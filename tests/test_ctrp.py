"""tests/test_ctrp.py -- §33 CTRP v1 (Calibrated Temporal Residual Prompt)
Test 1-10, per the GPT-frozen Step-3 plan (RUNCOMMANDS Sec.33, 2026-11-03).

  Test 1  off-path backward compatibility: ctrp_mode=off keeps the exact
           legacy behavior -- no ctrp_state in checkpoints, no ctrp_* keys
           in data splits, config defaults inert, legacy post_broadcast
           refit arm still constructs. (Full-run golden bit-identity vs
           the pre-patch HEAD is the user-run GPU smoke, see RUNCOMMANDS.)
  Test 2  config fail-closed: CTRP_V1_UNSUPPORTED_REFIT_TIMING, fixed_alpha
           alpha validation, grid validation, val_frac bounds, no dual
           sources of truth.
  Test 3  matched nested split: deterministic, class-stratified, disjoint,
           test side untouched; off = no ctrp_* keys.
  Test 4  anchor state machine + dominant-row-only deploy math: state 0
           no calibration; t>=1 deploy = (1-a)*prev + a*cand on the
           dominant row only; prompt_key untouched; alpha=1 skips the
           write; anchor advances to the deployed values.
  Test 5  premise gate: forward-actual route stats; nonslot>1% or slot
           change -> CTRP_PREMISE_FAIL (persisted BEFORE raising) for
           fixed_alpha/ctrp_val/ctrp_train; endpoint logs diagnostics and
           continues; <=1% nonslot passes; zero val batches fail closed.
  Test 6  checkpoint/resume: ctrp_state schema; frozen cases A (off + old
           ckpt), B (matched + pre-state0-hook ckpt -> fresh init),
           C (matched + transitions started + missing anchor ->
           CTRP_RESUME_ANCHOR_MISSING); grid/mode mismatch rejected.
  Test 7  selector semantics: pooled sample-weighted CE (NOT the
           equal-weight client mean), argmin over the ascending grid with
           the Probe-H tie-break (ties -> smallest alpha).
  Test 8  Probe-H replay equivalence gate (GPU + archived runs): the
           PRODUCTION evaluator kernel replays Probe-H's original row
           sets (probe-compat protocol lives in this test layer ONLY --
           seeds 51/52, <=512-row cap, batch 8) on the archived CIFAR +
           INR transitions. Gates: route slot equal; per-row/per-alpha
           CE max |diff| <= 1e-6 (else report the actual max diff, NO
           silent threshold relaxation); pooled val CE curve equal;
           selected alpha EXACTLY equal. Formal CTRP training may not
           start before this gate passes. The production kernel is the
           Probe-H shim intervention (Models/ctrp.py), NOT a new "more
           faithful" real-module forward (frozen Sec.5).
  Test 9  matched-split fairness: endpoint / fixed_alpha / ctrp_val /
           ctrp_train share BITWISE-identical core/val/test indices.
  Test 10 mid-task resume anchor integrity: a mid-task checkpoint stores
           the anchor from the LAST task end (never the current task's
           fusion/candidate intermediate state); resume restores it
           unchanged. (The full uninterrupted-vs-resumed GPU equivalence
           run is part of the user-run smoke.)

CPU-runnable subset (no CUDA / no datasets needed):

    python -m unittest tests.test_ctrp -k "Test1 or Test2 or Test3 or \
        Test4 or Test5 or Test6 or Test7 or Test9 or Test10" -v

Test 8 (GPU, archived runs, from the repo root):

    python tests/test_ctrp.py Test8ProbeHReplayEquivalence -v
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.ctrp import (DEFAULT_ALPHA_GRID, hash_prompt_module,
                          hash_tensor, parse_alpha_grid, select_alpha,
                          stratified_split)

CUDA = torch.cuda.is_available()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _ns(**kw):
    base = dict(nb_classes=20, seed=42, num_workers=0,
                refit_timing='local', head_refit=False,
                ctrp_mode='off', ctrp_alpha=None,
                ctrp_alpha_grid=None, ctrp_val_frac=0.20,
                no_instrumentation=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class _SynthSet(Dataset):
    """Tiny deterministic dataset: n rows, k classes (round-robin)."""

    def __init__(self, n=200, k=10):
        self.x = torch.arange(n)
        self.targets = [int(i % k) for i in range(n)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


class _FakePrompt(nn.Module):
    """Minimal Global_Prompt stand-in: prompt (pool, length, dim) values +
    prompt_key (pool, dim). state_dict layout matches the real module."""

    def __init__(self, pool=8, length=2, dim=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.prompt = nn.Parameter(
            torch.randn(pool, length, dim, generator=g))
        self.prompt_key = nn.Parameter(
            torch.randn(pool, dim, generator=g))


class _StubClient:
    """Stands in for Client_DF at the server hook: canned route slots and
    sweep results."""

    def __init__(self, cid, slots, sweep=None):
        self.id = cid
        self.slots = slots
        self.sweep = sweep
        self.sweep_calls = []
        self.data_split_indices = {}
        # minimal vit stand-in so the hook's device alignment can read
        # next(vit.parameters()).device
        self.vit = _FakePrompt()

    def _ctrp_route_slots(self, server_prompt_module, **kw):
        return list(self.slots)

    def _ctrp_alpha_sweep(self, prompt, anchor, slot, grid, **kw):
        self.sweep_calls.append(slot)
        return dict(self.sweep)

    def state_dict_for_checkpoint(self):
        return {"id": self.id}


def _sweep_res(ce_sum, n_rows, n_dropped=0, n_batches=None):
    return {
        "ce_sum_by_alpha": np.asarray(ce_sum, dtype=np.float64),
        "n_rows": int(n_rows),
        "n_dropped": int(n_dropped),
        "n_batches": int(n_batches if n_batches is not None else n_rows),
        "batch_slots": [],
        "row_ce": None,
        "row_meta": None,
    }


class _StubRunManager:
    def __init__(self):
        self.metrics_dir = Path(tempfile.mkdtemp(prefix="ctrp_test_"))
        self.records = []

    def append_jsonl(self, path, obj):
        self.records.append((Path(path).name, obj))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")

    def load_accuracy_records(self):
        return []


def _make_server(mode='off', ge=2, task_num=2, client_num=1, alpha=None,
                 val_frac=0.2, refit_timing='local', head_refit=False,
                 run_manager=None, grid=DEFAULT_ALPHA_GRID, seed=42,
                 with_data=False):
    from Models.Server_DF import Server_DF
    args = _ns(seed=seed, ctrp_mode=mode, ctrp_alpha=alpha,
               ctrp_alpha_grid=(grid if mode != 'off' else None),
               ctrp_val_frac=val_frac, refit_timing=refit_timing,
               head_refit=head_refit)
    if with_data:
        synth = _SynthSet(200, 10)
        client_data = [[Subset(synth, list(range(100)))
                        for _ in range(task_num)]
                       for _ in range(client_num)]
        class_mask = [[list(range(10))] * task_num
                      for _ in range(client_num)]
    else:
        client_data = [None] * client_num
        class_mask = [[[0, 1]] * task_num for _ in range(client_num)]
    server = Server_DF('Server', None, 'Tail_Anchor', client_num, task_num,
                      client_data, class_mask, 0.01, ge, 1, 16, 'cpu',
                      'fedta', 0.25, None, None, args, None, run_manager)
    return server


def _make_client(mode='off', seed=42, cid=0, k=10, n=200, alpha=None,
                 tasks=2):
    from Models.Client_DF import Client_DF
    args = _ns(seed=seed, ctrp_mode=mode, ctrp_alpha=alpha,
               ctrp_val_frac=0.2)
    return Client_DF(cid, None, 'Tail_Anchor', 1,
                     [_SynthSet(n, k) for _ in range(tasks)], 1, 16,
                     0.01, 'cpu', 'fedta',
                     [[list(range(k))] for _ in range(tasks)], args, None)


def _default_grid_list():
    return [float(x) for x in DEFAULT_ALPHA_GRID.split(",")]


# ----------------------------------------------------------------------
# Test 1 -- off-path backward compatibility
# ----------------------------------------------------------------------

class Test1OffPathIdentity(unittest.TestCase):

    def test_off_server_has_no_ctrp_state_and_legacy_keys(self):
        server = _make_server(mode='off', with_data=True)
        server.task_id = 1
        ckpt = server.build_checkpoint(3)
        self.assertNotIn("ctrp_state", ckpt)
        # legacy schema, unchanged by the CTRP patch
        for key in ("completed_round", "server_task_id", "server_model",
                    "global_head", "global_protos", "temp_protos",
                    "fix_keys", "existing_class", "server_prompt",
                    "clients", "rng_state", "data_split_state",
                    "outer_split_manifest", "args"):
            self.assertIn(key, ckpt)

    def test_off_get_data_records_no_ctrp_keys(self):
        client = _make_client(mode='off')
        torch.manual_seed(1234)
        client.get_data(0)
        split = client.data_split_indices[0]
        self.assertEqual(set(split.keys()), {"train", "test"})
        # traindata stays the full inner 70% (legacy semantics)
        self.assertEqual(list(client.traindata.indices), split["train"])

    def test_off_constructs_without_any_ctrp_attrs_in_args(self):
        # pre-CTRPTA checkpoints carry args WITHOUT ctrp_* keys; a server
        # built from a namespace lacking them must construct identically
        from Models.Server_DF import Server_DF
        legacy = types.SimpleNamespace(
            nb_classes=20, seed=42, num_workers=0,
            refit_timing='local', head_refit=False,
            no_instrumentation=False)
        server = Server_DF('Server', None, 'Tail_Anchor', 1, 2, [None],
                           [[0, 1], [0, 1]], 0.01, 2, 1, 16, 'cpu',
                           'fedta', 0.25, None, None, legacy, None)
        self.assertEqual(server._ctrp_mode, 'off')
        self.assertIsNone(server._ctrp_prev_deployed)

    def test_off_keeps_legacy_post_broadcast_refit_arm(self):
        # ctrp_mode=off + refit_timing=post_broadcast (the A5-v2 §28 arm)
        # must still construct -- the CTRP_V1 gate only fires when
        # ctrp_mode != off (frozen verdict Sec.1.1)
        server = _make_server(mode='off', refit_timing='post_broadcast',
                              head_refit=True)
        self.assertEqual(server._ctrp_mode, 'off')

    def test_config_defaults_match_probe_grid(self):
        # default alpha grid == the Probe-H lambda grid (21 points, 0.05)
        self.assertEqual(parse_alpha_grid(DEFAULT_ALPHA_GRID),
                         [round(i * 0.05, 2) for i in range(21)])


# ----------------------------------------------------------------------
# Test 2 -- config fail-closed
# ----------------------------------------------------------------------

class Test2ConfigFailClosed(unittest.TestCase):

    def _assert_exit_contains(self, mode, code_part, **kw):
        with self.assertRaises(SystemExit) as cm:
            _make_server(mode=mode, **kw)
        self.assertIn(code_part, str(cm.exception.code))

    def test_post_broadcast_rejected_for_matched_modes(self):
        for mode in ('endpoint', 'fixed_alpha', 'ctrp_val', 'ctrp_train'):
            with self.assertRaises(SystemExit) as cm:
                _make_server(mode=mode, refit_timing='post_broadcast',
                            head_refit=True, alpha=(0.5 if mode ==
                                                     'fixed_alpha' else None))
            self.assertIn("CTRP_V1_UNSUPPORTED_REFIT_TIMING",
                          str(cm.exception.code))

    def test_post_broadcast_rejected_even_without_head_refit(self):
        # raw refit_timing value carries the intent; CTRP rejects it
        # regardless of whether the legacy §28 check fired first
        with self.assertRaises(SystemExit) as cm:
            _make_server(mode='ctrp_val', refit_timing='post_broadcast',
                         head_refit=False)
        self.assertIn(
            "CTRP_V1_UNSUPPORTED_REFIT_TIMING"
            if "CTRP_V1_UNSUPPORTED_REFIT_TIMING" in str(cm.exception.code)
            else "requires --head_refit",
            str(cm.exception.code))

    def test_fixed_alpha_requires_alpha(self):
        self._assert_exit_contains('fixed_alpha',
                                   "requires --ctrp_alpha", alpha=None)

    def test_fixed_alpha_range(self):
        self._assert_exit_contains('fixed_alpha', "[0, 1]", alpha=1.5)
        self._assert_exit_contains('fixed_alpha', "[0, 1]", alpha=-0.1)

    def test_alpha_only_with_fixed_alpha(self):
        self._assert_exit_contains('ctrp_val', "only valid", alpha=0.5)

    def test_grid_validation(self):
        self._assert_exit_contains('ctrp_val', "strictly ascending",
                                   grid="0.5,0.2,1.0")
        self._assert_exit_contains('ctrp_val', "strictly ascending",
                                   grid="0.2,0.2,1.0")
        self._assert_exit_contains('ctrp_val', "must contain 1.0",
                                   grid="0.0,0.2,0.5")
        self._assert_exit_contains('ctrp_val', "within [0, 1]",
                                   grid="0.0,0.5,1.5")

    def test_val_frac_bounds(self):
        for bad in (0.0, 1.0, 1.5, -0.2):
            self._assert_exit_contains('ctrp_val', "(0, 1)",
                                       val_frac=bad)

    def test_valid_configs_construct(self):
        _make_server(mode='fixed_alpha', alpha=0.5)
        _make_server(mode='fixed_alpha', alpha=1.0)
        _make_server(mode='ctrp_val')
        _make_server(mode='ctrp_train')
        _make_server(mode='endpoint')


# ----------------------------------------------------------------------
# Test 3 -- matched nested split
# ----------------------------------------------------------------------

class Test3MatchedNestedSplit(unittest.TestCase):

    def _get_split(self, mode='ctrp_val', seed=42, cid=0):
        client = _make_client(mode=mode, seed=seed, cid=cid)
        torch.manual_seed(4321)   # controls the outer 70/30 random_split
        client.get_data(0)
        return client, dict(client.data_split_indices[0])

    def test_matched_split_created_and_traindata_is_core(self):
        client, split = self._get_split()
        self.assertEqual(
            set(split.keys()), {"train", "test", "ctrp_core", "ctrp_val"})
        self.assertEqual(list(client.traindata.indices), split["ctrp_core"])

    def test_deterministic_given_same_inner_train(self):
        _, s1 = self._get_split()
        _, s2 = self._get_split()
        self.assertEqual(s1["train"], s2["train"])
        self.assertEqual(s1["ctrp_core"], s2["ctrp_core"])
        self.assertEqual(s1["ctrp_val"], s2["ctrp_val"])
        self.assertEqual(s1["test"], s2["test"])

    def test_seed_depends_on_client_and_task(self):
        _, s0 = self._get_split(cid=0)
        _, s1 = self._get_split(cid=1)
        # same inner train (same torch.manual_seed) but a different
        # client id -> different a5_stable_seed -> different val draw
        self.assertEqual(s0["train"], s1["train"])
        self.assertNotEqual(s0["ctrp_val"], s1["ctrp_val"])

    def test_stratification_and_disjointness(self):
        client, split = self._get_split()
        train_set = set(split["train"])
        core_set = set(split["ctrp_core"])
        val_set = set(split["ctrp_val"])
        self.assertEqual(core_set | val_set, train_set)
        self.assertFalse(core_set & val_set)
        # val rows live strictly inside the inner train, never in test
        self.assertFalse(val_set & set(split["test"]))
        # per-class val count = Probe-H stratified formula
        synth = _SynthSet(200, 10)
        labels = {}
        for i in split["train"]:
            labels.setdefault(synth.targets[i], []).append(i)
        for c, rows in labels.items():
            n_val = sum(1 for i in split["ctrp_val"]
                        if synth.targets[i] == c)
            expected = max(1, int(round(0.2 * len(rows)))) \
                if len(rows) >= 2 else 0
            expected = min(expected, len(rows) - 1) if len(rows) >= 2 else 0
            self.assertEqual(n_val, expected,
                             f"class {c}: {n_val} != {expected}")

    def test_off_records_no_ctrp_keys(self):
        client, split = self._get_split(mode='off')
        self.assertEqual(set(split.keys()), {"train", "test"})


# ----------------------------------------------------------------------
# Test 4 -- anchor state machine + deploy math
# ----------------------------------------------------------------------

class _HookHarness:
    """Drives _ctrp_calibrate_task_end with stub clients."""

    def __init__(self, mode='ctrp_val', alpha=None, ge=2,
                 slots=(6, 6, 6, 6), sweep=None, client_num=1,
                 run_manager=None):
        self.server = _make_server(mode=mode, ge=ge, alpha=alpha,
                                   run_manager=run_manager, with_data=True,
                                   client_num=client_num)
        self.server.prompt = _FakePrompt(seed=1)
        self.server.clients = [
            _StubClient(i, list(slots), sweep) for i in range(client_num)]
        self.key_hash_before = hash_tensor(self.server.prompt.prompt_key)

    def run(self, round_id):
        self.server._ctrp_calibrate_task_end(round_id)


class Test4AnchorStateMachine(unittest.TestCase):

    def test_state0_sets_anchor_without_calibration(self):
        h = _HookHarness(mode='ctrp_val', ge=2)
        before = h.server.prompt.prompt.detach().clone()
        h.run(round_id=1)   # task 0 end
        s = h.server
        self.assertTrue(torch.equal(s._ctrp_prev_deployed, before))
        self.assertTrue(torch.equal(s.prompt.prompt.detach(), before))
        self.assertEqual(s._ctrp_prev_dominant_slot, 6)
        self.assertEqual(s._ctrp_history[-1]["transition"], "state0_init")
        self.assertEqual(hash_tensor(s.prompt.prompt_key),
                         h.key_hash_before)

    def test_transition_deploys_selected_alpha_on_dominant_row_only(self):
        # ce curve with unique argmin at alpha = 0.3 (index 6)
        grid = _default_grid_list()
        curve = [2.0] * len(grid)
        curve[6] = 1.0
        h = _HookHarness(mode='ctrp_val', ge=2,
                         sweep=_sweep_res(curve, n_rows=10))
        h.run(1)                                   # state 0
        anchor_before = h.server._ctrp_prev_deployed.clone()
        # task 1 produces a NEW candidate (fusion moved the values)
        gen = torch.Generator().manual_seed(7)
        h.server.prompt.prompt.data += torch.randn(
            h.server.prompt.prompt.shape, generator=gen)
        cand = h.server.prompt.prompt.detach().clone()
        h.run(3)                                    # task 1 end (t=1)
        s = h.server
        expected_row = (1.0 - 0.3) * anchor_before[6] + 0.3 * cand[6]
        self.assertTrue(torch.equal(s.prompt.prompt.detach()[6], expected_row))
        for j in range(cand.shape[0]):
            if j != 6:
                self.assertTrue(
                    torch.equal(s.prompt.prompt.detach()[j], cand[j]))
        # prompt_key untouched (route(alpha=0)==route(alpha=1) invariant)
        self.assertEqual(hash_tensor(s.prompt.prompt_key), h.key_hash_before)
        # anchor advanced to the deployed values
        self.assertTrue(torch.equal(s._ctrp_prev_deployed,
                                    s.prompt.prompt.detach()))
        self.assertEqual(s._ctrp_selected_alpha, 0.3)
        self.assertEqual(s._ctrp_prev_dominant_slot, 6)
        rec = s._ctrp_history[-1]
        self.assertEqual(rec["transition"], "0->1")
        self.assertEqual(rec["selected_alpha"], 0.3)
        self.assertEqual(rec["interior_alpha"], True)
        self.assertEqual(len(rec["val_ce_by_alpha"]), len(grid))
        # stub ce_sums are POOLED against n_rows=10 -> per-row CE
        self.assertAlmostEqual(rec["ce_at_alpha1"], 2.0 / 10, places=12)
        self.assertAlmostEqual(rec["selected_ce"], 1.0 / 10, places=12)

    def test_endpoint_skips_write(self):
        h = _HookHarness(mode='endpoint', ge=2)
        h.run(1)
        h.server.prompt.prompt.data += 1.0        # candidate moved
        cand = h.server.prompt.prompt.detach().clone()
        h.run(3)
        # alpha=1 -> deployed IS the candidate; the write is skipped so
        # the values stay bitwise identical (no float no-op drift)
        self.assertTrue(torch.equal(h.server.prompt.prompt.detach(), cand))
        self.assertEqual(h.server._ctrp_selected_alpha, 1.0)

    def test_fixed_alpha_deploys_fixed_without_sweep(self):
        h = _HookHarness(mode='fixed_alpha', alpha=0.5, ge=2,
                         sweep=None)   # sweep must NOT be called
        h.run(1)
        anchor_before = h.server._ctrp_prev_deployed.clone()
        gen = torch.Generator().manual_seed(9)
        h.server.prompt.prompt.data += torch.randn(
            h.server.prompt.prompt.shape, generator=gen)
        cand = h.server.prompt.prompt.detach().clone()
        h.run(3)
        expected = 0.5 * anchor_before[6] + 0.5 * cand[6]
        self.assertTrue(
            torch.equal(h.server.prompt.prompt.detach()[6], expected))
        self.assertEqual(h.server.clients[0].sweep_calls, [])

    def test_zero_val_batches_fail_closed(self):
        h = _HookHarness(mode='ctrp_val', ge=2, slots=())
        with self.assertRaises(RuntimeError) as cm:
            h.run(1)
        self.assertIn("CTRP_VAL_ROWS_EMPTY", str(cm.exception))


# ----------------------------------------------------------------------
# Test 5 -- premise gate
# ----------------------------------------------------------------------

class Test5PremiseGate(unittest.TestCase):

    def _run_transition_with_slots(self, mode, slots_t1, alpha=None,
                                   slots_t0=(6, 6, 6, 6)):
        rm = _StubRunManager()
        h = _HookHarness(mode=mode, alpha=alpha, ge=2, slots=slots_t0,
                         sweep=_sweep_res([2.0] * 21, n_rows=10),
                         run_manager=rm)
        h.run(1)
        h.server.prompt.prompt.data += 1.0
        h.server.clients[0].slots = list(slots_t1)
        try:
            h.run(3)
            outcome = ("ok", None)
        except RuntimeError as e:
            outcome = ("raise", str(e))
        return h.server, rm, outcome

    def test_nonslot_above_1pct_fails_closed_and_persists_first(self):
        server, rm, outcome = self._run_transition_with_slots(
            'ctrp_val', slots_t1=(6, 6, 7, 6, 6, 6, 6, 6))
        self.assertEqual(outcome[0], "raise")
        self.assertIn("CTRP_PREMISE_FAIL", outcome[1])
        # the reason was persisted BEFORE the exception was raised
        events = [r for name, r in rm.records
                  if name == "ctrp_events.jsonl"]
        self.assertTrue(events)
        self.assertEqual(events[-1]["event"], "CTRP_PREMISE_FAIL")
        self.assertIn("nonslot batch fraction", events[-1]["reason"])
        # premise fail = the candidate was NOT deployed/modified
        self.assertIsNone(server._ctrp_selected_alpha)

    def test_endpoint_continues_and_logs_diagnostics(self):
        server, rm, outcome = self._run_transition_with_slots(
            'endpoint', slots_t1=(6, 6, 7, 6, 6, 6, 6, 6))
        self.assertEqual(outcome[0], "ok")
        events = [r for name, r in rm.records
                  if name == "ctrp_events.jsonl"]
        self.assertTrue(any(e["event"] == "CTRP_PREMISE_DIAGNOSTIC"
                            for e in events))
        rec = server._ctrp_history[-1]
        self.assertEqual(rec["premise_status"], "diagnostic_only")
        # alpha=1: deployed == candidate, no write
        self.assertEqual(rec["selected_alpha"], 1.0)

    def test_slot_change_fails_for_all_strict_arms(self):
        for mode in ('fixed_alpha', 'ctrp_val', 'ctrp_train'):
            server, rm, outcome = self._run_transition_with_slots(
                mode, slots_t1=(7, 7, 7, 7),
                alpha=(0.5 if mode == 'fixed_alpha' else None))
            self.assertEqual(outcome[0], "raise", mode)
            self.assertIn("CTRP_PREMISE_FAIL", outcome[1])
            self.assertIn("dominant slot changed", outcome[1])

    def test_one_percent_nonslot_passes(self):
        # 99 batches at the dominant slot + 1 non-dominant = exactly 1%
        slots = (6,) * 99 + (7,)
        server, rm, outcome = self._run_transition_with_slots(
            'ctrp_val', slots_t1=slots)
        self.assertEqual(outcome[0], "ok")
        self.assertEqual(server._ctrp_history[-1]["premise_status"],
                         "pass")

    def test_premise_fail_reason_lands_in_run_dir(self):
        rm = _StubRunManager()
        h = _HookHarness(mode='ctrp_train', ge=2, slots=(6, 6, 6, 6),
                         sweep=_sweep_res([2.0] * 21, n_rows=10),
                         run_manager=rm)
        h.run(1)
        h.server.prompt.prompt.data += 1.0
        h.server.clients[0].slots = [7, 7, 7, 7]
        with self.assertRaises(RuntimeError):
            h.run(3)
        path = rm.metrics_dir / "ctrp_events.jsonl"
        self.assertTrue(path.exists())
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f.read().splitlines()]
        self.assertEqual(lines[-1]["event"], "CTRP_PREMISE_FAIL")


# ----------------------------------------------------------------------
# Test 6 -- checkpoint schema + resume cases A/B/C
# ----------------------------------------------------------------------

class Test6CheckpointResume(unittest.TestCase):

    def test_matched_checkpoint_carries_ctrp_state(self):
        server = _make_server(mode='ctrp_val', with_data=True)
        server.task_id = 0
        server.prompt = _FakePrompt()
        ckpt = server.build_checkpoint(1)   # ge=2 -> task-end round -> True
        st = ckpt["ctrp_state"]
        for key in ("mode", "current_task", "calibration_done",
                    "prev_deployed_prompt", "prev_dominant_slot",
                    "dominant_slot", "selected_alpha", "alpha_grid",
                    "val_ce_by_alpha", "candidate_prompt_hash",
                    "prev_deployed_prompt_hash", "deployed_prompt_hash",
                    "history"):
            self.assertIn(key, st)
        self.assertEqual(st["mode"], 'ctrp_val')
        self.assertTrue(st["calibration_done"])
        self.assertEqual(st["alpha_grid"], _default_grid_list())

    def test_mid_task_checkpoint_calibration_done_false(self):
        server = _make_server(mode='ctrp_val', with_data=True)
        ckpt = server.build_checkpoint(0)   # mid task (round 0 of 2)
        self.assertFalse(ckpt["ctrp_state"]["calibration_done"])

    # -- case A: off + legacy checkpoint ---------------------------------
    def test_case_a_off_with_legacy_checkpoint(self):
        server = _make_server(mode='off')
        # a pre-CTRPTA checkpoint has no ctrp_state at all
        server._ctrp_resume({"completed_round": 5, "server_task_id": 1})
        self.assertIsNone(server._ctrp_prev_deployed)

    # -- case B: matched + checkpoint before the state-0 hook point ----
    def test_case_b_fresh_init_before_state0_hook(self):
        server = _make_server(mode='ctrp_val')   # ge=2
        # completed_round 0 < ge-1=1: the state-0 hook still lies ahead
        server._ctrp_resume({"completed_round": 0, "server_task_id": 0})
        self.assertIsNone(server._ctrp_prev_deployed)

    def test_case_c_before_hook_but_task0_end_fails(self):
        # a legacy (non-CTRPTA) checkpoint taken at exactly the task-0
        # end: the state-0 calibration point has passed, so resuming a
        # matched arm lacks the anchor -> fail-closed (never guess from
        # server_prompt)
        server = _make_server(mode='ctrp_val')
        with self.assertRaises(RuntimeError) as cm:
            server._ctrp_resume({"completed_round": 1, "server_task_id": 0})
        self.assertIn("CTRP_RESUME_ANCHOR_MISSING", str(cm.exception))

    # -- case C: matched + transitions started + missing anchor ----------
    def test_case_c_transitions_started_missing_state(self):
        server = _make_server(mode='ctrp_train')
        with self.assertRaises(RuntimeError) as cm:
            server._ctrp_resume({"completed_round": 7, "server_task_id": 3})
        self.assertIn("CTRP_RESUME_ANCHOR_MISSING", str(cm.exception))
        self.assertIn("guess the anchor", str(cm.exception))

    def test_case_c_state_present_but_anchor_none(self):
        server = _make_server(mode='ctrp_val')
        ckpt = {"completed_round": 7, "server_task_id": 3,
                "ctrp_state": {"mode": "ctrp_val",
                               "prev_deployed_prompt": None}}
        with self.assertRaises(RuntimeError) as cm:
            server._ctrp_resume(ckpt)
        self.assertIn("CTRP_RESUME_ANCHOR_MISSING", str(cm.exception))

    def test_matched_state_restored(self):
        server = _make_server(mode='ctrp_val')
        anchor = torch.arange(8 * 2 * 4).float().reshape(8, 2, 4)
        history = [{"transition": "state0_init"}]
        ckpt = {"completed_round": 6, "server_task_id": 3,
                "ctrp_state": {
                    "mode": "ctrp_val",
                    "prev_deployed_prompt": anchor,
                    "prev_dominant_slot": 6,
                    "dominant_slot": 6,
                    "selected_alpha": 0.3,
                    "alpha_grid": _default_grid_list(),
                    "history": history,
                }}
        server._ctrp_resume(ckpt)
        self.assertTrue(torch.equal(server._ctrp_prev_deployed, anchor))
        self.assertEqual(server._ctrp_prev_dominant_slot, 6)
        self.assertEqual(server._ctrp_selected_alpha, 0.3)
        self.assertEqual(server._ctrp_history, history)

    def test_mode_mismatch_rejected(self):
        server = _make_server(mode='ctrp_val')
        ckpt = {"completed_round": 6, "server_task_id": 3,
                "ctrp_state": {"mode": "endpoint",
                               "prev_deployed_prompt": torch.zeros(1)}}
        with self.assertRaises(RuntimeError) as cm:
            server._ctrp_resume(ckpt)
        self.assertIn("mode mismatch", str(cm.exception))

    def test_grid_mismatch_rejected(self):
        server = _make_server(mode='ctrp_val')
        ckpt = {"completed_round": 6, "server_task_id": 3,
                "ctrp_state": {
                    "mode": "ctrp_val",
                    "prev_deployed_prompt": torch.zeros(1),
                    "alpha_grid": [0.0, 0.5, 1.0]}}
        with self.assertRaises(RuntimeError) as cm:
            server._ctrp_resume(ckpt)
        self.assertIn("alpha grid mismatch", str(cm.exception))

    def test_resume_must_match_includes_ctrp_args(self):
        from Models.Server_DF import Server_DF
        self.assertIn("ctrp_mode", Server_DF.RESUME_MUST_MATCH)
        self.assertIn("ctrp_alpha", Server_DF.RESUME_MUST_MATCH)
        self.assertIn("ctrp_alpha_grid", Server_DF.RESUME_MUST_MATCH)
        self.assertIn("ctrp_val_frac", Server_DF.RESUME_MUST_MATCH)


# ----------------------------------------------------------------------
# Test 7 -- selector semantics
# ----------------------------------------------------------------------

class Test7SelectorSemantics(unittest.TestCase):

    def test_tie_break_smallest_alpha(self):
        grid = _default_grid_list()
        self.assertEqual(select_alpha([2.0] * len(grid), grid), 0.0)
        curve = [3.0] * len(grid)
        curve[6] = curve[7] = 1.0      # tie between 0.3 and 0.35
        self.assertEqual(select_alpha(curve, grid), 0.3)

    def test_selector_equals_numpy_argmin(self):
        rng = np.random.default_rng(0)
        grid = _default_grid_list()
        for _ in range(20):
            curve = rng.normal(size=len(grid))
            self.assertEqual(select_alpha(curve, grid),
                             grid[int(np.argmin(curve))])

    def test_pooled_ce_is_sample_weighted_not_client_mean(self):
        # client A: 1 row, per-row CE [1, 2, 2, ...]; client B: 3 rows,
        # per-row CE [2, 1, 2, ...] (sweeps return CE SUMS, so B's stub
        # payload is 3x the per-row curve).
        # Pooled = (A_sum + B_sum) / (1+3) -> argmin at alpha=0.05.
        # The equal-weight client mean would tie ([1.5, 1.5, ...]) and
        # select 0.0, so this distinguishes the frozen aggregation.
        grid = _default_grid_list()
        ce_a = [2.0] * len(grid)
        ce_b_per_row = [2.0] * len(grid)
        ce_a[0], ce_a[1] = 1.0, 2.0
        ce_b_per_row[0], ce_b_per_row[1] = 2.0, 1.0
        ce_b = [3.0 * v for v in ce_b_per_row]
        h = _HookHarness(mode='ctrp_val', ge=2, client_num=2,
                         sweep=_sweep_res(ce_b, n_rows=3))
        h.server.clients[0].sweep = _sweep_res(ce_a, n_rows=1)
        h.server.clients[1].sweep = _sweep_res(ce_b, n_rows=3)
        h.run(1)
        h.server.prompt.prompt.data += 1.0
        h.run(3)
        rec = h.server._ctrp_history[-1]
        self.assertEqual(rec["selected_alpha"], 0.05)
        self.assertAlmostEqual(rec["val_ce_by_alpha"][0], 7 / 4, places=12)
        self.assertAlmostEqual(rec["val_ce_by_alpha"][1], 5 / 4, places=12)


# ----------------------------------------------------------------------
# Test 8 -- Probe-H replay equivalence gate (GPU + archived runs)
# ----------------------------------------------------------------------

class Test8ProbeHReplayEquivalence(unittest.TestCase):
    """Replays the archived Probe-H gate transitions (CIFAR 2->3 and
    INR 2->3) with the PRODUCTION evaluator kernel on the Probe-H
    original row sets.

    The probe-compat row protocol (stratified pick with derived seeds
    51/52, the <=512-row cap, batch 8) lives HERE, in the test layer --
    there is no production --probe_compat CLI (frozen verdict Sec.1.2).
    The kernel itself is shared (Models/ctrp.py).

    Gates (frozen Sec.6): route slot equal; per-row per-alpha CE
    max|diff| <= 1e-6 (if fp32 cannot reach it the test reports the
    actual max diff and FAILS -- no silent threshold relaxation);
    pooled val CE curve equal within the same gate; selected alpha
    EXACTLY equal. Formal CTRP training may not start before PASS.
    """

    CE_GATE = 1e-6

    REPLAYS = [
        {   # CIFAR-100, gate transition 2->3 (archived Probe-H run)
            "name": "cifar",
            "run_dir": "output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42",
            "cache_dir": "diagnostics_output/stepcalib_cifar",
            "cur_ckpt": "task_03_end.pth",
            "prev_state": "state_task_02_end",
            "cur_state": "state_task_03_end",
            "slot": 6,
            "t_to": 3,
            "summary": "diagnostics_output/stepcalib_cifar.summary.json",
        },
        {   # ImageNet-R, gate transition 2->3
            "name": "inr",
            "run_dir": "output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42",
            "cache_dir": "diagnostics_output/stepcalib_inr_seed42",
            "cur_ckpt": "task_03_end.pth",
            "prev_state": "state_task_02_end",
            "cur_state": "state_task_03_end",
            "slot": 53,
            "t_to": 3,
            "summary": "diagnostics_output/stepcalib_inr_seed42.summary.json",
        },
    ]

    def _paths_available(self, spec):
        return (Path(spec["run_dir"]).exists()
                and Path(spec["cache_dir"]).exists()
                and (Path(spec["cache_dir"])
                     / f"{spec['cur_state']}.npz").exists()
                and (Path(spec["cache_dir"])
                     / f"{spec['prev_state']}.npz").exists())

    def _replay(self, spec):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",
                              ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        device = "cuda"

        # the probe module imports diag_utils bare (it normally runs as a
        # script from diagnostics/); importable from here too
        diag_dir = str(REPO_ROOT / "diagnostics")
        if diag_dir not in sys.path:
            sys.path.insert(0, diag_dir)
        from diagnostics.probe_prompt_step_calibration import (
            _seed, _stratified_pick, _stratified_split)
        from diagnostics.diag_utils import bootstrap_server

        cache = Path(spec["cache_dir"])
        prev = np.load(cache / f"{spec['prev_state']}.npz")
        cur = np.load(cache / f"{spec['cur_state']}.npz")
        with open(cache / f"{spec['cur_state']}.json", encoding="utf-8") \
                as f:
            meta = json.load(f)
        fp = meta["fingerprint"]
        lam_grid = [float(x) for x in fp["lambdas"]]
        batch_size = int(fp["batch_size"])
        cap = int(fp["select_rows_per_client"])
        val_frac = float(fp["val_frac"])
        slot = int(spec["slot"])
        t_to = int(spec["t_to"])
        self.assertEqual(int(fp["monopoly_slot"]), slot)

        run_args, server = bootstrap_server(
            spec["run_dir"], checkpoint=spec["cur_ckpt"],
            data_path="./local_datasets", device=device)
        # probe runtime parity: everything on device, eval mode
        server.model.to(device)
        server.origin_model.to(device)
        server.origin_model.eval()
        server.prompt = server.prompt.to(device)
        for c in server.clients:
            c.model.to(device)

        # anchor = prev state's deployed pool values (identical across
        # clients in the archived run)
        pv_prev = prev["pv_pool"]
        for i in range(1, pv_prev.shape[0]):
            self.assertTrue(np.array_equal(pv_prev[0], pv_prev[i]),
                            "prev pv pools differ across clients")
        anchor = torch.tensor(np.asarray(pv_prev[0]))
        # the loaded server prompt IS the current state's broadcast
        # candidate == probe's pv_now (bitwise, float32)
        for i in range(1, cur["pv_pool"].shape[0]):
            self.assertTrue(np.array_equal(cur["pv_pool"][0],
                                           cur["pv_pool"][i]))
        self.assertTrue(np.array_equal(
            server.prompt.prompt.detach().cpu().numpy(),
            np.asarray(cur["pv_pool"][0]).astype(np.float32)))

        prod_ce_blocks = []
        prod_meta_blocks = []
        for c in server.clients:
            cid = c.id
            split = c.data_split_indices[t_to]
            train_idx = list(split["train"])
            # probe-compat row protocol (test layer only, frozen
            # Sec.1.2): stratified pick <= cap rows with derived seed
            # 51, then the stratified 20/80 split with derived seed 52
            labels = c._ctrp_read_labels(c.train_data[t_to], train_idx)
            rng_pick = np.random.default_rng(
                _seed(run_args.seed, 51, t_to, cid))
            keep_pos = _stratified_pick(labels, cap, rng_pick)
            kept_labels = [labels[i] for i in keep_pos]
            rng_split = np.random.default_rng(
                _seed(run_args.seed, 52, t_to, cid))
            flags = _stratified_split(kept_labels, val_frac, rng_split)
            sel_idx = [train_idx[i] for i in keep_pos]

            res = c._ctrp_alpha_sweep(
                server.prompt, anchor, slot, lam_grid, rows=sel_idx,
                batch_size=batch_size, record_row_ce=True)
            expected_batches = (len(sel_idx) + batch_size - 1) // batch_size
            self.assertEqual(res["n_batches"], expected_batches)
            self.assertTrue(all(s == slot
                                for s in res["batch_slots"]),
                            "production route hit a non-dominant slot")
            prod_ce_blocks.append(res["row_ce"])
            prod_meta_blocks.append(res["row_meta"])

        prod_ce = np.concatenate(prod_ce_blocks, axis=0)
        cached_ce = np.asarray(cur["sel_ce"], dtype=np.float32)
        self.assertEqual(prod_ce.shape, cached_ce.shape,
                         "row count mismatch: production "
                         f"{prod_ce.shape} vs cached {cached_ce.shape}")
        # row alignment: y_local sequence must match the cached rows
        prod_y_local = [m["y_local"] for block in prod_meta_blocks
                        for m in block]
        self.assertEqual(prod_y_local,
                         [int(v) for v in cur["sel_y_local"]])
        max_diff = float(np.max(np.abs(prod_ce.astype(np.float64)
                                       - cached_ce.astype(np.float64))))
        # pooled current-task VAL CE per alpha (Probe-H selector signal)
        split_flags = np.asarray(cur["sel_split"])
        va = split_flags == 1
        prod_pool = prod_ce[va].mean(axis=0, dtype=np.float64)
        cached_pool = cached_ce[va].mean(axis=0, dtype=np.float64)
        pool_max_diff = float(np.max(np.abs(prod_pool - cached_pool)))
        prod_alpha = select_alpha(prod_pool, lam_grid)
        cached_alpha = select_alpha(cached_pool, lam_grid)
        self.assertEqual(prod_alpha, cached_alpha,
                         "selected alpha differs from Probe-H")
        # cross-check against the archived probe summary
        if Path(spec["summary"]).exists():
            with open(spec["summary"], encoding="utf-8") as f:
                summary = json.load(f)
            if str(t_to) in summary["transitions"]:
                rec = summary["transitions"][str(t_to)]
                if rec.get("lam_hat_val") is not None:
                    self.assertEqual(
                        float(rec["lam_hat_val"]), cached_alpha,
                        "Probe-H summary lam_hat_val mismatch")
        detail = (f"[{spec['name']}] row CE max|diff|={max_diff:.3e}, "
                  f"pooled CE max|diff|={pool_max_diff:.3e}, "
                  f"selected alpha={prod_alpha} "
                  f"(probe {cached_alpha})")
        print("TEST8 " + detail)
        self.assertLessEqual(max_diff, self.CE_GATE,
                              detail + " -- exceeds the frozen 1e-6 gate; "
                              "report the actual max diff, do NOT relax "
                              "the threshold")
        self.assertLessEqual(pool_max_diff, self.CE_GATE, detail)
        self.assertEqual(prod_alpha, cached_alpha)

    def test_replay_cifar_transition(self):
        if not (CUDA and self._paths_available(self.REPLAYS[0])):
            self.skipTest("needs CUDA + the archived CIFAR Probe-H run "
                          "(output/cifar100/... + diagnostics_output/"
                          "stepcalib_cifar)")
        self._replay(self.REPLAYS[0])

    def test_replay_inr_transition(self):
        if not (CUDA and self._paths_available(self.REPLAYS[1])):
            self.skipTest("needs CUDA + the archived INR Probe-H run")
        self._replay(self.REPLAYS[1])


# ----------------------------------------------------------------------
# Test 9 -- matched-split fairness across the four arms
# ----------------------------------------------------------------------

class Test9MatchedSplitFairness(unittest.TestCase):

    def test_four_arms_share_bitwise_identical_splits(self):
        modes = [('endpoint', None), ('fixed_alpha', 0.5),
                 ('ctrp_val', None), ('ctrp_train', None)]
        splits = {}
        for mode, alpha in modes:
            client = _make_client(mode=mode, seed=42, cid=3, alpha=alpha)
            torch.manual_seed(999)     # identical outer split for all arms
            client.get_data(1)
            splits[mode] = client.data_split_indices[1]
        base = splits['endpoint']
        for key in ("train", "test", "ctrp_core", "ctrp_val"):
            for mode, _ in modes[1:]:
                self.assertEqual(splits[mode][key], base[key],
                                 f"{mode} diverged on {key}")


# ----------------------------------------------------------------------
# Test 10 -- mid-task resume anchor integrity
# ----------------------------------------------------------------------

class Test10MidTaskResumeAnchor(unittest.TestCase):

    def test_mid_task_checkpoint_keeps_last_deployed_anchor(self):
        # drive state 0 + transition 1, then simulate the next task's
        # mid-task fusion state (server_prompt drifting) and check the
        # checkpoint anchor is NEVER overwritten by the candidate
        grid = _default_grid_list()
        curve = [2.0] * len(grid)
        curve[6] = 1.0
        h = _HookHarness(mode='ctrp_val', ge=2,
                         sweep=_sweep_res(curve, n_rows=10))
        h.run(1)                      # state 0
        h.run(3)                      # transition 0->1 (alpha 0.3)
        deployed_1 = h.server.prompt.prompt.detach().clone()
        self.assertEqual(h.server._ctrp_selected_alpha, 0.3)

        # mid task 1... er, mid task 2: fusion moves the server prompt
        # (this is exactly the state that must NOT become the anchor)
        gen = torch.Generator().manual_seed(11)
        h.server.prompt.prompt.data += torch.randn(
            h.server.prompt.prompt.shape, generator=gen)
        fusion_state = h.server.prompt.prompt.detach().clone()
        h.server.task_id = 2

        ckpt = h.server.build_checkpoint(6)   # mid task 2 (ge=2)
        st = ckpt["ctrp_state"]
        self.assertFalse(st["calibration_done"])
        anchor_saved = st["prev_deployed_prompt"].float()
        self.assertTrue(torch.equal(anchor_saved, deployed_1))
        self.assertFalse(torch.equal(anchor_saved, fusion_state))

        # resume into a fresh matched server: the anchor comes back
        # unchanged (never the fusion/candidate intermediate state)
        fresh = _make_server(mode='ctrp_val', ge=2)
        fresh._ctrp_resume(ckpt)
        self.assertTrue(torch.equal(fresh._ctrp_prev_deployed,
                                     deployed_1))
        self.assertEqual(fresh._ctrp_prev_dominant_slot, 6)
        self.assertEqual(fresh._ctrp_selected_alpha, 0.3)
        self.assertEqual(len(fresh._ctrp_history), 2)


if __name__ == "__main__":
    unittest.main()
