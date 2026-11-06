"""tests/test_bsab_pair_capture.py -- Sec.34.2 0b BSAB_PAIR_CAPTURE hook.

Frozen gates (RUNCOMMANDS Sec.34.2, user 2026-11-04):
  * default OFF = bit-identical no-op (legacy args accepted, nothing
    written, no pending state);
  * capture ON: gate-1 strict same-sample pairing (the row manifest is
    independently REBUILT at the post half and the row_index_hash must
    match; labels/split/sample_index re-verified), gate-2 birth-head
    anchoring (full head state + hash saved BEFORE load_prompts, hash
    re-verified AFTER);
  * instrumentation neutrality (gate-3, unit level): across the
    broadcast boundary the capture-ON run is BITWISE identical to the
    capture-OFF run on the four RNG streams, every module tensor and
    every module mode; the full-run OFF/ON regression stays a separate
    pre-rerun gate (Sec.34.2.4);
  * paired-cache semantics: z_ref (frozen original model) is
    pre/post-invariant; z_op shifts by exactly the broadcast prompt
    delta; labels never leave the cache into anything else.

This file is instrumentation-only: no BSAB module exists here and the
BSAB architecture implementation stays FROZEN (Sec.34).
"""

import json
import os
import random
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.Client_DF import Client_DF  # noqa: E402
from Models.classification_head import Chead  # noqa: E402
from Models.ctrp import hash_prompt_module  # noqa: E402

_DEVICE = 'cpu'
_FEAT_DIM = 4
_CLIENT_ID = 7


class _BsabPrompt(nn.Module):
    """Prompt-module stand-in with a full registered state so
    hash_prompt_module sees the same fingerprint format as the real
    Global_Prompt."""

    def __init__(self, offset):
        super().__init__()
        self.register_buffer("offset", offset.detach().clone())


class _BsabVit(nn.Module):
    """feat = input + prompt.offset: z_op is the prompt-enhanced stream,
    so a broadcast must shift z_op by exactly the prompt delta while
    z_ref (from the frozen original model) stays invariant."""

    def __init__(self):
        super().__init__()
        self.prompt = _BsabPrompt(torch.zeros(1, _FEAT_DIM))

    def forward(self, input, task_id=None, cls_features=None, train=False):
        return {'feat': input + self.prompt.offset}

    def load_prompts(self, prompts):
        # real load_prompts deepcopies the full module state
        self.prompt = deepcopy(prompts)

    def get_prompts(self):
        return deepcopy(self.prompt)


class _BsabOrig(nn.Module):
    """Frozen original-model stand-in: z_ref = 0.5 * input (768-d feat
    slot of the real model shrunk to the test width)."""

    def forward(self, input):
        return {'pre_logits': input, 'feat': 0.5 * input}


def _make_client(capture=False, capture_dir=None, seed=123,
                 n_train=20, n_test=6, task_id=0, legacy_args=False):
    kwargs = dict(nb_classes=20, seed=seed)
    if not legacy_args:
        kwargs.update(bsab_pair_capture=capture,
                      refit_timing='local', refit_input='feat',
                      head_refit=True)
        if capture_dir is not None:
            kwargs['bsab_capture_dir'] = str(capture_dir)
    args = types.SimpleNamespace(**kwargs)
    class_mask = [[2 * t, 2 * t + 1] for t in range(10)]
    g = torch.Generator().manual_seed(seed + 31)
    n_rows = n_train + n_test
    feats = torch.randn(n_rows, _FEAT_DIM, generator=g)
    labels = torch.randint(2 * task_id, 2 * task_id + 2, (n_rows,),
                           generator=g)
    train_data = [TensorDataset(feats, labels)]
    torch.manual_seed(seed)
    client = Client_DF(_CLIENT_ID, _BsabOrig(), 'Tail_Anchor', 5,
                       train_data, 1, 16, 0.01, _DEVICE, 'fedta',
                       class_mask, args, _BsabVit())
    client.task_id = task_id
    client.data_split_indices = {
        0: {'train': list(range(n_train)),
            'test': list(range(n_train, n_rows))}}
    torch.manual_seed(seed + 5)
    client.heads[0] = Chead(20)
    client.feats, client.labels_ref = feats, labels
    # shared-vit residue vs the client's own birth prompt (bugfix
    # 2026-11-04): at a client's broadcast entry the SHARED vit holds
    # residue from another client (here offset 0.90); this client's own
    # post-training birth state lives in client.prompts (here 0.10).
    # pre-capture must forward z- under client.prompts and restore the
    # residue afterwards.
    client.prompts = _BsabPrompt(torch.full((1, _FEAT_DIM), 0.10))
    client.vit.prompt = _BsabPrompt(torch.full((1, _FEAT_DIM), 0.90))
    return client


def _np_rng_equal(a, b):
    return (a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
            and np.array_equal(a[3], b[3]) and a[4] == b[4])


def _rng_snapshot():
    return {
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _rng_equal(a, b):
    if not torch.equal(a["torch"], b["torch"]):
        return False
    if a["cuda"] is not None and b["cuda"] is not None:
        if len(a["cuda"]) != len(b["cuda"]):
            return False
        for x, y in zip(a["cuda"], b["cuda"]):
            if not torch.equal(x, y):
                return False
    if a["python"] != b["python"]:
        return False
    if not _np_rng_equal(a["numpy"], b["numpy"]):
        return False
    return True


def _state_equal(s1, s2):
    if set(s1.keys()) != set(s2.keys()):
        return False
    return all(torch.equal(s1[k], s2[k]) for k in s1)


class TestDefaultOffNoOp(unittest.TestCase):

    def test_off_is_bitwise_noop(self):
        tmp = Path(tempfile.mkdtemp())
        torch.manual_seed(900)
        client = _make_client(capture=False, capture_dir=tmp, seed=101)
        torch.manual_seed(4242)
        expected_rng = _rng_snapshot()
        bcast = _BsabPrompt(torch.full((1, _FEAT_DIM), 0.25))
        client.get_global_proto_and_head_no_test(None, None, bcast, 3)
        # nothing written, no pending state, broadcast took effect
        self.assertEqual(list(tmp.iterdir()), [])
        self.assertIsNone(client._bsab_pending)
        self.assertTrue(torch.all(client.vit.prompt.offset == 0.25))
        # RNG streams untouched by the (disabled) hook
        self.assertTrue(_rng_equal(expected_rng, _rng_snapshot()))

    def test_legacy_args_accepted(self):
        client = _make_client(legacy_args=True, seed=102)
        self.assertFalse(client._bsab_capture_on)
        bcast = _BsabPrompt(torch.full((1, _FEAT_DIM), 0.25))
        client.get_global_proto_and_head_no_test(None, None, bcast, 0)
        self.assertIsNone(client._bsab_pending)

    def test_no_birth_state_is_skipped(self):
        tmp = Path(tempfile.mkdtemp())
        client = _make_client(capture=True, capture_dir=tmp, seed=103)
        client.task_id = -1  # client has never seen any task
        client.data_split_indices = {}
        bcast = _BsabPrompt(torch.full((1, _FEAT_DIM), 0.25))
        client.get_global_proto_and_head_no_test(None, None, bcast, 0)
        self.assertIsNone(client._bsab_pending)
        self.assertEqual(list(tmp.iterdir()), [])


class TestCaptureContent(unittest.TestCase):

    def test_on_caches_paired_rows_and_anchors(self):
        tmp = Path(tempfile.mkdtemp())
        n_train, n_test = 20, 6
        client = _make_client(capture=True, capture_dir=tmp, seed=1,
                              n_train=n_train, n_test=n_test)
        pre_prompt_hash = hash_prompt_module(client.prompts)
        head_hash = hash_prompt_module(client.heads[0])
        delta = torch.full((1, _FEAT_DIM), 0.25)
        bcast = _BsabPrompt(delta)
        client.get_global_proto_and_head_no_test(None, None, bcast, 3)

        files = sorted(p.name for p in tmp.iterdir())
        stem = "bsab_c%d_t0_r0003_broadcast_no_test" % _CLIENT_ID
        self.assertEqual(files, [stem + ".json", stem + ".npz"])

        z = np.load(tmp / (stem + ".npz"))
        meta = json.loads((tmp / (stem + ".json")).read_text())
        n = n_train + n_test
        self.assertEqual(int(z["client_ids"][0]), _CLIENT_ID)
        self.assertTrue(np.all(z["task_ids"] == 0))
        self.assertTrue(np.all(z["round_ids"] == 3))
        # gate-1: exact recorded row manifest, train block first
        self.assertEqual(z["sample_index"].tolist(), list(range(n)))
        self.assertEqual(z["split"].tolist(), [0] * n_train + [1] * n_test)
        self.assertTrue(np.array_equal(
            z["label"], client.labels_ref.numpy().astype(np.int64)))
        self.assertEqual(int(meta["n_rows"]), n)
        self.assertEqual(int(meta["n_train"]), n_train)
        self.assertEqual(int(meta["n_test"]), n_test)
        self.assertEqual(meta["feature_shape"], [_FEAT_DIM])
        self.assertTrue(meta["low_power"])  # 20 < BSAB_CAPTURE_MIN_FIT
        # gate-2 anchors
        self.assertEqual(meta["pre_prompt_hash"], pre_prompt_hash)
        self.assertEqual(meta["post_prompt_hash"], hash_prompt_module(bcast))
        self.assertNotEqual(meta["pre_prompt_hash"],
                            meta["post_prompt_hash"])
        self.assertEqual(meta["head_hash"], head_hash)
        self.assertTrue(meta["gates"]["row_pairing_verified"])
        self.assertTrue(meta["gates"]["birth_head_hash_verified"])
        # paired feature semantics: z_ref pre/post-invariant (frozen
        # original model); z_op_pre = birth prompt (0.10), NOT the shared
        # vit residue (0.90); z_op_post = birth prompt + broadcast delta
        self.assertTrue(np.array_equal(z["z_ref_pre"], z["z_ref_post"]))
        self.assertTrue(np.array_equal(
            z["z_ref_pre"],
            (0.5 * client.feats.numpy()).astype(np.float32)))
        self.assertTrue(np.array_equal(
            z["z_op_pre"],
            (client.feats.numpy() + 0.10).astype(np.float32)))
        self.assertTrue(np.array_equal(
            z["z_op_post"],
            (client.feats.numpy() + 0.25).astype(np.float32)))
        # the anchored birth head state travels with the cache
        sd = client.heads[0].state_dict()
        self.assertEqual(sorted(k for k in z.files if k.startswith("head__")),
                         sorted("head__" + k for k in sd))
        for k, v in sd.items():
            self.assertTrue(np.array_equal(z["head__" + k],
                                           v.detach().cpu().numpy()), k)

    def test_pre_state_uses_clients_own_birth_prompts(self):
        """Regression (2026-11-04 OFF/ON arms: z_op_static=8/10): at a
        client's broadcast entry the shared vit holds residue from
        another client's training or an earlier broadcast -- NOT this
        client's own birth state. z- must be forwarded under
        client.prompts, and the residue restored afterwards."""
        tmp = Path(tempfile.mkdtemp())
        client = _make_client(capture=True, capture_dir=tmp, seed=21)
        client._bsab_capture_pre(3, entry='broadcast_no_test')
        # shared-vit residue (0.90) restored, not overwritten by the
        # birth prompt or left as the birth state
        self.assertTrue(torch.all(client.vit.prompt.offset == 0.90))
        pend = client._bsab_pending
        self.assertIsNotNone(pend)
        self.assertEqual(pend["pre_prompt_hash"],
                         hash_prompt_module(client.prompts))
        # z- forwarded under the birth prompt (0.10), not the residue
        z_op = np.asarray(pend["z_op_pre"])
        self.assertTrue(np.array_equal(
            z_op, (client.feats.numpy() + 0.10).astype(np.float32)))
        self.assertFalse(np.array_equal(
            z_op, (client.feats.numpy() + 0.90).astype(np.float32)))

    def test_missing_birth_prompts_fails_closed(self):
        tmp = Path(tempfile.mkdtemp())
        client = _make_client(capture=True, capture_dir=tmp, seed=22)
        client.prompts = None
        with self.assertRaises(RuntimeError) as ctx:
            client._bsab_capture_pre(3, entry='broadcast_no_test')
        self.assertIn("birth prompt", str(ctx.exception))
        # shared vit untouched by the failed capture
        self.assertTrue(torch.all(client.vit.prompt.offset == 0.90))
        self.assertIsNone(client._bsab_pending)

    def test_active_entry_records_broadcast_entry(self):
        tmp = Path(tempfile.mkdtemp())
        client = _make_client(capture=True, capture_dir=tmp, seed=2)
        bcast = _BsabPrompt(torch.full((1, _FEAT_DIM), 0.25))
        # evaluate is irrelevant to the capture halves; shadow it
        client.evaluate = lambda task, nb_classes=None: 0.0
        client.get_global_proto_and_head(None, None, bcast, 5)
        files = sorted(p.name for p in tmp.iterdir())
        self.assertEqual(files,
                         ["bsab_c7_t0_r0005_broadcast.json",
                          "bsab_c7_t0_r0005_broadcast.npz"])
        meta = json.loads((tmp / "bsab_c7_t0_r0005_broadcast.json").read_text())
        self.assertEqual(meta["entry"], "broadcast")

    def test_refuses_to_overwrite_earlier_capture(self):
        tmp = Path(tempfile.mkdtemp())
        client = _make_client(capture=True, capture_dir=tmp, seed=3)
        bcast = _BsabPrompt(torch.full((1, _FEAT_DIM), 0.25))
        client.get_global_proto_and_head_no_test(None, None, bcast, 2)
        with self.assertRaises(RuntimeError) as ctx:
            client.get_global_proto_and_head_no_test(
                None, None, _BsabPrompt(torch.full((1, _FEAT_DIM), 0.25)), 2)
        self.assertIn("refusing to overwrite", str(ctx.exception))

    def test_missing_capture_dir_fails_closed(self):
        client = _make_client(capture=True, capture_dir=None, seed=4)
        bcast = _BsabPrompt(torch.full((1, _FEAT_DIM), 0.25))
        with self.assertRaises(RuntimeError) as ctx:
            client.get_global_proto_and_head_no_test(None, None, bcast, 0)
        self.assertIn("capture directory was wired", str(ctx.exception))


class TestGateViolationsFailClosed(unittest.TestCase):

    def test_gate1_row_manifest_change_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        client = _make_client(capture=True, capture_dir=tmp, seed=5)
        client._bsab_capture_pre(3, entry='broadcast_no_test')
        self.assertIsNotNone(client._bsab_pending)
        # the split mutates between the two halves (e.g. a concurrent
        # rebuild): strict same-sample pairing is broken
        client.data_split_indices[0]["test"].pop()
        with self.assertRaises(RuntimeError) as ctx:
            client._bsab_capture_post(3, entry='broadcast_no_test')
        self.assertIn("GATE-1", str(ctx.exception))
        self.assertIsNone(client._bsab_pending)

    def test_gate2_birth_head_swap_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        client = _make_client(capture=True, capture_dir=tmp, seed=6)
        client._bsab_capture_pre(3, entry='broadcast_no_test')
        torch.manual_seed(64)
        client.heads[0] = Chead(20)  # different birth head after anchoring
        with self.assertRaises(RuntimeError) as ctx:
            client._bsab_capture_post(3, entry='broadcast_no_test')
        self.assertIn("GATE-2", str(ctx.exception))
        self.assertIsNone(client._bsab_pending)

    def test_mismatched_halves_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        client = _make_client(capture=True, capture_dir=tmp, seed=7)
        client._bsab_capture_pre(3, entry='broadcast_no_test')
        with self.assertRaises(RuntimeError):
            client._bsab_capture_post(4, entry='broadcast_no_test')


class TestInstrumentationNeutrality(unittest.TestCase):
    """Gate-3 (unit level): capture ON is bitwise identical to capture
    OFF across the broadcast boundary -- every module tensor, every
    module mode, all four RNG streams, and the next draw. The full-run
    OFF/ON regression (Sec.34.2.4) re-checks this at trajectory level
    before the official capture rerun."""

    def test_off_on_trajectory_bitwise_identical(self):
        tmp_off = Path(tempfile.mkdtemp())
        tmp_on = Path(tempfile.mkdtemp())
        off = _make_client(capture=False, capture_dir=tmp_off, seed=11)
        on = _make_client(capture=True, capture_dir=tmp_on, seed=11)
        # identical module states by construction (same seeds)
        self.assertTrue(_state_equal(off.model.state_dict(),
                                     on.model.state_dict()))
        # training modes are restored around the capture forward
        for c in (off, on):
            c.vit.train()
            c.model.train()
            c.original_model.train()
        delta = torch.full((1, _FEAT_DIM), 0.25)
        torch.manual_seed(777)
        off.get_global_proto_and_head_no_test(
            None, None, _BsabPrompt(delta), 2)
        torch.manual_seed(777)
        on.get_global_proto_and_head_no_test(
            None, None, _BsabPrompt(delta), 2)
        # module tensors bitwise identical
        self.assertTrue(_state_equal(off.model.state_dict(),
                                     on.model.state_dict()))
        self.assertTrue(torch.equal(off.vit.prompt.offset,
                                    on.vit.prompt.offset))
        self.assertTrue(_state_equal(off.heads[0].state_dict(),
                                     on.heads[0].state_dict()))
        # module modes restored
        for c in (off, on):
            self.assertTrue(c.vit.training)
            self.assertTrue(c.model.training)
            self.assertTrue(c.original_model.training)
        # four RNG streams bitwise identical + identical next draws
        # (each draw is taken from ITS OWN snapshot: the two draws happen
        # in one process, so the second one must re-seed from the second
        # snapshot, not follow the first draw's advance)
        rng_off = _rng_snapshot()
        rng_on = _rng_snapshot()
        self.assertTrue(_rng_equal(rng_off, rng_on))
        torch.set_rng_state(rng_off["torch"])
        random.setstate(rng_off["python"])
        np.random.set_state(rng_off["numpy"])
        draw_off = (torch.rand(4), random.random(), np.random.rand(3))
        torch.set_rng_state(rng_on["torch"])
        random.setstate(rng_on["python"])
        np.random.set_state(rng_on["numpy"])
        draw_on = (torch.rand(4), random.random(), np.random.rand(3))
        self.assertTrue(torch.equal(draw_off[0], draw_on[0]))
        self.assertEqual(draw_off[1], draw_on[1])
        self.assertTrue(np.array_equal(draw_off[2], draw_on[2]))
        # only the ON run produced a capture artifact
        self.assertEqual(list(tmp_off.iterdir()), [])
        self.assertEqual(len(list(tmp_on.iterdir())), 2)


if __name__ == '__main__':
    unittest.main()
