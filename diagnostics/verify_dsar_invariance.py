"""G1 DSAR invariance verifier (RUNCOMMANDS §34.4/§34.4a hard-gate tool).

First-phase B1 deliverable evidence -- NOT accuracy. Three checks per
client, replayed on a real round-boundary checkpoint (plain fedta
baseline run, e.g. cifar_base_p15_legacy task_04_end):

  TEST 1  OFF/ON(identity) bit-identical neutrality.
          Commit identity sidecars through the REAL runtime hook path
          (_dsar_fit_pre -> load_prompts -> _dsar_fit_post), then for
          every task k <= task_id run the official evaluate() twice
          from the SAME four-stream RNG point (shadow-comparison
          semantics: identical bs8 + shuffle=True + train=True batch
          manifest and dropout draws) -- once with the identity
          sidecar, once with the sidecar detached. Asserts:
          accuracy, per-batch adapted features and per-batch logits
          are bit-identical. Also asserts the identity-mode hooks
          themselves consume no RNG.

  TEST 2  Current/historical task readout dispatch wiring.
          Eval modes (deterministic), synthetic ACTIVE sidecars:
            (a) B+L active: current task  z_on == L(B(z_raw)),
                historical task z_on == B(z_raw) (bitwise), and the
                logits equal a manual frozen-readout forward of the
                expected features (validates the insertion point --
                corrected z feeds BOTH key-anchor routing and the Chead
                feature half);
            (b) B identity / L active (the B-identity != DSAR-identity
                trap): historical logits == raw bitwise (L must NOT
                reach historical tasks), current logits != raw;
            (c) B active / L identity: historical logits != raw.

  TEST 3  Real-fit path RNG/mode neutrality + smoke (in-run broadcast
          mechanics, no training-run launch). Runs the REAL B/L fits
          around a broadcast with a PERTURBED prompt module (pre/post
          states genuinely differ), then asserts: four-stream RNG
          restored, module train/eval flags restored, vit prompt state
          == broadcast, Tail_Anchor params / heads untouched, sidecar
          committed with a legal status (FITTED / IDENTITY_GUARD /
          LOW_POWER_IDENTITY), r8 factor shapes + finiteness ok. A
          full replay must reproduce the factors bit-identically
          (deterministic seeds, dedicated Generator streams).

Usage (repo root):
  OMP_NUM_THREADS=8 python diagnostics/verify_dsar_invariance.py \
      --run_dir output/cifar100/fedta/cifar_base_p15_legacy/seed_42 \
      --checkpoint checkpoints/task_04_end.pth

Exit code 0 = all PASS; 1 = any FAIL (printed per test).
Evidence JSON (default <run_dir>/dsar_g1_invariance.json) records every
assertion outcome for the §34.4a G1 gate.
"""

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diag_utils import load_args_from_run_dir, resolve_checkpoint  # noqa: E402
from Models.state_hashes import hash_prompt_module  # noqa: E402
from Models.DSAR import DSARSidecar  # noqa: E402

FEAT_DIM = 768
DSAR_RANK = 8
LEGAL_STATUSES = {"FITTED", "IDENTITY_GUARD", "LOW_POWER_IDENTITY"}


def parse_args():
    p = argparse.ArgumentParser("G1 DSAR invariance verifier")
    p.add_argument("--run_dir", default=(
        "output/cifar100/fedta/cifar_base_p15_legacy/seed_42"))
    p.add_argument("--checkpoint", default="checkpoints/task_04_end.pth")
    p.add_argument("--clients", default="all",
                   help="'all' or comma-separated client ids, e.g. '0,2'")
    p.add_argument("--out_json", default=None,
                   help="default: <run_dir>/dsar_g1_invariance.json")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


# ------------------------------------------------------------- utilities

def _feed_tensor(h, t):
    h.update(t.detach().cpu().contiguous().numpy().tobytes())


def hash_module_state(module):
    """Deterministic sha256 over a module's full state_dict (params +
    buffers), so 'untouched' assertions are bitwise, not approximate."""
    h = hashlib.sha256()
    sd = module.state_dict() if module is not None else {}
    for k in sorted(sd.keys()):
        h.update(k.encode())
        _feed_tensor(h, sd[k])
    return h.hexdigest()


def rng_snapshot(client):
    return client._rng_snapshot()


def rng_equal(a, b):
    if not torch.equal(a["torch_cpu"], b["torch_cpu"]):
        return False
    if (a["torch_cuda_all"] is None) != (b["torch_cuda_all"] is None):
        return False
    if a["torch_cuda_all"] is not None:
        for x, y in zip(a["torch_cuda_all"], b["torch_cuda_all"]):
            if not torch.equal(x, y):
                return False
    if a["python"] != b["python"]:
        return False
    na, nb = list(a["numpy"]), list(b["numpy"])
    na[1] = na[1].tobytes()
    nb[1] = nb[1].tobytes()
    return na == nb


def run_eval_with_capture(client, task, nb_classes):
    """One official evaluate() pass; captures per-batch (adapted z, pre
    logits) via a forward hook on the client's Tail_Anchor. The hook is
    observer-only (clone + no RNG)."""
    captured = []

    def hook(_mod, args, output):
        captured.append((args[0].detach().clone(),
                         output[0].detach().clone()))

    h = client.model.register_forward_hook(hook)
    try:
        acc = client.evaluate(task, nb_classes)
    finally:
        h.remove()
    return acc, captured


def synthetic_sidecar(device, b_active=True, l_active=True,
                      seed=20261106, scale=0.02):
    """Deterministic ACTIVE factors (test-only; never a method arm)."""
    g = torch.Generator().manual_seed(int(seed))

    def mk():
        M1 = (torch.randn(FEAT_DIM, DSAR_RANK, generator=g) * scale).to(device)
        M2 = torch.randn(FEAT_DIM, DSAR_RANK, generator=g).to(device)
        bias = (torch.randn(FEAT_DIM, generator=g) * scale).to(device)
        return M1, M2, bias

    bM1, bM2, bb = mk() if b_active else (None, None, None)
    lM1, lM2, lb = mk() if l_active else (None, None, None)
    return DSARSidecar(b_active, bM1, bM2, bb,
                       l_active, lM1, lM2, lb,
                       {"test": "synthetic", "seed": int(seed)})


def apply_manual(z, M1, M2, bias):
    """The deployment convention written out explicitly (so TEST 2 does
    not lean on DSARSidecar.apply to verify itself)."""
    return z + (z @ M1) @ M2.t() + bias


def perturb_prompt_module(mod, delta=0.01):
    """In-place deterministic perturbation so TEST 3's pre/post prompt
    states genuinely differ (birth vs broadcast semantics exercised)."""
    with torch.no_grad():
        next(mod.parameters()).add_(delta)
    return mod


def sidecar_factors_equal(a, b):
    for attr in ("b_M1", "b_M2", "b_bias", "l_M1", "l_M2", "l_bias"):
        x, y = getattr(a, attr), getattr(b, attr)
        if (x is None) != (y is None):
            return False
        if x is not None and not torch.equal(x, y):
            return False
    return bool(a.b_active) == bool(b.b_active) and \
        bool(a.l_active) == bool(b.l_active)


# ---------------------------------------------------------------- tests

def test_identity_neutrality(client, report):
    """TEST 1: OFF vs ON(identity) bit-identical logits."""
    nb = client.nb_classes
    r = {"name": "T1_off_vs_on_identity_bitwise"}
    client._dsar_on = True
    client._dsar_identity = True

    rng_before = rng_snapshot(client)
    client._dsar_fit_pre(0, "verify")
    broadcast = deepcopy(client.prompts)
    client.vit.load_prompts(broadcast)
    client._dsar_fit_post(0, "verify")
    rng_after = rng_snapshot(client)

    sc = client._dsar_sidecar
    r["sidecar_committed"] = sc is not None
    r["sidecar_is_identity"] = bool(sc.is_identity()) if sc else False
    r["prompt_hash_matches_broadcast"] = (
        sc is not None
        and sc.meta.get("prompt_hash") == hash_prompt_module(broadcast))
    r["hooks_rng_neutral"] = rng_equal(rng_before, rng_after)

    per_task = []
    ok = (r["sidecar_committed"] and r["sidecar_is_identity"]
          and r["prompt_hash_matches_broadcast"] and r["hooks_rng_neutral"])
    for k in range(client.task_id + 1):
        if k >= len(client.heads) or client.heads[k] is None:
            continue
        rng1 = rng_snapshot(client)
        acc_on, cap_on = run_eval_with_capture(client, k, nb)
        # rewind to the SAME RNG point; detach the sidecar (OFF arm)
        client._rng_restore(rng1)
        saved = client._dsar_sidecar
        client._dsar_sidecar = None
        acc_off, cap_off = run_eval_with_capture(client, k, nb)
        client._dsar_sidecar = saved
        z_eq = all(torch.equal(a[0], b[0])
                   for a, b in zip(cap_on, cap_off))
        lg_eq = all(torch.equal(a[1], b[1])
                    for a, b in zip(cap_on, cap_off))
        n_batches_match = len(cap_on) == len(cap_off)
        task_ok = (acc_on == acc_off and z_eq and lg_eq
                   and n_batches_match)
        per_task.append({"task": k, "acc_on": acc_on, "acc_off": acc_off,
                         "acc_equal": acc_on == acc_off,
                         "n_batches": len(cap_on),
                         "batch_manifest_equal": n_batches_match,
                         "features_bitwise_equal": bool(z_eq),
                         "logits_bitwise_equal": bool(lg_eq)})
        ok = ok and task_ok
    r["per_task"] = per_task
    r["verdict"] = "PASS" if ok else "FAIL"
    report.append(r)
    return ok


def test_dispatch_wiring(client, report):
    """TEST 2: frozen task dispatch on active synthetic sidecars."""
    nb = client.nb_classes
    t_cur = client.task_id
    t_hist = 0 if t_cur >= 1 else None
    r = {"name": "T2_readout_dispatch_wiring"}
    flags = (client.vit.training, client.model.training,
             client.original_model.training,
             tuple(None if h is None else h.training
                   for h in client.heads))
    try:
        client.original_model.eval()
        client.vit.eval()
        client.model.eval()
        # Chead carries Dropout(0.5) between its linears, and evaluate()
        # load_head()s a deepcopy of the STORED head (client.heads[task])
        # which inherits its training flag: unless the stored heads are
        # eval'd as well, the manual frozen-readout replica samples fresh
        # dropout masks at an unsynchronized RNG point and can never be
        # bitwise-comparable to the captured logits (G1 FAIL 2026-11-06,
        # clients 0-4: features_ok=True but manual_readout_logits_ok=False).
        # Official evaluate() train=True dropout semantics stay untouched
        # (T1 exercises them); this only pins T2's determinism intent.
        for hd in client.heads:
            if hd is not None:
                hd.eval()

        def paired(task, sidecar):
            """(adapted capture, raw capture) from the same RNG point."""
            rng = rng_snapshot(client)
            client._dsar_sidecar = sidecar
            _, cap_on = run_eval_with_capture(client, task, nb)
            client._rng_restore(rng)
            client._dsar_sidecar = None
            _, cap_off = run_eval_with_capture(client, task, nb)
            return cap_on, cap_off

        details = []

        def check_active(task, is_current, tag):
            syn = synthetic_sidecar(client.device, True, True,
                                    seed=20261106 + task)
            cap_on, cap_off = paired(task, syn)
            z_ok, lg_ok, active = True, True, False
            # expected features + expected logits (manual frozen readout:
            # model holds heads[task] after evaluate loaded it)
            for (z_on, lg_on), (z_raw, _) in zip(cap_on, cap_off):
                z_exp = apply_manual(z_raw, syn.b_M1, syn.b_M2, syn.b_bias)
                if is_current:
                    z_exp = apply_manual(z_exp, syn.l_M1, syn.l_M2,
                                         syn.l_bias)
                z_ok = z_ok and torch.equal(z_on, z_exp)
                with torch.no_grad():
                    pre, _, _ = client.model(z_exp, None)
                lg_ok = lg_ok and torch.equal(pre, lg_on)
                active = active or not torch.equal(lg_on, _)
            d = {"arm": tag, "task": task, "is_current": is_current,
                 "insertion_point_features_ok": bool(z_ok),
                 "manual_readout_logits_ok": bool(lg_ok),
                 "adapters_active": bool(active)}
            details.append(d)
            return z_ok and lg_ok and active

        ok = check_active(t_cur, True, "B+L(current)")

        def check_hist_semantics():
            good = True
            # (b) B identity / L active: historical untouched, current moved
            syn_bL = synthetic_sidecar(client.device, False, True,
                                       seed=20261107)
            cap_on, cap_off = paired(t_hist, syn_bL)
            hist_raw = all(torch.equal(a[1], b[1])
                           for a, b in zip(cap_on, cap_off))
            details.append({"arm": "B_identity_L_active(historical)",
                            "task": t_hist,
                            "logits_equal_raw_bitwise": bool(hist_raw)})
            good = good and hist_raw
            rng = rng_snapshot(client)
            client._dsar_sidecar = syn_bL
            _, cap_cur_on = run_eval_with_capture(client, t_cur, nb)
            client._rng_restore(rng)
            client._dsar_sidecar = None
            _, cap_cur_off = run_eval_with_capture(client, t_cur, nb)
            cur_moved = any(not torch.equal(a[1], b[1])
                            for a, b in zip(cap_cur_on, cap_cur_off))
            details.append({"arm": "B_identity_L_active(current)",
                            "task": t_cur, "logits_moved": bool(cur_moved)})
            good = good and cur_moved
            # (c) B active / L identity: historical moved
            syn_B = synthetic_sidecar(client.device, True, False,
                                      seed=20261108)
            cap_on, cap_off = paired(t_hist, syn_B)
            hist_moved = any(not torch.equal(a[1], b[1])
                             for a, b in zip(cap_on, cap_off))
            details.append({"arm": "B_active_L_identity(historical)",
                            "task": t_hist,
                            "logits_moved": bool(hist_moved)})
            good = good and hist_moved
            return good

        if t_hist is not None:
            ok = ok and check_hist_semantics() \
                and check_active(t_hist, False, "B+L(historical)")
        client._dsar_sidecar = None
    finally:
        client.vit.train(flags[0])
        client.model.train(flags[1])
        client.original_model.train(flags[2])
        for hd, was_training in zip(client.heads, flags[3]):
            if hd is not None:
                hd.train(was_training)
    r["details"] = details
    r["verdict"] = "PASS" if ok else "FAIL"
    report.append(r)
    return ok


def test_fit_neutrality(client, report):
    """TEST 3: REAL fit path RNG/mode neutrality + determinism smoke."""
    nb = client.nb_classes
    r = {"name": "T3_real_fit_rng_mode_neutrality"}

    client._dsar_on = True
    client._dsar_identity = False
    if client.prompts is None:
        r["verdict"] = "SKIP"
        r["reason"] = "client has no birth prompt state in the checkpoint"
        report.append(r)
        return True

    rng_before = rng_snapshot(client)
    flags = (client.vit.training, client.model.training,
             client.original_model.training)
    model_hash = hash_module_state(client.model)
    heads_hash = {k: hash_module_state(client.heads[k])
                  for k in range(client.task_id + 1)
                  if client.heads[k] is not None}
    birth_hash = hash_prompt_module(client.prompts)

    broadcast = perturb_prompt_module(deepcopy(client.prompts))
    broadcast_hash = hash_prompt_module(broadcast)

    client._dsar_fit_pre(1, "verify")
    client.vit.load_prompts(broadcast)
    client._dsar_fit_post(1, "verify")

    sc1 = client._dsar_sidecar
    r["sidecar_committed"] = sc1 is not None
    if sc1 is None:
        r["verdict"] = "FAIL"
        report.append(r)
        return False
    r["b_status"] = sc1.meta.get("b_status")
    r["l_status"] = sc1.meta.get("l_status")
    r["b_aux"] = {k: v for k, v in sc1.meta.get("b_aux", {}).items()
                  if isinstance(v, (int, float, str, bool))}
    r["l_final_ce"] = sc1.meta.get("l_final_ce")
    r["n_train"] = sc1.meta.get("n_train")
    r["statuses_legal"] = (sc1.meta.get("b_status") in LEGAL_STATUSES
                           and sc1.meta.get("l_status") in LEGAL_STATUSES)
    r["rank_shapes_ok"] = bool(sc1.rank_shapes_ok())
    r["factors_finite"] = bool(sc1.factors_finite())
    r["birth_hash_recorded"] = (sc1.meta.get("birth_prompt_hash")
                                 == birth_hash)
    r["prompt_hash_matches_perturbed_broadcast"] = (
        sc1.meta.get("prompt_hash") == broadcast_hash
        and broadcast_hash != birth_hash)

    rng_after = rng_snapshot(client)
    r["rng_restored"] = rng_equal(rng_before, rng_after)
    r["flags_restored"] = flags == (client.vit.training,
                                     client.model.training,
                                     client.original_model.training)
    r["vit_prompt_is_broadcast"] = (
        hash_prompt_module(client.vit.prompt) == broadcast_hash)
    r["tail_anchor_untouched"] = hash_module_state(client.model) == model_hash
    heads_now = {k: hash_module_state(client.heads[k])
                 for k in heads_hash}
    r["heads_untouched"] = heads_now == heads_hash

    # full replay: deterministic seeds + dedicated generators must
    # reproduce the factors bit-identically
    client._dsar_fit_pre(1, "verify")
    client.vit.load_prompts(broadcast)
    client._dsar_fit_post(1, "verify")
    sc2 = client._dsar_sidecar
    r["replay_reproduces_factors_bitwise"] = (
        sc2 is not None and sidecar_factors_equal(sc1, sc2))
    rng_final = rng_snapshot(client)
    r["replay_rng_restored"] = rng_equal(rng_before, rng_final)

    ok = all(r.get(k, False) for k in (
        "sidecar_committed", "statuses_legal", "rank_shapes_ok",
        "factors_finite", "birth_hash_recorded",
        "prompt_hash_matches_perturbed_broadcast", "rng_restored",
        "flags_restored", "vit_prompt_is_broadcast",
        "tail_anchor_untouched", "heads_untouched",
        "replay_reproduces_factors_bitwise", "replay_rng_restored"))
    # leave the client clean (no sidecar outside a real broadcast)
    client._dsar_sidecar = None
    r["verdict"] = "PASS" if ok else "FAIL"
    report.append(r)
    return ok


# ------------------------------------------------------------------ main

def main():
    cli = parse_args()
    run_dir = Path(cli.run_dir)
    out_json = (Path(cli.out_json) if cli.out_json
                else run_dir / "dsar_g1_invariance.json")

    args = load_args_from_run_dir(run_dir, data_path=cli.data_path,
                                  device=cli.device)
    # replay must be on the plain fedta baseline (B1 arm definition)
    if bool(getattr(args, "head_refit", False)):
        raise SystemExit("PROTOCOL_GATE_FAIL: run_dir is a head_refit run; "
                         "G1 DSAR verification requires the plain fedta "
                         "baseline")
    if str(getattr(args, "ctrp_mode", "off") or "off") != "off":
        raise SystemExit("PROTOCOL_GATE_FAIL: run_dir is a CTRP run; G1 "
                         "DSAR verification requires the plain fedta "
                         "baseline")
    args.dsar = False       # bootstrap OFF; the tests flip client flags
    args.dsar_identity = False

    from main import build_server
    server = build_server(args, run_manager=None)
    server.init_client()
    ckpt_path = resolve_checkpoint(run_dir, cli.checkpoint)
    server.load_checkpoint(ckpt_path)

    if cli.clients == "all":
        client_ids = list(range(server.client_num))
    else:
        client_ids = [int(x) for x in cli.clients.split(",") if x != ""]

    results = {}
    all_ok = True
    t0 = time.time()
    for ci in client_ids:
        client = server.clients[ci]
        if client.task_id < 0 or not client.data_split_indices:
            results[f"client_{ci}"] = [{"name": "skip",
                                        "reason": "no birth state"}]
            continue
        report = []
        c_ok = True
        c_ok = test_identity_neutrality(client, report) and c_ok
        c_ok = test_dispatch_wiring(client, report) and c_ok
        c_ok = test_fit_neutrality(client, report) and c_ok
        # restore OFF defaults
        client._dsar_on = False
        client._dsar_identity = False
        client._dsar_sidecar = None
        results[f"client_{ci}"] = report
        all_ok = all_ok and c_ok
        print("[G1] client %d: %s (%.0fs elapsed)"
              % (ci, "ALL PASS" if c_ok else "FAIL",
                 time.time() - t0), flush=True)

    evidence = {
        "gate": "G1_dsar_first_phase_invariance",
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "frozen_ref": "RUNCOMMANDS §34.4 / §34.4a",
        "tests": {
            "T1": "OFF vs ON(identity): official evaluate() bit-identical "
                  "(same RNG point, same bs8+shuffle+train=True manifest)",
            "T2": "frozen dispatch: current B->L, historical B-only; "
                  "insertion before self.model(); B identity != DSAR "
                  "identity",
            "T3": "real-fit path RNG/mode/state neutrality + deterministic "
                  "replay + legal statuses",
        },
        "results": results,
        "verdict": "PASS" if all_ok else "FAIL",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    print("\n=== G1 DSAR invariance: %s ===" % evidence["verdict"])
    for cname, report in results.items():
        for t in report:
            print("  %-8s %-42s %s"
                  % (cname, t.get("name", "?"), t.get("verdict", "?")))
    print("evidence: %s" % out_json)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
