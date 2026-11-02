"""§27 P1/P2 boundary attribution probe (RUNCOMMANDS §27, 2026-11-02).

Zero-training, read-only probe answering the two kill-questions that
must pass BEFORE any PGC (prototype-guided class separation) mechanism
code is written:

  P1 boundary attribution dichotomy -- is M2's boundary space ("the
     task was never learned that well in the first place", the sunken
     boundary rows, e.g. cifar T2 92.58) in the HEAD or in the FEATURES?
       retrain (mlp_local recipe on cached z_op) >> native refit head
         -> HEAD-LAG  (the head lags the features; PGC premise collapses)
       retrain ~= native after subtracting the T0 recipe-gap baseline
         -> FEAT      (head already saturates the features; PGC space holds)

  P2 geometric intrusion -- do the sunken task's new-class x1 features
     intrude deeper into the old-class global_protos region than the
     non-sunken tasks'?
       gap(sunken - others) >= 0.05 cos, or sunken absolute proto
       intrusion > its own old-old p95 reference
         -> INTRUDE (margin-squeeze mechanism has direct evidence)
       otherwise -> CLEAN (PGC mechanism assumption dies)

  Double gate (P1=FEAT AND P2=INTRUDE) -> §27.6 判据臂; any failure ->
  PGC dies at zero mechanism cost (same pre-registered-kill discipline
  as §26 A/B).

  Space registration (§27.0 code facts):
    P1 works on z_op 768 (client._a5_collect_head_inputs(feat_only=True)
    -- the exact M2' refit feature path);
    P2 works on x1 1536 (client._a5_collect_head_inputs(feat_only=False)
    -- the exact official proto-collection forward, same space as
    global_protos). The two spaces are different sources, hence each
    probe is registered in its own and never cross-converted.

  Discipline:
    - P1 VERDICT is PAIRED: the restored native refit head and the fresh
      mlp_local head both consume the SAME cached z_op test features
      (batchwise realisation noise -- measured at 0.4-0.7pp mean swings
      -- is common-mode and cancels in the delta). The native refit head
      is the M2' feat form (1536 input, anchor-half first-Linear weights
      hard-zeroed), so head(z||0) IS the native head function;
    - the official evaluate_all_seen_tasks serial replay is kept as a
      CATASTROPHE gate only (--sanity_tol, default 1.0pp, full-row check
      R[t][k] for every seen k): wrong-head/wrong-pool/wrong-task bugs
      deviate by multi-pp and die here; sub-1pp deviations are disclosed
      in the summary as realisation-noise evidence, not gate failures;
    - every feature collection / replay is RNG-snapshot wrapped so the
      pieces cannot pollute each other;
    - per-client protos equality check (global_protos is a single
      federated broadcast object; divergence means a probe bug);
    - NaN protos are skipped with a count report (official proto
      collection itself guards NaN); NaN x1 features abort (probe bug).

Usage (repo root, user tmux):

  python diagnostics/probe_boundary_attribution.py \
      --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
      --checkpoints task_00_end.pth,task_01_end.pth,task_02_end.pth,task_03_end.pth,task_04_end.pth \
      --out_csv diagnostics_output/boundary_probe/cifar_attribution.csv \
      --summary_json diagnostics_output/boundary_probe/cifar_attribution.json --device cuda
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.diag_utils import bootstrap_server  # noqa: E402
from diagnostics.probe_oracle_headroom import (  # noqa: E402
    fit_head, head_predict, stable_seed, _acc)
from runtime_utils import rng_state_dict, restore_rng_state  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("§27 P1/P2 boundary attribution probe")
    p.add_argument("--run_dir", required=True,
                   help=".../output/{ds}/{method}/{run}/seed_{seed}")
    p.add_argument("--checkpoints", required=True,
                   help="comma list, e.g. task_00_end.pth,...,task_04_end.pth")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--summary_json", required=True)
    p.add_argument("--data_path", default=None,
                   help="override dataset root if it moved")
    p.add_argument("--device", default="cuda")
    # mlp_local recipe (probe_oracle_headroom defaults; kept identical so
    # the T0 recipe-gap baseline stays comparable with §22.2 numbers)
    p.add_argument("--hidden", default=256, type=int)
    p.add_argument("--retrain_grid", default="std:256,std:512,nostd:256,nostd:512",
                   help="comma list of retrain recipe variants for the "
                        "§27.6 recipe-separation reading; each spec is "
                        "<std|nostd>:<hidden>. The std:<--hidden> entry is "
                        "the P1 main retrain column (pre-registered)")
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--wd", default=1e-4, type=float)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--seed_base", default=20261102, type=int)
    # pre-registered gates (§27.3; do not weaken without external review).
    # sanity_tol is a CATASTROPHE gate (wrong head / wrong pool / wrong
    # task restored -> multi-pp deviations), NOT a bit-replay gate: the
    # batchwise evaluation is shuffle-sensitive (measured 0.4-0.7pp mean
    # swings between equally-valid RNG realizations), so bit-wise
    # agreement is physically impossible. The paired same-cache design
    # (below) is what kills that noise for the P1 verdict.
    p.add_argument("--sanity_tol", default=1.0, type=float)
    p.add_argument("--p1_feat_gate", default=0.3, type=float)
    p.add_argument("--p1_lag_gate", default=1.0, type=float)
    p.add_argument("--p2_gap_gate", default=0.05, type=float)
    p.add_argument("--sunken_task", default=2, type=int,
                   help="pre-registered sunken task for the P2 reading")
    return p.parse_args()


_TASK_END_RE = re.compile(r"task_(\d+)_end\.pth$")


def parse_task_id(ckpt_name):
    m = _TASK_END_RE.search(Path(ckpt_name).name)
    if not m:
        raise SystemExit(
            "checkpoint %r is not a task_XX_end.pth boundary file -- this "
            "probe reads task boundaries only" % ckpt_name)
    return int(m.group(1))


def read_matrix(run_dir):
    """accuracy_matrix.csv -> {after_task: {test_task: acc}} (§8 schema)."""
    path = Path(run_dir) / "metrics" / "accuracy_matrix.csv"
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [c.strip() for c in header]  # after_task,T0,T1,...
        for row in reader:
            if not row or not row[0].strip():
                continue
            t = int(row[0])
            vals = {}
            for i, cell in enumerate(row[1:], start=1):
                cell = cell.strip()
                if cell:
                    vals[int(cols[i][1:])] = float(cell)
            out[t] = vals
    return out


def collect_test_zop(client, task, nb_classes):
    """Test-split z_op features (768, pre-anchor) -- native-evaluate
    forward semantics (original pre_logits -> vit(task_id) -> feat),
    sequential loader, no RNG consumption beyond the (deterministic)
    dataset order. Returns (z[N,768] np, y[N] np)."""
    xs, ys = [], []
    loader = DataLoader(client.test_loader[task], batch_size=16,
                        shuffle=False, num_workers=0)
    for input, target in loader:
        input = input.to(client.device, non_blocking=True)
        target = target.long()
        with torch.no_grad():
            cls_features = None
            if client.original_model is not None:
                cls_features = client.original_model(input)["pre_logits"]
            output = client.vit(input, task_id=client.task_id,
                                cls_features=cls_features, train=True)
        z = output["feat"]
        if torch.isnan(z).any():
            raise SystemExit(
                "NaN in test z_op features (client %d task %d) -- probe "
                "bug, refusing to continue" % (client.id, task))
        xs.append(z.detach().cpu().numpy())
        ys.append(target.numpy())
    return np.concatenate(xs, 0), np.concatenate(ys, 0)


def protos_equal_check(server):
    """global_protos is a single broadcast object; all clients must hold
    bitwise-identical copies (fail-closed: divergence = probe bug)."""
    c0 = server.clients[0]
    if c0.global_protos is None:
        raise SystemExit("client 0 global_protos is None after bootstrap")
    for c in server.clients[1:]:
        if c.global_protos is None or set(c.global_protos.keys()) != \
                set(c0.global_protos.keys()):
            raise SystemExit(
                "global_protos key mismatch at client %d -- expected a "
                "single broadcast copy" % c.id)
        for k in c0.global_protos.keys():
            if not np.array_equal(np.asarray(c0.global_protos[k]),
                                  np.asarray(c.global_protos[k])):
                raise SystemExit(
                    "global_protos value mismatch (client %d key %r) -- "
                    "expected a single broadcast copy" % (c.id, k))
    return c0.global_protos


def l2n(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def p2_metrics(client, task, protos, seen_before_all):
    """One (client, task) P2 reading. Returns dict or None (no old classes
    at this boundary -- T0 has nothing to intrude into)."""
    # x1 1536 features on the task's train split: the official
    # proto-collection forward, reusing the client's own cached-input
    # collector (zero logic drift, fail-closed on missing splits).
    snap = rng_state_dict()
    try:
        x1, ys = client._a5_collect_head_inputs(task, feat_only=False)
    finally:
        restore_rng_state(snap)
    if x1 is None:
        return None
    x1 = x1.detach().cpu().numpy().astype(np.float64)
    ys = (ys.detach().cpu().numpy() if torch.is_tensor(ys)
          else np.asarray(ys))
    # official proto collection skips NaN feature rows (Client_DF L753 --
    # x1 DOES have a NaN history); mirror that tolerance here
    ok = ~np.isnan(x1).any(axis=1)
    if not ok.all():
        print(f"[P2] client {client.id} task {task}: skipping "
              f"{int((~ok).sum())} NaN x1 rows (official proto semantics)")
    x1, ys = x1[ok], ys[ok]
    if x1.shape[0] == 0:
        return None

    old_keys = sorted(int(k) for k in protos.keys()
                      if int(k) in seen_before_all
                      and not np.isnan(np.asarray(protos[k])).any())
    nan_skipped = sum(
        1 for k in protos.keys()
        if int(k) in seen_before_all
        and np.isnan(np.asarray(protos[k])).any())
    if not old_keys:
        return None

    p_old = np.stack([np.asarray(protos[k], dtype=np.float64).reshape(-1)
                      for k in old_keys], axis=0)          # [n_old, D]
    p_old_n = l2n(p_old)

    new_classes = [int(x) for x in client.class_mask[task]]
    intru, margins, kept = [], [], []
    for cc in new_classes:
        rows = ys == cc
        if not rows.any():
            continue  # a (client,task) class can have 0 train rows (cifar
                      # public-class protocol; same tolerance as the head)
        p_new = l2n(x1[rows].mean(axis=0))                 # [D]
        sim_old = p_old_n @ p_new                          # [n_old]
        intru.append(float(sim_old.max()))
        kept.append(cc)
        # sample-level margin vs own proto: cos(f, own) - max cos(f, old)
        f = l2n(x1[rows])                                  # [n, D]
        own = f @ p_new                                     # [n]
        near_old = (f @ p_old_n.T).max(axis=1)              # [n]
        margins.append(float((own - near_old).mean()))
    if not intru:
        return None

    # old-old nearest-neighbour reference (how close different OLD class
    # protos sit to each other in this space -- the "healthy" distance)
    oo = p_old_n @ p_old_n.T
    np.fill_diagonal(oo, -np.inf)
    oo_nn = oo.max(axis=1)                                  # per-old nearest
    return {
        "proto_intrusion": float(np.mean(intru)),
        "proto_intrusion_max": float(np.max(intru)),
        "sample_margin": float(np.mean(margins)) if margins else None,
        "old_old_mean": float(oo_nn.mean()),
        "old_old_p95": float(np.percentile(oo_nn, 95)),
        "n_old_protos": len(old_keys),
        "nan_protos_skipped": int(nan_skipped),
        "n_new_protos": len(kept),
    }


def p1_verdict(corrected, feat_gate, lag_gate):
    """corrected = {t: delta_tilde} for t >= 1 (T0 IS the baseline)."""
    if not corrected:
        return "NO_DATA"
    if any(v >= lag_gate for v in corrected.values()):
        return "HEAD-LAG"
    if all(v < feat_gate for v in corrected.values()):
        return "FEAT"
    return "MIXED"


def p2_verdict(per_task, sunken, gap_gate):
    if sunken not in per_task or not per_task[sunken]:
        return "NO_DATA"
    others = [v["proto_intrusion"] for t, v in per_task.items()
              if t != sunken and v]
    if not others:
        return "NO_DATA"
    others_mean = float(np.mean(others))
    sink = per_task[sunken]["proto_intrusion"]
    gap = sink - others_mean
    cond2 = sink > per_task[sunken]["old_old_p95"]
    if gap >= gap_gate or cond2:
        return "INTRUDE"
    return "CLEAN"


def parse_grid(spec, main_hidden):
    """'std:256,std:512,nostd:256,nostd:512' -> [(True,256),(True,512),
    (False,256),(False,512)]. Fail-closed: the pre-registered P1 main
    column std:<main_hidden> must be present (the P1 verdict reads it)."""
    out = []
    seen_main = False
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        tag, sep, h = item.partition(":")
        if not sep or tag not in ("std", "nostd"):
            raise SystemExit("bad --retrain_grid spec %r" % item)
        entry = (tag == "std", int(h))
        if entry == (True, main_hidden):
            seen_main = True
        out.append(entry)
    if not seen_main:
        raise SystemExit(
            "--retrain_grid must contain the pre-registered main column "
            "std:%d" % main_hidden)
    return out


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    for c in ckpts:
        parse_task_id(c)  # fail early on a non-boundary file

    matrix = read_matrix(run_dir)
    hp = dict(hidden=args.hidden, epochs=args.epochs, lr=args.lr,
              wd=args.wd, dropout=args.dropout, batch_size=args.batch_size,
              device=args.device)
    grid = parse_grid(args.retrain_grid, args.hidden)

    rows = []                       # per (checkpoint, client) CSV rows
    p1_per_task, p2_per_task = {}, {}
    sanity = {}

    for ckpt in ckpts:
        t = parse_task_id(ckpt)
        print("=" * 72)
        print(f"[§27] bootstrap {run_dir.name} @ {ckpt} (task {t})")
        env_args, server = bootstrap_server(
            str(run_dir), ckpt, data_path=args.data_path,
            device=args.device)
        for c in server.clients:
            if int(c.task_id) != t:
                raise SystemExit(
                    "client %d task_id=%d != checkpoint task %d -- "
                    "task_XX_end checkpoints must sit at the task boundary"
                    % (c.id, c.task_id, t))
        protos = protos_equal_check(server)
        seen_before_all = set()
        for c in server.clients:
            for k in range(t):
                seen_before_all.update(int(x) for x in c.class_mask[k])

        # ---- P1 native: bit-faithful run_full_evaluation replay ------
        # The matrix numbers come from Server_DF.run_full_evaluation:
        # ONE rng snapshot, then per-client SERIAL evaluate_all_seen_tasks
        # (which calls evaluate_with_margin per seen task, in ascending
        # order), then one restore. The batchwise prompt evaluation is
        # shuffle-sensitive, so a per-client-restart loop here would NOT
        # reproduce the matrix (batch composition differs). Replicating
        # the official serial structure puts the restored checkpoint RNG
        # on the exact same stream -> native must match the matrix row.
        native = {}
        native_full = {}
        snap = rng_state_dict()
        try:
            for c in server.clients:
                results = c.evaluate_all_seen_tasks(env_args.nb_classes)
                native_full[c.id] = results
                if t not in results:
                    raise SystemExit(
                        "client %d: evaluate_all_seen_tasks at task %d has "
                        "no test_task %d entry" % (c.id, t, t))
                native[c.id] = float(results[t])
        finally:
            restore_rng_state(snap)
        native_mean = float(np.mean(list(native.values())))
        # full-row reconciliation: every seen k <= t must match R[t][k]
        # (client mean), not just the boundary cell -- a much stronger
        # drift catcher than the R[t][t]-only gate.
        row = matrix.get(t, {})
        row_checks = {}
        for k in range(t + 1):
            R_tk = row.get(k)
            if R_tk is None:
                raise SystemExit("accuracy_matrix has no R[%d][%d]" % (t, k))
            probe_mean = float(np.mean(
                [native_full[cid][k] for cid in sorted(native_full)]))
            diff = probe_mean - R_tk
            row_checks[k] = {"probe": probe_mean, "matrix": R_tk,
                             "diff": diff}
            if abs(diff) > args.sanity_tol:
                raise SystemExit(
                    "sanity gate FAILED at after_task=%d test_task=%d: "
                    "probe %.4f vs matrix %.4f (|diff| %.4f > %.2f) -- "
                    "probe semantics drifted, refusing to emit verdicts"
                    % (t, k, probe_mean, R_tk, abs(diff), args.sanity_tol))
        sanity[t] = row_checks
        diffs_str = ", ".join(
            "R[%d][%d] d%+.3f" % (t, k, v["diff"])
            for k, v in sorted(row_checks.items()))
        print(f"[P1] native mean {native_mean:.4f} vs matrix "
              f"{row[t]:.4f} -- full-row sanity OK ({diffs_str})")

        # ---- P1 paired comparison on ONE shared feature cache ---------
        # The batchwise evaluation is shuffle-sensitive (measured: two
        # probe structures gave per-client swings of 0.5-0.9pp against
        # the matrix -- the matrix itself is one realization). A serial
        # replay therefore cannot reproduce it bit-wise, and an unpaired
        # delta would drown in implementation noise. The CORRECT design
        # for the head-vs-features question is paired: the restored
        # native refit head and the freshly fitted mlp_local head both
        # consume the SAME cached z_op test features -> batchwise
        # realization noise is common-mode and cancels in the delta.
        # The native refit head is the M2' feat form: 1536 input with the
        # anchor half's first-Linear weights hard-zeroed (Client_DF
        # _a5_refit_one_head), so head(z || 0) is mathematically the
        # native evaluate() head function on the same feature stream.
        retrain = {}
        native_cached = {}
        eval_vs_cached = {}
        variants_by_client = {}
        for c in server.clients:
            snap = rng_state_dict()
            try:
                z_tr, y_tr = c._a5_collect_head_inputs(t, feat_only=True)
            finally:
                restore_rng_state(snap)
            if z_tr is None:
                raise SystemExit(
                    "client %d task %d: empty train cache (protocol bug)"
                    % (c.id, t))
            z_tr = z_tr.detach().cpu().numpy()
            y_tr = (y_tr.detach().cpu().numpy() if torch.is_tensor(y_tr)
                    else np.asarray(y_tr))
            z_te, y_te = collect_test_zop(c, t, env_args.nb_classes)
            classes = [int(x) for x in c.class_mask[t]]

            # (a) native head on the cache (eval mode: eval-side dropout
            # OFF = the corrected-eval semantics; train mode would inject
            # dropout noise into the paired reading)
            head_mod = c.heads[t]
            head_mod.eval()
            dev = next(head_mod.parameters()).device
            zt = torch.tensor(z_te, dtype=torch.float32)
            pad = torch.zeros(zt.shape[0], zt.shape[1], dtype=torch.float32)
            with torch.no_grad():
                logits = head_mod(torch.cat([zt, pad], dim=1).to(dev))
            logits = logits.detach().cpu().numpy()
            if logits.shape[1] < max(classes) + 1:
                raise SystemExit(
                    "client %d: class %d exceeds head output dim %d"
                    % (c.id, max(classes), logits.shape[1]))
            # Chead keeps the full nb_classes layout: output column ==
            # global class id (refit heads are trained masked but never
            # drop units). Integer column indexing (NOT boolean masking)
            # so the argmax->label map is exactly the column list below;
            # using c.class_mask[t]'s original order here previously
            # permuted the labels whenever the mask is not sorted
            # (that bug produced ~14% native-cached readings).
            cols = sorted(classes)
            allowed_logits = logits[:, cols]
            pred_native = np.asarray(
                [cols[i] for i in allowed_logits.argmax(axis=1)])
            native_cached[c.id] = _acc(pred_native, y_te)
            eval_vs_cached[c.id] = native[c.id] - native_cached[c.id]

            # (b) retrain recipe grid on the SAME cache. The
            # std:<args.hidden> entry keeps the historical seed so the P1
            # main column stays comparable across probe versions; every
            # other variant gets its own deterministic seed.
            variant_acc = {}
            for std_flag, h in grid:
                tag = ("std" if std_flag else "nostd") + str(h)
                if (std_flag, h) == (True, args.hidden):
                    vseed = stable_seed(args.seed_base, "p1",
                                        run_dir.name, c.id, t)
                else:
                    vseed = stable_seed(args.seed_base, "p1", tag,
                                        run_dir.name, c.id, t)
                handle = fit_head(
                    z_tr, y_tr, classes, seed=vseed,
                    hidden=h, epochs=args.epochs, lr=args.lr,
                    wd=args.wd, dropout=args.dropout,
                    batch_size=args.batch_size, device=args.device,
                    standardize=std_flag)
                pred = head_predict(handle, z_te, allowed=classes)
                variant_acc[tag] = _acc(pred, y_te)
            retrain[c.id] = variant_acc["std%d" % args.hidden]
            variants_by_client.setdefault(c.id, {})[t] = variant_acc
        retrain_mean = float(np.mean(list(retrain.values())))
        native_cached_mean = float(np.mean(
            list(native_cached.values())))

        # ---- P2 geometric intrusion ----------------------------------
        p2_clients = {}
        for c in server.clients:
            m = p2_metrics(c, t, protos, seen_before_all)
            if m is not None:
                p2_clients[c.id] = m
        if p2_clients:
            p2_per_task[t] = {
                k: float(np.mean([m[k] for m in p2_clients.values()]))
                for k in ("proto_intrusion", "proto_intrusion_max",
                          "sample_margin", "old_old_mean", "old_old_p95")
            }
            p2_per_task[t]["n_old_protos"] = int(max(
                m["n_old_protos"] for m in p2_clients.values()))
            p2_per_task[t]["nan_protos_skipped"] = int(max(
                m["nan_protos_skipped"] for m in p2_clients.values()))
        else:
            p2_per_task[t] = None  # T0: no old classes, nothing to intrude

        raw_delta = retrain_mean - native_cached_mean
        grid_tags = ["%s%d" % ("std" if s else "nostd", h)
                     for s, h in grid]
        variants_mean = {
            tag: float(np.mean(
                [variants_by_client[cid][t][tag]
                 for cid in sorted(variants_by_client)]))
            for tag in grid_tags}
        p1_per_task[t] = {
            "native_eval": native_mean,          # serial replay (info only)
            "native_cached": native_cached_mean,  # paired reading base
            "retrain": retrain_mean,
            "raw_delta": raw_delta,
            "eval_vs_cached_mean": float(np.mean(
                list(eval_vs_cached.values()))),
            "variants_mean": variants_mean,        # §27.6 recipe separation
            "variants_delta": {tag: variants_mean[tag] - native_cached_mean
                               for tag in grid_tags},
        }
        print(f"[P1] paired: native_cached {native_cached_mean:.4f} vs "
              f"retrain {retrain_mean:.4f} -> raw delta {raw_delta:+.4f} "
              f"(eval-replay {native_mean:.4f}, eval-cached swing "
              f"{p1_per_task[t]['eval_vs_cached_mean']:+.4f}); P2 "
              + (json.dumps(p2_per_task[t]) if p2_per_task[t] else "n/a"))

        for c in server.clients:
            m = p2_clients.get(c.id, {})
            row = {
                "checkpoint": ckpt, "task": t, "client": c.id,
                "native_eval": native[c.id],
                "native_cached": native_cached[c.id],
                "retrain": retrain[c.id],
                "raw_delta": retrain[c.id] - native_cached[c.id],
                "eval_vs_cached": eval_vs_cached[c.id],
                "proto_intrusion": m.get("proto_intrusion"),
                "old_old_mean": m.get("old_old_mean"),
                "old_old_p95": m.get("old_old_p95"),
                "sample_margin": m.get("sample_margin"),
            }
            for tag in grid_tags:
                row["retrain_" + tag] = variants_by_client[c.id][t][tag]
            rows.append(row)

    # ---- verdicts (§27.3 pre-registered) ----------------------------
    t0 = min(p1_per_task.keys())
    delta0 = p1_per_task[t0]["raw_delta"]
    corrected = {t: v["raw_delta"] - delta0
                 for t, v in p1_per_task.items() if t != t0}
    v1 = p1_verdict(corrected, args.p1_feat_gate, args.p1_lag_gate)
    v2 = p2_verdict({t: v for t, v in p2_per_task.items() if v},
                    args.sunken_task, args.p2_gap_gate)
    gate = ("PASS -> §27.6 判据臂定稿"
            if (v1 == "FEAT" and v2 == "INTRUDE")
            else "FAIL(P1=%s, P2=%s) -> PGC 判死" % (v1, v2))

    summary = {
        "run": str(run_dir), "checkpoints": ckpts, "device": args.device,
        "hyper": {k: v for k, v in hp.items()},
        "gates": {"sanity_tol": args.sanity_tol,
                  "p1_feat": args.p1_feat_gate, "p1_lag": args.p1_lag_gate,
                  "p2_gap": args.p2_gap_gate, "sunken_task": args.sunken_task},
        "sanity": sanity,
        "P1": {"per_task": p1_per_task, "delta0": delta0,
               "corrected": corrected, "verdict": v1},
        "P2": {"per_task": p2_per_task, "verdict": v2},
        "final_gate": gate,
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out_json = Path(args.summary_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)

    print("=" * 72)
    print(f"[§27] P1 verdict: {v1}  (delta0={delta0:+.4f}, "
          f"corrected={ {k: round(v, 4) for k, v in corrected.items()} })")
    print(f"[§27] P2 verdict: {v2}")
    print(f"[§27] FINAL GATE: {gate}")
    print(f"[§27] wrote {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
