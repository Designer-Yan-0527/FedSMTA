"""Phase 1-precheck: z_op batch sensitivity diagnostic (roadmap §5.9).

FedTA runs with batchwise_prompt=True: Global_Prompt.forward() aggregates
prompt frequencies over the whole batch and applies the major prompt to
every sample, so

    z_op(x) = f(x; B) != f(x)

The Phase-1 multimodality analysis assumes z_op is a stable sample-level
semantic representation. If z_op(x) changes with batch composition, a
later K=2 finding could be a batchwise-prompt-routing artifact instead of
real semantic multimodality. This script measures that instability BEFORE
any GMM is fitted.

Protocol (all on TRAIN split only, same checkpoint, cudnn deterministic):
  1. Class-stratified deterministic sample selection per (client, task)
     (fixed seed; NOT "first N sorted indices" -- those may carry
     dataset-order structure that biases the composition study).
  2. Batch-size sweep B in {1, 8, 16, 32, 64}: extract z_op AND z_ref for
     the same fixed-order samples.
       - z_ref control: origin_model is batch-independent in principle,
         so Delta_B^ref is the GPU numerical floor. Only
         Delta_B^op >> Delta_B^ref justifies attributing drift to
         batchwise prompt routing.
  3. Repeat-run floor: B=16, identical order, run twice -- pure GPU
     nondeterminism floor for the z_op path.
  4. Composition controls: B fixed at 16, several deterministic
     permutations (--composition_seeds, default 5 seeds) vs identity
     order -- isolates the batch-COMPOSITION effect from the batch-SIZE
     effect and avoids underestimation from a single lucky permutation.
  5. Per-sample Delta_B(x) = ||z^B(x) - z^1(x)||_2 / ||z^1(x)||_2 plus
     cosine similarity; aggregated mean/median/std/p05/p95/min/max per
     (client, task) and overall (cosine: p05/min are the informative
     tail; delta: p95/max).

Diagnostic only: no training logic is touched, no z_ref switch, no
batchwise_prompt changes (decision comes after seeing the numbers).

Usage (from repo root, on the server):
  python diagnostics/check_feature_batch_sensitivity.py \
      --run_dir output/regression/cifar100/fedta/reg_full/seed_42 \
      --checkpoint checkpoints/latest.pth \
      --device cuda
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diag_utils import bootstrap_server, resolve_checkpoint

EPS = 1e-8
DEFAULT_BATCH_SIZES = (1, 8, 16, 32, 64)
CONTROL_BATCH_SIZE = 16
SAMPLING_SEED = 2026
STAT_KEYS = ("mean", "median", "std", "p05", "p95", "min", "max")


def parse_args():
    p = argparse.ArgumentParser("Phase-1 precheck: z_op batch sensitivity")
    p.add_argument("--run_dir", required=True,
                   help="run directory (.../{data_name}/{method}/{run_name}/seed_{seed})")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint path or name relative to run_dir "
                        "(default: checkpoints/latest.pth)")
    p.add_argument("--batch_sizes", type=int, nargs="+",
                   default=list(DEFAULT_BATCH_SIZES),
                   help="batch sizes to audit (must contain 1 as baseline)")
    p.add_argument("--composition_seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4],
                   help="seeds for the B=16 permutation composition "
                        "controls (default: 5 seeds)")
    p.add_argument("--max_samples", type=int, default=256,
                   help="cap on samples per (client, task), class-"
                        "stratified deterministic sampling (default: 256)")
    p.add_argument("--out_dir", default="metrics",
                   help="output directory (default: metrics)")
    p.add_argument("--data_path", default=None,
                   help="override dataset root (default: value from args.json)")
    p.add_argument("--device", default=None,
                   help="override device (default: from args.json)")
    return p.parse_args()


def summarize(values):
    """mean/median/std/p05/p95/min/max of a 1-D array.

    For relative distance the bad tail is p95/max; for cosine similarity
    the bad tail is p05/min -- reporting both tails covers both metrics."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {k: float("nan") for k in STAT_KEYS}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def stratified_selection(targets, max_samples, seed):
    """Deterministic class-stratified positions into the local sample list.

    Proportional per-class rounding (>=1 per class), fixed RandomState,
    sorted output order. Never depends on dataset file order structure
    beyond the (already fixed) split indices order."""
    targets = np.asarray(targets)
    n = len(targets)
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    picked = []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        take = max(1, int(round(max_samples * len(idx) / n)))
        take = min(take, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    picked = sorted(set(picked))
    if len(picked) > max_samples:  # proportional rounding overflow
        picked = picked[:max_samples]
    return np.asarray(picked, dtype=np.int64)


def extract_feats(client, server, images, task_id, batch_size, device):
    """Forward a fixed image tensor, batched sequentially, through BOTH
    the frozen origin model (z_ref) and the client's CURRENT prompt state
    (z_op; same call signature as extract_features.py). Returns
    (z_ref, z_op) as [N, d] float32 arrays, row-aligned with `images`."""
    zr, zo = [], []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            inp = images[i:i + batch_size].to(device, non_blocking=True)
            out_ref = server.origin_model(inp)
            out_op = client.vit(
                inp, task_id=task_id,
                cls_features=out_ref["pre_logits"], train=True)
            zr.append(out_ref["feat"].float().cpu().numpy())
            zo.append(out_op["feat"].float().cpu().numpy())
    return np.concatenate(zr), np.concatenate(zo)


def pair_metrics(z_a, z_b):
    """Per-sample relative distance + cosine similarity of two [N, d]
    feature matrices (row-matched)."""
    norm_b = np.linalg.norm(z_b, axis=1)
    dist = np.linalg.norm(z_a - z_b, axis=1)
    delta = dist / (norm_b + EPS)
    a = z_a / (np.linalg.norm(z_a, axis=1, keepdims=True) + EPS)
    b = z_b / (np.linalg.norm(z_b, axis=1, keepdims=True) + EPS)
    cos = np.sum(a * b, axis=1)
    return delta, cos


def make_row(scope, client_id, task, n, comparison, delta, cos):
    row = {"scope": scope, "client": client_id, "task": task,
           "n_samples": n, "comparison": comparison}
    for k, v in summarize(delta).items():
        row[f"delta_{k}"] = v
    for k, v in summarize(cos).items():
        row[f"cos_{k}"] = v
    return row


def main():
    args_cli = parse_args()
    if 1 not in args_cli.batch_sizes:
        raise SystemExit("--batch_sizes must contain 1 (the baseline)")
    if CONTROL_BATCH_SIZE not in args_cli.batch_sizes:
        args_cli.batch_sizes.append(CONTROL_BATCH_SIZE)

    # CUBLAS deterministic workspace must be configured BEFORE any CUDA
    # context is created (i.e. before bootstrap builds models / loads data)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    ckpt_path = resolve_checkpoint(args_cli.run_dir, args_cli.checkpoint)
    run_args, server = bootstrap_server(
        args_cli.run_dir, checkpoint=ckpt_path,
        data_path=args_cli.data_path, device=args_cli.device)

    # deterministic flags MUST be applied AFTER bootstrap_server: diag_utils
    # resets args.deterministic=False and build_server() ->
    # setup_determinism(False) re-enables cudnn.benchmark=True, silently
    # undoing any flags set before bootstrap. The 1e-3..1e-2 level drift we
    # measure here demands the nondeterminism floor be minimized.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(run_args.device)
    server.origin_model.to(device)
    server.origin_model.eval()

    rows = []
    overall = {}  # comparison key -> list of per-client-task arrays
    t0 = time.time()

    def collect(comparison, delta, cos):
        overall.setdefault(comparison, ([], []))
        overall[comparison][0].append(delta)
        overall[comparison][1].append(cos)

    for client in server.clients:
        if client.task_id < 0:
            continue
        # the client's CURRENT prompts (end-of-training state)
        if client.prompts is not None:
            client.vit.load_prompts(client.prompts)
        for t in range(client.task_id + 1):
            if t not in client.data_split_indices:
                continue
            indices = sorted(int(i) for i in
                             client.data_split_indices[t]["train"])
            if not indices:
                continue
            subset = Subset(client.train_data[t], indices)
            # materialize the fixed-order images + targets once
            xs, ys = [], []
            for inp, tgt in DataLoader(subset, batch_size=64, shuffle=False,
                                       num_workers=run_args.num_workers):
                xs.append(inp)
                ys.append(tgt.numpy())
            images_all = torch.cat(xs, dim=0)
            targets_all = np.concatenate(ys)

            # class-stratified deterministic sampling (roadmap audit P0-1D)
            sel = stratified_selection(
                targets_all, args_cli.max_samples, SAMPLING_SEED)
            images = images_all[torch.from_numpy(sel)]
            n = images.shape[0]
            scope = f"c{client.id}_t{t}"

            # 1) batch-size sweep: z_ref (numerical control) and z_op
            z_ref_by_b, z_op_by_b = {}, {}
            for bsz in sorted(set(args_cli.batch_sizes)):
                z_ref_by_b[bsz], z_op_by_b[bsz] = extract_feats(
                    client, server, images, t, bsz, device)
            for bsz in sorted(set(args_cli.batch_sizes)):
                if bsz == 1:
                    continue
                for name, table in (("zref", z_ref_by_b), ("zop", z_op_by_b)):
                    delta, cos = pair_metrics(table[bsz], table[1])
                    comp = f"{name}_B={bsz}_vs_B=1"
                    rows.append(make_row(scope, client.id, t, n, comp,
                                         delta, cos))
                    collect(comp, delta, cos)

            # 2) repeat-run floor: identical batching, run twice
            zr_rep, zo_rep = extract_feats(
                client, server, images, t, CONTROL_BATCH_SIZE, device)
            delta, cos = pair_metrics(zo_rep, z_op_by_b[CONTROL_BATCH_SIZE])
            comp = f"zop_repeat_B={CONTROL_BATCH_SIZE}_run2_vs_run1"
            rows.append(make_row(scope, client.id, t, n, comp, delta, cos))
            collect(comp, delta, cos)
            delta, cos = pair_metrics(zr_rep, z_ref_by_b[CONTROL_BATCH_SIZE])
            comp = f"zref_repeat_B={CONTROL_BATCH_SIZE}_run2_vs_run1"
            rows.append(make_row(scope, client.id, t, n, comp, delta, cos))
            collect(comp, delta, cos)

            # 3) composition controls: fixed B, permuted deterministic order
            for seed in args_cli.composition_seeds:
                perm = np.random.RandomState(seed).permutation(n)
                _, zo_perm = extract_feats(
                    client, server, images[torch.from_numpy(perm)],
                    t, CONTROL_BATCH_SIZE, device)
                zo_ctrl = np.zeros_like(zo_perm)  # map back to sample pos
                zo_ctrl[perm] = zo_perm
                delta, cos = pair_metrics(
                    zo_ctrl, z_op_by_b[CONTROL_BATCH_SIZE])
                comp = (f"zop_perm{seed}_B={CONTROL_BATCH_SIZE}"
                        f"_vs_identity")
                rows.append(make_row(scope, client.id, t, n, comp,
                                     delta, cos))
                collect(comp, delta, cos)

            print(f"client {client.id} task {t}: {n} samples done "
                  f"({time.time()-t0:.0f}s)")

    # overall aggregates across all (client, task)
    overall_summary = {}
    for key, (deltas, coss) in overall.items():
        d = np.concatenate(deltas) if deltas else np.zeros(0)
        c = np.concatenate(coss) if coss else np.zeros(0)
        overall_summary[key] = {"n_samples": int(d.size),
                                "delta": summarize(d), "cos": summarize(c)}
        rows.append(make_row("ALL", -1, -1, int(d.size), key, d, c))

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_batch_sensitivity.csv"
    json_path = out_dir / "feature_batch_sensitivity.json"

    fieldnames = ["scope", "client", "task", "n_samples", "comparison"] + \
        [f"delta_{k}" for k in STAT_KEYS] + \
        [f"cos_{k}" for k in STAT_KEYS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    payload = {
        "run_dir": str(Path(args_cli.run_dir).resolve()),
        "checkpoint": str(ckpt_path),
        "round": int(getattr(server, "start_round", 0)) - 1,
        "data_name": run_args.data_name,
        "seed": run_args.seed,
        "batch_sizes": sorted(set(args_cli.batch_sizes)),
        "composition_seeds": list(args_cli.composition_seeds),
        "control_batch_size": CONTROL_BATCH_SIZE,
        "sampling": {"strategy": "class-stratified deterministic",
                     "seed": SAMPLING_SEED,
                     "max_samples_per_client_task": args_cli.max_samples},
        "cudnn": {  # actual runtime state (read back, never hardcoded)
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG", ""),
        },
        "split": "train",
        "overall": overall_summary,
        "reading_guide": {
            "zref_B=*_vs_B=1": "numerical floor (origin_model is "
                               "batch-independent in principle)",
            "zop_repeat_B=16_run2_vs_run1": "GPU nondeterminism floor "
                                            "for the z_op path",
            "zop_B=*_vs_B=1": "batch-SIZE effect on z_op",
            "zop_perm*_B=16_vs_identity": "batch-COMPOSITION effect at "
                                          "fixed B (batchwise prompt "
                                          "routing)",
            "attribution rule": "blame batchwise prompting only if "
                                "zop drift >> both floors",
        },
        "note": "Diagnostic only - interpretation and any z_op/z_ref "
                "decision come after review (roadmap §5.9).",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nsaved {csv_path}")
    print(f"saved {json_path}")
    print("\n===== overall summary (all client-tasks pooled) =====")
    for key, s in overall_summary.items():
        print(f"{key}: delta_mean={s['delta']['mean']:.6f} "
              f"delta_p95={s['delta']['p95']:.6f} "
              f"cos_p05={s['cos']['p05']:.6f} "
              f"cos_min={s['cos']['min']:.6f}")


if __name__ == "__main__":
    main()
