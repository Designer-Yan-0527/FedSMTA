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

Protocol (all on TRAIN split only, fixed sample order, same checkpoint):
  1. For batch sizes B in {1, 8, 16, 32, 64} extract z_op for the same
     samples in the same order, batching sequentially.
  2. For every sample x and every B > 1:
         Delta_B(x) = ||z_op^(B)(x) - z_op^(1)(x)||_2
                      / (||z_op^(1)(x)||_2 + eps)
     plus cosine similarity between the two.
  3. Control at fixed B=16: two DIFFERENT deterministic compositions
     (identity order vs fixed-seed permutation), same samples, same batch
     size -> isolates the pure batch-composition effect from the
     batch-size effect.
  4. Aggregate mean/median/std/p95/max per (client, task, B) and overall.

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
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diag_utils import bootstrap_server, resolve_checkpoint, REPO_ROOT

EPS = 1e-8
DEFAULT_BATCH_SIZES = (1, 8, 16, 32, 64)
CONTROL_BATCH_SIZE = 16
CONTROL_PERM_SEED = 12345


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
    p.add_argument("--max_samples", type=int, default=256,
                   help="cap on samples per (client, task) to keep the "
                        "B=1 pass cheap (default: 256)")
    p.add_argument("--out_dir", default="metrics",
                   help="output directory (default: metrics)")
    p.add_argument("--data_path", default=None,
                   help="override dataset root (default: value from args.json)")
    p.add_argument("--device", default=None,
                   help="override device (default: from args.json)")
    return p.parse_args()


def summarize(values):
    """mean/median/std/p95/max of a 1-D array (spec §5.9 aggregate)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {k: float("nan") for k in
                ("mean", "median", "std", "p95", "max")}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def extract_z_op(client, server, images, task_id, batch_size, device):
    """Forward a fixed image tensor through the client's CURRENT prompt
    state (same call signature as extract_features.py) and return z_op as
    an [N, 768] float32 numpy array. `images` is a CPU float tensor in a
    FIXED order; batches are sequential slices of it, so the composition
    is fully determined by (order, batch_size)."""
    n = images.shape[0]
    out = np.zeros((n, 768), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            inp = images[i:i + batch_size].to(device, non_blocking=True)
            out_ref = server.origin_model(inp)
            out_op = client.vit(
                inp, task_id=task_id,
                cls_features=out_ref["pre_logits"], train=True)
            b = inp.shape[0]
            out[i:i + b] = out_op["feat"].float().cpu().numpy()
    return out


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


def main():
    args_cli = parse_args()
    if 1 not in args_cli.batch_sizes:
        raise SystemExit("--batch_sizes must contain 1 (the baseline)")
    if CONTROL_BATCH_SIZE not in args_cli.batch_sizes:
        args_cli.batch_sizes.append(CONTROL_BATCH_SIZE)

    ckpt_path = resolve_checkpoint(args_cli.run_dir, args_cli.checkpoint)
    run_args, server = bootstrap_server(
        args_cli.run_dir, checkpoint=ckpt_path,
        data_path=args_cli.data_path, device=args_cli.device)
    device = torch.device(run_args.device)
    server.origin_model.to(device)
    server.origin_model.eval()

    rows = []          # CSV rows
    json_detail = []   # per (client, task) JSON records
    overall = {}       # key -> (delta, cos) arrays
    t0 = time.time()

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
            if args_cli.max_samples > 0:
                indices = indices[:args_cli.max_samples]
            subset = Subset(client.train_data[t], indices)
            # materialize the fixed-order image tensor once
            images = []
            for inp, _ in DataLoader(subset, batch_size=64, shuffle=False,
                                     num_workers=run_args.num_workers):
                images.append(inp)
            images = torch.cat(images, dim=0)
            n = images.shape[0]

            # 1) batch-size sweep (sequential composition per B)
            z_by_b = {}
            for bsz in sorted(set(args_cli.batch_sizes)):
                z_by_b[bsz] = extract_z_op(
                    client, server, images, t, bsz, device)
            base = z_by_b[1]

            # 2) control: fixed B, different deterministic composition
            perm = np.random.RandomState(CONTROL_PERM_SEED).permutation(n)
            z_perm = extract_z_op(
                client, server, images[torch.from_numpy(perm)],
                t, CONTROL_BATCH_SIZE, device)
            # map permuted rows back to original sample positions
            z_ctrl = np.zeros_like(z_perm)
            z_ctrl[perm] = z_perm

            record = {"client": client.id, "task": t, "n_samples": n,
                      "batch_sizes": {}}
            for bsz in sorted(set(args_cli.batch_sizes)):
                if bsz == 1:
                    continue
                delta, cos = pair_metrics(z_by_b[bsz], base)
                row = {"scope": f"c{client.id}_t{t}", "client": client.id,
                       "task": t, "n_samples": n,
                       "comparison": f"B={bsz}_vs_B=1"}
                for k, v in summarize(delta).items():
                    row[f"delta_{k}"] = v
                for k, v in summarize(cos).items():
                    row[f"cos_{k}"] = v
                rows.append(row)
                record["batch_sizes"][str(bsz)] = {
                    "delta": summarize(delta), "cos": summarize(cos)}
                overall.setdefault(f"B={bsz}_vs_B=1", ([], []))
                overall[f"B={bsz}_vs_B=1"][0].append(delta)
                overall[f"B={bsz}_vs_B=1"][1].append(cos)

            # control comparison: same B, permuted composition
            delta_c, cos_c = pair_metrics(z_ctrl, z_by_b[CONTROL_BATCH_SIZE])
            row = {"scope": f"c{client.id}_t{t}", "client": client.id,
                   "task": t, "n_samples": n,
                   "comparison": f"B={CONTROL_BATCH_SIZE}_perm_vs_B={CONTROL_BATCH_SIZE}_identity"}
            for k, v in summarize(delta_c).items():
                row[f"delta_{k}"] = v
            for k, v in summarize(cos_c).items():
                row[f"cos_{k}"] = v
            rows.append(row)
            record["control"] = {
                "comparison": f"B={CONTROL_BATCH_SIZE}: permuted_vs_identity",
                "perm_seed": CONTROL_PERM_SEED,
                "delta": summarize(delta_c), "cos": summarize(cos_c)}
            overall.setdefault(
                f"B={CONTROL_BATCH_SIZE}_perm_vs_identity", ([], []))
            overall[f"B={CONTROL_BATCH_SIZE}_perm_vs_identity"][0].append(delta_c)
            overall[f"B={CONTROL_BATCH_SIZE}_perm_vs_identity"][1].append(cos_c)

            json_detail.append(record)
            print(f"client {client.id} task {t}: {n} samples done "
                  f"({time.time()-t0:.0f}s)")

    # overall aggregates across all (client, task)
    overall_summary = {}
    for key, (deltas, coss) in overall.items():
        d = np.concatenate(deltas) if deltas else np.zeros(0)
        c = np.concatenate(coss) if coss else np.zeros(0)
        overall_summary[key] = {"n_samples": int(d.size),
                                "delta": summarize(d), "cos": summarize(c)}
        row = {"scope": "ALL", "client": -1, "task": -1,
               "n_samples": int(d.size), "comparison": key}
        for k, v in summarize(d).items():
            row[f"delta_{k}"] = v
        for k, v in summarize(c).items():
            row[f"cos_{k}"] = v
        rows.append(row)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_batch_sensitivity.csv"
    json_path = out_dir / "feature_batch_sensitivity.json"

    fieldnames = ["scope", "client", "task", "n_samples", "comparison"] + \
        [f"delta_{k}" for k in ("mean", "median", "std", "p95", "max")] + \
        [f"cos_{k}" for k in ("mean", "median", "std", "p95", "max")]
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
        "control": {"batch_size": CONTROL_BATCH_SIZE,
                    "perm_seed": CONTROL_PERM_SEED},
        "split": "train",
        "max_samples_per_client_task": args_cli.max_samples,
        "overall": overall_summary,
        "per_client_task": json_detail,
        "note": "Delta_B(x)=||z_op^B(x)-z_op^1(x)||/||z_op^1(x)||; "
                "control isolates batch-composition effect at fixed B. "
                "Diagnostic only - interpretation and any z_op/z_ref "
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
              f"delta_max={s['delta']['max']:.6f} "
              f"cos_mean={s['cos']['mean']:.6f}")


if __name__ == "__main__":
    main()
