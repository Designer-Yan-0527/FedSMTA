"""Phase 4 A1: historical key/anchor snapshot evaluation (zero retraining).

Deployment-time generation repair: when client c evaluates an OLD task t,
it uses the key/anchor generation that learned task t (from
checkpoints/task_t_end.pth), while everything else (prompt, task head,
protos) stays at the current generation. The current task is evaluated
with the unmodified deployment state.

This is exactly the "co" combination of the 2x2 protocol
(compatibility_drift.py) applied to ALL old tasks in one pass, so the
pre-registered 2x2 co values are the expected outcome (e.g. T1 71.4 ->
93.9 on seed 42).

For every (client, old task) the script also evaluates the baseline
deployment state (current key/anchor + current prompt) under the SAME
torch seed, so delta = a1 - baseline is free of batchwise-prompt voting
noise (~+-2pp).

Outputs:
  rows CSV:    client_id, task, ka_generation, accuracy,
               margin_mean, margin_median, margin_p10
  summary CSV: task, n_clients, a1_acc, baseline_acc, delta
               (+ console AA / old-task mean / recovery report)

Usage (from repo root, on the server):
  python diagnostics/a1_snapshot_eval.py \
      --run_dir output/cifar100/fedta/<run_name>/seed_42 \
      --out_csv diagnostics_output/a1_snapshot_eval.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from diag_utils import bootstrap_server
from runtime_utils import load_checkpoint_file, rng_state_dict, restore_rng_state


def parse_args():
    p = argparse.ArgumentParser("Phase 4 A1 snapshot evaluation")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--cur_ckpt", default=None,
                   help="current checkpoint (default: checkpoints/latest.pth)")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def resolve(run_dir, ckpt):
    path = Path(ckpt)
    if not path.is_absolute():
        path = Path(run_dir) / ckpt
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return str(path.resolve())


def eval_client_task(client, task, nb_classes, tag, rows, ka=None):
    """Evaluate one (client, task) and append a row. ka = optional
    {key, anchor_pool} state dict to load before evaluating (A1 arm).

    Strict combo order (same as compatibility_drift.py so A1 rows are
    directly comparable with the 2x2 grid):
      load current prompt -> load KA -> set seed -> evaluate."""
    if client.prompts is not None:
        client.vit.load_prompts(client.prompts)
    if ka is not None:
        client.model.load_state_dict(ka, strict=False)  # head untouched
    torch.manual_seed(9700 + 100 * client.id + task)
    acc = client.evaluate_with_margin(task, nb_classes)
    stats = client.last_margin_stats or {}
    rows.append({
        "client_id": client.id,
        "task": task,
        "ka_generation": tag,
        "accuracy": f"{acc:.2f}",
        "margin_mean": f"{stats.get('mean', float('nan')):.4f}",
        "margin_median": f"{stats.get('median', float('nan')):.4f}",
        "margin_p10": f"{stats.get('p10', float('nan')):.4f}",
    })
    print(f"client {client.id} task {task}: ka={tag:8s} acc={acc:.2f}")
    return acc


def main():
    args = parse_args()
    cur_path = (resolve(args.run_dir, args.cur_ckpt)
                if args.cur_ckpt
                else resolve(args.run_dir, "checkpoints/latest.pth"))

    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=cur_path,
        data_path=args.data_path, device=args.device)
    nb_classes = run_args.nb_classes

    rng_snapshot = rng_state_dict()
    rows = []
    try:
        for client in server.clients:
            cur_task = client.task_id
            if cur_task is None or cur_task < 0:
                print(f"client {client.id}: no task in checkpoint, skipped")
                continue

            # current key/anchor, CLONED (state_dict shares storage with the
            # live parameters; loading old generations copies in place)
            cur_ka = {k: v.clone() for k, v in client.model.state_dict().items()}

            # old-task generations from task_XX_end.pth
            old_ckpts = {}
            for t in range(cur_task):
                tpath = resolve(args.run_dir, f"checkpoints/task_{t:02d}_end.pth")
                old_ckpts[t] = load_checkpoint_file(tpath)["clients"][client.id]

            # current prompt is the deployment prompt (already loaded by
            # bootstrap); keep it for every evaluation below
            for t in range(cur_task + 1):
                if t < cur_task:
                    old_state = old_ckpts[t]
                    if old_state.get("tail_anchor_model") is None:
                        print(f"client {client.id}: task {t} KA missing, skipped")
                        continue
                    if client.heads[t] is None or t >= len(client.test_loader):
                        print(f"client {client.id}: task {t} head/loader missing, skipped")
                        continue
                    old_ka = old_state["tail_anchor_model"]
                    # A1: old-task KA generation + current prompt
                    eval_client_task(client, t, nb_classes, f"gen{t}", rows,
                                     ka={"key": old_ka["key"],
                                         "anchor_pool": old_ka["anchor_pool"]})
                    # baseline deployment state under the SAME seed
                    eval_client_task(client, t, nb_classes, "cur", rows,
                                     ka={"key": cur_ka["key"],
                                         "anchor_pool": cur_ka["anchor_pool"]})
                else:
                    # current task: unmodified deployment state
                    eval_client_task(client, t, nb_classes, "cur", rows)
    finally:
        restore_rng_state(rng_snapshot)

    # ---- write rows ----
    out_csv = args.out_csv or str(
        Path(args.run_dir) / "metrics" / "a1_snapshot_eval.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out_csv} ({len(rows)} rows)")

    # ---- per-task summary + headline metrics ----
    tasks = sorted({r["task"] for r in rows})
    summary = []
    for t in tasks:
        a1 = [float(r["accuracy"]) for r in rows
              if r["task"] == t and r["ka_generation"] != "cur"]
        base = [float(r["accuracy"]) for r in rows
                if r["task"] == t and r["ka_generation"] == "cur"]
        if not a1:  # current task: only deployment eval exists
            a1_mean, base_mean, delta = None, float(np.mean(base)), None
        else:
            a1_mean, base_mean = float(np.mean(a1)), float(np.mean(base))
            delta = a1_mean - base_mean
        summary.append({"task": t, "n_clients": len(base),
                        "a1_acc": a1_mean, "baseline_acc": base_mean,
                        "delta": delta})

    sum_csv = str(Path(out_csv).with_name(
        Path(out_csv).stem + "_summary.csv"))
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "n_clients", "a1_acc",
                                          "baseline_acc", "delta"])
        w.writeheader()
        w.writerows(summary)
    print(f"saved {sum_csv}")

    print("\n===== A1 snapshot evaluation (client-mean) =====")
    print(f"{'task':>4} {'A1 (gen t)':>11} {'baseline':>9} {'delta':>7}")
    for s in summary:
        a1 = f"{s['a1_acc']:.2f}" if s["a1_acc"] is not None else "   -"
        dl = f"{s['delta']:+.2f}" if s["delta"] is not None else "   -"
        print(f"{s['task']:>4} {a1:>11} {s['baseline_acc']:>9.2f} {dl:>7}")
    old = [s for s in summary if s["delta"] is not None]
    if old:
        rec = float(np.mean([s["delta"] for s in old]))
        print(f"\nold-task mean recovery (A1 - baseline): {rec:+.2f} pp "
              f"over {len(old)} tasks")
    aa_a1 = float(np.mean([s["a1_acc"] if s["a1_acc"] is not None
                           else s["baseline_acc"] for s in summary]))
    aa_base = float(np.mean([s["baseline_acc"] for s in summary]))
    print(f"AA  (A1): {aa_a1:.2f}   AA (baseline): {aa_base:.2f}")


if __name__ == "__main__":
    main()
