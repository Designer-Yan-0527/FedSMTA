"""Phase 1 diagnostic: compatibility 2x2 drift.

At task end the run already snapshots (inside every checkpoint):
  client prompts, Tail_Anchor key, anchor_pool, and per-task Chead.

This script loads an OLD checkpoint (end of task r) and the CURRENT
checkpoint (end of task t > r) and evaluates every client on its task-r
test set under the 4 combinations:

              old key/anchor     current key/anchor
  old prompt      (1)                (2)
  cur prompt      (3)                (4)

The task-specific Chead of task r is ALWAYS used (per the Phase-1 spec:
"the old task head always uses the corresponding task-specific Chead").

  (1) vs (4)  total drift
  (1) vs (2)  anchor/key drift alone
  (1) vs (3)  prompt drift alone

Output CSV: client_id, old_task, new_task, prompt, key_anchor,
accuracy, margin_mean, margin_median, margin_p10.

Usage (from repo root, on the server):
  python diagnostics/compatibility_drift.py \
      --run_dir output/cifar100/fedta/<run_name>/seed_42 \
      --old_ckpt checkpoints/task_00_end.pth \
      --new_ckpt checkpoints/task_03_end.pth \
      --out_csv diagnostics_output/compatibility_2x2.csv
"""

import argparse
import csv
from pathlib import Path

import torch

from diag_utils import bootstrap_server
from runtime_utils import (load_checkpoint_file, clone_prompt_template,
                           rng_state_dict, restore_rng_state)


def parse_args():
    p = argparse.ArgumentParser("Phase-1 compatibility 2x2 diagnostic")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--old_ckpt", required=True,
                   help="checkpoint at the END of the old task (relative to run_dir or absolute)")
    p.add_argument("--new_ckpt", default=None,
                   help="current checkpoint (default: checkpoints/latest.pth)")
    p.add_argument("--old_task", default=None, type=int,
                   help="override the old task id (default: task id stored in old_ckpt)")
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
    return str(path)


def main():
    args = parse_args()
    old_path = resolve(args.run_dir, args.old_ckpt)
    new_path = (resolve(args.run_dir, args.new_ckpt)
                if args.new_ckpt
                else resolve(args.run_dir, "checkpoints/latest.pth"))

    # environment from the CURRENT checkpoint (heads / test loaders /
    # current key+anchor / current prompts all restored)
    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=new_path,
        data_path=args.data_path, device=args.device)
    nb_classes = run_args.nb_classes

    old_ckpt = load_checkpoint_file(old_path)
    new_ckpt = load_checkpoint_file(new_path)

    # RNG guard: the diagnostic evaluation must not matter, but keep the
    # stream clean anyway (cheap).
    rng_snapshot = rng_state_dict()
    rows = []
    try:
        for client in server.clients:
            cid = client.id
            old_state = old_ckpt["clients"][cid]
            old_task = (args.old_task if args.old_task is not None
                        else old_state["task_id"])
            if old_task is None or old_task < 0:
                print(f"client {cid}: no old task in checkpoint, skipped")
                continue
            if old_task >= len(client.test_loader) or \
                    client.heads[old_task] is None:
                print(f"client {cid}: old task {old_task} not seen in the "
                      f"current checkpoint, skipped")
                continue

            # --- old prompt module (state -> module via vit template) ---
            old_prompt = None
            if old_state.get("prompts") is not None:
                old_prompt = clone_prompt_template(client.vit)
                old_prompt.load_state_dict(old_state["prompts"])
            cur_prompt = client.prompts  # already a module (or None)

            # --- key / anchor tensors ---
            old_ka = old_state["tail_anchor_model"]
            cur_ka = client.model.state_dict()

            prompt_variants = [("old", old_prompt), ("current", cur_prompt)]
            ka_variants = [("old", old_ka), ("current", cur_ka)]

            for (p_name, prompt_mod) in prompt_variants:
                if prompt_mod is None:
                    print(f"client {cid}: {p_name} prompts missing, combo skipped")
                    continue
                for (k_name, ka) in ka_variants:
                    client.vit.load_prompts(prompt_mod)
                    client.model.load_state_dict(
                        {"key": ka["key"], "anchor_pool": ka["anchor_pool"]},
                        strict=False)  # head untouched: evaluate() loads heads[old_task]
                    acc = client.evaluate(old_task, nb_classes)
                    stats = client.last_margin_stats or {}
                    rows.append({
                        "client_id": cid,
                        "old_task": old_task,
                        "new_task": client.task_id,
                        "prompt": p_name,
                        "key_anchor": k_name,
                        "accuracy": f"{acc:.2f}",
                        "margin_mean": f"{stats.get('mean', float('nan')):.4f}",
                        "margin_median": f"{stats.get('median', float('nan')):.4f}",
                        "margin_p10": f"{stats.get('p10', float('nan')):.4f}",
                    })
                    print(f"client {cid} task {old_task}: "
                          f"prompt={p_name:7s} key/anchor={k_name:7s} acc={acc:.2f}")
    finally:
        restore_rng_state(rng_snapshot)

    out_csv = args.out_csv or str(
        Path(args.run_dir) / "metrics" / "compatibility_2x2.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out_csv} ({len(rows)} rows)")

    # console 2x2 summary (client-averaged)
    import numpy as np
    combos = [("old", "old"), ("old", "current"),
              ("current", "old"), ("current", "current")]
    print("\n===== client-mean 2x2 (accuracy) =====")
    for (p_name, k_name) in combos:
        vals = [float(r["accuracy"]) for r in rows
                if r["prompt"] == p_name and r["key_anchor"] == k_name]
        if vals:
            print(f"prompt={p_name:7s} key/anchor={k_name:7s}: "
                  f"{np.mean(vals):.2f}")


if __name__ == "__main__":
    main()
