"""Prompt-pool row usage census (post-PSC diagnostic, pre-registered).

Question (decides the capacity-opening direction): across the FULL M2'
trajectory (t0..t4), does top-1 routing stay collapsed onto a few shared
prompt rows (usage starvation), or do new tasks open fresh rows naturally?

Evidence already in hand (PSC short arm, t1): ~98% of task-1 traffic hits
5-8 rows per client; merges never fired (no redundancy). If the census shows
the same collapse at t2..t4 -> the shared-bottleneck story is confirmed and
capacity opening is grounded. If active rows grow with tasks -> capacity is
NOT the bottleneck and the direction must be rethought.

Mechanics: for each task-boundary checkpoint, for each client, for each
SEEN task t', forward the task's train split through original_model
(pre_logits, the real routing query), take per-sample top-1 against the
checkpoint's broadcast prompt_key (exact _psc_prompt_top1 semantics),
and count per-row traffic. Reported per (client, task):
  n_active_rows   rows with >=1 top-1 hit
  rows_cover50/90 rows needed for 50%/90% of samples (concentration)
  top5_share      traffic share of the 5 heaviest rows
plus cross-task overlap: mean Jaccard of per-task support sets, and the
share of the NEWEST task's traffic that lands on rows already in the
previous tasks' support ("reuse vs opening").

Deterministic: sequential loader, first-N subsample (--max_per_task),
no RNG consumption (read-only probe on checkpoints).

Usage:
  python diagnostics/probe_row_usage.py \
    --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
    --out_csv diagnostics_output/row_usage/cifar_m2p_full.csv \
    --data_path ./local_datasets --device cuda
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.diag_utils import bootstrap_server  # noqa: E402


def task_top1_rows(client, task, prompt_key, device, max_per_task):
    """Per-sample top-1 row ids for one (client, task) train split."""
    idx = client.data_split_indices[task]["train"]
    if max_per_task > 0:
        idx = idx[:max_per_task]
    dataset = Subset(client.train_data[task], idx)
    loader = DataLoader(dataset, batch_size=16, num_workers=0,
                       shuffle=False)
    feats = []
    for input, _target in loader:
        input = input.to(device, non_blocking=True)
        with torch.no_grad():
            cls_features = client.original_model(input)['pre_logits']
        feats.append(cls_features)
    if not feats:
        return None, 0
    x = torch.cat(feats, dim=0)
    top1 = client._psc_prompt_top1(x, prompt_key.detach().to(x.device))
    return top1.cpu(), int(top1.numel())


def row_stats(top1, n_rows):
    counts = torch.zeros(n_rows, dtype=torch.long)
    counts.scatter_add_(0, top1, torch.ones_like(top1))
    n = max(int(top1.numel()), 1)
    order = torch.argsort(counts, descending=True)
    cum = torch.cumsum(counts[order], dim=0).float() / n
    cover50 = int((cum < 0.50).sum()) + 1
    cover90 = int((cum < 0.90).sum()) + 1
    top5 = order[:5].tolist()
    top5_share = float(counts[order[:5]].sum()) / n
    support = torch.nonzero(counts).flatten().tolist()
    return {
        "n_active_rows": int((counts > 0).sum()),
        "rows_cover50": cover50, "rows_cover90": cover90,
        "top5_share": round(top5_share, 4), "top5_rows": top5,
        "support": support,
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def census_checkpoint(run_name, run_dir, ckpt_name, data_path, device,
                     max_per_task):
    args, server = bootstrap_server(run_dir, ckpt_name,
                                    data_path=data_path, device=device)
    server_task = int(server.task_id)
    prompt_key = server.prompt.prompt_key.detach().cpu()
    n_rows = prompt_key.shape[0]
    rows = []
    per_client_support = {}   # client -> {task: support set}
    for c in server.clients:
        per_client_support[c.id] = {}
        for t in range(server_task + 1):
            if t not in c.data_split_indices:
                continue
            top1, n_samples = task_top1_rows(c, t, prompt_key, device,
                                             max_per_task)
            if top1 is None:
                continue
            st = row_stats(top1, n_rows)
            st.update(run=run_name, checkpoint=ckpt_name, server_task=server_task,
                      client=c.id, task=t, n_samples=n_samples)
            rows.append(st)
            per_client_support[c.id][t] = st["support"]
    # cross-task overlap per client, averaged
    jac_new_prev, jac_all_pairs = [], []
    for cid, tasks in per_client_support.items():
        for t in tasks:
            prev_keys = [p for p in tasks if p < t]
            if not prev_keys:
                continue
            prev_union = sorted({r for p in prev_keys for r in tasks[p]})
            jac_new_prev.append(jaccard(tasks[t], prev_union))
            for p in prev_keys:
                jac_all_pairs.append(jaccard(tasks[t], tasks[p]))
    # newest task traffic on previously-seen rows (share of samples)
    del server
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = {
        "checkpoint": ckpt_name, "server_task": server_task,
        "active_rows_mean_all_tasks":
            sum(r["n_active_rows"] for r in rows) / max(len(rows), 1),
        "active_rows_per_task": {
            str(t): round(sum(r["n_active_rows"] for r in rows
                              if r["task"] == t)
                          / max(sum(1 for r in rows if r["task"] == t), 1), 1)
            for t in sorted({r["task"] for r in rows})},
        "top5_share_mean":
            round(sum(r["top5_share"] for r in rows) / max(len(rows), 1), 4),
        "jaccard_new_vs_prev_mean":
            round(sum(jac_new_prev) / max(len(jac_new_prev), 1), 4),
        "jaccard_task_pairs_mean":
            round(sum(jac_all_pairs) / max(len(jac_all_pairs), 1), 4),
    }
    return rows, summary


def main():
    p = argparse.ArgumentParser("prompt row usage census")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoints",
                  default="task_00_end.pth,task_01_end.pth,task_02_end.pth,"
                          "task_03_end.pth,task_04_end.pth",
                  help="comma-separated checkpoint names (missing ones "
                       "are skipped)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--summary_json", default=None)
    p.add_argument("--data_path", default="./local_datasets")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_per_task", default=2048, type=int,
                   help="first-N train samples per (client, task); 0 = all")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_name = "/".join(run_dir.parts[-3:-1])
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows, all_summaries = [], []
    with open(out_path, "w") as f:
        f.write("run,checkpoint,server_task,client,task,n_samples,"
                "n_active_rows,rows_cover50,rows_cover90,top5_share,top5_rows\n")
        for ckpt in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
            if not (run_dir / "checkpoints" / ckpt).is_file():
                print(f"[skip] {ckpt}: not found")
                continue
            print(f"[census] {run_name} {ckpt}")
            rows, summary = census_checkpoint(
                run_name, run_dir, ckpt, args.data_path, args.device,
                args.max_per_task)
            all_rows.extend(rows)
            all_summaries.append(summary)
            for r in rows:
                f.write(f"{r['run']},{r['checkpoint']},{r['server_task']},"
                        f"{r['client']},{r['task']},{r['n_samples']},"
                        f"{r['n_active_rows']},{r['rows_cover50']},"
                        f"{r['rows_cover90']},{r['top5_share']},"
                        f"\"{r['top5_rows']}\"\n")
            print(f"  -> {json.dumps(summary, ensure_ascii=False)}")

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(all_summaries, f, indent=1, ensure_ascii=False)
    print(f"\nDONE: {len(all_rows)} (client,task) rows -> {out_path}")


if __name__ == "__main__":
    main()
