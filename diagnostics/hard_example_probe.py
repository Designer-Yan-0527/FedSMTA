"""Hard-example diagnostic probe (A2-HF Gate 0, zero training).

Tests the focal-loss mechanism hypothesis BEFORE any counterfactual
training: if current-task errors do NOT concentrate in low-margin /
moderate-p_y samples, A2-HF is rejected without spending GPU hours.

For each probed task t, the environment is bootstrapped at the
checkpoints/task_{t:02d}_end.pth state (right after task t was learned,
same state that produced CA_t = R[t][t]) and every client is evaluated on
its own task-t test set with per-sample recording:

  - true/predicted class (task-masked, same masking as evaluate_with_margin)
  - p_y = softmax prob of the true class on the masked logits
  - margin = logit_y - max_{j!=y} logit_j (masked, identical to
    evaluate_with_margin)
  - per-class accuracy (8 private classes per client-task)

Eval seed is fixed per (client, task): torch.manual_seed(9700+100*cid+t),
matching a1_snapshot_eval.py, so the probe is reproducible.

Outputs:
  rows CSV:    one row per test sample
  summary CSV: one row per (task, client) + one pooled row per task
  console:     error structure vs correct structure + focal-premise
               indicators (confidently-wrong fraction, error p_y, dead
               classes, top confused pairs)

Usage (from repo root, on the server):
  python diagnostics/hard_example_probe.py \
      --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
      --tasks 4,0
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from diag_utils import bootstrap_server


def parse_args():
    p = argparse.ArgumentParser("A2-HF hard-example probe")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--tasks", default="4,0",
                   help="comma-separated task ids, each evaluated at its "
                        "task_XX_end.pth state (default: 4,0)")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def probe_client_task(client, task, nb_classes):
    """Replicates evaluate_with_margin's forward chain exactly, but records
    per-sample (true, pred, p_y, margin) instead of aggregate stats."""
    if client.original_model is None:
        raise RuntimeError("client.original_model is None; unexpected state")
    if client.heads[task] is None or task >= len(client.test_loader):
        raise RuntimeError(f"client {client.id}: head/loader for task {task} missing")

    test_data = client.test_loader[task]
    loader = DataLoader(test_data, batch_size=8, shuffle=True)
    client.model.load_head(client.heads[task])
    client.model.to(client.device)

    torch.manual_seed(9700 + 100 * client.id + task)

    rows = []
    mask = client.class_mask[task]
    not_mask = np.setdiff1d(np.arange(nb_classes), mask)
    not_mask_t = torch.tensor(not_mask, dtype=torch.int64).to(client.device)

    for input, target in loader:
        input = input.to(client.device, non_blocking=True)
        target_d = target.to(client.device, non_blocking=True)
        with torch.no_grad():
            output = client.original_model(input)
            output = output['pre_logits'].requires_grad_(False)
            output = client.vit(input, task_id=client.task_id,
                                cls_features=output, train=True)
            pre, _, _ = client.model(output['feat'].to(client.device), target_d)
        logits = pre.index_fill(dim=1, index=not_mask_t, value=float('-inf'))
        probs = torch.softmax(logits, dim=1)
        pred = torch.max(probs, dim=1)[1]
        logit_y = logits.gather(1, target_d.unsqueeze(1)).squeeze(1)
        other = logits.scatter(1, target_d.unsqueeze(1), float('-inf'))
        margin = logit_y - other.max(dim=1)[0]
        p_y = probs.gather(1, target_d.unsqueeze(1)).squeeze(1)
        for i in range(len(target)):
            rows.append({
                "client_id": client.id,
                "task": task,
                "true": int(target[i]),
                "pred": int(pred[i].item()),
                "correct": int(int(pred[i].item()) == int(target[i])),
                "p_y": f"{float(p_y[i]):.4f}",
                "margin": f"{float(margin[i]):.4f}",
            })
    return rows


def summarize(rows):
    """(task, client) -> summary dict with focal-premise indicators."""
    out = {}
    by_tc = {}
    for r in rows:
        by_tc.setdefault((r["task"], r["client_id"]), []).append(r)
    for (task, cid), rs in sorted(by_tc.items()):
        corr = [r for r in rs if r["correct"] == 1]
        err = [r for r in rs if r["correct"] == 0]
        acc = 100.0 * len(corr) / len(rs)
        cm = [float(r["margin"]) for r in corr]
        em = [float(r["margin"]) for r in err]
        ep = [float(r["p_y"]) for r in err]
        cm_med = float(np.median(cm)) if cm else float("nan")
        # per-class accuracy
        by_class = {}
        for r in rs:
            c = r["true"]
            by_class.setdefault(c, [0, 0])
            by_class[c][1] += 1
            by_class[c][0] += r["correct"]
        class_acc = {c: 100.0 * v[0] / v[1] for c, v in by_class.items()}
        dead = sorted([c for c, a in class_acc.items() if a < 20.0])
        # top confused pairs
        conf = {}
        for r in err:
            conf[(r["true"], r["pred"])] = conf.get((r["true"], r["pred"]), 0) + 1
        top_conf = sorted(conf.items(), key=lambda kv: -kv[1])[:3]
        out[(task, cid)] = {
            "n": len(rs), "acc": acc,
            "n_err": len(err),
            "err_margin_med": float(np.median(em)) if em else float("nan"),
            "err_margin_p75": float(np.quantile(em, 0.75)) if em else float("nan"),
            "err_p_y_med": float(np.median(ep)) if ep else float("nan"),
            "corr_margin_med": cm_med,
            "conf_wrong_frac": (float(np.mean([m > cm_med for m in em]))
                                if em else float("nan")),
            "class_acc_min": min(class_acc.values()),
            "class_acc_max": max(class_acc.values()),
            "dead_classes": " ".join(map(str, dead)),
            "top_confused": "; ".join(f"{a}->{b}:{n}" for (a, b), n in top_conf),
        }
    return out


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    tasks = [int(t) for t in args.tasks.split(",") if t.strip() != ""]

    all_rows = []
    for t in tasks:
        ckpt = run_dir / "checkpoints" / f"task_{t:02d}_end.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        print(f"[probe] bootstrapping task {t} state from {ckpt}")
        run_args, server = bootstrap_server(
            args.run_dir, checkpoint=str(ckpt),
            data_path=args.data_path, device=args.device)
        nb_classes = run_args.nb_classes
        for client in server.clients:
            if client.task_id is None or client.task_id != t:
                print(f"[probe] client {client.id}: task_id={client.task_id}, "
                      f"expected {t} - skipped (checkpoint not at task end?)")
                continue
            rows = probe_client_task(client, t, nb_classes)
            acc = 100.0 * sum(r["correct"] for r in rows) / max(len(rows), 1)
            print(f"[probe] client {client.id} task {t}: "
                  f"n={len(rows)} acc={acc:.2f}")
            all_rows.extend(rows)

    # ---- write per-sample rows ----
    out_csv = args.out_csv or str(run_dir / "metrics" / "hard_example_probe.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"saved {out_csv} ({len(all_rows)} rows)")

    # ---- summary + focal-premise indicators ----
    summary = summarize(all_rows)
    sum_csv = str(Path(out_csv).with_name(
        Path(out_csv).stem + "_summary.csv"))
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "client_id"] +
                           list(next(iter(summary.values())).keys()))
        w.writeheader()
        for (task, cid), s in summary.items():
            w.writerow({"task": task, "client_id": cid, **s})
    print(f"saved {sum_csv}")

    print("\n===== hard-example probe summary =====")
    for (task, cid), s in summary.items():
        print(f"task {task} client {cid}: acc={s['acc']:.2f} "
              f"n_err={s['n_err']} | err_margin med/p75="
              f"{s['err_margin_med']:.2f}/{s['err_margin_p75']:.2f} vs "
              f"corr_margin med={s['corr_margin_med']:.2f} | "
              f"conf_wrong={100 * s['conf_wrong_frac']:.0f}% | "
              f"err_p_y med={s['err_p_y_med']:.3f} | "
              f"class acc {s['class_acc_min']:.0f}-"
              f"{s['class_acc_max']:.0f}"
              + (f" | dead(<20%): {s['dead_classes']}" if s["dead_classes"] else ""))
        if s["top_confused"]:
            print(f"          top confused: {s['top_confused']}")

    # pooled per task
    print("\n----- pooled per task -----")
    for t in tasks:
        rs = [r for r in all_rows if r["task"] == t]
        if not rs:
            continue
        err = [r for r in rs if r["correct"] == 0]
        corr = [r for r in rs if r["correct"] == 1]
        cm_med = float(np.median([float(r["margin"]) for r in corr]))
        em = [float(r["margin"]) for r in err]
        ep = [float(r["p_y"]) for r in err]
        acc = 100.0 * len(corr) / len(rs)
        print(f"task {t}: pooled acc={acc:.2f} | err margin med="
              f"{np.median(em):.2f} | conf_wrong(margin>corr med)="
              f"{100 * float(np.mean([m > cm_med for m in em])):.0f}% | "
              f"err p_y: <0.05: {100 * float(np.mean([p < 0.05 for p in ep])):.0f}% "
              f"0.05-0.5: {100 * float(np.mean([0.05 <= p < 0.5 for p in ep])):.0f}% "
              f">0.5: {100 * float(np.mean([p >= 0.5 for p in ep])):.0f}%")


if __name__ == "__main__":
    main()
