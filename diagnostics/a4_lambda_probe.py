"""Phase 6 A4: lambda calibration probe (tangential gradient-ratio +
direction-cosine + lambda-cancel), run BEFORE any fedta_a4 training.

Why this probe exists (A3 lesson): the direction loss
  L_dir = mean_{i in P} || norm(k_i) - sg(norm(k_i^old)) ||^2
back-propagates through the l2-normalization Jacobian ~ 1/||k||, so its
GRADIENT scale is numerically large even when the loss itself is tiny.
Choosing a4_lambda from loss magnitudes or raw gradient norms is wrong
twice over:

  1. at task start K = K_old, so L_dir AND its gradient are exactly 0 —
     calibration must happen AFTER natural drift, never at the initial
     point;
  2. routing depends only on the key DIRECTION, so the relevant base
     quantity is the TANGENTIAL component of the base gradient
     (g_base - (g_base . k_hat) k_hat), not the full norm (a large radial
     component only rescales the key and never changes routing);
  3. a matched gradient RATIO still does not imply opposition — A3 had
     cos(g_base, g_KL) ~= 0. We therefore also report
     cos(g_base_tan, g_dir) and the cancellation scale
     lambda_cancel = -<g_base_tan, g_dir> / |g_dir|^2  (only meaningful
     when cos < 0, i.e. the regularizer actually fights the drift).

Protocol (probe state is DISCARDED, never polluting a real run):
  1. bootstrap the run environment from {run_dir}/args.json and load the
     task_{t-1}_end.pth round-boundary checkpoint (= the exact state the
     real run is in when round t*global_epoch begins);
  2. for every client: update_data(probe_round) (identical data split /
     RNG consumption as the real round), then rebuild the A2-style
     protection set + task-start key snapshot (client.a2_arm='key';
     _a2_init_task_protection is RNG-neutral and idempotent);
  3. per client: lambda=0 drift phase — replicate the phase-2 loss
     (criterion + 0.2*T*InfoNCE - 0.1*pull_off) and Adam over
     model.parameters() exactly as train() does, for max_steps steps;
  4. at the probe steps (default 1/5/20, AFTER the step's update), do two
     extra measurement backwards on a FRESH batch forward:
       g_base_tan = tangential part of d(base loss)/d(key)[protected rows]
       g_dir      = d(L_dir)/d(key)[protected rows]
     and report aggregate + per-row median/IQR of
       r = |g_base_tan| / |g_dir|,  c = cos(g_base_tan, g_dir),
       lambda_cancel = -<g_base_tan, g_dir> / |g_dir|^2,
       plus the current drift  cos(k_hat, k_hat_old) per row.

Known limitation (documented, accepted for a magnitude probe): the drift
phase replicates phase-2 only; the real round runs phase-1 (prompt
optimization) first, which shifts z before phase-2 starts. This makes the
probe a slightly CONSERVATIVE (task-start-like) estimate; lambda chosen
from it is a starting point, to be refined from the [A4] epoch logs.

Usage (from repo root, on the server):
  python diagnostics/a4_lambda_probe.py \
      --run_dir output/cifar100/fedta/<run_name>/seed_42 \
      --probe_task 1 \
      --out_json diagnostics_output/a4_lambda_probe.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from diag_utils import bootstrap_server


def parse_args():
    p = argparse.ArgumentParser("A4 lambda calibration probe")
    p.add_argument("--run_dir", required=True,
                   help="run directory of ANY method (fedta / fedta_a2 / ...); "
                        "the task-boundary checkpoint provides the start state")
    p.add_argument("--probe_task", type=int, default=1,
                   help="task the probe drifts INTO (checkpoint used: "
                        "task_{probe_task-1}_end.pth). Default 1.")
    p.add_argument("--probe_steps", default="1,5,20",
                   help="comma-separated step indices (after the update) at "
                        "which to measure. Default 1,5,20.")
    p.add_argument("--max_steps", type=int, default=20,
                   help="total lambda=0 drift steps (>= max probe step).")
    p.add_argument("--clients", default="all",
                   help="'all' or comma-separated client ids.")
    p.add_argument("--checkpoint", default=None,
                   help="explicit checkpoint (default: "
                        "task_{probe_task-1:02d}_end.pth)")
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def phase2_forward_loss(client, run_args, round_id, input, target):
    """Replicate the official phase-2 loss of Client_DF.train() for ONE
    batch (lambda = 0: no A4 term). Mirrors train() line by line:
    criterion + 0.2*task_per_global_epoch*InfoNCE - 0.1*pull_off2, with
    the not-current-class logits masked to -inf."""
    with torch.no_grad():
        output = client.original_model(input)
        cls_features = output['pre_logits']
    output = client.vit(input, task_id=client.task_id,
                        cls_features=cls_features, train=True)
    pre, output_mixed, pull_off2 = client.model(
        output['feat'].to(client.device), target.to(client.device))
    logits = pre

    if client.global_protos is None:
        loss_InfoNCE = 0
    else:
        count = 0
        loss_InfoNCE = None
        for i, label in enumerate(target):
            if label.item() in client.global_protos.keys():
                count += 1
                feature = output_mixed[i].unsqueeze(0)
                loss_instance = client.calculate_infonce(
                    feature, label.item(),
                    (round_id + 1) % client.task_per_global_epoch == 0)
                if loss_InfoNCE is None:
                    loss_InfoNCE = loss_instance
                else:
                    loss_InfoNCE = loss_InfoNCE + loss_instance
        if count != 0:
            loss_InfoNCE = loss_InfoNCE / count
        else:
            loss_InfoNCE = 0

    mask = client.current_class
    not_mask = np.setdiff1d(np.arange(run_args.nb_classes), mask)
    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(client.device)
    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

    criterion = torch.nn.CrossEntropyLoss().to(client.device)
    return (criterion(logits, target)
            + 0.2 * client.task_per_global_epoch * loss_InfoNCE
            - 0.1 * pull_off2)


def measure_point(client, run_args, round_id, input, target):
    """Two measurement backwards on a fresh forward; returns the per-row
    tangential statistics dict for the current key state."""
    slots = client._a2_protected_slots

    # g_base: full base loss (no A4 term) w.r.t. the protected key rows
    base_loss = phase2_forward_loss(client, run_args, round_id, input, target)
    client.model.zero_grad()
    base_loss.backward()
    g_base = client.model.key.grad[slots].detach().clone()

    # g_dir: the A4 direction loss w.r.t. the same rows
    client.model.zero_grad()
    dir_loss = client._a4_dir_loss()
    dir_loss.backward()
    g_dir = client.model.key.grad[slots].detach().clone()
    client.model.zero_grad()

    # tangential decomposition of the base gradient (per row):
    # routing only sees the key DIRECTION, so the radial component
    # (g . k_hat) k_hat never matters for routing
    k = client.model.key.data[slots]
    k_hat = client.model.l2_normalize(k, dim=1)
    g_base_tan = g_base - (g_base * k_hat).sum(dim=1, keepdim=True) * k_hat

    # per-row statistics
    gb_n = g_base_tan.norm(dim=1)                    # [n]
    gd_n = g_dir.norm(dim=1)                         # [n]
    dot = (g_base_tan * g_dir).sum(dim=1)            # [n]
    ratio_row = gb_n / gd_n.clamp_min(1e-30)
    cos_row = dot / (gb_n * gd_n).clamp_min(1e-30)
    lam_cancel_row = -dot / gd_n.pow(2).clamp_min(1e-30)
    # current drift: cos(k_now, k_old) per row (1 = undrifted)
    k_old_hat = client.model.l2_normalize(
        client._a2_key_snapshot.to(k.dtype), dim=1)
    drift_row = (k_hat * k_old_hat).sum(dim=1)

    def med_iqr(x):
        x = x.cpu().numpy().astype(np.float64)
        q25, q50, q75 = np.percentile(x, [25, 50, 75])
        return {"median": float(q50), "iqr": float(q75 - q25)}

    # aggregates (all protected rows flattened into one vector)
    gb_flat = g_base_tan.reshape(-1)
    gd_flat = g_dir.reshape(-1)
    cos_agg = float((torch.dot(gb_flat, gd_flat)
                     / (gb_flat.norm() * gd_flat.norm()).clamp_min(1e-30)))
    lam_cancel_agg = float(-torch.dot(gb_flat, gd_flat)
                           / gd_flat.norm().pow(2).clamp_min(1e-30))
    ratio_agg = float(gb_flat.norm() / gd_flat.norm().clamp_min(1e-30))

    return {
        "loss_base": float(base_loss.detach()),
        "loss_dir": float(dir_loss.detach()),
        "n_protected": int(slots.numel()),
        "ratio_agg": ratio_agg,
        "cos_agg": cos_agg,
        "lam_cancel_agg": lam_cancel_agg,
        "ratio_row": med_iqr(ratio_row),
        "cos_row": med_iqr(cos_row),
        "lam_cancel_row": med_iqr(lam_cancel_row),
        "drift_cos_row": med_iqr(drift_row),
        "key_norm_row": med_iqr(k.norm(dim=1)),
    }


def summarize(reports):
    """Final lambda recommendation from the LAST probe point of each
    client (most-drifted, closest to the sustained regime)."""
    per_client = []
    for rep in reports:
        last = rep["points"][-1]
        per_client.append((rep["client_id"], last["cos_agg"],
                           last["lam_cancel_agg"]))
    n_opposing = sum(1 for _, c, _ in per_client if c < 0)
    lams = [lc for _, c, lc in per_client if c < 0]
    rec = {
        "clients_opposing_drift": f"{n_opposing}/{len(per_client)}",
        "note": "lam_cancel is only meaningful where cos_agg < 0 "
                "(regularizer gradient opposes the base tangential drift); "
                "lambda = lam_cancel FULLY cancels the tangential base "
                "gradient on the protected rows, 0.5*lam_cancel is a "
                "half-strength starting point. Cross-check stability across "
                "probe steps and refine from the [A4] epoch logs "
                "(dir_loss should stay small but NON-zero; a flat-zero "
                "dir_loss means over-constraining ~ A2 hard freeze).",
    }
    if lams:
        rec["lam_cancel_median"] = float(np.median(lams))
        rec["suggested_lambda_start"] = float(0.5 * np.median(lams))
    return rec


def main():
    args = parse_args()
    probe_steps = sorted({int(s) for s in args.probe_steps.split(",")})
    if max(probe_steps) > args.max_steps:
        raise SystemExit("--max_steps must be >= max probe step")
    ckpt = args.checkpoint or f"checkpoints/task_{args.probe_task - 1:02d}_end.pth"

    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=ckpt,
        data_path=args.data_path, device=args.device)
    probe_round = args.probe_task * run_args.global_epoch
    client_ids = (list(range(server.client_num)) if args.clients == "all"
                  else [int(c) for c in args.clients.split(",")])

    reports = []
    for cid in client_ids:
        client = server.clients[cid]
        # identical to the real round: update_data does the task switch
        # (data split + RNG consumption) exactly as train_clients would
        client.update_data(round=probe_round, args=run_args)
        if client.task_id != args.probe_task:
            raise SystemExit(
                f"client {cid}: expected task_id {args.probe_task} after "
                f"update_data, got {client.task_id} (bad checkpoint task?)")
        # A4 protection set + task-start key snapshot (RNG-neutral,
        # idempotent; a2_arm forced to 'key' regardless of the source run)
        client.a2_arm = 'key'
        client._a2_init_task_protection(run_args)
        if client._a2_protected_slots is None:
            print(f"[probe] client {cid}: empty protection set, skipped")
            continue

        # mirror train(): prompt state + original_model mode
        client.original_model.eval()
        if client.prompts is not None:
            client.vit.load_prompts(client.prompts)
        client.model.to(client.device)
        client.vit.to(client.device)
        train_loader = DataLoader(client.traindata, batch_size=16,
                                  num_workers=run_args.num_workers,
                                  shuffle=True)
        optimizer = torch.optim.Adam(client.model.parameters(),
                                     lr=client.lr, weight_decay=1e-03)

        rep = {"client_id": cid, "points": []}
        step = 0
        done = False
        while not done:
            for input, target in train_loader:
                step += 1
                input = Variable(input, requires_grad=False).to(
                    client.device, non_blocking=True)
                target = target.long().to(client.device, non_blocking=True)
                # lambda = 0 drift step (official phase-2 loss only)
                loss = phase2_forward_loss(client, run_args, probe_round,
                                           input, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step in probe_steps:
                    # measurement backwards on a FRESH batch forward
                    rep["points"].append(
                        {"step": step,
                         **measure_point(client, run_args, probe_round,
                                         input, target)})
                if step >= args.max_steps:
                    done = True
                    break

        print(f"\n===== client {cid}: {rep['points'][0]['n_protected']} "
              f"protected rows =====")
        for pt in rep["points"]:
            print(f" step {pt['step']:3d}: base={pt['loss_base']:.4f} "
                  f"dir={pt['loss_dir']:.3e} | "
                  f"r_agg={pt['ratio_agg']:.3e} "
                  f"cos_agg={pt['cos_agg']:+.3f} "
                  f"lam_cancel={pt['lam_cancel_agg']:.3e} | "
                  f"drift_cos_med={pt['drift_cos_row']['median']:.4f} "
                  f"keynorm_med={pt['key_norm_row']['median']:.4f}")
        reports.append(rep)

    if not reports:
        raise SystemExit("no client produced a protection set; "
                         "probe_task must be >= 1")

    out = {
        "run_dir": str(args.run_dir),
        "checkpoint": ckpt,
        "probe_task": args.probe_task,
        "probe_round": probe_round,
        "probe_steps": probe_steps,
        "clients": reports,
        "recommendation": summarize(reports),
    }
    out_json = args.out_json or str(
        Path("diagnostics_output") /
        f"a4_lambda_probe_task{args.probe_task}.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_json}")

    # flat CSV for cross-run comparison
    out_csv = args.out_csv or str(Path(out_json).with_suffix(".csv"))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "client_id", "step", "n_protected", "loss_base", "loss_dir",
            "ratio_agg", "cos_agg", "lam_cancel_agg",
            "ratio_row_median", "ratio_row_iqr",
            "cos_row_median", "cos_row_iqr",
            "lam_cancel_row_median", "lam_cancel_row_iqr",
            "drift_cos_row_median", "key_norm_row_median"])
        w.writeheader()
        for rep in reports:
            for pt in rep["points"]:
                w.writerow({
                    "client_id": rep["client_id"], "step": pt["step"],
                    "n_protected": pt["n_protected"],
                    "loss_base": pt["loss_base"], "loss_dir": pt["loss_dir"],
                    "ratio_agg": pt["ratio_agg"], "cos_agg": pt["cos_agg"],
                    "lam_cancel_agg": pt["lam_cancel_agg"],
                    "ratio_row_median": pt["ratio_row"]["median"],
                    "ratio_row_iqr": pt["ratio_row"]["iqr"],
                    "cos_row_median": pt["cos_row"]["median"],
                    "cos_row_iqr": pt["cos_row"]["iqr"],
                    "lam_cancel_row_median": pt["lam_cancel_row"]["median"],
                    "lam_cancel_row_iqr": pt["lam_cancel_row"]["iqr"],
                    "drift_cos_row_median": pt["drift_cos_row"]["median"],
                    "key_norm_row_median": pt["key_norm_row"]["median"],
                })
    print(f"saved {out_csv}")

    rec = out["recommendation"]
    print("\n===== lambda recommendation =====")
    for k, v in rec.items():
        print(f" {k}: {v}")


if __name__ == "__main__":
    main()
