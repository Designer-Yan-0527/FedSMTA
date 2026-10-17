"""Phase 6 A4: lambda calibration probe (tangential gradient-ratio +
direction-cosine + lambda-cancel), run BEFORE any fedta_a4 training.

Why this probe exists (A3 lesson): the direction loss
  L_dir = mean_{i in P} || norm(k_i) - sg(norm(k_i^old)) ||^2
back-propagates through the l2-normalization Jacobian ~ 1/||k||, so its
GRADIENT scale is numerically large even when the loss itself is tiny.
Choosing a4_lambda from loss magnitudes or raw gradient norms is wrong
three times over:

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
  2. ALL clients first complete the task transition IN SERVER ORDER
     (update_data for client 0..N-1, then the A2-style protection set +
     task-start key snapshot per client). No probe training may start
     before every client's random_split has consumed the same global RNG
     stream positions as the real round — otherwise later clients get a
     DIFFERENT train split and therefore a different protected set;
  3. per probed client, the probe replicates the REAL train() shape:
       phase-1  — full prompt optimization (local_epoch passes, Adam over
                  vit.parameters(), vit forward OUTSIDE no_grad, official
                  phase-1 loss), because the real phase-2 (and thus the
                  real g_base) sees z AFTER phase-1 moved the prompts;
       phase-2  — lambda=0 drift: the official phase-2 loss
                  (criterion + 0.2*T*InfoNCE - 0.1*pull_off, vit forward
                  INSIDE no_grad exactly as train()) and Adam over
                  model.parameters(), for max_steps steps;
  4. at the probe steps (default 1/5/20, AFTER the step's update), the
     measurement uses n_measure INDEPENDENT batches (never the batch that
     just produced the displacement — reusing it would artificially
     inflate the negative correlation between g_base_tan and g_dir, since
     Delta_k ~ -eta*g_base and g_dir points back at k_old). Per batch:
       g_base_tan = tangential part of d(base loss)/d(key)[protected rows]
       g_dir      = d(L_dir)/d(key)[protected rows]
     reporting aggregate + per-row median/IQR of
       r = |g_base_tan| / |g_dir|,  c = cos(g_base_tan, g_dir),
       lambda_cancel = -<g_base_tan, g_dir> / |g_dir|^2,
     plus the current drift cos(k_hat, k_hat_old) per row and the radial
     leakage |<g_dir, k_hat>| / |g_dir| (theory: ~0; the Jacobian
     (I - k_hat k_hat^T)/||k|| projects g_dir onto the tangent space).

Even with phase-1 replicated, this is still a magnitude/direction probe
on 20 steps of one round: the suggested lambda is a STARTING POINT, to be
refined from the [A4] epoch logs (dir_loss should stay small but
NON-zero; a flat-zero dir_loss means over-constraining ~ A2 hard freeze).

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
    p.add_argument("--n_measure", type=int, default=3,
                   help="independent measurement batches per probe point.")
    p.add_argument("--no_phase1", action="store_true",
                   help="skip the phase-1 prompt-optimization replica "
                        "(fast preliminary mode; NOT a faithful calibration "
                        "— the real phase-2 sees prompts after phase-1).")
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


def phase1_forward_loss(client, run_args, input, target):
    """Replicate the official phase-1 loss of Client_DF.train() for ONE
    batch: vit logits (NOT Tail_Anchor), loss = criterion - 0.1*pull_off,
    vit forward OUTSIDE torch.no_grad() (phase-1 trains the prompts)."""
    with torch.no_grad():
        output = client.original_model(input)
        cls_features = output['pre_logits']
    output = client.vit(input, task_id=client.task_id,
                        cls_features=cls_features, train=True)
    logits = output['logits']
    pull_off = output['reduce_sim']
    mask = client.current_class
    not_mask = np.setdiff1d(np.arange(run_args.nb_classes), mask)
    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(client.device)
    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
    criterion = torch.nn.CrossEntropyLoss().to(client.device)
    return criterion(logits, target) - 0.1 * pull_off


def phase2_forward_loss(client, run_args, round_id, input, target):
    """Replicate the official phase-2 loss of Client_DF.train() for ONE
    batch (lambda = 0: no A4 term). Mirrors train() line by line —
    including the vit forward INSIDE torch.no_grad() (phase-2 never
    trains the ViT; building its graph would not change the key gradient
    but would not be an exact replica):
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


def measure_batch(client, run_args, round_id, input, target):
    """Two measurement backwards on ONE independent batch; returns the
    per-row tangential statistics dict for the current key state."""
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
    # radial leakage of g_dir: theory says <g_dir, k_hat> ~ 0 (the
    # normalization Jacobian projects onto the tangent space)
    radial_row = (g_dir * k_hat).sum(dim=1).abs() / gd_n.clamp_min(1e-30)
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
        "radial_leak_row": med_iqr(radial_row),
        "drift_cos_row": med_iqr(drift_row),
        "key_norm_row": med_iqr(k.norm(dim=1)),
    }


def combine_measures(per_batch):
    """Median/min/max across the n_measure independent batches."""
    out = {}
    keys = [k for k in per_batch[0] if k != "n_protected"]
    out["n_protected"] = per_batch[0]["n_protected"]
    scalars = ["loss_base", "loss_dir", "ratio_agg", "cos_agg",
               "lam_cancel_agg"]
    for key in keys:
        if key in scalars:
            vals = [b[key] for b in per_batch]
            out[key] = {"median": float(np.median(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals))}
        else:  # per-row med_iqr dicts -> median of medians / of iqrs
            out[key] = {
                "median": float(np.median([b[key]["median"]
                                           for b in per_batch])),
                "iqr": float(np.median([b[key]["iqr"]
                                        for b in per_batch])),
            }
    return out


def summarize(reports):
    """Final lambda recommendation from the LAST probe point of each
    client (most-drifted, closest to the sustained regime), median across
    the independent measurement batches."""
    per_client = []
    for rep in reports:
        last = rep["points"][-1]
        per_client.append((rep["client_id"], last["cos_agg"]["median"],
                           last["lam_cancel_agg"]["median"],
                           last["radial_leak_row"]["median"]))
    n_opposing = sum(1 for _, c, _, _ in per_client if c < 0)
    lams = [lc for _, c, lc, _ in per_client if c < 0]
    rec = {
        "clients_opposing_drift": f"{n_opposing}/{len(per_client)}",
        "radial_leak_median_max": float(max(rl for *_, rl in per_client)),
        "note": "lam_cancel is only meaningful where cos_agg < 0 "
                "(regularizer gradient opposes the base tangential drift); "
                "lambda = lam_cancel FULLY cancels the tangential base "
                "gradient on the protected rows, 0.5*lam_cancel is a "
                "half-strength starting point. This is a 20-step probe "
                "value: cross-check stability across probe steps and "
                "refine from the [A4] epoch logs (dir_loss should stay "
                "small but NON-zero; a flat-zero dir_loss means "
                "over-constraining ~ A2 hard freeze).",
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
    if args.n_measure < 1:
        raise SystemExit("--n_measure must be >= 1")
    ckpt = args.checkpoint or f"checkpoints/task_{args.probe_task - 1:02d}_end.pth"

    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=ckpt,
        data_path=args.data_path, device=args.device)
    probe_round = args.probe_task * run_args.global_epoch
    client_ids = (list(range(server.client_num)) if args.clients == "all"
                  else [int(c) for c in args.clients.split(",")])

    # ---- stage 1: EVERY client completes the task transition first ----
    # Mirrors Server_DF.train_clients(): all update_data() calls (which
    # consume the shared global RNG via random_split) happen BEFORE any
    # local training. Interleaving probe training with later clients'
    # update_data would shift their RNG stream and silently change their
    # train split -> protected set -> gradients.
    for cid in range(server.client_num):
        client = server.clients[cid]
        client.update_data(round=probe_round, args=run_args)
        if client.task_id != args.probe_task:
            raise SystemExit(
                f"client {cid}: expected task_id {args.probe_task} after "
                f"update_data, got {client.task_id} (bad checkpoint task?)")
        # A4 protection set + task-start key snapshot (RNG-neutral,
        # idempotent; a2_arm forced to 'key' regardless of the source run)
        client.a2_arm = 'key'
        client._a2_init_task_protection(run_args)

    reports = []
    for cid in client_ids:
        client = server.clients[cid]
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
        batch_iter = iter(train_loader)

        def next_batch():
            nonlocal batch_iter
            try:
                input, target = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                input, target = next(batch_iter)
            input = Variable(input, requires_grad=False).to(
                client.device, non_blocking=True)
            target = target.long().to(client.device, non_blocking=True)
            return input, target

        # ---- phase-1 replica: full prompt optimization ----
        # (the real phase-2 — and thus the real g_base — sees z AFTER
        # phase-1 moved the prompts; skipping it is only a preliminary
        # magnitude mode)
        if not args.no_phase1:
            optimizer_p1 = torch.optim.Adam(client.vit.parameters(),
                                            lr=client.lr, weight_decay=1e-03)
            for _epoch in range(client.local_epoch):
                for _input, _target in train_loader:
                    _input = Variable(_input, requires_grad=False).to(
                        client.device, non_blocking=True)
                    _target = _target.long().to(client.device,
                                                non_blocking=True)
                    loss = phase1_forward_loss(client, run_args,
                                               _input, _target)
                    optimizer_p1.zero_grad()
                    loss.backward()
                    optimizer_p1.step()

        # ---- phase-2 replica: lambda=0 drift + probe-point measurement ----
        optimizer_p2 = torch.optim.Adam(client.model.parameters(),
                                        lr=client.lr, weight_decay=1e-03)
        rep = {"client_id": cid, "phase1_replicated": not args.no_phase1,
               "points": []}
        step = 0
        done = False
        while not done:
            for _ in range(len(train_loader)):
                step += 1
                input, target = next_batch()
                # lambda = 0 drift step (official phase-2 loss only)
                loss = phase2_forward_loss(client, run_args, probe_round,
                                           input, target)
                optimizer_p2.zero_grad()
                loss.backward()
                optimizer_p2.step()
                if step in probe_steps:
                    # measurement on INDEPENDENT batches (never the batch
                    # that just produced the displacement: Delta_k ~ -eta*
                    # g_base, so re-measuring on it would artificially
                    # inflate the negative g_base_tan / g_dir correlation)
                    per_batch = []
                    for _ in range(args.n_measure):
                        m_input, m_target = next_batch()
                        per_batch.append(
                            measure_batch(client, run_args, probe_round,
                                          m_input, m_target))
                    rep["points"].append(
                        {"step": step, **combine_measures(per_batch)})
                if step >= args.max_steps:
                    done = True
                    break
            if not done:
                batch_iter = iter(train_loader)  # epochs restart, as train()

        pt0 = rep["points"][0]
        print(f"\n===== client {cid}: {pt0['n_protected']} protected rows "
              f"(phase1={'on' if not args.no_phase1 else 'OFF'}) =====")
        for pt in rep["points"]:
            print(f" step {pt['step']:3d}: base={pt['loss_base']['median']:.4f} "
                  f"dir={pt['loss_dir']['median']:.3e} | "
                  f"r_agg={pt['ratio_agg']['median']:.3e} "
                  f"cos_agg={pt['cos_agg']['median']:+.3f} "
                  f"lam_cancel={pt['lam_cancel_agg']['median']:.3e} | "
                  f"radial_leak={pt['radial_leak_row']['median']:.2e} "
                  f"drift_cos={pt['drift_cos_row']['median']:.4f} "
                  f"keynorm={pt['key_norm_row']['median']:.4f}")
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
        "n_measure": args.n_measure,
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
    fields = ["client_id", "step", "n_protected",
              "loss_base_median", "loss_dir_median",
              "ratio_agg_median", "cos_agg_median", "lam_cancel_agg_median",
              "radial_leak_row_median", "drift_cos_row_median",
              "key_norm_row_median"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rep in reports:
            for pt in rep["points"]:
                w.writerow({
                    "client_id": rep["client_id"], "step": pt["step"],
                    "n_protected": pt["n_protected"],
                    "loss_base_median": pt["loss_base"]["median"],
                    "loss_dir_median": pt["loss_dir"]["median"],
                    "ratio_agg_median": pt["ratio_agg"]["median"],
                    "cos_agg_median": pt["cos_agg"]["median"],
                    "lam_cancel_agg_median": pt["lam_cancel_agg"]["median"],
                    "radial_leak_row_median": pt["radial_leak_row"]["median"],
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
