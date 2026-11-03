"""DHT feasibility death-check probe (RUNCOMMANDS Sec.29, offline only).

Zero-training offline probe for the DHT (Deployment-state Head Transport)
candidate: fits near-identity ridge affine transports between adjacent
deployment-state checkpoints (task_XX_end.pth = post-broadcast states) on
cached z_op features, and measures whether transported historical heads
recover aged-readout accuracy. Nothing here touches training code; the
probe only READS preserved runs.

Four-layer death chain (pre-registered, Sec.29):
  Probe-A  affine feasibility of the drift on probing (current-task)
           rows: held-out residual ratio must be clearly < 1.
  Probe-B  cross-task extrapolation, 3 arms:
             B0 identity (do nothing),
             B1 transport fitted on CURRENT-task train rows (deployable,
               no replay),
             B2 transport fitted on OLD-task train rows (diagnostic
               oracle only, never a method).
           Old-task feature recovery separates "affine dead" from
           "probing distribution dead".
  Probe-C  head-level causal aging: H_j^sync is born post-broadcast at
           boundary j (M2'' recipe: hidden 512, standardize False --
           identical semantics to the arm's refit head), then never sees
           task-j data again. At every later boundary r we compare
             acc_stale  (H_j^sync on z_r, un-transported),
             acc_rec    (H_j^sync on recursively transported z_r),
             acc_dir    (direct transport fitted on task-r train rows),
             acc_refit  (diagnostic pool ceiling: refit head at S_r on
               task-j train rows -- the F4/consolidation upper bound,
               replay access level, NEVER a method).
  Probe-D  recursive-vs-direct: composition error is visible as
           acc_rec < acc_dir on the same rows.

Gates (frozen defaults; do not weaken without external review):
  feasibility: median held-out B1 recovery ratio <= 0.8
  signature  : mean old-task Delta(acc_rec - acc_stale) over clients at
               the final boundary >= green (1.0pp) / <= red (0.5pp)
  transport efficiency (informative): Delta_rec / Delta_refit >= 0.5

F4 overlap audit context (Sec.29.0): the consolidation arm (M5) died for
(a) replay access-level cost and (b) an argued 1.4-1.7pp pool ceiling.
DHT is no-replay so death (a) does not apply, but (b) caps the pool this
probe measures -- acc_refit is exactly that ceiling, measured directly.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.ncm_probe import ensure_features  # noqa: E402
from diagnostics.probe_oracle_headroom import (  # noqa: E402
    fit_head, head_predict, _acc, stable_seed)

# M2'' refit head recipe (Client_DF A5_REFIT_*; kept in sync with the arm)
HIDDEN = 512
EPOCHS = 100
LR = 1e-3
WD = 1e-4
DROPOUT = 0.1
BATCH = 64


def parse_args():
    p = argparse.ArgumentParser("DHT feasibility death-check probe")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoints",
                   default="task_00_end.pth,task_01_end.pth,"
                           "task_02_end.pth,task_03_end.pth,"
                           "task_04_end.pth")
    p.add_argument("--features_dir", default="diagnostics_output")
    p.add_argument("--feat", default="z_op", choices=["z_op", "z_ref"])
    p.add_argument("--data_path", default=None,
                   help="forwarded to extract_features.py if extraction "
                        "is needed")
    p.add_argument("--device", default="cpu",
                   help="'cpu' (deterministic) or 'cuda' (also used for "
                        "feature extraction)")
    p.add_argument("--prompt_task_mode", default="current",
                   choices=["test", "current"])
    p.add_argument("--lam", default=1.0, type=float,
                   help="ridge strength on the residual map dA; effective "
                        "lambda = lam * trace(Zn^T Zn)/d (scale-invariant)")
    p.add_argument("--seed_base", default=20261103, type=int)
    p.add_argument("--rank", default=16, type=int,
                   help="Probe-E low-rank / dynamic bridge rank")
    p.add_argument("--dyn_epochs", default=200, type=int,
                   help="full-batch Adam steps for the dynamic bridge")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--feas_ratio_gate", default=0.8, type=float)
    p.add_argument("--gate_green", default=1.0, type=float)
    p.add_argument("--gate_red", default=0.5, type=float)
    p.add_argument("--no_pool_ceiling", action="store_true",
                   help="skip the diagnostic refit ceiling (faster)")
    return p.parse_args()


def fit_transport(z_old, z_new, lam):
    """Near-identity ridge affine: z_old ~ (I + dA) z_new + b.
    Closed form on centered rows; scale-invariant lambda. Returns (A, b)
    with A = I + dA. Deterministic (pure linear algebra)."""
    zo = torch.as_tensor(z_old, dtype=torch.float64)
    zn = torch.as_tensor(z_new, dtype=torch.float64)
    n, d = zn.shape
    mo, mn = zo.mean(0), zn.mean(0)
    oc, nc = zo - mo, zn - mn
    lam_eff = lam * (nc.T @ nc).trace() / d
    eye = torch.eye(d, dtype=torch.float64)
    # centered fit: oc ~ nc @ dA^T  ->  dA^T = (nc^T nc + lam I)^-1 nc^T oc
    dat = torch.linalg.solve(nc.T @ nc + lam_eff * eye, nc.T @ oc)
    A = eye + dat.T
    b = mo - A @ mn
    return A.numpy().astype(np.float32), b.numpy().astype(np.float32)


def apply_transport(z, A, b):
    return z @ A.T + b


def spectral_stats(A, z_new, z_old, ranks=(1, 4, 8, 16, 32, 64)):
    """Low-rank feasibility of the residual map dA = A - I (Sec.29.1b,
    STBB low-rank bridge hypothesis). Truncated SVD of dA is its optimal
    rank-r approximation (Frobenius), so the cumulative-energy curve AND
    the held-out recovery of the truncated map directly answer 'is the
    broadcast-induced drift low-dimensional' without any extra fitting."""
    dA = A - np.eye(A.shape[0], dtype=np.float32)
    s = np.linalg.svd(dA, compute_uv=False)
    energy = s ** 2
    cum = {int(r): round(float(energy[:r].sum() / energy.sum()), 4)
           for r in ranks if r <= len(s)}
    return cum


def fit_transport_lowrank(z_old, z_new, lam, rank):
    """Rank-r truncation of the full affine solution (optimal Frobenius
    approximation) -- Probe-E 'Low-rank Linear' arm."""
    A, b = fit_transport(z_old, z_new, lam)
    dA = A - np.eye(A.shape[0], dtype=np.float32)
    u, s, vt = np.linalg.svd(dA, full_matrices=False)
    r = min(rank, len(s))
    A_lr = (np.eye(A.shape[0], dtype=np.float32)
            + (u[:, :r] * s[:r]) @ vt[:r])
    return A_lr, b


def fit_transport_dynamic(z_old, z_new, lam, rank, seed, epochs=200,
                          lr=1e-3):
    """Probe-E 'Low-rank Dynamic' arm: B(z) = z + U (g(z) odot V^T z),
    g(z) = sigmoid(M V^T z + a).  Sample-adaptive, NOT foldable into W1
    (the DS-TBB candidate). Warm-started from the low-rank SVD of the
    full affine solution (U0 V0^T = trunc(dA), M0=0, a0=0, b0 ridge);
    full-batch Adam on the probing pairs; deterministic under seed.
    Returns a numpy->numpy callable."""
    A, b = fit_transport(z_old, z_new, lam)
    dA = A - np.eye(A.shape[0], dtype=np.float32)
    u, s, vt = np.linalg.svd(dA, full_matrices=False)
    r = min(rank, len(s))
    U0 = torch.tensor((u[:, :r] * s[:r]), dtype=torch.float32)
    V0 = torch.tensor(vt[:r].T, dtype=torch.float32)
    M0 = torch.zeros(r, r, dtype=torch.float32)
    a0 = torch.zeros(r, dtype=torch.float32)
    b0 = torch.tensor(b, dtype=torch.float32)
    zn = torch.tensor(np.asarray(z_new, dtype=np.float32))
    zo = torch.tensor(np.asarray(z_old, dtype=np.float32))
    torch.manual_seed(seed)
    params = [U0.requires_grad_(True), V0.requires_grad_(True),
              M0.requires_grad_(True), a0.requires_grad_(True),
              b0.requires_grad_(True)]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        q = zn @ V0
        g = torch.sigmoid(q @ M0.T + a0)
        pred = zn + (g * q) @ U0.T + b0
        loss = torch.nn.functional.mse_loss(pred, zo)
        loss.backward()
        opt.step()

    @torch.no_grad()
    def apply(z):
        zt = torch.tensor(np.asarray(z, dtype=np.float32))
        q = zt @ V0
        g = torch.sigmoid(q @ M0.T + a0)
        return (zt + (g * q) @ U0.T + b0).numpy()

    return apply


def compose(ts):
    """Compose a chain of (A, b): z -> A1(A2(...(Ak z + bk)...) + b1)."""
    A, b = ts[0]
    for A2, b2 in ts[1:]:
        A = A @ A2
        b = A2 @ b + b2
    return A, b


def resid_ratio(z_old, z_new, A, b):
    """||old - T(new)||^2 / ||old - new||^2: 1 = no better than identity."""
    drift = float(np.sum((z_old - z_new) ** 2))
    if drift <= 0:
        raise RuntimeError("zero drift between deployment states -- "
                           "checkpoint pair is not two states")
    resid = float(np.sum((z_old - apply_transport(z_new, A, b)) ** 2))
    return resid / drift


def load_boundary(run_dir, ckpt, feat_dir, data_path, device, mode, feat):
    npz_path, meta_path = ensure_features(
        run_dir, ckpt, feat_dir, data_path,
        device if device != "cpu" else None, mode)
    data = np.load(npz_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    masks = {int(cid): [list(map(int, ts)) for ts in tmask]
             for cid, tmask in meta["client_class_masks"].items()}
    return data, masks


def canon(data, feat):
    """Canonicalize a boundary cache by the join key (client, task,
    split, sample_idx_relative). Boundary caches cover tasks 0..t ONLY
    (rows are appended as tasks load), so the same logical sample lives
    at different absolute positions in different states; after this
    sort every (client, task, split) block is identical across all
    states that contain it."""
    order = np.lexsort((data["sample_idx_relative"], data["splits"],
                        data["task_ids"], data["client_ids"]))
    return {k: data[k][order] for k in
            ("labels", "client_ids", "task_ids", "splits",
             "sample_idx_relative", feat)}


def _keyblock(s, j):
    """Join-key fingerprint of task-j rows (client, split, rel_idx,
    labels) -- must be identical across every state containing j."""
    m = s["task_ids"] == j
    return np.column_stack([s["client_ids"][m], s["splits"][m],
                            s["sample_idx_relative"][m], s["labels"][m]])


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    feat_dir = Path(args.features_dir)
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    bounds = []
    for c in ckpts:
        m = re.fullmatch(r"task_(\d+)_end\.pth", Path(c).name)
        if not m:
            raise SystemExit(f"--checkpoints only accepts task_XX_end.pth, "
                             f"got {c!r}")
        bounds.append(int(m.group(1)))
    order = np.argsort(bounds)
    ckpts = [ckpts[i] for i in order]
    bounds = [bounds[i] for i in order]

    # load every deployment state once; canonicalize by join key; fail
    # closed on any mismatch for tasks shared between two states
    states, masks = {}, None
    for t, ck in zip(bounds, ckpts):
        data, msk = load_boundary(run_dir, ck, feat_dir, args.data_path,
                                  args.device, args.prompt_task_mode,
                                  args.feat)
        states[t] = data
        masks = msk
    S = {t: canon(states[t], args.feat) for t in bounds}
    for j in bounds:
        holders = [t for t in bounds if t >= j]
        if len(holders) < 2:
            continue
        k0 = _keyblock(S[holders[0]], j)
        for t in holders[1:]:
            if not np.array_equal(k0, _keyblock(S[t], j)):
                raise RuntimeError(
                    f"task-{j} rows differ across checkpoints "
                    f"({holders[0]} vs {t}) on the join key (client, "
                    f"split, rel_idx, labels) -- do not trust this "
                    f"probe run")

    def M(t, c, task, split):
        s = S[t]
        return ((s["client_ids"] == c) & (s["task_ids"] == task)
                & (s["splits"] == split))

    def MLE(t, c, tmax, split):
        s = S[t]
        return ((s["client_ids"] == c) & (s["task_ids"] <= tmax)
                & (s["splits"] == split))

    clients = sorted(masks)
    hp = dict(hidden=HIDDEN, epochs=EPOCHS, lr=LR, wd=WD, dropout=DROPOUT,
              batch_size=BATCH, device=args.device)
    rows = []
    feas_rows = []

    for cid in clients:
        # ---- transports per adjacent pair. Boundary caches cover tasks
        # 0..t ONLY (Sec.29.1c): the strict no-replay probing rows
        # (task-nxt samples seen by the state-t model) do NOT exist
        # offline, so the arms are re-scoped to what both states share:
        # B1' = fit on task-t train rows (the most recent current task
        # -- deployable proxy), B2 = all tasks <= t (diagnostic oracle).
        trans_cur, trans_old = {}, {}
        for t, nxt in zip(bounds[:-1], bounds[1:]):
            zo, zn = S[t][args.feat], S[nxt][args.feat]
            ct, cn = M(t, cid, t, 0), M(nxt, cid, t, 0)
            if not ct.any():
                raise RuntimeError(f"client {cid}: no task-{t} train rows "
                                   f"at state {t} -- wrong checkpoint ladder")
            A1, b1 = fit_transport(zo[ct], zn[cn], args.lam)
            trans_cur[(t, nxt)] = (A1, b1)
            ot, on = MLE(t, cid, t, 0), MLE(nxt, cid, t, 0)
            A2, b2 = fit_transport(zo[ot], zn[on], args.lam)
            trans_old[(t, nxt)] = (A2, b2)

            # Probe-A: probing-distribution feasibility (held-out rows)
            ctt, ctn = M(t, cid, t, 1), M(nxt, cid, t, 1)
            r_tr = resid_ratio(zo[ct], zn[cn], A1, b1)
            r_te = (resid_ratio(zo[ctt], zn[ctn], A1, b1)
                    if ctt.any() else float("nan"))
            dA = float(np.linalg.norm(A1 - np.eye(A1.shape[0]), "fro"))
            feas_rows.append({
                "client_id": cid, "pair": f"{t}->{nxt}",
                "n_probe": int(ct.sum()),
                "resid_ratio_train": round(r_tr, 4),
                "resid_ratio_heldout": round(r_te, 4) if ctt.any() else None,
                "dA_fro": round(dA, 4),
                "dA_spectrum_cum": spectral_stats(A1),
            })

            # Probe-B: old-task feature recovery, 3 arms (test rows)
            for j in [b for b in bounds if b <= t]:
                ojt, ojn = M(t, cid, j, 1), M(nxt, cid, j, 1)
                if not ojt.any():
                    continue
                r1 = resid_ratio(zo[ojt], zn[ojn], A1, b1)
                r2 = resid_ratio(zo[ojt], zn[ojn], A2, b2)
                rows.append({
                    "client_id": cid, "born_task": j, "boundary": nxt,
                    "kind": "feature_recovery",
                    "age": nxt - j,
                    "drift": round(float(np.mean(
                        np.linalg.norm(zo[ojt] - zn[ojn], axis=1))), 4),
                    "resid_B1_cur": round(r1, 4),
                    "resid_B2_old": round(r2, 4),
                })

        # ---- Probe-C/D: causal head aging ----
        for j in [b for b in bounds if b < bounds[-1]]:
            trj = M(j, cid, j, 0)
            if not trj.any():
                continue
            allowed = masks[cid][j]
            H = fit_head(S[j][args.feat][trj], S[j]["labels"][trj],
                         allowed,
                         seed=stable_seed(args.seed_base, "sync", cid, j),
                         standardize=False, **hp)
            for r in [b for b in bounds if b > j]:
                tej_r = M(r, cid, j, 1)
                if not tej_r.any():
                    continue
                z_te = S[r][args.feat][tej_r]
                y_te = S[r]["labels"][tej_r]
                acc_stale = _acc(head_predict(H, z_te, allowed), y_te)
                chain = [trans_cur[(t, t2)] for t, t2 in
                         zip(bounds[:-1], bounds[1:]) if j <= t < r]
                A_rec, b_rec = compose(chain)
                acc_rec = _acc(head_predict(
                    H, apply_transport(z_te, A_rec, b_rec), allowed), y_te)
                # direct (composition oracle): the deployable probing
                # rows (task-r under state j) do not exist in boundary
                # caches, so direct fits on the rows both states share
                # (task-j train) -- Sec.29.1c re-scope; Probe-E ladder
                # uses the same shared-row fit distribution
                dj, dr = M(j, cid, j, 0), M(r, cid, j, 0)
                zj_d, zr_d = S[j][args.feat][dj], S[r][args.feat][dr]
                A_d, b_d = fit_transport(zj_d, zr_d, args.lam)
                acc_dir = _acc(head_predict(
                    H, apply_transport(z_te, A_d, b_d), allowed), y_te)
                A_lr, b_lr = fit_transport_lowrank(zj_d, zr_d,
                                                   args.lam, args.rank)
                acc_lr = _acc(head_predict(
                    H, apply_transport(z_te, A_lr, b_lr), allowed), y_te)
                dyn = fit_transport_dynamic(
                    zj_d, zr_d, args.lam, args.rank,
                    seed=stable_seed(args.seed_base, "dyn", cid, j, r),
                    epochs=args.dyn_epochs)
                acc_dyn = _acc(head_predict(H, dyn(z_te), allowed), y_te)
                acc_refit = float("nan")
                if not args.no_pool_ceiling:
                    trj_r = M(r, cid, j, 0)
                    H2 = fit_head(S[r][args.feat][trj_r],
                                  S[r]["labels"][trj_r], allowed,
                                  seed=stable_seed(args.seed_base, "refit",
                                                   cid, j, r),
                                  standardize=False, **hp)
                    acc_refit = _acc(head_predict(H2, z_te, allowed), y_te)
                tej_j = M(j, cid, j, 1)
                rows.append({
                    "client_id": cid, "born_task": j, "boundary": r,
                    "kind": "head_aging",
                    "age": r - j,
                    "drift": round(float(np.mean(
                        np.linalg.norm(S[r][args.feat][tej_r]
                                       - S[j][args.feat][tej_j], axis=1))), 4),
                    "acc_stale": round(acc_stale, 2),
                    "acc_rec": round(acc_rec, 2),
                    "acc_dir": round(acc_dir, 2),
                    "acc_lr": round(acc_lr, 2),
                    "acc_dyn": round(acc_dyn, 2),
                    "acc_refit": round(acc_refit, 2),
                    "d_rec": round(acc_rec - acc_stale, 2),
                    "d_dir": round(acc_dir - acc_stale, 2),
                    "d_lr": round(acc_lr - acc_stale, 2),
                    "d_dyn": round(acc_dyn - acc_stale, 2),
                    "d_refit": round(acc_refit - acc_stale, 2),
                })

    # ---- aggregation + gates ----
    last = bounds[-1]
    aging = [r for r in rows if r["kind"] == "head_aging"
             and r["boundary"] == last and r["born_task"] < last]
    feat_rows = [r for r in rows if r["kind"] == "feature_recovery"]

    def colmean(rs, key):
        per = {}
        for r in rs:
            per.setdefault(r["born_task"], []).append(r[key])
        return {k: round(float(np.mean(v)), 2) for k, v in sorted(per.items())}

    sig_rec = float(np.mean([r["d_rec"] for r in aging])) if aging else float("nan")
    sig_refit = float(np.mean([r["d_refit"] for r in aging])) if aging else float("nan")
    sig_dir = float(np.mean([r["d_dir"] for r in aging])) if aging else float("nan")
    sig_lr = float(np.mean([r["d_lr"] for r in aging])) if aging else float("nan")
    sig_dyn = float(np.mean([r["d_dyn"] for r in aging])) if aging else float("nan")
    held = [f["resid_ratio_heldout"] for f in feas_rows
            if f["resid_ratio_heldout"] is not None]
    feas_med = float(np.median(held)) if held else float("nan")
    feas_ok = feas_med <= args.feas_ratio_gate

    if not np.isnan(sig_rec):
        verdict = ("GREEN" if (sig_rec >= args.gate_green and feas_ok)
                   else "RED" if sig_rec <= args.gate_red else "YELLOW")
    else:
        verdict = "N/A"

    ages = np.array([r["age"] for r in aging], dtype=float)
    gains = np.array([r["d_rec"] for r in aging], dtype=float)
    drifts = np.array([r["drift"] for r in aging], dtype=float)
    corr_age = (float(np.corrcoef(ages, gains)[0, 1])
                if len(ages) > 2 and np.std(ages) > 0 else float("nan"))
    corr_drift = (float(np.corrcoef(drifts, gains)[0, 1])
                  if len(drifts) > 2 and np.std(drifts) > 0 else float("nan"))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(
            {k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    with open(out_csv.with_suffix(".summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "hyper": vars(args),
            "gates": {"feasibility_ratio": feas_med,
                      "feasibility_gate": args.feas_ratio_gate,
                      "feasibility_pass": bool(feas_ok),
                      "signature_d_rec": round(sig_rec, 2),
                      "pool_ceiling_d_refit": round(sig_refit, 2),
                      "probe_E": {"d_dir_full_affine": round(sig_dir, 2),
                                  "d_lr_lowrank": round(sig_lr, 2),
                                  "d_dyn_dynamic": round(sig_dyn, 2),
                                  "rank": args.rank,
                                  "note": "direct-fit ladder at final "
                                          "boundary; dynamic > linear by "
                                          "a clear margin motivates the "
                                          "DS-TBB arm, dynamic ~ linear "
                                          "kills it (capacity without "
                                          "mechanism)"},
                      "transport_efficiency": (round(sig_rec / sig_refit, 3)
                                               if sig_refit else None),
                      "green_pp": args.gate_green, "red_pp": args.gate_red,
                      "verdict": verdict,
                      "corr_gain_age": round(corr_age, 3),
                      "corr_gain_drift": round(corr_drift, 3)},
            "per_born_task_rec": colmean(aging, "d_rec"),
            "per_born_task_refit": colmean(aging, "d_refit"),
            "feasibility_rows": feas_rows,
        }, f, indent=2)

    print(f"[dht-probe] rows -> {out_csv}")
    print(f"[dht-probe] feasibility: held-out B1 resid median "
          f"{feas_med:.3f} (gate <= {args.feas_ratio_gate}) -> "
          f"{'PASS' if feas_ok else 'FAIL'}")
    print(f"[dht-probe] signature (boundary {last}, old tasks): "
          f"d_rec {sig_rec:+.2f}pp vs pool ceiling d_refit "
          f"{sig_refit:+.2f}pp -> {verdict}")
    print(f"[dht-probe] corr(gain, age) = {corr_age:.3f}; "
          f"corr(gain, drift) = {corr_drift:.3f}")
    print(f"[dht-probe] Probe-E ladder (boundary {last}): "
          f"full-affine {sig_dir:+.2f} / lowrank(r={args.rank}) "
          f"{sig_lr:+.2f} / dynamic {sig_dyn:+.2f} / refit ceiling "
          f"{sig_refit:+.2f}")


if __name__ == "__main__":
    main()
