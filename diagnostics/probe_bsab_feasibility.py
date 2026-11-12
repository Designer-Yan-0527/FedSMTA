"""diagnostics/probe_bsab_feasibility.py -- B_pre: archived aging stress probe
(BSAB pre-registration, Sec.34; offline, training-loop-free diagnostic).

STATUS: PRIOR-LEVEL EVIDENCE ONLY. This probe measures cross-task-end
state transportability on the archived M2' (cifar_a5v2_M2_feat_full)
feature snapshots. It does NOT gate birth-BSAB: aging RED does not
imply birth RED, aging GREEN does not imply birth GREEN (GPT ruling
2026-11-04). The birth question (single-broadcast mismatch, rho_birth
anchored to M2'' deltas) requires the BSAB_PAIR_CAPTURE rerun (B0).

Frozen design (user + GPT review 2026-11-04):
  * pairs: (k, j), j > k, per client c; rows = task-k rows of client c
    (matched by (client_id, sample_idx_relative) across snapshots)
  * fit on train-split rows, eval on test-split rows (fit ∩ eval = ∅,
    original split semantics; snapshots carry split codes 0/1)
  * z_ref stability sanity gate: rel diff < EPS across snapshots for
    the same rows; violation => ZREF_INVARIANCE_VIOLATION, structural
    conclusions halted (NO auto-fallback to 1536-d bridging)
  * bridge operates on the z_op stream only (z_ref is state-invariant
    in expectation); head input = [z_ref ; B(z_op)]
  * full-affine bridge via identity-anchored GCV ridge (pre-registered
    penalty lambda*||B-I||^2): fit the DRIFT zm-zp, so heavy shrinkage
    collapses to the safe no-op B=I (a plain-coefficient ridge pulls
    A->0, a constant map that destroys the features -- artifact caught
    and fixed 2026-11-04, first run); rank variants
    B_r(z)=z+U_r S_r V_r^T z + b from SVD of (A - I)
  * labels are used ONLY for offline accuracy; bridge fitting never
    touches labels
  * rho_age = (Acc(H(B(z_j))) - Acc(H(z_j))) / (Acc(H(z_k)) - Acc(H(z_j)))
    denominator = "birth-state reference gap" (NOT an M2'' equivalent);
    validity gate: reference gap >= 0.5pp per client for the main
    statistic; pairs without any valid client are recorded only
  * per-pair reporting: client mean/median/worst rho_age (valid
    clients), row-weighted pooled accuracies, 2x2 quadrant counts
    (feat: R_feat<=0.5 ; head: rho_age>=0.5)
  * B1 structure evidence: spectra sigma(dZ) (fit rows) and sigma(A-I),
    cumulative energy E_r, rank sweep r in {1,4,8,16,32,64}
  * reproduction check: offline accuracy vs official accuracy_matrix
    entries (macro over clients); large deltas => PROTOCOL_MISMATCH
    flag (rho_age ratios remain internally valid)

READ-ONLY: no checkpoint/run writes; output only to
diagnostics_output/bsab_pre/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch


def svd_robust(M, compute_uv=True):
    """torch CPU float64 SVD. This conda env's numpy/scipy LAPACK stack
    is corrupted (verified 2026-11-04, two independent probes):
      * numpy gesdd: wrong spectra (sigma off by ~1e2) and NaN vectors;
      * scipy gesvd: sigma agrees with torch to 1e-13 but the singular
        VECTORS are corrupted (reconstruction err ~6.5, |U^T U - I| ~1);
      * numpy float64 dgemm fails even x @ I == x for some shapes.
    torch CPU (independent BLAS) passes eye/small-product bit tests and
    SVD reconstruction to ~1e-14 -- the only trusted float64 path here.
    Bridge parameters are cast to float32 on return: the head/feature
    pipeline is float32 end-to-end and float32 gemm is bitwise-correct
    (official accuracy reproduction worst delta 0.16pp)."""
    Mt = torch.from_numpy(
        np.ascontiguousarray(np.asarray(M, dtype=np.float64)))
    U, S, Vh = torch.linalg.svd(Mt, full_matrices=False)
    if compute_uv:
        return U.numpy(), S.numpy(), Vh.numpy()
    return S.numpy()

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUN_DIR = REPO_ROOT / "output/cifar100/fedRelink/cifar_a5v2_M2_feat_full/seed_42"
SNAP_DIR = REPO_ROOT / "diagnostics_output"
OUT_DIR = SNAP_DIR / "bsab_pre"

N_TASKS = 5
N_CLIENTS = 5
EPS_ZREF = 1e-4          # relative z_ref stability tolerance
MIN_REF_GAP = 0.5         # pp; denominator validity gate
FEAT_GOOD = 0.5           # R_feat <= FEAT_GOOD  -> "feature recovery good"
HEAD_GOOD = 0.5           # rho_age >= HEAD_GOOD  -> "head recovery good"
RANKS = (1, 4, 8, 16, 32, 64)
E_R_REPORT = (1, 4, 8, 16, 32, 64, 128, 256)


def load_snapshot(k):
    z = np.load(SNAP_DIR / f"features_cifar100_task_0{k}_end_promptcurrent.npz",
                allow_pickle=True)
    return {key: z[key] for key in
            ("z_ref", "z_op", "labels", "client_ids", "task_ids",
             "splits", "sample_idx_relative")}


def load_heads(k):
    ck = torch.load(RUN_DIR / f"checkpoints/task_0{k}_end.pth",
                    map_location="cpu")
    heads = {}
    masks = {}
    for c in range(N_CLIENTS):
        h = ck["clients"][c]["heads"][k]
        heads[c] = (h["head.0.weight"].numpy(), h["head.0.bias"].numpy(),
                   h["head.3.weight"].numpy(), h["head.3.bias"].numpy())
        masks[c] = np.asarray(ck["clients"][c]["class_mask"][k])
    return heads, masks


def head_logits(head, x):
    """Chead forward, eval mode (Dropout inactive): L3(ReLU(L0(x)))."""
    w0, b0, w3, b3 = head
    return np.maximum(x @ w0.T + b0, 0.0) @ w3.T + b3


def masked_acc(head, x, labels, allowed):
    logits = head_logits(head, x)[:, allowed]
    return 100.0 * float((allowed[logits.argmax(1)] == labels).mean())


def rows_of(snap, client, split, task):
    sel = ((snap["client_ids"] == client) & (snap["splits"] == split)
           & (snap["task_ids"] == task))
    order = np.argsort(snap["sample_idx_relative"][sel], kind="stable")
    idx = np.where(sel)[0][order]
    return idx, snap["sample_idx_relative"][idx]


def paired_rows(snapK, snapJ, client, split, tasks):
    """Concatenated aligned row indices of `client` over `tasks`,
    present in BOTH snapshots (same sample_idx_relative order)."""
    ik_all, ij_all = [], []
    for t in tasks:
        ik, rk = rows_of(snapK, client, split, t)
        ij, rj = rows_of(snapJ, client, split, t)
        if not np.array_equal(rk, rj):
            raise RuntimeError(f"row-set mismatch c={client} t={t} "
                               f"split={split}")
        ik_all.append(ik)
        ij_all.append(ij)
    return np.concatenate(ik_all), np.concatenate(ij_all)


def fit_bridge(zp, zm, fixed_lam=None):
    """Identity-anchored GCV ridge: [zp | 1] -> DRIFT (zm - zp), so the
    penalty acts on ||B - I||^2 as pre-registered. Heavy shrinkage then
    collapses to the SAFE no-op B = I. (A plain-coefficient ridge on the
    absolute target zm pulls A -> 0, a constant map that destroys the
    features -- artifact caught in the first run 2026-11-04: sigma(A-I)
    flat ~1.0, E_r == r/768, acc collapse to 3-19%.) Lambda selected by
    GCV on FIT rows only (label-free, eval rows untouched); the pure
    identity (dA = 0, DoF 0) is an EXPLICIT candidate so GCV can never
    pick a map that is reconstruction-better but worse than no-op.
    fixed_lam: skip GCV (sanity controls: 0 = plain LS, huge = no-op).
    ALL internal float64 math runs on torch CPU (this env's numpy/
    scipy float64 BLAS/LAPACK is corrupted, see svd_robust docstring);
    A/b are returned as float32 so downstream gemm stays on the
    bitwise-verified float32 path."""
    X = torch.from_numpy(np.hstack(
        [zp, np.ones((len(zp), 1), dtype=np.float32)]).astype(np.float64))
    D = torch.from_numpy((zm - zp).astype(np.float64))
    U, s, Vt = torch.linalg.svd(X, full_matrices=False)
    Uty = U.T @ D
    smean_sq = float((s ** 2).mean())
    n = len(zp)
    identity_selected = False
    if fixed_lam is not None:
        lam = float(fixed_lam)
        dof = float((s ** 2 / (s ** 2 + lam)).sum())
        gcv = float("nan")
    else:
        best = (None, np.inf)
        for lam in np.logspace(-8, 3, 23) * smean_sq:
            lam_t = torch.tensor(lam, dtype=torch.float64)
            dof = float((s ** 2 / (s ** 2 + lam_t)).sum())
            resid = D - X @ ((Vt.T * (s / (s ** 2 + lam_t))) @ Uty)
            gcv = float((resid ** 2).sum() / n) / (1 - dof / n) ** 2
            if gcv < best[1]:
                best = (lam, gcv)
        # explicit identity candidate (dA = 0): resid = D, DoF = 0
        gcv_identity = float((D ** 2).sum() / n)
        if gcv_identity <= best[1]:
            identity_selected = True
            return (np.eye(zp.shape[1], dtype=np.float32),
                    np.zeros(zp.shape[1], dtype=np.float32),
                    {"ridge_lambda": None, "gcv": gcv_identity,
                     "dof": 0.0, "identity_candidate_selected": True})
        lam = best[0]
        lam_t = torch.tensor(lam, dtype=torch.float64)
        dof = float((s ** 2 / (s ** 2 + lam_t)).sum())
        gcv = best[1]
    denom = s ** 2 + lam
    coef = torch.where(denom > 0, s / denom, torch.zeros_like(s))
    W = (Vt.T * coef) @ Uty
    A = torch.eye(zp.shape[1], dtype=torch.float64) + W[:768].T
    return (A.numpy().astype(np.float32),
            W[768].numpy().astype(np.float32),
            {"ridge_lambda": float(lam), "gcv": gcv, "dof": dof,
             "identity_candidate_selected": identity_selected})


def apply_bridge(A, b, x):
    return x @ A.T + b


def rank_variant(A, b, x, r):
    """B_r(z) = z + U_r S_r V_r^T z + b from the SVD of (A - I).
    SVD + truncation math on torch float64; applied on the float32
    path (broken numpy/scipy float64 BLAS, see svd_robust)."""
    d = A.shape[0]
    Mt = (torch.from_numpy(A.astype(np.float64))
          - torch.eye(d, dtype=torch.float64))
    U, S, Vt = torch.linalg.svd(Mt, full_matrices=False)
    Mr = (U[:, :r] * S[:r]) @ Vt[:r]
    A_r = (Mr + torch.eye(d, dtype=torch.float64)).numpy() \
        .astype(np.float32)
    return x @ A_r.T + b


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snaps = {k: load_snapshot(k) for k in range(N_TASKS)}
    heads_all = {k: load_heads(k) for k in range(N_TASKS)}
    report = {"probe": "B_pre_aging_stress", "status": "PRIOR_LEVEL_ONLY",
              "design_frozen": "user+GPT 2026-11-04", "pairs": [],
              "zref_violation": [], "notes": []}

    # ---- z_ref stability gate over all snapshot pairs (same rows) ----
    zref_stable = True
    for k in range(N_TASKS):
        for j in range(k + 1, N_TASKS):
            for c in range(N_CLIENTS):
                for split, tasks in ((0, range(k + 1)), (1, (k,))):
                    try:
                        ik, ij = paired_rows(snaps[k], snaps[j], c,
                                             split, tasks)
                    except RuntimeError as e:
                        report["zref_violation"].append(str(e))
                        zref_stable = False
                        continue
                    d = np.linalg.norm(snaps[j]["z_ref"][ij] -
                                        snaps[k]["z_ref"][ik])
                    n0 = np.linalg.norm(snaps[k]["z_ref"][ik])
                    rel = float(d / max(n0, 1e-12))
                    if rel > EPS_ZREF:
                        report["zref_violation"].append(
                            {"k": k, "j": j, "c": c, "split": split,
                             "rel_diff": rel})
                        zref_stable = False
    if report["zref_violation"]:
        report["status"] = "ZREF_INVARIANCE_VIOLATION__STRUCTURAL_HALTED"

    # ---- per-pair bridge fits ----
    for k in range(N_TASKS - 1):
        heads_k, masks_k = heads_all[k]
        for j in range(k + 1, N_TASKS):
            pair = {"k": k, "j": j, "clients": {}, "n_valid": 0,
                    "rho_age": {"mean": None, "median": None, "worst": None},
                    "quadrants": {"feat_good_head_good": 0,
                                  "feat_good_head_bad": 0,
                                  "feat_bad_head_good": 0,
                                  "feat_bad_head_bad": 0}}
            rho_list, pooled = [], {"n_eval": 0, "stale": 0, "ref": 0,
                                    "full": 0, "repro_stale": None}
            for c in range(N_CLIENTS):
                ik_tr, ij_tr = paired_rows(snaps[k], snaps[j], c, 0,
                                           range(k + 1))
                ik_ev, _ = rows_of(snaps[k], c, 1, k)
                ij_ev, _ = rows_of(snaps[j], c, 1, k)
                labels_ev = snaps[k]["labels"][ik_ev]
                allowed = masks_k[c]

                A, b, fit_info = fit_bridge(snaps[j]["z_op"][ij_tr],
                                            snaps[k]["z_op"][ik_tr])
                x_ev = snaps[j]["z_op"][ij_ev]
                xm_ev = snaps[k]["z_op"][ik_ev]
                zr_ev = snaps[j]["z_ref"][ij_ev]
                zr_ev_k = snaps[k]["z_ref"][ik_ev]

                zp_full = apply_bridge(A, b, x_ev)
                r_feat = float(
                    (np.linalg.norm(zp_full - xm_ev) ** 2) /
                    max(np.linalg.norm(x_ev - xm_ev) ** 2, 1e-12))

                acc_stale = masked_acc(heads_k[c],
                                       np.hstack([x_ev, zr_ev]),
                                       labels_ev, allowed)
                acc_ref = masked_acc(heads_k[c],
                                     np.hstack([xm_ev, zr_ev_k]),
                                     labels_ev, allowed)
                acc_full = masked_acc(heads_k[c],
                                      np.hstack([zp_full, zr_ev]),
                                      labels_ev, allowed)
                rank_acc = {}
                for r in RANKS:
                    zr_r = rank_variant(A, b, x_ev, r)
                    rank_acc[r] = masked_acc(heads_k[c],
                                            np.hstack([zr_r, zr_ev]),
                                            labels_ev, allowed)

                gap = acc_ref - acc_stale
                valid = gap >= MIN_REF_GAP
                rho = ((acc_full - acc_stale) / gap) if valid else None
                if valid:
                    rho_list.append(rho)
                    pair["n_valid"] += 1
                    q = ("feat_good" if r_feat <= FEAT_GOOD else "feat_bad",
                         "head_good" if rho >= HEAD_GOOD else "head_bad")
                    pair["quadrants"][q[0] + "_" + q[1]] += 1

                # spectra (fit rows)
                dZ = snaps[j]["z_op"][ij_tr] - snaps[k]["z_op"][ik_tr]
                sv_dz = svd_robust(dZ - dZ.mean(0), compute_uv=False)
                M = A - np.eye(768)
                sv_m = svd_robust(M, compute_uv=False)
                e_r = {r: float((sv_m[:r] ** 2).sum() / (sv_m ** 2).sum())
                       for r in E_R_REPORT}
                e_r_dz = {r: float((sv_dz[:r] ** 2).sum()
                                   / (sv_dz ** 2).sum())
                          for r in E_R_REPORT}

                pooled["n_eval"] += len(labels_ev)
                pooled["stale"] += acc_stale * len(labels_ev)
                pooled["ref"] += acc_ref * len(labels_ev)
                pooled["full"] += acc_full * len(labels_ev)

                pair["clients"][c] = {
                    "n_fit": int(len(ij_tr)), "n_eval": int(len(ij_ev)),
                    "low_power_fit": bool(len(ij_tr) < 2 * 768),
                    "fit": fit_info, "acc_stale": acc_stale,
                    "acc_ref_birth_state": acc_ref, "acc_full_bridge": acc_full,
                    "rank_acc": rank_acc, "R_feat": r_feat,
                    "ref_gap_pp": gap, "rho_age": rho, "valid": valid,
                    "sigma_dZ_top8": [float(s) for s in sv_dz[:8]],
                    "sigma_AminusI_top8": [float(s) for s in sv_m[:8]],
                    "E_r_sigma_dZ": e_r_dz,
                    "E_r_sigma_AminusI": e_r,
                }
            if rho_list:
                pair["rho_age"] = {"mean": float(np.mean(rho_list)),
                                   "median": float(np.median(rho_list)),
                                   "worst": float(min(rho_list))}
            pair["pooled_micro"] = {
                "acc_stale": pooled["stale"] / pooled["n_eval"],
                "acc_ref_birth_state": pooled["ref"] / pooled["n_eval"],
                "acc_full_bridge": pooled["full"] / pooled["n_eval"]}
            report["pairs"].append(pair)

    # ---- reproduction check vs official matrix (macro over clients) ----
    lines = (RUN_DIR / "metrics/accuracy_matrix.csv").read_text().strip().splitlines()
    official = [[float(x) if x else None for x in ln.split(",")]
                for ln in lines[1:]]
    repro = []
    for p in report["pairs"]:
        k, j = p["k"], p["j"]
        macro_stale = np.mean([p["clients"][c]["acc_stale"]
                               for c in range(N_CLIENTS)])
        macro_ref = np.mean([p["clients"][c]["acc_ref_birth_state"]
                             for c in range(N_CLIENTS)])
        repro.append({"k": k, "j": j,
                      "offline_macro_stale": float(macro_stale),
                      "official_after_task_j_Tk": official[j][k + 1],
                      "stale_delta": float(
                          macro_stale - official[j][k + 1]),
                      "offline_macro_ref_birth": float(macro_ref),
                      "official_after_task_k_Tk": official[k][k + 1],
                      "ref_delta": float(macro_ref - official[k][k + 1])})
    report["reproduction_check"] = repro
    worst = max(abs(r["stale_delta"]) for r in repro)
    if worst > 1.0:
        report["status"] += " |PROTOCOL_MISMATCH_worst_delta=%.2fpp" % worst

    (OUT_DIR / "bsab_pre_report.json").write_text(json.dumps(
        report, indent=1))

    # ---- console summary ----
    print("== B_pre aging stress probe (PRIOR LEVEL ONLY) ==")
    print("z_ref stable:", zref_stable)
    for p in report["pairs"]:
        rh = p["rho_age"]
        print(f"(k={p['k']}->j={p['j']}) valid={p['n_valid']}/5 "
              f"rho mean/med/worst = "
              f"{rh['mean'] if rh['mean'] is None else round(rh['mean'],3)}/"
              f"{rh['median'] if rh['median'] is None else round(rh['median'],3)}/"
              f"{rh['worst'] if rh['worst'] is None else round(rh['worst'],3)} "
              f"quadrants={p['quadrants']}")
    print("reproduction worst |delta| = %.2fpp" % worst)
    print("report ->", OUT_DIR / "bsab_pre_report.json")


if __name__ == "__main__":
    main()
