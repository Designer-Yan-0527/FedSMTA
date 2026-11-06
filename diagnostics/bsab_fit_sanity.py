"""diagnostics/bsab_fit_sanity.py -- fit-parameterization sanity gates
for the B_pre bridge (BSAB Sec.34; offline, training-loop-free).

Ruling (user, 2026-11-04): B_PRE_RUN_01 = FIT_SANITY_FAIL. The plain-
coefficient ridge collapsed A -> 0 (constant map). Before ANY B_pre
rerun, these invariants must hold on one client / one transition:

  (1) identity control     : B(z) = z via apply_bridge must bitwise
                             reproduce acc_stale
  (2) synthetic identity   : fit_bridge(X, X) must give A ~ I, b ~ 0
                             and leave accuracy untouched
  (3) lambda -> inf control: fixed_lam huge must collapse to B = I,
                             acc -> acc_stale (NOT a constant map)
  (4) lambda = 0 control   : plain LS on a n_fit > 2d client; report
                             held-out R_feat / acc (informational:
                             does unrestricted affine even help?)
  (5) mean-shift-only      : B(z) = z + mean(D_fit); report (separates
                             pure mean shift from linear map)

Gate = (1)+(2)+(3) PASS. (4)+(5) are recorded, no verdict.
Test cell: (k=2 -> j=4), client 0, n_fit = 4410 > 2*768.

READ-ONLY; output -> diagnostics_output/bsab_pre/fit_sanity_gate.json
"""

import json
from pathlib import Path

import numpy as np

from probe_bsab_feasibility import (apply_bridge, fit_bridge, load_heads,
                                     load_snapshot, masked_acc, paired_rows,
                                     rows_of)

K, J, C = 2, 4, 0
OUT = Path(__file__).resolve().parent.parent / "diagnostics_output/bsab_pre"


def pred_labels(head, xin, allowed):
    """head = (w0, b0, w3, b3); head[0] eats the FULL 1536-d input
    [z_op ; z_ref] (same math as probe's head_logits)."""
    w0, b0, w3, b3 = head
    logits = np.maximum(xin @ w0.T + b0, 0.0) @ w3.T + b3
    return allowed[logits[:, allowed].argmax(1)]


def main():
    snapK, snapJ = load_snapshot(K), load_snapshot(J)
    heads, masks = load_heads(K)
    head, allowed = heads[C], masks[C]

    ik_tr, ij_tr = paired_rows(snapK, snapJ, C, 0, range(K + 1))
    ik_ev, _ = rows_of(snapK, C, 1, K)
    ij_ev, _ = rows_of(snapJ, C, 1, K)
    labels = snapK["labels"][ik_ev]

    zp_tr, zm_tr = snapJ["z_op"][ij_tr], snapK["z_op"][ik_tr]
    x_ev, xm_ev = snapJ["z_op"][ij_ev], snapK["z_op"][ik_ev]
    zr_ev = snapJ["z_ref"][ij_ev]
    n_eval = len(labels)

    # z_ref stability on this cell
    d_ref = float(np.linalg.norm(zr_ev - snapK["z_ref"][ik_ev])
                  / np.linalg.norm(snapK["z_ref"][ik_ev]))

    xin_ev = np.hstack([x_ev, zr_ev])
    acc_stale = masked_acc(head, xin_ev, labels, allowed)
    stale_pred = pred_labels(head, xin_ev, allowed)
    rep = {"cell": {"k": K, "j": J, "client": C,
                    "n_fit": int(len(ij_tr)), "n_eval": int(n_eval),
                    "zref_rel_diff": d_ref},
           "acc_stale": acc_stale, "controls": {}}

    def eval_map(A, b):
        zB = apply_bridge(A, b, x_ev)
        xB = np.hstack([zB, zr_ev])
        acc = masked_acc(head, xB, labels, allowed)
        pred = pred_labels(head, xB, allowed)
        r_feat = float(np.linalg.norm(zB - xm_ev) ** 2
                       / max(np.linalg.norm(x_ev - xm_ev) ** 2, 1e-12))
        return acc, pred, r_feat

    # (1) identity control (float32: the whole head pipeline is float32
    # and this env's numpy float64 dgemm is corrupted -- x64 @ I != x64)
    acc1, pred1, _ = eval_map(np.eye(768, dtype=np.float32),
                              np.zeros(768, dtype=np.float32))
    c1 = {"acc": acc1,
          "bitwise_pred_match": bool(np.array_equal(pred1, stale_pred)),
          "pass": bool(np.array_equal(pred1, stale_pred)
                       and acc1 == acc_stale)}

    # (2) synthetic identity recovery: Y = X through the real fitter
    A2, b2, info2 = fit_bridge(zp_tr, zp_tr)
    acc2, pred2, r2 = eval_map(A2, b2)
    rel_dev = float(np.linalg.norm(A2 - np.eye(768))
                   / np.linalg.norm(np.eye(768)))
    rel_b = float(np.linalg.norm(b2) / max(np.linalg.norm(x_ev.mean(0)),
                                          1e-12))
    resid_syn = float(np.linalg.norm(apply_bridge(A2, b2, x_ev) - x_ev)
                      ** 2 / np.linalg.norm(x_ev) ** 2)
    c2 = {"A_minus_I_rel": rel_dev, "b_rel": rel_b,
          "eval_resid_rel": resid_syn, "acc": acc2,
          "fit_info": info2,
          "pass": bool(rel_dev < 1e-6 and rel_b < 1e-6 and resid_syn < 1e-6
                       and np.array_equal(pred2, stale_pred))}

    # (3) lambda -> inf control
    A3, b3, info3 = fit_bridge(zp_tr, zm_tr, fixed_lam=1e12)
    acc3, pred3, r3 = eval_map(A3, b3)
    dev3 = float(np.linalg.norm(A3 - np.eye(768)))
    c3 = {"A_minus_I_frob": dev3, "b_norm": float(np.linalg.norm(b3)),
          "acc": acc3, "R_feat": r3, "fit_info": info3,
          "pass": bool(dev3 < 1e-3 and np.linalg.norm(b3) < 1e-3
                       and np.array_equal(pred3, stale_pred))}

    # (4) lambda = 0 (plain LS) control -- informational
    A4, b4, info4 = fit_bridge(zp_tr, zm_tr, fixed_lam=0.0)
    acc4, pred4, r4 = eval_map(A4, b4)
    c4 = {"acc": acc4, "R_feat": r4, "fit_info": info4,
          "pass": None, "info": "plain LS, no gate (n_fit > 2d satisfied)"}

    # (5) mean-shift-only control -- informational
    b5 = (zm_tr - zp_tr).mean(0)
    acc5, pred5, r5 = eval_map(np.eye(768), b5)
    c5 = {"acc": acc5, "R_feat": r5, "pass": None,
          "info": "B(z)=z+mean(D_fit), no gate"}

    # (6) production GCV fit (identity candidate included) -- context
    A6, b6, info6 = fit_bridge(zp_tr, zm_tr)
    acc6, _, r6 = eval_map(A6, b6)
    c6 = {"acc": acc6, "R_feat": r6, "fit_info": info6, "pass": None,
          "info": "production GCV path after identity-anchored fix"}

    rep["controls"] = {"identity": c1, "synthetic_identity": c2,
                       "lambda_inf": c3, "lambda_zero": c4,
                       "mean_shift_only": c5, "gcv_production": c6}
    rep["gate_pass"] = bool(c1["pass"] and c2["pass"] and c3["pass"])
    rep["status"] = "FIT_SANITY_PASS" if rep["gate_pass"] \
        else "FIT_SANITY_FAIL"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit_sanity_gate.json").write_text(json.dumps(rep, indent=1))

    print(f"cell (k={K}->j={J}, c={C}) n_fit={len(ij_tr)} n_eval={n_eval} "
          f"zref_rel={d_ref:.2e}")
    print(f"acc_stale = {acc_stale:.2f}")
    for name, key in (("identity", "identity"),
                      ("synthetic Y=X", "synthetic_identity"),
                      ("lambda->inf", "lambda_inf"),
                      ("lambda=0 LS", "lambda_zero"),
                      ("mean-shift", "mean_shift_only"),
                      ("GCV production", "gcv_production")):
        ctl = rep["controls"][key]
        acc = ctl.get("acc")
        r = ctl.get("R_feat")
        print(f"  {name:16s} acc={acc if acc is None else round(acc, 3)} "
              f"R_feat={r if r is None else round(r, 4)} "
              f"pass={ctl['pass']}")
    print("GATE:", rep["status"])
    print("report ->", OUT / "fit_sanity_gate.json")


if __name__ == "__main__":
    main()
