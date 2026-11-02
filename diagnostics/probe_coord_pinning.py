"""Coordinate-pinning probe: is z_op drift affine-recoverable?

Motivation (measured ledger, 2026-10-31): after M2' fixes readout
staleness and A2 fixes routing, the one rotting interface left is the
COORDINATE SYSTEM itself -- z_op of old tasks drifts as prompts evolve
(cifar T0: fresh head scored 96.78 on t1-time z, only ~94.4 on t4-time
z, a ~2.3pp drift no readout can fix; INR NCM signature: prototypes
from t0 features score 82.49 vs 80.62 from t4 features). Proposed
mechanism "coordinate pinning": each client stores, at first sight of a
task, per-class feature statistics (anchors); every later round it fits
a CLOSED-FORM map (affine or orthogonal Procrustes) pulling current
features back into the historical coordinate system, so the M2' refit
head works on effectively time-invariant coordinates.

This probe is the pre-registered falsification test BEFORE any
training-side work, using the SAME-SAMPLE correspondence between two
already-cached extractions of the same run:

  - reference time t_ref = task_00_end.pth (old task's OWN prompts);
  - current  time t_cur = task_04_end.pth (old task's rows re-extracted
    under the CURRENT task's prompts -- prompt_task_mode 'current',
    which is exactly what native evaluate() and the deployed M2' see);
  - task-0 rows are joined 1:1 across the two npz files by
    (client_id, split, sample_idx_relative) -- fail closed on any
    mismatch or duplicate; a z_ref invariance check (md5 of the joined
    task-0 rows in both files) validates the join, since z_ref is
    computed by the frozen backbone and must be time-invariant.

Per-client arms (head = the SAME recipe as probe_oracle_headroom:
hidden 256, ep 100, lr 1e-3, wd 1e-4, dropout 0.1, seed hashed per
identity; candidates restricted to the (client, task-0) class_mask):

  acc_ref : head refit on t_ref train rows, eval t_ref test rows
            -- the no-drift reference ceiling;
  acc_raw : head refit on t_cur train rows, eval t_cur test rows
            -- what a fresh head gets from DRIFTED coordinates (this is
            M2'-in-deployment's regime);
  acc_pin : closed-form map fitted on ANCHOR rows only (t_cur train ->
            t_ref train of the same samples; m anchors per class,
            m=0 = all matched train rows = correspondence ORACLE),
            then head refit on pinned train rows, eval pinned test;
  acc_ht  : head-transfer -- the acc_ref head evaluated on pinned
            t_cur test rows. If ~= acc_ref, coordinates are fully
            maintained (the pure interface story: untouched head +
            pinned coords).

DEPLOYMENT-REALISM AXIS: real clients store class statistics, not per-
sample features -- hence the anchors-per-class sweep (20 / 5 / 1 vs
oracle all). The sweep directly answers how many stored anchors the
mechanism actually needs; m=1 is the stats-only operating point.

PRE-REGISTERED GATE (per dataset, feat=z_op, do not weaken without
external review):

  ORACLE-VIABLE : at m=0 (all anchors) recovery_pct >= 30% and
                   recovered >= 0.5pp -- drift is affine-recoverable
                   at least in principle; else the mechanism is DEAD
                   (non-affine drift cannot be pinned closed-form).
  GREEN          : at m=20 recovery_pct >= 50% and recovered >= 0.8pp
                   -> training-side implementation is justified.
  RED            : best realistic arm (m>=1) recovered <= 0.3pp
                   -> noise-level; pinning is dead here.

recovered = acc_pin - acc_raw; recovery_pct = recovered /
(acc_ref - acc_raw) * 100 (undefined when drift <= 0).

Usage (from repo root):

  python diagnostics/probe_coord_pinning.py \
      --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
      --features_dir diagnostics_output/ncm_probe/ncm_inr_a2_seed42 \
      --out_csv diagnostics_output/coord_pinning/pin_inr_a2_t0t4.csv \
      --device cuda
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.ncm_probe import ensure_features  # noqa: E402
from diagnostics.probe_oracle_headroom import (  # noqa: E402
    fit_head, head_predict, stable_seed)


def parse_args():
    p = argparse.ArgumentParser("coordinate-pinning probe")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt_ref", default="task_00_end.pth",
                   help="reference-time task-boundary checkpoint (the "
                        "old task's OWN prompts)")
    p.add_argument("--ckpt_cur", default="task_04_end.pth",
                   help="current-time task-boundary checkpoint")
    p.add_argument("--features_dir", default="diagnostics_output")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--feat", default="z_op", choices=["z_op", "z_ref"])
    p.add_argument("--prompt_task_mode", default="current",
                   choices=["test", "current"])
    p.add_argument("--anchors_per_class", default="0,20,5,1",
                   help="comma-separated anchor counts per class; 0 = "
                        "all matched train rows (correspondence oracle)")
    p.add_argument("--map_types", default="affine,procrustes")
    p.add_argument("--affine_ridge", default=1e-2, type=float,
                   help="relative ridge for the affine map (see "
                        "fit_affine docstring)")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default="cpu")
    # head recipe -- MUST stay identical to probe_oracle_headroom so the
    # arms are directly comparable to its mlp_local numbers
    p.add_argument("--hidden", default=256, type=int)
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--wd", default=1e-4, type=float)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--seed_base", default=20261028, type=int)
    p.add_argument("--threads", default=4, type=int,
                   help="torch CPU thread cap -- keep small on shared "
                        "servers; the probe's CPU work is light and "
                        "does not scale with more threads")
    # pre-registered gate thresholds (pp / %); frozen, do not weaken
    p.add_argument("--gate_oracle_rec_pp", default=0.5, type=float)
    p.add_argument("--gate_oracle_rec_pct", default=30.0, type=float)
    p.add_argument("--gate_green_rec_pp", default=0.8, type=float)
    p.add_argument("--gate_green_rec_pct", default=50.0, type=float)
    p.add_argument("--gate_red_rec_pp", default=0.3, type=float)
    return p.parse_args()


def _boundary(ckpt_name):
    m = re.fullmatch(r"task_(\d+)_end\.pth", Path(ckpt_name).name)
    if not m:
        raise SystemExit(f"--ckpt_ref/--ckpt_cur only accept "
                         f"task_XX_end.pth, got {ckpt_name!r}")
    return int(m.group(1))


def load_boundary(run_dir, ckpt, feat_dir, data_path, device, prompt_mode):
    npz_path, meta_path = ensure_features(
        run_dir, ckpt, feat_dir, data_path,
        device if device != "cpu" else None, prompt_mode)
    data = np.load(npz_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    masks = {int(cid): [list(map(int, ts)) for ts in tmask]
             for cid, tmask in meta["client_class_masks"].items()}
    return data, masks


def join_task0(data_ref, data_cur, t_ref):
    """1:1 join of the reference-boundary rows (tasks 0..t_ref; with the
    default t_ref=0 these are exactly task-0 rows) against the SAME
    samples in the current-boundary file. Join key: (client_id, split,
    sample_idx_relative). Returns index arrays (idx_ref, idx_cur) plus
    the shared metadata arrays, fail-closed on duplicates, missing
    rows, or label disagreement."""
    ref_rows = np.arange(data_ref["labels"].shape[0])
    cur_rows = np.where(data_cur["task_ids"] <= t_ref)[0]

    def keys(d, rows):
        return (d["client_ids"][rows].astype(np.int64) * 4
                + d["splits"][rows].astype(np.int64)) * 10 ** 9 \
            + d["sample_idx_relative"][rows].astype(np.int64)

    k_ref, k_cur = keys(data_ref, ref_rows), keys(data_cur, cur_rows)
    if len(np.unique(k_ref)) != len(k_ref):
        raise RuntimeError("duplicate (client, split, sample_idx) keys "
                           "in the REFERENCE npz -- join cannot proceed")
    if len(np.unique(k_cur)) != len(k_cur):
        raise RuntimeError("duplicate (client, split, sample_idx) keys "
                           "in the CURRENT npz -- join cannot proceed")
    order = {k: i for i, k in enumerate(k_cur)}
    missing = [int(k) for k in k_ref if k not in order]
    if missing:
        raise RuntimeError(f"{len(missing)} reference rows have no "
                           f"counterpart in the current npz -- the two "
                           f"extractions do not cover the same samples")
    # `order` values are positions WITHIN cur_rows, not absolute row
    # indices (the current npz holds all tasks, not only 0..t_ref) --
    # translate before indexing the full arrays
    idx_cur = cur_rows[np.asarray([order[k] for k in k_ref],
                                  dtype=np.int64)]
    if not np.array_equal(data_ref["labels"], data_cur["labels"][idx_cur]):
        raise RuntimeError("joined rows disagree on labels -- the two "
                           "npz files are not the same samples")
    if not np.array_equal(data_ref["client_ids"],
                          data_cur["client_ids"][idx_cur]):
        raise RuntimeError("joined rows disagree on client_id")
    if not np.array_equal(data_ref["splits"],
                          data_cur["splits"][idx_cur]):
        raise RuntimeError("joined rows disagree on split")
    if not np.array_equal(data_ref["task_ids"],
                          data_cur["task_ids"][idx_cur]):
        raise RuntimeError("joined rows disagree on task_id")
    return ref_rows, idx_cur


def fit_affine(x, y, ridge):
    """Ridge-regularized affine x -> y with bias. Ridge is RELATIVE:
    lam = ridge * trace(X'X)/ncol -- without it, the m=0 arm sits on the
    interpolation cliff (per-client train rows ~ d+1, the raw least
    squares memorizes the anchor rows and behaves arbitrarily off them,
    observed 2026-10-31: train resid 0.03 but pinned-test accuracy
    COLLAPSED below raw). Small anchor sets (m=1/5) are underdetermined;
    the ridge shrinks them instead of exploding. Solved with
    torch.linalg in float64: this machine's numpy LAPACK gelsd is
    broken (NaN on well-posed systems, verified 2026-10-31), torch's
    LAPACK is not."""
    X = torch.tensor(np.concatenate(
        [x, np.ones((len(x), 1), dtype=np.float64)], axis=1))
    Y = torch.tensor(np.asarray(y, dtype=np.float64))
    G = X.T @ X
    lam = ridge * float(torch.diagonal(G).sum()) / G.shape[0]
    A = torch.linalg.solve(G + lam * torch.eye(G.shape[0],
                                               dtype=torch.float64),
                           X.T @ Y)
    def apply(z):
        Z = torch.tensor(np.concatenate(
            [z, np.ones((len(z), 1), dtype=np.float64)], axis=1))
        return (Z @ A).numpy().astype(np.float32)
    return apply


def fit_procrustes(x, y):
    """Orthogonal Procrustes with translation and global scale:
    minimize ||s*(x-mu_x)R - (y-mu_y)||_F s.t. R orthogonal. Same
    torch/float64 rationale as fit_affine."""
    mu_x = torch.tensor(np.asarray(x, dtype=np.float64).mean(axis=0))
    mu_y = torch.tensor(np.asarray(y, dtype=np.float64).mean(axis=0))
    Xc = torch.tensor(np.asarray(x, dtype=np.float64)) - mu_x
    Yc = torch.tensor(np.asarray(y, dtype=np.float64)) - mu_y
    U, S, Vh = torch.linalg.svd(Xc.T @ Yc)
    R = U @ Vh
    denom = float((Xc ** 2).sum())
    s = (float(S.sum()) / denom) if denom > 1e-12 else 1.0
    def apply(z):
        Z = torch.tensor(np.asarray(z, dtype=np.float64))
        return (s * ((Z - mu_x) @ R) + mu_y).numpy().astype(np.float32)
    return apply


def pick_anchors(y_tr, m, seed):
    """Positions (WITHIN the client's matched train rows) used to fit
    the pinning map: up to m rows per class, deterministically sampled;
    m=0 -> all matched train rows (correspondence ORACLE)."""
    idx = np.arange(len(y_tr))
    if m <= 0:
        return idx
    rng = np.random.RandomState(seed)
    picked = []
    for c in np.unique(y_tr):
        rows = idx[y_tr == c]
        picked.extend(rows if len(rows) <= m
                      else rng.choice(rows, size=m, replace=False))
    return np.asarray(sorted(picked), dtype=np.int64)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    run_dir = Path(args.run_dir)
    feat_dir = Path(args.features_dir)
    t_ref, t_cur = _boundary(args.ckpt_ref), _boundary(args.ckpt_cur)
    if t_ref >= t_cur:
        raise SystemExit(f"reference boundary ({t_ref}) must precede "
                         f"current boundary ({t_cur})")
    m_levels = [int(x) for x in args.anchors_per_class.split(",")]
    map_types = [x.strip() for x in args.map_types.split(",")]

    d_ref, _ = load_boundary(run_dir, args.ckpt_ref, feat_dir,
                             args.data_path, args.device,
                             args.prompt_task_mode)
    d_cur, masks = load_boundary(run_dir, args.ckpt_cur, feat_dir,
                                 args.data_path, args.device,
                                 args.prompt_task_mode)
    idx_ref, idx_cur = join_task0(d_ref, d_cur, t_ref)

    feat = args.feat
    z_r, z_c = d_ref[feat][idx_ref], d_cur[feat][idx_cur]
    labels = d_ref["labels"][idx_ref]
    cids = d_ref["client_ids"][idx_ref]
    tr = d_ref["splits"][idx_ref] == 0
    te = d_ref["splits"][idx_ref] == 1

    # join validation: z_ref comes from the frozen backbone and must be
    # bit-identical across the two extraction times -- if it is not,
    # either the join is wrong or z_ref is not invariant; report both.
    md5 = lambda a: hashlib.md5(  # noqa: E731
        np.ascontiguousarray(a)).hexdigest()
    zref_same = (md5(d_ref["z_ref"][idx_ref]) ==
                 md5(d_cur["z_ref"][idx_cur]))
    print(f"[pin-probe] joined {len(idx_ref)} rows "
          f"(boundary {t_ref} -> {t_cur}, feat={feat}); "
          f"z_ref time-invariant: {zref_same}")

    hp = dict(hidden=args.hidden, epochs=args.epochs, lr=args.lr,
              wd=args.wd, dropout=args.dropout, batch_size=args.batch_size,
              device=args.device)

    rows = []
    for cid in sorted(masks):
        sel = cids == cid
        sel_tr, sel_te = sel & tr, sel & te
        allowed = masks[cid][0]
        ytr, yte = labels[sel_tr], labels[sel_te]
        if len(yte) == 0:
            continue
        z_tr_r, z_te_r = z_r[sel_tr], z_r[sel_te]
        z_tr_c, z_te_c = z_c[sel_tr], z_c[sel_te]

        head_ref = fit_head(z_tr_r, ytr, allowed,
                            seed=stable_seed(args.seed_base, "ref",
                                             cid, 0, feat), **hp)
        acc_ref = head_predict(head_ref, z_te_r, allowed)
        head_raw = fit_head(z_tr_c, ytr, allowed,
                            seed=stable_seed(args.seed_base, "raw",
                                             cid, 0, feat), **hp)
        acc_raw = head_predict(head_raw, z_te_c, allowed)

        base = {"client_id": int(cid), "feat": feat,
                "n_train": int(sel_tr.sum()), "n_test": int(sel_te.sum()),
                "acc_ref": float(np.mean(acc_ref == yte)),
                "acc_raw": float(np.mean(acc_raw == yte))}
        for mt in map_types:
            fitter = (lambda a, b: fit_affine(a, b, args.affine_ridge)) \
                if mt == "affine" else fit_procrustes
            for m in m_levels:
                a_idx = pick_anchors(ytr, m,
                                     stable_seed(args.seed_base, "anchor",
                                                 cid, mt, m))
                mapper = fitter(z_tr_c[a_idx], z_tr_r[a_idx])
                pin_tr, pin_te = mapper(z_tr_c), mapper(z_te_c)
                resid = float(np.linalg.norm(pin_tr - z_tr_r)
                              / (np.linalg.norm(z_tr_r) + 1e-12))
                head_pin = fit_head(pin_tr, ytr, allowed,
                                   seed=stable_seed(args.seed_base, "pin",
                                                    cid, mt, m, feat), **hp)
                acc_pin = head_predict(head_pin, pin_te, allowed)
                acc_ht = head_predict(head_ref, pin_te, allowed)
                rows.append({**base, "map_type": mt,
                             "anchors_per_class": m,
                             "n_anchor_rows": int(len(a_idx)),
                             "resid": round(resid, 4),
                             "acc_pin": float(np.mean(acc_pin == yte)),
                             "acc_head_transfer": float(np.mean(
                                 acc_ht == yte))})

    for r in rows:
        for k in ("acc_ref", "acc_raw", "acc_pin", "acc_head_transfer"):
            r[k] = round(100.0 * r[k], 2)
        r["drift_pp"] = round(r["acc_ref"] - r["acc_raw"], 2)
        r["recovered_pp"] = round(r["acc_pin"] - r["acc_raw"], 2)
        r["recovery_pct"] = round(100.0 * r["recovered_pp"] / r["drift_pp"],
                                   1) if r["drift_pp"] > 0 else None

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summaries = []
    for mt in map_types:
        for m in m_levels:
            sub = [r for r in rows if r["map_type"] == mt
                   and r["anchors_per_class"] == m]
            if not sub:
                continue
            mean = lambda k: round(float(np.mean([r[k] for r in sub])), 2)  # noqa: E731
            drift = round(mean("acc_ref") - mean("acc_raw"), 2)
            rec = round(mean("acc_pin") - mean("acc_raw"), 2)
            entry = {"map_type": mt, "anchors_per_class": m,
                     "feat": feat, "zref_invariant": bool(zref_same),
                     "aa_acc_ref": mean("acc_ref"),
                     "aa_acc_raw": mean("acc_raw"),
                     "aa_acc_pin": mean("acc_pin"),
                     "aa_acc_head_transfer": mean("acc_head_transfer"),
                     "aa_resid": round(mean("resid"), 4),
                     "drift_pp": drift, "recovered_pp": rec,
                     "recovery_pct": round(100.0 * rec / drift, 1)
                     if drift > 0 else None}
            if m == 0:
                viable = (rec >= args.gate_oracle_rec_pp
                          and (entry["recovery_pct"] or 0)
                          >= args.gate_oracle_rec_pct)
                entry["gate_oracle_viable"] = bool(viable)
            if m >= 1:
                if rec >= args.gate_green_rec_pp and \
                        (entry["recovery_pct"] or 0) >= args.gate_green_rec_pct:
                    verdict = "GREEN"
                elif rec <= args.gate_red_rec_pp:
                    verdict = "RED"
                else:
                    verdict = "YELLOW"
                entry["gate"] = {"verdict": verdict,
                                 "recovered_pp": rec,
                                 "recovery_pct": entry["recovery_pct"]}
            summaries.append(entry)

    with open(out_csv.with_suffix(".summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"hyper": {k: getattr(args, k) for k in
                             ("hidden", "epochs", "lr", "wd", "dropout",
                              "batch_size", "seed_base", "device")},
                   "boundaries": {"ref": t_ref, "cur": t_cur},
                   "gate": {"oracle_rec_pp": args.gate_oracle_rec_pp,
                            "oracle_rec_pct": args.gate_oracle_rec_pct,
                            "green_rec_pp": args.gate_green_rec_pp,
                            "green_rec_pct": args.gate_green_rec_pct,
                            "red_rec_pp": args.gate_red_rec_pp},
                   "summaries": summaries}, f, indent=2)

    print(f"[pin-probe] per-(client, map, m) rows -> {out_csv}")
    print(f"[pin-probe] AA summary (feat={feat}, "
          f"{t_ref}->{t_cur}, zref_invariant={zref_same}):")
    hdr = (f"{'map':>10} {'m':>4} {'ref':>7} {'raw':>7} {'pin':>7} "
           f"{'headtr':>7} {'resid':>7} {'drift':>6} {'rec':>6} "
           f"{'rec%':>6} {'gate':>7}")
    print(hdr)
    for s in summaries:
        v = s.get("gate", {}).get("verdict",
                                 "VIABLE" if s.get("gate_oracle_viable")
                                 else "DEAD")
        rp = s["recovery_pct"] if s["recovery_pct"] is not None else float("nan")
        print(f"{s['map_type']:>10} {s['anchors_per_class']:>4} "
              f"{s['aa_acc_ref']:>7.2f} {s['aa_acc_raw']:>7.2f} "
              f"{s['aa_acc_pin']:>7.2f} {s['aa_acc_head_transfer']:>7.2f} "
              f"{s['aa_resid']:>7.4f} {s['drift_pp']:>6.2f} "
              f"{s['recovered_pp']:>6.2f} {rp:>6.1f} {v:>7}")


if __name__ == "__main__":
    main()
