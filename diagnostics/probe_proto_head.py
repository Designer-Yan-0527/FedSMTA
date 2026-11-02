"""Prototype-readout probe: local vs pooled vs semi-shared class prototypes.

Fits closed-form prototype heads directly on cached features (no training)
and compares three aggregation policies per (client, test_task):
  * local      -- prototypes from the evaluating client's own train rows
  * global     -- prototypes pooled across ALL clients (classic NCM)
  * semi       -- public classes pooled (federated consensus), private
                  classes local-only (never leave the client)
plus per-client / global standardized Euclidean NCM variants. Cosine is
scale-invariant so it needs no standardization decision.

Judgment targets (pre-registered):
  1. does any prototype variant match the tuned-MLP oracle ceiling
     (aa_mlp_local in oracle_headroom outputs) for the same run?
  2. does 'semi' beat 'global' (aggregation gain without interference)?
  3. how far is the best variant above the run's actual M2' final AA?

Feature identity is fail-closed via ncm_probe.ensure_features (meta must
match run_dir + seed + checkpoint + prompt mode). Every allowed class must
have a prototype for the policy in use -- ncm_predict raises otherwise.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from ncm_probe import ensure_features, load_native, ncm_predict

VARIANTS_COS = ["cos_local", "cos_global", "cos_semi"]
VARIANTS_EUC = ["eucstd_local", "eucstd_global", "eucstd_semi"]
ALL_VARIANTS = VARIANTS_COS + VARIANTS_EUC


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoints", default="task_04_end.pth",
                   help="comma-separated task_XX_end.pth files")
    p.add_argument("--features_dir", default="diagnostics_output")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--feat", default="z_op", choices=["z_ref", "z_op"])
    p.add_argument("--prompt_task_mode", default="current",
                   choices=["test", "current"])
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def protos_semi(local_p, global_p, allowed, public_set):
    """Semi-shared policy: public classes take the pooled (federated
    consensus) prototype, private classes the local one. Fail closed if a
    class is missing from both sources."""
    out = {}
    for c in allowed:
        src = global_p if c in public_set else local_p
        if c not in src:
            raise RuntimeError(
                f"class {c} has no prototype under the semi policy "
                f"(public={c in public_set})")
        out[c] = src[c]
    return out


def protos_local_fb(allowed, protos_loc, protos_glob):
    """Strictly-local policy with EXPLICIT coverage fallback: public
    classes with no local train rows (data held by other clients) borrow
    the pooled prototype so the choice set never shrinks silently. Returns
    (protos, n_fallback) -- the fallback count is reported per row."""
    out, n_fb = {}, 0
    for c in allowed:
        if c in protos_loc:
            out[c] = protos_loc[c]
        else:
            out[c] = protos_glob[c]
            n_fb += 1
    return out, n_fb


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    native = load_native(run_dir)
    rows = []

    for ckpt in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        m = re.fullmatch(r"task_(\d+)_end\.pth", Path(ckpt).name)
        if not m:
            raise SystemExit(f"--checkpoints only accepts task_XX_end.pth, "
                             f"got {ckpt!r}")
        t = int(m.group(1))
        npz_path, meta_path = ensure_features(
            run_dir, ckpt, Path(args.features_dir), args.data_path,
            args.device, args.prompt_task_mode)
        data = np.load(npz_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        masks = {int(cid): [list(map(int, ts)) for ts in tmask]
                 for cid, tmask in meta["client_class_masks"].items()}
        public_set = set(int(c) for c in meta["public_classes"])

        z = data[args.feat]
        labels = data["labels"]
        client_ids = data["client_ids"]
        task_ids = data["task_ids"]
        splits = data["splits"]
        # continual accumulation semantics: only rows from tasks seen up to
        # boundary t may be used to build prototypes
        tr_rows = (splits == 0) & (task_ids <= t)

        def build(z_, sel):
            out = {}
            lab = labels[sel]
            for c in np.unique(lab):
                msk = lab == c
                out[int(c)] = z_[sel][msk].mean(axis=0)
            return out

        # raw-space prototypes
        protos_global = build(z, tr_rows)
        protos_local = {cid: build(z, tr_rows & (client_ids == cid))
                        for cid in sorted(masks)}
        # global-standardized space (single consistent scaling for pooled /
        # semi policies)
        mu_g = z[tr_rows].mean(axis=0)
        sd_g = z[tr_rows].std(axis=0)
        sd_g[sd_g < 1e-6] = 1e-6
        z_gs = (z - mu_g) / sd_g
        protos_global_gs = build(z_gs, tr_rows)
        protos_local_gs = {cid: build(z_gs, tr_rows & (client_ids == cid))
                           for cid in sorted(masks)}

        for cid in sorted(masks):
            # per-client standardization stats (local deployment scaling)
            tr_c = tr_rows & (client_ids == cid)
            if not tr_c.any():
                raise RuntimeError(f"client {cid} has no train rows <= task "
                                   f"{t}")
            mu_c = z[tr_c].mean(axis=0)
            sd_c = z[tr_c].std(axis=0)
            sd_c[sd_c < 1e-6] = 1e-6
            z_cs = (z - mu_c) / sd_c
            protos_local_cs = build(z_cs, tr_c)

            for k in range(t + 1):
                sel = ((splits == 1) & (client_ids == cid)
                       & (task_ids == k))
                if not sel.any():
                    continue
                zt = z[sel]
                yt = labels[sel]
                allowed = masks[cid][k]
                rec = {
                    "after_task": t, "test_task": k, "client_id": cid,
                    "n_test": int(sel.sum()), "feat": args.feat,
                    "acc_native": native[(cid, t, k)],
                }
                # local policy with explicit coverage fallback: public
                # classes with no local train rows borrow the pooled
                # prototype (counted, never silent)
                p_loc_cos, n_fb = protos_local_fb(
                    allowed, protos_local[cid], protos_global)
                # same fallback in the client's standardized space: the
                # affine image of the raw pooled prototype under
                # (z - mu_c) / sd_c
                p_loc_euc = {
                    c: (protos_local_cs[c] if c in protos_local_cs
                        else (protos_global[c] - mu_c) / sd_c)
                    for c in allowed}
                rec["n_fallback_local"] = n_fb
                # cosine family (raw space, scale-invariant)
                cos_preds = {
                    "cos_local": ncm_predict(zt, p_loc_cos, allowed)[0],
                    "cos_global": ncm_predict(zt, protos_global, allowed)[0],
                    "cos_semi": ncm_predict(
                        zt, protos_semi(protos_local[cid], protos_global,
                                        allowed, public_set), allowed)[0],
                }
                # standardized Euclidean family
                zt_cs = z_cs[sel]
                euc_preds = {
                    "eucstd_local": ncm_predict(zt_cs, p_loc_euc,
                                                allowed)[1],
                    "eucstd_global": ncm_predict(z_gs[sel],
                                                 protos_global_gs,
                                                 allowed)[1],
                    "eucstd_semi": ncm_predict(
                        z_gs[sel],
                        protos_semi(protos_local_gs[cid], protos_global_gs,
                                    allowed, public_set), allowed)[1],
                }
                for name, preds in {**cos_preds, **euc_preds}.items():
                    acc = 100.0 * float(np.mean(
                        [p == int(g) for p, g in zip(preds, yt)]))
                    rec[f"acc_{name}"] = round(acc, 2)
                rows.append(rec)

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # summary: per variant, AA = mean over seen tasks of client-mean
    print(f"[proto-probe] per-(client, task) rows -> {args.out_csv}")
    print(f"[proto-probe] AA summary (feat={args.feat}):")
    header = f"{'after_task':>10} {'native':>8} " + " ".join(
        f"{v:>13}" for v in ALL_VARIANTS)
    print(header)
    for t in sorted({r["after_task"] for r in rows}):
        sub_t = [r for r in rows if r["after_task"] == t]
        nat = np.mean([r["acc_native"] for r in sub_t])
        cells = []
        for v in ALL_VARIANTS:
            per_task = []
            for k in sorted({r["test_task"] for r in sub_t}):
                per_task.append(np.mean(
                    [r[f"acc_{v}"] for r in sub_t
                     if r["test_task"] == k]))
            cells.append(f"{np.mean(per_task):>13.2f}")
        print(f"{t:>10} {nat:>8.2f} " + " ".join(cells))
    print("[proto-probe] variants: cos = cosine similarity; eucstd = "
          "Euclidean NCM on standardized features "
          "(local = per-client stats, global = pooled stats); "
          "semi = public classes pooled + private classes local.")


if __name__ == "__main__":
    main()
