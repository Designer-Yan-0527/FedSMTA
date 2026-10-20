"""P1 probe (A5 Gate A): frozen NCM classification on existing features.

Question (decides whether A5 needs a trainable shared layer): how much of
C2Prompt's advantage over A2 on ImageNet-R can be recovered by a BETTER
CLASS-LEVEL CLASSIFICATION INTERFACE alone -- class prototypes in a fixed
feature space, zero training?

Design (frozen 2026-10-20, P0 audit + external review converged):

  - feature spaces: z_ref (frozen pretrained ViT feat -- the stable
    historical-interface coordinate system A5 would build on) and, as a
    secondary reading, z_op (current prompt-enhanced pre-anchor feat);
  - prototypes: per-class MEAN over the inner TRAIN split of the tasks
    SEEN at that boundary only (extract_features at task_XX_end already
    limits rows to tasks 0..XX). NO surrogate, NO test samples, NO
    future-task classes -> leakage-free by construction;
  - evaluation: per (client, test_task k <= t) on the inner TEST split,
    candidates restricted to the (client, task) class_mask -- the same
    mask semantics as the native R[t][k] evaluation, so the joined
    native numbers (run metrics client CSVs) are directly comparable;
  - NCM variants: cosine and Euclidean nearest-mean;
  - native path: joined from {run_dir}/metrics/client_X_accuracy.csv
    (recorded at the same boundaries during the A2 run).

T0-gap criterion (from the review): if NCM already closes a large part
of the ~4pp T0 gap to C2Prompt, A5's shared interface is primarily a
base-discrimination fix; gains that only appear at later boundaries
support the continuing-transfer narrative instead.

Usage (from repo root):

  python diagnostics/ncm_probe.py \
      --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
      --checkpoints task_00_end.pth,task_04_end.pth \
      --out_csv diagnostics_output/ncm_probe_inr_a2.csv \
      [--c2p "86.86,91.14"]   # optional per-boundary reference AAs

Features are extracted on demand via diagnostics/extract_features.py
(--split both) and cached as diagnostics_output/features_*.npz.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACT = Path(__file__).resolve().parent / "extract_features.py"


def parse_args():
    p = argparse.ArgumentParser("P1 frozen NCM probe")
    p.add_argument("--run_dir", required=True,
                   help="run directory (.../{dataset}/{method}/{run_name}/"
                        "seed_{seed})")
    p.add_argument("--checkpoints", default="task_00_end.pth,task_04_end.pth",
                   help="comma-separated checkpoint names relative to "
                        "run_dir (each must be a task-boundary "
                        "task_XX_end.pth)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--features_dir", default="diagnostics_output")
    p.add_argument("--c2p", default=None,
                   help="optional comma-separated C2Prompt reference AAs, "
                        "one per checkpoint in the same order (e.g. "
                        "'86.86,91.14')")
    p.add_argument("--data_path", default=None,
                   help="forwarded to extract_features.py if extraction "
                        "is needed")
    p.add_argument("--device", default=None)
    return p.parse_args()


def ensure_features(run_dir, ckpt_rel, feat_dir, data_path, device):
    """Return (npz_path, meta_path), extracting via extract_features.py
    (--split both) if the cached npz does not exist yet."""
    with open(Path(run_dir) / "args.json", "r", encoding="utf-8") as f:
        run_args = json.load(f)
    # diag_utils.resolve_checkpoint resolves bare names against run_dir, not
    # run_dir/checkpoints -- normalize a bare task_XX_end.pth to the
    # checkpoints/-relative form
    if "/" not in ckpt_rel and not Path(run_dir, ckpt_rel).is_file():
        ckpt_rel = f"checkpoints/{ckpt_rel}"
    stem = f"features_{run_args['data_name']}_{Path(ckpt_rel).stem}"
    npz = feat_dir / f"{stem}.npz"
    meta = feat_dir / f"{stem}.json"
    if not npz.is_file():
        feat_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(EXTRACT),
               "--run_dir", str(run_dir),
               "--checkpoint", str(ckpt_rel),
               "--split", "both",
               "--out_dir", str(feat_dir)]
        if data_path:
            cmd += ["--data_path", data_path]
        if device:
            cmd += ["--device", device]
        print(f"[ncm-probe] extracting features: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    if not npz.is_file() or not meta.is_file():
        raise RuntimeError(f"feature files missing after extraction: {npz}")
    return npz, meta


def build_protos(z, labels):
    """Per-class mean over the given (train-split) rows."""
    protos = {}
    for c in np.unique(labels):
        m = labels == c
        if m.sum() > 0:
            protos[int(c)] = z[m].mean(axis=0)
    return protos


def ncm_predict(z, protos, allowed):
    """allowed: list of class ids with prototypes. Returns predicted
    class ids under cosine and Euclidean NCM."""
    classes = [c for c in allowed if c in protos]
    if not classes:
        raise RuntimeError("no prototype covers the class mask")
    P = np.stack([protos[c] for c in classes])  # (K, d)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    cos_pred = [classes[i] for i in (zn @ Pn.T).argmax(axis=1)]
    d2 = ((z[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    euc_pred = [classes[i] for i in d2.argmin(axis=1)]
    return cos_pred, euc_pred


def load_native(run_dir):
    """(client, after_task, test_task) -> accuracy from the run's
    metrics client CSVs."""
    native = {}
    metrics = Path(run_dir) / "metrics"
    for path in sorted(metrics.glob("client_*_accuracy.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                native[(int(row["client_id"]), int(row["after_task"]),
                        int(row["test_task"]))] = float(row["accuracy"])
    if not native:
        raise RuntimeError(f"no client accuracy CSVs under {metrics}")
    return native


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    feat_dir = Path(args.features_dir)
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    c2p = ([float(x) for x in args.c2p.split(",")]
           if args.c2p else [None] * len(ckpts))

    native = load_native(run_dir)
    rows = []
    summaries = []

    for ckpt, ref in zip(ckpts, c2p):
        m = re.fullmatch(r"task_(\d+)_end\.pth", Path(ckpt).name)
        if not m:
            raise SystemExit(f"--checkpoints only accepts task_XX_end.pth, "
                             f"got {ckpt!r}")
        t = int(m.group(1))
        npz_path, meta_path = ensure_features(run_dir, ckpt, feat_dir,
                                              args.data_path, args.device)
        data = np.load(npz_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        masks = {int(cid): [list(map(int, ts)) for ts in tmask]
                 for cid, tmask in meta["client_class_masks"].items()}

        for feat in ("z_ref", "z_op"):
            z = data[feat]
            labels = data["labels"]
            tr = data["splits"] == 0
            protos = build_protos(z[tr], labels[tr])

            for cid in sorted(masks):
                for k in range(t + 1):
                    sel = ((data["splits"] == 1) & (data["client_ids"] == cid)
                           & (data["task_ids"] == k))
                    if not sel.any():
                        continue
                    zt = z[sel]
                    yt = labels[sel]
                    allowed = masks[cid][k]
                    cos_pred, euc_pred = ncm_predict(zt, protos, allowed)
                    acc_cos = 100.0 * float(np.mean(
                        [p == int(g) for p, g in zip(cos_pred, yt)]))
                    acc_euc = 100.0 * float(np.mean(
                        [p == int(g) for p, g in zip(euc_pred, yt)]))
                    rows.append({
                        "after_task": t, "test_task": k, "client_id": cid,
                        "n_test": int(sel.sum()), "feat": feat,
                        "acc_native": native[(cid, t, k)],
                        "acc_ncm_cos": round(acc_cos, 2),
                        "acc_ncm_euc": round(acc_euc, 2),
                    })

        # per-boundary summary per feature space: client-mean per task,
        # then AA over seen tasks (identical aggregation to the run's
        # accuracy_matrix)
        for feat in ("z_ref", "z_op"):
            sub = [r for r in rows
                   if r["after_task"] == t and r["feat"] == feat]
            per_task = {}
            for r in sub:
                per_task.setdefault(r["test_task"], []).append(
                    (r["acc_native"], r["acc_ncm_cos"], r["acc_ncm_euc"]))
            nat = np.mean([np.mean([v[0] for v in ts])
                           for ts in per_task.values()])
            cos = np.mean([np.mean([v[1] for v in ts])
                           for ts in per_task.values()])
            euc = np.mean([np.mean([v[2] for v in ts])
                           for ts in per_task.values()])
            summaries.append({"after_task": t, "feat": feat,
                              "aa_native": round(float(nat), 2),
                              "aa_ncm_cos": round(float(cos), 2),
                              "aa_ncm_euc": round(float(euc), 2),
                              "c2p_ref": ref})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_csv.with_suffix(".summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"[ncm-probe] per-(client, task) rows -> {out_csv}")
    print(f"[ncm-probe] AA summary (client-mean per task, then mean over "
          f"seen tasks):")
    hdr = (f"{'after_task':>10} {'feat':>6} {'native':>8} {'ncm_cos':>8} "
           f"{'ncm_euc':>8} {'c2p_ref':>8}")
    print(hdr)
    for s in summaries:
        print(f"{s['after_task']:>10} {s['feat']:>6} {s['aa_native']:>8.2f} "
              f"{s['aa_ncm_cos']:>8.2f} {s['aa_ncm_euc']:>8.2f} "
              f"{(s['c2p_ref'] if s['c2p_ref'] is not None else float('nan')):>8.2f}")


if __name__ == "__main__":
    main()
