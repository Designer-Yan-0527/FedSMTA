"""Oracle headroom probe (A5 Gate A-2): trainable-interface ceiling.

Question this probe answers BEFORE any A5-v2 engineering is started:
the NCM probe (2026-10-20) proved the STATISTICAL readout on our frozen
and prompt-enhanced features is saturated (z_op NCM ~= native Chead ~=
82.5 on ImageNet-R T0), which killed the pure-statistical A5 interface
(the A5-P2-v1 form). But that verdict does NOT prove a TRAINABLE
interface can extract more: C2Prompt's 86.86 is achieved on C2P's OWN
features (its prompt + contrastive training), so the ~4pp T0 gap may
live in feature quality -- reachable by no interface of ours. This
probe measures how much a supervised head can actually extract from OUR
feature npz files, at three data-access levels.

Design (frozen before the first run; the gate below is pre-registered):

  - three supervised probes per feature space, all trained on the inner
    TRAIN split only and evaluated on the inner TEST split with
    candidates restricted to the (client, task) class_mask -- the exact
    native R[t][k] / ncm_probe semantics, so numbers are directly
    comparable to native and to the NCM verdict:
      * linear-local : per-(client, task) multinomial logistic
        regression. Sanity anchor -- both it and the native Chead are
        trainable LINEAR interfaces on the same data, so it should land
        near native; a large deviation means a probe bug, not a finding.
      * mlp-local    : per-(client, task) 2-layer MLP = what a LOCAL-ONLY
        trainable semantic interface could reach (the future
        "local-only g" ablation, answered before building it).
      * mlp-pooled   : ONE MLP per boundary trained on the POOLED train
        rows of all clients' seen tasks = the information-content
        ORACLE. Cross-client pooling with labels violates the federated
        protocol -- it is a DIAGNOSTIC upper bound on "is the
        information in the features at all", NOT a method and NOT a
        claim. Its evaluation is still per-(client, task) with the same
        mask restriction.
  - decomposition:
        mlp_pooled - native   = total extractable headroom (Gate input)
        mlp_local  - native   = headroom reachable WITHOUT federation
        mlp_pooled - mlp_local= the federated premise preview (what
                                FedAvg of a shared interface could at
                                best add; ~0 weakens A5-v2's FedAvg
                                claim)
  - features come through ncm_probe.ensure_features verbatim: same cache
    files, same fail-closed identity checks (run_dir/seed/checkpoint/
    nb_classes/prompt_task_mode), same on-demand extraction via
    extract_features.py --split both.
  - aa_native is joined from the run's client CSVs and cross-checked
    against accuracy_matrix.csv (fail closed on >0.02pp), exactly like
    ncm_probe.

PRE-REGISTERED GATE (T0 row only, z_op primary reading -- z_op under
--prompt_task_mode current matches native evaluate() and at T0 is fully
online-legal; do not weaken thresholds without external review):

  GREEN : aa_mlp_pooled - aa_native >= 2.00pp
          -> A5-v2 (trainable shared semantic interface) 立项
  RED   : aa_mlp_pooled - aa_native <= 1.00pp
          -> the interface line is DEAD; the gap lives in feature
             quality -> pivot to feature-shaping components
  YELLOW: 1.00pp < delta < 2.00pp
          -> escalate to external review with the numbers

Readings at later boundaries and on z_ref are CONTEXT ONLY (z_op beyond
T0 re-extracts old tasks under current prompts = retrospective
re-access), never Gate input.

Usage (from repo root):

  python diagnostics/probe_oracle_headroom.py \
      --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
      --checkpoints task_00_end.pth,task_04_end.pth \
      --out_csv diagnostics_output/oracle_headroom_inr_a2.csv \
      --c2p 86.86,91.14
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

from diagnostics.ncm_probe import ensure_features, load_native  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("A5 Gate A-2 oracle headroom probe")
    p.add_argument("--run_dir", required=True,
                   help="run directory (.../{dataset}/{method}/{run_name}/"
                        "seed_{seed})")
    p.add_argument("--checkpoints", default="task_00_end.pth,task_04_end.pth",
                   help="comma-separated task-boundary checkpoints relative "
                        "to run_dir")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--features_dir", default="diagnostics_output")
    p.add_argument("--c2p", default=None,
                   help="optional comma-separated C2Prompt reference AAs, "
                        "one per checkpoint in the same order")
    p.add_argument("--prompt_task_mode", default="current",
                   choices=["test", "current"],
                   help="z_op prompt condition (ncm_probe semantics); "
                        "'current' (default) matches native evaluate()")
    p.add_argument("--data_path", default=None,
                   help="forwarded to extract_features.py if extraction "
                        "is needed")
    p.add_argument("--device", default="cpu",
                   help="'cpu' (default; fully deterministic) or 'cuda'")
    # probe hyper-parameters (recorded into the summary for provenance)
    p.add_argument("--hidden", default=256, type=int,
                   help="MLP hidden width (0 = linear head)")
    p.add_argument("--epochs", default=100, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--wd", default=1e-4, type=float)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--seed_base", default=20261028, type=int)
    # pre-registered gate thresholds (pp); defaults are frozen, do not
    # weaken without external review
    p.add_argument("--gate_green", default=2.0, type=float)
    p.add_argument("--gate_red", default=1.0, type=float)
    return p.parse_args()


def stable_seed(*parts):
    """Deterministic 31-bit seed from arbitrary key parts (probe identity:
    boundary / feat / scope / client / task)."""
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


class HeadHandle(object):
    """A fitted supervised head + its train-only standardization stats."""

    def __init__(self, model, mu, sd, classes, train_classes, device):
        self.model = model
        self.mu = mu
        self.sd = sd
        self.classes = list(classes)
        self.idx_of = {int(c): i for i, c in enumerate(self.classes)}
        self.train_classes = set(int(c) for c in train_classes)
        self.device = device


def fit_head(z_tr, y_tr, classes, *, seed, hidden, epochs, lr, wd, dropout,
             batch_size, device="cpu"):
    """Train a supervised head on (z_tr, y_tr) with global class ids in
    `classes` (output unit per class). Standardization stats are fit on
    the TRAIN rows only (fit on anything else would leak eval rows into
    the probe). Returns a HeadHandle; deterministic under `seed`."""
    z_tr = np.asarray(z_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    mu = z_tr.mean(axis=0)
    sd = z_tr.std(axis=0) + 1e-8
    z = (z_tr - mu) / sd
    n, d = z.shape
    remap = {int(c): i for i, c in enumerate(classes)}
    yt = torch.tensor([remap[int(c)] for c in y_tr], dtype=torch.long,
                      device=device)  # KeyError on unknown class = fail loud
    x = torch.tensor(z, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    C = len(classes)
    if hidden <= 0:
        model = torch.nn.Linear(d, C)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, C))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    bs = min(batch_size, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return HeadHandle(model, mu, sd, classes, np.unique(y_tr), device)


def head_predict(handle, z_te, allowed):
    """Masked-argmax prediction restricted to `allowed` class ids -- the
    native class_mask semantics. Fail closed: every allowed class must be
    a class the head was TRAINED on; silently dropping candidates would
    shrink the choice set and inflate accuracy (ncm_probe precedent)."""
    allowed = list(dict.fromkeys(int(c) for c in allowed))
    missing = [c for c in allowed if c not in handle.train_classes]
    if missing:
        raise RuntimeError(f"allowed classes the head was never trained on: "
                           f"{sorted(missing)}")
    zt = torch.tensor((np.asarray(z_te, dtype=np.float32) - handle.mu)
                      / handle.sd, dtype=torch.float32, device=handle.device)
    with torch.no_grad():
        logits = handle.model(zt).cpu().numpy()
    cols = [handle.idx_of[c] for c in allowed]
    best = logits[:, cols].argmax(axis=1)
    return np.asarray([allowed[i] for i in best], dtype=np.int64)


def _acc(pred, gold):
    return 100.0 * float(np.mean([p == int(g) for p, g in zip(pred, gold)]))


def _hyper(args):
    return {"hidden": args.hidden, "epochs": args.epochs, "lr": args.lr,
            "wd": args.wd, "dropout": args.dropout,
            "batch_size": args.batch_size, "seed_base": args.seed_base,
            "device": args.device}


def probe_boundary(data, masks, t, feat, args):
    """One boundary x feature space. Returns per-(client, task) rows.

    Probe scopes:
      linear-local / mlp-local: trained on THIS (client, task) train rows
        only -- the local-only interface ceiling;
      mlp-pooled: ONE head per (boundary, feat) trained on the pooled
        train rows of ALL clients' seen tasks -- the cross-client
        information-content oracle (diagnostic only, never a method).
    """
    z = data[feat]
    labels = data["labels"]
    cids = data["client_ids"]
    tids = data["task_ids"]
    tr = data["splits"] == 0
    te = data["splits"] == 1

    hp = dict(hidden=args.hidden, epochs=args.epochs, lr=args.lr,
              wd=args.wd, dropout=args.dropout, batch_size=args.batch_size,
              device=args.device)

    pooled_classes = sorted(int(c) for c in np.unique(labels[tr]))
    pooled = fit_head(z[tr], labels[tr], pooled_classes,
                      seed=stable_seed(args.seed_base, "pooled", t, feat),
                      **hp)

    rows = []
    for cid in sorted(masks):
        for k in range(t + 1):
            sel_tr = tr & (cids == cid) & (tids == k)
            sel_te = te & (cids == cid) & (tids == k)
            if not sel_te.any():
                continue
            allowed = masks[cid][k]
            zt = z[sel_te]
            yt = labels[sel_te]
            # local heads train on this (client, task) train rows only;
            # fit_head's remap fails loud if a train label is outside the
            # mask, head_predict fails loud if a mask class has no train
            # row -- the same fail-closed coverage as ncm_probe
            lin = fit_head(z[sel_tr], labels[sel_tr], allowed,
                           seed=stable_seed(args.seed_base, "lin", t, feat,
                                            cid, k),
                           hidden=0, **{kk: vv for kk, vv in hp.items()
                                        if kk != "hidden"})
            loc = fit_head(z[sel_tr], labels[sel_tr], allowed,
                           seed=stable_seed(args.seed_base, "mlp", t, feat,
                                            cid, k),
                           **hp)
            rows.append({
                "after_task": t, "test_task": k, "client_id": cid,
                "n_train": int(sel_tr.sum()), "n_test": int(sel_te.sum()),
                "feat": feat,
                "acc_linear_local": round(_acc(head_predict(lin, zt,
                                                            allowed), yt), 2),
                "acc_mlp_local": round(_acc(head_predict(loc, zt,
                                                         allowed), yt), 2),
                "acc_mlp_pooled": round(_acc(head_predict(pooled, zt,
                                                          allowed), yt), 2),
            })
    return rows


def aggregate(rows, t, feat, native):
    """Client-mean per task, then AA over seen tasks -- identical
    aggregation to ncm_probe and the run's accuracy_matrix."""
    sub = [r for r in rows
           if r["after_task"] == t and r["feat"] == feat]
    per_task = {}
    for r in sub:
        per_task.setdefault(r["test_task"], []).append(r)
    aas = {"aa_native": round(float(np.mean([
        np.mean([native[(r["client_id"], t, r["test_task"])]
                 for r in ts]) for ts in per_task.values()])), 2)}
    for key in ("acc_linear_local", "acc_mlp_local", "acc_mlp_pooled"):
        aas["aa_" + key[len("acc_"):]] = round(float(np.mean([
            np.mean([r[key] for r in ts]) for ts in per_task.values()])), 2)
    return aas


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    feat_dir = Path(args.features_dir)
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    c2p = ([float(x) for x in args.c2p.split(",")]
           if args.c2p else [None] * len(ckpts))
    if args.c2p and len(c2p) != len(ckpts):
        raise SystemExit(f"--c2p has {len(c2p)} values but --checkpoints "
                         f"has {len(ckpts)} -- one reference AA per "
                         f"checkpoint, same order")

    native = load_native(run_dir)
    rows = []
    summaries = []

    for ckpt, ref in zip(ckpts, c2p):
        m = re.fullmatch(r"task_(\d+)_end\.pth", Path(ckpt).name)
        if not m:
            raise SystemExit(f"--checkpoints only accepts task_XX_end.pth, "
                             f"got {ckpt!r}")
        t = int(m.group(1))
        npz_path, meta_path = ensure_features(
            run_dir, ckpt, feat_dir, args.data_path, args.device
            if args.device != "cpu" else None, args.prompt_task_mode)
        data = np.load(npz_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        masks = {int(cid): [list(map(int, ts)) for ts in tmask]
                 for cid, tmask in meta["client_class_masks"].items()}

        for feat in ("z_ref", "z_op"):
            rows += probe_boundary(data, masks, t, feat, args)
            aas = aggregate(rows, t, feat, native)
            entry = {"after_task": t, "feat": feat, "c2p_ref": ref, **aas}
            entry["delta_local"] = round(entry["aa_mlp_local"]
                                         - entry["aa_native"], 2)
            entry["delta_pooled"] = round(entry["aa_mlp_pooled"]
                                          - entry["aa_native"], 2)
            entry["federated_premise"] = round(entry["aa_mlp_pooled"]
                                               - entry["aa_mlp_local"], 2)
            if t == 0:
                # pre-registered gate: T0 row only, z_op is the primary
                # reading (later boundaries / z_ref are context only)
                dp = entry["delta_pooled"]
                if dp >= args.gate_green:
                    verdict = "GREEN"
                elif dp <= args.gate_red:
                    verdict = "RED"
                else:
                    verdict = "YELLOW"
                entry["gate"] = {
                    "verdict": verdict, "delta_pooled": dp,
                    "green_at": args.gate_green, "red_at": args.gate_red,
                    "primary": "z_op",
                }
            summaries.append(entry)

        # integrity check: aa_native must reproduce the run's own
        # accuracy_matrix row (same aggregation), fail closed on >0.02pp
        matrix = {}
        with open(run_dir / "metrics" / "accuracy_matrix.csv", newline="",
                  encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vals = [float(v) for k, v in row.items()
                        if k.startswith("T") and v not in (None, "")]
                matrix[int(row["after_task"])] = float(np.mean(vals))
        for feat in ("z_ref", "z_op"):
            aa_native = [s["aa_native"] for s in summaries
                         if s["after_task"] == t and s["feat"] == feat][0]
            if t not in matrix:
                raise RuntimeError(f"accuracy_matrix.csv has no row for "
                                   f"after_task {t}")
            if abs(matrix[t] - aa_native) > 0.02:
                raise RuntimeError(
                    f"aa_native from client CSVs ({aa_native:.2f}) diverges "
                    f"from accuracy_matrix.csv row ({matrix[t]:.2f}) at "
                    f"after_task {t} -- join key mismatch, do not trust "
                    f"this probe run")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_csv.with_suffix(".summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"hyper": _hyper(args),
                   "gate": {"green_pp": args.gate_green,
                            "red_pp": args.gate_red,
                            "note": "T0 z_op delta_pooled only; later "
                                    "boundaries and z_ref are context"},
                   "summaries": summaries}, f, indent=2)

    print(f"[headroom-probe] per-(client, task) rows -> {out_csv}")
    print(f"[headroom-probe] AA summary (client-mean per task, then mean "
          f"over seen tasks):")
    hdr = (f"{'after_task':>10} {'feat':>6} {'native':>8} {'linear':>8} "
           f"{'mlp_loc':>8} {'mlp_pool':>9} {'d_loc':>6} {'d_pool':>7} "
           f"{'fed':>6} {'c2p_ref':>8} {'gate':>7}")
    print(hdr)
    for s in summaries:
        gate = s.get("gate", {}).get("verdict", "-")
        ref = s["c2p_ref"] if s["c2p_ref"] is not None else float("nan")
        print(f"{s['after_task']:>10} {s['feat']:>6} {s['aa_native']:>8.2f} "
              f"{s['aa_linear_local']:>8.2f} {s['aa_mlp_local']:>8.2f} "
              f"{s['aa_mlp_pooled']:>9.2f} {s['delta_local']:>6.2f} "
              f"{s['delta_pooled']:>7.2f} {s['federated_premise']:>6.2f} "
              f"{ref:>8.2f} {gate:>7}")


if __name__ == "__main__":
    main()
