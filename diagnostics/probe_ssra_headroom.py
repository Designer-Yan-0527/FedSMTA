"""A5 Gate: SSRA headroom probe v2 (offline, read-only; 2026-10-28).

v2 (external review fix): epoch selection by a held-out VALIDATION split
carved from inner-train; TEST IS EVALUATED EXACTLY ONCE at the best-val
checkpoint. v1 selected the epoch by test accuracy = test leakage, and P1
(+0.2M params) could win the gate by fitting test fluctuations.

Question (pre-registered, Roadmap A5-P2 / NCM probe verdict "A5 共享
语义层必须可训练"):
  Does the minimal SSRA function class -- zero-init residual adapter
  A(z) = W2 GELU(W1 LN(z)), 768->128->768, ~0.2M params -- extract
  >= 2pp of task-in supervised classification headroom from the FIXED
  A2 T0 representation (z_op), above a trained linear head on the same
  representation?

Scope: "task-in supervised headroom probe on the FIXED A2 T0
representation" -- a screening gate, NOT a mathematical upper bound of
full SSRA (federation / cross-task init / contrastive supervision are
not simulated). Cache provenance verified: inr_a2_key_p40 seed_42,
method=fedta_a2, a2_arm=key, route_fix absent (= off) -- the SSRA
proposal requires the route_fix=off A2 baseline, and this cache is it.

Protocol per (client, task 0), 8-way over the client's own class mask
(native R[0][0] semantics):
  inner-train -> stratified 15% VALIDATION carve (per seed, identical
                 across arms) / remaining TRAIN
  inner-test  -> TEST, evaluated ONCE at best-val epoch
  Arms (identical optimizer/epochs/batch/wd/seed/split):
    P0 Linear(z_op)                          calibration arm
    P1 Linear(z_op + A(z_op))  A zero-init    SSRA function class
    P2 Linear(MLP(z_op))       no residual    attachment: capacity vs
                                              residual-identity value
  Aggregation: per-client acc -> unweighted mean over 5 clients
  (native T0 82.87 aggregation; verified 82.866). Seeds 42/43/44.

Diagnostics beyond accuracy (external review additions):
  per-client accs with std/min/max + clients-with-gain count
  (a shared federated interface should lift MOST clients, not one);
  Fisher ratio (between/within class scatter) on train features pre-
  vs post-transform (is the gain discriminability, not just capacity);
  train acc at best-val epoch + generalization gap.

Gate (reviewer-approved):
  calibration |mean(P0) - native_T0| <= 0.5pp else CALIBRATION_FAIL
  (protocol misaligned -> fix protocol; NOT an SSRA verdict)
  delta = mean over seeds of (P1_agg - P0_agg)
  GREEN >= 2.0 | YELLOW 1.0-2.0 | RED < 1.0
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

NATIVE_T0_DEFAULT = 82.87  # inr_a2_key_p40 seed_42 accuracy_matrix R[0][0]
REQUIRED_KEYS = {"z_op", "labels", "client_ids", "task_ids", "splits"}


class ResidualAdapter(torch.nn.Module):
    """SSRA function class: zero-init W2 => z_ssra == z at step 0."""

    def __init__(self, dim=768, bottleneck=128):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.down = torch.nn.Linear(dim, bottleneck)
        self.act = torch.nn.GELU()
        self.up = torch.nn.Linear(bottleneck, dim)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class PlainMLP(torch.nn.Module):
    """P2 attachment: same transform WITHOUT the residual identity."""

    def __init__(self, dim=768, bottleneck=128):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.down = torch.nn.Linear(dim, bottleneck)
        self.act = torch.nn.GELU()
        self.up = torch.nn.Linear(bottleneck, dim)

    def forward(self, x):
        return self.up(self.act(self.down(self.norm(x))))


def build_arm(arm, dim, n_classes):
    head = torch.nn.Linear(dim, n_classes)
    params = sum(p.numel() for p in head.parameters())
    if arm == "P0":
        return None, head, params
    feat = (ResidualAdapter(dim) if arm == "P1" else PlainMLP(dim))
    params += sum(p.numel() for p in feat.parameters())
    return feat, head, params


def fisher_ratio(X, y):
    """between-class / within-class scatter on raw features (diagnostic)."""
    mu = X.mean(dim=0)
    between, within = 0.0, 0.0
    for c in torch.unique(y):
        Xc = X[y == c]
        mu_c = Xc.mean(dim=0)
        between += Xc.shape[0] * float(((mu_c - mu) ** 2).sum())
        within += float(((Xc - mu_c) ** 2).sum())
    return between / max(within, 1e-12)


def _eval_acc(feat, head, X, y):
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], 1024):
            xb = X[i:i + 1024]
            h = feat(xb) if feat is not None else xb
            correct += int((head(h).argmax(dim=1) == y[i:i + 1024]).sum())
    return 100.0 * correct / X.shape[0]


def train_one(arm, X_tr, y_tr, X_val, y_val, X_te, y_te, seed,
              epochs, batch_size, lr, weight_decay, device, fisher_in):
    """One (client, seed, arm) run. VALIDATION selects the epoch; TEST is
    evaluated exactly once at the best-val state (no test leakage)."""
    torch.manual_seed(seed)
    dim = X_tr.shape[1]
    n_classes = int(max(y_tr.max(), y_val.max(), y_te.max())) + 1
    feat, head, n_params = build_arm(arm, dim, n_classes)
    feat = feat.to(device) if feat is not None else None
    head = head.to(device)
    mods = [m for m in (feat, head) if m is not None]
    opt = torch.optim.Adam(
        [p for m in mods for p in m.parameters()], lr=lr,
        weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    Xs = {n: torch.from_numpy(a).float().to(device)
          for n, a in (("tr", X_tr), ("val", X_val), ("te", X_te))}
    ys = {n: torch.from_numpy(a).long().to(device)
          for n, a in (("tr", y_tr), ("val", y_val), ("te", y_te))}
    n = Xs["tr"].shape[0]
    g = torch.Generator()
    g.manual_seed(seed)

    best_val, best_epoch, best_state = -1.0, -1, None
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            h = feat(Xs["tr"][idx]) if feat is not None else Xs["tr"][idx]
            loss = criterion(head(h), ys["tr"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val = _eval_acc(feat, head, Xs["val"], ys["val"])
            if val > best_val:
                best_val, best_epoch = val, epoch
                best_state = {
                    "feat": (copy.deepcopy(feat.state_dict())
                             if feat is not None else None),
                    "head": copy.deepcopy(head.state_dict())}
    if best_state is None:  # epochs < 5 edge
        best_epoch, best_state = epochs - 1, {
            "feat": (copy.deepcopy(feat.state_dict())
                     if feat is not None else None),
            "head": copy.deepcopy(head.state_dict())}
    if feat is not None:
        feat.load_state_dict(best_state["feat"])
    head.load_state_dict(best_state["head"])

    tr = _eval_acc(feat, head, Xs["tr"], ys["tr"])
    te = _eval_acc(feat, head, Xs["te"], ys["te"])  # single eval
    fisher_out = fisher_in  # P0: feature unchanged
    if feat is not None:
        with torch.no_grad():
            fisher_out = fisher_ratio(feat(Xs["tr"]), ys["tr"])
    return {
        "test_acc": round(te, 4),
        "val_acc_at_best": round(best_val, 4),
        "train_acc_at_best": round(tr, 4),
        "best_epoch": best_epoch,
        "generalization_gap": round(tr - te, 4),
        "param_count": n_params,
        "fisher_input": round(fisher_in, 4),
        "fisher_transformed": round(float(fisher_out), 4),
    }


def load_cache(npz_path, json_path):
    """Fail-closed cache load + provenance checks."""
    with open(json_path) as f:
        meta = json.load(f)
    d = np.load(npz_path)
    missing = REQUIRED_KEYS - set(d.files)
    if missing:
        raise SystemExit("cache missing keys: %s" % sorted(missing))
    if meta.get("prompt_task_mode") != "current":
        raise SystemExit("probe expects the promptcurrent cache, got %r"
                         % meta.get("prompt_task_mode"))
    z_op = d["z_op"]
    if z_op.shape[1] != 768:
        raise SystemExit("expected 768-d features, got %d" % z_op.shape[1])
    masks = meta["client_class_masks"]
    subsets = {}
    for cid in sorted(int(k) for k in masks):
        t0_classes = sorted(int(c) for c in masks[str(cid)][0])
        sel = ((d["client_ids"] == cid) & (d["task_ids"] == 0))
        for sp, name in ((0, "train"), (1, "test")):
            m = sel & (d["splits"] == sp)
            labels = d["labels"][m]
            if sorted(int(v) for v in np.unique(labels)) != t0_classes:
                raise SystemExit("client %d split %s labels != task-0 mask"
                                 % (cid, name))
            cls_idx = {c: i for i, c in enumerate(t0_classes)}
            y = np.array([cls_idx[int(v)] for v in labels], dtype=np.int64)
            subsets[(cid, name)] = (z_op[m].astype(np.float32), y)
    return subsets, meta


def carve_val(y, seed, frac=0.15):
    """Stratified validation carve, per (client, seed) -- identical across
    arms for a given seed. Fails closed if a class vanishes from train."""
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val_idx.extend(idx[:max(1, int(round(frac * len(idx))))])
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[val_idx] = True
    if len(np.unique(y[~val_mask])) < len(np.unique(y)):
        raise SystemExit("val carve emptied a class in train")
    return (~val_mask, val_mask)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", required=True,
                   help="features_ImageNet-R_task_00_end_promptcurrent.npz")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--native_t0", type=float, default=NATIVE_T0_DEFAULT)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_json",
                   default="diagnostics_output/ssra_headroom_probe.json")
    p.add_argument("--out_csv",
                   default="diagnostics_output/ssra_headroom_probe.csv")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = "cpu"

    npz = Path(args.features)
    subsets, meta = load_cache(str(npz), str(npz.with_suffix(".json")))
    clients = sorted({cid for (cid, _) in subsets})
    print("cache: %s (run %s, %d clients x task 0, 8-way, val_frac %.2f,"
          " test evaluated ONCE at best-val epoch)"
          % (npz.name, meta.get("run_dir"), len(clients), args.val_frac))

    arms = ["P0", "P1", "P2"]
    results = {}   # (seed, arm, client) -> run dict
    agg = {}       # (seed, arm) -> dict(mean, std, min, max)
    for seed in args.seeds:
        splits = {}
        for cid in clients:
            X, y = subsets[(cid, "train")]
            tr_m, va_m = carve_val(y, seed, args.val_frac)
            splits[cid] = ((X[tr_m], y[tr_m]), (X[va_m], y[va_m]),
                           subsets[(cid, "test")])
        for arm in arms:
            accs, run_rows = [], {}
            for cid in clients:
                (Xtr, ytr), (Xva, yva), (Xte, yte) = splits[cid]
                with torch.no_grad():
                    fin = fisher_ratio(torch.from_numpy(Xtr).float()
                                       .to(device),
                                       torch.from_numpy(ytr).long()
                                       .to(device))
                r = train_one(arm, Xtr, ytr, Xva, yva, Xte, yte, seed,
                              args.epochs, args.batch_size, args.lr,
                              args.weight_decay, device, float(fin))
                results[(seed, arm, cid)] = r
                run_rows[cid] = r
                accs.append(r["test_acc"])
            agg[(seed, arm)] = {
                "mean": round(float(np.mean(accs)), 4),
                "std": round(float(np.std(accs)), 4),
                "min": round(float(np.min(accs)), 4),
                "max": round(float(np.max(accs)), 4)}
            print("seed %d %s: per-client %s | mean %.4f (std %.2f,"
                  " min %.2f, max %.2f)"
                  % (seed, arm, ["%.2f" % a for a in accs],
                     agg[(seed, arm)]["mean"], agg[(seed, arm)]["std"],
                     agg[(seed, arm)]["min"], agg[(seed, arm)]["max"]))

    p0_mean = float(np.mean([agg[(s, "P0")]["mean"] for s in args.seeds]))
    p1_mean = float(np.mean([agg[(s, "P1")]["mean"] for s in args.seeds]))
    p2_mean = float(np.mean([agg[(s, "P2")]["mean"] for s in args.seeds]))
    calib_delta = round(p0_mean - args.native_t0, 4)
    calibrated = abs(calib_delta) <= 0.5
    delta_per_seed = [round(agg[(s, "P1")]["mean"] - agg[(s, "P0")]["mean"],
                         4) for s in args.seeds]
    delta = round(p1_mean - p0_mean, 4)
    gain_counts = []
    for s in args.seeds:
        gain_counts.append(sum(
            1 for cid in clients
            if results[(s, "P1", cid)]["test_acc"]
            > results[(s, "P0", cid)]["test_acc"]))
    fisher_p1 = round(float(np.mean(
        [results[(s, "P1", c)]["fisher_transformed"] / max(
            results[(s, "P1", c)]["fisher_input"], 1e-12)
         for s in args.seeds for c in clients])), 4)
    fisher_p0 = round(float(np.mean(
        [results[(s, "P0", c)]["fisher_transformed"] / max(
            results[(s, "P0", c)]["fisher_input"], 1e-12)
         for s in args.seeds for c in clients])), 4)

    if not calibrated:
        verdict = "CALIBRATION_FAIL"
        meaning = ("P0 %.2f deviates from native T0 %.2f by %+.2fpp "
                   "(>0.5pp): probe protocol not aligned -- fix protocol "
                   "first; NOT an SSRA verdict"
                   % (p0_mean, args.native_t0, calib_delta))
    elif delta >= 2.0:
        verdict = "GREEN"
        meaning = ("SSRA function class extracts >=2pp headroom from the "
                   "fixed A2 T0 representation -> fedta_a2_ssra gets the "
                   "engineering green light (own commit + review cycle)")
    elif delta >= 1.0:
        verdict = "YELLOW"
        meaning = ("headroom exists but below the 2pp bar for the federated "
                   "engineering cost -> study capacity/regularization first")
    else:
        verdict = "RED"
        meaning = ("the CURRENT minimal SSRA function class shows no "
                   "sufficient direct discriminative headroom on the "
                   "CURRENT A2 T0 representation -> downgrade A5-P2 as "
                   "currently designed (does NOT rule out all shared "
                   "semantic adapters: federation/contrastive supervision "
                   "not simulated)")

    summary = {
        "probe": "ssra_headroom_v2",
        "interpretation_scope": ("task-in supervised headroom probe on the "
                                 "FIXED A2 T0 representation (z_op); "
                                 "screening gate, not a mathematical upper "
                                 "bound of full SSRA"),
        "epoch_selection": "validation (stratified 15% carve); test "
                           "evaluated ONCE at best-val checkpoint",
        "cache": {"npz": str(npz), "run_dir": meta.get("run_dir"),
                  "checkpoint": meta.get("checkpoint"),
                  "provenance": "method fedta_a2 / a2_arm key / "
                                "route_fix off (verified via args.json)"},
        "protocol": {"seeds": args.seeds, "epochs": args.epochs,
                     "batch_size": args.batch_size, "lr": args.lr,
                     "weight_decay": args.weight_decay,
                     "optimizer": "Adam", "val_frac": args.val_frac,
                     "aggregation": "unweighted mean over clients "
                                    "(native R[0][0] semantics)"},
        "native_t0": args.native_t0,
        "p0_linear_mean": round(p0_mean, 4),
        "p1_ssra_mean": round(p1_mean, 4),
        "p2_plain_mlp_mean": round(p2_mean, 4),
        "calibration": {"delta_pp": calib_delta,
                        "within_0_5pp": calibrated},
        "delta_per_seed": delta_per_seed,
        "delta_mean_pp": delta,
        "clients_with_gain_p1_vs_p0_per_seed": gain_counts,
        "fisher_ratio_relative_p0": fisher_p0,
        "fisher_ratio_relative_p1": fisher_p1,
        "verdict": verdict,
        "verdict_meaning": meaning,
        "per_seed_arm_agg": {"%d_%s" % k: v for k, v in agg.items()},
        "runs": {"%d_%s_c%d" % k: v for k, v in results.items()},
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "arm", "client", "test_acc",
                    "val_acc_at_best", "train_acc_at_best", "best_epoch",
                    "generalization_gap", "param_count", "fisher_input",
                    "fisher_transformed"])
        for (seed, arm, cid), r in sorted(results.items()):
            w.writerow([seed, arm, cid, r["test_acc"],
                        r["val_acc_at_best"], r["train_acc_at_best"],
                        r["best_epoch"], r["generalization_gap"],
                        r["param_count"], r["fisher_input"],
                        r["fisher_transformed"]])
        for (seed, arm), v in sorted(agg.items()):
            w.writerow([seed, arm, "AGG_MEAN", v["mean"], "", "", "", "",
                        results[(seed, arm, clients[0])]["param_count"],
                        "", ""])
    print("\n=== A5 Gate: SSRA headroom probe (v2, val-selected) ===")
    print("P0 linear  mean %.4f  [native T0 %.2f, calib %+.2fpp]"
          % (p0_mean, args.native_t0, calib_delta))
    print("P1 SSRA    mean %.4f  -> delta = %+.4fpp" % (p1_mean, delta))
    print("P2 plainMLP mean %.4f (capacity vs residual attachment)"
          % p2_mean)
    print("clients with gain (P1>P0): %s / %d" % (gain_counts, len(clients)))
    print("Fisher ratio (transformed/input): P0 %.3fx, P1 %.3fx"
          % (fisher_p0, fisher_p1))
    print("VERDICT: %s" % verdict)
    print(meaning)
    print("wrote %s and %s" % (out_json, args.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
