"""H1/H2 real-system head retraining probe (Track B1; read-only, 2026-10-28).

Questions (reviewer ruling 2026-10-28, after the offline SSRA probe found
P0 Linear(z_op)=85.00 vs native T0 82.87, a +2.13pp gap that cannot be
attributed to head optimization alone because the probe and native also
differ in INPUT INTERFACE):
  H1: is the native Chead (1536->512->ReLU->Dropout(0.5)->200, trained
      jointly with the whole Tail_Anchor during phase-2, no val
      selection) simply under-trained?  -> freeze everything, retrain
      a FRESH Chead on the real [feat, anchor] inputs with a
      well-optimized protocol (val-selected epoch, test evaluated once).
  H2: does the anchor channel help or hurt?  -> retrain a pure
      Linear(768) on the SAME feat channel, same protocol.

This runs INSIDE the real system path: bootstrap_server rebuilds the
exact run environment (data partition bit-identical via main.build_server
+ args.json), task_00_end.pth restores the end-of-task-0 state, and head
inputs are captured from the REAL forward (original_model -> vit
(task_id=0, train=True, native evaluate semantics) -> Tail_Anchor
routing) via a forward_pre_hook on client.model.head.

Splits (native semantics, restored deterministically by the checkpoint):
  train = client.traindata      (the exact 70% split native trained on)
  test  = client.test_loader[0] (the exact 30% split behind R[0][0])

Arms, identical optimizer/epochs/batch/wd/seed offline protocol:
  native : restored head on cached inputs (calibration vs R[0][0]=82.87,
           plus its TRAIN acc -> underfit vs overfit indicator)
  H1     : fresh Chead(nb_classes), native class-mask eval
  H2     : fresh Linear(768 -> 8-way over the client's task-0 classes)
Aggregation: unweighted mean over 5 clients (native R[0][0] semantics,
verified 82.866). Seeds 42/43/44 (head init + stratified 15% val carve
+ batch order). Known deviation: extraction uses shuffle=False (native
evaluate shuffles batch 8 -- batchwise prompt majority composition
differs slightly; reported via the native-on-cache calibration number).

Interpretation (reviewer-approved):
  H1 ~85, native ~82.9     -> native head under-trained (cheap win:
                              retrain heads inside the real system)
  H1 ~83.x, H2 ~85         -> the anchor channel / Tail_Anchor interface
                              itself hurts discrimination
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.classification_head import Chead

NATIVE_T0_DEFAULT = 82.87


def extract_client(client, device):
    """One pass over the client's task-0 train + native-test splits,
    capturing the REAL head input x1=[feat,anchor] (1536) and feat (768)
    from the native forward. Deterministic (shuffle=False, no_grad)."""
    client.model.load_head(client.heads[0])
    client.model.to(device)
    captured = []

    def hook(_module, inp):
        captured.append(inp[0].detach().cpu())

    h = client.model.head.register_forward_pre_hook(hook)
    cache = {}
    try:
        for split, dataset, bs in (("train", client.traindata, 16),
                                   ("test", client.test_loader[0], 8)):
            loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            x1s, fs, ys = [], [], []
            for input, target in loader:
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.no_grad():
                    out = client.original_model(input)
                    out = out['pre_logits'].requires_grad_(False)
                    out = client.vit(input, task_id=0, cls_features=out,
                                     train=True)
                    client.model(out['feat'].to(device), target)
                x1s.append(captured.pop())
                fs.append(out['feat'].detach().cpu())
                ys.append(target.cpu())
            cache[split] = {
                "x1": torch.cat(x1s).numpy(),
                "feat": torch.cat(fs).numpy(),
                "y": torch.cat(ys).numpy(),
            }
    finally:
        h.remove()
    return cache


def native_head_acc(client, cache, split, nb_classes, device):
    """Restored head on the cached inputs, native masking semantics."""
    head = copy.deepcopy(client.model.head).to(device).eval()
    X = torch.from_numpy(cache[split]["x1"]).float().to(device)
    y = torch.from_numpy(cache[split]["y"]).long().to(device)
    mask = torch.tensor(list(client.class_mask[0]), dtype=torch.int64)
    not_mask = torch.tensor(
        np.setdiff1d(np.arange(nb_classes), mask.numpy()),
        dtype=torch.int64).to(device)
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], 1024):
            logits = head(X[i:i + 1024])
            logits = logits.index_fill(dim=1, index=not_mask,
                                       value=float('-inf'))
            correct += int((logits.argmax(dim=1) == y[i:i + 1024]).sum())
    return 100.0 * correct / X.shape[0]


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
    return ~val_mask, val_mask


def train_head(arm, client, cache, seed, epochs, batch_size, lr,
               weight_decay, nb_classes, device):
    """Offline head retraining on cached REAL inputs.

    H1: fresh Chead (the native head CLASS: 1536->512->ReLU->Dropout .5
        ->nb_classes), trained with global class ids, native masking at
        eval. H2: fresh Linear(768->8-way) on the feat channel.
    Protocol: Adam(lr, wd), batch_size, fixed epochs; stratified val
    carve selects the epoch; TEST is evaluated ONCE at best-val state.
    """
    torch.manual_seed(seed)
    cls_ids = sorted(int(c) for c in client.class_mask[0])
    pos = {c: i for i, c in enumerate(cls_ids)}
    X = cache["train"]
    y = X["y"]
    tr_m, va_m = carve_val(y, seed)

    if arm == "H1":
        model = Chead(nb_classes).to(device)
        def forward(xb):
            return model(xb)
        ytr = torch.from_numpy(y[tr_m]).long().to(device)
        yva = torch.from_numpy(y[va_m]).long().to(device)
        yte = torch.from_numpy(cache["test"]["y"]).long().to(device)
        not_mask = torch.tensor(
            np.setdiff1d(np.arange(nb_classes),
                         np.array(cls_ids)), dtype=torch.int64).to(device)

        def acc(logits, yy):
            lg = logits.index_fill(dim=1, index=not_mask,
                                   value=float('-inf'))
            return float((lg.argmax(dim=1) == yy).float().mean()) * 100
    else:  # H2
        model = torch.nn.Linear(X["feat"].shape[1],
                               len(cls_ids)).to(device)
        def forward(xb):
            return model(xb)
        ytr = torch.from_numpy(
            np.array([pos[int(v)] for v in y[tr_m]])).long().to(device)
        yva = torch.from_numpy(
            np.array([pos[int(v)] for v in y[va_m]])).long().to(device)
        yte = torch.from_numpy(
            np.array([pos[int(v)]
                      for v in cache["test"]["y"]])).long().to(device)

        def acc(logits, yy):
            return float((logits.argmax(dim=1) == yy).float().mean()) * 100

    feats = torch.from_numpy(
        X["x1"] if arm == "H1" else X["feat"]).float().to(device)
    X_te = torch.from_numpy(
        cache["test"]["x1"] if arm == "H1"
        else cache["test"]["feat"]).float().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    src = feats[torch.from_numpy(tr_m).to(device)]
    src_va = feats[torch.from_numpy(va_m).to(device)]
    g = torch.Generator()
    g.manual_seed(seed)

    best_val, best_epoch, best_state = -1.0, -1, None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(src.shape[0], generator=g).to(device)
        for i in range(0, src.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            loss = criterion(forward(src[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                va = acc(forward(src_va), yva)
            if va > best_val:
                best_val, best_epoch = va, epoch
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr_acc = acc(forward(src), ytr)
        te_acc = acc(forward(X_te), yte)
    return {
        "test_acc": round(te_acc, 4),
        "val_acc_at_best": round(best_val, 4),
        "train_acc_at_best": round(tr_acc, 4),
        "best_epoch": best_epoch,
        "generalization_gap": round(tr_acc - te_acc, 4),
        "param_count": sum(p.numel() for p in model.parameters()),
    }


def bootstrap_with_new_arg_defaults(run_dir, checkpoint, device=None):
    """bootstrap_server, plus defaults for args introduced AFTER the A2
    run recorded its args.json (route_fix etc.) -- main.build_server
    passes route_fix unconditionally, and the A2 baseline predates it.
    Diagnostic-only: never touches training behavior."""
    from diagnostics.diag_utils import (load_args_from_run_dir,
                                         resolve_checkpoint)
    from main import build_server
    args = load_args_from_run_dir(run_dir, device=device)
    for key, default in (("route_fix", False), ("route_tau", 0.1),
                         ("route_detach_ce_key_grad", False),
                         ("route_selected_ce_grad", False),
                         ("disable_pull", False)):
        if not hasattr(args, key):
            setattr(args, key, default)
    server = build_server(args, run_manager=None)
    server.init_client()
    server.load_checkpoint(resolve_checkpoint(run_dir, checkpoint))
    return args, server


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True,
                   help="output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42")
    p.add_argument("--checkpoint", default="checkpoints/task_00_end.pth")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--native_t0", type=float, default=NATIVE_T0_DEFAULT)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--out_json",
                   default="diagnostics_output/head_retrain_probe.json")
    p.add_argument("--out_csv",
                   default="diagnostics_output/head_retrain_probe.csv")
    args = p.parse_args()

    aargs, server = bootstrap_with_new_arg_defaults(
        args.run_dir, args.checkpoint, device=args.device)
    device = aargs.device
    nb_classes = aargs.nb_classes
    clients = server.clients
    print("bootstrapped %s | %d clients | nb_classes %d | device %s"
          % (args.run_dir, len(clients), nb_classes, device))

    caches, native = {}, {}
    for c in clients:
        caches[c.id] = extract_client(c, device)
        native[c.id] = {
            "test_acc": round(native_head_acc(c, caches[c.id], "test",
                                              nb_classes, device), 4),
            "train_acc": round(native_head_acc(c, caches[c.id], "train",
                                              nb_classes, device), 4),
        }
        print("client %d: native head on cache: test %.2f (native R[0][0] "
              "reference %.2f), train %.2f"
              % (c.id, native[c.id]["test_acc"], args.native_t0,
                 native[c.id]["train_acc"]))
    nat_test_mean = round(float(np.mean(
        [native[c.id]["test_acc"] for c in clients])), 4)

    results = {}
    agg = {}
    for seed in args.seeds:
        for arm in ("H1", "H2"):
            accs = []
            for c in clients:
                r = train_head(arm, c, caches[c.id], seed, args.epochs,
                               args.batch_size, args.lr, args.weight_decay,
                               nb_classes, device)
                results[(seed, arm, c.id)] = r
                accs.append(r["test_acc"])
            agg[(seed, arm)] = round(float(np.mean(accs)), 4)
            print("seed %d %s: per-client %s | mean %.4f"
                  % (seed, arm, ["%.2f" % a for a in accs],
                     agg[(seed, arm)]))

    h1_mean = round(float(np.mean([agg[(s, "H1")] for s in args.seeds])), 4)
    h2_mean = round(float(np.mean([agg[(s, "H2")] for s in args.seeds])), 4)

    summary = {
        "probe": "head_retrain_H1_H2",
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "protocol": {"seeds": args.seeds, "epochs": args.epochs,
                     "batch_size": args.batch_size, "lr": args.lr,
                     "weight_decay": args.weight_decay,
                     "optimizer": "Adam",
                     "val_frac": 0.15,
                     "selection": "val epoch; test evaluated once",
                     "aggregation": "unweighted mean over clients",
                     "extraction": "real forward, train=True vit "
                                   "semantics, shuffle=False (native "
                                   "evaluate shuffles -- slight "
                                   "batchwise-majority composition "
                                   "difference, reported by native-on-"
                                   "cache calibration)"},
        "native_t0_matrix": args.native_t0,
        "native_on_cache": native,
        "native_on_cache_mean": nat_test_mean,
        "h1_retrained_chead_mean": h1_mean,
        "h2_retrained_linear_feat_mean": h2_mean,
        "h1_minus_native": round(h1_mean - nat_test_mean, 4),
        "h2_minus_native": round(h2_mean - nat_test_mean, 4),
        "h1_minus_h2": round(h1_mean - h2_mean, 4),
        "runs": {"%d_%s_c%d" % k: v for k, v in results.items()},
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "client", "test_acc",
                    "val_acc_at_best", "train_acc_at_best", "best_epoch",
                    "generalization_gap", "param_count"])
        for cid, r in native.items():
            w.writerow(["native", "", cid, r["test_acc"], "",
                        r["train_acc"], "", "",
                        ""])
        for (seed, arm, cid), r in sorted(results.items()):
            w.writerow([arm, seed, cid, r["test_acc"],
                        r["val_acc_at_best"], r["train_acc_at_best"],
                        r["best_epoch"], r["generalization_gap"],
                        r["param_count"]])
    print("\n=== H1/H2 real-system head retraining probe ===")
    print("native head on cache (calibration): %.4f  [matrix R[0][0] "
          "%.2f]" % (nat_test_mean, args.native_t0))
    print("H1 retrained Chead([feat,anchor]): %.4f" % h1_mean)
    print("H2 retrained Linear(feat)        : %.4f" % h2_mean)
    print("H1-native %.2fpp | H2-native %.2fpp | H1-H2 %.2fpp"
          % (summary["h1_minus_native"], summary["h2_minus_native"],
             summary["h1_minus_h2"]))
    print("wrote %s and %s" % (out_json, args.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
