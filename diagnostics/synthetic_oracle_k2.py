"""Phase 1 diagnostic: SYNTHETIC Oracle-K2 stress test (sanity check).

*** NOT the real Oracle-K2 experiment ***

This script answers ONLY: "if we ARTIFICIALLY inject two well-separated
modes into the features (z' = z + s*delta*v_c) and hand the model the
TRUE mode labels, can K=2 mode-specific key/anchor slots exploit them?"
It is a synthetic sanity check of the anchor machinery, NOT evidence
about real CIFAR/ImageNet-R multimodality. The real Phase-2 Oracle-K2
must be built from semantic modes discovered on real TRAIN features by
analyze_multimodality.py. Never cite this script's numbers as
"Oracle-K2 proves real multimodality improves FedTA".

Protocol (feature level, on the pre-anchor features extracted by
diagnostics/extract_features.py -- NEVER [z;anchor]):
  1. Controlled mode separation on pre-anchor features:
         z' = z + s * delta * v_c,   s in {-1,+1} (random per sample, seeded),
         v_c = fixed random unit vector per class (seeded).
     The true mode label k = (s+1)/2 is therefore KNOWN (oracle).
  2. FedTA-K1  : feature-level replica of the official Tail_Anchor
                 (1 key/anchor slot per class, Top-1 cosine routing over the
                 whole key bank at train AND test -- no labels for routing).
  3. Oracle-K2 : 2 slots per class, slot(c,k) = 2c+k.
                 training   : uses the TRUE class + TRUE mode (fully oracle);
                 inference  : TRUE mode only; class-conditional candidate
                 scoring over the task's classes (upper bound, not a method).

Both methods share: identical perturbed features, identical loss structure
(official phase-2: CE + 0.2*global_epoch*InfoNCE(global fused protos)
- 0.1*pull), local-only anchor modules (never federated, like official
FedTA), per-round head snapshots, task-aware class masks at eval.

Simplifications vs the official pipeline (documented deviations, applied
EQUALLY to both methods so the K1-vs-K2 comparison stays controlled):
  - global prototypes use simple per-class mean fusion (fuse_protos)
    instead of BGPS;
  - head outputs nb_classes logits (the official 200-slot Chead quirk for
    cifar100 is not replicated);
  - dropout inactive at evaluation (eval mode).

Usage (from repo root, on the server):
  python diagnostics/synthetic_oracle_k2.py \
      --features diagnostics_output/features_cifar100_task_04_end.npz \
      --out_dir diagnostics_output/synthetic_oracle_k2 \
      --deltas 0,0.25,0.5,1.0,2.0
"""

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# feature-level replica of the official Tail_Anchor / Chead structure
# ---------------------------------------------------------------------------

def l2_normalize(x, dim=1, epsilon=1e-12):
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(
        torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm


class CheadReplica(nn.Module):
    """Same structure as Models.classification_head.Chead."""
    def __init__(self, label_num):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(768 * 2, 512), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(512, label_num))

    def forward(self, x):
        return self.head(x)


class AnchorBank(nn.Module):
    """Key + anchor_pool + Chead, with K slots per class (K=1: FedTA-style,
    K=2: oracle mode-specific slots, slot(c,k) = K*c + k)."""
    def __init__(self, nb_classes, k_per_class, dim=768):
        super().__init__()
        self.nb_classes = nb_classes
        self.k_per_class = k_per_class
        self.dim = dim
        n_slots = nb_classes * k_per_class
        self.key = nn.Parameter(torch.randn(n_slots, dim))
        nn.init.uniform_(self.key, -1, 1)
        self.anchor_pool = nn.Parameter(torch.randn(n_slots, dim))
        self.head = CheadReplica(nb_classes)

    # ---- official-style routing (no labels) ----
    def forward_k1(self, z, head=None):
        head = self.head if head is None else head
        zn = l2_normalize(z)
        kn = l2_normalize(self.key)
        sim = zn @ kn.t()
        idx = torch.topk(sim, k=1, dim=1)[1].squeeze(1)
        anchor = self.anchor_pool[idx]
        mixed = torch.cat([z, anchor], dim=1)
        logits = head(mixed)
        pull = (zn * l2_normalize(self.key[idx])).sum() / self.dim
        return logits, mixed, pull

    # ---- oracle routing (true class + true mode) ----
    def forward_oracle_train(self, z, y, mode):
        slot = self.k_per_class * y + mode
        anchor = self.anchor_pool[slot]
        mixed = torch.cat([z, anchor], dim=1)
        logits = self.head(mixed)
        zn = l2_normalize(z)
        pull = (zn * l2_normalize(self.key[slot])).sum() / self.dim
        return logits, mixed, pull

    # ---- oracle-mode inference: class-conditional candidate scoring ----
    def forward_oracle_eval(self, z, mode, cand_classes, head=None):
        head = self.head if head is None else head
        scores = z.new_zeros(len(z), len(cand_classes))
        for j, c in enumerate(cand_classes):
            anchor = self.anchor_pool[self.k_per_class * c + mode]
            mixed = torch.cat([z, anchor.unsqueeze(0).expand(len(z), -1)], dim=1)
            logits = head(mixed)
            scores[:, j] = logits[:, c]
        return scores


# ---------------------------------------------------------------------------
# controlled perturbation z' = z + s * delta * v_c
# ---------------------------------------------------------------------------

def make_class_directions(seed, nb_classes, dim=768):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(nb_classes, dim, generator=g)
    return v / v.norm(dim=1, keepdim=True)


def make_block_signs(seed, client, task, n):
    """s in {-1,+1} per sample; depends only on (seed, client, task, index)
    so every delta/method sees the SAME mode assignment."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + client * 1009 + task * 101 + 7)
    return torch.randint(0, 2, (n,), generator=g) * 2 - 1


# ---------------------------------------------------------------------------
# loss helpers (mirror official Client_DF phase-2 training)
# ---------------------------------------------------------------------------

def mask_logits(logits, task_classes, nb_classes):
    not_mask = np.setdiff1d(np.arange(nb_classes), task_classes)
    idx = torch.tensor(not_mask, dtype=torch.int64, device=logits.device)
    return logits.index_fill(dim=1, index=idx, value=float("-inf"))


def infonce_loss(mixed, y, proto_matrix, proto_classes, is_last, device):
    """Vectorized replica of Client_DF.calculate_infonce
    (cos / 0.2, exp, 1 - log(sum_pos) on the last round)."""
    matched = torch.isin(y, proto_classes)
    if matched.sum() == 0:
        return mixed.new_zeros(())
    f = mixed[matched]
    ym = y[matched]
    l = torch.nn.functional.cosine_similarity(
        f.unsqueeze(1), proto_matrix.unsqueeze(0), dim=2) / 0.2
    exp_l = torch.exp(l)
    pos_exp = exp_l.gather(
        1, (ym.unsqueeze(1) == proto_classes.unsqueeze(0)).float()
        .argmax(dim=1, keepdim=True)).squeeze(1)
    if is_last:
        return (1 - torch.log(pos_exp)).mean()
    return (-torch.log(pos_exp / exp_l.sum(dim=1))).mean()


def margin_stats(logits_masked, y):
    logit_y = logits_masked.gather(1, y.unsqueeze(1)).squeeze(1)
    other = logits_masked.scatter(1, y.unsqueeze(1), float("-inf"))
    m = logit_y - other.max(dim=1)[0]
    return (float(m.mean()), float(m.median()),
            float(torch.quantile(m, 0.10)) if m.numel() else float("nan"))


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def load_blocks(npz_path, device):
    """{client: {task: {train/test: (z, y)}}} + task class masks."""
    data = np.load(npz_path)
    z = torch.from_numpy(data["z_op"]).float()  # pre-anchor prompt features
    y = torch.from_numpy(data["labels"]).long()
    c = torch.from_numpy(data["client_ids"]).long()
    t = torch.from_numpy(data["task_ids"]).long()
    sp = torch.from_numpy(data["splits"]).long()

    blocks, task_classes = {}, {}
    for ci in c.unique().tolist():
        blocks[ci] = {}
        for ti in t[c == ci].unique().tolist():
            sel = (c == ci) & (t == ti)
            classes = sorted(y[sel].unique().tolist())
            task_classes[(ci, ti)] = classes
            for split_name, flag in (("train", 0), ("test", 1)):
                m = sel & (sp == flag)
                blocks[ci][ti] = blocks[ci].get(ti, {})
                blocks[ci][ti][split_name] = (z[m].to(device), y[m].to(device))
    nb_classes = int(y.max().item()) + 1
    return blocks, task_classes, nb_classes


def run_method(method, blocks, task_classes, nb_classes, delta, V, seed,
               args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    k_per_class = 1 if method == "fedta_k1" else 2
    E = args.global_epoch

    clients = [AnchorBank(nb_classes, k_per_class).to(device)
               for _ in blocks.keys()]
    heads_hist = [dict() for _ in clients]   # {task: head state_dict}
    global_protos = None                     # {class: 1536-vector}

    def perturb(z, y, s):
        return z + s.unsqueeze(1).float() * delta * V[y]

    rows = []
    task_ids = sorted(next(iter(blocks.values())).keys())
    rnd = 0
    for t in task_ids:
        for _ in range(E):
            is_last = (rnd + 1) % E == 0
            # ---- local training (per client, current task) ----
            local_protos = []
            for ci, bank in enumerate(clients):
                z, y = blocks[ci][t]["train"]
                s = make_block_signs(seed, ci, t, len(y)).to(device)
                zp = perturb(z, y, s)
                mode = (s + 1) // 2
                task_cls = task_classes[(ci, t)]
                opt = torch.optim.Adam(bank.parameters(), lr=args.lr,
                                       weight_decay=1e-3)
                ce = nn.CrossEntropyLoss()
                n = len(zp)
                for _ in range(args.local_epoch):
                    perm = torch.randperm(n, device=device)
                    for b in range(0, n, args.batch_size):
                        idx = perm[b:b + args.batch_size]
                        zb, yb, mb = zp[idx], y[idx], mode[idx]
                        if method == "fedta_k1":
                            logits, mixed, pull = bank.forward_k1(zb)
                        else:
                            logits, mixed, pull = bank.forward_oracle_train(zb, yb, mb)
                        logits = mask_logits(logits, task_cls, nb_classes)
                        if global_protos is not None:
                            proto_classes = torch.tensor(
                                sorted(global_protos.keys()), device=device)
                            proto_matrix = torch.stack(
                                [global_protos[int(k)] for k in proto_classes])
                            loss_nce = infonce_loss(
                                mixed, yb, proto_matrix, proto_classes,
                                is_last, device)
                        else:
                            loss_nce = 0.0
                        loss = (ce(logits, yb) + 0.2 * E * loss_nce
                                - 0.1 * pull)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                # head snapshot (official saves heads[task] every round)
                heads_hist[ci][t] = deepcopy(bank.head.state_dict())
                # local mixed protos on train data (class means)
                with torch.no_grad():
                    bank.eval()
                    if method == "fedta_k1":
                        _, mixed, _ = bank.forward_k1(zp)
                    else:
                        _, mixed, _ = bank.forward_oracle_train(zp, y, mode)
                    bank.train()
                    lp = {}
                    for cls in task_cls:
                        m = y == cls
                        if m.any():
                            lp[cls] = mixed[m].mean(0)
                    local_protos.append(lp)

            # ---- fuse global protos (simple mean; BGPS not replicated) ----
            fused = {}
            for lp in local_protos:
                for cls, vec in lp.items():
                    fused.setdefault(cls, []).append(vec)
            global_protos = {cls: torch.stack(v).mean(0)
                             for cls, v in fused.items()}

            # ---- evaluate all seen tasks (RNG-safe recording only) ----
            for ci, bank in enumerate(clients):
                bank.eval()
                for k in range(t + 1):
                    if k not in heads_hist[ci] or k not in blocks[ci]:
                        continue
                    zte, yte = blocks[ci][k]["test"]
                    if len(yte) == 0:
                        continue
                    ste = make_block_signs(seed, ci, k, len(yte)).to(device)
                    zpe = perturb(zte, yte, ste)
                    mode_e = (ste + 1) // 2
                    task_cls = task_classes[(ci, k)]
                    with torch.no_grad():
                        eval_head = CheadReplica(nb_classes).to(device)
                        eval_head.load_state_dict(heads_hist[ci][k])
                        eval_head.eval()
                        if method == "fedta_k1":
                            logits, _, _ = bank.forward_k1(zpe, head=eval_head)
                            logits = mask_logits(logits, task_cls, nb_classes)
                            pred = logits.argmax(dim=1)
                            mm, md, p10 = margin_stats(logits, yte)
                        else:
                            scores = bank.forward_oracle_eval(
                                zpe, mode_e, task_cls, head=eval_head)
                            pos = torch.tensor(task_cls, device=device)
                            pred = pos[scores.argmax(dim=1)]
                            y_pos = (yte.unsqueeze(1) == pos.unsqueeze(0)).float()
                            score_y = (scores * y_pos).sum(1)
                            max_other = (scores * (1 - y_pos) - 1e12 * (1 - y_pos)).max(1)[0]
                            mrg = score_y - max_other
                            mm, md, p10 = (float(mrg.mean()), float(mrg.median()),
                                           float(torch.quantile(mrg, 0.10)))
                    acc = 100.0 * (pred == yte).float().mean().item()
                    rows.append({
                        "delta": delta, "method": method, "client": ci,
                        "round": rnd, "after_task": t, "test_task": k,
                        "accuracy": f"{acc:.2f}",
                        "margin_mean": f"{mm:.4f}",
                        "margin_median": f"{md:.4f}",
                        "margin_p10": f"{p10:.4f}",
                    })
                bank.train()
            rnd += 1
    return rows


def summarize(rows):
    """per (delta, method): task-end metrics (AA / forgetting / BWT /
    retention) averaged over clients."""
    out = []
    key = {}
    for r in rows:
        k = (float(r["delta"]), r["method"])
        key.setdefault(k, []).append(r)
    for (delta, method), rs in sorted(key.items()):
        acc = {(r["client"], int(r["after_task"]), int(r["test_task"])):
               float(r["accuracy"]) for r in rs}
        clients = sorted({r["client"] for r in rs})
        t_end = max(int(r["after_task"]) for r in rs)
        # client-level matrices -> metrics -> mean over clients
        aas, fgs, bwts, rets = [], [], [], []
        for c in clients:
            R = {(a, k): v for (cc, a, k), v in acc.items() if cc == c}
            aa = [np.mean([R[(a, k)] for k in range(a + 1)
                           if (a, k) in R]) for a in range(t_end + 1)
                  if any((a, k) in R for k in range(a + 1))]
            aas.append(np.mean(aa))
            for a in range(1, t_end + 1):
                f, b, rt = [], [], []
                for k in range(a):
                    if (a, k) not in R or (k, k) not in R:
                        continue
                    best = max(R[(l, k)] for l in range(k, a) if (l, k) in R)
                    f.append(max(0.0, best - R[(a, k)]))
                    b.append(R[(a, k)] - R[(k, k)])
                    rt.append(100.0 * R[(a, k)] / R[(k, k)])
                if f:
                    fgs.append(np.mean(f))
                    bwts.append(np.mean(b))
                    rets.append(np.mean(rt))
        out.append({
            "delta": delta, "method": method,
            "final_avg_accuracy": f"{np.mean([float(r['accuracy']) for r in rs if int(r['after_task']) == t_end]):.2f}",
            "avg_accuracy_over_task_ends": f"{np.mean(aas):.2f}",
            "forgetting": f"{np.mean(fgs):.2f}" if fgs else "",
            "bwt": f"{np.mean(bwts):.2f}" if bwts else "",
            "retention": f"{np.mean(rets):.2f}" if rets else "",
        })
    return out


def mode_separation(blocks, V, delta, nb_classes):
    """empirical mean ||mu_mode0 - mu_mode1|| over classes (train split)."""
    sums = {}
    for ci, tasks in blocks.items():
        for t, sp in tasks.items():
            z, y = sp["train"]
            signs = make_block_signs(0, ci, t, len(y)).to(z.device)
            for c in range(nb_classes):
                m = (y == c).cpu()
                if m.sum() == 0:
                    continue
                s = signs[m.to(z.device)]
                zp = z[m.to(z.device)] + s.unsqueeze(1).float() * delta * V[c]
                for mode in (0, 1):
                    mm = s == (mode * 2 - 1)
                    if mm.any():
                        sums.setdefault(c, [0, 0, 0, 0])
                        sums[c][mode] = sums[c][mode] + zp[mm].sum(0)
                        sums[c][2 + mode] += int(mm.sum())
    dists = [(sums[c][0] / max(sums[c][2], 1) -
              sums[c][1] / max(sums[c][3], 1)).norm().item()
             for c in sums]
    return float(np.mean(dists)) if dists else 0.0


def main():
    p = argparse.ArgumentParser(
        "Phase-1 SYNTHETIC Oracle-K2 stress test (sanity check)")
    p.add_argument("--features", required=True,
                   help="features .npz from extract_features.py (needs train+test splits: "
                        "extract with --split both)")
    p.add_argument("--out_dir", default="diagnostics_output/synthetic_oracle_k2")
    p.add_argument("--deltas", default="0,0.25,0.5,1.0,2.0",
                   help="comma-separated mode separations (delta)")
    p.add_argument("--methods", default="fedta_k1,oracle_k2")
    p.add_argument("--global_epoch", default=5, type=int)
    p.add_argument("--local_epoch", default=30, type=int)
    p.add_argument("--batch_size", default=16, type=int,
                   help="official local training batch size")
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    blocks, task_classes, nb_classes = load_blocks(args.features, device)
    V = make_class_directions(args.seed, nb_classes).to(device)
    deltas = [float(d) for d in args.deltas.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, summaries = [], []
    for delta in deltas:
        sep_emp = mode_separation(blocks, V, delta, nb_classes)
        print(f"\n===== delta={delta} (theoretical separation 2*delta="
              f"{2*delta:.2f}, empirical={sep_emp:.2f}) =====")
        for method in methods:
            t0 = time.time()
            rows = run_method(method, blocks, task_classes, nb_classes,
                              delta, V, args.seed, args, device)
            all_rows.extend(rows)
            final = [r for r in rows if int(r["after_task"]) == max(
                int(x["after_task"]) for x in rows)]
            accs = [float(r["accuracy"]) for r in final]
            print(f"[{method}] final mean acc = {np.mean(accs):.2f} "
                  f"({time.time()-t0:.0f}s)")
        for s in summarize([r for r in all_rows if float(r["delta"]) == delta]):
            s["mode_separation_theory"] = f"{2*delta:.2f}"
            s["mode_separation_empirical"] = f"{sep_emp:.2f}"
            summaries.append(s)

    with open(out_dir / "oracle_k2_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(out_dir / "oracle_k2_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    meta = {"experiment": "synthetic_oracle_k2_stress_test",
            "features": str(args.features), "seed": args.seed,
            "deltas": deltas, "methods": methods,
            "global_epoch": args.global_epoch,
            "local_epoch": args.local_epoch,
            "batch_size": args.batch_size, "lr": args.lr,
            "note": "SYNTHETIC modes (z' = z + s*delta*v_c) with oracle mode "
                    "labels; sanity check of the K2 anchor machinery only -- "
                    "NOT evidence about real-data multimodality. The real "
                    "Phase-2 Oracle-K2 must use semantic modes discovered on "
                    "real TRAIN features by analyze_multimodality.py.",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_dir / "oracle_k2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nsaved {out_dir/'oracle_k2_results.csv'}")
    print(f"saved {out_dir/'oracle_k2_summary.csv'}")
    print("\n===== Synthetic Oracle-K2 Stress Test (sanity check; "
          "NOT the real Oracle-K2 on FedTA semantic modes) =====")
    print("\n===== summary =====")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
