"""Phase 4 D1: slot-level collision diagnostic (pre-repair mechanism evidence).

Motivation (FEDSMTA_ROADMAP.md v3 §6.0): the "slot hijack" reading of the
KA channel is currently correlation-level. This diagnostic turns it into
testable mechanism evidence by measuring, for every old-class key/anchor
slot i of every client:

  1. routing hits — how often each SUBSEQUENT task's train-split samples
     select slot i as top-1, replicating Tail_Anchor.forward exactly
     (l2_normalize -> matmul -> topk(k=1)); measured under two boundary
     states per subsequent task k: the state at its START (task_{k-1}_end
     checkpoint) and at its END (task_k_end checkpoint);
  2. parameter displacement — ||key_i(t_end) - key_i(final)||_2 and the
     same for anchor_i. Hijack-mediated vs weight-decay-mediated
     displacement are separated by contrasting slots with hits>0 against
     slots with hits==0 (Adam wd=1e-3 moves even untouched slots);
  3. KA damage — per-class oo - oc joined from the 4-combo CSV
     (analyze_class_retention.py output) by (client, old_task, class).

If hits -> displacement -> damage forms a stable chain, and zero-hit slots
displace less, the routing-collision hypothesis gains mechanism evidence
and A2's intervention (gradient mask vs freeze+wd-exemption) is decided.

Observer-only: no training code touched; RNG snapshot/restored; the only
client-state mutation is vit prompt loading (restored afterwards).

Output CSV: client_id, old_task, class_id, scope, n_test, hits_start,
hits_end, hits_rate_start, n_train_sub, key_disp, anchor_disp, ka_damage.

Usage (repo root, on the server):
  python diagnostics/slot_collision_diagnostic.py \
      --run_dir output/cifar100/fedta/<run>/seed_42 \
      --task_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth,checkpoints/task_04_end.pth \
      --combo_csv diagnostics_output/class_retention_4combo.csv \
      --out_csv diagnostics_output/slot_collision.csv \
      --out_json diagnostics_output/slot_collision.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diag_utils import bootstrap_server
from runtime_utils import (load_checkpoint_file, clone_prompt_template,
                           rng_state_dict, restore_rng_state)


def parse_args():
    p = argparse.ArgumentParser("Phase-4 D1 slot-level collision diagnostic")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--task_ckpts", required=True,
                   help="comma-separated task_end checkpoints IN TASK ORDER "
                        "(task_00_end.pth .. task_04_end.pth), relative to "
                        "run_dir or absolute")
    p.add_argument("--combo_csv", required=True,
                   help="4-combo per-class CSV (analyze_class_retention.py "
                        "output), cwd-relative or absolute")
    p.add_argument("--out_csv", default="diagnostics_output/slot_collision.csv")
    p.add_argument("--out_json", default="diagnostics_output/slot_collision.json")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def resolve(run_dir, ckpt):
    path = Path(ckpt)
    if not path.is_absolute():
        path = Path(run_dir) / ckpt
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return str(path.resolve())


def l2_normalize(x, dim=1, epsilon=1e-12):
    """Exact replica of Tail_Anchor.l2_normalize."""
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(
        square_sum, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm


def get_train_split(client, task_k):
    """data_split_indices keys may be int (fresh) or str (post-checkpoint);
    handle both."""
    dsi = client.data_split_indices
    if task_k in dsi:
        return dsi[task_k]["train"]
    if str(task_k) in dsi:
        return dsi[str(task_k)]["train"]
    raise KeyError(f"data_split_indices missing task {task_k}")


def collect_hits(client, task_k, prompt_module, key_tensor, nb_classes,
                 batch_size):
    """Top-1 slot histogram of task_k's train-split samples under the given
    boundary prompt state and key tensor. Routing math replicates
    Tail_Anchor.forward (l2_normalize -> matmul -> topk(k=1)); features come
    from the same forward path as evaluate() (vit train=True, task_id=k).

    Observer-only; consumes RNG (dropout in the prompt path) — the caller
    fixes the seed beforehand.
    """
    split_idx = get_train_split(client, task_k)
    dataset = Subset(client.train_data[task_k], split_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    key = key_tensor.to(client.device)
    key_norm = l2_normalize(key, dim=1)          # (pool_size, 768)
    # NOTE: the key pool is Tail_Anchor(10, 768, 200) — 200 slots while only
    # nb_classes (100) exist; indices 100-199 are never-trained noise slots
    # (routing into them is itself a meaningful signal).
    hits = torch.zeros(key.shape[0], dtype=torch.long)

    client.vit.load_prompts(prompt_module)
    client.vit.to(client.device)
    for input, _target in loader:
        input = input.to(client.device, non_blocking=True)
        with torch.no_grad():
            output = client.original_model(input)
            output = output['pre_logits'].requires_grad_(False)
            output = client.vit(input, task_id=task_k,
                                cls_features=output, train=True)
            feat = output['feat'].to(client.device)
            x_norm = l2_normalize(feat, dim=1)
            similarity = torch.matmul(x_norm, key_norm.t())
            index = torch.topk(similarity, k=1)[1].squeeze(1)
        hits.scatter_add_(0, index.cpu(),
                          torch.ones_like(index.cpu()))
    return hits


def load_combo_join(combo_csv):
    """(cid, old_task, class) -> {scope, n_test, oo, oc, ka_damage}."""
    join = {}
    with open(combo_csv, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["client_id"]), int(row["old_task"]),
                   int(row["class_id"]))
            entry = join.setdefault(key, {
                "scope": row["scope"], "n_test": int(row["n_test"])})
            combo = (row["prompt"], row["key_anchor"])
            if combo == ("old", "old"):
                entry["oo"] = float(row["acc"])
            elif combo == ("old", "current"):
                entry["oc"] = float(row["acc"])
    for key, entry in join.items():
        if "oo" in entry and "oc" in entry:
            entry["ka_damage"] = entry["oo"] - entry["oc"]
        else:
            entry["ka_damage"] = None
    return join


def spearman(x, y):
    """Spearman rho + p, nan-safe."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan"), float("nan")
    from scipy.stats import spearmanr
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def main():
    args = parse_args()
    ckpt_paths = [resolve(args.run_dir, p) for p in args.task_ckpts.split(",")]
    n_bounds = len(ckpt_paths)

    run_args, server = bootstrap_server(
        args.run_dir, checkpoint=ckpt_paths[-1],
        data_path=args.data_path, device=args.device)
    nb_classes = run_args.nb_classes

    # boundary states: load each task_end checkpoint ONCE
    # bounds[b][cid] = {"prompt_module": None-until-per-client, "key": ...}
    print("loading boundary checkpoints ...", flush=True)
    bounds = []
    for path in ckpt_paths:
        ckpt = load_checkpoint_file(path)
        per_client = {}
        for client in server.clients:
            st = ckpt["clients"][client.id]
            per_client[client.id] = {
                "prompts_state": st["prompts"],
                "key": st["tail_anchor_model"]["key"].detach().cpu().clone(),
                "anchor": st["tail_anchor_model"]["anchor_pool"]
                              .detach().cpu().clone(),
            }
        bounds.append(per_client)
        del ckpt
    final_key = bounds[-1]

    combo_join = load_combo_join(args.combo_csv)

    rng_snapshot = rng_state_dict()
    rows = []
    try:
        for client in server.clients:
            cid = client.id
            cur_prompt = client.prompts  # restored afterwards
            my_tasks = [t for t in range(len(client.train_data))
                        if client.heads[t] is not None]

            # per-boundary prompt modules (structure cloned once)
            prompt_mods = {}
            for b in range(n_bounds):
                pm = clone_prompt_template(client.vit)
                pm.load_state_dict(bounds[b][cid]["prompts_state"])
                prompt_mods[b] = pm

            # hits[k][b] for subsequent tasks k (b in {k-1 start, k end})
            hits = {}
            n_train = {}
            for k in my_tasks:
                if k == 0:
                    continue
                n_train[k] = len(get_train_split(client, k))
                for b in (k - 1, k):
                    torch.manual_seed(9700 + 100 * cid + 10 * k + b)
                    hits[(k, b)] = collect_hits(
                        client, k, prompt_mods[b], bounds[b][cid]["key"],
                        nb_classes, args.batch_size)
                    print(f"client {cid} task {k}: boundary {b} "
                          f"({Path(ckpt_paths[b]).name}) done", flush=True)

            # routing into never-trained noise slots (index >= nb_classes)
            # — direct signal of routing corruption under boundary states
            noise_hits = 0
            total_hits = 0
            for k in [kk for kk in my_tasks if kk > 0]:
                for b in (k - 1, k):
                    noise_hits += int(hits[(k, b)][nb_classes:].sum())
                    total_hits += int(hits[(k, b)].sum())
            print(f"client {cid}: noise-slot routing share = "
                  f"{noise_hits}/{total_hits} "
                  f"({100.0 * noise_hits / max(total_hits, 1):.1f}%)",
                  flush=True)

            # per (old_task, class): aggregated hits + displacement + damage
            pool_size = bounds[0][cid]["key"].shape[0]
            for t in my_tasks:
                if t >= n_bounds - 1:
                    continue
                sub_tasks = [k for k in my_tasks if k > t]
                n_train_sub = sum(n_train.get(k, 0) for k in sub_tasks)
                hits_start = torch.zeros(pool_size, dtype=torch.long)
                hits_end = torch.zeros(pool_size, dtype=torch.long)
                for k in sub_tasks:
                    hits_start += hits[(k, k - 1)]
                    hits_end += hits[(k, k)]
                key_t = bounds[t][cid]["key"]
                anchor_t = bounds[t][cid]["anchor"]
                for c in client.class_mask[t]:
                    entry = combo_join.get((cid, t, int(c)))
                    if entry is None or entry["ka_damage"] is None:
                        continue
                    key_disp = float(torch.linalg.vector_norm(
                        key_t[c] - final_key[cid]["key"][c]))
                    anchor_disp = float(torch.linalg.vector_norm(
                        anchor_t[c] - final_key[cid]["anchor"][c]))
                    hs = int(hits_start[c])
                    he = int(hits_end[c])
                    rows.append({
                        "client_id": cid,
                        "old_task": t,
                        "class_id": int(c),
                        "scope": entry["scope"],
                        "n_test": entry["n_test"],
                        "hits_start": hs,
                        "hits_end": he,
                        "hits_rate_start": (hs / n_train_sub
                                           if n_train_sub else float("nan")),
                        "n_train_sub": n_train_sub,
                        "key_disp": key_disp,
                        "anchor_disp": anchor_disp,
                        "ka_damage": entry["ka_damage"],
                    })

            # restore current prompt module (cleanliness)
            if cur_prompt is not None:
                client.vit.load_prompts(cur_prompt)
    finally:
        restore_rng_state(rng_snapshot)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_csv}")

    # ---------- analysis: hits -> displacement -> damage ----------
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"class-units = {len(df)}")

    print("\n===== chain test: hits vs key_disp (expect >0 if hijack) =====")
    for col in ["hits_start", "hits_end"]:
        for scope in ["all", "public", "private"]:
            sub = df if scope == "all" else df[df.scope == scope]
            r, p = spearman(sub[col], sub["key_disp"])
            print(f"{col:11s} {scope:8s}: rho={r:+.3f}  p={p:.4f}")

    print("\n===== chain test: displacement vs ka_damage =====")
    for col in ["hits_start", "key_disp", "anchor_disp"]:
        for scope in ["all", "public", "private"]:
            sub = df if scope == "all" else df[df.scope == scope]
            r, p = spearman(sub[col], sub["ka_damage"])
            print(f"{col:11s} {scope:8s}: rho={r:+.3f}  p={p:.4f}")

    print("\n===== zero-hit vs hit contrast (wd-mediated vs hijack-mediated) =====")
    for label, sub in [("hits_start==0", df[df.hits_start == 0]),
                       ("hits_start>0 ", df[df.hits_start > 0])]:
        if len(sub) == 0:
            continue
        print(f"{label}: n={len(sub):3d}  key_disp={sub.key_disp.mean():.4f}  "
              f"anchor_disp={sub.anchor_disp.mean():.4f}  "
              f"ka_damage={sub.ka_damage.mean():+.2f}")

    print("\n===== per (client, old_task) summary (sample-weighted means) =====")
    for (cid, t), sub in df.groupby(["client_id", "old_task"]):
        w = sub.n_test
        def wmean(col):
            return float((sub[col] * w).sum() / w.sum()) if w.sum() else float("nan")
        print(f"c{cid} t{t}: hits_start={int(sub.hits_start.sum()):5d}  "
              f"key_disp={wmean('key_disp'):.3f}  anchor_disp={wmean('anchor_disp'):.3f}  "
              f"KA_c={wmean('ka_damage'):+.2f}")

    # ---------- verdict hints ----------
    r_hits_key, _ = spearman(df["hits_start"], df["key_disp"])
    r_hits_ka, _ = spearman(df["hits_start"], df["ka_damage"])
    zero = df[df.hits_start == 0]
    hit = df[df.hits_start > 0]
    print("\n===== verdict hints (D1) =====")
    if len(zero) and len(hit):
        zkd, hkd = zero.key_disp.mean(), hit.key_disp.mean()
        print(f"zero-hit key_disp={zkd:.4f} vs hit key_disp={hkd:.4f} "
              f"(ratio {hkd / zkd:.2f}x)")
        print("=> displacement is " + (
            "hijack-dominated (A2: gradient mask on non-current slots)"
            if hkd > 2 * zkd else
            "partly wd-mediated (A2: freeze + wd exemption; mask only if "
            "hit slots displace much more)"))
    print(f"hits->key_disp rho={r_hits_key:+.3f}; hits->ka_damage "
          f"rho={r_hits_ka:+.3f}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    chain = {}
    pairs = [("hits_start", "key_disp"), ("hits_end", "key_disp"),
             ("hits_start", "ka_damage"), ("hits_end", "ka_damage"),
             ("key_disp", "ka_damage"), ("anchor_disp", "ka_damage")]
    for x_col, y_col in pairs:
        for scope in ["all", "public", "private"]:
            sub = df if scope == "all" else df[df.scope == scope]
            r, p = spearman(sub[x_col], sub[y_col])
            chain[f"{x_col}->{y_col}|{scope}"] = {"rho": r, "p": p}
    metrics = {
        "n_units": len(df),
        "chain": chain,
        "zero_hit": ({"n": int(len(zero)),
                      "key_disp": float(zero.key_disp.mean()),
                      "anchor_disp": float(zero.anchor_disp.mean()),
                      "ka_damage": float(zero.ka_damage.mean())}
                     if len(zero) else None),
        "hit": ({"n": int(len(hit)),
                 "key_disp": float(hit.key_disp.mean()),
                 "anchor_disp": float(hit.anchor_disp.mean()),
                 "ka_damage": float(hit.ka_damage.mean())}
                if len(hit) else None),
    }
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
