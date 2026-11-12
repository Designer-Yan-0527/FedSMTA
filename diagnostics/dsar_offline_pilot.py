"""DSAR 离线试点（diagnostics-only，不改任何训练代码）。

在 baseline run（官方协议，无 head refit）的 round checkpoint 上离线评估
两个"状态对齐"结构模块的即时读出修复量：

  raw      : Acc[H_end_j(z-)]        local 训练后冻结读出直接作用于出生表示
  +L_t     : Acc[H_end_j(L(z-))]    低秩 residual adapter（CE, 有标签），
                                     只在当前任务 train split 拟合
  damaged  : Acc[H_end_j(z+)]        server prompt 广播后的表示（损伤量）
  +B_t     : Acc[H_end_j(B(z+))]    label-free ridge 桥 z+ -> z-（GCV 选 λ，
                                     含 identity 候选）+ 低秩残差变体

z- = client 自己的 birth prompts 前向；z+ = 该轮 server_prompt 前向。
(read r, client prompts = 第 r 轮本地训练后状态；server_prompt = 即将
 广播给 r+1 的状态，因此 (z-, z+) 恰为 r+1 广播边界的 (pre, post) 配对。)

阶段：
  --stage extract : GPU 前向，特征缓存 <run_dir>/dsar_pilot/feats_round_%04d.npz
  --stage fit     : CPU 拟合评估（torch CPU；本机 numpy/scipy float64 BLAS 损坏，
                    数值一律 torch），产 cells.csv + summary.json

fail-closed：数据行配对不一致 / heads 缺失 / checkpoint 缺 key → 报错退出。
c0 零漂移对照（KD no-op 发现）：仅报告位移统计，不阻断。

用法（repo 根目录）：
  python diagnostics/dsar_offline_pilot.py --stage extract --rounds 10   # 冒烟
  python diagnostics/dsar_offline_pilot.py --stage extract              # 全量
  python diagnostics/dsar_offline_pilot.py --stage fit
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diag_utils import bootstrap_server

FEAT_DIM = 768
LT_RANKS = (8, 16, 32)
BT_RANKS = (8, 16, 32)
LT_EPOCHS = 200
LT_LR = 1e-3
LT_WD = 1e-4


def parse_args():
    p = argparse.ArgumentParser("DSAR offline pilot")
    p.add_argument("--run_dir", default=(
        "output/cifar100/fedRelink/cifar_base_p15/seed_42"))
    p.add_argument("--stage", default="both",
                   choices=["extract", "fit", "both"])
    p.add_argument("--rounds", default="all",
                   help="'all' 或逗号分隔的 round id，如 '10' 或 '0,5,10'")
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--out_dir", default=None,
                   help="default: <run_dir>/dsar_pilot")
    return p.parse_args()


def resolve_rounds(spec, run_dir):
    ckpt_dir = Path(run_dir) / "checkpoints"
    if spec == "all":
        avail = sorted(int(m.stem.split("_")[1]) for m in
                       ckpt_dir.glob("round_*.pth"))
        return avail
    return [int(x) for x in spec.split(",") if x.strip() != ""]


# ---------------------------------------------------------------- extract

def forward_rows(client, origin_model, base_dataset, indices, device,
                 batch_size, num_workers, seed):
    """一条 split 的 (z, y, sample_idx)：与 extract_features.py 完全一致的
    前向配方（native evaluate 语义：task_id=client.task_id, train=True）。"""
    loader = DataLoader(Subset(base_dataset, indices), batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    zs, ys, idxs = [], [], []
    n_done = 0
    for inp, tgt in loader:
        inp = inp.to(device, non_blocking=True)
        torch.manual_seed(seed + n_done)  # 若 vit 前向含随机性，两次 pass 同 mask
        with torch.no_grad():
            out_ref = origin_model(inp)
            out_op = client.vit(
                inp, task_id=client.task_id,
                cls_features=out_ref["pre_logits"], train=True)
            z = out_op["feat"].float().cpu()
        b = z.shape[0]
        zs.append(z)
        ys.append(tgt.cpu().to(torch.int64))
        idxs.append(torch.tensor(indices[n_done:n_done + b], dtype=torch.int64))
        n_done += b
    if not zs:
        return None
    return (torch.cat(zs), torch.cat(ys), torch.cat(idxs))


def run_extract(args_cli):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_dir = Path(args_cli.run_dir)
    out_dir = Path(args_cli.out_dir) if args_cli.out_dir else run_dir / "dsar_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = resolve_rounds(args_cli.rounds, run_dir)
    if not rounds:
        raise SystemExit(f"no round checkpoints under {run_dir}/checkpoints")

    run_args, server = bootstrap_server(str(run_dir), checkpoint=None,
                                         device=None)
    device = torch.device(run_args.device)
    server.origin_model.to(device)
    server.origin_model.eval()

    # 数据行配对锚（fail-closed）：各 round 的 data_split_indices 必须一致
    split_ref = None
    t0 = time.time()
    for r in rounds:
        ckpt_path = run_dir / "checkpoints" / f"round_{r:04d}.pth"
        server.load_checkpoint(str(ckpt_path))

        # ---- 恢复 vit 的 prompt 装载状态 ----
        def _to_vit_device(prompt_mod, vit):
            dev = next(vit.parameters()).device
            return prompt_mod.to(dev) if next(prompt_mod.parameters()).device != dev \
                else prompt_mod

        zm_all, zp_all = [], []
        y_all, cl_all, tk_all, sp_all, ix_all = [], [], [], [], []
        for client in server.clients:
            seen = [t for t in range(client.task_id + 1)
                    if t in client.data_split_indices]
            cur_split = {t: {k: list(v) for k, v in
                             client.data_split_indices[t].items()}
                         for t in seen}
            if split_ref is None:
                split_ref = {client.id: cur_split}
            else:
                prev = split_ref.setdefault(client.id, cur_split)
                if prev != cur_split:
                    raise SystemExit(
                        f"PAIRING_GATE_FAIL: client {client.id} "
                        f"data_split_indices changed across rounds")

            per_task = {}
            for t in seen:
                base_dataset = client.train_data[t]
                for split_name in ("train", "test"):
                    indices = cur_split[t][split_name]
                    if not indices:
                        continue
                    per_task[(t, split_name)] = forward_rows(
                        client, server.origin_model, base_dataset,
                        indices, device, args_cli.batch_size,
                        run_args.num_workers, seed=1234 + 17 * client.id)

            # pass 1: z-（client 自己的 birth prompts）
            client.vit.load_prompts(_to_vit_device(client.prompts, client.vit))
            client.vit.to(device)
            for (t, split_name), (z, y, ix) in per_task.items():
                zm, _, ix2 = forward_rows(
                    client, server.origin_model, client.train_data[t],
                    cur_split[t][split_name], device, args_cli.batch_size,
                    run_args.num_workers, seed=1234 + 17 * client.id)
                if not torch.equal(ix2, ix):
                    raise SystemExit(f"PAIRING_GATE_FAIL: row manifest "
                                     f"mismatch client {client.id} task {t}")
                per_task[(t, split_name)] = (zm, y, ix)

            # pass 2: z+（该轮 server_prompt）
            if server.prompt is None:
                raise SystemExit("server.prompt is None after load_checkpoint")
            client.vit.load_prompts(_to_vit_device(server.prompt, client.vit))
            for (t, split_name), (zm, y, ix) in per_task.items():
                zp, _, ix2 = forward_rows(
                    client, server.origin_model, client.train_data[t],
                    cur_split[t][split_name], device, args_cli.batch_size,
                    run_args.num_workers, seed=1234 + 17 * client.id)
                if not torch.equal(ix2, ix):
                    raise SystemExit(f"PAIRING_GATE_FAIL: z+ pass row "
                                     f"mismatch client {client.id} task {t}")
                b = zp.shape[0]
                zm_all.append(zm)
                zp_all.append(zp)
                y_all.append(y)
                cl_all.append(torch.full((b,), client.id, dtype=torch.int64))
                tk_all.append(torch.full((b,), t, dtype=torch.int64))
                sp_all.append(torch.full(
                    (b,), 0 if split_name == "train" else 1,
                    dtype=torch.int64))
                ix_all.append(ix)

        npz_path = out_dir / f"feats_round_{r:04d}.npz"
        np.savez_compressed(
            npz_path,
            feats_minus=torch.cat(zm_all).numpy().astype(np.float32),
            feats_plus=torch.cat(zp_all).numpy().astype(np.float32),
            labels=torch.cat(y_all).numpy(),
            client_ids=torch.cat(cl_all).numpy(),
            task_ids=torch.cat(tk_all).numpy(),
            splits=torch.cat(sp_all).numpy(),
            sample_idx=torch.cat(ix_all).numpy(),
        )
        print(f"round {r:4d}: {sum(x.shape[0] for x in zm_all)} rows -> "
              f"{npz_path} ({time.time()-t0:.0f}s)", flush=True)
    print("extract done")


# ---------------------------------------------------------------- fit

def l2n(x, dim=1, eps=1e-12):
    return x * torch.rsqrt(torch.clamp((x * x).sum(dim=dim, keepdim=True),
                                       min=eps))


def ta_logits(z, key, anchor_pool, hsd):
    """Tail_Anchor 前向的 CPU 复刻（drop 项关闭 = eval 语义）。
    hsd: heads[j] 的 state_dict（head.0.weight / head.0.bias /
    head.3.weight / head.3.bias）。"""
    xn = l2n(z)
    kn = l2n(key, dim=1)
    sim = xn @ kn.t()
    _, idx = torch.topk(sim, k=1, dim=1)
    anchor = anchor_pool[idx].reshape(-1, FEAT_DIM)
    x1 = torch.stack((z, anchor), dim=1).reshape(-1, FEAT_DIM * 2)
    h = F.relu(F.linear(x1, hsd["head.0.weight"], hsd["head.0.bias"]))
    return F.linear(h, hsd["head.3.weight"], hsd["head.3.bias"])


def masked_acc(logits, y, allowed):
    """官方口径：class_mask 限定类 argmax。"""
    keep = torch.full((logits.shape[1],), float("-inf"))
    keep[allowed] = 0.0
    pred = (logits + keep).argmax(dim=1)
    return 100.0 * (pred == y).float().mean().item()


def fit_lt(zm_train, y_train, key, anchor_pool, hsd_cur, r, seed):
    """L_t: z + U(V^T z) + b，CE 训练（head/readout 全冻结）。"""
    torch.manual_seed(seed)
    d = zm_train.shape[1]
    U = (torch.randn(d, r) * 0.01).requires_grad_(True)
    V = (torch.randn(d, r) * 0.01).requires_grad_(True)
    b = torch.zeros(d).requires_grad_(True)
    opt = torch.optim.Adam([U, V, b], lr=LT_LR, weight_decay=LT_WD)
    for _ in range(LT_EPOCHS):
        opt.zero_grad()
        z_adj = zm_train + (zm_train @ V) @ U.t() + b
        loss = F.cross_entropy(ta_logits(z_adj, key, anchor_pool, hsd_cur),
                               y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        z_adj = zm_train + (zm_train @ V) @ U.t() + b
        final = F.cross_entropy(
            ta_logits(z_adj, key, anchor_pool, hsd_cur), y_train).item()
    def apply_(z):
        return z + (z @ V) @ U.t() + b
    return apply_, final


def fit_bt(zp_train, zm_train, seed):
    """B_t: label-free ridge z+ -> z-（仿射，含 intercept）。
    GCV 选 λ（相对尺度网格），并与 identity（ΔB=0）按 train MSE 比较。
    返回 (Wz, c) 及 identity 是否胜出；低秩变体由调用方做 SVD 截断。"""
    torch.manual_seed(seed)
    X64 = torch.cat([zp_train.double(),
                     torch.ones(zp_train.shape[0], 1, dtype=torch.float64)],
                    dim=1)
    Y64 = zm_train.double()
    n, dx = X64.shape
    XtX = X64.t() @ X64
    XtY = X64.t() @ Y64
    scale = float(torch.diagonal(XtX).mean())
    best = None  # (gcv, lam, mse)
    for expo in torch.linspace(-3, 4, 29):
        lam = float(10 ** expo) * scale
        reg = torch.eye(dx, dtype=torch.float64) * lam
        reg[-1, -1] = 0.0  # intercept 不正则
        W = torch.linalg.solve(XtX + reg, XtY)
        resid = X64 @ W - Y64
        mse = float((resid * resid).sum() / (n * Y64.shape[1]))
        # GCV（hat 矩阵迹）
        P = torch.linalg.solve(XtX + reg, X64.t())        # (dx, n)
        tr = float(torch.einsum("ij,ji->", X64, P)) / n
        gcv = mse / max(1e-12, (1.0 - tr) ** 2)
        if best is None or gcv < best[0]:
            best = (gcv, lam, mse, W)
    _, lam, mse, W = best
    # identity 候选（label-free 选择）
    mse_id = float(((zp_train.double() - zm_train.double()) ** 2).sum()
                   / (n * zm_train.shape[1]))
    if mse_id <= mse:
        return torch.eye(FEAT_DIM), torch.zeros(FEAT_DIM), True, lam, mse_id
    Wz = W[:FEAT_DIM].float()
    c = W[FEAT_DIM].float()
    return Wz, c, False, lam, mse


def run_fit(args_cli):
    run_dir = Path(args_cli.run_dir)
    out_dir = Path(args_cli.out_dir) if args_cli.out_dir else run_dir / "dsar_pilot"
    rounds = resolve_rounds(args_cli.rounds, run_dir)
    rows = []
    disp_stats = []
    for r in rounds:
        npz_path = out_dir / f"feats_round_{r:04d}.npz"
        ckpt_path = run_dir / "checkpoints" / f"round_{r:04d}.pth"
        if not npz_path.is_file():
            raise SystemExit(f"missing {npz_path} (run --stage extract first)")
        data = np.load(npz_path)
        ck = torch.load(ckpt_path, map_location="cpu")
        cur = r // 5
        for ci, cstate in enumerate(ck["clients"]):
            tam = cstate["tail_anchor_model"]
            key = tam["key"].float()
            anchor_pool = tam["anchor_pool"].float()
            class_mask = cstate["class_mask"]
            heads = cstate["heads"]
            m_cl = torch.tensor(data["client_ids"] == ci)
            m_tr = torch.tensor(data["splits"] == 0)
            m_te = torch.tensor(data["splits"] == 1)
            zm = torch.tensor(data["feats_minus"])
            zp = torch.tensor(data["feats_plus"])
            y = torch.tensor(data["labels"])
            tk = torch.tensor(data["task_ids"])
            seen = sorted(int(t) for t in torch.unique(tk[m_cl]).tolist())
            if any(heads[j] is None for j in seen):
                raise SystemExit(f"round {r} client {ci}: heads None for a "
                                 f"seen task {seen}")
            # 当前任务 train 行：拟合材料
            m_fit = m_cl & m_tr & (tk == cur)
            if m_fit.sum() == 0:
                raise SystemExit(f"round {r} client {ci}: no current-task "
                                 f"train rows")
            zm_fit, zp_fit = zm[m_fit], zp[m_fit]
            y_fit = y[m_fit]
            # 位移统计（c0 零漂移对照，报告用）
            diff = (zp_fit - zm_fit).norm(dim=1)
            disp_stats.append(dict(round=r, client=ci, cur=cur,
                                   disp_mean=float(diff.mean()),
                                   disp_max=float(diff.max())))
            # ---- L_t：每 rank 拟合一次，跨任务共用 ----
            hsd_cur = heads[cur]
            lt = {}
            for r_rank in LT_RANKS:
                lt[r_rank] = fit_lt(zm_fit, y_fit, key, anchor_pool, hsd_cur,
                                   r_rank, seed=1000 + 97 * r + 31 * ci + r_rank)
            # ---- B_t：GCV ridge + identity 候选 ----
            Wz, c, identity_won, lam, mse = fit_bt(zp_fit, zm_fit,
                                                   seed=2000 + 97 * r + 31 * ci)
            dW = Wz - torch.eye(FEAT_DIM)
            U_, S_, Vt_ = torch.linalg.svd(dW, full_matrices=False)
            bt_rank = {}
            for r_rank in BT_RANKS:
                dW_r = (U_[:, :r_rank] * S_[:r_rank]) @ Vt_[:r_rank]
                bt_rank[r_rank] = torch.eye(FEAT_DIM) + dW_r
            # ---- 逐任务评估 ----
            for j in seen:
                m_ev = m_cl & m_te & (tk == j)
                if m_ev.sum() == 0:
                    continue
                zm_ev, zp_ev, y_ev = zm[m_ev], zp[m_ev], y[m_ev]
                allowed = torch.tensor([int(x) for x in class_mask[j]])
                hsd_j = heads[j]
                row = dict(round=r, client=ci, task=j,
                          n_test=int(m_ev.sum()),
                          acc_raw=masked_acc(
                              ta_logits(zm_ev, key, anchor_pool, hsd_j),
                              y_ev, allowed),
                          acc_damaged=masked_acc(
                              ta_logits(zp_ev, key, anchor_pool, hsd_j),
                              y_ev, allowed),
                          bt_identity=int(identity_won), bt_lam=lam,
                          disp_mean=disp_stats[-1]["disp_mean"],
                          disp_max=disp_stats[-1]["disp_max"])
                for r_rank in LT_RANKS:
                    apply_, _ = lt[r_rank]
                    row[f"acc_lt_r{r_rank}"] = masked_acc(
                        ta_logits(apply_(zm_ev), key, anchor_pool, hsd_j),
                        y_ev, allowed)
                row["acc_bt_full"] = masked_acc(
                    ta_logits(zp_ev @ Wz + c, key, anchor_pool, hsd_j),
                    y_ev, allowed)
                for r_rank in BT_RANKS:
                    row[f"acc_bt_r{r_rank}"] = masked_acc(
                        ta_logits(zp_ev @ bt_rank[r_rank] + c,
                                  key, anchor_pool, hsd_j),
                        y_ev, allowed)
                rows.append(row)
        print(f"fit round {r} done ({len(rows)} cells)", flush=True)

    # ---- 输出 ----
    csv_path = out_dir / "cells.csv"
    fields = sorted({k for r_ in rows for k in r_},
                    key=lambda k: (k not in ("round", "client", "task"),
                                   k))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def _mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    arm_cols = (["acc_raw", "acc_damaged", "acc_bt_full", "bt_identity"] +
                [f"acc_lt_r{r_}" for r_ in LT_RANKS] +
                [f"acc_bt_r{r_}" for r_ in BT_RANKS])
    summary = {"n_cells": len(rows),
               "overall": {c: _mean([r_[c] for r_ in rows])
                           for c in arm_cols},
               "old_task_only": {c: _mean([r_[c] for r_ in rows
                                           if r_["task"] < r_["round"] // 5])
                                 for c in arm_cols},
               "current_task_only": {c: _mean([r_[c] for r_ in rows
                                               if r_["task"] == r_["round"] // 5])
                                      for c in arm_cols},
               "displacement": {f"c{ci}": _mean([d["disp_mean"]
                                                 for d in disp_stats
                                                 if d["client"] == ci])
                                for ci in sorted({d["client"] for d in disp_stats})}}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== DSAR pilot summary ({len(rows)} cells) ===")
    for grp in ("overall", "old_task_only", "current_task_only"):
        print(f"\n[{grp}]")
        for c in arm_cols:
            print(f"  {c:16s} {summary[grp][c]:7.2f}")
    print("\n[displacement mean per client]")
    for k, v in summary["displacement"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nsaved {csv_path} and {out_dir/'summary.json'}")


def main():
    args_cli = parse_args()
    if args_cli.stage in ("extract", "both"):
        run_extract(args_cli)
    if args_cli.stage in ("fit", "both"):
        run_fit(args_cli)


if __name__ == "__main__":
    main()
