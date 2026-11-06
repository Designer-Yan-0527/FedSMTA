"""DSAR birth-semantics preview（diagnostics-only，不改任何训练代码）。

在 capture run（cifar_bsab_capture_M2matched，M2'-matched 轨迹）已落盘的
in-run 真配对 artifacts 上，离线预览两个"状态对齐"结构模块的 birth 语义：

  raw      : Acc[H_end(z-)]     冻结联训读出直接作用于出生表示（local 失配量）
  +L_t     : Acc[H_end(L(z-))]  低秩 residual adapter（CE, 有标签），只在当前
                                 任务 train split 拟合，head/readout 全冻结
  ceiling  : Acc[H-(z-)]         npz birth head（M2' refit oracle）作用于 z-
  damaged  : Acc[H_end(z+)]      广播后表示直接进 H_end（广播损伤量）
  +B_t     : Acc[H_end(B(z+))]   label-free ridge 桥 z+ -> z-（GCV 选 λ，含
                                 identity 候选）+ 低秩残差变体

与 dsar_offline_pilot 的关键差别（§34.2.6a 协议发现的修正版语义）：
checkpoint 的 clients[i]['prompts'] = post-broadcast 状态，birth prompts 不可
从任何 checkpoint 恢复；本脚本的 (z-, z+) 全部来自 bsab_capture/*.npz 的
in-run 真配对（z_op_pre / z_op_post），H_end 来自 round checkpoint 的
tail_anchor_model head.head.*（联训头，不受广播覆写），H- 来自 npz 的
head__*（广播边界锚定的 birth refit head，并与 checkpoint heads[task] 做逐位
一致性 gate）。

覆盖范围与限制（PREVIEW，非判决）：
  * 仅当前任务行（capture 协议只采当前任务 train/test split）；
  * M2'-matched 轨迹（A5 RNG 中立：joint 轨迹与官方 run 逐位一致）；
  * 全部 cell 为 LOW_POWER（n_train < 2048，拟合集小）；
  * 仅覆盖已 capture 的轮次（capture run 仍在跑）。

数值：全程 torch CPU（本机 numpy/scipy float64 BLAS 损坏）。
fail-closed：npz 与 checkpoint 的 birth head 不一致 / labels 越出 class_mask /
head 键集异常 -> 报错退出。c0 零漂移对照（KD no-op）：仅报告位移统计，不阻断。

用法（repo 根目录，CPU）：
  python diagnostics/dsar_birth_preview.py
  python diagnostics/dsar_birth_preview.py --rounds 0,5,7
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

FEAT_DIM = 768
LT_RANKS = (8, 16, 32)
BT_RANKS = (8, 16, 32)
LT_EPOCHS = 200
LT_LR = 1e-3
LT_WD = 1e-4
CHEAD_KEYS = {"head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias"}
LOW_POWER_MIN_FIT = 2048  # 与 Client_DF.BSAB_CAPTURE_MIN_FIT 同口径
NPZ_NAME = re.compile(
    r"^bsab_c(\d+)_t(\d+)_r(\d+)_(broadcast|broadcast_no_test)\.npz$")


def parse_args():
    p = argparse.ArgumentParser("DSAR birth-semantics preview")
    p.add_argument("--run_dir", default=(
        "output/cifar100/fedta/cifar_bsab_capture_M2matched/seed_42"))
    p.add_argument("--rounds", default="all",
                   help="'all' 或逗号分隔的 round id，如 '0,5,7'")
    p.add_argument("--out_dir", default=None,
                   help="default: <run_dir>/dsar_birth_preview")
    return p.parse_args()


# ------------------------------------------------- 与 pilot 逐字一致的工具

def l2n(x, dim=1, eps=1e-12):
    return x * torch.rsqrt(torch.clamp((x * x).sum(dim=dim, keepdim=True),
                                       min=eps))


def ta_logits(z, key, anchor_pool, hsd):
    """Tail_Anchor 前向的 CPU 复刻（drop 项关闭 = eval 语义）。
    hsd: Chead 的 state_dict（head.0.* / head.3.*）。
    refit_input=feat 的头锚半区权重恒 0，x1 路径同构，无需特判。"""
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
    best = None  # (gcv, lam, mse, W)
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


# ----------------------------------------------------------------- 主流程

def discover_cells(run_dir):
    """(round, client, task, entry) -> npz path。同 (c,t,r) 出现两个
    entry = fail-closed（协议上每轮每 client 只有一个广播入口）。"""
    cap_dir = run_dir / "bsab_capture"
    cells = {}
    for path in sorted(cap_dir.glob("bsab_c*.npz")):
        m = NPZ_NAME.match(path.name)
        if m is None:
            raise SystemExit(f"unrecognized capture file name: {path}")
        ci, t, r, entry = (int(m.group(1)), int(m.group(2)),
                          int(m.group(3)), m.group(4))
        key = (r, ci)
        if key in cells:
            raise SystemExit(
                f"PAIRING_GATE_FAIL: duplicate capture for round {r} "
                f"client {ci} ({cells[key][0].name} and {path.name})")
        cells[key] = (t, entry, path)
    if not cells:
        raise SystemExit(f"no capture npz under {cap_dir}")
    return cells


def run_preview(args_cli):
    run_dir = Path(args_cli.run_dir)
    out_dir = (Path(args_cli.out_dir) if args_cli.out_dir
               else run_dir / "dsar_birth_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = discover_cells(run_dir)

    rounds_avail = sorted({r for r, _ in cells})
    if args_cli.rounds == "all":
        rounds = rounds_avail
    else:
        want = [int(x) for x in args_cli.rounds.split(",") if x.strip() != ""]
        missing = [r for r in want if r not in rounds_avail]
        if missing:
            raise SystemExit(f"no capture npz for rounds {missing} "
                             f"(available: {rounds_avail})")
        rounds = want

    # run 级协议核验（args.json）：必须是 M2'-matched capture run
    args_json = run_dir / "args.json"
    if args_json.is_file():
        with open(args_json, encoding="utf-8") as f:
            ra = json.load(f)
        ok = (bool(ra.get("head_refit"))
              and str(ra.get("refit_timing", "local")) == "local"
              and bool(ra.get("bsab_pair_capture")))
        if not ok:
            raise SystemExit(
                "PROTOCOL_GATE_FAIL: run_dir is not an M2'-matched "
                "capture run (need head_refit + refit_timing=local + "
                "bsab_pair_capture)")

    rows = []
    skipped_no_ckpt = []
    t0 = time.time()
    for r in rounds:
        ckpt_path = run_dir / "checkpoints" / f"round_{r:04d}.pth"
        if not ckpt_path.is_file():
            # capture run 仍在跑：该轮 npz 已写、checkpoint 尚未保存
            skipped_no_ckpt.append(r)
            continue
        ck = torch.load(ckpt_path, map_location="cpu")
        by_client = {ci: v for (rr, ci), v in cells.items() if rr == r}
        for ci in sorted(by_client):
            t, entry, npz_path = by_client[ci]
            cstate = ck["clients"][ci]
            if int(cstate.get("task_id", -1)) != t:
                raise SystemExit(
                    f"PAIRING_GATE_FAIL: round {r} client {ci} task "
                    f"mismatch npz={t} ckpt={cstate.get('task_id')}")
            tam = cstate["tail_anchor_model"]
            key = tam["key"].float()
            anchor_pool = tam["anchor_pool"].float()
            # H_end = 联训头：tail_anchor_model 的 head.head.*（嵌套键，
            # refit-ON 臂由 _a5_tail_anchor_state 用联训头覆写）
            hsd_end = {k[len("head."):]: v.float() for k, v in tam.items()
                       if k.startswith("head.head.")}
            if set(hsd_end) != CHEAD_KEYS:
                raise SystemExit(
                    f"STRUCT_GATE_FAIL: round {r} client {ci} "
                    f"unexpected tail_anchor head keys: "
                    f"{sorted(k for k in tam if k.startswith('head.'))}")
            class_mask = cstate["class_mask"]
            heads_ckpt = cstate["heads"]

            data = np.load(npz_path)
            t_npz = sorted({int(v) for v in data["task_ids"]})
            if t_npz != [t]:
                raise SystemExit(f"PAIRING_GATE_FAIL: {npz_path.name} "
                                 f"task_ids {t_npz} != {t}")
            zm = torch.tensor(data["z_op_pre"])   # birth 语义 z⁻
            zp = torch.tensor(data["z_op_post"])  # 广播后 z⁺
            y = torch.tensor(data["label"]).to(torch.int64)
            m_tr = torch.tensor(data["split"] == 0)
            m_te = torch.tensor(data["split"] == 1)
            if zm.shape != zp.shape or zm.shape[0] != y.shape[0]:
                raise SystemExit(f"PAIRING_GATE_FAIL: {npz_path.name} "
                                 f"row shape mismatch")
            if int(m_tr.sum()) == 0 or int(m_te.sum()) == 0:
                raise SystemExit(f"PAIRING_GATE_FAIL: {npz_path.name} "
                                 f"empty train or test block")

            # H⁻ = birth refit head（npz head__*），并与 checkpoint heads[t]
            # 做逐位一致性 gate（广播不改 heads；不一致 = 配对断裂）
            hsd_refit = {k[len("head__"):]: torch.tensor(data[k])
                         for k in data.files
                         if k.startswith("head__")}
            if set(hsd_refit) != CHEAD_KEYS:
                raise SystemExit(
                    f"STRUCT_GATE_FAIL: {npz_path.name} unexpected "
                    f"birth head keys: {sorted(hsd_refit)}")
            heads_t = heads_ckpt[t]
            if heads_t is None:
                raise SystemExit(f"STRUCT_GATE_FAIL: round {r} client "
                                 f"{ci} checkpoint heads[{t}] is None")
            for k in sorted(CHEAD_KEYS):
                if not torch.equal(hsd_refit[k], heads_t[k].float()):
                    raise SystemExit(
                        f"PAIRING_GATE_FAIL: round {r} client {ci} birth "
                        f"head mismatch between npz and checkpoint "
                        f"({k}); strict pairing is broken")

            allowed = torch.tensor([int(x) for x in class_mask[t]])
            if not set(torch.unique(y).tolist()) <= set(allowed.tolist()):
                raise SystemExit(f"PAIRING_GATE_FAIL: {npz_path.name} "
                                  f"labels outside class_mask[{t}]")

            zm_fit, zp_fit, y_fit = zm[m_tr], zp[m_tr], y[m_tr]
            n_train = int(m_tr.sum())
            low_power = n_train < LOW_POWER_MIN_FIT

            # 位移统计（c0 零漂移对照，仅报告）
            diff = (zp_fit - zm_fit).norm(dim=1)
            disp_mean, disp_max = float(diff.mean()), float(diff.max())

            # ---- L_t：CE 拟合（H_end 冻结读出）----
            lt = {}
            for r_rank in LT_RANKS:
                lt[r_rank] = fit_lt(
                    zm_fit, y_fit, key, anchor_pool, hsd_end, r_rank,
                    seed=1000 + 97 * r + 31 * ci + r_rank)

            # ---- B_t：label-free ridge + identity 候选 ----
            Wz, c, identity_won, lam, _ = fit_bt(
                zp_fit, zm_fit, seed=2000 + 97 * r + 31 * ci)
            dW = Wz - torch.eye(FEAT_DIM)
            U_, S_, Vt_ = torch.linalg.svd(dW, full_matrices=False)
            bt_rank = {}
            for r_rank in BT_RANKS:
                dW_r = (U_[:, :r_rank] * S_[:r_rank]) @ Vt_[:r_rank]
                bt_rank[r_rank] = torch.eye(FEAT_DIM) + dW_r

            # ---- 测试行四臂评估 ----
            zm_ev, zp_ev, y_ev = zm[m_te], zp[m_te], y[m_te]
            row = dict(round=r, client=ci, task=t, entry=entry,
                       n_train=n_train, n_test=int(m_te.sum()),
                       low_power=int(low_power),
                       acc_raw=masked_acc(
                           ta_logits(zm_ev, key, anchor_pool, hsd_end),
                           y_ev, allowed),
                       acc_ceiling=masked_acc(
                           ta_logits(zm_ev, key, anchor_pool, hsd_refit),
                           y_ev, allowed),
                       acc_damaged=masked_acc(
                           ta_logits(zp_ev, key, anchor_pool, hsd_end),
                           y_ev, allowed),
                       bt_identity=int(identity_won), bt_lam=lam,
                       disp_mean=disp_mean, disp_max=disp_max)
            for r_rank in LT_RANKS:
                apply_, _ = lt[r_rank]
                row[f"acc_lt_r{r_rank}"] = masked_acc(
                    ta_logits(apply_(zm_ev), key, anchor_pool, hsd_end),
                    y_ev, allowed)
            row["acc_bt_full"] = masked_acc(
                ta_logits(zp_ev @ Wz + c, key, anchor_pool, hsd_end),
                y_ev, allowed)
            for r_rank in BT_RANKS:
                row[f"acc_bt_r{r_rank}"] = masked_acc(
                    ta_logits(zp_ev @ bt_rank[r_rank] + c,
                              key, anchor_pool, hsd_end),
                    y_ev, allowed)
            rows.append(row)
        print(f"round {r:4d}: {len(by_client)} clients done "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit(
            f"no usable cells (rounds {rounds} have no round checkpoints "
            f"yet; capture run still in progress?)")

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

    arm_cols = (["acc_raw", "acc_ceiling", "acc_damaged", "acc_bt_full",
                 "bt_identity"] +
                [f"acc_lt_r{r_}" for r_ in LT_RANKS] +
                [f"acc_bt_r{r_}" for r_ in BT_RANKS])
    tasks = sorted({r_["task"] for r_ in rows})
    summary = {
        "n_cells": len(rows),
        "rounds": sorted({r_["round"] for r_ in rows}),
        "skipped_rounds_no_checkpoint": skipped_no_ckpt,
        "low_power_cells": int(sum(r_["low_power"] for r_ in rows)),
        "overall": {c: _mean([r_[c] for r_ in rows]) for c in arm_cols},
        "per_task": {f"task_{t}": {c: _mean([r_[c] for r_ in rows
                                             if r_["task"] == t])
                                    for c in arm_cols} for t in tasks},
        "gaps": {
            "headroom_ceiling_minus_raw":
                _mean([r_["acc_ceiling"] - r_["acc_raw"] for r_ in rows]),
            "lt_r8_minus_raw":
                _mean([r_["acc_lt_r8"] - r_["acc_raw"] for r_ in rows]),
            "lt_r8_vs_ceiling_share":
                _mean([(r_["acc_lt_r8"] - r_["acc_raw"]) /
                       max(1e-9, r_["acc_ceiling"] - r_["acc_raw"])
                       if r_["acc_ceiling"] > r_["acc_raw"] else float("nan")
                       for r_ in rows]),
            "broadcast_damage_raw_minus_damaged":
                _mean([r_["acc_raw"] - r_["acc_damaged"] for r_ in rows]),
            "bt_full_minus_damaged":
                _mean([r_["acc_bt_full"] - r_["acc_damaged"] for r_ in rows]),
        },
        "displacement": {f"c{ci}": _mean([r_["disp_mean"] for r_ in rows
                                         if r_["client"] == ci])
                         for ci in sorted({r_["client"] for r_ in rows})},
        "notes": ("PREVIEW, not verdict: birth semantics from in-run "
                  "capture pairs (z_op_pre/z_op_post); M2'-matched "
                  "trajectory (joint trajectory bit-identical to official "
                  "per A5 RNG-neutrality); current-task rows only; "
                  "LOW_POWER (n_train<2048) on all cells; partial rounds "
                  "(capture run in progress); CPU torch per "
                  "ENVIRONMENT_AUDIT float64 BLAS breakage."),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== DSAR birth preview ({len(rows)} cells, "
          f"rounds {summary['rounds']}) ===")
    if skipped_no_ckpt:
        print(f"[skipped rounds without checkpoints: {skipped_no_ckpt}]")
    print("\n[overall]")
    for c in arm_cols:
        print(f"  {c:18s} {summary['overall'][c]:7.2f}")
    for t in tasks:
        print(f"\n[task_{t}]")
        for c in arm_cols:
            print(f"  {c:18s} {summary['per_task'][f'task_{t}'][c]:7.2f}")
    print("\n[gaps]")
    for k, v in summary["gaps"].items():
        print(f"  {k:34s} {v:7.2f}")
    print("\n[displacement mean per client] (c0 = KD no-op zero-drift)")
    for k, v in summary["displacement"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nsaved {csv_path} and {out_dir/'summary.json'}")


def main():
    run_preview(parse_args())


if __name__ == "__main__":
    main()
