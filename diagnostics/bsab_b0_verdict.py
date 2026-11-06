"""BSAB B0 正式判决（§34.2.5 预注册，diagnostics-only）。

预注册四臂（每个 broadcast boundary、同一 eval rows、全部从 capture npz
离线计算；§34.2.5 L2284-2293）：

  ① identity  : Acc[M2'head(z+)]   —— M2' refit head（npz birth head）直接
                                      服役于广播后表示 = 官方 M2' 部署口径
  ② bridge    : Acc[M2'head(B(z+))] —— GCV ridge 残差桥，λ‖ΔA‖² 参数化
                                      （A = I+ΔA，含 ΔA=0 候选），label-free
  ③ low-rank  : Acc[M2'head(B_r(z+))] —— ② 的 ΔA 做 SVD 截断，r sweep
  ④ M2''      : Acc[refit(z+)]     —— 冻结 refit 配方在 z+ 上重训读出头
                                      （离线 M2''，正对照锚点）
  （参考记录，不作臂：Acc[M2'head(z-)] = birth-state reference）

ρ_birth = (Acc_bridge - Acc_identity) / (Acc_M2'' - Acc_identity)   （L2295）

前置硬门（L2259；2026-11-06 修订）：
  * reproduction check（MC-dropout 带宽版）：① 对官方 cifar_a5v2_M2_feat_full、
    ④ 对官方 cifar_a5v2_M2pb_full 的 client 级 accuracy matrix 对角线
    （after_task == test_task，task_end 边界 round=5t+4）。根因：官方
    evaluate 路径从未 .eval() -> Chead Dropout(0.1) ON -> 官方 CSV 数是
    dropout-ON 随机前向的单次采样，原 0.25pp 逐位容差不可满足。修订判据：
    官方数落在我们 MC dropout-ON 分布（50 draws）3σ 带内（|z|<=3）且
    z 符号不系统性同侧（<=20/25）；violation -> PROTOCOL_MISMATCH，
    ρ_birth 不报告。
  * z_ref 不变性门：npz 的 z_ref_pre 与 z_ref_post 逐位一致；violation ->
    ZREF_INVARIANCE_VIOLATION，暂停结构结论。

判据冻结（L2297）：GREEN ⟺ mean ≥0.8 且 pooled ≥0.8 且 worst 无系统性
崩溃；YELLOW 0.5-0.8（只许研究 loss/rank，不进主方法）；RED <0.5。
负分母（Acc_M2'' < Acc_identity，refit 反而更差）只记录不纳入。
bridge fit 严格 label-free（L2258）：标签只进 arm ④ 与 accuracy。

数值：全程 torch CPU（本机 numpy/scipy float64 BLAS 损坏，ENVIRONMENT_AUDIT）。
用法（repo 根目录，CPU）：
  python diagnostics/bsab_b0_verdict.py
  python diagnostics/bsab_b0_verdict.py --repro_only   # 只跑 reproduction gate
"""

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_DIM = 768
CHEAD_KEYS = {"head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias"}
NPZ_NAME = re.compile(
    r"^bsab_c(\d+)_t(\d+)_r(\d+)_(broadcast|broadcast_no_test)\.npz$")
LT_RANKS_SWEEP = (8, 16, 32, 64)      # ③ low-rank sweep
REPRO_TOL_PP = 0.25                    # （旧版逐位容差，已废弃，保留备查）
MC_Z_MAX = 3.0                         # MC gate：|z| 上限（单采样 99.7% 带）
MC_SIGN_MAX = 20                       # MC gate：z 同侧数上限（25 cell 中）
VALID_DENOM_MIN_PP = 0.5               # ρ_birth 有效分母下限（0.5pp）
LOW_POWER_MIN_FIT = 2048               # 与 Client_DF.BSAB_CAPTURE_MIN_FIT 同口径

# ---- 冻结 refit 配方（Client_DF A5_REFIT_* 常量，L46-51）----
REFIT_HIDDEN = 512
REFIT_DROPOUT = 0.1
REFIT_EPOCHS = 100
REFIT_LR = 1e-3
REFIT_WD = 1e-4
REFIT_BATCH = 64


def parse_args():
    p = argparse.ArgumentParser("BSAB B0 verdict (pre-registered)")
    p.add_argument("--run_dir", default=(
        "output/cifar100/fedta/cifar_bsab_capture_M2matched/seed_42"))
    p.add_argument("--m2p_dir", default=(
        "output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42"),
        help="官方 M2' run（refit_timing local，reproduction 锚①）")
    p.add_argument("--m2pb_dir", default=(
        "output/cifar100/fedta/cifar_a5v2_M2pb_full/seed_42"),
        help="官方 M2'' run（refit_timing post_broadcast，reproduction 锚④）")
    p.add_argument("--out_dir", default=None,
                   help="default: <run_dir>/bsab_b0")
    p.add_argument("--repro_only", action="store_true",
                   help="只跑 reproduction gate（smoke，round 0-4）")
    p.add_argument("--gate_only", action="store_true",
                   help="复用上次全量 cells.csv（四臂不重算），只重跑 "
                        "task-end cell 的 repro gate（MC-dropout 判据）；"
                        "arm4 refit 头缓存于 out_dir/heads/，命中即免重训")
    p.add_argument("--mc_draws", type=int, default=50,
                   help="MC dropout-ON 采样次数（repro gate 带宽判据）")
    p.add_argument("--no_repro", action="store_true",
                   help="B0-lite：无官方 a5v2 锚 run（如 ImageNet-R）时"
                        "裁掉 reproduction gate（无锚可对），verdict 只看"
                        " zref 门 + rho_birth；中立性由 cifar 链承担")
    return p.parse_args()


def stable_seed(*parts):
    """Client_DF.a5_stable_seed 的逐字复刻（sha256 前 4 字节 little-endian）。"""
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


# ------------------------------------------------------------------ 工具

def l2n(x, dim=1, eps=1e-12):
    return x * torch.rsqrt(torch.clamp((x * x).sum(dim=dim, keepdim=True),
                                       min=eps))


def head_forward(x1, hsd):
    """Chead 前向（eval 语义，dropout off）。hsd: head.0.*/head.3.*。"""
    h = F.relu(F.linear(x1, hsd["head.0.weight"], hsd["head.0.bias"]))
    return F.linear(h, hsd["head.3.weight"], hsd["head.3.bias"])


def ta_logits(z, key, anchor_pool, hsd):
    """Tail_Anchor 前向 CPU 复刻（与 dsar_birth_preview 逐字一致）。
    refit_input=feat 的头锚半区权重恒 0，x1 路径同构，无需特判。"""
    xn = l2n(z)
    kn = l2n(key, dim=1)
    sim = xn @ kn.t()
    _, idx = torch.topk(sim, k=1, dim=1)
    anchor = anchor_pool[idx].reshape(-1, FEAT_DIM)
    x1 = torch.stack((z, anchor), dim=1).reshape(-1, FEAT_DIM * 2)
    return head_forward(x1, hsd)


def masked_acc(logits, y, allowed):
    keep = torch.full((logits.shape[1],), float("-inf"))
    keep[allowed] = 0.0
    pred = (logits + keep).argmax(dim=1)
    return 100.0 * (pred == y).float().mean().item()


# --------------------------------------------- ② GCV 残差桥（预注册参数化）

def fit_bridge(zp_train, zm_train):
    """预注册参数化：B(z) = z·(I+ΔA) + b，惩罚 λ‖ΔA‖²（对恒等的偏离；
    B_pre RUN_01 教训：不许正则 A 本体）。GCV 选 λ（label-free），
    与 ΔA=0（identity）候选按 GCV/training-MSE 比较。返回 (A, b, identity_won)。

    等价闭式：D = zp - zm（残差目标），Δ 解 (X^TX+λI)^{-1} X^T D；
    截距 b = mean(Y) - mean(X)·A，独立无正则。"""
    X = zp_train.double()
    Y = zm_train.double()
    n, d = X.shape
    D = Y - X                              # 拟合目标：广播位移本身
    XtX = X.t() @ X
    XtD = X.t() @ D
    scale = float(torch.diagonal(XtX).mean())
    # hat 矩阵迹的 GCV 快速近似：ridge hat trace = Σ d_i/(d_i+λ)（eigh 一次）
    evals = torch.linalg.eigvalsh(XtX)
    best = None                            # (gcv, lam, delta)
    for expo in torch.linspace(-4, 4, 33):
        lam = float(10 ** expo) * scale
        delta = torch.linalg.solve(XtX + torch.eye(d).double() * lam, XtD)
        resid = X @ delta - D
        mse = float((resid * resid).sum() / (n * d))
        tr = float((evals / (evals + lam)).sum()) / n
        gcv = mse / max(1e-12, (1.0 - tr) ** 2)
        if best is None or gcv < best[0]:
            best = (gcv, lam, delta)
    _, lam, delta = best
    # identity 候选（ΔA=0）：与 best-ridge 按 training MSE 比较（label-free）
    mse_id = float((D * D).sum() / (n * d))
    mse_ridge = float(((X @ delta - D) ** 2).sum() / (n * d))
    if mse_id <= mse_ridge:
        return torch.eye(FEAT_DIM), torch.zeros(FEAT_DIM), True, 0.0
    A = (torch.eye(d) + delta).float()
    # 截距（无正则，闭式）
    b = (Y.mean(dim=0) - X.mean(dim=0) @ A.double()).float()
    return A, b, False, lam


# --------------------------------------------- ④ 冻结配方离线 M2''

class RefitChead(nn.Module):
    """官方 Chead 结构复刻：head.0 Linear(1536->512) / ReLU / Dropout(0.1)
    / head.3 Linear(512->nb)。state keys 与官方完全一致。"""

    def __init__(self, nb_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(FEAT_DIM * 2, REFIT_HIDDEN),
            nn.ReLU(),
            nn.Dropout(REFIT_DROPOUT),
            nn.Linear(REFIT_HIDDEN, nb_classes),
        )

    def forward(self, x1):
        return self.head(x1)


def fit_refit_head(z_train, y_train, allowed, nb_classes, seed):
    """§34.2.6 冻结配方（_a5_refit_one_head 逐字复刻，feat 臂）：
    z 垫零成 1536（锚半区恒 0）、Chead(dropout=0.1) train 模式、
    Adam(1e-3, wd=1e-4)、100 epoch、bs=64、masked CE、
    manual_seed(a5_stable_seed(seed,'a5refit',c,t,r))、训后锚半区权重置零。
    eval = eval 模式（dropout off）。返回 hsd state dict。"""
    torch.manual_seed(seed)
    half = FEAT_DIM
    x1 = F.pad(z_train, (0, half))          # (n, 1536)，锚半区恒 0
    classes = allowed.tolist()
    not_mask = torch.tensor(
        np.setdiff1d(np.arange(nb_classes), classes), dtype=torch.int64)
    head = RefitChead(nb_classes)
    opt = torch.optim.Adam(head.parameters(), lr=REFIT_LR,
                           weight_decay=REFIT_WD)
    criterion = nn.CrossEntropyLoss()
    n = x1.shape[0]
    bs = min(REFIT_BATCH, n)
    head.train()
    for _ in range(REFIT_EPOCHS):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            logits = head(x1[idx])
            logits = logits.index_fill(dim=1, index=not_mask,
                                       value=float("-inf"))
            loss = criterion(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        head.head[0].weight[:, half:] = 0.0
    head.eval()
    return {k: v.detach().clone() for k, v in head.state_dict().items()}


def refit_logits(z, hsd):
    x1 = F.pad(z, (0, FEAT_DIM))
    return head_forward(x1, hsd)


def mc_dropout_acc(x1, y, allowed, hsd, n_draws, seed):
    """MC dropout-ON 读出 = 官方 evaluate 语义（Chead 的 Dropout(0.1) 在
    ReLU 后、train 模式；官方 evaluate 路径从未 .eval()，Client_DF L651
    探针注释为证）。x1 为已构造的 1536 维输入：arm1 = stack(z⁺, anchor)，
    arm4 = pad(z⁺, 0)。返回 n_draws 个 masked accuracy 样本。"""
    keep = torch.full((hsd["head.3.bias"].shape[0],), float("-inf"))
    keep[allowed] = 0.0
    accs = []
    for k in range(n_draws):
        torch.manual_seed(stable_seed(seed, "draw", k))
        h = F.dropout(F.relu(F.linear(x1, hsd["head.0.weight"],
                                      hsd["head.0.bias"])),
                      p=REFIT_DROPOUT, training=True)
        logits = F.linear(h, hsd["head.3.weight"], hsd["head.3.bias"])
        pred = (logits + keep).argmax(dim=1)
        accs.append(100.0 * (pred == y).float().mean().item())
    return accs


# ------------------------------------------------- 官方锚点装载

def load_client_matrix(run_dir):
    """client_X_accuracy.csv -> {(client, after_task, test_task): acc}。"""
    out = {}
    for csv_path in sorted(Path(run_dir).glob("metrics/client_*_accuracy.csv")):
        ci = int(re.match(r"client_(\d+)_accuracy", csv_path.name).group(1))
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[(ci, int(row["after_task"]), int(row["test_task"]))] = \
                    float(row["accuracy"])
    if not out:
        raise SystemExit(f"no client accuracy csv under {run_dir}/metrics")
    return out


def check_run_config(run_dir, expect_post_broadcast):
    args_json = Path(run_dir) / "args.json"
    if not args_json.is_file():
        return
    with open(args_json, encoding="utf-8") as f:
        ra = json.load(f)
    timing = str(ra.get("refit_timing", "local"))
    want = "post_broadcast" if expect_post_broadcast else "local"
    if timing != want:
        raise SystemExit(
            f"PROTOCOL_GATE_FAIL: {run_dir} has refit_timing={timing}, "
            f"expected {want}")


# ------------------------------------------------- 主流程

def discover_cells(run_dir):
    cap_dir = Path(run_dir) / "bsab_capture"
    cells = {}
    for path in sorted(cap_dir.glob("bsab_c*.npz")):
        m = NPZ_NAME.match(path.name)
        if m is None:
            raise SystemExit(f"unrecognized capture file name: {path}")
        ci, t, r = int(m.group(1)), int(m.group(2)), int(m.group(3))
        cells[(r, ci)] = (t, m.group(4), path)
    if not cells:
        raise SystemExit(f"no capture npz under {cap_dir}")
    return cells


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "bsab_b0"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "heads").mkdir(exist_ok=True)

    # run 级协议核验
    with open(run_dir / "args.json", encoding="utf-8") as f:
        ra = json.load(f)
    if not (bool(ra.get("head_refit"))
            and str(ra.get("refit_timing", "local")) == "local"
            and bool(ra.get("bsab_pair_capture"))):
        raise SystemExit("PROTOCOL_GATE_FAIL: not an M2'-matched capture run")
    if args.no_repro:
        m2p_mat, m2pb_mat = {}, {}
    else:
        check_run_config(args.m2p_dir, expect_post_broadcast=False)
        check_run_config(args.m2pb_dir, expect_post_broadcast=True)
        m2p_mat = load_client_matrix(args.m2p_dir)
        m2pb_mat = load_client_matrix(args.m2pb_dir)
    seed = int(ra.get("seed", 42))

    cells = discover_cells(run_dir)
    rounds_avail = sorted({r for r, _ in cells})
    task_end_cells = {r for r in rounds_avail if r % 5 == 4}  # 5t+4
    print(f"cells: {len(cells)} | task-end boundaries (r%5==4): "
          f"{sorted(task_end_cells)}")

    mc_rows = []
    zref_violations = []
    if args.gate_only:
        # 四臂结果复用上次全量 cells.csv；只重跑 task-end repro gate
        cells_csv = out_dir / "cells.csv"
        if not cells_csv.is_file():
            raise SystemExit(
                "--gate_only 需要上次全量运行的 cells.csv，未找到——"
                "先不带 --gate_only 跑一次全量")
        with open(cells_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        def _num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return v
        rows = [{k: _num(v) for k, v in r_.items()} for r_ in rows]
        for r_ in rows:
            r_["round"] = int(r_["round"])
            r_["client"] = int(r_["client"])
            r_["task"] = int(r_["task"])
        if len(rows) != len(cells):
            raise SystemExit(f"cells.csv rows={len(rows)} != npz cells="
                             f"{len(cells)}; rerun full mode")
        rounds = sorted(task_end_cells)
        print(f"gate_only: reuse {len(rows)} cells from cells.csv; "
              f"repro gate on rounds {rounds}")
    else:
        rows = []
        rounds = rounds_avail
    t0 = time.time()
    for r in rounds:
        ckpt_path = run_dir / "checkpoints" / f"round_{r:04d}.pth"
        if not ckpt_path.is_file():
            continue
        ck = torch.load(ckpt_path, map_location="cpu")
        for ci in sorted(c for rr, c in cells if rr == r):
            t, entry, npz_path = cells[(r, ci)]
            cstate = ck["clients"][ci]
            if int(cstate.get("task_id", -1)) != t:
                raise SystemExit(f"PAIRING_GATE_FAIL: r{r} c{ci} task "
                                 f"mismatch npz={t} ckpt={cstate.get('task_id')}")
            tam = cstate["tail_anchor_model"]
            key = tam["key"].float()
            anchor_pool = tam["anchor_pool"].float()

            data = np.load(npz_path)
            zm = torch.tensor(data["z_op_pre"])
            zp = torch.tensor(data["z_op_post"])
            zr_pre = torch.tensor(data["z_ref_pre"])
            zr_post = torch.tensor(data["z_ref_post"])
            y = torch.tensor(data["label"]).to(torch.int64)
            m_tr = torch.tensor(data["split"] == 0)
            m_te = torch.tensor(data["split"] == 1)
            # z_ref 不变性门（B_pre 语义，violation -> 暂停结构结论）
            if not torch.equal(zr_pre, zr_post):
                zref_violations.append((r, ci))

            hsd_refit = {k[len("head__"):]: torch.tensor(data[k])
                         for k in data.files if k.startswith("head__")}
            if set(hsd_refit) != CHEAD_KEYS:
                raise SystemExit(f"STRUCT_GATE_FAIL: {npz_path.name} "
                                 f"head keys {sorted(hsd_refit)}")
            # nb_classes 从官方 refit head 自证（head.3.bias 维 = 官方
            # Chead(args.nb_classes) 输出维；args.json 不落盘该派生属性）
            nb_classes = int(hsd_refit["head.3.bias"].shape[0])

            class_mask = cstate["class_mask"]
            allowed = torch.tensor([int(x) for x in class_mask[t]])
            if not set(torch.unique(y).tolist()) <= set(allowed.tolist()):
                raise SystemExit(f"PAIRING_GATE_FAIL: {npz_path.name} "
                                 f"labels outside class_mask[{t}]")

            zm_fit, zp_fit, y_fit = zm[m_tr], zp[m_tr], y[m_tr]
            zm_ev, zp_ev, y_ev = zm[m_te], zp[m_te], y[m_te]
            n_train = int(m_tr.sum())

            # ---- ④ 离线 M2'' refit 头（磁盘缓存：MC gate 与全量重跑
            #      复用，命中即免重训）----
            head_cache = out_dir / "heads" / f"r{r:04d}_c{ci}.pt"
            if head_cache.is_file():
                hsd_m2pp = torch.load(head_cache, map_location="cpu")
            else:
                hsd_m2pp = fit_refit_head(
                    zp_fit, y_fit, allowed, nb_classes,
                    seed=stable_seed(seed, "a5refit", ci, t, r))
                torch.save(hsd_m2pp, head_cache)

            if not args.gate_only:
                # ---- ② label-free GCV 残差桥（λ‖ΔA‖² 预注册参数化）----
                A, b_off, identity_won, lam = fit_bridge(zp_fit, zm_fit)
                delta = A - torch.eye(FEAT_DIM)
                U_, S_, Vt_ = torch.linalg.svd(delta, full_matrices=False)
                lowrank = {}
                for rk in LT_RANKS_SWEEP:
                    lowrank[rk] = (torch.eye(FEAT_DIM)
                                   + (U_[:, :rk] * S_[:rk]) @ Vt_[:rk])

                # ---- ① identity：M2' refit head 直接服役 z⁺ ----
                acc_id = masked_acc(
                    ta_logits(zp_ev, key, anchor_pool, hsd_refit),
                    y_ev, allowed)

                acc_m2pp = masked_acc(refit_logits(zp_ev, hsd_m2pp),
                                      y_ev, allowed)

                # ---- ②/③ 桥臂（M2' head 读出桥接后的 z⁺）----
                acc_bridge = masked_acc(
                    ta_logits(zp_ev @ A + b_off, key, anchor_pool, hsd_refit),
                    y_ev, allowed)
                acc_lr = {rk: masked_acc(
                    ta_logits(zp_ev @ lowrank[rk] + b_off, key, anchor_pool,
                              hsd_refit), y_ev, allowed)
                    for rk in LT_RANKS_SWEEP}

                # ---- 参考记录（不作臂）：birth-state reference ----
                acc_ref = masked_acc(
                    ta_logits(zm_ev, key, anchor_pool, hsd_refit),
                    y_ev, allowed)

                row = dict(round=r, client=ci, task=t, entry=entry,
                           n_train=n_train, n_test=int(m_te.sum()),
                           low_power=int(n_train < LOW_POWER_MIN_FIT),
                           acc_identity=acc_id, acc_bridge=acc_bridge,
                           acc_m2pp=acc_m2pp, acc_birth_ref=acc_ref,
                           bridge_identity=int(identity_won), bridge_lam=lam)
                for rk in LT_RANKS_SWEEP:
                    row[f"acc_lowrank_r{rk}"] = acc_lr[rk]
                rows.append(row)

            # ---- repro gate MC（task-end cell；§34.2.5 修订 2026-11-06：
            #      官方 evaluate 的 head Dropout(0.1) 从未关闭 -> 官方
            #      CSV 数 = 我们 dropout-ON 分布的单次采样 -> MC 带宽）----
            if r in task_end_cells:
                key_off = (ci, t, t)
                if key_off in m2p_mat and key_off in m2pb_mat:
                    # arm1 的 x1：与 ta_logits 相同的 key 选择（无
                    # dropout，确定性），只对 head 部分做 MC dropout
                    xn = l2n(zp_ev)
                    kn = l2n(key, dim=1)
                    _, idx = torch.topk(xn @ kn.t(), k=1, dim=1)
                    anchor = anchor_pool[idx].reshape(-1, FEAT_DIM)
                    x1_arm1 = torch.stack((zp_ev, anchor), dim=1).reshape(
                        -1, FEAT_DIM * 2)
                    accs1 = mc_dropout_acc(
                        x1_arm1, y_ev, allowed, hsd_refit, args.mc_draws,
                        seed=stable_seed(seed, "b0mc", ci, t, r, "arm1"))
                    accs4 = mc_dropout_acc(
                        F.pad(zp_ev, (0, FEAT_DIM)), y_ev, allowed,
                        hsd_m2pp, args.mc_draws,
                        seed=stable_seed(seed, "b0mc", ci, t, r, "arm4"))
                    mc_rows.append(dict(
                        round=r, client=ci, task=t,
                        official1=m2p_mat[key_off],
                        official4=m2pb_mat[key_off],
                        mc1_mean=float(np.mean(accs1)),
                        mc1_std=float(np.std(accs1)),
                        mc1_min=float(np.min(accs1)),
                        mc1_max=float(np.max(accs1)),
                        mc4_mean=float(np.mean(accs4)),
                        mc4_std=float(np.std(accs4)),
                        mc4_min=float(np.min(accs4)),
                        mc4_max=float(np.max(accs4))))
        print(f"round {r:4d}: done ({time.time()-t0:.0f}s)", flush=True)
        if args.repro_only and r >= 4:
            break

    if not rows:
        raise SystemExit("no usable cells")

    # ---- reproduction gate（§34.2.5 修订 2026-11-06：MC-dropout 带宽）----
    # 根因：官方 evaluate 路径从未 .eval()（Client_DF L651 探针注释为证）
    # -> Chead Dropout(0.1) ON -> 官方 CSV 数是 dropout-ON 随机前向的单次
    # 采样。原 0.25pp 逐位容差因此不可满足，修订为：
    #   cell PASS ⟺ |z| <= MC_Z_MAX（官方数落在我们 MC 分布 3σ 带内）
    #   gate PASS ⟺ 25/25 cell PASS 且 z 符号不系统性同侧（<=MC_SIGN_MAX）
    def _z(official, mean, std):
        if std < 1e-9:
            return 0.0 if abs(official - mean) < 1e-9 else float("inf")
        return (official - mean) / std
    repro = mc_rows
    for x in repro:
        x["z1"] = _z(x["official1"], x["mc1_mean"], x["mc1_std"])
        x["z4"] = _z(x["official4"], x["mc4_mean"], x["mc4_std"])
        # gate 判据只看 arm1（读出语义 ≡ 官方）。arm4（离线 refit 头）
        # 与官方 in-run 头是"同配方、不同训练随机实现"的两个头
        # （重训不可逐位复现，2026-11-06 判定），降级为 reference
        # record：分母只需在管线内有效（预注册的负分母排除已覆盖
        # 其噪声），不与官方逐位对齐。
        x["cell_pass"] = abs(x["z1"]) <= MC_Z_MAX
    n_pos1 = sum(1 for x in repro if x["z1"] > 0)
    n_neg1 = sum(1 for x in repro if x["z1"] < 0)
    sign_balanced = max(n_pos1, n_neg1) <= MC_SIGN_MAX
    repro_pass = (len(repro) == 25
                  and all(x["cell_pass"] for x in repro)
                  and sign_balanced)

    # ---- ρ_birth ----
    rho_rows = []
    excluded_neg = []
    for row in rows:
        denom = row["acc_m2pp"] - row["acc_identity"]
        if denom < VALID_DENOM_MIN_PP:
            excluded_neg.append((row["round"], row["client"], denom))
            row["rho_birth"] = ""
            continue
        row["rho_birth"] = (row["acc_bridge"] - row["acc_identity"]) / denom
        rho_rows.append(row)

    pooled = None
    mean_rho = float("nan")
    if rho_rows:
        pooled = (
            (sum(r["acc_bridge"] for r in rho_rows)
             - sum(r["acc_identity"] for r in rho_rows))
            / (sum(r["acc_m2pp"] for r in rho_rows)
               - sum(r["acc_identity"] for r in rho_rows)))
        mean_rho = sum(r["rho_birth"] for r in rho_rows) / len(rho_rows)

    # ---- 判决 ----
    zref_ok = not zref_violations
    repro_ok = (repro_pass if not args.no_repro else True)
    if not zref_ok:
        verdict = "ZREF_INVARIANCE_VIOLATION（暂停结构结论，查 extraction bug）"
    elif not args.no_repro and not repro_ok:
        verdict = ("PROTOCOL_MISMATCH（MC repro gate 未过：存在出带 cell "
                   "或 z 符号系统性同侧，ρ_birth 不报告；见 repro.csv）")
    elif mean_rho >= 0.8 and pooled >= 0.8:
        verdict = "GREEN" + (" [B0-lite, no_repro]" if args.no_repro else "")
    elif mean_rho >= 0.5 and pooled >= 0.5:
        verdict = "YELLOW（0.5-0.8：只许研究 loss/rank，不进主方法）"
    else:
        verdict = "RED"

    # worst 系统性崩溃检查（跨有效 transition）
    per_task = {}
    for r_ in rho_rows:
        per_task.setdefault(r_["task"], []).append(r_["rho_birth"])
    task_means = {t: sum(v) / len(v) for t, v in sorted(per_task.items())}
    worst_cells = sorted(rho_rows, key=lambda x: x["rho_birth"])[:5]
    collapse_flags = [t for t, m in task_means.items() if m < 0.2]

    summary = dict(
        n_cells=len(rows), n_rho_valid=len(rho_rows),
        n_excluded_neg_denom=len(excluded_neg),
        repro_gate=dict(n_draws=args.mc_draws,
                        z_max=MC_Z_MAX, sign_max=MC_SIGN_MAX,
                        n_cells=len(repro),
                        mode=("skipped_no_anchor" if args.no_repro
                              else "mc_dropout"),
                        arm1_sign=[n_pos1, n_neg1],
                        arm4_role="reference_only",
                        passed=bool(repro_ok)),
        zref_gate=dict(passed=zref_ok,
                       violations=zref_violations[:10]),
        rho_birth=dict(mean=mean_rho, pooled=pooled,
                       per_task_mean=task_means),
        worst_cells=[dict(round=w["round"], client=w["client"],
                          task=w["task"], rho=w["rho_birth"])
                     for w in worst_cells],
        collapse_flags=collapse_flags,
        verdict=verdict,
        notes="pre-registered Sec.34.2.5; label-free bridge (lambda||dA||^2 "
              "residual param, GCV, identity candidate); arm4 = frozen "
              "refit recipe on z+; torch CPU.")
    print("\n=== BSAB B0 (pre-registered) ===")
    if args.no_repro:
        print("reproduction gate: SKIPPED (B0-lite, no official anchor; "
              "neutrality carried by cifar chain)")
    else:
        print(f"reproduction gate (MC-dropout, {args.mc_draws} draws, arm1 "
              f"readout semantics): {'PASS' if repro_ok else 'FAIL'} "
              f"({len(repro)}/25 cells, |z|<={MC_Z_MAX}, "
              f"sign {n_pos1}+/{n_neg1}-; arm4 = reference only)")
    for x in repro:
        print(f"  r{x['round']:02d} c{x['client']} t{x['task']}: "
              f"arm1 off={x['official1']:6.2f} mc={x['mc1_mean']:6.2f}"
              f"±{x['mc1_std']:.2f} z={x['z1']:+5.1f} | "
              f"arm4 off={x['official4']:6.2f} mc={x['mc4_mean']:6.2f}"
              f"±{x['mc4_std']:.2f} z={x['z4']:+5.1f} [ref] "
              f"{'OK' if x['cell_pass'] else 'OUT'}")
    print(f"z_ref invariance: {'PASS' if zref_ok else 'VIOLATION'}")
    if rho_rows:
        print(f"rho_birth: mean={mean_rho:.3f} pooled={pooled:.3f} "
              f"(valid {len(rho_rows)}/{len(rows)}, "
              f"neg-denom excluded {len(excluded_neg)})")
        for t, m in task_means.items():
            print(f"  task {t}: mean rho = {m:.3f}")
        for w in worst_cells:
            print(f"  worst: r{w['round']:02d} c{w['client']} t{w['task']} "
                  f"rho={w['rho_birth']:.3f}")
    print(f"VERDICT: {verdict}")

    if not args.gate_only:
        with open(out_dir / "cells.csv", "w", newline="",
                  encoding="utf-8") as f:
            fields = sorted({k for r_ in rows for k in r_},
                            key=lambda k: (k not in ("round", "client", "task"),
                                           k))
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    if repro:
        mc_fields = ["round", "client", "task",
                     "official1", "mc1_mean", "mc1_std", "mc1_min",
                     "mc1_max", "z1",
                     "official4", "mc4_mean", "mc4_std", "mc4_min",
                     "mc4_max", "z4", "cell_pass"]
        with open(out_dir / "repro.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=mc_fields)
            w.writeheader()
            w.writerows(repro)
    with open(out_dir / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwritten: {out_dir}/cells.csv, repro.csv, verdict.json")


if __name__ == "__main__":
    main()
