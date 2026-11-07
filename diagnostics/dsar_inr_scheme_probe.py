"""INR 方案离线网格 probe（§34.6/§34.6a 之后向；diagnostics-only）。

问题：CIFAR DSAR 配方（γ*=0.396 / r8 / B+L / paired-MSE guard）在 INR 上
信号失效（γ*_INR=2.058、B0 RED、R_r 分母 124/125 退化）。§34.6a 已终审
方向（L16 主导 + B8 历史迁移臂），但**部署口径的方案效果从未被离线验证**：
Stage 0 preview 的 L 是 z- 基底 200ep full-batch 拟合，且 CE guard（Q2
裁决）没有实现。本工具在 izfree 的 INR capture npz（125 cell）上做
**零训练-run 离线 replay 网格**，回答"哪个方案在 INR 上有效"：

网格（评估统一 = z⁺ test 行、birth head（npz M2' head）Tail_Anchor 读出、
class-masked accuracy；anchor routing 见校正后 z，与部署语义一致）：

  基线   off                : identity（z⁺ raw）—— 损伤起点
  参考   ref_zm             : z⁻ birth 读出（preview acc_raw 口径）
  L 臂   L{r}_{zp|zm}_{ce|no}
         r ∈ {8,16,32}；基底 zp = 在 z⁺ train 行拟合（INR-R0 自然语义，
         §34.6a Q2 CE guard 的部署对象 L16(z⁺)）；基底 zm = 在 z⁻ train 行
         拟合（CIFAR 生产语义，L 部署在 B(z⁺) 的位置这里直接 z⁺）。
         guard ce = §34.6a Q2 部署态 CE guard：80% 行拟合（不 refit）→
         剩余 20% paired CE（CE_L < CE_I 才 active，平局 identity）；
         no = 全 train 行拟合、无条件部署。
  B 臂   B{r}_{γ}_{mse|no}
         r ∈ {8,16}；γ ∈ inr(2.0579 冻结)/cifar(0.396419)/gcv(label-free
         sweep)/lam0(min-norm 无 shrink)；label-free centered residual
         ridge z⁺→z⁻（生产 dsar_fit_bridge 参数化：λ=N·s·γ，SVD r 截断 +
         截断算子下 bias 重算）。guard mse = 生产 80/20 hash split +
         r 截断候选 vs identity 的 cal paired-MSE（胜出才全行 refit 再截断）；
         no = 全行直接拟合部署。
  组合   chain_B8inr_L16zp / chain_B8inr_L16zm
         B8(γ_inr, mse guard) 之后 L16（§34.6 组合可加性复核）。

L 拟合配方 = 生产 dsar_fit_l 逐字语义：100ep / bs64 / Adam(lr1e-3,
wd1e-4) / U randn*0.01、V=0、b=0 init（恒等起步）/ masked CE 穿冻结
birth-head 读出 / 专用 torch.Generator（不碰全局 RNG）。CE/读出前向
全部 eval 语义（dropout off，确定性），guard 两侧同一确定性前向（Q2
"固定随机前向协议"的离线实现）。

产出：<out_dir>/cells.csv（cell × 配置宽表）+ summary.json（配置排名，
主键 = task-end pooled delta，Stage 3 门 +0.8pp 的口径）+ 控制台排名。

用法（izfree repo 根目录；CPU 可跑，--device cuda 更快）：
  python diagnostics/dsar_inr_scheme_probe.py \
      --run_dir output/ImageNet-R/fedta/inr_bsab_capture_M2matched_p40/seed_42
  python diagnostics/dsar_inr_scheme_probe.py --run_dir ... --rounds 4,9   # 冒烟
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
import torch.nn.functional as F

FEAT_DIM = 768
CHEAD_KEYS = {"head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias"}
NPZ_NAME = re.compile(
    r"^bsab_c(\d+)_t(\d+)_r(\d+)_(broadcast|broadcast_no_test)\.npz$")

# ---- 冻结常量（§34.6a）----
GAMMA_INR = 2.0579118185313345      # γ*_INR 全精度（Q4 冻结，仅 B 验证臂）
GAMMA_CIFAR = 0.396419              # Models/DSAR.GAMMA_STAR
L_RANKS = (8, 16, 32)               # r16 冻结；r32 探索对照；r8 参考
B_RANKS = (8, 16)
GCV_EXPOS = tuple(float(x) for x in torch.linspace(-4, 4, 33))
MIN_FIT_ROWS = 32                   # LOW_POWER_IDENTITY（生产口径）
MIN_CAL_ROWS = 8

# ---- L 拟合配方（生产 dsar_fit_l 冻结常量）----
L_EPOCHS = 100
L_BATCH = 64
L_LR = 1e-3
L_WD = 1e-4


def parse_args():
    p = argparse.ArgumentParser("INR scheme offline probe")
    p.add_argument("--run_dir", default=(
        "output/ImageNet-R/fedta/inr_bsab_capture_M2matched_p40/seed_42"))
    p.add_argument("--out_dir", default=None,
                   help="default: <run_dir>/dsar_inr_probe")
    p.add_argument("--rounds", default="all",
                   help="'all' 或逗号分隔 round id（冒烟用）")
    p.add_argument("--device", default="cpu",
                   help="L 拟合/读出的张量设备（cuda 可显著加速）")
    p.add_argument("--l_epochs", default=0, type=int,
                   help="0 = 生产冻结值 100（冒烟可调小，结果不可采信）")
    return p.parse_args()


def stable_seed(*parts):
    """与 Client_DF.a5_stable_seed / DSAR.dsar_stable_seed 同构。"""
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],
                           "little")


def split_fit_cal(sample_idx):
    """生产 dsar_split_fit_cal 逐字复刻（sample-index hash 80/20）。"""
    n = len(sample_idx)
    fit = [True] * n
    cal = [False] * n
    for i, idx in enumerate(sample_idx):
        h = int.from_bytes(
            hashlib.sha256(("dsar_split:%d" % int(idx)).encode())
            .digest()[:4], "little")
        if h % 5 == 0:
            fit[i] = False
            cal[i] = True
    return fit, cal


# ------------------------------------------------------------------ 读出

def l2n(x, dim=1, eps=1e-12):
    return x * torch.rsqrt(torch.clamp((x * x).sum(dim=dim, keepdim=True),
                                       min=eps))


def head_forward(x1, hsd):
    h = F.relu(F.linear(x1, hsd["head.0.weight"], hsd["head.0.bias"]))
    return F.linear(h, hsd["head.3.weight"], hsd["head.3.bias"])


def ta_logits(z, key, anchor_pool, hsd):
    """Tail_Anchor 前向（eval 语义，dropout off）；routing 见输入 z。"""
    xn = l2n(z)
    kn = l2n(key, dim=1)
    sim = xn @ kn.t()
    _, idx = torch.topk(sim, k=1, dim=1)
    anchor = anchor_pool[idx].reshape(-1, FEAT_DIM)
    x1 = torch.stack((z, anchor), dim=1).reshape(-1, FEAT_DIM * 2)
    return head_forward(x1, hsd)


def masked_acc(logits, y, allowed):
    keep = torch.full((logits.shape[1],), float("-inf"),
                     device=logits.device)
    keep[allowed.to(logits.device)] = 0.0
    pred = (logits + keep).argmax(dim=1)
    return 100.0 * (pred == y.to(logits.device)).float().mean().item()


def masked_ce(z, y, key, anchor_pool, hsd, allowed, nb_classes):
    """部署态 CE（eval 确定性前向，guard 两侧同一协议）。"""
    logits = ta_logits(z, key, anchor_pool, hsd)
    keep = torch.full((nb_classes,), float("-inf"), device=z.device)
    keep[allowed.to(z.device)] = 0.0
    return F.cross_entropy(logits + keep, y)


# ------------------------------------------------------------------ B 拟合

def b_solve(zp_rows, zm_rows, lam):
    """生产参数化 centered residual ridge 的一次求解（CPU float64）。
    返回 (dA, xbar, ybar)；lam=0 时走 min-norm lstsq（欠定安全）。"""
    X = zp_rows.double()
    Y = zm_rows.double()
    n_, d_ = X.shape
    xbar, ybar = X.mean(dim=0), Y.mean(dim=0)
    xc, yc = X - xbar, Y - ybar
    XtX = xc.t() @ xc
    D = yc - xc
    if lam > 0.0:
        dA = torch.linalg.solve(XtX + torch.eye(d_).double() * lam,
                                xc.t() @ D)
    else:
        dA = torch.linalg.lstsq(XtX, xc.t() @ D).solution
    if not bool(torch.isfinite(dA).all()):
        raise RuntimeError("b_solve: non-finite ridge solution (fail-closed)")
    return dA, xbar, ybar


def b_scale(zp_rows):
    X = zp_rows.double()
    xc = X - X.mean(dim=0)
    return float((xc * xc).sum()) / float(X.shape[0] * X.shape[1])


def b_lambda(mode, zp_fit):
    """λ 选择：inr/cifar = N·s·γ 固定；lam0 = 0（gcv 走 b_lambda_gcv）。
    返回 (lam, gcv_expo)。"""
    n = zp_fit.shape[0]
    s = b_scale(zp_fit)
    if mode in ("inr", "cifar"):
        return float(n) * s * (GAMMA_INR if mode == "inr" else GAMMA_CIFAR), \
            float("nan")
    if mode == "lam0":
        return 0.0, float("nan")
    raise AssertionError(f"gcv mode must use b_lambda_gcv, got {mode}")


def b_lambda_gcv(zp_fit, zm_fit):
    """GCV label-free λ（b0 fit_bridge 的 GCV 口径，centered 残差形式）。"""
    X = zp_fit.double()
    Y = zm_fit.double()
    n_, d_ = X.shape
    xc, yc = X - X.mean(dim=0), Y - Y.mean(dim=0)
    XtX = xc.t() @ xc
    D = yc - xc
    XtD = xc.t() @ D
    scale = float(torch.diagonal(XtX).mean())
    evals = torch.linalg.eigvalsh(XtX)
    best = None
    for expo in GCV_EXPOS:
        lam = float(10 ** expo) * scale
        dA = torch.linalg.solve(XtX + torch.eye(d_).double() * lam, XtD)
        resid = xc @ dA - D
        mse = float((resid * resid).sum() / (n_ * d_))
        tr = float((evals / (evals + lam)).sum()) / n_
        gcv = mse / max(1e-12, (1.0 - tr) ** 2)
        if best is None or gcv < best[0]:
            best = (gcv, lam, float(expo))
    return best[1], best[2]


def b_trunc(dA, rank):
    """SVD r 截断：dA_r = M1 @ M2^T（M1 = U_r S_r, M2 = V_r）。"""
    U_, S_, Vt_ = torch.linalg.svd(dA, full_matrices=False)
    return U_[:, :rank] * S_[:rank], Vt_[:rank].t().contiguous()


def b_apply(z, M1, M2, bias):
    """部署读出约定（生产 _apply_factors）：z + (z@M1)@M2^T + bias。"""
    return z + (z @ M1) @ M2.t() + bias


def fit_b(zp, zm, sample_idx, mode, rank, guard):
    """一个 B 配置的完整拟合。返回 (M1, M2, bias 或 None[identity],
    status, aux)。guard=True 走生产流程（80/20 + 截断候选 cal-MSE 比较 +
    胜出全行 refit）；guard=False 全行直接拟合。"""
    n = zp.shape[0]
    if n < MIN_FIT_ROWS:
        return None, None, None, "LOW_POWER_IDENTITY", {}
    if guard:
        fit, cal = split_fit_cal(sample_idx)
        idx_f = torch.tensor(fit)
        idx_c = torch.tensor(cal)
        n_fit, n_cal = int(idx_f.sum()), int(idx_c.sum())
        if n_fit < MIN_FIT_ROWS or n_cal < MIN_CAL_ROWS:
            return None, None, None, "LOW_POWER_IDENTITY", {}
        zp_f, zm_f, zp_c, zm_c = zp[idx_f], zm[idx_f], zp[idx_c], zm[idx_c]
        if mode == "gcv":
            lam, expo = b_lambda_gcv(zp_f, zm_f)
        else:
            lam, expo = b_lambda(mode, zp_f)
        dA, xbar, ybar = b_solve(zp_f, zm_f, lam)
        M1, M2 = b_trunc(dA, rank)
        bias = ybar - xbar - xbar @ (M1 @ M2.t())
        out = b_apply(zp_c.double(), M1, M2, bias)
        resid = out - zm_c.double()
        mse_r = float((resid * resid).mean())
        diff = zp_c.double() - zm_c.double()
        mse_id = float((diff * diff).mean())
        aux = {"lam": lam, "lam_expo": expo,
               "mse_gain_cal": mse_id - mse_r}
        if mse_id <= mse_r:
            return None, None, None, "IDENTITY_GUARD", aux
        # 胜出：全行 refit + 再截断（生产流程）
        if mode == "gcv":
            lam, expo = b_lambda_gcv(zp, zm)
        else:
            lam, expo = b_lambda(mode, zp)
        dA, xbar, ybar = b_solve(zp, zm, lam)
        M1, M2 = b_trunc(dA, rank)
        bias = ybar - xbar - xbar @ (M1 @ M2.t())
        aux["lam_refit"] = lam
        return M1.float(), M2.float(), bias.float(), "FITTED", aux
    # guard off：全行直接拟合
    if mode == "gcv":
        # label-free λ 仍可在全行上选（GCV 无需 held-out）
        lam, expo = b_lambda_gcv(zp, zm)
    else:
        lam, expo = b_lambda(mode, zp)
    dA, xbar, ybar = b_solve(zp, zm, lam)
    M1, M2 = b_trunc(dA, rank)
    bias = ybar - xbar - xbar @ (M1 @ M2.t())
    return M1.float(), M2.float(), bias.float(), "FITTED", \
        {"lam": lam, "lam_expo": expo}


# ------------------------------------------------------------------ L 拟合

def fit_l(z_fit, y_fit, key, anchor_pool, hsd, allowed, nb_classes, rank,
          seed, device, epochs):
    """生产 dsar_fit_l 逐字语义：U randn*0.01 / V=0 / b=0 init（恒等
    起步）、100ep、bs64、Adam(1e-3, wd1e-4)、masked CE 穿冻结 birth-head
    读出、专用 CPU Generator。返回 (M1=V, M2=U, bias) 或 None。"""
    n = z_fit.shape[0]
    if n < MIN_FIT_ROWS:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    U = (torch.randn(FEAT_DIM, rank, generator=g) * 0.01).to(device)
    V = torch.zeros(FEAT_DIM, rank, device=device)
    b = torch.zeros(FEAT_DIM, device=device)
    U.requires_grad_(True)
    V.requires_grad_(True)
    b.requires_grad_(True)
    z = z_fit.to(device)
    y = y_fit.to(device)
    key_d = key.to(device)
    pool_d = anchor_pool.to(device)
    hsd_d = {k: v.to(device) for k, v in hsd.items()}
    keep = torch.full((nb_classes,), float("-inf"), device=device)
    keep[allowed] = 0.0
    opt = torch.optim.Adam([U, V, b], lr=L_LR, weight_decay=L_WD)
    bs = min(L_BATCH, n)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            zb, yb = z[idx], y[idx]
            z_adj = zb + (zb @ V) @ U.t() + b
            logits = ta_logits(z_adj, key_d, pool_d, hsd_d)
            loss = F.cross_entropy(logits + keep, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        if not (bool(torch.isfinite(U).all()) and bool(torch.isfinite(V).all())
                and bool(torch.isfinite(b).all())):
            raise RuntimeError("fit_l: non-finite factors (fail-closed)")
        return V.detach(), U.detach(), b.detach()


def l_apply(z, M1, M2, bias):
    return z + (z @ M1) @ M2.t() + bias


# ---------------------------------------------------------- 配置空间

L_CONFIGS = [(rk, basis, guard)
             for rk in L_RANKS for basis in ("zp", "zm")
             for guard in ("ce", "no")]
B_CONFIGS = [(rk, gm, guard)
             for rk in B_RANKS for gm in ("inr", "cifar", "gcv", "lam0")
             for guard in ("mse", "no")]
CHAIN_CONFIGS = ("chain_B8inr_L16zp", "chain_B8inr_L16zm")


def cfg_name_l(rk, basis, guard):
    return f"L{rk}_{basis}_{guard}"


def cfg_name_b(rk, gm, guard):
    return f"B{rk}_{gm}_{guard}"


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
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "dsar_inr_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    epochs = args.l_epochs if args.l_epochs > 0 else L_EPOCHS

    with open(run_dir / "args.json", encoding="utf-8") as f:
        ra = json.load(f)
    if not (bool(ra.get("head_refit"))
            and str(ra.get("refit_timing", "local")) == "local"
            and bool(ra.get("bsab_pair_capture"))):
        raise SystemExit("PROTOCOL_GATE_FAIL: not an M2'-matched capture run")
    seed = int(ra.get("seed", 42))

    cells = discover_cells(run_dir)
    rounds_avail = sorted({r for r, _ in cells})
    if args.rounds == "all":
        rounds = rounds_avail
    else:
        rounds = [int(x) for x in args.rounds.split(",") if x.strip() != ""]
        rounds = [r for r in rounds if r in rounds_avail]
    if not rounds:
        raise SystemExit(f"no usable rounds (avail {rounds_avail})")
    print(f"cells: {len(cells)} | probe rounds: {rounds} | "
          f"L epochs: {epochs} | device: {device}")

    rows = []
    zref_violations = []
    t0 = time.time()
    for r in rounds:
        ckpt_path = run_dir / "checkpoints" / f"round_{r:04d}.pth"
        if not ckpt_path.is_file():
            print(f"round {r}: no checkpoint, skip")
            continue
        ck = torch.load(ckpt_path, map_location="cpu")
        for ci in sorted(c for rr, c in cells if rr == r):
            t, entry, npz_path = cells[(r, ci)]
            cstate = ck["clients"][ci]
            if int(cstate.get("task_id", -1)) != t:
                raise SystemExit(
                    f"PAIRING_GATE_FAIL: r{r} c{ci} task mismatch "
                    f"npz={t} ckpt={cstate.get('task_id')}")
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
            if "sample_index" not in data.files:
                raise SystemExit(
                    f"STRUCT_GATE_FAIL: {npz_path.name} has no sample_index "
                    "(B guard 的 80/20 hash split 需要行索引)")
            sample_idx = torch.tensor(data["sample_index"]).to(torch.int64)
            if not torch.equal(zr_pre, zr_post):
                zref_violations.append((r, ci))

            hsd = {k[len("head__"):]: torch.tensor(data[k])
                   for k in data.files if k.startswith("head__")}
            if set(hsd) != CHEAD_KEYS:
                raise SystemExit(f"STRUCT_GATE_FAIL: {npz_path.name} "
                                 f"head keys {sorted(hsd)}")
            nb_classes = int(hsd["head.3.bias"].shape[0])

            class_mask = cstate["class_mask"]
            allowed = torch.tensor([int(x) for x in class_mask[t]])
            if not set(torch.unique(y).tolist()) <= set(allowed.tolist()):
                raise SystemExit(f"PAIRING_GATE_FAIL: {npz_path.name} "
                                 f"labels outside class_mask[{t}]")

            zm_fit, zp_fit, y_fit = zm[m_tr], zp[m_tr], y[m_tr]
            zm_ev, zp_ev, y_ev = zm[m_te], zp[m_te], y[m_te]
            n_train = int(m_tr.sum())
            idx_fit = sample_idx[m_tr].tolist()

            row = dict(round=r, client=ci, task=t, entry=entry,
                       n_train=n_train, n_test=int(m_te.sum()),
                       low_power=int(n_train < MIN_FIT_ROWS))
            # ---- 基线 / 参考 ----
            row["acc_off"] = masked_acc(
                ta_logits(zp_ev, key, anchor_pool, hsd), y_ev, allowed)
            row["acc_ref_zm"] = masked_acc(
                ta_logits(zm_ev, key, anchor_pool, hsd), y_ev, allowed)

            zp_ev_d = zp_ev.to(device)
            zm_ev_d = zm_ev.to(device)
            y_ev_d = y_ev.to(device)
            key_d = key.to(device)
            pool_d = anchor_pool.to(device)
            hsd_d = {k: v.to(device) for k, v in hsd.items()}
            allowed_d = allowed.to(device)

            # ---- B 臂 ----
            b_factors = {}
            for rk, gm, guard in B_CONFIGS:
                name = cfg_name_b(rk, gm, guard)
                M1, M2, bias, status, aux = fit_b(
                    zp_fit, zm_fit, idx_fit, gm, rk,
                    guard == "mse")
                row[f"{name}_status"] = status
                for k, v in aux.items():
                    row[f"{name}_{k}"] = v
                if status == "FITTED":
                    b_factors[name] = (M1.to(device), M2.to(device),
                                       bias.to(device))
                    z_c = b_apply(zp_ev_d, M1.to(device), M2.to(device),
                                  bias.to(device))
                    row[f"acc_{name}"] = masked_acc(
                        ta_logits(z_c, key_d, pool_d, hsd_d), y_ev_d,
                        allowed_d)
                else:
                    row[f"acc_{name}"] = row["acc_off"]

            # ---- L 臂 ----
            # 拟合基底：zp = z⁺ train 行（INR-R0 语义）；zm = z⁻ train 行
            # （CIFAR 生产语义）。guard=ce：80% 拟合 + 20% paired CE
            # （不 refit）；guard=no：全行拟合无条件部署。
            fit_mask, cal_mask = split_fit_cal(idx_fit)
            idx_f = torch.tensor(fit_mask)
            idx_c = torch.tensor(cal_mask)
            n_fit, n_cal = int(idx_f.sum()), int(idx_c.sum())
            row["n_lfit"] = n_fit
            row["n_lcal"] = n_cal
            bases = {"zp": zp_fit, "zm": zm_fit}
            for rk, basis, guard in L_CONFIGS:
                name = cfg_name_l(rk, basis, guard)
                if n_train < MIN_FIT_ROWS:
                    row[f"{name}_active"] = 0
                    row[f"acc_{name}"] = row["acc_off"]
                    continue
                if guard == "ce" and (n_fit < MIN_FIT_ROWS
                                      or n_cal < MIN_CAL_ROWS):
                    row[f"{name}_active"] = 0
                    row[f"acc_{name}"] = row["acc_off"]
                    continue
                z_base = bases[basis]
                if guard == "ce":
                    z_fit = z_base[idx_f]
                    y_fit_g = y_fit[idx_f]
                else:
                    z_fit = z_base
                    y_fit_g = y_fit
                seed_l = stable_seed(seed, "probeL", ci, t, r, rk, basis)
                factors = fit_l(z_fit, y_fit_g, key_d, pool_d, hsd_d,
                                allowed_d, nb_classes, rk, seed_l, device,
                                epochs)
                if factors is None:
                    row[f"{name}_active"] = 0
                    row[f"acc_{name}"] = row["acc_off"]
                    continue
                M1, M2, bias = factors
                if guard == "ce":
                    # 部署态 CE guard：20% paired CE（Q2：CE_L < CE_I，
                    # 平局 identity，不 refit）
                    z_cal = zp_fit[idx_c].to(device)
                    y_cal = y_fit[idx_c].to(device)
                    with torch.no_grad():
                        ce_i = float(masked_ce(z_cal, y_cal, key_d, pool_d,
                                               hsd_d, allowed_d, nb_classes))
                        z_l = l_apply(z_cal, M1, M2, bias)
                        ce_l = float(masked_ce(z_l, y_cal, key_d, pool_d,
                                              hsd_d, allowed_d, nb_classes))
                    row[f"{name}_ce_i"] = ce_i
                    row[f"{name}_ce_l"] = ce_l
                    active = int(ce_l < ce_i)
                else:
                    active = 1
                row[f"{name}_active"] = active
                if active:
                    z_c = l_apply(zp_ev_d, M1, M2, bias)
                    row[f"acc_{name}"] = masked_acc(
                        ta_logits(z_c, key_d, pool_d, hsd_d), y_ev_d,
                        allowed_d)
                else:
                    row[f"acc_{name}"] = row["acc_off"]

            # ---- 组合臂（B8 inr mse guard 之后 L16；复核可加性）----
            b_name = cfg_name_b(8, "inr", "mse")
            bfac = b_factors.get(b_name)
            for chain in CHAIN_CONFIGS:
                basis = "zp" if chain.endswith("L16zp") else "zm"
                rk = 16
                if bfac is None or n_train < MIN_FIT_ROWS:
                    # B 被 guard 拦下 -> chain 退化为 L16 单独
                    row[f"{chain}_b_status"] = "IDENTITY_FALLBACK"
                else:
                    row[f"{chain}_b_status"] = "FITTED"
                if bfac is not None:
                    M1b, M2b, bb = bfac
                    zp_b_fit = b_apply(zp_fit.to(device), M1b, M2b, bb)
                    base_fit = zp_b_fit if basis == "zp" else zm_fit.to(device)
                else:
                    base_fit = bases[basis].to(device)
                seed_l = stable_seed(seed, "probeL", ci, t, r, rk,
                                     "chain_" + basis)
                factors = fit_l(base_fit, y_fit, key_d, pool_d, hsd_d,
                                allowed_d, nb_classes, rk, seed_l, device,
                                epochs)
                if factors is None:
                    row[f"acc_{chain}"] = row["acc_off"]
                    continue
                M1, M2, bias = factors
                z_b_ev = (b_apply(zp_ev_d, M1b, M2b, bb) if bfac is not None
                          else zp_ev_d)
                z_c = l_apply(z_b_ev, M1, M2, bias)
                row[f"acc_{chain}"] = masked_acc(
                    ta_logits(z_c, key_d, pool_d, hsd_d), y_ev_d, allowed_d)
            rows.append(row)
        print(f"round {r:4d}: done ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        raise SystemExit("no usable cells")

    # ---------------------------------------------------------- 汇总排名
    acc_cols = (["acc_off", "acc_ref_zm"]
                + [f"acc_{cfg_name_l(*c)}" for c in L_CONFIGS]
                + [f"acc_{cfg_name_b(*c)}" for c in B_CONFIGS]
                + [f"acc_{c}" for c in CHAIN_CONFIGS])
    def _mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    task_end = [r_ for r_ in rows if r_["round"] % 5 == 4]
    per_task = {}
    for r_ in rows:
        per_task.setdefault(r_["task"], []).append(r_)
    configs = {}
    for col in acc_cols:
        name = col[len("acc_"):]
        deltas = [r_[col] - r_["acc_off"] for r_ in rows]
        te_deltas = [r_[col] - r_["acc_off"] for r_ in task_end]
        entry = dict(
            mean_acc=_mean([r_[col] for r_ in rows]),
            mean_delta=_mean(deltas),
            task_end_pooled_delta=_mean(te_deltas),
            n_task_end=len(task_end),
            per_task_delta={str(t): _mean([r_[col] - r_["acc_off"]
                                           for r_ in v])
                            for t, v in sorted(per_task.items())},
            worst_cells=sorted(
                ({"round": r_["round"], "client": r_["client"],
                  "task": r_["task"], "delta": r_[col] - r_["acc_off"]}
                 for r_ in rows), key=lambda d: d["delta"])[:3])
        acol = f"{name}_active"
        if any(acol in r_ for r_ in rows):
            entry["guard_active_rate"] = _mean(
                [r_[acol] for r_ in rows if acol in r_])
        scols = [k for k in rows[0] if k.startswith(f"{name}_status")]
        if scols:
            entry["fit_status"] = {
                s: sum(1 for r_ in rows if r_.get(scols[0]) == s)
                for s in sorted({r_.get(scols[0]) for r_ in rows})}
        configs[name] = entry
    ranked = sorted(
        (k for k in configs if k not in ("off", "ref_zm")),
        key=lambda k: configs[k]["task_end_pooled_delta"], reverse=True)

    summary = dict(
        tool="dsar_inr_scheme_probe", run_dir=str(run_dir),
        n_cells=len(rows), n_task_end=len(task_end),
        zref_gate=dict(passed=not zref_violations,
                       violations=zref_violations[:10]),
        l_recipe=dict(epochs=epochs, batch=L_BATCH, lr=L_LR, wd=L_WD,
                      frozen_production=(epochs == L_EPOCHS)),
        gamma_inr=GAMMA_INR, gamma_cifar=GAMMA_CIFAR,
        ranking=[dict(rank=i + 1, config=k,
                      task_end_pooled_delta=configs[k][
                          "task_end_pooled_delta"],
                      mean_delta=configs[k]["mean_delta"],
                      guard_active_rate=configs[k].get("guard_active_rate"),
                      fit_status=configs[k].get("fit_status"))
                 for i, k in enumerate(ranked)],
        configs=configs,
        notes="offline replay on capture npz; zero training-runs; readout = "
              "npz birth M2' head, eval semantics (dropout off), routing on "
              "corrected z; current-task rows only (historical transfer "
              "needs GPU extract, next stage).")
    print("\n=== INR scheme probe: ranking (task-end pooled delta, pp) ===")
    print(f"{'config':28s} {'te_pooled':>10s} {'mean':>8s} "
          f"{'guard':>7s}")
    print(f"{'[baseline] off':28s} {0.0:10.3f} {0.0:8.3f}")
    print(f"{'[ref] ref_zm':28s} "
          f"{configs['ref_zm']['task_end_pooled_delta']:10.3f} "
          f"{configs['ref_zm']['mean_delta']:8.3f}")
    for k in ranked:
        c = configs[k]
        ga = c.get("guard_active_rate")
        ga_s = f"{ga:7.2f}" if ga is not None else "      -"
        print(f"{k:28s} {c['task_end_pooled_delta']:10.3f} "
              f"{c['mean_delta']:8.3f} {ga_s}")
    print(f"\nz_ref invariance: "
          f"{'PASS' if not zref_violations else 'VIOLATION'}")

    with open(out_dir / "cells.csv", "w", newline="",
              encoding="utf-8") as f:
        fields = sorted({k for r_ in rows for k in r_},
                        key=lambda k: (k not in ("round", "client", "task"),
                                       k))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwritten: {out_dir}/cells.csv, summary.json")


if __name__ == "__main__":
    main()
