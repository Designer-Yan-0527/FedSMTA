"""G2 DSAR 数值签核（RUNCOMMANDS §34.4b 五断言表，B1 formal 解锁前置）。

对 CIFAR M2'-matched capture run 的全部 125 个真实 cell，双侧对账：

  生产侧   : Models/DSAR.py::dsar_fit_bridge（B1 正式实现路径，含 aux
             mse_bridge_r8_cal / mse_identity_cal / lam_fit / lam_refit）
  独立侧   : 本脚本自带的配方复刻（不 import DSAR 的任何实现函数；
             GAMMA_STAR=0.396419 / rank 8 / 32/8 阈值 / 80/20 hash split
             全部本地重写），fit-half solve -> SVD r8 -> b8 -> cal MSE
             -> (胜出才) 全行 refit -> 再截断 -> 再 b8。

五断言（§34.4b 冻结）：
  G2_RANK8_GUARD  独立复刻的 r8 候选 cal-MSE 与 aux mse_bridge_r8_cal
                  一致（相对容差 1e-9）——证明生产 guard 比较的就是
                  实际部署的 r8 候选，而非 full-ridge。
  G2_BIAS8        部署 bias 与独立公式 b8 一致（float32 往返 ≤1e-5）；
                  c_full≠b8 的 cell 上部署 bias 不得等于 c_full（防口径
                  回退）；聚合 max|c_full−b8| > 1e-5 证明口径切换实质性
                  （冒烟证据 2.1e-2）。
  G2_IDENTITY_TIE status==IDENTITY_GUARD ⟺ mse_identity_cal ≤
                  mse_bridge_r8_cal（identity 赢平局），两侧数字各自
                  成立且互证。
  G2_REFIT        FITTED ⇒ 全部 train rows + γ*=0.396419 重拟合
                  （aux lam_refit 与独立重算 λ=N·s·γ* 一致，N=全部
                  train rows），并再次截断 r8 + 重算 b8（M1/M2/bias
                  与独立复刻一致）。
  G2_LOW_POWER    fit rows<32 或 cal rows<8 ⇒ LOW_POWER_IDENTITY
                  （无因子部署 = bias 全清零语义）；反向亦然；非 FITTED
                  不得返回因子；完整性异常必须 raise（本工具以
                  "全部 125 cell 零异常完成"核验）。

附带 pairing gate（fail-closed）：npz manifest 的
(round_id, client_id, task_id, n_train, n_test) 与文件名/行数对账，
沿 §34.4c PAIRING_GATE 先例。

写盘纪律（G1 工程条款）：唯一写盘点 = 输出 verdict JSON（默认
<run_dir>/bsab_b0/dsar_g2_signoff.json），不触碰任何封板文件。

数值：CPU torch float64（本机 numpy/scipy BLAS 损坏，ENVIRONMENT_AUDIT）。
用法（repo 根目录，CPU）：
  python diagnostics/dsar_g2_signoff.py
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.DSAR import dsar_fit_bridge, DSAR_VERSION  # noqa: E402  (生产侧)

FEAT_DIM = 768
NPZ_NAME = re.compile(
    r"^bsab_c(\d+)_t(\d+)_r(\d+)_(broadcast|broadcast_no_test)\.npz$")

# ---- 独立侧冻结常数（与 Models/DSAR.py 刻意分开维护，防同源抄写）----
GAMMA_STAR_IND = 0.396419   # §34.4c corrected export (full: 0.396419312045252)
RANK_IND = 8
MIN_FIT_IND = 32
MIN_CAL_IND = 8

REL_TOL = 1e-9              # G2_RANK8_GUARD / G2_REFIT(λ) 相对容差
BIAS_TOL = 1e-5             # G2_BIAS8 float32 往返容差
FACTOR_TOL = 1e-6           # M1/M2 独立复刻一致性容差（期望逐位相等）
CFULL_SEP = 1e-5            # c_full 与 b8 视为可分辨的阈值


def parse_args():
    p = argparse.ArgumentParser("G2 DSAR numerical signoff")
    p.add_argument("--run_dir", default=(
        "output/cifar100/fedRelink/cifar_bsab_capture_M2matched/seed_42"))
    p.add_argument("--out_json", default=None,
                   help="default: <run_dir>/bsab_b0/dsar_g2_signoff.json")
    return p.parse_args()


def stable_seed(*parts):
    key = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],
                          "little")


# ------------------------------------------------- 独立侧配方复刻

def ind_split_fit_cal(sample_idx):
    """80/20 fit/cal，规则本地重写：sha256('dsar_split:<idx>') 前 4 字节
    little-endian，h%5==0 -> cal。不消费 RNG。"""
    fit, cal = [], []
    for idx in sample_idx:
        h = int.from_bytes(
            hashlib.sha256(("dsar_split:%d" % int(idx)).encode())
            .digest()[:4], "little")
        cal.append(h % 5 == 0)
        fit.append(h % 5 != 0)
    return fit, cal


def ind_solve(zp_rows, zm_rows):
    """一次 fixed-lambda residual-ridge solve（float64，独立表达式）。
    返回 (dA, xbar, ybar, lam, S)；S 供尾能量统计。"""
    X = zp_rows.double()
    Y = zm_rows.double()
    n, d = X.shape
    xbar, ybar = X.mean(dim=0), Y.mean(dim=0)
    xc, yc = X - xbar, Y - ybar
    s = float((xc * xc).sum()) / float(n * d)
    lam = float(n) * s * GAMMA_STAR_IND
    dA = torch.linalg.solve(xc.t() @ xc
                            + torch.eye(d, dtype=torch.float64) * lam,
                            xc.t() @ (yc - xc))
    if not bool(torch.isfinite(dA).all()):
        raise RuntimeError("ind_solve: non-finite solution")
    U_, S_, Vt_ = torch.linalg.svd(dA, full_matrices=False)
    return dA, xbar, ybar, lam, S_


def ind_trunc8(dA):
    """SVD 截断 r8：dA_8 = M1 @ M2^T（M1 = U_r*S_r, M2 = V_r）。"""
    U_, S_, Vt_ = torch.linalg.svd(dA, full_matrices=False)
    return U_[:, :RANK_IND] * S_[:RANK_IND], Vt_[:RANK_IND].t().contiguous()


def ind_bias8(xbar, ybar, M1, M2):
    """b8 = mu_y - mu_x - mu_x @ (M1 @ M2^T)（截断算子下重算）。"""
    return ybar - xbar - xbar @ (M1 @ M2.t())


def ind_mse_r8(zp_rows, zm_rows, M1, M2, b8):
    Z = zp_rows.double()
    out = Z + (Z @ M1) @ M2.t() + b8
    resid = out - zm_rows.double()
    return float((resid * resid).mean())


def ind_fit_cell(zp_tr, zm_tr, sample_idx_tr):
    """独立复刻五步流程。返回 dict（status/mse/lambda/因子/c_full 等）。"""
    fit, cal = ind_split_fit_cal(sample_idx_tr)
    n_fit, n_cal = int(sum(fit)), int(sum(cal))
    out = {"n_fit": n_fit, "n_cal": n_cal}
    if n_fit < MIN_FIT_IND or n_cal < MIN_CAL_IND:
        out["status"] = "LOW_POWER_IDENTITY"
        return out
    zp_fit, zm_fit = zp_tr[fit], zm_tr[fit]
    zp_cal, zm_cal = zp_tr[cal], zm_tr[cal]
    # -- guard：fit-half solve -> r8 -> b8 -> cal MSE --
    dA_g, xbar_g, ybar_g, lam_g, S_g = ind_solve(zp_fit, zm_fit)
    M1_g, M2_g = ind_trunc8(dA_g)
    b_g = ind_bias8(xbar_g, ybar_g, M1_g, M2_g)
    mse_r8 = ind_mse_r8(zp_cal, zm_cal, M1_g, M2_g, b_g)
    diff = zp_cal.double() - zm_cal.double()
    mse_id = float((diff * diff).mean())
    out.update({"lam_fit": lam_g, "mse_bridge_r8_cal": mse_r8,
                "mse_identity_cal": mse_id})
    if mse_id <= mse_r8:
        out["status"] = "IDENTITY_GUARD"
        return out
    # -- 胜出：全部 train rows refit -> 再截断 -> 再 b8 --
    dA_f, xbar_f, ybar_f, lam_f, S_f = ind_solve(zp_tr, zm_tr)
    M1_f, M2_f = ind_trunc8(dA_f)
    b8_f = ind_bias8(xbar_f, ybar_f, M1_f, M2_f)
    # 全秩截距（旧口径，仅用于 G2_BIAS8 口径回退检查，不部署）
    c_full = ybar_f - xbar_f - xbar_f @ dA_f
    out.update({
        "status": "FITTED", "lam_refit": lam_f,
        "M1": M1_f.float(), "M2": M2_f.float(), "bias": b8_f.float(),
        "c_full": c_full,
        "tail_energy_ratio": float(
            torch.linalg.vector_norm(S_f[RANK_IND:])
            / torch.linalg.vector_norm(S_f)),
    })
    return out


# ------------------------------------------------------------------ 主流程

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


def rel_err(a, b):
    return abs(a - b) / max(1.0, abs(b))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_json = (Path(args.out_json) if args.out_json
                else run_dir / "bsab_b0" / "dsar_g2_signoff.json")

    with open(run_dir / "args.json", encoding="utf-8") as f:
        ra = json.load(f)
    seed = int(ra.get("seed", 42))

    cells = discover_cells(run_dir)
    print(f"cells: {len(cells)} | production: Models/DSAR.py "
          f"({DSAR_VERSION}) | independent: local replica "
          f"(gamma*={GAMMA_STAR_IND}, rank={RANK_IND})")

    rows = []
    integrity_raises = []
    t0 = time.time()
    for (r, ci) in sorted(cells.keys()):
        t, entry, npz_path = cells[(r, ci)]
        data = np.load(npz_path)
        # pairing gate（fail-closed）
        mf = json.loads(npz_path.with_suffix(".json").read_text())
        if (int(mf["round_id"]) != r or int(mf["client_id"]) != ci
                or int(mf["task_id"]) != t):
            raise SystemExit(f"PAIRING_GATE_FAIL: manifest mismatch {npz_path}")
        m_tr = data["split"] == 0
        if int(m_tr.sum()) != int(mf["n_train"]) or \
                int((data["split"] == 1).sum()) != int(mf["n_test"]):
            raise SystemExit(f"PAIRING_GATE_FAIL: row counts {npz_path}")

        zp_tr = torch.tensor(data["z_op_post"][m_tr])
        zm_tr = torch.tensor(data["z_op_pre"][m_tr])
        idx_tr = data["sample_index"][m_tr]

        # -- 生产侧（B 路径 RNG-free；seed 仅 provenance 记录）--
        p_seed = stable_seed(seed, "dsar_g2", ci, t, r)
        try:
            factors_p, status_p, aux_p = dsar_fit_bridge(
                zp_tr, zm_tr, idx_tr, p_seed)
        except RuntimeError as e:
            integrity_raises.append((r, ci, str(e)[:120]))
            continue

        # -- 独立侧 --
        ind = ind_fit_cell(zp_tr, zm_tr, idx_tr)

        row = dict(round=r, client=ci, task=t, entry=entry,
                   n_train=int(m_tr.sum()),
                   n_fit=ind["n_fit"], n_cal=ind["n_cal"],
                   status_prod=status_p, status_ind=ind["status"])
        guard_reached = status_p != "LOW_POWER_IDENTITY"
        if guard_reached:
            row["mse_r8_prod"] = aux_p["mse_bridge_r8_cal"]
            row["mse_r8_ind"] = ind["mse_bridge_r8_cal"]
            row["mse_id_prod"] = aux_p["mse_identity_cal"]
            row["mse_id_ind"] = ind["mse_identity_cal"]
            row["lam_fit_prod"] = aux_p["lam_fit"]
            row["lam_fit_ind"] = ind["lam_fit"]
        if status_p == "FITTED":
            row["lam_refit_prod"] = aux_p["lam_refit"]
            row["lam_refit_ind"] = ind["lam_refit"]
            row["bias_diff"] = float(
                (factors_p["bias"] - ind["bias"]).abs().max())
            row["M1_diff"] = float(
                (factors_p["M1"] - ind["M1"]).abs().max())
            row["M2_diff"] = float(
                (factors_p["M2"] - ind["M2"]).abs().max())
            row["c_full_minus_b8_max"] = float(
                (ind["c_full"].double()
                 - ind["bias"].double()).abs().max())
            row["tail_energy_ratio"] = ind["tail_energy_ratio"]
            row["bias_vs_c_full"] = float(
                (factors_p["bias"].double()
                 - ind["c_full"].float().double()).abs().max())  # noqa: E501
        rows.append(row)
        if ci == 4:
            print(f"  round {r:4d} done ({time.time()-t0:.0f}s)", flush=True)

    if len(rows) != len(cells):
        raise SystemExit(f"processed {len(rows)}/{len(cells)} cells "
                         f"(integrity raises: {integrity_raises})")

    # ------------------------------------------------ 断言汇总
    fails = {k: [] for k in ("STATUS_MATCH", "G2_RANK8_GUARD", "G2_BIAS8",
                             "G2_IDENTITY_TIE", "G2_REFIT", "G2_LOW_POWER")}
    n_status = {"FITTED": 0, "IDENTITY_GUARD": 0, "LOW_POWER_IDENTITY": 0}
    for row in rows:
        key = f"r{row['round']:02d}c{row['client']}"
        n_status[row["status_prod"]] = \
            n_status.get(row["status_prod"], 0) + 1
        if row["status_prod"] != row["status_ind"]:
            fails["STATUS_MATCH"].append(key)
        # G2_RANK8_GUARD（含 guard 结构证明：独立 r8 候选数值 == aux）
        if row["status_prod"] != "LOW_POWER_IDENTITY":
            if rel_err(row["mse_r8_ind"], row["mse_r8_prod"]) > REL_TOL:
                fails["G2_RANK8_GUARD"].append(key)
        # G2_BIAS8
        if row["status_prod"] == "FITTED":
            if row["bias_diff"] > BIAS_TOL:
                fails["G2_BIAS8"].append(key + "(bias!=b8)")
            if (row["c_full_minus_b8_max"] > CFULL_SEP
                    and row["bias_vs_c_full"] <= BIAS_TOL):
                fails["G2_BIAS8"].append(key + "(bias==c_full caliber)")
        # G2_IDENTITY_TIE（identity 赢平局；两侧各自成立且互证）
        if row["status_prod"] != "LOW_POWER_IDENTITY":
            tie_p = (row["mse_id_prod"] <= row["mse_r8_prod"])
            tie_i = (row["mse_id_ind"] <= row["mse_r8_ind"])
            is_guard = row["status_prod"] == "IDENTITY_GUARD"
            if is_guard != tie_p or is_guard != tie_i:
                fails["G2_IDENTITY_TIE"].append(key)
        # G2_REFIT（全行 N + 同 γ*；M1/M2/bias == 独立复刻）
        if row["status_prod"] == "FITTED":
            if rel_err(row["lam_refit_ind"], row["lam_refit_prod"]) > REL_TOL:
                fails["G2_REFIT"].append(key + "(lam)")
            if (row["M1_diff"] > FACTOR_TOL or row["M2_diff"] > FACTOR_TOL
                    or row["bias_diff"] > BIAS_TOL):
                fails["G2_REFIT"].append(key + "(factors)")
        # G2_LOW_POWER（双向 + 非 FITTED 无因子）
        low_rows = row["n_fit"] < MIN_FIT_IND or row["n_cal"] < MIN_CAL_IND
        if low_rows != (row["status_prod"] == "LOW_POWER_IDENTITY"):
            fails["G2_LOW_POWER"].append(key)
        if row["status_prod"] != "FITTED" and "lam_refit_prod" in row:
            fails["G2_LOW_POWER"].append(key + "(factors leaked)")

    fitted = [r_ for r_ in rows if r_["status_prod"] == "FITTED"]
    max_cfull_gap = (max((r_["c_full_minus_b8_max"] for r_ in fitted),
                         default=0.0) if fitted else 0.0)
    if fitted and max_cfull_gap <= CFULL_SEP:
        fails["G2_BIAS8"].append("AGGREGATE(c_full==b8 everywhere: "
                                 "caliber switch not substantive)")

    n_integrity = len(integrity_raises)
    n_low = n_status.get("LOW_POWER_IDENTITY", 0)
    if n_integrity:
        fails["G2_LOW_POWER"].append(
            f"AGGREGATE({n_integrity} integrity raises — must raise, "
            f"not downgrade; investigate before signoff)")

    assertions = {
        "G2_RANK8_GUARD": dict(
            passed=not fails["G2_RANK8_GUARD"] and not fails["STATUS_MATCH"],
            tol_rel=REL_TOL, n_cells=len(rows) - n_low,
            max_rel_err=max((rel_err(r_["mse_r8_ind"], r_["mse_r8_prod"])
                             for r_ in rows
                             if r_["status_prod"] != "LOW_POWER_IDENTITY"),
                            default=0.0),
            failures=fails["G2_RANK8_GUARD"][:20]),
        "G2_BIAS8": dict(
            passed=not fails["G2_BIAS8"],
            bias_tol=BIAS_TOL, n_fitted=len(fitted),
            max_bias_diff=max((r_["bias_diff"] for r_ in fitted), default=0.0),
            max_c_full_minus_b8=max_cfull_gap,
            n_cfull_distinguishable=sum(
                1 for r_ in fitted if r_["c_full_minus_b8_max"] > CFULL_SEP),
            failures=fails["G2_BIAS8"][:20]),
        "G2_IDENTITY_TIE": dict(
            passed=not fails["G2_IDENTITY_TIE"],
            n_guard_cells=len(rows) - n_low - len(fitted),
            failures=fails["G2_IDENTITY_TIE"][:20]),
        "G2_REFIT": dict(
            passed=not fails["G2_REFIT"],
            gamma_star=GAMMA_STAR_IND, n_fitted=len(fitted),
            max_lam_rel_err=max((rel_err(r_["lam_refit_ind"],
                                         r_["lam_refit_prod"])
                                 for r_ in fitted), default=0.0),
            max_M1_diff=max((r_["M1_diff"] for r_ in fitted), default=0.0),
            max_M2_diff=max((r_["M2_diff"] for r_ in fitted), default=0.0),
            failures=fails["G2_REFIT"][:20]),
        "G2_LOW_POWER": dict(
            passed=not fails["G2_LOW_POWER"] and n_integrity == 0,
            min_fit_rows=MIN_FIT_IND, min_cal_rows=MIN_CAL_IND,
            n_low_power=n_low, n_integrity_raises=n_integrity,
            integrity_raise_cells=[list(map(str, x[:2]))
                                   for x in integrity_raises[:10]],
            failures=fails["G2_LOW_POWER"][:20]),
    }
    verdict = "PASS" if (all(a["passed"] for a in assertions.values())
                         and not fails["STATUS_MATCH"]) else "FAIL"

    out = dict(
        gate="G2", verdict=verdict,
        run_dir=str(run_dir), production_version=DSAR_VERSION,
        independent="local replica (diagnostics/dsar_g2_signoff.py); "
                    "no DSAR implementation imports",
        gamma_star=GAMMA_STAR_IND, rank=RANK_IND,
        n_cells=len(rows),
        status_counts={k: n_status.get(k, 0) for k in
                       ("FITTED", "IDENTITY_GUARD", "LOW_POWER_IDENTITY")},
        status_match_failures=fails["STATUS_MATCH"][:20],
        assertions=assertions,
        cells=rows,
        notes="Sec.34.4b frozen five-assertion table; production aux from "
              "Models/DSAR.py dsar_fit_bridge on real M2'-matched capture "
              "train rows (split==0); torch CPU float64.",
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n=== G2 DSAR signoff ({len(rows)} cells) ===")
    print(f"status counts: {out['status_counts']}")
    for name, a in assertions.items():
        print(f"{name}: {'PASS' if a['passed'] else 'FAIL'}")
    print(f"max bias |dep-b8|      = "
          f"{assertions['G2_BIAS8']['max_bias_diff']:.3e}")
    print(f"max |c_full-b8|       = {max_cfull_gap:.3e} "
          f"(caliber switch substance)")
    print(f"max lam_refit rel err = "
          f"{assertions['G2_REFIT']['max_lam_rel_err']:.3e}")
    print(f"VERDICT: {verdict}")
    print(f"written: {out_json}")


if __name__ == "__main__":
    main()
