"""B1 预注册前置：从已封板 B0 artifact 导出固定 ridge 归一化常数 γ*。

§34.4 B1 preregistration 冻结条款 2：B_t 在线不跑 GCV，λ_{c,r} =
N_{c,r}·s_{c,r}·γ*，其中
  s_{c,r} = tr(X^T X)/(N·d)，X = 当前 cell 的 z_op_post train 行中心化
  γ* = median_i λ_i^GCV/(N_i·s_i)，i ∈ B0 全部 nonidentity cell
只读 B0 已落盘产物（cells.csv + capture npz），不触碰任何训练代码。

2026-11-06 勘误（ERRATUM-gamma-split，§34.6 Stage 0 复审发现）：
原版以 `split == 1` 取行 —— 与 capture writer 编码（("train", 0),
("test", 1)，Client_DF._bsab_forward_rows）相反，s 误建立在 test 行上。
本版已修正为 split == 0。对 cifar B0 产物的影响已量化：
γ* = 0.390324（test 行，即现行冻结值）→ 0.396419（train 行），−1.5%，
处于 cell 间 spread（p25≈0.37 p75≈0.65）之内。B1 cifar 冻结常数的
处置（保留 0.390324 或修订 0.396419）属预注册变更，待用户/GPT 裁决；
本脚本仅保证此后所有导出（含 γ*_INR）使用正确行集。

用法（repo 根目录，CPU；run_dir 为含 bsab_capture/ 与 cells.csv 的
seed 目录）：
  python diagnostics/bsab_gamma_star.py --run_dir <seed_dir> \
      [--cells bsab_b0/cells.csv | dsar_birth_preview/cells.csv] \
      [--out <seed_dir>/bsab_b0/gamma_star_inr.json]

cells.csv 列名自动兼容两套产物：B0 offline pilot（bridge_identity /
bridge_lam）与 dsar_birth_preview（bt_identity / bt_lam）。GCV 网格
边界命中仅在 cells.csv 提供 rr_gcv_expo 列（Stage 0 扩展 preview）时
可报告；boundary=1 表示最优 λ 落在网格端点（10^-3 或 10^4 相对尺度）。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FEAT_DIM = 768


def parse_args():
    p = argparse.ArgumentParser("gamma* export (frozen GCV-median rule)")
    p.add_argument("--run_dir", required=True,
                   help="seed dir containing bsab_capture/ and cells.csv")
    p.add_argument("--cells", default="bsab_b0/cells.csv",
                   help="cells.csv path (absolute or relative to run_dir)")
    p.add_argument("--out", default=None,
                   help="output json (default <run_dir>/bsab_b0/"
                        "gamma_star.json)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cells_path = Path(args.cells)
    if not cells_path.is_absolute():
        cells_path = run_dir / cells_path
    out_path = Path(args.out) if args.out else (
        run_dir / "bsab_b0" / "gamma_star.json")

    with open(cells_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 列名兼容：B0 pilot (bridge_*) / birth preview (bt_*)
    first = rows[0] if rows else {}
    if "bridge_identity" in first:
        col_id, col_lam = "bridge_identity", "bridge_lam"
    elif "bt_identity" in first:
        col_id, col_lam = "bt_identity", "bt_lam"
    else:
        raise SystemExit(
            f"CELLS_GATE_FAIL: unrecognized columns in {cells_path}: "
            f"{sorted(first)}")

    gammas, skipped, boundary_hits, total_nonid = [], 0, 0, 0
    for row in rows:
        if int(row[col_id]) == 1:
            skipped += 1  # identity 胜出 cell 的 λ 不代表 ridge 强度
            continue
        total_nonid += 1
        lam = float(row[col_lam])
        n_train = int(row["n_train"])
        npz = (run_dir / "bsab_capture" /
               f"bsab_c{int(row['client'])}_t{int(row['task'])}"
               f"_r{int(row['round']):04d}_{row['entry']}.npz")
        data = np.load(npz)
        m_tr = data["split"] == 0  # writer 编码：0=train, 1=test（勘误后）
        if int(m_tr.sum()) != n_train:
            raise SystemExit(
                f"PAIRING_GATE_FAIL: {npz.name} split==0 rows "
                f"{int(m_tr.sum())} != cells.csv n_train {n_train}")
        X = data["z_op_post"][m_tr].astype(np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        s = float((X * X).sum() / (X.shape[0] * FEAT_DIM))
        gammas.append(lam / (n_train * s))
        # GCV 网格边界命中（§34.6 条件 B）：优先 Stage 0 扩展 preview 的
        # rr_gcv_expo，否则 b0_verdict 的 bridge_lam_expo（additive 列，
        # 2026-11-06）；boundary=1 表示最优 λ 落网格端点（10^-3 或 10^4
        # 相对尺度）——大量命中说明当前网格无法可靠确定 γ*，导出方须报
        # GCV_GRID_BOUNDARY_WARNING 而非直接采信 median。
        expo_raw = row.get("rr_gcv_expo") or row.get("bridge_lam_expo") or ""
        if expo_raw:
            expo = float(expo_raw)
            if expo != expo:  # nan = identity 网格无意义
                pass
            elif expo <= -3.0 + 1e-9 or expo >= 4.0 - 1e-9:
                boundary_hits += 1

    if not gammas:
        raise SystemExit("no nonidentity cells to calibrate gamma*")

    gammas = np.array(gammas)
    out = dict(
        split_encoding="train=0 (ERRATUM-gamma-split 2026-11-06)",
        cells_csv=str(cells_path),
        n_cells_total=len(rows), skipped_identity=skipped,
        n_nonidentity=total_nonid,
        gamma_median=float(np.median(gammas)),
        gamma_mean=float(gammas.mean()),
        gamma_min=float(gammas.min()), gamma_max=float(gammas.max()),
        gamma_p25=float(np.percentile(gammas, 25)),
        gamma_p75=float(np.percentile(gammas, 75)),
        gcv_boundary_hits=boundary_hits,
        gcv_boundary_reported=bool(
            rows and (rows[0].get("rr_gcv_expo")
                      or rows[0].get("bridge_lam_expo"))),
        # §34.6 条件 B：命中数超过 nonidentity 的 20% 即挂警告——此时
        # median 不可直接采信，须先修订校准协议再冻结 γ*。
        gcv_grid_boundary_warning=bool(
            boundary_hits > max(0, total_nonid // 5)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"nonidentity cells: {total_nonid}/{len(rows)} "
          f"(skipped identity: {skipped})")
    print(f"gamma* (median)  : {out['gamma_median']:.6g}")
    print(f"IQR              : p25={out['gamma_p25']:.3g} "
          f"p75={out['gamma_p75']:.3g} "
          f"min={out['gamma_min']:.3g} max={out['gamma_max']:.3g}")
    if out["gcv_boundary_reported"]:
        print(f"GCV grid boundary hits: {boundary_hits}/{total_nonid}"
              + ("  ** GCV_GRID_BOUNDARY_WARNING **"
                 if out["gcv_grid_boundary_warning"] else ""))
    else:
        print("GCV grid boundary hits: n/a "
              "(cells.csv lacks rr_gcv_expo/bridge_lam_expo; use Stage 0 "
              "preview or post-2026-11-06 b0_verdict output)")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
