"""INR 历史任务行 DSAR-B 离线探针（验证缺口 2：B 的历史迁移增益，
baseline 联训头口径 = 论文部署口径）。

生产部署图（Models/DSAR.py §34.4a）历史任务 k < t：

    y_k = TA_k(B_{c,r}(z_op+))          （B-only，L 禁用于历史任务）

  - z_op+   = 任务 k 的 test 行过 task t end（round r=5t+4，广播后）
              checkpoint 状态、当前任务 t 的 prompt 前向（=
              Client_DF.evaluate 的 task_id=self.task_id 语义），由
              extract_features.py --prompt_task_mode current --split test
              提取（stage 1，GPU）
  - B_{c,r} = 在当前任务 t 的 broadcast boundary r 的 capture npz
              （z_op_post -> z_op_pre train 行对）上拟合，mse guard
              （rank/γ 参数化，fit_b 与 dsar_inr_scheme_probe 同实现）
  - TA_k     = heads[k]（checkpoint per-task Chead）+ 当前 key/anchor
              路由（cstate["tail_anchor_model"]）+ class_mask[k]
              （§34.4a 不变式 3：head 选择保持 heads[k] + 原 key
              路由 + class mask；修正后 z 的锚定槽位允许自然改变）

两级流水（izfree；GPU 只在 stage 1）：

  stage 1  每个 task-end checkpoint 一次（task_01..task_04_end；
           task_00 无历史任务）：
             python diagnostics/extract_features.py \
                 --run_dir <run_dir> \
                 --checkpoint <run_dir>/checkpoints/task_01_end.pth \
                 --prompt_task_mode current --split test \
                 --out_dir <run_dir>/dsar_hist_feats
  stage 2  本工具（CPU 闭式拟合 + 读出）。

口径说明：eval 特征来自 stage-1 确定性 extract 协议（bs=16），
B 拟合来自 capture npz 的 boundary 特征——两者协议同族但非逐位同源
（capture 在训练进程内提取），这是离线 replay 的固有口径；结论按
“读出侧修复的结构增益”解释。历史行无预注册门（+0.8pp 参考门是
scheme probe 当前任务行口径）；本工具出数后交 GPT 裁决冻结。
当前任务行（k=t）作为对照一并落盘：其 mean delta 应与 scheme probe
的 te_pooled 大致同向（不同 eval 协议，不做逐位对账）。

用法（repo 根目录）：
  python diagnostics/dsar_inr_hist_probe.py \
      --run_dir output/ImageNet-R/fedta/inr_bsab_capture_baseline_p40/seed_42 \
      --b_ranks 8,24 --b_gammas inr,cifar
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from dsar_inr_scheme_probe import (
    CHEAD_KEYS,
    NPZ_NAME,                       # noqa: F401  (import 供审计对账)
    b_apply,
    cfg_name_b,
    discover_cells,
    fit_b,
    masked_acc,
    parse_b_gammas,
    ta_logits,
)


def parse_args():
    p = argparse.ArgumentParser("INR historical-task DSAR-B probe")
    p.add_argument("--run_dir", default=(
        "output/ImageNet-R/fedta/inr_bsab_capture_baseline_p40/seed_42"))
    p.add_argument("--feat_dir", default=None,
                   help="stage-1 extract 目录；default <run_dir>/dsar_hist_feats")
    p.add_argument("--out_dir", default=None,
                   help="default: <run_dir>/dsar_hist_probe")
    p.add_argument("--b_ranks", default="8,24",
                   help="B rank 列表（逗号分隔）")
    p.add_argument("--b_gammas", default="inr,cifar",
                   help="B γ 模式列表（inr/cifar 或固定值；同 scheme probe）")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    feat_dir = (Path(args.feat_dir) if args.feat_dir
                else run_dir / "dsar_hist_feats")
    out_dir = (Path(args.out_dir) if args.out_dir
               else run_dir / "dsar_hist_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- run 级协议门：必须是 baseline 联训头 pair-capture run ----
    with open(run_dir / "args.json", encoding="utf-8") as f:
        ra = json.load(f)
    if not bool(ra.get("bsab_pair_capture")):
        raise SystemExit("PROTOCOL_GATE_FAIL: not a pair-capture run")
    if str(ra.get("refit_timing", "local")) != "local":
        raise SystemExit("PROTOCOL_GATE_FAIL: unsupported refit_timing")
    if bool(ra.get("head_refit")):
        raise SystemExit("PROTOCOL_GATE_FAIL: M2'-matched capture; this "
                         "probe requires the baseline_joint (paper) readout")
    print("capture run type: baseline_joint")

    ranks = [int(x) for x in args.b_ranks.split(",") if x.strip()]
    if not ranks:
        raise SystemExit("empty --b_ranks")
    gammas = parse_b_gammas(args.b_gammas)
    b_configs = [(rk, gm) for rk in ranks for gm in gammas]
    b_names = [cfg_name_b(rk, gm, "mse") for rk, gm in b_configs]

    cells = discover_cells(run_dir)
    task_end_rounds = sorted({r for r, _ in cells if r % 5 == 4})
    print(f"cells: {len(cells)} | task-end rounds: {task_end_rounds}")

    rows = []
    zref_violations = []
    missing_extracts = []
    t0 = time.time()
    for r_te in task_end_rounds:
        cs_here = sorted(ci for rr, ci in cells if rr == r_te)
        tasks_here = {cells[(r_te, ci)][0] for ci in cs_here}
        if len(tasks_here) != 1:
            raise SystemExit(f"SYNC_GATE_FAIL: r{r_te} clients on mixed "
                             f"tasks {sorted(tasks_here)}")
        t = tasks_here.pop()
        if t == 0:
            continue
        matches = sorted(feat_dir.glob(
            f"features_*_task_{t:02d}_end_promptcurrent.npz"))
        if len(matches) != 1:
            missing_extracts.append(t)
            print(f"round {r_te} (task {t}): stage-1 extract npz missing "
                  f"under {feat_dir} -- skip")
            continue
        ck = torch.load(run_dir / "checkpoints" / f"round_{r_te:04d}.pth",
                        map_location="cpu")
        feats = np.load(matches[0])
        zop = torch.tensor(feats["z_op"]).float()
        f_ci = torch.tensor(feats["client_ids"]).to(torch.int64)
        f_t = torch.tensor(feats["task_ids"]).to(torch.int64)
        f_sp = torch.tensor(feats["splits"]).to(torch.int64)
        f_y = torch.tensor(feats["labels"]).to(torch.int64)
        f_r = torch.tensor(feats["round_ids"]).to(torch.int64)
        if zop.shape[1] != 768:
            raise SystemExit(f"STRUCT_GATE_FAIL: z_op dim {zop.shape[1]}")
        if set(torch.unique(f_r).tolist()) != {r_te}:
            raise SystemExit(
                f"PAIRING_GATE_FAIL: extract round_ids {sorted(set(torch.unique(f_r).tolist()))} "
                f"!= task-end round {r_te}（stage-1 checkpoint 与该 boundary 不匹配）")

        for ci in cs_here:
            cstate = ck["clients"][ci]
            if int(cstate.get("task_id", -1)) != t:
                raise SystemExit(f"PAIRING_GATE_FAIL: r{r_te} c{ci} ckpt "
                                 f"task {cstate.get('task_id')} != npz task {t}")
            tam = cstate["tail_anchor_model"]
            key = tam["key"].float()
            anchor_pool = tam["anchor_pool"].float()
            heads = cstate["heads"]
            class_mask = cstate["class_mask"]

            # ---- B 拟合（当前任务 boundary 对，guard=mse 固定）----
            _, entry, npz_path = cells[(r_te, ci)]
            data = np.load(npz_path)
            if not torch.equal(torch.tensor(data["z_ref_pre"]),
                               torch.tensor(data["z_ref_post"])):
                zref_violations.append((r_te, ci))
            if "sample_index" not in data.files:
                raise SystemExit(f"STRUCT_GATE_FAIL: {npz_path.name} has no "
                                 "sample_index")
            m_tr = torch.tensor(data["split"] == 0)
            zp_fit = torch.tensor(data["z_op_post"]).float()[m_tr]
            zm_fit = torch.tensor(data["z_op_pre"]).float()[m_tr]
            sample_idx = data["sample_index"][m_tr].tolist()
            fits = {}
            for rk, gm in b_configs:
                name = cfg_name_b(rk, gm, "mse")
                fits[name] = fit_b(zp_fit, zm_fit, sample_idx, gm, rk, True)

            # ---- 历史任务行 k<t（主）+ 当前任务行 k=t（对照）----
            for k in range(t + 1):
                hsd = heads[k]
                if hsd is None:
                    raise SystemExit(f"STRUCT_GATE_FAIL: r{r_te} c{ci} "
                                     f"heads[{k}] is None")
                if set(hsd) != CHEAD_KEYS:
                    raise SystemExit(f"STRUCT_GATE_FAIL: r{r_te} c{ci} "
                                     f"heads[{k}] keys {sorted(hsd)}")
                mask = (f_ci == ci) & (f_t == k) & (f_sp == 1)
                n_test = int(mask.sum())
                if n_test == 0:
                    continue
                zp_ev, y_ev = zop[mask], f_y[mask]
                allowed = torch.tensor([int(x) for x in class_mask[k]])
                if not set(torch.unique(y_ev).tolist()) <= set(allowed.tolist()):
                    raise SystemExit(f"PAIRING_GATE_FAIL: extract labels "
                                     f"outside class_mask[{k}] (r{r_te} c{ci})")
                row = dict(round=r_te, client=ci, cur_task=t, hist_task=k,
                           is_hist=int(k < t), distance=t - k,
                           n_test=n_test)
                row["acc_off"] = masked_acc(
                    ta_logits(zp_ev, key, anchor_pool, hsd), y_ev, allowed)
                for rk, gm in b_configs:
                    name = cfg_name_b(rk, gm, "mse")
                    M1, M2, bias, status, _aux = fits[name]
                    row[f"{name}_status"] = status
                    if M1 is not None:
                        z_b = ta_logits(b_apply(zp_ev, M1, M2, bias),
                                        key, anchor_pool, hsd)
                        row[f"acc_{name}"] = masked_acc(z_b, y_ev, allowed)
                    else:
                        row[f"acc_{name}"] = row["acc_off"]
                rows.append(row)
        print(f"round {r_te:4d} (task {t}): done ({time.time()-t0:.0f}s)",
              flush=True)

    if missing_extracts:
        raise SystemExit(
            "EXTRACT_GATE_FAIL: stage-1 extracts missing for tasks "
            f"{sorted(set(missing_extracts))}; run extract_features.py per "
            "task-end checkpoint first (see module docstring)")
    if not rows:
        raise SystemExit("no usable cells")

    # ---------------------------------------------------------- 汇总
    hist = [r_ for r_ in rows if r_["is_hist"]]
    cur = [r_ for r_ in rows if not r_["is_hist"]]
    if not hist:
        raise SystemExit("no historical cells (task 0 only?)")

    def _mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    per_dist = {}
    for r_ in hist:
        per_dist.setdefault(r_["distance"], []).append(r_)
    configs = {}
    for name in ["off"] + b_names:
        col = "acc_off" if name == "off" else f"acc_{name}"
        dh = [r_[col] - r_["acc_off"] for r_ in hist]
        entry = dict(
            mean_acc_hist=_mean([r_[col] for r_ in hist]),
            mean_delta_hist=_mean(dh) if name != "off" else 0.0,
            n_hist=len(hist),
            pos_fraction_hist=(float(np.mean([d > 0 for d in dh]))
                               if name != "off" else None),
            per_distance_delta_hist=(
                {str(d): _mean([r_[col] - r_["acc_off"] for r_ in v])
                 for d, v in sorted(per_dist.items())}
                if name != "off" else
                {str(d): _mean([r_[col] for r_ in v])
                 for d, v in sorted(per_dist.items())}),
            mean_acc_cur_ctrl=_mean([r_[col] for r_ in cur]),
            mean_delta_cur_ctrl=( _mean([r_[col] - r_["acc_off"] for r_ in cur])
                                  if name != "off" else 0.0),
            worst_hist_cells=(None if name == "off" else sorted(
                ({"round": r_["round"], "client": r_["client"],
                  "cur_task": r_["cur_task"], "hist_task": r_["hist_task"],
                  "delta": r_[col] - r_["acc_off"]} for r_ in hist),
                key=lambda d: d["delta"])[:5]),
        )
        scol = f"{name}_status"
        if name != "off" and scol in rows[0]:
            entry["fit_status"] = {
                s: sum(1 for r_ in rows if r_.get(scol) == s)
                for s in sorted({r_.get(scol) for r_ in rows})}
        configs[name] = entry
    ranked = sorted((k for k in configs if k != "off"),
                    key=lambda k: configs[k]["mean_delta_hist"], reverse=True)

    summary = dict(
        tool="dsar_inr_hist_probe", run_dir=str(run_dir),
        capture_run_type="baseline_joint",
        n_cells=len(rows), n_hist=len(hist), n_cur_ctrl=len(cur),
        zref_gate=dict(passed=not zref_violations,
                       violations=zref_violations[:10]),
        b_grid=dict(ranks=args.b_ranks, gammas=args.b_gammas,
                    guard="mse"),
        ranking=[dict(rank=i + 1, config=k,
                      mean_delta_hist=configs[k]["mean_delta_hist"],
                      pos_fraction_hist=configs[k]["pos_fraction_hist"],
                      mean_delta_cur_ctrl=configs[k]["mean_delta_cur_ctrl"],
                      fit_status=configs[k].get("fit_status"))
                 for i, k in enumerate(ranked)],
        configs=configs,
        notes="historical readout y_k = TA_k(B_{c,r}(z+)) per DSAR.py "
              "sec-34.4a: z+ from extract_features --prompt_task_mode current "
              "--split test at task-end checkpoints; B fit on current-task "
              "boundary capture pairs (mse guard); heads[k] + current "
              "key/anchor routing + class_mask[k]. Eval protocol bs=16 "
              "deterministic extract (not bitwise capture-native). No "
              "pre-registered gate for historical rows yet (GPT pending).")
    print("\n=== INR historical-task probe (B-only, paper readout) ===")
    print(f"cells: {len(rows)} total | {len(hist)} historical | "
          f"{len(cur)} current-ctrl")
    print(f"{'config':22s} {'hist_mean':>10s} {'pos':>6s} {'cur_ctrl':>9s}")
    print(f"{'[off] baseline':22s} {'-':>10s} {'-':>6s} "
          f"{configs['off']['mean_acc_cur_ctrl']:9.2f}")
    for k in ranked:
        c = configs[k]
        pf = c["pos_fraction_hist"]
        pf_s = f"{pf:6.2f}" if pf is not None else "     -"
        print(f"{k:22s} {c['mean_delta_hist']:+10.3f} {pf_s} "
              f"{c['mean_delta_cur_ctrl']:+9.3f}")
    print("\nper-distance (t-k) historical delta:")
    dist_keys = sorted(per_dist)
    print(f"{'config':22s} " +
          " ".join(f"{('d' + str(d)):>7}" for d in dist_keys))
    for k in ["off"] + ranked:
        pd = configs[k]["per_distance_delta_hist"]
        print(f"{k:22s} " + " ".join(f"{pd[str(d)]:+7.2f}" for d in dist_keys))
    print(f"\nz_ref invariance: "
          f"{'PASS' if not zref_violations else 'VIOLATION'}")

    with open(out_dir / "cells.csv", "w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r_ in rows for k in r_},
                        key=lambda k: (k not in ("round", "client",
                                                 "cur_task", "hist_task"),
                                       k))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nwritten: {out_dir}/cells.csv, summary.json")


if __name__ == "__main__":
    main()
