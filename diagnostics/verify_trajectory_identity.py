"""两 run 训练轨迹逐位一致性 gate（Gate 0C 方法的目录对目录版本）。

问题：capture run（cifar_bsab_capture_M2matched，--head_refit
--refit_input feat --refit_timing local + --bsab_pair_capture）的
**表征侧训练轨迹**是否与 baseline run（cifar_base_p15，官方协议无
refit 无 capture）逐位一致？A5 的 RNG 中立性设计预言：一致。本脚本
在两边共有的 round checkpoint 上逐字段验证这一预言。

字段分三档：

  HARD（必须逐位一致，任一不等 = GATE FAIL）
    server: server_model / global_head / server_prompt / global_protos /
            temp_protos / fix_keys / existing_class / server_task_id /
            data_split_state / outer_split_manifest
    client: prompts / tail_anchor_model(含 head.*=联训头) /
            global_protos / local_protos / data_split_indices /
            class_mask / current_class / task_id
  EXPECTED_DIFF（设计上就该不同，只报告不判负）
    client: heads[]（refit 头 vs 官方头）/ a5_joint_head（仅 ON 臂存在）
  INFO（信息项）
    rng_state（理想下也应一致）/ args（两 run flags 本就不同）

数值口径：tensor 用 torch.equal（逐位，不容浮点误差）；NaN 位置相同
仍记为 FAIL 但标注 NaN-pattern-equal 以便定位。fail-loud：第一个
不一致即完整报告路径。

用法（repo 根目录）：
  python diagnostics/verify_trajectory_identity.py
  # 或显式指定：
  python diagnostics/verify_trajectory_identity.py \
      --baseline_dir output/cifar100/fedta/cifar_base_p15/seed_42 \
      --treatment_dir output/cifar100/fedta/cifar_bsab_capture_M2matched/seed_42
"""

import argparse
import sys
from pathlib import Path

# 允许从 repo 根目录直接运行：python diagnostics/verify_trajectory_identity.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from runtime_utils import load_checkpoint_file

MAX_DIFF_DETAIL = 20


def parse_args():
    p = argparse.ArgumentParser("two-run trajectory identity gate")
    p.add_argument("--baseline_dir", default=(
        "output/cifar100/fedta/cifar_base_p15/seed_42"))
    p.add_argument("--treatment_dir", default=(
        "output/cifar100/fedta/cifar_bsab_capture_M2matched/seed_42"))
    return p.parse_args()


def classify(path):
    """('clients', i, 'heads', ...) -> expected_diff 等。"""
    if len(path) >= 3 and path[0] == "clients" and path[2] == "heads":
        return "expected_diff"
    if (len(path) >= 3 and path[0] == "clients"
            and path[2] == "a5_joint_head"):
        return "expected_diff"
    if path and path[0] in ("rng_state", "args"):
        return "info"
    return "hard"


def walk(a, b, path, diffs):
    """递归比较，把差异记成 (path, kind, detail) 三元组。"""
    kind = classify(path)
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            diffs.append((kind, path, "type mismatch",
                          type(a).__name__, type(b).__name__))
        elif a.shape != b.shape:
            diffs.append((kind, path, "shape mismatch",
                          tuple(a.shape), tuple(b.shape)))
        elif not torch.equal(a, b):
            na, nb = torch.isnan(a), torch.isnan(b)
            sub = ("NaN-pattern-equal" if torch.equal(na, nb) else "value")
            amax = (a.float() - b.float()).abs().max().item() \
                if a.is_floating_point() and not sub.startswith("NaN") else -1.0
            diffs.append((kind, path, f"tensor mismatch ({sub})",
                          amax, None))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            # 注意：缺失字段的分类必须按子路径算（父路径的 kind 会把
            # a5_joint_head 这类插桩专属字段误判为 hard）
            ckind = classify(path + (str(k),))
            if k not in a:
                diffs.append((ckind, path + (str(k),), "only-in-treatment",
                             None, None))
            elif k not in b:
                diffs.append((ckind, path + (str(k),), "only-in-baseline",
                             None, None))
            else:
                walk(a[k], b[k], path + (str(k),), diffs)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append((kind, path, "length mismatch", len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + (str(i),), diffs)
        return
    if type(a) is not type(b):
        diffs.append((kind, path, "value mismatch",
                      repr(a)[:50], repr(b)[:50]))
        return
    try:
        unequal = bool(a != b)
    except (ValueError, TypeError):
        # numpy 数组等容器：!= 返回数组，改用逐位比较
        unequal = not np.array_equal(a, b)
    if unequal:
        diffs.append((kind, path, "value mismatch",
                      repr(a)[:50], repr(b)[:50]))


def main():
    args = parse_args()
    base_ck = Path(args.baseline_dir) / "checkpoints"
    treat_ck = Path(args.treatment_dir) / "checkpoints"

    def rounds_of(d):
        return {int(m.stem.split("_")[1]) for m in d.glob("round_*.pth")}

    rb, rt = rounds_of(base_ck), rounds_of(treat_ck)
    common = sorted(rb & rt)
    if not common:
        raise SystemExit(f"GATE ABORT: no common round checkpoints "
                         f"(baseline {sorted(rb)}, treatment {sorted(rt)})")
    print(f"baseline rounds: {len(rb)} | treatment rounds: {len(rt)} | "
          f"comparing common: {common}")

    n_fail = 0
    for r in common:
        cb = load_checkpoint_file(str(base_ck / f"round_{r:04d}.pth"))
        ct = load_checkpoint_file(str(treat_ck / f"round_{r:04d}.pth"))

        diffs = []
        walk(cb, ct, (), diffs)

        hard = [d for d in diffs if d[0] == "hard"]
        exp = [d for d in diffs if d[0] == "expected_diff"]
        info = [d for d in diffs if d[0] == "info"]

        ok = not hard
        n_fail += 0 if ok else 1
        print(f"\nround {r:4d}: {'PASS' if ok else 'FAIL'}"
              f"  (expected_diff fields: {len(exp)}, info fields: {len(info)})")
        for kind, path, what, x, y in hard[:MAX_DIFF_DETAIL]:
            print(f"  [HARD] {'.'.join(path) or '<root>'}: {what} "
                  f"{'' if x is None else x} {'' if y is None else y}")
        if len(hard) > MAX_DIFF_DETAIL:
            print(f"  [HARD] ... {len(hard) - MAX_DIFF_DETAIL} more")
        if r == common[0]:
            # 首轮打印预期差异与信息项的明细，供人工核对口径
            for kind, path, what, x, y in exp:
                print(f"  [EXPECTED_DIFF] {'.'.join(path)}: {what}")
            for kind, path, what, x, y in info:
                print(f"  [INFO] {'.'.join(path)}: {what}")

    print("\n" + "=" * 60)
    if n_fail == 0:
        print(f"TRAJECTORY IDENTITY GATE: PASS "
              f"({len(common)}/{len(common)} rounds bit-identical on all "
              f"HARD fields; refit+capture instrumentation is "
              f"RNG-neutral on the real full run)")
    else:
        print(f"TRAJECTORY IDENTITY GATE: FAIL "
              f"({n_fail}/{len(common)} rounds have HARD-field mismatches)")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
