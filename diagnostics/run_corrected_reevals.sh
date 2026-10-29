#!/bin/bash
# corrected 口径全量 reeval（用户 tmux 执行版，v2 语法：--checkpoints 单值）
# 用法：bash diagnostics/run_corrected_reevals.sh
# 输出：diagnostics_output/corrected_reeval/<arm>_<ckpt>_corrected.{csv,json,log}
set -u
cd "$(dirname "$0")/.."
D=diagnostics_output/corrected_reeval
mkdir -p $D

run () {  # run <run_dir> <name> <ckpt-stem>
  local dir=$1 name=$2 ck=$3
  if [ ! -f "$dir/args.json" ]; then
    echo "SKIP $name/$ck（run_dir 不存在，先跑对应训练）"; return
  fi
  if [ ! -f "$dir/checkpoints/$ck.pth" ]; then
    echo "SKIP $name/$ck（checkpoint 不存在）"; return
  fi
  echo "=== $name/$ck $(date +%T) ==="
  python diagnostics/eval_mode_reeval.py --run_dir "$dir" \
      --checkpoints "$ck.pth" \
      --out_csv $D/${name}_${ck}_corrected.csv \
      --out_json $D/${name}_${ck}_corrected.json \
      > $D/${name}_${ck}_corrected.log 2>&1 \
    && echo "OK  $name/$ck" || echo "FAIL $name/$ck（看 ${name}_${ck}_corrected.log）"
}

# ---- cifar ----
for c in task_00_end task_04_end; do
  run output/cifar100/fedta/cifar_base_p15/seed_42             cifar_base_p15 $c
  run output/cifar100/fedta_a2/a2_key/seed_42                  cifar_a2_key   $c
  run output/cifar100/fedta_a2/cifar_a2_key_hsn/seed_42       cifar_a2_key_hsn $c
  run output/cifar100/fedta/cifar_hsn/seed_42                  cifar_hsn      $c
done
run output/cifar100/fedta/cifar_a5v2_M1_drop0/seed_42          cifar_M1_drop0     task_00_end
run output/cifar100/fedta/cifar_a5v2_M2_refit/seed_42         cifar_M2_refit     task_00_end
run output/cifar100/fedta/cifar_a5v2_M3_refit_std/seed_42     cifar_M3_refit_std task_00_end
run output/cifar100/fedta/cifar_a5v2_M4_full_noconsol/seed_42 cifar_M4_full      task_00_end
run output/cifar100/fedta/cifar_a5v2_M5_refit_consol/seed_42  cifar_M5_refit_consol task_00_end

# ---- INR（每 ckpt 重建环境 ~3min） ----
for c in task_00_end task_04_end; do
  run output/ImageNet-R/fedta/inr_base_p40/seed_42             inr_base   $c
  run output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42        inr_a2_key $c
  run output/ImageNet-R/fedta/inr_hsn_p40/seed_42              inr_hsn    $c
done

# ---- M2' Gate（若已跑完） ----
run output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42      inr_M2feat_p40 task_00_end
run output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42     cifar_M2feat_full task_00_end
run output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42     cifar_M2feat_full task_04_end

echo "ALL DONE $(date +%T)"
