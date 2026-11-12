# RUN — 最终方案（A2 + L + B）运行命令

> 本文件只放当前冻结方案的发车命令。完整档案（证据链 / GPT 裁决 / 勘误 / 工具用法）见 RUNCOMMANDS.md。
> 所有命令在 izfree 的 repo 根目录执行，先 `git pull`（版本 ≥ d1e515d）。
> 两个数据集的 DSAR 配方由 `--data_name` 自动分派，不需要额外参数：CIFAR = γ\*0.396419 / L r8（§34.4 冻结）；INR = γ\*2.0579 / L r16（§34.7 dsar-inr-v1）。

## 方案速览（四臂矩阵）

| 臂 | 开关 | 职责 |
|---|---|---|
| baseline | 无 | 对照 |
| A2 | `--a2_arm key` | 训练侧：旧任务支撑 key 槽位 post-step restore |
| DSAR | `--dsar` | 读出侧：B 对齐桥（历史任务）+ L 适配头（当前任务） |
| A2+DSAR | `--a2_arm key --dsar` | 组合臂，验证超加性 |

## CIFAR-100

```bash
# 1) baseline
python main.py datasets_delay \
    --data_name cifar100 --data_path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch_size 16 \
    --lr 0.001 --seed 42 --run_name cifar_base_p15

# 2) A2 臂
python main.py datasets_delay \
    --data_name cifar100 --data_path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch_size 16 \
    --lr 0.001 --seed 42 --a2_arm key --run_name cifar_a2_key_v2

# 3) DSAR 臂（B1 formal ON）
python main.py datasets_delay \
    --data_name cifar100 --data_path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch_size 16 \
    --lr 0.001 --seed 42 --dsar --run_name cifar_b1_dsar_on

# 4) A2+DSAR 组合臂
python main.py datasets_delay \
    --data_name cifar100 --data_path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch_size 16 \
    --lr 0.001 --seed 42 --a2_arm key --dsar --run_name cifar_a2_dsar_on
```

## ImageNet-R

```bash
# 1) baseline
python main.py datasets_delay \
    --data_name ImageNet-R --data_path ./local_datasets \
    --surrogate_num 5 --client_num 5 --task_num 5 --private_class_num 40 \
    --global_epoch 5 --local_epoch 30 --batch_size 16 \
    --lr 0.001 --seed 42 --run_name inr_base_p40

# 2) A2 臂
python main.py datasets_delay \
    --data_name ImageNet-R --data_path ./local_datasets \
    --surrogate_num 5 --client_num 5 --task_num 5 --private_class_num 40 \
    --global_epoch 5 --local_epoch 30 --batch_size 16 \
    --lr 0.001 --seed 42 --a2_arm key --run_name inr_a2_key

# 3) DSAR 臂（INR §34.7 配方自动生效）
python main.py datasets_delay \
    --data_name ImageNet-R --data_path ./local_datasets \
    --surrogate_num 5 --client_num 5 --task_num 5 --private_class_num 40 \
    --global_epoch 5 --local_epoch 30 --batch_size 16 \
    --lr 0.001 --seed 42 --dsar --run_name inr_dsar_v1

# 4) A2+DSAR 组合臂
python main.py datasets_delay \
    --data_name ImageNet-R --data_path ./local_datasets \
    --surrogate_num 5 --client_num 5 --task_num 5 --private_class_num 40 \
    --global_epoch 5 --local_epoch 30 --batch_size 16 \
    --lr 0.001 --seed 42 --a2_arm key --dsar --run_name inr_a2_dsar_on
```

## 发车自检

每条命令启动后核对：

1. `output/{data_name}/fedRelink/{run_name}/seed_42/args.json`：`a2_arm` / `dsar` 旗标与臂一致、`data_name` 正确
2. 进入任务 1 后 train.log 出现 `[A2] client X task 1 arm=key: protected n/... slots`（任务 0 无保护集是正常的）
3. DSAR 臂首个广播边界后有 sidecar 提交记录（`b_status` / `l_status` 出现 `FITTED`）
4. INR DSAR 臂首个 checkpoint 的 sidecar meta 里 `recipe: "dsar-inr-v1"`（CIFAR 臂为 `"dsar-b1-cifar"`）

## 注意事项

- **正在跑的 `inr_b1_dsar_on`（CIFAR 配方）**：pull 新代码后不要 resume 它——r8 L 因子在 INR 新配方下形状校验 fail-closed 拒绝（防串配方）。让它一口气跑完，作"CIFAR 配方在 INR"的对照证据。
- **INR baseline 可省**：`inr_bsab_capture_baseline_p40`（izfree 已跑完）的 capture 是 instrumentation-only、训练与 baseline 逐位一致，可直接充当 INR 基线臂；`inr_base_p40` 仅在想统一 run 命名时才需要跑。
- **磁盘**：checkpoint 约 350MB+/轮，紧张时加 `--save_every 5 --keep_last 2`。
- 对照冻结配方发车：不要叠加 `--bsab_pair_capture` 等插桩开关。
- 产物：`output/{data_name}/fedRelink/{run_name}/seed_42/metrics/accuracy_matrix.csv`（R[t][k] 矩阵）与 `summary_metrics.csv`（AA / Forgetting / BWT）。
