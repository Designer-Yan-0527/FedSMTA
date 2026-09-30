# FedTA 运行命令

本项目只支持 `cifar100` 和 `ImageNet-R` 两个数据集，通过 `--data_name` 选择。
所有参数定义在 `config/datasets_delay.py`，两个数据集共用同一套参数。

> 参数名写法：`--data-path` / `--data_path`、`--batch-size` / `--batch_size` 两种拼写均被接受（argparse alias），推荐统一用连字符形式。

## 1. 基本调用格式

```bash
python main.py datasets_delay --data_name {cifar100|ImageNet-R} [其他参数]
```

> `datasets_delay` 是子命令名（必带），与数据集无关；数据集由 `--data_name` 决定。

## 2. 运行 CIFAR-100

```bash
python main.py datasets_delay \
    --data_name cifar100 \
    --data-path ./local_datasets \
    --client_num 5 \
    --task_num 5 \
    --private_class_num 15 \
    --global_epoch 5 \
    --local_epoch 30 \
    --batch-size 16 \
    --lr 0.001 \
    --seed 42
```

数据集准备：**本地导入，不联网下载**。`--data-path` 目录下需已存在 `cifar-100-python/`（解压后的 CIFAR-100）；数据缺失时直接报错，不会自动下载。

## 3. 运行 ImageNet-R

```bash
python main.py datasets_delay \
    --data_name ImageNet-R \
    --data-path ./local_datasets \
    --surrogate_num 5 \
    --client_num 5 \
    --task_num 5 \
    --private_class_num 15 \
    --global_epoch 5 \
    --local_epoch 30 \
    --batch-size 16 \
    --lr 0.001 \
    --seed 42
```

数据集准备（**本地导入，不联网下载**）：

1. `--data-path` 目录下需已存在 `imagenet-r/`（解压后的 200 个类别目录）或 `imagenet-r.tar`（存在 tar 且未解压时会本地解压）；数据缺失时直接报错，不会自动下载；
2. 首次运行会做**一次性**的 80/20 随机划分（把图片移动到 `imagenet-r/train/`、`imagenet-r/test/` 子目录，之后直接复用，不会重复划分）；
3. ImageNet-R 共 30000 张图片、200 类，`args.nb_classes` 自动设为 200。

> **Phase 0 严格 baseline 说明**：官方 FedTA 对 ImageNet-R 固定每类 5 个 surrogate（`process_testdata(5)`），本仓库已恢复该行为——`--surrogate_num` 只对 cifar100 生效（官方默认 20）。本地训练 batch size 也已恢复官方硬编码 16（`--batch-size` 只影响 SIKF 蒸馏加载器，与官方一致）。

## 4. 常用参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_name` | `cifar100` | 数据集，可选 `cifar100` / `ImageNet-R`（区分大小写） |
| `--data-path` | `./local_datasets` | 数据集根目录（`--data_path` 亦可） |
| `--output_dir` | `output/` | 训练产物根目录 |
| `--client_num` | 5 | 客户端数量 |
| `--task_num` | 5 | 任务（阶段）数量 |
| `--private_class_num` | 15 | 每个客户端的私有类别数 |
| `--global_epoch` | 5 | 每个任务的联邦通信轮数 |
| `--local_epoch` | 30 | 每轮本地训练 epoch 数 |
| `--batch-size` | 16 | SIKF 蒸馏加载器 batch size（本地训练固定官方 16，不受此参数影响；`--batch_size` 亦可） |
| `--lr` | 0.001 | 学习率 |
| `--threshold` | 0.25 | BGPS 原型修复阈值（现已生效，旧代码硬编码 0.25） |
| `--surrogate_num` | 20 | 每类代理样本数（仅 cifar100 生效；ImageNet-R 固定官方值 5） |
| `--seed` | 42 | 随机种子 |
| `--num_workers` | 2 | DataLoader 进程数 |
| `--device` | `cuda` | 训练设备 |

## 5. 运行管理参数（Phase 0 工程，不影响算法）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--run_name` | `''`（自动生成） | 实验名。留空自动生成 `{dataset}_{method}_c.._t.._ge.._le.._lr.._seed.._{hash}` |
| `--resume` | `''` | `''` = 全新运行；`auto` = 从 `RUN_DIR/checkpoints/latest.pth` 恢复；或传 checkpoint 显式路径 |
| `--save_every` | 1 | 每 N 个全局轮保存一次 `round_XXXX.pth` |
| `--keep_last` | 0 | 只保留最近 N 个 `round_XXXX.pth`（0 = 全部保留）。`latest.pth` 和 `task_XX_end.pth` 始终保留 |
| `--eval_all_every` | 0 | 每 N 轮做一次全任务评估（0 = 仅在任务结束时评估） |
| `--deterministic` | off | 可选确定性模式（cudnn.deterministic、关闭 benchmark、deterministic algorithms）。默认关闭 = 官方行为 |
| `--max_rounds` | 0 | 总轮数上限，用于提前停止（0 = 完整运行）。仅截断训练循环终点，不改变任何每轮逻辑 |
| `--no_instrumentation` | off | 关闭 Phase 0 观察者插桩（train_log.csv 本地阶段评估、全任务评估、margin 输出）。仅供 observer-invariance 测试使用，用于证明插桩不扰动训练轨迹 |

## 6. 断点续训（resume）

```bash
# 方式一：自动从最近 checkpoint 恢复（要求 --run_name 等参数与原运行一致）
python main.py datasets_delay --data_name cifar100 --run_name <原run_name> --resume auto

# 方式二：显式指定 checkpoint 路径（run_dir 自动从路径推断）
python main.py datasets_delay --data_name cifar100 \
    --resume output/cifar100/fedta/<run_name>/seed_42/checkpoints/task_02_end.pth
```

说明：

- checkpoint 在**轮边界**保存，恢复时从下一轮继续，训练循环、数据划分、RNG 状态均精确还原；
- 恢复时校验关键参数（数据集、模型、client/task 数、lr、seed 等），不一致会拒绝恢复；
- 旧日志以追加方式保留，accuracy matrix 会接续完整历史。

## 7. 磁盘占用提醒

checkpoint 包含完整 ViT state_dict，**约 350MB+/轮**。默认每轮都保存（`--save_every 1`），一个 25 轮的运行约 9GB。磁盘紧张时建议：

```bash
--save_every 5 --keep_last 2   # 每 5 轮存一次，只保留最近 2 个 round checkpoint
```

## 8. 运行产物目录结构

```
output/
└── {data_name}/                  # cifar100 / ImageNet-R
    └── {method}/                 # 默认 fedta
        └── {run_name}/
            └── seed_{seed}/
                ├── command.txt           # 完整启动命令（追加式）
                ├── args.json             # 全部参数快照（仅首次运行写入）
                ├── git_commit.txt
                ├── logs/
                │   └── train.log         # 终端输出的完整副本（追加式）
                ├── checkpoints/
                │   ├── latest.pth        # 每轮更新，resume 默认入口
                │   ├── round_0001.pth    # 按 --save_every 保存
                │   └── task_00_end.pth   # 每个任务结束时的快照
                └── metrics/
                    ├── accuracy_matrix.csv    # 全任务准确率矩阵
                    ├── summary_metrics.csv    # AA / Forgetting / BWT / per-task retention 汇总（每 client + mean 行）
                    ├── margins.csv            # 各旧任务 margin 统计（mean / median / p10，全任务评估时追加）
                    ├── train_log.csv          # 结构化训练日志（round, task, client, accuracy, phase, test_task）
                    ├── client_0_accuracy.csv  # 各客户端准确率
                    └── round_metrics.jsonl    # 每轮指标流水
```

## 9. Phase 1 诊断工具（不改变正式训练）

诊断脚本通过共享的 `build_server()` 复刻训练的 RNG 流（seed → model default_cfg → 数据划分 → 建客户端），保证得到与训练完全相同的 client 数据划分。所有命令在项目根目录执行；`<run_dir>` 指 `output/{data_name}/fedta/{run_name}/seed_{seed}`。

### 9.1 提取特征（z_ref / z_op）

```bash
python diagnostics/extract_features.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --checkpoint output/cifar100/fedta/<run_name>/seed_42/checkpoints/task_04_end.pth \
    --out_dir diagnostics_output \
    --split both
```

- 输出 `diagnostics_output/features_{data_name}_{ckpt名}.npz`（z_ref = frozen ViT 特征、z_op = 当前 prompt 增强的 pre-anchor 特征；含 labels / client_ids / task_ids / round_ids / splits / sample_idx）+ 同名 `.json` 元信息（含 round、client_class_masks、public classes）
- `--checkpoint` 缺省时自动使用 `checkpoints/latest.pth`；`--data_path` / `--device` 可覆盖 args.json 中的值

### 9.2 多模态分析（SSE / NLL / BIC / silhouette / rho / D_cover / client entropy）

```bash
python diagnostics/analyze_multimodality.py \
    --features diagnostics_output/features_cifar100_task_04_end.npz \
    --plot_dir diagnostics_output/plots
```

- 默认只分析 public classes（类出现在 ≥2 个 client 的 class mask 中，由数据协议定义），加 `--all_classes` 分析全部；**只用 TRAIN split 发现 semantic modes——`--split` 仅接受 `train`，test/all 已被硬性禁止（test-leakage 防护）**
- 每类统计：`sse/delta_sse`、80/20 heldout `nll/delta_nll`、`bic/delta_bic`（p_K = K·d + K + (K−1)）、`silhouette`、`mode_distance`、`rho`（mode separation）、`mode1/mode2_mass`、`coverage_distortion`（单原型失真 D_cover）、每 mode 的 `client_entropy`（spatial heterogeneity 证据），另保留 S_W/S_B/H_c
- 输出（默认在特征文件同目录，可用 `--out_csv` / `--aggregate_json` 覆盖）：
  - `multimodality_summary.csv` — 每 (feature_type, class) 一行
  - `multimodality_aggregate.json` — 各指标 mean / median / std / p25 / p75 / bootstrap 95% CI
- `--plot_dir` 可选：每类 PCA 散点图（颜色=client，marker=task，星=FedTA 单原型，叉=K2 mode 中心）；PCA 仅用于可视化，不参与任何指标

### 9.3 兼容性 2x2 诊断（prompt × key-anchor）

```bash
python diagnostics/compatibility_drift.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --old_ckpt output/cifar100/fedta/<run_name>/seed_42/checkpoints/task_02_end.pth \
    --new_ckpt output/cifar100/fedta/<run_name>/seed_42/checkpoints/task_04_end.pth \
    --old_task 2
```

- 评估 old/current prompt × old/current key-anchor 四种组合（旧 task 始终用对应 task-specific Chead）
- 输出默认写到 `<run_dir>/metrics/compatibility_2x2.csv`；`--old_task` 缺省时从 old_ckpt 推断

### 9.4 Synthetic Oracle-K2 压力测试（sanity check，**非真实 Oracle-K2**）

```bash
python diagnostics/synthetic_oracle_k2.py \
    --features diagnostics_output/features_cifar100_task_04_end.npz \
    --out_dir diagnostics_output/synthetic_oracle_k2 \
    --deltas 0,0.25,0.5,1.0,2.0 \
    --methods fedta_k1,oracle_k2
```

- **定位**：人为注入两个 mode（z' = z + s·δ·v_c）并给定 oracle 真值，只回答"K2 anchor 机制能否利用已知的两个 mode"——是 anchor 机制的 synthetic sanity check，**不能**作为"真实数据多模态性提升 FedTA"的证据。真正的 Phase 2 Oracle-K2 必须等 Phase 1 结果出来后，用真实 train features 构建的 semantic modes
- 纯特征级联邦持续学习模拟：FedTA-K1 vs Oracle-K2（s 只依赖 seed/client/task/样本，两法严格一致）
- 输出 `oracle_k2_results.csv`（每轮各任务 accuracy/margin）、`oracle_k2_summary.csv`（AA / Forgetting / BWT / retention / mode separation）、`oracle_k2_meta.json`（含 `experiment: synthetic_oracle_k2_stress_test` 标记与 synthetic 定位说明）
- 服务器上先跑小 delta 网格确认流程，再跑完整网格

## 10. Resume 回归测试（Phase 0）

```bash
python tests/test_resume_regression.py \
    --data_name cifar100 \
    --data_path ./local_datasets \
    --rounds 4 --split_at 2 \
    --global_epoch 2 --task_num 2 --local_epoch 2 \
    --device cuda
```

- A = 连续跑 4 轮；B = 用 `--max_rounds 2` 跑 2 轮后 `--resume auto` 续跑 2 轮
- 自动附加 `--deterministic`，逐项比较最终 model / head / prompt / protos / fix_keys / 各 client tail_anchor / RNG states / accuracy_matrix.csv / 数据划分索引
- checkpoint 现已包含**真正的联邦外层划分 manifest**（每 client/task 的原始样本 raw_indices + class_mask，位于 70/30 内层划分之前）；resume 时会重新生成划分并与之逐位比对，不一致直接拒绝恢复
- 输出 `output/regression/metrics/baseline_verification.json`：两个 run 的 baseline 记录（outer split manifest hash、inner 70/30 split index hash、每 client/task train/test index hash、protos / head / prompt / key / anchor / RNG checksum、accuracy matrix hash）+ 比较字段（`accuracy_max_diff`、`global_proto_max_diff`、`prompt_max_diff`、`head_max_diff`、`key_max_diff`、`anchor_max_diff`、`split_equal`、`outer_split_equal`）
- **Gate 0**：`result != PASS` 时禁止进入 Phase 1

## 11. Observer-invariance 测试（Phase 0）

```bash
python tests/test_observer_invariance.py \
    --data_name cifar100 \
    --data_path ./local_datasets \
    --rounds 3 \
    --global_epoch 2 --task_num 2 --local_epoch 2 \
    --device cuda
```

- 证明 Phase 0 插桩（train_log.csv 本地阶段评估、全任务评估、margin 记录）是**纯观察者**：`--no_instrumentation`（官方循环，无任何额外评估）vs 默认（插桩开启）两个短程 run 的训练轨迹必须逐位一致
- 默认 `--rounds = global_epoch + 1`（跑完 task 0 + 进入 task 1 第 1 轮），因此比较项含**next-task split**（task 1 的 70/30 划分索引，由 RNG 流决定）
- 比较项：server model / prompt / global protos / global head / 各 client heads / key / anchor / client prompts / fix_keys / RNG states / 内层 split / 外层 manifest
- 输出 `output/regression/metrics/observer_invariance.json`；`result != PASS` 说明插桩扰动了训练随机流（检查 `_log_local_phase` / `run_full_evaluation` 的 RNG 保护）
- 与第 10 节互补：回归测试证明 resume 不改变当前程序自身轨迹；本测试证明插桩本身不改变轨迹
