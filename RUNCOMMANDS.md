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
    --out_dir diagnostics_output
```

- 输出 `diagnostics_output/features_{data_name}_{ckpt名}.npz`（z_ref = frozen ViT 特征、z_op = 当前 prompt 增强的 pre-anchor 特征；含 labels / client_ids / task_ids / round_ids / splits / sample_idx）+ 同名 `.json` 元信息（含 round、client_class_masks、public classes）
- **正式 Phase 1 协议默认值：`--split train`、`--batch_size 16`**（z_op 依赖 batch 组成，B 必须与 precheck 通过的取值一致；`--split` 的 test/both 仅限非诊断用途，多模态分析侧已硬性只接受 train）
- `--checkpoint` 缺省时自动使用 `checkpoints/latest.pth`；`--data_path` / `--device` 可覆盖 args.json 中的值

### 9.2 多模态分析（NLL / BIC / rho / D_cover(+rel) / Dirichlet 校正指标 / stability）

```bash
python diagnostics/analyze_multimodality.py \
    --features diagnostics_output/features_cifar100_task_04_end.npz \
    --plot_dir diagnostics_output/plots
```

- 默认只分析 public classes（类出现在 ≥2 个 client 的 class mask 中，由数据协议定义），加 `--all_classes` 分析全部；**metadata 缺失 `public_classes` 时直接 RuntimeError**（public/private 是联邦协议属性，禁止从特征样本反推）；**只用 TRAIN split 发现 semantic modes——`--split` 仅接受 `train`，test/all 已被硬性禁止（test-leakage 防护）**
- **模型选择与描述性几何严格分离**：held-out GMM（80/20）只出 `nll/delta_nll`、`bic/delta_bic`（p_K = K·d + K + (K−1)）；`sse/delta_sse` 由全量 TRAIN 上的专用 KMeans(K=2) 计算（K=1 解是 K=2 的可行特例，数学上严格保证 SSE₂ ≤ SSE₁；held-out GMM 的中心只见 80% 数据，不得用于 SSE）
- 每类统计：`delta_sse`、`delta_nll`、`delta_bic`、`silhouette`、`mode_distance`、`rho`、`mode1/mode2_mass`（**canonical ordering：mode1 = major mode，π₁ ≥ π₂**，否则 GMM component label 跨 seed/class 不可比——label switching）、`mode_mass_minor`（label-invariant）、`coverage_distortion`（D_cover）+ **`coverage_distortion_rel`**（失真占类内散度比例）、每 mode 的 `client_entropy` + Dirichlet 校正指标：**`client_mode_mi`（I(I;K|C=c)）与 `js_weighted`（Σₖ πₖ·D_JSₖ）为主证据（均 label-invariant）**，辅助 `client_entropy_ratio_k`（H_k / H(p(i|c))，**是 ratio 而非 [0,1] 归一化，可 >1**）与 `js_divergence_k`，另保留 S_W/S_B/H_c
- **报告口径 = 跨 run median**（`--heldout_seeds 0 1 2 3 4` × `--gmm_seeds 0 1 2` 配对轮转共 5 runs）：每类正式指标为 `*_median`（附 `*_std`），CSV 中 run-0 快照降级为 `*_primary` **debug 列**，不作为论文主结果；`p_nll` / `p_bic`（Δ>0 的 run 比例）同口径；**robust multimodal** 判定 = p_nll ≥ 0.8 且 p_bic ≥ 0.8 且 `rho_median` > `--rho_threshold`(默认 1.0) 且 `dcover_rel_median` > `--dcover_rel_threshold`(默认 0.1)——**这些是 operational thresholds 而非理论阈值**，aggregate JSON 内含 `robust_sensitivity` 网格（ρ ∈ {0.75, 1.0, 1.25} × D_rel ∈ {0.05, 0.10, 0.15}）供阈值敏感性检查
- 输出（默认在特征文件同目录，可用 `--out_csv` / `--aggregate_json` 覆盖）：
  - `multimodality_summary.csv` — 每 (feature_type, class) 一行（含 stability 列与 robust 标志）
  - `multimodality_aggregate.json` — 各指标 mean / median / std / p25 / p75 / bootstrap 95% CI + **robust 类计数/比例/清单（prevalence，Phase 1 Gate 判据）**
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

## 10. 官方 FedTA parity 测试（Gate 0A）

```bash
# Gate 0A smoke（默认配置：task_num=1, rounds=5, local_epoch=1，已覆盖官方 `!= 4` SIKF 分支）
python tests/test_official_fedta_parity.py \
    --official_repo /path/to/official_FedTA \
    --data_path ./local_datasets \
    --global_epoch 5 --task_num 1 --rounds 5 --local_epoch 1 \
    --batch_size 16 --surrogate_num 20 --threshold 0.25 \
    --seed 42 --device cuda

# Gate 0A 正式短程（覆盖 task transition）
python tests/test_official_fedta_parity.py \
    --official_repo /path/to/official_FedTA \
    --data_path ./local_datasets \
    --global_epoch 5 --task_num 2 --rounds 10 --local_epoch 1 \
    --batch_size 16 --surrogate_num 20 --threshold 0.25 \
    --seed 42 --device cuda
```

- 证明最强命题：`FedSMTA --method fedta --no_instrumentation ≡ Official FedTA`（Gate 0B 只能证明"自己的代码连续跑 = 自己的代码 resume 跑"，不能证明等于官方）
- **Gate 0A 强制 `global_epoch=5`**（传其他值直接退出）：官方 FedTA 硬编码 SIKF 分支为 `i % global_epoch != 4`，与本项目参数化的 `!= global_epoch - 1` 仅在 `global_epoch=5` 时等价；Phase 0 冻结的就是 `global_epoch=5` 的行为。**Gate 0B/0C 保持 `global_epoch=2` 不矛盾**——它们比较 ours vs ours（resume / observer），只验证工程机制，不是官方 parity
- 官方仓库无 checkpoint 机制：测试会在官方仓库内生成一个一次性 driver 脚本，进程内复刻官方 `main()` 流程（含 RNG 消耗顺序），跑完后导出最终状态；结束后自动清理 driver/payload/dump（`--keep_driver` 保留，FAIL 时也会保留以便调试）
- fork 侧以冻结的短程 parity 配置运行（`global_epoch=5, batch_size=16, surrogate_num=20, threshold=0.25, seed=42`，smoke 为 `task_num=1/rounds=5/local_epoch=1`，正式为 `task_num=2/rounds=10/local_epoch=1`），并强制 `--deterministic --no_instrumentation`；`--rounds` 必须等于 `task_num*global_epoch`（官方仓库总是跑完整任务循环）
- 前置条件：`--official_repo` 为官方 FedTA 仓库的本地克隆（含 `main.py`/`Models`/`config`）；官方仓库内需有 `pretrain_model/ViT-B_16.npz`；官方代码若硬编码数据集路径需提前准备（被 drop 的 kwarg 会在 JSON 中报告）；官方仓库若需要不同环境可用 `--official_python` 指定解释器
- 配置漂移防护：实验定义参数（client_num/task_num/.../model_name）在官方 config 中逐 key 断言存在，缺失直接报错并列出可用 key（fail loudly，绝不静默错配）；config 模块自动尝试 `config.cifar100` / `config.cifar100_delay`，可用 `--config_module` 覆盖
- 比较项：server model / global head / global+temp protos / server prompt / 各 client heads / key / anchor / 完整 tail_anchor state / client prompts / fix_keys / 外层 manifest（raw indices + class_mask）/ 内层 70/30 split（通过 patch `random_split` + 包装 `get_data` 在官方侧记录）；两侧 inner split 均做结构校验（缺失/空/缺 train-test 键/缺已到达 task 一律 FAIL，杜绝 `{} == {}` 假 PASS）
- 封板断言：fork checkpoint 必须记录 `no_instrumentation=True`、`deterministic=True`、跑满全部 rounds；两侧 manifest 必须存在且非空（无 `{} == {}` 假 PASS）
- 输出 `output/regression/metrics/official_fedta_parity.json`；**PASS 要求 D=0（不容忍浮点误差）**——若出现极小非零差异，先怀疑 GPU 算子非确定性并重跑，不允许放宽 tolerance
- v1 限制（已记录在 JSON notes，不作 gating）：accuracy 不直接比较（所有模型状态与两个 split 的 D=0 蕴含评估输出一致）；RNG states 无法比较（官方无保存机制）

## 11. Resume 回归测试（Gate 0B）

```bash
python tests/test_resume_regression.py \
    --data_name cifar100 \
    --data_path ./local_datasets \
    --rounds 4 --split_at 2 \
    --global_epoch 2 --task_num 2 --local_epoch 2 \
    --seed 42 --device cuda
```

- A = 连续跑 4 轮；B = 用 `--max_rounds 2` 跑 2 轮后 `--resume auto` 续跑 2 轮（resume 命令显式带 `--max_rounds`，不依赖"恰巧跑满任务循环"的隐式行为）
- 自动附加 `--deterministic`，逐项比较最终 model / head / prompt / global+temp protos / client prompts / fix_keys / 各 client 完整 tail_anchor state / client 级 global/local protos / existing_class / RNG states / accuracy_matrix.csv / 数据划分索引
- checkpoint 现已包含**真正的联邦外层划分 manifest**（每 client/task 的原始样本 raw_indices + class_mask，位于 70/30 内层划分之前）；resume 时会重新生成划分并与之逐位比对，不一致直接拒绝恢复
- **封板（假 PASS 防护）**：两侧 checkpoint 缺失或空 manifest 时测试直接 FAIL（`require_outer_manifest` fail-fast，杜绝 `{} == {}` 空比较通过）；**inner split 同样 fail-fast**（`require_inner_split_state`：缺失/空/缺 train-test 键/缺已到达 task 一律 FAIL，且每 client 必须含运行所到达的全部 task 的划分）；**accuracy matrix 缺失即 FAIL**（不再 `None == None → 0.0`，Gate B 插桩开启、矩阵必须存在）；`Server_DF.load_checkpoint()` 对缺失/空/不匹配 manifest 一律 `RuntimeError`（不再接受无 manifest 的旧 checkpoint）；每次运行前自动清理旧的测试 run 目录（防 stale checkpoint 触发 auto-resume）
- 输出 `output/regression/metrics/baseline_verification.json`：两个 run 的 baseline 记录（outer split manifest hash、inner 70/30 split index hash、每 client/task train/test index hash、protos / head / prompt / key / anchor / RNG checksum、accuracy matrix hash）+ 比较字段（`accuracy_max_diff`、`global_proto_max_diff`、`temp_proto_max_diff`、`prompt_max_diff`、`client_prompt_max_diff`、`head_max_diff`、`key_max_diff`、`anchor_max_diff`、`tail_anchor_state_max_diff`、`client_global_proto_max_diff`、`client_local_proto_max_diff`、`split_equal`、`outer_split_equal`、`existing_class_equal`）
- **Gate 0B**：`result != PASS` 时禁止进入 Phase 1

## 12. Observer-invariance 测试（Gate 0C）

```bash
python tests/test_observer_invariance.py \
    --data_name cifar100 \
    --data_path ./local_datasets \
    --rounds 3 \
    --global_epoch 2 --task_num 2 --local_epoch 2 \
    --seed 42 --device cuda
```

- 证明 Phase 0 插桩（train_log.csv 本地阶段评估、全任务评估、margin 记录）是**纯观察者**：`--no_instrumentation`（官方循环，无任何额外评估）vs 默认（插桩开启）两个短程 run 的训练轨迹必须逐位一致
- 默认 `--rounds = global_epoch + 1`（跑完 task 0 + 进入 task 1 第 1 轮），因此比较项含**next-task split**（task 1 的 70/30 划分索引，由 RNG 流决定）
- 比较项：server model / prompt / global+temp protos / global head / 各 client heads / key / anchor / 完整 tail_anchor state / client prompts / client 级 global/local protos / fix_keys / existing_class / RNG states / 内层 split / 外层 manifest
- **封板断言**：checkpoint 必须记录 `no_instrumentation` off=True / on=False（否则比较是空洞的）；manifest fail-fast（缺失/空即 FAIL）；**inner split 同样 fail-fast**（`require_inner_split_state`，含 next-task split 覆盖检查）；artifact 区分验证——OFF run 不得产生任何 observer 文件（`train_log.csv`/`round_metrics.jsonl`/`accuracy_matrix.csv`/`margins.csv`），ON run 必须产生（证明 `--no_instrumentation` 真正等于 "FedTA + checkpoint only"）
- 输出 `output/regression/metrics/observer_invariance.json`；`result != PASS` 说明插桩扰动了训练随机流（检查 `_log_local_phase` / `run_full_evaluation` 的 RNG 保护）
- 与第 11 节互补：回归测试证明 resume 不改变当前程序自身轨迹；本测试证明插桩本身不改变轨迹
- Gate 0B/0C 使用 `global_epoch=2` 是合法的：它们比较 ours vs ours（工程机制验证），不涉及官方 `!= 4` 硬编码的等价性（那是 Gate 0A 专属约束，见第 10 节）

## 13. Phase 0 Freeze（三个 Gate 全部 PASS 后）

```bash
# 顺序：Gate 0A smoke（第 10 节）→ Gate 0A 正式 → Gate 0B（第 11 节）→ Gate 0C（第 12 节）
# 三者全部 PASS 后：
git add .
git commit -m "freeze verified FedTA baseline"
git tag phase0-fedta-baseline
git push origin main
git push origin phase0-fedta-baseline
```

## 14. Phase 1-precheck：z_op batch sensitivity 诊断（tag 之后、正式 Phase 1 之前）

```bash
python diagnostics/check_feature_batch_sensitivity.py \
    --run_dir output/regression/cifar100/fedta/reg_full/seed_42 \
    --checkpoint checkpoints/latest.pth \
    --device cuda
```

- 背景（roadmap §5.9）：`batchwise_prompt=True` 下 `z_op(x) = f(x; B)` 依赖 batch 组成；若不稳定，后续 K=2 结论可能是 prompt routing 伪影而非真实语义多模态
- 协议（三层对照 + 两个 floor）：
  - **batch-size 扫描**：B = 1/8/16/32/64，同时提取 `z_op` 与 `z_ref`——`zref_B=*_vs_B=1` 是 GPU 数值 floor（origin_model 理论上 batch 无关）
  - **重复运行 floor**：B=16 同一顺序跑两遍（`zop_repeat_*_run2_vs_run1`）——纯 GPU 非确定性 floor（脚本内已强制 cudnn deterministic）
  - **组批对照**：B=16 恒定、`--composition_seeds 0 1 2 3 4` 五次 deterministic permutation vs identity，隔离"组批效应"与"batch size 效应"，杜绝单次 permutation 恰好没换 major prompt 的低估
  - 样本选择：**class-stratified deterministic sampling**（固定种子，非"前 N 个 sorted index"）
- 输出 `metrics/feature_batch_sensitivity.csv` + `.json`：每 (client, task, 对照) 与汇总（ALL）的 delta/cosine 的 mean/median/std/**p05**/p95/**min**/max（delta 看 p95/max 尾部，cosine 看 p05/min 尾部）；JSON 含 reading_guide（归因规则：只有 `zop` 漂移 ≫ 两个 floor 才能归因 batchwise prompt）
- **deterministic 设置顺序（v7 修复）**：`cudnn.deterministic / benchmark / use_deterministic_algorithms` 必须在 `bootstrap_server()` **之后**设置——`diag_utils` 会强制 `args.deterministic=False`，`build_server()` → `setup_determinism(False)` 会重开 `cudnn.benchmark=True`，静默覆盖 bootstrap 前设的任何 flags；`CUBLAS_WORKSPACE_CONFIG` 则必须在任何 CUDA context 创建**之前**写入 env。JSON 中的 `cudnn` 字段从 `torch.backends` **实读**（不硬编码），可直接验证运行时真实状态
- 解读（决策见 roadmap §5.9，看到数字前不擅自改用 z_ref / 不关 batchwise_prompt）：$\bar{\Delta}_B \approx 0$ → `z_op` 可直接用于 Phase 1；显著非零且 ≫ floor → 需重新决定 Semantic Bank 用 `z_ref` / sample-wise prompt feature / 严格定义 prompt context
- 注意：`--run_dir` 指向**已有完整训练产物**的 run（如 Gate 0B 的 `reg_full`），不是新开训练

- 正式 Phase 1 的特征提取 / 多模态分析命令见第 9 节（`extract_features.py` / `analyze_multimodality.py`），在 precheck 结论通过后执行

## 15. Freeze 后的硬约束（长期有效）

- 从 tag 起，`--method fedta` 视为 **read-only scientific baseline**：不得改动 FedTA loss / optimizer / BGPS / SIKF / FedAvg head / Tail Anchor 结构 / prompt 逻辑 / CIFAR 联邦划分
- 之后的顺序：`diagnostics/check_feature_batch_sensitivity.py`（Phase 1-precheck）→ 正式 Phase 1；z_op batch sensitivity 属于 Phase 1-precheck，不是 Phase 0 Gate
