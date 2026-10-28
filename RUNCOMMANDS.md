# FedSMTA 运行命令

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
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 \
    --lr 0.001 --seed 42
```

数据集准备：**本地导入，不联网下载**。`--data-path` 目录下需已存在 `cifar-100-python/`；缺失直接报错。

## 3. 运行 ImageNet-R

```bash
python main.py datasets_delay \
    --data_name ImageNet-R \
    --data-path ./local_datasets \
    --surrogate_num 5 \
    --client_num 5 --task_num 5 --private_class_num 40 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 \
    --lr 0.001 --seed 42
```

> **协议要点（勿改）**：官方论文 §5.1 规定 ImageNet-R 为每客户端 40 私有类、0 公共类、每任务 8 类（P0 manifest 审计 2026-10-20 证实 200 类全部单客户端）。`--private_class_num 40` 是唯一正确取值；误用 15 会得到 125 公共类的错误协议（2026-10-14 误跑已判废）。

数据集准备（本地导入）：`--data-path` 下需已存在 `imagenet-r/`（200 个类别目录）或 `imagenet-r.tar`；首次运行做一次性 80/20 划分（移动到 train/test 子目录后复用）。`--surrogate_num` 只对 cifar100 生效（INR 固定官方值 5）；本地训练 batch size 固定官方硬编码 16。

## 4. 常用参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_name` | `cifar100` | 数据集（区分大小写） |
| `--data-path` | `./local_datasets` | 数据集根目录 |
| `--output_dir` | `output/` | 训练产物根目录 |
| `--client_num` / `--task_num` | 5 / 5 | 客户端 / 任务数 |
| `--private_class_num` | 15 | 每客户端私有类数（INR 必须 40） |
| `--global_epoch` / `--local_epoch` | 5 / 30 | 联邦轮数 / 本地 epoch 数 |
| `--lr` | 0.001 | 学习率 |
| `--threshold` | 0.25 | BGPS 原型修复阈值 |
| `--surrogate_num` | 20 | 每类代理样本数（仅 cifar100 生效） |
| `--seed` | 42 | 随机种子 |
| `--num_workers` / `--device` | 2 / cuda | 加载进程 / 设备 |
| `--skip_fed_avg_head` | off | **HS-N 修复开关**：跳过 fed_avg_head（论文无 head 聚合，该调用唯一活效应 = ×2^(N-1)/轮别名污染，详见 INR_PHASE1_HEAD_DIVERGENCE.md §4.5）。默认 off = 官方行为逐位一致；与 `--method` 正交，可叠加 fedta / fedta_a2 |
| `--route_fix` | off | **A2-R 路由修复开关**（R1+R2，探针 `diagnostics/probe_route_fix.py` 判决 keyH 5/5 复活 +3~4 bits 且准确率守住）：R1 = prompt_key 正交球面初始化（独立 RNG 流，seed=`--seed`）+ R2 = 可导软路由 `w=softmax(sim/τ)` 全池加权（token 数 length×top_k 不变；训练/推理同款全软）。默认 off = 官方行为逐位一致；与 `--method`/`--a2_arm`/`--skip_fed_avg_head` 正交 |
| `--route_tau` | 0.1 | R2 软路由温度 τ（仅 `--route_fix` 时生效；探针验证值 0.1） |

## 5. 运行管理参数（Phase 0 工程，不影响算法）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--run_name` | 自动生成 | `{dataset}_{method}_c.._t.._ge.._le.._lr.._seed.._{hash}` |
| `--resume` | `''` | `''` 全新 / `auto` 从 latest.pth / 显式 checkpoint 路径 |
| `--save_every` / `--keep_last` | 1 / 0 | checkpoint 频率与保留数（latest 与 task_XX_end 恒保留） |
| `--eval_all_every` | 0 | 每 N 轮全任务评估（0 = 仅任务结束时） |
| `--deterministic` | off | 确定性模式（默认关闭 = 官方行为） |
| `--max_rounds` | 0 | 总轮数上限（仅截断终点） |
| `--no_instrumentation` | off | 关闭插桩（observer-invariance 测试用） |

## 6. 断点续训（resume）

```bash
# 自动从最近 checkpoint 恢复（要求 --run_name 等参数与原运行一致）
python main.py datasets_delay --data_name cifar100 --run_name <原run_name> --resume auto

# 显式指定 checkpoint 路径
python main.py datasets_delay --data_name cifar100 \
    --resume output/cifar100/fedta/<run_name>/seed_42/checkpoints/task_02_end.pth
```

checkpoint 在轮边界保存，恢复时 RNG 精确还原；关键参数校验不一致拒绝恢复。

## 7. 磁盘占用提醒

checkpoint 含完整 ViT state_dict，**约 350MB+/轮**。磁盘紧张时：`--save_every 5 --keep_last 2`。

## 8. 运行产物目录结构

```
output/{data_name}/{method}/{run_name}/seed_{seed}/
    ├── command.txt / args.json / git_commit.txt
    ├── logs/train.log              # 终端输出完整副本（追加式）
    ├── checkpoints/                # latest.pth / round_XXXX.pth / task_XX_end.pth
    └── metrics/
        ├── accuracy_matrix.csv     # R[t][k] 全任务准确率矩阵
        ├── summary_metrics.csv     # AA / Forgetting / BWT 汇总
        ├── train_log.csv           # 结构化训练日志（round, task, client, accuracy, phase, test_task）
        ├── client_X_accuracy.csv   # 各客户端准确率
        ├── margins.csv / round_metrics.jsonl
```

外部 baseline adapter（§17–19）产物落 `output/{dataset}/{method}/manifest_adapter/seed_{seed}/`，metrics CSV 格式与上述一致（多 `meta.json` + `records.jsonl`）。

## 9. 诊断工具（不改变正式训练）

所有命令在项目根目录执行；`<run_dir>` 指 `output/{data_name}/{method}/{run_name}/seed_{seed}`。诊断脚本通过共享的 `build_server()` 复刻训练 RNG 流，保证与训练完全相同的数据划分。各脚本的完整判决记录见 FEDSMTA_ROADMAP.md。

### 9.1 提取特征（z_ref / z_op）

```bash
python diagnostics/extract_features.py \
    --run_dir <run_dir> \
    --checkpoint <run_dir>/checkpoints/task_04_end.pth \
    --out_dir diagnostics_output
```

- 输出 `features_{data_name}_{ckpt名}.npz`（z_ref = frozen ViT 特征、z_op = prompt 增强特征；含 labels/client_ids/task_ids/round_ids/splits/sample_idx_relative）+ 同名 `.json` 元信息
- 正式协议默认 `--split train --batch_size 16`（z_op 依赖 batch 组成，B 必须与 precheck 一致）；`--prompt_task_mode current` 使 z_op 用当前任务 prompt 对所有已见任务计算（= native evaluate 的 task_id=self.task_id 语义；默认 test = 被抽取任务自身的 prompt，非默认 mode 进文件名防缓存串用）
- deterministic extraction 与 precheck 同协议（CUBLAS env 前置 + cudnn flags bootstrap 后设置）

### 9.3 兼容性 2×2 诊断（compatibility_drift.py）

```bash
python diagnostics/compatibility_drift.py \
    --run_dir <run_dir> \
    --old_ckpt <run_dir>/checkpoints/task_02_end.pth \
    --new_ckpt <run_dir>/checkpoints/task_04_end.pth \
    --old_task 2
```

评估 old/current prompt × old/current key-anchor 四组合（旧 task 用 task-specific Chead）。输出 `<run_dir>/metrics/compatibility_2x2.csv`。

### 9.3.1 4 组合 per-class 通道分解（analyze_class_retention.py）

```bash
python diagnostics/analyze_class_retention.py \
    --run_dir <run_dir> \
    --old_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth \
    --new_ckpt checkpoints/task_04_end.pth \
    --out_csv diagnostics_output/class_retention_4combo.csv
```

P 通道 = oo−co（聚合 prompt 漂移）、KA 通道 = oo−oc（本地 key/anchor 漂移）；public/private 由 manifest 自证（≥2 客户端持有 = public）；固定 shuffle 种子 `9700+100*cid+old_task`；样本加权。

### 9.3.2 子机制判别探针（probe_channel_correlates.py）

```bash
python diagnostics/probe_channel_correlates.py --csv diagnostics_output/class_retention_4combo.csv
```

消费 9.3.1 的 CSV，检验间接共适应 vs 协议结构性脆弱两候选子机制（判决：后者否证、前者修正版成立）。

### 9.3.3a 路由结构识别（verify_slot_binding.py，D0）

```bash
python diagnostics/verify_slot_binding.py \
    --run_dir <run_dir> \
    --task_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth,checkpoints/task_04_end.pth \
    --out_csv diagnostics_output/slot_binding.csv \
    --out_slot_csv diagnostics_output/slot_binding_slots.csv \
    --out_npz diagnostics_output/slot_binding_matrices.npz
```

绑定集中度 / 槽位共享 / 路由 2×2 分解 / 完整 class×slot 绑定矩阵。固定种子物化一份增强视图隔离增强噪声；无自动判决。

### 9.3.3b D1 v2 碰撞暴露诊断（collision_exposure_diagnostic.py，零前向）

```bash
python diagnostics/collision_exposure_diagnostic.py \
    --run_dir <run_dir> \
    --task_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth,checkpoints/task_04_end.pth \
    --binding_csv diagnostics_output/slot_binding.csv \
    --binding_npz diagnostics_output/slot_binding_matrices.npz \
    --combo_csv diagnostics_output/class_retention_4combo.csv \
    --out_csv diagnostics_output/collision_exposure.csv \
    --out_json diagnostics_output/collision_exposure.json
```

E_future → d_key/d_anchor（含方向/范数分解）→ KA 链条检验；纯张量运算约 1 分钟。判决须过 (client,task) FE（Roadmap §3.7）。

### 9.3.3c 落点分析（analyze_landing_spots.py，本地离线）

```bash
python diagnostics/analyze_landing_spots.py \
    --binding_npz diagnostics_output/slot_binding_matrices.npz \
    --damage_csv diagnostics_output/collision_exposure_local.csv \
    --out_csv diagnostics_output/landing_spots.csv \
    --out_json diagnostics_output/landing_spots.json
```

集中坍缩/attractor 假说检验；秒级本地运行。

### 9.3.4 TKR 精确口径复算（verify_tkr.py，零 GPU）

```bash
python diagnostics/verify_tkr.py \
    --run_dir <run_dir> \
    --out_prefix diagnostics_output/tkr_seed42
# 或直接指定 train_log：
python diagnostics/verify_tkr.py --train_log <path>/train_log.csv --global_epoch 5 --out_prefix diagnostics_output/tkr_seed42
```

按作者 issue 回复定义复算 TKR（分母 = round4 聚合全局模型、分子 = t>0 本地模型）。三种子复算结案：平均 96.4/98.4/96.1/89.6% 吻合论文 "around 98%"（Roadmap §3.8）。

### 9.3.5 P1 frozen NCM probe（ncm_probe.py，A5 Gate A）

```bash
python diagnostics/ncm_probe.py \
    --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
    --checkpoints task_00_end.pth,task_04_end.pth \
    --out_csv diagnostics_output/ncm_probe_inr_a2.csv \
    --c2p "86.86,91.14"
```

- z_ref（主结果，可在线增量维护）+ z_op（oracle，回溯访问上界）上的 cosine/Euclidean NCM vs native；原型仅来自已见任务 inner.train（防泄漏）；候选限制在 (client, task) class_mask（与 native R[t][k] 同语义）
- fail-closed：缓存身份校验（run_dir/seed/checkpoint/mode 不匹配即报错）、原型全覆盖、aa_native 与 accuracy_matrix.csv 对账（>0.02pp 拒绝）
- `--prompt_task_mode current`（默认）使 z_op 对齐 native evaluate 语义；特征缓存按 mode 分文件
- **T0-gap 判据**：T0 行对照 C²P 参考（86.86）——NCM 若大幅缩小 ~4pp 差距，A5 优先做分类接口；增益只在 T4 出现才支持持续迁移叙事
- **已跑结果（INR seed42，2026-10-20）**：AA T0/T4 = native 82.87/79.43、z_ref NCM 79.06/76.04、z_op NCM 82.49/80.62（C²P 86.86/91.14）。判决：T0-gap 非分类接口瓶颈（z_op NCM≈native Chead），纯统计原型分类接口判负，A5 共享语义层必须可训练（Roadmap §6）

### 9.3.6 Oracle headroom probe（probe_oracle_headroom.py，A5 Gate A-2）

**问题**：NCM 判负只证明"统计读出饱和"，没有证明"可训练接口能多榨出 ~4pp"——C²P 的 86.86 是在它自己特征上拿的，T0 差距可能根本不在我们特征里。本探针在**任何 A5-v2 工程开工前**判决：可训练接口在我们特征上有没有 headroom。

```bash
# INR A2 臂（特征缓存已存在于 ncm_inr_a2_seed42/，零提取，纯 CPU 分钟级）
python diagnostics/probe_oracle_headroom.py \
    --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
    --checkpoints task_00_end.pth,task_04_end.pth \
    --features_dir diagnostics_output/ncm_inr_a2_seed42 \
    --out_csv diagnostics_output/oracle_headroom_inr_a2.csv \
    --c2p 86.86,91.14

# CIFAR-100 cifar_hsn 臂（唯一有完整 checkpoint + 标准 metrics 的全协议 cifar run；
# 需先提特征，GPU 几分钟；探针本身 CPU）
python diagnostics/probe_oracle_headroom.py \
    --run_dir output/cifar100/fedta/cifar_hsn/seed_42 \
    --checkpoints task_00_end.pth,task_04_end.pth \
    --out_csv diagnostics_output/oracle_headroom_cifar_hsn.csv
```

- **cifar 辅助读数的 caveat**：cifar_hsn 带 `skip_fed_avg_head=True`（HS-N 修复臂，非纯官方）。z_ref（冻结主干）读数不受任何影响；z_op 的 prompt 训练路径不含 vit.head，大概率等价官方但未验证。官方 D 臂（route_fix off）不可用于本探针：`max_rounds=1` 短程协议无 client CSV / accuracy_matrix / task_XX_end.pth，native 对账管线按设计 fail-closed 拒绝。Phase-0 官方全协议 cifar run 的 checkpoint 已清理，同样不可用。

- **三探针**（每特征空间 z_ref/z_op 各一套，全部只在 inner.train 上训练、inner.test 上评估、候选限制 (client, task) class_mask = native R[t][k] 同语义）：
  - `linear_local`：每 (client, task) 逻辑回归——**sanity 锚**，应落在 native 附近（同为线性可训练接口）
  - `mlp_local`：每 (client, task) 2 层 MLP——本地-only 可训练接口天花板（未来 "local-only g" 消融的预演）
  - `mlp_pooled`：每边界一个 MLP，池化训练全部客户端已见任务 train 行——**信息量 oracle**（跨客户端池化+标签，违反联邦协议的纯诊断上界，不是方法）
- **分解读法**：`delta_pooled = mlp_pooled − native` = 总可提取 headroom（Gate 输入）；`delta_local = mlp_local − native` = 不联邦也能拿到的部分；`federated_premise = mlp_pooled − mlp_local` = FedAvg 共享接口的收益预览（≈0 则 A5-v2 的联邦主张降级）
- **fail-closed**：复用 ncm_probe 全套（缓存身份校验、aa_native 与 accuracy_matrix.csv 对账 >0.02pp 拒绝、mask 类训练行全覆盖缺失即报错）；标准化统计只 fit train 行；seed 派生自探针身份（SHA256），逐位可复现（单测 8/8 PASS）
- **预注册 Gate（只看 T0 行、z_op 主读数；T4 与 z_ref 仅作上下文，不进 Gate）**：
  - GREEN：`delta_pooled ≥ +2.00pp` → A5-v2（可训练共享语义接口）立项
  - RED：`delta_pooled ≤ +1.00pp` → 接口线判死，T0 差距在特征质量（C²P 的 prompt+对比训练），方向转 feature-shaping 组件——**A5-v2 整个方案免建**
  - YELLOW：之间 → 连数字送外审裁决
- 探针超参（hidden 256 / epochs 100 / lr 1e-3 / wd 1e-4 / dropout 0.1 / batch 64）与 Gate 阈值均写入 summary.json 溯源

### 9.4 已删除命令（tombstone）

多模态分析（旧 §9.2）、Synthetic Oracle-K2（旧 §9.4）、z_op precheck（旧 §14）、原版 slot 碰撞诊断（旧 §9.3.3）——所属假说已判负或设计判死，存档见 FEDSMTA_ROADMAP.md §4，原命令在 git 历史。

## 10. 官方 FedTA parity 测试（Gate 0A，已 PASS 封板）

```bash
# smoke（task_num=1/rounds=5/local_epoch=1）与正式（task_num=2/rounds=10/local_epoch=1）：
python tests/test_official_fedta_parity.py \
    --official_repo /path/to/official_FedTA \
    --data_path ./local_datasets \
    --global_epoch 5 --task_num 2 --rounds 10 --local_epoch 1 \
    --batch_size 16 --surrogate_num 20 --threshold 0.25 \
    --seed 42 --device cuda
```

证明 `--method fedta --no_instrumentation ≡ 官方 FedTA`（13 项 D=0）。Gate 0A 强制 `global_epoch=5`（官方 SIKF 硬编码 `!= 4` 仅此时等价）；`--rounds` 必须等于 `task_num*global_epoch`；PASS 不容忍浮点误差；配置漂移 fail loudly。细节见脚本 docstring 与 git 历史。

## 11. Resume 回归测试（Gate 0B，已 PASS 封板）

```bash
python tests/test_resume_regression.py \
    --data_name cifar100 --data_path ./local_datasets \
    --rounds 4 --split_at 2 --global_epoch 2 --task_num 2 --local_epoch 2 \
    --seed 42 --device cuda
```

连续跑 vs 断点续跑逐位一致（模型/RNG/数据划分/accuracy matrix）；manifest 缺失即 FAIL（防 `{} == {}` 假 PASS）。加 `--method fedta_a2 --a2_arm anchor` 验证 a2_state 逐位比较。

## 12. Observer-invariance 测试（Gate 0C，已 PASS 封板）

```bash
python tests/test_observer_invariance.py \
    --data_name cifar100 --data_path ./local_datasets \
    --rounds 3 --global_epoch 2 --task_num 2 --local_epoch 2 \
    --seed 42 --device cuda
```

插桩 ON/OFF 训练轨迹逐位一致 + artifact 确证区分。

## 13. Phase 0 Freeze（已执行）

三 Gate 全 PASS 后 commit + tag `phase0-fedta-baseline`。此后 `--method fedta` read-only。

## 14. 硬约束（长期有效）

- 不得改动 FedTA loss / optimizer / BGPS / SIKF / FedAvg head / Tail Anchor 结构 / prompt 逻辑 / 联邦划分
  （例外：`--skip_fed_avg_head` 开关——默认关闭时官方路径逐位不变，开启 = HS-N 论文语义修复，GPT 裁决批准走开关不直接改基线）
- 活跃路线见 FEDSMTA_ROADMAP.md（A2 接口保护 + A5 共享语义接口 + 外部对比实验）；完整约束清单见 Roadmap §9

## 16. Phase 4 修复实验（A2 训练臂 + A1 快照评估）

> A2 = post-step restore（freeze + wd 豁免，优化器零改动）；保护集 = 已完成任务训练特征经当前 key top-1 路由的支撑槽位并集。判读（预注册）：anchor 臂 KA_rel 降 = 主通道因果成立；both 臂逼近 co 上界且 cc 不受损。

```bash
# A2 三臂训练（CIFAR-100 seed 42，逐臂依次）
python main.py --method fedta_a2 --a2_arm anchor --seed 42 --global_epoch 5 --threshold 0.25 --data-path <DATA_PATH>
python main.py --method fedta_a2 --a2_arm key    --seed 42 --global_epoch 5 --threshold 0.25 --data-path <DATA_PATH>
python main.py --method fedta_a2 --a2_arm both   --seed 42 --global_epoch 5 --threshold 0.25 --data-path <DATA_PATH>

# ImageNet-R（固定 surrogate_num 5）
python main.py --method fedta_a2 --a2_arm anchor --seed 42 --global_epoch 5 --threshold 0.25 --surrogate_num 5 --data-path <DATA_PATH>

# A2 smoke（1 任务应输出 protected 0/200 slots 并与 control 无差异）
python main.py --method fedta_a2 --a2_arm both --seed 42 --task_num 1 --global_epoch 5 --local_epoch 1 --run_name a2_smoke

# A1 快照评估（纯推理，对 baseline 或 A2 run 均可）
python diagnostics/a1_snapshot_eval.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --out_csv diagnostics_output/a1_snapshot_eval.csv
```

- `--a2_arm` 仅在 `--method fedta_a2` 下生效；A2 checkpoint 含 a2_state，resume fail-closed
- A1 预期 = 2×2 co 组合预注册值（T1 71.4→93.9 等）；±2pp 内 batchwise_prompt 噪声正常

### 16.1 A2/A1 回归测试（正式四臂前必须全过）

```bash
python tests/test_resume_regression.py --method fedta_a2 --a2_arm anchor --data-path <DATA_PATH>  # T1
python tests/test_a2_rejections.py --data-path <DATA_PATH>                                        # T2
python tests/test_a1_parity.py --data-path <DATA_PATH>                                            # T3
# T0 = Gate 0A smoke + 0B/0C 重跑（§10–12）
```

执行顺序：T0 → T1 → T2 → T3 → A1 正式评估 → A2 三臂正式训练。

## 17. 外部 baseline：FedMGP（原仓库零编辑 + protocol adapter）

> 官方仓库克隆于 `external_baselines/FedMGP`，零编辑；注入/修复经 `protocol_adapters/` 运行时 monkey-patch。`local_datasets`/`pretrain_model` 为 symlink。

```bash
# 17.1 协议 manifest（已生成冻结，四重 seal PASS）
python diagnostics/export_split_manifest.py \
    --reference-checkpoint output/cifar100/fedta_a2/a2_anchor/seed_42/checkpoints/latest.pth \
    --verify-task0-replay \
    -o protocol_manifests/cifar100_c5t5p15s20_seed42.json

# 17.2 protocol equivalence gate（训练前必须 PASS，已 PASS 2026-10-18）
python protocol_adapters/gate_fedmgp.py \
    --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json

# 17.3 正式训练（tmux）
tmux new -s fedmgp_seed42
cd /root/autodl-tmp/users/ych/FedSMTA
python protocol_adapters/run_fedmgp.py \
    --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
    --device cuda
```

- 协议冻结（与 manifest 不符退出）：cifar100、5c×5t、p15、s20、ge5、seed 42；local_epoch 默认 30；method-native：lr 0.001875 / threshold 0.1 / batch 16；未知 flag 转发 FedMGP config
- 官方仓库 3 个 bug（update_data 缺失 / PIL 转换被注释 / np.arange(200) CUDA assert）已运行时修复
- 空闲 GPU 约 12–15h；FedSMTA conda env 直接可用（torch 1.12.1 + timm 0.6.7，deeplake 已 stub）

## 18. 外部 baseline：C²Prompt（ImageNet-R，原仓库零编辑 + adapter）

> 只在 ImageNet-R 对齐（官方 CIFAR 路径是私有 500 虚拟类协议，不兼容）。`external_baselines/NeurIPS2025-C2Prompt` 零编辑；注入经 `protocol_adapters/run_c2p.py`（main.py 7 组 8 处 + vit_coda_p_hard.py 2 处源码补丁 + dataset override + nni stub）。训练 = manifest inner.train；评测 = 原生 global eval（累计已训类）× manifest test_pool。
> **数据访问 = prev-task full local re-access**（mover 分支 L673-685 每新 task 对本客户端 prev task 全量训练数据重跑 train_class；官方日志逐位一致证实激活）——数据不出本地但违反 exemplar-free 设定，主表数据访问列照此标注。
> 官方仓库 2 个 cadence-latent bug（global_class_stats 残留 std UnboundLocalError + 温度硬编码表溢出）已运行时修复（边界轮官方语义逐位不变）。
> 结果（seed42 after-T4 AA = 91.14，T0 = 86.86）：T0 领先来自 per-class 原型头+对比损失；低遗忘部分被 prev-task 重训兜底（Roadmap §6 差距记账）。

```bash
# 18.1 manifest（已生成冻结，四重 seal PASS）
python diagnostics/export_split_manifest.py --dataset ImageNet-R \
    --reference-checkpoint output/ImageNet-R/fedta/inr_base_p40/seed_42/checkpoints/latest.pth \
    --verify-task0-replay \
    -o protocol_manifests/inr_c5t5p40s5_seed42.json

# 18.2 gate（已 PASS 2026-10-18）
python protocol_adapters/gate_c2p.py \
    --manifest protocol_manifests/inr_c5t5p40s5_seed42.json \
    --data-root local_datasets/imagenet-r \
    -o protocol_adapters/gate_c2p_report.json

# 18.3 冒烟（已通过 2026-10-18：1-task Avg=87.70 / 2-task Avg=90.43）
python protocol_adapters/run_c2p.py \
    --manifest protocol_manifests/inr_c5t5p40s5_seed42.json \
    --epochs_global 6 --tasks_global 6 --epochs_local 1

# 18.4 正式训练（AutoDL——izfree 24GB OOM）
python protocol_adapters/run_c2p.py \
    --manifest protocol_manifests/inr_c5t5p40s5_seed42.json
```

- 协议冻结：INR、200 类、5c×5t、epochs_global 25、seed 42；task progression 替换为全 client 前进（官方 50% 采样是 3-task 专用语义）
- method-native：lr 0.005 / batch 64 / epochs_local 10 / global_update_lr 100 / prompt_param `25 10 10 8 0 0 6 10 8`
- 环境：torch 1.12.1 + timm 0.6.7 + numpy 1.23.5；首次运行下载 timm augreg 权重 ~330MB

## 19. FL-only 构造基线（FedAvg / FedProx / FedNova，自实现 adapter）

> 主表 FL 下界三行，构造基线（FedTA 论文 Table 1 同样如此，官方仓库无独立实现），标注 "constructed following FedTA paper Table 1 setting"。`protocol_adapters/run_fl_baseline.py`：ViT-B/16 预训练 + 全参数联邦微调 + 固定 nb_classes 头（class-IL）+ 顺序任务流无 replay。数据 = manifest 冻结划分（训练 inner.train、评测 inner.test + class_mask 掩码）。
> 方法差异：fedavg 样本加权；fedprox 全参 prox（主表 μ=0.01）；fednova 步数归一化再加权（同质步数精确退化为 fedavg）。本地优化器对齐 FedTA 血统：Adam(0.001, wd 1e-3)、CE、batch 16、30 epochs × 5 rounds/task。

```bash
# CIFAR-100（manifest 复用 §17.1）
for m in fedavg fedprox fednova; do
  python protocol_adapters/run_fl_baseline.py \
    --manifest protocol_manifests/cifar100_c5t5p15s20_seed42.json \
    --method $m
done

# ImageNet-R（manifest 复用 §18.1；FL 行预期 2-8 级别近乎全崩，量级对照 FedTA Table 1）
for m in fedavg fedprox fednova; do
  python protocol_adapters/run_fl_baseline.py \
    --manifest protocol_manifests/inr_c5t5p40s5_seed42.json \
    --method $m
done
```

- 常用 flag：`--fedprox_mu 0.01`（仅 fedprox）；`--save_ckpt`（每个 ~350MB）；`--resume auto`（每轮写 latest.pth）；`--max_tasks N`（冒烟）
- 产物落 `output/{dataset}/{method}/manifest_adapter/seed_{seed}/`，CSV 格式与主线一致，直接进三级（分数据集/分种子/分任务）聚合
- 无 `--seed` 参数：seed 由 manifest 锁定（换种子需先 export 对应 manifest）
- 冒烟已通过（2026-10-20）+ 聚合公式单测（含 fednova 退化精确性）
- 显存约 6-8GB + 权重缓冲 ~1.7GB，24GB 4090 可跑；无 tmux 时直接顺序跑（izfree 不需要后台保护）

## 20. A2-R 路由修复消融（`--route_fix`，2×2 矩阵新增两臂）

> 诊断结论：官方 FedTA 的 Global_Prompt 路由结构性死锁——top-k 硬选择不可导 + uniform 初始化 + pull 只拉选中槽 → 全样本 100% 命中单槽（keyH≈0.004 bits），prompt 池退化为固定 bias。微训练探针 `diagnostics/probe_route_fix.py`（已验证）判决 R1+R2 使路由 **5/5 客户端复活（keyH +3~4 bits）且准确率守住**。
> **2026-10-27 修订（全轨迹判决）**：首臂 round-0 key 塌缩（有效秩 87.8→6.5、mean|cos| 0.03→0.49），首版全池软 pull（`reduce_sim = top_k·Σ(w·sim)/B`）与塌缩同时出现，判定为**强候选驱动机制**（w 近均匀时把全部 key 拉向特征均值方向，FedAvg 抵消各客户端判别分量后仅存共性方向；是否为唯一根因待 A 方案真实训练对照验证）。修订为 **A 方案 pull 解耦**：pull 槽位选择逐字复刻官方路径（detach 的 sim 做 top-k + batchwise 批内多数投票），pull 梯度仅经未 detach 的被选中相似度回传；CE 路径保持全池软路由。纯 torch 自检 4/4 PASS（pull 梯度仅选中槽 / CE 梯度 100 槽全达 / gather 版与官方公式**数学值与梯度等价**（浮点累加顺序不同，逐位等价需容差测试验证）/ CE−0.1×pull 形态下 pull 梯度隔离）。τ 保持 0.1，R3' 暂不加入，一次只检验一个机制。
> 配套只读门控：`diagnostics/route_weight_stats.py`（round checkpoint → 路由熵 / 最大权重 / top-1 使用 / 样本间 w 余弦 + key 有效秩）。**门控判据（全局聚合与每客户端聚合须同时满足，防全局指标掩盖单客户端塌缩）**：样本间 w 余弦 < 0.5 + 使用熵 > 3 bits + key 有效秩 > 50（key 池全局共享，有效秩只判全局一次）；任一全局/客户端聚合余弦 > 0.9 判 DEGENERATE；client-task 级余弦 > 0.9 仅作信息性提示（小样本噪声，不翻门控）。**退出码：0=PASS / 1=FAIL 或 DEGENERATE / 2=NOT_APPLICABLE**；`route_fix=False` 的 checkpoint 判 NOT_APPLICABLE（官方硬路由不消费 softmax 权重，其 PASS 无放行资格，仅作管线冒烟）。脚本为**分布级健康检查**（manifest 重建外层划分 + 新采样随机变换，非训练前向逐样本复现）；仅支持 `embedding_key='cls'`（校验不满足即报错）。
> R1 = prompt_key 正交球面初始化（独立 RNG 流，seed=`--seed`，零全局流扰动）；R2 = 可导软路由 `w = softmax(sim/τ)` 全池加权混合（τ=`--route_tau` 默认 0.1），**token 数保持 length×top_k 不变**（pos_embed / 官方聚合 / head_type='prompt' 消费者不受影响）；训练与推理同款全软。默认 off = 官方行为逐位一致；开启 run 无需与基线逐位对齐（RNG 消耗变化属方案的一部分）。
> 2×2 矩阵（route_fix × A2）：已有两臂 = 官方基线（fedta）与 A2 key+HSN 宿主形态（`cifar_a2_key_hsn`）；本节补齐两个新增臂。与 `--skip_fed_avg_head` / `--a2_arm` 正交（A2 在 Client post-step restore 操作 Tail_Anchor key/anchor_pool，路由在前向 Global_Prompt，无相互作用）。
> **新流程**：第一臂 round-0 checkpoint 落地后先跑 route_weight_stats 门控，PASS（退出码 0）才继续完整轨迹（判据：样本间 w 余弦 < 0.5 + 使用熵 > 3 bits + key 有效秩 > 50，且全局聚合与每个客户端聚合须同时满足）。

```bash
# 20.1 新增臂 1：route_fix 单独（官方基线协议 + --route_fix，CIFAR-100 seed42 全轨迹）
python main.py datasets_delay \
    --data_name cifar100 --method fedta \
    --route_fix --route_tau 0.1 \
    --seed 42 --global_epoch 5 --threshold 0.25 \
    --run_name cifar_route_fix \
    --data-path <DATA_PATH>

# 20.1a 只读路由门控（round-0 checkpoint 落地后、放行完整轨迹前）：
python diagnostics/route_weight_stats.py \
    --run_dir output/cifar100/fedta/cifar_route_fix/seed_42 \
    --round 0 --device cuda \
    --data_path <DATA_PATH>

# 20.1b 诊断脚本管线冒烟（用未删除的 route_fix=False 基线 checkpoint；
#      预期 verdict=NOT_APPLICABLE、退出码 2，仅验证加载/重建/特征提取链路）
python diagnostics/route_weight_stats.py \
    --run_dir output/cifar100/fedta/cifar_hsn/seed_42 \
    --round 0 --device cuda \
    --data_path <DATA_PATH>

# 20.2 新增臂 2：route_fix × A2 组合（与已跑 a2_key_hsn 宿主形态全同 + --route_fix；20.1 门控 PASS 后再启动）
python main.py datasets_delay \
    --data_name cifar100 --method fedta_a2 --a2_arm key \
    --skip_fed_avg_head \
    --route_fix --route_tau 0.1 \
    --seed 42 --global_epoch 5 --threshold 0.25 \
    --run_name cifar_a2_key_hsn_routefix \
    --data-path <DATA_PATH>
```

- 判读（预注册）：route_fix 臂 train_log/checkpoint 侧 keyH 相对基线复活（探针阈值 ΔkeyH > 0.5 bits）、准确率不劣于 −2pp；组合臂与 20.1 分离 A2 接口保护与路由修复的各自贡献
- resume：`route_fix`/`route_tau` 随 args 全量入 checkpoint（`build_checkpoint.args`）；行为开关以命令行传参为准（模型参数从 state_dict 恢复），跨开关 resume 自行保证传参一致
- 冒烟（1 任务 + protected 0 slots 预期）：
  `python main.py datasets_delay --data_name cifar100 --method fedta_a2 --a2_arm both --skip_fed_avg_head --route_fix --task_num 1 --global_epoch 5 --local_epoch 1 --seed 42 --run_name routefix_smoke`

### 20.3 `--route_audit` 塌缩取证插桩（A 方案门控 FAIL 后的定位诊断，2026-10-27）

> **触发事由**：A 方案（pull 解耦，commit b4311c8+be2d491）round-0 门控 **FAIL**（key 有效秩 87.8→6.59、样本间 w 余弦 0.8726、使用熵 0.85 bits，全局与 5 客户端全挂）——全池软 pull 被**证伪为塌缩的必要条件**。剩余嫌疑：CE 全池软路由梯度路径、客户端本地训练、SIKF 蒸馏融合（本仓库 prompt key **无任何 FedAvg**：`fed_avg_prompt` 死代码、5 客户端共享同一 ViT、keys 的"聚合"= `kd_fusion_prompt` 蒸馏）。
> **结构性疑点（插桩将实证）**：`kd_fusion_prompt` 的 Adam 作用于 `my_result.parameters()`，但前向经 `load_prompts()` deepcopy 出的 `self.model.prompt` 运行——梯度落在模型内的一次性副本上，`my_result` 的 `.grad` 恒为 None（Adam 跳过 grad-None 参数）→ **蒸馏疑似结构性 no-op**（server prompt ≡ client-0 分支原样）。若属实，塌缩完全发生在本地训练阶段。
> **实现**：`--route_audit`（默认 off = 官方路径零改动）。ON 时只读插桩：A 时刻（分支起点 key 快照 + sha256 实证 5 个 round-0 再初始化是否逐字节一致，不因共享 route_seed 而假设）、每客户端首 minibatch CE/pull 梯度审计（`torch.autograd.grad`，retain_graph，不写 `.grad`、不耗 RNG，训练逐位不变——单测 `tests/test_route_audit_invariance.py` 5/5 PASS 含 off/on 逐位等价）、B 时刻（各自 `self.prompts` 快照，**不读共享 ViT**）、跨分支对齐（同槽余弦/跨分支 Gram/子空间主角度——区分"分支内部塌缩"与"分支间子空间漂移"）、C 时刻（pre/post 融合 + 蒸馏首 batch 对**模型侧 key 与优化器侧 key 双梯度**审计，直接判定 no-op）。输出 `<run_dir>/route_audit/*.jsonl+npz`。

```bash
# 20.3a 塌缩取证 run（与 20.1 全同协议 + --route_audit + --max_rounds 1，新 run_name 保隔离）
python main.py datasets_delay \
    --data_name cifar100 --method fedta \
    --route_fix --route_tau 0.1 \
    --route_audit \
    --seed 42 --global_epoch 5 --threshold 0.25 \
    --run_name cifar_route_audit \
    --max_rounds 1 \
    --data-path <DATA_PATH>

# 20.3b 插桩不变性集成验证：本 run（audit ON）的 round_0000.pth 应与
#      be2d491 无插桩 run（cifar_route_fix/seed_42，同 seed 同协议）逐位一致
python - <<'EOF'
import torch
a = torch.load("output/cifar100/fedta/cifar_route_audit/seed_42/checkpoints/round_0000.pth",
               map_location="cpu")
b = torch.load("output/cifar100/fedta/cifar_route_fix/seed_42/checkpoints/round_0000.pth",
               map_location="cpu")
mism = [k for k in ("server_model", "server_prompt", "global_head")
        if not all(torch.equal(a[k][t], b[k][t])
                   for t in a[k] if torch.is_tensor(a[k][t]))]
print("MISMATCH:", mism if mism else "NONE (bitwise identical)")
EOF

# 20.3c 单元测试（含 off/on 逐位不变性 + no-op 检测器）
python -m unittest tests.test_route_audit_invariance -v

# 20.3d 官方 parity 回归（route_audit 默认 off 硬约束；官方仓库克隆所在机器执行）
python tests/test_official_fedta_parity.py \
    --official_repo /path/to/official_FedTA \
    --data_path ./local_datasets \
    --global_epoch 5 --task_num 2 --rounds 10 --local_epoch 1 \
    --batch_size 16 --surrogate_num 20 --threshold 0.25 \
    --seed 42 --device cuda
```

- 判读顺序（20.3a 跑完后）：① 5 个 `init` 快照 sha256 应一致（R1 再播种确定性）；② `init→post_local` 有效秩分解塌缩量（本地训练责任）；③ `cross_branch` 同槽余弦/子空间主角度（分支间漂移）；④ `fusion_audit` 的 `optimizer_target_grad_was_none`（True = 蒸馏 no-op 实证）；⑤ `pre_fusion` vs `post_fusion` sha256（不一致才有实际融合发生）
- 约束：`--route_audit` 不改变 loss/优化器/数据/RNG/客户端顺序；resume 跨 audit 开关安全（观察者性质，不入 RESUME_MUST_MATCH）

### 20.4 A2-R 塌缩归因消融矩阵（route_audit 取证判决后的 B/C/D 臂，2026-10-28）

> **取证判决（20.3a，seed 42）**：塌缩 100% 发生在本地训练（init 99 → post_local 3.5-6.6，5/5 分支）；SIKF 融合实证 no-op（`optimizer_target_grad_was_none=true`，pre/post sha256 相同）；pull 单槽集中、CE 全池耦合（softmax 分母把梯度送达全部 key）。审查裁决：优先隔离 CE 的 key 梯度（B 臂），τ=1.0 温度消融暂缓（同时改变前向与反向，且高温会使样本间权重余弦趋近 1，即使 key 秩改善也会挂在原门控上）。**暂不修复 SIKF no-op、不启动完整训练与 A2 组合臂**——先分清本地训练两条 key 梯度路径的责任，再决定修复组合（修 no-op 前必须先控制塌缩：教师输出两两余弦已 0.77-0.87，接通 1.3e5 量级蒸馏梯度只会加速趋同）。

| 臂 | CE 路由对 key 的梯度 | pull | 初始化 / τ | 已有结果 |
|---|---|---|---|---|
| A | 开 | 开 | 正交 / 0.1 | **塌缩**（cifar_route_audit，有效秩 3.5-6.6） |
| B | **关** | 开 | 正交 / 0.1 | 待跑 |
| C | 开 | **关** | 正交 / 0.1 | 待跑 |
| D | 官方硬路由（CE 本就无 key 通道） | 官方 | uniform | 待跑（历史参照） |

B/C/D 全部 `--max_rounds 1` 短程；B/C 带 `--route_audit`（D 也带：`log_client_grad_audit` 已兼容官方硬路由 `pull_idx=None`，还能实测官方 CE 对 key 无梯度通道这一结构性事实）。D 的 init 快照 sha256 各客户端**不**一致属预期（官方 uniform 消耗全局 RNG，每客户端构造结果不同）。

```bash
# 20.4a B 臂：CE-key 梯度隔离（--route_detach_ce_key_grad）
python main.py datasets_delay \
    --data_name cifar100 --method fedta \
    --route_fix --route_tau 0.1 \
    --route_detach_ce_key_grad \
    --route_audit \
    --seed 42 --global_epoch 5 --threshold 0.25 \
    --run_name cifar_route_ablate_B_detachce \
    --max_rounds 1 \
    --data-path /root/autodl-tmp/users/ych/FedSMTA/local_datasets

# 20.4b C 臂：去 pull（--disable_pull）
python main.py datasets_delay \
    --data_name cifar100 --method fedta \
    --route_fix --route_tau 0.1 \
    --disable_pull \
    --route_audit \
    --seed 42 --global_epoch 5 --threshold 0.25 \
    --run_name cifar_route_ablate_C_nopull \
    --max_rounds 1 \
    --data-path /root/autodl-tmp/users/ych/FedSMTA/local_datasets

# 20.4c D 臂：官方基线（route_fix off；审计兼容官方硬路由路径）
python main.py datasets_delay \
    --data_name cifar100 --method fedta \
    --route_audit \
    --seed 42 --global_epoch 5 --threshold 0.25 \
    --run_name cifar_route_ablate_D_official \
    --max_rounds 1 \
    --data-path /root/autodl-tmp/users/ych/FedSMTA/local_datasets

# 20.4d 单测（B/C 开关语义 + 默认 off 逐位不变性）
python -m unittest tests.test_route_ablation_switches -v
```

判读表（对照 A 臂 init 99 → post_local 3.5-6.6）：

- **B 不塌、C 塌**：CE 全池软路由梯度是主驱动 → 修复方向 = 改造 CE 路由梯度（detach / 只达选中槽 / 调 τ）
- **B 塌、C 不塌**：pull 或其与 CE 的交互是主驱动 → 修复方向 = 改造 pull
- **B、C 都不塌**：两条路径共同促成 → 需进一步检查交互效应（可能要 B+C 组合臂）
- **B、C 都塌**：嫌疑转向其他本地因素（优化器 / weight_decay=1e-3 / key 范数塌缩参与率口径），需逐槽更新轨迹分析

判据口径与 20.3 相同：post_local 有效秩（主要指标，>50 健康线）、样本间/分支间余弦、sha256。注意单 minibatch 的梯度范数（grad_audit 首批快照）不代表整个 30 epoch 的累计作用，责任判定以 post_local 快照为准。D 臂不能与 A 直接归因到单一梯度通道（route_fix 同时切换 R1 初始化，非干净对照），只作历史参照 + 官方"CE 无 key 通道"的结构性实证。

### 20.5 E1 修复臂：forward-soft / backward-selected CE routing（2026-10-28 终审裁决）

**判读实测（§20.4 出案，终审批准）**：命中判读表第一行——B 断 CE→key 后 4/5 客户端保秩 81-91（c0 个例 3.92 仍塌），C 留 CE 去 pull 则 5/5 全塌 3.4-8.3，D 官方结构性无 CE→key 通道（g_ce_exists 全 False）且从不塌。正式表述：**CE 全池软路由梯度是主驱动（dominant but not exclusive）**；c0 残余失败提示 pull+wd 次级退化通路。B 臂副作用警告：纯 detach 后 key 范数全体萎缩（mean 0.0067，wd 磨掉无梯度 key）——rank healthy ≠ key pool healthy，纯 detach 不可作正式方案。

**E1 = --route_selected_ce_grad**：前向与 route_fix 逐位一致；CE 对 key 梯度只经 per-sample detached top-k 选中槽回传（刻意不做 batchwise 多数——修复目标正是"全池被共同拉向同一方向"）；CE 对样本特征梯度完整（sim_no_key 提供全量 x 通道）；pull 保持官方语义。暂不调 τ（τ 会同时改前向 mixture/熵/梯度幅值/Jacobian，混因果；E1 成功后 τ 才作为第二阶段性能超参数）、暂不做范数维持补丁（机制绕）。

```bash
# 20.5a E1 臂（--max_rounds 1 短程，与 B/C/D 同协议对照）
python main.py datasets_delay \
    --data_name cifar100 --method fedta \
    --route_fix --route_tau 0.1 \
    --route_selected_ce_grad \
    --route_audit \
    --seed 42 --global_epoch 5 --threshold 0.25 \
    --run_name cifar_route_repair_E1_selectedce \
    --max_rounds 1 \
    --data-path /root/autodl-tmp/users/ych/FedSMTA/local_datasets
```

**四 Gate（E1 升格为正式方法模块的前置条件，全部逐 client 报告，不只报均值）**：

| Gate | 判据 | 参照 |
|---|---|---|
| 1. Key geometry | post_local 有效秩保持在明显高秩区（**>70**） | A 臂塌到 3.5-6.6；D 官方 ≈88 |
| 2. Key norm | 无 B 臂式萎缩：key_norm_mean 量级正常、`key_frac_norm_lt_1e-3`/`key_frac_norm_lt_1e-2` 两档 ≈0 | B 臂 mean 0.0067（route_audit 已新增两档 near-zero fraction 字段） |
| 3. Routing utilization | 不退回官方单槽垄断：熵 ↑、top-1 dominance ↓ | route_audit 各时点 |
| 4. Accuracy | **Acc_E ≥ Acc_A2**（最终最重要；不能再"指标变漂亮、准确率轻微下降"） | 与 §20 同协议 run 对照 |

**c0 重点观察**：若 E1 后 c0 仍显著低于其余客户端（如 4/86/88/90/84 形态），则 pull+WD 次级通路升级为独立问题，开第二层诊断；若 85/88/90/87/89 形态则 Track A 修复闭合。

**E1 实测结果（2026-10-28，cifar_route_repair_E1_selectedce/seed_42）**：Gate 1 FAIL（c0 4.85 全塌；c1-c4 = 54.84/65.00/56.70/52.21，从 A 的 3.5-6.6 半修复但低于 70 门槛与 B 臂 81-91）；Gate 2 FAIL（`key_frac_norm_lt_1e-3` 0.97-0.98 全体，与 B 臂 0.0067 同量级——E1 按设计保留原 wd，预期之中）；Gate 3 c0 FAIL（跨客户端 top-10 子空间余弦 0.917，暂不解释）；Gate 4 PASS（92.70 vs A 92.15 vs B 92.38，c0=95.65 全场最高）。**两个新发现**：① wd 萎缩与方向劣化疑似同机制——key 范数被磨小后，CE 梯度经 `l2_normalize` 的 Jacobian（∝1/|K|）被放大，小更新大幅扭转已萎缩 key 的方向（E1 c1-c4 秩 52-65 < B 臂 81-91 的解释）；② c0 对一切 CE 侧干预免疫（A 6.59 / B 3.92 / E1 4.85 三臂全塌）且准确率解耦，触发预注册的 pull+WD 第二层诊断条件。

**E2 放行（2026-10-28 第三轮终审）**：E2 = E1 + `prompt_key` no-weight-decay，严格单变量（R1 init、τ=0.1、selected-slot CE backward、pull、prompt/value wd、optimizer/LR/epochs/data/seed 全不变），仅 `prompt_key` 的 weight decay 1e-3→0（optimizer 双参数组）。fail-closed：必须同时开 `--route_fix` 与 `--route_selected_ce_grad`。判读重点：E2 同时检验 Gate 1 与 Gate 2 是否同一 wd 机制（若 E2 后秩回升至 >70 且 near-zero 两档≈0，则一次修复两个 Gate）；**c0 是 E2 的第一刀诊断**——c0 复活 → WD 是 E1 条件下的主要次级通路；c0 仍塌 → 预注册 E3 = E2 + disable_pull（此时才干净归因 pull，勿提前）。正确结论表述（审查者措辞）：E1 证明了 selected-slot CE backward 消除了 dense CE coupling 的大部分几何塌缩（3-6 → 52-65），但保留原 WD 时产生 dormant-key norm collapse，单独不足以形成最终修复——不写"selected-slot CE 没用"。

```bash
# 20.5b 单测（E1 梯度拓扑 + E2 参数组不变量/fail-closed/零梯度 wd 判别 + route_audit E2 证据字段 + 前向逐位一致 + 默认 off 不变）
python -m unittest tests.test_route_ablation_switches -v
```

### 20.5c E2 修复臂：E1 + prompt_key no-weight-decay（2026-10-28 第三轮终审放行）

```bash
# 20.5c E2 臂（--max_rounds 1 短程，与 A/B/E1 同协议对照；仅在 E1 命令上单变量追加 --route_key_no_wd）
python main.py datasets_delay \
    --data_name cifar100 --method fedta \
    --route_fix --route_tau 0.1 \
    --route_selected_ce_grad \
    --route_key_no_wd \
    --route_audit \
    --seed 42 --global_epoch 5 --threshold 0.25 \
    --run_name cifar_route_repair_E2_keynowd \
    --max_rounds 1 \
    --data-path /root/autodl-tmp/users/ych/FedSMTA/local_datasets
```

**判读（四 Gate 全部逐 client，Gate 2 为 E2 核心因果指标；Gate 4 须 corrected 口径）**：

| Gate | E2 判据 | 说明 |
|---|---|---|
| 1. Key geometry | 秩 >70，最好接近 D 官方 ≈88（不要求 99） | E1 c1-c4 只有 52-65 |
| 2. Key norm | `key_frac_norm_lt_1e-3` **≈0**、`key_frac_norm_lt_1e-2` 显著下降；看 median/p10/p25（route_audit 已加，mean 易被少数大 key 掩盖） | E1 为 0.97-0.98 |
| 3. Routing utilization | 熵↑、top-1 dominance↓，不退回官方单槽 | 同 E1 |
| 4. Accuracy | Acc_E2 ≥ Acc_A2，且用 Track E corrected 口径判（短程 native 数字仅作同协议臂间参照） | 审查者要求 |

**新增机制证据（审查者要求的 WD 因果记录）**：route_audit post_local 行新增 `key_norm_ratio_*`（|K_post|/|K_init| 的 mean/median/p10/p25/min/max）与 `n_selected_slots_{ce,pull,union}` + `key_norm_ratio_{selected,unselected}_mean`（按"本轮被 CE selected-slot backward 和/或 pull 触碰过"分组）。判读形态：若 E1 未选中 key ratio≈0、E2 未选中 key ratio≈1，即 wd 因果链的漂亮证据。若 E2 后 c0 复活（如 82/88/90/87/86 形态）→ WD 是主要次级通路；c0 仍 3-5 → 跑预注册 E3 = E2 + `--disable_pull`。0.917 子空间趋同：等 E2 去除 near-zero key 后再解释，暂不立项。

## 21. Track B/E 探针命令（SSRA headroom / H1-H2 / 评测 Dropout 因果与重评，2026-10-28）

> **Track E 定性（审查者裁决）**：官方 `Client_DF.evaluate()` 全程不调用 `.eval()`，Chead（`Linear(1536→512) → ReLU → Dropout(0.5) → 200`）的 Dropout 在推理时保持激活 → 评测是随机采样且系统性低估（T0 实测：matrix 82.87 → eval-mode 84.81，+1.94pp）。仓库实现层面定性为 **evaluation-mode implementation bug**；论文算法与代码实现是两个层次，官方报告值保留为 official-compatible，不覆盖。H1/H2 已结案：native head 无欠训练（重训仅 +0.35pp）、Tail_Anchor 在 T0 判别贡献 ≈ +0.15pp；82.87→85.00 的差距主体是评测协议效应而非 head 优化余量。

### 21.1 Track E 因果三态探针（`probe_eval_dropout_causal.py`，先行闭环）

同一个 checkpoint、同一个 test loader 上三种模式：native（逐字官方 evaluate）/ head_eval（仅 Tail_Anchor 含 Chead `.eval()`，Tail_Anchor 无 BN、Chead 外无 Dropout → 精确隔离 Dropout 因果）/ full_eval（model+vit+original_model 全 `.eval()` = **Dropout 关闭的 corrected 评测，但保留官方 shuffle 语义**；vit `train=True` 前向语义逐字保留）。native 重复 15 次展示随机波动，eval 模式重复 3 次验证稳定性。

**第三轮裁决补充（2026-10-28，shuffle × batchwise_prompt 混淆）**：官方 `evaluate()` 每次调用都重建 `DataLoader(test_data, batch_size=8, shuffle=True)`，且 `Global_Prompt` 的 batchwise_prompt 按 batch 多数投票选 prompt 并扩展给整个 batch——预测不仅依赖样本本身，还依赖**它这次随机和哪些样本同批**。因此即使 Dropout 全关，重复 full_eval 理论上仍可能不同，"deterministic corrected evaluation" 不能提前成立，须由本探针实测。判读标准：`eval_repeat_spread_max` ≤0.05pp → shuffle 混淆可忽略，Track E 结案为"Dropout 是主要直接原因"，21.2 直接放行；>0.1pp → corrected 仍有实质波动，须先做 paired-RNG 版本（同一 RNG state 分别喂给 native/head_eval/full_eval，batch grouping 相同、唯一差异是 module mode）或 corrected 多重复取均值，再全量重评。JSON 已带 `corrected_repeat_stability`（negligible/minor/material）与 `reeval_gate_21_2` 字段。

**复审修复（2026-10-28 第二轮，阻断一）**：官方 `evaluate()` 内部 `load_head()` 会 `deepcopy(heads[task])` 重新替换 `self.head`，deepcopy 继承 `heads[task]` 自己的 training flag——只切 `client.model.eval()` 会被绕过，head_eval/full_eval 未必真的关闭最终参与推理的 Dropout。现 `eval_in_mode` 在 head_eval/full_eval 下同时把源头 `client.heads[task]` 切到 `.eval()`（finally 中恢复其原始标志）。运行时证据经 forward_pre_hook 在**前向时刻**采集（调用后读取不可靠：finally 恢复 `model.train(True)` 是递归的，会把已装进去的 head 翻回 train），写入 JSON 字段 `loaded_head_training_evidence`。`CAUSAL_CONFIRMED` 判据新增两条：native 前向时 head 标志全 True、head_eval/full_eval 前向时全 False。

```bash
python diagnostics/probe_eval_dropout_causal.py \
    --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
    --checkpoint checkpoints/task_00_end.pth \
    --task 0 --native_repeats 15 --eval_repeats 3
```

预期：native ≈ 82.87 且重复间有波动；head_eval ≈ full_eval ≈ 84.81 且重复几乎一致。`CAUSAL_CONFIRMED` 判据：dropout 效应 ≥1pp、|head_eval−full_eval|≤0.5pp、native 有随机性、eval 重复稳定。产物 `diagnostics_output/eval_dropout_causal.{json,csv}`。

### 21.2 Track E 全量 corrected-eval 只读重评（`eval_mode_reeval.py`，**条件放行**：21.1 结案后跑）

对每个 `task_XX_end.pth`、每个 (client, 已见任务 k) 输出双口径 `official_compat_acc` / `corrected_eval_acc`，外加 R[t][k] 矩阵与 AA。**只读**：不覆盖 run_dir 内任何历史产物，输出仅落 `diagnostics_output/eval_mode_reeval.{csv,json}`。不预设"所有历史结果都涨 1-2pp"——逐 checkpoint 实测。corrected 口径复用 21.1 的 load_head 绕过修复，并做 fail-closed 校验：任一 checkpoint 的 corrected 评测若经 forward_pre_hook 在前向时刻检测到 head 仍处 train mode，打 WARNING 并计入 JSON 字段 `corrected_head_train_mode_failures`（判读时应为全 0，非 0 即修复未生效、该批 corrected 数字不可信）。

**第三轮条件（2026-10-28）**：每个 corrected cell 目前只评一次。若 21.1 显示 corrected repeat spread 可忽略（`eval_repeat_spread_max` ≤0.05pp）→ 本命令直接放行，现有单次口径即可；若 corrected 仍有实质波动（>0.1pp）→ 单次 corrected 数字依赖某次随机 batch grouping，须先升级：**首选**官方兼容口径（保留 `shuffle=True`，corrected 每 cell 重复多次取均值，不改 batchwise routing 协议）；可另设 `shuffle=False` 固定 batch 口径作为第二严格确定性参照。升级完成并复审后再跑本命令。

```bash
python diagnostics/eval_mode_reeval.py \
    --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
    --native_repeats 1
```

### 21.3 B2 SSRA 结构搜索（`probe_ssra_headroom.py --mode b2`，val 选结构、test 只评一次）——**第三轮裁决：正式放行，可直接执行**

B2 使用缓存好的 `z_op` 特征，不再经过 `Global_Prompt` 的 batchwise routing，因此不受第三轮发现的 shuffle × batchwise_prompt 混淆影响。

网格：bottleneck {64,128,256} × residual scale {fixed, learnable α} × dropout {0, 0.2} × wd {1e-3}（12 配置）。**协议（审查者裁决）**：所有结构只比 3-seed validation（选 epoch 的 val acc），胜者 freeze 后 TEST 只评一次；P0 线性参照臂照旧 val+test 一次。任何搜索结构不产生 test 数字。Gate 仍是 GREEN ≥2pp / YELLOW 1-2pp / RED <1pp（作用于 winner−P0 的 test-once delta）。约 4× v2 探针计算量（195 次训练 × 150 epochs，cached 特征上，GPU 需求很小）。

**复审修复（2026-10-28 第二轮，阻断二/三）**：① 搜索运行真正 test-untouched（fail-closed）：b2 模式下 `train_one` 以 `X_te=None, y_te=None` 调用，`evaluate_test=False` 时 n_classes 只从 y_tr/y_val 推断、完全不物化 test 张量；`evaluate_test=True` 时传 None 直接 ValueError。② val/train/test 统一 eval-mode 口径（`_acc_eval_mode`：评估前保存 training 标志并临时 `.eval()`，评估后恢复），消除"dropout>0 的候选用带随机 Dropout 的 val 排名、胜者却用关 Dropout 的 test 报结果"的协议不一致；dropout=0 配置数值不变，v2/gate 旧结果不受影响。

```bash
python diagnostics/probe_ssra_headroom.py \
    --features diagnostics_output/ncm_inr_a2_seed42/features_ImageNet-R_task_00_end_promptcurrent.npz \
    --mode b2 \
    --bottlenecks 64 128 256 --scale_modes fixed learnable \
    --dropouts 0.0 0.2 --wds 1e-3
```

产物 `diagnostics_output/ssra_b2_search.{json,csv}`。**模式语义（第二轮裁决同步）**：`--mode gate`（默认）已改为现行裁决——P0=85.00 是强线性参照，verdict 直接按 winner−P0 的 test-once delta 三档判定，native 对齐仅作信息项（不再 CALIBRATION_FAIL）；v2 逐位旧行为（含 CALIBRATION_FAIL）保留在 `--mode legacy_gate`，仅供历史复现，不再代表当前裁决。

### 21.4 单测（Track E 切换语义 + B2 无 test 泄漏）

```bash
python -m unittest tests.test_track_e_and_b2 -v
```

覆盖：ResidualAdapter 零初始化恒等（fixed/learnable）、α=1.0 可学习、dropout 模块按需挂载、eval_in_mode 三态标志与恢复、native 随机 vs eval 确定、`evaluate_test=False` 不产生任何 test 字段、冻结 best_state 经 `test_from_state` 重放确定。第二轮修复新增回归：`test_head_eval_survives_load_head_bypass`（FakeClient 复刻官方 `load_head` deepcopy 绕过路径：head_eval/full_eval 前向时刻 head 标志必须为 False 且结果确定，源头 `heads[task]` 与各模块 training 标志必须恢复）、`test_search_run_never_touches_test`（搜索传 None 正常训练）、`test_evaluate_test_true_with_none_raises`（fail-closed）、`test_acc_eval_mode_deterministic_and_restores`（eval 口径确定 + 标志恢复）、`test_frozen_state_replay_is_deterministic`。当前 17 项全 PASS。

### 21.5 终审结案（2026-10-28，审查者裁决）

**21.1 实测（Track E = CLOSED / confirmed evaluation bug）**：`CAUSAL_CONFIRMED` 全判据通过，且 head_eval = full_eval = 84.8066 逐位相同、`eval_repeat_spread_max = 0`——第三轮担心的 shuffle × batchwise_prompt 混淆在实测中不存在；额外把 vit/original_model 切 eval 不产生任何可测变化。**核心发现是"82.xx 本身不是稳定性能值"**：native 15 次重复 spread 达 7.17pp，历史 82.87 只是随机 Dropout 评测分布中的一次 realization，不应继续当作模型真实确定性性能。**措辞规范（论文用）**：写"关闭 Chead Dropout 后，确定性准确率为 84.81%；相较 15 次原评测重复的均值 82.12%，提高约 2.69pp"；不写"A2 准确率固定提高 2.69pp"（+1.94pp 对应的是历史那一次 realization 的差值）。

**21.2 = DIRECT PASS**：corrected spread 可忽略，现有单次口径直接跑，无需 paired-RNG/多重复升级。全矩阵出来后重算 T0-T4 / after-T4 AA / forgetting / BWT / A2 与其他方法的真实差距——不再假设所有任务都是 +1.94 或 +2.69pp，逐 checkpoint 实测。

**21.3 实测（Track B / minimal SSRA = RED / stop engineering）**：winner `ssra_b64_fixed_d0`（val 88.87 排第一）test-once 85.72，P0 test 85.00（与强线性参照逐位吻合）→ **+0.72pp**（逐 seed +0.96/+0.76/+0.43），未过 ≥1pp 门槛。结构趋势 b64>b128>b256、fixed≥learnable、d0>d0.2：容量增加不产生 task-in 判别余量。**停止一切联邦工程**：Adapter FedAvg / task snapshot / checkpoint-resume / prototype loss 接入 / SSRA audit 全部不做。措辞规范：不写"B2 推翻了 v2 的 +1.28pp"（v2 b128 与 B2 winner 非同一路径，B2 也故意没有逐候选读 test）；写"早期单配置探针提示 +1.28pp，但在更严格的预注册 val-only 结构选择协议下，入选 SSRA 配置在冻结 test 上仅 +0.72pp，未过 ≥1pp 续作门槛"。

**状态板（终审）**：

| 路线 | 状态 |
|---|---|
| Track E | **CLOSED / confirmed evaluation bug** |
| Track B 极简 SSRA | **RED / stop engineering** |
| Track A（prompt routing → h 形成过程） | **ACTIVE / 当前主要机制研发线**（B/C/D 归因待跑，§20.4） |
| A5 | **重新设计**：仅否 A5-P2-v1（fixed feature + post-feature adapter）形态，共享语义学习整体未判负；待 21.2 + B/C/D 完成后定形态 |

**下一步（顺序）**：① 跑 21.2（上面命令）；② 跑 §20.4 B/C/D 三臂（Track A 归因，最直接的上游 representation 机制实验）；③ 两件事完成后重新设计 A5，先不开新模块。判负不是坏结果：H1（+0.35pp）+ H2（+0.15pp）+ B2（+0.72pp）闭环证明 T0 余量主体是评测协议而非结构，一个廉价离线 Gate 提前砍掉了整条低收益联邦工程线。
