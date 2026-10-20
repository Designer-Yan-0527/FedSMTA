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
