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
| `--bsab_pair_capture` | off | **§34.2 0b pair capture 开关**（instrumentation-only）：每个 broadcast boundary 缓存 pre/post 广播配对特征行（当前任务全行 train+test、z_ref/z_op 双侧、pre/post prompt hash、锚定 birth head state+hash、row_index_hash），输出 `<run_dir>/bsab_capture/`。默认 off = 逐位一致。仅限 M2'-matched capture rerun（与 `--refit_timing post_broadcast` fail-closed 互斥：`BSAB_CAPTURE_UNSUPPORTED_REFIT_TIMING`）；不加 BSAB module、不改任何训练/广播行为，BSAB 结构实现仍 FROZEN |

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
    --out_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/class_retention_4combo.csv
```

P 通道 = oo−co（聚合 prompt 漂移）、KA 通道 = oo−oc（本地 key/anchor 漂移）；public/private 由 manifest 自证（≥2 客户端持有 = public）；固定 shuffle 种子 `9700+100*cid+old_task`；样本加权。

### 9.3.2 子机制判别探针（probe_channel_correlates.py）

```bash
python diagnostics/probe_channel_correlates.py --csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/class_retention_4combo.csv
```

消费 9.3.1 的 CSV，检验间接共适应 vs 协议结构性脆弱两候选子机制（判决：后者否证、前者修正版成立）。

### 9.3.3a 路由结构识别（verify_slot_binding.py，D0）

```bash
python diagnostics/verify_slot_binding.py \
    --run_dir <run_dir> \
    --task_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth,checkpoints/task_04_end.pth \
    --out_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/slot_binding.csv \
    --out_slot_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/slot_binding_slots.csv \
    --out_npz output/cifar100/fedta/cifar_base_p15_legacy/seed_42/slot_binding_matrices.npz
```

绑定集中度 / 槽位共享 / 路由 2×2 分解 / 完整 class×slot 绑定矩阵。固定种子物化一份增强视图隔离增强噪声；无自动判决。

### 9.3.3b D1 v2 碰撞暴露诊断（collision_exposure_diagnostic.py，零前向）

```bash
python diagnostics/collision_exposure_diagnostic.py \
    --run_dir <run_dir> \
    --task_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth,checkpoints/task_04_end.pth \
    --binding_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/slot_binding.csv \
    --binding_npz output/cifar100/fedta/cifar_base_p15_legacy/seed_42/slot_binding_matrices.npz \
    --combo_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/class_retention_4combo.csv \
    --out_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/collision_exposure.csv \
    --out_json output/cifar100/fedta/cifar_base_p15_legacy/seed_42/collision_exposure.json
```

E_future → d_key/d_anchor（含方向/范数分解）→ KA 链条检验；纯张量运算约 1 分钟。判决须过 (client,task) FE（Roadmap §3.7）。

### 9.3.3c 落点分析（analyze_landing_spots.py，本地离线）

```bash
python diagnostics/analyze_landing_spots.py \
    --binding_npz output/cifar100/fedta/cifar_base_p15_legacy/seed_42/slot_binding_matrices.npz \
    --damage_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/collision_exposure_local.csv \
    --out_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/landing_spots.csv \
    --out_json output/cifar100/fedta/cifar_base_p15_legacy/seed_42/landing_spots.json
```

集中坍缩/attractor 假说检验；秒级本地运行。

### 9.3.4 TKR 精确口径复算（verify_tkr.py，零 GPU）

```bash
python diagnostics/verify_tkr.py \
    --run_dir <run_dir> \
    --out_prefix diagnostics_output/tkr/tkr_seed42
# 或直接指定 train_log：
python diagnostics/verify_tkr.py --train_log <path>/train_log.csv --global_epoch 5 --out_prefix diagnostics_output/tkr/tkr_seed42
```

按作者 issue 回复定义复算 TKR（分母 = round4 聚合全局模型、分子 = t>0 本地模型）。三种子复算结案：平均 96.4/98.4/96.1/89.6% 吻合论文 "around 98%"（Roadmap §3.8）。

### 9.3.5 P1 frozen NCM probe（ncm_probe.py，A5 Gate A）

```bash
python diagnostics/ncm_probe.py \
    --run_dir output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42 \
    --checkpoints task_00_end.pth,task_04_end.pth \
    --out_csv diagnostics_output/ncm_probe/ncm_probe_inr_a2.csv \
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
    --features_dir diagnostics_output/ncm_probe/ncm_inr_a2_seed42 \
    --out_csv diagnostics_output/oracle_headroom/oracle_headroom_inr_a2.csv \
    --c2p 86.86,91.14

# CIFAR-100 cifar_hsn 臂（唯一有完整 checkpoint + 标准 metrics 的全协议 cifar run；
# 需先提特征，GPU 几分钟；探针本身 CPU）
python diagnostics/probe_oracle_headroom.py \
    --run_dir output/cifar100/fedta/cifar_hsn/seed_42 \
    --checkpoints task_00_end.pth,task_04_end.pth \
    --out_csv diagnostics_output/oracle_headroom/oracle_headroom_cifar_hsn.csv
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

### 10.1 重新封板（2026-11-06，izfree）

**重跑 PASS（同日第三次，本次有效）**：§34.2 改动推送、izfree `git pull` 后重跑 Gate 0A：**PASS**。至此本节可正式宣称 **Sec.34 状态封板**——证据链三环闭合：

```
官方 FedTA ≡[Gate 0A，izfree，Sec.34 状态]≡ FedSMTA 当前代码（开关全关）
FedSMTA baseline ≡[verify_trajectory_identity.py，本机环2 PASS 25/25，§34.2.6b]≡ capture run（表征轨迹）
capture run 读数 ≡[repro gate arm1 MC 带宽 PASS，§34.2.7]≡ 官方 evaluate 语义
```

**勘误（同日，第一次测的历史记录）**：第一次 PASS 实际验证的是 Sec.33 状态——§34.2 改动当时尚未推送，izfree 代码不含 capture 插桩（`--bsab_pair_capture` unrecognized 报错暴露）。Sec.33 状态 PASS 仍有价值（CTRP 时代代码与官方一致），但不覆盖 §34。此段为历史勘误记录，最终结论以上方重跑 PASS 为准。

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
    --out_csv output/cifar100/fedta/cifar_base_p15_legacy/seed_42/a1_snapshot_eval.csv
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

预期：native ≈ 82.87 且重复间有波动；head_eval ≈ full_eval ≈ 84.81 且重复几乎一致。`CAUSAL_CONFIRMED` 判据：dropout 效应 ≥1pp、|head_eval−full_eval|≤0.5pp、native 有随机性、eval 重复稳定。产物 `diagnostics_output/misc/eval_dropout_causal.{json,csv}`。

### 21.2 Track E 全量 corrected-eval 只读重评（`eval_mode_reeval.py`，**条件放行**：21.1 结案后跑）

对每个 `task_XX_end.pth`、每个 (client, 已见任务 k) 输出双口径 `official_compat_acc` / `corrected_eval_acc`，外加 R[t][k] 矩阵与 AA。**只读**：不覆盖 run_dir 内任何历史产物，输出仅落 `diagnostics_output/eval_mode_reeval/eval_mode_reeval.{csv,json}`。不预设"所有历史结果都涨 1-2pp"——逐 checkpoint 实测。corrected 口径复用 21.1 的 load_head 绕过修复，并做 fail-closed 校验：任一 checkpoint 的 corrected 评测若经 forward_pre_hook 在前向时刻检测到 head 仍处 train mode，打 WARNING 并计入 JSON 字段 `corrected_head_train_mode_failures`（判读时应为全 0，非 0 即修复未生效、该批 corrected 数字不可信）。

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
    --features diagnostics_output/ncm_probe/ncm_inr_a2_seed42/features_ImageNet-R_task_00_end_promptcurrent.npz \
    --mode b2 \
    --bottlenecks 64 128 256 --scale_modes fixed learnable \
    --dropouts 0.0 0.2 --wds 1e-3
```

产物 `diagnostics_output/ssra/ssra_b2_search.{json,csv}`。**模式语义（第二轮裁决同步）**：`--mode gate`（默认）已改为现行裁决——P0=85.00 是强线性参照，verdict 直接按 winner−P0 的 test-once delta 三档判定，native 对齐仅作信息项（不再 CALIBRATION_FAIL）；v2 逐位旧行为（含 CALIBRATION_FAIL）保留在 `--mode legacy_gate`，仅供历史复现，不再代表当前裁决。

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

## 22. Track A 终局 + A5 Gate A-2 双数据集判读 + 送审包（2026-10-28，待终审）

> 本节为送审正文，若终审通过则取代 §21.5 状态板中 Track A / A5 两行。

### 22.1 E2 四 Gate 判读（§20.5c 预注册）与 kill criterion 触发

E2 = E1 + `--route_key_no_wd`（`cifar_route_repair_E2_keynowd`，seed 42，短程协议）。判读源：`route_audit/route_audit.jsonl` + `logs/train.log`。

| Gate | 指标 | E1 | E2 | 判定 |
|---|---|---|---|---|
| 1 秩 | 有效秩逐 client（init 99） | 54.84/65.00/56.70/52.21/4.85 | **96.47/96.14/97.47/97.68/95.20**，post_fusion 96.47 | **PASS**（>70） |
| 2 范数（核心因果） | frac<1e-3 | 0.97-0.98 | **0.0**（全部 client + fusion 后） | **PASS** |
| 2 范数分组 | 未选中 key ratio / 选中 key ratio | 萎缩 | **1.0 逐位不动 / 1.03-1.06** | wd=0 下未触碰 key 完全保持 |
| 附 | mean_abs_offdiag_cosine | — | 0.0057（近正交保持） | — |
| 4 准确率 | train.log 末轮均值 | 92.69 | **≈92.5**（两列口径 92.61/92.48） | **FAIL**（≤ D 94.50） |

四臂对照（同一短程协议）：**D 官方 94.50 > E1 92.69 ≈ E2 92.5 > B 92.38 > A 92.15**。

**Kill criterion 触发**（§20.5c 预注册）："E2 几何全绿但准确率仍 ≤ D → Track A 修复线关闭，基线回到官方 D"。几何全绿 ✓ + 准确率 ≤ D ✓ → **触发**。

**因果结论（Track A 的最终科学产出）**：
1. **wd 是 key 范数饥饿的主驱动——证实**：免 wd 后未选中 key 逐位不动（Adam 零梯度 + wd=0 不移动）、零饥饿；E1 的"wd 经 l2_normalize Jacobian（∝1/|K|）放大 CE 梯度扭转"假说被 E2 直接证实。
2. **c0 之谜解开**：c0 在 A/B/E1 三臂全塌且对 CE 侧干预免疫，E2 下复活（96.47）→ c0 塌缩主因是 wd 而非 CE；按第六轮预注册"c0 恢复 → WD 是主次级通路"，**E3（E2+disable_pull）不再需要**。
3. **塌缩与当前任务准确率双向解耦**：E1 塌着 92.69 / E2 修干净 ≈92.5——修复 key 池一分准确率不还。"修好 key 池准确率回来"的立项假设被否证；route_fix 家族 −1.9pp 的成本来自软路由 prompt 混合与 R1 init 本身，与 key 健康无关。
4. **关闭措辞（论文用）**：不写"机制修复无效"；写"key 池健康与当前任务准确率无因果通路（双向解耦实测）；机制修复的假想收益在多轮保持维度，未测且不构成当前优先级"。软路由复活须先在 A2 INR 主线拿出多轮 retention 证据（维持原判据）。

### 22.2 A5 Gate A-2：oracle headroom 双数据集判读

工具 `diagnostics/probe_oracle_headroom.py`（§9.3.6，Gate 预注册：只看 T0 z_op `delta_pooled`，≥2 GREEN / ≤1 RED）。探针运行中修复三件事：`diag_utils` 老 run args 回填改 parser 默认值自动派生（2f5f1c2）；`Server_DF.load_checkpoint` global_head 按保存结构重建（75ead14，skip_fed_avg_head 臂 resume 真 bug）；mask 类零训练行保留被压低的输出单元（24f9810，CIFAR 公共类 (client,task) 份额可为 0-1 样本的协议现实，native Chead 同语义）。

| 数据集 | run | T0 z_op native / mlp_local / mlp_pooled | d_loc | d_pool | fed | Gate |
|---|---|---|---|---|---|---|
| INR（主判读） | inr_a2_key_p40 | 82.87 / 86.91 / 85.75 | +4.04 | **+2.88** | **−1.16** | **GREEN** |
| INR z_ref | 同上 | 82.87 / 84.01 / 83.09 | +1.14 | +0.22 | −0.92 | RED（上下文） |
| CIFAR | cifar_hsn | 94.67 / 96.93 / 96.13 | +2.26 | **+1.46** | **−0.80** | **YELLOW** |

INR 参考：mlp_local 86.91 ≈ C²P T0 86.86（T0 差距被本地重训接口完全吃掉）。CIFAR native 94.67 与 D 官方 94.50 吻合（hsn 臂 z_ref 读数有效性 sanity）。

**判读**：
1. **headroom 双重确认**（INR GREEN / CIFAR YELLOW）：可训练接口在 OUR 特征上可多榨 +1.5~4pp，且信息在 z_op（prompt 增强特征）不在 z_ref——prompt 有效，接口是瓶颈。
2. **"联邦共享"组件被否证（决定性）**：CIFAR 有 25 个跨客户端公共类，池化 oracle 仍输给本地头（96.13 < 96.93）；双数据集、有无类重叠两形态 `fed` 全负。**A5-v2 的"共享 g + FedAvg"设计无证据支持。**
3. **native 接口训练浪费**：连线性头干净重训都 +2.21（CIFAR）/ +2.90（INR）——Chead 的训练方式（Dropout 0.5 / 与 prompt 联训 / 优化配置）把 2-3pp 留在桌上。
4. **T4 上下文（不进 Gate，回溯重访口径）**：CIFAR native AA 74.65 vs 本地重训头 96.53（**+20pp**）；INR 84.82 vs 79.43（+5.4）。遗忘的主体是**接口漂移**（旧任务头与漂移后 prompt 特征失配），不是信息丢失——"本地数据巩固/接口重对齐"组件的 headroom 证据（协议合法，数据不出域）。

### 22.3 A5-v2 方向修正（待审）

原设计"轻量共享语义投影 g + FedAvg"按 22.2-2 撤回。修正设计要点：
- **基座**：官方 FedTA 全部不动（hard routing、official prompt 训练、corrected 口径 Gate 4）。
- **方法主体**：本地语义接口的解耦/巩固训练——利用本地已见任务数据（不出域）周期性重对齐接口，吸收 22.2-3 的 T0 headroom 与 22.2-4 的保持 headroom。
- **机制归因消融**（预注册）：Chead Dropout 0.5 / 联训 vs 后置 / 输入标准化 三因素拆解 native 接口为何弱。
- **Gate（预注册）**：当前任务 corrected 准确率 ≥ 官方 D+1pp = GREEN；约束 BWT_corrected ≥ −2pp。
- 联邦聚合作为增强臂保留（fed<0 不立项、只作消融），不再作为方法主张。

### 22.4 送审问题（请审查者裁决）

1. **Track A 关闭**：22.1 kill criterion 触发是否成立？E3 按预注册（c0 复活 → WD 归因）判"不需要"是否同意？
2. **A5-v2 方向修正**：由"共享语义层 + FedAvg"转向"本地接口解耦训练"，依据 22.2-2/3/4（双数据集 fed<0、linear refit +2~3pp、T4 +20pp 漂移证据）——是否成立？
3. **A5-v2 立项**：INR GREEN / CIFAR YELLOW 的 Gate 判定与 22.3 设计要点是否放行？若无异议，下一步产出 A5-v2 正式 spec（方法节 + 消融矩阵 + 实验计划）。

## 23. A5-v2 正式 spec：解耦接口 refit + 本地巩固重对齐（2026-10-28，已开工）

> 用户 2026-10-28 授权不等终审直接开工；若审查意见返回后与本 spec 冲突，以用户指令为准（冲突点须显式指出）。立项依据 = §22.2（headroom 双重确认 / fed<0 / 接口漂移 +20pp）。

### 23.1 方法定义（实现即本节，代码为准）

**基座不变**：官方 FedTA（hard routing、official prompt 训练、frozen ViT）一概不动；phase-1/phase-2/BGPS/SIKF 联训全部逐位保留。

**组件一（--head_refit，因素 F2=解耦）**：每轮 phase-2 官方联训结束后，对本任务做 readout-only 后置 refit——

1. 特征缓存：当前接口状态下（prompts 已加载、模型原状态），train split 顺序过一遍（no_grad、shuffle=False、batch 16），取 Tail_Anchor 的 head 输入 `x1`（1536 维 anchor-concat——native 接口的真实输入；探针 z_op 是 768 维 pre-anchor，见 23.6 已知偏差）；
2. 重训一个 Chead 形状读出头（`1536→512→ReLU→Dropout(0.1)→C`，C 与官方 Chead 同维，masked CE 同 phase-2 口径，Adam lr=1e-3 / wd=1e-4 / 100 epoch / batch 64，确定性本地种子 sha256(seed,'refit',client,task,round)），梯度只到读出头；
3. 替换 `heads[task_id]` 存储（评测即生效）；**joint head 轨迹独立保存**（`_a5_joint_head`），下一轮 phase-2 前恢复 → 联训 head 轨迹与官方 run 逐位一致。

**因素 F1（--head_dropout，默认 0.5=官方）**：phase-2 Chead 的 train 端 Dropout p。与 Track E 已结案的 eval 端 bug 不同物：corrected 口径（§21.5）已修复 eval 端；F1 是 train 端训练动力学消融，p=0.0 改变 phase-2 前向 RNG 消耗（预期内，臂内轨迹自洽）。若消融显示 headroom 全来自 F1 → 组件一按预注册预案降级为"基线卫生"，不作方法主张。

**因素 F3（--head_standardize，默认 off）**：refit 输入用 train-split mean/std 标准化（mu/sd 作为 buffer 随头存取，StandardizedChead 模块，评测路径经 load_head 深拷贝自然兼容）。fail-closed：必须与 --head_refit 或 --interface_consolidate 之一联用。

**组件二（--interface_consolidate，巩固重对齐）**：每任务边界轮（该 task 最后一轮）train() 末尾，对本 client 全部旧任务 t'<t 用**当前**接口状态 refit `heads[t']`（同 refit 配方；数据 = 旧任务 train split，`data_split_indices[t']`，test split 永不触碰）。当前任务的 refit 专属 --head_refit（两开关正交，巩固只管旧任务）。吸收 §22.2-4 的接口漂移 headroom（CIFAR T4 +20pp 证据）。

**RNG 中立（强归因性质，单测覆盖）**：refit/consolidation 全程四流快照恢复（torch cpu/cuda/python/numpy）+ DataLoader shuffle=False——(a) 全部开关 OFF = 官方行为逐位一致；(b) 开关 ON 时**表征侧轨迹（prompt/key/anchor/joint head）仍与官方 run 逐位一致**，任何 corrected 指标变化 100% 归因于接口替换。这是论文归因的核心卖点。

**replay 访问级别定位（预注册声明）**：巩固 = replay 访问级别（每边界重访旧任务 train split），与 C²P 的 consolidation_train 公平（C²P 每轮本就重访 origin+target，prompt_hard.py L675-715）。三臂消融（readout-only vs 表征 replay vs stored-logits 蒸馏）中本 spec 只实现 readout-only；后两臂待本矩阵 Gate 后再议。venue 若要求 no-replay → 双记账口径预案。

### 23.2 消融矩阵（预注册）

| 臂 | flags | 检验内容 |
|---|---|---|
| M0 | 无（官方） | 基线 D |
| M1 | `--head_dropout 0.0` | F1：train 端 Dropout 归因 |
| M2 | `--head_refit` | F2：联训 vs 后置 refit |
| M3 | `--head_refit --head_standardize` | F2+F3：加标准化 |
| M4 | `--head_dropout 0.0 --head_refit --head_standardize` | F1+F2+F3 |
| M5 | M4 + `--interface_consolidate` | 全方法（巩固需全程 5 任务） |

严格单变量逐臂叠加；M1-M4 在 T0 短程（`--max_rounds 5`）可判（巩固在 T0 无旧任务、M5≡M4），M5 需全程。

**Gate（预注册，不变）**：当前任务 corrected 准确率 ≥ D+1pp = GREEN；约束 BWT_corrected ≥ −2pp。种子 42 先行，GREEN 后 3 种子。主判读场：迭代场 cifar100（D=94.50）；主战场 ImageNet-R（headroom 证据地，D 参考 §21.5 corrected 口径 84.81，注意"vs 均值/一次 realization"措辞纪律）。

**预注册预案**：M1 单独达到 GREEN → 组件一降级为"基线卫生"；M4 达标 M5 加巩固无增益 → 巩固组件只作旧任务保持约束手段（仍受 BWT ≥ −2pp 检验）。

### 23.3 命令

T0 短程归因（cifar，各臂，约 5 轮）：

```bash
# M0 官方基线（已有 D=94.50 可复用，无需重跑；如需对账用 §2 命令 + --max_rounds 5）

# M1 F1 train端dropout
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_dropout 0.0 --max_rounds 5 --run_name cifar_a5v2_M1_drop0

# M2 F2 后置refit
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --max_rounds 5 --run_name cifar_a5v2_M2_refit

# M3 F2+F3 加标准化
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --head_standardize --max_rounds 5 --run_name cifar_a5v2_M3_refit_std

# M4 F1+F2+F3
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_dropout 0.0 --head_refit --head_standardize --max_rounds 5 --run_name cifar_a5v2_M4_full_noconsol
```

全程臂（M5 全方法；M0-M4 全程版按需把 --max_rounds 去掉即可）：

```bash
# M5 全方法（巩固需全程 5 任务）
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_dropout 0.0 --head_refit --head_standardize --interface_consolidate \
    --run_name cifar_a5v2_M5_full
```

ImageNet-R 版：`--data_name ImageNet-R --private_class_num 40 --surrogate_num 5`，其余参数同 §3，run_name 前缀换 `inr_a5v2_`。

### 23.4 判读表

| 判读项 | 口径 | 判据 |
|---|---|---|
| T0 短程归因 | `metrics/accuracy_matrix.csv` after_task=0 行（corrected 单次口径，§21.5 DIRECT PASS） | M0→M1→M2→M3→M4 逐步增量拆账：F1 = M1−M0，F2 = M2−M0，F3 = M3−M2，联合残差 = M4−M3−（M1−M0）的近似交互项 |
| T0 headroom 对账 | 各臂 T0 AA vs 探针 mlp_local 96.93（CIFAR）/ 86.91（INR） | refit 臂若显著低于探针 → 查 23.6 已知偏差（输入差异 1536 vs 768、hidden 512 vs 256、augmentation） |
| 全程 Gate | 当前任务 corrected（T4 时点）≥ D+1pp | GREEN 立项方法主张 |
| 保持约束 | BWT_corrected ≥ −2pp | M5 vs M4 的旧任务列差 = 巩固组件净效应 |
| 轨迹不变性 sanity | M2 与 M0 的 key/prompt checkpoint（如需） | 开关 ON 不改变表征侧（RNG 中立设计保证，抽查即可） |

数据组织三级口径：分数据集、分种子、分任务（目录结构 §8 天然满足：`seed_{seed}/metrics/*`）。

### 23.5 实现清单与单测

实现（本次 commit）：
1. `config/datasets_delay.py`：`--head_dropout`（默认 0.5）/ `--head_refit` / `--head_standardize` / `--interface_consolidate` 四开关；
2. `Models/classification_head.py`：`Chead(label_num, dropout=0.5)` 参数化（默认=官方逐位一致）+ `StandardizedChead`（mu/sd buffer + Chead 形状 MLP）；
3. `Models/Tail_Anchor.py`：dropout 透传；
4. `Models/Client_DF.py`：refit 机制（特征缓存→readout 重训→heads 替换）、`_a5_joint_head` 轨迹保存/恢复（phase-2 前恢复）、巩固（边界轮旧任务全量 refit）、RNG 四流快照、checkpoint 存取（schema 仅 ON 臂新增 key，OFF 臂逐位不变）、`rebuild_client_head`（resume 时按 state keys 识别 Chead/StandardizedChead，外来 keys fail-closed）；
5. `Models/Server_DF.py`：F3 fail-closed 组合校验 + RESUME_MUST_MATCH 增补四键（two-sided，老 checkpoint 兼容）。

单测（`tests/test_a5v2_refit_switches.py`）：默认 OFF 逐位一致（Chead 构造 RNG 中性、legacy args 无属性容错）、refit 梯度隔离（key/anchor/prompt 无梯度且逐位不动，仅头变）、RNG 四流恢复、heads 存储替换语义 + joint 轨迹恢复、rebuild_client_head 三分支（Chead/StandardizedChead/外来 keys 拒绝）、fail-closed（standardize 单独用、resume 缺 a5 key）。

```bash
python -m unittest tests.test_a5v2_refit_switches -v
```

### 23.6 已知偏差与边界（如实报告）

1. **输入差异**：探针 mlp_local 的 +4.04/+2.26 是 768 维 pre-anchor z_op 上取得的；refit 用 native 接口真实输入 x1（1536 维 anchor-concat，含 top-1 硬路由选 anchor）。若 M2/M3 仍显著低于探针，此为第一嫌疑——**2026-10-29 已定案为 INR 上的实际病灶（判读见下），`--refit_input feat` 臂已落地**：训练输入 = z_op⊕0 垫零成 1536，训练后 first Linear 锚半区权重硬置零 → 部署语义 head(x1) 功能上 = z_op 头（x1[:, :768] 即 z_op），evaluate/checkpoint/resume 路径与 x1 臂完全同构（state keys 不变），无需另改 evaluate 路径。fail-closed：必须与 --head_refit 或 --interface_consolidate 联用。
2. **hidden 512 vs 探针 256**：为保持 heads 存储 = Chead schema（评测路径零改动）选 512；容量非三因素之一，若归因不干净再回头。
3. **augmentation**：refit 特征来自带增强的 train split（与官方 protos 计算同数据流）；探针 npz 特征的增强状态以 extract_features.py 实际为准。臂内自洽，臂间对照时注意。
4. **dropout(0.5) train 端是作者设计选择**：只作归因消融因素，不作方法主张（用户已敲定）。
5. resume 的 diag 工具兼容性：部分诊断（compatibility_drift 等）对 StandardizedChead keys 可能 fail-closed 拒绝——预期行为，判读用 corrected metrics 口径即可。

### 23.7 §23.6-1 修复臂（M2'）落地与判读定案（2026-10-29）

**INR 双探针诊断判读**（probe_refit_input_ablation.py，锚点列修复后，corrected 口径，client 均值，探针直接实测）：

| 臂 | r0 | r1 |
|---|---|---|
| native（本地联训头） | 77.50 | 77.45 |
| saved_refit（hook 落盘头） | 77.96 | 79.35 |
| x1_rf（x1 配方） | 78.67 | 79.09 |
| z_rf（z 配方） | 78.77 | 80.27 |
| x1_lt（x1_light 轻量配方） | 78.90 | 80.41 |
| z_lt（z_light 轻量配方） | 79.23 | 80.33 |
| x1_pb（x1_probe256 探针配方） | 79.27 | 79.39 |
| z_pb（z_probe256 探针配方） | 79.41 | 80.59 |

判读结论（预注册判据兑现）：
1. **嫌疑 2（x1 路径）r1 过阈**：Δ(z_rf−x1_rf)=+1.18 —— refit 头吃进的 1536 维 anchor-concat（含 top-1 硬路由）在 INR 200 类上是污染源；r0 不动（+0.10，随训练轮次激活）。
2. **嫌疑 1（配方过拟合）r1 过阈**：Δ(x1_lt−x1_rf)=+1.32 —— 但**换 z 输入后配方敏感性消失**（z_rf/z_lt/z_pb 极差 ≤0.32）→ 配方过拟合计入 x1 的病，单变量修输入（z 替换）即可同时消两嫌疑。
3. 预期涨点口径（探针直接实测，非 oracle 外推）：INR r0 +0.81 / r1 +0.92（z 臂 vs native/saved_refit 锚点）。

**Gate 参照修正（口径审计）**：§23.2 中 D 参考 84.81 实为 **inr_a2_key_p40（A2 臂）** 的 reeval，非官方基线；对照物必须是 inr_base_p40 的 corrected T0。**补测定案（2026-10-29，eval_mode_reeval task_00_end.pth）：base corrected T0 = 84.8066（official 81.666，eval-dropout 折扣 +3.14pp）**——与 A2 臂的 84.81 数值几乎重合，口径缺口的实际影响 ≈0，Gate 参照数字维持 84.81 但来源已站住（base 臂）。

**M2' 臂命令**：

```bash
# INR M2' 判据臂（feat 输入修复，T0 短程；划分复刻 inr_a5v2_M2_refit：
# task_num 10 / private 20 / surrogate 默认 20——探针 +0.81/+0.92 即在此划分实测）
python main.py datasets_delay --data_name ImageNet-R --data-path ./local_datasets \
    --client_num 5 --task_num 10 --private_class_num 20 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --max_rounds 5 --run_name inr_a5v2_M2_feat

# cifar M2' 对照臂（验证 feat 输入在 cifar 上不掉点）
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --max_rounds 5 --run_name cifar_a5v2_M2_feat
```

**修订判据（预注册）**：INR M2' 的 r0/r1 允许 ≈baseline 或略输（早期联邦同步优势 ~1pp 预期内：baseline 日志数字 = fed_avg 广播头 5×池化，refit 头 = 纯本地小样本头）；**round 4（T0 边界）必须 ≥ inr_base_p40 corrected T0 = 84.81**——反超不成则 INR 上 M2 方法不成立、主战场退守 cifar。cifar M2' 预计不掉点（x1 三成本项在 cifar ≈0），若掉点则 feat 输入的通用性存疑，需复查。

**划分混淆披露（2026-10-29 口径审计时发现，如实报告）**：inr_a5v2_M2_refit 与 inr_base_p40 的任务划分不同（M2：task_num 10 / private 20 / surrogate 20；base：task_num 5 / private 40 / surrogate 5）——探针判读（native 77.50 vs 广播头 ≥78.41 的联邦同步优势项）与"round 4 vs base corrected"对照均为跨划分比较，只作方向性参考。M2' 复刻 M2 划分（探针预测 +0.81/+0.92 的同划分验证），主判读 = M2' vs M2 同划分单变量差；若 M2' round 4 确认 feat 修复有效且需正式 Gate，再补 base 同划分（task_num 5 / private 40 / surrogate 5）的 M2' 臂对 base corrected。

**M2 vs baseline 差距三层分解**（用户"可是不如 baseline"的定案回答）：
1. 早期联邦同步优势（~1pp）：baseline 日志数字 = fed_avg 广播头（5×池化）；refit 头 = 纯本地 ~1k 行头；本地联训头 corrected 77.50 vs 广播头 ≥78.41（日志口径）。
2. x1 路径 + 配方过拟合（−1.2~−1.5）：两嫌疑已证（上文判读），feat 臂修复对象。
3. eval-dropout 口径：INR 上官方头（p=0.5）eval 被打 ~3pp 折扣（T0: 81.67→84.81），refit 头 p=0.1 受罚小 → 日志口径偏袒 M2，corrected 差距更大。

**oracle 外推参考**（diagnostics_output/oracle_headroom/oracle_headroom_inr_a2.summary.json，inr_a2_key_p40）：T0 z_op 分解 mlp_local=86.91 > mlp_pooled=85.75 > native corrected=84.81；federated_premise=−1.16（INR 上跨 client 池化负收益 → "refit 头做 FedAvg"路线被 oracle 判死，不做）。

## 24. 目录结构规范（2026-10-29 起，历史产物已按此归档）

### 24.1 output/（训练产物，run_manager 固定四层）

```
output/<dataset>/<method>/<run_name>/seed_<seed>/
├── args.json / command.txt / git_commit.txt
├── checkpoints/   round_XXXX.pth / task_XX_end.pth / latest.pth
├── metrics/       accuracy_matrix.csv / summary_metrics.csv / train_log.csv / ...
└── logs/          train.log
```

- `<method>` = main.py 方法目录名（fedta / fedta_a2 / fedavg / fedprox / c2prompt / fedmgp ...）；
- `run_name` 命名：`<数据集缩写>_<实验线>_<臂>[_<variant>]`，小写下划线；**划分敏感的臂必须带 `_pN` 后缀**标明 private_class_num（如 inr_base_p40 / inr_a2_key_p40），避免再出现 §23.7 披露的划分混淆；
- legacy 例外（历史 run，只读保留，不再新增）：
  - `output/cifar100/fedta/cifar_base_p15_legacy/seed_{42,1999,2025}/`——旧格式：csv 平铺在 run 根、**无 checkpoints/**（corrected 无法重评的根因）；
  - `output/regression*/`——单测产物，与实验数据无关，禁止混入实验判读；
- **正在跑的 run 目录禁止移动/改名**。

### 24.2 diagnostics_output/（诊断产物，按产出脚本归档）

```
diagnostics_output/
├── corrected_reeval/     # corrected 口径统一定案（每臂 *_corrected.csv/.json/.log）
├── eval_mode_reeval/     # eval_mode_reeval.py 输出（含 a5v2 各臂历史）
├── oracle_headroom/      # oracle 上限分解
├── head_retrain_probe/   # 头重训探针
├── refit_input_probe/    # refit 输入消融探针（INR 双探针判读数据）
├── ncm_probe/            # NCM 探针 + 特征子目录
├── ssra/  ├── tkr/  ├── features/  ├── misc/
```

- 规则：**一个诊断脚本一个家族目录**；文件名保留脚本默认名（不二次改名，保持 `--out_*` 兼容）；
- 新诊断脚本必须支持 `--out_csv/--out_json` 并默认写入自己的家族目录，不允许平铺到根；
- 判读引用路径一律以本节为准；本文档历史路径已于 2026-10-29 批量更新。

## 25. PSC：prompt 子空间巩固（2026-11-01 预注册，已开工）

> 结构级创新（用户定案：非修复级、须与 M2' 正交可叠加、不破坏时间/空间保留率）。用户 2026-11-01 批准执行顺序：改代码 → 短程判据臂（用户跑）→ t1 边界 oracle refit 探针（我读 checkpoint 跑）。

### 25.1 方法定义（实现即本节，代码为准）

**动机与对象**：prompt 池（`--size 100`，cls-key 硬路由 top-1）中，跨任务/跨客户端长期承载相似语义的行被反复训练拉扯——旧任务的判别方向在后续任务训练中被覆盖。PSC 把池行显式分为两类：**core 行**（跨任务继续吃训练信号，承载公共方向）与 **per-task residual 行**（冻结锁旧任务判别方向，承载差异方向）。

**轮流程**（基座逐位不动，全部挂在既有轮序上）：

1. server 广播池（官方不变）+ 广播 core 行集合（`_psc_core_slots`，幂等）；
2. 客户端 task 开始（`update_data`）时计算旧任务 emergent 支撑：全部旧任务 t'<t 的 train split 过 `original_model` 取 cls features → 在**当前** key 下 per-sample top-1 路由并集 = S_old；residual = S_old − core（A2 机器同构，行集合换成 prompt 接口行）；四流 RNG 快照后冻结 residual 行的 prompt_key + prompt 内容；
3. 本地 phase-1/phase-2 照常训练；phase-1 每次 `optimizer.step()` 后把 residual 行恢复快照值（复用 A2 post-step restore 机器）；phase-2 与 BGPS/SIKF 不动；
4. M2' refit 原样叠加（正交：M2' 动 heads[] 读出头，PSC 动 prompt 行 + 路由 key）；
5. 上传对象零新增（官方四件套），server 照常 kd_fusion；
6. **仅在任务边界融合轮**（`round_i % global_epoch == ge-2`，即每 task 最后一次融合；提交后广播恰为新 task 起点）执行 `consolidate_prompts()`。

**合并语义（"公共方向入 core、差异方向入 residual、只合并不删除"）**：对候选对 (i,j), i<j：row i := (i+j)/2（prompt_key 与 prompt 内容都取平均）并标记为 core 行；**row j 原值不动**——旧任务路由到 j 的流量零扰动，差异方向保留在 residual 行中；无行删除 → span 不变性。已合并的行对不重复合并；共享行的对跳过（防 drift 滚回间接撤销先前已提交合并）。

**测量协议（server 侧，RNG 中立）**：每次融合轮后，surrogate 数据上**全部已见类**并集（注意：非 kd_fusion 的当前类口径）做一次路由 pass（四流快照/恢复、shuffle=False、num_workers=0），得每行 top-1 邻域质心；行对相似度 = 质心余弦（O(P²)）。候选门：**连续 `--psc_streak` 个融合轮**相似度 ≥ `--psc_tau`（本轮无法测量的对——行已转 core 或邻域 <2——streak 必须重置，"连续"语义严格）；边界轮按 (sim, streak) 排序取 ≤ `--psc_max_merge` 对。

**drift 滚回门**：合并前后各测一次 per-sample top-1；变化中**非纯 i↔j 交换**（既抓"离开对"也抓"被偷流量"，对称定义）的比例 > `--psc_drift` → 恢复行 i、streak 清零、该对滚回不提交。

**惰性证据日志（A2 组合臂教训，判读必查）**：客户端 phase-1 每批次统计 residual 行 top-1 命中（样本数/去重行数），轮末打印 `[PSC] ... (laziness check)`——冻结行若从不被路由到 = 无效冻结，判读时以此为否决证据。

**fail-closed**：`--psc` 与 `--method fedta_a2` 互斥（A2 已判死，惰性保护教训）、与 `--route_fix` 互斥；`--psc_max_merge < 0` 拒绝启动；resume 时 ON 臂缺 `psc_state`（server 侧 core_slots/pair_streak/merged_pairs；client 侧 task_id/core_slots/residual_slots/key_snapshot/prompt_snapshot）双侧拒绝；`RESUME_MUST_MATCH` 增补 `psc/psc_tau/psc_streak/psc_max_merge/psc_drift` 五键（跨 ON/OFF 与跨参数 resume 均拒绝）。checkpoint schema 仅 ON 臂新增 `psc_state` key，frozen 基线逐位不变。

**默认参数（首次注册即默认）**：`--psc_tau 0.95` / `--psc_streak 3` / `--psc_max_merge 2` / `--psc_drift 0.05`。

### 25.2 短程判据臂（预注册）

双臂对照 + 探针天花板比较。PSC 机制需 ≥2 任务才有旧任务 residual（task-0 内同任务合并除外），判据臂 = t0+t1 两任务（`--max_rounds 10`）：

```bash
# cifar 臂 A：M2' + PSC
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --psc --max_rounds 10 \
    --run_name cifar_psc_M2p_psc_t1

# cifar 臂 B：M2' 对照（同协议去 --psc）
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --max_rounds 10 \
    --run_name cifar_psc_M2p_ctrl_t1

# cifar 安全臂（可选，先行）：T0 单任务，验证 PSC 在 T0 近 no-op 不掉点
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --psc --max_rounds 5 \
    --run_name cifar_psc_M2p_psc_t0

# INR 臂 A：M2' + PSC（划分复刻 p40 家族：inr_base_p40 / inr_a5v2_M2_feat_p40 /
# inr_hsn_p40——t10/p20 的 inr_a5v2_M2_feat 是已废弃的 crash run，禁用作对照）
python main.py datasets_delay --data_name ImageNet-R --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 40 --surrogate_num 5 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --psc --max_rounds 10 \
    --run_name inr_psc_M2p_p40_psc_t1
# INR 臂 B = 同上去掉 --psc，--run_name inr_psc_M2p_p40_ctrl_t1
```

全轨迹臂（GREEN 后再跑，2×2：PSC-only / PSC+M2'，同 §2/§3 模板去 `--max_rounds`）。

### 25.3 判读表（预注册）

| 判读项 | 口径 | 判据 |
|---|---|---|
| T0 安全 | 安全臂 T0 边界 corrected vs 已有 M2' T0（cifar_a5v2_M2_feat） | 差 ≥ −0.3pp（PSC 在 T0 至多同任务内合并，预期近 no-op；跌破则查合并日志是否 task-0 行被误并） |
| t0 段不掉（A vs B） | 两臂 round 4（T0 边界）corrected AA | 差 ≥ −0.3pp（A2 教训：掉点即 RED，机制有反作用） |
| 惰性证据 | A 臂 train.log `[PSC] ... laziness check` | task 1 各轮 residual 行命中样本数 > 0；恒 0 → 冻结无效，RED（不许以"无害"名义放行） |
| 探针天花板（主判读） | 两臂 t1 边界 checkpoint（round_0009）oracle refit 探针（§21 head_retrain 家族，corrected） | A 臂天花板 ≥ B 臂 + 0.3pp → GREEN（合并应释放 core 容量给新任务、residual 锁旧方向 → 新任务天花板更高且旧任务不掉） |
| t1 段直接效应 | 两臂 round 9 corrected AA | A − B 作参考量级（非 Gate；短程两任务效应 < 全轨迹） |

预注册预案：天花板 GREEN 但 t1 直接效应 ≈0 → 照跑全轨迹 2×2（短程不敏感属预期，机制红利在长任务流）；天花板 RED → PSC 判死，只保留 §25.5 披露。

### 25.4 实现清单与单测（本次 commit）

1. `config/datasets_delay.py`：`--psc` + `--psc_tau/--psc_streak/--psc_max_merge/--psc_drift` 五参数（各带 help）；
2. `Models/Client_DF.py`：task-start 冻结（`_psc_init_task_protection`：旧任务支撑集 + residual 行四流快照）、phase-1 post-step restore（`_psc_post_step_restore`，复用 A2 机器）、惰性计数（`_psc_count_route_hits`/`_psc_log_round`，只挂钩主流 train()，train_only_prompts 故意不挂）、checkpoint `psc_state`（ON 臂新增，OFF 逐位不变）+ fail-closed load；
3. `Models/Server_DF.py`：`consolidate_prompts()`（边界轮 streak 候选 → 快照 → 逐对合并 → drift 门滚回 → 提交）、`_psc_surrogate_routing`（RNG 中立测量 pass，全部已见类）、`_psc_seen_classes`、core 集合每轮广播、F3 互斥/参数校验、RESUME_MUST_MATCH +5 键、checkpoint 存取 + fail-closed resume；
4. `tests/test_psc_consolidation.py`（6 测试全 PASS）：T1 streak+边界合并提交、T2 drift 滚回、T3 max_merge+共享行排除、T4 邻域萎缩 streak 重置、T5 客户端冻结 restore/top1 尺度不变/stale-task no-op、T6 merge-only 不变量（j 行原值保留）；
5. 回归：test_a5v2_refit_switches（修复 TestRefitInputFeat 漏 skipUnless 的 3 个 CUDA 测试）/test_route_ablation_switches/test_track_e_and_b2/test_global_head_rebuild/test_diag_utils_compat 全绿。

### 25.5 已知边界与诚实披露

1. 相似度是 surrogate 代理测量（cls features 上 top-1 邻域质心余弦），非各 client 真任务分布；τ=0.95 高阈值 + streak=3 + drift 滚回三重防假合并，但残余风险如实声明；
2. 超参敏感性（τ/K/max_merge/δ）不在本判据臂范围，GREEN 后如需再议消融；
3. 合并只动 prompt_key/prompt 两张表，anchor/表征侧与官方逐位一致；RNG 中立设计下 OFF = 逐位 no-op（单测覆盖），ON 时表征侧差异 100% 归因于 residual 冻结 + 边界合并；
4. PSC 与 A2/route_fix 互斥是设计决定（A2 判死；route_fix 是修复级）——不是能力缺失；
5. residual 冻结会让旧任务方向在 t1 内不被新任务训练覆写，但若 S_old 支撑集随任务数增长逼近全池，residual 比例会膨胀（core 容量被挤占）——探针天花板判读正是对这一点的直接检验；
6. 结果阶梯参照（全轨迹 corrected 终值，本判据臂的直接背景）：cifar base 75.04 / M2' 93.08 / oracle 96.21；INR base 77.12 / M2' 81.93 / oracle 84.82——PSC 的主张空间 = M2' 与 oracle 之间的差距部分归因于 prompt 池方向覆写。

## 26. Fusion 侵蚀探针（候选方向 A/B 立项判据，2026-11-02 预注册）

### 26.0 背景与机制空洞（代码 + 普查事实）

- 普查（diagnostics_output/row_usage/）：cifar 路由坍缩到 1–2 共享行（跨客户端 Jaccard 0.47–1.0）；INR 行使用按数据自然分区（跨任务 Jaccard 0.13–0.19，跨客户端 0.125–0.22）。capacity opening 因 INR 侧不满足"Jaccard ≥0.5"不立项（§25.3 预注册普查判据的换方向分支）。
- 代码事实：`kd_fusion_prompt` 学生 = `deepcopy(clients[thisclients[0]].prompts)`（thisclients 每轮全 5 客户端 → **锚定恒为 client 0**），蒸馏 surrogate **只装当前任务类**（L736 current_class），融合后广播对所有客户端**全池整体替换**（`get_global_proto_and_head`）。旧任务功能在融合中无信号路径（仅从锚定者被动 deepcopy 继承）。
- 候选 A（按行归属融合）：非锚定客户端本地训练更新的行每轮被广播清零（其行在锚定者副本中 = 上一轮广播态）。INR 侧机制，cifar 预期退化 ≈ 无损。
- 候选 B（轨迹感知融合）：融合蒸馏对旧任务零约束，10 epoch Adam(+wd) 只受当前任务压力 → 旧任务功能被侵蚀，逐轮累积。cifar 3.6pp 轨迹损失（边界 96.7 → 终局 93.08）与 INR 2.9pp 的共同空洞。PSC 教训（冻结参数饿死适应）的反命题：保输出不保参数。

### 26.1 探针设计

Resume M2' 全轨迹 run 一个融合轮（round 20 = task 4 首轮，旧任务 0–3），`--fusion_probe_dump` 在融合前后各 dump 一次池（纯读，RNG 快照保护，不影响训练流）。离线脚本 `diagnostics/probe_fusion_erosion.py` 从探针 run 的 `round_0020.pth` bootstrap，对每客户端每旧任务做**配对评估**（同 (client,task) 固定种子，三池变体共享同一 shuffle/dropout 序列）：

- **ΔA(c,t) = acc(own_local) − acc(fused)**（非锚定客户端，own head）：广播替换对非锚定者旧任务的丢弃量（A 的天花板）
- **ΔB(c,t) = acc(anchor_local) − acc(fused)**（全部客户端，own head）：蒸馏+wd 对旧任务功能的侵蚀量（B 的天花板）

heads/protos/模型体全部来自 checkpoint 且三变体共享 → 差异 100% 归因 prompt 池。Sanity gate：dump 的 fused 与 checkpoint 里客户端 prompts 逐位一致，不一致 = 探针失效。

### 26.2 命令（用户 tmux 执行，两条独立可并行）

```bash
# INR：resume inr_a5v2_M2_feat_p40 的 task_03_end，只跑 round 20
python main.py datasets_delay --data_name ImageNet-R --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 40 --surrogate_num 5 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --max_rounds 21 \
    --resume output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42/checkpoints/task_03_end.pth \
    --fusion_probe_dump diagnostics_output/fusion_probe/inr \
    --run_name inr_fusion_probe_r20

# 离线评估（GPU，分钟级）
python diagnostics/probe_fusion_erosion.py \
    --run_dir output/ImageNet-R/fedta/inr_fusion_probe_r20/seed_42 \
    --checkpoint round_0020.pth \
    --dump_dir diagnostics_output/fusion_probe/inr \
    --out_csv diagnostics_output/fusion_probe/inr_erosion.csv \
    --summary_json diagnostics_output/fusion_probe/inr_erosion.json --device cuda

# cifar：resume cifar_a5v2_M2_feat_full 的 task_03_end，同构
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --max_rounds 21 \
    --resume output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42/checkpoints/task_03_end.pth \
    --fusion_probe_dump diagnostics_output/fusion_probe/cifar \
    --run_name cifar_fusion_probe_r20

python diagnostics/probe_fusion_erosion.py \
    --run_dir output/cifar100/fedta/cifar_fusion_probe_r20/seed_42 \
    --checkpoint round_0020.pth \
    --dump_dir diagnostics_output/fusion_probe/cifar \
    --out_csv diagnostics_output/fusion_probe/cifar_erosion.csv \
    --summary_json diagnostics_output/fusion_probe/cifar_erosion.json --device cuda
```

注：RESUME_MUST_MATCH 不含 max_rounds（§25.2 已验证），dump 无 RNG 消耗 + resume 恢复 RNG 态。

### 26.2b run_dir 推断坑 + 已执行修正（2026-11-02 事故记录，先于判读）

**事故**：`--resume` 给显式 checkpoint 路径时 RunManager **从 checkpoint 路径推断 run_dir（`ckpt.parent.parent`），`--run_name` 被静默忽略**（runtime_utils.py L211-219）。INR 探针的产物因此写进了原 M2' run 目录 `inr_a5v2_M2_feat_p40/seed_42`，而非计划的 `inr_fusion_probe_r20`——§26.2 原命令有误（断言"产物落在新目录"前未验证 RunManager 行为）。

**损伤与恢复（已完成）**：
- `round_0020.pth` 被 probe 覆写，原版不可恢复——probe 版是同轨迹 resume 的等价状态（RNG 态恢复 + dump 无 RNG 消耗，但 cudnn 非确定性下不逐位一致）。终态数据不受影响（分析用 task_XX_end / accuracy_matrix）；
- `latest.pth` 已从 `task_04_end.pth` 恢复（内容逐位一致）；
- `train_log.csv` 追加的 20 行 probe round-20 行已剥离（恢复 451 行 = header+450 原始数据；probe 会话备份于 `diagnostics_output/fusion_probe/inr/probe_session_backup/`）；
- `train.log` 已截断回原 4249 行（probe 会话尾部单独备份）；
- `task_XX_end.pth`、`round_0024.pth`、`accuracy_*.csv`、`args.json` 全程未动。

**修正后的离线评估命令（INR，可立即跑）**——probe 的 round_0020.pth（融合后、广播后状态）就在原 run 目录，直接以原 run 为 bootstrap 源（只读，sanity gate 校验 dump 与其逐位一致）：

```bash
python diagnostics/probe_fusion_erosion.py \
    --run_dir output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42 \
    --checkpoint round_0020.pth \
    --dump_dir diagnostics_output/fusion_probe/inr \
    --out_csv diagnostics_output/fusion_probe/inr_erosion.csv \
    --summary_json diagnostics_output/fusion_probe/inr_erosion.json --device cuda
```

**cifar resume 必须走 staging 副本**（避免同样的覆写；checkpoint 副本让 run_dir 推断落到 staging 目录，RunManager 会在那里写 args.json/command.txt，原 run 只读）：

```bash
mkdir -p diagnostics_output/fusion_probe/cifar/staging/seed_42/checkpoints
cp output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42/checkpoints/task_03_end.pth \
   diagnostics_output/fusion_probe/cifar/staging/seed_42/checkpoints/
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --max_rounds 21 \
    --resume diagnostics_output/fusion_probe/cifar/staging/seed_42/checkpoints/task_03_end.pth \
    --fusion_probe_dump diagnostics_output/fusion_probe/cifar \
    && python diagnostics/probe_fusion_erosion.py \
    --run_dir diagnostics_output/fusion_probe/cifar/staging/seed_42 \
    --checkpoint round_0020.pth \
    --dump_dir diagnostics_output/fusion_probe/cifar \
    --out_csv diagnostics_output/fusion_probe/cifar_erosion.csv \
    --summary_json diagnostics_output/fusion_probe/cifar_erosion.json --device cuda
```

### 26.3 判读表（预注册，先于数据）

| 判读项 | 口径 | GREEN | YELLOW | RED |
|---|---|---|---|---|
| A：广播丢弃量 | mean ΔA（非锚定客户端 × 旧任务 0–3，corrected） | ≥ 1.0pp → A 立项 | 0.3–1.0pp → 量小，需评估实现成本再议 | < 0.3pp → A 判死 |
| B：蒸馏侵蚀量 | mean ΔB（全部客户端 × 旧任务 0–3，corrected） | ≥ 1.0pp → B 立项 | 0.3–1.0pp | < 0.3pp → B 判死 |
| 辅助方向一致性 | ΔA/ΔB 正值 cell 占比 | ≥ 60% | — | < 60% 且 mean 靠近门限 → 降一档 |
| Sanity | dump fused vs checkpoint 客户端 prompts | 逐位一致 | — | 不一致 = 探针失效，重跑 |
| B 的双数据集门 | 26.3 B 两数据集判读 | 双 GREEN → 写机制 | 单 GREEN → 只披露 | — |

A 立项只需 INR GREEN（A 本质是 INR 侧机制，cifar 预期近 no-op，cifar 判读只作无害性参考）；B 瞄准两数据集共同空洞，按防堆参数纪律须双 GREEN 才写机制（不允许各自调参）。两方向均 RED → 融合侧判死，转向 head/proto 侧诊断。

### 26.4 已知边界（预注册时声明）

1. 单轮采样：round 20 是任务 4 首轮（刚换任务，本地训练对旧行扰动最大的阶段之一），量级或高于轮均值——ΔA/ΔB 是"该轮丢弃/侵蚀"的瞬时量，跨轮复利不外推，只作立项判据；
2. evaluate 语义官方原样（`train=True` 前向、shuffle），配对种子使噪声在变体间共模抵消，但绝对值含 dropout 噪声；
3. heads[t] 在 task 4 首轮尚无 heads[4]，当前任务无参考 cell（判读只依赖旧任务 cell，不受影响）；
4. 探针 run 的 round 20 与原 run bit 一致仅当 dump 无 RNG 消耗成立（快照保护 + torch.save 不消费 torch RNG）；
5. ΔB 含 Adam wd 的被动侵蚀与蒸馏梯度的主动侵蚀，二者不分离（对立项判据无差别，对机制设计有——GREEN 后需 ablation）。

### 26.5 INR 探针结果与判读（2026-11-02，按 §26.3 预注册判据）

| 判读项 | 实测 | 预注册判决 |
|---|---|---|
| A 广播丢弃量 | mean ΔA = **−0.35pp**（n=16，正值 cell 44%） | **RED**（<0.3pp；判死） |
| B 蒸馏侵蚀量 | mean ΔB = **0.000pp，全部 20 cell 精确为 0** | **RED**（<0.3pp；判死） |

**两个尖锐的证伪事实**：

1. **ΔB ≡ 0（逐 cell 精确为零）**：INR 任务间路由分区（跨任务 Jaccard 0.13–0.19）意味着当前任务的蒸馏批次只路由进当前任务行——旧任务行的 prompt 与 key **收不到任何梯度**，Adam(+wd) 对 grad=None 的参数跳过 → 融合对旧任务功能是**逐位 pass-through**。"轨迹感知融合"(B) 瞄准的侵蚀在 INR 上数学上不存在（§26.4-5 的 wd/蒸馏不分离问题就此消失——两者都是零）。
2. **ΔA < 0（广播是稳定器不是破坏者）**：非锚定客户端本地池相对融合池在旧任务上平均**低 0.35pp**——本地训练对邻近旧任务（t2 −1.9pp、t3 −0.7pp）的扰动大于对老任务（t0/t1 +0.6pp）的小改善；广播把大家重置回共识，净效应为正。"按行归属融合"(A) 要保留的"被丢弃更新"净值为负，保留反而更差。

**合并推论**：INR 旧任务功能在池侧的演化路径 = 旧任务专属行任务结束后**永久冻结**（无梯度来源）+ 共享行（13–19% 流量）经锚定者谱系被当前任务压力缓慢漂移。池侧干扰是小头；INR 轨迹损失主要不在融合侧。

**教训（预注册制度的价值）**：探针前的预期是 A/B 至少一个 ≥1.0pp GREEN——两者均被证伪。若跳过探针直接写机制，将复刻 PSC 式沉没成本。普查的"锚定不对称 +0.53pp"签名真实存在，但机制解释反了：锚定者优势不是"保住了自己的更新"，而是"从未离开共识"。

**按 §26.3 预注册预案执行**：两方向均 RED → **融合侧判死，转向 head/proto 侧诊断**。cifar 臂（§26.2b staging 命令）仍值得跑一次作为披露级事实：cifar 共享行拓扑与 INR 相反——当前任务蒸馏梯度会真实进入旧任务所用行，若 cifar ΔB ≠ 0 则定位了 cifar 3.6pp 轨迹损失的融合分量，为下一个方向提供路标；不改变 RED 判决，不作为立项门。

## 27. PGC：原型引导的类间几何先验（2026-11-02 预注册，探针先行）

> 用户定案（2026-11-02）：巩固臂（M5）带轻 replay 档位代价，放弃；要 no-replay 方案。
> 本节先注册 PGC 方案与 P1/P2 探针判据（先于数据），探针双门过后才写判据臂。

### 27.0 背景与方案定义

**瞄准点（no-replay 档内最大空白）**：边界内空间——"学的时候就没学好"。证据：

- cifar M2' 边界行 96.09/96.67/**92.58**/94.46/94.11（T2 凹陷 2–4pp），base 在 T2 同样凹陷（90.02）→ 凹陷是协议结构性的；
- INR M2' 边界行 86.22/82.59/**81.49**/84.43/82.03（T2 全阵最低）；
- T0（无旧类干扰）refit 配方只差 0.69–0.84pp（M2' T0 边界 vs oracle 探针 mlp_local：cifar 96.09 vs 96.93、INR 86.22 vs 86.91）——**多任务边界多出来的 1.5–3pp 不是头配方问题**；
- 残余轨迹分量（1.4–1.7pp）已论证低于 GREEN 门，不立项（M5 判死的另一面）。

**机制假设（class_mask 语义下的 margin 挤压）**：native 评估带 class mask，新任务样本不可能被预测到旧类——损失路径不是"错分到旧类"，而是：新任务样本路由进旧类占用的行（cifar 共享核跨任务 Jaccard 0.75/0.89）→ 新旧类特征在同几行上混合 → **新类之间的判别 margin 被挤窄** → 新类间混淆增加 → 边界内损失。

**方案**：FedTA 协议每轮把全类原型广播到每个客户端（`global_protos`，BGPS threshold=0.25 修复 + fix_keys 冻结维护），但客户端**训练时从未消费它们**（唯一用途是 server 侧修复；evaluate 不用原型）——协议里存在一条**广播了却无人使用的信息流**。PGC 在 phase-2 训练中激活它：损失从 `CE` 变为 `CE + λ·L_sep`，`L_sep` = 当前 batch 新类样本特征与**旧类**原型的排斥项（margin 余弦形式），让新任务学习不挤占旧类几何区域。

**空间配准（代码事实，探针依赖）**：

- `local_protos[c]` = **x1 1536 维 anchor-concat** 空间的类均值（Client_DF L746-767：original pre_logits → vit(task_id) 前向 → `model(feat)` 的第二个返回值 = Tail_Anchor.forward 的 `x1`，Tail_Anchor.py L59）；
- global_protos = 各客户端 local_protos 的 BGPS 择优/修复联邦共识（Server_DF L385-528），每轮广播，客户端存 `self.global_protos`（Client_DF L983），checkpoint 持久化并经 bootstrap 恢复；
- M2' refit feat 头的输入 = z_op 768 维 pre-anchor（`_a5_collect_head_inputs(feat_only=True)`，Client_DF L825-865），部署语义 = 垫零 1536 + 锚半区权重置零（L867-880）——**与 protos 的 x1 空间不同源**（P1/P2 因此各用各的空间，见 27.2）；
- 梯度通路：prompt 特征依赖可训练池 → `λ·L_sep` 对 prompt/key 有梯度；ViT 冻结、Tail_Anchor、heads 全不碰 → 与 M2'（动 heads 存储）**正交叠加**。

**实现预告（双门过后写进判据臂）**：`L_sep = mean_batch Σ_oldc relu(cos(f_i, p_oldc) − m_sep)`，f_i 为 batch 内新类样本的 prompt 增强特征，旧类原型集合 = global_protos 减去 current_class；`--pgc` 开关 + `λ`/`m_sep` 两个标量（预注册即默认，不扫描）；评估路径零依赖原型（训练时消费、corrected 评估零改动）。

### 27.1 结构判据自检与死亡名单对照

| 标准 | 检验 |
|---|---|
| 信息流形状 | 新建消费连接（protos → phase-2 损失），组件角色改变（protos 从"server 维护对象"变"客户端训练先验"）——激活协议内既有的死信息流，非换数据口径 |
| 机制假设 | margin 挤压假说（27.0），且可被零训练探针（P1/P2）预先证伪 |
| 新参数 | 1 开关 + λ + m_sep（预注册即默认） |
| 与 M2' 正交 | 表征侧 vs 接口侧；OFF 逐位官方路径 |
| no-replay | 原型是内存向量，官方协议本来就在收集/广播/checkpoint 它们——访问级别零变化 |

死亡名单：≠ A2/PSC（不冻结、不合并任何行）；≠ route_fix（不动路由算子）；≠ 融合 A/B（不碰 SIKF）；≠ NCM 读出（判死的是原型做**读出**；PGC 是**训练约束**，评估零依赖）；≠ SIKF-Seen（不换蒸馏数据，新建训练信号，且该路线已被用户判为修复级弃用）；≠ 池化（无聚合）。

### 27.2 探针设计（P1+P2 一体，`diagnostics/probe_boundary_attribution.py`，零训练）

对 M2' run 的每个 `task_XX_end.pth`（cifar `cifar_a5v2_M2_feat_full` / INR `inr_a5v2_M2_feat_p40`，均 seed 42）：

**P1 边界归属二分**（问题：边界空间的损失在头侧还是特征侧）：

1. **配对核心（v3 修订，2026-11-02 按两次实测证据修正设计）**：每客户端用 `_a5_collect_head_inputs(t, feat_only=True)`（客户端方法直接复用，零逻辑漂移）收集任务 t train split 的 z_op；test split z_op 由探针收集（同前向语义、顺序 loader）；**恢复的 native refit 头（heads[t]，eval 模式，z⊕0 垫零前向——M2' feat 形状锚半区权重恒零，head(z||0) 数学上即 native 头功能）与 `fit_head`（import probe_oracle_headroom.py，mlp_local 同配方：768→256→ReLU→Dropout(0.1)→C、Adam 1e-3/wd 1e-4、100 epoch、batch 64、masked CE）重训头消费同一份缓存特征** → `rawΔ_t = R_retrain − R_native_cached`，batchwise 实现噪声在配对内共模抵消；
2. 官方 `evaluate_all_seen_tasks` 串联重放（单快照 → 逐客户端 → 恢复，复刻 run_full_evaluation 结构）保留为**灾难 gate**：全行对账 R[t][k]（k=0..t，客户端均值）|diff| > 1.0pp 即 SystemExit——错头/错池/错任务类 bug 偏 multi-pp，在此拦截；亚 1pp 偏差作为实现噪声证据披露进 summary，不作 gate 失败；
3. 配方差基线 `Δ0 = rawΔ_0`（T0 无多任务干扰，预期 0.5–1.2pp 带内，见 27.4-3）；校正 `Δ̃_t = rawΔ_t − Δ0`。

**P2 几何侵入测量**（问题：凹陷任务的新类特征是否被旧类原型区域拉扯）：

1. 原型：`client.global_protos`（checkpoint 恢复的联邦共识单份，x1 1536 空间）；旧类 = protos keys 中不属于任务 t 当前类、且属于任务 < t 已见类的键；
2. 新类特征：任务 t train split 的 x1 前向（与官方 proto 收集同链：original pre_logits → vit(task_id) → feat → model(feat, y) 的 x1），按类均值得新类原型；
3. 度量：`proto_intrusion(t)` = 新类原型到最近旧类原型的余弦（按新类取均值）；old-old 参照 = 旧类原型彼此最近邻余弦的 mean/p95；样本级 margin = cos(f, own_proto) − max cos(f, old_protos) 的均值。

### 27.3 判读表（预注册，先于数据）

| 判读项 | 口径 | 门 |
|---|---|---|
| 灾难 gate（sanity） | 官方重放 vs accuracy_matrix 全行 R[t][k]（k=0..t） | \|diff\| ≤ 1.0pp；错头/错池/错任务类 bug 偏 multi-pp 在此拦截。亚 1pp 偏差 = batchwise 实现噪声（27.4-6 实测证据），披露不拦截 |
| P1 特征说（FEAT） | 全部任务校正 `Δ̃_t`（配对口径 rawΔ = retrain − native_cached） | **全部 < +0.3pp** → 头已吃满特征，PGC 立项空间成立 |
| P1 头质量说（HEAD-LAG） | 任一任务校正 `Δ̃_t` | **≥ +1.0pp** → 头滞后主导，PGC 前提坍塌，方案死 |
| P1 混合带 | 0.3–1.0pp | per-task 报告，判"部分"，按最大任务定夺 |
| P2 侵入证据（INTRUDE） | 凹陷任务（cifar/INR 均为 T2）`proto_intrusion` vs 非凹陷任务均值 | **差 ≥ +0.05 cos** 或凹陷任务绝对值 > old-old 参照 p95 |
| P2 无侵入（CLEAN） | 同上 | 两条件均不满足 → 与任务凹陷无关联，PGC 机制假设死 |
| **双门（立项判据）** | P1=FEAT **且** P2=INTRUDE | 过 → 27.6 判据臂定稿；任一失败 → PGC 判死，零机制成本（与 A/B/PSC 预注册判死同构） |

主判读场 cifar（共享核拓扑、T2 凹陷证据链完整）；INR 分区路由预期侵入弱，单独判读——INR 双门失败不否决 cifar 单数据集主张，但按防堆参数纪律，**cifar-only GREEN 的机制主张限定"共享核型场景"措辞**。

### 27.4 已知边界（预注册时声明）

1. **类语义相似 confound**：T2 的类与旧类可能天然相似（数据集固有）——P2 判读以"跨任务相对差"为主证据（凹陷 vs 非凹陷），绝对值仅作辅助；
2. **原型坐标滞后**：global_protos 是旧类最后训练时点的 BGPS 共识快照，特征漂移后 P2 度量的"新类 vs 冻结原型"几何可能低估当前真实侵入——方向上偏保守（低估侵入 → 偏向 CLEAN），不破坏 INTRUDE 判决的可信度；
3. **Δ0 基线内生**：配方差基线取自 T0 任务实测，若 T0 本身异常（边界内损失在 T0 已有），基线失真——判读时先检查 rawΔ_0 落在 0.5–1.2pp 预期带；
4. **单 seed**：探针不扫种子（判死性探针，非效应量估计）；判据臂若 GREEN 才按惯例扩 3 种子；
5. **P1/P2 都是判死性探针**：双门设计保证机制代码零沉没成本；若 P1 判 HEAD-LAG，其副产品（refit 机器在多任务边界退化的定位）转交修复级路线处理，不作为结构创新主张；
6. **batchwise 评估的实现噪声（2026-11-02 两次实测，v3 修订依据）**：批内多数投票的 prompt 选择依赖 batch 组成 → 评估结果依赖 shuffle 实现。实测：探针两种合法 RNG 结构对 matrix 的 mean 偏差 0.44 / 0.68pp（cifar T0），官方 run 内双口径自差 0.08pp（server phase 96.17 vs matrix 96.09）；c0（首个评估客户端）跨结构逐位一致证明恢复无 bug，偏差全部来自 c1–c4 的序列敏感（±0.5–0.9pp/client）。推论：**matrix 的 R[t][k] 是单次实现，自带 ±0.5pp 级实现噪声带**；本探针的 P1 判读因此用同缓存配对设计（共模消噪），官方重放降级为灾难 gate；跨 run 比较任何 <1pp 的差异时须引用本条。

### 27.5 探针命令（用户 tmux 执行，GPU 各约 15–30 分钟）

```bash
# cifar 主判读场（5 checkpoints 串行）
python diagnostics/probe_boundary_attribution.py \
    --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
    --checkpoints task_00_end.pth,task_01_end.pth,task_02_end.pth,task_03_end.pth,task_04_end.pth \
    --out_csv diagnostics_output/boundary_probe/cifar_attribution.csv \
    --summary_json diagnostics_output/boundary_probe/cifar_attribution.json --device cuda

# INR 对照场
python diagnostics/probe_boundary_attribution.py \
    --run_dir output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42 \
    --checkpoints task_00_end.pth,task_01_end.pth,task_02_end.pth,task_03_end.pth,task_04_end.pth \
    --out_csv diagnostics_output/boundary_probe/inr_attribution.csv \
    --summary_json diagnostics_output/boundary_probe/inr_attribution.json --device cuda
```

（产物按 §24.2 落 `diagnostics_output/boundary_probe/` 家族目录。）

### 27.5 cifar 探针结果与判读（2026-11-02，按 §27.3 预注册判据）

探针执行记录：v1/v2 两版因 batchwise 实现噪声误设逐位 sanity gate 而 fail（修订见 27.4-6）；v3 曾有 native_cached 标签错位 bug（class_mask 原序 vs 布尔掩码升序错位 → 14% 假读数，HEAD-LAG 判决作废），修复为整数列索引后重跑。下表为修复后读数。

**P1 配对读数（native_cached = M2' refit 头在共享 z_op 缓存上；retrain = mlp_local 同缓存）**：

| t | native_cached | retrain | rawΔ | 校正 Δ̃（−Δ0=0.59） |
|---|---|---|---|---|
| 0 | 96.34 | 96.93 | +0.59 | （配方差基线） |
| 1 | 96.69 | 97.65 | +0.96 | +0.37 |
| 2 | 92.71 | 95.30 | **+2.59** | **+2.00** |
| 3 | 94.68 | 96.47 | +1.79 | +1.20 |
| 4 | 94.22 | 96.94 | +2.72 | **+2.13** |

**P2 侵入读数**：T1 0.608 / **T2 0.740** / T3 0.671 / T4 0.755——T2 vs 非凹陷均值 gap +0.062 ≥ 0.05（弱 INTRUDE，T4 高侵入不凹陷破坏共变图景）；sample_margin 跨任务递减 0.449→0.274（与挤压方向一致）；第二条件（绝对值 > old-old p95 0.919）不成立。

**双门判决：P1=HEAD-LAG → PGC 判死**（按 27.3 预注册，双门须 P1=FEAT）。

**判读（关键事实链）**：

1. **头滞后主导，特征几何次要**：T4 的 retrain 96.94 ≈ T0 的 96.93（同为"刚学完"任务、特征无干扰差异）——T4 边界 2.1pp 损失 100% 在头侧；T2 retrain 95.30 距 T0 基线仅 −1.6pp（含任务难度）。头滞后随任务数递增（t1 +0.37 → t4 +2.13）。PGC 治的病（特征 margin 挤压）是次要成分——判死成立。
2. **副产品（按 27.4-5 预案转交修复级）**：M2' refit 头配方在多任务边界吃不满特征——mlp_local（hidden 256 + **train-only 标准化**）比 M2' refit（hidden 512 + 无标准化）在 T2/T4 高 2.0–2.7pp。两因素（标准化 / hidden 容量）未分离，均为修复级消融。`--head_standardize`×`--refit_input feat` 组合语义已验证正确（Client_DF L891-906：统计量在垫零前的 768 维 z_op 上计算、锚半区统计量恒等 0/1、训练后锚半区权重置零兜底——训练/部署分布一致，零代码改动可直接跑）；hidden 消融需参数化 `A5_REFIT_HIDDEN`。**这是 no-replay 档内当前最明确的 +2pp 级增益路径**。
3. P2 的弱 INTRUDE 与 margin 递减趋势作为披露保留（几何分量存在但被头滞后掩盖），不构成行动项。

### 27.6 判据臂（已作废，随 PGC 判死）→ 修复级配方分离探针（2026-11-02 预注册）

原占位（cifar + INR 2 任务短程 M2' vs M2'+`--pgc`）不再执行——探针双门 FAIL，`--pgc` 机制代码零行实现（预注册判死第三次兑现：融合 A/B、PSC、PGC）。

**27.5-2 转交项的执行方案（配方分离探针，零训练）**：探针 v3 的 retrain 列升级为 2×2 配方网格 `--retrain_grid std:256,std:512,nostd:256,nostd:512`（`fit_head` 新增 `standardize` 关键字，默认 True = 封板语义逐位不变；主列 std:256 保留历史 seed 维持跨版本可比）。四变体在同一份缓存特征上配对训练评估，读数为逐任务 rawΔ（vs native_cached）。**判级声明：本探针与后续训练臂均为配方卫生（修复级最轻档），不构成方法贡献主张，只产数字。**

**分离判读映射（预注册，先于数据）**：

| 读数形态（以 T2/T4 校正滞后为准，≥1.0pp 显著） | 解释 | 行动 |
|---|---|---|
| nostd512 ≈ 0 且 std256/std512 显著 | 标准化是主因 | 训练臂直接跑 `--head_standardize`（现成开关，命令见下） |
| nostd256 ≈ std256 显著，nostd512/std512 ≈ 0 | hidden 容量是主因 | 先参数化 `A5_REFIT_HIDDEN`（新开关），再跑训练臂 |
| 仅 std256 显著，其余 ≈ 0 | 联合效应不可分 | 训练臂配方 = std256 完整版（标准化 + hidden 参数化都要） |
| 四变体全部 ≈ 0 | 差异不在两因素（残余候选：lr/epoch/batch/masked-CE 细节） | 暂停，报告后另议 |

**离线读数预测训练臂的合法性**：M2' 的 heads[t] 即边界轮最后一次 refit 的产物，离线用边界 checkpoint 的接口状态重做 refit 与训练臂的边界头同构——训练臂只需跑胜出配方一次全程，避免盲跑。

**训练臂命令（胜出配方为标准化主因时；cifar 全程）**：

```bash
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --head_standardize \
    --run_name cifar_a5v2_M2std_full
```

主判据 = corrected 边界列 T2（vs M2' 92.58，≥ +1.0pp GREEN）；探针对账带 = T2 预期 ~95.3±0.5（27.5-1 表 retrain 读数）；约束 = T0/T1 边界与终局 AA 不降。

**分离探针实测（2026-11-02，cifar，按 27.6 预注册判读）——第四形态触发，训练臂暂停**：

| 校正读数 | std256 | std512 | nostd256 | nostd512 |
|---|---|---|---|---|
| t1 | +0.37 | +0.71 | +0.39 | +0.38 |
| t2 | +2.00 | +2.52 | +2.11 | +1.88 |
| t3 | +1.20 | +1.87 | +1.41 | +1.31 |
| t4 | +2.13 | +2.51 | +2.34 | +2.28 |

判读：**标准化效应 ≈ 0、hidden 效应 ≈ 0**（四配方在每任务互相重叠 ±0.3pp 内）；公共效应（任一 mlp_local 配方 vs M2' refit 头的滞后读数）稳健显著（t2/t4 ≈ 2pp）。两因素排除 + `A5_REFIT_*` 常量逐项核对与纸面配方一致（epoch/batch/lr/wd/dropout 全同）——**fit_head 与 refit 的残余差异不在已测因素**，剩余候选：softmax 归一化域（refit 的 Chead(200) 全类 masked CE vs fit_head 紧凑 15 列）、头维度/初始化（1536→512→200 vs 768→h→15）。

**按预注册第四行执行：`--head_standardize` 训练臂暂停不跑**（标准化无分离效应，预期拿不到 2pp；预注册判死纪律避免一次 4–8h 白跑）。头滞后 2pp 事实保留在案（27.5-1），配方归因转 softmax-域假说探针（待议），优先级让位于仓库清理与投稿主线。

### 27.7 2pp 归因阶梯探针（2026-11-02 预注册，用户指令"先归因 2pp"）

单变量阶梯（共享缓存上训练、配对评估），每步只动一个因素：

| 阶梯 | 输入 | 结构 | CE 口径 | 阶梯差分离 |
|---|---|---|---|---|
| A = std256 | z_op 768 | 768→256→C_task | 紧凑 softmax | — |
| B = z7m:256 | z_op 768 | 768→256→label_num | **refit masked CE**（not_mask=0..nb_classes−1 非任务列填 −inf；label_num=200 含裸奔列——复刻 Chead(200) 写死语义） | A−B = CE 域 |
| C1 = x1m:512 | z⊕0 1536 | 1536→512→label_num | refit masked CE | B−C1 = 结构维度 |
| native_cached | — | 1536→512→label_num | — | C1−native = **训练环境**（特征时点） |

**C1−native 的机制背景**：refit 头的训练特征 = 广播前（本地 phase-2 后）池状态；评估与探针缓存 = 广播后池状态。若 C1（完全复刻 refit 配方、只换广播后缓存）显著高于 native，则 2pp 归因于**部署-训练特征时点错位**——与头滞后随任务数递增（t1 +0.37 → t4 +2.13，广播前后特征差随任务数变大）的模式吻合。

**预注册判读（cifar 主场，以 T2/T4 校正读数为准）**：

| 阶梯差 | 门 | 归因 |
|---|---|---|
| C1 − native_cached ≥ +1.5pp | 主 | **特征时点错位**（refit 应移到广播后训练——训练协议改动，另议立项） |
| B − C1 ≥ +1.0pp | 主 | 结构维度因素（1536 垫零结构劣于 768 紧凑结构） |
| A − B ≥ +1.0pp | 主 | CE 归一化域因素（紧凑 softmax 优于全类 masked） |
| 多段同时 ≥ +0.8pp | 报告 | 多因素共担，按比例披露 |
| 全部 < +0.3pp | 复核 | 与 27.5-1 的 2pp 稳健读数矛盾 → 探针 bug，查实现 |

判级提醒：归因结论服务于**修复级配方/协议改进**（不构成方法创新主张）；若判"特征时点错位"，其行动项（广播后 refit）是训练协议结构改动，需另立判据臂验证，本探针只归因不立项。

### 27.7-1 阶梯实测与归因判决（2026-11-02，cifar，按 27.7 预注册门）

| 校正读数（T2 / T4） | 实测 | 预注册门 | 判决 |
|---|---|---|---|
| A−B（CE 域） | +0.04 / −0.15 | ≥ +1.0 | **排除** |
| B−C1（结构维度） | +0.10 / +0.12 | ≥ +1.0 | **排除**（B→C1 含 hidden 256→512，前轮网格已证 hidden 效应 ≈0） |
| C1−native（特征时点） | **+1.97 / +2.36** | ≥ +1.5 | **触发——主因成立** |

（判决算式澄清：第三段 = x1m512 变体自身的校正读数——C1 与 native_cached 的配方完全同款，唯一差异是训练特征时点：C1 用探针的广播后缓存，native 头是训练流程在广播前缓存上产出的。绝对值对照：T2 x1m512=95.00 vs native=92.71；T4 96.90 vs 94.22。）

**归因判决：2pp 头滞后的真凶 = 训练-评估特征时点错位**。refit 头每轮在"本地 phase-2 后（广播前）"的池状态上缓存特征训练，而评估/部署发生在"广播后"的池状态上；错位量随任务数递增（t1 +0.66 → t4 +2.36 校正）——与广播对池的重置幅度随任务数累积一致（本地训练对共享行扰动越大、广播重置差异越大），机制与 §26.5-2"广播是稳定器"互证。CE 域、结构维度、标准化、hidden 四因素全部排除（互相重叠 ±0.3pp 内）；mlp_local 全族的 ~2pp 优势 = 它天然在探针的广播后缓存上训练，与配方无关。

**行动方向（按 27.7 判级声明，另立判据臂验证，本探针不立项）**：refit 特征缓存时点移到广播后（客户端收到 server 池之后再 refit）——训练协议级改动。预期收益 = 头滞后读数的主体（T2 ~2pp / T4 ~2.4pp 边界），代价与交互（与每轮广播循环的顺序、RNG 中立设计是否保持、每轮 refit 语义变化）需在判据臂 spec 里预注册后再动。

## 28. broadcast-post-refit 判据臂（§27.7-1 判决的落地，2026-11-02 预注册）

> 立项依据 = §27.7-1 归因判决：oracle mlp_local 相对 M2' 的 ~2pp 优势来自**特征时点错配**——M2' 的 refit 头在"本地 phase-2 后（广播前）"的接口状态上缓存特征训练，而评估/部署发生在"广播后"状态（客户端 vit 已装载 server 融合 prompt）。判级声明同 §27.7：**修复级协议改进（配方只产数字），不构成方法创新主张**。

### 28.1 方法定义（实现即本节，代码为准）

**开关**：`--refit_timing {local,post_broadcast}`，默认 `local` = M2' 现行为逐位不变；fail-closed：`post_broadcast` 必须与 `--head_refit` 联用（无作用对象拒绝启动）。与 `--refit_input` / `--head_standardize` 正交（同一次 refit 的输入路径与标准化配方照常生效）。

**`post_broadcast` 语义**（相对 M2' 的唯一变量 = refit 时点）：

1. `train()` 末尾钩子退化为纯快照：`_a5_joint_head` 照常快照（phase-2 后、两 evaluate 前），**不再**在本地时点 refit；`heads[task_id]` 保持官方语义（= phase-2 后联训头）。
2. refit 移到 `get_global_proto_and_head()`（广播路径）内、`vit.load_prompts(广播 prompt)` 之后、两个 evaluate 之前：特征缓存在**广播后接口状态**（融合 prompt + 本地 phase-2 后 Tail_Anchor 状态）上采集——与随后的 evaluate 特征流完全同源；refit 配方（masked CE / Adam 1e-3,wd 1e-4 / 100 epoch / batch 64 / 稳定种子）与 M2' 完全一致，替换 `heads[task_id]` 后再 evaluate（server 相位的日志/corrected 口径即反映新头）。
3. 非参与客户端（`get_global_proto_and_head_no_test` 路径）不 refit——与 M2' 一致（refit 只属于 train() 参与者）。

**RNG 中立（不变式，单测覆盖）**：refit 全程四流快照恢复，插入点前后（train() 两 evaluate、广播路径两 evaluate）的 RNG 消费流与官方逐位一致 → 表征侧轨迹（prompt/key/anchor/joint head）与官方 run 逐位一致；本臂 vs M2' 的任何 corrected 指标差 100% 归因于 refit 特征时点。checkpoint schema 不变（heads[task_id] = 广播后 refit 头、a5_joint_head = 联训轨迹；边界轮 checkpoint 在广播后保存，天然捕获新头）；`refit_timing` 增补 RESUME_MUST_MATCH（two-sided，老 checkpoint 兼容）。

**已知边界（如实报告）**：

1. 广播后时点的特征仍带 train split 增强（与 M2' 同数据流）；探针对账时注意。
2. train() 末尾两 evaluate（本地相位日志）在 M2' 用 refit 头、在本臂用联训头——本地相位读数两臂语义不同，跨臂对照一律用 server 相位/corrected 边界口径。
3. `local` 显式值与 legacy args（无该键）逐位一致（getattr 默认路径，单测覆盖）。

### 28.2 判读（预注册，先于数据）

| 判读项 | 口径 | 门 |
|---|---|---|
| 主判据 | cifar M2''（post_broadcast）vs M2'（local）corrected 边界列 T2/T4 差 | **≥ +1.0pp = GREEN**（对齐 §27.6 门槛；§27.7-1 预期主体 ~2pp，对账带 = T2 ~+2.0 / T4 ~+2.4） |
| 早期轮约束 | T0/T1 边界差不降 | 广播-本地特征差随任务数累积（t1 仅 +0.66），早期轮预期 ≈0；若显著为负则时点改动有未建模代价 |
| 探针对账 | 边界列 vs §27.7-1 x1m512 读数（T2 95.00 / T4 96.90） | 收益主体兑现则差值 ≈ 0±0.5；显著低于探针 → 查增强/划分差异 |
| INR 复核 | INR M2'' vs M2'（81.93 corrected）同划分 | 预期同向（错位量随任务数增长机制通用）；不设 GREEN 门，作方向性证据 |
| 轨迹不变性 sanity | 两臂 key/anchor/joint-head checkpoint 抽查 | 逐位一致（RNG 中立设计保证） |

GREEN 后按惯例补 3 种子（1999/2025）；判级不变：修复级配方，不作方法主张。

### 28.3 命令（GPT 审核通过后执行；先跑 T0 三 Gate 回归确认基线逐位不变）

```bash
# T0 回归（三 Gate：官方基线 / M2' OFF / M2' ON 逐位复现，--max_rounds 2）

# 判据臂（cifar 全程；M2'' = M2' + --refit_timing post_broadcast 单变量）
python main.py datasets_delay --data_name cifar100 --data-path ./local_datasets \
    --client_num 5 --task_num 5 --private_class_num 15 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --refit_timing post_broadcast \
    --run_name cifar_a5v2_M2pb_full

# INR 复核臂（M2' 同划分：task_num 5 / private 40 / surrogate 5，
# 与 inr_a5v2_M2_feat_p40 单变量对照）
python main.py datasets_delay --data_name ImageNet-R --data-path ./local_datasets \
    --surrogate_num 5 \
    --client_num 5 --task_num 5 --private_class_num 40 \
    --global_epoch 5 --local_epoch 30 --batch-size 16 --lr 0.001 --seed 42 \
    --head_refit --refit_input feat --refit_timing post_broadcast \
    --run_name inr_a5v2_M2pb_p40
```

主判读场 = cifar（`cifar_a5v2_M2_feat_full` 为 M2' 对照，92.58 T2 / 94.22→96.90 见 §27.7-1）；INR 对照 = `inr_a5v2_M2_feat_p40`（81.93 corrected，task_num 5 / private 40 / surrogate 5）。GREEN 后种子扩展同 §23.2 惯例。

## 29. DHT 可行性判死探针（历史头跨状态迁移候选，2026-11-03 预注册，探针先行）

> 候选动机：M2'' 只解决**当前头**的即时错配；历史头在"出生边界校准后随部署状态演化而老化"（M2'' 未覆盖的时间尺度）。候选方案 DHT（Deployment-state Head Transport）：在相邻部署状态间用当前任务样本拟合 near-identity ridge 仿射映射 `z_{r-1} ≈ (I+ΔA) z_r + b`，把旧头第一层解析迁移（`W1'←W1·A`，部署时可吸收、0 推理开销）。**判死链先行，探针不过不写训练臂。**

### 29.0 F4 重叠审计（硬 gate，已完成）

巩固臂（M5，存档 §23.6-3 / git e7d7a42）= 边界轮用当前接口状态 + 旧任务 train split（**replay**）直接重 refit 旧头——即 DHT 目标池的"有数据直接版"（上界）。判死理由两条：

1. **replay 档位代价**（用户定案 2026-11-02：带轻 replay 代价，放弃，要 no-replay 方案）——DHT 恰好 no-replay（当前任务样本只作 probing 点），**不重叠于死因**；
2. **池上界被论证为 1.4–1.7pp，低于独立 GREEN 门**（"残余轨迹分量…不立项"）——这条**封顶** DHT：DHT（无数据仿射近似）≤ F4（有数据直接 refit）≤ 池。探针直接实测该上界（`acc_refit` 项）。

结论：不算重蹈 F4，但生死门必须按 ≤1.4–1.7pp 的池量级设计（拿 ≥1.0pp = 池的 ≥60%，紧但非不可能）。

### 29.1 判死链（`diagnostics/probe_dht_feasibility.py`，离线零训练，只读保留 run）

| 层 | 检验 | 判死区分 |
|---|---|---|
| Probe-A 仿射可行性 | 相邻部署 checkpoint 对（task_XX_end = 广播后状态）当前任务 train 行拟合 `I+ΔA` ridge；held-out 残差比 | 残差比 ≥1 = 漂移非仿射，仿射假设死 |
| Probe-B 跨任务外推三臂 | B0 identity / B1 当前任务拟合（可部署，no-replay）/ B2 旧任务拟合（仅诊断 oracle）；旧任务 test 行特征恢复比 | B1≈B2 且 <1 = probing 分布成立；B2 有效 B1 无效 = probing 死；均无效 = 仿射死 |
| Probe-C 头级因果老化 | `H_j^sync` = 出生边界 j 的 M2'' 同配方头（hidden 512 / standardize False / 稳定种子），之后不再见任务 j 数据；各后续边界 r 比较 stale / 递归迁移 / 直接迁移 / refit 上界四读数 | Δ(acc_rec−acc_stale) = DHT 净效应；Δ(acc_refit−acc_stale) = 池上界（F4 等价，仅诊断） |
| Probe-D 递归 vs 直接 | 递归复合迁移 vs 单步直接拟合 | direct 好而 recursive 差 = 复合误差累积，非原理死 |
| Probe-E 动态必要性阶梯 | 同一 direct-fit 分布下四级：Identity(stale) / Full Affine / Low-rank Linear(r=16 截断 SVD) / **Low-rank Dynamic**（`B(z)=z+U(g(z)⊙V^⊤z)`，`g=σ(MV^⊤z+a)`，SVD warm-start + 200 步全批 Adam，样本自适应、不可折叠进 W1 = DS-TBB 候选） | 见 29.2b 分支表 |

**29.1b 低秩/动态谱增补**：Probe-A 同时输出拟合残差映射 ΔA 的奇异值累计能量曲线（r=1,4,8,16,32,64）——"广播位移低维性"与"动态桥必要性"共用同一条判死链，零额外拟合成本。

**29.1c 边界缓存布局与臂重映射（2026-11-03 实跑发现）**：边界 checkpoint 的特征缓存**只覆盖截至该边界已加载的任务**（task_00_end 仅 task 0 行，task_03_end 为 task 0–3 行）；行按 (client, task, split, rel_idx) 规范化对齐后跨状态可比。由此探针臂重映射：
- **B1'**（原 B1）= 逐对在 **task-t train 行**上拟合（state t 的最近当前任务，两状态共有）——部署臂的可部署 proxy。**严格的 no-replay 当前任务 probing（task-nxt 样本过 state-t 模型）在离线缓存中不存在**，留待训练臂 Gate 阶段验证（轮开始前缓存，GPT §6 语义）；
- **B2** = tasks ≤ t 全体 train 行拟合（诊断 oracle，不变）；
- **Probe-C direct** = 在两状态共有的 task-j train 行上直接拟合（复合误差 oracle；角色与 recursive 互换参照：rec = 可部署 proxy 链、dir = oracle 上界）；
- **Probe-E 阶梯** = 同共有行拟合分布下的四级对照（机制可分性判读不变，但"可部署分布下的动态必要性"降级为 proxy 口径，正式口径在训练臂 T0 Gate 补）。

门数值（可行性 ≤0.8 / 签名 ≥+1.0pp / Δ_dyn−Δ_lin ≥+0.3pp）不变。

**不变式**：探针只用 z_op 缓存特征（`ensure_features` 复用既有提取，行对齐 fail-closed）；不触碰训练代码；所有种子派生稳定可复现。transport 在特征空间评估（与第一层吸收数学等价，`W1'←W1·A` 等价于对输入做逆迁移）。

### 29.2 判读（预注册，先于数据）

| 判读项 | 门 |
|---|---|
| 仿射可行性（门 1） | held-out B1 残差比中位数 ≤ 0.8 |
| 主签名（门 2） | 末边界旧任务列 client-mean Δ(acc_rec) ≥ +1.0pp = GREEN / ≤ +0.5pp = RED |
| 迁移效率（对账） | Δ_rec / Δ_refit ≥ 0.5（拿不到池的一半则"近恒等仿射"叙事不成立） |
| 机制签名（对账） | corr(gain, drift) ≥ corr(gain, age)：真理论变量是累计特征位移而非任务数 |
| 约束 | 当前任务头读数不受迁移影响（transport 只作用于历史头路径） |

双门全过 → DHT 写训练臂（`M2'' + DHT` 单变量叠加，沿用 §23 范式：RNG 中立 / fail-closed / T0 回归）；门 1 过门 2 RED → 机制成立但池不够，按 F4 预案降级为约束手段不立项；门 1 不过 → 仿射死，当天关闭，不写臂。

### 29.2b Probe-E 分支判读（预注册；DS-TBB 立项与否的唯一依据）

| 数据形态 | 判决 |
|---|---|
| Low-rank ≈ Full Affine ≈ refit 上界 | 位移低维且仿射充分——**不需要动态结构**，DS-TBB 动机死（结构为加而加）；走 DHT/STBB 吸收式路线 |
| Linear 有效，Dynamic 显著更优（Δ_dyn − Δ_lin ≥ +0.3pp）且两者均 < refit 上界 | 漂移含样本自适应分量——**DS-TBB 立项**（真 architecture 主张：动态桥 + 低维位移机制发现） |
| Linear 弱，Dynamic 明显有效 | 漂移非全局仿射——动态结构反而是主机制，DS-TBB 立项且叙事升级 |
| 两者都无效（≤ Identity） | temporal bridge 家族整体关闭（含一切显式/吸收/动态形态） |

判级预声明：若走 DS-TBB 臂，§28 的"不作方法创新主张"在 §30 预注册中**明文升级**为 architecture-level 主张，升级依据 = Probe-B 外推 + Probe-E 阶梯 + 谱曲线三组证据；若 Probe-E 首行成立（线性充分），判级保持协议/机制级，不拔高。

### 29.4 判决（2026-11-03，cifar 4 边界 T3 口径 seed 42；commit c9db197 探针）

读数：门 1 held-out B1 残差比中位数 **1.944**（>1 = 劣于 identity）→ **硬 FAIL**；签名 d_rec +1.72pp（YELLOW，但受门 1 约束不作数）；oracle 分布 direct full-affine 头级 +2.44 vs refit 上界 +2.74；lowrank(r=16) **−75.17pp** / dynamic **−48.65pp**（灾难性摧毁，准确率崩至随机量级）；corr(gain,age)=−0.668 与 corr(gain,drift)=+0.663 方向打架，n=15 不作机制依据。

**判决：门 1 不过 → temporal bridge 全家族（DHT/TABB/STBB/DS-TBB）判死关闭，不写训练臂**（§29.2 预注册条款）。Probe-E 同时实测死掉低秩与动态假设：截断 SVD 保留的主导奇异方向是拟合噪声，r=16 截断即摧毁读数——"广播位移低维"假设无数据支持。

机制注记（写作素材）：z_op 来自离散 top-k prompt 检索路由，跨部署状态漂移含离散跳变成分，非平滑仿射位移——768 维 ridge 在 task-t train 行上过拟合（同任务 held-out 残差即 >1），外推必然失败。头级正效应（rec +1.72 / oracle direct +2.44）解释为 near-identity 正则下温和收缩的判别鲁棒性收益，非可控坐标对齐；且 +2.44 仅在 oracle 分布（旧任务数据拟合）下成立——与 F4 replay 判死口径一致，不可部署。**结论：修复层级选择 = 读出头时点重校准（M2''）优于特征空间 transport 家族**——此负结果直接支撑论文"修复层级"论证（附录：仿射家族排除记录）。

INR 复核：可选存档级（判死不需双确认；若为论文附录完整性补跑一次，成本 ~10 分钟）。

### 29.3 命令（GPT 审核通过后执行；与 M2'' 训练无 GPU 抢占——特征提取为轻前向，建议错峰）

```bash
# cifar 主判读场（保留 run 的 5 个边界 checkpoint）
# 注意 --features_dir 必须指向独立新目录：缓存文件名不含 run 指纹，
# 与其他 run 的旧缓存同目录会触发 fail-closed 拒绝（2026-11-03 实跑踩坑）
python diagnostics/probe_dht_feasibility.py \
    --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
    --data_path ./local_datasets --device cuda \
    --features_dir diagnostics_output/dht_cifar \
    --out_csv diagnostics_output/dht_cifar.csv

# INR 复核（同划分保留 run）
python diagnostics/probe_dht_feasibility.py \
    --run_dir output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42 \
    --data_path ./local_datasets --device cuda \
    --features_dir diagnostics_output/dht_inr \
    --out_csv diagnostics_output/dht_inr.csv
```

产物：`dht_{ds}.csv`（逐 client/born_task/boundary 行）+ `dht_{ds}.summary.json`（门读数 + verdict + 相关性）。判级声明：探针级诊断，产判据不产数字；GREEN 后的训练臂另立 §30 预注册。

## 30. Probe-F：路由不连续机制识别（`probe_routing_flip.py`，预注册 2026-11-03）

> §29 判死后 §30 按用户+GPT 终审冻结方向重立为 **Probe-F**（原"§29 GREEN 后训练臂预注册"路径已随 §29 判死作废）。候选结构 = **DTCPR**（Dual-Timescale Consensus Prompt Router：fast/slow 双 key 池 + consensus gate + 分歧感知过渡混合），是最后一个结构候选，不再设计第三个。定位（GPT 冻结措辞）：
> **Probe-F = routing-discontinuity mechanism identification**——先回答 `non-affine drift ←?— discrete routing transition` 因果链是否成立；成立后 DTCPR 才获得设计依据。探针零训练、只读保留 run、fail-closed、预注册先于数据。

### 30.0 F0：A2-R 死因审计（硬前置）

先例记录（§20/§20.6，2026-10-28，Track A 关闭）：四臂同协议 **D 官方 94.50 > E1 92.69 ≈ E2 ≈92.5 > B 92.38 > A 92.15**；kill criterion 触发（E2 几何全绿——key 有效秩 96+、范数健康、路由熵复活——准确率仍 ≤ D）。因果结论原文：**"塌缩与当前任务准确率双向解耦：E1 塌着 92.69 / E2 修干净 ≈92.5——修复 key 池一分准确率不还"**；关闭措辞："key 池健康与当前任务准确率无因果通路（双向解耦实测）"。

| 项目 | A2-R（已判死） | DTCPR（本探针服务对象） |
|---|---|---|
| 攻击对象 | routing geometry / 可导性 / **当下**检索质量 | **跨 deployment state** 的时序路由稳定性 |
| 目标 | 路由熵复活、key 池健康 | 抑制相邻状态离散换路 |
| route 类型 | 软/正交（R1 init + softmax 全池） | fast-slow temporal consensus（未实现） |
| 是否改 z_op 生成路径 | 是 | **是** |
| 主失败指标 | **实测：routing metric↑ 但 Acc −1.9~−2pp** | 待 F3 实测 |
| accuracy pool | 与 Acc 解耦（历史先例） | F3 直接测 |

**先例含义（冻结）**：routing 侧指标与准确率解耦是实测先例，非推测——因此 **F1 为显式判死门**（换路不存在 → DTCPR 当天关闭，不写臂），**F3 主门 ≥1pp 不放宽**。A2-R 的死因不直接否证 DTCPR（"状态内路由质量" ≠ "跨状态路由稳定性"，对象不同），但它把 DTCPR 的举证责任抬高：必须证明**跨状态** route change 拥有**跨状态**性能退化池（F3b 因果口径），而非"换路样本更难"（F3a 观测口径须防的混杂）。

### 30.1 实际路由语义审计（代码事实，2026-11-03 实读冻结；探针以此为准）

读 `Models/global_prompt.py` + `vision_transformer__l2p.py` + `Client_DF.evaluate` + 保留 run `args.json`：

1. **保留 run 配置**（cifar_a5v2_M2_feat_full）：pool_size=100、length=10、**top_k=1**、prompt_key=True（uniform init）、**use_prompt_mask=False**、**batchwise_prompt=True**、embedding_key='cls'、head_type='prompt'、ViT 骨干全冻结（freeze=blocks/patch_embed/cls_token/norm/pos_embed）、weight_decay=0。
2. **use_prompt_mask=False ⇒ 训练与推理（含 native evaluate）都走全池 top-k 相似度路由**（prompt_mask≡None、task_id 完全不参与路由；L2P 式 task-mask 路径在这些 run 上是死代码）。GPT 的"离散 top-k 路由"前提成立，但按事实修正三处：
   - **K=1**：FlipSeverity_i ∈ {0,1}，无 3/4→2/4 连续剂量；连续剂量改由 **batch vote margin** 承担（F2/F4）；
   - **batchwise 多数投票**：路由是 batch 级量——同 batch 全体样本共享同一 top-1 槽。per-sample FlipSeverity 以其 batch 多数槽为准（部署真相）；per-sample 自身 top-1 偏好翻转另记参考量；
   - **§20 先验**：官方硬路由结构性单槽垄断（keyH≈0.004 bits、全样本命中单槽）→ F1 很可能直接判死；预注册退化分支见 30.4。
3. **路由只依赖 key 池漂移**：query q(x) = 冻结 original model 的 CLS 特征（状态无关）；`sim = l2norm(q)·l2norm(prompt_key)^T`——R_{t−1}≠R_t 完全由 prompt_key 漂移驱动。
4. **z_op 完全由注入 prompt token 决定**：head_type='prompt' 下 z_op = prompt token 输出均值（`x[:,1:1+L].mean(1)`），骨干冻结 ⇒ 部署特征 = `FrozenViT(x ⊕ prompt[R])`。由此 **route × contents 2×2 因子分解穷尽全部漂移通道**（route=选哪个槽；contents=槽内 prompt 值）——比纯翻转相关更强的机制武器，且零训练可测。
5. **native evaluate 语义**：heads[task]（各任务训练结束存的头；M2 臂只 refit 当前任务头，旧任务头跨边界应恒定——探针逐边界记录 heads[j] 张量哈希验证"同头"前提）+ 当前状态路由特征 + class_mask[task] 掩码 + **batch_size=8、shuffle=True**；且全程不调 `.eval()`（Chead Dropout 0.5 保持激活 = Track E 已定性的 evaluation-mode implementation bug）。**探针偏差（有意，预注册）**：同一人群（test_loader[t] ≡ data_split_indices[t]['test']）、固定顺序、batch=8——跨状态/跨臂同 batch 配对成立的前提；读数固定 `head.eval()`（确定性优先；绝对值较 native matrix 系统性高 ~1-2pp，已知偏置，门全部作用于跨状态/跨臂差值故偏置消去；与 accuracy_matrix 对账为 informative annex，容差 3pp）。
6. **Debt-C 部分解答（代码事实落地）**：prompt 为 token 前拼接（`cat([batched_prompt, x_embed])`），DTCPR 的 consensus 融合若在 token 层合成单一 prompt 集，则 ViT 主体只 forward 一次——推理预算 = **2×routing（廉价）+ 1×transformer**，非 2× full model；诊断探针（Oracle 双路完整 forward）可以 2×，正式架构不必。

### 30.2 假设措辞（冻结，先于数据）

> **Hypothesis: discrete top-k route switching may contribute to non-affine deployment-state drift.**

不写 "drift is caused by top-k switching"——因果判定按 30.4 判决树出，措辞不抢跑。

### 30.3 探针设计

**人群与配对**：4 个相邻状态对（task_00→01 … 03→04）；每对分析人群 = 全部 5 client 的旧任务（j ≤ t−1）test 行；跨状态按 (client, task, rel_idx) 规范化对齐（§29.1c 同款 canon）。

**逐状态臂表**（每边界 checkpoint；B/C/λ 臂仅对翻转批次追加）：

| 臂 | route | contents | 作用 |
|---|---|---|---|
| A 天然 | R_t（本状态多数投票） | prompt_t | 部署真相；F2 跳变与 F3 ΔErr 的跨状态两端（t−1 侧的 A 即 D） |
| B 换路隔离 | R_{t−1}（强制旧槽） | prompt_t | 同状态强制旧路 → **route 通道因果量（F3b 主门）** |
| C 内容隔离 | R_t | prompt_{t−1} | 槽不变换值 → contents 通道（2×2 补全） |
| D 全旧路 | R_{t−1} | prompt_{t−1} | = state t−1 的 A（天然臂直接复用，零额外 forward） |

2×2 读法：A−B = route 效应；A−C = contents 效应；(A−B)−(D−C) = 交互；无翻转时 B≡A、C≡D，因子分解退化为 A vs D 的 contents 归因——依然穷尽。

**F1 换路存在性（显式判死门）**：逐 (client, transition) 的 FlipRate（多数槽变更批次占比）+ per-sample 偏好翻转率 + **逐状态路由使用熵（keyH 复刻，直接对照 §20 的 0.004 bits）**。门（第一版工程门，外审可调）：任一状态对 flip 批次占比 ≥5% **或**全局多数槽跨状态轮换 ≥1 次 → F1 过；全部状态对零翻转且全局单槽恒定 → F1 死。

**F2 归一化特征跳变（佐证门）**：per-sample `D_rel = ‖z_t−z_{t−1}‖₂/(‖z_{t−1}‖₂+ε)`、`D_cos = 1−cos(z_t,z_{t−1})`（天然臂跨状态）。检验 flip vs stable 分布差异（K=1 下剂量二值；连续剂量由 t−1 状态 batch vote margin 分层承担）。门（第一版）：flip 中位数 ≥1.2× stable 中位数且 D_rel/D_cos 双指标同向 → F2 过。

**F3 跨状态预测退化（主门，GPT 冻结）**：
- **F3a 观测版**（防"flip 样本本来就难"混杂）：per-sample ΔErr_i = ℓ_t(x_i) − ℓ_{t−1}(x_i)（同头 heads[j]、class_mask[j] 掩码后的 CE）与正确性变化 C_{t−1}−C_t；比较 E[ΔErr|flip] vs E[ΔErr|stable]。要证的是**跨状态 route change 与跨状态性能退化相关**，不是"换路样本更难"。
- **F3b 因果版（主门）**：`route_pool = Acc(B at t) − Acc(A at t)`（同状态、同头、同人群，唯一差异 = 强制旧槽 vs 天然槽）。**门：过渡均值 ≥ +1.0pp**（逐过渡值并报；孤立单过渡超阈且均值 <0.5pp 判 YELLOW 转外审）。
- **1.4–1.7pp 定位修正（GPT 冻结措辞）**：F4 审计的 1.4–1.7pp 是 **readout-side recoverable margin 的历史参考，不是 representation-side routing 干预的上界**；DTCPR 的真上界参考 = F3b 自测池。

**F4 路由边界前兆（增强门，不作硬判死门）**：t−1 状态 per-sample cutoff margin `m_i = s₍₁₎−s₍₂₎`（K=1 即 top1−top2 相似度差）+ batch 级 vote margin（多数票−次多票票数差/B）。检验 m↓ ⇒ P(flip)↑ 分层曲线。论文目标形态的完整机制链：small margin → route flip → feature jump → accuracy drop → smooth transition oracle repairs。

**F5 两层 λ-mix oracle（区分冻结）**：λ ∈ {0, 0.25, 0.5, 0.75, 1}；人群 = 翻转旧任务行；判读用 heads[j] + class_mask[j]：
- **Oracle-1 预测层诊断 oracle**：`l_λ = λ·l(A at t) + (1−λ)·l(D)`——两端点 = 两条部署路径的完整 forward logits，**离线可算**。只回答"两条路径间有无互补信息"，**不构成 DTCPR 结构依据**。
- **Oracle-2 结构合法 prompt 过渡（DTCPR 判据）**：token 层 `P_λ = λ·prompt_t[R_t] + (1−λ)·prompt_{t−1}[R_{t−1}]`（子臂 2a = 同状态两槽混合 λ·C_t[R_t]+(1−λ)·C_t[R_{t−1}]，route-only 参照）。结构合法性：prompt 是 embedding 空间 token 前拼接，凸混合是合法前向；它不对应任何离散路由状态——**这正是 DTCPR consensus gate 会计算的量**（off-manifold 警告由此在语义上消解，仍记录在案）。**判据：Oracle-2b 存在 λ∈(0,1) 使 Acc ≥ max(两端点)+0.25pp 才支持 DTCPR**；仅 Oracle-1 过而 2b 不过 = "互补信息存在但平滑过渡结构不可行"，须换结构而非 prompt 插值。

**fail-closed**：`use_prompt_mask=True` 即中止（探针语义只对 top-k 路由路径定义）；bootstrap 复刻训练划分（checkpoint manifest 校验内置）；跨状态 heads[j] 哈希不一致 → 报告头漂移通道并将 F3 标记 CONTAMINATED（M2 臂预期恒定）；逐边界记录 client 间 key 池哈希（广播后应一致，不一致报告为发现）；A 臂与 accuracy_matrix R[t][j] 对账（informative annex，容差 3pp = dropout 偏置 + batch 组成）。缓存落独立目录 `diagnostics_output/routeflip_{ds}/`（防 §29 同名缓存串用踩坑重演）。

### 30.4 判决树（冻结）

```
F0  A2 先例成立（routing 指标↔Acc 解耦实测）→ F1 显式判死 + F3 ≥1pp 不放宽
F1  换路非平凡？—— 否 → DTCPR 判死（措辞：route switching 未达非平凡水平；
    fast/slow 双路由在部署路径上将恒一致，consensus gate 恒不激活，结构 inert）。
    2×2 因子分解同时给出 contents 通道归因（预期主通道）；representation-side
    叙事记录为"prompt 内容时序稳定性"，不自动立项。
F2  FlipSeverity ↔ D_rel/D_cos 剂量关系（佐证，须与 F1 同向）
F3  主门：F3b route_pool 过渡均值 ≥1pp（F3a 同向佐证）
F4  margin 前兆（增强，论文价值，不作死门）
F5  Oracle-2b > hard switch +0.25pp → DTCPR 获得设计依据，写 §31 训练臂预注册；
    仅 Oracle-1 过 → 互补信息在、平滑过渡结构存疑，换结构再审；
    两者皆不过 → 判死。
```

全链成立形态（论文级）：margin↓ → flip → feature jump → accuracy drop → oracle 修复——**由实测机制直接推出的 representation-side 结构创新**，立项依据硬于 PBRS/TABB/STBB/DS-TBB 全家族。

### 30.5 三债务预注册（GPT 冻结；DTCPR 过 F5 时生效）

- **Debt-A 归因重置**：DTCPR 改 z_op 生成路径 ⇒ F^{DTCPR} ≠ F^{Baseline}，§28 的 bit-identical attribution framework 不再自动适用；M2'' 的 2pp 归因须重建单变量链 Baseline → DTCPR → DTCPR+M2''，T0 逐位 Gate 不适用（改分布级回归 + 逐位锚点子集）。
- **Debt-B 联邦语义**：K^slow 采用 `K^slow_t = μ·K^slow_{t−1} + (1−μ)·K^{fast,global}_t`（对**广播后的全局 key** 做 EMA——"slow 分支 = global deployment routing state 的时间平均"，非各 client 私有历史；GPT 倾向，μ 待训练臂预注册冻结）。
- **Debt-C 推理预算**：见 30.1-6——2×routing + 1×transformer；不写死 2× full forward。

### 30.6 命令（GPT 审核通过后执行；GPU 轻前向，与 M2'' 训练错峰）

```bash
# cifar 主判读场（保留 run 5 边界 checkpoint；天然臂 + native_draws 4 + 翻转批次追加臂，GPU 约 25–35 分钟）
python diagnostics/probe_routing_flip.py \
    --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
    --data_path ./local_datasets --device cuda \
    --cache_dir diagnostics_output/routeflip_cifar \
    --native_draws 4 \
    --out_csv diagnostics_output/routeflip_cifar.csv \
    --out_rows_csv diagnostics_output/routeflip_cifar_rows.csv \
    --out_json diagnostics_output/routeflip_cifar.summary.json

# INR 复核（判死不需双确认；DTCPR 过 F1 后立项前补）
python diagnostics/probe_routing_flip.py \
    --run_dir output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42 \
    --data_path ./local_datasets --device cuda \
    --cache_dir diagnostics_output/routeflip_inr \
    --native_draws 4 \
    --out_csv diagnostics_output/routeflip_inr.csv \
    --out_rows_csv diagnostics_output/routeflip_inr_rows.csv \
    --out_json diagnostics_output/routeflip_inr.summary.json
```

产物：`routeflip_{ds}.csv`（过渡×client×task 聚合 + 2×2 因子分解读数）+ `routeflip_{ds}_rows.csv`（per-row 分析即用表：FlipSeverity / D_rel / D_cos / ΔErr / TopKMargin / vote margin）+ `routeflip_{ds}.summary.json`（F0–F5 门读数 + verdict + 机制链判定）。判级声明：探针级诊断，产判据不产数字；F5 全链成立才写 DTCPR 训练臂预注册（§31）。

> **工程冒烟记录（2026-11-03，2 状态 task_00→01，cifar 保留 run；工程验证非正式判读）**：链路全通（提取 / canon 对齐 / 缓存复现 / shim 强制槽=天然槽时与官方路由**逐位一致** maxdiff 0.0）。过渡 0→1 **零翻转批次、零多数槽轮换**；keyH = 0.0079 / 0.0058 bits（与 §20 单槽垄断先验 ≈0.004 bits 一致，垄断槽 = slot 6，份额 99.94/99.95%）；annex 与 client CSV 最大差 0.59pp；跨状态 heads[0] 哈希恒定（F3 同头前提成立）。正式判读仍以 5 状态全量跑为准——但单槽垄断若在全部 5 状态成立，F1 将按预注册判死分支触发（2×2 因子分解同时给出 contents 通道归因）。

### 30.7 第二轮审核修订（2026-11-03，`6178c74` 有条件批准 → 4 锁死点全部落实；probe v3）

审核结论：APPROVE Probe-F full run, conditional on evaluator/statistical sanity。四项修订全部预注册落实：

1. **统计单位 = batch exposure**：`batchwise_prompt=True` 下 route 是 R_{t,b}（batch 级量），同 batch 样本共享 flip，不得当独立 route observation。F1 已按 batch flip rate（v1 即如此）；F2/F3 保留 sample feature/loss 但**显著性按 batch 聚类**：flip vs stable 对比量配 **paired batch bootstrap 95% CI**（B=1000，`--n_boot`/`--boot_seed` 稳定种子，重抽样单位 = batch）；F2 门升级为"中位数比值 ≥1.2 **且** 池化 batch-bootstrap CI 下界 >0（D_rel/D_cos 双指标）"。**F4 重写为 batch routing-boundary precursor**：m_{t−1,b}↓ ⇒ P(R_{t,b}≠R_{t−1,b})↑——batch 级 vote margin / mean margin 预测 batch 级 flip，per-sample margin 只是批内背景量，不再作前兆单位。
2. **确定性同 batch 分区 + 组成敏感性 annex**：跨 checkpoint 完全相同 deterministic partition 已由 canon 键连接 fail-closed 断言保证（分区由划分 manifest 锁定，非运行时抽样）。新增**零 GPU annex**：同一 checkpoint 下用缓存 per-sample 偏好构造备选固定分区（canon 序 circular shift 1..B−1），统计是否有任何 batch 多数槽改变——若换分组就大量改 route，则 "temporal route" 根本不是样本固有属性，DTCPR 死得更硬（判读措辞预注册："routing is batch-composition sensitive"）。
3. **2×2 标量归因严谨化**：向量恒等式 D−A = Δr + Δp + Δ_int 只对**向量**成立，不对其范数成立——feature 端只报**向量幅值**（mean ‖Δr‖/‖Δp‖/‖Δ_int‖/‖D−A‖），明文禁止百分比归因。标量归因双轨：① 准确率计数池（route_pool/content_pool/interaction_pool）——正确性指示变量逐行可加，恒等式**精确成立**（保留为主归因）；② 新增 **loss 两因素 Shapley**：φ_r = ½[(L_B−L_A)+(L_D−L_C)]、φ_p = ½[(L_C−L_A)+(L_D−L_B)]，φ_r+φ_p = L_D−L_A 精确（nats，逐过渡并报）。
4. **probe-vs-native 0.59pp 差异定位（正式 F3 verdict 前置）**：`--native_draws N`（默认 2）逐 (state, client, task) 做**同 checkpoint/head/人群的 native 语义抽样**（Chead Dropout ON + shuffle=True batch 8，种子派生稳定可复现），把 gap 分解为 (i) native 抽样随机性（draw spread，CSV 本身也是该过程的一次抽样）(ii) eval 模式抬升（probe − mean draw）(iii) 未解释残差；报告 frac_residual_within_draw_spread、max_unexplained_pp、worst_key。**残差未被定位前 F3 verdict 不作数**（0.59pp 已达 1pp 门的过半）。冒烟（2 状态）首批分解：draw spread 均值 0.64pp / 最大 1.23pp（**native Dropout 抽样噪声主导**——与 Track E 的 Dropout 定性一致）、eval 抬升均值 +0.42pp、13/15 键残差完全落在抽样随机性内，仅 state0/client2/task0 有 0.45pp 未解释（3 抽样的 spread 估计本身偏低，疑似功效不足而非语义分歧）；**正式跑提 `--native_draws 4` 收紧 spread 估计**。
5. **F5 净收益门（A2-R 先例约束）**：A2-R 死因 = 常开软混合本身 −1.9pp ⇒ flip 子集 oracle 赢不能读作净赢。预注册：可部署 transition gate 必须 **disagreement-gated**（stable batch 上 G_stable ≡ 0 by design，§31 训练臂的证明义务）；探针显式报告 **G_net = P(flip)·G_flip**（部署级期望池）。G_flip 过 0.25pp 但 G_net 不过同一工程底线 → verdict = `PASS_NET_REVIEW`（转外审，不直接立项）。

**判读优先阶梯（审核冻结，替代原线性树的第一读序）**：Route exists? → Route changes temporally? → Route explains drift? → Route owns accuracy pool?——**前两步死则不救 DTCPR，直接读 2×2 的 content channel**。F1 判死措辞升级为**拒绝自身机制假说**（hypothesis rejected，非"探针失败"）：若 5 状态 FlipRate≈0 且 keyH≈0，则写 "the discrete temporal route-switching hypothesis is rejected"，DTCPR 当天判死不写臂；2×2 同时回答 "route fixed + feature still drifts"——若 Δ_route≈0 而 Δ_content≫0，DHT 的非仿射漂移获得全新且更扎实的解释：**非仿射部署漂移由 prompt 内容演化主导，而非离散槽位切换**（探针推翻设计假说后找到的新机制，论文素材）。

### 30.8 正式判决（2026-11-03，5 状态全量，probe v3 `routingflip-v3`，`--native_draws 4`）

按 §30.6 cifar 主判读场命令执行（保留 run `cifar_a5v2_M2_feat_full/seed_42`，5 边界 checkpoint 全量提取；4 过渡合计 3938 batch、14992 行/末状态）。产物：`diagnostics_output/routeflip_cifar{.csv,_rows.csv,.summary.json}`。判读依据 = §30.4 判决树 + §30.7 审核冻结阶梯，无任何事后放宽。

**读数一：F1 路由状态（5 状态全量，单槽垄断成立）**

| state (ckpt) | keyH (bits) | 垄断份额 | 垄断槽 | key 池跨 client 一致 |
|---|---|---|---|---|
| 0 (task_00_end) | 0.0079 | 99.94% | slot 6 | 一致 |
| 1 (task_01_end) | 0.0058 | 99.95% | slot 6 | 一致 |
| 2 (task_02_end) | 0.0074 | 99.95% | slot 6 | 一致 |
| 3 (task_03_end) | 0.0061 | 99.96% | slot 6 | 一致 |
| 4 (task_04_end) | 0.0060 | 99.96% | slot 6 | 一致 |

keyH 全状态 0.0058–0.0079 bits，与 §20 单槽垄断先验（≈0.004 bits）量级一致；垄断槽 slot 6 跨 5 状态恒定。4 过渡 batch flip rate 全 = 0.0（0/417、0/814、0/1158、0/1549；门 0.05），多数槽轮换事件 = 0。

**读数二：2×2 因子分解（准确率计数池，恒等式精确成立）**

| 过渡 | 天然臂 Acc t−1→t（探针口径，%） | drift | route | content | interaction |
|---|---|---|---|---|---|
| 0→1 | 96.15 → 96.63 | −0.481 | 0.000 | −0.481 | 0.000 |
| 1→2 | 96.65 → 96.22 | +0.432 | 0.000 | +0.432 | 0.000 |
| 2→3 | 95.12 → 92.47 | +2.646 | 0.000 | +2.646 | 0.000 |
| 3→4 | 93.03 → 92.67 | +0.365 | 0.000 | +0.365 | 0.000 |
| **均值** | — | **+0.74** | **0.00** | **+0.74** | **0.00** |

零翻转下 route_pool≡0、content=drift、interaction≡0 是恒等式的退化分支（R_t ≡ R_{t−1} 逐 batch 成立 ⇒ 向量 Δr=0），非测量捷径；loss Shapley/特征幅值按预注册只定义在翻转行（臂 B/C 仅在翻转 batch 前向），零翻转 ⇒ 空集且结构上 φ_r≡0、φ_p=L_D−L_A 退化恒等成立。

**门判定**：
- **F0**：A2-R 先例维持（F1 显式判死门 + F3b ≥1pp 不放宽生效）。
- **F1 DEAD**：flip rates [0,0,0,0] + 零轮换 + keyH≈垄断先验——"Route exists? → Route changes temporally?" 两步全死。
- **F2 N/A**（无翻转对照列，池化 batch bootstrap 无 flip 臂）。
- **F3a**：flip 臂 NaN；stable 侧逐行 CE 变化均值 [−0.018, +0.006, +0.083, +0.014] nats（2→3 过渡 CE 抬升最重，与 drift 同向）——仅佐证通道，无门作用。
- **F3b FAIL**：route_pool 逐过渡全 0.0pp，均值 0.0pp vs 门 1.0pp。
- **F4 N/A**：3938 batch 无一 flip，point-biserial 无定义——前兆问题随 F1 死而消失。
- **F5 N/A**：oracle 人群 = 翻转行，pooled=0；Oracle-1 complementary=False、Oracle-2b 空曲线。
- heads 跨状态哈希漂移事件 = 0（F3 同头前提成立）；key 池广播后跨 client 全一致（5/5 状态）。

**审核锁死点闭合核验（§30.7 四项 + F5 净收益门，全部按预注册执行）**：
1. **batch exposure 单位**：F1 按 batch flip rate（0/3938）；F2/F3a 的 paired batch bootstrap 因无对照臂自动 N/A——统计单位修订正确落实且无损判决。
2. **组成敏感性 annex**：5 状态 × 75 个 (client, task) 组、circular shift 1..B−1 全部备选固定分区下多数槽改变 = **0**——"routes composition-insensitive；the batch majority is a stable population statistic on this run"（垄断使 batchwise 多数票退化为总体统计量，非 batch 伪象）。
3. **标量归因**：准确率计数池主归因完整且 interaction ≡ 0.000；向量幅值未做任何百分比归因。
4. **native gap 定位（F3b verdict 前置条件）**：75/75 键配 4 抽样——native diff max 0.84pp / mean 0.02pp；draw spread mean 1.06pp / max 3.06pp（native Dropout 抽样噪声主导，与 Track E 定性一致）；eval 抬升 mean +0.41pp；残差 73/75 严格落在 draw spread 内、其余 2 键超出量小于报告精度 0.01pp，**max_unexplained ≈ 0.0pp**（冒烟期 state0/client2/task0 的 0.45pp 疑点被 4 抽样的 spread 估计完全闭合）。结论："the gap is located, not a semantics divergence"——锁死点 4 解除，本 F3b verdict 作数。
5. **F5 净收益门**：无翻转人群 ⇒ G_flip/G_net 均为空集，不适用；A2-R 旧死法（常开软混合 −1.9pp）无需引入。

**判决（预注册冻结措辞，§30.4/30.7）**：

> **HYPOTHESIS_REJECTED: the discrete temporal route-switching hypothesis is REJECTED**（flip rates [0,0,0,0]；零多数槽轮换；keyH per state [0.0079, 0.0058, 0.0074, 0.0061, 0.0060] bits）。这是**拒绝我们自身的机制假说，不是探针失败**。

DTCPR **判死，不写训练臂**：fast/slow 双路由在部署路径上将恒一致、consensus gate 永不激活（结构 inert）。按最终裁定（2026-11-03），不再设计第三个结构候选。INR 复核按 §30.6 预注册豁免（判死不需双确认）。判级声明不变：单 run 单种子，产判据不产数字。

**机制注记（content channel 读法，审核阶梯指定路径）**：
- "Route fixed + features still drift" 实测成立：route 通道 0.00pp / 全部 batch 零变化，而漂移池 0.74pp **全部**由 contents 通道拥有（interaction ≡ 0）。
- **最大单次漂移（2→3 过渡 +2.646pp，95.12→92.47）100% content-owned**——本 run 最重的一次遗忘事件与路由零相关。
- 新机制叙事（**记录在案，不自动立项**，§30.4 冻结）：非仿射部署漂移由 **prompt 内容演化**主导，而非离散槽位切换——DHT 探针（§29）观察到的非仿射漂移由此获得更扎实的解释。
- 方向指向（非承诺）：representation-side 干预若未来立项，目标应为 **prompt 值的时序内容稳定性**（EMA/正则/锚定一类），而非任何路由结构；路由侧结构创新在本 run 部署路径上结构性 inert。§31 训练臂预注册按 F5 未过**不触发**。

### 30.9 Probe-F 封档记录（GPT 冻结三句式，2026-11-03）

> 1. **Probe-F rejects the discrete temporal route-switching hypothesis. Across all four deployment transitions, the selected prompt slot remained unchanged, with a persistent >99.9% monopoly of slot 6.**
> 2. **Despite invariant routing identities, substantial representation drift remained, and the 2×2 intervention assigned the observed deployment drift entirely to the prompt-content channel rather than route identity.**
> 3. **However, content evolution is not uniformly detrimental: the accuracy effect is strongly boundary-dependent and is concentrated at the 2→3 transition (+2.646pp recoverable degradation). Therefore, prompt-content evolution is promoted to a mechanism hypothesis, not yet to a method candidate.**

## 31. Probe-G：同槽 prompt 内容漂移机制解剖（`probe_prompt_content_drift.py`，预注册 2026-11-03）

> §30.8 判死后按 GPT 判后复核冻结方向设立 **Probe-G = same-slot prompt content drift**——只解剖 slot 6，不再碰任何 routing 侧（A2-R + Probe-F 已从两个独立方向证明 routing 不是干预面）。**本节不预注册任何结构**：Stable–Plastic Factorized Prompt（P_t = P^S_t + P^P_t，正交/子空间解耦）仅在 G3 GREEN 后才获得 §32 立项资格，且须先过 INR 机制复核 + 外审。探针零训练、只读保留 run、fail-closed、预注册先于数据。

### 31.0 机制衔接（判死链推进，非重立候选）

两次判死（DHT §29 / DTCPR §30）的净产出 = 机制定位上移一层：Probe-F 的 2×2 把全部漂移池归给 contents 通道（content 0.74pp vs route 0.00pp，interaction ≡ 0），且 route 侧结构性 inert（flip=0、keyH 0.0058–0.0079 bits、slot 6 垄断 99.94–99.96%、batch 组成不敏感）。问题已从"模型换了哪个 prompt"改为**"同一个 prompt 自己变了会怎样"**：

> R_{t−1}(x) = R_t(x), P_{6,t−1} ≠ P_{6,t} → representation drift（same prompt identity + evolving prompt content）

且 §30.8 已实测**内容演化非均匀有害**（0→1 受益 −0.481pp；2→3 集中损伤 +2.646pp），真正的科学问题冻结为：**When does prompt-content evolution become destructive?**——2→3 为主解剖对象，0→1 为良性对照。

### 31.1 代码事实审计（Probe-F 5 状态实测冻结 + 官方代码实读；探针以此为准）

1. **slot-6 prompt 值是每状态单一全局张量**：probe-F meta `prompt_value_hashes` 5/5 状态跨 client 单值（广播后逐位一致）；池形状 (100, 10, 768) = (pool_size, length, embed_dim) ⇒ P_{6,t} 良定义，swap/λ 臂不需要 per-client 分支。
2. **路由只读 prompt_key、不读 prompt 值**（`Global_Prompt.forward`：sim = l2norm(q)·l2norm(prompt_key)^T，`self.prompt[idx]` 只消费值）⇒ **只改 slot-6 值不改变路由身份**，swap/λ 臂与天然臂逐 batch 同 route（前提断言进 fail-closed）。
3. **同槽 token 插值语义合法**（GPT 冻结）：P_λ = (1−λ)·P_{6,t−1} + λ·P_{6,t} 是**同一 slot 的 prompt tensor** 凸混合；prompt 是 embedding 空间 token 前拼接（`cat([batched_prompt, x_embed])`），一一对应语义成立——与 DTCPR 跨槽混合的 off-manifold 问题不同类。
4. **Probe-F 实测基线（本探针对照锚点）**：slot-6 值逐边界 rel_norm 漂移 0→1 **0.9767** / 1→2 0.7323 / 2→3 0.6548 / 3→4 0.6645——**幅值不预测损伤**（0→1 漂移最大却是受益过渡）；天然臂特征漂移中位数 D_rel/D_cos：0→1 0.613/0.188、1→2 0.580/0.163、2→3 0.809/0.343、3→4 **0.851/0.349**（3→4 特征漂移最大却只伤 0.365pp）；探针口径天然臂 Acc 2→3 95.12→92.47；非 slot-6 路由行份额 ≈0.06%。⇒ **G1 预注册预期 = 幅值剂量响应 FAIL**（非探针缺陷，是 G2/G3 方向假说的立项动机，见 31.2）。
5. **heads 跨状态恒定**（probe-F 哈希漂移事件 = 0）⇒ 同头因果对比前提成立；本探针复刻断言。
6. **backbone 全冻结 + weight_decay=0** ⇒ 部署状态间除 prompt 池（值+key）外无其它可变前向输入，key 只影响选槽（恒 slot 6）⇒ **slot-6 prompt 值是该部署路径上唯一实质演化的表征输入**。

### 31.2 假设措辞（冻结，先于数据）

> **Hypothesis: same-slot prompt-content evolution causally owns the boundary-dependent deployment accuracy drift (concentrated at the 2→3 transition), and the damage is carried by the old-task-sensitive directions of the prompt update rather than its raw magnitude.**

判读措辞冻结：
- G1 是**可证伪臂**：预期措辞预注册为 "magnitude is not the dose"（§30.8 先验）；若幅值反而单调对应损伤 → 写 "magnitude-dominant (unexpected)" 转外审（机制比方向假说更简单）。
- **content evolution 晋升为 mechanism hypothesis，而非 method candidate**——即便 G3 GREEN，措辞也只写到 causal mechanism 为止。
- **G3 判负 ⇒ prompt content 不是干预面**（冻结措辞）："prompt-content evolution explains representation-state change but is not the recoverable-accuracy intervention surface; the structure path remains closed"——结构继续判死，不写 §32 臂。
- Stable–Plastic Factorized Prompt 仅作为 G3 GREEN 后的 §32 候选浮名，**本节不预注册其任何设计细节**。

### 31.3 探针设计（G0–G4）

**人群与臂覆盖**：每边界 (s→s+1) 在状态 s+1 上前向（F_{s+1} 其余状态不动）；人群 = 状态 s+1 全部 (client, task) test 行（旧任务 j ≤ s + 当前任务 j = s+1——G4 塑性侧需要当前任务行）；**臂覆盖 = slot-6 多数票 batch**（垄断前提，fail-closed），非 slot-6 batch（实测 ≈0.06% 行）不进臂、份额报告、天然臂照常含全部行；跨状态 canon join 沿用 §29.1c/§30 同款（(client, task, rel_idx)，fail-closed 单调断言）。

| 臂 | slot-6 token | 语义 |
|---|---|---|
| natural (λ=1) | P_{6,s+1} | 部署真相（G1 特征漂移跨状态两端；0→1 时 D 侧复用 s 状态天然臂） |
| swap (λ=0) | P_{6,s} | **G3 因果臂**：same route, old content |
| λ∈{0.25, 0.5, 0.75} | (1−λ)P_{6,s} + λP_{6,s+1} | **G4 剂量臂**（同槽插值，语义合法） |

**G0 前提门（fail-closed，任一违反即拒）**：top_k=1 + batchwise_prompt=True + use_prompt_mask=False（部署路径）；每状态 batch 多数票众数槽 == slot 6 且非 slot-6 batch 占比 ≤1%；垄断槽跨状态恒定；prompt 值跨 client 单值；heads 跨状态哈希恒定（违反则 G3 标记 CONTAMINATED，不作数）。

**G1 prompt 自身漂移（可证伪臂，无门）**：逐边界 rel_norm = ‖ΔP_6‖₂/(‖P_{6,s}‖₂+ε)、cos drift = 1−cos(P_{6,s}, P_{6,s+1})，与天然臂跨状态特征漂移中位数 D_rel/D_cos 并列成表。预期 "magnitude is not the dose"。

**G2 old-sensitive 方向分解（描述性，佐证）**：把 ΔP_6 分解为 ΔP_∥（会改变历史任务损失的方向）+ ΔP_⊥——真正要修的不是 |ΔP| 而是 |Π_old-sensitive ΔP|。估计器（冻结）：在状态 s、head.eval() 确定性前向下，对旧任务行 masked CE 逐行求 g_i = ∂CE_i/∂P_{6,s}（autograd 仅用于分析、零训练合规、不写任何参数）；行梯度集合 G（种子派生稳定、(client, task) 分层、≤256 行、只取 slot-6 batch 行）；SVD 取累积能量 ≥90% 的前 k（k≤128）方向张成 old-sensitive 子空间；报 **frac_old_sensitive = ‖Π_G ΔP‖²/‖ΔP‖²**、一阶预测 mean(g_i^T ΔP)（nats）与同子样本行实测 ΔCE 对照 + 逐行 Pearson。判读：2→3 的 frac/pred 应高于 0→1（方向剂量假说）；无硬门，喂 G3 判读。

**G3 prompt-swap causal oracle（主门，GPT 冻结）**：状态 s+1 前向、同 route、同 heads、同人群，唯一差异 = slot-6 值换 P_{6,s}。
recovery = Acc(swap, 旧任务行) − Acc(natural, 旧任务行)（pp，配对逐行差）。
**门（冻结）：2→3 过渡 recovery ≥ +0.5pp 且 paired batch bootstrap 95% CI 下界 > 0**（重抽样单位 = batch，B=1000，稳定种子）→ G3 PASS（因果证据：new prompt content → historical performance loss，而非相关）；recovery ≥ 门但 CI 含 0 → YELLOW_REVIEW；否则 FAIL。并报 recovery 占该边界 drift 池比例（2→3 池 = +2.646pp，预期显著占比）及其余过渡 recovery（0→1 预期 ≈0 或负 = 良性/受益演化的对照证据）。

**G4 同槽 λ 插值（剂量-响应佐证）**：λ ∈ {0, 0.25, 0.5, 0.75, 1}；旧任务 + 当前任务两条 acc(λ) 曲线。判读（冻结）：2→3 上 Spearman(acc_old, λ) < 0 且 Spearman(acc_cur, λ) > 0 → G4 PASS（stability–plasticity trajectory 实测）；任一曲线全程极差 <0.05pp 记 FLAT。G3 PASS + G4 PASS/FLAT → 机制确认；G3 PASS + G4 FAIL → PASS_REVIEW 转外审；**G4 单独不救 G3 FAIL**。

**Annex（informative）**：swap 臂特征恢复（旧行 median ‖z_swap−z_s‖/‖z_s‖ vs 天然 D_rel——swap 若同时恢复特征坐标则因果链完整）；native draw annex 沿用 §30.7-4 同款（probe-vs-CSV gap 定位，容差 3pp，`--native_draws`，正式跑 4）。

### 31.4 判决树（冻结）

```
G0  前提门 fail → REFUSED（fail-closed，无判决）
G3  2→3 recovery ≥ +0.5pp 且 CI_lo > 0？
    是 → G4 佐证？
        单调（old 随 λ↓、cur 随 λ↑）或 FLAT → CONTENT_MECHANISM_CONFIRMED：
            same-slot prompt-content evolution 因果拥有 2→3 集中损伤；
            晋升 mechanism hypothesis（冻结措辞，仍非 method candidate）；
            Stable–Plastic Factorized Prompt 获得 §32 立项资格——
            前置 = INR 机制复核 + 外审，本探针不写臂。
        G4 FAIL → CONTENT_CONFIRMED_REVIEW（因果在、剂量-响应不单调）→ 转外审
    否（<门 或 CI 含 0）→ CONTENT_NOT_INTERVENTION_SURFACE：
        prompt-content evolution explains representation-state change but is
        not the recoverable-accuracy intervention surface——结构路径继续关闭，
        不写 §32 臂。
```

### 31.5 fail-closed 清单

1. 部署路径前提（use_prompt_mask=True / top_k≠1 / batchwise_prompt=False）→ 拒；
2. 垄断前提：任一状态众数槽 ≠ 6、非 slot-6 batch 占比 >1%、或垄断槽跨状态变化 → 拒（G 系列语义只对该部署路径定义）；
3. prompt 值跨 client 不一致 → 拒（P_{6,t} 全局张量前提破坏）；
4. heads 跨状态哈希漂移 → G3 标记 CONTAMINATED，不作数；
5. canon join 缺行 / 不单调 → 拒（沿用 §30）；
6. G2 梯度只走分析路径（torch.autograd.grad，不写 .grad、无 optimizer、无任何参数更新）；梯度 batch 若多数槽 ≠ 6 → 该批子样本行剔除并报告数量；
7. 缓存指纹含 PROBE_VERSION=contentdrift-v1 + run/checkpoint/batch/monopoly_slot/lambdas/g2_rows，不匹配即拒；缓存目录独立 `diagnostics_output/contentdrift_cifar/`（§29 串用教训）；
8. `--device cuda` 必须（Tail_Anchor 硬编码 .to('cuda')）。

### 31.6 命令（GPT 审核通过后执行；GPU 轻前向 ≈5×天然臂 + 256 行梯度/边界，预计 25–45 分钟）

```bash
# cifar 主判读场（保留 run 5 边界 checkpoint）
python diagnostics/probe_prompt_content_drift.py \
    --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
    --data_path ./local_datasets --device cuda \
    --cache_dir diagnostics_output/contentdrift_cifar \
    --native_draws 4 \
    --out_csv diagnostics_output/contentdrift_cifar.csv \
    --out_rows_csv diagnostics_output/contentdrift_cifar_rows.csv \
    --out_json diagnostics_output/contentdrift_cifar.summary.json
```

产物：`contentdrift_cifar.csv`（过渡×client 聚合 + G1–G4 读数）+ `contentdrift_cifar_rows.csv`（per-row：d_rel/d_cos、correct@{tm1, t, swap, λ}、特征恢复）+ `contentdrift_cifar.summary.json`（G0–G4 门 + verdict + annex）。判级声明：探针级诊断，单 run 单种子，产判据不产数字；INR 复核仅在 G3 GREEN 且拟立项 §32 时才需要（判负不需双确认，延续 §30 惯例）。

### 31.7 工程冒烟记录（2026-11-03）

用户指令跳过独立冒烟、直接正式跑（GPU 共享时段）。正式跑中发现并修复一处工程 bug（不影响任何门读数）：`extract_state` 末状态（`is_last_state=True`）跳过 G2 时走的兜底 dict 键名少 `_rows` 后缀（`n_dropped_batches` vs `n_dropped_batches_rows`），致 state 4 保存时 KeyError；修复后原命令重跑，state 0–3 从缓存秒级加载（14:35–15:17 已落盘），state 4 重抽取，全量完成。此前另有一处冒烟期已修的 `retain_graph=True`（G2 批内逐行 backward 二次通过图崩溃）。两处均为分析/序列化路径，臂前向、门逻辑、统计口径未动。

### 31.8 正式判决（2026-11-03，5 状态全量，probe `contentdrift-v1`，`--native_draws 4`）

**判决：CONTENT_CONFIRMED_REVIEW（预注册判决树 G3 PASS + G4 FAIL 分支）——same-slot prompt-content evolution 因果拥有跨边界部署精度漂移，但剂量-响应不单调，转外审后才谈 §32 立项。**

**G0 前提门 OK**：5/5 状态众数槽 = 6、非 slot-6 batch 占比 0.0%、prompt 值跨 client 单值（哈希逐状态互异：0fdd1457/5b497b8b/9f1f2033/18075eda/d07c1e4a，证实 P_{6,t} 逐状态全局演化）、heads 跨状态哈希漂移事件 0、行数 3324/6487/9222/12328/14992 与 probe-F 逐位一致。

**主结果（G3 swap causal oracle，主门 PASS）**：逐边界 swap recovery（pp，paired batch bootstrap B=1000）：

| Transition | drift 池 | recovery | CI95 | 占池比 | swap Acc（=t−1 天然 Acc，逐位复原） |
|---|---:|---:|---|---:|---|
| 0→1 | −0.481 | −0.481 | [−1.054, +0.090] | 100% | 96.15 |
| 1→2 | +0.432 | **+0.432** | [+0.031, +0.832] | 100% | 96.65 |
| 2→3 | **+2.646** | **+2.646** | **[+2.126, +3.150]** | **100%** | **95.12** |
| 3→4 | +0.365 | +0.365 | [−0.113, +0.836] | 100% | 93.03 |

- 2→3 过 +0.5pp 门且 CI 下界远大于 0 → **因果确认**（冻结措辞：new prompt content → historical performance loss，非相关）。
- **每个边界 swap 都复原该边界 100% 漂移池**，且 swap 臂特征中位 rel drift = 0.0（旧行 z_swap ≡ z_{t−1} 逐位复原；frozen backbone + 同 route + 同 heads 下的解析必然，实测坐实）⇒ **slot-6 prompt 值是该部署路径上跨状态精度漂移的唯一因果载体**——损伤（2→3）与收益（0→1）都由它承载，方向是边界依赖的（§30 结论的强化版）。

**G1（幅值，可证伪臂，按预期 FAIL）**：P_6 逐边界 rel_norm 0.9767/0.7323/0.6548/0.6645、cos drift 0.6715/0.2568/0.2227/0.2318；幅值-损伤 Spearman = −0.2 → **"magnitude is not the dose" 确认**（0→1 漂移近 100% 范数却是受益过渡；预注册措辞照记）。顺带关键读数：**每边界 prompt 值近乎全量重写（0→1 cos drift 0.67）而路由从不翻转**——key 演化与值演化彻底解耦，值演化才是部署实质。

**G2（old-sensitive 方向分解，描述性预期不成立）**：frac_old_sensitive 逐边界 0.1389/0.0208/**0.0141**/0.1061——2→3 反而**最低**；一阶预测 mean(g_i^TΔP) 与实测 ΔCE 对照（2→3：预测 −0.0135 nats vs 实测 +0.1369 nats，逐行 Pearson 0.19）⇒ **一阶梯度子空间投影也不定位损伤**。损伤方向不可由线性 old-sensitive 投影刻画（弱化 Stable–Plastic 的朴素线性正交分解版本，转外审的理由之一）。

**G4（λ 插值，2→3 剂量臂 FAIL=非单调）**：acc_old(λ=0→1) = 95.12 / **95.99** / 95.41 / 93.76 / 92.47（Spearman −0.7）；acc_cur(λ) = 95.94 / **96.56** / 96.46 / 95.81 / 94.69（Spearman **同为 −0.7**）——两条曲线都在 λ=0.25 处峰值：**凸混合 prompt 在旧任务与当前任务两侧同时优于两个部署端点**。损伤集中在 λ∈[0.5, 1] 段，不是平滑剂量-响应。其余过渡（对照）：0→1 两条曲线随 λ 单调上升（Spearman +0.82/+0.90，新内容对旧任务有益）；1→2 双 FLAT（±0.2 内）；3→4 cur 单调 +0.90。

**Annex（informative）**：native gap 定位复刻 §30 结论——75 组对照 max 未解释残差 0.0pp（97.3% 组落 Dropout draw 传播范围内），probe-vs-CSV 差异是随机性非语义分歧；头身份恒定。

**机制注记（喂外审）**：因果链已完整闭合——same route + evolving same-slot content → 全部特征漂移 → 全部边界精度漂移（含损伤与收益），swap 完全可逆。但**"何时成为破坏性"仍未参数化**：幅值不预测（G1）、一阶敏感子空间不预测（G2）、λ 剂量非单调且混合点优于端点（G4）。G4 内点占优（λ=0.25 双侧同时最好）提示损伤不是"新内容剂量"的单调函数，更像新内容某分段（λ≥0.5）与旧判别几何的非线性相互作用。**content evolution 维持 mechanism hypothesis 措辞、不升 method candidate**；Stable–Plastic Factorized Prompt 不因 G3 PASS 自动立项——其线性正交分解前提已被 G2/G4 两侧削弱，是否以及何种形式的 §32 臂交外审裁决。

**封档记录（GPT 三句式，2026-11-03）**：

> **Probe-G causally confirms same-slot prompt-content evolution as the sole carrier of the cross-state deployment accuracy drift: the swap oracle recovers 100% of the drift pool at every boundary (2→3: +2.646pp, CI95 [+2.13, +3.15]) with bit-identical feature restoration, establishing new prompt content as the cause of historical performance loss.**

> **However, the damage is not yet parametrized: prompt drift magnitude does not predict it (Spearman −0.2), the first-order old-sensitive gradient subspace does not localize it (frac at 2→3 is the lowest, 0.0141), and the lambda interpolation is non-monotone with an interior blend (λ=0.25) dominating both deployment endpoints on BOTH old and current tasks.**

> **The pre-registered verdict is CONTENT_CONFIRMED_REVIEW: the causal mechanism stands, the dose-response structure is falsified, and any Sec.32 candidate (Stable–Plastic Factorized Prompt) — whose naive linear-orthogonal premise is now weakened on both the G2 and G4 sides — must pass external review before candidacy.**

### 31.9 判后复核修订（GPT，2026-11-03；§31.8 原文全文保留作审计痕迹，本节只做修订记录）

**判决标签改写（冻结）**：`CONTENT_CONFIRMED_REVIEW` → **`CONTENT-CAUSAL / DIRECTION-MECHANISM-UNRESOLVED / OVERSHOOT-HYPOTHESIS-PROMOTED`**。三层含义：

1. **第一层确认**：same-slot prompt content → representation state 因果链成立；2→3 过渡它承担 +2.646pp 历史性能损失且 CI 离零（[+2.126, +3.150]）。**G3 措辞收紧**：swap 恢复 100% 漂移池在 backbone/head/route 恒定条件下本质是恒等干预的预期结果（D_rel^swap = 0 逐位复原），不包装为独立发现；科学价值在于**性能差随 prompt swap 被恢复且 CI 明确离零**。
2. **第二层否定**：大范数（G1）与 old-sensitive 线性子空间（G2）都不能解释"为何某些更新有害"——2→3 的 frac_old_sensitive = 0.0141 反而是四过渡最低（0→1 = 0.1389 最高），一阶预测 ΔCE −0.0135 nats 与实测 +0.1369 nats 连方向都不一致。**"carried by old-task-sensitive directions" 从 hypothesis 措辞中删除**。修订后 hypothesis（冻结）：
   > **Same-slot prompt-content evolution causally controls the deployment-state representation change, while raw update magnitude does not explain its boundary-dependent accuracy effect.**
   （有害方向目前未知。）
3. **第三层（G4 "FAIL" 中的主信号，新机制候选）**：**interior prompt states repeatedly outperform the learned endpoint**。逐过渡当前任务 test 最优 λ（probe-G 粗网格，test-oracle 口径）：

   | Transition | 当前任务最优 λ | 相对 λ=1 old gain | 相对 λ=1 cur gain |
   |---|---:|---:|---:|
   | 0→1 | 1.00 | 0 | 0 |
   | 1→2 | **0.50** | **+0.84pp** | **+0.74pp** |
   | 2→3 | **0.25** | **+3.52pp** | **+1.87pp** |
   | 3→4 | **0.75** | **+0.69pp** | **+0.79pp** |

   后三个过渡中，仅按当前任务最优点选出的中间 prompt **双侧同时改善**（均值 old +1.26pp / cur +0.85pp）⇒ 新 prompt 不是"为当前任务牺牲旧任务"，连当前任务都次优于中间态 ⇒ 机制从 stability–plasticity conflict 改写为 **the learned prompt endpoint is temporally miscalibrated**（temporal prompt update overshoot）；最优步长逐边界不同（1 / 0.5 / 0.25 / 0.75）⇒ 需 **state-adaptive temporal step calibration**，而非固定 EMA。

**处置决定（冻结）**：
- **§32 Stable–Plastic Factorized Prompt 不立项**（G2 未提供其线性正交分解前提的任何机制支撑）。
- 下一轮 = **§32 Probe-H: Temporal Prompt Step Calibration**——唯一核心问题：**Can the harmful endpoint be avoided using current-task information alone?**
- Probe-H 过门后的方法候选名（浮名，仍须 INR 机制复核 + 外审才立项）：**CTRP (Calibrated Temporal Residual Prompt)**——P_t^deploy = P_{t−1} + α_t·ΔP_t，0 ≤ α_t ≤ 1，α_t 由当前任务训练信号自适应确定（不依赖 replay/old validation、非固定超参）。

## 32. Probe-H：temporal prompt step calibration（`probe_prompt_step_calibration.py`，预注册 2026-11-03）

> §31.9 判后复核冻结方向设立。探针零训练、只读保留 run、fail-closed、预注册先于数据。**本节不预注册任何结构/训练臂**：CTRP 仅在 Probe-H 过主门后获得 §33 立项资格，且须先过 INR 机制复核 + 外审。

### 32.0 机制衔接

§31 判决链已给出：prompt content 是跨状态漂移的唯一因果载体（G3）、幅值与 old-sensitive 线性子空间都不定位损伤（G1/G2）、同槽内点 prompt 反复双侧优于学习端点（G4，§31.9 表）。剩下的**可部署性问题只有一个**：G4 的最优 λ 是 test-oracle 选的（看了历史 test，不能部署）。若**完全不碰旧任务数据**、仅凭当前任务信号就能找到 α < 1 的步长且双侧不劣于端点，则 overshoot 是 current-task-signal 可修复的部署缺陷 ⇒ CTRP 立项依据直接建立在实测的 endpoint overshoot 上，无需假设任何 G2 未找到的"旧任务敏感子空间"。若合法选择器找不到（而 oracle 找得到），则 overshoot 记录为 selector-blind、同样不可部署。

### 32.1 代码事实审计（probe-G 5 状态实测 + 官方代码实读冻结；探针以此为准）

1. **slot-6 prompt 值每状态单一全局张量**（跨 client 逐位一致，哈希 0fdd1457/5b497b8b/9f1f2033/18075eda/d07c1e4a 逐状态互异）⇒ P_{6,t} 良定义，λ 臂无需 per-client 分支；池形状 (100, 10, 768)。
2. **路由只读 prompt_key 不读值** ⇒ 只改 slot-6 值不改变路由身份（逐 batch 断言进 fail-closed）；非 slot-6 路由行份额 ≈0.06%。
3. **同槽 token 插值语义合法**（§31.1-3 同款：embedding 空间 token 前拼接的凸混合）。
4. **heads 跨状态哈希恒定**（probe-G 漂移事件 0）⇒ 同头对比前提成立；backbone 全冻结 + weight_decay=0 ⇒ slot-6 值是该部署路径唯一实质演化的表征输入。
5. **inner train 划分**：`Client_DF.get_data` 用 `random_split(trainset, [70%, 30%])` 落 `data_split_indices[task_id]['train'/'test']`（Client_DF.py L147-155）——划分由 bootstrap_server 复刻的训练 RNG 流锁定；**Probe-H 选择行 = 当前任务 `data_split_indices[t]['train']`**（与训练同 manifest，无二次划分随机性；test 行读数人群沿用 `['test']`，与 probe-G 逐位同源，行数应复现 3324/6487/9222/12328/14992）。
6. **probe-vs-native gap 已在 probe-G 于本 run 定位**（75/75 组、max_unexplained ≈ 0.0pp、Dropout draw 噪声主导）⇒ Probe-H 沿用同一确定性前向协议（head.eval() + 固定 batch 8 + 固定顺序；门全部作用于跨臂差值、偏置消去），**native-draw annex 不重复执行**（GPU 预算；声明沿用 §31.3）。
7. **探针口径天然臂应逐位复现 probe-G 读数**（同 run/同 ckpt/同协议/同种子派生）：2→3 天然 95.12→92.47、λ 粗网格臂值（0.25 处 old 95.99 / cur 96.56）——不一致即工程 bug，fail-closed 对照（分析层对账，容差 0.0：确定性前向必须逐位）。

### 32.2 假设措辞（冻结，先于数据）

> **Hypothesis: the learned prompt endpoint is temporally miscalibrated — a current-task-signal-calibrated interior step α ∈ [0,1) avoids the endpoint's harm on BOTH the old and the current tasks, without accessing any old-task data.**

判读措辞冻结：
- 过门只把 overshoot 晋升为 **current-task-signal-exploitable mechanism**；CTRP 仍是浮名，本探针不写任何训练臂。
- 判负措辞（冻结）：**OVERSHOOT_NOT_EXPLOITABLE** —— "the interior gain is not locatable by current-task signals (or does not exist at the fine grid); the overshoot is recorded but not deployable"。

### 32.3 探针设计（H0 前提门 + H1 细网格扫描与合法选择器）

**臂与人群**：每过渡 (s→s+1) 在状态 s+1 前向（其余一切不动）；P_λ = (1−λ)·P_{6,s} + λ·P_{6,s+1}，**λ ∈ {0, 0.05, …, 1}（21 点细网格）**；臂覆盖 = slot-6 多数票 batch（§31 垄断前提）；读数人群 = 状态 s+1 全部 (client, task ≤ s+1) test 行（acc_old 取 task ≤ s，acc_cur 取 task = s+1，均限臂覆盖行）。

**选择信号（部署合法、零旧任务信息）**：每 client 从当前任务 `data_split_indices[t]['train']` 作**按类分层、派生种子稳定子采样**，上限 512 行/client（`--select_rows_per_client`，纯 GPU 预算控制；统计上 2560 行池化 CE 均值已远超比较 21 个 λ 所需功效），再按类分层拆 **20% val / 80% train**（`--val_frac`；类内行数 ≥2 时 val ≥1 行，行数 =1 时留 train）。选择行与非 slot-6 batch 剔除计数（占比 >1% 拒）。信号前向与读数同协议（head.eval() 确定性、掩码后 CE = logsumexp − logit[y]）。

**选择器（冻结）**：
- **λ̂_val-current（主）** = argmin 跨 client 池化当前任务 val CE；
- **λ̂_train-current（次级佐证）** = argmin 池化 train CE；
- 并列取最小 λ；对照（非法、仅报告）：**λ\*_current** = argmax 当前任务 test acc。

**主门（GPT 冻结，2→3 主判场，primary selector = λ̂_val-current）**：

> **λ̂ < 1 且 Acc_cur(λ̂) ≥ Acc_cur(λ=1) 且 Acc_old(λ̂) > Acc_old(λ=1)**，且 old gain 配 paired batch bootstrap 95% CI（B=1000，单位 = batch，重抽样覆盖旧任务行）**下界 > 0**。

其余过渡全部同表报告（逐过渡 λ̂/λ\* 与双侧增益 = state-adaptive 校准必要性的直接证据），门只在 2→3 上作用。

### 32.4 判决树（冻结）

```
H0  前提门 fail（部署路径 flags / 垄断 / prompt 值跨 client 单值 / heads 恒定 /
    canon join / 选择行垄断校验，任一违反）→ REFUSED（fail-closed，无判决）
主门（2→3，λ̂ = λ̂_val-current）：
    λ̂ < 1 且 Acc_cur(λ̂) ≥ Acc_cur(λ=1) 且 old_gain > 0 ？
        是 → CI95_lo(old_gain) > 0 ？
            是 → OVERSHOOT_EXPLOITABLE_CURRENT_ONLY：
                temporal overshoot 可仅凭当前任务信号修复；
                CTRP 获得 §33 立项资格（仍须 INR 机制复核 + 外审；本探针不写臂）
            否 → YELLOW_REVIEW（方向成立但显著性不足，转外审）
        否 → OVERSHOOT_NOT_EXPLOITABLE
            annex 注记二分（informative）：
            · selector-blind：test-oracle λ* < 1 且 λ* 双侧优于 λ=1
              —— 内点增益存在但合法选择器定位不到；
            · no-interior-gain：oracle 也取 λ* = 1
              —— 细网格下端点即最优，§31.9 表在 2→3 之外的推广不成立。
次级（报告不作门）：train-current selector 同门读数；mean old/cur gain
    @λ̂_val 跨过渡均值；λ̂ 逐过渡是否相异（state-adaptive 证据）。
```

### 32.5 fail-closed 清单

1. 部署路径前提（use_prompt_mask=True / top_k≠1 / batchwise_prompt=False）→ 拒；
2. 垄断前提：任一状态 batch 众数槽 ≠ 6 或非 slot-6 batch 占比 >1% → 拒；
3. **选择行侧垄断校验**：非 slot-6 batch 选择行占比 >1% → 拒（计数入报告）；选择行为 0（状态 ≥1）→ 拒；
4. prompt 值跨 client 不一致 → 拒；
5. heads 跨状态哈希漂移 → 主门标记 CONTAMINATED 不作数；
6. canon join 缺行/不单调 → 拒（§29.1c/§30/§31 同款）；
7. 零训练合规：本探针无梯度臂、无 autograd、无 optimizer、无任何参数更新；
8. 缓存指纹含 PROBE_VERSION=stepcalib-v1 + run/checkpoint/batch/monopoly_slot/λ 网格(21)/select_rows_per_client/val_frac，不匹配即拒；**缓存目录独立 `diagnostics_output/stepcalib_cifar/`**（§29 教训；不复用也不读 contentdrift_cifar）；
9. `--device cuda` 必须（Tail_Anchor 硬编码 .to('cuda')）。

### 32.6 命令（GPT 审核通过后执行；21 臂 + 选择行 ≈ probe-G 前向量的 2–2.5×，预计 1.5–2.5 小时；逐状态缓存、可断点续跑）

```bash
# cifar 主判读场（保留 run 5 边界 checkpoint）
python diagnostics/probe_prompt_step_calibration.py \
    --run_dir output/cifar100/fedta/cifar_a5v2_M2_feat_full/seed_42 \
    --data_path ./local_datasets --device cuda \
    --cache_dir diagnostics_output/stepcalib_cifar \
    --out_csv diagnostics_output/stepcalib_cifar.csv \
    --out_rows_csv diagnostics_output/stepcalib_cifar_rows.csv \
    --out_json diagnostics_output/stepcalib_cifar.summary.json
```

产物：`stepcalib_cifar.csv`（过渡级聚合 + 逐客户端 old gain + 21 点 acc_old/acc_cur 曲线）+ `stepcalib_cifar_rows.csv`（per-row 分析即用表：覆盖行 correct@{λ 21 点}，old/current 两人口）+ `stepcalib_cifar.summary.json`（H0 前提 + 选择器读数 + 主门判决 + oracle annex + probe-G 对照对账）。判级声明：探针级诊断，单 run 单种子，产判据不产数字；INR 复核仅在 Probe-H 过门且拟立项 §33 时需要（判负不需双确认，延续 §30/§31 惯例）。

### 32.7 工程记录（正式跑 2026-11-03）

- **执行**：用户终端跑正式 5 状态全量（§32.6 命令原样），17:07–19:09 约 2 小时。GPU 与 8 进程共享（用户自有 `cifar_a5v2_M2pb_full` / `inr_a5v2_M2pb_p40` 两训练 + 其它租户 6 任务，100% util），逐状态耗时 ~2 / 21 / 26 / 32 / 40 分钟（行数 3324/6487/9222/12328/14992 + 21 臂 + 选择行 2561/2559/2562/2562）。
- **零 bug**：无 crash、无 fail-closed 触发；H0 全过（nonslot batch frac = 0.0、选择行 drop frac = 0.0/剔除 0 行、prompt 值跨 client 单值、heads 跨状态恒定）；探针无运行期修补，§31.7 两个已知坑（retain_graph / 兜底键名）因臂结构不同未复现。
- **跨探针对账**：probe-G 连续性检查 8/8 天然读数逐位一致（`contentdrift_cifar.csv` 的 acc_nat_tm1/acc_nat_t，全部 bit-identical）——确定性前向协议在两探针间可复现，分析层对账通过。
- **缓存**：`stepcalib_cifar/` 5 状态 180MB，逐状态落盘正常；正式跑一次通过，未用到断点续跑。
- **冒烟说明**：我侧沙箱拦截 CUDA 设备访问（cudaGetDeviceCount error 304，两次尝试含非沙箱模式均被 /proc 限制），冒烟改由正式跑直接覆盖验证——正式跑零 bug 反证代码路径完好。
- 产物：`stepcalib_cifar.csv`（4 过渡行 + 12 client 行 + 42 曲线列）、`stepcalib_cifar_rows.csv`（覆盖行 × 21 点 correct，old/current 两人口）、`stepcalib_cifar.summary.json`（判决 + 选择器读数 + CE 曲线 + oracle annex）。

### 32.8 正式判决（2026-11-03，判级：探针级诊断，单 run 单种子，产判据不产数字）

**`OVERSHOOT_EXPLOITABLE_CURRENT_ONLY`**（主门 PASS，且以压倒性余量通过）。

主门（2→3，primary = λ̂_val-current）：

| 读数 | λ̂ | Acc_old | old gain | CI95 (paired batch bootstrap) | Acc_cur gain |
|---|---:|---:|---:|---|---:|
| **λ̂_val（主）** | **0.30** | 95.98 | **+3.51pp** | **[+3.02, +4.00]** | **+1.67pp** |
| λ̂_train（次级） | 0.40 | 95.73 | +3.26pp | [+2.79, +3.74] | +1.90pp |
| λ\* test-oracle（非法对照） | 0.35 | 96.02 | +3.55pp | [+3.07, +4.05] | +1.96pp |

两个合法选择器几乎精确命中 test-oracle 区域（0.3/0.4 vs 0.35），且 **train 与 val 双选择器独立同向**。

逐过渡全表（全部 CI 下界 > 0）：

| 过渡 | λ̂_val | old gain | cur gain | λ̂_train | old / cur gain |
|---|---:|---:|---:|---:|---|
| 0→1 | 0.85 | +0.42pp [0.06, 0.81] | +0.10pp | 0.95 | +0.36 / +0.16 |
| 1→2 | 0.95 | +0.19pp [0.06, 0.32] | +0.11pp | 0.65 | +0.84 / +0.52 |
| 2→3 | **0.30** | **+3.51pp [3.02, 4.00]** | +1.67pp | 0.40 | +3.26 / +1.90 |
| 3→4 | 0.70 | +0.54pp [0.20, 0.90] | +0.98pp | 0.65 | +0.91 / +1.01 |

@λ̂_val 跨过渡均值：old +1.165pp / cur +0.715pp。**λ̂ 逐过渡互异（0.85/0.95/0.30/0.70）⇒ state-adaptive 校准必要性再次确证**，固定 EMA 不可替代。

细网格对 probe-G 粗网格的两点修正/升级（informative annex）：
1. **0→1 的"λ\*=1"是网格分辨率伪影**——细网格下 0→1 同样内点占优（acc_old 在 λ=0.8 达 97.20 vs 端点 96.63，oracle λ\*=0.9）。内点占优从"后三过渡"升级为**四过渡全部成立**。
2. 2→3 的 oracle 最优从粗网格 0.25 修正为 0.35，且 acc_old 与 acc_cur 的内点峰同位（都在 0.35 附近）——overshoot 是两侧共认的次优点，不存在 stability–plasticity 交换。

**机制结论（§32.2 hypothesis 确认）**：the learned prompt endpoint is temporally miscalibrated，且该 miscalibration **完全可由当前任务信号定位与修复**（val CE argmin 即可，零旧任务信息、零 replay、零 test-oracle）。联邦持续学习把 prompt 更新推到了一个连当前任务都次优的 temporal endpoint；这不是 stability–plasticity 冲突，而是**步长校准缺陷**。

**CTRP 立项资格（§33）**：Calibrated Temporal Residual Prompt——`P_deploy = P_{t−1} + α_t·ΔP_t`，α_t 由当前任务信号（当前任务 val/train CE 在 {0, 0.05, …, 1} 网格 argmin）自适应确定。前置条件不变：**须先过 INR 机制复核（同一探针、ImageNet-R p40 保留 run）+ 外审**；本探针不写任何训练臂。

**GPT 三句式封档记录（frozen for external review）**：

> **The pre-registered main gate at the 2→3 transition PASSES with decisive margin: the val-current selector (current-task validation CE only, zero old-task information) picks λ̂=0.30<1 with Acc_old +3.51pp over the natural endpoint (paired batch-bootstrap 95% CI [+3.02, +4.00], lower bound ≫ 0) and Acc_cur +1.67pp, and the independent train-current selector concurs (λ̂=0.40, +3.26pp CI [+2.79, +3.74], +1.90pp) — both legal selectors land on the test-oracle region (λ\*=0.35).**
>
> **The effect is boundary-general and state-adaptive: at ALL four boundaries the current-task selectors find interior steps that improve BOTH populations with CI lower bounds > 0 (λ̂_val = 0.85/0.95/0.30/0.70; mean old +1.165pp / cur +0.715pp), the optimal step differs per boundary (fixed-EMA insufficient), and the fine grid upgrades Probe-G's coarse-grid "λ\*=1 at 0→1" to a grid-resolution artifact (acc_old 97.20 @ λ=0.8 vs 96.63 endpoint) — interior dominance holds at every transition.**
>
> **The pre-registered verdict is OVERSHOOT_EXPLOITABLE_CURRENT_ONLY: the temporal prompt-update overshoot is a deployment-repairable, current-task-signal-locatable step-calibration defect rather than a stability–plasticity conflict, and CTRP (Calibrated Temporal Residual Prompt: P_deploy = P_{t−1} + α_t·ΔP_t, α_t calibrated on current-task signals alone) earns Sec.33 candidacy, conditional on INR mechanism replication + external review; no training arm is pre-registered by this probe.**

## 32.9 INR 机制复核（Probe-H replicate，预注册 2026-11-03，GPT 外审批板冻结；先冻结判据后运行，不看结果调门）

**身份**：CTRP §33 立项前置条件之一（§32.8 冻结）。CIFAR-100 的 2→3 主判场是从已观察到的性能损失选出来的；INR 是**跨数据集外部验证**，无先验主判场——若人为挑一个"类似 2→3"的边界即引入选择偏差。因此 INR 采用 **All eligible adjacent transitions**，主统计量为各过渡**等权**的历史任务准确率改善均值：

G_old = (1/T) Σ_{t=1..T} [ Acc_old,t(α̂_val,t) − Acc_old,t(1) ]

每个 transition 单独报告，再算总体；**绝不把所有过渡的样本并成一个超大独立样本池**（CI 用层次 bootstrap 实现：每个 rep 在每个 transition 内独立重抽 batch、rep 内等权平均）。

### 32.9.1 冻结参数（GPT 拍板 + 一处工程修订）

| 参数 | 冻结值 |
|---|---|
| run_dir | `output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42` |
| cache_dir | `diagnostics_output/stepcalib_inr_seed42` |
| 主判场 | **全部相邻 transition（T=4），等权聚合**，不指定单一边界 |
| 门模式 | `--gate_mode all` |
| 其余 | 沿用 §32 原始门槛，**不因外部验证降低标准** |

**与 GPT 拍板的唯一偏离（工程修订，外审可否决）**：cache_dir 由拍板的 `contentdrift_inr_seed42` 改为 **`stepcalib_inr_seed42`**——`contentdrift_*` 是 probe-G 的命名空间，若未来 probe-G 补跑 INR 会用同名；虽然缓存指纹会 fail-closed 拒绝跨探针串用，但按 §29 教训不应主动制造同名冲突。探针版本 `stepcalib-v1`（同一探针的新运行模式，非新探针）。

### 32.9.2 判决条件冻结表

| 检查项 | 预注册要求 |
|---|---|
| G0（H0 前提门） | 路由稳定（部署路径 top_k=1/batchwise/无 task mask）、prompt 值跨客户端单值、heads 跨状态恒定；**主导槽由初始 checkpoint 自动发现**（`--monopoly_slot -1`），所有后续状态必须保持同一主导槽且非主导 batch ≤1%，否则判"单槽内容演化前提不成立"拒（不强行套用 CTRP） |
| G1 | 逐过渡主导槽行的 prompt 更新度量（rel_norm / cosine）+ 状态性能漂移（drift pool） |
| H1 主门 | 当前任务 val CE 选择的 α̂（逐过渡独立选择，零历史任务信息）改善历史任务：G_old > 0 且层次 paired batch-bootstrap 95% CI 下界 > 0 |
| H2 | 当前任务非劣效：各过渡 cur gain 等权均值 ≥ 0（沿用原不等式，聚合层级） |
| H3 | **所有相邻 transition 逐过渡全表单独报告**（λ̂/old gain/cur gain/CI95/oracle），不允许挑选最佳边界 |
| Oracle | test-oracle 仅作非法参考上界，不参与步长选择；annex 报告逐过渡 oracle 内点占优计数 |
| 统计 | 沿用 §32.7 方案：paired batch bootstrap，B=1000，seed=20261103（聚合门 +137 派生 train 选择器种子）；选择器 = 当前任务 val/train CE argmin、并列取最小 λ |
| 选择信号隔离 | α̂ = argmin CE_current,val(α)；历史任务数据不参与步长选择；val/train 与 test 分离（inner-train 分层子采样 80/20） |

### 32.9.3 判决树（`--gate_mode all`，冻结）

```
H0 fail（含主导槽跨状态漂移 / 任一状态选择信号退化）→ REFUSED
G_old > 0 且 mean_cur ≥ 0 且层次 CI95_lo(G_old) > 0
    → OVERSHOOT_EXPLOITABLE_CURRENT_ONLY（INR 复核通过；§33 立项前置条件满足，待外审）
G_old > 0 且 mean_cur ≥ 0 但 CI95_lo ≤ 0 → YELLOW_REVIEW
否则 → OVERSHOOT_NOT_EXPLOITABLE（annex：合法选择器内点计数 / oracle 内点占优计数）
次级：train-current selector 同聚合读数（报告不作门）；
逐过渡 λ̂ 互异性 + prompt_drift annex（G1）= state-adaptive 复核证据。
```

### 32.9.4 与 §32 CIFAR 版的实现差异（全部已实现 + 回归验证）

1. `--monopoly_slot -1`：初始状态自动发现主导槽（仅 t_index=0 允许 auto，后续状态模式槽不一致即拒）；发现的槽记入 `slot_effective` 并落盘 meta。
2. `--gate_mode all`：等权聚合门 + `_aggregate_boot_ci` 层次 bootstrap（rep 内逐过渡独立重抽、等权平均、不并池）。
3. G1 annex：`annex.prompt_drift_slot`（逐过渡主导槽行 rel_norm/cosine）。
4. probe-G 对账：probe-G 从未跑过 INR，`--probe_g_csv` 指向不存在的路径 → 检查按预注册降级为 informative note（协议保留，数据无源）。
5. **回归验证（2026-11-03）**：single 模式在 CIFAR 既有缓存上重分析与正式 run 逐位一致（含 FINAL/gate/secondary/全部过渡记录）；all 模式在同一缓存上读数 mean_old +1.165pp、agg CI95 [0.996, 1.333]、mean_cur +0.715pp、secondary（train）+1.343pp [1.163, 1.517]、interior 4/4——与 §32.8 手算完全吻合。

### 32.9.5 fail-closed 增补

auto 槽仅在 state 0 发现；后续状态 batch-majority 模式槽 ≠ 发现槽即拒；聚合门要求**每个** eligible transition 都有有效选择信号（任一退化 → REFUSED）；其余沿用 §32.5 全部条款。

### 32.9.6 命令（GPT 外审通过 §32.9 后执行；INR 行规模小于 CIFAR，预计 30–60 分钟，GPU 共享时加倍；逐状态缓存可断点续跑）

```bash
/root/autodl-tmp/users/ych/.conda/envs/FedSMTA/bin/python diagnostics/probe_prompt_step_calibration.py \
    --run_dir output/ImageNet-R/fedta/inr_a5v2_M2_feat_p40/seed_42 \
    --data_path ./local_datasets --device cuda \
    --monopoly_slot -1 --gate_mode all \
    --probe_g_csv diagnostics_output/absent_probe_g_inr.csv \
    --cache_dir diagnostics_output/stepcalib_inr_seed42 \
    --out_csv diagnostics_output/stepcalib_inr_seed42.csv \
    --out_rows_csv diagnostics_output/stepcalib_inr_seed42_rows.csv \
    --out_json diagnostics_output/stepcalib_inr_seed42.summary.json
```

### 32.9.7 工程记录（INR 正式跑 2026-11-03）

- **执行**：用户终端跑 §32.9.6 命令原样，一次通过，零 bug、零 fail-closed 触发。
- **auto 槽发现是必要的**：INR 的主导槽是 **slot 53**（≠ CIFAR 的 slot 6）——外审"不要硬编码 slot 6"的要求被数据直接证实；5 状态 batch-majority 模式槽恒为 53、非主导 batch frac = 0.0、prompt 值跨客户端单值、heads 跨状态恒定，H0 全过。
- **覆盖**：选择信号正常，old 行覆盖 1438/3001/4407/5793 = 各过渡全部 old 行（t1–t4，INR 行规模约为 CIFAR 一半以下）。
- **probe-G 对账**：按 §32.9.4-4 预注册自跳过（CSV 无源，informative note 落盘）。
- **展示性小瑕疵（不判级、不改判）**：summary 的 per-state 记录未回显 `slot_effective`/`selection_rows`/`selection_drop_frac`（分析层展示键，非判决输入——premise 由模式槽恒定性 + auto 解析 + 缓存指纹把关）；下版可补。
- 产物：`stepcalib_inr_seed42.csv` / `_rows.csv` / `.summary.json` + 缓存目录（逐状态落盘）。

### 32.9.8 正式判决（2026-11-03，判级：探针级诊断，单 run 单种子，产判据不产数字）

**`OVERSHOOT_EXPLOITABLE_CURRENT_ONLY`（INR replication PASS）**——CTRP §33 立项前置条件（跨数据集机制复核）**全部满足，待最终外审**。

聚合门（primary = λ̂_val-current，`--gate_mode all` 预注册判决树）：

| 读数 | 值 |
|---|---|
| G_old（等权均值） | **+0.79pp** |
| 层次 paired batch-bootstrap CI95 | **[+0.45, +1.13]**（下界 > 0） |
| mean cur gain（H2 非劣效） | **+0.32pp** ≥ 0 |
| interior λ̂ | **4/4** 过渡 |
| oracle 内点占优（非法对照） | 4/4 |
| 次级（train selector，报告不作门） | **同判 PASS**：+0.53pp CI [+0.24, +0.80]，mean_cur +0.48pp |

逐过渡全表（H3，全部报告、不挑边界）：

| 过渡 | λ̂_val | old gain | 单过渡 CI95 | cur gain | λ\* oracle |
|---|---:|---:|---|---:|---:|
| 0→1 | 0.80 | +0.35pp | [−0.49, +1.24]（跨零） | +0.00pp | 0.75 |
| 1→2 | 0.65 | **+1.67pp** | [+0.84, +2.54] | **+1.49pp** | 0.80 |
| 2→3 | 0.90 | +0.25pp | [−0.11, +0.61]（跨零） | +0.15pp | 0.90 |
| 3→4 | 0.80 | **+0.89pp** | [+0.38, +1.40] | −0.36pp | 0.95 |

诚实读数（供外审）：INR 效应弱于 CIFAR（+0.79 vs +1.165pp），t1/t3 单过渡 CI 跨零——**按预注册设计只有层次聚合 CI 有判决权**（这正是全过渡聚合门的意义：不挑边界、也不让单一边界一票否决）；t4 有 −0.36pp 的当前任务小代价但 H2 聚合非劣效通过；state-adaptive 再次成立（λ̂ = 0.80/0.65/0.90/0.80 互异）；G1 annex 首次落盘 INR 逐过渡 prompt 更新度量（rel_norm 0.57–0.68 / cosine 0.73–0.83）。

**累计判决标签更新（待外审确认）**：`CONTENT-CAUSAL / DIRECTION-MECHANISM-UNRESOLVED / OVERSHOOT-HYPOTHESIS-PROMOTED` → **`OVERSHOOT-CONFIRMED-CROSS-DATASET / CTRP-SEC33-READY (pending external review)`**。

**GPT 三句式封档记录（frozen for external review）**：

> **The pre-registered all-transitions aggregate gate PASSES on ImageNet-R: across all 4 eligible adjacent transitions the val-current selectors (zero old-task information) pick interior steps with equal-weighted mean old-task gain +0.79pp (hierarchical paired batch-bootstrap 95% CI [+0.45, +1.13], lower bound > 0) and mean current-task gain +0.32pp, and the independent train-current selector concurs (+0.53pp CI [+0.24, +0.80], mean_cur +0.48pp).**
>
> **The effect is weaker but structurally identical to CIFAR-100: per-transition old gains are +0.35/+1.67/+0.25/+0.89pp with individual CIs crossing zero at t1/t3 (by pre-registration only the hierarchical aggregate CI decides, which is exactly why the no-cherry-picking all-transitions design was frozen), the optimal step is again state-adaptive (λ̂ = 0.80/0.65/0.90/0.80), and the auto-discovered monopoly slot is 53 (≠ CIFAR's slot 6) — the no-hardcoding protocol was not just prudent but necessary.**
>
> **The verdict is OVERSHOOT_EXPLOITABLE_CURRENT_ONLY (INR replication): the temporal prompt-update overshoot replicates cross-dataset as a deployment-repairable, current-task-signal-locatable step-calibration defect, CTRP's Sec.33 candidacy preconditions (INR mechanism replication) are now all satisfied pending final external review, and no training arm is pre-registered by this probe.**

---

## 34. BSAB：Broadcast-State Alignment Bridge（2026-11-04 立项裁决冻结；B_pre→B0→B1 执行树）

**状态标记（user 裁决 2026-11-04）**：`B_pre: VALID + STRONG POSITIVE PRIOR` / `B0 birth test: NOT YET OBSERVED` / `BSAB architecture implementation: STILL FROZEN`。

### 34.0 提案与执行树

BSAB：冻结 head，在 evolving representation 与 head 之间学显式无标签低秩坐标 bridge `B(z)=z+ΔAz+b`，替代 M2'' post-broadcast refit（§28 判据臂）；birth 问题 = single-broadcast mismatch `z_t⁺ → z_t⁻`。

冻结执行树（GPT 外审 2026-11-04）：

* **B_pre（archived aging stress probe）**：跨任务端 accumulated transport，只作先验，**不判 birth 生死**（`aging RED ⇏ birth RED` 且 `aging GREEN ⇏ birth GREEN`：多步 transport 非单调，可部分抵消/复合变复杂）。
* **B0（true birth-pair probe）**：pre/post-broadcast 配对缓存上的真 birth 检验，`ρ_birth ≥ 0.8` GREEN 门。
* **B1（rank/structure selection）**：shared low-rank vs per-task low-rank vs full/high-rank。

ρ_age 与 ρ_birth **不混名不混表**。ρ_age 分母 = birth-state reference gap（Acc(H_k(z_k))−Acc(H_k(z_j))，valid ⟺ ≥0.5pp，不叫 ceiling——bridge 可超过 birth-state accuracy）；ρ_birth 分母锚 M2''−M2'。负 gap（stale > birth-state ref）只记录不纳入。

### 34.1 B_pre 存档记录（`diagnostics/probe_bsab_feasibility.py`，只读 M2' run artifacts）

**RUN_01 = FIT_SANITY_FAIL / NOT_EVALUABLE / PRIMARY_SYMPTOM=A_MINUS_I_SPECTRUM≈NEGATIVE_IDENTITY**（`bsab_pre_report_run01_fitsanityfail.json`）。双重根因：
① 参数化 bug：ridge 正则了 A 而非 ΔA=A−I，λ 大时 A→0（常数映射摧毁特征；σ(A−I) 平坦≈1、E_r=r/768、DoF 2–18 为谱指纹）；② 环境损坏：本机 numpy/scipy float64 BLAS/LAPACK 无效（`ENVIRONMENT_AUDIT.md §1`）——RUN_01 的 acc_full/rank_acc 数值作废。

**RUN_02 = VALID**（`bsab_pre_report.json`，torch CPU 数值后端 + fit-sanity 6/6 PASS：`Y=X⇒A=I`、`λ→∞⇒B(z)=z`、`Acc_identity=Acc_stale` bitwise；z_ref 稳定门 PASS 0 violations；对官方 accuracy_matrix reproduction worst |Δ|=0.16pp；GCV 显式含 ΔA=0 候选，50/50 未选中）。

**判读（PRELIMINARY / PRIOR-LEVEL ONLY，M2pb task4 完成前不出 FINAL）**：

* `AGING TRANSPORTABILITY PRIOR = STRONGLY POSITIVE`：32/37 valid client-cells ρ_age≥0.5，多对 mean 0.8–1.2（可超 birth-state reference）。措辞冻结：*accumulated deployment-state drift appears substantially transportable by an unlabeled coordinate bridge*。禁止写 "BSAB can replace M2''"。
* 2×2 象限：feat_good_head_good=6 / **feat_bad_head_good=26** / feat_bad_head_bad=5 / feat_good_head_bad=0——feature 级 R_feat 不是 BSAB 主目标，**not all feature drift is semantically relevant**；"feature MSE 一般 + head recovery 强" 在 birth-B0 应视为正结果。
* **叙事修正（论文级）**：σ(ΔZ) E_64 仅 0.40–0.50——漂移能量**不低秩**；但 rank sweep 显示 r=8–32 即拿到大部分 head 恢复且与 full GCV 互有胜负。冻结表述：*the classifier-relevant component of broadcast-induced drift is concentrated in a low-dimensional corrective subspace*（D_head ⊊ D_energy, dim(D_head) ≪ dim(D_energy)），不写 "drift is low-rank"。
* 负 reference gap 21/50 cells 只记录（aging 有时利于旧 head——真正变量是 state displacement relative to head-sensitive directions，非 task age）。

### 34.2 0b：`BSAB_PAIR_CAPTURE` instrumentation-only hook spec（预注册；**hook + 单测已实现 2026-11-04**，见 34.2.6；授权范围独立于"只读 artifact"）

**授权边界**：单独 instrumentation-only scope；不碰 CTRP/endpoint INR/当前 M2pb run；**正式 BSAB 结构实现仍冻结**，等 true B0。载体 = M2pb 完成后单独开 deterministic、matched 的 `BSAB_PAIR_CAPTURE` rerun（同配置同种子同划分；若未来本就需要同配置 rerun 可搭车，但必须先过 34.2.4 RNG-neutral regression）。

**实现裁决（user，2026-11-04 第二轮）**：`BSAB_PAIR_CAPTURE: APPROVED FOR IMPLEMENTATION`；**capture run = M2'-timing 冻结**（不改为 matched to M2pb：M2' 给出最干净的 birth 基准 `z⁻→H_birth`，直接回答 "能否用 coordinate synchronization 替代 classifier retraining"；若 matched to M2pb 则 birth head 反而变成需额外解释的中间状态）；**M2'' = offline reconstructed positive-control arm**，subject to reproduction gate。排期：等 M2pb Task-4 FINAL 的同时**并行实现 hook + OFF/ON regression**；不启动正式 BSAB 结构实验。状态：`B_pre: VALID / STRONG POSITIVE PRIOR` / `B0: APPROVED FOR INSTRUMENTATION, NOT YET OBSERVED` / `BSAB architecture: FROZEN` / `M2'-matched capture: APPROVED` / `M2'' offline positive control: APPROVED, subject to reproduction gate`。

**六条实现 hard gate（user 2026-11-04，逐条必须成立）**：

1. **严格同一批 sample 配对**：z⁻ᵢ 与 z⁺ᵢ 同 sample、同顺序、同 preprocessing；直接保存 sample/index hash，不依赖 dataloader 顺序。
2. **birth head 在 broadcast 前锚定**：`load_prompts` 前保存 H_birth + `head_hash`；所有 frozen-head arm 验证 hash 不变。
3. **capture 不影响正式轨迹**：四流 RNG snapshot/restore 之外，OFF/ON regression 同时核对 prompt hash / head hash / optimizer-state 或关键 tensor hash / accuracy matrix / 后续 round candidate hash；目标是 **OFF/ON trajectory bitwise identical**，不止 accuracy 一致。
4. **bridge fit 严格 label-free**：标签只进最终 accuracy evaluation 与 M2'' positive control，不进 bridge selection/fitting。
5. **M2'' reproduction 先过关再算 ρ_birth**：离线 M2'' 与官方 M2pb matrix 超出预冻结容差 → ρ_birth 不报告（分母锚点不可靠）。
6. **不许因 B_pre positive 提前调 B0**：strongly positive prior 不构成调 rank/λ/挑 task 的依据，B0 保持 confirmatory birth test。

**34.2.1 配置与捕获点**（rerun matched to M2'：`--refit_timing local`，pre 状态即真实 M2' birth state）：

* **pre（z⁻）**：client 完成本轮 local training 与每轮 M2' refit 后、`get_global_proto_and_head*` 装载 server fused prompt **之前**（`Client_DF.py` L924 `self.prompts = prompt` 之前，此时 `vit` 仍持本地训练后 prompts = L378 `deepcopy(get_prompts())` 状态）。
* **post（z⁺）**：`vit.load_prompts(广播 prompt)`（L925）之后、`_a5_post_broadcast_interface` / 任何 post-broadcast refit / evaluate **之前**。与 §28 广播路径钩子同位。
* **same rows**：两侧同一 sample index 集合；**缓存该 client 该任务全部可用 train/test rows**（head refit 输入为 feat，全行缓存 ⇒ M2'/M2'' 两 refit 臂与全部 bridge 臂可离线复现）。N_fit<2048/client 时用全部可用行并记 `LOW_POWER`；N_fit≲2d≈1536 为 full-affine 欠定线。eval 独立：完整 test split 或 ≥512 rows。
* cadence：**每个 broadcast boundary（每轮）**捕获，25 boundaries（5 task × 5 round），带 round_id。

**34.2.2 每行字段**：

```text
client_id / task_id / round_id / sample_index / split / label   # label 仅离线 accuracy
z_ref_pre / z_op_pre / z_ref_post / z_op_post                    # float32
# state anchors:
pre_prompt_hash / post_prompt_hash / head_hash / row_index_hash / feature_shape
```

**bridge fit sees no labels**（z_ref 的 pre/post 双缓存用于在真 birth 场景重验 invariance）。

**34.2.3 forward 卫生**：`model.eval()` + `torch.no_grad()` + 前后恢复 model mode；固定 index 的 deterministic diagnostic loader，**不复用正式训练 loader 的 shuffle/iterator 状态**。

**34.2.4 RNG neutrality 硬 gate**：`RNG_before capture == RNG_after capture`，覆盖 Python / NumPy / Torch CPU / Torch CUDA 四流（复用 `_a5_rng_snapshot/_a5_rng_restore`，§23.1 单测先例）；再做 `capture OFF vs capture ON` 小规模 regression——后续 candidate/prompt hash/训练状态**逐位一致**才准上正式 rerun。通过后 capture 方为 instrumentation-neutral。

**34.2.5 B0 对照臂与判据**（离线诊断，全部从缓存计算；官方 evaluate dropout-ON deviation 沿 §34 候选记录）：

```text
每个 broadcast boundary 上、同一 eval rows：
  ① Acc_pre-head on z⁺（≡ identity bridge ≡ M2' deployed head 直接服役）
  ② Acc_full-residual-bridge（GCV ridge，λ‖A−I‖²，含 ΔA=0 候选）
  ③ Acc_low-rank-bridge（r sweep）
  ④ Acc_M2''（离线按冻结配方在 z⁺ 上 refit；对官方 M2pb 复现校验）
  （参考记录：Acc_M2' pre-head on z⁻ = birth-state reference，不作臂）
```

`ρ_birth = (Acc_bridge − Acc_identity) / (Acc_M2'' − Acc_identity)`（identity 臂 = M2' 实现的显式 control）。前置：离线 refit 臂对官方 M2'/M2pb accuracy matrix 的 reproduction check（超差 → PROTOCOL_MISMATCH，ρ 不出）。

判据冻结：**GREEN** ⟺ mean ≥0.8 且 pooled ≥0.8 且 worst 无系统性崩溃（跨有效 transition）；**YELLOW** 0.5–0.8（只许研究 loss/rank，不进主方法）；**RED** <0.5（full residual affine 都 <0.5 才真 RED）。z_ref 不变性门沿 B_pre 语义：violation → `ZREF_INVARIANCE_VIOLATION` 暂停结构结论，查 extraction bug，**不自动 fallback 1536 维**。

**stopping-time 纪律**：M2pb task4 完成前只出 PRELIMINARY（task0–3），FINAL B0 VERDICT 等全有效 transition 集。

**34.2.6 实现记录（2026-11-04，instrumentation-only diff）**：

* `--bsab_pair_capture`（`config/datasets_delay.py`，默认 OFF = 逐位 no-op；legacy args 无该字段亦安全）。`Server_DF.__init__` fail-closed：与 `--refit_timing post_broadcast` 互斥（`BSAB_CAPTURE_UNSUPPORTED_REFIT_TIMING`，capture run = M2'-matched 冻结）；诊断构建（run_manager=None）拒绝启动；capture 目录 = `<run_dir>/bsab_capture/`（经 `args.bsab_capture_dir` 接线，client 不自行推导路径）。
* `Client_DF.py`：`_bsab_capture_pre/post` 挂入两个广播入口（pre 在 `self.prompts = prompt` 前 = birth state；post 在 `load_prompts` 后、`_a5_post_broadcast_interface`/evaluate 前）。gate-1：pre/post 两半独立重建 row manifest（当前任务全行 train[=refit cache 口径]+test，固定顺序），`row_index_hash` + labels/split/sample_index 逐位重验；gate-2：pre 半段锚定 birth head（`heads[task_id]` 完整 state + hash 随缓存落盘），post 半段重验 hash；gate-3：四流 RNG snapshot/restore + 全模块 mode 恢复 + 独立 deterministic loader（eval+no_grad、shuffle=False、num_workers=0、batch=16 与正式缓存口径一致）；同名文件已存在 = fail-closed 拒绝覆盖。每 boundary 产物：`bsab_c{client}_t{task}_r{round}_{entry}.npz`（client/task/round/sample_index/split/label、z_ref_pre/z_op_pre/z_ref_post/z_op_post float32、`head__*` birth head state）+ `.json`（pre/post prompt hash、head_hash、row_index_hash、n_train/n_test、LOW_POWER 标记、feature_shape、gates 校验位、label-free 契约 note）。entry ∈ {broadcast（active client）, broadcast_no_test（passive client）}。
* 单测 `tests/test_bsab_pair_capture.py`（11/11 PASS，CPU）：OFF = 逐位 no-op（含 legacy args、无 birth state skip）；ON 缓存内容逐字段校验（z_ref pre/post 不变性、z_op 精确偏移 broadcast delta、row manifest/head state/hash 全等）；gate-1/gate-2/半段错配/覆盖保护/缺目录全 fail-closed；**instrumentation neutrality：OFF vs ON 广播边界后 model/vit/head 张量、全模块 mode、四流 RNG state 与 next-draw 逐位一致**（run 级 OFF/ON regression 仍为正式 rerun 前的独立硬门，34.2.4）。
* 既有回归：`tests.test_a5v2_refit_switches` + `tests.test_ctrp` 合跑 81 tests，除 Test8 CUDA replay 2 例（GPU 与运行中 M2pb 竞争的间歇 OOM；单跑 PASS 复现不了）外全过。

#### 34.2.6a 协议发现：checkpoint 的 `prompts` 字段语义 = post-broadcast（2026-11-05）

* **发现载体**：DSAR 离线试点冒烟（`diagnostics/dsar_offline_pilot.py`，cifar_base_p15 round_0010，用户执行）。症状：displacement c0–c4 **全部恰为 0.0**、`bt_identity=1.0`——z⁺ 与 z⁻ 逐位相同，B_t 三臂退化。
* **根因**：广播入口 `Client_DF.py` L1234/L1261 `self.prompts = prompt` 在轮末 checkpoint 保存**之前**覆写 client 的 birth prompts → 所有 round/task_end checkpoint 里 `clients[i]['prompts'] == server_prompt`（全部 5 个 client；baseline/M2'/M2pb 所有 run 同构）。恢复后两遍前向逐位一致，与症状完全自洽。
* **含义**：c1–c4 的 birth prompt 值**不可从任何 checkpoint 恢复**（c0 例外：KD no-op ⇒ fusion 轮 fused == c0 birth）。round 级 (z⁻, z⁺) 真配对的唯一来源 = in-run capture artifacts（`bsab_capture/*.npz`）——离线重构路线对 B_t 不成立，等价于回到预注册 B0 路径。
* **连带裁决**：(1) DSAR pilot `--rounds all` 取消（B_t 列无效、L_t 列语义偏差）；(2) L_t（birth 语义）的修正试点 = capture npz 的 z⁻/labels + round checkpoint 的 tail_anchor（key/anchor_pool/head 不受广播覆写，H_end 可恢复）按 (client, task, round) 对齐——等 capture rerun 跑完后与 B0 一并做。
* **留档的冒烟读数**（仅 round_0010、15 cells、PRELIMINARY 非门）：L_t 在**部署态（广播后）表示**上 +2.2pp overall / +3.3pp current-task，r=8 即近饱和——部署态读出存在低秩可修 slack 的初步迹象（注意：这是 L(z⁺) 语义，非设计的 birth-state L_t）。

#### 34.2.6b 环2 identity gate：capture run ≡ baseline（2026-11-06，PASS）

`diagnostics/verify_trajectory_identity.py`（本窗口新写，两 run 版 Gate 0C）：cifar_base_p15 vs cifar_bsab_capture_M2matched，全 25 个 round checkpoint 逐张量零容差比对。**PASS：25/25 轮全部 HARD 字段（server_model/global_head/server_prompt/protos/clients[i] 的 prompts+tail_anchor_model+data_split_indices+class_mask+outer_split_manifest）bit-identical**；expected_diff 仅有 refit 头（`heads[]`）与 `a5_joint_head`（后者经验证与 tail_anchor 内 head 逐位相同的冗余快照，仅 refit run 落盘）。含义：**refit + capture 插桩在真实全量 run 上 RNG 中立，capture run 表征轨迹 ≡ baseline 轨迹**——B0 材料对 baseline 合法；与 izfree Gate 0A（注意：Sec.34 状态那次尚未补跑，见 §10.1 勘误）拼成"官方 ≡ 代码 ≡ capture"证据链。capture run 终态：25 round checkpoints + 125 npz boundary 全齐，integrity 持续更新到最后。

#### 34.2.7 B0 正式判决：GREEN（2026-11-06 封板）

**脚本**：`diagnostics/bsab_b0_verdict.py`（预注册 §34.2.5 四臂实现；125 cell 全量 + gate_only 增量模式，arm4 refit 头缓存 `bsab_b0/heads/`）。产出：`bsab_b0/{cells.csv, repro.csv, verdict.json, heads/}`。

**§34.2.5 gate 勘误（两轮修订，均记录在案）**：
1. **0.25pp 逐位容差 → MC-dropout 带宽**（根因：官方 evaluate 路径从未 `.eval()` ⇒ Chead Dropout(0.1) ON ⇒ 官方 CSV 数 = dropout-ON 随机前向的单次采样，逐位容差数学上不可满足；Client_DF L651 探针注释为证）。修订判据：官方数落在我们 MC 分布（50 draws）3σ 带内（|z|≤3）且 z 符号不系统性同侧（同侧 ≤20/25）。
2. **arm4 降级 reference_only**（MC 结果两臂分裂的判定）：arm1（npz in-run 锚定头）25/25 PASS（|z|≤2.5，7+/18−）= 读出语义 ≡ 官方；arm4（离线重训头）6 cell 出带（z −9.4~+6.5）但符号 13+/12− 双向均衡 ⇒ 非系统偏差，是**重训随机性**（in-run refit 的 batch/shuffle/dropout RNG 流不可离线复刻，同配方不同随机实现）。裁决：ρ 分母只需管线内有效（负分母排除已覆盖其噪声），不与官方抽签头逐位对齐 ⇒ gate = arm1-only + zref，arm4 对照留档 repro.csv。

**判决**（verdict.json）：

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| repro gate (arm1 MC) | 25/25 \|z\|≤3 + 符号均衡 | 25/25，\|z\|≤2.5，7+/18− | PASS |
| z_ref 不变性 | 25/25 逐位 | 25/25 | PASS |
| ρ_birth mean | ≥0.8 | **1.006** | PASS |
| ρ_birth pooled | ≥0.8 | **1.018** | PASS |
| task 崩溃 | 无 | 无 | PASS |

**VERDICT: GREEN —— B1 结构实现解冻。**

**四臂读数**（125 cell，M2' refit head 读出）：① identity 94.83 / ② bridge 96.64 / ③ lowrank r8-r64 96.40-96.61 / ④ M2'' 96.57 / ref z⁻ 97.24。ρ per-task：t0 0.707 / t1 0.990 / t2 0.968 / t3 1.245 / t4 1.033（valid 84，负分母排除 41）。两论文点：免标签桥 ρ≈1（零标签追平重训上限；20/125 cell GCV 自动选 identity = 安全阀；41 个 refit 失效 cell 上桥仍 +0.15pp）；低秩 r8（~1% 参数）拿回 87% 修复量、sweep 平坦。

**baseline 部署口径参照**（`dsar_birth_preview/summary.json`，H_end 联训头读出，PREVIEW）：damaged 93.52 → +B_t 95.86 → +B_t+L_t ≈96.3-96.8（current-task +2.8~3.3pp，oracle 3.72pp 的 ~80%）；损伤随任务放大 t0 −1.56pp → t4 −4.69pp，桥修复随之放大（t4 86%）。注意：current-task 诊断口径 ≠ 论文主表 AA；旧任务保持收益待 B1 正式 run。

**下一步**：B1 预注册（部署图 H_end(L(B(z⁺)))、B_t GCV 在线化、L_t/B_t 拟合时机、rank 冻结、通信开销、旧任务度量）→ 实现 → INR B0-lite（依赖 izfree capture）→ 多 seed。

#### 34.3 INR B0-lite 预注册（2026-11-06，跑前冻结）

* **材料**：izfree 的 ImageNet-R capture run（M2'-matched 同款协议；izfree 无 INR baseline run ⇒ 环2 identity gate 免除，中立性由 cifar 链（§10.1 三环 + §34.2.6b + §34.2.7）承担——跨机器不可逐位一致，此前已裁决）。**〔勘误 2026-11-06 晚〕**"INR 无官方 a5v2 锚 run"前提**错误**：autodl 本机存在 `inr_a5v2_M2_feat_p40` / `inr_a5v2_M2pb_p40`（官方 M2'/M2'' 完整 run，含 client CSV 与 checkpoint）——当时只确认了 izfree 侧未查本机。判决不追溯翻案（按落盘预注册执行），repro gate 改为 **post-hoc robustness 补跑**（见 §34.5 补跑条目）；baseline 对照锚 = `inr_base_p40` 官方对角线 82.87/77.56/78.62/79.05/78.76（mean 79.37，server 聚合口径）。
* **判据裁剪**：INR 无官方 a5v2 refit 锚 run ⇒ **reproduction gate 裁除**（`bsab_b0_verdict.py --no_repro`，verdict 标 `[B0-lite, no_repro]`）；**z_ref 不变性门保留**（npz 自含）；**ρ_birth 四臂 + GREEN/YELLOW/RED 阈值原样**（run 内配对，跨机器分析合法）。
* **协议差异**：INR 官方协议 = 每客户端 40 私有类、0 公共类、每任务 8 类、surrogate 固定 5、batch 16 硬编码——与 cifar 的划分不同是预期行为，不作 fail 条件。
* **流程**：izfree `git pull` 最新 diagnostics → 完整性检查（25 ckpt + 125 npz + args.json flags）→ `--repro_only --no_repro` 冒烟（round 0-4）→ 全量 `--no_repro`。

#### 34.4 B1 preregistration v1（2026-11-06 冻结，外部审查版采纳；跑前冻结，方可动训练代码）

**一句话冻结**：per-broadcast client-local r8 DSAR sidecar；fixed normalized ridge + unlabeled identity guard；zero added FL payload；training trajectory unchanged.

**B0 遗留口径修正（随封板生效）**：① ρ>1 论文措辞不写"超过重训上限"，写"达到/饱和 M2'' reference 的恢复量"（M2'' 自带重训随机性）；② 今后始终同时报 raw accuracy 与 valid-denominator ρ（41/125 负分母限制 ρ 的全 cell 解释力，但不推翻 bridge 的 all-cell raw 增益）；③ 500/1000 次 MC 为论文 robustness 附加项，非解锁条件；④ 20/125 identity no-op（transport 不需要时自动退化）与 41 refit-negative（有监督 reference 自身不稳定）不冲突，各归各叙事。

**六项冻结**：

1. **部署图 = 纯 readout sidecar**。`ŷ = H_end(z_ref ⊕ L(B(z⁺_op)))`：B/L 只碰 768-D operational branch，**z_ref 全程 bypass**；broadcast hook 只负责拟合/更新 sidecar 状态，不改 forward、不进 `load_prompts`、不 monkey-patch feature tensor。正式部署唯一入口 = 统一 `DSAR readout` helper（被 evaluate/predict 调用）。**local-training forward 不经过 B/L**——prompt/backbone/head 训练轨迹与 FedTA baseline 相同。结构不变量（硬门材料）：DSAR_ON vs OFF 的 base prompt/head/checkpoint tensors **逐轮逐位一致**，差异只允许存在于 sidecar 参数与读出结果。
2. **B_t 在线 λ 冻结**（不跑 GCV、无在线 sweep）：`λ_{c,r} = N_{c,r}·s_{c,r}·γ*`，s = tr(XᵀX)/(N·d)（X = 当前 cell z⁺ train 行中心化）。**γ* = 0.396419**（`diagnostics/bsab_gamma_star.py` 从封板 B0 artifact 导出：105 nonidentity cell 的 λ_GCV/(N·s) 中位数，IQR 0.374–0.652，落盘 `bsab_b0/gamma_star_corrected.json`。**ERRATUM-gamma-split §34.4c**：superseded γ\*=0.390324 的 s 误建于 test 行，2026-11-06 修订为 train 行口径，详见 §34.4c）。**无标签 identity guard**：train rows 按 sample-index hash 固定切 80% fit / 20% calibration（不消费 RNG）；固定 λ 在 80% 拟合，20% 上比较 `MSE(B(z⁺),z⁻) < MSE(z⁺,z⁻)` 才启用，否则严格 B=I；启用后用全部 train rows 重拟合一次。**test/val 不参与 λ、identity、rank 任何决策。**
3. **拟合时机 = per-broadcast、per-client**（与 B0 证据单元 125 cell 同构，不改为 task boundary）。每轮时序冻结：local training 正常结束 → H_end → 冻结 backbone/prompt/head → identity 初始化拟合 L_{c,r} → 固定 index loader 取 z⁻_op（current-task train split 全行）→ load server broadcast → 同一 index manifest 取 z⁺_op → 拟合 B_{c,r} → evaluate 走 B→L→H。B 全程无标签；L 可用 current-task train labels（冻结-head CE，预注册职责）。**每轮 fresh-fit from identity：不 warm-start、不跨轮 compose**（B_r∘B_{r-1} 累积动力学未经 B0 验证，禁止）。
4. **rank 冻结 r=8**（B 与 L 统一）：B = residual ridge ΔA + 确定性 SVD 截断 r8 部署为低秩 factors；L = rank-8 residual 参数化直接训练。r16/32/64 只作**主 run 完成后的结构消融**，禁止看 B1 结果后改主方法 rank。依据：B0 r8=96.40 vs full 96.64（87% 修复量）、r64=96.61（+0.03pp）。
5. **adapter FL 通信 = 0**。B_{c,r} 是 client-local broadcast compensation（目标 z⁻ 是该 client 的本地坐标，跨 client 聚合 A 无 B0 证据且会平均坏 local coordinate target）——server 只传原 FedTA payload，client 收到后本地拟合 B。本地只存 r8 factors（U,V ∈ R^{768×8}, b ∈ R^768，13,056 floats ≈ 51 KiB/module，B+L ≈ 102 KiB/client）；full 768² 矩阵仅限求解期内存，**禁止进 uplink/downlink**；若模拟器由 server 代存 client-local state，只存/回传 r8 factors 并在论文通信核算中标明 simulator state-management。
6. **判据 = 结构硬门 + 效果联合 GREEN 门**（官方 FedTA 聚合口径不变；a_{t,k} = 完成 task t 后对 task k 的 accuracy）。指标：AA_T = mean_k a_{T,k}；F = mean_{k<T}(max_{s≥k} a_{s,k} − a_{T,k})；**G_old = mean_{t>k}(a^{DSAR}_{t,k} − a^{Base}_{t,k})**；Δ_k = a^{DSAR}_{T,k} − a^{Base}_{T,k}。

**GREEN 联合规则**（跑前冻结，不许跑完挑指标）：

```
Structural hard gate: PASS        # 任一失败 → 直接 RED，不看 accuracy
  - DSAR_OFF ≡ matched baseline（逐位复现）
  - DSAR_ON 时 base prompt/head/checkpoint 逐轮 hash ≡ OFF
  - 无 test leakage；B/L fit failure → fail-safe 回 identity
  - 无 NaN/Inf；正式 evaluate 命中 B→L→H，training forward 不命中
ΔAA_T ≥ +0.50 pp
G_old  ≥ +0.25 pp                  # 防止只把 current task 做漂亮
ΔF     ≤ +0.25 pp                  # Forgetting 允许小波动、不许明显恶化
min_k Δ_k ≥ −1.00 pp               # 单任务不许崩超 1pp
```

ΔAA_T>0 但联合门未全过 → **YELLOW**（保留结构，不宣称端到端成功）；ΔAA_T≤0 且 G_old<0 超出噪声，或任何硬门失败 → **RED**。**B1 single-seed GREEN 仅 = 结构进入正式复现阶段的门**；论文统计显著性由多 seed 决定，不提前包装。

**Shadow comparison 细则**：Base-vs-DSAR 在同一 checkpoint 上做 paired evaluation——评 raw readout 前 snapshot RNG、评 DSAR 前恢复同一 RNG，两臂看到相同 dropout masks（B0 暴露的官方 dropout-ON 语义不复辟为 .eval()，但把比较噪声压到最低）。官方语义矩阵照原样保存，paired matrix 作为 B1 gate 的因果读数。

#### 34.4a B1 Dispatch Audit Addendum — APPROVED（2026-11-06 冻结；历史 B-only 分支 + 部署图勘误）

**Audit verdict**: `SHARED_PATH_CONFIRMED`。Evidence: `diagnostics/dsar_task_dispatch_audit.json`（Client_DF / Server_DF / Tail_Anchor 六项静态调用图证据，文件:行号齐全）。关键链：`vit(input, task_id=self.task_id, train=True)` → z_op → `self.model(z_op, target)` → key-anchor routing → Chead（Client_DF.evaluate L1344-1357）；历史任务评估 vit 前向恒用**当前任务 id**（L1347）且广播 prompt 已先装载（L1235）——历史 z_op 与本轮 B 所修共享**同一次 broadcast 参数状态变换**（注意：不等于所有样本位移向量相同，F⁺(x)−F⁻(x) 非 constant；历史有效性仍由 G_old 实验检验，静态审计仅证明架构合法性）。key/anchor_pool 为 client 本地参数、不进任何 fed_avg payload——B 插入不改变路由算法/head 选择/class mask（anchor index 允许随修正特征自然变化，即 B0 preview 已验证的部署语义，不得人为固定 baseline anchor index）。

**Deployment erratum**: §34.4 原部署图 `H_end(z_ref ⊕ L(B(z_op)))` 不符合真实 Tail_Anchor 前向——**z_ref 是 frozen original-model 参照流，不参与正式 head 输入**（Client_DF L992）。实际 Chead 输入 1536 = `[corrected z_op ‖ routed anchor(corrected z_op)]`。正确部署算子（TA_k(z) = Mask_k(Chead_k([z‖a(z)]))，a(z) 为原 key-anchor routing）：

- **当前任务 k=t**：`ŷ_t = TA_t(L_{c,r}(B_{c,r}(z_op⁺)))`
- **历史任务 k<t**：`ŷ_k = TA_k(B_{c,r}(z_op⁺))`（**B-only，禁用 L**）

**四项不可变语义**：① B 插在 `self.model()` 入口之前，修正特征同时参与 key routing 与 Chead feature half；② k<t 只 B→Tail_Anchor；③ head 选择恒 `self.heads[k]` + 原 anchor pool/key routing/class mask；④ B 的 active/identity 仅由当前任务 train split 无标签 calibration 决定，**禁止以历史 task accuracy 或测试标签决定**。**B identity ≠ DSAR identity**：guard 选 B=I 时当前任务的 L 照常生效，不得短路成 `return original_readout(z)`。

**Engineering constants（批准冻结；γ\* 经 §34.4c 勘误修订）**：γ\*=0.396419（ERRATUM-gamma-split，见 §34.4c；superseded 0.390324）；rank_B=rank_L=8；LOW_POWER_IDENTITY：fit rows<32 或 calibration rows<8 → exact identity（含 bias 全清零；hash mismatch/NaN/solver 错误等完整性异常**不得**用此理由降级）；L 配方：100 epochs、bs=64、`torch.optim.Adam`（不得换 AdamW）、lr=1e-3、weight_decay=1e-4、λ_L=0，U 非零/V 零/bias 零初始化，无 early stopping、不以 val/test 选参。

**Evaluation integrity**：官方 evaluate 的 `train=True`、bs=8、shuffle 语义不被 DSAR 修改；所有额外拟合/诊断维护 RNG/mode/主训练状态中立性；**task-end 全任务评估必须读取与该时刻 broadcast 状态匹配的最新已提交 sidecar**（不得仅凭 broadcast 内两次 evaluate 验证历史部署）；B/L 的 ON 结果按实际 bs8+train=True 官方口径统计，不以 bs16 离线缓存替代。**G4 必须验证**：OFF 的正式 evaluate() 完全未改；OFF/ON(identity) 同 RNG 同 batch manifest 下 logits 逐位一致；额外 fit/capture 不改变后续官方训练或评估 RNG 状态。

**Implementation authorization**: `Models/DSAR.py` Shell + `Client_DF.py` 受控接线 + `verify_dsar_invariance.py` 即刻开工；G1 未 GREEN 前不启动正式 B1 训练。第一阶段交付 = 真实 diff + OFF/ON(identity) 逐位中立性结果 + 当前/历史任务 readout 接线测试（非 accuracy）。

#### 34.4b B1 数值口径修正 + G2 断言冻结 — APPROVED（2026-11-06 方案级 PASS；断言先行冻结，待 G2 执行时验收）

**两项数学修正闭环（复审裁决：不再重开设计）**：

1. **截断后 bias 重算**：B1 唯一有效实现是 r8 截断后重算
   `b₈ = μ_y − μ_x − μ_x@ΔA₈`（等价 `μ_y − μ_x(I+ΔA₈)`；部署 refit 步用全部 train rows 的均值）。B0 `bt_rank` 保留全拟合截距的做法自此降级为**历史实验事实**，不再作为 B1 实现依据（不登记勘误——B1 从未以旧口径跑过正式臂）。实现：`Models/DSAR.py` `_bias8()` + 部署步（全行 refit → `_trunc8` → `_bias8`）。
2. **identity guard 裁决对象 = 实际部署的 r8 候选**：五步流程 `80% ridge → r8 truncation → b₈ → 20% calibration paired-MSE（identity 赢平局）→ 胜出才全行 refit 再截断`。guard 不得比较未截断 full-ridge（对抗反例实证 `mse_full=0.0205 < mse_id=0.0208 < mse_r8`：旧 guard 会部署一个部署后更差的 bridge，INR preview task0 −0.19pp 即该漏洞的真实数据投影；新 guard 正确返回 IDENTITY_GUARD）。实现：`Models/DSAR.py` guard 块（`_trunc8` + `_bias8` + `_mse_r8`）；aux 键 `mse_bridge_r8_cal` / `mse_identity_cal` / `lam_fit` / `lam_refit` 专供 G2 断言。

**G2 断言表（冻结，G2 执行时逐项验收）**：

| 断言 | 要求 |
|---|---|
| `G2_RANK8_GUARD` | guard 比较实际 r8 部署候选：外部独立代码路径复刻冻结配方（fit-half solve → SVD r8 → b₈ → cal MSE），与 aux `mse_bridge_r8_cal` 数值一致（相对容差 1e-9） |
| `G2_BIAS8` | 部署 bias 与独立公式 b₈（全行均值 + 截断算子）一致（float32 往返容差 ≤1e-5）；全秩截距 c_full 不得被部署（rank>8 位移场景 c_full≠b₈ 必须成立，防口径回退） |
| `G2_IDENTITY_TIE` | `status==IDENTITY_GUARD ⟺ mse_identity_cal ≤ mse_bridge_r8_cal`（identity 赢平局）；每个拟合 cell 断言决策/公式等价 |
| `G2_REFIT` | FITTED ⇒ 全部 train rows + 同一 γ\*=0.396419（§34.4c 修订值）重拟合（aux `lam_refit` 与外部重算 λ=N·s·γ\* 一致），并再次截断 r8 + 重算 b₈ |
| `G2_LOW_POWER` | fit rows<32 或 cal rows<8 ⇒ LOW_POWER_IDENTITY（含 bias 全清零）；hash mismatch / NaN / Inf / solver 错误等完整性异常必须 raise，**不得**经此条款降级 |

CPU 冒烟证据（2026-11-06 全 PASS）：外部复刻对账逐位一致（mse_r8=0.1004968726 两侧相同；b₈ 公式 vs 部署 bias 最大差 3.7e-9；c_full 与 b₈ 差 2.1e-2，证明口径切换实质性）；对抗扫描命中 2 个 full-wins/r8-loses 配置，均被新 guard 正确拒绝；重放逐位确定性、LOW_POWER、L 拟合、sidecar dispatch 全绿。

**G1 写盘纪律（工程条款，冻结）**：G1 工具写入既有 baseline run 目录时**只允许新增明确命名的诊断产物**（默认 `<run_dir>/dsar_g1_invariance.json`），不得覆盖/修改原始 checkpoint、args.json、accuracy matrix 或任何 B0 封板文件。当前实现审计：`verify_dsar_invariance.py` 唯一写盘点 = 输出 JSON（`open(out_json, "w")`）；checkpoint 经 `load_checkpoint` 只读装载；`diag_utils` 对 run_dir 仅读 `args.json`；`build_server(run_manager=None)` 不触碰 accuracy records；每客户端测试结束后内存旗标恢复 OFF 默认。

#### 34.4c CIFAR B1 Gamma Normalization Erratum — APPROVED（2026-11-06 预注册勘误，pre-formal-run correction）

**Decision: APPROVED — PRE-FORMAL-RUN CORRECTION**。§34.6 Stage 0 复审发现：原冻结 γ\*=0.390324 的导出脚本（`bsab_gamma_star.py` 旧版）以 `split == 1` 取行计算归一化尺度 s，而 capture writer 编码为 **0=train, 1=test**（`Client_DF._bsab_forward_rows`；artifact 级证据 = capture manifest JSON 的 n_train/n_test/n_rows 声明 + 105/105 cell `count(split==0)==n_train` 对账），即 s 误建立在 **test rows** 上，违反 train-only calibration 纪律。修复 split 编码后，使用 train rows（`split==0`）重新导出。

**勘误登记表**：

| 字段 | 值 |
|---|---|
| `gamma_star_old` | 0.390324 |
| `gamma_star_corrected` | **0.396419**（导出全精度 0.396419312045252） |
| `reason` | normalization s computed on test rather than train rows（split 编码误读） |
| `effective_from` | CIFAR B1 formal run 前（B1 尚未开跑，`prior_formal_runs_affected = None`） |
| `prior_formal_runs_affected` | None |
| `b0_verdict_changed` | **False**（错误仅存在于 B1 常数冻结的 γ 导出环节，B0 四臂判决不重开） |

**Effective rule**：CIFAR B1 正式实现唯一使用 corrected γ\*=0.396419（`Models/DSAR.py::GAMMA_STAR`，DSAR_VERSION bump `dsar-b1-shell-v2-gamma-erratum`），在线 λ 公式 `λ=N·s·γ*` 不变。修订原因仅为数据行集勘误，**不依据任何正式 B1 ON/OFF accuracy 选择**。B0 GREEN、B1 r8、identity guard、部署图（§34.4a）、L 优化配方和四项联合 GREEN 门均不改变。差异约 +1.56%，但 identity guard 是离散选择边界（`MSE_{r8} < MSE_I`），个别 cell 可能在 `FITTED` 与 `IDENTITY_GUARD` 间切换，故 G1 及受影响 G2 数值断言必须重跑并记录前后 `FITTED`/`IDENTITY_GUARD` 数量变化（不要求新旧判决逐位一致）。

**Artifact 处理**：旧版 `bsab_b0/gamma_star.json` 原文与哈希保留（superseded 历史证据），修正版单独落盘 `bsab_b0/gamma_star_corrected.json`：

- superseded `gamma_star.json` SHA-256: `25ea22fb8ea6f8b4245e6ebbb403c41e5633805d8b832e3890af9ece71da52ba`（γ_median=0.3903236188504265, n=105）
- corrected `gamma_star_corrected.json` SHA-256: `f3174a508922de548e51368b4a3672f580272681c485670d6e2d1beefeafa0b4`（γ_median=0.396419312045252, n_nonidentity=105, p25=0.3735, p75=0.6546, min=0.2105, max=2.2244；含 gcv_grid_boundary_warning 字段，2026-11-06 条件 B 扩展后脚本导出）

**Required validation（四项一致性检查，通过前 B1 formal run 继续 LOCKED）**：

1. **Cell 集合对账**：更正前后相同的 105 个 nonidentity cell（20 identity skip 不变）；导出脚本内建 PAIRING_GATE（`count(split==0)==n_train`）105/105 通过。✅ 2026-11-06
2. **GCV 来源对账**：λ_GCV、候选网格（10^expo×scale, expo∈[−4,4] 33 点）、拟合行集均直接读自封板 `bsab_b0/cells.csv` 与 capture npz，未因修复改变（split 行集选择除外——即勘误本体）。✅
3. **Split 编码 artifact 证据**：`0=train / 1=test` 由 capture manifest JSON（n_train/n_test/n_rows）+ 105/105 行数对账确认，不依赖代码注释。✅
4. **新值独立复现**：修正脚本重复导出两次 SHA-256 逐位一致（`f3174a50…`，条件 B 扩展后脚本版本）；此前 inline 独立计算与脚本导出同值 0.396419。✅

**连带同步（已完成）**：`Models/DSAR.py`（常数+注释+版本）、RUNCOMMANDS §34.4 冻结项 2 / §34.4a Engineering constants / §34.4b G2_REFIT 断言三处同步为 0.396419，避免文档与代码保留两个不同正式 γ\*。

**验证闭环（2026-11-07）**：四项一致性检查通过后，CPU 数值断言重跑全绿——新旧 γ\* 在 125 cell 判决**零翻转**（FITTED 105 / IDENTITY_GUARD 20 不变；λ +1.56% 未触碰任何 guard 离散边界）；外部独立配方复刻 1e-9 对账；对抗扫描 full-wins/r8-loses 样本仍被新 guard 正确拒绝；`G2_IDENTITY_TIE` 全量 cell 成立。G1 在新 γ\* 下重跑 **PASS（15/15）**，宿主 `cifar_base_p15/seed_42`；旧 γ 版本 G1 证据归档为 `dsar_g1_invariance_gamma0.390324_superseded.json`（历史工程证据保留）。§34.4c 勘误验证链闭合，γ\* 相关 LOCKED 条件解除；B1 formal run 的其余前置（G2 运行时断言等）按 §34.4 原时间线不变。

#### 34.5 INR B0-lite 判决：RED，裁剪 INR 臂（2026-11-06 封板）

**执行**：izfree `--no_repro` 全量 125 cell（repro gate 按 §34.3 裁除；zref PASS 25/25）。**VERDICT: RED**：ρ mean=0.315 / pooled=0.418，t1=−0.109、t4=−0.226，worst 全部集中 c4（ρ −6.0~−1.8）。

**归因（分解诊断后确认，A+C 混合；B 排除）**：

| task | 损伤 | bridge 修复 | m2pp 修复 | 桥修复率 |
|---|---|---|---|---|
| 0 | 0.68 | +0.50 | +0.64 | 78% |
| 1 | 1.23 | **−0.15** | +0.42 | 塌 |
| 2 | 2.28 | +0.81 | +1.48 | 36% |
| 3 | 0.76 | +0.66 | +0.66 | 87% |
| 4 | 1.82 | **+0.09** | +0.82 | 5% |
| pooled | ~1.35 | +0.38 | +0.80 | **~28%（cifar 82%）** |

- **A（ρ 噪声主导）**：INR 损伤本小（0.68–2.28pp vs cifar 2.85/4.69pp），分母普遍 <1pp ⇒ 64/125 负分母、ρ 爆炸值（−6.0）为噪声；ρ 指标在 INR 多数 cell 不适用。
- **C（实质弱点，raw 增益口径确认）**：损伤大的 task（t1/t4）桥反而修不动（负/近零）——与 cifar"损伤越大修复率越高"相反。修复率 28% vs 82% 是结构信号非噪声。
- **损伤小的机制归因（位移对照，同口径 125 cell npz）**：INR 位移 mean 0.010 vs cifar 0.0148（小 32%）+ 头敏感度 ~135 vs ~193 pp/unit（钝 30%）⇒ 位移小一半贡献 + 头钝感一半贡献。协议根源：INR 40 全私有类/每任务 8 类/类空间压力小（cifar 75 类区段+公共类冲突大）。
- **修复率低的候选机制**（未定论，事后可查）：~650 行/cell 拟合 768 维仿射欠定更严重；损伤方向或大部分不在仿射桥可达跨度（B_pre RUN_01 镜像）。

**处置（预注册 RED 条款执行）**：
1. **INR 臂裁剪**：DSAR 不做 INR 端到端部署（收益上限 <0.8pp 撑不起结构成本）；论文 scope 声明"部署于损伤显著的数据集"，INR 作 limitation/边界案例诚实报告（损伤小 + 仿射不可达）。
2. **cifar B1 照常推进**（§34.4 不受影响）。
3. INR capture/诊断材料保留（cells.csv + verdict.json + 位移统计），供论文 limitation 段与审稿答复。

### 34.6 INR Rescue Amendment — Stage 0 只读诊断（2026-11-07 执行与读数；Stage 1 冻结待裁决）

**授权与边界**（GPT 2026-11-07 裁决）：批准 Stage 0 工具扩展与只读诊断；与 CIFAR γ\* 勘误两轨道独立并行。目标从"确认 INR 局限"修订为"修复 INR"（B0-lite RED 只证明 §34.2 残差桥参数化不足，不禁止其他修复算子）。B0 GREEN 与 §34.5 RED 封板**不重开**；Stage 0 仍是探索性诊断，**不直接解锁 INR 正式 Rescue run**——正式部署前必须先完成 Stage 1（rank、γ\*_INR、L guard、判决口径冻结）。

**条件 A（knee 规则，冻结）**：R_r=(MSE_I−MSE_r)/(MSE_I−MSE_full)≥0.9 为 knee 判据；聚合口径 = 仅对有效正分母 cell（容差 1e-6·MSE_I）取固定宏平均，同时上报 n_valid、median、p10 与全量 guard-active rate；无 rank 达标 → 记 `NO_KNEE_FOUND`，不得临时降阈值。Stage 0 的 test accuracy **只作机制解释，禁止作为 rank 选择依据**。

**条件 B（γ\*_INR 多统计，冻结）**：导出须含 γ\*、n_nonidentity、p25/p75（IQR）、min/max 及 GCV 网格两端命中数；nonidentity 命中率 >20% → 记 `GCV_GRID_BOUNDARY_WARNING`，median 不直接采信，须先裁决是否修改校准协议。

**执行记录**：izfree，`INR_RUN=output/ImageNet-R/fedta/inr_bsab_capture_M2matched_p40/seed_42`（find 核实，非猜测），125 cell / rounds 0–24 串行三命令：① `bsab_b0_verdict.py --no_repro` 重跑（新增 `bridge_lam_expo` 列）→ ② `bsab_gamma_star.py`（canonical fit_bridge 口径）→ ③ `dsar_birth_preview.py` 全量。

**Stage 0 读数（preview overall，125 cell）**：

| 指标 | 值 | 判读 |
|---|---|---|
| acc_raw / damaged / ceiling | 83.29 / 81.95 / 85.10 | 损伤 1.34pp，headroom 1.80pp |
| bt_full / bt_r8 | 82.74 / 82.12 | B(full) 仅修 25% 损伤-天花板跨度 |
| lt_r8 / r16 / r32 | 83.70 / 83.98 / 84.08 | **L 修 68% 跨度**，r32 仍在上行 |
| 最佳 chain (lt32_btfull) | 83.62 | **< L 单独 84.08**，组合不可加 |
| guard active rate | r8 0.48 → full 0.01 | 部署 r8 会被近半 cell 拦为 identity |
| R_r / n_valid | NO_KNEE_FOUND；**1/125** | **分母退化**（见结论 1） |
| 谱 | eff_rank_90=343.2；cum_energy r32=0.13, r256=0.69 | 损伤高秩，低秩 B 结构性不足 |
| GCV 边界命中（canonical fit_bridge） | 11/105 = 10.5% < 20% | 无 `GCV_GRID_BOUNDARY_WARNING`（preview fit_bt 口径 21 为另一实现，仅参考） |
| 位移 | c0 0.15 vs c1–c4 2.10–2.29 | KD 客户端高位移高秩 |

**五条结构结论**：

1. **R_r 框架在 INR 结构性退化**：124/125 cell 的分母（MSE_I−MSE_full）≤容差——full-ridge 在 held-out cal 行上对 identity 的 MSE 改善≈0。`NO_KNEE_FOUND` 的成因是参考系失效而非 rank 不足；互锁证据 = 高秩谱（343）+ 欠定（n_fit 均值 510 < d=768）+ 高位移。
2. **INR 广播损伤在 cell 尺度上不可被线性残差桥泛化修复**；bt_full 的 +0.78pp 准确率收益与 MSE 退化并存，说明其来自对 head 判别面敏感的微小系统性偏移，非行级重建。
3. **L 主导当前任务修复**（68% vs B 的 25%），且 L rank 收益未饱和（r8→r32 再 +0.38）。
4. **B→L 组合收益为负**：所有 chain 低于 L 单独（task_1 甚至低于 damaged）——B 与 L 针对重叠损伤，B 在 L 修复后的特征上过度校正；§34.6 组合可加性假设被否决。
5. **task 异质性**：t2/t4 损伤大且修复空间大，t1 损伤小且所有手段最弱。

**γ\*_INR 与 B0 重跑补录（2026-11-07）**：

- **γ\*_INR = 2.05791**（median，n_nonidentity=105；p25=2.03 / p75=3.6 / min=0.000115 / max=6.44；GCV 边界命中 11/105=10.5%<20%，条件 B 通过无警告）。为 CIFAR γ\*（0.396419）的 5.2 倍——per-dataset 校准必要性坐实。min 离群（低 median 近 5 个数量级）对应边界命中 cell 的近无正则解；median 对其稳健。
- **B0-lite 重跑一致性 PASS**：rho mean=0.315 / pooled=0.418 / task 分布（t3 0.991、t1 −0.109、t4 −0.226）/ worst cell（c4 r09 −6.0 等）与 §34.5 封板判决逐项一致，VERDICT RED 不变；additive `bridge_lam_expo` 列未扰动四臂——重跑验收条件满足。

**状态（2026-11-07）**：Stage 0 完整（① verdict 一致性 + ② γ*_INR + ③ preview 读数），待 GPT 对完整包做 Stage 1 冻结裁决。Stage 1 方向初判：B 臂降级候选（低秩覆盖不足 + full 泛化失败 + 组合负收益），L 臂为 INR Rescue 主力（rank 候选 r16/r32），修复主战场转向 L + 历史任务路径。**guard 判据的方法学注意**：R_r 分母退化（结论 1）同时意味着 cal-MSE 对 INR 损伤不敏感——"guard 放行"≠"有效修复"，INR 的 guard 判据是否需换口径（如部署态 CE）列入 Stage 1 裁决议题。INR 正式 Rescue run 继续 LOCKED。
