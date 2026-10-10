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

## 9. 诊断工具（不改变正式训练）

诊断脚本通过共享的 `build_server()` 复刻训练的 RNG 流（seed → model default_cfg → 数据划分 → 建客户端），保证得到与训练完全相同的 client 数据划分。所有命令在项目根目录执行；`<run_dir>` 指 `output/{data_name}/fedta/{run_name}/seed_{seed}`。

### 9.1 提取特征（z_ref / z_op）

```bash
python diagnostics/extract_features.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --checkpoint output/cifar100/fedta/<run_name>/seed_42/checkpoints/task_04_end.pth \
    --out_dir diagnostics_output
```

- 输出 `diagnostics_output/features_{data_name}_{ckpt名}.npz`（z_ref = frozen ViT 特征、z_op = 当前 prompt 增强的 pre-anchor 特征；含 labels / client_ids / task_ids / round_ids / splits / **sample_idx_relative**——`data_split_indices` 存储的**值**（client.train_data[t] 的索引），raw dataset index 不可恢复；与 precheck per-sample CSV 同坐标系，(client, task, sample_idx_relative) 可直接 join）+ 同名 `.json` 元信息（含 round、client_class_masks、public classes、**deterministic 运行态实读记录**）
- **正式 Phase 1 协议默认值：`--split train`、`--batch_size 16`**（z_op 依赖 batch 组成，B 必须与 precheck 通过的取值一致；`--split` 的 test/both 仅限非诊断用途，多模态分析侧已硬性只接受 train）
- **deterministic extraction**：与 precheck 脚本同协议（CUBLAS env 前置 + cudnn flags 在 bootstrap 后设置），保证正式提取与 precheck 测 floor 时的执行条件一致
- `--checkpoint` 缺省时自动使用 `checkpoints/latest.pth`；`--data_path` / `--device` 可覆盖 args.json 中的值

### 9.2 多模态分析 —— 已删除（已测验失败）

- **Phase 1 多模态方向判负（2026-10-10）**：robust multimodal prevalence 0/25（ρ、D_cover、client 归因、切分形态、z_ref 对照五证据链全阴性），halt criterion 触发，Phase 2 主线（Semantic-K2 / Oracle-K2 / UOT）冻结。存档见 FEDSMTA_ROADMAP.md §11；原命令与完整说明见 git 历史。

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

### 9.3.1 4 组合 per-class 通道分解（public vs private × 4 组合，联邦独有机制判决实验）

```bash
python diagnostics/analyze_class_retention.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --old_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth \
    --new_ckpt checkpoints/task_04_end.pth \
    --out_csv diagnostics_output/class_retention_4combo.csv
```

- 对每个 (client, old task) 评估 **4 组合**（old/old、old/cur、cur/old、cur/cur，顺序与 compatibility_drift 一致，旧 task 始终用 task-specific Chead），做 **per-class** 精度评估（复刻 evaluate 的 mask/target 语义，`predicts`/`target` 均为全局 class id）
- 通道定义：P 通道 = oo−co（聚合 prompt 漂移，联邦独有候选）；KA 通道 = oo−oc（本地 key/anchor 漂移，架构固有）；总漂移 = oo−cc
- public/private 身份由 `outer_split_manifest` 自证：出现在 >1 个客户端的类 = public（协议定义），缺失 manifest 时直接 RuntimeError
- 固定 shuffle 种子 `9700+100*cid+old_task`（与 compatibility_drift 一致，四组合同 batch 顺序）；全程 RNG snapshot/restore；评估后恢复 current 组件
- 判决目标：public 类超额损伤（相对 private）若集中在 P 通道 → 聚合介导的跨客户端接口失配坐实（联邦独有机制）；若同样在 KA 通道 → 机制架构固有，联邦只是调制项，论文降级措辞
- 聚合必须样本加权（本地测试集 Dirichlet 切到类级）；脚本 per-client 值与 compatibility_drift 输出精确对账
- 输出 CSV 列：client_id, old_task, class_id, scope, n_clients, n_test, prompt, key_anchor, acc（每类每组合一行）；console 汇总按 (task, scope) 样本加权通道分解 + public/private 判决 + 各客户端任务构成

### 9.3.2 子机制判别探针（公共类 KA 超额损伤的归因，消费 9.3.1 的 CSV）

```bash
python diagnostics/probe_channel_correlates.py \
    --csv diagnostics_output/class_retention_4combo.csv
```

- 在 4combo CSV 上按 (client, old_task, class) 单元计算 `KA_c=oo−oc`、`P_c=oo−co`，检验两个候选子机制：(i) 间接共适应（匹配强度下 public KA > private、KA_c~P_c 正相关）；(ii) 协议结构性脆弱（KA_c~n_test 负相关）
- 交互表：n_test 分桶（1-15/16-40/41-80/81-171）× scope 的加权 KA/oo，plus 匹配强度对比（public n_test≥40 vs private all）与逐客户端对照
- 汇总指标另存 `channel_correlates.json`（与输入 CSV 同目录）
- 判决（seed42）：(ii) 否证（方向相反：r=+0.346，n_test≤15 公共类近乎免疫 wKA=+1.8）；(i) 修正版成立（匹配强度下 public +19.7 vs private +11.4，4/5 客户端，P 通道协同抬升）

### 9.3.3a 路由结构识别（Phase 4 D0：gateway，先于 D1 的有效性检查）

```bash
python diagnostics/verify_slot_binding.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --task_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth,checkpoints/task_04_end.pth \
    --out_csv diagnostics_output/slot_binding.csv \
    --out_slot_csv diagnostics_output/slot_binding_slots.csv \
    --out_npz diagnostics_output/slot_binding_matrices.npz
```

- **动机**：`Tail_Anchor.forward(x, class_mask)` 的 class_mask 形参在函数体内未使用——路由纯相似度、无显式类绑定；slot i ↔ class i 只能涌现形成。D0 是**描述性路由结构识别实验**，输出决定 D1 用单槽位归因、加权多槽位归因还是完整路由分布分析；脚本本身不输出机制结论
- **固定输入协议**：train_data 带 RandomResizedCrop+HorizontalFlip——先在固定种子下物化**一份**增强视图（张量缓存），t_end 状态与部署态（task_04_end prompt+key）两次路由共用同一批输入 → routing_stability 纯测模型状态变化，增强噪声完全隔离
- 四个维度输出：①绑定集中度（diag_share、modal_share、routing entropy）；②槽位共享（within-task + **cross-task** purity/sharing——task 3 抢 task 0 槽位是劫持核心形态，within-task 看不见）；③**路由 2×2 分解**（R_oo/R_oc = key 漂移、R_oo/R_co = feature/prompt 漂移、R_oo/R_cc = 总 routing flip + 非可加性交互——与 accuracy 2×2 同坐标系，直接对桥 KA/P 通道分解）；④完整 class×slot 绑定矩阵 B（.npz：per task + per client 全序列 + top 跨任务争抢槽位表）
- 槽位池大小取自 key.shape[0]（200，含 100–199 未训练噪声槽）；prompt 恢复在 finally 内；空结果硬报错
- **无自动判决**：0.9/0.8 等阈值仅作描述性分组；人工读取三个输出文件后决定 D1 归因形态

### 9.3.3b D1 v2 离线碰撞暴露诊断（collision_exposure_diagnostic.py，零前向）

```bash
python diagnostics/collision_exposure_diagnostic.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --task_ckpts checkpoints/task_00_end.pth,checkpoints/task_01_end.pth,checkpoints/task_02_end.pth,checkpoints/task_03_end.pth,checkpoints/task_04_end.pth \
    --binding_csv diagnostics_output/slot_binding.csv \
    --binding_npz diagnostics_output/slot_binding_matrices.npz \
    --combo_csv diagnostics_output/class_retention_4combo.csv \
    --out_csv diagnostics_output/collision_exposure.csv \
    --out_json diagnostics_output/collision_exposure.json
```

- **输入**：D0 的绑定 CSV + npz 矩阵、5 个 task_end checkpoint（只读 key/anchor 张量）、4combo 损伤 CSV
- **机制链检验**：E_future（未来任务流量对旧类接口的碰撞暴露）→ d_key/d_anchor（p(j|c) 加权接口位移）→ KA_c（oo−oc 损伤），全链 Spearman 相关
- **A2 设计岔路口**：支撑槽按未来流量 F_j>0 / F_j=0 分层的加权位移对比——劫持介导 vs 衰减介导两种损伤路径的分离；另输出 D_traffic（流量加权位移 Σ p(j|c)·(F_j/ΣF)·||ΔK_j||），区分"被访问"与"被重度访问"
- **stabK=0.00 sanity**：旧类 modal slot 的未来流量（modal_future_traffic）——旧接口是被复用（劫持）还是被废弃（衰减）
- **CEI_all**（GPT 形式争抢指数）与 E_future（因果方向的前瞻版）双口径；d_key_next 单窗口位移（事件驱动检查）；public/private 分层
- **key 位移方向/范数分解**：d_key_dir（l2 归一化后 1−cos，路由相关量）+ d_key_norm_ratio（范数比，wd 存活签名）——D1 v2 FE 判决（2026-10-10）：raw ||ΔK|| 是盲指标（hit/nohit 比值 0.91×，耦合 wd Adam 使未命中槽方向随机游走+范数塌缩、命中槽被梯度钉住，两者产生相当的 raw 位移）；d_key_dir 是逐出相关量（vs modal_preserve FE rho=−0.622），脚本内打印的链相关为 pooled、判决须过 (client,task) FE（见 Roadmap §6.0 判决）
- 纯张量/CSV 运算，约 1 分钟，无 GPU 前向；描述性审计，因果主张待 A2 干预

### 9.3.3c 落点分析（analyze_landing_spots.py，本地离线，检验集中坍缩/attractor 假说）

```bash
python diagnostics/analyze_landing_spots.py \
    --binding_npz diagnostics_output/slot_binding_matrices.npz \
    --damage_csv diagnostics_output/collision_exposure_local.csv \
    --out_csv diagnostics_output/landing_spots.csv \
    --out_json diagnostics_output/landing_spots.json
```

- **输入**：D0 重跑 npz（含 `_oc` 落点矩阵：同一批固定增强视图的旧特征在最终 key 下的路由）+ D1-lite 损伤 CSV（158 class-units，已验证 158/158 modal 对齐）
- **预注册判读**：R1 落点暴露 H_landing vs KA_rel（FE）；R2 support_preserve 保护因子；R3 强绑定类 all-or-none 双峰性；R4 跨类落点坍缩
- **已得判决（2026-10-10）**：R1/R5 证伪（落点去向无预测力，与支撑集暴露同死）、R2 存活（support_preserve FE rho=−0.197, p=0.014，迄今最强类级保护因子）、R3 单边坍缩（90.4% 强绑定类 modal_preserve<0.2，但损伤量级远小于坍缩量级）、R4 强确认（32 旧类 modal 坍缩到 4–8 个吸引子槽，top-10 落点槽承载 86–96% 质量，64.6% 落入未训练噪声区）
- 秒级本地运行，无 GPU；描述性审计，因果主张待 A2 干预

### 9.3.4 TKR 精确口径复算（verify_tkr.py，零 GPU，消费 train_log.csv）

```bash
python diagnostics/verify_tkr.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --out_prefix diagnostics_output/tkr_seed42
# 或直接指定 train_log.csv：
python diagnostics/verify_tkr.py \
    --train_log <path>/train_log.csv --global_epoch 5 \
    --out_prefix diagnostics_output/tkr_seed42
```

- **目的**：按 FedTA 作者 2026-10-10 issue 回复的定义复算 TKR——分母 = round4 **聚合全局模型**在 task 0 本地测试集精度（`phase=server, test_task=0, round=4` 行），分子 = t>0 后**本地模型**回评 task 0（`phase=local, test_task=0, round=t*5+4` 行，本地训练后、聚合前的状态）
- **注意**：task-end checkpoint 存的是广播后 prompt，无法重建分子态，train_log.csv 是唯一数据源（要求 run 开着 instrumentation）
- 输出三口径（author local/global、local/local、server/server）× 两种聚合（ratio-of-means、mean-of-ratios）+ t=0 client-drift premium
- **已得判决（2026-10-10，seed 42）**：author 口径 = 84.6/92.2/95.3/95.6%——**未复现论文 105–115%**，local/global prompt 不对称解释被实证否证（Roadmap §4.10）；server/server 口径与此前部署态 KRt（88.6/93.0/90.1/90.7）精确吻合，交叉验证通过

### 9.3.3 slot 级碰撞诊断 —— 已删除（已测验失败）

- **原版 D1（slot_collision_diagnostic.py）已被 D1 v2（§9.3.3b collision_exposure_diagnostic.py）取代**：其"slot id ≈ class id"单槽归因设计在 D0 判死后即不可用（diag_share=0.000），且核心假说（hits→位移→损伤线性劫持链）被 D1-lite / D1 v2 FE 判决证伪。存档见 FEDSMTA_ROADMAP.md §11；原命令见 git 历史。

### 9.4 Synthetic Oracle-K2 压力测试 —— 已删除（方向已冻结）

- **所属 Phase 2 主线（多模态 K2 anchor）随 Phase 1 判负冻结（2026-10-10）**，不再执行。存档见 FEDSMTA_ROADMAP.md §11；原命令见 git 历史。

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

## 14. Phase 1-precheck（z_op batch sensitivity）—— 已删除（已测验失败）

- **服务对象（Phase 1 多模态分析）已判负（2026-10-10），precheck 不再执行**。脚本 `check_feature_batch_sensitivity.py` 保留在仓库，其 deterministic 设置顺序教训（cudnn flags 必须在 `bootstrap_server()` 之后设置）已并入工程约定。存档见 FEDSMTA_ROADMAP.md §11；原命令见 git 历史。

## 15. Freeze 后的硬约束（长期有效）

- 从 tag 起，`--method fedta` 视为 **read-only scientific baseline**：不得改动 FedTA loss / optimizer / BGPS / SIKF / FedAvg head / Tail Anchor 结构 / prompt 逻辑 / CIFAR 联邦划分
- 当前活跃路线：**Phase 4 接口稳定性**（FEDSMTA_ROADMAP.md §6）——A2 干预在新方法 flag（`--method fedta_a2`）下实验；已测验失败方案的命令已从本文件移除，总存档见 FEDSMTA_ROADMAP.md §11

## 16. Phase 4 修复实验（A2 训练臂 + A1 快照评估）

> A2 三臂各自跑一次完整训练（control 复用既有 seed42 baseline run，不新增计算）。
> A1 为纯推理零重训实验，baseline run 跑完后几分钟即可出结果。
> 实现机制：post-step restore（等价 freeze + wd 豁免），官方 FedTA 优化器零改动；
> 保护集 = 已完成任务训练特征经当前 key top-1 路由的支撑槽位并集（D0 定义，运行时计算）。
> 判读标准（预注册，roadmap §6.2）：anchor 臂 KA_rel 显著下降 = 主通道因果成立；both 臂逼近 co 上界且 cc 不受损。

```bash
# A2 三臂训练（CIFAR-100，seed 42；逐臂依次运行）
python main.py --method fedta_a2 --a2_arm anchor --seed 42 --global_epoch 5 --threshold 0.25 --data-path <DATA_PATH>
python main.py --method fedta_a2 --a2_arm key    --seed 42 --global_epoch 5 --threshold 0.25 --data-path <DATA_PATH>
python main.py --method fedta_a2 --a2_arm both   --seed 42 --global_epoch 5 --threshold 0.25 --data-path <DATA_PATH>

# ImageNet-R（固定 surrogate_num 5）
python main.py --method fedta_a2 --a2_arm anchor --seed 42 --global_epoch 5 --threshold 0.25 --surrogate_num 5 --data-path <DATA_PATH>

# A2 smoke（先验证接线：1 任务应输出 protected 0/200 slots 并与 control 无差异）
python main.py --method fedta_a2 --a2_arm both --seed 42 --task_num 1 --global_epoch 5 --local_epoch 1 --run_name a2_smoke

# A1 快照评估（纯推理；对 baseline run 或 A2 run 均可运行）
python diagnostics/a1_snapshot_eval.py \
    --run_dir output/cifar100/fedta/<run_name>/seed_42 \
    --out_csv diagnostics_output/a1_snapshot_eval.csv
```

注意事项：
- `--a2_arm` 仅在 `--method fedta_a2` 下生效；`--method fedta` 路径零改动（A2 代码全部藏在 method flag 后）
- A2 checkpoint 含保护快照（a2_state），跨 arm resume 会被 arm 一致性检查拒绝（RuntimeError）
- A1 预期结果 = 2×2 co 组合预注册值（T1 71.4→93.9 等）；显著偏离时先查 batchwise_prompt 噪声（±2pp 内正常）
- A2 判读需配合 D1 脚本在 A2 run 上重算 KA_rel / 路由保持率，与 seed42 baseline 对照
