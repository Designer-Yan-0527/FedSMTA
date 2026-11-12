# FedRelink —— 项目工作记录（A2 + L + B）

> 记录生成于 2026-11-12。只收录与最终方案 **A2 + L + B**（FedRelink）直接相关的工作，包括支撑它的探针链。
> 判死方向（route_fix / PGC / PSC / M5 / DHT / A5 / 多模态主线等）与外部对比基线（FedProTIP / HePCo / GLFC / Powder / FedMGP / C²Prompt）的详细档案不在本记录，见 RUNCOMMANDS.md 各节与 git 历史。
> 方法名沿革：载体为官方 FedTA（CVPR 2025）→ 项目方法初名 `fedta_a2`/`fedta` → 2026-11-12 更名 **`fedRelink`**（四臂矩阵 = base / `--a2_arm key` / `--dsar` / 组合臂）。

---

## 1. 一句话方案

联邦持续学习（FCL）中 prompt-based 方法的表观遗忘，主体不是知识擦除，而是**路由接口跨世代失配**（A1 实验可恢复占比 86.4%）。我们将失配分解为三个独立损伤源，给出最小干预的三模块组合：

| 模块 | 损伤源 | 作用层 | 时机 | 保险 |
|---|---|---|---|---|
| **A2**（训练侧保护） | key/anchor 支撑槽位被新任务训练冲刷 | 训练侧参数保护 | 任务起点定保护集 + step 级回写 | 默认 off = 基线逐位一致 |
| **B**（DSAR 广播坐标桥） | server prompt 融合致 z⁺ 漂移（历史任务共享广播态） | 读出侧 | 每轮广播边界 per-client fresh-fit | paired-MSE identity guard |
| **L**（DSAR 读出适配器） | 当前特征 ↔ 出生/联训头失配 | 读出侧 | 同上，冻结头 CE 训练 | worst-cell 有界（无 CE guard，证据见 §4.4） |

全部零回放、零联邦通信增量（sidecar ≈ 102 KiB/客户端）、训练轨迹逐位中立（G1–G4 硬门）、最坏情况退化为 baseline。

## 2. 基线纪律（一切干预的地基）

- **Phase 0 三门全部 bit-exact PASS** 后冻结为只读科学基线（git tag `phase0-fedta-baseline`）：
  - **0A** 官方 FedTA 逐位 parity（D=0，global_epoch=5 / threshold=0.25）；
  - **0B** resume 等价（checkpoint+续训 ≡ 连续训练，含 outer_split_manifest 逐位校验）；
  - **0C** 观察者不变性（插桩 ON/OFF 轨迹逐位一致）。
- 方法学承诺：后续一切干预与官方基线**逐位对账**，差异只允许出现在声明过的位置。
- 划分真值冻结入库：`protocol_manifests/{cifar100_c5t5p15s20, inr_c5t5p40s5}_seed42.json`。

## 3. 核心发现：接口遗忘（interface-mismatch forgetting）

seed 42 表观遗忘 F=18.89pp，三个异常签名：retention 多处 >100%、遗忘事件驱动非单调（task 2 崩塌后自发恢复）、TKR 作者口径复算（96.5/98.5/96.2%）与表观值割裂。

**A1 干预实验**（`a1_snapshot_eval`：推理时把旧任务特征换回出生世代 key/anchor）逐任务分解 F = 接口可恢复 + 残余：

| 任务 | 总遗忘 | 接口可恢复 | 占比 |
|---|---|---|---|
| T0 | 8.59 | 7.39 | 86% |
| T1 | 25.17 | 22.75 | 90% |
| T2 | 20.22 | 13.38 | 66%（残余最大 = 事件任务 anchor 污染） |
| T3 | 21.57 | 21.81 | 101% |
| **均值** | **18.89** | **16.33** | **86.4%** |

多 seed 稳健性（三 seed）：对角线 93.8±0.3、F 11.1±6.8、AA 85.1±5.7——波动全部来自旧任务保持。

## 4. A2：训练侧接口保护

### 4.1 发现与定位链（探针）

1. **compatibility_drift 2×2**（新旧 key/anchor 交叉换装）：确认损伤来自接口世代差。
2. **D0 `verify_slot_binding`**：槽位-类绑定是涌现的（路由纯相似度、无显式类绑定）→ 归因前必须先验证绑定。
3. **D1 v2 `collision_exposure_diagnostic`**（固定效应回归 + `analyze_landing_spots` 落点分析）：未来流量碰撞、槽位劫持、落点去向、路由不稳定、key 直接因果**五条直觉损伤路径全部判死**；唯一存活通道 = anchor 位移（FE rho=+0.259, p=0.001）→ 机制修正为 emergent interface sharing。
4. **A2 预注册四臂干预（决定性）**：干预推翻相关性归因——**key 单冻臂**才是有效通道：
   - F 18.89 → **7.11**（削减 63%）、AA 78.62 → **87.09**（+8.5pp）、可塑性代价仅 0.6pp；
   - anchor 臂有害（F 恶化 5.5pp，破坏 key-anchor 共适应）；both 与 key 打平但多付可塑性；
   - A1 上界 91.67，A2 捕获其 68%；残余 = prompt 漂移（→ 由 B/L 接手）+ c4 路由碰撞。

### 4.2 机制（frozen Phase-4 spec，commit f527ab7）

- 任务起点：RNG 中立特征回访，已完成任务 train 特征经**当前 key** top-1 路由命中的槽位取并集 = 保护集（D0 涌现支撑）。
- 训练期：每个 phase-2 optimizer step 后回写保护槽位（freeze + weight-decay 豁免；≡ freeze + wd 豁免，不改官方优化器与损失）。

### 4.3 工程封印与沿革

- checkpoint 携带 `a2_state`；resume fail-closed（arm 不匹配即 RuntimeError）；开关门 `--a2_arm {off,key,anchor,both}`，默认 off = 与基线逐位一致；与 `--dsar` 正交组合。
- 沿革：11-02 大清理中旧 `--method fedta_a2` 方法线入判死档案（laziness check 否决）；**11-08 从 f37590d^ 恢复 frozen Phase-4 规格并改为开关门实现**（开关语义、可与 DSAR 组合）。新四臂矩阵是 A2 的现役证据载体，不复用 10 月旧四臂数字。

## 5. DSAR = B + L：读出侧修复

### 5.1 发现链（探针）

1. **读出侧 headroom 确认**：M2' 头重训 CIFAR +2.36pp（corrected 93.07）；**oracle headroom probe**（`probe_oracle_headroom.py`）上界 96.21（CIFAR）/ 84.82（INR）——一大块遗忘不动 backbone 即可拿回。
2. **损伤时点定位**（§27.7 归因阶梯探针）：2pp 头滞后真凶 = 训练-评估特征时点错位（**HEAD-LAG**）→ §28 post-broadcast refit 判据臂落地结论。
3. **机制链推进**（Probe-G `probe_prompt_content_drift` + Probe-H `probe_prompt_step_calibration`，CIFAR+INR 双数据集判决）：同槽 prompt 内容漂移与时间步校准的机制解剖，直接引出 BSAB 立项（Broadcast-State Alignment Bridge）。
4. **birth pair 可测量化**（§34.2 **`BSAB_PAIR_CAPTURE`** instrumentation-only 插桩，2026-11-04 实现并单测）：每广播边界缓存同行 (z⁻, z⁺) 配对 + prompt/head/row-index 哈希 + 全行 train/test 缓存；四流 RNG 快照/恢复；OFF = 逐位 no-op，capture run ≡ baseline 的环 2 identity gate 验证（M2'-matched 冻结口径）。INR 的 `inr_bsab_capture_baseline_p40`（izfree）已跑完，兼作 INR 基线臂证据（instrumentation-only、与 baseline 逐位一致）。
5. **B_pre aging probe**（`probe_bsab_feasibility.py`，只读 M2' artifacts）：跨任务累积漂移被免标签坐标桥大幅可迁移（32/37 valid cell ρ_age≥0.5）→ 免标签桥先验成立。
6. **B0 birth probe（决定性）**：125 cell 四臂离线判决 **GREEN**——ρ_birth mean=1.006 / pooled=1.018；identity 94.83 / **bridge 96.64** / 参照 z⁻ 97.24 / M2'' 96.57；免标签桥恢复量达到监督重训参照水平；20/125 cell GCV 自动选 identity = 安全阀有效。
7. **B1 rank 冻结**：r8（~1% 参数）= 87% 修复量，r64 仅 +0.03pp → **冻结 r8**。

### 5.2 机制（冻结规格 §34.4/§34.4a）

- **B_t 广播桥**：每轮每客户端 fresh-fit `B(z)=z+ΔAz+b`（残差 ridge，λ=N·s·γ*；SVD 截断 r8 + 截断后 bias 重算 b₈）；无标签 identity guard：80% 拟合 → 20% 校准 paired-MSE，桥不赢则严格 B=I。
- **L_t 读出适配器**：rank-8 残差参数化，冻结头 CE 训练（100ep/bs64/Adam/lr1e-3/wd1e-4）。
- **部署图**：当前任务 `ŷ_t = TA_t(L(B(z⁺_op)))`；历史任务 `ŷ_k = TA_k(B(z⁺_op))`（B-only，禁 L）。训练前向不经 B/L。
- **γ\* 冻结**：CIFAR 0.396419（含 §34.4c train/test 行集勘误，四项一致性检查闭环，125 cell 判决零翻转）。

### 5.3 INR 支线：数据集条件配方（§34.5–34.7）

- B0-lite 在 INR 判 **RED**：修复率 28% vs CIFAR 82%；机制 = 位移小 32% + 头钝感 30% + 高秩谱（eff_rank 343）+ 欠定（n_fit≈510 < d=768）。
- Stage 0/1 救援三轮离线证据（全部 baseline 联训头 = 论文口径）：
  1. 方案网格：**L8_zp_no +2.361 te-pooled 全场第一**；B24 +1.13~1.16 过门；无 guard min-norm 灾难（−5.5~−8.3）→ 修的正是读出失配；
  2. **guard 反相关分解**：Q2 CE guard 拒绝的 16% cell 真实增益 ≈ +11.5pp——guard 恰好杀最大赢面（CE 惩罚自信错误，identity 穿失配头输出低置信垃圾 CE 反而不高）→ **L 无 guard 部署**，改用 worst-cell 安全证据（L16_zp_no min=−1.20、p5=−0.56，有界温和）；
  3. **历史行探针**：B8_inr 唯一全距离为正——rank 交互在历史行反转（低秩外推稳）→ **B 保 r8**。
- **INR 冻结配方 `dsar-inr-v1`**：γ\*=2.0579118185313345（provenance 6cd83d9）、L rank 16、B rank 8、L 无 guard、dispatch 与 CIFAR 完全同构——论文叙事 = 同一框架、两个数据集参数。配方由 `--data_name` 自动分派，CIFAR 臂 sidecar meta `recipe: "dsar-b1-cifar"`，INR 臂 `"dsar-inr-v1"`（防串配方 fail-closed）。

### 5.4 在线验证现状（四项 GREEN 门：ΔAA≥+0.50 ∧ G_old≥+0.25 ∧ ΔF≤+0.25 ∧ minΔ_k≥−1.00）

| 数据集 | ΔAA | G_old | ΔF | minΔ_k | 判决 |
|---|---|---|---|---|---|
| CIFAR（cifar_b1_dsar_on） | +2.19 ✓ | ✓ | +1.60 ✗ | −1.56 ✗ | 2/4 |
| INR（inr_dsar_v1） | +1.02 ✓ | +0.525 ✓ | +2.56 ✗ | −0.68 ✓ | 3/4 |

- INR 对角线均值 +3.18（T3 出生 +5.31）、末行旧任务全部为正、每客户端 4 正 1 微负。
- 两处 ΔF 红门同构 = **峰值抬升伪影**（forgetting 是"从峰值跌落"的相对量，对角线被抬高后机械变大；绝对值上旧任务无退化）——处置待终审（metric critique 进 discussion 或换 retention 口径）。

## 6. 判死名单（design space falsification，A2+L+B 的反面证据链）

| 竞争方案 | 判死证据（全部预注册封板） |
|---|---|
| route_fix（修 key 几何） | E2 几何全绿但准确率不回升——修复几何 ≠ 修复分数 |
| PGC（原型几何先验） | 边界损失主体在头侧（HEAD-LAG +2.0~2.1pp） |
| 融合候选 A/B（改广播） | ΔA=−0.35pp、ΔB≡0——广播是稳定器，旧行侵蚀数学上不存在 |
| PSC（prompt 子空间巩固） | 8/8 全负 + laziness 否决 |
| M5（边界巩固臂） | 收益 0.5–1.2pp + replay 档位代价 |
| DHT（历史头迁移） | 可行性探针判死 |
| A5（共享语义接口/标准化头） | Gate A-2 后判死，11-12 随清理删除（`StandardizedChead` 分支） |

DSAR 胜出性质：① label-free（B 无标签，L 仅当前任务 train 标签）；② 零 FL 通信；③ 轨迹逐位中立（G1–G4，38/38 断言）；④ identity guard 安全阀；⑤ 低秩便宜；⑥ 直接命中已测 z⁻→z⁺ 损伤（ρ≈1 因果支撑）。

## 7. 探针/插桩清单（仅 A2+L+B 相关）

| 工具 | 性质 | 判决/用途 |
|---|---|---|
| `diagnostics/extract_features.py` | 只读，复刻训练 RNG 流 | z_ref/z_op 特征提取（一切下游探针的输入） |
| `diagnostics/compatibility_drift.py` | 离线 2×2 换装 | 损伤来自接口世代差（§4.1-1） |
| `diagnostics/analyze_class_retention.py` | 4 组合 per-class 分解 | 类保持通道分解（发现链辅助） |
| `diagnostics/verify_tkr.py` | 零 GPU 复算 | TKR 作者口径与表观遗忘割裂（§3 签名三） |
| `diagnostics/verify_slot_binding.py`（D0） | 零训练 | 槽位-类绑定 = 涌现，非显式 |
| `diagnostics/collision_exposure_diagnostic.py`（D1 v2） | 零前向固定效应 | 五直觉路径判死，anchor 通道唯一存活 |
| `diagnostics/analyze_landing_spots.py` | 离线落点 | D1 归因辅助 |
| `diagnostics/a1_snapshot_eval.py` | 只读重评 | A1 干预：接口遗忘 86.4%（§3） |
| `diagnostics/probe_oracle_headroom.py`（A5 Gate A-2） | 只读探针 | 读出侧上界 96.21/84.82（DSAR headroom 依据） |
| §27.7 归因阶梯探针 | 预注册判据臂 | HEAD-LAG 定位（§5.1-2） |
| `--bsab_pair_capture`（§34.2 hook） | instrumentation-only 插桩 | birth pair 可测量化；OFF=逐位 no-op；capture≡baseline（环 2 gate） |
| `diagnostics/probe_bsab_feasibility.py`（B_pre） | 只读 M2' artifacts | aging probe：漂移可迁移 32/37 cell |
| B0 birth probe（离线四臂） | 只读 capture 产物 | 判决 GREEN：免标签桥 ≈ 监督重训参照（§5.1-6） |
| INR Stage 0/1 离线网格 + guard 反相关分解 + 历史行探针 | 只读 | dsar-inr-v1 配方三要素的证据（§5.3） |
| Probe-G / Probe-H（`probe_prompt_content_drift` / `probe_prompt_step_calibration`） | 探针级诊断 | prompt 漂移机制解剖 → BSAB 立项前置（§5.1-3） |
| `evaluate_with_margin` margin stats | observer-only | 评估期观察，不进正式口径 |

其余探针（probe_routing_flip / probe_dht_feasibility / probe_boundary_attribution / PSC 短程判据臂 / Track E-B2 等）均服务于已判死方向，档案在 RUNCOMMANDS.md §21/§25–§30，不在本记录范围。

## 8. 工程与验证纪律

- **RunManager 四层产物树**：`output/{data}/{method}/{run_name}/seed_{seed}/`（args.json / command.txt / git_commit.txt / logs / checkpoints / metrics）；诊断产物按产出脚本归档 `diagnostics_output/`。
- **checkpoint 一致性**：`data_split_indices` 记录 70/30 划分；outer_split_manifest 逐位校验；RNG 状态最后恢复；原子写；resume `--resume auto`。
- **逐位中立性 gates**：G1–G4（38/38 断言）覆盖 A2/DSAR/capture 的默认与部署路径；每模块带单测（tests/：Gate 0A/0B/0C、A2/A1 回归、bsab_pair_capture、diag_utils_compat、global_head_rebuild、oracle_headroom_probe 等 29 项 CPU 单测）。
- **fail-closed 纪律**：resume arm 不匹配、配方串用、strict load 失败等一律 RuntimeError 拒绝，不允许静默兼容。
- **预注册纪律**：判据先于数据冻结；一切判读按预注册门执行，修订留审计痕迹。

## 9. 命名重构与清理（2026-11-12，commit a89bb7d）

- `--method` 与输出树 `fedta` → **`fedRelink`**；`--model_name` 收敛 `['Tail_Anchor']`；21 个 timm 连字符旗标统一下划线（`--data_path` 等，旧拼写 argparse 拒绝）；`vision_transformer__l2p.py` → `vision_transformer_l2p.py`。
- 删除 A5 死分支（`StandardizedChead` + rebuild 兼容路径，简化为 strict load fail-closed）与死代码文件。
- 双仓库推送：新仓 **FedRelink**（github.com/Designer-Yan-0527/FedRelink）+ 原仓 FedSMTA 同步。
- 验证：import / 29 项单测 / argparse 行为 / GPU base 冒烟 round 0 完整跑通（融合→广播→checkpoint，无 Traceback）；resume 冒烟验证到 checkpoint 加载 + test loader 重建 + round 1 启动。

## 10. 当前状态与剩余工作

**四臂矩阵进度**（正式口径 = 命令在 RUN.md）：

| 臂 | CIFAR-100 | ImageNet-R |
|---|---|---|
| baseline | ✓ | ✓（capture baseline 兼作） |
| A2（--a2_arm key） | 待发车 | 待发车（izfree） |
| DSAR（--dsar） | ✓ 2/4 门 | ✓ 3/4 门 |
| A2+DSAR | 待发车 | 待发车（izfree） |

剩余：
1. 双数据集 A2、A2+DSAR 四格发车（检验超加性，A2 论文最终判据）；
2. ΔF 峰值抬升伪影的措辞终审（metric critique / retention 口径）；
3. B1 verdict 工具（四项门 JSON + §34.4d 登记）。

部署：izfree 手册见 SETUP_IZFREE.md；外部对比基线见 RUN_BASELINES.md（另行执行，不在本记录）。
