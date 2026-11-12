# FedSMTA 论文大纲与研究素材梳理

> 生成于 2026-11-10。证据锚点：RUNCOMMANDS.md §34.x（DSAR/BSAB 全链）、§判死档案（git e7d7a42 及更早）、project_memory、commit f527ab7（A2 恢复）。
> 实验状态：CIFAR base ✓ / DSAR ✓（2/4 门）；INR base ✓ / DSAR ✓（3/4 门）；A2 与 A2+DSAR 两臂待发车。

---

## 0. 一句话故事

联邦持续学习（FCL）中 prompt-based 方法（以 FedTA 为载体）的表观遗忘，主体不是知识擦除，而是**路由接口跨世代失配**（A1 实验恢复 86.4%）。我们将失配分解为三个独立损伤源，给出最小干预的三模块组合方案 **A2 + L + B**：训练侧保护（A2，冻结旧任务支撑槽位）+ 读出侧修复（DSAR = B 广播坐标桥 + L 读出适配器），全部零回放、零联邦通信增量、训练轨迹逐位中立、最坏情况退化为 baseline。

---

## 1. 背景与问题发现（论文 §1–§2 素材）

### 1.1 载体与基线纪律

- 载体：FedTA（CVPR 2025）。核心结构：Tail_Anchor 路由接口（key 相似度路由 + anchor 原型池）、per-task 客户端头、BGPS/SIKF、FedAvg 头聚合。
- Phase 0 三门全部 bit-exact PASS 后冻结为只读科学基线（git tag `phase0-fedta-baseline`）：
  - 0A 官方逐位 parity（D=0，global_epoch=5 / threshold=0.25）
  - 0B resume 等价（checkpoint+续训 ≡ 连续训练，含 outer_split_manifest）
  - 0C 观察者不变性（插桩 ON/OFF 轨迹逐位一致）
- 方法学卖点：**后续一切干预都与官方基线逐位对账**（差异只允许出现在声明过的位置）。

### 1.2 核心发现：接口遗忘（interface-mismatch forgetting）

seed 42 表观遗忘 F=18.89pp，三个异常签名：

1. retention 多处 >100%（知识未被擦除）；
2. 遗忘事件驱动、非单调（task 2 崩塌后自发恢复）；
3. TKR 作者口径复算（三 seed 平均 96.5/98.5/96.2%）与表观值割裂。

**A1 干预实验**（推理时将旧任务特征换回出生世代 key/anchor）：逐任务分解 F = 接口可恢复 + 残余：

| 任务 | 总遗忘 | 接口可恢复 | 占比 |
|---|---|---|---|
| T0 | 8.59 | 7.39 | 86% |
| T1 | 25.17 | 22.75 | 90% |
| T2 | 20.22 | 13.38 | 66%（残余最大 = 事件任务 anchor 污染，两证据互证） |
| T3 | 21.57 | 21.81 | 101% |
| **均值** | **18.89** | **16.33** | **86.4%** |

多 seed 稳健性：三 seed 对角线 93.8±0.3（学新任务极稳）、旧任务终值 82.7±6.9、F 11.1±6.8、AA 85.1±5.7——波动全部来自旧任务保持。

### 1.3 三误差源 → 三模块映射（方案骨架）

| 损伤源 | 时机 | 模块 | 职责 |
|---|---|---|---|
| key/anchor 槽位侵蚀（旧任务支撑槽位被新任务训练冲刷） | 训练期 | **A2** | 训练侧预防 |
| 广播 z⁺ 漂移（server prompt 融合致特征位移；历史任务共享广播态） | 每轮广播边界 | **B**（DSAR 桥） | 读出侧修复，主场=历史任务 |
| 读出失配（当前特征 ↔ 出生/联训头） | 部署态 | **L**（DSAR 适配器） | 读出侧修复，主场=当前任务 |
| 保险 | — | identity guard | 最坏退化为 baseline |

方案原则：最小干预、零回放、零联邦通信增量（sidecar 全 client-local，B+L ≈ 102 KiB/客户端）、训练轨迹逐位中立（G1–G4 硬门）。

---

## 2. DSAR：问题发现、选择依据、机制（论文方法章 §3 素材）

### 2.1 发现链

1. **读出侧 headroom 确认**：M2' 头重训 CIFAR +2.36pp（93.07 corrected）；oracle headroom probe 上界 96.21（CIFAR）/ 84.82（INR）——一大块遗忘不动 backbone 即可拿回。
2. **损伤时点定位**（§27.7-1 归因阶梯）：2pp 头滞后真凶 = 训练-评估特征时点错位（HEAD-LAG）；§28 post-broadcast refit 落地结论。
3. **birth pair 可测量化**（§34.2）：`BSAB_PAIR_CAPTURE` 插桩 hook 缓存每广播边界 (z⁻, z⁺) 同行配对（四流 RNG 快照、OFF/ON 逐位中立性 gate、capture run ≡ baseline 环2 identity gate）。
4. **B_pre aging probe**：跨任务累积漂移被免标签坐标桥大幅可迁移（32/37 valid cell ρ_age≥0.5）→ 先验。
5. **B0 birth probe（决定性）**：125 cell 四臂离线判决 GREEN——ρ_birth mean=1.006 / pooled=1.018；identity 94.83 / bridge 96.64 / 参照 z⁻ 97.24 / M2'' 96.57。免标签桥恢复量达到监督重训参照水平；20/125 cell GCV 自动选 identity = 安全阀。
6. **B1 rank 冻结**：r8（~1% 参数）= 87% 修复量，r64 仅 +0.03pp → 冻结 r8。

### 2.2 方案选择：design space falsification（判死名单作为反面证据链）

| 竞争方案 | 判死证据（全部预注册封板） |
|---|---|
| route_fix（修 key 几何） | E2 几何全绿但准确率不回升——修复几何 ≠ 修复分数 |
| PGC（原型几何先验） | 边界损失主体在头侧（HEAD-LAG +2.0~2.1pp） |
| 融合候选 A/B（改广播） | ΔA=−0.35pp、ΔB≡0——广播是稳定器，旧行侵蚀数学上不存在 |
| PSC（prompt 子空间巩固） | 8/8 全负 + laziness 否决 |
| M5（边界巩固臂） | 收益 0.5–1.2pp + replay 档位代价 |
| DHT（历史头迁移） | 可行性探针判死 |

DSAR 胜出性质：① label-free（B 无标签，L 仅当前任务 train 标签）；② 零 FL 通信；③ 轨迹逐位中立（G1–G4，38/38 断言）；④ identity guard 安全阀；⑤ 低秩便宜；⑥ 直接命中已测 z⁻→z⁺ 损伤（ρ≈1 因果支撑）。

### 2.3 机制（冻结规格 §34.4/§34.4a）

- **B_t 广播桥**：每轮每客户端 fresh-fit `B(z)=z+ΔAz+b`（残差 ridge，λ=N·s·γ*；SVD 截断 r8 + 截断后 bias 重算 b₈）；无标签 identity guard：80% 拟合 → 20% 校准 paired-MSE，桥不赢则严格 B=I（裁决对象 = 实际部署的 r8 候选）。
- **L_t 读出适配器**：rank-8 残差参数化，冻结头 CE 训练（100ep/bs64/Adam/lr1e-3/wd1e-4）。
- **部署图**：当前任务 `ŷ_t = TA_t(L(B(z⁺_op)))`；历史任务 `ŷ_k = TA_k(B(z⁺_op))`（B-only，禁 L）。训练前向不经 B/L。
- **γ\***：CIFAR 0.396419（含 §34.4c train/test 行集勘误，四项一致性检查闭环，125 cell 判决零翻转）。

### 2.4 INR 支线：数据集条件配方（§34.5–34.7）

- B0-lite 在 INR 判 RED：修复率 28% vs CIFAR 82%；机制 = 位移小 32% + 头钝感 30% + 高秩谱（eff_rank 343）+ 欠定（n_fit≈510 < d=768）。
- Stage 0/1 救援三轮离线证据（全部 baseline 联训头=论文口径）：
  1. 方案网格：L8_zp_no +2.361 te-pooled 全场第一（M2' 口径判死 ↔ 论文口径复活 = L 修的正是读出失配）；B24 +1.13~1.16 过门；无 guard min-norm 灾难（−5.5~−8.3）。
  2. guard 反相关分解：Q2 CE guard 拒绝的 16% cell 真实增益 ≈ +11.5pp——guard 恰好杀最大赢面（CE 惩罚自信错误，identity 穿失配头输出低置信垃圾 CE 反而不高）→ **L 无 guard 部署**，改用 worst-cell 安全证据（L16_zp_no min=−1.20、p5=−0.56，有界温和）。
  3. 历史行探针：B8_inr 唯一全距离为正——rank 交互在历史行反转（B 在当前任务拟合、应用到历史特征是外推，低秩外推稳）→ B 保 r8。
- **INR 冻结配方（dsar-inr-v1）**：γ\*=2.0579118185313345（provenance 6cd83d9）、L rank 16、B rank 8、L 无 guard、dispatch 与 CIFAR 完全同构——论文叙事 = 同一框架、两个数据集参数。

### 2.5 在线验证现状（四项 GREEN 门：ΔAA≥+0.50 ∧ G_old≥+0.25 ∧ ΔF≤+0.25 ∧ minΔ_k≥−1.00）

| 数据集 | ΔAA | G_old | ΔF | minΔ_k | 判决 |
|---|---|---|---|---|---|
| CIFAR（cifar_b1_dsar_on） | +2.19 ✓ | ✓ | +1.60 ✗ | −1.56 ✗ | 2/4 |
| INR（inr_dsar_v1） | +1.02 ✓ | +0.525 ✓ | +2.56 ✗ | −0.68 ✓ | 3/4 |

- INR 对角线均值 +3.18（T3 出生 +5.31）、末行旧任务全部为正（+0.31~+0.81）、每客户端 4 正 1 微负（最弱 c2 +1.13）。
- 两处 ΔF 红门同构 = 峰值抬升伪影（forgetting 是"从峰值跌落"的相对量，对角线被抬高后机械变大；绝对值上旧任务无退化）——待 GPT 裁决定措辞（metric critique 进 discussion 或换 retention 口径）。

---

## 3. A2：问题发现、定位测试、机制（论文方法章 §4 素材）

### 3.1 发现与定位链

1. **相关性诊断**：compatibility_drift 2×2 组合（新旧 key/anchor 交叉换装）确认损伤来自接口世代差。
2. **D0**：`verify_slot_binding` 判定槽位-类绑定是涌现的（路由纯相似度、无显式类绑定）→ 归因前必须先验证绑定。
3. **D1 v2 固定效应回归**：未来流量碰撞、槽位劫持、落点去向、路由不稳定、key 直接因果**五条直觉损伤路径全部判死**；唯一存活通道 = anchor 位移（FE rho=+0.259, p=0.001）→ 机制修正为 emergent interface sharing → anchor prototype contamination。
4. **A2 预注册四臂干预（决定性）**：干预推翻相关性归因——**key 单冻臂**才是有效通道：F 18.89→7.11（削减 63%）、AA 78.62→87.09（+8.5pp）、可塑性代价仅 0.6pp；anchor 臂有害（F 恶化 5.5pp，破坏 key-anchor 共适应）；both 与 key 打平但多付可塑性。A1 上界 91.67，A2 捕获 68%；残余 = prompt 漂移 + c4 路由碰撞。

### 3.2 为什么选 A2

- 因果证据：唯一通过预注册干预验证的通道（相关性诊断的 anchor 假说被同一实验否定，key 才是真因——干预越小，机制证明强度越高）。
- 最小干预：post-step restore ≡ freeze + wd 豁免，不改官方优化器与训练损失。
- 零通信、零回放、微开销；不需推理期任务身份，可直接部署。

### 3.3 机制（f527ab7，frozen Phase-4 spec）

- 任务起点：RNG 中立特征回访，已完成任务 train 特征经**当前 key** top-1 路由命中的槽位取并集 = 保护集（D0 涌现支撑）。
- 训练期：每个 phase-2 optimizer step 后回写保护槽位（freeze + wd 豁免）。
- 工程封印：checkpoint `a2_state` + fail-closed resume（arm 不匹配即 RuntimeError）；开关门 `--a2_arm {off,key,anchor,both}`，默认 off = 与基线逐位一致；与 `--dsar` 正交组合。

### 3.4 诚实披露（limitation/appendix）

11-02 大清理中当时的 A2 方法线（`--method fedta_a2`）入判死档案（laziness check：冻结行从不被路由命中）；11-08 为终局方案从 f37590d^ 恢复 frozen Phase-4 规格并改为开关门实现。**新四臂矩阵是 A2 的现役证据载体**——A2 与 A2+DSAR 臂跑完后以其数据为论文最终判据，不复用 10 月旧四臂数字。

---

## 4. 模块 × 问题总表（论文主表/框架图素材）

| 模块 | 解决的损伤 | 作用层 | 时机 | 标签 | 通信 | 主场 | 保险 |
|---|---|---|---|---|---|---|---|
| A2 | 训练期 key/anchor 支撑槽位侵蚀 | 训练侧（参数保护） | 任务起点定保护集 + step 级回写 | 旧任务 train 特征回访（不进损失） | 0 | 旧任务保持 | off 默认 = 基线 |
| B | 广播 z⁺ 漂移 | 读出侧（特征坐标桥） | 每广播边界 per-client fresh-fit | 零标签 | 0（client-local） | 历史任务 | paired-MSE identity guard |
| L | 读出失配（当前特征 ↔ 联训头） | 读出侧（低秩残差适配） | 同上 | 当前任务 train 标签 | 0 | 当前任务 | CIFAR/INR 均无 CE guard（反相关证据），worst-cell 有界 |
| 组合 | A2 ∩ DSAR = ∅（参数集/时机正交） | — | — | — | — | 超加性由四臂矩阵检验 | — |

四臂矩阵（base / A2 / DSAR / A2+DSAR）双数据集进度：CIFAR base ✓、DSAR ✓；INR base ✓、DSAR ✓；**A2、A2+DSAR 四格待发车**（RUN.md 有冻结命令）。

---

## 5. 写作建议

1. 先写发现链（§1.2 接口遗忘 + 86.4% 分解是最强 hook）。
2. 判死名单写成 "design space falsification" 小节——系统性排除是审稿人最认的部分。
3. ΔF 红门等 GPT 裁决后定措辞：metric critique 进 discussion，或换 retention 口径重表。
4. 预注册纪律本身可作为 methodology 贡献写（判据先于数据冻结、bit-exact 对账、fail-closed）。

## 6. 剩余实验工作（阻塞主表）

1. CIFAR A2 臂 + A2+DSAR 臂发车（GPU 空出后按 RUN.md）。
2. INR A2 臂 + A2+DSAR 臂（izfree）。
3. B1 verdict 工具（四项门 JSON + RUNCOMMANDS §34.4d 登记）。
4. GPT 裁决材料：CIFAR 2/4 + INR 3/4 的 ΔF/minΔ_k 处置。
