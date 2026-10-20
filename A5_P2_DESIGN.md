# A5 · P2 v1.2 设计：Route-Preserving Shared Semantic Adaptation（已冻结，2026-10-20）

> 状态：**冻结版预注册方案**。v1.1 经终审补充三技术约束（梯度归属 / 单因子聚合 / 原型同步时序）后升版冻结。实现须逐条遵守；任何偏离 = 新版本 + 重新评审。
> 证据基础：P0 manifest 审计 + P1 NCM probe（§0；FEDSMTA_ROADMAP.md §6）。

## 变更摘要

| 版本 | 变更 |
|---|---|
| v1 → v1.1 | 四阻塞项：梯度死区（U=0 无门控）、路由后残差、阶段 2 训练、固定空间原型；蒸馏公式纠错后置；数据访问三级披露 |
| v1.1 → v1.2 | 终审三约束：**A** 阶段 2 梯度归属表 + proto-CE 两侧 W 梯度明确；**B** 固定共享 V 单因子训练（有效变换 FedAvg 严格成立），双因子降为容量消融；**C** 任务边界原型先同步后训练的 5 步时序 + 全局已见类定义。新增两个必需机制对照（Shared CE-only / Local Adapter）。P1 叙事收紧 |

## 0. 证据链与设计约束

### 0.1 P1 NCM probe 原始结果（INR seed42，A2 run，2026-10-20）

| after_task | feat | native | NCM-cos | NCM-euc | C²P 参考 |
|---|---|---|---|---|---|
| T0 | z_ref（冻结特征） | 82.87 | 79.06 | 77.70 | 86.86 |
| T0 | z_op（当前 prompt 特征） | 82.87 | 82.49 | 81.32 | 86.86 |
| T4 | z_ref | 79.43 | 76.04 | 74.92 | 91.14 |
| T4 | z_op | 79.43 | 80.62 | 79.21 | 91.14 |

不变性对账：z_ref 旧行 T0/T4 逐 client 一致（diff=+0.00 ×5）。完整数据：`diagnostics_output/ncm_probe_inr_a2.csv`。

### 0.2 参照系（同协议 seed42）

| 方法 | T0 AA | T4 AA | 数据访问 |
|---|---|---|---|
| FedTA | 82.87 | 77.12 | 当前任务 |
| A2（ours） | 82.87 | 79.43 | 当前任务 + 任务起始历史特征回访（无历史梯度训练） |
| C²Prompt | 86.86 | 91.14 | prev-task full re-access + 重训练（mover 分支） |

### 0.3 证据 → 约束

| 证据 | 结论 | 约束 |
|---|---|---|
| P1：z_op NCM(82.49)≈native(82.87) @T0 | **当前证据不支持以固定特征 NCM 替换 Chead 作为主要精度提升路径**（不排除线性 probe / 判别式原型分类器） | P2 优先验证共享可训练特征残差 + 类级判别目标 |
| P1：z_ref NCM 倒退 3.8 | 冻结特征单独不足 | 共享层要真正改特征 |
| P1：z_op>z_ref（+3.4/+4.6） | prompt 特征有价值 | 残差与 prompt 路径并行，不替代 |
| P1：T4 oracle NCM 仅 +1.19 | A2 后残余接口损伤小 | 收益预期主体在基础精度 |
| C²P T0 即 +3.99 | 原型/对比目标改表征 | 目标函数是关键变量 |
| C²P mover 重训 | 其保持部分靠重训兜底 | 共享层**无历史梯度训练**（非价值判断——C²P 机制在其异步设定下合理；不采用是为归因洁净，机制研究放 C²P 三臂独立消融） |
| P0：INR 0 公共类 | 同类跨客户端补偿关闭 | INR 只能走跨类共享；prototype-CE 分母含其他客户端已见私有类 |

## 1. 架构：路由后共享残差（单因子）

```
x ──► F_prompt(x) = h_p（FedTA 原路径, SIKF 演化, 阶段1 训练）
x ──► F_frozen(x) = z_ref（origin_model, 恒冻结）

j* = argmax_j sim(h_p, k_j)          ← 路由只看 h_p，共享层不介入
v  = ( h_p + Δθ(z_ref),  a_{j*} )    ← 1536 维不变，无第二个 head
ŷ  = Chead_t(v)                        ← 原生评估路径零改动

Δθ(z) = U V z,  U∈R^{768×r} 可训练,  V∈R^{r×768} 固定共享,  r=8
初始化：U=0；V = 服务器生成的固定随机投影（任务 0 前广播，全程不变）
```

- **单因子 rationale（约束 B）**：固定 V 下 `∑pₖ(UₖV) = (∑pₖUₖ)V`——adapter 的 FedAvg = **有效变换的精确样本加权平均**。双因子 `Ū V̄ ≠ ∑pₖUₖVₖ`（交叉协方差项），只作容量消融（factor-wise FedAvg，须监控有效变换聚合偏差）
- **零残差 + 无死区**：U=0 → 初始残差为零（首步前 logits 与基线一致）；∂L/∂U ∝ Vz ≠ 0 首步即有梯度
- **三种更新规则并存**：A2 保护历史路由接口（key/anchor 支撑槽快照恢复）+ FedTA prompt 经 SIKF 持续演化 + A5 共享残差每轮 FedAvg——参数集不相交，路由被 h_p 侧隔离
- 深层 LoRA（block 内 Q/V）为后置架构消融：改变路由 = 重开损伤通道
- shared=off 与 FedTA/A2 逐位一致；shared=on 只保证**训练开始前**输出等价（"原生损失不变"指定义与系数不变，开启 adapter 后 loss 数值当然改变）

## 2. 训练（阶段 2 only，梯度归属表 = 约束 A）

| 参数 | 阶段 1 | 阶段 2：原生 CE / InfoNCE / pull-off | 阶段 2：proto-CE（新增） |
|---|---|---|---|
| ViT prompt | ✓（原样） | ✗（no_grad，原样） | ✗ |
| frozen backbone | ✗ | ✗ | ✗ |
| key 路由 / pull-off | — | 用原始 h_p（原样） | — |
| Tail_Anchor（可训练部分）/ Chead | — | ✓（原样） | ✗ |
| adapter U | ✗ | ✓（融合特征 v 参与原生损失） | ✓ |

```
L_stage2 = L_CE^native + α·L_InfoNCE^native − 0.1·L_pull + λ·L_proto
```

- **λ=0 = Shared CE-only 对照**（必需机制对照之一，区分"共享可训练容量"与"跨客户端语义目标"）
- **proto-CE 梯度路径**：`sim(W_θz, W_θμ_c)`，W_θ=I+UV 为**同一当前投影，两侧均接收梯度**（μ_c 数值固定于 z_ref 空间、不随 W 演化——判别目标，非教师目标；若改 detach μ 侧则属另一种教师目标定义，未经评审不得混用）
- τ=0.07，相似度 = **归一化内积 / τ**（`normalize(W_θz)·normalize(W_θμ_c)/0.07`，再对全部已见类 logits 取 cross_entropy；禁止未归一化内积直接除 τ）
- 数值监控（记入日志）：L_proto 数值、最大 logit、有限值比例、proto-CE 与原生 CE 在 U 上的梯度范数比
- λ=0.5 固定；消融 {0.1/0.5/1.0} 后置

### L_proto（跨客户端已见类 prototype-CE）

- 分母 = **当前全局已见类**（服务器原型表全集，含其他客户端私有类），绝不含未来类
- 带梯度输入 = 仅当前任务 inner-train 样本
- 冷启动：见 §4 时序步骤 2

## 3. 聚合

- adapter U：每轮样本加权 FedAvg（= 有效变换的精确加权平均，见 §1 单因子 rationale）
- 原型表：sum/count 精确合并；**不因 adapter 更新而重算**（固定 z_ref 空间，无需重访）
- A2 保护集：维持现状（纯本地，不聚合）

## 4. 任务边界原型同步时序（约束 C，5 步写死）

1. **Task boundary**：所有客户端进入当前任务，确定 inner-train
2. **无梯度统计**：各客户端对当前任务 inner.train 计算固定 z_ref 空间的类级 sum/count（一次前向遍历，计入预算，四臂一致）
3. **服务器同步**：按类精确合并，发布统一、**带版本号**的 `global_ref_protos`
4. **正常 FedTA 轮次**：阶段 1 训练 prompt → 阶段 2 训练 adapter / Tail_Anchor / Chead
5. **每轮聚合**：FedAvg adapter；原型统计不重算

- **全局已见类定义**：遵循同步联邦协议——服务器原型表含所有客户端已见类。CIFAR-100 上公共类可能对当前客户端是"未来任务"但已在表中（提前暴露）；**必须统计暴露率并报告**，实现不得私自改用客户端本地已见类集合（否则不可比）
- 同步阶段只含合法已见类，记录额外计算量与通信量

## 5. 兼容性监控与补救阶梯

- 残差会改变旧 Chead 决策边界——旧任务损伤监控不因路由隔离而豁免
- compatibility_drift 2×2 扩展 adapter 通道（old/new adapter × old/new prompt）
- Shared-only 旧任务退化 ≤2pp：门控充分；超标进入补救阶梯：
  1. 残差幅度正则 `L_res = E‖Δθ(z_s)‖²`（surrogate 集）
  2. 历史决策蒸馏 `τ²·KL(p_teacher ‖ p_A5)`（教师 = 任务边界快照、同 mask；启用前先审计 INR surrogate 对旧任务决策区域的覆盖度）

## 6. 消融与对照

### 6.1 四臂主消融

| 臂 | A2 | Shared | 回答 |
|---|---|---|---|
| FedTA | ✗ | ✗ | 基线 |
| A2 only | ✓ | ✗ | 接口保护独立贡献 |
| Shared only | ✗ | ✓ | 共享层是否伤旧任务（风险臂；退化**不是 A5 否决条件**——A2+Shared 安全即证 A2 互补） |
| A5 = A2+Shared | ✓ | ✓ | 双速组合；交互项 I=Y₁₁−Y₁₀−Y₀₁+Y₀₀ |

### 6.2 必需机制对照（判定"加了模块"还是"共享语义有贡献"）

| 对照 | 配置 | 区分 |
|---|---|---|
| **Shared CE-only** | λ=0（无 proto-CE） | 共享可训练容量 vs 跨客户端语义目标 |
| **Local Adapter** | 同参数量同预算，adapter 不 FedAvg（纯本地） | FedAvg 共享 vs 新增容量 |

- 臂间一致性硬规则：除被消融变量外完全一致（A2 保护、surrogate 访问、冷启动遍历、轮次、本地预算）
- AA / BWT / Forgetting 分开报 + 逐旧任务精度变化
- T0-gap 判据贯穿：Shared 臂 T0 ≥ FedTA+1.5pp 为表征修复信号

## 7. 预注册判据（三层）

**研发通过（seed 42，探索性种子——已参与方法设计，正式报告中标为探索性结果）**：
- A5 INR T4 AA ≥ 81.43（=A2 79.43+2.0），**且** BWT ≥ 同 seed A2−0.5，**且**旧任务逐行无被 AA 掩盖的显著退化（联合判断）

**机制支持**：
- Shared CE-only 与 Local Adapter 两对照就位后：FedAvg > Local-only（共享有贡献）且 proto-CE > CE-only（跨客户端语义有贡献）——这两条把"加了模块"转化为"共享语义演化改善 FCIL"的可验证主张

**论文主张（确认性运行）**：
- 预冻结种子 42/1999/2025（42 = 探索性，1999/2025 = 确认性）；不重新调参；配对改变量 + 不确定性区间；两新种子不作过强显著性声明

**工程信号（非否决）**：Shared-only T0 ≥ 84.37（=FedTA 82.87+1.5）为表征修复研发信号

**实现陷阱断言（实施前必须过）**：`fedta_a5` 显式启用 A2；`--a2_arm none` 断言保护集为空。

## 8. 实施前单元测试（8 项，smoke 内断言）

1. shared=off：FedTA 与 A2 逐位复现
2. shared=on、U=0：首步前 logits 与对应基线一致
3. 首次反向传播后 U 梯度非零
4. λ=0 且 shared=off：阶段 1 参数轨迹与基线一致（原生路径未被替换）
5. adapter 开/关不改变同一 h_p 的 Tail Anchor 选槽（路由不变性）
6. 各客户端每轮收到同一全局 U（及同一 V）；checkpoint/resume 恢复 U + V + 原型表 + 优化器 + RNG
7. 原型全部来自已见 inner-train、无未来类；服务器合并前后类计数精确对账；原型表版本号随任务边界单调
8. 评估仍用原生 task mask + Chead + AA 汇总定义（评估协议零改动）

## 9. 数据访问披露（主表口径）

| 方法 | 历史数据使用 |
|---|---|
| FedTA | 常规优化仅当前任务 |
| A2 | 任务起始回访历史 inner-train 提特征算保护槽；**无历史样本梯度训练** |
| A5 | 继承 A2 的保护槽统计回访；共享 adapter 梯度训练**只用当前任务**（含任务起始无梯度冷启动遍历） |
| C²Prompt | prev-task full re-access 且重训练（mover 分支；其异步设定下合理，非价值判断） |

A5 的准确主张 = "共享层不使用历史样本做梯度重训练"，非"方法历史数据零访问"。严格 exemplar-free 主张需另行改造 A2（任务首次训练时保存路由统计、证明不读旧样本可达同等保护）——独立方法贡献，不并入 P2。

## 10. 冻结配置表

| 模块 | 冻结值 |
|---|---|
| 方法名 | Route-Preserving Shared Semantic Adaptation |
| 特征路径 | frozen z_ref + 原始 prompt h_p |
| 路由 | 原始 h_p 决定 top-1 key |
| 共享残差 | Δ(z)=UVz，r=8；**U 可训练（U=0 init），V 固定共享（服务器生成广播）** |
| 训练阶段 | Stage 2 only（梯度归属表 §2） |
| 训练目标 | 原生 CE/InfoNCE（梯度含 U）+ λ·proto-CE（梯度仅 U） |
| λ / τ | 0.5 / 0.07（归一化内积 / τ） |
| 原型统计 | 固定 z_ref 空间，精确 sum/count，带版本号 |
| 负类范围 | 当前全局已见类（CIFAR-100 统计提前暴露率） |
| 同步时机 | 每任务开始、本地训练之前（§4 五步） |
| 服务器共享 | 每轮 FedAvg U（=有效变换精确加权平均） |
| 蒸馏 | 主方法关闭，预定义补救消融 |
| 主对照 | FedTA / A2 / Shared-only / A5 |
| 必需机制对照 | Shared CE-only（λ=0）、Local Adapter |
| 工程判据 | INR T4 AA ≥81.43%，BWT 非劣（联合） |
| 最终验证 | INR 先行 + CIFAR-100 必跑；42/1999/2025（42 探索性，1999/2025 确认性）配对 |
| 容量消融（后置） | 双因子 U,V 训练（factor-wise FedAvg + 聚合偏差监控）、r 扫描、深层 LoRA |
