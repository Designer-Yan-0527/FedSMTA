# A5 · P2 v1.1 设计：Route-Preserving Shared Semantic Adaptation（供评审）

> 状态：设计稿 v1.1（2026-10-20，吸收外部评审四阻塞项后重写），未实现。冻结后动工。
> 证据基础：P0 manifest 审计 + P1 NCM probe（见 §0 与 FEDSMTA_ROADMAP.md §6）。

## 变更摘要（v1 → v1.1）

| # | v1 问题 | v1.1 修正 |
|---|---|---|
| A | 零门控 × LoRA B=0 初始化 = 严格梯度死区 | 输出投影 U=0 初始化 + 无门控（g≡1），∂L/∂U ∝ Vz ≠ 0 |
| B | Q/V LoRA 改 ViT 内部 → 改变 key 路由 = 重造 A2 刚修复的失配通道 | 共享残差置于 **Tail Anchor 路由之后、Chead 之前**，路由只看原始 h_p |
| C | FedTA 两阶段训练（阶段 2 vit 在 no_grad 内），ViT 内 adapter 接不到损失梯度 | adapter 只依赖冻结 z_ref，**在阶段 2 训练**（z_ref 为常量输入，Δθ(z_ref) 对 U,V 可微） |
| D | 原型冻结在演化特征空间 = 跨世代失配重演 | 原型统计存**固定 z_ref 空间**；线性 W_θ=I+UV 满足 W·E[z]=E[W·z]，损失时用当前 W_θ 投影 |
| E | 蒸馏公式 μ‖gA−sg(f_ref)‖ 在 gA=0 时非零（推动残差复制整条教师特征） | v1.1 **不上蒸馏**；预定义补救阶梯见 §5 |
| F | "A5 只用当前任务数据"声明过强（A2 保护槽计算本身回访历史 inner-train） | 三级数据访问披露表（§9）；A5 只主张共享层无历史梯度训练 |

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
| P1：z_op NCM(82.49)≈native(82.87) @T0 | 读出无瓶颈 | 必须练表征 |
| P1：z_ref NCM 倒退 3.8 | 冻结特征单独不足 | 共享层要真正改特征 |
| P1：z_op>z_ref（+3.4/+4.6） | prompt 特征有价值 | 残差与 prompt 路径并行，不替代 |
| P1：T4 oracle NCM 仅 +1.19 | A2 后残余接口损伤小 | 收益预期主体在基础精度 |
| C²P T0 即 +3.99 | 原型/对比目标改表征 | 目标函数是关键变量 |
| C²P mover 重训 | 其保持部分靠重训兜底 | 共享层**无历史梯度训练**（非价值判断——C²P 机制在其异步设定下合理；不采用是为归因洁净，机制研究放在 C²P 三臂独立消融） |
| P0：INR 0 公共类 | 同类跨客户端补偿关闭 | INR 只能走跨类共享；prototype-CE 分母含其他客户端已见私有类 |

## 1. 架构：路由后共享残差

```
x ──► F_prompt(x) = h_p（FedTA 原路径, SIKF 演化, 阶段1 训练）
x ──► F_frozen(x) = z_ref（origin_model, 恒冻结）

j* = argmax_j sim(h_p, k_j)          ← 路由只看 h_p，共享层不介入
v  = ( h_p + Δθ(z_ref),  a_{j*} )    ← 1536 维不变，无第二个 head
ŷ  = Chead_t(v)                        ← 原生评估路径零改动

Δθ(z) = UV z,  U∈R^{768×r}, V∈R^{r×768}, r=8
初始化：U=0, V≠0（随机）；无门控
```

- **T0 parity 血统**：shared=off 与 FedTA/A2 逐位一致；shared=on 初始残差为零（首步前 logits 与基线一致），U 首步即得非零梯度（无死区）
- **三种更新规则并存**（修正 v1 措辞）：A2 保护历史路由接口（key/anchor 支撑槽快照恢复）+ FedTA prompt 经 SIKF 持续演化 + A5 共享残差每轮 FedAvg——参数集不相交，功能上路由被 h_p 侧隔离
- 深层 LoRA（block 内 Q/V）**降为后置架构消融**：它改变路由 = 第一版就重开损伤通道

## 2. 训练（评审焦点：阶段归属与梯度隔离）

- **阶段 1（prompt 优化）原样不动**：adapter 不参与，不接收任何新损失梯度
- **阶段 2（Tail_Anchor + Chead 优化）中训练 adapter**：z_ref 由冻结 origin_model 计算（可为 no_grad 常量），Δθ(z_ref) 对 U,V 可微——无需触碰 vit 前向或 no_grad 结构
- **损失**：
  - 原生阶段 2 损失（masked CE + global prototype InfoNCE）**逐位不动**；
  - 新增 `L = λ·L_proto`，**梯度只到 U,V**（原型与 h_p 侧全部 stop-grad）；
  - 对照控制：设"只加 proto-CE / native InfoNCE 不变"臂，防止把 FedTA 已有原型机制包装成新贡献
- λ=0.5 固定起步；τ=0.07；消融 {λ: 0.1/0.5/1.0} 后置

### L_proto（跨客户端已见类 prototype-CE）

```
L_proto = −log exp(sim(W_θ z, W_θ μ_y^ref)/τ) / Σ_{c∈C_seen} exp(sim(W_θ z, W_θ μ_c^ref)/τ)
W_θ = I + UV（当前），μ_c^ref = 固定 z_ref 空间类均值（stop-grad）
```

- 分母 = **全部已见类**（含其他客户端私有类——INR 0 公共类下这是唯一的跨客户端语义共享通道），绝不含未来类
- 带梯度输入 = 仅当前任务 inner-train 样本
- 线性恒等式 W·E[z]=E[W·z] 保证原型无需随 W_θ 演化重算（阻塞项 D 解）
- **冷启动（预注册）**：每任务首次训练前，对当前任务 inner.train 做一次无梯度统计遍历入表（计入预算，四臂一致）

## 3. 聚合

- adapter（U,V）：每轮样本加权 FedAvg，独立参数组
- 原型表：sum/count 精确合并（无近似）；INR 合并恒等（单客户端类），CIFAR-100 上 25 公共类做真跨客户端对齐
- A2 保护集：维持现状（纯本地，不聚合）

## 4. 兼容性监控

- 残差仍会改变旧 Chead 决策边界（输入变了）——旧任务损伤监控不因路由隔离而豁免
- compatibility_drift 2×2 扩展 adapter 通道（old/new adapter × old/new prompt）
- Shared-only 臂旧任务退化 ≤2pp 为门控充分信号（同 v1）；超标进入 §5 补救阶梯

## 5. 补救阶梯（预定义，v1.1 默认全部关闭）

1. **残差幅度正则** `L_res = E‖Δθ(z_s)‖²`（surrogate 集）——最简单，只是幅度约束
2. **历史决策蒸馏** `L_KD = τ²·KL(p_teacher^τ(x_s) ‖ p_A5^τ(x_s))`，教师 = 任务边界快照、同 class mask——真兼容性保护；启用前必须先审计 INR surrogate 对旧任务决策区域的覆盖度

## 6. 四臂析因（论文主消融，不变）

| 臂 | A2 | Shared | 回答 |
|---|---|---|---|
| FedTA / A2 only / Shared only / A5(both) | | | 接口保护、共享层风险、双速组合与交互项 I=Y₁₁−Y₁₀−Y₀₁+Y₀₀ |

- **臂间一致性硬规则**：除共享层本身外完全一致（A2 保护、surrogate 访问、原型冷启动遍历、轮次、本地预算）；Shared-only 不得意外获得更高预算或更宽历史数据权限
- AA / BWT / Forgetting 分开报 + **逐旧任务精度变化**（防 AA 上升掩盖旧任务损伤）
- Shared-only 旧任务退化**不是 A5 的否决条件**（A2+Shared 安全保留增益恰好证明 A2 互补价值）
- T0-gap 判据贯穿：Shared 臂 T0 ≥ FedTA+1.5pp 判"基础精度修复"信号

## 7. 预注册判据（v1.1 修订）

**工程准入（seed 42，探索/调试种子，不作论文结论）**：
- A5 T4 AA ≥ 81.43（=A2 79.43+2.0），**且** BWT ≥ A2−0.5 **联合判断**（若收益全来自新任务、T0–T3 均值明显退化 → 不算成功）
- Shared-only T0 ≥ 84.37 = 表征修复研发信号

**正式主张（预冻结三种子 42/1999/2025，对齐 FedTA 官方复现协议）**：
- 配对种子改变量 + 不确定性区间；单 seed 跨门槛永不构成论文结论

**实现陷阱测试（实施前必须断言）**：`fedta_a5` 显式启用 A2；`--a2_arm none` 下断言保护集为空——否则四臂实际只有三个方法。

## 8. 实施前单元测试（8 项，smoke 内断言，不靠日志肉眼）

1. shared=off：FedTA 与 A2 逐位复现
2. shared=on、零初始化残差：首步前 logits 与对应基线一致
3. 首次反向传播后 adapter 梯度非零
4. 关闭 proto-CE 后，原生 prompt 更新路径未被替换（阶段 1 参数轨迹与基线一致）
5. adapter 开/关不改变同一 h_p 的 Tail Anchor 选槽（路由不变性）
6. 各客户端每轮收到同一全局 adapter；checkpoint/resume 恢复 adapter + 原型表 + 优化器 + RNG
7. 原型全部来自已见 inner-train、无未来类；聚合前后类计数精确对账
8. 评估仍用原生 task mask + Chead + AA 汇总定义（评估协议零改动）

## 9. 数据访问披露（主表口径）

| 方法 | 历史数据使用 |
|---|---|
| FedTA | 常规优化仅当前任务 |
| A2 | 任务起始回访历史 inner-train 提特征算保护槽；**无历史样本梯度训练** |
| A5 | 继承 A2 的保护槽统计回访；共享 adapter 梯度训练**只用当前任务** |
| C²Prompt | prev-task full re-access 且重训练（mover 分支；其异步设定下合理，非价值判断） |

A5 的准确主张是"共享层不使用历史样本做梯度重训练"，**不是**"整个方法历史数据零访问"。严格 exemplar-free 主张需另行改造 A2（任务首次训练时保存路由统计、证明不读旧样本可达同等保护）——独立方法贡献，不并入 P2-v1.1。

## 10. 剩余待决（评审表态）

| # | 问题 | 倾向 |
|---|---|---|
| ⑧ | τ=0.07 是否合理 | 标准值起步，消融后置 |
| ⑨ | adapter 是否也接阶段 1 梯度 | 否——阶段 2 only，归因最干净 |
| ⑩ | 预注册阈值数值（§7） | 待定 |
| ⑪ | CIFAR-100 首轮参与还是 INR 判决后 | INR 先行，CIFAR 必须参与最终机制验证 |
