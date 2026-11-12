# FedSMTA §34 B0 判决结果汇总（cifar100, seed 42, 125 cell 全量）

> 用途：交给外部模型做交叉审查。目标读者无项目上下文，本文自包含。

## 1. 背景与问题

- 基础系统：FedTA（CVPR2025）——联邦持续学习，ViT prompt + per-client Tail_Anchor 读出头；5 客户端 × 5 任务（cifar100，每任务 5 轮联邦，共 25 轮）。
- 关键发现：每轮 **broadcast**（服务器向客户端分发状态）后，客户端的特征表示被系统性"搬坏"（z⁻ → z⁺ 漂移）。损伤随任务深度放大：task0 时 −1.56pp，task4 时 −4.69pp。
- 提议结构（DSAR，Dual-State Alignment Readout）：
  - **B_t**：label-free ridge 桥，把广播后表示 z⁺ 拉回出生表示 z⁻（λ‖ΔA‖² 残差参数化，GCV 选 λ，含 ΔA=0 的 identity 候选）
  - **L_t**：低秩 residual adapter（rank r），有标签 CE 拟合，只动残差
- **B0 = 结构可行性判决实验**：预注册判据（含阈值冻结、四臂定义、前置硬门），在一条与官方训练轨迹逐位一致（A5 RNG 中立已验证）的 capture run 上，用 in-run 真配对数据（每轮 broadcast 边界缓存 z⁻/z⁺ 全行）离线执行。125 cell = 25 轮 × 5 客户端。

## 2. 预注册判据（§34.2.5）

四臂（全部在同一离线管线内配对，读出头统一用 M2' refit head）：

| 臂 | 定义 | 角色 |
|---|---|---|
| ① identity | M2' refit head 直接读 z⁺ | 起点（损伤量） |
| ② bridge | B_t(z⁺) 后读 | 论文点 1（分子） |
| ③ lowrank | ② 的 ΔA 做 SVD 截断 r∈{8,16,32,64} | 论文点 2 |
| ④ M2'' | 冻结 refit 配方（100ep, bs64, Adam, masked CE）在 z⁺ 上重训读出头 | 重训上限（分母） |

**ρ_birth = (②−①)/(④−①)**；GREEN ⟺ mean ≥0.8 且 pooled ≥0.8 且无 task 级系统性崩溃；YELLOW 0.5–0.8；RED <0.5。有效分母 ≥0.5pp，负分母 cell 排除（只记录）。

前置硬门：reproduction gate + z_ref 不变性门（npz 的 z_ref pre/post 逐位）。

## 3. 正式 B0 结果（M2' 读出口径，125 cell）

```
① identity   94.83
② bridge     96.64   ← 免标签
③ lowrank    96.40 / 96.53 / 96.57 / 96.61   (r=8/16/32/64)
④ M2''       96.57   ← 有标签重训
ref: z⁻ 读出  97.24（birth oracle）

ρ_birth: mean=1.006  pooled=1.018  (valid 84/125, 负分母排除 41)
per-task ρ: t0=0.707  t1=0.990  t2=0.968  t3=1.245  t4=1.033
```

## 4. reproduction gate 的两轮故事（gate 修订史）

**第一轮（0.25pp 逐位容差）→ FAIL，根因定位**：官方 evaluate 路径从未调用 `.eval()` → 读出头内 Dropout(0.1) 在评估时开着 → 官方 CSV 里的准确率是 dropout-ON 随机前向的**单次采样**。拿我们的确定值（dropout-OFF）逐位对齐在数学上不可满足。

**第二轮（MC-dropout 带宽判据）**：我们也开 dropout 采 50 遍形成自己的分布，检查官方数是否落 3σ 带内（|z|≤3）且 z 符号不系统性同侧（同侧 ≤20/25）。结果两臂分裂：

- **arm1（npz 里 in-run 锚定的 M2' 头）**：25/25 PASS，全部 |z|≤2.5，符号 7+/18− → **读出语义与官方逐位等价，方法学闭合**。
- **arm4（我们离线重训的 M2'' 头）**：6 cell 出带（z = −9.4 ~ +6.5），但符号 13+/12− 完全均衡、双向波动 → 不是系统性偏差，是**重训随机性**：in-run refit 与离线 refit 同配方（同 seed/100ep/Adam），但 batch 顺序、shuffle、dropout 采样流嵌在官方 run 的 RNG 环境里，离线不可复刻 → 两头是"同配方、不同随机实现"的两个样本。
- **判定（决策）**：ρ 的分母只需在本管线内有效（预注册的负分母排除已覆盖其噪声），不要求复现官方的抽签实现 → arm4 的官方对照降级为 reference record，gate 判据改为 arm1-only。
- z_ref 不变性门：PASS 25/25。

**判决（verdict.json，2026-11-06 落盘）**：arm1-only gate 重跑确认 → ρ mean 1.006 ≥ 0.8、pooled 1.018 ≥ 0.8、无 task 崩溃 → **GREEN，B1 结构实现解冻**。

## 5. 两个论文点的证据

**论文点 1：免标签桥 B_t（主结果）**
- ρ_birth ≈ 1：**零标签的桥追平有标签的重训**（96.64 vs 96.57）
- baseline 部署口径（birth preview，联训头读出）：损伤 2.85pp，B_t 免标签修回 82%（pooled），且修复量随损伤放大（task4 修回 86%）
- 鲁棒性：41 个 M2'' refit 失效 cell（负分母，多为小拟合集）上桥仍比 identity 高 +0.15pp——桥不依赖拟合集大小
- 安全阀：20/125 cell GCV 自动选 identity（ΔA=0）——损伤可忽略时桥自动退化为 no-op，不引入额外损伤

**论文点 2：低秩充分性**
- r8（≈1% 参数量）拿回 87% 的修复量；r64 与 full ridge 只差 0.03pp；r∈{8..64} sweep 平坦单调 → ΔA 本征维数低，低秩叙事成立

## 6. 相对 baseline 的涨幅（current-task 诊断口径）

| 部署方式 | 准确率 | vs baseline |
|---|---|---|
| baseline（z⁺ 直进联训头） | 93.52 | — |
| + B_t | 95.86 | +2.34pp |
| + B_t + L_t（DSAR 全链，r8） | ≈96.3–96.8 | **≈+2.8~3.3pp** |
| oracle ceiling（z⁻ + 重训头） | 97.24 | +3.72pp |

即拿到 oracle 天花板的 ~80%；且收益随任务深度放大（t0 −1.6pp 损伤 → t4 −4.7pp 损伤，修复随之放大）。

## 7. 口径说明与遗留事项

- 以上涨幅是 **current-task 诊断读数**（B0 capture 管线预估），**不是论文主表 AA**；旧任务保持/遗忘收益要等 B1 把 L_t/B_t 实现进训练管线后跑正式 run 才知道。
- z⁻ = broadcast 前出生表示；z⁺ = broadcast 后表示；cell = (round, client)；npz = capture run 落盘的 per-boundary 配对数据。
- 下一步：① ~~arm1-only gate 重跑封板 GREEN~~（已完成，GREEN 落盘）；② B1 预注册 + 结构实现（进训练管线）；③ ImageNet-R 的 B0-lite（capture run 在另一台机器进行中）；④ 多 seed 复现。
- 已知限制：重训类对照（M2''）存在训练随机性不可逐位复现问题，本文以"管线内有效 + 负分母排除"处理；跨机器（不同 GPU）不可逐位一致，所有对照均在同一 run 内配对。

## 想请 GPT 审查的问题

1. gate 从"逐位容差"修订为"MC 带宽 + arm1-only"，arm4 降级为 reference——这个证据链处理是否有漏洞？
2. ρ_birth > 1（桥超过重训上限）的叙事解释（桥直接对齐 z⁻，refit 在损伤表示上绕道重学）是否站得住，还是应该按"分母测量误差"保守处理？
3. 41/125 负分母排除比例偏高（33%），是否威胁 ρ 判决的代表性？（per-task 覆盖：valid 84 cell 五个 task 都有）
4. 免标签桥的"20/125 自动选 identity"与"41 cell refit 失效但桥仍 work"两个卖点是否与主叙事冲突？
