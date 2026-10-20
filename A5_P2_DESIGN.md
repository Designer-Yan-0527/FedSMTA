# A5 · P2 可训练共享语义层设计（v1，供评审）

> 状态：设计稿，未实现。评审通过后冻结为预注册方案再动工。
> 证据基础：P0 manifest 审计 + P1 NCM probe（均 2026-10-20，见 FEDSMTA_ROADMAP.md §6）。

## 0. 证据链与设计约束

| 证据 | 结论 | 对 P2 的直接约束 |
|---|---|---|
| P0：INR 0 公共类（200 类全单客户端） | LCDC 跨客户端同类补偿通道关闭 | INR 上只能走**跨类**语义共享 |
| P0：CIFAR-100 = 75 私有 + 25 公共 | public/private 分层判别可行 | CIFAR-100 做机制分离消融 |
| P1：z_op NCM(82.49) ≈ native Chead(82.87) @T0 | 分类读出无瓶颈 | **换分类器没用**，必须练表征 |
| P1：z_ref NCM 倒退 3.8 | 冻结特征不足 | 共享层要真正改特征 |
| P1：z_op > z_ref 一贯（+3.4/+4.6） | prompt 适配特征有价值 | 共享层与 prompt 路径并行而非替代 |
| P1：T4 oracle NCM 仅 +1.19 | A2 后残余接口损伤很小 | P2 的收益预期主体在**基础精度**而非保持 |
| C²P 审计：T0 即 +3.99 | 对比损失+原型训练改表征 | 目标函数是关键变量 |
| C²P 审计：prev-task full re-access | 其低遗忘部分靠重训兜底 | **P2 明确不复制**：只用当前任务数据（exemplar-free 纪律） |

双速架构定式：A2 快照保护私有任务 prompt+key/anchor（稳定历史接口）；A5 新增**每轮 FedAvg 的可训练共享语义层**（演化共享接口）。二者参数集不相交。

## 1. 架构（评审焦点 ①）

```
input ──► ViT block (frozen) ──► [prompt 路径: FedTA 原样, A2 保护] ──┐
              │                                                      ├─► concat/pre_logits ──► Chead (原样)
              └─► [共享路径: Adapter(共享, FedAvg)] ──► g·Adapter(x) ──┘
```

- **形态三选一**：
  - (a) **LoRA**（r=8，Q/V，后 K=4 个 block）——参数最少，与 prompt 无参数交叠；
  - (b) **bottleneck adapter**（MLP 后并行，down/up + 残差）——容量稍大；
  - (c) **共享类原型 prompt**（C²P 式 per-class prompt pool）——最贴 C²P 血统但引入第二套路由接口，与 A2 的接口失配教训相冲。
  - **倾向 (a)**：最小侵入、与 A2 保护集天然正交、(c) 有接口失配前科。
- **注入门控 g**：零初始化标量（或 per-block 门控），训练中自学幅度。零初始化保证 T0 起点 = FedTA（不破坏 Gate 0 parity 血统），也压制旧任务特征漂移。
- **注入位置**：只动 pre_logits 之前的特征通路；**Chead 与评估路径完全不动**，保证 R[t][k] 与主线/A2 可比。

## 2. 训练目标（评审焦点 ②）

```
L = L_CE(FedTA 原样, 不动) + λ · L_proto(共享层特征空间)
```

- `L_proto` 候选（按侵入度递增）：
  - (i) **prototype-CE**：类原型（running mean，在线 sum/count 更新，无旧数据回访）作为辅助分类头，取 CE；
  - (ii) **SupCon**：共享层特征上的监督对比损失（SupCon, Khodak et al.）；
  - (iii) 两者并用。
  - **倾向 (i) 起步**：实现简单、直接对应 P1 判决（原型在**被训练的特征空间**里才有效），(ii) 作为增消融。
- λ 固定 0.5 起步，不搞 schedule（消融变量控制在最少）。
- 原型维护：**只来自当前任务** inner.train 的在线统计（EMA β=0.99 或精确 sum/count）；任务结束后原型冻结进全局表——这是与 C²P mover 重训的本质区别，主表数据访问列标 "current-task only"。

## 3. 聚合（评审焦点 ③）

- **共享 Adapter**：每轮标准 FedAvg（样本加权），与 prompt 池聚合同相位；
- **全局类原型表**：sum/count 精确合并（与 FedNova 记账同型，无近似）；INR 0 公共类下合并恒等（每类单客户端），CIFAR-100 上 25 公共类做真正的跨客户端统计对齐——这本身就是 public/private 分层判别的实验载体；
- **A2 保护集**：不参与 FedAvg（维持现状，纯本地快照恢复）。

## 4. 兼容性风险与对策（评审焦点 ④，A5 核心风险）

共享层每轮漂移 = 在新通路上**重造 A2 刚修掉的接口失配**（旧任务用当前 adapter 评估）。

对策（按顺序叠加，消融臂）：
1. **零初始化门控**（§1，默认有）；
2. **surrogate 约束蒸馏**：adapter 输出对齐 origin_model 特征（surrogate 集 s=5/20，协议内合法数据），L += μ·‖g·A(x_s) − stopgrad(f_ref(x_s))‖；
3. 若 1+2 仍致旧任务退化 → 兼容性约束升格为核心贡献（这正是四臂析因要回答的）。

监控：compatibility_drift 2×2 扩展 adapter 通道（old/new adapter × old/new prompt），复用现有诊断。

## 5. 四臂析因（论文主消融）

| 臂 | A2 | Shared | 回答 |
|---|---|---|---|
| FedTA | ✗ | ✗ | 基线 |
| A2 only | ✓ | ✗ | 接口保护独立贡献（已有 INR 数据） |
| Shared only | ✗ | ✓ | 共享层是否重造失配（风险臂） |
| A5 = A2+Shared | ✓ | ✓ | 双速组合 |

- 指标：AA / BWT / Forgetting **分开报**；交互项 I = Y₁₁−Y₁₀−Y₀₁+Y₀₀；
- **T0-gap 判据贯穿**：Shared 臂若在 T0 就缩差距 → 表征修复成立（P1 预测的主收益通道）；若只在 T4 有增益 → 持续迁移叙事；
- 数据集：INR（跨类迁移干净试验场）+ CIFAR-100（public/private 分层）；种子 42 先行，43/44 复验。

## 6. 预注册判据（评审通过后冻结，防事后合理化）

1. **主判据**：A5(both) 在 INR after-T4 AA ≥ A2(79.43) + 2.0pp，且 BWT 不劣化（≥ A2 − 0.5pp）——否则 A5 不进主表；
2. Shared-only 臂旧任务（T0 行）相对 FedTA 退化 ≤ 2pp → 门控+蒸馏足够；> 2pp → 兼容性约束升为核心贡献线；
3. T0-gap：Shared 臂 T0 AA ≥ FedTA(82.87) + 1.5pp 判"表征修复成立"；C²P(86.86) 不作为达标线（其含重训兜底，非公平上界）；
4. 若主判据失败但 Shared 臂 T0 显著提升 → A5 重定位为"基础精度修复"线，保持交给 A2。

## 7. 实施管道

- `--method fedta_a5`（含 `--a2_arm` 继承）+ `--a5_shared {none,lora,adapter}` / `--a5_proto_loss {none,proto_ce,supcon}` / `--a5_lambda` / `--a5_gate zero|learn`；
- 评估路径零改动（native evaluate + Chead）→ 现有 RunManager CSV、三级聚合、A1/A2 回归测试全部复用；
- 新增回归：resume 含 adapter/原型表状态；T0 parity（gate=0 时与 FedTA 逐位一致——零初始化保证）；
- 跑位：A5 是我们自己的方法 → 全部 AutoDL；先 INR seed42 smoke（1 task）→ 1 task 全程 → 5 task 正式。

## 8. 待决问题清单（请评审按编号表态）

| # | 问题 | 我的倾向 |
|---|---|---|
| ① | adapter 形态 (a)LoRA / (b)bottleneck / (c)类原型 prompt | (a) |
| ② | L_proto (i)proto-CE / (ii)SupCon / (iii)both | (i) 起步 |
| ③ | 原型维护 EMA vs 精确 sum/count | sum/count（可验证，无超参） |
| ④ | 门控零初始化是否足以防旧任务漂移，还是直接上 surrogate 蒸馏 | 先只上零门控，蒸馏留作臂 |
| ⑤ | λ=0.5 固定是否可接受 | 接受，消融 {0.1, 0.5, 1.0} 后置 |
| ⑥ | CIFAR-100 是否首轮就跑，还是 INR 判决后再上 | INR 先行（机制最干净） |
| ⑦ | 预注册阈值（§6 数值）是否合理 | 待定 |
