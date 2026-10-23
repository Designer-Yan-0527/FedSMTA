# INR Phase-1 Scaffold Head 数值发散（独立 P0 机制问题）

状态: 已证实（checkpoint 证据），病因未定（round 粒度证据缺失，见 §6）
首次发现: 2026-10-23，A2-JE 梯度一致性探针（commit a420d90）结果追查
影响范围: ImageNet-R 全部 run（fedta 基线与 fedta_a2，逐位相同）；CIFAR-100 不受影响
基线约束: FedTA/A2 冻结不动；任何修复走新 method 开关（见 §8）

## 1. 现象

ImageNet-R seed_42，task-end checkpoint 中共享 ViT 的内部分类头 `vit.head`
（`nn.Linear(768, 200)`，phase-1 脚手架，非部署头）权重指数发散：

| checkpoint | fedta (inr_base_p40) | fedta_a2 (inr_a2_key_p40) |
|---|---|---|
| task_00_end | 6.65e4 | 6.65e4 |
| task_01_end | 6.98e10 | 6.98e10 |
| task_02_end | 7.32e16 | 7.32e16 |
| task_03_end | 7.67e22 | 7.67e22 |
| task_04_end | 8.04e28 | 8.04e28 |

- 数值为 `|server_model['head.weight']|.max()`，两家 run 逐位一致
  （A2 只保护 Tail_Anchor key/anchor 行，不触碰 vit.head）
- 每任务精确 ×10^6；从 task_00_end 已开始（6.65e4）
- bias 同步发散（|b|max ~4e24 @T4）；最大权重行（类 12, 195, 37, 80, 153）

**CIFAR-100 对照健康**：a2_key / a2_anchor / a2_both（seed_42）全部
task-end |W|max = 5-8，全程无发散 → 问题特定于 ImageNet-R 配置。

## 2. 为什么训练日志里看不到

phase-1 的 CE 对 logits 做 task mask（`index_fill(-inf)` 于非当前类），
爆炸 logits 集中在非当前类行上，被 mask 遮住；类内 logits 仍有限，
损失值不直接暴露发散。

## 3. 代码路径（发散的土壤）

- 共享对象：`Server_DF.init_client()`（Server_DF.py L74-80）把**同一个**
  `self.model`（ViT）传给全部 5 个客户端；客户端按 0→4 顺序串行 `train()`
- phase-1 优化器：`Adam(self.vit.parameters(), lr, weight_decay=1e-3)`
  （Client_DF.py L306）——prompt、vit.head、fc_norm 一起训练
- phase-1 损失：CE 于 `output['logits']` 即 vit 内部头（L320, L327-329）
- 部署路径完全不经过该头：`vit.feat → Tail_Anchor → Chead`；
  phase-2 的 vit 前向在 `torch.no_grad()` 内（L238-242）

即：**一个被全部客户端串行改写的脚手架头决定 prompt 学习方向，
而部署分类器从未反向塑形 prompt。**

## 4. 任务末状态的失效形态（JE probe 实测，T0/T4 同型）

| 客户端 | phase-1 masked CE | G1 = ∇_prompt CE | 形态 |
|---|---|---|---|
| c0-c3 | ≈ ln(8) = 2.0794（8 类均匀） | 0.001-0.02 | 类内 logits≈0，G1 近噪声 |
| c4 | 1200-3200 | 24000-44000 | 类内 logits 达 1.6e4，高置信全错 |

机制候选（未单独证实）：任务末 head = 本轮最后训练的客户端（c4）
phase-1 留下的状态；c0-c3 的类内行在后续 4 客户端 × 30 epoch 的
weight decay 下衰减归零。"×10^6/task" 的确切来源（逐轮稳定倍增 vs
跨 task 跳变）未知——见 §6 限制。

对照：G4（部署 Chead CE 对 prompt 的梯度）全部客户端量级健康（0.26-5.7）。

## 5. 对已有结论的影响

### 5.1 A2-JE 探针（INR）可解释性受限

INR 上 cos(G1,G4)≈0（10/10 组合，-0.10~+0.17）**不能**读作
"两个良定义目标方向失配"：c0-c3 的 G1 是退化目标（均匀分布），
c4 的 G1 被爆炸 head 主导。原始 JE 前提在 INR 上无法干净检验。
判读以 CIFAR-100（健康 head）对照探针为准（机制鉴别实验）。

### 5.2 C²Prompt +4pp 归因收紧

旧表述（不再使用）："C²P T0 86.86 vs FedTA/A2 82.87 → 4pp 主要来自
更好的端到端目标函数。"

收紧后表述：**C²P T0 表明冻结骨干 + prompt 范式内至少存在约 4pp 的
可恢复空间；当前证据提示 FedTA 的两阶段目标设计与 INR-specific
phase-1 head 病态都可能贡献这一差距，尚不能完全归因于端到端目标函数。**

## 6. Round-trajectory audit 的证据限制

本 run 只有 task_end 级 checkpoint（5 个）+ latest（=task_04_end），
**没有逐轮 checkpoint**；train.log 与 round_metrics.jsonl 也不含
head 范数。因此以下问题现有证据无法回答：

1. 发散从 T0 第几个 round 开始
2. client 0→4 顺序效应（C0 rows↑ → C1 训练后 C0 rows↓ → … → C4 存活）是否存在
3. ×10^6/task 来自逐轮稳定倍增还是跨 task 跳变

如需 round 粒度证据，需 instrumented 重跑（仅加 head-norm 日志，
不改训练逻辑），须另行评审批准。

## 7. CIFAR-100 对照 JE probe 结果（2026-10-23，已完成）

健康 head 环境（|W|max 5-8，无发散）。逐客户端 phase-1 CE 损失实测
（8 batch 均值，ln(8)=2.0794 为 8 类均匀基线）：

| | c0 | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|
| T0 CE | 2.00 | **2.0795** | **2.0794** | 2.63 | **0.28** |
| T0 G1 范数 | 1.83 | 4.6e-4 | 3.7e-4 | 7.19 | 3.01 |
| T0 cos(G1,G4) | +0.02 | -0.07* | +0.14* | +0.15 | **+0.49** |
| T4 CE | 2.62 | 2.18 | **2.0794** | **2.0794** | **0.95** |
| T4 G1 范数 | 8.66 | 10.7 | 1.7e-3 | 1.5e-13 | 17.9 |
| T4 cos(G1,G4) | +0.11 | +0.29 | +0.06* | NaN* | **+0.47** |

*号为 G1 近零（均匀退化），其 cos 为噪声，不具解读价值。

### 7.1 跨数据集的结构性发现（比 INR 爆炸更根本）

**ln(8) 均匀退化不是 INR 特有，而是共享 head 串行训练的结构性质**：
CIFAR（head 完全健康）上同样出现——c4（本轮最后训练的客户端）
head 行拟合良好（CE 0.28/0.95），而部分其他客户端（T0 的 c1/c2，
T4 的 c2/c3）类内行被后续客户端 × 30 epoch weight decay 抹平到
均匀输出（CE 精确 = ln(8)，G1→0）。

即脚手架系统有两层缺陷：
1. **结构性（两数据集共有）**：单个共享 head 串行服务 5 个客户端的
   不相交 8 类 mask，非最后训练的客户端行被 weight decay 抹平
2. **INR 特有**：在此结构上权重再指数爆炸（×10^6/task）

### 7.2 机制鉴别结论（对 GPT 裁决表）

- **B2（cos ≥0.7 居多 → JE 降级）被排除**：10 个组合中 0 个 ≥0.7，
  最高为 c4 的 0.47/0.49
- **B1（健康 G1 下普遍失配）部分成立**：G1 健康的 6 个组合中
  4 个 cos < 0.3（+0.02~+0.29），但 c4 两次达 ~0.47-0.49（中等）
- 细致图景：脚手架目标并非与部署目标内在正交——当 head 行新鲜且
  拟合当前客户端数据时（c4），对齐度中等（~0.48）；结构性衰减使
  其余客户端信号退化。JE 前提（prompt 从未被部署边界塑形）成立，
  但"持续性方向冲突"证据依然不存在（负余弦比例 0.12-0.38）

## 8. 后续候选（严格分离，禁止合并归因）

- **A2-HS（Head Stabilization）**：仅修 phase-1 脚手架头数值稳定性，
  保留两阶段范式。回答：phase-1 目标恢复健康后，当前任务 CA 能否提升。
- **A2-JE（Joint/deployed-space optimization）**：让真实 Chead 决策
  反向塑形 prompt。回答：即使脚手架健康，直接绑定部署边界是否仍更好。

两者实现前必须先完成 §6/§7 的病因与机制鉴别。JE 设计问题清单
（不写死实现）：替换还是联合 phase-1 CE；更新哪些参数块；
G4/G1 的 100-1000× 尺度差处理；InfoNCE/pull_off2 是否反传 prompt；
30+30 epoch 等预算口径；scaffold head 去留；CIFAR/INR 同一冻结方案。

## 9. 证据文件

- `output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42/metrics/je_gradient_probe{,_summary}.csv`
- `output/cifar100/fedta_a2/a2_key/seed_42/metrics/je_gradient_probe{,_summary}.csv`
- checkpoint head 范数扫描（§1 表）与逐 batch 损失/梯度诊断（§4、§7 表）
  为 2026-10-23 会话内一次性脚本输出，数字已录入本文档
