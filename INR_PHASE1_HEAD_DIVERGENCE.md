# INR Phase-1 Scaffold Head 数值发散（独立 P0 机制问题）

状态: **根因已定位**（fed_avg_head 原地别名累加，见 §4.5；2026-10-23 定案）
首次发现: 2026-10-23，A2-JE 梯度一致性探针（commit a420d90）结果追查
影响范围: ImageNet-R 全部 run（fedta 基线与 fedta_a2，逐位相同）；CIFAR-100
全部 run 同样触发（有界不发散）；官方 FedTA 代码自带（初始提交 d4756ae 未改）
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

机制（§4.5 根因定案后的完整链条，§7 CIFAR 证据支持）：任务末 head =
本轮最后训练的客户端（c4）phase-1 留下的状态；c0-c3 的类内行在同一轮内
被后续 4 个客户端的 phase-1（wd-only Adam，每步 ~lr 侵蚀）抹回
~1e-3 平衡态，×16 轮末重标定后停留 ~1e-2 → logits≈0 → 均匀输出。
对照：G4（部署 Chead CE 对 prompt 的梯度）全部客户端量级健康（0.26-5.7）。

## 4.5 根因（已证实）：fed_avg_head 原地别名累加

`Server_DF.fed_avg_head`（[Server_DF.py L962-982](Models/Server_DF.py)，
初始提交 d4756ae 原文，fork 未改）：

```python
heads = []
for i in chosen_clients:
    heads.append(self.clients[i].vit.head)   # 5 个引用指向同一共享对象
result_head = deepcopy(heads[0].state_dict())
for k in result_head.keys():
    for i in range(len(heads)):
        local_model_params = heads[i].state_dict()   # 返回活引用
        if i == 0:
            result_head[k] = local_model_params[k]   # 覆盖 deepcopy，绑到活张量
        else:
            result_head[k] += local_model_params[k]  # 原地累加同一个张量!
    result_head[k] = result_head[k] / len(chosen_clients)  # 非原地，生成新张量
```

因为 5 个客户端共享同一个 ViT 对象（init_client L78-79 传同一 `self.model`），
`state_dict()` 返回的都是同一批活张量：i=1 时 h+=h→2h，i=2→4h，i=3→8h，
i=4→16h。**每轮聚合后活 head 整体 ×16 = 2^(N-1)**；`/5` 只作用于新张量，
故 global_head = 16h/5 = 3.2h。

**四重独立验证：**
1. 纯 torch 最小复现（5 个引用同一 Linear，逐行复刻上述代码）：
   活 head=16.0，global_head=3.20，比值 5.000
2. 全部 checkpoint 实测 server_head/global_head 最大行比值 = **5.000**
   （INR fedta / INR fedta_a2 / CIFAR a2_key，全部 task_end）
3. 逐任务倍率精确 = 2^20 = 1.05e6（= 16^5，5 轮/任务）：
   6.65e4→6.98e10→7.32e16→7.67e22→8.04e28，每步比值 1.048-1.050e6
4. 初始规模吻合：Linear 初始化 |W|max≈0.06 × 16^5 ≈ 6.3e4 ≈ T0 实测 6.65e4

**出处**：`git log -L` 确认该函数自初始提交（官方 FedTA 导入）未改动；
Gate 0A 官方 parity（D=0，含 head 状态，5 轮后 head 已 ×16^5）只能在该
bug 同样存在于官方仓库时通过 → 官方 FedTA 自带此 bug。

**论文规范核对（2026-10-23，arXiv 2412.18355v2 / CVPR 2025 FedTA）**：
论文中 head 不参与聚合，`fed_avg_head` 是论文中不存在的操作：
- Fig. 3 上传箭头仅两个：Knowledge Base（input enhancement 参数）与
  Local Prototypes；MLP Head 留在本地，无 upload 箭头。Introduction 同述：
  "the server will separately process the parameters added during input
  enhancement ... and the local prototype"
- §4.1 定义 scaffold head 为 per-client（"Each client has a classification
  head used for training input enhancement parameters, H_e^i"）——代码的
  共享 vit.head 亦偏离论文（该偏离归 HS-S）
- Table 1 将 head 聚合表述为 FedViT 的特征（"only the classification head
  is involved in training and aggregation"），非 FedTA 的设计
- 代码侧：global_head 唯一消费点 `evaluate_on_global_testset` 全仓库零调用
  （死代码）→ `fed_avg_head` 的唯一净效应 = ×16/轮污染；且因 5 客户端共享
  同一 ViT 对象，其"平均"在数学上恒等（5 个相同张量求平均 = 自身）
- 修复形式推论：跳过 fed_avg_head（论文忠实）与修别名（保留调用）对训练
  轨迹逐位等价（正确平均 = 恒等）；HS-S 的 per-client scaffold head
  方向与论文本意重合

**动力学平衡（解释 CIFAR 有界 vs INR 失控）：**
- 每轮 ×16 重标定 vs phase-1 wd-only Adam 侵蚀（~lr/步）形成平衡：
  无 CE 信号的行停在 ~1e-2（中位行实测 8.6e-3）
- c4（每轮最后训练）：其 8 行在保存前刚被 CE 重建，唯一存活者
- CIFAR：c4 的 CE 快速拟合（CE 0.28-0.95）→ 自身 phase 后期 wd 主导
  侵蚀 → 与 ×16 平衡 → 停在 ~46（log10=1.66，跨任务稳定）
- INR：c4 的 CE 永不拟合（T4 CE≈2900，高置信全错）→ CE 步进持续
  累积 → ×16 逐轮复合 → 失控至 1e28+
- c0-c3（两数据集）：CE 建立量（~lr×步数 ≈ 5）< 后续 4 个客户端的
  侵蚀容量（~4×lr×步数 ≈ 6.4）→ 每轮被完全抹平 → 均匀输出

**伴生发现（2026-10-23 修正：fed_avg_prompt 为死代码）**：
`fed_avg_prompt`（L937-958）有相同的别名模式，但全仓库**零调用**——
prompt 的实际聚合路径是 `kd_fusion_prompt`（SIKF 蒸馏，L166，与论文
§4.3 一致，任务末轮跳过），此前的"瞬态无害"表述不再需要。`fedavg`
（L83-99）同样零调用，且其 i=0 分支为 `* w`（乘法生成新张量），
**写法本身无别名问题**——代码库中存在正确模板但未被采用。
实际运行的聚合面 = BGPS（论文 §4.4）+ SIKF（论文 §4.3）+
`fed_avg_head`（论文中不存在）——唯一偏离论文的聚合恰好是有害的那个。

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

## 6. Round-trajectory audit：已被 §4.5 根因定案取代（原证据限制记录）

本 run 只有 task_end 级 checkpoint（5 个）+ latest（=task_04_end），
无逐轮 checkpoint。原计划回答的三个问题现状：

1. 发散从何时开始 → **已回答**：从第 1 轮聚合起（×16/轮是构造性的，
   初始 |W|max≈0.06 × 16^5 ≈ T0 实测 6.65e4）
2. client 顺序效应是否存在 → **已回答**：是（§4 动力学 + §7 CIFAR
   行为证据；c4 唯一存活 = 最后训练 + 保存前 CE 重建）
3. ×10^6/task 的来源 → **已回答**：2^20 = 16^5，逐轮 ×2^(N-1) 精确复合

instrumented 重跑不再必要（§4.5 四重验证已闭环）。

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

### 7.1 跨数据集的结构性发现（根因定案前记录，机制解释见 §4.5）

**ln(8) 均匀退化不是 INR 特有**：CIFAR（head 有界、无发散）上同样出现
——c4（本轮最后训练的客户端）head 行拟合良好（CE 0.28/0.95），而部分
其他客户端（T0 的 c1/c2，T4 的 c2/c3）类内行 CE 精确 = ln(8)（G1→0）。
注意：CIFAR 并非"无 bug"——×16/轮同样活跃（ratio=5.000 实测），
只是 CE-拟合动力学与重标定达到有界平衡（§4.5）。

结构性两层缺陷（现均可归因到 §4.5 根因 + 共享 head 串行训练）：
1. **结构性（两数据集共有）**：单个共享 head 串行服务 5 个客户端的
   不相交 8 类 mask，非最后训练的客户端行被同一轮内后续 phase 的
   wd 侵蚀抹平（建立量 < 侵蚀容量）
2. **数值性（INR 放大）**：fed_avg_head ×16/轮重标定在 CE 不拟合
   条件下复合失控（×2^20/任务）

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

- **A2-HS（Head Stabilization）**：修复脚手架头系统。根因定案后
  拆分为两个正交子问题（对应 GPT 的 HS-S / HS-N）：
  - HS-N（数值）：**已实现**——`--skip_fed_avg_head` 开关（2026-10-23，
    方案 A：跳过 fed_avg_head 调用，论文忠实语义；默认 OFF = 官方行为
    逐位一致，orthogonal to --method）。修别名版本（方案 B）与 A 训练
    逐位等价，未采用（保留无意义死计算路径）
  - HS-S（结构）：共享 head 串行训练的客户端顺序干扰（最后训练者
    唯一存活）——候选 per-client scaffold head 或激活行独立更新
    （与论文 §4.1 per-client H_e^i 本意重合）
  两者都必须走新 method 开关：该 bug 是官方基线行为，直接修复会
  改变所有既有结果的归因口径（fedta 与 a2 的全部数据都在此行为下取得）
- **A2-JE（Joint/deployed-space optimization）**：让真实 Chead 决策
  反向塑形 prompt。回答：即使脚手架健康，直接绑定部署边界是否仍更好。
  注意 §4.5 定案后，JE 的动机更强：phase-1 在两数据集上对 4/5 客户端
  的 prompt 仅有噪声级 CE 信号（均匀头），prompt 实际由 reduce_sim
  和 SGD 噪声主导；唯一有效信号（c4）在 INR 上还是病态的。

两者实现前必须完成 GPT 裁定的机制鉴别与设计评审。JE 设计问题清单
（不写死实现）：替换还是联合 phase-1 CE；更新哪些参数块；
G4/G1 的 100-1000× 尺度差处理；InfoNCE/pull_off2 是否反传 prompt；
30+30 epoch 等预算口径；scaffold head 去留；CIFAR/INR 同一冻结方案。

## 9. 证据文件

- `output/ImageNet-R/fedta_a2/inr_a2_key_p40/seed_42/metrics/je_gradient_probe{,_summary}.csv`
- `output/cifar100/fedta_a2/a2_key/seed_42/metrics/je_gradient_probe{,_summary}.csv`
- checkpoint head 范数扫描（§1 表）与逐 batch 损失/梯度诊断（§4、§7 表）
  为 2026-10-23 会话内一次性脚本输出，数字已录入本文档
