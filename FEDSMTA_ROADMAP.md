# FedSMTA 研发路线（v3，2026-10-10 全面重写）

## 0. 文档说明

v3 是**全面重写**：多模态主线（Semantic statistics / Oracle-K2 / Semantic-K2 / Dynamic-K / UOT）已于 2026-10-10 判决死亡，相关数学体系与方法设计不再占用正文。本版只保留**已验证事实、失败判决存档与活跃方向**。

旧版完整内容（含 §2 数学体系、§5–10 Phase 1–7 方法设计、§12 代码审查全记录）在 git 历史中：

```bash
git show cff49dd:FEDSMTA_ROADMAP.md
```

## 1. 项目定位（当前生效）

### 1.1 论文定位

**FCIL 中被忽视的接口失配型表观遗忘（interface-mismatch apparent forgetting）**：
SOTA 方法（FedTA, CVPR 2025）的"无遗忘"结论是评估协议盲区的产物；真实遗忘（18.89pp）由推理期组件接口失配主导（非知识擦除），可在**零重训**的推理/接口层修复。最终措辞（2026-10-10 判决）：**"机制通用、签名联邦"**——核心机制（KA 接口失配、事件驱动、世代共适应）为通用机制（与 centralized CL 的 transport keys 工作互补），联邦独有的是公共/私有不对称遗忘签名。

**目标**：不做 FedTA++（单点工程改进），提出可推广的联邦持续学习**接口稳定性**方法；机制贡献 > 性能贡献；顶会论文。

### 1.2 研究链条

审计协议 → 机制归因 → 联邦签名 → 接口稳定性修复 → 多方法验证（2 方法 × 2 数据集）。

## 2. FedTA 基线事实（源码确认，方法设计的基础）

### 2.1 架构事实

- 冻结 ViT backbone（特征提取器不变，遗忘 100% 发生在可演化组件）
- prompt 模块：**SIKF 聚合，全局共享**（唯一受聚合影响的旧任务推理组件）
- Tail_Anchor：key + anchor_pool 各 200 个 per-class 槽位（768 维），top-1 相似度路由，anchor 与特征拼接后进 per-task Chead；**纯本地，从不聚合**（[Tail_Anchor.py](file:///d:/MyWork/FedSMTA/Models/Tail_Anchor.py#L33-L66)）
- 优化器 Adam、weight_decay=1e-3，作用于全部 key/anchor 槽位（含未被路由命中的槽位）
- per-task Chead，旧任务推理恒用任务专属 head

### 2.2 协议事实（CIFAR-100）

- 5 clients × 5 tasks × 8 classes；每客户端 15 private + 25 public（[cifar100_subset_spliter.py](file:///d:/MyWork/FedSMTA/data/cifar100_subset_spliter.py#L43-L106)）
- **公共类在同一客户端自己的后续任务中不重现**（40 类 shuffle 后按 8/任务切块，每类只出现一次）
- 公共类本地训练数据被 Dirichlet 切分（均值 ~100 样本/类 vs 私有类全量 500）——n_test 是本地数据量的 30% 内切代理
- 官方硬编码：本地训练 batch=16、surrogate_num=20、threshold=0.25；KRt 评估脚本未随代码发布

### 2.3 复现基线（Gate 0A 已证明逐位一致）

- 宿主 run：`cifar100_fedta_c5_t5_ge5_le30_lr0.001_seed42_951bdc3b`，seed 42（seed 0/1 已复跑待分析）
- 对角线 94.25/96.27/91.67/92.70/93.77（均值 93.73 ≈ 论文 93.70，plasticity 对齐）
- 部署态（after4）：T0=85.66 / T1=71.10 / T2=71.45 / T3=71.13 / T4=93.77；AA=78.62，标准遗忘指标 F=18.89
- 本仓库与官方 FedTA 在 13 项组件上 D=0（tag: `phase0-fedta-baseline`），`--method fedta` 为 read-only scientific baseline

## 3. Phase 0：三 Gate 判决（已封板，2026-10-08）

- **Gate 0A（官方 parity）PASS**：FedSMTA（`--method fedta --no_instrumentation`）与官方 FedTA 在 global protos / head / prompt / key / anchor / client heads / accuracy 上逐位一致（13 项 D=0；协议：global_epoch=5、threshold=0.25 对齐官方硬编码）
- **Gate 0B（resume 等价）PASS**：22 项全绿——连续训练 = checkpoint+resume（全部模型组件、四路 RNG、inner+outer split manifest 逐位一致）
- **Gate 0C（observer 不变性）PASS**：29 项全绿——插桩 ON/OFF 训练轨迹逐位一致，artifact 确证区分

## 4. 已验证结论（方向 (c) 证据链，全部有数据留档）

### 4.1 协议盲区：论文指标看不见遗忘

- Table 1 = 对角线表（只报当前任务精度，结构上看不见遗忘）
- KRt（eq.11）只度量 task 0；且 task 0 受每轮 broadcast `evaluate(0)` 持续监控，存在结构性偏护（T0 总漂移 −8.8 远小于 T1–T3 的 −20~−25）
- KRt 测量状态未定义：同一 run 不同口径可报 84.6%~100.3%
- T1–T3 部署态崩溃至 ~71（保留率 ~74%），对 Table 1 / KRt / KRs 三指标同时不可见

### 4.2 表观遗忘判决：接口失配，非知识擦除

- old/old 组合精确复原训练时精度（T0 94.49 vs 94.25 / T1 96.27 / T2 92.15 vs 91.67 / T3 92.89 vs 92.70，±0.5pp 内）→ KRt=100% 在参数存储层面成立
- 损伤非单调可回升（T1：96.27 → 80.90 → 65.61 → 71.10）→ 排除知识擦除

### 4.3 2×2 归因（四对全闭合，compatibility_drift.py）

| 任务对 | 窗口 | oo | oc | co | cc | 总漂移 | KA（占比） | P（占比） | 交互 |
|---|---|---|---|---|---|---|---|---|---|
| T0×T4 | 4 | 94.49 | 89.13 | 93.05 | 85.67 | −8.82 | −5.36（61%） | −1.44 | −2.02 |
| T1×T4 | 3 | 96.27 | 76.13 | 93.85 | 71.41 | −24.86 | −20.14（81%） | −2.42 | −2.30 |
| T2×T4 | 2 | 92.15 | 80.65 | 84.83 | 72.42 | −19.73 | −11.50（58%） | −7.32（37%） | −0.91 |
| T3×T4 | 1 | 92.89 | 71.41 | 92.94 | 70.53 | −22.36 | −21.48（96%） | +0.05 | −0.93 |

- KA（key/anchor）主导：占总漂移 58–96%；cc 精确复现部署态（2×2 协议内部效度成立）
- KA 分量高度异质：受害者随任务切换（c3 在 T1 免疫、T2 最惨 −30.8）

### 4.4 margin 归因（先行指标）

T0：oo 8.47 / oc 3.31 / co 7.94 / cc 2.69；T1：oo 8.61 / oc 2.70 / co 7.56 / cc 1.81——KA 分量 −5.2~−5.9 主导；部署态 p10 转负；margin 比 accuracy 敏感 ~4×。

### 4.5 事件驱动：单窗口饱和

T3×T4（窗口=1）总漂移 −22.4 ≈ T1×T4（窗口=3）的 −24.9 → 遗忘由任务切换时的 anchor 更新事件一次性驱动，非渐进累积。

### 4.6 世代共适应

c1 在 T1 上 old-prompt+cur-KA（57.76）< cur-prompt+cur-KA（63.79）——prompt 与 key/anchor 成对漂移，单换任一接口反而更糟 → 修复必须保持接口世代一致。

### 4.7 公共/私有不对称（联邦独有现象）

样本加权 retention：public 74.4% < private 83.9%（T1/T3 上 public delta −31 vs private −18/−19）——持续被其他客户端再训练的公共类比再无人训练的私有类损伤更大。

### 4.8 4 组合 per-class 通道分解（analyze_class_retention.py，class_retention_4combo.csv）

| scope | 总漂移 | P 通道 oo−co | KA 通道 oo−oc |
|---|---|---|---|
| public | +23.41 | +4.22（18%） | +18.31（78%） |
| private | +15.27 | +2.34（15%） | +11.40（75%） |
| **公共类超额** | +8.14 | **+1.88（23%）** | **+6.91（85%）** |

超额损伤走本地 KA 通道而非聚合 prompt 通道。但因公共类在本客户端后续任务不重现 + key/anchor 纯本地，纯本地类无关漂移应一视同仁——公共类 KA 超额仍需联邦层面的解释（间接传导）。

### 4.9 子机制判别（probe_channel_correlates.py，158 单元）

- **脆弱假说否证（方向相反）**：public 内 KA~n_test **正**相关（r=+0.346，p<1e-4，5/5 客户端同向）；n_test≤15 公共类近乎免疫（wKA=+1.80，woo=71 排除天花板效应），16-40 → +20.79，41-80 → +15.10，81-171 → +33.55
- **间接共适应修正版成立**：匹配强度（woo≈95）下扎根公共类（n_test≥40，n=31）wKA=+19.71 vs private +11.40，4/5 客户端方向一致（c3 反转与已知异质性一致）；P 通道同步抬升（+5.23 vs +2.34）；KA_c~P_c 弱正相关（r=+0.353，p=0.10）
- **"扎根×演化交互"降级为待检验假说**（诚实条款）：当前仅为相关性——private 无低样本对照臂、pub/priv 未控制任务/客户端/oo，**不能称超可加**。因果链待 D1（slot 级诊断）+ A2（路由干预）坐实
- 机制解释候选（与弱槽免疫一致）：弱训练 key 近随机初始化 → 不参与路由拓扑 → 不被劫持；扎根槽位参与拓扑，且公共类表征区在联邦侧持续演化 → 本地 KA 共适应拖走旧接口

### 4.10 KRt 对账与 20pp 复现差距

- 配置协议逐项一致（args.json vs 论文 5.1）；对角线均值 93.73 ≈ 论文 93.70
- KRt 三口径均达不到论文 105–115%：广播态 93.0→94.5→98.0→98.4%；部署态 88.6→93.0→90.1→90.7%；old/old ~100.5%
- 差距两段分解：①协议段（铁证，可发表）②残余段 ~7–15pp（supra-baseline 不可达）；三候选死了两个（公共类强化假说否证），剩：未发布 KRt 脚本 / 硬编码参数 ≠ 论文配置 / seed（待 seed 0/1 界定方差上界）
- GitHub issue 已发官方 repo（KRt 测量状态 / 硬编码参数 / seed 数三问），等作者回应

## 5. 失败判决存档（何者已死、为何死）

| 假说/主线 | 判决 | 依据 |
|---|---|---|
| 多模态主线（Semantic-K2 / Oracle-K2 / UOT） | **死亡** | Phase 1：robust multimodal prevalence 0/25（ρ、D_cover、client 归因、切分形态、z_ref 对照五证据链全阴性）；Phase 2 保持冻结 |
| 公共类跨客户端强化假说 | **死亡** | retention 方向相反（public 74.4% < private 83.9%） |
| 聚合介导的联邦独有新失配机制 | **死亡** | 4 组合：公共类超额损伤 85% 走本地 KA 通道，直接聚合通道仅 23% |
| 协议结构性脆弱假说（Dirichlet 欠训练 → 槽位脆弱） | **死亡** | 方向相反：KA~n_test 正相关，弱 share 公共类免疫 |
| "超可加交互"措辞 | **降级** | 仅为相关性证据；缺对照臂与协变量控制 |
| FedTA++ 作为引用支柱 | **放弃** | DataSciMI 2026 低质量；训练层 EMA 与本工作正交；一句话差异化即可 |
| supra-baseline KRt（105–115%）复现 | **不可达** | 三口径对账全部低 7–15pp；作附录 reproducibility note |

## 6. Phase 4：接口稳定性路线（当前活跃，2026-10-10 定稿）

**原则**：只动推理/接口层验证因果，不动训练 loss / BGPS / SIKF / FedAvg head / CIFAR split / Tail Anchor 结构 / prompt 逻辑（A2 干预在新方法 flag 下实验，baseline 不动）。

### 6.0 D1：slot 级碰撞诊断（先于一切修复，observer-only，消费现有 checkpoint）

对每个旧类槽位 i 统计后续任务中的三元关系：

1. **路由命中次数**：后续任务训练样本在任务边界 key/prompt 状态下 top-1 槽位选择频次（复刻 Tail_Anchor.forward 的 normalize→matmul→topk(1)）
2. **参数位移**：key_i / anchor_i 从任务 t 末到各边界再到 task_04_end 的范数位移
3. **KA 损伤**：join 4combo CSV（已有）

**关键设计点**：Adam wd=1e-3 使未命中槽位也会位移——必须按 hits>0 / hits=0 分层对比位移，分离"劫持介导"与"衰减介导"两种损伤路径，直接决定 A2 用梯度掩码还是冻结+衰减豁免。

**判据**：按 (client, old_task) 分组 + public/private 分层，检验 hits↑ → Δ↑ → KA_c↑ 链条稳定性。

### 6.1 A1：历史 key/anchor 分槽（低成本恢复基线）

旧任务推理时取用该任务学习世代的 key/anchor 快照（存储 ~6MB/客户端）。检验"旧接口隔离是否足以消除遗忘"。预期效果 = 2×2 的 co 组合（见 §6.5 预注册表）。

### 6.2 A2：路由碰撞抑制（机制验证核心）

新任务训练时保护非当前类槽位（梯度掩码 / 路由约束）。若干预后旧类 KA 损伤显著下降，"槽位劫持"因果链闭合——论文机制贡献的关键实验。注意世代共适应（§4.6）：纯冻结可能损害当前任务，需与 A3 对比。

### 6.3 A3：跨世代兼容性校准

用旧原型/代理特征约束当前 prompt 与历史 key/anchor 的功能一致性，替代"全部冻结"——突破 co 上界的增量空间（补 P 通道）。

### 6.4 A4：兼容性感知联邦聚合

A3 有效后，服务端聚合层降低跨客户端接口冲突——论文"联邦侧"贡献落点。

### 6.5 A1/A2 部署态等价与增益预注册

A1（快照恢复）与 A2（训练期冻结）最终都落在 co 组合（当前 prompt + 旧 key/anchor），其精度**已被 2×2 诊断直接测出**：

| old task | 部署态 cc | A1/A2 预期（= co，预注册） |
|---|---|---|
| T0 | 85.67 | 93.05 |
| T1 | 71.41 | 93.85 |
| T2 | 72.42 | 84.83 |
| T3 | 70.53 | 92.94 |

修复实验若与 co 对齐 → 机制归因闭环。A2 独立价值 = 训练中途连续性保护（每轮 `evaluate(0)` 广播态受益）+ 劫持假说因果证据。

### 6.6 修复验证标准

- KRt → ~100%（对齐 old/old 组合精度）
- AA 78.6 → 8x；T1–T3 保留率恢复（~71 → 9x）
- T0 复核：区分机制修复 vs 指标修复（T0 已受结构性偏护）

## 7. 论文定位与新颖性格局

- **差异化支柱**：① 公共/私有不对称是联邦独有遗忘签名（经本地接口演化执行，因果强度按 D1/A2 定稿）；② 组件级归因（prompt/key/anchor/head 四件 2×2）vs 近亲工作的层级拼接；③ SOTA"无遗忘"主张的协议盲区批判（bit-exact 复现 + 20pp 差距）；④ 零重训的世代分槽修复
- **近亲必引**：arXiv 2606.02860《Forgetting is Not Erasure: Recovering Latent Knowledge via Transport Keys》（centralized CL，2026-06）——interface drift + stitched evaluation + transport keys，结构与我们的 2×2/分槽同构；必须引用差异化，不能声称概念首创；定位为"并发现象的 centralized 证据"，我们为 FCL 实例化 + 更深因果分析
- **联邦侧空档已确认**：FCL 文献（HiGP、STSA、Fed-TaLoRA、ACM 2025-12 survey）无"接口失配≠知识擦除、组件级世代交叉评估、公共/私有类不对称、零重训恢复"任一主张
- **叙事策略**：概念被外部（2606.02860）验证反而降低论证风险；我们提供 FCL 侧审计协议 + 因果链 + 联邦签名 + 通用修复框架
- 多方法验证：2 方法（FedTA + GLFC 或其他 prompt-based FCIL）× 2 数据集（CIFAR-100 + ImageNet-R）

## 8. 未决事项

1. seed 0/1 复跑分析：三 seed KRt 方差上界（残余段差距的 seed 假设检验）
2. FedTA 作者 GitHub issue 回应（KRt 测量状态 / 硬编码参数 / seed 数）
3. CVF 高清版 Fig 4(b) FedTA 曲线五个精确值读取
4. ImageNet-R 方向 (c) 诊断（2×2 协议不依赖 public classes，适用）
5. GLFC（或第二方法）2×2 审计 → 多方法验证
6. Phase 4 D1 → A1 → A2 → A3 → A4 依序实施（§6）

## 9. 硬约束与工程约定（仍然有效）

- 不修改：FedTA 训练 loss、optimizer、BGPS、SIKF、FedAvg head、CIFAR federated split、Tail Anchor 结构、prompt 逻辑 / batchwise_prompt；`--method fedta` read-only
- 修复方案只动推理/接口层；训练期干预（A2）在新方法 flag 下实验
- 所有新增 diagnostic/evaluation 必须 observer-only；消耗 RNG 的代码必须 snapshot/restore
- 语义/特征诊断只用 TRAIN split；test data 永远禁止（analyze_multimodality 类脚本 `--split` 硬校验）
- 数据集仅 cifar100 / ImageNet-R；本地导入（download=False）；`--data-path` 指定目录
- 跨类聚合必须样本加权（本地测试集 Dirichlet 切到类级，未加权会产生假偏差）
- 对账验证协议：diagnostic 评估值须与 compatibility_drift console 输出精确吻合（固定种子 `9700+100*cid+old_task` 坐标系）
- checkpoint 兼容：`torch_load_compat()`；`load_checkpoint()` 对缺失/空 `outer_split_manifest` 抛 RuntimeError；resume 必须重生成并逐位比对 manifest，不一致拒绝恢复
- 每新增测试命令同步更新 `RUNCOMMANDS.md`
- 评估协议对齐：`global_epoch=5`、`threshold=0.25`（Gate 0A 口径）

## 10. 命令索引

全部运行/诊断/测试命令见 [RUNCOMMANDS.md](file:///d:/MyWork/FedSMTA/RUNCOMMANDS.md)。关键诊断入口：

- §9.3 compatibility_drift.py（2×2） / §9.3.1 analyze_class_retention.py（4 组合 per-class） / §9.3.2 probe_channel_correlates.py（子机制探针）
- 数据文件（服务器）：`diagnostics_output/compatibility_2x2_t{00..03}_t04.csv`、`class_retention_4combo.csv`、`channel_correlates.json`
