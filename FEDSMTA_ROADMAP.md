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

> 2026-10-10 起完整失败方案总存档合并至文末 **§11**（含 Phase 4 判死项与 RUNCOMMANDS 命令清理记录）；本表保留 Phase 4 之前的早期判决。

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

### 6.0 D0 路由拓扑识别（已完成，2026-10-10，verify_slot_binding.py）

D0 是 D1 的前置 gateway：**slot id = class id 假设已判死**（diag_share=0.000/198 类），per-class 槽位归因必须走经验绑定。结果：

- **绑定涌现且高度集中**：modal_share=0.906、entropy 0.41 bits——每类 ~90% 样本路由到单一 modal slot，modal_slot 可作类接口代理
- **枢纽拓扑（hub-and-spoke）**：within-task purity 0.920，但跨任务争抢严重——65/198 exclusive，top 枢纽槽位被 10–28 个类共享（单槽 1400+ hits，占该客户端全部样本 ~60%）；modal slot 大量落在 ≥100 的未训练噪声槽区（接口位置任意但功能确定）
- **路由 2×2 分解与 accuracy 2×2 对桥成功**：key 通道路由翻转 74.1%（stab(R_oo,R_oc)=0.259）vs feature 通道 7.0%，总翻转 73.4%——与 accuracy 级 KA 份额（58–96%）同量级，"key 漂移→误路由→KA 损伤"链条第一段（key 漂移→误路由）实证成立；交互 0.077 为**重叠**（双通道共翻样本），非协同
- **t0/t1 旧任务 key 通道路由完全洗牌（stabK=0.00）**，越旧翻转越彻底
- **解释注意**：purity_client 按累积序计算，只含 ≤t 的任务——**未来任务碰撞（D1 核心方向）尚未测**，但可从 D0 已存的 npz 矩阵离线计算（offline D1-lite，零 GPU）

**D1 归因形态判决：加权多槽位**（E_c = Σ_j B(c,j)·Conflict(j)，用完整 B 分布而非单槽）。

**D1-lite 判决（2026-10-10，本地离线：D0 npz/csv + 4combo join，158 class-units）——简单碰撞链证伪**：

- E_future（未来流量对旧支撑集的碰撞暴露）vs KA：**rho=−0.197（p=0.013，负向）**；CEI_all vs KA：rho=−0.218（负向）——方向与预期相反
- per-class routing_stability / stab_key vs KA：rho≈0（**范围受限**：加权 stab≈0.11 近乎全类翻转，无分辨方差）
- 唯一显著保护因子：**modal_share vs KA：rho=+0.239（p=0.003）**——绑定越强损伤越小
- **失败原因诊断——支撑集错位（support misalignment）**：E_future/CEI 度量旧支撑集（旧特征曾去之处）的未来流量；但 key 通道路由翻转 74% 后，旧特征在当前 key 下**落在别处**（R_oc 落点），损伤发生在**落点槽**的 anchor 状态，不在旧支撑集。我们度量了旧特征已不再光顾的区域
- **存活结论**：聚合级 KA 主导（accuracy 2×2）、key 通道路由主导（D0 routing 2×2）、A1 修复（co 预注册增益）均不受影响；被证伪的只是"旧支撑集碰撞→损伤"这个最简版本
- **修正方向（attractor 假说）**：当前 key 下旧任务特征坍缩到少数吸引子槽（其 anchor 服务当前类）→ 混合注入当前类 anchor → 旧 head 误判。类间损伤方差来自落点分布与落点 anchor 状态——需要 R_oc 落点分布（D0 已算出 slots_oc 但未存矩阵，已补存，需重跑 D0 一次）【勘误 2026-10-10：落点分析（见下方判决）否证了"吸引子槽 anchor 服务当前类"与"方差来自落点 anchor 状态"两处——落点 64.6% 在噪声区、去向无预测力；坍缩本身与"离开支撑集"两个事实存活】
- 教训：预注册链检验的价值——先定判据再看数据，证伪同样是产出

**D1-lite FE 补充判决（2026-10-10，two-way (client,task) FE + 相对损伤重标定）**：

- **pooled 相关全部是混杂伪影**：E_future 的负相关（−0.197）在 FE 下消失（beta=−0.009, p=0.35）；绝对 KA 的主导预测因子是 **baseline 精度 oo（beta=+0.77, p<1e-10）**——绝对损伤 ∝ 基线精度（天花板机械效应），损伤分析必须用相对量 KA/oo
- **相对损伤（KA/oo、total/oo）的 within-cell 存活因子**：
  - **modal_share 正向**（KA_rel beta=+0.55, p=0.003；total_rel +0.70, p=0.018）——**方向反转**：pooled 的"绑定保护"（+0.239）是 Simpson 伪影；同 cell 内**绑定越强、相对损伤越大**。机制解释（集中坍缩假说）：强绑定类的特征质量集中于 key[j*] 附近，key[j*] 漂移时**整类样本一次性全部误路由**（all-or-none），弱绑定类支撑分散、只部分误路由
  - **routing_stability 负向**（total_rel beta=−0.46, **p=0.036**）——**类级正向链证据首次出现**：路由保持的类相对漂移更小
  - n_test_log 正向（KA_rel p=0.02）——小样本类的损伤被统计噪声压低
- **E_future / CEI 在所有规格（pooled/FE/绝对/相对）下均无预测力**——"旧支撑集碰撞→损伤"故事彻底判死，支撑集错位诊断成立，落点（attractor）测量是唯一活着的替代假说
- **P/oo 相对损伤无可预测因子**（within-R2=0.012）——prompt 通道损伤呈类无关噪声形态
- 对 A2 的含义：干预目标不是"保护旧支撑集流量"而是**防止整类集中误路由**（集中坍缩机制）——与"按类保护 modal slot"的 A2 原型一致，但保护判据应是"绑定强度 × 路由保持"而非流量

**落点分析判决（2026-10-10，analyze_landing_spots.py，D0 重跑 npz 的 `_oc` 落点矩阵 × D1-lite 损伤，158 class-units，158/158 modal 对齐验证通过）**：

- **R1 证伪（落点版碰撞故事同死）**：落点槽未来流量 H_landing vs KA_rel：FE rho=+0.055（p=0.49）——支撑集暴露与落点暴露在两个坐标系下**均无预测力**；"碰撞→损伤"的流量形式全面死亡。落点去向三分（noise/future_served/idle）的 KA_rel 完全平坦（0.10–0.14），on_noise、landing_owner_share、landing_entropy FE 全部归零——**损伤不在"落到哪"，只在"是否离开支撑集"**
- **R2 存活（迄今最强类级保护因子）**：support_preserve vs KA_rel：FE rho=−0.197（**p=0.014**）；vs total_rel：−0.180（p=0.025）——留在自身支撑集内的类相对损伤更小；modal_preserve 单独无预测力（支撑集内任何槽位保持皆有保护作用，非仅 modal）
- **R3 单边坍缩判决**：强绑定类（modal_share>0.8，占 83% 测试质量）中 **90.4% modal_preserve<0.2、仅 4.8%≥0.8**——非 U 型、非渐进，是**近乎完全的单边坍缩**：key 漂移对旧 modal 路由几乎彻底抹除。但坍缩类 wKA_rel 仅 0.12——**损伤量级远小于路由坍缩量级**：anchor 为轻度加性修正，错 anchor 只温和扰动 head（与"路由翻转 74% vs KA 遗忘 ~19pp"的量级差一致）
- **R4 强确认（attractor 坍缩）**：每客户端 32 旧类的 modal 槽从 12–17 个坍缩到 **4–8 个吸引子槽**，top-10 落点槽承载 86–96% 落点质量；**64.6% 旧类的落点 modal 落在 ≥100 未训练噪声区**
- **公共/私有不对称首次获得路由层坐标**：public 类 modal_preserve 0.059 < private 0.100，support_preserve 0.207 < 0.297——公共类被逐出支撑更彻底，与其更高 KA_rel（0.176 vs 0.119）方向一致；联邦签名在路由层有了直接对应物
- **关键推理（D1 v2 判读指引）**：64.6% 旧类被**方向冻结**的噪声槽 key argmax 胜出 ⇒ 旧支撑槽 key 的**方向必然移动过**（路由只看方向）。但边界态流量测不到这些槽的访问（E_future 无预测力）⇒ 支撑槽 key 方向改变来自**训练轮内的瞬态路由命中**（task-end 边界态测量系统性漏检）。判决量：D1 v2 的 **d_key_dir**（l2 归一化 1−cos，路由相关位移）应正向预测 KA_rel，而 d_key_norm_ratio（wd 签名）不应；hit/nohit 边界态分层预期失灵——瞬态命中是漏检的混杂【勘误 2026-10-10 D1 v2 FE 判决（见下方）：此推理两条皆废——"噪声槽方向冻结"不成立（耦合 wd Adam 使未命中槽同样随机游走），瞬态命中假说被更简洁的 wd 随机游走机制替代；"d_key_dir 正向预测 KA_rel"的预期也证伪（FE +0.010 n.s.）——key 位移不直接致伤，损伤走 anchor 通道】
- **A2 设计输入修正**：损伤机制最终形态为 **"key 方向漂移 → 旧特征被逐出支撑集 → 错 anchor（去向无关）→ 轻度 head 扰动累积"**。干预目标是**旧支撑槽 key 的方向稳定性**（freeze + wd 豁免 > 流量掩码），保护判据 = 支撑集保持（support_preserve）。另新增廉价修复候选：旧特征坍缩到 4–8 个吸引子槽且高度集中 ⇒ **少量吸引子槽 anchor 校准**（对齐旧类原型，零重训、每客户端仅数行参数）可作为 A3 的输入形态

**D1 v2 FE 判决（2026-10-10，collision_exposure_diagnostic.py 服务器运行 + 本地 (client,task) FE 重算，158 class-units 与落点分析 join）——机制链闭合，损伤主通道 = anchor 内容失配**：

- **链 1（暴露→扰动）真实且双通道分裂**（FE）：E_future vs d_anchor **+0.274（p=5e-4）**、vs d_key（raw）+0.390、vs norm_ratio **+0.631（p=6e-19）**；但 vs d_key_dir **−0.477（p=2e-10，负向）**
- **wd 随机游走机制确认**（替代瞬态命中假说）：`Client_DF` 的 Tail_Anchor 挂在**耦合 wd 的 torch.optim.Adam**（L2 进梯度 + 逐元素自适应缩放）下——未命中槽位的 wd 步长与范数无关（≈lr·sign 翻转），方向**随机游走**、范数塌缩至 ~1e-2（norm_ratio 中位数 0.016）；命中槽位被梯度钉住方向、维持范数。两类槽产生**相当的 raw 位移**（hit/nohit 比值 0.91×）——raw ||ΔK|| 是盲指标，方向/范数分解是必需的。此前"瞬态命中假说"不再必要；"噪声槽方向冻结"推断作废（耦合 wd 下噪声槽同样随机游走）
- **链 2（扰动→损伤）单通道存活**（FE）：**d_anchor vs KA_rel +0.259（p=0.001）**，partial 掉 support_preserve 后 **+0.268（p=8e-4）**仍显著；d_key（raw +0.109 n.s. / d_key_dir +0.010 / norm_ratio +0.058）**全部无预测力**——key 位移不直接致伤
- **逐出链强但不通向损伤**：d_key_dir vs modal_preserve **−0.622（p=2.6e-18，全 Phase 4 最强单环）**——方向扰动驱动路由逐出；但 modal_preserve vs KA_rel −0.012（n.s.），落点去向无关（R1）——**逐出本身不是主要损伤路径**（错 anchor 仅轻度加性扰动，量级解耦见 R3）
- **损伤机制最终形态（三因子独立存活，partial 检验后均显著）**：KA_rel ~ **anchor 位移**（+0.268，正）+ **support_preserve**（−0.221，保护）+ **modal_share**（+0.310，脆弱）。语义：未来流量命中旧支撑槽 → anchor 被改写服务新类（主损伤）；未命中槽 wd 随机游走 → 路由逐出（扰动但轻伤）；绑定越集中整类越脆弱（集中坍缩）；全程池级污染解释"落点去向无关"——池内所有 anchor 均被扰动（改写或随机化），落哪都拿到错 anchor
- **联邦签名第三层证据**：public 类 wE_future=284 > private=215、w_d_anchor=6.19 > 5.42；d_anchor→KA_rel 链 public +0.228（p=0.08）而 **private −0.027（消失）**——公共类的支撑槽更多被未来流量命中、anchor 改写更深，anchor 通道损伤集中于公共类
- **A2 设计定稿**：冻结旧类支撑槽的 **key + anchor 两者**（freeze + wd 豁免必须——耦合 wd 下纯冻结仍会被随机游走破坏）；槽位选择判据 = D0 涌现绑定（per-class support），非流量。差异化预注册预测：双冻结应同时恢复 modal 路由（d_key_dir→modal_preserve 链）与 anchor 内容（d_anchor→KA_rel 链）；只冻结 anchor 预测损伤部分下降但路由仍坍缩——三臂 A2（key-only / anchor-only / both）可直接判别双通道
- **routing-instability 家族判死（GPT-D1.5 提议的离线检验，2026-10-10）**：KL(P_own‖P_land) vs KA_rel FE +0.028（p=0.73）、JS +0.119（p=0.14）、routing_stability（=1−RI）−0.098（p=0.22）——全部 null；头对头 partial：KL | d_anchor = +0.088（死），d_anchor | KL = **+0.280（p=4e-4，存活）**——anchor 通道严格支配整个 routing-instability 家族。**routing instability 定位为碰撞的表现形式（eviction 现象），不是损伤原因**。KL/JS 为排除性负结果，正式留档；不再扩展 routing 指标挖掘
- **机制命名升级（外部审计收敛，2026-10-10）**：论文措辞弃用 "slot hijacking"（过强且已被证伪），改为 **"emergent interface sharing → anchor prototype contamination"**——涌现接口共享缺少语义隔离，后续任务的梯度与 wd 使 anchor 原型被持续污染；key 方向随机游走导致路由逐出（真实但轻伤），损伤主通道是 anchor 内容失配。一句话主张：**FedTA 的遗忘不是因为旧槽位被动了，而是共享路由接口允许语义 anchor 被覆写**。注意三因子结构保留：support_preserve（−0.221）独立存活，key/routing 非纯伴随而是弱独立保护/脆弱因子（modal_share +0.310）
- **public/private 独立 FE（补算）**：public 内 modal_share +0.295（p=0.022）显著、d_anchor +0.228（p=0.08）边缘；private 各 (client,task) cell 样本 <5 被剔除后 FE 欠功率，null 不作确认结论

对每个旧类的 modal slot 及完整路由分布统计三元关系：

1. **碰撞暴露 E_c**：后续任务训练样本（边界态路由）落入旧类 modal slot / B(c,·) 支撑集的加权频次——离线版可直接从 D0 矩阵算（未来任务类按各自 task_end 态路由进旧类槽位的 hits）
2. **参数位移**：modal slot 的 key/anchor 从 t 末到各边界再到 task_04_end 的位移，**按 hits>0/hits=0 分层**——分离"劫持介导"与"wd 衰减介导"两种损伤路径（Adam wd=1e-3 使未命中槽位也位移），直接决定 A2 用梯度掩码还是冻结+衰减豁免
3. **KA 损伤**：join 4combo CSV（已有）

**判据**：按 (client, old_task) 分组 + public/private 分层，检验 E_c↑ → Δ_slot↑ → KA_c↑ 链条稳定性；命中粒度取决于 round checkpoint 是否保留（逐轮 vs 任务边界代理）。

**D1 v2 预注册阅读顺序（2026-10-10 审阅定稿；跑完先看这四条，再看其余 rho）**：

1. E_future → d_key：未来暴露是否导致接口位移
2. d_key → KA_rel：接口位移是否与相对损伤相关
3. E_future → KA_rel：完整链是否成立
4. hit/nohit 加权位移比 D_hit/D_nohit：**>2× 支持劫持介导 → A2 选梯度掩码；≈1× 说明 wd 衰减主导 → A2 选冻结+衰减豁免**

辅助量：D_traffic（流量加权位移，Σ_j p(j|c)·(F_j/ΣF)·||ΔK_j||）——hit/nohit 是槽位级二分，D_traffic 分辨"被访问"与"被重度访问"；若 d_hit>d_nohit 但 D_traffic 无额外预测力，只能下"被访问 slot 更易漂移"的弱结论。主指标 E_future + H_weighted，CEI 放 supplementary；主文相对损伤 KA_rel，绝对 KA 放补充。

**论文级分析（跑完数据后执行，不进 D1 脚本）**：层级回归 KA_rel ~ β1·E_future + β2·d_key + α_client + γ_task + ε，d_key 作 mediator（E_future → d_key → KA_rel 路径），单位 (client, task, class) 需控制 client/task 效应——描述性 Spearman 仅为探索，机制主张以 FE 回归 + A2 干预为准。

### 6.1 A1：历史 key/anchor 分槽（低成本恢复基线）

旧任务推理时取用该任务学习世代的 key/anchor 快照（存储 ~6MB/客户端）。检验"旧接口隔离是否足以消除遗忘"。预期效果 = 2×2 的 co 组合（见 §6.5 预注册表）。

### 6.2 A2：接口污染干预（四臂因果实验，机制验证核心）

机制待因果判定（D1 v2 FE 判决）：**anchor 内容污染为损伤主通道**（d_anchor→KA_rel +0.259），key 方向随机游走为路由逐出通道（弱独立保护因子 support_preserve −0.197/−0.221），绑定集中度为脆弱因子（modal_share +0.292/+0.310）。

**臂设计**（baseline 复用 seed42 run，零新增计算；三臂各需一次完整 5 任务联邦训练）：

| 臂 | 保护对象 | 预注册预测 |
|---|---|---|
| control | 无（复用现有 run） | — |
| A2-key | 旧支撑槽 key 行 | modal/support_preserve 恢复高位；KA_rel **部分**改善（仅来自 support 保持保护因子）；改善幅度 < anchor 臂 |
| A2-anchor | 旧支撑槽 anchor 行 | d_anchor(protected)→0；**KA_rel 显著下降（主通道判定）**；modal_preserve 仍低（路由仍坍缩）；若 KA_rel 不降 → anchor 污染机制**证伪** |
| A2-both | key + anchor 双冻结 | KA_rel 降至 §6.5 co 组合预注册水平；逼近 A1 上界；cc（当前任务精度）不显著受损为接受门槛 |

**实现（freeze + wd 豁免的干净形式）**：训练期 **post-step restore**——任务开始时快照保护行，本地 Tail_Anchor 每次 `optimizer.step()` 后从快照恢复。不改优化器构造、不消耗额外 RNG、不动 loss/训练逻辑——Adam 动量中残留的 wd/梯度冲量被覆盖（等价 wd 豁免），未保护行更新语义与 baseline 逐位一致。

**保护集**（进入新任务 t 时计算）：客户端所有已完成任务 t'<t 的训练特征经当前 key top-1 路由，逐类累计路由质量 >0 的槽位（=D0 支撑集定义，B_t>0）取并集。运行时从活体模型计算，不依赖离线文件。

**新方法 flag**：`--method fedta_a2` + `--a2_arm {key,anchor,both}`；`--method fedta` 基线路径保持只读（Phase 0 冻结约束）。

**世代共适应风险**（§4.6）：both 臂压缩新任务可用槽位，若 cc 显著受损则冻结路线让位于 A3 校准路线——此为 A3 的存在性判据。

**证伪也是产出**：任何臂方向反号或量级严重偏离三因子模型排序（anchor > key）→ 相关模型未含全部损伤路径，回补诊断并按预注册留档。

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

- §9.3 compatibility_drift.py（2×2） / §9.3.1 analyze_class_retention.py（4 组合 per-class） / §9.3.2 probe_channel_correlates.py（子机制探针） / §9.3.3a verify_slot_binding.py（D0） / §9.3.3b collision_exposure_diagnostic.py（D1 v2） / §9.3.3c analyze_landing_spots.py（落点分析）
- 数据文件（服务器）：`diagnostics_output/compatibility_2x2_t{00..03}_t04.csv`、`class_retention_4combo.csv`、`channel_correlates.json`、`slot_binding.csv` / `slot_binding_matrices.npz`、`collision_exposure.csv/.json`、`landing_spots.csv/.json`
- 已测验失败方案的命令已从 RUNCOMMANDS.md 移除（留 tombstone 保引用），见 §11

## 11. 已测验失败方案总存档（2026-10-10 整理）

> 本节是全项目失败判决的**唯一汇总处**（§5 表为其 Phase 4 之前子集）。所有条目均有数据留档与预注册判据；失败判决等同产出，是排除法证据链的组成部分。关联命令已从 RUNCOMMANDS.md 删除（节号留 tombstone），旧命令文本在 git 历史（commit cd26db0 及之前）。

### 11.1 Phase 1 / Phase 2 方向（整线判负）

| 方案 | 判决 | 依据与日期 |
|---|---|---|
| 多模态主线：Semantic-K2 / Oracle-K2 / UOT（Phase 2 全线） | **已测验失败（判负）** | Phase 1（halt criterion）：robust multimodal prevalence **0/25**（ρ、D_cover、client 归因、切分形态、z_ref 对照五证据链全阴性）；Phase 2 冻结。RUNCOMMANDS 已删：§9.2 多模态分析、§9.4 synthetic Oracle-K2、§14 z_op precheck |
| Synthetic Oracle-K2 压力测试 | **随主线废弃** | synthetic sanity check 依附于 K2 方向，方向死即无意义 |

### 11.2 Phase 4 诊断链内被证伪的假说（证据链本身存活，负结果已并入论文排除法）

| 方案 | 判决 | 依据与日期 |
|---|---|---|
| "slot id = class id" 显式绑定假设 | **已测验失败** | D0（verify_slot_binding.py）：diag_share=0.000 / 198 类；路由纯相似度、绑定涌现形成 |
| 简单碰撞链：旧支撑集未来流量 → 损伤（E_future → KA） | **已测验失败** | D1-lite FE：beta=−0.009 (p=0.35)，pooled 负相关为 Simpson 伪影；D1 v2 FE 复确认无预测力 |
| slot hijacking（流量劫持 modal slot → 损伤） | **已测验失败** | D1 v2：hit/nohit raw 位移比 0.91×（无分离）；损伤不随边界态流量分层；论文措辞弃用 |
| 落点流量假说（H_landing → 损伤）与 attractor 服务当前类故事 | **已测验失败** | 落点分析 R1：FE rho=+0.055 (p=0.49)；落点去向三分 KA_rel 全平坦；64.6% 落噪声区 |
| 瞬态命中假说 + "噪声槽方向冻结" 推理 | **已测验失败（作废）** | D1 v2 FE：E_future vs d_key_dir **负向** −0.477（方向相反）；被耦合 wd Adam 方向随机游走机制替代；d_key_dir → KA_rel 亦 null（+0.010） |
| D1.5 routing instability → KA（KL / JS / agreement 家族） | **已测验失败** | 2026-10-10 离线检验：KL +0.028 (p=0.73)、JS +0.119 (p=0.14)、routing_stability −0.098 (p=0.22) 全 null；d_anchor \| KL = +0.280 存活而 KL \| d_anchor = +0.088 死——routing instability 仅为碰撞的表现形式，非损伤原因 |
| key 位移（raw / 方向 / 范数）→ KA 直接因果 | **已测验失败** | D1 v2 FE：+0.109 / +0.010 / +0.058 全 n.s.；损伤主通道为 anchor 内容失配（d_anchor → KA_rel +0.259, p=0.001，partial 后存活） |
| 原版 D1 单槽归因脚本（slot_collision_diagnostic.py） | **设计判死，未运行** | 依附 slot id 假设（11.2 第 1 条）；被 D1 v2（collision_exposure_diagnostic.py）取代。RUNCOMMANDS 已删 §9.3.3 |

### 11.3 早期假说（§5 表详解，此处仅存目）

公共类跨客户端强化假说（retention 反向 74.4% < 83.9%）、聚合介导联邦独有机制（85% 走本地 KA 通道）、协议结构性脆弱假说（方向相反）、"超可加交互"措辞（降级为相关性）、FedTA++ 引用支柱（放弃）、supra-baseline KRt 复现（不可达，作附录 reproducibility note）——判决依据见 §5 表。

### 11.4 存活结论速查（失败存档的反面，防误伤）

以下结论**不受上述证伪影响**，为当前论文证据链主干：表观遗忘判决（old/old 复原 KRt=100%）、KA 通道主导（58–96%）、公共/私有不对称（74.4% vs 83.9% + anchor 通道 public 内边缘显著）、anchor 内容失配主通道（三因子模型）、A1/A2 修复预注册（co 组合增益）、集中坍缩与支撑保持保护因子。
