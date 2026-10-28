# FedSMTA 研发路线（v4，2026-10-20 精简重写）

> v4 删除 v3 中诊断链的逐步演进叙述与勘误嵌套，只保留**最终判决形态**。完整过程性记录（D0/D1-lite/D1 v2/落点分析逐步判决、TKR 复算细节）在 git 历史：`git show 2face47:FEDSMTA_ROADMAP.md` 及更早。

## 1. 项目定位

**FCIL 中被忽视的接口失配型表观遗忘**：FedTA (CVPR 2025) 的"无遗忘"结论是评估协议盲区的产物；真实遗忘（18.89pp）由推理期组件接口失配主导（非知识擦除），可在零重训的接口层修复。措辞定稿：**"机制通用、签名联邦"**——KA 接口失配为通用机制，联邦独有的是公共/私有不对称遗忘签名。机制贡献 > 性能贡献；目标顶会。

研究链条：审计协议 → 机制归因 → 联邦签名 → 接口稳定性修复（A2）→ 共享语义接口（A5）→ 多方法验证。

## 2. 基线事实与复现（Gate 0 已封板）

- 架构：冻结 ViT-B/16；prompt 池 SIKF 聚合全局共享；Tail_Anchor（key+anchor 各 200 槽、top-1 路由）纯本地不聚合；per-task Chead 旧任务恒用任务专属 head
- 协议（CIFAR-100）：5c×5t×8 类；每客户端 15 私有+25 公共；公共类在本客户端后续任务不重现；公共类本地数据 Dirichlet 切分（~100/类 vs 私有 500）。ImageNet-R：5c×5t、每客户端 40 私有类、**0 公共类**、surrogate 固定 5/类
- Gate 0A/0B/0C 全 PASS（2026-10-08）：与官方 FedTA 13 项 D=0；resume 22 项全绿；observer 不变性 29 项全绿。`--method fedta` 为 read-only baseline（tag `phase0-fedta-baseline`）
- seed42 部署态（after4）：T0–T4 = 85.66/71.10/71.45/71.13/93.77；AA=78.62，标准遗忘 F=18.89；对角线均值 93.73 ≈ 论文 93.70

## 3. 已验证结论主干（全部有数据留档）

1. **协议盲区**：Table 1 是对角线表；KRt 只测 task 0 且受每轮 `evaluate(0)` 偏护；T1–T3 崩至 ~71 对三指标同时不可见
2. **表观遗忘**：old/old 组合精确复原训练时精度（±0.5pp）→ KRt=100% 在参数层面成立；损伤非单调可回升 → 排除知识擦除
3. **2×2 归因**：KA（key/anchor）通道占总漂移 58–96%（compatibility_drift.py）
4. **事件驱动**：单窗口损伤 ≈ 多窗口 → 任务切换时一次性驱动
5. **世代共适应**：prompt 与 key/anchor 成对漂移，单换任一接口反而更糟 → 修复必须保持接口世代一致
6. **公共/私有不对称（联邦签名）**：public retention 74.4% < private 83.9%；超额损伤 85% 走本地 KA 通道；anchor 通道损伤集中于公共类（public 内 +0.228 vs private −0.027）
7. **损伤机制最终形态（三因子，D1 v2 FE 判决）**：`KA_rel ~ anchor 位移(+0.268) + support_preserve(−0.221) + modal_share(+0.310)`。语义：未来流量命中旧支撑槽 → anchor 被改写服务新类（**主通道**）；未命中槽耦合 wd Adam 随机游走 → 路由逐出（轻伤）；绑定越集中整类越脆弱（集中坍缩）。一句话：**FedTA 的遗忘不是旧槽位被动了，而是共享路由接口允许语义 anchor 被覆写**
8. **TKR 复现结案（2026-10-20）**：作者口径三种子平均（42/1999/2025，= 作者协议）= 96.4/98.4/96.1/89.6%，t=1..3 吻合论文 "around 98%"；此前 20pp 差距 = seed_42 单种子（恰为最差种子）。唯一残余 = Fig 4(b) 105–115% 读数复核（可能我方读图误读）

## 4. 失败判决存档（唯一汇总处，含预注册判据与数据留档）

| 假说 | 判决 | 一句话依据 |
|---|---|---|
| 多模态主线（Semantic-K2/Oracle-K2/UOT） | 死亡（2026-10-10） | robust multimodal prevalence 0/25，halt criterion 触发 |
| 公共类跨客户端强化 | 死亡 | retention 方向相反 |
| 聚合介导联邦独有失配 | 死亡 | 超额损伤 85% 走本地 KA 通道 |
| 协议结构性脆弱（Dirichlet 欠训练） | 死亡 | KA~n_test 正相关，方向相反 |
| "slot id = class id" 绑定 | 死亡（D0） | diag_share=0.000，绑定涌现 |
| 简单碰撞链（E_future→损伤） | 死亡（D1-lite/v2 FE） | 全规格无预测力，pooled 为 Simpson 伪影 |
| slot hijacking / 落点流量 / attractor 服务当前类 | 死亡 | hit/nohit 位移比 0.91×；落点去向 KA_rel 全平坦 |
| 瞬态命中 + 噪声槽方向冻结 | 作废 | 被耦合 wd Adam 随机游走机制替代 |
| routing instability（KL/JS 家族）→ 损伤 | 死亡 | 全 null；d_anchor 严格支配（+0.280 存活） |
| key 位移直接致伤 | 死亡 | d_key raw/dir/norm 全 n.s.，损伤走 anchor 通道 |
| "超可加交互"措辞 | 降级 | 仅相关性，缺对照臂 |
| supra-baseline KRt 复现 | 已解释（结案） | seed 方差主导，见 §3.8 |

存活反面（防误伤，论文证据链主干）：§3 全部结论 + 集中坍缩/支撑保持保护因子 + A1/A2 预注册。

## 5. Phase 4：A2 接口保护（已实现，INR 已跑，机制验证核心）

- **机制**：post-step restore（任务开始快照保护行，每次 `optimizer.step()` 后恢复 = freeze + wd 豁免，零优化器改动）；保护集 = 已完成任务训练特征经当前 key top-1 路由的支撑槽位并集
- **臂**：key / anchor / both（`--method fedta_a2 --a2_arm {key,anchor,both}`）；预注册判读：anchor 臂 KA_rel 降 = 主通道因果成立；both 臂逼近 2×2 co 上界且 cc 不受损
- **co 上界预注册**（CIFAR-100 seed42）：T0–T3 部署 cc = 85.67/71.41/72.42/70.53 → co 预期 = 93.05/93.85/84.83/92.94
- **已得结果（INR seed42，2026-10-20）**：FedTA 77.12 / A2 79.43 / C²P 91.14（after-T4 AA）；A2 遗忘压至 2.5；差距在基础精度（T0 即落后 C²P 3.99）而非保持
- 回归测试三件套（T1 resume / T2 拒绝语义 / T3 A1 parity）见 RUNCOMMANDS §16.1

## 6. A5：共享语义接口（当前活跃方向，2026-10-20 定稿）

**架构**：Stable Historical Interface + Evolving Shared Semantic Interface（双速接口）。A2 快照保护私有任务 prompt+key/anchor；全局类原型/共享层每轮向全局统计对齐。C²P 与 A2 作用于不同参数集与失效通道，正交可组合。

**P0 判决（manifest 审计，2026-10-20）**：CIFAR-100 = 75 私有类 + 25 公共类（每公共类 5 客户端全持有）；**ImageNet-R = 200 类全部单客户端、0 公共类**（协议算术 200−5×40=0，与 seed 无关）。推论：INR 上 LCDC 跨客户端同类补偿通道确定关闭（模块残余正则化作用需消融分离）；C²P 在 INR 的 +11.71 全部来自跨类机制（CPA/分布感知原型头/类级辅助 prompt）；INR 是干净的"跨类语义迁移"试验场，public/private 分层判别只能在 CIFAR-100 做。

**差距记账（INR after-T4）**：C²P−A2 = 11.71pp = T0 起点差 3.99（原型头+对比损失的容量优势，非迁移）+ 后续差距扩大 7.72（持续演化贡献的**上界**，含 prev-task 重训兜底；真归因需完整 matrix+BWT+受控消融）。C²P 数据访问 = prev-task full re-access（mover 分支 L673-685，官方日志逐位一致证实激活），强度高于 GLFC exemplar buffer。

**P1：frozen NCM probe（已完成，2026-10-20，INR seed42，`diagnostics/ncm_probe.py`）**：z_ref（冻结特征，可在线增量维护）+ z_op（当前 prompt 特征，oracle/辅助）上的 cosine/Euclidean NCM vs native 路径。防泄漏构造（原型仅来自已见任务 inner.train）；缓存身份 fail-closed 校验；`--prompt_task_mode current` 对齐 native evaluate 的 task_id。**T0-gap 判据**：NCM 若在 T0 缩小 ~4pp 差距 → A5 首先是基础判别能力修复；增益只在后续边界 → 支持持续迁移叙事。INR 上的 NCM 增益只能证明分类接口价值，不得写成 federated transfer。

**P1 结果与判决**：AA（T0/T4）：native 82.87/79.43；z_ref NCM 79.06/76.04；z_op NCM 82.49/80.62；C²P 86.86/91.14。不变性对账通过（z_ref 旧行 T0/T4 逐位一致，probe 完整性确认）。判决：**T0-gap 非分类接口瓶颈**——z_op NCM(82.49)≈native Chead(82.87)，换分类器在 T0 不缩差距，冻结特征还倒退 3.8；T4 oracle NCM 仅 +1.19（量化了 A2 残余可分类器级恢复的接口损伤）。**纯统计原型分类接口判负**：差距在表征层（C²P 的对比损失 + per-class 原型训练在同数据预算下 T0 即 +4）→ A5 的共享语义层必须**可训练**（P2 走参数聚合 + 对比/原型目标路线）。z_op > z_ref 一贯（+3.4/+4.6）确认 prompt 适配特征的判别价值。

**P2（按 P1 结果定）**：小型共享语义层，普通类掩码聚合 vs 类感知聚合 → 兼容性约束（surrogate 约束残差融合）→ FedTA/A2/Shared/A5 四臂析因（交互项 I=Y₁₁−Y₁₀−Y₀₁+Y₀₀，BWT 与 forgetting 分开报）。T0-gap 判据同样适用。LCDC/CPA/FedAvg 均不预设为必需组件。

**P2-minimal（post-feature SSRA）Gate 判负（2026-10-28 终审，B2 结构搜索）**：固定 A2 T0 特征上的零初始化残差 adapter（768→bottleneck→768 + 线性头），12 配置（bottleneck {64,128,256} × scale {fixed, learnable α} × dropout {0, 0.2}）按预注册 val-only 协议选结构、winner 冻结后 test 只评一次：winner（b64/fixed/d0）−P0 = **+0.72pp**（逐 seed +0.96/+0.76/+0.43），未过 ≥1pp 续作门槛 → **RED，停止一切联邦工程**（Adapter FedAvg / task snapshot / checkpoint-resume / prototype loss 接入 / SSRA audit 全部不做）。结构趋势 b64>b128>b256、fixed≥learnable、d0>d0.2：容量增加不产生可利用的 task-in 判别余量。与 H1（head 重训 +0.35pp）、H2（anchor 通道 +0.15pp）闭环：T0 剩余问题不是"固定特征后堆更强 classifier/MLP"能解决的。**注意 scope：判负的仅是 A5-P2-v1（fixed feature + small residual adapter）这一形态；prompt-aware 语义学习、表征级对比训练、共享原型反向影响特征形成、feature distillation、prompt 条件化共享模块、backbone/prompt 联合低秩 adaptation 均未被本探针测试、均未被否**。新 A5 待 21.2 全量重评 + Track A B/C/D 归因完成后重新设计，重心从 post-feature 变换移向 h 的形成过程（x→ViT→Global Prompt routing→h）。措辞规范（论文用）：不写"B2 推翻了 v2 的 +1.28pp"；写"早期单配置探针提示 +1.28pp，但在更严格的预注册 val-only 结构选择协议下，入选 SSRA 配置在冻结 test 上仅 +0.72pp，未过 ≥1pp 续作门槛"。

**A2 槽位容量挤压风险**：A5 落地前用 verify_slot_binding.py 量化受保护槽比例/跨任务槽争抢。

## 7. 外部对比实验格局（主表 ~14 行）

FL 层：FedAvg/FedProx/FedNova（✅ 已实现 `protocol_adapters/run_fl_baseline.py`，izfree 跑）→ FL+CL 层：FedLwF/FedViT/FedL2P/FedDualP（队列，L2P 是 FedTA prompt pool 血统源头）→ FCL 层：GLFC/TARGET/MFCL/FedMGP（FedMGP ✅ adapter 就绪；CIFAR 在 izfree 跑中）→ C²P(INR) ✅ → ours（FedTA/A2/A5）。

双机分工 v2：izfree（4090 24GB）跑对比实验（C²P 除外，OOM）；AutoDL（4090 48GB）跑 FedSMTA 主线。

## 8. 未决事项

1. P1 NCM probe 正式运行 + 判读（INR seed42 A2 run）
2. TKR Fig 4(b) 高清读数复核（唯一残余）+ issue 回复作者（附三种子平均）
3. ImageNet-R 方向 (c) 诊断（2×2 协议不依赖 public classes）
4. 第二方法（GLFC）多方法验证
5. seed 43/44 manifest export + 三种子对比实验
6. A5 四臂消融（P1/P2 结果定形态后）

## 9. 硬约束与工程约定（长期有效）

- 不修改：FedTA 训练 loss / optimizer / BGPS / SIKF / FedAvg head / 联邦划分 / Tail Anchor 结构 / prompt 逻辑；`--method fedta` read-only
- 修复方案只动接口层；训练期干预（A2/A5）在新 method flag 下
- 新增 diagnostic 必须 observer-only；消耗 RNG 的代码 snapshot/restore
- 语义/特征诊断只用 TRAIN split；test data 永远禁止
- 数据集仅 cifar100 / ImageNet-R；本地导入；ImageNet-R `--private_class_num 40`（勿用 15，2026-10-14 误跑判废）
- 跨类聚合必须样本加权；对账验证协议（固定种子 `9700+100*cid+old_task` 坐标系）
- checkpoint 兼容 `torch_load_compat()`；manifest 不一致拒绝 resume
- 每新增测试命令同步更新 RUNCOMMANDS.md；评估协议 `global_epoch=5`、`threshold=0.25`
- 外部基线 = 官方仓库零编辑 + manifest 冻结划分 + method-native 行为；FL 三基线为构造基线（同 FedTA 论文 Table 1 做法），标注 "constructed following FedTA paper Table 1 setting"

## 10. 命令索引

见 [RUNCOMMANDS.md](file:///root/autodl-tmp/users/ych/FedSMTA/RUNCOMMANDS.md)。关键入口：§9 诊断（2×2 / 4combo / D0 / D1 v2 / 落点 / TKR / NCM probe）、§16 A2/A1、§17–19 外部基线（FedMGP / C²P / FL 三基线）。
