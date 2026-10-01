# FedSMTA 研发路线与优化方案总结

> 本文档整理自与 GPT 的完整方案讨论（2026-09-30），涵盖：项目核心思想、数学体系、七阶段开发路线（含 Gate/止损条件）、各阶段具体方案（公式级）、以及 Phase 0/1 代码审查中发现的问题与修复记录。
>
> **v2 修订（2026-09-30，按 GPT 复核意见）**：① Phase 4 效应更名 $\Delta_A \to \Delta_{KA}$（Key–Anchor compatibility effect），组件组合改用 $\operatorname{Eval}(H_\tau;\ P_t,\ Q_t,\ A_t)$ 记号；② Oracle-K2 2A 拆为 Positive-only / Full 两步，消除负样本数量混杂；③ client entropy 增加 Dirichlet 失衡校正（JS 散度 / 条件互信息）；④ Oracle Recovery Ratio 增加方向归一与有效域判定；⑤ $D_{\rm cover}$ 增加 normalized version；⑥ Phase 1 Gate 改为 prevalence + effect size + stability；⑦ Phase 5/6 机制全部降级为 candidate（由 Phase 1–4 结果决定）；⑧ 修正 Phase 0 状态描述（实现接近完成，但 Gate 未通过）。方案按置信度分三层管理，见 §3.1。
>
> 仓库：`Designer-Yan-0527/FedSMTA`（基线：`SkyOfBeginning/FedTA_CVPR2025`）

---

## 目录

1. [项目定位与核心思想](#1-项目定位与核心思想)
2. [核心数学体系](#2-核心数学体系)
3. [七阶段开发路线总览](#3-七阶段开发路线总览)
4. [Phase 0：Baseline Freeze & Verification](#4-phase-0baseline-freeze--verification)
5. [Phase 1：Multimodality Diagnostic](#5-phase-1multimodality-diagnostic)
6. [Phase 2：Oracle-K2](#6-phase-2oracle-k2)
7. [Phase 3：Semantic-K2](#7-phase-3semantic-k2)
8. [Phase 4：Head–Anchor Compatibility Diagnostic](#8-phase-4headanchor-compatibility-diagnostic)
9. [Phase 5：Fixed-K FedSMTA（完整方法）](#9-phase-5fixed-k-fedsmta完整方法)
10. [Phase 6：Dynamic-K + UOT](#10-phase-6dynamic-k--uot)
11. [Phase 7：Final Experiments & Ablation](#11-phase-7final-experiments--ablation)
12. [Phase 0/1 代码审查问题与修复记录](#12-phase-01-代码审查问题与修复记录)
13. [当前状态与下一步](#13-当前状态与下一步)

---

## 1. 项目定位与核心思想

### 1.1 论文暂定名

**Beyond Point Prototypes: Semantic Mode-Guided Tail Anchoring for Federated Continual Learning under Spatio-Temporal Heterogeneity**

### 1.2 核心思想

$$\boxed{\text{Global Semantic Alignment} + \text{Local Discriminative Adaptation}}$$

把 FedTA 的"每类单一 global point prototype"升级为"每类多模态 semantic mode bank"，但 **semantic statistics 与 trainable Tail Anchor 必须严格解耦**。

论文需要建立的逻辑链：

```
Spatio-temporal semantic heterogeneity
        ↓
multimodal class distribution
        ↓
single-prototype distortion（点原型失真）
        ↓
semantic mode bank（语义模式库）
        ↓
mode-guided local Tail Anchors（模式引导的局部尾锚）
        ↓
head-anchor compatibility preservation（头-锚兼容性保持）
```

### 1.3 两层架构

**Server 端：Global Semantic Mode Bank（不可训练的统计量）**

$$\mathcal{S}_c = \left\{ s_{c,k} \right\}_{k=1}^{K_c}, \quad s_{c,k} = \left( u_{c,k},\ \mu_{c,k},\ \sigma^2_{c,k},\ \pi_{c,k},\ n_{c,k},\ \text{active}_{c,k} \right)$$

| 符号 | 含义 |
|---|---|
| $u_{c,k}$ | persistent mode UID（永不复用） |
| $\mu_{c,k}$ | semantic mean |
| $\sigma^2_{c,k}$ | isotropic variance |
| $\pi_{c,k}$ | mode mass（样本占比） |
| $n_{c,k}$ | 有效样本数 |
| active | mode 是否活跃 |

全部是 **statistics, not trainable parameters**，必须满足 $\nabla \mu_{c,k} = 0$。

**Client 端：Local Discriminative Key–Anchor Bank（可训练）**

$$\mathcal{A}_{i,c} = \left\{ (q_{i,c,k},\ a_{i,c,k}) \right\}_{k=1}^{K_c}$$

- $q$：trainable discriminative key
- $a$：trainable Tail Anchor

**核心约束**：$\boxed{\mu_{c,k} \neq a_{i,c,k}}$，semantic center 与 discriminative anchor 绝不能混为一个参数，二者不共享梯度。

### 1.4 已确认的 FedTA 源码事实（方法设计的基础）

- 本地 Tail Anchor 实际是 `key + anchor_pool + Chead`，key/anchor 均为可训练 `nn.Parameter`
- 实际混合表示是 $h = [z;\ a] \in \mathbb{R}^{1536}$（768-D feature 与 768-D anchor 拼接），**并非相加**
- 服务器：`global_protos ≠ anchor_pool`；BGPS 产生 class-wise global prototype，主要参与第二阶段 InfoNCE；SIKF/`kd_fusion_prompt()` 融合 prompt；`vit.head` 通过 `fed_avg_head()` 聚合；**local Tail Anchor/key 不做 server FedAvg**；Chead 按 task 本地保存
- 旧 task 测试时使用：`old task head + current prompt + current key/anchor`，因此真实存在 **Head–Anchor Compatibility Drift**

### 1.5 CIFAR-100 数据协议的重要事实

官方划分：5 clients × 15 private classes = 75 private classes，剩余约 **25 public classes** 跨客户端 Dirichlet 分布；同一 client 内一个 class 基本只出现在一个 task。

因此不能把标准 FedTA CIFAR-100 描述为"同一客户端同一类别持续 temporal concept drift"，更准确的表述是：

> **asynchronous progressive revelation of multimodal semantic structure**

多模态诊断优先分析 25 个 public classes（最能体现 cross-client semantic heterogeneity）。

---

## 2. 核心数学体系

### 2.1 Semantic statistics 的数据来源（铁律）

所有 semantic statistics 必须来自 Tail Anchor **之前**的 ViT feature：

$$\boxed{z = f_\theta(x) \in \mathbb{R}^{768}}$$

- 不能统计 $[z;\ a]$，因为 $a$ 是 trainable parameter，会污染统计
- 统计时必须 `z = z.detach()`（stop-gradient）
- 维度恒定：无论 $K$ 取多少，Tail Anchor 输入保持 $h = [z;\ \bar{a}] \in \mathbb{R}^{1536}$，**不要拼成 2304/3072**

### 2.2 Local 统计量定义（isotropic Gaussian mode）

对客户端 $i$、类别 $c$、mode $k$ 的样本集 $\mathcal{Z}_{i,c,k} = \{z_1, \dots, z_n\}$：

$$\mu_{i,c,k} = \frac{1}{n} \sum_{j=1}^{n} z_j$$

$$\boxed{\sigma^2_{i,c,k} = \frac{1}{d\,n} \sum_{j=1}^{n} \|z_j - \mu_{i,c,k}\|_2^2}$$

**注意必须是 $\frac{1}{dn}$ 而不是 $\frac{1}{n}$**（isotropic 假设下每维平均）。

mode mass：

$$\pi_{c,k} = \frac{n_{c,k}}{\sum_l n_{c,l}}$$

### 2.3 多客户端统计量合并（Moment Merge）

假设多个 local mode 已匹配到同一 global mode，$\{(\mu_j, \sigma_j^2, n_j)\}_{j=1}^{M}$，$N = \sum_j n_j$：

$$\boxed{\mu = \frac{1}{N} \sum_j n_j \mu_j}$$

$$\boxed{\sigma^2 = \frac{1}{N} \sum_j n_j \left[ \sigma_j^2 + \frac{\|\mu_j - \mu\|_2^2}{d} \right]}$$

**关键**：$\frac{1}{d}$ 必须保留，否则 between-client mean shift 会被错误放大 $d$ 倍。该公式同时涵盖 within-client variance 和 between-client mean variance，需配 unit test（summary-statistics merge 与从 raw samples 直接计算数值一致）。

### 2.4 Mode 匹配距离（Gaussian Wasserstein-2）

两个 isotropic Gaussian $P = \mathcal{N}(\mu_1, \sigma_1^2 I)$、$Q = \mathcal{N}(\mu_2, \sigma_2^2 I)$：

$$\boxed{W_2^2(P, Q) = \|\mu_1 - \mu_2\|_2^2 + d\,(\sigma_1 - \sigma_2)^2}$$

**注意第二项是 $(\sigma_1 - \sigma_2)^2$（标准差之差的平方），不是 $(\sigma_1^2 - \sigma_2^2)^2$。**

---

## 3. 七阶段开发路线总览

| Phase | 名称 | 是否改算法 | 目的 |
|---|---|---|---|
| 0 | Baseline Freeze & Verification | 否 | 确保工程底座可信 |
| 1 | Multimodality Diagnostic | 否 | 证明问题真的存在 |
| 2 | Oracle-K2 | 是（仅诊断） | 判断多模态是否真的有价值 |
| 3 | Semantic-K2 | 是 | 无 oracle 地发现模式 |
| 4 | Compatibility Diagnostic | 基本否 | 验证 Head–Anchor Drift |
| 5 | Fixed-K FedSMTA | 是 | 建立完整核心方法 |
| 6 | Dynamic-K + UOT | 是 | 完成顶刊方法 |
| 7 | Final Experiments | 否/消融 | 论文实验 |

**三条止损线（Gate）：**

$$\boxed{\text{Phase 1 fails} \Rightarrow \text{不做 multimodal FedSMTA}}$$

$$\boxed{\text{Oracle-K2 fails} \Rightarrow \text{直接停止 K2 主线}}$$

$$\boxed{\text{Oracle succeeds, Semantic fails} \Rightarrow \text{只修 mode discovery/matching，不上 UOT}}$$

核心开发原则：

$$\text{先证明问题存在} \rightarrow \text{再证明理想解有效} \rightarrow \text{再做可实现解} \rightarrow \text{最后增加复杂机制}$$

绝对不要用 UOT、dynamic K、更多 loss 去掩盖 Semantic-K2 本身不工作的事实。

### 3.1 方案置信度分层（v2 新增）

本方案按置信度分三层管理，防止把"暂定设计"当成"最终方法定义"写进论文：

**第一层：已锁定，不再修改**

- 阶段顺序：Phase 0 → 1 → Oracle-K2 → Semantic-K2 → Compatibility → Fixed-K → Dynamic-K/UOT
- 核心约束：$\mu_{c,k} \neq a_{i,c,k}$（semantic center 与 discriminative anchor 解耦）
- 数据来源：$z$ = pre-anchor feature，$\sigma^2 = \frac{1}{dn}\sum\|z-\mu\|^2$
- 匹配距离：$W_2^2 = \|\mu_1-\mu_2\|^2 + d(\sigma_1-\sigma_2)^2$

**第二层：Phase 1/2 实验设计（本文档 v2 已修正）**

- Oracle 2A 负样本数量混杂 → 拆分 Positive-only / Full 两步（§6.1）
- client entropy 被 Dirichlet 失衡混淆 → 增加 JS 散度 / 条件互信息（§5.7）
- Phase 1 Gate 改为 prevalence + effect size + stability（§5.10）

**第三层：candidate mechanism（全部由 Phase 1–4 实验结果决定，现在不锁死）**

- SMCL 的 replace-vs-auxiliary 形式（§9.3）
- Compatibility regularizer $\mathcal{L}_{\rm HAC}$ / $\mathcal{L}_K$（§9.4）
- soft joint router（§9.1）
- UOT、birth/death（§10）

在 Phase 1–4 数据出来之前，第三层任何机制不得写进论文方法定义。

---

## 4. Phase 0：Baseline Freeze & Verification

### 4.1 目标

$$\boxed{\text{FedSMTA(method=fedta)} \simeq \text{Official FedTA}}$$

Phase 0 完成后**禁止再修改 baseline 算法文件的数学逻辑**，`--method fedta` 永远得到原 FedTA。

### 4.2 CIFAR-100 固定协议

```
client_num = 5          task_num = 5           private_class_num = 15
global_epoch = 5        local_epoch = 30       batch_size = 16
surrogate_num = 20      threshold = 0.25
```

（注意：CIFAR-100 官方默认 `surrogate_num=20`；ImageNet-R 官方硬编码 `process_testdata(5)`，即 surrogate_num=5。）

### 4.3 Checkpoint / Resume 机制

- Checkpoint 在**完整 global round 结束后**保存
- 保存内容：server state、global head/protos、prompt、client Tail Anchor state、task-specific heads、task/client 状态、**data split indices（内外两层）**、RNG states
- 恢复时必须从 `completed_round + 1` 继续，RNG 最后恢复
- **外层 federated partition 必须固化**：`outer_split_manifest = {client_id: {task_id: {"raw_indices": [...], "classes": [...]}}}`；resume 时重新生成并 exact 比较，不一致 `RuntimeError`
- 客户端内部 70/30 `random_split()` 的 indices 也必须保存（否则 resume 后旧 task test set 改变，遗忘结果不可信）

推荐目录结构：

```
runs/
  dataset/
    method/
      run_name/
        seed_x/
          command.txt    args.json
          logs/          checkpoints/    metrics/
```

### 4.4 Accuracy Matrix 与评价指标

定义 $A_{t,j}$ = 训练完 task $t$ 后在 task $j$ 上的 accuracy，形成下三角矩阵。

平均准确率：

$$ACC_t = \frac{1}{t+1} \sum_{j=0}^{t} A_{t,j}, \qquad ACC = \frac{1}{T} \sum_{j=0}^{T-1} A_{T-1,j}$$

Forgetting：

$$F_j = \max_{t \in [j, T-1]} A_{t,j} - A_{T-1,j}, \qquad F = \frac{1}{T-1} \sum_{j=0}^{T-2} F_j$$

Backward Transfer：

$$BWT = \frac{1}{T-1} \sum_{j=0}^{T-2} \left( A_{T-1,j} - A_{j,j} \right)$$

### 4.5 三个 Gate（Phase 0 封板判定）

Phase 0 总目标固定为：

$$\boxed{\text{FedSMTA(method=fedta)} \equiv \text{Official FedTA}}$$

同时证明 checkpoint/resume 与 observer 都不改变这条训练轨迹。

| Gate    | 要证明什么                            | 最终判定            |
| ------- | ----------------------------------- | ------------------- |
| Gate 0A | FedSMTA 的 `fedta` 路径 = 官方 FedTA | baseline parity     |
| Gate 0B | 连续训练 = checkpoint + resume       | resume equivalence  |
| Gate 0C | instrumentation OFF = ON 的训练轨迹  | observer invariance |

z_op batch sensitivity 属于 **Phase 1-precheck**，不放在 Phase 0 Gate（见 §5.9）。

**Gate 0A：Official parity**（`test_official_fedta_parity.py`，Phase 0 最重要的一关）：

Gate 0B 只能证明"自己的代码连续跑 = 自己的代码 resume 跑"，**不能证明等于官方**，所以必须单独做。固定 CIFAR-100 短程 parity 配置（task_num=2, global_epoch=2, local_epoch=2, batch_size=16, surrogate_num=20, threshold=0.25, seed=42, rounds=4），官方 FedTA vs `FedSMTA --method fedta --no_instrumentation` 逐项比较：

$$D_{\rm proto}^{(r)} = \max_c \|\mu^{official}_{c,r} - \mu^{ours}_{c,r}\|_\infty, \qquad D_{\rm head}^{(r)}, \qquad D_{\rm prompt}^{(r)}, \qquad D_q^{(r)}, \qquad D_a^{(r)}, \qquad D_{Chead}^{(r)}$$

以及 outer split / inner 70/30 split / current classes / fix_keys。官方仓库无 checkpoint 机制，测试通过生成的 driver 脚本在官方仓库内进程内复刻 `main()` 流程并导出状态。**理想 PASS 为 $D=0$**；只有 GPU 算子确实导致极小 nondeterminism 时才考虑浮点容差，不能一开始就用宽松 tolerance 掩盖逻辑差异。v1 限制（不作 gating）：accuracy 不直接比较（全部状态 $D=0$ 蕴含评估一致）；RNG 无法比较（官方无保存机制）。

**Gate 0B：Resume 等价性**（`test_resume_regression.py`）：

```
Run A: round 0 ──────────────→ N（连续）
Run B: round 0 ──→ k，checkpoint，resume，k+1 ──→ N
```

比较：accuracy matrix、global prototypes、prompt、global head、client heads、key、anchor、split indices、RNG states，验证 $A_N = B_N$（exact equality）。封板要求：两侧 checkpoint 缺失/空 `outer_split_manifest` 直接 FAIL（`require_outer_manifest` fail-fast，杜绝 `{} == {}` 假 PASS）；`load_checkpoint()` 对缺失/空/不匹配 manifest 一律 `RuntimeError`。

**Gate 0C：Observer 不变性**（`test_observer_invariance.py`）：

- `--no_instrumentation` OFF vs ON 两组运行（rounds 默认 = global_epoch + 1，跨 task 边界，比较 next-task split）
- 验证核心状态完全一致：server model、global head、global protos、prompt、client heads、key、anchor_pool、RNG、inner split、next-task split
- 封板断言：`ckpt_off["args"]["no_instrumentation"] is True` 且 `ckpt_on[...]["no_instrumentation"] is False`（否则比较是空洞的）
- 验证 instrumentation artifact 确实不同：OFF run 不产生任何 observer 文件（train_log.csv / round_metrics.jsonl / accuracy_matrix.csv / margins.csv），ON run 必须产生（证明 `--no_instrumentation` 真正等于 "FedTA + checkpoint only"）

三个 Gate 全部 PASS 后打 Git tag：`phase0-fedta-baseline`。任一 Gate 不通过则**禁止进入 Phase 1**。

---

## 5. Phase 1：Multimodality Diagnostic

论文第一个真正的科学实验。**绝对不修改训练**：只允许 `z.detach()` 后保存 feature；禁止 `loss += ...`；禁止新增 trainable parameter。

### 5.1 Feature Collection

- 只用 **TRAIN split** 发现 mode，测试集永远不能用于 mode discovery
- 优先分析 25 个 public classes
- 同时保存两种 feature（对论文非常有价值的 control）：
  - $z_{\rm ref}$：frozen pretrained representation（`server.origin_model(inp)`）
  - $z_{\rm op}$：prompt-enhanced pre-anchor representation（`client.vit(...)`，未经 Tail Anchor）
- 每条记录：feature、label、client_id、task_id、round_id、sample_index、split
- Feature 不塞入 checkpoint，输出到 `runs/.../diagnostics/`

### 5.2 K=1 vs K=2 的 SSE

$$\mu_c = \frac{1}{N_c} \sum_n z_n, \qquad SSE_1 = \sum_n \|z_n - \mu_c\|^2$$

$$SSE_2 = \sum_{k=1}^{2} \sum_{z \in C_k} \|z - \mu_{c,k}\|^2$$

$$\Delta_{\rm SSE} = \frac{SSE_1 - SSE_2}{SSE_1 + \epsilon}$$

**严谨性警告**：K=2 的 SSE 必然不高于 K=1，**不能只凭 SSE 宣称 multimodality**。

### 5.3 Held-out NLL（真正的 multimodality 判据）

TRAIN feature 再分 80% diagnostic-fit / 20% diagnostic-validation，只用 fit 集拟合 isotropic GMM：

$$p_K(z) = \sum_{k=1}^{K} \pi_k\, \mathcal{N}(z;\ \mu_k,\ \sigma_k^2 I)$$

$$NLL_K = -\frac{1}{N_{\rm val}} \sum_{z \in V} \log p_K(z), \qquad \Delta NLL = NLL_1 - NLL_2$$

$\Delta NLL > 0$ 说明 K=2 对未参与拟合的 feature 有更强泛化解释能力。

### 5.4 BIC

$$BIC_K = -2 \log L_K + p_K \log N, \qquad p_K = Kd + K + (K-1)$$

参数计数：$Kd$（means）+ $K$（variances）+ $K-1$（mixture weights）。$\Delta BIC = BIC_1 - BIC_2 > 0$ 偏向 K=2。

### 5.5 Mode Separation

$$\boxed{\rho_c = \frac{\|\mu_{c,1} - \mu_{c,2}\|}{\sqrt{d\,(\sigma_{c,1}^2 + \sigma_{c,2}^2)/2}}}$$

- $\rho \ll 1$：K=2 很可能只是把一个 blob 强行切成两半
- $\rho > 1$：才开始具有比较明确的模式分离
- 1 不是理论硬阈值，只是诊断参考

### 5.6 Single-Prototype Distortion（Coverage Distortion）

FedTA 单 prototype 近似为 $\mu_c^{\rm single} = \sum_k \pi_{c,k} \mu_{c,k}$，可能落在两个真实 mode 中间：

$$\boxed{D_{\rm cover} = \sum_k \pi_{c,k} \left\| \mu_{c,k} - \mu_c^{\rm single} \right\|^2}$$

直接量化 point prototype 离真实 semantic modes 有多远。

raw 值跨 class 不易直接比较，需额外输出 normalized version：

$$\boxed{D_{\rm cover}^{\rm rel} = \frac{D_{\rm cover}}{\frac{1}{N} \sum_n \left\| z_n - \mu_c^{\rm single} \right\|^2 + \epsilon}}$$

含义直观：single prototype distortion 占该类别总 feature dispersion 的比例。

### 5.7 Client-mode Heterogeneity（含 Dirichlet 失衡校正）

mode $k$ 内 client $i$ 的比例 $r_{i,k} = n_{i,k} / \sum_j n_{j,k}$，entropy：

$$H_k = -\sum_i r_{i,k} \log r_{i,k}$$

低 entropy 说明 mode 主要由少数客户端贡献——**但这不能直接叫 spatial semantic heterogeneity 的证据**：public class 本来就是 Dirichlet non-IID，若 class $c$ 的样本 80% 本就来自 client A，则某个 mode 80% 来自 A 并不能证明 mode 与 client 有关联（entropy 被 client imbalance 混淆）。

需在控制类别本身 client 分布后使用更强指标。记 class $c$ 的原始 client 分布：

$$p(i \mid c) = \frac{n_{i,c}}{\sum_j n_{j,c}}$$

与 mode 内分布 $p(i \mid c, k)$，计算 JS 散度：

$$D_{\rm JS}\left( p(i \mid c, k)\ \Vert\ p(i \mid c) \right)$$

或直接用**条件互信息**（正式论文指标）：

$$\boxed{I(I;\ K \mid C = c)}$$

即 client identity 与 mode assignment 在给定 class 后的 mutual information。它回答：**在控制该类别本身 client imbalance 后，mode membership 是否仍然依赖 client？** 比单纯 client entropy 强得多。

输出约定：`client_entropy`（保留，描述性）、`normalized_client_entropy`（增加）、`client_mode_mi` / `js_divergence`（正式论文指标）。

### 5.8 输出与可视化

- `metrics/multimodality_summary.csv`：class_id、sample_num、sse1/sse2、delta_sse、nll1/nll2、delta_nll、bic1/bic2、delta_bic、silhouette、mode_distance、rho、mode1_mass、mode2_mass、coverage_distortion、coverage_distortion_rel、client_entropy_mode1/2、normalized_client_entropy、client_mode_mi、js_divergence
- `metrics/multimodality_aggregate.json`：mean / median / std / p25 / p75 / bootstrap 95% CI
- PCA/UMAP **只用于画图**，不用于计算核心指标；图上：颜色 = client，marker = task，★ = FedTA single prototype，× = K2 mode centers
- 最理想的发现：client A → mode 1、client B → mode 2、single prototype 落在两个 cluster 中间；但如果实际不是，必须如实接受

### 5.9 z_op 的 batch sensitivity 问题（方法学关键）

FedTA 默认 `batchwise_prompt=True`：`Global_Prompt.forward()` 会统计整个 batch 的 prompt 频率并给全 batch 用同一 major prompt，因此：

$$z_{\rm op}(x) = f(x;\ \mathcal{B}) \neq f(x)$$

同一样本与不同样本组 batch 时 prompt 可能不同，得到的 $z_{\rm op}$ 也不同。若不处理，最终看到的 K=2 可能部分来自 batch-wise prompt routing 差异而非真正 semantic multimodality（审稿人必质疑）。

**方案**：新增 `diagnostics/check_feature_batch_sensitivity.py`，同一 checkpoint、同一批固定顺序 TRAIN samples，分别用 batch size = 1, 8, 16, 32, 64 提取 $z_{\rm op}$，对每个样本计算：

$$\Delta_B(x) = \frac{\left\| z_{\rm op}^{(B)}(x) - z_{\rm op}^{(1)}(x) \right\|_2}{\left\| z_{\rm op}^{(1)}(x) \right\|_2 + \epsilon}$$

汇总 mean/median/std/p95/max 及 cosine similarity，另做同 batch_size=16 不同 deterministic 组批的对照。若 $\bar{\Delta}_B \approx 0$ 可放心用 $z_{\rm op}$；否则需重新决定用 $z_{\rm ref}$ 构造 Semantic Bank、或构造 sample-wise prompt feature、或严格定义 prompt context。**先只做 diagnostic，不擅自改用 z_ref、不关闭 batchwise_prompt。**

### 5.10 Phase 1 Gate（prevalence + effect size + stability）

Gate 不采用"大多数 public classes 必须 multimodal"的粗粒度多数判据——真实情况可能是 25 个 public classes 中仅 8–10 个有很强 multi-mode、其余 unimodal；此时 FedSMTA 仍可能成立（且更符合 Dynamic-K 的 $K_c = 1\ \text{or}\ 2$ 设定）。

定义 **robust multimodal class**：同时满足

$$\Delta NLL > 0, \qquad \Delta BIC > 0, \qquad \rho > \rho_0$$

且在 bootstrap / 多 seeds 下稳定。Gate 从三个维度判：

1. **prevalence**：robust multimodal classes 的比例（而非简单 >50%）
2. **effect size**：这些类上 $\Delta NLL$、$D_{\rm cover}^{\rm rel}$ 的幅度
3. **stability**：bootstrap 95% CI 是否稳定偏离 0

若 robust classes 比例极低、effect size 微弱、CI 跨 0 不稳定：

$$\boxed{\text{停止 multimodal 主线，不要硬做 FedSMTA}}$$

---

## 6. Phase 2：Oracle-K2

核心问题不是"能不能实现 K2"，而是：

$$\boxed{\text{如果 semantic mode 是正确的，是否真的能提高 FCL？}}$$

- Oracle 数据可用所有 **TRAIN** features（允许跨 client、跨未来 task，因为这是 upper bound），**test data 永远禁止**
- public class：$K_c = 2$；private class：$K_c = 1$
- Oracle semantic center 必须 detached，不得设为 `nn.Parameter`

### 6.1 Phase 2A：只改 semantic prototype（两步走，消除负样本数量混杂）

Tail Anchor 完全保持 FedTA。样本 $z$ 选择最近的 mode：

$$k^* = \arg\min_k \|z - \mu_{c,k}\|^2$$

**混杂变量警告**：原 FedTA 中每个负类 $c' \to 1$ 个 prototype；若负类也直接展开为 $K=2$ 个 negative modes，denominator 中负样本数量直接翻倍——2A 的收益/损失将无法区分是 semantic mode improvement 还是 negatives 数量/强度的改变。因此 2A 必须拆成两步：

**2A-1（首选，最干净）：Oracle-K2 Positive-only**

只把 true-class positive $\mu_c \to \mu_{c,k^*}$，负类保持 FedTA 原 single global prototype：

$$\boxed{\mathcal{L}_{\rm mode}^{+} = -\log \frac{e^{s(z,\ \mu_{c,k^*})/\tau}}{e^{s(z,\ \mu_{c,k^*})/\tau} + \sum_{c' \neq c} e^{s(z,\ \mu_{c'})/\tau}}}$$

回答最纯的问题：**单点 prototype 对正确类别的 semantic guidance 是否真的存在 distortion？**

**2A-2（仅当 2A-1 有效才做）：Oracle-K2 Full Mode Contrastive**

负类也 mode-aware，但必须 class-normalized（避免 K=2 的类天然贡献两倍 negatives）：

$$S_{c'}(z) = \sum_l \pi_{c',l}\, e^{s(z,\ \mu_{c',l})/\tau}$$

$$\mathcal{L}_{\rm mode}^{\rm full} = -\log \frac{e^{s(z,\ \mu_{c,k^*})/\tau}}{e^{s(z,\ \mu_{c,k^*})/\tau} + \sum_{c' \neq c} S_{c'}(z)}$$

其中 $s(x,y) = \frac{x^\top y}{\|x\|\|y\|}$。

### 6.2 Phase 2B：Mode-specific Tail Anchor（仅 2A 有效才做）

将 $(q_{i,c},\ a_{i,c})$ 扩展为 $(q_{i,c,1},\ a_{i,c,1})$、$(q_{i,c,2},\ a_{i,c,2})$，按 oracle mode 选择：

$$h = [\,z;\ a_{i,c,k^*}\,], \qquad \dim(h) = 768 + 768 = 1536$$

**禁止** $[z;\ a_1;\ a_2]$。

### 6.3 Parameter Control（必须做）

实现 **K2-extra-param control**：同样两个 anchor slots，但随机分配/不使用 semantic routing。用于验证：

$$\text{gain} \neq \text{just more parameters}$$

### 6.4 Phase 2 Gate

比较组：`FedTA` vs `Oracle-K2 Positive-only (2A-1)` vs `Oracle-K2 Full (2A-2)` vs `K2 ExtraParam Control`（做 2B 再加 `Oracle-K2 ModeAnchor`）。

不只看 ACC，重点看：$ACC$、$F$、$BWT$、public-class accuracy、classification margin：

$$m(x) = f_y(x) - \max_{j \neq y} f_j(x)$$

若 Oracle-K2（含 2A-1）不能稳定提高 retention / margin / public accuracy → **停止**。

---

## 7. Phase 3：Semantic-K2

目标：$$\text{不共享 raw feature，也能恢复 Oracle 的收益}$$

### 7.1 客户端本地统计与上传

客户端只用 LOCAL TRAIN feature 聚类，上传 $(\mu_{i,c,k},\ \sigma^2_{i,c,k},\ n_{i,c,k})$。**禁止上传 raw image / raw feature / anchor / key / head**。

### 7.2 Fixed K=2 Matching（先不用 UOT）

cost：$C_{rk} = \|\mu_r - \mu_k\|^2 + d(\sigma_r - \sigma_k)^2$，采用 **Hungarian** 匹配。

Global 初始化用 weighted KMeans（输入点：local mode means；权重：$n_r$；K=2），必须 deterministic。

### 7.3 Moment Merge

按 §2.3 公式合并（配 unit test 验证与 raw samples 直接计算一致）。

### 7.4 Gaussian Router

$$g_{c,k}(z) = -\frac{\|z - \mu_{c,k}\|^2}{2(\sigma^2_{c,k} + \epsilon)} - \frac{d}{2} \log(\sigma^2_{c,k} + \epsilon) + \log(\pi_{c,k} + \epsilon)$$

$$\boxed{r_{c,k}(z) = \mathrm{softmax}_k\left(g_{c,k}(z)\right)}$$

支持 hard（$k^* = \arg\max_k r_{c,k}$）/ soft 两种模式；输出 occupancy、assignment entropy/confidence、client composition。

### 7.5 Oracle Recovery Ratio（方向归一 + 有效域判定）

设 baseline metric $M_B$、Oracle $M_O$、Semantic $M_S$。注意：ACC / BWT / margin 是 higher-is-better，Forgetting 是 lower-is-better，直接套同一公式方向会反；且 $M_O \approx M_B$ 时 ratio 失去意义甚至爆炸。先做方向归一：

$$U(M) = \begin{cases} M, & \text{higher is better} \\ -M, & \text{lower is better} \end{cases}$$

$$\boxed{R_O = \frac{U(M_S) - U(M_B)}{U(M_O) - U(M_B)}}$$

**有效域判定**：仅当 $U(M_O) - U(M_B) > \delta$ 时报告 $R_O$，否则输出：

```text
N/A: oracle does not provide a meaningful positive upper-bound gain.
```

回答：非 Oracle 方法恢复了多少 ideal semantic information 的收益？

### 7.6 Phase 3 Gate

若 $Oracle \gg Semantic$：核心问题不是多模态没用，而是 distributed mode discovery 不行 → 优化 matching/statistics，**不要直接进入 UOT**。本阶段禁止直接加 regularization。

---

## 8. Phase 4：Head–Anchor Compatibility Diagnostic

验证第二个核心问题。FedTA 旧任务 $\tau$ 在时间 $t$ 的实际评估是**模型组件组合**（不是向量相加；最终表示仍为 $h = [z;\ a]$），记作：

$$\boxed{\operatorname{Eval}(H_\tau;\ P_t,\ Q_t,\ A_t)}$$

（旧 task head $H_\tau$、当前 prompt $P_t$、当前 key $Q_t$、当前 anchor $A_t$）

### 8.1 Task Snapshot 与 2×2 反事实评估

task $\tau$ 结束时保存 $H_\tau,\ P_\tau,\ Q_\tau,\ A_\tau$。未来 task $t$ 对旧 task 评估四组：

| 组合 | prompt | anchor/key |
|---|---|---|
| $E_{11}$ | current | current |
| $E_{10}$ | current | old |
| $E_{01}$ | old | current |
| $E_{00}$ | old | old |

（旧 head 固定不动。当前 `compatibility_drift.py` 改的是 key 与 anchor 整体，测得的是 **Key–Anchor joint drift**；若未来要拆分纯 Anchor effect，需增加 key-only / anchor-only 的 2×2 factorial——old/new key × old/new anchor 四组。）

### 8.2 效应分解

$$\boxed{\Delta_{KA} = \tfrac{1}{2}\left[(E_{10} - E_{11}) + (E_{00} - E_{01})\right]} \quad \text{(Key–Anchor compatibility effect)}$$

**命名约束**：counterfactual 改的是 key 与 anchor 整体，因此该效应只能叫 **Key–Anchor compatibility effect（$\Delta_{KA}$）**，不能写成纯 Anchor effect——除非按 §8.1 增加 key-only / anchor-only factorial 后才可拆分命名。

$$\Delta_P = \tfrac{1}{2}\left[(E_{01} - E_{11}) + (E_{00} - E_{10})\right] \quad \text{(prompt effect)}$$

$$I = E_{00} - E_{01} - E_{10} + E_{11} \quad \text{(interaction)}$$

### 8.3 Drift 与 Routing Stability

$$D_a = \frac{\|a_t - a_\tau\|}{\|a_\tau\| + \epsilon}, \qquad D_q = \frac{\|q_t - q_\tau\|}{\|q_\tau\| + \epsilon}$$

$$R_{\rm flip} = \frac{1}{N} \sum_x \mathbf{1}\left[k_t(x) \neq k_\tau(x)\right]$$

若 $\Delta_{KA} > 0$（即 $E_{10} > E_{11}$）且 $D_a \uparrow$、$R_{\rm flip} \uparrow$，则 Head–Anchor Compatibility Drift 有实证支持。

**所有 counterfactual evaluation 必须在执行前保存 model/prompt/anchor-key/RNG state，并在 finally 中完全恢复**（observer-only）。输出 `metrics/compatibility_matrix.csv`、`metrics/compatibility_summary.json`。本阶段不得实现 compatibility regularizer，先获得实验结论。

---

## 9. Phase 5：Fixed-K FedSMTA（完整方法）

前置条件：Phase 1–4 全部成立。

### 9.1 Joint Routing（semantic + key 双分数，candidate）

$$s^{\rm sem}_{c,k} = \log(r_{c,k}(z) + \epsilon), \qquad s^{\rm key}_{i,c,k} = \cos(z,\ q_{i,c,k})$$

$$s_{i,c,k} = \beta_s\, s^{\rm sem}_{c,k} + \beta_k\, \frac{s^{\rm key}_{i,c,k}}{\tau_q}$$

（routing 分数权重用 $\beta_s,\ \beta_k$，与 loss 权重 $\lambda$ 系列区分，避免符号冲突。soft joint router 本身为 candidate mechanism。）

$$\boxed{\alpha_{i,c,k} = \mathrm{softmax}_k\left(\frac{s_{i,c,k}}{\tau_r}\right)}$$

### 9.2 Soft Tail Anchor

$$\bar{a}_{i,c} = \sum_k \alpha_{i,c,k}\, a_{i,c,k}, \qquad h = [\,z;\ \bar{a}\,], \quad \dim = 1536$$

支持 hard/soft routing ablation。

### 9.3 Mode-aware Contrastive Loss（SMCL，candidate）

$$\mathcal{L}_{\rm SMCL} = -\log \frac{\sum_k r_{c,k}(z)\, e^{\mathrm{sim}(z,\ \mu_{c,k})/\tau}}{\sum_k r_{c,k}(z)\, e^{\mathrm{sim}(z,\ \mu_{c,k})/\tau} + \sum_{c' \neq c} \sum_l \pi_{c',l}\, e^{\mathrm{sim}(z,\ \mu_{c',l})/\tau}}$$

所有 $\mu$ 带 stop-gradient；负类部分沿用 §6.1 的 class-normalized 形式。

**暂定说明（不锁死）**：SMCL 与原 FedTA single-prototype InfoNCE 的关系是 **replace 还是 auxiliary（相加）**，由 Phase 2A 实验结果决定。若 $\mathcal{L}_{\rm FedTA}$ 已含原 InfoNCE，直接相加会变成 "single-prototype loss + multi-mode loss" 并存，未必符合 Beyond Point Prototypes 的设计目标。Phase 2A 结论出来前，本条为 candidate design。

### 9.4 Compatibility Regularizer（candidate，由 Phase 4 决定是否启用）

> **candidate design**：Phase 4 尚未回答主要问题是否是 anchor drift、key drift 是否关键、prompt interaction 是否更大、key–anchor interaction 是否主要来源。在 Phase 4 结论出来之前，$\mathcal{L}_{\rm HAC}$ / $\mathcal{L}_K$ 只是候选设计——activated only if Phase 4 supports the corresponding mechanism；不得在未诊断前预设答案是 anchor/key regularization。

旧 head $W_\tau = [W^z_\tau,\ W^a_\tau]$，anchor 依赖度 $\Omega_\tau = \|W^a_\tau\|_F^2$，保存 reference anchor/key：

$$\mathcal{L}_{\rm HAC} = \sum_{\text{old modes}} \Omega_{c,k} \left\| a_{i,c,k} - \mathrm{stopgrad}(a^{\rm ref}_{i,c,k}) \right\|^2$$

$$\mathcal{L}_{K} = \sum \omega_{c,k} \left\| q_{i,c,k} - \mathrm{stopgrad}(q^{\rm ref}_{i,c,k}) \right\|^2$$

### 9.5 总目标（candidate 形式）

$$\mathcal{L} = \mathcal{L}_{\rm FedTA} + \lambda_m \mathcal{L}_{\rm SMCL} + \lambda_a \mathcal{L}_{\rm HAC} + \lambda_k \mathcal{L}_{K}$$

（loss 权重统一用 $\lambda_m,\ \lambda_a,\ \lambda_k$；SMCL 的 replace-vs-auxiliary 形式见 §9.3 暂定说明；$\mathcal{L}_{\rm HAC}$ / $\mathcal{L}_K$ 是否启用见 §9.4。）

**所有新增 loss weight 默认允许设为 0；当全部为 0 且 K=1 时，必须精确退化为 FedTA**（需 `test_k1_equivalence.py` 验证）。

---

## 10. Phase 6：Dynamic-K + UOT（candidate mechanism）

> 本阶段全部机制（UOT、birth/death、dynamic K）为 **candidate mechanism**，是否及以何种形式启用，由 Phase 1–4 实验结论决定；Phase 3/4 Gate 未通过前不得进入本阶段。

### 10.1 Persistent Mode UID

每个 mode 记录：mode_uid、class、birth_round、last_seen、active、mass、μ/variance trajectory（输出 `metrics/mode_history.jsonl`）。

$$\boxed{\text{UID 永不复用，禁止用 list index 作为永久身份}}$$

推荐 `ModuleDict` keyed by UID 或固定 $K_{\max}$ + active mask，必须保证 old-head/anchor mapping 不变。

### 10.2 Unbalanced Optimal Transport

local mass $p_r = n_r / \sum_s n_s$，global mass $q_k = \pi_k$，cost $C_{rk} = \|\mu_r - \mu_k\|^2 + d(\sigma_r - \sigma_k)^2$：

$$\min_{\Gamma \geq 0}\ \langle \Gamma, C \rangle + \epsilon\, \mathrm{KL}(\Gamma \| pq^\top) + \tau_p\, \mathrm{KL}(\Gamma \mathbf{1} \| p) + \tau_q\, \mathrm{KL}(\Gamma^\top \mathbf{1} \| q)$$

保存 Γ、transport cost、matched/unmatched mass；transported effective count $m_{rk} = N_{\rm class} \cdot \Gamma_{rk}$ 用于 fractional moment merge。

### 10.3 Birth / Death

**Birth**：local mode $r$ 的匹配比例 $\eta_r = \frac{\sum_k \Gamma_{rk}}{p_r + \epsilon}$。若 $\eta_r < \eta_{\rm match}$ 且 $\min_k C_{rk} > \delta_{\rm birth}$ 连续 $L_b$ 轮且 active modes < $K_{\max}$ → 创建新 mode（新 UID = global monotonic counter）。

**Death**：若 $\pi_{c,k} < \pi_{\min}$ 且 incoming transport mass $\sum_r \Gamma_{rk} < \eta_{\rm death}$ 连续 $L_d$ 轮 → $active = False$。

$$\boxed{\text{绝不能物理删除 slot / 压缩 anchor / 重排 mode index / 复用 UID}}$$

否则旧 head-anchor 映射会被破坏。

---

## 11. Phase 7：Final Experiments & Ablation

### 11.1 主表方法列表

```
FedTA / Oracle-K2 Positive-only (2A-1) / Oracle-K2 Full (2A-2) /
Oracle-K2 ModeAnchor (2B) / Semantic-K2 / Semantic-K2 + ModeAnchor /
Semantic-K2 + Compatibility / Dynamic-K without UOT / Full FedSMTA
```

### 11.2 指标

ACC、task-wise final accuracy、forgetting、BWT、worst-task accuracy、public/private accuracy、classification margin、temporal/spatial retention、active modes 数、birth/death 计数、routing entropy、communication bytes、trainable parameters、server memory。

### 11.3 Ablation 清单

K1 vs K2 / Oracle vs Semantic / random K2 / extra-param K2 / mean only vs mean+variance / hard vs soft router / Hungarian vs UOT / compatibility OFF-ON / dynamic K OFF-ON / birth-death OFF-ON / ModeInfoNCE OFF-ON / public-only vs all-class modes。

### 11.4 统计规范

开发期至少 3 seeds，最终 5 seeds，报告 mean ± std（每 seed raw result 保留），最好加 paired bootstrap 95% CI。

### 11.5 必备测试

`test_k1_equivalence.py`、`test_moment_merge.py`、`test_w2.py`、`test_mode_uid_stability.py`、`test_resume_equivalence.py`、`test_observer_rng_invariance.py`。

---

## 12. Phase 0/1 代码审查问题与修复记录

GPT 对仓库多轮源码级审查（官方 FedTA vs FedSMTA 逐文件对照）的完整结论与处置：

### 12.1 审查发现的基准等价性问题

| # | 问题 | 严重度 | 状态 | 说明 |
|---|---|---|---|---|
| 1 | 本地训练 `batch_size=16` 硬编码改为 `args.batch_size` | 🟡 | 已确认等价 | 实际固定 bs=16 时行为与官方一致；形式变化但协议未变 |
| 2 | ImageNet-R `process_testdata(5)` 改为 `args.surrogate_num` | 🔴→🟢 | 已确认等价 | ImageNet-R 必须设 `--surrogate_num 5`；CIFAR-100 官方默认 20，不手动覆盖即可 |
| 3 | BGPS `threshold=0.25`、SIKF `i % global_epoch != 4` 改为参数化 `threshold`、`!= global_epoch-1` | 🟡 | 已确认等价 | 默认 threshold=0.25、global_epoch=5 时等价；改参数后语义不同，Phase 0 不建议顺手修官方写法 |
| 4 | ImageNet-R 官方协议为 client=10/task=10/private=40/size=20/length=15/local_epoch=20，当前沿用 CIFAR config（5/5/15/100/10/30） | 🔴 | 未修（仅 ImageNet-R） | 当前 ImageNet-R 运行**不是官方协议**；Phase 1 主线是 CIFAR-100，暂不阻塞 |
| 5 | ImageNet-R loader 增加 `sorted(os.listdir())` | 🟡 | 保留 | 工程上更合理（避免文件系统顺序导致 label 错位），但属行为变化，注意 `class_public.remove(72)` 的 index mapping 影响 |

### 12.2 已修复的工程问题

| # | 问题 | 修复方案 |
|---|---|---|
| 1 | **Full seen-task evaluation 污染 RNG**（🔴 最隐蔽）：新增评估的 DataLoader `shuffle=True` 消耗 Torch RNG，改变下一 task 的 `random_split()` 与训练轨迹 | `run_full_evaluation()` / `_log_local_phase()` 前后 snapshot/restore RNG（`try/finally`），且 checkpoint 在评估后保存 |
| 2 | **参数名不一致**：`test_resume_regression.py` 用 `--data_path` 但 argparse 注册 `--data-path`，直接 argparse error | argparse 增加 aliases（`'--data-path', '--data_path'` 双写法），测试统一连字符格式 |
| 3 | **Test leakage**：`analyze_multimodality.py` 允许 `--split test/all` 拟合 GMM | `--split` 限制为 `choices=["train"]` + 硬校验 `raise ValueError` |
| 4 | **public class 判定**：原从提取样本反推 | 改为从 `client.class_mask` 判定：$\mathcal{C}_{\rm public} = \{c: \|\{i: c \in \mathcal{C}_i\}\| \geq 2\}$（public/private 是数据协议属性而非 feature 统计属性） |
| 5 | **`--no_instrumentation` 未生效**（🔴）：Server_DF 未读取该参数，OFF=ON，observer test 假通过 | Server_DF 增加 `self.instrumentation_enabled = not getattr(args, "no_instrumentation", False)`，只关闭 Phase-0 新增 observer（`_log_local_phase`、`run_full_evaluation`、margin 评估），**绝不关闭原版 `get_global_proto_and_head()` 内的 evaluate()** |
| 6 | **outer_split_manifest 缺失**（🔴）：test 读取该字段但 checkpoint 未保存，`ckpt.get(...) or {}` 导致假 PASS | `build_server()` 在生成 `client_data/client_mask` 后立即构造 `{client_id: {task_id: {"raw_indices": [...], "classes": [...]}}}` 存入 checkpoint；resume 时 exact 比较，不一致 `RuntimeError`；test 缺字段/空 manifest 直接 `AssertionError` |
| 7 | **oracle_k2.py 命名误导**：它是人工构造 $z' = z + s\delta v_c$ 的 synthetic stress test，非真实 Oracle-K2 | 改名 `synthetic_oracle_k2.py`，明确标注不可用于论文宣称真实 multimodality 结论 |
| 8 | **文档参数名错误**：RUNCOMMANDS.md 用 `--batch_size`/`--data_path` | 修正为连字符格式 |
| 9 | **outer manifest 假 PASS 残留**（第三轮复核确认）：`test_resume_regression.py` / `test_observer_invariance.py` 仍用 `ckpt.get("outer_split_manifest") or {}`，两侧都缺字段时 `{}` == `{}` 仍会 PASS；`Server_DF.load_checkpoint()` 仍宽松接受无 manifest 旧 checkpoint | 新增 `require_outer_manifest()` fail-fast 辅助函数（缺失/空 manifest 直接 AssertionError）并替换全部使用点；`load_checkpoint()` 改为缺失/空/不匹配三种情况一律 `RuntimeError` |
| 10 | **`--no_instrumentation` 未彻底等于 "FedTA + checkpoint only"**：`append_train_log_rows(server_rows)` 与 `append_round_metrics(i)` 无条件执行，OFF run 仍产生部分新增 observer artifact | 两处均加 `self.instrumentation` 门控（只门控 CSV/JSONL 写入）；原 FedTA 的 `get_global_proto_and_head()` 广播/评估循环保持无条件执行，**绝不放进 if** |
| 11 | **observer 测试缺封板断言**：未验证两组 run 的 `no_instrumentation` args（A 可能根本不是 OFF），也未验证 artifact 确实区分 | 增加 `ckpt_off["args"]["no_instrumentation"] is True` / `ckpt_on[...] is False` 硬断言 + artifact 区分验证（OFF 无 observer 文件、ON 必须有） |
| 12 | **Gate 0A 无专门测试**：官方 parity 检查此前只有人工对照计划 | 新增 `tests/test_official_fedta_parity.py`：driver 脚本进程内复刻官方 `main()` 流程并导出状态、严格 key 存在性断言（fail loudly）、`D=0` 判定，输出 `metrics/official_fedta_parity.json` |
| 13 | **测试潜伏 bug（第三轮新发现）**：两个测试的 `--work_dir` 默认为 str 但代码用 `/` 运算符（运行时 TypeError）；stale run 目录会触发 auto-resume 破坏等价性比较 | `--work_dir` 加 `type=Path`；三个测试运行前 rmtree 清理各自 run 目录 |
| 14 | **Gate 0A 默认 `global_epoch=2` 与官方不等价**（第四轮复核，🔴 阻塞）：官方 FedTA 硬编码 SIKF 分支为 `i % global_epoch != 4`，本仓库参数化为 `!= global_epoch - 1`，两者仅当 `global_epoch=5` 时等价（$4 = 5-1$）；用 `global_epoch=2` 跑 parity 比较的是两条不同训练路径，PASS 反而说明测试失效 | Gate 0A 强制 `global_epoch=5`（其他值直接 `SystemExit`）；默认改为 smoke 配置 `task_num=1, rounds=5, local_epoch=1`（第 4 轮已覆盖官方 `!=4` 分支），正式短程用 `task_num=2, rounds=10`。**Gate 0B/0C 保持 `global_epoch=2` 合法**：它们比较 ours vs ours（resume/observer 机制），不涉及官方硬编码等价性 |
| 15 | **inner split 假 PASS 残留**（第四轮复核，🟠）：B/C/A 仍用 `ckpt.get("data_split_state") or {}`，两侧都缺字段时 `{}` == `{}` 仍 PASS | 新增 `require_inner_split_state()` + `validate_split_state()` fail-fast（缺失/空/缺 train-test 键/train-test 空列表一律 AssertionError；且每 client 必须含运行到达的全部 task 的划分——`thisclients` 为全量无采样）；A/B/C 全部接入，官方侧 dump 的 `inner_splits` 同样做结构校验 |
| 16 | **Gate 0B accuracy matrix 假 PASS**（第四轮复核，🟠）：`matrix_diff()` 中 `ma is None and mb is None → 0.0`，两侧都缺 `accuracy_matrix.csv` 仍 PASS | Gate B 插桩开启、矩阵必须存在：两侧显式 `assert` 矩阵文件存在；`matrix_diff()` 单侧 None 直接 `inf` |
| 17 | **Gate B B2 resume 未显式传 `--max_rounds`**（第四轮复核）：默认配置下 `rounds == task_num*global_epoch` 恰好掩盖问题，非默认 rounds 时 resume 会跑满完整任务循环破坏等价性 | B2 命令显式附加 `--max_rounds args.rounds` |
| 18 | **B/C 漏比较若干 checkpoint 核心状态**（第四轮复核，🟡）：`temp_protos`、client 级 global/local protos、完整 tail_anchor state、client prompts（B 缺）、`existing_class`、`server_task_id` 均未比较 | B/C 比较矩阵补齐上述全部字段；`temp_protos`/client protos 增加"必须存在且非 None"断言（fuse_protos / 训练每轮必然写入） |
| 19 | **margin 计算混在官方 `evaluate()` 路径**（第五轮复核，⚠️ 不改算法但破坏严格 baseline execution path）：`Client_DF.evaluate()` 内嵌 margin 记录（gather/scatter/quantile），且不受 `--no_instrumentation` 控制——违反 OFF = "FedTA + checkpoint only" 的严格定义 | `evaluate()` 恢复官方原样（accuracy only，唯一增加是不影响计算的 `return float(acc)`）；新增 observer-only 的 `evaluate_with_margin()`（同一 accuracy 计算 + margin 记录）；`evaluate_all_seen_tasks()` 与 `diagnostics/compatibility_drift.py` 改调 wrapper（`_log_local_phase` 只需 accuracy，保持官方 `evaluate()`）；`__init__` 初始化 `last_margin_stats=None`。margin 代码不消耗 RNG，此改动对 Gate B/C 轨迹中立 |
| 20 | **Gate 0A driver 复刻错入口：官方 npz 加载路径**（服务器首跑 smoke 暴露）：driver 用 fork 的 `create_model(pretrained=True, pretrained_cfg=...)` 调用官方仓库的 ViT 文件，而官方 `_create_vision_transformer` 没有 fork 添加的 `pretrained_custom_load='npz' in url` 适配 → 新版 timm 把 `ViT-B_16.npz` 当 torch checkpoint `torch.load` 直接崩（`hasRecord("version")`）。官方 main.py 实际流程：`create_model(pretrained=False)` ×2 后**显式** `model.load_pretrained('pretrain_model/ViT-B_16.npz')`（官方 ViT 文件自带的 l2p `_load_weights` Flax 加载器） | driver 改为忠实复刻官方：两处 `create_model(pretrained=False)`（不传 `pretrained_cfg`）+ 显式 `load_pretrained(pretrained_path)`（含文件存在性 fail-fast）。fork 侧仍走 `pretrained=True` + custom load，最终同样进入 `_load_weights`，两条路径均 RNG 中立、初始权重逐位相同——由 Gate 0A 的 D=0 直接验证 |
| 21 | **`torch.load(weights_only=False)` 在服务器旧版 torch 上崩**（服务器 smoke 二跑暴露：两侧训练均完整跑完 "All Process completes"，仅测试进程加载 dump 比较时 TypeError: `weights_only` is an invalid keyword argument——服务器 torch < 1.13 无此参数）：三个测试文件 6 处裸调用（runtime_utils.py 的 load_checkpoint 已有兼容写法，未受影响） | 三个测试文件各新增 `torch_load_compat()`：`try: torch.load(..., weights_only=False)` / `except TypeError: torch.load(...)`，同时覆盖旧版 torch（无参数）与 torch>=2.6（默认 `weights_only=True` 会拒绝 checkpoint 内的 argparse.Namespace），6 处调用全部替换 |

第五轮复核同时确认：CIFAR-100 冻结协议下 Client 两阶段训练 / Tail Anchor / InfoNCE / Global Prompt（含 batchwise majority routing）/ BGPS（`threshold=0.25` 时）/ SIKF（`global_epoch=5` 时）/ FedAvg head / CIFAR partition 均与官方一致；单客户端 SIKF 分支的 `client.prompts` bugfix 在标准 5-client 全参与协议下不触发；ImageNet-R 不宣称 strict official parity。

> **封板状态说明（v5）**：第 5、6 两项连同第三轮（第 9–11 项）、第四轮（第 14–18 项）、第五轮（第 19 项）复核及 Gate 0A 服务器首跑（第 20、21 项）发现的假 PASS / 阻塞 / baseline 纯度问题**代码侧修复已全部落地**：
>
> - `--no_instrumentation`：Server_DF 门控 + observer args 断言 + artifact 区分验证 + **官方 `evaluate()` 路径去 margin 化**（margin 隔离进 `evaluate_with_margin()`）均在代码中封板
> - `outer_split_manifest` / `data_split_state`（inner split）/ accuracy matrix：三处 `or {}` / `None == None` 假 PASS 模式全部替换为 fail-fast 结构校验
> - Gate 0A 协议修正：强制 `global_epoch=5` + `threshold=0.25`（官方硬编码的唯一等价点），smoke 默认 `task_num=1/rounds=5/local_epoch=1`
>
> 但"实现完成 ≠ Gate 通过"：三个 Gate 测试（`test_official_fedta_parity.py` / `test_resume_regression.py` / `test_observer_invariance.py`）**均尚未在服务器真实跑通**。在服务器跑通并打 tag `phase0-fedta-baseline` 之前，上述各项只算"实现完成"，不算"已验证修复"。

### 12.3 待处理

| # | 问题 | 优先级 |
|---|---|---|
| 1 | **Phase 0 三个 Gate 测试未在服务器跑**：`test_official_fedta_parity.py`（Gate 0A）、`test_resume_regression.py`（Gate 0B）、`test_observer_invariance.py`（Gate 0C）均已实现（含假 PASS 封板），但未在服务器验证；第 5、6、9–11 项修复的最终确认依赖它们（见 §12.2 封板状态说明） | P0（Phase 0 封板前置） |
| 2 | **Gate 0A 前置准备**：服务器上需有官方 FedTA 仓库的本地克隆（含 `pretrain_model/ViT-B_16.npz`，数据集按官方路径约定就位）；若官方 config 模块名不在候选列表（`config.cifar100` / `config.cifar100_delay`）需用 `--config_module` 指定 | P0 |
| 3 | **z_op batch sensitivity 诊断未落地**：`diagnostics/check_feature_batch_sensitivity.py` 尚未实现（batchwise_prompt=True 导致 $z_{\rm op}$ 依赖 batch 组成，见 §5.9）；属 **Phase 1-precheck**（三个 Gate 通过 + 打 tag 之后、正式 Phase 1 之前） | P0 |
| 4 | ImageNet-R 独立官方 config（10/10/40/20/15/20） | P1 |

### 12.4 审查确认正确的部分

- CIFAR-100 外层划分（private/public、Dirichlet、task partition）与官方一致；`root` 参数化 + `download=False` 属环境工程修改
- 客户端内部 70/30 `random_split()` 保持原版，额外保存 indices 用于 resume 是正确增强
- Phase 1 数学实现全部正确：spherical GMM ↔ $\mathcal{N}(\mu, \sigma^2 I)$、80/20 held-out NLL、BIC 参数计数 $p_K = Kd + K + (K-1)$、mode separation $\rho$、coverage distortion $D_{\rm cover}$
- `build_server()` 重构保持 RNG stream 一致；diag_utils 通过 `build_server()` 重建数据环境正确
- feature extraction 正确取 pre-anchor $z$（未经 `client.model()`），同时保存 $z_{\rm ref}$ 与 $z_{\rm op}$ 双通道

---

## 13. 当前状态与下一步

### 状态快照（截至 2026-09-30，v4）

$$\boxed{\text{Phase 0 = implementation complete (3-Gate sealed in code, round-4 hardening done), Gate NOT passed}}$$

- **Phase 0**：三个 Gate（0A/0B/0C）的测试代码**已全部实现并封板**（假 PASS 防护、args 断言、artifact 区分验证、第四轮 inner-split/matrix/核心状态加固均落地，见 §12.2 第 9–18 项），**但 Gate 尚未通过**——三个测试在服务器真实跑通并打 tag `phase0-fedta-baseline` 之前，baseline **不算冻结**
- **Phase 1**：数学实现基本通过。待实现 batch sensitivity 诊断（Phase 1-precheck）+ 服务器跑特征提取/多模态分析
- **Phase 2+**：未开始，**一行 Phase 2 代码都不要继续加**（synthetic_oracle_k2.py 仅为 synthetic sanity check，不计入）

### 下一步行动（按序）

1. 服务器按顺序跑通三个 Gate（实现完成 ≠ Gate 通过；Gate 0A 强制 `global_epoch=5`，smoke 先行）：
   ```bash
   # Gate 0A smoke：官方 parity（需 --official_repo 指向官方 FedTA 仓库本地克隆）
   python tests/test_official_fedta_parity.py \
       --official_repo /path/to/official_FedTA \
       --data_path ./local_datasets \
       --global_epoch 5 --task_num 1 --rounds 5 --local_epoch 1 \
       --batch_size 16 --surrogate_num 20 --threshold 0.25 \
       --seed 42 --device cuda

   # Gate 0A 正式短程（覆盖 task transition）
   python tests/test_official_fedta_parity.py \
       --official_repo /path/to/official_FedTA \
       --data_path ./local_datasets \
       --global_epoch 5 --task_num 2 --rounds 10 --local_epoch 1 \
       --batch_size 16 --surrogate_num 20 --threshold 0.25 \
       --seed 42 --device cuda

   # Gate 0B：resume 回归（resume 正好跨 task boundary；ours vs ours，global_epoch=2 合法）
   python tests/test_resume_regression.py \
       --data_name cifar100 --data_path ./local_datasets \
       --rounds 4 --split_at 2 --global_epoch 2 --task_num 2 --local_epoch 2 \
       --seed 42 --device cuda

   # Gate 0C：observer 不变性（rounds=3 跨 task 边界，覆盖 next-task split）
   python tests/test_observer_invariance.py \
       --data_name cifar100 --data_path ./local_datasets \
       --rounds 3 --global_epoch 2 --task_num 2 --local_epoch 2 \
       --seed 42 --device cuda
   ```
2. 三个 Gate 全部 PASS 后打 tag `phase0-fedta-baseline`，冻结 baseline（`--method fedta` 从此视为 read-only scientific baseline）
3. 实现 `diagnostics/check_feature_batch_sensitivity.py`（§5.9 规格）并运行，确认 $z_{\rm op}$ 是否为稳定的 sample-level semantic representation（**Phase 1-precheck**，不是 Phase 0 Gate）
4. 跑 Phase 1 真实诊断：25 public classes × ($z_{\rm ref}$, $z_{\rm op}$)，产出 $\Delta NLL$、$\Delta BIC$、$\rho$、$D_{\rm cover}^{\rm rel}$、client-mode MI / JS divergence 及 bootstrap CI（Gate 判据见 §5.10）
5. **看到真实 multimodality 结果之前，不开发 Semantic-K2 / UOT / dynamic K；现在一行 Phase 2 代码都不要继续加**

### 给后续开发的硬约束（再次强调）

- 不修改：FedTA 训练 loss、optimizer、BGPS、SIKF、FedAvg head、CIFAR federated split、Tail Anchor 结构、prompt 逻辑 / batchwise_prompt
- 所有新增 diagnostic/evaluation 必须 observer-only；消耗 RNG 的代码必须 snapshot/restore
- Semantic mode discovery 只用 TRAIN split；test data 永远禁止
- 所有新增 loss weight 默认 0；全 0 且 K=1 时必须精确退化为 FedTA
- 每新增测试命令同步更新 `RUNCOMMANDS.md`
