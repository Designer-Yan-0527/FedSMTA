# FedSMTA A5 PR-2 验收报告（对 commit `9829638` 外部审查的回应）

日期：2026-10-21
审查对象：`9829638`（feat(A5 PR-2): prototype sync + proto-CE + U-only FedAvg）+ 后续补充提交 `9b72b48`（B7 测试与审计脚本）、`da94b60`（FL 基线适配器 resume 修复）
方法学基线：P2 v1.2 冻结设计（固定 V、只聚合 U、固定 z_ref 空间类统计、Stage-2 prototype-CE）

**总裁决：PR-2 A/B/C/D 验收通过。** 外部审查的三个 P0 项全部以新增测试闭环；七项验收表全部有 checkpoint 级证据；四路基线保护回归全部 PASS、逐位一致。唯一 open item 是审查 II.1（PR-1 时代 A5 checkpoint 恢复到 PR-2 的兼容性策略），现状为"缺键静默跳过"，未按建议改为默认拒绝——详见 §5.1，已排入 PR-3。

---

## 1. 审查 P0 项验证

### P0-A 原型表类别标签映射 —— 已闭环（新增测试 B7）

- 事实确认：T0 原型表的 66 个类 ID **确实非连续**（CIFAR-100 全局标签空间的任意子集），审查的假设成立，此风险是真实风险。
- 实现：proto logits 第 j 列对应 `classes[j]`（sorted 顺序），batch 标签映射为 `target(y) = index_{C_seen}(y)`；标签不在表内时 fail-closed 报错（B5），无静默跳过路径。
- 测试 `test_b7_noncontiguous_class_ids`（`tests/test_a5_pr2b.py`，commit `9b72b48`）：
  - 用非连续类集 {2, 7, 19, 53, 91} 且插入顺序故意打乱（53 先于 2）；
  - 断言独立重算的 loss 与实现逐位一致（<1e-5）；
  - 断言标签→列映射精确（y=[91,2,19,53,7,2] → 列 [4,0,2,3,1,0]）；
  - 断言**表的插入顺序不变性**：同一张表按 sorted 顺序重建后 loss 与 U 梯度完全一致（allclose 1e-6）——即不存在任何依赖 dict 顺序的隐性映射。
- 结果：PASS。

### P0-B proto-CE 对 U 的独立梯度 —— 已闭环（B1/B2/B6）

- B1 梯度隔离：`L_proto` 的 autograd 图仅触碰 U。断言 `‖∇_U L_proto‖ > 0`；V、μ_c、z_ref 为非梯度张量且逐位不变；Chead / Tail Anchor / prompt 不在计算图中（`test_b1_b3_gradient_isolation_and_formula`）。
- W 两侧通路：W·z 与 W·μ 两侧均未落入 `no_grad`——B2 用玩具问题证明**仅优化 U** 即可单调降低 L_proto（30 步 Adam，首末 loss 严格下降）。
- 公式精确性（B3）：`logits == cos(Wz, Wμ_c)/τ`（τ=0.07）独立重算逐位一致。
- λ=0 等价（B4）：`a5_proto_lambda=0` 时生产 gate 谓词为 False，proto-CE 完全不被调用（patch 为 raise 验证）。
- 数值稳定（B6）：特征尺度跨 5 个数量级（至 1e4）无 NaN/Inf（loss 与 U.grad 均有限）。
- 结果：全部 PASS。

### P0-C 任务边界同步幂等性 —— 已闭环（PR-2A 单测 + 精确对账）

- 单测（`test_a5_pr2a.py`）：同一任务版本重复提交同步，sum/count 不变（幂等）；版本号严格递增。
- 集成级精确对账（PR-2D smoke，非仅单测）：用 checkpoint 内 `outer_split_manifest` + `data_split_state` + CIFAR-100 原始 targets **离线独立重建**每类样本数：
  - version-0 表 == task-0 全体客户端 inner-train 的精确重建（逐类相等）；
  - version-1 计数 == version-0 基线 + task-1 精确贡献（100 类全对账；24 个重叠类为合法累积、34 个新类，**无任何类被重复累积**）。
- 结果：PASS。对账脚本已入库（`diagnostics/audit_a5_pr2_smoke.py`，commit `9b72b48`）。

---

## 2. PR-2D 集成验收（CIFAR-100 smoke：10 客户端 × 2 任务，T0 2 轮 + T1 resume 2 轮）

| 验收项 | 结果 | 证据 |
|---|---|---|
| T0→T1 原型表版本 | ✅ | version 0→1；表 66→100 类；重叠类合法累积、新类正确加入（§1 P0-C 的精确对账） |
| U 聚合 | ✅ | 有效残差样本数加权；聚合后**全部客户端 U 逐位一致**且非零（‖U‖≈0.33） |
| V 固定 | ✅ | checkpoint 内所有客户端 V 与确定性重生成（seed×100003+7）**逐位相等**，客户端间亦逐位相等 |
| Proto-CE | ✅ | loss 有限（T0 各轮 0.92–1.58，低于随机均匀 ln66≈4.19；仅陈述可分性，不主张语义迁移已发生——见 §6） |
| Checkpoint 恢复 | ✅ | T1 从 checkpoint 恢复：表/版本/类集/U/V/优化器状态正确；μ == sums/counts 精确重建（审计 A1/A2） |
| Native 指标 | ✅ | AA 矩阵正常；审计 A4：union(任务类掩码) == 表键；审计 A5：表内无 future-only 类 |
| 66/46 口径 | ✅ | 每客户端 own(t≤0)=20 类、exposed=46 类；task-1 后 own=40、exposed=60；逐客户端类别集合固化在 checkpoint 的 `outer_split_manifest`（键：client×task → 原始样本索引 + class_mask），可离线复算 |

审计脚本 `audit_a5_pr2_smoke.py` 的全部断言针对**磁盘 checkpoint**而非训练日志，直接回应"exit code 0 不等于验收"。

---

## 3. 基线保护回归（四路，全部 PASS）

| 测试 | 内容 | 结论 |
|---|---|---|
| `test_resume_regression` | fedta 连续 4 轮 vs 第 2 轮存档恢复续跑 | **PASS**：数据划分 hash、accuracy matrix、global protos、global head、server prompt、各 client task head / key / anchor pool、RNG 四路状态（torch/numpy/python/cuda）**全部逐位一致**，所有 max_diff = 0.0（产物 `output/regression/metrics/baseline_verification.json`，result=PASS） |
| `test_a1_parity` | A1 消融 vs 2×2 基线，同冻结划分 | **PASS**：10 行精度矩阵**逐位一致**（任务覆盖 [0,1]） |
| `test_a2_rejections` | A2 fail-closed 路径 | **PASS**：R1 arm-mismatch / R2 missing-a2_state / R3 cross-method 全部按预期拒绝；正向 sanity：a2_state 非空、arm 一致、保护集非空（5 客户端） |
| `test_observer_invariance` | Phase-0 插桩开关两遍 10 轮 | **PASS**：模型/原型/精度矩阵/RNG 全部 max_diff = 0.0；且带 seal 断言——obs_off 侧证明无插档产物、obs_on 侧证明有插档产物（排除空洞对比） |

结论：PR-2 的改动（纯插入 + 方法守卫）对 fedta / fedta_a2 旧方法**零行为影响**，证据等级为逐位一致，非统计近似。

---

## 4. 过程中发现并处理的两个缺陷（均不涉及基线方法代码）

1. **a2_rejections 的 CPU 失败**：测试曾以 `--device cpu` 运行并崩溃。根因是基线代码 `Models/Tail_Anchor.py:50`（commit `d4756ae`，早于本 PR 全部改动）硬编码 `.to('cuda')`。处置：GPU 上重跑全部通过；**基线代码未做任何修改**。教训：该基线仅支持 CUDA device，已记入回归运行约定。
2. **FL 基线适配器 resume 崩溃**（commit `da94b60`）：`protocol_adapters/run_fl_baseline.py` 用 `map_location=device` 加载 checkpoint，将 CPU RNG 状态 ByteTensor 搬上 GPU 导致 `torch.set_rng_state` 报错。修复为 `map_location="cpu"`。该文件是自研协议适配器（非基线方法本体）；修复只影响恢复路径，从头训练语义与已完成的 FedAvg/FedProx 结果均不受影响。

---

## 5. 边界风险现状

### 5.1 PR-1 A5 checkpoint 恢复到 PR-2 —— **open item，未按建议实现默认拒绝**

当前行为（`Models/Server_DF.py` RESUME_MUST_MATCH）：`a5_proto_lambda` / `a5_proto_tau` 已加入必配键，但**键缺失时静默跳过**（代码注释明示 PR-1 时代 checkpoint 缺这两键）。即：PR-1 A5 checkpoint（无 λ/τ、无 a5_proto_state）在 PR-2 代码下可直接恢复，且 λ 会取当前命令行默认 0.5——这正是审查所指"未记录的方法变更"。

- 审查建议的三分策略中，"旧 FedTA/A2 不受影响"与"PR-2 完整 A5 严格校验"两条已满足；**"PR-1 A5 默认拒绝、仅允许显式迁移"未实现**。
- 风险敞口评估：PR-1 时代仅产生过两个 A5 smoke run（均已废弃，不进入任何正式数据）；正式 A5 实验全部在 PR-2 代码下从零启动，不存在生产场景的 PR-1→PR-2 恢复需求。
- 计划：排入 PR-3，做法为 fail-closed——`method==fedta_a5` 且 checkpoint 缺 `a5_proto_state` 或缺 λ/τ 键时默认拒绝，需显式 `--allow_proto_migration` 才放行，并补 fail-closed 回归测试。

### 5.2 66/46 跨客户端暴露类 —— 已固化，正式协议审计待做

- smoke 已把逐客户端类别集合与 class_mask 固化进 checkpoint 的 `outer_split_manifest`，66/46 与 40/60 均可从 checkpoint 离线复算（审计 A4 自动对账）。
- 审查正确指出：两任务 smoke 的口径**不能替代**正式五任务协议的类集审计。此项将随 INR/CIFAR 正式五任务 run 逐 manifest 复算后出具。

---

## 6. 明确不主张的事项

1. smoke 中 79–83% 的准确率为 CIFAR-100 短程（2 轮/任务）smoke，仅用于机制跑通验证，**不作为 A5 的任何精度主张**。
2. `proto_ce=1.05 < ln66` 仅说明既有特征空间的可分性；共享语义迁移是否发生，需正式实验的对照臂（PR-3 四臂）判定。
3. A5 正式判据：INR 五任务 T4 终局 AA（对照 A2 四臂 control 78.62 / key 87.09 / A1 上界 91.67，seed42 探索性口径），按分数据集、分种子、分任务三级组织。

---

## 7. 相关提交与测试资产

| commit | 内容 |
|---|---|
| `9829638` | PR-2 主体（A/B/C）+ 15 项测试 |
| `9b72b48` | B7 非连续类 ID 测试 + `diagnostics/audit_a5_pr2_smoke.py` checkpoint 级审计 |
| `da94b60` | FL 基线适配器 resume RNG 修复（不影响基线方法语义） |

测试清单：`test_a5_pr1.py`（7/7）、`test_a5_pr2a.py`、`test_a5_pr2b.py`（B1–B7）、`test_a5_pr2c.py`、`test_resume_regression.py`、`test_a1_parity.py`、`test_a2_rejections.py`、`test_observer_invariance.py`——全部 PASS。

## 8. 下一步

PR-3：四臂开关（`a2_arm none` = Shared-only；`--a5_proto_lambda 0` = CE-only；Local Adapter = 关 U 聚合；完整 A5）+ §5.1 的 fail-closed 迁移守卫 + 8 项回归；随后进入 CIFAR/INR 五任务正式实验。
