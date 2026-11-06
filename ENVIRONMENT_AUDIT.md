# ENVIRONMENT_AUDIT — 实验环境完整性登记（独立于任何单个探针/报告）

## 1. `NUMPY_SCIPY_FLOAT64_BLAS_INVALID_ON_HOST`（2026-11-04 确认）

**登记级别：HOST-WIDE，影响一切本机 numpy/scipy float64 线性代数。与 BSAB 算法无关，不得埋进 BSAB 报告解释。**

conda env：`/root/autodl-tmp/users/ych/.conda/envs/FedSMTA`（numpy 1.23.x / scipy 1.10.x 栈）。

### 1.1 证据（全部可独立复跑）

| 检验 | 结果 |
|---|---|
| `x64 @ eye64 == x64`（恒等映射） | **FAIL**（不 bitwise 相等） |
| 同一 head 输入，float32 vs float64 路径 | 89.98 vs 25.48（60pp 级漂移） |
| 随机 dgemm (512×1536)@(1536×512)，numpy vs torch CPU | max diff **254**（torch 为对） |
| head layer-1 matmul，numpy float64 vs torch float64 | max diff 2.0 |
| `np.linalg.svd`（gesdd） | σ 错误（同一矩阵 vs torch 差 ~129）+ NaN 向量 |
| `scipy.linalg.svd(lapack_driver='gesvd')` | σ 与 torch 一致（1e-13），**但奇异向量损坏**：重构误差 6.5、\|UᵀU−I\|≈1.0 |
| numpy float32 gemm / torch CPU（float32 与 float64） | **全部 PASS**（eye bitwise、小矩阵精确、SVD 重构 1.4e-14、U 正交 2e-15） |

### 1.2 裁决（user，2026-11-04）

* 本机后续一切 float64 线性代数（SVD/求解/谱分析/大规模 gemm）**一律走 torch CPU**（独立 BLAS build，经精确用例验证）。
* **主训练 CUDA 路径不受影响**（torch CUDA/cuBLAS；numpy 仅做整数级操作）。
* 依赖 float64 奇异向量 / float64 matmul 的既有诊断结果**废弃**；仅经 torch 独立复核的标量/奇异值结果可保留。
* 具体波及：`B_PRE_RUN_01` 的 `acc_full_bridge`/`rank_acc` 数值（经损坏 float64 dgemm 路径）作废；其 σ 谱标量经 torch 复核一致可留档。见 RUNCOMMANDS §34。

### 1.3 已迁移的代码路径

* `diagnostics/probe_bsab_feasibility.py`：`svd_robust`/`fit_bridge`/`rank_variant` 全部 torch CPU float64；bridge 参数出口转 float32，head/feature 管线保持 float32 端到端（对官方 accuracy 矩阵 reproduction worst \|Δ\|=0.16pp 验证）。
* `diagnostics/bsab_fit_sanity.py`：同构。

### 1.4 后续守则

* 任何新诊断脚本如需 numpy/scipy float64 数值路径，必须先过本文件的 bitwise/exact 套件；不过则走 torch。
* 若未来更换/重建 conda env，须先重跑 1.1 检验套件再恢复 numpy/scipy float64 使用资格。
