# FedReLink 论文数据清单（paper_data/）

> 生成：2026-11-10，机器：autodl（本机）。所有文件均为**复制**，原始文件未移动、未修改。
> 用途：论文图表数据归档。在线训练数据后续将在 izfree 机器重跑以保证口径一致，本清单记录每份文件的原始来源路径，供核对哪些数据产自 autodl。
> 拷贝规则：仅 csv/json/md/txt/html + 小 npz（47KB）；排除 *.pt / *.pth / checkpoints / >20MB 的 .npz。

## 目录总览

| 文件夹 | 内容 | 文件数 |
|---|---|---|
| fig1_motivation/ | 多 seed 矩阵、TKR 复算、retention、遗忘轨迹 | 56 |
| fig_interface_swap/ | compatibility 2×2 换装 + D1 碰撞暴露（含 D0 槽位绑定） | 6 |
| fig4_hsp_intervention/ | 旧 A2 四臂干预报告（HTML + 图表数据 js） | 2 |
| fig5_bcb_birthpair/ | CIFAR B0 birth probe 四臂判决 + B_pre aging probe + birth preview | 11 |
| fig6_drift_geometry/ | （空，见"本机未找到"） | 0 |
| fig_guard_anticorr/ | （空，见"本机未找到"） | 0 |
| fig_worstcell_cdf/ | （空，见"本机未找到"） | 0 |
| fig_rank_interaction/ | （空，见"本机未找到"） | 0 |
| online_runs/ | 4 个在线 run 的 metrics 完整拷贝 + 口径三件套 | 66 |
| docs_reports/ | 两份 GPT 审查报告 | 2 |
| **合计** | | **143**（约 1.3MB，另有本 README） |

---

## online_runs/（4 个在线 run，全部在本机 autodl ✓）

结构：`<run名>/seed_42/metrics/...` + 口径三件套（`args.json` / `command.txt` / `git_commit.txt`，用于 izfree 重跑时核对命令与 commit）。

| run | 原始路径（autodl） | git_commit | 备注 |
|---|---|---|---|
| cifar_base_p15 | output/cifar100/fedta/cifar_base_p15/seed_42/ | fa06fe78 | metrics 9 文件 + dsar_g1_invariance.json（G1 轨迹中立门）+ superseded 旧版 γ 门 |
| cifar_b1_dsar_on | output/cifar100/fedta/cifar_b1_dsar_on/seed_42/ | 见目录内 git_commit.txt | metrics 9 文件 |
| inr_dsar_v1 | output/ImageNet-R/fedta/inr_dsar_v1/seed_42/ | f96c95af | metrics 9 文件 + 9 个 `.bak_dupes`（11-09 去重前旧数据备份，一并保留作溯源） |
| inr_base_p40 | output/ImageNet-R/fedta/inr_base_p40/seed_42/ | 见目录内 git_commit.txt | metrics 11 文件（多 a3_calibration.csv/json） |

注意：
- `inr_base_p40` 本机**存在**（任务曾提示可能 MISSING-on-autodl，已核实不缺）。
- 各 run 原目录下的 `checkpoints/`（*.pth）与 `logs/train.log` 未拷贝（规则排除/非 metrics）。
- 原 `output/ImageNet-R/fedta/inr_dsar_v1/seed_42/metrics.bak_dupes/` 为**空目录**，未拷。

## fig1_motivation/（A1 分解、遗忘轨迹、多 seed 稳健性、TKR 复算）

| 文件 | 原始路径 | 大小 | 备注 |
|---|---|---|---|
| tkr/tkr{,_seed42,_seed1999,_seed2025}.{csv,json} | diagnostics_output/tkr/ | 各 ~2.2–3.0KB | verify_tkr.py 作者口径 TKR 复算（三 seed） |
| cifar_base_p15_legacy/seed_42/accuracy_matrix.csv 等 8 文件 | output/cifar100/fedta/cifar_base_p15_legacy/seed_42/ | 142B–24KB | 遗忘轨迹（accuracy/summary/margins/round_metrics/train_log）+ tkr_out.csv + class_retention{,_4combo}.csv（retention>100% 证据；4combo 同时是 D1 的输入） |
| cifar_base_p15_legacy/seed_1999/metrics/（5 文件） | output/cifar100/fedta/cifar_base_p15_legacy/seed_1999/metrics/ | 共 ~17KB | 多 seed 矩阵（无 client_*_accuracy.csv，原目录即无） |
| cifar_base_p15_legacy/seed_2025/metrics/（5 文件） | output/cifar100/fedta/cifar_base_p15_legacy/seed_2025/metrics/ | 共 ~17KB | 同上 |
| inr_hsn_p40/seed_{42,1999,2025}/metrics/（各 10 文件） | output/ImageNet-R/fedta/inr_hsn_p40/seed_*/metrics/ | 各 ~18KB | INR 多 seed（HS-N 修复版 run，--skip_fed_avg_head）。PAPER_OUTLINE 的多 seed 数字（93.8±0.3 等）为 CIFAR；INR 多 seed 用途待论文定稿确认，一并归档 |

**A1 接口恢复分解数据本机未找到**（见"本机未找到"节），PAPER_OUTLINE.md §1.2 的分解表（T0 8.59/7.39/86% … 均值 86.4%）目前只有文档数字。

## fig_interface_swap/（compatibility_drift 2×2 换装 + D1 固定效应回归）

| 文件 | 原始路径 | 大小 | 备注 |
|---|---|---|---|
| compatibility_2x2_t00_t04_fixed.csv | output/cifar100/fedta/cifar_base_p15_legacy/seed_42/ | 1020B | key/anchor 新旧世代 2×2 交叉换装（t00 出生态 vs t04） |
| compatibility_2x2_t01_t04.csv | 同上 | 1021B | t01 vs t04 换装 |
| collision_exposure.csv | 同上 | 56KB | D1 v2 碰撞暴露诊断（collision_exposure_diagnostic.py）行级数据，(client,task) FE 回归输入 |
| collision_exposure.json | 同上 | 2.4KB | D1 相关性汇总（spearman E_future/d_key/d_anchor→KA 链）；PAPER_OUTLINE 的 FE rho=+0.259, p=0.001（anchor 位移通道）由该数据算出 |
| slot_binding.csv | 同上 | 29.5KB | D0 槽位绑定（verify_slot_binding，D1 前置） |
| slot_binding_matrices.npz | 同上 | 47KB | 绑定矩阵（<20MB，放行） |

来源 run：cifar_base_p15_legacy（seed_42），autodl 本机产。

## fig4_hsp_intervention/（旧 A2 四臂干预）

| 文件 | 原始路径 | 大小 | 备注 |
|---|---|---|---|
| a2_verdict_report.html | reports/a2_verdict_report/ | 41KB | "复现·问题发现·A2 定型·A3 展望"报告，含旧 A2 四臂数字（F 18.89→7.11、AA 78.62→87.09 等） |
| assets/charts.js | reports/a2_verdict_report/assets/ | 10.5KB | 报告图表数据（_shared/ 字体与 echarts.min.js 为前端库，未拷） |

注意：旧 A2 四臂的**原始 run 目录**（output/cifar100/fedta_a2/...）本机不存在（见"本机未找到"），四臂数字仅存于此报告。新四臂矩阵（A2 / A2+DSAR 臂）尚待发车（RUN.md）。

## fig5_bcb_birthpair/（CIFAR B0 birth probe，125 cell）

| 文件 | 原始路径 | 大小 | 备注 |
|---|---|---|---|
| cifar_b0/cells.csv | output/cifar100/fedta/cifar_bsab_capture_M2matched/seed_42/bsab_b0/ | 24.9KB | 125 cell 四臂读数；**含 rank sweep**：acc_lowrank_r8 / r16 / r32 / r64 列齐全，无需单独拷 rank sweep 数据 |
| cifar_b0/verdict.json | 同上 | 1.3KB | GREEN 判决（ρ mean=1.006 / pooled=1.018） |
| cifar_b0/repro.csv | 同上 | 5.3KB | reproduction gate（arm1 25/25 PASS） |
| cifar_b0/gamma_star_corrected.json | 同上 | 546B | 修正版 γ*=0.396419（SHA-256 f3174a...） |
| cifar_b0/gamma_star.json | 同上 | 258B | superseded 旧版 γ=0.390324（历史证据保留） |
| cifar_b0/dsar_g2_signoff.json | 同上 | 90KB | G2 signoff |
| dsar_birth_preview/{cells.csv, summary.json} | output/cifar100/fedta/cifar_bsab_capture_M2matched/seed_42/dsar_birth_preview/ | 32KB+3.7KB | baseline 部署口径（联训头）：损伤 2.85pp、B_t 免标签修回 82%（b0_results_for_gpt.md §5/6 数字来源） |
| bsab_pre/{bsab_pre_report.json, bsab_pre_report_run01_fitsanityfail.json, fit_sanity_gate.json} | diagnostics_output/bsab_pre/ | 96KB+81KB+1.6KB | B_pre aging probe（32/37 valid cell ρ_age≥0.5，DSAR 发现链先验） |

排除：`bsab_b0/heads/*.pt`（30 个重训头缓存，规则排除）。
来源 run：cifar_bsab_capture_M2matched（capture run，与官方轨迹逐位一致），autodl 本机产。

## docs_reports/

| 文件 | 原始路径 | 大小 | 备注 |
|---|---|---|---|
| b0_results_for_gpt.md | 项目根目录 | 6.9KB | CIFAR B0 判决汇总（含 §6 涨幅表） |
| inr_scheme_probe_for_gpt.md | 项目根目录 | 9.9KB | INR 方案有效性探针汇总（B 网格 4×5、guard、冻结配方 dsar-inr-v1） |

## online 数据与图的对应关系（口径提示）

- 遗忘轨迹 / ΔF / AA 主表：online_runs/ 的 accuracy_matrix.csv + summary_metrics.csv（cifar_base_p15 vs cifar_b1_dsar_on；inr_base_p40 vs inr_dsar_v1）。
- 在线验证四门（ΔAA/G_old/ΔF/minΔ_k）读数由上述 base 与 dsar run 的 metrics 差分得到（PAPER_OUTLINE §2.5）。
- inr_dsar_v1 的 metrics 有 11-09/11-10 两个时间层（.bak_dupes 为去重前旧层），用主文件为准。

---

## 本机未找到（大概率在 izfree 机器）

以下数据在 autodl 全盘搜索未命中，按 RUNCOMMANDS.md 执行记录，均在 **izfree** 机器产生（INR capture run：`output/ImageNet-R/fedta/inr_bsab_capture_M2matched_p40/seed_42`）：

| 图 | 缺失数据 | 搜索过的关键词 | 目录 |
|---|---|---|---|
| fig1_motivation | **A1 接口恢复分解**（a1_snapshot_eval.csv） | a1, a1_snapshot, interface, recovery, birth, snapshot_eval, 22.75, 86.4 | diagnostics_output/、output/ 全域、reports/；RUNCOMMANDS.md L343-345 记录其应输出到 cifar_base_p15_legacy/seed_42/ 但该文件不存在（可能未跑或被清理）；代码 diagnostics/a1_snapshot_eval.py 在 |
| fig6_drift_geometry | **INR B0-lite**（RED 判决：ρ 0.315/0.418、eff_rank 343） | b0_lite, inr*b0, B0-lite, eff_rank, effective_rank | RUNCOMMANDS §34.5：izfree `--no_repro` 125 cell |
| fig6_drift_geometry | **INR γ\***（γ*_INR=2.0579118185313345, provenance 6cd83d9） | gamma, gamma_star | 本机仅有 CIFAR 版（已入 fig5）；INR 版为 izfree `bsab_gamma_star.py` 输出 |
| fig6_drift_geometry | **位移统计 / 有效秩对照**（位移小 32% 等） | displacement, eff_rank, spectrum, spectral, norm_shift | diagnostics_output/ 全部 json/csv 内容级 grep 无命中 |
| fig_guard_anticorr | **INR cer vs ce 对照、guard 反相关分解 cells**（Q2 拒绝 16% cell 真实增益 +11.5pp） | cer, guard, anticorr, Q2 | 产物应为 `dsar_inr_probe/`（izfree） |
| fig_worstcell_cdf | **INR scheme probe L16/L8 worst-cell**（L16_zp_no min=-1.20 / p5=-0.56 等） | worst, scheme_probe, dsar_inr_probe, L16 | RUNCOMMANDS §34.7：数字存于 izfree 的 cells.csv 复算 |
| fig_rank_interaction | **dsar_inr_hist_probe 历史行 B rank**（B8_inr 唯一全距离为正） | hist_probe, dsar_inr_hist | 代码 diagnostics/dsar_inr_hist_probe.py 在本机，产物在 izfree |
| fig4_hsp_intervention | **旧 A2 四臂原始 run**（output/cifar100/fedta_a2/a2_{anchor,key,both}/...） | fedta_a2, a2_arm, a2_anchor, four_arm, interven | output/cifar100/ 与 output/ImageNet-R/ 下均无 fedta_a2 目录；四臂数字仅存 fig4 的 HTML 报告 |

补充说明：
- inr_scheme_probe_for_gpt.md 记载产物目录 `dsar_inr_probe/`、`dsar_inr_probe_bgrid/` 在本机不存在（已确认 tmp/ 为空、diagnostics_output/ 29 个子目录逐一盘点无此项）。
- izfree 重跑在线训练时，建议同时回收上述 8 项离线探针产物并按本目录结构补齐 fig6 / fig_guard_anticorr / fig_worstcell_cdf / fig_rank_interaction。

## diagnostics_output/ 盘点备注（29 子目录，未拷部分）

以下子目录属于判死档案/中间探针，不在 8 个图的数据清单内，未拷贝：boundary_probe、contentdrift_cifar(+csv/json)、contentdrift_smoke、coord_pinning、corrected_reeval、ctrp_forensic_inr_task1_round9、ctrp_gate_amendment_regression、dht_cifar(+csv)、eval_mode_reeval、feat_base、feat_inr_base、features、fusion_probe、head_retrain_probe、misc、ncm_probe、oracle_headroom、proto_head、refit_input_probe、routeflip_cifar(+csv)、routeflip_smoke（空）、row_usage、ssra、stepcalib_cifar(+csv)、stepcalib_inr_seed42(+csv)。其中含多个 >20MB 的 .npz 特征转储（28–284MB），按规则排除。
