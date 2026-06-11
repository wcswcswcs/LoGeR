# Stream4D v12 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/stream4d_v12_executable_object_explanation_plan_for_codex.md`  
执行日志: `docs/stream4d_v12_执行日志.md`  
结论先行: v12 实现了 measurement bank -> object explanation -> posterior support -> unified evaluation 的最小闭环，但没有得到可作为正式 method table 改进的结果。M5 证明 negative evidence/WTA 和真实 D4RT temporal evidence 有正贡献，但 own AP 仍低于 gate，S0/S1 cross-support 仍崩溃。修复 M5r1/M5r2/M5r3 未解决核心问题，tune30/final 未启动。

## 结果边界

- 所有 AP/AP50/AP25 来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt`。
- Phase 2 oracle 使用 GT selection，只能作为 diagnostic upper bound，不能进入 method table。
- M4/M5/M6/M7/M5r1/M5r2/M5r3 是无 GT method-result 配置，reportable scan clean。
- Cross-support 行是 diagnostic-only，不写成方法成功。
- 没有把 `NA`、`nan`、空预测或 oracle 结果改写成有利数字。

## Phase 0 统一基准

输出: `Stream3D/outputs/audit/v12_phase0/unified_eval_matrix_probe5.json`

| row | AP | AP50 | AP25 | pre% | union% | conflict | best IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 Stream3D on S0 | 0.235730 | 0.414306 | 0.537786 | 84.6744 | 84.6744 | 0.2213 | 0.630750 |
| P0 Stream3D on S1 | 0.399213 | 0.597171 | 0.742535 | 4.5145 | 84.6744 | 0.2213 | 0.736970 |
| P2 B1 on S2 | 0.328439 | 0.629266 | 0.884363 | 3.9861 | 3.9861 | 8.4307 | 0.691363 |
| P2 B1 on S0 | 0.000635 | 0.004294 | 0.010768 | 84.6744 | 3.9861 | 8.4307 | 0.031298 |
| P4 O38 on S0 | 0.033012 | 0.123089 | 0.392066 | 84.6744 | 66.6809 | 3.9025 | 0.362653 |
| P5 R1b on S1 | 0.064218 | 0.169101 | 0.415508 | 4.5145 | 20.6836 | 0.0000 | 0.570806 |

证据链:

- 完整矩阵为 6 predictions x 6 supports，共 36 行。
- `P0 Stream3D on S0` 与 v10 Phase 0 probe5 baseline 一致。
- `P0 Stream3D on S1/S5` AP=0.399213，说明稀疏 support 仍强烈改变指标。
- B1/O1 在 tiny support 上 own 高，但 S0/S1 崩溃的历史结论再次成立。

分析:

- support 本身是强混杂变量；v12 后续不能只看 own-support。
- S1/S5 support 对 Stream3D 很友好，但 B1/O1/M5 类方法在 S1 上仍远低于 Stream3D，说明问题是 object assignment/coverage，不是 evaluator。

## Phase 1 Measurement Bank

输出:

- `Stream3D/outputs/v12_measurement_bank/*/measurement_bank.npz`
- `Stream3D/outputs/audit/v12_measurement_bank/measurement_bank_probe5.json`
- `Stream3D/outputs/audit/v12_measurement_bank/measurement_bank_probe5.md`

| metric | value |
|---|---:|
| num_surfels_mean | 16384.0 |
| uv_in01_rate_mean | 0.9858451843261719 |
| visible_ok_rate_mean | 0.8302947998046875 |
| track_length_visible_mean | 13.284716796875 |
| self_uv_error_p90_mean | 1.5708253622055055 |
| cycle_uv_error_p90_mean | 3.2737268686294554 |
| mean_positive_observations_per_surfel | 1.6324951171875 |
| mask_to_surfel_count_mean | 941.2109458504808 |
| boundary_safe_surfel_ratio | 0.83087158203125 |
| ambiguous_surfel_ratio | 0.012196359731626518 |
| unobserved_surfel_ratio | 0.14576416015625 |

Per scene:

| scene | uv in01 | visible len | obs/surfel | mask->surfel | boundary safe | ambiguous | unobserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.990829 | 15.7715 | 1.9639 | 918.4571 | 0.9881 | 0.0039 | 0.0030 |
| scene0011_00 | 0.959251 | 9.6402 | 1.1931 | 602.2903 | 0.6526 | 0.0119 | 0.3293 |
| scene0030_00 | 0.999809 | 15.9829 | 1.9926 | 1925.2353 | 0.9944 | 0.0020 | 0.0000 |
| scene0081_01 | 0.982063 | 11.5381 | 1.4441 | 1045.3636 | 0.7234 | 0.0210 | 0.2458 |
| scene0591_00 | 0.997272 | 13.4908 | 1.5687 | 214.7083 | 0.7958 | 0.0222 | 0.1507 |

Gate:

- uv/cycle/self error 通过 Phase 1 carrier-quality gate。
- mean positive observations per surfel = 1.6325，通过 1.5 threshold。
- mask_to_surfel_count_mean 远高于 16。
- 风险: 当前 mask cache 仍只有 2/16 frames，scene0011/0081/0591 unobserved surfel 比例偏高。

## Phase 2 Candidate Oracle

输出: `Stream3D/outputs/audit/v12_candidate_oracle/candidate_oracle_matrix_probe5.json`

| oracle | AP | AP50 | AP25 | pre% | union% | conflict | best IoU | method table |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C_mask | 0.224691 | 0.453333 | 0.648889 | 60.8842 | 46.3060 | 2.3829 | 0.461338 | False |
| C_regionlet | 0.338574 | 0.613208 | 0.829643 | 18.5455 | 7.2183 | 0.0965 | 0.605980 | False |
| C_surfel_cluster | 0.395062 | 0.750000 | 0.993360 | 4.2916 | 3.7916 | 12.9326 | 0.652205 | False |
| C_hybrid | 0.256757 | 0.495495 | 0.702512 | 52.7795 | 32.5679 | 1.6555 | 0.510990 | False |

证据链:

- Oracle configs manifest 写明 `uses_gt_for_prediction=true`、`uses_gt_for_diagnostic=true`、`is_diagnostic_only=true`、`is_method_result=false`。
- `evaluation.evaluate` 仅通过 `--allow-oracle-eval` 评估 oracle。
- C_surfel_cluster oracle 很高，但 support 只有 4.2916%，属于 tiny-support upper bound。
- C_hybrid support 大到 52.7795%，但 AP=0.256757、AP50=0.495495，没有给出足够强 broad-support 上界。

分析:

- v12 primitive 有局部上界: C_surfel tiny oracle AP50=0.75、AP25=0.993。
- broad support 的 C_hybrid 上界不够，说明把当前 primitive 直接扩成完整 reconstruction 仍不可靠。
- M5 method AP50=0.4536 明显低于 C_surfel oracle AP50=0.75，说明 inference/set-packing 还有差距；但 C_surfel 的 tiny support 也解释了 cross-support 失败。

## Phase 3 Object Explanation

输出:

- `Stream3D/outputs/audit/v12_object_explanation/object_explanation_matrix_probe5.json`
- `Stream3D/outputs/audit/v12_object_explanation/object_explanation_internal_probe5.json`
- `Stream3D/outputs/audit/v12_object_explanation/reportable_config_scan_v12_object_probe5.json`

| row | AP | AP50 | AP25 | pre% | union% | conflict | best IoU | method table |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M4 no negative own | 0.140914 | 0.262547 | 0.459596 | 4.2958 | 4.2958 | 99.1226 | 0.689433 | True |
| M5 negative own | 0.226671 | 0.453586 | 0.745367 | 4.2958 | 4.2958 | 55.2235 | 0.651884 | True |
| M6 shuffled D4RT own | 0.006107 | 0.032675 | 0.107804 | 3.6947 | 3.6947 | 82.4188 | 0.281376 | True |
| M7 no temporal own | 0.138615 | 0.258599 | 0.429016 | 4.2958 | 4.2958 | 99.1226 | 0.689433 | True |
| P0 Stream3D on M5 | 0.358958 | 0.549096 | 0.740956 | 4.2958 | 84.6744 | 0.2213 | 0.724195 | False |
| M5 on S0 | 0.002042 | 0.005558 | 0.010704 | 84.6744 | 4.2958 | 55.2235 | 0.031388 | False |
| M5 on S1 | 0.023149 | 0.049873 | 0.148286 | 4.5145 | 4.2958 | 55.2235 | 0.167469 | False |

Ablation:

| comparison | delta AP | delta AP50 | delta AP25 | result |
|---|---:|---:|---:|---|
| M5 - M4 | +0.085757 | +0.191039 | +0.285771 | negative/WTA helps |
| M5 - M6 | +0.220564 | +0.420911 | +0.637562 | real D4RT >> shuffled |
| M5 - M7 | +0.088056 | +0.194987 | +0.316350 | temporal evidence helps |

Internal evidence:

| config | slots | assigned | core | reject | explained | cannot-link |
|---|---:|---:|---:|---:|---:|---:|
| M4 | 14.80 | 0.1148 | 0.1147 | 0.0000 | 0.5292 | 14.20 |
| M5 | 12.80 | 0.0998 | 0.0997 | 0.0002 | 0.5292 | 0.00 |
| M6 | 5.00 | 0.0082 | 0.0082 | 0.0000 | 0.2900 | 5.80 |
| M7 | 14.80 | 0.1145 | 0.1140 | 0.0000 | 0.5292 | 14.20 |

分析:

- M5 成功证明 signed evidence 有效果: cannot-link violations 从 M4 的 14.20 降到 0，conflict 从 99.1% 降到 55.2%。
- M5 明显强于 shuffle/no-temporal，说明 D4RT temporal/source evidence 真正进入了 object inference。
- 但是 M5 own AP=0.226671 < 0.30，AP50=0.453586 < 0.55，S0/S1 远低于 gate。
- Same-support 上 `P0 on M5 support` AP=0.358958，高于 M5 own AP=0.226671，说明同一 support 对 Stream3D 可用，M5 的 object assignment 仍弱。

## Repair Attempts

输出: `Stream3D/outputs/audit/v12_object_explanation_repair/object_explanation_repair_matrix_probe5.json`

| row | AP | AP50 | AP25 | pre% | conflict | best IoU |
|---|---:|---:|---:|---:|---:|---:|
| M5r1 relaxed birth own | 0.228270 | 0.424798 | 0.723425 | 4.4771 | 55.2381 | 0.662796 |
| M5r1 on S0 | 0.002028 | 0.005431 | 0.010291 | 84.6744 | 55.2381 | 0.033446 |
| M5r1 on S1 | 0.022217 | 0.046343 | 0.152798 | 4.5145 | 55.2381 | 0.175302 |
| M5r2 target birth own | 0.139815 | 0.237037 | 0.419444 | 0.6388 | 0.6379 | 0.484909 |
| M5r2 on S0 | 0.000000 | 0.000000 | 0.000000 | 84.6744 | 0.6379 | 0.003204 |
| M5r2 on S1 | 0.000000 | 0.000000 | 0.024096 | 4.5145 | 0.6379 | 0.014486 |
| M5r3 strict posterior own | 0.237230 | 0.455495 | 0.729781 | 4.2756 | 55.1810 | 0.662560 |
| M5r3 on S0 | 0.002042 | 0.005558 | 0.012047 | 84.6744 | 55.1810 | 0.031654 |
| M5r3 on S1 | 0.022346 | 0.046761 | 0.156118 | 4.5145 | 55.1810 | 0.166754 |

修复判断:

- M5r1 放宽 birth/core 后没有增加有效 cross-support，只把 union 从 4.2958% 提到 4.4771%。
- M5r2 target birth 明显退化，平均只导出约 1 个 object。
- M5r3 strict posterior 是本轮最高 own AP=0.237230，但仍低于 gate，并且 S0/S1 仍近零。

## Gate 判断

| lane | gate | result | evidence |
|---|---|---|---|
| Phase 0 protocol | complete matrix + manifest pass | pass | 36 rows, manifest pass true |
| Phase 1 measurement bank | uv>=0.95, self<=3, cycle<=6, obs/surfel>=1.5 | pass | 0.985845, 1.5708, 3.2737, 1.6325 |
| Phase 2 primitive oracle | broad-support upper bound competitive | partial/fail | C_surfel tiny high; C_hybrid AP=0.2568 |
| M5 own | AP>=0.30, AP50>=0.55, AP25>=0.75 | fail | 0.2267/0.4536/0.7454 |
| M5 S0 | AP>=0.08, AP50>=0.18, AP25>=0.45 | fail | 0.0020/0.0056/0.0107 |
| M5 S1 | AP>=0.18, AP50>=0.35, AP25>=0.60 | fail | 0.0231/0.0499/0.1483 |
| Ablation M5 > M4/M6/M7 | real D4RT and negative help | pass | M5 AP50 deltas +0.191/+0.421/+0.195 |
| Repair attempts | fix M5 gate | fail | best M5r3 0.2372/0.4555/0.7298 |
| tune30/final | Phase 3 probe5 pass | not started | gate failed |

## 主要 Insight

1. v12 object explanation 闭环是可执行的，但还不是足够强的方法。它从 measurement bank 产生 slot/posterior/support，并能进入统一评估，但 AP/cross-support 不达标。
2. Negative evidence 和 WTA 是真实有效因素。M5 相比 M4 大幅降低 cannot-link 与 conflict，并显著提升 AP/AP50/AP25。
3. D4RT temporal evidence 不是摆设。M5 明显强于 M6 shuffle 和 M7 no-temporal，说明 real D4RT signal 进入了 posterior。
4. 失败仍然是 tiny-support + object assignment 双重问题。M5/M5r3 support 约 4.3%，S0/S1 基本崩；同 support 下 Stream3D 又强于 M5，说明 support 和 assignment 都不够。
5. Oracle 暴露了上界分裂: tiny C_surfel oracle 很高，但 broad C_hybrid 上界不强。当前 primitive 能解释局部 clean object，不能稳定解释完整 scene。
6. 当前 mask observation 频率仍是瓶颈。Measurement bank carrier 指标通过，但只有 2/16 mask frames，unobserved surfel ratio 在部分 scene 偏高。

## 结论

Stream4D v12 在 probe5 上没有获得可报告 SOTA 或超过 Stream3D 的方法结果。最强 method-result 是 M5r3 strict posterior own `0.237230 / 0.455495 / 0.729781`，仍低于 probe5 minimal gate，且 `on S0/S1` 分别只有 `0.002042 / 0.005558 / 0.012047` 和 `0.022346 / 0.046761 / 0.156118`。

本轮最可靠的正结论是: signed evidence、negative/WTA 和真实 D4RT temporal evidence 都有因果贡献。最可靠的负结论是: 当前 measurement density 和 deterministic slot posterior 仍不足以把 tiny clean evidence 扩展为完整 object reconstruction。下一步若继续，应优先增加更密集的 semantic/video masklet observations，或引入更强的 object-level set packing/MDL/split-merge inference；不能把 v12 当前结果写成正式方法成功。

## 审计材料

审计包:

- `stream4d_v12_probe5_code_review_packet.zip`
- SHA256: 见 `stream4d_v12_probe5_code_review_packet.sha256`
- filelist: `stream4d_v12_probe5_filelist.txt`
- git diff: `stream4d_v12_probe5_git_diff.patch`
- git status: `stream4d_v12_probe5_git_status.txt`
- zip test: `stream4d_v12_probe5_ziptest.log`

审计通过:

```text
py_compile: pass
unittest discover tests: Ran 16 tests ... OK
reportable scan: num_reportable_method_configs=7, num_suspicious_configs=0, num_uses_gt_for_prediction=0
metric integrity: phase0_pass=True
```
