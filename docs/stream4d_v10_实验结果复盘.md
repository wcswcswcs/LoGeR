# Stream4D v10 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/stream4d_v10_unified_eval_regionlet_d4rt_geometry_plan_for_codex.md`  
执行日志: `docs/stream4d_v10_执行日志.md`  
结论先行: v10 没有得到可作为正式 method table 的改进结果。Phase 0 确认历史 own-support 高分存在 support illusion；Regionlet birth/repair 未通过 cross-support gate；D4RT geometry 替换只得到 diagnostic-only 证据，不能作为正式方法结果。

## 结果边界

- 所有 AP/AP50/AP25 来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt`，由 matrix 工具解析。
- 所有 support/union/conflict/best-IoU 来自 `outputs/audit/*/*.json` 的诊断字段。
- Fresh D4RT carrier 来自 `outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67`。
- G1-G5 D4RT geometry 使用 diagnostic anchors，因此 `is_method_result=false`，`is_diagnostic_only=true`，不能进入 method table。
- 没有把 NA、nan 或空预测改写成有利数字。G1 raw 的 own AP 在 matrix 中保持 `NA`。

## Phase 0 统一评估

输出: `Stream3D/outputs/audit/v10_phase0/unified_eval_matrix_probe5.json`

| row | AP | AP50 | AP25 | pre% | union% | conflict | method table |
|---|---:|---:|---:|---:|---:|---:|---|
| P0 Stream3D on S0 | 0.235730 | 0.414306 | 0.537786 | 84.6744 | 84.6744 | 0.2213 | False |
| P0 Stream3D on S1 | 0.399213 | 0.597171 | 0.742535 | 4.5145 | 84.6744 | 0.2213 | False |
| P2 B1 own | 0.328439 | 0.629266 | 0.884363 | 3.9861 | 3.9861 | 8.4307 | True |
| P2 B1 on S0 | 0.000635 | 0.004294 | 0.010768 | 84.6744 | 3.9861 | 8.4307 | False |
| P2 B1 on S1 | 0.016837 | 0.047534 | 0.168162 | 4.5145 | 3.9861 | 8.4307 | False |
| O1 core-only own | 0.349893 | 0.526807 | 0.861595 | 3.6634 | 3.6634 | 0.0000 | True |
| O1 core-only on S0 | 0.000286 | 0.001150 | 0.009731 | 84.6744 | 3.6634 | 0.0000 | False |
| O1 core-only on S1 | 0.013431 | 0.042531 | 0.130559 | 4.5145 | 3.6634 | 0.0000 | False |
| O38 c055 own | 0.081038 | 0.219225 | 0.492501 | 66.6809 | 66.6809 | 3.9025 | True |
| O38 c055 on S0 | 0.033012 | 0.123089 | 0.392066 | 84.6744 | 66.6809 | 3.9025 | False |
| O38 c055 on S1 | 0.096291 | 0.226815 | 0.559930 | 4.5145 | 66.6809 | 3.9025 | False |

证据链:

- reportable scan: `num_reportable_method_configs=6`，`num_suspicious_configs=0`，`num_uses_gt_for_prediction=0`。
- metric integrity: `phase0_pass=True`。
- B1/O1 在 own-support 高，但换到 S0/S1 基本崩溃。O38 support 更大，S0/S1 稳定性更好一些，但 own AP 低。

分析:

- 历史“超过 Stream3D”的现象主要来自 own-support 裁剪差异，而不是对象形成机制真的优于 Stream3D。
- `P0 Stream3D on S1` AP=0.399213 说明稀疏 support 本身能显著改变指标，这正是 v10 必须强制 cross-support 的原因。
- 后续方法必须同时提高 own 和 same-support，否则不能宣称解决问题。

## Fresh D4RT Carrier

输出:

- `Stream3D/outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67/summary.json`
- `Stream3D/outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67/summary.md`

关键指标:

| metric | value |
|---|---:|
| num_windows | 5 |
| num_ok_windows | 5 |
| num_failed_windows | 0 |
| uv_in01_rate_mean | 0.9858451843261719 |
| track_length_visible_mean_mean | 13.284716796875 |
| self_uv_error_p90_mean | 1.5708253622055055 |
| cycle_uv_error_p90_mean | 3.2737268686294554 |
| surfel_coverage_2d_per_frame_mean_mean | 0.13198280334472656 |

每 scene:

| scene | uv in01 | visible len mean | self p90 px | cycle p90 px | coverage |
|---|---:|---:|---:|---:|---:|
| scene0050_00 | 0.990829 | 15.7715 | 1.38513 | 2.31369 | 0.161015 |
| scene0011_00 | 0.959251 | 9.6402 | 1.55750 | 4.13698 | 0.133202 |
| scene0030_00 | 0.999809 | 15.9829 | 1.61023 | 2.81068 | 0.130849 |
| scene0081_01 | 0.982063 | 11.5381 | 1.65579 | 3.89239 | 0.134019 |
| scene0591_00 | 0.997272 | 13.4908 | 1.64548 | 3.21488 | 0.100830 |

GPU/缓存结论:

- 该 carrier 是 v10 fresh run，不是 v8 cache。
- 用户指出缓存风险后，R3/R4/repair/geometry 的 `DEBUG_ROOT` 已改到 fresh v10 root。
- D4RT image-space tracking 本身质量并不差，uv/cycle 指标可用；后面失败主要发生在 object primitive、support、mesh geometry materialization 和 cross-support 对齐层面。

## Regionlet Birth R0-R4

输出: `Stream3D/outputs/audit/v10_regionlet/regionlet_matrix_probe5.json`

| row | AP | AP50 | AP25 | pre% | conflict | best IoU |
|---|---:|---:|---:|---:|---:|---:|
| P0 Stream3D on R0 | 0.353928 | 0.505814 | 0.678428 | 4.7467 | 0.2213 | 0.726528 |
| R0 full-mask own | 0.136117 | 0.316625 | 0.426390 | 4.7467 | 70.2060 | 0.635427 |
| R0 full-mask on S0 | 0.000509 | 0.004505 | 0.012763 | 84.6744 | 70.2060 | 0.035717 |
| R0 full-mask on S1 | 0.020240 | 0.059677 | 0.124805 | 4.5145 | 70.2060 | 0.187240 |
| R1 mask-core own | 0.172123 | 0.291682 | 0.421048 | 4.0122 | 67.4841 | 0.696796 |
| R1 mask-core on S0 | 0.001003 | 0.004754 | 0.009630 | 84.6744 | 67.4841 | 0.032378 |
| R1 mask-core on S1 | 0.015567 | 0.033303 | 0.084522 | 4.5145 | 67.4841 | 0.176701 |
| R2 depth-split own | 0.000168 | 0.001392 | 0.121278 | 4.7986 | 74.0562 | 0.320640 |
| R3 D4RT-seeded own | 0.142857 | 0.256714 | 0.358752 | 1.5783 | 48.9373 | 0.620748 |
| R4 combined own | 0.014373 | 0.059587 | 0.168899 | 2.3179 | 58.0068 | 0.400221 |
| R4 combined on S0 | 0.000000 | 0.000000 | 0.000457 | 84.6744 | 58.0068 | 0.010128 |
| R4 combined on S1 | 0.000288 | 0.002591 | 0.006401 | 4.5145 | 58.0068 | 0.057090 |

证据链:

- reportable scan: `num_configs=5`，`num_reportable_method_configs=5`，`num_suspicious_configs=0`。
- R0/R1 own-support 有一定 AP，但 cross-support 到 S0/S1 几乎归零。
- R2/R4 过度切分明显，AP 退化。

分析:

- Regionlet 的局部切分没有形成稳定 object primitive。
- conflict 很高，说明点归属重叠严重。
- D4RT seed 降低了 support 面积和部分 conflict，但没有解决 object formation。

## Regionlet Repair R0b/R1b/R4b

输出: `Stream3D/outputs/audit/v10_regionlet_repair/regionlet_repair_matrix_probe5.json`

| row | AP | AP50 | AP25 | pre% | conflict | best IoU |
|---|---:|---:|---:|---:|---:|---:|
| P0 Stream3D on R0b | 0.338713 | 0.538104 | 0.685444 | 22.9309 | 0.2213 | 0.685119 |
| R0b full-mask 32f WTA own | 0.021122 | 0.098676 | 0.331286 | 22.9309 | 0.0000 | 0.447109 |
| R0b on S0 | 0.000429 | 0.003032 | 0.053284 | 84.6744 | 0.0000 | 0.141810 |
| R0b on S1 | 0.060831 | 0.155245 | 0.402146 | 4.5145 | 0.0000 | 0.548491 |
| R1b mask-core 32f WTA own | 0.033785 | 0.127959 | 0.395010 | 20.6836 | 0.0000 | 0.521286 |
| R1b on S0 | 0.000288 | 0.002181 | 0.054594 | 84.6744 | 0.0000 | 0.139745 |
| R1b on S1 | 0.064218 | 0.169101 | 0.415508 | 4.5145 | 0.0000 | 0.570806 |
| R4b combined 32f WTA own | 0.000262 | 0.001621 | 0.104822 | 21.4094 | 0.0000 | 0.343440 |
| R4b on S0 | 0.000000 | 0.000000 | 0.004463 | 84.6744 | 0.0000 | 0.093709 |
| R4b on S1 | 0.003630 | 0.017324 | 0.174300 | 4.5145 | 0.0000 | 0.377634 |

Repair 诊断:

| config | regions/mask | objects/scene | pre points/scene | raw conflict | after WTA | removed |
|---|---:|---:|---:|---:|---:|---:|
| R0b | 1.7967 | 222.8 | 47723.2 | 0.8344 | 0.0000 | 0.8314 |
| R1b | 1.7647 | 216.8 | 43128.8 | 0.8058 | 0.0000 | 0.8112 |
| R4b | 6.0692 | 395.2 | 44622.8 | 0.8479 | 0.0000 | 0.8444 |

分析:

- WTA 成功把 conflict 清零，但 AP 没有提升到可接受水平。
- 这说明主要问题不只是 overlap，而是 regionlet object formation 本身: object 过碎、support 与语义实例的对齐不足。
- Repair 后 `P0 Stream3D on R0b/R1b/R4b` 仍明显强于 R 方法 own，说明这些 support 对 Stream3D 有用，但 R 方法自身的预测/对象组织不够。

## D4RT Geometry G1-G5

输出:

- `Stream3D/outputs/audit/v10_d4rt_geometry/d4rt_geometry_matrix_probe5.json`
- `Stream3D/outputs/v10_d4rt_geometry/*_summary.md`
- `Stream3D/outputs/audit/v10_d4rt_geometry/reportable_config_scan_d4rt_geometry_probe5.json`

Matrix:

| row | AP | AP50 | AP25 | pre% | conflict | best IoU | method table | manifest pass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| G0 Stream3D RGB-D own | 0.235730 | 0.414306 | 0.537786 | 84.6744 | 0.2213 | 0.630750 | False | True |
| G6 D4RT image-space evidence with RGB-D geometry | 0.328439 | 0.629266 | 0.884363 | 3.9861 | 8.4307 | 0.691363 | True | True |
| G1 D4RT raw own | NA | NA | NA | 0.0090 | 14.1772 | NA | False | True |
| P0 Stream3D on G1 support | NA | NA | NA | 0.0090 | 0.2213 | NA | False | True |
| G1 on S0 | 0.000000 | 0.000000 | 0.000000 | 84.6744 | 14.1772 | 0.000000 | False | True |
| G1 on S1 | 0.000000 | 0.000000 | 0.000000 | 4.5145 | 14.1772 | 0.000000 | False | True |
| G2 scene Sim3 own | 0.188825 | 0.364384 | 0.486341 | 1.4370 | 75.1744 | 0.633245 | False | True |
| P0 Stream3D on G2 support | 0.314343 | 0.413636 | 0.552727 | 1.4370 | 0.2213 | 0.668357 | False | True |
| G2 on S0 | 0.000000 | 0.000000 | 0.000112 | 84.6744 | 75.1744 | 0.007395 | False | True |
| G2 on S1 | 0.000000 | 0.000000 | 0.004523 | 4.5145 | 75.1744 | 0.034475 | False | True |
| G3 window Sim3 own | 0.173962 | 0.334164 | 0.461727 | 1.4386 | 74.1746 | 0.625980 | False | True |
| G3 on S0 | 0.000000 | 0.000000 | 0.000112 | 84.6744 | 74.1746 | 0.007512 | False | True |
| G3 on S1 | 0.000000 | 0.000000 | 0.004523 | 4.5145 | 74.1746 | 0.034818 | False | True |
| G4 scene Sim3 density own | 0.188825 | 0.364384 | 0.486341 | 1.4370 | 75.1744 | 0.633245 | False | True |
| G4 on S0 | 0.000000 | 0.000000 | 0.000112 | 84.6744 | 75.1744 | 0.007395 | False | True |
| G4 on S1 | 0.000000 | 0.000000 | 0.004523 | 4.5145 | 75.1744 | 0.034475 | False | True |
| G5 window Sim3 density own | 0.173962 | 0.334164 | 0.461727 | 1.4386 | 74.1746 | 0.625980 | False | True |
| G5 on S0 | 0.000000 | 0.000000 | 0.000112 | 84.6744 | 74.1746 | 0.007512 | False | True |
| G5 on S1 | 0.000000 | 0.000000 | 0.004523 | 4.5145 | 74.1746 | 0.034818 | False | True |

Geometry export evidence:

| config | example evidence |
|---|---|
| G1 raw | 4 scenes had 0 projected masks; scene0591_00 had 12 objects, 79 points. Own AP stayed NA. |
| G2 scene Sim3 | 7200 anchors/scene; p90 residual range 0.0577556 to 0.287136; objects per scene 15 to 92. |
| G3 window Sim3 | Same anchor/residual pattern as G2; objects per scene 15 to 90. |
| G4/G5 density | Same AP as G2/G3 respectively in this probe5 run. |

Reportable scan:

```text
num_configs=5
num_diagnostic_only_configs=5
num_reportable_method_configs=0
num_suspicious_configs=0
num_uses_gt_for_prediction=0
```

分析:

- G2/G4 own AP=0.188825，G3/G5 own AP=0.173962，低于 G0 Stream3D RGB-D own AP=0.235730。
- G2/G4/G3/G5 在 S0/S1 cross-support 基本为 0，说明其 object/support 与 Stream3D support universe 不兼容。
- G1 raw 无有效 AP，原因是 raw D4RT 坐标没有直接落到 ScanNet evaluator mesh 的有效 support 上。
- Sim3 能把 D4RT 点映射到 mesh，但它使用 diagnostic anchors，因此不能当作训练自由 method result。
- D4RT carrier 的 2D tracking 指标可用，不等于 3D object formation 可用。失败点是从 correspondence 到 instance primitive 的转换。

## Gate 判断

| lane | gate | result |
|---|---|---|
| Phase 0 protocol | manifest/integrity clean | pass |
| Regionlet R0-R4 | own 与 cross-support 同时可竞争 | fail |
| Regionlet repair | conflict 降低且 AP/cross 改善 | fail |
| D4RT geometry | 形成可报告 method result 且 same-support 不崩 | fail |
| tune30/full 扩展 | probe5 gate 通过后才启动 | not started |

## 主要 insight

1. own-support 高分不是充分证据。B1/O1 在 own-support 分数较高，但 S0/S1 cross-support 基本崩溃。
2. support 不是越小越好，也不是越大越好。S1 sparse support 能让 Stream3D AP 提高到 0.399213，但同一 support 对新方法不一定有效。
3. Regionlet 的第一问题不是 overlap 本身。WTA 把 conflict 清零后，R0b/R1b/R4b AP 仍低，说明 object primitive 过碎或边界不可靠。
4. D4RT image-space evidence 有价值。fresh carrier 的 uv/cycle 指标可用，但它需要新的 3D object formation，而不是简单 backproject 或 Sim3 materialize。
5. G2/G4 的 diagnostic own AP 接近但未超过 G0，并且不能进 method table。任何后续 claim 必须绕开 diagnostic anchor，改为纯预测路径。

## 结论

Stream4D v10 在 probe5 上没有获得新的可报告 SOTA 或超过 Stream3D 的结果。最可靠结论是: 继续堆 top-k、NMS、WTA、support completion 或简单 D4RT backprojection 不能解决核心问题。下一步如果继续，应直接改 object formation 机制，例如从 D4RT tracks 构建稳定 3D object proposal，再在同一 support 下和 Stream3D 比较，而不是再对现有 mask 做后处理。

## 审计材料

审计包:

- `stream4d_v10_code_audit_packet_20260609_1134_probe5_final.zip`
- SHA256: 见 `stream4d_v10_code_audit_packet_20260609_1134_probe5_final.sha256`
- filelist: `stream4d_v10_code_audit_packet_20260609_1134_probe5_final_filelist.txt`
- git diff: `stream4d_v10_code_audit_packet_20260609_1134_probe5_final_git_diff.patch`
- zip test: `stream4d_v10_code_audit_packet_20260609_1134_probe5_final_ziptest.log`
