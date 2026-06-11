# Stream4D v11 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/stream4d_v11_deep_algorithmic_reboot_plan_for_codex.md`  
执行日志: `docs/stream4d_v11_执行日志.md`  
结论先行: v11 没有获得可作为正式 method table 的改进结果。Phase 1 说明 tiny clean candidate 有 oracle headroom，但 broad union candidate 不足以支撑继续调 selection；Phase 2 说明 D4RT propagation 增加 temporal observations，但与 shuffle 接近；Phase 3/4 posterior proxy 没有证明 real D4RT 优于 shuffle/no-track；Phase 6 D4RT geometry adapter 给出可审计归因，但仍是 diagnostic-only，且 cross-support 崩溃。

## 执行完整性自查

- v11 计划真正的“本质改变”是: 不再把 mask/regionlet/carrier component 当 object，而是建立 measurement bank + latent object slot / ownership posterior，再导出 core/fringe/reject。
- 本轮没有完成这个完整新算法。没有产出 `outputs/v11_measurement_bank/...`，没有实现真正的 latent object slot posterior，也没有完成 Phase 5 core/fringe/reject export。
- 本轮实际完成的是 v11 诊断闭环和代理实验: candidate oracle、measurement density、failure attribution、S2-S5 posterior proxy、D4RT geometry protocol audit。
- S2-S5 被称为 posterior proxy，不是完整 v11 object-slot method。它复用了既有 `tools.export_v9_b1_controls` 风格的 maskcount/area/no-track/shuffle 控制路径，用来验证 real D4RT 是否明显优于 control；结果没有通过 gate。
- v8/v9/v10 的 B1/O1/O38/regionlet/repair 结果只作为 candidate pool、control baseline、failure attribution 输入和对照，不算 v11 新方法结果。
- 因此不能把本轮说成“执行并验证了 v11 新算法核心”。更准确的说法是: v11 核心算法尚未实现；本轮先做了必要诊断，结果显示当前旧 primitive/control proxy 不足以支撑继续扩展。

## 结果边界

- 所有 AP/AP50/AP25 来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt`，由 matrix 工具解析。
- Candidate oracle 和 GT failure attribution 使用 GT，只能作为 diagnostic upper bound / attribution，不能进入 method table。
- 严格协议边界: D4RT 在生成 prediction/TMP 前只能做 D4RT-internal self-alignment；任何对 GT/reference scene geometry 的对齐只能发生在评估/测指标阶段。G2-G5 使用 ScanNet RGB-D depth/pose world points 在 diagnostic materialization/export 阶段做 Sim3，因此不满足该严格协议。它们保留为审计中的 anchor-assisted 反例，不进入 method table，也不作为有效 geometry gate 证据。
- S2-S5 posterior proxy 是本轮唯一被 reportable scan 计为 method result 的 v11 配置，但没有通过 S0/S1、same-support gap 或 D4RT real-vs-shuffle gate。
- S2-S5 不是完整 v11 latent object-slot posterior；它们只是用于 gate 判断的轻量 proxy/control。不能拿它们冒充 v11 核心方法。
- 没有把 `NA`、`nan` 或空预测改成有效数字。G1 raw own AP 保持 NA。

## Phase 1 Candidate Upper Bound

输出:

- `Stream3D/outputs/audit/v11_candidate_oracle/candidate_oracle_matrix_probe5.json`
- `Stream3D/outputs/audit/v11_candidate_oracle/candidate_oracle_matrix_probe5.csv`
- `Stream3D/outputs/audit/v11_candidate_oracle/candidate_oracle_matrix_probe5.md`

| pool | AP | AP50 | AP25 | pre% | union% | GT crop/full | best IoU mean | GT>=.25 | GT>=.50 | method table |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C0 Stream3D oracle pool | 0.390788 | 0.628205 | 0.760684 | 83.7810 | 62.1779 | 40.6/40.6 | 0.632363 | 163 | 135 | False |
| C1 B1+O1 tiny clean | 0.476608 | 0.763158 | 0.918763 | 3.8660 | 3.6139 | 8.0/40.6 | 0.697902 | 25 | 21 | False |
| C2 O38 large memory | 0.223210 | 0.444444 | 0.635556 | 66.5926 | 51.2985 | 40.2/40.6 | 0.457088 | 127 | 93 | False |
| C3 v10 regionlet birth | 0.396825 | 0.809524 | 0.904762 | 4.7191 | 2.8149 | 7.8/40.6 | 0.631296 | 28 | 24 | False |
| C4 v10 regionlet repair | 0.218145 | 0.559633 | 0.834862 | 21.5143 | 7.1375 | 19.6/40.6 | 0.525520 | 79 | 54 | False |
| C5 C1+C2+C3 union | 0.232232 | 0.459459 | 0.666667 | 57.7528 | 44.2017 | 40.0/40.6 | 0.464317 | 131 | 94 | False |

证据链:

- 所有 oracle configs manifest 均为 `uses_gt_for_prediction=true`、`uses_gt_for_diagnostic=true`、`is_diagnostic_only=true`、`is_method_result=false`。
- `evaluate.py` 只在 `--allow-oracle-eval` 且 diagnostic-only 非 method 时允许 oracle evaluation。
- C1/C3 oracle AP50/AP25 很高，但 GT crop 只有 8.0/40.6 和 7.8/40.6。
- C5 broad union 覆盖到 40.0/40.6 GT crop，但 AP=0.232232，AP50=0.459459。

分析:

- tiny clean subset 的高 oracle 说明候选空间中有局部好对象，这解释了 v8-v10 的 own-support 高分。
- broad union 不是强上界: AP 与 v10 G0 Stream3D AP=0.235730 基本持平，AP50 只比 Stream3D AP50=0.414306 高 0.045153，未达到计划要求的 +0.05 headroom。
- 因此继续做 top-k、score、NMS 或 selection sweep 不符合 v11 gate。

## Phase 2 Measurement Density

输出:

- `Stream3D/outputs/audit/v11_measurement_density/measurement_density_probe5.json`
- `Stream3D/outputs/audit/v11_measurement_density/measurement_density_probe5.csv`
- `Stream3D/outputs/audit/v11_measurement_density/measurement_density_probe5.md`

| mode | ok rows | mask density | surfel obs rate | obs/surfel | visible unobserved | self consistency | contradiction |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 CropFormer available frames | 5/5 | 0.125000 | 0.854236 | 1.615100 | 0.878835 | 0.982253 | 0.017747 |
| M1 framewise no propagation | 5/5 | 0.125000 | 0.854236 | 1.615100 | 0.878835 | 0.982253 | 0.017747 |
| M2 D4RT source propagation | 5/5 | 0.125000 | 0.122559 | 1.632495 | 0.877120 | 0.992518 | 0.007482 |
| M3 frozen video masklet | 0/5 | NA | NA | NA | NA | NA | NA |
| M4 no-track source only | 5/5 | 0.125000 | 0.122559 | 0.122559 | 0.877441 | 0.000000 | 0.000000 |
| M5 shuffled propagation | 5/5 | 0.125000 | 0.122522 | 1.626343 | 0.877726 | 0.981484 | 0.018516 |
| M5b shuffled uv target control | 5/5 | 0.125000 | 0.858264 | 1.587817 | 0.881308 | 0.980186 | 0.019814 |

证据链:

- 该 diagnostic 不读取 GT，不报告 AP。
- 当前 workspace 没有 denser framewise mask cache，所以 M0 与 M1 相同。
- 当前 workspace 没有 frozen video masklet cache，所以 M3 记录为 `not_available`。
- 5 个 probe scenes 均只有 2/16 mask frames，mask_frame_density=0.125。

分析:

- D4RT propagation 相比 no-track M4 将 obs/surfel 从 0.122559 提高到 1.632495，说明 temporal propagation 确实增加 observation 数量。
- 但 M2 与 M5 shuffle 的 obs/surfel 很接近，且 visible_unobserved 均约 0.877，说明密度提升本身没有证明 object identity 被正确利用。
- 这支持 v11 计划中的判断: 当前问题不仅是 measurement 数量，还是 ownership posterior/object primitive。

## Phase 3/4 Posterior Proxy

输出:

- `Stream3D/outputs/audit/v11_posterior_proxy/posterior_proxy_matrix_probe5.json`
- `Stream3D/outputs/audit/v11_posterior_proxy/posterior_proxy_matrix_probe5.csv`
- `Stream3D/outputs/audit/v11_posterior_proxy/posterior_proxy_matrix_probe5.md`
- `Stream3D/outputs/audit/v11_posterior_proxy/reportable_config_scan_v11_posterior_proxy_probe5.json`

执行边界:

- 这里执行的是 posterior proxy/control，不是 v11 计划中的完整 measurement bank + latent object slot ownership posterior。
- 该 proxy 用于回答一个较小问题: real D4RT maskcount signal 是否明显强于 area/shuffle/no-track controls。
- 因为 proxy 没有证明 real D4RT 明显优于 controls，所以没有继续实现 Phase 5 core/fringe/reject，也没有启动 tune30/final。

| row | AP | AP50 | AP25 | pre% | conflict | best IoU | same-support P0 AP | gap AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 area same-count own | 0.144503 | 0.305339 | 0.536614 | 4.2354 | 68.8770 | 0.600767 | 0.372722 | -0.228220 |
| P0 on S2 support | 0.372722 | 0.511111 | 0.727160 | 4.2354 | 0.2213 | 0.727700 | 0.372722 | 0.000000 |
| S3 real D4RT maskcount own | 0.156653 | 0.313452 | 0.552663 | 4.1903 | 68.5051 | 0.625784 | 0.373232 | -0.216580 |
| P0 on S3 support | 0.373232 | 0.500325 | 0.720779 | 4.1903 | 0.2213 | 0.731692 | 0.373232 | 0.000000 |
| S3 real D4RT on S0 | 0.000373 | 0.002503 | 0.006594 | 84.6744 | 68.5051 | 0.029091 | NA | NA |
| S3 real D4RT on S1 | 0.009604 | 0.024097 | 0.086757 | 4.5145 | 68.5051 | 0.162611 | NA | NA |
| S4 shuffled D4RT own | 0.147827 | 0.301142 | 0.534694 | 4.1948 | 69.4148 | 0.603247 | 0.367316 | -0.219489 |
| S5 no-track own | 0.146705 | 0.305507 | 0.536841 | 4.2371 | 68.5676 | 0.603649 | 0.372722 | -0.226017 |

D4RT contribution:

| comparison | delta AP | delta AP50 | delta AP25 | gate |
|---|---:|---:|---:|---|
| S3 real - S4 shuffle | +0.008826 | +0.012310 | +0.017969 | fail |
| S3 real - S5 no-track | +0.009947 | +0.007946 | +0.015821 | fail |

Reportable scan:

```text
num_configs=4
num_reportable_method_configs=4
num_diagnostic_only_configs=0
num_suspicious_configs=0
num_uses_gt_for_prediction=0
```

Metric integrity:

```text
phase0_pass=True
num_reportable_method_configs=4
num_suspicious_configs=0
num_uses_gt_for_prediction=0
```

分析:

- S3 real D4RT 是本轮最好的 posterior proxy，但只比 shuffle/no-track 高不到 0.02 AP25，未达到计划要求的 +0.05 AP50/AP25。
- S3 on S0/S1 基本崩溃，低于 S0 Gate 和 S1 Gate。
- 同一 support 上 Stream3D 明显强于 S3 own，说明 S3 的 support 对 Stream3D 有用，但 S3 自身 object assignment 仍差。

## GT Failure Attribution

输出:

- `Stream3D/outputs/audit/v11_failure_attribution/b1_vs_c5.md`
- `Stream3D/outputs/audit/v11_failure_attribution/o38_vs_c5.md`
- `Stream3D/outputs/audit/v11_failure_attribution/r1b_vs_c4.md`
- `Stream3D/outputs/audit/v11_failure_attribution/s3_vs_c5.md`

| case | num GT | mean pool best IoU | mean method best IoU | no candidate | filtered candidate | boundary bad | wrong/frag | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 vs C5 | 236 | 0.306059 | 0.026017 | 103 (0.436441) | 126 (0.533898) | 2 (0.008475) | 4 (0.016949) | 1 (0.004237) |
| O38 vs C5 | 236 | 0.306059 | 0.300022 | 103 (0.436441) | 3 (0.012712) | 73 (0.309322) | 0 | 57 (0.241525) |
| R1b vs C4 | 236 | 0.103764 | 0.097447 | 195 (0.826271) | 2 (0.008475) | 36 (0.152542) | 1 (0.004237) | 2 (0.008475) |
| S3 vs C5 | 236 | 0.306059 | 0.023775 | 103 (0.436441) | 125 (0.529661) | 3 (0.012712) | 4 (0.016949) | 1 (0.004237) |

证据链:

- 该工具显式使用 GT，只做 diagnostic attribution，不产生 method result。
- B1/S3 用 C5 pool 做参照，R1b 用 C4 repair pool 做参照。
- failure class 由 pool best IoU 与 method best IoU 的阈值关系确定。

分析:

- B1/S3 的 filtered_candidate 占比超过 52%，说明过筛是一个真实问题。
- O38 的 filtered_candidate 很少，但 boundary_bad 达到 30.9%，matched 也只有 24.2%，说明大 support 下主要是 object boundary/primitive 质量问题。
- R1b 的 no_candidate 达 82.6%，说明 v10 repair candidate pool 本身无法表达多数 GT object。
- 归因符合用户补充判断: 不是单纯 overfilter，也不是单纯 coverage，而是 mask-to-object ownership/matching 层面的问题。

## Phase 6 D4RT Geometry Adapter

输出:

- `Stream3D/outputs/audit/v11_d4rt_geometry/stream4d_v11_adapter_scene_sim3_probe5_summary.md`
- `Stream3D/outputs/audit/v11_d4rt_geometry/d4rt_geometry_matrix_probe5.json`
- `Stream3D/outputs/audit/v11_d4rt_geometry/reportable_config_scan_d4rt_geometry_probe5.json`

严格协议自查:

- 允许: D4RT 自己和自己对齐，例如 window 内/跨 window 的 D4RT-internal coordinate normalization 或 self-consistency 对齐。
- 允许: prediction/TMP 已冻结后，在评估/测指标阶段为了和 GT/reference mesh 计算误差而做 post-hoc alignment。
- 不允许: 在生成 prediction/TMP artifact 前，用 GT/reference scene geometry 对 D4RT 做 Sim3，然后基于 aligned D4RT 生成预测。
- 自查结果: G2-G5 没有使用 instance/semantic GT label，但使用了 ScanNet RGB-D depth/pose backprojection world points，并且 Sim3 发生在 diagnostic materialization/export 阶段。因此按严格协议，G2-G5 是 anchor-assisted invalid artifact，只能保留为审计反例，不能作为有效 D4RT geometry replacement 证据。
- S2-S5 posterior proxy 没有使用该 Sim3，是本轮唯一被 reportable scan 计为 method result 的 v11 配置。

Adapter scene Sim3 geometry evidence:

| scene | anchors | median residual | p90 residual | spacing q50 | frames | empty mask mean |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 7200 | 0.0370485 | 0.0761211 | 0.00476558 | 16 | 0 |
| scene0011_00 | 7200 | 0.0324988 | 0.0577556 | 0.0140756 | 16 | 0.03125 |
| scene0030_00 | 7200 | 0.103001 | 0.152568 | 0.0042935 | 16 | 0.0555556 |
| scene0081_01 | 7200 | 0.0386918 | 0.101404 | 0.0114319 | 16 | 0.0384615 |
| scene0591_00 | 7200 | 0.181572 | 0.287136 | 0.00409783 | 16 | 0.0398309 |

D4RT geometry matrix:

| row | AP | AP50 | AP25 | pre% | conflict | best IoU | method table |
|---|---:|---:|---:|---:|---:|---:|---|
| G0 Stream3D RGB-D own | 0.235730 | 0.414306 | 0.537786 | 84.6744 | 0.2213 | 0.630750 | False |
| G1 D4RT raw own | NA | NA | NA | 0.0090 | 14.1772 | NA | False |
| P0 on G1 support | NA | NA | NA | 0.0090 | 0.2213 | NA | False |
| G1 on S0 | 0.000000 | 0.000000 | 0.000000 | 84.6744 | 14.1772 | 0.000000 | False |
| G1 on S1 | 0.000000 | 0.000000 | 0.000000 | 4.5145 | 14.1772 | 0.000000 | False |
| G2 scene Sim3 own | 0.188825 | 0.364384 | 0.486341 | 1.4370 | 75.1744 | 0.633245 | False |
| P0 on G2 support | 0.314343 | 0.413636 | 0.552727 | 1.4370 | 0.2213 | 0.668357 | False |
| G2 on S0 | 0.000000 | 0.000000 | 0.000112 | 84.6744 | 75.1744 | 0.007395 | False |
| G2 on S1 | 0.000000 | 0.000000 | 0.004523 | 4.5145 | 75.1744 | 0.034475 | False |
| G3 window Sim3 own | 0.173962 | 0.334164 | 0.461727 | 1.4386 | 74.1746 | 0.625980 | False |
| P0 on G3 support | 0.311594 | 0.433110 | 0.568562 | 1.4386 | 0.2213 | 0.670186 | False |
| G3 on S0 | 0.000000 | 0.000000 | 0.000112 | 84.6744 | 74.1746 | 0.007512 | False |
| G3 on S1 | 0.000000 | 0.000000 | 0.004523 | 4.5145 | 74.1746 | 0.034818 | False |
| G4 scene Sim3 density own | 0.188825 | 0.364384 | 0.486341 | 1.4370 | 75.1744 | 0.633245 | False |
| G5 window Sim3 density own | 0.173962 | 0.334164 | 0.461727 | 1.4386 | 74.1746 | 0.625980 | False |

Reportable scan:

```text
num_configs=5
num_reportable_method_configs=0
num_diagnostic_only_configs=5
num_suspicious_configs=0
num_uses_gt_for_prediction=0
```

分析:

- Scene Sim3 residual 多数 scene 低于 median 0.15m、p90 0.35m，scene0591 median=0.181572 高于 median threshold 但 p90=0.287136 仍低于 0.35m。
- G2/G4 own AP=0.188825，低于 G0 Stream3D RGB-D own AP=0.235730；AP50 drop=0.049922，接近计划阈值 0.05；AP drop=0.046905，高于计划允许 drop 0.03。
- G2/G3 在 S0/S1 基本归零，说明当前 materialized D4RT support 不是完整 Stream3D geometry replacement。
- 更重要的是，G2-G5 在生成 artifact 前已经用 RGB-D depth/pose 做 Sim3 anchor alignment，所以在严格协议下不能作为有效 geometry replacement 实验。它们只能说明: 即便采用这种不允许的 anchor-assisted materialization，AP/cross-support 仍没有过关。
- 真正符合严格协议的 no-external-alignment raw lane 是 G1；G1 own AP 为 NA，G1 on S0/S1 为 0，说明 raw D4RT geometry 当前不能直接落到 ScanNet evaluator support。

## Gate 判断

| lane | gate | result | evidence |
|---|---|---|---|
| Candidate oracle | broad pool AP50 >= Stream3D AP50 + 0.05 | fail | C5 AP50=0.459459，比 Stream3D 0.414306 高 0.045153 |
| Measurement density | real D4RT clearly beats shuffle/no-track | partial/fail | M2 beats no-track on obs/surfel，但与 M5 shuffle 接近 |
| Posterior proxy | S3 > S4 by +0.05 AP50/AP25 | fail | +0.012310 AP50, +0.017969 AP25 |
| S0 Gate | AP>=0.08, AP50>=0.18, AP25>=0.45 | fail | S3 on S0 = 0.000373/0.002503/0.006594 |
| S1 Gate | AP>=0.18, AP50>=0.35, AP25>=0.60 | fail | S3 on S1 = 0.009604/0.024097/0.086757 |
| Same-support Gap | P0 on M support - M own AP < 0.08 | fail | P0 on S3 AP 0.373232 vs S3 own 0.156653 |
| D4RT geometry replacement | D4RT 预测前不得和 GT/reference geometry 对齐；AP drop <=0.03 且 cross-support stable | fail/invalid | G1 raw 合规但无有效 AP；G2-G5 在 materialization/export 阶段用了 RGB-D depth/pose Sim3，严格协议下剔除 |
| tune30/final | probe5 gate passed | not started | gates failed |

## 主要 Insight

1. Tiny-support oracle 上界确实高，但 broad support 上界不够强。C1/C3 解释了 own-support 高分，C5 解释了为什么扩大 support 后仍难超过 Stream3D。
2. D4RT correspondence 能增加 temporal observations，但当前 semantic mask 稀疏且 shuffle control 接近 real D4RT。现有使用方式没有证明 object identity 被正确提取。
3. Posterior proxy 没有关闭 same-support gap。同一 S3 support 上 Stream3D AP=0.373232，而 S3 own AP=0.156653，说明 support 不是废的，object assignment 才是主要问题。
4. Failure attribution 同时支持两个问题: B1/S3 是 filtered_candidate 主导，O38 是 boundary_bad/no_candidate 主导。根因不是单一阈值，而是 mask-to-object ownership/matching。
5. D4RT geometry lane 需要重述: G1 raw 是合规 no-external-alignment lane，但无有效 AP；G2-G5 虽然没有用 instance/semantic GT label，却在 prediction/TMP 生成前用了 RGB-D depth/pose Sim3，因此严格协议下只能作为无效 anchor-assisted 反例。当前不能 claim metric geometry replacement。

## 结论

Stream4D v11 在 probe5 上没有获得新的可报告 SOTA 或超过 Stream3D 的结果。更重要的是，本轮没有真正实现 v11 计划中最核心的 measurement bank + latent object slot posterior + core/fringe/reject 方法；只完成了诊断、归因和 proxy/control 实验。S2-S5 是 method-result 配置且 metric integrity 通过，但它们只是 posterior proxy，不是完整 v11 新算法，并且 AP、cross-support、same-support gap 和 D4RT contribution gates 全部失败。Oracle 和 failure attribution 使用 GT，只能作为 diagnostic-only。D4RT geometry 中 G1 raw 合规但失败；G2-G5 在 prediction/TMP 生成前使用 RGB-D depth/pose Sim3，严格协议下是 invalid anchor-assisted artifact，不能进入 method table，也不能作为有效 geometry gate 证据。

最可靠结论是: 现有 training-free 2D mask measurements + D4RT surfel evidence，在当前 primitive design 下不足以稳定恢复 ScanNet 静态实例分割。下一步不应继续 top-k/NMS/WTA/score sweep，而应引入更强的 frozen video masklet prior、更密集 semantic observations，或重新设计 latent object slot ownership posterior；否则项目主张应转向 D4RT structurally 更有优势的 dynamic correspondence，而不是声称替代 RGB-D static segmentation。

## 本轮修复/实现是否合理的审计点

- `evaluate.py` 的 oracle 允许逻辑只放开 diagnostic-only 非 method oracle，不降低正式方法安全性。
- `v11_candidate_pool_oracle.py` 明确写入 GT 使用 manifest，结果只作为 upper bound。
- `v11_measurement_density_diagnostic.py` 不读 GT，不报告 AP；M3 缺失按 `not_available` 处理。
- `v11_gt_failure_attribution.py` 使用 GT 但只做 diagnostic attribution。
- `d4rt_stream3d_geometry_adapter.py` 明确写出 `is_complete_stream3d_replacement=false`；复盘进一步修正为: 若在 prediction/TMP 生成前使用 RGB-D depth/pose Sim3，则严格协议下不能作为有效 D4RT geometry replacement 结果。
- 本轮没有实现完整 v11 latent object-slot method；复盘已把 S2-S5 标成 proxy/control，避免拿旧 primitive 的代理结果冒充 v11 新方法。
- S2-S5 method configs 通过 reportable scan 和 metric integrity，`num_uses_gt_for_prediction=0`。

## 审计材料

审计包:

- `stream4d_v11_final_code_review_packet.zip`
- SHA256: 见 `stream4d_v11_final_code_review_packet.sha256`
- filelist: `stream4d_v11_final_code_review_packet_filelist.txt`
- git diff: `stream4d_v11_final_code_review_packet_git_diff.patch`
- git status: `stream4d_v11_final_code_review_packet_git_status.txt`
- zip test: `stream4d_v11_final_code_review_packet_ziptest.log`

干净解包验证已通过: `py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py` 通过，`unittest discover tests` 为 15 tests OK，v11 S2-S5 method scan clean，metric integrity `phase0_pass=True`。
