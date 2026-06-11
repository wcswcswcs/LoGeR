# Stream4D v15 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/stream4d_v15_executable_global_measurement_explanation_plan_for_codex.md`  
执行日志: `docs/stream4d_v15_执行日志.md`  
结论先行: v15 完成了 manifest/audit 修复、multi-atom union oracle、mask-region measurement candidate、region split/materialization blocker 尝试、region union oracle 和 D4RT Sim3 geometry diagnostic。没有得到可作为正式 method success 的结果；本轮所有 v15 prediction configs 都是 diagnostic-only。好消息是: atom-as-measurement 的 multi-atom upper bound 明显强于 v14 atom-as-object，A3/A4 target K=8 oracle 达到 `0.297386 / 0.529412 / 0.764706`，说明 object-level set packing 方向确实有信号。坏消息是: support 仍只有 `3-5%`，Phase 1/2 broad-support gate 明确失败，因此 Phase 3/4 solver、Phase 5 method materialization、tune30/final 均未启动。

## 结果边界

- 所有 AP/AP50/AP25 来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 或 v15 summary JSON。
- Oracle 使用 GT selection，只是 diagnostic upper bound，不能进入 method table。
- Mask-region candidates 不读 GT 生成 prediction，但仍是 measurement candidate diagnostic，不是 method result。
- GT diagnostic 只用于 region purity/completeness、union oracle selection 和 evaluator。
- D4RT Sim3 alignment 只用于 Phase 6 diagnostic，不用于 prediction grouping/filtering/score。
- 没有把 `NA`、空预测、oracle 结果或 diagnostic candidate 改写为 method 成功。

## 审计通过

```text
py_compile: pass
unittest all: Ran 30 tests in 1.570s ... OK
reportable scan:
  num_configs=40
  num_diagnostic_only_configs=40
  num_oracle_configs=36
  num_reportable_method_configs=0
  num_suspicious_configs=0
  num_uses_gt_for_prediction=0
  num_gt_selected_output_and_method_result=0
  num_forbidden_for_method_table_and_method_result=0
  num_alignment_used_for_prediction=0
metric integrity:
  phase0_pass=True
  gt_files_read_by_rescore=False
```

## 代码修改审计说明

本轮核心修改是把 v15 计划中的“measurement 不等于 object”和“oracle 不进 method table”落到代码里。

- `prediction_manifest.py`: 新增 `gt_selected_output`、`forbidden_for_method_table`、`alignment_source`、`alignment_used_for_prediction`、`alignment_used_for_diagnostic`。
- `scan_reportable_configs.py`: 新增对 GT-selected output、forbidden method table、prediction alignment 的 suspicious 检查。
- `oracle_candidate_upper_bound.py`: oracle 输出强制标记 `gt_selected_output=true` 和 `forbidden_for_method_table=true`。
- `diagnose_v15_union_oracle.py`: 新增 multi-measurement union oracle，用 GT 只做 diagnostic selection。
- `build_v15_mask_region_measurements.py`: 新增 non-GT mask-region measurement candidate 生成和可选 GT diagnostic。
- `test_v15_pure.py`: 新增 v15 纯测试。
- `reproduce_v15_*.sh`: 新增 Phase 0/1/2/6 复现脚本。

## Phase 1: Atom-as-Measurement Union Oracle

v14 的 target-dominant atom 单个 candidate oracle 很低。v15 改问: 如果一个 object 允许由多个 atoms 联合解释，upper bound 会不会提升？

| source | best K | oracle AP | AP50 | AP25 | candidate pre% | selected union pre% | mean best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A3 target atom | 8 | 0.297386 | 0.529412 | 0.764706 | 3.0295 | 2.0379 | 0.500310 | False |
| A4 target atom | 8 | 0.294118 | 0.529412 | 0.764706 | 3.0469 | 2.0379 | 0.499494 | False |
| A4 target minpts5 atom | 8 | 0.238095 | 0.485714 | 0.742857 | 3.4740 | 1.9971 | 0.458516 | False |

证据链:

1. A3/A4 target atom single-candidate oracle 在 v14 是 `0.068627 / 0.117647 / 0.558824`。
2. v15 A3/A4 K=8 union oracle 提升到 AP50 `0.529412`，AP25 `0.764706`。
3. mean best IoU 提升到约 `0.50`，平均每个 GT 使用约 `2.73` 个 atoms。
4. 但 candidate pre% 仍只有 `3.03-3.47%`，selected union pre% 只有约 `2.0%`。

分析:

- 这是 v15 最重要的好消息: atom 作为 measurement basis 有真实 object signal。
- 但它仍不是 broad-support object basis。AP50 没过 `0.60`，AP25 对 A3/A4 只有 `0.764706`，略低于 `0.78`，pre% 更远低于 `25%`。
- 因此不能直接写 global solver method claim；solver 可能能选出更好的 subset，但没有足够 dense support 形成完整 object field。

## Phase 2: Mask-Region Measurement Candidate

本轮实现了三个 non-GT mask-region measurement variants:

```text
R0 component: predicted mask connected component
R0b component r010: R0 的 materialization radius 从 0.05 放宽到 0.10
R1 seed_voronoi: 用 D4RT surfel seeds 做 grid/Voronoi split
R2 boundary_core: 对 component 做边界 erosion
```

直接 candidate 结果:

| variant | AP | AP50 | AP25 | pre% | regions | hit rate | weighted purity | contamination | completeness | best region IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 component | 0.000132 | 0.000478 | 0.036858 | 5.4552 | 151.2 | 0.075273 | 0.585791 | 0.414209 | 0.060564 | 0.016472 |
| R0b r010 | 0.000092 | 0.000475 | 0.036856 | 5.6308 | 151.8 | 0.077466 | 0.575856 | 0.424144 | 0.060050 | 0.016797 |
| R1 seed_voronoi | 0.000146 | 0.001065 | 0.025048 | 4.5435 | 173.4 | 0.055488 | 0.697447 | 0.302553 | 0.043924 | 0.016551 |
| R2 boundary_core | 0.000137 | 0.000495 | 0.036890 | 5.3673 | 141.6 | 0.074881 | 0.587792 | 0.412208 | 0.061695 | 0.016070 |

修复结果:

- R0b 放宽 NN radius 只把 pre% 从 `5.4552%` 提高到 `5.6308%`，hit rate 从 `0.075273` 到 `0.077466`，没有改善 AP。
- R1 seed_voronoi 是有效 purity repair: weighted purity 从 `0.585791` 到 `0.697447`，contamination 从 `0.414209` 降到 `0.302553`。
- R1 的代价是 support/completeness 下降: pre% 从 `5.4552%` 到 `4.5435%`，completeness 从 `0.060564` 到 `0.043924`。
- R2 boundary_core 基本没改善 R0。

分析:

- 当前 mask-region materialization 的 RGB-D backprojection 命中率低，约 `5.5-7.7%`，是一个真实 materialization blocker。
- 但单纯增大 NN radius 不解决根因，说明不是半径小一点这么简单。
- seed_voronoi split 可以降低跨物体污染，但它进一步碎片化 support，仍不能形成 broad-support object。

## Phase 2: Region Union Oracle

为了确认“多个 regions 联合解释 object”是否能救回 region primitive，本轮对 R0/R1/R2 运行 union oracle。

| source | best K | oracle AP | AP50 | AP25 | candidate pre% | selected union pre% | mean best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0 component | 32 | 0.054374 | 0.170213 | 0.539435 | 5.4552 | 4.9739 | 0.296611 | False |
| R1 seed_voronoi | 8 | 0.130556 | 0.450000 | 0.800000 | 4.5435 | 3.8349 | 0.447398 | False |
| R2 boundary_core | 32 | 0.055270 | 0.171347 | 0.551510 | 5.3673 | 4.8855 | 0.298155 | False |

证据链:

1. R1 split 后 AP25 oracle 达到 `0.800000`，说明多 region union 的确能覆盖一部分 object coarse support。
2. R1 AP50 只有 `0.450000`，低于 `0.60` gate。
3. R1 candidate pre% 只有 `4.5435%`，仍远低于 `25%` broad-support gate。
4. R0/R2 即使 K=32，AP50 也只有约 `0.17`。

分析:

- R1 是比 R0/R2 更好的 measurement primitive，但仍不是合格 broad-support basis。
- region split 解决了部分 purity，但没有解决 dense support/materialization。
- 当前不应进入 Phase 3/4 solver，因为 solver 只能在低 support、低 AP50 upper bound 上优化，容易制造误导性 method result。

## Phase 6: D4RT Geometry Diagnostic

输出:

- `Stream3D/outputs/audit/v15_phase6/d4rt_sim3_residual_probe5.{json,csv,md}`

结果:

| metric | value |
|---|---:|
| num windows | 5 |
| ok windows | 5 |
| failed windows | 0 |
| anchor count mean | 431.2 |
| Sim3 scale mean/min/max | 0.560101 / 0.194515 / 0.978872 |
| residual median mean | 0.468208 |
| residual p90 mean | 0.859581 |
| residual p95 mean | 1.077516 |
| uv in01 rate mean | 0.985845 |

分析:

- 5/5 windows 都可拟合 Sim3，UV in-bounds 率高，说明 correspondence 不是完全坏。
- 但 residual median mean `0.468m`、p90 mean `0.860m`，尺度波动也大。
- 这支持 v15 的核心判断: D4RT 当前更适合作 semantic measurement/correspondence backbone，不适合作直接替换 ScanNet RGB-D metric geometry 的主方法卖点。

## Gate

v15 broad-support gate:

```text
oracle AP50 >= 0.60
oracle AP25 >= 0.78
pre% >= 25
```

最好观测:

| criterion | best observed | pass |
|---|---:|---|
| atom union AP50 | 0.529412 | False |
| atom union AP25 | 0.764706 | False |
| atom candidate pre% | 3.4740 | False |
| region union AP50 | 0.450000 | False |
| region union AP25 | 0.800000 | True |
| region candidate pre% | 5.6308 | False |
| any Phase 1/2 broad-support gate pass | False | False |

因此:

```text
Phase 3 object proposal controls: not started
Phase 4 global set packing solver: not started
Phase 5 reportable materialization method: not started
tune30/final: not started
```

## 主要 Insight

1. v15 相比 v14 的真正正结果是 atom union oracle。A3/A4 target atom 从 single AP50 `0.117647` 到 K=8 AP50 `0.529412`，证明 atom primitive 不是没有 object signal，而是不该直接作为 object 导出。
2. 这个正结果仍被 support 卡死。best atom candidate pre% 只有 `3.4740%`，selected union pre% 约 `2%`，距离 broad-support object field 很远。
3. mask-region primitive 不是自然解药。完整 component support 也只有 `5.4552%`，direct AP/AP50 几乎为零。
4. seed_voronoi split 是有效 purity repair，但不是 completeness repair。它把 weighted purity 提到 `0.697447`、contamination 降到 `0.302553`，同时 pre% 和 completeness 下降。
5. region union oracle 的最好结果 `0.130556 / 0.450000 / 0.800000` 说明 split region 可以解释 coarse AP25，但 AP50 和 support 仍不足。
6. R0b 放宽 materialization radius 没有实质收益，说明 raw mask pixels 到 ScanNet mesh 的 support collapse 不是简单 NN radius 超参问题。
7. v15 audit 字段比 v14 更安全。GT-selected oracle 输出现在明确有 `gt_selected_output=true` 和 `forbidden_for_method_table=true`，scanner 会阻止它们进入 method table。
8. D4RT Sim3 diagnostic 再次说明，D4RT correspondence 有价值，但 metric geometry replacement 风险很高。

## 结论

Stream4D v15 在 probe5 上没有获得可报告的正式方法成功，也没有启动 tune30/final。最强的正结论是: atom-as-measurement + multi-atom set cover 明显优于 atom-as-object，证明 global latent object explanation 方向有科学信号。最强的负结论是: 当前 measurement materialization/support 仍太窄，best atom/region candidate pre% 只有 `3-6%`，不能支撑 broad object field。

下一步如果继续，不应直接写 Phase 4 solver，而应先解决 support materialization 和 region/object dense extent 问题。更具体地说，需要找到一种 non-GT 的 owned-region materialization，让 candidate pre% 从 `3-6%` 提升到至少 `15-25%`，同时保持 R1 级别的 purity/contamination；否则 solver 只会在 tiny subset 上变得更聪明，不能形成完整 4D semantic reconstruction。

## 审计材料

核心输出:

- `Stream3D/outputs/audit/v15_phase0/reportable_config_scan_v15_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase0/metric_integrity_v15_probe5.{json,md}`
- `Stream3D/outputs/audit/v15_phase1/a3t16_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase1/a4t16_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase1/a4t16mp5_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase2/r0_component_region_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase2/r0b_component_region_r010_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase2/r1_seed_voronoi_region_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase2/r2_boundary_core_region_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase2/r1_seed_voronoi_region_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase6/d4rt_sim3_residual_probe5.{json,csv,md}`

复现脚本:

- `Stream3D/scripts/reproduce_v15_phase0_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase1_union_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase2_regions_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase6_geometry_probe5.sh`
