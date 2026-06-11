# Stream4D v21.3 实验结果复盘

日期: 2026-06-10
计划文档: `docs/stream4d_v21_3_gt_guard_opend4rt_geometry_diagnostic_plan_for_codex.md`
执行日志: `docs/stream4d_v21_3_执行日志.md`
结果根目录: `Stream3D/outputs/audit`

结论先行: v21.3 完成了 Phase A guard / OpenD4RT source alignment / checkpoint-aware chunking / Sim3 / occupancy primary path 的实现与审计，并跑通真实 checkpoint smoke。继续执行后，Phase D 也完成了真正 Stream3D 内部 `GeometryProvider` replacement diagnostic rerun: G0-G6 都通过 `main -> mask_graph_construction -> frame_backprojection -> Stream3D/post_process` 路径生成，不是旧 v10/v11 那种 export-level adapter。再次继续后，按 blocker repair 方向补跑了 dense128-grid carrier support repair、mask-interior provider gate、scale-normalized self-stitch provider smoke，以及 probe5 真实 D4RT occupancy D0/D1/D2/D3/D4/D5 diagnostic。

结果没有形成可报告方法成功。Phase B/C 仍不满足进入 full-scene object formation 的 gate: cached carrier 的 `uv_in01_rate=0.633619`、`visible_track_length_mean=0.196961` 低于 `0.90/0.60`；修正真实 overlap frame 匹配后，scene0050 多窗口 self-Sim3 的 `self_sim3_scale_std=0.047587` 已过 `<=0.10`，但 plain chain `accumulated_scale_drift=0.269684` 仍高于 `0.20`。scale-normalized bundle 可以把 drift 降到约 `0`，但 provider-level G9/G10 在 scene0050 128f 上 AP 仍为 `0.0`，pre% 只有 `0.003108/0.006797`，说明主问题不是 chain scale drift。Phase D sparse/dense/interior 证据确认 geometry/materialization 是强瓶颈: G0 RGB-D baseline probe5 为 AP/AP50/AP25 `0.324948/0.497839/0.650992`；dense128 + interior 的最佳 eval-Sim3 diagnostic G7 只有 `0.144135/0.286865/0.563043`。occupancy repair 有进展但不够: 同半径 D2r4 coverage 高于 D4/D5，接入 provider 后，D2r4 + interior/outlier 的最佳 eval-Sim3 diagnostic 为 `0.198368/0.315906/0.628571`，比 dense128/interior 更好，但仍显著低于 G0，且 `pre_points_ratio=0.031738`。不使用 eval-Sim3 的 D2r4 raw/self probe5 对照只有 `0.032407/0.072917/0.625000`，G6 self-density 还在 scene0591 失败。继续按推荐思路修了 D2 overlap-window exporter，并在用户指出 self-stitch 不应更差、尺度超参也应变化后做了实现自查: 修正 self-stitch overlap frame 默认 all-window union 的污染风险，改为 best-confidence window selection；补了 Sim3 方向 synthetic test；做了 fixed-radius 与 density-alpha sweep。scene0050 64f self-stitch 仍未接近 G0，best no-GT self-stitch row 是 G10 alpha0.5 `0.037037/0.166667/0.250000`，旧 all-window G6 `0.058201/0.214286/0.357143` 也仍很低。D5 overlap warmstart 生效但 provider/AP 负修复: G3/G5 AP 只有 `0.025905/0.017455`。因此不启动 Phase E/F，不生成 method table。

本文件只基于本次落盘 artifact 和明确读取的旧 artifact 汇总；未运行字段写为未运行/NA。

## Phase A: Guard 与 OpenD4RT 对齐

新增代码:

- `Stream3D/stream4d_native/*`
- `Stream3D/stream4d_native/OPEND4RT_SOURCE_NOTES.md`
- `Stream3D/tools/scan_native_manifests.py`
- `Stream3D/tools/audit_opend4rt_source_alignment.py`
- `Stream3D/tests/test_native_chunking_and_sim3.py`
- `Stream3D/tests/test_native_occupancy_and_builder.py`

关键修复:

1. `rgbd_eval` 历史路径降级为 diagnostic-only:
   - `Stream3D/stream4d/run_scannet.py`
   - `Stream3D/stream4d/export_scannet.py`
2. manifest 新增 v21.3 guard 字段:
   - `uses_rgbd_for_prediction`
   - `uses_pose_for_prediction`
   - `uses_scannet_mesh_for_prediction`
   - `uses_gt_sim3_for_prediction`
   - `uses_d4rt_self_sim3`
   - `uses_rgbd_for_evaluation`
   - `chunking_policy`
   - `opend4rt_reference_policy`
3. 新增 `GeometryProvider` hook:
   - `Stream3D/utils/mask_backprojection.py:frame_backprojection`

审计结果:

| item | result |
|---|---:|
| native py_compile | pass |
| native unittest | 14 tests OK |
| full Stream3D tests | 50 tests OK, skipped 1 |
| source files scanned | 7 |
| forbidden import count | 0 |
| method path forbidden imports count | 0 |
| native method configs with GT/RGB-D geometry | 0 |
| OpenD4RT required files present | True |
| OpenD4RT required helpers present | True |
| source notes present | True |
| chunk size policy pass | True |
| occupancy primary path present | True |

Phase A 判定: 通过。可以做 geometry diagnostic，但不能把旧 RGB-D bridge 结果写成 method。

## Native occupancy GPU smoke

该 smoke 只验证真实 D4RT checkpoint + native occupancy path 可运行，不是 dense/probe5 实验。

设置:

| item | value |
|---|---|
| GPU | 6 |
| scene | scene0050_00 |
| frames | 0, 10, 20, 30 |
| checkpoint | `OpenD4RT_32CLIP_9Dataset_NoAUG` |
| temporal chunk size / stride / overlap | 32 / 16 / 16 |
| query budget | 4 |

结果:

| metric | value |
|---|---:|
| uses_spatiotemporal_occupancy | True |
| naive_source_query_count | 5018112 |
| actual_source_query_count | 4 |
| num_output_tubes | 2 |
| pixel_occupancy_coverage_mean | 0.0000781170 |
| query_budget_hit | True |
| total_d4rt_time_sec | 0.509300 |

解释: `query_budget_hit=True` 且 query budget 只有 4，因此不能报告 dense tracking speedup 或 coverage 成功。它只证明新 native occupancy path 能通过真实 checkpoint smoke。

Artifact: `Stream3D/outputs/audit/v21_3_geometry/native_occupancy_gpu_smoke.json`

## Phase B: D4RT geometry quality cached diagnostic

输入:

- probe5 split: `Stream3D/splits/scannet_v6_probe5.txt`
- carrier cache: `Stream3D/outputs/stream4d_debug_full_32f_ioc075_fixmem`
- variant 名称: `cached_mask_carrier_32clip_ioc075`

注意: 这是旧 cached carrier diagnostic，不是新 D0-D5 occupancy dense tracking full run。

Method-internal metrics mean:

| metric | mean |
|---|---:|
| uv_in01_rate | 0.633619 |
| self_uv_error_p90 | 0.014398 |
| visible_track_length_mean | 0.196961 |
| visibility_mean | 0.212568 |
| confidence_mean | 0.998792 |
| trajectory_acceleration_p90 | 0.046394 |
| local_neighbor_stretch_p90 | 0.234798 |
| local_neighbor_outlier_rate | 0.014355 |
| mask_interior_coverage_mean | 0.991041 |
| mask_boundary_coverage_mean | 0.602354 |

Per-scene highlights:

| scene | uv_in01 | visible track len | self uv p90 | boundary cov |
|---|---:|---:|---:|---:|
| scene0050_00 | 0.395775 | 0.218336 | 0.018674 | 0.730264 |
| scene0011_00 | 0.614117 | 0.078951 | 0.016305 | 0.437070 |
| scene0030_00 | 0.520935 | 0.387942 | 0.012463 | 0.685683 |
| scene0081_01 | 0.799623 | 0.057445 | 0.012220 | 0.464911 |
| scene0591_00 | 0.837644 | 0.242131 | 0.012330 | 0.693844 |

Evaluation-only Sim3 mean:

| metric | mean |
|---|---:|
| anchor_candidates | 189491.2 |
| anchor_valid | 31550.2 |
| anchor_count | 6233.8 |
| sim3_scale | 0.223768 |
| median_residual | 0.643861 |
| p90_residual | 1.057819 |
| p95_residual | 1.182737 |
| inlier_ratio | 0.899996 |
| translation_norm | 5.015110 |

Phase B 判定:

- image-space gate 不过: `uv_in01_rate=0.633619 < 0.90`。
- track length gate 不过: `visible_track_length_mean=0.196961 < 0.60`。
- eval-only Sim3 residual 偏大: p90 residual mean `1.057819`。
- 这说明当前 cached D4RT geometry 不能作为稳定 full-scene geometry backbone。即使 mask id 覆盖看起来高，也不能抵消 UV out-of-bounds 和 visibility 长度不足。

Artifact:

- `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics.md`
- `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics.json`
- `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_phase_b_rows.csv`

## Phase C: chunk overlap self-Sim3

输入:

- multi-window cache: `Stream3D/outputs/stream4d_debug_scene0050_128f_ioc075_fixmem`
- scene: `scene0050_00`
- windows: 7
- pairs: 6
- match signal: D4RT uv/xyz/visibility only；不使用 GT/RGB-D 参与 self-Sim3。

审计修复:

- 初始 Phase C artifact 使用相同 local frame index 匹配相邻 windows，未按真实 ScanNet frame id 对齐 overlap。对 32/16 sliding window，这会把非 overlap frame 混入 self-Sim3。
- 已修复 `tools/native_geometry_diagnostics.py`，通过 `summary.json` 恢复每个 window 的真实 frame id，只在共同 frame 上做 UV nearest-neighbor matching。
- 以下以 `native_geometry_diagnostics_corrected_overlap.*` 为准；旧 `native_geometry_diagnostics_phase_c_rows.csv` 仅保留为历史错误诊断，不作为最终结论。

Corrected aggregate:

| metric | value |
|---|---:|
| num_pairs | 6 |
| alignment_fail_count | 0 |
| overlap_frame_count_mean | 16.0 |
| overlap_anchor_count_mean | 7568.0 |
| self_sim3_inlier_ratio_mean | 0.899932 |
| self_sim3_residual_p90_mean | 0.111853 |
| self_sim3_scale_std | 0.047587 |
| accumulated_scale_drift | 0.269684 |

Corrected pair rows:

| pair | overlap frames | anchors | scale | p90 residual |
|---|---:|---:|---:|---:|
| w000 -> w001 | 16 | 7160 | 0.945616 | 0.148605 |
| w001 -> w002 | 16 | 5657 | 1.012924 | 0.132771 |
| w002 -> w003 | 16 | 8015 | 0.901958 | 0.114902 |
| w003 -> w004 | 16 | 8192 | 1.002363 | 0.090686 |
| w004 -> w005 | 16 | 8192 | 0.955421 | 0.096526 |
| w005 -> w006 | 16 | 8192 | 0.882699 | 0.087626 |

Gate sweep:

| setting | anchors mean | residual p90 | scale std | accumulated drift | gate |
|---|---:|---:|---:|---:|---|
| uv0.010 m512 | 7568.0 | 0.111853 | 0.047587 | 0.269684 | False |
| uv0.005 m512 | 6159.667 | 0.107457 | 0.048867 | 0.255112 | False |
| uv0.0025 m512 | 3787.667 | 0.104421 | 0.050291 | 0.246115 | False |
| uv0.005 vis/conf0.7 | 5156.333 | 0.098947 | 0.049296 | 0.254580 | False |
| uv0.005 vis/conf0.9 | 1883.833 | 0.093877 | 0.048916 | 0.266656 | False |
| uv0.010 m2048 | 15373.667 | 0.111948 | 0.048123 | 0.266380 | False |
| uv0.005 m2048 | 9108.833 | 0.107804 | 0.049220 | 0.253983 | False |

Scale-normalized local bundle:

| metric | value |
|---|---:|
| scale_bias_removed | 0.948968 |
| normalized_scale_std | 0.050146 |
| normalized_accumulated_scale_drift | 0.000000 |
| normalized_residual_median_mean | 0.096214 |
| normalized_residual_p90_mean | 0.179600 |

Phase C 判定:

- overlap frame 匹配修复后，pairwise self-Sim3 质量显著好于初始报告。
- overlap anchors 足够: corrected mean `7568.0`, 高于计划要求 `200`。
- inlier ratio 足够: mean `0.899932`, 高于 `0.60`。
- scale stability 已过: `self_sim3_scale_std=0.047587 <= 0.10`。
- accumulated scale drift 在普通链式 pairwise 下仍失败: best sweep `0.246115 > 0.20`。
- 简单收紧 UV radius / visibility / confidence / match cap 不能把 drift 压到 gate 内。
- 非 GT scale-normalized local bundle 可以把 accumulated drift 降到约 `0`，normalized residual p90 mean `0.179600` 没有爆炸。这说明 Phase C 的 drift 可能可由 D4RT-only scale prior / local bundle 缓解。
- 但该 bundle 仍只是 diagnostic；Phase B image-space gate 和 Phase D provider/materialization gap 未过，所以不能据此启动 Phase E/F。

Artifact:

- Initial stale artifact: `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_phase_c_rows.csv`
- Corrected baseline: `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap.*`
- Scale-normalized bundle: `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_bundle.*`
- Corrected sweeps: `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005.*`, `native_geometry_diagnostics_corrected_overlap_uv0025.*`, `native_geometry_diagnostics_corrected_overlap_uv005_vc07.*`, `native_geometry_diagnostics_corrected_overlap_uv005_vc09.*`, `native_geometry_diagnostics_corrected_overlap_m2048.*`, `native_geometry_diagnostics_corrected_overlap_uv005_m2048.*`

## Phase D: Stream3D GeometryProvider replacement

继续执行后已完成 v21.3 要求的 G0-G6 diagnostic rerun。关键修复:

1. 新增 `D4RTCarrierProjectionProvider`:
   - 从 `carriers_window*.npz` 直接读 D4RT carrier。
   - 用 `src_frame_global` 恢复真实 ScanNet frame id，修复旧 helper 把 local `0..31` 当真实帧号的问题。
   - 在 `frame_backprojection` 内返回 `mask_id -> scene point ids`，下游仍走 Stream3D set-cover / manifold refining / neighbor merging / historical merge。
2. 新增 `tools/run_v21_3_stream3d_provider_replacement.py`:
   - 对 G0-G6 使用同一 Stream3D 内部路径 rerun。
   - G0-G6 都限制到 D4RT debug cache 覆盖的同一批 32-frame support，避免把 frame coverage mismatch 当 geometry failure。
3. 修复 Stream3D 导出边界:
   - 0-object prediction 写成 `(N, 0)`，不崩溃。
   - 新 config 名用 `dataset.seq_name` 写 TMP pre_points。
4. Manifest 修复:
   - G3/G4/G5 使用 evaluation-only GT/RGB-D Sim3 生成 diagnostic output，因此 `uses_gt_for_prediction=true`、`is_diagnostic_only=true`、`forbidden_for_method_table=true`。
   - 这些 row 不能进入 method table。

Phase D G0-G6 结果:

| variant | AP | AP50 | AP25 | pre% | union% | #pred | projection hit | empty mask | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G0 RGBD baseline | 0.324948 | 0.497839 | 0.650992 | 0.229350 | 0.229350 | 57.8 | NA | NA | ok |
| G1 D4RT raw | 0.000000 | 0.000000 | 0.500000 | 0.000607 | 0.000607 | 1.0 | 0.014528 | 0.910255 | ok |
| G2 D4RT self-stitched | 0.000000 | 0.000000 | 0.500000 | 0.000607 | 0.000607 | 1.0 | 0.014528 | 0.910255 | ok |
| G3 D4RT eval-Sim3 | 0.067447 | 0.170455 | 0.581163 | 0.018251 | 0.018251 | 26.8 | 0.676159 | 0.161914 | ok |
| G4 eval-Sim3 + outlier filter | 0.085961 | 0.227257 | 0.616071 | 0.016314 | 0.016314 | 27.2 | 0.607961 | 0.191265 | ok |
| G5 eval-Sim3 + density | 0.051376 | 0.126974 | 0.388286 | 0.022382 | 0.022382 | 33.2 | 0.840654 | 0.086268 | ok |
| G6 self-stitched + density | 0.072531 | 0.222222 | 1.000000 | 0.001005 | 0.001005 | 2.2 | 0.033196 | 0.896859 | ok |

Deltas:

- `delta_d4rt_eval_sim3 = AP(G0) - AP(G3) = 0.257501`
- `delta_self_stitch = AP(G3) - AP(G2) = 0.067447`
- `delta_outlier = AP(G4) - AP(G3) = 0.018514`
- `delta_density_threshold = AP(G5) - AP(G3) = -0.016071`

Phase D 判定:

- full provider replacement diagnostic 已完成。
- G3 仍远低于 G0，说明 D4RT geometry/materialization 即使用 evaluation-only Sim3 对齐后仍不足。
- G4 小幅提升，local outlier 是次要 blocker。
- G5 projection hit 更高但 AP 更低，说明 density/radius 扩张引入污染，不能作为有效修复。
- G2 与 G1 相同是因为 probe5 debug cache 每 scene 只有 1 个 window，`self_stitch_pair_count=0`，不能在该 cache 上证明 self-stitch 修复有效。
- 所有 row 都是 diagnostic-only，不是 method result。

## Phase D Repair: dense128-grid carrier support

目的: 按 Phase D 后续修复方向，测试 D4RT carrier/object-surface support 是否主要受每 mask carrier 太稀、随机采样不稳影响。本轮没有改算法代码，只新增 3 个单 scene split 文件用于多 GPU 并行:

- `Stream3D/splits/scannet_v21_3_dense128_scene0030.txt`
- `Stream3D/splits/scannet_v21_3_dense128_scene0081.txt`
- `Stream3D/splits/scannet_v21_3_dense128_scene0591.txt`

设置:

| item | value |
|---|---|
| checkpoint | `OpenD4RT_32CLIP_9Dataset_NoAUG` |
| split | `splits/scannet_v6_probe5.txt` |
| frames | stride 10, max 32 |
| window | 32 / 16 |
| sampling | `grid_inside_mask` |
| max/min points per mask | 128 / 8 |
| merged debug root | `Stream3D/outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1` |

Dense cache per-scene generation:

| scene | carriers | props | objects | total sec | export hit |
|---|---:|---:|---:|---:|---:|
| scene0011_00 | 49664 | 86 | 86 | 179.000 | 0.683929 |
| scene0030_00 | 44160 | 12 | 12 | 186.893 | 0.835200 |
| scene0050_00 | 88064 | 17 | 17 | 357.486 | 0.928501 |
| scene0081_01 | 106624 | 492 | 492 | 380.236 | 0.590275 |
| scene0591_00 | 185216 | 271 | 271 | 918.571 | 0.847446 |

Provider replacement dense result:

| variant | AP | AP50 | AP25 | pre% | union% | #pred | projection hit | empty mask | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G0 RGBD baseline | 0.324948 | 0.497839 | 0.650992 | 0.229350 | 0.229350 | 57.8 | NA | NA | ok |
| G1 D4RT raw | 0.086785 | 0.201183 | 0.497041 | 0.003953 | 0.003953 | 6.8 | 0.013630 | 0.887625 | ok |
| G2 D4RT self-stitched | 0.086785 | 0.201183 | 0.497041 | 0.003953 | 0.003953 | 6.8 | 0.013630 | 0.887625 | ok |
| G3 D4RT eval-Sim3 | 0.140696 | 0.289128 | 0.537907 | 0.090106 | 0.090106 | 70.0 | 0.676236 | 0.078927 | ok |
| G4 eval-Sim3 + outlier filter | 0.136916 | 0.281781 | 0.556021 | 0.082324 | 0.082324 | 67.8 | 0.608571 | 0.095128 | ok |
| G5 eval-Sim3 + density | 0.116184 | 0.234043 | 0.454701 | 0.111051 | 0.111051 | 87.8 | 0.898392 | 0.045044 | ok |
| G6 self-stitched + density | 0.079138 | 0.169082 | 0.543860 | 0.006601 | 0.006601 | 9.0 | 0.037451 | 0.860509 | ok |

Dense vs sparse comparison:

| variant | sparse AP | dense AP | sparse pre% | dense pre% | sparse hit | dense hit |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 0.000000 | 0.086785 | 0.000607 | 0.003953 | 0.014528 | 0.013630 |
| G2 | 0.000000 | 0.086785 | 0.000607 | 0.003953 | 0.014528 | 0.013630 |
| G3 | 0.067447 | 0.140696 | 0.018251 | 0.090106 | 0.676159 | 0.676236 |
| G4 | 0.085961 | 0.136916 | 0.016314 | 0.082324 | 0.607961 | 0.608571 |
| G5 | 0.051376 | 0.116184 | 0.022382 | 0.111051 | 0.840654 | 0.898392 |
| G6 | 0.072531 | 0.079138 | 0.001005 | 0.006601 | 0.033196 | 0.037451 |

Dense deltas:

- `delta_d4rt_eval_sim3 = AP(G0) - AP(G3) = 0.184252`
- `delta_self_stitch = AP(G3) - AP(G2) = 0.053911`
- `delta_outlier = AP(G4) - AP(G3) = -0.003780`
- `delta_density_threshold = AP(G5) - AP(G3) = -0.024511`

Audit:

| item | value |
|---|---:|
| dense configs scanned | 7 |
| reportable method configs | 0 |
| diagnostic-only configs | 7 |
| suspicious configs | 3 |
| uses_gt_for_prediction | 3 |
| uses_gt_for_prediction_and_method_result | 0 |
| missing manifest | 0 |
| missing eval policy | 0 |
| tests | 51 OK, skipped 1 |

Dense repair 判定:

- 128/grid carrier 确实缓解了 support 稀疏问题: G3 pre_points ratio `0.018251 -> 0.090106`，AP `0.067447 -> 0.140696`。
- 但它没有解决 D4RT-native geometry: raw/self G1/G2 仍只有 pre% `0.003953`、projection hit `0.013630`，说明不经 eval-Sim3 时 D4RT geometry 仍无法稳定落到 ScanNet point universe。
- G5 的 projection hit 达 `0.898392`，pre% 达 `0.111051`，但 AP 低于 G3，说明半径/density 扩张继续引入污染；这与 v19 dilation repair 和 Phase D sparse G5 一致。
- G2 等同 G1，`self_stitch_pair_count=0`，因为 dense probe5 cache 每 scene 仍只有 1 个 window，不能用它证明 self-stitch 修复。
- 因此 dense128-grid 是有效诊断修复，但不是可报告方法成功，不改变 Phase E/F stop decision。

## Phase D Repair: mask-interior provider gate

目的: dense128-grid 后继续尝试减少 3D 半径扩张和 2D mask 边界附近 carrier 带来的污染。本轮修改只作用于 diagnostic provider:

- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 新增 `min_mask_interior_px`。
  - 使用 `distance_transform_edt(mask == mask_id)` 过滤当前 UV 落入 mask 边界附近的 carrier。
  - 新增诊断字段 `interior_filtered_point_count`。
- `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - 新增 G7/G8 diagnostic variants。
- `Stream3D/tests/test_v21_3_geometry_provider.py`
  - 新增边界过滤单测。

G7/G8 结果:

| variant | AP | AP50 | AP25 | pre% | projection hit | interior filtered | #pred | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G7 eval-Sim3 + interior gate | 0.144135 | 0.286865 | 0.563043 | 0.086586 | 0.614394 | 2131.131 | 70.2 | ok |
| G8 eval-Sim3 + density + interior gate | 0.122724 | 0.240363 | 0.493827 | 0.106406 | 0.813372 | 2131.131 | 86.0 | ok |

对照:

- G7 vs G3 dense: AP `0.144135` vs `0.140696`，小幅 +`0.003439`；pre% `0.086586` vs `0.090106`。
- G8 vs G5 dense: AP `0.122724` vs `0.116184`，小幅 +`0.006540`；仍低于 G3/G7。

Audit:

| item | value |
|---|---:|
| scanned configs | 2 |
| reportable method configs | 0 |
| diagnostic-only configs | 2 |
| suspicious configs | 2 |
| uses_gt_for_prediction | 2 |
| uses_gt_for_prediction_and_method_result | 0 |
| tests | 52 OK, skipped 1 |

Mask-interior repair 判定:

- 边界污染存在: G7/G8 都比对应 G3/G5 稍好。
- 但收益很小: 最佳 G7 AP `0.144135` 仍比 G0 `0.324948` 低 `0.180813`。
- G8 仍低于 G7/G3，说明 density/radius 扩张污染没有被 2D interior gate 根治。
- 该修复不能改变 Phase E/F stop decision。

## Phase D Repair: provider-level scale-normalized self-stitch

目的: 将 Phase C scale-normalized local bundle 从表格诊断接入 `D4RTCarrierProjectionProvider`，验证 plain chain scale drift 被消掉后，raw/self D4RT support 是否能接近 Stream3D RGB-D baseline。本轮不使用 GT/RGB-D Sim3，仍是 diagnostic-only。

代码修改:

- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 新增 `self_stitched_scale_normalized` 和 `self_stitched_scale_normalized_density`。
  - 估计相邻窗口 curr->prev 的 D4RT-only local pair Sim3。
  - 用 pair scales 的几何均值作为 `scale_bias`，分摊共同 scale 偏置后 compose 到 canonical frame。
- `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - 新增 G9/G10 diagnostic variants。

scene0050 128f provider smoke:

| variant | AP | AP50 | AP25 | pre% | hit | #pred | stitch drift | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G0 RGBD baseline | 0.251972 | 0.492424 | 0.650866 | 0.436771 | NA | 84 | NA | ok |
| G2 self-stitched | 0.000000 | 0.000000 | 0.666667 | 0.002852 | 0.013647 | 5 | NA | ok |
| G6 self-stitched density | 0.000000 | 0.000000 | 0.656250 | 0.005426 | 0.040477 | 8 | NA | ok |
| G9 scale-normalized | 0.000000 | 0.000000 | 0.333333 | 0.003108 | 0.013806 | 5 | 0.000000 | ok |
| G10 scale-normalized density | 0.000000 | 0.000000 | 0.720000 | 0.006797 | 0.047125 | 11 | 0.000000 | ok |

G9/G10 diagnostics:

| metric | value |
|---|---:|
| self_stitch_scale_bias_removed_mean | 0.950389 |
| self_stitch_accumulated_scale_drift_mean | 0.000000 |
| self_stitch_residual_p90_mean_mean | 0.162034 |

Audit:

| item | value |
|---|---:|
| scanned configs | 5 |
| reportable method configs | 0 |
| diagnostic-only configs | 5 |
| suspicious configs | 0 |
| uses_gt_for_prediction | 0 |
| uses_gt_for_prediction_and_method_result | 0 |

判定:

- scale-normalized bundle 在 provider 层确实把 self-stitch accumulated drift 归零。
- 但 G9/G10 AP 仍为 `0.0`，pre% 只有 `0.003108/0.006797`，hit 只有 `0.013806/0.047125`。
- 因此 Phase C drift 可被缓解，但 raw D4RT carrier 落点/coverage 仍是主 blocker，不能启动 Phase E/F。

## Occupancy Repair: scene0050 32f D0/D2/D3/D4 smoke

目的: 补上计划 3.7 的真实 D4RT occupancy 对照 smoke，验证 occupancy dense tracking 是否已经能在小规模真实 checkpoint 调用中改善 coverage/query tradeoff。本轮只跑 `scene0050_00` 32-frame window，不是 probe5 full D0-D5。

代码修改:

- `Stream3D/stream4d_native/occupancy_state.py`
  - 新增 mask-aware source priority: mask interior -> boundary -> uniform。
  - 新增 mask interior / boundary coverage 统计。
- `Stream3D/tools/run_v21_3_native_occupancy_ablation.py`
  - 新增真实 `D4RTAdapter` occupancy ablation runner。
  - 支持 D0 fixed full-grid、D2 mask-aware fixed、D3 occupancy uniform、D4 occupancy mask-aware。
- `Stream3D/tests/test_native_occupancy_and_builder.py`
  - 新增 mask-aware sampling 与 mask coverage summary 单测。

审计说明:

- r1 artifact 暴露 runner 汇总字段 bug: fixed D0/D2 的 coverage summary 覆盖了 `uses_spatiotemporal_occupancy`、`adaptive_speedup_vs_naive` 和 `total_d4rt_time_sec`。已修复 runner，r1 不作为最终数据。
- D0/D2 以 r2 为准；D3/D4 的 UV/visibility 统计以 r3 为准。

正式结果:

| variant | queries | occupancy | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| D0 fixed full-grid32 | 32768 | False | 0.145756 | NA | NA | 0.410795 | 0.244301 | False | 109.647530 |
| D2 mask-aware fixed32 | 22016 | False | 0.074613 | 0.072570 | 0.156897 | 0.393523 | 0.215321 | False | 73.819152 |
| D3 occupancy uniform | 4096 | True | 0.019867 | NA | NA | 0.248396 | 0.248818 | True | 25.849630 |
| D4 occupancy mask-aware | 4096 | True | 0.019924 | 0.020016 | 0.015978 | 0.248865 | 0.249287 | True | 25.472391 |

关键派生指标:

| comparison | result |
|---|---:|
| D3 query reduction vs D0 | 8.0x fewer queries |
| D4 query reduction vs D2 | 5.375x fewer queries |
| D4 mask interior coverage vs D2 | 0.020016 vs 0.072570 |
| D4 mask boundary coverage vs D2 | 0.015978 vs 0.156897 |
| D4 semantic adaptive speedup | 9742.976318 |

继续修复:

- r3 暴露 D4 greedy priority 的 boundary starvation: boundary coverage `0.015978`。
- 将 sampler 改为 priority-balanced 后，第一版 r4 因提前构建 full-image uniform candidates 卡在 CPU，已终止，不作为正式结果。
- 修复为只在剩余配额时构建 uniform candidates 后，r4/r5/r6/r7 成功落盘。

Repair 结果:

| variant | queries | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| D4 r4 priority-balanced | 4096 | 0.017740 | 0.017121 | 0.052089 | 0.226539 | 0.226877 | True | 31.612875 |
| D4 r5 priority-balanced radius4 | 4096 | 0.055019 | 0.053288 | 0.149907 | 0.223559 | 0.223820 | True | 31.467228 |
| D4 r7 priority-balanced radius4 budget7168 | 7168 | 0.091768 | 0.089131 | 0.237754 | 0.222065 | 0.222354 | True | 54.388076 |
| D4 r6 priority-balanced radius4 budget8192 | 8192 | 0.103342 | 0.100476 | 0.261895 | 0.221797 | 0.222070 | True | 61.887372 |

Scene0050 best repair vs default D2:

| variant | queries | query reduction vs D2 default | mask interior | mask boundary | time sec |
|---|---:|---:|---:|---:|---:|
| D2 default mask-aware radius2 | 22016 | 1.00x | 0.072570 | 0.156897 | 73.819152 |
| D4 r7 priority-balanced radius4 budget7168 | 7168 | 3.07x fewer | 0.089131 | 0.237754 | 54.388076 |

重要更正: 上表不是公平半径对照。继续补跑 D2 `mark_radius_px=4` 后，D4 r7 的 coverage 胜出不再成立。

Probe5 D0/D1/D2r4/D3/D4/D5 mean:

| variant | query mean | pixel cov | mask interior | mask boundary | uv_in01 | visible len | time sec | budget hit |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D0 full-grid32 radius4 | 32768.0 | 0.329996 | NA | NA | 0.583287 | 0.224036 | 107.196207 | False |
| D1 full-grid48 radius4 | 73728.0 | 0.537018 | NA | NA | 0.585857 | 0.228456 | 240.800609 | False |
| D2 fixed mask-aware radius4 | 23686.4 | 0.150638 | 0.149194 | 0.220562 | 0.631871 | 0.194399 | 77.563186 | False |
| D3 occupancy uniform r7 | 1945.6 | 0.023628 | NA | NA | 0.235452 | 0.235616 | 12.372525 | False |
| D4 r7 priority-balanced radius4 | 7168.0 | 0.080009 | 0.079015 | 0.145672 | 0.195286 | 0.195694 | 57.572386 | True |
| D5 overlap warmstart64 | 21504.0 | 0.113402 | 0.111991 | 0.207054 | 0.219479 | 0.219934 | 167.707026 | True |

D2r4 vs D4 r7 fair comparison:

| scene | D2r4 q | D4 q | D2r4 interior | D4 interior | D2r4 boundary | D4 boundary | D2r4 uv | D4 uv | D2r4 time | D4 time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | 12416 | 7168 | 0.053335 | 0.038749 | 0.097173 | 0.087170 | 0.615436 | 0.093873 | 41.129 | 56.299 |
| scene0030_00 | 11040 | 7168 | 0.186100 | 0.141894 | 0.184685 | 0.210721 | 0.515614 | 0.356110 | 36.236 | 53.312 |
| scene0050_00 | 22016 | 7168 | 0.180171 | 0.089131 | 0.324923 | 0.237754 | 0.393523 | 0.222065 | 73.586 | 54.235 |
| scene0081_01 | 26656 | 7168 | 0.080427 | 0.041652 | 0.143080 | 0.060915 | 0.797363 | 0.100243 | 86.291 | 61.744 |
| scene0591_00 | 46304 | 7168 | 0.245935 | 0.083650 | 0.352951 | 0.131802 | 0.837422 | 0.204136 | 150.574 | 62.273 |

Wins:

```text
D4 r7 > D2r4 interior coverage: 0 / 5 scenes
D4 r7 > D2r4 boundary coverage: 1 / 5 scenes
```

D5 overlap warmstart64 per-scene:

| scene | queries | windows | warmstart | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| scene0011_00 | 21504 | 3 | 8823 | 0.105969 | 0.103476 | 0.221116 | 0.219482 | 0.220039 | True | 161.481327 |
| scene0030_00 | 21504 | 3 | 10602 | 0.156739 | 0.156122 | 0.221561 | 0.296525 | 0.297231 | True | 165.621760 |
| scene0050_00 | 21504 | 3 | 9062 | 0.131435 | 0.127957 | 0.352909 | 0.255265 | 0.255636 | True | 156.868990 |
| scene0081_01 | 21504 | 3 | 7302 | 0.058262 | 0.057717 | 0.085110 | 0.107751 | 0.107971 | True | 175.380746 |
| scene0591_00 | 21504 | 3 | 8673 | 0.114603 | 0.114685 | 0.154577 | 0.218374 | 0.218793 | True | 179.182308 |

D5 window coverage progression:

| scene | w0 interior | w1 interior | w2 interior | w0 boundary | w1 boundary | w2 boundary | w1 warm | w2 warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | 0.038749 | 0.120824 | 0.150855 | 0.087170 | 0.276466 | 0.299711 | 3588 | 5235 |
| scene0030_00 | 0.141894 | 0.166736 | 0.159738 | 0.210721 | 0.230689 | 0.223273 | 5722 | 4880 |
| scene0050_00 | 0.089131 | 0.131195 | 0.163544 | 0.237754 | 0.317191 | 0.503781 | 4780 | 4282 |
| scene0081_01 | 0.041652 | 0.064146 | 0.067355 | 0.060915 | 0.096085 | 0.098331 | 3699 | 3603 |
| scene0591_00 | 0.083650 | 0.126842 | 0.133563 | 0.131802 | 0.158567 | 0.173361 | 4585 | 4088 |

判定:

- Occupancy path 现在确实调用真实 D4RT checkpoint，而不只是 fake-model unit test。
- 原始 D4 在 4096 budget 下达到很高 query-count speedup，但 `query_budget_hit=True`，且 coverage 明显低于 D2。
- priority-balanced + radius4 是有效修复，但只相对原始 D4 / 默认 radius2 D2 成立；公平的 D2r4 对照显示 D4 r7 coverage 仍不足。
- D0/D1 full-grid 能提高 pixel coverage，但 query/time 很高，且没有 mask interior/boundary 语义；D1 `73728` queries / `240.800609s` 不是可接受的 method 路径。
- D3 uniform 很省 query，但 pixel coverage 只有 `0.023628`，没有解决 object support。
- D5 overlap warmstart 生效: w1/w2 的 interior/boundary 通常高于 w0，mean mask interior/boundary 到 `0.111991/0.207054`。但它仍低于 D2r4 的 `0.149194/0.220562`，且 5/5 scenes 都 `query_budget_hit=True`。
- 到本小节为止 occupancy 还没有接入 provider/object AP；后续 provider/AP repair 已单列。D0-D5 coverage 诊断本身仍不是方法成功，它把 blocker 收窄到“需要 D4RT-native support 的 coverage/uv 共同提升，并验证能否转化为 provider/AP”。

## Phase D Repair: occupancy carrier provider/AP diagnostic

目的: 把 D2r4 / D5 occupancy support 落成 `D4RTCarrierProjectionProvider` 可读取的 `carriers_window*.npz` cache，再走完整 Stream3D provider replacement + evaluator。该实验仍是 eval-Sim3 diagnostic-only；G3/G4/G5/G7/G8 都使用 GT/RGB-D Sim3 对齐，不能进入 method table。

新增代码:

- `Stream3D/tools/export_v21_3_occupancy_carrier_cache.py`
  - D2: 导出 mask-aware fixed D4RT carrier cache。
  - D5: 导出 overlap warmstart 后每个 window 的 accepted D4RT tubes。
  - 每个 `carriers_window*.npz` 同步写 manifest，标明 diagnostic-only、无 GT/RGB-D/pose/mesh prediction。

D2r4 / D5 provider diagnostic:

| row | AP | AP50 | AP25 | pre% | hit | empty | #pred | interior px | nn radius |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2r4 G3 | 0.104381 | 0.247014 | 0.552210 | 0.044965 | 0.679178 | 0.099930 | 53.6 | 0.0 | 0.050 |
| D2r4 G5 | 0.086880 | 0.183997 | 0.462698 | 0.056575 | 0.900515 | 0.053667 | 77.2 | 0.0 | 0.124 |
| D5 G3 | 0.025905 | 0.054767 | 0.465900 | 0.013054 | 0.166395 | 0.458584 | 20.0 | 0.0 | 0.050 |
| D5 G5 | 0.017455 | 0.042783 | 0.270536 | 0.024586 | 0.372644 | 0.316338 | 31.2 | 0.0 | 0.119 |
| D2r4 G7 i2 | 0.131037 | 0.275046 | 0.588235 | 0.042806 | 0.615016 | 0.121368 | 50.0 | 2.0 | 0.050 |
| D2r4 G8 i2 | 0.100882 | 0.244156 | 0.441371 | 0.053493 | 0.812637 | 0.068842 | 75.2 | 2.0 | 0.124 |
| D2r4 G3 i4 | 0.158133 | 0.274194 | 0.577545 | 0.038769 | 0.518655 | 0.197842 | 46.0 | 4.0 | 0.050 |
| D2r4 G3 i5 | 0.178432 | 0.320269 | 0.557541 | 0.037032 | 0.483039 | 0.232110 | 43.4 | 5.0 | 0.050 |
| D2r4 G3 i6 | 0.188584 | 0.293836 | 0.556523 | 0.034907 | 0.443836 | 0.273008 | 43.6 | 6.0 | 0.050 |
| D2r4 G3 i7 | 0.165676 | 0.329428 | 0.610881 | 0.033233 | 0.415857 | 0.304924 | 39.2 | 7.0 | 0.050 |
| D2r4 G3 i8 | 0.176619 | 0.318841 | 0.578814 | 0.031443 | 0.388774 | 0.331040 | 38.8 | 8.0 | 0.050 |
| D2r4 G3 i6 r0.03 | 0.148719 | 0.313457 | 0.522748 | 0.026865 | 0.310994 | 0.317438 | 34.6 | 6.0 | 0.030 |
| D2r4 G3 i6 r0.07 | 0.148766 | 0.285707 | 0.566323 | 0.039151 | 0.520232 | 0.253598 | 47.6 | 6.0 | 0.070 |
| D2r4 G4 i5 | 0.162343 | 0.273845 | 0.542824 | 0.033586 | 0.434551 | 0.254694 | 43.2 | 5.0 | 0.050 |
| D2r4 G4 i6 | 0.198368 | 0.315906 | 0.628571 | 0.031738 | 0.399259 | 0.296387 | 40.6 | 6.0 | 0.050 |
| D2r4 G4 i7 | 0.198176 | 0.357224 | 0.585234 | 0.030273 | 0.374129 | 0.326661 | 40.2 | 7.0 | 0.050 |

Best diagnostic row:

| row | AP | AP50 | AP25 | delta AP vs G0 | delta AP vs dense128+interior G7 |
|---|---:|---:|---:|---:|---:|
| D2r4 G4 interior6 | 0.198368 | 0.315906 | 0.628571 | -0.126580 | +0.054233 |

No-eval-Sim3 raw/self check:

| row | status | AP | AP50 | AP25 | pre% | hit | empty | #pred | error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| D2r4 G1 raw | ok | 0.032407 | 0.072917 | 0.625000 | 0.001710 | 0.013862 | 0.894686 | 3.2 | NA |
| D2r4 G2 self-stitched | ok | 0.032407 | 0.072917 | 0.625000 | 0.001710 | 0.013862 | 0.894686 | 3.2 | NA |
| D2r4 G6 self-density | failed | NA | NA | NA | 0.003370 | 0.042824 | 0.856939 | 8.25 | `scene0591_00: IndexError: index -1 is out of bounds for axis 0 with size 0` |

Provider repair 判定:

- D2r4 provider cache 能转成 AP，且 interior/outlier gate 明显有效。G3 baseline `0.104381`，G4+interior6 到 `0.198368`。
- 这说明 D4RT support 的边界污染/外点确实是 blocker 之一；过滤后 AP 提升，但 pre% 降到 `0.031738`，仍远低于 G0 的 `0.229350`。
- 不使用 eval-Sim3 时 D2r4 不可作为 method。G1/G2 的 projection hit 只有 `0.013862`，pre% 只有 `0.001710`；G6 density 路径虽然 hit 到 `0.042824`，但 scene0591 失败，且部分结果仍远低于 G0。
- D5 warmstart 在 occupancy coverage 上有效，但 provider/AP 上是负修复。D5 G3 residual p90 `1.696078`、projection hit `0.166395`，说明 overlap warmstart accepted tubes 不形成稳定 metric support。
- density 半径扩张仍是负修复。D2r4 G5 hit 到 `0.900515` 但 AP 降到 `0.086880`；G8 也低于 G7/G4。
- 当前最佳仍使用 eval-Sim3 GT/RGB-D 对齐，因此只能证明“在诊断对齐下，D2r4 + interior/outlier 比 dense128/interior 更好”，不能报告为 method。

新输出审计:

| scan | result |
|---|---:|
| scanned occupancy provider configs | 53 |
| diagnostic-only configs | 53 |
| reportable method configs | 0 |
| missing manifest | 0 |
| missing eval policy | 0 |
| suspicious configs | 22 |
| uses_gt_for_prediction | 22 |
| alignment_used_for_prediction | 22 |
| uses_gt_for_prediction_and_method_result | 0 |
| scanner exit code | 5 |

说明: scanner exit code 5 来自 22 个 eval-Sim3 diagnostic 都显式 `uses_gt_for_prediction=true` / `alignment_used_for_prediction=true`；其余 raw/self/G0 diagnostic 不使用 GT Sim3，但也不是 method。全部 53 个配置同时 `is_method_result=false`、`is_diagnostic_only=true`、`forbidden_for_method_table=true`，没有 method 泄漏。

## Phase D Repair: D2 overlap-window self-stitch smoke

目的: probe5 raw/self D2r4 失败后，排除“D2 只有单窗口，所以 self-stitch 没有机会发挥”的解释。本轮修改 `Stream3D/tools/export_v21_3_occupancy_carrier_cache.py`，让 D2 在 `max_frames > window_size` 时按 `window_size/window_stride` 导出多个窗口；`CarrierSampler` 已用 global frame/x/y 生成 stable carrier id，因此 overlap frames 中相同 source pixel 可以作为 self-stitch shared anchor。新增 `test_d2_export_uses_overlap_windows_with_stable_ids` 覆盖该行为。

scene0050 64f D2r4 overlap-window cache:

| scene | windows | queries/carriers | time sec | window uv_in01 |
|---|---:|---:|---:|---|
| scene0050_00 | 3 | 57152 | 191.969105 | 0.393523 / 0.373964 / 0.331015 |

scene0050 64f provider smoke:

| row | AP | AP50 | AP25 | pre% | hit | empty | #pred | self pairs | self fail | residual p90 | scale std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 RGB-D baseline | 0.263124 | 0.462112 | 0.669151 | 0.336807 | NA | NA | 86.0 | NA | NA | NA | NA |
| G1 raw | 0.000000 | 0.000000 | 0.225000 | 0.017568 | 0.050440 | 0.577493 | 32.0 | NA | NA | NA | NA |
| G2 self-stitched | 0.000000 | 0.000000 | 0.555556 | 0.007043 | 0.025262 | 0.858742 | 15.0 | 2.0 | 0.0 | 0.143379 | 0.008392 |
| G3 eval-Sim3 | 0.064815 | 0.194444 | 0.750000 | 0.006490 | 0.232136 | 0.350870 | 14.0 | NA | NA | 2.185261 | NA |
| G4 eval-Sim3 outlier | 0.000000 | 0.000000 | 0.500000 | 0.006211 | 0.208822 | 0.378279 | 15.0 | NA | NA | 2.185261 | NA |
| G6 self-density | 0.058201 | 0.214286 | 0.357143 | 0.012242 | 0.077385 | 0.828692 | 14.0 | 2.0 | 0.0 | 0.143379 | 0.008392 |
| G9 scale-normalized | 0.027778 | 0.250000 | 0.250000 | 0.007474 | 0.025398 | 0.858742 | 13.0 | 2.0 | 0.0 | 0.146869 | 0.034045 |
| G10 scale-normalized density | 0.030093 | 0.090278 | 0.454545 | 0.013240 | 0.080160 | 0.823519 | 14.0 | 2.0 | 0.0 | 0.146869 | 0.034045 |

判定:

- exporter 修复成功产生 overlap windows 和 shared anchors；provider 诊断里 `self_stitch_pair_count=2`、`self_stitch_fail_count=0`，说明这次 G2/G6/G9/G10 不是“没有 pair”。
- 但 self-stitch 没转化成 method 可用 geometry。G2 AP 仍 `0.0`，G6 只有 `0.058201`，G9/G10 只有 `0.027778/0.030093`，都远低于 G0 `0.263124`。
- eval-Sim3 在 64f scene0050 上也只有 G3 AP `0.064815`，低于 32f D2r4 scene0050 G3 `0.111274`，说明简单拉长到多窗口会进一步暴露 cross-window/materialization 不稳定。
- 因此不扩到 probe5，不启动 Phase E/F；下一步如果继续，应修 D4RT window metric consistency / canonical support，而不是只让 D2 多窗口化。

### Self-stitch 实现自查与尺度 sweep

用户指出 self-stitch 没道理无条件更差，并提醒尺度不一致时超参也要跟着变化。基于这个反馈，本轮重新自查并补修:

代码修复/测试:

- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 新增 `overlap_policy`。
  - self-stitch variants 改为 `best_confidence`，对 overlap frame 默认只选择一个质量最高的 window，避免把多个窗口的 canonical 噪声直接 union 进 Stream3D point universe。
  - frame diagnostics 新增 `candidate_source_windows` 和 `overlap_policy`。
- `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - 新增 `--overlap-policy`。
  - G2/G6/G9/G10 默认使用 `best_confidence`。
- `Stream3D/tests/test_v21_3_geometry_provider.py`
  - 新增 best-confidence overlap window selection 单测。
  - 新增已知 scale/translation 的 synthetic self-stitch 方向测试，确认第二窗口能被映射回 canonical scene points。

best-confidence rerun:

| row | AP | AP50 | AP25 | pre% | hit | source windows | candidate windows | nn radius |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 raw all-window | 0.000000 | 0.000000 | 0.225000 | 0.017568 | 0.050440 | 1.5 | 1.5 | 0.050000 |
| G2 self best-window | 0.000000 | 0.000000 | 0.133333 | 0.005236 | 0.022957 | 1.0 | 1.5 | 0.050000 |
| G6 self-density best-window | 0.000000 | 0.000000 | 0.375000 | 0.009673 | 0.071769 | 1.0 | 1.5 | 0.158941 |
| G9 scale-normalized best-window | 0.000000 | 0.000000 | 0.500000 | 0.005610 | 0.023278 | 1.0 | 1.5 | 0.050000 |
| G10 scale-normalized density best-window | 0.031250 | 0.093750 | 0.343750 | 0.010520 | 0.075393 | 1.0 | 1.5 | 0.163794 |

Fixed-radius sweep:

| radius | G2 AP/AP50/AP25 | G2 pre% | G2 hit | G9 AP/AP50/AP25 | G9 pre% | G9 hit |
|---:|---|---:|---:|---|---:|---:|
| 0.02 | 0.000000 / 0.000000 / 0.250000 | 0.002791 | 0.008265 | 0.000000 / 0.000000 / 0.250000 | 0.002867 | 0.008073 |
| 0.05 | 0.000000 / 0.000000 / 0.133333 | 0.005236 | 0.022957 | 0.000000 / 0.000000 / 0.500000 | 0.005610 | 0.023278 |
| 0.10 | 0.000000 / 0.000000 / 0.400000 | 0.007578 | 0.045393 | 0.032407 / 0.097222 / 0.233333 | 0.008155 | 0.045900 |
| 0.20 | 0.000000 / 0.000000 / 0.375000 | 0.010828 | 0.089884 | 0.000000 / 0.000000 / 0.481481 | 0.011494 | 0.092293 |

Density-alpha sweep (`nn_radius=0.02`, effective radius from D4RT spacing):

| alpha | G6 radius | G6 AP/AP50/AP25 | G6 pre% | G6 hit | G10 radius | G10 AP/AP50/AP25 | G10 pre% | G10 hit |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 0.5 | 0.039735 | 0.000000 / 0.000000 / 0.187500 | 0.004593 | 0.017848 | 0.040948 | 0.037037 / 0.166667 / 0.250000 | 0.004891 | 0.018758 |
| 1.0 | 0.079470 | 0.000000 / 0.000000 / 0.357143 | 0.006731 | 0.036271 | 0.081897 | 0.000000 / 0.000000 / 0.400000 | 0.007393 | 0.037971 |
| 2.0 | 0.158941 | 0.000000 / 0.000000 / 0.375000 | 0.009673 | 0.071769 | 0.163794 | 0.031250 / 0.093750 / 0.343750 | 0.010520 | 0.075393 |
| 4.0 | 0.317882 | 0.000000 / 0.000000 / 0.196970 | 0.012422 | 0.116439 | 0.327587 | 0.000000 / 0.000000 / 0.576923 | 0.013197 | 0.119539 |

自查判定:

- `best_confidence` 确认修掉了 overlap frame 多窗口 union 的污染风险，source windows 从 `1.5` 降到 `1.0`；但 AP 没有改善，说明这不是唯一主因。
- synthetic test 确认 Sim3 方向在已知 scale/translation 下可把后续窗口映回 canonical points；当前没有证据表明 transform 方向反了。
- 尺度超参确实关键: 半径/alpha 增大时 hit/pre 基本上升，但 AP 不单调，过大半径继续污染。最佳 self-stitch no-GT row 只有 G10 alpha0.5 `0.037037/0.166667/0.250000`，仍远低于 scene0050 G0 `0.263124/0.462112/0.669151`。
- 因此不能把此前结果简单解释成“self-stitch 算法一定更差”；更准确结论是: 修掉 obvious overlap selection 风险并做尺度 sweep 后，当前 D2r4 self-aligned canonical support 仍不足以形成可报告 method。

## 是否启动 Phase E/F

| phase | status | reason |
|---|---|---|
| Phase E D4RT-native semantic object formation | 未运行 | Phase B image-space gate 不过；G9/G10 provider-level bundle AP 仍为 0；best occupancy-provider diagnostic D2r4 G4 i6 仍低于 G0 且使用 eval-Sim3 GT/RGB-D 对齐；probe5 D2r4 raw/self AP 只有 `0.032407` 且 G6 失败；scene0050 D2 overlap-window self-stitch 经 best-window 修复和尺度 sweep 后仍远低于 G0 |
| Phase F evaluation-only ScanNet export | 未运行 | 没有 frozen method output；dense128 / interior / bundle / occupancy 都是 diagnostic-only |
| method table | 未生成 | 无 reportable method result |

## Insight 与证据链

1. v21.3 先修住了方法边界。`rgbd_eval` 现在被标为 diagnostic-only，native source scanner 也能阻止 `stream4d_native` 回到 ScanNet depth/pose/mesh/GT 路径。这解决的是“旧结果被误当 method”的审计问题。

2. OpenD4RT 对齐比旧实现更清晰。chunk length 从 checkpoint `model.input.clip_frames` 读取；32CLIP 默认 `32/16/16`，48CLIP 可读为 `48`；query batch size 与 temporal chunk size 分离。

3. Occupancy path 已经能跑真实 checkpoint，并已从 4-frame/4-query smoke 推进到 probe5 D0/D1/D2/D3/D4/D5 diagnostic。D4/D5 使用真实 D4RT checkpoint，但仍 `query_budget_hit=True`，且公平对照下 coverage 低于 D2r4，不能拿来证明 dense tracking 已成功。

4. cached Phase B 显示 D4RT image-space 几何仍不稳。`uv_in01_rate=0.633619` 远低于 `0.90`，`visible_track_length_mean=0.196961` 远低于 `0.60`。这比单看 mask coverage 更关键，因为 out-of-bounds 和短 visibility 会直接破坏 long-range tube/object memory。

5. corrected Phase C 的 blocker 不是 anchor 数量，也不是 pairwise residual。真实 overlap frame 匹配后，overlap anchors mean `7568.0`，inlier ratio `0.899932`，residual p90 `0.111853`，scale std `0.047587` 已过阈值；普通链式累计 drift `0.269684 > 0.20`，但 scale-normalized local bundle 可把 drift 降到约 `0`，residual p90 mean `0.179600`。后续 G9/G10 provider smoke 证明该 bundle 能在 provider 层把 drift 归零，但 AP 仍为 `0.0`，所以 drift 不是唯一或主导 blocker。

6. Phase D 现在给出更强证据。G0 RGB-D baseline 在同一 32-frame support 上仍有 `0.324948/0.497839/0.650992`，说明 Stream3D 本身在该 support 下能形成一定 object；G3 eval-Sim3 D4RT 只有 `0.067447/0.170455/0.581163`，差距 `0.257501 AP`，所以主要失败不能推给 set-cover / manifold / historical merge，而是 D4RT geometry/materialization 进入 Stream3D point universe 后覆盖太小、落点不足。

7. raw D4RT 几乎不可用。G1/G2 的 projection hit 只有 `0.014528`，pre_points ratio `0.000607`，AP/AP50 为 `0/0`。这与 Phase B 的 UV out-of-bounds / short visibility 诊断一致。

8. eval-Sim3 能修落点但不能修完整 object support。G3 projection hit 升到 `0.676159`，empty mask 降到 `0.161914`，但 pre_points ratio 仍只有 `0.018251`，远低于 G0 的 `0.229350`。这说明问题不是单纯全局尺度，而是 D4RT carrier/materialization 的 object surface support 太稀。

9. outlier filtering 是小修，density expansion 是负修。G4 比 G3 AP +`0.018514`，但 G5 projection hit 到 `0.840654` 后 AP 下降到 `0.051376`，说明扩大半径会带来污染，与 v19 dilation repair 的结论一致。

10. dense128-grid 证明“carrier support 太稀”是一个真实 blocker。G3 AP 从 `0.067447` 到 `0.140696`，pre_points ratio 从 `0.018251` 到 `0.090106`，说明更多、更均匀的 mask 内 carrier 可以显著改善 eval-Sim3 上界。

11. dense128-grid 同时证明“只加 carrier”不够。G3 dense 仍比 G0 少 `0.184252 AP`，pre% `0.090106` 仍远低于 G0 `0.229350`；raw/self D4RT 仍只有 pre% `0.003953`。这把问题收窄到: D4RT native canonical geometry / self alignment / object-surface support 仍没有把 carrier 可靠落入可评估 ScanNet scene point universe。

12. dense G5 是一次负向修复。它把 projection hit 提到 `0.898392`、pre% 提到 `0.111051`，但 AP 降到 `0.116184`，低于 G3 `0.140696`。因此后续不能简单继续放大 NN radius / density alpha，应做边界感知或 object-surface-aware 的 support selection。

13. mask-interior gate 证实边界污染是次要因素。G7 相比 G3 只提升 `0.003439 AP`，G8 相比 G5 只提升 `0.006540 AP`；它没有把 D4RT geometry 推近 G0。当前最大缺口仍是 D4RT native geometry / object support 的落点和完整性，而不是单纯边界清洗。

14. provider-level scale-normalized self-stitch 是一次有用的负证据。G9/G10 的 drift 为 `0.0`，但 pre% 仍只有 `0.003108/0.006797`，AP 仍为 `0.0`。这说明继续只修 self-Sim3 chain drift 不足以解决 Stream3D point universe 落点。

15. 真实 occupancy smoke 先证明 query efficiency 和 coverage 没平衡好。原始 D4 mask-aware occupancy 只用 `4096` queries，semantic adaptive speedup 为 `9742.976318`，但 `query_budget_hit=True`，mask interior/boundary coverage 只有 `0.020016/0.015978`，低于 D2 fixed mask-aware 的 `0.072570/0.156897`。

16. priority-balanced + radius4 是有效 occupancy 修复，但旧 D2 对照半径不公平。scene0050 r7 相对默认 D2 radius2 更好，但同样 `mark_radius_px=4` 的 D2r4 在 probe5 mean 上是 `0.149194/0.220562`，明显高于 D4 r7 `0.079015/0.145672`。

17. D4 r7 仍不是完整解决。probe5 D4 r7 全部 `query_budget_hit=True`，D4 只在 boundary 上 1/5 scene 高于 D2r4，interior 0/5 scene 高于 D2r4，且 uv_in01 mean `0.195286` 远低于 D2r4 `0.631871`。这说明 adaptive occupancy 的 query efficiency 没有转化成足够的可用 geometry。

18. D5 overlap warmstart 是有用但不足的修复。它在 probe5 上把 mean mask interior/boundary 推到 `0.111991/0.207054`，并且每个 scene 的 w1/w2 通常高于 w0，证明 overlap warmstart 真正生效；但 D5 仍低于 D2r4 的 `0.149194/0.220562`，5/5 scenes `query_budget_hit=True`，所以还不能启动 Phase E/F。

19. D2r4 occupancy support 接回 provider 后得到新的最好 diagnostic row。D2r4 G4 interior6 达到 `0.198368/0.315906/0.628571`，超过 dense128+interior G7 的 `0.144135/0.286865/0.563043`，说明 mask-aware fixed support + 更强 interior/outlier 清洗比单纯 dense carrier 更有价值。

20. 但 provider repair 仍没有达到 method success。D2r4 G4 interior6 比 G0 低 `0.126580 AP`，pre% 只有 `0.031738`，而 G0 约 `0.229350`；它还依赖 eval-Sim3 GT/RGB-D 对齐，因此不能进入 method table。

21. raw/self D2r4 对照确认 occupancy 最佳 row 不能转成 method。G1/G2 不使用 eval-Sim3 时 AP/AP50 只有 `0.032407/0.072917`，pre% `0.001710`；G6 self-density 在 scene0591 触发 `IndexError`，部分统计也只有 pre% `0.003370`、hit `0.042824`。这说明当前可见提升主要来自 eval-Sim3 诊断对齐后的 support 清洗，而不是 D4RT-native canonical geometry 已可用。

22. D2 overlap-window repair 排除了“D2 只有单窗口导致 self-stitch无效”的解释，并进一步自查了实现和尺度。scene0050 64f 有 3 个窗口、2 个 self-stitch pairs、0 fail；修正 overlap frame window selection 后 source windows 从 `1.5` 降到 `1.0`；synthetic Sim3 test 通过；fixed radius / density-alpha sweep 说明尺度超参会改变 hit/pre，但最佳 no-GT self-stitch 仍只有 G10 alpha0.5 `0.037037/0.166667/0.250000`。self-stitch 数学上能配准局部窗口，不等于投到 Stream3D point universe 后能形成 object support。

23. D5 warmstart 是 coverage 有效、metric/AP 无效的负证据。D5 occupancy coverage 提升，但 provider G3/G5 AP 只有 `0.025905/0.017455`，Sim3 residual p90 `1.696078`，说明 accepted tubes 的 metric consistency 很差。

24. 下一步 occupancy 不能只追求更少 query、更多窗口或单一半径；需要同时保住 D2r4+interior/outlier 的 object support 质量、提升 D4RT self-aligned canonical support，并让尺度自适应阈值在去掉 eval-Sim3 后仍保持 provider/AP 改善。

## 必答问题

1. v21.3 是否得到可报告方法成功: `False`。
2. Phase A guard 是否通过: `True`。
3. 是否修复旧 RGB-D bridge method 语义: `True`，未来 `rgbd_eval` manifest 为 diagnostic-only / forbidden-for-method-table。
4. 是否完成 OpenD4RT source alignment audit: `True`。
5. 是否实现 occupancy primary path: `True`，并通过 unit test + 真实 checkpoint smoke；已补 probe5 D0/D1/D2/D3/D4/D5 diagnostic。
6. Phase B geometry gate 是否通过: `False`。
7. Phase C self-Sim3 gate 是否通过: corrected pairwise `True` for anchors/inlier/scale_std；plain chain accumulated drift `False`；scale-normalized diagnostic/provider smoke `True` for drift，但 AP/pre% 仍失败。
8. Phase D full provider replacement 是否完成: `True`，但只作为 diagnostic-only，不能进入 method table。
9. dense128-grid repair 是否达成目标: `False` for method success；`True` for diagnostic repair evidence。它显著提升 G3，但未追上 G0，raw/self 仍失败。
10. mask-interior repair 是否达成目标: `False` for method success；只得到小幅 diagnostic 改善，最佳 G7 AP `0.144135`。
11. provider-level scale-normalized bundle 是否达成目标: `False` for method success；`True` for drift repair evidence。G9/G10 drift 归零，但 AP 为 `0.0`。
12. occupancy D0-D5 是否完成: `True` for diagnostic。D0/D1/D2r4/D3/D4 为 probe5 32f diagnostic，D5 为 probe5 64f overlap warmstart diagnostic；D2r4/D5 也已接入 provider/AP diagnostic；另补 scene0050 D2r4 64f overlap-window self-stitch smoke、best-window 实现修复和尺度 sweep。
13. Phase D 是否证明 D4RT geometry 可用: `False` for method。D2r4 G4 interior6 eval-Sim3 diagnostic 提升到 AP `0.198368`，但仍低于 G0 `0.324948`，且依赖 GT/RGB-D Sim3；D2r4 raw/self 无 eval-Sim3 只有 AP `0.032407`，G6 self-density 失败；scene0050 D2 overlap-window self-stitch 经实现自查和尺度 sweep 后最佳 no-GT row 只有 G10 alpha0.5 AP `0.037037`。
14. 是否运行 Phase E/F 或生成 method table: `False`。
15. 是否有 GT/RGB-D/pose/mesh 泄漏到 native method result: 本次没有 native method result；native scanner `forbidden_import_count=0`，`num_method_configs_with_gt_or_rgbd_geometry=0`。G3/G4/G5/G7/G8 明确标为 `uses_gt_for_prediction=true` 且 diagnostic-only / forbidden-for-method-table。dense128、interior、G9/G10、occupancy scans 都显示 `num_reportable_method_configs=0`、`num_uses_gt_for_prediction_and_method_result=0`；最终 occupancy all scan 为 53 configs，其中 22 个 eval-Sim3 diagnostic 标记 `uses_gt_for_prediction=true`，全部 `is_method_result=false`。

## 后续修复方向

按计划优先级:

1. 继续修 D4RT carrier/object-surface support，但不要再只做 brute-force density 或简单 2D interior filtering。dense128-grid + interior 最佳 AP `0.144135`，下一步目标仍是接近 G0 `0.324948` / pre% `0.229350`，需要更可靠的 D4RT-native canonical support。
2. 不再把 Phase C scale drift 当作唯一主线。scale-normalized provider smoke 已证明 drift 可归零但 AP 仍为 0；下一步应把 bundle 与更可靠 carrier落点/coverage 机制一起验证。
3. Occupancy 后续应从“少 query/多窗口/单一半径”转为“coverage + uv_in01 + scale-aware AP 联合目标”。D2r4 + interior/outlier 是当前最强 diagnostic support；但 raw/self AP 只有 `0.032407`，scene0050 D2 overlap-window self-stitch 在尺度 sweep 后仍不行，D5 证明 warmstart coverage 有效但 metric/AP 失败。下一步应把 D2r4 质量迁移到非 GT self-aligned provider，或修 D5 accepted-tube metric consistency。
4. 用 provider runner 作为后续所有 geometry repair 的固定 Phase D 审计入口；只有 B/C/D 同时给出可解释改善后，再启动 Phase E/F。

## 最终验证

| item | result |
|---|---|
| final unittest discover | 59 tests OK |
| dense128 reportable method configs | 0 |
| interior reportable method configs | 0 |
| G9/G10 reportable method configs | 0 |
| occupancy provider reportable method configs | 0 |
| occupancy all scanned configs | 53 |
| occupancy all uses_gt_for_prediction | 22 |
| occupancy all uses_gt_for_prediction_and_method_result | 0 |
| occupancy all scan exit code | 5, expected for eval-Sim3 diagnostic uses_gt_for_prediction |
| Phase E/F | 未运行 |

## 审计材料

- Phase A:
  - `Stream3D/outputs/audit/v21_3_phaseA/native_manifest_scan.*`
  - `Stream3D/outputs/audit/v21_3_phaseA/opend4rt_source_alignment.*`
- GPU smoke:
  - `Stream3D/outputs/audit/v21_3_geometry/native_occupancy_gpu_smoke.json`
- Phase B/C cached diagnostics:
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics.*`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_phase_b_rows.csv`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_phase_c_rows.csv`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap.*`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005.*`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv0025.*`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005_vc07.*`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005_vc09.*`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_m2048.*`
  - `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005_m2048.*`
- Phase D provider replacement:
  - `Stream3D/outputs/audit/v21_3_phaseD/D4RT_geometry_replacement_stream3d_probe5.md`
  - `Stream3D/outputs/audit/v21_3_phaseD/D4RT_geometry_replacement_stream3d_probe5.csv`
  - `Stream3D/outputs/audit/v21_3_phaseD/D4RT_geometry_replacement_stream3d_probe5.json`
  - `Stream3D/outputs/audit/v21_3_phaseD/stream4d_v21_3_provider_probe5_r1_g*_provider_diagnostics.json`
  - `Stream3D/outputs/audit/v21_3_phaseD/reportable_config_scan_provider_probe5_after_manifest_fix.*`
- Phase D dense128-grid repair:
  - `Stream3D/outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1`
  - `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5/D4RT_geometry_replacement_stream3d_probe5.*`
  - `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5_self/D4RT_geometry_replacement_stream3d_probe5.*`
  - `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5/reportable_config_scan_dense128_provider_probe5.*`
- Phase D mask-interior repair:
  - `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5_interior/D4RT_geometry_replacement_stream3d_probe5.*`
  - `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5_interior/reportable_config_scan_dense128_interior_provider_probe5.*`
- Phase D scale-normalized provider smoke:
  - `Stream3D/outputs/audit/v21_3_phaseD_scene0050_128f_bundle/D4RT_geometry_replacement_stream3d_probe5.*`
  - `Stream3D/outputs/audit/v21_3_phaseD_scene0050_128f_bundle/reportable_config_scan_scene0050_128f_bundle_provider.*`
- Occupancy scene0050 smoke:
  - `Stream3D/outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r2.*`
  - `Stream3D/outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r3_d3d4.*`
  - `Stream3D/outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r4_d4_priority_balanced.*`
  - `Stream3D/outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r5_d4_priority_balanced_radius4.*`
  - `Stream3D/outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r6_d4_priority_balanced_radius4_budget8192.*`
  - `Stream3D/outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r7_d4_priority_balanced_radius4_budget7168.*`
  - `Stream3D/outputs/audit/v21_3_occupancy_probe5_d4_r7/*`
  - `Stream3D/outputs/audit/v21_3_occupancy_probe5_d2_fixed/*`
  - `Stream3D/outputs/audit/v21_3_occupancy_probe5_d0_grid32_radius4/*`
  - `Stream3D/outputs/audit/v21_3_occupancy_probe5_d1_grid48_radius4/*`
  - `Stream3D/outputs/audit/v21_3_occupancy_probe5_d2_fixed_radius4/*`
  - `Stream3D/outputs/audit/v21_3_occupancy_probe5_d3_uniform_r7/*`
  - `Stream3D/outputs/audit/v21_3_occupancy_d5_smoke/*`
  - `Stream3D/outputs/audit/v21_3_occupancy_probe5_d5_warmstart64/*`
- Occupancy provider/AP repair:
  - `Stream3D/outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1`
  - `Stream3D/outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1`
  - `Stream3D/outputs/audit/v21_3_occupancy_provider_cache_d2r4/*`
  - `Stream3D/outputs/audit/v21_3_occupancy_provider_cache_d5_warmstart64/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d5_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_interior_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_interior4_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_interior5_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_interior6_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_interior7_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_interior8_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_i6_r003_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_i6_r007_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_i5_g4_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_i6_g4_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_i7_g4_provider_probe5/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_rawself_provider_probe5/*`
  - `Stream3D/outputs/stream4d_debug_v21_3_occupancy_d2r4_win64_scene0050_r1`
  - `Stream3D/outputs/audit/v21_3_occupancy_provider_cache_d2r4_win64_scene0050/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_provider/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_provider_bestwin/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_r002/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_r005/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_r010/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_r020/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_a05/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_a1/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_a2/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_a4/*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_provider_reportable_scan.*`
  - `Stream3D/outputs/audit/v21_3_phaseD_occupancy_all_reportable_scan.*`
- Source notes:
  - `Stream3D/stream4d_native/OPEND4RT_SOURCE_NOTES.md`

## Stream4D 当前有什么好消息

有好消息，但不是“已经得到可报告 method success”。本轮最好的消息是: 问题边界比 v19/v18 清楚很多，且诊断链条已经能稳定区分 method 和 diagnostic。

1. 审计边界守住了。最终 occupancy all scan 覆盖 `53` 个 configs，全部是 diagnostic-only，`num_reportable_method_configs=0`，`num_uses_gt_for_prediction_and_method_result=0`。这说明没有把 GT/RGB-D/pose/mesh/GT-Sim3 辅助结果误写成 method。

2. Stream3D 内部 geometry replacement 已经跑通。G0-G10 不是旧的 export-level adapter，而是通过 `main -> mask_graph_construction -> frame_backprojection -> Stream3D/post_process` 真实进入 Stream3D 内部路径。这给后续所有 geometry repair 留下了可复用审计入口。

3. D2r4 + interior/outlier 是当前最强 diagnostic repair。best row 是 D2r4 G4 interior6: AP/AP50/AP25 `0.198368 / 0.315906 / 0.628571`，高于 dense128+interior G7 的 `0.144135 / 0.286865 / 0.563043`。这说明 mask-aware fixed support 加上更强边界/外点清洗确实比单纯 dense carrier 更有效。

4. self-stitch 的明显实现风险已经排查了一轮。修了 overlap frame 多窗口 union 的污染风险，新增 best-confidence window selection；补了 synthetic Sim3 方向测试；也做了 fixed-radius 和 density-alpha scale-aware sweep。结果仍不够好，但不能再简单归因成一个低级 transform 方向 bug。

5. 尺度超参被验证为真实因素。radius/alpha 增大时 hit/pre 会明显变化，说明用户指出的“尺度不一样，超参也要跟着变化”是对的。只是当前 sweep 后 best no-GT self-stitch row 仍只有 G10 alpha0.5 `0.037037 / 0.166667 / 0.250000`，没有达到 method 级别。

6. blocker 被进一步收窄。现在更像主问题的是 D4RT-native self-aligned canonical support 还不能可靠落进 Stream3D point universe，而不是单纯缺窗口、单纯 scale drift、单纯 carrier density、单纯边界污染或单纯半径没调好。

一句话结论: v21.3 没有赢在 method table，但赢在把实验从“可能混了 RGB-D/GT 的不可信结果”推进到“诊断边界干净、best diagnostic 有提升、几个假方向被排掉、下一步 blocker 明确”的状态。
