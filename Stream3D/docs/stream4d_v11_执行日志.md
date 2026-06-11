# Stream4D v11 执行日志

日期: 2026-06-09  
计划文档: `docs/stream4d_v11_deep_algorithmic_reboot_plan_for_codex.md`  
结果复盘: `docs/stream4d_v11_实验结果复盘.md`  
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`  
Stream3D 目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D`  
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`  
GPU: `CUDA_VISIBLE_DEVICES=6,7`  
结论先行: v11 在 probe5 上完成了 candidate oracle、measurement density、posterior proxy、GT failure attribution 和 D4RT geometry adapter 归因。没有产生可进入正式 method table 且超过 Stream3D 的新方法结果；tune30/final 未启动。

## 执行原则

- 不编造数据，不把空预测、`nan`、`NA` 改写成有利数字。
- 所有 AP/AP50/AP25 只引用 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 解析结果。
- oracle 和 GT failure attribution 使用 GT，因此全部标记为 diagnostic-only，不能进入 method table。
- 严格协议边界: D4RT 在生成 prediction/TMP 前只能做 D4RT-internal self-alignment；任何对 GT/reference scene geometry 的对齐只能发生在评估/测指标阶段。G2-G5 使用 ScanNet RGB-D depth/pose world points 在 diagnostic materialization/export 阶段做 Sim3，因此严格协议下是 invalid anchor-assisted artifact，只保留为审计反例。
- v11 method scan 只扫描非 oracle 的 S2-S5 posterior proxy 配置。
- 用户补充的 pasted text 指出: 过筛和 mask/object matching 都存在，根因是 mask-to-object ownership/matching；本轮对应实现了 candidate upper bound、oracle/filtered attribution、per-GT failure attribution。

## 代码修改

### `Stream3D/evaluation/evaluate.py`

目的: 让 diagnostic oracle 可以被显式评估，同时继续阻止 GT 泄漏进入 method result。

修改:

- 默认仍拒绝 `uses_gt_for_prediction=true` 的配置。
- 只有同时满足 `--allow-oracle-eval`、`is_diagnostic_only=true`、`is_method_result=false` 时才允许 oracle evaluation。
- 这样 Phase 1 oracle AP 能被真实计算，但不会被误报为正式方法结果。

### `Stream3D/tools/v11_candidate_pool_oracle.py`

目的: Phase 1 candidate upper-bound diagnostic。

功能:

- 从多个历史预测配置合并 candidate pool。
- 使用 `--dedup-threshold 0.95` 和 `--dedup-overlap-mode min_ioc` 去重。
- 使用 GT 只做 oracle candidate selection。
- 输出 prediction/TMP manifest，并写明 `uses_gt_for_prediction=true`、`uses_gt_for_diagnostic=true`、`is_diagnostic_only=true`、`is_method_result=false`。
- 输出 JSON/CSV/MD summary。

### `Stream3D/tools/v11_measurement_density_diagnostic.py`

目的: Phase 2 measurement density diagnostic。

功能:

- 不读取 GT，不输出 AP。
- 比较当前 sparse CropFormer mask、D4RT source-mask propagation、no-track、shuffle controls。
- 当 frozen video masklet cache 不存在时，把 M3 记录为 `not_available`，不伪造结果。

### `Stream3D/tools/v11_gt_failure_attribution.py`

目的: 对用户要求的 per-GT failure attribution 做 diagnostic-only 归因。

功能:

- 使用 GT 统计每个 GT object 的 pool best IoU 和 method best IoU。
- 分类为 `no_candidate`、`filtered_candidate`、`wrong_assignment_or_fragmentation`、`boundary_bad`、`matched`。
- 输出 JSON/CSV/MD。

### `Stream3D/stream4d/d4rt_stream3d_geometry_adapter.py`

目的: Phase 6 D4RT geometry adapter。

功能:

- materialize per-frame D4RT local point cloud。
- 生成 per-frame 2D mask 到 D4RT point index 的 mapping。
- 输出每个 scene 的 `geometry_manifest.json`。
- 明确记录 `is_complete_stream3d_replacement=false`，因为本轮没有完整重跑原版 Stream3D local proposal/set-cover/manifold stages。

### `Stream3D/tools/run_stream3d_with_d4rt_geometry.py`

目的: Phase 6 adapter CLI。

功能:

- 跑 `raw`、`scene_sim3`、`window_sim3` adapter modes。
- 汇总 anchor/residual/spacing/empty mask 指标。
- 与 v10 minimal projection geometry matrix 串联，做 diagnostic attribution。

### 新增复现脚本

- `Stream3D/scripts/reproduce_v11_candidate_oracle_probe5.sh`
- `Stream3D/scripts/reproduce_v11_measurement_density_probe5.sh`
- `Stream3D/scripts/reproduce_v11_posterior_proxy_probe5.sh`
- `Stream3D/scripts/reproduce_v11_failure_attribution_probe5.sh`
- `Stream3D/scripts/reproduce_v11_d4rt_geometry_probe5.sh`

### 新增 matrix 配置

- `Stream3D/scripts/v11_candidate_oracle_matrix_probe5.json`
- `Stream3D/scripts/v11_posterior_proxy_matrix_probe5.json`
- `Stream3D/scripts/v11_d4rt_geometry_matrix_probe5.json`

## Phase 1 Candidate Oracle

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v11_candidate_oracle_probe5.sh
```

脚本内部关键命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.v11_candidate_pool_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pool-name C5_c1_c2_c3_union \
  --pool-configs stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5,stream4d_v10_r0_fullmask_probe5,stream4d_v10_r1_maskcore_probe5,stream4d_v10_r2_depthsplit_probe5,stream4d_v10_r3_d4rtseed_probe5,stream4d_v10_r4_combined_probe5 \
  --output-config stream4d_v11_oracle_c5_c1_c2_c3_union_probe5 \
  --support-mode union \
  --summary-root outputs/audit/v11_candidate_oracle \
  --min-candidate-points 100 \
  --min-select-iou 0.25 \
  --dedup-threshold 0.95 \
  --dedup-overlap-mode min_ioc
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v11_oracle_c5_c1_c2_c3_union_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v11_oracle_c5_c1_c2_c3_union_probe5_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v11_oracle_c5_c1_c2_c3_union_probe5 \
  --no_class \
  --require-manifest \
  --allow-oracle-eval
```

输出:

- `Stream3D/outputs/audit/v11_candidate_oracle/*_upper_bound.json`
- `Stream3D/outputs/audit/v11_candidate_oracle/*_upper_bound.csv`
- `Stream3D/outputs/audit/v11_candidate_oracle/*_upper_bound.md`
- `Stream3D/outputs/audit/v11_candidate_oracle/candidate_oracle_matrix_probe5.json`
- `Stream3D/outputs/audit/v11_candidate_oracle/candidate_oracle_matrix_probe5.csv`
- `Stream3D/outputs/audit/v11_candidate_oracle/candidate_oracle_matrix_probe5.md`
- logs: `Stream3D/logs/stream4d_v11_candidate_oracle_*`

关键结果:

| pool | AP | AP50 | AP25 | pre% | GT crop/full | method result |
|---|---:|---:|---:|---:|---:|---|
| C0 Stream3D oracle pool | 0.390788 | 0.628205 | 0.760684 | 83.7810 | 40.6/40.6 | False |
| C1 B1+O1 tiny clean | 0.476608 | 0.763158 | 0.918763 | 3.8660 | 8.0/40.6 | False |
| C2 O38 large memory | 0.223210 | 0.444444 | 0.635556 | 66.5926 | 40.2/40.6 | False |
| C3 v10 regionlet birth | 0.396825 | 0.809524 | 0.904762 | 4.7191 | 7.8/40.6 | False |
| C4 v10 regionlet repair | 0.218145 | 0.559633 | 0.834862 | 21.5143 | 19.6/40.6 | False |
| C5 C1+C2+C3 union | 0.232232 | 0.459459 | 0.666667 | 57.7528 | 40.0/40.6 | False |

判断:

- C1/C3 tiny pool oracle 很高，但 support 很小，只说明 clean subset 有上界。
- C5 broad union AP=0.232232，基本等于 v10 G0 Stream3D AP=0.235730；AP50=0.459459 只比 Stream3D AP50=0.414306 高 0.045153，低于计划要求的 +0.05 headroom。
- 不满足继续大规模 selection/score 调参的条件。

## Phase 2 Measurement Density

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.v11_measurement_density_diagnostic \
  --debug-root outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67 \
  --seq-list splits/scannet_v6_probe5.txt \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --output-prefix outputs/audit/v11_measurement_density/measurement_density_probe5
```

等价脚本:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v11_measurement_density_probe5.sh
```

输出:

- `Stream3D/outputs/audit/v11_measurement_density/measurement_density_probe5.json`
- `Stream3D/outputs/audit/v11_measurement_density/measurement_density_probe5.csv`
- `Stream3D/outputs/audit/v11_measurement_density/measurement_density_probe5.md`

关键结果:

| mode | ok rows | mask density | obs/surfel | visible unobserved | self consistency | contradiction |
|---|---:|---:|---:|---:|---:|---:|
| M0 CropFormer available frames | 5/5 | 0.125000 | 1.615100 | 0.878835 | 0.982253 | 0.017747 |
| M1 framewise no propagation | 5/5 | 0.125000 | 1.615100 | 0.878835 | 0.982253 | 0.017747 |
| M2 D4RT source propagation | 5/5 | 0.125000 | 1.632495 | 0.877120 | 0.992518 | 0.007482 |
| M3 frozen video masklet | 0/5 | NA | NA | NA | NA | NA |
| M4 no-track source only | 5/5 | 0.125000 | 0.122559 | 0.877441 | 0.000000 | 0.000000 |
| M5 shuffled propagation | 5/5 | 0.125000 | 1.626343 | 0.877726 | 0.981484 | 0.018516 |
| M5b shuffled uv target control | 5/5 | 0.125000 | 1.587817 | 0.881308 | 0.980186 | 0.019814 |

判断:

- 当前 mask cache 只有 2/16 frames，mask_frame_density=0.125。
- M2 相比 M4 no-track 在 obs/surfel 上约 13.32x，但 M2 与 M5 shuffle 非常接近。
- 没有 frozen video masklet cache，所以不能声称 M3 完成。

## Phase 3/4 Posterior Proxy

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v11_posterior_proxy_probe5.sh
```

脚本使用 fresh v10 D4RT carrier:

```text
outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67
```

输出:

- `Stream3D/outputs/v11_posterior_proxy/*_summary.json`
- `Stream3D/outputs/v11_posterior_proxy/*_summary.csv`
- `Stream3D/outputs/v11_posterior_proxy/*_summary.md`
- `Stream3D/outputs/audit/v11_posterior_proxy/posterior_proxy_matrix_probe5.json`
- `Stream3D/outputs/audit/v11_posterior_proxy/posterior_proxy_matrix_probe5.csv`
- `Stream3D/outputs/audit/v11_posterior_proxy/posterior_proxy_matrix_probe5.md`

关键结果:

| row | AP | AP50 | AP25 | pre% | conflict | same-support P0 AP | gap AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| S2 area same-count own | 0.144503 | 0.305339 | 0.536614 | 4.2354 | 68.8770 | 0.372722 | -0.228220 |
| S3 real D4RT maskcount own | 0.156653 | 0.313452 | 0.552663 | 4.1903 | 68.5051 | 0.373232 | -0.216580 |
| S4 shuffled D4RT own | 0.147827 | 0.301142 | 0.534694 | 4.1948 | 69.4148 | 0.367316 | -0.219489 |
| S5 no-track own | 0.146705 | 0.305507 | 0.536841 | 4.2371 | 68.5676 | 0.372722 | -0.226017 |
| S3 real D4RT on S0 | 0.000373 | 0.002503 | 0.006594 | 84.6744 | 68.5051 | NA | NA |
| S3 real D4RT on S1 | 0.009604 | 0.024097 | 0.086757 | 4.5145 | 68.5051 | NA | NA |

判断:

- S3 - S4 delta = AP +0.008826、AP50 +0.012310、AP25 +0.017969，远低于计划要求的 +0.05 AP50/AP25。
- Same-support gap 很大: P0 on S3 AP=0.373232 vs S3 own AP=0.156653。
- S0/S1 gate 均失败。

## GT Failure Attribution

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v11_failure_attribution_probe5.sh
```

输出:

- `Stream3D/outputs/audit/v11_failure_attribution/b1_vs_c5.json`
- `Stream3D/outputs/audit/v11_failure_attribution/b1_vs_c5.csv`
- `Stream3D/outputs/audit/v11_failure_attribution/b1_vs_c5.md`
- `Stream3D/outputs/audit/v11_failure_attribution/o38_vs_c5.*`
- `Stream3D/outputs/audit/v11_failure_attribution/r1b_vs_c4.*`
- `Stream3D/outputs/audit/v11_failure_attribution/s3_vs_c5.*`

聚合结果:

| case | num GT | mean pool best IoU | mean method best IoU | no candidate | filtered | boundary bad | wrong/frag | matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 vs C5 | 236 | 0.306059 | 0.026017 | 103 | 126 | 2 | 4 | 1 |
| O38 vs C5 | 236 | 0.306059 | 0.300022 | 103 | 3 | 73 | 0 | 57 |
| R1b vs C4 | 236 | 0.103764 | 0.097447 | 195 | 2 | 36 | 1 | 2 |
| S3 vs C5 | 236 | 0.306059 | 0.023775 | 103 | 125 | 3 | 4 | 1 |

判断:

- B1/S3 的主要失败是 `filtered_candidate`，说明过筛/tiny clean subset 现象真实存在。
- O38 的 filtered 很少，但 `boundary_bad` 和 `no_candidate` 很多，说明 support 变大后仍有 object boundary/primitive 质量问题。
- R1b 的 pool 本身 `no_candidate` 占 195/236，说明 regionlet repair candidate 表达能力不足。

## Phase 6 D4RT Geometry Adapter

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v11_d4rt_geometry_probe5.sh
```

adapter 关键命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_stream3d_with_d4rt_geometry \
  --debug-root outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-name stream4d_v11_adapter_scene_sim3_probe5 \
  --mode scene_sim3 \
  --backbone Cropformer \
  --output-root outputs/v11_d4rt_stream3d_geometry_adapter \
  --summary-root outputs/audit/v11_d4rt_geometry \
  --min-visibility 0.5 \
  --min-confidence 0.5 \
  --max-anchors 8000 \
  --robust-trim-percentile 90
```

输出:

- `Stream3D/outputs/v11_d4rt_stream3d_geometry_adapter/*`
- `Stream3D/outputs/audit/v11_d4rt_geometry/stream4d_v11_adapter_*_summary.json`
- `Stream3D/outputs/audit/v11_d4rt_geometry/stream4d_v11_adapter_*_summary.csv`
- `Stream3D/outputs/audit/v11_d4rt_geometry/stream4d_v11_adapter_*_summary.md`
- `Stream3D/outputs/audit/v11_d4rt_geometry/d4rt_geometry_matrix_probe5.json`
- `Stream3D/outputs/audit/v11_d4rt_geometry/d4rt_geometry_matrix_probe5.csv`
- `Stream3D/outputs/audit/v11_d4rt_geometry/d4rt_geometry_matrix_probe5.md`
- `Stream3D/outputs/audit/v11_d4rt_geometry/reportable_config_scan_d4rt_geometry_probe5.json`

重要边界:

- 允许: D4RT 自己和自己对齐，包括 D4RT-internal coordinate/self-consistency alignment。
- 允许: prediction/TMP 冻结后，在评估/测指标阶段为了算误差而与 GT/reference scene geometry 做 post-hoc alignment。
- 不允许: 在 prediction/TMP 生成前，使用 GT/reference scene geometry 对 D4RT 做 Sim3，然后导出预测。
- 自查结果: G2-G5 没有用 instance/semantic GT label，但用 ScanNet RGB-D depth/pose backprojection world points 在 materialization/export 阶段做 Sim3，因此严格协议下不是有效 geometry replacement 实验，只能作为 anchor-assisted 反例。G1 raw 是 no-external-alignment lane，但 AP 为 NA 且 S0/S1 为 0。

scene-level Sim3 adapter evidence:

| scene | anchors | median residual | p90 residual | spacing q50 | frames | empty mask mean |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 7200 | 0.0370485 | 0.0761211 | 0.00476558 | 16 | 0 |
| scene0011_00 | 7200 | 0.0324988 | 0.0577556 | 0.0140756 | 16 | 0.03125 |
| scene0030_00 | 7200 | 0.103001 | 0.152568 | 0.0042935 | 16 | 0.0555556 |
| scene0081_01 | 7200 | 0.0386918 | 0.101404 | 0.0114319 | 16 | 0.0384615 |
| scene0591_00 | 7200 | 0.181572 | 0.287136 | 0.00409783 | 16 | 0.0398309 |

geometry matrix 关键结果:

| row | AP | AP50 | AP25 | pre% | conflict | best IoU | method table |
|---|---:|---:|---:|---:|---:|---:|---|
| G0 Stream3D RGB-D own | 0.235730 | 0.414306 | 0.537786 | 84.6744 | 0.2213 | 0.630750 | False |
| G1 raw own | NA | NA | NA | 0.0090 | 14.1772 | NA | False |
| G2 scene Sim3 own | 0.188825 | 0.364384 | 0.486341 | 1.4370 | 75.1744 | 0.633245 | False |
| P0 on G2 support | 0.314343 | 0.413636 | 0.552727 | 1.4370 | 0.2213 | 0.668357 | False |
| G2 on S0 | 0.000000 | 0.000000 | 0.000112 | 84.6744 | 75.1744 | 0.007395 | False |
| G2 on S1 | 0.000000 | 0.000000 | 0.004523 | 4.5145 | 75.1744 | 0.034475 | False |
| G3 window Sim3 own | 0.173962 | 0.334164 | 0.461727 | 1.4386 | 74.1746 | 0.625980 | False |
| G3 on S0 | 0.000000 | 0.000000 | 0.000112 | 84.6744 | 74.1746 | 0.007512 | False |
| G3 on S1 | 0.000000 | 0.000000 | 0.004523 | 4.5145 | 74.1746 | 0.034818 | False |

判断:

- Adapter 产出的 Sim3 residual 是 anchor-assisted materialization 的诊断记录；由于 Sim3 在 prediction/TMP 生成前使用 RGB-D depth/pose world points，G2/G3/G4/G5 在严格协议下剔除出有效 geometry gate。
- G2/G3 own AP 低于 G0，且 S0/S1 cross-support 基本归零；这只能作为“不允许的 anchor-assisted 路径仍未过关”的审计反例。
- G1 raw 是合规 no-external-alignment lane，但 raw own AP 为 NA，G1 on S0/S1 为 0。
- 本轮没有完整替换原版 Stream3D local proposal/set-cover/manifold pipeline；adapter manifest 已明确记录边界。

## 最终验证

### py_compile

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v11_candidate_pool_oracle.py \
  tools/v11_measurement_density_diagnostic.py \
  tools/v11_gt_failure_attribution.py \
  tools/run_stream3d_with_d4rt_geometry.py \
  stream4d/d4rt_stream3d_geometry_adapter.py \
  evaluation/evaluate.py
```

结果: 通过。

### unittest

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
...............
----------------------------------------------------------------------
Ran 15 tests in 0.130s

OK
```

### v11 method reportable scan

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v11_s2_area_same_count_probe5,stream4d_v11_s3_d4rt_maskcount_probe5,stream4d_v11_s4_shuffle_maskcount_probe5,stream4d_v11_s5_no_track_probe5 \
  --output outputs/audit/v11_posterior_proxy/reportable_config_scan_v11_posterior_proxy_probe5.md \
  --require-manifest \
  --require-eval-policy
```

结果:

```text
num_configs=4
num_configs_missing_eval_policy=0
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=4
num_suspicious_configs=0
num_uses_gt_for_prediction=0
```

### v11 method metric integrity

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs stream4d_v11_s2_area_same_count_probe5,stream4d_v11_s3_d4rt_maskcount_probe5,stream4d_v11_s4_shuffle_maskcount_probe5,stream4d_v11_s5_no_track_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v11_final/metric_integrity_v11_methods.md \
  --require-manifest
```

结果:

```text
phase0_pass=True
num_reportable_method_configs=4
num_suspicious_configs=0
num_uses_gt_for_prediction=0
num_uses_gt_for_diagnostic_and_method_result=0
```

## Blocker 与处理

| blocker | 处理 | 结果 |
|---|---|---|
| evaluator 默认拒绝 `uses_gt_for_prediction=true` oracle | 修改 `evaluate.py`，只允许 `--allow-oracle-eval` 且 diagnostic-only 非 method 的 oracle eval | oracle AP 可真实计算，method table 仍受保护 |
| M3 frozen video masklet cache 不存在 | Phase 2 中记录为 `not_available` | 没有伪造 M3 结果 |
| mask cache 只有 2/16 frames | 量化为 mask_frame_density=0.125，比较 M0/M2/M4/M5 | M2 对 no-track 提升大，但与 shuffle 接近 |
| C5 broad oracle 没有达到 +0.05 AP50 headroom | 不进入大规模 selection/score sweep | 按计划停止 tune30/final |
| 完整 Stream3D geometry replacement 成本超出本轮且原 pipeline 接口不直接支持 | 实现 D4RT geometry adapter，输出 per-frame geometry/mapping/manifest，并显式标注 `is_complete_stream3d_replacement=false` | 得到可审计 geometry attribution，但不宣称完整 replacement |
| “GT/reference 对齐”协议边界误读 | 自查确认真正规则是: 生成 prediction/TMP 前只能做 D4RT-internal self-alignment；和 GT/reference scene geometry 的对齐只能在评估/测指标阶段做。G2-G5 在 materialization/export 阶段用了 RGB-D depth/pose Sim3 | 日志修正: G2-G5 是 invalid anchor-assisted artifact，只作为审计反例；G1 raw 才是合规 no-external-alignment lane |

## 未启动项

- tune30/final: 因 S0/S1 gate、same-support gap gate、D4RT real-vs-shuffle gate 均未通过，按计划不启动。
- Phase 5 core/fringe/reject 完整 export: posterior proxy 未证明 D4RT real signal，因此未继续扩展为主方法。
- M3 frozen video masklet: workspace 中没有对应 cache，未伪造。

## 审计包

最终审计包在根目录生成:

- `stream4d_v11_final_code_review_packet.zip`
- `stream4d_v11_final_code_review_packet.sha256`
- `stream4d_v11_final_code_review_packet_filelist.txt`
- `stream4d_v11_final_code_review_packet_git_diff.patch`
- `stream4d_v11_final_code_review_packet_git_status.txt`
- `stream4d_v11_final_code_review_packet_ziptest.log`

包内包含:

- `Stream3D/stream4d/*.py`
- `Stream3D/tools/*.py`
- `Stream3D/evaluation/evaluate.py`
- `Stream3D/tests/*.py`
- `Stream3D/scripts/reproduce_v11_*.sh`
- `Stream3D/scripts/v11_*.json`
- `Stream3D/docs/stream4d_v11_执行日志.md`
- `Stream3D/docs/stream4d_v11_实验结果复盘.md`
- `Stream3D/outputs/audit/v11_*/*.json`
- `Stream3D/outputs/audit/v11_*/*.md`
- `Stream3D/outputs/audit/v11_*/*.csv`
- `Stream3D/data/evaluation/scannet/stream4d_v11*_class_agnostic.txt`
- probe5 prediction/TMP manifests for v11 configs
- `Stream3D/splits/scannet_v6_probe5.txt`
- probe5 GT text files under `Stream3D/data/scannet/gt/`

干净解包验证:

```bash
rm -rf /tmp/stream4d_v11_packet_test
mkdir -p /tmp/stream4d_v11_packet_test
unzip -q stream4d_v11_final_code_review_packet.zip -d /tmp/stream4d_v11_packet_test
cd /tmp/stream4d_v11_packet_test/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v11_s2_area_same_count_probe5,stream4d_v11_s3_d4rt_maskcount_probe5,stream4d_v11_s4_shuffle_maskcount_probe5,stream4d_v11_s5_no_track_probe5 \
  --output outputs/audit/v11_posterior_proxy/packet_test_reportable_scan.md \
  --require-manifest \
  --require-eval-policy
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs stream4d_v11_s2_area_same_count_probe5,stream4d_v11_s3_d4rt_maskcount_probe5,stream4d_v11_s4_shuffle_maskcount_probe5,stream4d_v11_s5_no_track_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v11_final/packet_test_metric_integrity.md \
  --require-manifest
```

结果: `py_compile` 通过，`unittest discover tests` 为 `Ran 15 tests ... OK`，method scan 为 `num_reportable_method_configs=4`、`num_suspicious_configs=0`、`num_uses_gt_for_prediction=0`，metric integrity 为 `phase0_pass=True`。
