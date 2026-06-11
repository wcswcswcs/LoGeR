# Stream4D v12 执行日志

日期: 2026-06-09  
计划文档: `docs/stream4d_v12_executable_object_explanation_plan_for_codex.md`  
结果复盘: `docs/stream4d_v12_实验结果复盘.md`  
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`  
Stream3D 目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D`  
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`  
GPU: `CUDA_VISIBLE_DEVICES=6,7`  
结论先行: v12 完成 Phase 0/1/2/3 probe5 和三条 M5 修复尝试。没有通过 Phase 3 minimal gate，tune30/final 未启动。

## 执行原则

- 不编造数据，不把 `NA`、`nan`、空预测或 diagnostic oracle 写成 method success。
- 所有 AP/AP50/AP25 只引用 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt`。
- GT oracle 只作为 diagnostic-only upper bound；`is_method_result=false`。
- M4/M5/M6/M7 和 M5r1/M5r2/M5r3 是无 GT method-result 配置；cross-support 行是 diagnostic-only。
- Phase 3 gate 未过，所以未启动 tune30/final。

## 代码修改

新增核心代码:

- `Stream3D/stream4d/measurement_bank.py`
  - 构建 v12 measurement bank，保存 D4RT surfel、target mask observation、source mask propagation、negative observation、boundary distance、RGB feature、可见性等字段。
  - 修复一次诊断口径 bug: `uv_in01_rate` 改为 raw valid+in-bounds rate；另保留 `visible_ok_rate`。
- `Stream3D/stream4d/object_slot.py`
  - 定义 `ObjectSlot(core/fringe/unknown/reject)` 和导出 adapter record。
- `Stream3D/stream4d/evidence_terms.py`
  - 实现 birth、posterior、negative evidence、boundary risk、appearance、temporal consistency 和 measurement voting。
  - 修复尝试中新增 `enable_target_births`，允许从 target-frame measurement 出生。
- `Stream3D/stream4d/object_explanation.py`
  - 实现 deterministic object explanation 原型；M5/M5r* 使用 measurement WTA 去除 same-frame cannot-link。
- `Stream3D/tools/build_v12_measurement_bank.py`
- `Stream3D/tools/diagnose_v12_measurement_bank.py`
- `Stream3D/tools/export_v12_object_explanation.py`
- `Stream3D/tools/diagnose_v12_object_explanation.py`
- `Stream3D/tests/test_v12_object_explanation.py`

新增复现脚本和 matrix:

- `Stream3D/scripts/reproduce_v12_phase0_probe5.sh`
- `Stream3D/scripts/reproduce_v12_measurement_bank_probe5.sh`
- `Stream3D/scripts/reproduce_v12_candidate_oracle_probe5.sh`
- `Stream3D/scripts/reproduce_v12_object_explanation_probe5.sh`
- `Stream3D/scripts/reproduce_v12_object_explanation_repairs_probe5.sh`
- `Stream3D/scripts/v12_phase0_matrix_probe5.json`
- `Stream3D/scripts/v12_candidate_oracle_matrix_probe5.json`
- `Stream3D/scripts/v12_object_explanation_matrix_probe5.json`
- `Stream3D/scripts/v12_object_explanation_repair_matrix_probe5.json`

## Phase 0 统一矩阵

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v12_phase0_probe5.sh
```

脚本做了:

- 更新旧 v6 compact artifact 的 manifest metadata:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.update_config_manifest_fields \
  --root . \
  --config stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve \
  --eval-policy own_recompute_paper_style \
  --support-source own \
  --geometry-source rgbd_eval_bridge \
  --uses-gt-for-prediction false \
  --uses-gt-for-diagnostic false \
  --is-method-result true \
  --is-diagnostic-only false \
  --reason "v12 Phase0 protocol completion for pre-existing v6 compact artifact"
```

- 对 6 个 prediction 和 6 个 support 跑完整 `prediction x support` cross-prepoints evaluation。
- 汇总:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.summarize_v10_unified_eval \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --matrix-json scripts/v12_phase0_matrix_probe5.json \
  --output-prefix outputs/audit/v12_phase0/unified_eval_matrix_probe5 \
  --plot-dir outputs/audit/v12_phase0 \
  --dataset scannet \
  --stream3d-config scannet
```

输出:

- `Stream3D/outputs/audit/v12_phase0/unified_eval_matrix_probe5.json`
- `Stream3D/outputs/audit/v12_phase0/unified_eval_matrix_probe5.csv`
- `Stream3D/outputs/audit/v12_phase0/unified_eval_matrix_probe5.md`
- heatmap/bar plots in `Stream3D/outputs/audit/v12_phase0/`
- logs: `Stream3D/logs/stream4d_v12_phase0_*`

关键结果:

- `P0 Stream3D on S0`: `0.235730 / 0.414306 / 0.537786`
- `P0 Stream3D on S1`: `0.399213 / 0.597171 / 0.742535`
- `P2 B1 on S2`: `0.328439 / 0.629266 / 0.884363`
- `P2 B1 on S0`: `0.000635 / 0.004294 / 0.010768`
- `P4 O38 on S0`: `0.033012 / 0.123089 / 0.392066`
- `P5 R1b on S1`: `0.064218 / 0.169101 / 0.415508`

## Phase 1 Measurement Bank

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v12_measurement_bank_probe5.sh
```

使用 fresh v10 D4RT carrier:

```text
outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67
```

输出:

- `Stream3D/outputs/v12_measurement_bank/*/measurement_bank.npz`
- `Stream3D/outputs/v12_measurement_bank/*/measurement_bank_summary.json`
- `Stream3D/outputs/audit/v12_measurement_bank/measurement_bank_probe5.json`
- `Stream3D/outputs/audit/v12_measurement_bank/measurement_bank_probe5.csv`
- `Stream3D/outputs/audit/v12_measurement_bank/measurement_bank_probe5.md`
- overlays: `Stream3D/outputs/audit/v12_measurement_bank/visuals/*measurement_bank_overlay*.png`

Blocker/fix:

- 初版 `uv_in01_rate` 误用了 `visible_ok`，把 visibility/confidence 阈值也算进 in-bounds rate，导致均值为 `0.830295`。
- 修复后 `uv_in01_rate_mean=0.9858451843261719`，`visible_ok_rate_mean=0.8302947998046875` 单独记录。

关键结果:

```text
num_surfels_mean = 16384
uv_in01_rate_mean = 0.9858451843261719
track_length_visible_mean = 13.284716796875
self_uv_error_p90_mean = 1.5708253622055055
cycle_uv_error_p90_mean = 3.2737268686294554
mean_positive_observations_per_surfel_mean = 1.6324951171875
mask_to_surfel_count_mean_mean = 941.2109458504808
boundary_safe_surfel_ratio_mean = 0.83087158203125
ambiguous_surfel_ratio_mean = 0.012196359731626518
unobserved_surfel_ratio_mean = 0.14576416015625
```

## Phase 2 Candidate Oracle

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v12_candidate_oracle_probe5.sh
```

脚本先导出 diagnostic-only surfel cluster candidate:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.export_v12_object_explanation \
  --bank-root outputs/v12_measurement_bank \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v12_c_surfel_cluster_candidate_probe5 \
  --mode surfel_cluster_candidate \
  --summary-root outputs/v12_candidate_oracle \
  --diagnostic-candidate-only
```

随后用 `tools.v11_candidate_pool_oracle` 生成四个 GT oracle diagnostic:

- `stream4d_v12_oracle_c_mask_probe5`
- `stream4d_v12_oracle_c_regionlet_probe5`
- `stream4d_v12_oracle_c_surfel_cluster_probe5`
- `stream4d_v12_oracle_c_hybrid_probe5`

输出:

- `Stream3D/outputs/audit/v12_candidate_oracle/*_upper_bound.json`
- `Stream3D/outputs/audit/v12_candidate_oracle/candidate_oracle_matrix_probe5.json`
- `Stream3D/outputs/audit/v12_candidate_oracle/candidate_oracle_matrix_probe5.md`
- logs: `Stream3D/logs/stream4d_v12_candidate_*`

关键结果:

| oracle | AP | AP50 | AP25 | pre% |
|---|---:|---:|---:|---:|
| C_mask | 0.224691 | 0.453333 | 0.648889 | 60.8842 |
| C_regionlet | 0.338574 | 0.613208 | 0.829643 | 18.5455 |
| C_surfel_cluster | 0.395062 | 0.750000 | 0.993360 | 4.2916 |
| C_hybrid | 0.256757 | 0.495495 | 0.702512 | 52.7795 |

注意: 以上全部使用 GT selection，只能诊断 primitive upper bound，不能进入 method table。

## Phase 3 Object Explanation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v12_object_explanation_probe5.sh
```

脚本导出并评估:

- M4: `stream4d_v12_m4_no_negative_probe5`
- M5: `stream4d_v12_m5_with_negative_probe5`
- M6: `stream4d_v12_m6_shuffled_d4rt_probe5`
- M7: `stream4d_v12_m7_no_d4rt_temporal_probe5`

并生成每个 config 的:

- own
- `Stream3D on method support`
- `method on S0`
- `method on S1`

输出:

- `Stream3D/outputs/v12_object_explanation/*_summary.json`
- `Stream3D/outputs/audit/v12_object_explanation/object_explanation_matrix_probe5.json`
- `Stream3D/outputs/audit/v12_object_explanation/object_explanation_internal_probe5.json`
- `Stream3D/outputs/audit/v12_object_explanation/reportable_config_scan_v12_object_probe5.json`
- logs: `Stream3D/logs/stream4d_v12_object_*`

关键结果:

| row | AP | AP50 | AP25 | pre% | conflict |
|---|---:|---:|---:|---:|---:|
| M4 no negative own | 0.140914 | 0.262547 | 0.459596 | 4.2958 | 99.1226 |
| M5 negative own | 0.226671 | 0.453586 | 0.745367 | 4.2958 | 55.2235 |
| M6 shuffled D4RT own | 0.006107 | 0.032675 | 0.107804 | 3.6947 | 82.4188 |
| M7 no temporal own | 0.138615 | 0.258599 | 0.429016 | 4.2958 | 99.1226 |
| M5 on S0 | 0.002042 | 0.005558 | 0.010704 | 84.6744 | 55.2235 |
| M5 on S1 | 0.023149 | 0.049873 | 0.148286 | 4.5145 | 55.2235 |
| P0 on M5 support | 0.358958 | 0.549096 | 0.740956 | 4.2958 | 0.2213 |

Internal diagnostic:

| config | slots | rejected | assigned | core | reject | explained | cannot-link |
|---|---:|---:|---:|---:|---:|---:|---:|
| M4 | 14.80 | 0.00 | 0.1148 | 0.1147 | 0.0000 | 0.5292 | 14.20 |
| M5 | 12.80 | 0.00 | 0.0998 | 0.0997 | 0.0002 | 0.5292 | 0.00 |
| M6 | 5.00 | 36.40 | 0.0082 | 0.0082 | 0.0000 | 0.2900 | 5.80 |
| M7 | 14.80 | 0.00 | 0.1145 | 0.1140 | 0.0000 | 0.5292 | 14.20 |

## Repair Attempts

Phase 3 gate 失败后按计划做三条修复。

### M5r1 relaxed birth

方向: 降低 birth/core/measurement 阈值，提高 object birth recall。  
配置要点:

```text
birth_min_surfels=8
birth_min_boundary_safe_ratio=0.50
birth_max_ambiguous_ratio=0.50
core_posterior_threshold=0.62
min_core_surfels_per_object=6
measurement_min_surfels=2
measurement_min_core_ratio=0.04
```

结果:

```text
own = 0.228270 / 0.424798 / 0.723425
P0 on M5r1 = 0.353801 / 0.523631 / 0.696026
M5r1 on S0 = 0.002028 / 0.005431 / 0.010291
M5r1 on S1 = 0.022217 / 0.046343 / 0.152798
```

### M5r2 target birth

方向: 新增 `enable_target_births`，允许从 target-frame measurement mask observation 出生。  
结果:

```text
own = 0.139815 / 0.237037 / 0.419444
P0 on M5r2 = 0.237011 / 0.281481 / 0.681481
M5r2 on S0 = 0 / 0 / 0
M5r2 on S1 = 0 / 0 / 0.024096
```

判断: target birth 在当前 posterior/WTA 下被过度压掉，平均只导出 1 个 object，退化。

### M5r3 strict posterior

方向: 提高 posterior/export 置信度，增强 negative/boundary risk，减少 conflict。  
配置要点:

```text
birth_min_surfels=16
birth_min_boundary_safe_ratio=0.70
birth_max_ambiguous_ratio=0.20
core_posterior_threshold=0.78
fringe_posterior_threshold=0.55
reject_negative_threshold=0.30
visible_outside_negative_weight=1.2
boundary_risk_weight=0.7
min_export_points_per_object=120
```

结果:

```text
own = 0.237230 / 0.455495 / 0.729781
P0 on M5r3 = 0.358958 / 0.549096 / 0.740956
M5r3 on S0 = 0.002042 / 0.005558 / 0.012047
M5r3 on S1 = 0.022346 / 0.046761 / 0.156118
```

输出:

- `Stream3D/outputs/audit/v12_object_explanation_repair/object_explanation_repair_matrix_probe5.json`
- `Stream3D/outputs/audit/v12_object_explanation_repair/object_explanation_repair_matrix_probe5.md`

## 审计验证

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v12_m4_no_negative_probe5,stream4d_v12_m5_with_negative_probe5,stream4d_v12_m6_shuffled_d4rt_probe5,stream4d_v12_m7_no_d4rt_temporal_probe5,stream4d_v12_m5r1_relaxed_birth_probe5,stream4d_v12_m5r2_target_birth_probe5,stream4d_v12_m5r3_strict_posterior_probe5 \
  --output outputs/audit/v12_final/reportable_config_scan_v12_methods_probe5.md \
  --require-manifest \
  --require-eval-policy
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs stream4d_v12_m4_no_negative_probe5,stream4d_v12_m5_with_negative_probe5,stream4d_v12_m6_shuffled_d4rt_probe5,stream4d_v12_m7_no_d4rt_temporal_probe5,stream4d_v12_m5r1_relaxed_birth_probe5,stream4d_v12_m5r2_target_birth_probe5,stream4d_v12_m5r3_strict_posterior_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v12_final/metric_integrity_v12_methods_probe5.md \
  --require-manifest
```

结果:

```text
py_compile: pass
unittest discover tests: Ran 16 tests ... OK
scan_reportable_configs:
  num_configs=7
  num_reportable_method_configs=7
  num_suspicious_configs=0
  num_uses_gt_for_prediction=0
  num_configs_missing_eval_policy=0
metric_integrity:
  phase0_pass=True
```

## Gate 结果

| gate | result |
|---|---|
| Phase 0 complete matrix | pass |
| Phase 1 measurement bank | pass for D4RT carrier quality; mask sparsity remains |
| Phase 2 oracle diagnostic | partial: tiny C_surfel oracle high, broad C_hybrid not enough |
| Phase 3 M5 own minimal | fail |
| Phase 3 M5 S0/S1 | fail |
| D4RT real > shuffle | pass on own-support ablation |
| tune30/final | not started |

## 审计材料

审计包:

- `stream4d_v12_probe5_code_review_packet.zip`
- SHA256: 见 `stream4d_v12_probe5_code_review_packet.sha256`
- filelist: `stream4d_v12_probe5_filelist.txt`
- git diff: `stream4d_v12_probe5_git_diff.patch`
- git status: `stream4d_v12_probe5_git_status.txt`
- zip test: `stream4d_v12_probe5_ziptest.log`
