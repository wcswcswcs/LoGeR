# Stream4D v17 执行日志

日期: 2026-06-09  
计划文档: `docs/stream4d_v17_non_gt_object_explanation_solver_plan_for_codex.md`  
结果复盘: `docs/stream4d_v17_实验结果复盘.md`  
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D`  
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`  
GPU: `CUDA_VISIBLE_DEVICES=6,7`

## 执行原则

- 不把 oracle 当 method。
- method 生成/selection/scoring 不读取 GT。
- GT 只用于 evaluation、oracle diagnostic label、可视化 diagnostic。
- 所有 AP/AP50/AP25 从 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 或统一 summary JSON 读取。
- 失败和 blocker repair 都记录，不补造数据。

## 代码改动

新增:

- `Stream3D/tools/diagnose_measurement_bank_v17.py`
- `Stream3D/tools/diagnose_v17_oracle_feature_separation.py`
- `Stream3D/tools/export_v17_object_explanation_solver.py`
- `Stream3D/tools/summarize_v17_object_explanation.py`
- `Stream3D/scripts/reproduce_v17_phase0_audit_probe5.sh`
- `Stream3D/scripts/reproduce_v17_phase1_measurement_bank_probe5.sh`
- `Stream3D/scripts/reproduce_v17_phase2_feature_separation_probe5.sh`
- `Stream3D/scripts/reproduce_v17_phase3_solver_probe5.sh`
- `Stream3D/scripts/reproduce_v17_phase3_repairs_probe5.sh`
- `Stream3D/scripts/v17_object_explanation_matrix_probe5.json`

修改:

- `Stream3D/tools/update_config_manifest_fields.py`
  - 增加 `--algorithm-name`、`--algorithm`、`--forbidden-for-method-table`、`--gt-selected-output`，用于补齐 repair 产物 manifest。
- `Stream3D/tools/export_v17_object_explanation_solver.py`
  - 在第一轮失败后增加 `--w-conflict`，用于 Phase 2 指出的 low-conflict repair。

## Phase 1 Measurement Bank

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
bash scripts/reproduce_v17_phase1_measurement_bank_probe5.sh
```

输出:

- `outputs/audit/v17_phase1/measurement_bank_fixed_probe5.{json,csv,md}`
- log: `outputs/audit/v17_logs/v17_phase1_measurement_bank_fixed.log`

关键结果:

```text
positive_observation_count_per_surfel_mean = 13.0517333984375
source_propagated_count_per_surfel_mean = 1.6324951171875
surfel_positive_observation_rate = 0.99215087890625
unobserved_surfel_ratio = 0.00784912109375
measurement_density_not_main_bottleneck = True
```

## Phase 2 Oracle Feature Separation

命令:

```bash
bash scripts/reproduce_v17_phase2_feature_separation_probe5.sh
```

输出:

- `outputs/audit/v17_phase2/c_hybrid_oracle_feature_separation_probe5.{json,md}`
- `outputs/audit/v17_phase2/c_hybrid_oracle_feature_separation_probe5_{features,candidates}.csv`
- log: `outputs/audit/v17_logs/v17_phase2_c_hybrid_feature_separation.log`

审计修正:

- 初次运行时误把 `oracle_selected` label 本身收入 feature 列，产生 AUC=1.0 假信号。
- 已修正 `diagnose_v17_oracle_feature_separation.py`，显式排除 `oracle_selected` 和 bool label 后重跑。

重跑后关键结果:

```text
num_candidates = 851
num_oracle_selected_candidates = 474
best_feature_auc_directional = 0.7959826075277843
num_features_auc_ge_0p62 = 12
has_single_feature_auc_ge_0p70 = True
has_three_features_auc_ge_0p62 = True
```

Top feature:

```text
conflict_rate low AUC = 0.7959826075277843
boundary_risk_proxy low AUC = 0.7833411677802773
surfel_candidate_coverage low AUC = 0.7702044790652386
```

## Phase 3 Solver And Controls

命令:

```bash
bash scripts/reproduce_v17_phase3_solver_probe5.sh
```

生成 method/control configs:

- `stream4d_v17_m17_real_probe5`
- `stream4d_v17_m17_shuffle_probe5`
- `stream4d_v17_m17_no_temporal_probe5`
- `stream4d_v17_m17_no_negative_probe5`
- `stream4d_v17_m17_area_only_probe5`
- `stream4d_v17_m17_random_same_count_probe5`

每个 config 跑:

- own
- `Stream3D on M`
- `M on S0`
- `M on S1`
- inherit C_hybrid parent

输出:

- `outputs/audit/v17_phase3/object_explanation_matrix_probe5.{json,csv,md}`
- `outputs/audit/v17_phase4/object_explanation_summary_probe5.{json,csv,md}`
- `outputs/audit/v17_phase3/cross_prepoints/*_summary.json`
- logs: `outputs/audit/v17_logs/*`

第一轮关键结果:

```text
M17-real own = 0.016074 / 0.042181 / 0.227249
M17-real own pre% = 41.831394
M17-real on S1 = 0.006206 / 0.022883 / 0.215747
M17-shuffle own = 0.006643 / 0.021480 / 0.152819
M17-no-temporal own = 0.023292 / 0.060455 / 0.218871
M17-area-only own = 0.034478 / 0.072073 / 0.226008
```

Gate:

```text
minimum_pass = False
strong_pass = False
real_minus_shuffle_ap50_ge_0p05 = False
real_minus_no_temporal_ap25_ge_0p05 = False
```

Oracle overlap diagnostic:

```text
oracle_selected_count = 474
solver_selected_count = 424
selected_overlap_with_oracle = 326
oracle_recall_by_solver = 0.6877637130801688
solver_precision_vs_oracle = 0.7688679245283019
```

## Blocker Repairs

复现脚本:

```bash
bash scripts/reproduce_v17_phase3_repairs_probe5.sh
```

实际执行过的 repair configs:

- `stream4d_v17_m17_repair_conflict_probe5`
- `stream4d_v17_m17_repair_strict_probe5`
- `stream4d_v17_m17_repair_cmask_probe5`
- `stream4d_v17_m17_repair_cmask_surfel_probe5`
- `stream4d_v17_m17_repair_cmask_strict_probe5`

输出:

- `outputs/audit/v17_repairs/repair_summary_probe5.{json,csv,md}`
- `outputs/audit/v17_repairs/cross_prepoints/*_summary.json`
- logs: `outputs/audit/v17_logs/stream4d_v17_m17_repair_*`

Repair 结果摘要:

```text
base_real own = 0.016074 / 0.042181 / 0.227249
repair_conflict own = 0.071127 / 0.159550 / 0.294447
repair_strict own = 0.066604 / 0.136194 / 0.268990
repair_cmask own = 0.101653 / 0.248464 / 0.494844
repair_cmask_surfel own = 0.062632 / 0.159308 / 0.333776
repair_cmask_strict own = 0.106253 / 0.220706 / 0.404878
```

Best repair by own AP50:

```text
repair_cmask = 0.101653 / 0.248464 / 0.494844
repair_cmask on S1 = 0.102883 / 0.242779 / 0.576250
repair_cmask pre% = 60.8353
```

仍未达到 v17 minimum own gate:

```text
required own = AP >= 0.16, AP50 >= 0.35, AP25 >= 0.55, pre% >= 25
best own = 0.101653 / 0.248464 / 0.494844, pre% = 60.8353
```

## Diagnostic Visuals

命令为一次性 matplotlib diagnostic 生成，输出:

- `outputs/audit/v17_visuals/visualization_manifest.json`
- `outputs/audit/v17_visuals/feature_hist_conflict_rate.png`
- `outputs/audit/v17_visuals/feature_hist_boundary_risk_proxy.png`
- `outputs/audit/v17_visuals/feature_hist_surfel_candidate_coverage.png`
- `outputs/audit/v17_visuals/feature_hist_surfel_max_min_ioc.png`
- `outputs/audit/v17_visuals/m17_controls_own_ap_bars.png`
- `outputs/audit/v17_visuals/v17_repairs_own_ap_bars.png`
- `outputs/audit/v17_visuals/m17_real_oracle_solver_overlap.png`
- `outputs/audit/v17_visuals/best_repair_stream3d_same_support_ap50.png`

说明:

- 这些图全部为 diagnostic-only。
- 因 minimum gate 未通过，Phase 5 posterior materialization 未启动，因此没有 core/fringe/unknown/reject ownership map。

## Phase 0 Final Audit

命令:

```bash
CONFIGS='stream4d_v17_m17_real_probe5,stream4d_v17_m17_shuffle_probe5,stream4d_v17_m17_no_temporal_probe5,stream4d_v17_m17_no_negative_probe5,stream4d_v17_m17_area_only_probe5,stream4d_v17_m17_random_same_count_probe5,stream4d_v17_m17_repair_conflict_probe5,stream4d_v17_m17_repair_strict_probe5,stream4d_v17_m17_repair_cmask_probe5,stream4d_v17_m17_repair_cmask_surfel_probe5,stream4d_v17_m17_repair_cmask_strict_probe5' \
  bash scripts/reproduce_v17_phase0_audit_probe5.sh
```

输出:

- `outputs/audit/v17_phase0/reportable_config_scan_v17_probe5.{json,csv,md}`
- `outputs/audit/v17_phase0/metric_integrity_v17_probe5.{json,md}`
- logs:
  - `outputs/audit/v17_logs/v17_phase0_py_compile.log`
  - `outputs/audit/v17_logs/v17_phase0_unittest.log`
  - `outputs/audit/v17_logs/v17_phase0_reportable_scan.log`
  - `outputs/audit/v17_logs/v17_phase0_metric_integrity.log`

核心审计结果:

```text
py_compile: pass
unittest: Ran 30 tests in 1.475s OK
num_configs = 11
num_reportable_method_configs = 11
num_configs_missing_manifest = 0
num_configs_missing_eval_policy = 0
num_suspicious_configs = 0
num_uses_gt_for_prediction = 0
num_gt_selected_output_and_method_result = 0
num_alignment_used_for_prediction = 0
metric_integrity phase0_pass = True
gt_files_read_by_rescore = False
```

## 未启动阶段

- Phase 5 posterior materialization: 未启动，因为 Phase 3/repair minimum gate 未通过。
- Phase 6 D4RT geometry adapter: 未启动，本轮主要 blocker 在 non-GT object selection/materialization 前的 candidate-union method gate。
- Tune30/final: 未启动，因为 probe5 strong pass 未通过。
