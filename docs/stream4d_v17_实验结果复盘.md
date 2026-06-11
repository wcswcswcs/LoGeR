# Stream4D v17 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/stream4d_v17_non_gt_object_explanation_solver_plan_for_codex.md`  
执行日志: `docs/stream4d_v17_执行日志.md`  
结论先行: v17 产出了真正的非 GT method configs 和 controls，但没有达到 probe5 minimum gate。最强初始 C_hybrid solver 是 `0.016074 / 0.042181 / 0.227249`，远低于目标；按 blocker 修复后，最强 broad-support repair 是 C_mask-source 的 `repair_cmask`，own `0.101653 / 0.248464 / 0.494844`，S1 `0.102883 / 0.242779 / 0.576250`。它证明非 GT low-conflict/broad C_mask source 比 C_hybrid slot-growth 更有效，但仍不足以 claim v17 success。因此 Phase 5 materialization、Phase 6 geometry adapter、tune30/final 均未启动。

## 结果边界

- 所有 method selection/scoring 不读取 GT。
- Phase 2 oracle-selected/rejected feature analysis 使用 GT oracle label，只是 diagnostic。
- 所有 AP/AP50/AP25 来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 或 `outputs/audit/v17_*/*.json` 汇总。
- `repair_cmask` 是本轮 best method repair，但仍未通过 minimum own gate。
- C_mask-source repair 不等于原计划理想的 C_hybrid multi-measurement slot solver；它是 blocker repair，用来验证 candidate source / score direction 问题。

## 审计通过

```text
py_compile: pass
unittest: Ran 30 tests in 1.475s OK
reportable scan:
  num_configs=11
  num_reportable_method_configs=11
  num_configs_missing_manifest=0
  num_configs_missing_eval_policy=0
  num_suspicious_configs=0
  num_uses_gt_for_prediction=0
  num_gt_selected_output_and_method_result=0
  num_alignment_used_for_prediction=0
metric integrity:
  phase0_pass=True
  gt_files_read_by_rescore=False
```

核心审计输出:

- `Stream3D/outputs/audit/v17_phase0/reportable_config_scan_v17_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v17_phase0/metric_integrity_v17_probe5.{json,md}`
- `Stream3D/outputs/audit/v17_phase3/object_explanation_matrix_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v17_phase4/object_explanation_summary_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v17_repairs/repair_summary_probe5.{json,csv,md}`

## 做了什么修改

1. `Stream3D/tools/diagnose_measurement_bank_v17.py`
   - 新增 v17 measurement-bank 统计诊断。
   - 修正 v16 口径风险: 单独报告 target `positive_observation` 与 source `source_positive_propagated`。

2. `Stream3D/tools/diagnose_v17_oracle_feature_separation.py`
   - 新增 oracle-selected vs rejected 的非 GT feature AUC 诊断。
   - 修复初次运行时把 `oracle_selected` label 当 feature 的问题；修复后重跑，最佳 AUC 为 `0.7959826075277843`。

3. `Stream3D/tools/export_v17_object_explanation_solver.py`
   - 新增非 GT candidate-union solver。
   - 支持 `real/shuffle/no_temporal/no_negative/area_only/random_same_count` controls。
   - 支持 seed + growth + packing + support floor。
   - blocker repair 后新增 `--w-conflict`，用于直接惩罚 Phase 2 指出的 high-conflict 候选。

4. `Stream3D/tools/summarize_v17_object_explanation.py`
   - 新增 v17 gate / oracle recovery / oracle overlap 汇总。

5. `Stream3D/tools/update_config_manifest_fields.py`
   - 增加 `--algorithm-name`、`--algorithm`、`--forbidden-for-method-table`、`--gt-selected-output`。
   - 用于修正 `repair_cmask_surfel` manifest。

6. 复现脚本
   - `Stream3D/scripts/reproduce_v17_phase0_audit_probe5.sh`
   - `Stream3D/scripts/reproduce_v17_phase1_measurement_bank_probe5.sh`
   - `Stream3D/scripts/reproduce_v17_phase2_feature_separation_probe5.sh`
   - `Stream3D/scripts/reproduce_v17_phase3_solver_probe5.sh`
   - `Stream3D/scripts/reproduce_v17_phase3_repairs_probe5.sh`
   - `Stream3D/scripts/v17_object_explanation_matrix_probe5.json`

## Phase 1: Measurement Bank Fixed Statistics

v17 修正后的关键点是: v16 的 `mean_positive_observations_per_surfel=1.632495` 实际对应 source propagated count，而不是 target positive observation count。

| metric | value | gate |
|---|---:|---|
| num_mask_frames_available | 16.0 | pass |
| num_mask_frames_missing | 0.0 | pass |
| uv_in01_rate | 0.985845 | pass |
| visible_ok_rate | 0.830295 | diagnostic |
| positive_observation_count_per_surfel_mean | 13.051733 | pass vs 3.0 |
| source_propagated_count_per_surfel_mean | 1.632495 | old v16-like source propagated count |
| surfel_positive_observation_rate | 0.992151 | pass |
| unobserved_surfel_ratio | 0.007849 | pass |
| ambiguous_surfel_ratio | 0.045913 | diagnostic |
| boundary_safe_surfel_ratio | 0.972583 | diagnostic |

结论:

- measurement density 不是本轮主瓶颈。
- v16 的 “positive observations/surfel fail” 是统计命名风险，不应继续解释为 target mask density 稀疏。
- 真正的 blocker 是 non-GT object selection / ownership，而不是 mask frame 是否可用。

## Phase 2: Oracle-Selected vs Rejected Feature Separation

使用 v16 `C_hybrid K8` oracle label 作为 diagnostic label，特征本身不读取 GT。

```text
num_candidates = 851
num_oracle_selected_candidates = 474
best_feature_auc_directional = 0.7959826075277843
num_features_auc_ge_0p62 = 12
has_single_feature_auc_ge_0p70 = True
has_three_features_auc_ge_0p62 = True
```

Top features:

| feature | direction | AUC | selected mean | rejected mean |
|---|---|---:|---:|---:|
| conflict_rate | low | 0.795983 | 0.931901 | 3.084171 |
| boundary_risk_proxy | low | 0.783341 | 0.728276 | 2.586772 |
| surfel_candidate_coverage | low | 0.770204 | 0.196414 | 0.604521 |
| surfel_max_min_ioc | low | 0.758705 | 0.216321 | 0.608210 |
| surfel_max_iou | low | 0.733483 | 0.110321 | 0.210109 |
| max_candidate_overlap | low | 0.707716 | 0.376245 | 0.678710 |

解读:

1. C_hybrid oracle-selected candidates 确实能被非 GT proxy 部分区分。
2. 最强方向不是更大 score，也不是更强 surfel coverage，而是 low conflict / low boundary risk。
3. 这直接驱动了后续 `--w-conflict` 和 single-measurement low-conflict repair。

## Phase 3: Initial M17 Solver

M17-real 是 C_hybrid candidate-union solver，使用 seed/growth/packing，不读 GT。

| config | own AP | own AP50 | own AP25 | pre% | S1 AP | S1 AP50 | S1 AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M17-real | 0.016074 | 0.042181 | 0.227249 | 41.8314 | 0.006206 | 0.022883 | 0.215747 |
| M17-shuffle | 0.006643 | 0.021480 | 0.152819 | 38.9364 | 0.005901 | 0.021094 | 0.186536 |
| M17-no-temporal | 0.023292 | 0.060455 | 0.218871 | 44.4416 | 0.006492 | 0.015659 | 0.113210 |
| M17-no-negative | 0.016144 | 0.042346 | 0.227693 | 41.8228 | 0.006206 | 0.022883 | 0.215747 |
| M17-area-only | 0.034478 | 0.072073 | 0.226008 | 38.9839 | 0.003022 | 0.005881 | 0.079385 |
| M17-random-same-count | 0.073999 | 0.168353 | 0.302491 | 11.6333 | 0.025554 | 0.079259 | 0.131342 |

Minimum gate:

```text
required own = AP >= 0.16, AP50 >= 0.35, AP25 >= 0.55, pre% >= 25
M17-real own = 0.016074 / 0.042181 / 0.227249, pre% = 41.8314
required S1 = AP >= 0.10, AP50 >= 0.22, AP25 >= 0.40
M17-real S1 = 0.006206 / 0.022883 / 0.215747
real - shuffle AP50 = 0.020701 < 0.05
real - no-temporal AP25 = 0.008377 < 0.05
minimum_pass = False
```

Oracle recovery:

```text
AP recovery = -0.055346188207599074
AP50 recovery = -0.053109037215402645
AP25 recovery = 0.0023046280574453622
```

Oracle selected overlap:

```text
oracle_selected_count = 474
solver_selected_count = 424
selected_overlap_with_oracle = 326
oracle_recall_by_solver = 0.6877637130801688
solver_precision_vs_oracle = 0.7688679245283019
```

关键矛盾:

- Solver selected indices 和 oracle 有较高 overlap，但 AP 仍接近失败。
- 这说明“选中了 oracle 常选的 measurement index”不等于“生成了可评估 object mask”。
- C_hybrid slot growth 可能把候选边界/ownership 组合坏了，或者 C_hybrid pool 的 duplicate/conflict 让排序分数无法转化为 object AP。

## Blocker Repairs

按计划尝试了 score direction、negative/conflict、slot-conditioned marginal、seed purity、candidate source 修复。

| repair | own AP | own AP50 | own AP25 | S1 AP | S1 AP50 | S1 AP25 | pre% | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base_real | 0.016074 | 0.042181 | 0.227249 | 0.006206 | 0.022883 | 0.215747 | 41.8314 | 初始失败 |
| repair_conflict | 0.071127 | 0.159550 | 0.294447 | 0.035567 | 0.090712 | 0.216459 | 42.0572 | conflict penalty 有效但不够 |
| repair_strict | 0.066604 | 0.136194 | 0.268990 | 0.021219 | 0.051509 | 0.155793 | 35.7085 | 更严格 packing 变差 |
| repair_cmask | 0.101653 | 0.248464 | 0.494844 | 0.102883 | 0.242779 | 0.576250 | 60.8353 | 最强 repair，S1 过 minimum |
| repair_cmask_surfel | 0.062632 | 0.159308 | 0.333776 | 0.078317 | 0.183893 | 0.443170 | 59.8411 | 加 C_surfel anchors 变差 |
| repair_cmask_strict | 0.106253 | 0.220706 | 0.404878 | 0.062486 | 0.159154 | 0.401461 | 50.8625 | AP 略升但 AP50/AP25 降 |

Best repair:

```text
repair_cmask own = 0.101653 / 0.248464 / 0.494844
repair_cmask S1 = 0.102883 / 0.242779 / 0.576250
repair_cmask pre% = 60.8353
```

仍失败:

```text
own AP target 0.16, got 0.101653
own AP50 target 0.35, got 0.248464
own AP25 target 0.55, got 0.494844
```

Repair 证据链:

1. `repair_conflict` 从 initial real 的 AP50 `0.042181` 提升到 `0.159550`，说明 Phase 2 的 low-conflict signal 是真实有用的。
2. `repair_strict` 更干净但变差，说明仅靠更少、更纯的 C_hybrid measurement 不能过 gate。
3. `repair_cmask` 大幅超过 C_hybrid-source repairs，说明 broad source 更适合 C_mask 而不是 C_hybrid slot-growth。
4. `repair_cmask` 在 S1 上过了 minimum，但 own 没过，说明 selected support 上 object boundary/duplicate/ownership 仍不足。
5. `repair_cmask_surfel` 下降，说明直接把 tiny C_surfel anchors 加回输出不是解法；高 precision tiny objects 不能简单补 broad support。

## Gate 判定

v17 minimum:

```text
Method own:
  AP   >= 0.16
  AP50 >= 0.35
  AP25 >= 0.55
  pre_points % >= 25

Method on S1:
  AP   >= 0.10
  AP50 >= 0.22
  AP25 >= 0.40

D4RT contribution:
  real - shuffle AP50 >= 0.05
  real - no-temporal AP25 >= 0.05
```

Best initial C_hybrid solver:

```text
M17-real own = 0.016074 / 0.042181 / 0.227249
M17-real S1 = 0.006206 / 0.022883 / 0.215747
pass = False
```

Best repair:

```text
repair_cmask own = 0.101653 / 0.248464 / 0.494844
repair_cmask S1 = 0.102883 / 0.242779 / 0.576250
own gate pass = False
S1 gate pass = True
overall v17 minimum pass = False
```

Final:

```text
minimum_pass=False
strong_pass=False
phase5_materialization_started=False
phase6_geometry_adapter_started=False
tune30_started=False
final_started=False
```

## 主要 Insight

1. v17 成功修正了 v16 measurement-bank 误判。target positive observation count mean 是 `13.0517`，不是 `1.6325`；后者是 source propagated count。因此 bank density 不该再作为主 blocker。
2. oracle selection 不是纯 GT 幻觉。低 conflict / 低 boundary risk 的非 GT proxy 对 oracle-selected candidates 有 AUC `0.796`，说明有可分性。
3. 但 C_hybrid slot-growth 仍不能把可分性转成 AP。M17-real 与 oracle index overlap 很高，但 AP 很低，说明 mask ownership/materialized object shape 仍坏。
4. C_hybrid 的 union/growth 容易引入 duplicate/conflict。直接加 conflict penalty 后 AP50 提升约 3.8 倍，但仍远低于 gate。
5. C_mask broad source 是本轮最大 method 正信号。`repair_cmask` 比 C_hybrid real 强很多，并且 S1 gate 通过；这说明 broad non-GT method 不是完全无望，但 v17 原始 C_hybrid object explanation 形式不够。
6. tiny high-precision C_surfel 不能简单加入 broad output。`repair_cmask_surfel` 下降，说明 high-precision anchors 更适合做 birth/diagnostic，不适合直接扩成 reportable broad method。
7. `Stream3D on repair_cmask support = 0.224924 / 0.401511 / 0.577226`，远高于 `repair_cmask own = 0.101653 / 0.248464 / 0.494844`。同 support 上仍有明显 method gap，说明不是 support 本身完全不行，而是 object proposal/score/ownership 不够。

## 结论

v17 没有达到计划定义的 method success。它完成了从 oracle 到非 GT method 的第一步，并且给出了可审计的 controls 和 repairs，但最强 broad-support non-GT result 仍是 `repair_cmask own = 0.101653 / 0.248464 / 0.494844`，低于 `0.16 / 0.35 / 0.55` minimum gate。

更准确的失败结论是:

```text
C_hybrid K8 oracle upper bound is partially aligned with non-GT low-conflict signals,
but current non-GT C_hybrid slot-growth solver cannot materialize that signal into object AP.
C_mask broad source plus low-conflict selection is stronger, but still below v17 minimum.
```

下一步如果继续，不应直接写 Phase 5 materializer，也不应继续只调 C_hybrid K/threshold。更合理方向:

1. 以 C_mask broad support 为主，重新设计 object ownership / split-explain，而不是 C_hybrid union growth。
2. 把 low-conflict/boundary-risk 从 candidate-level 改成 mask-internal split cue，避免整 mask 进入 object。
3. 保留 C_regionlet/C_surfel 作为 seed/anchor，但不要直接把 tiny anchors 合并成 broad output。
4. 先缩小 `Stream3D on M support` 与 `M own` 的 same-support gap，再考虑 Phase 5 materialization。

## 审计材料

审计包:

- `stream4d_v17_code_review_packet.zip`
- `stream4d_v17_code_review_packet.sha256`
- `stream4d_v17_code_review_packet_filelist.txt`
- `stream4d_v17_code_review_packet_ziptest.log`
- `stream4d_v17_code_review_packet_git_status.txt`
- `stream4d_v17_code_review_packet_git_diff.patch`

核心输出:

- `Stream3D/outputs/audit/v17_phase1/measurement_bank_fixed_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v17_phase2/c_hybrid_oracle_feature_separation_probe5.{json,md}`
- `Stream3D/outputs/audit/v17_phase2/c_hybrid_oracle_feature_separation_probe5_{features,candidates}.csv`
- `Stream3D/outputs/audit/v17_phase3/object_explanation_matrix_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v17_phase4/object_explanation_summary_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v17_repairs/repair_summary_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v17_visuals/visualization_manifest.json`
- `Stream3D/outputs/audit/v17_phase0/reportable_config_scan_v17_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v17_phase0/metric_integrity_v17_probe5.{json,md}`

复现脚本:

- `Stream3D/scripts/reproduce_v17_phase0_audit_probe5.sh`
- `Stream3D/scripts/reproduce_v17_phase1_measurement_bank_probe5.sh`
- `Stream3D/scripts/reproduce_v17_phase2_feature_separation_probe5.sh`
- `Stream3D/scripts/reproduce_v17_phase3_solver_probe5.sh`
- `Stream3D/scripts/reproduce_v17_phase3_repairs_probe5.sh`
- `Stream3D/scripts/v17_object_explanation_matrix_probe5.json`
