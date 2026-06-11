# ACL2 v41 执行日志：ReadFirst HealthGated SemanticGeometry Target30

日期：2026-05-26（Asia/Singapore）

计划文件：

```text
docs/ACL2_v41_ReadFirst_HealthGated_SemanticGeometry_Target30_Plan.md
```

主结果目录：

```text
results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/
```

本轮原则：

```text
1. 只记录实际执行和落盘结果，不编造数据。
2. 不把 Phase 0/1/2 诊断、short rollout 或 blocked downstream stage 写成
   deployable online success。
3. 每个 LoGeR 进程只绑定一张 GPU。
4. 用户确认 GPU 0,1,2,3,4,5,6,7 可用；并行只用于无 RUN_NAME 冲突的独立 row。
5. 若遇到 blocker，先按计划修复方向处理，并把修复与边界写入复盘。
```

---

## 2026-05-26 初始化与计划审计

### 1. 阅读 v41 计划

```bash
sed -n '1,260p' docs/ACL2_v41_ReadFirst_HealthGated_SemanticGeometry_Target30_Plan.md
sed -n '260,520p' docs/ACL2_v41_ReadFirst_HealthGated_SemanticGeometry_Target30_Plan.md
sed -n '520,780p' docs/ACL2_v41_ReadFirst_HealthGated_SemanticGeometry_Target30_Plan.md
sed -n '780,1040p' docs/ACL2_v41_ReadFirst_HealthGated_SemanticGeometry_Target30_Plan.md
```

要点：

```text
主线 = read-first / health-gated / training-free。
Phase 1 用 v40 health atlas 选择 bad chunks，不能用固定 chunk id 或 ATE 作为 runtime 条件。
Phase 3 只在 health-selected chunks 上跑 R1-R5 READ-only h10；
只有 h10 过 gate 的 rows 才允许 h15；
只有 h15-qualified candidate 才允许 full online。
```

### 2. 建立结果目录并复用 parent snapshots

```bash
mkdir -p results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30

ln -s \
  ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots \
  results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase0_parent_snapshots

ls -l results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30
```

结果：

```text
phase0_parent_snapshots -> ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots
```

### 3. 审计可复用 v40/v39 artifacts

```bash
find results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30 -maxdepth 3 -type f | sort

sed -n '1,80p' \
  results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30/health_atlas/chunk_health_timeline.csv

sed -n '1,80p' \
  results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30/phase2a_read/report_h10_R1/read_h10_effects.csv

find results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal/phase0_semantic_appearance/report_R1 -maxdepth 2 -type f | sort
```

确认存在：

```text
v40 Phase 0 no-op report
v40 Phase 1 health atlas
v40 Phase 2A READ h10 report
v39 semantic appearance proxy visual summaries
H9/C9 chunk 6/10/16 parent snapshots
```

### 4. 审计已有 launcher alias 与 attention-mass instrumentation

```bash
rg -n "READ_A[1-5]|P0_00|case \"\\$CANDIDATE_ID\"|Unsupported CANDIDATE_ID" \
  tools/run_v24_candidate_rollout.sh

sed -n '540,670p' tools/run_v24_candidate_rollout.sh

rg -n "CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS|ATTENTION_MASS|record_attention_mass|context_source_skip_record_attention_mass" \
  tools/run_v24_candidate_rollout.sh tools/run_attention_cue_experiment.sh run_pipeline_abc_v2.py loger -g '!*.pyc'
```

结论：

```text
v40 READ_A1-A5 aliases exist.
default-off context source-skip attention mass instrumentation exists and can
be enabled through CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1.
v41 R1-R5 aliases still need to be added.
```

---

## 2026-05-26 工程修改与验证

### 5. 添加 v41 runtime aliases / reports

修改文件：

```text
tools/run_v24_candidate_rollout.sh
tools/v41_health_detector_report.py
tools/v41_read_mechanism_report.py
tools/v41_read_gate_report.py
tools/v41_read_washout_report.py
tools/v41_final_summary_report.py
```

新增候选：

```text
R1_READ_HIGH_INFLUENCE_ANOMALY
R2_READ_SKY_APP_ANOMALY
R3_READ_ANOMALY_PLUS_STATIC_RESCUE
R4_NEG_CONTROL_SKY_NO_SOURCE_MASS
R5_NEG_CONTROL_STATIC_RESCUE_ONLY
```

Boundary：

```text
R1/R2 复用 v40 READ_A2/A4 家族。
R3/R4/R5 是由现有 LoGeR training-free 控制拼接出的 runtime proxy。
没有引入 learned trigger/router/classifier。
没有使用 GT SemanticKITTI label 作为 runtime action。
```

### 6. 语法验证

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v41_final_summary_report.py \
  tools/v41_read_washout_report.py \
  tools/v41_health_detector_report.py \
  tools/v41_read_mechanism_report.py \
  tools/v41_read_gate_report.py

bash -n tools/run_v24_candidate_rollout.sh
```

结果：

```text
PASS
```

---

## 2026-05-26 Phase 1：Health Detector

### 7. 生成 health-selected bad chunks

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_health_detector_report.py \
  --root results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30
```

输出：

```text
phase1_health_detector/chunk_health_table.csv
phase1_health_detector/rolling_window_health_alignment.csv
phase1_health_detector/health_component_by_chunk.csv
phase1_health_detector/health_vs_rolling_ate_scatter.png
phase1_health_detector/chunk_health_timeline.png
phase1_health_detector/bad_chunk_report.md
phase1_health_detector/v41_health_detector_summary.json
phase1_health_detector/selected_bad_chunks.json
```

关键结果：

```text
phase1_gate_pass = true
selected_bad_chunks = [10]
selected_bad_chunk_ratio = 0.3333333333
top3_health_risk_chunks = [10, 6, 16]
top_rolling100_bad_chunk_diagnostic = 6
stress_window_overlap_chunks_diagnostic = [10, 6]
selection_uses_ATE = false
selection_uses_fixed_chunk_or_segment = false
```

Decision：

```text
Phase 1 gate = pass.
Phase 3 only runs on health-selected chunk10.
```

---

## 2026-05-26 Phase 2：READ mechanism / attribution

### 8. h3 scalar attention-mass supplement

```bash
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30 \
V36B_PHASE_NAME=phase2_read_mechanism/h3_R1 \
V36B_RUN_PREFIX_BASE=V41_MECH_H3_R1 \
V36B_CANDIDATES=R1_READ_HIGH_INFLUENCE_ANOMALY,R2_READ_SKY_APP_ANOMALY,R5_NEG_CONTROL_STATIC_RESCUE_ONLY \
V36B_CHUNKS=10 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=3 \
V36B_GPUS=0,1,2,3,4,5 \
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

结果：

```text
rows = 6/6 DONE
parents = H9,C9
chunk = 10
candidates = R1,R2,R5
```

### 9. 生成 mechanism report

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_read_mechanism_report.py \
  --root results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30
```

输出：

```text
phase2_read_mechanism/read_a2_a4_attribution.csv
phase2_read_mechanism/per_label_removed_source_mass.csv
phase2_read_mechanism/action_mask_overlap.csv
phase2_read_mechanism/read_a4_sky_causality_report.md
phase2_read_mechanism/READ_A2_A4_attribution_report.md
phase2_read_mechanism/scalar_attention_mass_rows.csv
phase2_read_mechanism/overlays/chunk010_proxy_rgb_frame_strip.png
phase2_read_mechanism/overlays/chunk010_proxy_semantic_mask_overlay.png
phase2_read_mechanism/overlays/chunk010_proxy_appearance_anomaly_heatmap.png
phase2_read_mechanism/overlays/spatial_attention_boundary.json
phase2_read_mechanism/v41_read_mechanism_summary.json
```

关键结果：

```text
mechanism_decision = B_general_high_influence_anomaly_preferred
best_READ_A2_ATE_delta = -0.5994474373
best_READ_A4_stress_delta = -6.3477371145
scalar_attention_mass_rows = 144
proxy_overlays_copied = 3
sky_causality_proven = false
evidence_level =
    candidate_level_metrics_plus_proxy_overlays;
    per-label spatial attention and affected masks missing
```

Boundary：

```text
本阶段补到了 scalar attention-mass rows，但仍没有 per-label spatial
attention / affected-mask tensor，因此 sky causality 不被证明。
```

---

## 2026-05-26 Phase 3：READ h10

### 10. 运行 h10 short rollout

```bash
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30 \
V36B_PHASE_NAME=phase3_read_h10/h10_R1 \
V36B_RUN_PREFIX_BASE=V41_READ_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,R1_READ_HIGH_INFLUENCE_ANOMALY,R2_READ_SKY_APP_ANOMALY,R3_READ_ANOMALY_PLUS_STATIC_RESCUE,R4_NEG_CONTROL_SKY_NO_SOURCE_MASS,R5_NEG_CONTROL_STATIC_RESCUE_ONLY \
V36B_CHUNKS=10 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3,4,5,6,7 \
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

结果：

```text
rows = 12/12 DONE
failures = 0
```

### 11. 生成 h10 durability report

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v38_durability_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/h10_R1/rollouts \
  --run-prefix V41_READ_H10_R1 \
  --parents H9,C9 \
  --chunks 10 \
  --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,R1_READ_HIGH_INFLUENCE_ANOMALY,R2_READ_SKY_APP_ANOMALY,R3_READ_ANOMALY_PLUS_STATIC_RESCUE,R4_NEG_CONTROL_SKY_NO_SOURCE_MASS,R5_NEG_CONTROL_STATIC_RESCUE_ONLY \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/report_h10_R1 \
  --report-prefix read_h10 \
  --mode h10
```

### 12. 生成 v41 h10 gate report

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_read_gate_report.py \
  --effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/report_h10_R1/read_h10_effects.csv \
  --mode h10 \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/report_h10_R1 \
  --report-prefix read_h10
```

输出：

```text
phase3_read_h10/report_h10_R1/read_h10_effects.csv
phase3_read_h10/report_h10_R1/read_h10_report.md
phase3_read_h10/report_h10_R1/read_h10_summary.json
phase3_read_h10/report_h10_R1/read_h10_v41_gate_report.md
phase3_read_h10/report_h10_R1/read_h10_v41_gate_summary.json
```

关键结果：

```text
v41 h10 gate_pass = true
best_ATE_candidate = R1_READ_HIGH_INFLUENCE_ANOMALY
best_ATE_parent = C9
best_ATE_delta_vs_base = -0.5994474373
best_rolling100_candidate = R4_NEG_CONTROL_SKY_NO_SOURCE_MASS
best_rolling100_delta = -3.3097470786
best_stress_candidate = R3_READ_ANOMALY_PLUS_STATIC_RESCUE
best_stress_delta = -8.5850808794
```

h10 gate pass rows：

```text
H9 R1_READ_HIGH_INFLUENCE_ANOMALY        reason = stress_window_with_downstream
H9 R3_READ_ANOMALY_PLUS_STATIC_RESCUE    reason = rolling100
H9 R4_NEG_CONTROL_SKY_NO_SOURCE_MASS     reason = rolling100
H9 R5_NEG_CONTROL_STATIC_RESCUE_ONLY     reason = rolling100
C9 R1_READ_HIGH_INFLUENCE_ANOMALY        reason = stress_window_with_downstream
C9 R4_NEG_CONTROL_SKY_NO_SOURCE_MASS     reason = rolling100
```

Boundary：

```text
R4/R5 are negative/control-like rows that pass rolling100 but have large
downstream regression. They are continued only because the v41 gate report
marks them as h10 pass rows; they are not interpreted as safe deployable rows.
```

---

## 2026-05-26 Phase 3：READ h15 continuation

### 13. 运行 H9 h15 rows

```bash
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30 \
V36B_PHASE_NAME=phase3_read_h15/h15_R1 \
V36B_RUN_PREFIX_BASE=V41_READ_H15_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,R1_READ_HIGH_INFLUENCE_ANOMALY,R3_READ_ANOMALY_PLUS_STATIC_RESCUE,R4_NEG_CONTROL_SKY_NO_SOURCE_MASS,R5_NEG_CONTROL_STATIC_RESCUE_ONLY \
V36B_CHUNKS=10 \
V36B_PARENTS=H9 \
V36B_HORIZON=15 \
V36B_GPUS=0,1,2,3,4 \
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

### 14. 运行 C9 h15 rows

```bash
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30 \
V36B_PHASE_NAME=phase3_read_h15/h15_R1 \
V36B_RUN_PREFIX_BASE=V41_READ_H15_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,R1_READ_HIGH_INFLUENCE_ANOMALY,R4_NEG_CONTROL_SKY_NO_SOURCE_MASS \
V36B_CHUNKS=10 \
V36B_PARENTS=C9 \
V36B_HORIZON=15 \
V36B_GPUS=5,6,7 \
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

结果：

```text
rows = 8/8 DONE
H9 rows = 5/5 DONE
C9 rows = 3/3 DONE
```

### 15. 生成 h15 durability / gate reports

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v38_durability_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/h15_R1/rollouts \
  --run-prefix V41_READ_H15_R1 \
  --parents H9 \
  --chunks 10 \
  --horizon 15 \
  --candidates V31_BASE_H9_REFERENCE,R1_READ_HIGH_INFLUENCE_ANOMALY,R3_READ_ANOMALY_PLUS_STATIC_RESCUE,R4_NEG_CONTROL_SKY_NO_SOURCE_MASS,R5_NEG_CONTROL_STATIC_RESCUE_ONLY \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_H9 \
  --report-prefix read_h15_H9 \
  --mode h15 \
  --h15-ate-threshold -1.5 \
  --h15-rolling100-threshold -3.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v38_durability_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/h15_R1/rollouts \
  --run-prefix V41_READ_H15_R1 \
  --parents C9 \
  --chunks 10 \
  --horizon 15 \
  --candidates V31_BASE_H9_REFERENCE,R1_READ_HIGH_INFLUENCE_ANOMALY,R4_NEG_CONTROL_SKY_NO_SOURCE_MASS \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_C9 \
  --report-prefix read_h15_C9 \
  --mode h15 \
  --h15-ate-threshold -1.5 \
  --h15-rolling100-threshold -3.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_read_gate_report.py \
  --effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_H9/read_h15_H9_effects.csv \
  --mode h15 \
  --h10-effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/report_h10_R1/read_h10_effects.csv \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_H9 \
  --report-prefix read_h15_H9

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_read_gate_report.py \
  --effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_C9/read_h15_C9_effects.csv \
  --mode h15 \
  --h10-effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/report_h10_R1/read_h10_effects.csv \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_C9 \
  --report-prefix read_h15_C9
```

关键结果：

```text
H9 h15 gate_pass = false
H9 best ATE delta = -0.1660176140
H9 best rolling100 delta = -1.6499290343
H9 best stress [200,300) delta = -4.7873967631

C9 h15 gate_pass = false
C9 best ATE delta = -0.1484353180
C9 best rolling100 delta = -1.6218203375
C9 best stress [200,300) delta = -4.8163673672
```

Decision：

```text
h15 gate = fail for H9 and C9.
No boundary h20 diagnostic was launched because h15 failed the primary signal
thresholds directly.
No full-online continuation is allowed.
```

---

## 2026-05-26 Phase 5：Washout / failure routing

### 16. 生成 proxy-only washout reports

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_read_washout_report.py \
  --h10-effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/report_h10_R1/read_h10_effects.csv \
  --h15-effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_H9/read_h15_H9_effects.csv \
  --parent H9 \
  --candidate R1_READ_HIGH_INFLUENCE_ANOMALY \
  --chunk 10 \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase5_washout/R1_H9

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_read_washout_report.py \
  --h10-effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/report_h10_R1/read_h10_effects.csv \
  --h15-effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_C9/read_h15_C9_effects.csv \
  --parent C9 \
  --candidate R1_READ_HIGH_INFLUENCE_ANOMALY \
  --chunk 10 \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase5_washout/R1_C9

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_read_washout_report.py \
  --h10-effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h10/report_h10_R1/read_h10_effects.csv \
  --h15-effects results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase3_read_h15/report_h15_R1_H9/read_h15_H9_effects.csv \
  --parent H9 \
  --candidate R3_READ_ANOMALY_PLUS_STATIC_RESCUE \
  --chunk 10 \
  --out-dir results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/phase5_washout/R3_H9
```

输出：

```text
phase5_washout/R1_H9/read_washout_summary.json
phase5_washout/R1_C9/read_washout_summary.json
phase5_washout/R3_H9/read_washout_summary.json
```

关键结果：

```text
evidence_level = proxy_only_no_tensor_state_snapshots

R1 H9:
    stress_200_300_delta durability = 0.8911326709
    ATE_delta durability = 0.2773474850
    rolling100_best_delta durability = 0.8577451063
    ttt_state tail/h10 proxy = 0.4259168776

R1 C9:
    stress_200_300_delta durability = 0.8861286118
    ATE_delta durability = 0.2476202392
    rolling100_best_delta durability = 0.8360702240
    ttt_state tail/h10 proxy = 0.4238951346

R3 H9:
    stress_200_300_delta durability = 0.5129897295
    ATE_delta durability = 2.6045714512
    rolling100_best_delta durability = 0.1205039811
    ttt_state tail/h10 proxy = 0.4393180545
```

Boundary：

```text
Washout evidence is JSONL proxy-only.
No tensor-state overwrite norm or true memory-state causality is claimed.
```

---

## 2026-05-26 Final reports

### 17. 生成 final summary

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v41_final_summary_report.py \
  --root results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30
```

输出：

```text
final_reports/health_detector_report.md
final_reports/read_mechanism_report.md
final_reports/read_h10_candidate_report.md
final_reports/read_h15_candidate_report.md
final_reports/washout_report.md
final_reports/full_online_report.md
final_reports/failure_routing_summary.md
final_reports/v41_final_summary.json
```

Final summary：

```text
phase1_health_detector_gate_pass = true
selected_bad_chunks = [10]
mechanism_decision = B_general_high_influence_anomaly_preferred
sky_causality_proven = false
scalar_attention_mass_rows = 144
h10_gate_pass = true
h10_best_ATE_delta = -0.5994474373
h10_best_rolling100_delta = -3.3097470786
h10_best_stress_delta = -8.5850808794
h15_H9_gate_pass = false
h15_H9_best_ATE_delta = -0.1660176140
h15_H9_best_rolling100_delta = -1.6499290343
h15_H9_best_stress_delta = -4.7873967631
h15_C9_gate_pass = false
h15_C9_best_ATE_delta = -0.1484353180
h15_C9_best_rolling100_delta = -1.6218203375
h15_C9_best_stress_delta = -4.8163673672
full_online_allowed = false
full_online_launched = false
target30_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029
```

Decision：

```text
v41 completed.
No full-online row was launched because h15 gate failed.
No Target-30 result was produced.
No new deployable online success is produced.
```
