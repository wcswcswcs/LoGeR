# ACL2 v40 执行日志：QualityGated SemanticGeometry MemoryController Target30

日期：2026-05-25（Asia/Singapore）

计划文件：

```text
docs/ACL2_v40_QualityGated_SemanticGeometry_MemoryController_Target30_Plan.md
```

主结果目录：

```text
results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30/
```

原则：

```text
1. 只记录实际运行命令和落盘结果。
2. 不把 no-op health audit、short rollout、proxy visualization、blocked stage 写成 deployable online success。
3. 每个 LoGeR 进程只绑定一张 GPU。
4. 用户指定 GPU 0,1,2,3 可用；GPU 4,5,6,7 当前被其他任务占用，不使用。
```

---

## 2026-05-25 20:34 setup / code validation

Commands:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v40_health_atlas_report.py \
  tools/v40_final_summary_report.py \
  tools/v38_durability_report.py \
  tools/v39_semantic_appearance_atlas.py

bash -n \
  tools/run_attention_cue_experiment.sh \
  tools/run_v24_candidate_rollout.sh \
  tools/run_v36b_path_h10.sh

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

mkdir -p results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
if [ ! -e results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30/phase0_parent_snapshots ]; then
  ln -s ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots \
    results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30/phase0_parent_snapshots
fi
```

Output summary:

```text
py_compile = PASS
bash -n = PASS

GPU status:
    0, 1 MiB, 0%
    1, 1 MiB, 0%
    2, 1 MiB, 0%
    3, 4 MiB, 0%
    4, 19334 MiB, 38%
    5, 20196 MiB, 72%
    6, 20148 MiB, 100%
    7, 19070 MiB, 100%

phase0_parent_snapshots symlink created/reused:
    results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30/phase0_parent_snapshots
```

Code changes validated:

```text
tools/run_attention_cue_experiment.sh:
    forwarded existing runtime read-quality / gram / beta-policy arguments:
        READ_QUALITY_MASS_MIN
        READ_QUALITY_MASS_MAX
        READ_QUALITY_ANCHOR_MAX
        READ_QUALITY_FRAG_MAX
        GRAM_LAYER_GROUPS
        BETA_POLICY
        BETA_ENERGY_TARGET
        BETA_MIN
        BETA_MAX

tools/run_v24_candidate_rollout.sh:
    added v40 Phase 0 / Phase 1 no-op health aliases.
    added v40 READ_A / SWA_B / TTT_C training-free aliases.
    added run_config fields for read cue / quality thresholds / beta policy.

tools/v40_health_atlas_report.py:
    new landed-artifact-only no-op and health atlas report.

tools/v40_final_summary_report.py:
    new landed-artifact-only final report aggregation.
```

---

## 2026-05-25 20:35 Phase 0 no-op health gate launched

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
V36B_RESULT_ROOT="$V40_ROOT" \
V36B_PHASE_NAME=phase0_health/h3_R1 \
V36B_RUN_PREFIX_BASE=V40_PHASE0_H3_R1 \
V36B_CANDIDATES=P0_00_C9_REFERENCE,P0_01_HEALTH_LOGGING_ONLY,P0_02_SEMANTIC_PASSIVE_ONLY,P0_03_APPEARANCE_AUDIT_ONLY \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=3 \
V36B_GPUS=0,1,2,3 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

---

## 2026-05-26 04:38 Phase 2C TTT h10 postprocess

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V40_ROOT/phase2c_ttt/h10_R1/rollouts" \
  --run-prefix V40_TTT_H10_R1 \
  --parents H9,C9 --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,TTT_C1_STRUCTURE_LOW_RISK_POSITIVE_LONG,TTT_C2_DYNAMIC_HIGH_RISK_NO_LONG_WRITE,TTT_C3_VEGETATION_HIGH_D_CONFLICT_SHORT_NEGATIVE,TTT_C4_SKY_ANOMALY_NO_POSITIVE_LONG_NEUTRAL,TTT_C5_COMBINED_LIFECYCLE,TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL \
  --out-dir "$V40_ROOT/phase2c_ttt/report_h10_R1" \
  --report-prefix ttt_h10 --mode h10
```

Output summary:

```text
all_rows_done = true
rows = 42
missing_rows = 0
gate_pass = false
best_ATE_candidate = TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL
best_ATE_parent = H9
best_ATE_chunk = 10
best_ATE_delta_vs_base = -0.2368777135m
best_rolling_100f_candidate = TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL
best_rolling_100f_parent = C9
best_rolling_100f_chunk = 6
best_rolling_100f_best_delta = -1.5074228818m
best_downstream_400_600_delta_for_best_ATE = -0.0530026027m
H9 best [200,300) delta = -0.8702571558m
C9 best [200,300) delta = -0.8097502047m
```

Decision:

```text
TTT h10 gate = fail.
No TTT h15 continuation is allowed.
```

---

## 2026-05-26 04:39 Final summary report

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v40_final_summary_report.py --root "$V40_ROOT"
```

Output summary:

```text
phase0_gate_pass = true
read_h10_gate_pass = false
read_h15_gate_pass = null
swa_h10_gate_pass = false
swa_h15_gate_pass = null
ttt_h10_gate_pass = false
ttt_h15_gate_pass = null
reset_h10_gate_pass = null
full_online_allowed = false
full_online_launched = false
target30_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029
```

Decision:

```text
Phase 2D RESET was not launched because the plan only allows it under severe
global-state health flag, and no such landed flag exists in the v40 aggregate
health atlas.

No Phase 2 h10 candidate passed continuation gate.
No h15 row was launched.
No Phase 3 minimal combination was allowed.
No Phase 4 full-online row was launched.
```

---

## 2026-05-26 03:43 Phase 2C TTT second single-row supplemental acceleration on GPU 2

Context:

```text
The C9 chunk10 TTT_C6 supplemental row landed DONE at 2026-05-26 03:43.
The official TTT queue was still on C9 chunk6.
GPU 2 was reused for one additional far-future C9 chunk10 row.
```

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
V36B_RESULT_ROOT="$V40_ROOT" \
V36B_PHASE_NAME=phase2c_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V40_TTT_H10_R1 \
V36B_CANDIDATES=TTT_C5_COMBINED_LIFECYCLE \
V36B_CHUNKS=10 \
V36B_PARENTS=C9 \
V36B_HORIZON=10 \
V36B_GPUS=2 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

---

## 2026-05-26 03:18 Phase 2B SWA h10 postprocess

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V40_ROOT/phase2b_swa/h10_R1/rollouts" \
  --run-prefix V40_SWA_H10_R1 \
  --parents H9,C9 --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP,SWA_B2_OVERLAP_K_PRESERVE_V_ATTEN_RISKY,SWA_B3_STRUCTURE_OVERLAP_ANCHOR_PROTECT,SWA_B4_SKY_HORIZON_NEUTRAL_K_KEEP_V_ATTEN_IF_ANOMALOUS,SWA_B5_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_OVERLAP_PROTECT \
  --out-dir "$V40_ROOT/phase2b_swa/report_h10_R1" \
  --report-prefix swa_h10 --mode h10
```

Output summary:

```text
all_rows_done = true
rows = 36
missing_rows = 0
gate_pass = false
best_ATE_candidate = SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP
best_ATE_parent = C9
best_ATE_chunk = 6
best_ATE_delta_vs_base = -0.7081471064m
best_rolling_100f_candidate = SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP
best_rolling_100f_parent = C9
best_rolling_100f_chunk = 6
best_rolling_100f_best_delta = -1.3354202480m
best_downstream_400_600_delta_for_best_ATE = +0.4147398163m
H9 best [200,300) delta = -2.3916353409m
C9 best [200,300) delta = -2.3564699828m
```

Decision:

```text
SWA h10 gate = fail.
No SWA h15 continuation is allowed.
```

---

## 2026-05-26 03:17 Phase 2C TTT single-row supplemental acceleration on GPU 2

Context:

```text
Phase2B SWA exited with code 0 at 2026-05-26 03:16.
GPU 2 was released.
The official Phase2C TTT queue was still on C9 chunk6.
To avoid broad duplicate scheduling, launch only one far-future row:
    parent = C9
    chunk = 10
    candidate = TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL
    same phase/prefix
    FORCE = 0
```

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
V36B_RESULT_ROOT="$V40_ROOT" \
V36B_PHASE_NAME=phase2c_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V40_TTT_H10_R1 \
V36B_CANDIDATES=TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL \
V36B_CHUNKS=10 \
V36B_PARENTS=C9 \
V36B_HORIZON=10 \
V36B_GPUS=2 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Expected rows:

```text
parents = H9,C9
chunks = 6,10,16
candidates = 4
rows = 24
GPU = 0,1,2,3
```

Result:

```text
rows = 6/6
launcher failures = 0
```

Postprocess command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v40_health_atlas_report.py \
  --root "$V40_ROOT" \
  --phase0-rollout-root "$V40_ROOT/phase0_health/h3_R1/rollouts" \
  --phase0-prefix V40_PHASE0_H3_R1 \
  --phase0-candidates P0_00_C9_REFERENCE,P0_01_HEALTH_LOGGING_ONLY,P0_02_SEMANTIC_PASSIVE_ONLY,P0_03_APPEARANCE_AUDIT_ONLY \
  --phase1-rollout-root "$V40_ROOT/phase1_passive_health/h10_R1/rollouts" \
  --phase1-prefix V40_PHASE1_H10_R1 \
  --phase1-candidates P1_00_HEALTH_LOGGING_ONLY \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 3 \
  --phase1-horizon 10 \
  --out-dir "$V40_ROOT/phase1_passive_health/report_R1"
```

Postprocess output:

```text
phase0_gate_pass = true
health_rows_total = 30
cue_quality_rows_total = 162
source_influence_rows_total = 162
swa_health_rows_total = 162
ttt_health_rows_total = 162
context_empty_source_events_total = 0.0
attention_mass_rows_total = 0
appearance_evidence_level = explainability_missing_runtime_rollout_spatial_tensors
```

---

## 2026-05-25 21:10 Phase 2A READ h10 launched

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
V36B_RESULT_ROOT="$V40_ROOT" \
V36B_PHASE_NAME=phase2a_read/h10_R1 \
V36B_RUN_PREFIX_BASE=V40_READ_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,READ_A1_HIGH_INFLUENCE_ANOMALY_V_ATTEN,READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT,READ_A3_DYNAMIC_VEG_SHADOW_HIGHD_SKIP_STRUCT_RESCUE,READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH,READ_A5_STATIC_ANCHOR_RESCUE_ONLY \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Expected rows:

```text
parents = H9,C9
chunks = 6,10,16
candidates = 6
rows = 36
GPU = 0,1
```

## 2026-05-25 21:11 Phase 2B SWA h10 launched

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
V36B_RESULT_ROOT="$V40_ROOT" \
V36B_PHASE_NAME=phase2b_swa/h10_R1 \
V36B_RUN_PREFIX_BASE=V40_SWA_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP,SWA_B2_OVERLAP_K_PRESERVE_V_ATTEN_RISKY,SWA_B3_STRUCTURE_OVERLAP_ANCHOR_PROTECT,SWA_B4_SKY_HORIZON_NEUTRAL_K_KEEP_V_ATTEN_IF_ANOMALOUS,SWA_B5_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_OVERLAP_PROTECT \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=2,3 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Expected rows:

```text
parents = H9,C9
chunks = 6,10,16
candidates = 6
rows = 36
GPU = 2,3
```

Result:

```text
rows = 24/24
launcher failures = 0
```

Postprocess command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v40_health_atlas_report.py \
  --root "$V40_ROOT" \
  --phase0-rollout-root "$V40_ROOT/phase0_health/h3_R1/rollouts" \
  --phase0-prefix V40_PHASE0_H3_R1 \
  --phase0-candidates P0_00_C9_REFERENCE,P0_01_HEALTH_LOGGING_ONLY,P0_02_SEMANTIC_PASSIVE_ONLY,P0_03_APPEARANCE_AUDIT_ONLY \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 3 \
  --out-dir "$V40_ROOT/phase0_health/report_R1"
```

Postprocess output:

```text
phase0_rows_done = 24
phase0_missing_rows = 0
phase0_gate_pass = true
max_abs_ATE_delta_vs_noop_reference = 0.0
max_raw_pose_abs_diff_vs_noop_reference = 0.0
required_health_streams_nonempty = true
context_empty_source_events_total = 0.0
cue_quality_rows_total = 96
source_influence_rows_total = 96
swa_health_rows_total = 96
ttt_health_rows_total = 96
attention_mass_rows_total = 0
appearance_evidence_level = explainability_missing_runtime_rollout_spatial_tensors
```

---

## 2026-05-25 20:55 Phase 1 passive health atlas launched

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
V36B_RESULT_ROOT="$V40_ROOT" \
V36B_PHASE_NAME=phase1_passive_health/h10_R1 \
V36B_RUN_PREFIX_BASE=V40_PHASE1_H10_R1 \
V36B_CANDIDATES=P1_00_HEALTH_LOGGING_ONLY \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Expected rows:

```text
parents = H9,C9
chunks = 6,10,16
candidates = 1
rows = 6
GPU = 0,1,2,3
```

---

## 2026-05-25 23:45 Phase 2C TTT h10 launched after READ released GPU 0/1

Context:

```text
Phase2A READ queue exited with code 0 at 2026-05-25 23:42.
Phase2B SWA was still running on GPU 2,3.
GPU 0,1 were reused for Phase2C TTT to keep the user-approved GPU set busy.
No duplicate RUN_NAME overlap with READ/SWA phases.
```

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
V36B_RESULT_ROOT="$V40_ROOT" \
V36B_PHASE_NAME=phase2c_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V40_TTT_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,TTT_C1_STRUCTURE_LOW_RISK_POSITIVE_LONG,TTT_C2_DYNAMIC_HIGH_RISK_NO_LONG_WRITE,TTT_C3_VEGETATION_HIGH_D_CONFLICT_SHORT_NEGATIVE,TTT_C4_SKY_ANOMALY_NO_POSITIVE_LONG_NEUTRAL,TTT_C5_COMBINED_LIFECYCLE,TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Expected rows:

```text
parents = H9,C9
chunks = 6,10,16
candidates = 7
rows = 42
GPU = 0,1
```

---

## 2026-05-25 23:49 Phase 2A READ h10 postprocess

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V40_ROOT/phase2a_read/h10_R1/rollouts" \
  --run-prefix V40_READ_H10_R1 \
  --parents H9,C9 --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,READ_A1_HIGH_INFLUENCE_ANOMALY_V_ATTEN,READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT,READ_A3_DYNAMIC_VEG_SHADOW_HIGHD_SKIP_STRUCT_RESCUE,READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH,READ_A5_STATIC_ANCHOR_RESCUE_ONLY \
  --out-dir "$V40_ROOT/phase2a_read/report_h10_R1" \
  --report-prefix read_h10 --mode h10
```

Output summary:

```text
all_rows_done = true
rows = 36
missing_rows = 0
gate_pass = false
best_ATE_candidate = READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT
best_ATE_parent = C9
best_ATE_chunk = 6
best_ATE_delta_vs_base = -1.3698298799m
best_rolling_100f_candidate = READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH
best_rolling_100f_parent = C9
best_rolling_100f_chunk = 6
best_rolling_100f_best_delta = -3.4811567463m
best_downstream_400_600_delta_for_best_ATE = +0.7460894838m
H9 best [200,300) delta = -6.3477371145m
C9 best [200,300) delta = -6.3241080599m
```

Decision:

```text
READ h10 is a strong local diagnostic but fails the v40 continuation gate.
No READ h15 continuation is allowed from this report.
```

---

## 2026-05-26 02:19 Phase 2C TTT supplemental acceleration on GPU 3

Context:

```text
Phase2B SWA was still running one long C9 chunk10 row on GPU 2.
GPU 3 was idle.
The official Phase2C TTT queue was still on H9 chunk16, far before C9 chunk16.
To use the user-approved GPU set without colliding with active RUN_NAME rows,
launch a far-future supplemental TTT subset:
    parent = C9
    chunk = 16
    same phase/prefix
    FORCE = 0

If the supplemental rows finish before the official queue reaches C9 chunk16,
the official queue will skip DONE rows. If any row is not DONE, only landed
DONE artifacts are counted.
```

Command:

```bash
V40_ROOT=results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30
V36B_RESULT_ROOT="$V40_ROOT" \
V36B_PHASE_NAME=phase2c_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V40_TTT_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,TTT_C1_STRUCTURE_LOW_RISK_POSITIVE_LONG,TTT_C2_DYNAMIC_HIGH_RISK_NO_LONG_WRITE,TTT_C3_VEGETATION_HIGH_D_CONFLICT_SHORT_NEGATIVE,TTT_C4_SKY_ANOMALY_NO_POSITIVE_LONG_NEUTRAL,TTT_C5_COMBINED_LIFECYCLE,TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL \
V36B_CHUNKS=16 \
V36B_PARENTS=C9 \
V36B_HORIZON=10 \
V36B_GPUS=3 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```
