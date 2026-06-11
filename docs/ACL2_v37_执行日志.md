# ACL2 v37 执行日志：TrainingFree SemanticInfluence MemorySurgery Target30

日期：2026-05-24（Asia/Singapore）

计划文件：`docs/ACL2_v37_TrainingFree_SemanticInfluence_MemorySurgery_Target30_Plan.md`

实验结果复盘：`docs/ACL2_v37_实验复盘.md`

主结果目录：`results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/`

原则：

```text
1. 不训练 trigger / selector / router / classifier。
2. 不使用 GT runtime action。
3. 不使用 absolute chunk id 作为 deployable runtime 条件。
4. 不把 short rollout、fixed diagnostic、instrumentation smoke 写成 deployable online success。
5. 每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立 row。
6. 所有 blocker、修复尝试、落盘命令和复盘结论必须可审计。
```

---

## 0. 初始化

读取计划：

```bash
sed -n '1,260p' docs/ACL2_v37_TrainingFree_SemanticInfluence_MemorySurgery_Target30_Plan.md
sed -n '260,620p' docs/ACL2_v37_TrainingFree_SemanticInfluence_MemorySurgery_Target30_Plan.md
sed -n '620,980p' docs/ACL2_v37_TrainingFree_SemanticInfluence_MemorySurgery_Target30_Plan.md
```

创建结果目录：

```bash
mkdir -p results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/{phase0_action_influence,phase1_frameglobal,phase2_swa,phase3_ttt,phase4_semc23,phase5_full_online,matrix_logs}
```

GPU preflight：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Observed：

```text
0, 1, 23028, 0
1, 1, 23028, 0
2, 1, 23028, 0
3, 4, 23028, 0
4, 4, 23028, 0
5, 4, 23028, 0
6, 1, 23028, 0
7, 4, 23028, 0
```

Decision：

```text
GPU 0-7 当前可用于独立 row 并行；仍保持一进程一 GPU。
```

Parent snapshot reuse：

```bash
ln -sfn ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots \
  results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase0_parent_snapshots
```

Boundary：

```text
v37 reuses the landed v36B H9/C9 parent state + merge snapshots for chunks 6,10,16.
No synthetic parent snapshot was created.
```

Runtime semantic source audit：

```bash
find results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase0_action_influence/smoke_R1/rollouts \
  -path '*semantic_group_summary.jsonl' | head -2 | \
  xargs -r -I{} sh -c 'echo --- {}; head -2 {}'
```

Observed source fields：

```text
fine_label_source = MaskletOutput.L_sem
semantic_group_source = MaskletOutput.G_sem
semantic_group_taxonomy = stage_c_coarse_5_groups
```

Boundary：

```text
v37 runtime semantic policies use VideoMasklet frontend / Stage-C semantic
cache outputs, not GT SemanticKITTI labels.

Projected SemanticKITTI / KITTI 3D labels from v29C are offline audit and trust
calibration evidence only; they are not used as runtime semantic action labels
in these v37 rollout commands.
```

---

## 1. Track 0 Smoke R1

Purpose：

```text
Run h3 action/influence smoke over H9/C9 parents and chunks 6,10,16.
This is not trajectory evidence; it is for H0A/H0B/H0C action realism and
source influence audit.
```

Execution：

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase0_action_influence/smoke_R1 \
V36B_RUN_PREFIX_BASE=V37_TRACK0_SMOKE_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_RISK_01_CONFLICT_TRI \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=3 \
V36B_GPUS=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

---

## 8. Final Completion Addendum

This section records the final commands/results after the long-running Track 3
and Track 4 queues completed. Earlier R1/R2 partial notes above are retained as
audit trail; this section is the authoritative completion state.

### Track 3 h10 R2 reports

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase3_ttt/h10_R2/rollouts \
  --run-prefix V37_T3_TTT_H10_R2_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_03_SCALE_CONDITIONED,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_05_STRUCTURE_PROTECT,TTT_FINE_RISK_01_CONFLICT_TRI,TTT_FINE_RISK_02_SCALE_STATE,TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase3_ttt/report_h10_R2_H9 \
  --report-prefix track3_h10_H9 \
  --ate-threshold -1.5 --segment-threshold -3.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase3_ttt/h10_R2/rollouts \
  --run-prefix V37_T3_TTT_H10_R2_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_03_SCALE_CONDITIONED,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_05_STRUCTURE_PROTECT,TTT_FINE_RISK_01_CONFLICT_TRI,TTT_FINE_RISK_02_SCALE_STATE,TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase3_ttt/report_h10_R2_C9 \
  --report-prefix track3_h10_C9 \
  --ate-threshold -1.5 --segment-threshold -3.0 --downstream-threshold 1.0
```

Result:

```text
Track3 h10 R2:
    rows = 48/48
    failures = 0
    H9 gate_pass = true
    H9 best = TTT_FINE_RISK_02_SCALE_STATE chunk10
    H9 ATE delta = -1.8431656072m
    H9 [200,300) delta = -1.7711993876m
    H9 [400,600) delta = -2.5522426503m
    C9 gate_pass = true
    C9 best = TTT_FINE_RISK_02_SCALE_STATE chunk10
    C9 ATE delta = -1.9880251151m
    C9 [200,300) delta = -1.9522337557m
    C9 [400,600) delta = -2.7310700062m
```

### Track 3 h15 R1 command and reports

Triggered because Track3 h10 passed under both H9 and C9.

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase3_ttt/h15_R1 \
V36B_RUN_PREFIX_BASE=V37_T3_TTT_H15_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,TTT_FINE_RISK_02_SCALE_STATE \
V36B_CHUNKS=10 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=15 \
V36B_GPUS=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase3_ttt/h15_R1/rollouts \
  --run-prefix V37_T3_TTT_H15_R1_H9 \
  --chunks 10 --horizon 15 \
  --candidates V31_BASE_H9_REFERENCE,TTT_FINE_RISK_02_SCALE_STATE \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase3_ttt/report_h15_R1_H9 \
  --report-prefix track3_h15_H9 \
  --ate-threshold -3.0 --segment-threshold -5.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase3_ttt/h15_R1/rollouts \
  --run-prefix V37_T3_TTT_H15_R1_C9 \
  --chunks 10 --horizon 15 \
  --candidates V31_BASE_H9_REFERENCE,TTT_FINE_RISK_02_SCALE_STATE \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase3_ttt/report_h15_R1_C9 \
  --report-prefix track3_h15_C9 \
  --ate-threshold -3.0 --segment-threshold -5.0 --downstream-threshold 1.0
```

Result:

```text
Track3 h15 R1:
    rows = 4/4
    failures = 0
    H9 gate_pass = false
    H9 ATE delta = -0.9431819076m
    H9 [200,300) delta = -1.5502750675m
    H9 [400,600) delta = -1.6984935722m
    C9 gate_pass = false
    C9 ATE delta = -1.0265231323m
    C9 [200,300) delta = -1.7469440517m
    C9 [400,600) delta = -1.8278025868m
```

### Track 4 h10 R1 reports

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase4_semc23/h10_R1/rollouts \
  --run-prefix V37_T4_SEMC23_H10_R1_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,SEM_C23_01_READ_ONLY_RESID,SEM_C23_02_NO_TTT,SEM_C23_03_NO_SWA,SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY,SEM_C23_05_STATIC_RESCUE_RESID \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase4_semc23/report_h10_R1_H9 \
  --report-prefix track4_h10_H9 \
  --ate-threshold -999 --segment-threshold -5.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase4_semc23/h10_R1/rollouts \
  --run-prefix V37_T4_SEMC23_H10_R1_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,SEM_C23_01_READ_ONLY_RESID,SEM_C23_02_NO_TTT,SEM_C23_03_NO_SWA,SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY,SEM_C23_05_STATIC_RESCUE_RESID \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase4_semc23/report_h10_R1_C9 \
  --report-prefix track4_h10_C9 \
  --ate-threshold -999 --segment-threshold -5.0 --downstream-threshold 1.0
```

Result:

```text
Track4 h10 R1:
    rows = 36/36
    failures = 0
    H9 gate_pass = false
    H9 best candidate = SEM_C23_02_NO_TTT
    H9 best [200,300) delta = -1.2784762533m
    C9 gate_pass = false
    C9 best candidate = SEM_C23_02_NO_TTT
    C9 best [200,300) delta = -1.2952954518m
```

### Final report generation

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v37_final_summary_report.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v37_final_summary_report.py \
  --root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30
```

Generated:

```text
final_reports/track1_frameglobal_report.md
final_reports/track2_swa_report.md
final_reports/track3_ttt_report.md
final_reports/track4_semantic_c23_report.md
final_reports/track5_full_online_report.md
final_reports/failure_routing_summary.md
final_reports/segment_ate_bar_chart.png
final_reports/h10_h15_durability_curve.png
final_reports/v37_final_summary.json
```

Final state:

```text
track0_gate_pass = true
track1_gate_pass = false
track2_h10_gate_pass = true
track2_h15_gate_pass = false
track3_h10_gate_pass = true
track3_h15_gate_pass = false
track4_h10_gate_pass = false
track5_full_online_allowed = false
track5_full_online_launched = false
target30_success = false
```

Track 1 reports:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/h10_R1/rollouts \
  --run-prefix V37_T1_FG_H10_R1_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05 \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/report_h10_R1_H9 \
  --report-prefix track1_h10_H9 \
  --ate-threshold -1.5 --segment-threshold -5.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/h10_R1/rollouts \
  --run-prefix V37_T1_FG_H10_R1_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05 \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/report_h10_R1_C9 \
  --report-prefix track1_h10_C9 \
  --ate-threshold -1.5 --segment-threshold -5.0 --downstream-threshold 1.0
```

Track 1 status:

```text
rows = 30/30
failures = 0
H9 gate_pass = false; best [200,300) delta = -0.3418221710m
C9 gate_pass = false; best [200,300) delta = -0.2606290156m
No Track 1 h15 launched.
```

---

## 3. Track 2 / Track 3 R1 Invalidated

Blocker:

```text
Track 2 / Track 3 R1 were active while Track 4 alias repair edited
tools/run_v24_candidate_rollout.sh. Some still-running bash instances later
misrouted V37_T2/V37_T3 rows to:
    results/kitti01_hmc_v2/attention_cue_library_v1/
with default full-sequence/default Stage-C settings.
```

Repair:

```text
1. Stop Track 2 / Track 3 R1 schedulers and child rows.
2. Move intended-root partial rows:
       phase2_swa/h10_R1_20260524T1620_launcher_edit_invalid
       phase3_ttt/h10_R1_20260524T1620_launcher_edit_invalid
3. Move misrouted default-root rows:
       invalid_misrouted_attention_cue_library_R1/
4. Finish launcher edits and validate bash -n before R2 launch.
```

Boundary:

```text
Track 2 R1 and Track 3 R1 are invalid / partial.
No R1 row is used for any gate or conclusion.
```

---

## 4. Track 2 SWA h10/h15 R2/R1

Track 2 h10 R2 command:

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase2_swa/h10_R2 \
V36B_RUN_PREFIX_BASE=V37_T2_SWA_H10_R2 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_02_SKY_PARTIAL_KEEP,SWA_FINE_03_VEGETATION_CONDITIONAL,SWA_FINE_04_BOUNDARY_PROTECT,SWA_FINE_05_CACHE_LIFECYCLE \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

Track 2 h10 R2 report commands:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase2_swa/h10_R2/rollouts \
  --run-prefix V37_T2_SWA_H10_R2_H9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_02_SKY_PARTIAL_KEEP,SWA_FINE_03_VEGETATION_CONDITIONAL,SWA_FINE_04_BOUNDARY_PROTECT,SWA_FINE_05_CACHE_LIFECYCLE \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase2_swa/report_h10_R2_H9 \
  --report-prefix track2_h10_H9 \
  --ate-threshold -1.0 --segment-threshold -3.0 --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase2_swa/h10_R2/rollouts \
  --run-prefix V37_T2_SWA_H10_R2_C9 \
  --chunks 6,10,16 --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_02_SKY_PARTIAL_KEEP,SWA_FINE_03_VEGETATION_CONDITIONAL,SWA_FINE_04_BOUNDARY_PROTECT,SWA_FINE_05_CACHE_LIFECYCLE \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase2_swa/report_h10_R2_C9 \
  --report-prefix track2_h10_C9 \
  --ate-threshold -1.0 --segment-threshold -3.0 --downstream-threshold 1.0
```

Track 2 h10 R2 summary:

```text
rollout dirs = 36
DONE = 36
FAIL = 0

H9 gate_pass = true
H9 best ATE delta = -1.2136622509m, SWA_FINE_01, chunk6
H9 best [200,300) delta = -3.7351609926m, SWA_FINE_01, chunk10
H9 best [400,600) delta for best ATE = +0.9505750525m

C9 gate_pass = true
C9 best ATE delta = -1.2466940180m, SWA_FINE_01, chunk6
C9 best [200,300) delta = -3.4943516930m, SWA_FINE_01, chunk10
C9 best [400,600) delta for best ATE = +0.9348534197m
```

Boundary diagnostics were generated with `tools/v27_swa_boundary_diagnostics.py`
for passing candidates `SWA_FINE_01`, `SWA_FINE_02`, `SWA_FINE_04` on chunks 6
and 10 under H9/C9:

```text
phase2_swa/boundary_R2_H9_chunk6/
phase2_swa/boundary_R2_H9_chunk10/
phase2_swa/boundary_R2_C9_chunk6/
phase2_swa/boundary_R2_C9_chunk10/
```

Boundary summary:

```text
H9 chunk6 best boundary_10f delta = -0.5939405842m
H9 chunk10 best boundary_10f delta = +0.0723922636m
C9 chunk6 best boundary_10f delta = -0.6149793460m
C9 chunk10 best boundary_10f delta = +0.2221317699m
```

Track 2 h15 R1 command:

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase2_swa/h15_R1 \
V36B_RUN_PREFIX_BASE=V37_T2_SWA_H15_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_02_SKY_PARTIAL_KEEP,SWA_FINE_04_BOUNDARY_PROTECT \
V36B_CHUNKS=6,10 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=15 \
V36B_GPUS=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

Track 2 h15 R1 summary:

```text
rollout dirs = 16
DONE = 16
FAIL = 0

H9 h15 gate_pass = false
H9 best ATE delta = -0.8292622200m, SWA_FINE_01, chunk6
H9 best [200,300) delta = -2.2466407151m, SWA_FINE_01, chunk6
H9 chunk10 best [200,300) delta = -1.1991180979m

C9 h15 gate_pass = false
C9 best ATE delta = -0.8404528162m, SWA_FINE_01, chunk6
C9 best [200,300) delta = -2.2988546997m, SWA_FINE_01, chunk6
C9 chunk10 best [200,300) delta = -0.8636036060m
```

Washout attribution command pattern:

```bash
for parent in H9 C9; do
  for chunk in 6 10; do
    for cand in SWA_FINE_01_OVERLAP_STRUCTURE_KEEP SWA_FINE_02_SKY_PARTIAL_KEEP SWA_FINE_04_BOUNDARY_PROTECT; do
      /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_f_washout_attribution.py \
        --h10-run-dir "$h10_root/V37_T2_SWA_H10_R2_${parent}_${cand}_chunk${chunk}_h10_globalgate_H9parent_SWKS3" \
        --h15-run-dir "$h15_root/V37_T2_SWA_H15_R1_${parent}_${cand}_chunk${chunk}_h15_globalgate_H9parent_SWKS3" \
        --h10-report-csv "$base/phase2_swa/report_h10_R2_${parent}/track2_h10_${parent}_effects.csv" \
        --h15-report-csv "$base/phase2_swa/report_h15_R1_${parent}/track2_h15_${parent}_effects.csv" \
        --candidate "$cand" --chunk "$chunk" \
        --out-dir "$base/phase2_swa/washout_R1/${parent}_chunk${chunk}_${cand}"
    done
  done
done
```

Washout summary:

```text
reports = 12
evidence_level = proxy_only_no_tensor_state_snapshots
C9 chunk10 SWA_FINE_01 h15/h10 [200,300) durability = 0.2471427261
C9 chunk6 SWA_FINE_01 h15/h10 [200,300) durability = 0.8868817382
```

Decision:

```text
Track 2 has h10 local signal but fails h15 durability.
No Track 2 full-online continuation is allowed.
```

---

## 5. Track 3 TTT h10 R2

Track 3 h10 R2 command:

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase3_ttt/h10_R2 \
V36B_RUN_PREFIX_BASE=V37_T3_TTT_H10_R2 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_03_SCALE_CONDITIONED,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_05_STRUCTURE_PROTECT,TTT_FINE_RISK_01_CONFLICT_TRI,TTT_FINE_RISK_02_SCALE_STATE,TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

Current status at this log update:

```text
Track 3 h10 R2 still running.
Partial status observed: dirs = 45, DONE = 44, FAIL = 0.
No Track 3 metric is claimed yet.
```

---

## 6. Track 4 Semantic C23 h10 R1

Track 4 aliases added to `tools/run_v24_candidate_rollout.sh`:

```text
SEM_C23_01_READ_ONLY_RESID
SEM_C23_02_NO_TTT
SEM_C23_03_NO_SWA
SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY
SEM_C23_05_STATIC_RESCUE_RESID
```

Validation:

```bash
bash -n tools/run_v24_candidate_rollout.sh tools/run_v36b_path_h10.sh
```

Track 4 h10 R1 command:

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase4_semc23/h10_R1 \
V36B_RUN_PREFIX_BASE=V37_T4_SEMC23_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,SEM_C23_01_READ_ONLY_RESID,SEM_C23_02_NO_TTT,SEM_C23_03_NO_SWA,SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY,SEM_C23_05_STATIC_RESCUE_RESID \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

Current status at this log update:

```text
Track 4 h10 R1 is running.
No Track 4 metric is claimed yet.
```

Completion audit：

```bash
root=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/h10_R1/rollouts
printf 'dirs='; find "$root" -maxdepth 1 -type d -name 'V37_T1_FG_H10_R1_*' | wc -l
printf 'status_files='; find "$root" -maxdepth 2 -name run_status.txt | wc -l
printf 'done='; find "$root" -maxdepth 2 -name run_status.txt -print0 | \
  xargs -0 rg -l '^\[.*\] DONE V37_T1_FG_H10_R1_' | wc -l
printf 'fail='; find "$root" -maxdepth 2 -name run_status.txt -print0 | \
  xargs -0 rg -l '^\[.*\] FAIL V37_T1_FG_H10_R1_' | wc -l
```

Observed：

```text
dirs=30
status_files=30
done=30
fail=0
```

Trajectory reports：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/h10_R1/rollouts \
  --run-prefix V37_T1_FG_H10_R1_H9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05 \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/report_h10_R1_H9 \
  --report-prefix track1_h10_H9 \
  --ate-threshold -1.5 \
  --segment-threshold -5.0 \
  --downstream-threshold 1.0

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v31_track_a_report.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/h10_R1/rollouts \
  --run-prefix V37_T1_FG_H10_R1_C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05 \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/report_h10_R1_C9 \
  --report-prefix track1_h10_C9 \
  --ate-threshold -1.5 \
  --segment-threshold -5.0 \
  --downstream-threshold 1.0
```

Context / attention-mass summaries：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v36b_context_skip_summary.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/h10_R1/rollouts \
  --run-prefix V37_T1_FG_H10_R1_H9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05 \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/report_h10_R1_H9_context

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v36b_context_skip_summary.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/h10_R1/rollouts \
  --run-prefix V37_T1_FG_H10_R1_C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05 \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase1_frameglobal/report_h10_R1_C9_context
```

Track 1 R1 summary：

```text
H9 gate_pass = false
H9 best ATE delta = -0.0473357966m (FG_SEM_04, chunk10)
H9 best [200,300) delta = -0.3418221710m (FG_RISK_00, chunk10)

C9 gate_pass = false
C9 best ATE delta = -0.0694789609m (FG_SEM_04, chunk6)
C9 best [200,300) delta = -0.2606290156m (FG_RISK_00, chunk10)

H9 context source_effect_rows = 12
C9 context source_effect_rows = 12
H9/C9 context_empty_source_events_total = 0
H9/C9 attention_mass_removed_available = true
H9/C9 attention_mass_status = sampled-qk-softmax-mass
H9/C9 attention_mass_rows = 12
```

Decision：

```text
Track 1 h10 gate = fail.
No Track 1 h15 is allowed.
Continue to Track 2 / Track 3 / Track 4 independent path probes.
```

---

## 3. Track 2 SWA Local-Continuity h10 R1

Execution：

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase2_swa/h10_R1 \
V36B_RUN_PREFIX_BASE=V37_T2_SWA_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_02_SKY_PARTIAL_KEEP,SWA_FINE_03_VEGETATION_CONDITIONAL,SWA_FINE_04_BOUNDARY_PROTECT,SWA_FINE_05_CACHE_LIFECYCLE \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

Scheduling：

```text
One worker queue per GPU.
Track 2 uses GPUs 0,1,2,3.
Track 3 uses GPUs 4,5,6,7 in parallel.
No single LoGeR process is bound to multiple GPUs.
```

---

## 4. Track 3 TTT Semantic Static / Negative h10 R1

Execution：

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase3_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V37_T3_TTT_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_03_SCALE_CONDITIONED,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_05_STRUCTURE_PROTECT,TTT_FINE_RISK_01_CONFLICT_TRI,TTT_FINE_RISK_02_SCALE_STATE,TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

---

## 5. Track 4 Alias Repair Preflight

Blocker：

```text
The v37 plan names Track 4 candidates:
    SEM_C23_01_READ_ONLY_RESID
    SEM_C23_02_NO_TTT
    SEM_C23_03_NO_SWA
    SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY
    SEM_C23_05_STATIC_RESCUE_RESID

These aliases did not exist in tools/run_v24_candidate_rollout.sh.
Running Track 4 without adding them would either fail immediately or reuse
older V31 aliases without auditable v37 identity.
```

Repair：

```text
tools/run_v24_candidate_rollout.sh:
    added SEM_C23_01_READ_ONLY_RESID
        read_path_only via readonly mode
        read_cue_source = v31.sem_resid_coarse_l025.c23past

    added SEM_C23_02_NO_TTT
        readonly mode
        read_cue_source = v31.sem_resid_coarse_l025.c23past
        frame/global compact_kv semantic-risk source skip

    added SEM_C23_03_NO_SWA
        hybrid mode
        read_cue_source = v31.sem_resid_coarse_l025.c23past
        semantic paths = frame,global,ttt
        no SWA policy enabled by this alias

    added SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY
        readonly mode
        read_cue_source = v31.sem_resid_coarse_l025.c23past
        frame/global compact_kv semantic-risk source skip

    added SEM_C23_05_STATIC_RESCUE_RESID
        readonly mode
        read_cue_source = mix.c23past_static_rescue_a025
        frame/global structure-rescue compact_kv protection
```

Validation：

```bash
bash -n tools/run_v24_candidate_rollout.sh
```

Result：

```text
PASS
```

---

## 6. Blocker：Track 2 / Track 3 R1 invalidated by live launcher edit

Observation：

```text
While Track 2 / Track 3 R1 queues were running, tools/run_v24_candidate_rollout.sh
was edited to add Track 4 SEM_C23 aliases.

Some still-running bash instances then produced duplicate/misrouted rows under:
    results/kitti01_hmc_v2/attention_cue_library_v1/

Those misrouted rows used default full-sequence / default Stage-C settings,
not the intended v37 short-rollout root and not the intended chunk window.
```

Invalidation / cleanup：

```bash
# stop the affected R1 schedulers and their child LoGeR rows
pgrep -f '[V]37_T2_SWA_H10_R1|[V]37_T3_TTT_H10_R1'
pgrep -f '[r]un_v36b_path_h10.sh'

# move partial intended-root rows aside
ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30
STAMP=20260524T1620_launcher_edit_invalid
mv "$ROOT/phase2_swa/h10_R1" "$ROOT/phase2_swa/h10_R1_${STAMP}"
mv "$ROOT/phase3_ttt/h10_R1" "$ROOT/phase3_ttt/h10_R1_${STAMP}"

# move misrouted default-root rows aside
MIS="$ROOT/invalid_misrouted_attention_cue_library_R1"
mkdir -p "$MIS"
find results/kitti01_hmc_v2/attention_cue_library_v1 -maxdepth 1 -type d \
  \( -name 'V37_T2_SWA_H10_R1_*' -o -name 'V37_T3_TTT_H10_R1_*' \) \
  -exec mv {} "$MIS/" \;
```

Moved invalid evidence：

```text
phase2_swa/h10_R1_20260524T1620_launcher_edit_invalid
phase3_ttt/h10_R1_20260524T1620_launcher_edit_invalid
invalid_misrouted_attention_cue_library_R1/
```

Decision：

```text
Track 2 R1 and Track 3 R1 are invalid / partial.
They are not used for any trajectory gate, report, or conclusion.

Repair before rerun:
    finish launcher edits first,
    validate bash -n,
    then rerun Track 2 / Track 3 as R2 without modifying the launcher while
    rows are active.
```

---

## 7. Track 2 / Track 3 h10 R2 Relaunch

Preflight：

```bash
ps -eo pid,stat,etime,cmd | rg 'V37_T2_SWA_H10_R1|V37_T3_TTT_H10_R1|V37_T2_SWA_H10_R2|V37_T3_TTT_H10_R2|tools/run_v36b_path_h10.sh' | rg -v rg || true
bash -n tools/run_v24_candidate_rollout.sh tools/run_v36b_path_h10.sh
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Observed：

```text
no active v37 Track2/Track3 rows
bash -n = PASS
GPU 0-7 available
```

Track 2 R2：

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase2_swa/h10_R2 \
V36B_RUN_PREFIX_BASE=V37_T2_SWA_H10_R2 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_02_SKY_PARTIAL_KEEP,SWA_FINE_03_VEGETATION_CONDITIONAL,SWA_FINE_04_BOUNDARY_PROTECT,SWA_FINE_05_CACHE_LIFECYCLE \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

Track 3 R2：

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase3_ttt/h10_R2 \
V36B_RUN_PREFIX_BASE=V37_T3_TTT_H10_R2 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_03_SCALE_CONDITIONED,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_05_STRUCTURE_PROTECT,TTT_FINE_RISK_01_CONFLICT_TRI,TTT_FINE_RISK_02_SCALE_STATE,TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```

Candidate mapping boundary：

```text
v37 exact policy names are not all implemented as first-class aliases yet.
Track 0 R1 uses the closest landed training-free semantic memory actions:
    FG_RISK_00
    FG_SEM_02
    FG_SEM_04
    FG_SEM_05
    SWA_FINE_01_OVERLAP_STRUCTURE_KEEP
    SWA_FINE_04_BOUNDARY_PROTECT
    TTT_FINE_01_STRUCTURE_POSITIVE
    TTT_FINE_04_LOWSTUFF_HIGHD_SHORT
    TTT_FINE_RISK_01_CONFLICT_TRI
The R1 audit will decide whether this action set is sufficiently distinguishable
or whether v37-specific aliases/instrumentation must be repaired before rollout.
```

Added Track 0 atlas aggregator：

```text
tools/v37_action_influence_atlas.py
```

Validation：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v37_action_influence_atlas.py
```

Result：

```text
PASS
```

Track 0 smoke R1 completed：

```text
rows expected = 60
rows DONE = 60
failures observed by scheduler = 0
```

Atlas aggregation：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v37_action_influence_atlas.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase0_action_influence/smoke_R1/rollouts \
  --run-prefix V37_TRACK0_SMOKE_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 3 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_RISK_01_CONFLICT_TRI \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase0_action_influence/report_R1
```

Initial aggregation blocker：

```text
The first aggregation returned h0c_influence_nontriviality_pass=false because
max_skipped_source_influence_mass became NaN even though attention_mass_rows=24
and max_influence_mass=0.12868116796016693.
```

Repair：

```text
tools/v37_action_influence_atlas.py:
    added _finite_float(...) and ignored NaN in source influence max.
```

Validation after repair：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v37_action_influence_atlas.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v37_action_influence_atlas.py \
  --rollout-root results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase0_action_influence/smoke_R1/rollouts \
  --run-prefix V37_TRACK0_SMOKE_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 3 \
  --candidates V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05,SWA_FINE_01_OVERLAP_STRUCTURE_KEEP,SWA_FINE_04_BOUNDARY_PROTECT,TTT_FINE_01_STRUCTURE_POSITIVE,TTT_FINE_04_LOWSTUFF_HIGHD_SHORT,TTT_FINE_RISK_01_CONFLICT_TRI \
  --out-dir results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/phase0_action_influence/report_R1
```

Track 0 R1 summary：

```text
rows_done = 60 / 60
missing_rows = 0
context_empty_source_events_total = 0
source_effect_rows = 24
swa_effect_rows = 12
ttt_effect_rows = 60
attention_mass_rows = 24
max_influence_mass = 0.12868116796016693
max_skipped_source_influence_mass = 0.12868116796016693
h0a_hook_reachability_pass = true
h0b_action_distinguishability_pass = true
h0c_influence_nontriviality_pass = true
track0_gate_pass = true
```

---

## 2. Track 1 Frame/Global Source Surgery h10 R1

Execution：

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30 \
V36B_PHASE_NAME=phase1_frameglobal/h10_R1 \
V36B_RUN_PREFIX_BASE=V37_T1_FG_H10_R1 \
V36B_CANDIDATES=V31_BASE_H9_REFERENCE,FG_RISK_00,FG_SEM_02,FG_SEM_04,FG_SEM_05 \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
tools/run_v36b_path_h10.sh
```
