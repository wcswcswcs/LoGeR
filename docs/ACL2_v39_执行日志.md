# ACL2 v39 执行日志：SemanticAppearanceCue MemoryPath Causal

日期：2026-05-25（Asia/Singapore）  
计划文件：`docs/ACL2_v39_SemanticAppearanceCue_MemoryPath_CausalPlan.md`  
实验复盘：`docs/ACL2_v39_实验复盘.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal/`  

原则：

```text
1. 只记录实际执行命令与实际落盘 artifact。
2. 不把 Track 0 audit、short rollout、proxy visualization、blocked stage 写成 deployable success。
3. Runtime semantic source 只能使用 VideoMasklet / Stage-C cache，不使用 GT SemanticKITTI runtime labels。
4. 每个 LoGeR 进程只绑定一张 GPU；本轮用户指定 GPU 0,1,2,3 可用。
5. 遇到 blocker 按计划修复方向尝试，修复结果写入实验复盘。
```

---

## 0. 初始化

读取计划：

```bash
sed -n '1,220p' docs/ACL2_v39_SemanticAppearanceCue_MemoryPath_CausalPlan.md
sed -n '220,520p' docs/ACL2_v39_SemanticAppearanceCue_MemoryPath_CausalPlan.md
```

检查 v38/v39 现有工具与 launcher alias：

```bash
sed -n '320,460p' tools/run_v24_candidate_rollout.sh
sed -n '460,660p' tools/run_v24_candidate_rollout.sh
sed -n '660,820p' tools/run_v24_candidate_rollout.sh
sed -n '1,260p' tools/run_v36b_path_h10.sh
sed -n '1,260p' tools/v37_action_influence_atlas.py
sed -n '1,260p' tools/v38_action_influence_postprocess.py
sed -n '1,620p' tools/v38_durability_report.py
```

建立结果目录并复用 v36B parent snapshots：

```bash
mkdir -p results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
ln -s ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots \
  results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal/phase0_parent_snapshots
```

确认 snapshot：

```bash
find -L results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots \
  -maxdepth 3 -type f | head -20
```

---

## 1. 工程修改

修改 launcher，添加 v39 candidate aliases：

```text
tools/run_v24_candidate_rollout.sh
```

新增工具：

```text
tools/v39_semantic_appearance_atlas.py
tools/v39_final_summary_report.py
```

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v39_semantic_appearance_atlas.py \
  tools/v39_final_summary_report.py \
  tools/v38_durability_report.py \
  tools/v37_action_influence_atlas.py

bash -n tools/run_v24_candidate_rollout.sh
```

结果：

```text
py_compile: PASS
bash -n: PASS
snapshot symlink and H9/C9 chunk 6/10/16 state snapshots: present
```

---

## 2. Track 0：Semantic-Appearance Influence Atlas h3

启动命令：

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T0_CANDIDATES='V31_BASE_H9_REFERENCE,FG_01_DYNAMIC_HIGHD_SKIP,FG_02_VEGETATION_HIGHD_SKIP,FG_03_SKY_APPANOM_WEAK_SKIP,FG_04_LOWTRUST_APPANOM_SKIP,FG_05_RISK_SKIP_STRUCTURE_RESCUE,FG_06_SHADOW_PROXY_SKIP,SWA_01_NONOVERLAP_DYNAMIC_REMOVE,SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC,SWA_03_STRUCTURE_OVERLAP_PROTECT,SWA_04_SKY_HORIZON_NEUTRAL,SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP,SWA_06_COMBINED_LOCAL_TOPOLOGY,TTT_01_STRUCTURE_LONG_ANCHOR,TTT_02_DYNAMIC_NO_LONG_WRITE,TTT_03_VEG_SHORT_NEGATIVE,TTT_04_SKY_NEUTRAL_NO_LONG,TTT_05_COMBINED_LIFECYCLE,TTT_06_SHADOW_LOWTRUST_NO_LONG,C23R_01_READ_ONLY_RESID,C23R_02_NO_TTT,C23R_03_NO_SWA,C23R_04_FRAMEGLOBAL_ONLY,C23R_05_APPANOM_SEM_Z,C23R_06_STATIC_RESCUE'
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase0_semantic_appearance/h3_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK0_H3_R1 \
V36B_CANDIDATES="$V39_T0_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=3 \
V36B_GPUS=0,1,2,3 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Blocker 1：

```text
At 2026-05-25 06:44:56, row failed:
    V39_TRACK0_H3_R1_H9_SWA_04_SKY_HORIZON_NEUTRAL_chunk6_h3...

Error:
    Unsupported CANDIDATE_ID for v24 rollout: SWA_04_SKY_HORIZON_NEUTRAL

Cause:
    v39 plan uses SWA_04_SKY_HORIZON_NEUTRAL, while launcher only had the
    earlier v38 alias SWA_05_SKY_HORIZON_NEUTRAL.
```

Repair：

```bash
# Edited tools/run_v24_candidate_rollout.sh:
#   added SWA_04_SKY_HORIZON_NEUTRAL as an alias for the sky horizon neutral
#   SWA policy implemented with causal_swa_sky_partial_keep.

bash -n tools/run_v24_candidate_rollout.sh
```

Repair validation：

```text
bash -n: PASS
```

Boundary：

```text
The failed row is not counted as DONE. It must be rerun after the current queue
finishes or is otherwise safely stopped.
```

Blocker 2：

```text
At 2026-05-25 06:48:47, row produced a DONE run_status but the wrapper returned
rc=2:
    V39_TRACK0_H3_R1_H9_SWA_01_NONOVERLAP_DYNAMIC_REMOVE_chunk6_h3...

Log excerpt:
    DONE V39_TRACK0_H3_R1_H9_SWA_01_NONOVERLAP_DYNAMIC_REMOVE...
    tools/run_v24_candidate_rollout.sh: line 1362:
        unexpected EOF while looking for matching `"'

Cause:
    tools/run_v24_candidate_rollout.sh was patched while some already-started
    shell instances were still running. Bash can read a script lazily from disk,
    so an in-place edit can invalidate an active shell even after the child
    LoGeR run itself landed DONE.
```

Repair / handling:

```text
Do not edit launcher again while Track 0 queue is active.
After the active queue exits, classify rows by landed run_status/artifacts:
    run_status DONE + prediction present => landed artifact, wrapper false fail
    no run_status DONE => missing row, rerun in repair phase
```

Track 0 queue exit audit：

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v37_action_influence_atlas.py \
  --rollout-root "$V39_ROOT/phase0_semantic_appearance/h3_R1/rollouts" \
  --run-prefix V39_TRACK0_H3_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 3 \
  --candidates "$V39_T0_CANDIDATES" \
  --out-dir "$V39_ROOT/phase0_semantic_appearance/report_R1"
```

Intermediate result：

```text
rows_expected = 150
rows_done = 149
missing_rows = 1
missing row = H9 / chunk6 / SWA_04_SKY_HORIZON_NEUTRAL

The four other fail_file rows were checked from landed artifacts:
    SWA_01_NONOVERLAP_DYNAMIC_REMOVE
    SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC
    SWA_03_STRUCTURE_OVERLAP_PROTECT
    TTT_02_DYNAMIC_NO_LONG_WRITE
All four had run_status DONE and prediction artifact present, so they were
classified as wrapper false-fails caused by the active launcher edit.
```

Repair rerun command for the one genuinely missing row：

```bash
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase0_semantic_appearance/h3_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK0_H3_R1 \
V36B_CANDIDATES="SWA_04_SKY_HORIZON_NEUTRAL" \
V36B_CHUNKS=6 \
V36B_PARENTS=H9 \
V36B_HORIZON=3 \
V36B_GPUS=0 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Repair result：

```text
[2026-05-25 09:55:27] START V39_TRACK0_H3_R1_H9_SWA_04_SKY_HORIZON_NEUTRAL_chunk6_h3...
[2026-05-25 10:02:58] DONE  V39_TRACK0_H3_R1_H9_SWA_04_SKY_HORIZON_NEUTRAL_chunk6_h3...
```

Final Track 0 aggregation：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v37_action_influence_atlas.py \
  --rollout-root "$V39_ROOT/phase0_semantic_appearance/h3_R1/rollouts" \
  --run-prefix V39_TRACK0_H3_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 3 \
  --candidates "$V39_T0_CANDIDATES" \
  --out-dir "$V39_ROOT/phase0_semantic_appearance/report_R1"
```

Final Track 0 result：

```text
rows_expected = 150
rows_done = 150
missing_rows = 0
h0a_hook_reachability_pass = true
h0b_action_distinguishability_pass = true
h0c_influence_nontriviality_pass = true
track0_gate_pass = true
context_empty_source_events_total = 0
source_effect_rows = 60
swa_effect_rows = 36
ttt_effect_rows = 150
attention_mass_rows = 42
max_influence_mass = 0.23769477009773254
max_skipped_source_influence_mass = 0.12868116796016693
```

Semantic-appearance postprocess：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v39_semantic_appearance_atlas.py \
  --atlas-dir "$V39_ROOT/phase0_semantic_appearance/report_R1" \
  --image-dir /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --stage-c-cache-dir results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full \
  --chunks 6,10,16 \
  --out-dir "$V39_ROOT/phase0_semantic_appearance/report_R1"
```

Semantic-appearance postprocess result：

```text
frame_rows = 96
masklet_rows = 20
semantic_label_rows = 7
sky_lab_delta_p90 = 2.301654845589792
sky_candidate_level_influence_mass_max = 0.23769477009773254
sky_causality_decision = not_proven_per_label_influence_missing

Generated:
    rgb_frame_strip_chunk006/010/016.png
    semantic_mask_overlay_chunk006/010/016.png
    appearance_anomaly_heatmap_chunk006/010/016.png

Not landed / not fabricated:
    D_g_heatmap
    source_attention_mass_heatmap
    SWA_overlap_nonoverlap_source_mass_map
    TTT_update_contribution_map
    full_combined_risk_map_with_Dg_attention_swa_ttt
```

---

## 3. Track 1：Frame/Global Source Surgery h10

启动命令：

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T1_CANDIDATES='V31_BASE_H9_REFERENCE,FG_01_DYNAMIC_HIGHD_SKIP,FG_02_VEGETATION_HIGHD_SKIP,FG_03_SKY_APPANOM_WEAK_SKIP,FG_04_LOWTRUST_APPANOM_SKIP,FG_05_RISK_SKIP_STRUCTURE_RESCUE,FG_06_SHADOW_PROXY_SKIP'
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase1_frameglobal/h10_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK1_H10_R1 \
V36B_CANDIDATES="$V39_T1_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Runtime note：

```text
Started 2026-05-25 10:05:23.
One LoGeR process per GPU on GPU 0,1,2,3.
```

GPU 4-7 monitoring：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '5,8p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | \
  rg 'GPU-(ab3a379b|59af916e|fc712b2c|9ff9ded3)' || true
```

Result：

```text
During Track 1, GPUs 4,5,6,7 were repeatedly occupied by xuetong active python
processes with about 17-18GB memory each and nonzero/high utilization.
Therefore no v39 rows were launched on GPU 4-7.
```

Completion：

```text
[2026-05-25 11:33:26] Track 1 h10 matrix completed.
rows = 42/42
```

Report command：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V39_ROOT/phase1_frameglobal/h10_R1/rollouts" \
  --run-prefix V39_TRACK1_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V39_T1_CANDIDATES" \
  --out-dir "$V39_ROOT/phase1_frameglobal/report_h10_R1" \
  --report-prefix track1_h10 \
  --mode h10
```

Track 1 h10 result：

```text
rows = 42
missing_rows = 0
all_rows_done = true
gate_pass = false
best_ATE_candidate = FG_01_DYNAMIC_HIGHD_SKIP
best_ATE_parent = C9
best_ATE_chunk = 6
best_ATE_delta_vs_base = -0.06947896088025018
best_rolling_100f_candidate = FG_02_VEGETATION_HIGHD_SKIP
best_rolling_100f_parent = H9
best_rolling_100f_chunk = 6
best_rolling_100f_best_delta = -0.15147176397447915

Decision:
    Track 1 h10 gate = fail.
    No Track 1 h15 launched.
```

---

## 4. Track 2：SWA Local-Continuity h10

启动前 GPU audit：

```text
GPU 0-3: free after Track 1 completion.
GPU 4-7: still occupied by xuetong active python processes, about 17-18GB
memory each, with nonzero/high utilization.
```

启动命令：

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T2_CANDIDATES='V31_BASE_H9_REFERENCE,SWA_01_NONOVERLAP_DYNAMIC_REMOVE,SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC,SWA_03_STRUCTURE_OVERLAP_PROTECT,SWA_04_SKY_HORIZON_NEUTRAL,SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP,SWA_06_COMBINED_LOCAL_TOPOLOGY'
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase2_swa/h10_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK2_H10_R1 \
V36B_CANDIDATES="$V39_T2_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Runtime note：

```text
Started 2026-05-25 11:34:42.
One LoGeR process per GPU on GPU 0,1,2,3.
```

Interruption / resume audit：

```text
After the session interruption, Track 2 had 40/42 rows with DONE in
run_status.txt.

Two rows had output artifacts and logs, but run_status.txt only contained START:
    V39_TRACK2_H10_R1_C9_SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP_chunk16_h10_globalgate_H9parent_SWKS3
    V39_TRACK2_H10_R1_C9_SWA_06_COMBINED_LOCAL_TOPOLOGY_chunk16_h10_globalgate_H9parent_SWKS3

Report attempt before repair:
    rows = 40
    missing_rows = 2
    all_rows_done = false

Boundary:
    Do not manually edit run_status.txt.
    Do not claim rows with only START as completed evidence.
```

Pre-repair report command：

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T2_CANDIDATES='V31_BASE_H9_REFERENCE,SWA_01_NONOVERLAP_DYNAMIC_REMOVE,SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC,SWA_03_STRUCTURE_OVERLAP_PROTECT,SWA_04_SKY_HORIZON_NEUTRAL,SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP,SWA_06_COMBINED_LOCAL_TOPOLOGY'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V39_ROOT/phase2_swa/h10_R1/rollouts" \
  --run-prefix V39_TRACK2_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V39_T2_CANDIDATES" \
  --out-dir "$V39_ROOT/phase2_swa/report_h10_R1_attempt_before_repair" \
  --report-prefix track2_h10 \
  --mode h10
```

Repair rerun command：

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase2_swa/h10_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK2_H10_R1 \
V36B_CANDIDATES="SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP,SWA_06_COMBINED_LOCAL_TOPOLOGY" \
V36B_CHUNKS=16 \
V36B_PARENTS=C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Repair runtime note：

```text
Started 2026-05-25 15:06:49 on GPUs 0,1.
The launcher moved the stale START-only directories to:
    *.INVALID_RERUN_20260525_150649
```

Repair completion：

```text
[2026-05-25 15:26:32] DONE
    V39_TRACK2_H10_R1_C9_SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP_chunk16_h10_globalgate_H9parent_SWKS3

[2026-05-25 15:27:25] DONE
    V39_TRACK2_H10_R1_C9_SWA_06_COMBINED_LOCAL_TOPOLOGY_chunk16_h10_globalgate_H9parent_SWKS3
```

Final Track 2 report command：

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T2_CANDIDATES='V31_BASE_H9_REFERENCE,SWA_01_NONOVERLAP_DYNAMIC_REMOVE,SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC,SWA_03_STRUCTURE_OVERLAP_PROTECT,SWA_04_SKY_HORIZON_NEUTRAL,SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP,SWA_06_COMBINED_LOCAL_TOPOLOGY'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V39_ROOT/phase2_swa/h10_R1/rollouts" \
  --run-prefix V39_TRACK2_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V39_T2_CANDIDATES" \
  --out-dir "$V39_ROOT/phase2_swa/report_h10_R1" \
  --report-prefix track2_h10 \
  --mode h10
```

Track 2 h10 result：

```text
rows = 42
missing_rows = 0
all_rows_done = true
gate_pass = false
best_ATE_candidate = SWA_01_NONOVERLAP_DYNAMIC_REMOVE
best_ATE_parent = C9
best_ATE_chunk = 6
best_ATE_delta_vs_base = -0.7081471063607552
best_rolling_100f_candidate = SWA_01_NONOVERLAP_DYNAMIC_REMOVE
best_rolling_100f_parent = C9
best_rolling_100f_chunk = 6
best_rolling_100f_best_delta = -1.3354202480089725
H9 best_200_300_delta_vs_base = -2.3916353408948012
C9 best_200_300_delta_vs_base = -2.3564699827626683

Decision:
    Track 2 h10 gate = fail.
    No Track 2 h15 launched.
```

Parallel Track 3 launch for GPU utilization：

```text
GPU 2/3 were free while Track 2 repair used GPU 0/1.
Track 3 h10 is independent from Track 2 h10, so it was launched in parallel.
GPU 4-7 were still occupied by xuetong active python processes, so they were
not used.
```

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T3_CANDIDATES='V31_BASE_H9_REFERENCE,TTT_01_STRUCTURE_LONG_ANCHOR,TTT_02_DYNAMIC_NO_LONG_WRITE,TTT_03_VEG_SHORT_NEGATIVE,TTT_04_SKY_NEUTRAL_NO_LONG,TTT_05_COMBINED_LIFECYCLE,TTT_06_SHADOW_LOWTRUST_NO_LONG'
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase3_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK3_H10_R1 \
V36B_CANDIDATES="$V39_T3_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=2,3 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

---

## 5. Track 4：Semantic C23 / Appearance-Anomaly h10

启动原因：

```text
Track 2 h10 gate failed, so GPU 0/1 were released.
Track 3 h10 was already running on GPU 2/3.
Track 4 is independent from Track 3, so Track 4 h10 was launched on GPU 0/1.
```

启动命令：

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T4_CANDIDATES='V31_BASE_H9_REFERENCE,C23R_01_READ_ONLY_RESID,C23R_02_NO_TTT,C23R_03_NO_SWA,C23R_04_FRAMEGLOBAL_ONLY,C23R_05_APPANOM_SEM_Z,C23R_06_STATIC_RESCUE'
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase4_semc23/h10_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK4_H10_R1 \
V36B_CANDIDATES="$V39_T4_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Runtime note：

```text
Started 2026-05-25 15:27:58.
One LoGeR process per GPU on GPU 0,1.
GPU 4-7 remained occupied by other-user processes and were not used.
```

Interruption recovery / live status：

```text
2026-05-25 16:21:05 +08 resumed after session interruption.

Track 3 h10:
    current status = running
    DONE rows observed = 8/42
    active GPUs = 2,3
    active rows include:
        H9 / TTT_01_STRUCTURE_LONG_ANCHOR / chunk10
        H9 / TTT_02_DYNAMIC_NO_LONG_WRITE / chunk10

Track 4 h10:
    current status = running
    DONE rows observed = 12/42
    active GPUs = 0,1
    active rows include:
        H9 / C23R_06_STATIC_RESCUE / chunk10
        H9 / C23R_05_APPANOM_SEM_Z / chunk10

GPU audit:
    GPU 4,5,6,7 still had other-user compute processes
    (/mnt/data/users/xuetong/miniconda3/envs/active/bin/python).
    They were not used.

Boundary:
    No run_status.txt was manually edited.
    No partial row is counted as DONE until landed run_status contains DONE.
```

### 2026-05-25 18:18-18:20 Track 4 h10 completed and report generated

Completion:

```text
phase4_semc23/h10_R1 rows = 42/42 DONE
last rows:
    V39_TRACK4_H10_R1_C9_C23R_05_APPANOM_SEM_Z_chunk16... DONE
    V39_TRACK4_H10_R1_C9_C23R_06_STATIC_RESCUE_chunk16... DONE
```

Report command:

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T4_CANDIDATES='V31_BASE_H9_REFERENCE,C23R_01_READ_ONLY_RESID,C23R_02_NO_TTT,C23R_03_NO_SWA,C23R_04_FRAMEGLOBAL_ONLY,C23R_05_APPANOM_SEM_Z,C23R_06_STATIC_RESCUE'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V39_ROOT/phase4_semc23/h10_R1/rollouts" \
  --run-prefix V39_TRACK4_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V39_T4_CANDIDATES" \
  --out-dir "$V39_ROOT/phase4_semc23/report_h10_R1" \
  --report-prefix track4_h10 \
  --mode h10
```

Report result:

```text
all_rows_done = true
rows = 42
missing_rows = 0
gate_pass = false
best_ATE_candidate = C23R_05_APPANOM_SEM_Z
best_ATE_parent = C9
best_ATE_chunk = 6
best_ATE_delta_vs_base = -0.5844090747247357
best_rolling_100f_candidate = C23R_05_APPANOM_SEM_Z
best_rolling_100f_parent = C9
best_rolling_100f_chunk = 6
best_rolling_100f_best_delta = -2.8143212230658907
best_downstream_400_600_delta_for_best_ATE = +1.0187841702235119

H9 best [200,300) delta = -4.976563854445843
C9 best [200,300) delta = -4.962398285051577
```

Decision:

```text
Track 4 h10 gate failed.
No Track 4 h15 continuation.
```

### 2026-05-25 18:20-18:22 Track 3 safe acceleration on freed GPU 0/1

Reason:

```text
Track 4 finished and released GPU 0/1.
Track 3 original scheduler was still running on GPU 2/3 with 22/42 DONE.
GPU 4/5/6/7 remained occupied by other-user xuetong python processes.

To reduce collision risk with the original Track 3 scheduler, only far-future
original-order rows were launched:
    C9 / chunk16 / TTT_03
    C9 / chunk16 / TTT_04
    C9 / chunk16 / TTT_05
    C9 / chunk16 / TTT_06

The original queue still had C9 chunk6/chunk10 and chunk16 earlier rows before
these rows, so the supplemental rows should land before the original queue
reaches them. FORCE=0 is used; already-DONE rows will be skipped by the original
launcher later.
```

Acceleration command:

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase3_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK3_H10_R1 \
V36B_CANDIDATES="TTT_03_VEG_SHORT_NEGATIVE,TTT_04_SKY_NEUTRAL_NO_LONG,TTT_05_COMBINED_LIFECYCLE,TTT_06_SHADOW_LOWTRUST_NO_LONG" \
V36B_CHUNKS=16 \
V36B_PARENTS=C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Runtime:

```text
Started 2026-05-25 18:22:14.
Session id = 48491.
Initial rows:
    GPU 0: C9 / TTT_04_SKY_NEUTRAL_NO_LONG / chunk16
    GPU 1: C9 / TTT_03_VEG_SHORT_NEGATIVE / chunk16
```

### 2026-05-25 18:53 Track 3 second acceleration on freed GPU 0

Reason:

```text
The first acceleration row C9 / chunk16 / TTT_06 completed and released GPU 0.
The original Track 3 queue was still on C9 chunk6, so C9 chunk16 base/TTT_01/
TTT_02 were still far enough ahead to accelerate with lower collision risk.
```

Command:

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase3_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK3_H10_R1 \
V36B_CANDIDATES="V31_BASE_H9_REFERENCE,TTT_01_STRUCTURE_LONG_ANCHOR,TTT_02_DYNAMIC_NO_LONG_WRITE" \
V36B_CHUNKS=16 \
V36B_PARENTS=C9 \
V36B_HORIZON=10 \
V36B_GPUS=0 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Runtime:

```text
Started 2026-05-25 18:53:09.
Session id = 72066.
Initial row:
    GPU 0: C9 / V31_BASE_H9_REFERENCE / chunk16
```

### 2026-05-25 19:11 Track 3 third acceleration on freed GPU 1

Reason:

```text
The first acceleration queue finished and released GPU 1.
Track 3 original queue had reached C9 chunk10 front rows; to reduce collision
risk, only the later chunk10 rows TTT_05/TTT_06 were launched on GPU 1.
```

Command:

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V36B_RESULT_ROOT="$V39_ROOT" \
V36B_PHASE_NAME=phase3_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V39_TRACK3_H10_R1 \
V36B_CANDIDATES="TTT_05_COMBINED_LIFECYCLE,TTT_06_SHADOW_LOWTRUST_NO_LONG" \
V36B_CHUNKS=10 \
V36B_PARENTS=C9 \
V36B_HORIZON=10 \
V36B_GPUS=1 \
FORCE=0 \
bash tools/run_v36b_path_h10.sh
```

Runtime:

```text
Started 2026-05-25 19:11:04.
Session id = 94584.
Initial row:
    GPU 1: C9 / TTT_05_COMBINED_LIFECYCLE / chunk10
```

Safety stop:

```text
At 2026-05-25 19:35, the original Track 3 queue had reached C9 chunk10
TTT_03/TTT_04 and was close to TTT_06. The supplemental TTT_06 chunk10 row was
still running, which created a duplicate RUN_NAME collision risk.

Action:
    terminated the supplemental process group 130014 with SIGTERM.

Boundary:
    This affected only the supplemental acceleration row.
    The original Track 3 scheduler on GPU 2/3 continued running.
    The START-only supplemental TTT_06 chunk10 directory is not counted as DONE;
    when the original queue reaches the same RUN_NAME, the launcher will move
    the stale non-DONE directory to .INVALID_RERUN_* and rerun it normally.
```

Outcome:

```text
The original queue later moved the stale supplemental TTT_06 chunk10 directory
to:
    V39_TRACK3_H10_R1_C9_TTT_06_SHADOW_LOWTRUST_NO_LONG_chunk10_h10_globalgate_H9parent_SWKS3.INVALID_RERUN_20260525_193856
and reran the official row successfully.

Supplemental acceleration rows that landed DONE were later skipped by the
original queue with "SKIP existing DONE run".
```

### 2026-05-25 19:59-20:00 Track 3 h10 completed and report generated

Completion:

```text
phase3_ttt/h10_R1 rows = 42/42 DONE
missing_rows = 0
all_rows_done = true
```

Report command:

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
V39_T3_CANDIDATES='V31_BASE_H9_REFERENCE,TTT_01_STRUCTURE_LONG_ANCHOR,TTT_02_DYNAMIC_NO_LONG_WRITE,TTT_03_VEG_SHORT_NEGATIVE,TTT_04_SKY_NEUTRAL_NO_LONG,TTT_05_COMBINED_LIFECYCLE,TTT_06_SHADOW_LOWTRUST_NO_LONG'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V39_ROOT/phase3_ttt/h10_R1/rollouts" \
  --run-prefix V39_TRACK3_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V39_T3_CANDIDATES" \
  --out-dir "$V39_ROOT/phase3_ttt/report_h10_R1" \
  --report-prefix track3_h10 \
  --mode h10
```

Report result:

```text
gate_pass = false
best_ATE_candidate = TTT_01_STRUCTURE_LONG_ANCHOR
best_ATE_parent = H9
best_ATE_chunk = 6
best_ATE_delta_vs_base = -0.10855437824889691
best_rolling_100f_candidate = TTT_01_STRUCTURE_LONG_ANCHOR
best_rolling_100f_parent = H9
best_rolling_100f_chunk = 6
best_rolling_100f_best_delta = -0.17132255368217386
best_downstream_400_600_delta_for_best_ATE = -0.08864076153853162

H9 best [200,300) delta = -0.16360952387594807
C9 best [200,300) delta = -0.13902892687064394
```

Decision:

```text
Track 3 h10 gate failed.
No Track 3 h15 continuation.
```

### 2026-05-25 20:00 final summary

Command:

```bash
V39_ROOT=results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v39_final_summary_report.py --root "$V39_ROOT"
```

Output:

```text
track0_gate_pass = true
track1_h10_gate_pass = false
track2_h10_gate_pass = false
track3_h10_gate_pass = false
track4_h10_gate_pass = false
track5_full_online_allowed = false
track5_full_online_launched = false
target30_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029
sky_causality_decision = not_proven_per_label_influence_missing
```
