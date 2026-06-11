# ACL2 v38 执行日志：TrainingFree SemanticMemory Durability Target30

日期：2026-05-24（Asia/Singapore）  
计划文件：`docs/ACL2_v38_TrainingFree_SemanticMemory_Durability_Target30_Plan.md`  
实验复盘：`docs/ACL2_v38_实验复盘.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/`

原则：

```text
1. 只记录实际执行的命令、脚本、输入路径与落盘输出。
2. 不把 Track 0 action/influence audit、short rollout、proxy attribution、
   instrumentation repair、blocked downstream stage 写成 deployable online success。
3. 每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立 row 分配到不同 GPU。
4. Runtime semantic source 使用 VideoMasklet frontend / Stage-C semantic cache，
   不使用 GT SemanticKITTI runtime labels。
5. 不编造数据；没有落盘的 tensor / pixel visualization 不补画、不声称。
```

---

## 0. Preflight

读取计划：

```bash
sed -n '1,260p' docs/ACL2_v38_TrainingFree_SemanticMemory_Durability_Target30_Plan.md
sed -n '260,980p' docs/ACL2_v38_TrainingFree_SemanticMemory_Durability_Target30_Plan.md
```

检查 GPU 空闲状态：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

输出：

```text
0, NVIDIA RTX A5000, 1 MiB, 23028 MiB, 0 %
1, NVIDIA RTX A5000, 1 MiB, 23028 MiB, 0 %
2, NVIDIA RTX A5000, 1 MiB, 23028 MiB, 0 %
3, NVIDIA RTX A5000, 4 MiB, 23028 MiB, 0 %
4, NVIDIA RTX A5000, 4 MiB, 23028 MiB, 0 %
5, NVIDIA RTX A5000, 4 MiB, 23028 MiB, 0 %
6, NVIDIA RTX A5000, 1 MiB, 23028 MiB, 0 %
7, NVIDIA RTX A5000, 4 MiB, 23028 MiB, 0 %
```

检查已有 v38 文件 / alias：

```bash
ls docs/ACL2_v38* 2>/dev/null || true
rg "FG_01_DYNAMIC|C23R_01|TTT_06_FULL" tools/run_v24_candidate_rollout.sh || true
pwd
```

输出：

```text
docs/ACL2_v38_TrainingFree_SemanticMemory_Durability_Target30_Plan.md
/mnt/data/users/chengshun.wang/pjs/LoGeR
```

初始判断：

```text
v38 尚未完成。
需要新增 v38 执行日志、实验复盘日志、结果目录、候选 alias / 报告工具，
然后按 Track 0-5 计划推进。
```

创建 v38 结果目录，并复用 v36B H9/C9 parent state + merge snapshots：

```bash
mkdir -p results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
ln -sfn ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots \
  results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/phase0_parent_snapshots
ls -l results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
```

输出：

```text
total 4
lrwxrwxrwx 1 chengshun.wang chengshun.wang 83 May 24 22:57 phase0_parent_snapshots -> ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots
```

Boundary：

```text
v38 不重新生成 parent snapshots。
复用 v36B 已落盘 H9/C9 snapshots:
    chunks = 6,10,16
    state snapshots complete
    merge snapshots complete
```

---

## 1. 工程修改：v38 candidate aliases

检查 v38 aliases 是否已存在：

```bash
rg "FG_01_DYNAMIC|C23R_01|TTT_06_FULL" tools/run_v24_candidate_rollout.sh || true
```

输出为空，说明 v38 aliases 尚未落地。

修改：

```text
tools/run_v24_candidate_rollout.sh:
    added v38 Track 1 aliases:
        FG_01_DYNAMIC_HIGHD_SKIP
        FG_02_VEGETATION_HIGHD_SKIP
        FG_03_LOWTRUST_HIGHD_SKIP
        FG_04_STRUCTURE_RESCUE
        FG_05_RISK_SKIP_STATIC_RESCUE
        FG_06_COMPACT_KV_TRUE
        FG_07_BIAS_ONLY_CONTROL

    added v38 Track 2 aliases:
        SWA_01_NONOVERLAP_RISK_REMOVE
        SWA_02_OVERLAP_K_KEEP_V_ATTEN
        SWA_03_STRUCTURE_OVERLAP_PROTECT
        SWA_04_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_PROTECT
        SWA_05_SKY_HORIZON_NEUTRAL
        SWA_06_SOURCE_TOPOLOGY_CONTROL

    added v38 Track 3 aliases:
        TTT_01_STRUCTURE_LONG
        TTT_02_STRUCTURE_LONG_DYNAMIC_NOLONG
        TTT_03_VEGETATION_CONDITIONAL_SHORTNEG
        TTT_04_LOWTRUST_SHORTNEG
        TTT_05_SKY_NEUTRAL
        TTT_06_FULL_LIFECYCLE_POLICY

    added v38 Track 4 aliases:
        C23R_01_READ_ONLY_RESID
        C23R_02_NO_TTT
        C23R_03_NO_SWA
        C23R_04_FRAMEGLOBAL_COMPACT_ONLY
        C23R_05_STATIC_RESCUE_RESID
        C23R_06_C9_COMPAT_READ_ONLY

    added v38 Track 0 synthetic / semantic / path-stress aliases:
        T0_SYN_ALL_PATCH_SKIP
        T0_SYN_CENTER_BOX_SKIP
        T0_SYN_RANDOM_20PCT_SKIP
        T0_SYN_LEFT_HALF_SKIP
        T0_SYN_ALL_DYNAMIC_ROLE
        T0_SYN_ALL_STATIC_ROLE
        T0_SEM_DYNAMIC_HIGHD
        T0_SEM_VEGETATION_HIGHD
        T0_SEM_SKY_HIGHD
        T0_SEM_LOWTRUST_HIGHD
        T0_SEM_STRUCTURE_LOWD
        T0_SEM_STRUCTURE_LOWD_LOWCONFLICT
        T0_PATH_FRAME_ONLY
        T0_PATH_GLOBAL_ONLY
        T0_PATH_FRAME_GLOBAL
        T0_PATH_SWA_ONLY
        T0_PATH_TTT_ONLY
```

Validation：

```bash
bash -n tools/run_v24_candidate_rollout.sh
```

输出为空，syntax pass。

After Track 0 alias addition：

```bash
bash -n tools/run_v24_candidate_rollout.sh
```

输出为空，syntax pass。

新增 durability 汇总脚本：

```text
tools/v38_durability_report.py:
    reads landed rollout artifacts only
    computes:
        ATE delta vs per-parent reference
        [200,300) / [400,600) segment delta
        rolling 50f / 100f / 200f window deltas
        high-error rolling-window best delta
        downstream regression gate
        h10 / h15 gate pass
    writes:
        *_effects.csv
        *_effects.json
        *_missing_rows.csv
        *_by_parent.csv
        *_summary.json
        *_report.md
```

Validation：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v38_durability_report.py
bash -n tools/run_v24_candidate_rollout.sh
```

输出为空，py_compile / bash syntax pass。

---

## 2. Track 0：h3 action/influence audit R1

启动：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T0_CANDIDATES='V31_BASE_H9_REFERENCE,T0_SYN_ALL_PATCH_SKIP,T0_SYN_CENTER_BOX_SKIP,T0_SYN_RANDOM_20PCT_SKIP,T0_SYN_LEFT_HALF_SKIP,T0_SYN_ALL_DYNAMIC_ROLE,T0_SYN_ALL_STATIC_ROLE,T0_SEM_DYNAMIC_HIGHD,T0_SEM_VEGETATION_HIGHD,T0_SEM_SKY_HIGHD,T0_SEM_LOWTRUST_HIGHD,T0_SEM_STRUCTURE_LOWD,T0_SEM_STRUCTURE_LOWD_LOWCONFLICT,T0_PATH_FRAME_ONLY,T0_PATH_GLOBAL_ONLY,T0_PATH_FRAME_GLOBAL,T0_PATH_SWA_ONLY,T0_PATH_TTT_ONLY'
V36B_RESULT_ROOT="$V38_ROOT" \
V36B_PHASE_NAME=phase0_action_influence/h3_R1 \
V36B_RUN_PREFIX_BASE=V38_TRACK0_H3_R1 \
V36B_CANDIDATES="$V38_T0_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=3 \
V36B_GPUS=0,1,2,3,4,5,6,7 \
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
bash tools/run_v36b_path_h10.sh
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/phase0_action_influence/h3_R1/rollouts/
results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/matrix_logs/phase0_action_influence/h3_R1/
```

Initial stdout：

```text
[2026-05-24 23:05:55] START parent=H9 candidate=T0_SEM_DYNAMIC_HIGHD chunk=6 h=3 gpu=0
[2026-05-24 23:05:55] START parent=H9 candidate=V31_BASE_H9_REFERENCE chunk=6 h=3 gpu=1
[2026-05-24 23:05:55] START parent=H9 candidate=T0_SYN_ALL_PATCH_SKIP chunk=6 h=3 gpu=2
[2026-05-24 23:05:55] START parent=H9 candidate=T0_SYN_CENTER_BOX_SKIP chunk=6 h=3 gpu=3
[2026-05-24 23:05:55] START parent=H9 candidate=T0_SYN_RANDOM_20PCT_SKIP chunk=6 h=3 gpu=4
[2026-05-24 23:05:55] START parent=H9 candidate=T0_SYN_LEFT_HALF_SKIP chunk=6 h=3 gpu=5
[2026-05-24 23:05:55] START parent=H9 candidate=T0_SYN_ALL_DYNAMIC_ROLE chunk=6 h=3 gpu=6
[2026-05-24 23:05:55] START parent=H9 candidate=T0_SYN_ALL_STATIC_ROLE chunk=6 h=3 gpu=7
```

Completion integrity check：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/phase0_action_influence/h3_R1/rollouts
printf 'run_dirs='; find "$ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l
printf 'done_status='; rg -l '^\[[^]]+\] DONE ' "$ROOT"/*/run_status.txt | wc -l
printf 'fail_logs='; rg -n 'FAIL|Traceback|CUDA out of memory|Unsupported|error:' \
  results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/matrix_logs/phase0_action_influence/h3_R1 || true
```

输出：

```text
run_dirs=108
done_status=108
fail_logs=
```

Track 0 atlas / postprocess：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T0_CANDIDATES='V31_BASE_H9_REFERENCE,T0_SYN_ALL_PATCH_SKIP,T0_SYN_CENTER_BOX_SKIP,T0_SYN_RANDOM_20PCT_SKIP,T0_SYN_LEFT_HALF_SKIP,T0_SYN_ALL_DYNAMIC_ROLE,T0_SYN_ALL_STATIC_ROLE,T0_SEM_DYNAMIC_HIGHD,T0_SEM_VEGETATION_HIGHD,T0_SEM_SKY_HIGHD,T0_SEM_LOWTRUST_HIGHD,T0_SEM_STRUCTURE_LOWD,T0_SEM_STRUCTURE_LOWD_LOWCONFLICT,T0_PATH_FRAME_ONLY,T0_PATH_GLOBAL_ONLY,T0_PATH_FRAME_GLOBAL,T0_PATH_SWA_ONLY,T0_PATH_TTT_ONLY'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v37_action_influence_atlas.py \
  --rollout-root "$V38_ROOT/phase0_action_influence/h3_R1/rollouts" \
  --run-prefix V38_TRACK0_H3_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 3 \
  --candidates "$V38_T0_CANDIDATES" \
  --out-dir "$V38_ROOT/phase0_action_influence/report_R1"
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_action_influence_postprocess.py \
  --atlas-dir "$V38_ROOT/phase0_action_influence/report_R1"
```

Key stdout：

```text
rows_expected = 108
rows_done = 108
missing_rows = 0
context_empty_source_events_total = 0
source_effect_rows = 78
swa_effect_rows = 6
ttt_effect_rows = 108
attention_mass_rows = 72
max_influence_mass = 0.18056412041187286
max_skipped_source_influence_mass = 0.18056412041187286
h0a_hook_reachability_pass = true
h0b_action_distinguishability_pass = true
h0c_influence_nontriviality_pass = true
track0_gate_pass = true

attention_mass_removed_before_after_rows = 432
source_attention_mass_removed_bar = true
swa_overlap_nonoverlap_keep_bar = true
ttt_role_mass_by_label_bar = true
influence_atlas_by_chunk = true
per_label_files_status = explainability_missing_when_not_landed
```

Generated report files：

```bash
find results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/phase0_action_influence/report_R1 \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

输出：

```text
action_jaccard_heatmap.png
action_jaccard_matrix.csv
action_keep_ratio_by_path.csv
action_summary_by_candidate.csv
attention_mass_removed_before_after.csv
context_empty_source_events.csv
frame_source_keep_ratio_by_label.csv
global_source_keep_ratio_by_label.csv
influence_atlas_by_chunk.png
missing_rows.csv
per_label_action_mass.csv
per_masklet_action_mass.csv
phase0_action_influence_report.md
phase0_action_influence_summary.json
policy_jaccard_matrix.csv
protected_token_count.csv
semantic_group_memory_path_heatmap.png
semantic_influence_atlas.csv
semantic_path_action_influence.csv
source_attention_mass_by_label.csv
source_attention_mass_removed_bar.png
swa_nonoverlap_keep_ratio_by_label.csv
swa_overlap_keep_ratio_by_label.csv
swa_overlap_nonoverlap_keep_bar.png
swa_source_attention_mass_by_label.csv
ttt_post_zp_update_norm_by_label.csv
ttt_role_mass_by_label.csv
ttt_role_mass_by_label_bar.png
v38_action_influence_postprocess_summary.json
```

---

## 3. Track 1：Frame/Global source surgery h10 R1

启动：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T1_CANDIDATES='V31_BASE_H9_REFERENCE,FG_01_DYNAMIC_HIGHD_SKIP,FG_02_VEGETATION_HIGHD_SKIP,FG_03_LOWTRUST_HIGHD_SKIP,FG_04_STRUCTURE_RESCUE,FG_05_RISK_SKIP_STATIC_RESCUE,FG_06_COMPACT_KV_TRUE,FG_07_BIAS_ONLY_CONTROL'
V36B_RESULT_ROOT="$V38_ROOT" \
V36B_PHASE_NAME=phase1_frameglobal/h10_R1 \
V36B_RUN_PREFIX_BASE=V38_TRACK1_H10_R1 \
V36B_CANDIDATES="$V38_T1_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3,4,5,6,7 \
CONTEXT_SOURCE_SKIP_RECORD_ATTENTION_MASS=1 \
CONTEXT_SOURCE_SKIP_ATTENTION_MASS_MAX_QUERIES=512 \
bash tools/run_v36b_path_h10.sh
```

输出目录：

```text
phase1_frameglobal/h10_R1/rollouts/
matrix_logs/phase1_frameglobal/h10_R1/
```

Initial stdout：

```text
[2026-05-25 00:13:47] START parent=H9 candidate=FG_07_BIAS_ONLY_CONTROL chunk=6 h=10 gpu=0
[2026-05-25 00:13:47] START parent=H9 candidate=V31_BASE_H9_REFERENCE chunk=6 h=10 gpu=1
[2026-05-25 00:13:47] START parent=H9 candidate=FG_01_DYNAMIC_HIGHD_SKIP chunk=6 h=10 gpu=2
[2026-05-25 00:13:47] START parent=H9 candidate=FG_02_VEGETATION_HIGHD_SKIP chunk=6 h=10 gpu=3
[2026-05-25 00:13:47] START parent=H9 candidate=FG_03_LOWTRUST_HIGHD_SKIP chunk=6 h=10 gpu=4
[2026-05-25 00:13:47] START parent=H9 candidate=FG_04_STRUCTURE_RESCUE chunk=6 h=10 gpu=5
[2026-05-25 00:13:47] START parent=H9 candidate=FG_05_RISK_SKIP_STATIC_RESCUE chunk=6 h=10 gpu=6
[2026-05-25 00:13:47] START parent=H9 candidate=FG_06_COMPACT_KV_TRUE chunk=6 h=10 gpu=7
```

Completion integrity check：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/phase1_frameglobal/h10_R1/rollouts
printf 'run_dirs='; find "$ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l
printf 'done_status='; rg -l '^\[[^]]+\] DONE ' "$ROOT"/*/run_status.txt | wc -l
printf 'fail_logs='; rg -n 'FAIL|Traceback|CUDA out of memory|Unsupported|error:' \
  results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/matrix_logs/phase1_frameglobal/h10_R1 || true
```

输出：

```text
run_dirs=48
done_status=48
fail_logs=
```

First report attempt：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py ...
```

失败：

```text
ValueError: operands could not be broadcast together with shapes (50,4,4) (50,3)
```

原因：

```text
tools/v38_durability_report.py rolling ATE path treated aligned 4x4 pose
matrices as xyz positions.
Rollout artifacts are valid; failure is report-script-only.
```

修复：

```text
tools/v38_durability_report.py:
    added _as_positions(...)
    rolling windows now convert 4x4 poses to translation vectors via [:3,3]
```

Validation：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v38_durability_report.py
```

输出为空，py_compile pass。

Rerun Track 1 h10 report after fix：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T1_CANDIDATES='V31_BASE_H9_REFERENCE,FG_01_DYNAMIC_HIGHD_SKIP,FG_02_VEGETATION_HIGHD_SKIP,FG_03_LOWTRUST_HIGHD_SKIP,FG_04_STRUCTURE_RESCUE,FG_05_RISK_SKIP_STATIC_RESCUE,FG_06_COMPACT_KV_TRUE,FG_07_BIAS_ONLY_CONTROL'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V38_ROOT/phase1_frameglobal/h10_R1/rollouts" \
  --run-prefix V38_TRACK1_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V38_T1_CANDIDATES" \
  --out-dir "$V38_ROOT/phase1_frameglobal/report_h10_R1" \
  --report-prefix track1_h10 \
  --mode h10
```

Key stdout：

```text
rows = 48
missing_rows = 0
all_rows_done = true
gate_pass = false

H9:
    best_ATE_candidate = FG_01_DYNAMIC_HIGHD_SKIP
    best_ATE_chunk = 10
    best_ATE_delta_vs_base = -0.0473357966m
    best_200_300_candidate = FG_02_VEGETATION_HIGHD_SKIP
    best_200_300_chunk = 10
    best_200_300_delta_vs_base = -0.3418221710m
    best_rolling_100f_candidate = FG_07_BIAS_ONLY_CONTROL
    best_rolling_100f_chunk = 10
    best_rolling_100f_best_delta = -0.2303364706m

C9:
    best_ATE_candidate = FG_01_DYNAMIC_HIGHD_SKIP
    best_ATE_chunk = 6
    best_ATE_delta_vs_base = -0.0694789609m
    best_200_300_candidate = FG_02_VEGETATION_HIGHD_SKIP
    best_200_300_chunk = 10
    best_200_300_delta_vs_base = -0.2606290156m
    best_rolling_100f_candidate = FG_07_BIAS_ONLY_CONTROL
    best_rolling_100f_chunk = 10
    best_rolling_100f_best_delta = -0.1896032366m
```

Decision：

```text
Track 1 h10 gate = fail.
No Track 1 h15.
Per plan failure routing, compact_kv, bias-only, static rescue, and high-risk
variants were already included; all remain far below h10 continuation gate.
Track 1 is demoted.
```

---

## 4. Track 2：SWA h10 R1

启动：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T2_CANDIDATES='V31_BASE_H9_REFERENCE,SWA_01_NONOVERLAP_RISK_REMOVE,SWA_02_OVERLAP_K_KEEP_V_ATTEN,SWA_03_STRUCTURE_OVERLAP_PROTECT,SWA_04_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_PROTECT,SWA_05_SKY_HORIZON_NEUTRAL,SWA_06_SOURCE_TOPOLOGY_CONTROL'
V36B_RESULT_ROOT="$V38_ROOT" \
V36B_PHASE_NAME=phase2_swa/h10_R1 \
V36B_RUN_PREFIX_BASE=V38_TRACK2_H10_R1 \
V36B_CANDIDATES="$V38_T2_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3,4,5,6,7 \
bash tools/run_v36b_path_h10.sh
```

输出目录：

```text
phase2_swa/h10_R1/rollouts/
matrix_logs/phase2_swa/h10_R1/
```

Initial stdout：

```text
[2026-05-25 01:06:11] START parent=H9 candidate=V31_BASE_H9_REFERENCE chunk=10 h=10 gpu=0
[2026-05-25 01:06:11] START parent=H9 candidate=V31_BASE_H9_REFERENCE chunk=6 h=10 gpu=1
[2026-05-25 01:06:11] START parent=H9 candidate=SWA_01_NONOVERLAP_RISK_REMOVE chunk=6 h=10 gpu=2
[2026-05-25 01:06:11] START parent=H9 candidate=SWA_02_OVERLAP_K_KEEP_V_ATTEN chunk=6 h=10 gpu=3
[2026-05-25 01:06:11] START parent=H9 candidate=SWA_03_STRUCTURE_OVERLAP_PROTECT chunk=6 h=10 gpu=4
[2026-05-25 01:06:11] START parent=H9 candidate=SWA_04_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_PROTECT chunk=6 h=10 gpu=5
[2026-05-25 01:06:11] START parent=H9 candidate=SWA_05_SKY_HORIZON_NEUTRAL chunk=6 h=10 gpu=6
[2026-05-25 01:06:11] START parent=H9 candidate=SWA_06_SOURCE_TOPOLOGY_CONTROL chunk=6 h=10 gpu=7
```

Tail stdout：

```text
[2026-05-25 03:11:47] DONE V38_TRACK2_H10_R1_C9_SWA_06_SOURCE_TOPOLOGY_CONTROL_chunk16_h10_globalgate_H9parent_SWKS3
[2026-05-25 03:11:47] END parent=C9 candidate=SWA_06_SOURCE_TOPOLOGY_CONTROL chunk=16 h=10 gpu=2
```

Integrity check：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/phase2_swa/h10_R1/rollouts
printf 'run_dirs='; find "$ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l
printf 'done_status='; rg -l '^\[[^]]+\] DONE ' "$ROOT"/*/run_status.txt | wc -l
printf 'fail_logs='; rg -n 'FAIL|Traceback|CUDA out of memory|Unsupported|error:' \
  results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/matrix_logs/phase2_swa/h10_R1 || true
```

Output：

```text
run_dirs=42
done_status=42
fail_logs=
```

Track 2 h10 report：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T2_CANDIDATES='V31_BASE_H9_REFERENCE,SWA_01_NONOVERLAP_RISK_REMOVE,SWA_02_OVERLAP_K_KEEP_V_ATTEN,SWA_03_STRUCTURE_OVERLAP_PROTECT,SWA_04_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_PROTECT,SWA_05_SKY_HORIZON_NEUTRAL,SWA_06_SOURCE_TOPOLOGY_CONTROL'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V38_ROOT/phase2_swa/h10_R1/rollouts" \
  --run-prefix V38_TRACK2_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V38_T2_CANDIDATES" \
  --out-dir "$V38_ROOT/phase2_swa/report_h10_R1" \
  --report-prefix track2_h10 \
  --mode h10
```

Report summary：

```text
rows = 42
missing_rows = 0
all_rows_done = true
gate_pass = false

H9:
    best_ATE_candidate = SWA_01_NONOVERLAP_RISK_REMOVE
    best_ATE_chunk = 6
    best_ATE_delta_vs_base = -0.6859359661m
    best_200_300_candidate = SWA_01_NONOVERLAP_RISK_REMOVE
    best_200_300_chunk = 10
    best_200_300_delta_vs_base = -2.3916353409m
    best_rolling_100f_candidate = SWA_01_NONOVERLAP_RISK_REMOVE
    best_rolling_100f_chunk = 6
    best_rolling_100f_best_delta = -1.3009993775m
    gate_pass = false

C9:
    best_ATE_candidate = SWA_01_NONOVERLAP_RISK_REMOVE
    best_ATE_chunk = 6
    best_ATE_delta_vs_base = -0.7081471064m
    best_200_300_candidate = SWA_01_NONOVERLAP_RISK_REMOVE
    best_200_300_chunk = 10
    best_200_300_delta_vs_base = -2.3564699828m
    best_rolling_100f_candidate = SWA_01_NONOVERLAP_RISK_REMOVE
    best_rolling_100f_chunk = 6
    best_rolling_100f_best_delta = -1.3354202480m
    gate_pass = false
```

Decision：

```text
Track 2 h10 gate = fail.
No Track 2 h15 is launched.
```

## 2026-05-25 Track 3 TTT h10 R1

Command：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T3_CANDIDATES='V31_BASE_H9_REFERENCE,TTT_01_STRUCTURE_LONG,TTT_02_STRUCTURE_LONG_DYNAMIC_NOLONG,TTT_03_VEGETATION_CONDITIONAL_SHORTNEG,TTT_04_LOWTRUST_SHORTNEG,TTT_05_SKY_NEUTRAL,TTT_06_FULL_LIFECYCLE_POLICY'
V36B_RESULT_ROOT="$V38_ROOT" \
V36B_PHASE_NAME=phase3_ttt/h10_R1 \
V36B_RUN_PREFIX_BASE=V38_TRACK3_H10_R1 \
V36B_CANDIDATES="$V38_T3_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3,4,5,6,7 \
bash tools/run_v36b_path_h10.sh
```

Output dirs：

```text
phase3_ttt/h10_R1/rollouts/
matrix_logs/phase3_ttt/h10_R1/
```

Initial stdout：

```text
[2026-05-25 03:13:02] START parent=H9 candidate=V31_BASE_H9_REFERENCE chunk=10 h=10 gpu=0
[2026-05-25 03:13:02] START parent=H9 candidate=V31_BASE_H9_REFERENCE chunk=6 h=10 gpu=1
[2026-05-25 03:13:02] START parent=H9 candidate=TTT_01_STRUCTURE_LONG chunk=6 h=10 gpu=2
[2026-05-25 03:13:02] START parent=H9 candidate=TTT_02_STRUCTURE_LONG_DYNAMIC_NOLONG chunk=6 h=10 gpu=3
[2026-05-25 03:13:02] START parent=H9 candidate=TTT_03_VEGETATION_CONDITIONAL_SHORTNEG chunk=6 h=10 gpu=4
[2026-05-25 03:13:02] START parent=H9 candidate=TTT_04_LOWTRUST_SHORTNEG chunk=6 h=10 gpu=5
[2026-05-25 03:13:02] START parent=H9 candidate=TTT_05_SKY_NEUTRAL chunk=6 h=10 gpu=6
[2026-05-25 03:13:02] START parent=H9 candidate=TTT_06_FULL_LIFECYCLE_POLICY chunk=6 h=10 gpu=7
```

Tail stdout：

```text
[2026-05-25 05:05:03] DONE V38_TRACK3_H10_R1_C9_TTT_06_FULL_LIFECYCLE_POLICY_chunk16_h10_globalgate_H9parent_SWKS3
[2026-05-25 05:05:03] END parent=C9 candidate=TTT_06_FULL_LIFECYCLE_POLICY chunk=16 h=10 gpu=2
```

Integrity check：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/phase3_ttt/h10_R1/rollouts
printf 'run_dirs='; find "$ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l
printf 'done_status='; find "$ROOT" -maxdepth 2 -name run_status.txt -print0 | \
  xargs -0 -r rg -l '^\[[^]]+\] DONE ' | wc -l
printf 'fail_logs='; rg -n 'FAIL|Traceback|CUDA out of memory|Unsupported|error:' \
  results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/matrix_logs/phase3_ttt/h10_R1 || true
```

Output：

```text
run_dirs=42
done_status=42
fail_logs=
```

Track 3 h10 report：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T3_CANDIDATES='V31_BASE_H9_REFERENCE,TTT_01_STRUCTURE_LONG,TTT_02_STRUCTURE_LONG_DYNAMIC_NOLONG,TTT_03_VEGETATION_CONDITIONAL_SHORTNEG,TTT_04_LOWTRUST_SHORTNEG,TTT_05_SKY_NEUTRAL,TTT_06_FULL_LIFECYCLE_POLICY'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V38_ROOT/phase3_ttt/h10_R1/rollouts" \
  --run-prefix V38_TRACK3_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V38_T3_CANDIDATES" \
  --out-dir "$V38_ROOT/phase3_ttt/report_h10_R1" \
  --report-prefix track3_h10 \
  --mode h10
```

Report summary：

```text
rows = 42
missing_rows = 0
all_rows_done = true
gate_pass = false

H9:
    best_ATE_candidate = TTT_04_LOWTRUST_SHORTNEG
    best_ATE_chunk = 6
    best_ATE_delta_vs_base = -0.1663926129m
    best_200_300_candidate = TTT_04_LOWTRUST_SHORTNEG
    best_200_300_chunk = 6
    best_200_300_delta_vs_base = -0.2098918009m
    best_rolling_100f_candidate = TTT_04_LOWTRUST_SHORTNEG
    best_rolling_100f_chunk = 6
    best_rolling_100f_best_delta = -0.3978539997m
    gate_pass = false

C9:
    best_ATE_candidate = TTT_04_LOWTRUST_SHORTNEG
    best_ATE_chunk = 6
    best_ATE_delta_vs_base = -0.1105288332m
    best_200_300_candidate = TTT_04_LOWTRUST_SHORTNEG
    best_200_300_chunk = 6
    best_200_300_delta_vs_base = -0.1461979325m
    best_rolling_100f_candidate = TTT_04_LOWTRUST_SHORTNEG
    best_rolling_100f_chunk = 6
    best_rolling_100f_best_delta = -0.2513232702m
    gate_pass = false
```

Decision：

```text
Track 3 h10 gate = fail.
No Track 3 h15 is launched.
```

## 2026-05-25 Track 4 Semantic C23 Residual Path Isolation h10 R1

Command：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T4_CANDIDATES='V31_BASE_H9_REFERENCE,C23R_01_READ_ONLY_RESID,C23R_02_NO_TTT,C23R_03_NO_SWA,C23R_04_FRAMEGLOBAL_COMPACT_ONLY,C23R_05_STATIC_RESCUE_RESID,C23R_06_C9_COMPAT_READ_ONLY'
V36B_RESULT_ROOT="$V38_ROOT" \
V36B_PHASE_NAME=phase4_semc23/h10_R1 \
V36B_RUN_PREFIX_BASE=V38_TRACK4_H10_R1 \
V36B_CANDIDATES="$V38_T4_CANDIDATES" \
V36B_CHUNKS=6,10,16 \
V36B_PARENTS=H9,C9 \
V36B_HORIZON=10 \
V36B_GPUS=0,1,2,3,4,5,6,7 \
bash tools/run_v36b_path_h10.sh
```

Output dirs：

```text
phase4_semc23/h10_R1/rollouts/
matrix_logs/phase4_semc23/h10_R1/
```

Initial stdout：

```text
[2026-05-25 05:06:20] START parent=H9 candidate=V31_BASE_H9_REFERENCE chunk=10 h=10 gpu=0
[2026-05-25 05:06:20] START parent=H9 candidate=V31_BASE_H9_REFERENCE chunk=6 h=10 gpu=1
[2026-05-25 05:06:20] START parent=H9 candidate=C23R_01_READ_ONLY_RESID chunk=6 h=10 gpu=2
[2026-05-25 05:06:20] START parent=H9 candidate=C23R_02_NO_TTT chunk=6 h=10 gpu=3
[2026-05-25 05:06:20] START parent=H9 candidate=C23R_03_NO_SWA chunk=6 h=10 gpu=4
[2026-05-25 05:06:20] START parent=H9 candidate=C23R_04_FRAMEGLOBAL_COMPACT_ONLY chunk=6 h=10 gpu=5
[2026-05-25 05:06:20] START parent=H9 candidate=C23R_05_STATIC_RESCUE_RESID chunk=6 h=10 gpu=6
[2026-05-25 05:06:20] START parent=H9 candidate=C23R_06_C9_COMPAT_READ_ONLY chunk=6 h=10 gpu=7
```

Tail stdout：

```text
[2026-05-25 05:54:48] DONE V38_TRACK4_H10_R1_C9_C23R_05_STATIC_RESCUE_RESID_chunk16_h10_globalgate_H9parent_SWKS3
[2026-05-25 05:54:48] END parent=C9 candidate=C23R_05_STATIC_RESCUE_RESID chunk=16 h=10 gpu=1
[2026-05-25 05:56:04] DONE V38_TRACK4_H10_R1_C9_C23R_06_C9_COMPAT_READ_ONLY_chunk16_h10_globalgate_H9parent_SWKS3
[2026-05-25 05:56:04] END parent=C9 candidate=C23R_06_C9_COMPAT_READ_ONLY chunk=16 h=10 gpu=2
```

Integrity check：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/phase4_semc23/h10_R1/rollouts
printf 'run_dirs='; find "$ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l
printf 'done_status='; find "$ROOT" -maxdepth 2 -name run_status.txt -print0 | \
  xargs -0 -r rg -l '^\[[^]]+\] DONE ' | wc -l
printf 'fail_logs='; rg -n 'FAIL|Traceback|CUDA out of memory|Unsupported|error:' \
  results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/matrix_logs/phase4_semc23/h10_R1 || true
```

Output：

```text
run_dirs=42
done_status=42
fail_logs=
```

Track 4 h10 report：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
V38_T4_CANDIDATES='V31_BASE_H9_REFERENCE,C23R_01_READ_ONLY_RESID,C23R_02_NO_TTT,C23R_03_NO_SWA,C23R_04_FRAMEGLOBAL_COMPACT_ONLY,C23R_05_STATIC_RESCUE_RESID,C23R_06_C9_COMPAT_READ_ONLY'
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v38_durability_report.py \
  --rollout-root "$V38_ROOT/phase4_semc23/h10_R1/rollouts" \
  --run-prefix V38_TRACK4_H10_R1 \
  --parents H9,C9 \
  --chunks 6,10,16 \
  --horizon 10 \
  --candidates "$V38_T4_CANDIDATES" \
  --out-dir "$V38_ROOT/phase4_semc23/report_h10_R1" \
  --report-prefix track4_h10 \
  --mode h10
```

Report summary：

```text
rows = 42
missing_rows = 0
all_rows_done = true
gate_pass = false

H9:
    best_ATE_candidate = C23R_02_NO_TTT
    best_ATE_chunk = 6
    best_ATE_delta_vs_base = -0.4807208363m
    best_200_300_candidate = C23R_02_NO_TTT
    best_200_300_chunk = 10
    best_200_300_delta_vs_base = -1.2784762533m
    best_rolling_100f_candidate = C23R_02_NO_TTT
    best_rolling_100f_chunk = 6
    best_rolling_100f_best_delta = -0.8477685297m
    gate_pass = false

C9:
    best_ATE_candidate = C23R_02_NO_TTT
    best_ATE_chunk = 6
    best_ATE_delta_vs_base = -0.4976377546m
    best_200_300_candidate = C23R_02_NO_TTT
    best_200_300_chunk = 10
    best_200_300_delta_vs_base = -1.2952954518m
    best_rolling_100f_candidate = C23R_02_NO_TTT
    best_rolling_100f_chunk = 6
    best_rolling_100f_best_delta = -0.8793798994m
    gate_pass = false
```

Decision：

```text
Track 4 h10 gate = fail.
No Track 4 h15 is launched.
```

## 2026-05-25 Final summary reports

Command：

```bash
V38_ROOT=results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v38_final_summary_report.py --root "$V38_ROOT"
```

Output：

```json
{
  "best_deployable_online": "C9_P0_R2",
  "best_deployable_online_ate": 33.7629421029,
  "target30_success": false,
  "track0_gate_pass": true,
  "track1_h10_gate_pass": false,
  "track2_h10_gate_pass": false,
  "track2_h15_gate_pass": null,
  "track3_h10_gate_pass": false,
  "track3_h15_gate_pass": null,
  "track4_h10_gate_pass": false,
  "track5_full_online_allowed": false,
  "track5_full_online_launched": false
}
```

Generated：

```text
final_reports/track0_action_influence_report.md
final_reports/track1_frameglobal_report.md
final_reports/track2_swa_report.md
final_reports/track3_ttt_report.md
final_reports/track4_semantic_c23_report.md
final_reports/track5_full_online_report.md
final_reports/failure_routing_summary.md
final_reports/short_rollout_delta_bar_chart.png
final_reports/h10_h15_durability_curve.png
final_reports/v38_final_summary.json
```

Final validation：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v38_durability_report.py \
  tools/v38_action_influence_postprocess.py \
  tools/v38_final_summary_report.py
bash -n tools/run_v24_candidate_rollout.sh
```

Output：

```text
PASS (commands produced no stderr/stdout)
```

Landed row count audit：

```bash
for phase in phase0_action_influence/h3_R1 phase1_frameglobal/h10_R1 \
  phase2_swa/h10_R1 phase3_ttt/h10_R1 phase4_semc23/h10_R1; do
  ROOT="results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/$phase/rollouts"
  printf '%s run_dirs=' "$phase"; find "$ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l
  printf '%s done=' "$phase"; find "$ROOT" -maxdepth 2 -name run_status.txt -print0 | \
    xargs -0 -r rg -l '^\[[^]]+\] DONE ' | wc -l
done
```

Output：

```text
phase0_action_influence/h3_R1 run_dirs=108
phase0_action_influence/h3_R1 done=108
phase1_frameglobal/h10_R1 run_dirs=48
phase1_frameglobal/h10_R1 done=48
phase2_swa/h10_R1 run_dirs=42
phase2_swa/h10_R1 done=42
phase3_ttt/h10_R1 run_dirs=42
phase3_ttt/h10_R1 done=42
phase4_semc23/h10_R1 run_dirs=42
phase4_semc23/h10_R1 done=42
```

新增 Track 0 v38 postprocess：

```text
tools/v38_action_influence_postprocess.py:
    augments v37-style action atlas outputs into v38 requested filenames:
        attention_mass_removed_before_after.csv
        frame_source_keep_ratio_by_label.csv
        global_source_keep_ratio_by_label.csv
        swa_overlap_keep_ratio_by_label.csv
        swa_nonoverlap_keep_ratio_by_label.csv
        ttt_role_mass_by_label.csv
        source_attention_mass_by_label.csv
        swa_source_attention_mass_by_label.csv
        ttt_post_zp_update_norm_by_label.csv
        source_attention_mass_removed_bar.png
        swa_overlap_nonoverlap_keep_bar.png
        ttt_role_mass_by_label_bar.png
        influence_atlas_by_chunk.png

Boundary:
    per-label/per-masklet tensor fields are written as explainability_missing
    when not landed. No pixel/tensor visualization is fabricated.
```

Validation：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v38_durability_report.py \
  tools/v38_action_influence_postprocess.py
bash -n tools/run_v24_candidate_rollout.sh
```

输出为空，py_compile / bash syntax pass。

新增 final summary 汇总脚本：

```text
tools/v38_final_summary_report.py:
    consumes landed Track 0-5 JSON reports only
    writes final_reports/ Markdown summaries and charts
    marks Track 5 as not launched when no h15-qualified candidate exists
```

Validation：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v38_durability_report.py \
  tools/v38_action_influence_postprocess.py \
  tools/v38_final_summary_report.py
```

输出为空，py_compile pass。
