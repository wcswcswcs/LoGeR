# ACL2 v42 执行日志：C9 Health-Gated READ Semantic Target30

日期：2026-05-26（Asia/Singapore）

计划文件：

```text
docs/ACL2_v42_C9_HealthGated_READ_Semantic_Target30_Plan.md
```

主结果目录：

```text
results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/
```

原则：

```text
只记录实际执行命令与落盘结果。
不把 health detector、mechanism audit、short/proxy evidence 写成 deployable online success。
每个 LoGeR 进程只绑定一张 GPU。
用户确认 GPU 0,1,2,3,4,5,6,7 可用。
```

---

## 2026-05-26 初始化

### 阅读计划

```bash
sed -n '1,220p' docs/ACL2_v42_C9_HealthGated_READ_Semantic_Target30_Plan.md
sed -n '220,520p' docs/ACL2_v42_C9_HealthGated_READ_Semantic_Target30_Plan.md
sed -n '520,760p' docs/ACL2_v42_C9_HealthGated_READ_Semantic_Target30_Plan.md
sed -n '760,960p' docs/ACL2_v42_C9_HealthGated_READ_Semantic_Target30_Plan.md
```

### 建立结果目录与 parent snapshot symlink

```bash
mkdir -p results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30
ln -sfn ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase0_parent_snapshots
```

结果：

```text
phase0_parent_snapshots -> ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots
```

### 工程审计

确认 C9 locked 配置来源：

```bash
rg -n "ttt_write_gradient|ttt_write_tri|mp_alpha|read_beta_frame_chunks|hmc_commit_mode|read_cue_source|hybrid_memory_mode" \
  results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/phase0_boundary/V16_P0_R2_C9_locked_exact_merge_input_SWKS3/hmc_config.yaml \
  results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/phase0_repro/V15_P0_A2_C9_REPEAT_no_state_save_SWKS3/hmc_config.yaml
```

确认 v42 需要的 action chunk gate 之前不存在：

```bash
rg -n "v32_semantic_cue_active_chunks|semantic_role_policy|enable_context_source_skip" \
  run_pipeline_abc_v2.py loger/pipeline/hybrid_memory_controller.py tools/run_attention_cue_experiment.sh
```

### 工程修改：health-selected chunk action gate

修改文件：

```text
loger/pipeline/hybrid_memory_controller.py
run_pipeline_abc_v2.py
tools/run_attention_cue_experiment.sh
```

修改内容：

```text
1. 新增 semantic_action_active_chunks 参数，支持逗号 chunk 和闭区间 ranges。
2. 在 HMC control bundle 中仅当 current_chunk_idx 位于 active chunk set 时启用：
       semantic_role_policy
       semantic_memory_paths
       context_source_skip
3. C9 read cue / TTT write / 原生 C9 SWA overlap replace 不由该 gate 改写。
   这样 v42 候选是 C9 baseline + health-selected chunk READ source filtering。
4. control bundle 记录：
       semantic_action_chunk_gate_mode
       semantic_action_chunk_gate_active
       semantic_action_active_chunks
       semantic_action_chunk_idx
5. 追加 semantic_action_inactive_read_cue_source：
       候选 READ cue 只在 active chunks 生效；
       inactive chunks 回退到 C9 原始 read cue。
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/pipeline/hybrid_memory_controller.py \
  run_pipeline_abc_v2.py

bash -n tools/run_attention_cue_experiment.sh
```

结果：

```text
PASS
```

### F0 R1 完成但 Phase 0 no-op gate 失败

F0 R1 完成：

```text
run = V42_P0_F0_C9_REFERENCE_REPEAT
status = DONE
hmc_rows = 38
```

生成 Phase 0 单行 report：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v42_full_online_report.py \
  --rollout-root \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/rollouts \
  --runs F0=V42_P0_F0_C9_REFERENCE_REPEAT \
  --reference-name F0 \
  --out-dir \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase0_noop_report
```

落盘结果：

```text
reference_ATE_full = 34.270599734227254
historical_c9_ate = 33.7629421029
delta_vs_historical_C9 = +0.507657631327254m
Phase 0 no-op gate = FAIL
```

按计划处理：

```text
Phase 0 no-op 不过时，不能解释 downstream strategy rows。
```

### Phase 1 detector R1（基于 F0 R1，仅作为诊断）

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v42_health_detector_report.py \
  --reference-run \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/rollouts/V42_P0_F0_C9_REFERENCE_REPEAT \
  --out-dir \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase1_health
```

落盘结果：

```text
phase1_gate_pass = true
selected_bad_chunks = [7,9,12,14,16,17,19]
selected_bad_chunk_ratio = 0.18421052631578946
selection_uses_ATE = false
selection_uses_fixed_chunk_or_segment = false
rolling100_used_for_selection = false
```

Boundary：

```text
Because F0 R1 failed the Phase 0 no-op gate, this detector output is diagnostic
only until F0 is repaired and rerun.
```

### Provisional F1-F5 chunk10 rows invalidated

Before Phase 1 completed, provisional F1-F5 were launched with v41 chunk10.
After Phase 1 R1 selected `[7,9,12,14,16,17,19]`, those rows were stopped and
moved aside:

```bash
for pg in 1460105 1460107 1460125 1460133 1460145; do
  kill -TERM -- -$pg 2>/dev/null || true
done

mkdir -p \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/provisional_invalid_chunk10_mismatch

mv phase3_full_online/rollouts/V42_F1_C9_R1_HEALTHGATED_READ \
   phase3_full_online/provisional_invalid_chunk10_mismatch/
mv phase3_full_online/rollouts/V42_F2_C9_R2_SKY_APP_READ \
   phase3_full_online/provisional_invalid_chunk10_mismatch/
mv phase3_full_online/rollouts/V42_F3_C9_R3_STATIC_RESCUE_READ \
   phase3_full_online/provisional_invalid_chunk10_mismatch/
mv phase3_full_online/rollouts/V42_F4_C9_R3_EPISODE_FOLLOW_READ \
   phase3_full_online/provisional_invalid_chunk10_mismatch/
mv phase3_full_online/rollouts/V42_F5_H9_R1_DIAG \
   phase3_full_online/provisional_invalid_chunk10_mismatch/
```

Boundary：

```text
These rows were partial START-only artifacts, not DONE evidence, and are not
counted in any v42 gate.
```

### Health-selected F1-F5 R1 also invalidated because Phase 0 failed

After Phase 1 R1 selected chunks, F1-F5 were relaunched with health-selected
chunks, but then F0 R1 no-op failure was detected. Per plan, these rows cannot
be interpreted and were stopped:

```bash
SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19 \
  tools/run_v42_full_candidate.sh 0 \
  V42_F1_HEALTHSEL_R1_HIGH_INFLUENCE_READ F1_R1_HIGH_INFLUENCE_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19 \
  tools/run_v42_full_candidate.sh 1 \
  V42_F2_HEALTHSEL_R2_SKY_APP_READ F2_R2_SKY_APP_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19 \
  tools/run_v42_full_candidate.sh 2 \
  V42_F3_HEALTHSEL_R3_STATIC_RESCUE_READ F3_R3_STATIC_RESCUE_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=7,8,9,10,12,13,14,15,16,17,18,19,20 \
  tools/run_v42_full_candidate.sh 3 \
  V42_F4_HEALTHSEL_R3_EPISODE_FOLLOW_READ F4_R3_EPISODE_FOLLOW_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19 \
  tools/run_v42_full_candidate.sh 4 \
  V42_F5_HEALTHSEL_H9_R1_DIAG F5_H9_R1_DIAG
```

Stop / quarantine:

```bash
for pg in 1524059 1524067 1524076 1524089 1524090; do
  kill -TERM -- -$pg 2>/dev/null || true
done

mkdir -p \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/invalid_phase0_noop_failed
```

Moved partial directories under:

```text
phase3_full_online/invalid_phase0_noop_failed/
```

Boundary：

```text
These health-selected rows were not allowed evidence because Phase 0 had not
passed. They are not counted in v42 final metrics.
```

### Blocker：v42 C9 launcher was not exact C9

Config diff audit:

```text
Historical C9:
    beta_frame = 4.75
    beta_swa = 4.75
    stage_c_mode = none
    stage_c_cache_mode = off
    ttt_write_commit_ema_alpha = 0.5
    ttt_write_commit_ema_branch_mask = 0
    ttt_write_commit_ema_chunks = 5,6
    ttt_write_native_mix_scales = 1.10,1.00,1.00

v42 F0 R1:
    beta_frame = 4.25
    beta_swa = 4.25
    stage_c_mode = reference
    stage_c_cache_mode = read
    ttt_write_commit_ema_alpha = 1.0
    ttt_write_commit_ema_branch_mask = all
    ttt_write_commit_ema_chunks = empty
    ttt_write_native_mix_scales = empty
```

修复：

```text
tools/run_v42_full_candidate.sh:
    restored C9-locked defaults for F0:
        beta = 4.75
        Stage C default off
        TTT commit EMA defaults restored
        native mix scales restored

    Stage C cache is now enabled only inside F1-F5 semantic READ candidates.

loger/pipeline/hybrid_memory_controller.py:
    outside semantic_action_active_chunks:
        read_cue_source falls back to C9 cue
        prior_output is now passed as None
    This prevents Stage-C semantic prior from perturbing non-selected chunks.
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/pipeline/hybrid_memory_controller.py \
  run_pipeline_abc_v2.py

bash -n tools/run_v42_full_candidate.sh tools/run_attention_cue_experiment.sh
```

结果：

```text
PASS
```

### Phase 0 repair rerun：F0 R2

启动：

```bash
tools/run_v42_full_candidate.sh \
  0 \
  V42_P0_F0_C9_REFERENCE_REPEAT_R2 \
  F0_C9_REFERENCE
```

状态：

```text
RUNNING on GPU 0 since 2026-05-26 08:09:54
```

配置抽检：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
import yaml
from pathlib import Path
old = Path('results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/phase0_boundary/V16_P0_R2_C9_locked_exact_merge_input_SWKS3/hmc_config.yaml')
new = Path('results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/rollouts/V42_P0_F0_C9_REFERENCE_REPEAT_R2/hmc_config.yaml')
...
PY
```

关键项对齐：

```text
beta_frame: old=4.75 new=4.75
beta_swa: old=4.75 new=4.75
stage_c_mode: old='none' new='none'
stage_c_cache_mode: old='off' new='off'
ttt_write_commit_ema_alpha: old=0.5 new=0.5
ttt_write_commit_ema_branch_mask: old='0' new='0'
ttt_write_commit_ema_chunks: old='5,6' new='5,6'
ttt_write_native_mix_scales: old='1.10,1.00,1.00' new='1.10,1.00,1.00'
read_cue_source: old=C9 cue new=C9 cue
read_calib_mode: old='none' new='none'
mp_alpha: old=0.1 new=0.1
hmc_commit_mode: old='probe_ttt_write' new='probe_ttt_write'
hmc_write_score_source: old='stage_d_x_dg_inv_sqrt' new='stage_d_x_dg_inv_sqrt'
enable_swa_overlap_source_replace: old=1 new=1
```

Runtime status check：

```text
2026-05-26 08:13 approx:
    hmc_rows = 5
    pred_files = 0
    GPU0 memory used ~14432 MiB
```

### Phase 0 F0 R2 完成并通过 no-op gate

F0 R2 完成：

```text
[2026-05-26 08:41:51] DONE V42_P0_F0_C9_REFERENCE_REPEAT_R2
hmc_rows = 38
pred_files = 1
```

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v42_full_online_report.py \
  --rollout-root \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/rollouts \
  --runs F0=V42_P0_F0_C9_REFERENCE_REPEAT_R2 \
  --reference-name F0 \
  --out-dir \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase0_noop_report_R2
```

落盘结果：

```text
reference_ATE_full = 33.76294210291885
historical_c9_ate = 33.7629421029
abs_delta = 0.00000000001885m
Phase 0 no-op gate = PASS
```

### Phase 1 R2 health detector

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v42_health_detector_report.py \
  --reference-run \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/rollouts/V42_P0_F0_C9_REFERENCE_REPEAT_R2 \
  --out-dir \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase1_health_R2
```

落盘结果：

```text
phase1_gate_pass = true
health_chunk_count = 38
hmc_rows = 38
selected_bad_chunks = [7,9,12,14,16,17,19]
selected_bad_chunk_ratio = 0.18421052631578946
selection_uses_ATE = false
selection_uses_fixed_chunk_or_segment = false
rolling100_used_for_selection = false
top_rolling100_bad_chunk_diagnostic = 6
```

### Phase 0 repaired short no-op smoke R3

命令：

```bash
START_FRAME=0 END_FRAME=120 \
  tools/run_v42_full_candidate.sh \
  5 \
  V42_P0_HEALTH_LOGGING_ONLY_H3_R3 \
  F0_C9_REFERENCE

START_FRAME=0 END_FRAME=120 \
  STAGE_C_CACHE_REQUIRE_HIT_OVERRIDE=0 \
  SEMANTIC_ACTION_ACTIVE_CHUNKS=999 \
  tools/run_v42_full_candidate.sh \
  6 \
  V42_P0_READ_HOOK_NOOP_H3_R3 \
  F1_R1_HIGH_INFLUENCE_READ
```

状态：

```text
V42_P0_HEALTH_LOGGING_ONLY_H3_R3:
    DONE at 2026-05-26 08:45:19
    hmc_rows = 5
    pred_files = 1

V42_P0_READ_HOOK_NOOP_H3_R3:
    DONE at 2026-05-26 08:50:31
    hmc_rows = 5
    pred_files = 1
```

### Phase 2 R2 mechanism report

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v42_read_mechanism_report.py \
  --selected-json \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase1_health_R2/selected_bad_chunks.json \
  --out-dir \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase2_read_mechanism_R2
```

落盘结果：

```text
mechanism_decision = B_general_high_influence_anomaly_preferred
explainability_level = incomplete_explainability_selected_chunk_not_chunk010_proxy
general_anomaly_supported = true
sky_causality_supported = false
static_anchor_misdamage_risk = not_proven_from_spatial_maps
scalar_attention_mass_rows = 144
proxy_overlays_copied = 3
```

Boundary：

```text
Selected v42 chunks do not include chunk10, while the richest v41 proxy
attribution evidence was chunk10. The report reuses only landed scalar/proxy
evidence and marks spatial causality incomplete rather than reconstructing it.
```

### Phase 3 official F1-F5 R2 full-online launch

F1/F2/F3/F5 use exactly selected chunks:

```text
SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19
```

F4 uses selected chunks plus one-step follow-through approximation:

```text
SEMANTIC_ACTION_ACTIVE_CHUNKS=7,8,9,10,12,13,14,15,16,17,18,19,20
```

命令：

```bash
SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19 \
  tools/run_v42_full_candidate.sh 0 \
  V42_F1_R2_HEALTHSEL_R1_HIGH_INFLUENCE_READ F1_R1_HIGH_INFLUENCE_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19 \
  tools/run_v42_full_candidate.sh 1 \
  V42_F2_R2_HEALTHSEL_R2_SKY_APP_READ F2_R2_SKY_APP_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19 \
  tools/run_v42_full_candidate.sh 2 \
  V42_F3_R2_HEALTHSEL_R3_STATIC_RESCUE_READ F3_R3_STATIC_RESCUE_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=7,8,9,10,12,13,14,15,16,17,18,19,20 \
  tools/run_v42_full_candidate.sh 3 \
  V42_F4_R2_HEALTHSEL_R3_EPISODE_FOLLOW_READ F4_R3_EPISODE_FOLLOW_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=7,9,12,14,16,17,19 \
  tools/run_v42_full_candidate.sh 4 \
  V42_F5_R2_HEALTHSEL_H9_R1_DIAG F5_H9_R1_DIAG
```

状态：

```text
RUNNING since 2026-05-26 08:42:42
```

Action gate audit while rows running：

```text
F1:
    chunk7 cue = old_dyn_switch_flow_sem_veto
    chunk7 chunk/frame source_skip = 6/6
    chunk7 empty_source_events = 0/0
    chunk7 attention_mass_removed_before = 0.1705431342
    chunk8 cue = C9 cue
    chunk8 chunk/frame source_skip = 0/0

F2:
    chunk7 cue = old_dyn_switch_flow_sem_veto
    chunk7 chunk/frame source_skip = 6/6
    chunk8 cue = C9 cue
    chunk8 chunk/frame source_skip = 0/0

F3:
    chunk7 cue = old_dyn_key_static_rescue
    chunk7 chunk/frame source_skip = 6/6
    chunk7 attention_mass_removed_before = 0.2078069001
    chunk8 cue = C9 cue
    chunk8 chunk/frame source_skip = 0/0

F4:
    chunk7 cue = old_dyn_key_static_rescue
    chunk7 chunk/frame source_skip = 6/6
    chunk8 cue = old_dyn_key_static_rescue
    chunk8 chunk/frame source_skip = 6/6
    This matches the one-step follow-through approximation.

F5:
    chunk7 cue = old_dyn_switch_flow_sem_veto
    chunk7 chunk/frame source_skip = 6/6
    chunk8 cue = C9 cue
    chunk8 chunk/frame source_skip = 0/0
```

### Phase 3 full-online R2 completion

Completion status：

```text
V42_F1_R2_HEALTHSEL_R1_HIGH_INFLUENCE_READ:
    DONE at 2026-05-26 10:06:00

V42_F2_R2_HEALTHSEL_R2_SKY_APP_READ:
    DONE at 2026-05-26 09:58:38

V42_F3_R2_HEALTHSEL_R3_STATIC_RESCUE_READ:
    DONE at 2026-05-26 10:03:15

V42_F4_R2_HEALTHSEL_R3_EPISODE_FOLLOW_READ:
    DONE at 2026-05-26 10:03:53

V42_F5_R2_HEALTHSEL_H9_R1_DIAG:
    DONE at 2026-05-26 09:58:07
```

Report command：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v42_full_online_report.py \
  --rollout-root \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/rollouts \
  --runs \
  F0=V42_P0_F0_C9_REFERENCE_REPEAT_R2,F1=V42_F1_R2_HEALTHSEL_R1_HIGH_INFLUENCE_READ,F2=V42_F2_R2_HEALTHSEL_R2_SKY_APP_READ,F3=V42_F3_R2_HEALTHSEL_R3_STATIC_RESCUE_READ,F4=V42_F4_R2_HEALTHSEL_R3_EPISODE_FOLLOW_READ,F5=V42_F5_R2_HEALTHSEL_H9_R1_DIAG \
  --reference-name F0 \
  --out-dir \
  results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/final_reports
```

Generated：

```text
final_reports/v42_full_online_report.md
final_reports/v42_full_online_summary.json
final_reports/full_online_registry.csv
```

Summary output：

```text
rows = 6
done_rows = 6
reference_ATE_full = 33.76294210291885
historical_c9_ate = 33.7629421029
best_read_candidate = F1
best_read_ATE_full = 34.7539112803563
best_read_delta_vs_reference = +0.9909691774374494
best_read_delta_vs_historical_c9 = +0.9909691774563001
minimum_progress_pass = false
stage_success_pass = false
strong_success_pass = false
target30_success = false
phase4_allowed = false
```

Full rows：

```text
F0:
    ATE = 33.7629421029
    [200,300) = 76.1021355543
    [400,600) = 41.8963642126

F1:
    ATE = 34.7539112804
    delta vs F0 = +0.9909691774
    [200,300) = 76.3653263460
    [400,600) = 43.6064222084

F2:
    ATE = 39.3352485068
    delta vs F0 = +5.5723064039
    [200,300) = 84.3972404066
    [400,600) = 49.6447573571

F3:
    ATE = 38.2193997350
    delta vs F0 = +4.4564576321
    [200,300) = 83.0529914780
    [400,600) = 47.7645482227

F4:
    ATE = 48.1789570268
    delta vs F0 = +14.4160149239
    [200,300) = 83.0021804111
    [400,600) = 62.2024339052

F5:
    ATE = 34.7524824725
    delta vs F0 = +0.9895403696
    [200,300) = 76.4623800233
    [400,600) = 43.5591122658
```

Decision：

```text
Phase 3 READ-only full online = fail.
Phase 4 memory barrier = not allowed.
No Target-30 success.
No deployable online success.
```

### Phase 3 provisional full-online launches

原因：

```text
F0 no-op full run was still running on GPU 0.
To use available GPUs, F1-F5 were launched with the v41 health-detector selected
chunk [10] as provisional action chunks.

Boundary:
    These rows are counted only if the v42 Phase 1 full-sequence detector
    confirms compatible selected chunks. If v42 Phase 1 selects different
    chunks, these rows must be marked provisional/invalid and rerun.
```

启动：

```bash
SEMANTIC_ACTION_ACTIVE_CHUNKS=10 \
  tools/run_v42_full_candidate.sh 1 V42_F1_C9_R1_HEALTHGATED_READ F1_R1_HIGH_INFLUENCE_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=10 \
  tools/run_v42_full_candidate.sh 2 V42_F2_C9_R2_SKY_APP_READ F2_R2_SKY_APP_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=10 \
  tools/run_v42_full_candidate.sh 3 V42_F3_C9_R3_STATIC_RESCUE_READ F3_R3_STATIC_RESCUE_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=10,11 \
  tools/run_v42_full_candidate.sh 4 V42_F4_C9_R3_EPISODE_FOLLOW_READ F4_R3_EPISODE_FOLLOW_READ

SEMANTIC_ACTION_ACTIVE_CHUNKS=10 \
  tools/run_v42_full_candidate.sh 5 V42_F5_H9_R1_DIAG F5_H9_R1_DIAG
```

状态：

```text
RUNNING since 2026-05-26 07:30:58
GPU 1: F1
GPU 2: F2
GPU 3: F3
GPU 4: F4
GPU 5: F5
```

### Phase 0 short no-op smokes

启动：

```bash
START_FRAME=0 END_FRAME=120 \
  tools/run_v42_full_candidate.sh 6 V42_P0_HEALTH_LOGGING_ONLY_H3 F0_C9_REFERENCE

START_FRAME=0 END_FRAME=120 SEMANTIC_ACTION_ACTIVE_CHUNKS=999 \
  tools/run_v42_full_candidate.sh 7 V42_P0_READ_HOOK_NOOP_H3 F1_R1_HIGH_INFLUENCE_READ
```

Boundary：

```text
V42_P0_READ_HOOK_NOOP_H3 uses a non-present action chunk 999 so the candidate
READ cue and semantic source-skip remain inactive for the h3 smoke frames and
fall back to C9 read cue. This is a hook no-op smoke, not a deployment row.
```

状态：

```text
RUNNING since 2026-05-26 07:32:23
GPU 6: health logging h3
GPU 7: READ hook inactive/no-op h3
```

### Blocker：short smoke partial chunk Stage-C cache miss

现象：

```text
V42_P0_HEALTH_LOGGING_ONLY_H3 failed:
    RuntimeError: Required Stage C cache miss for chunk 4:
    .../chunk_004_000116_000120/masklet.pt
```

原因：

```text
START_FRAME=0 END_FRAME=120 creates a non-standard partial chunk
chunk_004_000116_000120. The landed Stage-C cache contains standard chunk
windows, not this partial smoke window.
```

修复方向：

```text
Keep the smoke short, but allow Stage-C to compute cache-miss partial chunks
instead of requiring a cache hit. This affects only Phase 0 smoke rows, not the
full-online rows.
```

修复重跑：

```bash
START_FRAME=0 END_FRAME=120 STAGE_C_CACHE_REQUIRE_HIT=0 \
  tools/run_v42_full_candidate.sh 6 V42_P0_HEALTH_LOGGING_ONLY_H3_R2 F0_C9_REFERENCE
```

状态：

```text
RUNNING on GPU 6 since 2026-05-26 07:38:51
```

结果：

```text
V42_P0_HEALTH_LOGGING_ONLY_H3_R2 DONE at 2026-05-26 07:42:59.
V42_P0_READ_HOOK_NOOP_H3 also hit the same partial-chunk cache miss and failed.
```

READ hook no-op 修复重跑：

```bash
START_FRAME=0 END_FRAME=120 STAGE_C_CACHE_REQUIRE_HIT=0 SEMANTIC_ACTION_ACTIVE_CHUNKS=999 \
  tools/run_v42_full_candidate.sh 6 V42_P0_READ_HOOK_NOOP_H3_R2 F1_R1_HIGH_INFLUENCE_READ
```

状态：

```text
RUNNING on GPU 6 since 2026-05-26 07:43:50
```

结果：

```text
V42_P0_READ_HOOK_NOOP_H3_R2 DONE at 2026-05-26 07:52:08.
```

### 新增 Phase 2 READ mechanism report

新增：

```text
tools/v42_read_mechanism_report.py
```

功能：

```text
Reads selected_bad_chunks.json and landed v41 mechanism artifacts.
Copies proxy overlays when selected chunks overlap chunk010.
Marks missing per-label spatial source-attention / READ affected masks as
incomplete_explainability instead of reconstructing them.
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v42_read_mechanism_report.py
```

结果：

```text
PASS
```

### 新增 Phase 1 health detector report

新增：

```text
tools/v42_health_detector_report.py
```

功能：

```text
Reads F0 no-op hmc_state_hash.jsonl and builds:
    phase1_health/chunk_health_table.csv
    phase1_health/chunk_health_flags.jsonl
    phase1_health/health_component_by_chunk.csv
    phase1_health/selected_bad_chunks.json
    phase1_health/health_vs_rolling_window_diagnostic.csv
    phase1_health/chunk_health_timeline.png
    phase1_health/health_component_stackplot.png
    phase1_health/bad_chunk_report.md

Detector selection does not read ATE / segment labels.
rolling100 is written only as post-hoc diagnostic.
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v42_health_detector_report.py
```

结果：

```text
PASS
```

### Phase 0 / F0 C9 reference repeat

启动：

```bash
tools/run_v42_full_candidate.sh \
  0 \
  V42_P0_F0_C9_REFERENCE_REPEAT \
  F0_C9_REFERENCE
```

输出目录：

```text
results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/phase3_full_online/rollouts/V42_P0_F0_C9_REFERENCE_REPEAT/
```

状态：

```text
RUNNING on GPU 0 since 2026-05-26 07:23:27
```

### 新增 v42 full-online landed report

新增：

```text
tools/v42_full_online_report.py
```

功能：

```text
Reads only landed artifacts:
    01.txt
    run_status.txt
    hmc_state_hash.jsonl

Reports:
    full ATE / Rot / RPE / FinalErr
    [200,300), [400,600)
    rolling50 / rolling100 / rolling200
    action_active_chunks and context empty-source counts
    deltas vs F0 reference and historical C9
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v42_full_online_report.py
```

结果：

```text
PASS
```

### 新增 v42 full-online candidate launcher

新增：

```text
tools/run_v42_full_candidate.sh
```

用途：

```text
Run V42 F0-F5 full-online rows with C9 locked config defaults.
Each process binds one GPU through CUDA_VISIBLE_DEVICES inside
tools/run_attention_cue_experiment.sh.
Existing DONE rows with 01.txt are skipped.
```

候选：

```text
F0_C9_REFERENCE
F1_R1_HIGH_INFLUENCE_READ
F2_R2_SKY_APP_READ
F3_R3_STATIC_RESCUE_READ
F4_R3_EPISODE_FOLLOW_READ
F5_H9_R1_DIAG
```

验证：

```bash
chmod +x tools/run_v42_full_candidate.sh
bash -n tools/run_v42_full_candidate.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/pipeline/hybrid_memory_controller.py \
  run_pipeline_abc_v2.py
```

结果：

```text
PASS
```
