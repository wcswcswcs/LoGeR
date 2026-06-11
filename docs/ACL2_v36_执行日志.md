# ACL2 v36 执行日志：Training-Free Semantic Memory Control

日期：2026-05-24（Asia/Singapore）  
计划文件：`docs/ACL2_v36_TrainingFree_SemanticMemory_Control_Target30_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/`

本日志只记录实际执行过的命令、文件、修复和复现信息。不把未完成或被 gate 阻断的阶段写成成功。

---

## 0. Pre-flight：结束上次 v35B 残留

用户要求“上次实验先结束”。执行前先清理 v35B 残留进程，避免占用 GPU 或污染 v36 日志。

命令：

```bash
set -e
pids=$(ps -u "$USER" -o pid,cmd | awk '/V35B|run_v35b_controlled_probe|track_a_controlled_probe_R2/ && !/awk/ {print $1}' | tr '\n' ' ')
echo "V35B leftover pids: ${pids:-none}"
if [ -n "${pids:-}" ]; then kill $pids 2>/dev/null || true; sleep 2; fi
pids2=$(ps -u "$USER" -o pid,cmd | awk '/V35B|run_v35b_controlled_probe|track_a_controlled_probe_R2/ && !/awk/ {print $1}' | tr '\n' ' ')
echo "V35B remaining after TERM: ${pids2:-none}"
if [ -n "${pids2:-}" ]; then kill -9 $pids2 2>/dev/null || true; fi
ps -u "$USER" -o pid,ppid,stat,etime,cmd | rg 'V35B|run_v35b|controlled_probe|run_attention_cue_experiment|run_pipeline_abc_v2' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
```

结果：

```text
V35B leftover pids:
389353 400491 400492 400493 400494 400515 400516 400517
400518 400519 400520 400521 400522 400523 400524 400525
400526 400552 400553 400554 400555

V35B remaining after TERM: none

GPU memory after cleanup:
0, 1 MiB, 0 %
1, 1 MiB, 0 %
2, 1 MiB, 0 %
3, 4 MiB, 0 %
4, 4 MiB, 0 %
5, 4 MiB, 0 %
6, 1 MiB, 0 %
7, 4 MiB, 0 %
```

---

## 1. Plan Read / Hard Gate

读取计划：

```bash
sed -n '1,220p' docs/ACL2_v36_TrainingFree_SemanticMemory_Control_Target30_Plan.md
sed -n '221,520p' docs/ACL2_v36_TrainingFree_SemanticMemory_Control_Target30_Plan.md
sed -n '521,900p' docs/ACL2_v36_TrainingFree_SemanticMemory_Control_Target30_Plan.md
sed -n '901,1260p' docs/ACL2_v36_TrainingFree_SemanticMemory_Control_Target30_Plan.md
sed -n '1261,1700p' docs/ACL2_v36_TrainingFree_SemanticMemory_Control_Target30_Plan.md
```

关键结论：

```text
v36 不能先跑 rollout/full online。
必须先执行 H0 Action Realism / Hook Audit。
如果 synthetic mask stress 没有真实 effect、context empty、action collapse，
则禁止继续 H1-H5 rollout，必须先修 hook / projection / fallback / protected-token 逻辑。
```

---

## 2. 工程接线：v36 synthetic override

目的：

```text
H0 要求 synthetic mask stress test 不依赖 video masklet 预测。
因此新增 v36 synthetic non-semantic patch mask override，
直接覆盖 path-specific role streams：
    R_frame_tok
    R_global_tok
    R_swa_tok
    R_ttt_tok
```

修改文件：

```text
run_pipeline_abc_v2.py
loger/pipeline/hybrid_memory_controller.py
tools/run_attention_cue_experiment.sh
```

新增 CLI / env：

```text
--semantic_role_policy v36_synthetic_override
--v36_synthetic_mask
--v36_synthetic_path
--v36_synthetic_action

V36_SYNTHETIC_MASK
V36_SYNTHETIC_PATH
V36_SYNTHETIC_ACTION
```

支持 synthetic masks：

```text
all_patch_skip
center_box_skip
random_20pct_skip
left_half_skip
all_dynamic_role
all_static_role
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    loger/pipeline/hybrid_memory_controller.py

bash -n tools/run_attention_cue_experiment.sh
```

结果：

```text
PASS
```

---

## 3. H0 R1 首个 smoke 与 blocker

首个 smoke 命令：

```bash
ATTN_CUE_BASE=results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/action_realism/rollouts \
START_FRAME=290 END_FRAME=322 GLOBAL_CHUNK_OFFSET=10 \
STAGE_C_MODE=reference \
STAGE_C_CACHE_DIR=results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full \
STAGE_C_CACHE_MODE=read STAGE_C_CACHE_REQUIRE_HIT=1 \
SEMANTIC_ROLE_POLICY=v36_synthetic_override SEMANTIC_MEMORY_PATHS=frame \
ENABLE_CONTEXT_SOURCE_SKIP=1 \
CONTEXT_SOURCE_SKIP_IMPL=compact_kv \
CONTEXT_SOURCE_SKIP_SCOPE=frame \
CONTEXT_SOURCE_SKIP_MASK=semantic_role_negative \
CONTEXT_SOURCE_SKIP_LAYER_MODE=early \
V36_SYNTHETIC_MASK=all_patch_skip \
V36_SYNTHETIC_PATH=frame \
V36_SYNTHETIC_ACTION=source_skip \
bash tools/run_attention_cue_experiment.sh \
    0 V36_H0_R1_SYN01_ALL_FRAME_SKIP readonly
```

结果：

```text
run_status = DONE
frame_attention num_context_source_skip_applied = 6
frame_attention mean_context_source_keep_ratio = 0.9333299994
frame_attention max_context_source_skip_tokens = 8026
context_empty_source_events = 0
```

Blocker：

```text
SYN_01_all_patch_skip did not actually skip all synthetic patch sources.
Reason:
    CONTEXT_SOURCE_SKIP_MASK=semantic_role_negative still applies D_g q80
    filtering inside loger/models/pi3.py.
```

修复：

```text
run_pipeline_abc_v2.py:
    allowed context_source_skip_mask = v36_synthetic_role_negative

loger/models/pi3.py:
    v36_synthetic_role_negative uses negative role stream directly,
    without D_g quantile filtering.
```

边界：

```text
R1 is kept as blocker evidence.
It is not used as H0 pass evidence.
```

---

## 4. H0 R2 all-path smoke 与 cache blocker

R2 计划：

```text
并行 4 rows:
    V36_H0_R2_SYN01_ALL_FRAME_SKIP  -> GPU0
    V36_H0_R2_SYN01_ALL_GLOBAL_SKIP -> GPU1
    V36_H0_R2_SYN01_ALL_SWA_REMOVE  -> GPU2
    V36_H0_R2_SYN01_ALL_TTT_NEG     -> GPU3

START_FRAME=290
END_FRAME=354
GLOBAL_CHUNK_OFFSET=10
```

结果：

```text
All 4 rows ended with run_status = FAIL.
Reason:
    Required Stage C cache miss for chunk 12:
    chunk_012_000348_000354/masklet.pt
```

已落盘的前两块 hook evidence：

```text
V36_H0_R2_SYN01_ALL_FRAME_SKIP:
    frame_attention max_context_source_skip_tokens = 40128
    frame_attention num_context_empty_source_events = 0

V36_H0_R2_SYN01_ALL_GLOBAL_SKIP:
    chunk_attention max_context_source_skip_tokens = 40128
    chunk_attention num_context_empty_source_events = 0
```

处理：

```text
R2 is invalid for gate because run_status = FAIL.
Fix by rerunning with END_FRAME=351, avoiding the uncached partial chunk12.
```

---

## 5. H0 R3 clean all-path synthetic smoke

修复 R2 cache blocker 后，复用同一调度脚本并把窗口改成：

```text
START_FRAME=290
END_FRAME=351
GLOBAL_CHUNK_OFFSET=10
```

调度脚本：

```bash
/tmp/run_v36_h0_r2_allpath.sh
```

关键脚本内容：

```bash
COMMON_ENV=(
  ATTN_CUE_BASE=results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/action_realism/rollouts
  START_FRAME=290
  END_FRAME=351
  GLOBAL_CHUNK_OFFSET=10
  STAGE_C_MODE=reference
  STAGE_C_CACHE_DIR=results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full
  STAGE_C_CACHE_MODE=read
  STAGE_C_CACHE_REQUIRE_HIT=1
  SEMANTIC_ROLE_POLICY=v36_synthetic_override
  V36_SYNTHETIC_MASK=all_patch_skip
)

run_row 0 V36_H0_R3_SYN01_ALL_FRAME_SKIP  readonly frame  source_skip  frame frame
run_row 1 V36_H0_R3_SYN01_ALL_GLOBAL_SKIP readonly global source_skip  frame chunk
run_row 2 V36_H0_R3_SYN01_ALL_SWA_REMOVE  readonly swa    swa_remove   swa   both
run_row 3 V36_H0_R3_SYN01_ALL_TTT_NEG     hybrid   ttt    ttt_negative frame both stage_d
```

执行：

```bash
bash -n /tmp/run_v36_h0_r2_allpath.sh
bash /tmp/run_v36_h0_r2_allpath.sh
```

结果：

```text
V36_H0_R3_SYN01_ALL_FRAME_SKIP   DONE
V36_H0_R3_SYN01_ALL_GLOBAL_SKIP  DONE
V36_H0_R3_SYN01_ALL_SWA_REMOVE   DONE
V36_H0_R3_SYN01_ALL_TTT_NEG      DONE
```

关键 landed hook evidence：

| Row | Path evidence | Value |
|---|---|---:|
| `V36_H0_R3_SYN01_ALL_FRAME_SKIP` | `frame_attention.max_context_source_skip_tokens` | `40128` |
| `V36_H0_R3_SYN01_ALL_FRAME_SKIP` | `frame_attention.mean_context_source_keep_ratio` | `0.6666666865` |
| `V36_H0_R3_SYN01_ALL_FRAME_SKIP` | `frame_attention.num_context_empty_source_events` | `0` |
| `V36_H0_R3_SYN01_ALL_GLOBAL_SKIP` | `chunk_attention.max_context_source_skip_tokens` | `40128` |
| `V36_H0_R3_SYN01_ALL_GLOBAL_SKIP` | `chunk_attention.mean_context_source_keep_ratio` | `0.6666666865` |
| `V36_H0_R3_SYN01_ALL_GLOBAL_SKIP` | `chunk_attention.num_context_empty_source_events` | `0` |
| `V36_H0_R3_SYN01_ALL_SWA_REMOVE` | `swa_read.num_source_gate_applied` | `1` |
| `V36_H0_R3_SYN01_ALL_SWA_REMOVE` | `swa_read.mean_swa_gate` | `0.9673362374` |
| `V36_H0_R3_SYN01_ALL_SWA_REMOVE` | `swa_read.max_abs_gate_delta` | `0.1484375` |
| `V36_H0_R3_SYN01_ALL_TTT_NEG` | `memory_ttt_mean_rel_diff` | `0.0183690778` |
| `V36_H0_R3_SYN01_ALL_TTT_NEG` | `memory_ttt_max_rel_diff` | `0.0520210761` |

边界：

```text
R3 proves the synthetic role reaches frame/global source skip and SWA source
gate hooks without empty-source events.

R3 does not yet provide attention_mass_removed_before/after.
R3 TTT row also does not contain post-zp/action-delta tensor artifacts.
```

---

## 6. H0 R4 TTT trace repair

触发原因：

```text
H0 requires TTT update/post-zp delta norm evidence.
R3 TTT row landed memory state diff, but not post-zp/action-delta tensor trace.
```

补跑单行 trace：

```bash
cat > /tmp/run_v36_h0_r4_ttt_trace.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR
cd "$ROOT"
BASE=results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/action_realism/rollouts
TRACE=results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/action_realism/ttt_trace_R4/V36_H0_R4_SYN01_ALL_TTT_NEG_TRACE
mkdir -p results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/matrix_logs/action_realism_R4 "$TRACE"
export ATTN_CUE_BASE="$BASE"
export START_FRAME=290
export END_FRAME=351
export GLOBAL_CHUNK_OFFSET=10
export STAGE_C_MODE=reference
export STAGE_C_CACHE_DIR=results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full
export STAGE_C_CACHE_MODE=read
export STAGE_C_CACHE_REQUIRE_HIT=1
export SEMANTIC_ROLE_POLICY=v36_synthetic_override
export V36_SYNTHETIC_MASK=all_patch_skip
export SEMANTIC_MEMORY_PATHS=ttt
export V36_SYNTHETIC_PATH=ttt
export V36_SYNTHETIC_ACTION=ttt_negative
export READ_PATH=frame
export ENABLE_CONTEXT_SOURCE_SKIP=1
export CONTEXT_SOURCE_SKIP_IMPL=compact_kv
export CONTEXT_SOURCE_SKIP_SCOPE=both
export CONTEXT_SOURCE_SKIP_MASK=v36_synthetic_role_negative
export CONTEXT_SOURCE_SKIP_LAYER_MODE=early
export V11_PROJECTION_TRACE_DIR="$TRACE"
export V11_PROJECTION_ACTION_MODE=none
export V18_TRUE_ACTION_TRACE_LAYERS=0,6,12,17
export V18_TRUE_ACTION_TRACE_BRANCHES=all
bash tools/run_attention_cue_experiment.sh 3 V36_H0_R4_SYN01_ALL_TTT_NEG_TRACE hybrid dyn 1.0 stage_d \
  > results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/matrix_logs/action_realism_R4/V36_H0_R4_SYN01_ALL_TTT_NEG_TRACE.launcher.log 2>&1
EOF

bash -n /tmp/run_v36_h0_r4_ttt_trace.sh
bash /tmp/run_v36_h0_r4_ttt_trace.sh
```

结果：

```text
V36_H0_R4_SYN01_ALL_TTT_NEG_TRACE DONE
```

Trace artifacts：

```text
action_realism/ttt_trace_R4/V36_H0_R4_SYN01_ALL_TTT_NEG_TRACE/
    basis_projection_coefficients.csv
    post_zp_delta_before_after.pt
    per_layer_branch_post_zp_delta.pt
    per_token_to_post_zp_contribution_summary.pt
    basis_vector_bank.pt
    v11_trace_summary.jsonl
```

关键 trace summary：

```text
v18_true_tensor_basis = true
v18_artifact_layers = 4
v18_artifact_coeff_rows = 12 per chunk, 24 total
post_zp_action_delta_over_native_max = 0.6306188026
post_zp_action_delta_over_native_mean = 0.2624141406
```

边界：

```text
V11_PROJECTION_ACTION_MODE=none, so no GT projection action was applied.
The trace is logging-only and records online probe_ttt_write tensor deltas.
```

---

## 7. H0 action-realism report

新增报告器：

```text
tools/v36_action_realism_report.py
```

修复记录：

```text
1. Initial report parsed full run_status.txt as status, so DONE rows were
   counted as zero.
2. Fixed _status(...) to read the final DONE/FAIL line.
3. Added landed SWA gate summaries and TTT v18/post-zp trace summaries to CSV.
```

验证：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v36_action_realism_report.py
```

生成报告：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v36_action_realism_report.py \
    --run-root results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/action_realism/rollouts \
    --out-dir results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/action_realism
```

输出：

```text
action_realism/action_tensor_summary.csv
action_realism/action_jaccard_matrix.csv
action_realism/source_attention_mass_removed.csv
action_realism/swa_cache_effect_summary.csv
action_realism/ttt_update_effect_summary.csv
action_realism/hook_audit_summary.json
action_realism/hook_audit_report.md
```

Final H0 summary：

```text
rows_found = 10
rows_done = 6
context_empty_source_events_total = 0
synthetic_source_effect_rows = 3
attention_mass_removed_instrumented = false
ttt_post_zp_delta_instrumented = true
h0_gate_pass = false
```

Blocked reason：

```text
H0 cannot pass until attention-mass-removed instrumentation is present for the
landed smoke rows.
```

Final decision：

```text
H0 Action Realism / Hook Audit = fail / blocked.
Per v36 plan, H1-H5 rollout/full online is forbidden.
No v36 ATE / Target-30 result is claimed.
```
