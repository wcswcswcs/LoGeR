# ACL2 v36 实验复盘：Training-Free Semantic Memory Control Target30

日期：2026-05-24（Asia/Singapore）  
计划文件：`docs/ACL2_v36_TrainingFree_SemanticMemory_Control_Target30_Plan.md`  
执行日志：`docs/ACL2_v36_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/`

本轮原则：只记录实际落盘结果；不把 action audit、hook smoke、short rollout、blocked gate 或未启动 full online 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立 row 分配到不同 GPU。

---

## 0. 当前结论

v36 已按计划先执行 H0 Action Realism / Hook Audit，并在遇到 blocker 后按计划尝试修复：

```text
1. 修复 synthetic all_patch_skip 被 D_g quantile 过滤的问题。
2. 修复 R2 uncached partial chunk12 窗口问题。
3. 修复 H0 report 对 run_status.txt 的 DONE/FAIL 解析问题。
4. 补跑 TTT trace row，补齐 post-zp/action-delta tensor evidence。
```

但 H0 最终没有通过：

```text
attention_mass_removed_before/after 仍没有可审计落盘 instrumentation。
```

因此按 v36 hard gate：

```text
H1-H5 rollout / selector / full online validation 全部不允许启动。
```

v36 未达到 Target-30，也没有产生新的 deployable online result。

已完成并落盘：

```text
1. 阅读 v36 计划，确认 hard gate：
       H0 Action Realism / Hook Audit 必须先过。
       H0 不过时禁止 H1-H5 rollout/full online。

2. 新增 v36 synthetic override 接线：
       run_pipeline_abc_v2.py
       loger/pipeline/hybrid_memory_controller.py
       tools/run_attention_cue_experiment.sh

3. 新增 v36 H0 action realism report：
       tools/v36_action_realism_report.py

4. 编译 / shell 语法验证通过：
       run_pipeline_abc_v2.py
       loger/models/pi3.py
       loger/pipeline/hybrid_memory_controller.py
       tools/v36_action_realism_report.py
       tools/run_attention_cue_experiment.sh

5. H0 R3 clean all-path synthetic smoke 完成：
       frame source skip DONE
       global/chunk source skip DONE
       SWA remove DONE
       TTT negative DONE

6. H0 R4 TTT trace repair 完成：
       v18_true_tensor_basis = true
       post_zp_action_delta_over_native_max = 0.6306188026
       post_zp_action_delta_over_native_mean = 0.2624141406

7. H0 final report：
       rows_found = 10
       rows_done = 6
       context_empty_source_events_total = 0
       synthetic_source_effect_rows = 3
       attention_mass_removed_instrumented = false
       ttt_post_zp_delta_instrumented = true
       h0_gate_pass = false
```

最终边界：

```text
1. Frame/global synthetic source skip hook is real enough to change source skip summaries.
2. SWA synthetic remove reaches SWA source gate summary.
3. TTT synthetic negative has true post-zp/action-delta tensor trace.
4. Attention mass removed before/after is not instrumented.
5. H0 gate = fail / blocked.
6. No H1/H2/H3/H4/H5 rollout was launched.
7. No full online validation was launched.
8. No online Target-30 result was produced in v36.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

---

## 1. 工程修改

新增 / 修改：

```text
run_pipeline_abc_v2.py:
    added v36_synthetic_override semantic role policy.
    added CLI:
        --v36_synthetic_mask
        --v36_synthetic_path
        --v36_synthetic_action
    added _apply_v36_synthetic_intervention(...).
    It writes path-specific role tensors only and records:
        v36_synthetic_intervention
    in runtime debug.

loger/pipeline/hybrid_memory_controller.py:
    v36_synthetic_override is consumed as a path-role override policy,
    same mechanism class as v29c_masklet_override, but without masklet/GT data.

loger/models/pi3.py:
    added v36_synthetic_role_negative context source skip mask.
    It skips negative role stream tokens directly, without D_g quantile
    filtering, for H0 synthetic stress only.

tools/run_attention_cue_experiment.sh:
    forwards V36_SYNTHETIC_MASK / PATH / ACTION to run_pipeline_abc_v2.py.

tools/v36_action_realism_report.py:
    aggregates H0 synthetic smoke rows.
    Writes action tensor summary, Jaccard, source skip summaries, SWA summaries,
    TTT trace summaries, and H0 gate report.
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    loger/models/pi3.py \
    loger/pipeline/hybrid_memory_controller.py \
    tools/v36_action_realism_report.py

bash -n tools/run_attention_cue_experiment.sh

PASS
```

边界：

```text
This is an H0 audit hook, not a trajectory result.
No v36 result currently counts as deployable online success.
```

---

## 2. Blocker 与修复记录

### Blocker 1：H0 synthetic all_patch_skip 被 D_g quantile 过滤

现象：

```text
V36_H0_R1_SYN01_ALL_FRAME_SKIP:
    run_status = DONE
    requested synthetic mask = all_patch_skip
    requested path = frame
    requested action = source_skip

Observed hook effect:
    frame_attention num_context_source_skip_applied = 6
    frame_attention mean_context_source_keep_ratio = 0.9333299994
    frame_attention max_context_source_skip_tokens = 8026
    context_empty_source_events = 0
```

原因：

```text
The old CONTEXT_SOURCE_SKIP_MASK=semantic_role_negative still applied D_g q80
inside loger/models/pi3.py, so the synthetic all-patch negative role was not
used as a pure all-source stress test.
```

修复：

```text
run_pipeline_abc_v2.py:
    added context_source_skip_mask choice:
        v36_synthetic_role_negative

loger/models/pi3.py:
    v36_synthetic_role_negative skips negative role tokens directly and does
    not apply the D_g quantile filter.
```

边界：

```text
R1 is blocker evidence only.
It is not counted as H0 pass evidence.
```

### Blocker 2：H0 R2 窗口触发 uncached partial chunk12

现象：

```text
R2 all-path smoke used:
    START_FRAME=290
    END_FRAME=354
    GLOBAL_CHUNK_OFFSET=10

All four rows failed with:
    Required Stage C cache miss for chunk 12:
    chunk_012_000348_000354/masklet.pt
```

原因：

```text
The selected END_FRAME created a partial chunk12 [348,354).
The reference Stage-C cache contains full planned chunks, not this partial
temporary chunk.
```

修复方向：

```text
Rerun the same H0 all-path smoke with END_FRAME=351 so only full chunk10/11
are evaluated from the existing cache.
```

边界：

```text
R2 is invalid for H0 gate because run_status = FAIL.
Partial context summaries are kept only as blocker evidence.
```

### Blocker 3：H0 report initially counted DONE rows as zero

现象：

```text
hook_audit_summary.json initially showed:
    rows_found = 9
    rows_done = 0

But run_status.txt files contained final DONE lines.
```

原因：

```text
tools/v36_action_realism_report.py read the entire run_status.txt text as the
status string instead of extracting the final DONE/FAIL line.
```

修复：

```text
tools/v36_action_realism_report.py:
    _status(...) now scans run_status.txt from the end and returns DONE/FAIL.
```

结果：

```text
Final report:
    rows_found = 10
    rows_done = 6
```

### Blocker 4：TTT post-zp/action-delta evidence missing in R3

现象：

```text
V36_H0_R3_SYN01_ALL_TTT_NEG:
    memory_ttt_mean_rel_diff = 0.0183690778
    memory_ttt_max_rel_diff = 0.0520210761

But no post_zp/action-delta tensor artifact was available.
```

修复：

```text
Reran one TTT trace row:
    V36_H0_R4_SYN01_ALL_TTT_NEG_TRACE

with:
    V11_PROJECTION_TRACE_DIR=...
    V11_PROJECTION_ACTION_MODE=none
    V18_TRUE_ACTION_TRACE_LAYERS=0,6,12,17
    V18_TRUE_ACTION_TRACE_BRANCHES=all
```

结果：

```text
V36_H0_R4_SYN01_ALL_TTT_NEG_TRACE = DONE

Trace artifacts:
    basis_projection_coefficients.csv
    post_zp_delta_before_after.pt
    per_layer_branch_post_zp_delta.pt
    per_token_to_post_zp_contribution_summary.pt

Metrics:
    v18_true_tensor_basis = true
    v18_artifact_layers = 4
    v18_artifact_coeff_rows = 24 total
    post_zp_action_delta_over_native_max = 0.6306188026
    post_zp_action_delta_over_native_mean = 0.2624141406
```

边界：

```text
V11_PROJECTION_ACTION_MODE=none.
No GT projection action was used.
The trace is logging-only.
```

### Blocker 5：attention_mass_removed_before/after 未落盘

现象：

```text
H0 final report:
    attention_mass_removed_instrumented = false
    h0_gate_pass = false
```

原因：

```text
Current frame/global context source skip logs record:
    max_context_source_skip_tokens
    mean_context_source_keep_ratio
    num_context_empty_source_events

They do not record the actual pre-skip attention mass on skipped source tokens.
Without that, v36 H0 condition:
    skipped source original attention mass >= 0.05
cannot be honestly evaluated.
```

已尝试修复：

```text
1. Fixed synthetic mask hook so all_patch_skip reaches source skip.
2. Confirmed frame/global hooks skip 40128 source tokens with no empty-source.
3. Confirmed TTT trace can be instrumented with true post-zp/action tensors.
```

未能合法修复：

```text
No landed runtime artifact provides attention_mass_removed_before/after.
The report explicitly refuses to impute or approximate this metric.
```

Decision：

```text
H0 remains blocked.
Per v36 plan, no rollout/full-online continuation is allowed.
```

---

## 3. H0 Action Realism / Hook Audit

输出：

```text
results/kitti01_hmc_v2/acl2_v36_trainingfree_semanticmemory_control_target30/action_realism/
    action_tensor_summary.csv
    action_jaccard_matrix.csv
    source_attention_mass_removed.csv
    swa_cache_effect_summary.csv
    ttt_update_effect_summary.csv
    hook_audit_summary.json
    hook_audit_report.md
```

Clean R3 rows：

| Row | Status | Key landed evidence |
|---|---|---|
| `V36_H0_R3_SYN01_ALL_FRAME_SKIP` | DONE | `frame_attention.max_context_source_skip_tokens = 40128`; empty source `0` |
| `V36_H0_R3_SYN01_ALL_GLOBAL_SKIP` | DONE | `chunk_attention.max_context_source_skip_tokens = 40128`; empty source `0` |
| `V36_H0_R3_SYN01_ALL_SWA_REMOVE` | DONE | `swa_read.num_source_gate_applied = 1`; `mean_swa_gate = 0.9673362374` |
| `V36_H0_R3_SYN01_ALL_TTT_NEG` | DONE | `memory_ttt_mean_rel_diff = 0.0183690778`; `max = 0.0520210761` |
| `V36_H0_R4_SYN01_ALL_TTT_NEG_TRACE` | DONE | `post_zp_action_delta_over_native_max = 0.6306188026` |

Hook audit summary：

| Metric | Value |
|---|---:|
| `rows_found` | `10` |
| `rows_done` | `6` |
| `context_empty_source_events_total` | `0` |
| `synthetic_source_effect_rows` | `3` |
| `attention_mass_removed_instrumented` | `false` |
| `ttt_post_zp_delta_instrumented` | `true` |
| `h0_gate_pass` | `false` |

Decision：

```text
H0 Action Realism / Hook Audit = fail / blocked.

Reason:
    attention_mass_removed_before/after is required by the v36 plan and is not
    available in landed runtime logs.

No unavailable metric was imputed.
```

---

## 4. Not Started / Not Claimed

Not started because H0 failed:

```text
1. H1 VGGT4D-style group-level source skip rollout.
2. H2 SWA semantic local-continuity control.
3. H3 TTT semantic write lifecycle control.
4. H4 semantic-conditioned C23 residual/read-only C9 compatibility.
5. H5 deployable online validation.
6. Selector / learned trigger / learned router.
```

Boundary：

```text
No v36 row counts as deployable online TTT write success.
No v36 short rollout or full online ATE exists.
No online Target-30 result was produced.
```

---

## 5. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Pre-flight v35B cleanup | pass | leftover v35B processes killed |
| v36 synthetic override | pass | path role tensors created without GT/video masklet dependency |
| H0 R1 | blocker evidence | all_patch_skip was D_g-filtered |
| v36 synthetic source skip repair | done | `v36_synthetic_role_negative` bypasses D_g filter |
| H0 R2 | invalid | uncached partial chunk12 |
| H0 R3 | partial hook pass | frame/global/SWA/TTT hooks landed, empty-source `0` |
| H0 R4 TTT trace | pass for TTT instrumentation | post-zp/action tensor artifacts landed |
| H0 final gate | fail / blocked | attention mass removed not instrumented |
| H1-H5 | not started | H0 failed |
| Target-30 | not produced | no full online allowed |

---

## 6. Final Decision

v36 的真实成功点：

```text
1. Ended previous v35B leftover processes before starting v36.
2. Added an auditable v36 synthetic action override independent of video-masklet
   predictions and GT trajectory.
3. Found and fixed the D_g filtering bug that made synthetic all_patch_skip not
   a pure all-source stress test.
4. Proved frame/global synthetic source skip can reach real context-source skip
   hooks:
       max_context_source_skip_tokens = 40128
       context_empty_source_events = 0
5. Proved SWA synthetic remove reaches SWA source-gate summary:
       num_source_gate_applied = 1
       mean_swa_gate = 0.9673362374
6. Proved TTT trace instrumentation can land true post-zp/action tensor deltas:
       post_zp_action_delta_over_native_max = 0.6306188026
```

v36 的关键负结果：

```text
H0 still cannot pass because attention_mass_removed_before/after is missing.

The landed source skip summaries prove tokens were skipped, but do not prove
the skipped source originally carried attention mass >= 0.05.

Therefore the plan's H0 action-realism gate remains blocked.
```

Interpretation：

```text
The previous semantic negative results cannot yet be cleanly reinterpreted as
semantic hypothesis failure or success.

v36 did repair part of the hook/action path, but the H0 audit is intentionally
stricter than "tokens changed":
    it requires proof that the skipped source mattered to attention.

That proof is not available from current logs.
```

Conclusion type：

```text
H0 hook/action partial success, H0 action-realism gate failure.

Do not promote v36 to rollout/full online.
Do not claim Target-30.
```

Next required direction：

```text
Before any v36 H1-H5 continuation, add a real attention-mass instrumentation
path for frame/global source skip that records:
    attention_mass_removed_before
    attention_mass_removed_after
    structure/dynamic/semantic-group attention mass before/after

Only after H0 passes with those metrics should v36 proceed to H1 source-skip
trajectory rollouts.
```
