# ACL2 v42 实验复盘：C9 Health-Gated READ Semantic Target30

日期：2026-05-26（Asia/Singapore）  
计划文件：`docs/ACL2_v42_C9_HealthGated_READ_Semantic_Target30_Plan.md`  
执行日志：`docs/ACL2_v42_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/`

本轮原则：只记录实际落盘结果；不把 health detector、mechanism audit、short/proxy evidence、blocked downstream stage 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；用户确认 GPU 0,1,2,3,4,5,6,7 可用。

---

## 0. 当前状态

```text
v42 已按计划完成 Phase 0-5 与 final report 汇总。

已完成：
    1. 阅读 v42 计划。
    2. 建立主结果目录。
    3. 复用 v36B H9/C9 parent snapshots symlink。
    4. 新增 health-selected semantic action chunk gate。
    5. py_compile / bash -n 验证通过。
    6. F0 R1 C9 full repeat 完成，但 Phase 0 no-op gate 失败：
           F0 R1 ATE = 34.2705997342m
           historical C9_P0_R2 ATE = 33.7629421029m
           delta = +0.5076576313m
    7. Phase 1 R1 detector 生成诊断结果：
           selected_bad_chunks = [7,9,12,14,16,17,19]
           selected_bad_chunk_ratio = 0.1842105263
           selection_uses_ATE = false
           selection_uses_fixed_chunk_or_segment = false
       但由于 F0 R1 no-op failed，该 detector 只作为诊断，等待 F0 R2。
    8. 已停止并隔离两批 invalid partial downstream rows：
           provisional chunk10 rows
           health-selected rows launched before Phase 0 repair
    9. 已修复 v42 launcher 的 C9 baseline drift。
    10. F0 R2 C9 full repeat 完成并通过 Phase 0 no-op gate：
            ATE = 33.76294210291885m
            historical C9_P0_R2 = 33.7629421029m
            abs_delta = 0.00000000001885m
    11. Phase 1 R2 detector 通过：
            selected_bad_chunks = [7,9,12,14,16,17,19]
            selected_bad_chunk_ratio = 0.1842105263
            selection_uses_ATE = false
            selection_uses_fixed_chunk_or_segment = false
    12. Phase 2 R2 mechanism report 完成：
            mechanism_decision = B_general_high_influence_anomaly_preferred
            explainability_level =
                incomplete_explainability_selected_chunk_not_chunk010_proxy
    13. Phase 3 official F1-F5 R2 full-online 已完成：
            rows = 6/6 including F0
            best READ candidate = F1
            best READ ATE = 34.7539112804m
            delta vs F0/C9 = +0.9909691774m
            target30_success = false
    14. Phase 4 memory barrier 未启动：
            phase4_allowed = false
            reason = READ-only 没有 full ATE 收益
    15. final report 已生成。

尚未完成：
    None
```

Phase 0 repaired short smoke R3：

```text
V42_P0_HEALTH_LOGGING_ONLY_H3_R3:
    DONE
    hmc_rows = 5
    pred_files = 1

V42_P0_READ_HOOK_NOOP_H3_R3:
    DONE
    hmc_rows = 5
    pred_files = 1
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

---

## 1. 工程修改

Initial setup：

```text
Created result root:
    results/kitti01_hmc_v2/acl2_v42_c9_healthgated_read_semantic_target30/

Reused v36B H9/C9 parent state + merge snapshots through symlink:
    phase0_parent_snapshots ->
    ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots
```

新增 / 修改：

```text
loger/pipeline/hybrid_memory_controller.py:
    added semantic_action_active_chunks parser and runtime gate.
    The gate disables semantic_role_policy / semantic_memory_paths /
    context_source_skip outside the selected health chunks.
    added semantic_action_inactive_read_cue_source so candidate READ cues can
    fall back to the C9 read cue outside selected chunks.

run_pipeline_abc_v2.py:
    added --semantic_action_active_chunks and forwards it to HMC.

tools/run_attention_cue_experiment.sh:
    forwards SEMANTIC_ACTION_ACTIVE_CHUNKS to run_pipeline_abc_v2.py.

tools/run_v42_full_candidate.sh:
    added F0-F5 full-online launcher with C9 locked config defaults and
    per-row single-GPU binding through run_attention_cue_experiment.sh.
```

Boundary：

```text
The new gate does not change C9 read cue, native C9 TTT write, or C9 SWA
overlap source replacement outside selected chunks. It only limits additional
semantic READ source filtering actions to health-selected chunks.

After the Phase 0 failure audit, this boundary was strengthened:
    outside semantic_action_active_chunks, HMC now also receives prior_output =
    None, so Stage-C semantic prior cannot perturb non-selected chunks.
```

Validation：

```text
py_compile:
    loger/pipeline/hybrid_memory_controller.py
    run_pipeline_abc_v2.py
    PASS

bash -n:
    tools/run_attention_cue_experiment.sh
    tools/run_v42_full_candidate.sh
    PASS
```

### Phase 0 Blocker：C9 no-op repeat drift

F0 R1 output：

```text
run = V42_P0_F0_C9_REFERENCE_REPEAT
status = DONE
hmc_rows = 38
ATE = 34.2705997342m
historical C9_P0_R2 ATE = 33.7629421029m
delta_vs_historical_C9 = +0.5076576313m
Phase 0 no-op gate = fail
```

原因审计：

```text
v42 F0 R1 was not exact C9:
    beta_frame / beta_swa used 4.25 instead of historical 4.75.
    Stage C was enabled in reference/read mode instead of off.
    TTT commit EMA settings drifted:
        alpha 1.0 vs historical 0.5
        branch_mask all vs historical 0
        chunks empty vs historical 5,6
    TTT native mix scales were missing:
        historical = 1.10,1.00,1.00
```

修复：

```text
tools/run_v42_full_candidate.sh:
    restored C9 locked no-op defaults.
    Stage C is disabled by default for F0 and enabled only by semantic READ
    candidates.

loger/pipeline/hybrid_memory_controller.py:
    outside semantic_action_active_chunks, candidate READ cue falls back to C9
    cue and prior_output is now None.
```

当前修复状态：

```text
F0 R2 completed:
    V42_P0_F0_C9_REFERENCE_REPEAT_R2
    status = DONE
    ATE = 33.76294210291885m
    Phase 0 no-op gate = pass
```

## 2. Phase 1：Full-Sequence Health Detector

输出：

```text
phase1_health_R2/
```

Summary：

| Metric | Value |
|---|---:|
| `phase1_gate_pass` | `true` |
| `health_chunk_count` | `38` |
| `selected_bad_chunks` | `[7,9,12,14,16,17,19]` |
| `selected_bad_chunk_ratio` | `0.1842105263` |
| `selection_uses_ATE` | `false` |
| `selection_uses_fixed_chunk_or_segment` | `false` |
| `rolling100_used_for_selection` | `false` |
| `top_rolling100_bad_chunk_diagnostic` | `6` |

Decision：

```text
Phase 1 gate = pass.
Phase 3 may run health-gated READ candidates only on selected chunks.
```

## 3. Phase 2：READ Mechanism Audit

输出：

```text
phase2_read_mechanism_R2/
```

Summary：

| Metric | Value |
|---|---:|
| `mechanism_decision` | `B_general_high_influence_anomaly_preferred` |
| `explainability_level` | `incomplete_explainability_selected_chunk_not_chunk010_proxy` |
| `general_anomaly_supported` | `true` |
| `sky_causality_supported` | `false` |
| `static_anchor_misdamage_risk` | `not_proven_from_spatial_maps` |
| `scalar_attention_mass_rows` | `144` |
| `proxy_overlays_copied` | `3` |

Boundary：

```text
The v42 selected chunks do not include chunk10. The richest landed v41 proxy
attribution evidence was chunk10, so Phase 2 is explainability-incomplete.
No per-label spatial attention / affected-source maps are reconstructed.
```

## 4. Phase 3：Full-Online READ Candidates

状态：

```text
DONE.
```

Rows：

| Label | Run | Active chunks |
|---|---|---|
| `F1` | `V42_F1_R2_HEALTHSEL_R1_HIGH_INFLUENCE_READ` | `[7,9,12,14,16,17,19]` |
| `F2` | `V42_F2_R2_HEALTHSEL_R2_SKY_APP_READ` | `[7,9,12,14,16,17,19]` |
| `F3` | `V42_F3_R2_HEALTHSEL_R3_STATIC_RESCUE_READ` | `[7,9,12,14,16,17,19]` |
| `F4` | `V42_F4_R2_HEALTHSEL_R3_EPISODE_FOLLOW_READ` | `[7,8,9,10,12,13,14,15,16,17,18,19,20]` |
| `F5` | `V42_F5_R2_HEALTHSEL_H9_R1_DIAG` | `[7,9,12,14,16,17,19]` |

Boundary：

```text
F4 uses a static one-step follow-through approximation:
selected chunk plus next chunk. It is not a learned persistence detector.
```

Runtime action audit while rows are running：

```text
F1/F3/F5:
    selected chunk7/chunk9 show source_skip_applied > 0.
    non-selected chunk8 falls back to C9 cue with source_skip_applied = 0.

F4:
    chunk7 and chunk8 both show action, matching selected+next follow-through.

context_empty_source_events observed in audited chunks = 0.
```

Full-online metrics：

| Label | Run | ATE | Delta vs F0 | `[200,300)` | `[400,600)` | Action chunks |
|---|---|---:|---:|---:|---:|---|
| `F0` | `V42_P0_F0_C9_REFERENCE_REPEAT_R2` | `33.7629421029` | `0.0000000000` | `76.1021355543` | `41.8963642126` | `[]` |
| `F1` | `V42_F1_R2_HEALTHSEL_R1_HIGH_INFLUENCE_READ` | `34.7539112804` | `+0.9909691774` | `76.3653263460` | `43.6064222084` | `[7,9,12,14,16,17,19]` |
| `F2` | `V42_F2_R2_HEALTHSEL_R2_SKY_APP_READ` | `39.3352485068` | `+5.5723064039` | `84.3972404066` | `49.6447573571` | `[7,9,12,14,16,17,19]` |
| `F3` | `V42_F3_R2_HEALTHSEL_R3_STATIC_RESCUE_READ` | `38.2193997350` | `+4.4564576321` | `83.0529914780` | `47.7645482227` | `[7,9,12,14,16,17,19]` |
| `F4` | `V42_F4_R2_HEALTHSEL_R3_EPISODE_FOLLOW_READ` | `48.1789570268` | `+14.4160149239` | `83.0021804111` | `62.2024339052` | `[7,8,9,10,12,13,14,15,16,17,18,19,20]` |
| `F5` | `V42_F5_R2_HEALTHSEL_H9_R1_DIAG` | `34.7524824725` | `+0.9895403696` | `76.4623800233` | `43.5591122658` | `[7,9,12,14,16,17,19]` |

Full-online decision：

```text
minimum_progress_pass = false
stage_success_pass = false
strong_success_pass = false
target30_success = false
phase4_allowed = false

The best READ-only row is F1, but it regresses C9 by +0.9909691774m.
All F1-F4 READ-only C9 candidates are worse than F0/C9.
F4 one-step follow-through is especially harmful:
    ATE delta vs F0 = +14.4160149239m
    [400,600) delta vs F0 = +20.3060696926m
```

Interpretation：

```text
v42 validates the engineering path:
    exact C9 no-op reproduction was repaired,
    health detector selected chunks without ATE/fixed-segment gating,
    semantic READ actions were correctly gated to selected chunks,
    and full-online ATE was produced for the minimal F0-F5 matrix.

But the scientific result is negative:
    health-gated semantic READ filtering on the selected full-sequence chunks
    does not improve C9.

The v41 chunk10 local READ diagnostic does not transfer to the v42
full-sequence health-selected chunks. The selected chunks [7,9,12,14,16,17,19]
produce global regressions under F1/F2/F3/F4.
```

## 5. Phase 4：Memory Barrier

Decision：

```text
Phase 4 was not launched.

Reason:
    Phase 4 requires READ-only full-online benefit but insufficient final ATE.
    v42 Phase 3 has no READ-only benefit:
        best READ ATE = 34.7539112804m
        F0/C9 ATE = 33.7629421029m
        delta = +0.9909691774m

Therefore launching memory barrier would violate the v42 plan and would be
diagnostic fishing.
```

## 6. Final Decision

Final reports：

```text
final_reports/v42_full_online_report.md
final_reports/v42_full_online_summary.json
final_reports/full_online_registry.csv
```

Final summary：

```text
phase0_noop_gate_pass = true
phase1_gate_pass = true
phase2_mechanism_decision = B_general_high_influence_anomaly_preferred
phase2_explainability_level = incomplete_explainability_selected_chunk_not_chunk010_proxy
phase3_done_rows = 6
best_read_candidate = F1
best_read_ATE_full = 34.7539112804m
best_read_delta_vs_C9 = +0.9909691774m
phase4_allowed = false
full_online_launched = true
target30_success = false
deployable_online_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029m
```

Downstream decision：

| Stage | Status | Reason |
|---|---|---|
| Phase 0 C9 no-op | pass after repair | F0 R2 exactly reproduces C9 ATE |
| Phase 1 health detector | pass | selected chunk ratio `0.1842`; no ATE/fixed chunk selection |
| Phase 2 mechanism audit | incomplete but sufficient to continue | scalar/proxy evidence only; no selected-chunk spatial maps |
| Phase 3 READ-only full online | fail | best READ row F1 regresses C9 by `+0.9910m` |
| Phase 4 memory barrier | not launched | no READ-only benefit |
| Target-30 | fail | no row reaches `<=30m` |
| Deployable online success | no | C9 remains best deployable |

Conclusion：

```text
v42 is an engineering success but a deployment failure.

It fixed the C9 baseline reproduction problem and proved that health-selected
READ actions can be gated cleanly in full online. However, the actual
health-gated READ candidates all regress C9.

Do not promote v42 as deployable online success.
```

Next required direction：

```text
Do not launch v42 Phase 4 from current rows.
Do not continue this exact health-selected READ-only policy.

The closest useful lesson is negative:
    v41's chunk10 local signal is not enough to justify a full-sequence health
    detector that selects [7,9,12,14,16,17,19].

If continuing, first fix the health detector/causality evidence:
    land per-selected-chunk spatial attention / affected-source maps,
    validate selected chunks against source-contamination evidence rather than
    relying on aggregate health,
    or return to C9-native trajectory/scale/gauge correction.
```

### Invalidated Rows

The following partial rows are not counted:

```text
phase3_full_online/provisional_invalid_chunk10_mismatch/
    V42_F1_C9_R1_HEALTHGATED_READ
    V42_F2_C9_R2_SKY_APP_READ
    V42_F3_C9_R3_STATIC_RESCUE_READ
    V42_F4_C9_R3_EPISODE_FOLLOW_READ
    V42_F5_H9_R1_DIAG

phase3_full_online/invalid_phase0_noop_failed/
    V42_F1_HEALTHSEL_R1_HIGH_INFLUENCE_READ
    V42_F2_HEALTHSEL_R2_SKY_APP_READ
    V42_F3_HEALTHSEL_R3_STATIC_RESCUE_READ
    V42_F4_HEALTHSEL_R3_EPISODE_FOLLOW_READ
    V42_F5_HEALTHSEL_H9_R1_DIAG
```

Reason：

```text
The first batch used v41 chunk10 before v42 Phase 1 selected chunks.
The second batch was launched before the Phase 0 no-op drift was discovered.
Neither batch reached DONE, and neither is valid v42 evidence.
```
