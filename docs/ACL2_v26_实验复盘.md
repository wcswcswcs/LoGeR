# ACL2 v26 实验复盘：VideoMasklet SemanticRoleRouter AllMemory Parallel

日期：2026-05-22（Asia/Singapore）  
计划文件：`docs/ACL2_v26_VideoMasklet_SemanticRoleRouter_AllMemory_Parallel_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v26_videomasklet_semanticrole_router_allmemory_parallel/`

本轮原则：只记录实际落盘结果；不把 implementation audit、passive attribution、smoke、short rollout、failed gate、或未启动矩阵写成 deployable online success。没有通过 Phase 2 gate 时，不启动 Phase 3 pairwise、Phase 4 all-memory、no-GT selector 或 full online validation。

---

## 0. 当前结论

v26 已按 speed-gated Phase 2 策略执行到 stop rule，未达到 Target-25。

已完成并落盘：

```text
1. 阅读 v26 计划，并实现 fine-label path-specific semantic role router。
2. Phase 0 H0 implementation hard gate 通过。
3. Phase 0 no-op / pass-through / debug-only direct pose parity 通过：
       matched rows = 119
       max_translation_abs_diff = 0
       max_pose_abs_diff = 0
4. R_frame_tok / R_global_tok / R_swa_tok / R_ttt_tok 四条 role stream 均非空。
5. frame / global / SWA / TTT consumption flags 均为 true。
6. Phase 1 passive fine-label attribution 通过解释力 gate。
7. Phase 2 chunk10 h10 screen 完成：
       rows = 15/15
       failures = 0
8. Phase 2 h15 top confirmation 完成：
       rows = 5/5
       failures = 0
9. 按失败分流表追加推荐修复尝试：
       TTT fine semantic + update_conflict_energy / scale-state risk diagnostic
       h10 risk rows = 3/3
       h15 risk top row = 1/1
10. 按 h10 strong / h15 weak 分流补做 Phase 5 washout attribution：
       evidence_level = proxy_only_no_tensor_state_snapshots
       common HMC hash matches = 11/11
       h15 tail chunks = 5
11. 按 Phase 5 分流尝试 skip-aware TTT commit / W_long-W_short repair：
       TTT_FINE_REPAIR_01_SCALE_DUAL_LIFETIME h10 row = 1/1
       result = weaker than original scale-state risk row
12. 继续尝试更选择性的 scale-state + conflict-energy commit-filter repair：
       TTT_FINE_REPAIR_02_SCALE_CONFLICT_COMMIT_FILTER h10 row = 1/1
       result = weaker than original scale-state risk row
13. Phase 2 gate = fail。
```

最终边界：

```text
1. original h10 screen best ATE delta = -0.2713026491m。
2. original h10 screen best [200,300) delta = -1.2687051048m。
3. risk-coupled h10 best ATE delta = -1.9228654883m。
4. risk-coupled h10 best [200,300) delta = -1.8379048956m。
5. original h15 top best ATE delta = -0.1331930081m。
6. original h15 top best [200,300) delta = -1.0294599587m。
7. risk-coupled h15 top best ATE delta = -1.0253418796m。
8. risk-coupled h15 top best [200,300) delta = -1.6843312369m。
9. risk ATE durability abs(h15/h10) = 0.5332364019。
10. risk [200,300) durability abs(h15/h10) = 0.9164409061。
11. Phase 5 proxy attribution points to TTT/scale-state tail update as plausible washout path.
12. W_long/W_short repair h10 ATE delta = -0.4875482747m。
13. W_long/W_short repair h10 [200,300) delta = -0.4996539742m。
14. W_long/W_short repair did not qualify for h15.
15. scale-state + conflict commit-filter repair h10 ATE delta = -0.1640035255m。
16. scale-state + conflict commit-filter repair h10 [200,300) delta = -0.8490782836m。
17. scale-state + conflict commit-filter repair did not qualify for h15.
18. No Phase 2 candidate passed gate.
19. No Phase 3 pairwise combination was started.
20. No Phase 4 all-memory validation was started.
21. No no-GT selector was started.
22. No full online validation was launched.
23. No online Target-25 result was produced.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v26 不是 GT semantic 实验。
semantic_source = video_masklet_frontend_cache
uses_gt_semantic = false
uses_video_masklet_semantic = true

v26 已经把 fine label 接入 runtime role policy，
但当前实现只有 D_g / Q_mask 条件化。
update_conflict_energy 和 scale-state risk 尚不是 token-aligned runtime input，
因此不能声称完整验证了 fine label + conflict + scale-risk router。
本轮后续补测使用已有 TTT-native update_conflict_energy / v19 scale-state
作为 write-side diagnostic coupling；这是推荐修复方向的 diagnostic trial，
不是新的 token-aligned semantic condition implementation。
```

---

## 1. 工程修改

新增 / 修改：

```text
loger/pipeline/semantic_prior_generator.py:
    added stable fine-label ids and name mapping:
        road, sidewalk, building, wall, fence, sky, vegetation, grass, ...
    fixed L_sem_tok:
        previously token label stream could represent masklet index style ids;
        now it carries stable fine-label ids from MaskletOutput.L_sem.
    PriorOutput added:
        R_frame_tok
        R_global_tok
        R_swa_tok
        R_ttt_tok
        and patch-flat variants
    project_masklet_semantic_groups now emits:
        L_sem_tok
        G_sem_tok
        R_frame/R_global/R_swa/R_ttt role streams

loger/pipeline/hybrid_memory_controller.py:
    HybridMemoryControlPrior added:
        L_sem_tok
        R_frame_tok
        R_global_tok
        R_swa_tok
        R_ttt_tok
    added fine-label path router policies:
        fine_path_router
        fine_fg_*
        fine_swa_*
        fine_ttt_*
    added _fine_path_roles:
        fine label + D_g + Q_mask conditioned roles
    added per-label runtime audit fields:
        fine_label_condition_metrics
        fine_label_path_role_counts
    records unavailable condition signals:
        condition_signal_conflict_available = false
        condition_signal_scale_risk_available = false

loger/models/pi3.py:
    frame source skip consumes R_frame_tok when available.
    global/chunk source skip consumes R_global_tok when available.
    fallback remains R_sem_tok for older policies.

run_pipeline_abc_v2.py:
    CLI accepts fine_* semantic role policies.
    pass-through prior forwards path-specific role streams.
    semantic_group_summary.jsonl records fine-label counts.
    semantic_role_summary.jsonl records path_role_counts.
    semantic_memory_path_summary.jsonl records:
        runtime_fine_role_policy_available
        path_specific_role_streams_available
        R_frame/R_global/R_swa/R_ttt counts
        fine_label_condition_metrics
        fine_label_path_role_counts

tools/run_v24_candidate_rollout.sh:
    reused as trusted rollout launcher with v26 candidate aliases:
        P0_00..P0_06
        FG_FINE_*
        SWA_FINE_*
        TTT_FINE_*
    v26 aliases route to fine_* policies, not old coarse semantic policies.
    added risk-coupled diagnostic aliases after Phase 2 weak signal:
        TTT_FINE_RISK_01_CONFLICT_TRI
        TTT_FINE_RISK_02_SCALE_STATE
        TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER
    these enable existing TTT-native risk mechanisms:
        update_conflict_energy tri replay
        v19_scale_state projection_risk
        update_conflict_energy commit filter
    added Phase 5 W_long/W_short repair alias:
        TTT_FINE_REPAIR_01_SCALE_DUAL_LIFETIME
    this combines:
        fine_ttt_scale_conditioned
        v19_scale_state projection_risk
        dual_lifetime transient delta
        long_scale = 0.25
        ttl = 3
    added Phase 5 selective commit-filter repair alias:
        TTT_FINE_REPAIR_02_SCALE_CONFLICT_COMMIT_FILTER
    this combines:
        fine_ttt_scale_conditioned
        v19_scale_state projection_risk
        update_conflict_energy tail-overlap old_decay_by_risk commit filter
        branch0 commit scale observed around 0.15-0.35

tools/v26_implementation_self_audit.py:
    strict Phase 0 H0 audit:
        cache quality
        direct same-batch no-op pose parity
        runtime fine role availability
        four role streams non-empty
        path consumption flags
        context empty source events

tools/v26_passive_fine_label_attribution.py:
    Phase 1 passive fine-label attribution:
        per-label D_g / Q metrics
        per-label path role mass
        per-label path action
        prior candidate-level segment metrics
        label condition correlation summary
        aggregate dashboard plots

tools/v24_candidate_bank_report.py:
    added v26 candidate family labels for report/gate aggregation.
    added v26 risk-coupled diagnostic candidate labels.
    added v26 W_long/W_short repair candidate label.

tools/v26_phase5_washout_attribution.py:
    Phase 5 lightweight h10/h15 washout attribution for v26 risk candidate.
    Uses landed artifacts only:
        hmc_state_hash.jsonl
        semantic_memory_path_summary.jsonl
        hook_effect_summary.jsonl
        candidate_vs_H9_delta_by_horizon.csv
    Writes:
        phase5_washout_summary.json
        h10_h15_state_norms.csv
        path_overwrite_ratio.csv
        label_role_mass_h10_h15.csv
        memory_path_washout_summary.md
    Explicitly marks:
        evidence_level = proxy_only_no_tensor_state_snapshots
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    loger/pipeline/semantic_prior_generator.py \
    loger/pipeline/hybrid_memory_controller.py \
    loger/models/pi3.py \
    run_pipeline_abc_v2.py \
    tools/v26_implementation_self_audit.py \
    tools/v26_passive_fine_label_attribution.py \
    tools/v26_phase5_washout_attribution.py \
    tools/v24_candidate_bank_report.py

bash -n \
    tools/run_v24_candidate_rollout.sh \
    tools/run_v24_phase2_initial_queue.sh \
    tools/run_attention_cue_experiment.sh

PASS
```

---

## 2. Phase 0：Implementation Hard Gate

输出：

```text
implementation_audit/
matrix_logs/phase0_smoke_R1/
rollouts/V26_P0_SMOKE_R1_*
```

Phase 0 smoke：

```text
P0_00_H9_REFERENCE
P0_01_FINE_LABEL_LOADED_BUT_IGNORED
P0_02_FINE_ROLE_PASS_THROUGH_CONSUMED
P0_03_FINE_ROLE_DEBUG_ONLY_ALL_PATHS
P0_04_FRAME_SOURCE_SMOKE
P0_05_SWA_CACHE_SMOKE
P0_06_TTT_ROLE_SMOKE

chunk = 10
horizon = 3
rows completed = 7/7
```

H0 audit summary：

| Metric | Value |
|---|---:|
| `all_gate_pass` | `true` |
| `cache_hit_rate` | `1.0` |
| `chunks_with_masklets_ratio` | `1.0` |
| `mean_coverage` | `0.9816746437` |
| `focus_coverage_200_300` | `0.9781515747` |
| `fine_label_count` | `8` |
| `runtime_fine_role_policy_available` | `true` |
| `path_specific_role_streams_available` | `true` |
| `context_empty_source_events` | `0` |

Fine labels：

```text
building
fence
grass
road
sidewalk
sky
vegetation
wall
```

Role stream gate：

| Stream | Non-empty |
|---|---|
| `R_frame` | `true` |
| `R_global` | `true` |
| `R_swa` | `true` |
| `R_ttt` | `true` |

Path consumption：

| Path | Consumed |
|---|---|
| frame | `true` |
| global | `true` |
| SWA | `true` |
| TTT | `true` |

No-op parity：

| Candidate | Reference | matched rows | max translation diff | max pose diff |
|---|---|---:|---:|---:|
| `P0_00_H9_REFERENCE` | self | `119` | `0.0` | `0.0` |
| `P0_01_FINE_LABEL_LOADED_BUT_IGNORED` | `P0_00` | `119` | `0.0` | `0.0` |
| `P0_02_FINE_ROLE_PASS_THROUGH_CONSUMED` | `P0_00` | `119` | `0.0` | `0.0` |
| `P0_03_FINE_ROLE_DEBUG_ONLY_ALL_PATHS` | `P0_00` | `119` | `0.0` | `0.0` |

Runtime：

```text
Phase 0 wall_seconds:
    count = 7
    min = 175s
    max = 201s
    mean = 182.143s
```

Decision：

```text
Phase 0 H0 gate = pass.
Phase 1 passive fine-label attribution is allowed.
No deployable online result is claimed.
No selector/full online validation is allowed.
```

---

## 3. Phase 1：Passive Fine-Label Attribution

输出：

```text
phase1_passive_attribution/
phase1_visual_dashboard/
rollouts/V26_P1_PASSIVE_R1_P0_03_FINE_ROLE_DEBUG_ONLY_ALL_PATHS_*
```

Blocker / fix：

```text
blocker:
    Initial runtime summaries had fine_label_counts and role_counts separately,
    but did not show which fine label got which path role.
    That was insufficient for H1 attribution.

fix:
    hybrid_memory_controller.py now writes:
        fine_label_condition_metrics
        fine_label_path_role_counts
    run_pipeline_abc_v2.py forwards those fields to semantic_memory_path_summary.jsonl.

result:
    A new V26_P1_PASSIVE_R1 debug row was run and Phase 1 attribution was generated.
```

Phase 1 summary：

| Metric | Value |
|---|---:|
| `phase1_gate_pass` | `true` |
| `fine_label_count` | `9` |
| `condition_rows` | `9` |
| `memory_rows` | `116` |
| `action_rows` | `116` |
| `coarse_internal_label_diversity` | `true` |
| `path_role_diversity_within_coarse` | `true` |
| `D_mean_range_within_coarse_max` | `0.3703437358` |
| `conflict_signal_available` | `false` |
| `scale_risk_signal_available` | `false` |

Key attribution：

```text
STRUCTURE_ANCHOR labels:
    road/building/wall/fence/sidewalk
    D_mean range = 0.3703437358
    frame/global dominant roles include POSITIVE_LONG and PROTECT_NEUTRAL
    TTT dominant roles include POSITIVE_LONG and NEUTRAL_KEEP

LOW_VALUE_STUFF labels:
    sky/vegetation/grass
    D_mean range = 0.1181245149
    sky stays NEUTRAL_KEEP
    vegetation/grass can become NEGATIVE_SHORT when high-D
```

Important boundary：

```text
update_conflict_energy and scale-state risk are not token-aligned runtime inputs.
Therefore v26 Phase 2 tests fine-label + D_g + Q_mask conditioned policies,
not the full fine-label + D_g + conflict + scale-risk formula from the plan.
This is recorded in passive_attribution_summary.json and per-label CSVs.
```

Decision：

```text
Phase 1 gate = pass for explanation.
Phase 2 chunk10 h10 screen is allowed.
No pairwise/all-memory/selector/full validation is allowed yet.
```

---

## 4. Phase 2：Single-Path h10 Screen

输出：

```text
phase2_h10_screen_report_R1/
matrix_logs/phase2_h10_screen_R1/
rollouts/V26_P2_H10_R1_*
```

设置：

```text
chunk = 10
horizon = 10
GPUs = 0,1,2,3
probe_cache_mode = readwrite
probe_cache_payload = read_path_min
reference = V26_REF_R1_P0_00_H9_REFERENCE chunk10 h10
rows completed = 15/15
failures = 0
```

Candidates：

```text
FG_FINE_01_STRUCTURE_KEEP
FG_FINE_02_LOWSTUFF_HIGHD_SKIP
FG_FINE_03_SKY_NEUTRAL
FG_FINE_04_STRUCTURE_RESCUE
FG_FINE_05_CONFLICT_CONDITIONED
SWA_FINE_01_OVERLAP_STRUCTURE_KEEP
SWA_FINE_02_SKY_PARTIAL_KEEP
SWA_FINE_03_VEGETATION_CONDITIONAL
SWA_FINE_04_BOUNDARY_PROTECT
SWA_FINE_05_CACHE_LIFECYCLE
TTT_FINE_01_STRUCTURE_POSITIVE
TTT_FINE_02_SKY_NEUTRAL
TTT_FINE_03_SCALE_CONDITIONED
TTT_FINE_04_LOWSTUFF_HIGHD_SHORT
TTT_FINE_05_STRUCTURE_PROTECT
```

h10 gate summary：

| Metric | Best |
|---|---:|
| Best h10 ATE delta vs H9 | `-0.2713026491` |
| Best h10 ATE candidate | `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` |
| Best h10 `[200,300)` delta vs H9 | `-1.2687051048` |
| Best h10 `[200,300)` candidate | `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` |
| Best h10 `[400,600)` delta vs H9 | `-0.1100681466` |
| Phase 2 gate pass candidates | `[]` |

Key rows：

| Candidate | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta |
|---|---:|---:|---:|
| `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` | `-0.2713026491` | `-1.2687051048` | `-0.1100681466` |
| `SWA_FINE_02_SKY_PARTIAL_KEEP` | `-0.2713026491` | `-1.2687051048` | `-0.1100681466` |
| `SWA_FINE_04_BOUNDARY_PROTECT` | `-0.2713026491` | `-1.2687051048` | `-0.1100681466` |
| `FG_FINE_02_LOWSTUFF_HIGHD_SKIP` | `-0.0397311078` | `-0.1088859305` | `-0.0302639389` |
| `TTT_FINE_04_LOWSTUFF_HIGHD_SHORT` | `-0.0245481029` | `-0.0371714555` | `-0.0151619074` |
| `TTT_FINE_01_STRUCTURE_POSITIVE` | `+0.0843257993` | `+0.1456020323` | `+0.0873413379` |

Runtime：

```text
h10 rows with wall_seconds:
    count = 15
    min = 233s
    max = 500s
    mean = 459.0s
```

Decision：

```text
Phase 2 h10 local gate = fail.

Required:
    h10 [200,300) delta <= -3m
    or h10 ATE delta <= -1.5m with downstream regression <= +1m

Observed:
    best h10 [200,300) delta = -1.2687051048m
    best h10 ATE delta = -0.2713026491m
```

Because h10 did not pass gate, only a small h15 top confirmation was run. No chunk6/chunk16 expansion was launched.

---

## 5. Phase 2：h15 Top Confirmation

输出：

```text
phase2_h15_top_report_R1/
matrix_logs/phase2_h15_top_R1/
rollouts/V26_P2_H15_TOP_R1_*
```

设置：

```text
chunk = 10
horizon = 15
reference = V26_REF_R1_P0_00_H9_REFERENCE chunk10 h15
selected rows completed = 5/5
failures = 0
```

Selected candidates：

```text
SWA_FINE_01_OVERLAP_STRUCTURE_KEEP
SWA_FINE_03_VEGETATION_CONDITIONAL
FG_FINE_02_LOWSTUFF_HIGHD_SKIP
FG_FINE_01_STRUCTURE_KEEP
TTT_FINE_04_LOWSTUFF_HIGHD_SHORT
```

h15 gate summary：

| Metric | Best |
|---|---:|
| Best h15 ATE delta vs H9 | `-0.1331930081` |
| Best h15 ATE candidate | `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` |
| Best h15 `[200,300)` delta vs H9 | `-1.0294599587` |
| Best h15 `[200,300)` candidate | `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` |
| Best h15 `[400,600)` delta vs H9 | `-0.0518444969` |
| Phase 2 gate pass candidates | `[]` |

Rows：

| Candidate | h15 ATE delta | h15 `[200,300)` delta | h15 `[400,600)` delta |
|---|---:|---:|---:|
| `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` | `-0.1331930081` | `-1.0294599587` | `+0.0834705376` |
| `SWA_FINE_03_VEGETATION_CONDITIONAL` | `+0.0402787665` | `+0.0162013181` | `+0.0438690554` |
| `FG_FINE_02_LOWSTUFF_HIGHD_SKIP` | `-0.0639994441` | `-0.1722233807` | `-0.0518444969` |
| `FG_FINE_01_STRUCTURE_KEEP` | `+0.0005925980` | `-0.2142159294` | `+0.0241219029` |
| `TTT_FINE_04_LOWSTUFF_HIGHD_SHORT` | `-0.0709001534` | `-0.1343000885` | `-0.0400082445` |

Runtime：

```text
h15 rows with wall_seconds:
    count = 5
    min = 451s
    max = 713s
    mean = 582.2s
```

Decision：

```text
Phase 2 h15 confirmation gate = fail.

Required:
    h15 ATE delta <= -1.5m
    or h15 [200,300) delta <= -2.5m
    with [400,600) regression <= +1m

Observed:
    best h15 ATE delta = -0.1331930081m
    best h15 [200,300) delta = -1.0294599587m
```

---

## 6. Recommended Repair：TTT Risk-Coupled Diagnostic

触发原因：

```text
Phase 2 original h10 screen 没有强 semantic 单路径信号。
v26 plan failure routing says:
    h10 无强信号 -> semantic 单独不是主因 -> 转 D_g/conflict/scale-conditioned role。

当前 blocker:
    update_conflict_energy / scale-state risk 尚不是 HMC semantic router 的 token-aligned input。

可合法尝试的修复：
    使用已有 TTT-native risk path 做 write-side diagnostic coupling，
    不把它声明为完整 token-aligned fine semantic router。
```

代码修改：

```text
tools/run_v24_candidate_rollout.sh:
    added V26 risk-coupled diagnostic candidates:
        TTT_FINE_RISK_01_CONFLICT_TRI
        TTT_FINE_RISK_02_SCALE_STATE
        TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER

    TTT_FINE_RISK_01_CONFLICT_TRI:
        semantic_role_policy = fine_ttt_lowstuff_highd_short
        semantic_memory_paths = ttt
        ttt_write_gradient_reversal_mode = tri_replay
        ttt_write_gradient_reversal_risk_source = update_conflict_energy
        ttt_write_gradient_reversal_chunk_gammas = v22-style fixed chunk gamma map

    TTT_FINE_RISK_02_SCALE_STATE:
        semantic_role_policy = fine_ttt_scale_conditioned
        semantic_memory_paths = ttt
        ttt_write_scale_state_mode = projection_risk
        ttt_write_scale_state_proxy = pose_step_ema
        ttt_write_scale_state_carrier = structure_lowdg
        ttt_write_gradient_reversal_risk_source = v19_scale_state
        ttt_write_native_delta_gate_mode = orthogonal_suppress

    TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER:
        semantic_role_policy = fine_ttt_structure_positive
        semantic_memory_paths = ttt
        ttt_write_gradient_reversal_risk_source = update_conflict_energy
        ttt_write_commit_filter_mode = old_decay_by_risk
        ttt_write_commit_filter_risk_source = update_conflict_energy
        ttt_write_commit_filter_scope = tail_overlap
        ttt_write_commit_filter_stat = q90

tools/v24_candidate_bank_report.py:
    added family labels for these diagnostic candidates,
    so the report no longer drops them as unknown.
```

验证：

```text
bash -n tools/run_v24_candidate_rollout.sh

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v24_candidate_bank_report.py

PASS
```

Risk h10 输出：

```text
phase2_risk_h10_report_R1/
matrix_logs/phase2_risk_h10_R1/
rollouts/V26_P2_RISK_R1_*

chunk = 10
horizon = 10
rows completed = 3/3
failures = 0
reference = V26_REF_R1_P0_00_H9_REFERENCE chunk10 h10
```

Risk h10 rows：

| Candidate | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta | wall seconds |
|---|---:|---:|---:|---:|
| `TTT_FINE_RISK_01_CONFLICT_TRI` | `-0.0577597303` | `-0.2237371648` | `+0.1208392339` | `924` |
| `TTT_FINE_RISK_02_SCALE_STATE` | `-1.9228654883` | `-1.8379048956` | `-2.6743106654` | `751` |
| `TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER` | `+0.0828880496` | `-0.8644514192` | `+0.4836488261` | `1630` |

Risk h10 decision：

```text
TTT_FINE_RISK_02_SCALE_STATE produced the strongest v26 h10 signal:
    h10 ATE delta = -1.9228654883m
    h10 [200,300) delta = -1.8379048956m
    h10 [400,600) delta = -2.6743106654m

But it still did not pass the original Phase 2 h10 local gate:
    required h10 [200,300) delta <= -3m
    observed best h10 [200,300) delta = -1.8379048956m

Because the h10 ATE signal was materially stronger than the original semantic rows,
one h15 top confirmation was allowed for this diagnostic candidate.
```

Risk h15 输出：

```text
phase2_risk_h15_top_report_R1/
matrix_logs/phase2_risk_h15_top_R1/
rollouts/V26_P2_RISK_H15_TOP_R1_*

chunk = 10
horizon = 15
rows completed = 1/1
failures = 0
reference = V26_REF_R1_P0_00_H9_REFERENCE chunk10 h15
```

Risk h15 row：

| Candidate | h15 ATE delta | h15 `[200,300)` delta | h15 `[400,600)` delta | wall seconds |
|---|---:|---:|---:|---:|
| `TTT_FINE_RISK_02_SCALE_STATE` | `-1.0253418796` | `-1.6843312369` | `-1.7976685585` | `939` |

Risk h15 decision：

```text
Phase 2 h15 risk confirmation gate = fail.

Required:
    h15 ATE delta <= -1.5m
    or h15 [200,300) delta <= -2.5m
    with [400,600) regression <= +1m

Observed:
    h15 ATE delta = -1.0253418796m
    h15 [200,300) delta = -1.6843312369m
    h15 [400,600) delta = -1.7976685585m

This is a real improvement over original v26 semantic-only rows,
but it still does not pass the v26 Phase 2 entry gate.
```

Speed note：

```text
Risk-coupled TTT rows are substantially slower than original h10 screen rows:
    h10 scale-state row = 751s
    h10 conflict tri row = 924s
    h10 conflict commit-filter row = 1630s
    h15 scale-state row = 939s

This makes full risk-coupled chunk/horizon sweeping unjustified without a gate-passing h10/h15 row.
```

Boundary：

```text
These risk rows are diagnostic short rollouts.
They do not count as deployable online TTT write success.
They do not enable selector/full online validation.
They do not prove token-aligned fine-label conflict/scale-risk routing.
```

---

## 7. Phase 5：Durability / Washout Attribution

触发原因：

```text
TTT_FINE_RISK_02_SCALE_STATE:
    h10 ATE delta = -1.9228654883m
    h15 ATE delta = -1.0253418796m

This satisfies the "h10 stronger than h15" diagnostic condition,
even though it does not pass the Phase 2 h15/full-run gate.
Per v26 failure routing, h10 strong / h15 weak should trigger durability attribution.
```

输出：

```text
phase5_state_attribution_R1/
    phase5_washout_summary.json
    h10_h15_state_norms.csv
    path_overwrite_ratio.csv
    label_role_mass_h10_h15.csv
    memory_path_washout_summary.md
```

重要边界：

```text
No .pt HMC/merge state snapshots were found in the h10/h15 rollout dirs.
Therefore Phase 5 uses proxy evidence from landed JSONL/CSV summaries.
It does not claim full tensor-state overwrite attribution.
```

Phase 5 summary：

| Metric | Value |
|---|---:|
| `evidence_level` | `proxy_only_no_tensor_state_snapshots` |
| h10 ATE delta | `-1.9228654883` |
| h15 ATE delta | `-1.0253418796` |
| ATE durability `abs(h15/h10)` | `0.5332364019` |
| h10 `[200,300)` delta | `-1.8379048956` |
| h15 `[200,300)` delta | `-1.6843312369` |
| `[200,300)` durability `abs(h15/h10)` | `0.9164409061` |
| common chunks compared | `11` |
| common `hash_H_next` exact matches | `11/11` |
| h15-only tail chunks | `5` |
| tensor state snapshots available | `false` |

Path overwrite proxy：

| Path | h10 total side-effect | h15 tail side-effect | tail / h10 | Evidence |
|---|---:|---:|---:|---|
| `ttt` | `0.3113787164` | `0.1381085392` | `0.4435387761` | `hmc_state_hash.jsonl` |
| `frame_attention_bias` | `24.1899759769` | `10.7929420471` | `0.4461741530` | `hook_effect_summary` |
| `swa_source_replace` | `0.0` | `0.0` | `0.0` | `hook_effect_summary` |

Interpretation：

```text
Common h10/h15 chunks are hash-identical:
    common hash_H_next exact matches = 11/11

The h15 run then adds 5 tail chunks after the h10 endpoint.
Those tail chunks continue TTT side effects and frame-bias effects:
    TTT tail/h10 side-effect proxy ratio ~= 0.4435
    frame-bias tail/h10 side-effect proxy ratio ~= 0.4462
    SWA source replace tail/h10 proxy ratio = 0

Available evidence therefore points to TTT/scale-state tail update as the most
plausible washout path, not SWA source replacement.

But because tensor snapshots are missing, this is a proxy attribution only.
It does not legally enable selector/full online validation.
```

Decision：

```text
Phase 5 diagnostic = partial pass as a washout hypothesis:
    likely path = TTT / scale-state tail update

Phase 5 does not produce a durable correction action:
    the tested W_long/W_short repair failed h10 gate
    the tested scale-state + conflict-energy commit-filter repair failed h10 gate
    no h15 gate passed
    no selector/full online validation is allowed
```

Repair attempt：W_long/W_short dual lifetime

```text
Based on the Phase 5 proxy attribution, a minimal skip-aware TTT repair was tried:
    TTT_FINE_REPAIR_01_SCALE_DUAL_LIFETIME

Configuration:
    semantic_role_policy = fine_ttt_scale_conditioned
    semantic_memory_paths = ttt,lifecycle
    ttt_write_scale_state_mode = projection_risk
    ttt_write_gradient_reversal_risk_source = v19_scale_state
    ttt_write_gradient_reversal_transient_mode = dual_lifetime
    ttt_write_gradient_reversal_transient_long_scale = 0.25
    ttt_write_transient_delta_ttl = 3
```

Output：

```text
phase5_repair_h10_report_R1/
matrix_logs/phase5_repair_h10_R1/
rollouts/V26_P5_REPAIR_H10_R1_*

chunk = 10
horizon = 10
rows completed = 1/1
failures = 0
wall_seconds = 754
```

Repair h10 result：

| Candidate | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta | wall seconds |
|---|---:|---:|---:|---:|
| `TTT_FINE_REPAIR_01_SCALE_DUAL_LIFETIME` | `-0.4875482747` | `-0.4996539742` | `-0.6134233872` | `754` |

Runtime evidence：

```text
dlbank_transient_mode = dual_lifetime
dlbank_transient_applied_layer_count = 18 for active chunks
dlbank_short_stored = true for active chunks
dlbank_transient_long_scale = 0.25
dlbank_short_ttl_out decays to 0 by the final chunks
mean memory_ttt_mean_rel_diff = 0.0341914484
```

Decision：

```text
W_long/W_short dual-lifetime repair = fail.

It weakened the useful h10 scale-state signal:
    previous TTT_FINE_RISK_02_SCALE_STATE h10 ATE delta = -1.9228654883m
    dual-lifetime h10 ATE delta = -0.4875482747m

It also weakened the local disease-window result:
    previous h10 [200,300) delta = -1.8379048956m
    dual-lifetime h10 [200,300) delta = -0.4996539742m

Therefore no h15 confirmation was launched for this repair.
This suggests the useful part of scale-state correction requires long-term TTT state,
while the remaining h15 weakness is not solved by simply moving most correction
into short apply-only delta.
```

Repair attempt：scale-state + conflict-energy commit filter

```text
After the dual-lifetime split weakened h10, a second minimal skip-aware TTT
commit repair was tried:
    TTT_FINE_REPAIR_02_SCALE_CONFLICT_COMMIT_FILTER

Configuration:
    semantic_role_policy = fine_ttt_scale_conditioned
    semantic_memory_paths = ttt
    ttt_write_scale_state_mode = projection_risk
    ttt_write_gradient_reversal_risk_source = v19_scale_state
    ttt_write_commit_filter_mode = old_decay_by_risk
    ttt_write_commit_filter_risk_source = update_conflict_energy
    ttt_write_commit_filter_scope = tail_overlap
    ttt_write_commit_filter_stat = q90
    ttt_write_commit_filter_base = 0.35
    ttt_write_commit_filter_gain = 0.60
    ttt_write_commit_filter_min = 0.15
    ttt_write_commit_filter_max = 1.0
```

Output：

```text
phase5_repair2_h10_report_R1/
matrix_logs/phase5_repair2_h10_R1/
rollouts/V26_P5_REPAIR2_H10_R1_*

chunk = 10
horizon = 10
rows completed = 1/1
failures = 0
wall_seconds = 1381
```

Repair2 h10 result：

| Candidate | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta | wall seconds |
|---|---:|---:|---:|---:|
| `TTT_FINE_REPAIR_02_SCALE_CONFLICT_COMMIT_FILTER` | `-0.1640035255` | `-0.8490782836` | `+0.0548300882` | `1381` |

Runtime evidence：

```text
command-line flags include:
    ttt_write_scale_state_mode = projection_risk
    ttt_write_gradient_reversal_risk_source = v19_scale_state
    ttt_write_commit_filter_mode = old_decay_by_risk
    ttt_write_commit_filter_risk_source = update_conflict_energy

01.log evidence:
    ttt_write_commit_filter_applied True count = 11
    update_conflict_energy commit-filter risk source count = 209
    observed commit-filter scale count = 198
    commit-filter scale min = 0.15
    commit-filter scale max = 0.3489231601
    commit-filter scale mean = 0.2437206063
```

Decision：

```text
scale-state + conflict-energy commit-filter repair = fail.

It weakened the useful h10 scale-state signal more than dual-lifetime:
    previous TTT_FINE_RISK_02_SCALE_STATE h10 ATE delta = -1.9228654883m
    conflict-filter repair h10 ATE delta = -0.1640035255m

It also weakened the disease-window result:
    previous h10 [200,300) delta = -1.8379048956m
    conflict-filter repair h10 [200,300) delta = -0.8490782836m

Therefore no h15 confirmation was launched for this repair.
This suggests naive tail-overlap conflict-energy decay removes too much of the
beneficial scale-state correction instead of preserving it durably.
```

---

## 8. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Phase 0 implementation hard gate | pass | fine labels and four path role streams available; no-op parity exact |
| Phase 1 passive attribution | pass | coarse groups contain fine-label D/role differences |
| Conflict/scale token conditioning | diagnostic attempted | existing TTT-native risk coupling improved ATE but no token-aligned semantic risk stream |
| Phase 2 original h10 screen | fail | best `[200,300)` delta = `-1.2687m`, gate requires `<= -3m` |
| Phase 2 original h15 top | fail | best h15 ATE delta = `-0.1332m`, best `[200,300)` = `-1.0295m` |
| Phase 2 risk h10 diagnostic | fail | best h10 ATE delta = `-1.9229m`, but best `[200,300)` = `-1.8379m`, gate requires `<= -3m` |
| Phase 2 risk h15 diagnostic | fail | h15 ATE delta = `-1.0253m`, h15 `[200,300)` = `-1.6843m`, gate requires `<= -1.5m` or `<= -2.5m` respectively |
| Phase 5 washout attribution | partial diagnostic | proxy evidence points to TTT/scale-state tail update; no tensor snapshots, no durable repair action |
| Phase 5 W_long/W_short repair | fail | dual-lifetime short repair weakened h10 ATE to `-0.4875m` and `[200,300)` to `-0.4997m` |
| Phase 5 scale/conflict commit-filter repair | fail | conflict-energy tail commit filter weakened h10 ATE to `-0.1640m` and `[200,300)` to `-0.8491m` |
| Phase 3 pairwise | not started | no Phase 2 candidate passed |
| Phase 4 all-memory | not started | pairwise entry forbidden |
| No-GT selector | not started | gate failed |
| Full online validation | not started | selector/full-run entry forbidden |

Boundary：

```text
No v26 short-rollout result counts as deployable online TTT write success.
No GT semantic was used.
No no-GT selector was evaluated.
No full online validation was launched.
No online Target-25 result was produced in v26.
```

---

## 9. Final Decision

v26 的真实成功点：

```text
1. Fine labels are now runtime-visible:
       L_sem_tok uses stable fine-label ids from video-masklet L_sem.

2. Path-specific role streams are implemented:
       R_frame_tok
       R_global_tok
       R_swa_tok
       R_ttt_tok

3. H0 hard gate passed:
       cache_hit_rate = 1.0
       fine_label_count = 8
       no-op direct pose diff = 0
       all four paths consumed semantic role

4. Phase 1 found real attribution structure:
       STRUCTURE_ANCHOR internal D_mean range = 0.3703437358
       LOW_VALUE_STUFF internal D_mean range = 0.1181245149
       fine labels receive different roles within the same coarse group.
```

v26 的关键负结果：

```text
Fine-label path-specific semantic router alone did not produce strong trajectory correction.

risk-coupled diagnostic improved the signal but still failed the Phase 2 gate.

original h10 best:
    SWA_FINE_01_OVERLAP_STRUCTURE_KEEP
    ATE delta = -0.2713026491m
    [200,300) delta = -1.2687051048m

original h15 best:
    SWA_FINE_01_OVERLAP_STRUCTURE_KEEP
    ATE delta = -0.1331930081m
    [200,300) delta = -1.0294599587m

risk h10 best:
    TTT_FINE_RISK_02_SCALE_STATE
    ATE delta = -1.9228654883m
    [200,300) delta = -1.8379048956m
    [400,600) delta = -2.6743106654m

risk h15 best:
    TTT_FINE_RISK_02_SCALE_STATE
    ATE delta = -1.0253418796m
    [200,300) delta = -1.6843312369m
    [400,600) delta = -1.7976685585m

Phase 5 proxy attribution:
    common h10/h15 chunks hash match = 11/11
    h15-only tail chunks = 5
    TTT tail/h10 side-effect proxy ratio = 0.4435387761
    SWA source replace tail/h10 proxy ratio = 0
    likely washout path = TTT / scale-state tail update

W_long/W_short repair:
    TTT_FINE_REPAIR_01_SCALE_DUAL_LIFETIME
    h10 ATE delta = -0.4875482747m
    h10 [200,300) delta = -0.4996539742m
    no h15 confirmation launched

scale-state + conflict-energy commit-filter repair:
    TTT_FINE_REPAIR_02_SCALE_CONFLICT_COMMIT_FILTER
    h10 ATE delta = -0.1640035255m
    h10 [200,300) delta = -0.8490782836m
    no h15 confirmation launched

No candidate met:
    h10 [200,300) delta <= -3m
    h15 ATE delta <= -1.5m
    h15 [200,300) delta <= -2.5m
```

Interpretation：

```text
v26 answers the v25B engineering question:
    video-masklet fine labels can be projected,
    routed into path-specific role streams,
    consumed by frame/global/SWA/TTT,
    and audited without no-op drift.

But it does not support promoting semantic role routing as the Target-25 mainline.
The original semantic-only signal is a small SWA/local-continuity effect.
The recommended risk-coupled diagnostic found a stronger scale-state TTT signal,
but even that missed both h10 local and h15 durability gates.
Phase 5 proxy attribution suggests the remaining h15 decay is caused by extra
TTT/scale-state tail updates after the h10 endpoint, not by SWA source replacement.
The tested W_long/W_short dual-lifetime repair made h10 much weaker,
so the simple short-lifetime split is not the durable repair action.
The tested scale-state + conflict-energy tail commit filter also made h10 much weaker,
so naive conflict decay is not the durable repair action either.

The remaining missing semantic condition is token-aligned conflict/scale-risk.
The existing TTT-native risk path suggests scale-state coupling is more promising
than semantic-only routing, but it is still not strong enough to justify Phase 3/4
or selector/full online validation.
```

Conclusion type：

```text
Semantic fine-label router is valid infrastructure and useful diagnostic,
but current D/Q-conditioned semantic role routing is not sufficient for Target-25.
Existing scale-state risk coupling is a better repair direction than more semantic sweeps,
but both tested durable-repair actions weakened h10 before any selector/full validation.
```

Next required direction：

```text
Do not start Phase 3 / Phase 4 / selector / full online from v26 candidates.
Do not expand h15/chunk6/chunk16 sweeps for this semantic family.

Target-25 mainline should return to:
    explicit online trajectory-state,
    explicit online scale-state,
    more principled token-aligned risk routing than the failed dual-lifetime split
    and failed tail conflict-energy commit filter,
    skip-aware memory lifecycle,
    merge/gauge-aware correction,
    or a token-aligned conflict/scale-risk module before revisiting semantic routing.
```
