# ACL2 v28 实验复盘：SemanticPrior TokenExact RiskConditioned AllMemory Parallel

日期：2026-05-22（Asia/Singapore）
计划文件：`docs/ACL2_v28_SemanticPrior_TokenExact_RiskConditioned_AllMemory_Parallel_Plan.md`
主结果目录：`results/kitti01_hmc_v2/acl2_v28_semanticprior_tokenexact_riskconditioned_allmemory_parallel/`

本轮原则：只记录实际落盘结果；不把 implementation audit、passive attribution、smoke、short rollout、failed gate、proxy risk、或未启动矩阵写成 deployable online success。`C_ttt_conflict_tok` 和 `S_scale_risk_tok` 没有通过 token-level hard gate 时，不启动 Phase 2 candidate gate、h15、Phase 3/4、selector 或 full online validation。

---

## 0. 当前结论

v28 已按计划最小执行清单推进到 Phase 2 h10 stop rule，未达到 Target-25。

已完成并落盘：

```text
1. 阅读 v28 计划，并实现 token-exact / token-aligned risk-conditioned semantic role router 接线。
2. Phase 0 H0 implementation hard gate 通过：
       rows = 9/9
       failures = 0
       no-op parity pass
       C_ttt_conflict_tok token_exact/nonempty pass
       S_scale_risk_tok token_exact/nonempty pass
3. Phase 1 passive token-exact semantic-risk attribution 完成：
       R1 用 read-path-only debug row，conflict token unavailable，标记为无效 attribution evidence
       R2 改用 all-path debug row，conflict/scale risk available，phase1_gate_pass = true
4. Phase 2 chunk10 h10 screen 完成：
       rows = 12/12
       failures = 0
5. SWA boundary diagnostic 完成：
       candidates = 3
       boundary_gate_pass_candidates = []
6. Phase 2 gate = fail。
```

最终边界：

```text
1. h10 best ATE delta = -0.3641983213m.
2. h10 best [200,300) delta = -2.3655037795m.
3. h10 best [400,600) delta = -0.3077443114m.
4. best h10 candidates are SWA_RISK_ONLY / SWA_SEM_ONLY / SWA_SEM_RISK with identical trajectory deltas.
5. semantic-risk is not stronger than semantic-only by H1 criteria.
6. SWA boundary 10f / 20f metrics regress vs H9:
       mean boundary 10f delta = +0.4789576412m
       mean boundary 20f delta = +0.3681139080m
7. No Phase 2 candidate passed gate.
8. No h15 confirmation was launched.
9. No Phase 3 pairwise combination was started.
10. No Phase 4 all-memory validation was started.
11. No no-GT selector was started.
12. No full online validation was launched.
13. No online Target-25 result was produced in v28.
```

Gate result：

```text
Required to enter h15 / Phase 3:
    h10 [200,300) delta <= -3m
    or h10 ATE delta <= -1.5m

Observed:
    best h10 [200,300) delta = -2.3655037795m
    best h10 ATE delta = -0.3641983213m

SWA boundary requirement:
    boundary_10f_delta <= +0.25m
    boundary_20f_delta <= +0.25m
    chunk_boundary_pose_jump_delta <= +0.25m

Observed:
    boundary_10f_delta = +0.4789576412m
    boundary_20f_delta = +0.3681139080m
    pose_jump_delta = -0.0186129478m

Therefore:
    Phase 2 gate = fail.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v28 不是 GT semantic 实验。
semantic_source = video_masklet_frontend_cache
uses_gt_semantic = false
uses_video_masklet_semantic = true

C_ttt_conflict_tok:
    level = token_exact_selected_ttt_layers
    source = pre_role_probe_ttt_update_conflict_energy_token_vector

S_scale_risk_tok:
    level = token_window_proxy
    source = v19_scale_state_payload_abs_log_ratio_x_token_update_magnitude
    construction = norm01(U_i * R_window)
    finite_difference_sensitivity = false

因此 v28 可以声称 conflict token stream 是 selected-layer token-exact，
但 scale risk 只能声称为 token-to-window-risk proxy，不能声称已实现
finite-difference token scale sensitivity。
```

---

## 1. 工程修改记录

新增 / 修改：

```text
loger/pipeline/ttt_write_controller.py:
    _build_gradient_reversal_risk_flat records:
        _ttt_gradient_reversal_risk_source_vector
        _ttt_gradient_reversal_risk_source_vector_count
    This exposes selected-layer token-level update_conflict_energy risk
    for the v28 pre-role probe.

loger/pipeline/hybrid_memory_controller.py:
    HybridMemoryControlPrior added:
        C_ttt_conflict_tok
        S_scale_risk_tok
    set_semantic_condition_signals accepts:
        conflict_tok / conflict_token_exact
        scale_tok / scale_token_exact
    _fine_path_roles consumes token tensors when:
        token exists
        token_count matches D_tok
        token_exact flag is true
    runtime debug records:
        condition_signal_conflict_token_exact
        condition_signal_scale_risk_token_exact
        condition_signal_conflict_token_nonempty
        condition_signal_scale_risk_token_nonempty
        token_count / mean / p90

run_pipeline_abc_v2.py:
    added _extract_v28_conflict_condition
    added _v28_scale_token_condition
    rebuilds final control_prior after token risk conditions are set
    forwards token risk availability/exact/count/mean/p90 fields to:
        semantic_role_summary.jsonl
        semantic_memory_path_summary.jsonl

tools/run_v24_candidate_rollout.sh:
    added v28 Phase 0 aliases:
        V28_P0_00_H9_REFERENCE
        V28_P0_01_SEM_LOADED_IGNORED
        V28_P0_02_SEM_PASS_THROUGH_CONSUMED
        V28_P0_03_TOKEN_RISK_DEBUG_ONLY
        V28_P0_04_FRAME_SOURCE_SMOKE
        V28_P0_05_GLOBAL_SOURCE_SMOKE
        V28_P0_06_SWA_SOURCE_SMOKE
        V28_P0_07_TTT_WRITE_SMOKE
        V28_P0_08_ALL_PATH_DEBUG_ONLY
    added v28 Phase 2 aliases:
        FRAME_SEM_ONLY / FRAME_RISK_ONLY / FRAME_SEM_RISK
        GLOBAL_SEM_ONLY / GLOBAL_RISK_ONLY / GLOBAL_SEM_RISK
        SWA_SEM_ONLY / SWA_RISK_ONLY / SWA_SEM_RISK
        TTT_SEM_ONLY / TTT_RISK_ONLY / TTT_SEM_RISK

tools/v24_candidate_bank_report.py:
    added v28 candidate family labels.

tools/v28_implementation_self_audit.py:
    strict v28 H0 audit:
        cache quality
        direct same-batch no-op pose parity
        role streams non-empty
        frame/global/SWA/TTT consumed
        conflict/scale token streams available
        conflict/scale token_exact true
        conflict/scale token_nonempty true
        context empty source events = 0

tools/v26_passive_fine_label_attribution.py:
    added --run-prefix filter so v28 Phase 1 passive attribution can avoid
    mixing Phase 0/Phase 2 rows into the passive report.
    added v28 passive run-name stripping support.
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v28_implementation_self_audit.py \
    loger/pipeline/ttt_write_controller.py \
    loger/pipeline/hybrid_memory_controller.py \
    run_pipeline_abc_v2.py \
    tools/v24_candidate_bank_report.py \
    tools/v26_passive_fine_label_attribution.py

bash -n \
    tools/run_v24_candidate_rollout.sh \
    tools/run_attention_cue_experiment.sh

PASS
```

工程边界：

```text
The launcher aliases reuse existing causal/fine policies.
The *_RISK_ONLY rows are risk-conditioned diagnostic aliases, not a newly
implemented fully label-free ablation family in every path.
This is why H1 interpretation compares actual landed behavior, not idealized names.
```

---

## 2. Blocker 与修复记录

### Blocker 1：Phase 0 R1 OOM

现象：

```text
matrix_logs/phase0_smoke_R1/
rollouts/V28_P0_SMOKE_R1_*

early rows failed with CUDA out of memory.
```

原因：

```text
9 rows were launched at once on 4 GPUs.
Some GPUs had two LoGeR processes at the same time.
Each process used roughly 7-11GB on 22GB GPUs.
```

处理：

```text
R1 is invalid and not counted.
Rerun Phase 0 with at most one LoGeR process per GPU.
```

### Blocker 2：Phase 0 R2 role_debug UnboundLocalError

现象：

```text
R2 rows failed with:
    UnboundLocalError:
        cannot access local variable 'role_debug' where it is not associated with a value
```

原因：

```text
C_ttt_conflict_tok / S_scale_risk_tok fields were accidentally inserted into
the _apply_semantic_role_policy(...) call area before role_debug existed.
```

修复：

```text
loger/pipeline/hybrid_memory_controller.py:
    removed C/S token fields from the _apply_semantic_role_policy(...) call.
    added C/S token fields to the returned HybridMemoryControlPrior object.
```

结果：

```text
py_compile PASS.
R2 is invalid and not counted.
R3 was launched after the fix.
```

### Blocker 3：Phase 1 R1 passive attribution lacked conflict token

现象：

```text
V28_P1_PASSIVE_R1_V28_P0_03_TOKEN_RISK_DEBUG_ONLY_*

condition_signal_conflict_available = false
condition_signal_conflict_level = unavailable
condition_signal_conflict_summary.reason = preprobe_unavailable
condition_signal_conflict_summary.has_P_ttt_write = false

scale risk was available, but conflict token was not.
```

原因：

```text
V28_P0_03_TOKEN_RISK_DEBUG_ONLY is read-path-only.
It does not produce P_ttt_write for the pre-role update_conflict_energy probe,
so it is not a valid v28 passive conflict-token attribution row.
```

修复：

```text
Rerun Phase 1 with:
    V28_P0_08_ALL_PATH_DEBUG_ONLY

This candidate had already passed H0 with:
    condition_signal_conflict_available = true
    condition_signal_conflict_token_exact = true
    condition_signal_conflict_token_nonempty = true
```

结果：

```text
V28_P1_PASSIVE_R1 is recorded as invalid for conflict attribution.
V28_P1_PASSIVE_R2 is used for the Phase 1 report.
```

---

## 3. Phase 0：Implementation Hard Gate

输出：

```text
implementation_audit/
matrix_logs/phase0_smoke_R3/
rollouts/V28_P0_SMOKE_R3_*
```

Phase 0 smoke：

```text
chunk = 10
horizon = 3
rows completed = 9/9
failures = 0
```

Runtime：

| Candidate | wall seconds |
|---|---:|
| `V28_P0_00_H9_REFERENCE` | `165` |
| `V28_P0_01_SEM_LOADED_IGNORED` | `178` |
| `V28_P0_02_SEM_PASS_THROUGH_CONSUMED` | `196` |
| `V28_P0_03_TOKEN_RISK_DEBUG_ONLY` | `199` |
| `V28_P0_04_FRAME_SOURCE_SMOKE` | `188` |
| `V28_P0_05_GLOBAL_SOURCE_SMOKE` | `198` |
| `V28_P0_06_SWA_SOURCE_SMOKE` | `437` |
| `V28_P0_07_TTT_WRITE_SMOKE` | `498` |
| `V28_P0_08_ALL_PATH_DEBUG_ONLY` | `453` |

H0 audit summary：

| Metric | Value |
|---|---:|
| `all_gate_pass` | `true` |
| `cache_hit_rate` | `1.0` |
| `chunks_with_masklets_ratio` | `1.0` |
| `mean_coverage` | `0.9816746437` |
| `focus_coverage_200_300` | `0.9781515747` |
| `fine_label_count` | `8` |
| `noop_parity_gate_pass` | `true` |
| `runtime_fine_role_policy_available` | `true` |
| `path_specific_role_streams_available` | `true` |
| `condition_signal_conflict_available` | `true` |
| `condition_signal_scale_risk_available` | `true` |
| `condition_signal_conflict_token_exact` | `true` |
| `condition_signal_scale_risk_token_exact` | `true` |
| `condition_signal_conflict_token_nonempty` | `true` |
| `condition_signal_scale_risk_token_nonempty` | `true` |
| `context_empty_source_events` | `0` |

Path consumption：

| Path | Consumed |
|---|---|
| frame | `true` |
| global | `true` |
| SWA | `true` |
| TTT | `true` |

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

Token risk evidence：

```text
C_ttt_conflict_tok:
    level = token_exact_selected_ttt_layers
    source = pre_role_probe_ttt_update_conflict_energy_token_vector
    token_count = 40320

S_scale_risk_tok:
    level = token_window_proxy
    source = v19_scale_state_payload_abs_log_ratio_x_token_update_magnitude
    token_count = 40320
    finite_difference_sensitivity = false
```

Decision：

```text
Phase 0 H0 gate = pass.
Phase 1 passive attribution and Phase 2 chunk10 h10 screen are allowed.
No deployable online result is claimed.
```

---

## 4. Phase 1：Passive Token-Exact Semantic-Risk Attribution

输出：

```text
phase1_passive_attribution/
phase1_visual_dashboard/
rollouts/V28_P1_PASSIVE_R2_*
```

设置：

```text
candidate = V28_P0_08_ALL_PATH_DEBUG_ONLY
chunks = 6, 10, 16
horizon = 10
rows completed = 3/3
failures = 0
```

Phase 1 summary：

| Metric | Value |
|---|---:|
| `phase1_gate_pass` | `true` |
| `fine_label_count` | `9` |
| `condition_rows` | `9` |
| `memory_rows` | `1000` |
| `action_rows` | `1000` |
| `coarse_internal_label_diversity` | `true` |
| `path_role_diversity_within_coarse` | `false` |
| `D_mean_range_within_coarse_max` | `0.2683107930` |
| `conflict_signal_available` | `true` |
| `scale_risk_signal_available` | `true` |

Key attribution rows：

| Fine label | Coarse group | token count | D mean | conflict mean | scale-risk mean |
|---|---|---:|---:|---:|---:|
| `road` | `STRUCTURE_ANCHOR` | `430383` | `0.1599159226` | `0.0696426863` | `0.7032769239` |
| `building` | `STRUCTURE_ANCHOR` | `54292` | `0.3975008196` | `0.0553983310` | `0.5309490129` |
| `wall` | `STRUCTURE_ANCHOR` | `11025` | `0.4282267155` | `0.0693485617` | `0.6242665420` |
| `fence` | `STRUCTURE_ANCHOR` | `16346` | `0.3775920457` | `0.0720330386` | `0.5958423128` |
| `sky` | `LOW_VALUE_STUFF` | `493520` | `0.2191547727` | `0.0810930245` | `0.4446039587` |
| `vegetation` | `LOW_VALUE_STUFF` | `243277` | `0.2249761783` | `0.0640951741` | `0.4335248672` |
| `grass` | `LOW_VALUE_STUFF` | `52307` | `0.3157251313` | `0.0698830994` | `0.4370637995` |

Interpretation：

```text
Phase 1 attribution passes because:
    fine labels differ inside coarse groups,
    D/risk metrics are present,
    conflict and scale-risk conditions are available.

However:
    path_role_diversity_within_coarse = false in this passive R2 aggregate.
    This means the passive attribution is useful as risk/label evidence,
    but it does not by itself prove strong role diversity.
```

Visual boundary：

```text
phase1_visual_dashboard contains aggregate plots:
    per_label_Dg_distribution.png
    per_label_TTT_role_mass.png

Full spatial token map grids requested by the plan were not generated.
Because Phase 2 gate failed, no candidate is promoted to a stage where missing
spatial dashboards would be used to justify h15 / Phase 3 entry.
```

Decision：

```text
Phase 1 gate = pass for attribution.
Phase 2 h10 screen remains allowed.
No deployable online result is claimed.
```

---

## 5. Phase 2：chunk10 h10 Single-Path Screen

输出：

```text
phase2_h10_screen_report_R1/
matrix_logs/phase2_h10_screen_R1/
rollouts/V28_P2_H10_R1_*
rollouts/V28_REF_R1_V28_P0_00_H9_REFERENCE_chunk10_h10_globalgate_H9parent_SWKS3/
```

设置：

```text
chunk = 10
horizon = 10
GPUs = 0,1,2,3
reference = V28_REF_R1_V28_P0_00_H9_REFERENCE chunk10 h10
rows completed = 12/12
failures = 0
```

Candidates：

```text
FRAME_SEM_ONLY
FRAME_RISK_ONLY
FRAME_SEM_RISK
GLOBAL_SEM_ONLY
GLOBAL_RISK_ONLY
GLOBAL_SEM_RISK
SWA_SEM_ONLY
SWA_RISK_ONLY
SWA_SEM_RISK
TTT_SEM_ONLY
TTT_RISK_ONLY
TTT_SEM_RISK
```

Runtime：

| Candidate | wall seconds |
|---|---:|
| `FRAME_RISK_ONLY` | `466` |
| `FRAME_SEM_ONLY` | `483` |
| `FRAME_SEM_RISK` | `493` |
| `GLOBAL_RISK_ONLY` | `553` |
| `GLOBAL_SEM_ONLY` | `474` |
| `GLOBAL_SEM_RISK` | `507` |
| `SWA_RISK_ONLY` | `1209` |
| `SWA_SEM_ONLY` | `478` |
| `SWA_SEM_RISK` | `1259` |
| `TTT_RISK_ONLY` | `1226` |
| `TTT_SEM_ONLY` | `551` |
| `TTT_SEM_RISK` | `1223` |

h10 rows：

| Candidate | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta |
|---|---:|---:|---:|
| `FRAME_RISK_ONLY` | `0.0` | `0.0` | `0.0` |
| `FRAME_SEM_ONLY` | `0.0` | `0.0` | `0.0` |
| `FRAME_SEM_RISK` | `0.0` | `0.0` | `0.0` |
| `GLOBAL_RISK_ONLY` | `-0.0327972027` | `-0.2707409448` | `+0.0098508412` |
| `GLOBAL_SEM_ONLY` | `-0.0327972027` | `-0.2707409448` | `+0.0098508412` |
| `GLOBAL_SEM_RISK` | `-0.0397311078` | `-0.1088859305` | `-0.0302639389` |
| `SWA_RISK_ONLY` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` |
| `SWA_SEM_ONLY` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` |
| `SWA_SEM_RISK` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` |
| `TTT_RISK_ONLY` | `-0.0608005684` | `-0.0625564517` | `-0.0574940246` |
| `TTT_SEM_ONLY` | `+0.0843257993` | `+0.1456020323` | `+0.0873413379` |
| `TTT_SEM_RISK` | `-0.0338513522` | `+0.0609887703` | `-0.0709859863` |

Gate summary：

| Metric | Best |
|---|---:|
| Best h10 ATE delta vs H9 | `-0.3641983213` |
| Best h10 ATE candidate | `SWA_RISK_ONLY` |
| Best h10 `[200,300)` delta vs H9 | `-2.3655037795` |
| Best h10 `[200,300)` candidate | `SWA_RISK_ONLY` |
| Best h10 `[400,600)` delta vs H9 | `-0.3077443114` |
| Phase 2 gate pass candidates | `[]` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

H1 comparison：

```text
FRAME:
    semantic-only / risk-only / semantic-risk are all trajectory no-op.

GLOBAL:
    semantic-risk slightly improves ATE vs semantic-only:
        -0.0397311078 vs -0.0327972027
    but [200,300) becomes weaker:
        -0.1088859305 vs -0.2707409448
    H1 strength criterion is not met.

SWA:
    semantic-only / risk-only / semantic-risk are identical:
        ATE delta = -0.3641983213
        [200,300) delta = -2.3655037795
    semantic-risk is not stronger than semantic-only.

TTT:
    semantic-only regresses:
        ATE delta = +0.0843257993
    risk-only is weakly positive:
        ATE delta = -0.0608005684
    semantic-risk does not improve over risk-only:
        ATE delta = -0.0338513522
        [200,300) delta = +0.0609887703
```

Decision：

```text
Phase 2 h10 gate = fail.

Required:
    h10 [200,300) delta <= -3m
    or h10 ATE delta <= -1.5m

Observed:
    best h10 [200,300) delta = -2.3655037795m
    best h10 ATE delta = -0.3641983213m

H1 does not hold:
    semantic-risk is not materially stronger than semantic-only
    by >= 1.0m ATE or >= 2.0m [200,300).

Therefore no h15 confirmation is allowed.
No Phase 3 / Phase 4 / selector / full online validation is allowed.
```

---

## 6. SWA Boundary Diagnostic

触发原因：

```text
SWA rows produced the best local [200,300) improvement and involved the SWA path.
The v28 plan requires SWA candidates to pass boundary/local continuity checks,
not only trajectory ATE.
```

输出：

```text
phase2_swa_boundary_report_R1/
    swa_boundary_summary.json
    swa_boundary_summary.csv
    swa_boundary_by_candidate_boundary.csv
```

Candidates：

```text
SWA_SEM_ONLY
SWA_RISK_ONLY
SWA_SEM_RISK
```

Boundary summary：

| Metric | Value |
|---|---:|
| boundary candidates | `3` |
| boundary count per candidate | `11` |
| mean boundary 10f delta vs H9 | `+0.4789576412` |
| mean boundary 10f improvement ratio | `-0.0241153897` |
| mean boundary 20f delta vs H9 | `+0.3681139080` |
| mean boundary 20f improvement ratio | `-0.0180280938` |
| mean pose jump delta vs H9 | `-0.0186129478` |
| boundary gate pass candidates | `[]` |

Decision：

```text
SWA local trajectory improvement is real in h10:
    [200,300) delta = -2.3655037795m

But it misses the h10 entry gate and regresses boundary metrics:
    boundary_10f_delta = +0.4789576412m > +0.25m
    boundary_20f_delta = +0.3681139080m > +0.25m

Per the v28 plan, this cannot enter h15 or Phase 3.
```

---

## 7. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Phase 0 H0 implementation hard gate | pass | no-op, cache, path streams, conflict/scale token streams, path consumption passed |
| Phase 1 R1 passive attribution | invalid | read-path-only row lacked P_ttt_write, conflict token unavailable |
| Phase 1 R2 passive attribution | pass | conflict and scale-risk available; D/risk differ by label |
| Phase 2 h10 screen | fail | best ATE `-0.3642m`, best `[200,300)` `-2.3655m`; gate requires `<= -1.5m` ATE or `<= -3m` segment |
| H1 semantic-risk comparison | fail | semantic-risk is not materially stronger than semantic-only |
| SWA boundary diagnostic | fail | boundary 10f/20f deltas regress by `+0.4790m` / `+0.3681m` |
| h15 confirmation | not started | Phase 2 h10 gate failed |
| Phase 3 pairwise | not started | no Phase 2 candidate passed |
| Phase 4 all-memory | not started | pairwise entry forbidden |
| Phase 5 washout attribution | not started | no h10-strong / h15-weak candidate |
| Cross-dataset diagnostic | not started | no h15-gate candidate |
| No-GT selector | not started | gate failed |
| Full online validation | not started | selector/full-run entry forbidden |

Boundary：

```text
No v28 short-rollout result counts as deployable online TTT write success.
No GT semantic was used.
No no-GT selector was evaluated.
No full online validation was launched.
No online Target-25 result was produced in v28.
```

---

## 8. Final Decision

v28 的真实成功点：

```text
1. Token-exact selected-layer TTT conflict stream is implemented and audited:
       C_ttt_conflict_tok
       token_count = 40320
       token_exact = true

2. Token-aligned scale-risk proxy is implemented and audited:
       S_scale_risk_tok
       token_count = 40320
       token_exact = true
       level = token_window_proxy

3. H0 hard gate passed:
       all_gate_pass = true
       cache_hit_rate = 1.0
       no-op parity pass
       frame/global/SWA/TTT consumed
       context_empty_source_events = 0

4. Phase 1 R2 showed fine-label risk variation:
       STRUCTURE_ANCHOR D_mean range contributes to max range = 0.2683107930
       conflict and scale-risk signals are available
```

v28 的关键负结果：

```text
Token-exact / token-aligned risk-conditioned semantic role routing still did not
produce a gate-passing h10 correction.

Best h10:
    SWA_RISK_ONLY / SWA_SEM_ONLY / SWA_SEM_RISK
    ATE delta = -0.3641983213m
    [200,300) delta = -2.3655037795m
    [400,600) delta = -0.3077443114m

But:
    h10 [200,300) gate requires <= -3m
    h10 ATE gate requires <= -1.5m
    SWA boundary 10f / 20f metrics regress
    semantic-risk is not stronger than semantic-only
```

Interpretation：

```text
v28 answers the v27 open question more sharply:
    the failure is not just because conflict risk was broadcast.
    A selected-layer token-exact conflict stream can be constructed and consumed.

However, the current risk-conditioned semantic policies still do not show enough
causal power for Target-25. The strongest effect is again SWA-local:
    useful in [200,300),
    below gate,
    and harmful to boundary-local continuity.

The identical SWA_SEM_ONLY / SWA_RISK_ONLY / SWA_SEM_RISK trajectories suggest
the current SWA intervention is dominated by shared hook/source behavior rather
than by a successfully differentiated semantic-risk role tree.

The TTT rows remain weak:
    TTT_RISK_ONLY is small positive,
    TTT_SEM_ONLY regresses,
    TTT_SEM_RISK does not beat risk-only.
```

Conclusion type：

```text
v28 produced valid infrastructure and stronger auditability,
but not a successful Target-25 correction mechanism.

Semantic Prior Generator should not be promoted as the Target-25 mainline
from this family without a new mechanism beyond the tested token-exact
conflict + token-window scale-risk role router.
```

Next required direction：

```text
Do not start h15 / Phase 3 / Phase 4 / selector / full online from v28 candidates.
Do not expand chunk6/chunk16 or cross-dataset diagnostics for v28 candidates.

If continuing semantic work:
    implement true finite-difference token scale sensitivity,
    separate SWA hook trajectory intervention from semantic-risk role identity,
    design boundary-first SWA policies whose boundary metrics pass before h15.

Otherwise Target-25 mainline should return to:
    explicit online trajectory-state,
    explicit scale-state,
    skip-aware memory lifecycle,
    merge/gauge-aware correction,
    or TTT-native causal actions outside this semantic prior family.
```
