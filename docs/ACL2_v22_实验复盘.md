# ACL2 v22 实验复盘：Durable ContextSkip SemanticAllMemory TTT Target25

日期：2026-05-21（Asia/Singapore）  
计划文件：`docs/ACL2_v22_Durable_ContextSkip_SemanticAllMemory_TTT_Target25_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v22_durable_contextskip_semanticallmemory_ttt_target25/`

本轮原则：只记录实际落盘结果；不把 short rollout、sandbox oracle、GT audit、污染重跑目录、failed durability gate、或未实现的 H5 变体写成 deployable online success。没有通过 durability gate 时，不启动 no-GT selector，也不启动 full online validation。

---

## 0. 工程与配置复盘

新增 / 修改：

```text
tools/run_v22_candidate_rollout.sh:
    v22 trusted short-rollout launcher
    added support / read-only compact / semantic role / skip-aware TTT / skip-aware SWA / durability / lifecycle candidate families
    added V22_SAVE_ATTRIBUTION_STATES for h10/h15 state attribution
    added stale run directory invalidation:
        existing run dir is moved to .INVALID_RERUN_* before a forced/clean rerun

tools/run_v22_matrix.sh:
    v22 GPU matrix scheduler
    phases:
        phaseA_support_compact_semantic
        phaseB_read_only_compact
        phaseC_semantic_role_initial
        phaseD_v21_strongest_attribution
        phaseE_skip_aware_ttt
        phaseF_skip_aware_memory
        phaseG_ttt_durable_commit
        phaseH_ttt_lifecycle

tools/v22_candidate_bank_report.py:
    aggregate h10/h15 candidate rows
    recompute deltas against H9 on matched frame intersection
    report durability and selector/full-online gates

tools/v22_state_attribution.py:
    compare saved HMC / merge states at base, h10 endpoint, h15 endpoint
    report h10->h15 overwrite ratio

tools/run_attention_cue_experiment.sh:
    fixed readonly mode to forward CONTEXT_SOURCE_SKIP_* arguments
    before this fix, readonly compact_kv candidates did not actually apply compact_kv
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    tools/v22_candidate_bank_report.py \
    tools/v22_state_attribution.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v22_candidate_rollout.sh
bash -n tools/run_v22_matrix.sh

PASS
```

工程 blocker 与修复：

```text
blocker 1:
    Phase B read-only compact first run completed, but context_skip_summary showed:
        context_source_skip_requested = false
    Root cause:
        run_attention_cue_experiment.sh did not pass CONTEXT_SOURCE_SKIP_* args in readonly mode.

fix:
    added CONTEXT_SOURCE_SKIP_ARGS and forwarded them to readonly mode.
    Clean rerun confirmed:
        context_source_skip_impl = compact_kv
        context_source_skip_requested = true
        num_context_source_skip_applied = 6
        max_context_source_skip_tokens = 8026
        mean keep ratio about 0.933
        num_context_empty_source_events = 0

blocker 2:
    FORCE rerun initially reused an existing run directory, mixing old invalid JSONL with new valid rows.

fix:
    run_v22_candidate_rollout.sh now moves stale/forced run dirs to .INVALID_RERUN_* before launch.
    Contaminated V22_B_READ_R1 directories were moved under INVALID_BATCH_20260521_150306 and excluded.

blocker 3:
    v22_state_attribution.py failed on PyTorch 2.6 default weights_only=True when loading HybridMemoryState.

fix:
    torch.load(..., weights_only=False) is used for trusted local experiment snapshots.

blocker 4:
    v22_state_attribution.py could not unpickle loger classes when run as tools/v22_state_attribution.py.

fix:
    repo root is inserted into sys.path before torch.load.
```

边界说明：

```text
H9/C9/WINGAM full online boundary was not rerun in v22.
v22 uses the v16 trusted boundary and causal fork snapshots:
    H9_P0_R2 ATE = 34.1257769401m
    C9_P0_R2 ATE = 33.7629421029m
    WINGAM_P0_R3 ATE = 34.1902782732m

Current best deployable online TTT write remains:
    C9_P0_R2
    ATE = 33.7629421029m
```

---

## 1. Executed Matrix

有效落盘：

```text
Smoke:
    V22_SMOKE_R1_KVC_TTT_01... chunk10 h3

Phase A support compact/semantic:
    48/48 rows completed

Phase B read-only compact:
    16/16 clean rows completed after invalid run exclusion

Phase C semantic role:
    8/8 rows completed

Phase D attribution:
    1/1 row completed with saved HMC/merge states

Phase E skip-aware TTT:
    8/8 rows completed

Phase F skip-aware memory:
    8/8 rows completed

Phase G durable commit:
    8/8 rows completed

Phase H lifecycle:
    8/8 rows completed
```

所有这些 row 都是 trusted short-rollout diagnostic：

```text
diagnostic_only_short_rollout = true
counts_as_online_ttt_write_success = false
selector_allowed = false unless gate passes
full_online_validation_allowed = false unless gate passes
```

---

## 2. Phase A：Support + Compact/Semantic

输出：

```text
phaseA_support_report/
rollouts/V22_A_SUPPORT_R1_*
```

Gate summary：

| Metric | Best |
|---|---:|
| Best h10/h15 ATE delta vs H9 | `-0.8738393532` |
| Best ATE candidate | `SUP_LOCKED_C_STRUCTURE_RESCUE_COMPACT`, chunk `6`, h`15` |
| Best `[200,300)` delta vs H9 | `-3.6277354286` |
| Best `[200,300)` candidate | `SUP_LOCKED_C_STRUCTURE_RESCUE_COMPACT`, chunk `10`, h`10` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Key rows：

| Candidate | Chunk | h10 ATE delta | h15 ATE delta | h10 `[200,300)` | h15 `[200,300)` | h10 `[400,600)` | h15 `[400,600)` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SUP_LOCKED_C_STRUCTURE_RESCUE_COMPACT` | `6` | `-0.6236033343` | `-0.8738393532` | `-1.2422956662` | `-1.5361619675` | `+0.7799961922` | `-0.9595408243` |
| `SUP_LOCKED_C_STRUCTURE_RESCUE_COMPACT` | `10` | `-0.8190672795` | `+0.7361069995` | `-3.6277354286` | `-1.2015249497` | `-0.6624327907` | `+0.8258603352` |

Decision：

```text
Phase A gate = fail
Support variants did not produce a durable gate-passing candidate.
full_chunk / no-overlap did not unlock the target path.
No selector/full online validation allowed.
```

---

## 3. Phase B：Read-Only Compact K/V

输出：

```text
phaseB_read_only_report/
rollouts/V22_B_READ_R1_*
```

有效性审计：

```text
The clean rerun has real compact_kv:
    context_source_skip_impl = compact_kv
    context_source_skip_requested = true
    context_source_skip_mask = dg_q80
    num_context_source_skip_applied = 6
    max_context_source_skip_tokens = 8026
    mean_context_source_keep_ratio ~= 0.933
    num_context_empty_source_events = 0
```

Gate summary：

| Metric | Best |
|---|---:|
| Best h10/h15 ATE delta vs H9 | `-0.3904989913` |
| Best ATE candidate | `KVC_READ_04`, chunk `6`, h`10` |
| Best `[200,300)` delta vs H9 | `-1.7896792460` |
| Best `[200,300)` candidate | `KVC_READ_04`, chunk `10`, h`10` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Decision：

```text
Phase B read-only compact gate = fail
Read-only compact_kv has real but weak local effect.
It does not solve h15 durability.
```

---

## 4. Phase C：Semantic Role

输出：

```text
phaseC_semantic_role_report/
rollouts/V22_C_SEM_R1_*
```

Gate summary：

| Metric | Best |
|---|---:|
| Best h10/h15 ATE delta vs H9 | `-0.8738393532` |
| Best ATE candidate | `SEM_ROLE_01_STRUCTURE_RESCUE`, chunk `6`, h`15` |
| Best `[200,300)` delta vs H9 | `-3.6277354286` |
| Best `[200,300)` candidate | `SEM_ROLE_01_STRUCTURE_RESCUE`, chunk `10`, h`10` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Key rows：

| Candidate | Chunk | h10 ATE delta | h15 ATE delta | h10 `[200,300)` | h15 `[200,300)` | h15 `[400,600)` |
|---|---:|---:|---:|---:|---:|---:|
| `SEM_ROLE_01_STRUCTURE_RESCUE` | `6` | `-0.6236033343` | `-0.8738393532` | `-1.2422956662` | `-1.5361619675` | `-0.9595408243` |
| `SEM_ROLE_01_STRUCTURE_RESCUE` | `10` | `-0.8190672795` | `+0.7361069995` | `-3.6277354286` | `-1.2015249497` | `+0.8258603352` |
| `SEM_ROLE_02_LOWSTUFF_HIGHD_SKIP` | `10` | `-0.6164115009` | `+1.0386884981` | `-2.9123071925` | `-0.2265022849` | `+1.0832839725` |

Decision：

```text
Phase C gate = fail
Semantic structure rescue helps local h10, and chunk6 has a small h15 improvement.
But no row reaches:
    h10/h15 ATE delta <= -3m
    or [200,300) delta <= -5m
No selector/full online validation allowed.
```

---

## 5. Phase D：State Attribution

输出：

```text
phaseD_state_attribution/
rollouts/V22_D_ATTR_R1_TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT_chunk10_h15_globalgate_H9parent_SWKS3/
```

State overwrite summary：

| State | Group | base->h10 norm | h10->h15 norm | overwrite ratio |
|---|---|---:|---:|---:|
| HMC | all | `13856.9394083647` | `12098.5351008558` | `0.8731029807` |
| HMC | ttt | `13856.8345862250` | `12098.4158468734` | `0.8731009793` |
| merge | all | `69169.3855780770` | `39109.4837210925` | `0.5654160926` |

Interpretation：

```text
The v21 strongest h10 correction is followed by large h10->h15 state movement.
HMC TTT state movement after h10 is about 87.3% of the base->h10 movement.
Merge/gauge cursor movement after h10 is about 56.5% of base->h10 movement.

This supports the durability diagnosis:
    h10 improvement is not written as a stable long-horizon trajectory correction.
    later TTT/merge state updates can wash out the local read/source correction.
```

---

## 6. Phase E/F：Skip-Aware TTT and Skip-Aware Memory

Outputs：

```text
phaseE_skip_aware_ttt_report/
phaseF_skip_aware_memory_report/
rollouts/V22_E_TTT_R1_*
rollouts/V22_F_MEM_R1_*
```

Best rows：

| Phase | Candidate | h10 ATE delta | h15 ATE delta | h10 `[200,300)` | h15 `[200,300)` | h10 `[400,600)` | h15 `[400,600)` |
|---|---|---:|---:|---:|---:|---:|---:|
| E | `KVC_TTT_04_SOURCE_KEEP_GATED_WRITE` | `-0.7324282608` | `+0.8391261607` | `-3.4472737923` | `-0.9390895264` | `-0.5684630140` | `+0.9205697333` |
| F | `KVC_MEM_03_GLOBAL_CHUNK_SOURCE_SKIP` | `-0.7353211000` | `+0.5674474093` | `-3.4871305431` | `-1.4384892706` | `-0.6100760362` | `+0.6502941502` |

Decision：

```text
Phase E gate = fail
Phase F gate = fail

Skip-aware commit / SWA / global source policy improves h10 locally,
but h15 ATE regresses relative to H9.
No selector/full online validation allowed.
```

---

## 7. Phase G/H：Durable Commit and Lifecycle Split

Outputs：

```text
phaseG_ttt_durable_commit_report/
phaseH_ttt_lifecycle_report/
rollouts/V22_G_DUR_R1_*
rollouts/V22_H_LIFE_R1_*
```

Phase G best：

| Candidate | h10 ATE delta | h15 ATE delta | h10 `[200,300)` | h15 `[200,300)` | h15 `[400,600)` |
|---|---:|---:|---:|---:|---:|
| `TTT_DUR_04_POST_ZP_SKIP_BASIS_ROUTING` | `-0.7324282608` | `+0.8391261607` | `-3.4472737923` | `-0.9390895264` | `+0.9205697333` |

Phase H best：

| Candidate | h10 ATE delta | h15 ATE delta | h10 `[200,300)` | h15 `[200,300)` | h10 `[400,600)` | h15 `[400,600)` | Durability |
|---|---:|---:|---:|---:|---:|---:|---:|
| `TTT_LIFE_04_SCALE_LONG_HIGHD_SHORT` | `-1.8034599870` | `+0.2915878358` | `-4.2566491136` | `-1.6753970509` | `-2.2983352023` | `-0.3669997235` | `0.1616824528` |

Decision：

```text
Phase G gate = fail
Phase H status = weak, but gate = fail

TTT_LIFE_04 is the strongest v22 h10 signal:
    h10 ATE delta = -1.8034599870m
    h10 [200,300) delta = -4.2566491136m

It still fails both required entry conditions:
    h10/h15 ATE delta <= -3m is not met
    [200,300) delta <= -5m is not met
    h15 durability ratio is only 0.1616824528 < 0.45

No no-GT selector was started.
No full online validation was launched.
```

---

## 8. H5 Online Trajectory/Scale-State Branch

H5 was not counted as executed.

Reason：

```text
The current pipeline already has an older online_scale_state overlap-step mode.
However, the v22 H5 plan specifies new online trajectory/scale-state variants:
    SCALE_ONLINE_01: pose-step EMA only
    SCALE_ONLINE_02: pose-step EMA + TTT conflict energy
    SCALE_ONLINE_03: pose-step EMA + semantic structure coverage
    SCALE_ONLINE_04: pose-step EMA + context skip keep ratio

These four v22 H5 variants are not implemented as separate runtime modules in this run.
Therefore no H5 full-online row is reported and no H5 result is claimed.
```

Audit boundary：

```text
Do not reinterpret old overlap-step diagnostics as v22 H5 success.
Do not count H5 as TTT write success.
The correct next engineering step is to implement explicit online trajectory/scale-state modules
before any H5 full-online validation can be audited.
```

---

## 9. Downstream Phase Decision

| Phase | Status | Reason |
|---|---|---|
| v16 boundary reuse | pass | H9/C9/WINGAM trusted references reused |
| v22 smoke | pass as instrumentation only | compact / skip-aware path runs; not a deployable result |
| Phase A support | fail | best h15 ATE delta `-0.874m`, below `-3m`; best `[200,300)` `-3.628m`, below `-5m` |
| Phase B read-only compact | fail | best ATE delta `-0.390m`; weak local signal only |
| Phase C semantic role | fail | small h15 chunk6 improvement, but below gate |
| Phase D attribution | diagnostic | HMC h10->h15 overwrite ratio `0.873`; merge ratio `0.565` |
| Phase E skip-aware TTT | fail | h10 improves; h15 ATE regresses |
| Phase F skip-aware memory | fail | h10 improves; h15 ATE regresses |
| Phase G durable commit | fail | no durable gate-passing candidate |
| Phase H lifecycle | weak / fail | best h10 signal in v22, but h15 durability only `0.162` |
| H5 online trajectory-state | not executed | v22-specific modules not implemented in this run |
| No-GT selector | not started | durability / local gate not met |
| Full online validation | not started | selector/full-run entry forbidden |

Boundary：

```text
No v22 short-rollout result counts as deployable TTT write success.
No GT-selected candidate is counted.
No no-GT selector was evaluated.
No full online validation was launched.
No online Target-25 result was produced in v22.

Current best deployable online TTT write remains:
    C9_P0_R2
    ATE = 33.7629421029m
```

---

## 10. Final Decision

v22 的真实成功点：

```text
1. v22 durability-first matrix was executed through support, compact, semantic, TTT, SWA, durable-commit, and lifecycle branches.
2. The read-only compact_kv blocker was found and fixed; clean rerun confirms compact_kv actually applied.
3. State attribution was implemented and shows large h10->h15 HMC/merge movement:
       HMC all overwrite ratio   = 0.8731029807
       HMC TTT overwrite ratio   = 0.8731009793
       merge overwrite ratio     = 0.5654160926
4. The strongest v22 diagnostic improved over simple read-only compact:
       TTT_LIFE_04 h10 ATE delta       = -1.8034599870m
       TTT_LIFE_04 h10 [200,300) delta = -4.2566491136m
```

v22 的关键负结果：

```text
No v22 candidate passed the durability/full-run entry gate.

Best h10 local result:
    TTT_LIFE_04_SCALE_LONG_HIGHD_SHORT
    h10 ATE delta = -1.8034599870m
    h10 [200,300) delta = -4.2566491136m

But h15 durability is weak:
    h15 ATE delta = +0.2915878358m
    durability ratio = 0.1616824528
    required durability ratio >= 0.45
```

Interpretation：

```text
Context source skip, semantic rescue, skip-aware commit, SWA/global source controls,
and lifecycle split all create some short-window correction.

They still do not create a persistent trajectory correction.
The strongest evidence is the attribution result:
    later HMC/TTT state movement after h10 remains almost as large as the movement that created h10 correction.

Therefore v22 supports the plan's warning:
    current memory-source filtering and TTT write interface are useful stabilizers,
    but they are not enough by themselves for Target-25.
```

Next required direction：

```text
Do not start selector/full online validation from v22 candidates.
Do not keep micro-sweeping thresholds/beta/gamma without a new durable mechanism.

Implement explicit online trajectory-state / scale-state modules for H5:
    pose-step EMA state
    TTT-conflict-conditioned scale state
    semantic-structure-conditioned scale state
    context-keep-ratio-conditioned scale state

Only after an H5 or new TTT mechanism satisfies:
    h10/h15 ATE delta <= -3m,
    or [200,300) delta <= -5m with [400,600) regression <= +1m,
    and durability >= 0.45,
may no-GT selector and full online validation start.
```
