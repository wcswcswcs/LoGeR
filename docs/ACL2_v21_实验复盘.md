# ACL2 v21 实验复盘：ContextSkip SemanticAllMemory TTT Persistence Target25

日期：2026-05-21（Asia/Singapore）  
计划文件：`docs/ACL2_v21_ContextSkip_SemanticAllMemory_TTT_Persistence_Target25_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v21_contextskip_semanticallmemory_ttt_persistence_target25/`

本轮原则：只记录实际落盘结果；不把 short rollout、sandbox oracle、GT audit、proxy semantic mask、instrumentation smoke、failed durability gate 写成 deployable online success。没有通过 durability gate 时，不启动 no-GT selector，也不启动 full online validation。

---

## 0. 工程与配置复盘

新增 / 修改：

```text
loger/models/layers/attention.py:
    added compact_kv attention path
    Query tokens remain full length; Key/Value source tokens are compacted by source_keep_mask

loger/models/pi3.py:
    added context_source_skip_impl = bias / compact_kv
    added skip masks:
        dg_q80 / dg_q85 / dg_q90
        lowstuff_highd / sem_lowstuff_highd
        structure_rescue_dg_q80 / sem_structure_rescue_dg_q80
    fixed compact_kv dict mask handling in attention path

loger/pipeline/hybrid_memory_controller.py:
    HybridMemoryControlPrior now carries G_sem_tok / Q_sem_tok / L_sem_tok
    added true support aliases:
        full_chunk_true
        full_chunk_no_overlap
        past_plus_near_future12
        past_plus_future_light_real
    HMC forwards overlap_frames and context_source_skip_impl into model controls

loger/pipeline/semantic_prior_generator.py:
    added exact coarse semantic group projection from MaskletOutput
    taxonomy = stage_c_coarse_5_groups
    fine sky/vegetation split is not available and is not claimed

loger/pipeline/ttt_write_controller.py:
    added v19/v21 scale-state projection risk source
    added scale_state_mode / proxy / carrier / alpha / chunk controls
    added chunk-gated native mix and commit EMA controls

run_pipeline_abc_v2.py:
    added context_source_skip_impl CLI
    added context_skip_summary.jsonl and semantic_group_summary.jsonl
    added support/debug plumbing for v21
    fixed Stage C cache lookup for sliced causal fork to use global frame start/end

tools/run_attention_cue_experiment.sh:
    forwards CONTEXT_SOURCE_SKIP_IMPL

tools/v21_support_audit.py:
    support index audit for true support aliases

tools/run_v21_candidate_rollout.sh:
    v21 trusted short-rollout launcher

tools/run_v21_matrix.sh:
    v21 GPU matrix scheduler

tools/v21_candidate_bank_report.py:
    aggregate h10/h15 metrics and durability gate
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    loger/pipeline/semantic_prior_generator.py \
    loger/pipeline/hybrid_memory_controller.py \
    loger/models/pi3.py \
    loger/models/layers/attention.py \
    tools/v21_support_audit.py \
    tools/v21_candidate_bank_report.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v21_candidate_rollout.sh
bash -n tools/run_v21_matrix.sh

PASS
```

工程 blocker 与修复：

```text
blocker 1:
    first compact_kv smoke failed because attention mask became a dict and code still called .to(dtype).

fix:
    pi3.py detects compact_kv dict mask and skips tensor .to(dtype) conversion.

blocker 2:
    semantic compact candidates initially failed Stage C cache lookup:
        requested chunk_006_000000_000032/masklet.pt
    but causal fork chunks use global frames:
        chunk_006_000174_000206/masklet.pt

fix:
    run_pipeline_abc_v2.py now passes global_start/global_end into _run_stage_c_cached.
    Re-run confirmed Stage C cache hit and semantic masklets loaded.

blocker 3:
    initial Phase B matrix had one duplicate write race for KVC_01 chunk10 h15.

fix:
    contaminated duplicate directory was renamed INVALID_DUPLICATE_WRITE and excluded.
    clean KVC_01 chunk10 h15 was re-run and used in report.
```

边界说明：

```text
H9/C9/WINGAM full online boundary was not re-run in v21.
v21 uses the v16 trusted boundary and causal fork snapshots as parent/reference:
    H9_P0_R2 ATE = 34.1257769401m
    C9_P0_R2 ATE = 33.7629421029m
    WINGAM_P0_R3 ATE = 34.1902782732m

Current best deployable online TTT write remains:
    C9_P0_R2
    ATE = 33.7629421029m
```

---

## 1. Support Audit

输出：

```text
phaseA_support_audit/support_index_summary.csv
phaseA_support_audit/support_index_by_frame.jsonl
phaseA_support_audit/support_future_mass.csv
phaseA_support_audit/support_overlap_exclusion_check.csv
```

| Support | Count mean | Future ratio mean | Weighted future mass | Overlap seam support | Check |
|---|---:|---:|---:|---:|---|
| `past_only` | `15.5` | `0.0` | `0.0` | `279` | pass |
| `full_chunk_true` | `31.0` | `0.5` | `0.5` | `558` | pass |
| `full_chunk_no_overlap` | `25.1875` | `0.5` | `0.5` | `0` | pass |
| `past_plus_near_future12` | `17.40625` | `0.1902108892` | `0.1902108892` | `306` | pass |
| `past_plus_future_light_real` | `31.0` | `0.5` | `0.265625` | `558` | pass |

结论：

```text
full_chunk_no_overlap is now true no-overlap, not fallback.
past_plus_future_light_real is weighted future support, not sparse support.
past_plus_static_future_only was not implemented and was not run as if valid.
```

---

## 2. Smoke / Instrumentation

已落盘 smoke：

```text
V21_SMOKE_R2_KVC_01_FRAME_EARLY_DG_Q80_COMPACT_chunk10_h3_globalgate_H9parent_SWKS3
```

关键观测：

```text
context_source_skip_impl = compact_kv
frame attention num_context_source_skip_applied = 6
max_context_source_skip_tokens = 8026
num_context_empty_source_events = 0
mean keep ratio about 0.933
```

说明：

```text
Smoke run only verifies compact_kv path and logging.
It is not a gate success and does not count as deployable TTT write.
```

---

## 3. Phase A：Support Candidate Bank

输出：

```text
phaseA_support_report/
rollouts/V21_A_SUPPORT_R1_*
```

矩阵：

```text
candidates:
    S0_C23_PAST_LOCKED
    S1_C23_FULL_CHUNK_TRUE
    S2_C23_FULL_CHUNK_NO_OVERLAP_TRUE
    S3_C23_PAST_PLUS_NEAR_FUTURE12
    S4_C23_PAST_PLUS_FUTURE_LIGHT_REAL

chunks: 6, 10, 16
horizons: 10, 15
rows completed: 30/30
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-0.4359313833` |
| Best ATE candidate | `S0_C23_PAST_LOCKED`, chunk `6`, h`15` |
| Best `[200,300)` delta vs H9 | `-0.6936248406` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Decision：

```text
Phase A gate = fail
True support variants did not beat past_only strongly enough.
No selector/full online validation allowed.
```

---

## 4. Phase B：True K/V Compact Source Skip

输出：

```text
phaseB_kvcompact_report/
rollouts/V21_B_KVC_R1_*
```

矩阵：

```text
candidates:
    KVC_01_FRAME_EARLY_DG_Q80_COMPACT
    KVC_02_FRAME_EARLY_DG_Q90_COMPACT
    KVC_04_GLOBAL_EARLY_DG_Q80_COMPACT
    KVC_05_FRAME_GLOBAL_EARLY_DG_Q80_COMPACT
    KVC_06_FRAME_EARLY_DG_Q80_BIAS_REPEAT

chunks: 6, 10
horizons: 10, 15
rows completed: 20/20 after clean rerun
```

Gate summary：

| Metric | Best |
|---|---|
| Best h10/h15 ATE delta vs H9 | `-0.8899887250` |
| Best ATE candidate | `KVC_05_FRAME_GLOBAL_EARLY_DG_Q80_COMPACT`, chunk `6`, h`15` |
| Best `[200,300)` delta vs H9 | `-3.4871305431` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Interpretation：

```text
true K/V compaction has real local signal.
Best chunk10 h10 [200,300) improvement is about 3.49m.
It is still below the -5m segment gate and below the -3m ATE gate.
```

---

## 5. Phase B Follow-up / Phase D：Semantic Coarse Role and Static Rescue

输出：

```text
phaseB_rescue_report/
phaseD_semcoarse_report/
rollouts/V21_B_RESCUE_R1_*
rollouts/V21_D_SEMCOARSE_R1_*
```

已验证 Stage C cache 使用 exact cached masklet，示例 chunk 6：

```text
road        -> STRUCTURE_ANCHOR
fence       -> STRUCTURE_ANCHOR
grass       -> LOW_VALUE_STUFF
sky         -> LOW_VALUE_STUFF
vegetation  -> LOW_VALUE_STUFF
wall        -> STRUCTURE_ANCHOR
```

注意：

```text
This is coarse semantic grouping.
Fine sky-vs-vegetation separated exact role is not available in this cache and is not claimed.
```

矩阵：

```text
Phase B rescue candidates:
    KVC_03_FRAME_EARLY_LOWSTUFF_HIGHD_COMPACT
    KVC_08_FRAME_EARLY_DG_Q80_COMPACT_WITH_STATIC_RESCUE

Phase D semantic candidates:
    SEMFA_04_LOWSTUFF_HIGHD_FRAME_EARLY_COMPACT
    SEMFA_05_STRUCTURE_RESCUE_DGQ80_FRAME_EARLY_COMPACT

chunks: 6, 10
horizons: 10, 15
rows completed:
    Phase B rescue: 8/8
    Phase D: 8/8
```

Best rows：

| Candidate | Chunk | Horizon | ATE delta vs H9 | `[200,300)` delta vs H9 | `[400,600)` delta vs H9 |
|---|---:|---:|---:|---:|---:|
| `KVC_08_FRAME_EARLY_DG_Q80_COMPACT_WITH_STATIC_RESCUE` | `10` | `10` | `-0.8190672795` | `-3.6277354286` | `-0.6624327907` |
| `KVC_08_FRAME_EARLY_DG_Q80_COMPACT_WITH_STATIC_RESCUE` | `10` | `15` | `+0.7361069995` | `-1.2015249497` | `+0.8258603352` |
| `SEMFA_05_STRUCTURE_RESCUE_DGQ80_FRAME_EARLY_COMPACT` | `10` | `10` | `-0.8190672795` | `-3.6277354286` | `-0.6624327907` |
| `SEMFA_05_STRUCTURE_RESCUE_DGQ80_FRAME_EARLY_COMPACT` | `10` | `15` | `+0.7361069995` | `-1.2015249497` | `+0.8258603352` |

Decision：

```text
Static rescue improves local h10 signal slightly.
However h15 ATE regresses at chunk10, so durability is still poor.
No selector/full online validation allowed.
```

---

## 6. Phase E：Scale-State Compact Persistence

输出：

```text
phaseE_persistence_report/
rollouts/V21_E_PERSIST_R1_*
```

矩阵：

```text
candidates:
    TTTSSP_01_SCALECOMMIT_DGQ80_COMPACT
    TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT

chunk: 10
horizons: 10, 15
rows completed: 4/4
```

Best rows：

| Candidate | Horizon | ATE delta vs H9 | `[200,300)` delta vs H9 | `[400,600)` delta vs H9 | Runtime |
|---|---:|---:|---:|---:|---:|
| `TTTSSP_01_SCALECOMMIT_DGQ80_COMPACT` | `10` | `-2.5639878011` | `-4.9082055701` | `-3.3874144067` | `730s` |
| `TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT` | `10` | `-2.6117631916` | `-5.0388073200` | `-3.4284099296` | `797s` |
| `TTTSSP_01_SCALECOMMIT_DGQ80_COMPACT` | `15` | `-0.0885591764` | `-2.2958582956` | `-1.0543101889` | `950s` |
| `TTTSSP_02_SCALECOMMIT_DGQ80_STRUCTURE_RESCUE_COMPACT` | `15` | `-0.1846540636` | `-2.5291452036` | `-1.1370945588` | `1051s` |

Gate interpretation：

```text
Phase E short-horizon entry gate before durability:
    passed by [200,300) h10 delta = -5.0388073200m

Durability gate:
    failed

For TTTSSP_02:
    |h15 ATE delta| / |h10 ATE delta|
    = 0.1846540636 / 2.6117631916
    ~= 0.071
    required >= 0.45

selector_allowed_before_durability = true
selector_allowed = false
full_online_validation_allowed = false
```

Decision：

```text
Phase E found the strongest v21 diagnostic signal:
    TTTSSP_02 chunk10 h10
    ATE delta vs H9 = -2.6117631916m
    [200,300) delta vs H9 = -5.0388073200m

But the effect decays strongly by h15.
This is a short-horizon stabilizer, not a durable trajectory correction.
No no-GT selector was started.
No full online validation was launched.
No online Target-25 result was produced.
```

---

## 7. Downstream Phase Decision

| Phase | Status | Reason |
|---|---|---|
| Phase 0 Boundary | reused v16 | v21 did not rerun full H9/C9/WINGAM boundary |
| Support Audit | pass | true no-overlap support verified; no fallback |
| Smoke / compact_kv instrumentation | pass | compact_kv path ran with no empty source events |
| Phase A Support Bank | fail | best ATE delta `-0.436m`, below gate |
| Phase B compact_kv | fail | best `[200,300)` delta `-3.487m`, below gate |
| Phase B/D semantic rescue | fail | h10 improves, h15 chunk10 regresses |
| Phase E scale-state compact persistence | local gate only | h10 `[200,300)` `-5.039m`, durability ratio only `~0.071` |
| No-GT selector | not started | durability gate failed |
| Full online validation | not started | selector/full-run entry forbidden |

Boundary：

```text
No v21 short-rollout result counts as deployable TTT write success.
No GT-selected candidate is counted.
No no-GT selector was evaluated.
No full online validation was launched.
No online Target-25 result was produced in v21.

Current best deployable online TTT write remains:
    C9_P0_R2
    ATE = 33.7629421029m
```

---

## 8. Final Decision

v21 的真实成功点：

```text
1. true K/V source compaction is implemented and runs.
2. Stage C exact coarse semantic group routing is connected to HMC/model control.
3. full_chunk_no_overlap support is now actually no-overlap.
4. The strongest diagnostic improved over v20:
       TTTSSP_02 h10 [200,300) delta = -5.0388073200m
       TTTSSP_02 h10 ATE delta       = -2.6117631916m
```

v21 的关键负结果：

```text
The strongest v21 effect is not durable.
TTTSSP_02 h15 ATE delta is only -0.1846540636m.
Durability ratio is about 0.071, far below 0.45.

Therefore it is not a valid selector/full-online candidate.
Target-25 remains unreached.
```

Interpretation：

```text
Context source skip and semantic/static rescue reduce local disease-window pollution.
Scale-state commit turns that into a stronger h10 local stabilizer.
However, the correction still does not persist through h15.

This supports the v21 plan's concern:
    current source filtering + TTT write interface can stabilize short windows,
    but it still does not create durable trajectory-state correction.
```

Next required direction：

```text
Do not start selector/full online validation from v21 candidates.
Continue only with mechanisms that directly address durability:
    skip-aware write-probe TTT commit,
    skip-aware SWA cache commit,
    post-zp source-skip basis routing,
    or explicit online trajectory-state / scale-state module.

Any future full online Target-25 validation must still pass:
    h10/h15 ATE delta <= -3m,
    or [200,300) delta <= -5m with [400,600) regression <= +1m,
    and Durability >= 0.45,
    and a no-GT selector gate.
```
