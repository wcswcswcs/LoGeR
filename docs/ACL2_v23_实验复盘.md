# ACL2 v23 实验复盘：SemanticPrior AllMemory Durable Target25

日期：2026-05-21（Asia/Singapore）  
计划文件：`docs/ACL2_v23_SemanticPrior_AllMemory_Durable_Target25_Experiment_Plan_with_Codex_Audit.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v23_semanticprior_allmemory_durable_target25/`

本轮原则：只记录实际落盘结果；不把静态代码接线、自查脚本、smoke、short rollout、failed gate、污染目录、或未启动矩阵写成 deployable online success。没有通过 semantic all-memory implementation self-audit 和后续 durability gate 时，不启动 no-GT selector，也不启动 full online validation。

---

## 0. 当前结论

v23 已按当前计划执行到 stop rule，未达到 Target-25。

已完成：

```text
1. 阅读 v23 计划，并按 Codex self-audit hard gate 先做实现自查。
2. 实现 semantic role / all-memory 控制链路的静态接线。
3. 新增 v23 launcher / matrix / self-audit / report 脚本。
4. 修复 dynamic smoke 前置 blocker。
5. Phase 0 R3 no-op / pass-through / debug-only smoke 通过：
       ATE_delta_vs_H9 = 0
       raw_trans_max_diff = 0
6. implementation self-audit 当前 all_gate_pass = true。
```

最终边界：

```text
1. Phase 2 single-path matrix completed: 36/36 rows.
2. Phase 3 all-memory matrix completed: 36/36 rows.
3. Phase 3 best all-memory h15 ATE delta = -0.9514682517m.
4. Phase 3 best [200,300) delta = +0.8706640678m.
5. No candidate passed v23 durability/full-run entry gate.
6. No no-GT selector was started.
7. No full online validation was launched.
8. No online Target-25 result was produced.
```

当前 best deployable online TTT write 仍然是 v16/v21/v22 记录的：

```text
C9_P0_R2
ATE = 33.7629421029m
```

---

## 1. 工程修改

新增 / 修改：

```text
loger/pipeline/semantic_prior_generator.py:
    added semantic role constants:
        FALLBACK
        POSITIVE_LONG
        NEUTRAL_KEEP
        NEGATIVE_SHORT
        PROTECT_NEUTRAL
    PriorOutput added:
        V_sem_tok
        R_sem_tok
        R_sem_patch_flat
    project_masklet_semantic_groups now emits R_sem_patch_flat / R_sem_tok
    pass-through semantic prior can emit token-aligned semantic group/role streams

loger/pipeline/hybrid_memory_controller.py:
    HybridMemoryControlPrior added:
        V_sem_tok
        R_sem_tok
    added semantic role controls:
        semantic_role_policy
        semantic_memory_paths
        semantic_role_highd_quantile
        semantic_role_low_trust
        semantic_role_positive_scale
        semantic_role_neutral_scale
        semantic_role_negative_scale
        semantic_role_swa_negative_scale
    added _apply_semantic_role_policy
    semantic role can route to:
        frame/global source debug
        SWA write score path
        TTT write prior path
        lifecycle debug path
    model hmc_control now forwards V_sem_tok / R_sem_tok

loger/models/pi3.py:
    context_source_skip_mask added:
        semantic_role_negative
        semantic_role_source_skip
        semrole_negative
    source K/V skip can consume R_sem_tok
    negative-short role is skipped only when also high-D
    positive/protected roles are protected as source tokens

run_pipeline_abc_v2.py:
    added CLI args for semantic role policy and memory paths
    passes semantic role args into HybridMemoryController
    added semantic_role_summary.jsonl
    added semantic_memory_path_summary.jsonl
    pass-through prior now carries R_sem_tok / V_sem_tok
    fixed read_path_only semantic-prior generation:
        read_path_only now builds prior when semantic role/source skip is requested

tools/run_attention_cue_experiment.sh:
    forwards SEMANTIC_ROLE_* env vars into Python CLI
    readonly and hybrid modes both receive semantic role args
    supports LOGER_CHECKPOINT / LOGER_CONFIG overrides for warm-path runs

tools/v23_implementation_self_audit.py:
    static + dynamic implementation audit
    dynamic semantic role check now requires non-empty role counts
    dynamic semantic memory path check now requires non-empty group/role metrics
    writes:
        implementation_audit/codex_self_check_report.md
        implementation_audit/codex_self_check_summary.json
        implementation_audit/codex_self_check_failures.jsonl

tools/run_v23_candidate_rollout.sh:
    v23 trusted short-rollout launcher
    supports Phase 0 smoke, Phase 2 single-path, Phase 3 all-memory candidates
    stale run dirs are moved to .INVALID_RERUN_* before forced rerun
    fixed P0_02 to readonly mode so pass-through consumed smoke is no-op

tools/run_v23_matrix.sh:
    v23 matrix scheduler:
        phase0
        phase2
        phase3
        phase4_smoke

tools/v23_candidate_bank_report.py:
    v23 candidate family labels layered on v22/v18 reporting code
```

---

## 2. 验证

静态验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    run_pipeline_abc_v2.py \
    loger/pipeline/semantic_prior_generator.py \
    loger/pipeline/hybrid_memory_controller.py \
    loger/pipeline/ttt_write_controller.py \
    loger/models/pi3.py \
    loger/models/layers/attention.py \
    tools/v23_implementation_self_audit.py \
    tools/v23_candidate_bank_report.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v23_candidate_rollout.sh
bash -n tools/run_v23_matrix.sh

PASS
```

Self-audit 输出：

```text
results/kitti01_hmc_v2/acl2_v23_semanticprior_allmemory_durable_target25/
    implementation_audit/codex_self_check_report.md
    implementation_audit/codex_self_check_summary.json
    implementation_audit/codex_self_check_failures.jsonl
```

Self-audit 当前结果：

| Gate | Result |
|---|---|
| hard_static_gate_pass | `true` |
| dynamic_smoke_gate_pass | `true` |
| all_gate_pass | `true` |

Dynamic evidence：

```text
semantic_role_summary.jsonl:
    non-empty role_counts found in Phase 0 R3 P0_02/P0_03

semantic_memory_path_summary.jsonl:
    non-empty semantic_group_role_metrics found in Phase 0 R3 P0_02/P0_03

context_skip_summary.jsonl:
    present in Phase 0 R3 rollouts
```

注意：

```text
Phase 0 R3 uses debug_only/no-op style rows.
It proves role projection/logging and no-op parity, not performance improvement.
single-path/all-memory consumed-path correctness still needs Phase 2/path smoke evidence.
```

---

## 3. Blocker 与修复记录

### Blocker 1：dynamic smoke 卡在 I/O page-in，没有进入 CUDA

现象：

```text
RUN_PREFIX=V23_P0_SMOKE_R1
candidate = P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY
chunk = 10
horizon = 3
mode = readonly
GPU argument = 0

nvidia-smi --query-compute-apps:
    no compute app rows

process status:
    STAT = Dl
    WCHAN = wait_on_page_bit_common
    RSS ~= 8.9GB

01.log:
    0 bytes
```

原因判断：

```text
launcher 已经进入 run_pipeline_abc_v2.py 前置初始化，
但大 checkpoint / HMC snapshot / merge snapshot page-in 卡住，尚未到 CUDA context/model forward。
所以显存没有变化不是 CPU 跑完，而是还没进入 GPU forward。
```

相关大文件：

```text
ckpts/LoGeR/latest.pt = 4.7GB
H9 chunk10 HMC snapshot = 503MB
H9 chunk10 merge snapshot = 211MB
```

修复：

```text
将 checkpoint/config/chunk10 HMC/merge snapshot 复制到 /tmp/loger_v23_warm：
    /tmp/loger_v23_warm/ckpts/latest.pt
    /tmp/loger_v23_warm/ckpts/original_config.yaml
    /tmp/loger_v23_warm/snapshots/chunk_010_input.pt
    /tmp/loger_v23_warm/merge_snapshots/chunk_010_input.pt

运行时通过 LOGER_CHECKPOINT / LOGER_CONFIG / LOAD_HMC_STATE_AT_CHUNK /
LOAD_MERGE_STATE_AT_CHUNK 指向 warm-path 文件。
```

结果：

```text
R2/R3 smoke 进入 GPU forward。
GPU 0 显存使用约 14GB。
Phase 0 R3 四条 h3 row 均 DONE。
```

污染目录处理：

```text
未完成 R1 smoke 目录已移动为 INVALID_IO_BLOCKER_*，不纳入报告。
```

### Blocker 2：P0_02 pass-through consumed smoke 出现非 no-op 漂移

R2 结果：

```text
P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED:
    ATE_delta_vs_H9 = +0.0219249690m
    raw_trans_max_diff = 0.0093258056
```

原因：

```text
P0_02 被配置为 RUN_MODE=hybrid。
hybrid 路径启用了 probe_ttt_write / HMC commit，因此它不是纯 pass-through no-op smoke。
```

修复：

```text
tools/run_v23_candidate_rollout.sh:
    P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED 改为 RUN_MODE=readonly。
```

结果：

```text
Phase 0 R3 P0_02:
    ATE_delta_vs_H9 = 0.0
    raw_trans_max_diff = 0.0
```

### Blocker 3：read_path_only 下 semantic role 没有真正生成 prior

现象：

```text
早期 read_path_only semantic/debug row 中：
    semantic_role_available = false
    role_counts = {}
```

原因：

```text
run_pipeline_abc_v2.py 只在 ttt_write_only/hybrid 时构建 semantic prior。
read_path_only 只请求 semantic role/source skip 时没有进入 prior 构建分支。
```

修复：

```text
run_pipeline_abc_v2.py:
    added semantic_role_requested
    needs_prior now includes:
        mode == read_path_only and semantic_role_requested
```

结果：

```text
Phase 0 R3 P0_02 / P0_03 均写出非空 semantic_role_summary.jsonl。
```

---

## 4. Phase 0：Plumbing / No-op Gate

输出：

```text
phase0_plumbing_report_R3/
rollouts/V23_P0_SMOKE_R3_*
```

矩阵：

```text
K1_H9
P0_01_SEMANTIC_ROLE_NOOP_IGNORED
P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED
P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY

chunk = 10
horizon = 3
rows completed = 4/4
```

关键结果：

| Candidate | ATE delta vs H9 | `[200,300)` delta vs H9 | raw trans max diff | HMC hash mismatch |
|---|---:|---:|---:|---:|
| `K1_H9` | `0.0` | `0.0` | `0.0` | `0` |
| `P0_01_SEMANTIC_ROLE_NOOP_IGNORED` | `0.0` | `0.0` | `0.0` | `0` |
| `P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED` | `0.0` | `0.0` | `0.0` | `0` |
| `P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY` | `0.0` | `0.0` | `0.0` | `0` |

Semantic role evidence, P0_03 first chunk summary：

```text
semantic_role_available = true
semantic_role_policy = debug_only
semantic_memory_paths = all
token_count = 40320
role_counts:
    0 = 395
    1 = 10395
    2 = 21561
    3 = 6144
    4 = 1825
Q_sem_mean = 0.9801945686
V_sem_mean = 0.5887824893
D_mean = 0.2200119793
D_p90 = 0.7494726777
```

Decision：

```text
Phase 0 no-op/plumbing gate = pass
v23 implementation self-audit = pass
Phase 2 single-path smoke/performance matrix is allowed.
No deployable online result is claimed.
No selector/full online validation is allowed yet.
```

---

## 5. Path Consumption / Passive Semantic Audit

输出：

```text
phase1_passive_semantic_audit/
rollouts/V23_PATH_SMOKE_R1_*
```

Path smoke：

```text
FRAME_SEM_01_STRUCTURE_KEEP                 chunk10 h3
GLOBAL_SEM_01_STRUCTURE_KEEP                chunk10 h3
SWA_SEM_01_STRUCTURE_LONG_KEEP              chunk10 h3
TTT_SEM_01_STRUCTURE_POSITIVE               chunk10 h3
ALLSEM_05_FRAME_GLOBAL_SWA_TTT_ALL_ROLE     chunk10 h3
rows completed = 5/5
GPU used = true
```

Consumed-path evidence：

| Smoke | frame | global | SWA | TTT | context skip applied | empty source |
|---|---|---|---|---|---:|---:|
| `FRAME_SEM_01_STRUCTURE_KEEP` | `true` | `false` | `false` | `false` | `24` | `0` |
| `GLOBAL_SEM_01_STRUCTURE_KEEP` | `false` | `true` | `false` | `false` | `24` | `0` |
| `SWA_SEM_01_STRUCTURE_LONG_KEEP` | `false` | `false` | `true` | `false` | `0` | `0` |
| `TTT_SEM_01_STRUCTURE_POSITIVE` | `false` | `false` | `false` | `true` | `0` | `0` |
| `ALLSEM_05_FRAME_GLOBAL_SWA_TTT_ALL_ROLE` | `true` | `true` | `true` | `true` | `48` | `0` |

Passive audit aggregate：

```text
num_semantic_runs = 77
num_done_semantic_runs = 77
consumed_any_runs = 77
frame_consumed_runs = 50
global_consumed_runs = 44
swa_consumed_runs = 26
ttt_consumed_runs = 32
context_skip_requested_runs = 57
context_skip_empty_source_event_runs = 0
```

Decision：

```text
Semantic role is not only logged; it is consumed by frame/global/SWA/TTT paths.
compact_kv source skip was applied where requested.
No empty source event was observed.
This permits Phase 2 and Phase 3 performance matrices.
```

---

## 6. Phase 2：Semantic Role Single-Path Ablation

输出：

```text
phase2_singlepath_report/
rollouts/V23_P2_SINGLEPATH_R1_*
matrix_logs/phase2_singlepath_R1/
```

矩阵：

```text
candidates:
    FRAME_SEM_01_STRUCTURE_KEEP
    FRAME_SEM_02_LOWSTUFF_HIGHD_SKIP
    GLOBAL_SEM_01_STRUCTURE_KEEP
    SWA_SEM_01_STRUCTURE_LONG_KEEP
    TTT_SEM_01_STRUCTURE_POSITIVE
    TTT_SEM_02_LOWSTUFF_HIGHD_SHORT_NEG

chunks: 6, 10, 16
horizons: 10, 15
rows completed: 36/36
failures: 0
```

Gate summary：

| Metric | Best |
|---|---:|
| Best h10/h15 ATE delta vs H9 | `-1.0299370930` |
| Best ATE candidate | `FRAME_SEM_01_STRUCTURE_KEEP`, chunk `16`, h`15` |
| Best `[200,300)` delta vs H9 | `-0.8600984785` |
| Best `[200,300)` candidate | `GLOBAL_SEM_01_STRUCTURE_KEEP`, chunk `10`, h`10` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Key rows：

| Candidate | Chunk | h10 ATE delta | h15 ATE delta | h10 `[200,300)` | h15 `[200,300)` | h15 `[400,600)` |
|---|---:|---:|---:|---:|---:|---:|
| `FRAME_SEM_01_STRUCTURE_KEEP` | `16` | `-0.1624956227` | `-1.0299370930` | `nan` | `nan` | `-1.2256103362` |
| `FRAME_SEM_02_LOWSTUFF_HIGHD_SKIP` | `16` | `-0.1624956227` | `-1.0299370930` | `nan` | `nan` | `-1.2256103362` |
| `TTT_SEM_01_STRUCTURE_POSITIVE` | `16` | `-0.1781913684` | `-1.0165732576` | `nan` | `nan` | `-1.2321567052` |
| `GLOBAL_SEM_01_STRUCTURE_KEEP` | `10` | `-0.0971426695` | `+1.3561304336` | `-0.8600984785` | `+1.6144458745` | `+0.8232870931` |

Decision：

```text
Phase 2 does not pass full-run entry.
It has a weak single-path h15 signal at chunk16:
    h15 ATE delta ~= -1.03m
    downstream [400,600) improves, not regresses.

This satisfies the Phase 2 continuation criterion:
    h15 ATE delta <= -1m without downstream regression.

Therefore Phase 3 all-memory combination was allowed.
No selector/full online validation was allowed from Phase 2.
```

---

## 7. Phase 3：Semantic All-Memory Role Combination

输出：

```text
phase3_allmem_report/
rollouts/V23_P3_ALLMEM_R1_*
matrix_logs/phase3_allmem_R1/
```

矩阵：

```text
candidates:
    ALLSEM_01_FRAME_GLOBAL_STRUCTURE_KEEP
    ALLSEM_02_FRAME_GLOBAL_LOWSTUFF_HIGHD_SKIP
    ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP
    ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG
    ALLSEM_05_FRAME_GLOBAL_SWA_TTT_ALL_ROLE
    ALLSEM_06_ALL_ROLE_LONG_SHORT

chunks: 6, 10, 16
horizons: 10, 15
rows completed: 36/36
failures: 0
```

Gate summary：

| Metric | Best |
|---|---:|
| Best h10/h15 ATE delta vs H9 | `-0.9514682517` |
| Best ATE candidate | `ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG`, chunk `16`, h`15` |
| Best `[200,300)` delta vs H9 | `+0.8706640678` |
| Best `[200,300)` candidate | `ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG`, chunk `10`, h`10` |
| Durability gate pass | `false` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Best by chunk / horizon：

| Chunk | Horizon | Best ATE candidate | ATE delta vs H9 | Best `[200,300)` candidate | `[200,300)` delta |
|---:|---:|---|---:|---|---:|
| `6` | `10` | `ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP` | `+0.4820243174` | `ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP` | `+1.1604995445` |
| `6` | `15` | `ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP` | `+0.7393667065` | `ALLSEM_03_FRAME_GLOBAL_SWA_STRUCTURE_LONG_KEEP` | `+1.4124559014` |
| `10` | `10` | `ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG` | `+0.0852638560` | `ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG` | `+0.8706640678` |
| `10` | `15` | `ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG` | `+0.2982347683` | `ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG` | `+1.0487306072` |
| `16` | `10` | `ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG` | `-0.1631425799` | `ALLSEM_01_FRAME_GLOBAL_STRUCTURE_KEEP` | `nan` |
| `16` | `15` | `ALLSEM_04_FRAME_GLOBAL_TTT_POSNEG` | `-0.9514682517` | `ALLSEM_01_FRAME_GLOBAL_STRUCTURE_KEEP` | `nan` |

Decision：

```text
Phase 3 all-memory gate = fail.

Required:
    h10 [200,300) delta <= -5m
    h15 ATE delta <= -3m or h15 [200,300) delta <= -5m
    durability_ratio >= 0.45
    [400,600) regression <= +1m

Observed:
    best h15 ATE delta = -0.951m
    best [200,300) delta = +0.871m
    no candidate passed local/durability gate

No no-GT selector was started.
No full online validation was launched.
```

---

## 8. Phase 4 / Selector / Full Online Decision

Phase 4 attribution was not launched as a new v23 targeted repair stage.

Reason：

```text
Phase 4 is triggered when Phase 3 has strong h10 but weak h15,
so attribution can locate the overwrite source.

v23 Phase 3 did not produce strong h10:
    best [200,300) delta = +0.8706640678m
    best chunk10 h10 ATE delta = +0.0852638560m

Therefore this is not a "h10 strong, h15 washed out" case.
It is a "all-memory semantic role did not create a strong local or durable correction" case.
```

Downstream decision：

| Stage | Status | Reason |
|---|---|---|
| Phase 0 self-audit / no-op | pass | no drift, non-empty semantic role summary |
| Path consumption smoke | pass | frame/global/SWA/TTT consumed flags verified |
| Phase 2 single-path | weak | chunk16 h15 ATE about `-1.03m`, allowed Phase 3 |
| Phase 3 all-memory | fail | no h10/h15 gate-passing candidate |
| Phase 4 attribution | not started | no strong h10 candidate to attribute |
| No-GT selector | not started | durability/full-run entry gate failed |
| Full online validation | not started | selector/full-run entry forbidden |

Boundary：

```text
No v23 short-rollout result counts as deployable online TTT write success.
No GT-selected candidate is counted.
No no-GT selector was evaluated.
No full online validation was launched.
No online Target-25 result was produced in v23.

Current best deployable online TTT write remains:
    C9_P0_R2
    ATE = 33.7629421029m
```

---

## 9. Final Decision

v23 的真实成功点：

```text
1. Semantic role streams are implemented:
       G_sem_tok / Q_sem_tok / V_sem_tok / R_sem_tok
2. Semantic role projection and logging passed dynamic smoke.
3. No-op / pass-through consumed smoke no longer changes output:
       Phase 0 R3 ATE delta = 0
       raw_trans_max_diff = 0
4. Path consumption was verified:
       frame/global/SWA/TTT each consumed role in its own smoke
       ALLSEM_05 consumed all four paths
       context skip empty source events = 0
5. Phase 2 and Phase 3 matrices completed without failed rows:
       Phase 2: 36/36
       Phase 3: 36/36
```

v23 的关键负结果：

```text
Semantic all-memory role did not become a durable trajectory correction.

Phase 2 best:
    FRAME_SEM_01 chunk16 h15 ATE delta = -1.0299370930m

Phase 3 best:
    ALLSEM_04 chunk16 h15 ATE delta = -0.9514682517m

Phase 3 disease-window result regressed:
    best [200,300) delta = +0.8706640678m

No candidate met:
    h10 [200,300) delta <= -5m
    h15 ATE delta <= -3m
    h15 [200,300) delta <= -5m
    durability/full-run entry gate
```

Interpretation：

```text
v23 validates the engineering path:
    semantic role can be projected,
    forwarded,
    consumed by frame/global/SWA/TTT,
    and audited without no-op drift.

But the scientific hypothesis is not supported strongly enough:
    all-memory semantic role policy did not convert short-window correction
    into h15 durable correction.

The best effect appears as a small chunk16 h15 stabilizer, not as Target-25-scale correction.
The [200,300) disease window did not improve in Phase 3.
```

Conclusion type：

```text
Conclusion C:
    semantic memory is not sufficient to support Target-25 in this formulation.

Semantic role should be retained as a regularizer / source reliability signal,
but Target-25 mainline should move to explicit online trajectory-state / scale-state modules.
```

Next required direction：

```text
Do not start selector/full online validation from v23 candidates.
Do not micro-sweep semantic role thresholds from this family.

Use semantic role as auxiliary reliability for:
    explicit online trajectory-state,
    explicit online scale-state,
    merge/gauge-aware state correction,
    or a new durable mechanism that changes trajectory state directly.

Any future full online Target-25 validation must still pass:
    h15 ATE delta <= -3m,
    or h15 [200,300) delta <= -5m,
    durability_ratio >= 0.45,
    [400,600) regression <= +1m,
    and a no-GT selector gate.
```
