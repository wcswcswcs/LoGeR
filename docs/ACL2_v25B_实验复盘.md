# ACL2 v25B 实验复盘：VideoMasklet SemanticPrior AllMemory Parallel

日期：2026-05-22（Asia/Singapore）  
计划文件：`docs/ACL2_v25B_VideoMasklet_SemanticPrior_AllMemory_Parallel_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v25b_videomasklet_semanticprior_allmemory_parallel/`

本轮原则：只记录实际落盘结果；不把 GT semantic 缺失时的 predicted semantic 写成 GT 实验；不把 implementation audit、smoke、short rollout、failed gate、coarse fallback fine-label 候选、或未启动矩阵写成 deployable online success。没有通过 Phase 0 implementation hard gate、Phase 2/3/4 gate、以及 no-GT selector gate 时，不启动 full online validation。

---

## 0. 当前结论

v25B 已按 speed-gated Phase 2 策略执行到 stop rule，未达到 Target-25。

已完成并落盘：

```text
1. 阅读 v25B 计划，并执行 video-masklet cache / no-op / path-consumption hard gate。
2. 新增 v25B video-masklet audit 脚本。
3. Phase 0 cache quality gate 通过。
4. Phase 0 no-op / pass-through / debug-only parity 通过：
       direct K1_H9 pose compare
       matched rows = 119
       max_translation_abs_diff = 0
       max_pose_abs_diff = 0
5. Phase 0 path consumption audit 通过：
       frame/global/SWA/TTT all consumed
       context empty source event runs = 0
6. Phase 2 h10 screen 完成：
       rows = 12/12
       failures = 0
7. Phase 2 h15 top confirmation 完成：
       rows = 5/5
       failures = 0
8. Phase 2 gate = fail。
```

最终边界：

```text
1. h10 screen best ATE delta = -0.3175006931m。
2. h10 screen best [200,300) delta = -1.9404135647m。
3. h15 top confirmation best ATE delta = -0.0735579504m。
4. h15 top confirmation best [200,300) delta = -0.1254841547m。
5. No Phase 2 candidate passed gate.
6. No Phase 3 pairwise combination was started.
7. No Phase 4 all-memory validation was started.
8. No no-GT selector was started.
9. No full online validation was launched.
10. No online Target-25 result was produced.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v25B 不是 GT semantic 实验。
semantic_source = video_masklet_frontend_cache
uses_gt_semantic = false
uses_video_masklet_semantic = true

Video-masklet cache quality and plumbing are sufficient to start short-rollout diagnostics.
但当前 runtime semantic policy 仍是 coarse-group keyed，不是真正 fine-label runtime role policy。
fine label 可用于审计/诊断，但 sky/vegetation/fence 等 fine-name policy 只能标记为 coarse fallback。
```

---

## 1. 工程修改

新增：

```text
tools/v25b_video_masklet_audit.py:
    audits v25B Stage-C video-masklet cache quality
    never treats predicted semantic as GT semantic
    writes:
        implementation_audit/codex_self_check_report.md
        implementation_audit/codex_self_check_summary.json
        implementation_audit/codex_self_check_failures.jsonl
        implementation_audit/stage_c_cache_hit_audit.csv
        implementation_audit/fine_label_coverage_by_chunk.csv

    checks:
        stage_c_cache_hit_rate
        chunks_with_masklets_ratio
        mean_coverage
        focus_coverage_200_300
        fine_label availability
        direct K1_H9 no-op pose parity when Phase 0 report is supplied
```

修改：

```text
tools/v25b_video_masklet_audit.py:
    Phase 0 no-op gate now uses direct same-batch K1_H9 pose comparison
    instead of relying only on candidate_vs_H9_delta_by_horizon.csv.

    Reason:
        candidate_vs_H9_delta_by_horizon.csv can use an external old H9 reference
        for performance reporting. In Phase 0 R1, even K1_H9 showed
        ATE_delta_vs_H9 = -0.1596122975m against that old reference.
        That is not valid evidence of semantic no-op drift.

    The audit keeps external-H9 failures as diagnostic fields:
        noop_external_H9_report_gate_pass = false
    but the hard no-op parity gate is:
        noop_gate_method = direct_k1_pose_compare
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v25b_video_masklet_audit.py

PASS
```

复用：

```text
tools/run_v24_candidate_rollout.sh
tools/run_v24_phase2_initial_queue.sh
tools/v24_candidate_bank_report.py
tools/v24_passive_semantic_audit.py

These were used with:
    V24_ROOT=results/kitti01_hmc_v2/acl2_v25b_videomasklet_semanticprior_allmemory_parallel
    V24_RESULT_ROOT=...
    RUN_PREFIX=V25B_...
```

重要边界：

```text
No true independent runtime tensors:
    R_frame_tok
    R_global_tok
    R_swa_tok
    R_ttt_tok
were implemented in v25B.

Current runtime path behavior is path-gated coarse semantic role:
    one R_sem_tok
    semantic_memory_paths controls frame/global/SWA/TTT consumption

Therefore v25B can test coarse video-masklet semantic path effects,
but cannot claim true fine-label path-specific role-controller success.
```

---

## 2. Phase 0：Cache / No-op / Path Consumption Gate

输出：

```text
implementation_audit/
phase0_plumbing_report_R1/
phase0_path_consumption_audit_R1/
rollouts/V25B_P0_SMOKE_R1_*
```

Phase 0 smoke：

```text
K1_H9
P0_01_SEMANTIC_ROLE_NOOP_IGNORED
P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED
P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY
FRAMESEM_02_LOWSTUFF_HIGHD_SKIP
ALLMEM_03_FRAME_GLOBAL_SWA_TTT_PATHSPEC

chunk = 10
horizon = 3
rows completed = 6/6
```

Cache quality：

| Metric | Value |
|---|---:|
| `cache_hit_rate` | `1.0` |
| `chunks_with_masklets_ratio` | `1.0` |
| `needed_chunk_count` | `26` |
| `mean_masklets_per_chunk` | `6.5` |
| `mean_coverage` | `0.9816746437` |
| `focus_coverage_200_300` | `0.9781515747` |
| `mean_visible_masklet_frame_frac` | `0.7254197196` |
| `fine_label_count` | `8` |
| `cache_quality_gate_pass` | `true` |

Fine labels observed in Stage-C cache audit:

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

No-op parity direct comparison:

| Candidate | Reference | matched rows | max translation diff | max pose diff |
|---|---|---:|---:|---:|
| `K1_H9` | `K1_H9` | `119` | `0.0` | `0.0` |
| `P0_01_SEMANTIC_ROLE_NOOP_IGNORED` | `K1_H9` | `119` | `0.0` | `0.0` |
| `P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED` | `K1_H9` | `119` | `0.0` | `0.0` |
| `P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY` | `K1_H9` | `119` | `0.0` | `0.0` |

Path consumption aggregate:

| Metric | Value |
|---|---:|
| `num_runs_with_semantic_memory_metrics` | `6` |
| `major_semantic_group_count` | `3` |
| `fine_label_count` | `8` |
| `frame consumed count` | `8` |
| `global consumed count` | `4` |
| `swa consumed count` | `4` |
| `ttt consumed count` | `4` |
| `context_skip_empty_source_event_runs` | `[]` |
| `phase1_gate_pass` | `true` |

Runtime note:

```text
coarse groups seen:
    0
    3
    4

coarse_group_diversity_gate_pass = false
fine_label_diversity_gate_pass = true
runtime_fine_role_policy_available = false
```

Decision:

```text
Phase 0 gate = pass.
Video-masklet semantic cache and plumbing are good enough for short-rollout diagnostics.
No deployable online result is claimed.
No selector/full online validation is allowed.
```

---

## 3. Blocker 与修复记录

### Blocker 1：Phase 0 no-op report 用旧 external H9 reference 误判失败

现象：

```text
phase0_plumbing_report_R1/candidate_vs_H9_delta_by_horizon.csv:
    K1_H9 ATE_delta_vs_H9 = -0.15961229754918715
    K1_H9 raw_trans_max_diff = 0.1447277092819479

P0_01 / P0_02 / P0_03 与 K1_H9 数值相同，
也显示相同 external-H9 delta。
```

原因：

```text
该 report 的 H9 reference 是旧 v16 H9 causal-fork reference，
可用于跨版本性能诊断，但不能用于判断同批 v25B no-op semantic drift。
因为 K1_H9 自身已经对这个 external reference 非零。
```

修复：

```text
tools/v25b_video_masklet_audit.py:
    added direct same-batch K1_H9 pose comparison.
    no-op hard gate now compares P0 rows against landed V25B_P0_SMOKE_R1_K1_H9.
    external-H9 failure is retained as diagnostic only:
        noop_external_H9_report_gate_pass = false
```

结果：

```text
noop_gate_method = direct_k1_pose_compare
noop_parity_gate_pass = true
noop_rows_checked = 4
noop_failures = []
```

### Blocker 2：fine labels 存在，但 runtime fine-label role policy 不存在

现象：

```text
Stage-C cache audit has 8 fine labels.
Runtime semantic_memory_path_summary is still coarse-group keyed.
```

处理：

```text
Fine labels are recorded as predicted video-masklet audit evidence.
Fine-label-specific policies are marked blocked/coarse fallback.
No sky/vegetation/fence runtime-policy success is claimed.
```

结果：

```text
runtime_fine_role_policy_available = false
Fine-label policy H3 is not actually validated in v25B.
```

---

## 4. Phase 2：Single-Path h10 Screen

输出：

```text
phase2_h10_screen_report_R1/
matrix_logs/phase2_h10_screen_R1/
rollouts/V25B_P2_H10_R1_*
```

设置：

```text
chunk = 10
horizon = 10
GPUs = 0,1,2,3
probe_cache_mode = readwrite
probe_cache_payload = read_path_min
reference = V25B_REF_R1_K1_H9 chunk10 h10
rows completed = 12/12
failures = 0
```

Candidates:

```text
FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT
FRAMESEM_02_LOWSTUFF_HIGHD_SKIP
FRAMESEM_03_SKY_NEUTRAL_VEGETATION_HIGHD_SKIP
GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT
GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP
CHUNKSEM_03_PROTECT_SPECIAL_TOKENS
SWASEM_01_STRUCTURE_CACHE_KEEP
SWASEM_02_LOWSTUFF_HIGHD_CACHE_SOFTDROP
SWASEM_06_CURRENT_AND_PREVIOUS_COMPARE
TTTSEM_01_STRUCTURE_POSITIVE_LONG
TTTSEM_02_LOWSTUFF_HIGHD_NEGATIVE_SHORT
TTTSEM_04_SEMANTIC_PLUS_TTT_CONFLICT
```

h10 gate summary:

| Metric | Best |
|---|---:|
| Best h10 ATE delta vs H9 | `-0.3175006931` |
| Best h10 `[200,300)` delta vs H9 | `-1.9404135647` |
| Best h10 `[400,600)` delta vs H9 | `-0.3310072074` |
| Phase 2 gate pass candidates | `[]` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Key rows:

| Candidate | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta |
|---|---:|---:|---:|
| `GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT` | `-0.3175006931` | `-1.9404135647` | `-0.3310072074` |
| `GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP` | `-0.3175006931` | `-1.9404135647` | `-0.3310072074` |
| `CHUNKSEM_03_PROTECT_SPECIAL_TOKENS` | `-0.3175006931` | `-1.9404135647` | `-0.3310072074` |
| `TTTSEM_01_STRUCTURE_POSITIVE_LONG` | `-0.0923098418` | `-0.0919824963` | `-0.1106929803` |
| `TTTSEM_02_LOWSTUFF_HIGHD_NEGATIVE_SHORT` | `-0.0923098418` | `-0.0919824963` | `-0.1106929803` |
| `TTTSEM_04_SEMANTIC_PLUS_TTT_CONFLICT` | `-0.0923098418` | `-0.0919824963` | `-0.1106929803` |
| `SWASEM_01_STRUCTURE_CACHE_KEEP` | `+0.0605432506` | `+0.1014813364` | `+0.0798655977` |
| `SWASEM_02_LOWSTUFF_HIGHD_CACHE_SOFTDROP` | `+0.0382902280` | `+0.0966188843` | `+0.0401519478` |

Runtime:

```text
h10 rows with wall_seconds:
    count = 12
    min = 227s
    max = 500s
    mean = 411.083s

slowest:
    SWASEM_06_CURRENT_AND_PREVIOUS_COMPARE = 500s
    TTTSEM_04_SEMANTIC_PLUS_TTT_CONFLICT = 470s
    TTTSEM_02_LOWSTUFF_HIGHD_NEGATIVE_SHORT = 463s
    TTTSEM_01_STRUCTURE_POSITIVE_LONG = 457s
```

Decision:

```text
Phase 2 h10 local gate = fail.

Required:
    h10 [200,300) delta <= -3m

Observed:
    best h10 [200,300) delta = -1.9404135647m
```

---

## 5. Phase 2：h15 Top Confirmation

输出：

```text
phase2_h15_top_report_R1/
matrix_logs/phase2_h15_top_R1/
rollouts/V25B_P2_H15_TOP_R1_*
```

设置：

```text
chunk = 10
horizon = 15
reference = V25B_REF_R1_K1_H9 chunk10 h15
selected rows completed = 5/5
failures = 0
```

Selected candidates:

```text
FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT
GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT
GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP
CHUNKSEM_03_PROTECT_SPECIAL_TOKENS
TTTSEM_01_STRUCTURE_POSITIVE_LONG
```

h15 gate summary:

| Metric | Best |
|---|---:|
| Best h15 ATE delta vs H9 | `-0.0735579504` |
| Best h15 `[200,300)` delta vs H9 | `-0.1254841547` |
| Best h15 `[400,600)` delta vs H9 | `-0.0931423111` |
| Phase 2 gate pass candidates | `[]` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Rows:

| Candidate | h15 ATE delta | h15 `[200,300)` delta | h15 `[400,600)` delta |
|---|---:|---:|---:|
| `FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT` | `0.0` | `0.0` | `0.0` |
| `GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT` | `+0.9146850021` | `+0.2485983272` | `+1.0377147353` |
| `GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP` | `+0.9146850021` | `+0.2485983272` | `+1.0377147353` |
| `CHUNKSEM_03_PROTECT_SPECIAL_TOKENS` | `+0.9146850021` | `+0.2485983272` | `+1.0377147353` |
| `TTTSEM_01_STRUCTURE_POSITIVE_LONG` | `-0.0735579504` | `-0.1254841547` | `-0.0931423111` |

Runtime:

```text
h15 rows with wall_seconds:
    count = 5
    min = 436s
    max = 660s
    mean = 490.6s

slowest:
    TTTSEM_01_STRUCTURE_POSITIVE_LONG = 660s
```

Decision:

```text
Phase 2 h15 confirmation gate = fail.

Required:
    h15 ATE delta <= -1.5m
    or h15 [200,300) delta <= -2.5m

Observed:
    best h15 ATE delta = -0.0735579504m
    best h15 [200,300) delta = -0.1254841547m
```

---

## 6. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Phase 0 cache / no-op | pass | cache hit 1.0, coverage high, direct K1 parity exact |
| Path consumption | pass | frame/global/SWA/TTT consumed, empty source events 0 |
| Passive semantic audit | pass | fine labels exist in cache audit, path metrics non-empty |
| Runtime fine-label policy | blocked | runtime metrics are coarse-group keyed |
| Phase 2 h10 screen | fail | best `[200,300)` delta = `-1.9404m`, gate requires `<= -3m` |
| Phase 2 h15 top | fail | best h15 ATE delta = `-0.0736m`, best `[200,300)` = `-0.1255m` |
| Phase 3 pairwise | not started | no Phase 2 candidate passed |
| Phase 4 all-memory | not started | pairwise entry forbidden |
| Phase 5 persistence combination | not started | no strong semantic candidate; current semantic signal too weak |
| No-GT selector | not started | gate failed |
| Full online validation | not started | selector/full-run entry forbidden |

Boundary:

```text
No v25B short-rollout result counts as deployable online TTT write success.
No GT semantic was used.
No no-GT selector was evaluated.
No full online validation was launched.
No online Target-25 result was produced in v25B.
```

---

## 7. Final Decision

v25B 的真实成功点：

```text
1. Video-masklet Stage-C cache is usable:
       cache_hit_rate = 1.0
       mean_coverage = 0.9816746437
       focus_coverage_200_300 = 0.9781515747

2. Semantic no-op plumbing is clean:
       P0_01 / P0_02 / P0_03 direct K1 pose diff = 0

3. Semantic path consumption is real:
       frame/global/SWA/TTT all have consumed-path evidence
       empty source event runs = []

4. h10 and h15 speed-gated screens completed without failed rows:
       h10 = 12/12
       h15 top = 5/5
```

v25B 的关键负结果：

```text
Current video-masklet semantic role did not produce strong local or durable correction.

h10 best:
    GLOBALSEM_01 / GLOBALSEM_02 / CHUNKSEM_03
    ATE delta = -0.3175006931m
    [200,300) delta = -1.9404135647m

h15 best:
    TTTSEM_01_STRUCTURE_POSITIVE_LONG
    ATE delta = -0.0735579504m
    [200,300) delta = -0.1254841547m

Global/chunk h15 rows regressed:
    ATE delta = +0.9146850021m
    [400,600) delta = +1.0377147353m
```

Interpretation:

```text
The video-masklet semantic frontend is not the blocker for basic coverage/plumbing.
It provides dense predicted semantic evidence and can be consumed by memory paths.

But the current coarse runtime role design is too weak for Target-25:
    frame path is effectively no-op in this screen,
    global/chunk has moderate h10 local improvement but h15 regresses,
    SWA is weak or slightly harmful,
    TTT is stable but tiny.

This is not a "h10 strong, h15 washed out" case.
It is a "semantic single-path signal is below entry gate" case.
```

Conclusion type:

```text
Semantic Prior Generator with video-masklet cache is deployable and correctly connected,
but current video-masklet semantic coarse-role policy does not have enough causal power
for Target-25.

Semantic should be kept as auxiliary diagnostic / weak source policy,
not promoted as the main Target-25 driver from this family.
```

Next required direction:

```text
Do not start pairwise/all-memory/selector/full online from v25B candidates.
Do not expand h15/chunk6/chunk16 sweeps for this semantic family.

If semantic work continues, first implement true runtime fine-label/path-specific role tensors:
    R_frame_tok
    R_global_tok
    R_swa_tok
    R_ttt_tok
    L_sem_tok consumed by runtime policy

Otherwise return Target-25 mainline to:
    explicit online trajectory-state,
    explicit online scale-state,
    skip-aware memory lifecycle,
    or merge/gauge-aware state correction.
```
