# ACL2 v27 实验复盘：SemanticPrior CausalRoleRouter AllMemory Parallel

日期：2026-05-22（Asia/Singapore）  
计划文件：`docs/ACL2_v27_SemanticPrior_CausalRoleRouter_AllMemory_Parallel_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v27_semanticprior_causalrolerouter_allmemory_parallel/`

本轮原则：只记录实际落盘结果；不把 implementation audit、passive attribution、smoke、short rollout、failed gate、或未启动矩阵写成 deployable online success。没有通过 Phase 2 gate 时，不启动 Phase 3 pairwise、Phase 4 all-memory、no-GT selector 或 full online validation。

---

## 0. 当前结论

v27 已按 speed-gated Phase 2 策略执行到 stop rule，未达到 Target-25。

已完成并落盘：

```text
1. 阅读 v27 计划，并实现 causal semantic role router 的工程接线。
2. Phase 0 H0 implementation hard gate 通过。
3. Phase 1 passive semantic-risk attribution 完成并通过 H1 gate。
4. 发现 Phase 2 R1 single-path 污染并隔离，不纳入报告。
5. 修复 single-path launcher：
       FG rows = readonly
       SWA rows = hybrid + probe_native
       TTT rows = semantic TTT only，不再隐式启用 v26 scale-state commit
6. Phase 0 R2 H0 audit 复验通过。
7. Phase 2 chunk10 h10 R2 screen 完成：
       rows = 17/17
       failures = 0
8. 按 v27 failure routing 继续修复 SWA hook：
       修复 launcher READ_PATH_VALUE 未同步导致 SWA rows 实际 read_path=none 的问题
       R2 smoke 证明 swa_read hook 生效
9. 补跑 SWA hook h10 diagnostic：
       rows = 5/5
       failures = 0
       best [200,300) delta = -2.3655037795m
10. 补做 SWA boundary diagnostics：
       boundary gate pass candidates = []
       mean boundary 10f delta vs H9 = +0.4789576412m
       mean boundary 20f delta vs H9 = +0.3681139080m
11. Phase 2 gate = fail。
```

最终边界：

```text
1. Phase 2 h10 best ATE delta = -0.1140785981m.
2. Phase 2 h10 best [200,300) delta = -0.2707409448m.
3. Phase 2 h10 best [400,600) delta = -0.1256322862m.
4. SWA hook repair h10 best ATE delta = -0.3641983213m.
5. SWA hook repair h10 best [200,300) delta = -2.3655037795m.
6. SWA hook repair boundary 10f / 20f metrics regressed vs H9.
7. No Phase 2 candidate passed gate.
8. No h15 confirmation was launched.
9. No Phase 3 pairwise combination was started.
10. No Phase 4 all-memory validation was started.
11. No no-GT selector was started.
12. No full online validation was launched.
13. No online Target-25 result was produced in v27.
```

Gate result：

```text
Required to enter h15 / Phase 3:
    h10 [200,300) delta <= -3m
    or h10 ATE delta <= -1.5m with downstream regression <= +1m

Observed:
    best h10 [200,300) delta = -0.2707409448m
    best h10 ATE delta = -0.1140785981m

Post-fix SWA hook diagnostic:
    best h10 [200,300) delta = -2.3655037795m
    best h10 ATE delta = -0.3641983213m
    boundary 10f delta vs H9 = +0.4789576412m

Therefore:
    Phase 2 gate = fail.
```

Downstream boundary：

```text
No Phase 3 pairwise combination has been started.
No Phase 4 all-memory validation has been started.
No no-GT selector has been started.
No full online validation has been launched.
No online Target-25 result has been produced in v27.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v27 不是 GT semantic 实验。
semantic_source = video_masklet_frontend_cache
uses_gt_semantic = false
uses_video_masklet_semantic = true

v27 当前 conflict/scale risk 是 provenance-tagged broadcast 条件，
不是 token_exact 条件。
不能声称已经实现 token_exact conflict / scale-risk routing。
```

---

## 1. 工程修改

新增 / 修改：

```text
loger/pipeline/hybrid_memory_controller.py:
    added semantic condition provenance state:
        semantic_condition_conflict_level/source/value/summary
        semantic_condition_scale_level/source/value/summary
    added set_semantic_condition_signals(...)
    _fine_path_roles now supports causal_* policies
    causal role policy consumes:
        L_sem_tok
        D_g / Q_mask
        conflict condition C
        scale-risk condition S
    fine_label_condition_metrics now records:
        conflict_mean / conflict_p90
        scale_risk_mean / scale_risk_p90
    semantic_memory_path_summary records:
        condition_signal_conflict_available
        condition_signal_conflict_level/source/value/summary
        condition_signal_scale_risk_available
        condition_signal_scale_risk_level/source/value/summary

run_pipeline_abc_v2.py:
    added causal semantic role policy choices:
        causal_path_router
        causal_path_router_debug
        causal_fg_*
        causal_swa_*
        causal_ttt_*
    added v27 scale condition extraction from v19 scale-state payload
    added v27 pre-role conflict extraction from no-commit TTT update_conflict_energy probe
    final control_prior is rebuilt after conflict/scale condition signals are set
    semantic_memory_path_summary.jsonl now forwards causal condition provenance

tools/run_v24_candidate_rollout.sh:
    reused as trusted rollout launcher for v27 aliases
    added Phase 0 v27 aliases:
        V27_P0_00_H9_REFERENCE
        V27_P0_01_CAUSAL_LOADED_BUT_IGNORED
        V27_P0_02_CAUSAL_PASS_THROUGH_CONSUMED
        V27_P0_03_CAUSAL_DEBUG_ONLY_ALL_PATHS
        V27_P0_04_CAUSAL_FRAME_GLOBAL_SMOKE
        V27_P0_05_CAUSAL_SWA_SMOKE
        V27_P0_06_CAUSAL_TTT_SMOKE
    added Phase 2 v27 aliases:
        FG_RISK_00
        FG_SEM_01..FG_SEM_05
        SWA_SEM_01..SWA_SEM_05
        TTT_ROLE_00_RISK_ONLY
        TTT_ROLE_01_STRUCTURE_LOWCONFLICT_POS
        TTT_ROLE_02_LOWSTUFF_HIGHD_CONFLICT_SHORTNEG
        TTT_ROLE_03_VEGETATION_CONDITIONAL_NEG
        TTT_ROLE_04_BLOCK_HIGHCONFLICT_STRUCTURE_LONGWRITE
        TTT_ROLE_05_FULL_ROLE_TREE

tools/v24_candidate_bank_report.py:
    added v27 candidate family labels

tools/v27_implementation_self_audit.py:
    strict Phase 0 H0 audit for causal role router
    verifies:
        cache quality
        no-op direct pose parity
        R_frame/R_global/R_swa/R_ttt non-empty
        frame/global/SWA/TTT consumed
        conflict/scale condition available
        context empty source events = 0
    writes:
        codex_self_check_report.md
        codex_self_check_summary.json
        codex_self_check_failures.jsonl
        semantic_role_router_audit.csv
        per_token_condition_summary.csv
        noop_parity_metrics.csv

tools/v26_passive_fine_label_attribution.py:
    made v27-compatible
    now reads conflict/scale metrics and provenance from runtime summaries
    no longer hardcodes conflict/scale unavailable for v27

tools/v27_swa_boundary_diagnostics.py:
    offline SWA boundary diagnostic from landed trajectories only
    compares candidate vs same-slice H9 reference:
        boundary_10f_ATE
        boundary_20f_ATE
        chunk_boundary_pose_jump
    writes:
        swa_boundary_by_candidate_boundary.csv
        swa_boundary_summary.csv
        swa_boundary_summary.json
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    loger/pipeline/hybrid_memory_controller.py \
    run_pipeline_abc_v2.py \
    tools/v24_candidate_bank_report.py \
    tools/v27_implementation_self_audit.py \
    tools/v27_swa_boundary_diagnostics.py \
    tools/v26_passive_fine_label_attribution.py

bash -n \
    tools/run_v24_candidate_rollout.sh \
    tools/run_v24_phase2_initial_queue.sh \
    tools/run_attention_cue_experiment.sh

PASS
```

---

## 2. Blocker 与修复记录

### Blocker 1：v27 Phase 0 首次运行误写入 v24 root

现象：

```text
初始运行只设置了绝对 V24_RESULT_ROOT，
但没有同步设置 V24_ROOT。
tools/run_v24_candidate_rollout.sh 使用 V24_ROOT 生成 rollout root，
导致 V27_P0_SMOKE_R1_* 目录误写入 v24 result root。
```

处理：

```text
误写目录已移动到：
results/kitti01_hmc_v2/acl2_v24_semanticprior_pathspecific_allmemory_parallel/
    rollouts/.INVALID_V27_ROOT_MISROUTED_20260522T1416/

后续 v27 运行同时设置：
    V24_ROOT=results/kitti01_hmc_v2/acl2_v27_semanticprior_causalrolerouter_allmemory_parallel
    V24_RESULT_ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v27_semanticprior_causalrolerouter_allmemory_parallel
```

边界：

```text
misrouted rows are not counted in v27 reports.
```

### Blocker 2：causal role router 首次实现出现 undefined high_thr

现象：

```text
_fine_path_roles causal policy referenced high_thr,
but that local variable did not exist in the function scope.
```

修复：

```text
hybrid_memory_controller.py:
    replaced the bad high_thr reference with the already computed high_d condition.
```

结果：

```text
py_compile pass
Phase 0 rows can run.
```

### Blocker 3：conflict condition initially unavailable

现象：

```text
H0 audit initially failed:
    condition_signal_conflict_available = false
    condition_signal_scale_risk_available = true
```

原因：

```text
v27 pre-role TTT probe used gradient_reversal_gamma = 0.0.
TTTWriteController skips update_conflict_energy risk computation when effective gamma is zero.
Therefore no conflict risk statistic was produced.
```

修复：

```text
run_pipeline_abc_v2.py:
    pre-role no-commit conflict probe now uses diagnostic gamma = 1e-6
    branch/layer/head gamma routes are cleared during the diagnostic probe
    controller settings are restored after the probe

This triggers update_conflict_energy statistics without committing TTT state.
```

结果：

```text
H0 rerun:
    condition_signal_conflict_available = true
    condition_signal_conflict_level = chunk_broadcast
    condition_signal_conflict_source = pre_role_probe_ttt_update_conflict_risk_p90
    condition_signal_scale_risk_available = true
    condition_signal_scale_risk_level = chunk_broadcast
    condition_signal_scale_risk_source = v19_scale_state_payload_abs_log_ratio
```

边界：

```text
This is not token_exact risk conditioning.
It is provenance-tagged chunk_broadcast conditioning.
```

### Blocker 4：Phase 2 R1 single-path rows 被隐式 scale-state / TTT commit 污染

现象：

```text
Phase 2 R1 的 FG/SWA rows 原本应是 single-path causal semantic rows。
但 command line 显示它们继承了 v26 risk helper：
    --hybrid_memory_mode hybrid
    --hmc_commit_mode probe_ttt_write
    --ttt_write_scale_state_mode projection_risk
    --ttt_write_gradient_reversal_risk_source v19_scale_state

这会把 FG/SWA single-path screen 混入最终 TTT scale-state commit。
```

处理：

```text
停止旧 Phase 2 R1 队列。
已落盘/半截 R1 rows 和 R1 matrix logs 移动到：
results/kitti01_hmc_v2/acl2_v27_semanticprior_causalrolerouter_allmemory_parallel/
    rollouts/.INVALID_20260522T1530_SINGLEPATH_CONFONDED/
    matrix_logs/phase2_h10_screen_R1/.INVALID_20260522T1530_SINGLEPATH_CONFONDED/

R1 不纳入任何 Phase 2 gate/report。
```

代码修复：

```text
run_pipeline_abc_v2.py:
    added condition-only scale payload for causal router.
    If final ttt_write_scale_state_mode is off, v27 still computes
    scale condition from a local projection_risk payload without updating tracker
    or enabling final TTT commit.

tools/run_v24_candidate_rollout.sh:
    FG_RISK_00 / FG_SEM_* now run readonly.
    SWA_SEM_* now run hybrid with HMC_COMMIT_MODE=probe_native.
    TTT_ROLE_* no longer implicitly enables v26 scale-state commit.

tools/run_attention_cue_experiment.sh:
    hybrid mode now honors HMC_COMMIT_MODE env override.
```

验证：

```text
py_compile run_pipeline_abc_v2.py PASS
bash -n tools/run_attention_cue_experiment.sh tools/run_v24_candidate_rollout.sh PASS
Phase 0 R2 H0 audit PASS
```

### Blocker 5：SWA Phase 2 rows 没有实际 SWA hook intervention

现象：

```text
Phase 2 R2 SWA_SEM_* rows completed, but trajectory deltas are all zero:
    ATE_delta_vs_H9 = 0
    [200,300) delta = 0
    raw_trans_max_diff = 0

01.log hook evidence:
    swa_read num_enabled_layers = 0
    num_swa_overlap_bias_applied = 0
    num_swa_overlap_source_gate_applied = 0
    num_swa_overlap_source_replace_applied = 0
```

Interpretation：

```text
The semantic router can emit R_swa and H0 path-consumption metrics are present,
but the Phase 2 SWA aliases did not produce a real SWA read/write intervention.
Therefore SWA Phase 2 rows are counted as completed diagnostic rows,
but not claimed as effective SWA semantic corrections.
```

Decision：

```text
Do not claim SWA semantic role success from v27 Phase 2.
Do not launch SWA h15 confirmation.
Record this as a runtime hook wiring/alias limitation for future work.
```

### Blocker 6：SWA launcher read_path 修复与补测

触发原因：

```text
v27 plan failure routing says:
    SWA semantic 弱时，不继续大矩阵；
    先检查 boundary metrics / SWA cache path；
    如果 SWA cache mass 没变，检查 SWA role stream 是否被写入 cache path。

Blocker 5 证明原 Phase 2 R2 SWA rows 没有进入 swa_read hook。
因此需要先修正接线，再补一个小规模 SWA h10 diagnostic。
```

错误尝试：

```text
第一次手动 smoke 命令参数顺序写错：
    bash tools/run_v24_candidate_rollout.sh V27_P0_05_CAUSAL_SWA_SMOKE 10 3 1

script usage 实际是：
    GPU CANDIDATE_ID CHUNK_ID HORIZON

结果：
    Unsupported HORIZON: 1
    没有启动有效实验 row。
```

第一次修复：

```text
tools/run_v24_candidate_rollout.sh:
    enable_swa_policy() 里加入 export READ_PATH=swa

结果：
    V27_SWA_HOOK_FIX_SMOKE_R1 completed,
    但 hook 仍然显示：
        implemented_paths = ['ttt_update', 'frame_attention']
        swa_read num_enabled_layers = 0

原因：
    launcher 最终传给 run_attention_cue_experiment.sh 的是 READ_PATH_VALUE，
    而 READ_PATH_VALUE 在 candidate dispatch 前已经固定为 none。
```

最终修复：

```text
tools/run_v24_candidate_rollout.sh:
    enable_swa_policy() now sets:
        export READ_PATH=swa
        READ_PATH_VALUE=swa

bash -n tools/run_v24_candidate_rollout.sh
PASS
```

R2 smoke evidence：

```text
rollout:
    V27_SWA_HOOK_FIX_SMOKE_R2_V27_P0_05_CAUSAL_SWA_SMOKE_chunk10_h3_globalgate_H9parent_SWKS3

implemented_paths:
    ['ttt_update', 'swa_read']

swa_read:
    num_enabled_layers sum = 4
    num_source_gate_applied sum = 4
    mean_swa_gate avg = 0.9825138003
```

Decision：

```text
SWA hook activation blocker is fixed.
This permits a small SWA hook h10 diagnostic.
It does not retroactively make old SWA R2 rows valid SWA interventions.
```

---

## 3. Phase 0：Implementation Hard Gate

输出：

```text
implementation_audit/
matrix_logs/phase0_smoke_R1/
rollouts/V27_P0_SMOKE_R1_*
matrix_logs/phase0_smoke_R2/
rollouts/V27_P0_SMOKE_R2_*
```

Phase 0 smoke：

```text
V27_P0_00_H9_REFERENCE
V27_P0_01_CAUSAL_LOADED_BUT_IGNORED
V27_P0_02_CAUSAL_PASS_THROUGH_CONSUMED
V27_P0_03_CAUSAL_DEBUG_ONLY_ALL_PATHS
V27_P0_04_CAUSAL_FRAME_GLOBAL_SMOKE
V27_P0_05_CAUSAL_SWA_SMOKE
V27_P0_06_CAUSAL_TTT_SMOKE

chunk = 10
horizon = 3
R1 rows completed = 7/7
R2 rows completed = 7/7
```

H0 audit summary：

| Metric | Value |
|---|---:|
| `all_gate_pass` | `true` |
| `cache_hit_rate` | `1.0` |
| `cache_quality_gate_pass` | `true` |
| `chunks_with_masklets_ratio` | `1.0` |
| `mean_coverage` | `0.9816746437` |
| `focus_coverage_200_300` | `0.9781515747` |
| `fine_label_count` | `8` |
| `runtime_fine_role_policy_available` | `true` |
| `path_specific_role_streams_available` | `true` |
| `condition_signal_conflict_available` | `true` |
| `condition_signal_scale_risk_available` | `true` |
| `context_empty_source_events` | `0` |

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

```text
noop_parity_gate_pass = true
noop_rows_checked = 4
noop_failures = []
```

Decision：

```text
Phase 0 H0 gate = pass.
After Phase 2 R1 contamination was found, Phase 0 R2 was rerun with the
fixed launcher and H0 audit passed again:
    all_gate_pass = true
    noop_parity_gate_pass = true
    condition_signal_conflict_available = true
    condition_signal_scale_risk_available = true
    path_consumption_flags frame/global/swa/ttt = true
    context_empty_source_events = 0

Phase 1 passive semantic-risk attribution is allowed.
No deployable online result is claimed.
No selector/full online validation is allowed.
```

---

## 4. Phase 1：Passive Semantic-Risk Attribution

输出：

```text
phase1_passive_attribution/
phase1_visual_dashboard/
matrix_logs/phase1_passive_R1/
rollouts/V27_P1_PASSIVE_R1_*
```

设置：

```text
candidate = V27_P0_03_CAUSAL_DEBUG_ONLY_ALL_PATHS
chunks = 6, 10, 16
horizons = 10, 15
rows completed = 6/6
failures = 0
```

Runtime：

| Chunk | Horizon | wall seconds |
|---:|---:|---:|
| `6` | `10` | `498` |
| `6` | `15` | `717` |
| `10` | `10` | `453` |
| `10` | `15` | `659` |
| `16` | `10` | `513` |
| `16` | `15` | `765` |

Phase 1 summary：

| Metric | Value |
|---|---:|
| `phase1_gate_pass` | `true` |
| `fine_label_count` | `9` |
| `condition_rows` | `9` |
| `memory_rows` | `3028` |
| `action_rows` | `3028` |
| `coarse_internal_label_diversity` | `true` |
| `path_role_diversity_within_coarse` | `true` |
| `D_mean_range_within_coarse_max` | `0.2667130153` |
| `conflict_signal_available` | `true` |
| `scale_risk_signal_available` | `true` |

Fine-label D/risk evidence：

| Fine label | Coarse group | token count | weighted D mean | weighted conflict mean | weighted scale-risk mean |
|---|---|---:|---:|---:|---:|
| `road` | `STRUCTURE_ANCHOR` | `1305297` | `0.1636931081` | `0.0042573326` | `0.0564964038` |
| `building` | `STRUCTURE_ANCHOR` | `143144` | `0.4038646922` | `0.0021306254` | `0.0281012054` |
| `wall` | `STRUCTURE_ANCHOR` | `23485` | `0.4304061234` | `0.0003592270` | `0.0038163290` |
| `sky` | `LOW_VALUE_STUFF` | `1508609` | `0.2171635318` | `0.0047860439` | `0.0625187629` |
| `vegetation` | `LOW_VALUE_STUFF` | `787917` | `0.2280308183` | `0.0042865551` | `0.0564887804` |
| `grass` | `LOW_VALUE_STUFF` | `176171` | `0.3088524714` | `0.0042317204` | `0.0505300005` |

Interpretation：

```text
H1 passes by attribution criteria:
    fine labels differ inside the same coarse group;
    path roles differ inside coarse group;
    conflict and scale-risk conditions are present with provenance.

But conflict / scale-risk are currently chunk_broadcast style signals.
The passive attribution does not prove token_exact risk routing.
```

Decision：

```text
Phase 1 gate = pass.
Phase 2 chunk10 h10 single-path screen is allowed.
No pairwise/all-memory/selector/full validation is allowed yet.
```

---

## 5. Phase 2：Single-Path h10 Screen

输出：

```text
phase2_h10_screen_report_R2/
matrix_logs/phase2_h10_screen_R2/
rollouts/V27_P2_H10_R2_*
```

设置：

```text
chunk = 10
horizon = 10
GPUs = 1,2,3
probe_cache_mode = readwrite
probe_cache_payload = read_path_min
reference = V27_REF_R2_V27_P0_00_H9_REFERENCE chunk10 h10
rows completed = 17/17
failures = 0
```

Candidates：

```text
FG_RISK_00
FG_SEM_01
FG_SEM_02
FG_SEM_03
FG_SEM_04
FG_SEM_05
SWA_SEM_01
SWA_SEM_02
SWA_SEM_03
SWA_SEM_04
SWA_SEM_05
TTT_ROLE_00_RISK_ONLY
TTT_ROLE_01_STRUCTURE_LOWCONFLICT_POS
TTT_ROLE_02_LOWSTUFF_HIGHD_CONFLICT_SHORTNEG
TTT_ROLE_03_VEGETATION_CONDITIONAL_NEG
TTT_ROLE_04_BLOCK_HIGHCONFLICT_STRUCTURE_LONGWRITE
TTT_ROLE_05_FULL_ROLE_TREE
```

h10 gate summary：

| Metric | Best |
|---|---:|
| Best h10 ATE delta vs H9 | `-0.1140785981` |
| Best h10 ATE candidate | `TTT_ROLE_00_RISK_ONLY` |
| Best h10 `[200,300)` delta vs H9 | `-0.2707409448` |
| Best h10 `[200,300)` candidate | `FG_RISK_00 / FG_SEM_01 / FG_SEM_02 / FG_SEM_03` |
| Best h10 `[400,600)` delta vs H9 | `-0.1256322862` |
| Phase 2 gate pass candidates | `[]` |
| Selector allowed | `false` |
| Full online validation allowed | `false` |

Key rows：

| Candidate | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta |
|---|---:|---:|---:|
| `TTT_ROLE_00_RISK_ONLY` | `-0.1140785981` | `-0.1309044684` | `-0.1256322862` |
| `FG_RISK_00` | `-0.0327972027` | `-0.2707409448` | `+0.0098508412` |
| `FG_SEM_01` | `-0.0327972027` | `-0.2707409448` | `+0.0098508412` |
| `FG_SEM_04` | `-0.0397311078` | `-0.1088859305` | `-0.0302639389` |
| `SWA_SEM_01` | `0.0` | `0.0` | `0.0` |
| `SWA_SEM_05` | `0.0` | `0.0` | `0.0` |
| `TTT_ROLE_01_STRUCTURE_LOWCONFLICT_POS` | `+0.0380742595` | `+0.0913194733` | `+0.0328734481` |
| `TTT_ROLE_05_FULL_ROLE_TREE` | `+0.0380742595` | `+0.0913194733` | `+0.0328734481` |

Runtime：

```text
h10 rows with wall_seconds:
    count = 17
    min = 242s
    max = 1288s
    mean = 599.8823529412s

slowest:
    TTT_ROLE_03_VEGETATION_CONDITIONAL_NEG = 1288s
    TTT_ROLE_00_RISK_ONLY = 1284s
    TTT_ROLE_02_LOWSTUFF_HIGHD_CONFLICT_SHORTNEG = 1244s
    TTT_ROLE_05_FULL_ROLE_TREE = 1230s
    TTT_ROLE_01_STRUCTURE_LOWCONFLICT_POS = 1165s
    TTT_ROLE_04_BLOCK_HIGHCONFLICT_STRUCTURE_LONGWRITE = 1147s
```

Decision：

```text
Phase 2 h10 gate = fail.

Required:
    h10 [200,300) delta <= -3m
    or h10 ATE delta <= -1.5m with downstream regression <= +1m

Observed:
    best h10 [200,300) delta = -0.2707409448m
    best h10 ATE delta = -0.1140785981m

No h15 confirmation is allowed.
No Phase 3 / Phase 4 / selector / full online validation is allowed.
```

---

## 6. SWA Hook Repair Diagnostic

触发原因：

```text
Phase 2 R2 SWA rows were completed but no-op.
After fixing READ_PATH_VALUE propagation, the plan-required SWA cache/path check
was rerun as a small h10 diagnostic.
```

输出：

```text
phase2_swa_hook_h10_report_R1/
phase2_swa_hook_boundary_report_R1/
matrix_logs/phase2_swa_hook_h10_R1/
rollouts/V27_P2_SWA_HOOK_R1_SWA_SEM_*
```

设置：

```text
chunk = 10
horizon = 10
GPUs = 0,1,2,3
probe_cache_mode = readwrite
probe_cache_payload = read_path_min
reference = V27_REF_R2_V27_P0_00_H9_REFERENCE chunk10 h10
rows completed = 5/5
failures = 0
```

Hook evidence：

| Candidate | swa enabled layers | source gate applied | mean_swa_gate |
|---|---:|---:|---:|
| `SWA_SEM_01` | `11` | `11` | `0.9825323062` |
| `SWA_SEM_02` | `11` | `11` | `0.9825323062` |
| `SWA_SEM_03` | `11` | `11` | `0.9825323062` |
| `SWA_SEM_04` | `11` | `11` | `0.9825323062` |
| `SWA_SEM_05` | `11` | `11` | `0.9825323062` |

h10 report：

| Candidate | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta | wall seconds |
|---|---:|---:|---:|---:|
| `SWA_SEM_01` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` | `254` |
| `SWA_SEM_02` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` | `239` |
| `SWA_SEM_03` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` | `254` |
| `SWA_SEM_04` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` | `254` |
| `SWA_SEM_05` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` | `231` |

Gate summary：

```text
phase2_gate_pass_candidates = []
selector_allowed = false
full_online_validation_allowed = false

Required to enter h15:
    h10 ATE delta <= -1.5m
    or h10 [200,300) delta <= -3m

Observed:
    h10 ATE delta = -0.3641983213m
    h10 [200,300) delta = -2.3655037795m
```

Boundary diagnostic：

| Metric | Value |
|---|---:|
| boundary candidates | `5` |
| boundary count per candidate | `11` |
| mean boundary 10f delta vs H9 | `+0.4789576412` |
| mean boundary 10f improvement ratio | `-0.0241153897` |
| mean boundary 20f delta vs H9 | `+0.3681139080` |
| mean boundary 20f improvement ratio | `-0.0180280938` |
| mean pose jump delta vs H9 | `-0.0186129478` |
| boundary gate pass candidates | `[]` |

Decision：

```text
SWA hook repair produced a real intervention and a much stronger h10 local
[200,300) improvement than the original R2 no-op SWA rows.

However:
    h10 local gate still failed:
        -2.3655m > required -3m
    h10 ATE gate failed:
        -0.3642m > required -1.5m
    boundary 10f / 20f ATE regressed vs H9.

Therefore no h15 confirmation is allowed.
No Phase 3 / Phase 4 / selector / full online validation is allowed.
```

---

## 7. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Phase 0 H0 implementation hard gate | pass | cache/no-op/four role streams/condition provenance/path consumption passed |
| Phase 1 passive attribution | pass | fine labels have D/risk/path-role differences; conflict/scale are present |
| Phase 2 R1 | invalidated | single-path rows were confounded by implicit scale-state / final TTT commit |
| Phase 2 R2 h10 | fail | best ATE `-0.1141m`, best `[200,300)` `-0.2707m` |
| SWA Phase 2 R2 rows | diagnostic only | trajectory no-op; `swa_read` enabled layers = 0 |
| SWA hook repair h10 | fail | real `swa_read` hook, best `[200,300)` `-2.3655m`, gate requires `<= -3m`; boundary ATE regressed |
| h15 confirmation | not started | Phase 2 h10 gate failed |
| Phase 3 pairwise | not started | no Phase 2 candidate passed |
| Phase 4 all-memory | not started | pairwise entry forbidden |
| Phase 5 washout attribution | not started | no h10 strong / h15 weak candidate |
| No-GT selector | not started | gate failed |
| Full online validation | not started | selector/full-run entry forbidden |

Boundary：

```text
No v27 short-rollout result counts as deployable online TTT write success.
No GT semantic was used.
No no-GT selector was evaluated.
No full online validation was launched.
No online Target-25 result was produced in v27.
```

---

## 8. Final Decision

v27 的真实成功点：

```text
1. Causal semantic role router is implemented with provenance-tagged
   conflict / scale-risk conditions.
2. Phase 0 H0 passed after the conflict diagnostic gamma fix.
3. Phase 0 R2 revalidated H0 after the single-path launcher fix.
4. Phase 1 passive attribution passed and showed real fine-label D/risk variation.
5. Phase 2 R2 completed 17/17 clean h10 rows with no failed rows.
6. The SWA hook alias blocker was fixed and verified:
       swa_read enabled layers = 11 for h10 SWA hook rows
       source gate applied = 11
7. A small SWA hook diagnostic completed 5/5 rows.
```

v27 的关键负结果：

```text
Causal semantic role router did not produce a meaningful h10 correction.

Best h10 ATE:
    TTT_ROLE_00_RISK_ONLY
    delta = -0.1140785981m

Best h10 [200,300):
    FG_RISK_00 / FG_SEM_01 / FG_SEM_02 / FG_SEM_03
    delta = -0.2707409448m

Original SWA rows were no-op at trajectory level:
    delta = 0
    raw_trans_max_diff = 0
    hook evidence: swa_read enabled layers = 0

After SWA hook repair:
    SWA_SEM_01..05 h10 ATE delta = -0.3641983213m
    SWA_SEM_01..05 h10 [200,300) delta = -2.3655037795m
    boundary 10f delta vs H9 = +0.4789576412m
    boundary 20f delta vs H9 = +0.3681139080m
    boundary gate pass candidates = []

No candidate met:
    h10 [200,300) delta <= -3m
    h10 ATE delta <= -1.5m
```

Interpretation：

```text
v27 validates part of the engineering direction:
    semantic fine labels can be combined with D/Q and causal risk provenance,
    the condition availability is auditable,
    and no-op plumbing remains clean.

But the actual Phase 2 causal single-path effects are much weaker than v26.
The condition signals are chunk_broadcast, not token_exact, and the tested
causal role policies do not create Target-25-scale correction.

The SWA branch first exposed a runtime alias limitation:
    semantic SWA role streams existed, but Phase 2 R2 aliases did not activate
    a real SWA read intervention.

After fixing that launcher bug, SWA became a real intervention and produced
a larger local [200,300) improvement, but it still missed the h10 gate and
regressed boundary-local ATE. This means the failure is no longer simply
"SWA hook not wired"; the tested SWA semantic source policy itself is still
not strong/durable enough.
```

Conclusion type：

```text
Current v27 causal role router is valid infrastructure but not a successful
Target-25 correction mechanism.

Do not run h15, Phase 3, Phase 4, selector, or full online validation from v27.
Do not count any v27 result as deployable online TTT write success.
```

Next required direction：

```text
If continuing this family, the next work is not another threshold sweep.
Required engineering fixes are:
    token_exact or token-aligned conflict / scale-risk tensors,
    stronger SWA boundary/topology diagnostics and policies beyond the tested hook repair,
    and a runtime audit that separates semantic role availability from actual
    hook-side trajectory intervention.

Otherwise Target-25 mainline should move back to explicit online trajectory-state,
explicit scale-state, skip-aware lifecycle, or merge/gauge-aware correction.
```
