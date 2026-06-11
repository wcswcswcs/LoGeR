# ACL2 v32 实验复盘：SemanticCue Transfer Target30

日期：2026-05-23（Asia/Singapore）  
计划文件：`docs/ACL2_v32_SemanticCue_Transfer_Target30_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/`

本轮原则：只记录实际落盘结果；不把 fixed chunk diagnostic、runtime trigger 失败、C9 interaction 失败、repair 失败、或未启动矩阵写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立任务分配到不同 GPU。

---

## 0. 当前结论

v32 已按计划完成 local-to-full transfer audit、C9 interaction audit，并按失败分流尝试了一个 conservative residual repair。没有达到 Target-30，也没有产生新的 deployable online result。

已完成并落盘：

```text
1. 阅读 v32 计划，并实现 semantic cue fixed-window / runtime-trigger gating。
2. H1 R1 invalid：
       current_chunk_idx 没有在 build_control_prior 前同步，fixed chunk gate 实际看到 -1。
3. 修复 current_chunk_idx 同步后，H1 R2/R3 full online 完成：
       rows = 5/5 new rows + 2 existing references
       best = H1_FIXED_CH10_12
       full ATE = 36.0801185755m
       Target-30 fail
4. H1 runtime trigger 修复尝试完成：
       semantic_z_high_mass trigger failed
       semantic_d_mean_ema_mad trigger failed
       semantic_d_q90_ema_mad trigger failed
5. H2 C9 interaction 完成：
       C9 + SEM_Z all chunks completed
       C9 + runtime D-mean completed
       C9 + fixed chunk10-12 diagnostic completed
       all C9+semantic rows are worse than C9 full ATE
6. H3 conservative residual repair completed：
       C9 + SEM_RESID_COARSE_L025 all chunks completed
       worse than C9 full ATE
7. Full online Target-30 gate = fail。
```

最终边界：

```text
1. H1 fixed chunk10-12 diagnostic improves H9 full base:
       ATE delta vs H9 = -0.7390457418m
       [200,300) delta vs H9 = -1.2521234262m
   but fixed chunk-id activation is diagnostic only.

2. H1 runtime triggers did not transfer the v31 local effect:
       semantic_z_high_mass ATE delta vs H9 = +0.4207011488m
       semantic_d_mean_ema_mad ATE delta vs H9 = +0.0552074110m
       semantic_d_q90_ema_mad ATE delta vs H9 = +0.3083686367m

3. H2 C9 interaction gate failed:
       C9 reference ATE = 33.7629421029m
       best C9+semantic ATE = 34.2952287438m
       best C9+semantic delta vs C9 = +0.5322866409m

4. H3 residual repair failed:
       C9_RESID_COARSE_ALL ATE = 34.3258261120m
       delta vs C9 = +0.5628840091m

5. No v32 row reached full ATE <= 30m.
6. No runtime-trigger row improved C9.
7. No selector or additional all-memory matrix was launched.
8. No online Target-30 result was produced in v32.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v32 confirms that v31 semantic cue reconditioning has real local/full diagnostic
signal under fixed chunk activation, but the tested runtime triggers do not
recover it, and C9 interaction is negative.

Fixed chunk activation is not deployable because it uses KITTI01 chunk ids.
The deployable-style runtime triggers failed.
```

---

## 1. 工程修改

新增 / 修改：

```text
loger/pipeline/hybrid_memory_controller.py:
    added v32 semantic cue gating state:
        v32_semantic_cue_active_chunks
        v32_semantic_cue_trigger_mode
        v32_semantic_cue_trigger_high_threshold
        v32_semantic_cue_trigger_k
        v32_semantic_cue_trigger_min_mass
        v32_semantic_cue_trigger_warmup
        v32_semantic_cue_trigger_ema_alpha

    added runtime trigger modes:
        semantic_z_ema_mad
        semantic_d_mean_ema_mad
        semantic_d_q90_ema_mad

    added debug fields:
        v32_semantic_cue_gate_mode
        v32_semantic_cue_active
        v32_semantic_cue_gate_reason
        v32_semantic_cue_chunk_idx
        v32_semantic_z_high_mass
        v32_semantic_d_mean
        v32_semantic_d_q90
        v32_semantic_trigger_metric
        v32_semantic_trigger_threshold
        v32_semantic_trigger_ema/mad/count

run_pipeline_abc_v2.py:
    added CLI args for v32 gating / runtime trigger.
    forwards v32 debug fields into hmc_state_hash.jsonl.
    fixed current_chunk_idx synchronization before each build_control_prior call.

tools/run_attention_cue_experiment.sh:
    forwards V32_SEMANTIC_CUE_* env vars to run_pipeline_abc_v2.py.

tools/v32_transfer_report.py:
    aggregates full-online metrics from absolute or relative rollout dirs.
    Computes full ATE, segment ATEs, deltas vs reference, C9 delta, Target-30 gate,
    and v32 activation stats.
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    loger/pipeline/hybrid_memory_controller.py \
    run_pipeline_abc_v2.py \
    tools/v32_transfer_report.py

bash -n tools/run_attention_cue_experiment.sh

PASS
```

---

## 2. Blocker 与修复记录

### Blocker 1：H1 R1 fixed chunk gate 看到 chunk_idx = -1

现象：

```text
H1 R1 fixed chunk runs did not activate the requested chunks.
hmc_state_hash.jsonl showed:
    prior_v32_semantic_cue_chunk_idx = -1
```

原因：

```text
run_pipeline_abc_v2.py did not synchronize hmc.current_chunk_idx before
hmc.build_control_prior(...). The semantic cue gate therefore could not see
the actual full-online chunk id.
```

修复：

```text
run_pipeline_abc_v2.py:
    before each build_control_prior call:
        hmc.current_chunk_idx = int(effective_ci)
        hmc.ttt_update_controller.current_chunk_idx = int(effective_ci)
```

结果：

```text
H1 R2 fixed chunk runs recorded correct active chunks:
    H1_FIXED_CH10_12 active chunks = [10,11,12]
    H1_FIXED_CH6_12 active chunks = [6,7,8,9,10,11,12]
```

R1 decision：

```text
H1 R1 is invalid and not used for conclusions.
```

### Blocker 2：H2 fixed diagnostic first launch used wrong result-root env var

现象：

```text
The first V32_H2_03_C9_FIXED_CH10_12 launch used:
    ATTENTION_CUE_BASE

But tools/run_attention_cue_experiment.sh reads:
    ATTN_CUE_BASE

The row started under the default attention_cue_library_v1 root.
```

处理：

```text
The misrouted run was killed before completion and moved to:
    results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/
        rollouts_invalid_misrouted/V32_H2_03_C9_FIXED_CH10_12_WRONG_BASE_20260523T141130/

It is not used for any metric or gate.
```

修复：

```text
Relaunched as:
    V32_H2_03_C9_FIXED_CH10_12_R2
with:
    ATTN_CUE_BASE=results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h2_c9_combo/rollouts
```

### Blocker 3：H3 repair matrix log tee path was missing

现象：

```text
H3 repair launch printed:
    tee: .../matrix_logs/h3_repair_R1/...log: No such file or directory
```

处理：

```text
Created the missing matrix log directory after startup.
The LoGeR rollout itself continued and completed:
    run_status.txt = DONE
    trajectory / evaluation artifacts landed
```

边界：

```text
The shell wrapper session returned non-zero because tee failed, but the rollout
row is valid by run_status and landed trajectory metrics.
This is recorded as a logging-path blocker, not a trajectory failure.
```

---

## 3. H1：Local-to-Full Transfer Audit

输出：

```text
h1_local_to_full/report_R3/
h1_local_to_full/rollouts/V32_H1_R2_*
h1_local_to_full/rollouts/V32_H1_R3_*
```

References：

```text
H9_BASE:
    v31 full-online H9 base
V31_ALL_CHUNKS:
    v31 full-online SEM_Z_COARSE_BETA525 all chunks
```

Rows：

| Run | Full ATE | ATE delta vs H9 | `[200,300)` delta vs H9 | `[400,600)` delta vs H9 | Active chunks | Gate |
|---|---:|---:|---:|---:|---|---|
| `H9_BASE` | `36.8191643173` | `0.0` | `0.0` | `0.0` | `[]` | fail |
| `V31_ALL_CHUNKS` | `36.6905744722` | `-0.1285898451` | `-0.1845663526` | `-1.7120821455` | all / legacy | fail |
| `H1_FIXED_CH10_12` | `36.0801185755` | `-0.7390457418` | `-1.2521234262` | `-1.3071608818` | `[10,11,12]` | fail |
| `H1_FIXED_CH6_12` | `36.8427633206` | `+0.0235990033` | `-0.0042195864` | `+0.0545863006` | `[6..12]` | fail |
| `H1_RUNTIME_HIGHMASS` | `37.2398654661` | `+0.4207011488` | `+0.5545470559` | `+0.6658620319` | `[5,13,14,27,32]` | fail |
| `H1_RUNTIME_DMEAN` | `36.8743717283` | `+0.0552074110` | `+0.2707764157` | `+0.1112483383` | `[7,12,27,36]` | fail |
| `H1_RUNTIME_DQ90` | `37.1275329540` | `+0.3083686367` | `+0.4529562591` | `-0.8109244247` | `[5,19,27]` | fail |

Decision：

```text
H1 fixed chunk10-12 diagnostic recovers a small full-run improvement,
stronger than v31 all-chunks, but far below the v31 short-rollout local gain
and far above Target-30.

H1 runtime triggers did not recover the fixed-window effect.

Therefore:
    fixed chunk diagnostic supports local-to-full activation/state drift,
    but no deployable runtime trigger passed.
```

Boundary：

```text
H1_FIXED_CH10_12 is diagnostic only because it uses fixed KITTI01 chunk ids.
It cannot count as deployable online success.
```

---

## 4. H2：C9 Interaction Audit

输出：

```text
h2_c9_combo/report_R1_final/
h2_c9_combo/rollouts/V32_H2_*
```

Rows：

| Run | Full ATE | Delta vs C9 | `[200,300)` delta vs C9 | `[400,600)` delta vs C9 | Active chunks | Gate |
|---|---:|---:|---:|---:|---|---|
| `C9_REF` | `33.7629421029` | `0.0` | `0.0` | `0.0` | `[]` | fail Target-30 |
| `C9_SEM_ALL` | `34.5597307381` | `+0.7967886352` | `-0.0734166869` | `+1.6510089597` | all | fail |
| `C9_RUNTIME_DMEAN` | `34.8766743352` | `+1.1137322323` | `+0.5815532851` | `+4.1018725335` | `[7,12,27,36]` | fail |
| `C9_FIXED_CH10_12` | `34.2952287438` | `+0.5322866409` | `-1.1455007464` | `+3.1541071913` | `[10,11,12]` | fail |

C9 interaction gate：

```text
Required:
    improve C9 full ATE by >= 1.0m
    or improve [200,300) by >= 5m with [400,600) regression <= +1m

Observed best ATE row:
    C9_FIXED_CH10_12
    ATE delta vs C9 = +0.5322866409m
    [200,300) delta vs C9 = -1.1455007464m
    [400,600) delta vs C9 = +3.1541071913m

H2 gate = fail.
```

Decision：

```text
Semantic-conditioned C23 does not stack with the current best C9 protocol.
All tested C9+semantic rows are worse than C9 full ATE.
The fixed-window diagnostic improves [200,300) but causes large [400,600)
regression and remains non-deployable.
```

---

## 5. H3：Conservative Residual Repair

触发原因：

```text
H2 all-chunks semantic z worsened C9 full ATE and regressed [400,600).
Per v32 plan, try a more conservative normalization / residual cue rather than
expanding semantic action sweeps.
```

输出：

```text
h3_repair/report_R1/
h3_repair/rollouts/V32_H3_01_C9_RESID_COARSE_L025_ALL/
```

Row：

| Run | Full ATE | Delta vs C9 | `[200,300)` delta vs C9 | `[400,600)` delta vs C9 | Gate |
|---|---:|---:|---:|---:|---|
| `C9_RESID_COARSE_ALL` | `34.3258261120` | `+0.5628840091` | `-0.6684692189` | `+3.1115081606` | fail |

Decision：

```text
The conservative residual repair reduces the damage relative to C9_SEM_ALL,
but still worsens C9 full ATE and regresses [400,600).
H3 repair does not pass continuation gate.
```

---

## 6. Not Started / Not Claimed

Not started:

```text
1. No learned role router.
2. No no-GT selector.
3. No Phase 3 / Phase 4 all-memory matrix.
4. No H5 distributed static-anchor matrix.
5. No H6 semantic risk lifecycle matrix.
6. No cross-sequence runtime-trigger validation.
```

Reason：

```text
The v32 plan says if semantic-conditioned C23 fails in C9:
    stop semantic cue deployment line;
    keep semantic cue as diagnostic;
    test static anchor only if H5 has h15 signal;
    return Target-30 line to trajectory-state / scale-state.

Observed:
    H2 C9 interaction gate failed.
    No tested runtime trigger transferred local h15 effect to full online.
    No semantic full-online row improved C9.
```

Boundary：

```text
No v32 row counts as deployable online success.
No fixed chunk-id diagnostic counts as deployable.
No Target-30 result was produced.
```

---

## 7. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| H1 R1 | invalid | chunk_idx was not synchronized; fixed gate saw `-1` |
| H1 fixed chunk10-12 | diagnostic pass / deployable fail | improves H9 by `-0.7390m`, but fixed chunk id and ATE `36.0801m` |
| H1 runtime triggers | fail | all runtime triggers worsen H9 or fail to improve meaningfully |
| H2 C9 all chunks | fail | ATE `34.5597m`, worse than C9 |
| H2 C9 runtime D-mean | fail | ATE `34.8767m`, worse than C9 |
| H2 C9 fixed chunk10-12 | diagnostic fail | ATE `34.2952m`, `[400,600)` regression `+3.1541m` |
| H3 residual repair | fail | ATE `34.3258m`, worse than C9 |
| Full online Target-30 | fail | best v32 full ATE `34.2952m`, target requires `<=30m` |
| Deployable online success | no | C9 remains best deployable |

---

## 8. Final Decision

v32 的真实成功点：

```text
1. Implemented auditable v32 semantic cue fixed-window and runtime-trigger gating.
2. Fixed chunk index propagation so full-online HMC control priors know the
   correct chunk id.
3. Demonstrated that fixed chunk10-12 activation improves H9 full online more
   than all-chunks v31:
       H1_FIXED_CH10_12 ATE delta vs H9 = -0.7390457418m
       V31_ALL_CHUNKS ATE delta vs H9 = -0.1285898451m
4. Completed C9 interaction tests instead of stopping at H9-only evidence.
5. Completed a conservative residual repair attempt after C9 semantic z failed.
```

v32 的关键负结果：

```text
1. No runtime trigger transferred the local h10/h15 effect to full online.
2. Semantic-conditioned cue does not improve current best C9.
3. C9 + semantic rows all worsen full ATE:
       C9_SEM_ALL delta = +0.7967886352m
       C9_RUNTIME_DMEAN delta = +1.1137322323m
       C9_FIXED_CH10_12 delta = +0.5322866409m
       C9_RESID_COARSE_ALL delta = +0.5628840091m
4. Fixed chunk rows remain non-deployable diagnostics.
5. No v32 row reaches Target-30.
```

Interpretation：

```text
v32 sharpens the v31 conclusion:
    semantic cue reconditioning can be locally useful,
    and fixed-window full online diagnostics show some recoverable signal.

But the deployable path fails:
    runtime triggers choose the wrong/insufficient windows,
    all-chunk activation causes drift/regression,
    and the current best C9 TTT/SWA write path conflicts with semantic cue
    reconditioning rather than benefiting from it.

The local-to-full gap is therefore not solved by the tested semantic triggers.
```

Conclusion type：

```text
Semantic cue reconditioning remains a diagnostic / weak regularizer.
It should not be promoted as the Target-30 mainline from v32.
```

Next required direction：

```text
Do not launch selector / full validation from v32 semantic rows.
Do not continue broad semantic cue sweeps unless a new non-chunk-specific
trigger can first reproduce the fixed-window effect.

Per v32 failure routing, Target-30 mainline should return to:
    explicit online trajectory-state,
    explicit scale-state,
    merge/gauge-aware correction,
    or C9-native lifecycle / risk-state repair outside semantic labels.
```
