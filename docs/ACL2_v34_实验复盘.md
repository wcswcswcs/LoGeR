# ACL2 v34 实验复盘：SemanticCue RuntimeValueTrigger C9Compatibility Target30

日期：2026-05-23 至 2026-05-24（Asia/Singapore）  
计划文件：`docs/ACL2_v34_SemanticCue_RuntimeValueTrigger_C9Compatibility_Target30_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v34_semanticcue_value_trigger_c9compat_target30/`

本轮原则：只记录实际落盘结果；不把 snapshot 生成、reset-oracle short rollout、trigger learner 训练集拟合、repair 失败、或未启动 full online 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立 row 分配到不同 GPU。

---

## 0. 当前结论

v34 解决了 v33 的 parent snapshot coverage blocker，并完成了完整 reset-group H1 oracle；但 H2 no-GT runtime value trigger 在 leave-one-reset-group-out 验证中失败，因此按 v34 stop rule 没有启动 H3/H4/selector/full online。v34 未达到 Target-30，也没有产生新的 deployable online result。

已完成并落盘：

```text
1. 阅读 v34 计划，确认本轮核心是：
       先补齐 H9/C9 reset-group parent snapshots，
       再做完整 reset oracle，
       只有 no-GT value trigger 通过 held-out gate 才允许 C9-compatible full validation。

2. 生成 v34 H9/C9 full-sequence parent snapshots：
       parents = H9_V34_R1, C9_V34_R1
       reset starts = 0,5,10,15,20,25,30
       both rows DONE

3. Snapshot coverage audit 通过：
       complete_expected_reset_coverage = true
       trigger_training_allowed = true
       common_expected_reset_hits = [0,5,10,15,20,25,30]

4. H1 reset oracle bank 完成：
       parents = H9, C9
       horizons = h10, h15
       chunks = 0,5,10,15,20,25,30
       rollout dirs = 140/140
       DONE = 140
       FAIL = 0

5. H1 h15 找到多 reset positive：
       positive chunks = [0,10]
       H9/C9 均成立

6. H2 no-GT value trigger 完成：
       samples = 14
       positive samples = 4
       uses_absolute_chunk_id = false
       trained-all 2-atom rule fits training set
       leave-one-reset-group-out heldout positive recall = 0.0
       false positive rate = 0.4
       trigger_gate_pass = false

7. 按计划做 single-atom conservative repair：
       heldout positive recall = 0.0
       false positive rate = 0.6
       trigger_gate_pass = false

8. H2 gate = fail。
```

最终边界：

```text
1. v33 的 snapshot coverage blocker 已被 v34 修复。
2. H1 reset oracle 证明 semantic z coarse beta525 在 reset chunks 0 和 10 有 short diagnostic signal。
3. H1 h10 仍只在 chunk10 过 gate；h15 在 chunk0 和 chunk10 过 gate。
4. H2 value trigger 训练集可拟合，但 held-out reset group 完全召回失败。
5. Single-atom repair 失败且 false positive rate 更高。
6. No-GT runtime trigger 没有通过。
7. H3 C9 compatibility decomposition 未启动。
8. H4 full online validation 未启动。
9. No selector was launched.
10. No online Target-30 result was produced in v34.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v34 strengthened the reset-oracle evidence for semantic cue reconditioning,
but did not produce a deployable trigger.

Fixed/reset short rollout evidence is not deployable by itself.
The tested no-GT value triggers failed held-out reset-group validation.
Therefore no full online Target-30 claim is allowed.
```

---

## 1. 工程修改

新增：

```text
tools/v34_reset_oracle_report.py:
    aggregates v34 H1 reset-oracle short rollouts.
    Uses dynamic segment start based on chunk start frame.
    Computes:
        ATE delta vs same parent/chunk/horizon base
        [200,300) delta
        [400,600) delta
        gate pass candidates
        positive reset chunks
        multi-reset positive flag
    Writes per parent/horizon:
        v34_*_effects.csv
        v34_*_by_chunk.csv
        v34_*_summary.json

tools/v34_value_trigger_audit.py:
    trains/evaluates no-GT threshold value triggers from H1 oracle reports.
    Uses no absolute chunk id.
    Uses leave-one-reset-group-out validation.
    Writes:
        value_trigger_samples.csv
        value_trigger_loo_predictions.csv
        value_trigger_summary.json
```

临时调度脚本：

```text
/tmp/run_v34_snapshot_generation.sh:
    generates H9_V34_R1 and C9_V34_R1 parent snapshots.

/tmp/run_v34_reset_oracle.sh:
    launches independent H1 oracle rows across GPUs 0/1/2/3.
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v34_reset_oracle_report.py \
    tools/v34_value_trigger_audit.py

bash -n /tmp/run_v34_snapshot_generation.sh
bash -n /tmp/run_v34_reset_oracle.sh

PASS
```

边界：

```text
No production runtime code was changed for v34.
Existing v31 semantic cue aliases were reused.
```

---

## 2. Phase 0：Snapshot Generation / Coverage

输出：

```text
phase0_snapshot_coverage/
snapshot_coverage_audit/
```

Snapshot generation rows：

| Run | GPU | Start | Done | Status |
|---|---:|---|---|---|
| `V34_PHASE0_H9_SNAPSHOT_R1` | `0` | `2026-05-23 21:07:48` | `2026-05-23 21:42:39` | DONE |
| `V34_PHASE0_C9_SNAPSHOT_R1` | `1` | `2026-05-23 21:07:48` | `2026-05-23 21:42:29` | DONE |

Coverage audit summary：

| Metric | Value |
|---|---:|
| `parents` | `H9_V34_R1, C9_V34_R1` |
| `expected_reset_starts` | `[0,5,10,15,20,25,30]` |
| `common_available_chunks` | `[0,5,10,15,20,25,30]` |
| `common_expected_reset_hits` | `[0,5,10,15,20,25,30]` |
| `complete_expected_reset_coverage` | `true` |
| `trigger_training_allowed` | `true` |
| `warm_snapshot_chunks` | `[6,10,16]` |

Decision：

```text
Phase 0 snapshot coverage = pass.
The v33 reset-group coverage blocker is fixed for v34.
H1 complete reset oracle and H2 trigger audit are allowed.
```

---

## 3. H1：Reset Oracle Bank

输出：

```text
phase1_reset_oracle/rollouts/
phase1_reset_oracle/report_h10_H9/
phase1_reset_oracle/report_h15_H9/
phase1_reset_oracle/report_h10_C9/
phase1_reset_oracle/report_h15_C9/
```

设置：

```text
parents = H9, C9
horizons = h10, h15
chunks = 0,5,10,15,20,25,30
candidates per parent/chunk/horizon:
    V31_BASE_H9_REFERENCE
    V31_A0_ORIG_C23
    V31_A1B_SEM_Z_COARSE
    V31_A1_SEM_Z_FINE
    V31_A5B_SEM_RESID_COARSE_L025

rollout dirs = 140
DONE = 140
FAIL = 0
```

Report rows：

```text
Each parent/horizon report contains:
    7 base rows
    28 candidate effect rows
```

Gate：

```text
h10/h15 short diagnostic gate:
    ATE delta <= -1.5m
    or [200,300) delta <= -3m
    with [400,600) delta <= +1m
```

### H9 h10

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `positive_chunks` | `[10]` |
| `h1_multi_reset_positive` | `false` |
| Best ATE | `V31_A1_SEM_Z_FINE chunk5 = -0.7422223695m` |
| Best `[200,300)` | `V31_A1B_SEM_Z_COARSE chunk10 = -6.0251634306m` |
| Best ATE row `[400,600)` | `+0.0457528173m` |

H9 h10 chunk10：

| Candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---|---:|---:|---:|---|
| `V31_A1B_SEM_Z_COARSE` | `-0.5483503537` | `-6.0251634306` | `+0.6729546768` | pass |
| `V31_A1_SEM_Z_FINE` | landed pass | landed pass | landed pass | pass |

### H9 h15

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `positive_chunks` | `[0,10]` |
| `non_chunk10_positive_chunks` | `[0]` |
| `h1_multi_reset_positive` | `true` |
| Best ATE | `V31_A1B_SEM_Z_COARSE chunk0 = -1.6353453461m` |
| Best `[200,300)` | `V31_A1B_SEM_Z_COARSE chunk10 = -6.1564374201m` |
| Best ATE row `[400,600)` | `-1.7986829658m` |

H9 h15 key rows：

| Chunk | Best candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---:|---|---:|---:|---:|---|
| `0` | `V31_A1B_SEM_Z_COARSE` | `-1.6353453461` | landed positive | `-1.7986829658` | pass |
| `5` | `V31_A1_SEM_Z_FINE` | `-0.2931623029` | `-0.6326251680` | `+0.0758255460` | fail |
| `10` | `V31_A1B_SEM_Z_COARSE` | `-0.4809611282` | `-6.1564374201` | `+0.2109813077` | pass |

### C9 h10

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `positive_chunks` | `[10]` |
| `h1_multi_reset_positive` | `false` |
| Best ATE | `V31_A1_SEM_Z_FINE chunk5 = -0.7890634347m` |
| Best `[200,300)` | `V31_A1B_SEM_Z_COARSE chunk10 = -5.9916419717m` |
| Best ATE row `[400,600)` | `+0.0699194200m` |

C9 h10 chunk10：

| Candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---|---:|---:|---:|---|
| `V31_A1B_SEM_Z_COARSE` | `-0.5407172429` | `-5.9916419717` | `+0.6783464094` | pass |
| `V31_A1_SEM_Z_FINE` | landed pass | landed pass | landed pass | pass |

### C9 h15

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `positive_chunks` | `[0,10]` |
| `non_chunk10_positive_chunks` | `[0]` |
| `h1_multi_reset_positive` | `true` |
| Best ATE | `V31_A1B_SEM_Z_COARSE chunk0 = -1.6353453461m` |
| Best `[200,300)` | `V31_A1B_SEM_Z_COARSE chunk10 = -6.1250454121m` |
| Best ATE row `[400,600)` | `-1.7986829658m` |

C9 h15 key rows：

| Chunk | Best candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---:|---|---:|---:|---:|---|
| `0` | `V31_A1B_SEM_Z_COARSE` | `-1.6353453461` | landed positive | `-1.7986829658` | pass |
| `5` | `V31_A1_SEM_Z_FINE` | `-0.3163943751` | `-0.6937423463` | `+0.0573137323` | fail |
| `10` | `V31_A1B_SEM_Z_COARSE` | `-0.4737712828` | `-6.1250454121` | `+0.2168005677` | pass |

Decision：

```text
H1 reset oracle = diagnostic pass.

h15 produces two positive reset groups:
    chunks 0 and 10
for both H9 and C9 parents.

However:
    h10 remains positive only on chunk10,
    chunks 5/15/20/25/30 do not expose a strong semantic cue effect,
    this is still short-rollout oracle evidence, not deployable online success.
```

---

## 4. H2：No-GT Runtime Value Trigger

输出：

```text
phase2_value_trigger/
phase2_value_trigger_single_atom/
```

设置：

```text
Training/evaluation samples:
    parent/chunk h15 summaries from H9 and C9
    2 parents x 7 reset chunks = 14 samples

Positive labels:
    chunk 0 positive for H9/C9
    chunk 10 positive for H9/C9
    positive_samples = 4

Features:
    no-GT runtime/debug value features from H1 reports
    D patch mean/q90
    semantic z high mass
    semantic D mean/q90
    candidate-to-candidate value differences

Forbidden:
    absolute chunk id
```

### Two-atom trigger

Trained-all rule：

```text
coarse_minus_fine_mean_d >= -0.0002126246690750122
and coarse_minus_orig_q90_d <= 0.09732508659362793
```

Training-set confusion：

| TP | TN | FP | FN |
|---:|---:|---:|---:|
| `4` | `10` | `0` | `0` |

Leave-one-reset-group-out：

| Metric | Value |
|---|---:|
| `heldout_positive_recall` | `0.0` |
| `false_positive_rate` | `0.4` |
| `TP / TN / FP / FN` | `0 / 6 / 4 / 4` |
| `trigger_gate_pass` | `false` |

Failure pattern：

```text
Held-out chunk0:
    learned chunk10-like coarse_mean_d <= 0.4820718169
    missed both chunk0 positives.

Held-out chunk10:
    learned chunk0-like coarse_mean_d >= 0.4830932319
    missed both chunk10 positives.

Held-out chunks15/20:
    false positives appeared on both H9 and C9 rows.
```

### Single-atom repair

触发原因：

```text
Two-atom rule overfit reset groups despite fitting the full training set.
Per v34 repair direction, try a simpler conservative one-atom rule.
```

Trained-all rule：

```text
coarse_minus_orig_q90_d <= 0.09732508659362793
```

Training-set confusion：

| TP | TN | FP | FN |
|---:|---:|---:|---:|
| `4` | `8` | `2` | `0` |

Leave-one-reset-group-out：

| Metric | Value |
|---|---:|
| `heldout_positive_recall` | `0.0` |
| `false_positive_rate` | `0.6` |
| `TP / TN / FP / FN` | `0 / 4 / 6 / 4` |
| `trigger_gate_pass` | `false` |

Decision：

```text
H2 no-GT value trigger = fail.

The trigger can fit the complete H1 labels but cannot generalize when either
positive reset group is held out. The single-atom repair also fails and increases
false positives.

No deployable runtime trigger exists from v34.
```

---

## 5. Blocker 与修复记录

### Blocker 1：v33 reset-group coverage incomplete

现象：

```text
v33 common usable parent chunks:
    [5,10,16]

v34 plan requires reset starts:
    [0,5,10,15,20,25,30]
```

修复：

```text
Generate H9_V34_R1 and C9_V34_R1 full-sequence parent snapshots with:
    reset_every = 5
    save_hmc_state_snapshot_chunks = 0,5,10,15,20,25,30
    save_merge_state_snapshot_chunks = 0,5,10,15,20,25,30

Run H9 and C9 snapshot generation as independent one-GPU rows.
```

结果：

```text
complete_expected_reset_coverage = true
trigger_training_allowed = true
```

### Blocker 2：H2 value trigger overfits complete labels but fails held-out reset groups

现象：

```text
Two-atom trained-all rule:
    TP=4, TN=10, FP=0, FN=0

Leave-one-reset-group-out:
    heldout_positive_recall = 0.0
    false_positive_rate = 0.4
```

原因：

```text
Available no-GT value features separate the complete label table only when all
positive groups are visible. When either chunk0 or chunk10 is held out, learned
rules do not transfer to the other positive group.
```

按计划尝试的修复方向：

```text
Try a simpler one-atom value trigger instead of a broader semantic sweep.
Do not introduce absolute chunk id.
```

结果：

```text
Single-atom repair:
    heldout_positive_recall = 0.0
    false_positive_rate = 0.6
    trigger_gate_pass = false
```

Decision：

```text
No further v34 continuation is allowed without a new feature source or more
oracle evidence. Continuing to H3/H4/full online would use an overfit trigger
or fixed reset identity, which the plan forbids.
```

---

## 6. Not Started / Not Claimed

Not started：

```text
1. H3 C9 compatibility decomposition.
2. H4 C9-compatible full online validation.
3. Selector.
4. Additional all-memory matrix.
5. Cross-sequence runtime-trigger validation.
```

Reason：

```text
H2 no-GT value trigger failed.
No deployable trigger passed held-out reset-group validation.
```

Boundary：

```text
No v34 short rollout counts as deployable online TTT write success.
No fixed reset/chunk identity trigger is claimed.
No full online validation was launched.
No online Target-30 result was produced.
```

---

## 7. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Phase 0 snapshot generation | pass | H9/C9 v34 parent snapshots generated for all reset starts |
| Snapshot coverage audit | pass | common expected reset hits `[0,5,10,15,20,25,30]` |
| H1 h10 H9/C9 oracle | diagnostic pass | chunk10 passes; no multi-reset positive |
| H1 h15 H9/C9 oracle | diagnostic pass | chunks 0 and 10 pass |
| H2 two-atom trigger | fail | held-out positive recall `0.0`, FPR `0.4` |
| H2 single-atom repair | fail | held-out positive recall `0.0`, FPR `0.6` |
| H3/H4/full online | not started | no valid no-GT trigger |
| Target-30 | not produced | no full online candidate allowed |
| Deployable online success | no | C9 remains best deployable |

---

## 8. Final Decision

v34 的真实成功点：

```text
1. Fixed the v33 reset-group snapshot coverage blocker.
2. Completed a full H9/C9 reset oracle over chunks:
       0,5,10,15,20,25,30
3. Verified h15 semantic z coarse beta525 has reset-oracle positives on:
       chunk0 and chunk10
   under both H9 and C9 parents.
4. Verified C9-parent local compatibility remains real in the oracle setting.
5. Implemented and audited a no-absolute-chunk-id value trigger learner.
```

v34 的关键负结果：

```text
The no-GT value trigger does not generalize.

Two-atom trigger:
    training fit = perfect
    held-out positive recall = 0.0
    false positive rate = 0.4

Single-atom repair:
    held-out positive recall = 0.0
    false positive rate = 0.6

Therefore v34 cannot produce a deployable runtime trigger and cannot enter
full online validation.
```

Interpretation：

```text
v34 changes the v33 blocker status:
    lack of reset snapshots is no longer the reason we cannot continue.

The new blocker is scientific/runtime:
    the semantic cue oracle positives at reset chunks 0 and 10 do not share a
    no-GT value signature strong enough to survive held-out reset validation.

This means fixed/reset-window semantic cue effects are real, but the tested
value-trigger route cannot recover them deployably.
```

Conclusion type：

```text
Complete reset-oracle diagnostic success, no-GT runtime trigger failure.

Do not promote v34 as deployable online success.
Do not launch full online from v34 trigger rows.
```

Next required direction：

```text
Do not continue v34 through H3/H4/selector/full online without a new trigger
feature source or additional non-overfit oracle evidence.

If continuing this line, first add no-GT features that can plausibly distinguish
both positive reset groups without absolute chunk id:
    trajectory-state drift features,
    scale-state / gauge-risk features,
    C9-native lifecycle/risk-state features,
    or richer pre-action cue value summaries.

Otherwise Target-30 mainline should return to:
    explicit online trajectory-state,
    explicit scale-state,
    merge/gauge-aware correction,
    or C9-native lifecycle / risk-state repair outside semantic labels.
```
