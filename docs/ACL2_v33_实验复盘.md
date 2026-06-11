# ACL2 v33 实验复盘：SemanticCue LocalToFull C9Compatibility Target30

日期：2026-05-23（Asia/Singapore）  
计划文件：`docs/ACL2_v33_SemanticCue_LocalToFull_C9Compatibility_Target30_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v33_semanticcue_localtofull_c9compat_target30/`

本轮原则：只记录实际落盘结果；不把 fixed chunk diagnostic、short rollout、incomplete reset-group oracle、blocked trigger learner、或未启动 full validation 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立 row 分配到不同 GPU。

---

## 0. 当前结论

v33 完成了可用 parent snapshot 窗口上的 H1 local-to-full / C9-compatibility short diagnostic。结果确认：

```text
1. v31/v32 的 semantic cue 局部信号不是偶然：
       chunk10 上 SEM_Z_COARSE_BETA525 在 H9 parent 和 C9 parent 下，
       h10/h15 都通过 short diagnostic gate。

2. C9 parent 下也能复现 chunk10 局部收益：
       C9 h15 chunk10 SEM_Z_COARSE_BETA525
       [200,300) delta = -6.0886783722m
       [400,600) delta = +0.2510547794m

3. 但信号只在可测窗口里的 chunk10 稳定成立：
       chunk5 只有弱 ATE 改善，未过 gate。
       chunk16 semantic z 明显回退。

4. H2 trigger learner 被 hard-block：
       expected reset starts = [0,5,10,15,20,25,30]
       common H9/C9 usable parent snapshot chunks = [5,10,16]
       common expected reset hits = [5,10]
       complete_expected_reset_coverage = false
```

因此：

```text
H1 available-window short diagnostic = pass for chunk10 only.
H2 no-GT trigger learner = not allowed, because oracle labels are incomplete
and would collapse to absolute chunk-id overfit.
H3 / H4 / selector / full online validation = not started.
No v33 online Target-30 result was produced.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v33 strengthens the local causal interpretation of semantic cue reconditioning:
    chunk10 fixed-window semantic z can help even from a C9 parent snapshot.

But v33 does not solve deployability:
    no non-absolute runtime trigger was trained,
    reset-group coverage is incomplete,
    no C9-compatible full online candidate was launched,
    no Target-30 result exists.
```

---

## 1. 工程修改

新增：

```text
tools/v33_snapshot_coverage_audit.py:
    audits available H9/C9 parent state + merge snapshots for v33 reset-window
    oracle generation.

    Writes:
        snapshot_coverage_audit/snapshot_coverage.csv
        snapshot_coverage_audit/snapshot_coverage_summary.json
        snapshot_coverage_audit/snapshot_coverage_report.md

    Returns non-zero when expected reset-group coverage is incomplete.
```

临时调度脚本：

```text
/tmp/run_v33_trackA_h15.sh
/tmp/run_v33_trackA_h10.sh
```

用途：

```text
1. 每个 LoGeR 进程只绑定一张 GPU。
2. 多个独立 H1 rows 分配到不同 GPU 并行。
3. C9 rows 显式加载 C9 parent hmc / merge snapshots。
4. H9 rows 使用 H9 parent snapshots。
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v33_snapshot_coverage_audit.py

bash -n /tmp/run_v33_trackA_h10.sh
bash -n /tmp/run_v33_trackA_h15.sh

PASS
```

边界：

```text
No production runtime code was changed for v33.
Existing v31 semantic cue aliases and tools/v31_track_a_report.py were reused.
```

---

## 2. Snapshot Coverage Audit

输出：

```text
snapshot_coverage_audit/
```

Audit summary：

| Metric | Value |
|---|---:|
| `expected_reset_starts` | `[0,5,10,15,20,25,30]` |
| `parent H9 available chunks` | `[5,6,9,10,12,16]` |
| `parent C9 available chunks` | `[5,10,16]` |
| `common_available_chunks` | `[5,10,16]` |
| `common_expected_reset_hits` | `[5,10]` |
| `complete_expected_reset_coverage` | `false` |
| `trigger_training_allowed` | `false` |
| `warm_snapshot_chunks` | `[6,10,16]` |

Decision：

```text
H1 can only be run as an available-window diagnostic on chunks 5,10,16.
It cannot be treated as the full reset-group oracle required by H2.

H2 no-GT trigger training is blocked because the oracle labels would be sparse,
biased, and effectively tied to absolute KITTI01 chunks.
```

---

## 3. Blocker 与修复记录

### Blocker 1：Reset-group parent snapshots are incomplete

现象：

```text
v33 plan requires reset-group starts:
    0,5,10,15,20,25,30

Observed common H9/C9 usable parent snapshots:
    5,10,16

Only expected reset starts hit by both H9/C9:
    5,10
```

按计划尝试的修复方向：

```text
1. 扫描 /tmp/loger_v23_warm/snapshots。
2. 扫描 v16 phase1_causalfork H9/C9 state snapshots。
3. 扫描 matching merge_state_snapshots。
4. 只允许使用同时具备 HMC state + merge state 的 parent chunks。
```

结果：

```text
No complete reset-group parent snapshot set was found.
No synthetic or fabricated parent snapshots were created.
```

处理：

```text
Run H1 only on available chunks:
    5,10,16

Block H2 trigger learner and downstream deployable trigger/full validation.
```

### Blocker 2：Direct temp script execution failed

现象：

```text
Direct /tmp/run_v33_trackA_h15.sh execution failed due permission/noexec style
execution behavior.
```

修复：

```text
Invoke through bash:
    bash /tmp/run_v33_trackA_h15.sh
    bash /tmp/run_v33_trackA_h10.sh
```

结果：

```text
H1 h10/h15 rows completed:
    rows = 60/60
    failures = 0
```

### Blocker 3：C9 wrapper metadata still says H9 parent in run_config

现象：

```text
The reused rollout wrapper keeps some run_config labels as H9parent.
```

Audit：

```text
C9 row hmc_config.yaml records actual loaded parent files:
    state_snapshots/C9_P0_V16_R2/chunk_*.pt
    merge_state_snapshots/C9_P0_V16_R2/chunk_*.pt
```

处理：

```text
Treat wrapper run_config parent label as metadata limitation.
Use hmc_config loaded-state paths and landed trajectory metrics as validity evidence.
```

---

## 4. H1 Track A：Available-window h10

输出：

```text
trackA_h10_report_R1_H9/
trackA_h10_report_R1_C9/
rollouts/V33_TRACKA_H10_R1_*
```

设置：

```text
parents = H9, C9
chunks = 5,10,16
horizon = 10
candidates per parent/chunk:
    V31_A0_ORIG_C23
    V31_A1B_SEM_Z_COARSE   (beta = 5.25)
    V31_A1_SEM_Z_FINE      (beta = 4.75)
    V31_A5B_SEM_RESID_COARSE_L025

rows = 30/30
failures = 0
```

H9 h10 summary：

| Chunk | Best candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---:|---|---:|---:|---:|---|
| `5` | `V31_A1_SEM_Z_FINE` | `-0.7597608599` | `-1.0318304065` | `+0.0403095500` | fail |
| `10` | `V31_A1B_SEM_Z_COARSE` | `-0.5384379305` | `-5.9984295787` | `+0.6752013759` | pass |
| `16` | `V31_A0_ORIG_C23` | `0.0` | `nan` | `0.0` | fail |

H9 chunk10 key rows：

| Candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---|---:|---:|---:|---|
| `V31_A1B_SEM_Z_COARSE` | `-0.5384379305` | `-5.9984295787` | `+0.6752013759` | pass |
| `V31_A1_SEM_Z_FINE` | `-0.5002416554` | `-5.2260805866` | `+0.5411293958` | pass |
| `V31_A5B_SEM_RESID_COARSE_L025` | `-0.3023297413` | `-1.2649612255` | `-0.1130789383` | fail |

C9 h10 summary：

| Chunk | Best candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---:|---|---:|---:|---:|---|
| `5` | `V31_A1_SEM_Z_FINE` | `-0.7553786540` | `-1.0166113410` | `+0.0305457828` | fail |
| `10` | `V31_A1B_SEM_Z_COARSE` | `-0.5288888861` | `-5.9940499745` | `+0.6992505165` | pass |
| `16` | `V31_A0_ORIG_C23` | `0.0` | `nan` | `0.0` | fail |

C9 chunk10 key rows：

| Candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---|---:|---:|---:|---|
| `V31_A1B_SEM_Z_COARSE` | `-0.5288888861` | `-5.9940499745` | `+0.6992505165` | pass |
| `V31_A1_SEM_Z_FINE` | `-0.5077497526` | `-5.1848167858` | `+0.5311506394` | pass |
| `V31_A5B_SEM_RESID_COARSE_L025` | `-0.2876428068` | `-1.1925743862` | `-0.1021598552` | fail |

Decision：

```text
h10 available-window diagnostic passes on chunk10 for both H9 and C9 parents.
No other available chunk passes.
```

---

## 5. H1 Track A：Available-window h15

输出：

```text
trackA_h15_report_R1_H9/
trackA_h15_report_R1_C9/
rollouts/V33_TRACKA_H15_R1_*
```

设置：

```text
parents = H9, C9
chunks = 5,10,16
horizon = 15
rows = 30/30
failures = 0
```

H9 h15 summary：

| Chunk | Best candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---:|---|---:|---:|---:|---|
| `5` | `V31_A1_SEM_Z_FINE` | `-0.3048190193` | `-0.6568653974` | `+0.0632445845` | fail |
| `10` | `V31_A1B_SEM_Z_COARSE` | `-0.4705400793` | `-6.1265519528` | `+0.2170289959` | pass |
| `16` | `V31_A5B_SEM_RESID_COARSE_L025` | `-0.0187650589` | `nan` | `-0.0956840461` | fail |

H9 chunk10 key rows：

| Candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---|---:|---:|---:|---|
| `V31_A1B_SEM_Z_COARSE` | `-0.4705400793` | `-6.1265519528` | `+0.2170289959` | pass |
| `V31_A1_SEM_Z_FINE` | `-0.0706371822` | `-4.5645019571` | `+0.6317412549` | pass |
| `V31_A5B_SEM_RESID_COARSE_L025` | `-0.2492519164` | `-1.3311665565` | `-0.1249384540` | fail |

C9 h15 summary：

| Chunk | Best candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---:|---|---:|---:|---:|---|
| `5` | `V31_A1_SEM_Z_FINE` | `-0.3031799491` | `-0.6399667673` | `+0.0578259261` | fail |
| `10` | `V31_A1B_SEM_Z_COARSE` | `-0.4481391219` | `-6.0886783722` | `+0.2510547794` | pass |
| `16` | `V31_A5B_SEM_RESID_COARSE_L025` | `-0.0173959493` | `nan` | `-0.0870260193` | fail |

C9 chunk10 key rows：

| Candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---|---:|---:|---:|---|
| `V31_A1B_SEM_Z_COARSE` | `-0.4481391219` | `-6.0886783722` | `+0.2510547794` | pass |
| `V31_A1_SEM_Z_FINE` | `-0.0732507738` | `-4.5226188781` | `+0.6271374587` | pass |
| `V31_A5B_SEM_RESID_COARSE_L025` | `-0.2348993708` | `-1.2504655317` | `-0.1110389610` | fail |

Decision：

```text
h15 durability repeats the h10 pattern:
    chunk10 passes under both H9 and C9 parents.
    chunk5 does not pass.
    chunk16 semantic z is not useful and can regress.

This supports a real local semantic cue effect, but it is still fixed-window
diagnostic evidence.
```

---

## 6. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Snapshot coverage audit | fail / blocker | common expected reset hits only `[5,10]`, full reset groups missing |
| H1 available-window h10 | pass diagnostic | H9/C9 chunk10 pass; chunks 5/16 do not |
| H1 available-window h15 | pass diagnostic | H9/C9 chunk10 pass; chunks 5/16 do not |
| H2 trigger learner | not started | incomplete oracle labels; would overfit absolute chunks |
| H3 C9 compatibility decomposition | not started | no deployable trigger / reset-relative oracle from H2 |
| H4 residual/gated semantic C23 | not started | H2 blocked and v32 residual repair already failed in full C9 |
| Full online validation | not started | no non-absolute trigger or C9-compatible candidate passed continuation gate |
| Target-30 | fail / not produced | no v33 full online candidate launched |

Boundary：

```text
No v33 short rollout counts as deployable online TTT write success.
No absolute chunk-id trigger was trained or evaluated as deployable.
No selector was launched.
No full online validation was launched.
No online Target-30 result was produced in v33.
```

---

## 7. Final Decision

v33 的真实成功点：

```text
1. Built a v33-specific snapshot coverage audit and recorded the parent-state
   blocker explicitly.

2. Completed all available-window H1 short diagnostics:
       h10 rows = 30/30, failures = 0
       h15 rows = 30/30, failures = 0

3. Confirmed SEM_Z_COARSE_BETA525 chunk10 is durable under H9 parent:
       h15 [200,300) delta = -6.1265519528m
       h15 [400,600) delta = +0.2170289959m

4. Confirmed the same chunk10 effect survives under C9 parent:
       h15 [200,300) delta = -6.0886783722m
       h15 [400,600) delta = +0.2510547794m
```

v33 的关键负结果：

```text
1. The local effect is not broadly reset-relative in the available windows:
       chunk5 does not pass.
       chunk16 semantic z regresses.

2. Full reset-group parent snapshots are missing:
       no complete H9/C9 parent coverage for 0,5,10,15,20,25,30.

3. H2 trigger learner cannot be run honestly:
       training on only [5,10,16] with only chunk10 positive would be
       an absolute chunk-id proxy, forbidden by the plan.

4. No deployable runtime trigger, C9-compatible full candidate, selector, or
   Target-30 online result was produced.
```

Interpretation：

```text
v33 sharpens the v32 conclusion:
    semantic cue reconditioning has a real local causal signal,
    and that signal can exist even on top of C9 parent state in chunk10.

But v33 also shows why continuing would be unsafe without more parent-state
coverage:
    the only strong signal is still a fixed-window diagnostic,
    not a learned or deployable trigger.

The correct result is therefore not "semantic cue solved C9 compatibility";
it is "semantic cue has C9-parent local compatibility at chunk10, but deployable
local-to-full transfer remains blocked."
```

Conclusion type：

```text
Local C9-parent short diagnostic success, deployable Target-30 failure / blocked.

Do not promote v33 as deployable online success.
```

Next required direction：

```text
Do not train H2 trigger from the current incomplete oracle labels.
Do not launch selector or full online from v33 fixed-window rows.

If continuing this line, first generate or recover complete H9/C9 parent
state + merge snapshots for reset groups:
    0,5,10,15,20,25,30

Then rerun H1 as a true reset-group oracle and only train a no-GT trigger if:
    it does not use absolute chunk id,
    it has enough positive/negative windows,
    and it passes held-out precision/mean-gain gates.

Otherwise Target-30 mainline should continue through:
    explicit online trajectory-state,
    explicit scale-state,
    merge/gauge-aware correction,
    or C9-native lifecycle / risk-state repair outside semantic labels.
```
