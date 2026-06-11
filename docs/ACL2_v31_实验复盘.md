# ACL2 v31 实验复盘：SemanticCue Reconditioning DistributedMemory Target30

日期：2026-05-23（Asia/Singapore）  
计划文件：`docs/ACL2_v31_SemanticCue_Reconditioning_DistributedMemory_Target30_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v31_semanticcue_reconditioning_distributedmemory_target30/`

本轮原则：只记录实际落盘结果；不把 action audit、invalid/OOM row、short h10/h15、repair row、或 full online 失败写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立任务分配到不同 GPU。

---

## 0. 当前结论

v31 找到了一个可重复的 short-rollout 局部强信号，并通过一次小修复让 h15 durability gate 通过；但 full online 验证没有达到 Target-30，因此没有产生新的 deployable online result。

已完成并落盘：

```text
1. 阅读 v31 计划，并实现 semantic-conditioned C23 / D_g reconditioning 接线。
2. Track 0 R1 invalid:
       v31 aliases initially used noop, Stage-C semantic labels not loaded.
3. Track 0 R2 invalid/partial:
       scheduling mistake caused multiple heavy rows to share GPU and CUDA OOM.
4. Track 0 R3 clean smoke passed:
       semantic labels available
       semantic z / residual / original C23 cue produce distinguishable D_g maps
5. Track A h10 R1 completed:
       chunks = 6,10,16
       rows = 18/18
       gate pass candidates = V31_A1_SEM_Z_FINE and V31_A1B_SEM_Z_COARSE on chunk10
6. Track A h15 R1 completed:
       V31_A1_SEM_Z_FINE and bugged V31_A1B_SEM_Z_COARSE h15 did not pass.
7. Blocker fixed:
       coarse cue detection used ".coarse." and accidentally consumed L_sem_tok.
       fixed to detect "_coarse" and consume G_sem_tok.
8. Track A coarse-fix h10 R2 completed:
       rows = 6/6
       true coarse V31_A1B_SEM_Z_COARSE passed h10 gate on chunk10.
9. Track A coarse-fix h15 R2 completed:
       true coarse V31_A1B_SEM_Z_COARSE missed h15 gate by a small margin.
10. Minimal durability repair tried:
       V31_A1B_SEM_Z_COARSE beta 5.25
       h10 gate pass
       h15 durability gate pass
11. Full online R1 launched for:
       H9 read-only base
       SEM_Z_COARSE_BETA525 candidate
       both rows completed.
12. Full online Target-30 gate = fail.
```

最终边界：

```text
1. Best original h10 signal:
       V31_A1_SEM_Z_FINE chunk10
       [200,300) delta = -5.2260805866m
       ATE delta = -0.5002416554m
       [400,600) delta = +0.5411293958m

2. Original h15 did not pass:
       V31_A1_SEM_Z_FINE
       [200,300) delta = -4.5645019571m
       ATE delta = -0.0706371822m
       [400,600) delta = +0.6317412549m

3. True coarse after bug fix:
       h10 [200,300) delta = -4.9550490429m
       h15 [200,300) delta = -4.9470747902m
       h15 gate still fail.

4. Repair beta525:
       h10 [200,300) delta = -5.9984295787m
       h15 [200,300) delta = -6.1265519528m
       h15 [400,600) delta = +0.2170289959m
       h15 durability gate pass.

5. Full online:
       H9_BASE ATE = 36.8191643173m
       SEM_Z_COARSE_BETA525 ATE = 36.6905744722m
       delta vs H9_BASE = -0.1285898451m
       Target-30 gate fail.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v31 的 semantic cue repair 在 short h10/h15 上有真实局部效果，
但 full online ATE = 36.6905744722m，未达到 <= 30m。

因此 v31 不能计为 Target-30 deployable success。
```

---

## 1. 工程修改

新增 / 修改：

```text
loger/pipeline/hybrid_memory_controller.py:
    added _patch_labels_from_token_labels(...)
    added _semantic_z_recondition_patch(...)
    added read cue sources:
        v31.sem_z_fine.c23past
        v31.sem_z_coarse.c23past
        v31.sem_resid_fine_l025.c23past
        v31.sem_resid_coarse_l025.c23past
    bug fix:
        coarse cue now detects "_coarse" and uses G_sem_tok.

run_pipeline_abc_v2.py:
    forwards v31 semantic recondition debug fields into hmc_state_hash.jsonl.

tools/run_v24_candidate_rollout.sh:
    added v31 aliases:
        V31_BASE_H9_REFERENCE
        V31_A0_ORIG_C23
        V31_A1_SEM_Z_FINE
        V31_A1B_SEM_Z_COARSE
        V31_A5_SEM_RESID_FINE_L025
        V31_A5B_SEM_RESID_COARSE_L025
        V31_B0_STATIC_RESCUE_EXISTING

tools/v24_candidate_bank_report.py:
    added v31 family labels.

tools/v31_track_a_report.py:
    aggregates v31 h10/h15 cue-reconditioning rollouts and gates.

tools/v31_track_f_washout_attribution.py:
    lightweight h10/h15 durability attribution from landed JSONL summaries.

tools/v31_full_online_report.py:
    computes full-online ATE / segment metrics and Target-30 gate.
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    loger/pipeline/hybrid_memory_controller.py \
    run_pipeline_abc_v2.py \
    tools/v31_track_a_report.py \
    tools/v31_track_f_washout_attribution.py \
    tools/v31_full_online_report.py

bash -n \
    tools/run_v24_candidate_rollout.sh \
    tools/run_attention_cue_experiment.sh

PASS
```

---

## 2. Blocker 与修复记录

### Blocker 1：Track 0 R1 semantic labels unavailable

现象：

```text
V31 Track0 R1 semantic_group_summary:
    fine_label_available = false
    token_count = 0
    semantic z fallback ratio = 1.0
```

原因：

```text
Initial v31 aliases used semantic_role_policy = noop.
Stage-C / semantic prior cache was not forced into runtime.
```

修复：

```text
tools/run_v24_candidate_rollout.sh:
    v31 A0/A1/A5/B0 aliases now use:
        semantic_role_policy = fine_path_router_debug
        semantic_memory_paths = all

This loads runtime semantic labels while keeping the role policy debug/passive.
```

结果：

```text
Track 0 R3:
    semantic labels available
    fine_label_count = 7
    semantic z fallback ratio = 0
```

### Blocker 2：Track 0 R2 scheduling caused CUDA OOM

现象：

```text
R2 launched too many rows at once and placed multiple heavy rows on the same GPU.
Some rows failed with CUDA out of memory.
```

处理：

```text
R2 is invalid/partial and not used for any gate.
R3 and later experiments use wave scheduling:
    one LoGeR process per GPU
    independent tasks parallelized across GPUs 0/1/2/3
```

### Blocker 3：coarse cue consumed fine labels

现象：

```text
V31_A1B_SEM_Z_COARSE debug showed:
    prior_v31_semantic_label_field = L_sem_tok

Expected:
    G_sem_tok
```

原因：

```text
The implementation checked for ".coarse." in the read cue string, but the actual
cue name is "v31.sem_z_coarse.c23past".
```

修复：

```text
hybrid_memory_controller.py:
    label_field = "G_sem_tok" if "_coarse" in self.read_cue_source else "L_sem_tok"
```

结果：

```text
Track A coarse-fix R2 was rerun for affected coarse rows.
True coarse result is recorded separately and does not overwrite earlier bugged rows.
```

---

## 3. Track 0：Action / Cue Distinguishability Smoke

输出：

```text
matrix_logs/track0_smoke_R3/
rollouts/V31_TRACK0_SMOKE_R3_*
```

Clean R3 rows:

```text
V31_A0_ORIG_C23
V31_A1_SEM_Z_FINE
V31_A1B_SEM_Z_COARSE
V31_A5_SEM_RESID_FINE_L025
V31_A5B_SEM_RESID_COARSE_L025
V31_B0_STATIC_RESCUE_EXISTING
```

Key smoke evidence:

| Candidate group | mean D patch | q90 D patch | fallback |
|---|---:|---:|---:|
| A0 original C23 | `0.2148744315` | `0.7398606539` | `0.0` |
| semantic z | `0.4829037189` | `0.8421235681` | `0.0002741228` |
| semantic residual l025 | `0.2818817496` | `0.7652481794` | `0.0002741228` |
| B0 static rescue existing | `0.2100982368` | `0.7331724763` | `0.0` |

Decision:

```text
Track 0 cue/action distinguishability = pass.
Track A h10 is allowed.
```

---

## 4. Track A h10 R1：Semantic-conditioned C23

输出：

```text
track_a_h10_report_R1/
matrix_logs/track_a_h10_R1/
rollouts/V31_TRACKA_H10_R1_*
```

设置：

```text
chunks = 6,10,16
horizon = 10
rows = 18/18
missing_rows = 0
all_rows_done = true
```

Gate summary:

| Metric | Value |
|---|---:|
| h10 gate pass | `true` |
| gate pass candidates | `V31_A1_SEM_Z_FINE chunk10`, `V31_A1B_SEM_Z_COARSE chunk10` |
| best ATE delta | `-0.7358958756m` (`V31_A1_SEM_Z_FINE`, chunk6) |
| best `[200,300)` delta | `-5.2260805866m` (`V31_A1_SEM_Z_FINE`, chunk10) |
| downstream `[400,600)` for best chunk10 local row | `+0.5411293958m` |

Key chunk10 rows:

| Candidate | ATE delta | `[200,300)` delta | `[400,600)` delta |
|---|---:|---:|---:|
| `V31_A1_SEM_Z_FINE` | `-0.5002416554` | `-5.2260805866` | `+0.5411293958` |
| `V31_A1B_SEM_Z_COARSE` | `-0.5002416554` | `-5.2260805866` | `+0.5411293958` |
| `V31_A5_SEM_RESID_FINE_L025` | `-0.2483582230` | `-1.2272179132` | `-0.0674381017` |
| `V31_B0_STATIC_RESCUE_EXISTING` | `+0.0589303171` | `+0.1206484926` | `+0.0616193923` |

Boundary:

```text
The original coarse row above was later found to be bugged and actually used
L_sem_tok. It is retained as landed evidence but not treated as true coarse.
```

Decision:

```text
h10 gate = pass.
h15 confirmation is allowed for A1 semantic z.
```

---

## 5. Track A h15 R1：Original A1 Confirmation

输出：

```text
track_a_h15_report_R1/
matrix_logs/track_a_h15_R1/
rollouts/V31_TRACKA_H15_R1_*
```

设置：

```text
chunk = 10
horizon = 15
rows = 3/3 candidate rows plus H9 base
all_rows_done = true
```

Rows:

| Candidate | h15 ATE delta | h15 `[200,300)` delta | h15 `[400,600)` delta | Gate |
|---|---:|---:|---:|---|
| `V31_A1_SEM_Z_FINE` | `-0.0706371822` | `-4.5645019571` | `+0.6317412549` | fail |
| `V31_A1B_SEM_Z_COARSE` bugged | `-0.0706371822` | `-4.5645019571` | `+0.6317412549` | fail |
| `V31_A0_ORIG_C23` | `0.0` | `0.0` | `0.0` | fail |

Decision:

```text
Original A1 h15 gate = fail.
Required:
    h15 ATE delta <= -3m
    or h15 [200,300) delta <= -5m
Observed best:
    h15 [200,300) delta = -4.5645019571m
```

---

## 6. Track F：Washout Attribution

输出：

```text
track_f_washout_A1_FINE_R1/
track_f_washout_A1B_COARSE_R1/
```

Important boundary:

```text
No .pt tensor snapshots were available.
Evidence level = proxy_only_no_tensor_state_snapshots.
```

Fine A1 summary:

| Metric | h10 | h15 | Durability |
|---|---:|---:|---:|
| ATE delta | `-0.5002416554` | `-0.0706371822` | `0.1412061152` |
| `[200,300)` delta | `-5.2260805866` | `-4.5645019571` | `0.8734082594` |

Path proxy:

| Path | h10 total | h15 tail total | tail/h10 |
|---|---:|---:|---:|
| `ttt` | `0.3185427014` | `0.1377412269` | `0.4324105434` |
| `frame_attention_bias` | `16.2050579786` | `7.3509969711` | `0.4536236141` |
| `swa_source_replace` | `0.0` | `0.0` | `0.0` |

Interpretation:

```text
Common h10/h15 chunks hash match = 11/11.
h15 adds 5 tail chunks.
Available proxy evidence suggests the h15 tail continues frame-bias and TTT
side effects, while SWA source replacement remains zero.

This does not prove tensor-state overwrite, but supports trying a conservative
read-cue strength/persistence repair instead of SWA repair.
```

---

## 7. Coarse Cue Fix and Re-run

输出：

```text
track_a_h10_report_R2_coarse_fix/
track_a_h15_report_R2_coarse_fix/
matrix_logs/track_a_h10_R2_coarse_fix/
matrix_logs/track_a_h15_R2_coarse_fix/
```

True coarse h10:

| Candidate | Chunk | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta | Gate |
|---|---:|---:|---:|---:|---|
| `V31_A1B_SEM_Z_COARSE` | `10` | `-0.5024656204` | `-4.9550490429` | `+0.5446158898` | pass |
| `V31_A5B_SEM_RESID_COARSE_L025` | `10` | weaker | weaker | ok | fail |

True coarse h15:

| Candidate | h15 ATE delta | h15 `[200,300)` delta | h15 `[400,600)` delta | Gate |
|---|---:|---:|---:|---|
| `V31_A1B_SEM_Z_COARSE` | `-0.3807339205` | `-4.9470747902` | `+0.2596483441` | fail |

Decision:

```text
True coarse semantic z is real and stronger/more durable than the original fine
h15 result, but it still misses the h15 [200,300) gate by about 0.053m.
```

---

## 8. Minimal Repair：Coarse Semantic Z beta525

触发原因：

```text
True coarse h15 [200,300) delta = -4.9470747902m,
just above the -5m gate.
```

Repair:

```text
V31_A1B_SEM_Z_COARSE
BETA_VALUE = 5.25
H9 base remains beta = 4.75.
This is a small durability repair, not a broad threshold sweep.
```

输出：

```text
track_a_repair_beta525_h10_report_R1/
track_a_repair_beta525_h15_report_R1/
matrix_logs/track_a_repair_beta525_R1/
rollouts/V31_TRACKA_REPAIR_BETA525_R1_*
```

Rows:

| Horizon | ATE delta | `[200,300)` delta | `[400,600)` delta | Gate |
|---:|---:|---:|---:|---|
| h10 | `-0.5384379305` | `-5.9984295787` | `+0.6752013759` | pass |
| h15 | `-0.4705400793` | `-6.1265519528` | `+0.2170289959` | pass |

Durability:

```text
abs(h15/h10) by ATE = 0.873893...
abs(h15/h10) by [200,300) = 1.021359...
```

Decision:

```text
Short-rollout h15 durability gate = pass.
No deployable online result is claimed from this short rollout.
Full online validation is allowed.
```

---

## 9. Full Online R1

输出：

```text
full_online_R1/rollouts/
full_online_R1/report/
matrix_logs/full_online_R1/
```

设置：

```text
Rows:
    V31_FULL_R1_H9_BASE
    V31_FULL_R1_SEM_Z_COARSE_BETA525

Both rows:
    start_frame = 0
    end_frame = 10000
    no GT runtime action
    no offline trajectory rewrite
    no selector using GT
```

Full metrics:

| Run | Frames | Full ATE | Segment `[200,300)` | Segment `[400,600)` | Wall seconds |
|---|---:|---:|---:|---:|---:|
| `H9_BASE` | `1101` | `36.8191643173` | `78.9449969514` | `47.9161822387` | `1400` |
| `SEM_Z_COARSE_BETA525` | `1101` | `36.6905744722` | `78.7604305989` | `46.2041000931` | `1617` |

Delta vs H9 full base:

| Metric | Delta |
|---|---:|
| Full ATE | `-0.1285898451m` |
| `[200,300)` ATE | `-0.1845663526m` |
| `[400,600)` ATE | `-1.7120821455m` |

Decision:

```text
Full online Target-30 gate = fail.

Required:
    KITTI01 full ATE <= 30m

Observed:
    SEM_Z_COARSE_BETA525 full ATE = 36.6905744722m
```

Boundary:

```text
The short h10/h15 local correction did not transfer to full online Target-30.
No v31 result counts as deployable online success.
```

---

## 10. Not Started / Not Claimed

Not started:

```text
1. No learned role router was trained.
2. No no-GT selector was evaluated beyond fixed-rule full online.
3. No Phase 3 / Phase 4 all-memory matrix was launched.
4. No cross-sequence Track G was launched.
```

Reason:

```text
The only h15-passing v31 repair failed full online Target-30 by a wide margin.
The plan forbids claiming short rollout or local segment success as deployable.
```

---

## 11. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Track 0 R1 | invalid | semantic labels unavailable due noop alias |
| Track 0 R2 | invalid/partial | scheduling put multiple heavy rows on one GPU and caused CUDA OOM |
| Track 0 R3 | pass | semantic labels available; D_g maps distinguishable |
| Track A h10 R1 | pass | fine semantic z chunk10 `[200,300)` delta `-5.2261m` |
| Track A h15 R1 | fail | best `[200,300)` delta `-4.5645m`, gate requires `<= -5m` |
| Coarse cue fix | done | `_coarse` now consumes `G_sem_tok` |
| True coarse h10 | pass | `[200,300)` delta `-4.9550m` |
| True coarse h15 | fail | `[200,300)` delta `-4.9471m` |
| beta525 repair h10/h15 | pass | h15 `[200,300)` delta `-6.1266m`, downstream `+0.2170m` |
| Full online R1 | fail | full ATE `36.6906m`, target requires `<=30m` |
| Deployable online success | no | full target failed |

---

## 12. Final Decision

v31 的真实成功点：

```text
1. Semantic-conditioned C23 reconditioning produced the first strong local
   semantic cue signal in this family:
       h10 [200,300) delta <= -5m.

2. The coarse cue bug was found and fixed.

3. True coarse semantic z with a small beta525 repair passed both h10 and h15
   local durability gates:
       h15 [200,300) delta = -6.1265519528m
       h15 [400,600) delta = +0.2170289959m

4. The full online validation was actually launched and completed.
```

v31 的关键负结果：

```text
The h15-passing repair did not transfer to full online Target-30.

Full online:
    SEM_Z_COARSE_BETA525 ATE = 36.6905744722m
    Target-30 requires <= 30m

It also did not beat the current best deployable online TTT write:
    C9_P0_R2 ATE = 33.7629421029m
```

Interpretation：

```text
v31 changes the semantic-family conclusion:
    semantic cue reconditioning can create a strong local correction in h10/h15.

But it does not yet solve Target-30:
    the local correction is not enough when applied from the full online start.
    The full-run improvement over a read-only H9 base is only -0.1286m ATE.

Therefore semantic cue reconditioning remains a useful local diagnostic/regularizer,
not a deployable Target-30 mechanism in the current implementation.
```

Conclusion type：

```text
Short h10/h15 semantic cue success, full online Target-30 failure.

Do not promote v31 to deployable online success.
```

Next required direction：

```text
Do not claim Target-30 from v31.

If continuing from v31, the next work should explain why chunk10 h10/h15 parent
rollouts show strong local correction while full online from frame 0 collapses:
    full-run cue drift over early chunks,
    reset_every / chunk offset interaction,
    frame-bias accumulation,
    merge/gauge interaction,
    or combination with the current best TTT write path.

Otherwise Target-30 mainline should return to:
    explicit online trajectory-state,
    explicit scale-state,
    skip-aware lifecycle,
    read-cue source-skip outside semantic labels,
    merge/gauge-aware correction,
    or TTT-native causal actions.
```
