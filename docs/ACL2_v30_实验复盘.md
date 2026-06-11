# ACL2 v30 实验复盘：SemanticCue CausalMemoryBank Target30

日期：2026-05-23（Asia/Singapore）  
计划文件：`docs/ACL2_v30_SemanticCueCausalMemoryBank_Target30_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v30_semanticcue_causalmemorybank_target30/`

本轮原则：只记录实际落盘结果；不把 action audit、partial/OOM row、short h10 causal-bank、blocked gate、或未启动矩阵写成 deployable online success。没有 masklet causal oracle upper bound 时，不训练 learned role router，不启动 h15、Phase 3/4、selector 或 full online validation。

---

## 0. 当前结论

v30 已按计划先执行 Track A action distinguishability 和 Track E diverse masklet causal bank wave1。Track E h10 oracle gate 未通过，因此按 v30 stop rule 早停，未达到 Target-30。

已完成并落盘：

```text
1. 阅读 v30 计划，确认本轮不再扩大 semantic rule sweep，而是先测：
       action distinguishability
       diverse masklet causal oracle
       oracle 成立后才允许 learned router / h15 / full validation

2. 新增 v30 diverse masklet selector：
       tools/v30_select_diverse_masklet_bank.py
   从 v29C masklet-3D alignment 中选择 chunk10 diverse masklets。

3. 新增 Track A action distinguishability audit：
       tools/v30_action_distinguishability_audit.py
   Track A gate pass:
       same_path_action_distinguishability_gate_pass = true

4. 新增 v30 runtime causal-bank report：
       tools/v30_masklet_causal_bank_report.py

5. 扩展 rollout launcher:
       V30_BASE_H9_REFERENCE
       V30_MASKLET_FRAME_SKIP
       V30_MASKLET_GLOBAL_SKIP
       V30_MASKLET_SWA_ANCHOR
       V30_MASKLET_SWA_REMOVE
       V30_MASKLET_TTT_POS
       V30_MASKLET_TTT_NEG

6. Track E h10 R1 首次并发运行触发资源 blocker：
       CSV CRLF leaked into run_prefix
       4-way concurrent TTT/SWA rows caused CUDA OOM
   R1 marked invalid / partial and not used for gate.

7. 修复 CSV line ending，并用 single-GPU serial R2 重跑：
       base + 16 interventions
       rows reported = 16/16
       missing_or_invalid_rows = 0
       all rows done = true
       context_empty_source_events_total = 0

8. 补做 SWA boundary diagnostic：
       SWA candidates = 5
       boundary_gate_pass_candidates = []

9. Track E oracle gate = fail。
```

最终边界：

```text
1. Track A action distinguishability passed offline.
2. Track E diverse causal bank h10 completed clean R2.
3. Best h10 ATE delta = -0.3883615963m.
4. Best h10 [200,300) delta = -2.4341499157m.
5. Best h10 [400,600) delta for best ATE row = -0.3470529934m.
6. No candidate met oracle:
       h10 ATE delta <= -3m
       or h10 [200,300) delta <= -5m
7. SWA rows again collapsed to identical trajectory deltas across masklets/actions:
       ATE delta = -0.3641983213m
       [200,300) delta = -2.3655037795m
8. SWA boundary metrics regressed:
       mean boundary 10f delta = +0.4789576412m
       mean boundary 20f delta = +0.3681139080m
9. Track B semantic-conditioned C23 was not launched after Track E stop rule.
10. Track C/D/G were not launched.
11. Learned role router was not trained.
12. No h15 semantic candidate was launched.
13. No no-GT selector was launched.
14. No full online validation was launched.
15. No online Target-30 result was produced in v30.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v30 uses v29C projected SemanticKITTI sparse anchors and masklet-3D trust data
as causal-bank selection evidence.

uses_gt_projected_3d_semantic = true for sparse diagnostic anchors
uses_video_masklet_semantic_as_gt = false

The completed v30 trajectory rows are short h10 causal-oracle diagnostics.
They do not count as deployable online success.
```

---

## 1. 工程修改

新增 / 修改：

```text
tools/v30_select_diverse_masklet_bank.py:
    selects diverse chunk10 masklets from v29C masklet_alignment.csv.
    Records the feature gap when per-masklet D_g/conflict/scale/source-attention
    features are unavailable.
    Writes:
        track_e_masklet_selection/selected_masklets.csv
        track_e_masklet_selection/planned_interventions.csv
        track_e_masklet_selection/selection_summary.json

tools/v30_action_distinguishability_audit.py:
    offline Track A action distinguishability audit.
    Writes:
        track_a_action_audit/action_tensor_summary.csv
        track_a_action_audit/action_jaccard_matrix.csv
        track_a_action_audit/action_audit_summary.json

tools/v30_masklet_causal_bank_report.py:
    scans v30 R2 rollout dirs with per-row run prefixes.
    Joins planned_interventions.csv with landed trajectory/hmc artifacts.
    Computes:
        ATE delta vs clean H9 reference
        [200,300) delta
        [400,600) delta
        runtime intervention metadata
        oracle gate
    Writes:
        track_e_causal_bank_h10_report_R2/causal_effects.csv
        causal_effects_by_path.csv
        causal_effects_by_category.csv
        causal_bank_summary.json

tools/run_v24_candidate_rollout.sh:
    added V30 aliases:
        V30_BASE_H9_REFERENCE
        V30_MASKLET_FRAME_SKIP
        V30_MASKLET_GLOBAL_SKIP
        V30_MASKLET_SWA_ANCHOR
        V30_MASKLET_SWA_REMOVE
        V30_MASKLET_TTT_POS
        V30_MASKLET_TTT_NEG

tools/v24_candidate_bank_report.py:
    added V30 candidate family labels.
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v30_select_diverse_masklet_bank.py \
    tools/v30_action_distinguishability_audit.py \
    tools/v30_masklet_causal_bank_report.py \
    tools/v24_candidate_bank_report.py

bash -n \
    tools/run_v24_candidate_rollout.sh \
    tools/run_attention_cue_experiment.sh

PASS
```

---

## 2. Track A：Action Distinguishability Audit

输出：

```text
track_a_action_audit/
track_e_masklet_selection/
```

Selection summary：

| Metric | Value |
|---|---:|
| `selected_masklets` | `7` |
| `planned_interventions` | `16` |
| `selection_gate_pass` | `true` |
| `available_alignment_rows` | `7` |

Selected masklets：

| Category | Masklet | Video label | Projected majority | Support | Agreement | `q_3d` | `t_mask` |
|---|---:|---|---|---:|---|---:|---:|
| `road_high_support` | `0` | road | road | `305184` | `true` | `0.8145409271` | `0.9422205008` |
| `structure_high_support` | `2` | fence | terrain | `995` | `false` | `0.2857073556` | `0.4178862455` |
| `vegetation_high_support` | `5` | vegetation | fence | `102505` | `false` | `0.3904690060` | `0.6513583783` |
| `grass_or_terrain` | `3` | grass | terrain | `35571` | `true` | `0.6075571519` | `0.7330787086` |
| `disagreement` | `6` | wall | terrain | `949` | `false` | `0.3232454732` | `0.2671219785` |
| `low_trust` | `1` | building | fence | `259` | `true` | `0.2190729924` | `0.7424614004` |
| `large_area` | `4` | sky | fence | `430` | `unknown` | `0.2018683878` | `0.9959967381` |

Boundary：

```text
Per-masklet D_g / conflict / scale-risk / source-attention features were not
present in the landed v29C alignment table. Wave1 therefore uses only landed
alignment/trust/projection proxies. This is recorded in selection_summary.json
and selected_masklets.csv.

large_area was selected but not acted in wave1 because the first-wave action
budget was capped at 16 interventions.
```

Action audit summary：

| Metric | Value |
|---|---:|
| `same_path_action_distinguishability_gate_pass` | `true` |
| `planned_interventions` | `16` |
| `selected_masklets` | `7` |

Action mass：

| Path | Action | Masklets | Ratio |
|---|---|---:|---:|
| global | `source_skip` | `6` | `0.8571428571` |
| SWA | `swa_anchor_keep` | `2` | `0.2857142857` |
| SWA | `swa_remove` | `3` | `0.4285714286` |
| TTT | `ttt_positive` | `2` | `0.2857142857` |
| TTT | `ttt_negative` | `3` | `0.4285714286` |

Decision：

```text
Track A offline action distinguishability = pass.

Boundary:
    This only proves planned action tensors differ offline.
    It does not prove runtime hook effects or trajectory improvement.
```

---

## 3. Blocker 与修复记录

### Blocker 1：CSV CRLF leaked into run_prefix

现象：

```text
R1 rollout directory names contained a control character before the candidate id:
    ...source_skip\r_V30_MASKLET_GLOBAL_SKIP...
```

原因：

```text
Python csv.DictWriter default lineterminator is CRLF.
The shell loop read the final CSV column with a trailing \r.
```

修复：

```text
tools/v30_select_diverse_masklet_bank.py:
tools/v30_action_distinguishability_audit.py:
    csv.DictWriter(..., lineterminator="\\n")

R2 loop also stripped possible \r before launching rows.
```

结果：

```text
R2 rollout dirs are clean.
R1 is retained only as invalid/partial blocker evidence.
```

### Blocker 2：R1 4-way concurrent h10 wave caused CUDA OOM

现象：

```text
V30_H4_WAVE1_R1_vegetation_high_support_m05_global_source_skip failed with:
    torch.OutOfMemoryError: CUDA out of memory

Several R1 TTT/SWA rows were left without DONE/FAIL after the batch exited.
```

原因：

```text
4-way concurrent execution placed heavy TTT/SWA rows on memory-constrained GPUs.
This is a resource scheduling failure, not trajectory evidence.
```

修复：

```text
R2 rerun:
    single GPU
    serial execution
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

结果：

```text
R2 base + 16 interventions completed:
    failed = 0
    rows_reported = 16/16
    missing_or_invalid_rows = 0
```

R1 decision：

```text
R1 is invalid / partial and not used for gate or conclusions.
```

---

## 4. Track E：Diverse Masklet Causal Bank h10

输出：

```text
track_e_causal_bank_h10_report_R2/
rollouts/V30_H4_WAVE1_R2_*
```

设置：

```text
chunk = 10
horizon = 10
reference = V30_H4_WAVE1_R2_base_V30_BASE_H9_REFERENCE
rows expected = 16
rows reported = 16
missing_or_invalid_rows = 0
all_rows_done = true
context_empty_source_events_total = 0
```

Oracle gate：

```text
Required:
    h10 ATE delta <= -3m
    or h10 [200,300) delta <= -5m
    with [400,600) regression <= +1m
    and context_empty_source_events = 0

Observed:
    best h10 ATE delta = -0.3883615963m
    best h10 [200,300) delta = -2.4341499157m
    best [400,600) delta for best ATE row = -0.3470529934m

oracle_gate_pass = false
```

Best rows：

| Category | Masklet | Path | Action | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta |
|---|---:|---|---|---:|---:|---:|
| `road_high_support` | `0` | global | `source_skip` | `-0.3883615963` | `-2.4341499157` | `-0.3470529934` |
| `road_high_support` | `0` | SWA | `swa_anchor_keep` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` |
| `structure_high_support` | `2` | SWA | `swa_anchor_keep` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` |
| `vegetation_high_support` | `5` | SWA | `swa_remove` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` |
| `grass_or_terrain` | `3` | SWA | `swa_remove` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` |
| `disagreement` | `6` | SWA | `swa_remove` | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` |
| `vegetation_high_support` | `5` | global | `source_skip` | `-0.3492055299` | `-2.1585452359` | `-0.3296916584` |
| `structure_high_support` | `2` | TTT | `ttt_positive` | `-0.1717134765` | `-0.2475952837` | `-0.1894172419` |

By path：

| Path | Rows | Best ATE delta | Best `[200,300)` delta |
|---|---:|---:|---:|
| global | `6` | `-0.3883615963` | `-2.4341499157` |
| SWA | `5` | `-0.3641983213` | `-2.3655037795` |
| TTT | `5` | `-0.1717134765` | `-0.2475952837` |

Interpretation：

```text
The diverse masklet bank found the same scale of signal as v29C:
    weak-to-moderate local [200,300) improvement through global/SWA,
    far below oracle gate,
    no TTT row close to oracle threshold.

The best global row is essentially the same weak signal scale as v29C top-support global:
    v29C best global ATE delta ~= -0.3935m
    v30 R2 best global ATE delta = -0.3884m
This is effectively the same weak signal scale, not a new upper bound.
```

Decision：

```text
Track E h10 causal oracle = fail.
No h15 confirmation is allowed.
No learned role router is allowed.
No no-GT selector or full online validation is allowed.
```

---

## 5. SWA Boundary Diagnostic

触发原因：

```text
SWA rows were among the best local [200,300) rows, and v30 failure routing
requires boundary checks when SWA local improvement appears.
```

输出：

```text
track_e_swa_boundary_report_R2/
```

SWA summary：

| Metric | Value |
|---|---:|
| `num_candidates` | `5` |
| `boundary_gate_pass_candidates` | `[]` |
| `mean_boundary_10f_delta_vs_H9` | `+0.4789576412` |
| `mean_boundary_20f_delta_vs_H9` | `+0.3681139080` |
| `mean_pose_jump_delta_vs_H9` | `-0.0186129478` |

Important observation：

```text
All five SWA rows have identical trajectory and boundary metrics:
    road anchor
    structure anchor
    vegetation remove
    grass remove
    disagreement remove

This repeats the v29C SWA collapse:
    SWA trajectory effect is dominated by shared hook/source behavior,
    not by differentiated masklet identity or anchor/remove action identity.
```

Decision：

```text
SWA boundary gate = fail.
No SWA row can be promoted.
```

---

## 6. Not Started / Not Claimed

Not started due to Track E oracle fail and v30 stop rule:

```text
1. Track B semantic-conditioned C23 h10.
2. Track C static scale-anchor h10.
3. Track D risk short-negative h10.
4. Track F learned role router.
5. Track G washout attribution.
6. h15 confirmation.
7. Phase 3 / Phase 4 all-memory matrices.
8. No-GT selector.
9. Full online validation.
```

Reason：

```text
The plan explicitly says to stop early if:
    masklet oracle has no upper bound;
    no candidate achieves h10 ATE <= -1.5m or [200,300) <= -3m;
    SWA boundary regression dominates.

Observed:
    no oracle upper bound,
    best ATE = -0.3884m,
    best [200,300) = -2.4341m,
    SWA boundary 10f/20f regressed.
```

Boundary：

```text
No v30 short rollout counts as deployable online TTT write success.
No learned router was trained.
No h15 / selector / full online validation was launched.
No online Target-30 result was produced.
```

---

## 7. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Track A action distinguishability | pass | planned same-path action sets differ offline |
| Track E R1 | invalid | CRLF run_prefix pollution and concurrent CUDA OOM |
| Track E R2 h10 | fail | best ATE `-0.3884m`, best `[200,300)` `-2.4341m`, oracle requires `-3m` / `-5m` |
| SWA boundary diagnostic | fail | boundary 10f / 20f regress by `+0.4790m` / `+0.3681m` |
| Track B semantic-conditioned C23 | not started | stop rule triggered by no masklet oracle upper bound |
| Track C static anchors | not started | stop rule triggered |
| Track D risk short-negative | not started | stop rule triggered |
| Track F learned router | not started | Track E oracle failed |
| h15 confirmation | not started | h10 oracle failed |
| No-GT selector | not started | no candidate gate |
| Full online validation | not started | no selector/full-run entry |

---

## 8. Final Decision

v30 的真实成功点：

```text
1. Built a v30 diverse masklet selection and action audit pipeline.
2. Verified planned action tensors differ offline.
3. Repaired CSV/run-prefix and resource scheduling blockers.
4. Completed clean R2 h10 diverse masklet causal bank:
       16/16 rows
       failures = 0
5. Produced a v30-specific causal report with runtime intervention metadata.
6. Confirmed context empty source events remained zero.
```

v30 的关键负结果：

```text
Diverse masklet causal bank still has no Target-30-scale upper bound.

Best h10:
    road_high_support masklet 0
    global source_skip
    ATE delta = -0.3883615963m
    [200,300) delta = -2.4341499157m
    [400,600) delta = -0.3470529934m

Best SWA:
    identical across 5 SWA masklet/action rows
    ATE delta = -0.3641983213m
    [200,300) delta = -2.3655037795m
    boundary 10f delta = +0.4789576412m
    boundary 20f delta = +0.3681139080m

Best TTT:
    structure_high_support masklet 2
    ttt_positive
    ATE delta = -0.1717134765m
    [200,300) delta = -0.2475952837m
```

Interpretation：

```text
v30 answers the v29C open question more directly:
    the weak v29C result was not only because it tested top-support road.
    A more diverse chunk10 masklet bank still does not expose a strong semantic
    causal oracle.

Global source-skip has the strongest row, but it is still below even the weak
Track B h10 signal threshold:
    required for semantic-conditioned cue continuation:
        ATE <= -1.5m or [200,300) <= -3m
    observed:
        ATE = -0.3884m, [200,300) = -2.4341m

SWA remains especially bounded:
    different masklets and anchor/remove actions collapse to identical
    trajectory and boundary metrics.
```

Conclusion type：

```text
v30 does not find a semantic masklet causal upper bound.

Per v30 failure routing:
    downgrade Semantic Prior Generator mainline,
    keep semantic as diagnostics/trust calibration,
    do not train a learned semantic role router,
    redirect Target-30 work to TTT-native / scale-state / read-cue source-skip
    or non-semantic trajectory-state / merge-gauge correction.
```

Next required direction：

```text
Do not start h15 / Track F / selector / full online from current v30 rows.

If semantic work continues, it should not be another masklet rule sweep.
Required repairs before any new semantic promotion:
    make SWA anchor/remove produce auditable distinct hook effects,
    add real per-masklet D_g / conflict / scale-risk / source-attention features,
    and only then re-test oracle.

Otherwise Target-30 mainline should move back to:
    explicit online trajectory-state,
    explicit scale-state,
    skip-aware lifecycle,
    read-cue source-skip outside semantic labels,
    merge/gauge-aware correction,
    or TTT-native causal actions.
```
