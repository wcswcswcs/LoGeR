# ACL2 v37 实验复盘：TrainingFree SemanticInfluence MemorySurgery Target30

日期：2026-05-24（Asia/Singapore）  
计划文件：`docs/ACL2_v37_TrainingFree_SemanticInfluence_MemorySurgery_Target30_Plan.md`  
执行日志：`docs/ACL2_v37_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v37_trainingfree_semantic_influence_memory_surgery_target30/`

本轮原则：只记录实际落盘结果；不把 Track 0 action/influence audit、short rollout、instrumentation repair、reused diagnostic evidence、或 blocked downstream stage 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立 row 分配到不同 GPU。

---

## 0. 当前结论

v37 已按计划完成 Track 0-4 与 final report 汇总。Track 0 action/influence audit 通过；Track 1 frame/global source surgery h10 失败；Track 2 SWA h10 有局部信号但 h15 durability 失败；Track 3 TTT scale-state h10 通过但 h15 durability 失败；Track 4 semantic C23 path-isolation h10 失败。因此没有任何 h15-qualified candidate 进入 Track 5，full online validation 没有启动，v37 没有产生 Target-30 或新的 deployable online result。

已完成并落盘：

```text
1. 阅读 v37 计划，建立执行日志与实验复盘日志。
2. 复用 v36B H9/C9 parent snapshots:
       chunks = 6,10,16
       state + merge snapshots complete
3. Track 0 action/influence smoke 完成:
       rows = 60/60
       track0_gate_pass = true
4. Track 1 frame/global source surgery h10 完成:
       rows = 30/30
       H9/C9 gate_pass = false
5. Track 2 SWA h10 完成:
       rows = 36/36
       H9/C9 h10 gate_pass = true
6. Track 2 SWA h15 完成:
       rows = 16/16
       H9/C9 h15 gate_pass = false
7. Track 2 washout attribution 完成:
       evidence_level = proxy_only_no_tensor_state_snapshots
8. Track 3 TTT h10 完成:
       rows = 48/48
       H9/C9 h10 gate_pass = true via TTT_FINE_RISK_02_SCALE_STATE
9. Track 3 TTT h15 完成:
       rows = 4/4
       H9/C9 h15 gate_pass = false
10. Track 4 semantic C23 path isolation h10 完成:
       rows = 36/36
       H9/C9 gate_pass = false
11. Final reports and charts generated.
12. Track 5 full online not launched because no h15-qualified candidate exists.
```

最终边界：

```text
1. Track 0 proves the tested action families are reachable, distinguishable,
   and have nontrivial landed source-influence mass.
2. Track 2 SWA and Track 3 TTT both show h10 local diagnostic signal.
3. Both Track 2 and Track 3 wash out at h15, so neither can be promoted.
4. Track 4 semantic C23 path-isolation does not recover the v36B beta525
   local signal in the v37 policy family.
5. No v37 full-online row was launched.
6. No v37 row reaches full ATE <= 30m.
7. No new deployable online success is produced.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

Runtime semantic source boundary：

```text
v37 runtime semantic policies use VideoMasklet frontend / Stage-C semantic cache
outputs:
    fine_label_source = MaskletOutput.L_sem
    semantic_group_source = MaskletOutput.G_sem
    semantic_group_taxonomy = stage_c_coarse_5_groups

They do not use GT SemanticKITTI labels at runtime.
Projected SemanticKITTI / KITTI 3D labels from earlier experiments remain
offline audit / trust-calibration evidence only.
```

---

## 1. 工程修改

新增 / 修改：

```text
tools/v37_action_influence_atlas.py:
    aggregates Track 0 landed smoke artifacts into:
        phase0_action_influence_report.md
        semantic_influence_atlas.csv
        semantic_path_action_influence.csv
        action_summary_by_candidate.csv
        action_jaccard_matrix.csv
        policy_jaccard_matrix.csv
        action_keep_ratio_by_path.csv
        context_empty_source_events.csv
        protected_token_count.csv
        per_label_action_mass.csv
        per_masklet_action_mass.csv
        semantic_group_memory_path_heatmap.png
        action_jaccard_heatmap.png
    Missing per-label/per-masklet tensor granularity is explicitly written as
    explainability_missing rather than reconstructed.

tools/v37_final_summary_report.py:
    aggregates Track 1-5 summaries and writes:
        final_reports/track1_frameglobal_report.md
        final_reports/track2_swa_report.md
        final_reports/track3_ttt_report.md
        final_reports/track4_semantic_c23_report.md
        final_reports/track5_full_online_report.md
        final_reports/failure_routing_summary.md
        final_reports/segment_ate_bar_chart.png
        final_reports/h10_h15_durability_curve.png
        final_reports/v37_final_summary.json

tools/run_v24_candidate_rollout.sh:
    added v37 Track 4 aliases:
        SEM_C23_01_READ_ONLY_RESID
        SEM_C23_02_NO_TTT
        SEM_C23_03_NO_SWA
        SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY
        SEM_C23_05_STATIC_RESCUE_RESID
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v37_action_influence_atlas.py \
    tools/v37_final_summary_report.py

bash -n tools/run_v24_candidate_rollout.sh

PASS
```

---

## 2. Blocker 与修复记录

### Blocker 1：Track 0 atlas 被 NaN 污染

现象：

```text
Initial Track 0 aggregation returned h0c false because NaN from missing frame
attention mass contaminated max_skipped_source_influence_mass.
```

修复：

```text
tools/v37_action_influence_atlas.py:
    added _finite_float(...)
    ignores NaN when computing source influence maxima.
```

结果：

```text
h0c_influence_nontriviality_pass = true
max_skipped_source_influence_mass = 0.12868116796016693
track0_gate_pass = true
```

### Blocker 2：Track2/3 R1 launcher edit caused invalid/misrouted rows

现象：

```text
While Track2/3 R1 queues were active, Track4 alias repair edited
tools/run_v24_candidate_rollout.sh. Some rows then launched with inconsistent
launcher state / default attention_cue_library_v1 root.
```

处理：

```text
R1 rows were not used for any gate.
Invalid/misrouted artifacts were moved aside under:
    phase2_swa/h10_R1_20260524T1620_launcher_edit_invalid/
    phase3_ttt/h10_R1_20260524T1620_launcher_edit_invalid/
    invalid_misrouted_attention_cue_library_R1/
```

修复：

```text
Rerun Track2 and Track3 as R2 after launcher aliases stabilized.
Keep one LoGeR process per GPU.
```

结果：

```text
Track2 h10 R2:
    rows = 36/36
    failures = 0

Track3 h10 R2:
    rows = 48/48
    failures = 0
```

### Blocker 3：Visualization artifacts unavailable at pixel/tensor granularity

现象：

```text
The v37 plan requests semantic influence visualization, but v37 rollout reports
do not land full pixel-level RGB / semantic / trust / D_g / scale-risk /
source-attention / SWA / TTT overlay tensors.
```

处理：

```text
Generated only auditable landed summaries:
    semantic_group_memory_path_heatmap.png
    action_jaccard_heatmap.png
    segment_ate_bar_chart.png
    h10_h15_durability_curve.png

Pixel-level overlays and full trajectory overlays were not fabricated.
```

Decision：

```text
Visualization is sufficient for Track 0 action/influence audit, but not a
pixel-level semantic overlay claim.
```

---

## 3. Track 0：Semantic Influence Atlas / Action Realism

输出：

```text
phase0_action_influence/smoke_R1/
phase0_action_influence/report_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 3
candidates = 10
rows expected = 60
rows done = 60
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `h0a_hook_reachability_pass` | `true` |
| `h0b_action_distinguishability_pass` | `true` |
| `h0c_influence_nontriviality_pass` | `true` |
| `track0_gate_pass` | `true` |
| `context_empty_source_events_total` | `0` |
| `source_effect_rows` | `24` |
| `swa_effect_rows` | `12` |
| `ttt_effect_rows` | `60` |
| `attention_mass_rows` | `24` |
| `max_influence_mass` | `0.12868116796016693` |
| `max_skipped_source_influence_mass` | `0.12868116796016693` |

Decision：

```text
Track 0 gate = pass.
Track 1-4 short diagnostic probes are allowed.
```

Boundary：

```text
Track 0 is action/influence audit only.
It is not a trajectory improvement result and not deployable online success.
```

---

## 4. Track 1：Frame/Global Source Surgery h10

输出：

```text
phase1_frameglobal/h10_R1/
phase1_frameglobal/report_h10_R1_H9/
phase1_frameglobal/report_h10_R1_C9/
phase1_frameglobal/report_h10_R1_H9_context/
phase1_frameglobal/report_h10_R1_C9_context/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    FG_RISK_00
    FG_SEM_02
    FG_SEM_04
    FG_SEM_05
rows = 30/30
failures = 0
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `FG_SEM_04` |
| `best_ATE_chunk` | `10` |
| `best_ATE_delta` | `-0.0473357966m` |
| `best_[200,300)_candidate` | `FG_RISK_00` |
| `best_[200,300)_chunk` | `10` |
| `best_[200,300)_delta` | `-0.3418221710m` |
| `best_[400,600)_delta_for_best_ATE` | `-0.0437172725m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `FG_SEM_04` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta` | `-0.0694789609m` |
| `best_[200,300)_candidate` | `FG_RISK_00` |
| `best_[200,300)_chunk` | `10` |
| `best_[200,300)_delta` | `-0.2606290156m` |
| `best_[400,600)_delta_for_best_ATE` | `+0.0709069802m` |

Decision：

```text
Track 1 h10 gate = fail.
No Track 1 h15 or full-online continuation.
```

---

## 5. Track 2：SWA Local-Continuity

### h10

输出：

```text
phase2_swa/h10_R2/
phase2_swa/report_h10_R2_H9/
phase2_swa/report_h10_R2_C9/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    SWA_FINE_01_OVERLAP_STRUCTURE_KEEP
    SWA_FINE_02_SKY_PARTIAL_KEEP
    SWA_FINE_03_VEGETATION_CONDITIONAL
    SWA_FINE_04_BOUNDARY_PROTECT
    SWA_FINE_05_CACHE_LIFECYCLE
rows = 36/36
failures = 0
```

H9 h10：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_ATE_candidate` | `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta` | `-1.2136622509m` |
| `best_[200,300)_candidate` | `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` |
| `best_[200,300)_chunk` | `10` |
| `best_[200,300)_delta` | `-3.7351609926m` |
| `best_[400,600)_delta_for_best_ATE` | `+0.9505750525m` |

C9 h10：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_ATE_candidate` | `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta` | `-1.2466940180m` |
| `best_[200,300)_candidate` | `SWA_FINE_01_OVERLAP_STRUCTURE_KEEP` |
| `best_[200,300)_chunk` | `10` |
| `best_[200,300)_delta` | `-3.4943516930m` |
| `best_[400,600)_delta_for_best_ATE` | `+0.9348534197m` |

Boundary diagnostics：

| Parent | Chunk | Candidate | Boundary 10f delta |
|---|---:|---|---:|
| `H9` | `6` | `SWA_FINE_01` | `-0.5939405842m` |
| `H9` | `10` | `SWA_FINE_01` | `+0.0723922636m` |
| `C9` | `6` | `SWA_FINE_01` | `-0.6149793460m` |
| `C9` | `10` | `SWA_FINE_01` | `+0.2221317699m` |

### h15

输出：

```text
phase2_swa/h15_R1/
phase2_swa/report_h15_R1_H9/
phase2_swa/report_h15_R1_C9/
```

设置：

```text
parents = H9,C9
chunks = 6,10
horizon = 15
candidates =
    V31_BASE_H9_REFERENCE
    SWA_FINE_01_OVERLAP_STRUCTURE_KEEP
    SWA_FINE_02_SKY_PARTIAL_KEEP
    SWA_FINE_04_BOUNDARY_PROTECT
rows = 16/16
failures = 0
```

H9 h15：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_delta` | `-0.8292622200m` |
| `best_[200,300)_delta` | `-2.2466407151m` |
| `best_[400,600)_delta_for_best_ATE` | `-0.4959528739m` |

C9 h15：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_delta` | `-0.8404528162m` |
| `best_[200,300)_delta` | `-2.2988546997m` |
| `best_[400,600)_delta_for_best_ATE` | `-0.4782051590m` |

Washout attribution：

```text
phase2_swa/washout_R1/
evidence_level = proxy_only_no_tensor_state_snapshots
```

Important durability：

| Parent | Chunk | Candidate | h15/h10 `[200,300)` abs durability |
|---|---:|---|---:|
| `C9` | `10` | `SWA_FINE_01` | `0.2471427261` |
| `C9` | `6` | `SWA_FINE_01` | `0.8868817382` |

Decision：

```text
Track 2 has h10 local signal but fails h15 durability.
No Track 2 full-online continuation.
```

---

## 6. Track 3：TTT Static / Short-Negative

### h10

输出：

```text
phase3_ttt/h10_R2/
phase3_ttt/report_h10_R2_H9/
phase3_ttt/report_h10_R2_C9/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    TTT_FINE_01_STRUCTURE_POSITIVE
    TTT_FINE_03_SCALE_CONDITIONED
    TTT_FINE_04_LOWSTUFF_HIGHD_SHORT
    TTT_FINE_05_STRUCTURE_PROTECT
    TTT_FINE_RISK_01_CONFLICT_TRI
    TTT_FINE_RISK_02_SCALE_STATE
    TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER
rows = 48/48
failures = 0
```

H9 h10：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_candidate` | `TTT_FINE_RISK_02_SCALE_STATE` |
| `best_chunk` | `10` |
| `best_ATE_delta` | `-1.8431656072m` |
| `best_[200,300)_delta` | `-1.7711993876m` |
| `best_[400,600)_delta_for_best_ATE` | `-2.5522426503m` |

C9 h10：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_candidate` | `TTT_FINE_RISK_02_SCALE_STATE` |
| `best_chunk` | `10` |
| `best_ATE_delta` | `-1.9880251151m` |
| `best_[200,300)_delta` | `-1.9522337557m` |
| `best_[400,600)_delta_for_best_ATE` | `-2.7310700062m` |

### h15

输出：

```text
phase3_ttt/h15_R1/
phase3_ttt/report_h15_R1_H9/
phase3_ttt/report_h15_R1_C9/
```

设置：

```text
parents = H9,C9
chunk = 10
horizon = 15
candidates =
    V31_BASE_H9_REFERENCE
    TTT_FINE_RISK_02_SCALE_STATE
rows = 4/4
failures = 0
```

H9 h15：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `ATE_delta` | `-0.9431819076m` |
| `[200,300)_delta` | `-1.5502750675m` |
| `[400,600)_delta` | `-1.6984935722m` |

C9 h15：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `ATE_delta` | `-1.0265231323m` |
| `[200,300)_delta` | `-1.7469440517m` |
| `[400,600)_delta` | `-1.8278025868m` |

Decision：

```text
Track 3 h10 passes by ATE for TTT_FINE_RISK_02_SCALE_STATE,
but h15 gate fails.
No Track 3 full-online continuation.
```

---

## 7. Track 4：Semantic C23 Path Isolation h10

输出：

```text
phase4_semc23/h10_R1/
phase4_semc23/report_h10_R1_H9/
phase4_semc23/report_h10_R1_C9/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    SEM_C23_01_READ_ONLY_RESID
    SEM_C23_02_NO_TTT
    SEM_C23_03_NO_SWA
    SEM_C23_04_FRAMEGLOBAL_COMPACT_ONLY
    SEM_C23_05_STATIC_RESCUE_RESID
rows = 36/36
failures = 0
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_candidate` | `SEM_C23_02_NO_TTT` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta` | `-0.4807208363m` |
| `best_[200,300)_chunk` | `10` |
| `best_[200,300)_delta` | `-1.2784762533m` |
| `best_[400,600)_delta_for_best_ATE` | `+0.1168446428m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_candidate` | `SEM_C23_02_NO_TTT` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta` | `-0.4976377546m` |
| `best_[200,300)_chunk` | `10` |
| `best_[200,300)_delta` | `-1.2952954518m` |
| `best_[400,600)_delta_for_best_ATE` | `+0.1246525937m` |

Decision：

```text
Track 4 h10 gate = fail.
No Track 4 h15 or full-online continuation.
```

---

## 8. Track 5：Full Online

输出：

```text
final_reports/track5_full_online_report.md
```

Decision：

```text
No v37 full-online row was launched.

Reason:
    Track 5 requires h15-qualified candidates from Tracks 1-4.
    Track 1 failed h10.
    Track 2 passed h10 but failed h15.
    Track 3 passed h10 but failed h15.
    Track 4 failed h10.

full_online_allowed = false
full_online_launched = false
target30_success = false
```

Boundary：

```text
No full trajectory overlay was generated because no v37 full-online candidate
was allowed or launched.
```

---

## 9. Final Reports / Visualizations

输出：

```text
final_reports/
```

Generated：

```text
track1_frameglobal_report.md
track2_swa_report.md
track3_ttt_report.md
track4_semantic_c23_report.md
track5_full_online_report.md
failure_routing_summary.md
segment_ate_bar_chart.png
h10_h15_durability_curve.png
v37_final_summary.json
```

Final summary：

```text
track0_gate_pass = true
track1_gate_pass = false
track2_h10_gate_pass = true
track2_h15_gate_pass = false
track3_h10_gate_pass = true
track3_h15_gate_pass = false
track4_h10_gate_pass = false
track5_full_online_allowed = false
track5_full_online_launched = false
target30_success = false
```

Visualization boundary：

```text
Generated:
    phase0 semantic influence heatmap
    action Jaccard heatmap
    h10->h15 durability curve
    segment delta bar chart

Not generated:
    pixel-level RGB / semantic / trust / D_g / scale-risk /
    source-attention / SWA / TTT overlays

Reason:
    those tensors/images were not landed by v37 rollout reports at the needed
    spatial granularity. No overlay is fabricated.
```

---

## 10. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Track 0 action/influence audit | pass | hook reachability, distinguishability, and skipped-source influence mass landed |
| Track 1 frame/global h10 | fail | best local deltas far below gate |
| Track 2 SWA h10 | pass diagnostic | local signal on chunks 6/10 |
| Track 2 SWA h15 | fail | durability gate failed |
| Track 2 washout attribution | done | proxy-only; no tensor-state overwrite proof claimed |
| Track 3 TTT h10 | pass diagnostic | scale-state candidate passed by ATE on chunk10 |
| Track 3 TTT h15 | fail | durability gate failed |
| Track 4 semantic C23 h10 | fail | path-isolation aliases did not recover strong local gate |
| Track 5 full online | not launched | no h15-qualified candidate |
| Target-30 | fail / not produced | no full online candidate allowed |
| Deployable online success | no | C9 remains best deployable |

---

## 11. Final Decision

v37 的真实成功点：

```text
1. Built an auditable Semantic Influence Atlas / action realism report.
2. Verified Track 0 action families are reachable and nontrivial:
       track0_gate_pass = true
       max_skipped_source_influence_mass = 0.12868116796016693
3. Found Track 2 SWA h10 local diagnostic signal under H9/C9.
4. Found Track 3 TTT scale-state h10 diagnostic signal under H9/C9.
5. Completed h15 durability checks instead of promoting h10-only wins.
6. Generated final report artifacts and charts for future audit.
```

v37 的关键负结果：

```text
Track 1 frame/global source surgery failed h10.

Track 2 SWA:
    H9 h10 best [200,300) delta = -3.7351609926m
    C9 h10 best [200,300) delta = -3.4943516930m
    H9 h15 best [200,300) delta = -2.2466407151m
    C9 h15 best [200,300) delta = -2.2988546997m
    h15 gate = fail

Track 3 TTT scale-state:
    H9 h10 ATE delta = -1.8431656072m
    C9 h10 ATE delta = -1.9880251151m
    H9 h15 ATE delta = -0.9431819076m
    C9 h15 ATE delta = -1.0265231323m
    h15 gate = fail

Track 4 semantic C23 path isolation:
    H9 best [200,300) delta = -1.2784762533m
    C9 best [200,300) delta = -1.2952954518m
    h10 gate = fail

No candidate reached the continuation requirement for Track 5.
```

Interpretation：

```text
v37 answers the memory-surgery question more directly than v36B:
    the action/influence path is real enough to run,
    and both SWA and TTT can create short local improvements.

But the useful signals are not durable:
    SWA washes out by h15,
    TTT scale-state weakens below h15 gate,
    semantic C23 path isolation does not recover the prior beta525 local gain.

Therefore v37 does not provide a deployable semantic influence memory surgery
mechanism and does not justify full-online validation.
```

Conclusion type：

```text
Action/influence audit success, h10 local diagnostic signals found, h15
durability failure, no full-online Target-30 candidate.

Do not promote v37 as deployable online success.
```

Next required direction：

```text
Do not launch v37 Track 5 from current rows.
Do not claim Target-30 from h10-only diagnostics.

If continuing this family, the missing piece is durability:
    preserve Track2 SWA or Track3 TTT local gains through h15,
    land tensor-state snapshots for real washout attribution,
    or design a non-semantic C9-native lifecycle / scale-state repair that
    does not depend on fragile semantic path isolation.

Target-30 mainline should continue through:
    explicit online trajectory-state,
    scale-state / gauge-risk,
    C9-native lifecycle,
    merge/gauge-aware correction,
    or non-semantic TTT-native control.
```
