# ACL2 v39 实验复盘：SemanticAppearanceCue MemoryPath Causal

日期：2026-05-25（Asia/Singapore）  
计划文件：`docs/ACL2_v39_SemanticAppearanceCue_MemoryPath_CausalPlan.md`  
执行日志：`docs/ACL2_v39_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal/`

本轮原则：只记录实际落盘结果；不把 Track 0 semantic-appearance atlas、short rollout、proxy visualization、blocked downstream stage 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；本轮用户指定 GPU 0,1,2,3 可用。

---

## 0. 当前状态

```text
v39 已按计划完成 Track 0-4 与 final report 汇总。

Track 0 action/influence gate passed:
    rows = 150/150
    missing_rows = 0
    track0_gate_pass = true
    max_skipped_source_influence_mass = 0.12868116796016693
    max_influence_mass = 0.23769477009773254

Semantic-appearance audit landed RGB/Lab/masklet summaries and three proxy
visualization families. Per-label spatial D_g / source attention / SWA / TTT
maps were not landed and are explicitly marked missing.

Sky observation boundary:
    sky_lab_delta_p90 = 2.301654845589792
    sky_candidate_level_influence_mass_max = 0.23769477009773254
    sky_causality_decision = not_proven_per_label_influence_missing

Track 1 frame/global source surgery h10 completed and failed gate:
    rows = 42/42
    best ATE delta = -0.0694789609m

Track 2 SWA local-continuity h10 completed after repairing two START-only
rows by rerun:
    rows = 42/42
    best ATE delta = -0.7081471064m
    best rolling100 delta = -1.3354202480m
    best [200,300) delta = -2.3916353409m / -2.3564699828m
    gate_pass = false

Track 3 TTT lifecycle h10 completed after limited safe acceleration:
    rows = 42/42
    best ATE delta = -0.1085543782m
    best rolling100 delta = -0.1713225537m
    gate_pass = false

Track 4 semantic C23 / appearance-anomaly h10 completed:
    rows = 42/42
    best ATE delta = -0.5844090747m
    best rolling100 delta = -2.8143212231m
    best [200,300) delta = -4.9765638544m / -4.9623982851m
    downstream [400,600) delta = +1.0187841702m
    gate_pass = false

No Track reached the h10 continuation gate.
No h15 row was launched.
Track 5 full online was not allowed and not launched.
No Target-30 result was produced.
No new deployable online success is produced.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

Runtime semantic source boundary：

```text
v39 runtime semantic policies use VideoMasklet frontend / Stage-C semantic cache
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

Initial setup：

```text
Created result root:
    results/kitti01_hmc_v2/acl2_v39_semantic_appearance_cue_memory_path_causal/

Reused v36B H9/C9 parent state + merge snapshots through symlink:
    phase0_parent_snapshots ->
    ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots

Snapshot chunks available:
    6,10,16
```

Added / modified：

```text
tools/run_v24_candidate_rollout.sh:
    added v39 Track 1/2/3/4 candidate aliases.
    The aliases call existing training-free LoGeR causal semantic policies,
    context source skip hooks, SWA write controls, TTT role policies, and C23
    read-cue aliases. No learned trigger/router/classifier is introduced.

    Appearance-anomaly named aliases are implemented as runtime proxy controls
    over existing VideoMasklet semantic role / trust / high-D conditions:
        FG_03_SKY_APPANOM_WEAK_SKIP
        FG_04_LOWTRUST_APPANOM_SKIP
        FG_06_SHADOW_PROXY_SKIP
        C23R_05_APPANOM_SEM_Z

    Boundary:
        These aliases do not yet contain a learned or separately computed
        per-token Lab anomaly trigger. The true Lab/masklet appearance audit is
        produced by tools/v39_semantic_appearance_atlas.py and is reported as
        audit evidence, not as an online trained trigger.

tools/v39_semantic_appearance_atlas.py:
    computes landed RGB / Lab / masklet appearance statistics from KITTI images
    and Stage-C masklet cache, then joins candidate-level action/influence
    summaries from Track 0.

    It writes required v39 Phase 0 filenames when evidence exists, and explicitly
    writes explainability_missing when per-label D_g / attention / SWA / TTT
    tensors were not landed.

tools/v39_final_summary_report.py:
    aggregates Track 0-5 landed summaries and writes final Markdown/JSON reports.
```

Validation status：

```text
py_compile:
    tools/v39_semantic_appearance_atlas.py
    tools/v39_final_summary_report.py
    tools/v38_durability_report.py
    tools/v37_action_influence_atlas.py
    PASS

bash -n:
    tools/run_v24_candidate_rollout.sh
    PASS

snapshot symlink:
    H9/C9 chunk 6/10/16 state snapshots present
```

---

## 2. Blocker 与修复记录

### Blocker 1：v39 `SWA_04_SKY_HORIZON_NEUTRAL` alias 未注册

现象：

```text
Track 0 row failed:
    V39_TRACK0_H3_R1_H9_SWA_04_SKY_HORIZON_NEUTRAL_chunk6_h3...

stderr:
    Unsupported CANDIDATE_ID for v24 rollout: SWA_04_SKY_HORIZON_NEUTRAL
```

原因：

```text
v39 plan uses SWA_04_SKY_HORIZON_NEUTRAL.
Existing launcher only had the older v38 alias:
    SWA_05_SKY_HORIZON_NEUTRAL
```

修复：

```text
tools/run_v24_candidate_rollout.sh:
    added SWA_04_SKY_HORIZON_NEUTRAL to the SWA alias set.
    It maps to the existing training-free sky horizon neutral SWA policy:
        semantic_role_policy = causal_swa_sky_partial_keep
        SWA overlap source gate enabled
        V attenuation target
```

验证：

```text
bash -n tools/run_v24_candidate_rollout.sh = PASS
```

Boundary：

```text
The failed row is not counted as landed evidence.
It must be rerun after the active queue finishes or a safe rerun phase is
started. No metric is inferred for it.
```

### Blocker 2：active launcher edit caused wrapper false-fail

现象：

```text
Row landed DONE:
    V39_TRACK0_H3_R1_H9_SWA_01_NONOVERLAP_DYNAMIC_REMOVE_chunk6_h3...

But the wrapper returned rc=2:
    tools/run_v24_candidate_rollout.sh: line 1362:
        unexpected EOF while looking for matching `"'
```

原因：

```text
Blocker 1 was repaired by editing tools/run_v24_candidate_rollout.sh while
Track 0 rows were already active. Bash can read long shell scripts lazily from
disk, so an in-place script edit can corrupt an already-running shell wrapper
after the child LoGeR process has completed.
```

处理：

```text
No further launcher edits while the active Track 0 queue is running.
After queue exit, rows will be classified from landed artifacts:
    run_status DONE + prediction present -> landed artifact, wrapper false-fail
    otherwise -> missing row requiring repair rerun
```

Boundary：

```text
No metric is inferred from wrapper status alone.
Only landed run_status / prediction / report artifacts count.
```

Result：

```text
After queue exit, four wrapper fail rows were audited as landed artifacts:
    SWA_01_NONOVERLAP_DYNAMIC_REMOVE
    SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC
    SWA_03_STRUCTURE_OVERLAP_PROTECT
    TTT_02_DYNAMIC_NO_LONG_WRITE

The only genuinely missing row was:
    H9 / chunk6 / SWA_04_SKY_HORIZON_NEUTRAL

It was rerun after the alias repair and landed DONE at 2026-05-25 10:02:58.
```

---

## 3. Track 0：Semantic-Appearance Influence Atlas

输出：

```text
phase0_semantic_appearance/h3_R1/
phase0_semantic_appearance/report_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 3
candidates = 25
rows expected = 150
rows done = 150
failures after landed-artifact audit = 0
attention mass instrumentation = enabled
```

Action / influence summary：

| Metric | Value |
|---|---:|
| `h0a_hook_reachability_pass` | `true` |
| `h0b_action_distinguishability_pass` | `true` |
| `h0c_influence_nontriviality_pass` | `true` |
| `track0_gate_pass` | `true` |
| `context_empty_source_events_total` | `0` |
| `source_effect_rows` | `60` |
| `swa_effect_rows` | `36` |
| `ttt_effect_rows` | `150` |
| `attention_mass_rows` | `42` |
| `max_influence_mass` | `0.2376947701` |
| `max_skipped_source_influence_mass` | `0.1286811680` |

Semantic-appearance audit：

| Metric | Value |
|---|---:|
| `frame_rows` | `96` |
| `masklet_rows` | `20` |
| `semantic_label_rows` | `7` |
| `sky_lab_delta_p90` | `2.3016548456` |
| `sky_candidate_level_influence_mass_max` | `0.2376947701` |
| `sky_causality_decision` | `not_proven_per_label_influence_missing` |

Generated visual summaries：

```text
rgb_frame_strip_chunk006/010/016.png
semantic_mask_overlay_chunk006/010/016.png
appearance_anomaly_heatmap_chunk006/010/016.png
semantic_group_memory_path_heatmap.png
action_jaccard_heatmap.png
```

Visualization / causality boundary：

```text
Track 0 proves the candidate action families are reachable, distinguishable,
and have nontrivial landed source-influence mass.

The RGB/Lab audit detects sky appearance variation, but v39 rollout artifacts
do not land per-label spatial source-attention / SWA / TTT contribution maps.
Therefore sky causality is not proven at per-label tensor granularity.

Missing maps are recorded as missing rather than reconstructed:
    D_g_heatmap
    source_attention_mass_heatmap
    SWA_overlap_nonoverlap_source_mass_map
    TTT_update_contribution_map
    full_combined_risk_map_with_Dg_attention_swa_ttt
```

Decision：

```text
Track 0 gate = pass.
Track 1-4 h10 diagnostic probes are allowed.
Track 0 is not ATE evidence and not deployable online success.
```

---

## 4. Track 1：Frame/Global Source Surgery h10

输出：

```text
phase1_frameglobal/h10_R1/
phase1_frameglobal/report_h10_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    FG_01_DYNAMIC_HIGHD_SKIP
    FG_02_VEGETATION_HIGHD_SKIP
    FG_03_SKY_APPANOM_WEAK_SKIP
    FG_04_LOWTRUST_APPANOM_SKIP
    FG_05_RISK_SKIP_STRUCTURE_RESCUE
    FG_06_SHADOW_PROXY_SKIP
rows = 42/42
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `FG_01_DYNAMIC_HIGHD_SKIP` |
| `best_ATE_parent` | `C9` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.0694789609m` |
| `best_rolling_100f_candidate` | `FG_02_VEGETATION_HIGHD_SKIP` |
| `best_rolling_100f_parent` | `H9` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-0.1514717640m` |
| `best_downstream_400_600_delta_for_best_ATE` | `+0.0709069802m` |

By parent：

| Parent | Best ATE candidate | Best ATE delta | Best `[200,300)` candidate | Best `[200,300)` delta | Best rolling100 delta |
|---|---|---:|---|---:|---:|
| `H9` | `FG_01_DYNAMIC_HIGHD_SKIP` | `-0.0473357966` | `FG_02_VEGETATION_HIGHD_SKIP` | `-0.3418221710` | `-0.1514717640` |
| `C9` | `FG_01_DYNAMIC_HIGHD_SKIP` | `-0.0694789609` | `FG_02_VEGETATION_HIGHD_SKIP` | `-0.2606290156` | `-0.1436653637` |

Decision：

```text
Track 1 h10 gate = fail.
No Track 1 h15 or full-online continuation.

The best frame/global source surgery signal is far below the v39 continuation
thresholds:
    h10 ATE threshold = -1.5m
    h10 rolling100 threshold = -3.0m
Observed:
    best ATE delta = -0.0694789609m
    best rolling100 delta = -0.1514717640m
```

---

## 5. Track 2：SWA Local-Continuity h10

输出：

```text
phase2_swa/h10_R1/
phase2_swa/report_h10_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    SWA_01_NONOVERLAP_DYNAMIC_REMOVE
    SWA_02_OVERLAP_K_KEEP_V_ATTEN_DYNAMIC
    SWA_03_STRUCTURE_OVERLAP_PROTECT
    SWA_04_SKY_HORIZON_NEUTRAL
    SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP
    SWA_06_COMBINED_LOCAL_TOPOLOGY
rows = 42/42
failures = 0
```

Blocker / repair：

```text
After the session interruption, Track 2 had 40/42 rows with DONE.
Two C9 chunk16 rows had output artifacts and logs but run_status.txt contained
only START:
    SWA_05_VEG_HIGHCONFLICT_NONOVERLAP_SKIP
    SWA_06_COMBINED_LOCAL_TOPOLOGY

The pre-repair report treated them as missing:
    rows = 40
    missing_rows = 2
    all_rows_done = false

Repair:
    reran exactly the two missing C9/chunk16 rows with FORCE=0 on GPU 0,1.
    The launcher moved the stale START-only directories to
    .INVALID_RERUN_20260525_150649 and landed new DONE rows.

Boundary:
    run_status.txt was not manually edited.
    START-only rows were not counted as completed evidence.
```

Summary：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `SWA_01_NONOVERLAP_DYNAMIC_REMOVE` |
| `best_ATE_parent` | `C9` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.7081471064m` |
| `best_rolling_100f_candidate` | `SWA_01_NONOVERLAP_DYNAMIC_REMOVE` |
| `best_rolling_100f_parent` | `C9` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-1.3354202480m` |
| `best_downstream_400_600_delta_for_best_ATE` | `+0.4147398163m` |

By parent：

| Parent | Best ATE candidate | Best ATE delta | Best `[200,300)` candidate | Best `[200,300)` delta | Best rolling100 delta |
|---|---|---:|---|---:|---:|
| `H9` | `SWA_01_NONOVERLAP_DYNAMIC_REMOVE` | `-0.6859359661` | `SWA_01_NONOVERLAP_DYNAMIC_REMOVE` | `-2.3916353409` | `-1.3009993775` |
| `C9` | `SWA_01_NONOVERLAP_DYNAMIC_REMOVE` | `-0.7081471064` | `SWA_01_NONOVERLAP_DYNAMIC_REMOVE` | `-2.3564699828` | `-1.3354202480` |

Decision：

```text
Track 2 h10 gate = fail.
No Track 2 h15 or full-online continuation.

The best SWA signal remains local and below the v39 continuation thresholds:
    h10 ATE threshold = -1.5m
    h10 rolling100 threshold = -3.0m
Observed:
    best ATE delta = -0.7081471064m
    best rolling100 delta = -1.3354202480m
```

---

## 6. Track 3：TTT Lifecycle / Appearance-Anomaly h10

输出：

```text
phase3_ttt/h10_R1/
phase3_ttt/report_h10_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    TTT_01_STRUCTURE_LONG_ANCHOR
    TTT_02_DYNAMIC_NO_LONG_WRITE
    TTT_03_VEG_SHORT_NEGATIVE
    TTT_04_SKY_NEUTRAL_NO_LONG
    TTT_05_COMBINED_LIFECYCLE
    TTT_06_SHADOW_LOWTRUST_NO_LONG
rows = 42/42
failures = 0
```

Acceleration / repair boundary：

```text
Track 3 was originally launched on GPU 2/3.
After Track 4 completed, GPU 0/1 were used for limited far-future supplemental
rows to accelerate Track 3:
    C9 chunk16 TTT_03/TTT_04/TTT_05/TTT_06
    C9 chunk16 base/TTT_01/TTT_02
    C9 chunk10 TTT_05/TTT_06

When the original queue approached C9 chunk10 TTT_06, the still-running
supplemental TTT_06 process group was terminated to avoid duplicate RUN_NAME
collision. The original launcher then moved the stale START-only directory to:
    .INVALID_RERUN_20260525_193856
and reran the official row successfully.

No run_status.txt was manually edited.
Only landed DONE rows are counted.
Supplemental DONE rows were later skipped by the original queue.
```

Summary：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `TTT_01_STRUCTURE_LONG_ANCHOR` |
| `best_ATE_parent` | `H9` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.1085543782m` |
| `best_rolling_100f_candidate` | `TTT_01_STRUCTURE_LONG_ANCHOR` |
| `best_rolling_100f_parent` | `H9` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-0.1713225537m` |
| `best_downstream_400_600_delta_for_best_ATE` | `-0.0886407615m` |

By parent：

| Parent | Best ATE candidate | Best ATE delta | Best `[200,300)` delta | Best rolling100 delta |
|---|---|---:|---:|---:|
| `H9` | `TTT_01_STRUCTURE_LONG_ANCHOR` | `-0.1085543782` | `-0.1636095239` | `-0.1713225537` |
| `C9` | `TTT_01_STRUCTURE_LONG_ANCHOR` | `-0.0891135238` | `-0.1390289269` | `-0.1698753029` |

Decision：

```text
Track 3 h10 gate = fail.
No Track 3 h15 or full-online continuation.

The v39 TTT lifecycle / appearance-anomaly policy family is much weaker than
the continuation thresholds:
    h10 ATE threshold = -1.5m
    h10 rolling100 threshold = -3.0m
Observed:
    best ATE delta = -0.1085543782m
    best rolling100 delta = -0.1713225537m
```

---

## 7. Track 4：Semantic C23 / Appearance-Anomaly h10

输出：

```text
phase4_semc23/h10_R1/
phase4_semc23/report_h10_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    C23R_01_READ_ONLY_RESID
    C23R_02_NO_TTT
    C23R_03_NO_SWA
    C23R_04_FRAMEGLOBAL_ONLY
    C23R_05_APPANOM_SEM_Z
    C23R_06_STATIC_RESCUE
rows = 42/42
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `C23R_05_APPANOM_SEM_Z` |
| `best_ATE_parent` | `C9` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.5844090747m` |
| `best_rolling_100f_candidate` | `C23R_05_APPANOM_SEM_Z` |
| `best_rolling_100f_parent` | `C9` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-2.8143212231m` |
| `best_downstream_400_600_delta_for_best_ATE` | `+1.0187841702m` |

By parent：

| Parent | Best ATE candidate | Best ATE delta | Best `[200,300)` delta | Best rolling100 delta |
|---|---|---:|---:|---:|
| `H9` | `C23R_05_APPANOM_SEM_Z` | `-0.5764786446` | `-4.9765638544` | `-2.7640142822` |
| `C9` | `C23R_05_APPANOM_SEM_Z` | `-0.5844090747` | `-4.9623982851` | `-2.8143212231` |

Decision：

```text
Track 4 h10 gate = fail.
No Track 4 h15 or full-online continuation.

Although C23R_05_APPANOM_SEM_Z nearly reaches the local [200,300) stress-window
threshold, it fails the continuation gate because:
    best ATE delta = -0.5844090747m, weaker than -1.5m
    best rolling100 delta = -2.8143212231m, weaker than -3.0m
    downstream [400,600) delta = +1.0187841702m, slightly beyond +1.0m boundary
```

---

## 8. Track 5：Full Online

输出：

```text
final_reports/track5_full_online_report.md
```

Decision：

```text
No v39 full-online row was launched.

Reason:
    Track 5 requires at least one h15-qualified candidate from Track 1-4.
    Track 1 failed h10.
    Track 2 failed h10.
    Track 3 failed h10.
    Track 4 failed h10.

full_online_allowed = false
full_online_launched = false
target30_success = false
```

---

## 9. Final Reports / Decision

Final reports：

```text
final_reports/track0_semantic_appearance_report.md
final_reports/track1_frameglobal_report.md
final_reports/track2_swa_report.md
final_reports/track3_ttt_report.md
final_reports/track4_semantic_c23_report.md
final_reports/track5_full_online_report.md
final_reports/failure_routing_summary.md
final_reports/short_rollout_delta_bar_chart.png
final_reports/v39_final_summary.json
```

Final summary：

```text
track0_gate_pass = true
track1_h10_gate_pass = false
track1_h15_gate_pass = null
track2_h10_gate_pass = false
track2_h15_gate_pass = null
track3_h10_gate_pass = false
track3_h15_gate_pass = null
track4_h10_gate_pass = false
track4_h15_gate_pass = null
track5_full_online_allowed = false
track5_full_online_launched = false
target30_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029
sky_causality_decision = not_proven_per_label_influence_missing
```

Downstream decision：

| Stage | Status | Reason |
|---|---|---|
| Track 0 semantic appearance audit | pass | action/influence landed; sky causality remains not proven because per-label influence is missing |
| Track 1 frame/global h10 | fail | best ATE delta only `-0.0694789609m` |
| Track 2 SWA h10 | fail | best ATE `-0.7081471064m`, rolling100 `-1.3354202480m` |
| Track 3 TTT h10 | fail | best ATE `-0.1085543782m`, rolling100 `-0.1713225537m` |
| Track 4 C23/app-anomaly h10 | fail | local `[200,300)` improves about `-4.96m`, but ATE/rolling/downstream gate fails |
| Track 5 full online | not launched | no h15-qualified candidate |
| Target-30 | fail / not produced | no full-online candidate allowed |
| Deployable online success | no | C9 remains best deployable |

Conclusion：

```text
v39 built a semantic-appearance causal audit and found a real local C23
appearance-anomaly segment signal, but no candidate reached the h10 durability
continuation gate.

The strongest landed local signal is Track 4 C23R_05_APPANOM_SEM_Z:
    H9 [200,300) delta = -4.9765638544m
    C9 [200,300) delta = -4.9623982851m

However it is not deployable:
    best ATE delta = -0.5844090747m
    best rolling100 delta = -2.8143212231m
    downstream [400,600) delta = +1.0187841702m

No h15 row was launched.
No full-online row was launched.
No row reaches Target-30.

Do not promote v39 as deployable online success.
```

Next required direction：

```text
Do not launch v39 Track 5 from current rows.
Do not claim Target-30 from h10-only local segment improvements.

If continuing this family, the missing piece remains durable activation:
    per-label/per-masklet landed influence evidence,
    non-fragile lifecycle / activation control,
    or a C9-native non-semantic trajectory/scale/gauge repair.
```
