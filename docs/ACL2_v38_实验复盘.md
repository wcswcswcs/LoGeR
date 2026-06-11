# ACL2 v38 实验复盘：TrainingFree SemanticMemory Durability Target30

日期：2026-05-24（Asia/Singapore）  
计划文件：`docs/ACL2_v38_TrainingFree_SemanticMemory_Durability_Target30_Plan.md`  
执行日志：`docs/ACL2_v38_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/`

本轮原则：只记录实际落盘结果；不把 Track 0 action/influence audit、short rollout、proxy attribution、instrumentation repair、blocked downstream stage 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立 row 分配到不同 GPU。

---

## 0. 当前结论

```text
v38 已按计划完成 Track 0-4 与 final report 汇总。

Track 0 action/influence audit 通过：
    rows = 108/108
    track0_gate_pass = true
    max_skipped_source_influence_mass = 0.1805641204

Track 1 frame/global source surgery h10 失败：
    best ATE delta = -0.0694789609m

Track 2 SWA h10 失败：
    best ATE delta = -0.7081471064m
    best [200,300) delta = -2.3916353409m / -2.3564699828m
    best rolling100 delta = -1.3354202480m

Track 3 TTT h10 失败：
    best ATE delta = -0.1663926129m
    best rolling100 delta = -0.3978539997m

Track 4 semantic C23 residual path isolation h10 失败：
    best ATE delta = -0.4976377546m
    best [200,300) delta = -1.2952954518m
    best rolling100 delta = -0.8793798994m

No Track reached the v38 h10 continuation gate.
Therefore no h15 row was launched and Track 5 full-online was not allowed.

No v38 row reaches full ATE <= 30m.
No new deployable online success is produced.

Current best deployable online TTT write remains:
    C9_P0_R2
    ATE = 33.7629421029m
```

Runtime semantic source boundary：

```text
v38 runtime semantic policies use VideoMasklet frontend / Stage-C semantic cache
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
    results/kitti01_hmc_v2/acl2_v38_trainingfree_semanticmemory_durability_target30/

Reused v36B H9/C9 parent state + merge snapshots through symlink:
    phase0_parent_snapshots ->
    ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots

Snapshot chunks available:
    6,10,16
```

Added / modified：

```text
tools/run_v24_candidate_rollout.sh:
    added v38 Track 0/1/2/3/4 candidate aliases.

tools/v38_durability_report.py:
    added landed-artifact-only h10/h15 durability aggregation:
        ATE / segment / rolling-window metrics
        gate pass/fail summaries
        missing-row reports

tools/v38_action_influence_postprocess.py:
    added v38 Track 0 requested filenames / visualizations from landed
    aggregate summaries.
    Missing per-label/per-masklet tensor fields are explicitly marked as
    explainability_missing rather than reconstructed.

tools/v38_final_summary_report.py:
    added final landed-artifact-only Track 0-5 report aggregation.

Validation:
    py_compile tools/v38_durability_report.py tools/v38_action_influence_postprocess.py tools/v38_final_summary_report.py
    bash -n tools/run_v24_candidate_rollout.sh
    PASS
```

Audit note：

```text
The aliases are training-free runtime controls built from existing LoGeR
semantic role policies and Stage-C VideoMasklet semantic cache fields.
They do not introduce GT SemanticKITTI runtime labels or learned triggers.
```

---

## 2. Track 0：Semantic Influence Atlas v2

输出：

```text
phase0_action_influence/h3_R1/
phase0_action_influence/report_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 3
candidates = 18
rows expected = 108
rows done = 108
failures = 0
attention mass instrumentation = enabled
```

Summary：

| Metric | Value |
|---|---:|
| `h0a_hook_reachability_pass` | `true` |
| `h0b_action_distinguishability_pass` | `true` |
| `h0c_influence_nontriviality_pass` | `true` |
| `track0_gate_pass` | `true` |
| `context_empty_source_events_total` | `0` |
| `source_effect_rows` | `78` |
| `swa_effect_rows` | `6` |
| `ttt_effect_rows` | `108` |
| `attention_mass_rows` | `72` |
| `max_influence_mass` | `0.1805641204` |
| `max_skipped_source_influence_mass` | `0.1805641204` |

v38 postprocess：

```text
attention_mass_removed_before_after_rows = 432
source_attention_mass_removed_bar = true
swa_overlap_nonoverlap_keep_bar = true
ttt_role_mass_by_label_bar = true
influence_atlas_by_chunk = true
per_label_files_status = explainability_missing_when_not_landed
```

Boundary：

```text
Track 0 is action/influence audit only.
It is not a trajectory improvement result and not deployable online success.

Per-label/per-masklet/pixel tensor fields that were not landed are explicitly
recorded as explainability_missing. No missing spatial tensor evidence is
reconstructed.
```

Decision：

```text
Track 0 gate = pass.
Track 1-4 diagnostic probes are allowed.
```

---

## 3. Track 1：Frame/Global Source Surgery

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
    FG_03_LOWTRUST_HIGHD_SKIP
    FG_04_STRUCTURE_RESCUE
    FG_05_RISK_SKIP_STATIC_RESCUE
    FG_06_COMPACT_KV_TRUE
    FG_07_BIAS_ONLY_CONTROL
rows = 48/48
failures = 0
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `FG_01_DYNAMIC_HIGHD_SKIP` |
| `best_ATE_chunk` | `10` |
| `best_ATE_delta_vs_base` | `-0.0473357966m` |
| `best_200_300_candidate` | `FG_02_VEGETATION_HIGHD_SKIP` |
| `best_200_300_chunk` | `10` |
| `best_200_300_delta_vs_base` | `-0.3418221710m` |
| `best_rolling_100f_candidate` | `FG_07_BIAS_ONLY_CONTROL` |
| `best_rolling_100f_chunk` | `10` |
| `best_rolling_100f_best_delta` | `-0.2303364706m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `FG_01_DYNAMIC_HIGHD_SKIP` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.0694789609m` |
| `best_200_300_candidate` | `FG_02_VEGETATION_HIGHD_SKIP` |
| `best_200_300_chunk` | `10` |
| `best_200_300_delta_vs_base` | `-0.2606290156m` |
| `best_rolling_100f_candidate` | `FG_07_BIAS_ONLY_CONTROL` |
| `best_rolling_100f_chunk` | `10` |
| `best_rolling_100f_best_delta` | `-0.1896032366m` |

Decision：

```text
Track 1 h10 gate = fail.
No Track 1 h15 or full-online continuation.

Per plan failure routing, compact_kv, bias-only, static rescue, and high-risk
variants were already included in Track 1 R1. All variants are far below the
h10 continuation gate, so Track 1 is demoted rather than swept.
```

---

## 4. Track 2：SWA Local-Continuity Durability

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
    SWA_01_NONOVERLAP_RISK_REMOVE
    SWA_02_OVERLAP_K_KEEP_V_ATTEN
    SWA_03_STRUCTURE_OVERLAP_PROTECT
    SWA_04_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_PROTECT
    SWA_05_SKY_HORIZON_NEUTRAL
    SWA_06_SOURCE_TOPOLOGY_CONTROL
rows = 42/42
failures = 0
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `SWA_01_NONOVERLAP_RISK_REMOVE` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.6859359661m` |
| `best_200_300_candidate` | `SWA_01_NONOVERLAP_RISK_REMOVE` |
| `best_200_300_chunk` | `10` |
| `best_200_300_delta_vs_base` | `-2.3916353409m` |
| `best_rolling_100f_candidate` | `SWA_01_NONOVERLAP_RISK_REMOVE` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-1.3009993775m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `SWA_01_NONOVERLAP_RISK_REMOVE` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.7081471064m` |
| `best_200_300_candidate` | `SWA_01_NONOVERLAP_RISK_REMOVE` |
| `best_200_300_chunk` | `10` |
| `best_200_300_delta_vs_base` | `-2.3564699828m` |
| `best_rolling_100f_candidate` | `SWA_01_NONOVERLAP_RISK_REMOVE` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-1.3354202480m` |

Decision：

```text
Track 2 h10 gate = fail.
No Track 2 h15 or full-online continuation.

The best SWA row improves the [200,300) segment, but the v38 continuation gate
requires stronger full/rolling durability evidence:
    h10 ATE threshold = -1.5m
    h10 rolling100 threshold = -3.0m
Observed best:
    best ATE delta = -0.7081471064m
    best rolling100 delta = -1.3354202480m
Therefore Track 2 is demoted at h10.
```

---

## 5. Track 3：TTT Static-Anchor / Short-Negative Durability

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
    TTT_01_STRUCTURE_LONG
    TTT_02_STRUCTURE_LONG_DYNAMIC_NOLONG
    TTT_03_VEGETATION_CONDITIONAL_SHORTNEG
    TTT_04_LOWTRUST_SHORTNEG
    TTT_05_SKY_NEUTRAL
    TTT_06_FULL_LIFECYCLE_POLICY
rows = 42/42
failures = 0
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `TTT_04_LOWTRUST_SHORTNEG` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.1663926129m` |
| `best_200_300_candidate` | `TTT_04_LOWTRUST_SHORTNEG` |
| `best_200_300_chunk` | `6` |
| `best_200_300_delta_vs_base` | `-0.2098918009m` |
| `best_rolling_100f_candidate` | `TTT_04_LOWTRUST_SHORTNEG` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-0.3978539997m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `TTT_04_LOWTRUST_SHORTNEG` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.1105288332m` |
| `best_200_300_candidate` | `TTT_04_LOWTRUST_SHORTNEG` |
| `best_200_300_chunk` | `6` |
| `best_200_300_delta_vs_base` | `-0.1461979325m` |
| `best_rolling_100f_candidate` | `TTT_04_LOWTRUST_SHORTNEG` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-0.2513232702m` |

Decision：

```text
Track 3 h10 gate = fail.
No Track 3 h15 or full-online continuation.

The v38 TTT policy family did not reproduce the stronger v37 Track 3 h10
scale-state signal. The best landed row is far below the v38 continuation
thresholds:
    best ATE delta = -0.1663926129m
    best rolling100 delta = -0.3978539997m
```

---

## 6. Track 4：Semantic C23 Residual Path Isolation

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
    C23R_04_FRAMEGLOBAL_COMPACT_ONLY
    C23R_05_STATIC_RESCUE_RESID
    C23R_06_C9_COMPAT_READ_ONLY
rows = 42/42
failures = 0
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `C23R_02_NO_TTT` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.4807208363m` |
| `best_200_300_candidate` | `C23R_02_NO_TTT` |
| `best_200_300_chunk` | `10` |
| `best_200_300_delta_vs_base` | `-1.2784762533m` |
| `best_rolling_100f_candidate` | `C23R_02_NO_TTT` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-0.8477685297m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `C23R_02_NO_TTT` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.4976377546m` |
| `best_200_300_candidate` | `C23R_02_NO_TTT` |
| `best_200_300_chunk` | `10` |
| `best_200_300_delta_vs_base` | `-1.2952954518m` |
| `best_rolling_100f_candidate` | `C23R_02_NO_TTT` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-0.8793798994m` |

Decision：

```text
Track 4 h10 gate = fail.
No Track 4 h15 or full-online continuation.

The best path-isolation row is C23R_02_NO_TTT. It reproduces a modest local
[200,300) segment improvement but does not recover a deployable/durable
semantic C23 signal:
    best ATE delta = -0.4976377546m
    best rolling100 delta = -0.8793798994m
```

---

## 7. Track 5：Full Online

输出：

```text
final_reports/track5_full_online_report.md
```

Decision：

```text
No v38 full-online row was launched.

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

Boundary：

```text
No full trajectory overlay was generated because no v38 full-online candidate
was allowed or launched.
```

---

## 8. Blocker 与修复记录

### Blocker 1：v38 durability report rolling ATE shape mismatch

现象：

```text
Track 1 rollout rows completed 48/48, but first report aggregation failed:
    ValueError: operands could not be broadcast together with shapes
    (50,4,4) (50,3)
```

原因：

```text
tools/v38_durability_report.py used aligned 4x4 pose matrices directly in the
rolling-window ATE calculation instead of extracting xyz translations.
```

修复：

```text
tools/v38_durability_report.py:
    added _as_positions(...)
    rolling-window ATE now converts 4x4 poses to translation vectors with
    pose[:, :3, 3].
```

验证：

```text
py_compile tools/v38_durability_report.py = PASS
```

Boundary：

```text
This was a report-script bug only.
No Track 1 rollout artifact was modified or rerun for this fix.
```

---

## 9. Final Decision

Final reports：

```text
final_reports/track0_action_influence_report.md
final_reports/track1_frameglobal_report.md
final_reports/track2_swa_report.md
final_reports/track3_ttt_report.md
final_reports/track4_semantic_c23_report.md
final_reports/track5_full_online_report.md
final_reports/failure_routing_summary.md
final_reports/short_rollout_delta_bar_chart.png
final_reports/h10_h15_durability_curve.png
final_reports/v38_final_summary.json
```

Final summary：

```text
track0_gate_pass = true
track1_h10_gate_pass = false
track2_h10_gate_pass = false
track2_h15_gate_pass = null
track3_h10_gate_pass = false
track3_h15_gate_pass = null
track4_h10_gate_pass = false
track5_full_online_allowed = false
track5_full_online_launched = false
target30_success = false
```

Downstream decision：

| Stage | Status | Reason |
|---|---|---|
| Track 0 action/influence audit | pass | hooks/actions/influence mass landed; `max_skipped_source_influence_mass = 0.1805641204` |
| Track 1 frame/global h10 | fail | best ATE delta only `-0.0694789609m` |
| Track 2 SWA h10 | fail | best ATE delta `-0.7081471064m`, rolling100 `-1.3354202480m`; below v38 gate |
| Track 3 TTT h10 | fail | best ATE delta `-0.1663926129m` |
| Track 4 semantic C23 h10 | fail | best ATE delta `-0.4976377546m`; local segment improvement not strong/durable enough |
| Track 5 full online | not launched | no h15-qualified candidate |
| Target-30 | fail / not produced | no full online candidate allowed |
| Deployable online success | no | C9 remains best deployable |

Conclusion：

```text
Action/influence audit success, but all v38 durability candidate families fail
the h10 continuation gate.

The strongest landed improvements are local and insufficient:
    Track 2 SWA best [200,300) delta = -2.3916353409m / -2.3564699828m
    Track 4 C23R best [200,300) delta = -1.2784762533m / -1.2952954518m

However, none reaches the v38 h10 ATE or rolling-window gate:
    h10 ATE threshold = -1.5m
    h10 rolling100 threshold = -3.0m

Therefore v38 does not provide a durable training-free semantic memory
mechanism and does not justify Track 5 full-online validation.

Do not promote v38 as deployable online success.
```

Next required direction：

```text
Do not launch v38 Track 5 from current rows.
Do not claim Target-30 from h10-only local segment improvements.

If continuing this family, the missing piece is still durability:
    stronger lifecycle / activation control,
    real tensor-state washout attribution,
    or a C9-native non-semantic scale/gauge/trajectory-state repair.

Target-30 mainline should continue through:
    explicit online trajectory-state,
    scale-state / gauge-risk,
    C9-native lifecycle,
    merge/gauge-aware correction,
    or non-semantic TTT-native control.
```
