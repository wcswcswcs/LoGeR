# ACL2 v40 实验复盘：QualityGated SemanticGeometry MemoryController Target30

日期：2026-05-25（Asia/Singapore）  
计划文件：`docs/ACL2_v40_QualityGated_SemanticGeometry_MemoryController_Target30_Plan.md`  
执行日志：`docs/ACL2_v40_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v40_qualitygated_semanticgeometry_memorycontroller_target30/`

本轮原则：只记录实际落盘结果；不把 Phase 0 no-op health audit、short rollout、proxy visualization、blocked downstream stage 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；初始用户指定 GPU 0,1,2,3 可用，后续用户确认所有卡可用后，仅在无 RUN_NAME 冲突风险的远未来单行上使用额外空闲 GPU 加速。

---

## 0. 当前状态

```text
v40 已按计划完成 Phase 0-2 与 final report 汇总。

已完成并落盘：
    1. 阅读 v40 计划。
    2. 建立 v40 主结果目录。
    3. 复用 v36B H9/C9 parent snapshots:
           chunks = 6,10,16
    4. 新增 v40 runtime 候选别名与 landed-artifact-only 报告脚本。
    5. py_compile / bash -n 验证通过。
    6. Phase 0 no-op health gate 完成并通过：
           rows = 24/24
           max_abs_ATE_delta_vs_noop_reference = 0.0
           max_raw_pose_abs_diff_vs_noop_reference = 0.0
           required_health_streams_nonempty = true
           context_empty_source_events_total = 0.0
    7. Phase 1 passive health atlas 完成：
           rows = 6/6
           health_rows_total = 30
           cue_quality_rows_total = 162
           source_influence_rows_total = 162
           swa_health_rows_total = 162
           ttt_health_rows_total = 162
    8. Phase 2A READ h10 完成：
           rows = 36/36
           gate_pass = false
           best ATE delta = -1.3698298799m
           best rolling100 delta = -3.4811567463m
           best [200,300) delta = -6.3477371145m / -6.3241080599m
    9. Phase 2B SWA h10 完成：
           rows = 36/36
           gate_pass = false
           best ATE delta = -0.7081471064m
           best rolling100 delta = -1.3354202480m
           best [200,300) delta = -2.3916353409m / -2.3564699828m
    10. Phase 2C TTT h10 完成：
            rows = 42/42
            gate_pass = false
            best ATE delta = -0.2368777135m
            best rolling100 delta = -1.5074228818m
            best [200,300) delta = -0.8702571558m / -0.8097502047m
    11. Final reports generated.

未启动：
    Phase 2D RESET diagnostic:
        plan only allows it under severe global-state health flag;
        no such landed severe flag exists in the v40 aggregate health atlas.
    Phase 3 minimal combinations:
        no Phase 2 h10 candidate passed continuation gate.
    Phase 4 full online:
        no h15-qualified candidate exists.

Final:
    full_online_allowed = false
    full_online_launched = false
    target30_success = false
    no new deployable online success
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

---

## 1. 工程修改

新增 / 修改：

```text
tools/run_attention_cue_experiment.sh:
    forwarded existing runtime quality-gate related args:
        read_quality_mass_min/max
        read_quality_anchor_max
        read_quality_frag_max
        gram_layer_groups
        beta_policy / beta_energy_target / beta_min / beta_max

tools/run_v24_candidate_rollout.sh:
    added v40 aliases:
        P0_00_C9_REFERENCE
        P0_01_HEALTH_LOGGING_ONLY
        P0_02_SEMANTIC_PASSIVE_ONLY
        P0_03_APPEARANCE_AUDIT_ONLY
        P1_00_HEALTH_LOGGING_ONLY

        READ_A1_HIGH_INFLUENCE_ANOMALY_V_ATTEN
        READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT
        READ_A3_DYNAMIC_VEG_SHADOW_HIGHD_SKIP_STRUCT_RESCUE
        READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH
        READ_A5_STATIC_ANCHOR_RESCUE_ONLY

        SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP
        SWA_B2_OVERLAP_K_PRESERVE_V_ATTEN_RISKY
        SWA_B3_STRUCTURE_OVERLAP_ANCHOR_PROTECT
        SWA_B4_SKY_HORIZON_NEUTRAL_K_KEEP_V_ATTEN_IF_ANOMALOUS
        SWA_B5_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_OVERLAP_PROTECT

        TTT_C1_STRUCTURE_LOW_RISK_POSITIVE_LONG
        TTT_C2_DYNAMIC_HIGH_RISK_NO_LONG_WRITE
        TTT_C3_VEGETATION_HIGH_D_CONFLICT_SHORT_NEGATIVE
        TTT_C4_SKY_ANOMALY_NO_POSITIVE_LONG_NEUTRAL
        TTT_C5_COMBINED_LIFECYCLE
        TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL

tools/v40_health_atlas_report.py:
    landed-artifact-only Phase 0 no-op drift and health atlas aggregation.
    Missing spatial tensor evidence is marked explainability_missing.

tools/v40_final_summary_report.py:
    landed-artifact-only final report aggregation.
```

Validation：

```text
py_compile = PASS
bash -n = PASS
```

Runtime boundary：

```text
v40 aliases use existing training-free LoGeR controls:
    cue_quality / read_quality thresholds
    VideoMasklet / Stage-C semantic cache
    context source skip / compact_kv / bias attenuation
    SWA write controls
    TTT lifecycle and commit filter controls

No trained trigger/router/classifier is introduced.
No GT semantic label is used as runtime action.
```

Acceleration / scheduling boundary：

```text
Phase2A READ and Phase2B SWA were first run in parallel on disjoint GPU sets.
After READ released GPU 0/1, Phase2C TTT was launched on GPU 0/1.
After user later confirmed all cards were idle/available, only far-future TTT
single-row supplemental launches were used on idle GPU 2/3:
    C9 chunk16 all candidates through a supplemental queue
    C9 chunk10 TTT_C6
    C9 chunk10 TTT_C5

The official TTT queue later skipped those existing DONE rows.
No duplicate active RUN_NAME was intentionally launched, and run_status.txt was
not manually edited.
```

---

## 2. Phase 0：Health instrumentation / no-op gate

输出：

```text
phase0_health/h3_R1/
phase0_health/report_R1/
health_atlas/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 3
candidates =
    P0_00_C9_REFERENCE
    P0_01_HEALTH_LOGGING_ONLY
    P0_02_SEMANTIC_PASSIVE_ONLY
    P0_03_APPEARANCE_AUDIT_ONLY
rows = 24/24
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `phase0_gate_pass` | `true` |
| `phase0_rows_done` | `24` |
| `phase0_missing_rows` | `0` |
| `max_abs_ATE_delta_vs_noop_reference` | `0.0` |
| `max_raw_pose_abs_diff_vs_noop_reference` | `0.0` |
| `required_health_streams_nonempty` | `true` |
| `context_empty_source_events_total` | `0.0` |
| `cue_quality_rows_total` | `96` |
| `source_influence_rows_total` | `96` |
| `swa_health_rows_total` | `96` |
| `ttt_health_rows_total` | `96` |
| `attention_mass_rows_total` | `0` |

Boundary：

```text
Phase 0 proves v40 no-op health logging does not perturb h3 trajectories:
    raw pose max diff = 0
    ATE delta = 0

Runtime rollout artifacts did not land spatial per-label attention mass maps:
    attention_mass_rows_total = 0
    appearance_evidence_level =
        explainability_missing_runtime_rollout_spatial_tensors

Therefore Phase 1 may continue with aggregate health atlas, but per-label /
per-masklet spatial causality remains explainability_limited.
```

Decision：

```text
Phase 0 gate = pass.
Phase 1 passive health atlas is allowed.
```

---

## 3. Phase 1：Passive Health Atlas

输出：

```text
phase1_passive_health/h10_R1/
phase1_passive_health/report_R1/
health_atlas/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates = P1_00_HEALTH_LOGGING_ONLY
rows = 6/6
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `health_rows_total` | `30` |
| `cue_quality_rows_total` | `162` |
| `source_influence_rows_total` | `162` |
| `swa_health_rows_total` | `162` |
| `ttt_health_rows_total` | `162` |
| `context_empty_source_events_total` | `0.0` |
| `attention_mass_rows_total` | `0` |

Generated：

```text
health_atlas/chunk_health_timeline.csv
health_atlas/read_health_by_chunk.csv
health_atlas/swa_health_by_boundary.csv
health_atlas/ttt_health_by_chunk.csv
health_atlas/geometry_health_by_chunk.csv
health_atlas/appearance_health_by_semantic.csv
health_atlas/memory_path_influence_by_semantic.csv
health_atlas/health_flag_summary.json
health_atlas/chunk_health_timeline_heatmap.png
health_atlas/read_swa_ttt_health_timeline.png
```

Boundary：

```text
Phase 1 is passive aggregate health evidence only.
Per-label / per-masklet spatial attention and TTT contribution tensors are not
landed in v40 runtime rollouts; missing spatial evidence is recorded as
explainability_missing, not reconstructed.
```

Decision：

```text
Phase 1 aggregate health atlas = complete.
Phase 2 single-path h10 probes are allowed.
```

---

## 4. Phase 2A：READ Quality-Gated Intervention h10

输出：

```text
phase2a_read/h10_R1/
phase2a_read/report_h10_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    READ_A1_HIGH_INFLUENCE_ANOMALY_V_ATTEN
    READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT
    READ_A3_DYNAMIC_VEG_SHADOW_HIGHD_SKIP_STRUCT_RESCUE
    READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH
    READ_A5_STATIC_ANCHOR_RESCUE_ONLY
rows = 36/36
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT` |
| `best_ATE_parent` | `C9` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-1.3698298799m` |
| `best_rolling_100f_candidate` | `READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH` |
| `best_rolling_100f_parent` | `C9` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-3.4811567463m` |
| `best_downstream_400_600_delta_for_best_ATE` | `+0.7460894838m` |

By parent：

| Parent | Best ATE candidate | Best ATE delta | Best `[200,300)` candidate | Best `[200,300)` delta | Best rolling100 delta |
|---|---|---:|---|---:|---:|
| `H9` | `READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT` | `-1.3538693094` | `READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH` | `-6.3477371145` | `-3.4088925512` |
| `C9` | `READ_A2_HIGH_INFLUENCE_ANOMALY_KV_COMPACT` | `-1.3698298799` | `READ_A4_SKY_APPANOM_WEAK_ATTEN_ONLY_IF_SOURCE_MASS_HIGH` | `-6.3241080599` | `-3.4811567463` |

Decision：

```text
Phase 2A READ h10 gate = fail.
No READ h15 or full-online continuation is allowed from this report.

READ_A4 has a strong local [200,300) diagnostic signal and passes the rolling100
magnitude threshold, but the best full short ATE delta is only -1.3698298799m,
weaker than the v40 h10 ATE threshold of -1.5m.
```

---

## 5. Phase 2B：SWA Quality-Gated Intervention h10

输出：

```text
phase2b_swa/h10_R1/
phase2b_swa/report_h10_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP
    SWA_B2_OVERLAP_K_PRESERVE_V_ATTEN_RISKY
    SWA_B3_STRUCTURE_OVERLAP_ANCHOR_PROTECT
    SWA_B4_SKY_HORIZON_NEUTRAL_K_KEEP_V_ATTEN_IF_ANOMALOUS
    SWA_B5_DYNAMIC_NONOVERLAP_REMOVE_STRUCTURE_OVERLAP_PROTECT
rows = 36/36
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP` |
| `best_ATE_parent` | `C9` |
| `best_ATE_chunk` | `6` |
| `best_ATE_delta_vs_base` | `-0.7081471064m` |
| `best_rolling_100f_candidate` | `SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP` |
| `best_rolling_100f_parent` | `C9` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-1.3354202480m` |
| `best_downstream_400_600_delta_for_best_ATE` | `+0.4147398163m` |

By parent：

| Parent | Best ATE candidate | Best ATE delta | Best `[200,300)` delta | Best rolling100 delta |
|---|---|---:|---:|---:|
| `H9` | `SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP` | `-0.6859359661` | `-2.3916353409` | `-1.3009993775` |
| `C9` | `SWA_B1_NONOVERLAP_RISKY_REMOVE_OVERLAP_KEEP` | `-0.7081471064` | `-2.3564699828` | `-1.3354202480` |

Decision：

```text
Phase 2B SWA h10 gate = fail.
No SWA h15 or full-online continuation is allowed.

The best SWA row improves the [200,300) local segment, but both full short ATE
and rolling100 are far below v40 continuation thresholds:
    h10 ATE threshold = -1.5m
    h10 rolling100 threshold = -3.0m
```

---

## 6. Phase 2C：TTT Quality-Gated Intervention h10

输出：

```text
phase2c_ttt/h10_R1/
phase2c_ttt/report_h10_R1/
```

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    TTT_C1_STRUCTURE_LOW_RISK_POSITIVE_LONG
    TTT_C2_DYNAMIC_HIGH_RISK_NO_LONG_WRITE
    TTT_C3_VEGETATION_HIGH_D_CONFLICT_SHORT_NEGATIVE
    TTT_C4_SKY_ANOMALY_NO_POSITIVE_LONG_NEUTRAL
    TTT_C5_COMBINED_LIFECYCLE
    TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL
rows = 42/42
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL` |
| `best_ATE_parent` | `H9` |
| `best_ATE_chunk` | `10` |
| `best_ATE_delta_vs_base` | `-0.2368777135m` |
| `best_rolling_100f_candidate` | `TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL` |
| `best_rolling_100f_parent` | `C9` |
| `best_rolling_100f_chunk` | `6` |
| `best_rolling_100f_best_delta` | `-1.5074228818m` |
| `best_downstream_400_600_delta_for_best_ATE` | `-0.0530026027m` |

By parent：

| Parent | Best ATE candidate | Best ATE delta | Best `[200,300)` delta | Best rolling100 delta |
|---|---|---:|---:|---:|
| `H9` | `TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL` | `-0.2368777135` | `-0.8702571558` | `-1.1069840416` |
| `C9` | `TTT_C6_FILTERED_COMMIT_ON_HEALTH_FAIL` | `-0.1478018155` | `-0.8097502047` | `-1.5074228818` |

Decision：

```text
Phase 2C TTT h10 gate = fail.
No TTT h15 or full-online continuation is allowed.

The filtered commit candidate is the best TTT row but remains far below v40
continuation thresholds:
    h10 ATE threshold = -1.5m
    h10 rolling100 threshold = -3.0m
```

---

## 7. Phase 2D：Emergency RESET Diagnostic

Decision：

```text
Phase 2D RESET was not launched.

Reason:
    The v40 plan allows RESET_D only under a severe global-state health flag.
    The landed aggregate health atlas contains nonempty health streams and no
    landed severe global-state flag. Therefore launching RESET_D would violate
    the plan boundary and would be diagnostic fishing rather than planned
    failure routing.
```

---

## 8. Phase 3 / Phase 4

Decision：

```text
Phase 3 minimal combinations were not launched.

Reason:
    Phase 3 requires Phase 2 passing candidates from different non-conflicting
    paths. READ, SWA, and TTT all failed h10 continuation gate, and RESET was
    not allowed.

Phase 4 full online was not launched.

Reason:
    Full online requires h15-qualified candidates. No h15 row was launched
    because no Phase 2 h10 candidate passed continuation gate.
```

---

## 9. Final Reports / Decision

Final reports：

```text
final_reports/durability_report.md
final_reports/health_timeline_report.md
final_reports/path_action_report.md
final_reports/full_online_report.md
final_reports/failure_routing_summary.md
final_reports/short_rollout_delta_bar_chart.png
final_reports/v40_final_summary.json
```

Final summary：

```text
phase0_gate_pass = true
read_h10_gate_pass = false
read_h15_gate_pass = null
swa_h10_gate_pass = false
swa_h15_gate_pass = null
ttt_h10_gate_pass = false
ttt_h15_gate_pass = null
reset_h10_gate_pass = null
full_online_allowed = false
full_online_launched = false
target30_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029
```

Downstream decision：

| Stage | Status | Reason |
|---|---|---|
| Phase 0 no-op health gate | pass | no raw pose or ATE perturbation, health streams nonempty |
| Phase 1 passive health atlas | complete | aggregate health / path summaries landed; spatial tensors missing are marked |
| Phase 2A READ h10 | fail but strong diagnostic | best ATE `-1.3698m` misses `-1.5m`; local `[200,300)` `-6.3m` |
| Phase 2B SWA h10 | fail | best ATE `-0.7081m`, rolling100 `-1.3354m` |
| Phase 2C TTT h10 | fail | best ATE `-0.2369m`, rolling100 `-1.5074m` |
| Phase 2D RESET | not launched | no landed severe global-state health flag |
| Phase 3 minimal combo | not launched | no Phase 2 passing candidate |
| Phase 4 full online | not launched | no h15-qualified candidate |
| Target-30 | fail / not produced | no full-online candidate allowed |
| Deployable online success | no | C9 remains best deployable |

Conclusion：

```text
v40 successfully built and audited a quality-gated semantic-geometry controller
framework without perturbing no-op trajectories.

The main positive result is READ_A4:
    H9 [200,300) delta = -6.3477371145m
    C9 [200,300) delta = -6.3241080599m
    best rolling100 delta = -3.4811567463m

However, the best READ full short ATE is only:
    -1.3698298799m
which misses the v40 h10 ATE continuation threshold:
    -1.5m

SWA and TTT remain weaker. Therefore v40 finds a stronger local READ diagnostic
than v39, but still does not produce an h15-qualified or full-online Target-30
candidate.

Do not promote v40 as deployable online success.
```

Next required direction：

```text
Do not launch v40 full online from current rows.
Do not claim Target-30 from READ_A4 h10 local segment improvement.

If continuing this family, the closest lead is READ_A4 / READ_A2:
    preserve the READ_A4 local [200,300) gain,
    improve full short ATE past -1.5m without downstream regression,
    and land stronger attention/source-mass evidence for the quality gate.

The mainline still needs durable activation/lifecycle control or C9-native
non-semantic trajectory / scale / gauge correction.
```
