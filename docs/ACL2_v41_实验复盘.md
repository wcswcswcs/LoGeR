# ACL2 v41 实验复盘：ReadFirst HealthGated SemanticGeometry Target30

日期：2026-05-26（Asia/Singapore）  
计划文件：`docs/ACL2_v41_ReadFirst_HealthGated_SemanticGeometry_Target30_Plan.md`  
执行日志：`docs/ACL2_v41_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/`

本轮原则：只记录实际落盘结果；不把 health detector、mechanism audit、short rollout、proxy washout、blocked downstream stage 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；用户确认 GPU 0,1,2,3,4,5,6,7 可用。

---

## 0. 当前结论

```text
v41 已按计划完成 Phase 1-5 与 final report 汇总。

Phase 1 health detector 通过：
    selected_bad_chunks = [10]
    selected_bad_chunk_ratio = 0.3333333333
    selection_uses_ATE = false
    selection_uses_fixed_chunk_or_segment = false

Phase 2 READ mechanism audit 完成：
    mechanism_decision = B_general_high_influence_anomaly_preferred
    scalar_attention_mass_rows = 144
    sky_causality_proven = false

Phase 3 READ h10 通过 v41 gate：
    rows = 12/12
    h10_gate_pass = true
    best ATE delta = -0.5994474373m
    best rolling100 delta = -3.3097470786m
    best stress [200,300) delta = -8.5850808794m

Phase 3 READ h15 完成但失败：
    rows = 8/8
    H9 h15 gate_pass = false
    C9 h15 gate_pass = false
    H9 best ATE delta = -0.1660176140m
    C9 best ATE delta = -0.1484353180m
    H9 best stress [200,300) delta = -4.7873967631m
    C9 best stress [200,300) delta = -4.8163673672m

Phase 5 proxy washout attribution 完成：
    evidence_level = proxy_only_no_tensor_state_snapshots

No h15-qualified candidate exists.
No full-online row was launched.
No Target-30 result was produced.
No new deployable online success is produced.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

名称解释：

```text
C9_P0_R2 是历史锁定 baseline / recipe 名，不是 v41 新候选。

C9:
    v15/v16 中的 C9 控制配方，当前作为可部署 online TTT-write best。

P0:
    Phase 0 boundary / reproducibility 复现实验阶段。

R2:
    第 2 个 locked repeat / locked boundary run。

对应已落盘 run:
    v15 repeat:
        results/kitti01_hmc_v2/acl2_v15_ttt_repro_causal_sandbox_target25/
        phase0_repro/V15_P0_A2_C9_REPEAT_no_state_save_SWKS3

    v16 locked boundary:
        results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/
        phase0_boundary/V16_P0_R2_C9_locked_exact_merge_input_SWKS3

核心配置摘要:
    hybrid_memory_mode = hybrid
    hmc_commit_mode = probe_ttt_write
    hmc_write_score_source = stage_d_x_dg_inv_sqrt
    read_cue_source = acl2.gg.qq.low.g2_3.past_only.headmean.robustq
    mp_alpha = 0.1
    read_beta_frame_chunks = 5-9:4.85, 10-12/16:4.25
    ttt_write_gradient_reversal_chunk_gammas =
        5-9:0.005, 10-12:0.003, 16:0.0003
```

Runtime boundary：

```text
v41 remains training-free.
No learned trigger/router/classifier was introduced.
No GT SemanticKITTI label was used as runtime action.
Phase 1 selected chunk10 from v40 aggregate health metrics, not from a fixed
chunk id or runtime ATE condition.
```

---

## 1. 工程修改

Initial setup：

```text
Created result root:
    results/kitti01_hmc_v2/acl2_v41_readfirst_healthgated_semanticgeometry_target30/

Reused v36B H9/C9 parent state + merge snapshots through symlink:
    phase0_parent_snapshots ->
    ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots
```

新增 / 修改：

```text
tools/run_v24_candidate_rollout.sh:
    added v41 aliases:
        R1_READ_HIGH_INFLUENCE_ANOMALY
        R2_READ_SKY_APP_ANOMALY
        R3_READ_ANOMALY_PLUS_STATIC_RESCUE
        R4_NEG_CONTROL_SKY_NO_SOURCE_MASS
        R5_NEG_CONTROL_STATIC_RESCUE_ONLY

tools/v41_health_detector_report.py:
    builds a training-free health detector from landed v40 health atlas and
    v39 appearance proxy evidence.

tools/v41_read_mechanism_report.py:
    aggregates READ_A2/A4 mechanism evidence, copies proxy overlays, and records
    scalar attention-mass rows when available.

tools/v41_read_gate_report.py:
    applies v41 h10/h15 continuation gate logic from landed effect CSVs.

tools/v41_read_washout_report.py:
    creates proxy-only h10->h15 durability / path-tail attribution reports.

tools/v41_final_summary_report.py:
    aggregates final landed results into final_reports/.
```

Validation：

```text
py_compile:
    tools/v41_final_summary_report.py
    tools/v41_read_washout_report.py
    tools/v41_health_detector_report.py
    tools/v41_read_mechanism_report.py
    tools/v41_read_gate_report.py
    PASS

bash -n tools/run_v24_candidate_rollout.sh:
    PASS
```

Alias boundary：

```text
R1/R2 reuse v40 READ_A2/A4 family controls.
R3/R4/R5 are runtime proxies assembled from existing LoGeR training-free
semantic role, trust, high-D, compact_kv, and static-rescue controls.
They are not learned triggers and are not true per-token Lab anomaly triggers.
```

---

## 2. Phase 1：Health Detector

输出：

```text
phase1_health_detector/
```

Generated：

```text
chunk_health_table.csv
rolling_window_health_alignment.csv
health_component_by_chunk.csv
health_vs_rolling_ate_scatter.png
chunk_health_timeline.png
bad_chunk_report.md
v41_health_detector_summary.json
selected_bad_chunks.json
```

Summary：

| Metric | Value |
|---|---:|
| `phase1_gate_pass` | `true` |
| `selected_bad_chunks` | `[10]` |
| `selected_bad_chunk_ratio` | `0.3333333333` |
| `top3_health_risk_chunks` | `[10, 6, 16]` |
| `top_rolling100_bad_chunk_diagnostic` | `6` |
| `top3_covers_top_rolling100_bad_window` | `true` |
| `stress_window_overlap_chunks_diagnostic` | `[10, 6]` |
| `stress_window_has_health_high_risk_or_top3_explanation` | `true` |
| `selection_uses_ATE` | `false` |
| `selection_uses_fixed_chunk_or_segment` | `false` |

Boundary：

```text
The health detector uses training-free v40 health metrics plus v39 appearance
proxy evidence. ATE is used only as offline diagnostic alignment, not as runtime
selection.
```

Decision：

```text
Phase 1 gate = pass.
Phase 3 may run only on health-selected chunk10.
```

---

## 3. Phase 2：READ Mechanism / Attribution

输出：

```text
phase2_read_mechanism/h3_R1/
phase2_read_mechanism/
```

设置：

```text
parents = H9,C9
chunk = 10
horizon = 3
candidates = R1,R2,R5
attention mass instrumentation = enabled
rows = 6/6
```

Generated：

```text
read_a2_a4_attribution.csv
per_label_removed_source_mass.csv
action_mask_overlap.csv
read_a4_sky_causality_report.md
READ_A2_A4_attribution_report.md
scalar_attention_mass_rows.csv
overlays/chunk010_proxy_rgb_frame_strip.png
overlays/chunk010_proxy_semantic_mask_overlay.png
overlays/chunk010_proxy_appearance_anomaly_heatmap.png
overlays/spatial_attention_boundary.json
v41_read_mechanism_summary.json
```

Summary：

| Metric | Value |
|---|---:|
| `mechanism_decision` | `B_general_high_influence_anomaly_preferred` |
| `best_READ_A2_ATE_delta` | `-0.5994474373m` |
| `best_READ_A4_stress_delta` | `-6.3477371145m` |
| `scalar_attention_mass_rows` | `144` |
| `proxy_overlays_copied` | `3` |
| `sky_causality_proven` | `false` |

Interpretation：

```text
READ_A2/R1 is preferred because it has safer full short-ATE behavior.
READ_A4/R2 has strong stress-window signal but regresses downstream in v40 and
does not prove sky-specific causality.
```

Boundary：

```text
Supplemental h3 rows landed scalar attention-mass evidence, but v41 artifacts
still do not contain per-label spatial attention maps or affected masks.
Therefore sky causality remains not proven at tensor/spatial granularity.
```

---

## 4. Phase 3：READ h10

输出：

```text
phase3_read_h10/h10_R1/
phase3_read_h10/report_h10_R1/
```

设置：

```text
parents = H9,C9
chunks = 10
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    R1_READ_HIGH_INFLUENCE_ANOMALY
    R2_READ_SKY_APP_ANOMALY
    R3_READ_ANOMALY_PLUS_STATIC_RESCUE
    R4_NEG_CONTROL_SKY_NO_SOURCE_MASS
    R5_NEG_CONTROL_STATIC_RESCUE_ONLY
rows = 12/12
failures = 0
```

Summary：

| Metric | Value |
|---|---:|
| `h10_gate_pass` | `true` |
| `best_ATE_candidate` | `R1_READ_HIGH_INFLUENCE_ANOMALY` |
| `best_ATE_parent` | `C9` |
| `best_ATE_delta_vs_base` | `-0.5994474373m` |
| `best_rolling100_candidate` | `R4_NEG_CONTROL_SKY_NO_SOURCE_MASS` |
| `best_rolling100_parent` | `C9` |
| `best_rolling100_delta` | `-3.3097470786m` |
| `best_stress_candidate` | `R3_READ_ANOMALY_PLUS_STATIC_RESCUE` |
| `best_stress_parent` | `H9` |
| `best_stress_delta` | `-8.5850808794m` |

Gate-pass rows：

| Parent | Candidate | Reason |
|---|---|---|
| `H9` | `R1_READ_HIGH_INFLUENCE_ANOMALY` | `stress_window_with_downstream` |
| `H9` | `R3_READ_ANOMALY_PLUS_STATIC_RESCUE` | `rolling100` |
| `H9` | `R4_NEG_CONTROL_SKY_NO_SOURCE_MASS` | `rolling100` |
| `H9` | `R5_NEG_CONTROL_STATIC_RESCUE_ONLY` | `rolling100` |
| `C9` | `R1_READ_HIGH_INFLUENCE_ANOMALY` | `stress_window_with_downstream` |
| `C9` | `R4_NEG_CONTROL_SKY_NO_SOURCE_MASS` | `rolling100` |

Important row details：

| Parent | Candidate | ATE delta | `[200,300)` delta | `[400,600)` delta | Rolling100 best |
|---|---|---:|---:|---:|---:|
| `H9` | `R1` | `-0.5985906589` | `-5.3722604038` | `+0.1976874926` | `-2.5654805304` |
| `C9` | `R1` | `-0.5994474373` | `-5.4352915618` | `+0.2103211841` | `-2.5857378482` |
| `H9` | `R3` | `+1.1768136899` | `-8.5850808794` | `+3.4523526076` | `-3.0491654688` |
| `H9` | `R4` | `+3.5374375193` | `-0.6447533379` | `+6.0189397928` | `-3.2933776653` |
| `C9` | `R4` | `+3.5833004091` | `-0.3917509141` | `+6.0534247971` | `-3.3097470786` |

Interpretation：

```text
v41 successfully found h10 local diagnostic signal.

R1 is the safest h10 row:
    stress-window improvement passes and downstream remains below +1m.

R3/R4/R5 rolling-window passes are not safe deployment evidence:
    R3 has strong stress improvement but large downstream regression.
    R4/R5 are negative/control-like rows with large downstream regression.
```

Decision：

```text
Phase 3 h10 gate = pass.
Only h10 gate-pass rows plus corresponding baselines were continued to h15.
```

---

## 5. Phase 3：READ h15

输出：

```text
phase3_read_h15/h15_R1/
phase3_read_h15/report_h15_R1_H9/
phase3_read_h15/report_h15_R1_C9/
```

设置：

```text
H9 candidates:
    V31_BASE_H9_REFERENCE
    R1_READ_HIGH_INFLUENCE_ANOMALY
    R3_READ_ANOMALY_PLUS_STATIC_RESCUE
    R4_NEG_CONTROL_SKY_NO_SOURCE_MASS
    R5_NEG_CONTROL_STATIC_RESCUE_ONLY

C9 candidates:
    V31_BASE_H9_REFERENCE
    R1_READ_HIGH_INFLUENCE_ANOMALY
    R4_NEG_CONTROL_SKY_NO_SOURCE_MASS

rows = 8/8
failures = 0
```

H9 h15：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `R1_READ_HIGH_INFLUENCE_ANOMALY` |
| `best_ATE_delta_vs_base` | `-0.1660176140m` |
| `best_rolling100_candidate` | `R1_READ_HIGH_INFLUENCE_ANOMALY` |
| `best_rolling100_delta` | `-1.6499290343m` |
| `best_stress_candidate` | `R1_READ_HIGH_INFLUENCE_ANOMALY` |
| `best_stress_delta` | `-4.7873967631m` |
| `best_downstream_400_600_delta_for_best_ATE` | `+0.2449909563m` |

C9 h15：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `R1_READ_HIGH_INFLUENCE_ANOMALY` |
| `best_ATE_delta_vs_base` | `-0.1484353180m` |
| `best_rolling100_candidate` | `R1_READ_HIGH_INFLUENCE_ANOMALY` |
| `best_rolling100_delta` | `-1.6218203375m` |
| `best_stress_candidate` | `R1_READ_HIGH_INFLUENCE_ANOMALY` |
| `best_stress_delta` | `-4.8163673672m` |
| `best_downstream_400_600_delta_for_best_ATE` | `+0.2698713541m` |

Decision：

```text
Phase 3 h15 gate = fail for both H9 and C9.

The R1 local stress signal remains close to the -5m threshold, but misses it:
    H9 = -4.7873967631m
    C9 = -4.8163673672m

Full short ATE and rolling100 both weaken strongly at h15:
    H9 ATE = -0.1660176140m, rolling100 = -1.6499290343m
    C9 ATE = -0.1484353180m, rolling100 = -1.6218203375m

No boundary h20 diagnostic was launched because h15 fails the primary signal
thresholds directly. No full-online continuation is allowed.
```

---

## 6. Phase 5：Washout / Failure Routing

输出：

```text
phase5_washout/R1_H9/
phase5_washout/R1_C9/
phase5_washout/R3_H9/
```

Boundary：

```text
Evidence level = proxy_only_no_tensor_state_snapshots.
No tensor-state overwrite norm or true memory-state causality is claimed.
```

R1 H9：

| Metric | Value |
|---|---:|
| `ATE_delta durability` | `0.2773474850` |
| `rolling100_best_delta durability` | `0.8577451063` |
| `stress_200_300_delta durability` | `0.8911326709` |
| `downstream_400_600_delta durability` | `1.2392840428` |
| `chunk_attention_source_keep tail/h10` | `0.4542793707` |
| `frame_attention_bias tail/h10` | `0.4439106063` |
| `ttt_state tail/h10` | `0.4259168776` |
| `swa_source_replace tail/h10` | `0.0` |

R1 C9：

| Metric | Value |
|---|---:|
| `ATE_delta durability` | `0.2476202392` |
| `rolling100_best_delta durability` | `0.8360702240` |
| `stress_200_300_delta durability` | `0.8861286118` |
| `downstream_400_600_delta durability` | `1.2831391853` |
| `chunk_attention_source_keep tail/h10` | `0.4542664818` |
| `frame_attention_bias tail/h10` | `0.4388786346` |
| `ttt_state tail/h10` | `0.4238951346` |
| `swa_source_replace tail/h10` | `0.0` |

R3 H9：

| Metric | Value |
|---|---:|
| `ATE_delta durability` | `2.6045714512` |
| `rolling100_best_delta durability` | `0.1205039811` |
| `stress_200_300_delta durability` | `0.5129897295` |
| `downstream_400_600_delta durability` | `1.2101970660` |
| `ttt_state tail/h10` | `0.4393180545` |

Interpretation：

```text
R1 preserves much of its stress-window improvement from h10 to h15:
    H9 stress durability = 0.8911
    C9 stress durability = 0.8861

But the absolute h15 stress improvement still misses -5m, and full short ATE
falls to about -0.15m to -0.17m. Thus the failure is not a pure immediate
collapse; it is a near-threshold local signal that is too weak globally.

Proxy tail activity remains nonzero for chunk_attention/source_keep,
frame_attention_bias, and ttt_state, while SWA replace is 0. This supports a
read-path / TTT-tail interaction hypothesis, but it is not tensor-state proof.
```

---

## 7. Full Online

输出：

```text
final_reports/full_online_report.md
```

Decision：

```text
No v41 full-online row was launched.

Reason:
    Full online requires h15-qualified candidates.
    H9 h15 gate_pass = false.
    C9 h15 gate_pass = false.

full_online_allowed = false
full_online_launched = false
target30_success = false
```

---

## 8. Final Reports / Decision

Final reports：

```text
final_reports/health_detector_report.md
final_reports/read_mechanism_report.md
final_reports/read_h10_candidate_report.md
final_reports/read_h15_candidate_report.md
final_reports/washout_report.md
final_reports/full_online_report.md
final_reports/failure_routing_summary.md
final_reports/v41_final_summary.json
```

Final summary：

```text
phase1_health_detector_gate_pass = true
selected_bad_chunks = [10]
selected_bad_chunk_ratio = 0.3333333333
mechanism_decision = B_general_high_influence_anomaly_preferred
sky_causality_proven = false
scalar_attention_mass_rows = 144
h10_gate_pass = true
h10_best_ATE_delta = -0.5994474373
h10_best_rolling100_delta = -3.3097470786
h10_best_stress_delta = -8.5850808794
h15_H9_gate_pass = false
h15_H9_best_ATE_delta = -0.1660176140
h15_H9_best_rolling100_delta = -1.6499290343
h15_H9_best_stress_delta = -4.7873967631
h15_C9_gate_pass = false
h15_C9_best_ATE_delta = -0.1484353180
h15_C9_best_rolling100_delta = -1.6218203375
h15_C9_best_stress_delta = -4.8163673672
full_online_allowed = false
full_online_launched = false
target30_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029
```

Downstream decision：

| Stage | Status | Reason |
|---|---|---|
| Phase 1 health detector | pass | selected chunk10 from landed health metrics without ATE/fixed-chunk runtime gate |
| Phase 2 mechanism audit | complete | scalar attention mass landed; sky causality still not proven spatially |
| Phase 3 READ h10 | pass diagnostic | R1 safe stress-window signal; R3/R4/R5 rolling-only rows unsafe due downstream |
| Phase 3 READ h15 | fail | R1 remains closest but misses stress, ATE, and rolling thresholds |
| Phase 5 washout | complete | proxy-only; no tensor-state causality claimed |
| Full online | not launched | no h15-qualified candidate |
| Target-30 | fail / not produced | no full-online candidate allowed |
| Deployable online success | no | C9 remains best deployable |

Conclusion：

```text
v41 produced the clearest read-first diagnostic so far:
    health detector selected chunk10 without using ATE/fixed chunk gating,
    R1_READ_HIGH_INFLUENCE_ANOMALY passed h10 with safe downstream behavior,
    R3 found a very strong h10 stress-window local improvement.

However, h15 did not qualify:
    R1 H9 stress = -4.7873967631m
    R1 C9 stress = -4.8163673672m
    both miss the -5m stress threshold,
    and short ATE/rolling100 weaken far below continuation thresholds.

Therefore v41 is a read-first local diagnostic success but not a durable
Target-30 solution.

Do not promote v41 as deployable online success.
```

Next required direction：

```text
Do not launch v41 full online from current rows.
Do not claim Target-30 from h10-only or near-threshold h15 local signals.

If continuing this family, the closest lead is R1:
    preserve its h10/h15 stress-window behavior,
    lift h15 stress past -5m,
    improve full short ATE beyond diagnostic-only scale,
    and land real per-label/spatial source-attention or tensor-state evidence.

R3/R4-style rolling-only improvements need downstream regression control before
they can be considered for deployment.
```
