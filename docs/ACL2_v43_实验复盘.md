# ACL2 v43 实验复盘：C9 Dechunk Attribution SemanticRead Target30

日期：2026-05-26（Asia/Singapore）  
计划文件：`docs/ACL2_v43_C9_Dechunk_Attribution_SemanticRead_Target30_Plan.md`  
执行日志：`docs/ACL2_v43_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/`

本轮原则：只记录实际落盘结果；不把 component attribution、short/proxy/action audit、blocked downstream stage 写成 deployable online success。每个 LoGeR 进程只绑定一张 GPU；用户确认 GPU 0,1,2,3,4,5,6,7 可用。

---

## 0. 当前状态

```text
v43 已按计划完成 Phase 0-5 与 final report 汇总。

已完成：
    1. 阅读 v43 计划。
    2. 建立主结果目录。
    3. 复用 v36B H9/C9 parent snapshots symlink。
    4. 新增 v43 full-online launcher。
    5. 新增 Phase 0 no-op gate report。
    6. py_compile / bash -n 验证通过。
    7. Phase 0 C9 locked repeat 完成并通过 no-op gate：
           ATE = 33.76294210291885m
           historical C9_P0_R2 = 33.7629421029m
           abs_delta = 0.00000000001885m
           hmc_rows = 38
           effective_config_unexpected_diff_count = 0
    8. Phase 1 C9-flat dechunk candidates 完成：
           best flat = FLAT_01
           best flat ATE = 35.29521801485317m
           delta vs C9 = +1.5322759119343203m
           c9_flat_acceptable = false
    9. Phase 2 component attribution 完成：
           largest positive component = TTT tri-replay
           ATTR_05 remove TTT tri-replay ATE = 36.2098947787m
           delta vs C9 = +2.4469526758m
           ATTR_02 remove tri gamma chunk map ATE = 34.7339675087m
           delta vs C9 = +0.9710254058m
    10. Phase 3 C9-base semantic READ 完成：
           best candidate = SEM_READ_03_C23_RESID_READ_ONLY
           best ATE = 33.4875667508m
           delta vs C9 = -0.2753753521m
           minimum_progress_pass = false
    11. Phase 4 combos not launched:
           best semantic improvement < 0.3m and ATE > 33.3m
    12. Phase 5 cross-sequence sanity not launched:
           KITTI01 did not reach <=33.0m or >=0.5m improvement

尚未完成：
    None

Final:
    full_online_launched = true
    target30_success = false
    deployable_online_success = false
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

---

## 1. 工程修改

Initial setup：

```text
Created result root:
    results/kitti01_hmc_v2/acl2_v43_c9_dechunk_attribution_semanticread_target30/

Reused v36B H9/C9 parent state + merge snapshots through symlink:
    phase0_parent_snapshots ->
    ../acl2_v36b_nooverblocking_semanticmemory_control_target30/phase0_parent_snapshots
```

新增：

```text
tools/run_v43_full_candidate.sh:
    added v43 full-online launcher for:
        P0_F0_C9_LOCKED_REPEAT
        FLAT_01..04
        ATTR_01..07
        SEM_READ_01..04
        COMBO_01..04 placeholders

    The launcher starts from repaired v42 C9 locked defaults:
        Stage C off by default
        C9 read cue
        C9 read_beta_frame_chunks
        C9 tri-replay chunk gammas
        C9 commit EMA chunks
        C9 native mix scales
        C9 SWA overlap source replacement

    Semantic READ candidates enable Stage C explicitly and do not use
    semantic_action_active_chunks by default, matching v43's no absolute
    chunk-id runtime boundary.

tools/v43_noop_gate_report.py:
    reads landed F0 artifacts and writes:
        effective_config.yaml
        effective_config_diff_vs_C9.json
        noop_gate_summary.json

tools/v43_registry_summarize.py:
    summarizes landed full-online registries into v43 phase reports.
    For Phase 2 it additionally writes component contribution CSVs and
    available plots from landed metrics.
```

Validation：

```text
bash -n tools/run_v43_full_candidate.sh = PASS
py_compile tools/v43_noop_gate_report.py tools/v43_registry_summarize.py = PASS
```

Runtime boundary：

```text
v43 remains training-free.
No learned trigger/router/classifier is introduced.
No GT SemanticKITTI label is used as runtime action.
Phase 1+ dechunk candidates remove absolute chunk-id maps only where the plan
explicitly specifies that candidate change.
```

---

## 2. Phase 0：C9 no-op locked repeat

输出：

```text
phase0_c9_repeat/rollouts/V43_P0_F0_C9_LOCKED_REPEAT/
phase0_c9_repeat/report_R1/
```

Summary：

| Metric | Value |
|---|---:|
| `phase0_noop_gate_pass` | `true` |
| `ATE` | `33.76294210291885m` |
| `historical_C9_P0_R2_ATE` | `33.7629421029m` |
| `abs_delta_vs_historical_C9` | `0.00000000001885m` |
| `hmc_rows` | `38` |
| `effective_config_unexpected_diff_count` | `0` |
| `stage_c_disabled` | `true` |
| `semantic_action_disabled` | `true` |

Decision：

```text
Phase 0 gate = pass.
Phase 1 full-online C9-flat / dechunk candidates are allowed.
```

---

## 3. Phase 1：C9-flat dechunk full runs

状态：

```text
DONE
```

Rows：

| Candidate | Run | Status | ATE | Delta vs C9 |
|---|---|---|---:|---:|
| `FLAT_01` | `V43_P1_FLAT_01` | `DONE` | `35.2952180149m` | `+1.5322759119m` |
| `FLAT_02` | `V43_P1_FLAT_02` | `DONE` | `35.5004971353m` | `+1.7375550324m` |
| `FLAT_03` | `V43_P1_FLAT_03` | `DONE` | `35.3608497931m` | `+1.5979076901m` |
| `FLAT_04` | `V43_P1_FLAT_04` | `DONE` | `36.5229452729m` | `+2.7600031699m` |

Summary：

| Metric | Value |
|---|---:|
| `best_flat_candidate` | `FLAT_01` |
| `best_flat_ATE` | `35.29521801485317m` |
| `best_flat_delta_vs_C9` | `+1.5322759119343203m` |
| `c9_flat_acceptable` | `false` |
| `c9_flat_promising` | `false` |
| `c9_flat_breakthrough` | `false` |

Decision：

```text
Phase 1 C9-flat = fail.
All FLAT rows are worse than C9 by more than +0.5m.

Per v43 failure routing, C9's chunk-id maps / interactions are not harmless
details. C9 remains the deployment baseline but should be treated as a
diagnostic best rather than a mechanism-clean final strategy.

Phase 2 contribution attribution is required.
Phase 3 semantic READ uses C9 locked as the official base.
```

---

## 4. Phase 2：C9 component attribution

状态：

```text
DONE
```

Rows started before Phase 1 completion because they do not depend on Phase 1 best gamma / best flat:

| Candidate | Run | GPU | Component |
|---|---|---:|---|
| `ATTR_01_C9_MINUS_READ_MAP_TO_FLAT` | `V43_P2_ATTR_01_C9_MINUS_READ_MAP_TO_FLAT` | `4` | read beta chunk map -> global 4.75 |
| `ATTR_03_C9_MINUS_COMMIT_EMA` | `V43_P2_ATTR_03_C9_MINUS_COMMIT_EMA` | `5` | disable commit EMA chunks |
| `ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE` | `V43_P2_ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE` | `6` | disable SWA overlap source replacement |
| `ATTR_05_C9_MINUS_TTT_TRI_REPLAY` | `V43_P2_ATTR_05_C9_MINUS_TTT_TRI_REPLAY` | `7` | disable TTT tri-replay |
| `ATTR_06_C9_MINUS_NATIVE_MIX` | `V43_P2_ATTR_06_C9_MINUS_NATIVE_MIX` | `3` | native mix scales -> `1.00,1.00,1.00` |
| `ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT` | `V43_P2_ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT` | `0` | tri gamma chunk map -> global `0.003` |

Pending / reused:

```text
ATTR_07_C9_NO_CHUNK_ID_ALL:
    represented by exact landed FLAT_01 row because ATTR_07 is defined as same
    as Phase 1 best flat. This is exact artifact reuse, not an inferred metric.
```

Early raw landed results:

```text
ATTR_01_C9_MINUS_READ_MAP_TO_FLAT:
    raw results_sim3 ATE = 33.789524m

ATTR_03_C9_MINUS_COMMIT_EMA:
    raw results_sim3 ATE = 34.251324m

ATTR_04_C9_MINUS_SWA_OVERLAP_REPLACE:
    raw results_sim3 ATE = 33.819190m

ATTR_05_C9_MINUS_TTT_TRI_REPLAY:
    raw results_sim3 ATE = 36.209889m

ATTR_06_C9_MINUS_NATIVE_MIX:
    raw results_sim3 ATE = 33.854642m

ATTR_02_C9_MINUS_TRI_CHUNKMAP_TO_FLAT:
    raw results_sim3 ATE = 34.733965m
```

Preliminary interpretation：

```text
Read beta chunk map removal is nearly neutral by raw ATE.
Commit EMA removal is moderately harmful.
SWA overlap source replacement removal is near-neutral/slightly harmful.
Removing TTT tri-replay is strongly harmful.

The final ledger below uses exact full-online registry metrics.
```

Final contribution ledger：

| Candidate | ATE | Delta vs C9 | Class | Interpretation |
|---|---:|---:|---|---|
| `ATTR_01` | `33.7895226653m` | `+0.0265805624m` | neutral | read beta chunk map is nearly neutral |
| `ATTR_02` | `34.7339675087m` | `+0.9710254058m` | major positive | tri gamma chunk map is important |
| `ATTR_03` | `34.2513275668m` | `+0.4883854639m` | moderate positive | commit EMA contributes moderately |
| `ATTR_04` | `33.8191923480m` | `+0.0562502451m` | neutral | SWA overlap replacement is near-neutral by full ATE |
| `ATTR_05` | `36.2098947787m` | `+2.4469526758m` | major positive | TTT tri-replay is the largest positive component |
| `ATTR_06` | `33.8546448528m` | `+0.0917027499m` | neutral | native mix is near-neutral |
| `ATTR_07` | `35.2952180149m` | `+1.5322759119m` | major positive | full no-chunk-id flat strongly regresses |

Decision：

```text
Phase 2 contribution ledger = complete.

C9's main full-online gain is not from semantic READ; it is dominated by TTT
tri-replay and chunk-specific tri gamma / no-chunk-id interaction. Commit EMA
is a moderate positive component. Read beta map, SWA overlap source replacement,
and native mix are near-neutral by full ATE.

This supports the v43 hypothesis that C9 is an effective but chunk-map-heavy
diagnostic best, not a mechanism-clean final policy.
```

---

## 5. Phase 3：Minimal semantic READ full online

状态：

```text
C9-BASE OFFICIAL DONE
```

Boundary：

```text
These C9-base semantic READ rows were launched early to use idle GPUs after
several Phase 2 rows finished.

After Phase 1 completed, best C9-flat was not acceptable
(`+1.5322759119m` vs C9), so these rows are now the official Phase 3 C9-base
semantic READ matrix.
```

Rows：

| Candidate | Run | GPU |
|---|---|---:|
| `SEM_READ_01_HIGH_INFLUENCE_ANOMALY` | `V43_P3_C9BASE_SEM_READ_01_HIGH_INFLUENCE_ANOMALY` | `4` |
| `SEM_READ_02_GUARDED_SKY_VEG_DYNAMIC` | `V43_P3_C9BASE_SEM_READ_02_GUARDED_SKY_VEG_DYNAMIC` | `5` |
| `SEM_READ_03_C23_RESID_READ_ONLY` | `V43_P3_C9BASE_SEM_READ_03_C23_RESID_READ_ONLY` | `6` |
| `SEM_READ_04_ANOMALY_STATIC_RESCUE` | `V43_P3_C9BASE_SEM_READ_04_ANOMALY_STATIC_RESCUE` | `7` |

Full-online metrics：

| Candidate | ATE | Delta vs C9 | `[200,300)` delta | `[400,600)` delta | Rolling100 best delta | Attention mass rows | Context empty source events |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SEM_READ_01` | `35.9580864086m` | `+2.1951443057m` | `-3.7952646498m` | `+2.1065748253m` | `-3.9784499836m` | `38` | `0` |
| `SEM_READ_02` | `54.4017022529m` | `+20.6387601500m` | `+8.3245155082m` | `+22.3763732284m` | `+1.5303879826m` | `0` | `0` |
| `SEM_READ_03` | `33.4875667508m` | `-0.2753753521m` | `-0.4103324649m` | `-0.7851119221m` | `-1.1407966182m` | `0` | `0` |
| `SEM_READ_04` | `53.9372336259m` | `+20.1742915230m` | `+2.4472796283m` | `+22.9961337776m` | `-3.6011151265m` | `38` | `0` |

Summary：

| Metric | Value |
|---|---:|
| `best_candidate` | `SEM_READ_03_C23_RESID_READ_ONLY` |
| `best_ATE_full` | `33.487566750822836m` |
| `best_delta_vs_C9` | `-0.27537535209601316m` |
| `minimum_progress_pass` | `false` |
| `stage_success_pass` | `false` |
| `strong_success_pass` | `false` |
| `target30_success` | `false` |

Interpretation：

```text
SEM_READ_03 is the only v43 semantic READ row with full-online ATE improvement:
    ATE = 33.4875667508m
    delta vs C9 = -0.2753753521m

This is real landed positive signal, but it is below the v43 minimum progress
boundary:
    required improvement >= 0.3m or ATE <= 33.3m
    observed improvement = 0.2753753521m
    observed ATE = 33.4875667508m

SEM_READ_01 has local [200,300) improvement but regresses downstream by more
than +1m. SEM_READ_02 and SEM_READ_04 are severe regressions.
```

Decision：

```text
Phase 3 semantic READ = fail minimum progress.
No Phase 3 row reaches Target-30.
No Phase 3 row is deployable online success.
```

---

## 6. Phase 4：Minimal combinations

Decision：

```text
Phase 4 was not launched.

Reason:
    Phase 4 requires a semantic READ candidate with either:
        improvement >= 0.3m vs C9, or
        full ATE <= 33.3m.

The best semantic READ row is SEM_READ_03:
    ATE = 33.4875667508m
    improvement vs C9 = 0.2753753521m

This misses both continuation criteria, so launching COMBO rows would violate
the v43 plan and would be diagnostic fishing.
```

---

## 7. Phase 5：Cross-sequence sanity

Decision：

```text
Phase 5 was not launched.

Reason:
    Phase 5 requires KITTI01 evidence strong enough to justify cross-sequence
    validation:
        full ATE <= 33.0m, or
        improvement >= 0.5m vs C9.

The best v43 KITTI01 semantic READ result is:
    SEM_READ_03 ATE = 33.4875667508m
    improvement = 0.2753753521m

Therefore cross-sequence validation is not allowed from current v43 rows.
```

---

## 8. Final Reports / Decision

Final reports：

```text
final_reports/v43_final_summary.md
final_reports/v43_final_summary.json

phase1_flat/report_R1/v43/phase1_flat_report.md
phase2_attribution/report_R1/v43/phase2_attribution_report.md
phase2_attribution/report_R1/v43/component_contribution_ate.csv
phase2_attribution/report_R1/v43/component_contribution_segments.csv
phase2_attribution/report_R1/v43/component_contribution_rolling.csv
phase3_semantic_read/report_R1/v43/phase3_semantic_read_report.md
```

Final summary：

```text
phase0_noop_gate_pass = true
c9_ate = 33.76294210291885
phase1_best_flat = FLAT_01
phase1_best_flat_ate = 35.29521801485317
phase1_best_flat_delta_vs_C9 = +1.5322759119343203
phase1_c9_flat_acceptable = false
phase2_largest_positive_component = ATTR_05_C9_MINUS_TTT_TRI_REPLAY
phase2_largest_positive_component_delta = +2.4469526757787676
phase3_best_semantic_read = SEM_READ_03_C23_RESID_READ_ONLY
phase3_best_semantic_read_ate = 33.487566750822836
phase3_best_semantic_read_delta_vs_C9 = -0.27537535209601316
phase3_minimum_progress_pass = false
phase4_launched = false
phase5_launched = false
target30_success = false
deployable_online_success = false
best_deployable_online = C9_P0_R2
best_deployable_online_ate = 33.7629421029
```

Downstream decision：

| Stage | Status | Reason |
|---|---|---|
| Phase 0 C9 no-op | pass | exact locked C9 repeat, `abs_delta = 1.885e-11m` |
| Phase 1 C9-flat | fail | best flat regresses C9 by `+1.5323m` |
| Phase 2 attribution | complete | TTT tri-replay and tri gamma chunk map are main positive C9 components |
| Phase 3 semantic READ | fail minimum progress | SEM_READ_03 improves by `-0.2754m`, below `0.3m` continuation boundary |
| Phase 4 combinations | not launched | no semantic READ candidate passed continuation boundary |
| Phase 5 cross-sequence | not launched | KITTI01 did not reach `<=33.0m` or `>=0.5m` improvement |
| Target-30 | fail | no row reaches `<=30m` |
| Deployable online success | no | C9 remains best deployable |

Conclusion：

```text
v43 answered two important questions.

First, dechunking C9 is not a free simplification:
    all C9-flat candidates regress substantially.
    C9 depends strongly on chunk-specific TTT tri-replay / tri gamma structure.

Second, semantic READ can produce a small full-online improvement:
    SEM_READ_03_C23_RESID_READ_ONLY reaches 33.4875667508m,
    improving C9 by 0.2753753521m.

But this does not meet the v43 minimum progress boundary and is still far from
Target-30. It is a useful diagnostic signal, not a deployable success.

Do not promote v43 as deployable online success.
```

Next required direction：

```text
Do not launch v43 Phase 4 or Phase 5 from current rows.
Do not claim Target-30 from SEM_READ_03.

If continuing, the closest lead is C9 + C23 residual read-only, but it needs:
    at least +0.3m to +0.5m robust full-online improvement,
    no severe downstream regression,
    and better causal evidence for where the residual read path helps.

The component ledger also says that any future simplification must preserve the
TTT tri-replay / tri gamma mechanism or replace it with a non-chunk-id
mechanism that proves equivalent full-online behavior.
```
