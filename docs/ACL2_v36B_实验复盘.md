# ACL2 v36B 实验复盘：NoOverblocking SemanticMemory Control Target30

日期：2026-05-24（Asia/Singapore）  
计划文件：`docs/ACL2_v36B_NoOverblocking_SemanticMemory_Control_Target30_Plan.md`  
执行日志：`docs/ACL2_v36B_执行日志.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v36b_nooverblocking_semanticmemory_control_target30/`

本轮原则：只记录实际落盘结果；不把 H0 smoke、short rollout、fixed/repair diagnostic、复用 full evidence 或 blocked stage 写成新的 deployable online success。每个 LoGeR 进程只绑定一张 GPU；并行只来自多个独立 row 分配到不同 GPU。

---

## 0. 当前结论

v36B 修正了 v36 的 overblocking：H0 attention-mass 缺失不再阻止 short rollout；但最终仍未达到 Target-30，没有产生新的 deployable online result。

已完成并落盘：

```text
1. 生成 v36B H9/C9 parent snapshots:
       chunks = 6,10,16
       state + merge snapshots complete

2. H0C action distinguishability smoke passed:
       rows_done = 7
       any_non_base_pair_distinguishable = true
       any_source_skip_effect = true
       context_empty_source_events_total = 0

3. H0B attention-mass feasibility audit completed:
       source_removal_semantics_match_vggt4d = true
       old landed_attention_mass_available = false
       sample_records_checked = 198
       sample_records_with_any_attention_mass = 0

4. H0B runtime instrumentation repair completed on one representative row:
       attention_mass_removed_available = true
       attention_mass_status = sampled-qk-softmax-mass
       mass_rows = 4
       mean_removed_before = 0.0549059110
       mean_removed_after = 0.0
       mean_retained_before = 0.9450940639
       mean_retained_after = 1.0

5. H1 frame/global source-skip h10 R2 completed:
       rows = 18/18
       gate_pass = false

6. H2 SWA h10 completed:
       H9/C9 chunk10 short gate pass
       boundary no-regression check pass under v36B <= +0.25m rule

7. H2 SWA h15 completed:
       gate_pass = false

8. H2 washout attribution completed:
       evidence_level = proxy_only_no_tensor_state_snapshots
       H9 h15/h10 [200,300) durability = 0.3210351835
       C9 h15/h10 [200,300) durability = 0.2471427261

9. H3 TTT semantic static/negative h10 completed:
       rows = 24/24
       gate_pass = false

10. H4 semantic C23 path-isolation h10 completed:
       initial beta 4.75 nearly passed but missed -5m gate

11. H4 minimal beta525 repair completed:
       h10 pass
       h15 pass

12. H5 full-online evidence was first evaluated by reusing exact landed
   full-online rows from v32:
       C9 + SEM_Z_COARSE_BETA525 all chunks
       full ATE = 34.5597307381m
       Target-30 fail
       C9 + SEM_RESID_COARSE_L025 all chunks
       full ATE = 34.3258261120m
       Target-30 fail

13. Per v36B 10.6, H5 path-isolation full online was then launched:
       C9_NO_SWA_BASE
       SEM_Z_NO_SWA
       RESID_NO_SWA
       SEM_Z_READONLY
       RESID_READONLY
   all rows DONE, all worse than C9, Target-30 fail.
```

最终边界：

```text
1. H4 beta525 repair has real h10/h15 local diagnostic signal.
2. The all-chunks full-online semantic-z/residual versions are worse than C9.
3. The path-isolation full rows also fail:
       best new path-isolation row = RESID_NO_SWA
       full ATE = 38.2740049300m
       delta vs C9 = +4.5110628270m
4. H0B attention-mass repair is representative instrumentation evidence only,
   not a trajectory-improvement result.
5. No v36B row reaches full ATE <= 30m.
6. No new deployable online success is produced.
7. Current best deployable online TTT write remains C9_P0_R2.
```

当前 best deployable online TTT write：

```text
C9_P0_R2
ATE = 33.7629421029m
```

---

## 1. 工程修改

新增 / 使用：

```text
tools/run_v36b_snapshot_generation.sh:
    generate H9/C9 parent state + merge snapshots for chunks 6,10,16.

tools/run_v36b_h0c_action_smoke.sh:
    run H0C action distinguishability smoke.

tools/v36b_h0c_action_smoke_report.py:
    aggregate H0C action equivalence / distinguishability.

tools/v36b_context_skip_summary.py:
    summarize H1 context source-skip landed effects and attention-mass boundary.

tools/v36b_h0b_attention_mass_feasibility_audit.py:
    compare LoGeR compact K/V source-removal implementation against VGGT4D
    source-only K/V compaction and audit whether landed v36B artifacts contain
    before/after attention-mass fields.

loger/models/layers/attention.py:
    added default-off sampled qk-softmax mass recording inside compact_kv SDPA.

loger/models/pi3.py:
    passes the optional mass collector through compact_kv attention and writes
    per-layer scalar mass fields into HMC trace records.

loger/pipeline/hybrid_memory_controller.py:
    added default-off context_source_skip_record_attention_mass config and
    aggregates attention mass into hook_effect_summary.

run_pipeline_abc_v2.py:
tools/run_attention_cue_experiment.sh:
tools/run_v24_candidate_rollout.sh:
    added / forwarded default-off H0B attention-mass instrumentation switches.

tools/run_v36b_h1_h10.sh:
    H1 scheduler repaired to one worker queue per GPU.

tools/run_v36b_path_h10.sh:
    common one-worker-per-GPU scheduler for H2/H3/H4 h10/h15.
    supports explicit v36B parent snapshots through LOAD_HMC_STATE_AT_CHUNK and
    LOAD_MERGE_STATE_AT_CHUNK.

tools/v31_track_f_washout_attribution.py:
    reused for H2 h10-effective / h15-failed washout attribution.
    Evidence is proxy-only when tensor snapshots are absent.

tools/v32_transfer_report.py:
    reused to aggregate landed v32 full-online SEM_Z and RESID rows under the
    v36B H5 evidence section.
```

验证：

```text
bash -n: PASS
py_compile: PASS
```

Production runtime boundary：

```text
v36B did not require new production runtime hooks beyond landed v36/v31-v32
hook/cue support. v36B changes were scheduling and reporting tools.
```

---

## 2. H0：工程自查

H0A inherited v36 hook reachability evidence：

```text
frame/global synthetic source skip:
    max_context_source_skip_tokens > 0
    context_empty_source_events = 0

SWA synthetic remove:
    num_source_gate_applied > 0

TTT synthetic negative:
    post-zp/action-delta tensor trace changed
```

H0B：

```text
phase0b_attention_mass_feasibility_audit_R1:
    source_removal_semantics_match_vggt4d = true
    landed_attention_mass_available = false
    sample_files_checked = 6
    sample_records_checked = 198
    sample_records_with_any_attention_mass = 0

Code-level conclusion:
    LoGeR compact_kv source-removal keeps all query rows and removes K/V source
    tokens through source_keep_mask, matching VGGT4D's source-only K/V subset
    pattern for dynamic/source removal.

Artifact boundary:
    attention_mass_removed_before/after not landed.
    q/k tensors or attention probability tensors were not landed.
    Therefore H1/H2 source-skip/SWA mechanism claims are labeled:
        source-count/action-summary supported
        attention-mass causality not yet proven

Repair:
    Added default-off sampled qk-softmax mass instrumentation for compact_kv.

phase0b_attention_mass_rerun_R1:
    representative row = H9 / FG_RISK_00 / chunk10 / h3
    run_status = DONE
    attention_mass_removed_available = true
    attention_mass_status = sampled-qk-softmax-mass
    mass_rows = 4
    paths_with_mass = [chunk_attention]
    chunks_with_mass = [10,11,12,13]
    mean_removed_before = 0.0549059110
    mean_removed_after = 0.0
    mean_retained_before = 0.9450940639
    mean_retained_after = 1.0
    context_empty_source_events_total = 0

Boundary:
    This is a representative H0B instrumentation validation, not a full H1/H2
    matrix rerun and not a Target-30 trajectory result.
```

H0C：

```text
h0c_action_smoke_gate_pass = true
rows_done = 7
missing_rows = 0
any_non_base_pair_distinguishable = true
any_source_skip_effect = true
context_empty_source_events_total = 0
```

Action collapse:

```text
FG_RISK_00 / FG_SEM_01 / FG_SEM_02 / FG_SEM_03 collapse.
FG_SEM_04 / FG_SEM_05 collapse.
```

Decision：

```text
H1 allowed only with representative source-skip candidates.
H2/H3/H4 allowed as independent path probes.
```

---

## 3. Blocker 与修复记录

### Blocker 1：H1 R1 scheduler caused CUDA OOM

现象：

```text
V36B_H1_H10_R1_H9_FG_SEM_04_chunk10_h10...
failed with CUDA OOM.
```

原因：

```text
Initial scheduler reused GPU before a heavy row finished.
```

修复：

```text
tools/run_v36b_h1_h10.sh:
    one worker queue per GPU
    independent GPUs run in parallel
    no GPU receives two LoGeR rows at once
```

结果：

```text
H1 R2 completed 18/18 rows with failures = 0.
H1 R1 kept only as invalid/partial blocker evidence.
```

### Blocker 2：GPU 4-7 cannot safely steal active queued rows

现象：

```text
User reported GPU 4,5,6,7 temporarily free.
```

审计：

```text
tools/run_v24_candidate_rollout.sh moves an existing non-DONE run directory to
.INVALID_RERUN_* if the same RUN_NAME is launched again.
```

处理：

```text
Do not launch duplicate H3/H4 rows with same prefix while old queues are active.
Use GPU 4/5/6/7 only for new non-conflicting beta525 repair phases.
```

结果：

```text
phase4_h4_semc23_repair_beta525_h10 used GPU 4/5/6/7.
phase4_h4_semc23_repair_beta525_h15 used GPU 4/5/6/7.
No active run directory was invalidated.
```

### Blocker 3：H4 beta 4.75 nearly passed but missed gate

现象：

```text
H4 initial h10:
    H9 best [200,300) delta = -4.9916037591m
    C9 best [200,300) delta = -4.9458168459m
Gate requires <= -5m.
```

修复方向：

```text
Use a single prior landed conservative repair:
    beta = 5.25
This was not a sweep and was already validated as v31 beta525 repair.
```

结果：

```text
H4 beta525 h10 pass.
H4 beta525 h15 pass.
```

### Blocker 4：H2 h10 signal washed out at h15

现象：

```text
H2 SWA chunk10 h10 passed for H9/C9:
    H9 best [200,300) delta = -3.7351609926m
    C9 best [200,300) delta = -3.4943516930m

H2 SWA chunk10 h15 failed:
    H9 best [200,300) delta = -1.1991180979m
    C9 best [200,300) delta = -0.8636036060m
```

按计划修复 / 审计方向：

```text
Plan 10.3 says h10-effective / h15-failed rows require washout attribution
instead of more threshold sweeping.
```

结果：

```text
phase2_h2_swa_washout_R1 generated proxy-only reports for:
    H9/C9 x SWA_FINE_01/SWA_FINE_04

Evidence level:
    proxy_only_no_tensor_state_snapshots
```

Decision：

```text
H2 is treated as local diagnostic only.
No H2 full-online continuation is allowed.
```

### Blocker 5：C9 full online semantic all-chunks regressed

现象：

```text
C9 + SEM_Z_COARSE_BETA525_ALL reused full row:
    ATE = 34.5597307381m
    delta vs C9 = +0.7967886352m
```

按计划修复 / 审计方向：

```text
Plan 10.6 says C9 full online regression requires path isolation / disabling
conflicting paths. Reuse the already landed v32 conservative residual repair
as full-online evidence, without claiming it as a new v36B run.
```

结果：

```text
C9 + SEM_RESID_COARSE_L025_ALL reused full row:
    ATE = 34.3258261120m
    delta vs C9 = +0.5628840091m
    [400,600) delta vs C9 = +3.1115081606m
```

Decision：

```text
Residual repair reduces the full ATE damage relative to semantic-z all-chunks,
but still regresses C9 and fails Target-30. No H5 selector/full continuation.
```

### Blocker 6：H0B attention-mass fields unavailable in landed artifacts

现象：

```text
v36B plan asks for attention-mass explainability:
    attention_mass_removed_before/after
    retained/group attention mass before/after source removal

Landed v36B context/hmc JSONL samples contain source skip counts / keep ratios,
but no attention-mass fields.
```

按计划审计 / 修复方向：

```text
1. Read third_party/VGGT4D source-removal implementation:
       third_party/VGGT4D/vggt4d/layers/attention.py

2. Read LoGeR compact K/V source-skip implementation:
       loger/models/layers/attention.py
       loger/models/pi3.py

3. Add feasibility audit:
       tools/v36b_h0b_attention_mass_feasibility_audit.py

4. Audit representative landed v36B context/hmc JSONL artifacts.
5. Add default-off sampled attention mass instrumentation.
6. Rerun one representative compact_kv source-skip row.
```

结果：

```text
source_removal_semantics_match_vggt4d = true
sample_records_checked = 198
sample_records_with_any_attention_mass = 0
landed_attention_mass_available = false

After repair representative row:
    attention_mass_removed_available = true
    mass_rows = 4
    mean_removed_before = 0.0549059110
    mean_removed_after = 0.0
    mean_retained_before = 0.9450940639
    mean_retained_after = 1.0
```

Decision：

```text
The source-removal implementation is code-auditable and matches the VGGT4D
source-only K/V compaction pattern.

Old landed artifacts cannot support attention-mass claims, but the repair row
now lands true sampled qk-softmax mass for compact_kv chunk_attention:
    removed source mass before compaction is non-zero,
    removed source mass after compaction is 0,
    retained mass after compaction is 1.

This validates the H0B instrumentation path on a representative row. It is not
retroactive proof for old rows and not a full matrix rerun.
```

---

## 4. H1：Frame/Global Source Skip h10

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates = V31_BASE_H9_REFERENCE, FG_RISK_00, FG_SEM_04
rows = 18/18
failures = 0
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `FG_SEM_04` |
| `best_ATE_delta` | `-0.0473357966m` |
| `best_[200,300)_candidate` | `FG_RISK_00` |
| `best_[200,300)_delta` | `-0.3418221710m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_candidate` | `FG_SEM_04` |
| `best_ATE_delta` | `-0.0694789609m` |
| `best_[200,300)_candidate` | `FG_RISK_00` |
| `best_[200,300)_delta` | `-0.2606290156m` |

Context source:

| Parent | Source-effect rows | Empty-source events | Attention mass |
|---|---:|---:|---|
| `H9` | `6` | `0` | `attention-mass-unverified` |
| `C9` | `6` | `0` | `attention-mass-unverified` |

Decision：

```text
H1 h10 gate = fail.
No H1 h15.
```

---

## 5. H2：SWA Local-Continuity

### h10

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    SWA_FINE_01_OVERLAP_STRUCTURE_KEEP
    SWA_FINE_04_BOUNDARY_PROTECT
    SWA_FINE_05_CACHE_LIFECYCLE
```

H9 h10：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_ATE_delta` | `-1.2136622509m` |
| `best_[200,300)_delta` | `-3.7351609926m` |
| `passing_chunk` | `10` |
| `passing_candidates` | `SWA_FINE_01`, `SWA_FINE_04` |
| `boundary_10f_delta` | `+0.0723922636m` |
| `boundary_20f_delta` | `-0.0176117235m` |

C9 h10：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_ATE_delta` | `-1.2466940180m` |
| `best_[200,300)_delta` | `-3.4943516930m` |
| `passing_chunk` | `10` |
| `passing_candidates` | `SWA_FINE_01`, `SWA_FINE_04` |
| `boundary_10f_delta` | `+0.2221317699m` |
| `boundary_20f_delta` | `+0.1452799891m` |

Boundary note：

```text
tools/v27_swa_boundary_diagnostics.py reports its own h3_boundary_gate_pass=false
because that script expects positive improvement. v36B only requires no large
boundary regression for H2 continuation:
    boundary_10f_delta <= +0.25
    boundary_20f_delta <= +0.25
Both H9 and C9 satisfy the v36B no-regression criterion.
```

### h15

H9 h15：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_[200,300)_delta` | `-1.1991180979m` |

C9 h15：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_[200,300)_delta` | `-0.8636036060m` |

Decision：

```text
H2 has h10 local signal but fails h15 durability.
No H2 full online.
```

### washout attribution

输出：

```text
phase2_h2_swa_washout_R1/
```

Boundary：

```text
Evidence level = proxy_only_no_tensor_state_snapshots.
No tensor snapshot proof of memory overwrite is claimed.
```

| Parent | Candidate | h10 `[200,300)` delta | h15 `[200,300)` delta | h15/h10 abs durability | TTT tail/h10 proxy | Frame-bias tail/h10 | SWA replace tail/h10 |
|---|---|---:|---:|---:|---:|---:|---:|
| `H9` | `SWA_FINE_01` | `-3.7351609926` | `-1.1991180979` | `0.3210351835` | `0.4499735800` | `0.0` | `0.0` |
| `H9` | `SWA_FINE_04` | `-3.7351609926` | `-1.1991180979` | `0.3210351835` | `0.4499735800` | `0.0` | `0.0` |
| `C9` | `SWA_FINE_01` | `-3.4943516930` | `-0.8636036060` | `0.2471427261` | `0.4492046627` | `0.0` | `0.0` |
| `C9` | `SWA_FINE_04` | `-3.4943516930` | `-0.8636036060` | `0.2471427261` | `0.4492046627` | `0.0` | `0.0` |

Interpretation：

```text
The local SWA benefit is not durable:
    H9 retains about 32.1% of the h10 [200,300) improvement at h15.
    C9 retains about 24.7%.

The landed proxy does not show SWA source-replace tail movement, while TTT tail
side-effect proxy remains about 45% of h10. This is not a full tensor-state
proof, but supports stopping H2 rather than expanding SWA rules.
```

---

## 6. H3：TTT Semantic Static / Negative

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    TTT_FINE_01_STRUCTURE_POSITIVE
    TTT_FINE_04_LOWSTUFF_HIGHD_SHORT
    TTT_FINE_RISK_01_CONFLICT_TRI
rows = 24/24
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_delta` | `-0.0955755272m` |
| `best_[200,300)_delta` | `-0.1236567417m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_ATE_delta` | `-0.0962599959m` |
| `best_[200,300)_delta` | `-0.1663533136m` |

Decision：

```text
H3 h10 gate = fail.
No H3 h15.
```

---

## 7. H4：Semantic C23 Path Isolation

### Initial beta 4.75 h10

设置：

```text
parents = H9,C9
chunks = 6,10,16
horizon = 10
candidates =
    V31_BASE_H9_REFERENCE
    V31_A1B_SEM_Z_COARSE
    V31_A5B_SEM_RESID_COARSE_L025
    V31_B0_STATIC_RESCUE_EXISTING
```

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_[200,300)_delta` | `-4.9916037591m` |
| `best_[400,600)_delta` | `+0.5515281383m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `false` |
| `best_[200,300)_delta` | `-4.9458168459m` |
| `best_[400,600)_delta` | `+0.5476704041m` |

### beta525 repair h10

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_[200,300)_delta` | `-6.0997514572m` |
| `best_[400,600)_delta` | `+0.6826731549m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_[200,300)_delta` | `-6.0435649622m` |
| `best_[400,600)_delta` | `+0.6822210800m` |

### beta525 repair h15

H9：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_ATE_delta` | `-0.4434033455m` |
| `best_[200,300)_delta` | `-6.1462634267m` |
| `best_[400,600)_delta` | `+0.2681140634m` |

C9：

| Metric | Value |
|---|---:|
| `gate_pass` | `true` |
| `best_ATE_delta` | `-0.4523904071m` |
| `best_[200,300)_delta` | `-6.1250511839m` |
| `best_[400,600)_delta` | `+0.2516740832m` |

Decision：

```text
H4 beta525 repair passes h10 and h15 local diagnostic gates.
H5 full online evidence is allowed.
```

---

## 8. H5：Full Online Evidence

### 8.1 Reused v32 Full Evidence

Evidence source：

```text
Reused exact landed v32 full-online rows:
results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h2_c9_combo/rollouts/V32_H2_01_C9_SEM_Z_COARSE_BETA525_ALL
results/kitti01_hmc_v2/acl2_v32_semanticcue_transfer_target30/h3_repair/rollouts/V32_H3_01_C9_RESID_COARSE_L025_ALL
```

Why reuse is valid：

```text
The semantic-z landed row is C9 + all-chunks semantic z coarse beta525:
    mode = hybrid
    read_cue_source = v31.sem_z_coarse.c23past
    beta_frame = 5.25
    beta_swa = 5.25
    hmc_commit_mode = probe_ttt_write
    hmc_write_score_source = stage_d_x_dg_inv_sqrt
    frames = 1101

The residual landed row is the conservative residual repair:
    read_cue_source = v31.sem_resid_coarse_l025.c23past
    mode = hybrid
    frames = 1101
```

Boundary：

```text
These are not claimed as newly launched v36B full-online rows.
They are reused landed full-online evidence for the same semantic C23 family
and its conservative residual repair.
```

Full metrics：

| Run | Full ATE | `[200,300)` ATE | `[400,600)` ATE | Delta vs C9 |
|---|---:|---:|---:|---:|
| `C9_REF` | `33.7629421029` | `76.1021355543` | `41.8963642126` | `0.0` |
| `SEM_Z_COARSE_BETA525_ALL` | `34.5597307381` | `76.0287188674` | `43.5473731723` | `+0.7967886352` |
| `SEM_RESID_COARSE_L025_ALL` | `34.3258261120` | `75.4336663355` | `45.0078723731` | `+0.5628840091` |

Residual deltas vs C9：

```text
[200,300) delta = -0.6684692189m
[400,600) delta = +3.1115081606m
```

Decision：

```text
Full online Target-30 gate = fail.
Full online C9 compatibility = fail.
The short h10/h15 local gain does not transfer to all-chunks C9 full online.
The residual repair is less bad in full ATE than semantic-z, but still worse
than C9 and has a large [400,600) regression.
```

### 8.2 Path-Isolation Full Rows

触发原因：

```text
v36B plan 10.6 says C9 full-online regression requires path isolation:
    disable conflicting paths,
    and if semantic read-only still regresses, stop semantic deployment line.

The reused v32 rows were all-chunks hybrid rows with:
    TTT write enabled
    SWA write enabled
    SWA source replacement enabled

Therefore v36B had to test noSWA/read-only path-isolation directly.
```

输出：

```text
phase5_h5_path_isolation_full_R1/rollouts/
phase5_h5_path_isolation_full_R1/report_R1/
```

Rows launched：

| Run | Mode | Cue | SWA write/source replace | Status |
|---|---|---|---|---|
| `V36B_H5_R1_C9_NO_SWA_BASE` | `hybrid` | original C9 cue | off / off | DONE |
| `V36B_H5_R1_SEM_Z_NO_SWA` | `hybrid` | `v31.sem_z_coarse.c23past` | off / off | DONE |
| `V36B_H5_R1_RESID_NO_SWA` | `hybrid` | `v31.sem_resid_coarse_l025.c23past` | off / off | DONE |
| `V36B_H5_R1_SEM_Z_READONLY` | `read_path_only` | `v31.sem_z_coarse.c23past` | off / off | DONE |
| `V36B_H5_R1_RESID_READONLY` | `read_path_only` | `v31.sem_resid_coarse_l025.c23past` | off / off | DONE |

Full metrics：

| Run | Full ATE | `[200,300)` ATE | `[400,600)` ATE | Delta vs C9 |
|---|---:|---:|---:|---:|
| `C9_REF` | `33.7629421029` | `76.1021355543` | `41.8963642126` | `0.0` |
| `V36B_C9_NO_SWA_BASE` | `38.4008982950` | `74.4364805502` | `51.0119935782` | `+4.6379561921` |
| `V36B_SEM_Z_NO_SWA` | `38.6185780372` | `76.7424498908` | `51.5305074135` | `+4.8556359343` |
| `V36B_RESID_NO_SWA` | `38.2740049300` | `74.8701546766` | `50.8510186878` | `+4.5110628270` |
| `V36B_SEM_Z_READONLY` | `38.6598083906` | `76.8223138233` | `51.3057963914` | `+4.8968662877` |
| `V36B_RESID_READONLY` | `38.5202961329` | `75.0664994878` | `51.0993316597` | `+4.7573540299` |

Additional interpretation：

```text
NoSWA baseline itself regresses badly vs C9:
    C9_NO_SWA_BASE delta vs C9 = +4.6379561921m
    [400,600) delta vs C9 = +9.1156293656m

Residual noSWA is slightly better than C9_NO_SWA_BASE:
    38.2740049300m vs 38.4008982950m
but it is still far worse than C9:
    delta vs C9 = +4.5110628270m

Read-only / noTTT semantic rows also regress:
    SEM_Z_READONLY delta vs C9 = +4.8968662877m
    RESID_READONLY delta vs C9 = +4.7573540299m
```

Decision：

```text
H5 path-isolation full online = fail.

NoSWA does not rescue C9 compatibility.
Read-only / noTTT does not rescue C9 compatibility.
Semantic read-only still regresses vs C9, so per v36B 10.6 the semantic
deployment line should stop.
```

---

## 9. Not Started / Not Claimed

Not launched:

```text
1. No selector.
2. No additional all-memory combo.
3. No learned trigger/router.
4. No further semantic deployment full-online row after path-isolation fail.
```

Not claimed:

```text
1. No H0/H1/H2/H3/H4 short rollout counts as deployable.
2. No beta525 local repair counts as deployable.
3. No reused full evidence is claimed as a newly launched row.
4. No path-isolation full row counts as Target-30 success.
5. No Target-30 result was produced.
```

---

## 10. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| H0A hook reachability | pass | inherited v36 hook evidence |
| H0B attention mass | representative repair pass | old artifacts had `0/198` sampled records with mass fields; new representative row lands sampled qk mass with removed-before `0.0549`, removed-after `0.0` |
| H0C action smoke | pass | action distinguishability and source effects landed |
| H1 frame/global h10 | fail | best local delta far below gate |
| H2 SWA h10 | pass | chunk10 local `[200,300)` improvement with boundary no-regression |
| H2 SWA h15 | fail | h15 local delta collapses |
| H2 washout attribution | done | proxy-only; h15 retains only `0.3210` H9 / `0.2471` C9 of h10 local segment gain |
| H3 TTT h10 | fail | all deltas tiny |
| H4 initial h10 | near miss / fail | beta 4.75 misses `-5m` gate |
| H4 beta525 h10/h15 | pass diagnostic | chunk10 local durability passes |
| H5 reused all-chunks full evidence | fail | SEM_Z ATE `34.5597m`; RESID ATE `34.3258m`; both worse than C9 |
| H5 path-isolation full rows | fail | best new row RESID_NO_SWA ATE `38.2740m`, delta vs C9 `+4.5111m`; read-only rows also regress |
| Target-30 | fail | no full ATE <= 30m |

---

## 11. Final Decision

v36B 的真实成功点：

```text
1. Corrected the overblocking issue from v36.
2. Ran independent H1/H2/H3/H4 path probes under one-process-per-GPU scheduling.
3. Audited H0B source-removal semantics against VGGT4D:
       source_removal_semantics_match_vggt4d = true
       old landed_attention_mass_available = false
   Then added default-off sampled qk mass instrumentation and verified one
   representative compact_kv row:
       mean_removed_before = 0.0549059110
       mean_removed_after = 0.0
       mean_retained_after = 1.0
   This prevents overclaiming old artifacts while proving the repair path works.
4. Found H2 SWA h10 local signal, but it washed out at h15.
5. Added H2 washout attribution instead of continuing a weak SWA sweep:
       evidence = proxy_only_no_tensor_state_snapshots
       C9 h15/h10 [200,300) durability = 0.2471427261
6. Confirmed TTT semantic static/negative aliases are weak in h10.
7. Recovered strong H4 semantic C23 local signal with the prior beta525 repair:
       C9 h15 [200,300) delta = -6.1250511839m
       C9 h15 [400,600) delta = +0.2516740832m
8. Completed the v36B 10.6 full path-isolation branch:
       noSWA base
       semantic-z noSWA
       residual noSWA
       semantic-z read-only/noTTT
       residual read-only/noTTT
```

v36B 的关键负结果：

```text
The only durable local signal is still semantic C23 beta525 at chunk10.
Its all-chunks C9 full-online version is worse than C9:
    C9 ATE = 33.7629421029m
    C9 + SEM_Z_COARSE_BETA525_ALL ATE = 34.5597307381m
    delta = +0.7967886352m

The conservative residual all-chunks repair is also worse than C9:
    C9 + SEM_RESID_COARSE_L025_ALL ATE = 34.3258261120m
    delta = +0.5628840091m
    [400,600) delta = +3.1115081606m

The newly launched path-isolation full rows are all worse than C9:
    C9_NO_SWA_BASE ATE = 38.4008982950m
    SEM_Z_NO_SWA ATE = 38.6185780372m
    RESID_NO_SWA ATE = 38.2740049300m
    SEM_Z_READONLY ATE = 38.6598083906m
    RESID_READONLY ATE = 38.5202961329m

Therefore the local diagnostic repair does not solve Target-30, and disabling
SWA/TTT/read-only paths does not rescue C9 compatibility.
```

Interpretation：

```text
v36B confirms that v36's H0 overblocking was too strict:
    useful short diagnostics can be run safely with H0A/H0C and without H0B.

It also confirms the H0B boundary precisely:
    source-removal semantics are consistent with VGGT4D-style K/V source
    compaction,
    old rows cannot be retroactively converted into attention-mass evidence,
    but the new representative row shows sampled compact_kv removed-source mass
    is nonzero before compaction and zero after compaction.

But the scientific conclusion is still bounded:
    semantic memory control has real local cue value in H4,
    yet all-chunks full online conflicts with C9 rather than improving it.

The path-isolation result closes the v36B 10.6 repair branch:
    noSWA worsens C9 strongly,
    semantic residual noSWA is only slightly better than the noSWA baseline but
    still much worse than C9,
    and semantic read-only/noTTT also regresses.
Thus semantic C23 should remain diagnostic rather than deployable in v36B.
```

Conclusion type：

```text
NoOverblocking execution success, local H4 diagnostic success, full-online
Target-30 failure.

Do not promote v36B as deployable online success.
```

Next required direction：

```text
Do not continue broad semantic memory sweeps.
If continuing this family, the missing piece is not another semantic label rule
or simple path toggle; it is a deployable, non-overfit activation/lifecycle
mechanism that preserves the chunk10 H4 local benefit without applying semantic
C23 all chunks.

If broader attention-mass evidence is required, rerun the specific H1/H2 rows
with the new default-off instrumentation enabled.

Target-30 mainline should return to:
    explicit online trajectory-state,
    scale-state / gauge-risk,
    C9-native lifecycle,
    merge/gauge-aware correction,
    or non-semantic TTT-native control.
```
