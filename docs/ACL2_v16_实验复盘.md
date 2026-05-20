# ACL2 v16 实验复盘：TTT Write CausalFork CandidateBank Target25

日期：2026-05-19（Asia/Singapore）  
计划文件：`docs/ACL2_v16_TTT_Write_CausalFork_CandidateBank_Target25_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v16_ttt_causalfork_candidatebank_target25/`

本轮原则：只记录实际落盘结果；不把 offline trajectory rewrite、GT oracle、sandbox smoke、partial rollout、或 failed parity run 写成 deployable TTT success。

---

## 0. 工程与配置复盘

新增 / 修改：

```text
run_pipeline_abc_v2.py:
    --save_merge_states
    --save_merge_state_chunks
    --save_merge_state_kinds
    --load_merge_state_at_chunk
    --load_merge_state_at_chunk_index
    --global_chunk_offset
    merge/gauge cursor save/load/hash
    raw_prediction_buffer_summary.jsonl
    loaded merge cursor 后使用 aligned local predictions 输出 trajectory
    sliced run 的 chunk-gated read/TTT controls 使用 global chunk id

tools/run_attention_cue_experiment.sh:
    SAVE_MERGE_STATES
    SAVE_MERGE_STATE_CHUNKS
    SAVE_MERGE_STATE_KINDS
    LOAD_MERGE_STATE_AT_CHUNK
    LOAD_MERGE_STATE_AT_CHUNK_INDEX
    GLOBAL_CHUNK_OFFSET

tools/ttt_short_rollout_sandbox.py:
    added raw_pose_mean_trans_diff_vs_full
    added timestamp_mapping_equal / window frame counts

tools/run_v16_candidate_rollout.sh:
    Phase 2 trusted short-rollout candidate launcher

tools/v16_candidate_value.py:
    numeric helper for shell candidate launcher

tools/v16_candidate_oracle_report.py:
    aggregate R6 candidate-bank rows from landed rollout artifacts
```

验证：

```text
python -m py_compile run_pipeline_abc_v2.py tools/ttt_short_rollout_sandbox.py tools/v15_config_diff_report.py tools/v16_candidate_value.py tools/v16_candidate_oracle_report.py
bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v16_candidate_rollout.sh
PASS
```

重要 blocker 与修复：

```text
R2 causal fork initially loaded correct HMC/merge input state,
but sliced rollout still used local chunk id 0/1/... for chunk-gated read beta / TTGR / EMA.

Fix:
    added --global_chunk_offset
    passed GLOBAL_CHUNK_OFFSET in launcher
    used effective global chunk id for chunk-gated controls and debug rows
    kept load_hmc/load_merge index as local index
```

---

## 1. Phase 0 Boundary

输出：

```text
registry_v16_phase0_boundary_R3.csv
registry_v16_phase0_boundary_R3.json
phase1_causalfork/state_snapshots/
phase1_causalfork/merge_state_snapshots/
```

无效尝试记录：

```text
第一次 v16 boundary attempt 漏掉 locked H9/C9 action env，H9/C9/WINGAM ATE 约 38m。
这些 run 标记为 config-mismatch invalid，不计入 gate。

WINGAM_R2 误继承 H9/C9 read_beta_frame_chunks / chunk16 gamma，标记为 config-mismatch invalid。
```

有效 boundary：

| Run | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[400,600)` | hmc rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `H9_P0_R2` | `34.1257769401` | `6.5414` | `92.4053` | `6.189399` | `74.409927` | `44.353638` | `38` |
| `C9_P0_R2` | `33.7629421029` | `6.5259` | `92.3871` | `5.666384` | `76.102136` | `41.896364` | `38` |
| `WINGAM_P0_R3` | `34.1902782732` | `6.5666` | `92.4202` | `6.195236` | `75.576021` | `42.280485` | `38` |

说明：

```text
WINGAM_P0_R3 的 Python pipeline 和 benchmark 已 DONE 且 artifact 完整。
该 shell session 因中途修改 launcher，在 DONE 后读到旧 bash 尾部并返回 EOF syntax error；
本轮按已落盘 01.txt / kitti_benchmark.log / hmc_state_hash.jsonl 计数。
当前 bash -n 已通过。
```

Snapshot footprint：

```text
H9/C9 Phase 1 input HMC + merge snapshots, including derived H9 chunks 6/9/12:
18 files
8.37 GB
<= 20 GB boundary batch limit
```

Phase 0 conclusion：

```text
Phase 0 gate = pass
H9/C9/WINGAM all match v15 locked references within 0.03m
input-only HMC + merge snapshots did not change valid boundary outputs
```

---

## 2. Phase 1 Trajectory-Level Causal Fork Parity

输出：

```text
phase1_causalfork/phase1_causalfork_gate_summary_R3.csv
phase1_causalfork/phase1_causalfork_gate_summary_R3.json
phase1_causalfork/native_rollouts/
phase1_causalfork/audit_R3_*_globalgate/
```

关键修复效果：

```text
v15 H9 chunk5 h1 raw trans diff = 11.9086m
v16 R3 H9 chunk5 h1 raw trans diff = 0.0m
```

R3 gate：

| Run | Horizon | ATE diff vs full | Raw pose max abs diff | Raw trans max diff | HMC hash mismatch | Gate |
|---|---:|---:|---:|---:|---:|---|
| `H9_chunk5_h1_R3_globalgate` | `1` | `0.0` | `0.0` | `0.0` | `0` | pass |
| `H9_chunk5_h3_R3_globalgate` | `3` | `0.0` | `0.0` | `0.0` | `0` | pass |
| `H9_chunk5_h5_R3_globalgate` | `5` | `0.0` | `0.0` | `0.0` | `0` | pass |
| `H9_chunk10_h3_R3_globalgate` | `3` | `0.0` | `0.0` | `0.0` | `0` | pass |
| `H9_chunk16_h3_R3_globalgate` | `3` | `0.0` | `0.0` | `0.0` | `0` | pass |
| `C9_chunk5_h3_R3_globalgate` | `3` | `0.0` | `0.0` | `0.0` | `0` | pass |
| `C9_chunk10_h3_R3_globalgate` | `3` | `0.0` | `0.0` | `0.0` | `0` | pass |
| `C9_chunk16_h3_R3_globalgate` | `3` | `0.0` | `0.0` | `0.0` | `0` | pass |

Phase 1 conclusion：

```text
Phase 1 gate = pass
HMC fast-weight state + merge/gauge state + global chunk id is sufficient to reproduce full-run trajectory suffix.
Candidate bank may start.
```

---

## 3. Phase 2 Candidate Commit Bank Oracle

输出：

```text
phase2_candidate_bank/candidate_manifest_R3.csv / .json
phase2_candidate_bank/oracle_table_R3.csv / .json
phase2_candidate_bank/oracle_table_R5_expanded.csv / .json
phase2_candidate_bank/oracle_summary_R5_expanded.csv
phase2_candidate_bank/candidate_manifest_R6_targeted.csv / .json
phase2_candidate_bank/oracle_table_R6_targeted.csv / .json
phase2_candidate_bank/oracle_summary_R6_targeted.csv / .json
phase2_candidate_bank/candidate_manifest_R7_gt_scale_projection.csv / .json
phase2_candidate_bank/oracle_table_R7_gt_scale_projection.csv / .json
phase2_candidate_bank/oracle_summary_R7_gt_scale_projection.csv / .json
phase2_candidate_bank/candidate_manifest_R10_online_scale_state_corrected.csv / .json
phase2_candidate_bank/oracle_table_R10_online_scale_state_corrected.csv / .json
phase2_candidate_bank/oracle_summary_R10_online_scale_state_corrected.csv / .json
phase2_candidate_bank/candidate_h3_ate_heatmap_R3.csv
phase2_candidate_bank/candidate_h3_ate_delta_heatmap_R3.csv
```

重要边界：

```text
Phase 2 是 trusted short-rollout GT oracle audit。
GT 只用于离线评分，不用于 runtime action。
所有 Phase 2 row 都是 diagnostic_only_short_rollout = true。
counts_as_ttt_write_success = false。
没有启动 full online selector validation。
```

第一批 candidate family：

```text
K1_H9
K4_TTGR_ZERO
K4_TTGR_DOUBLE
K7_EMA090_W0
K8_NATIVE_MIX075_W0
K5_CAP_W0_050
K10_DUAL_K2_A025
```

R3 结果：

```text
max h3 ATE improvement = 0.315463m
max [200,300) eval-intersection improvement = 0.413888m
H2 stage pass = false
H2 fail = true
```

按计划处理 blocker：

```text
没有启动 no-GT selector。
没有启动 full online validation。
转向扩展 action family，只跑 sandbox oracle。
```

扩展 candidate family：

```text
K9_OVERLAP_PSEUDO_KV
K11_ORTHO_SUPPRESS_W0
K12_OVERLAP_STATIC_TOPK
```

R5 expanded best by chunk：

| Chunk | Best candidate | h3 ATE | h3 ATE delta vs H9 | `[200,300)` delta vs H9 | `[400,600)` delta vs H9 |
|---:|---|---:|---:|---:|---:|
| `5` | `K4_TTGR_ZERO` | `1.172982` | `-0.027376` | `-0.079896` | `nan` |
| `6` | `K4_TTGR_ZERO` | `1.818655` | `-0.315463` | `-0.338885` | `nan` |
| `9` | `K11_ORTHO_SUPPRESS_W0` | `7.054944` | `-0.163526` | `-0.064826` | `nan` |
| `10` | `K11_ORTHO_SUPPRESS_W0` | `5.811254` | `-0.507247` | `-1.028527` | `nan` |
| `12` | `K11_ORTHO_SUPPRESS_W0` | `3.362973` | `-0.104322` | `nan` | `-0.087764` |
| `16` | `K8_NATIVE_MIX075_W0` | `2.479493` | `-0.011373` | `nan` | `-0.011373` |

R5 decision：

```text
max h3 ATE improvement = 0.507247m
max [200,300) eval-intersection improvement = 1.028527m
H2 strong pass = false
H2 stage pass = false
H2 hard fail = false by the <0.5m threshold, but still below selector/full-run entry gate
```

Interpretation：

```text
K11_ORTHO_SUPPRESS_W0 is the first weak positive family above 0.5m h3 local improvement.
However, improvement is still far below the v16 H2 stage gate:
    h3/full-equivalent improvement >= 1.0m
    or [200,300) improvement >= 5m without downstream regression.

Therefore the current candidate bank does not justify no-GT selector or full online validation.
```

R6 targeted blocker follow-up：

按 v16 计划的 blocker 处理路线，继续只在 sandbox oracle 中扩展 action family，没有启动 selector/full run。

新增 candidate family：

```text
K13_ORTHO_RHO025_W0
K14_TTGR_ZERO_ORTHO_W0
K15_COSCAP_W0_025_050
K16_COMMIT_CONFLICT_NATIVE025
K17_COMMIT_CONFLICT_NATIVE050_Q90
K18_OVERLAP_PSEUDO_V050
K19_ORTHO_SUPPRESS_W1
K20_ORTHO_SUPPRESS_W2
K21_ORTHO_SUPPRESS_ALL
```

R6 targeted rows：

| Candidate | Chunk | h3 ATE | h3 ATE delta vs H9 | `[200,300)` delta vs H9 | hmc rows |
|---|---:|---:|---:|---:|---:|
| `K13_ORTHO_RHO025_W0` | `6` | `9.415234` | `+7.281116` | `+0.556011` | `4` |
| `K14_TTGR_ZERO_ORTHO_W0` | `6` | `9.619511` | `+7.485393` | `+0.829822` | `4` |
| `K13_ORTHO_RHO025_W0` | `10` | `6.048254` | `-0.270247` | `-1.397592` | `4` |
| `K14_TTGR_ZERO_ORTHO_W0` | `10` | `6.144277` | `-0.174224` | `-1.246220` | `4` |
| `K15_COSCAP_W0_025_050` | `10` | `6.322177` | `+0.003676` | `-0.883729` | `4` |
| `K16_COMMIT_CONFLICT_NATIVE025` | `10` | `6.441271` | `+0.122770` | `-0.677076` | `4` |
| `K17_COMMIT_CONFLICT_NATIVE050_Q90` | `10` | `6.433938` | `+0.115437` | `-0.676889` | `4` |
| `K18_OVERLAP_PSEUDO_V050` | `10` | `6.365010` | `+0.046509` | `-0.716175` | `4` |
| `K19_ORTHO_SUPPRESS_W1` | `10` | `6.468193` | `+0.149693` | `-0.558555` | `4` |
| `K20_ORTHO_SUPPRESS_W2` | `10` | `6.445835` | `+0.127334` | `-0.569475` | `4` |
| `K21_ORTHO_SUPPRESS_ALL` | `10` | `5.883534` | `-0.434967` | `-1.765624` | `4` |

R6 combined decision：

```text
best h3 ATE delta remains:
    K11_ORTHO_SUPPRESS_W0 chunk10 = -0.507247m

best [200,300) eval-intersection delta becomes:
    K21_ORTHO_SUPPRESS_ALL chunk10 = -1.765624m

H2 strong pass = false
H2 stage pass = false
No no-GT selector/full online validation is allowed.
```

Interpretation：

```text
The plan-recommended follow-up did not unlock a stage-pass candidate.
Residual routing has a real but weak local effect on chunk10.
Commit-risk filtering and overlap pseudo replay do not beat H9 in h3 ATE.
The chunk6 routing variants are unstable and regress h3 ATE by >7m.
```

R7/R10 H5 follow-up：

R6 没有达到 H2 stage pass 后，按 v16 H5 路线继续做 scale-state diagnostic。重要边界：

```text
K22/K23/K24 use GT runtime scale projection.
They are oracle/diagnostic only and cannot count as deployable.

K25/K26/K27/K28 are explicit online trajectory-scale-state modules.
They do not use GT at runtime, but they bypass TTT fast-weight write.
They also cannot count as TTT write success.
```

R7 GT scale-projection TTT oracle：

| Candidate | Chunk | h3 ATE | h3 ATE delta vs H9 | `[200,300)` delta vs H9 | uses GT runtime |
|---|---:|---:|---:|---:|---|
| `K22_GT_SCALE_PROJ_BASE` | `6` | `8.911667` | `+6.777549` | `+0.012590` | `true` |
| `K22_GT_SCALE_PROJ_BASE` | `10` | `6.533392` | `+0.214891` | `-0.402687` | `true` |
| `K23_GT_SCALE_PROJ_STR2` | `10` | `6.533392` | `+0.214891` | `-0.402687` | `true` |
| `K24_GT_SCALE_PROJ_DOUBLE` | `10` | `6.504734` | `+0.186233` | `-0.460531` | `true` |

Interpretation：

```text
GT scale residual routed into current TTT tri-replay risk does not unlock H2.
It worsens h3 ATE on chunk6 and chunk10.
This weakens the hypothesis that current TTT fast-weight write interface can directly consume scale-state.
```

R8/R9 exposed a reporting audit issue：

```text
Old oracle table K1_H9 chunk6 baseline = 2.134118m.
Direct recomputation from matching H9 frames [174,293] gives ~8.900814m.
Therefore new online-scale-state deltas are reported from R10 corrected table,
where H9 baseline is recomputed from the same candidate frame intersection.
Old R5/R6 rows are left untouched and should be interpreted as their original audit convention.
```

R10 corrected online scale-state diagnostic：

| Candidate | Chunk | h3 ATE | corrected h3 ATE delta vs H9 | `[200,300)` delta vs H9 | uses GT runtime | counts as TTT success |
|---|---:|---:|---:|---:|---|---|
| `K25_ONLINE_SCALE_OVERLAP_STEP` | `6` | `8.911667` | `+0.010853` | `+0.012590` | `false` | `false` |
| `K26_ONLINE_SCALE_OVERLAP_TIGHT` | `6` | `8.911667` | `+0.010853` | `+0.012590` | `false` | `false` |
| `K27_ONLINE_SCALE_INV_STEP` | `6` | `8.911667` | `+0.010853` | `+0.012590` | `false` | `false` |
| `K28_ONLINE_SCALE_INV_TIGHT` | `6` | `8.911667` | `+0.010853` | `+0.012590` | `false` | `false` |
| `K25_ONLINE_SCALE_OVERLAP_STEP` | `10` | `6.533392` | `-0.131429` | `-0.402688` | `false` | `false` |
| `K26_ONLINE_SCALE_OVERLAP_TIGHT` | `10` | `6.533391` | `-0.131430` | `-0.402691` | `false` | `false` |
| `K27_ONLINE_SCALE_INV_STEP` | `10` | `6.533390` | `-0.131432` | `-0.402692` | `false` | `false` |
| `K28_ONLINE_SCALE_INV_TIGHT` | `10` | `6.533390` | `-0.131432` | `-0.402692` | `false` | `false` |

Interpretation：

```text
The first explicit online scale-state module has only weak local effect.
chunk10 improves by ~0.13m h3 and ~0.40m on [200,300), far below target-25 leverage.
chunk6 is effectively unchanged or slightly worse.
This module is not a deployable TTT write result.
```

---

## 4. Downstream Phase Decision

| Phase | Status | Reason |
|---|---|---|
| Phase 0 Boundary | pass | H9/C9/WINGAM locked references reproduced |
| Phase 1 Causal Fork | pass | raw pose / ATE suffix / HMC hash parity all exact for required H9/C9 windows |
| Phase 2 Candidate Bank Oracle | weak signal, no stage pass | R6 best h3 improvement `0.507m`; best `[200,300)` improvement `1.766m`, below gate |
| H5 Scale-State Diagnostic | weak signal, no target-25 path yet | GT TTT projection worsened; explicit online scale-state improved chunk10 only `0.131m` |
| Phase 3 No-GT Selector | not started | oracle stage gate not met |
| Phase 4 Full New Action Family Validation | not started | no sandbox oracle stage pass |
| Phase 5 Full Online Validation | not started | forbidden by Phase 2 result |

Boundary：

```text
No Phase 2 result counts as deployable TTT write success.
No GT-selected candidate is counted as online success.
No no-GT selector was evaluated.
No full online target-25 validation was launched.
No online target-25 result was produced in v16.
No scale-state diagnostic row counts as TTT write success.
```

---

## 5. Final Decision

v16 的真正成功点：

```text
Trajectory-level causal fork parity is fixed.
Phase 1 is now a usable trusted short-horizon causal bank.
This resolves the v15 blocker.
```

v16 的 candidate-bank conclusion：

```text
Current fast-weight action candidates show weak local causal signal.
Best observed:
    K11_ORTHO_SUPPRESS_W0 at chunk10
    h3 ATE delta vs H9 = -0.507247m
    [200,300) eval-intersection delta = -1.028527m

Best R6 segment-local effect:
    K21_ORTHO_SUPPRESS_ALL at chunk10
    [200,300) eval-intersection delta = -1.765624m

This is not enough for H2 stage pass and not enough to start a no-GT selector.
```

v16 H5 diagnostic conclusion：

```text
GT scale projection into TTT write did not produce useful upper bound.
The first explicit online overlap-step scale-state module produced only weak chunk10 local improvement.
Target-25 remains unreached.
The next viable direction is a stronger explicit trajectory-state module, not more scalar TTT write sweeps.
```

Next required direction：

```text
Do not run selector/full validation yet.
Continue Phase 2/4 only in trusted sandbox with stronger action families:
    true post-zeropower residual basis routing
    overlap-geometry auxiliary replay objective
    scale-state-conditioned TTT write
    or explicit online trajectory/scale-state module if TTT write-only remains below gate
```
