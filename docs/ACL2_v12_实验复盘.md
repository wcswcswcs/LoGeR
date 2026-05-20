# ACL2 v12 实验复盘：TTT Write WindowScale/MPC Target25

日期：2026-05-18 至 2026-05-19（Asia/Singapore）

本文件的 Phase 0-2A 由 `tools/v12_ttt_write_autopsy.py` 从已有落盘 artifact 生成；Phase 3-4 由本轮实际运行结果追加。没有把缺失数据补写成实验结果。

## 0. 固定边界

- Best online HMC / TTT-write baseline: `H9_REPEAT`, ATE `34.1258`, Rot `6.5414`, counts_as_ttt_write=`True`。
- Best diagnostic-only no-GT pose proxy: `NOGTPOSE_27`, ATE `22.4012`, counts_as_ttt_write=`False`, uses_offline_trajectory_rewrite=`True`。
- Best v11 projection oracle: `ORACLE_TTT_01`, ATE `34.8647`, uses_gt_runtime_action=`True`, deployable_success=`False`。

## 1. Phase 0 Registry

输出：

```text
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/v12_result_registry.csv
```

Phase 0 gate 结论：H9 是 deployable online baseline；NOGTPOSE_27 是 offline diagnostic；ORACLE_TTT_* 是 GT runtime oracle-only，不允许写成 deployable TTT success。

## 2. Phase 1 Projection Autopsy

| Metric | Value |
|---|---:|
| Spearman helpful vs scale improvement | `0.0128` |
| Spearman helpful vs segment ATE improvement | `-0.2781` |
| normal+inverse oracle worse than H9 | `True` |
| projection action coordinate failed | `True` |

输出：

```text
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/phase1_projection_autopsy/projection_chunk_autopsy.csv
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/phase1_projection_autopsy/projection_correlation_summary.csv
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/phase1_projection_autopsy/layer_branch_projection_summary.csv
```

Native-vs-candidate layer delta 对比没有被伪造：H9 没有匹配的 v11 per-layer projection trace，因此 `native_update_norm/update_cos_to_native/candidate_minus_native_norm` 在 layer summary 中保留为 NaN，并明确标注 unavailable。

## 3. Phase 2A Offline Selector Audit

| Check | Value |
|---|---:|
| candidate_count | `9` |
| chunk_selected_H9_or_better_ratio | `0.2500` |
| selected_oracle_with_worse_global_ATE_count | `10` |
| Spearman proxy score vs ATE | `0.2833` |
| best proxy run | `ORACLE_TTT_01` |
| gate_pass | `False` |

输出：

```text
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/phase2_selector_offline_audit/offline_selector_run_scores.csv
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/phase2_selector_offline_audit/offline_selector_chunk_choices.csv
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/phase2_selector_offline_audit/offline_selector_gate.json
```

## 4. Phase 3 Auxiliary TTT Full Runs

新增工程：

```text
loger/pipeline/ttt_write_controller.py:
    overlap_pseudo_replay_v / overlap_pseudo_replay_kv

run_pipeline_abc_v2.py:
    --ttt_write_replay_feature_gate_mode choices include overlap pseudo replay modes
```

说明：`AUX_TTT_03/04` 使用的是 frame-static feature-center proxy，不是完整的 plan 中 scale-risk-gated structure replay；这里按实际实现记录，不把 proxy 写成 scale-state 成功。

输出：

```text
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/phase3_aux_ttt_registry.csv
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/global_drift_dashboard_phase3_aux_ttt/
```

| Run | Mechanism | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `H9_REPEAT` | reference | `34.1258` | `6.5414` | `92.4053` | `6.189` | `74.410` | `54.651` | `44.354` |
| `AUX_TTT_01` | overlap pseudo replay `0.05` | `34.3417` | `6.2433` | `92.4080` | `4.115` | `74.911` | `55.016` | `44.673` |
| `AUX_TTT_02` | overlap pseudo replay `0.10` | `34.4391` | `6.2323` | `92.4086` | `3.884` | `75.115` | `55.147` | `44.667` |
| `AUX_TTT_03` | frame-static proxy `0.05` | `34.2658` | `6.1954` | `92.4078` | `4.325` | `75.044` | `55.088` | `44.275` |
| `AUX_TTT_04` | frame-static proxy `0.10` | `34.3782` | `6.1588` | `92.4075` | `3.649` | `75.085` | `55.131` | `44.546` |
| `AUX_TTT_05` | apply mismatch suppression | `34.5371` | `6.2751` | `92.4129` | `4.353` | `75.847` | `55.666` | `44.135` |
| `AUX_TTT_06` | overlap + suppression | `34.5031` | `6.2849` | `92.4127` | `4.380` | `75.858` | `55.670` | `44.057` |

Phase 3 gate:

```text
strong success: fail
weak success: fail
best AUX ATE = 34.2658 (AUX_TTT_03), still worse than H9 by 0.1400m
best AUX [200,300) = 74.911, worse than H9 74.410
best AUX [400,600) = 44.057, but ATE is 34.5031 and [200,300) worsens
```

结论：auxiliary replay / suppression 没有产生新的 deployable TTT write candidate，也没有打开 target-25 空间。

## 5. Phase 4 Dual-Lifetime TTT

新增工程：

```text
run_pipeline_abc_v2.py:
    --ttt_write_gradient_reversal_transient_apply_scale

tools/run_attention_cue_experiment.sh:
    TTT_WRITE_GRADIENT_REVERSAL_TRANSIENT_APPLY_SCALE

loger/pipeline/ttt_write_controller.py:
    dual_lifetime short delta apply scale

loger/pipeline/hybrid_memory_controller.py:
    pass apply scale into TTTWriteController
```

`V12_P4_SMOKE_DLTTT01_e180_SWKS3` 工程 smoke：

```text
END_FRAME=180
ATE / Rot = 3.8888 / 3.6299
debug confirms:
    ttt_gradient_reversal_transient_applied = True
    ttt_gradient_reversal_transient_apply_scale = 0.25
    ttt_transient_delta_mode_out = dual_lifetime
    ttt_dual_lifetime_long_old_override = True
```

Full 输出：

```text
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/phase4_dual_lifetime_ttt_registry.csv
results/kitti01_hmc_v2/acl2_v12_ttt_write_windowscale_mpc_target25/global_drift_dashboard_phase4_dual_lifetime_ttt/
```

| Run | Setting | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `H9_REPEAT` | reference | `34.1258` | `6.5414` | `92.4053` | `6.189` | `74.410` | `54.651` | `44.354` |
| `DLTTT_01` | `K=1`, `alpha=0.25` | `36.2058` | `6.3919` | `92.4444` | `3.886` | `78.115` | `57.231` | `46.874` |
| `DLTTT_02` | `K=2`, `alpha=0.25` | `36.1339` | `6.3946` | `92.4439` | `3.858` | `78.010` | `57.163` | `46.698` |
| `DLTTT_03` | `K=2`, `alpha=0.50` | `35.9173` | `6.4000` | `92.4402` | `3.805` | `78.167` | `57.252` | `45.933` |
| `DLTTT_04` | `K=3`, `alpha=0.25` | `36.1354` | `6.3870` | `92.4428` | `3.902` | `77.941` | `57.121` | `46.874` |
| `DLTTT_05` | `K=2`, body chunks `5-9` | `36.2217` | `6.3920` | `92.4444` | `3.889` | `78.144` | `57.275` | `46.890` |
| `DLTTT_06` | `K=2`, body + conservative exit | `36.1380` | `6.3951` | `92.4438` | `3.836` | `78.010` | `57.161` | `46.709` |

Phase 4 gate:

```text
strong success ATE <= 30.0: fail
stage success ATE <= 32.5: fail
stage success [200,300) <= 60 and [400,600) <= 44.354: fail
weak success ATE <= 33.8 and [200,300) improves >=3m: fail
best DLTTT ATE = 35.9173
best DLTTT [200,300) = 77.941, worse than H9 by 3.531m
best DLTTT [400,600) = 45.933, worse than H9 by 1.580m
```

结论：当前 dual-lifetime short overlay 会改善 FinalErr/Rot 一部分读数，但显著伤害 ATE、`[200,300)` 和 `[400,600)`；不能作为 target-25 主线。

## 6. Final Decision

v12 停止规则触发：

```text
Phase 2A selector gate = fail
Phase 3 auxiliary objective = fail
Phase 4 dual-lifetime = fail
AUX_TTT_01-06 + DLTTT_01-06 共 12 条 online TTT full run
    none beats H9 by >= 0.3m
    none beats H9 at all
```

因此按 v12 plan 11.3，`Phase 5 finite-difference / MPC sensitivity` 不再启动；当前已经满足 TTT write target-25 主线降级条件。

最终结论：

```text
No new deployable online TTT write candidate.
No online target-25.
Best deployable online TTT/HMC remains H9_REPEAT:
    ATE / Rot = 34.1258 / 6.5414

NOGTPOSE_27 remains diagnostic-only:
    ATE = 22.4012
    counts_as_ttt_write = false

TTT write should be kept as Rot / FinalErr / local regularizer.
target-25 mainline should move to online pose-state / trajectory-state module.
```

GPU boundary note: after the user constraint, all continued runs were launched only on GPU `0/1/2/3`. Earlier failed GPU `4/5` attempts are not counted as valid v12 full results.
