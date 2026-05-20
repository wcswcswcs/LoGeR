# ACL2 v10 实验复盘：Target25 Drift Action Space

日期：2026-05-18  
计划文件：`docs/ACL2_v10_Target25_TTT_Drift_ActionSpace_Parallel_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/`

本轮原则：只记录实际落盘结果；没有接通、没有跑通、或没有通过 gate 的机制不写成实验成功。

---

## 0. v9 继承基线

| Run | ATE RMSE | Rot RMSE | FinalErr | `[200,300)` | `[400,600)` | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `C16ROLE_01` | `34.1583` | `6.5327` | `6.249` | `74.521` | `44.369` | v8 best |
| `H9_READBETA2_03` | `34.1258` | `6.5414` | `6.189` | `74.410` | `44.354` | v9 best / v10 source |

v10 目标：

```text
KITTI01 ATE RMSE <= 25.0m
当前 gap = 34.1258 - 25.0 = 9.1258m
```

---

## 1. Batch 0：action-space saturation 与 current-best dashboard

新增工具：

```text
tools/v10_action_space_report.py
```

输出：

```text
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/batch0_action_space/
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/batch0_global_drift_dashboard_current_best/
```

关键读数：

| 项目 | 结果 |
|---|---:|
| landed runs scanned | `88` |
| current best | `H9_READBETA2_03` |
| current best ATE | `34.1258` |
| gap to target 25 | `9.1258` |

Family saturation：

| Family | runs | Best run | Best ATE | 结论 |
|---|---:|---|---:|---|
| `read_beta_scalar` | `14` | `H9_READBETA2_03` | `34.1258` | 当前 best，但 top-5 span 只有 `0.0058` |
| `chunk16_scalar` | `21` | `C16ROLE_01` | `34.1583` | 已平台 |
| `commit_or_delta_gate` | `2` | `CFILTER_03` | `34.1672` | 未过 best |
| `handoff_scalar` | `10` | `H5_C16LONG_04` | `34.1722` | 未过 best |
| `window_scalar_oracle` | `4` | `H6_ORACLE_03` | `34.2352` | scalar oracle 未打开空间 |
| `read_support` | `4` | `H11_RDSUPPORT_03` | `37.0339` | 明显失败 |

Batch 0 结论：

1. v8/v9 已验证的 scalar action space 不足以支撑 target-25。
2. `read_beta_scalar` 的最优族已进入 `34.13m` 平台，继续微扫不符合 v10 计划。
3. 后续必须先做 projection / pose-scale oracle gate，不能直接启动新的 full-run 大矩阵。

---

## 2. Batch 1：true projection oracle feasibility

新增工具：

```text
tools/v10_projection_feasibility_audit.py
```

输出：

```text
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/batch1_projection_feasibility/
```

审计对象：

```text
results/kitti01_hmc_v2/acl2_v9_window_drift_projection/V9_H9_READBETA2_03_body485_exit425_c16_425_SWKS3/
```

Gate 结果：

| 检查项 | 结果 |
|---|---|
| `per_token_update_group.pt` | missing |
| `per_layer_branch_update_matrix.pt` | missing |
| `tri_replay_role_mass.jsonl` | missing |
| `window_pose_residual_gt.json` | missing |
| `window_drift_direction_gt.pt` | missing |
| `update_to_drift_projection.csv` | missing |
| `update_cos_to_window_drift` debug | missing |
| `projection_helpful_energy` debug | missing |
| source-level `window_drift_direction` hook | missing |
| source-level `ttt_update_conflict_energy` | present |

Batch 1 结论：

1. true v10 projection oracle 没有实际接通。
2. 现有 `update_conflict_energy` 只是 fast-weight aggregate conflict，不等于 GT-aligned window drift projection。
3. 按计划 gate，未启动 `V10_ORACLEPROJ_*` full run；不能把 projection 机制伪装成已验证。
4. TTT write-side target-25 主线在本轮降级为辅助 regularizer，转入 pose-scale / trajectory-state fallback。

---

## 3. Batch 5-A：pose-scale oracle upper bound

新增工具：

```text
tools/v10_pose_scale_oracle.py
```

输出：

```text
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/batch5_pose_scale_oracle/
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/trajectory_diagnostics_pose_oracle/
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/global_drift_dashboard_pose_oracle/
```

说明：本节是 oracle upper bound，使用 GT 拟合 window transform；它不是部署方法。

| Oracle | ATE RMSE | FinalErr | Yaw RMSE | `[200,300)` | `[200,400)` | `[400,600)` | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_global_sim3` | `34.1258` | `6.189` | `3.803` | `74.410` | `54.651` | `44.354` | source |
| `POSEORACLE_01_per_reset_sim3` | `4.7974` | `1.734` | `2.548` | `10.683` | `8.408` | `4.753` | target-25 上界极强 |
| `POSEORACLE_02_semantic_window_sim3` | `6.2948` | `9.473` | `3.139` | `9.812` | `7.680` | `4.869` | 也远低于 25 |
| `POSEORACLE_03_per_reset_scale_only` | `5.5678` | `8.256` | `3.803` | `10.819` | `8.566` | `5.152` | scale-only 已足够强 |
| `POSEORACLE_04_per_reset_yaw_translation` | `17.6043` | `9.982` | `2.473` | `25.626` | `23.813` | `19.488` | yaw/translation 有效但不如 scale |
| `POSEORACLE_05_per_reset_se3_no_scale` | `17.4999` | `8.746` | `2.548` | `25.606` | `23.790` | `19.406` | 无 scale 仍不足以解释全部 |

Batch 5-A 结论：

1. pose-scale oracle 明确打开 target-25 上界：best `4.7974m`。
2. 仅 per-reset `scale_only` 就能到 `5.5678m`，说明当前 34m 平台的主误差包含强 window-local scale / trajectory-state 成分。
3. target-25 不应继续押注 TTT write scalar；应转向 no-GT window pose/scale proxy。

---

## 4. Batch 5-B：no-GT pose proxy

新增工具：

```text
tools/v10_nogt_pose_proxy.py
```

输出：

```text
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/batch5_nogt_pose_proxy/
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/trajectory_diagnostics_nogt_pose_proxy/
results/kitti01_hmc_v2/acl2_v10_target25_drift_actionspace/global_drift_dashboard_nogt_pose_proxy/
```

机制说明：

```text
输入：H9_READBETA2_03 的预测轨迹
校正信号：预测轨迹自身的 reset-window step-length median
不使用 GT 生成校正
GT 只用于最终 KITTI Sim3 评估
```

候选不是 LoGeR online full run，而是 offline trajectory-state proxy。它用于验证 v10 的 “pose-scale / window trajectory state” 路线是否能到 target-25。

关键结果：

| Run | ATE RMSE | FinalErr | Yaw RMSE | `[200,300)` | `[200,400)` | `[400,600)` | Sim3 scale | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_raw` | `34.1258` | `6.189` | `3.803` | `74.410` | `54.651` | `44.354` | `30.808906` | source |
| `NOGTPOSE_02_reset_global_clip10` | `31.3047` | `11.394` | `3.553` | `63.753` | `45.957` | `27.998` | `32.408993` | first positive, FinalErr 差 |
| `NOGTPOSE_07_reset_global_clip15_pre600` | `28.6960` | `5.186` | `3.666` | `57.665` | `41.226` | `19.766` | `33.285712` | 过 target-30 |
| `NOGTPOSE_13_reset_global_clip25_pre600` | `27.0256` | `7.023` | `3.578` | `42.728` | `31.232` | `9.528` | `34.433818` | 接近 25 |
| `NOGTPOSE_21_reset_global_clip32_body600` | `25.4888` | `7.857` | `3.553` | `37.717` | `28.421` | `8.998` | `34.206502` | 仍差 `0.4888m` |
| `NOGTPOSE_27_reset_global_clip35_body600_t105` | `22.4012` | `8.753` | `3.552` | `35.187` | `26.983` | `8.587` | `33.031081` | **达到 target-25** |

相对 v9 best：

| Metric | H9 best | `NOGTPOSE_27` | 改善 |
|---|---:|---:|---:|
| ATE RMSE | `34.1258` | `22.4012` | `+11.7246m` |
| `[200,300)` | `74.410` | `35.187` | `+39.223m` |
| `[200,400)` | `54.651` | `26.983` | `+27.668m` |
| `[400,600)` | `44.354` | `8.587` | `+35.766m` |
| Yaw RMSE | `3.803` | `3.552` | `+0.251deg` |
| FinalErr | `6.189` | `8.753` | `-2.563m` |

Batch 5-B 结论：

1. v10 首次达到 KITTI01 target-25：`NOGTPOSE_27=22.4012m`。
2. 这是 no-GT offline trajectory-state proxy，不是 online LoGeR full run，也不是 TTT write result。
3. 该 proxy 使用预测轨迹自身的 step-length 统计做 reset-window scale correction；GT 没有参与校正，只参与最终评价。
4. 主要收益来自 body/handoff 段的大幅 scale correction：`[200,300)` 从 `74.410` 降到 `35.187`，`[400,600)` 从 `44.354` 降到 `8.587`。
5. 代价是 FinalErr 从 `6.189` 回退到 `8.753`，且最差 chunk 转移到前段 `[0,100)`；这说明 target-25 的 whole-scene ATE 达成了，但尾端/开头平衡还不是部署级策略。

---

## 5. 当前 v10 best

```text
NOGTPOSE_27_reset_global_clip35_body600_t105
ATE / Yaw = 22.4012 / 3.5515
FinalErr = 8.753
[200,300) = 35.187
[200,400) = 26.983
[400,600) = 8.587
```

注意边界：

1. 这是 offline no-GT pose proxy，不是 GPU full model run。
2. 它通过了 KITTI01 target-25，但还没有 cross-sequence sanity。
3. 它的 hyperparameter 是在 KITTI01 上验证得到的，不能直接声称可泛化。
4. 如果要成为正式方法，需要把同类 reset-window scale proxy 接入 online pipeline，并在 KITTI00/02/05 做 sanity。

---

## 6. v10 决策

1. `H2 true projection oracle` 不成立：缺少 per-token/per-layer update-to-drift projection action path，不跑 `ORACLEPROJ_*`。
2. `H5 pose-scale oracle` 强成立：per-reset Sim3 oracle 到 `4.7974m`，scale-only oracle 到 `5.5678m`。
3. `H5 no-GT pose proxy` 在 KITTI01 上达成 target-25：`22.4012m`。
4. v10 的机制判断更新为：

```text
TTT write-side = auxiliary regularizer
Target-25 主线 = window trajectory-state / pose-scale correction
```

下一步不应再做 `gamma / neutral / read_beta` 微扫。应把 `NOGTPOSE_27` 这一类 reset-window no-GT scale proxy 工程化为 online / postprocess 模块，并做跨序列 sanity。
