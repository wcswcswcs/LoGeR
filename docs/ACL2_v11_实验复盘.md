# ACL2 v11 实验复盘：No-Postprocess TTT Write Scale-Aware Action Space

日期：2026-05-18  
计划文件：`docs/ACL2_v11_NoPostprocess_TTT_Write_ScaleAware_ActionSpace_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/`

本轮原则：只记录实际落盘结果；后处理轨迹修正不计入 TTT write success；没有接通的 projection hook 不写成 projection 失败。

---

## 0. v10 / v9 继承基线

| Run | 类型 | ATE RMSE | Rot RMSE | FinalErr | `[200,300)` | `[400,600)` | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `H9_READBETA2_03` | online HMC / TTT write | `34.1258` | `6.5414` | `6.189` | `74.410` | `44.354` | v9 online best |
| `NOGTPOSE_27` | offline no-GT pose proxy | `22.4012` | n/a | `8.753` | `35.187` | `8.587` | target-25 diagnostic，不算 TTT write |

v11 的核心边界：

```text
Online HMC full run + TTT commit affects future state = 可计入 TTT write candidate
Offline trajectory rewrite / no-GT pose proxy / GT oracle = diagnostic 或 oracle，不计入 TTT write success
```

---

## 1. Phase 0：reference repeat 与边界确认

新增工具：

```text
tools/v11_metric_registry.py
tools/v11_projection_gate_audit.py
```

输出：

```text
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/phase0_repeats/
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/phase0_nogt_pose_proxy/
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/trajectory_diagnostics_phase0_repeats/
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/global_drift_dashboard_phase0_repeats/
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/v11_result_registry.csv
```

### 1.1 Online reference repeat

固定协议继承 v9 best / v8 references，均为 `run_pipeline_abc_v2.py` online full run，`hmc_commit_mode=probe_ttt_write`。

| Run | GPU | Start | Done | Walltime |
|---|---:|---|---|---:|
| `V11_P0_REPEAT_H9_READBETA2_03_body485_exit425_c16_425_SWKS3` | `0` | `2026-05-18 15:17:23` | `15:53:32` | `36.2 min` |
| `V11_P0_REPEAT_C16ROLE_01_SWKS3` | `1` | `2026-05-18 15:17:23` | `15:51:52` | `34.5 min` |
| `V11_P0_REPEAT_WINGAM_03_SWKS3` | `2` | `2026-05-18 15:17:23` | `15:48:19` | `30.9 min` |

Global metrics 与 trajectory diagnostics：

| Run | ATE RMSE | Rot RMSE | RPE t | RPE r | FinalErr | Yaw RMSE | Sim3 scale | `[200,300)` | `[200,400)` | `[400,600)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `H9_REPEAT` | `34.1258` | `6.5414` | `92.4053` | `0.0082` | `6.189` | `3.803` | `30.808906` | `74.410` | `54.651` | `44.354` |
| `C16ROLE_REPEAT` | `34.1583` | `6.5327` | `92.4059` | `0.0082` | `6.249` | `3.803` | `30.809795` | `74.521` | `54.740` | `44.369` |
| `WINGAM_REPEAT` | `34.1903` | `6.5666` | `92.4202` | `0.0083` | `6.195` | `3.774` | `30.994373` | `75.576` | `55.428` | `42.280` |

State / boundary flags from `v11_result_registry.csv`：

| Run | counts_as_ttt_write | hmc rows | commit mode ok | state changed count | no postprocess |
|---|---:|---:|---:|---:|---:|
| `H9_READBETA2_03_repeat` | `true` | `38` | `true` | `38` | `true` |
| `C16ROLE_01_repeat` | `true` | `38` | `true` | `38` | `true` |
| `WINGAM_03_repeat` | `true` | `38` | `true` | `38` | `true` |

Phase 0 online gate：

```text
H9 repeat ATE = 34.12577694
reference H9 ATE = 34.1258
delta < 0.0001m <= 0.03m
PASS
```

### 1.2 NOGTPOSE diagnostic reproduction

重跑：

```text
tools/v10_nogt_pose_proxy.py
source = V9_H9_READBETA2_03_body485_exit425_c16_425_SWKS3/01.txt
```

关键结果：

| Run | ATE RMSE | FinalErr | Yaw RMSE | Sim3 scale | `[200,300)` | `[200,400)` | `[400,600)` | registry 分类 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `NOGTPOSE_13_reset_global_clip25_pre600` | `27.0256` | `7.023` | `3.578` | `34.433818` | `42.728` | `31.233` | `9.528` | diagnostic only |
| `NOGTPOSE_21_reset_global_clip32_body600` | `25.4888` | `7.857` | `3.553` | `34.206502` | `37.717` | `28.421` | `8.998` | diagnostic only |
| `NOGTPOSE_27_reset_global_clip35_body600_t105` | `22.4012` | `8.753` | `3.552` | `33.031081` | `35.187` | `26.983` | `8.587` | diagnostic only |

Boundary flags：

```text
counts_as_ttt_write = false
diagnostic_only = true
output_from_online_hmc = false
no_postprocess_flag = false
```

Phase 0 diagnostic gate：

```text
NOGTPOSE_27 repeat ATE = 22.40124708
v10 recorded ATE = 22.4012
delta < 0.0001m <= 0.05m
diagnostic flags present in v11_result_registry.csv
PASS
```

---

## 2. Phase 1：projection instrumentation smoke / gate

有效 smoke：

| Run | GPU | Start | Done | 备注 |
|---|---:|---|---|---|
| `V11_P1_SMOKE_projection_trace_e160_R1_SWKS3` | `3` | `2026-05-18 15:25:06` | `15:29:30` | `END_FRAME=160` logging/instrumentation smoke |

说明：早先一次后台启动 `V11_P1_SMOKE_projection_trace_e160_SWKS3` 只有 start 标记、无有效输出，不计入实验。

审计输出：

```text
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/phase1_projection_gate_audit/
```

Required artifact coverage：

| Required artifact | Present | Count |
|---|---:|---:|
| `per_token_update_group.pt` | `false` | `0` |
| `per_layer_branch_update_matrix.pt` | `false` | `0` |
| `tri_replay_role_mass.jsonl` | `false` | `0` |
| `window_pose_residual_gt.json` | `false` | `0` |
| `window_drift_direction_gt.pt` | `false` | `0` |
| `update_to_drift_projection.csv` | `false` | `0` |
| `update_cos_to_window_drift.jsonl` | `false` | `0` |
| `projection_helpful_energy.jsonl` | `false` | `0` |
| `ttt_update_conflict_energy.jsonl` | `false` | `0` |

Debug/source hook audit：

| Check | Result |
|---|---:|
| required artifact coverage | `0 / 9 = 0.000` |
| `update_cos_to_window_drift` debug rows | `0` |
| `projection_helpful_energy` debug rows | `0` |
| source-level `window_drift_direction` hook | `false` |
| source-level `update_to_drift_projection` hook | `false` |
| source-level `per_token_update_group` hook | `false` |

Phase 1 gate：

```text
gate_pass = false
phase2_allowed = false
reason = required online projection instrumentation is incomplete
```

机制结论：

1. 这不是 “projection oracle 失败”；这是 **true projection instrumentation 没有接通**。
2. 当前代码没有落盘 v11 计划要求的 token / layer / branch update-to-drift projection artifacts。
3. 因此按 v11 gate，不能启动 `ORACLE_TTT_*` full run，也不能把 no-GT projection 近似矩阵写成有效实验。

---

## 3. 后续 Phase 决策

| Phase | 状态 | 原因 |
|---|---|---|
| Phase 2 `ORACLE_TTT_*` | 已完成 | R4 后 artifact coverage `9/9`，已运行 projection-routed oracle full runs，见第 6 节 |
| Phase 3 `SELECT_TTT_*` | 未启动 | Phase 2 oracle 未过弱 gate，不进入 no-GT TTT selector |
| Phase 4 `DLTTT_*` | 未启动 | Phase 2 没有出现 segment 正信号，不进入 dual-lifetime TTT |
| Phase 5 online pose-state module | 未实现 | v11 结论转向 pose-state / trajectory-state，但当前仅有 offline `v10_nogt_pose_proxy.py` 诊断工具 |
| Phase 6 cross-seq | 未启动 | 没有 online TTT candidate `ATE <= 32`，也没有 online pose-state candidate `ATE <= 25` |

---

## 4. 当前 v11 结论

1. v11 成功复现了在线 best 边界：`H9_REPEAT=34.1258 / 6.5414`，与 v9 best 严格一致。
2. v11 成功复现了 no-GT pose proxy 诊断边界：`NOGTPOSE_27=22.4012`，但 registry 明确标为 `counts_as_ttt_write=false`。
3. v11 没有产生新的 TTT write candidate，也没有达到 online target-25。
4. Phase 1 gate 失败是本轮的关键结论：true projection oracle 仍未接通，不能下 `TTT projection action 无效` 的机制结论。
5. 下一步如果继续 v11，必须先做工程接线，而不是跑更多 full run：

```text
必须新增在线落盘：
    per_token_update_group.pt
    per_layer_branch_update_matrix.pt
    tri_replay_role_mass.jsonl
    window_pose_residual_gt.json
    window_drift_direction_gt.pt
    update_to_drift_projection.csv
    update_cos_to_window_drift.jsonl
    projection_helpful_energy.jsonl
    ttt_update_conflict_energy.jsonl

这些文件覆盖率达到 100% 后，才允许启动 ORACLE_TTT_01-04。
```

当前 best 分类保持：

```text
Best online HMC / TTT-write candidate:
    H9_READBETA2_03_repeat
    ATE / Rot = 34.1258 / 6.5414

Best diagnostic-only no-GT pose proxy:
    NOGTPOSE_27_reset_global_clip35_body600_t105
    ATE / Yaw = 22.4012 / 3.5515
    counts_as_ttt_write = false
```

---

## 5. 继续执行记录：Phase 1 instrumentation R4

追加工程接线：

```text
run_pipeline_abc_v2.py:
    --v11_projection_trace_dir
    --v11_projection_gt_path
    logging-only v11 trace writer

tools/run_attention_cue_experiment.sh:
    V11_PROJECTION_TRACE_DIR
    V11_PROJECTION_GT_PATH
```

边界说明：这些 projection / GT 文件只作为 oracle/debug instrumentation 落盘；不进入 forward，不改写输出轨迹，不改变 committed HMC state。

### 5.1 R4 smoke

| Run | GPU | Start | Done | Scope |
|---|---:|---|---|---|
| `V11_P1_SMOKE_projection_trace_e160_R4_SWKS3` | `0` | `2026-05-18 19:20:53` | `19:24:59` | `END_FRAME=160` logging-only smoke |

Trace 目录：

```text
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/phase1_projection_smoke/
  V11_P1_SMOKE_projection_trace_e160_R4_SWKS3/v11_projection_trace/
```

落盘文件：

```text
per_token_update_group.pt
per_layer_branch_update_matrix.pt
tri_replay_role_mass.jsonl
window_pose_residual_gt.json
window_drift_direction_gt.pt
update_to_drift_projection.csv
update_cos_to_window_drift.jsonl
projection_helpful_energy.jsonl
ttt_update_conflict_energy.jsonl
v11_trace_summary.jsonl
```

行数概览：

| Artifact | Rows |
|---|---:|
| `v11_trace_summary.jsonl` | `6` |
| `tri_replay_role_mass.jsonl` | `108` |
| `ttt_update_conflict_energy.jsonl` | `6` |
| `update_cos_to_window_drift.jsonl` | `324` |
| `projection_helpful_energy.jsonl` | `324` |

Logging-only invariance check：

```text
cmp R2/01.txt R4/01.txt = identical
R4 smoke ATE on first 160 matched poses = 3.1580m
```

### 5.2 R4 gate audit

输出：

```text
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/phase1_projection_gate_audit_R4/
```

结果：

| Check | Result |
|---|---:|
| required artifact coverage | `9 / 9 = 1.000` |
| `update_cos_to_window_drift` debug rows | `6` |
| `projection_helpful_energy` debug rows | `6` |
| `projection_harmful_energy` debug rows | `6` |
| `projection_role_mass` debug rows | `6` |
| `ttt_update_conflict_energy` debug rows | `6` |
| source-level `window_drift_direction` hook | `true` |
| source-level `update_to_drift_projection` hook | `true` |
| `gate_pass` | `true` |
| `phase2_allowed` | `true` |

Phase 1 更新结论：

```text
initial R1/R2 gate: failed, artifact coverage 0/9
continued R4 gate: pass, artifact coverage 9/9
```

但 R4 仍然不是 TTT write candidate：

```text
END_FRAME=160 only
logging-only instrumentation smoke
uses GT only for oracle/debug artifact generation
projection_method = diagnostic_delta_mean_x_gt_window_scale_direction
no projection-routed write action has been run as a full online candidate
```

当前允许进入下一步，但下一步必须是 **projection-routed TTT write action** 的 engineering/full-run，而不是把 R4 logging smoke 记为 `ORACLE_TTT_*` 成功。

---

## 6. Phase 2：projection-routed TTT write oracle full runs

新增工程接线：

```text
loger/pipeline/ttt_write_controller.py:
    gradient_reversal_risk_source = v11_gt_scale_projection
    v11 projection-routed token risk:
        raw_w0_token_outer_mean x GT step-scale residual

run_pipeline_abc_v2.py:
    --v11_projection_action_mode
    --v11_projection_action_chunks
    --v11_projection_action_strength
    --v11_projection_action_deadband

tools/run_attention_cue_experiment.sh:
    V11_PROJECTION_ACTION_MODE
    V11_PROJECTION_ACTION_CHUNKS
    V11_PROJECTION_ACTION_STRENGTH
    V11_PROJECTION_ACTION_DEADBAND
```

分类边界：

```text
这些 run 是 online full run，输出轨迹没有后处理；
但 runtime action 使用 GT scale residual 定义 projection direction；
因此只能归类为 oracle-only TTT write upper-bound，不算 deployable TTT write success。
```

### 6.1 Action smoke

| Run | Scope | ATE / Rot | 关键审计 |
|---|---|---:|---|
| `V11_P2_SMOKE_gt_scale_projection_e180_c5_R2_SWKS3` | `END_FRAME=180`, chunk 5 action | `4.0041 / 3.7728` | chunk 5 `v11_projection_risk_applied=true`, `conflict_rows=18`, `role_rows=18` |

chunk 5 action debug 摘要：

```text
risk_source = v11_gt_scale_projection
scale_log = 3.2266
tri_replay active branches = [0]
pos / neutral / neg mass = 0.35 / 0.529985 / 0.120015
```

Smoke 结论：

```text
projection action 已进入真实 probe_ttt_write commit；
不是 logging-only；
trace writer 已能记录真实 probe commit 的逐层 projection risk。
```

### 6.2 Full oracle results

结果登记：

```text
results/kitti01_hmc_v2/acl2_v11_no_postprocess_ttt_scaleaware/
    phase2_oracle_ttt_registry.csv
    global_drift_dashboard_phase2_oracle_ttt/
```

| Run | Runtime GT action | Active chunks | ATE | Rot | RPE_t | FinalErr | `[200,300)` | `[200,400)` | `[400,600)` | Gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `H9_REPEAT` | no | - | `34.1258` | `6.5414` | `92.4053` | `6.189` | `74.410` | `54.651` | `44.354` | reference |
| `ORACLE_TTT_01` body projection | yes | `5-12` | `34.8647` | `6.6735` | `92.4324` | `5.788` | `75.772` | `55.349` | `44.513` | fail |
| `ORACLE_TTT_02` body+handoff | yes | `5-12,16` | `34.8690` | `6.6004` | `92.4154` | `5.992` | `74.560` | `54.709` | `46.826` | fail |
| `ORACLE_TTT_03` scale c5-16 | yes | `5-16` | `34.9331` | `6.5492` | `92.4269` | `6.136` | `75.622` | `55.294` | `45.508` | fail |
| `ORACLE_TTT_04` top-layer | yes | `5-12` | `36.3971` | `6.6635` | `92.4450` | `5.669` | `77.552` | `56.906` | `47.839` | fail |
| `ORACLE_TTT_05_INV` inverse body | yes | `5-12` | `35.5911` | `6.6745` | `92.4360` | `6.973` | `77.572` | `56.807` | `45.436` | fail |
| `ORACLE_TTT_06_INV` inverse c5-16 | yes | `5-16` | `35.7524` | `6.6115` | `92.4303` | `7.909` | `77.324` | `56.723` | `46.821` | fail |

### 6.3 Gate decision

Phase 2 强 gate：

```text
ATE <= 30.0
and [200,300) <= 55
and [400,600) <= 44.354
```

Phase 2 弱 gate：

```text
ATE <= 32.0
or [200,300) <= 60 and ATE <= 34.0
```

实际结果：

```text
best oracle ATE = 34.8647
best oracle [200,300) = 74.560
best oracle [400,600) = 44.513
inverse controls also failed: 35.5911 / 35.7524
```

结论：

```text
Phase 2 gate = fail
TTT projection oracle did run, and it was not a no-op.
Both normal and inverse projection routing failed to beat H9_REPEAT.
Therefore v11 不能再声称 “projection 没接通”；现在的结论是：
    this v11 scale-projection TTT write action space lacks target-25 upper bound.
```

### 6.4 v11 final conclusion

v11 最终状态：

```text
Best online HMC / TTT-write candidate:
    H9_READBETA2_03_repeat
    ATE / Rot = 34.1258 / 6.5414

Best v11 projection-routed oracle-only run:
    ORACLE_TTT_01
    ATE / Rot = 34.8647 / 6.6735
    counts_as_ttt_write_success = false
    reason = uses GT runtime projection direction

Best diagnostic-only no-GT pose proxy:
    NOGTPOSE_27_reset_global_clip35_body600_t105
    ATE / Yaw = 22.4012 / 3.5515
    counts_as_ttt_write = false
```

停止规则触发：

```text
true TTT projection oracle ATE > 33.5
=> TTT write-side 降级为 auxiliary regularizer
=> target-25 主线应转向 online pose-state / trajectory-state module
```

这不是 R4 时的 “没接通所以不能判断”。本轮已经完成接线、smoke、full oracle 和 inverse 控制；v11 的有效结论是：

```text
No new deployable TTT write candidate.
No online target-25.
TTT projection-routed write upper bound, in this implementation, is insufficient.
```
