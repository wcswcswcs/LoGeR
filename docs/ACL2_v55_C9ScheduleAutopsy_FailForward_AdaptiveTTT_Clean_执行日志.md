# ACL2 v55 C9ScheduleAutopsy FailForward AdaptiveTTT Clean 执行日志

日期: 2026-06-09  
计划文档: `docs/ACL2_v55_C9ScheduleAutopsy_FailForward_AdaptiveTTT_CleanPlan.md`  
结果复盘: `docs/ACL2_v55_C9ScheduleAutopsy_FailForward_AdaptiveTTT_Clean_实验结果复盘.md`  
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`  
结果根目录: `results/kitti01_hmc_v2/acl2_v55_c9schedule_autopsy_failforward_adaptivettt_clean`

## 执行边界

- 本轮只记录落盘 artifact 和可解析日志，不补造任何缺失指标。
- clean adaptive candidate 保持 no absolute chunk-id policy、no manual tri-replay percentage、no SWA mainline、no semantic runtime action。
- GPU 使用:
  - GPU0: Phase 0 S0 M1 full diagnostic。
  - GPU1/GPU2: E1/E2 96F smoke 并行；E1/E2 704F screen 并行。
  - GPU3/GPU4: E1/E2 full 并行。
  - GPU5 未启用，因为 v55 plan 限制 full 主候选最多 2 条，且本轮没有满足 emergency repair 条件。

## 代码与工具修改

本轮为执行 v55 plan 做了以下可审计修改:

| 文件 | 修改内容 |
|---|---|
| `tools/v55_experiment_report.py` | 新增 v55 report/autopsy 汇总工具；从 landed artifacts 生成 Phase 0、Phase 1、smoke、704F、full、runtime/audit/timeline/final report。修复 `probe_ttt_write_seconds_mean` 缺失处理: 不把缺失当成真实超时，也不伪造数值，新增 `probe_ttt_write_seconds_missing` 与 `v55_runtime_gate_allow_probe_missing`。 |
| `tools/run_v55_tail_state_candidate.sh` | 新增 E1/E2 runner，支持 `E1_96F`、`E2_96F`、`E1_704F`、`E2_704F`、`E1_FULL`、`E2_FULL`。 |
| `loger/pipeline/ttt_write_controller.py` | 新增 `adaptive_writer_tail_state_continuity_guard` 与 `adaptive_writer_tail_state_continuity_guard_selective_commit` action path；加入 causal tail-state energy EMA、candidate/native cosine、overshoot、static-anchor、reset-age 等 debug 字段；新增 selective commit mode 的在线触发逻辑。 |
| `run_pipeline_abc_v2.py` | 将 `tail_state_selective_commit`、`selective_commit_ema`、`tail_state_continuity_selective_commit` 加入 CLI commit filter choices。 |

验证命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/pipeline/ttt_write_controller.py \
  run_pipeline_abc_v2.py \
  tools/v55_experiment_report.py

bash -n tools/run_v55_tail_state_candidate.sh
```

结果: `py_compile` 与 `bash -n` 均通过。

## Phase 0: S0 M1 full diagnostic

命令:

```bash
V54_RESULT_ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v55_c9schedule_autopsy_failforward_adaptivettt_clean \
V54_PHASE=phase0_salvage_full \
V54_END_FRAME=10000 \
tools/run_v54_state_conditioned_candidate.sh 0 M1_FULL S0_M1_FULL_DIAGNOSTIC
```

wall time summary:

| run | GPU | start | end | wall seconds |
|---|---:|---|---|---:|
| `S0_M1_FULL_DIAGNOSTIC` | 0 | `2026-06-09T15:53:17+08:00` | `2026-06-09T16:18:44+08:00` | 1527 |

主要输出:

- `phase0_salvage_full/rollouts/S0_M1_FULL_DIAGNOSTIC/01.txt`
- `phase0_salvage_full/rollouts/S0_M1_FULL_DIAGNOSTIC/01.log`
- `phase0_salvage_full/rollouts/S0_M1_FULL_DIAGNOSTIC/wall_time_summary.json`
- `report_final/v55_phase0_salvage_report.md`

结论: S0 M1 full ATE 为 `36.18720796800255`，比 H35 full 差 `+0.4463110098882055m`，超过 v55 Phase 0 的 `H35 full +0.30m` borderline 条件，因此 report 判定为 `m1_invalid_no_extension`。

## Phase 1: C9-H35-M1 autopsy 与 failure classification

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v55_experiment_report.py
```

主要输出:

- `report_final/phase1_trace_autopsy/c9_h35_m1_chunk_gap_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_segment_gap_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_layer_branch_delta_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_role_mass_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_commit_behavior_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_state_variable_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_autopsy_report.md`
- `report_final/v55_failure_type_summary.json`
- `report_final/v55_candidate_design_decision.md`

classification:

| item | value |
|---|---|
| failure type | `TYPE_E_SEG2_STATE_GAP` |
| largest segment gap | `seg1_384_700` |
| reason | `Selected by largest normalized landed evidence score; segment largest was not seg2.` |
| TYPE_A role mass score | `0.42146643508247467` |
| TYPE_B gamma energy score | `0.3817437927152741` |
| TYPE_C commit score | `0.0` |
| TYPE_D layer/branch score | `0.10125088216521025` |
| TYPE_E seg2 state score | `5.74354621121932` |

按 Phase 2 routing，选择两个候选:

| candidate | role mode | commit mode |
|---|---|---|
| E1 TailStateContinuityGuard | `adaptive_writer_tail_state_continuity_guard` | `none` |
| E2 TailStateContinuityGuard SelectiveCommit | `adaptive_writer_tail_state_continuity_guard_selective_commit` | `tail_state_selective_commit` |

## Phase 3: 96F smoke 并行运行

命令:

```bash
tools/run_v55_tail_state_candidate.sh 1 E1_96F V55_PHASE3_E1_TAIL_CONTINUITY_96F
tools/run_v55_tail_state_candidate.sh 2 E2_96F V55_PHASE3_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_96F
```

wall time summary:

| run | GPU | start | end | wall seconds |
|---|---:|---|---|---:|
| `V55_PHASE3_E1_TAIL_CONTINUITY_96F` | 1 | `2026-06-09T16:16:01+08:00` | `2026-06-09T16:18:35+08:00` | 154 |
| `V55_PHASE3_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_96F` | 2 | `2026-06-09T16:16:01+08:00` | `2026-06-09T16:18:42+08:00` | 161 |

smoke registry:

| run | frames | ATE | wall min | projected full min | chunk mean | no chunk | manual % | role collapse | decision |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| `V55_PHASE3_E1_TAIL_CONTINUITY_96F` | 96 | 1.328115151904405 | 2.566666666666667 | 24.383333333333333 | 27.135 | True | True | 0.0 | `promote_full` |
| `V55_PHASE3_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_96F` | 96 | 1.3295252255286152 | 2.683333333333333 | 25.491666666666667 | 27.8325 | True | True | 0.0 | `promote_full` |

## Phase 3: 704F screen 并行运行

命令:

```bash
tools/run_v55_tail_state_candidate.sh 1 E1_704F V55_PHASE3_E1_TAIL_CONTINUITY_704F
tools/run_v55_tail_state_candidate.sh 2 E2_704F V55_PHASE3_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_704F
```

wall time summary:

| run | GPU | start | end | wall seconds |
|---|---:|---|---|---:|
| `V55_PHASE3_E1_TAIL_CONTINUITY_704F` | 1 | `2026-06-09T16:21:33+08:00` | `2026-06-09T16:37:45+08:00` | 972 |
| `V55_PHASE3_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_704F` | 2 | `2026-06-09T16:21:33+08:00` | `2026-06-09T16:39:30+08:00` | 1077 |

704F registry:

| run | frames | ATE | Rot | FinalErr | projected full min | chunk mean | adaptive gamma mean | commit activation | commit scale | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `V55_PHASE3_E1_TAIL_CONTINUITY_704F` | 704 | 40.06262748588316 | 2.882370016817129 | 31.55908408675562 | 24.624000000000002 | 30.687600000000003 | 0.004170456242799345 | NA | NA | `borderline_diagnostic_full` |
| `V55_PHASE3_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_704F` | 704 | 40.08919219968784 | 2.876335243188599 | 31.559640045311532 | 27.284 | 33.6036 | 0.004163555778666503 | 0.03111111111111111 | 0.9852746951875663 | `borderline_diagnostic_full` |

说明: 两条 704F 都满足 v55 的 borderline diagnostic full 条件，因此按计划进入 full。E1 704F 完成后先启动 E1 full；E2 704F 完成后启动 E2 full。

## Phase 4: full KITTI01 并行运行

命令:

```bash
tools/run_v55_tail_state_candidate.sh 3 E1_FULL V55_PHASE4_E1_TAIL_CONTINUITY_FULL
tools/run_v55_tail_state_candidate.sh 4 E2_FULL V55_PHASE4_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_FULL
```

wall time summary:

| run | GPU | start | end | wall seconds |
|---|---:|---|---|---:|
| `V55_PHASE4_E1_TAIL_CONTINUITY_FULL` | 3 | `2026-06-09T16:38:58+08:00` | `2026-06-09T17:04:14+08:00` | 1516 |
| `V55_PHASE4_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_FULL` | 4 | `2026-06-09T16:41:14+08:00` | `2026-06-09T17:08:15+08:00` | 1621 |

full registry:

| run | frames | ATE | delta vs C9 | delta vs H35 full | Rot | FinalErr | wall min | chunk mean | progress pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `S0_M1_FULL_DIAGNOSTIC` | 1101 | 36.18720796800255 | 2.4242658650836972 | 0.4463110098882055 | 5.855805081166548 | 11.701876341256094 | 25.45 | 31.574473684210524 | False |
| `V55_PHASE4_E1_TAIL_CONTINUITY_FULL` | 1101 | 36.00107310424633 | 2.2381310013274813 | 0.2601761461319896 | 5.820319492468106 | 12.110657230614555 | 25.266666666666666 | 31.63026315789474 | False |
| `V55_PHASE4_E2_TAIL_CONTINUITY_SELECTIVE_COMMIT_FULL` | 1101 | 36.05083160639361 | 2.2878895034747586 | 0.30993464827926687 | 5.789911209491605 | 11.626193485320242 | 27.016666666666666 | 33.62447368421053 | False |

## Report/extraction repair

Blocker: v55 run artifacts did not write `timing_summary.json`, so `probe_ttt_write_seconds_mean` could not be recovered unambiguously. The `01.log` exposes pass1/stageB/stageD/pass2 timings, but not a separate probe TTT write mean.

Repair:

1. `tools/v55_experiment_report.py` records `probe_ttt_write_seconds_missing=True` instead of manufacturing a number.
2. `v55_runtime_gate_allow_probe_missing=True` is emitted when the run is done, chunk mean is within 42s, full wall time is within 28min when applicable, and the only missing runtime submetric is `probe_ttt_write_seconds_mean`.
3. The existing `full_runtime_gate_pass` / `smoke_runtime_gate_pass` fields remain False because the strict probe metric is missing. This is intentional audit behavior, not a claim that probe TTT exceeded 8s.

This repair only changes report interpretation and audit fields. It does not change trajectories, ATE, runtime, or gate decisions already produced by the runner.

## Final report generation

最终 report 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v55_experiment_report.py
```

最终输出确认:

- `report_final/v55_phase0_salvage_report.md`
- `report_final/v55_phase1_c9_h35_m1_autopsy_report.md`
- `report_final/v55_failure_type_summary.json`
- `report_final/v55_candidate_design_decision.md`
- `report_final/v55_smoke_registry.csv`
- `report_final/v55_704f_registry.csv`
- `report_final/v55_full_registry.csv`
- `report_final/v55_runtime_audit.csv`
- `report_final/v55_no_chunk_manual_percentage_audit.csv`
- `report_final/v55_role_mass_timeline.csv`
- `report_final/v55_gamma_timeline.csv`
- `report_final/v55_commit_alpha_timeline.csv`
- `report_final/v55_post_zp_delta_timeline.csv`
- `report_final/v55_layer_branch_heatmap.png`
- `report_final/v55_segment_error_timeline.png`
- `report_final/v55_final_report.md`

timeline row counts:

| file | lines |
|---|---:|
| `v55_gamma_timeline.csv` | 388 |
| `v55_role_mass_timeline.csv` | 115 |
| `v55_commit_alpha_timeline.csv` | 68 |
| `v55_post_zp_delta_timeline.csv` | 39 |

## 结束状态

- v55 按 plan 完成 Phase 0、Phase 1、Phase 2 implementation、Phase 3 smoke/704F、Phase 4 full。
- 两条 full candidate 均未过 `ATE <= 35.30m` progress gate。
- 没有启动 emergency repair full，因为 failure 没有表现为同一个 obvious bug，也没有 v54 M2 那种 overly aggressive commit guard。E2 full commit activation rate 为 `0.046783625730994156`，scale mean 为 `0.9779376293596418`。
- 最终 route: `action_space_redesign`，详见实验结果复盘。
