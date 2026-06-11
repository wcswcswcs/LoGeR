# ACL2 v55 C9ScheduleAutopsy FailForward AdaptiveTTT Clean 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/ACL2_v55_C9ScheduleAutopsy_FailForward_AdaptiveTTT_CleanPlan.md`  
执行日志: `docs/ACL2_v55_C9ScheduleAutopsy_FailForward_AdaptiveTTT_Clean_执行日志.md`  
结果根目录: `results/kitti01_hmc_v2/acl2_v55_c9schedule_autopsy_failforward_adaptivettt_clean`  
结论先行: v55 完成了 v54 M1 full salvage、C9-H35-M1 autopsy、TYPE_E 路由下的 E1/E2 TailStateContinuityGuard smoke、704F 和 full。最强 v55 full 是 E1，ATE `36.00107310424633`，比 C9 差 `+2.2381310013274813m`，比 H35 full 差 `+0.2601761461319896m`，未通过 `ATE <= 35.30m` progress gate。E2 full ATE `36.05083160639361`，也未通过 progress gate。v55 不能写成方法成功。

## 当前结果边界

- C9 reference full ATE: `33.76294210291885`。
- H35 reference full ATE: `35.74089695811434`，来自 v53 landed artifact。
- v55 S0/E1/E2 full 均为 1101 frames full KITTI01。
- 704F 只作为 screen/borderline diagnostic full gate，不当作 full success。
- `probe_ttt_write_seconds_mean` 全部保持 `NA`。原因是本轮 run 未写出 `timing_summary.json`，`01.log` 不能无歧义恢复该字段。本轮 report 没有替代或补造该数字。
- runtime 审计中 `full_runtime_gate_pass=False` 是严格 probe metric 缺失造成的形式结果；另有 `v55_runtime_gate_allow_probe_missing=True`，表示 wall/chunk/runtime 可继续审计，且没有把缺失当作真实超时。

## Phase 0: v54 M1 salvage full

输出: `report_final/v55_phase0_salvage_report.md`

| run | frames | ATE | delta vs C9 | delta vs H35 full | Rot | FinalErr | wall min | chunk mean | no chunk | manual % | role collapse | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| S0 M1 full diagnostic | 1101 | 36.18720796800255 | 2.4242658650836972 | 0.4463110098882055 | 5.855805081166548 | 11.701876341256094 | 25.45 | 31.574473684210524 | True | True | 0.0 | `m1_invalid_no_extension` |

判断:

1. v54 M1 full 没有被 704F gate 误杀成“潜在成功”。它 full 后比 H35 full 差 `+0.4463110098882055m`，超过 v55 Phase 0 的 `H35 +0.30m` borderline 条件。
2. S0 比 v54 M1 704F 更有诊断价值，因为它证明 state-energy M1 在 full tail 上也没有突破 H35。
3. S0 仍满足 no-chunk、no-manual、role-collapse audit，失败不是审计违规。

## Phase 1: C9-H35-M1 Autopsy

输出:

- `report_final/phase1_trace_autopsy/c9_h35_m1_autopsy_report.md`
- `report_final/phase1_trace_autopsy/c9_h35_m1_segment_gap_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_role_mass_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_commit_behavior_table.csv`
- `report_final/phase1_trace_autopsy/c9_h35_m1_state_variable_table.csv`
- `report_final/v55_failure_type_summary.json`

H35 full vs C9 segment evidence:

| segment | C9 RMSE | H35 full RMSE | H35 delta vs C9 |
|---|---:|---:|---:|
| seg0 000-384 | 46.39902061769452 | 44.943948305639466 | -1.4550723120550515 |
| seg1 384-700 | 34.656626116071514 | 40.48296121701416 | 5.826335100942643 |
| seg2 700-end | 11.034547189236443 | 16.778093400455763 | 5.74354621121932 |
| window 200-300 | 76.10213555431245 | 71.16459519271248 | -4.937540361599972 |
| window 400-600 | 41.896364212570404 | 48.12023088757293 | 6.223866675002526 |

Failure classification:

| field | value |
|---|---|
| selected failure type | `TYPE_E_SEG2_STATE_GAP` |
| largest segment gap | `seg1_384_700` |
| reason | `Selected by largest normalized landed evidence score; segment largest was not seg2.` |
| TYPE_A role mass gap | 0.42146643508247467 |
| TYPE_B gamma energy gap | 0.3817437927152741 |
| TYPE_C commit schedule gap | 0.0 |
| TYPE_D layer/branch gap | 0.10125088216521025 |
| TYPE_E seg2 state score | 5.74354621121932 |

解释边界:

1. 最大单段 gap 是 seg1，不是 seg2。report 没有隐瞒这个事实。
2. Formal routing 仍选 `TYPE_E_SEG2_STATE_GAP`，因为 seg2 gap 与 seg1 非常接近，且 v55 plan 的 TYPE_E 本质指向 long-tail / reset-state / memory-state 规则缺失。
3. role mass、post-zp energy、layer/branch pattern 都有 gap，但分数低于 segment/state gap。候选因此不继续扫 rho/layer/role threshold，而是按 TYPE_E 实现 TailStateContinuityGuard。

## Phase 2 Candidate Design

输出: `report_final/v55_candidate_design_decision.md`

| candidate | role mode | commit mode | intended action |
|---|---|---|---|
| E1 TailStateContinuityGuard | `adaptive_writer_tail_state_continuity_guard` | `none` | long-tail continuity risk 高时收缩 aggressive negative write，并保持 neutral continuity。 |
| E2 TailStateContinuityGuard SelectiveCommit | `adaptive_writer_tail_state_continuity_guard_selective_commit` | `tail_state_selective_commit` | 在 E1 基础上，只在 overshoot/low-cosine/risk state 下启用 selective commit。 |

实现证据:

- `loger/pipeline/ttt_write_controller.py` 增加 tail-state role mode 与 selective commit path。
- `run_pipeline_abc_v2.py` 增加对应 commit filter CLI choices。
- `tools/run_v55_tail_state_candidate.sh` 固定 E1/E2 runner，继承 no-chunk/no-manual 约束。
- `tools/v55_experiment_report.py` 统一生成 registry、timeline、audit 与 final report。

## Phase 3: 96F Smoke

输出: `report_final/v55_smoke_registry.csv`

| run | ATE96 | Rot | FinalErr | projected full min | chunk mean | adaptive gamma mean | commit activation | no chunk | manual % | role collapse | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| E1 96F | 1.328115151904405 | 1.8862757567245894 | 2.2206593095556943 | 24.383333333333333 | 27.135 | 0.004685535027723138 | NA | True | True | 0.0 | `promote_full` |
| E2 96F | 1.3295252255286152 | 1.8870539708630487 | 2.225603382352325 | 25.491666666666667 | 27.8325 | 0.004683221476928641 | 0.013888888888888888 | True | True | 0.0 | `promote_full` |

Smoke 结论:

- 两条 candidate 没有 no-chunk/manual audit 违规。
- role collapse rate 为 0。
- E2 selective commit 在 smoke 中 activation rate 很低，不是 v54 M2 的全局强收缩模式。
- `probe_ttt_write_seconds_mean` 仍缺失，所以该字段不用于伪造通过。

## Phase 3: 704F Screen

输出: `report_final/v55_704f_registry.csv`

| run | ATE704 | H35 704 delta | Rot | FinalErr | seg0 RMSE | seg1 RMSE | seg2 700-end RMSE | rolling100 p90 | projected full min | chunk mean | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E1 704F | 40.06262748588316 | 0.2643797653975287 | 2.882370016817129 | 31.55908408675562 | 46.25812424526701 | 31.047783817511935 | 30.118269474749376 | 65.21666029590304 | 24.624000000000002 | 30.687600000000003 | `borderline_diagnostic_full` |
| E2 704F | 40.08919219968784 | 0.29094447920190224 | 2.876335243188599 | 31.559640045311532 | 46.22495075354808 | 31.183869532231192 | 30.12207771653947 | 65.20895700126918 | 27.284 | 33.6036 | `borderline_diagnostic_full` |

704F 结论:

1. E1/E2 都没有达到 promote full 条件，但都在 `H35_704 +0.35m` 内，并且 projected runtime 小于 28min，因此按 v55 plan 进入 borderline diagnostic full。
2. 704F 的 seg2 只有 frames 700-704 的 4 帧，不能当成 full 后段成功证据。
3. E2 704F commit activation rate 为 `0.03111111111111111`，scale mean 为 `0.9852746951875663`，已经避免了 v54 M2 activation rate `1.0` 的过强问题，但 704F ATE 仍未优于 H35。

## Phase 4: Full KITTI01

输出: `report_final/v55_full_registry.csv`

| run | ATE | delta vs C9 | delta vs H35 full | Rot | FinalErr | frames | wall min | chunk mean | progress | soft | close | excellent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| S0 M1 full diagnostic | 36.18720796800255 | 2.4242658650836972 | 0.4463110098882055 | 5.855805081166548 | 11.701876341256094 | 1101 | 25.45 | 31.574473684210524 | False | False | False | False |
| E1 TailStateContinuityGuard full | 36.00107310424633 | 2.2381310013274813 | 0.2601761461319896 | 5.820319492468106 | 12.110657230614555 | 1101 | 25.266666666666666 | 31.63026315789474 | False | False | False | False |
| E2 TailStateContinuityGuard SelectiveCommit full | 36.05083160639361 | 2.2878895034747586 | 0.30993464827926687 | 5.789911209491605 | 11.626193485320242 | 1101 | 27.016666666666666 | 33.62447368421053 | False | False | False | False |

Full segment detail:

| run | seg0 000-384 RMSE | delta vs H35 seg0 | seg1 384-700 RMSE | delta vs H35 seg1 | seg2 700-end RMSE | delta vs H35 seg2 | window 200-300 RMSE | window 400-600 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 M1 full | 45.19259648628828 | 0.24864818064881433 | 41.37259557730417 | 0.8896343602900103 | 17.052799923668296 | 0.27470652321253297 | 71.48199361201024 | 49.1574436018587 |
| E1 full | 45.107472031237066 | 0.16352372559759942 | 41.052755879364895 | 0.5697946623507377 | 16.7940814495255 | 0.01598804906973683 | 71.50496854476629 | 48.842647653941164 |
| E2 full | 45.08193930007217 | 0.13799099443270535 | 41.233088460618355 | 0.7501272436041972 | 16.804658166530675 | 0.026564766074912427 | 71.50591322306683 | 49.04410698541849 |

Rolling detail:

| run | rolling50 mean | rolling50 p90 | rolling50 worst | rolling100 mean | rolling100 p90 | rolling100 worst | rolling200 mean | rolling200 p90 | rolling200 worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 M1 full | 31.039076952432808 | 60.68307961242692 | 80.11250103605043 | 32.29751361461802 | 57.40553583947136 | 71.6195378081881 | 34.72904020870987 | 54.264522750455455 | 55.98682584833823 |
| E1 full | 30.796057371101497 | 60.36220102151173 | 80.12441365276673 | 32.06304612433349 | 57.10346717918294 | 71.63780007477573 | 34.51281051754143 | 54.26270809813783 | 56.014984495436146 |
| E2 full | 30.845234688026462 | 60.62833316377229 | 80.14613665575341 | 32.12257915108841 | 57.34003920185977 | 71.64239070190105 | 34.57927988654417 | 54.2836289063261 | 56.02537450374316 |

Runtime detail:

| run | wall min | chunk mean | chunk max | pass1 mean | stageB mean | stageD mean | pass2 mean | probe TTT mean |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S0 M1 full | 25.45 | 31.574473684210524 | 33.88 | 11.513684210526316 | 8.455526315789474 | 1.1157894736842104 | 10.489473684210527 | NA |
| E1 full | 25.266666666666666 | 31.63026315789474 | 33.21 | 11.367105263157894 | 8.660526315789475 | 1.108421052631579 | 10.494210526315788 | NA |
| E2 full | 27.016666666666666 | 33.62447368421053 | 41.89 | 11.832894736842107 | 9.522105263157895 | 1.1057894736842109 | 11.163684210526316 | NA |

Internal evidence:

| run | adaptive gamma mean | neutral lambda mean | commit activation | commit scale | no chunk | manual % | role collapse |
|---|---:|---:|---:|---:|---|---|---:|
| S0 M1 full | 0.00505745034186273 | 1.0 | NA | NA | True | True | 0.0 |
| E1 full | 0.003979641922725071 | 1.0 | NA | NA | True | True | 0.0 |
| E2 full | 0.0039707241324253096 | 1.0 | 0.046783625730994156 | 0.9779376293596418 | True | True | 0.0 |

## Full Decision

Best v55 full candidate: E1 TailStateContinuityGuard full.

| threshold | condition | E1 result | pass |
|---|---|---:|---|
| progress pass | ATE <= 35.30m | 36.00107310424633 | False |
| soft pass | ATE <= 34.60m | 36.00107310424633 | False |
| close-to-C9 pass | ATE <= 34.30m | 36.00107310424633 | False |
| excellent pass | ATE <= 34.06m | 36.00107310424633 | False |

Decision: no v55 full candidate passed. v55 stops with `action_space_redesign` routing.

No emergency repair was launched because:

1. Both full candidates passed no-chunk/manual/role-collapse audits.
2. Runtime wall time stayed below 28min for E1/E2 full.
3. E2 selective commit was not overly aggressive: activation rate `0.046783625730994156`, scale mean `0.9779376293596418`.
4. Failure pattern is action-space insufficiency, not a single obvious bug or a guard that can be repaired with one low-risk emergency full.

## Report/Extraction Repair

During v55 reporting, `probe_ttt_write_seconds_mean` was unavailable because the runner wrote `wall_time_summary.json` but did not write `timing_summary.json`. I modified `tools/v55_experiment_report.py` to:

1. Preserve `probe_ttt_write_seconds_mean` as NA.
2. Add `probe_ttt_write_seconds_missing`.
3. Add `v55_runtime_gate_allow_probe_missing`.
4. Keep strict `full_runtime_gate_pass` / `smoke_runtime_gate_pass` false when the probe field is missing.

This repair is report-only. It did not change any rollout trajectory, ATE, segment RMSE, runtime, or gate decision.

## 主要分析

1. v54 M1 的 full salvage 证明 v54 704F gate 不是误杀成功候选。S0 full 比 H35 full 差 `+0.4463110098882055m`，因此 M1 action-space 不应继续扩展。
2. Phase 1 的最大单段 gap 是 seg1，seg2 gap 也非常接近。把 failure type 路由成 `TYPE_E_SEG2_STATE_GAP` 是一个 formal routing 决策，不代表“seg2 是唯一最大 gap”。更准确地说，H35 与 C9 的差距集中在 mid-tail/long-tail state，而不是前 384F。
3. E1 确实比 S0 有小幅改善: full ATE 从 `36.18720796800255` 到 `36.00107310424633`，seg1 从 `41.37259557730417` 到 `41.052755879364895`，seg2 从 `17.052799923668296` 到 `16.7940814495255`。但这些改善仍不足以超过 H35 full，更不足以接近 C9。
4. E2 修复了 v54 M2 的“commit 每个 chunk 都强收缩”问题。E2 full commit activation rate 只有 `0.046783625730994156`，scale mean `0.9779376293596418`。但是 selective commit 变健康之后，性能并没有明显提升，说明缺失的不是简单 commit activation rule。
5. no-chunk/no-manual 约束没有破。所有 v55 runs 的 audit 都是 True，role collapse rate 都是 0。失败不是因为审计违规。
6. runtime 主体已经可用。E1 full wall `25.266666666666666min`，E2 full wall `27.016666666666666min`，都低于 28min。但 probe TTT write metric 缺失必须继续修报告/runner 输出，不能在论文式结论里写成已测得。

## Insight

1. TailStateContinuityGuard 是一个可执行、审计干净的 fail-forward candidate，但它只产生了小幅修正，没有改变 H35 平台。
2. v55 的负结果比 v54 更可靠，因为 v55 补了 M1 full diagnostic，并且对 E1/E2 都跑了 full，而不是停在 704F。
3. 当前 adaptive split + gamma shrink + selective commit 的 action space 仍然太接近 H35。它能调节写入强度，但没能学到 C9 的 layer/branch action pattern 或 READ-TTT joint schedule。
4. 继续做 rho/layer/threshold 小扫不符合证据。E1/E2 已经按最大 failure type 修了 causal tail-state action，仍没有 progress pass。
5. 下一轮应改变 TTT action space，而不是给 E1/E2 加小常数。更有根据的方向是 branch/layer-specific replay、filtered/no-long-write commit、或 READ-TTT joint state controller。若继续 clean TTT 线，应先把 mid-tail state 变量和 layer/branch action 的对应关系做成新的 autopsy，而不是直接命名 M4/M5。

## 最终结论

v55 没有得到可升级为成功的 clean adaptive TTT candidate。最强 v55 full 是 E1 TailStateContinuityGuard，ATE `36.00107310424633`，仍比 H35 full 差 `+0.2601761461319896m`，比 C9 差 `+2.2381310013274813m`，未通过 progress gate。E2 selective commit full ATE `36.05083160639361`，也未通过 progress gate。

可靠负结论: 当前 no-chunk adaptive split/gamma/commit 修补不足以复现 C9 clean TTT writing behavior。

可靠正结论: v55 打通了 fail-forward 实验链路，补跑了 v54 M1 full，完成了 C9-H35-M1 autopsy，按 failure type 自动实现并并行评估了 E1/E2，且所有候选保持 no-chunk/no-manual audit 通过。下一步应该重写 action space，而不是继续小扫当前 formula。
