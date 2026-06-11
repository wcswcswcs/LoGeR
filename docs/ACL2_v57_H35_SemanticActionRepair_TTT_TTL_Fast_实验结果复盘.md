# ACL2 v57 H35 SemanticActionRepair TTT TTL Fast 实验结果复盘

日期: 2026-06-09
计划文档: `docs/ACL2_v57_H35_SemanticActionRepair_TTT_TTL_FastPlan.md`
执行日志: `docs/ACL2_v57_H35_SemanticActionRepair_TTT_TTL_Fast_执行日志.md`
结果根目录: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast`

结论先行: 本文件只基于已落盘 artifact 生成；未运行、未写出或不可推断的字段保持 NA/unavailable。v57 修复并验证了 semantic action-realization，但没有得到可报告成功: SREAD03 在 704F 有改善，full 反而比 H35 差；TTT 三条 704F 全部回退，因此没有 TTT full 和 combo。

## H35 参照

- H35 full ATE: `35.740897`。
- H35 704F ATE: `39.798248`。
- semantic minimum progress gate: ATE <= `35.240900`；target success gate: ATE <= `33.740900`。
- TTT target success gate: ATE <= `34.740900`。

## Phase 0: v56 修复性审计

- v56 semantic action inactive rows: `10`。
- semantic audit: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/report_final/semantic_action_realization_audit.md`。
- TTT audit: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/report_final/ttt_action_regression_audit.md`。
- 关键修复结论: 如果 v56 semantic rows 的 source token / attention mass 字段为 0 或 NA，只能判定 action inactive 或 evidence missing，不能写成语义无用。

## Phase 1: Semantic READ action smoke

| row | run | ATE | stage_hit | labels | groups | src_mean | mass_before | mass_after | empty | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| S0_96F | V57R1_S0_FORCED_SEMANTIC_SOURCE_SKIP_TRACE_96F | 1.279743 | 1.000000 | NA | 3.000000 | 2354.750000 | 0.118548 | 0.000000 | 0 | True |
| S1_96F | V57R1_S1_SKY_LOWSTUFF_SOURCE_SKIP_TRACE_96F | 1.279743 | 1.000000 | NA | 3.000000 | 2354.750000 | 0.118548 | 0.000000 | 0 | True |
| S2_96F | V57R1_S2_HIGH_D_SEMANTIC_SOURCE_SKIP_TRACE_96F | 1.279673 | 1.000000 | 32558.500000 | 3.000000 | 2354.625000 | 0.118524 | 0.000000 | 0 | True |
| S0_96F | V57_S0_FORCED_SEMANTIC_SOURCE_SKIP_96F | 1.279743 | 1.000000 | NA | NA | 2354.750000 | 0.118548 | 0.000000 | 0 | True |
| S1_96F | V57_S1_SKY_LOWSTUFF_SOURCE_SKIP_96F | 1.279743 | 1.000000 | NA | NA | 2354.750000 | 0.118548 | 0.000000 | 0 | True |
| S2_96F | V57_S2_HIGH_D_SEMANTIC_SOURCE_SKIP_96F | 1.279673 | 1.000000 | NA | NA | 2354.625000 | 0.118524 | 0.000000 | 0 | True |

Phase1 gate pass: `True`。

注: `V57_S*_96F` 是第一批 action hook smoke；`V57R1_*_TRACE_96F` 是补充 semantic prior/debug 落盘后重跑的审计 smoke。两批都保留在 registry 中，最终 label/group/source-mass 证据以 trace rows 为主。

## Phase 2: Semantic READ 704F

| row | ATE | dH35_704 | rolling100p90 | src_mean | mass_before | mass_after | static_protect | decision |
|---|---|---|---|---|---|---|---|---|
| SREAD01_704F | 44.687484 | 4.889237 | 73.595006 | 2592.720000 | 0.120387 | 0.000000 | 2697.920000 | semantic_active_regression_repair_required |
| SREAD02_704F | 44.712281 | 4.914033 | 73.620321 | 2485.220000 | 0.115275 | 0.000000 | NA | semantic_active_regression_repair_required |
| SREAD03_704F | 38.209718 | -1.588530 | 59.986602 | 2481.520000 | 0.117725 | 0.000000 | 7713.840000 | promote_full |
| SREAD04_704F | 44.687484 | 4.889237 | 73.595006 | 2592.720000 | 0.120387 | 0.000000 | 7752.080000 | semantic_active_regression_repair_required |

Semantic promoted to full: `SREAD03_704F`。

Semantic 704F blocker repair 记录:
- SREAD01 action active but regressed: delta vs H35_704 `4.889237`；SREAD04 加 static rescue 后 protected mass `7752.080000`，但 ATE 仍为 `44.687484`，未修复回退。
- SREAD02 只有 group-level semantic evidence 也确实 active: affected source tokens mean `2485.220000`，attention mass `0.115275` -> `0.000000`，但 704F delta `4.914033`。
- SREAD03 是唯一 promoted semantic row: 704F ATE `38.209718`，delta vs H35_704 `-1.588530`，rolling100_p90 `59.986602`。

## Phase 3: Semantic READ full

| row | ATE | dH35 | Rot | FinalErr | wall | min_progress | target_success |
|---|---|---|---|---|---|---|---|
| SREAD03_FULL | 36.422824 | 0.681927 | 3.787143 | 20.435976 | 27.950000 | False | False |

Semantic full 判定: `SREAD03_FULL` ATE `36.422824`，delta vs H35 `0.681927`，runtime `27.950000` min。minimum progress=`False`，target success=`False`。按计划，ATE >= H35+0.3m 属 hard fail。

## Phase 4: New TTT action smoke/704/full

| row | ATE | stable | risk | no_long | short_norm | native_cos | gate |
|---|---|---|---|---|---|---|---|
| TTT01_96F | 1.227985 | 0.458740 | 0.000000 | 0.541260 | NA | 0.930831 | True |
| TTT02_96F | 1.254267 | NA | NA | NA | 0.000050 | NA | True |
| TTT03_96F | 1.260858 | 0.894860 | 0.105140 | 0.105140 | NA | NA | True |

TTT smoke gate pass: `True`。

| row | ATE | dH35_704 | rolling100p90 | no_long | short_norm | post_zp | decision |
|---|---|---|---|---|---|---|---|
| TTT01_704F | 41.476049 | 1.677802 | 67.512335 | 0.540455 | NA | NA | repair_no_long_too_broad |
| TTT02_704F | 41.781139 | 1.982891 | 68.104662 | NA | 0.000038 | NA | stop_no_ttt_screen_signal |
| TTT03_704F | 41.816010 | 2.017762 | 68.087778 | 0.133331 | NA | NA | stop_no_ttt_screen_signal |

TTT promoted to full: `none`。

TTT 704F blocker repair 记录:
- TTT01 复现 broad no-long 过宽问题: no_long mass `0.540455`，delta vs H35_704 `1.677802`。
- TTT02 short residual/TTL 机制触发，但 short_residual_norm `0.000038`，704F delta `1.982891`，没有 screen signal。
- TTT03 将 no-long 收窄到 `0.133331`，但 704F delta 仍为 `2.017762`，说明简单 high-risk/high-influence 收窄不足。

无 landed run。

## Phase 5: Combo

无 landed run。

## 关键实验结论

- Best semantic full: `SREAD03_FULL` ATE `36.422824`, delta vs H35 `0.681927`，minimum progress `False`，target success `False`。
- TTT full 未运行或未完成；若 smoke/704 gate 未通过，这是按计划停止。
- 最终 gate: semantic_min_progress=`False`, semantic_target=`False`, ttt_min_progress=`False`, ttt_target=`False`, combo_run=`False`。
- Combo 未启动；只有单目标 full improvement >=0.5m 才允许组合。

## Insight 与证据链

- 第一层证据是 action-realization，而不是 ATE。v57 先修复 v56 CLI 参数未传入 HMC 的 wiring blocker，再用 S0/S1/S2 smoke 检查 source token 与 attention mass。
- S0/S1/S2 修复后证明 semantic projection -> HMC prior -> pi3 compact K/V source skip 的代码路径通；v56 的语义负结果不能解释成语义本身无效。
- 704F 上 SREAD03 的 C23/action-guard 路线是真信号，但 full 没有继承这个收益，说明当前 READ action 的局部改善不能稳定覆盖 full trajectory。
- SREAD04 的 static rescue 没能修复 SREAD01 的 active regression；后续不能继续做宽泛语义过滤，需要更细的作用域/trajectory-state gate。
- TTT 路线不再扩大 broad no-long-write；TTT03 已按计划收窄但仍回退，TTL/short residual 当前实现也没有优于 H35 的 704F 证据。
- dense overlay/heatmap 图若为 no-data，是因为当前 runner 没有落盘对应空间/tensor trace，复盘不会把缺失解释成 0。

## 必答问题

1. v56 semantic failure 是否主要因为 action inactive: `True`，证据见 Phase0 audit。
2. 修复后 semantic 是否选中真实 source tokens: `yes`。
3. semantic READ 是否达到 0.5m/2m full 收益: `not proven` / `not proven`。
4. TTL/short residual 是否优于 broad no-long-write: `not proven`。当前只证明 TTT02 机制触发但 704F 回退。
5. new TTT action 是否达到 1m full 收益: `not proven`。
6. 两个目标失败的当前证据: semantic action-realization 已修复，但 READ 局部 704F 收益不能转成 full；TTT 简单生命周期/no-long action space 不足。还不能证明 H35 主要误差完全不来自 READ/TTT。
7. 后续方向: 语义若继续，应聚焦 SREAD03 这类局部收益的 full-transfer/trajectory-state gate；TTT 不建议继续 broad no-long 或同形 TTL 小扫，应转向 trajectory-state、merge-gauge、pose-scale controller 等更底层状态控制。

## 审计材料

- registries: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/report_final`
- figures: `results/kitti01_hmc_v2/acl2_v57_h35_semantic_action_repair_ttt_ttl_fast/report_final/figures`
- `semantic_action_realization_audit.md` / `ttt_action_regression_audit.md`
