# ACL2 v54 Fast State-Conditioned Adaptive TTT Clean-to-C9 实验结果复盘

日期: 2026-06-09  
计划文档: `docs/ACL2_v54_Fast_StateConditioned_AdaptiveTTT_Clean_to_C9_Plan.md`  
执行日志: `docs/ACL2_v54_Fast_StateConditioned_AdaptiveTTT_Clean_to_C9_执行日志.md`  
结论先行: v54 实现并运行了 M1 State-Energy Matched Split 与 M2 State-Energy + Directional Commit Guard 的 704F screen，但两者都未通过 promotion gate，因此按计划没有启动 full。M1 704F ATE `40.09525341863154`，比 H35 704F 差 `+0.2970056981459063m`；M2 704F ATE `41.38387973914448`，比 H35 704F 差 `+1.5856320186588473m`，且 projected full runtime `28.474666666666668min` 超过 28min gate。v54 当前结果不能写成方法成功。

## 当前结果边界

- C9 reference constant: `33.76294210291885`。
- H35 reference 来自 v53 landed artifact，不是本轮重跑。
- Phase 1 teacher/student autopsy 复用 v52 landed trace；`V52_EnergyMatched` 没有在 `run_overview.csv` 中找到 landed row，因此记录为 missing，不补数字。
- v54 M1/M2 704F 已完成。
- 因 704F gate 未通过，没有 full run；full 指标保持 `NA`，不能用 704F 当 full 成功。
- `probe_ttt_write_seconds_mean` 在 v54 registry 中为 `NA`，原因是本轮 run 未写出 `timing_summary.json`，report fallback 只能从 `01.log` 解析 `pass2_control_seconds_mean`，不能无歧义恢复 `probe_ttt_write_seconds_mean`。该字段未被替代或补造。

## Phase 0 Reference

输出: `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_reference_registry.csv`

| reference | frames | ATE | Rot | FinalErr | wall min | chunk mean | TTT mean | no chunk | manual % |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| H35 704F | 704 | 39.798247720486 | 2.883889157786 | 31.091606862781 | 16.383333333333 | 37.501508817673 | 4.921872768402 | True | True |
| H35 full | 1101 | 35.740896958114 | 5.750002143364 | 11.138076035141 | 24.783333333333 | 37.640955912439 | 4.905587773574 | True | True |

H35 segment reference:

| reference | seg0 000-384 RMSE | seg1 384-700 RMSE | seg2 700-end RMSE | rolling100 p90 |
|---|---:|---:|---:|---:|
| H35 704F | 46.051212827057 | 30.667454243705 | 29.647952695105 | 64.931366192051 |
| H35 full | 44.943948305639 | 40.482961217014 | 16.778093400456 | 56.285288896203 |

说明:

- 704F 的 `seg2_700_end` 只有 4 帧，不能当成 full 后段成功证据，只用于计划要求的 gate 字段。
- H35 full 仍是 v53 best clean adaptive baseline，ATE = `35.74089695811434`。

## Phase 1 Teacher/Student Autopsy

输出:

- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_phase1_teacher_student_autopsy.csv`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/v54_phase1_teacher_student_autopsy.md`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/teacher_student_post_zp_delta_timeline.png`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_initial/teacher_student_layer_branch_heatmap.png`

| source | run | ATE | delta vs C9 | risk source | role source | gamma mean |
|---|---|---:|---:|---|---|---:|
| v52 autopsy | C9_exact_teacher | 33.762943 | 0.000001 | control_prior_D_tok_quantile_proxy;update_conflict_energy | control_prior_D_tok_quantile_proxy;controller_debug | 0.003811 |
| v52 autopsy | V46B_fixed_F111_teacher | 36.650736 | 2.887794 | update_conflict_energy | controller_debug | 0.004000 |
| v52 autopsy | V50_split_resxdg_student | 35.985305 | 2.222363 | ttt_residual_x_dg | controller_debug | 0.004607 |
| v53 reference | H35 704F | 39.798248 | 6.035306 | ttt_residual_x_dg | adaptive_writer_sc_gamma | 0.016451 |
| expected but missing | V52_EnergyMatched | NA | NA | NA | NA | NA |

分析:

1. v52/v53 证据支持 v54 的设计假设: role split 可以进入 controller 并被审计，但 post-zp delta energy、gamma timing、commit behavior 仍与 C9 有差距。
2. H35 证明 `ttt_residual_x_dg + layer 0/8/17 rho0.0075` 是 v53 里最强 clean adaptive 起点；v54 M1/M2 固定使用这组全局层常数作为 `gamma0`，不再做 rho/layer 小扫。
3. 当前没有 landed 的 `V52_EnergyMatched` row；本轮报告不补数字，只把缺失作为证据链限制。

## Phase 2/3 v54 Candidate Results

输出:

- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_final/v54_704_registry.csv`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_final/v54_704_report.md`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_final/v54_code_audit.md`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_final/v54_state_energy_gamma_timeline.csv`
- `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_final/v54_commit_alpha_timeline.csv`

| run | ATE704 | H35 delta | seg2 RMSE | rolling100 p90 | projected full min | chunk mean | pass2 mean | TTT mean | no chunk | manual % | collapse | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| M1 state-energy | 40.095253418632 | +0.297005698146 | 30.466111819241 | 65.204726842289 | 25.764000000000 | 31.874800000000 | 10.324000000000 | NA | True | True | 0 | fail |
| M2 directional commit | 41.383879739144 | +1.585632018659 | 30.614086845687 | 67.420535169239 | 28.474666666667 | 34.943200000000 | 10.505999999999 | NA | True | True | 0 | fail |

Gate reasons:

- M1: `ATE704_not_0.20m_better_than_H35`, `seg2_or_rolling100_not_0.50m_better_than_H35`。
- M2: `projected_runtime_gt_28min`, `ATE704_not_0.20m_better_than_H35`, `seg2_or_rolling100_not_0.50m_better_than_H35`。

Segment detail:

| run | seg0 000-384 RMSE | seg1 384-700 RMSE | seg2 700-end RMSE | rolling50 p90 | rolling100 p90 | rolling200 p90 |
|---|---:|---:|---:|---:|---:|---:|
| H35 704F ref | 46.051212827057 | 30.667454243705 | 29.647952695105 | 64.904113678620 | 64.931366192051 | 54.558942007435 |
| M1 state-energy | 46.249898412675 | 31.152032898906 | 30.466111819241 | 65.195654438245 | 65.204726842289 | 54.814395441515 |
| M2 directional commit | 47.763853842852 | 32.113689582328 | 30.614086845687 | 67.231907041002 | 67.420535169239 | 56.525868431110 |

Runtime detail:

| run | frames | timing chunks | wall min | projected full min | chunk mean | chunk max | pass1 mean | stageB mean | stageD mean | pass2 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H35 704F ref | 704 | 25 | 16.383333333333 | 24.902666666667 | 37.501508817673 | 42.879514455795 | 11.407875318527 | 8.496997842789 | 1.065997438431 | 9.981320409775 |
| M1 state-energy | 704 | 25 | 16.950000000000 | 25.764000000000 | 31.874800000000 | 34.490000000000 | 11.858800000000 | 8.624000000000 | 1.068000000000 | 10.324000000000 |
| M2 directional commit | 704 | 25 | 18.733333333333 | 28.474666666667 | 34.943200000000 | 40.500000000000 | 12.132000000000 | 11.227600000000 | 1.077600000000 | 10.506000000000 |

Internal evidence:

- M1/M2 role source: `adaptive_writer_state_energy`。
- M1/M2 role collapse rows: `0 / 450`。
- M1 adaptive gamma mean from registry: `0.005237599410789294`。
- M2 adaptive gamma mean from registry: `0.004881805181503296`。
- M2 commit filter mode: `state_energy_directional_commit`。
- M2 commit filter active rows: `25`，applied rows: `25`，activation rate mean: `1.0`，scale mean: `0.28364215257213743`。
- Timeline extraction repair after run generated:
  - `v54_state_energy_gamma_timeline.csv`
  - `v54_commit_alpha_timeline.csv`
  - `v54_state_energy_gamma_timeline.png`
  - `teacher_student_commit_delta_timeline.png`

Interpretation:

1. M1 did enter the intended action path: state-energy gamma was logged per chunk, no manual role percentages were used, and role collapse was 0. However, it did not beat H35; it was worse on ATE, seg1, seg2, and rolling100 p90.
2. M2 made performance and runtime worse. Commit alpha mean around `0.284` with activation rate `1.0` indicates the guard was active on every chunk and aggressively pulled candidate toward native. That did not protect trajectory continuity; it degraded ATE by `+1.5856m` vs H35 and exceeded projected runtime.
3. Both candidates failed the plan's promotion condition, so no full run is valid or allowed.

## Full Decision

输出: `results/kitti01_hmc_v2/acl2_v54_fast_state_conditioned_adaptive_ttt_clean_to_c9/report_final/v54_full_report.md`

| run | ATE | delta vs C9 | frames | wall min | gate |
|---|---:|---:|---:|---:|---|
| v54 full | NA | NA | NA | NA | not started |

Reason:

- `704F gate did not pass; full run was not allowed.`

This is an intentional stop, not an unfinished full experiment.

## Report/Extraction Repair

During reporting, `v54_state_energy_gamma_timeline.csv` and `v54_commit_alpha_timeline.csv` were initially empty although `01.log` contained the relevant debug fields. The cause was report extraction only reading `hmc_state_hash.jsonl`, whose compact rows did not preserve full layer debug. I modified `tools/v54_experiment_report.py` to parse both `hmc_state_hash.jsonl` and `01.log`, then reran `py_compile` and regenerated `report_final`.

This repair changed only report artifact extraction. It did not change M1/M2 trajectories, ATE, runtime, or gate decisions.

## Failure Routing 与下一步唯一改法

计划在 `9.1 如果 M1/M2 704F 都失败` 里要求: 不允许继续做 rho/layer/role-threshold 小扫；必须分析 missing state variable 与 layer/branch energy mismatch，并且只提出一个 action-space change。本轮遵守该停止条件，所以没有继续启动 M3/M4 或 full。

失败定位:

1. M1 的 scalar state-energy target 不够。它把 `E_t` 定义成 `EMA(E_native)`，能让 gamma 进入因果能量闭环，但不能区分 C9 在 late/seg2 状态下的离散时序动作。结果 M1 的 ATE、seg1、seg2、rolling100 都差于 H35。
2. M2 的 directional commit guard 不是缺失变量的替代品。它只看 candidate/native 方向和能量比，实际 activation rate = `1.0`，scale mean = `0.28364215257213743`，说明 guard 几乎无条件强收缩，导致 runtime 和 ATE 同时变差。
3. 当前可见的 `risk/prior/native/candidate energy` 仍然存在 state aliasing: 一些 chunk 在局部 risk/energy 上相似，但 C9 的写入动作不同。v54 的 M1/M2 没有引入足以拆开这些状态的 online variable。

下一步唯一建议改法:

```text
M3: Causal Late-State Disambiguated Energy Target
```

只改一个 action-space: 把 M1 的

```text
E_t = EMA(E_native)
```

替换为一个 no-GT、no-chunk-map 的 state-conditioned target:

```text
E_t = EMA(E_native) * f(S_late)
```

其中 `S_late` 必须只来自 causal runtime state，不能来自 chunk id 或 GT。建议组成:

- accumulated native path-length / pose-step EMA，用来表示 sequence progress，但不使用 absolute chunk id；
- pass1/pass2 pose disagreement EMA，例如 `pass1_pass2_pose_t_mean`、`pass1_pass2_pose_r_deg_mean`；
- TTT state continuity drift，例如 `memory_ttt_w0_mean_rel_diff` 或 `memory_ttt_mean_rel_diff`；
- layer8/17 branch0 post-zp candidate/native energy ratio 与 cosine mismatch；
- dynamic/read state summary，例如 `prior_mean_D_tok`、`prior_q90_D_tok`、`prior_hmc_write_score_mean`。

`f(S_late)` 应该是固定公式或 robust-threshold gate，不是 learned router，不用 GT，不用 C9 chunk map。它的目的不是扫 rho，而是让同一套 state-energy writer 在 late/seg2 drift 状态下选择不同 target energy。这样才针对计划里指出的 state aliasing，而不是继续调 M1/M2 的局部常数。

本轮没有实现 M3 的原因:

- v54 计划明确 Phase 2 只允许实现 M1/M2。
- M1/M2 704F gate 均失败，计划规定不允许 full，也不允许在同轮继续开大矩阵。
- 继续跑 M3 会超出本轮 plan scope；应作为 v55 或单独 addendum，并先做 C9 vs M1/M2 state-feature autopsy。

## 主要 Insight

1. v54 的实现闭环是可执行的，但效果失败。M1/M2 都产生了合法 704F artifact、进入统一 report/gate，但都没有超过 H35。
2. State-energy gamma 没有解决 H35 平台。M1 使用 causal energy EMA 后，ATE `40.0953`，仍差于 H35 704F `39.7982`，说明仅用当前 `risk/prior/native/candidate energy` 仍不能分辨 C9 的关键时序动作。
3. Directional commit guard 过强且无益。M2 commit activation rate `1.0`、scale mean `0.2836`，说明几乎每个 chunk 都被强收缩；结果 ATE 和 runtime 都变差。
4. no-chunk/no-manual 约束没有破。两个 run 的 no-chunk audit 与 manual percentage audit 均为 True，role collapse 为 0，失败不是因为审计违规。
5. 下一步不应继续在 rho/layer/role threshold 上小扫。v54 失败支持计划里的 failure routing: 需要一个能区分 C9 late/seg2 行为的更强 online state variable，或更换 action space；不是继续把局部 risk/gamma 调细。

## 结论

v54 没有得到可升级 full 的 clean adaptive TTT candidate。最强 v54 candidate 是 M1 state-energy，704F ATE `40.09525341863154`，仍比 H35 704F 差 `0.2970056981459063m`，且未改善 seg2/rolling100 gate。M2 directional commit 更差，704F ATE `41.38387973914448`，projected full runtime `28.474666666666668min` 超过 28min gate。

因此本轮按计划停止在 704F，不启动 full，不声称 v54 成功。可靠负结论是: 当前 state-energy matched split 与 simple directional commit guard 仍不足以复现 C9 的 clean no-chunk TTT writing behavior。可靠正结论是: v54 的审计边界、报告链路和 M1/M2 runtime execution path 已经打通，后续可以基于这些 artifact 做 action-space 级别的失败分析。
