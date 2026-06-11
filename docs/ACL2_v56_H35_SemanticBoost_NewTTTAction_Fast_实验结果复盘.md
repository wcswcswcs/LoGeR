# ACL2 v56 H35 SemanticBoost NewTTTAction Fast 实验结果复盘

日期: 2026-06-09
计划文档: `docs/ACL2_v56_H35_SemanticBoost_NewTTTAction_FastPlan.md`
执行日志: `docs/ACL2_v56_H35_SemanticBoost_NewTTTAction_Fast_执行日志.md`
结果根目录: `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast`

结论先行: 本文件只基于已落盘结果生成；未运行或未写出的指标保持 NA/unavailable。v56 没有证明 semantic 能改善 H35 重建，也没有证明 new TTT action 能改善 H35。最佳 full 仍是 H35 等价结果，ATE `35.74089695811434`，没有达到 `<=35.2409` minimum progress gate。

## Phase 0 H35 baseline

- H35 landed reference full ATE: `35.740897`。
- H35 landed reference 704F ATE: `39.798248`。
- H35 repeat full ATE: `35.740897`，drift vs landed H35: `0.000000`。
- H35 repeat Rot/FinalErr: `5.750002` / `11.138076`。
- H35 repeat runtime: wall `25.283333min`, chunk mean `31.551579s`, probe TTT mean `NA`。

## Track A 704F screen

| candidate | ATE | delta_vs_H35_704 | stage_c_hit | sem_labels | source_tokens | decision |
|---|---|---|---|---|---|---|
| A1 | 40.077571 | 0.279324 | 1.000000 | NA | 0.000000 | stop_semantic_704_regression |
| A2 | 39.798248 | 0.000000 | 1.000000 | NA | 0.000000 | borderline_full_allowed |
| A3 | 39.798248 | 0.000000 | 1.000000 | NA | 0.000000 | borderline_full_allowed |
| A4 | 40.077571 | 0.279324 | 1.000000 | NA | 0.000000 | stop_semantic_704_regression |

## Track A full

| candidate | ATE | delta_vs_H35 | Rot | FinalErr | wall_min | progress |
|---|---|---|---|---|---|---|
| A2 | 35.740897 | 0.000000 | 5.750002 | 11.138076 | 29.616667 | False |
| A3 | 35.740897 | 0.000000 | 5.750002 | 11.138076 | 28.216667 | False |

## Track B 704F screen

| candidate | ATE | delta_vs_H35_704 | stable_mass | risk_mass | no_long | decision |
|---|---|---|---|---|---|---|
| B1 | 41.439231 | 1.640983 | 0.459595 | 0.000000 | 0.540405 | stop_ttt_704_regression |
| B2 | 41.727226 | 1.928979 | 0.866673 | 0.133327 | 0.133327 | stop_ttt_704_regression |
| B3 | 40.867985 | 1.069737 | NA | NA | NA | stop_ttt_704_regression |
| B4 | 41.233692 | 1.435444 | NA | NA | NA | stop_ttt_704_regression |

## Track B full

无 landed run。

## 关键分析

- Track A best full 是 `A2`，ATE `35.740897`，相对 H35 full `0.000000`。
- Track B 没有 landed full；若 704F 未过 gate，按计划不继续 full。
- 全部 v56 best full 是 `A2`，ATE `35.740897`。
- `probe_ttt_write_seconds_mean` 如果为 NA，原因是该 run 未写出 `timing_summary.json` 中的 probe 字段；本报告没有替代或补造。
- requested dense overlays 若显示 no-data，是因为当前 runner 未落盘对应空间图，不代表对应量为 0。

## Blocker 修复证据

### Stage C short-tail cache miss

第一次 A1-A4 96F 全部失败在 Stage C require-hit: runner 需要 `chunk_003_000087_000096/masklet.pt`，但 cache 中只有完整 chunk `chunk_003_000087_000119/masklet.pt`。修复是在 `run_pipeline_abc_v2.py` 中加入 short-tail superset cache read，并裁剪 `MaskletOutput` 时间维。这个修复没有关闭 require-hit，也没有启用 inline Stage C fallback。

修复后 A1-A4 FIX1 smoke 全部完成。证据: `v56_track_a_smoke_registry.csv` 中四条 FIX1 smoke 的 `stage_c_cache_hit_rate=1.0`、`stage_c_cache_hit_count=4`、`stage_c_cache_seen=4`、`stage_c_cache_superset_hit_count=1`、`stage_c_cache_missing_count=0`。

### B4 native delta gate enum

B4 96F 第一次失败是 argparse choices 漏掉 `orthogonal_suppress`。代码中 `ttt_write_controller.py` 已有该实现，因此修复是在 `run_pipeline_abc_v2.py` parser choices 中加入该枚举，保留计划原始动作语义。修复后 `V56_B4_PROJECTION_COMMIT_FIX1_96F` 和 `V56_B4_PROJECTION_COMMIT_704F` 均完成。

## 语义结论

Track A 没有证明 semantic help:

- A1/A4 704F 使用 semantic residual cue 后 ATE `40.077571`，相对 H35 704F `+0.279324`，是回退。
- A2/A3 704F ATE `39.798248`，相对 H35 704F `0.000000`，只是持平。
- A2/A3 full ATE 都是 `35.74089695811434`，相对 H35 full `0.000000`，没有达到 `<=35.2409`。
- A2/A3 full runtime 也未过 28min gate: A2 `29.616667min`，A3 `28.216667min`。
- A2/A3 的 `context_source_skip_applied_count=0`、`affected_source_token_count_max=0.0`，说明 high-influence anomaly READ filtering / static rescue 在当前路径没有实际选中 source tokens；`semantic_label_count_mean=NA` 表示当前 runner 未落盘细粒度 semantic label count 证据。

Insight: Stage C cache 被正确读取并命中，但语义路径没有转化成有效 source filtering 或 ATE 改善。当前失败更像是 semantic action 未实际影响 HMC read/write 决策，而不是 Stage C cache 缺失。

## New TTT Action 结论

Track B 没有证明 new TTT action help。96F smoke 能跑通且有 debug 证据，但 704F 全部回退:

- B1 Binary Stable-Anchor Replay: 704F ATE `41.439231`，相对 H35 704F `+1.640983`。debug: `stable_anchor_token_mass_mean=0.459595`，`no_long_write_token_mass_mean=0.540405`。
- B2 Risk-Veto Commit: 704F ATE `41.727226`，相对 H35 704F `+1.928979`。debug: `risk_token_mass_mean=0.133327`，`no_long_write_token_mass_mean=0.133327`。
- B3 Two-Lifetime Commit: 704F ATE `40.867985`，相对 H35 704F `+1.069737`。debug: `dual_lifetime_long_override_count=20`。
- B4 Projection Commit: 704F ATE `41.233692`，相对 H35 704F `+1.435444`。debug: `native_delta_gate_scale_mean=0.35`，`native_delta_gate_cos_mean=0.736172`。

Insight: action mechanisms did trigger, but longer screen revealed instability/regression. This is not a smoke-only implementation failure; it is a 704F behavioral failure, so Track B full was correctly skipped.

## Combo / Cross-Sequence 判定

- Combo 未运行: 计划要求 Track A 和 Track B 都有 full ATE `<=35.2409`；实际 Track A best full 是 `35.740897`，Track B 无 full。
- Cross-sequence 未运行: 计划要求 KITTI01 full ATE `<=35.2409`；实际没有任何 v56 full 达到该门槛。

## 关键 Insight

1. H35 baseline 稳定，drift 为 `0.0`，所以本轮负结果不能归咎于基线复现失败。
2. Stage C cache short-tail blocker 是真实工程问题，修复后 cache artifact hit 证据完整；但 semantic ATE 没有改善。
3. A2/A3 的 semantic source-skip 相关计数为 0，说明计划中的语义过滤没有实际影响上下文源选择。
4. A1/A4 的 semantic residual cue 在 704F 上回退，提示该 cue 可能引入了无效或有害读偏置。
5. B1/B2/B3/B4 的动作 debug 均显示机制触发，但 704F 均显著回退，说明新动作不是只缺少实现，而是当前策略在中长窗口不稳。
6. v56 最佳 full 与 H35 完全相同，不是成功；把持平当成功会误导后续计划。

## 判定

- Track A 未证明 semantic full ATE 相对 H35 改善 >=0.5m。
- Track B 未证明 new TTT action full ATE 相对 H35 改善 >=0.5m。
- 本轮没有进入 combo 或 cross-sequence 的资格。

## 产物

- final report: `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/report_final/v56_final_report.md`
- registries: `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/report_final`
- figures: `results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/report_final/figures`
