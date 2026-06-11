# ACL2 v61 CleanSemanticResidualRead TTTWriting ScaleState 实验结果复盘

日期: 2026-06-11
计划文档: `docs/ACL2_v61_CleanSemanticResidualRead_TTTWriting_ScaleState_Plan.md`
执行日志: `docs/ACL2_v61_CleanSemanticResidualRead_TTTWriting_ScaleState_执行日志.md`
结果根目录: `results/kitti01_hmc_v2/acl2_v61_clean_semantic_residual_read_ttt_scale_state`

结论先行: 本文件只基于已落盘 artifact 生成；缺失字段保持 NA/unavailable。当前 full 只有 baseline repeat `A0_H35_CLEAN_REPEAT`，ATE `41.979343`。minimum progress gate `<= 35.2409`: `False`；semantic target `<= 33.7409`: `False`。当前 A0_FULL 相对 landed H35 full 漂移 `6.238446`，A0_704F 相对历史 H35_704 漂移 `5.545510`，因此不生成可报告 method result，不启动 semantic candidate full。

## Phase 0

- `h35_landed_reference_dir`: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v56_h35_semanticboost_newtttaction_fast/phase0_h35_repeat/rollouts/V56_PHASE0_H35_FULL_REPEAT`
- `h35_landed_reference_available`: `True`
- `h35_landed_reference_ate`: `35.74089695811434`
- `c9_reference_ate`: `33.76294210291885`
- `semantic_cache_dir`: `/mnt/data/users/chengshun.wang/pjs/LoGeR/results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full`
- `semantic_cache_dir_exists`: `True`
- `stage_c_smoke_hit_rate_min`: `1.0`
- `d_i_available_in_smoke`: `True`
- `ttt_residual_logging_available`: `False`
- `scale_metric_script_available`: `True`
- `scale_metric_rows_written`: `208`

## 96F/256F smoke

| row | candidate | track | frames | ATE | dH35_704 | dH35_full | read_act | ttt_act | scale_var | scale_rows | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_256F | A0_H35_CLEAN_REPEAT | baseline_or_geometry_control | 256 | 3.898257 | NA | NA | 0.000000 | NA | 0.003440 | 9 | False |
| A1_256F | A1_SEM_RESID_C23_READ | track_a_semantic_read | 256 | 3.943052 | NA | NA | 0.000000 | NA | 0.002743 | 9 | False |
| A2_256F | A2_SEM_CONDITIONED_DG_READ | track_a_semantic_read | 256 | 4.664563 | NA | NA | 0.000000 | NA | 0.004267 | 9 | False |
| A3_256F | A3_SEM_ANCHOR_RESCUE_READ | track_a_semantic_read | 256 | 3.908078 | NA | NA | 96.000000 | NA | 0.003401 | 9 | False |
| A4_256F | A4_SEM_TRANSIENT_RISK_READ | track_a_semantic_read | 256 | 3.883521 | NA | NA | 54.000000 | NA | 0.002844 | 9 | False |
| B0_256F | B0_H35_CLEAN_REPEAT | baseline_or_geometry_control | 256 | 3.898257 | NA | NA | 0.000000 | NA | 0.003440 | 9 | False |
| B1_256F | B1_SEM_ANCHOR_WRITE_FLOOR | track_b_semantic_ttt | 256 | 3.684943 | NA | NA | 0.000000 | 961.000000 | 0.003486 | 9 | False |
| B2_256F | B2_SEM_TRANSIENT_RISK_BOOST | track_b_semantic_ttt | 256 | 3.783711 | NA | NA | 0.000000 | NA | 0.003549 | 9 | False |
| B4_256F | B4_SEM_CONDITIONED_DG_TTT | track_b_semantic_ttt | 256 | 3.866625 | NA | NA | 0.000000 | NA | 0.002824 | 9 | False |
| NA1_256F | NA1_RANDOM_SAME_MASS_READ | track_a_negative_control | 256 | 3.896167 | NA | NA | 54.000000 | NA | 0.002869 | 9 | False |
| NA2_256F | NA2_SHUFFLED_SEMANTIC_READ | track_a_negative_control | 256 | 3.910930 | NA | NA | 96.000000 | NA | 0.003419 | 9 | False |
| NA3_256F | NA3_GEOMETRY_ONLY_RESIDUAL_READ | track_a_negative_control | 256 | 3.853954 | NA | NA | 54.000000 | NA | 0.002904 | 9 | False |
| NA4_256F | NA4_SEMANTIC_ONLY_READ | track_a_negative_control | 256 | 3.908078 | NA | NA | 96.000000 | NA | 0.003401 | 9 | False |
| NB1_256F | NB1_RANDOM_SAME_MASS_TTT | track_b_negative_control | 256 | 3.691956 | NA | NA | 0.000000 | 1122.000000 | 0.003485 | 9 | False |
| NB2_256F | NB2_SHUFFLED_SEMANTIC_TTT | track_b_negative_control | 256 | 3.721844 | NA | NA | 0.000000 | 10261.000000 | 0.003492 | 9 | False |
| NB3_256F | NB3_H35_CLEAN_REPEAT | baseline_or_geometry_control | 256 | 3.898257 | NA | NA | 0.000000 | NA | 0.003440 | 9 | True |
| A0_96F | A0_H35_CLEAN_REPEAT | baseline_or_geometry_control | 96 | 1.289640 | NA | NA | 0.000000 | NA | 0.001675 | 4 | True |
| A1_96F | A1_SEM_RESID_C23_READ | track_a_semantic_read | 96 | 0.977569 | NA | NA | 0.000000 | NA | 0.001146 | 4 | False |
| A2_96F | A2_SEM_CONDITIONED_DG_READ | track_a_semantic_read | 96 | 1.285054 | NA | NA | 0.000000 | NA | 0.001632 | 4 | False |
| A3_96F | A3_SEM_ANCHOR_RESCUE_READ | track_a_semantic_read | 96 | 1.288906 | NA | NA | 48.000000 | NA | 0.001681 | 4 | True |
| A4_96F | A4_SEM_TRANSIENT_RISK_READ | track_a_semantic_read | 96 | 1.128423 | NA | NA | 24.000000 | NA | 0.001025 | 4 | True |
| B0_96F | B0_H35_CLEAN_REPEAT | baseline_or_geometry_control | 96 | 1.289640 | NA | NA | 0.000000 | NA | 0.001675 | 4 | False |
| B1_96F | B1_SEM_ANCHOR_WRITE_FLOOR | track_b_semantic_ttt | 96 | 1.265343 | NA | NA | 0.000000 | 374.000000 | 0.001600 | 4 | True |
| B2_96F | B2_SEM_TRANSIENT_RISK_BOOST | track_b_semantic_ttt | 96 | 1.273697 | NA | NA | 0.000000 | NA | 0.001643 | 4 | False |
| B4_96F | B4_SEM_CONDITIONED_DG_TTT | track_b_semantic_ttt | 96 | 0.992916 | NA | NA | 0.000000 | NA | 0.001169 | 4 | False |
| NA1_96F | NA1_RANDOM_SAME_MASS_READ | track_a_negative_control | 96 | 1.064991 | NA | NA | 24.000000 | NA | 0.001087 | 4 | False |
| NA2_96F | NA2_SHUFFLED_SEMANTIC_READ | track_a_negative_control | 96 | 1.286487 | NA | NA | 48.000000 | NA | 0.001671 | 4 | True |
| NA3_96F | NA3_GEOMETRY_ONLY_RESIDUAL_READ | track_a_negative_control | 96 | 1.138140 | NA | NA | 24.000000 | NA | 0.001009 | 4 | False |
| NA4_96F | NA4_SEMANTIC_ONLY_READ | track_a_negative_control | 96 | 1.288906 | NA | NA | 48.000000 | NA | 0.001681 | 4 | True |
| NB1_96F | NB1_RANDOM_SAME_MASS_TTT | track_b_negative_control | 96 | 1.267023 | NA | NA | 0.000000 | 537.000000 | 0.001621 | 4 | False |
| NB2_96F | NB2_SHUFFLED_SEMANTIC_TTT | track_b_negative_control | 96 | 1.267555 | NA | NA | 0.000000 | 3611.000000 | 0.001644 | 4 | True |
| NB3_96F | NB3_H35_CLEAN_REPEAT | baseline_or_geometry_control | 96 | 1.289640 | NA | NA | 0.000000 | NA | 0.001675 | 4 | True |

## 704F screen

| row | candidate | track | frames | ATE | dH35_704 | dH35_full | read_act | ttt_act | scale_var | scale_rows | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_704F | A0_H35_CLEAN_REPEAT | baseline_or_geometry_control | 704 | 45.343758 | 5.545510 | NA | 0.000000 | NA | 0.020002 | 25 | NA |
| A1_704F | A1_SEM_RESID_C23_READ | track_a_semantic_read | 704 | 41.085636 | 1.287388 | NA | 0.000000 | NA | 0.019618 | 25 | NA |
| A3_704F | A3_SEM_ANCHOR_RESCUE_READ | track_a_semantic_read | 704 | 45.332443 | 5.534195 | NA | 120.000000 | NA | 0.019997 | 25 | NA |
| A4_704F | A4_SEM_TRANSIENT_RISK_READ | track_a_semantic_read | 704 | 40.859855 | 1.061607 | NA | 150.000000 | NA | 0.019406 | 25 | NA |
| B1_704F | B1_SEM_ANCHOR_WRITE_FLOOR | track_b_semantic_ttt | 704 | 45.216610 | 5.418363 | NA | 0.000000 | 1054.000000 | 0.020065 | 25 | NA |
| B2_704F | B2_SEM_TRANSIENT_RISK_BOOST | track_b_semantic_ttt | 704 | 45.197581 | 5.399334 | NA | 0.000000 | NA | 0.020099 | 25 | NA |
| B4_704F | B4_SEM_CONDITIONED_DG_TTT | track_b_semantic_ttt | 704 | 41.430838 | 1.632590 | NA | 0.000000 | NA | 0.019609 | 25 | NA |
| NA1_704F | NA1_RANDOM_SAME_MASS_READ | track_a_negative_control | 704 | 41.165665 | 1.367418 | NA | 150.000000 | NA | 0.019520 | 25 | NA |
| NA3_704F | NA3_GEOMETRY_ONLY_RESIDUAL_READ | track_a_negative_control | 704 | 40.909326 | 1.111079 | NA | 150.000000 | NA | 0.019392 | 25 | NA |
| NB1_704F | NB1_RANDOM_SAME_MASS_TTT | track_b_negative_control | 704 | 45.199655 | 5.401408 | NA | 0.000000 | 1214.000000 | 0.020105 | 25 | NA |
| NB2_704F | NB2_SHUFFLED_SEMANTIC_TTT | track_b_negative_control | 704 | 45.174002 | 5.375754 | NA | 0.000000 | 11058.000000 | 0.020001 | 25 | NA |

## Full runs

| row | candidate | track | frames | ATE | dH35_704 | dH35_full | read_act | ttt_act | scale_var | scale_rows | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_FULL | A0_H35_CLEAN_REPEAT | baseline_or_geometry_control | 1101 | 41.979343 | NA | 6.238446 | 0.000000 | NA | 0.023931 | 38 | NA |

## Scale Metric Notes

- `scale_variance_all` comes from actual per-chunk overlap point-map Sim(3) only when `per_chunk_geometry` exists.
- `scale_variance_sem` remains NA unless point-level semantic weights are saved; v61 does not backfill it with zeros.
- Historical H35 full artifact is used as landed ATE baseline; new v61 rows are reported separately.
- A0_FULL was rerun as a blocker repair after A0_704F failed to match the historical H35_704 reference.

## Phase Decisions

- Baseline stability: `False`; current A0_704F `45.343758` vs historical `39.798248`, delta `5.545510`.
- Current A0_FULL `41.979343` vs landed H35 full `35.740897`, delta `6.238446`.
- Track A 704F apparent gains against current A0_704F are not reportable because the current baseline drifted and NA1/NA3 controls show the same scale/ATE pattern.
- Track B 704F does not pass: B1/B2 are indistinguishable from NB1/NB2 controls, and B4 has no semantic TTT write action count and remains worse than historical H35_704.
- Phase 3 causal fork and Phase 4 semantic candidate full were not run by design because 704F promotion and baseline stability gates failed.

## 必答问题

1. semantic residual READ 是否有收益: `not reportable: no Track A row beats historical H35_704 by 0.50m, and best semantic 704F is matched by controls`。
2. semantic residual READ 是否影响 current chunk scale metrics: `scale metrics available; baseline-relative improvement must be judged from paired rows`。
3. semantic TTT writing 是否改变 write mass/post-zp: `partial: B1 writes changed, but post-zp is unavailable and B1/B2 do not beat matched controls`。
4. semantic TTT writing 是否影响 future scale residual: `not proven unless causal fork rows are present`。
5. semantic 是否优于 controls: `not proven`。
6. 主要收益来源: `no promoted source; apparent 704F gains come from READ-style source attenuation also reproduced by controls`。
7. 如果失败，主要瓶颈: `current H35 baseline drifted; controls explain READ gains; semantic TTT writing lacks causal improvement`。

## Insight 与证据链

1. Landed row count: `44`; smoke rows: `32`; 704F rows: `11`; full rows: `1`.
2. Best 704F row is `A4_SEM_TRANSIENT_RISK_READ` with ATE `40.859855` and delta vs H35_704 `1.061607`.
3. Best full row is `A0_H35_CLEAN_REPEAT` with ATE `41.979343` and delta vs H35 `6.238446`.
4. Scale diagnostics wrote rows for `44` runs; semantic point-weighted scale remains unavailable unless explicitly saved.
5. Smoke gate failures include `A0_H35_CLEAN_REPEAT, A1_SEM_RESID_C23_READ, A2_SEM_CONDITIONED_DG_READ, A3_SEM_ANCHOR_RESCUE_READ, A4_SEM_TRANSIENT_RISK_READ, B0_H35_CLEAN_REPEAT, B1_SEM_ANCHOR_WRITE_FLOOR, B2_SEM_TRANSIENT_RISK_BOOST`; inspect action counts, context empty events, runtime and scale availability before promotion.
6. Current A0_704F ATE `45.343758` is worse than historical H35_704 `39.798248` by `5.545510`.
7. Current A0_FULL ATE `41.979343` is worse than landed H35 full `35.740897` by `6.238446`; this blocks reportable method claims.
8. A4_704F is close to controls: A4 `40.859855`, NA1 `41.165665`, NA3 `40.909326`; the READ/scale improvement is therefore not semantic-specific.
9. B1/B2 704F do not beat TTT controls: B1 `45.216610`, B2 `45.197581`, NB1 `45.199655`, NB2 `45.174002`.
