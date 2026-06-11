# ACL2 v52 C9Clean AdaptiveTTT SemanticGeometry 实验复盘

日期：2026-06-09（Asia/Singapore）  
执行日志：`docs/ACL2_v52_C9Clean_AdaptiveTTT_SemanticGeometry_执行日志.md`  
计划文件：`docs/ACL2_v52_C9Clean_AdaptiveTTT_SemanticGeometry_Experiment_Plan.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/`

本复盘只记录真实落盘数据。没有完成的 run 不补指标；partial/missing_prediction 不作为方法结果。

## 当前状态

```text
Phase 0 code audit / bugfix pass。
Phase 1 C9 attribution pass。
Phase 2 adaptive failure autopsy pass。
Phase 3 smoke pass。
Phase 3 AW111 full B/C 完成，但全部 fail soft pass。
Phase 3 no-SWA diagnostic 完成，也 fail soft pass。
Phase 4/5 因 Phase 3 全部候选 fail soft pass，不启动。
v52 已收尾；当前没有继续 full run 的有效依据。
Runtime efficiency audit 已新增，后续 full run 必须先过效率 gate。
```

## 进度过慢的复盘

这轮慢，主要不是计算资源完全不足，而是执行策略前半段不够激进：

```text
1. full KITTI01 rollout 是 1101 frames，单 run 约几十分钟。
2. launcher 只能设置 V47_END_FRAME，不能按 start/end 切分一条 sequence 到多卡。
3. 第一批 full B/C 没有 DONE prediction，必须重跑 R2；这消耗了额外 wall time。
4. 为保证审计可信，partial/missing_prediction 没有被写入结果表。
5. 用户指出后，已把可并行的 B/C no-SWA diagnostic 同时放到 GPU2/GPU3；no-SWA 已完成并落盘。
```

GPU0-5 可用后，实际可加速的部分是“不同候选并行跑”，不是“单条 rollout 多卡拆分”。计划里 Phase 3 只允许 A/B/C 主候选和必要 no-SWA diagnostic；A 因 smoke 太慢降级，B/C full 已完成，D1/Phase4/Phase5 都需要 soft pass 才能启动。因此没有继续用 GPU4/GPU5 开无解释意义的额外网格。

追加效率审计：

```text
新增 tools/v52_runtime_profile_report.py。
run_pipeline_abc_v2.py 新增 --empty_cuda_cache_each_chunk。
tools/run_attention_cue_experiment.sh 透传 EMPTY_CUDA_CACHE_EACH_CHUNK。
tools/run_v47_adaptive_ttt_writer_candidate.sh 后续新启动 run 默认 V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0。

本地 R2 DONE timing:
  B1 AW111 R2 wall = 25.35min, chunk mean = 38.22s
  C1 AW111 R2 wall = 24.52min, chunk mean = 37.04s
  B1 no-SWA R1 wall = 26.17min, chunk mean = 39.48s
  C1 no-SWA R1 wall = 26.57min, chunk mean = 40.19s

如果调度环境观测到 61-62min wall time，则按计划 11.3b 直接视为效率 blocker。
无论采用本地 wall 还是外部观测，chunk mean 已经说明 full run 不能继续作为默认筛选。
```

## Phase 0 结论

修复并审计了两个会污染解释的问题：

```text
1. past_plus_future_light_real 在 qq/qk/kk 三分支统一为 weighted support。
2. tri replay debug 字段不再让 ttt_two_replay_applied 误导报告。
```

关键审计：

| Item | Result |
|---|---:|
| `py_compile` | pass |
| shell `bash -n` | pass |
| support alias pass | `True` |
| qq/qk/kk max diff vs manual weighted | `0.0 / 0.0 / 0.0` |
| tri replay debug audit | `True` |
| code packet missing/unpackaged | `0 / 0` |

结论：

```text
v52 后续实验建立在修复后的 support alias 和 debug audit 上。
Phase 0 不改变历史指标，但修复了解释链。
```

## Phase 1 结论：C9 的收益来源

clean no-chunk factorial 说明：

| Row | ATE | Gain vs F000 |
|---|---:|---:|
| `F000_NONE` | `41.750205342038015` | `0.0` |
| `F100_ONLY_FRAME_ATTN` | `38.593412804190564` | `3.1567925378474513` |
| `F010_ONLY_TTT` | `39.51910428415088` | `2.2311010578871375` |
| `F001_ONLY_SWA` | `41.73704103737674` | `0.013164304661273718` |
| `F110_FRAME_ATTN_TTT` | `36.668910841145234` | `5.081294500892781` |
| `F111_ALL_THREE` | `36.65074045045878` | `5.099464891579238` |

关键判断：

```text
READ 和 TTT 是主要正贡献。
SWA 单独贡献几乎为 0。
READ+TTT 有约 1.9245m 非加性增益。
```

exact C9 attribution：

| Component | delta vs C9 |
|---|---:|
| `tri_replay` | `2.4469526757787676` |
| `tri_gamma_chunk_map` | `0.971025405816512` |
| `commit_ema` | `0.48838546389362136` |
| `native_mix` | `0.091702749922284` |
| `swa_overlap_replace` | `0.056250245127017706` |
| `read_beta_map` | `0.0265805624180544` |

C9-Clean：

```text
Exact C9 F0 = 33.76294210291885
D7 C9-Clean = 35.500497135292775
Delta = +1.7375550323739262
```

结论：

```text
C9 的核心不是语义，也不是 SWA。
C9 优势主要来自 TTT tri replay、chunk gamma map、commit EMA 的交互。
去 chunk-id 后，fixed global substitute 无法复现 C9。
```

## Phase 2 结论：v50 为什么仍失败

| Run | ATE | delta vs C9 | delta vs soft 34.60 | probe TTT sec mean |
|---|---:|---:|---:|---:|
| `C9_exact_teacher` | `33.762943` | `0.000000897081150697` | `-0.837057` | `nan` |
| `V46B_fixed_F111_teacher` | `36.650736` | `+2.88779389708` | `+2.050736` | `62.7037344983` |
| `V50_split_resxdg_student` | `35.985305` | `+2.22236289708` | `+1.385305` | `5.78059238509` |

关键判断：

```text
v50 split 比 fixed F111 trace 好 0.665431m，但仍离 C9 远。
fixed/update-conflict teacher 太慢，不能直接作为 runtime。
失败更像 split replay 恢复后的 action geometry / energy mismatch，
不是单纯 role mass 有无的问题。
```

## Phase 3 smoke

| Run | Method | ATE | chunk sec mean | probe TTT sec mean | 结论 |
|---|---|---:|---:|---:|---|
| `A1 ConflictLite` | selected-layer conflict proxy | `1.315722` | `45.389658` | `14.290215` | 太慢，降级 diagnostic |
| `B1 EnergyMatched` | energy-matched adaptive split | `1.349152` | `40.185260` | `5.951500` | 进入 full |
| `C1 Cluster3D` | 3D feature cluster split | `1.366865` | `35.026832` | `5.108346` | 进入 full |

判断：

```text
ConflictLite 虽然有动机，但 probe TTT mean > 8s/chunk，
按计划 11.3 降级，不作为 full 主线。
```

## Phase 3 full AW111

| Run | ATE | delta vs C9 | delta vs soft 34.60 | delta vs fixed F111 | probe TTT sec mean | audit |
|---|---:|---:|---:|---:|---:|---|
| `B1 EnergyMatched AW111 R2` | `35.967687` | `+2.204745` | `+1.367687` | `-0.683053` | `4.980390` | `True/True` |
| `C1 Cluster3D AW111 R2` | `36.427703` | `+2.664761` | `+1.827703` | `-0.223037` | `5.029483` | `True/True` |

Gate：

```text
C9 reference = 33.76294210291885
soft pass = ATE <= 34.60
best v52 AW111 = 35.967687
success=False
```

解释：

```text
EnergyMatched 和 Cluster3D 都满足 no-chunk / no-manual-percentage audit，
并且比 fixed F111 trace 有改善。
但它们没有接近 C9，也没有达到 soft pass。
```

## no-SWA diagnostic

| Run | ATE | delta vs C9 | delta vs soft 34.60 | delta vs fixed TTT | wall min | chunk mean | probe TTT mean | audit |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `B1 EnergyMatched AW110 no-SWA R1` | `35.974144` | `+2.211202` | `+1.374144` | `-0.694766` | `26.17` | `39.476774` | `5.132336` | `True/True` |
| `C1 Cluster3D AW110 no-SWA R1` | `36.452058` | `+2.689115` | `+1.852058` | `-0.216853` | `26.57` | `40.192329` | `5.518366` | `True/True` |

判断：

```text
no-SWA 没有改变 Phase 3 结论。
B1 no-SWA 35.974144 与 B1 AW111 35.967687 基本同级；
C1 no-SWA 36.452058 与 C1 AW111 36.427703 基本同级。

因此失败不是 SWA 引起的，而是 adaptive split 本身没有恢复 C9 的 chunk-gamma / commit-EMA / tri-replay 效果。
```

## Phase 4/5 判断

计划要求：

```text
Phase 4 semantic minimal retest 只在 Phase 3 adaptive TTT hard/soft pass 后启动。
Phase 5 cross sequence sanity 也只在 adaptive baseline 达到 soft pass 后启动。
```

当前：

```text
best Phase 3 = 35.967687 > 34.60
因此 Phase 4/5 不启动。
```

## 总结

```text
v52 已回答 C9 机制问题：
  READ/TTT 是主贡献，SWA 很小；
  exact C9 的优势来自 tri replay + chunk gamma + commit EMA。

v52 也完成了 adaptive TTT v2 的两条可跑主线：
  EnergyMatched
  Cluster3D

但当前 no-chunk / no-manual-percentage adaptive TTT 仍不能复现 C9。
best Phase 3 ATE = 35.967687，距离 C9 仍 +2.204745m。
```

## 结论

```text
success=False

当前 adaptive TTT 公式没有达到 C9 close/soft pass。
继续盲扫 risk threshold、role threshold 或简单 gamma clamp 信心很低。
按计划 11.4，应写明：
  当前无 chunk / 无手工 percentage adaptive TTT 不能复现 C9。
  C9 的性能依赖 fixed percentage / chunk gamma / EMA 交互。
  下一步只能做 C9 teacher behavior distillation 或重新考虑 TTT write action space。
```

## Insight

```text
最关键的失败信号不是“adaptive 完全没用”，而是：
  v50、EnergyMatched、Cluster3D 都能比 fixed F111 改善，
  但都卡在 35.9m 左右，离 C9 的 33.76m 仍很远。

这说明 split replay 的动作空间方向是对的，
但当前自适应策略没有学到 C9 的 chunk gamma / EMA / action energy timing。

下一步不应再用单一 risk proxy 决定 pos/neu/neg，
而应转成 teacher-behavior distillation：
  1. 从 C9 trace 学 action delta norm / branch timing / commit behavior；
  2. 不使用 absolute chunk id；
  3. 学到的是 state-conditioned action rule，而不是固定 chunk table；
  4. 先在 KITTI01 复现 C9 action geometry，再谈语义几何闭环。
```

## 证据链索引

```text
Phase 0:
  results/.../phase0_code_audit/bugfix_report.md
  results/.../phase0_code_audit/support_alias_unit_audit_summary.{json,csv}
  results/.../phase0_code_audit/adaptive_writer_debug_field_audit.json
  results/.../phase0_code_audit/code_packet_completeness_audit.md

Phase 1:
  results/.../phase1_c9_attribution/c9_component_attribution_report.md
  results/.../phase1_c9_attribution/positive_only_factorial_table.csv
  results/.../phase1_c9_attribution/exact_c9_clean_rows.csv
  results/.../phase1_c9_attribution/exact_c9_knockout_table.csv

Phase 2:
  results/.../phase2_adaptive_failure_audit/adaptive_failure_autopsy.md
  results/.../phase2_adaptive_failure_audit/run_overview.csv
  results/.../phase2_adaptive_failure_audit/phase2_adaptive_failure_summary.json
  results/.../phase2_adaptive_failure_audit/post_zp_delta_ratio_by_chunk.csv

Phase 3:
  results/.../phase3_adaptive_ttt_v2_smoke/report_R1/v47_adaptive_ttt_writer_report.md
  results/.../phase3_adaptive_ttt_v2_full/report_R1/v47_adaptive_ttt_writer_report.md
  results/.../phase3_adaptive_ttt_v2_full/report_R2_noswa_done/v47_adaptive_ttt_writer_report.md
  results/.../phase3_adaptive_ttt_v2_full/rollouts/V52_FULL_B1_ENERGYMATCH_AW111_R2/
  results/.../phase3_adaptive_ttt_v2_full/rollouts/V52_FULL_C1_CLUSTER3D_AW111_R2/
  results/.../phase3_adaptive_ttt_v2_full/rollouts/V52_FULL_B1_ENERGYMATCH_AW110_NOSWA_R1/
  results/.../phase3_adaptive_ttt_v2_full/rollouts/V52_FULL_C1_CLUSTER3D_AW110_NOSWA_R1/

Runtime:
  results/.../runtime_efficiency_audit/v52_runtime_profile_summary.{json,csv,md}
```
