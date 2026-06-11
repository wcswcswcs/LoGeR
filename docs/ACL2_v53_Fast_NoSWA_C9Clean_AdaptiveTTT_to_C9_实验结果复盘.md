# ACL2 v53 Fast No-SWA C9-Clean Adaptive TTT to C9 实验结果复盘

日期：2026-06-09  
计划文件：`docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9`

## 不造假声明

本文只记录已经落盘或已有历史 artifact 中可追溯的数据。没有实际运行得到的数据不填数值；字段缺失时明确写缺失原因。

## 已确认参考数据

| 来源 | run/candidate | ATE | runtime / 备注 |
|---|---|---:|---|
| v53 plan | `C9_P0_R2` | `33.76294210291885` | C9 reference |
| v50 artifact | `V50_FULL_SPLIT_RESXDG_AW111` | `35.985306009701524` | probe TTT mean `5.398126s`，hmc rows `38` |
| v52 artifact | EnergyMatched AW111 R2 | `35.967687` | chunk mean `38.215938s`，probe TTT mean `4.980390s` |
| v52 artifact | EnergyMatched AW110 no-SWA | `35.974144` | wall `26.17min`，chunk mean `39.476774s`，probe TTT mean `5.132336s` |
| v52 artifact | ConflictLite 96F smoke | `1.315722` | chunk mean `45.389658s`，probe TTT mean `14.290215s`，runtime gate failed |

## Phase 1 复用证据

复用目录：

`results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit`

该目录包含：

- `run_overview.csv`
- `role_mass_timeline.csv`
- `post_zp_delta_ratio_by_chunk.csv`
- `post_zp_delta_by_chunk_layer_branch.csv`
- `adaptive_failure_autopsy.md`
- `teacher_student_role_mass_timeline.png`

已确认的 Phase 1 gap：

- v50/v52 adaptive split 已经优于 fused，但仍停在约 `35.97m` ATE 平台，距离 C9 约 `+2.20m`。
- v52 autopsy 显示 role assignment 不是唯一瓶颈；post-zp update energy、action/native cosine、gamma timing 和 commit behavior 仍与 C9 有显著差异。
- 因此 v53 修改方向是先补 `SC-GammaSplit` 和 `SC-GammaCommit`，而不是继续扫 fixed role percentage。

## 已做代码修改与审计意义

- `adaptive_writer_sc_gamma_split`：实现 no-percentage role split；塌缩时记录 collapse，不使用 fixed top percentage 修补。
- `adaptive_writer_sc_gamma_commit_split`：在 A 的基础上启用 state-conditioned commit filter，用 candidate/native update 距离自适应缩短 commit。
- `tools/run_v47_adaptive_ttt_writer_candidate.sh`：强制清空 chunk-id map，并落盘 no-chunk / manual-percentage audit 文件和复现命令。
- `tools/v53_experiment_report.py`：汇总 artifact，不参与模型运行；用于后续审计 registry、runtime、audit 和 failure routing。

## 初始审计结论

截至初始审计阶段，v53 代码入口已通过编译与 shell 语法检查；尚未产生 v53 smoke/runtime/full 的新 ATE。下一步必须先跑 96F smoke，不能直接 full。

## Phase 3 / 96F smoke 结果

数据来源：

- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R1_smoke/v53_candidate_registry.csv`
- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R1_smoke/v53_failure_routing_report.md`
- 各 run 的 `timing_summary.json`、`wall_time_summary.json`、`adaptive_ttt_audit.json`、`chunk_id_policy_audit.json`、`01.log`

| candidate | 96F ATE | chunk mean | probe TTT mean | audit | role collapse | 决策 |
|---|---:|---:|---:|---|---|---|
| A `SC-GammaSplit` | `1.3260134756601656` | `33.56557655334473` | `4.709445774555206` | no-chunk true / manual true | `0/72` | 进入 384F |
| B `SC-GammaCommit` | `1.2925985616830291` | `34.65052515268326` | `4.784034490585327` | no-chunk true / manual true | `0/72` | 进入 384F |
| C `ConflictLite-Split` | `1.3073629931206907` | `44.106241106987` | `13.946743845939636` | no-chunk true / manual true | SC-gamma 字段不适用 | 停止，不进 384F/full |

分析：

- A/B 的 smoke runtime 都低于 `chunk mean <= 42s` 和 `probe TTT <= 8s`，且 audit 通过，因此可以按计划进入 384F runtime projection。
- B 的 96F ATE 比 A 低约 `0.0334m`，但 96F 不用于 claim 方法有效，只用于筛掉明显失败候选。
- C 重复了 v52 ConflictLite 的主要问题：risk signal 可能有用，但当前实现的 TTT 写入耗时仍超过 gate。根据 v53 计划，不能用 full `update_conflict_energy` 代替，也不能送 full。

## Phase 3 / 384F runtime projection 结果

数据来源：

- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R2_screen/v53_candidate_registry.csv`
- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R2_screen_runtime/v52_runtime_profile_summary.md`

| candidate | 384F ATE_short | wall min | projected full min | chunk mean | probe TTT mean | audit | role collapse | 决策 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| A `SC-GammaSplit` | `33.813351532934135` | `9.783333333333333` | `26.554761904761907` | `39.52451760428293` | `5.157917380332947` | no-chunk true / manual true | `0.0` | 进入 full |
| B `SC-GammaCommit` | `33.96840141632595` | `9.0` | `24.428571428571427` | `36.32406502110617` | `4.731844697679792` | no-chunk true / manual true | `0.0` | 进入 full |

分析：

- A/B 均通过 384F runtime projection；没有触发 efficiency repair。
- A 的 384F ATE_short 优于 B，但 B 的 runtime 明显更低。384F 仍不是最终有效性结论，只用于允许 full。
- C 不进入 384F/full，符合 stop rule。

## Phase 4 / full 结果

数据来源：

- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_candidate_registry.csv`
- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_full_metrics_summary.md`
- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final_runtime/v52_runtime_profile_summary.md`

| full candidate | ATE | delta vs C9 | wall min | chunk mean | probe TTT mean | frames/hmc | audit | 判定 |
|---|---:|---:|---:|---:|---:|---|---|---|
| A `SC-GammaSplit` | `35.912858666859364` | `+2.149916563940515` | `26.65` | `40.57448160648346` | `5.432522616888347` | `1101/38` | pass | runtime pass, ATE fail |
| B `SC-GammaCommit` | `36.317006895477675` | `+2.554064792558826` | `27.35` | `41.662026831978245` | `5.646422605765493` | `1101/38` | pass | runtime pass, ATE fail |
| C `ConflictLite layer0 sample2048` | `36.08133075676421` | `+2.3183886538453606` | `24.283333333333335` | `37.00016983559257` | `5.0161768825430615` | `1101/38` | pass | runtime pass, ATE fail |

rolling window evidence:

| full candidate | rolling50 mean/p90/worst | rolling100 mean/p90/worst | rolling200 mean/p90/worst |
|---|---|---|---|
| A | `30.754175659031393 / 60.092918544733564 / 79.88573799028862` | `32.00159050856109 / 56.784130552238544 / 71.43510550708994` | `34.42390643451079 / 54.104031598262296 / 55.84389625519551` |
| B | `31.162076259992013 / 60.11454536637435 / 81.06931981229218` | `32.33900382123641 / 56.561473264691756 / 72.3434766398543` | `34.65120562374255 / 54.76036704443822 / 56.44890164096004` |
| C | `30.69178299078554 / 61.14824071532603 / 80.66612054868085` | `32.00375797929032 / 57.9171232248693 / 72.0656565894848` | `34.54191085177192 / 54.549914704732025 / 56.333479290314976` |

Full 结论：

- A 是本轮 best full：`35.912858666859364`。
- A/B/C 都过了 full runtime hard gate，但都没有达到 `ATE <= 35.30` progress-pass。
- B 的 commit filtering 在 log 中确认启用：`commit_filter_applied_true=38/38` chunks，`commit_w0_triggered_true=289/684` layer rows；但 full ATE 比 A 更差。
- C 的 sampled conflict 在 log 中确认启用：`sampled_true=38/38`，`sample_tokens_used=2048`；它显著改善 runtime，但 full ATE 仍没有带来 progress。

## ConflictLite blocker 修复记录

原始 C：

```text
V53_SMOKE_C_CONFLICTLITE_AW110_96F
chunk_total_seconds_mean = 44.106241106987
probe_ttt_write_seconds_mean = 13.946743845939636
decision = runtime fail
```

修复 1：selected layers 从 `{0,8,17}` 减到单层。

| repair | 96F ATE | projected full min | chunk mean | probe TTT mean | 384F 结果 |
|---|---:|---:|---:|---:|---|
| layer0 | `1.3402113070904205` | `27.075` | `37.37825036048889` | `7.972570598125458` | 384F runtime fail：chunk `42.322086232049124`，TTT `8.67749570097242` |
| layer17 | `1.3044604606864516` | `25.333333333333332` | `34.44476252794266` | `6.98481285572052` | 384F TTT fail：TTT `8.066814933504377` |
| layer8 | `1.34019069300634` | `27.55` | `38.08236104249954` | `8.26403135061264` | 96F probe TTT fail，停止 |

修复 2：sampled-token conflict proxy，selected layer 采样 2048 tokens。

| repair | 96F ATE | 96F TTT | 384F ATE | 384F projected full min | 384F chunk mean | 384F TTT | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| layer0 sample2048 | `1.3395832605016933` | `5.507285892963409` | `33.92511490882306` | `27.45952380952381` | `40.93911133493696` | `5.553112268447876` | 进入唯一 C full |
| layer17 sample2048 | `1.341075549605423` | `4.690415143966675` | `34.31631528249291` | `25.65` | `38.2044951234545` | `5.075153061321804` | 384F ATE 较差，不 full |

修复结论：

- selected-layer 单独修复还不够，384F 会重新触发 runtime fail。
- sampled-token proxy 确实解决了 ConflictLite runtime blocker。
- 但是 sampled C 的 full ATE 仍 fail，说明 conflict-like proxy 的短窗口信号没有转化为 full trajectory gain。

## 最终分析

1. no-chunk / no-manual-percentage adaptive TTT candidate 已产生，且所有 full audit 通过。

2. runtime 目标达成：
   - A wall `26.65min`
   - B wall `27.35min`
   - C wall `24.283333333333335min`
   都小于 28min。

3. 性能目标失败：
   - best full ATE 是 A 的 `35.912858666859364`
   - 距 C9 `33.76294210291885` 仍差 `+2.149916563940515m`
   - 三条 full 都大于 `35.30m` progress-pass 线。

4. commit filtering 不是当前主瓶颈：
   - B 的 commit filter 确实触发了大量 rows，但 ATE 从 A 的 `35.912858666859364` 退到 `36.317006895477675`。
   - 这说明当前 state-conditioned commit 距离规则没有恢复 C9 的 commit behavior，甚至可能削弱了有用 update。

5. ConflictLite 的问题被拆成两个部分：
   - runtime 可以通过 selected-layer + sampled-token 修复；
   - 但 full ATE 仍 fail，说明当前 conflict proxy 或它进入 role split 的方式没有抓到 C9 的真正长期写入节奏。

6. 384F 短窗口不能代表 full：
   - C layer0 sample2048 384F ATE `33.92511490882306`，看起来接近 C9；
   - full 却是 `36.08133075676421`。
   - 后续不能用 384F short ATE claim 方法有效，只能作为 runtime/catastrophic filter。

## 结论

```text
success_runtime = True
success_clean_adaptive_candidate = True
success_progress_pass = False
best_full = V53_FULL_A_SCGAMMASPLIT_AW110
best_full_ATE = 35.912858666859364
delta_vs_C9 = +2.149916563940515
```

本轮没有产出 C9Clean-AdaptiveTTT-v1 soft/progress baseline。按 v53 stop rule，算法尝试停止，进入 failure report。

下一步不应继续扫 role threshold 或小幅 gamma clamp。更合理的方向是：

- 重新审计 C9 是否依赖 branch/layer-level action，而不是 token role split。
- 查 C9 的 post-zp action/native cosine 与 current adaptive 的长期下游 drift。
- 重新设计 state-conditioned commit，不只用 candidate/native 距离触发 alpha。
- 若继续 ConflictLite，应研究为什么 384F short gain 没有转成 full gain，而不是继续加更多 selected layers。

## Insight

本轮最重要的 insight 是：

```text
runtime blocker 可以修，但 action space blocker 还在。
```

SC-GammaSplit、SC-GammaCommit、sampled ConflictLite 都能在 28 分钟内完成 full KITTI01，并且满足 no-chunk/no-manual-percentage 审计；这说明工程边界已经干净了。

但 full ATE 仍然卡在 `35.9--36.3m`，接近 v50/v52 平台，说明当前 adaptive writer 仍主要是在 token role 和轻量 gamma 上做局部修正，没有学到 C9 的真正长期写入行为。C9 的优势更可能来自 layer/branch-level action timing、commit/native interaction 或 post-zp update geometry，而不是单纯的 token-level positive/negative split。

## 证据链索引

```text
Final report:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_final_report.md
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_candidate_registry.csv
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_full_metrics_summary.md
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_failure_routing_report.md
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_phase0_efficiency_audit.md
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_no_chunk_policy_audit.json
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final/v53_manual_percentage_audit.json

Full rollouts:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase4_full/rollouts/V53_FULL_A_SCGAMMASPLIT_AW110
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase4_full/rollouts/V53_FULL_B_SCGAMMACOMMIT_AW110
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase4_full/rollouts/V53_FULL_C_CONFLICTLITE_LAYER0_SAMPLE2048_AW110

Runtime report:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final_runtime/v52_runtime_profile_summary.md

Teacher/student autopsy reused:
  results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/adaptive_failure_autopsy.md
  results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/post_zp_delta_ratio_by_chunk.csv
  results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/role_mass_timeline.csv

Code:
  loger/pipeline/ttt_write_controller.py
  run_pipeline_abc_v2.py
  tools/run_v47_adaptive_ttt_writer_candidate.sh
  tools/v53_experiment_report.py
```

## 2026-06-09 continuation 复盘：GPU 2/3/4/5 clean 优化

本节记录在用户提醒“GPU 2/3/4/5 空闲，请充分利用”之后继续按 v53 计划执行的真实结果。所有数值来自 R8--R22 报告和对应 rollout artifacts；未跑出的结论不补写。

## 修改审计补充

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `loger/pipeline/ttt_write_controller.py` | 为 v19 scale-state projection risk 新增 `scale_state_sample_tokens`；支持 evenly-spaced token sampling；补充 `v19_scale_state_sample_tokens_requested/sampled/used` debug 字段 | H3 full-token scale-state smoke 运行过慢，按计划 blocker 修复方向尝试 sampled-token approximation，而不是直接放弃 scale-state 方向 |
| `loger/pipeline/hybrid_memory_controller.py` | 新增 `ttt_write_scale_state_sample_tokens` pass-through | 让 pipeline controller 可以显式配置 scale-state sampling，避免只在局部对象手工改参数 |
| `run_pipeline_abc_v2.py` | 新增 CLI `--ttt_write_scale_state_sample_tokens`，并在 v19 scale-state controller summary 中记录 `sample_tokens` | 复现实验需要命令级可审计配置 |
| `tools/run_attention_cue_experiment.sh` | 透传 `TTT_WRITE_SCALE_STATE_SAMPLE_TOKENS` 到 pipeline | 让候选脚本可统一运行 sampled scale-state |
| `tools/run_v47_adaptive_ttt_writer_candidate.sh` | 新增 `V47_TTT_SCALE_STATE_SAMPLE_TOKENS`；写入 effective config、`adaptive_ttt_audit.json`、`chunk_id_policy_audit.json`、`reproduce_command.sh` | 保证 scale-state sampling 不是隐藏实验条件，后续可复现和审计 |

验证：

```text
py_compile:
  loger/pipeline/ttt_write_controller.py
  loger/pipeline/hybrid_memory_controller.py
  run_pipeline_abc_v2.py
  tools/v53_experiment_report.py
  pass

bash -n:
  tools/run_attention_cue_experiment.sh
  tools/run_v47_adaptive_ttt_writer_candidate.sh
  pass
```

## Continuation 候选筛选结果

### 96F smoke

| Run | ATE | wall min | projected full min | chunk mean | probe TTT mean | gate |
|---|---:|---:|---:|---:|---:|---|
| H1 branch-all | `1.3260134756601656` | `2.5833333333333335` | `24.541666666666668` | `33.12454253435135` | `4.520475924015045` | pass |
| H2 native_mix0.75 | `1.3019107729155959` | `2.6333333333333333` | `25.016666666666666` | `33.788400053977966` | `5.039144814014435` | pass |
| H3 scale-state full-token | `1.234120946980877` | `4.5` | `42.75` | `61.89534771442413` | `34.31854671239853` | runtime fail |
| H3 sampled2048 repair | `1.238825345252657` | `3.033333333333333` | `28.816666666666666` | `39.66757506132126` | `7.1920828223228455` | pass |
| H4 native_mix0.50 | `1.2950077303214693` | `2.816666666666667` | `26.758333333333333` | `36.81181448698044` | `5.747905254364014` | pass |
| H5 native_mix0.90 | `1.3133437044947083` | `2.85` | `27.075` | `36.88674467802048` | `5.681598246097565` | pass |
| H6 native_mix0.25 | `1.2843283665192422` | `2.966666666666667` | `28.183333333333334` | `38.74092394113541` | `5.894661843776703` | pass |
| H7 alpha1 sample2048 | `1.3046822937448517` | `2.75` | `26.125` | `35.80425089597702` | `6.269121468067169` | pass |
| H8 alpha3 sample2048 | `1.2380288828304868` | `2.7333333333333334` | `25.966666666666665` | `35.7044934630394` | `6.1488922238349915` | pass |
| H9 alpha2 sample1024 | `1.2388245468910044` | `2.6` | `24.7` | `33.770372092723846` | `5.0376317501068115` | pass |
| H10 alpha4 sample2048 | `1.2349877829102671` | `2.65` | `25.175` | `34.14753848314285` | `5.722457766532898` | pass |
| H11 alpha5 sample2048 | `1.2381165931580862` | `2.6666666666666665` | `25.333333333333332` | `34.60253643989563` | `5.940308630466461` | pass |
| H12 online pose-EMA | `1.2349877829102671` | `3.0166666666666666` | `28.658333333333335` | `39.564329385757446` | `6.647333443164825` | pass |
| H13 online overlap | `1.2349877829102671` | `2.8833333333333333` | `27.391666666666666` | `37.68193966150284` | `6.283405065536499` | pass |

修复判断：

```text
H3 full-token scale-state:
  probe TTT mean = 34.31854671239853s
  projected full = 42.75min
  runtime fail

H3 sampled2048:
  probe TTT mean = 7.1920828223228455s
  projected full = 28.816666666666666min
  runtime pass
  debug 记录 sample requested/sampled/used，sampling 条件可审计。

结论：
  scale-state runtime blocker 被 sampled-token repair 解决；
  但这只是 runtime repair，不等价于 full ATE 成功。
```

### 384F screen

| Run | ATE | delta vs C9 | wall min | projected full min | chunk mean | probe TTT mean | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| H2 native_mix0.75 | `34.17316081147195` | `0.41021870855310283` | `9.533333333333333` | `25.876190476190473` | `38.512014167649404` | `5.296329004423959` | full |
| H3 alpha2 sample2048 | `33.8902169923413` | `0.12727488942245202` | `9.4` | `25.514285714285716` | `37.78704789706639` | `5.885575481823513` | full |
| H4 native_mix0.50 | `34.63753779356436` | `0.8745956906455135` | `9.283333333333333` | `25.19761904761905` | `37.360947711127146` | `5.0753229175295145` | stop |
| H6 native_mix0.25 | `35.17743595421384` | `1.414493851294992` | `9.35` | `25.378571428571426` | `37.707017404692515` | `5.1447978019714355` | stop |
| H8 alpha3 sample2048 | `33.77653108321121` | `0.013588980292361441` | `9.183333333333334` | `24.926190476190474` | `37.02048448153904` | `5.635223065103803` | full |
| H9 alpha2 sample1024 | `33.91375953077707` | `0.15081742785822172` | `9.25` | `25.107142857142858` | `37.37098090989249` | `5.221473370279584` | stop |
| H10 alpha4 sample2048 | `33.82920983193383` | `0.06626772901498157` | `9.366666666666667` | `25.423809523809528` | `37.80639869826181` | `5.886151092393058` | stop |
| H13 alpha4 online overlap | `33.82920983193383` | `0.06626772901498157` | `9.316666666666666` | `25.28809523809524` | `37.61891397408077` | `5.871012875011989` | full as repair test |
| H14 alpha3 online overlap | `33.77653108321121` | `0.013588980292361441` | `10.183333333333334` | `27.640476190476193` | `41.02354213169643` | `6.220972163336618` | stop, same ATE as H8 but slower |

384F 判断：

```text
H8/H14 的 384F ATE = 33.77653108321121，几乎等于 C9。
H10/H13 的 384F ATE = 33.82920983193383，也有明显短窗正信号。
因此 sampled scale-state alpha3/alpha4 有短窗作用。
但 384F 只能作为 catastrophic/runtime filter，不能作为 clean progress-pass 证据。
```

## Full continuation 结果

| Run | status | frames | hmc rows | ATE | delta vs C9 | wall min | chunk mean | probe TTT mean | no-chunk/manual audit | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `V53_CONT_FULL_H2_NATIVEMIX075_SCGAMMA_AW110` | done | 1101 | 38 | `36.27180630457207` | `2.5088642016532177` | `26.366666666666667` | `40.14183219483024` | `5.4306485339214925` | pass | runtime pass, ATE fail |
| `V53_CONT_FULL_H3_SCALESTATE_OVERLAP_SAMPLE2048_AW110` | done | 1101 | 38 | `36.481771395468634` | `2.718829292549785` | `27.033333333333335` | `41.15486602406753` | `6.266185089161522` | pass | runtime pass, ATE fail |
| `V53_CONT_FULL_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110` | done | 1101 | 38 | `36.30791120013488` | `2.5449690972160326` | `27.3` | `41.552999684685155` | `6.262588739395142` | pass | runtime pass, ATE fail |
| `V53_CONT_FULL_H13_SCALESTATE_ALPHA4_ONLINESCALE_OVERLAP_AW110` | done | 1101 | 38 | `36.346746515095305` | `2.5838044121764554` | `26.333333333333332` | `40.04427994552412` | `6.1080278471896525` | pass | runtime pass, ATE fail |

与既有 best 对比：

```text
C9 reference ATE:
  33.76294210291885

progress-pass threshold:
  35.30

best continuation full:
  V53_CONT_FULL_H2_NATIVEMIX075_SCGAMMA_AW110
  ATE = 36.27180630457207
  delta vs C9 = +2.5088642016532177

best overall v53 full remains:
  V53_FULL_A_SCGAMMASPLIT_AW110
  ATE = 35.912858666859364
  delta vs C9 = +2.149916563940515

continuation did not improve the previous best full result.
all continuation full runs failed the 35.30 progress-pass threshold.
```

## Continuation 分析

1. runtime blocker 确实被修复：
   - full-token scale-state H3 smoke 的 projected full 是 `42.75min`；
   - sampled2048 repair 把 projected full 降到 `28.816666666666666min`；
   - 后续 H3/H8/H13 full 都在约 `26.33--27.30min` 完成。

2. native-mix 有短窗收益，但 full 不成立：
   - H2 384F ATE 是 `34.17316081147195`；
   - H2 full ATE 是 `36.27180630457207`；
   - 它是 continuation full best，但仍差于既有 A 的 `35.912858666859364`。

3. scale-state alpha3/alpha4 的短窗信号最强，但不是 full 解法：
   - H8/H14 384F ATE `33.77653108321121`，只比 C9 差 `0.013588980292361441`；
   - H8 full 回落到 `36.30791120013488`。

4. online-scale overlap 没有把短窗信号转成 full 信号：
   - H13 384F 与 H10 相同，ATE `33.82920983193383`；
   - H13 full ATE `36.346746515095305`，仍 fail。

5. 384F screen 的用途需要降级：
   - 本轮最强 384F 不但没有转成 full success，甚至 full 低于既有 best；
   - 因此后续不能用 384F 接近 C9 作为 progress-pass claim，只能用来排除 runtime/catastrophic 失败。

6. 当前失败更像 full-sequence 长期 drift：
   - 短窗 ATE 可接近 C9；
   - full 从 384F 之后明显回落；
   - 单纯继续扫 alpha/native_mix/online scale clamp 预计收益很低。

## Continuation 结论

```text
success_runtime = True
success_scale_state_runtime_repair = True
success_progress_pass = False
best_continuation_full = V53_CONT_FULL_H2_NATIVEMIX075_SCGAMMA_AW110
best_continuation_full_ATE = 36.27180630457207
best_overall_v53_full = V53_FULL_A_SCGAMMASPLIT_AW110
best_overall_v53_full_ATE = 35.912858666859364
requires_new_plan = True
```

这轮 continuation 按计划充分利用 GPU 2/3/4/5 跑了 native mix、scale-state sampling、alpha sweep、online-scale overlap 和 4 条 full。结果没有产出 clean progress-pass，也没有刷新 v53 既有 best full。

按 v53 计划的 stop rule，继续做小幅 alpha/native_mix/scale clamp sweep 已经不合理。下一步需要新的 full-sequence failure autopsy 计划，重点不是再扩短窗参数，而是解释为什么 384F 接近 C9 后 full 失效。

## Continuation insight

```text
短窗已经不是主要瓶颈；full-sequence drift 才是。
```

sampled scale-state 证明可以在 runtime 内近似 full-token scale-state risk；H8/H14 证明这个 risk 在前 384 帧可以几乎贴近 C9。但是 full 结果全部回到 `36.27--36.48m`，说明当前 controller 缺少跨 chunk 的长期状态约束，或者 C9 的优势来自更具体的 branch/layer/action timing，而不是当前 token-level risk/gamma/native-mix 组合。

后续应优先做：

- 对比 full 的前 384 帧与 384 帧后的 chunk drift。
- 审计 scale/alignment drift、branch/layer action、commit/native interaction timeline。
- 分析 H8/H10/H13 在 384F 内接近 C9 的具体 chunk 行为，定位失效从哪个 chunk 开始。
- 基于 full trajectory state 设计新的 state-conditioned controller，而不是继续局部扫 alpha/native_mix。

## Continuation 证据链索引

```text
Final continuation report:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_final_continuation_with_online_scale/v53_final_report.md
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_final_continuation_with_online_scale/v53_candidate_registry.csv
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_final_continuation_with_online_scale/v53_full_metrics_summary.md
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_final_continuation_with_online_scale/v53_no_chunk_policy_audit.json
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_final_continuation_with_online_scale/v53_manual_percentage_audit.json

Runtime reports:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_full_runtime/v52_runtime_profile_summary.md
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_online_scale_screen_runtime/v52_runtime_profile_summary.md
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R16_scale_repair_runtime/v52_runtime_profile_summary.md

Full continuation rollouts:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_full/rollouts/V53_CONT_FULL_H2_NATIVEMIX075_SCGAMMA_AW110
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_full/rollouts/V53_CONT_FULL_H3_SCALESTATE_OVERLAP_SAMPLE2048_AW110
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_full/rollouts/V53_CONT_FULL_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_full/rollouts/V53_CONT_FULL_H13_SCALESTATE_ALPHA4_ONLINESCALE_OVERLAP_AW110

Screen/smoke evidence:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_screen/rollouts/V53_CONT_SCREEN_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110_384F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_screen/rollouts/V53_CONT_SCREEN_H10_SCALESTATE_ALPHA4_SAMPLE2048_AW110_384F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_online_scale_screen/rollouts/V53_CONT_SCREEN_H13_SCALESTATE_ALPHA4_ONLINESCALE_OVERLAP_AW110_384F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_online_scale_screen/rollouts/V53_CONT_SCREEN_H14_SCALESTATE_ALPHA3_ONLINESCALE_OVERLAP_AW110_384F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase5_clean_continuation_scale_repair/rollouts/V53_CONT_REPAIR_H3_SCALESTATE_OVERLAP_SAMPLE2048_AW110_96F

Code:
  loger/pipeline/ttt_write_controller.py
  loger/pipeline/hybrid_memory_controller.py
  run_pipeline_abc_v2.py
  tools/run_attention_cue_experiment.sh
  tools/run_v47_adaptive_ttt_writer_candidate.sh
```

## 2026-06-09 Phase6 复盘：full-sequence autopsy 与 704F commit/branch/layer screen

本节只记录真实落盘结果。没有 full promotion 的 run 不写作 full 结果；commit filter 的 activation 没有落盘 debug，因此不 claim activation rate。

### 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `tools/v53_full_sequence_drift_autopsy.py` | 新增 full-sequence drift autopsy 工具；输出 by-run/by-chunk/segments/rolling/prefix-diff CSV、JSON、MD 和 timeline png | 按 v53 §9.6，解释 384F 接近 C9 但 full 失败的长期 drift；工具只读已落盘 artifacts，不生成或补造实验数据 |

### R1 autopsy 结论

```text
H8 screen/full prefix consistency:
  common frames = 384
  translation RMSE = 0.006200570735239549
  translation max = 0.08722549334340368
  rotation fro mean = 0.0001275102135586111
```

解释：

```text
H8 384F screen 与 H8 full 的前 384 帧几乎一致。
因此 H8/H14 的 384F 接近 C9 不是 screen/full prefix artifact。
失败主要发生在 384 帧之后。
```

分段证据：

| run | full ATE | prefix384 self ATE | seg0 rmse | seg1 rmse | seg2 rmse |
|---|---:|---:|---:|---:|---:|
| `C9_REF` | `33.76294210291885` | `33.37157053761927` | `46.39902061769452` | `34.656626116071514` | `11.034547189236443` |
| `V53_A_BEST` | `35.912858666859364` | `33.83297324070255` | `45.04645244540594` | `40.85251832537275` | `16.81711362434614` |
| `V53_H8_FULL` | `36.30791120013488` | `33.80040449316786` | `45.55351876783694` | `41.24700757795206` | `17.07735967628986` |
| `V53_H13_FULL` | `36.346746515095305` | `33.85379625939145` | `45.84565695969239` | `40.97596351054843` | `17.06974111460667` |

### H15-H18 inverse-scale 704F

| run | ATE | wall min | projected full min | chunk mean | TTT mean | 判定 |
|---|---:|---:|---:|---:|---:|---|
| H15 alpha3 inverse | `40.3156963118558` | `17.483333333333334` | `26.574666666666666` | `40.284730968475344` | `6.245894966125488` | 不升级 |
| H16 alpha4 inverse | `40.39964760231925` | `18.133333333333333` | `27.56266666666667` | `41.64413722038269` | `6.302977046966553` | 不升级 |
| H17 alpha3 inverse up-only | `40.3156963118558` | `17.85` | `27.132` | `41.03594367980957` | `6.161382150650025` | 不升级 |
| H18 residual inverse | `39.96013035108` | `16.55` | `25.156` | `37.95244373321533` | `5.013090181350708` | 不升级 |

对照：

```text
A-best 704F self ATE = 39.92846746159833
H18 ATE = 39.96013035108
```

结论：

```text
inverse-scale 方向没有超过 A-best 704F，不升级 full。
```

### H19-H26 commit/branch/layer 704F

| run | ATE | wall min | projected full min | chunk mean | TTT mean | no-chunk | manual % |
|---|---:|---:|---:|---:|---:|---|---|
| H19 statecommit075 | `40.185368766362735` | `16.716666666666665` | `25.409333333333333` | `38.33181634902954` | `5.228257350921631` | pass | pass |
| H20 olddecay q90 | `40.17659973494846` | `17.95` | `27.284` | `41.30676031112671` | `6.984069814682007` | pass | pass |
| H21 native2cand q90 | `40.158801504115694` | `18.916666666666668` | `28.753333333333334` | `43.4792165851593` | `6.960243282318115` | pass | pass |
| H22 layergamma 0/8/17 | `39.96013035108` | `16.083333333333332` | `24.446666666666665` | `36.96695454597473` | `4.7673742389678955` | pass | pass |
| H23 branchall | `45.219899861528525` | `16.3` | `24.776` | `37.418483638763426` | `5.160757207870484` | pass | pass |
| H24 branchall statecommit075 | `44.73271874335227` | `16.15` | `24.548` | `37.05656575202942` | `5.3126442432403564` | pass | pass |
| H25 branchall olddecay q90 | `43.34215610485726` | `18.283333333333335` | `27.790666666666667` | `42.046777153015135` | `7.290578470230103` | pass | pass |
| H26 branchall native2cand q90 | `44.38401578166757` | `17.05` | `25.916` | `39.154375534057614` | `6.925587244033814` | pass | pass |

分段证据：

| run | seg0 rmse | seg1 rmse | seg2 rmse |
|---|---:|---:|---:|
| C9_REF | `46.39902061769452` | `34.656626116071514` | `11.034547189236443` |
| V53_A_BEST | `45.04645244540594` | `40.85251832537275` | `16.81711362434614` |
| H22 | `46.17639098579641` | `30.899712092537136` | `30.223477905413752` |
| H23 branchall | `52.17797929050439` | `35.08404124288639` | `35.88080647300726` |
| H25 branchall olddecay | `50.264211654270225` | `33.11354573466735` | `38.161421805556` |

审计限制：

```text
H19/H20/H21/H24/H25/H26 的 commit_filter 配置已写入 effective_config.yaml 和 hmc_config.yaml。
但落盘 summary 没有 commit_filter debug 字段；R24 registry 的 commit_filter_debug_seen=0。
因此本阶段不能 claim commit_filter activation rate 或 per-layer commit-filter 生效细节。
这几条只能作为“配置下的轨迹结果”，并记录 commit-filter observability audit gap。
```

### Phase6 分析

1. 384F 接近 C9 的信号是真实前缀信号，但不能代表 full success。  
   R1 显示 H8 screen/full 前 384 帧 translation RMSE 只有 `0.006200570735239549`，因此 prefix 一致；full 失败来自后段。

2. inverse-scale 没有解决 704F 对照。  
   H18 是 inverse-scale best，ATE `39.96013035108`，仍差于 A-best 704F self ATE `39.92846746159833`。

3. selected-layer gamma 轻微接近但不够。  
   H22 是 H19-H26 best，ATE `39.96013035108`，与 H18 相同，仍不超过 A-best 704F；不应升级 full。

4. branch-all 明显伤害前缀/早期行为。  
   H23-H26 ATE 范围 `43.34215610485726--45.219899861528525`，seg0 rmse 升到约 `50--52`，远差于 A-best seg0 `45.04645244540594`。

5. commit-filter 方向存在审计 blocker。  
   配置被写入，但 summary 没有 activation debug；继续用 commit filter 做结论前，需要先修复/增强可观测性。

### Phase6 结论

```text
success_progress_pass = False
best_overall_v53_full = V53_FULL_A_SCGAMMASPLIT_AW110
best_overall_v53_full_ATE = 35.912858666859364
best_phase6_704F = H22_LAYERGAMMA_0_8_17 or H18_RESIDUAL_INVSCALE
best_phase6_704F_ATE = 39.96013035108
A_best_704F_self_ATE = 39.92846746159833
full_promotion = False
```

下一步不应再继续 branch-all 或 inverse-scale 小扫。更合理的方向：

```text
1. 修复 commit-filter 可观测性，确认 activation/debug 是否应该落盘。
2. 若 commit filter 确认未生效，再修复 activation path 后重测一个最小 704F。
3. 若 commit filter 已生效但只是不落盘，则补报告 parser/summary 字段，再基于真实 activation 做判断。
4. 设计显式 trajectory-state controller，目标是 384F 之后的 drift，而不是继续优化前 384F。
```

### Phase6 insight

```text
当前瓶颈不是“能不能在前 384 帧接近 C9”，而是“怎样在 384 帧后保持 C9 的长期状态行为”。
```

branch-all 的负例很强：扩大 branch action 并没有修复长期 drift，反而破坏 prefix。selected-layer/global layer action 也只让 704F 接近但没有超过 A-best。下一步应从长期状态控制和 commit-filter 审计入手，而不是继续扩同类参数。

### Phase6 证据链

```text
Code:
  tools/v53_full_sequence_drift_autopsy.py

Reports:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R23_phase6_inverse_scale_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R24_phase6_commit_layer_screen

Autopsy:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R1
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R2_inverse_scale_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R3_commit_layer_screen

Rollouts:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_inverse_scale_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_commit_layer_screen/rollouts
```

## 2026-06-09 Phase7 复盘：commit-filter 可观测性修复与 sc-gamma layergamma 链路修复

本节只记录真实落盘结果。Phase7 产生了一个新的 v53 full best，但仍未达到 progress-pass。

### Phase7 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `run_pipeline_abc_v2.py` | 新增 `path_length_ema` / `path_length_prev` trajectory scale-state proxy；在 hybrid debug summary 中落盘 commit-filter top-level 和 per-layer 字段 | 按 Phase6 blocker，修复 commit-filter 不可观测问题，并尝试用无 GT 轨迹状态控制 384F 后 drift |
| `tools/v53_experiment_report.py` | 增强 `commit_filter_stats`，统计 active mode seen、debug seen、applied rows、risk/scale/mode | 让报告能判断 commit-filter 是否真的进入 debug，而不是只看 config |
| `loger/pipeline/ttt_write_controller.py` | 修复 `adaptive_writer_sc_gamma_split` 中 `V47_TTT_LAYER_GAMMAS` 没有进入 sc-gamma `rho` 的问题；新增 configured rho / rho source debug | Phase6 H22/H30 的 layergamma config 写入了配置但没有改变轨迹；这是实际代码链路 bug |

验证：

```text
py_compile:
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile run_pipeline_abc_v2.py tools/v53_experiment_report.py loger/pipeline/ttt_write_controller.py
  pass
```

### Phase6 解释修正

Phase7 审计发现：

```text
H30 01.txt 与旧 H19 01.txt 完全逐行相同。
H22 01.txt 与 H18 01.txt 完全逐行相同。
```

因此 Phase6 中“selected-layer gamma 轻微接近”的解释需要修正为：

```text
旧 H22/H30 的 layergamma 配置写入了 config，但没有进入 sc-gamma rho 计算；
所以那些 run 不能作为 layergamma 有效/无效的证据，只能作为配置未生效状态下的轨迹结果。
```

### H27-H30：trajectory path proxy 与 commit-filter debug

| run | GPU | ATE 704F | wall min | projected full min | chunk mean | TTT mean | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| H27 pathlen ema alpha3 | 2 | `40.897348622677924` | `16.516666666666666` | `25.10533333333333` | `37.85437040328979` | `5.802112550735473` | 失败 |
| H28 pathlen ema alpha4 | 3 | `40.830079619661` | `17.766666666666666` | `27.005333333333333` | `40.85793343544006` | `6.074894361495971` | 失败 |
| H29 pathlen prev alpha3 | 4 | `40.81389341714244` | `16.533333333333335` | `25.130666666666666` | `37.96012343406677` | `5.639145746231079` | 失败 |
| H30 old layergamma + statecommit075 | 5 | `40.185368766362735` | `16.783333333333335` | `25.51066666666667` | `38.56237456321716` | `5.138150844573975` | 与 H19 逐行相同；不算新优化 |

H30 commit-filter debug 已可见：

```text
commit_filter_debug_seen = 25
commit_filter_active_mode_seen = 25
commit_filter_applied_debug_rows = 25
commit_filter_activation_rate_mean = 0.4333333333333333
commit_filter_scale_mean = 0.8916666666666666
commit_filter_modes_seen = state_conditioned_commit
```

结论：

```text
trajectory path-length proxy 没有改善 704F；
commit-filter 可观测性 blocker 已修复，但 H30 不提升性能。
```

### H31-H37：sc-gamma layergamma 修复后 704F

修复后 H31/H32/H33/H34 的 `01.txt` 均不再与 H18 完全相同，first diff 从 frame 29 出现，说明 layergamma 链路真实改变了轨迹。

| run | rho map | GPU | ATE 704F | wall min | projected full min | chunk mean | TTT mean | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| H31 | `0/8/17:0.003` | 2 | `39.90181810063874` | `16.533333333333335` | `25.130666666666666` | `37.97590750694275` | `4.856322240829468` | beat A-best 704F |
| H32 | `0/8/17:0.006` | 3 | `39.9844379268327` | `16.766666666666666` | `25.485333333333337` | `38.35878821372986` | `5.174338655471802` | 不升级 |
| H33 | `0/8/17:0.009` | 4 | `39.86624170202318` | `16.583333333333332` | `25.206666666666663` | `37.99921131134033` | `5.168029804229736` | beat A-best 704F |
| H34 | `0/8/17:0.012` | 5 | `39.965776688405064` | `17.55` | `26.676` | `40.30042562484741` | `5.258063020706177` | 不升级 |
| H35 | `0/8/17:0.0075` | 4 | `39.798247720485634` | `16.383333333333333` | `24.90266666666667` | `37.50150881767273` | `4.9218727684021` | best 704F，升级 full |
| H36 | `0/8/17:0.0105` | 5 | `39.95144397608373` | `16.35` | `24.852` | `37.562337217330935` | `4.873052206039429` | 不升级 |
| H37 | `0/8/17:0.007` | 5 | `39.91245830855871` | `15.95` | `24.244000000000003` | `36.56959966659546` | `4.8340124416351316` | 不升级 |

对照：

```text
A-best 704F self ATE = 39.92846746159833
H35 704F improvement vs A-best = 0.13021974111269685
```

### Phase7 full：H31/H33/H35

| run | rho map | frames | ATE full | delta vs C9 | Rot | FinalErr | wall min | chunk mean | TTT mean | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H31 full | `0/8/17:0.003` | `1101` | `35.83167463692937` | `2.068732534010521` | `5.783755` | `11.782992` | `24.983333333333334` | `38.06106244262896` | `5.019238898628636` | pass |
| H33 full | `0/8/17:0.009` | `1101` | `35.769087844821314` | `2.006145741902467` | `5.726937` | `11.623176` | `26.933333333333334` | `40.951174811313024` | `5.409016904078032` | pass |
| H35 full | `0/8/17:0.0075` | `1101` | `35.74089695811434` | `1.9779548551954917` | `5.750002` | `11.138076` | `24.783333333333335` | `37.640955912439445` | `4.905587773574026` | pass |

R9 分段：

| run | seg0 rmse | seg1 rmse | seg2 rmse |
|---|---:|---:|---:|
| C9_REF | `46.39902061769452` | `34.656626116071514` | `11.034547189236443` |
| V53_A_BEST | `45.04645244540594` | `40.85251832537275` | `16.81711362434614` |
| H31 | `45.00057513987618` | `40.688663174256426` | `16.77224281258083` |
| H33 | `44.931824306935155` | `40.66141901286172` | `16.6334465944543` |
| H35 | `44.943948305639466` | `40.48296121701416` | `16.778093400455763` |

最终 Phase7 判定：

```text
old best full = V53_FULL_A_SCGAMMASPLIT_AW110
old best ATE = 35.912858666859364
new best full = V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075
new best ATE = 35.74089695811434
improvement = 0.17196170874502315

progress-pass threshold = 35.30
gap to progress-pass = 0.4408969581143438
gap to C9 = 1.9779548551954917

success_progress_pass = False
best_full_updated = True
```

### Phase7 分析

1. path-length trajectory proxy 是负结果。  
   H27-H29 的 704F ATE 都在 `40.81--40.90`，明显差于 A-best 704F `39.92846746159833`。

2. commit-filter 可观测性 blocker 已解决，但当前 statecommit 配置不是性能来源。  
   H30 能看到 activation/debug 字段，activation rate mean `0.4333333333333333`；但轨迹与 H19 完全相同，704F ATE `40.185368766362735`。

3. 真正有效的是 sc-gamma layergamma 链路修复。  
   修复前 H22/H30 的 layergamma 不改变轨迹；修复后 H31/H33/H35 均改变轨迹，且 H35 full 刷新 v53 best。

4. H35 的收益主要来自中段 seg1。  
   V53_A_BEST seg1 `40.85251832537275`，H35 seg1 `40.48296121701416`，改善约 `0.36955710835859`；seg2 基本没有解决，仍为 `16.778093400455763`，远差于 C9 的 `11.034547189236443`。

5. 704F 与 full 的方向一致但幅度有限。  
   H35 704F best，full 也是 best；但 full 仍差 progress-pass `0.4408969581143438`，说明当前全局三层同 rho 只是小幅修复，不是最终 clean 解法。

### Phase7 结论

```text
success_progress_pass = False
best_overall_v53_full = V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075
best_overall_v53_full_ATE = 35.74089695811434
best_overall_improvement_vs_previous = 0.17196170874502315
runtime_gate = pass
no_chunk_policy = pass
manual_percentage_audit = pass
```

下一步不应继续 path-length proxy 或旧 statecommit 小扫。更合理方向：

```text
1. 沿着已验证有效的 sc-gamma rho 链路，做 layer-specific / branch-specific 704F sweep。
2. 重点观察 seg1 与 seg2：H35 改善 seg1，但 seg2 仍远差 C9。
3. 候选必须先超过 A-best/H35 704F，再升级 full。
4. full 仍需满足 28min runtime、no-chunk、manual percentage audit。
```

### Phase7 insight

```text
这轮最关键的 insight 是：之前一部分“参数无效”其实是链路没接上。
```

`adaptive_writer_sc_gamma_split` 的 sc-gamma 分支一直固定 `rho=0.005`，导致 `V47_TTT_LAYER_GAMMAS` 没有实际进入强度计算。修复后，rho sweep 立刻产生轨迹差异并刷新 full best。这说明 clean adaptive TTT 还有可优化空间，但当前全局 0/8/17 同 rho 只解决了一部分中段 drift，后段仍是主要缺口。

### Phase7 证据链

```text
Code:
  loger/pipeline/ttt_write_controller.py
  run_pipeline_abc_v2.py
  tools/v53_experiment_report.py

Reports:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R25_phase7_trajectory_state_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R26_phase7_layergamma_fix_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R27_phase7_layergamma_refine_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R29_phase7_layergamma_refine2_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R30_phase7_layergamma_fix_full_final

Autopsy:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_full_sequence_autopsy/report_R4_trajectory_state_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_full_sequence_autopsy/report_R5_layergamma_fix_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_full_sequence_autopsy/report_R6_layergamma_refine_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_full_sequence_autopsy/report_R8_layergamma_refine2_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_full_sequence_autopsy/report_R9_layergamma_fix_full_final

Rollouts:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_trajectory_state_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_layergamma_fix_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_layergamma_refine_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_layergamma_refine2_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_layergamma_fix_full/rollouts

Logs:
  logs/stream_v53_phase7_h27_pathlen_ema_a3_704f.log
  logs/stream_v53_phase7_h28_pathlen_ema_a4_704f.log
  logs/stream_v53_phase7_h29_pathlen_prev_a3_704f.log
  logs/stream_v53_phase7_h30_layergamma_statecommit075_704f.log
  logs/stream_v53_phase7_h31_layergammafix_rho003_704f.log
  logs/stream_v53_phase7_h32_layergammafix_rho006_704f.log
  logs/stream_v53_phase7_h33_layergammafix_rho009_704f.log
  logs/stream_v53_phase7_h34_layergammafix_rho012_704f.log
  logs/stream_v53_phase7_h35_layergammafix_rho0075_704f.log
  logs/stream_v53_phase7_h36_layergammafix_rho0105_704f.log
  logs/stream_v53_phase7_h37_layergammafix_rho007_704f.log
  logs/stream_v53_phase7_full_h31_layergammafix_rho003.log
  logs/stream_v53_phase7_full_h33_layergammafix_rho009.log
  logs/stream_v53_phase7_full_h35_layergammafix_rho0075.log
```

## 2026-06-09 Phase8 复盘：layer/branch/rho 细化没有突破 H35

Phase8 充分利用 GPU 0/1/2/3/4/5 做 704F 并行筛选。本阶段没有 full promotion；H35 full 仍是当前 best。

### C9 配置复核

查到最近 C9 repeat：

```text
results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts/V45_P0_C9_REPEAT/effective_config.yaml
```

关键 chunk-wise 配置：

```text
read_beta_frame_chunks:
  5-9: 4.85
  10-12: 4.25
  16: 4.25

ttt_write_gradient_reversal_chunk_gammas:
  5-9: 0.005
  10-12: 0.003
  16: 0.0003

ttt_write_tri_replay_chunk_params:
  5-12: 0.35/0.12/0.85
  16: 0.35/0.08/0.85

ttt_write_commit_ema_alpha:
  0.5

ttt_write_commit_ema_chunks:
  5,6
```

解释：

```text
C9 不是自动在某些 chunk 关闭 TTT。
C9 是手工指定少数 chunk 的 read beta、TTT correction gamma、tri replay params 和 commit EMA。
clean 版失败的重点不是平均参数值差很多，而是没有学到这些离散时序动作。
```

### Phase8 704F 结果

| run | 配置 | ATE704 | wall min | chunk mean | TTT mean | 判定 |
|---|---|---:|---:|---:|---:|---|
| H38 | layer0 rho0.0075 | `39.960135` | `16.633333333333333` | `38.20559416770935` | `5.0138212585449216` | 不升级 |
| H39 | layer8 rho0.0075 | `39.859164` | `17.583333333333332` | `40.30065856933594` | `5.506328716278076` | 不升级 |
| H40 | layer17 rho0.0075 | `39.965632` | `17.5` | `40.19906923294067` | `5.392944402694702` | 不升级 |
| H41 | layer8+17 rho0.0075 | `39.798242` | `16.583333333333332` | `37.97287831306458` | `5.052609062194824` | 等同 H35 704F |
| H42 | layer0+8 rho0.0075 | `39.859164` | `17.716666666666665` | `40.55417562484741` | `5.259451179504395` | 不升级 |
| H43 | layer0+17 rho0.0075 | `39.965632` | `17.133333333333333` | `39.19534448623657` | `5.148021402359009` | 不升级 |
| H44 | branch0 0/8/17 rho0.0075 | `39.798242` | `17.6` | `39.1620478439331` | `5.123666524887085` | 等同 H35/H41 |
| H45 | branch1 0/8/17 rho0.0075 | `41.896211` | `17.45` | `38.7483429145813` | `3.2366672801971434` | audit false；失败 |
| H46 | branch2 0/8/17 rho0.0075 | `41.896211` | `16.8` | `37.27957184791565` | `3.2869395065307616` | audit false；失败 |
| H47 | branchall 0/8/17 rho0.0075 | `39.798242` | `16.7` | `37.10262540817261` | `4.83673641204834` | 等同 H35/H41 |
| H48 | layer8+17 rho0.0065 | `39.933831` | `17.95` | `41.27058617591858` | `5.566148166656494` | 不升级 |
| H49 | layer8+17 rho0.0085 | `39.986971` | `17.15` | `39.30914575576782` | `5.1856677532196045` | 不升级 |
| H50 | layer8+17 rho0.0055 | `39.891210` | `18.65` | `41.887183818817135` | `5.6431180858612064` | 不升级 |
| H51 | layer8+17 rho0.0095 | `39.896061` | `17.183333333333334` | `38.435900869369505` | `5.135310764312744` | 不升级 |
| H52 | layer8 rho0.0065, layer17 rho0.0085 | `39.871933` | `16.983333333333334` | `37.94002234458923` | `5.009374647140503` | 不升级 |
| H53 | layer8 rho0.0085, layer17 rho0.0065 | `39.942017` | `18.033333333333335` | `40.4485304069519` | `5.357725143432617` | 不升级 |

R10 autopsy:

```text
H35_704 / H41_L8_17 / H44_BRANCH0:
  ATE recompute = 39.798248
  seg0 = 46.051213
  seg1 = 30.667454
  seg2 = 29.647953

H52_ASYM:
  ATE = 39.871933
  seg0 = 46.059087
  seg1 = 30.859968
  seg2 = 30.113921
```

### Phase8 分析

1. H35 的 704F 有效部分基本来自 layer8+17，layer0 不是必要条件。  
   H41 layer8+17 与 H35 704F 完全等价到报告精度；H38 layer0 单独退化到 `39.960135`。

2. branch1/branch2 不是可用方向。  
   H45/H46 ATE 都是 `41.896211`，且 manual percentage audit false / split_debug_count=0；这两条只保留为审计负例。

3. branchall 不增加收益。  
   H47 与 H41/H35 等价，说明在当前实现里把 branch 全打开没有带来新的有效控制。

4. layer8+17 的 rho/asym 细化没有突破。  
   H48-H53 全部差于 H41/H35；最接近的 H52 也只有 `39.871933`，明显差于 `39.798242`。

5. 当前 sc-gamma layer/rho 空间已经到平台。  
   继续在 `8/17 + branch0 + rho` 上小扫，预期收益很低。

### 自适应问题判断

```text
不是完全没有信号；
是当前自适应算法可见的状态信号不足以模拟 C9 的离散时序动作。
```

证据：

```text
有信号：
  Phase7 修通 sc-gamma rho 后，full 从 35.912858666859364 提到 35.74089695811434。

信号不足：
  Phase8 在 layer/branch/rho/asym 上 16 条 704F 没有实质突破。
  当前 risk/prior/native-delta 只能调局部写入强度，不能判断 C9 的 chunk 5/6 commit EMA、10-12 gamma 降档、16 极弱 gamma 的时机。
```

结论：

```text
当前问题更像 state aliasing：
  对当前可见局部特征而言，一些 chunk 看起来相似；
  但 C9 手工策略对它们采取不同动作。

所以 clean 版差距不是“参数平均值不接近 C9”，
而是“缺少能复现 C9 timing 的状态变量/控制器”。
```

### Phase8 结论

```text
success_progress_pass = False
full_promotion = False
best_phase8_704F = H41/H44/H47
best_phase8_704F_ATE = 39.798242
best_overall_v53_full = V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075
best_overall_v53_full_ATE = 35.74089695811434
```

下一步方向：

```text
1. 不再继续 sc-gamma layer/rho 小扫。
2. 把已经有效的 layer8+17 sc-gamma 与 state-conditioned commit filter 组合起来测。
3. 目标是模拟 C9 的 commit EMA chunks 5/6，但不能使用 chunk id。
4. 如果仍失败，需要做 teacher-vs-student timing classifier/autopsy，而不是继续纯参数搜索。
```

### Phase8 证据链

```text
Reports:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R31_phase8_layer_specific_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R32_phase8_branch_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R33_phase8_layer817_refine_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R34_phase8_layer817_asym_screen

Autopsy:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase8_full_sequence_autopsy/report_R10_phase8_screen

Rollouts:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase8_layer_specific_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase8_branch_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase8_layer817_refine_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase8_layer817_asym_screen/rollouts

Logs:
  logs/stream_v53_phase8_h38_layer0_rho0075_704f.log
  logs/stream_v53_phase8_h39_layer8_rho0075_704f.log
  logs/stream_v53_phase8_h40_layer17_rho0075_704f.log
  logs/stream_v53_phase8_h41_layer8_17_rho0075_704f.log
  logs/stream_v53_phase8_h42_layer0_8_rho0075_704f.log
  logs/stream_v53_phase8_h43_layer0_17_rho0075_704f.log
  logs/stream_v53_phase8_h44_branch0_rho0075_704f.log
  logs/stream_v53_phase8_h45_branch1_rho0075_704f.log
  logs/stream_v53_phase8_h46_branch2_rho0075_704f.log
  logs/stream_v53_phase8_h47_branchall_rho0075_704f.log
  logs/stream_v53_phase8_h48_layer8_17_rho0065_704f.log
  logs/stream_v53_phase8_h49_layer8_17_rho0085_704f.log
  logs/stream_v53_phase8_h50_layer8_17_rho0055_704f.log
  logs/stream_v53_phase8_h51_layer8_17_rho0095_704f.log
  logs/stream_v53_phase8_h52_layer8_0065_layer17_0085_704f.log
  logs/stream_v53_phase8_h53_layer8_0085_layer17_0065_704f.log
```

## Phase9：commit 时序模拟复盘

### 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `run_pipeline_abc_v2.py` | 从 HEAD 恢复缺失文件：`git show HEAD:run_pipeline_abc_v2.py > run_pipeline_abc_v2.py` | 主入口曾处于 deleted 状态，launcher 无法继续实验；这是恢复缺失入口，不是回滚结果 |
| `run_pipeline_abc_v2.py` | 新增/恢复 `state_conditioned_commit`、`native2candidate_by_risk` 等 commit filter CLI choices | Phase9 要测试 controller 已实现的 commit filter；原 parser 拦截导致 H54-H59 rc=2 |
| `run_pipeline_abc_v2.py` | 新增 `ttt_write_native_mix_chunks`、`ttt_write_tri_replay_role_mode`、`ttt_write_gradient_reversal_transient_apply_scale`、`ttt_write_commit_ema_chunks`、`ttt_write_scale_state_*` 并传入 `HybridMemoryController` | 恢复入口与 controller 的参数一致性，确保实验旋钮真实生效 |
| `run_pipeline_abc_v2.py` | 新增 `v11_projection_*`、`online_scale_state_*`、`probe_cache_*`、`empty_cuda_cache_each_chunk`、`v32_semantic_cue_*`、`context_source_skip_*`、`semantic_role_*`、`v29c_masklet_*`、`v36_synthetic_*` compatibility parser | 统一 launcher 会传这些历史诊断参数；本轮取值均为 off/none/empty，不改变 clean 行为，只解除入口 blocker |
| `tools/v53_experiment_report.py` | `_iter_run_dirs` 支持 `run_status.txt + 01.txt` rollout；`_timing_stats` 从 `01.log` 解析阶段耗时；`_commit_filter_stats` 扫描 `01.log` debug | Phase9 没落 `timing_summary.json`，原 report 得到 0 rows 且误报 commit filter debug=0；修复后不补造缺失 TTT 子阶段耗时 |

### Blocker 处理

```text
H54-H59 首次启动失败：
  invalid choice: state_conditioned_commit

H54-H59 第二次启动失败：
  unrecognized arguments: v32/context/semantic/v29c/v36 compatibility args

这些均为 launcher/入口兼容 blocker，不作为算法负例。
修复后 r3 六个 run 均完成。
```

### Phase9 结果

| Run | 方法 | ATE 704F | Rot | wall min | chunk mean | pass2 mean | commit debug | scale mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| H54 | state commit min0.50 | `40.3783773752881` | `2.8965180125128684` | `16.966666666666665` | `31.9072` | `10.650000000000002` | `25/25` | `0.789258757710796` | 差于 H35 |
| H55 | state commit min0.65 | `40.24293894797938` | `2.9142385463126335` | `16.333333333333332` | `30.8224` | `10.242` | `25/25` | `0.8522222222222224` | 差于 H35 |
| H56 | state commit min0.75 | `40.07617220766744` | `2.8707779660815924` | `17.433333333333334` | `32.8824` | `10.8748` | `25/25` | `0.8938888888888888` | 差于 H35 |
| H57 | old decay mean | `39.953848909964535` | `2.8879959100201225` | `17.7` | `31.957200000000004` | `10.405200000000002` | `25/25` | `0.9646988984745823` | Phase9 best，但差于 H35 |
| H58 | old decay q90 | `40.216765335248425` | `2.9071219475031573` | `17.833333333333332` | `32.2676` | `10.365599999999999` | `25/25` | `0.8813276774436235` | 差于 H35 |
| H59 | native2candidate mean | `40.29482875365769` | `2.87096569799561` | `18.566666666666666` | `33.8516` | `11.236400000000001` | `25/25` | `0.7852583289601737` | 差于 H35 |

对照：

```text
H35 704F = 39.798247720485634
H41/H44 704F = 39.798242
Phase9 best H57 = 39.953848909964535
Phase9 best is worse than H35 704F by about 0.155601189478901
```

### 分析

```text
1. commit filter 不是没接上。
   R35 report 显示 H54-H59 commit_filter_debug_seen=25，applied rows=25。
   H54-H56 state_conditioned_commit activation_rate_mean 约 0.422-0.424。

2. 但这种状态条件 commit 没有产生 C9 少数 chunk 时序收益。
   H54-H56 越放松 scale floor，ATE 仍在 40.08-40.38，全部差于 H35。
   H57 risk old-decay 最好，也只是 39.953849，仍差于 39.798248。

3. 这说明当前风险/状态不是完全无信号，但不足以分辨 C9 hidden schedule。
   Phase7 的 sc-gamma fix 曾让 full ATE 从 35.912858666859364 提到 35.74089695811434；
   因此不是“完全没信号”。
   但 Phase8+Phase9 说明继续用局部 risk/prior/native-delta/commit-distance
   做连续强度调节，无法复现 C9 的离散 chunk 动作。

4. 当前失败更像算法状态定义问题，而不是单纯调参不够。
   C9 的关键不是所有 chunk 参数均值差异大，
   而是少数 chunk 的 read beta / gamma / commit EMA 时序动作。
   当前 clean 自适应没有足够的状态变量去识别“这个 chunk 应该降档/commit EMA/极弱 gamma”。
```

### 对用户问题的直接回答

```text
问题：为什么现在的自适应算法没办法模拟 C9？是算法有问题还是没有信号？

答案：
  不是完全没有信号。
  但当前自适应算法的状态表示和控制方式有问题，准确说是 state aliasing：
    对局部可见的 risk/prior/native-delta 来说，一些 chunk 看起来很像；
    C9 却对其中少数 chunk 做了不同的离散时序动作。

因此当前算法能调“写强度”，但不会可靠学到“什么时候该切换写/commit/read 节奏”。
```

### Phase9 结论

```text
success_progress_pass = False
full_promotion = False
best_phase9_704F = H57
best_phase9_704F_ATE = 39.953848909964535
best_overall_v53_full remains V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075
best_overall_v53_full_ATE remains 35.74089695811434
```

### 下一步判断

```text
不建议继续盲扫 sc-gamma rho/layer/branch 或简单 commit scale。
下一步如果继续优化，需要新计划：
  用 teacher-vs-student timing autopsy/classifier 先验证 no-GT 特征是否能区分 C9 特殊 chunk；
  如果可分，再做离散 action controller；
  如果不可分，说明 clean 版缺少可观测信号，继续在线自适应会反复贴近 H35 平台。
```

### Phase9 证据链

```text
Report:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R35_phase9_commit_layer817_screen

Rollouts:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H54_L817_SCOMMIT050_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H55_L817_SCOMMIT065_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H56_L817_SCOMMIT075_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H57_L817_OLDDECAY_RESXDG_MEAN_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H58_L817_OLDDECAY_RESXDG_Q90_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H59_L817_NATIVE2CAND_RESXDG_MEAN_704F

Logs:
  logs/stream_v53_phase9_h54_l817_scommit050_704f.log
  logs/stream_v53_phase9_h54_l817_scommit050_704f_r2.log
  logs/stream_v53_phase9_h54_l817_scommit050_704f_r3.log
  logs/stream_v53_phase9_h55_l817_scommit065_704f_r3.log
  logs/stream_v53_phase9_h56_l817_scommit075_704f_r3.log
  logs/stream_v53_phase9_h57_l817_olddecay_resxdg_mean_704f_r3.log
  logs/stream_v53_phase9_h58_l817_olddecay_resxdg_q90_704f_r3.log
  logs/stream_v53_phase9_h59_l817_native2cand_resxdg_mean_704f_r3.log
  logs/stream_v53_phase9_report_R35_r3.log
```
