# ACL2 v53 Fast No-SWA C9-Clean Adaptive TTT to C9 执行日志

日期：2026-06-09  
工作目录：`/mnt/data/users/chengshun.wang/pjs/LoGeR`  
计划文件：`docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9`  
Python：`/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`  
可用 GPU：`0,1,2,3,4,5`

## 执行原则

- 不编造数据；所有 ATE、runtime、audit 结果只能来自落盘 artifact。
- full KITTI01 hard gate：`wall_time_min <= 28`，`chunk_total_seconds_mean <= 42`，`probe_ttt_write_seconds_mean <= 8`，`hmc_rows = 38`，`frames = 1101`，`status = DONE`。
- 候选必须 no chunk-id：`read_beta_frame_chunks_empty`、`ttt_gradient_reversal_chunk_gammas_empty`、`ttt_tri_replay_chunk_params_empty`、`ttt_commit_ema_chunks_empty`、`native_mix_chunks_empty`、`semantic_action_active_chunks_empty` 均为 true。
- tri replay 不允许手工 percentage：`manual_positive_frac = 0`，`manual_negative_frac = 0`，`manual_neutral_lambda = 0`，role mode 必须是 adaptive/state-conditioned/no-percentage 类。
- 先 96F smoke，再 384F/512F runtime projection，只有通过 gate 才进入 full。

## 2026-06-09 04:42 +08：计划阅读与代码审计

已阅读 `ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md` 的硬约束、Phase 0-5 和 stop rules。v53 的三个候选：

- A：`SC-GammaSplit`
- B：`SC-GammaCommit`
- C：`ConflictLite-Split`

复用的真实历史证据：

- C9 reference：`C9_P0_R2`，ATE `33.76294210291885m`。
- v50 robust split：ATE `35.985306009701524m`。
- v52 EnergyMatched AW111：ATE `35.967687m`。
- v52 EnergyMatched AW110 no-SWA：ATE `35.974144m`，wall `26.17min`，chunk mean `39.476774s`，probe TTT mean `5.132336s`。
- v52 ConflictLite smoke：96F ATE `1.315722`，chunk mean `45.389658s`，probe TTT mean `14.290215s`，已超过 v53 runtime gate，因此 v53 C 必须重新以轻量实现验证，若仍超时立即停止。

## 2026-06-09 04:42 +08：代码修改

修改文件：

- `loger/pipeline/ttt_write_controller.py`
- `run_pipeline_abc_v2.py`
- `tools/run_v47_adaptive_ttt_writer_candidate.sh`
- 新增 `tools/v53_experiment_report.py`

关键修改：

- 增加 `adaptive_writer_sc_gamma_split` / `adaptive_writer_sc_gamma_commit_split` role mode。
- 增加 SC-Gamma role assignment：用当前 chunk 内的 safety/danger robust margin 划分 positive/neutral/negative；塌缩时记录 `ttt_tri_replay_role_collapsed=True`，不 fallback 到 fixed percentage。
- 增加 SC-Gamma branch gamma：基于 native update energy、negative branch energy、risk variance 和 prior variance 计算 gamma；不依赖 chunk id。
- 增加 state-conditioned commit filter：`native_distance_adaptive_ema` / `candidate_native_distance_ema` / `state_conditioned_commit`，用 candidate-native 距离与 cosine distance 的 robust threshold 决定 commit alpha；不使用 `commit_ema_chunks`。
- launcher 写出 `effective_config.yaml`、`chunk_id_policy_audit.json`、`adaptive_ttt_audit.json`、`reproduce_command.sh`。
- 新增 v53 汇总脚本，只从已有 artifact 生成 registry、runtime summary、audit summary、failure routing 和 final report。
- 2026-06-09 05:16 后续修复：为 ConflictLite 增加 `conflict_lite_layer0` / `conflict_lite_layer8` / `conflict_lite_layer17` 单层 alias。
- 2026-06-09 05:33 后续修复：为 ConflictLite 增加 `conflict_lite_layer0_sample2048` / `conflict_lite_layer17_sample2048` sampled-token alias；selected 层只用 2048 个 evenly-spaced token 估计 conflict proxy，并写出 `ttt_update_conflict_sample_tokens_used` / `ttt_update_conflict_sampled` debug 字段。

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile run_pipeline_abc_v2.py loger/pipeline/ttt_write_controller.py tools/v47_adaptive_ttt_writer_report.py tools/v52_runtime_profile_report.py
bash -n tools/run_v47_adaptive_ttt_writer_candidate.sh && bash -n tools/run_attention_cue_experiment.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v53_experiment_report.py loger/pipeline/ttt_write_controller.py run_pipeline_abc_v2.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v53_experiment_report.py --result-root results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R0
```

结果：

- `py_compile` 通过。
- shell 语法检查通过。
- 初始 `report_R0` 生成成功；此时尚无 v53 run row。

## 2026-06-09 04:47--04:51 +08：Phase 3 / 96F smoke

并行启动 3 个 smoke，均为 AW110 no-SWA：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_smoke/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW110_FRAME_ADAPTIVE_TTT V53_SMOKE_A_SCGAMMASPLIT_AW110_96F
```

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_smoke/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_commit_split \
V47_TTT_COMMIT_FILTER_MODE=native_distance_adaptive_ema \
V47_TTT_COMMIT_FILTER_MIN=0.35 \
V47_TTT_COMMIT_FILTER_MAX=1.0 \
V47_TTT_COMMIT_FILTER_BRANCH_MASK=0 \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW110_FRAME_ADAPTIVE_TTT V53_SMOKE_B_SCGAMMACOMMIT_AW110_96F
```

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_smoke/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=conflict_lite_selected_layers \
V47_TTT_ROLE_MODE=adaptive_writer_conflictlite_split \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW110_FRAME_ADAPTIVE_TTT V53_SMOKE_C_CONFLICTLITE_AW110_96F
```

后处理命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v53_experiment_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
  --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R1_smoke

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
  --rollout-root results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_smoke/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R1_smoke_v47

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v52_runtime_profile_report.py \
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_smoke/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R1_smoke_runtime \
  --max-wall-seconds 1680
```

真实结果：

| run | status | frames | ATE | wall min | projected full min | chunk mean | probe TTT mean | no-chunk | manual % | split/fused debug | role collapse |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| `V53_SMOKE_A_SCGAMMASPLIT_AW110_96F` | done | 96 | `1.3260134756601656` | `2.60` | `24.70` | `33.56557655334473` | `4.709445774555206` | true | true | `72/0` | `0/72` |
| `V53_SMOKE_B_SCGAMMACOMMIT_AW110_96F` | done | 96 | `1.2925985616830291` | `2.6666666666666665` | `25.333333333333332` | `34.65052515268326` | `4.784034490585327` | true | true | `72/0` | `0/72` |
| `V53_SMOKE_C_CONFLICTLITE_AW110_96F` | done | 96 | `1.3073629931206907` | `3.316666666666667` | `31.508333333333333` | `44.106241106987` | `13.946743845939636` | true | true | `72/0` | no SC-gamma field |

Phase 3 决策：

- A 通过 96F gate，进入 384F runtime projection。
- B 通过 96F gate，进入 384F runtime projection。
- C 因 `chunk_total_seconds_mean > 42` 且 `probe_ttt_write_seconds_mean > 8`，按 v53 stop rule 停止 ConflictLite line；不进入 384F 或 full。

## 2026-06-09 04:53--05:03 +08：Phase 3 / 384F runtime projection

只运行 A/B；C 已因 96F runtime fail 停止。

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_runtime_screen/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_END_FRAME=384 \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW110_FRAME_ADAPTIVE_TTT V53_SCREEN_A_SCGAMMASPLIT_AW110_384F
```

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_runtime_screen/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_END_FRAME=384 \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_commit_split \
V47_TTT_COMMIT_FILTER_MODE=native_distance_adaptive_ema \
V47_TTT_COMMIT_FILTER_MIN=0.35 \
V47_TTT_COMMIT_FILTER_MAX=1.0 \
V47_TTT_COMMIT_FILTER_BRANCH_MASK=0 \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW110_FRAME_ADAPTIVE_TTT V53_SCREEN_B_SCGAMMACOMMIT_AW110_384F
```

后处理命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v53_experiment_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
  --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R2_screen

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v52_runtime_profile_report.py \
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_runtime_screen/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R2_screen_runtime \
  --max-wall-seconds 1680
```

真实结果：

| run | status | frames | ATE_short | wall min | projected full min | chunk mean | probe TTT mean | no-chunk | manual % | split/fused debug | role collapse |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| `V53_SCREEN_A_SCGAMMASPLIT_AW110_384F` | done | 384 | `33.813351532934135` | `9.783333333333333` | `26.554761904761907` | `39.52451760428293` | `5.157917380332947` | true | true | `252/0` | `0.0` |
| `V53_SCREEN_B_SCGAMMACOMMIT_AW110_384F` | done | 384 | `33.96840141632595` | `9.0` | `24.428571428571427` | `36.32406502110617` | `4.731844697679792` | true | true | `252/0` | `0.0` |

Phase 3 决策：

- A/B 均满足 projected full wall <= 28min、chunk mean <= 42s、probe TTT <= 8s、no-chunk audit true、manual-percentage audit true、role collapse 0。
- A 384F ATE 更低；B runtime 更快。
- 按 v53 Phase 4 最小 full 验证，启动 A/B 两条 full；C 不进入 full。

## 2026-06-09 05:04--05:32 +08：Phase 4 / A-B full

启动命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase4_full/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW110_FRAME_ADAPTIVE_TTT V53_FULL_A_SCGAMMASPLIT_AW110
```

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase4_full/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_commit_split \
V47_TTT_COMMIT_FILTER_MODE=native_distance_adaptive_ema \
V47_TTT_COMMIT_FILTER_MIN=0.35 \
V47_TTT_COMMIT_FILTER_MAX=1.0 \
V47_TTT_COMMIT_FILTER_BRANCH_MASK=0 \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW110_FRAME_ADAPTIVE_TTT V53_FULL_B_SCGAMMACOMMIT_AW110
```

真实结果：

| run | status | frames | hmc rows | ATE | delta vs C9 | wall min | chunk mean | probe TTT mean | no-chunk | manual % | split/fused debug | role collapse | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| `V53_FULL_A_SCGAMMASPLIT_AW110` | done | 1101 | 38 | `35.912858666859364` | `2.149916563940515` | `26.65` | `40.57448160648346` | `5.432522616888347` | true | true | `684/0` | `0.0` | runtime pass, ATE fail |
| `V53_FULL_B_SCGAMMACOMMIT_AW110` | done | 1101 | 38 | `36.317006895477675` | `2.554064792558826` | `27.35` | `41.662026831978245` | `5.646422605765493` | true | true | `684/0` | `0.0` | runtime pass, ATE fail |

决策：

- A/B 都满足 full runtime hard gate。
- A/B 均 `ATE > 35.30`，因此都不是 progress-pass。
- B 的 commit filter 没带来 full ATE 改善；当前 evidence 更支持继续查 conflict / update-energy proxy，而不是继续扩大 SC-GammaCommit。

## 2026-06-09 05:16--05:48 +08：ConflictLite blocker repair

用户提醒 GPU 2/3/4/5 空闲后，按 v53 blocker 修复方向并行利用空闲 GPU，但不违反 stop rule：

1. 原始 C `conflict_lite_selected_layers={0,8,17}` 96F 超时，停止 direct C。
2. 第一轮修复：单层 selected layer。
3. 第二轮修复：sampled-token conflict proxy，selected layer 只采样 2048 tokens。

第一轮修复命令示例：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_conflictlite_repair/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=conflict_lite_layer17 \
V47_TTT_ROLE_MODE=adaptive_writer_conflictlite_split \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 4 AW110_FRAME_ADAPTIVE_TTT V53_REPAIR_C_CONFLICTLITE_LAYER17_AW110_96F
```

第一轮真实结果：

| run | frames | ATE | projected full min | chunk mean | probe TTT mean | decision |
|---|---:|---:|---:|---:|---:|---|
| `V53_REPAIR_C_CONFLICTLITE_LAYER0_AW110_96F` | 96 | `1.3402113070904205` | `27.075` | `37.37825036048889` | `7.972570598125458` | 进入 384F，但很临界 |
| `V53_REPAIR_C_CONFLICTLITE_LAYER17_AW110_96F` | 96 | `1.3044604606864516` | `25.333333333333332` | `34.44476252794266` | `6.98481285572052` | 进入 384F |
| `V53_REPAIR_C_CONFLICTLITE_LAYER8_AW110_96F` | 96 | `1.34019069300634` | `27.55` | `38.08236104249954` | `8.26403135061264` | probe TTT fail，停止 |

第一轮 384F 真实结果：

| run | frames | ATE_short | projected full min | chunk mean | probe TTT mean | decision |
|---|---:|---:|---:|---:|---:|---|
| `V53_SCREEN_C_REPAIR_CONFLICTLITE_LAYER0_AW110_384F` | 384 | `33.78770294045761` | `28.273809523809526` | `42.322086232049124` | `8.67749570097242` | runtime fail，停止 |
| `V53_SCREEN_C_REPAIR_CONFLICTLITE_LAYER17_AW110_384F` | 384 | `34.22957526718698` | `27.45952380952381` | `40.96893012523651` | `8.066814933504377` | probe TTT fail，停止 |

第二轮 sampled-token 修复命令示例：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase3_conflictlite_sampled_repair/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_END_FRAME=96 \
V47_TTT_RISK_SOURCE=conflict_lite_layer17_sample2048 \
V47_TTT_ROLE_MODE=adaptive_writer_conflictlite_split \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW110_FRAME_ADAPTIVE_TTT V53_REPAIR_C_CONFLICTLITE_LAYER17_SAMPLE2048_AW110_96F
```

sampled debug 复核：

```text
V53_REPAIR_C_CONFLICTLITE_LAYER17_SAMPLE2048_AW110_96F sampled_debug_count=4 sample_tokens_used_first=['2048','2048','2048']
V53_REPAIR_C_CONFLICTLITE_LAYER0_SAMPLE2048_AW110_96F sampled_debug_count=4 sample_tokens_used_first=['2048','2048','2048']
```

第二轮 96F 真实结果：

| run | frames | ATE | projected full min | chunk mean | probe TTT mean | decision |
|---|---:|---:|---:|---:|---:|---|
| `V53_REPAIR_C_CONFLICTLITE_LAYER0_SAMPLE2048_AW110_96F` | 96 | `1.3395832605016933` | `27.55` | `37.72022384405136` | `5.507285892963409` | 进入 384F |
| `V53_REPAIR_C_CONFLICTLITE_LAYER17_SAMPLE2048_AW110_96F` | 96 | `1.341075549605423` | `25.175` | `34.227492332458496` | `4.690415143966675` | 进入 384F |

第二轮 384F 真实结果：

| run | frames | ATE_short | delta vs C9 | projected full min | chunk mean | probe TTT mean | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `V53_SCREEN_C_CONFLICTLITE_LAYER0_SAMPLE2048_AW110_384F` | 384 | `33.92511490882306` | `0.16217280590421268` | `27.45952380952381` | `40.93911133493696` | `5.553112268447876` | 启动唯一 C full |
| `V53_SCREEN_C_CONFLICTLITE_LAYER17_SAMPLE2048_AW110_384F` | 384 | `34.31631528249291` | `0.5533731795740593` | `25.65` | `38.2044951234545` | `5.075153061321804` | 不跑 full，ATE_short 较差 |

## 2026-06-09 05:48 +08：Phase 4 / C sampled ConflictLite full

启动命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase4_full/rollouts \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
V47_TTT_RISK_SOURCE=conflict_lite_layer0_sample2048 \
V47_TTT_ROLE_MODE=adaptive_writer_conflictlite_split \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW110_FRAME_ADAPTIVE_TTT V53_FULL_C_CONFLICTLITE_LAYER0_SAMPLE2048_AW110
```

状态：

- run 已完成。

最终后处理命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v53_experiment_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
  --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v52_runtime_profile_report.py \
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase4_full/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final_runtime \
  --max-wall-seconds 1680

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
  --rollout-root results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase4_full/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final_v47
```

最终 full 真实结果：

| run | status | frames | hmc rows | ATE | delta vs C9 | Rot | FinalErr | wall min | chunk mean | probe TTT mean | no-chunk | manual % | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `V53_FULL_A_SCGAMMASPLIT_AW110` | done | 1101 | 38 | `35.912858666859364` | `2.149916563940515` | `5.809712796892106` | `11.645295504302135` | `26.65` | `40.57448160648346` | `5.432522616888347` | true | true | runtime pass, ATE fail |
| `V53_FULL_B_SCGAMMACOMMIT_AW110` | done | 1101 | 38 | `36.317006895477675` | `2.554064792558826` | `5.667585251473343` | `10.294665409438705` | `27.35` | `41.662026831978245` | `5.646422605765493` | true | true | runtime pass, ATE fail |
| `V53_FULL_C_CONFLICTLITE_LAYER0_SAMPLE2048_AW110` | done | 1101 | 38 | `36.08133075676421` | `2.3183886538453606` | `5.837362114451481` | `13.476248480651599` | `24.283333333333335` | `37.00016983559257` | `5.0161768825430615` | true | true | runtime pass, ATE fail |

rolling window 指标：

| run | rolling50 mean/p90/worst | rolling100 mean/p90/worst | rolling200 mean/p90/worst |
|---|---|---|---|
| `V53_FULL_A_SCGAMMASPLIT_AW110` | `30.754175659031393 / 60.092918544733564 / 79.88573799028862` | `32.00159050856109 / 56.784130552238544 / 71.43510550708994` | `34.42390643451079 / 54.104031598262296 / 55.84389625519551` |
| `V53_FULL_B_SCGAMMACOMMIT_AW110` | `31.162076259992013 / 60.11454536637435 / 81.06931981229218` | `32.33900382123641 / 56.561473264691756 / 72.3434766398543` | `34.65120562374255 / 54.76036704443822 / 56.44890164096004` |
| `V53_FULL_C_CONFLICTLITE_LAYER0_SAMPLE2048_AW110` | `30.69178299078554 / 61.14824071532603 / 80.66612054868085` | `32.00375797929032 / 57.9171232248693 / 72.0656565894848` | `34.54191085177192 / 54.549914704732025 / 56.333479290314976` |

额外 debug 复核：

```text
V53_FULL_B_SCGAMMACOMMIT_AW110:
  commit_filter_applied_seen = 38
  commit_filter_applied_true = 38
  commit_w0_triggered_seen = 684
  commit_w0_triggered_true = 289

V53_FULL_C_CONFLICTLITE_LAYER0_SAMPLE2048_AW110:
  sampled_seen = 38
  sampled_true = 38
  sample_tokens_used_first = ['2048','2048','2048','2048','2048']
```

最终判定：

- 本轮产出了 no-chunk / no-manual-percentage adaptive TTT candidates。
- 所有 3 条 full 均满足 runtime hard gate。
- 所有 3 条 full 均 `ATE > 35.30`，因此没有 progress-pass / soft-pass / close-pass。
- best full 是 A：`35.912858666859364`，距离 C9 `+2.149916563940515m`。
- 按 v53 stop rule：主候选 full 全部 fail，停止算法尝试，输出 failure report 和复盘分析。

## 最终交付物

主报告目录：

`results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final`

包含：

- `v53_phase0_efficiency_audit.md`
- `v53_c9_teacher_student_autopsy.md`
- `teacher_student_role_mass_timeline.png`
- `teacher_student_gamma_timeline.png`
- `teacher_student_post_zp_delta_norm.png`
- `candidate_vs_native_cosine_timeline.png`
- `risk_spread_vs_delta_norm_scatter.png`
- `v53_candidate_registry.csv`
- `v53_candidate_registry.json`
- `v53_full_metrics_summary.md`
- `v53_runtime_profile_summary.csv`
- `v53_runtime_profile_by_chunk.png`
- `v53_no_chunk_policy_audit.json`
- `v53_manual_percentage_audit.json`
- `v53_failure_routing_report.md`
- `v53_final_report.md`

补充 runtime 报告：

- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final_runtime/v52_runtime_profile_summary.{json,csv,md}`

补充 v47 registry：

- `results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R7_final_v47/v47_adaptive_ttt_writer_registry.{csv,json}`

## 2026-06-09 06:37--08:10 +08：continuation，按计划继续优化 clean 版并使用 GPU 2/3/4/5

用户指出还有 GPU `2,3,4,5` 空闲，且 clean 版仍未成功，要求继续按计划优化。本阶段继续遵守：

```text
no chunk-id policy
no manual tri-replay percentage
no SWA variable
full wall <= 28min
先 96F / 384F screen，再 full
```

### 代码修改与验证

修改文件：

- `loger/pipeline/ttt_write_controller.py`
- `loger/pipeline/hybrid_memory_controller.py`
- `run_pipeline_abc_v2.py`
- `tools/run_attention_cue_experiment.sh`
- `tools/run_v47_adaptive_ttt_writer_candidate.sh`

修改内容：

- 新增 `ttt_write_scale_state_sample_tokens` / `V47_TTT_SCALE_STATE_SAMPLE_TOKENS` / `TTT_WRITE_SCALE_STATE_SAMPLE_TOKENS` 参数链路。
- `v19_scale_state` projection risk 支持 evenly-spaced token sampling；默认 `0` 保持全 token 行为。
- sampled scale-state debug 落盘字段：
  - `v19_scale_state_sample_tokens_requested`
  - `v19_scale_state_sampled`
  - `v19_scale_state_sample_tokens_used`
- sampled path 使用当前 tensor device 计算，避免原 full-token CPU projection 的 runtime blocker。
- wrapper 的 `effective_config.yaml`、`adaptive_ttt_audit.json`、`chunk_id_policy_audit.json`、`reproduce_command.sh` 均记录 sample tokens。

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  loger/pipeline/ttt_write_controller.py \
  loger/pipeline/hybrid_memory_controller.py \
  run_pipeline_abc_v2.py \
  tools/v53_experiment_report.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v47_adaptive_ttt_writer_candidate.sh
```

结果：

```text
py_compile pass
bash -n pass
```

### 共用复现模板

所有 continuation run 均使用下面 wrapper，差异只在 env delta、GPU、RUN_NAME、phase/base：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9 \
V47_ROLLOUT_BASE=<phase_rollout_root> \
V47_PLAN_NOTE=docs/ACL2_v53_Fast_NoSWA_C9Clean_AdaptiveTTT_to_C9_Plan.md \
<env deltas> \
tools/run_v47_adaptive_ttt_writer_candidate.sh <gpu> AW110_FRAME_ADAPTIVE_TTT <run_name>
```

每个 run 目录均包含：

```text
effective_config.yaml
v47_effective_config.yaml
adaptive_ttt_audit.json
chunk_id_policy_audit.json
reproduce_command.sh
v47_reproduce_command.sh
01.log
01.txt
kitti_benchmark.log
timing_summary.json
wall_time_summary.json
```

### Continuation run 清单

| run | phase | GPU | end frame | 关键 env delta |
|---|---|---:|---:|---|
| `V53_CONT_SMOKE_H1_BRANCHALL_SCGAMMA_AW110_96F` | `phase5_clean_continuation_smoke` | 0 | 96 | `V47_TTT_BRANCH_MASK=all` |
| `V53_CONT_SMOKE_H2_NATIVEMIX075_SCGAMMA_AW110_96F` | `phase5_clean_continuation_smoke` | 1 | 96 | `V47_NATIVE_MIX_SCALES=0.75,1.00,1.00` |
| `V53_CONT_SMOKE_H3_SCALESTATE_OVERLAP_SCGAMMA_AW110_96F` | `phase5_clean_continuation_smoke` | 2 | 96 | `V47_TTT_RISK_SOURCE=v19_scale_state`, `alpha=2.0`, no sampling |
| `V53_CONT_REPAIR_H3_SCALESTATE_OVERLAP_SAMPLE2048_AW110_96F` | `phase5_clean_continuation_scale_repair` | 3 | 96 | `V47_TTT_RISK_SOURCE=v19_scale_state`, `alpha=2.0`, `sample_tokens=2048` |
| `V53_CONT_SMOKE_H4_NATIVEMIX050_SCGAMMA_AW110_96F` | `phase5_clean_continuation_native_mix_sweep` | 4 | 96 | `V47_NATIVE_MIX_SCALES=0.50,1.00,1.00` |
| `V53_CONT_SMOKE_H5_NATIVEMIX090_SCGAMMA_AW110_96F` | `phase5_clean_continuation_native_mix_sweep` | 5 | 96 | `V47_NATIVE_MIX_SCALES=0.90,1.00,1.00` |
| `V53_CONT_SMOKE_H6_NATIVEMIX025_SCGAMMA_AW110_96F` | `phase5_clean_continuation_native_mix_sweep` | 5 | 96 | `V47_NATIVE_MIX_SCALES=0.25,1.00,1.00` |
| `V53_CONT_SMOKE_H7_SCALESTATE_ALPHA1_SAMPLE2048_AW110_96F` | `phase5_clean_continuation_scale_repair` | 4 | 96 | `alpha=1.0`, `sample_tokens=2048` |
| `V53_CONT_SMOKE_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110_96F` | `phase5_clean_continuation_scale_repair` | 4 | 96 | `alpha=3.0`, `sample_tokens=2048` |
| `V53_CONT_SMOKE_H9_SCALESTATE_ALPHA2_SAMPLE1024_AW110_96F` | `phase5_clean_continuation_scale_repair` | 5 | 96 | `alpha=2.0`, `sample_tokens=1024` |
| `V53_CONT_SMOKE_H10_SCALESTATE_ALPHA4_SAMPLE2048_AW110_96F` | `phase5_clean_continuation_scale_repair` | 5 | 96 | `alpha=4.0`, `sample_tokens=2048` |
| `V53_CONT_SMOKE_H11_SCALESTATE_ALPHA5_SAMPLE2048_AW110_96F` | `phase5_clean_continuation_scale_repair` | 2 | 96 | `alpha=5.0`, `sample_tokens=2048` |
| `V53_CONT_SMOKE_H12_SCALESTATE_ALPHA4_ONLINESCALE_POSEEMA_AW110_96F` | `phase5_clean_continuation_online_scale` | 2 | 96 | `alpha=4.0`, `sample_tokens=2048`, `ONLINE_SCALE_STATE_MODE=pose_step_ema`, clamp `0.90..1.10` |
| `V53_CONT_SMOKE_H13_SCALESTATE_ALPHA4_ONLINESCALE_OVERLAP_AW110_96F` | `phase5_clean_continuation_online_scale` | 3 | 96 | `alpha=4.0`, `sample_tokens=2048`, `ONLINE_SCALE_STATE_MODE=overlap_step`, clamp `0.90..1.10` |
| `V53_CONT_SCREEN_H2_NATIVEMIX075_SCGAMMA_AW110_384F` | `phase5_clean_continuation_screen` | 2 | 384 | H2 384F |
| `V53_CONT_SCREEN_H3_SCALESTATE_OVERLAP_SAMPLE2048_AW110_384F` | `phase5_clean_continuation_screen` | 3 | 384 | H3 sampled 384F |
| `V53_CONT_SCREEN_H4_NATIVEMIX050_SCGAMMA_AW110_384F` | `phase5_clean_continuation_screen` | 4 | 384 | H4 384F |
| `V53_CONT_SCREEN_H6_NATIVEMIX025_SCGAMMA_AW110_384F` | `phase5_clean_continuation_screen` | 5 | 384 | H6 384F |
| `V53_CONT_SCREEN_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110_384F` | `phase5_clean_continuation_screen` | 4 | 384 | H8 384F |
| `V53_CONT_SCREEN_H9_SCALESTATE_ALPHA2_SAMPLE1024_AW110_384F` | `phase5_clean_continuation_screen` | 5 | 384 | H9 384F |
| `V53_CONT_SCREEN_H10_SCALESTATE_ALPHA4_SAMPLE2048_AW110_384F` | `phase5_clean_continuation_screen` | 5 | 384 | H10 384F |
| `V53_CONT_SCREEN_H13_SCALESTATE_ALPHA4_ONLINESCALE_OVERLAP_AW110_384F` | `phase5_clean_continuation_online_scale_screen` | 2 | 384 | H13 384F |
| `V53_CONT_SCREEN_H14_SCALESTATE_ALPHA3_ONLINESCALE_OVERLAP_AW110_384F` | `phase5_clean_continuation_online_scale_screen` | 3 | 384 | H14 384F |
| `V53_CONT_FULL_H2_NATIVEMIX075_SCGAMMA_AW110` | `phase5_clean_continuation_full` | 2 | full | H2 full |
| `V53_CONT_FULL_H3_SCALESTATE_OVERLAP_SAMPLE2048_AW110` | `phase5_clean_continuation_full` | 3 | full | H3 full |
| `V53_CONT_FULL_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110` | `phase5_clean_continuation_full` | 4 | full | H8 full |
| `V53_CONT_FULL_H13_SCALESTATE_ALPHA4_ONLINESCALE_OVERLAP_AW110` | `phase5_clean_continuation_full` | 2 | full | H13 full |

备注：

```text
H2 384F 启动后，运行中修改过 shell 脚本，bash 进程结束时出现一次 EOF parse message；
但该 run 的 01.txt、kitti_benchmark.log、wall_time_summary.json、run_status.txt 均已落盘，
R10/R11/R22 registry 将其判定为 done/done/384。该事件作为执行噪声记录，不手工改数据。
```

### Continuation smoke 真实结果

| run | ATE | wall min | projected full min | chunk mean | probe TTT mean | smoke gate |
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

### 384F / screen 真实结果

| run | ATE | delta vs C9 | wall min | projected full min | chunk mean | probe TTT mean | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| H2 native_mix0.75 | `34.17316081147195` | `0.41021870855310283` | `9.533333333333333` | `25.876190476190473` | `38.512014167649404` | `5.296329004423959` | full |
| H3 alpha2 sample2048 | `33.8902169923413` | `0.12727488942245202` | `9.4` | `25.514285714285716` | `37.78704789706639` | `5.885575481823513` | full |
| H4 native_mix0.50 | `34.63753779356436` | `0.8745956906455135` | `9.283333333333333` | `25.19761904761905` | `37.360947711127146` | `5.0753229175295145` | stop |
| H6 native_mix0.25 | `35.17743595421384` | `1.414493851294992` | `9.35` | `25.378571428571426` | `37.707017404692515` | `5.1447978019714355` | stop |
| H8 alpha3 sample2048 | `33.77653108321121` | `0.013588980292361441` | `9.183333333333334` | `24.926190476190474` | `37.02048448153904` | `5.635223065103803` | full |
| H9 alpha2 sample1024 | `33.91375953077707` | `0.15081742785822172` | `9.25` | `25.107142857142858` | `37.37098090989249` | `5.221473370279584` | stop |
| H10 alpha4 sample2048 | `33.82920983193383` | `0.06626772901498157` | `9.366666666666667` | `25.423809523809528` | `37.80639869826181` | `5.886151092393058` | stop |
| H13 alpha4 online overlap | `33.82920983193383` | `0.06626772901498157` | `9.316666666666666` | `25.28809523809524` | `37.61891397408077` | `5.871012875011989` | full as full-sequence repair test |
| H14 alpha3 online overlap | `33.77653108321121` | `0.013588980292361441` | `10.183333333333334` | `27.640476190476193` | `41.02354213169643` | `6.220972163336618` | stop: same ATE as H8, worse runtime |

### Continuation full 真实结果

| run | status | frames | hmc rows | ATE | delta vs C9 | wall min | chunk mean | probe TTT mean | no-chunk | manual % | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `V53_CONT_FULL_H2_NATIVEMIX075_SCGAMMA_AW110` | done | 1101 | 38 | `36.27180630457207` | `2.5088642016532177` | `26.366666666666667` | `40.14183219483024` | `5.4306485339214925` | true | true | runtime pass, ATE fail |
| `V53_CONT_FULL_H3_SCALESTATE_OVERLAP_SAMPLE2048_AW110` | done | 1101 | 38 | `36.481771395468634` | `2.718829292549785` | `27.033333333333335` | `41.15486602406753` | `6.266185089161522` | true | true | runtime pass, ATE fail |
| `V53_CONT_FULL_H8_SCALESTATE_ALPHA3_SAMPLE2048_AW110` | done | 1101 | 38 | `36.30791120013488` | `2.5449690972160326` | `27.3` | `41.552999684685155` | `6.262588739395142` | true | true | runtime pass, ATE fail |
| `V53_CONT_FULL_H13_SCALESTATE_ALPHA4_ONLINESCALE_OVERLAP_AW110` | done | 1101 | 38 | `36.346746515095305` | `2.5838044121764554` | `26.333333333333332` | `40.04427994552412` | `6.1080278471896525` | true | true | runtime pass, ATE fail |

Continuation full 判定：

```text
best continuation full = V53_CONT_FULL_H2_NATIVEMIX075_SCGAMMA_AW110
best continuation full ATE = 36.27180630457207
best continuation delta vs C9 = +2.5088642016532177
best overall v53 full remains V53_FULL_A_SCGAMMASPLIT_AW110
best overall v53 full ATE = 35.912858666859364
progress-pass threshold = 35.30
success_progress_pass = False
```

### 新报告目录

最终 continuation 报告：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_final_continuation_with_online_scale
```

补充 runtime 报告：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_full_runtime
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R22_online_scale_screen_runtime
```

关键中间报告：

```text
report_R8_continuation_smoke
report_R9_continuation_after_smokes
report_R11_screen_h3_h4
report_R14_h8_h9_screen
report_R20_final_continuation
```

### Continuation 最终判定

```text
sampled scale-state 修复了 H3 full-token runtime blocker。
native_mix、scale-state alpha、sample_tokens、online_scale overlap 都未产生 full progress-pass。
H8/H14 384F 曾达到 33.77653108321121，几乎等于 C9，但 full 仍回落到 36.30791120013488。
这证明 384F early screen 不能作为 full success 代理。
按计划 §9.6，需要停止继续小扫，进入 full-sequence failure autopsy / 新计划。
```

## 2026-06-09 08:15--09:29 +08：full-sequence autopsy + GPU 2/3/4/5 704F screen

用户再次提醒 GPU `2,3,4,5` 空闲。本阶段继续按 v53 计划 §9.6 执行：

```text
不使用 SWA。
不使用 absolute chunk-id gamma/replay/EMA map。
不使用手工 tri-replay 百分比。
优先审计 full-sequence drift、commit/branch/layer action 和 scale-state controller。
```

### 新增工具

```text
tools/v53_full_sequence_drift_autopsy.py
```

用途：只读取落盘 artifacts，生成 by-run/by-chunk/segment/rolling-window/prefix-diff CSV/JSON/MD 和 timeline png；不生成新实验数据，不补写缺失指标。

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/v53_full_sequence_drift_autopsy.py
```

### R1 full-sequence autopsy

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
OUT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R1
mkdir -p "$OUT"
$PY -m py_compile tools/v53_full_sequence_drift_autopsy.py > "$OUT/py_compile.log" 2>&1
$PY tools/v53_full_sequence_drift_autopsy.py --out-dir "$OUT" 2>&1 | tee "$OUT/run.log"
```

第一次运行时曾在创建 out-dir 前使用 `tee`，导致 shell 退出码为 1；随后创建目录后重跑，退出码 0。有效结果采用重跑后的 `report_R1`。

关键结果：

```text
H8 screen/full prefix consistency:
  common frames = 384
  translation RMSE = 0.006200570735239549
  translation max = 0.08722549334340368
  rotation fro mean = 0.0001275102135586111

C9_REF:
  full ATE = 33.76294210291885
  prefix384 self ATE = 33.37157053761927
  seg0/seg1/seg2 rmse = 46.39902061769452 / 34.656626116071514 / 11.034547189236443

V53_A_BEST:
  full ATE = 35.912858666859364
  prefix384 self ATE = 33.83297324070255
  seg0/seg1/seg2 rmse = 45.04645244540594 / 40.85251832537275 / 16.81711362434614

V53_H8_FULL:
  full ATE = 36.30791120013488
  prefix384 self ATE = 33.80040449316786
  seg0/seg1/seg2 rmse = 45.55351876783694 / 41.24700757795206 / 17.07735967628986

V53_H13_FULL:
  full ATE = 36.346746515095305
  prefix384 self ATE = 33.85379625939145
  seg0/seg1/seg2 rmse = 45.84565695969239 / 40.97596351054843 / 17.06974111460667
```

判断：

```text
H8 384F screen 与 H8 full 的前 384 帧几乎一致。
384F 好结果不是 screen/full prefix artifact。
主要失败来自 384 帧之后的 full-sequence drift。
```

### H15-H18 inverse-scale 704F screen

共同命令模板：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9
BASE=$ROOT/phase6_inverse_scale_screen/rollouts
V47_RESULT_ROOT="$ROOT" \
V47_ROLLOUT_BASE="$BASE" \
V47_END_FRAME=704 \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh <GPU> AW110_FRAME_ADAPTIVE_TTT <RUN_NAME>
```

报告：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R23_phase6_inverse_scale_screen
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R2_inverse_scale_screen
```

结果：

| run | GPU | ATE | wall min | projected full min | chunk mean | TTT mean |
|---|---:|---:|---:|---:|---:|---:|
| H15 alpha3 inverse | 2 | `40.3156963118558` | `17.483333333333334` | `26.574666666666666` | `40.284730968475344` | `6.245894966125488` |
| H16 alpha4 inverse | 3 | `40.39964760231925` | `18.133333333333333` | `27.56266666666667` | `41.64413722038269` | `6.302977046966553` |
| H17 alpha3 inverse up-only | 4 | `40.3156963118558` | `17.85` | `27.132` | `41.03594367980957` | `6.161382150650025` |
| H18 residual inverse | 5 | `39.96013035108` | `16.55` | `25.156` | `37.95244373321533` | `5.013090181350708` |

判定：

```text
A-best 704F self ATE = 39.92846746159833。
H18 最接近，但 ATE = 39.96013035108，仍略差于 A-best 704F。
不升级 full。
```

### H19-H26 commit/branch/layer 704F screen

H23 第一次用后台子 shell 提交，只写出配置和 START，未进入 pipeline；该尝试不计入结果。随后用阻塞式会话重跑 H23，真实完成。

共同约束：

```text
V47_END_FRAME=704
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split
V47_TTT_RISK_SOURCE=ttt_residual_x_dg
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0
```

报告：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R24_phase6_commit_layer_screen
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R3_commit_layer_screen
```

结果：

| run | GPU | ATE | wall min | projected full min | chunk mean | TTT mean | clean audit |
|---|---:|---:|---:|---:|---:|---:|---|
| H19 statecommit075 | 2 | `40.185368766362735` | `16.716666666666665` | `25.409333333333333` | `38.33181634902954` | `5.228257350921631` | no-chunk/manual pass |
| H20 olddecay q90 | 3 | `40.17659973494846` | `17.95` | `27.284` | `41.30676031112671` | `6.984069814682007` | no-chunk/manual pass |
| H21 native2cand q90 | 4 | `40.158801504115694` | `18.916666666666668` | `28.753333333333334` | `43.4792165851593` | `6.960243282318115` | no-chunk/manual pass |
| H22 layergamma 0/8/17 | 5 | `39.96013035108` | `16.083333333333332` | `24.446666666666665` | `36.96695454597473` | `4.7673742389678955` | no-chunk/manual pass |
| H23 branchall | 5 | `45.219899861528525` | `16.3` | `24.776` | `37.418483638763426` | `5.160757207870484` | no-chunk/manual pass |
| H24 branchall statecommit075 | 2 | `44.73271874335227` | `16.15` | `24.548` | `37.05656575202942` | `5.3126442432403564` | no-chunk/manual pass |
| H25 branchall olddecay q90 | 3 | `43.34215610485726` | `18.283333333333335` | `27.790666666666667` | `42.046777153015135` | `7.290578470230103` | no-chunk/manual pass |
| H26 branchall native2cand q90 | 4 | `44.38401578166757` | `17.05` | `25.916` | `39.154375534057614` | `6.925587244033814` | no-chunk/manual pass |

R3 segment 摘要：

```text
C9_REF seg0/seg1/seg2 = 46.39902061769452 / 34.656626116071514 / 11.034547189236443
V53_A_BEST seg0/seg1/seg2 = 45.04645244540594 / 40.85251832537275 / 16.81711362434614
H22 seg0/seg1/seg2 = 46.17639098579641 / 30.899712092537136 / 30.223477905413752
H23 branchall seg0/seg1/seg2 = 52.17797929050439 / 35.08404124288639 / 35.88080647300726
H25 branchall olddecay seg0/seg1/seg2 = 50.264211654270225 / 33.11354573466735 / 38.161421805556
```

审计限制：

```text
H19/H20/H21/H24/H25/H26 的 commit_filter 配置已写入 effective_config.yaml 和 hmc_config.yaml。
但落盘 summary 没有 commit_filter debug 字段；R24 registry 的 commit_filter_debug_seen=0。
因此本阶段不能 claim commit_filter activation rate 或 per-layer commit-filter 生效细节。
只报告这些配置下的轨迹结果和该审计缺口。
```

判定：

```text
best 704F in this batch = H22 layergamma 0/8/17
H22 ATE = 39.96013035108
A-best 704F self ATE = 39.92846746159833
H22 did not beat A-best 704F; do not promote to full.

branch-all variants H23-H26 ATE range = 43.34215610485726--45.219899861528525
branch-all clearly worsens prefix/early behavior.

success_progress_pass = False
new_full_run_launched = False
next blocker = commit-filter observability/activation audit gap + need better trajectory-state controller
```

### 本阶段证据链

```text
Code:
  tools/v53_full_sequence_drift_autopsy.py

Autopsy:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R1
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R2_inverse_scale_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_full_sequence_autopsy/report_R3_commit_layer_screen

Reports:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R23_phase6_inverse_scale_screen
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R24_phase6_commit_layer_screen

Rollout roots:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_inverse_scale_screen/rollouts
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase6_commit_layer_screen/rollouts
```

## 2026-06-09 09:30--11:05 +08：Phase7 commit-filter 可观测性修复 + sc-gamma layergamma 链路修复 + full 验证

本阶段继续按 v53 clean 目标推进。用户质疑“忙了三小时是否无收获”后，本阶段重新审计 Phase6 的两个疑点：

```text
1. commit-filter 配置存在但 debug 不可见，不能 claim activation。
2. H22/H30 的 layergamma 配置看似生效，但轨迹与无 layergamma run 完全一致。
```

### 代码修改与验证

修改：

```text
run_pipeline_abc_v2.py
  - 新增 no-GT path_length_ema / path_length_prev trajectory scale-state proxy。
  - 在 hybrid debug summary 中落盘 commit-filter top-level 和 per-layer summary 字段。

tools/v53_experiment_report.py
  - 增强 commit_filter_stats，统计 active mode/debug seen/applied/risk/scale/modes。

loger/pipeline/ttt_write_controller.py
  - 修复 adaptive_writer_sc_gamma_split 中 layer/branch gamma 未进入 sc-gamma rho 的问题。
  - 原先 sc-gamma 分支固定 rho=0.005，导致 V47_TTT_LAYER_GAMMAS 只写入 config，不改变轨迹。
  - 修复后若当前 layer/branch 有正 gamma，则作为 sc-gamma rho；否则保持默认 rho=0.005。
```

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  run_pipeline_abc_v2.py tools/v53_experiment_report.py loger/pipeline/ttt_write_controller.py
```

结果：

```text
py_compile pass
```

### Phase7 H27-H30：trajectory path proxy + commit-filter debug

共同配置：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9
BASE=$ROOT/phase7_trajectory_state_screen/rollouts
V47_RESULT_ROOT="$ROOT"
V47_ROLLOUT_BASE="$BASE"
V47_END_FRAME=704
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0
tools/run_v47_adaptive_ttt_writer_candidate.sh <GPU> AW110_FRAME_ADAPTIVE_TTT <RUN_NAME>
```

H27-H29 用 `V47_TTT_RISK_SOURCE=trajectory_scale_state` 和 path-length proxy；H30 用旧 layergamma + statecommit 复核 commit-filter debug。

报告：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R25_phase7_trajectory_state_screen
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_full_sequence_autopsy/report_R4_trajectory_state_screen
```

结果：

| run | GPU | ATE 704F | wall min | projected full min | chunk mean | TTT mean | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| H27 pathlen ema alpha3 | 2 | `40.897348622677924` | `16.516666666666666` | `25.10533333333333` | `37.85437040328979` | `5.802112550735473` | 失败 |
| H28 pathlen ema alpha4 | 3 | `40.830079619661` | `17.766666666666666` | `27.005333333333333` | `40.85793343544006` | `6.074894361495971` | 失败 |
| H29 pathlen prev alpha3 | 4 | `40.81389341714244` | `16.533333333333335` | `25.130666666666666` | `37.96012343406677` | `5.639145746231079` | 失败 |
| H30 old layergamma + statecommit075 | 5 | `40.185368766362735` | `16.783333333333335` | `25.51066666666667` | `38.56237456321716` | `5.138150844573975` | 与 H19 逐行相同；不算新优化 |

H30 commit-filter debug：

```text
commit_filter_debug_seen = 25
commit_filter_active_mode_seen = 25
commit_filter_applied_debug_rows = 25
commit_filter_activation_rate_mean = 0.4333333333333333
commit_filter_scale_mean = 0.8916666666666666
commit_filter_modes_seen = state_conditioned_commit
```

额外审计：

```text
H30 01.txt 与旧 H19 01.txt 完全逐行相同。
H22 01.txt 与 H18 01.txt 完全逐行相同。
因此 Phase6 文中“selected-layer gamma”解释需要更正：
  旧 H22/H30 的 layergamma 配置写入了 config，但没有改变 sc-gamma 行为/轨迹。
```

### Phase7 H31-H34：sc-gamma layergamma 链路修复后 704F sweep

共同命令模板：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9
BASE=$ROOT/phase7_layergamma_fix_screen/rollouts
V47_RESULT_ROOT="$ROOT" \
V47_ROLLOUT_BASE="$BASE" \
V47_END_FRAME=704 \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_LAYER_GAMMAS='<rho map>' \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh <GPU> AW110_FRAME_ADAPTIVE_TTT <RUN_NAME>
```

报告：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R26_phase7_layergamma_fix_screen
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_full_sequence_autopsy/report_R5_layergamma_fix_screen
```

结果：

| run | rho map | GPU | ATE 704F | wall min | projected full min | chunk mean | TTT mean |
|---|---|---:|---:|---:|---:|---:|---:|
| H31 | `0/8/17:0.003` | 2 | `39.90181810063874` | `16.533333333333335` | `25.130666666666666` | `37.97590750694275` | `4.856322240829468` |
| H32 | `0/8/17:0.006` | 3 | `39.9844379268327` | `16.766666666666666` | `25.485333333333337` | `38.35878821372986` | `5.174338655471802` |
| H33 | `0/8/17:0.009` | 4 | `39.86624170202318` | `16.583333333333332` | `25.206666666666663` | `37.99921131134033` | `5.168029804229736` |
| H34 | `0/8/17:0.012` | 5 | `39.965776688405064` | `17.55` | `26.676` | `40.30042562484741` | `5.258063020706177` |

轨迹有效性审计：

```text
H31/H32/H33/H34 的 01.txt 均不再与 H18 逐行相同。
first_diff 均从 frame 29 出现。
说明 layergamma 修复后确实改变了轨迹，不再是无效配置。
```

判定：

```text
A-best 704F self ATE = 39.92846746159833
H31 = 39.90181810063874, beat A-best 704F
H33 = 39.86624170202318, beat A-best 704F
H33 当时最优，因此升级 full；H31 同时升级 full。
```

### Phase7 H35-H37：rho 细化

H35/H36:

```text
report_R27_phase7_layergamma_refine_screen
phase7_full_sequence_autopsy/report_R6_layergamma_refine_screen
```

H37:

```text
report_R29_phase7_layergamma_refine2_screen
phase7_full_sequence_autopsy/report_R8_layergamma_refine2_screen
```

结果：

| run | rho map | GPU | ATE 704F | wall min | projected full min | chunk mean | TTT mean | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| H35 | `0/8/17:0.0075` | 4 | `39.798247720485634` | `16.383333333333333` | `24.90266666666667` | `37.50150881767273` | `4.9218727684021` | best 704F，升级 full |
| H36 | `0/8/17:0.0105` | 5 | `39.95144397608373` | `16.35` | `24.852` | `37.562337217330935` | `4.873052206039429` | 不升级 |
| H37 | `0/8/17:0.007` | 5 | `39.91245830855871` | `15.95` | `24.244000000000003` | `36.56959966659546` | `4.8340124416351316` | 不升级 |

### Phase7 full：H31/H33/H35

共同命令模板：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9
BASE=$ROOT/phase7_layergamma_fix_full/rollouts
V47_RESULT_ROOT="$ROOT" \
V47_ROLLOUT_BASE="$BASE" \
V47_END_FRAME=10000 \
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_LAYER_GAMMAS='<rho map>' \
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0 \
tools/run_v47_adaptive_ttt_writer_candidate.sh <GPU> AW110_FRAME_ADAPTIVE_TTT <RUN_NAME>
```

报告：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R30_phase7_layergamma_fix_full_final
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase7_full_sequence_autopsy/report_R9_layergamma_fix_full_final
```

结果：

| run | rho map | GPU | frames | ATE full | wall min | chunk mean | TTT mean | full runtime gate | no-chunk | manual % |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| H31 full | `0/8/17:0.003` | 2 | `1101` | `35.83167463692937` | `24.983333333333334` | `38.06106244262896` | `5.019238898628636` | pass | pass | pass |
| H33 full | `0/8/17:0.009` | 3 | `1101` | `35.769087844821314` | `26.933333333333334` | `40.951174811313024` | `5.409016904078032` | pass | pass | pass |
| H35 full | `0/8/17:0.0075` | 4 | `1101` | `35.74089695811434` | `24.783333333333335` | `37.640955912439445` | `4.905587773574026` | pass | pass | pass |

R9 segment 摘要：

```text
C9_REF seg0/seg1/seg2 = 46.39902061769452 / 34.656626116071514 / 11.034547189236443
V53_A_BEST seg0/seg1/seg2 = 45.04645244540594 / 40.85251832537275 / 16.81711362434614
H31 seg0/seg1/seg2 = 45.00057513987618 / 40.688663174256426 / 16.77224281258083
H33 seg0/seg1/seg2 = 44.931824306935155 / 40.66141901286172 / 16.6334465944543
H35 seg0/seg1/seg2 = 44.943948305639466 / 40.48296121701416 / 16.778093400455763
```

最终判定：

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

## 2026-06-09 Phase8：layer/branch/rho 细化 704F sweep

目的：

```text
Phase7 H35 full 是当前 v53 best，但仍未达到 progress-pass。
H35 使用 layer 0/8/17 同 rho=0.0075、branch0。
本阶段检查 H35 的收益来自哪些 layer/branch，并确认继续细扫 rho/asym 是否还有 704F 提升。
```

共同约束：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9
V47_RESULT_ROOT=$ROOT
V47_END_FRAME=704
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split
V47_TTT_RISK_SOURCE=ttt_residual_x_dg
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0
tools/run_v47_adaptive_ttt_writer_candidate.sh <GPU> AW110_FRAME_ADAPTIVE_TTT <RUN_NAME>
```

并行资源使用：

```text
第一批使用 GPU 2/3/4/5 跑 H38-H41。
用户提醒 GPU 0/1 空闲后，补开 H42-H43。
H38-H43 完成后，GPU 2/3/4/5 继续跑 H44-H47。
H42-H43 完成后，GPU 0/1 继续跑 H48-H49。
H44-H47 完成后，GPU 2/3/4/5 继续跑 H50-H53。
本阶段最多 6-way 704F 并行。
```

报告：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R31_phase8_layer_specific_screen
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R32_phase8_branch_screen
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R33_phase8_layer817_refine_screen
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R34_phase8_layer817_asym_screen
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase8_full_sequence_autopsy/report_R10_phase8_screen
```

### C9 chunk-wise 配置复核

为回答“C9 是否在某些 chunk 自动关闭 TTT”，查到最近 C9 repeat 配置：

```text
artifact:
  results/kitti01_hmc_v2/acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay/phase0_hard_gate/rollouts/V45_P0_C9_REPEAT/effective_config.yaml

read_beta_frame_chunks:
  5:4.85,6:4.85,7:4.85,8:4.85,9:4.85,10:4.25,11:4.25,12:4.25,16:4.25

ttt_write_gradient_reversal_chunk_gammas:
  5:0.005,6:0.005,7:0.005,8:0.005,9:0.005,10:0.003,11:0.003,12:0.003,16:0.0003

ttt_write_tri_replay_chunk_params:
  5-12: 0.35/0.12/0.85
  16: 0.35/0.08/0.85

ttt_write_commit_ema_alpha:
  0.5

ttt_write_commit_ema_chunks:
  5,6
```

判断：

```text
C9 不是“自动关闭 TTT”。
它是人工 chunk-wise 控制 read beta、TTT correction gamma、tri replay params 和 commit EMA。
clean 版不能照抄 chunk id，需要用状态变量模拟这些时机。
```

### Phase8 结果表

| run | 配置 | ATE 704F | Rot | wall min | chunk mean | TTT mean | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| H38 | layer0 rho0.0075 | `39.960135` | `6.263346` | `16.633333333333333` | `38.20559416770935` | `5.0138212585449216` | 不升级 |
| H39 | layer8 rho0.0075 | `39.859164` | `6.190298` | `17.583333333333332` | `40.30065856933594` | `5.506328716278076` | 不升级 |
| H40 | layer17 rho0.0075 | `39.965632` | `6.162321` | `17.5` | `40.19906923294067` | `5.392944402694702` | 不升级 |
| H41 | layer8+17 rho0.0075 | `39.798242` | `6.264927` | `16.583333333333332` | `37.97287831306458` | `5.052609062194824` | 等同 H35 704F；不升 full |
| H42 | layer0+8 rho0.0075 | `39.859164` | `6.190298` | `17.716666666666665` | `40.55417562484741` | `5.259451179504395` | 不升级 |
| H43 | layer0+17 rho0.0075 | `39.965632` | `6.162321` | `17.133333333333333` | `39.19534448623657` | `5.148021402359009` | 不升级 |
| H44 | branch0 0/8/17 rho0.0075 | `39.798242` | `6.264927` | `17.6` | `39.1620478439331` | `5.123666524887085` | 等同 H35/H41；不升 full |
| H45 | branch1 0/8/17 rho0.0075 | `41.896211` | `5.564021` | `17.45` | `38.7483429145813` | `3.2366672801971434` | manual percentage audit false；失败 |
| H46 | branch2 0/8/17 rho0.0075 | `41.896211` | `5.564021` | `16.8` | `37.27957184791565` | `3.2869395065307616` | manual percentage audit false；失败 |
| H47 | branchall 0/8/17 rho0.0075 | `39.798242` | `6.264927` | `16.7` | `37.10262540817261` | `4.83673641204834` | 等同 H35/H41；不升 full |
| H48 | layer8+17 rho0.0065 | `39.933831` | `6.261888` | `17.95` | `41.27058617591858` | `5.566148166656494` | 不升级 |
| H49 | layer8+17 rho0.0085 | `39.986971` | `6.282328` | `17.15` | `39.30914575576782` | `5.1856677532196045` | 不升级 |
| H50 | layer8+17 rho0.0055 | `39.891210` | `6.174457` | `18.65` | `41.887183818817135` | `5.6431180858612064` | 不升级 |
| H51 | layer8+17 rho0.0095 | `39.896061` | `6.124135` | `17.183333333333334` | `38.435900869369505` | `5.135310764312744` | 不升级 |
| H52 | layer8 rho0.0065, layer17 rho0.0085 | `39.871933` | `6.147249` | `16.983333333333334` | `37.94002234458923` | `5.009374647140503` | 不升级 |
| H53 | layer8 rho0.0085, layer17 rho0.0065 | `39.942017` | `6.307412` | `18.033333333333335` | `40.4485304069519` | `5.357725143432617` | 不升级 |

R10 autopsy 摘要：

```text
H35_704 ATE = 39.798248
H41_L8_17 ATE = 39.798248
H44_BRANCH0 ATE = 39.798248
H52_ASYM ATE = 39.871933

H35/H41/H44 segment:
  seg0 = 46.051213
  seg1 = 30.667454
  seg2 = 29.647953

H52 segment:
  seg0 = 46.059087
  seg1 = 30.859968
  seg2 = 30.113921
```

### Phase8 判定

```text
best_phase8_704F = H41/H44/H47
best_phase8_704F_ATE = 39.798242
previous_best_704F = H35
previous_best_704F_ATE = 39.798247720485634

The numeric difference is around 5e-6 and R10 recomputes all as 39.798248.
This is not meaningful enough to justify a full promotion.

full_promotion = False
success_progress_pass = False
best_overall_v53_full remains H35 full
best_overall_v53_full_ATE remains 35.74089695811434
```

审计说明：

```text
H45/H46 branch1/branch2 单独 run 的 no-chunk policy pass，
但 manual percentage audit false / split_debug_count=0，
且 ATE=41.896211 明显更差。
它们只作为失败/审计负例，不作为有效 clean 候选。
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

## Phase9：layer8+17 sc-gamma + commit 时序模拟

时间：2026-06-09 12:19-12:46（Asia/Singapore）

目的：

```text
Phase8 证明 layer/branch/rho 小扫已到平台。
Phase9 继续按计划测试“能否用 no-chunk state-conditioned commit / risk commit filter
模拟 C9 的少数 chunk commit/时序动作”。
全部 run 仍禁止 read_beta_frame_chunks、tri gamma chunk map、tri replay chunk params、
commit_ema_chunks、native_mix_chunks、semantic_action_active_chunks。
```

### Phase9 preflight / blocker 修复

真实 blocker：

```text
run_pipeline_abc_v2.py 曾处于 deleted 状态，导致 launcher 会找不到主入口。
修复动作：
  git show HEAD:run_pipeline_abc_v2.py > run_pipeline_abc_v2.py

说明：
  这是从 HEAD 恢复缺失入口文件，不是回滚实验结果。
  由于文件较大，当时用了 shell 重建文件；后续增量修复均用 apply_patch。
```

恢复后出现入口兼容 blocker：

```text
H54-H59 第一次启动失败：
  error: --ttt_write_commit_filter_mode invalid choice: state_conditioned_commit

修复：
  run_pipeline_abc_v2.py 新增/恢复 CLI：
    --ttt_write_native_mix_chunks
    --ttt_write_tri_replay_role_mode
    --ttt_write_gradient_reversal_transient_apply_scale
    --ttt_write_commit_ema_chunks
    --ttt_write_commit_filter_mode choices: native2candidate_by_risk/state_conditioned_commit 等
    --ttt_write_scale_state_*
    --online_scale_state_*
    --v11_projection_*
    --probe_cache_*
    --empty_cuda_cache_each_chunk

  run_pipeline_abc_v2.py 同步把 native_mix_chunks / tri_replay_role_mode /
  transient_apply_scale / commit_ema_chunks / scale_state_* 传入 HybridMemoryController。

H54-H59 第二次启动失败：
  error: unrecognized arguments: --v32_semantic_cue_* --read_beta_frame_chunks
  --enable_context_source_skip ... --semantic_role_policy ...
  --v29c_masklet_* --v36_synthetic_*

修复：
  run_pipeline_abc_v2.py 增加这些历史诊断/干预参数的 no-op compatibility parser。
  本轮这些参数取值均为 off/none/empty，不改变 clean 实验行为。
```

验证：

```text
python -m py_compile run_pipeline_abc_v2.py loger/pipeline/hybrid_memory_controller.py loger/pipeline/ttt_write_controller.py
python run_pipeline_abc_v2.py --help | rg "state_conditioned|native2candidate|scale_state|v32_semantic|context_source_skip"
```

报告工具修复：

```text
tools/v53_experiment_report.py 原本只扫描 timing_summary.json；
Phase9 run 只落 wall_time_summary.json + 01.log + 01.txt。
修复：
  _iter_run_dirs 支持 run_status.txt + 01.txt 的 rollout。
  _timing_stats 从 01.log 解析 Pass1/StageB/StageD/Pass2 done-in 秒数；
  解析不到的 probe_ttt_write_seconds 保持 NA，不补造。
  _commit_filter_stats 增加扫描 01.log 中的 Python dict debug，
  修复 commit_filter_debug_seen 误报 0 的问题。
```

### Phase9 命令模板

共同环境：

```bash
ROOT=results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9
BASE=$ROOT/phase9_commit_layer817_screen/rollouts
V47_RESULT_ROOT=$ROOT
V47_ROLLOUT_BASE=$BASE
V47_END_FRAME=704
V47_TTT_ROLE_MODE=adaptive_writer_sc_gamma_split
V47_TTT_RISK_SOURCE=ttt_residual_x_dg
V47_TTT_BRANCH_MASK=0
V47_TTT_LAYER_GAMMAS='8:0.0075,17:0.0075'
V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0
bash tools/run_v47_adaptive_ttt_writer_candidate.sh <GPU> AW110_FRAME_ADAPTIVE_TTT <RUN_NAME>
```

具体并行分配：

| Run | GPU | 额外环境 |
|---|---:|---|
| H54 `V53_PHASE9_SCREEN_H54_L817_SCOMMIT050_704F` | 0 | `V47_TTT_COMMIT_FILTER_MODE=state_conditioned_commit V47_TTT_COMMIT_FILTER_MIN=0.50 V47_TTT_COMMIT_FILTER_MAX=1.0` |
| H55 `V53_PHASE9_SCREEN_H55_L817_SCOMMIT065_704F` | 1 | `V47_TTT_COMMIT_FILTER_MODE=state_conditioned_commit V47_TTT_COMMIT_FILTER_MIN=0.65 V47_TTT_COMMIT_FILTER_MAX=1.0` |
| H56 `V53_PHASE9_SCREEN_H56_L817_SCOMMIT075_704F` | 2 | `V47_TTT_COMMIT_FILTER_MODE=state_conditioned_commit V47_TTT_COMMIT_FILTER_MIN=0.75 V47_TTT_COMMIT_FILTER_MAX=1.0` |
| H57 `V53_PHASE9_SCREEN_H57_L817_OLDDECAY_RESXDG_MEAN_704F` | 3 | `V47_TTT_COMMIT_FILTER_MODE=old_decay_by_risk V47_TTT_COMMIT_FILTER_RISK_SOURCE=ttt_residual_x_dg V47_TTT_COMMIT_FILTER_STAT=mean V47_TTT_COMMIT_FILTER_BASE=1.0 V47_TTT_COMMIT_FILTER_GAIN=0.25 V47_TTT_COMMIT_FILTER_MIN=0.75 V47_TTT_COMMIT_FILTER_MAX=1.0` |
| H58 `V53_PHASE9_SCREEN_H58_L817_OLDDECAY_RESXDG_Q90_704F` | 4 | `V47_TTT_COMMIT_FILTER_MODE=old_decay_by_risk V47_TTT_COMMIT_FILTER_RISK_SOURCE=ttt_residual_x_dg V47_TTT_COMMIT_FILTER_STAT=q90 V47_TTT_COMMIT_FILTER_BASE=1.0 V47_TTT_COMMIT_FILTER_GAIN=0.25 V47_TTT_COMMIT_FILTER_MIN=0.75 V47_TTT_COMMIT_FILTER_MAX=1.0` |
| H59 `V53_PHASE9_SCREEN_H59_L817_NATIVE2CAND_RESXDG_MEAN_704F` | 5 | `V47_TTT_COMMIT_FILTER_MODE=native2candidate_by_risk V47_TTT_COMMIT_FILTER_RISK_SOURCE=ttt_residual_x_dg V47_TTT_COMMIT_FILTER_STAT=mean V47_TTT_COMMIT_FILTER_BASE=0.75 V47_TTT_COMMIT_FILTER_GAIN=0.25 V47_TTT_COMMIT_FILTER_MIN=0.75 V47_TTT_COMMIT_FILTER_MAX=1.0` |

### Phase9 运行结果

Report：

```text
results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R35_phase9_commit_layer817_screen
```

| Run | ATE 704F | Rot | wall min | chunk mean from log | pass2 mean | commit filter mode | applied rows | scale mean | no-chunk | manual % |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| H54 | `40.3783773752881` | `2.8965180125128684` | `16.966666666666665` | `31.9072` | `10.650000000000002` | `state_conditioned_commit` | `25` | `0.789258757710796` | True | True |
| H55 | `40.24293894797938` | `2.9142385463126335` | `16.333333333333332` | `30.8224` | `10.242` | `state_conditioned_commit` | `25` | `0.8522222222222224` | True | True |
| H56 | `40.07617220766744` | `2.8707779660815924` | `17.433333333333334` | `32.8824` | `10.8748` | `state_conditioned_commit` | `25` | `0.8938888888888888` | True | True |
| H57 | `39.953848909964535` | `2.8879959100201225` | `17.7` | `31.957200000000004` | `10.405200000000002` | `old_decay_by_risk` | `25` | `0.9646988984745823` | True | True |
| H58 | `40.216765335248425` | `2.9071219475031573` | `17.833333333333332` | `32.2676` | `10.365599999999999` | `old_decay_by_risk` | `25` | `0.8813276774436235` | True | True |
| H59 | `40.29482875365769` | `2.87096569799561` | `18.566666666666666` | `33.8516` | `11.236400000000001` | `native2candidate_by_risk` | `25` | `0.7852583289601737` | True | True |

对照：

```text
H35/H41/H44 704F = 39.798247720485634 / 39.798242 / 39.798242
Phase9 best = H57 = 39.953848909964535
Phase9 best is worse than H35 704F by about 0.155601189478901
full promotion = False
```

### Phase9 判定

```text
success_progress_pass = False
full_promotion = False
best_overall_v53_full remains V53_PHASE7_FULL_H35_LAYERGAMMAFIX_RHO0075
best_overall_v53_full_ATE remains 35.74089695811434
```

本轮结论：

```text
commit filter 确实接入并生效：
  H54-H56 state_conditioned_commit activation_rate_mean about 0.422-0.424；
  all Phase9 runs commit_filter_debug_seen=25, applied rows=25。

但所有 Phase9 704F 都差于 H35/H41。
所以“简单 state-conditioned commit / risk decay”没有模拟出 C9 少数 chunk 时序收益。
```

证据链：

```text
Rollouts:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H54_L817_SCOMMIT050_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H55_L817_SCOMMIT065_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H56_L817_SCOMMIT075_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H57_L817_OLDDECAY_RESXDG_MEAN_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H58_L817_OLDDECAY_RESXDG_Q90_704F
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/phase9_commit_layer817_screen/rollouts/V53_PHASE9_SCREEN_H59_L817_NATIVE2CAND_RESXDG_MEAN_704F

Report:
  results/kitti01_hmc_v2/acl2_v53_fast_noswa_c9clean_adaptive_ttt_to_c9/report_R35_phase9_commit_layer817_screen

Logs:
  logs/stream_v53_phase9_h54_l817_scommit050_704f.log        # first rc=2 invalid choice
  logs/stream_v53_phase9_h54_l817_scommit050_704f_r2.log     # second rc=2 missing compat args
  logs/stream_v53_phase9_h54_l817_scommit050_704f_r3.log     # completed
  logs/stream_v53_phase9_h55_l817_scommit065_704f_r3.log
  logs/stream_v53_phase9_h56_l817_scommit075_704f_r3.log
  logs/stream_v53_phase9_h57_l817_olddecay_resxdg_mean_704f_r3.log
  logs/stream_v53_phase9_h58_l817_olddecay_resxdg_q90_704f_r3.log
  logs/stream_v53_phase9_h59_l817_native2cand_resxdg_mean_704f_r3.log
  logs/stream_v53_phase9_report_R35_r3.log
```
