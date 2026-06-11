# ACL2 v52 C9Clean AdaptiveTTT SemanticGeometry 执行日志

日期：2026-06-09（Asia/Singapore）  
计划文件：`docs/ACL2_v52_C9Clean_AdaptiveTTT_SemanticGeometry_Experiment_Plan.md`  
结果根目录：`results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/`

本日志只记录真实执行过的命令、落盘文件和可审计状态。没有完成或没有落盘 prediction 的 run 不补写指标。

## 执行节奏与加速调整

```text
Phase 0 code audit / bugfix 完成。
Phase 1 C9 attribution 完成。
Phase 2 adaptive failure autopsy 完成。
Phase 3 smoke 完成。
Phase 3 full AW111 B/C 完成。
Phase 3 no-SWA diagnostic 完成。
Phase 4/5 依赖 Phase 3 soft pass；AW111 与 no-SWA diagnostic 均 fail soft pass，因此 Phase 4/5 不启动。
Runtime efficiency audit 已新增；后续 full run 必须先过效率 gate。
```

用户指出进度过慢后，已调整为：

```text
GPU0: EnergyMatched AW111 full R2，已完成。
GPU1: Cluster3D AW111 full R2，已完成。
GPU2: EnergyMatched AW110 no-SWA diagnostic，已完成。
GPU3: Cluster3D AW110 no-SWA optional diagnostic，已完成。
GPU4/GPU5: 未启动额外 full run；计划约束下无可解释的新增网格。
```

慢的主要原因：

```text
1. Phase 2 true trace rerun 和 Phase 3 full run 都是 KITTI01 1101-frame rollout。
2. 每个 full run 是单 GPU、不可按 frame 拆分；launcher 只有 V47_END_FRAME，没有 start-frame 分片接口。
3. 第一批 Phase 3 full B/C 变成 missing_prediction/partial，不能写结果，只能重跑 R2。
4. 为避免伪造或引用 partial 指标，所有表只采用 DONE + prediction 落盘后的报告。
```

效率修正：

```text
新增 tools/v52_runtime_profile_report.py，用于汇总 timing_summary.json / wall_time_summary.json。
run_pipeline_abc_v2.py 新增 --empty_cuda_cache_each_chunk。
tools/run_attention_cue_experiment.sh 透传 EMPTY_CUDA_CACHE_EACH_CHUNK。
tools/run_v47_adaptive_ttt_writer_candidate.sh 后续新启动 run 默认 V47_EMPTY_CUDA_CACHE_EACH_CHUNK=0。

注意：
  这些改动只影响后续新启动的 run。
  GPU2/GPU3 no-SWA diagnostic 启动早于该开关改动，因此未被改命令或中断。
```

## Phase 0：代码修复与审计

修改：

```text
loger/pipeline/hybrid_memory_controller.py
  修复 past_plus_future_light_real 在 qq/qk/kk 的统一 weighted support。

loger/pipeline/ttt_write_controller.py
  tri replay 路径显式 ttt_two_replay_applied=False；
  新增 ttt_two_replay_debug_note，避免报告误读。

tools/v52_support_alias_unit_audit.py
tools/v52_phase0_debug_audit.py
tools/v52_code_packet_audit.py
  新增 v52 审计工具。
```

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  run_pipeline_abc_v2.py \
  loger/pipeline/hybrid_memory_controller.py \
  loger/pipeline/ttt_write_controller.py \
  tools/v47_adaptive_ttt_writer_report.py \
  tools/v52_support_alias_unit_audit.py \
  tools/v52_phase0_debug_audit.py \
  tools/v52_code_packet_audit.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v45_full_candidate.sh
bash -n tools/run_v46b_factorial_candidate.sh
bash -n tools/run_v47_adaptive_ttt_writer_candidate.sh

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v52_support_alias_unit_audit.py \
  --out-dir results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase0_code_audit

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v52_phase0_debug_audit.py \
  --out results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase0_code_audit/adaptive_writer_debug_field_audit.json

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v52_code_packet_audit.py \
  --out-dir results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase0_code_audit
```

关键结果：

```text
py_compile: pass
bash -n: pass
support alias pass: True
qq/qk/kk max diff vs manual weighted: 0.0 / 0.0 / 0.0
tri replay debug audit pass: True
code packet missing_in_repo=0, unpackaged=0
code packet sha256=7bba31dde4af448c982594143d2647c2669ea4cc8afe5461c67437030ab34070
```

证据：

```text
results/.../phase0_code_audit/bugfix_report.md
results/.../phase0_code_audit/support_alias_unit_audit_summary.{json,csv}
results/.../phase0_code_audit/adaptive_writer_debug_field_audit.json
results/.../phase0_code_audit/code_packet_completeness_audit.md
```

## Phase 1：C9 组件归因

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v52_c9_attribution_report.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v52_c9_attribution_report.py \
  --out-dir results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase1_c9_attribution
```

关键落盘结果：

```text
v46B F000 = 41.750205342038015
READ-only gain = 3.1567925378474513
TTT-only gain = 2.2311010578871375
SWA-only gain = 0.013164304661273718
READ+TTT gain = 5.081294500892781
READ+TTT incremental margin = 1.92450196304533
F111 gain = 5.099464891579238

Exact C9 F0 = 33.76294210291885
D7 C9-Clean = 35.500497135292775
D7 delta vs C9 = +1.7375550323739262

tri_replay contribution = 2.4469526757787676
tri_gamma_chunk_map contribution = 0.971025405816512
commit_ema contribution = 0.48838546389362136
```

证据：

```text
results/.../phase1_c9_attribution/c9_component_attribution_report.md
results/.../phase1_c9_attribution/positive_only_factorial_table.csv
results/.../phase1_c9_attribution/exact_c9_clean_rows.csv
results/.../phase1_c9_attribution/exact_c9_knockout_table.csv
results/.../phase1_c9_attribution/phase1_attribution_summary.json
```

## Phase 2：Adaptive failure autopsy

true trace rerun 命令：

```bash
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/rollouts \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_robust_split \
V11_PROJECTION_TRACE_DIR=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/rollouts/V52_TRACE_V50_SPLIT_RESXDG_AW111/v11_projection_trace \
V11_PROJECTION_GT_PATH=/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW111_FRAME_ADAPTIVE_TTT_SWA V52_TRACE_V50_SPLIT_RESXDG_AW111

V46B_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/rollouts \
V11_PROJECTION_TRACE_DIR=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit/rollouts/V52_TRACE_V46B_F111_FIXED/v11_projection_trace \
V11_PROJECTION_GT_PATH=/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt \
tools/run_v46b_factorial_candidate.sh 1 F111_ALL_THREE V52_TRACE_V46B_F111_FIXED
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/v52_adaptive_failure_audit.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v52_adaptive_failure_audit.py \
  --out-dir results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase2_adaptive_failure_audit
```

关键结果：

```text
C9_exact_teacher ATE = 33.762943
V46B_fixed_F111_teacher ATE = 36.650736
V50_split_resxdg_student ATE = 35.985305
V50 delta vs C9 = +2.22236289708
V50 delta vs soft 34.60 = +1.385305
V50 beats fixed F111 by 0.665431m in traced rerun.

fixed/update-conflict teacher probe TTT mean = 62.7037344983s/chunk
V50 student probe TTT mean = 5.78059238509s/chunk
```

证据：

```text
results/.../phase2_adaptive_failure_audit/adaptive_failure_autopsy.md
results/.../phase2_adaptive_failure_audit/run_overview.csv
results/.../phase2_adaptive_failure_audit/phase2_adaptive_failure_summary.json
results/.../phase2_adaptive_failure_audit/role_mass_timeline.csv
results/.../phase2_adaptive_failure_audit/post_zp_delta_ratio_by_chunk.csv
```

## Phase 3：Adaptive TTT v2 实现

修改：

```text
loger/pipeline/ttt_write_controller.py
  新增 role modes:
    adaptive_writer_conflictlite_split
    adaptive_writer_energy_matched_split
    adaptive_writer_cluster3d_split
  新增 conflict_lite_selected_layers risk source。
  新增 energy-matched gamma/lambda。
  新增 cluster3d role masks。

run_pipeline_abc_v2.py
  新增 risk source / role mode choices。
```

验证命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  run_pipeline_abc_v2.py \
  loger/pipeline/ttt_write_controller.py \
  tools/v47_adaptive_ttt_writer_report.py \
  tools/v52_adaptive_failure_audit.py

bash -n tools/run_v47_adaptive_ttt_writer_candidate.sh
bash -n tools/run_attention_cue_experiment.sh
```

## Phase 3 smoke

命令：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_smoke/rollouts \
V47_END_FRAME=96 V47_TTT_RISK_SOURCE=conflict_lite_selected_layers \
V47_TTT_ROLE_MODE=adaptive_writer_conflictlite_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW111_FRAME_ADAPTIVE_TTT_SWA V52_SMOKE_A1_CONFLICTLITE_AW111_96F

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_smoke/rollouts \
V47_END_FRAME=96 V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_energy_matched_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW111_FRAME_ADAPTIVE_TTT_SWA V52_SMOKE_B1_ENERGYMATCH_AW111_96F

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_smoke/rollouts \
V47_END_FRAME=96 V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_cluster3d_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW111_FRAME_ADAPTIVE_TTT_SWA V52_SMOKE_C1_CLUSTER3D_AW111_96F

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
  --rollout-root results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_smoke/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_smoke/report_R1
```

结果：

| Run | ATE | frames | chunk sec mean | probe TTT sec mean | audit |
|---|---:|---:|---:|---:|---|
| `V52_SMOKE_A1_CONFLICTLITE_AW111_96F` | `1.315722` | `96` | `45.389658` | `14.290215` | `True/True` |
| `V52_SMOKE_B1_ENERGYMATCH_AW111_96F` | `1.349152` | `96` | `40.185260` | `5.951500` | `True/True` |
| `V52_SMOKE_C1_CLUSTER3D_AW111_96F` | `1.366865` | `96` | `35.026832` | `5.108346` | `True/True` |

判断：

```text
ConflictLite probe TTT mean = 14.290215s/chunk，超过计划 8s/chunk 分流阈值。
因此 A 降级为 diagnostic，不作为 full 主线。
B/C 进入 full。
```

## Phase 3 full AW111

第一批 B/C full run 出现 missing_prediction/partial，不能作为结果。随后启动 R2：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/rollouts \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_energy_matched_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 0 AW111_FRAME_ADAPTIVE_TTT_SWA V52_FULL_B1_ENERGYMATCH_AW111_R2

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/rollouts \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_cluster3d_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 1 AW111_FRAME_ADAPTIVE_TTT_SWA V52_FULL_C1_CLUSTER3D_AW111_R2
```

R2 完成状态：

```text
V52_FULL_B1_ENERGYMATCH_AW111_R2 DONE 2026-06-09 03:17:38
V52_FULL_C1_CLUSTER3D_AW111_R2 DONE 2026-06-09 03:17:04
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
  --rollout-root results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/report_R1
```

结果：

| Run | Row | ATE | delta vs C9 | delta vs fixed TTT | chunk sec mean | probe TTT sec mean | audit |
|---|---|---:|---:|---:|---:|---:|---|
| `V52_FULL_B1_ENERGYMATCH_AW111_R2` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `35.967687` | `+2.204745` | `-0.683053` | `38.215938` | `4.980390` | `True/True` |
| `V52_FULL_C1_CLUSTER3D_AW111_R2` | `AW111_FRAME_ADAPTIVE_TTT_SWA` | `36.427703` | `+2.664761` | `-0.223037` | `37.039348` | `5.029483` | `True/True` |

Gate：

```text
C9 reference = 33.76294210291885
soft pass threshold = 34.60
best AW111 = 35.967687
best delta vs C9 = +2.204745
best delta vs soft = +1.367687
soft pass = False
```

## Phase 3 no-SWA diagnostic

用户要求加速后，no-SWA diagnostic 并行启动并已完成：

```bash
V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/rollouts \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_energy_matched_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 2 AW110_FRAME_ADAPTIVE_TTT V52_FULL_B1_ENERGYMATCH_AW110_NOSWA_R1

V47_RESULT_ROOT=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
V47_ROLLOUT_BASE=results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/rollouts \
V47_TTT_RISK_SOURCE=ttt_residual_x_dg \
V47_TTT_ROLE_MODE=adaptive_writer_cluster3d_split \
tools/run_v47_adaptive_ttt_writer_candidate.sh 3 AW110_FRAME_ADAPTIVE_TTT V52_FULL_C1_CLUSTER3D_AW110_NOSWA_R1
```

完成状态：

```text
V52_FULL_B1_ENERGYMATCH_AW110_NOSWA_R1 DONE 2026-06-09 03:41:44, wall=1570s
V52_FULL_C1_CLUSTER3D_AW110_NOSWA_R1 DONE 2026-06-09 03:42:07, wall=1594s
```

报告命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v47_adaptive_ttt_writer_report.py \
  --result-root results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry \
  --rollout-root results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/report_R2_noswa_done
```

结果：

| Run | Row | ATE | delta vs C9 | delta vs soft 34.60 | delta vs fixed TTT | chunk mean | TTT mean | audit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `V52_FULL_B1_ENERGYMATCH_AW110_NOSWA_R1` | `AW110_FRAME_ADAPTIVE_TTT` | `35.974144` | `+2.211202` | `+1.374144` | `-0.694766` | `39.476774` | `5.132336` | `True/True` |
| `V52_FULL_C1_CLUSTER3D_AW110_NOSWA_R1` | `AW110_FRAME_ADAPTIVE_TTT` | `36.452058` | `+2.689115` | `+1.852058` | `-0.216853` | `40.192329` | `5.518366` | `True/True` |

判断：

```text
no-SWA 结果与 AW111 SWA 结果几乎同级：
  B1 no-SWA 35.974144 vs B1 AW111 35.967687
  C1 no-SWA 36.452058 vs C1 AW111 36.427703

因此 v52 Phase 3 fail soft pass 不是 SWA 引起的。
EnergyMatched/Cluster3D adaptive split 比 fixed TTT 有改善，但没有接近 C9。
```

## Runtime efficiency audit

用户指出 full rollout wall time 不可接受后，新增 runtime profile 工具并对当前 full rollout 目录做轻量汇总：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  run_pipeline_abc_v2.py \
  tools/v52_runtime_profile_report.py

bash -n tools/run_attention_cue_experiment.sh
bash -n tools/run_v47_adaptive_ttt_writer_candidate.sh

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  tools/v52_runtime_profile_report.py \
  results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase3_adaptive_ttt_v2_full/rollouts \
  --out-dir results/kitti01_hmc_v2/acl2_v52_c9clean_adaptive_ttt_semantic_geometry/runtime_efficiency_audit \
  --max-wall-seconds 1800
```

当前汇总：

| Run | status | wall min | chunk mean | pass1 mean | StageB mean | pass2 mean | TTT mean | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `V52_FULL_B1_ENERGYMATCH_AW111_R2` | done | `25.35` | `38.22` | `11.50` | `8.58` | `10.51` | `4.98` | pass under 30min |
| `V52_FULL_C1_CLUSTER3D_AW111_R2` | done | `24.52` | `37.04` | `10.85` | `8.59` | `10.04` | `5.03` | pass under 30min |
| `V52_FULL_B1_ENERGYMATCH_AW110_NOSWA_R1` | done | `26.17` | `39.48` | `11.96` | `9.01` | `10.45` | `5.13` | pass under 30min |
| `V52_FULL_C1_CLUSTER3D_AW110_NOSWA_R1` | done | `26.57` | `40.19` | `11.59` | `9.30` | `10.89` | `5.52` | pass under 30min |

解释：

```text
1. 本地 DONE R2/no-SWA wall_time_summary 显示 24.5-26.6min，不是 61-62min。
2. 但 chunk mean 37-40s 仍然太重，full run 不能继续作为默认筛选。
3. 主要开销不是单独 TTT write，而是 pass1 probe + StageB + pass2 control + TTT write 累加。
4. 后续计划已改为 runtime gate：projected full wall >30min 或 chunk mean >30s 时，不允许进入 full，除非是唯一必要复现。
```

证据：

```text
results/.../runtime_efficiency_audit/v52_runtime_profile_summary.{json,csv,md}
```

## Phase 4/5 Gate

```text
Phase 4 semantic minimal retest 只在 Phase 3 adaptive baseline <=34.60 后启动。
Phase 5 cross sequence sanity 也只在 adaptive baseline 达到 soft pass 后启动。

当前 best Phase 3 = 35.967687 > 34.60。
因此 Phase 4/5 不启动。
```
