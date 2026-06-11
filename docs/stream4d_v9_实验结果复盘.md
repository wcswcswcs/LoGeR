# Stream4D v9 unified eval and method 实验结果复盘

日期：2026-06-09（Asia/Singapore）  
执行日志：`docs/stream4d_v9_执行日志.md`  
计划文件：`docs/stream4d_v9_unified_eval_and_method_plan_for_codex.md`

本复盘只记录真实执行得到的数据。没有跑出的指标不补写；blocker 和修复尝试单独记录。

## 当前状态

```text
正确 Python 环境：/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
GPU: CUDA_VISIBLE_DEVICES=6,7
probe split: Stream3D/splits/scannet_v6_probe5.txt

Day0 unified eval 完成。
B1 controls 完成。
Phase4 O1/O2/O3 core/fringe probe5 完成。
Dynamic Replica v9 env check 完成，但 usable_scene_count=0。
S3/G1 parent support 未跑 AP：当前 G1 carrier 不是 ScanNet mesh pre_points support。

最强 method AP:
  O1/O2 core support = 0.3498933087840023 / 0.5268071075480882 / 0.8615948481633923

最强 AP50/AP25 仍是 B1:
  B1 = 0.32843947812986807 / 0.6292662056580957 / 0.8843628978668244

统一评估结论：
  B1/O1/O2/O3 在 own sparse support 上有正信号；
  B1 在同 S2 support 上只比 Stream3D 高 +0.001725 AP，但 AP50/AP25 明显高；
  B1/O1/O2/O3 一旦放到 S0/S1 support 上 AP 接近崩掉；
  因此不能 claim full ScanNet 或 cross-support unified victory。
```

## 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/tools/evaluate_cross_prepoints.py` | 新增 `--eval-policy`，manifest 增加 `eval_policy`、`prediction_config`、`pre_points_config` | v9 需要区分 own/cross/fixed-support 评估，避免把 diagnostic 当 method result |
| `Stream3D/tools/scan_reportable_configs.py` | 增加 eval policy 扫描和 `--require-eval-policy` | reportable artifact 缺评估策略时必须失败 |
| `Stream3D/tools/summarize_v9_unified_eval.py` | 新增 unified eval matrix 汇总，解析官方 evaluator AP，并计算 support/IoU diagnostic | 统一比较 S0/S1/S2/controls/Phase4，不手工拼表 |
| `Stream3D/tools/export_v9_b1_controls.py` | 新增 no-track、shuffle、random、area、maskcount controls | 检查 B1 的收益是否来自 D4RT ownership，而不是面积/数量/随机 support |
| `Stream3D/tools/split_core_fringe_prediction.py` | 新增 `--eval-policy` 写入 manifest | Phase4 首跑发现 O1/O2/O3 缺 eval policy；修复后 reportable scan pass |
| `Stream3D/scripts/reproduce_v9_day0.sh` | 新增 Day0 复现入口 | 复现 B1 gap matrix、controls、audit |
| `Stream3D/scripts/v9_day0_matrix_probe5.json` | 新增 Day0 matrix spec | 避免手工选择行 |
| `Stream3D/scripts/reproduce_v9_phase4.sh` | 新增 Phase4 复现入口，并在修复后传入 O1/O2/O3 eval policy | 复现 core/fringe 方法比较 |
| `Stream3D/scripts/v9_phase4_matrix_probe5.json` | 新增 Phase4 matrix spec | 统一汇总 O1/O2/O3 和对应 cross-support rows |
| `Stream3D/data/prediction/stream4d_v8_b1_surfacelet_singlemask_probe5_class_agnostic/config_manifest.json` | 补充 `eval_policy=own_recompute_paper_style` | 既有 v8 B1 进入 v9 reportable scan 时缺 metadata |
| `Stream3D/data/TMP/stream4d_v8_b1_surfacelet_singlemask_probe5/config_manifest.json` | 同上 | 保持 pred/TMP manifest 一致 |
| `docs/stream4d_v9_执行日志.md`、`docs/stream4d_v9_实验结果复盘.md` | 新增日志 | 满足执行记录、结果复盘和审计要求 |

## Lane 0 / 审计验证

```text
Day0 py_compile: pass
Day0 import smoke: v9 import smoke OK
Day0 unittest: Ran 13 tests in 0.121s, OK
Phase4 py_compile: pass
Phase4 bash -n: pass

Day0 reportable scan r2:
  num_configs=6
  num_configs_missing_manifest=0
  num_configs_missing_eval_policy=0
  num_oracle_configs=0
  num_reportable_method_configs=6
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0

Day0 metric integrity:
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  gt_files_read_by_rescore=False
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=6
  num_suspicious_configs=0

Phase4 reportable scan:
  num_configs=3
  num_configs_missing_manifest=0
  num_configs_missing_eval_policy=0
  num_oracle_configs=0
  num_reportable_method_configs=3
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0

Phase4 metric integrity:
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  gt_files_read_by_rescore=False
```

## v9 Day0 unified eval matrix

支持定义：

```text
S0 = Stream3D original ScanNet support
S1 = historical 32f support / stream4d_32f_self_probe5
S2 = B1 support / stream4d_v8_b1_surfacelet_singlemask_probe5
```

| Row | AP | AP50 | AP25 | pre ratio | pred union ratio | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `P0 Stream3D on S0` | `0.23572958766757215` | `0.41430630093420795` | `0.5377857284834029` | `0.8467441995295546` | `0.8467441995295546` | Stream3D own |
| `P0 Stream3D on S1` | `0.3992127932017927` | `0.5971712938711367` | `0.7425353588266108` | `0.04514451433782776` | `0.8467441995295546` | Stream3D on historical sparse support |
| `P0 Stream3D on S2` | `0.32671440506026966` | `0.4967776584317938` | `0.7266380236305048` | `0.03986074960713631` | `0.8467441995295546` | Stream3D on B1 support |
| `P2 B1 on S2` | `0.32843947812986807` | `0.6292662056580957` | `0.8843628978668244` | `0.03986074960713631` | `0.03986074960713631` | B1 own |
| `P2 B1 on S0` | `0.00063537285759508` | `0.004293854293854294` | `0.010767703576280312` | `0.8467441995295546` | `0.03986074960713631` | B1 masks evaluated on S0 support |
| `P2 B1 on S1` | `0.01683726779673` | `0.04753430011191438` | `0.16816170430299288` | `0.04514451433782776` | `0.03986074960713631` | B1 masks evaluated on S1 support |
| `P1 v6 compact on S1` | `0.28483247256897415` | `0.5039622641509434` | `0.6719147248897401` | `0.04514451433782776` | `0.0415780150820649` | historical compact baseline on S1 |

关键差分：

```text
B1 on S2 - Stream3D on S2:
  AP  +0.001725073069598404
  AP50 +0.13248854722630188
  AP25 +0.15772487423631965

B1 on S1 - v6 compact on S1:
  AP  -0.26799520477224414
  AP50 -0.45642796403902897
  AP25 -0.5037530205867473

B1 on S0 - Stream3D on S0:
  AP  -0.23509421480997708
  AP50 -0.41001244664035363
  AP25 -0.5270180249071226
```

判断：

```text
B1 own support AP50/AP25 有明显优势；
但 AP 与 Stream3D-on-S2 几乎持平，只高 0.0017。
B1 on S1 远低于 v6 compact on S1，不满足 v9 Phase1 的 same-support 成立条件。
B1 on S0 几乎为零，说明 B1 mask/support 强绑定，不能跨 support claim。
```

## Phase2 B1 controls

| Control | AP | AP50 | AP25 | pre ratio | conflict | B1 - control AP/AP50/AP25 |
|---|---:|---:|---:|---:|---:|---|
| `C0 no-track` | `0.146705342203801` | `0.30550667509417073` | `0.5368412416412567` | `0.042371332157667585` | `0.6856756530048784` | `+0.18173413592606708 / +0.3237595305639249 / +0.34752165622556774` |
| `C1 shuffle` | `0.16674974377662624` | `0.33897812717969683` | `0.5883478771535553` | `0.04221031493352989` | `0.6809801381489401` | `+0.16168973435324183 / +0.2902880784783988 / +0.29601502071326913` |
| `C2 random_same_count_s0` | `0.2273565066334571` | `0.4015896372151457` | `0.6553753622298587` | `0.029335128915637516` | `0.2177296214091228` | `+0.10108297149641096 / +0.22767656844294998 / +0.22898753563696572` |
| `C3 area_same_count` | `0.14450252518514864` | `0.30533872554758557` | `0.5366137826394362` | `0.04235368763870692` | `0.6887702267956342` | `+0.18393695294471943 / +0.3239274801105101 / +0.3477491152273883` |
| `C4 maskcount_same_count` | `0.15665269373330934` | `0.3134522651691879` | `0.5526626668422495` | `0.04190280349975758` | `0.6850507489338424` | `+0.17178678439655873 / +0.31581394048890776 / +0.3317002310245749` |

对应 Stream3D-on-control-support：

| Support | Stream3D AP | Stream3D AP50 | Stream3D AP25 |
|---|---:|---:|---:|
| no-track support | `0.3727224607860679` | `0.5111111111111111` | `0.7271604938271605` |
| shuffle support | `0.38356261022927685` | `0.5257142857142857` | `0.7479365079365079` |
| random support | `0.39904420549581837` | `0.5913978494623656` | `0.7497556207233627` |
| area support | `0.3727224607860679` | `0.5111111111111111` | `0.7271604938271605` |
| maskcount support | `0.37323232323232325` | `0.5003246753246753` | `0.7207792207792207` |

判断：

```text
B1 相比 no-track/shuffle/area/maskcount/random controls 均有明显 own-support 提升。
尤其 shuffle control 差分为 +0.1617 AP / +0.2903 AP50 / +0.2960 AP25。
这支持 “D4RT ownership/track signal 有贡献”，而不是简单随机或面积选择。

但所有 controls on S0 与 B1 on S0 一样接近零；
且 Stream3D on control supports 仍普遍高于 control own methods。
所以 controls 证明了 B1 内部 ownership 优于 naive controls，
但没有解决跨 support 泛化问题。
```

## Phase4 core/fringe methods

本轮 Phase4 是保守 mesh-support postprocess，不是完整 surfel posterior。它从 B1 prediction/support 出发，清理或重分配 conflict points。

| Method | AP | AP50 | AP25 | pre ratio | conflict | same-support gap vs Stream3D | vs B1 |
|---|---:|---:|---:|---:|---:|---|---|
| `O0 B1 full-mask own` | `0.32843947812986807` | `0.6292662056580957` | `0.8843628978668244` | `0.03986074960713631` | `0.08430650606572185` | `+0.001725073069598404 / +0.13248854722630188 / +0.15772487423631965` | baseline |
| `O1 core-only own` | `0.3498933087840023` | `0.5268071075480882` | `0.8615948481633923` | `0.03663422658631823` | `0.0` | `+0.017863051982578426 / +0.04625790846342004 / +0.1788145277972596` | `+0.021453830654134254 / -0.10245909811000742 / -0.02276804970343216` |
| `O2 core+radius fringe own` | `0.3498933087840023` | `0.5268071075480882` | `0.8615948481633923` | `0.03663422658631823` | `0.0` | `+0.017863051982578426 / +0.04625790846342004 / +0.1788145277972596` | `+0.021453830654134254 / -0.10245909811000742 / -0.02276804970343216` |
| `O3 WTA-negative fringe own` | `0.34052646499896333` | `0.5804108107848305` | `0.876844735148573` | `0.039852776941988244` | `0.0` | `+0.013812059938693666 / +0.0836331523530367 / +0.1502067115180682` | `+0.012086986869095262 / -0.04885539487326518 / -0.007518162718251453` |

Phase4 support transfer：

| Row | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `P0 Stream3D on O1` | `0.3320302568014239` | `0.4805491990846682` | `0.6827803203661327` |
| `O1 on S0` | `0.00028603084158639714` | `0.0011497761497761497` | `0.009730750333444716` |
| `O1 on S1` | `0.013430692074848725` | `0.04253098125317383` | `0.13055920817487343` |
| `P0 Stream3D on O2` | `0.3320302568014239` | `0.4805491990846682` | `0.6827803203661327` |
| `O2 on S0` | `0.00028603084158639714` | `0.0011497761497761497` | `0.009730750333444716` |
| `O2 on S1` | `0.013430692074848725` | `0.04253098125317383` | `0.13055920817487343` |
| `P0 Stream3D on O3` | `0.32671440506026966` | `0.4967776584317938` | `0.7266380236305048` |
| `O3 on S0` | `0.00028603084158639714` | `0.0011497761497761497` | `0.010767703576280312` |
| `O3 on S1` | `0.014575660363574435` | `0.04193785951547378` | `0.15909158503678383` |

Phase4 export diagnostic：

```text
O1 mean instances after = 15.6
O1 mean output union = 7851.2
O1 mean output support conflict ratio = 0.0
O1 mean core ratio = 0.7359904700557497

O2 mean instances after = 15.6
O2 mean output union = 7851.2
O2 mean output support conflict ratio = 0.0
O2 radius growth added points = 0.0

O3 mean instances after = 16.2
O3 mean output union = 8511.8
O3 mean output support conflict ratio = 0.0
O3 mean core ratio = 0.8666684364728209
```

判断：

```text
O1/O2 把 conflict 从 B1 的 0.0843 降到 0，并把 AP 从 0.3284 提到 0.3499。
但 O1/O2 的 AP50/AP25 低于 B1，说明 core-only 清理提高排序/严格 AP 的同时丢了中低 IoU recall。
O2 与 O1 完全一致，且 radius growth added points = 0；这个 fringe 设定没有实际扩张 support。
O3 在 AP/AP50/AP25 上介于 B1 和 O1/O2 之间，AP50 比 O1/O2 好但仍低于 B1。
所有 O-method on S0/S1 仍接近零，跨 support 问题没有解决。
```

## G1 / Lane2 证据边界

v9 复用并核验 v8 连续帧 G1 / Lane2 输出，不把它们改写成 v9 新 AP。

G1 correspondence probe5：

```text
num_ok_windows=5
uv_in01_rate_mean=0.9858451843261719
track_length_visible_mean_mean=13.284716796875
self_uv_error_p90_mean=1.57081866979599
cycle_uv_error_p90_mean=3.2913891315460204
surfel_coverage_2d_per_frame_mean_mean=0.13198394775390626
```

G1 Sim3 geometry：

```text
sim3_anchor_count_mean=431.2
sim3_residual_median_mean=0.46820781478265117
sim3_residual_p90_mean=0.8595804531797114
sim3_residual_median_max=0.6804238543343015
sim3_residual_p90_max=1.1890475588672236
```

Lane2 mask measurement coverage：

```text
diagnostic_only=True
uses_gt=False
num_ok_windows=5
num_frames_mean=16.0
num_mask_frames_available_mean=2.0
num_mask_frames_missing_mean=14.0
carrier_assignment_rate_all_frames_mean=0.12105091419117497
carrier_assignment_rate_available_mask_frames_mean=0.9806140838500657
surfel_positive_observation_rate_mean=0.91259765625
mean_positive_observations_per_surfel_mean=1.72508544921875
```

判断：

```text
D4RT 连续帧 correspondence 是可用的；
但 Sim3 residual 仍高，不能 claim D4RT metric geometry 替代 ScanNet mesh。
mask observation 只有 2/16 帧，不能假装有 dense semantic measurement。
```

## Dynamic Replica

```text
data_root_exists=True
split_dir_exists=True
annotation_exists=True
all_required_camera_fields_present=True
usable_scene_count=0
can_report_official_instance_tracking=False
can_report_d4rt_trajectory_metrics=False
can_report_only_qualitative_consistency=False
```

场景样本均为：

```text
images=300
depths=0
trajectories=300
semantic GT=False
instance GT=False
object IDs=False
```

判断：

```text
本地 Dynamic Replica valid split 有 image/trajectory metadata，
但没有 depth 对齐和 instance/object ID GT。
因此不能报告 IDF1、MOTA、4D IoU、official instance tracking 或 D4RT trajectory metrics。
```

## Blocker 和修复结果

### B1：Day0 B1 manifest 缺 eval_policy

```text
问题：
  v8 B1 既有 pred/TMP manifest 没有 eval_policy。
  v9 reportable scan 加 --require-eval-policy 后不应放过。

修复：
  给 B1 pred/TMP manifest 补充 eval_policy=own_recompute_paper_style。

结果：
  reportable_config_scan_controls_probe5_r2:
    num_configs_missing_eval_policy=0
    num_configs_missing_manifest=0
    num_suspicious_configs=0
```

### B2：Phase4 O1/O2/O3 manifest 缺 eval_policy

```text
问题：
  Phase4 第一次执行完成评估和 matrix，但最后 reportable scan 退出码 4。
  scan summary:
    num_configs=3
    num_configs_missing_eval_policy=3

修复：
  tools/split_core_fringe_prediction.py 新增 --eval-policy。
  scripts/reproduce_v9_phase4.sh 对 O1/O2/O3 export 显式传入 eval policy。
  重跑 Phase4。

结果：
  reportable_config_scan_phase4_probe5:
    num_configs_missing_eval_policy=0
    num_suspicious_configs=0
  metric_integrity_phase4_probe5:
    phase0_pass=True
```

### B3：S3 / G1 parent support 不能直接评估

```text
问题：
  计划里的 S3 = B1 parent/G1 support 需要和 ScanNet evaluator 的 mesh pre_points 对齐。
  当前 G1 carrier artifact 是 D4RT query/surfel NPZ，不是 data/TMP/<config>/<scene>_pre_points.npy。

实际 G1 carrier keys:
  carrier_id
  confidence_prob
  src_frame
  src_frame_global
  src_mask_id
  src_uv
  src_xy
  uv_pred
  valid
  visibility_prob
  xyz_ref

处理：
  本轮不伪造 P0 on S3 / B1 on S3 / O-method on S3 AP。
  记录 blocker。

后续修复方向：
  新增 surfel-to-mesh support materializer。
  必须审计 Sim3/nearest-neighbor 投影误差和 support policy 后，才能跑 S3 AP。
```

## 分析

```text
1. B1 的 v8 own-support 成绩不是完全由 support selection 解释。
   在同 S2 support 上，B1 AP 只比 Stream3D 高 0.0017，
   但 AP50/AP25 分别高 0.1325 和 0.1577。
   说明 B1 的 mask ownership 在中低 IoU 阈值上有真实收益。

2. 但 B1 不具备跨 support 稳定性。
   B1 on S0 AP=0.000635，B1 on S1 AP=0.016837。
   这不是轻微退化，而是几乎失效。
   因此 v9 不能把 B1 表述为统一 support 下优于 Stream3D/v6 的方法。

3. Controls 支持 D4RT ownership 的贡献。
   shuffle/no-track/area/maskcount controls 的 conflict 都约 0.68，
   B1 conflict 只有 0.0843。
   B1 相比 shuffle control 高 +0.1617 AP / +0.2903 AP50 / +0.2960 AP25。

4. Phase4 core cleaning 提高 AP 但牺牲 AP50/AP25。
   O1/O2 AP=0.3499，是本轮最高 AP；
   但 AP50=0.5268、AP25=0.8616，低于 B1 的 0.6293/0.8844。
   core-only 让 support 更干净，但也丢掉了部分 recall。

5. O2 radius fringe 没有实际生效。
   O2 与 O1 完全同分，growth_added_points=0.0。
   当前 radius/growth setting 没有恢复 B1 被 core-only 丢掉的 AP50/AP25。

6. O3 WTA-negative 是折中，但不是最终解。
   O3 AP=0.3405，AP50=0.5804，AP25=0.8768；
   AP/AP50/AP25 都高于 Stream3D on O3 support，
   但 AP50/AP25 仍低于 B1，AP 低于 O1/O2。

7. D4RT correspondence 和 metric geometry 是两个不同问题。
   uv/cycle 指标很好，但 Sim3 median mean=0.468m。
   当前可以继续 image-space/surfel ownership 研究，
   但不能 claim metric mesh replacement。
```

## 结论

```text
success=True for v9 Day0 unified eval execution
success=True for B1 control experiments
success=True for Phase4 O1/O2/O3 probe5 execution
success=False for S3/G1 parent support AP
success=False for full ScanNet / Dynamic Replica official claim

可以报告：
  1. B1 在 own S2 support 上超过 controls，说明 D4RT ownership 有真实贡献。
  2. B1 在同 S2 support 上相对 Stream3D 只小幅提升 AP，但显著提升 AP50/AP25。
  3. O1/O2 core-only 是本轮最高 AP method result：AP=0.3498933087840023。
  4. 所有 v9 reportable artifacts uses_gt=False，reportable scan 和 metric integrity 通过。

不能报告：
  1. B1/O1/O2/O3 在 S0/S1 support 上优于 Stream3D/v6。
  2. full ScanNet result。
  3. Dynamic Replica official tracking 或 trajectory metric。
  4. D4RT metric geometry 替代 ScanNet RGB-D mesh。
  5. S3/G1 support AP。
```

## Insight

```text
这轮最重要的变化是把 “own-support 好看” 拆开了。

B1 的好处不是假的：
  controls 说明 D4RT ownership 明显降低 conflict，并带来大幅 AP/AP50/AP25 提升。

但 B1 的好处也不是完整方法胜利：
  它强绑定自己的 sparse support；
  换到 S0/S1 support 就几乎失效；
  同 S2 support 上 AP 只比 Stream3D 高一点，主要优势在 AP50/AP25。

Phase4 给出一个清晰方向：
  clean ownership 可以提升 AP；
  但 aggressive core-only 会损失 AP50/AP25 recall；
  下一步应做的不是盲目扩 support，而是设计可审计的 fringe recovery。

S3/G1 blocker 也很关键：
  现在的 G1 surfel carrier 不是 mesh evaluator support。
  如果直接写 S3 AP，就是把 D4RT query support 和 ScanNet mesh vertex support 混为一谈。
  正确下一步是先做 surfel-to-mesh materializer 和误差审计。
```

## 证据链索引

Code:

```text
Stream3D/tools/evaluate_cross_prepoints.py
Stream3D/tools/scan_reportable_configs.py
Stream3D/tools/summarize_v9_unified_eval.py
Stream3D/tools/export_v9_b1_controls.py
Stream3D/tools/split_core_fringe_prediction.py
Stream3D/scripts/reproduce_v9_day0.sh
Stream3D/scripts/v9_day0_matrix_probe5.json
Stream3D/scripts/reproduce_v9_phase4.sh
Stream3D/scripts/v9_phase4_matrix_probe5.json
```

Day0:

```text
Stream3D/outputs/audit/v9_day0/unified_eval_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_day0/reportable_config_scan_controls_probe5_r2.{json,csv,md}
Stream3D/outputs/audit/v9_day0/metric_integrity_controls_probe5.{json,md}
Stream3D/outputs/v9_b1_controls/stream4d_v9_b1_*_probe5_summary.json
Stream3D/data/evaluation/scannet/stream4d_v9_*_probe5_class_agnostic.txt
Stream3D/data/prediction/stream4d_v9_*_probe5_class_agnostic/*.npz
Stream3D/data/TMP/stream4d_v9_*_probe5/*_pre_points.npy
```

Phase4:

```text
Stream3D/outputs/audit/v9_phase4/phase4_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_phase4/reportable_config_scan_phase4_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_phase4/metric_integrity_phase4_probe5.{json,md}
Stream3D/outputs/v9_core_fringe/stream4d_v9_o1_b1_core_only_probe5_summary.json
Stream3D/outputs/v9_core_fringe/stream4d_v9_o2_b1_core_radius_fringe_probe5_summary.json
Stream3D/outputs/v9_core_fringe/stream4d_v9_o3_b1_wta_negative_fringe_probe5_summary.json
Stream3D/data/evaluation/scannet/stream4d_v9_o*_probe5_class_agnostic.txt
Stream3D/data/prediction/stream4d_v9_o*_probe5_class_agnostic/*.npz
Stream3D/data/TMP/stream4d_v9_o*_probe5/*_pre_points.npy
```

G1 / Lane2 / Dynamic Replica:

```text
Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger/summary.{json,csv,md}
Stream3D/outputs/audit/v8_g1_grid32m002_probe5_16f_stride1_loger_geometry.{json,csv,md}
Stream3D/outputs/audit/v8_mask_measurement_coverage_probe5_stride1_loger.{json,csv,md}
Stream3D/outputs/audit/v9_day0/dynamic_replica_env_v9.{json,md}
```

Logs:

```text
Stream3D/logs/stream4d_v9_py_compile.log
Stream3D/logs/stream4d_v9_import_smoke.log
Stream3D/logs/stream4d_v9_unittest.log
Stream3D/logs/stream4d_v9_unified_eval_matrix_probe5.log
Stream3D/logs/stream4d_v9_reportable_scan_controls_probe5.log
Stream3D/logs/stream4d_v9_reportable_scan_controls_probe5_r2.log
Stream3D/logs/stream4d_v9_metric_integrity_controls_probe5.log
Stream3D/logs/stream4d_v9_dynamic_replica_env.log
Stream3D/logs/stream4d_v9_phase4_py_compile.log
Stream3D/logs/stream4d_v9_phase4_matrix_probe5.log
Stream3D/logs/stream4d_v9_phase4_reportable_scan.log
Stream3D/logs/stream4d_v9_phase4_metric_integrity_probe5.log
Stream3D/logs/stream4d_v9_o1_core_only_export.log
Stream3D/logs/stream4d_v9_o2_core_radius_fringe_export.log
Stream3D/logs/stream4d_v9_o3_wta_negative_fringe_export.log
```

Environment:

```text
Stream3D/pip_freeze_v9_loger.txt
```

## 审计包

```text
latest packet:
  stream4d_v9_code_audit_packet_20260609_0603_probe5_unified.zip

latest sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_0603_probe5_unified.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

latest filelist:
  stream4d_v9_code_audit_packet_20260609_0603_probe5_unified_filelist.txt

latest zip test:
  No errors detected in compressed data of stream4d_v9_code_audit_packet_20260609_0603_probe5_unified.zip.

latest file count:
  649
```

## 追加复盘：cross-support top priority

用户追加要求：

```text
优先解决 cross-support。
```

追加执行结论：

```text
cross-support 目标仍未达成。
O6/O7/O8/O9 都没有让 reportable method 在 S0/S1 support 上接近 Stream3D/v6。
本轮没有编造成功；新增结果全部保留为真实失败或 diagnostic upper bound。
```

## 追加修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/tools/build_union_prepoints.py` | 新增 support union materializer | 构造 S4=union(S0,S2)，验证 union support 不能自动解决 cross-support |
| `Stream3D/tools/split_core_fringe_prediction.py` | 新增 `--growth-candidate-mode scene` | 允许 O4 从 B1 core 向 scene-wide mesh candidate 做半径 fringe 扩张 |
| `Stream3D/tools/diagnose_v9_d4rt_mask_propagation.py` | 新增 D4RT mask propagation coverage diagnostic | CropFormer higher-frequency blocked 后，按计划验证 propagation 是否能补 temporal measurement |
| `Stream3D/tools/export_v9_propagated_slot_field.py` | 新增 O5 propagated slot field exporter | 把 propagated mask slots materialize 成 AP method prototype，检验是否改善 cross-support |
| `Stream3D/tools/complete_prediction_to_support.py` | 新增 O6 support completion 工具 | 直接把 target support 点按最近 object core 分配给方法 mask，测试 point-level completion 是否能修复 S0/S1 |
| `Stream3D/tools/export_v8_surfel_object_field.py` | 新增 `--eval-policy` 写入 manifest | 新 v9 variants 必须带 eval policy，避免 reportable scan 缺字段 |
| `Stream3D/tools/export_v8_surfel_object_field.py` | 修复多窗口 `positive_mask_sample_rate` 聚合 | O9 多窗口 summary 原先把 rate 相加导致 >1，修复为 positive/valid count ratio |
| `Stream3D/tools/slotwise_candidate_select.py` | 新增 manifest、`--eval-policy`、`--diagnostic-only` | O8 需要把 `scannet` candidate upper bound 标为 diagnostic-only，防止误报方法结果 |
| `Stream3D/scripts/reproduce_v9_s4_phase4b.sh`、`v9_s4_phase4b_matrix_probe5.json` | 新增 S4/O4 复现脚本与矩阵 | 复现 union support 和 scene-fringe cross-support 结果 |
| `Stream3D/scripts/reproduce_v9_phase5_o5_cross_support.sh`、`v9_phase5_o5_matrix_probe5.json` | 新增 O5 cross-support 复现脚本与矩阵 | 复现 D4RT propagated slot field 结果 |
| `Stream3D/scripts/reproduce_v9_o6_support_completion_cross_support.sh`、`v9_o6_support_completion_matrix_probe5.json` | 新增 O6 support completion 复现脚本与矩阵 | 复现 point-level support completion 结果 |
| `Stream3D/scripts/reproduce_v9_o7_birth_recall_cross_support.sh`、`v9_o7_birth_recall_matrix_probe5.json` | 新增 O7 birth-recall 复现脚本与矩阵 | 复现降低 min_carriers 增加 object slots 的结果 |
| `Stream3D/scripts/reproduce_v9_o8_slot_candidate_cross_support.sh`、`v9_o8_slot_candidate_matrix_probe5.json` | 新增 O8 full-scene candidate diagnostic 复现脚本与矩阵 | 区分 full-scene candidate upper bound 与非 GT obs-bank candidate 失败 |
| `Stream3D/scripts/reproduce_v9_o9_multwindow_cross_support.sh`、`v9_o9_multwindow_matrix_probe5.json` | 新增 O9 multi-window D4RT 复现脚本与矩阵 | 验证增加连续 D4RT windows/mask frames 是否改善 support coverage 与 cross-support |
| `docs/stream4d_v9_执行日志.md`、`docs/stream4d_v9_实验结果复盘.md` | 追加 cross-support 日志和复盘 | 满足用户要求，记录真实命令、blocker、修复尝试和失败结论 |

## Cross-support 追加结果总表

所有数值是官方 evaluator AP/AP50/AP25 的原始 0-1 小数。

### S4/O4：union support 与 scene fringe

| Row | AP | AP50 | AP25 | pre ratio | conflict | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `P0 Stream3D on S4` | `0.23504494357207534` | `0.41430630093420795` | `0.5377857284834029` | `0.8476265009146591` | `0.2213` | S4 基本等于 S0，未给新方法优势 |
| `B1 on S4` | `0.00063537285759508` | `0.004293854293854294` | `0.010767703576280312` | `0.8476265009146591` | `0.0843` | 与 B1 on S0 一样崩 |
| `O4 scene fringe r0.02 own` | `0.27332782077312223` | `0.45816899831266267` | `0.7948049043158266` | `0.049321168880599946` | `0.04267301667886557` | own 有信号但低于 B1 AP50；S0/S1 仍失败 |
| `O4 r0.02 on S0` | `0.0015343295361216508` | `0.004767730775795292` | `0.01636819036404976` | `0.8467441995295546` | `0.0427` | 未解决 S0 |
| `O4 r0.02 on S1` | `0.016861638556381193` | `0.04599825458290972` | `0.17003319795984578` | `0.04514451433782776` | `0.0427` | 未超过 v6 compact |
| `O4 scene fringe r0.05 own` | `0.1688754827908881` | `0.3331513158372127` | `0.6930771801917053` | `0.0744724956764278` | `0.22938707041611142` | 扩张过多导致 conflict/AP 崩 |

S4 support summary：

```text
mean_union_points=186354.8
mean_scannet_points=186164.8
mean_B1_points=8513.2
mean_union_over_max_input=1.0010538535786635
```

判断：

```text
S4 几乎就是 S0 加少量 B1 点，不能解决方法 mask 只覆盖 tiny observed subset 的问题。
O4 r0.02 的 own AP25 高，但 method-on-S0/S1 仍接近 B1，cross-support 未解决。
```

### Phase5 mask frequency / propagation

CropFormer blocker：

```text
python third_party/Cropformer.py --help:
  ModuleNotFoundError: No module named 'mask2former'

PYTHONPATH=third_party/detectron2/projects/CropFormer python third_party/Cropformer.py --help:
  ImportError: cannot import name '_C' from 'detectron2'
```

D4RT mask propagation diagnostic：

```text
diagnostic_only=True
uses_gt=False
is_method_result=False
num_ok_scenes=5
num_frames_mean=16
num_positive_source_carriers_mean=2008.0
positive_source_carrier_rate_mean=0.12255859375
num_positive_source_slots_mean=41.8
frames_with_propagated_measurement_mean=16
propagated_frame_rate_mean=1.0
propagated_positive_carriers_per_frame_mean_mean=1671.675
positive_carrier_observations_mean_mean=13.30330313048482
collision_point_rate_mean_mean=0.000362135764865486
conflicting_pixel_rate_mean_mean=0.00018127008076854632
unique_propagated_slots_per_frame_mean_mean=35.1125
```

O5 propagated slot AP：

| Row | AP | AP50 | AP25 | pre ratio | conflict |
|---|---:|---:|---:|---:|---:|
| `O5 own` | `0.04758295716672683` | `0.2181824246604962` | `0.5138809947918372` | `0.022943926035371806` | `0.30732105087179745` |
| `P0 Stream3D on O5` | `0.3264414983164983` | `0.45880681818181823` | `0.7140151515151515` | `0.022943926035371806` | `0.2213` |
| `O5 on S0` | `0.0` | `0.0` | `0.0006439678787592312` | `0.8467441995295546` | `0.3073` |
| `O5 on S1` | `0.00015237561073077048` | `0.0010845187639320693` | `0.0165409758608464` | `0.04514451433782776` | `0.3073` |
| `O5 on S4` | `0.0` | `0.0` | `0.0006439678787592312` | `0.8476265009146591` | `0.3073` |

判断：

```text
Propagation 能补 temporal measurement coverage，但 O5 object partition/overlap 质量差。
O5 没有修复 cross-support，甚至 own AP 明显低于 B1/O1。
```

### O6：support completion

| Row | AP | AP50 | AP25 | pre ratio | conflict | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `O6 r0.02 own` | `0.256582` | `0.450681` | `0.818187` | `0.050152` | `0.0` | own 尚可，但低于 B1/O1 |
| `O6 r0.02 on S0` | `0.001534` | `0.004768` | `0.016368` | `0.846744` | `0.0` | S0 仍崩 |
| `O6 r0.05 own` | `0.186596` | `0.393051` | `0.709579` | `0.072518` | `0.0` | support 更大但 AP 更低 |
| `O6 r0.05 on S0` | `0.003225` | `0.008379` | `0.023870` | `0.846744` | `0.0` | S0 仍崩 |
| `O6 r0.10 own` | `0.143044` | `0.320326` | `0.640611` | `0.091365` | `0.0` | 半径继续扩大损伤 AP |
| `O6 r0.10 on S0` | `0.004289` | `0.010638` | `0.026037` | `0.846744` | `0.0` | S0 仍崩 |
| `O6 all own` | `0.006259` | `0.020778` | `0.075250` | `0.847512` | `0.0` | 覆盖全 support 但 object 粗糙，失败 |
| `O6 all on S0` | `0.006259` | `0.020778` | `0.075250` | `0.846744` | `0.0` | 完整 S0 support 仍远低于 Stream3D |

判断：

```text
最近 core 的 point-level completion 不能替代 object boundary/candidate inference。
即使 all 覆盖 S0，只有约 14.4 个 object 被铺满全场，AP 仍很低。
```

### O7：object birth recall

| Row | AP | AP50 | AP25 | pre ratio | #pred | conflict |
|---|---:|---:|---:|---:|---:|---:|
| `O7 min8 own` | `0.315568` | `0.613131` | `0.861687` | `0.040310` | `18.40` | `0.087819` |
| `O7 min8 on S0` | `0.000635` | `0.004294` | `0.010768` | `0.846744` | `18.40` | `0.087819` |
| `O7 min4 own` | `0.299951` | `0.572588` | `0.840145` | `0.040609` | `20.20` | `0.090412` |
| `O7 min4 on S0` | `0.000635` | `0.004294` | `0.010768` | `0.846744` | `20.20` | `0.090412` |
| `O7 min2 own` | `0.299951` | `0.572588` | `0.840145` | `0.040609` | `20.20` | `0.090412` |
| `O7 min2 on S0` | `0.000635` | `0.004294` | `0.010768` | `0.846744` | `20.20` | `0.090412` |

判断：

```text
降低 min_carriers 只增加少量 slots，GT crop 仍是 8.20/40.60。
单个 16f clip 的 temporal/scene coverage 是根本限制，birth threshold 不是主要瓶颈。
```

### O8：full-scene candidate diagnostic

| Row | AP | AP50 | AP25 | pre ratio | #pred | conflict | method/diagnostic |
|---|---:|---:|---:|---:|---:|---:|---|
| `O8 scannet slot upper own S0` | `0.270727` | `0.371378` | `0.447508` | `0.232794` | `11.20` | `0.2067` | diagnostic-only |
| `O8 scannet slot upper on S1` | `0.139846` | `0.178689` | `0.286773` | `0.045145` | `11.20` | `0.2067` | diagnostic-only |
| `O8 obs-bank slot-only own` | `0.000743` | `0.003383` | `0.032805` | `0.552481` | `16.40` | `0.109886` | method result |
| `O8 obs-bank slot-only on S0` | `0.000337` | `0.001619` | `0.012895` | `0.846744` | `16.40` | `0.109886` | method result |
| `O8 obs-bank slot+top80 own` | `0.000975` | `0.004807` | `0.039085` | `0.552481` | `96.40` | `0.704240` | method result |
| `O8 obs-bank slot+top80 on S0` | `0.000368` | `0.001806` | `0.018056` | `0.846744` | `96.40` | `0.704240` | method result |

O8 audit：

```text
reportable scan:
  num_configs=3
  num_reportable_method_configs=2
  num_diagnostic_only_configs=1
  num_uses_gt_and_method_result=0
  num_configs_missing_eval_policy=0

metric integrity:
  phase0_pass=True
  num_reportable_method_configs=2
```

判断：

```text
使用 Stream3D/scannet full-scene candidates 的 diagnostic upper 能明显提高 S0 AP，
说明 D4RT/B1 slot 与高质量 full-scene candidates 可以对齐。
但这不能作为方法结果，因为 source candidate 就是 Stream3D baseline。
非 GT obs-bank candidates 即使覆盖 55% pre_points，AP 仍接近 0，说明 raw mask bank 质量/去重/排序远不够。
```

### O9：multi-window D4RT

G1 multi-window diagnostic：

```text
run=stream4d_v9_g1_grid16m002_probe5_96f_stride20_loger
num_windows=25
num_ok_windows=25
grid_size=16
window_size=16
window_stride=20
max_frames=96
num_source_queries_mean=4096.0
uv_in01_rate_mean=0.9480364990234375
self_uv_error_p90_mean=1.6126432371139527
cycle_uv_error_p90_mean=3.6816897602081298
track_length_visible_mean_mean=12.14859375
surfel_coverage_2d_per_frame_mean_mean=0.04088371276855469
```

O9 AP：

| Row | AP | AP50 | AP25 | pre ratio | GT crop/full | #pred | conflict |
|---|---:|---:|---:|---:|---|---:|---:|
| `O9 own` | `0.040167` | `0.124304` | `0.294343` | `0.082626` | `12.60/40.60` | `63.40` | `0.684667` |
| `P0 Stream3D on O9` | `0.349368` | `0.543819` | `0.646154` | `0.082626` | `12.60/40.60` | `128.20` | `0.2213` |
| `O9 on S0` | `0.000137` | `0.000900` | `0.005557` | `0.846744` | `40.60/40.60` | `63.40` | `0.684667` |
| `O9 on S1` | `0.014236` | `0.042087` | `0.080372` | `0.045145` | `19.60/40.60` | `63.40` | `0.684667` |

O9 object summary after rate fix：

```text
num_available_mask_frames_mean=10.0
num_exported_objects_mean=63.4
num_exported_points_mean=17448.4
positive_mask_sample_rate_mean=0.987537692406441
export_conflict_rate_mean=0.6846670938545043
```

判断：

```text
多窗口确实提升 coverage：
  B1 GT crop 8.20/40.60 -> O9 12.60/40.60
  B1 pre ratio 0.03986 -> O9 0.08263
  B1 #pred 16.4 -> O9 63.4

但 object competition/merge 没跟上：
  conflict 0.0843 -> 0.6847
  own AP/AP50/AP25 大幅降为 0.0402/0.1243/0.2943
  Stream3D on O9 support = 0.3494/0.5438/0.6462，远高于 O9

所以“更多 windows/mask frames”方向是必要但不充分；
必须加入跨窗口 duplicate suppression、cannot-link/merge、slot state，而不能简单把每个窗口的 mask-owned surfacelet 全部导出。
```

## Cross-support blocker 修复尝试汇总

```text
B6 S4 union support:
  修复尝试：构造 S4=union(S0,S2)。
  结果：S4 几乎等于 S0；B1/O1/O3/O4 on S4 仍接近 S0 失败。

B7 scene-wide fringe:
  修复尝试：O4 r0.02/r0.05 从 B1 core 向 scene candidate 扩张。
  结果：r0.02 own 可用但 S0/S1 未修复；r0.05 conflict 升高，AP 降低。

B8 higher-frequency masks:
  修复尝试：运行本地 CropFormer。
  结果：mask2former/detectron2 _C 缺失，不能可靠生成新增 masks。

B9 D4RT propagation:
  修复尝试：传播已有 mask measurement 到 16f 所有帧，再导出 O5。
  结果：coverage 变好，但 O5 AP 低，S0/S1 仍崩。

B10 support completion:
  修复尝试：O6 最近 core 分配 target support points。
  结果：small radius 不够覆盖，all 覆盖 S0 但边界/object 数质量极差。

B11 object birth recall:
  修复尝试：O7 降低 min_carriers 到 8/4/2。
  结果：slots 从 16.4 增到 20.2，GT crop 不变，S0/S1 不变。

B12 full-scene candidate diagnostic:
  修复尝试：O8 用 B1 slots 选择 scannet full-scene candidates 和非 GT obs-bank candidates。
  结果：scannet candidate 是有效 upper bound 但 diagnostic-only；obs-bank method 几乎为 0。

B13 multi-window D4RT:
  修复尝试：O9 前 96 连续帧、25 windows、10 mask frames/scene。
  结果：coverage 增加，但 duplicate/conflict 爆炸，AP 大幅下降，S0/S1 仍失败。

B14 O9 summary rate bug:
  修复尝试：修正多窗口 positive_mask_sample_rate 聚合。
  结果：rate=0.9875，AP 不变。
```

## 追加分析

```text
1. cross-support 失败已经不是 evaluator protocol 问题。
   同一矩阵中 P0 Stream3D on method support 往往很高：
   O9 support 上 P0 = 0.3494/0.5438/0.6462，而 O9 own = 0.0402/0.1243/0.2943。
   说明 support universe 本身可以评价好 object；方法 object quality/competition 不够。

2. 单 clip B1 的主要瓶颈是 scene/object coverage。
   O7 降低 birth 阈值后 GT crop 仍是 8.20/40.60，说明一个 16f window 看不到足够 scene。

3. 多窗口方向能提高 coverage，但会引入强重复和冲突。
   O9 把 mask frames 从 2 提到 10、GT crop 提到 12.60、#pred 到 63.4，
   但 conflict 升到 0.6847，AP 反而从 B1 的 0.3284 降到 0.0402。

4. full-scene candidate 质量是关键瓶颈。
   O8 scannet slot upper diagnostic 可达 0.2707 AP，
   但 obs-bank candidates 覆盖 55% pre_points 仍只有 0.001 AP 量级。
   因此不是“覆盖大就好”，而是需要高质量 full-scene masks / robust object memory。

5. support completion 不能替代 object inference。
   O6 all 把 S0 support 填满但 AP 只有 0.0063，说明最近邻扩张没有语义边界。

6. 目前最可信的后续方向不是继续调阈值，而是：
   multi-window object slot state + cross-window duplicate suppression/merge + cannot-link negative evidence。
   换句话说，需要把 O9 的多窗口 coverage 和 O1/B1 的 clean ownership 结合，而不是简单 union/export。
```

## 追加结论

```text
success=False for cross-support top priority
success=True for exhaustive probe5 repair attempts and audit logging
success=True for identifying likely next blocker: cross-window object memory / duplicate suppression

当前不能 claim：
  B1/O1/O4/O5/O6/O7/O8/O9 robustly outperform Stream3D。
  method-on-S0 或 method-on-S1 达标。
  full ScanNet 结果。

可以 claim：
  B1/O1 在 own sparse support 上有真实 D4RT ownership signal。
  O8 diagnostic upper bound 证明 B1 slots 能匹配 high-quality full-scene candidates。
  O9 证明增加 D4RT windows 可以提高 coverage，但 naive per-window export 会产生严重重复/conflict。

下一步优先级：
  1. 在 O9 上做 cross-window duplicate suppression：同一 mask/object slot 跨 windows 只能保留/合并一个 owner。
  2. 加 cannot-link：同帧不同 mask 或强重叠冲突不能被后续 weak positive edge 合并。
  3. 用 pending/lost object state 合并 temporal slots，而不是每个 window 独立导出。
  4. 若要提高 mask frequency，需要先修 detectron2/CropFormer 环境或换可运行的 2D mask backend。
```

## 追加 Insight

```text
这轮 cross-support 的关键 insight 是：

“more support” 和 “better objects” 是两件事。

O6 all 给了更多 support，但没有 object boundary，所以失败。
O7 给了更多 births，但仍局限在同一小 clip，所以失败。
O8 obs-bank 给了更多 full-scene mask candidates，但 candidate quality/duplication 太差，所以失败。
O9 给了更多 D4RT windows 和 mask frames，但没有跨窗口 object memory，所以 conflict 爆炸。

唯一看起来可行的信号是 O8 scannet upper bound：
  当 full-scene candidate 本身足够好时，B1/D4RT slots 可以选出非零且相对强的 S0 diagnostic AP。

因此 v9 后续若要真正解决 cross-support，核心不是再扩一圈 support，
而是把 D4RT surfel ownership 变成跨窗口 object memory：
  birth -> update -> merge/reject -> lost/pending -> export one stable object。
```

## 追加证据链索引

Code:

```text
Stream3D/tools/build_union_prepoints.py
Stream3D/tools/diagnose_v9_d4rt_mask_propagation.py
Stream3D/tools/export_v9_propagated_slot_field.py
Stream3D/tools/complete_prediction_to_support.py
Stream3D/tools/export_v8_surfel_object_field.py
Stream3D/tools/slotwise_candidate_select.py
Stream3D/tools/split_core_fringe_prediction.py
Stream3D/scripts/reproduce_v9_s4_phase4b.sh
Stream3D/scripts/reproduce_v9_phase5_o5_cross_support.sh
Stream3D/scripts/reproduce_v9_o6_support_completion_cross_support.sh
Stream3D/scripts/reproduce_v9_o7_birth_recall_cross_support.sh
Stream3D/scripts/reproduce_v9_o8_slot_candidate_cross_support.sh
Stream3D/scripts/reproduce_v9_o9_multwindow_cross_support.sh
Stream3D/scripts/v9_s4_phase4b_matrix_probe5.json
Stream3D/scripts/v9_phase5_o5_matrix_probe5.json
Stream3D/scripts/v9_o6_support_completion_matrix_probe5.json
Stream3D/scripts/v9_o7_birth_recall_matrix_probe5.json
Stream3D/scripts/v9_o8_slot_candidate_matrix_probe5.json
Stream3D/scripts/v9_o9_multwindow_matrix_probe5.json
```

Outputs:

```text
Stream3D/outputs/audit/v9_s4_phase4b/s4_phase4b_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_phase5/d4rt_mask_propagation_probe5_vis05.{json,csv,md}
Stream3D/outputs/audit/v9_phase5/o5_cross_support_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o6_support_completion/o6_support_completion_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o7_birth_recall/o7_birth_recall_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o8_slot_candidate/o8_slot_candidate_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o9_multwindow/o9_multwindow_matrix_probe5.{json,csv,md}
Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v9_g1_grid16m002_probe5_96f_stride20_loger/summary.{json,csv,md}
Stream3D/outputs/v9_multwindow/stream4d_v9_o9_b1_multwin96_grid16_mc08_probe5_summary.{json,csv,md}
```

Logs:

```text
Stream3D/logs/stream4d_v9_s4_phase4b_*.log
Stream3D/logs/stream4d_v9_phase5_*.log
Stream3D/logs/stream4d_v9_o6_*.log
Stream3D/logs/stream4d_v9_o7_*.log
Stream3D/logs/stream4d_v9_o8_*.log
Stream3D/logs/stream4d_v9_o9_*.log
```

## 追加审计包

```text
latest cross-support packet:
  stream4d_v9_code_audit_packet_20260609_0652_cross_support.zip

latest cross-support sha256:
  6e842bdd773097ffd9fc04800e65f9ff68a7a4d69c183ac0a492b128401a9b76

latest cross-support filelist:
  stream4d_v9_code_audit_packet_20260609_0652_cross_support_filelist.txt

latest cross-support zip test:
  No errors detected in compressed data of stream4d_v9_code_audit_packet_20260609_0652_cross_support.zip.

latest cross-support file count:
  1007
```

## O10 overlap suppression 追加结果

当前状态更新：

```text
cross-support top priority 仍未达成。
O10 是对 O9 multi-window 的跨窗口 duplicate/conflict 修复尝试。
O10 minIoC 0.50 显著改善 O9 own support AP 和 conflict，但 method-on-S0 / method-on-S1 仍失败。
因此不能 claim v9 已解决 cross-support。
```

### O10 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/tools/support_aware_object_rank.py` | 新增 `--tmp-policy {input,recompute}`、`--eval-policy`、`--diagnostic-only`，并写 `config_manifest.json` | O10 需要把 overlap suppression 后的输出作为 reportable method config；TMP 必须可 recompute，避免 own support 仍指向 O9 输入 |
| `Stream3D/scripts/reproduce_v9_o10_overlap_suppression_cross_support.sh` | 新增 O10 一键复现脚本 | 记录 overlap suppression 的 export/eval/cross-eval/matrix/reportable/integrity 全流程，方便复现 |
| `Stream3D/scripts/v9_o10_overlap_suppression_matrix_probe5.json` | 新增 O10 cross-support matrix | 统一比较 O10 own、P0-on-O10、O10-on-S0、O10-on-S1 |
| `docs/stream4d_v9_执行日志.md`、`docs/stream4d_v9_实验结果复盘.md` | 追加 O10 命令、结果和分析 | 满足 cross-support top priority 的审计记录要求 |

### O10 方法约束

```text
输入：
  O9 multi-window D4RT object predictions
  config = stream4d_v9_o9_b1_multwin96_grid16_mc08_probe5

方法：
  support-aware object rank
  quality_mode = score_support_area_conflict_penalty
  score_weight = 0.25
  overlap_mode = min_ioc
  overlap_threshold = 0.50 / 0.70 / 0.85
  min_support_area = 20
  tmp_policy = recompute

uses_gt=False
is_method_result=True
diagnostic_only=False
```

### O10 probe5 AP matrix

| Row | AP | AP50 | AP25 | pre ratio | GT crop/full | #pred | conflict |
|---|---:|---:|---:|---:|---|---:|---:|
| `O9 multi-window own` | `0.040167` | `0.124304` | `0.294343` | `0.082626` | `12.60/40.60` | `63.40` | `0.684667` |
| `O10 minIoC 0.50 own` | `0.157425` | `0.376554` | `0.686405` | `0.070985` | `12.60/40.60` | `24.60` | `0.153755` |
| `P0 Stream3D on O10 0.50` | `0.368456` | `0.556364` | `0.664697` | `0.070985` | `12.60/40.60` | `128.20` | `0.2213` |
| `O10 0.50 on S0` | `0.000223` | `0.001530` | `0.012000` | `0.846744` | `40.60/40.60` | `24.60` | `0.153755` |
| `O10 0.50 on S1` | `0.029484` | `0.089440` | `0.202250` | `0.045145` | `19.60/40.60` | `24.60` | `0.153755` |
| `O10 minIoC 0.70 own` | `0.101223` | `0.265244` | `0.562828` | `0.076435` | `12.60/40.60` | `35.00` | `0.300720` |
| `O10 0.70 on S0` | `0.000182` | `0.001250` | `0.009998` | `0.846744` | `40.60/40.60` | `35.00` | `0.300720` |
| `O10 0.70 on S1` | `0.025296` | `0.076849` | `0.167189` | `0.045145` | `19.60/40.60` | `35.00` | `0.300720` |
| `O10 minIoC 0.85 own` | `0.068145` | `0.201929` | `0.443341` | `0.081425` | `12.60/40.60` | `48.60` | `0.550321` |
| `O10 0.85 on S0` | `0.000150` | `0.001041` | `0.007994` | `0.846744` | `40.60/40.60` | `48.60` | `0.550321` |
| `O10 0.85 on S1` | `0.023424` | `0.075299` | `0.130966` | `0.045145` | `19.60/40.60` | `48.60` | `0.550321` |

O10 export summary：

```text
minIoC 0.50:
  mean_num_instances_before=63.4
  mean_num_instances_after_competition=24.6
  mean_num_suppressed_by_overlap=38.8
  mean_output_union_count=15067.4
  mean_num_score_support_points=17448.4
  mean_mean_conflict_ratio=0.8634897583301928
  mean_mean_unique_ratio=0.13651024166980721

minIoC 0.70:
  mean_num_instances_after_competition=35.0
  mean_num_suppressed_by_overlap=28.4
  mean_output_union_count=16145.6

minIoC 0.85:
  mean_num_instances_after_competition=48.6
  mean_num_suppressed_by_overlap=14.8
  mean_output_union_count=17174.0
```

O10 审计：

```text
reportable scan:
  num_configs=3
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=3
  num_diagnostic_only_configs=0
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
  num_configs_missing_eval_policy=0

metric integrity:
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  gt_files_read_by_rescore=False
  num_reportable_method_configs=3
```

### O10 判断

```text
O10 minIoC 0.50 是 O10 三个阈值中最强：
  own AP/AP50/AP25 = 0.157425 / 0.376554 / 0.686405
  conflict 从 O9 的 0.684667 降到 0.153755
  #pred 从 O9 的 63.40 降到 24.60

这证明 duplicate suppression 是正确修复方向之一。

但 O10 仍没有解决 cross-support：
  on S0 只有 0.000223 / 0.001530 / 0.012000
  on S1 只有 0.029484 / 0.089440 / 0.202250

P0 Stream3D on O10 support = 0.368456 / 0.556364 / 0.664697，
说明 O10 的 support crop 内仍存在可被高质量 full-scene object 做好的评价空间；
O10 自身 object proposal 仍缺 full-scene coverage 和稳定 object identity。
```

## O10 后 blocker 修复尝试补充

```text
B15 cross-window duplicate suppression:
  问题：
    O9 multi-window 使 coverage 变大，但同一 object 跨 window 被重复导出，conflict=0.684667。

  修复尝试：
    O10 support-aware object rank + min-IoC suppression。
    threshold = 0.50 / 0.70 / 0.85。
    tmp_policy=recompute，manifest reportable。

  结果：
    O10 0.50 own AP 从 O9 的 0.040167 提升到 0.157425；
    AP50 从 0.124304 提升到 0.376554；
    AP25 从 0.294343 提升到 0.686405；
    conflict 从 0.684667 降到 0.153755。

  失败点：
    O10 0.50 on S0 仍只有 0.000223 / 0.001530 / 0.012000；
    O10 0.50 on S1 仍只有 0.029484 / 0.089440 / 0.202250。

  判断：
    overlap suppression 修复了重复，但没有补足 scene-wide object coverage；
    cross-support 仍未达成。
```

## O10 后分析

```text
1. O10 证明 O9 的主要局部错误之一确实是重复/conflict。
   minIoC 0.50 直接把 conflict 降到 0.1538，own AP/AP50/AP25 全部显著提升。

2. O10 也证明“只做 suppression”不够。
   它保留更干净的 24.6 个 objects，但 GT crop/full 仍是 12.60/40.60。
   也就是说它没有新增 missing object，只是在已有 O9 coverage 内清理重复。

3. O10 on S0 几乎为零，说明方法输出仍不是 full-scene prediction。
   S0 的 pre ratio 是 0.846744，O10 union 只有约 0.07；
   当强行用 full-scene support 评价 sparse O10 object 时，大多数 GT 仍 missed。

4. O10 on S1 有小幅可见信号，但离达标很远。
   O10 0.50 on S1 AP25 = 0.202250，高于 O9 on S1 的 0.080372；
   这说明 suppression 改善了 32f support 上的局部 object quality，但不是 robust cross-support。

5. 下一步不能只调 threshold。
   0.50/0.70/0.85 呈现明确 tradeoff：
     0.50 最干净、own AP 最好；
     0.85 保留更多 support，但 conflict 回升、AP 下降。
   因此瓶颈不是单一阈值，而是缺少跨窗口 object memory 和 missing-object birth/update。
```

## O10 后结论

```text
success=False for cross-support top priority
success=True for O9 duplicate/conflict repair attempt
success=True for reportable method audit on O10 configs

当前最强 cross-support 修复结果：
  O10 minIoC 0.50 own = 0.157425 / 0.376554 / 0.686405
  O10 minIoC 0.50 on S0 = 0.000223 / 0.001530 / 0.012000
  O10 minIoC 0.50 on S1 = 0.029484 / 0.089440 / 0.202250

因此不能 claim cross-support solved。

目前最可信下一步：
  1. 在 O9/O10 基础上做真正 object memory，而不是一次性 overlap suppression；
  2. object state 需要 birth/update/merge/reject/lost；
  3. 同一 object 跨 windows 合并后只导出一个稳定实例；
  4. missing-object coverage 需要更多高质量 mask candidates 或可运行的 dense mask backend；
  5. cannot-link negative evidence 需要阻止不同 mask 的重复弱合并。
```

## O10 Insight

```text
O10 给出的新 insight 是：

duplicate suppression 是必要条件，但不是充分条件。

O9 的失败像是“看见更多，但每次都重新命名同一批东西”；
O10 把重复命名压掉后，own support 质量明显回升。
但 cross-support 要求的是 scene-level object field，
它需要同时解决两个问题：
  已见 object 的跨窗口合并；
  未见或低覆盖 object 的 birth / candidate completion。

所以 v9 的核心路线应从“把 window proposals 导出得更干净”
升级为“维护一个跨窗口 object memory，再从 memory 导出 scene object field”。
```

## O10 证据链追加

Code:

```text
Stream3D/tools/support_aware_object_rank.py
Stream3D/scripts/reproduce_v9_o10_overlap_suppression_cross_support.sh
Stream3D/scripts/v9_o10_overlap_suppression_matrix_probe5.json
```

Outputs:

```text
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o10_o9_overlap_mioc050_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o10_o9_overlap_mioc070_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o10_o9_overlap_mioc085_probe5_summary.json
Stream3D/outputs/audit/v9_o10_overlap_suppression/o10_overlap_suppression_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o10_overlap_suppression/reportable_config_scan_o10_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o10_overlap_suppression/metric_integrity_o10_probe5.{json,md}
```

Logs:

```text
Stream3D/logs/stream4d_v9_o10_overlap_suppression_py_compile.log
Stream3D/logs/stream4d_v9_o10_o9_overlap_mioc050_export.log
Stream3D/logs/stream4d_v9_o10_o9_overlap_mioc070_export.log
Stream3D/logs/stream4d_v9_o10_o9_overlap_mioc085_export.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc050_probe5_eval.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc070_probe5_eval.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc085_probe5_eval.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc050_on_s0_probe5.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc050_on_s1_probe5.log
Stream3D/logs/stream4d_v9_o10_overlap_suppression_matrix_probe5.log
Stream3D/logs/stream4d_v9_o10_overlap_suppression_reportable_scan.log
Stream3D/logs/stream4d_v9_o10_overlap_suppression_metric_integrity.log
Stream3D/logs/stream4d_v9_o10_final_py_compile.log
Stream3D/logs/stream4d_v9_o10_final_bash_n.log
Stream3D/logs/stream4d_v9_o10_final_unit_tests.log
```

## O10 追加审计包

```text
latest cross-support O10 packet:
  stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10.zip

latest cross-support O10 sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10.sha256

latest cross-support O10 filelist:
  stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10_filelist.txt

latest cross-support O10 zip test:
  No errors detected in compressed data of stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10.zip.

latest cross-support O10 file count:
  1091

previous cross-support packet:
  stream4d_v9_code_audit_packet_20260609_0652_cross_support.zip
```

## O11-O13 继续推进结果

当前状态更新：

```text
cross-support top priority 仍未达成。
O11/O12/O13 继续尝试了三类局部修复：
  O11: obs-bank candidates overlap suppression
  O12: O10 D4RT-clean + O11 obs-bank-clean fusion，再 suppression
  O13: O12 best core 最近邻 completion 到 S0 support

当前 strongest own-support repair = O12 fused overlap 0.50:
  AP/AP50/AP25 = 0.177930 / 0.380577 / 0.643669

当前 strongest S1 repair = O13 r0.10:
  AP/AP50/AP25 = 0.054274 / 0.146604 / 0.309673

但 S0 仍失败：
  O12 on S0 = 0.000301 / 0.001498 / 0.026504
  O13 all own with near-full S0 support = 0.000692 / 0.001785 / 0.065276
```

### O11 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/scripts/reproduce_v9_o11_obsbank_overlap_cross_support.sh` | 新增 O11 复现脚本 | 将 O10 的 overlap suppression 迁移到 O8 非 GT obs-bank top80 candidates |
| `Stream3D/scripts/v9_o11_obsbank_overlap_matrix_probe5.json` | 新增 O11 unified/cross-support matrix | 统一比较 O8 obs-bank top80、O11 own、P0-on-O11、O11-on-S0、O11-on-S1 |

O11 方法约束：

```text
input_config=stream4d_v9_o8_obsbank_slot_top80_probe5
quality_mode=score_support_area_conflict_penalty
score_weight=0.25
overlap_mode=min_ioc
threshold=0.50 / 0.70
tmp_policy=recompute
uses_gt=False
is_method_result=True
```

O11 结果：

| Row | AP | AP50 | AP25 | pre ratio | GT crop/full | #pred | conflict |
|---|---:|---:|---:|---:|---|---:|---:|
| `O8 obs-bank slot+top80 own` | `0.000975` | `0.004807` | `0.039085` | `0.552481` | `37.80/40.60` | `96.40` | `0.704240` |
| `O11 minIoC 0.50 own` | `0.136121` | `0.296384` | `0.654162` | `0.071923` | `11.60/40.60` | `20.40` | `0.179091` |
| `O11 0.50 on S0` | `0.000429` | `0.002271` | `0.020027` | `0.846744` | `40.60/40.60` | `20.40` | `0.179091` |
| `O11 0.50 on S1` | `0.023517` | `0.063834` | `0.146866` | `0.045145` | `19.60/40.60` | `20.40` | `0.179091` |
| `O11 minIoC 0.70 own` | `0.108074` | `0.223622` | `0.618033` | `0.081563` | `12.00/40.60` | `33.40` | `0.405934` |
| `O11 0.70 on S0` | `0.000429` | `0.002271` | `0.018948` | `0.846744` | `40.60/40.60` | `33.40` | `0.405934` |
| `O11 0.70 on S1` | `0.022953` | `0.064411` | `0.144321` | `0.045145` | `19.60/40.60` | `33.40` | `0.405934` |

O11 审计：

```text
reportable scan:
  num_reportable_method_configs=2
  num_configs_missing_manifest=0
  num_configs_missing_eval_policy=0
  num_uses_gt_and_method_result=0

metric integrity:
  phase0_pass=True
  gt_files_read_by_rescore=False
```

判断：

```text
O11 证明 O8 obs-bank top80 的主要局部问题也是 duplicate/conflict。
去重后 own AP 从 0.000975 提到 0.136121。
但去重会把 pre ratio 从 0.552481 降到 0.071923，S0/S1 仍失败。
```

### O12 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/tools/fuse_prediction_configs.py` | 新增 `--eval-policy`，并写入 manifest extra | O12 union config 必须通过 reportable scan；第一次执行暴露缺 eval_policy |
| `Stream3D/scripts/reproduce_v9_o12_fused_o10_o11_cross_support.sh` | 新增 O12 fusion + suppression 复现脚本 | 测试 O10 D4RT-clean 与 O11 obs-bank-clean 是否互补 |
| `Stream3D/scripts/v9_o12_fused_o10_o11_matrix_probe5.json` | 新增 O12 unified/cross-support matrix | 统一比较 O10、O11、O12 union、O12 suppression、cross-support |

O12 blocker 修复：

```text
第一次执行失败：
  reportable scan: num_configs_missing_eval_policy=1

修复：
  fuse_prediction_configs.py 新增 --eval-policy；
  O12 union export 添加 --eval-policy own_recompute_o10_o11_union。

重跑结果：
  reportable scan pass
  metric integrity phase0_pass=True
```

O12 结果：

| Row | AP | AP50 | AP25 | pre ratio | GT crop/full | #pred | conflict |
|---|---:|---:|---:|---:|---|---:|---:|
| `O10 D4RT overlap 0.50 own` | `0.157425` | `0.376554` | `0.686405` | `0.070985` | `12.60/40.60` | `24.60` | `0.153755` |
| `O11 obs-bank overlap 0.50 own` | `0.136121` | `0.296384` | `0.654162` | `0.071923` | `11.60/40.60` | `20.40` | `0.179091` |
| `O12 fused union own` | `0.079668` | `0.217072` | `0.543820` | `0.091712` | `13.00/40.60` | `45.00` | `0.577206` |
| `O12 fused overlap 0.50 own` | `0.177930` | `0.380577` | `0.643669` | `0.085801` | `12.60/40.60` | `26.00` | `0.192014` |
| `P0 Stream3D on O12 overlap` | `0.359558` | `0.567568` | `0.682199` | `0.085801` | `12.60/40.60` | `128.20` | `0.2213` |
| `O12 overlap on S0` | `0.000301` | `0.001498` | `0.026504` | `0.846744` | `40.60/40.60` | `26.00` | `0.192014` |
| `O12 overlap on S1` | `0.045242` | `0.126188` | `0.232727` | `0.045145` | `19.60/40.60` | `26.00` | `0.192014` |

O12 fusion summary：

```text
O12 union:
  mean_num_primary_instances=24.6
  mean_num_secondary_instances=20.4
  mean_num_output_instances=45.0
  mean_output_union_count=19558.4
  mean_output_union_ratio=0.09171196302908038

O12 overlap:
  mean_num_instances_before=45.0
  mean_num_instances_after_competition=26.0
  mean_num_suppressed_by_overlap=19.0
  mean_output_union_count=18322.4
```

判断：

```text
O12 是当前 own-support 最强局部修复：
  AP=0.177930，比 O10/O11 都高。
O12 on S1 也提升到 0.045242 / 0.126188 / 0.232727。
但 O12 on S0 仍只有 0.000301 / 0.001498 / 0.026504；
因此 fusion 不能解决 full-scene cross-support。
```

### O13 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/scripts/reproduce_v9_o13_o12_support_completion_cross_support.sh` | 新增 O13 复现脚本 | 验证 O12 sparse support 是否能通过最近邻 completion 扩到 S0 |
| `Stream3D/scripts/v9_o13_o12_support_completion_matrix_probe5.json` | 新增 O13 matrix | 统一比较 O12、O13 r0.10、O13 all、S1 cross-support |

O13 方法约束：

```text
input_config=stream4d_v9_o12_o10_o11_union_overlap_mioc050_probe5
target_support_config=scannet
max_radius=0.10 / -1.0(all)
keep_core_points=True
uses_gt=False
is_method_result=True
```

O13 结果：

| Row | AP | AP50 | AP25 | pre ratio | GT crop/full | #pred | conflict |
|---|---:|---:|---:|---:|---|---:|---:|
| `O12 overlap own` | `0.177930` | `0.380577` | `0.643669` | `0.085801` | `12.60/40.60` | `26.00` | `0.192014` |
| `O13 r0.10 own` | `0.064846` | `0.167195` | `0.450757` | `0.166961` | `14.00/40.60` | `26.00` | `0.100829` |
| `O13 r0.10 on S1` | `0.054274` | `0.146604` | `0.309673` | `0.045145` | `19.60/40.60` | `26.00` | `0.100829` |
| `O13 all own` | `0.000692` | `0.001785` | `0.065276` | `0.848995` | `40.60/40.60` | `26.00` | `0.022499` |
| `O13 all on S1` | `0.042918` | `0.114857` | `0.352031` | `0.045145` | `19.60/40.60` | `26.00` | `0.022499` |

O13 support summary：

```text
O13 r0.10:
  num_output_points_mean=35021.8
  output_point_ratio_mean=0.16696074059544355
  target_support_fill_ratio_mean=0.1986002059966709
  conflict_rate_mean=0.10082923725773804

O13 all:
  num_output_points_mean=186634.0
  output_point_ratio_mean=0.848994888108882
  target_support_fill_ratio_mean=1.0027125358278444
  conflict_rate_mean=0.022498879427461715
```

判断：

```text
O13 r0.10 增加 coverage，但 own AP 从 O12 的 0.177930 降到 0.064846。
O13 all 几乎填满 S0 support，GT crop/full 达到 40.60/40.60，但 AP 崩到 0.000692。
这说明 cross-support 失败不是单纯 support 缺口；
最近邻 completion 会破坏 object boundary/identity，不能作为解法。
```

## O11-O13 追加分析

```text
1. suppression 是必要但不充分。
   O10/O11/O12 都显示去重可以显著提升 own AP；
   但去重后的支持仍是约 7%-9% scene points，无法通过 S0。

2. fusion 能提高局部对象质量，但没有形成 scene-level field。
   O12 own AP 最高，为 0.177930；
   但 S0 AP 仍 0.000301，说明它仍是局部 sparse field。

3. 最近邻 completion 明确失败。
   O13 all 已把 support ratio 提到 0.848995、GT crop/full 到 40.60/40.60；
   但 AP/AP50/AP25 只有 0.000692/0.001785/0.065276。
   这排除了“只要填满 support 就能解决 cross-support”的假设。

4. S1 有一些持续改善，但不是最终目标。
   O12 on S1 = 0.045242/0.126188/0.232727；
   O13 r0.10 on S1 = 0.054274/0.146604/0.309673。
   这说明局部 32f support 上 object quality 在变好，
   但 full-scene S0 仍需要真正的 scene-level object memory/candidates。

5. 当前可执行的局部修复路线已经基本耗尽。
   O4/O6/O10/O11/O12/O13 分别验证了 fringe、completion、suppression、obs-bank、fusion、completion-all。
   失败模式一致：没有高质量 full-scene object boundary 和稳定 object identity。
```

## O11-O13 后结论

```text
success=False for cross-support top priority
success=True for O11/O12/O13 repair attempts and audit logging
success=True for narrowing blocker:
  cross-support 不是 evaluator bug；
  不是单纯 duplicate/conflict；
  不是单纯 support coverage；
  是缺 scene-level object memory / high-quality full-scene candidates。

当前最强 method rows:
  best own AP:
    O12 fused overlap 0.50 = 0.177930 / 0.380577 / 0.643669

  best S1:
    O13 r0.10 on S1 = 0.054274 / 0.146604 / 0.309673

  S0 仍失败:
    O12 on S0 = 0.000301 / 0.001498 / 0.026504
    O13 all own with S0-like support = 0.000692 / 0.001785 / 0.065276

因此不能 claim v9 cross-support solved。
下一步不应继续只调阈值；
需要实现跨窗口 object memory 或修复/替换 2D mask backend 以得到高质量 full-scene candidates。
```

## O11-O13 Insight

```text
这组三个追加实验把 cross-support 失败拆得更清楚：

O11 说明：
  obs-bank candidates 不是完全没信号，去重后 own AP 可到 0.136；
  但它的 full-scene support 被压掉后仍回到 sparse field。

O12 说明：
  D4RT-clean 和 obs-bank-clean 互补有限；
  fusion 后再去重能得到当前最好的 own AP，但仍不是 full-scene object field。

O13 说明：
  full support 本身不够；
  如果 object boundary/identity 错，填满 S0 只会把错误扩散到全场景。

所以 cross-support 的真正缺口是：
  从局部 window proposals 到 scene-level persistent object memory 的建模。
这个 memory 需要做 birth/update/merge/reject/lost，
并且需要可靠 mask candidates 或新的 2D/3D object proposal source。
```

## O11-O13 证据链追加

Code:

```text
Stream3D/tools/fuse_prediction_configs.py
Stream3D/scripts/reproduce_v9_o11_obsbank_overlap_cross_support.sh
Stream3D/scripts/reproduce_v9_o12_fused_o10_o11_cross_support.sh
Stream3D/scripts/reproduce_v9_o13_o12_support_completion_cross_support.sh
Stream3D/scripts/v9_o11_obsbank_overlap_matrix_probe5.json
Stream3D/scripts/v9_o12_fused_o10_o11_matrix_probe5.json
Stream3D/scripts/v9_o13_o12_support_completion_matrix_probe5.json
```

Outputs:

```text
Stream3D/outputs/audit/v9_o11_obsbank_overlap/o11_obsbank_overlap_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o11_obsbank_overlap/reportable_config_scan_o11_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o11_obsbank_overlap/metric_integrity_o11_probe5.{json,md}
Stream3D/outputs/audit/v9_o12_fused_o10_o11/o12_fused_o10_o11_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o12_fused_o10_o11/reportable_config_scan_o12_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o12_fused_o10_o11/metric_integrity_o12_probe5.{json,md}
Stream3D/outputs/audit/v9_o13_o12_support_completion/o13_o12_support_completion_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o13_o12_support_completion/reportable_config_scan_o13_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o13_o12_support_completion/metric_integrity_o13_probe5.{json,md}
Stream3D/outputs/v9_fusion/stream4d_v9_o12_o10_o11_union_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o11_obsbank_overlap_mioc050_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o11_obsbank_overlap_mioc070_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o12_o10_o11_union_overlap_mioc050_probe5_summary.json
Stream3D/outputs/v9_support_completion/stream4d_v9_o13_o12_complete_s0_r010_probe5_summary.{json,csv,md}
Stream3D/outputs/v9_support_completion/stream4d_v9_o13_o12_complete_s0_all_probe5_summary.{json,csv,md}
```

Logs:

```text
Stream3D/logs/stream4d_v9_o11_*.log
Stream3D/logs/stream4d_v9_o12_*.log
Stream3D/logs/stream4d_v9_o13_*.log
Stream3D/logs/stream4d_v9_o13_final_py_compile.log
Stream3D/logs/stream4d_v9_o13_final_bash_n.log
Stream3D/logs/stream4d_v9_o13_final_unit_tests.log
```

## O13 后最终验证

```text
py_compile: pass
bash -n: pass
unit tests: Ran 13 tests in 0.123s, OK
```

## O13 最终审计包

```text
latest final cross-support packet:
  stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.zip

latest final cross-support sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.sha256

latest final cross-support filelist:
  stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final_filelist.txt

latest final cross-support zip test:
  see sibling file stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final_ziptest.log

latest final cross-support file count:
  1250

previous O10 packet:
  stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10.zip
```

## O14-O28 继续复盘：cross-support top priority

日期：2026-06-09（Asia/Singapore）  
新增执行日志位置：`docs/stream4d_v9_执行日志.md` 的 “O14-O28 cross-support 继续修复” 章节。  
本节只记录真实跑出的数据。matrix markdown 里的 AP 是百分数；本复盘表格统一写 evaluator 原始小数。

### 当前状态

```text
cross-support top priority 仍未达成。

当前最强 reportable method-on-S0：
  O26 inside050 logarea on S0 = 0.021517 / 0.100351 / 0.291694

当前最强 reportable AP25 on S0：
  O25 inside050 on S0 = 0.021388 / 0.100300 / 0.292631

当前最强 diagnostic-only S0-aware row：
  O28 S0-aware rank diagnostic on S0 = 0.023637 / 0.095270 / 0.267036

P0 Stream3D on S0 reference:
  0.235730 / 0.414306 / 0.537786

因此不能 claim v9 已解决 cross-support。
```

### 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/tools/merge_overlapping_prediction_masks.py` | 新增 overlap connected-component merge，按 prediction mask min-IoC 合并，不读 GT，写 manifest | O15/O16/O18 用于测试 full-span candidates duplicate/conflict 是否可通过 merge 修复 |
| `Stream3D/tools/object_competition_rank.py` | 新增 `--eval-policy` 并写入 manifest | O17 competition/ranking rescue 需要可审计 eval policy |
| `Stream3D/tools/rescore_prediction_scores.py` | 新增 manifest、`--eval-policy`、`inverse_area/inverse_sqrt_area/inverse_log_area` score features | O19/O26 测试非 GT score calibration 是否能提升 cross-support |
| `Stream3D/tools/self_discovered_boundary_refine.py` | 新增 manifest、`--eval-policy` | O25 用 RGB-D + non-GT 2D mask 做 boundary/negative evidence refine，必须可审计 |
| `Stream3D/tools/self_discovered_silhouette_score.py` | 新增 manifest、`--eval-policy`、`--diagnostic-only` | O27 用 self-discovered silhouette agreement 重排；修复原工具缺 manifest 的问题 |
| `Stream3D/scripts/reproduce_v9_o14_fullspan_grid8_cross_support.sh`、`v9_o14_fullspan_grid8_matrix_probe5.json` | 新增 O14 full-span ws100 复现 | 测试全场景 D4RT windows 是否改善 S0 |
| `Stream3D/scripts/reproduce_v9_o15_o14_overlap_merge_cross_support.sh`、`v9_o15_o14_overlap_merge_matrix_probe5.json` | 新增 O15 merge 复现 | 测试 raw full-span candidates 的 duplicate merge |
| `Stream3D/scripts/reproduce_v9_o16_o14_overlap_merge_threshold_cross_support.sh`、`v9_o16_o14_overlap_merge_threshold_matrix_probe5.json` | 新增 O16 threshold sweep | 测试 merge threshold 对 S0/S1 的影响 |
| `Stream3D/scripts/reproduce_v9_o17_o14_competition_rescue_cross_support.sh`、`v9_o17_o14_competition_rescue_matrix_probe5.json` | 新增 O17 ranking/rescue 复现 | 测试 ranking/competition 是否能救 full-span candidates |
| `Stream3D/scripts/reproduce_v9_o18_o14_merge_fine_threshold_cross_support.sh`、`v9_o18_o14_merge_fine_threshold_matrix_probe5.json` | 新增 O18 fine threshold sweep | 细扫 overlap merge threshold |
| `Stream3D/scripts/reproduce_v9_o19_score_calibration_cross_support.sh`、`v9_o19_score_calibration_matrix_probe5.json` | 新增 O19 score calibration 复现；修正 matrix cross output_config 命名 | 测试 merge 后是否主要是 score/order 问题 |
| `Stream3D/scripts/reproduce_v9_o20_fullspan_grid8_ws50_cross_support.sh`、`v9_o20_fullspan_grid8_ws50_matrix_probe5.json` | 新增 O20 ws50 复现 | 提高 full-span window density |
| `Stream3D/scripts/reproduce_v9_o21_o20_merge_threshold_cross_support.sh`、`v9_o21_o20_merge_threshold_matrix_probe5.json` | 新增 O21 O20 threshold sweep | 测试 ws50 merge threshold |
| `Stream3D/scripts/reproduce_v9_o22_fullspan_grid8_ws25_cross_support.sh`、`v9_o22_fullspan_grid8_ws25_matrix_probe5.json` | 新增 O22 ws25 复现 | 进一步提高 window density |
| `Stream3D/scripts/reproduce_v9_o23_o22_merge_threshold_cross_support.sh`、`v9_o23_o22_merge_threshold_matrix_probe5.json` | 新增 O23 O22 threshold sweep | 找 ws25 最佳 cross-support tradeoff |
| `Stream3D/scripts/reproduce_v9_o24_maskaware_ws50_cross_support.sh`、`v9_o24_maskaware_ws50_matrix_probe5.json` | 新增 O24 mask-aware query 复现 | 测试 mask-aware D4RT query densification |
| `Stream3D/scripts/reproduce_v9_o25_boundary_refine_cross_support.sh`、`v9_o25_boundary_refine_matrix_probe5.json` | 新增 O25 boundary refine 复现；修正 matrix cross output_config 命名 | 当前最有效 S0 修复方向 |
| `Stream3D/scripts/reproduce_v9_o26_boundary_refine_rescore_cross_support.sh`、`v9_o26_boundary_refine_rescore_matrix_probe5.json` | 新增 O26 O25 后 score calibration；修正 matrix cross output_config 命名 | 测试 O25 后排序是否仍有收益 |
| `Stream3D/scripts/reproduce_v9_o27_silhouette_score_cross_support.sh`、`v9_o27_silhouette_score_matrix_probe5.json` | 新增 O27 silhouette score 复现 | 测试 non-GT silhouette agreement 排序 |
| `Stream3D/scripts/reproduce_v9_o28_support_aware_rank_cross_support.sh`、`v9_o28_support_aware_rank_matrix_probe5.json` | 新增 O28 target-support-aware diagnostic；修复 `scannet` TMP 别名不能被 rank 工具直接读取 | 诊断即使用 target support 排序是否能救 O25 masks |
| `docs/stream4d_v9_执行日志.md`、`docs/stream4d_v9_实验结果复盘.md` | 追加 O14-O28 执行与复盘 | 满足审计和复现要求 |

### O14 full-span grid8 ws100

G1 / exporter summary：

```text
num_windows=129
num_ok_windows=129
num_failed_windows=0
num_source_queries_mean=1024
uv_in01_rate_mean=0.8860752785852714
self_uv_error_p90_mean=1.7681998460791835
track_length_visible_mean_mean=11.87823249757752
surfel_coverage_2d_per_frame_mean_mean=0.010712187419566073

raw object export:
  num_windows_mean=25.8
  num_available_mask_frames_mean=51.6
  num_raw_clusters_total_mean=592.4
  object_dict_size_mean=329.8
  num_exported_objects_mean=301.6
  num_exported_points_mean=117129.8
  positive_mask_sample_rate=0.9753585417336186
  export_conflict_rate=0.518337149340269
```

O14 matrix：

| Row | AP | AP50 | AP25 | pre ratio | #pred | conflict |
|---|---:|---:|---:|---:|---:|---:|
| `O14 raw own` | `0.013983` | `0.061317` | `0.206398` | `0.530185` | `301.60` | `0.518337` |
| `O14 overlap0.50 own` | `0.037778` | `0.114894` | `0.343145` | `0.487851` | `149.80` | `0.249676` |
| `P0 on O14 overlap` | `0.276978` | `0.465197` | `0.603135` | `0.487851` | `128.20` | `0.2213` |
| `O14 overlap on S0` | `0.000873` | `0.004532` | `0.189368` | `0.846744` | `149.80` | `0.249676` |
| `O14 overlap on S1` | `0.050867` | `0.157824` | `0.331157` | `0.045145` | `149.80` | `0.249676` |

判断：

```text
全场景 windows 明显提升 support/coverage，但 raw duplicate/conflict 很高。
O14 overlap 对 S1 有正信号，但 S0 AP/AP50 仍很低。
```

### O15-O18 merge / ranking sweep

| Row | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| `O15 merge0.50 own` | `0.043781` | `0.134961` | `0.331306` | merge 后 #pred=122.0，conflict=0.247371 |
| `O15 on S0` | `0.004527` | `0.018030` | `0.213390` | S0 有提升但仍远低 P0 |
| `O15 on S1` | `0.080973` | `0.203431` | `0.395525` | S1 比 O14 更好 |
| `O16 merge0.30 own` | `0.064622` | `0.152794` | `0.360815` | conflict=0.102500 |
| `O16 merge0.30 on S0` | `0.005703` | `0.025106` | `0.222277` | S0 小幅提升 |
| `O16 merge0.30 on S1` | `0.057982` | `0.140941` | `0.311544` | 阈值过低损 S1 |
| `O16 merge0.70 own` | `0.025851` | `0.096629` | `0.275602` | duplicate 更多 |
| `O16 merge0.70 on S0` | `0.002193` | `0.010508` | `0.163663` | 低于 0.30 |
| `O16 merge0.70 on S1` | `0.060543` | `0.168654` | `0.299582` | S1 中等 |
| `O17 area_unique g030 own` | `0.130996` | `0.282872` | `0.500056` | strict ranking own 最强之一 |
| `O17 area_unique g030 on S0` | `0.000997` | `0.005157` | `0.167036` | S0 崩 |
| `O17 area_unique g030 on S1` | `0.025559` | `0.071080` | `0.230187` | S1 下降 |
| `O17 unique_compact g050 rescue on S0` | `0.002297` | `0.011357` | `0.197719` | support/shape 仍不足 |
| `O17 unique_compact g050 rescue on S1` | `0.043957` | `0.127742` | `0.342780` | S1 有信号 |
| `O18 merge0.20 on S0` | `0.003798` | `0.014934` | `0.163178` | 过低 threshold 不够好 |
| `O18 merge0.40 on S0` | `0.005749` | `0.025290` | `0.217116` | 接近 O16 0.30 |
| `O18 merge0.40 on S1` | `0.075136` | `0.203101` | `0.443958` | S1 较强 |

判断：

```text
merge/suppression 是正确局部修复方向：可以降低 duplicate/conflict，并提升 S0 到 0.005-0.006 AP。
但 ranking 过严会把 S0 support 需要的 recall 丢掉。
O17 own support 变强不等于 cross-support 变强。
```

### O19 score calibration

| Row | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| `O19 O16 large-first own` | `0.064117` | `0.157997` | `0.377456` | log_area |
| `O19 O16 large-first on S0` | `0.005641` | `0.025284` | `0.246358` | AP25 提升 |
| `O19 O16 large-first on S1` | `0.054427` | `0.139066` | `0.296509` | S1 降 |
| `O19 O16 small-first own` | `0.041158` | `0.108962` | `0.215530` | inverse area |
| `O19 O16 small-first on S0` | `0.002879` | `0.014439` | `0.140589` | 负向 |
| `O19 O18 large-first own` | `0.060545` | `0.161403` | `0.399989` | O18 0.40 log_area |
| `O19 O18 large-first on S0` | `0.007324` | `0.030070` | `0.247198` | 当时最佳 S0 |
| `O19 O18 large-first on S1` | `0.082004` | `0.214578` | `0.456606` | S1 也较强 |
| `O19 O18 small-first on S0` | `0.002789` | `0.013545` | `0.122402` | 负向 |

判断：

```text
large-first score 对 O18 有帮助，small-first 明显变差。
但 O19 仍只有 S0 AP 0.007324，说明排序不是唯一瓶颈。
```

### O20-O24 window density / mask-aware query

G1 / raw export：

```text
O20 ws50:
  num_windows=257
  num_ok_windows=257
  uv_in01_rate_mean=0.8857258006292559
  track_length_visible_mean_mean=11.80042786357004
  raw num_available_mask_frames_mean=102.4
  raw num_exported_objects_mean=591.8
  raw num_exported_points_mean=147225.4
  raw export_conflict_rate=0.6781698289308404

O22 ws25:
  num_windows=510
  num_ok_windows=510
  uv_in01_rate_mean=0.8826017491957721
  track_length_visible_mean_mean=11.818889782475491
  raw num_available_mask_frames_mean=203.6
  raw num_exported_objects_mean=1156.2
  raw num_exported_points_mean=168260.6
  raw export_conflict_rate=0.7881821935464426

O24 mask-aware ws50:
  num_windows=257
  num_ok_windows=257
  num_source_queries_mean=1233.070038910506
  uv_in01_rate_mean=0.88387713003626
  surfel_coverage_2d_per_frame_mean_mean=0.012507928484608691
  raw num_exported_objects_mean=943.8
  raw export_conflict_rate=0.6915147124
```

Matrix：

| Row | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| `O20 raw own` | `0.006217` | `0.035016` | `0.121951` | duplicate 爆炸 |
| `O20 merge0.40 logarea on S0` | `0.013746` | `0.070405` | `0.270659` | ws50 明显优于 O19 |
| `O20 merge0.40 logarea on S1` | `0.075767` | `0.167640` | `0.412176` | S1 strong |
| `O21 merge0.50 logarea on S0` | `0.013961` | `0.060077` | `0.272304` | S0 AP25 类似 |
| `O21 merge0.50 logarea on S1` | `0.084816` | `0.181128` | `0.450340` | S1 best in O20/O21 |
| `O22 raw own` | `0.001628` | `0.007575` | `0.068081` | ws25 raw conflict=0.788182 |
| `O22 merge0.50 logarea on S0` | `0.014275` | `0.078108` | `0.259100` | ws25 有提升 |
| `O22 merge0.50 logarea on S1` | `0.054004` | `0.151406` | `0.222549` | S1 下降 |
| `O23 merge0.40 on S0` | `0.017217` | `0.091101` | `0.277387` | O23 当前最好 full-span merge |
| `O23 merge0.40 on S1` | `0.035126` | `0.080355` | `0.201905` | S1 弱 |
| `O23 merge0.60 on S1` | `0.050557` | `0.134464` | `0.236624` | S1 较好但 S0 低 |
| `O24 mask-aware 0.50 logarea on S0` | `0.013162` | `0.063595` | `0.276403` | 不如 O23 S0 |
| `O24 mask-aware 0.50 logarea on S1` | `0.062447` | `0.137396` | `0.380767` | S1 尚可 |

判断：

```text
full-span window density 是有效方向：S0 从 O19 的 0.007324/0.030070 提到 O23 的 0.017217/0.091101。
但密度越高 raw duplicate/conflict 越严重，必须配合 merge/refine。
mask-aware query 没有超过固定 grid8 ws25；更多 query 不等于更好 object field。
```

### O25 boundary refine

O25 refine summary：

```text
inside050:
  mean_changed_instances = 78.8 / 80.6
  mean_union_before = 168260.6
  mean_union_after = 154585.2
  mean_point_keep_ratio = 0.8069584563
  mean_used_observations = 11.1343
  mean_inside_ratio = 0.71059

inside070+interior010:
  mean_union_after = 150709.6
  mean_point_keep_ratio = 0.7061683036
```

Matrix：

| Row | AP | AP50 | AP25 | pre ratio | #pred | conflict |
|---|---:|---:|---:|---:|---:|---:|
| `O25 inside050 own` | `0.053131` | `0.164664` | `0.319182` | `0.703530` | `80.60` | `0.113479` |
| `O25 inside050 on S0` | `0.021388` | `0.100300` | `0.292631` | `0.846744` | `80.60` | `0.113479` |
| `O25 inside050 on S1` | `0.055330` | `0.112214` | `0.267113` | `0.045145` | `80.60` | `0.113479` |
| `O25 inside070+interior010 own` | `0.055074` | `0.160528` | `0.304713` | `0.686093` | `80.60` | `0.104560` |
| `O25 inside070+interior010 on S0` | `0.015151` | `0.081885` | `0.277673` | `0.846744` | `80.60` | `0.104560` |
| `O25 inside070+interior010 on S1` | `0.046814` | `0.096448` | `0.246778` | `0.045145` | `80.60` | `0.104560` |

判断：

```text
O25 inside050 是 O14-O28 中最有效的 reportable S0 修复之一：
  S0 = 0.021388 / 0.100300 / 0.292631。

它显著超过 O23 S0:
  O23 = 0.017217 / 0.091101 / 0.277387。

更严格 inside070+interior010 降低 conflict，但 AP/AP50/AP25 反而下降。
说明 O25 需要保留一定 fringe recall，不能只追求更干净边界。
```

### O26 boundary refine 后 score calibration

| Row | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| `O26 inside050 logarea own` | `0.053088` | `0.164607` | `0.317119` | 与 O25 own 基本持平 |
| `O26 inside050 logarea on S0` | `0.021517` | `0.100351` | `0.291694` | AP/AP50 微升，AP25 微降 |
| `O26 inside050 logarea on S1` | `0.054010` | `0.110206` | `0.267419` | S1 基本持平 |
| `O26 inside070 logarea own` | `0.054857` | `0.160241` | `0.303397` | 与 O25 inside070 接近 |
| `O26 inside070 logarea on S0` | `0.015916` | `0.085157` | `0.277151` | 仍低于 inside050 |
| `O26 inside070 logarea on S1` | `0.046023` | `0.095655` | `0.245616` | 仍低于 inside050 |

判断：

```text
O26 说明 O25 之后 score/order 已接近饱和。
inside050 logarea 是当前 reportable method-on-S0 AP/AP50 最高：
  0.021517 / 0.100351 / 0.291694。
但这个提升只有微小幅度，不能视为解决 cross-support。
```

### O27 silhouette score

O27 self-discovered silhouette summary：

```text
mean_self_silhouette_quality_mean = 0.47257861495018005
mean_inside_visible_ratio_mean = 0.6025588693171791
mean_interior_ratio_mean = 0.5834642090249296
mean_used_observations_mean = 6.359917936384678
mean_visible_points_mean = 710.9449039215453
num_removed = 0
```

Matrix：

| Row | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| `O27 score-silhouette-area own` | `0.044803` | `0.129561` | `0.246050` | 低于 O25/O26 |
| `O27 score-silhouette-area on S0` | `0.019739` | `0.086878` | `0.223127` | 负结果 |
| `O27 score-silhouette-area on S1` | `0.054254` | `0.105638` | `0.259337` | S1 接近 |
| `O27 silhouette-area own` | `0.041388` | `0.116918` | `0.213008` | 更低 |
| `O27 silhouette-area on S0` | `0.018974` | `0.079954` | `0.194873` | 更低 |
| `O27 silhouette-area on S1` | `0.055862` | `0.109112` | `0.245639` | S1 AP 小升但 AP25 降 |

判断：

```text
O27 是负结果。
非 GT silhouette agreement 能算出有效观测，但用它重排会降低 S0。
因此 O25 后剩余瓶颈不是简单 score/order 问题，而是 object mask 的 scene-level recall/boundary/identity 本身不足。
```

### O28 target-support-aware rank diagnostic

注意：

```text
O28 明确是 diagnostic-only。
它使用 S0 或 S1 target support 的 overlap/area/conflict 统计做 ranking/suppression。
因此不能作为 reportable method claim，只用于定位 blocker。
```

Matrix：

| Row | AP | AP50 | AP25 | #pred | conflict |
|---|---:|---:|---:|---:|---:|
| `O28 S0-aware rank diagnostic on S0` | `0.023637` | `0.095270` | `0.267036` | `34.80` | `0.074840` |
| `O28 S0-aware rank diagnostic on S1` | `0.053300` | `0.102318` | `0.236443` | `34.80` | `0.074840` |
| `O28 S1-aware rank diagnostic on S1` | `0.061123` | `0.115058` | `0.247119` | `7.20` | `0.031706` |
| `O28 S1-aware rank diagnostic on S0` | `0.006089` | `0.024712` | `0.086213` | `7.20` | `0.031706` |

Reportable scan：

```text
num_configs=2
num_reportable_method_configs=0
num_diagnostic_only_configs=2
num_suspicious_configs=0
num_uses_gt_and_method_result=0
num_configs_missing_eval_policy=0
```

判断：

```text
即使用 target support 做 diagnostic 排序，O28 S0 AP 也只有 0.023637，
AP50 反而低于 O25/O26。

这基本排除“只差排序/去重”的解释。
当前 O25/O26 masks 本身缺 scene-level object quality；
target support 统计只能略调 AP，不能接近 P0 Stream3D on S0。
```

### Blocker 和修复尝试

#### B6：matrix output_config 命名不一致

```text
影响：
  O19/O25/O26 首次 eval 已写出，但 summarize_v9_unified_eval 找不到 cross TMP。

原因：
  复现脚本 cross_eval 使用 `${output_config}_on_s0_probe5`。
  当 output_config 自身已经以 `_probe5` 结尾时，真实输出会是 `..._probe5_on_s0_probe5`。
  初始 matrix 写成了 `..._on_s0_probe5`。

修复：
  修正 O19/O25/O26 matrix JSON output_config 名称。
  只重跑 matrix/reportable scan/metric integrity，不重跑已经成功的 official eval。

结果：
  O19/O25/O26 matrix/reportable scan/metric integrity 全部完成。
```

#### B7：O28 `scannet` TMP alias 不能被 rank 工具直接读取

```text
影响：
  O28 首次运行失败：
    FileNotFoundError: data/TMP/scannet/scene0050_00_pre_points.npy

原因：
  `scannet` 是 evaluator/matrix 层的特殊别名；
  data/TMP/scannet 只有 config_manifest.json，没有 per-scene pre_points。

修复：
  support_aware_object_rank 的 S0 scoring support 改用 materialized config：
    stream4d_v9_p0_on_s0_scannet_probe5
  cross-eval 仍用 scannet S0 protocol。

结果：
  O28 rerun 成功，reportable scan 标记为 diagnostic-only。
```

#### B8：`C_core_fringe_reject` 当前不是可依赖的 v8 exporter 修复路线

```text
观察：
  检查 export_v8_surfel_object_field.py 后发现，
  `C_core_fringe_reject` 当前主要是 parser option/method 名称；
  真正 core/fringe logic 在 ScanNetExporter.export_rgbd_eval 路径里，
  并没有作为 v8 object_dict exporter 的完整 core/fringe/reject 逻辑接入。

影响：
  不能把继续调 `C_core_fringe_reject` 当成已实现的 cross-support 修复。

处理：
  本轮转向 full-span windows、overlap merge、boundary refine、silhouette score、support-aware diagnostic。
```

### 分析

```text
1. full-span coverage 是必要但不充分。
   O14/O20/O22 把 mask frames 和 exported points 大幅增加，
   S0 从 O13 的接近零提升到 O23 的 0.017217 / 0.091101 / 0.277387。
   但 raw duplicate/conflict 随 window density 爆炸，不能直接使用。

2. overlap merge 是有效局部修复，但不能解决 scene-level object identity。
   O15/O16/O18 能把 conflict 降下来，并让 S1/S0 有正信号。
   最好的 merge-only S0 仍只有约 0.005-0.007 AP。

3. score calibration 有边际收益。
   O19 large-first 在 O18 上把 S0 提到 0.007324 / 0.030070 / 0.247198。
   O26 在 O25 后只从 0.021388/0.100300 提到 0.021517/0.100351，
   说明边界修复后排序空间很小。

4. window density 的收益主要来自更多 scene coverage，但有明显瓶颈。
   ws100 -> ws50 -> ws25 提升 S0 AP50，
   但 O22 raw #pred=1156.2、conflict=0.7882，说明没有 object memory 的 dense window aggregation 会制造大量重复。

5. mask-aware query densification 是负/弱结果。
   O24 query 更多，coverage 略增，但 S0 不超过 O23。
   问题不只是 query 点数量。

6. self-discovered boundary refine 是本轮最有效的 reportable method 修复。
   O25 inside050 使用非 GT RGB-D/mask observation 裁剪后，
   S0 达到 0.021388 / 0.100300 / 0.292631。
   这是 O14-O28 中最清楚的正向修复。

7. 过严 boundary refinement 会伤 recall。
   O25 inside070+interior010 conflict 更低，但 S0 降到 0.015151 / 0.081885 / 0.277673。
   当前需要保留部分 fringe，而不是只追求 interior purity。

8. silhouette score 和 target-support-aware diagnostic 都说明“排序不是主因”。
   O27 silhouette 重排降低 S0；
   O28 即使用 S0 support 统计排序也只有 0.023637 / 0.095270 / 0.267036。
   这说明 masks 本身的 object boundary/recall/identity 不足，不是简单 ranking bug。
```

### 结论

```text
success=False for cross-support top priority
success=True for identifying the best current repair direction
success=True for reportable method audit on O25/O26/O27
success=True for diagnostic-only audit on O28

当前可以报告：
  O25/O26 将 S0 cross-support 从 O13 时代的接近零提升到约 0.0215 AP / 0.100 AP50 / 0.292 AP25。
  self-discovered boundary refine 是目前最有效的非 GT reportable 修复方向。

不能报告：
  v9 cross-support solved。
  full ScanNet final method 达标。
  Dynamic Replica 或 official tracking 成绩。
  O28 作为 method result。

下一步真正需要的是：
  scene-level persistent object memory：
    birth/update/merge/reject/lost；
    跨窗口 identity 合并，而不是事后 overlap merge；
    对每个 object 维护正观测和 negative evidence。

  或者替换/增强 proposal source：
    更可靠的 2D/3D full-scene object candidates；
    让 D4RT tracks 只做关联与边界校验，而不是从 dense noisy windows 后处理出 object field。
```

### Insight

```text
这轮最重要的 insight 是：

1. cross-support 不是 protocol 假象。
   O14-O28 都使用 unified matrix 和 manifest audit，
   P0 Stream3D on S0 reference 稳定为 0.235730 / 0.414306 / 0.537786。
   新方法在 S0 上仍远低于它。

2. support completion 已经被排除，target-support-aware ranking 也基本被排除。
   O13 all 填满 S0 会崩；
   O28 用 S0 support 排序也只能到 0.023637 AP。
   所以“把点铺满”或“换个排序”不是答案。

3. boundary negative evidence 是有效信号。
   O25 是本轮最大正向增量，说明 RGB-D + 2D mask observation 能帮助 cut 掉外溢区域。
   但它仍不能自动产生缺失 object，也不能可靠合并跨窗口 identity。

4. 当前系统缺的不是更多局部 proposal，而是 object memory。
   ws25 已经有 203.6 个 mask frames/scene、1156.2 raw objects/scene，
   问题反而变成 duplicate/conflict 爆炸。
   没有持续 object identity 的 window aggregation 只能制造更大的后处理负担。
```

### O14-O28 证据链索引

Code:

```text
Stream3D/tools/merge_overlapping_prediction_masks.py
Stream3D/tools/object_competition_rank.py
Stream3D/tools/rescore_prediction_scores.py
Stream3D/tools/self_discovered_boundary_refine.py
Stream3D/tools/self_discovered_silhouette_score.py
Stream3D/tools/support_aware_object_rank.py
Stream3D/tools/summarize_v9_unified_eval.py
Stream3D/tools/evaluate_cross_prepoints.py
Stream3D/tools/scan_reportable_configs.py
Stream3D/tools/verify_stream4d_metric_integrity.py
```

Scripts:

```text
Stream3D/scripts/reproduce_v9_o14_fullspan_grid8_cross_support.sh
Stream3D/scripts/reproduce_v9_o15_o14_overlap_merge_cross_support.sh
Stream3D/scripts/reproduce_v9_o16_o14_overlap_merge_threshold_cross_support.sh
Stream3D/scripts/reproduce_v9_o17_o14_competition_rescue_cross_support.sh
Stream3D/scripts/reproduce_v9_o18_o14_merge_fine_threshold_cross_support.sh
Stream3D/scripts/reproduce_v9_o19_score_calibration_cross_support.sh
Stream3D/scripts/reproduce_v9_o20_fullspan_grid8_ws50_cross_support.sh
Stream3D/scripts/reproduce_v9_o21_o20_merge_threshold_cross_support.sh
Stream3D/scripts/reproduce_v9_o22_fullspan_grid8_ws25_cross_support.sh
Stream3D/scripts/reproduce_v9_o23_o22_merge_threshold_cross_support.sh
Stream3D/scripts/reproduce_v9_o24_maskaware_ws50_cross_support.sh
Stream3D/scripts/reproduce_v9_o25_boundary_refine_cross_support.sh
Stream3D/scripts/reproduce_v9_o26_boundary_refine_rescore_cross_support.sh
Stream3D/scripts/reproduce_v9_o27_silhouette_score_cross_support.sh
Stream3D/scripts/reproduce_v9_o28_support_aware_rank_cross_support.sh
Stream3D/scripts/v9_o14_fullspan_grid8_matrix_probe5.json
Stream3D/scripts/v9_o15_o14_overlap_merge_matrix_probe5.json
Stream3D/scripts/v9_o16_o14_overlap_merge_threshold_matrix_probe5.json
Stream3D/scripts/v9_o17_o14_competition_rescue_matrix_probe5.json
Stream3D/scripts/v9_o18_o14_merge_fine_threshold_matrix_probe5.json
Stream3D/scripts/v9_o19_score_calibration_matrix_probe5.json
Stream3D/scripts/v9_o20_fullspan_grid8_ws50_matrix_probe5.json
Stream3D/scripts/v9_o21_o20_merge_threshold_matrix_probe5.json
Stream3D/scripts/v9_o22_fullspan_grid8_ws25_matrix_probe5.json
Stream3D/scripts/v9_o23_o22_merge_threshold_matrix_probe5.json
Stream3D/scripts/v9_o24_maskaware_ws50_matrix_probe5.json
Stream3D/scripts/v9_o25_boundary_refine_matrix_probe5.json
Stream3D/scripts/v9_o26_boundary_refine_rescore_matrix_probe5.json
Stream3D/scripts/v9_o27_silhouette_score_matrix_probe5.json
Stream3D/scripts/v9_o28_support_aware_rank_matrix_probe5.json
```

Outputs:

```text
Stream3D/outputs/audit/v9_o14_fullspan_grid8/
Stream3D/outputs/audit/v9_o15_o14_overlap_merge/
Stream3D/outputs/audit/v9_o16_o14_merge_threshold/
Stream3D/outputs/audit/v9_o17_o14_competition_rescue/
Stream3D/outputs/audit/v9_o18_o14_merge_fine_threshold/
Stream3D/outputs/audit/v9_o19_score_calibration/
Stream3D/outputs/audit/v9_o20_fullspan_grid8_ws50/
Stream3D/outputs/audit/v9_o21_o20_merge_threshold/
Stream3D/outputs/audit/v9_o22_fullspan_grid8_ws25/
Stream3D/outputs/audit/v9_o23_o22_merge_threshold/
Stream3D/outputs/audit/v9_o24_maskaware_ws50/
Stream3D/outputs/audit/v9_o25_boundary_refine/
Stream3D/outputs/audit/v9_o26_boundary_refine_rescore/
Stream3D/outputs/audit/v9_o27_silhouette_score/
Stream3D/outputs/audit/v9_o28_support_aware_rank/
Stream3D/outputs/v9_mask_merge/
Stream3D/outputs/v9_object_competition/
Stream3D/outputs/v9_score_calibration/
Stream3D/outputs/v9_boundary_refine/
Stream3D/outputs/v9_silhouette_score/
Stream3D/outputs/v9_support_aware_rank/
Stream3D/data/evaluation/scannet/stream4d_v9_o14*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o15*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o16*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o17*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o18*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o19*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o20*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o21*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o22*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o23*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o24*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o25*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o26*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o27*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o28*_class_agnostic.txt
```

Logs:

```text
Stream3D/logs/stream4d_v9_o14_*.log
Stream3D/logs/stream4d_v9_o15_*.log
Stream3D/logs/stream4d_v9_o16_*.log
Stream3D/logs/stream4d_v9_o17_*.log
Stream3D/logs/stream4d_v9_o18_*.log
Stream3D/logs/stream4d_v9_o19_*.log
Stream3D/logs/stream4d_v9_o20_*.log
Stream3D/logs/stream4d_v9_o21_*.log
Stream3D/logs/stream4d_v9_o22_*.log
Stream3D/logs/stream4d_v9_o23_*.log
Stream3D/logs/stream4d_v9_o24_*.log
Stream3D/logs/stream4d_v9_o25_*.log
Stream3D/logs/stream4d_v9_o26_*.log
Stream3D/logs/stream4d_v9_o27_*.log
Stream3D/logs/stream4d_v9_o28_*.log
Stream3D/logs/stream4d_v9_o28_final_py_compile.log
Stream3D/logs/stream4d_v9_o28_final_bash_n.log
Stream3D/logs/stream4d_v9_o28_final_unit_tests.log
```

### O14-O28 最终验证

```text
py_compile: pass
bash -n: pass
unit tests: Ran 13 tests in 0.119s, OK
```

### O14-O28 final 审计包

```text
latest final cross-support packet:
  stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.zip

latest final cross-support sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

latest final cross-support filelist:
  stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final_filelist.txt

latest final cross-support zip test:
  see sibling file stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final_ziptest.log

latest final cross-support file count:
  1331

previous final cross-support packet:
  stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.zip
```

## O29-O38 cross-support 收尾复盘

本节是用户要求“cross-support top priority”后的继续推进记录。只记录真实完成的结果；O39 因用户要求收尾而终止在导出阶段，不进入有效结果表。

### 当前状态

```text
success=False for cross-support top priority。
best reportable method-on-S0 = O37 O35 new-points logarea on S0。
O37 S0 AP/AP50/AP25 = 0.03290778791710753 / 0.12668964832396762 / 0.41826598588387504。
P0 Stream3D on S0 reference = 0.235730 / 0.414306 / 0.537786。
O37 仍远低于 P0，不能 claim cross-support solved，也不能 claim robust Stream3D replacement。
```

### 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/tools/scene_object_memory_from_predictions.py` | 新增 scene-level object memory：birth/update/reject、ambiguous bridge rejection、`union/keep_slot/new_points_only` update modes、`none/score/small_area/large_area` exclusivity | 按 O28 结论转向 object memory，避免继续只做 support completion |
| `Stream3D/scripts/reproduce_v9_o29_scene_memory_cross_support.sh` | O29/O30 scene memory 复现脚本 | 固化 scene memory + refine 的 own/S0/S1/matrix/integrity 流程 |
| `Stream3D/scripts/reproduce_v9_o31_memory_exclusive_cross_support.sh` | O31 exclusivity ablation | 验证 zero-conflict 是否真有利于 S0 |
| `Stream3D/scripts/reproduce_v9_o32_o31_score_calibration_cross_support.sh` | O32 score calibration | 测试 logarea / inverse-logarea 是否是主要瓶颈 |
| `Stream3D/scripts/reproduce_v9_o33_o31_boundary_refine_cross_support.sh` | O33 O31 boundary refine | 测试非 GT RGB-D/mask observation 对 memory masks 的裁剪收益 |
| `Stream3D/scripts/reproduce_v9_o34_o33_score_calibration_cross_support.sh` | O34 O33 score calibration | 将 O33 inside035 接 logarea |
| `Stream3D/scripts/reproduce_v9_o35_memory_update_mode_cross_support.sh` | O35 memory update-mode ablation | 针对过度合并，比较 `keep_slot` 与 `new_points_only` |
| `Stream3D/scripts/reproduce_v9_o36_o35_boundary_refine_cross_support.sh` | O36 O35 boundary refine | 测试 O35 best 的 boundary refine tradeoff |
| `Stream3D/scripts/reproduce_v9_o37_o35_o36_score_calibration_cross_support.sh` | O37 O35/O36 score calibration | 测试 score calibration 能否保留 O35 recall 并抬 AP/AP50 |
| `Stream3D/scripts/reproduce_v9_o38_memory_threshold_logarea_cross_support.sh` | O38 memory threshold sweep | 比较偏合并 c055 与偏分裂 c075split |
| `Stream3D/scripts/reproduce_v9_o39_c075split_exclusive_cross_support.sh` | O39 small-area exclusivity attempt | 尝试验证 split 后 object competition；未完成，不进入有效结果 |
| `docs/stream4d_v9_执行日志.md`、`docs/stream4d_v9_实验结果复盘.md` | 追加 O29-O38/O39 aborted 记录 | 满足复现和审计要求 |

### 主要结果表

| Run | AP | AP50 | AP25 | pre% | #pred | conflict | 判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| `P0 Stream3D on S0` | `0.235730` | `0.414306` | `0.537786` | `0.846744` | `128.20` | `0.2213` | reference |
| `O26 inside050 logarea on S0` | `0.021517` | `0.100351` | `0.291694` | `0.846744` | `80.60` | `0.113479` | O28 前 best AP/AP50 |
| `O31 c065 no-exclusive on S0` | `0.023980` | `0.102236` | `0.423419` | `0.846744` | `54.80` | `0.177452` | memory 正向，AP25 大升 |
| `O32 no-exclusive logarea on S0` | `0.023991` | `0.102364` | `0.424218` | `0.846744` | `54.80` | `0.177452` | AP25 best in O31/O32 |
| `O33 inside035 on S0` | `0.025754` | `0.111061` | `0.412295` | `0.846744` | `54.80` | `0.130407` | boundary refine 提升 AP/AP50 |
| `O34 inside035 logarea on S0` | `0.026925` | `0.117447` | `0.413656` | `0.846744` | `54.80` | `0.130407` | O33 后 score 小幅正向 |
| `O35 keep-slot on S0` | `0.002759` | `0.015338` | `0.294378` | `0.846744` | `107.20` | `0.090768` | 负结果，birth-only recall 不足 |
| `O35 new-points-only on S0` | `0.031794` | `0.122303` | `0.413561` | `0.846744` | `57.80` | `0.057197` | 核心正向修复 |
| `O36 new-points inside050 on S0` | `0.029340` | `0.113117` | `0.398909` | `0.846744` | `57.80` | `0.039425` | own 变强，S0 变弱 |
| `O37 new-points logarea on S0` | `0.032908` | `0.126690` | `0.418266` | `0.846744` | `57.80` | `0.057197` | 本轮 best reportable S0 |
| `O38 c055 logarea on S0` | `0.033012` | `0.123089` | `0.392066` | `0.846744` | `46.20` | `0.039025` | AP 微升但 AP50/AP25 降 |
| `O38 c075split logarea on S0` | `0.025503` | `0.106879` | `0.376944` | `0.846744` | `105.20` | `0.136208` | S1/diagnostics 好，S0 差 |

### O29-O38 分析

```text
1. scene object memory 是有效方向，但不是充分解。
   O31 no-exclusive 把 O26 S0 从 0.021517 / 0.100351 / 0.291694
   提到 0.023980 / 0.102236 / 0.423419。
   最大收益来自 AP25/recall，而不是高 IoU AP50。

2. zero-conflict 不是目标。
   O29 score-exclusive 和 O31 small-area-exclusive 都比 O31 no-exclusive 弱。
   O35 new_points_only 保留少量 overlap/conflict 后 S0 反而最佳。

3. `new_points_only` 是本轮最关键正向修复。
   它防止 update 阶段把重复 proposal 完整吞进已有大 slot，
   同时又允许新增 points 补充 recall。
   O35 new_points_only on S0 达到 0.031794 / 0.122303 / 0.413561。

4. boundary refine 有明显 own-support 收益，但会伤 S0。
   O36 inside050 own = 0.112869 / 0.273510 / 0.525436，
   但 on S0 降到 0.029340 / 0.113117 / 0.398909。
   说明 S0 仍依赖一些 fringe/low-IoU recall，不能只追求干净边界。

5. logarea score calibration 有边际正向。
   O37 在 O35 new_points_only 上把 S0 提到 0.032908 / 0.126690 / 0.418266。
   但提升幅度仍很小，排序不是主要瓶颈。

6. 阈值 sweep 暴露 split/merge tradeoff。
   O38 c055 合并更强，AP 微升到 0.033012，但 AP50/AP25 降；
   O38 c075split 的 S0 best-IoU mean = 0.4372，GT IoU>=0.50 count = 87，
   但 #pred=105.20、conflict=0.136208，official AP 下降。
   更好 best-IoU diagnostics 没有自动转成 evaluator AP。
```

### Blocker / 修复尝试

#### B9：scene memory score-exclusive 过硬

```text
问题：
  O29 score-exclusive 把 conflict 降到 0，但 S0 AP/AP50 低于 O26/O31。

修复：
  O31 添加 no-exclusive 和 small-area-exclusive 对比。

结果：
  no-exclusive on S0 = 0.023980 / 0.102236 / 0.423419；
  small-area-exclusive on S0 = 0.023499 / 0.096988 / 0.373138。

判断：
  强制唯一 ownership 会删除对 S0 有用的 fringe/recall。
```

#### B10：memory union 过度合并

```text
问题：
  O31/O33 large mask ratio 高，O33 inside035 on S0 large>1000 = 55.0160%。

修复：
  O35 增加 update-mode ablation：
    keep_slot
    new_points_only

结果：
  keep_slot on S0 = 0.002759 / 0.015338 / 0.294378；
  new_points_only on S0 = 0.031794 / 0.122303 / 0.413561。

判断：
  keep_slot 丢 recall；
  new_points_only 是当前最合理折中。
```

#### B11：boundary refine 提高 own-support 但伤 cross-support

```text
问题：
  O35 new_points_only 仍有 boundary/large-mask 问题。

修复：
  O36 对 O35 new_points_only 做 self-discovered boundary refine。

结果：
  O36 inside050 own = 0.112869 / 0.273510 / 0.525436；
  O36 inside050 on S0 = 0.029340 / 0.113117 / 0.398909。

判断：
  该 refine 能提升 cropped own-support AP/AP50，
  但对 S0 的 recall/fringe 不利，不能作为 cross-support best。
```

#### B12：split 后 conflict/排序未解决

```text
问题：
  O38 c075split best-IoU diagnostics 较好，但 official S0 AP 下降。

修复：
  计划 O39 c075split + small-area exclusivity + logarea。

结果：
  O39 因用户要求收尾终止在 export 阶段，未产生 summary/eval/matrix。

判断：
  O39 不进入有效结果；
  但 O38 已足够说明 split 本身不能直接解决 cross-support。
```

### 结论

```text
success=False for cross-support top priority
success=True for reportable method audit through O38
success=True for identifying one useful local fix: memory update_mode=new_points_only
success=False for robust/full Stream3D replacement

当前 best:
  O37 O35 new-points logarea on S0 =
    0.03290778791710753 / 0.12668964832396762 / 0.41826598588387504

相对 O26:
  AP  +0.0113908
  AP50 +0.0263386
  AP25 +0.126572

相对 P0 Stream3D on S0:
  AP  -0.202822
  AP50 -0.287617
  AP25 -0.119520

所以，本轮没有达成 cross-support 目标。
可以报告的是：scene object memory + new_points_only + logarea 显著优于 O28 前 best，
但距离 Stream3D S0 reference 仍很远，不能 claim solved。
```

### 新计划建议

```text
1. 停止继续在 O35/O37 上做小阈值 sweep。
   O38 表明阈值能改 best-IoU/conflict，但不能把 AP50 拉近 P0。

2. 下一轮要先做 failure autopsy，而不是继续加变体：
   对 O37 on S0 按 GT instance 分桶：
     missed GT
     best IoU 0.25-0.50
     best IoU >=0.50 但 AP 不吃分
     duplicate / ranking failure
   这一步只做诊断，标 diagnostic-only。

3. 如果主要是 ranking failure：
   再做非 GT confidence model，输入只能来自 mask observation consistency、slot lifetime、area、conflict、boundary statistics。

4. 如果主要是 boundary / split failure：
   实现真正 per-slot cannot-link / negative evidence，
   而不是 post-hoc global exclusivity。

5. 如果主要是 missing GT recall：
   需要新的 proposal source 或更高 mask frequency；
   不能继续靠当前 ws25 proposal 后处理硬补。

6. 动态/4D claim 暂停。
   先把 ScanNet S0 cross-support 从 0.033 AP / 0.127 AP50 提到可用水平，
   再谈 Dynamic Replica 或 official tracking。
```

### Insight

```text
本轮最重要的 insight 是：
  cross-support 不是“support 多一点”或“conflict 低一点”能解决的问题。

O35 new_points_only 有效，是因为它改变了 memory update 的语义：
  candidate 可以贡献新区域，但不能把整个 noisy proposal 反复并进 slot。

O36/O38 的负结果也很有价值：
  过强 boundary refine 会切掉 S0 需要的 fringe；
  过强 split 会提高 best-IoU diagnostics，却因为 conflict/ranking/duplicates 损失 official AP。

因此下一轮应该从 per-GT AP failure autopsy 开始，
把 error 拆成 recall、boundary、duplicate、ranking 四类，
而不是继续凭单个总 AP 盲扫阈值。
```

### O29-O38 证据链索引

Code:

```text
Stream3D/tools/scene_object_memory_from_predictions.py
Stream3D/tools/rescore_prediction_scores.py
Stream3D/tools/self_discovered_boundary_refine.py
Stream3D/tools/summarize_v9_unified_eval.py
Stream3D/tools/evaluate_cross_prepoints.py
Stream3D/tools/scan_reportable_configs.py
Stream3D/tools/verify_stream4d_metric_integrity.py
```

Scripts:

```text
Stream3D/scripts/reproduce_v9_o29_scene_memory_cross_support.sh
Stream3D/scripts/reproduce_v9_o31_memory_exclusive_cross_support.sh
Stream3D/scripts/reproduce_v9_o32_o31_score_calibration_cross_support.sh
Stream3D/scripts/reproduce_v9_o33_o31_boundary_refine_cross_support.sh
Stream3D/scripts/reproduce_v9_o34_o33_score_calibration_cross_support.sh
Stream3D/scripts/reproduce_v9_o35_memory_update_mode_cross_support.sh
Stream3D/scripts/reproduce_v9_o36_o35_boundary_refine_cross_support.sh
Stream3D/scripts/reproduce_v9_o37_o35_o36_score_calibration_cross_support.sh
Stream3D/scripts/reproduce_v9_o38_memory_threshold_logarea_cross_support.sh
Stream3D/scripts/reproduce_v9_o39_c075split_exclusive_cross_support.sh
Stream3D/scripts/v9_o29_scene_memory_matrix_probe5.json
Stream3D/scripts/v9_o31_memory_exclusive_matrix_probe5.json
Stream3D/scripts/v9_o32_o31_score_calibration_matrix_probe5.json
Stream3D/scripts/v9_o33_o31_boundary_refine_matrix_probe5.json
Stream3D/scripts/v9_o34_o33_score_calibration_matrix_probe5.json
Stream3D/scripts/v9_o35_memory_update_mode_matrix_probe5.json
Stream3D/scripts/v9_o36_o35_boundary_refine_matrix_probe5.json
Stream3D/scripts/v9_o37_o35_o36_score_calibration_matrix_probe5.json
Stream3D/scripts/v9_o38_memory_threshold_logarea_matrix_probe5.json
Stream3D/scripts/v9_o39_c075split_exclusive_matrix_probe5.json
```

Audit outputs:

```text
Stream3D/outputs/audit/v9_o29_scene_memory/
Stream3D/outputs/audit/v9_o31_memory_exclusive/
Stream3D/outputs/audit/v9_o32_o31_score_calibration/
Stream3D/outputs/audit/v9_o33_o31_boundary_refine/
Stream3D/outputs/audit/v9_o34_o33_score_calibration/
Stream3D/outputs/audit/v9_o35_memory_update_mode/
Stream3D/outputs/audit/v9_o36_o35_boundary_refine/
Stream3D/outputs/audit/v9_o37_o35_o36_score_calibration/
Stream3D/outputs/audit/v9_o38_memory_threshold_logarea/
```

Prediction/evaluation outputs:

```text
Stream3D/data/evaluation/scannet/stream4d_v9_o29*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o30*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o31*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o32*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o33*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o34*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o35*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o36*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o37*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o38*_class_agnostic.txt
Stream3D/data/prediction/stream4d_v9_o29*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o30*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o31*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o32*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o33*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o34*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o35*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o36*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o37*_class_agnostic/
Stream3D/data/prediction/stream4d_v9_o38*_class_agnostic/
Stream3D/data/TMP/stream4d_v9_o29*/
Stream3D/data/TMP/stream4d_v9_o30*/
Stream3D/data/TMP/stream4d_v9_o31*/
Stream3D/data/TMP/stream4d_v9_o32*/
Stream3D/data/TMP/stream4d_v9_o33*/
Stream3D/data/TMP/stream4d_v9_o34*/
Stream3D/data/TMP/stream4d_v9_o35*/
Stream3D/data/TMP/stream4d_v9_o36*/
Stream3D/data/TMP/stream4d_v9_o37*/
Stream3D/data/TMP/stream4d_v9_o38*/
```

Logs:

```text
Stream3D/logs/stream4d_v9_o29_*.log
Stream3D/logs/stream4d_v9_o30_*.log
Stream3D/logs/stream4d_v9_o31_*.log
Stream3D/logs/stream4d_v9_o32_*.log
Stream3D/logs/stream4d_v9_o33_*.log
Stream3D/logs/stream4d_v9_o34_*.log
Stream3D/logs/stream4d_v9_o35_*.log
Stream3D/logs/stream4d_v9_o36_*.log
Stream3D/logs/stream4d_v9_o37_*.log
Stream3D/logs/stream4d_v9_o38_*.log
Stream3D/logs/stream4d_v9_o39_*.log
Stream3D/logs/stream4d_v9_o38_final_py_compile.log
Stream3D/logs/stream4d_v9_o38_final_bash_n.log
Stream3D/logs/stream4d_v9_o38_final_unit_tests.log
```

Final validation:

```text
py_compile: pass
bash -n: pass
unit tests: Ran 13 tests in 0.124s, OK
```

### O29-O38 audit packet

```text
latest final cross-support packet:
  stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final.zip

latest final cross-support sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

latest final cross-support filelist:
  stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final_filelist.txt

latest final cross-support zip test:
  see sibling file stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final_ziptest.log

latest final cross-support file count:
  371

previous final cross-support packet:
  stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.zip
```
