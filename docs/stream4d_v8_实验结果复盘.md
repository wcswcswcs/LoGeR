# Stream4D v8 D4RT-native surfel field 实验结果复盘

日期：2026-06-09（Asia/Singapore）  
执行日志：`docs/stream4d_v8_执行日志.md`  
计划文件：`docs/stream4d_v8_d4rt_native_surfel_experiment_plan_for_codex.md`

本复盘只记录真实执行得到的数据。没有跑出的指标不补写；blocker 和修复尝试必须单独记录。

## 当前状态

```text
正确 D4RT/Stream4D Python 环境已确认：/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python。
曾误用 4D env 跑出 G1 diagnostic；该输出不进入有效结果表。
Lane 0 py_compile/import smoke/unit test pass。
Lane 1 stride-10 G1 失败，已定位为 D4RT clip temporal-scale/preprocessing 问题。
Lane 1 连续 16 帧 G1 grid32 margin 0.02 已扩到 probe5。
连续帧 G1 通过 uv/cycle/track-length gate，但 Sim3 metric geometry 仍弱。
Lane 2 mask measurement coverage diagnostic 完成，uses_gt=False。
Lane 3 A/B/C lightweight surfel object field prototypes 已完成 probe5 method eval。
当前 best = B1 surfacelet_singlemask，AP/AP50/AP25 = 0.32843947812986807 / 0.6292662056580957 / 0.8843628978668244。
B1 通过 reportable scan 和 metric integrity；uses_gt=False，phase0_pass=True。
但 B1 仍是 16f/probe5/sparse mask-frame prototype，不能 claim full ScanNet 或 Dynamic Replica 官方结果。
Dynamic Replica v8 loger 环境检查完成，但 usable_scene_count=0，不能报告官方动态指标。
```

## 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/tools/export_d4rt_grid_surfel_field_v8.py` | 新增 G1 grid/semi-dense surfel field diagnostic，生成 carrier-like surfel NPZ、coverage/self/cycle summary、overlay | v8 Lane 1 必需；验证 D4RT native semi-dense query 是否比 sparse carrier 更值得继续 |
| `Stream3D/tools/export_d4rt_grid_surfel_field_v8.py` | 新增 `--grid-margin-ratio` | 按 OpenD4RT helper 的 `margin_ratio=0.02` 排查边缘 query / normalized uv 问题 |
| `Stream3D/tools/compare_d4rt_adapter_official_v8.py` | 新增 adapter-vs-official helper 等价性诊断 | 按计划 Lane 1 blocker 修复方向，确认 D4RTAdapter 是否偏离 OpenD4RT helper |
| `Stream3D/stream4d/scannet_stream.py` | `load_window(..., require_masks=False)` 支持缺 mask 帧用 0 mask 占位；默认仍 require mask | 连续帧 D4RT geometry diagnostic 不应被 stride-10 mask cache 阻塞 |
| `Stream3D/tools/export_d4rt_grid_surfel_field_v8.py` | 新增 `--allow-missing-masks` | 允许 Lane 1 纯几何/轨迹诊断使用连续 ScanNet frames |
| `Stream3D/tools/diagnose_v8_mask_measurement_coverage.py` | 新增 Lane 2 非 GT mask measurement coverage diagnostic | 量化连续 D4RT clip 中可用 2D mask observation 覆盖，不读 GT、不产 AP |
| `Stream3D/tools/export_v8_surfel_object_field.py` | 新增 Lane3 surfel object field prototype exporter；支持 A signed mask-history、B single-mask surfacelet、C core/reject 方向 | 计划 Lane3 要并行比较非 connected-component 的 surfel partition 原型，并产出 method AP |
| `Stream3D/scripts/reproduce_v8_lane1.sh` | 新增 scene0050_00 16f grid16 复现脚本 | 后续复现 G1 diagnostic 不需要重新读代码 |
| `Stream3D/scripts/reproduce_v8_lane3.sh` | 新增 B1 最小复现脚本 | 方便复现当前 best Lane3 method result 和审计 |
| `docs/stream4d_v8_执行日志.md`、`docs/stream4d_v8_实验结果复盘.md` | 新增日志 | 满足本轮执行与复盘审计要求 |

## 环境结论

```text
系统 python3 和 base conda python 均无 torch，不能运行 D4RT 模型。
loger conda 环境有 torch 2.6.0+cu124，CUDA 可用。
D4RT 32-frame checkpoint 存在。
scene0050_00 ScanNet RGB-D/mask/mesh 均存在。
本轮 D4RT/v8 Lane 1 固定使用 loger 环境；系统/base python 的无 torch结果只作为环境探测，不作为实验失败。
4D env 跑出的 G1 grid16/grid32 是错误环境 diagnostic，不作为有效 gate 依据。
```

## Lane 0 验证

```text
py_compile loger: pass
import smoke loger: v8 loger import smoke OK
unit tests loger: Ran 13 tests in 0.125s, OK
final py_compile loger: pass
reproduce script bash -n: pass
final unittest loger: Ran 13 tests in 0.121s, OK
Lane3 final py_compile: pass
Lane3 final import smoke: pass
Lane3 final unittest: Ran 13 tests in 0.129s, OK
reproduce_v8_lane3.sh bash -n: pass
reportable method configs: Lane3 A/B/C 共 5 个 method configs，scan clean。
```

## Lane 1：G1 grid/semi-dense surfel field

有效环境：

```text
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CUDA_VISIBLE_DEVICES=6/7
```

错误环境处理：

```text
曾用 /mnt/data/users/chengshun.wang/miniconda3/envs/4D/bin/python 跑过 G1 grid16/grid32。
这些输出真实存在，但环境不符合历史 Stream4D 约定。
因此不进入 v8 有效结果表，不用于 gate 判断。
```

### G1 stride-10 失败结果

这些是 loger 环境真实结果，但后来被 E10/E11 证明主要是 clip temporal scale 设置错误；保留为反例，不作为最终 Lane 1 成立/失败结论。

| Run | queries | valid tracks | visible obs | uv in01 | visibility mean | confidence mean | track len mean | track len p10 | self p90 px | cycle p90 px | coverage mean | Sim3 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `grid16_stride10_loger` | `4096` | `3979` | `34209` | `0.593902587890625` | `0.4975601136684418` | `0.9999560713768005` | `8.351806640625` | `2.0` | `6.634120941162109` | `34.77774353027344` | `0.032052040100097656` | `0.45260965177544044` |
| `grid32_stride10_loger` | `16384` | `16150` | `141300` | `0.6092605590820312` | `0.5138279795646667` | `0.9999561309814453` | `8.624267578125` | `3.0` | `6.3304579257965115` | `38.27891311645508` | `0.12543678283691406` | `0.45915942109306296` |
| `grid32m002_stride10_loger` | `16384` | `16375` | `146552` | `0.6279640197753906` | `0.5324155688285828` | `0.9999563694000244` | `8.94482421875` | `3.0` | `6.181506586074829` | `39.24992179870606` | `0.12944412231445312` | `0.4602868233077916` |

### E10 adapter-vs-official helper

| Diff | mean abs | p90 abs | max abs |
|---|---:|---:|---:|
| `uv` | `3.855219983961433e-05` | `8.618831634521484e-05` | `0.0006253048777580261` |
| `xyz_ref0` | `0.00034643031540326774` | `0.0007100462913513185` | `0.002689838409423828` |
| `visibility_prob` | `0.00015662208897992969` | `0.00039833784103393555` | `0.0204317569732666` |
| `confidence_prob` | `3.018067218363285e-08` | `1.1920928955078125e-07` | `1.5497207641601562e-06` |

判断：

```text
D4RTAdapter 与 OpenD4RT helper 在同一 clip/query 上几乎一致。
stride-10 失败不是 adapter query order/API 不一致导致。
```

### E11 scene0050 连续帧修复

| Run | uv in01 | track len mean | self p90 px | cycle p90 px | coverage mean | Sim3 median | Sim3 p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `grid32m002_stride10_loger` | `0.6279640197753906` | `8.94482421875` | `6.181506586074829` | `39.24992179870606` | `0.12944412231445312` | `0.4602868233077916` | `0.7802417144670429` |
| `grid32m002_stride1_loger` | `0.9908294677734375` | `15.771484375` | `1.3851011872291565` | `2.381803274154663` | `0.16101455688476562` | `0.2619269225708374` | `0.4944150956084864` |

结论：

```text
连续帧修复显著解决 uv/cycle blocker。
初始 stride-10 G1 不能作为 D4RT native surfel field 失败证据。
```

### E12 probe5 连续帧 G1 summary

| Metric | mean | min | max |
|---|---:|---:|---:|
| `uv_in01_rate` | `0.9858451843261719` | `0.9592514038085938` | `0.9998092651367188` |
| `track_length_visible_mean` | `13.284716796875` | `9.64019775390625` | `15.98291015625` |
| `track_length_visible_p10` | `7.4` | `1.0` | `16.0` |
| `self_uv_error_p90` | `1.57081866979599` | `1.3851011872291565` | `1.6558191418647772` |
| `cycle_uv_error_p90` | `3.2913891315460204` | `2.381803274154663` | `4.192905950546264` |
| `coverage mean` | `0.13198394775390626` | `0.1008310317993164` | `0.16101455688476562` |
| `Sim3 residual median` | `0.46820781478265117` | `0.2619269225708374` | `0.6804238543343015` |
| `Sim3 residual p90` | `0.8595804531797114` | `0.4944150956084864` | `1.1890475588672236` |

Lane 1 最新 gate：

```text
uv_in01_rate_mean >= 0.70:
  0.9858451843261719
  pass

track_length_visible_mean >= 6:
  13.284716796875
  pass

self_uv_error_p90 <= 8 px:
  1.57081866979599
  pass

cycle_uv_error_p90 <= 12 px:
  3.2913891315460204
  pass

semi-dense coverage >= current sparse union 3x:
  coverage mean = 0.13198394775390626
  v6/v7 sparse support reference about 0.04514451433782776
  3x reference about 0.13543354301348328
  near threshold, mean slightly below by about 0.00343; scene0050 passes, scene0591 low.

Sim3 residual median mean 明显低于 0.680m:
  0.46820781478265117 < 0.6801382969694636
  pass for correspondence diagnostic
  but >0.30m, so ScanNet mesh AP remains diagnostic only, not a D4RT metric-geometry claim.
```

结论：

```text
连续帧 G1 已经支持继续做 image-space / observation-space surfel field diagnostic。
但它还不能支撑“D4RT metric geometry 替代 ScanNet RGB-D mesh”的 claim。
Lane 3 若启动，必须避开把 coarse D4RT xyz 直接当高质量 ScanNet mesh geometry 主体。
```

## Blocker 修复尝试

### B1：环境误用修复

```text
问题：
  最初误用 4D env 跑 G1；用户指出正确环境是 loger。

修复：
  将复现脚本和后续命令固定为 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python。
  4D 输出标记为错误环境 diagnostic，不进入有效结果表。

结果：
  loger 下 env check、py_compile、import smoke、G1 grid16/grid32/grid32m002、Sim3 diagnostic 均完成。
```

### B2：grid margin / normalized uv 修复

```text
问题：
  uv_in01_rate 低，可能来自边缘 query 或和 OpenD4RT 官方 grid 采样不一致。

修复：
  对照 OpenD4RT/infer_track_3d.py 的 `_grid_query_points(..., margin_ratio=0.02)`，
  新增 --grid-margin-ratio 并重跑 grid32 margin 0.02。

结果：
  uv_in01_rate: 0.6092605590820312 -> 0.6279640197753906
  track_length_visible_mean: 8.624267578125 -> 8.94482421875
  self_uv_error_p90: 6.3304579257965115 -> 6.181506586074829
  cycle_uv_error_p90: 38.27891311645508 -> 39.24992179870606
  Sim3 residual median: 0.45915942109306296 -> 0.4602868233077916

判断：
  margin 有小幅正向，但在 stride-10 clip 下没有解决 uv/cycle/geometry blocker。
```

### B3：official infer sanity 入口检查

```text
命令：
  python ../Open-d4rt/infer_track_3d.py --help

结果：
  logs/stream4d_v8_official_infer_track_help_loger.log size = 0 bytes。
  `infer_track_3d.py` 没有 argparse/CLI main；它是 helper module。

判断：
  不能直接用 --help/CLI 做 adapter-vs-official sanity，因此新增 helper API 对比脚本。
```

### B4：adapter-vs-official helper 对比

```text
问题：
  stride-10 G1 的 uv/cycle 失败可能来自 D4RTAdapter query order/API 偏差。

修复/诊断：
  新增 tools.compare_d4rt_adapter_official_v8，
  对同一 scene0050 clip 和同一 grid4 query 同时跑 D4RTAdapter 与 OpenD4RT helper。

结果：
  uv mean abs = 3.855219983961433e-05
  xyz_ref0 mean abs = 0.00034643031540326774
  visibility_prob mean abs = 0.00015662208897992969
  confidence_prob mean abs = 3.018067218363285e-08

判断：
  adapter 与 official helper 基本等价。
  stride-10 失败不是 adapter API 偏差导致。
```

### B5：连续帧 / mask 缺失修复

```text
问题：
  使用 frame_stride=10 会把 ScanNet 0..150 当成 16-frame D4RT clip，破坏 temporal scale。
  改成 frame_stride=1 后，Cropformer mask/1.png 等不存在。

修复：
  ScanNetStream.load_window 新增 require_masks=False。
  export_d4rt_grid_surfel_field_v8 新增 --allow-missing-masks。
  缺 mask 只用于 Lane 1 source metadata 的 0-mask 占位；默认路径仍 require mask。

结果：
  scene0050 cycle_uv_error_p90: 39.24992179870606 -> 2.381803274154663
  scene0050 uv_in01_rate: 0.6279640197753906 -> 0.9908294677734375
  scene0050 Sim3 residual median: 0.4602868233077916 -> 0.2619269225708374

判断：
  frame_stride=10 是主要 uv/cycle blocker。
  连续帧 G1 能成立为 D4RT correspondence diagnostic。
```

## Lane 2：mask measurement coverage

```text
diagnostic_only=True
uses_gt=False
is_method_result=False
num_ok_windows=5
num_mask_frames_available_mean=2.0
num_mask_frames_missing_mean=14.0
carrier_assignment_rate_all_frames_mean=0.12105091419117497
carrier_assignment_rate_available_mask_frames_mean=0.9806140838500657
surfel_positive_observation_rate_mean=0.91259765625
mean_positive_observations_per_surfel_mean=1.72508544921875
```

解释：

```text
连续 16 帧 clip 中只有 frame 0 和 10 有 Cropformer masks。
在实际有 mask 的帧上，visible surfels 几乎都能落入非空 mask；
但从 16-frame field 角度看，mask observation 时间覆盖很稀疏。
Lane 3 不能假装有 dense per-frame mask measurement。
```

## Lane 3：surfel object field prototypes

方法共同约束：

```text
输入 = Lane1 连续 16-frame G1 grid32 margin0.02 D4RT surfel tracks。
semantic measurement = Cropformer mask frames available in each 16f clip, currently frame 0 and frame 10。
uses_gt=False。
is_method_result=True。
export = RGB-D mask backproject 到 ScanNet mesh evaluator，pre_points_policy=recompute。
同一 (frame_id, mask_id) 在 support expansion 前只允许一个 cluster/object owner。
```

新增工具：

```text
Stream3D/tools/export_v8_surfel_object_field.py
```

Prototype 定义：

| Run | Direction | 定义 |
|---|---|---|
| `A1` | signed mask-history | surfel 需要至少 2 个 mask observations；按 exact mask history 成组；每个 object 最多 2 个 owned masks |
| `A2` | signed mask-history recall | surfel 只需 1 个 mask observation；测试单帧召回是否能弥补 mask 稀疏 |
| `B1` | surfacelet single-mask | `max_observations=1`，把单帧 mask-owned surfacelet proposal 作为 object field 原型 |
| `C1` | core/fringe/reject | surfel 需 2 个 observations，且 object 必须拥有 2 个 masks，否则 reject |
| `A1s1` | export resolution check | 与 A1 相同，但 mask backproject stride 从 2 改为 1 |

Probe5 AP：

| Run | AP | AP50 | AP25 | objects/scene | pre_points/scene | conflict | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `A1 signed_history_m2` | `0.31800136425080006` | `0.5499198184432726` | `0.8688954479508236` | `15.2` | `9482.8` | `0.09091808474220602` | 接近 AP gate，AP50/AP25 pass |
| `A2 signed_history_m1` | `0.3060221611896139` | `0.5496301734726751` | `0.888005513856092` | `17.2` | `9733.8` | `0.11135387397243113` | 单帧召回提高 AP25，但 AP 降低 |
| `B1 surfacelet_singlemask` | `0.32843947812986807` | `0.6292662056580957` | `0.8843628978668244` | `16.4` | `8513.2` | `0.08430650606572185` | 当前 best，pass probe5 gate |
| `C1 core_owned2` | `0.31583318558955015` | `0.5295522312291147` | `0.8468851431724259` | `13.6` | `9218.4` | `0.0820415825059075` | 过严 reject 损失 AP/AP50 |
| `A1s1 stride1 export` | `0.29073598846418347` | `0.5581574610318252` | `0.8679909541120018` | `15.2` | `10302.6` | `0.11975014288387284` | 更密回投增加冲突，AP 降低 |

B1 per-scene：

| Scene | AP | AP50 | AP25 | objects | pre_points | conflict |
|---|---:|---:|---:|---:|---:|---:|
| `scene0050_00` | `0.40079365079365076` | `0.7142857142857142` | `0.7142857142857142` | `14` | `8593` | `0.06237635284533923` |
| `scene0011_00` | `0.2904170675004008` | `0.7850108225108225` | `0.9052015692640693` | `12` | `5980` | `0.1588628762541806` |
| `scene0030_00` | `0.5142857142857142` | `0.5547619047619048` | `0.857142857142857` | `7` | `10790` | `0.026042632066728452` |
| `scene0081_01` | `0.2457010582010582` | `0.7331349206349207` | `0.9603174603174605` | `10` | `8912` | `0.1099640933572711` |
| `scene0591_00` | `0.2875921917588584` | `0.48261002886002885` | `0.972020202020202` | `39` | `8291` | `0.06428657580508985` |

B1 审计：

```text
reportable scan:
  num_configs=5
  num_configs_missing_manifest=0
  num_diagnostic_only_configs=0
  num_oracle_configs=0
  num_reportable_method_configs=5
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0

metric integrity for B1:
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  gt_files_read_by_rescore=False
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
  mean_pre_points_ratio=0.03986074960713631
  mean_prediction_union_ratio=0.03986074960713631
  pre_points_policy={"recompute_like": 5}
  object_dict_pred_alignment_mean/min=1.0/1.0
  object_dict_pred_alignment_failed_instances=0
```

与历史 baseline 对比：

```text
Best v6/v7 dense-object baseline P3:
  0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401

v7 same-support Stream3D diagnostic P0 on S1:
  0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108

v8 B1:
  0.32843947812986807 / 0.6292662056580957 / 0.8843628978668244

B1 vs P3:
  AP  +0.043607005560894
  AP50 +0.1253039415071523
  AP25 +0.21244817297708433

B1 vs P0-on-S1:
  AP  -0.07077331507192463
  AP50 +0.03209491178695906
  AP25 +0.1418275390402136
```

判断：

```text
B1 是 v8 当前第一个通过 probe5 gate 的非-GT method result：
  AP >= 0.32 pass
  AP50 >= 0.54 pass
  AP25 >= 0.70 pass

但必须限制 claim：
  B1 只验证 5-scene probe5；
  pre_points ratio 只有 0.03986，仍是 sparse support；
  semantic observations 只有每个 16f clip 中的 2 个 mask frames；
  不能 claim full ScanNet；
  不能 claim Dynamic Replica official tracking；
  不能 claim D4RT metric geometry 已替代 RGB-D mesh。
```

## Dynamic Replica

```text
data_root_exists=True
split_dir_exists=True
annotation_exists=True
scene_count=20
usable_scene_count=0
can_report_official_instance_tracking=False
can_report_d4rt_trajectory_metrics=False
can_report_only_qualitative_consistency=False
```

解释：

```text
Dynamic Replica v2 valid 目录和 annotation 存在，20 scenes 有 images 和 trajectories，
但本地检查没有 depth 对齐，且无 instance/object ID GT。
因此不能报告 IDF1、MOTA、4D IoU、official instance tracking 或 D4RT trajectory metrics。
```

## 分析

```text
1. 初始 stride-10 G1 失败不是 D4RTAdapter API 偏差。
   adapter-vs-official helper diff 很小：uv mean abs 3.86e-05，xyz mean abs 3.46e-04。

2. 真正关键修复是连续帧。
   scene0050 stride1 把 cycle p90 从 39.25px 降到 2.38px，
   uv_in01 从 0.628 提到 0.991，Sim3 median 从 0.460m 降到 0.262m。

3. probe5 连续帧 G1 通过 correspondence 相关最低 gate。
   uv mean 0.9858，track length mean 13.28，self p90 1.57px，cycle p90 3.29px。

4. 但 metric geometry 仍弱。
   Sim3 median mean 0.468m，虽然低于 v7 sparse aggregate 0.680m，
   但仍大于 0.30m，且 scene0081/scene0591 很高。

5. Lane 2 coverage 说明 2D mask observation 可用但时间稀疏。
   有 mask 帧 assignment mean 0.9806，但 16 帧中只有 2 帧有 mask。
   所以后续 object partition 必须是 sparse measurement model，而不是 dense mask stream。

6. Lane3 B1 说明 v8 的 object-field 表示有真实正信号。
   单帧 surfacelet ownership 比 exact 2-frame history 更好：
   B1 AP 0.3284，高于 A1 0.3180，也高于 v6/v7 P3 0.2848。

7. A1s1 负例说明更密 mask backproject 不是免费收益。
   stride=1 让 pre_points 增加到 10302.6/scene，conflict 升到 0.1198，AP 降到 0.2907。
   因此当前收益来自 ownership/partition，而不是简单扩大 support。
```

## 结论

```text
success=True for probe5 Lane3 object/AP gate
success=False for full ScanNet / Dynamic Replica final goal
success=True for Lane 1 correspondence sanity after frame_stride=1 repair

当前最清楚的结论是：
  D4RT native semi-dense surfel correspondence 在连续 frames 上成立；
  但 ScanNet metric geometry 还不能替代 RGB-D mesh；
  当前 2D mask cache 只有 stride-10 observations，必须显式建模 sparse measurement；
  B1 surfacelet single-mask ownership 在 probe5 上超过 v6/v7 dense-object baseline。

因此本轮可以报告 Lane3 probe5 method AP，但不能扩大为 full/dynamic claim。
下一步应做：
  1. 将 B1 扩到更大 ScanNet split，确认是否只是 probe5/chair support 偏置；
  2. 补齐连续 frames 的 2D mask observations，测试 dense measurement 是否进一步提升 AP；
  3. 对 scene0591_00 的低 AP50 做 failure autopsy；
  4. Dynamic Replica 仍需可用 instance/depth 数据后才能报告 official tracking。
```

## Insight

```text
这轮最有价值的信号有两个。

第一是“stride matters”：
  stride-10 clip 看起来像 D4RT field 失败；
  连续 16 帧立刻把 uv/cycle 指标修好。

第二是“ownership before support expansion matters”：
  B1 不做跨帧 full-mask component merge，而是把单帧 mask-owned surfacelet 当 object field proposal；
  它用更少 pre_points 和更低 conflict 拿到当前最高 AP/AP50/AP25。

metric geometry 和 semantic observation 仍是两个不同瓶颈：
  D4RT correspondence 已经能支持 image-space surfel tracking；
  Sim3 residual 说明它还不是可靠 ScanNet metric mesh；
  B1 说明 sparse semantic measurement 也能产生 object AP；
  但 full 4D object field 仍需要更多时间观测和动态数据验证。
```

## 证据链索引

```text
Code:
  Stream3D/tools/export_d4rt_grid_surfel_field_v8.py
  Stream3D/tools/compare_d4rt_adapter_official_v8.py
  Stream3D/tools/diagnose_v8_mask_measurement_coverage.py
  Stream3D/tools/export_v8_surfel_object_field.py
  Stream3D/stream4d/scannet_stream.py
  Stream3D/scripts/reproduce_v8_lane1.sh
  Stream3D/scripts/reproduce_v8_lane3.sh

G1 loger outputs:
  Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid16_scene0050_16f_loger/summary.{json,csv,md}
  Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32_scene0050_16f_loger/summary.{json,csv,md}
  Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_scene0050_16f_loger/summary.{json,csv,md}
  Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_scene0050_16f_stride1_loger/summary.{json,csv,md}
  Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger/summary.{json,csv,md}

G1 geometry:
  Stream3D/outputs/audit/v8_g1_grid16_scene0050_16f_loger_geometry.{json,csv,md}
  Stream3D/outputs/audit/v8_g1_grid32_scene0050_16f_loger_geometry.{json,csv,md}
  Stream3D/outputs/audit/v8_g1_grid32m002_scene0050_16f_loger_geometry.{json,csv,md}
  Stream3D/outputs/audit/v8_g1_grid32m002_scene0050_16f_stride1_loger_geometry.{json,csv,md}
  Stream3D/outputs/audit/v8_g1_grid32m002_probe5_16f_stride1_loger_geometry.{json,csv,md}

Adapter sanity:
  Stream3D/outputs/audit/v8_adapter_vs_official_scene0050_grid4_loger.{json,md}

Lane2:
  Stream3D/outputs/audit/v8_mask_measurement_coverage_probe5_stride1_loger.{json,csv,md}

Lane3 results:
  Stream3D/outputs/v8_surfel_object_field/stream4d_v8_a1_signed_history_m2_probe5_summary.{json,csv,md}
  Stream3D/outputs/v8_surfel_object_field/stream4d_v8_a2_signed_history_m1_probe5_summary.{json,csv,md}
  Stream3D/outputs/v8_surfel_object_field/stream4d_v8_b1_surfacelet_singlemask_probe5_summary.{json,csv,md}
  Stream3D/outputs/v8_surfel_object_field/stream4d_v8_c1_core_owned2_probe5_summary.{json,csv,md}
  Stream3D/outputs/v8_surfel_object_field/stream4d_v8_a1s1_signed_history_m2_probe5_summary.{json,csv,md}
  Stream3D/data/evaluation/scannet/stream4d_v8_*_probe5_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v8_b1_surfacelet_singlemask_probe5_scene*_class_agnostic.txt
  Stream3D/data/prediction/stream4d_v8_b1_surfacelet_singlemask_probe5_class_agnostic/*.npz
  Stream3D/data/TMP/stream4d_v8_b1_surfacelet_singlemask_probe5/*_pre_points.npy

Lane3 integrity:
  Stream3D/outputs/audit/v8_reportable_config_scan_lane3abc_probe5.{json,csv,md}
  Stream3D/outputs/audit/v8_metric_integrity_b1_surfacelet_singlemask_probe5.{json,md}

Logs:
  Stream3D/logs/stream4d_v8_g1_loger_py_compile.log
  Stream3D/logs/stream4d_v8_g1_loger_import_smoke.log
  Stream3D/logs/stream4d_v8_adapter_official_compare_py_compile.log
  Stream3D/logs/stream4d_v8_adapter_official_compare_import_smoke.log
  Stream3D/logs/stream4d_v8_adapter_vs_official_scene0050_grid4_loger.log
  Stream3D/logs/stream4d_v8_stride1_missing_mask_fix_py_compile.log
  Stream3D/logs/stream4d_v8_stride1_missing_mask_fix_import_smoke.log
  Stream3D/logs/stream4d_v8_g1_grid16_scene0050_16f_loger.log
  Stream3D/logs/stream4d_v8_g1_grid32_scene0050_16f_loger.log
  Stream3D/logs/stream4d_v8_g1_grid32m002_scene0050_16f_loger.log
  Stream3D/logs/stream4d_v8_g1_grid32m002_scene0050_16f_stride1_loger.log
  Stream3D/logs/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger.log
  Stream3D/logs/stream4d_v8_g1_grid16_scene0050_16f_loger_geometry.log
  Stream3D/logs/stream4d_v8_g1_grid32_scene0050_16f_loger_geometry.log
  Stream3D/logs/stream4d_v8_g1_grid32m002_scene0050_16f_loger_geometry.log
  Stream3D/logs/stream4d_v8_g1_grid32m002_scene0050_16f_stride1_loger_geometry.log
  Stream3D/logs/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger_geometry.log
  Stream3D/logs/stream4d_v8_mask_measurement_coverage_py_compile.log
  Stream3D/logs/stream4d_v8_mask_measurement_coverage_import_smoke.log
  Stream3D/logs/stream4d_v8_mask_measurement_coverage_probe5_stride1_loger.log
  Stream3D/logs/stream4d_v8_lane3a_py_compile.log
  Stream3D/logs/stream4d_v8_lane3a_import_smoke.log
  Stream3D/logs/stream4d_v8_a1_signed_history_m2_scene0050_export.log
  Stream3D/logs/stream4d_v8_a1_signed_history_m2_scene0050_eval.log
  Stream3D/logs/stream4d_v8_a1_signed_history_m2_probe5_export.log
  Stream3D/logs/stream4d_v8_a1_signed_history_m2_probe5_eval.log
  Stream3D/logs/stream4d_v8_a2_signed_history_m1_probe5_export.log
  Stream3D/logs/stream4d_v8_a2_signed_history_m1_probe5_eval.log
  Stream3D/logs/stream4d_v8_a1s1_signed_history_m2_probe5_export.log
  Stream3D/logs/stream4d_v8_a1s1_signed_history_m2_probe5_eval.log
  Stream3D/logs/stream4d_v8_b1_surfacelet_singlemask_probe5_export.log
  Stream3D/logs/stream4d_v8_b1_surfacelet_singlemask_probe5_eval.log
  Stream3D/logs/stream4d_v8_c1_core_owned2_probe5_export.log
  Stream3D/logs/stream4d_v8_c1_core_owned2_probe5_eval.log
  Stream3D/logs/stream4d_v8_b1_surfacelet_singlemask_probe5_per_scene_eval.log
  Stream3D/logs/stream4d_v8_reportable_config_scan_lane3abc_probe5_r2.log
  Stream3D/logs/stream4d_v8_metric_integrity_b1_surfacelet_singlemask_probe5_r2.log
  Stream3D/logs/stream4d_v8_lane3abc_final_py_compile.log
  Stream3D/logs/stream4d_v8_lane3abc_final_import_smoke.log
  Stream3D/logs/stream4d_v8_lane3abc_final_unit_tests.log
  Stream3D/logs/stream4d_v8_reproduce_lane3_bash_n.log
  Stream3D/logs/stream4d_v8_official_infer_track_help_loger.log
  Stream3D/logs/stream4d_v8_dynamic_replica_env_loger.log
  Stream3D/logs/stream4d_v8_loger_unit_tests_unittest.log

Dynamic Replica:
  Stream3D/outputs/audit/dynamic_replica_env_v8_loger.{json,md}

Environment:
  Stream3D/pip_freeze_v8_loger.txt
```

## 审计包

```text
latest packet:
  stream4d_v8_code_audit_packet_20260609_0503_lane3_probe5_final.zip

latest sha256:
  see sibling file stream4d_v8_code_audit_packet_20260609_0503_lane3_probe5_final.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

latest filelist:
  stream4d_v8_code_audit_packet_20260609_0503_lane3_probe5_final_filelist.txt

latest zip test:
  No errors detected in compressed data of stream4d_v8_code_audit_packet_20260609_0503_lane3_probe5_final.zip.

latest file count:
  273

previous Lane1 packet:
  stream4d_v8_code_audit_packet_20260609_0440_lane1_final.zip

previous Lane1 sha256:
  see sibling file stream4d_v8_code_audit_packet_20260609_0440_lane1_final.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

previous Lane1 filelist:
  stream4d_v8_code_audit_packet_20260609_0440_lane1_final_filelist.txt

previous Lane1 zip test:
  see stream4d_v8_code_audit_packet_20260609_0440_lane1_final_ziptest.log

previous Lane1 file count:
  216
```
