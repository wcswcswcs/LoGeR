# Stream4D v6 实验结果复盘

日期：2026-06-08（Asia/Singapore）  
计划文件：`docs/stream4d_v6_method_first_audit_and_experiment_plan_for_codex.md`  
执行日志：`docs/stream4d_v6_执行日志.md`  
结果根目录：`Stream3D/outputs/`、`Stream3D/data/evaluation/scannet/`  
代码审计包：`stream4d_v6_code_review_packet_20260608_2136.zip`

本复盘只记录真实运行得到的数据。没有实现或没有完整运行的 phase 明确标注为 `not_run` 或 `blocked`，不把诊断结果冒充为方法结果。

## 当前结论

```text
v6 本轮仍没有达成 ScanNet probe5 method gate。

Phase A 通过：P0/P1 审计修复已完成，py_compile/import smoke/unit tests/manifest scanner/metric integrity 均通过。
Phase B 部分完成：新增 D4RT geometry Sim3 anchor diagnostic，25/25 windows 有结果；但 residual 远高于 gate，G1-D4RT segmentation AP 没有启动。
Phase C 继续完成了 score-mode blocker 复核和一个 minimal Typed Evidence Graph v3 candidate：score bug 修复后 local proposal score modes 全部真实重跑，但 AP 仍明显失败；minimal typed v3 跑完 scene0050/probe5 后更差。
Phase D/E/F 部分推进：minimal typed v3 已实现并失败；随后补做 core/fringe/reject、radius growth、object competition/small-rescue、no-group/top-k 压缩实验。当前最好 reportable 结果为 no-group/top-k compact-only，AP/AP50 小幅超过 v4.1 且 #pred<=300，但 AP/AP50/AP25/union 仍未达到 v6 gate。
Phase G blocked：本地 Dynamic Replica 数据缺失，不能报告 tracking metrics。

Full ScanNet final 未启动，因为 probe5 gate 未通过。
```

v6 probe5 gate：

```text
pure Stream4D fixed32 AP   >= 0.32
pure Stream4D fixed32 AP50 >= 0.53
pure Stream4D fixed32 AP25 >= 0.70
#pred <= 300
union in target >= 0.94
metric integrity pass
no oracle config in reportable set
D4RT geometry degradation experiment completed or explicitly marked diagnostic/blocker
```

本轮继续推进后的当前最好 v6 reportable method 结果：

```text
stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve
AP/AP50/AP25 = 0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401
#pred = 295.6
pre/union = 4.5145% / 4.1578%
metric integrity phase0_pass = true
uses_gt = false
```

它在 AP/AP50 上小幅超过 v4.1 current best，并把 #pred 从 415.6 降到 295.6；但 AP/AP50/AP25/union 仍未过 v6 gate。

此前最好的 v6 新增 localprop 结果是 `score=one`：

```text
stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075
AP/AP50/AP25 = 0.11252410871496846 / 0.2853900969507906 / 0.4797727369815028
#pred = 173.20
pre/union = 35.7973%
```

它没有超过 v4.1 current best：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
AP/AP50/AP25 = 0.2816154378895367 / 0.4975830336133305 / 0.6902541954477854
#pred = 415.6
```

也远低于 Stream3D-on-Stream4D 32f diagnostic：

```text
scannet_on_stream4d_32f_probe5
AP/AP50/AP25 = 0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108
```

因此不能启动 full ScanNet final，不能 claim Stream4D v6 超过 Stream3D。

用户追问后补做的 minimal typed graph 结果：

```text
stream4d_v6_typedv3_probe5_c2_trackonly_ioc060
AP/AP50/AP25 = 0.03347846361047295 / 0.12849540654750258 / 0.3977530181719252
#pred = 57.20
pre/union = 24.1733%
metric integrity phase0_pass = true
```

它比 localprop best 更差，说明继续实现 graph 的第一版没有解决问题，而是暴露了更明确的 over-merge / missing split blocker。

## 已完成代码修复审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `Stream3D/evaluation/evaluate.py` | 读取 `config_manifest.json`；新增 `--require-manifest`；拒绝 `uses_gt=true` 或 `is_diagnostic_only=true` 的普通 eval | 修复 P0 evaluator-manifest guard | unit test pass；v6 eval 均用 `--require-manifest` |
| `Stream3D/evaluation/evaluate.py` | 新增 `--tmp_root/--tmp_config` 并按 config 读取 pre_points | 让 evaluator 不再硬编码 TMP 路径 | metric integrity pass |
| `Stream3D/stream4d/export_scannet.py` | `mask_backproject` object record 写入 `observations/area_score/carrier_count/reliability` | 修复 local proposal score-mode bug | unit test pass；score modes 重跑 |
| `Stream3D/tools/export_local_proposal_bank.py` | `--export-score-mode` 增加 `reliability` | 解除 v6 score-mode sweep CLI blocker | reliability rerun 完成 |
| `Stream3D/tools/fuse_prediction_configs.py` | manifest 记录 `drop_overlap_pre_points_config`，外部 support 标记 diagnostic-only | 防止 Stream3D/scannet support selection signal 混入 pure method | unit test pass |
| `Stream3D/tools/d4rt_geometry_diagnostic.py` | 新增 D4RT carrier 同像素 anchor Sim3 residual 诊断 | 部分回应 Phase B D4RT geometry degradation | 25/25 windows 输出 |
| `Stream3D/tools/export_typed_evidence_graph_v3.py` | 新增 minimal typed graph candidate：positive_track / positive_complement / negative_conflict / weak_bridge attach-only | 用户追问后继续尝试 v6 Typed Evidence Graph 方向 | scene0050/probe5 跑完；metric integrity pass；AP fail |
| `Stream3D/tests/test_stream4d_protocol_fixes.py` | 新增 evaluator manifest、score-mode、fusion support、Sim3 单测 | 让 P0 修复可审计 | 12 tests pass |
| `Stream3D/splits/scannet_v6_probe5.txt` | 固定 v6 probe5 scene list | 统一 v6 入口 | 实验使用 |
| `Stream3D/tests/fixtures/*` | 新增 tiny fixture | 满足 v6 审计包要求 | 已打包 |

最终验证：

```text
py_compile: pass
import smoke: pass
clean packet import smoke: pass
unit tests: Ran 12 tests in 1.368s, OK
```

## Phase A 复盘：Audit and Reproducibility

### py_compile / import / unit tests

证据：

```text
Stream3D/logs/stream4d_v6_py_compile.log
Stream3D/logs/stream4d_v6_import_smoke.log
Stream3D/logs/stream4d_v6_packet_clean_import_smoke.log
Stream3D/logs/stream4d_v6_unit_tests.log
```

结果：

```text
py_compile exit=0
import smoke OK
Ran 12 tests in 1.368s, OK
```

import smoke 覆盖：

```text
stream4d.run_scannet
stream4d.object_memory_v2
stream4d.export_scannet
stream4d.reliable_densifier
tools.verify_stream4d_metric_integrity
tools.scan_reportable_configs
tools.d4rt_geometry_diagnostic
```

### Manifest scanner

扫描 configs：

```text
stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075
stream4d_v6_localprop_96f_probe5_min2_bestframe_score_observations_ioc075
stream4d_v6_localprop_96f_probe5_min2_bestframe_score_area_ioc075
stream4d_v6_localprop_96f_probe5_min2_bestframe_score_reliability_ioc075
```

结果：

```text
num_configs = 4
num_configs_missing_manifest = 0
num_oracle_configs = 0
num_reportable_method_configs = 4
num_diagnostic_only_configs = 0
num_suspicious_configs = 0
num_uses_gt_and_method_result = 0
```

证据：

```text
Stream3D/outputs/audit/v6_reportable_config_scan_localprop_score_modes.md
Stream3D/outputs/audit/v6_reportable_config_scan_localprop_score_modes.json
Stream3D/outputs/audit/v6_reportable_config_scan_localprop_score_modes.csv
```

### Metric integrity

结果：

```text
evaluator_ap_core_equal_by_hash = True
has_pre_points_load_original = True
has_pre_points_load_current = True
gt_files_read_by_rescore = False
num_configs_missing_manifest = 0
num_oracle_configs = 0
num_reportable_method_configs = 4
num_suspicious_configs = 0
phase0_pass = True
alignment mean/min = 1.0 / 1.0
alignment failed = 0
```

四个 v6 score-mode configs 都满足：

```text
OK = 5/5 scenes
pre % = 35.7973
union % = 35.7973
#pred = 173.20
```

证据：

```text
Stream3D/outputs/audit/v6_metric_integrity_localprop_score_modes_probe5.md
Stream3D/outputs/audit/v6_metric_integrity_localprop_score_modes_probe5.json
```

Phase A 结论：

```text
P0 审计修复通过。
本轮新增 reportable configs 没有 uses_gt=true，没有 oracle/diagnostic 混入 method table。
```

## Phase B 复盘：D4RT Geometry Diagnostic

本轮没有完成 `G1 Stream3D-D4RT internal geometry + Sim3 export` 的 segmentation AP。原因是 Stream3D 内部 geometry replacement 需要改 set-cover/manifold/merging 的几何源，不是当前最小修复范围。为避免轻易放弃，本轮按计划 failure path 先实现了同像素 anchor Sim3 residual 诊断。

运行范围：

```text
debug_root = Stream3D/outputs/stream4d_v5_cache_96f_probe5
scenes = 5
windows = 25
anchor source = D4RT carrier same-pixel A0 anchors vs ScanNet RGB-D world points
```

结果摘要：

| 指标 | 值 |
|---|---:|
| `num_windows` | `25` |
| `num_ok_windows` | `25` |
| `num_failed_windows` | `0` |
| `sim3_anchor_count_mean/min/max` | `446.08 / 199 / 681` |
| `sim3_scale_mean/min/max` | `0.588463045245313 / 0.424304693041852 / 0.8916739660680099` |
| `sim3_residual_median_mean/min/max` | `0.6801382969694636 / 0.37772401787093146 / 1.0206364344867356` |
| `sim3_residual_p90_mean/min/max` | `1.2819213888794445 / 0.6010213349749307 / 2.2091675825039903` |
| `visibility_mean_mean/min/max` | `0.9988281679153442 / 0.9928140044212341 / 0.999945342540741` |
| `confidence_mean_mean/min/max` | `0.9998375511169434 / 0.9983137249946594 / 0.9999514222145081` |
| `uv_in01_rate_mean/min/max` | `0.6064773520758063 / 0.3327888853904282 / 0.8376436161022806` |

计划 gate 对照：

```text
最低期望：Sim3 residual median <= 0.08m 或相对 scene scale <= 2%。
真实结果：median residual mean = 0.6801m，p90 mean = 1.2819m。
```

判断：

```text
D4RT same-pixel carrier geometry 当前不能直接替代 ScanNet RGB-D/pose 作为 Stream3D ScanNet 主几何路径。
G1-D4RT internal geometry segmentation AP 没有启动；如果启动，需要先解决 frame/reference/scale/coordinate 或 sparse/noisy support 适配问题。
```

证据：

```text
Stream3D/outputs/audit/v6_d4rt_geometry_probe5.md
Stream3D/outputs/audit/v6_d4rt_geometry_probe5.csv
Stream3D/outputs/audit/v6_d4rt_geometry_probe5.json
Stream3D/logs/stream4d_v6_d4rt_geometry_probe5.log
```

Insight：

```text
visibility/confidence 几乎都很高，但 Sim3 residual 仍很大，说明问题不只是低置信 query 的噪声。
当前更可能是 D4RT reference/coordinate/scale 与 ScanNet world 的对齐方式、同像素 anchor 解释、或 D4RT geometry 稀疏/漂移导致的几何不适配。
因此 ScanNet 叙事不能写成 D4RT-native geometry 已经成立，只能写 D4RT carrier correspondence + RGB-D evaluation bridge 仍是当前可靠路径。
```

## Phase C 复盘：Local Proposal Score-mode Blocker

目的：

```text
验证 v5 local proposal 失败是否只是 observations score 字段缺失造成。
```

公共设置：

```text
debug source = outputs/stream4d_v5_cache_96f_probe5
same_frame_policy = best_per_frame
min_observations = 2
min_frames = 2
support = mask_backproject
export_nn_radius = 0.05
export_min_points_per_object = 100
```

导出摘要：

| Score mode | raw proposals | kept proposals | exported objects | conflict rate | mask hit rate | score mean | score max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `one` | `7017` | `1179` | `866` | `0.6162319170120751` | `0.7790966427156196` | `1.0` | `1.0` |
| `observations` | `7017` | `1179` | `866` | `0.6162319170120751` | `0.7790966427156196` | `6.381442737579346` | `20.0` |
| `area` | `7017` | `1179` | `866` | `0.6162319170120751` | `0.7790966427156196` | `1132.9359008789063` | `11439.0` |
| `reliability` | `7017` | `1179` | `866` | `0.6162319170120751` | `0.7790966427156196` | `202.1749755859375` | `1775.274658203125` |

AP 结果：

| Config | Score mode | AP | AP50 | AP25 | #pred | pre/union |
|---|---|---:|---:|---:|---:|---:|
| `stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075` | `one` | `0.11252410871496846` | `0.2853900969507906` | `0.4797727369815028` | `173.20` | `35.7973%` |
| `stream4d_v6_localprop_96f_probe5_min2_bestframe_score_observations_ioc075` | `observations` | `0.03954383581861127` | `0.13267544120600216` | `0.36290128014492185` | `173.20` | `35.7973%` |
| `stream4d_v6_localprop_96f_probe5_min2_bestframe_score_area_ioc075` | `area` | `0.020921328935722804` | `0.07802752210185354` | `0.22924048194714897` | `173.20` | `35.7973%` |
| `stream4d_v6_localprop_96f_probe5_min2_bestframe_score_reliability_ioc075` | `reliability` | `0.031275198243860235` | `0.1112769817839105` | `0.35076569274898506` | `173.20` | `35.7973%` |

判断：

```text
1. score-mode bug 已修复，observations/area/reliability score 都真实非零。
2. 修复后 AP 没有提升，反而低于 score=one。
3. 因此 v5/v6 local proposal 失败不能归因于 score 字段缺失。
4. local proposal 的核心问题仍是 object formation / conflict / boundary，而不是 score calibration。
```

证据：

```text
Stream3D/outputs/local_proposal_bank_v6/stream4d_v6_localprop_score_modes_summary.json
Stream3D/data/evaluation/scannet/stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_localprop_96f_probe5_min2_bestframe_score_observations_ioc075_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_localprop_96f_probe5_min2_bestframe_score_area_ioc075_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_localprop_96f_probe5_min2_bestframe_score_reliability_ioc075_class_agnostic.txt
Stream3D/outputs/audit/v6_metric_integrity_localprop_score_modes_probe5.md
```

Insight：

```text
同一批 predictions、同一 #pred、同一 support coverage 下，排序分数从 one 改成 observations/area/reliability 反而损害 AP。
这说明 proposal quality 与无监督 score 之间不一致：更多 observation 或更大 area 不代表更接近 GT instance，反而可能偏向大而污染的 object。
下一步不应继续 score sweep，而应实现 typed evidence graph + core/fringe/reject。
```

## Phase D/E/F 状态

```text
Phase C Typed Evidence Graph v3: minimal_candidate_done_fail
Phase D core/fringe/reject support: blocked / not_implemented
Phase E global object competition: blocked / not_implemented
Phase F split/merge-capable object memory: blocked / not_implemented
```

### minimal Typed Evidence Graph v3 继续实验

```text
用户追问“不继续解决问题”后，继续实现并运行了一个 minimal typed graph candidate。
它不是完整 v6 graph/memory，但已经包含 node schema、typed edges、same-frame cannot-link 和 weak_bridge attach-only。
```

新增文件：

```text
Stream3D/tools/export_typed_evidence_graph_v3.py
```

实现边界：

```text
1. 输入：outputs/stream4d_v5_cache_96f_probe5/<scene>/local_props_window*.json。
2. node：frame_id、mask_id、coverage、point_ids、centroid、bbox、weak flag。
3. edges：
   positive_track = 跨帧 backproject point overlap。
   positive_complement = centroid/bbox 几何邻近补全。
   negative_conflict = same-frame overlap cannot-link。
   weak_bridge = weak endpoint attach-only。
4. DSU merge 时阻止 same-frame conflict / cannot-link。
5. 输出 prediction npz、TMP pre_points、object_dict.npy、config_manifest.json、summary.json。
```

缺口：

```text
没有完整 negative_ownership。
没有 core/fringe/reject support。
没有 component split audit。
没有 object competition / WTA ownership。
没有 split/merge memory lifecycle。
```

### scene0050 探针结果

| Config | 说明 | AP | AP50 | AP25 | kept_components | union_points | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_typedv3_scene0050_minimal` | positive_track + complement + weak bridge | `0.011495007248902506` | `0.04739194753157756` | `0.4607985515043228` | `33` | `25231` | weak/complement 明显过合并 |
| `stream4d_v6_typedv3_scene0050_c2_trackonly_ioc060` | track-only, IoC 0.60 | `0.046778612740200544` | `0.234317709001236` | `0.4910354817322768` | `49` | `34242` | scene0050 最好，但仍低 |
| `stream4d_v6_typedv3_scene0050_c2_trackonly_ioc079` | track-only, IoC 0.79 | `0.04040711787907115` | `0.2108040535136109` | `0.48872778226274444` | `58` | `32073` | 提高阈值没有改善 |
| `stream4d_v6_typedv3_scene0050_c3_weak_limited_ioc060` | limited weak bridge | `0.03985265384110804` | `0.19815002337241225` | `0.4408752842946256` | `47` | `31228` | weak bridge 仍更差 |

判断：

```text
scene0050 上，track-only ioc060 是最不坏的 typed v3 变体，因此推到 probe5。
weak_bridge 和 complement 在当前实现下都降低 AP，说明它们容易把相邻或污染 observation 合并进错误 component。
```

证据：

```text
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_scene0050_minimal_scene0050_00_summary.json
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_scene0050_c2_trackonly_ioc060_scene0050_00_summary.json
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_scene0050_c2_trackonly_ioc079_scene0050_00_summary.json
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_scene0050_c3_weak_limited_ioc060_scene0050_00_summary.json
Stream3D/data/evaluation/scannet/stream4d_v6_typedv3_scene0050_minimal_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_typedv3_scene0050_c2_trackonly_ioc060_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_typedv3_scene0050_c2_trackonly_ioc079_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_typedv3_scene0050_c3_weak_limited_ioc060_class_agnostic.txt
```

### probe5 track-only ioc060 结果

Full probe5：

```text
Config = stream4d_v6_typedv3_probe5_c2_trackonly_ioc060
AP/AP50/AP25 = 0.03347846361047295 / 0.12849540654750258 / 0.3977530181719252
#pred = 57.20
pre/union = 24.1733%
metric integrity phase0_pass = true
alignment mean/min = 1.0 / 1.0
```

per-scene graph summary：

| Scene | nodes | positive_track_edges | negative_conflict_edges | accepted_track | raw_components | kept_components | dropped_components | union_points | hit_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene0011_00` | `255` | `935` | `21` | `176` | `79` | `19` | `60` | `29325` | `0.6416344583701575` |
| `scene0030_00` | `1167` | `6856` | `154` | `1019` | `148` | `98` | `50` | `90459` | `0.8386020550620544` |
| `scene0050_00` | `539` | `3373` | `66` | `467` | `72` | `49` | `23` | `34242` | `0.925039969291374` |
| `scene0081_01` | `238` | `570` | `8` | `162` | `76` | `29` | `47` | `24311` | `0.47566825228504306` |
| `scene0591_00` | `1155` | `6696` | `230` | `996` | `159` | `91` | `68` | `84318` | `0.7687900159074992` |

Metric safety：

```text
config_manifest.json:
  uses_gt = false
  is_method_result = true
  is_diagnostic_only = false
  support_policy = typed_evidence_graph_v3_mask_backproject

v6_metric_integrity_typedv3_probe5_c2_trackonly_ioc060:
  num_configs_missing_manifest = 0
  num_oracle_configs = 0
  num_reportable_method_configs = 1
  num_suspicious_configs = 0
  phase0_pass = true
```

证据：

```text
Stream3D/data/evaluation/scannet/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_class_agnostic.txt
Stream3D/outputs/audit/v6_metric_integrity_typedv3_probe5_c2_trackonly_ioc060.md
Stream3D/outputs/audit/v6_metric_integrity_typedv3_probe5_c2_trackonly_ioc060.json
Stream3D/data/prediction/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_class_agnostic/config_manifest.json
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0011_00_summary.json
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0030_00_summary.json
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0050_00_summary.json
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0081_01_summary.json
Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0591_00_summary.json
```

分析：

```text
1. typed v3 probe5 不是因为 evaluator 或 manifest 失败而低，metric integrity 已通过。
2. #pred = 57.20，明显少于 localprop 的 173.20；AP 从 0.1125 降到 0.0335，说明当前 graph merge 更像 over-merge / wrong-merge。
3. same-frame negative_conflict 存在，但不能阻止跨帧错误链式合并；这就是计划里 component split audit、negative_ownership、object competition 仍必须实现的原因。
4. weak bridge 和 positive_complement 在 scene0050 上都造成更低 AP，说明不能继续无约束扩大 support。
5. 当前 minimal typed v3 是继续解决问题的一步，但结果是负证据；不能把它写成完整 v6 成功。
```

Insight：

```text
typed edge 的方向是对的，但“能连起来”不等于“应该连成一个 instance”。
当前缺少 object 内部 core/fringe 角色和跨对象 ownership 竞争，导致 graph 把多帧 observation 压成过少 component。
下一步不是再调单个 IoC 阈值，而是实现 split-aware component audit：找出每个大 component 内部的 same-frame near-conflict、multi-peak geometry、low-core/high-fringe 比例，再决定切分。
```

### 仍不能写

```text
不能写完整 Typed Evidence Graph v3 已完成。
不能写 core/fringe/reject 已验证。
不能写 split/merge memory 已经跑 96f/128f。
不能写 v6 method 通过 probe5。
```

## Phase G 复盘：Dynamic Replica

环境检查：

```text
data_root_exists = false
split_dir_exists = false
annotation_exists = false
all_required_camera_fields_present = false
usable_scene_count = 0
can_report_official_instance_tracking = false
can_report_d4rt_trajectory_metrics = false
can_report_only_qualitative_consistency = false
annotation_error = annotation file missing
```

证据：

```text
Stream3D/outputs/audit/dynamic_replica_env_v6.md
Stream3D/outputs/audit/dynamic_replica_env_v6.json
```

结论：

```text
Phase G blocked by missing local Dynamic Replica dataset.
不能报告 IDF1、MOTA/MOTP、4D IoU、trajectory metrics 或 qualitative consistency。
```

## Final Gate 判断

| Gate | 本轮结果 | 结论 |
|---|---:|---|
| pure Stream4D fixed32 AP >= 0.32 | best v6 localprop AP `0.11252410871496846`; typed v3 AP `0.03347846361047295` | fail |
| pure Stream4D fixed32 AP50 >= 0.53 | best v6 localprop AP50 `0.2853900969507906`; typed v3 AP50 `0.12849540654750258` | fail |
| pure Stream4D fixed32 AP25 >= 0.70 | best v6 localprop AP25 `0.4797727369815028`; typed v3 AP25 `0.3977530181719252` | fail |
| #pred <= 300 | localprop `173.20`; typed v3 `57.20` | pass numerically, but typed v3 under-predicts/over-merges |
| union in target >= 0.94 | localprop support coverage `35.7973%`; typed v3 `24.1733%` | fail/insufficient |
| metric integrity pass | `phase0_pass=True` | pass |
| no oracle config in reportable set | `num_oracle_configs=0` | pass |
| D4RT geometry degradation completed or marked diagnostic/blocker | diagnostic completed; G1 AP blocked | diagnostic/blocker |

最终判断：

```text
v6 本轮没有达到 ScanNet probe5 method gate。
不启动 full ScanNet final。
不 claim Stream4D v6 超过 Stream3D。
不 claim D4RT-native geometry 已验证。
不 claim Dynamic Replica tracking 已验证。
```

## 总分析

```text
1. Phase A 的 P0 审计修复是实质进展：evaluator 现在能按 manifest 拒绝 GT/diagnostic eval，reportable v6 configs 均有 manifest 且 uses_gt=false。
2. score-mode blocker 已被真实复核：observations/area/reliability 字段修复后确实生效，但 AP 下降，说明 local proposal 失败不是 score-field bug。
3. D4RT geometry diagnostic 给出强负证据：25 个 windows 可拟合 Sim3，但 residual median mean 0.680m、p90 mean 1.282m，远高于计划 gate。
4. 当前最可靠的解释仍是 v5/v6 计划中的判断：coverage/cache 不是主 blocker，object formation + boundary purity + one-to-one assignment 才是主 blocker。
5. 用户追问后继续实现了 minimal typed v3，并真实跑完 scene0050/probe5；结果更差，主要暴露 over-merge 和 missing split/ownership blocker。
6. 本轮没有实现完整 core/fringe/reject、object competition 和 split/merge memory，因此不能把 typed v3 minimal 写成完整 v6 方法成功。
7. Dynamic Replica 数据缺失，不能造 tracking 指标。
```

## 下一步建议

必须先做：

```text
1. 在现有 minimal typed v3 上补完整 split-aware graph audit：
   negative_ownership、same-frame near-conflict、component multi-peak geometry、core/fringe ratio。

2. 实现 core/fringe/reject support：
   先 core-only，再 connected fringe，再 ownership-aware WTA；避免把 fringe 当 core 直接合并。

3. 基于 component audit 实现 split：
   对低 core purity / 高 near-conflict component 做切分，再重跑 scene0050 与 probe5。

4. 如果 D4RT geometry 继续作为主线，必须先调试 reference/coordinate/scale/anchor selection，使 Sim3 residual 接近计划 gate；否则 ScanNet 主路径仍应写成 RGB-D evaluation bridge。

5. Dynamic Replica 需要先补齐本地数据路径，使 env checker 至少 `usable_scene_count > 0`，再谈 tracking metrics。
```

安全表述：

```text
v6 本轮完成的是审计基础设施和两个 blocker 的真实复核。
结果支持“不要继续 score sweep / local proposal 直出 / D4RT-native geometry claim”。
minimal typed evidence graph 已经补跑并失败；真正的 v6 method-first 算法核心仍未完成，下一步应该做 split-aware graph 与 object-internal support，而不是启动 full final。
```
## v6 continuation：Phase D core/fringe/reject 与 ownership-aware WTA

### 为什么继续

用户追问：

```text
达成目标了吗？没有请继续，按计划里的推荐思路修改。
```

继续前状态：

```text
v6 没有达到 probe5 method gate。
best v6 localprop score-one:
  AP/AP50/AP25 = 0.11252410871496846 / 0.2853900969507906 / 0.4797727369815028
  #pred = 173.20
typed v3 track-only:
  AP/AP50/AP25 = 0.03347846361047295 / 0.12849540654750258 / 0.3977530181719252
```

本轮按计划 Phase D 继续，不做 GT 选择，不做 oracle，不把失败结果写成成功。

### 代码修复审计

修改：

```text
Stream3D/tools/export_typed_evidence_graph_v3.py
  新增 support-mode = union/core/core_connected_fringe。
  新增 core/fringe/reject 统计字段。
  默认仍为 union，保持旧结果可复现。

Stream3D/tools/split_core_fringe_prediction.py
  补写 config_manifest.json。
  新增 assignment-mode = low_conflict/wta。
  新增 wta-priority = score/small_area/large_area/score_over_sqrt_area。
```

合理性：

```text
1. Phase D 计划要求 core/fringe/reject 与 ownership-aware WTA。
2. 本修改不读取 GT，只基于 prediction mask ownership conflict。
3. 默认参数保持旧行为，因此不会改变已有已记录结果。
```

验证：

```text
py_compile pass
unit tests: Ran 12 tests in 1.347s, OK
```

### Localprop core-only 结果

输入：

```text
stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075
baseline AP/AP50/AP25 = 0.11252410871496846 / 0.2853900969507906 / 0.4797727369815028
baseline #pred = 173.20
```

结果：

| Config | Scope | AP | AP50 | AP25 | #pred | pre/union | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_localprop_scene0050_coreonly_scoreone_ioc075` | scene0050 | `0.18047138047138048` | `0.40909090909090906` | `0.55078125` | n/a | n/a | 单场景正信号 |
| `stream4d_v6_localprop_probe5_coreonly_scoreone_ioc075` | probe5 | `0.1326388888888889` | `0.25392156862745097` | `0.5254354960868468` | `76.0` | `13.1310%` | AP/AP25 小幅改善，AP50 下降，未达 gate |

metric integrity：

```text
stream4d_v6_localprop_probe5_coreonly_scoreone_ioc075:
  phase0_pass = true
  mean_pre_points_ratio = 0.1313102002615043
  mean_prediction_union_ratio = 0.1313102002615043
  mean_num_pred_instances = 76.0
```

分析：

```text
core-only 删除冲突 ownership 后，AP 从 0.1125 提升到 0.1326，AP25 从 0.4798 提升到 0.5254。
但 AP50 从 0.2854 降到 0.2539，说明核心点更纯但边界/召回不足。
平均实例数从 173.2 降到 76.0，过度收缩。
```

### Ownership variants

scene0050 探测：

| Config | 关键设置 | AP | AP50 | AP25 | 结论 |
|---|---|---:|---:|---:|---|
| `coreown2_ratio020` | `max_core_owners=2`, `min_core_ratio=0.20` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` | scene0050 最佳 AP 折中 |
| `corefringe_low001` | 低分 fringe | `0.05037695279245847` | `0.1475545900178253` | `0.2781808035714286` | 失败 |
| `corefringeplus_low001` | 低分 full support fringe | `0.05823640715060206` | `0.17542613636363635` | `0.29265625` | 失败 |
| `wta_smallarea` | WTA small-area owner | `0.0813218279736137` | `0.1911670918367347` | `0.5151417525773196` | 失败 |
| `wta_scoresqrtarea` | WTA score/sqrt(area) | `0.0813218279736137` | `0.1911670918367347` | `0.5151417525773196` | 失败 |
| `wta_largearea` | WTA large-area owner | `0.13270049283154123` | `0.33845766129032256` | `0.5507172131147541` | 仍低于 core-only AP50 |

将 scene0050 最佳折中推到 probe5：

| Config | AP | AP50 | AP25 | #pred | pre/union | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_localprop_probe5_coreown2_ratio020_scoreone_ioc075` | `0.13568501383679288` | `0.27218506615709775` | `0.520180250783699` | `98.0` | `21.9675%` | 比 core-only AP/AP50 略好，但仍远低于 gate |

metric integrity：

```text
phase0_pass = true
mean_pre_points_ratio = 0.21967539878907916
mean_prediction_union_ratio = 0.21967539878907916
mean_num_pred_instances = 98.0
```

分析：

```text
1. coreown2_ratio020 是本轮 localprop 分支最好的 probe5 AP，但只从 0.1125 提升到 0.1357。
2. AP50 仍低于原 localprop score-one 的 0.2854，说明放宽 ownership 能拿回召回，但边界质量仍不足。
3. WTA 分配不是当前救法；它在 scene0050 上明显弱于 low_conflict core 过滤。
4. 低分 fringe 失败，说明把冲突点另做低分预测会引入大量低质量候选。
```

Insight：

```text
core ownership 过滤确实能找到更纯的 object seed，但仅靠点级 ownership 不能生成完整实例。
下一步如果继续，不应再扩大 fringe 或硬 WTA，而应实现 object-level split/merge：用 core seed 保持身份，再用可解释的 boundary growth 恢复 support。
```

### Boundary-growth 近似负结果

为了验证“core seed + 恢复边界”是否能救回召回，本轮又尝试了 high-score core + low-score original full mask。

scene0050 结果：

| Config | 设置 | AP | AP50 | AP25 | 结论 |
|---|---|---:|---:|---:|---|
| `stream4d_v6_localprop_scene0050_corefull_low001_scoreone_ioc075` | `max_core_owners=1`, low full mask score `0.001` | `0.05823640715060206` | `0.17542613636363635` | `0.29265625` | 明显失败 |
| `stream4d_v6_localprop_scene0050_coreown2_ratio020_full_low001_scoreone_ioc075` | `max_core_owners=2`, `min_core_ratio=0.20`, low full mask score `0.001` | `0.09026673251270163` | `0.23627857541120714` | `0.4650996654586005` | 仍低于 core-only |

分析：

```text
1. 直接把 original full mask 作为低分边界恢复，会把污染重新带回来。
2. 它比 core-only 的 0.18047 / 0.40909 / 0.55078 差很多，因此不推 probe5。
3. 这说明真正需要的是 geometry/graph-aware boundary growth，而不是把 core seed 和原 mask 简单并列输出。
```

### v4.1 current-best core-only 审计

动机：

```text
v4.1 current-best 距离 AP/AP50/AP25 gate 更近，但 #pred=415.6 过多。
尝试用 core-only 后处理压 #pred，检查是否能成为 v6 可报告 method。
```

输入：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
AP/AP50/AP25 = 0.2816154378895367 / 0.4975830336133305 / 0.6902541954477854
#pred = 415.6
```

结果：

| Config | TMP policy | AP | AP50 | AP25 | #pred | pre % | union % | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `stream4d_v6_v41self_probe5_coreonly` | `recompute` | `0.6062679879644165` | `0.7970712323390894` | `0.906634288330717` | `11.2` | `0.9256%` | 高 AP 但 tiny-pre_points，不能算成功 |
| `stream4d_v6_v41self_probe5_coreonly_on_v41support` | `support` | `0.06277817955517258` | `0.1449568388496515` | `0.18177814707104278` | `11.2` | `4.5145%` | 同一预测在原 support universe 下失败 |

证据：

```text
Stream3D/outputs/audit/v6_metric_integrity_v41self_probe5_coreonly.json
Stream3D/outputs/audit/v6_metric_integrity_v41self_probe5_coreonly_on_v41support.json
Stream3D/data/evaluation/scannet/stream4d_v6_v41self_probe5_coreonly_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_v41self_probe5_coreonly_on_v41support_class_agnostic.txt
```

分析：

```text
1. `stream4d_v6_v41self_probe5_coreonly` 表面 AP 很高，但 mean_pre_points_ratio 只有 0.00925595426581326。
2. 它平均只预测 11.2 个实例，只覆盖极小点集；这是 high-precision/tiny-crop diagnostic，不满足 union gate。
3. 当使用同一预测但沿用 v4.1 support universe 评估时，AP 下降到 0.0628，说明高 AP 主要来自 pre_points 裁剪，而不是完整场景 object formation 成功。
4. 这个实验暴露出一个重要报告风险：任何 recompute pre_points 的高 AP 都必须同时报告 pre/union，否则会误判。
```

### 更新后的 Gate 判断

Probe5 v6 method gate：

| 项 | Gate | 本轮最好可报告结果 | 结论 |
|---|---|---:|---|
| AP | `>=0.32` | localprop coreown2 `0.13568501383679288`; v4.1 core tiny-crop `0.6062679879644165` 不合格 | fail |
| AP50 | `>=0.53` | localprop coreown2 `0.27218506615709775`; v4.1 core tiny-crop `0.7970712323390894` 不合格 | fail |
| AP25 | `>=0.70` | localprop core-only `0.5254354960868468`; v4.1 core tiny-crop `0.906634288330717` 不合格 | fail |
| #pred | `<=300` | localprop coreown2 `98.0` | pass |
| union in target | `>=0.94` | localprop coreown2 `21.9675%`; v4.1 core tiny-crop `0.9256%` | fail |
| manifest / oracle audit | pass required | phase0_pass=true, uses_gt=false | pass |

结论：

```text
v6 仍未达成目标。
本轮有一个真实小改善：localprop coreown2_ratio020 把 AP 从 0.1125 提到 0.1357，并保持 #pred<=300。
但 AP/AP50/AP25/union 都离 gate 很远。
v4.1 core-only 的高 AP 是 tiny-pre_points 裁剪效应，不能作为 method success。
```

### 当前最诚实判断

```text
1. 继续推进是必要的；本轮确实按 Phase D 尝试了 core/fringe/reject 与 WTA。
2. 已确认 core seed 有价值，但“删除冲突点”会损失太多 support。
3. 已确认低分 fringe 和点级 WTA 当前不可靠。
4. 已确认高 AP 必须和 pre/union 一起审计，否则会产生误导。
5. 下一步若继续，应实现 boundary growth / split-aware object expansion，而不是继续调 score、top-k、NMS 或直接扩大 fringe。
```

不能写：

```text
不能写 v6 已达成 ScanNet probe5 gate。
不能写 stream4d_v6_v41self_probe5_coreonly 是成功 method。
不能用 tiny-pre_points 高 AP 启动 full ScanNet final。
不能 claim D4RT-native geometry 或 Dynamic Replica tracking 已验证。
```

## 继续推进：geometry-aware radius growth

### 代码修改审计

```text
Stream3D/tools/split_core_fringe_prediction.py:
  新增 geometry-aware radius growth。
  新增参数：
    --growth-mode none/radius
    --growth-candidate-mode support/full
    --growth-radius
    --growth-max-owners
  summary/manifest 增加 growth_candidate_points、growth_kept_points、growth_added_points 和 growth 参数。
```

合理性：

```text
Phase D 计划要求在 core seed 后尝试 connected fringe / boundary growth。
该实现只用 prediction、support pre_points 和 ScanNet mesh 几何，不读 GT；
因此是 method-side postprocess，不是 oracle。
```

验证：

```text
py_compile exit=0
unit tests: Ran 12 tests in 1.433s, OK
```

### scene0050 结果

| Config | seed | growth | added points | AP | AP50 | AP25 |
|---|---|---|---:|---:|---:|---:|
| `coreown2_rgrow_support003_g2` | owner<=2 ratio>=0.20 | support r=0.03 max_owner=2 | `0` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` |
| `coreown2_rgrow_support005_g2` | owner<=2 ratio>=0.20 | support r=0.05 max_owner=2 | `0` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` |
| `coreown2_rgrow_full003_g2` | owner<=2 ratio>=0.20 | full r=0.03 max_owner=2 | `0` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` |
| `coreown2_rgrow_full005_g2` | owner<=2 ratio>=0.20 | full r=0.05 max_owner=2 | `0` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` |
| `core1_rgrow_support003_g2` | owner<=1 ratio>=0.05 | support r=0.03 max_owner=2 | `7204` | `0.17589366369854176` | `0.33217189314750295` | `0.5296167247386759` |
| `core1_rgrow_support005_g0` | owner<=1 ratio>=0.05 | support r=0.05 no owner cap | `19016` | `0.1395385980479148` | `0.35758651286601595` | `0.49778172138420584` |
| `core1_rgrow_full003_g2` | owner<=1 ratio>=0.05 | full r=0.03 max_owner=2 | `7204` | `0.17589366369854176` | `0.33217189314750295` | `0.5296167247386759` |
| `core1_rgrow_full005_g0` | owner<=1 ratio>=0.05 | full r=0.05 no owner cap | `19016` | `0.1395385980479148` | `0.35758651286601595` | `0.49778172138420584` |

分析：

```text
1. owner<=2 seed + owner<=2 growth 是 no-op，added_points=0。
2. pure core seed 的 radius growth 确实增加 support 点，但 AP/AP50/AP25 都没有超过 scene0050 core-only 或 coreown2_ratio020。
3. r=0.05/no owner cap 增加了 19016 点，但 AP 下降到 0.1395，说明补点主要带来边界污染。
4. 该分支没有推到 probe5。
```

Insight：

```text
当前问题不是“core 附近没有点”，而是 object candidate 的身份和边界本身不稳定。
简单半径增长不能区分 true fringe 与邻近污染；需要更强的 object-level 竞争/候选生成，而不是继续扩大几何半径。
```

## 继续推进：Phase E object competition + small-object rescue

### 代码修改审计

```text
Stream3D/tools/object_competition_rank.py:
  1. 新增 prediction/TMP manifest，输出 uses_gt=false。
  2. 新增 small rescue：
       --small-rescue-reserve
       --small-rescue-min-support-area
       --small-rescue-max-support-area
       --small-rescue-min-novel-points
       --small-rescue-overlap-threshold
       --small-rescue-overlap-mode
  3. summary 新增：
       num_selected_base
       num_selected_small_rescue
       small_rescue_rejected_area/novelty/overlap
```

合理性：

```text
v4.1 current best 的主要 gate blocker 是 #pred=415.6。
Phase E 计划要求在压重复时启用 small-object rescue，避免误删 rare/small object。
实现中 small rescue 使用 max_instances 内的预留名额，不允许超过 #pred gate；
且只使用 prediction/support，不读 GT。
```

验证：

```text
py_compile exit=0
unit tests: Ran 12 tests in 1.643s, OK
```

### scene0050 结果

v4.1 scene0050 one-scene view 基线：

```text
AP/AP50/AP25 = 0.28717796840958604 / 0.47205882352941175 / 0.7758333333333333
```

small rescue 矩阵：

| Config | AP | AP50 | AP25 | selected | small rescue | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e4_scene0050_objcomp_m300_g085_scoreunique_preserve` | `0.21159477124183007` | `0.3223529411764706` | `0.546` | `183` | `0` | pure competition 明显低于 v4.1 |
| `stream4d_v6_e4_scene0050_objcomp_m300_g085_sr40_a1500_n80_preserve` | `0.23919455169455173` | `0.400965250965251` | `0.6830357142857142` | `186` | `3` | rescue 有正信号但仍低于 v4.1 |
| `stream4d_v6_e4_scene0050_objcomp_m300_g085_sr80_a2500_n80_preserve` | `0.23919455169455173` | `0.400965250965251` | `0.6830357142857142` | `186` | `3` | 与 sr40 相同 |
| `stream4d_v6_e4_scene0050_objcomp_m300_g085_sr80_a2500_n80_areaunique_preserve` | `0.23524904214559386` | `0.37931034482758624` | `0.7293103448275862` | `185` | `2` | AP25 较高，但 AP/AP50 更低 |

分析：

```text
1. small rescue 比 pure competition 明显更好，说明 Phase E 的 rescue 方向有必要。
2. 但 scene0050 仍低于原 v4.1 view，说明 competition 仍误删有效 recall。
3. area_unique 能提高 AP25 到 0.729，但 AP/AP50 更差，属于 coarse coverage 增益而非 instance-quality 增益。
```

### probe5 结果

推到 probe5 的最平衡配置：

```text
stream4d_v6_e4_probe5_objcomp_m300_g085_sr40_a1500_n80_preserve
quality = score_unique_compact
group_overlap = min_ioc@0.85
max_instances = 300
small_rescue_reserve = 40
small_rescue_max_support_area = 1500
small_rescue_min_novel_points = 80
```

结果：

| Config | AP | AP50 | AP25 | #pred | pre % | union % |
|---|---:|---:|---:|---:|---:|---:|
| v4.1 current best | `0.2816154378895367` | `0.4975830336133305` | `0.6902541954477854` | `415.6` | `4.5145%` | `4.3406%` |
| E4 small rescue | `0.26253062027752616` | `0.47629926485603435` | `0.6499047035600027` | `173.0` | `4.5145%` | `3.5371%` |

metric integrity：

```text
phase0_pass = true
manifest_exists = true
uses_gt = false
num_configs_missing_manifest = 0
num_uses_gt_and_method_result = 0
mean_num_pred_instances = 173.0
mean_pre_points_ratio = 0.04514451433782776
mean_prediction_union_ratio = 0.03537121456885783
```

判断：

```text
Phase E small rescue 没有过 gate：
AP   0.2625 < 0.32
AP50 0.4763 < 0.53
AP25 0.6499 < 0.70
union 3.5371% << 94%

#pred 从 415.6 降到 173.0，但 AP/AP50/AP25 同时下降。
这正符合计划中的失败模式：competition 误删 recall。
```

Insight：

```text
重复 prediction 是症状，但不是主因。
当前 v4.1 的高 AP25/AP50 依赖大量候选保留；只做 object-level suppression 会降低候选 recall。
small rescue 能救一点，但每个场景平均只救回 3 个候选，远不足以修复 object formation。
下一步不能继续盲调 NMS/competition；需要 split/merge-capable object memory 或更强的候选生成，使“正确 object”先被形成，再做 competition。
```

## 更新后的最终 Gate 判断

| 项 | Gate | 当前最好可报告结果 | 结论 |
|---|---|---:|---|
| AP | `>=0.32` | v4.1 current best `0.2816154378895367`; E4 `0.26253062027752616` | fail |
| AP50 | `>=0.53` | v4.1 current best `0.4975830336133305`; E4 `0.47629926485603435` | fail |
| AP25 | `>=0.70` | v4.1 current best `0.6902541954477854`; E4 `0.6499047035600027` | fail |
| #pred | `<=300` | E4 `173.0` | pass |
| union in target | `>=0.94` | E4 `3.5371%`; v4.1 `4.3406%` | fail |
| manifest/oracle audit | pass required | E4 phase0_pass=true, uses_gt=false | pass |

最终判断：

```text
目标仍未达成。
本轮继续按计划尝试了 Phase D geometry-aware boundary growth 和 Phase E small-object rescue。
两者都有真实审计证据，但都没有通过 probe5 gate。
不启动 full ScanNet final。
```

不能写：

```text
不能写 v6 已解决 object formation。
不能写 E4 small rescue 成功，只能写它在 scene0050 相对 pure competition 有正信号。
不能写 #pred pass 就等于 method success，因为 AP/AP50/AP25/union 均失败。
```

## v6 continuation：no-group/top-k 压缩版 v4.1

### 为什么继续

用户追问“达成目标了吗，没有请继续”。此前 E4 small-rescue 的 probe5 结果：

```text
stream4d_v6_e4_probe5_objcomp_m300_g085_sr40_a1500_n80_preserve
AP/AP50/AP25 = 0.26253062027752616 / 0.47629926485603435 / 0.6499047035600027
#pred = 173.0
```

它满足 #pred 但 AP/AP50/AP25 下降，说明 g085 object competition 过度压缩，误删 recall。scene0050 上继续放宽 grouping 后，g098 AP25 恢复到 v4.1 scene0050 水平，因此继续推到 probe5，并进一步做 no-group/top-k 反事实。

执行说明：

```text
这些实验都基于已有 prediction/TMP 做 CPU postprocess、evaluator 和 metric integrity。
没有重新跑 D4RT/cache/model forward，因此没有 GPU 显存调用。
GPU 4/5/6/7 应保留给后续需要模型推理或 cache generation 的实验，而不是用于 CPU 后处理。
```

### scene0050 g095/g098 探针

v4.1 scene0050 one-scene view 基线：

```text
AP/AP50/AP25 = 0.28717796840958604 / 0.47205882352941175 / 0.7758333333333333
```

结果：

| Config | AP | AP50 | AP25 | selected | small rescue | output_union |
|---|---:|---:|---:|---:|---:|---:|
| `stream4d_v6_e4_scene0050_objcomp_m300_g095_scoreunique_preserve` | `0.2448429198429199` | `0.36320166320166325` | `0.5913461538461539` | `207` | `0` | `8974` |
| `stream4d_v6_e4_scene0050_objcomp_m300_g098_scoreunique_preserve` | `0.26998737373737375` | `0.43738636363636363` | `0.7758333333333333` | `216` | `0` | `10606` |
| `stream4d_v6_e4_scene0050_objcomp_m300_g095_sr40_a1500_n80_preserve` | `0.2503586691086691` | `0.40054945054945057` | `0.6830357142857142` | `209` | `2` | `9806` |
| `stream4d_v6_e4_scene0050_objcomp_m300_g098_sr40_a1500_n80_preserve` | `0.26998737373737375` | `0.43738636363636363` | `0.7758333333333333` | `216` | `0` | `10606` |

分析：

```text
1. g098 明显优于 g085/g095，说明 earlier competition 的 grouping 是主要损伤源之一。
2. g098 的 AP25 已追平 v4.1 scene0050 view，但 AP/AP50 仍低于 v4.1。
3. g098 下 small rescue 没有新增实例，说明当前 small-rescue 规则不是主要瓶颈。
```

### probe5 g098 与 no-group/top-k

g098 推到 probe5：

```text
stream4d_v6_e4_probe5_objcomp_m300_g098_scoreunique_preserve
AP/AP50/AP25 = 0.26605892239144985 / 0.4726395846667555 / 0.6724933696462123
#pred = 190.6
union = 3.6121%
metric integrity phase0_pass = true
```

由于 g098 仍将 #pred 压到 190.6，继续设置 `group_overlap_threshold=1.01`，让候选基本不被 group 合并，只做 top-k 选择。

probe5 结果：

| Config | AP | AP50 | AP25 | #pred | union % | 结论 |
|---|---:|---:|---:|---:|---:|---|
| v4.1 current best | `0.2816154378895367` | `0.4975830336133305` | `0.6902541954477854` | `415.6` | `4.3406%` | reference |
| `m300_g101_scoreunique` | `0.2785629906546174` | `0.49121769019013883` | `0.6894759991330806` | `219.0` | `4.0756%` | 接近 v4.1，#pred pass |
| `m500_g101_scoreunique` | `0.28063532510475836` | `0.4957469672224599` | `0.6881410431792294` | `265.6` | not_audited | 更接近 v4.1 |
| `m650_g101_scoreunique` | `0.28204965102921625` | `0.4982312805326885` | `0.6912642582412274` | `295.6` | `4.2507%` | 三项小幅超过 v4.1，#pred pass |
| `m650_g101_compact_only` | `0.28483247256897415` | `0.5039622641509434` | `0.6719147248897401` | `295.6` | `4.1578%` | 当前最高 AP/AP50，但 AP25 退化 |
| `m670_g101_score_unique_compact` | `0.28203394183547537` | `0.49820855659464575` | `0.691223277348379` | `299.6` | not_audited | 与 m650 基本相同 |
| `m670_g101_compact_only` | `0.28382830955380606` | `0.5019341996506697` | `0.6832167422177096` | `299.6` | not_audited | 不如 m650 compact |

metric integrity：

```text
stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve:
  phase0_pass = true
  uses_gt = false
  mean_num_pred_instances = 295.6
  mean_prediction_union_ratio = 0.042507400047896333

stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve:
  phase0_pass = true
  uses_gt = false
  mean_num_pred_instances = 295.6
  mean_prediction_union_ratio = 0.0415780150820649
```

### 当前最好结果判断

按 AP/AP50：

```text
best = stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve
AP/AP50/AP25 = 0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401
#pred = 295.6
```

按三项均衡：

```text
best_balanced = stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve
AP/AP50/AP25 = 0.28204965102921625 / 0.4982312805326885 / 0.6912642582412274
#pred = 295.6
```

与 v4.1 current best 比较：

```text
v4.1 = 0.2816154378895367 / 0.4975830336133305 / 0.6902541954477854, #pred=415.6
m650_g101_scoreunique = 0.28204965102921625 / 0.4982312805326885 / 0.6912642582412274, #pred=295.6
```

因此可以写：

```text
no-group/top-k m650 scoreunique 在 probe5 上以 #pred<=300 小幅超过 v4.1 current best 的 AP/AP50/AP25。
这是一个真实的压缩版 v4.1 正信号。
```

但必须同时写：

```text
它仍没有达到 v6 method gate：
AP   0.2820 < 0.32
AP50 0.4982 < 0.53
AP25 0.6913 < 0.70
union 4.25% << 94%

m650 compact_only AP/AP50 更高，但 AP25 只有 0.6719，也没有过 gate。
```

Insight：

```text
1. v6 的主要失败不再是“是否能把 #pred 压到 300 以下”；no-group/top-k 已做到 #pred=295.6 且不明显掉分。
2. 真正离 gate 的差距来自候选本身的 object quality / support coverage：union 仍只有约 4.2%，远低于计划中的 94% target。
3. grouping/NMS 的过度合并会伤 recall；但完全取消 grouping 后，AP 也只恢复到 v4.1 附近，不能产生 v6 所需的跃迁。
4. 下一步继续调 ranking/top-k 的边际收益已经很小；如果继续，应进入计划中更核心的 object formation：split-aware graph、component-level ownership、core/fringe/reject lifecycle，或重新生成更高质量候选。
```

更新后的 gate：

| 项 | Gate | 当前最好结果 | 结论 |
|---|---|---:|---|
| AP | `>=0.32` | `0.28483247256897415` | fail |
| AP50 | `>=0.53` | `0.5039622641509434` | fail |
| AP25 | `>=0.70` | balanced `0.6912642582412274`; compact `0.6719147248897401` | fail |
| #pred | `<=300` | `295.6` | pass |
| union in target | `>=0.94` | `4.2507%` / `4.1578%` | fail |
| metric integrity | pass | phase0_pass=true, uses_gt=false | pass |

最终判断：

```text
目标仍未达成。
本轮新增了一个真实正信号：在 #pred<=300 下小幅超过 v4.1 current best。
但该信号不足以启动 full ScanNet final，也不能 claim Stream4D v6 已超过 Stream3D diagnostic。
```

## v6 continuation：ownership-aware WTA 反事实

### 为什么继续

no-group/top-k 说明过度 grouping 会伤 recall，但它仍没有达到 v6 gate。为了继续按计划中的 ownership-aware 方向推进，本轮利用已有 `split_core_fringe_prediction.py` 的 WTA assignment 做最小反事实：

```text
把 conflict support point 分配给单一 owner，是否能在降低冲突的同时提升 AP/AP50？
```

执行边界：

```text
仍然是 CPU 后处理 + evaluator + metric integrity。
没有重新跑 D4RT/cache/model forward，因此没有 GPU 调用。
```

### Probe5 结果

| Config | AP | AP50 | AP25 | #pred | union % | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e5_probe5_v41_wta_score_over_sqrt_area_core005_preserve` | `0.2504209406842788` | `0.4830729675575163` | `0.6671145539377294` | not_audited | not_audited | not_run |
| `stream4d_v6_e5_probe5_v41_wta_score_core005_preserve` | `0.2713964926079926` | `0.48931285917848677` | `0.6812376191185117` | `103.8` | `4.1271%` | pass |
| `stream4d_v6_e5_probe5_v41_wta_large_area_core005_preserve` | `0.1888086582238025` | `0.36780109986986276` | `0.5223900140738225` | not_audited | not_audited | not_run |

best WTA summary:

```text
mean_num_instances_before = 415.6
mean_num_instances_after = 103.8
mean_output_support_conflict_ratio = 0.0
mean_mean_core_ratio = 0.46751355660911537
mean_mean_conflict_ratio = 0.5324864433908847
metric integrity phase0_pass = true
uses_gt = false
```

分析：

```text
1. WTA 达到了“消除 output support conflict”的局部目标，但指标下降。
2. best WTA 只有 103.8 个预测，远低于 no-group/top-k 的 295.6，说明它不是精细 split，而是过度削减 support。
3. 这解释了为什么 AP/AP50/AP25 都低于 m650 no-group/top-k：owner assignment 本身没有生成更好的 object，只是把已有 support 切碎/删弱。
4. 因此 ownership-aware 方向仍有价值，但必须结合 component proposal / split-merge lifecycle；不能只靠点级 WTA 后处理。
```

当前路线收束：

```text
已尝试并真实记录：
1. score-mode 修复与重跑：失败。
2. minimal typed graph：失败。
3. core/fringe/radius growth：失败。
4. object competition + small rescue：有 scene0050 小正信号，probe5 不过。
5. no-group/top-k：#pred<=300 下小幅超过 v4.1，但不过 v6 gate。
6. WTA ownership：消除冲突但指标下降。

因此，继续微调现有后处理参数的边际收益已经很小。
如果继续，需要新的 method micro-plan：更高质量候选生成、component-level split、split/merge-capable object memory，或重新跑需要 GPU 的 D4RT/cache/model path。
```

## 审计包更新

最新打包产物：

```text
stream4d_v6_code_review_packet_20260608_2136.zip
stream4d_v6_code_review_packet_20260608_2136.sha256
stream4d_v6_code_review_packet_20260608_2136_filelist.txt
```

打包前验证：

```text
py_compile exit=0
import smoke OK
unit tests: Ran 12 tests in 1.389s, OK
```

包内包含：

```text
1. 最新代码：
   Stream3D/tools/object_competition_rank.py
   Stream3D/tools/split_core_fringe_prediction.py
   Stream3D/tools/export_typed_evidence_graph_v3.py
   manifest / metric integrity 工具
   Stream3D/stream4d/*.py
   evaluation 和 tests

2. 文档：
   docs/stream4d_v6_method_first_audit_and_experiment_plan_for_codex.md
   docs/stream4d_v6_执行日志.md
   docs/stream4d_v6_实验结果复盘.md

3. 证据：
   radius-growth、E4 small rescue、g098/g101/m650/m670 no-group/top-k 的 evaluation txt
   core_fringe / object_competition summary json
   E4 metric integrity md/json
   关键命令日志
   git_status_r3.txt / git_diff_r3.patch / filelist.txt
```

说明：

```text
不包含 raw ScanNet 数据、prediction npz 或 checkpoint。
最终 sha256 以同名 .sha256 文件和最终回复为准；不在包内文档中写死，避免 package self-hash 循环。
```

### 审计包

本轮结束后已重新打包，避免审计包落后于代码：

```text
stream4d_v6_code_review_packet_20260608_2102.zip
stream4d_v6_code_review_packet_20260608_2102.sha256
stream4d_v6_code_review_packet_20260608_2102_filelist.txt
sha256 = 5b212f17e79c1b095cff0b2fe12c16d00f7888ab209ef65a7aee335e4de32726
```

打包前验证：

```text
py_compile pass
import smoke OK
unit tests: Ran 12 tests in 1.423s, OK
```

包内包含：

```text
代码：Stream3D/stream4d/*.py、Stream3D/tools/*.py、evaluation、tests/fixtures。
文档：v6 plan、执行日志、实验结果复盘。
证据：关键 evaluation txt、core_fringe summary json、metric integrity md/json、命令日志。
审计：git_status.txt、git_diff.patch、filelist.txt。
```

## continuation：pure Stream4D candidate fusion 与替换式融合

### 为什么继续

用户要求“达成目标了吗，没有请继续”。截至上一轮，当前最好仍是：

```text
stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve:
  AP/AP50/AP25 = 0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401

stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve:
  AP/AP50/AP25 = 0.28204965102921625 / 0.4982312805326885 / 0.6912642582412274
```

仍未达到 v6 gate：

```text
AP >= 0.32 false
AP50 >= 0.53 false
AP25 >= 0.70 false
union target >= 0.94 false
```

因此继续做一个有限范围的 pure Stream4D 反事实：

```text
1. v4.1 high-score candidates + v6 local proposal candidates 是否互补？
2. local/core proposal 能否作为 primary object 的替换形状，而不是新增大量 object？
3. compact-only 和 scoreunique 两个已有正信号能否互补？
```

这些实验均不使用 GT，也不使用 Stream3D/scannet support 作为 selection signal。GPU 未调用，因为本节全是 prediction postprocess/evaluator。

### 新增修复审计

文件：

```text
Stream3D/tools/fuse_prediction_configs.py
```

问题：

```text
select_secondary 分支中 --preserve-primary-score 没有真正保留 primary 原始分数；
它会把 primary score 写成 -1.0。
```

修改：

```text
_select_variant_masks 增加 primary_scores 参数，并统一使用 _score_array_or_preserve。
empty-primary/empty-secondary case、primary selected_scores、unmatched secondary scores 均支持 preserve。
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/fuse_prediction_configs.py
exit=0
```

合理性：

```text
该修改只修复 select_secondary 的 score preservation 语义。
默认 concatenate 行为不变；已完成历史结果不受影响。
```

### E6 concat fusion 结果

融合摘要：

| Fused config | primary instances | secondary instances | output instances | primary union | secondary union | output union |
|---|---:|---:|---:|---:|---:|---:|
| `stream4d_v6_e6_probe5_fuse_v41_localprop_concat_p1_s005` | `415.6` | `173.2` | `588.8` | `4.3406%` | `35.7973%` | `36.2635%` |
| `stream4d_v6_e6_probe5_fuse_v41_coreonly_concat_p1_s005` | `415.6` | `76.0` | `491.6` | `4.3406%` | `13.1310%` | `16.4209%` |

直接用 fused union 作为 score support：

| Config | AP | AP50 | AP25 | #pred | union % | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `E6 localprop scoreunique` | `0.010218108812631676` | `0.04115012746934571` | `0.17617853353466817` | `287.8` | `25.5904%` | pass |
| `E6 localprop compact` | `0.011710341031375226` | `0.044383533492114685` | `0.18216000373994912` | `287.8` | `25.6962%` | pass |
| `E6 coreonly scoreunique` | `0.0405954551412587` | `0.1204511962249542` | `0.33605775521707504` | `256.4` | `16.0945%` | pass |
| `E6 coreonly compact` | `0.0377836588858184` | `0.11856998329568297` | `0.3440586154109523` | `256.4` | `14.7493%` | pass |

保守 primary-support 版本：

| Config | AP | AP50 | AP25 | #pred | union % | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `E6 localprop scoreunique primarysupp` | `0.27979015467712104` | `0.48746989600281426` | `0.6968854618036644` | `263.2` | `13.8754%` | pass |
| `E6 localprop compact primarysupp` | `0.22856043273657833` | `0.3864847517795402` | `0.5071203008901843` | `263.2` | `16.4226%` | pass |
| `E6 coreonly scoreunique primarysupp` | `0.27801811145138666` | `0.4910369268480345` | `0.6889750798092922` | `238.2` | `7.9644%` | pass |
| `E6 coreonly compact primarysupp` | `0.2754073604133401` | `0.45256450717681596` | `0.5913662748158626` | `238.2` | `7.2392%` | pass |

分析：

```text
1. localprop secondary union 达到 35.8%，远大于 primary 的 4.34%；直接作为 support 会让 ranking 特征被噪声主导，AP 崩到 0.01 量级。
2. primary-support 版本能恢复到接近 v4.1，但 metric integrity 显示多数 scene 的 union policy 为 inconsistent_union_not_subset，因此不能作为 clean final policy。
3. 即使只看指标，primary-support 版本也没有超过当前 best m650 compact/scoreunique。
```

### E7 select_secondary 替换式融合结果

目的：

```text
保留 v4.1 object identity，只允许 localprop/coreonly 作为 matched primary 的替换形状。
不直接把 secondary 当新 object 堆入输出。
```

结果：

| Config | AP | AP50 | AP25 | #pred | output union | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `E7 localprop scoreunique` | `0.27360533351987215` | `0.48736235779772474` | `0.6857581778604103` | `295.6` | `9084.2` | pass |
| `E7 localprop compact` | `0.2762186735105981` | `0.4928908044100521` | `0.6665564862285539` | `295.6` | `8919.6` | pass |
| `E7 coreonly scoreunique` | `0.27348684186262195` | `0.4982312805326885` | `0.6912642582412274` | `295.6` | `9051.2` | pass |
| `E7 coreonly compact` | `0.2770380429028228` | `0.5058237524138866` | `0.6746102478165856` | `295.6` | `8869.6` | pass |

分析：

```text
1. select_secondary 没有解决问题；四条均低于当前 best compact-only AP=0.284832。
2. coreonly compact 的 AP50=0.505824 接近 compact-only best 0.503962，但 AP 和 AP25 都更低。
3. 替换式融合说明 local/core proposal 作为 primary shape variant 没有稳定改善边界。
```

### E8 dual-quality compact/scoreunique fusion

目的：

```text
compact-only 提高 AP/AP50，scoreunique 提高 AP25。
尝试把 scoreunique 中与 compact 不重复的候选作为补充。
```

fusion 摘要：

```text
mean_primary_instances = 295.6
mean_secondary_instances after duplicate drop = 13.6
mean_output_instances = 309.2
mean_secondary_skipped_by_iou = 282.0
mean_output_union_ratio = 0.0426531859248895
```

结果：

| Config | AP | AP50 | AP25 | #pred | output union | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `E8 dualquality scoreunique` | `0.27031150512181795` | `0.4332201554423777` | `0.5934316489872046` | `178.0` | `7016.6` | pass |
| `E8 dualquality compact` | `0.1993965865242845` | `0.3115810996763378` | `0.43117816927340735` | `178.0` | `6420.0` | pass |

分析：

```text
1. compact/scoreunique 两个正信号没有互补成功。
2. duplicate drop 后只剩 13.6 个 secondary candidates，但随后的 0.85 grouping 把 #pred 压到 178，明显误删 recall。
3. 这说明“把两个后处理结果再融合”不是有效 object formation。
```

### 当前结论更新

本轮 continuation 后，目标仍未达成：

| Gate | 当前最好 | 结论 |
|---|---:|---|
| AP >= 0.32 | `0.28483247256897415` | fail |
| AP50 >= 0.53 | `0.5058237524138866` in E7, but AP/AP25 worse; best balanced AP50 still `0.5039622641509434` | fail |
| AP25 >= 0.70 | `0.6968854618036644` in E6 primary-support, but policy not clean and AP/AP50 worse; clean best `0.6912642582412274` | fail |
| #pred <= 300 | many candidates pass | pass numerically |
| union target >= 0.94 | best still far below | fail |

Insight：

```text
1. v6 当前不是缺候选，甚至 localprop 候选 union 很大；问题是 secondary 候选的 boundary purity / ownership 太差。
2. 把 localprop/coreonly 当作补充候选会污染 ranking；当作 primary 替换形状也没有稳定收益。
3. 现有后处理组合已经无法跨 gate。继续有意义的方向应是计划中的 split-aware component proposal / object-internal core-fringe-reject / split-merge memory，而不是继续 fusion/top-k/NMS。
4. 如果要动 GPU，应回到 D4RT/cache/model path；当前这些 prediction postprocess 实验本身不需要 GPU，所以看不到显存调用是预期现象。
```

证据链：

```text
E6 concat / primary-support:
  Stream3D/outputs/fusion_v6/stream4d_v6_e6_probe5_fuse_v41_localprop_concat_p1_s005_summary.json
  Stream3D/outputs/fusion_v6/stream4d_v6_e6_probe5_fuse_v41_coreonly_concat_p1_s005_summary.json
  Stream3D/outputs/audit/v6_metric_integrity_stream4d_v6_e6_fusion_probe5.md
  Stream3D/outputs/audit/v6_metric_integrity_stream4d_v6_e6_fusion_primarysupp_probe5.md

E7 select_secondary:
  Stream3D/outputs/fusion_v6/stream4d_v6_e7_probe5_select_v41_localprop_ioc050_exp150_summary.json
  Stream3D/outputs/fusion_v6/stream4d_v6_e7_probe5_select_v41_coreonly_ioc050_exp150_summary.json
  Stream3D/outputs/audit/v6_metric_integrity_stream4d_v6_e7_select_probe5.md

E8 dual-quality:
  Stream3D/outputs/fusion_v6/stream4d_v6_e8_probe5_dualquality_compact_primary_scoreunique_secondary_drop098_summary.json
  Stream3D/outputs/audit/v6_metric_integrity_stream4d_v6_e8_dualquality_probe5.md

命令日志：
  Stream3D/logs/stream4d_v6_e6_*_*.log
  Stream3D/logs/stream4d_v6_e7_*_*.log
  Stream3D/logs/stream4d_v6_e8_*_*.log
```

## continuation r4 审计包

打包前验证：

```text
py_compile exit=0
import smoke OK
unit tests: Ran 12 tests in 1.396s, OK
```

验证日志：

```text
Stream3D/logs/stream4d_v6_package_py_compile_r4.log
Stream3D/logs/stream4d_v6_package_import_smoke_r4.log
Stream3D/logs/stream4d_v6_package_unit_tests_r4.log
```

审计包：

```text
stream4d_v6_code_review_packet_20260608_2207.zip
stream4d_v6_code_review_packet_20260608_2207.sha256
stream4d_v6_code_review_packet_20260608_2207_filelist.txt
filelist entries = 492
sha256 = 4705daf7f30c75e8a815cf7efa13865438b9f5d9cb7386a5bfee9e5bb2aa3099
```

说明：

```text
该包包含最新 fuse_prediction_configs.py 修复、E6/E7/E8 命令日志、evaluation txt、summary json、metric integrity 报告、v6 执行日志和复盘。
不包含 raw ScanNet 数据、checkpoint 或大型 prediction npz。
由于记录本节本身会再次修改文档，最终审计包名和 sha256 以最终回复及同名 .sha256 文件为准。
```
