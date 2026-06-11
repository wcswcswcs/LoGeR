# Stream4D v7 深度审计、差距确认与方法改进实验复盘

日期：2026-06-08 至 2026-06-09（Asia/Singapore）  
执行日志：`docs/stream4d_v7_执行日志.md`  
计划文件：`docs/stream4d_v7_deep_audit_gap_method_plan_for_codex.md`  
主要结果目录：`Stream3D/outputs/audit/`、`Stream3D/outputs/v7_carrier_tracklet_graph/`、`Stream3D/data/evaluation/scannet/`

本复盘只记录真实运行得到的数据。`nan` 表示 evaluator 结果文件实际落盘为 `nan`；没有指标的地方不补写。

## 当前状态

```text
Phase A gap matrix 完成。
Phase B D4RT geometry degradation 完成，并按 blocker 方向尝试修复。
Phase C carrier-tracklet C2/C3/C4 probe5 完成。
Phase D Dynamic Replica 环境检查完成，但本地无数据，不能报告官方指标。
Metric integrity phase0_pass=True。
最终 py_compile/import smoke/unit test pass；pytest 入口因环境无 pytest 失败，已用 unittest 重跑通过。
C11/C12/C13 track-bucket duplicate suppression 已完成，未取得有效提升。
C14 复现 C8P，预测文件 sha256 完全一致，不是新性能。
C15 strict posttrack merge 已完成并审计通过，但指标低于 C8P/C14。
v7 当前证据链显示差距没有实质缩小，需要进入 v8 新实验计划，而不是继续在当前阈值/后处理线上扫参。
```

## 修改审计

| 文件 | 修改 | 审计理由 |
|---|---|---|
| `Stream3D/stream4d/reliable_densifier.py` | WTA 后重算 `area_score/score/reliability/dense_quality/selection_quality` | 修复 WTA 改变 `point_ids` 后 score stale 的 P1 问题 |
| `Stream3D/tests/test_stream4d_protocol_fixes.py` | 增加 WTA score 重算测试 | 防止同类 bug 回归 |
| `Stream3D/evaluation/evaluate.py` | torch optional + numpy CPU fallback；AP core 未改 | 解除当前环境无 torch 的测试 blocker；metric integrity 证明 AP core hash 一致 |
| `Stream3D/tools/evaluate_cross_prepoints.py` | cross-support diagnostic manifest 与 evaluator guard | gap matrix 不进入 method table，必须可审计 |
| `Stream3D/tools/make_union_prepoints_config.py` | union support S6 | Phase A 必需 |
| `Stream3D/tools/diagnose_prediction_quality.py` | prediction quality diagnostic | 支撑差距分析 |
| `Stream3D/tools/export_d4rt_geometry_degradation_v7.py` | D4RT geometry degradation / Sim3 residual / G1/G3 export | Phase B 必需 |
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | carrier co-membership object formation C2/C3/C4 | Phase C 主线 |
| `Stream3D/tools/summarize_v7_gap_matrix.py` | 汇总 gap matrix | 生成统一证据表 |
| `Stream3D/tools/diagnose_trackbucket_suppression_v7.py` | 新增 GT-only suppression diagnostic，显式 `diagnostic_only=True, uses_gt=True` | 判断 C11/C12/C13 删除候选是否伤害真阳性；不作为 method result |
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | object_dict 写入 processed 目录失败时 fallback 到 `outputs/v7_carrier_tracklet_graph/object_dicts/...`，summary 记录 fallback path/error | 处理当前 sandbox 下 processed object 目录只读 blocker，保留 object_dict 审计材料 |
| `Stream3D/tools/verify_stream4d_metric_integrity.py` | alignment 查找支持 summary fallback object_dict 路径 | 修复 C14/C15 因 fallback 路径导致 alignment skipped 的审计噪声；r6 审计 alignment mean/min=1.0/1.0 |

## 验证与完整性

```text
py_compile final: pass
import smoke final: pass
unit tests final: unittest 13 tests pass
metric integrity phase0_pass=True
evaluator AP core equal by hash=True
rescore_gt_read_check.gt_files_read_by_rescore=False
reportable config scan:
  num_configs=3
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=3
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

注意：

```text
logs/stream4d_v7_unit_tests_final.log 是 pytest 入口失败，原因是 conda 环境没有 pytest。
有效最终单测日志是 logs/stream4d_v7_unit_tests_final_unittest.log。
```

## Phase A：gap matrix 关键结果

完整表：

```text
Stream3D/outputs/audit/v7_gap_matrix.{json,csv,md,png}
```

关键 cell：

| Cell | Prediction | Support | AP | AP50 | AP25 | target pre_points ratio |
|---|---|---|---:|---:|---:|---:|
| `p0_s0` | `P0_scannet` | `S0_own` | `0.23572958766757215` | `0.41430630093420795` | `0.5377857284834029` | `0.8467441995295546` |
| `p0_s1` | `P0_scannet` | `S1_32f` | `0.3992127932017927` | `0.5971712938711367` | `0.7425353588266108` | `0.04514451433782776` |
| `p1_s1` | `P1_32f_self` | `S1_32f` | `0.14423832897728478` | `0.28834385159686365` | `0.46471600688468157` | `0.04514451433782776` |
| `p2_s2` | `P2_v41_best` | `S2_v41` | `0.2816154378895367` | `0.4975830336133305` | `0.6902541954477854` | `0.04514451433782776` |
| `p3_s3` | `P3_v6_compact` | `S3_v6compact` | `0.28483247256897415` | `0.5039622641509434` | `0.6719147248897401` | `0.04514451433782776` |
| `p4_s4` | `P4_v6_scoreunique` | `S4_v6scoreunique` | `0.2820496510292162` | `0.4982312805326885` | `0.6912642582412274` | `0.04514451433782776` |
| `p5_s0` | `P5_v6_localprop` | `S0_own` | `0.11252410871496846` | `0.2853900969507906` | `0.4797727369815028` | `0.35797330322212584` |
| `p6_s0` | `P6_v6_typedv3` | `S0_own` | `0.03347846361047295` | `0.12849540654750258` | `0.3977530181719252` | `0.2417333067446915` |
| `p6_s1` | `P6_v6_typedv3` | `S1_32f` | `0.05614973324754232` | `0.15686703876511696` | `0.29328637508974403` | `0.04514451433782776` |
| `p6_s5` | `P6_v6_typedv3` | `S5_scannet` | `0.006373642124203465` | `0.02518243184133715` | `0.19137308037590808` | `0.8467441995295546` |

每个 prediction 的 best support：

| Prediction | Own AP/AP50/AP25 | Best cell | Best AP/AP50/AP25 |
|---|---|---|---|
| `P0_scannet` | `0.23572958766757215 / 0.41430630093420795 / 0.5377857284834029` | `p0_s1` | `0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108` |
| `P1_32f_self` | `0.14423832897728478 / 0.28834385159686365 / 0.46471600688468157` | `p1_s0` | `0.14423832897728478 / 0.28834385159686365 / 0.46471600688468157` |
| `P2_v41_best` | `0.2816154378895367 / 0.4975830336133305 / 0.6902541954477854` | `p2_s0` | `0.2816154378895367 / 0.4975830336133305 / 0.6902541954477854` |
| `P3_v6_compact` | `0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401` | `p3_s0` | `0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401` |
| `P4_v6_scoreunique` | `0.2820496510292162 / 0.4982312805326885 / 0.6912642582412274` | `p4_s0` | `0.2820496510292162 / 0.4982312805326885 / 0.6912642582412274` |
| `P5_v6_localprop` | `0.11252410871496846 / 0.2853900969507906 / 0.4797727369815028` | `p5_s1` | `0.20140876139125907 / 0.37935125115848006 / 0.5310212813872311` |
| `P6_v6_typedv3` | `0.03347846361047295 / 0.12849540654750258 / 0.3977530181719252` | `p6_s1` | `0.05614973324754232 / 0.15686703876511696 / 0.29328637508974403` |

判断：

```text
Best overall = P0_scannet on S1_32f
AP/AP50/AP25 = 0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108

Best Stream4D-like v6 object result remains P3/P4 around AP 0.282-0.285.
Gap to Stream3D-on-32f-support is about 0.114-0.117 AP and 0.093-0.099 AP50.
这不是 evaluator 或 support 名字造成的小差异，而是 object quality 差距。
```

## Phase A：prediction quality 证据

来源：

```text
Stream3D/outputs/audit/v7_prediction_quality_diagnostic.json
```

关键 aggregate：

| Config | pred union ratio | per-GT best IoU mean | best IoU >=0.25 | best IoU >=0.50 | missed GT IoU<0.25 | support conflict rate | tiny mask ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scannet` | `0.8467441995295546` | `0.5462737589068828` | `0.8350185008396803` | `0.6307326471879421` | `9.2` | `0.0022130316719709987` | `0.32300259193734576` |
| `stream4d_32f_self_probe5` | `0.04514451433782776` | `0.051127850185773815` | `0.020098322729901676` | `0.007142857142857143` | `40.0` | `0.20515044669111138` | `0.9001249995541526` |
| `P3_v6_compact` | `0.0415780150820649` | `0.05814536588629949` | `0.020098322729901676` | `0.007142857142857143` | `40.0` | `0.6664408020292352` | `0.7788380688380688` |
| `P4_v6_scoreunique` | `0.042507400047896333` | `0.058034844069406455` | `0.020098322729901676` | `0.007142857142857143` | `40.0` | `0.6348266097606893` | `0.7843765303765303` |
| `P5_v6_localprop` | `0.35797330322212584` | `0.24917957992888787` | `0.4733974649320242` | `0.1372253618290397` | `22.0` | `0.6162319170120751` | `0.0` |
| `P6_v6_typedv3` | `0.2417333067446915` | `0.2246585908757892` | `0.41360351759844466` | `0.1593089535702091` | `24.0` | `0.2648217091330786` | `0.0` |

解释：

```text
Stream4D sparse high-score variants在小 support 上 AP 可以较高，但 per-GT best IoU 极低，missed GT 接近 40/40.6。
Localprop/typedv3 提高 coverage 后 AP 仍低，说明不是单纯 support 太小，而是 object ownership / split-merge 质量差。
```

## Phase B：D4RT geometry degradation

Sim3 residual summary：

```text
num_windows=25
num_ok_windows=25
num_failed_windows=0
sim3_anchor_count_mean=446.08
sim3_anchor_count_min/max=199.0 / 681.0
sim3_scale_mean=0.588463045245313
sim3_scale_min/max=0.424304693041852 / 0.8916739660680099
sim3_residual_median_mean=0.6801382969694636
sim3_residual_median_min/max=0.37772401787093146 / 1.0206364344867356
sim3_residual_p90_mean=1.2819213888794445
sim3_residual_p95_mean=1.5032774278399892
uv_in01_rate_mean=0.6064773520758063
visibility_mean_mean=0.9988281679153442
confidence_mean_mean=0.9998375511169434
```

Segmentation/eval：

| Config | 修复尝试 | AP | AP50 | AP25 | 结论 |
|---|---|---:|---:|---:|---|
| `stream4d_v7_g1_d4rt_geometry_probe5` | default | `nan` | `nan` | `nan` | 0 objects |
| `stream4d_v7_g3_d4rt_geometry_norm_probe5` | default | `nan` | `nan` | `nan` | 0 objects |
| `stream4d_v7_g1_d4rt_geometry_probe5_r050` | mesh NN radius 0.50 | `nan` | `nan` | `nan` | 仍 0 objects |
| `stream4d_v7_g3_d4rt_geometry_norm_probe5_r050` | mesh NN radius 0.50 | `nan` | `nan` | `nan` | 仍 0 objects |
| `stream4d_v7_g1_d4rt_geometry_probe5_r050_min1` | radius 0.50 + min_points 1 | `0.0` | `0.0` | `0.0` | 变成大量碎片，仍无 AP |

Blocker 处理结论：

```text
已按计划方向尝试 radius 放宽和 min_points 极限 diagnostic。
结果不是“没跑通就放弃”，而是显示 D4RT shared-reference geometry 在当前缓存/映射下无法直接替代 ScanNet RGB-D/pose geometry。
高 Sim3 residual 与 G1/G3 的 0 objects / 0 AP 一致。
```

## Phase C：carrier-tracklet object formation

方法定义：

```text
C2 = carrier core only
C3 = carrier core + selected mask fringe
C4 = C3 + WTA ownership
所有 C2/C3/C4 summary 均写 uses_mask_node_point_overlap=False。
所有 reportable C configs manifest uses_gt=False, is_method_result=True。
```

Probe5 AP：

| Config | AP | AP50 | AP25 | evaluator |
|---|---:|---:|---:|---|
| `stream4d_v7_c2_carrier_core_probe5` | `nan` | `nan` | `nan` | `data/evaluation/scannet/stream4d_v7_c2_carrier_core_probe5_class_agnostic.txt` |
| `stream4d_v7_c3_carrier_corefringe_probe5` | `0.00898913041811256` | `0.037895752433249094` | `0.1537215682032027` | `data/evaluation/scannet/stream4d_v7_c3_carrier_corefringe_probe5_class_agnostic.txt` |
| `stream4d_v7_c4_carrier_corefringe_wta_probe5` | `0.020512013069687883` | `0.07965152929245886` | `0.43874809492877437` | `data/evaluation/scannet/stream4d_v7_c4_carrier_corefringe_wta_probe5_class_agnostic.txt` |

Probe5 formation summary：

| Config | exported objects sum/mean | exported points sum/mean | conflict mean | mask backproject queries sum | selected masks sum |
|---|---:|---:|---:|---:|---:|
| `C2` | `2 / 0.4` | `182 / 36.4` | `0.0` | `0` | `56` |
| `C3` | `1902 / 380.4` | `584600 / 116920.0` | `0.9697554565162196` | `450635654` | `23334` |
| `C4` | `340 / 68.0` | `576236 / 115247.2` | `0.0` | `450635654` | `23334` |

C4 WTA：

```text
pre_conflict_rate_mean=0.9697554565162196
removed_assignment_rate_mean=0.9051216051506223
export_conflict_rate_mean=0.0
```

Per-scene C4：

| Scene | candidate records | exported objects | exported points | pre-conflict rate | removed assignment rate |
|---|---:|---:|---:|---:|---:|
| `scene0011_00` | `235` | `57` | `86663` | `0.9448025969869465` | `0.9196465720334508` |
| `scene0030_00` | `471` | `76` | `151564` | `0.9777457457979768` | `0.9013181692379978` |
| `scene0050_00` | `322` | `56` | `77799` | `0.9801986445726113` | `0.8785825927060987` |
| `scene0081_01` | `296` | `69` | `138796` | `0.9547291662211578` | `0.9016910052479956` |
| `scene0591_00` | `578` | `82` | `121414` | `0.9913011290024061` | `0.9243696865275688` |

判断：

```text
C2 core-only 太稀疏，5 个场景只导出 2 个对象，eval 为 nan。
C3 能形成大量对象，但冲突率约 0.97，AP 极低。
C4 WTA 能把冲突率清零，AP25 从 0.1537 提升到 0.4387，但 AP/AP50 仍只有 0.0205/0.0797。
因此 carrier-tracklet 方向有可测信号，但当前 object split/merge/ownership 仍远未解决。
```

## Phase D：Dynamic Replica

```text
data_root_exists=False
split_dir_exists=False
annotation_exists=False
scene_count=0
usable_scene_count=0
can_report_official_instance_tracking=False
can_report_d4rt_trajectory_metrics=False
can_report_only_qualitative_consistency=False
```

结论：

```text
本地没有 Dynamic Replica v2 valid 数据。
不能报告 IDF1、official instance tracking、D4RT trajectory metrics 或 qualitative consistency。
```

## Gate 判断

```text
Stream3D same-support diagnostic:
  P0_scannet on S1_32f = 0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108

Best v7 new carrier method:
  C4 = 0.020512013069687883 / 0.07965152929245886 / 0.43874809492877437

Best prior Stream4D-like v6 object result in matrix:
  P3_v6_compact = 0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401
  P4_v6_scoreunique = 0.2820496510292162 / 0.4982312805326885 / 0.6912642582412274

success=False
```

不能 claim：

```text
不能 claim v7 carrier-tracklet method 超过 Stream3D。
不能 claim D4RT geometry replacement 可直接替代 RGB-D/pose geometry。
不能 claim Dynamic Replica 官方结果。
```

可以 claim：

```text
gap matrix 已定量确认差距。
D4RT geometry direct replacement 在当前实现/缓存上严重失败。
carrier-tracklet graph 使用 D4RT identity 而非 mask-node point overlap，且 WTA 有明确冲突修复效果。
但 C4 仍不足以缩小到 Stream3D/object-quality baseline。
```

## 分析

```text
1. gap matrix 证明当前最大差距不是 evaluator，而是 object formation。
   P0_scannet 放到 S1_32f support 能达到 AP 0.3992，而 Stream4D best object variants 约 0.282-0.285。

2. prediction quality diagnostic 说明 sparse support 的高 AP 有 tiny-support diagnostic 风险。
   Stream4D sparse variants missed GT 接近 40/40.6，per-GT best IoU mean 约 0.05-0.06。

3. D4RT geometry replacement 的失败和 Sim3 residual 一致。
   median residual mean 0.68m、p90 mean 1.28m；默认 G1/G3 0 objects，极限 min_points=1 也只有 AP 0。

4. carrier-tracklet C3/C4 证明 D4RT identity 信号不是完全无效。
   C4 相比 C3：AP 0.009 -> 0.0205，AP25 0.1537 -> 0.4387，冲突率 0.97 -> 0。
   但 WTA 同时移除了约 90.5% assignments，object split/merge 不可靠。

5. 当前 C4 AP50 只有 0.0797，说明 coarse overlap 可能改善了 AP25，但精确实例边界和 one-to-one object quality 仍失败。
```

## 结论

```text
v7 完成了计划要求的深度审计和三条关键实验：
1. 同 support gap matrix。
2. D4RT geometry degradation。
3. D4RT carrier-tracklet object formation。

结果是明确负结论：
D4RT geometry 直接替代 ScanNet RGB-D/pose 不可行；
carrier-tracklet + WTA 有正向信号但远未达到 Stream3D 或 v6 best；
Dynamic Replica 因数据缺失不能验证。
```

## Insight

```text
“用 D4RT carrier identity 做 object backbone”比 v6 的 mask-node overlap 更接近正确方向，
但当前 edge/component 仍太粗：C3 过度扩张，C4 再用 WTA 大量剪除，等于先污染再强行排他。

下一步不应继续调 WTA 分数或 top-k，而应在 object formation 阶段提前加入：
1. carrier trajectory consistency 的 split criterion；
2. per-frame mask exclusivity before component merge；
3. object-level temporal birth/death，而不是 window 内一次性 component；
4. 用 RGB-D geometry 做静态 dense support，D4RT carrier 只负责 identity/track consistency。

换句话说，D4RT 当前更适合作为 identity backbone，不适合作为 ScanNet 静态 dense geometry backbone。
```

## 证据链索引

```text
Phase A:
  Stream3D/outputs/audit/v7_gap_matrix.json
  Stream3D/outputs/audit/v7_prediction_quality_diagnostic.json
  Stream3D/logs/stream4d_v7_gap_matrix.log

Phase B:
  Stream3D/outputs/audit/v7_d4rt_geometry_degradation.json
  Stream3D/logs/stream4d_v7_d4rt_geometry_residual.log
  Stream3D/logs/stream4d_v7_d4rt_geometry_segmentation*.log
  Stream3D/data/evaluation/scannet/stream4d_v7_g*_class_agnostic.txt

Phase C:
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c*_probe5_scene*_summary.json
  Stream3D/data/evaluation/scannet/stream4d_v7_c*_probe5_class_agnostic.txt
  Stream3D/logs/stream4d_v7_carrier_*.log

Integrity:
  Stream3D/outputs/audit/v7_metric_integrity_carrier_probe5.json
  Stream3D/outputs/audit/v7_reportable_config_scan_carrier_probe5.json
  Stream3D/logs/stream4d_v7_unit_tests_final_unittest.log
  Stream3D/logs/stream4d_v7_import_smoke_final.log
  Stream3D/logs/stream4d_v7_py_compile_final.log

Dynamic Replica:
  Stream3D/outputs/audit/dynamic_replica_env_v7.json
```

## 追加实验 C5/C6（2026-06-09）

追加目标：

```text
v7 初版 C4 未达标。
按上文 Insight 继续尝试提前约束 object formation：
1. carrier trajectory consistency split criterion；
2. per-frame mask purity / exclusivity before component merge；
3. 尽量避免 C3/C4 “先过扩张再靠 WTA 强剪”的模式。

仍保持约束：
uses_gt=False
uses_mask_node_point_overlap=False
evaluation 使用 --require-manifest
```

## 追加修改审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | 新增 `seeded_fringe` / `seeded_fringe_wta` support mode；新增 seeded mask 参数；seeded mode 从 carrier core seeds 出发，只在 seed-connected / seed-near RGB-D mask 区域补支持点 | 尝试在 component merge 之前限制 fringe 扩张，避免 full-mask fringe 污染 | `logs/stream4d_v7_c5_py_compile.log` pass；`logs/stream4d_v7_c5_import_smoke.log` pass |
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | `core_fringe_wta` 支持更严格轨迹与 mask purity 参数组合：`min_shared_frames=3`、`min_positive_ratio=0.65`、`max_pair_distance_variance=0.005`、`max_component_carriers=180`、`min_mask_carriers=6`、`min_frame_mask_ratio=0.65` | 测试提前 trajectory-consistency split 是否优于 C4 默认 merge | C6 full probe5 完成；metric integrity `phase0_pass=True`；reportable scan clean；13 unittest pass |

## C5/C6 scene0050 smoke

| Run | support mode | 关键参数 | AP | AP50 | AP25 | objects | points | WTA pre-conflict | WTA removed assignment | selected masks | mask backproject queries |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 scene0050 ref | `core_fringe_wta` | C4 默认 | `0.05774536692856975` | `0.2280865778908927` | `0.6270272194960086` | `56` | `77799` | `0.9801986445726113` | `0.8785825927060987` | `3433` | `69877900` |
| C5 | `seeded_fringe_wta` | seeded 默认 | `0.023971830447578644` | `0.16721706226350191` | `0.4777910127368332` | `32` | `16791` | `0.5481265039532486` | `0.40254672417334153` | `655` | `2797563` |
| C5W | `seeded_fringe_wta` | `seeded_max_masks=12, distance=64, selection=coverage` | `0.03023042499035112` | `0.13771709976696125` | `0.3725548278647442` | `32` | `31147` | `0.6413786555260159` | `0.4778901447551061` | `661` | `8931588` |
| C6 smoke | `core_fringe_wta` | strict-track | `0.06702122663545979` | `0.275824858166817` | `0.6505328341003658` | `54` | `77767` | `0.9753710135775182` | `0.8565431195054248` | `3043` | `61881851` |

判断：

```text
C5/C5W 证明 seeded fringe 能显著降低 backprojection 和 WTA 冲突，但 AP/AP50/AP25 均低于 C4 scene0050。
这不是有效修复。

C6 strict-track 在 scene0050 上优于 C4：
AP  +0.009275859706890037
AP50 +0.047738280275924305
AP25 +0.023505614604357183

因此 C6 值得扩到 full probe5。
```

## C6 full probe5 结果

| Run | AP | AP50 | AP25 | vs C4 AP | vs C4 AP50 | vs C4 AP25 | h/scene |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3 `carrier_corefringe` | `0.00898913041811256` | `0.037895752433249094` | `0.1537215682032027` |  |  |  | 5/5 |
| C4 `carrier_corefringe_wta` | `0.020512013069687883` | `0.07965152929245886` | `0.43874809492877437` | ref | ref | ref | 5/5 |
| C6 `strict_track_wta` | `0.02655669594520481` | `0.11179541279756663` | `0.46564072814373947` | `+0.006044682875516927` | `+0.03214388350510777` | `+0.0268926332149651` | 5/5 |

与 v6/object baseline 对比：

```text
v6 P3 compact:
  0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401

v6 P4 scoreunique:
  0.2820496510292162 / 0.4982312805326885 / 0.6912642582412274

C6 strict_track_wta:
  0.02655669594520481 / 0.11179541279756663 / 0.46564072814373947

结论：
  C6 相比 C4 有小幅修复，但仍远低于 v6/object-component 路线。
```

## C4 vs C6 per-scene 证据

| Scene | C4 AP | C6 AP | delta AP | C4 AP50 | C6 AP50 | delta AP50 | C4 AP25 | C6 AP25 | delta AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene0050_00` | `0.05774536692856975` | `0.06702122663545979` | `+0.009275859706890037` | `0.2280865778908927` | `0.275824858166817` | `+0.047738280275924305` | `0.6270272194960086` | `0.6505328341003658` | `+0.023505614604357183` |
| `scene0011_00` | `0.012751145797862292` | `0.015016324783933273` | `+0.0022651789860709814` | `0.054561214339796826` | `0.09398114555263243` | `+0.0394199312128356` | `0.41261434858595114` | `0.40339257315593413` | `-0.00922177543001701` |
| `scene0030_00` | `0.0224098070200366` | `0.03279020032913967` | `+0.010380393309103071` | `0.07113548816712265` | `0.11267285949799695` | `+0.041537371330874304` | `0.4585943909769171` | `0.4528696935428941` | `-0.0057246974340230405` |
| `scene0081_01` | `0.027453545958821487` | `0.03313173979579032` | `+0.005678193836968836` | `0.11967213455825426` | `0.1468326900289278` | `+0.02716055547067353` | `0.4554026421918954` | `0.521475109206907` | `+0.06607246701501157` |
| `scene0591_00` | `0.012805626620859071` | `0.015579294522583899` | `+0.0027736679017248272` | `0.0443114718947924` | `0.046731525862118636` | `+0.0024200539673262383` | `0.41765338151298226` | `0.46214841494214837` | `+0.04449503342916611` |

解释：

```text
C6 在 5/5 场景 AP 都高于 C4，但只有 scene0050 的绝对 AP 达到 0.067。
其他四个场景 AP 仍只有 0.015 到 0.033。
所以 C6 是一致的小幅改善，不是根因解决。
```

## C4 vs C6 object formation 证据

| Scene | C4 objects | C6 objects | C4 points | C6 points | C4 pre-conflict | C6 pre-conflict | C4 removed assign | C6 removed assign | C4 selected masks | C6 selected masks | C4 traj rejects | C6 traj rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene0050_00` | `56` | `54` | `77799` | `77767` | `0.9801986445726113` | `0.9753710135775182` | `0.8785825927060987` | `0.8565431195054248` | `3433` | `3043` | `16` | `220` |
| `scene0011_00` | `57` | `47` | `86663` | `85591` | `0.9448025969869465` | `0.9234076985319616` | `0.9196465720334508` | `0.8836815313001691` | `3370` | `2354` | `1404` | `1766` |
| `scene0030_00` | `76` | `75` | `151564` | `151404` | `0.9777457457979768` | `0.9763442406494712` | `0.9013181692379978` | `0.8814043628583389` | `6795` | `5800` | `402` | `1568` |
| `scene0081_01` | `69` | `62` | `138796` | `137975` | `0.9547291662211578` | `0.9266816207977454` | `0.9016910052479956` | `0.8666066953672171` | `2628` | `2015` | `233` | `585` |
| `scene0591_00` | `82` | `77` | `121414` | `121337` | `0.9913011290024061` | `0.9860479787767514` | `0.9243696865275688` | `0.8934721790220201` | `7108` | `5377` | `441` | `1087` |

证据解释：

```text
1. C6 selected masks 比 C4 少，trajectory rejects 比 C4 多。
   说明 strict-track 参数确实提前过滤了一部分不稳定 merge。

2. C6 pre-conflict 与 removed assignment rate 均比 C4 略低。
   说明它降低了一点 WTA 负担。

3. 但 C6 仍保留 0.923 到 0.986 的 WTA pre-conflict rate，
   removed assignment 仍是 0.856 到 0.893。
   这说明 full-mask fringe 仍然高度重叠，仍在依赖 WTA 大量剪除。

4. AP25 有小幅提升，但 AP/AP50 仍低，说明粗 overlap 稍有改善，
   精确 object boundary 和 one-to-one instance formation 仍失败。
```

## C6 审计结果

```text
py_compile:
  logs/stream4d_v7_c6_py_compile.log
  pass

import smoke:
  logs/stream4d_v7_c6_import_smoke.log
  tools.export_carrier_tracklet_graph_v7 OK
  stream4d.reliable_densifier OK

unit test:
  logs/stream4d_v7_c6_unit_tests.log
  Ran 13 tests in 0.088s
  OK

metric integrity:
  outputs/audit/v7_metric_integrity_c6_strict_track_wta_probe5.md
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  alignment mean/min=1.000000/1.000000

reportable scan:
  outputs/audit/v7_reportable_config_scan_c6_strict_track_wta_probe5.json
  num_configs=1
  num_configs_missing_manifest=0
  num_diagnostic_only_configs=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

## 追加结论

```text
C5 seeded fringe：
  工程上减少了 backprojection 和冲突，但损失太多 dense support，scene0050 AP 明显低于 C4。
  结论：不能作为有效修复。

C6 strict-track WTA：
  相比 C4 在 full probe5 上有一致小幅改善：
    AP  +0.006044682875516927
    AP50 +0.03214388350510777
    AP25 +0.0268926332149651
  但绝对指标仍为：
    0.02655669594520481 / 0.11179541279756663 / 0.46564072814373947
  远低于 v6/object-component baseline。

success=False
```

不能 claim：

```text
不能 claim C5/C6 解决了 v7 object formation gap。
不能 claim strict-track WTA 已接近 Stream3D 或 v6 object-component baseline。
不能 claim D4RT carrier-tracklet 可以替代 RGB-D dense support。
```

可以 claim：

```text
strict-track merge 约束方向是正向的，小幅降低 WTA 冲突和 assignment 删除率，并稳定提高 AP/AP50。
但只靠更严格 edge/component 参数无法消除 full-mask fringe 的高重叠本质。
```

## 追加 Insight

```text
C6 的小幅收益说明“更早的 trajectory-consistency 约束”是对的，
但 residual high pre-conflict rate 说明瓶颈不是阈值，而是 object support 表示。

full-mask fringe 把同一片静态 geometry 同时分给多个 carrier component，
WTA 只能在最后选择赢家，无法恢复被错误 merge 或错误 dense support 污染的实例边界。

C5 反方向证明：只保留 seed-near support 又会过度稀疏，AP/AP50/AP25 下降。
因此下一步不是简单放宽/收紧 fringe，而应改成：
1. object-level temporal birth/death；
2. per-frame one-to-one mask assignment before support expansion；
3. RGB-D dense support 作为 geometry 主体；
4. D4RT carrier 只作为 identity / track consistency backbone；
5. support expansion 需要 object-level ownership model，而不是 component 后处理 WTA。
```

## 追加证据链索引

```text
C5/C5W smoke:
  Stream3D/logs/stream4d_v7_c5_seeded_fringe_wta_scene0050_smoke.log
  Stream3D/logs/stream4d_v7_c5_seeded_fringe_wta_scene0050_eval.log
  Stream3D/logs/stream4d_v7_c5w_seeded_fringe_wta_scene0050_smoke.log
  Stream3D/logs/stream4d_v7_c5w_seeded_fringe_wta_scene0050_eval.log
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c5*_scene0050_00_summary.json
  Stream3D/data/evaluation/scannet/stream4d_v7_c5*_scene0050_class_agnostic.txt

C6 full:
  Stream3D/logs/stream4d_v7_c6_strict_track_wta_probe5.log
  Stream3D/logs/stream4d_v7_c6_strict_track_wta_probe5_eval.log
  Stream3D/logs/stream4d_v7_c6_strict_track_wta_probe5_per_scene_eval.log
  Stream3D/logs/stream4d_v7_c4_carrier_corefringe_wta_probe5_per_scene_eval.log
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c6_strict_track_wta_probe5_scene*_summary.json
  Stream3D/data/evaluation/scannet/stream4d_v7_c6_strict_track_wta_probe5_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c6_strict_track_wta_probe5_scene*_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c4_carrier_corefringe_wta_probe5_scene*_class_agnostic.txt

C6 integrity:
  Stream3D/logs/stream4d_v7_c6_py_compile.log
  Stream3D/logs/stream4d_v7_c6_import_smoke.log
  Stream3D/logs/stream4d_v7_c6_unit_tests.log
  Stream3D/logs/stream4d_v7_metric_integrity_c6_strict_track_wta_probe5.log
  Stream3D/logs/stream4d_v7_reportable_config_scan_c6_strict_track_wta_probe5.log
  Stream3D/outputs/audit/v7_metric_integrity_c6_strict_track_wta_probe5.json
  Stream3D/outputs/audit/v7_reportable_config_scan_c6_strict_track_wta_probe5.json
```

## 追加实验 C7：pre-expansion mask ownership（2026-06-09）

目标：

```text
C6 证明 strict trajectory split 有小幅收益，但 full-mask fringe 仍高度重叠。
C7 直接测试：
  在 support expansion 前做 per-frame mask ownership，
  每个 (frame_id, mask_id) 只能由一个 carrier component 使用。
```

## C7 修改审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | 新增 `core_owned_fringe` / `core_owned_fringe_wta` support mode | 在 full-mask backprojection 前对 mask 做 component ownership，减少同一 RGB-D mask 被多个 object 同时扩张 | `logs/stream4d_v7_c7_py_compile_final.log` pass；`logs/stream4d_v7_c7_import_smoke_final.log` pass |
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | 新增 `_build_mask_ownership`，ownership score = `count * ratio * sqrt(component_size) * sqrt(frame_count)`；落盘 ownership diagnostics | 记录竞争 mask、保留/丢弃 claim 数，方便审计 C7 是否真的做了 pre-expansion exclusivity | C7 summaries 写出 `support_ownership_*` 字段；metric integrity `phase0_pass=True` |

## C7 scene0050 smoke

| Run | AP | AP50 | AP25 | objects | points | WTA pre-conflict | WTA removed assignment | queries | selected masks | ownership competing | ownership dropped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C6 scene0050 | `0.06702122663545979` | `0.275824858166817` | `0.6505328341003658` | `54` | `77767` | `0.9753710135775182` | `0.8565431195054248` | `61881851` | `3043` | n/a | n/a |
| C7 scene0050 | `0.07021919799297623` | `0.3289020603973451` | `0.6961639578901516` | `57` | `78332` | `0.9235454683313956` | `0.7008287209267157` | `34756620` | `1765` | `789` | `1191` |

判断：

```text
C7 scene0050 明显正向：
  AP  +0.003197971357516441
  AP50 +0.05307720223052809
  AP25 +0.045631123789785843

同时 C7 降低 backprojection 和 WTA 负担：
  queries: 61881851 -> 34756620
  WTA removed assignment: 0.8565431195054248 -> 0.7008287209267157

因此扩到 full probe5。
```

## C7 full probe5 结果

| Run | AP | AP50 | AP25 | vs C6 AP | vs C6 AP50 | vs C6 AP25 | vs C4 AP | vs C4 AP50 | vs C4 AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 `carrier_corefringe_wta` | `0.020512013069687883` | `0.07965152929245886` | `0.43874809492877437` |  |  |  | ref | ref | ref |
| C6 `strict_track_wta` | `0.02655669594520481` | `0.11179541279756663` | `0.46564072814373947` | ref | ref | ref | `+0.006044682875516927` | `+0.03214388350510777` | `+0.0268926332149651` |
| C7 `owned_fringe_wta` | `0.03130332812699374` | `0.12217319256494784` | `0.49161195683671977` | `+0.004746632181788932` | `+0.010377779767381204` | `+0.025971228692980297` | `+0.010791315057305857` | `+0.04252166327248898` | `+0.0528638619079454` |

与 v6/object baseline 对比：

```text
C7 owned_fringe_wta:
  0.03130332812699374 / 0.12217319256494784 / 0.49161195683671977

v6 P3 compact:
  0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401

v6 P4 scoreunique:
  0.2820496510292162 / 0.4982312805326885 / 0.6912642582412274

结论：
  C7 是当前 carrier-tracklet 分支里最好的非-oracle 结果，
  但 AP/AP50 仍远低于 v6/object baseline。
```

## C6 vs C7 per-scene 证据

| Scene | C6 AP | C7 AP | delta AP | C6 AP50 | C7 AP50 | delta AP50 | C6 AP25 | C7 AP25 | delta AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene0050_00` | `0.06702122663545979` | `0.07021919799297623` | `+0.003197971357516441` | `0.275824858166817` | `0.3289020603973451` | `+0.05307720223052809` | `0.6505328341003658` | `0.6961639578901516` | `+0.045631123789785843` |
| `scene0011_00` | `0.015016324783933273` | `0.025578546718713922` | `+0.010562221934780649` | `0.09398114555263243` | `0.11378856777089513` | `+0.019807422218262705` | `0.40339257315593413` | `0.4448661332149625` | `+0.04147356005902836` |
| `scene0030_00` | `0.03279020032913967` | `0.035497197949020995` | `+0.0027069976198813225` | `0.11267285949799695` | `0.10290309568828235` | `-0.009769763809714602` | `0.4528696935428941` | `0.4682466411265928` | `+0.01537694758369873` |
| `scene0081_01` | `0.03313173979579032` | `0.04320145377213323` | `+0.01006971397634291` | `0.1468326900289278` | `0.16198228101801132` | `+0.015149590989083528` | `0.521475109206907` | `0.5006290517386844` | `-0.020846057468222545` |
| `scene0591_00` | `0.015579294522583899` | `0.019932370202365532` | `+0.004353075679781633` | `0.046731525862118636` | `0.05150672636396844` | `+0.004775200501849805` | `0.46214841494214837` | `0.48633603132728864` | `+0.024187616385140276` |

解释：

```text
C7 在 5/5 场景 AP 都高于 C6。
AP50 在 4/5 场景高于 C6；scene0030_00 AP50 下降。
AP25 在 4/5 场景高于 C6；scene0081_01 AP25 下降。

整体 full probe5 仍提升，说明 ownership 方向比 C6 更稳。
但四个低 AP 场景仍低，尤其 scene0591_00 只有 AP 0.0199。
```

## C6 vs C7 object formation 证据

| Scene | C6 objects | C7 objects | C6 points | C7 points | C6 pre-conflict | C7 pre-conflict | C6 removed assign | C7 removed assign | C6 queries | C7 queries | C6 selected masks | C7 selected masks | C7 competing | C7 dropped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene0050_00` | `54` | `57` | `77767` | `78332` | `0.9753710135775182` | `0.9235454683313956` | `0.8565431195054248` | `0.7008287209267157` | `61881851` | `34756620` | `3043` | `1765` | `789` | `1191` |
| `scene0011_00` | `47` | `50` | `85591` | `85806` | `0.9234076985319616` | `0.8092617853096895` | `0.8836815313001691` | `0.7364180941760057` | `62535600` | `30217320` | `2354` | `1245` | `650` | `1063` |
| `scene0030_00` | `75` | `75` | `151404` | `151506` | `0.9763442406494712` | `0.7763415973638789` | `0.8814043628583389` | `0.6803464209507767` | `83084503` | `35804833` | `5800` | `2677` | `1687` | `3068` |
| `scene0081_01` | `62` | `65` | `137975` | `138315` | `0.9266816207977454` | `0.841160737396106` | `0.8666066953672171` | `0.6855508579955235` | `54117495` | `27367747` | `2015` | `1212` | `497` | `781` |
| `scene0591_00` | `77` | `84` | `121337` | `121832` | `0.9860479787767514` | `0.9043322760227861` | `0.8934721790220201` | `0.731174328610148` | `80273004` | `35913325` | `5377` | `2806` | `1490` | `2474` |

证据解释：

```text
1. C7 在所有场景都显著降低 WTA pre-conflict 和 removed assignment rate。
   说明 pre-expansion mask ownership 确实减少了 object 间共享污染。

2. C7 queries 也大幅降低：
   例如 scene0030_00 从 83084503 降到 35804833。
   说明它不是用更多回投换指标，而是更早裁剪竞争 mask。

3. C7 object/points 数没有大幅塌缩，和 C6 相近或略多。
   所以 C7 优于 C5：不是靠过度稀疏化换低冲突。

4. 但 C7 residual pre-conflict 仍有 0.776 到 0.923；
   full AP 只有 0.0313，说明 ownership 只能缓解 full-mask fringe 污染，
   不能解决 object boundary 和 temporal identity 的根本缺口。
```

## C7 审计结果

```text
py_compile:
  logs/stream4d_v7_c7_py_compile_final.log
  pass

import smoke:
  logs/stream4d_v7_c7_import_smoke_final.log
  tools.export_carrier_tracklet_graph_v7 OK
  stream4d.reliable_densifier OK

unit test:
  logs/stream4d_v7_c7_unit_tests.log
  Ran 13 tests in 0.095s
  OK

metric integrity:
  outputs/audit/v7_metric_integrity_c7_owned_fringe_wta_probe5.md
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  alignment mean/min=1.000000/1.000000

reportable scan:
  outputs/audit/v7_reportable_config_scan_c7_owned_fringe_wta_probe5.json
  num_configs=1
  num_configs_missing_manifest=0
  num_diagnostic_only_configs=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

## C7 追加结论

```text
C7 是正向修复：
  相比 C6：
    AP  +0.004746632181788932
    AP50 +0.010377779767381204
    AP25 +0.025971228692980297
  相比 C4：
    AP  +0.010791315057305857
    AP50 +0.04252166327248898
    AP25 +0.0528638619079454

但 C7 仍未达标：
  C7 full = 0.03130332812699374 / 0.12217319256494784 / 0.49161195683671977
  v6 P3  = 0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401
  v6 P4  = 0.2820496510292162 / 0.4982312805326885 / 0.6912642582412274

success=False
```

## C7 Insight

```text
pre-expansion exclusivity 是当前最有证据支持的修复方向：
它同时提高 AP/AP50/AP25，降低 WTA 冲突，降低 backprojection 成本。

但它仍是在 component/window 内做 mask ownership，
没有真正建模 object-level temporal birth/death。
低 AP 场景没有被根本抬起，说明错误不只来自同帧 mask 竞争，
还来自跨窗口 object identity 和 dense support ownership 的长期一致性缺失。

下一步更可能有效的是：
1. 将 ownership 从 window 内提升到 scene-level temporal object track；
2. 同一 mask 的 owner 不只看当前 window component score，还看跨窗口 track consistency；
3. 使用 RGB-D dense support 建静态几何主体，D4RT carrier 只提供 identity linking；
4. 在 support expansion 前做 object-track-level one-to-one assignment，而不是 window component-level。
```

## C7 证据链索引

```text
C7 code / logs:
  Stream3D/tools/export_carrier_tracklet_graph_v7.py
  Stream3D/logs/stream4d_v7_c7_py_compile.log
  Stream3D/logs/stream4d_v7_c7_import_smoke.log
  Stream3D/logs/stream4d_v7_c7_owned_fringe_wta_scene0050_smoke.log
  Stream3D/logs/stream4d_v7_c7_owned_fringe_wta_scene0050_eval.log
  Stream3D/logs/stream4d_v7_c7_owned_fringe_wta_probe5.log
  Stream3D/logs/stream4d_v7_c7_owned_fringe_wta_probe5_eval.log
  Stream3D/logs/stream4d_v7_c7_owned_fringe_wta_probe5_per_scene_eval.log

C7 results:
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c7_owned_fringe_wta_scene0050_scene0050_00_summary.json
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c7_owned_fringe_wta_probe5_scene*_summary.json
  Stream3D/data/evaluation/scannet/stream4d_v7_c7_owned_fringe_wta_scene0050_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c7_owned_fringe_wta_probe5_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c7_owned_fringe_wta_probe5_scene*_class_agnostic.txt

C7 integrity:
  Stream3D/logs/stream4d_v7_c7_py_compile_final.log
  Stream3D/logs/stream4d_v7_c7_import_smoke_final.log
  Stream3D/logs/stream4d_v7_c7_unit_tests.log
  Stream3D/logs/stream4d_v7_metric_integrity_c7_owned_fringe_wta_probe5.log
  Stream3D/logs/stream4d_v7_reportable_config_scan_c7_owned_fringe_wta_probe5.log
  Stream3D/outputs/audit/v7_metric_integrity_c7_owned_fringe_wta_probe5.json
  Stream3D/outputs/audit/v7_reportable_config_scan_c7_owned_fringe_wta_probe5.json
```

## 追加实验 C8：scene-level carrier track linking（2026-06-09）

目标：

```text
C7 的 window-level pre-expansion ownership 有效，但没有跨 window object track。
C8 测试 D4RT carrier_id 是否可用于 scene-level temporal linking。

设计了三种 smoke：
1. C8 pre-WTA track merge default；
2. C8S pre-WTA track merge strict；
3. C8P post-WTA track merge default。
```

## C8 修改审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | 新增 `core_owned_track_fringe` / `core_owned_track_fringe_wta` / `core_owned_fringe_wta_posttrack` | 测试跨 window carrier-overlap scene track linking | `logs/stream4d_v7_c8p_py_compile_final.log` pass；`logs/stream4d_v7_c8p_import_smoke_final.log` pass |
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | 新增 `_merge_scene_track_records`，按 shared carrier IDs + overlap ratio + window gap + 同帧 mask 冲突约束合并 records | 将 C7 的 window object 提升为 scene track，避免重复 FP | C8P full probe5 完成；metric integrity `phase0_pass=True` |
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | 新增 `scene_link_*` diagnostics | 审计合并强度、候选、拒绝原因、输出 records 数 | summaries 已写出 `scene_link_candidate_pairs_raw/candidate_pairs/accepted_pairs/output_records/...` |

carrier_id overlap 检查：

```text
scene0050_00 adj_inter [3736, 1872, 1304, 1808]
scene0011_00 adj_inter [1416, 1328, 1720, 2064]
scene0030_00 adj_inter [1656, 3176, 3864, 3824]
scene0081_01 adj_inter [3544, 2648, 1008, 2360]
scene0591_00 adj_inter [5240, 3512, 3736, 3088]
```

解释：

```text
D4RT carrier_id 在相邻 windows 间确实有大量重叠。
因此 scene-level linking 有数据基础，不是凭空合并。
```

## C8 scene0050 smoke 对比

| Run | Link timing | Link params | AP | AP50 | AP25 | objects | points | scene links | link out records | WTA pre-conflict | WTA removed assignment |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C7 ref | none | n/a | `0.07021919799297623` | `0.3289020603973451` | `0.6961639578901516` | `57` | `78332` | n/a | n/a | `0.9235454683313956` | `0.7008287209267157` |
| C8 | pre-WTA | shared `10`, ratio `0.15` | `0.06367143810934084` | `0.2581980856602934` | `0.6924028518327863` | `55` | `78481` | `70` | `132` | `0.5540110195622504` | `0.5002147386504308` |
| C8S | pre-WTA | shared `32`, ratio `0.50` | `0.0609502809098266` | `0.2796438286570385` | `0.6770596341069114` | `55` | `78334` | `23` | `179` | `0.7189127028256584` | `0.6180039970263669` |
| C8P | post-WTA | shared `10`, ratio `0.15` | `0.07728015570932228` | `0.3411865410038038` | `0.7220291203300045` | `55` | `78501` | `70` | `132` | `0.9235454683313956` | `0.7008287209267157` |

判断：

```text
C8/C8S pre-WTA track merge 虽然降低 WTA 冲突，但 AP/AP50 下降。
原因推断：在 WTA 前合并 dense support 会把跨 window 的错误支持也一起合并，伤实例边界。

C8P post-WTA track merge 明显优于 C7。
原因推断：先让 C7 WTA 排他化点归属，再做 track-level instance merge，
能减少跨 window duplicate FP，同时不把未排他的 dense support 提前污染到同一 track。
```

## C8P full probe5 结果

| Run | AP | AP50 | AP25 | vs C7 AP | vs C7 AP50 | vs C7 AP25 | vs C4 AP | vs C4 AP50 | vs C4 AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 `carrier_corefringe_wta` | `0.020512013069687883` | `0.07965152929245886` | `0.43874809492877437` |  |  |  | ref | ref | ref |
| C7 `owned_fringe_wta` | `0.03130332812699374` | `0.12217319256494784` | `0.49161195683671977` | ref | ref | ref | `+0.010791315057305857` | `+0.04252166327248898` | `+0.0528638619079454` |
| C8P `posttrack_owned_wta` | `0.04249776562268287` | `0.17857097183073392` | `0.5404251478272147` | `+0.01119443749568913` | `+0.05639777926578608` | `+0.04881319099049497` | `+0.021985752552994987` | `+0.09891944253827505` | `+0.10167705289844034` |

与 v6/object baseline 对比：

```text
C8P:
  0.04249776562268287 / 0.17857097183073392 / 0.5404251478272147

v6 P3 compact:
  0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401

v6 P4 scoreunique:
  0.2820496510292162 / 0.4982312805326885 / 0.6912642582412274

结论：
  C8P 是当前 carrier-tracklet 分支最佳结果，
  但 AP/AP50 仍远低于 v6/object-component baseline。
```

## C7 vs C8P per-scene 证据

| Scene | C7 AP | C8P AP | delta AP | C7 AP50 | C8P AP50 | delta AP50 | C7 AP25 | C8P AP25 | delta AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene0050_00` | `0.07021919799297623` | `0.07728015570932228` | `+0.0070609577163460585` | `0.3289020603973451` | `0.3411865410038038` | `+0.012284480606458703` | `0.6961639578901516` | `0.7220291203300045` | `+0.02586516243985293` |
| `scene0011_00` | `0.025578546718713922` | `0.05644319591105128` | `+0.030864649192337355` | `0.11378856777089513` | `0.20134293904073744` | `+0.08755437126984231` | `0.4448661332149625` | `0.5430033753699983` | `+0.09813724215503583` |
| `scene0030_00` | `0.035497197949020995` | `0.03571591463028091` | `+0.00021871668125991728` | `0.10290309568828235` | `0.1794471883559009` | `+0.07654409266761855` | `0.4682466411265928` | `0.5192252384492815` | `+0.05097859732268867` |
| `scene0081_01` | `0.04320145377213323` | `0.04893204758316747` | `+0.005730593811034239` | `0.16198228101801132` | `0.1724726523481816` | `+0.010490371330170278` | `0.5006290517386844` | `0.5190651044061318` | `+0.01843605266744741` |
| `scene0591_00` | `0.019932370202365532` | `0.030686421506734` | `+0.010754051304368469` | `0.05150672636396844` | `0.12235372049825174` | `+0.0708469941342833` | `0.48633603132728864` | `0.5290872102462931` | `+0.042751178919004484` |

解释：

```text
C8P 在 5/5 场景 AP、AP50、AP25 都高于 C7。
低 AP 场景被明显抬升：
  scene0011_00 AP: 0.0256 -> 0.0564
  scene0591_00 AP: 0.0199 -> 0.0307

这说明 post-WTA scene track linking 是真实正向修复。
```

## C7 vs C8P object formation 证据

| Scene | C7 objects | C8P objects | C7 points | C8P points | C8P input records | C8P link records | C8P accepted links | C8P merged groups | C8P max group |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene0050_00` | `57` | `55` | `78332` | `78501` | `202` | `132` | `70` | `49` | `5` |
| `scene0011_00` | `50` | `43` | `85806` | `85935` | `130` | `93` | `37` | `18` | `7` |
| `scene0030_00` | `75` | `65` | `151506` | `151969` | `268` | `181` | `87` | `49` | `7` |
| `scene0081_01` | `65` | `62` | `138315` | `138389` | `163` | `121` | `42` | `37` | `4` |
| `scene0591_00` | `84` | `80` | `121832` | `121951` | `298` | `227` | `71` | `48` | `4` |

解释：

```text
C8P 没有减少 pre_points coverage，points 反而略多。
它主要减少输出 object 数，说明提升更像来自跨 window duplicate instance 合并，
而不是 dense support 塌缩。

C8/C8S 的负例也很重要：
同样的 carrier-overlap link 如果放在 WTA 前，会降低 AP/AP50。
所以当前证据支持：
  C7 ownership + WTA first，
  then C8P scene-level posttrack merge。
```

## C8P 审计结果

```text
py_compile:
  logs/stream4d_v7_c8p_py_compile_final.log
  pass

import smoke:
  logs/stream4d_v7_c8p_import_smoke_final.log
  tools.export_carrier_tracklet_graph_v7 OK
  stream4d.reliable_densifier OK

unit test:
  logs/stream4d_v7_c8p_unit_tests.log
  Ran 13 tests in 0.079s
  OK

metric integrity:
  outputs/audit/v7_metric_integrity_c8p_posttrack_owned_wta_probe5.md
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  alignment mean/min=1.000000/1.000000

reportable scan:
  outputs/audit/v7_reportable_config_scan_c8p_posttrack_owned_wta_probe5.json
  num_configs=1
  num_configs_missing_manifest=0
  num_diagnostic_only_configs=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

## C8P 追加结论

```text
C8P 是当前 v7 carrier-tracklet 分支最强版本：
  C4  = 0.020512013069687883 / 0.07965152929245886 / 0.43874809492877437
  C7  = 0.03130332812699374 / 0.12217319256494784 / 0.49161195683671977
  C8P = 0.04249776562268287 / 0.17857097183073392 / 0.5404251478272147

相比 C7:
  AP  +0.01119443749568913
  AP50 +0.05639777926578608
  AP25 +0.04881319099049497

相比 C4:
  AP  +0.021985752552994987
  AP50 +0.09891944253827505
  AP25 +0.10167705289844034

但仍未达成 Stream3D/v6 object-component 级别：
  v6 P3 AP = 0.28483247256897415
  C8P AP  = 0.04249776562268287

success=False
```

## C8P Insight

```text
正确顺序非常关键：
  pre-WTA track merge 会伤害实例质量；
  post-WTA track merge 会显著提升。

这说明当前 pipeline 的主要错误链是：
1. window 内 dense support 先需要 C7 ownership + WTA 做排他化；
2. 排他化之后，跨 window duplicates 才能通过 carrier overlap 安全合并；
3. 如果在 WTA 前合并，会把 dense support 污染带进 track，导致 AP/AP50 下降。

下一步不是继续简单调 scene-link 阈值，而是把 C8P 的 posttrack merge 与
RGB-D dense object-component 主体结合：
  D4RT carrier track 负责跨 window identity merge；
  RGB-D/object-component 负责更可靠的 dense geometry and boundary。
```

## C8 证据链索引

```text
C8 code / logs:
  Stream3D/tools/export_carrier_tracklet_graph_v7.py
  Stream3D/logs/stream4d_v7_c8_py_compile.log
  Stream3D/logs/stream4d_v7_c8_import_smoke.log
  Stream3D/logs/stream4d_v7_c8_track_owned_wta_scene0050_smoke.log
  Stream3D/logs/stream4d_v7_c8_track_owned_wta_scene0050_eval.log
  Stream3D/logs/stream4d_v7_c8s_track_owned_wta_scene0050_smoke.log
  Stream3D/logs/stream4d_v7_c8s_track_owned_wta_scene0050_eval.log
  Stream3D/logs/stream4d_v7_c8p_posttrack_owned_wta_scene0050_smoke.log
  Stream3D/logs/stream4d_v7_c8p_posttrack_owned_wta_scene0050_eval.log
  Stream3D/logs/stream4d_v7_c8p_posttrack_owned_wta_probe5.log
  Stream3D/logs/stream4d_v7_c8p_posttrack_owned_wta_probe5_eval.log
  Stream3D/logs/stream4d_v7_c8p_posttrack_owned_wta_probe5_per_scene_eval.log

C8 results:
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c8*_scene0050_00_summary.json
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c8p_posttrack_owned_wta_probe5_scene*_summary.json
  Stream3D/data/evaluation/scannet/stream4d_v7_c8*_scene0050_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c8p_posttrack_owned_wta_probe5_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c8p_posttrack_owned_wta_probe5_scene*_class_agnostic.txt

C8 integrity:
  Stream3D/logs/stream4d_v7_c8p_py_compile_final.log
  Stream3D/logs/stream4d_v7_c8p_import_smoke_final.log
  Stream3D/logs/stream4d_v7_c8p_unit_tests.log
  Stream3D/logs/stream4d_v7_metric_integrity_c8p_posttrack_owned_wta_probe5.log
  Stream3D/logs/stream4d_v7_reportable_config_scan_c8p_posttrack_owned_wta_probe5.log
  Stream3D/outputs/audit/v7_metric_integrity_c8p_posttrack_owned_wta_probe5.json
  Stream3D/outputs/audit/v7_reportable_config_scan_c8p_posttrack_owned_wta_probe5.json
```

## 追加实验 C9：dense object-component + C8P carrier support（2026-06-09）

目标：

```text
C8P 证明 D4RT carrier track 做 post-WTA duplicate merge 是正向，
但 C8P 自身仍远低于 v6 object-component baseline。

C9 按 C8P Insight 测试：
  RGB-D/object-component 作为 dense geometry 主体；
  C8P carrier track 作为 identity/support signal。

不使用 GT。
本轮不改代码，复用已有工具：
  tools.fuse_prediction_configs
  tools.object_competition_rank
```

## C9 修改审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| n/a | C9 未修改代码 | 先用现有工具验证 C8P track support 是否能增强 dense object-component | `logs/stream4d_v7_c9_py_compile_fixed.log` pass |

## C9 输入

```text
v6 dense compact primary:
  stream4d_v6_e4_probe5_objcomp_m670_g101_compact_only_preserve
  0.28382830955380606 / 0.5019341996506697 / 0.6832167422177096

v6 dense scoreunique primary:
  stream4d_v6_e4_probe5_objcomp_m670_g101_score_unique_compact_preserve
  0.28203394183547537 / 0.49820855659464575 / 0.691223277348379

C8P carrier track secondary/support:
  stream4d_v7_c8p_posttrack_owned_wta_probe5
  0.04249776562268287 / 0.17857097183073392 / 0.5404251478272147

candidate bank:
  stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
```

## C9 结果总表

| Run | Method | AP | AP50 | AP25 | vs compact AP | vs scoreunique AP | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `C9A` | dense compact primary + low-score C8P secondary, `min_ioc=0.50` | `0.10219162364608642` | `0.2038022542680502` | `0.4126460380157163` | `-0.18163668590771963` | `-0.17984231818938895` | 负向 |
| `C9B` | dense compact primary + low-score C8P secondary, `min_ioc=0.30` | `0.13289377146375558` | `0.2596510816552011` | `0.43066751254252333` | `-0.15093453809005048` | `-0.1491401703717198` | 负向 |
| `C9C` | object competition，用 C8P support，`tmp_policy=score_support`，compact | `0.0008465063976877719` | `0.005090165689382481` | `0.049447566929227645` | `-0.28298180315611827` | `-0.2811874354377876` | 错误设置导致灾难性下降 |
| `C9D` | object competition，用 C8P support，`tmp_policy=score_support`，scoreunique | `0.0008458586177464427` | `0.005086113157248588` | `0.04979020001581942` | `-0.2829824509360596` | `-0.2811880832177289` | 错误设置导致灾难性下降 |
| `C9E` | object competition，C8P 只参与 ranking，`tmp_policy=inherit`，compact | `0.28364997416761173` | `0.501653928415879` | `0.683017785427424` | `-0.00017833538619432848` | `+0.0016160323321363627` | 接近 v6 compact，但未超过 |
| `C9F` | object competition，C8P 只参与 ranking，`tmp_policy=inherit`，scoreunique | `0.28201848259966583` | `0.49818619479107373` | `0.6911829392930654` | `-0.0018098269541402268` | `-0.000015459235809541147` | 接近 v6 scoreunique，但未超过 |

判断：

```text
C9 best = C9E
AP = 0.28364997416761173

与 v6 compact baseline 比：
  delta AP  = -0.00017833538619432848
  delta AP50 = -0.0002802712347907176
  delta AP25 = -0.00019895679028559692

与 C8P carrier-only 比：
  delta AP  = +0.24115220854492887
  delta AP50 = +0.32308295658514506
  delta AP25 = +0.14259263760020934

success=False
C9 没有超过 v6 dense object-component baseline。
```

## C9 证据链

### C9A/C9B fusion 证据

```text
C9A aggregate:
  mean_num_primary_instances = 299.6
  mean_num_secondary_instances = 31.4
  mean_num_output_instances = 331.0
  mean_primary_union_ratio = 0.041981204148124125
  mean_secondary_union_ratio = 0.18202767239288514
  mean_output_union_ratio = 0.2225681686940458

C9B aggregate:
  mean_num_primary_instances = 299.6
  mean_num_secondary_instances = 26.6
  mean_num_output_instances = 326.2
  mean_primary_union_ratio = 0.041981204148124125
  mean_secondary_union_ratio = 0.13913892425047605
  mean_output_union_ratio = 0.18074331635854182
```

解释：

```text
即使 C8P secondary 低分，仍大幅扩大 union：
  compact primary union ratio ≈ 4.20%
  C9A output union ratio ≈ 22.26%
  C9B output union ratio ≈ 18.07%

这带来大量低分 FP tail，AP 从 0.2838 降到 0.102/0.133。
说明 C8P track masks 不适合作为 dense primary 的直接低分补召回。
```

### C9C/C9D blocker 与修复

```text
C9C/C9D 使用：
  --score-pre-points-config stream4d_v7_c8p_posttrack_owned_wta_probe5
  --tmp-policy score_support

metric integrity:
  policy = {"inconsistent_union_not_subset": 5}
  C9C pre % = 54.2822, union % = 4.2142
  C9D pre % = 54.2822, union % = 4.2518
```

分析：

```text
这不是合理的 dense-main pipeline。
它把 evaluator/TMP support 也替换成 C8P track union，
导致 prediction union 不再是 pre_points 的自然子集。

因此按修复方向改为 C9E/C9F：
  C8P support 只参与 object ranking；
  tmp_policy=inherit，保持 dense RGB-D/object-component TMP 主体。
```

### C9E/C9F 修复后证据

```text
C9E metric integrity:
  policy = {"inherit_or_fixed_superset": 5}
  AP/AP50/AP25 = 0.28364997416761173 / 0.501653928415879 / 0.683017785427424

C9F metric integrity:
  policy = {"inherit_or_fixed_superset": 5}
  AP/AP50/AP25 = 0.28201848259966583 / 0.49818619479107373 / 0.6911829392930654

reportable scan:
  num_configs=6
  num_configs_missing_manifest=0
  num_diagnostic_only_configs=0
  num_oracle_configs=0
  num_reportable_method_configs=6
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

解释：

```text
C9E/C9F 修复了 TMP 错误后恢复到 v6 baseline 附近，
但没有带来有效增益。

C9E 与 v6 compact 的 AP 差只有 -0.000178m，
C9F 与 v6 scoreunique 的 AP 差只有 -0.000015m。
这说明 C8P support/ranking signal 对当前 dense candidate bank 几乎没有区分力。
```

## C9 审计结果

```text
py_compile:
  logs/stream4d_v7_c9_py_compile_fixed.log
  pass

metric integrity:
  outputs/audit/v7_metric_integrity_c9_probe5.md
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=6
  num_suspicious_configs=0

reportable scan:
  outputs/audit/v7_reportable_config_scan_c9_probe5.json
  num_configs=6
  num_configs_missing_manifest=0
  num_diagnostic_only_configs=0
  num_oracle_configs=0
  num_reportable_method_configs=6
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

## C9 结论

```text
C9 没有达成目标。

C8P track mask 直接作为 secondary recall 会显著拉低 AP。
C8P track union 作为 dense candidate ranking support，在修复 TMP policy 后基本等价于原 v6 ranking，
没有超过 v6 dense object-component baseline。

因此“RGB-D dense 主体 + C8P support/ranking”这条简单组合不够。
```

## C9 Insight

```text
C8P 的价值是 scene-level duplicate identity merge，
不是 dense object boundary，也不是候选质量排序。

原因链：
1. C8P carrier masks 很大，union ratio 远高于 v6 dense primary。
2. 作为 secondary 输出会引入大量低分 FP。
3. 作为 score support 时，C8P union 对 dense candidates 覆盖过宽，缺少 true-object-level discriminative signal。
4. 修复 TMP 后结果接近 v6 baseline，说明 C8P ranking signal 没能改变 object-component 的关键排序。

下一步若继续，不能再把 C8P 当 support union 使用。
更合理的方向是 object-level bipartite assignment：
  以 v6 dense object 为边界主体；
  C8P track 只约束跨 window / duplicate 的 one-to-one identity；
  不输出 C8P coarse mask，不用 C8P union 替换 evaluation support；
  合并/压制必须基于 dense object 与 track 的稳定多帧关系，而不是单个 union overlap。
```

## C9 证据链索引

```text
C9 logs:
  Stream3D/logs/stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050_fuse.log
  Stream3D/logs/stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030_fuse.log
  Stream3D/logs/stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050_eval.log
  Stream3D/logs/stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030_eval.log
  Stream3D/logs/stream4d_v7_c9c_objcomp_c8p_support_compact_m670_g101.log
  Stream3D/logs/stream4d_v7_c9d_objcomp_c8p_support_scoreunique_m670_g101.log
  Stream3D/logs/stream4d_v7_c9c_objcomp_c8p_support_compact_m670_g101_eval.log
  Stream3D/logs/stream4d_v7_c9d_objcomp_c8p_support_scoreunique_m670_g101_eval.log
  Stream3D/logs/stream4d_v7_c9e_objcomp_c8p_rank_inherit_compact_m670_g101.log
  Stream3D/logs/stream4d_v7_c9f_objcomp_c8p_rank_inherit_scoreunique_m670_g101.log
  Stream3D/logs/stream4d_v7_c9e_objcomp_c8p_rank_inherit_compact_m670_g101_eval.log
  Stream3D/logs/stream4d_v7_c9f_objcomp_c8p_rank_inherit_scoreunique_m670_g101_eval.log

C9 summaries:
  Stream3D/outputs/stream4d_v7_c9_fusion/stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050_summary.json
  Stream3D/outputs/stream4d_v7_c9_fusion/stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030_summary.json
  Stream3D/outputs/stream4d_v7_c9_object_competition/stream4d_v7_c9c_objcomp_c8p_support_compact_m670_g101_summary.json
  Stream3D/outputs/stream4d_v7_c9_object_competition/stream4d_v7_c9d_objcomp_c8p_support_scoreunique_m670_g101_summary.json
  Stream3D/outputs/stream4d_v7_c9_object_competition/stream4d_v7_c9e_objcomp_c8p_rank_inherit_compact_m670_g101_summary.json
  Stream3D/outputs/stream4d_v7_c9_object_competition/stream4d_v7_c9f_objcomp_c8p_rank_inherit_scoreunique_m670_g101_summary.json

C9 results:
  Stream3D/data/evaluation/scannet/stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c9c_objcomp_c8p_support_compact_m670_g101_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c9d_objcomp_c8p_support_scoreunique_m670_g101_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c9e_objcomp_c8p_rank_inherit_compact_m670_g101_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c9f_objcomp_c8p_rank_inherit_scoreunique_m670_g101_class_agnostic.txt

C9 integrity:
  Stream3D/logs/stream4d_v7_metric_integrity_c9_probe5.log
  Stream3D/logs/stream4d_v7_metric_integrity_c9_probe5_fixed.log
  Stream3D/logs/stream4d_v7_reportable_config_scan_c9_probe5_fixed.log
  Stream3D/outputs/audit/v7_metric_integrity_c9_probe5.md
  Stream3D/outputs/audit/v7_metric_integrity_c9_probe5.json
  Stream3D/outputs/audit/v7_reportable_config_scan_c9_probe5.md
  Stream3D/outputs/audit/v7_reportable_config_scan_c9_probe5.json
```

## 追加实验 C10：dense masks + C8P track-overlap score-only rescore（2026-06-09）

目标：

```text
C9 证明不能直接输出 C8P coarse masks，也不能用 C8P union 替换 dense support。

C10 进一步收窄尝试：
  v6 dense compact masks 不变；
  v6 dense TMP/pre_points 不变；
  C8P carrier track 只作为 score tiebreaker。

这测试“track-overlap feature 是否能改进 dense object score ordering”。
```

## C10 修改审计

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| n/a | C10 未修改源码；用 Python heredoc 生成 prediction artifact | 测试 score-only track overlap cue，避免 C9 的 coarse mask / support union 污染 | `outputs/audit/v7_metric_integrity_c10_probe5.md` pass |

## C10 方法定义

```text
primary_config = stream4d_v6_e4_probe5_objcomp_m670_g101_compact_only_preserve
track_config   = stream4d_v7_c8p_posttrack_owned_wta_probe5

对每个 dense object 与每个 C8P track 计算 overlap：
  inter = dense_mask.T @ c8p_track_mask

best_primary_ioc = max_track inter / dense_object_area
best_track_ioc   = max_track inter / c8p_track_area

输出：
  pred_masks   = primary pred_masks
  pred_classes = primary pred_classes
  TMP/pre_points = primary TMP/pre_points
  pred_score = original_score + weight * feature
```

## C10 结果

| Run | Feature | Weight | AP | AP50 | AP25 | vs v6 compact AP | vs v6 compact AP50 | vs v6 compact AP25 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `C10A` | `best_primary_ioc` | `0.01` | `0.15166185826289458` | `0.33012994011446756` | `0.596671879087483` | `-0.13216645129091148` | `-0.17180425953620214` | `-0.08654486313022661` |
| `C10B` | `best_primary_ioc` | `0.10` | `0.15166185826289458` | `0.33012994011446756` | `0.596671879087483` | `-0.13216645129091148` | `-0.17180425953620214` | `-0.08654486313022661` |
| `C10C` | `best_track_ioc` | `0.10` | `0.20392158827794155` | `0.4043240507748191` | `0.5627806069152249` | `-0.07990672127586451` | `-0.09761014887585072` | `-0.1204361353024847` |
| `C10D` | `best_track_ioc` | `0.50` | `0.20371311021912747` | `0.403070414779716` | `0.5622845830274148` | `-0.08011519933467859` | `-0.09886378487095372` | `-0.12093215919029458` |

判断：

```text
C10 best = C10C
AP = 0.20392158827794155

仍低于 v6 compact baseline：
  v6 compact = 0.28382830955380606 / 0.5019341996506697 / 0.6832167422177096
  C10C      = 0.20392158827794155 / 0.4043240507748191 / 0.5627806069152249

success=False
```

## C10 feature 证据

```text
best_primary_ioc feature:
  mean = 0.6746636509895325
  p50  = 0.6725598454475403
  p90  = 0.989612340927124
  max  = 1.0

best_track_ioc feature:
  mean = 0.05441656038165092
  p50  = 0.011941186571493744
  p90  = 0.167325259745121
  max  = 0.5258764505386353
```

解释：

```text
best_primary_ioc 对大部分 dense object 都很高，区分力不足；
加 0.01 和 0.10 得到完全相同 AP，说明它主要打乱排序而不是改善排序。

best_track_ioc 更稀疏，但仍明显负向：
它奖励“dense object 覆盖了多少 C8P track”，而 C8P track 本身很粗，
该信号与 true object quality 不一致。
```

## C10 审计结果

```text
metric integrity:
  outputs/audit/v7_metric_integrity_c10_probe5.md
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=4
  num_suspicious_configs=0

reportable scan:
  outputs/audit/v7_reportable_config_scan_c10_probe5.json
  num_configs=4
  num_configs_missing_manifest=0
  num_diagnostic_only_configs=0
  num_oracle_configs=0
  num_reportable_method_configs=4
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

## C10 结论

```text
C10 没有达成目标。

即使保持 dense masks/TMP 完全不变，只用 C8P track overlap 调整 score，
AP 仍明显低于 v6 compact baseline。

这说明 C8P track overlap 不是可靠的 score-ordering cue。
```

## C10 Insight

```text
C8P track 的有效使用范围进一步收窄：
  可以做 post-WTA duplicate merge；
  不适合直接输出 coarse mask；
  不适合替换 dense support；
  不适合作为简单 score cue。

下一步如果继续，必须做更结构化的 object-level assignment：
1. 不输出 C8P mask；
2. 不用 C8P union；
3. 不用 single overlap feature 排序；
4. 只在 dense object candidates 之间存在真实 duplicate/temporal conflict 时，
   用 C8P track 作为 cannot-link / must-link 的辅助证据。
```

## C10 证据链索引

```text
C10 logs:
  Stream3D/logs/stream4d_v7_c10_dense_track_overlap_rescore_generate.log
  Stream3D/logs/stream4d_v7_c10a_dense_compact_track_pioc_w001_eval.log
  Stream3D/logs/stream4d_v7_c10b_dense_compact_track_pioc_w010_eval.log
  Stream3D/logs/stream4d_v7_c10c_dense_compact_track_tioc_w010_eval.log
  Stream3D/logs/stream4d_v7_c10d_dense_compact_track_tioc_w050_eval.log

C10 summaries:
  Stream3D/outputs/stream4d_v7_c10_track_rescore/stream4d_v7_c10a_dense_compact_track_pioc_w001_summary.json
  Stream3D/outputs/stream4d_v7_c10_track_rescore/stream4d_v7_c10b_dense_compact_track_pioc_w010_summary.json
  Stream3D/outputs/stream4d_v7_c10_track_rescore/stream4d_v7_c10c_dense_compact_track_tioc_w010_summary.json
  Stream3D/outputs/stream4d_v7_c10_track_rescore/stream4d_v7_c10d_dense_compact_track_tioc_w050_summary.json

C10 results:
  Stream3D/data/evaluation/scannet/stream4d_v7_c10a_dense_compact_track_pioc_w001_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c10b_dense_compact_track_pioc_w010_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c10c_dense_compact_track_tioc_w010_class_agnostic.txt
  Stream3D/data/evaluation/scannet/stream4d_v7_c10d_dense_compact_track_tioc_w050_class_agnostic.txt

C10 integrity:
  Stream3D/logs/stream4d_v7_metric_integrity_c10_probe5.log
  Stream3D/logs/stream4d_v7_reportable_config_scan_c10_probe5.log
  Stream3D/outputs/audit/v7_metric_integrity_c10_probe5.md
  Stream3D/outputs/audit/v7_metric_integrity_c10_probe5.json
  Stream3D/outputs/audit/v7_reportable_config_scan_c10_probe5.md
  Stream3D/outputs/audit/v7_reportable_config_scan_c10_probe5.json
```

## 追加实验 C11/C12/C13（2026-06-09）

追加目标：

```text
C10 证明 C8P track overlap 不适合简单 score cue。
按 C10 insight 继续更保守地测试：
  不输出 C8P mask；
  不使用 C8P union / TMP；
  dense primary masks 和 TMP 作为唯一输出 support；
  C8P track 只作为 dense candidates 之间 duplicate / temporal bucket 的辅助证据。

所有 C11/C12/C13 均 uses_gt=False，is_method_result=True，
evaluation 使用 --require-manifest。
```

新增复现脚本：

| 文件 | 修改 | 目的 | 验证 |
|---|---|---|---|
| `Stream3D/tools/export_trackbucket_dense_variants_v7.py` | 固化 C11/C12/C13 dense-primary track-bucket suppression 生成逻辑 | 原始结果来自 Python heredoc；新增脚本让后续可按 group/config 重跑复现同类 artifact | `logs/stream4d_v7_trackbucket_repro_py_compile.log` pass；`logs/stream4d_v7_trackbucket_repro_import_smoke.log` pass |

注意：

```text
C11/C12/C13 指标来自已经落盘的真实 heredoc 运行和 evaluator 结果。
新增脚本是 r5 追加的复现工具，不倒填为当时的生成命令。
```

## C11：track-bucket NMS

方法定义：

```text
primary_config:
  compact     = stream4d_v6_e4_probe5_objcomp_m670_g101_compact_only_preserve
  scoreunique = stream4d_v6_e4_probe5_objcomp_m670_g101_score_unique_compact_preserve

track_config = stream4d_v7_c8p_posttrack_owned_wta_probe5

对 dense object 和 C8P track 计算 primary IoC。
当 best_primary_ioc >= assign_primary_ioc_min 时，将 dense object 放入该 track bucket。
bucket 内按 mask overlap 做 NMS；输出仍只保留 dense primary masks/classes/scores，TMP 继承 primary。
```

C11 过程 blocker 与修复：

```text
1. 初始全量 Python heredoc 生成命令过慢，手动终止；日志为空。
2. full-scene dense matrix 版本被 kill，exit 143；日志为空。
3. 修复方向：按计划避免回到粗暴全量重计算，改成 sparse primary-union overlap，只在 dense primary union / track bucket 范围内计算；该版本成功生成 C11A-E。
```

C11 结果：

| Run | Primary | assign | overlap | AP | AP50 | AP25 | vs primary AP | #suppressed mean | union before -> after |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `C11A` | compact | `0.50` | `min_ioc@0.90` | `0.2704320873317211` | `0.47988173584030724` | `0.6695362536371277` | `-0.013396222222084975` | `47.4` | `8912.6 -> 8209.4` |
| `C11B` | compact | `0.50` | `min_ioc@0.75` | `0.2686413052853578` | `0.47453179052845046` | `0.6644916609507336` | `-0.015187004268448268` | `64.2` | `8912.6 -> 8117.2` |
| `C11C` | compact | `0.50` | `iou@0.50` | `0.27236752539720466` | `0.47994690773425164` | `0.6696705657993772` | `-0.0114607841566014` | `46.6` | `8912.6 -> 8527.6` |
| `C11D` | scoreunique | `0.50` | `min_ioc@0.90` | `0.26886596867628404` | `0.47656996919717703` | `0.6780442110000419` | `-0.013167973159191326` | `38.8` | `9010.0 -> 8382.8` |
| `C11E` | scoreunique | `0.50` | `min_ioc@0.75` | `0.2671394709133604` | `0.4714001637618435` | `0.6731127314002505` | `-0.014894470922114988` | `58.2` | `9010.0 -> 8284.4` |

判断：

```text
C11 明显 over-suppression。
它删除 38.8 到 64.2 个 instances/scene，并显著降低 prediction union。
AP/AP50/AP25 全部低于各自 primary baseline。

结论：C8P track bucket 不能用中等阈值做 broad duplicate NMS。
```

## C12：ultra-conservative track-bucket suppression

修复方向：

```text
C11 删除太多 dense candidates。
C12 提高 assignment/overlap 阈值：
  assign_primary_ioc_min = 0.90 或 0.95
  min_ioc@0.99 或 iou@0.85
目标是只删除几乎确定的 duplicate。
```

C12 结果：

| Run | Primary | assign | overlap | AP | AP50 | AP25 | vs primary AP | vs primary AP50 | vs primary AP25 | #suppressed mean |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `C12A` | compact | `0.90` | `min_ioc@0.99` | `0.2800747423738896` | `0.49685112752837807` | `0.6829500187596522` | `-0.003753567179916484` | `-0.0050830721222915876` | `-0.00026672345805733855` | `9.0` |
| `C12B` | compact | `0.90` | `iou@0.85` | `0.283520751520795` | `0.5023288260789226` | `0.6829700373610922` | `-0.0003075580330110794` | `+0.00039462642825294214` | `-0.0002467048566173746` | `5.8` |
| `C12C` | compact | `0.95` | `min_ioc@0.99` | `0.2799994121344577` | `0.49673714634563515` | `0.6827419329888822` | `-0.0038288974193483316` | `-0.005197053305034505` | `-0.0004748092288273664` | `3.8` |
| `C12D` | scoreunique | `0.90` | `min_ioc@0.99` | `0.27834384843765725` | `0.4932120531494701` | `0.6916215209066441` | `-0.003690093397818117` | `-0.00499650344517566` | `+0.00039824355826512114` | `6.0` |
| `C12E` | scoreunique | `0.90` | `iou@0.85` | `0.28169453388969407` | `0.49853735125740806` | `0.6916715496834056` | `-0.0003394079457813004` | `+0.0003287946627623106` | `+0.0004482723350266449` | `3.4` |

判断：

```text
C12 把 C11 的伤害大幅降下来，但仍没有超过 primary AP。
C12B/C12E 有极小 AP50 或 AP25 局部收益，但 AP 仍下降。

结论：track bucket 可以找到少量 near-duplicate，
但当前信号无法可靠区分“应删 duplicate”和“低分但有用 candidate”。
```

## C13：conservative IoU grid

修复方向：

```text
C12 best 来自 iou@0.85。
C13 继续缩小搜索，只测试更保守 IoU：
  iou@0.90 / iou@0.95
  assign_primary_ioc_min=0.90 / 0.95
```

C13 结果：

| Run | Primary | assign | overlap | AP | AP50 | AP25 | vs primary AP | vs primary AP50 | vs primary AP25 | #suppressed mean |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `C13A` | compact | `0.90` | `iou@0.90` | `0.2837834686112302` | `0.5021096487158756` | `0.6827009115465468` | `-0.00004484094257584115` | `+0.00017544906520594594` | `-0.0005158306711627869` | `3.0` |
| `C13B` | compact | `0.90` | `iou@0.95` | `0.28382830955380606` | `0.5019341996506697` | `0.6824298780157181` | `0.0` | `0.0` | `-0.0007868642019914773` | `0.0` |
| `C13C` | compact | `0.95` | `iou@0.90` | `0.28375652456257155` | `0.5020692354906476` | `0.682628369001877` | `-0.00007178499123450255` | `+0.00013503583997798163` | `-0.0005883732158326049` | `1.6` |
| `C13D` | compact | `0.95` | `iou@0.95` | `0.28382830955380606` | `0.5019341996506697` | `0.6824298780157181` | `0.0` | `0.0` | `-0.0007868642019914773` | `0.0` |
| `C13E` | scoreunique | `0.90` | `iou@0.90` | `0.28195599230250884` | `0.49833569328009064` | `0.6914188730867814` | `-0.00007794953296652585` | `+0.0001271366854448952` | `+0.00019559573840244315` | `1.4` |
| `C13F` | scoreunique | `0.90` | `iou@0.95` | `0.28203394183547537` | `0.49820855659464575` | `0.691223277348379` | `0.0` | `0.0` | `0.0` | `0.0` |

注意：

```text
C13B/C13D 的 mean_num_suppressed=0。
AP/AP50 与 compact baseline 完全相同，但 AP25 比 baseline 低 0.0007868642019914773；
这很可能来自导出时同分 candidate 顺序变化导致的 evaluator tie-order 微差。
因此不能把 no-suppression row 当成真实方法收益。
```

判断：

```text
C13 失败。
当阈值保守到安全时，suppression 约等于 0，结果退回 baseline；
一旦实际删除少量 candidates，AP 仍不升。

当前 best：
  compact line: C13A AP50 +0.00017544906520594594，但 AP -0.00004484094257584115。
  scoreunique line: C13E AP50 +0.0001271366854448952, AP25 +0.00019559573840244315，但 AP -0.00007794953296652585。

这些都不能 claim 改进。
```

## C11/C12/C13 审计结果

```text
C11:
  outputs/audit/v7_metric_integrity_c11_probe5.md
  phase0_pass=True
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=5
  num_suspicious_configs=0
  reportable scan clean

C12:
  outputs/audit/v7_metric_integrity_c12_probe5.md
  phase0_pass=True
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=5
  num_suspicious_configs=0
  reportable scan clean

C13:
  outputs/audit/v7_metric_integrity_c13_probe5.md
  phase0_pass=True
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=6
  num_suspicious_configs=0
  reportable scan clean
```

## C11/C12/C13 分析

```text
1. C11 的负结果说明 C8P track bucket 的 grouping 太粗。
   中等阈值会删除大量 dense candidates，并同步降低 union 和 AP。

2. C12/C13 把阈值收紧后，metric 接近 baseline。
   这说明确实能找到少量 near-duplicate，但删掉这些 object 并不能稳定提升 AP。

3. 局部 AP50/AP25 微小提升不能支撑方法进展。
   它们伴随 AP 下降，幅度在 1e-4 到 4e-4 级别，且 no-suppression tie-order 已能带来 AP25 级别微差。

4. 证据链继续支持 C10 的判断：
   C8P track 不适合做 output mask、support union、score cue、或 broad NMS。
   它最多能作为更晚阶段的可解释 diagnostic / cannot-link 线索，
   但当前 C8P quality 不足以直接驱动 dense object selection。
```

## C11/C12/C13 结论与 Insight

```text
结论：
  track-bucket duplicate suppression 这条修复线没有达成目标。
  C11 太 aggressive，C12/C13 太 conservative 时退回 baseline；
  没有出现“删除少量 duplicate 后 AP 稳定上升”的区间。

Insight：
  当前瓶颈不是缺少一个 NMS 阈值，而是缺少可靠的 object-level identity evidence。
  C8P track 对 dense object 的映射更多反映 coarse coverage，而不是 true instance equivalence。
  下一步如果继续，不应再做 track-bucket 阈值网格，
  而应回到 plan 中 carrier-tracklet graph 的可视化/诊断：
    1. 检查 carrier clusters 是否跨真实物体；
    2. 检查 same-frame cannot-link 是否在 merge 前生效；
    3. 检查低分 dense candidates 被删时是否对应独立 GT；
    4. 引入 object birth/death 和 per-frame exclusivity，而不是后验删 prediction。
```

## C11/C12/C13 证据链索引

```text
Repro script:
  Stream3D/tools/export_trackbucket_dense_variants_v7.py
  Stream3D/logs/stream4d_v7_trackbucket_repro_py_compile.log
  Stream3D/logs/stream4d_v7_trackbucket_repro_import_smoke.log

C11:
  Stream3D/logs/stream4d_v7_c11_track_bucket_nms_generate.log
  Stream3D/logs/stream4d_v7_c11_track_bucket_nms_generate_fast.log
  Stream3D/logs/stream4d_v7_c11_track_bucket_nms_generate_sparse.log
  Stream3D/logs/stream4d_v7_c11*_eval.log
  Stream3D/outputs/stream4d_v7_c11_track_bucket_nms/*_summary.json
  Stream3D/outputs/audit/v7_metric_integrity_c11_probe5.{md,json}
  Stream3D/outputs/audit/v7_reportable_config_scan_c11_probe5.{md,json,csv}

C12:
  Stream3D/logs/stream4d_v7_c12_conservative_track_bucket_generate.log
  Stream3D/logs/stream4d_v7_c12*_eval.log
  Stream3D/outputs/stream4d_v7_c12_conservative_track_bucket/*_summary.json
  Stream3D/outputs/audit/v7_metric_integrity_c12_probe5.{md,json}
  Stream3D/outputs/audit/v7_reportable_config_scan_c12_probe5.{md,json,csv}

C13:
  Stream3D/logs/stream4d_v7_c13_conservative_iou_grid_generate.log
  Stream3D/logs/stream4d_v7_c13*_eval.log
  Stream3D/outputs/stream4d_v7_c13_conservative_iou_grid/*_summary.json
  Stream3D/outputs/audit/v7_metric_integrity_c13_probe5.{md,json}
  Stream3D/outputs/audit/v7_reportable_config_scan_c13_probe5.{md,json,csv}
```

## 最新审计包

```text
packet:
  stream4d_v7_code_review_packet_20260609_022358_r5.zip

sha256:
  see sibling file stream4d_v7_code_review_packet_20260609_022358_r5.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

filelist:
  stream4d_v7_code_review_packet_20260609_022358_r5_filelist.txt

zip test:
  see r5 打包执行日志
```

## 收尾实验 C14/C15（2026-06-09）

收尾目标：

```text
C11/C12/C13 track-bucket duplicate suppression 没有稳定收益。
为避免误判，继续做两类确认：
1. GT-only diagnostic：被删除 candidates 是否其实是有用真阳性。
2. C15 strict posttrack merge：C8P/C14 是否只是 posttrack over-merge。

所有 GT diagnostic 均标记 diagnostic_only=True, uses_gt=True，不进入 method result。
C14/C15 method result 均 uses_gt=False，并通过 --require-manifest audit。
```

## C14：track-bucket suppression GT diagnostic

修改审计：

```text
新增文件：
  Stream3D/tools/diagnose_trackbucket_suppression_v7.py

用途：
  读取 GT 只用于错误分析，判断 C11/C12/C13 suppression 删除的 candidates 是否有 GT 覆盖价值。

审计标记：
  diagnostic_only=True
  uses_gt=True
  purpose=error analysis only

不作为：
  method result
  model selection table
```

Blocker 处理：

```text
第一次跑全量 C11/C12/C13 diagnostic 超过约 90s 仍无输出，手动 Ctrl-C。
没有落盘指标，因此不补写结果。

随后修复方向：
  给 diagnostic 工具增加 --configs allowlist；
  增加 progress print；
  改跑代表配置。
```

代表配置结果：

| Config | suppressed total/mean | assigned mean | suppressed GT IoU mean | GT IoU >=25/50 | same GT | harmful@25/50 | safe duplicate-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| `stream4d_v7_c11a_compact_trackbucket_minioc090_a050` | `198 / 39.60` | `202.80` | `0.036966402022634876` | `0.010101010101010102 / 0.010101010101010102` | `0.7626262626262627` | `0.0 / 0.0` | `1.0` |
| `stream4d_v7_c12b_compact_trackbucket_iou085_a090` | `22 / 4.40` | `54.60` | `0.02014944291361396` | `0.0 / 0.0` | `0.5909090909090909` | `0.0 / 0.0` | `1.0` |
| `stream4d_v7_c13a_compact_trackbucket_iou090_a090` | `12 / 2.40` | `54.60` | `0.034950627838649474` | `0.0 / 0.0` | `0.6666666666666666` | `0.0 / 0.0` | `1.0` |
| `stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090` | `4 / 0.80` | `54.60` | `0.056628791615366936` | `0.0 / 0.0` | `0.5` | `0.0 / 0.0` | `1.0` |

解释：

```text
被删除 candidates 几乎没有有效 GT 覆盖：
  C12/C13 suppressed GT IoU >=0.25/0.50 全为 0；
  C11 也只有 0.0101/0.0101。

harmful@25/50 全为 0，safe_duplicate_like 全为 1.0。

因此 C11/C12/C13 没提升，不是因为 suppression 删掉了大量真正有用的 GT candidates。
更合理的解释是：
  当前 dense baseline 的 AP/ranking 缺口不在这些低质量重复项；
  删除它们只能轻微改变排序/覆盖，无法修复 object quality。
```

证据链：

```text
Stream3D/tools/diagnose_trackbucket_suppression_v7.py
Stream3D/logs/stream4d_v7_trackbucket_suppression_gt_diagnostic_representative.log
Stream3D/outputs/audit/v7_trackbucket_suppression_gt_diagnostic_representative.{json,csv,md}
```

## C14：C8P reproduction 与 object_dict fallback

修改审计：

```text
文件：
  Stream3D/tools/export_carrier_tracklet_graph_v7.py

修改：
  object_dict.npy 写入 processed object dir 失败时 fallback 到：
    outputs/v7_carrier_tracklet_graph/object_dicts/<config>/<scene>/object_dict.npy

原因：
  当前 sandbox 中 data/scannet/processed/.../object/<config> 为只读。
  prediction npz 和 TMP pre_points 可写，不影响 evaluator；
  但 object_dict 是审计 alignment 所需，不能静默丢失。

summary 新增：
  object_dict_write_fallback
  object_dict_write_path
  object_dict_write_error
```

C14 结果：

| Run | AP | AP50 | AP25 | phase0 | scan clean | object_dict fallback |
|---|---:|---:|---:|---|---|---:|
| `C8P posttrack owned WTA` | `0.04249776562268287` | `0.17857097183073392` | `0.5404251478272147` | previously pass | previously clean | n/a |
| `C14 strict_owned_posttrack` | `0.04249776562268287` | `0.17857097183073392` | `0.5404251478272147` | `True` | clean | `5/5` |

C8P vs C14 prediction sha256：

| Scene | sha256 |
|---|---|
| `scene0011_00` | `89250f283adb49233fac07dc61a64d852bd2c747bef6dc8ea227b3c3885cf839` |
| `scene0030_00` | `89a90e0bb5a5dffbdb428cbb07a6fe8cc894c8ffa21018d5af31ae1a12dc9510` |
| `scene0050_00` | `f7bf55fb1bda4f79f8195e88a09d251cb1aa31b1dff6cca1f0c87a6f7b2f3278` |
| `scene0081_01` | `adbe3dce2a0e61d71870e1a2603d326a002fcfef9f6f2fbd53106cbd8367f5aa` |
| `scene0591_00` | `df1f70d8260904dd79d299e5ceb7f6e82dd885e8850ff343b74857bf8c1c6b7c` |

判断：

```text
C14 与 C8P 的 5 个 scene npz sha256 完全一致。
因此 C14 不是新性能，只是复现 C8P，并验证 object_dict fallback 可工作。
```

## C15：strict posttrack merge

方法：

```text
保持 C8P/C14 的 window formation 不变：
  support_mode=core_owned_fringe_wta_posttrack
  min_shared_frames=3
  min_positive_ratio=0.65
  max_pair_distance_variance=0.005
  max_component_carriers=180
  min_mask_carriers=6
  min_frame_mask_ratio=0.65

只收紧 posttrack scene link：
  scene_link_min_shared_carriers=32
  scene_link_min_overlap_ratio=0.50
```

结果：

| Run | AP | AP50 | AP25 | vs C8P AP | vs C8P AP50 | vs C8P AP25 | phase0 | scan clean |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `C8P/C14` | `0.04249776562268287` | `0.17857097183073392` | `0.5404251478272147` | ref | ref | ref | `True` | clean |
| `C15 strict_posttrack_merge` | `0.03494669836186661` | `0.14738590648397554` | `0.5024238943440352` | `-0.007551067260816258` | `-0.03118506534675838` | `-0.03800125348317946` | `True` | clean |

C8P/C14 vs C15 formation summary：

| Metric | C8P/C14 sum/mean | C15 sum/mean |
|---|---:|---:|
| `num_candidate_records` | `1061 / 212.2` | `1061 / 212.2` |
| `scene_link_candidate_pairs_raw` | `693 / 138.6` | `693 / 138.6` |
| `scene_link_candidate_pairs` | `320 / 64.0` | `64 / 12.8` |
| `scene_link_accepted_pairs` | `307 / 61.4` | `63 / 12.6` |
| `scene_link_output_records` | `754 / 150.8` | `998 / 199.6` |
| `scene_link_merged_groups` | `201 / 40.2` | `55 / 11.0` |
| `scene_link_mean_group_size` | `7.068698244507041 / 1.4137396489014082` | `5.3167531277206965 / 1.0633506255441394` |
| `num_exported_objects` | `305 / 61.0` | `325 / 65.0` |
| `num_exported_points` | `576745 / 115349.0` | `576035 / 115207.0` |
| `densify_wta_pre_conflict_rate` | `4.254641864423856 / 0.8509283728847713` | `4.254641864423856 / 0.8509283728847713` |
| `densify_wta_removed_assignment_rate` | `3.5343184226591697 / 0.7068636845318339` | `3.5343184226591697 / 0.7068636845318339` |
| `export_conflict_rate` | `0.0 / 0.0` | `0.0 / 0.0` |

C15 per-scene formation：

| Scene | accepted links | output records | exported objects | exported points |
|---|---:|---:|---:|---:|
| `scene0011_00` | `4` | `126` | `48` | `85806` |
| `scene0030_00` | `15` | `253` | `74` | `151682` |
| `scene0050_00` | `23` | `179` | `56` | `78385` |
| `scene0081_01` | `8` | `155` | `64` | `138327` |
| `scene0591_00` | `13` | `285` | `83` | `121835` |

解释：

```text
C15 确实大幅减少 posttrack merge：
  accepted links: 307 -> 63
  output records: 754 -> 998
  objects: 305 -> 325

但 AP/AP50/AP25 全部低于 C8P/C14。
因此 C8P/C14 的主要问题不是简单 posttrack over-merge。
收紧 merge 会让 object 更碎，反而损失 AP。
```

## r6 audit fallback 结果

修改审计：

```text
文件：
  Stream3D/tools/verify_stream4d_metric_integrity.py

修改：
  object_dict alignment 优先查原 processed object_dict；
  若缺失，则读取 exporter summary 中的 object_dict_write_path。

目的：
  C14/C15 因 processed object dir 只读使用 fallback 路径；
  r5 audit 虽 phase0_pass=True，但 alignment 显示 missing/skipped。
  r6 修复后，alignment 可真实检查。
```

r6 metric integrity：

| Config | phase0 | alignment checked | path source | alignment mean/min | failed instances |
|---|---|---:|---|---:|---:|
| `stream4d_v7_c14_strict_owned_posttrack_probe5` | `True` | `5/5` | `summary_fallback 5/5` | `1.0 / 1.0` | `0` |
| `stream4d_v7_c15_strict_posttrack_merge_probe5` | `True` | `5/5` | `summary_fallback 5/5` | `1.0 / 1.0` | `0` |

证据链：

```text
Stream3D/logs/stream4d_v7_r6_metric_integrity_fallback_py_compile.log
Stream3D/logs/stream4d_v7_metric_integrity_c14_strict_owned_posttrack_probe5_r6.log
Stream3D/logs/stream4d_v7_metric_integrity_c15_strict_posttrack_merge_probe5_r6.log
Stream3D/outputs/audit/v7_metric_integrity_c14_strict_owned_posttrack_probe5_r6.{json,md}
Stream3D/outputs/audit/v7_metric_integrity_c15_strict_posttrack_merge_probe5_r6.{json,md}
```

## 收尾 Gate 判断

最新有效 method 结果：

| Line | AP | AP50 | AP25 | 结论 |
|---|---:|---:|---:|---|
| Stream3D same-support diagnostic `P0_scannet on S1_32f` | `0.3992127932017927` | `0.5971712938711367` | `0.7425353588266108` | 目标参考，不是 Stream4D method |
| Best dense/object baseline `P3_v6_compact` | `0.28483247256897415` | `0.5039622641509434` | `0.6719147248897401` | 当前 Stream4D-like dense/object best |
| Best v7 carrier branch `C8P/C14` | `0.04249776562268287` | `0.17857097183073392` | `0.5404251478272147` | carrier 分支 best，但远低 |
| C15 strict posttrack merge | `0.03494669836186661` | `0.14738590648397554` | `0.5024238943440352` | 收紧 merge 失败 |
| Best trackbucket dense local row `C13B/C13D compact` | `0.28382830955380606` | `0.5019341996506697` | `0.6824298780157181` | AP/AP50 未超过 m670 compact；近似 baseline/tie-order |

差距：

```text
Best dense/object baseline vs Stream3D same-support:
  AP gap  = 0.3992127932017927 - 0.28483247256897415 = 0.11438032063281853
  AP50 gap= 0.5971712938711367 - 0.5039622641509434  = 0.09320902972019326

Best v7 carrier C8P/C14 vs Stream3D same-support:
  AP gap  = 0.3567150275791098
  AP50 gap= 0.4186003220404028

C15 vs C8P/C14:
  AP delta  = -0.007551067260816258
  AP50 delta= -0.03118506534675838
  AP25 delta= -0.03800125348317946
```

判断：

```text
success=False
stream4d vs stream3d gap 没有实质缩小。
C11-C13 证明 track-bucket suppression 不能修复 dense/object baseline。
C14 证明 strict-owned-posttrack 与 C8P 完全相同。
C15 证明单纯收紧 posttrack merge 不是救点。
```

## 收尾分析

```text
1. GT diagnostic 排除了一个重要误判：
   C11/C12/C13 失败不是因为 suppression 大量误删真阳性；
   被删 candidates 的 GT IoU 很低，harmful@25/50 全为 0。

2. C14 排除了“参数没复现”的可能：
   C14 与 C8P 五个 scene npz sha256 完全一致；
   所以 C8P 当前 best 是稳定可复现的。

3. C15 排除了“posttrack over-merge 是主因”的简单解释：
   C15 将 accepted links 从 307 降到 63，但指标下降。
   这说明当前 object records 更可能同时存在 split、boundary、ranking、support ownership 多重问题。

4. r6 audit fallback 修复了审计链，而不是方法链：
   object_dict/prediction alignment 现在能在 fallback 路径上验证为 1.0/1.0；
   这增强代码审查可信度，但不改变方法性能。
```

## 收尾结论

```text
v7 计划已经给出清楚负结论：
  继续在 track-bucket threshold、posttrack merge strictness、或当前 carrier WTA 后处理上扫参，信心很低。

需要新实验计划 v8。

新计划不应再把重点放在后验删除或更严格合并阈值；
应回到 object formation 的结构设计：
  1. object-level temporal birth/death；
  2. per-frame one-to-one mask assignment before support expansion；
  3. RGB-D dense support 作为 geometry 主体；
  4. D4RT carrier 只作为 identity / track consistency backbone；
  5. 在 component merge 前建立 ownership，而不是 merge 后靠 WTA/NMS 修。
```

## 收尾 Insight

```text
这轮最有价值的 insight 不是某个指标提升，而是排除了三条诱人的错误路线：
1. 删除 track bucket 内重复项不能修 AP。
2. 收紧 posttrack merge 不能修 AP。
3. D4RT carrier-only geometry / carrier-only object backbone 不能替代 RGB-D dense geometry。

Stream4D 和 Stream3D 的差距没有缩小，原因不是一个阈值未调好。
当前证据指向：
  object support representation 和 temporal ownership model 才是核心瓶颈。
```

## 收尾证据链索引

```text
GT diagnostic:
  Stream3D/tools/diagnose_trackbucket_suppression_v7.py
  Stream3D/outputs/audit/v7_trackbucket_suppression_gt_diagnostic_representative.{json,csv,md}
  Stream3D/logs/stream4d_v7_trackbucket_suppression_gt_diagnostic_representative.log

C14:
  Stream3D/logs/stream4d_v7_c14_object_dict_fallback_py_compile.log
  Stream3D/logs/stream4d_v7_c14_strict_owned_posttrack_probe5_fixed.log
  Stream3D/logs/stream4d_v7_c14_strict_owned_posttrack_probe5_eval.log
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c14_strict_owned_posttrack_probe5_scene*_summary.json
  Stream3D/data/evaluation/scannet/stream4d_v7_c14_strict_owned_posttrack_probe5_class_agnostic.txt
  Stream3D/outputs/audit/v7_metric_integrity_c14_strict_owned_posttrack_probe5.{json,md}
  Stream3D/outputs/audit/v7_reportable_config_scan_c14_strict_owned_posttrack_probe5.{json,md,csv}

C15:
  Stream3D/logs/stream4d_v7_c15_strict_posttrack_merge_probe5.log
  Stream3D/logs/stream4d_v7_c15_strict_posttrack_merge_probe5_eval.log
  Stream3D/outputs/v7_carrier_tracklet_graph/stream4d_v7_c15_strict_posttrack_merge_probe5_scene*_summary.json
  Stream3D/data/evaluation/scannet/stream4d_v7_c15_strict_posttrack_merge_probe5_class_agnostic.txt
  Stream3D/outputs/audit/v7_metric_integrity_c15_strict_posttrack_merge_probe5.{json,md}
  Stream3D/outputs/audit/v7_reportable_config_scan_c15_strict_posttrack_merge_probe5.{json,md,csv}

r6 alignment audit:
  Stream3D/tools/verify_stream4d_metric_integrity.py
  Stream3D/outputs/audit/v7_metric_integrity_c14_strict_owned_posttrack_probe5_r6.{json,md}
  Stream3D/outputs/audit/v7_metric_integrity_c15_strict_posttrack_merge_probe5_r6.{json,md}
```

## 最新审计包 r6

```text
packet:
  stream4d_v7_code_review_packet_20260609_030044_r6.zip

sha256:
  see sibling file stream4d_v7_code_review_packet_20260609_030044_r6.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

filelist:
  stream4d_v7_code_review_packet_20260609_030044_r6_filelist.txt

zip test:
  No errors detected in compressed data of stream4d_v7_code_review_packet_20260609_030044_r6.zip.

file count:
  669
```
