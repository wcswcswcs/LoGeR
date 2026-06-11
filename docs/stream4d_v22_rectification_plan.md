# Stream4D v22 整改计划：D4RT-native 几何审计、自身重建指标与 Self-Stitch 修复

日期：2026-06-10  
面向：Codex / 后续代码实现与实验执行  
目标：彻底修复 v21.3 暴露的 self-stitch、D4RT geometry provider、occupancy persistent tube、scale-sensitive 阈值与直接 D4RT 重建评估问题。  
公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`。

---

## 0. 本轮整改的根本目标

v21.3 的最大收获不是出现了可报告方法成功，而是把旧错误和新 blocker 彻底暴露出来：

1. 旧主线把 D4RT 证据、ScanNet RGB-D / pose / mesh bridge、evaluation support、object formation 混在一起，因此不能判断 D4RT geometry、scale、outlier、materialization 和语义方法各自的问题。
2. v21.3 已经开始修正边界：`rgbd_eval` 被降级为 diagnostic-only，所有 eval-Sim3 / GT/RGB-D alignment 结果不能进入 method table。
3. v21.3 已完成真正 Stream3D 内部 `GeometryProvider` replacement diagnostic，不再只是旧 export-level adapter；但结果仍显示 D4RT geometry / materialization 与 RGB-D baseline 差距明显。
4. 当前 self-stitch / provider / occupancy 仍有 P0 级实现和诊断风险，尤其是 inlier ratio 定义、overlap matching、persistent tube id、warmstart identity、scale-sensitive radius、multi-hop Sim3 test 覆盖不足。
5. 这次必须额外直接回答：**D4RT 在 ScanNet 上的重建指标究竟如何**，而不是只通过 “D4RT 替换 Stream3D 几何后 AP 掉多少” 间接判断。

因此 v22 的总目标是：

```text
先修代码可信度和 self-stitch 基础实现；
再建立 D4RT-on-ScanNet direct reconstruction benchmark；
再重跑 D4RT 替换 Stream3D 几何；
最后才决定是否继续 semantic 4D object formation。
```

---

## 1. v21.3 当前结论与不能跳过的事实

### 1.1 方法边界修正是对的，但没有可报告 method success

v21.3 的 final report 显示：

```text
v21.3 是否得到可报告方法成功: False
Phase A guard 是否通过: True
Phase D full provider replacement 是否完成: True
Phase E/F 是否运行: False
method table: 未生成
```

这说明代码边界和 diagnostic 框架比旧版干净，但还没有证明 D4RT-native semantic 4D reconstruction 成立。v21.3 还明确记录：所有 occupancy provider configs 都是 diagnostic-only，`num_reportable_method_configs=0`，`num_uses_gt_for_prediction_and_method_result=0`。这点必须继续保持。

### 1.2 D4RT 替换 Stream3D 几何后的结果仍远低于 RGB-D baseline

v21.3 的真实 Stream3D provider replacement 结果显示：

```text
G0 RGB-D baseline:
AP/AP50/AP25 = 0.324948 / 0.497839 / 0.650992

G3 D4RT eval-Sim3:
AP/AP50/AP25 = 0.067447 / 0.170455 / 0.581163

G4 eval-Sim3 + outlier:
AP/AP50/AP25 = 0.085961 / 0.227257 / 0.616071

dense128-grid G3:
AP/AP50/AP25 = 0.140696 / 0.289128 / 0.537907

D2r4 G4 interior6:
AP/AP50/AP25 = 0.198368 / 0.315906 / 0.628571
```

当前 best diagnostic D2r4 G4 interior6 仍比 G0 低：

```text
AP drop = 0.324948 - 0.198368 = 0.126580
```

并且 D2r4 G4 interior6 仍依赖 eval-Sim3 / GT-RGBD diagnostic alignment，不能作为 method。无 eval-Sim3 的 D2r4 raw/self 只有：

```text
D2r4 G1/G2:
AP/AP50/AP25 = 0.032407 / 0.072917 / 0.625000
pre% = 0.001710
hit = 0.013862
```

这说明：当前 D4RT-native self-aligned geometry 还不能稳定替代 RGB-D / pose geometry。

### 1.3 v21.3 还没有直接回答 D4RT 在 ScanNet 上的重建质量

v21.3 主要从两类角度看 D4RT 几何：

```text
1. Stream3D provider replacement 后的 segmentation AP；
2. provider / materialization / projection hit / pre% / Sim3 residual diagnostic。
```

这仍然是间接的。v22 必须新增 **D4RT-on-ScanNet direct reconstruction benchmark**，直接报告：

```text
Depth quality
Camera / pose quality
Point cloud quality
Track / correspondence quality
Chunk self-stitch quality
Scale drift
Outlier rate
Completeness / accuracy / F-score
```

这样才能区分：

```text
D4RT 整体几何差？
D4RT 局部 outlier 多？
D4RT scale / chunk stitching 差？
D4RT geometry 可用但 Stream3D provider/materialization 不适配？
```

---

## 2. 本轮 P0 整改原则

### 2.1 非评估阶段使用 GT / RGB-D / pose / mesh 是错误

本轮必须继续强调：

```text
方法内部禁止使用：
  ScanNet depth
  ScanNet camera pose
  ScanNet mesh
  GT labels
  GT/RGB-D Sim3
  RGB-D backprojection
  mesh nearest-neighbor assignment
```

任何使用这些信息的脚本或 artifact 必须标记为：

```text
is_method_result = false
is_diagnostic_only = true
forbidden_for_method_table = true
```

这个规则不是形式问题，而是科学问题。旧 pipeline 的根本错误就是把 RGB-D/pose bridge 混进方法链路，导致我们无法判断 D4RT geometry 或 semantic 4D method 是否真的成立。

### 2.2 OpenD4RT 源码逻辑优先于自造轮子

D4RT 长序列和 occupancy dense tracking 不允许 Codex 任意重写一套不可控逻辑。实现必须参考 OpenD4RT 的源码思想：

```text
OpenD4RT anchor-clip long tracking:
  infer_track_3d.py::_make_anchor_clip_indices
  infer_track_3d.py::_infer_tracks

OpenD4RT model helper:
  src.eval.tasks._model_clip_frames
  src.eval.tasks._encode_model_memory
  src.eval.tasks._run_model_for_queries

OpenD4RT sliding-window + overlap Sim3 设计:
  vis/build_like_demo.py
  vis/build_like_demo_for_worldtrack.py
```

如果 OpenD4RT demo helper 在当前 zip 中缺失或 import 不完整，Codex 不能直接 import 坏路径，而应在 `stream4d_native/` 中重写等价 helper，并写单元测试证明行为一致。

### 2.3 chunk size 必须由 checkpoint 决定

所有 D4RT temporal chunks 必须满足：

$$
L_{chunk} \leq L_{ckpt}
$$

默认必须是：

$$
L_{chunk}=L_{ckpt}, \quad S_{chunk}=\lfloor L_{ckpt}/2 \rfloor
$$

例如：

```text
32CLIP:
  temporal_chunk_size = 32
  temporal_chunk_stride = 16
  temporal_chunk_overlap = 16

48CLIP:
  temporal_chunk_size = 48
  temporal_chunk_stride = 24
  temporal_chunk_overlap = 24
```

禁止把 `query_batch_size` 和 `temporal_chunk_size` 混用。CLI 必须使用：

```text
--temporal-chunk-size
--temporal-chunk-stride
--query-batch-size
```

### 2.4 Self-stitch 必须先可信，不能带 bug 进方法

v21.3 self-stitch 已暴露以下风险：

```text
inlier ratio 由 p90 residual 阈值定义，天然约 0.9，不是真 inlier ratio；
Phase C diagnostic 用 UV nearest-neighbor，而 provider 用 carrier_id，matching 不一致；
D5 warmstart 没有 persistent tube id，不能形成真正跨 chunk material identity；
scale-normalized multi-hop Sim3 测试不足；
self-stitch 修正 overlap union 后 AP 仍低，但还不能直接判定 self-stitch 思路失败。
```

v22 必须先修这些，再重跑 self-stitch / provider / reconstruction benchmark。

---

## 3. Phase A：补齐代码包与核心依赖

### 3.1 目标

修复 code review packet 不完整、核心 provider tests 无法独立复跑的问题。所有 geometry provider / self-stitch / native D4RT tests 必须能在干净解压目录通过。

### 3.2 必须做的代码整改

Codex 必须保证以下文件进入 code review packet：

```text
Stream3D/tools/d4rt_geometry_diagnostic.py
Stream3D/tools/materialize_d4rt_aligned_geometry_for_stream3d.py
Stream3D/stream4d_native/sim3.py
Stream3D/stream4d_native/chunk_alignment.py
Stream3D/geometry_provider/d4rt_carrier_provider.py
Stream3D/tools/native_geometry_diagnostics.py
Stream3D/tools/export_v21_3_occupancy_carrier_cache.py
Stream3D/tools/run_v21_3_stream3d_provider_replacement.py
Stream3D/tests/test_v21_3_geometry_provider.py
Stream3D/tests/test_native_chunking_and_sim3.py
```

如果 `geometry_provider/d4rt_carrier_provider.py` 需要 `fit_sim3_umeyama` 或 `_backproject_xy_world`，不要从不打包的旧工具里 import。应改为：

```text
将 Sim3 / geometry helper 移入 stream4d_native/sim3.py 和 geometry_provider/common.py；
provider 和 diagnostics 统一 import 新 helper。
```

### 3.3 必须运行

```bash
python -m py_compile evaluation/*.py stream4d/*.py stream4d_native/*.py geometry_provider/*.py tools/*.py tests/*.py
python -m unittest tests.test_v21_3_geometry_provider
python -m unittest tests.test_native_chunking_and_sim3
python -m unittest discover tests
```

### 3.4 必须记录指标

```text
clean_packet_unittest_pass
missing_core_file_count
missing_core_file_list
provider_imports_resolved
sim3_helper_single_source_of_truth
full_unittest_count
full_unittest_pass
```

### 3.5 成功标准

```text
clean_packet_unittest_pass = true
missing_core_file_count = 0
provider_imports_resolved = true
```

### 3.6 不满足条件时 Codex 先尝试

```text
如果仍缺 helper:
  把 helper 移进 stream4d_native/ 或 geometry_provider/，不要依赖旧 tools 文件。

如果 open3d 缺失导致测试失败:
  拆分 pure-python tests 和 eval-only open3d tests。
  provider self-stitch tests 不能依赖 open3d。
```

---

## 4. Phase B：Self-stitch 实现整改

### 4.1 目标

修复 v21.3 self-stitch 的 P0 诊断问题，确保 scale / alignment / matching 结论可信。

### 4.2 B1：修正 inlier ratio 伪通过问题

旧问题：

```python
threshold = percentile(residual, 90)
inliers = residual <= threshold
```

这会让 inlier ratio 天然约 0.9，不能作为真实 gate。

新实现必须报告多种 inlier：

```text
inlier_ratio_abs005
inlier_ratio_abs010
inlier_ratio_rel001
inlier_ratio_rel002
inlier_ratio_mad
residual_median
residual_p90
residual_p95
residual_mad
```

其中相对阈值定义为：

$$
\tau_{rel} = \alpha \cdot \text{scene\_scale}
$$

MAD 阈值为：

$$
\tau_{mad} = \text{median}(r) + k \cdot \text{MAD}(r)
$$

### 4.3 B2：统一 overlap matching

当前存在两套 matching：

```text
Phase C diagnostic:
  UV nearest-neighbor

Provider self-stitch:
  exact carrier_id
```

v22 必须统一成三层 matching：

```text
primary:
  stable persistent_tube_id or stable carrier_id

secondary:
  same global frame + same source pixel / source grid id

fallback:
  mutual UV nearest-neighbor + cycle consistency + appearance patch consistency
```

每个 self-stitch pair 必须记录：

```text
match_source_stable_id_count
match_source_same_source_pixel_count
match_source_mutual_uv_count
match_source_rejected_count
mutual_uv_match_ratio
stable_id_match_ratio
cycle_consistency_pass_ratio
appearance_consistency_pass_ratio
```

### 4.4 B3：D5 warmstart 必须输出 persistent tube id

旧问题：

```text
D5 overlap warmstart 能提高 coverage，但没有把 previous chunk 的 tube identity 作为 persistent tube_id 写入当前 window cache。
因此 warmstart 只是 occupancy precoverage，不是真正跨 chunk material tube identity。
```

整改：

```text
warmstart track 保留 previous persistent_tube_id；
current window 的 accepted tubes 写 persistent_tube_id；
provider self-stitch 优先使用 persistent_tube_id；
object memory 后续也使用 persistent_tube_id；
cache 中同时保存 local_carrier_id 和 persistent_tube_id。
```

必须新增字段：

```text
persistent_tube_id
parent_tube_id
warmstart_source_chunk
warmstart_source_frame
is_warmstarted
```

### 4.5 B4：补三窗口 synthetic Sim3 test

必须新增 synthetic tests：

```text
window0 canonical
window1 = Sim3_1^{-1}(canonical)
window2 = Sim3_2^{-1}(canonical)
```

验证：

```text
window1 local -> canonical
window2 local -> canonical
multi-hop compose direction
non-identity rotation
non-unit scale
translation
partial overlap
missing pair fallback
bad fit rejection
scale-normalized chain
```

### 4.6 B5：best-confidence overlap policy 继续保留，但要做 ablation

对 overlap frame，多窗口候选不能直接 union。必须比较：

```text
overlap_policy = all_window_union
overlap_policy = best_confidence
overlap_policy = lowest_residual
overlap_policy = newest_window
```

记录：

```text
candidate_source_windows_mean
selected_source_windows_mean
duplicate_window_hit_rate
projection_hit
pre%
AP if provider diagnostic
```

### 4.7 成功标准

```text
true inlier ratios available and not defined by quantile artifact
stable_id_match_ratio reported
persistent_tube_id propagated for D5
three-window synthetic tests pass
self-stitch diagnostics use same matching logic as provider
```

如果不满足：

```text
不允许继续跑 Phase E/F；
不允许宣称 self-stitch 成立或失败；
只报告 implementation blocker。
```

---

## 5. Phase C：D4RT-on-ScanNet 直接重建指标

### 5.1 目标

直接回答：

```text
D4RT 在 ScanNet 上的重建质量究竟如何？
```

不能只通过 Stream3D provider replacement 间接判断。必须建立 D4RT direct reconstruction benchmark。

### 5.2 输入与输出

输入：

```text
RGB video frames
prepared masks only for mask-aware query variants
frozen D4RT model
```

Evaluation-only 输入：

```text
ScanNet depth
ScanNet pose
ScanNet mesh
ScanNet GT instance labels
```

输出：

```text
D4RT predicted depth maps
D4RT point clouds
D4RT camera relative poses
D4RT point tracks / tube tracks
D4RT stitched scene point cloud / tube field
```

### 5.3 Reconstruction variants

必须至少跑：

```text
R0: D4RT single-chunk depth / point cloud, no stitching
R1: D4RT sliding-window point cloud, raw no stitching
R2: D4RT sliding-window self-Sim3 stitched
R3: D4RT sliding-window scale-normalized self-Sim3
R4: D4RT eval-only per-scene Sim3 aligned
R5: D4RT eval-only per-chunk Sim3 aligned
R6: D4RT occupancy-dense tubes
R7: D4RT fixed-grid dense tubes
R8: D4RT mask-aware fixed tubes
R9: D4RT mask-aware occupancy tubes
```

### 5.4 Direct depth metrics

对于每个 frame，query D4RT depth：

$$
q=(u,v,t,t,t)
$$

记录 raw depth 与 aligned depth 两套结果。

指标：

```text
AbsRel
SqRel
RMSE
RMSE_log
MAE
delta1
delta2
delta3
valid_pixel_ratio
depth_scale_raw
depth_scale_aligned
```

标准 depth threshold：

$$
\delta_i = \% \left(\max\left(\frac{d}{d^*}, \frac{d^*}{d}\right) < 1.25^i \right)
$$

必须分开报告：

```text
raw D4RT depth
median-scale aligned depth
least-squares scale-shift aligned depth
eval-only GT depth aligned depth
```

判断：

```text
如果 raw 差但 scale-aligned 好:
  scale 是主要问题。

如果 scale-aligned 仍差:
  局部几何 / depth 结构本身差。

如果 valid_pixel_ratio 低:
  D4RT visibility / query coverage 是瓶颈。
```

### 5.5 Point cloud metrics

D4RT 点云与 ScanNet mesh / RGB-D point cloud 对齐后评估。必须报告 raw、自对齐和 eval-only aligned。

指标：

```text
Chamfer-L1
Chamfer-L2
Accuracy@1cm / 5cm / 10cm / 20cm
Completeness@1cm / 5cm / 10cm / 20cm
F-score@5cm / 10cm / 20cm
Precision@5cm / 10cm / 20cm
Recall@5cm / 10cm / 20cm
normal consistency if normals available
point density
outlier rate beyond 20cm / 50cm
```

定义：

$$
\text{Accuracy}_\tau = \frac{1}{|P|} \sum_{p \in P} \mathbf{1}[\min_{g \in G}\|p-g\| < \tau]
$$

$$
\text{Completeness}_\tau = \frac{1}{|G|} \sum_{g \in G} \mathbf{1}[\min_{p \in P}\|g-p\| < \tau]
$$

$$
F_\tau = \frac{2 \cdot P_\tau \cdot R_\tau}{P_\tau + R_\tau + \epsilon}
$$

其中 $P_\tau$ 是 precision，$R_\tau$ 是 recall。

### 5.6 Camera / pose metrics

D4RT paper 可以通过查询同一 3D 点在不同 camera reference 下估计相对 camera transform。必须用 D4RT own predictions 估计 camera poses，再与 ScanNet pose 做 evaluation-only comparison。

指标：

```text
ATE after Sim3
RPE rotation
RPE translation
rotation error deg
translation direction error
scale drift over frames
pose chain drift
loop consistency if applicable
```

判断：

```text
如果 depth/point cloud 局部好但 pose drift 大:
  chunk stitching / camera consistency 是主瓶颈。

如果 pose 好但 point cloud差:
  local geometry / point prediction 是主瓶颈。
```

### 5.7 Tracking / correspondence metrics

在 ScanNet 上没有真实 dense correspondence，但可以使用 evaluation-only pseudo-correspondence：

```text
RGB-D + pose + mesh visibility 生成可见 surface correspondences
或使用 frame-to-frame depth reprojection consistency。
```

指标：

```text
2D endpoint error
PCK@1px / 3px / 5px
cycle error
visibility precision / recall
track length
out-of-frame rate
occlusion false positive rate
```

这些只用于诊断，不进入 method optimization。

### 5.8 Chunk stitching metrics

必须在 direct reconstruction benchmark 中记录：

```text
self_sim3 residual median / p90 / p95
true inlier ratios at absolute thresholds
scale per pair
scale std
accumulated scale drift
GT-Sim3 per chunk scale for comparison
self-vs-GT Sim3 delta
post-stitch Chamfer / F-score
```

### 5.9 Per-instance geometry coverage metrics

为了和后续 object formation 对齐，必须记录每个 GT instance 的 D4RT coverage：

```text
GT instance surface area
D4RT point coverage ratio
D4RT tube coverage ratio
mean nearest D4RT point distance
covered surface component count
fragmentation of D4RT coverage
coverage by object size bucket
coverage by class if labels available
```

这一步直接回答：

```text
D4RT geometry 是否覆盖了 ScanNet object surfaces？
是大物体覆盖差，还是小物体覆盖差？
是全部都差，还是局部 outlier 影响？
```

### 5.10 可视化

必须输出：

```text
D4RT depth vs GT depth heatmap
depth error heatmap
D4RT point cloud vs ScanNet mesh side-by-side
per-scene point cloud colored by residual
per-instance coverage heatmap
chunk scale drift plot
pre/post self-stitch point cloud overlay
outlier tubes overlay on video
```

### 5.11 成功标准

不是要求 D4RT 达到 RGB-D 水平，而是要求能清楚定位原因。最低成功标准：

```text
完整生成 R0-R9 benchmark 表；
每个指标分 raw / self-stitch / eval-Sim3；
每个 scene 有 direct reconstruction report；
能将失败归因为：
  scale-only,
  local outlier,
  pose/chunk drift,
  coverage insufficiency,
  materialization mismatch,
  or object-level issue。
```

数值 gate 初版：

```text
F-score@10cm >= 0.25 after eval-Sim3
Completeness@20cm >= 0.40 after eval-Sim3
Depth delta1 >= 0.50 after scale alignment
ATE / scale drift reported, not necessarily pass
per-instance covered GT ratio >= 0.50 after eval-Sim3
```

如果连 eval-Sim3 后都远低于这些：

```text
D4RT geometry itself is insufficient for ScanNet dense 3D reconstruction under current query/preprocess setting.
Do not proceed to semantic object formation on ScanNet AP.
```

如果 eval-Sim3 好但 raw/self 差：

```text
scale/reference/stitching 是主瓶颈。
优先修 self-Sim3 / canonical frame。
```

如果 raw/self geometry reasonably good but Stream3D provider still bad：

```text
provider/materialization / scale-sensitive Stream3D hyperparams 是主瓶颈。
```

---

## 6. Phase D：D4RT 替换 Stream3D 几何整改重跑

### 6.1 目标

回答：

```text
如果只替换 Stream3D 的几何源，Stream3D 性能掉多少？
这个掉分来自 D4RT 几何本身，还是来自尺度敏感超参 / provider materialization？
```

### 6.2 必须固定的 provider variants

```text
G0: Stream3D + RGB-D / pose geometry baseline
G1: Stream3D + raw D4RT local geometry
G2: Stream3D + D4RT self-stitched geometry
G3: Stream3D + D4RT scale-normalized self-stitched geometry
G4: Stream3D + eval-only scene Sim3 D4RT geometry
G5: Stream3D + eval-only chunk Sim3 D4RT geometry
G6: Stream3D + eval-only Sim3 + outlier filtering
G7: Stream3D + eval-only Sim3 + mask-interior gate
G8: Stream3D + eval-only Sim3 + density-normalized thresholds
G9: Stream3D + raw/self geometry + scale-adaptive thresholds
G10: Stream3D + eval-Sim3 geometry + scale-adaptive thresholds
```

### 6.3 Scale-sensitive Stream3D hyperparams 必须重设

替换几何后不能复用 RGB-D metric-scale 超参。必须定义基于 D4RT 点间距的自适应阈值：

```text
point_spacing_median
point_spacing_p10 / p90
local_density
object_scale_proxy
```

所有半径类超参改为：

$$
r = \alpha \cdot \text{median\_point\_spacing}
$$

需要 sweep：

```text
alpha = 0.5, 1.0, 2.0, 4.0, 8.0
```

涉及参数：

```text
NN projection radius
set-cover keypoint neighbor radius
manifold refining distance
historical merge centroid distance
component fill radius
outlier filter radius
min points per object, as density-relative threshold
```

### 6.4 必须记录

```text
AP/AP50/AP25
pre%
union%
projection_hit
empty_mask_rate
#pred
point_spacing_median
effective_radius
set_cover_keypoints
manifold_neighbor_count
component_count
merge_count
split_count
AP vs radius curve
```

### 6.5 判断标准

如果：

```text
G4/G5/G6/G7/G10 still far below G0:
```

说明即使 eval alignment + scale-aware threshold，D4RT geometry / support 仍不足。

如果：

```text
G4/G10 close to G0 but G2/G3 bad:
```

说明主要是 self-stitch / scale / canonical frame 问题。

如果：

```text
G3/G9 close to G0:
```

说明 D4RT-native geometry path 可用，可以进入 semantic method。

### 6.6 不满足条件时 Codex 先尝试

```text
1. 检查 provider 真实走完整 Stream3D internal path，不是 export-level adapter。
2. 检查 frame id / local id 混淆。
3. 检查 source window overlap policy。
4. 检查 radius 是否基于 spacing，而不是固定米制。
5. 检查 D4RT direct reconstruction benchmark 是否支持几何可用。
```

---

## 7. Phase E：Occupancy path 整改

### 7.1 目标

修复 D4 / D5 的核心问题：

```text
query 少但 coverage 低；
warmstart coverage 有效但 metric/AP 失败；
D5 没有 persistent tube identity；
accepted tubes metric consistency 差。
```

### 7.2 E1：D4/D5 不再以 query speed 为唯一目标

旧 occupancy 的问题是：

```text
semantic adaptive speedup 很高，但 query_budget_hit=true，mask coverage 不够。
```

新目标必须同时优化：

```text
query efficiency
mask interior coverage
boundary coverage
uv_in01
visible length
provider projection hit
```

### 7.3 E2：coverage-aware query budget

对于每个 scene，不能固定 4096/7168/8192 后硬停。要按 coverage target 自适应：

```text
while coverage_target_not_met and query_budget_not_exceeded:
    query more
```

targets：

```text
mask_interior_coverage >= 0.15 on probe5
mask_boundary_coverage >= 0.22 on probe5
uv_in01_rate >= 0.45 minimum
visible_track_length >= 0.20 minimum
```

注意：这些 gate 只是进入 provider diagnostic 的最低线，不是最终成功标准。

### 7.4 E3：persistent tube warmstart

D5 必须实现：

```text
previous tube -> current overlap -> same persistent_tube_id
```

并记录：

```text
persistent_tube_retention_rate
warmstart_tube_acceptance_rate
warmstart_metric_residual
warmstart_uv_consistency
warmstart_rejected_bad_geometry
```

### 7.5 E4：D5 accepted tube metric consistency gate

D5 accepted tubes 在进入 provider 前必须通过：

```text
overlap self-Sim3 residual gate
local neighbor stretch gate
uv cycle gate
visibility length gate
```

否则不能写入 provider cache。

### 7.6 必须记录

```text
actual_queries
query_budget_hit
adaptive_speedup
mask_interior_coverage
mask_boundary_coverage
uv_in01
visible_length
persistent_tube_count
persistent_tube_retention
provider_projection_hit
provider_AP_if_eval
```

### 7.7 成功标准

```text
D5 coverage >= D2r4 with fewer or comparable queries
D5 uv_in01 not worse than D2r4 by more than 0.1
D5 provider G3/G4 >= D2r4 provider G3/G4
D5 raw/self G1/G2 improves over D2r4 raw/self
```

如果 D5 coverage 高但 AP 低：

```text
accepted tubes have metric inconsistency;
increase geometry gate or reject cross-window warmstart.
```

---

## 8. Phase F：重新判定是否可以进入语义方法

只有满足以下条件才允许进入 TubeCover / object formation：

```text
Phase A pass
Phase B self-stitch implementation fixed
Phase C direct D4RT reconstruction benchmark completed
Phase D provider replacement rerun completed
Phase E occupancy/persistent tube identity fixed or clearly rejected
```

进入语义方法的最低条件：

```text
D4RT direct reconstruction after self-stitch has reasonable direct geometry metrics
or eval-Sim3 direct reconstruction shows geometry is structurally usable
and raw/self provider AP is not near zero
and D4RT-real geometry clearly beats shuffled/no-temporal controls
```

如果不满足：

```text
暂停 semantic 4D object formation；
报告 D4RT geometry limitation；
切换到 D4RT correspondence-only semantic tracking diagnostic，而不是 3D AP。
```

---

## 9. 本轮必须生成的表格

### 9.1 Self-stitch code audit table

| item | value |
|---|---|
| clean tests pass |  |
| stable id matching implemented |  |
| persistent tube id implemented |  |
| inlier ratio fixed |  |
| three-window Sim3 test pass |  |
| provider/diagnostic matching unified |  |

### 9.2 D4RT direct reconstruction table

| variant | depth AbsRel | depth δ1 | Chamfer | F@5cm | F@10cm | completeness@20cm | pose ATE | scale drift | outlier@20cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 9.3 D4RT geometry replacement table

| variant | AP | AP50 | AP25 | pre% | hit | empty | spacing | radius | #pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 9.4 Occupancy correction table

| variant | queries | speedup | interior | boundary | uv | visible | persistent retention | provider AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

### 9.5 Failure attribution table

| failure type | evidence | next action |
|---|---|---|
| overall geometry bad | direct reconstruction low after eval-Sim3 | stop 3D AP method |
| local outliers | high outlier residual but good median | improve filtering |
| scale drift | eval-Sim3 good but self bad | fix stitching |
| materialization/provider | direct geometry good but provider bad | fix provider / thresholds |
| occupancy coverage | fixed better than occupancy | fix query policy |
| semantic grouping | geometry good but object AP bad | proceed TubeCover |

---

## 10. 可视化要求

Codex 必须输出：

```text
self-stitch residual histogram
self-stitch scale drift over chunks
stable-id vs UV-fallback match overlay
D4RT depth vs GT depth heatmaps
D4RT point cloud vs ScanNet mesh
point cloud residual color map
per-GT-instance D4RT coverage heatmap
provider projection hit visualization
radius/AP curve plots
occupancy coverage progression per round
persistent tube id timeline across chunks
```

---

## 11. 本轮 Stop Rules

### Stop A：代码包不完整

```text
clean packet tests cannot run
```

停止实验，先补包。

### Stop B：self-stitch metrics 仍不可信

```text
inlier ratio still quantile-defined
provider matching differs from diagnostic matching
persistent tube id absent in warmstart
```

停止 provider / method 结论。

### Stop C：direct D4RT reconstruction eval-Sim3 后仍极差

```text
F@10cm < 0.10
depth δ1 < 0.30
completeness@20cm < 0.20
```

停止 ScanNet 3D AP 主线，改为报告 D4RT geometry limitation。

### Stop D：eval-Sim3 direct reconstruction 好但 self-stitch 坏

```text
direct R4/R5 good, R2/R3 bad
```

继续修 scale / stitching，不进入语义。

### Stop E：direct reconstruction 好但 provider AP 坏

```text
D4RT point cloud F-score acceptable, but Stream3D provider AP low
```

修 provider、scale-sensitive thresholds、set-cover universe，不归因给 D4RT geometry。

### Stop F：occupancy 比 fixed 差

```text
D4/D5 coverage and provider AP lower than D2r4
```

不得 claim occupancy dense path 成功；回到 OpenD4RT occupancy logic 检查 visited update、query priority 和 persistent identity。

---

## 12. 本轮完成后应能回答的问题

这次整改后，必须能明确回答：

1. self-stitch 代码到底有没有 bug？
2. overlap matching 是否真的是同一物理点？
3. D5 warmstart 是否产生 persistent material tube identity？
4. D4RT 在 ScanNet 上的 depth / point cloud / pose / track 直接重建指标是多少？
5. D4RT 几何是整体差、局部 outlier 差、scale drift 差，还是 provider/materialization 差？
6. D4RT 替换 Stream3D RGB-D 几何后，在 scale-aware 超参下掉多少？
7. eval-Sim3 能否显著救 D4RT geometry？
8. 如果 eval-Sim3 能救但 self-stitch 不能救，scale / canonical alignment 要怎么修？
9. 如果 eval-Sim3 也不能救，D4RT geometry 在 ScanNet 上是否不适合做 dense 3D AP？
10. 我们是否有资格进入 semantic 4D object formation？

只有这些回答清楚，才允许继续写新方法。
