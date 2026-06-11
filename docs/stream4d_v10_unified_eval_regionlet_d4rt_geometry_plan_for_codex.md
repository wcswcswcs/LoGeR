# Stream4D v10：统一评估、算法重构与 D4RT 几何归因实验计划书

面向 Codex 的执行计划。本文档不再把目标定义为“Stream3D + D4RT”，也不再继续围绕 top-k、NMS、WTA、score sweep、support completion 这类后处理堆叠。当前实验已经足够说明：如果基本 object primitive 和 object formation 机制不改，静态 ScanNet 都无法稳定接近 Stream3D，更不可能支撑 feed-forward semantic 4D reconstruction and tracking。

本文所有公式均使用 Typora 友好的 `$...$` 或 `$$...$$` 格式，不使用 display 公式的方括号语法。

---

## 0. 本计划的硬约束

从 v10 开始，任何结果都必须同时报告两类评估：

1. **paper-style own-support / recompute**  
   这是原版 Stream3D-style cropped-TMP 主评估方式。每个方法使用自己的 prediction 和自己的 `TMP/<config>/<scene>_pre_points.npy`。这对应论文主表的“每个方法自己的输出 universe”。

2. **same-support / cross-support diagnostic**  
   这不是 Stream3D paper 的主指标，但它是判断方法是否靠缩小 support 获利的必要诊断。每个方法必须和 Stream3D 在同一 support 上比较。否则不能判断“到底和 Stream3D 差多少”。

每个 method config 至少输出四行：

```text
M own:
  prediction = M
  pre_points = M

Stream3D on M:
  prediction = Stream3D baseline
  pre_points = M

M on Stream3D:
  prediction = M
  pre_points = Stream3D baseline S0

M on S1 historical sparse support:
  prediction = M
  pre_points = S1 historical 32f support
```

如果方法由 parent config 后处理得到，还必须输出：

```text
M inherit parent:
  prediction = M
  pre_points = parent
```

禁止只汇报 own-support 高分。所有表格必须包含以下字段：

```text
AP
AP50
AP25
eval_policy
prediction_config
pre_points_config
pre_points %
prediction union %
union in target scene %
union in target pre_points %
GT crop/full
#pred
mean points/object
conflict rate
tiny mask ratio <100
large mask ratio >1000
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per matched GT
runtime
manifest_integrity_pass
```

所有 method result 的 manifest 必须包含：

```text
uses_gt_for_prediction = false
uses_gt_for_diagnostic = false
is_method_result = true
is_diagnostic_only = false
eval_policy = own_recompute_paper_style | cross_fixed_support | inherit_parent_support
support_source = own | stream3d_s0 | stream4d_s1 | parent | named_config
geometry_source = rgbd_eval_bridge | d4rt_sim3_aligned | d4rt_raw | mixed
```

如果某个工具读取 GT 只是为了诊断，manifest 必须写：

```text
uses_gt_for_prediction = false
uses_gt_for_diagnostic = true
is_method_result = false
is_diagnostic_only = true
```

普通 evaluator/report scanner 必须拒绝 `uses_gt_for_prediction=true` 或 `is_diagnostic_only=true` 的结果进入 method table。

---

## 1. 当前结果的独立判断

### 1.1 评估协议本身不是假，但历史汇报方式造成了严重误判

原版 Stream3D paper-style 评估使用每个配置自己的 cropped-TMP support。因此 `recompute_pre_points` 对应 paper-style own-support 主协议。`inherit_pre_points` 和 cross-support 是我们新增的诊断协议，不是原论文主表协议。

但是，过去大量结果只报 own-support，导致看起来像超过 Stream3D。历史数据说明这会误导判断。

v3 full ScanNet 上，Stream4D adaptive 在 own-support 下略高：

```text
Stream4D v3 adaptive own:
20.3718 / 35.5222 / 55.0649

Stream3D baseline own:
20.1139 / 34.4654 / 50.2268
```

但同一 Stream4D prediction 在 inherited MVP support 下变成：

```text
Stream4D v3 adaptive inherit:
12.2851 / 23.3147 / 41.6773
```

更关键的是，同一 sparse adaptive support 上，Stream3D 反而明显更强：

```text
Stream3D on Stream4D adaptive support:
32.6669 / 50.1160 / 67.6487
```

这说明 v3 的提升不是 object quality 胜利，而是 paper-style support shrinking 下的 high-precision sparse proposal 结果。

v9 Day0 也给出相同结论。B1 在自己的 S2 support 上：

```text
B1 own S2:
0.328439 / 0.629266 / 0.884363

Stream3D on S2:
0.326714 / 0.496778 / 0.726638
```

这说明 B1 在 tiny own support 上确实有 AP50/AP25 正信号；但 B1 放到 Stream3D S0/S1 support 上几乎崩掉：

```text
B1 on S0:
0.000635 / 0.004294 / 0.010768

B1 on S1:
0.016837 / 0.047534 / 0.168162
```

因此以后所有结果必须同时给出 own-support 和 same-support。

### 1.2 现在不是单纯 support 不够，而是 object hypothesis 本身错了

早期问题主要是 support 小。现在 v9 O29-O38 已经证明，单纯扩大 support 也不能解决。

O38 c055 own support 已经达到约 66% scene vertices，但 own AP 只有：

```text
O38 c055 own:
0.0810 / 0.2192 / 0.4925
```

O38 c055 放到 S0 上：

```text
O38 c055 on S0:
0.033012 / 0.123089 / 0.392066
```

而 Stream3D on S0 是：

```text
Stream3D on S0:
0.235730 / 0.414306 / 0.537786
```

这说明当前主要问题不是“没有点”，而是“有了大量点仍然不能组成正确 object”。当前方法的失败模式是：

```text
small clean support -> own AP 高，但不是完整 reconstruction；
large noisy support -> coverage 上来，但 object boundary / one-to-one assignment 崩；
postprocess memory -> 可以微调 trade-off，但不能改变 object primitive 错误。
```

所以 v10 不能继续主推 support completion、scene object memory threshold sweep、logarea score、zero-conflict exclusivity 这类后处理。它们只能移动 AP/AP50/AP25 的权衡，不能解决 object formation。

### 1.3 当前算法的错误假设

当前做法有四个关键假设是错的：

**错误假设 A：2D mask 可以直接当 object primitive。**  
2D VFM mask 是 noisy measurement。它可能跨多个物体，也可能只覆盖物体一部分。B1 的 single-mask ownership 高分说明“clean mask birth”是有用信号，但 full-span 多 mask 结果爆炸说明“mask as object”不是完整解。

**错误假设 B：D4RT carrier 落在同一 mask 就应该合并。**  
同 mask co-membership 是弱正证据，不是强 identity。一个坏 mask 会把多个 object 的 carriers 连成 clique。v7 carrier-tracklet C3/C4 已经证明：C3 full fringe 冲突率约 0.97，C4 WTA 能把冲突清零，但 AP/AP50 仍只有 0.0205/0.0797。

**错误假设 C：先扩 full-mask fringe，再用 WTA 修复。**  
这等于先污染再清理。WTA 只能决定冲突点最后给谁，不能把跨物体 mask 切开，也不能恢复正确边界。

**错误假设 D：support 扩大后自然接近 Stream3D。**  
O38 反例已经说明，support 可以很大但 AP 很低。扩大 support 必须建立在 region-level ownership 正确的基础上。

### 1.4 现在正确的算法方向

v10 的目标不是“Stream3D + D4RT”。Stream3D 只作为 baseline、评估协议参照和 negative control。新的方法必须从 object primitive 开始改：

```text
full 2D mask as object
  -> 错

sparse carrier component as object
  -> 太稀疏，且 co-membership 容易错误传播

regionlet / surfel-supported object primitive
  -> v10 主线
```

新的表示应该是：

```text
D4RT semi-dense surfels provide temporal/material evidence.
2D masks provide noisy semantic measurements.
RGB-D/mesh is used only for ScanNet evaluation bridge and regionlet geometry diagnostics.
Object is inferred as a partition/ownership field over regionlets/surfels, not as a connected component of masks.
```

---

## 2. v10 总体目标

v10 的总体目标是建立一个能真正推进的最小闭环：

```text
D4RT semi-dense surfel tracks
+ 2D VFM mask measurements
+ regionlet-level object birth
+ ownership-before-support-expansion
+ unified own/cross support evaluation
+ D4RT geometry attribution experiment
```

v10 不要求一步达到 full ScanNet 超越，但必须满足三个硬目标：

1. **评估目标**：每个实验都能明确回答“与 Stream3D 到底差多少”。  
2. **算法目标**：证明 regionlet/surfel ownership 比 full-mask object 或 sparse-carrier component 更接近正确 object formation。  
3. **归因目标**：通过“D4RT 几何对齐到 GT 几何后给 Stream3D 使用”的实验，分清 D4RT 几何精度到底对结果有多大影响。

v10 的主结论只能来自以下 gate：

```text
Probe5 Gate A:
  own-support AP >= B1 或 O1 的对应强项；
  S0 AP 明显超过 O38 c055 on S0；
  S1 AP 明显超过 B1/O1 on S1；
  Stream3D-on-method-support 不再显著高于 method own。

Probe5 Gate B:
  method on S0 AP >= 0.08
  method on S0 AP50 >= 0.18
  method on S0 AP25 >= 0.45
  method on S1 AP >= 0.18
  method on S1 AP50 >= 0.35
  method on S1 AP25 >= 0.60

Tune30 Gate:
  如果 probe5 gate 通过，固定参数跑 tune30；
  own and cross-support 不能出现 probe5 好、tune30 崩。

Final Gate:
  final split 只允许跑一次 locked config；
  不允许 final split 继续调阈值。
```

这些 gate 比最终超 Stream3D 低很多，但它们能证明方向是否真的前进。

---

## 3. 每轮提交给人工审核的代码包要求

Codex 每轮必须提交完整审计包，不允许只提交部分脚本或只提交结果表。审计包必须包含：

```text
stream4d_v10_<phase>_code_review_packet.zip
stream4d_v10_<phase>_code_review_packet.sha256
stream4d_v10_<phase>_filelist.txt
stream4d_v10_<phase>_ziptest.log
```

zip 内必须包含：

```text
Stream3D/tools/*.py
Stream3D/stream4d/*.py
Stream3D/evaluation/evaluate.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_*.sh
Stream3D/scripts/*.json
Stream3D/docs/stream4d_v10_执行日志.md
Stream3D/docs/stream4d_v10_实验结果复盘.md
Stream3D/outputs/audit/*.json
Stream3D/outputs/audit/*.md
Stream3D/outputs/audit/*.csv
Stream3D/data/evaluation/scannet/*_class_agnostic.txt
Stream3D/data/prediction/<new_configs>_class_agnostic/config_manifest.json
Stream3D/data/TMP/<new_configs>/config_manifest.json
probe5 prediction/TMP npz/npy artifacts sufficient for rerun
```

每轮必须提供并通过：

```text
python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
python -m unittest discover tests
python -m tools.scan_reportable_configs --require-manifest --require-eval-policy
python -m tools.verify_stream4d_metric_integrity --require-manifest
```

如果本地审计环境缺少 `open3d` 或某些库，测试必须拆分为：

```text
pure python tests
open3d-required tests
gpu-required tests
```

不能让 import 阶段因为 optional dependency 缺失而导致所有 protocol tests 不可运行。

---

## 4. Phase 0：统一评估矩阵与差距确认

### 4.1 目标

在继续任何算法实验前，先把当前 best、Stream3D baseline、B1/O1/O38、v6 compact 的差距矩阵固定下来，作为 v10 的统一比较基准。该阶段不改方法，只修 eval/report 工具。

### 4.2 假设

$H_0$：当前最大的混乱来自结果汇报不统一。只要强制 own-support 与 cross-support 同时报，就能准确判断每个方法是否真的接近 Stream3D。

### 4.3 实验设置

固定支持集：

```text
S0 = Stream3D original ScanNet support
S1 = historical 32f support
S2 = B1 support
S3 = O1/O2 core support
S4 = O38 support
S5 = v6 compact support
```

固定 prediction：

```text
P0 = Stream3D baseline
P1 = v6 compact
P2 = B1 surfacelet singlemask
P3 = O1 core-only
P4 = O3 WTA-negative fringe
P5 = O38 c055 logarea
P6 = O38 c075split logarea
```

Codex 运行 `tools/evaluate_cross_prepoints.py` 生成完整 matrix：

```text
prediction P_i x support S_j
```

但 method report table 只允许包含：

```text
P_i own
P0 on S_i
P_i on S0
P_i on S1
P_i inherit parent, if applicable
```

完整 matrix 作为 diagnostic appendix。

### 4.4 必须记录的指标

```text
AP / AP50 / AP25
pre_points %
prediction union %
union in target scene %
union in target support %
GT crop/full
#pred
points/object mean and median
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per GT
support conflict rate
tiny mask ratio
large mask ratio
```

### 4.5 判断标准

Phase 0 通过标准：

```text
1. Stream3D self-inherit 与 Stream3D baseline 完全一致或差异 < 1e-6。
2. 所有 method config 都有 eval_policy。
3. 所有 matrix 行都能追溯 prediction_config 和 pre_points_config。
4. 复盘文档明确列出每个 method 相对 Stream3D 的 own-support gap 和 same-support gap。
```

如果不满足：

```text
先修 evaluator / manifest / pre_points materialization。
不要继续方法实验。
```

### 4.6 可视化

输出：

```text
outputs/audit/v10_gap_matrix_heatmap_AP.png
outputs/audit/v10_gap_matrix_heatmap_AP50.png
outputs/audit/v10_support_ratio_bar.png
outputs/audit/v10_gt_crop_full_bar.png
outputs/audit/v10_method_vs_stream3d_same_support_delta.png
```

---

## 5. Phase 1：D4RT 几何对齐到 GT 几何后给 Stream3D 使用

这是用户特别要求的实验。该实验是**几何归因诊断**，不是主方法。它回答：

```text
如果把 D4RT 几何通过 GT/RGB-D anchors 对齐到 ScanNet metric geometry，然后让原版 Stream3D 使用这个几何，会掉多少？
```

这个实验的目的不是证明 D4RT 几何一定能替代 RGB-D，而是分解误差：

```text
D4RT geometry error
vs
object formation error
vs
evaluation/export adapter error
```

### 5.1 背景和动机

历史 v7 中，直接用 D4RT geometry 失败：

```text
G1/G3 default = nan, 0 objects
radius 0.50 = nan, 仍 0 objects
radius 0.50 + min_points 1 = 0 AP
```

但这不能直接证明 D4RT 几何无用，因为当时存在两个问题：

```text
1. 直接复用了 RGB-D meter-scale 超参，例如 NN radius / min_points / manifold threshold。
2. stride-10 clip 后来被证明会破坏 D4RT temporal scale，连续帧修复后 uv/cycle 明显改善。
```

因此 v10 必须重新做一个公平的几何归因实验：先把 D4RT 几何通过 GT/RGB-D anchors 对齐到 ScanNet geometry，再用适配后的 metric coordinate 给 Stream3D 使用。

### 5.2 核心假设

$H_{G1}$：若 D4RT 几何经过 robust Sim3 对齐后仍导致 Stream3D AP 大幅下降，则 ScanNet 上的主瓶颈之一是 D4RT metric geometry 精度；若下降很小，则过去 D4RT geometry 失败主要来自 scale/threshold/export 适配错误。

$H_{G2}$：D4RT 的 image-space correspondence 可用，不等价于 metric geometry 可用。因此必须同时报告 uv/cycle 指标和 metric residual，不能用一个代替另一个。

### 5.3 实验变体

Codex 实现一个新的 geometry adapter：

```text
tools/materialize_d4rt_aligned_geometry_for_stream3d.py
```

输入：

```text
ScanNet RGB frames
ScanNet GT depth/pose/intrinsics, only for geometry alignment diagnostic
D4RT model/checkpoint
CropFormer 2D masks
Stream3D original pipeline
```

输出：

```text
data/scannet_d4rt_aligned/<scene>/depth_like/*.npy
data/scannet_d4rt_aligned/<scene>/pointcloud/*.npy
data/scannet_d4rt_aligned/<scene>/poses_or_reference.json
data/scannet_d4rt_aligned/<scene>/geometry_manifest.json
```

必须比较以下几组：

```text
G0: Stream3D original RGB-D/pose geometry
G1: Stream3D + D4RT raw geometry, no scale adaptation
G2: Stream3D + D4RT scene-level Sim3 aligned geometry
G3: Stream3D + D4RT window-level Sim3 aligned geometry
G4: Stream3D + D4RT scene-level Sim3 + scale-normalized thresholds
G5: Stream3D + D4RT window-level Sim3 + scale-normalized thresholds
G6: Stream3D + D4RT image-space correspondence but RGB-D metric geometry, diagnostic only
```

G0 是原版 baseline。G1 是故意保留的 negative control。G2/G3 是核心实验。G4/G5 用来测试“对齐后仍需阈值适配”的影响。G6 判断 D4RT tracking signal 与 D4RT metric geometry 的贡献能否拆开。

### 5.4 D4RT-to-GT Sim3 对齐

对每个 scene 或 window，构建 anchor pairs：

```text
x_i^D4RT = D4RT query output in reference t_cam
x_i^GT = ScanNet RGB-D point in world coordinate
```

只使用无 GT instance/semantic 的几何 anchors：

```text
valid depth
valid pose
D4RT visibility high
D4RT confidence high
uv in range
optional low-motion/static-like anchors
```

估计：

$$
T^*=(s,R,t)=\arg\min_{s,R,t}\sum_i \omega_i\|sRx_i^{D4RT}+t-x_i^{GT}\|_2^2
$$

其中 $\omega_i$ 来自：

```text
D4RT visibility
D4RT confidence
depth validity
cycle consistency
self reprojection consistency
```

禁止：

```text
使用 GT instance labels 选择 anchors
使用 semantic labels 选择 anchors
根据 AP 选择 Sim3 variant
把 Sim3 residual 反馈给 method object selection
```

允许：

```text
使用 GT depth/pose 作为几何对齐 anchor，因为这是本实验的目的：隔离几何精度影响。
```

### 5.5 Stream3D 如何使用 D4RT-aligned geometry

Codex 需要把原版 Stream3D 中依赖 RGB-D/pose 生成 point cloud 的入口抽象为 geometry provider：

```text
GeometryProviderRGBD
GeometryProviderD4RTAligned
```

对于 `GeometryProviderD4RTAligned`，需要提供：

```text
per-frame local point cloud P^t
accumulated point cloud P^{1:t}
2D pixel -> 3D point mapping
3D point -> scene vertex or unified point id mapping
scale metadata
point density metadata
```

如果无法完全替换内部 point cloud，先实现最小可行替换：

```text
只替换 2D mask projection 到 3D mask 的几何点；
Stream3D set-cover / merging / manifold 使用这些 D4RT-aligned 3D masks；
evaluation 用与 G0 相同的 cropped-TMP policy 和 cross-support matrix。
```

### 5.6 超参适配原则

因为 D4RT 几何原始尺度不稳定，G1 直接复用 RGB-D 超参只是 negative control。G2/G3 Sim3 对齐后可以使用 meter-scale Stream3D 原超参，但仍必须检查 point density 差异。

G4/G5 使用 scale-normalized / density-normalized 阈值：

```text
nn_radius = q75(nearest_neighbor_distance) * alpha
manifold_delta = q90(intra_mask_point_spacing) * beta
min_points_per_object = max(1, round(rgbd_min_points * d4rt_density_ratio))
fps_keypoint_count = min(original_count, available_d4rt_points)
```

所有超参不能在 final 上调。先在 probe5 和 tune30 选择，再 locked 到 final。

### 5.7 必须记录的指标

几何指标：

```text
anchor_count
Sim3 scale
Sim3 rotation determinant
translation norm
inlier ratio
median residual
p90 residual
p95 residual
per-frame residual median
per-window scale drift
self UV error p90
cycle UV error p90
uv_in01_rate
track_length_visible_mean
D4RT point density
D4RT point spacing q25/q50/q75/q90
GT depth RMSE after Sim3
GT depth AbsRel after Sim3
mask projection hit rate
```

Stream3D pipeline 指标：

```text
num_2d_masks
num_3d_masks_after_projection
3D mask area distribution
empty_projected_mask_ratio
set-cover selected mask count
manifold refining removed point ratio
historical merge count
new/dynamic/static division counts
```

AP 指标：

```text
AP / AP50 / AP25 own-support
Stream3D original on G_i support
G_i on Stream3D S0 support
G_i on S1 support
pre_points %
prediction union %
GT crop/full
#pred
```

### 5.8 判断标准

该实验的核心判断不是必须超过 Stream3D，而是定量回答“D4RT 几何精度影响多大”。

几何可接受标准：

```text
median residual < 0.15m
p90 residual < 0.35m
uv_in01_rate > 0.95
cycle p90 < 5 px
empty_projected_mask_ratio < 30%
```

Stream3D-D4RT geometry 可用标准：

```text
G2/G3 相对 G0 AP drop <= 3 points
G2/G3 相对 G0 AP50 drop <= 5 points
G2/G3 相对 G0 AP25 drop <= 7 points
```

如果 drop 大于上述标准：

```text
结论：D4RT-aligned metric geometry 当前不足以替代 ScanNet RGB-D/pose 运行 Stream3D。
后续主方法不能 claim ScanNet metric geometry replacement。
D4RT 只能作为 correspondence / ownership evidence 继续研究。
```

如果 G2/G3 接近 G0：

```text
结论：过去 D4RT geometry failure 主要来自 scale/threshold/export 适配。
后续可把 D4RT metric geometry 纳入 method ablation。
```

如果 G4/G5 明显优于 G2/G3：

```text
结论：D4RT geometry 需要 density-normalized thresholds。
后续所有 D4RT geometry 实验禁止复用 RGB-D fixed meter-scale 超参。
```

### 5.9 可视化

每个 scene 必须输出：

```text
D4RT aligned point cloud vs ScanNet mesh overlay
Sim3 residual heatmap on image and mesh
per-frame projected D4RT points on RGB
2D mask -> RGB-D 3D mask vs D4RT-aligned 3D mask overlay
set-cover selected masks comparison
manifold refining before/after point cloud
failure cases: high residual, empty projection, over-merged masks
```

### 5.10 不满足条件时 Codex 先尝试

如果 Sim3 residual 大：

```text
检查 D4RT t_cam reference 是否固定。
检查 frame indexing 是否和 ScanNet frame id 一致。
检查 normalized UV 到 pixel coordinate 的转换。
检查 coordinate axis convention。
尝试 scene-level Sim3、window-level Sim3、frame-pair Sim3。
过滤低 cycle consistency anchors。
只使用 background-like low-motion anchors。
```

如果 residual 可接受但 AP 仍低：

```text
检查 D4RT point density 是否太稀疏。
降低 min_points_per_object。
用 density-normalized manifold delta。
检查 mask projection empty ratio。
比较每个 2D mask 的 RGB-D projected 3D mask 与 D4RT projected 3D mask IoU。
```

如果 G4/G5 仍 0 AP：

```text
停止把 D4RT metric geometry 当 ScanNet 主路径。
回到 image-space/surfel ownership 方法，不继续浪费时间。
```

---

## 6. Phase 2：Regionlet object birth，替代 full-mask object birth

### 6.1 目标

当前最根本的问题是 object primitive 错误。v10 Phase 2 把 object birth 从 full 2D mask 改为 regionlet。Regionlet 是一个比 full mask 更小、更可验证、更不容易跨 object 的局部区域。

### 6.2 假设

$H_{R1}$：把 2D mask 先拆成 regionlets，再做 object birth，比直接 full-mask backprojection 更能减少跨物体污染，同时不牺牲太多 recall。

$H_{R2}$：regionlet 的边界由 RGB-D geometry、2D mask distance transform、D4RT surfel seeds 共同定义时，比只用 connected component 或只用 carrier seed-near 更稳。

### 6.3 Regionlet 定义

对每个 frame $t$ 的 2D mask $m_{t,r}$，生成 regionlets：

$$
m_{t,r} \rightarrow \{r_{t,r,1}, r_{t,r,2}, \dots, r_{t,r,n}\}
$$

每个 regionlet 是满足以下条件的局部区域：

```text
位于同一个 2D mask 内；
在 depth/normal 上近似连续；
包含或接近若干 D4RT surfel seeds；
离 mask boundary 有可记录距离；
面积在合理范围内；
不跨明显 depth discontinuity。
```

第一版实现三种 regionlet 生成方式：

```text
R0: full-mask baseline
R1: mask connected component + distance transform core
R2: depth/normal discontinuity split
R3: D4RT surfel-seeded watershed inside mask
R4: R2 + R3 combined
```

### 6.4 Regionlet scoring

每个 regionlet 记录：

```text
area_2d
area_3d
num_d4rt_surfels
surfel_density
mean_distance_to_mask_boundary
depth_variance
normal_variance
rgb_feature_variance
mask_confidence
temporal_observation_count
```

Regionlet 初始质量分：

$$
q(r)=
w_s\log(1+n_{surfel})
+w_b d_{boundary}
-w_d \sigma_{depth}
-w_n \sigma_{normal}
-w_c conflict(r)
$$

第一版不要学习权重，使用固定归一化。

### 6.5 实验设置

在 probe5 上比较：

```text
R0 full-mask object birth
R1 mask-core regionlet
R2 depth-split regionlet
R3 D4RT-seeded regionlet
R4 combined regionlet
```

每个 variant 都必须输出：

```text
own-support
Stream3D on method support
method on S0
method on S1
```

不能只看 own AP。

### 6.6 必须记录的指标

```text
num_regionlets/frame
num_regionlets/mask
regionlet area distribution
regionlet depth variance
regionlet surfel count
regionlet boundary distance
regionlet conflict rate before object merge
empty_regionlet_ratio
over-small_regionlet_ratio
regionlet-to-object assignment count
AP / AP50 / AP25, all eval policies
per-GT best IoU mean
GT IoU>=0.25 / IoU>=0.50 count
duplicate predictions per GT
```

### 6.7 判断标准

R-phase 成立需要：

```text
R4 on S0 AP > O38 c055 on S0 by at least +0.03 absolute
R4 on S1 AP > B1/O1 on S1 by at least +0.05 absolute
R4 own AP50 not lower than B1 by more than 0.08
Stream3D on R4 support - R4 own AP gap < 0.10
regionlet conflict rate < full-mask conflict rate by at least 30%
```

如果只提升 AP25，不提升 AP/AP50：

```text
说明 regionlet 增加了粗 recall，但没有改善高 IoU instance quality。
Codex 下一步必须改 ownership model，而不是继续扩 regionlet。
```

如果 S0/S1 仍接近 0：

```text
说明 regionlet 仍只是 tiny support clean subset。
需要增加 object recall，而不是继续修边界。
```

### 6.8 可视化

每个 scene 保存：

```text
RGB + original 2D masks
RGB + regionlets colored
D4RT surfel seeds over regionlets
depth discontinuity overlay
regionlet accepted/rejected overlay
final object support overlay
missed GT objects list
false positive examples
same 2D mask split into multiple regionlets examples
```

### 6.9 不满足条件时 Codex 先尝试

如果 regionlets 太碎：

```text
合并相邻 regionlets，但必须满足 depth/normal/surfel ownership consistency。
调高 minimum area。
使用 mask distance transform core 作为 merge seed。
```

如果 regionlets 仍跨 object：

```text
提高 depth discontinuity sensitivity。
加入 D4RT surfel motion/trajectory incompatibility split。
按 same-frame competing object seeds 做 watershed。
```

如果 AP50 下降：

```text
regionlet 太保守，fringe recovery 需要加入但必须在 ownership 后进行。
不要回到 full-mask backproject。
```

---

## 7. Phase 3：Surfel ownership posterior，而不是 connected component

### 7.1 目标

把 D4RT surfels 从“mask selector”变成 object ownership evidence。2D mask 只作为 measurement，不直接产生 object。对象是 regionlets/surfels 的 ownership posterior。

### 7.2 假设

$H_{S1}$：D4RT surfel tracks 对 object ownership 有真实贡献，并且这种贡献不能被 area/random/no-track controls 解释。

$H_{S2}$：把 surfel assignment 建模成 posterior，比 carrier co-membership connected component 更稳定。

### 7.3 表示

每个 D4RT surfel：

$$
s_i=\{X_i(t), \pi_i(t), v_i(t), c_i(t), f_i(t), z_i(t)\}_{t=1}^{T}
$$

其中：

```text
X_i(t): D4RT predicted 3D position
π_i(t): projected 2D position
v_i(t): visibility
c_i(t): confidence
f_i(t): image/appearance feature
z_i(t): mask/regionlet observation
```

每个 object slot $O_k$ 维护：

$$
p(y_i=k \mid \mathcal{Z}, \mathcal{G}, \mathcal{A})
$$

其中 $\mathcal{Z}$ 是 2D mask/regionlet observations，$\mathcal{G}$ 是 D4RT geometry/correspondence，$\mathcal{A}$ 是 appearance evidence。

### 7.4 Ownership evidence

Positive evidence：

```text
surfel consistently appears in regionlets owned by object slot
surfel trajectory stays spatially coherent with object core
surfel appearance/crop feature consistent with object memory
surfel visibility supports temporal continuation
```

Negative evidence：

```text
surfel visible but outside object-owned regionlet
same-frame cannot-link
two slots repeatedly claim same surfel
object contains multiple incompatible motion modes
object support crosses strong depth boundary
```

Ownership score：

$$
E(i,k)=
\psi_{obs}(i,k)
+\psi_{track}(i,k)
+\psi_{app}(i,k)
-\psi_{neg}(i,k)
-\psi_{conflict}(i,k)
$$

第一版用 fixed weights，不训练。

### 7.5 实验设置

比较：

```text
S0: B1 single-mask ownership baseline
S1: regionlet birth without D4RT
S2: regionlet birth + D4RT surfel posterior
S3: regionlet birth + shuffled D4RT surfel posterior
S4: regionlet birth + no-track source-frame-only posterior
S5: regionlet birth + random same-count surfels
```

### 7.6 必须记录的指标

```text
AP / AP50 / AP25 under all eval policies
D4RT contribution delta over no-track
D4RT contribution delta over shuffle
surfel assignment entropy
surfel positive evidence count
surfel negative evidence count
visible-outside-object rate
same-frame cannot-link violations
slot conflict rate
object purity proxy
object fragmentation proxy
```

### 7.7 判断标准

D4RT ownership 成立需要：

```text
S2 own AP/AP50/AP25 > S3 shuffle by at least +0.05 / +0.08 / +0.08
S2 on S0/S1 > S1 no-D4RT by at least +0.02 AP
same-frame cannot-link violation lower than B1/O1
Stream3D-on-S2-support gap smaller than B1 gap
```

如果 S2 和 S3 接近：

```text
D4RT signal 没有被真正利用。
Codex 必须检查 surfel-to-regionlet measurement 是否错误，或者当前 signal 仍只是 area/mask-count proxy。
```

如果 S2 own 好但 S0/S1 仍崩：

```text
D4RT ownership 只做了 clean subset selection。
下一步必须增加 verified object recall，不要继续优化 own AP。
```

---

## 8. Phase 4：Core / Fringe / Reject support，替代 full-mask backprojection

### 8.1 目标

B1 的 full-mask backprojection 是捷径。v10 必须把 support 分为 core、fringe、reject：

```text
core: 高置信属于 object 的 regionlet/surfel support
fringe: 可能属于 object，但需要 ownership 支持的边界/扩展区域
reject: mask 内但没有 D4RT/geometry/appearance 支持，或与其他 object 冲突
```

### 8.2 假设

$H_F$：在 ownership 后进行 conservative fringe recovery，可以提升 S0/S1 recall，同时不显著牺牲 own AP50。

### 8.3 实验变体

```text
F0 core-only
F1 core + boundary-safe fringe
F2 core + surfel-near fringe
F3 core + depth/normal-consistent fringe
F4 core + ownership posterior fringe
F5 core + all fringe, negative control
```

### 8.4 必须记录的指标

```text
core points/object
fringe points/object
reject points/object
fringe accepted ratio
fringe rejected ratio
fringe conflict rate
WTA removed assignment rate
own AP / AP50 / AP25
S0 AP / AP50 / AP25
S1 AP / AP50 / AP25
per-GT best IoU mean
GT IoU>=0.50 count
duplicate prediction count
```

### 8.5 判断标准

F-phase 成立需要：

```text
F4 on S0 AP > F0 on S0 by +0.03
F4 on S1 AP > F0 on S1 by +0.05
F4 own AP50 drop <= 0.05 relative to F0
WTA removed assignment rate < 0.35
fringe conflict rate < 0.20
```

如果 F5 提升最多：

```text
说明 ownership model 太保守，但不能直接用 full fringe 作为方法。
Codex 需要分析 F5 中 true-positive fringe 与 false-positive fringe 的无 GT proxy 差异。
```

如果 F0 最好：

```text
说明 fringe recovery 当前只引入噪声。
下一轮先改 regionlet birth / ownership，不继续扩 support。
```

---

## 9. Phase 5：Object partition，不再使用普通 connected components

### 9.1 目标

当前 connected component 的问题是 weak positive edge 会传递，把一次 mask 错误扩散成整个 component 错误。v10 需要把 object formation 改为 constrained partition / set packing。

### 9.2 假设

$H_P$：object formation 是 explanation problem，不是 connectivity problem。使用 constrained partition 能减少 over-merge 和 duplicate predictions。

### 9.3 方法

候选 object slots 来自 regionlet/surfel ownership。然后求解一组 object hypotheses：

$$
\max_{\mathcal{O}} \sum_k Q(O_k) - \lambda_1 \sum_{i<j} overlap(O_i,O_j) - \lambda_2 \sum_k complexity(O_k)
$$

约束：

```text
same-frame strong cannot-link 不能合并
一个 surfel/regionlet 只能被一个 object core 拥有
fringe 可以 pending，但不能同时成为多个 object core
object 必须有最小 temporal/regionlet support
object 内部不能有明显多峰 motion/depth cluster
```

第一版可以使用 greedy set packing：

```text
sort hypotheses by quality
accept if it explains new regionlets/surfels
reject if strong conflict with accepted object
allow split_candidate if object is multi-modal
```

### 9.4 指标

```text
accepted object count
rejected object count
pending object count
split_candidate count
same-frame cannot-link violation count
duplicate predictions per GT
per-GT best IoU
AP / AP50 / AP25 all eval policies
object quality score vs actual best IoU diagnostic correlation
```

### 9.5 判断标准

Partition 成立需要：

```text
duplicate predictions per matched GT 下降至少 25%
AP/AP50 同时提升，不只是 AP25
S0 AP 超过 O38 c055
S1 AP 超过 v6 compact gap baseline 的一半
```

如果 object count 大幅下降但 AP 不升：

```text
发生 under-recall。
降低 acceptance threshold，允许 pending fringe，但不要重新变成 full-mask union。
```

如果 duplicate 下降但 AP50 下降：

```text
竞争过度删除真阳性。
改用 soft score penalty 而不是 hard reject。
```

---

## 10. Phase 6：ScanNet tune30 和 final

只有 Phase 2-5 在 probe5 达到 gate 后，才允许进入 tune30。tune30 只允许调：

```text
regionlet min/max area
depth discontinuity threshold
surfel seed density
ownership score weights
fringe threshold
partition conflict threshold
```

禁止调：

```text
直接根据 final AP 调任何阈值
直接对 GT IoU 选候选
在 final split 中反复试 config
```

### 10.1 tune30 必须输出

```text
own-support table
cross-support table
Stream3D-on-method-support table
method-on-Stream3D-support table
per-scene failure ranking
```

### 10.2 final 判断标准

最终 claim 的最低标准：

```text
paper-style own-support:
  method AP >= Stream3D AP + 0.5 point
  method AP50 >= Stream3D AP50 + 1.0 point

same-support:
  Stream3D-on-method-support - method-own AP <= 5 points
  method-on-S0 AP not near zero
  method-on-S1 AP improves over B1/O1 by clear margin

coverage:
  pre_points % cannot be <5% unless explicitly reported as high-precision partial reconstruction
  GT crop/full must be reported
```

如果 own-support 好但 cross-support 失败：

```text
只能写 high-precision observed subset，不允许 claim robust ScanNet win。
```

---

## 11. Phase 7：Dynamic Replica / Replica-Dynamic 只在静态 gate 后推进

当前 Dynamic Replica 本地检查曾出现 usable_scene_count=0，无法报告 official tracking。v10 仍需保留动态计划，但不能在静态 object partition 未过关时把动态作为逃避 ScanNet 的借口。

### 11.1 数据 gate

Codex 先运行：

```text
tools/check_dynamic_replica_env.py
```

必须确认：

```text
images exists
depths exists or evaluation supports image/trajectory-only
camera fields complete
instance/object IDs exist
semantic labels exist, if semantic AP required
trajectory GT exists, if APD3D/AJ/OA required
```

如果没有 instance/object ID：

```text
不能报告 IDF1、MOTA、official instance tracking、4D IoU。
只能报告 qualitative 或 diagnostic consistency。
```

### 11.2 动态实验只验证 D4RT-native 优势

如果数据可用，比较：

```text
D0 framewise 2D mask baseline
D1 Stream3D-style static overlap, if depth available
D2 v10 regionlet/surfel ownership memory
D3 D4RT-only track propagation diagnostic
```

指标：

```text
IDF1
IDSW
fragmentation
reactivation after occlusion
4D IoU
trajectory EPE
track purity
object lifetime accuracy
semantic consistency over time
```

判断标准：

```text
v10 在 IDF1 / IDSW / reactivation 上显著优于 static-overlap baseline。
如果只在 static ScanNet 有局部 AP，动态没有 tracking 优势，则论文主线不成立。
```

---

## 12. 必须生成的可视化总表

每轮实验至少保存以下图：

```text
1. own/cross support AP matrix heatmap
2. support ratio and GT crop/full bar chart
3. method vs Stream3D same-support delta chart
4. regionlet split overlay on RGB
5. D4RT surfel ownership overlay
6. core/fringe/reject overlay
7. final prediction vs GT mesh, at least 20 examples
8. missed GT object panels
9. duplicate prediction panels
10. D4RT geometry Sim3 residual heatmap
11. Stream3D RGB-D geometry vs D4RT-aligned geometry projected mask comparison
12. per-object best IoU histogram
```

可视化命名：

```text
outputs/visualization/v10_<phase>/<scene>/<method>_<view>.png
```

每个 figure 必须有一份 JSON sidecar：

```text
scene
method
eval_policy
AP/AP50/AP25 scene-level if available
pre_points %
union %
GT crop/full
failure tags
```

---

## 13. 不再作为主线继续做的方向

以下方向已经有足够负证据，不再作为 v10 主线：

```text
adaptive top-k ratio sweep
mask_count / area / reliability / logarea score sweep
point NMS
point IoC merge
support completion by nearest object core
full-mask fringe + final WTA
zero-conflict exclusivity
scene object memory over already-wrong masks
more windows -> more proposals without object partition
D4RT raw geometry direct replacement with RGB-D meter-scale thresholds
```

如果 Codex 需要做这些，只能作为 ablation 或 negative control，不能作为主要推进方向。

---

## 14. v10 成功时可以写什么，失败时必须写什么

### 14.1 成功时

如果 regionlet/surfel ownership 在 own-support 和 cross-support 都明显改善，可以写：

```text
v10 shows that the key missing component is not post-hoc ranking but object formation. By replacing full-mask object birth with regionlet/surfel ownership inference, Stream4D reduces support shrinking artifacts and improves same-support object quality relative to prior Stream4D variants.
```

如果 D4RT-aligned geometry with Stream3D 接近 RGB-D Stream3D，可以写：

```text
D4RT metric geometry, after evaluation-only Sim3 alignment and density-aware thresholds, is sufficiently accurate to support Stream3D-style 3D mask projection on ScanNet probe/tune splits.
```

如果不接近，只能写：

```text
D4RT correspondence is useful for ownership evidence, but current D4RT metric geometry is not yet a reliable replacement for ScanNet RGB-D/pose in static 3D segmentation.
```

### 14.2 失败时

如果 regionlet/surfel ownership 仍失败，必须明确写：

```text
Current 2D mask measurements and D4RT surfel evidence are insufficient to infer stable object partitions under training-free constraints on ScanNet. The method should either introduce stronger 2D mask generation / denser measurements / stronger appearance priors, or reposition the work around dynamic correspondence rather than ScanNet segmentation.
```

如果 D4RT geometry aligned experiment失败，必须写：

```text
The performance drop is attributable to metric geometry / point-density / alignment errors rather than only Stream3D hyperparameter mismatch, after scale-aware threshold controls.
```

---

## 15. Codex 最小执行清单

Codex 按以下顺序并行推进：

```text
Lane 0: unified eval matrix
  完成 Phase 0，锁定差距表。

Lane 1: D4RT geometry aligned to GT -> Stream3D
  完成 Phase 1 G0-G5，给出 AP drop 和 geometry residual。

Lane 2: regionlet object birth
  完成 Phase 2 R0-R4，先 probe5。

Lane 3: surfel ownership posterior
  完成 Phase 3 S0-S5 controls，证明 D4RT 是否真有贡献。

Lane 4: core/fringe/reject + partition
  只在 Lane 2/3 有正信号后启动 Phase 4/5。
```

每条 lane 都要独立输出：

```text
run script
config json
method manifests
evaluation text files
audit summary
visualization
failure analysis
code review packet
```

如果 48 小时内 Lane 2/3 没有超过 O38 S0 或 B1/O1 S1 诊断：

```text
停止继续写后处理。
先做 failure autopsy：随机抽 20 个 missed GT 和 20 个 duplicate prediction，人工查看 regionlet/surfel/mask evidence 是否支持正确 object。
```

这一步是为了避免继续盲跑脚本。
