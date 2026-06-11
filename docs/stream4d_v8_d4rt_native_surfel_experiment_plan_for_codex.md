# Stream4D v8：D4RT-native Semantic 4D Surfel Field 实验计划书

面向 Codex 的执行计划。本文档的核心目的不是继续堆后处理工具，也不是继续做“Stream3D + D4RT”的工程拼接，而是重新验证和推进一个真正的新问题：**如何从 D4RT 的 feed-forward 4D correspondence 构建 training-free semantic 4D reconstruction and tracking**。

本文档公式只使用 `$...$` 或 `$$...$$`，适合 Typora。本文所有实验都必须遵守：方法内部不读取 GT instance label；GT 只允许进入显式标记为 `diagnostic_only=true`、`uses_gt=true` 的诊断脚本；所有 reportable method result 必须有 manifest、命令日志、config、metric integrity 输出和可复现实验包。

---

## 0. 当前结论：为什么必须换算法思路

当前 v3 到 v7 的实验已经足够说明：继续在已有 prediction 上做 top-k、NMS、WTA、score sweep、point merge、mask fusion、track bucket suppression，没有希望解决核心问题。v7 的 same-support gap matrix 给出了最清楚的数字：Stream3D 在 Stream4D 32f support 上达到 `0.3992 / 0.5972 / 0.7425`，而当前最强 Stream4D-like dense/object baseline 只有 `0.2848 / 0.5040 / 0.6719`。差距约为 `AP -0.1144`、`AP50 -0.0932`、`AP25 -0.0706`。carrier-tracklet 分支更差，C6 strict-track WTA 只有 `0.0266 / 0.1118 / 0.4656`，说明当前 carrier component 方案不是接近成功，而是表示本身不对。

这个结果说明三件事。

第一，Stream4D 当前不是缺某个后处理，而是没有可靠解决从 noisy 2D mask observations 和 D4RT carriers 到 object-level instances 的离散化问题。大量实验已经证明：稀疏 high-precision support 可以在 own recompute 下拿高分，但一旦评估 support 固定或扩大，false negative 爆炸；增加 support 又会引入大量 noisy object、over-merge、重复实例和边界污染。

第二，当前 carrier branch 的基本假设错了。过去的做法近似假设“落在同一 2D mask 里的 carriers 属于同一个 object”，再用 connected component 得到 object。这在真实 VFM mask 上不成立。2D mask 是 noisy observation，不是 object identity。一个欠分割 mask 可以把桌子和椅子连成 clique；一个过分割 mask 又会把同一物体拆碎。connected component 会让一次错误正边通过传递闭包扩散成整块 component 错误。

第三，当前实现没有真正使用 D4RT 的核心能力。D4RT 的关键不是“从 mask 中采几个 sparse carrier”，而是统一 query interface：

$$
q=(u,v,t_{src},t_{tgt},t_{cam}) \rightarrow X \in \mathbb{R}^3
$$

这个接口允许对任意 source pixel 查询其在任意 target time、任意 reference camera 下的 3D position。D4RT 论文还提出了 dense all-pixel tracking algorithm，用 occupancy grid 避免朴素 $O(T^2HW)$ 查询，从而得到 dense dynamic correspondence。也就是说，D4RT 的正确使用方式应该是构建 dense 或 semi-dense material surfel field，而不是把它降级成 sparse mask support selector。

因此 v8 的目标是从根上换表示：

```text
过去错误路线：
2D mask -> sparse carriers -> carrier/mask co-membership graph -> component -> full-mask fringe -> WTA -> 3D mask export

v8 目标路线：
video -> D4RT dense/semi-dense surfel tracks -> mask/feature observations as measurements -> semantic likelihood over surfels -> object field partition -> queryable semantic 4D object field
```

Stream3D 从现在开始只作为 baseline、协议参照和失败对比，不作为方法组件。D4RT 也不能只被当作辅助点采样器；D4RT 应该是 4D material field 的几何/时间骨架。

---

## 1. 当前研究进展对我们的启发

### 1.1 D4RT 的启发：从 frame-level decoding 转向 on-demand 4D material queries

D4RT 的核心贡献是将 depth、camera、point cloud、point tracking 和 dynamic correspondence 统一到同一个 query interface。它不是要求每帧 dense decode，也不是为每个任务做不同 decoder，而是把 global scene representation 固定后，对任意 query 独立预测 3D point position。

这对我们最重要的启发是：semantic 4D reconstruction 不应该从“frame mask merging”开始，而应该从“material surfel field”开始。每个 surfel 本身跨时间存在，object tracking 是 surfel partition 的时间可见结果，而不是后验地把每帧 mask 接起来。

D4RT 的 dense all-pixel tracking 也给出一个明确方向：当前 sparse carrier 方案过早牺牲了 D4RT 的能力。我们必须先验证 semi-dense 或 adaptive dense surfel field 是否能在 ScanNet / Replica-Dynamic 中成立，再谈 semantic object inference。

### 1.2 Stream3D 的启发：它强在 ScanNet 的 dense RGB-D static prior，但这不是我们的路线

Stream3D 的形式化任务假设输入包括 RGB 图像、由 depth 和 pose 生成的 reconstructed point cloud，以及投影到点云上的 3D masks。它的 set-cover 和 manifold refining 都围绕静态 dense point cloud 设计。在 ScanNet 这种静态 RGB-D benchmark 上，这些假设很强：同一个物体在 3D 中连续，不同物体在 3D 中相对分离，多视角 mask 可以通过 3D overlap 合并。

我们不应该复制这个归纳偏置。Stream3D 是要被超过和解释的 baseline，不是 v8 的 backbone。当前失败恰好说明：如果只用 sparse carriers 去挑战 Stream3D 的 dense point-cloud pipeline，会天然吃亏。v8 要做的是另一个问题：D4RT-native semantic 4D surfel field。ScanNet 用来检验静态退化情况是否合理，动态场景才是 4D tracking 优势的主战场。

### 1.3 4D semantic field 研究的启发：语义必须变成 field，不是最后贴标签

2025–2026 的相关方向已经在往 4D language/semantic field 走。4D LangSplat 强调 dynamic 4D language field 和 time-sensitive open-vocabulary queries；Feature4X 将 2D foundation model 能力蒸馏进 4D Gaussian feature field；4DLangVGGT 进一步把 4D geometry 和 language grounding 推向 feed-forward framework。这些工作说明一个趋势：语义不再只是 instance segmentation 后的 class label，而是时空场上的可查询属性。

但这些工作也给了我们差异化空间：很多方法依赖 per-scene optimization、Gaussian feature distillation 或训练新的语言对齐模块。v8 应该坚持 training-free / frozen-model：不训练新的 3D/4D semantic model，不使用 3D/4D semantic labels，不做 per-scene optimization-heavy fitting；用 D4RT dense surfel field 和 frozen 2D/VLM observations 做 object field inference。

---

## 2. v8 总目标和必须解决的问题

v8 的整体目标是建立一个最小但真实的 D4RT-native semantic 4D reconstruction and tracking 闭环。它不以“立刻 full ScanNet 超过 Stream3D”为第一目标，而以验证下面四个科学假设为核心。

### 2.1 总目标

给定一段视频 $V$ 和冻结的 2D segmentation / feature observations，构建一个 visible-surface semantic 4D object field：

$$
\mathcal{S} = \{O_k\}_{k=1}^{K}
$$

每个 object 是一组 D4RT surfels 的 partition：

$$
O_k = \{s_i \mid y_i=k\}
$$

每个 surfel 是跨时间可查询的 material-like unit：

$$
s_i = \{X_i(t), \pi_i(t), v_i(t), c_i(t), f_i(t), z_i(t)\}_{t=1}^{T}
$$

其中：

```text
X_i(t): D4RT 预测的 3D position
pi_i(t): surfel 在 frame t 的 2D projection
v_i(t): visibility
c_i(t): confidence
f_i(t): appearance / feature observation
z_i(t): mask / semantic observation
```

2D masks 不再直接变成 object。它们只作为 measurement：

$$
m_{t,r} \Rightarrow p(z_{i,t}=r \mid y_i)
$$

object inference 的目标不是 connected component，而是解释观测：

$$
E(Y)=\sum_i \psi_i(y_i)+\sum_{i,j}\psi_{ij}(y_i,y_j)+\sum_k \Omega(O_k)
$$

这里 $Y=\{y_i\}$ 是 surfel assignment，$\psi_i$ 是 unary observation likelihood，$\psi_{ij}$ 是 pairwise consistency/cannot-link，$\Omega$ 是 object-level validity。

### 2.2 v8 必须解决的四类问题

**问题 A：D4RT dense/semi-dense surfel field 是否成立。** 当前 D4RT geometry sparse export 的 Sim3 residual 很大，G1/G3 segmentation AP 失败。v8 不能继续在 sparse carrier 上做对象推断，必须先验证 D4RT 原生 dense/semi-dense query 是否能构建足够可靠的 surfel field。

**问题 B：2D masks 是否能作为 surfel-level semantic observations。** 过去把 mask co-membership 当强正边是错的。v8 要先诊断：同一个真实 object 内的 surfels 是否有更一致的 mask/feature observations，不同 objects 是否有可分性。GT 只能用于 diagnostic，不参与方法。

**问题 C：object inference 不能再用普通 connected component。** weak positive 不能传递式合并；strong negative 必须优先。v8 要比较 signed correlation clustering、energy-based partition、core/fringe/reject ownership，而不是继续 graph connected component。

**问题 D：动态场景必须被纳入主实验。** 没有 Replica-Dynamic / Dynamic Replica，工作就只是在静态 ScanNet 上和 Stream3D 硬拼，无法体现 D4RT 的动态 correspondence 优势。v8 必须把动态数据补齐作为 P0 blocker，而不是继续延期。

---

## 3. 实验执行总览：并行推进，而不是串行耗时

为加快实验，v8 分成五条并行 lane。Codex 不要等某条 lane 完全结束再启动下一条；只要依赖数据存在，就并行跑。

```text
Lane 0: 审计与提交规范。确保每轮代码可审核、指标无 GT 泄漏。
Lane 1: D4RT dense/semi-dense surfel field sanity。先验证几何/轨迹场是否成立。
Lane 2: mask-as-measurement separability。验证 2D observations 是否能支持 object partition。
Lane 3: surfel object inference。实现并比较新的 object field partition 方法。
Lane 4: ScanNet gap / Dynamic Replica tracking。分别做静态退化和动态主任务验证。
```

任何 full ScanNet final 之前必须满足：

```text
1. Lane 0 metric integrity pass。
2. Lane 1 surfel field sanity 至少达到最低 gate，或者明确标记为 geometry blocker。
3. Lane 3 在 probe5 上超过当前 v6/v7 best baseline 的固定 support 指标。
4. 所有方法结果 manifest uses_gt=false。
```

---

## 4. Lane 0：代码审计、指标安全和提交规范

### 4.1 目标

过去几轮审计包经常缺完整依赖、缺 prediction/TMP、缺运行环境，导致只能做静态审计，不能完全复跑。v8 开始，每轮 Codex 必须提交可审核代码包。否则本轮结果不得进入方法表。

### 4.2 必须提交的 code audit packet

每轮提交一个 zip，命名：

```text
stream4d_v8_code_audit_packet_<YYYYMMDD_HHMMSS>.zip
```

zip 内必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/*.py
Stream3D/tests/*.py
Stream3D/docs/*.md
Stream3D/splits/*.txt
Stream3D/configs/*.json
Stream3D/outputs/audit/*.json
Stream3D/outputs/audit/*.md
Stream3D/logs/*.log
Stream3D/scripts/reproduce_v8_<phase>.sh
Stream3D/environment_v8.yml 或 pip_freeze_v8.txt
```

如果某个实验要审核 AP，必须附带 probe5 的最小可复现数据：

```text
Stream3D/data/prediction/<config>_class_agnostic/*.npz, only probe5 scenes
Stream3D/data/TMP/<config>/*_pre_points.npy, only probe5 scenes
Stream3D/data/evaluation/scannet/<config>_class_agnostic.txt
config_manifest.json
```

如果文件太大，可以只提交 5-scene probe 包，但必须提供完整工程路径和复现命令。不能只提交 summary markdown。

### 4.3 指标安全检查

每个 reportable method config 必须通过：

```text
python -m tools.scan_reportable_configs --configs <configs>
python -m tools.verify_stream4d_metric_integrity --configs <configs>
python -m unittest tests.test_stream4d_protocol_fixes
python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
```

必须记录：

```text
evaluator_ap_core_equal_by_hash
num_configs_missing_manifest
num_uses_gt_and_method_result
num_oracle_configs
num_diagnostic_only_configs
object_dict_pred_alignment_mean_iou
object_dict_pred_alignment_min_iou
pre_points_policy
support_policy
uses_gt
is_method_result
is_diagnostic_only
```

### 4.4 判断标准

Lane 0 通过条件：

```text
1. py_compile pass。
2. unit tests pass。
3. reportable configs 中 uses_gt=false。
4. oracle diagnostic output 必须 uses_gt=true 且 diagnostic_only=true。
5. evaluator 默认拒绝 uses_gt=true 的普通评估。
6. object_dict 与 pred columns 对齐，mean IoU=1.0 或明确 cannot_verify_alignment。
```

不满足时 Codex 的先行修复方向：

```text
如果 import 失败，优先补齐缺失模块或提供 environment 文件。
如果 metric integrity 失败，先修 manifest / evaluator guard，不跑新 AP。
如果 oracle output 可被普通 evaluator 读取，立即修 evaluator guard。
如果 object_dict/pred 不对齐，停止 rescore 和 score sweep，先加 alignment map。
```

---

## 5. Lane 1：D4RT dense/semi-dense surfel field sanity

### 5.1 目标

验证 D4RT 是否能在 ScanNet 和 Dynamic Replica 上生成足够可靠的 dense/semi-dense material surfel field。如果这一层不成立，后续 semantic object inference 都会建立在错误几何上。

### 5.2 核心假设

$H_{G1}$：当前失败主要来自 sparse carrier adapter，而不是 D4RT 4D correspondence 本身。使用 D4RT 原生 dense/all-pixel tracking 或 adaptive dense query 后，visible surface coverage、track cycle consistency 和 Sim3 residual 会显著优于过去 sparse carrier export。

$H_{G2}$：D4RT 在 ScanNet 上不一定能替代 RGB-D metric geometry，但它应该能提供足够稳定的 material correspondence，用于 semantic object field partition。也就是说，metric residual 可以不是完美，但 track consistency 和 surfel observation consistency必须可用。

### 5.3 实验设计

在 probe5 scenes 先做三种 surfel field：

```text
G0_sparse_existing: 当前 sparse mask carrier cache，作为失败基线。
G1_grid_dense: 每帧固定 grid 或 mask-aware grid 的 semi-dense D4RT query。
G2_occupancy_dense: 仿 D4RT Algorithm 1，用 occupancy grid 避免重复 tracks，生成 adaptive dense tracks。
```

G1 先实现，成本低。采样策略：

```text
image grid 16x16 / 32x32 / 64x64
mask-aware oversampling: 每个 2D mask 内至少 N 个 grid/sampled surfels
boundary-aware sampling: mask boundary 附近加密，但只用于观测，不作为强正边
```

G2 后实现，如果 G1 有正信号。G2 需要维护 occupancy grid $G \in \{0,1\}^{T\times H\times W}$，每条 visible track 标记经过的 spatio-temporal pixels，未访问区域继续发起 query。

### 5.4 必须记录的几何/轨迹指标

每个 scene、每个 window、每种 sampling density 记录：

```text
num_source_queries
num_surfel_tracks
num_valid_tracks
num_visible_observations
surfel_coverage_2d_per_frame
surfel_coverage_3d_after_export
uv_in01_rate_mean / median / p10
visibility_mean / median / p10
confidence_mean / median / p10
track_length_visible_mean
track_length_visible_p10
duplicate_track_rate
hole_ratio_2d
```

用 ScanNet RGB-D 只做 evaluation diagnostic，记录：

```text
Sim3 anchor count
Sim3 scale
Sim3 residual median / p90 / p95
Sim3 inlier ratio at 0.05m / 0.10m / 0.20m
D4RT depth-vs-ScanNet depth MAE for tsrc=ttgt=tcam
D4RT point-vs-RGBD point error after Sim3
```

记录 cycle/self consistency：

```text
self_uv_error_mean / p90
cycle_uv_error_mean / p90
cycle_3d_error_mean / p90
forward_backward_visibility_consistency
```

### 5.5 成立标准

G1/G2 不要求立刻超过 Stream3D AP。它们先判断 D4RT surfel field 是否值得继续。

最低 gate：

```text
1. uv_in01_rate_mean >= 0.70，或比当前 sparse carrier均值提升至少 +0.15。
2. track_length_visible_mean >= 6 frames on 32f windows，或动态数据中达到可解释轨迹长度。
3. self_uv_error_p90 <= 8 px at resized model resolution，或明确单位换算后合理。
4. cycle_uv_error_p90 <= 12 px。
5. semi-dense surfel exported coverage 至少达到当前 sparse union 的 3x。
6. Sim3 residual median mean 明显低于当前 0.680m；若仍大于 0.30m，ScanNet mesh AP 只能作为 diagnostic，不可作为 D4RT geometry claim。
```

强 gate：

```text
1. Sim3 residual median <= 0.15m 或 <= 5% scene diagonal。
2. 2D coverage per frame >= 50% of visible object-mask pixels。
3. dense/semi-dense surfel field 不再只有 5%-7% support。
```

### 5.6 不满足条件时 Codex 先尝试什么

如果 Sim3 residual 很高但 self/cycle UV 正常：

```text
优先检查坐标系和 reference camera：
1. tsrc=ttgt=tcam depth query 是否和 ScanNet depth 一致。
2. normalized uv 是否和 D4RT 输入 resize/crop 后坐标一致。
3. tcam reference 是否固定到同一个 frame。
4. 是否把 local clip coordinate 错当 world coordinate。
5. 是否需要按 D4RT 官方 intrinsics/extrinsics query 先估 camera，再对齐。
```

如果 uv_in01_rate 低：

```text
检查 video preprocessing：resize、padding、aspect ratio token、normalized coordinate。
用官方 infer_track_3d.py 对同一 clip 生成 reference 输出，对比 adapter 输出。
```

如果 dense query 太慢：

```text
先做 G1 32x32 grid，不做 all-pixel。
启用 query batching 和 cache。
只跑 probe5 的 16f/32f windows。
不要因为速度问题退回 sparse mask-only carrier。
```

如果 dense surfel field 几何仍失败：

```text
停止 ScanNet mesh AP 主线，把 ScanNet 仅作为 diagnostic。
转到 Dynamic Replica 的 image-space / trajectory-space object tracking 验证，避免错误地用不可靠 mesh export 否定 D4RT correspondence。
```

### 5.7 可视化要求

每个 probe scene 必须保存：

```text
frame overlay: sampled surfels colored by visibility/confidence
frame overlay: tracks colored by object/motion direction
2D heatmap: surfel coverage per frame
3D point cloud: D4RT surfels after Sim3, colored by residual
track cycle error histogram
Sim3 residual histogram
failed tracks montage: high confidence but high residual
```

---

## 6. Lane 2：mask-as-measurement separability 诊断

### 6.1 目标

验证 2D mask / feature observations 是否能在 D4RT surfel field 上提供 object partition 信号。过去的方法把 co-mask membership 当强正边，这是错误假设。v8 必须先把它降级为 noisy likelihood，并测量它到底有没有区分力。

### 6.2 核心假设

$H_{O1}$：同一真实 object 的 surfels 在多帧中应有更相似的 mask observation history、appearance feature 和 motion/geometry consistency；不同 objects 即使偶尔落入同一 2D mask，也会被 negative evidence 或 feature/geometry/motion 分开。

$H_{O2}$：如果只用 mask co-membership，pairwise object classification AUC 会低且 over-merge；加入 boundary safety、DINO/CLIP feature、trajectory consistency、same-frame cannot-link 后，AUC 和 clusterability 会显著提升。

### 6.3 实验设计

这条 lane 分两类诊断。

第一类是 method-free diagnostic。用 GT instance 只评估 observation separability，不参与任何方法输出：

```text
输入：D4RT surfels + 2D masks + ScanNet GT instance labels mapped to mesh/surfel for diagnostic。
输出：observation separability metrics。
manifest: uses_gt=true, diagnostic_only=true。
```

第二类是 method signal construction，不用 GT：

```text
输入：D4RT surfels + 2D masks + RGB/DINO/CLIP features。
输出：surfel unary likelihoods and pairwise signed evidence。
manifest: uses_gt=false, is_method_result=false until partition export。
```

### 6.4 要构造的 observation features

对每个 surfel $s_i$，记录：

```text
mask_id_history: 每个可见帧落入的 mask id 或 background/unknown
mask_boundary_distance_history
mask_area_history
mask_confidence_history if available
DINO feature pooled at projected pixel or crop region
CLIP crop feature for mask/object observation
RGB local patch histogram, only as low-level fallback
D4RT motion vector history
3D local neighbor consistency
```

对 surfel pair $(i,j)$，构造 signed evidence：

```text
positive_mask_covis: 多帧落入同一 mask，但只能作为 weak positive
positive_feature: DINO/appearance similarity
positive_motion: trajectory/motion consistency
positive_geometry: canonical adjacency / local 3D proximity
negative_same_frame_exclusive: 同一帧落入不同 non-overlapping masks
negative_boundary: 二者跨越稳定 mask/depth/appearance boundary
negative_motion: trajectory mode incompatible
negative_geometry: local adjacency断裂或3D距离长期不合理
negative_ownership: 同一 surfel 被多个 object hypotheses 强占用
```

### 6.5 必须记录的指标

GT diagnostic 指标：

```text
pairwise_same_object_AUC for each signal and combined signal
positive_edge_precision@topK
negative_edge_precision@topK
within-object observation entropy
between-object observation distance
mask co-membership false-positive rate
mask co-membership false-negative rate
same-frame cannot-link recall
GT object surfel coverage ratio
GT object dominant-mask purity
```

非 GT method signal 指标：

```text
num_surfels_with_observation
num_surfels_without_observation
mean observations per surfel
positive_edge_count by type
negative_edge_count by type
positive/negative edge ratio
edge degree distribution
large connected component size before negative constraints
```

### 6.6 成立标准

Lane 2 成立条件：

```text
1. mask co-membership alone 的 pairwise AUC 不能作为唯一信号；如果它 <0.65，必须禁止强正边传递。
2. combined non-GT signal 的 diagnostic AUC >=0.75，或相比 mask-only 提升 >=0.10。
3. same-frame cannot-link precision >=0.90。
4. boundary / feature / motion 至少有一个信号能显著降低 cross-object false positive。
```

如果不满足：

```text
若 mask observation 不可分，优先换 2D masks：Cropformer vs SAM2 vs SAM2.1/Mask2Former，先在 probe5 比较。
若 feature 不可分，改用 DINOv2 patch feature，不要继续 RGB histogram。
若 motion 不可分，说明静态 ScanNet 上 motion signal 弱，动态场景再验证；ScanNet 静态用 geometry/feature/boundary。
若所有信号都不可分，暂停 object partition，回到 surfel projection/coordinate/mask alignment 检查。
```

### 6.7 可视化要求

每个 probe scene 输出：

```text
surfel observation history strip: 每条 track 在时间上的 mask id
pairwise evidence heatmap sampled from top GT objects, diagnostic only
mask co-membership false-positive examples
same-frame cannot-link examples
feature embedding t-SNE/UMAP colored by GT diagnostic and predicted clusters
boundary failure montage: one 2D mask spans multiple objects
```

---

## 7. Lane 3：D4RT-native surfel object inference

### 7.1 目标

实现真正的新算法原型：object 不再由 mask connected component 或 carrier co-membership component 得到，而是由 surfel field partition 得到。2D masks 是 observation，D4RT surfels 是 field primitive，object 是 partition。

### 7.2 三个候选方法方向，并行验证

Codex 需要并行实现三个轻量原型，不要一次只赌一个。

#### 方向 A：Signed correlation clustering over surfels / surfacelets

把 surfel 或 surfacelet 作为节点，边为 signed evidence。目标是让正边尽量在同一 cluster，负边尽量跨 cluster：

$$
E(Y)=\sum_{(i,j)\in E^+}w^+_{ij}\mathbf{1}[y_i\ne y_j]+\sum_{(i,j)\in E^-}w^-_{ij}\mathbf{1}[y_i=y_j]
$$

这不是普通 connected component。weak positive 不允许无限传递；strong negative 可以阻止或切断 cluster。

第一版优化器可以 training-free：greedy agglomerative merge with negative-edge veto、signed modularity、local search split-merge。关键不是优化器花哨，而是必须能处理 negative constraints。

#### 方向 B：Surfacelet region proposal + ownership inference

先把 dense surfels 按空间/时间/feature 局部一致性聚成 surfacelets，再对 surfacelets 做 object ownership。surfacelet 比单个 surfel稳定，也比整张 2D mask小。

surfacelet 生成规则：

```text
同一 source/target 局部邻域
mask boundary 不跨越稳定边界
D4RT track motion 差异小
DINO/RGB feature相似
```

object ownership 先于 support export：一个 2D mask 内可以有多个 surfacelets 属于不同 objects，不能整张 mask被一个 component 拥有。

#### 方向 C：Energy-based core/fringe/reject field

每个 object 不是一个硬 mask，而是三层 support：

```text
core: 高置信 surfels，参与 object identity。
fringe: 中置信 surfels，只能被 core 解释后加入，不触发 merge。
reject: noisy observations，记录但不加入 object。
```

核心约束：fringe 不允许把两个 object 连接起来；reject 不参与 object score；只有 core 可以建立强正证据。

### 7.3 共同输入和输出

输入：

```text
D4RT surfel tracks from Lane 1
mask/feature observations from Lane 2
optional RGB-D for evaluation bridge only, not for object identity if testing pure D4RT-native variant
```

输出：

```text
object_surfel_partition.json
object_field.npz
object_tracks.json
prediction export for ScanNet diagnostic
visualization package
```

object field 必须支持 query：

```text
query object k at time t -> visible surfels, 2D support, 3D support
query surfel i -> object id over time
query text label -> ranked objects, if semantic layer enabled
```

### 7.4 实验设置

先跑 probe5：

```text
scene0011_00
scene0030_00
scene0050_00
scene0081_01
scene0591_00
```

每个方向至少跑：

```text
16f quick debug
32f comparable with v6/v7
96f multi-window if cache exists
```

对比对象：

```text
P0 Stream3D on Stream4D 32f support diagnostic: 0.3992 / 0.5972 / 0.7425
P3 v6 compact: 0.2848 / 0.5040 / 0.6719
P4 v6 scoreunique: 0.2820 / 0.4982 / 0.6913
C6 strict-track WTA: 0.0266 / 0.1118 / 0.4656
new v8-A/B/C
```

### 7.5 必须记录的指标

ScanNet class-agnostic metrics：

```text
AP / AP50 / AP25 under own recompute
AP / AP50 / AP25 under fixed 32f support
AP / AP50 / AP25 under Stream3D support diagnostic
num_pred_instances
prediction_union_ratio
union_in_target_ratio
```

object quality diagnostics：

```text
per-GT best IoU mean
GT count with best IoU >= 0.25
GT count with best IoU >= 0.50
missed GT count with best IoU < 0.25
duplicate predictions per GT
support conflict rate
WTA removed assignment rate, if WTA used
component/object size distribution
tiny object ratio
large object overmerge ratio
```

surfel partition diagnostics：

```text
num_surfels
num_surfacelets
num_objects
mean surfels per object
core/fringe/reject ratio
positive edge violation count
negative edge violation count
same-frame cannot-link violation count
object temporal span
object visible frames count
object reactivation count
```

GT diagnostic only：

```text
cluster purity
adjusted Rand index
overmerge count
undermarge count
pairwise precision/recall of same-object relation
```

注意：GT diagnostic 指标不能进入 method table，只用于解释失败。

### 7.6 成立标准

v8 probe5 最低 method gate：

```text
AP >= 0.32
AP50 >= 0.54
AP25 >= 0.70
#pred between 80 and 300, unless有明确 object count 解释
support conflict rate < 0.20
WTA removed assignment rate < 0.35 if WTA exists
same-frame cannot-link violation near 0
```

v8 strong gate：

```text
same fixed 32f support 下接近或超过 Stream3D diagnostic：
AP gap <= 0.05
AP50 gap <= 0.04
AP25 gap <= 0.03
```

如果 own recompute 高但 fixed-support 低：

```text
说明仍是 sparse clean core，不是 reconstruction。回到 Lane 1 coverage 或 Lane 3 fringe ownership。
```

如果 AP25 高但 AP/AP50 低：

```text
说明 coarse object 对，但边界/one-to-one assignment 不行。优先做 surfacelet split、boundary-aware ownership，不做 top-k。
```

如果 #pred 很少且 AP低：

```text
over-merge。检查 negative edge violation、large component、多峰 geometry。
```

如果 #pred 很多且 AP低：

```text
fragmentation。检查 core threshold 是否太严格、positive edge是否太弱、同一 object surfacelets 是否未合并。
```

如果 GT oracle candidate pool 高而 method低：

```text
selection/scoring 问题；建立 object validity score，不扩候选。
```

如果 GT oracle candidate pool也低：

```text
object generation问题；回到 surfel sampling、observation projection、surfacelet generation。
```

### 7.7 可视化要求

每个方向每个 scene 保存：

```text
3D object field colored by predicted object id
same scene colored by GT object id, diagnostic only
surfel graph sampled visualization: positive/negative edges
large component split visualization
core/fringe/reject overlay on frames
object tracks over time as colored surfel trajectories
failure montage: overmerge / undermerge / boundary leak / missing object
precision-recall curve by object score
```

---

## 8. Lane 4A：ScanNet 静态退化实验和与 Stream3D 差距确认

### 8.1 目标

ScanNet 不是 v8 的唯一目标，但必须用它回答：D4RT-native semantic 4D surfel field 在静态室内场景上是否至少合理，和 Stream3D 还差多少，差距来自 geometry、observation、object partition 还是 export adapter。

### 8.2 关键假设

$H_{S1}$：只要 D4RT-native surfel field 质量足够，v8 在 same support 下的 object quality 应该逐步缩小与 Stream3D 的 AP/AP50 gap。若 gap 长期存在，失败不在 evaluator，而在 object partition 或 surfel field。

$H_{S2}$：D4RT geometry 直接替代 Stream3D RGB-D geometry 在 ScanNet 上大概率下降；该实验作为 negative control，不作为方法主线。它用于证明我们不应该把 D4RT 当 metric geometry replacement，而应该把 D4RT 用作 4D surfel correspondence backbone。

### 8.3 实验矩阵

每个新方法输出都要进入 gap matrix：

```text
Prediction configs:
P0 Stream3D scannet baseline
P1 current Stream4D 32f self
P2 v6 compact
P3 best v8-A signed clustering
P4 best v8-B surfacelet ownership
P5 best v8-C core/fringe/reject

Support configs:
S0 own recompute
S1 Stream4D 32f fixed support
S2 v8 surfel support
S3 Stream3D baseline support diagnostic
S4 union support diagnostic
```

禁止只报 own recompute。每个 config 至少报：

```text
own recompute
same 32f fixed support
Stream3D support diagnostic
```

### 8.4 Stream3D 使用 D4RT geometry 的 negative control

这不是方法主线，但必须跑，用来回答审稿人可能问的问题：“为什么不直接把 D4RT 几何喂给 Stream3D？”

实验：

```text
G0: 原版 Stream3D RGB-D/pose geometry。
G1: Stream3D projection 使用 D4RT tsrc=ttgt=tcam point/depth，经 evaluation-only Sim3 导出。
G2: Stream3D projection 使用 D4RT shared-reference point map，经 Sim3 对齐。
G3: D4RT geometry + normalized manifold threshold。
G4: D4RT geometry + track-consistency manifold replacement。
```

方法内部如果使用 Sim3，必须标记为 diagnostic，不能算 D4RT-native method result。推荐先做 probe5，不直接 full ScanNet。

记录指标：

```text
AP / AP50 / AP25
num projected 3D masks
projection hit rate
D4RT-vs-RGBD point error
Sim3 residual median/p90
manifold refinement keep ratio
object count
prediction union ratio
```

判断标准：

```text
如果 G1/G2 相比 G0 AP drop > 3 或 AP50 drop > 5，明确结论为：D4RT geometry replacement is not the ScanNet path。
如果 G4 明显优于 G1/G2，说明 D4RT tracks 对 identity/refinement 有用，但仍不代表直接替代 RGB-D metric geometry。
```

### 8.5 成立标准

ScanNet v8 阶段目标分三档。

最低可推进：

```text
v8 在 probe5 same 32f support 上超过 v6 compact：
AP >= 0.32, AP50 >= 0.54, AP25 >= 0.70
```

可写成方法进展：

```text
v8 在 tune30 fixed support 上稳定超过 v6 compact，且不依赖 own support 缩小。
```

可考虑 full final：

```text
tune30 locked config 在 final split 上保持：
AP gap to Stream3D <= 5 points 或至少 AP50/AP25 接近；
同时 dynamic dataset 上有明显 tracking advantage。
```

如果 ScanNet 静态仍落后很多但动态明显强：

```text
论文定位应避免 claim ScanNet 静态 SOTA。ScanNet 作为静态退化和诊断；主贡献放在 dynamic semantic 4D tracking。
```

---

## 9. Lane 4B：Replica-Dynamic / Dynamic Replica 动态实验

### 9.1 目标

验证 D4RT-native semantic 4D field 的真正优势：动态 object identity、遮挡重现、运动一致性和 time-sensitive semantic query。没有动态实验，Stream4D 的核心 claim 不成立。

### 9.2 P0 数据环境检查

Codex 先运行：

```text
python -m tools.check_dynamic_replica_env --root <dynamic_replica_root>
```

必须输出：

```text
data_root_exists
split_dir_exists
annotation_exists
usable_scene_count
camera_fields_present
instance_annotation_present
can_report_official_instance_tracking
can_report_trajectory_metrics
```

如果数据缺失：

```text
不能报告 IDF1、MOTA、4D IoU、official instance tracking。
只能提交 blocker report，并向用户列出缺失目录/文件名。
不要用 pseudo label 伪造 official metrics。
```

### 9.3 动态实验任务定义

任务 1：dynamic object track consistency。

```text
输入视频，输出每个 object 的 time-indexed visible surfel support。
评估 object identity 是否随时间一致。
```

任务 2：occlusion reactivation。

```text
物体消失后重新出现，是否保持同一 object id。
```

任务 3：moving-object 4D support。

```text
同一个物体在不同空间位置仍保持同一 identity，不依赖 static 3D overlap。
```

任务 4：semantic query。

```text
文本 query: moved chair, person/object being moved, object before/after interaction。
先做 qualitative + retrieval@K，如果有 GT semantic label 再做 quantitative。
```

### 9.4 指标

若有官方 instance / track GT：

```text
IDF1
ID precision / ID recall
ID switches
fragmentation
MOTA / MOTP if applicable
track purity
4D IoU over time
mean trajectory error for matched surfels
occlusion reactivation accuracy
```

若只有 2D masks / pseudo masks：

```text
temporal mask consistency
track survival length
reappearance same-id rate under pseudo association
cycle consistency
qualitative only, no official claims
```

semantic metrics：

```text
text-object retrieval R@1/R@5 if labels/text available
time-sensitive query accuracy if annotations available
object caption consistency over time, diagnostic only if generated by VLM
```

### 9.5 对比方法

```text
D0: frame-wise 2D mask + no memory
D1: current sparse carrier component baseline
D2: v8 surfel field partition
D3: static-overlap baseline resembling Stream3D historical matching, if data supports RGB-D
D4: oracle trajectory diagnostic, uses_gt=true, diagnostic_only=true
```

### 9.6 成立标准

动态主线成立需要：

```text
1. v8 相比 static-overlap / sparse carrier baseline 明显降低 ID switches。
2. reactivation accuracy 提升。
3. moving objects 不因空间位置变化被拆成多个 IDs。
4. qualitative visualization 清楚展示 object 轨迹、遮挡、重现。
```

如果动态数据无法跑：

```text
本轮计划不得进入 paper-claim 阶段。
Codex 必须优先补数据环境或生成最小 synthetic dynamic test，而不是继续 ScanNet 后处理。
```

### 9.7 可视化

每个动态 scene 输出：

```text
video overlay: object id colors over time
3D/4D trajectory visualization: object centroid path + surfel tracks
occlusion timeline: visible/lost/reactivated states
ID switch montage
text query results montage, if semantic layer enabled
```

---

## 10. Semantic 4D field 层：只在几何和 object partition 通过后启动

### 10.1 启动条件

只有当 Lane 1 和 Lane 3 至少满足最低 gate 后，才启动 semantic layer。否则 semantic AP 低只会反映 instance mask 失败，不会提供新信息。

启动条件：

```text
probe5 AP50 >= 0.54 or dynamic IDF1 shows improvement
surfel field coverage pass
object partition conflict rate acceptable
```

### 10.2 目标

将 object field 变成 open-vocabulary semantic 4D field。语义不再是最后给 3D mask 贴 label，而是 object/surfel/time 上的属性：

$$
p(l \mid O_k,t), \quad p(q_{text} \mid O_k,t)
$$

### 10.3 实验

```text
Sema0: current CLIP representative crop baseline
Sema1: surfel feature aggregation with CLIP/DINO
Sema2: object-time feature aggregation, time-sensitive embedding
Sema3: VLM caption as weak observation, diagnostic only unless cross-validated by visual evidence
```

### 10.4 指标

ScanNet semantic if labels exist：

```text
semantic AP / AP50 / AP25
class-agnostic AP separated from semantic AP
classification accuracy on matched instances
open-vocabulary text retrieval R@1/R@5
```

Dynamic semantic：

```text
time-sensitive query success
object state change retrieval
caption consistency, diagnostic only
```

### 10.5 风险控制

VLM captions 只能作为 weak evidence。必须记录 hallucination guards：

```text
caption source
visual support frames
object mask evidence
language-only claim flag
```

不能让 MLLM 生成的文字直接成为 GT。

---

## 11. 需要停止的方向

从 v8 开始，以下方向不再作为主线，除非作为诊断对照：

```text
1. adaptive top-k / mask_count rescore 作为核心方法。
2. full-mask fringe -> WTA 作为主 object formation。
3. sparse carrier connected component 直接输出 object。
4. mask co-membership 作为强正边。
5. point merge / point NMS / fusion / track bucket suppression 继续扫阈值。
6. own recompute AP 作为唯一成功指标。
7. D4RT sparse geometry direct NN export 作为 ScanNet 主几何路径。
8. Stream3D dense candidate bank 作为内部 backbone。
```

这些方向可以保留为 baseline 和 negative control，但不要再消耗主力时间。

---

## 12. 里程碑和决策树

### 12.1 第 1 天：可审计环境和几何 sanity

必须完成：

```text
Lane 0 audit packet template
Lane 1 G1_grid_dense on scene0050_00 16f/32f
D4RT adapter vs official infer sanity
Sim3/self/cycle diagnostic
```

Day 1 go/no-go：

```text
如果 dense/semi-dense surfel field 比 sparse carrier 没有任何 coverage/consistency提升，先修 D4RT adapter。
如果 adapter 与官方输出不一致，停止后续 object inference。
```

### 12.2 第 2-3 天：observation separability 和第一个 object partition

必须完成：

```text
Lane 2 GT diagnostic pairwise separability
Lane 3A signed clustering prototype
Lane 3B surfacelet prototype quick version
```

Day 3 go/no-go：

```text
如果 combined signal AUC <0.70，先换/加强 2D masks和features，不做 clustering调参。
如果 signed clustering 不超过 v6 compact 的 object quality，检查 failure 是 overmerge还是 fragmentation。
```

### 12.3 第 4-5 天：probe5 fixed-support gate

必须完成：

```text
v8-A/B/C 在 probe5 的 full metrics
gap matrix
failure visualizations
negative control: Stream3D-D4RT geometry probe5 if geometry adapter可运行
```

Day 5 decision：

```text
如果 AP>=0.32/AP50>=0.54/AP25>=0.70，进入 tune30。
如果 AP25高但AP50低，优先 boundary/surfacelet ownership。
如果AP/AP50/AP25都低，回到 surfel field or observation separability。
```

### 12.4 第 6-7 天：Dynamic Replica 和 tune30

必须完成：

```text
Dynamic Replica env fixed or blocker report
ScanNet tune30 for best v8 direction
dynamic qualitative/tracking sanity if data present
```

Full final 启动条件：

```text
1. probe5 gate pass。
2. tune30 不退化。
3. metric integrity clean。
4. dynamic data至少有 sanity结果或明确blocker。
```

---

## 13. Codex 最终交付格式

每轮执行结束必须生成：

```text
docs/stream4d_v8_<phase>_implementation.md
docs/stream4d_v8_<phase>_实验复盘.md
outputs/audit/v8_<phase>_metric_integrity.json
outputs/audit/v8_<phase>_gap_matrix.{json,csv,md,png}
outputs/audit/v8_<phase>_failure_cases.md
logs/stream4d_v8_<phase>_*.log
scripts/reproduce_v8_<phase>.sh
stream4d_v8_code_audit_packet_<timestamp>.zip
```

复盘必须包括：

```text
1. 实际跑了什么，不允许写计划中没跑的内容。
2. 每个 config 的 AP/AP50/AP25、support、#pred、coverage、conflict。
3. 是否使用 GT，若使用只能 diagnostic。
4. 和 Stream3D 的差距，不允许只报自己最好的 own recompute。
5. 失败案例和可视化路径。
6. 下一步根据失败类型自动选择的尝试方向。
```

---

## 14. v8 成功的最低定义

v8 不以“所有指标立刻 SOTA”为唯一成功标准。真正成功是证明我们从错误路线转到了正确科学问题，并用实验给出清楚 yes/no。

最低成功定义：

```text
1. D4RT dense/semi-dense surfel field 的 coverage/consistency 明显优于 sparse carrier。
2. mask-as-measurement separability 证明 2D observations 能支持 object partition，且不再把 co-mask当强正边。
3. surfel object inference 在 probe5 fixed support 上显著超过 v6/v7 Stream4D best。
4. Dynamic Replica 数据要么跑出 tracking sanity，要么明确 blocker，不再伪造或跳过。
5. 所有结果可审计，无虚假指标，无 GT 泄漏。
```

如果这五点中前两点失败，说明 D4RT-native semantic field 当前基础不成立，应停止 ScanNet AP 优化，先修 D4RT surfel field。若前两点成立但第三点失败，说明 object inference 设计仍错，继续研究 signed partition / surfacelet ownership。若前三点成立但动态失败，说明论文主 claim 需要补动态数据而不是继续静态 ScanNet。

---

## 15. 最终提醒

v8 的核心不是“再试一个后处理”。它要回答一个更本质的问题：

**D4RT 的 feed-forward dense 4D correspondence 能否作为 semantic 4D object field 的基本表示？**

如果答案是 yes，ScanNet 上应至少能得到合理的静态退化结果，动态场景上应展示明显 identity/tracking 优势。如果答案是 no，也要诚实地定位失败在 D4RT geometry、observation projection、object inference 还是 evaluation adapter。只有这样，后续才不会继续浪费时间在阈值和工程堆砌上。
