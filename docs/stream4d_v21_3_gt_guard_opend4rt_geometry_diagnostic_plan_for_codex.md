# Stream4D v21.3：GT Guard、OpenD4RT 源码对齐、D4RT 几何替换 Stream3D 诊断与 D4RT-Native Pipeline 重建计划

面向 Codex 的执行计划。本版本基于 `stream4d_v21_2_opend4rt_chunking_occupancy_plan_for_codex.md` 修改，专门补上三件必须完成的事：

1. **强制强调并修复历史错误：任何非评估阶段使用 GT / ScanNet RGB-D / pose / mesh / GT Sim3 几何都是错误的。** 过去的 `rgbd_eval`、RGB-D backprojection、ScanNet pose/mesh materialization 结果只能保留为 diagnostic / bridge baseline，不能作为主方法。
2. **D4RT 长序列处理、checkpoint-aware chunking、occupancy dense tracking 必须参考 OpenD4RT 源码和 D4RT paper 的既有设计，不允许 Codex 自己拍脑袋造轮子。** 必须优先复用或严格移植 OpenD4RT 的 encoder/query helpers、anchor-clip 逻辑、sliding-window Sim3 思路和 dense occupancy 算法。
3. **必须完成“D4RT 几何替换 Stream3D 几何”的真正诊断实验。** 这次必须回答：D4RT 几何质量到底会让 Stream3D 掉多少？以前的 export-level diagnostic 不够，必须实现 Stream3D 内部 geometry provider 替换。

本文档所有公式使用 Typora 友好的 `$...$` 或 `$$...$$`，不使用 display 公式方括号语法。

---

## 0. 总体结论：这次必须从 pipeline 层重写，不再修旧 `rgbd_eval` 主线

旧 ScanNet 主线的问题不是某个参数、某个 solver、某个 densifier 写错，而是目标和实现错位。目标是：

```text
RGB video + prepared 2D masks + frozen D4RT
-> feed-forward semantic 4D reconstruction and tracking
```

但旧实现大量结果实际是：

```text
D4RT carrier / surfel evidence
-> select / refine 2D masks
-> use ScanNet RGB-D / pose / mesh bridge to materialize 3D support
-> evaluate AP
```

这条路线不能证明 D4RT-native geometry，也不能证明 RGB-only feed-forward 4D reconstruction。它只能说明：在 ScanNet RGB-D / pose bridge 的帮助下，D4RT temporal evidence 能否辅助 mask selection / postprocess。这个不是目标。

因此 v21.3 的第一原则是：

```text
Method prediction path must use only:
  RGB video
  prepared 2D masks
  frozen D4RT outputs
  D4RT self-alignment between chunks
  image-space / mask / appearance / D4RT consistency signals

Method prediction path must NOT use:
  ScanNet depth
  ScanNet camera pose
  ScanNet mesh
  GT labels
  GT / RGB-D / mesh Sim3 alignment
  RGB-D backprojection
  mesh nearest-neighbor assignment
```

ScanNet RGB-D / pose / mesh / GT / GT Sim3 只允许用于：

```text
evaluation
metric computation
diagnostic geometry attribution
visualization
oracle upper-bound analysis
```

这不是措辞问题，是方法正确性问题。任何非评估阶段使用 GT / RGB-D / pose / mesh 几何的结果，都不能进入 method table。

---

## 1. 必须明确承认的历史错误

### 1.1 旧 pipeline 中非评估阶段使用 GT/RGB-D/pose/mesh 几何是错误的

过去旧主线里有如下步骤：

```text
selected mask pixels
-> RGB-D / pose backproject
-> ScanNet mesh vertices
-> object support
```

这是错误的，原因是：

```text
1. 它把 ScanNet 几何放进了 prediction/export path。
2. 它没有验证 D4RT 是否能作为几何和 4D correspondence backbone。
3. 它让 AP 混杂了 D4RT evidence、RGB-D/pose bridge、support cropping、mesh materialization 等变量。
4. 它无法回答几何差、局部 outlier、chunk scale drift、materialization fail、semantic grouping fail 到底哪个是瓶颈。
```

从本版本开始，旧结果统一降级：

```text
rgbd_eval result:
  diagnostic-only / RGB-D bridge baseline

reliable densification with RGB-D backprojection:
  diagnostic-only / oracle-geometry-assisted mask refinement

Stream3D fixed-support AP with method masks:
  diagnostic-only support attribution
```

禁止写成：

```text
D4RT-native semantic 4D reconstruction result
D4RT geometry replacement result
RGB-only feed-forward 4D reconstruction result
```

### 1.2 Manifest 和 scanner 必须强制执行

所有 artifact 必须写 manifest：

```text
uses_rgbd_for_prediction
uses_pose_for_prediction
uses_scannet_mesh_for_prediction
uses_gt_for_prediction
uses_gt_sim3_for_prediction
uses_d4rt_self_sim3
uses_rgbd_for_evaluation
uses_gt_for_diagnostic
is_method_result
is_diagnostic_only
geometry_source
alignment_source
chunking_policy
opend4rt_reference_policy
```

普通 method scanner 必须拒绝任何满足以下条件的 config：

```text
uses_rgbd_for_prediction = true
uses_pose_for_prediction = true
uses_scannet_mesh_for_prediction = true
uses_gt_for_prediction = true
uses_gt_sim3_for_prediction = true
is_diagnostic_only = true
```

新增测试：

```text
test_native_method_rejects_rgbd_import
test_native_method_rejects_pose_import
test_native_method_rejects_mesh_materialization
test_native_manifest_blocks_gt_sim3_prediction
test_eval_only_allows_gt_sim3_with_diagnostic_flag
```

成功标准：

```text
method_path_forbidden_imports_count = 0
num_method_configs_with_gt_or_rgbd_geometry = 0
num_eval_only_configs_marked_diagnostic = all
scanner_exit_code = 0 only if above holds
```

不满足时 Codex 先做：

```text
1. 把调用 ScanNet depth/pose/mesh 的代码移动到 eval_export_only.py。
2. 给所有旧 rgbd_eval result 补 manifest，并标记 diagnostic-only。
3. 禁止 native method 调用 old export_scannet.py / reliable_densifier.py / mask_backprojection.py。
4. 如果代码确实需要 mesh 输出，只能通过 eval-only adapter 产生，不能参与 method inference。
```

---

## 2. OpenD4RT 源码对齐：不能让 Codex 重新造轮子

Codex 必须先读并复用或等价移植 OpenD4RT 源码中的长序列和 query 逻辑。不能手写一个与 OpenD4RT 语义不一致的 D4RT runner。

### 2.1 必须参考的 OpenD4RT 文件

```text
Open-d4rt-main/infer_track_3d.py
Open-d4rt-main/eval_track3d_in_worldtrack.py
Open-d4rt-main/run_eval_worldtrack.sh
Open-d4rt-main/vis/build_like_demo.py
Open-d4rt-main/vis/build_like_demo_for_worldtrack.py
Open-d4rt-main/src/eval/tasks.py
Open-d4rt-main/checkpoints/*/model.yaml
```

必须在新代码中写出 `OPEND4RT_SOURCE_NOTES.md`，记录每个借鉴点对应的 OpenD4RT 源文件和函数名。

### 2.2 Anchor-clip long tracking：用于 long-range sparse tracking diagnostic

OpenD4RT `infer_track_3d.py` 中已有：

```text
_make_anchor_clip_indices(num_frames, clip_frames, target_idx, source_idx)
_infer_tracks(...)
```

其逻辑是：当长视频 $T$ 大于 checkpoint 支持的 `clip_frames` 时，不把整段视频硬塞进 D4RT，而是为每个 target frame 构造一个合法 local clip，使该 clip 同时包含 source frame 和 target frame，然后 grouped encode/decode，再 scatter 回全局时间轴。

这条路径适合：

```text
long-range sparse point tracking
object re-identification diagnostic
first-frame anchor tracking
D4RT consistency sanity check
```

不适合作为主几何重建，因为：

```text
1. 它没有明确形成 scene-level canonical geometry。
2. 它没有显式 overlap self-Sim3 stitching。
3. 它主要为 sparse tracking benchmark 设计。
```

Codex 要做：

```text
1. 在 stream4d_native/opend4rt_long_video.py 中封装 anchor-clip runner。
2. 尽量复用 OpenD4RT _make_anchor_clip_indices 的逻辑。
3. 写测试确认：source frame 和 target frame 永远在 clip 内；clip length 不超过 checkpoint clip_frames。
4. 该路径只能作为 diagnostic / sparse tracking baseline，不进入主 scene geometry builder。
```

### 2.3 Sliding-window + overlap Sim3：用于主 D4RT-native scene geometry

OpenD4RT `vis/build_like_demo.py` 中有 sliding-window + Umeyama / Sim3 stitching 的设计痕迹：

```text
--umeyama-slide-window
--umeyama-slide-window-dense
_infer_point_cloud_ref0(...)
_predict_camera_branches(...)
```

其核心思路：

```text
long video
  -> sliding windows with overlap
  -> local D4RT point/tube field per window
  -> in overlap frames estimate Sim3 from high-confidence D4RT points
  -> transform current window into previous/global reference
  -> choose better overlapping predictions by confidence
```

但是当前上传的 OpenD4RT zip 中，这条 demo path 不能直接 import，因为 `vis/build_like_demo.py` 试图从 `infer_track_3d.py` import：

```text
_make_sliding_window_clip_ranges
_estimate_overlap_sim3
_apply_sim3_to_xyz
```

而当前 `infer_track_3d.py` 中这些 helper 不完整或不存在。

因此 Codex 不允许直接依赖这个 broken demo import。必须在 `stream4d_native/` 中重写等价 helper，并以 OpenD4RT demo 的调用语义为参考。

必须新增：

```text
stream4d_native/opend4rt_long_video.py
stream4d_native/chunk_alignment.py
stream4d_native/sim3.py
```

最低 helper：

```python
def make_sliding_window_clip_ranges(num_frames: int, clip_frames: int, stride: int | None = None):
    ...

def estimate_overlap_sim3(prev_xyz_qt3, curr_xyz_qt3, prev_vis_qt, curr_vis_qt, prev_conf_qt, curr_conf_qt):
    ...

def apply_sim3_to_xyz(xyz, scale, rot, trans):
    ...

def compose_sim3(a, b):
    ...

def invert_sim3(t):
    ...
```

测试：

```text
test_make_sliding_window_clip_ranges_respects_checkpoint
test_make_sliding_window_clip_ranges_has_overlap
test_estimate_overlap_sim3_recovers_known_transform
test_estimate_overlap_sim3_rejects_low_inliers
test_apply_sim3_to_xyz_batch_shapes
test_compose_and_invert_sim3_roundtrip
```

### 2.4 checkpoint chunk policy：chunk size 必须和 checkpoint 对应

从 OpenD4RT `model.yaml` 读取 checkpoint clip length。不要手写 16/32/48。

规则：

$$
L_{chunk} \leq L_{ckpt}
$$

默认：

$$
L_{chunk}=L_{ckpt}, \quad S_{chunk}=\lfloor L_{ckpt}/2 \rfloor, \quad O_{chunk}=L_{chunk}-S_{chunk}
$$

默认配置：

```text
OpenD4RT 32CLIP checkpoint:
  clip_frames = 32
  temporal_chunk_size = 32
  temporal_chunk_stride = 16
  temporal_chunk_overlap = 16

OpenD4RT 48CLIP checkpoint:
  clip_frames = 48
  temporal_chunk_size = 48
  temporal_chunk_stride = 24
  temporal_chunk_overlap = 24
```

严格禁止：

```text
window_size > checkpoint.clip_frames
window_stride >= window_size for full-scene method
manual chunk_size that does not match checkpoint unless explicitly marked diagnostic
```

必须区分：

```text
temporal_chunk_size:
  输入 D4RT encoder 的时间窗口长度。

query_batch_size:
  D4RT decoder 每批处理多少 point queries，只影响显存和速度，不是时间长度。
```

CLI 必须使用清晰命名：

```text
--temporal-chunk-size
--temporal-chunk-stride
--query-batch-size
```

测试：

```text
test_chunk_size_from_checkpoint_32clip
test_chunk_size_from_checkpoint_48clip
test_window_never_exceeds_checkpoint
test_full_scene_method_requires_overlap
test_query_batch_size_not_used_as_temporal_length
```

---

## 3. D4RT 时空占用状态表：必须作为主路径，不是效率小优化

D4RT paper 的 dense tracking 不是固定 grid 越密越好，而是维护 spatio-temporal occupancy grid，避免对同一物理点在不同帧重复发起 query。这个机制必须进入我们的 D4RT-native tube extraction 主路径。

OpenD4RT checkpoint config 中也有：

```yaml
dense_tracking:
  algorithm: occupancy_grid_tracking_all_pixels
  enable: true
```

即使当前源码中没有完整 production-grade implementation，Codex 也必须按 D4RT paper Algorithm 1 的语义实现，并使用 OpenD4RT 的 query/encode helpers，而不是自写不一致的 decoder 调用。

### 3.1 Occupancy dense tracking 的目的

朴素 dense tracking 会对所有 frame 的所有 pixels 都作为 source 发起 query，复杂度接近：

$$
O(T^2HW)
$$

D4RT occupancy 策略维护：

$$
G \in \{0,1\}^{T \times H \times W}
$$

初始所有 pixel-time states 都是 unvisited。每解码一条 material track，就把它在所有 target frames 上可见、可信的位置标为 visited。下一轮只从 still-unvisited pixels 中采样新的 source points。这样同一物理 surface track 在多个 frames 中只需要被 query 一次，避免重复。

D4RT paper 报告该 adaptive dense tracking 可带来约 5--15× speedup。我们不能忽视这点，因为如果没有 occupancy，就无法有效做 dense / adaptive-dense material tubes。

### 3.2 Semantic occupancy 扩展

基础 pixel occupancy：

```text
G_pixel[t, y, x]:
  spatio-temporal pixel 是否已经被 accepted D4RT tube 解释。
```

为了 semantic 4D reconstruction，需要扩展：

```text
G_mask[t, mask_id]:
  该 2D mask interior 是否已有足够 tube coverage。

G_boundary[t, mask_id]:
  该 mask boundary 是否已有足够 boundary-near tubes。

G_overlap[t, y, x]:
  该 pixel 是否属于 chunk overlap，用于 self-Sim3 anchors。

G_uncertain[t, y, x]:
  high cycle error / low visibility / mask ambiguity 区域，优先补 query。
```

query priority：

```text
1. overlap anchor unvisited pixels
2. large mask interiors with insufficient coverage
3. mask boundaries with insufficient coverage
4. uncertain / high-disagreement regions
5. uniform unvisited pixels
```

### 3.3 必须参考 OpenD4RT query helpers

Codex 必须通过 OpenD4RT helper 或等价 wrapper 发起 query：

```text
src.eval.tasks._model_clip_frames
src.eval.tasks._encode_model_memory
src.eval.tasks._run_model_for_queries
infer_track_3d._build_query_for_targets
infer_track_3d._run_clip_queries_for_target_indices or equivalent
```

禁止：

```text
手写与 OpenD4RT query embedding / timestep indexing 不一致的调用。
手写不检查 clip_frames 的 D4RT runner。
忽略 aspect ratio / resize / normalized uv 处理。
```

### 3.4 Occupancy-guided tube extraction 伪代码

```python
def query_d4rt_tubes_with_spatiotemporal_occupancy(
    d4rt_model,
    frames,
    masks,
    checkpoint_clip_frames,
    query_batch_size,
    coverage_targets,
    query_budget,
):
    assert len(frames) <= checkpoint_clip_frames

    state = SpatioTemporalOccupancyState(
        num_frames=len(frames),
        image_height=frames.shape[1],
        image_width=frames.shape[2],
        masks=masks,
    )

    tubes = []
    total_source_queries = 0

    memory = opend4rt_encode_once(
        d4rt_model=d4rt_model,
        frames=frames,
    )

    while not state.coverage_satisfied(coverage_targets):
        if total_source_queries >= query_budget.max_source_points:
            break

        source_points = state.sample_unvisited_source_points(
            batch_size=query_budget.source_points_per_round,
            priority_order=[
                "overlap_anchor_unvisited",
                "large_mask_interior_uncovered",
                "mask_boundary_uncovered",
                "uncertain_region_uncovered",
                "uniform_unvisited",
            ],
        )

        if len(source_points) == 0:
            break

        tracks = opend4rt_decode_full_tracks(
            d4rt_model=d4rt_model,
            encoded_memory=memory,
            frames=frames,
            source_points=source_points,
            query_batch_size=query_batch_size,
            reference_frame_local=0,
        )

        tracks = filter_tracks_before_marking_occupancy(
            tracks,
            min_visibility=coverage_targets.min_visibility,
            min_confidence=coverage_targets.min_confidence,
            max_cycle_error_px=coverage_targets.max_cycle_error_px,
        )

        for track in tracks:
            tube_id = len(tubes)
            tubes.append(track)

            state.mark_visible_track_as_visited(
                track=track,
                tube_id=tube_id,
                mark_radius_px=coverage_targets.mark_radius_px,
            )

            state.update_mask_interior_coverage(track, tube_id)
            state.update_mask_boundary_coverage(track, tube_id)
            state.update_overlap_anchor_coverage(track, tube_id)
            state.update_uncertainty_map(track, tube_id)

        total_source_queries += len(source_points)

    diagnostics = state.summarize(
        num_source_queries=total_source_queries,
        num_output_tubes=len(tubes),
        naive_source_query_count=len(frames) * frames.shape[1] * frames.shape[2],
    )

    return tubes, diagnostics
```

### 3.5 与 chunk overlap self-Sim3 联动

对于 chunk $j>0$，应使用前一 chunk 的 canonical tubes 对 overlap frames warm start：

```text
previous chunk canonical tubes
-> project / match into current overlap frames using D4RT-only overlap evidence
-> mark already-covered overlap pixels as low priority
-> reserve uncovered overlap regions for new self-Sim3 anchors
```

这个过程只允许使用：

```text
D4RT tubes
D4RT uv
D4RT xyz
D4RT confidence / visibility
chunk overlap frame indices
```

不允许使用：

```text
ScanNet pose
ScanNet depth
ScanNet mesh
GT Sim3
```

### 3.6 必须记录的 efficiency 与 coverage 指标

每个 chunk 必须记录：

```text
uses_spatiotemporal_occupancy
naive_source_query_count
actual_source_query_count
adaptive_speedup_vs_naive
semantic_adaptive_speedup
num_output_tubes
num_visible_track_observations
pixel_occupancy_coverage_mean / p10
mask_interior_coverage_mean / p10
mask_boundary_coverage_mean / p10
overlap_anchor_coverage
unvisited_large_mask_count
unvisited_boundary_count
duplicate_track_rate
redundant_query_rate
coverage_saturation_round
query_budget_hit
encoder_time_sec
query_decode_time_sec
occupancy_update_time_sec
total_d4rt_time_sec
visible_tracks_per_second
```

定义：

$$
\text{adaptive\_speedup\_vs\_naive}
= \frac{T \cdot H \cdot W}{N_{source\_queries}}
$$

mask-aware setting 中记录：

$$
\text{semantic\_adaptive\_speedup}
= \frac{N_{candidate\_mask\_pixels}}{N_{source\_queries}}
$$

注意：query-count speedup 不等于 wall-clock speedup，wall-clock 必须单独记录。

### 3.7 Occupancy 实验分层

必须添加对照：

```text
D0_fixed_grid32
D1_fixed_grid48
D2_mask_aware_fixed
D3_occupancy_dense_uniform
D4_occupancy_dense_mask_aware
D5_occupancy_dense_overlap_warmstart
```

同一个 checkpoint、同一个 temporal chunk policy 下比较：

```text
coverage per query
boundary coverage per query
overlap anchor coverage
self-Sim3 stability
duplicate physical track reduction
wall-clock speed
```

成功标准：

```text
adaptive_speedup_vs_naive >= 5 for dense/all-pixel setting, or explicitly explain scope difference.
semantic_adaptive_speedup >= 3 for mask-aware dense setting.
mask_interior_coverage_mean improves over fixed grid32 at same or lower query count.
mask_boundary_coverage_mean improves over fixed grid32 at same or lower query count.
duplicate_track_rate <= 0.25.
query_budget_hit = false for probe5 default setting.
```

如果 speedup 不足，Codex 先排查：

```text
1. mark_radius_px 是否太小。
2. visible track 是否只标记 source frame 而没有标记 all target frames。
3. uv normalization / D4RT resize mapping 是否错误。
4. occupancy dilation 是否过小或过大。
5. mask-aware scope 是否太小，不适合追求 full all-pixel 5x。
```

没有 occupancy dense tracking 的结果只能叫：

```text
semi-dense D4RT tube diagnostic
```

不能叫：

```text
dense feed-forward semantic 4D reconstruction
```

---

## 4. D4RTNativeSceneBuilder：优先级最高的重写模块

Codex 必须优先实现：

```text
Stream3D/stream4d_native/d4rt_scene_builder.py
Stream3D/stream4d_native/opend4rt_long_video.py
Stream3D/stream4d_native/chunk_alignment.py
Stream3D/stream4d_native/sim3.py
Stream3D/stream4d_native/occupancy_state.py
Stream3D/stream4d_native/occupancy_dense_tracker.py
```

最低接口：

```python
class D4RTNativeSceneBuilder:
    def __init__(self, d4rt_model, checkpoint_config):
        self.clip_frames = read_checkpoint_clip_frames(d4rt_model, checkpoint_config)
        self.temporal_chunk_size = self.clip_frames
        self.temporal_chunk_stride = self.clip_frames // 2
        self.temporal_chunk_overlap = self.temporal_chunk_size - self.temporal_chunk_stride

    def build_chunks(self, rgb_video):
        ...

    def extract_local_tubes_with_occupancy(self, rgb_video, masks_by_frame, chunk):
        ...

    def estimate_overlap_self_sim3(self, previous_chunk, current_chunk):
        ...

    def stitch_to_canonical(self, local_chunks):
        ...

    def build_scene_tubes(self, rgb_video, masks_by_frame):
        ...
```

硬要求：

```text
1. 自动读取 checkpoint clip_frames。
2. 默认 temporal_chunk_size = clip_frames。
3. 默认 stride = clip_frames // 2。
4. assert temporal_chunk_size <= clip_frames。
5. assert overlap > 0 for full-scene method。
6. 使用 occupancy-guided adaptive dense tracking 作为主 tube extraction。
7. fixed grid 只能作为 baseline / ablation。
8. 相邻 chunk 使用 D4RT-only overlap self-Sim3。
9. stitching 失败的 chunk 标记 weak_alignment，不能强 3D merge。
10. 方法内部禁止 RGB-D / pose / mesh / GT Sim3。
11. OpenD4RT helper usage must be documented in OPEND4RT_SOURCE_NOTES.md。
```

新增测试：

```text
test_chunk_size_from_checkpoint_32clip
test_chunk_size_from_checkpoint_48clip
test_window_never_exceeds_checkpoint
test_sliding_window_has_overlap
test_occupancy_marks_all_target_frames
test_occupancy_reduces_duplicate_queries
test_occupancy_warmstart_uses_no_gt
test_self_sim3_recovers_known_transform
test_self_sim3_rejects_low_inlier_alignment
test_no_gt_geometry_in_native_scene_builder
```

---

## 5. Phase A：重写 pipeline guard 与源码对齐审计

### 5.1 目标

在跑任何 AP 前，先证明新 pipeline 不会回到旧错误：

```text
no RGB-D / pose / mesh / GT geometry in prediction path
chunk size follows checkpoint
D4RT long-sequence code follows OpenD4RT source semantics
occupancy tracking is implemented as primary dense tube extraction
```

### 5.2 实验 / 审计命令

```bash
python -m py_compile Stream3D/stream4d_native/*.py Stream3D/tools/native_*.py Stream3D/tests/test_native_*.py
python -m unittest discover Stream3D/tests -p 'test_native_*.py'
python -m Stream3D.tools.scan_native_manifests --require-no-gt-prediction
python -m Stream3D.tools.audit_opend4rt_source_alignment --opend4rt-root Open-d4rt-main
```

### 5.3 必须记录

```text
py_compile_pass
unit_tests_pass
forbidden_import_count
opend4rt_helpers_reused_or_ported
opend4rt_source_notes_present
chunk_size_policy_pass
occupancy_primary_path_present
rgbd_eval_old_path_blocked
```

### 5.4 成功标准

```text
forbidden_import_count = 0
chunk_size_policy_pass = true
occupancy_primary_path_present = true
opend4rt_source_notes_present = true
all native tests pass
```

如果不满足：

```text
不要跑任何 D4RT/native AP。
先修 guard 和 source alignment。
```

---

## 6. Phase B：D4RT geometry quality 拆解实验

### 6.1 目标

必须区分三类问题：

```text
1. D4RT 整体 metric geometry 差。
2. D4RT 局部几何总体可以，但有 outliers。
3. 单 chunk 几何尚可，但 chunk 间 scale / reference drift 严重。
```

### 6.2 设置

对 probe5 scenes，使用 checkpoint-aware chunks：

```text
32CLIP: chunk 32 stride 16 overlap 16
48CLIP: chunk 48 stride 24 overlap 24
```

对比：

```text
B0 fixed_grid32
B1 fixed_grid48
B2 occupancy_dense_uniform
B3 occupancy_dense_mask_aware
B4 occupancy_dense_overlap_warmstart
```

### 6.3 Method-internal metrics

不使用 GT：

```text
uv_in01_rate
cycle_uv_error_p90
self_uv_error_p90
visible_track_length_mean
local_neighbor_stretch_p90
local_neighbor_outlier_rate
trajectory_acceleration_p90
duplicate_track_rate
mask_interior_coverage_mean
mask_boundary_coverage_mean
overlap_anchor_coverage
```

### 6.4 Evaluation-only geometry diagnostics

允许用 RGB-D / GT，只能 diagnostic：

```text
eval_sim3_residual_median
eval_sim3_residual_p90
eval_sim3_inlier_ratio
eval_scale_per_chunk
eval_scale_std
gt_outlier_rate_after_sim3
```

### 6.5 判断标准

单 chunk image-space 通过：

```text
uv_in01_rate >= 0.90
cycle_uv_error_p90 <= 6 px
visible_track_length_mean >= 0.6 * chunk_length
```

局部 outlier 可控：

```text
local_neighbor_outlier_rate <= 0.25
trajectory_acceleration_p90 not exploding
```

如果 image-space 好但 eval Sim3 residual 高：

```text
不能说 D4RT 完全无效；应判定为 metric geometry unstable。
继续自对齐和 D4RT-native qualitative/tube metrics，不可用 GT 修 prediction。
```

如果局部 outlier 高：

```text
Codex 先实现 cycle / visibility / confidence / neighbor-stretch outlier filter。
```

---

## 7. Phase C：chunk overlap self-Sim3 与 scale drift 实验

### 7.1 目标

回答：full-scene D4RT geometry 是否能用 overlap self-Sim3 拼成稳定 canonical frame。

### 7.2 设置

必须使用：

```text
chunk_size = checkpoint.clip_frames
stride = clip_frames // 2
overlap = clip_frames // 2
```

对照：

```text
C0 no stitching
C1 pairwise chain self-Sim3
C2 pairwise-to-reference self-Sim3
C3 local bundle self-Sim3, no GT
C4 eval-only GT/RGB-D Sim3, diagnostic only
```

### 7.3 记录指标

```text
num_chunks
overlap_frames
overlap_anchor_count
self_sim3_scale_per_pair
self_sim3_scale_std
self_sim3_residual_median
self_sim3_residual_p90
self_sim3_inlier_ratio
accumulated_scale_drift
alignment_fail_count
weak_alignment_chunk_count
```

Evaluation-only：

```text
gt_sim3_scale_per_chunk
gt_sim3_residual_p90
self_vs_gt_scale_delta
self_vs_gt_transform_delta
```

### 7.4 成功标准

```text
overlap_anchor_count_per_pair >= 200
self_sim3_inlier_ratio >= 0.60
scale_std <= 0.10
accumulated_scale_drift <= 0.20
```

如果失败：

```text
1. 增加 overlap。
2. 增加 overlap occupancy priority。
3. 用 stricter cycle/visibility/confidence gate。
4. 用 pairwise-to-reference 或 local bundle 替代链式累积。
5. 如果 GT Sim3 可以救但 self-Sim3 不行，说明 self-alignment 是瓶颈，不能启动 full-scene object memory。
```

---

## 8. Phase D：真正完成 D4RT 几何替换 Stream3D 的诊断实验

这是本版本必须完成的核心 deliverable。以前只做过 export-level / support-level diagnostic，不够。这次必须把 D4RT 几何接进 Stream3D 内部 pipeline，回答：

```text
如果只替换几何，不换 Stream3D 的 set-cover / manifold / historical merge，性能掉多少？
```

### 8.1 目标

拆开几何质量和算法质量：

$$
\Delta_{geometry}^{Stream3D}
=
AP(Stream3D, G_{RGBD}) - AP(Stream3D, G_{D4RT})
$$

如果 D4RT 几何一换就大幅下降，说明 D4RT geometry 是强瓶颈；不能用 D4RT geometry 下的低 AP 判断 semantic algorithm。

### 8.2 必须实现真正 GeometryProvider 替换

新增：

```text
Stream3D/geometry_provider/base.py
Stream3D/geometry_provider/rgbd_provider.py
Stream3D/geometry_provider/d4rt_raw_provider.py
Stream3D/geometry_provider/d4rt_self_stitched_provider.py
Stream3D/geometry_provider/d4rt_eval_sim3_provider.py
```

必须让 Stream3D 的以下步骤通过 provider 取几何：

```text
2D-to-3D mask projection
local point cloud construction
set-cover key point universe
neighbor point merging
manifold refining distance
local-to-historical mask localization
historical merge geometry
```

禁止只在最终 export 阶段替换点。那不回答问题。

### 8.3 对照实验

```text
G0: Stream3D + original RGB-D / pose geometry
G1: Stream3D + raw D4RT local geometry, no stitching
G2: Stream3D + D4RT self-stitched geometry, no GT
G3: Stream3D + D4RT eval-only GT/RGB-D Sim3 aligned geometry
G4: Stream3D + D4RT eval-only Sim3 + local outlier filtering
G5: Stream3D + D4RT eval-only Sim3 + density-normalized Stream3D thresholds
G6: Stream3D + D4RT self-stitched + density-normalized thresholds
```

### 8.4 记录指标

Performance：

```text
AP/AP50/AP25
pre_points %
prediction union %
GT crop/full
#pred
```

Geometry provider diagnostics：

```text
projection_hit_rate
local_point_count
local_point_density
setcover_keypoint_count
mean_points_per_2d_mask
mask_projection_empty_rate
manifold_distance_distribution
historical_merge_candidate_count
merge_success_rate
```

D4RT geometry diagnostics：

```text
self_stitch_scale_drift
self_stitch_residual_p90
eval_sim3_residual_p90
local_outlier_rate
chunk_alignment_fail_count
```

### 8.5 必须输出的结论表

Codex 必须生成：

```text
D4RT_geometry_replacement_stream3d_probe5.md
D4RT_geometry_replacement_stream3d_probe5.csv
D4RT_geometry_replacement_stream3d_probe5.json
```

表格必须包含：

```text
G0 RGBD baseline
G1 D4RT raw
G2 D4RT self-stitched
G3 D4RT eval-Sim3
G4 D4RT eval-Sim3 + outlier filter
G5 D4RT eval-Sim3 + density thresholds
G6 D4RT self-stitched + density thresholds
```

并计算：

$$
\Delta_{D4RT\_evalSim3} = AP(G0) - AP(G3)
$$

$$
\Delta_{selfStitch} = AP(G3) - AP(G2)
$$

$$
\Delta_{outlier} = AP(G4) - AP(G3)
$$

$$
\Delta_{densityThreshold} = AP(G5) - AP(G3)
$$

### 8.6 判断标准

```text
If G3 << G0:
  D4RT geometry/materialization is insufficient even when aligned to GT/RGB-D for evaluation.

If G3 close to G0 but G2 bad:
  self-stitching / chunk scale drift is the main bottleneck.

If G4 >> G3:
  local outliers are the main bottleneck.

If G5 >> G3:
  Stream3D meter-scale thresholds are incompatible with D4RT geometry density/scale.

If G2 close to G0:
  D4RT self-stitched geometry is viable for Stream3D-like geometric reasoning.
```

This diagnostic must not enter the main method table, because G3/G4/G5 use evaluation-only GT/RGB-D Sim3. But it must be completed before claiming anything about D4RT geometry quality.

如果 Codex 做不到 provider 替换：

```text
不要再交 export-level diagnostic。
必须明确标记 blocker: Stream3D geometry provider replacement not implemented.
```

---

## 9. Phase E：D4RT-native semantic 4D object formation

只有 Phase B/C/D 给出清楚几何结论后，才启动语义 object formation。

### 9.1 输入

```text
RGB video
prepared 2D masks
D4RT canonical tubes from Phase C
```

禁止：

```text
RGB-D backprojection
ScanNet pose
ScanNet mesh
GT Sim3
```

### 9.2 Mask-as-measurement

每个 mask $m_{t,r}$ 转成 measurement：

```text
inside_tubes
visible_outside_tubes
boundary_crossing_edges
boundary_safe_merge_edges
appearance feature
ambiguity score
boundary risk
```

### 9.3 TubeCover

选择可靠 measurements：

$$
\mathcal{M}^*
=
\arg\min_{\mathcal{M}'}
|\mathcal{M}'|
+
\alpha U(\mathcal{M}')
+
\beta B(\mathcal{M}')
+
\gamma T(\mathcal{M}')
$$

记录：

```text
num_measurements_before/after
tube_node_coverage_before/after
tube_edge_coverage_before/after
boundary_conflict_before/after
temporal_inconsistency_before/after
D4RT-real vs shuffle vs no-temporal differences
```

### 9.4 Tube graph partition

构造 signed graph：

$$
E(Y)
=
\sum_{(i,j)}
w_{ij}^{cut}\mathbf{1}[y_i=y_j]
+
w_{ij}^{merge}\mathbf{1}[y_i \neq y_j]
+
\Omega(Y)
$$

输出 object tubes：

```text
object_id
tube_ids
per-frame visible tubes
canonical xyz trajectory
uv trajectory
visibility/confidence
measurement history
chunk provenance
alignment uncertainty
```

### 9.5 成功标准

Method-internal：

```text
D4RT-real > shuffle/no-temporal in tube consistency metrics
object temporal length_mean >= 0.5 * chunk length
unknown_tube_ratio <= 0.5
alignment-uncertain chunks do not cause aggressive merge
```

Evaluation-only：

```text
ScanNet export allowed only after method output is frozen.
AP cannot be used to tune method unless explicitly in tune split.
```

---

## 10. Phase F：Evaluation-only ScanNet export

该阶段只用于评估，不参与方法。

### 10.1 Variants

```text
E0 nearest mesh after eval Sim3
E1 component fill after eval Sim3
E2 mask-region fill after eval Sim3, diagnostic only
E3 qualitative no-mesh D4RT tube visualization
```

### 10.2 记录

```text
eval_alignment_anchor_count
eval_sim3_residual_median/p90
mesh_vertex_coverage
covered_gt_instance_ratio
exported_objects
AP/AP50/AP25
pre_points %
GT crop/full
```

### 10.3 Stop rule

```text
If mesh coverage < 0.15 or covered_gt_instance_ratio < 0.50:
  Do not claim ScanNet AP success.
  Report D4RT-native 4D tube metrics and geometry diagnostic instead.
```

---

## 11. Execution order for Codex

### Week 1 / 必须先做

```text
1. Phase A: guards + OpenD4RT source alignment audit.
2. Implement D4RTNativeSceneBuilder.
3. Implement checkpoint-aware chunking.
4. Implement occupancy dense tracking using OpenD4RT query helpers.
5. Implement self-Sim3 helper with unit tests.
6. Start Phase D geometry provider abstraction for Stream3D.
```

### Week 2 / 几何诊断

```text
1. Run Phase B local geometry split: overall geometry vs local outlier.
2. Run Phase C chunk self-Sim3 scale drift audit.
3. Complete Phase D Stream3D geometry provider replacement.
4. Produce geometry impact table G0-G6.
```

### Week 3 / 语义 object formation only if geometry path is interpretable

```text
1. Build mask-as-measurement bank.
2. Run TubeCover controls: real / shuffle / no-temporal / mask-only.
3. Run tube graph partition.
4. Freeze method output.
5. Run evaluation-only ScanNet export.
```

---

## 12. What Codex must not do

```text
Do not use rgbd_eval inside native prediction path.
Do not use ScanNet depth/pose/mesh inside method.
Do not use GT Sim3 to align chunks inside method.
Do not report export-level D4RT geometry diagnostic as Stream3D geometry provider replacement.
Do not use fixed grid only and call it dense D4RT reconstruction.
Do not set chunk_size manually without reading checkpoint clip_frames.
Do not run non-overlapping chunks for full-scene method.
Do not optimize AP before geometry diagnostics are completed.
```

---

## 13. Expected deliverables

Codex must deliver:

```text
stream4d_native source code
OPEND4RT_SOURCE_NOTES.md
native manifest scanner
checkpoint-aware chunking tests
occupancy dense tracking tests
self-Sim3 tests
Phase A audit report
Phase B local geometry report
Phase C chunk scale drift report
Phase D Stream3D geometry replacement report
Phase E object tube output, if geometry passes
Phase F evaluation-only export, if object tubes exist
```

The most important deliverable is not AP. It is the geometry attribution table:

```text
G0 Stream3D RGB-D
G1 Stream3D D4RT raw
G2 Stream3D D4RT self-stitched
G3 Stream3D D4RT eval-Sim3
G4 Stream3D D4RT eval-Sim3 + outlier filter
G5 Stream3D D4RT eval-Sim3 + density thresholds
G6 Stream3D D4RT self-stitched + density thresholds
```

Without this table, we still do not know whether D4RT geometry quality is the bottleneck.

