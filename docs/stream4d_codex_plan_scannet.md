# Stream3D + D4RT：Feed-forward Semantic 4D Reconstruction and Tracking 方案计划书

> 给 Codex 的实现计划。目标是在现有 `Code_Stream3D` 和 `Open-d4rt-main` 代码基础上，做一个 **training-free / frozen-model / feed-forward / streaming-bounded** 的 semantic 4D reconstruction and tracking 原型，并首先在 **ScanNet** 上完成可复现实验。

---

## 0. 先读代码后的关键结论

### 0.1 OpenD4RT 代码结论

已检查 `Open-d4rt-main`，重点文件如下：

- `src/model/d4rt.py`
  - `D4RTModel` 暴露两个关键接口：
    - `encode_video(video, aspect_ratio)`：输入 `[B,T,C,H,W]`，输出 global scene representation。
    - `decode_queries(video, query, memory)`：输入 query dict，输出 query-level 几何/运动预测。
  - query 字段为 `u, v, t_src, t_tgt, t_cam`。
- `src/model/query_embedding.py`
  - query token = Fourier UV embedding + learned time embeddings + local RGB patch embedding。
  - local patch 默认 9x9，对边界细节很重要。
- `src/model/heads.py`
  - 输出包含：`xyz_3d`, `uv_2d`, `visibility`, `displacement`, `normal`, `confidence`。
- `infer_track_3d.py`
  - `_infer_tracks(...)` 可以给一批 source UV 点生成多帧 tracks。
  - 当前默认 anchor-clip 逻辑支持长视频窗口，但不是完整 paper 中 dense all-pixel occupancy-grid tracking。
  - `query_src_indices_global` 已经支持不同 source frame 的 query，这对从多帧 2D mask 采样 carrier 很重要。
- `eval_track3d_in_worldtrack.py`
  - evaluator 默认报告的是 `global median scale alignment` 后的 APD/EPE。
  - 代码里也计算 `sim3` / `sim3_closed`，但这是评估对齐，不应写成方法本身依赖 GT Sim3。

非常重要：**不要把 “D4RT 需要 Sim3 才能用” 写成方法假设。** 这里的 Sim3/scale 主要是 evaluation protocol 或 long-video chunk stitching 里的对齐工具。我们的方法推理阶段不能用 GT Sim3。ScanNet 上如果为了把 D4RT 坐标导出到官方 mesh 点云坐标，需要单独写成 **evaluation/export adapter**，不能把它描述为核心方法。

### 0.2 Stream3D 代码结论

已检查 `Code_Stream3D`，重点文件如下：

- `main.py`
  - 入口先构建局部 mask graph，然后调用 `Stream3D(...)` 做 streaming merging。
  - ScanNet 默认 `frame_id = get_frame_list(10)`，即每 10 帧取一帧。
  - 局部窗口 `Step_num = 20`，FPS ratio 0.05，IoU overlap 0.2，manifold distance 0.05。
- `dataset/scannet.py`
  - 依赖 `data/scannet/processed/{seq}`。
  - 需要 `color/`, `depth/`, `pose/`, `intrinsic/`, `{seq}_vh_clean_2.ply`, 以及 2D mask 输出目录。
- `graph/construction.py`
  - `mask_graph_construction(...)` 使用 `frame_backprojection(...)` 把 2D mask 投影到全局 scene point cloud。
  - 当前范式是：2D mask → depth/pose backprojection → static scene point index set。
- `utils/mask_backprojection.py`
  - `turn_mask_to_point(...)` 依赖 RGB-D、intrinsic、extrinsic 和全局 scene points。
  - `backproject(...)` 里使用了未定义的 `DEPTH_TRUNC`，当前代码导入后会有潜在运行错误。
- `utils/Stream3D.py`
  - `Stream3D(...)` 内部实现 local multi-view set cover、IoU merging、DBSCAN/manifold refining、bbox-based historical update。
  - historical update 的核心仍然是 static scene point ID overlap。
- `semantics/*`
  - 语义是 post-hoc：先得到 object_dict，再用代表性 2D masks 提取 CLIP image features，最后和 label text features 匹配。
- `evaluation/evaluate.py`
  - 读取 `data/prediction/{config}` 或 `{config}_class_agnostic` 下的 `.npz`。
  - 代码里有硬编码 TMP 路径，Codex 需要修掉，否则复现实验容易挂。

### 0.3 当前代码阻塞点

Codex 先处理这些阻塞，不然新模块很难稳定跑：

1. `utils/config.py` import 了 `dataset.demo.DemoDataset`，但当前 zip 没有 `dataset/demo.py`。
2. `graph/construction.py` import 了 `utils.MC_mask_backprojection`，但当前 zip 没有该文件。
3. `utils/mask_backprojection.py` 的 `backproject()` 使用未定义变量 `DEPTH_TRUNC`。
4. `evaluation/evaluate.py` 和 `utils/Stream3D.py` 中存在硬编码 TMP 路径，不能复现。
5. 代码大量 `.cuda()`，没有 device 参数；MVP 可以要求 CUDA，但新模块不要继续扩大这个问题。
6. OpenD4RT 上传 zip 只有 `model.yaml`，没有 `.ckpt` 权重。Codex 运行时必须检查 checkpoint 是否存在，不存在就明确报错并给下载路径提示。
7. `Open-d4rt-main/vis/build_like_demo.py` 引用了 `infer_track_3d.py` 中不存在的旧函数。新工程不要依赖该 demo 文件。

---

## 1. 研究目标和方法边界

### 1.1 目标

把 Stream3D 从：

$$
\text{2D masks} \rightarrow \text{RGB-D projected 3D point masks} \rightarrow \text{historical 3D mask merging}
$$

升级为：

$$
\text{2D masks} \rightarrow \text{D4RT 4D carriers} \rightarrow \text{track-carried semantic 4D field} \rightarrow \text{object-level 4D tracking}
$$

最终输出不是简单的 `M^{1:t}` static point masks，而是可查询的 4D object memory：

$$
\mathcal{S}^{1:t}=\{\mathcal{O}_k\}_{k=1}^{K_t}, \quad
\mathcal{O}_k = (C_k, G_k^{1:t}, F_k^{1:t}, L_k, V_k^{1:t})
$$

其中：

- $C_k$：object 绑定的 D4RT carrier 集合。
- $G_k^{1:t}$：object 在每个时间的 4D geometry support。
- $F_k^{1:t}$：视觉/语言语义证据。
- $L_k$：open-vocabulary semantic state 或 label distribution。
- $V_k^{1:t}$：visibility / confidence / lifecycle。

### 1.2 不能写错的边界

- **Training-free** 指不训练新的 3D/4D segmentation 或 semantic-field 模型，不使用 3D/4D 语义标注监督，不做 per-scene optimization-heavy training。可以使用 frozen D4RT、SAM2/Cropformer、CLIP。
- **Feed-forward** 指每个 sliding clip 由 frozen D4RT 一次 encode，然后 query decode；不是 per-scene 训练。
- **Streaming** 第一版写成 bounded-latency streaming：用长度不超过 D4RT `clip_frames` 的 sliding window，并维护 persistent object memory。
- **4D field** 第一版是 visible-surface semantic 4D field，不要声称补全不可见背面或未观测体积。
- **ScanNet** 是静态室内 benchmark。它能验证 semantic 3D reconstruction / open-vocabulary 3D instance segmentation，但不能证明动态物体 tracking。动态 tracking 后续需要 WorldTrack / TAPVid-3D / dynamic video data。

---

## 2. 问题本质：为什么不能只做 “D4RT depth + Stream3D”

Stream3D 当前公式可概括为：

$$
M^{1:t}=\mathcal{F}_{\text{Loc2His}}\left(M^{t-k+1:t}\mid M^{1:t-k},P^{1:t}\right)
$$

其中 $P^{1:t}$ 是 accumulated point cloud，$M$ 是 point index set。这个表示的根本问题是：**它把物理 surface identity、时间、运动、语义都压成了静态点云索引集合**。

D4RT 的核心接口是：

$$
q_i=(u_i,v_i,t_i^{src},t^{tgt},t^{cam}), \quad
\mathbf{x}_i^{tgt|cam}=D(q_i,F)
$$

这不是单纯的 depth/pose 估计，而是 source-pixel anchored 的 Lagrangian correspondence。正确结合方式是：

$$
\text{semantic observation} \rightarrow \text{carrier } c_i=(u_i,v_i,t_i^{src}) \rightarrow \text{trajectory } \{\mathbf{x}_i^\tau\}_{\tau}
$$

也就是说，Stream3D 的基本单元应从 **3D point mask** 改为 **4D carrier**。

---

## 3. 总体架构

建议新建模块，不要大面积改坏原 Stream3D：

```text
Code_Stream3D/
  stream4d/
    __init__.py
    d4rt_adapter.py
    scannet_stream.py
    carrier_sampler.py
    carrier_store.py
    mask_evidence.py
    local_4d_filter.py
    object_memory.py
    export_scannet.py
    run_scannet.py
    diagnostics.py
  configs/
    stream4d_scannet.json
```

整体流程：

```text
ScanNet RGB frames + precomputed 2D masks
        |
        v
D4RTAdapter: encode window, decode carrier trajectories
        |
        v
CarrierStore: 4D carriers with xyz/uv/visibility/confidence
        |
        v
MaskEvidenceBuilder: assign 2D mask observations to carriers
        |
        v
Local4DFilter: set-cover filtering and carrier-space mask merging
        |
        v
ObjectMemory4D: streaming object identity update
        |
        v
ScanNetExporter: export official .npz masks for ScanNet evaluation
```

---

## 4. 核心数据结构

### 4.1 Carrier

每个 carrier 是 D4RT 语义载体：

$$
c_i=(u_i,v_i,t_i^{src})
$$

D4RT 预测：

$$
\mathbf{x}_i^\tau = D(u_i,v_i,t_i^{src},\tau,t_r), \quad
\mathbf{p}_i^\tau = \pi_i^\tau, \quad
\rho_i^\tau = \sigma(v_i^\tau) \cdot \sigma(s_i^\tau)
$$

其中：

- $\mathbf{x}_i^\tau$：D4RT 参考坐标系下的 3D position。
- $\mathbf{p}_i^\tau$：D4RT 预测的 target UV。
- $v_i^\tau$：visibility logit。
- $s_i^\tau$：confidence logit 或 score。
- $\rho_i^\tau$：用于语义证据融合的可见置信权重。

实现建议：

```python
@dataclass
class CarrierBatch:
    carrier_id: np.ndarray          # [N]
    src_frame: np.ndarray           # [N]
    src_uv: np.ndarray              # [N,2], normalized
    xyz_ref: np.ndarray             # [T,N,3]
    uv_pred: np.ndarray             # [T,N,2], normalized
    visibility_prob: np.ndarray     # [T,N]
    confidence_prob: np.ndarray     # [T,N]
    valid: np.ndarray               # [T,N]
```

### 4.2 MaskObservation

2D mask 不直接投影为 3D point mask，而是作为 carrier 的 noisy semantic observation：

$$
e_{i,\tau}=(c_i, \tau, m_\tau(\mathbf{p}_i^\tau), \rho_i^\tau)
$$

实现建议：

```python
@dataclass
class MaskObservation:
    frame_id: int
    mask_id: int
    carrier_ids: np.ndarray
    weights: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area: int
```

### 4.3 Object4D

```python
@dataclass
class Object4D:
    object_id: int
    carrier_ids: set[int]
    frame_support: dict[int, np.ndarray]
    mask_observations: list[tuple[int, int]]
    feature_sum: np.ndarray | None
    feature_weight: float
    last_seen: int
    birth_frame: int
    state: str  # active / lost / merged
```

---

## 5. 模块实现计划

### 5.1 第一步：先修代码健康问题

Codex 先做最小修复，保证原 Stream3D 能 import：

1. `utils/config.py`
   - 将 `from dataset.demo import DemoDataset` 改成 try/except。
   - 如果 `args.dataset == 'demo'` 且 DemoDataset 不存在，再显式报错。
2. `graph/construction.py`
   - 将 `from utils.MC_mask_backprojection import MC_frame_backprojection` 改成 try/except。
   - 当前代码里该函数调用是注释掉的，所以不要让 import 阻塞。
3. `utils/mask_backprojection.py`
   - 定义 `DEPTH_TRUNC = DEPTH_TRUNC_MAX`，或者把 `backproject()` 的 `depth_trunc=DEPTH_TRUNC` 改为 `DEPTH_TRUNC_MAX`。
4. `evaluation/evaluate.py` 和 `utils/Stream3D.py`
   - 移除硬编码 TMP 路径。
   - 用 `data/TMP/{config}/{scene_id}_pre_points.npy`。
5. 新增一个 `tools/check_env.py`
   - 检查 OpenD4RT checkpoint、CLIP checkpoint、ScanNet processed data、2D mask 文件是否存在。

验收标准：

```bash
python - <<'PY'
import utils.config
import graph.construction
import utils.mask_backprojection
print('Stream3D imports OK')
PY
```

### 5.2 第二步：实现 D4RTAdapter

文件：`stream4d/d4rt_adapter.py`

职责：加载 OpenD4RT model，冻结参数，给输入视频窗口和 carrier source points 输出 trajectories。

接口：

```python
class D4RTAdapter:
    def __init__(self, d4rt_root, model_config, ckpt_path, device='cuda'):
        ...

    def infer_carriers(
        self,
        video_rgb_uint8: np.ndarray,          # [T,H,W,3]
        src_uv_norm: np.ndarray,             # [N,2]
        src_frame_local: np.ndarray,          # [N]
        target_frames_local: np.ndarray | None = None,
        t_cam_local: int = 0,
        query_chunk_size: int = 4096,
    ) -> CarrierBatch:
        ...
```

实现要点：

- 直接使用 `D4RTModel.encode_video()` 和 `decode_queries()`，不要调用 stale demo。
- 输入 resize 到 D4RT config 的 `model.input.image_size`，但 `u,v` 保持 normalized。
- 对每个 carrier 查询所有 target frames：

$$
Q=\{(u_i,v_i,t_i^{src},\tau,t_r) \mid i=1...N, \tau=1...T\}
$$

- `visibility` 用 sigmoid 得到概率。
- `confidence` 先用 sigmoid 规整到 $[0,1]$，不要直接拿 raw logit 当权重。
- 如果 checkpoint 不存在，抛出清晰错误，不要 silently fallback。

### 5.3 第三步：ScanNet 输入流

文件：`stream4d/scannet_stream.py`

职责：复用 `dataset/scannet.py`，但提供更稳定的 streaming API：

```python
class ScanNetStream:
    def __init__(self, seq_name, backbone='SAM2'):
        ...

    def frame_ids(self, stride=10, max_frames=None) -> list[int]:
        ...

    def load_window(self, frame_ids: list[int]) -> dict:
        return {
            'rgb': np.ndarray,          # [T,H,W,3]
            'mask': np.ndarray,         # [T,H,W], instance ids
            'depth': optional,
            'pose': optional,
            'intrinsics': optional,
            'scene_points': optional,
        }
```

注意：ScanNet 第一版可以继续读取 existing precomputed 2D masks，不要把 SAM2/Cropformer 推理纳入 MVP。这样能更快验证 4D fusion 本身。

### 5.4 第四步：carrier sampling

文件：`stream4d/carrier_sampler.py`

目标：从 2D masks 中采样 source carriers，而不是全图密集采样。优先保证小物体不丢。

建议采样策略：

1. 对每个 frame 的每个 non-zero mask，采样最多 `max_points_per_mask` 个 source pixels。
2. 对小 mask 至少采 `min_points_per_mask`，避免 set cover 偏向大物体。
3. 支持三种策略：
   - `uniform_mask_pixels`：MVP 默认。
   - `grid_inside_mask`：可复现、稳定。
   - `boundary_plus_interior`：后续 ablation。

采样后的 carrier 集合：

$$
\mathcal{C}_w = \bigcup_{\tau \in w}\bigcup_{m \in \mathcal{M}_\tau}\text{Sample}(m, \tau)
$$

验收指标：

- 每个 window 输出 carrier 数量可控，例如 5k～30k。
- 每个有效 2D mask 至少有 carrier，除非 mask 面积太小。
- 保存 `carrier_sources.npz` 便于复查。

### 5.5 第五步：mask evidence builder

文件：`stream4d/mask_evidence.py`

输入 `CarrierBatch` 和 window 内 2D mask images，输出每个 2D mask 的 carrier support。

对每个 carrier $c_i$ 和 target frame $\tau$：

1. 检查 $\rho_i^\tau > \rho_{min}$。
2. 检查 predicted UV 在图像范围内。
3. 从 segmentation image 读取 mask id：

$$
\ell_i^\tau = M_\tau(\mathbf{p}_i^\tau)
$$

4. 若 $\ell_i^\tau > 0$，把 carrier 作为该 mask 的 observation。

MVP 可先用 hard assignment；后续可加邻域 soft voting：

$$
P(\ell \mid c_i,\tau)=\sum_{\Delta u,\Delta v} K(\Delta u,\Delta v) \cdot \mathbf{1}[M_\tau(\mathbf{p}_i^\tau+\Delta)=\ell]
$$

不要在第一版引入太复杂的 VLM 判断。先让几何-carrier融合闭环跑通。

### 5.6 第六步：4D local mask filtering

文件：`stream4d/local_4d_filter.py`

复用 Stream3D set-cover 思想，但 universe 从 3D key points 改为 carrier-time observations。

原 Stream3D 做：

$$
\min_{\mathcal{M}\subseteq\mathcal{C}} |\mathcal{M}|, \quad
\text{s.t. } \forall p\in\mathcal{K}, \exists m\in\mathcal{M}:p\in m
$$

新版本做：

$$
\min_{\mathcal{A}\subseteq\mathcal{B}_w} |\mathcal{A}|, \quad
\text{s.t. } \forall z\in\mathcal{K}_w, \exists b\in\mathcal{A}: z\in S(b)
$$

其中：

- $\mathcal{B}_w$ 是 window 内所有 2D mask observations。
- $z=(c_i,\tau)$ 是 carrier-time observation。
- $S(b)$ 是某个 2D mask 覆盖的 carrier-time 集合。
- $\mathcal{K}_w$ 是 key carrier-time observations，可由 carrier FPS 或 mask-balanced sampling 得到。

贪心规则不要只按 mask size 排序，建议按 weighted new coverage：

$$
\text{gain}(b)=\sum_{z\in S(b)\cap U}\rho(z) - \lambda \cdot \text{redundancy}(b)
$$

MVP 可以先实现：

```python
while uncovered:
    choose mask observation with max weighted_new_coverage
    mark covered
```

### 5.7 第七步：carrier-space merging

文件：`stream4d/local_4d_filter.py`

不要用 3D point IoU。对两个 mask observations $a,b$，用 carrier support overlap：

$$
\text{IoC}(a,b)=\frac{|C_a\cap C_b|}{\min(|C_a|,|C_b|)+\epsilon}
$$

其中 $C_a$ 是 mask observation 支持的 carrier set。用 `IoC` 而不是 IoU 的原因是：同一物体不同视角可见区域不同，IoU 太苛刻；containment-style overlap 更适合跨视角 mask matching。

综合边权：

$$
s(a,b)=\lambda_c\text{IoC}(a,b)+\lambda_t\text{TemporalAdj}(a,b)+\lambda_g\text{MotionCoherence}(a,b)+\lambda_f\cos(\mathbf{f}_a,\mathbf{f}_b)
$$

MVP 先实现前两项：

$$
s(a,b)=\lambda_c\text{IoC}(a,b)+\lambda_t\exp(-|t_a-t_b|/\eta)
$$

如果 $s(a,b)>\theta_{local}$，连边，connected components 得到 local object proposals。

### 5.8 第八步：ObjectMemory4D

文件：`stream4d/object_memory.py`

用 persistent object memory 替换 Stream3D historical point mask pool。

每个 local proposal $P_j$ 与历史 object $\mathcal{O}_k$ 的匹配分数：

$$
S(P_j,\mathcal{O}_k)=
\alpha \frac{|C(P_j)\cap C(\mathcal{O}_k)|}{\min(|C(P_j)|,|C(\mathcal{O}_k)|)+\epsilon}
+\beta \cos(\mathbf{f}_{P_j},\mathbf{f}_{\mathcal{O}_k})
+\gamma \text{MotionCompat}(P_j,\mathcal{O}_k)
$$

MVP：先用 carrier overlap + temporal decay。

更新规则：

- 若最高分 $>\theta_{hist}$，merge 到已有 object。
- 若没有匹配，创建新 object。
- 如果 object 若干 window 未出现，标记 `lost`，但不要立即删除。
- 如果 object 重现且 carrier overlap 高，恢复为 active。

这一步的目标是解决 Stream3D 的本质短板：moving/reappearing object 不应依赖 static 3D overlap。

---

## 6. ScanNet 导出与评估

### 6.1 为什么需要单独 export adapter

我们的方法内部输出在 D4RT reference coordinates 和 carrier space 中；但 ScanNet 官方 evaluation 需要对 `{seq}_vh_clean_2.ply` 的每个 vertex 给出 instance mask。因此要写一个 **ScanNet-only export adapter**。

这不是方法本体，而是为了复用 ScanNet AP evaluator。

### 6.2 两种导出模式

文件：`stream4d/export_scannet.py`

#### 模式 A：`rgbd_eval`，MVP 默认

用途：先验证 semantic carrier fusion 是否优于/接近 Stream3D。

做法：用 ScanNet depth/pose 把 object 的 carrier observations 映射到官方 scene point indices。

步骤：

1. 对 object 的每个 frame support，取 carrier predicted UV 或 source UV。
2. 在 ScanNet depth 上取深度，用 ScanNet intrinsics/pose backproject 到 scene world。
3. 用 KDTree 找最近 scene point。
4. 半径内投票给 object。

注意：这个模式用了 ScanNet RGB-D 几何，是 **evaluation adapter**。论文中不能声称这是 RGB-only 4D reconstruction 的推理输出。

#### 模式 B：`d4rt_nn`，研究版

用途：评估 D4RT geometry 是否足够直接导出。

做法：用 D4RT `xyz_ref` 转到 ScanNet scene coordinate，再 nearest-neighbor 到 scene points。

第一版可用 frame0 pose 做参考坐标转换：

$$
\mathbf{x}^{world}=T^{world}_{cam0}\mathbf{x}^{d4rt}_{cam0}
$$

如果尺度有偏，允许使用 depth-only scale calibration：

$$
s=\operatorname{median}_{i\in\Omega}\frac{z_i^{ScanNetDepth}}{z_i^{D4RT}}
$$

这个 scale 只能写作 ScanNet export calibration / evaluation bridge，不是 GT label alignment，也不是核心算法。

### 6.3 输出格式

保持兼容 Stream3D evaluator：

```python
pred_dict = {
    'pred_masks': np.ndarray[bool],   # [N_scene_points, N_instances]
    'pred_score': np.ones(N_instances),
    'pred_classes': np.zeros(N_instances, dtype=np.int32),
}
```

路径：

```text
data/prediction/stream4d_scannet_class_agnostic/{seq}.npz
```

并输出 object_dict，供 open-vocabulary 语义使用：

```text
data/scannet/processed/{seq}/output_{backbone}/object/stream4d_scannet/object_dict.npy
```

`object_dict` 兼容现有 `semantics/get_open-voc_features.py`：

```python
object_dict[obj_id] = {
    'point_ids': np.ndarray,
    'mask_list': list[(frame_id, mask_id, coverage)],
    'repre_mask_list': list[(frame_id, mask_id, coverage)],
    'carrier_ids': optional list[int],
}
```

---

## 7. Codex 具体执行清单

### Phase A：最小可运行基础

- [ ] 修复 import blocker。
- [ ] 修复 `DEPTH_TRUNC`。
- [ ] 修复 TMP hard-code。
- [ ] 新增 `stream4d/` 包。
- [ ] 新增 `stream4d/d4rt_adapter.py`。
- [ ] 新增 `stream4d/scannet_stream.py`。
- [ ] 新增 `stream4d/run_scannet.py` 参数解析。
- [ ] 新增 `tools/check_stream4d_env.py`。

验收命令：

```bash
python -m tools.check_stream4d_env \
  --d4rt-root ../Open-d4rt-main \
  --stream3d-root . \
  --seq-name scene0050_00 \
  --backbone SAM2
```

### Phase B：D4RT carrier extraction smoke test

- [ ] 从 ScanNet `scene0050_00` 读取 16～48 帧 RGB 和 mask。
- [ ] 每个 mask 采样少量 carriers。
- [ ] 调用 D4RTAdapter 得到 `CarrierBatch`。
- [ ] 保存：

```text
outputs/stream4d_debug/scene0050_00/carriers_window000.npz
```

必须包含：

```text
src_frame, src_uv, xyz_ref, uv_pred, visibility_prob, confidence_prob, valid
```

诊断图：

- 每帧 carrier projected UV overlay 到 RGB。
- 每帧 carrier mask assignment overlay。
- 可见 carrier 数量曲线。

### Phase C：mask evidence + local 4D filter

- [ ] 实现 `MaskEvidenceBuilder`。
- [ ] 实现 carrier-time set cover。
- [ ] 实现 carrier-space connected component merging。
- [ ] 保存 local proposals：

```text
outputs/stream4d_debug/scene0050_00/local_props_window000.json
```

诊断字段：

```json
{
  "num_raw_mask_observations": 0,
  "num_selected_mask_observations": 0,
  "num_local_proposals": 0,
  "mean_carriers_per_proposal": 0,
  "carrier_coverage": 0.0
}
```

### Phase D：ObjectMemory4D streaming update

- [ ] 实现 object memory。
- [ ] 支持 sliding windows。
- [ ] 支持 lost/active lifecycle。
- [ ] 输出 memory timeline。

保存：

```text
outputs/stream4d_debug/scene0050_00/object_memory.json
```

### Phase E：ScanNet class-agnostic export

- [ ] 实现 `rgbd_eval` exporter。
- [ ] 实现 `d4rt_nn` exporter。
- [ ] 默认先跑 `rgbd_eval`，保证 AP evaluator 可跑。
- [ ] 输出 `data/prediction/stream4d_scannet_class_agnostic/{seq}.npz`。

验收命令：

```bash
python -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt-main \
  --d4rt-config ../Open-d4rt-main/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt-main/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone SAM2 \
  --frame-stride 10 \
  --max-frames 48 \
  --window-size 48 \
  --max-points-per-mask 32 \
  --export-mode rgbd_eval \
  --output-config stream4d_scannet
```

### Phase F：ScanNet evaluation

单场景 smoke：

```bash
python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_scannet_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --output_file data/evaluation/scannet/stream4d_scannet_class_agnostic.txt
```

全 ScanNet val：

```bash
python -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt-main \
  --d4rt-config ../Open-d4rt-main/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt-main/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --seq-list splits/scannet.txt \
  --backbone SAM2 \
  --frame-stride 10 \
  --window-size 48 \
  --export-mode rgbd_eval \
  --output-config stream4d_scannet

python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_scannet_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class
```

### Phase G：open-vocabulary semantic ScanNet

复用现有语义流程：

```bash
python -m semantics.get_open-voc_features \
  --config stream4d_scannet \
  --seq_name_list scene0050_00 \
  --backbone SAM2

python -m semantics.extract_label_featrues \
  --config stream4d_scannet \
  --backbone SAM2

python -m semantics.open-voc_query \
  --config stream4d_scannet \
  --seq_name scene0050_00 \
  --backbone SAM2

python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_scannet \
  --gt_path data/scannet/gt \
  --dataset scannet
```

---

## 8. Baselines 和 Ablations

### 8.1 必跑 baseline

1. 原 Stream3D-SAM2 on ScanNet。
2. 原 Stream3D-Cropformer on ScanNet。
3. 新方法 `stream4d_scannet` with SAM2。
4. 新方法 `stream4d_scannet` with Cropformer。

### 8.2 必跑 ablation

| Ablation | 目的 |
|---|---|
| no D4RT tracking，只用 source-frame mask | 验证 D4RT carrier correspondence 是否真正有用 |
| no local 4D set cover | 验证 carrier-space noise filtering |
| no object memory，只做 window local | 验证 historical memory |
| IoU merging vs IoC merging | 验证 carrier containment overlap 更适合跨视角 |
| `rgbd_eval` export vs `d4rt_nn` export | 分离 semantic fusion 与 D4RT geometry export 误差 |
| max carriers per mask 8/16/32/64 | 观察性能/速度权衡 |

### 8.3 诊断指标

除了 AP/AP50/AP25，必须记录：

- `carrier_coverage`：有多少 2D mask area 被 carriers 覆盖。
- `carrier_visibility_rate`：D4RT visibility 后保留比例。
- `mean_mask_carrier_count`：每个 2D mask 的 carrier 数。
- `local_selected_mask_ratio`：set cover 选中 mask / 原始 mask。
- `object_reactivation_count`：lost object 被重新匹配次数。
- `export_nn_hit_rate`：carrier/object support 映射到 ScanNet scene point 的成功率。
- `export_conflict_rate`：同一 scene point 被多个 object 竞争的比例。

---

## 9. 重要实现细节和坑

### 9.1 不要误用 D4RT 的 `uv_2d`

先确认 `uv_2d` 的范围。写断言和统计：

```python
print(np.nanmin(uv_pred), np.nanmax(uv_pred), np.nanmean((uv_pred >= 0) & (uv_pred <= 1)))
```

如果发现不是 normalized UV，需要在 adapter 中统一转换。

### 9.2 visibility/confidence 不能硬相信

D4RT 输出 head 是 raw tensor。第一版用：

$$
\rho=\sigma(visibility)\cdot\sigma(confidence)
$$

并保留阈值 `rho_min`。不要把低置信 tracks 强行用于 object memory。

### 9.3 ScanNet exporter 只是评估桥

在 markdown、README、论文草稿里明确写：

- `rgbd_eval` 使用 ScanNet RGB-D/pose 是为了输出官方 vertex masks。
- 方法本体是 carrier semantic field。
- 真正 RGB-only geometry export 看 `d4rt_nn`，但第一版 AP 可能会受到几何尺度/坐标误差影响。

### 9.4 Stream3D 原始 evaluator 有 TMP 依赖

`evaluation/evaluate.py` 会从 TMP 中读取 `_pre_points.npy` 来裁剪 GT。Codex 必须确保新 exporter 写入相同语义的文件，路径稳定：

```text
data/TMP/stream4d_scannet/{scene_id}_pre_points.npy
```

并让 evaluator 从 `--tmp_root` 参数读取，而不是硬编码。

### 9.5 不要全量 dense query 起步

D4RT paper 有 dense all-pixel tracking 思想，但 OpenD4RT 当前可用接口更适合 sparse query。MVP 从 mask-sampled carriers 开始，避免 $O(T^2HW)$ 爆炸。

### 9.6 ScanNet 是静态场景

在结果分析里不要用 ScanNet 声称解决动态 tracking。ScanNet 只验证：

- Feed-forward 4D carrier semantic fusion 能否完成 3D semantic reconstruction。
- 与 Stream3D 的 static point mask merging 相比，是否更稳健。
- 代码框架是否具备 4D tracking 接口。

动态 tracking 后续另设实验。

---

## 10. 最小配置文件建议

新增 `configs/stream4d_scannet.json`：

```json
{
  "dataset": "scannet",
  "frame_stride": 10,
  "window_size": 48,
  "window_stride": 24,
  "backbone": "SAM2",
  "max_points_per_mask": 32,
  "min_points_per_mask": 4,
  "rho_min": 0.35,
  "local_ioc_threshold": 0.25,
  "history_match_threshold": 0.30,
  "lost_tolerance_windows": 3,
  "export_mode": "rgbd_eval",
  "export_nn_radius": 0.05,
  "output_config": "stream4d_scannet"
}
```

---

## 11. 预期结果和里程碑

### M0：代码健康

- Stream3D imports OK。
- OpenD4RT model loads OK。
- 单个 ScanNet scene 数据路径检查 OK。

### M1：carrier extraction

- 单 window 能输出 D4RT carriers。
- overlay 显示 projected UV 大体落在合理区域。
- 可见率不是全 0 或全 1。

### M2：local 4D proposals

- 单 window 输出 local object proposals。
- proposal 数量和 2D masks 数量相比显著减少。
- 小物体不会因为 set cover 全部消失。

### M3：streaming object memory

- 多 window 后 object id 稳定。
- object memory JSON 可读。
- lost/active 逻辑正常。

### M4：ScanNet AP

- 能在 `scene0050_00` 导出 `.npz` 并跑 evaluator。
- 能在 full ScanNet val 跑 class-agnostic AP。
- 与原 Stream3D-SAM2/Cropformer 做表格对比。

### M5：semantic AP

- 能复用 CLIP pipeline 得到 semantic prediction。
- 输出 ScanNet semantic AP/AP50/AP25。

---

## 12. 最终要交付的文件

Codex 完成后应至少有：

```text
Code_Stream3D/
  stream4d/
    d4rt_adapter.py
    scannet_stream.py
    carrier_sampler.py
    carrier_store.py
    mask_evidence.py
    local_4d_filter.py
    object_memory.py
    export_scannet.py
    run_scannet.py
    diagnostics.py
  configs/stream4d_scannet.json
  tools/check_stream4d_env.py
  docs/stream4d_scannet.md
```

输出结果目录：

```text
outputs/stream4d_debug/{seq}/
  carriers_window*.npz
  local_props_window*.json
  object_memory.json
  overlays/*.png

data/prediction/stream4d_scannet_class_agnostic/{seq}.npz
data/prediction/stream4d_scannet/{seq}.npz
data/evaluation/scannet/stream4d_scannet*.txt
```

---

## 13. 一句话定位

这不是 “用 D4RT 预测 depth 再跑 Stream3D”。真正目标是：

$$
\boxed{\text{Stream3D point-mask merging} \Rightarrow \text{D4RT carrier-carried semantic 4D field}}
$$

ScanNet 第一阶段只做静态 benchmark 验证；方法设计必须保留 4D carrier、object memory、time-indexed support 和 tracking query 能力，为后续动态数据集实验铺路。
