# ACL2 v29B：SemanticKITTI 3D 标注投影 + Video Masklet Quality-aware Semantic Prior Generator 实验计划

日期：2026-05-22  
对象：LoGeR / HMC Pipeline v2 / Video Masklet Front-end / Semantic Prior Generator / SemanticKITTI 3D semantic projection  
主目标：在短时间无法获得 KITTI01 dense 2D semantic GT 的情况下，把 **SemanticKITTI 3D 点级语义标注投影到 KITTI image_2**，作为稀疏但高可信的语义诊断与 masklet 质量校准信号，继续探索语义对 frame attention、global attention、SWA、TTT 四类 memory path 的作用。

---

## 0. 本轮计划的核心变化

v25 原本希望使用 KITTI01 dense 2D GT semantic，但本地数据审计发现 KITTI01 没有可覆盖目标 frame 的 dense 2D GT semantic，GT semantic Phase 0 hard gate 被合法停止。v25B / v26 / v27 / v28 继续使用 Video Masklet Front-end 输出，工程接线越来越完整，但性能仍然很弱。v28 已经做到 fine label、token-exact TTT conflict、token-level scale proxy 都进入 runtime，但 h10 最好 ATE delta 只有约 $-0.36m$，最好 $[200,300)$ delta 约 $-2.37m$，并且 SWA boundary 退化。

本轮不再继续扩大 coarse / fine semantic rule 矩阵，而是引入第三种语义证据：

```text
SemanticKITTI 3D semantic labels
    ↓ projection to KITTI camera image_2
Sparse projected 2D semantic anchors
    ↓
calibrate / verify / correct video masklet labels
    ↓
quality-aware semantic memory role router
```

这不是 dense 2D GT semantic。它只能覆盖 LiDAR 可见点，很多像素没有标签，尤其天空通常没有 LiDAR 返回。因此它不能直接替代 Video Masklet Front-end；它的主要作用是：

```text
1. 作为稀疏高可信语义锚点；
2. 诊断 video masklet label 是否错误；
3. 诊断 video masklet 是否时序一致；
4. 校准 Semantic Prior Generator 的 trust/routing；
5. 判断语义到底有没有 masklet-level causal 上界。
```

一句话：**本轮不是把 SemanticKITTI 投影结果当 dense semantic map 使用，而是把它当作稀疏语义校验器和高可信 memory anchor。**

---

## 1. 实验整体目标

本轮实验不以单个 KITTI01 数字打榜为目标，也不允许为 KITTI01 单独调参。本轮要回答五个更根本的问题。

### 1.1 问题 A：Video Masklet 的语义预测到底有多可信？

之前 v25B / v26 / v27 / v28 证明 video masklet cache 覆盖率、no-op parity、memory path consumption 基本都过了，但效果很弱。这可能有两个解释：

```text
解释 1：video masklet 语义本身不准，label 或时序一致性差；
解释 2：语义即使准确，也不是当前 trajectory drift 的主因果变量。
```

3D semantic projection 可以提供稀疏但可信的对照，用来检查：

```text
1. masklet 预测 label 是否和 LiDAR 语义一致；
2. 同一 masklet 在多帧中的 3D semantic 支持是否稳定；
3. masklet 是否覆盖了多个互相冲突的 3D label；
4. video masklet 的 fine label 和 coarse group 是否过粗或错误。
```

### 1.2 问题 B：Projected 3D semantic 是否能给 memory control 提供比 video-only semantic 更强的上界？

Projected 3D semantic 是稀疏的，但它的 label 更可信。如果语义本身有强因果力，即便只用 LiDAR-visible 区域，也应该在 road、building、wall、fence、vegetation、vehicle 等可见点上产生更清楚的 role signal。

本轮要比较：

```text
Video-only semantic role
Projected-3D-only semantic role
Video-masklet role calibrated by projected 3D semantic
```

### 1.3 问题 C：语义在四条 memory path 中的作用是否不同？

LoGeR 的 memory path 不是同一种东西：

```text
frame/global attention:
    当前窗口的 K/V source routing；

SWA:
    adjacent chunk 的 local source / overlap cache；

TTT:
    compressed global fast-weight memory；

global/chunk source:
    global context 与跨 chunk source 的长期传播。
```

因此，Projected 3D semantic 也不应该生成一个统一的 $A_{tok}$。它应该帮助判断不同 path 中的 role：

```text
R_frame
R_global
R_swa
R_ttt
```

### 1.4 问题 D：3D semantic projection 能否解释 h10 有效但 h15 衰减？

历史实验多次出现：

```text
h10 / short window 有改善；
h15 / longer window 衰减；
full online 不允许启动。
```

本轮需要检查：

```text
1. h10 有效的 masklet 是否有 3D semantic 支持；
2. h15 失效时，相关 masklet 是否被后续 TTT / SWA / global / merge state 覆盖；
3. 3D semantic 支持的结构区域是否能作为 durable anchor；
4. 没有 3D 支持的 predicted semantic 区域是否更容易被洗掉。
```

### 1.5 问题 E：语义路线是否还值得作为 Target-25 主线？

如果用 Projected 3D semantic 校准后，masklet-level oracle 仍然没有明显上界，那么需要诚实地把语义主线降级为辅助诊断，而不是继续扩大语义矩阵。

本轮要建立一个清晰判断：

```text
如果 projected 3D semantic + video masklet causal bank 没有上界：
    语义不是 Target-25 主因果变量；
    转回 TTT-native / scale-state / source-skip / trajectory-state 主线。

如果 projected 3D semantic 有上界但 video masklet 没有：
    语义预测质量是瓶颈；
    优先改 video masklet frontend 和 trust calibration。

如果 projected 3D semantic 与 video masklet 都有上界：
    继续做 quality-aware Semantic Prior Generator。
```

---

## 2. 实验原则

### 2.1 Projected 3D semantic 是 sparse GT，不是 dense GT

SemanticKITTI 3D labels 投影到相机后，只覆盖 LiDAR 点落到图像上的位置。未覆盖像素必须标为 ignore。

禁止做：

```text
1. 把 ignore 当成 sky；
2. 用最近邻把稀疏标签填满整张图后当主实验输入；
3. 把 projected 3D semantic 结果写成 dense 2D GT semantic；
4. 把使用 3D GT projection 的结果写成 deployable semantic success。
```

允许做：

```text
1. 用 projected labels 评估 video masklet label purity；
2. 用 projected labels 校准 masklet trust；
3. 用 projected labels 产生 sparse high-confidence anchors；
4. 用 projected labels 做 diagnostic / upper bound。
```

### 2.2 仍然不针对数据集调参

本轮可以诊断 KITTI01，也可以后续诊断 KITTI00/02/05 或其他带 LiDAR semantic 的序列，但不能做：

```text
KITTI01 专用 label rule
KITTI01 专用 threshold
KITTI01 chunk10 专用策略
某个数据集专用 gamma
```

所有规则必须使用同一套参数。跨数据集只用于观察 failure mode，而不是调参打榜。

### 2.3 先校准 masklet，再做 memory action

本轮不直接把 projected 3D labels 投到 token 后就控制 memory。必须先做 masklet quality audit：

```text
3D semantic projection
    ↓
masklet label purity / support / conflict / temporal consistency
    ↓
trust score T_masklet
    ↓
quality-aware semantic role
    ↓
memory path action
```

### 2.4 先 action distinguishability，再 rollout

如果两个 semantic policy 最后产生的实际 action mask 几乎一样，不允许继续 rollout。必须先检查：

```text
frame_source_keep_mask
swa_cache_keep_mask
ttt_positive_mask
ttt_negative_mask
source_keep_ratio
attention_mass_removed
```

这条原则来自 v28 的关键异常：`SWA_SEM_ONLY / SWA_RISK_ONLY / SWA_SEM_RISK` 得到相同 trajectory delta，说明不同策略可能在实际 action 上等价。

---

## 3. 数据与投影实现

### 3.1 需要的数据布局

优先支持 SemanticKITTI / KITTI Odometry layout：

```text
sequences/01/image_2/{frame:06d}.png
sequences/01/velodyne/{frame:06d}.bin
sequences/01/labels/{frame:06d}.label
sequences/01/calib.txt
```

必须要求：

```text
velodyne frame id 与 image_2 frame id 对齐；
labels frame id 与 velodyne frame id 对齐；
calib.txt 中存在 P2 与 Tr 或 Tr_velo_to_cam；
如果存在 R0_rect，则使用；否则默认 identity。
```

### 3.2 投影公式

对 LiDAR 点 $p_v = [x, y, z, 1]^T$，先变换到相机坐标：

$$
p_c = R_0 \, T_{velo\rightarrow cam} \, p_v
$$

然后用相机投影矩阵 $P_2$ 投到 image_2：

$$
\tilde u = P_2 p_c
$$

像素坐标为：

$$
u = \frac{\tilde u_x}{\tilde u_z}, \quad v = \frac{\tilde u_y}{\tilde u_z}
$$

只保留满足以下条件的点：

```text
p_c.z > 0
0 <= u < W
0 <= v < H
label is not ignore
```

同一个像素有多个 LiDAR 点时使用 z-buffer，保留最近点：

$$
L_{3d}(u,v) = L_i \quad \text{where} \quad z_i = \min_j z_j
$$

未被 LiDAR 点覆盖的像素：

```text
L_3d(u,v) = IGNORE
valid_3d(u,v) = 0
```

### 3.3 语义标签处理

SemanticKITTI `.label` 通常是 uint32，每个点的 semantic id 使用低 16 bits：

$$
label_{sem} = label_{raw} \; \& \; 0xFFFF
$$

本轮要输出两套标签：

```text
L_3d_fine:
    SemanticKITTI original semantic id mapped to fine class name

G_3d_coarse:
    mapped to current memory coarse group
```

示例 coarse mapping：

```text
road, sidewalk, parking, other-ground -> STRUCTURE_GROUND
building, wall, fence, pole, traffic-sign -> STRUCTURE_VERTICAL
car, truck, bus, person, bicyclist, motorcyclist -> MOVABLE_THING
vegetation, trunk, terrain -> LOW_VALUE_OR_AMBIG_STUFF
unlabeled / outlier / other-object -> UNCERTAIN_REGION
ignore / no projection -> IGNORE
```

注意：**no projection 不等于 sky**。Sky 通常没有 LiDAR 返回，不能从 SemanticKITTI projection 中确认 sky。

---

## 4. 新增输出接口

### 4.1 Projected semantic cache

每个 chunk 生成：

```text
stage_c_3dproj_cache/chunk_{idx}_{start}_{end}/
    L_3d_fine_pix.pt        # [T,H,W], int16/int32, ignore for uncovered pixels
    G_3d_coarse_pix.pt      # [T,H,W], int16/int32
    V_3d_pix.pt             # [T,H,W], bool valid projection mask
    Z_3d_pix.pt             # [T,H,W], projected depth, inf for invalid
    projection_debug.json
    per_frame_projection_stats.csv
```

### 4.2 Patch/token projection

同时投影到 patch token：

```text
L_3d_fine_tok
G_3d_coarse_tok
V_3d_tok
N_3d_points_tok
P_3d_purity_tok
```

patch 内聚合规则：

$$
N_{k,i} = \sum_{u \in patch_i} \mathbf{1}[V_{3d}(u)=1] \mathbf{1}[L_{3d}(u)=k]
$$

$$
L^{tok}_{3d,i} = \arg\max_k N_{k,i}
$$

$$
P^{tok}_{3d,i} = \frac{\max_k N_{k,i}}{\sum_k N_{k,i} + \epsilon}
$$

如果 patch 内有效 3D 点数小于 $N_{min}$，则：

```text
V_3d_tok = 0
L_3d_fine_tok = IGNORE
```

### 4.3 Masklet-3D semantic alignment

对每个 video masklet $M_j$，统计其内部 projected 3D labels：

$$
n_{j,k} = \sum_{t,u} M_j(t,u) V_{3d}(t,u) \mathbf{1}[L_{3d}(t,u)=k]
$$

$$
N_j = \sum_k n_{j,k}
$$

$$
purity_j = \frac{\max_k n_{j,k}}{N_j + \epsilon}
$$

$$
label^{3d}_j = \arg\max_k n_{j,k}
$$

输出：

```text
masklet_3d_alignment.csv:
    chunk_id
    masklet_id
    video_label
    video_coarse_group
    projected_label_top1
    projected_group_top1
    projected_support_count
    projected_coverage_ratio
    projected_purity
    video_vs_3d_agree
    conflict_type
    temporal_support_frames
    temporal_label_consistency
```

---

## 5. Trust-gated Semantic Prior Generator

### 5.1 Masklet trust score

本轮新增 $T_j^{sem}$，表示是否信任 video masklet 的语义标签。

建议定义：

$$
T_j^{3d} = \mathbf{1}[N_j \ge N_{min}] \cdot purity_j \cdot q_j
$$

其中 $q_j$ 是 video masklet 自身质量，来自 masklet frontend，例如 mask stability、temporal consistency、Q_mask。

最终 trust：

$$
T_j^{sem} = \max(T_j^{3d}, T_j^{video} \cdot \mathbf{1}[N_j < N_{min}])
$$

解释：

```text
如果有足够 3D projected support，就优先相信 3D projection；
如果没有 3D support，不能直接判定 video label 错，只能降低 trust 或 fallback。
```

### 5.2 Label correction rule

当 3D support 足够且 purity 高时：

```text
if N_j >= N_min and purity_j >= 0.75:
    trusted_label = projected_label_top1
    trusted_group = projected_group_top1
    semantic_source = projected_3d
```

当 3D support 足够但 purity 低时：

```text
trusted_label = video_label or unknown
trusted_group = UNCERTAIN_REGION
semantic_source = conflict
```

当没有足够 3D support 时：

```text
trusted_label = video_label
trusted_group = video_group
semantic_source = video_masklet_only
trust reduced by factor r_no3d
```

### 5.3 Memory role 不直接由 label 决定

Memory role 必须由：

```text
trusted semantic label / group
D_g
C_ttt_conflict_tok
S_scale_risk_tok
Q_mask / T_sem
memory path id
```

共同决定。

例如：

```text
road/building/wall/fence + lowD + low conflict + high trust:
    frame/global source keep
    SWA anchor keep if overlap
    TTT positive long

road/building/wall/fence + high conflict:
    frame/global source maybe keep
    TTT long write blocked or neutral

vehicle/person + highD + high conflict:
    frame/global source skip
    SWA non-overlap source skip
    TTT short negative or no-long-write

vegetation/terrain + lowD + low conflict:
    frame/global soft keep
    SWA neutral
    TTT neutral, not positive long

vegetation/terrain + highD or high conflict:
    frame/global source skip
    TTT short negative or no-long-write

no 3D support but video says sky:
    do not hard skip by default
    neutral context unless D_g/conflict/scale-risk high
```

---

## 6. 核心假设与实验设计

## H0：3D projection cache 和 no-op parity 必须可靠

### 假设

SemanticKITTI projection 可以稳定投到 KITTI01 image_2，且加载该 cache 不扰动 HMC / trajectory。

### 实验

运行：

```text
P0_00_H9_REFERENCE
P0_01_3DPROJ_LOADED_IGNORED
P0_02_3DPROJ_PASS_THROUGH_CONSUMED
P0_03_3DPROJ_DEBUG_ONLY_ALL_PATHS
```

### 记录指标

```text
cache_hit_rate
projection_frame_hit_rate
mean_projected_points_per_frame
mean_projected_pixels_per_frame
focus_200_300_projected_coverage
per_label_projected_pixel_count
pose max abs diff vs H9
raw_trans_max_diff
ATE_delta_vs_H9
hmc_state_hash_diff
```

### 成立标准

```text
projection_frame_hit_rate = 1.0 for needed frames
no-op pose max diff = 0
ATE_delta_vs_H9 = 0
no stale run directory
no predicted fallback used in 3D projection rows
```

如果 H0 不成立：

```text
Codex 先修 projection / frame id / calibration / crop-resize 对齐；
不得启动任何 semantic candidate。
```

---

## H1：Projected 3D semantic 能识别 video masklet label error / inconsistency

### 假设

Video Masklet Front-end 的语义预测可能不准；Projected 3D semantic 可以帮助发现 label error、coarse group error、masklet fragmentation 和 temporal inconsistency。

### 实验

对 chunk 6 / 10 / 16 和全部 focus chunks 生成 masklet alignment：

```text
masklet_3d_alignment.csv
masklet_temporal_consistency.csv
masklet_trust_summary.csv
```

### 记录指标

```text
per-masklet:
    video_label
    projected_label_top1
    projected_support_count
    projected_purity
    video_vs_3d_agree
    trust_score
    temporal_label_flip_rate
    temporal_mask_iou_mean
    temporal_mask_iou_p10
    fragmentation_score
    birth_death_count

per-label:
    agreement_rate
    mean_purity
    mean_support
    D_g_mean
    conflict_mean
    scale_risk_mean
```

### 成立标准

H1 成立不是要求所有 masklet 都准确，而是要求 3D projection 能提供有用的质量分层：

```text
At least 20% of high-area masklets have sufficient 3D support;
trusted masklets show higher temporal stability than untrusted masklets;
semantic disagreements are nontrivial but explainable;
trusted masklets have lower D_g/conflict for structure labels or higher D_g/conflict for dynamic labels.
```

如果 H1 不成立：

```text
若 3D support 太少：
    降低 3D projection 到 diagnostic-only，不用于 role control；
    继续 video masklet quality-only plan。

若 3D projection 与视频明显错位：
    Codex 检查 calib / image resolution / frame id / z-buffer / crop-resize。

若 3D label 与 video label 冲突很多但 projection 质量正常：
    优先修 video masklet taxonomy / label mapping。
```

---

## H2：Projected 3D semantic + video masklet trust 能产生比 video-only semantic 更强的 single-path memory signal

### 假设

如果语义噪声是主要瓶颈，那么 3D-projected trust-gated semantic role 应该强于 video-only semantic role。

### 实验矩阵

对每条 path 先做 single-path：

```text
FRAME_VIDEO_ONLY
FRAME_3DPROJ_ONLY
FRAME_VIDEO_3DTRUST

GLOBAL_VIDEO_ONLY
GLOBAL_3DPROJ_ONLY
GLOBAL_VIDEO_3DTRUST

SWA_VIDEO_ONLY
SWA_3DPROJ_ONLY
SWA_VIDEO_3DTRUST

TTT_VIDEO_ONLY
TTT_3DPROJ_ONLY
TTT_VIDEO_3DTRUST
```

固定：

```text
parent = H9 causal fork
chunks = 6,10,16
horizons = h10,h15
```

### 记录指标

```text
h10/h15 ATE delta
h10/h15 [200,300) delta
h10/h15 [400,600) delta
Rot delta
FinalErr delta
boundary_10f / boundary_20f for SWA
source_keep_ratio
attention_mass_removed
TTT positive/neutral/negative/no-write mass
per-label role mass
trusted_masklet_ratio
untrusted_fallback_ratio
```

### 成立标准

进入下一阶段必须满足至少一个：

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
or h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -3m
```

同时：

```text
[400,600) regression <= +1m
SWA boundary regression <= +0.25m for SWA candidates
```

如果 H2 不成立：

```text
若 3DPROJ_ONLY 和 VIDEO_ONLY 都弱：
    semantic not a primary causal signal -> downgrade semantic to diagnostic.

若 3DPROJ_ONLY 强而 VIDEO_ONLY 弱：
    video masklet quality is bottleneck -> focus on trust calibration and frontend repair.

若 VIDEO_3DTRUST 与 VIDEO_ONLY action 等价：
    repair role projection / trust threshold before rollout.
```

---

## H3：Masklet-level causal bank 可以发现语义区域在不同 memory path 中的真实角色

### 假设

当前人工 role rule 太粗。真正有效的策略应该先通过 masklet-level intervention 找到 causal role，再归纳规则。

### 实验设计

对每个关键 chunk 选择 top masklets：

```text
chunks = 6, 10, 16
masklets per chunk = 12 to 20
selection features:
    area
    D_g p90
    C_ttt_conflict p90
    S_scale_risk p90
    source attention mass
    projected_3d_support_count
    semantic group diversity
```

对每个 masklet $j$ 和 path $p$ 做少量 action：

```text
frame/global:
    keep source
    soft skip source
    hard skip source

SWA:
    keep cache
    remove previous-source cache
    protect overlap anchor

TTT:
    positive long
    neutral keep
    short negative
    no long write
```

定义 causal effect：

$$
E^{h}_{j,p,a} = ATE^h_{base} - ATE^h_{j,p,a}
$$

$$
E^{[200,300]}_{j,p,a} = Err^{[200,300]}_{base} - Err^{[200,300]}_{j,p,a}
$$

### 记录指标

```text
masklet id / label / group
3D support count
3D purity
video-vs-3D agreement
trust score
path
action
h5/h10/h15 ATE delta
[200,300] delta
[400,600] delta
boundary metrics
source mass changed
TTT update norm changed
action validity flags
```

### 成立标准

Masklet-level semantic route 有上界，当至少一个 action family 满足：

```text
h10 ATE delta <= -3m
or h15 ATE delta <= -3m
or [200,300) delta <= -5m
```

如果 masklet-level oracle 都不成立：

```text
semantic mainline downgrade;
stop expanding semantic role matrix;
return to TTT-native / scale-state / trajectory-state mainline.
```

---

## H4：Projected 3D semantic 能帮助区分 SWA source filtering 与 SWA local-continuity damage

### 假设

SWA 的语义策略不能照搬 TTT。Projected 3D semantic 可以帮助识别哪些 overlap source 是真正的 local alignment anchor。

### 实验

只对 SWA path 做专项：

```text
SWA_KEEP_3D_STRUCTURE_OVERLAP
SWA_SKIP_3D_MOVABLE_NONOVERLAP
SWA_KEEP_ROAD_BUILDING_OVERLAP
SWA_SOFT_SKIP_VEGETATION_HIGHD_NONOVERLAP
SWA_PROTECT_PROJECTED_ANCHORS
```

### 记录指标

```text
boundary_10f_delta
boundary_20f_delta
chunk_boundary_pose_jump_delta
overlap source keep ratio
overlap projected-label mass
removed source attention mass
[200,300] delta
[400,600] delta
```

### 成立标准

```text
[200,300] delta <= -3m
and boundary_10f_delta <= +0.25m
and boundary_20f_delta <= +0.25m
and [400,600] regression <= +1m
```

如果 `[200,300)` 改善但 boundary 退化：

```text
Codex must switch from hard skip to:
    soft keep
    key keep + value attenuation
    non-overlap-only skip
    protect projected structure anchors
```

---

## H5：Projected 3D semantic 如果有上界，应作为 Video Masklet role router 的训练信号，而不是直接替代 masklet

### 假设

Projected 3D semantic 是稀疏的，不能直接作为 dense memory controller。但它可以监督 video masklet role router。

### 实验

如果 H3 oracle 成立，训练或拟合简单可解释 router：

```text
features:
    video fine label
    projected label / agreement / purity
    trust score
    D_g mean / p90
    C_ttt_conflict mean / p90
    S_scale_risk mean / p90
    Q_mask
    mask area
    temporal consistency
    source attention mass
    overlap membership

target:
    best action per masklet per path
```

候选模型：

```text
decision tree
rule list
logistic regression
small random forest
```

### 记录指标

```text
train chunk -> test chunk generalization
action accuracy
weighted action gain
h10/h15 ATE delta
action interpretability report
label-specific failure cases
```

### 成立标准

```text
train chunk10 -> test chunk6/chunk16:
    h10 ATE delta <= -1.5m
    or [200,300] delta <= -3m

No dataset-specific feature:
    no sequence id
    no absolute chunk id as input
    no KITTI01-specific rule table
```

如果 router 只在训练 chunk 有效：

```text
Do not run full online.
Codex should reduce label-specific rules and add causal features like attention mass / conflict / scale-risk.
```

---

## 7. 必须记录的文件

### 7.1 Projection artifacts

```text
projection_cache/
    L_3d_fine_pix.pt
    G_3d_coarse_pix.pt
    V_3d_pix.pt
    Z_3d_pix.pt
    L_3d_fine_tok.pt
    G_3d_coarse_tok.pt
    V_3d_tok.pt
    projection_debug.json
    per_frame_projection_stats.csv
```

### 7.2 Masklet alignment artifacts

```text
masklet_3d_alignment.csv
masklet_trust_summary.csv
masklet_temporal_consistency.csv
per_label_agreement_summary.csv
video_vs_projected_confusion_matrix.csv
```

### 7.3 Memory action artifacts

```text
semantic_action_tensor_summary.jsonl
frame_source_keep_mask.pt
global_source_keep_mask.pt
swa_cache_keep_mask.pt
ttt_role_masks.pt
action_jaccard_matrix.csv
attention_mass_removed.csv
swa_boundary_by_candidate.csv
ttt_update_role_mass.csv
```

### 7.4 Candidate metrics

```text
candidate_bank.csv
candidate_vs_H9_delta_by_horizon.csv
segment_delta_200_300.csv
segment_delta_400_600.csv
boundary_metrics.csv
durability_metrics.csv
```

---

## 8. 必须可视化

### 8.1 Projection visualization

每个 focus chunk 输出：

```text
RGB
Projected 3D labels overlay
Projected valid mask
Projected depth / z-buffer map
Video masklet boundaries
Video label vs projected label disagreement map
```

### 8.2 Masklet trust visualization

```text
masklet trust heatmap
temporal label consistency strip
per-masklet projected purity bar chart
per-label agreement confusion matrix
```

### 8.3 Memory action visualization

```text
R_frame / R_global / R_swa / R_ttt maps
source keep / skip maps
TTT positive / neutral / negative maps
action difference maps between VIDEO_ONLY and VIDEO_3DTRUST
```

### 8.4 Trajectory visualization

```text
H9 vs candidate trajectory XY
segment [200,300] error curve
segment [400,600] error curve
h10-to-h15 durability plot
boundary 10f/20f SWA plot
```

---

## 9. Codex 并行任务分配

### Codex A：SemanticKITTI projection implementation and audit

任务：

```text
1. implement / verify SemanticKITTI projection cache;
2. ensure lower-16-bit semantic id extraction;
3. verify P2 / Tr_velo_to_cam / R0_rect;
4. implement z-buffer;
5. produce projection visualizations;
6. no-op parity gate.
```

失败分流：

```text
if projection_frame_hit_rate < 1:
    inspect frame id alignment;
    inspect KITTI root / sequences/01 paths;
    inspect start_frame/end_frame slicing.

if projected points visually misalign:
    inspect calibration order;
    inspect image resolution / resize / crop;
    inspect z-buffer.
```

### Codex B：Video-masklet vs 3D projection quality audit

任务：

```text
1. compute masklet_3d_alignment;
2. compute label purity and support;
3. compute trust score;
4. identify video label errors;
5. generate confusion matrices.
```

失败分流：

```text
if 3D support too sparse:
    use only high-support masklets;
    don't run role matrix over unsupported regions.

if video-vs-3D disagreement high:
    repair taxonomy mapping first;
    don't run memory control until semantic mapping is trustworthy.
```

### Codex C：Single-path projected semantic memory role

任务：

```text
Run FRAME / GLOBAL / SWA / TTT single-path candidates:
    VIDEO_ONLY
    3DPROJ_ONLY
    VIDEO_3DTRUST
```

失败分流：

```text
if 3DPROJ_ONLY weak and VIDEO_ONLY weak:
    semantic mainline downgrade.

if 3DPROJ_ONLY strong but VIDEO_ONLY weak:
    focus on frontend quality / trust-gated router.

if SWA improves [200,300] but boundary regresses:
    switch to soft keep / key keep + value attenuation / overlap-anchor protection.
```

### Codex D：Masklet causal bank

任务：

```text
1. select top masklets per chunk;
2. generate masklet-specific actions;
3. run h5/h10/h15 short rollout;
4. output causal_effect.csv.
```

失败分流：

```text
if oracle has no upper bound:
    stop semantic mainline.

if oracle has upper bound but actions don't generalize:
    learn role router with causal features, not label-only rules.
```

### Codex E：Durability / washout attribution

任务：

```text
1. trace h10 effective correction;
2. compare h15 state movement;
3. identify whether TTT / SWA / global / merge-gauge washes out correction;
4. recommend persistence mechanism.
```

失败分流：

```text
if TTT washes out:
    try positive long commit for trusted projected structure anchors.

if SWA washes out:
    try source cache persistence / overlap anchor protection.

if merge/gauge washes out:
    semantic memory path is not enough; switch to trajectory-state module.
```

---

## 10. 总体 gate 与停止规则

### 10.1 工程 gate

```text
projection_frame_hit_rate = 1.0
no-op parity exact
no predicted fallback in projected semantic rows
no stale run directory contamination
```

### 10.2 Action distinguishability gate

不同 policy 必须满足至少一个：

```text
Jaccard(action_A, action_B) <= 0.85
or source_keep_ratio difference >= 0.05
or TTT role mass difference >= 0.05
or attention_mass_removed difference >= 0.05
```

否则不跑 rollout。

### 10.3 Single-path gate

```text
h10 ATE delta <= -1.5m
or h10 [200,300) delta <= -3m
or h15 ATE delta <= -1.5m
or h15 [200,300) delta <= -3m
```

### 10.4 Oracle gate

```text
h10 ATE delta <= -3m
or h15 ATE delta <= -3m
or [200,300) delta <= -5m
```

### 10.5 Full online entry gate

```text
h15 ATE delta <= -3m
or h15 [200,300) delta <= -5m

and durability_ratio >= 0.45
and [400,600) regression <= +1m
and SWA boundary regression <= +0.25m if SWA involved
```

---

## 11. 预期结论分支

### 分支 A：Projected 3D semantic 强，video masklet 弱

结论：

```text
语义路线有潜力；video masklet frontend / label trust 是瓶颈。
```

下一步：

```text
改进 video masklet frontend；
用 3D projection 训练/校准 masklet trust；
不要继续 label-rule matrix。
```

### 分支 B：Projected 3D semantic 弱，video masklet 也弱

结论：

```text
语义不是 Target-25 主因果变量。
```

下一步：

```text
Semantic Prior Generator 降级为 diagnostic / weak regularizer；
主线转 TTT-native / scale-state / source-skip / trajectory-state。
```

### 分支 C：Masklet oracle 强，但 learned router 弱

结论：

```text
有上界，但规则学习不够；不能直接 full online。
```

下一步：

```text
增加 causal features；
减少 label-specific hand rules；
做跨 chunk validation。
```

### 分支 D：h10 强、h15 弱

结论：

```text
语义能短期修正，但没有 durable memory persistence。
```

下一步：

```text
做 washout attribution；
决定是 TTT long commit、SWA cache persistence，还是 trajectory-state module。
```

---

## 12. 本轮最终判断标准

本轮成功不要求 full online Target-25，但必须至少回答一个问题：

```text
1. Projected 3D semantic 是否比 video semantic 更强？
2. Video masklet label / temporal consistency 是否是主要瓶颈？
3. Semantic masklet-level causal oracle 是否有上界？
4. 语义是否应该继续作为 Target-25 主线？
```

如果这些问题都没有得到明确答案，说明计划执行失败。

如果答案是“semantic 没有上界”，则这是有价值的负结果：后续不要继续把 Semantic Prior Generator 作为主线，而应转向更直接的 trajectory-state / scale-state correction。
