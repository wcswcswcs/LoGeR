# ACL2 v29C: SemanticKITTI Download + Sparse 3D-to-2D Projection + VideoMasklet Quality-Aware Semantic Prior Plan

日期：2026-05-23  
对象：LoGeR / HMC Pipeline v2 / Video Masklet Front-end / Semantic Prior Generator / SemanticKITTI sparse projection  
主目标：在本地 KITTI01 缺少 `velodyne/*.bin`、`labels/*.label` 和有效 `Tr / Tr_velo_to_cam` 的情况下，指导 Codex 从官方数据源补齐 SemanticKITTI / KITTI Odometry 必需数据，并将其作为 **稀疏 2D 语义锚点** 用来校验 video masklet，而不是把 predicted masklet 冒充 GT。

---

## 0. 本轮为什么要改计划

v29B 已经合法停止在 Phase 0 projection hard gate：本地 KITTI01 有 `image_2` 和 `calib.txt`，但没有 `velodyne`、没有 `labels`，本地 `calib.txt` 也没有可用于 LiDAR-to-camera 投影的 `Tr / Tr_velo_to_cam`，所以 projection frame hit rate 是 `0 / 757 = 0.0`。在这种情况下，Codex 不能继续做 SemanticKITTI 3D projection，也不能把 VideoMasklet predicted semantic fallback 写成 projected 3D semantic。

因此 v29C 的目标不是继续扩大 SemanticPrior 矩阵，而是先建立一个可靠的数据获取与投影链路：

```text
Official SemanticKITTI / KITTI Odometry data
    ↓
sequence 01 velodyne + labels + calib
    ↓
LiDAR point semantic labels projected into image_2
    ↓
sparse 2D semantic anchors
    ↓
video masklet label / consistency / trust calibration
    ↓
quality-aware Semantic Prior Generator
    ↓
frame / global / SWA / TTT path-specific memory role
```

关键边界：SemanticKITTI projection 是 **sparse semantic anchor**，不是 dense 2D semantic GT。未投影到的像素必须是 `ignore`，不能补成 sky、road 或 background。

---

## 1. 数据源原则

### 1.1 只从官方或用户允许的数据源下载

Codex 首选官方来源：

```text
SemanticKITTI official dataset page:
    provides SemanticKITTI label data
    instructs users to also download KITTI Odometry velodyne and calibration

KITTI Odometry official page:
    provides odometry velodyne laser data
    provides odometry calibration files
```

如果官方下载需要登录、403、license confirmation 或人工操作，Codex 不允许绕过。Codex 必须生成一个 `MANUAL_DOWNLOAD_REQUEST.md`，列出需要用户手动下载的文件、目标路径和验证命令。

允许的下载文件：

```text
Required:
    data_odometry_velodyne.zip
    data_odometry_calib.zip
    data_odometry_labels.zip

Optional:
    data_odometry_color.zip  # only if local image_2 is missing; current local KITTI01 image_2 已存在，不优先下载
```

禁止：

```text
1. 不允许用 Mask2Former / VideoMasklet predicted semantic 伪装成 GT projection。
2. 不允许把 KITTI 2015 semantic benchmark 的 200 张 dense semantic 图强行匹配到 KITTI Odometry 01。
3. 不允许从不明 mirror 下载，除非用户明确批准。
4. 不允许下载失败后继续跑 semantic projection candidate。
```

### 1.2 本轮不为 KITTI01 打榜调参

SemanticKITTI 3D projection 只用于诊断：

```text
1. video masklet label 是否经常错？
2. video masklet temporal consistency 是否可靠？
3. 哪些 semantic group 和 D_g / TTT conflict / scale risk 之间有关系？
4. 哪些 masklet 在 memory path 中有因果作用？
```

不得做：

```text
KITTI01 专用 label value
KITTI01 专用 threshold
KITTI01 专用 chunk id rule
sequence-specific semantic role table
```

可以做：

```text
用 KITTI01 诊断 failure mode；
用同一套规则在 KITTI00/02/05 或其他数据上看模式是否一致；
不同数据集只报告差异，不针对数据集调参。
```

---

## 2. Codex Phase 0A：数据获取计划

### 2.1 设置目录

Codex 应创建独立数据根目录，不要把下载 zip 混进当前 LoGeR results：

```bash
export SEMKITTI_ROOT=/mnt/data/users/chengshun.wang/data/semantickitti_odometry
mkdir -p "$SEMKITTI_ROOT/downloads"
mkdir -p "$SEMKITTI_ROOT/dataset"
```

最终目标结构应为：

```text
$SEMKITTI_ROOT/dataset/sequences/01/
    image_2/              # 可以软链到现有 KITTI01 image_2
    calib.txt             # 必须来自 KITTI odometry calib zip，且含 P2 和 Tr 或 Tr_velo_to_cam
    velodyne/000000.bin
    velodyne/000001.bin
    ...
    labels/000000.label
    labels/000001.label
    ...
```

如果本地已有 `image_2`，Codex 可以软链：

```bash
ln -s /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
      "$SEMKITTI_ROOT/dataset/sequences/01/image_2"
```

### 2.2 下载命令模板

Codex 应优先使用可恢复下载：

```bash
cd "$SEMKITTI_ROOT/downloads"

# KITTI Odometry velodyne point clouds, official KITTI download.
aria2c -c -x 8 -s 8 -k 1M \
  -o data_odometry_velodyne.zip \
  'https://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip'

# KITTI Odometry calibration files, official KITTI download.
aria2c -c -x 4 -s 4 -k 1M \
  -o data_odometry_calib.zip \
  'https://www.cvlibs.net/download.php?file=data_odometry_calib.zip'

# SemanticKITTI labels.
aria2c -c -x 8 -s 8 -k 1M \
  -o data_odometry_labels.zip \
  'https://semantic-kitti.org/assets/data_odometry_labels.zip'
```

如果 `aria2c` 不存在，用 `wget -c`：

```bash
wget -c -O data_odometry_velodyne.zip 'https://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip'
wget -c -O data_odometry_calib.zip 'https://www.cvlibs.net/download.php?file=data_odometry_calib.zip'
wget -c -O data_odometry_labels.zip 'https://semantic-kitti.org/assets/data_odometry_labels.zip'
```

### 2.3 如果下载被登录页面拦截

有些环境中 KITTI 官方下载可能返回 HTML 登录页，而不是 zip。Codex 必须检查文件类型：

```bash
file data_odometry_velodyne.zip
file data_odometry_calib.zip
file data_odometry_labels.zip
```

必须满足：

```text
*.zip -> Zip archive data
```

如果得到：

```text
HTML document
text/html
```

Codex 必须停止，并生成：

```text
$SEMKITTI_ROOT/MANUAL_DOWNLOAD_REQUEST.md
```

其中写明：

```text
Please manually download:
1. KITTI Odometry Velodyne point clouds: data_odometry_velodyne.zip
2. KITTI Odometry calibration files: data_odometry_calib.zip
3. SemanticKITTI label data: data_odometry_labels.zip

Then place them in:
$SEMKITTI_ROOT/downloads/
```

不允许继续 projection。

### 2.4 磁盘预算 gate

Velodyne zip 约 80GB，解压后更大。Codex 必须先检查：

```bash
df -h "$SEMKITTI_ROOT"
du -sh "$SEMKITTI_ROOT/downloads" || true
```

最低建议：

```text
free disk >= 180GB
```

如果磁盘不足：

```text
1. 不启动下载；
2. 写 DISK_SPACE_BLOCKER.md；
3. 建议用户提供 sequence 01 only 的已解压 velodyne / labels / calib；
4. 或者将 SEMKITTI_ROOT 指向更大磁盘。
```

---

## 3. Codex Phase 0B：解压与 sequence 01 对齐

### 3.1 只解压需要的 sequence 01

虽然 zip 需要整体下载，但解压时只取 `sequences/01`：

```bash
cd "$SEMKITTI_ROOT"

unzip -q downloads/data_odometry_velodyne.zip \
  'dataset/sequences/01/velodyne/*' \
  -d .

unzip -q downloads/data_odometry_calib.zip \
  'dataset/sequences/01/calib.txt' \
  -d .

unzip -q downloads/data_odometry_labels.zip \
  'dataset/sequences/01/labels/*' \
  -d .
```

如果 zip 内部路径不是 `dataset/sequences/01/...`，Codex 应先列出：

```bash
unzip -l downloads/data_odometry_velodyne.zip | head -100
unzip -l downloads/data_odometry_calib.zip | head -100
unzip -l downloads/data_odometry_labels.zip | head -100
```

然后写入 `ZIP_LAYOUT_SCAN.md`，不要猜路径。

### 3.2 软链 image_2

如果 `$SEMKITTI_ROOT/dataset/sequences/01/image_2` 不存在：

```bash
ln -s /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
      "$SEMKITTI_ROOT/dataset/sequences/01/image_2"
```

### 3.3 校准文件要求

Codex 必须解析：

```text
P2
Tr or Tr_velo_to_cam
```

允许别名：

```text
Tr
Tr_velo_to_cam
Tr_velo_cam
```

如果只找到本地旧 calib 且没有 `Tr`，不要使用它。必须使用 downloaded odometry calibration zip 中的 `calib.txt`。

如果 downloaded calib 也缺失 `Tr`：

```text
1. stop projection;
2. write CALIB_BLOCKER.md;
3. do not invent transform;
4. do not use identity transform.
```

---

## 4. Codex Phase 0C：文件完整性 audit

Codex 必须新增或扩展：

```text
tools/v29c_semantickitti_download_and_projection_audit.py
```

该脚本应输出：

```text
implementation_audit/download_audit_summary.json
implementation_audit/download_audit_report.md
implementation_audit/sequence01_file_counts.csv
implementation_audit/calib_parse_report.json
implementation_audit/frame_hit_audit.csv
implementation_audit/label_velodyne_point_count_audit.csv
```

### 4.1 文件 count gate

检查：

```text
image_2 count
velodyne count
labels count
calib exists
```

对 selected frames：

```text
chunks = 6,10,16
horizons = 10,15
```

必须满足：

```text
projection_frame_hit_rate >= 0.95
labels_hit_rate >= 0.95
velodyne_hit_rate >= 0.95
calib_parse_pass = true
```

否则：

```text
phase0_data_gate_pass = false
no projection cache
no masklet calibration
no candidate rollout
```

### 4.2 label 与 point count gate

每个 frame：

```python
points = np.fromfile(velodyne_bin, dtype=np.float32).reshape(-1, 4)
labels = np.fromfile(label_file, dtype=np.uint32)
assert points.shape[0] == labels.shape[0]
semantic_id = labels & 0xFFFF
instance_id = labels >> 16
```

必须记录：

```text
frame_id
num_points
num_labels
count_match
unique_semantic_ids
```

如果 count mismatch：

```text
stop projection for that frame;
mark sequence invalid unless mismatch rate <= 1% and mismatched frames not in selected windows.
```

### 4.3 calibration parse gate

记录：

```text
P2 shape = 3x4
Tr shape = 3x4 or 4x4
R0_rect optional; if missing use identity and log explicitly
```

不要默默假设。

---

## 5. Codex Phase 1：SemanticKITTI sparse projection cache

### 5.1 Projection algorithm

对每个 selected frame：

```text
1. load velodyne points [N,4]
2. load labels uint32 [N]
3. semantic_id = label & 0xFFFF
4. instance_id = label >> 16
5. transform points from velodyne to camera using Tr / Tr_velo_to_cam
6. rectify with R0_rect if present
7. project with P2
8. keep z > 0
9. keep u in [0,W), v in [0,H)
10. z-buffer: if multiple points land on same pixel, keep nearest depth
11. output sparse image-aligned semantic map
```

Outputs per frame:

```text
projection_cache/seq01/{frame:06d}_sem_sparse.npy
projection_cache/seq01/{frame:06d}_inst_sparse.npy
projection_cache/seq01/{frame:06d}_depth_sparse.npy
projection_cache/seq01/{frame:06d}_valid_mask.npy
projection_cache/seq01/{frame:06d}_meta.json
```

Pixel convention:

```text
unprojected pixels semantic = IGNORE_ID = 255
valid_mask = 0 for unprojected pixels
```

Do not densify.

### 5.2 Projection quality metrics

For each frame:

```text
num_projected_points
num_unique_projected_pixels
projected_pixel_coverage = unique_pixels / (H*W)
per_label_projected_pixel_count
per_label_depth_mean
per_label_depth_p50
per_label_depth_p90
```

For chunks:

```text
chunk_projected_pixel_coverage_mean
chunk_projected_pixel_coverage_p10
chunk_projected_label_entropy
focus_200_300_projected_coverage
```

Success gate:

```text
frame_hit_rate >= 0.95
median unique_projected_pixels >= 2000
focus_200_300_projected_coverage > 0
```

The projected pixel coverage is allowed to be sparse. It should not be compared against dense 2D semantic coverage.

---

## 6. Codex Phase 2：VideoMasklet vs projected 3D semantic alignment

### 6.1 Goal

Use sparse projected 3D semantic as **anchor evidence** for VideoMasklet trust calibration.

It answers:

```text
1. Is this masklet label likely correct?
2. Is this masklet temporally consistent?
3. Does this masklet cover LiDAR-visible semantic anchors?
4. Should Semantic Prior Generator trust its semantic label or fallback to geometry / D_g / TTT conflict?
```

### 6.2 Per-masklet alignment metrics

For each video masklet $j$:

```text
masklet_id
fine_label_pred
coarse_label_pred
num_frames_visible
area_mean
mask_temporal_iou_mean
mask_fragmentation
label_flip_count
projected_point_support_count
projected_pixel_support_count
projected_majority_semantic_id
projected_majority_semantic_name
projected_majority_ratio
projected_entropy
agreement_pred_vs_projected
support_depth_mean
support_depth_p90
```

Define:

$$
Q_{3d}(j)=\min\left(1, \frac{N_{proj}(j)}{N_{min}}\right) \cdot (1-H_{sem}(j))
$$

where:

```text
N_proj(j) = number of projected valid pixels inside masklet j
H_sem(j) = normalized entropy of projected semantic labels inside masklet j
N_min = 50 initially
```

Define label agreement:

$$
A_{label}(j)=\mathbf{1}[L_{pred}(j)=L_{proj-majority}(j)]
$$

For incompatible label taxonomies, map both to shared groups:

```text
road / sidewalk / parking / ground -> GROUND_STRUCTURE
building / wall / fence -> STRUCTURE_ANCHOR
vegetation / trunk / terrain -> VEGETATION_STUFF
sky -> SKY_STUFF if present in video masklet only; projected LiDAR usually lacks sky
car / truck / bicycle / person -> MOVABLE_THING
```

Important: SemanticKITTI LiDAR cannot validate sky because sky is not hit by LiDAR. A masklet labeled sky with low projected support is not automatically wrong.

### 6.3 Trust score

The trust score should combine VideoMasklet quality and sparse 3D anchor:

$$
T_{mask}(j)=\operatorname{clip}\left(
0.35Q_{mask}(j)+0.25Q_{temporal}(j)+0.25Q_{3d}(j)+0.15A_{label}(j),0,1
\right)
$$

But if projected support is low:

$$
N_{proj}(j)<N_{min}\Rightarrow Q_{3d}(j)=\text{unknown},\; A_{label}(j)=\text{unknown}
$$

Then do not penalize the masklet solely because LiDAR did not hit it.

### 6.4 Outputs

```text
masklet_3d_alignment/masklet_alignment.csv
masklet_3d_alignment/masklet_alignment.jsonl
masklet_3d_alignment/per_chunk_alignment_summary.csv
masklet_3d_alignment/low_trust_masklet_gallery/
masklet_3d_alignment/high_disagreement_gallery/
```

Gate to continue semantic role experiments:

```text
at least 30% of non-sky / non-far-stuff masklets have projected support >= N_min
and at least 70% of supported structure / ground masklets agree with projected semantic group
```

If this fails:

```text
Do not run semantic candidate matrix.
First improve VideoMasklet trust / taxonomy mapping / projection alignment.
```

---

## 7. Codex Phase 3：Quality-aware Semantic Prior Generator

### 7.1 Modify Semantic Prior Generator input

Add:

```text
T_mask(j): masklet trust score
A_label(j): projected 3D agreement flag or unknown
Q_3d(j): sparse 3D support quality or unknown
L_proj_group(j): projected majority group or unknown
```

Semantic Prior Generator should not directly replace predicted label with projected label unless support is strong.

Recommended rule:

```text
if Q_3d known and A_label == 1:
    trust semantic branch more
elif Q_3d known and A_label == 0:
    downgrade semantic branch, fallback to geometry / D_g / conflict
else:
    use VideoMasklet trust only; do not punish sky/far stuff due to no LiDAR support
```

### 7.2 Role generation

Do not use semantic alone. Role should be:

```text
role = f(
    fine_label_pred,
    coarse_label_pred,
    T_mask,
    D_g,
    C_ttt_conflict,
    S_scale_risk,
    memory_path
)
```

Examples:

```text
road/building/wall/fence + high T_mask + lowD + low conflict:
    TTT positive long
    frame/global source keep
    SWA overlap anchor keep

vegetation + highD + high conflict + high T_mask:
    frame/global source skip
    TTT short negative or no-long-write

sky + lowD:
    neutral context, not TTT positive, not hard negative

sky + highD + high scale-risk:
    frame/global weak source skip, TTT neutral/no-long-write

masklet low trust:
    fallback to geometry / D_g / update_conflict_energy, not semantic rule
```

### 7.3 Role output streams

Generate:

```text
R_frame_tok
R_global_tok
R_swa_tok
R_ttt_tok
T_mask_tok
Q_3d_tok
A_label_tok
```

Memory path semantics:

```text
frame/global:
    source keep / soft skip / hard skip

SWA:
    cache keep / overlap anchor protect / non-overlap skip / value attenuation

TTT:
    positive long / neutral keep / short negative / no-long-write
```

---

## 8. Codex Phase 4：Action distinguishability before rollout

Before running h10/h15 candidate rollouts, Codex must verify different role policies produce different actions.

For each policy:

```text
frame_source_keep_mask
global_source_keep_mask
swa_cache_keep_mask
ttt_positive_mask
ttt_neutral_mask
ttt_negative_mask
ttt_no_write_mask
```

Compare:

```text
SEM_ONLY vs QUALITY_AWARE_SEM
QUALITY_AWARE_SEM vs RISK_ONLY
SEM_RISK vs QUALITY_AWARE_SEM_RISK
```

Metrics:

```text
Jaccard(action_A, action_B)
source_keep_ratio_difference
TTT_role_mass_difference
attention_mass_removed_difference
SWA_overlap_source_mass_removed_difference
post_zp_update_norm_difference
```

Gate:

```text
Jaccard <= 0.85
or source_keep_ratio_difference >= 0.05
or TTT_role_mass_difference >= 0.05
```

If not:

```text
Do not rollout.
Codex should debug projection from masklet role to token action.
```

---

## 9. Codex Phase 5：Masklet-level causal bank with 3D trust

### 9.1 Goal

Before writing large semantic rules, test the causal effect of individual masklets.

For selected chunks:

```text
chunks = 6,10,16
horizons = h5,h10,h15
```

Select top masklets by:

```text
projected support count
T_mask
D_g p90
C_ttt_conflict p90
S_scale_risk p90
source attention mass
semantic diversity
```

### 9.2 Interventions

For each masklet $j$ and path $p$:

```text
frame:
    keep source
    soft skip source
    hard skip source

global:
    keep source
    soft skip source
    hard skip source

SWA:
    keep in cache
    remove from previous-source cache
    protect as overlap anchor
    attenuate value only

TTT:
    positive long
    neutral keep
    short negative
    no long write
```

Causal effect:

$$
E_{j,p,a}^{h}=ATE_{base}^{h}-ATE_{intervention(j,p,a)}^{h}
$$

Disease-window effect:

$$
E_{j,p,a}^{[200,300]}=Err_{base}^{[200,300]}-Err_{intervention(j,p,a)}^{[200,300]}
$$

### 9.3 Outputs

```text
masklet_causal_bank/causal_effects.csv
masklet_causal_bank/causal_effects_by_label.csv
masklet_causal_bank/causal_effects_by_path.csv
masklet_causal_bank/top_positive_masklets.csv
masklet_causal_bank/top_negative_masklets.csv
masklet_causal_bank/action_gallery/
```

Oracle gate:

```text
h10 ATE delta <= -3m
or h15 ATE delta <= -3m
or [200,300) delta <= -5m
```

If oracle fails:

```text
Semantic masklet route is not the main Target-25 action.
Keep semantic as diagnostic / weak regularizer.
Do not launch pairwise/all-memory semantic matrix.
```

---

## 10. Codex Phase 6：Learn simple quality-aware role router

Only if Phase 5 oracle passes.

### 10.1 Features

```text
fine label
coarse group
masklet area
mask temporal IoU
mask fragmentation
T_mask
Q_3d
A_label
D_g mean/p90
C_ttt_conflict mean/p90
S_scale_risk mean/p90
frame attention source mass
global attention source mass
SWA source mass
TTT update norm
overlap membership
boundary proximity
```

### 10.2 Model

Use interpretable models first:

```text
decision tree
rule list
logistic regression
small random forest
```

Target:

```text
best action per masklet per memory path
```

### 10.3 Generalization gate

Do not train and test on the same chunk only.

```text
train chunk10 -> test chunk6/chunk16
train chunk6 -> test chunk10/chunk16
```

Pass condition:

```text
test h10 ATE delta <= -1.5m
or test [200,300) delta <= -3m
and [400,600) regression <= +1m
```

If fail:

```text
Do not full-run.
Diagnose overfitting / add non-semantic features / reduce label-specificity.
```

---

## 11. Codex Phase 7：Durability and washout attribution

If h10 strong but h15 weak:

Measure:

```text
HMC state movement
TTT state movement
SWA cache movement
global source summary movement
merge/gauge movement
role mass over time
masklet trust over time
```

Questions:

```text
1. Does TTT tail update overwrite correction?
2. Does SWA cache refresh remove useful masklet role?
3. Does global source update erase source filtering?
4. Does merge/gauge state dominate the trajectory change?
```

Decision:

```text
TTT washout:
    try positive long commit only for validated masklets

SWA washout:
    try persistent overlap-anchor cache for validated masklets

Global source washout:
    try source-role persistence across chunks

Merge/gauge washout:
    semantic memory path is not enough; switch mainline to trajectory-state module
```

---

## 12. Codex Phase 8：Full online validation

Only if all gates pass:

```text
1. projection data gate pass
2. masklet trust gate pass
3. action distinguishability gate pass
4. masklet causal oracle gate pass
5. router generalization gate pass
6. durability gate pass
```

Then run full online:

```text
K1_H9 reference
C9 reference
quality-aware semantic router candidate
risk-only counterpart
semantic-only counterpart
```

Metrics:

```text
ATE
Rot
RPE_t
RPE_r
FinalErr
[200,300]
[200,400]
[400,600]
50f/100f/200f mean ATE
YawRMSE
Sim3Scale
boundary_10f_ATE
boundary_20f_ATE
```

Success:

```text
Strong:
    ATE <= 25m

Stage:
    ATE <= 30m
    and [200,300] <= 50m
    and [400,600] regression <= +1m

Weak:
    ATE improves over C9 by >= 1.5m
    and h15 durability was predicted correctly
```

---

## 13. Codex failure routing

### Case A：download blocked

```text
write MANUAL_DOWNLOAD_REQUEST.md
stop all projection-related experiments
continue VideoMasklet-only diagnostics if requested
```

### Case B：download exists but sequence 01 incomplete

```text
write SEMKITTI_SEQUENCE01_INCOMPLETE.md
list missing frames
stop projection cache generation
```

### Case C：calib missing Tr

```text
write CALIB_BLOCKER.md
use downloaded odometry calib if available
never invent identity Tr
```

### Case D：projection sparse support too low

```text
only use projection as diagnostic;
do not penalize masklets with no LiDAR support;
continue VideoMasklet trust from temporal / D_g / conflict metrics
```

### Case E：masklet-projection disagreement high

```text
do not run semantic memory candidates;
first repair taxonomy mapping or trust routing;
output high_disagreement_gallery
```

### Case F：action masks equivalent

```text
do not rollout;
check role projection collapse;
check protected token fallback;
check source skip hooks;
```

### Case G：masklet causal oracle has no upper bound

```text
semantic route demoted to diagnostic;
resume TTT-native / scale-state / context source non-semantic mainline
```

### Case H：oracle has local signal but h15 fails

```text
run durability attribution;
route to TTT/SWA/global/merge washout repair;
do not tune semantic threshold
```

---

## 14. Final decision rule

v29C should answer:

```text
1. Can Codex obtain valid SemanticKITTI sequence 01 projection data?
2. Does sparse 3D projection improve VideoMasklet trust calibration?
3. Do trusted semantic masklets have masklet-level causal upper bound?
4. Can a simple quality-aware role router generalize across chunks?
5. Does it produce durable h15 improvement?
```

If answers 2-5 are mostly no:

```text
Semantic Prior Generator remains useful for diagnosis and weak regularization,
but it is not the Target-25 mainline.
```

If answers 2-5 are yes:

```text
Proceed to full online validation with quality-aware semantic role router.
```

