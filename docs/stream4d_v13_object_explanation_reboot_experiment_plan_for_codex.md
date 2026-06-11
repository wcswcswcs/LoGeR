# Stream4D v13：D4RT-native Object Explanation Reboot 实验计划书

面向 Codex 的可执行实验计划。本文档以 v12 probe5 结果和代码审计为基础，目标是停止继续堆 top-k、WTA、NMS、score sweep、mask-level memory 这类后处理，转向真正的 **feed-forward semantic 4D reconstruction and tracking**：从 D4RT material correspondences 与 noisy 2D semantic measurements 中，training-free 地推断 latent semantic 4D object field。

本文所有公式均使用 Typora 友好的 `$...$` 或 `$$...$$`，不使用方括号 display 公式。

---

## 0. 本轮结论先行

v12 的主要价值不是性能，而是把失败定位得更清楚：

1. **D4RT correspondence 是有用的。** 连续帧 D4RT surfel 的 `uv_in01_rate_mean=0.9858`、`self_uv_error_p90≈1.57px`、`cycle_uv_error_p90≈3.27px`，说明 D4RT image-space material correspondence 不是主要噪声源。
2. **真实 D4RT temporal evidence 有贡献。** M5 明显强于 shuffled D4RT M6 和 no-temporal M7，说明 D4RT 信息进入了 posterior。
3. **negative evidence / WTA 有贡献。** M5 相比 M4 降低 conflict，并明显提升 AP/AP50/AP25。
4. **但当前方法仍然失败。** M5/M5r3 own support 只有约 4.3%，S0/S1 cross-support 基本崩溃；同一 method support 上 Stream3D 仍强于 M5。
5. **current primitive 上界分裂。** Tiny C_surfel oracle 很高，但 broad C_hybrid oracle 不强；说明当前 primitive 能解释局部 clean object，不能解释完整 scene。
6. **当前 object explanation 仍不是真正的 object explanation。** 代码实际上仍以 source mask group 为 birth，posterior 只在 birth group 内做 surfel 分类，最后导出仍用 mask_backproject；它还没有做全局 object set packing，也没有把 low-confidence measurements 作为待解释证据统一管理。

因此 v13 的核心不是“调 M5”，而是把算法改成：

```text
D4RT surfels + semantic masklets/regionlets measurements
-> global object explanation / set packing
-> posterior surfel ownership field
-> conservative geometry export and 4D tracking query
```

---

## 1. 当前结果的独立判断

### 1.1 v12 没有接近 Stream3D

v12 Phase 0 的 probe5 基准显示：

```text
Stream3D on S0:
AP / AP50 / AP25 = 0.235730 / 0.414306 / 0.537786

Stream3D on S1:
AP / AP50 / AP25 = 0.399213 / 0.597171 / 0.742535
```

而 v12 best method-level 结果仍是 tiny-support：

```text
M5 negative own:
0.226671 / 0.453586 / 0.745367
pre% = 4.2958

M5 on S0:
0.002042 / 0.005558 / 0.010704

M5 on S1:
0.023149 / 0.049873 / 0.148286

M5r3 strict posterior own:
0.237230 / 0.455495 / 0.729781
pre% = 4.2756

M5r3 on S0:
0.002042 / 0.005558 / 0.012047

M5r3 on S1:
0.022346 / 0.046761 / 0.156118
```

结论：M5/M5r3 own AP 接近或略高于 Stream3D S0 AP，但这只是 tiny support 上的 observed-subset 结果；一旦放到 S0/S1 统一 support，几乎无效。不能写成 static ScanNet object reconstruction 成功。

### 1.2 v12 有真实正信号，但不是足够强的方法

M5 相比 M4、M6、M7：

```text
M5 - M4:
AP +0.085757, AP50 +0.191039, AP25 +0.285771

M5 - M6 shuffled D4RT:
AP +0.220564, AP50 +0.420911, AP25 +0.637562

M5 - M7 no temporal:
AP +0.088056, AP50 roughly +0.195
```

这说明 signed evidence、negative evidence 和真实 D4RT temporal evidence 都有用。问题不是 D4RT 没用，而是当前 inference 仍然无法把 useful evidence 变成 complete object field。

### 1.3 代码层面的核心缺陷

v12 代码审计显示：

1. `py_compile` 通过，但 `unittest discover tests` 在当前审计容器由于缺少 `open3d` 失败。后续测试必须拆成 `pure_python`、`open3d_required`、`gpu_required`，不能让协议测试被 optional dependency 卡住。
2. `tools/export_v12_object_explanation.py` 调用 `explain_objects()` 后，最终仍通过 `ScanNetExporter.export_object_dict_mask_backproject()` 导出 object support。也就是说，support 仍来自 selected 2D masks 的 RGB-D backprojection，不是 posterior surfel field 的直接 support。
3. `stream4d/object_explanation.py` 的 slot birth 来自 `birth_groups()`，而 `birth_groups()` 默认按 `(src_frame, src_mask_id)` 分组。也就是说，object candidate 的出生仍是 source 2D mask，不是多 measurement explanation。
4. `posterior_for_group()` 只对该 birth group 内的 surfels 做 posterior，无法把同一 object 在其他 masks/frames 中的 surfels 吸收进来，也无法把一个 bad birth mask 拆成多个 competing object slots。
5. `measurement_votes()` 只是从 core surfels 投票出 mask observations，随后 exporter 又 backproject 整个 selected masks。这导致 inference 的 posterior 没有真正控制 support 边界。
6. `negative_observation = visible & mask_frame_available & source_positive & (target_mask_id <= 0)` 只把 visible-but-outside-any-mask 当负证据。它没有建模“visible inside another object's mask but not this slot”的强反证，也没有 same-frame object competition 的完整项。

这说明 v12 只是 minimal closure，不是完整 object explanation framework。

---

## 2. v13 总体目标

v13 的目标是验证一个新的、顶会级别的 formulation：

> **Training-free object explanation over D4RT material fields.**

基本定义：D4RT 提供 material surfel field，2D masks / propagated masklets / regionlets / features 是 noisy measurements，object 是 latent slot，用来解释 measurements 并拥有 surfels。

D4RT surfel：

$$
s_i = \{X_i(t), \pi_i(t), v_i(t), c_i(t), f_i(t)\}_{t=1}^{T}
$$

semantic measurement：

$$
z_m = (t, R_m, e_m, b_m, q_m)
$$

latent object assignment：

$$
y_i \in \{1, \dots, K, \text{unknown}\}
$$

object field：

$$
O_k = \{s_i \mid y_i = k\}
$$

object visible support at time $t$：

$$
O_k(t) = \{(X_i(t), \pi_i(t)) \mid s_i \in O_k, v_i(t)=1\}
$$

v13 要证明的不是“某个后处理 AP 上升”，而是以下三个科学假设。

### H1：当前失败是 inference/ownership 问题，不是 D4RT correspondence 无效

如果真实 D4RT evidence 相比 shuffle/no-temporal 能持续提升 object ownership，并且在 oracle 上界中候选池有高 IoU object，那么问题在 inference。

### H2：full mask / regionlet / carrier component 都不能直接当 object；object 应该是解释 measurements 的 latent entity

如果 mask-as-object、regionlet-as-object、carrier-component-as-object 都在 cross-support 上失败，而 object-explanation set packing 能同时提高 own 与 S0/S1，则新 formulation 成立。

### H3：semantic measurement density 是当前完整 reconstruction 的硬瓶颈

如果只使用 2/16 mask frames，任何 inference 都只能产生 tiny clean subset；需要 D4RT-propagated video masklets 或更密集 2D semantic measurements，才能从 clean object birth 走向 broad support reconstruction。

---

## 3. 统一评估硬规则

所有 method config 必须输出以下四行，缺一不可：

```text
M own:
  prediction = M
  pre_points = M

Stream3D on M:
  prediction = Stream3D baseline P0
  pre_points = M

M on S0:
  prediction = M
  pre_points = Stream3D original S0

M on S1:
  prediction = M
  pre_points = historical 32f support S1
```

如果方法来自 parent 后处理，还要输出：

```text
M inherit parent:
  prediction = M
  pre_points = parent
```

必须记录：

```text
AP
AP50
AP25
pre_points %
prediction union %
union in target support %
GT crop/full
#pred
mean points/object
median points/object
conflict rate
same-frame conflict count
measurement explained ratio
measurement multi-explained ratio
unexplained measurement ratio
assigned surfel ratio
core/fringe/unknown/reject surfel ratio
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per GT
support source
geometry source
uses_gt_for_prediction
uses_gt_for_diagnostic
is_method_result
```

判断标准：

```text
必须同时看 own 和 cross-support。
如果 own 高但 M on S0/S1 近零，只能写 high-precision observed subset。
如果 Stream3D on M 明显高于 M own，说明 method support 本身不难，是 object assignment 失败。
如果 broad-support oracle 不强，说明 primitive 不能支撑完整 reconstruction。
```

---

## 4. Phase A：代码审计与 artifact 修复

### A.1 目标

让下一轮实验可复跑、可审计，不再出现“测试因为 open3d 缺失整体失败”或“代码包缺核心模块”的问题。

### A.2 Codex 必须做

1. 把 tests 拆成：

```text
tests/test_protocol_pure_python.py
tests/test_manifest_and_eval_policy.py
tests/test_object_explanation_pure.py
tests/test_export_scannet_open3d.py
tests/test_d4rt_gpu_optional.py
```

2. 增加不依赖 `open3d` 的 pure python tests：

```text
measurement bank load/save roundtrip
birth_groups split/merge edge cases
posterior_for_group negative evidence behavior
measurement_votes deterministic behavior
manifest scanner rejects uses_gt method result
metric integrity refuses diagnostic-only as method
```

3. 将 `open3d` import 从 module top-level 移入需要它的函数，或在测试里通过 skip 处理。

4. 每轮提交完整 code audit packet：

```text
stream4d_v13_<phase>_code_review_packet.zip
stream4d_v13_<phase>_code_review_packet.sha256
stream4d_v13_<phase>_filelist.txt
stream4d_v13_<phase>_ziptest.log
stream4d_v13_<phase>_git_diff.patch
stream4d_v13_<phase>_git_status.txt
```

zip 必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/*.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_v13_*.sh
Stream3D/docs/stream4d_v13_执行日志.md
Stream3D/docs/stream4d_v13_实验结果复盘.md
Stream3D/outputs/audit/**/*.json
Stream3D/outputs/audit/**/*.md
Stream3D/data/evaluation/scannet/*_class_agnostic.txt
new method prediction/TMP artifacts for probe5
```

### A.3 记录指标

```text
py_compile_status
pure_python_tests_status
open3d_tests_status
gpu_tests_status
reportable_scan_summary
metric_integrity_phase0_pass
num_missing_manifest
num_missing_eval_policy
num_uses_gt_for_prediction
num_uses_gt_for_diagnostic_and_method_result
object_dict_alignment_mean/min IoU
```

### A.4 成功标准

```text
pure python tests 必须通过。
open3d/gpu tests 可以 skipped，但必须清楚标注 skipped reason。
reportable configs suspicious=0。
metric integrity phase0_pass=True。
```

### A.5 不满足条件时的尝试方向

如果 `open3d` 仍阻塞 import，Codex 必须改 lazy import；如果 tests 仍依赖真实 ScanNet 文件，Codex 必须构造 tiny synthetic masks/point ids fixture，不能要求人工先配数据。

---

## 5. Phase B：失败归因矩阵，不再只看 AP

### B.1 目标

回答：现在是 candidate primitive 本身错，还是 object inference 选不出来？这个问题必须先回答，否则继续改算法会盲目。

### B.2 实验设计

构建同一批 measurement/candidate bank，包含四类 primitive：

```text
C_mask: 原始 2D mask backprojection candidate
C_regionlet: mask 内按 depth/normal/boundary/D4RT seed 切的 local fragment
C_surfel_cluster: D4RT surfel cluster candidate
C_masklet: D4RT-propagated video masklet candidate，见 Phase C
C_hybrid: 以上候选 union
```

对每类 candidate，运行两个版本：

```text
method-free oracle diagnostic：GT oracle selects best subset，只做 upper bound
unsupervised simple baseline：score by size/stability/observation count，不读 GT
```

Oracle 只能 diagnostic，manifest 必须：

```text
uses_gt_for_prediction=true
uses_gt_for_diagnostic=true
is_diagnostic_only=true
is_method_result=false
```

### B.3 记录指标

```text
oracle AP/AP50/AP25
unsupervised AP/AP50/AP25
candidate count
candidate union %
pre_points %
conflict rate
per-GT best IoU mean
GT with best IoU >= 0.25 / 0.50 / 0.75
no-candidate GT count
candidate-filtered GT count
wrong-assignment GT count
boundary-bad GT count
mean candidates per GT
mean GT per candidate
candidate duplicate ratio
```

### B.4 判断标准

如果：

```text
C_hybrid oracle on broad support AP < Stream3D S0 AP
```

则当前 primitive 本身不足，必须转向 Phase C 的 masklet/measurement generation，不能继续调 set packing。

如果：

```text
oracle high but unsupervised low
```

则 candidate 有潜力，问题在 inference/selection，进入 Phase D。

如果：

```text
C_surfel tiny oracle high but C_hybrid broad oracle weak
```

说明局部 clean object 有上界，但完整 reconstruction primitive 仍不足。此时必须增加 semantic measurement density，不能只调 posterior threshold。

### B.5 不满足条件时的尝试方向

如果所有 oracle 都低，Codex 不要继续实现复杂 solver；先扩充 measurement 类型：video masklets、DINO/CLIP local features、depth discontinuity regionlets、SAM2/DEVA/2D tracker outputs。如果 oracle 高但 method 低，Codex 进入 Phase D，优先实现 MDL set packing，而不是 score sweep。

---

## 6. Phase C：Semantic Measurement Density 方向，解决 2/16 mask frames 过稀疏问题

### C.1 目标

v12 measurement bank 中 `num_mask_frames_available_mean=2/16`，这意味着 semantic observations 太稀疏。D4RT track 很好，但没有足够 semantic measurements，就无法从 birth object 走向 full object field。本阶段要构建 **D4RT-propagated video masklet measurements**。

### C.2 新方向

不再只用已有 CropFormer mask frames。对每个 high-confidence birth mask $m_{t_0}$，用 D4RT 将其内部 surfels track 到其他 frames，生成 propagated masklet：

$$
\hat{R}_{k,t} = \{\pi_i(t) \mid s_i \in R_{m,t_0}, v_i(t)=1\}
$$

然后用 image-space consistency 校验：

```text
cycle consistency
local RGB appearance consistency
2D boundary compactness
mask overlap with existing CropFormer/SAM masks when available
visible-outside negative evidence
```

输出不是 object，而是 measurement：

```text
masklet measurement = (slot proposal id, frame t, propagated region, confidence, negative support)
```

### C.3 实验设置

实现：

```text
tools/build_v13_video_masklet_measurements.py
stream4d/video_masklet.py
```

输入：

```text
v12 measurement_bank.npz
D4RT carrier tracks
source masks
RGB frames
optional existing CropFormer masks
```

输出：

```text
outputs/v13_masklet_measurements/<scene>/masklets.npz
outputs/audit/v13_masklet_density/masklet_density_probe5.{json,md,csv}
```

比较四种 semantic measurement sources：

```text
C0 original sparse CropFormer only
C1 source-mask propagated by D4RT only
C2 propagated + existing mask consistency gate
C3 propagated + appearance + boundary + cycle gates
```

### C.4 记录指标

```text
num_mask_frames_available
num_effective_semantic_frames_per_surfel
positive observations per surfel
unobserved surfel ratio
masklet count per scene
masklet frames per object birth
masklet compactness
masklet area growth ratio
cycle error p50/p90 for masklet surfels
RGB appearance drift
available-mask agreement IoU when mask exists
negative visible-outside ratio
ambiguous surfel ratio
```

Diagnostic with GT only：

```text
masklet-to-GT best IoU
masklet purity
masklet object coverage
masklet over-merge rate
masklet under-segmentation rate
```

GT diagnostic 必须单独标记，不进入 method table。

### C.5 判断标准

C3 必须满足：

```text
effective semantic frames per surfel >= 5.0
unobserved surfel ratio <= v12 by at least 30%
masklet purity diagnostic not worse than original mask by >5%
masklet broad oracle AP improves over C_hybrid oracle by >= 0.05 AP or >= 0.07 AP50
```

如果 C3 做不到，说明 D4RT propagation 无法可靠增加 semantic measurements，下一步应换 2D video segmentation source，而不是继续扩 masks。

### C.6 不满足条件时的尝试方向

若 propagated masklet 过脏：

```text
提高 cycle/appearance gate；只保留 core masklet，不输出 full propagated region；引入 unknown，不强制扩 support。
```

若 propagated masklet 太稀疏：

```text
使用更密 grid surfels；允许 multi-birth masklets；引入 2D video tracker 或 SAM2/DEVA 作为 semantic measurement source。
```

若 masklet across time 漂移：

```text
仅用 D4RT propagation 产生 candidate support，再用 existing masks/appearance 修正；不要把 propagation 当 hard label。
```

---

## 7. Phase D：Object Explanation as MDL Set Packing，而不是 connected component

### D.1 目标

把 object 从 observation primitive 中分离出来。Object 不再等于 mask/regionlet/surfel cluster，而是一个 latent explanation slot，用来解释一组 measurements。

### D.2 新算法

实现：

```text
stream4d/object_explanation_mdl.py
tools/export_v13_object_explanation_mdl.py
```

定义候选 object slot $O_k$，它解释 measurements $Z_k$ 和 surfels $S_k$。能量：

$$
E(O) = C_{model}(O) + C_{miss}(O) + C_{conflict}(O) + C_{boundary}(O) + C_{motion}(O) + C_{appearance}(O)
$$

全局目标：

$$
\min_{\mathcal{O}} \sum_{O_k \in \mathcal{O}} E(O_k) + \lambda_{overlap} \sum_{i} \max(0, n_i - 1) + \lambda_{unexplained} |Z_{unexplained}|
$$

其中 $n_i$ 是 surfel 被多少 object 解释。允许 unknown：

```text
unknown surfels 不惩罚过重；ambiguous measurements 可以 pending，不直接并入 object。
```

核心要求：

```text
weak positive 不可传递式合并。
strong negative 优先于 positive co-membership。
一个 2D mask 可以被多个 object slots split-explain。
一个 object 可以解释多个 masks/masklets。
```

### D.3 Candidate generation

从 Phase C measurement bank 产生 object proposals：

```text
birth proposals: high-confidence single clean mask / masklet core
split proposals: one mask contains multiple D4RT/geometry components
merge proposals: two slots share stable masklets and no strong negative
fringe proposals: uncertain region attached to nearest compatible slot
reject proposals: measurement cannot be explained without conflict
```

每个 proposal 必须记录：

```text
core_surfels
fringe_surfels
unknown_surfels
reject_surfels
explained_measurements
negative_measurements
appearance_summary
motion_summary
source masks/masklets
```

### D.4 Inference solver

第一版使用 deterministic greedy MDL set packing：

1. 按 score 排序 candidate slots。
2. 逐个加入，如果违反 hard cannot-link 或 overlap penalty 太高则拒绝。
3. 对 overlapping slots 尝试 split、trim fringe、或 keep both with unknown overlap。
4. 最后做 local search：merge/split/swap，要求能量下降才接受。

不要实现复杂训练；全部 training-free。

### D.5 记录指标

```text
num_candidate_slots
num_selected_slots
selected/unselected ratio
explanation energy components
explained measurement ratio
unexplained measurement ratio
multi-explained measurement ratio
surfel overlap conflict
core/fringe/unknown/reject ratio
slot split count
slot merge count
slot swap accepted count
measurement split count
mean measurements per selected slot
mean surfels per selected slot
AP/AP50/AP25 own/S0/S1/Stream3D-on-M
```

### D.6 成功标准

Probe5：

```text
M own AP >= 0.30
M own AP50 >= 0.55
M own AP25 >= 0.75
M on S0 AP >= 0.08
M on S0 AP50 >= 0.18
M on S0 AP25 >= 0.45
M on S1 AP >= 0.18
M on S1 AP50 >= 0.35
M on S1 AP25 >= 0.60
Stream3D on M - M own AP <= 0.08
```

如果 own 提升但 S0/S1 不提升，说明仍是 tiny-support。若 S0/S1 提升但 own 低，说明 broad support 有 recall 但 object precision 不够，需要 tighter core/fringe posterior。

### D.7 不满足条件时的尝试方向

如果 conflict 高：提高 overlap penalty 或 split measurements。

如果 AP25 高但 AP50 低：边界/fringe 太粗，强化 boundary and appearance term。

如果 AP50 高但 AP低：ranking/duplicate objects 多，强化 MDL model cost 和 one-to-one competition。

如果 Stream3D-on-M 仍远高于 M own：method support 是可用的，但 object ownership 错。不要扩 support，改 inference。

---

## 8. Phase E：Posterior Support Export，不再 full-mask backproject

### E.1 目标

v12 最大代码问题之一是 posterior 推断后仍用 full selected mask backproject。v13 必须让 posterior 真正控制 exported support。

### E.2 新导出方式

实现：

```text
ScanNetExporter.export_object_slot_posterior_support()
```

输入 object slot：

```text
core_surfels
fringe_surfels
unknown_surfels
reject_surfels
owned measurements
```

导出规则：

```text
core support: high-posterior surfel seeds backproject or RGB-D local component around seeds
fringe support: only region connected to core and passing boundary/appearance gate
unknown: never exported by default
reject: never exported
```

禁止默认导出整张 owned mask。整张 mask 只允许作为 upper bound 或 ablation。

### E.3 记录指标

```text
core exported points
fringe candidate points
fringe kept points
unknown points not exported
reject points not exported
core-to-fringe ratio
export conflict rate
export hit rate
boundary erosion ratio
connected component count per object
points/object distribution
```

### E.4 判断标准

相比 full-mask export：

```text
conflict rate 降低 >= 30%
AP50 不下降
S0/S1 AP 不下降
per-GT best IoU mean 不下降
```

如果 AP25 降而 AP50 升，说明 export 更干净但 recall 变低；可允许作为 core-only ablation，但主方法需要 calibrated fringe。

### E.5 不满足条件时的尝试方向

若 export 太稀疏：放宽 fringe gate，但只在 connected-to-core 区域扩。

若 export 太脏：禁止 multi-mask union，改成 per-frame best component voting。

若 Scene mesh NN 误差大：输出 image-space overlay 先确认不是 geometry adapter 错。

---

## 9. Phase F：D4RT geometry aligned -> Stream3D diagnostic

### F.1 目标

单独回答：D4RT geometry 精度对 Stream3D 3D segmentation pipeline 的影响有多大。这个实验是 diagnostic，不是主方法。

### F.2 设计

比较：

```text
G0: original Stream3D RGB-D/pose geometry
G1: D4RT raw geometry, scale-normalized thresholds
G2: D4RT scene-level Sim3 aligned to RGB-D/GT geometry
G3: D4RT window-level Sim3 aligned
G4: D4RT Sim3 aligned + density-normalized manifold thresholds
G5: D4RT image-space evidence + RGB-D geometry bridge
```

关键要求：

```text
不能复用 RGB-D meter-scale radius/min_points。
NN radius 使用 D4RT point spacing quantiles。
manifold threshold 使用 aligned D4RT local density。
Sim3 只在 evaluation/export diagnostic 中使用，不进入 method inference。
```

### F.3 记录指标

```text
Sim3 residual median/p90/p95
aligned point spacing q10/q50/q90
D4RT-to-RGBD reprojection error
Stream3D AP/AP50/AP25 under each geometry
num exported objects
pre% / union%
mesh NN hit rate
per-object mask IoU vs RGB-D Stream3D projection
manifold refine keep/drop ratio
failure reason: no objects / fragmented / geometry mismatch / threshold mismatch
```

### F.4 判断标准

```text
If G2/G4 AP drop vs G0 <= 3 AP and AP50 drop <= 5:
  D4RT geometry is sufficiently accurate for Stream3D-style projection diagnostic.

If drop is large but image-space D4RT evidence is good:
  D4RT should be used as correspondence/identity, not metric geometry replacement on ScanNet.

If raw fails but Sim3-aligned works:
  scale/reference alignment was the main issue.

If Sim3-aligned also fails:
  current D4RT metric geometry adapter is not adequate for static ScanNet AP.
```

### F.5 不满足条件时的尝试方向

若 G1/G2 0 objects：自动 sweep density-normalized `min_points` and NN radius, but record as diagnostic grid, not method tuning.

若 Sim3 residual high：check frame indexing, intrinsics resize, D4RT query normalization, reference camera `tcam`, and stride.

若 geometry AP nonzero but low：visualize per-scene D4RT/RGB-D projected masks to separate geometry error from object inference error.

---

## 10. Phase G：Tune30 / Final gate

### G.1 目标

防止 probe5 过拟合。只有 Phase D/E 在 probe5 过 gate 才能进入 tune30。

### G.2 Tune30

固定 probe5 top two configs，允许只在 tune30 上选择：

```text
one MDL weight setting
one fringe export setting
one masklet density setting
```

Tune30 必须报告 own/S0/S1/Stream3D-on-M 四行。

Tune30 gate：

```text
own AP >= 0.28
own AP50 >= 0.52
M on S1 AP >= 0.16
M on S1 AP50 >= 0.32
M on S0 AP >= 0.06
Stream3D-on-M - M-own AP <= 0.10
```

### G.3 Final

Final split 只允许跑一次 locked config。不能在 final 上修改阈值。

必须报告：

```text
Stream3D original
Stream3D on method support
method own
method on Stream3D support
method on S1 if available
```

如果 final 失败，复盘必须写失败，不允许回头调 final。

---

## 11. Phase H：Dynamic Replica 只在静态 object explanation 过关后推进

### H.1 数据 gate

先运行：

```text
tools/check_dynamic_replica_env.py
```

必须确认：

```text
images exist
camera trajectories exist
instance/object IDs exist
semantic labels exist if semantic AP required
depth exists if RGB-D export required
trajectory GT exists if 4D tracking metric required
```

如果没有 object IDs：

```text
不能报告 IDF1、MOTA、4D IoU、official instance tracking。
只能报告 qualitative 或 diagnostic consistency。
```

### H.2 动态指标

如果数据可用，比较：

```text
D0 framewise 2D mask baseline
D1 static overlap / Stream3D-like baseline if depth available
D2 v13 object explanation field
D3 D4RT masklet-only diagnostic
```

指标：

```text
IDF1
IDSW
fragmentation
reactivation after occlusion
4D IoU
track purity
object lifetime accuracy
semantic consistency over time
trajectory EPE for surfel/object tracks
```

成功标准：

```text
v13 must reduce IDSW and fragmentation vs static-overlap baseline.
v13 must improve occlusion reactivation.
If ScanNet improves but dynamic tracking does not, feed-forward semantic 4D tracking claim is not supported.
```

---

## 12. 每轮必须生成的可视化

每个 method 至少保存：

```text
1. own/cross support AP matrix heatmap
2. support ratio and GT crop/full bar chart
3. method vs Stream3D same-support delta chart
4. measurement bank overlay: surfels colored by source mask / target mask
5. masklet propagation video strip
6. object slot core/fringe/unknown/reject overlay
7. final prediction vs GT mesh, at least 20 examples
8. missed GT object panels
9. duplicate prediction panels
10. same-frame cannot-link violation panels
11. one large bad mask split into multiple slots example
12. D4RT geometry Sim3 residual heatmap
```

每张图必须有 JSON sidecar：

```text
scene
method
eval_policy
AP/AP50/AP25 if available
pre_points %
union %
GT crop/full
failure tags
```

---

## 13. 明确不再作为主线的方向

以下方向已经有足够负证据，只能作为 ablation，不作为主线：

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
D4RT raw geometry with RGB-D meter-scale thresholds
connected component over mask co-membership
regionlet direct output as object
carrier component direct output as object
```

---

## 14. v13 成功与失败时的论文边界

如果 Phase D/E/G 成功，可以写：

```text
We reformulate training-free semantic 4D reconstruction as object explanation over D4RT material surfels and noisy semantic measurements, rather than 2D/3D mask merging. The proposed MDL-style object slot inference improves both own-support and same-support diagnostics, demonstrating that the gain is not merely caused by shrinking the evaluation universe.
```

如果 Phase C 证明 measurement density 是瓶颈，可以写：

```text
D4RT correspondence is reliable, but sparse 2D semantic measurements are insufficient for full object field reconstruction. D4RT-propagated masklets or video semantic measurements are necessary for robust 4D semantic field construction.
```

如果 Phase F 失败，可以写：

```text
D4RT metric geometry, even after evaluation-only Sim3 alignment, is not accurate enough to replace ScanNet RGB-D geometry for Stream3D-style static mesh segmentation. We therefore use D4RT as material correspondence and object identity evidence, not as a metric mesh replacement.
```

不能写：

```text
v13 proves full semantic 4D reconstruction if only tiny own-support improves.
v13 exceeds Stream3D if Stream3D-on-method-support is still stronger.
Dynamic tracking works if Dynamic Replica lacks object ID GT.
D4RT geometry replaces RGB-D if G2/G4 diagnostic fails.
```

---

## 15. 最低交付清单

Codex 下一轮必须至少交付：

```text
1. Complete code audit packet with pure tests pass.
2. Phase B candidate/oracle failure attribution matrix.
3. Phase C video masklet measurement density diagnostic.
4. Phase D MDL object explanation prototype.
5. Phase E posterior support export prototype.
6. Phase F D4RT geometry aligned -> Stream3D diagnostic.
7. Unified own/S0/S1/Stream3D-on-M evaluation table for all reportable methods.
8. At least 20 failure/success visualizations with JSON sidecars.
```

只有在 Phase D/E probe5 gate 通过后，才允许启动 tune30。只有 tune30 通过后，才允许启动 final。

