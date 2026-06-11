# Stream4D v12 可执行实验计划：D4RT-native Semantic 4D Object Explanation

面向 Codex 的执行计划。本文档不再把目标写成“Stream3D + D4RT”，也不再继续围绕 top-k、WTA、NMS、score sweep、support completion 做后处理堆叠。当前实验已经足够说明：如果继续把 `2D mask / regionlet / carrier component / memory slot` 当成 object primitive，静态 ScanNet 都无法稳定接近 Stream3D，更不可能支撑 feed-forward semantic 4D reconstruction and tracking。

本文所有公式均使用 Typora 友好的 `$...$` 或 `$$...$$`，不使用 display 公式的方括号语法。

---

## 0. 计划结论先行

当前失败的根因不是单纯“mask 被过滤太狠”，也不是单纯“support 不够”，而是：

**我们把 observation 当成 object 了。**

过去几轮的做法反复在尝试：

```text
2D mask -> object
regionlet -> object
carrier component -> object
mask history -> object
memory slot over existing wrong masks -> object
```

这些 primitive 都不是真正的 object。它们只是 measurement。真正的 object 应该是一个 latent entity，用来解释一组 D4RT surfels、2D masks、regionlets、appearance features、visibility、motion 和 negative evidence。

新的主线必须改成：

```text
D4RT semi-dense surfels
+ noisy 2D semantic measurements
-> latent object explanation
-> semantic 4D object field
```

而不是：

```text
mask / regionlet / carrier component
-> object candidate
-> WTA / NMS / score / memory 修补
```

v12 的目标不是马上 full ScanNet 超越 Stream3D，而是先建立一个真正正确的最小闭环：**measurement bank -> object explanation -> posterior support -> unified evaluation**。如果这个闭环在 probe5 上不能同时改善 own-support 和 cross-support，就不要进入 tune30 / final。

---

## 1. 历史结果给出的硬事实

### 1.1 B1/O1 的高分是 tiny-support high-precision subset，不是完整 reconstruction

历史 B1 own-support：

```text
B1 own:
AP / AP50 / AP25 = 0.328439 / 0.629266 / 0.884363
pre% = 3.9861
GT crop/full = 8.20 / 40.60
```

但 B1 放到 Stream3D S0/S1 support 后几乎崩溃：

```text
B1 on S0:
0.000635 / 0.004294 / 0.010768

B1 on S1:
0.016837 / 0.047534 / 0.168162
```

这说明 B1 不是完整 object reconstruction，而是 D4RT-assisted clean-mask subset selection。

### 1.2 O38 证明“support 大”也不能自动解决问题

O38 c055 own support 已经很大：

```text
O38 c055 own:
pre% = 66.6809
AP / AP50 / AP25 = 0.081038 / 0.219225 / 0.492501
```

但 O38 on S0 仍远低于 Stream3D：

```text
O38 c055 on S0:
0.033012 / 0.123089 / 0.392066

Stream3D on S0:
0.235730 / 0.414306 / 0.537786
```

这说明现在已经不能把主要问题说成“coverage 不够”。coverage 上来后 AP 仍低，说明 object hypothesis 本身错误。

### 1.3 Regionlet direct-output 失败说明 regionlet 也不是 object

R0/R1 regionlet own-support 有一定 AP，但 conflict 极高。WTA repair 后 conflict 清零，AP 仍然很低：

```text
R1b WTA own:
0.033785 / 0.127959 / 0.395010
conflict = 0

R1b on S0:
0.000288 / 0.002181 / 0.054594
```

这说明 WTA 只能清点级冲突，不能把错误 primitive 修成正确 object。

### 1.4 D4RT signal 有正贡献，但目前只体现为 mask selection

B1 相比 no-track、shuffle、area、maskcount、random controls 均明显更强。例如相比 shuffle：

```text
B1 - shuffle:
+0.1617 AP
+0.2903 AP50
+0.2960 AP25
```

这说明 D4RT ownership/track signal 不是假的。但它现在只是帮助挑出干净 masks，还没有变成 semantic 4D object field。

### 1.5 D4RT image-space correspondence 可用，metric geometry 仍需谨慎

连续帧 D4RT surfel diagnostic 显示：

```text
uv_in01_rate_mean ≈ 0.9858
self_uv_error_p90_mean ≈ 1.57 px
cycle_uv_error_p90_mean ≈ 3.27 px
```

说明 D4RT image-space correspondence 是可用的。与此同时，Sim3 residual 仍偏大，不能把 D4RT metric geometry 直接当作 ScanNet RGB-D/pose 替代品。因此 v12 的主线使用 D4RT 作为 material correspondence / ownership evidence，而不是主张 D4RT metric geometry 已经替代 ScanNet 几何。

---

## 2. v12 总体目标

v12 的整体目标是：

**实现并验证一个 training-free 的 D4RT-native semantic 4D object explanation 原型。**

具体目标分三层。

第一层是静态 ScanNet 的 object formation sanity：

```text
在 probe5 上，证明 measurement-to-object explanation 比 mask-as-object、regionlet-as-object、carrier-component-as-object 更好。
```

第二层是统一评估和差距确认：

```text
每个方法同时报告 own-support、Stream3D-on-method-support、method-on-S0、method-on-S1，不能只报 own-support。
```

第三层是 4D 表示闭环：

```text
最终 object 必须表示为 D4RT surfel ownership field，而不仅是 3D mask npz。
ScanNet npz 只是 evaluation adapter。
```

v12 不追求在第一轮直接 full ScanNet 超越。它追求可判定的科学进展：

```text
1. 证明当前失败是 primitive 上界问题，还是 object explanation 问题。
2. 证明 D4RT signal 进入了 object ownership，而不是只进入 mask selection。
3. 证明 negative evidence / unknown state 能同时改善 precision 和 recall trade-off。
4. 如果 probe5 通过，再固定参数跑 tune30；final 只允许 locked config 跑一次。
```

---

## 3. 核心科学假设

### H1：当前失败主要来自 object primitive 错误，而不是 evaluator 或单个阈值

如果 `mask / regionlet / carrier component` 本身作为 candidate 的 GT oracle 上界很低，则说明 primitive 错了；如果 oracle 上界高而当前 method 低，则说明 ownership inference 错了。

### H2：D4RT surfel 是 material correspondence evidence，不是 object seed 本身

D4RT surfel 不应该直接变 object，也不应该简单按同 mask co-membership connected component。它应该提供：

```text
positive evidence: surfel 在某 mask 内、跨帧一致、可见性高；
negative evidence: surfel 可见但长期在 object mask 外、motion/appearance 不兼容；
temporal evidence: object 在无 mask 帧仍能通过 surfel track 维持 identity。
```

### H3：2D mask / regionlet 应该作为 measurement，而不是 object

一个 mask 可以由多个 objects 解释；一个 object 也可以由多个 masks 支持。因此 object inference 的基本问题不是 mask merging，而是 measurement explanation。

### H4：unknown / reject 是必要输出

当前方法常把 ambiguous support 硬塞给某个 object，导致 large-support AP 崩。新的 posterior 必须允许：

```text
core
fringe
unknown
reject
```

不确定区域不应该强制分配。

### H5：成功方法必须同时改善 own-support 和 cross-support

如果只改善 own-support，而 `method on S0/S1` 仍接近 0，则方法仍是 tiny-support selector。如果只改善 support 但 AP 低，则 object hypothesis 仍错误。

### H6：D4RT aligned geometry -> Stream3D 是 diagnostic，不是主方法

需要单独验证 D4RT 几何精度对 Stream3D pipeline 的影响，但这个实验不能被写成我们的主方法。它的作用是归因：D4RT metric geometry、object formation、evaluation adapter 各自贡献多少误差。

---

## 4. 统一评估协议：所有实验必须遵守

每个新 config `M` 都必须输出以下行：

```text
M own:
  prediction = M
  pre_points = M
  eval_policy = own_recompute_paper_style

Stream3D on M:
  prediction = Stream3D baseline
  pre_points = M
  eval_policy = cross_fixed_support

M on S0:
  prediction = M
  pre_points = Stream3D S0
  eval_policy = cross_fixed_support

M on S1:
  prediction = M
  pre_points = historical 32f S1
  eval_policy = cross_fixed_support

M inherit parent:
  prediction = M
  pre_points = parent
  eval_policy = inherit_parent_support
  only required if M is a postprocess of parent
```

每行必须记录：

```text
AP
AP50
AP25
pre_points %
prediction union %
union in target scene %
union in target support %
GT crop/full
#pred
mean points/object
median points/object
conflict rate
tiny mask ratio <100
large mask ratio >1000
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per matched GT
runtime
GPU hours
manifest_integrity_pass
```

所有 method result manifest 必须写：

```text
uses_gt_for_prediction = false
uses_gt_for_diagnostic = false
is_method_result = true
is_diagnostic_only = false
eval_policy = own_recompute_paper_style | cross_fixed_support | inherit_parent_support
support_source = own | stream3d_s0 | stream4d_s1 | parent | named_config
geometry_source = rgbd_eval_bridge | d4rt_sim3_aligned | d4rt_raw | mixed
method_family = object_explanation | baseline | diagnostic
```

所有 GT oracle / diagnostic 工具必须写：

```text
uses_gt_for_prediction = false
uses_gt_for_diagnostic = true
is_method_result = false
is_diagnostic_only = true
```

如果普通 report scanner 发现 `is_diagnostic_only=true` 进入 method table，直接 fail。

---

## 5. 每轮必须提交的代码审计包

Codex 每轮必须提交完整审计包，不允许只给日志或只给结果表。

压缩包命名：

```text
stream4d_v12_<phase>_code_review_packet.zip
stream4d_v12_<phase>_code_review_packet.sha256
stream4d_v12_<phase>_filelist.txt
stream4d_v12_<phase>_ziptest.log
```

zip 内必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/evaluate.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_v12_*.sh
Stream3D/scripts/*.json
Stream3D/docs/stream4d_v12_执行日志.md
Stream3D/docs/stream4d_v12_实验结果复盘.md
Stream3D/outputs/audit/**/*.json
Stream3D/outputs/audit/**/*.md
Stream3D/outputs/audit/**/*.csv
Stream3D/data/evaluation/scannet/*_class_agnostic.txt
Stream3D/data/prediction/<new_configs>_class_agnostic/config_manifest.json
Stream3D/data/TMP/<new_configs>/config_manifest.json
probe5 prediction/TMP artifacts sufficient for rerun
```

每轮必须通过：

```bash
python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
python -m unittest discover tests
python -m tools.scan_reportable_configs --require-manifest --require-eval-policy
python -m tools.verify_stream4d_metric_integrity --require-manifest
```

如果本地缺 `open3d` 或 GPU 依赖，必须拆分测试：

```text
pure_python_tests
open3d_required_tests
gpu_required_tests
```

不能让 optional dependency 在 import 阶段导致所有 protocol tests 不可运行。

---

## 6. Phase 0：统一基准和差距矩阵

### 6.1 目标

固定 v12 的比较基线，避免后续继续在不同 support 下混乱解释结果。该阶段不改方法，只修评估和汇总工具。

### 6.2 假设

$H_0$：如果评估矩阵统一，则可以明确判断当前方法到底比 Stream3D 差在哪里：own-support、same-support、S0、S1、还是 candidate quality。

### 6.3 实验设置

固定 support：

```text
S0 = Stream3D original ScanNet support
S1 = historical 32f support
S2 = B1 support
S3 = O1/O2 core support
S4 = O38 c055 support
S5 = v6 compact support
```

固定 prediction：

```text
P0 = Stream3D baseline
P1 = v6 compact
P2 = B1 surfacelet singlemask
P3 = O1 core-only
P4 = O38 c055
P5 = best regionlet repaired, if available
```

生成完整 matrix：

```text
prediction P_i x support S_j
```

### 6.4 必须记录

除统一指标外，额外记录：

```text
每个 support 的 GT crop/full
每个 support 的 Stream3D AP/AP50/AP25
每个 method own 与 Stream3D-on-method-support 的差值
每个 method on S0 与 Stream3D-on-S0 的差值
每个 method on S1 与 Stream3D-on-S1 的差值
```

### 6.5 成功标准

Phase 0 通过条件：

```text
1. Stream3D on S0 与 Stream3D baseline 完全一致或差异 < 1e-6。
2. 每个 method 都有 own / Stream3D-on-method / on-S0 / on-S1。
3. 所有 config 都有 manifest 和 eval_policy。
4. 输出 gap matrix heatmap。
```

如果失败：

```text
先修 evaluator、pre_points materialization、manifest，不进入 Phase 1。
```

---

## 7. Phase 1：Measurement Bank 构建，不直接输出 object

### 7.1 目标

把所有 2D masks、regionlets、D4RT surfels、appearance / boundary / visibility evidence 存成 measurement bank。该阶段不导出最终 AP method，只做数据结构和质量诊断。

### 7.2 核心定义

D4RT surfel：

$$
s_i = \{X_i(t), \pi_i(t), v_i(t), c_i(t), a_i(t)\}_{t=1}^{T}
$$

其中：

```text
X_i(t): D4RT 预测的 3D position
π_i(t): frame t 的 2D projection
v_i(t): visibility
c_i(t): confidence
a_i(t): local RGB/DINO/CLIP-like appearance feature
```

2D measurement：

$$
z_m = (t, R_m, e_m, b_m, q_m)
$$

其中：

```text
t: frame index
R_m: mask or regionlet pixel set
e_m: appearance embedding or RGB histogram
b_m: boundary distance map
q_m: mask quality diagnostic
```

Measurement bank 记录：

```text
surfel_id
frame_id
mask_id
regionlet_id
inside_mask
inside_regionlet
boundary_distance
visibility
confidence
local_rgb_feature
depth_consistency
normal_consistency
uv_cycle_error
positive_observation
negative_observation
```

### 7.3 实验设置

用 probe5，固定连续 16 frames，grid32 / margin 0.02 D4RT surfels。使用已有 CropFormer masks。不要直接导出 object。

Codex 实现：

```text
Stream3D/stream4d/measurement_bank.py
Stream3D/tools/build_v12_measurement_bank.py
Stream3D/tools/diagnose_v12_measurement_bank.py
```

### 7.4 必须记录的指标

```text
num_surfels
num_valid_tracks
track_length_visible_mean
uv_in01_rate
self_uv_error_p90
cycle_uv_error_p90
surfel_2d_coverage_per_frame
num_mask_frames_available
num_mask_frames_missing
surfel_positive_observation_rate
mean_positive_observations_per_surfel
median_positive_observations_per_surfel
surfel_negative_observation_rate
mask_to_surfel_count_mean
mask_to_surfel_count_p10/p50/p90
regionlet_to_surfel_count_mean
boundary_safe_surfel_ratio
ambiguous_surfel_ratio: surfel inside multiple conflicting masks/regions
unobserved_surfel_ratio
```

### 7.5 判断标准

Phase 1 通过条件：

```text
1. uv_in01_rate_mean >= 0.95
2. self_uv_error_p90_mean <= 3 px
3. cycle_uv_error_p90_mean <= 6 px
4. mean_positive_observations_per_surfel >= 1.5
5. mask_to_surfel_count_mean >= 16
6. ambiguous_surfel_ratio 可被记录，不要求低，但必须可解释
```

如果不满足：

```text
若 uv/cycle 失败：
  检查是否误用 stride-10 clip；强制连续帧。
  检查 D4RT adapter vs official helper。
  降低 grid density，只修 correctness，不追 AP。

若 positive observations 太少：
  增加 mask frame frequency。
  补跑更多 frames 的 2D mask。
  使用 D4RT 将 sparse mask measurement propagate 到无 mask 帧，但只作为 diagnostic，不直接导出 AP。

若 mask_to_surfel_count 太低：
  grid32 -> grid48 或 adaptive sampling。
  在大 mask 内增加 boundary-safe surfel sampling。
```

### 7.6 可视化

每个 scene 输出：

```text
measurement_bank_overlay_frame_*.png
surfel_tracks_colored_by_mask_obs.mp4
mask_with_inside_surfels_and_boundary_distance.png
ambiguous_surfels_overlay.png
unobserved_surfels_overlay.png
```

---

## 8. Phase 2：Candidate Primitive 上界诊断

### 8.1 目标

用 GT-only diagnostic 回答：当前 measurement bank 里的 primitive 有没有足够上界？如果 oracle 都低，说明 primitive 错；如果 oracle 高而 method 低，说明 inference 错。

该阶段所有输出必须是 diagnostic-only，不允许进入 method table。

### 8.2 假设

$H_1$：如果 measurement primitive 质量足够，GT oracle 可以从候选集合中选出高 AP object set。否则必须重做 primitive，不能继续调 inference。

### 8.3 候选 primitive 集合

构造四类 candidate，不改变 method：

```text
C_mask:
  原始 2D mask backprojection candidate。

C_regionlet:
  每个 2D mask 内按 depth/normal/boundary/surfel seeds 生成 regionlet candidate。

C_surfel_cluster:
  在 D4RT surfel graph 上按 local geometry + appearance + same-mask weak evidence 生成小 cluster。
  注意：cluster 不作为 method object，只作为 candidate upper-bound diagnostic。

C_hybrid:
  regionlet + surfel cluster support 的 conservative union。
```

### 8.4 Oracle 诊断

GT oracle 只能用于诊断：

```text
oracle_top1_per_gt
oracle_set_packing
oracle_best_subset_under_#pred_budget
```

记录：

```text
Oracle AP/AP50/AP25
Oracle per-GT best IoU mean
Oracle GT IoU>=0.25 count
Oracle GT IoU>=0.50 count
Oracle candidate count
Oracle selected count
Oracle duplicate rate
Candidate purity
Candidate coverage
Candidate boundary error
Candidate over-merge rate
Candidate under-fragmentation rate
```

### 8.5 判断标准

如果：

```text
oracle AP50 < Stream3D AP50 on same support - 0.10
```

则 primitive 不合格。Codex 必须尝试：

```text
1. regionlet 分裂更细：depth/normal/RGB boundary 优先。
2. 允许 object 由多个 regionlets 解释，而不是一个 regionlet 输出 object。
3. 提高 D4RT surfel density。
4. 增加 mask observation 帧数。
```

如果：

```text
oracle AP50 >= Stream3D AP50 on same support
method AP50 << oracle AP50
```

则 primitive 有潜力，主要问题是 object explanation / ownership inference。进入 Phase 3。

如果 oracle 高但 candidate count 极大：

```text
记录 oracle-selected vs raw count ratio。
这说明需要 MDL / set-packing / object explanation，不是直接导出 candidate。
```

---

## 9. Phase 3：Object Explanation 原型

### 9.1 目标

实现第一个真正意义上的 object explanation prototype：mask/regionlet/surfel 都是 measurement，object 是 latent slot。该阶段是 v12 核心。

### 9.2 关键表示

Object slot：

$$
O_k = (S_k^{core}, S_k^{fringe}, S_k^{unknown}, S_k^{reject}, \theta_k)
$$

其中：

```text
S_k^{core}: 高置信 surfels / regionlets
S_k^{fringe}: 可能属于 object，但证据不足
S_k^{unknown}: 不确定，不导出
S_k^{reject}: 被 negative evidence 排除
θ_k: object appearance、trajectory、mask prototype、semantic embedding、lifecycle state
```

Surfel ownership posterior：

$$
p(y_i=k \mid Z,S) = \sigma(\ell_{ik})
$$

第一版 score：

$$
\ell_{ik}=w_p P_{ik}+w_a A_{ik}+w_t T_{ik}+w_g G_{ik}-w_n N_{ik}-w_b B_{ik}-w_x X_{ik}
$$

其中：

```text
P_ik: positive mask/regionlet support
A_ik: appearance consistency
T_ik: D4RT temporal consistency
G_ik: geometry/local adjacency consistency
N_ik: visible-outside negative evidence
B_ik: boundary risk
X_ik: conflict with other object slot
```

Measurement explanation energy：

$$
E(Y,O)=
\sum_i \psi_i(y_i)
+\sum_{i,j}\psi_{ij}(y_i,y_j)
+\sum_k \Omega(O_k)
+\sum_m \Phi(z_m,O)
$$

第一版不需要复杂优化器，采用 deterministic EM-like split-merge：

```text
Object birth:
  用 B1-like clean single mask / boundary-safe surfel core 生成 object slot。
  该 mask 只用于 birth，不等于 final object。

E-step:
  对 surfel / regionlet 计算所有 object slots 的 ownership score。
  只把 high posterior 放入 core。
  中等 posterior 放入 fringe。
  低 posterior 或 high conflict 放入 unknown/reject。

M-step:
  更新 object appearance prototype。
  更新 D4RT trajectory / visibility summary。
  更新 owned measurement list。
  计算 object validity。

Split:
  如果一个 slot 内部存在两个不连通 regionlet groups、
  或者 motion / appearance / negative evidence 强烈多峰，
  则拆分 slot。

Merge:
  只有当两个 slots 在多帧 D4RT surfel、appearance、geometry 上一致，
  且没有 same-frame cannot-link，才允许合并。

Reject:
  如果 slot 只能被单个不稳定 measurement 支持、
  或 visible-outside negative ratio 太高，
  则不导出。
```

### 9.3 Codex 需要实现的文件

```text
Stream3D/stream4d/object_explanation.py
Stream3D/stream4d/object_slot.py
Stream3D/stream4d/evidence_terms.py
Stream3D/tools/export_v12_object_explanation.py
Stream3D/tools/diagnose_v12_object_explanation.py
Stream3D/tests/test_v12_object_explanation.py
```

### 9.4 关键默认参数

先用 conservative 参数，不要盲目扫大网格：

```text
birth_min_surfels = 16
birth_min_boundary_safe_ratio = 0.65
birth_max_ambiguous_ratio = 0.25
core_posterior_threshold = 0.70
fringe_posterior_threshold = 0.45
reject_negative_threshold = 0.40
visible_outside_negative_weight = 1.0
boundary_risk_weight = 0.5
appearance_weight = 0.3
d4rt_temporal_weight = 0.5
max_slots_per_frame_mask = 3
min_core_surfels_per_object = 12
min_export_points_per_object = 100
unknown_export = false
```

### 9.5 实验对照

必须比较：

```text
M0 = B1 single-mask selector
M1 = mask-as-object baseline
M2 = regionlet-as-object baseline
M3 = carrier-component baseline
M4 = object explanation without negative evidence
M5 = object explanation with negative evidence
M6 = object explanation with shuffled D4RT evidence
M7 = object explanation with no D4RT temporal evidence
```

### 9.6 必须记录的指标

除了统一评估指标，还必须记录 object explanation 内部指标：

```text
num_birth_slots
num_active_slots
num_rejected_slots
num_split_events
num_merge_events
num_unknown_surfels
assigned_surfel_ratio
core_surfel_ratio
fringe_surfel_ratio
reject_surfel_ratio
visible_outside_negative_ratio
same_frame_cannot_link_violations
mean_object_positive_evidence
mean_object_negative_evidence
mean_object_boundary_risk
mean_object_appearance_consistency
mean_object_temporal_consistency
measurement_explained_ratio
measurement_multi_explained_ratio
measurement_unexplained_ratio
```

GT diagnostic：

```text
per-GT failure category:
  no_candidate
  filtered_candidate
  wrong_assignment
  over_merge
  over_split
  boundary_bad
```

### 9.7 成功标准

Probe5 minimal pass：

```text
M5 own AP >= max(B1 own AP - 0.03, 0.30)
M5 own AP50 >= 0.55
M5 own AP25 >= 0.75

M5 on S0 AP >= 0.08
M5 on S0 AP50 >= 0.18
M5 on S0 AP25 >= 0.45

M5 on S1 AP >= 0.18
M5 on S1 AP50 >= 0.35
M5 on S1 AP25 >= 0.60

Stream3D on M5 support - M5 own <= 0.08 AP
or M5 AP50/AP25 better while AP gap <= 0.05
```

Ablation pass：

```text
M5 > M4 on AP50 or S0 AP
M5 > M6 by at least 0.05 AP50
M5 > M7 by at least 0.03 AP50
```

如果 M5 own 高但 S0/S1 仍接近 B1：

```text
说明仍是 tiny-support selector。
Codex 必须提高 measurement coverage 或 object growth，但不能 full-mask backproject。
```

如果 M5 support 大但 AP 低：

```text
说明 ownership posterior 太宽或 negative evidence 太弱。
Codex 优先提高 reject/unknown，不要继续扩 support。
```

如果 M5 和 shuffled D4RT 差不多：

```text
说明 D4RT temporal evidence 没有真正进入 object inference。
Codex 必须检查 evidence_terms.py 是否只用了 mask area / boundary，重新设计 D4RT temporal term。
```

---

## 10. Phase 4：Negative Evidence 和 Unknown 状态专项实验

### 10.1 目标

验证“允许 unknown / reject”是否能打破过去的二元困境：

```text
少过滤 -> noisy support
多过滤 -> tiny support
```

### 10.2 实验设置

在 M5 基础上做 ablation：

```text
N0: no negative evidence, no unknown
N1: negative evidence only
N2: unknown only
N3: negative + unknown
N4: negative + unknown + split
N5: negative + unknown + split + conservative fringe export
```

### 10.3 必须记录

```text
unknown_surfel_ratio
reject_surfel_ratio
exported_surfel_ratio
GT crop/full
AP/AP50/AP25
per-GT best IoU mean
missed GT count
duplicate predictions per GT
over_merge_count
over_split_count
visible_outside_negative_violations
```

### 10.4 判断标准

如果 N3/N4/N5 相比 N0：

```text
AP50 提升 >= 0.05
duplicate predictions per GT 下降
over_merge_count 下降
missed GT 不显著上升
```

则 H4 成立。

如果 unknown 比例过高导致 S0/S1 崩：

```text
降低 unknown 阈值；
增加 object birth recall；
加入 measurement propagation；
但禁止直接 full-mask backproject。
```

如果 negative evidence 使 AP25 大幅下降：

```text
negative 太强或误伤 recall。
Codex 先降低 visible-outside 权重，保留 boundary negative。
```

---

## 11. Phase 5：D4RT aligned geometry -> Stream3D 归因实验

### 11.1 目标

单独回答：

```text
如果把 D4RT 几何对齐到 ScanNet GT/RGB-D geometry 后给原版 Stream3D 使用，指标会如何变化？
```

该实验是 diagnostic，不是主方法。它用于判断 D4RT metric geometry 精度对 Stream3D pipeline 的影响。

### 11.2 关键原则

不能再直接复用 RGB-D meter-scale 超参。D4RT geometry 必须先做：

```text
1. 连续帧 D4RT clips。
2. scene-level 或 window-level Sim3 evaluation-only alignment。
3. density-normalized NN radius。
4. density-normalized manifold threshold。
5. min_points 根据 D4RT point density 自适应。
```

Sim3 只能用于几何归因和 evaluation adapter，不能影响 object grouping / semantic inference。

### 11.3 实验设置

配置：

```text
G0: 原版 Stream3D RGB-D/pose geometry。
G1: Stream3D + D4RT raw geometry，density-normalized thresholds。
G2: Stream3D + D4RT scene-level Sim3 aligned geometry。
G3: Stream3D + D4RT window-level Sim3 aligned geometry。
G4: Stream3D + D4RT scene-level Sim3 + density-normalized manifold refining。
G5: Stream3D + D4RT window-level Sim3 + track-consistency manifold replacement。
```

Codex 实现：

```text
Stream3D/tools/materialize_d4rt_geometry_for_stream3d_v12.py
Stream3D/tools/run_stream3d_with_geometry_adapter_v12.py
Stream3D/tools/diagnose_d4rt_geometry_to_stream3d_v12.py
```

### 11.4 必须记录

Geometry metrics：

```text
sim3_anchor_count
sim3_scale
sim3_residual_median
sim3_residual_p90
sim3_residual_p95
D4RT point spacing p10/p50/p90
NN radius used
manifold threshold used
depth-like coverage
mesh materialization hit rate
per-frame projection error
per-object geometric compactness
```

Stream3D metrics：

```text
AP/AP50/AP25
pre%
union%
GT crop/full
#pred
mean points/object
Stream3D local set-cover selected masks
manifold refining kept ratio
manifold refining rejected ratio
object fragmentation
object merge count
```

### 11.5 判断标准

如果 G2/G4 相比 G0：

```text
AP drop <= 3 points
AP50 drop <= 5 points
```

说明 D4RT geometry after alignment 可作为未来几何路径候选。

如果 G2/G4 明显低于 G0，但非零：

```text
说明 D4RT metric geometry 对 ScanNet still imperfect。
论文中不要 claim geometry replacement；只 claim correspondence / semantic field path。
```

如果 G1/G2/G3 仍为 0 或 nan：

```text
检查 density normalization 是否生效；
检查 materialized point cloud 是否真的进入 Stream3D local point cloud；
检查 D4RT xyz coordinate/ref frame；
不要把 0 AP 直接解释为 D4RT 不行。
```

---

## 12. Phase 6：Tune30 和 locked final

### 12.1 进入条件

只有当 Phase 3/4 probe5 通过以下 gate 才能进入 tune30：

```text
M5 on S0 AP >= 0.08
M5 on S1 AP >= 0.18
M5 own AP >= 0.30
D4RT ablation real > shuffle
metric integrity pass
```

### 12.2 Tune30 规则

只允许在 tune30 上选择：

```text
birth_min_surfels
core_posterior_threshold
fringe_posterior_threshold
reject_negative_threshold
max_slots_per_frame_mask
min_export_points_per_object
```

不允许在 tune30 上无限扫大网格。最多：

```text
3 values per parameter
total configs <= 36
```

每个 config 必须跑 full unified eval。

### 12.3 Final 规则

选定唯一 config 后，final split 只跑一次。final 报告：

```text
own
Stream3D on method support
method on S0
method on S1
inherit parent, if applicable
```

如果 final own 好但 cross-support 崩：

```text
只允许写 visible-subset result，不允许写超越 Stream3D。
```

---

## 13. Phase 7：Dynamic / Replica-Dynamic 准备实验

### 13.1 进入条件

静态 ScanNet probe5 必须至少通过 Phase 3 minimal pass。否则不要把动态结果作为遮羞布。

### 13.2 数据检查

Codex 先运行：

```text
check_dynamic_replica_env_v12.py
```

记录：

```text
data_root_exists
split_dir_exists
annotation_exists
images_count
depth_count
trajectory_count
semantic_gt_exists
instance_gt_exists
object_id_exists
camera_fields_present
usable_scene_count
```

如果缺 instance/object ID：

```text
不能报告 IDF1、MOTA、official 4D IoU。
只能报告 qualitative 或 pseudo consistency，且标为 diagnostic。
```

### 13.3 动态指标

如果 GT 可用，记录：

```text
IDF1
MOTA
ID switches
fragmentation
track recall
track precision
4D IoU over time
occlusion reactivation accuracy
moving-object AP/AP50/AP25
static-object AP/AP50/AP25
time-sensitive query success
```

如果 GT 不可用，记录：

```text
D4RT trajectory self-consistency
object slot persistence
mask observation consistency
occlusion qualitative panels
```

---

## 14. 可视化要求

每个 phase 必须输出可视化，不允许只给 AP 表。

### 14.1 ScanNet object failure panels

每个 scene 输出至少：

```text
top false positives overlay
top missed GT overlay
duplicate predictions for same GT
over-merge example
over-split example
method vs Stream3D on same support
```

### 14.2 Measurement / explanation 可视化

输出：

```text
D4RT surfels colored by object posterior
surfel core/fringe/unknown/reject map
mask measurement explained by multiple object slots
visible-outside negative evidence overlay
object slot birth/growth timeline
split/merge decisions timeline
```

### 14.3 统一图表

```text
AP heatmap: prediction x support
support ratio bar
GT crop/full bar
method-vs-Stream3D same-support delta
oracle gap plot
D4RT real vs shuffled ablation bar
unknown/reject ratio vs AP scatter
```

---

## 15. 并行执行安排

为了加速，Codex 分 4 个 lane 并行：

### Lane A：评估和审计

```text
Phase 0
manifest scanner
metric integrity
gap matrix
visualization
```

如果 Lane A 不通过，其他 lane 的 AP 只能作为临时日志，不进入复盘主表。

### Lane B：Measurement Bank

```text
Phase 1
surfel field
mask/regionlet measurement bank
coverage diagnostics
```

### Lane C：Oracle Upper Bound

```text
Phase 2
candidate primitive upper bound
failure category
oracle gap
```

### Lane D：Object Explanation

```text
Phase 3/4
object slot posterior
negative evidence
unknown/reject
export
```

Lane B/C/D 共享 measurement bank。不要每个 lane 重跑 D4RT，除非 measurement bank 被判定有问题。

---

## 16. 不满足条件时的尝试方向

### 16.1 如果 own-support 仍高但 S0/S1 崩

说明仍是 tiny-support selector。Codex 尝试：

```text
1. 增加 object birth recall，但保持 object slot posterior，不要 full-mask export。
2. 使用 regionlet measurement，而不是 entire mask support。
3. 降低 birth_min_surfels，但提高 negative evidence。
4. 增加 mask frame frequency。
5. 引入 unknown，不要把 ambiguous surfels 强制归类。
```

### 16.2 如果 support 大但 AP 低

说明 ownership 太宽或 primitive 污染。Codex 尝试：

```text
1. 提高 core_posterior_threshold。
2. 增强 visible-outside negative evidence。
3. 加强 same-frame cannot-link。
4. 切分大 mask into regionlets。
5. 只导出 core+conservative fringe。
```

### 16.3 如果 D4RT ablation 不显著

说明 D4RT 没被用上。Codex 尝试：

```text
1. 检查 shuffled-D4RT 是否真的破坏 surfel-to-mask assignment。
2. 确认 temporal term T_ik 进入 posterior。
3. 加入 cycle error / visibility consistency。
4. 对比 source-frame-only vs multi-frame。
```

### 16.4 如果 oracle upper bound 低

说明候选 primitive 错。Codex 尝试：

```text
1. regionlet 更细分。
2. 提高 surfel density。
3. 增加 mask observations。
4. 增加 boundary-safe region proposal。
5. 引入 local RGB/depth superpixel。
```

### 16.5 如果 oracle 高但 method 低

说明 inference 错。Codex 尝试：

```text
1. 更强 object-level set packing。
2. 增加 MDL penalty，限制重复解释同一 measurement。
3. 增加 split/merge search。
4. 加强 negative evidence。
```

---

## 17. 本计划的最低可接受交付

Codex 本轮至少必须交付：

```text
1. 完整 code audit packet。
2. Phase 0 unified matrix。
3. Phase 1 measurement bank diagnostic。
4. Phase 2 oracle upper bound diagnostic。
5. Phase 3 object explanation prototype M4/M5/M6/M7。
6. 每个 config 的 own / Stream3D-on-method / on-S0 / on-S1。
7. 一份清楚说明：失败是 primitive 上界低，还是 inference 低于 oracle。
```

如果 Phase 3 没有超过现有 O38 on S0：

```text
不要继续包装为方法成功。
必须回到 Phase 2 的 oracle/primitive 诊断。
```

如果 Phase 3 通过 probe5 gate：

```text
固定小范围参数进入 tune30。
```

如果 tune30 通过：

```text
锁定 config 跑 final。
```

---

## 18. 禁止事项

禁止把以下结果写成 method success：

```text
只在 own-support 高，但 S0/S1 接近 0。
GT oracle selected candidates。
D4RT geometry diagnostic。
Dynamic Replica 无 GT 的 qualitative result。
Stream3D-on-method-support 高分。
```

禁止继续作为主线：

```text
top-k / score sweep
full-mask fringe + WTA
regionlet direct output
carrier connected component
zero-conflict as objective
scene memory over wrong masks
own-support only reporting
```

这些可以作为 ablation 或 negative evidence，不能作为新的顶会方法核心。

---

## 19. 论文级创新点对齐

如果 v12 成功，论文贡献应写为：

1. **New formulation**  
   Semantic 4D reconstruction is formulated as training-free object explanation over D4RT material surfels, not 2D/3D mask merging.

2. **Measurement-as-evidence representation**  
   2D masks, regionlets, visibility, appearance, and language features are noisy measurements; object is a latent explanatory variable.

3. **Signed evidence for object ownership**  
   The method uses both positive support and negative evidence such as visible-outside contradiction, cannot-link, boundary risk, and motion incompatibility.

4. **Feed-forward queryable semantic 4D field**  
   The output is not a 3D mask list but an object-surfels ownership field:

   $$
   O_k(t)=\{X_i(t), \pi_i(t) \mid y_i=k, v_i(t)=1\}
   $$

5. **Training-free and no per-scene optimization**  
   Frozen D4RT and frozen 2D/vision-language models provide measurements; object inference is deterministic / optimization-light and uses no 3D/4D semantic labels.

这才是 feed-forward semantic 4D reconstruction and tracking 的主线。
