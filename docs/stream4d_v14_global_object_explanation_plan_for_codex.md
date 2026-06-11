# Stream4D v14：从 mask 后处理转向全局 object explanation 的可执行实验计划

本文面向 Codex 执行。v14 不再把目标写成“Stream3D + D4RT”，也不再继续围绕 top-k、WTA、NMS、score sweep、scene memory over wrong masks、full-mask backprojection 做工程堆叠。v13 的结果已经足够说明：当前方法连静态 ScanNet 的 object formation 都没有站稳。下一步必须从 **object primitive** 和 **inference formulation** 改起。

本文所有公式使用 Typora 友好的 `$...$` 或 `$$...$$` 格式，不使用 `\[\]`。

---

## 0. 硬性边界：GT / Sim3 / 评估协议

### 0.1 GT 对齐只允许出现在评估或测试诊断中

本项目中“对齐”特指 Sim3、Umeyama、Procrustes、GT/RGB-D anchor based alignment 等会把 D4RT 坐标系映射到 ScanNet GT/RGB-D/mesh/world 坐标系的操作。规则如下：

```text
允许：
1. evaluation-only / diagnostic-only 的 Sim3 对齐。
2. D4RT geometry aligned -> Stream3D 的归因实验。
3. 计算指标、误差、upper bound、failure attribution 时读取 GT。
4. D4RT 自己窗口之间的 self-alignment / self-stitching，只使用 D4RT 预测、可见性、confidence 或视频自身信息，不使用 GT。

禁止：
1. 方法内部 object birth / ownership / selection / memory / support export 使用 GT Sim3。
2. 使用 ScanNet GT instance mask 或 GT semantic label 生成 prediction。
3. 用 GT-aligned geometry 的结果进入 method table。
4. 把 diagnostic-only 结果写成 reportable method result。
```

所有涉及 GT 或 GT/RGB-D anchor 的工具必须写 manifest：

```text
uses_gt_for_prediction = false
uses_gt_for_diagnostic = true
is_method_result = false
is_diagnostic_only = true
```

普通 report scanner 必须拒绝这些结果进入 method table。D4RT 自己和自己对齐，例如 window-to-window stitching，只要不使用 GT/RGB-D/ScanNet mesh anchor，可以进入方法内部，但必须记录：

```text
alignment_source = d4rt_self_only
uses_gt_for_prediction = false
uses_gt_for_diagnostic = false
```

### 0.2 每个实验必须统一报告四类评估

任何方法 $M$ 都必须输出以下四行，缺一不可：

```text
M own:
  prediction = M
  pre_points = M
  eval_policy = own_recompute_paper_style

Stream3D on M:
  prediction = Stream3D baseline
  pre_points = M
  eval_policy = stream3d_on_method_support

M on S0:
  prediction = M
  pre_points = Stream3D S0 support
  eval_policy = method_on_stream3d_support

M on S1:
  prediction = M
  pre_points = historical 32f support
  eval_policy = method_on_s1_support
```

如果 $M$ 是从 parent config 后处理得到，还必须报告：

```text
M inherit parent:
  prediction = M
  pre_points = parent
```

每行必须记录：

```text
AP / AP50 / AP25
pre_points %
prediction union %
union in target scene %
union in target pre_points %
GT crop/full
#pred
objects/scene
points/object
conflict rate
tiny mask ratio <100 vertices
large mask ratio >1000 vertices
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per matched GT
method manifest path
uses_gt_for_prediction
uses_gt_for_diagnostic
is_method_result
is_diagnostic_only
runtime
```

如果某个方法 own-support 高，但 `Stream3D on M` 明显更高，不能 claim object quality 胜利。如果某个方法 own-support 高，但 `M on S0/S1` 近零，必须写成 tiny-support high-precision subset，不得写成 reconstruction success。

---

## 1. v13 独立结论：现在到底错在哪里

### 1.1 代码审计边界

我解压了 `stream4d_v13_probe5_code_review_packet.zip`。该包包含 `stream4d/`、`tools/`、`tests/`、logs、outputs、prediction/TMP artifacts，但缺少完整 `evaluation/` 源码目录；在干净目录里：

```text
python3 -m py_compile stream4d/*.py tools/*.py tests/*.py 通过。
python3 -m unittest discover tests 失败，原因是 tests/test_stream4d_protocol_fixes.py import tools.oracle_candidate_upper_bound，进而 import evaluation.constants，但 zip 内没有 evaluation package。
```

因此 v13 包比 v12 更完整，但仍不是完全 self-contained runnable packet。Codex 下一轮必须把 `evaluation/evaluate.py`、`evaluation/constants.py`、`evaluation/__init__.py` 和 minimal evaluator dependencies 打包，否则人工审计无法独立复跑 protocol tests。

指标安全方面，v13 final reportable scan 显示 9 个 reportable method configs 均有 manifest，没有 `uses_gt_for_prediction`，没有 suspicious method config，metric integrity 的 AP core hash 一致。当前没有看到非 oracle 方法用 GT 生成 prediction 的证据。

### 1.2 v13 的主要数值结论

v13 最强的 reportable own-support 结果仍然是小 support 或 narrow support：

```text
M13c full-mask ablation own:
AP / AP50 / AP25 = 0.224575 / 0.419119 / 0.781728
pre% = 4.3855%
GT crop/full = 8.2 / 40.6

M13d posterior WTA repair own:
AP / AP50 / AP25 = 0.161109 / 0.427857 / 0.793144
pre% = 2.0522%
conflict = 0

C_surfel unsup own:
AP / AP50 / AP25 = 0.228316 / 0.460285 / 0.778069
pre% = 4.2916%
```

同 support 对比更关键：

```text
Stream3D on M13c support:
0.338269 / 0.511111 / 0.694444
M13c own:
0.224575 / 0.419119 / 0.781728
AP gap = -0.113695

Stream3D on C_surfel support:
0.358958 / 0.549096 / 0.740956
C_surfel own:
0.228316 / 0.460285 / 0.778069
AP gap = -0.130642

Stream3D on C_regionlet support:
0.400714 / 0.568847 / 0.705466
C_regionlet own:
0.045679 / 0.122830 / 0.266596
AP gap = -0.355035
```

放到 S0/S1 后，M13a/M13b/M13c/M13d 基本仍然崩溃。M13c on S0 只有：

```text
0.002030 / 0.005451 / 0.010602
```

这说明 v13 的 object explanation 仍然没有解决静态 ScanNet 的 object formation。它最多说明：D4RT evidence 可以在 tiny observed support 上找到一些高质量局部物体，但无法变成完整 object field。

### 1.3 v13 最关键的负证据：broad-support upper bound 不够，tiny-support upper bound 很高

v13 oracle upper bound 很有信息量：

```text
C_surfel oracle:
0.395062 / 0.750000 / 0.993360
pre% = 4.2916%

C_regionlet oracle:
0.338574 / 0.613208 / 0.829643
pre% = 18.5455%

C_hybrid oracle:
0.256256 / 0.495495 / 0.702512
pre% = 52.8088%
```

这个结果不是在说“solver 再好一点就可以”。它说明候选空间本身分裂成两种坏状态：

```text
tiny support primitive:
  上界很高，但只覆盖非常小的 observed subset。

broad support primitive:
  coverage 高，但 oracle upper bound 也不够强，说明 primitive 本身跨物体、错边界、碎片化严重。
```

因此 v14 的第一优先级不是继续调 MDL 参数，而是重做 primitive。只有当 broad-support candidate pool 的 oracle AP/AP50 明显提高，才有必要投入更复杂的 object solver。

### 1.4 v13 的“video masklet density”没有真正解决 semantic density

C1/C2/C3 masklet 的统计几乎完全相同：

```text
masklets/scene = 247.8
frames/birth = 14.466
available mask agreement ≈ 0.9925
unobserved surfel ratio ≈ 0.8832
```

C0 原始 measurement bank 的 unobserved ratio 是 `0.1458`，而 C1/C2/C3 masklet 变成 `0.8832`。这不是 semantic density 提升，而是说明 masklet 只沿着 birth surfels 传播，绝大多数 surfels 没有被 object/masklet 覆盖。代码也验证了这一点：`build_video_masklet_bank()` 对每个 birth group 只在 `birth_surfels` 内按 visibility 生成 rows，不会把新的未观测 surfels 吸收到 object hypothesis 里。

这解释了为什么 masklet candidate own AP 低、S0/S1 更低。v14 必须把“masklet”从 birth-surfels-only 传播改成 **measurement proposal over atom bank**：masklet 可以给 atom 提供 likelihood，但不能自己成为 object，也不能只复用 birth surfels。

### 1.5 v13 的 MDL 不是全局 MDL，只是 source-mask slot filtering

`object_explanation_mdl.py` 有 MDL 相关变量，例如 `model_cost`、`overlap_penalty`、`unexplained_penalty`，但当前 selection 的实质是：

```text
1. birth_groups(bank) 按 (src_frame, src_mask_id) 分组。
2. 对每个 group 内 surfels 计算 posterior。
3. 每个 group 形成一个 ObjectSlot。
4. 根据单 slot energy 排序。
5. 贪心选择，主要通过 core overlap 防止重复。
6. unexplained_measurement_penalty 只在 diagnostic 中统计，没有参与全局优化。
```

这不是全局 object explanation。它不能主动选择一组 object slots 去解释最多 measurements，也不能 split 一个 bad mask，也不能把多个局部 atoms 合成一个 object，也不能为了覆盖未解释 measurements 引入新 slot。它仍然是：

```text
source mask birth -> local posterior -> object candidate
```

这就是为什么 M13c full-mask ablation 比 posterior-support M13a/M13b 更好：full mask backprojection 虽然脏，但至少补了一点 support；posterior support 更干净但太小，S0/S1 基本为零。

### 1.6 D4RT 几何诊断的边界

v13 geometry diagnostic 复用了 v10 G1-G5。G2/G4 在 diagnostic own-support 下有：

```text
18.8825 / 36.4384 / 48.6341
Sim3 median residual ≈ 0.0786
conflict ≈ 75.17%
```

但这些是 diagnostic-only，并且使用 GT/RGB-D anchors 做 Sim3 评估对齐。按照本计划的硬约束，它们不能作为 method result。它们的价值只是说明：

```text
D4RT geometry after evaluation Sim3 alignment can project something useful into ScanNet space, but conflict very high and S0/S1 still near zero。
```

v14 仍然保留 D4RT geometry aligned -> Stream3D diagnostic，但只用于归因，不进入方法主表。

---

## 2. v14 的总体目标

v14 的目标是验证一个真正可能成为顶会贡献的方向：

```text
D4RT material surfels + noisy semantic measurements -> global latent object explanation -> queryable semantic 4D field
```

不要再把 mask、regionlet、carrier component、masklet 或 memory slot 当 object。它们只能作为 measurements 或 atoms。v14 要建立一个最小但真实的闭环：

```text
D4RT surfels
  -> atom bank
  -> measurement bank
  -> candidate object factors
  -> global object set packing / explanation
  -> posterior support export
  -> unified own/cross evaluation
```

核心问题拆成四个假设：

```text
H1: 当前失败的直接原因是 broad-support primitive 上界不足，而不是 solver 参数没调好。
H2: 将 full mask / regionlet / masklet 降级为 measurement，并构建更小、更纯的 surfel atoms，可以提升 broad-support oracle upper bound。
H3: 只有在 atom upper bound 充足后，全局 set-packing / object explanation 才能缩小和 Stream3D same-support 的差距。
H4: D4RT 的价值应体现在 surfel ownership、split/merge、temporal consistency，而不是 GT-aligned metric geometry 或 tiny mask selection。
```

---

## 3. Phase 0：代码审计与统一评估基线

### 3.1 目标

确保 v14 开始前，评估协议、代码包、manifest、metric integrity 都可审计。任何算法结果都必须能和 Stream3D 在同 support 下比较。

### 3.2 Codex 任务

1. 修复 code audit packet 打包问题。下一轮 zip 必须包含：

```text
Stream3D/evaluation/evaluate.py
Stream3D/evaluation/constants.py
Stream3D/evaluation/__init__.py
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_v14_*.sh
Stream3D/data/evaluation/scannet/*_class_agnostic.txt
Stream3D/data/prediction/<v14 configs>_class_agnostic/config_manifest.json
Stream3D/data/TMP/<v14 configs>/config_manifest.json
Stream3D/outputs/audit/v14_*/*.json, *.csv, *.md, *.png
```

2. 将 tests 拆成：

```text
pure-python tests
open3d-required tests
gpu-required tests
```

必须保证：

```text
python -m unittest discover tests -p '*pure*.py
```

在无 open3d/GPU 的干净审计环境中通过。

3. 固定 Phase 0 baseline matrix：

```text
P0 Stream3D on S0
P0 Stream3D on S1
B1 on S0/S1/own
O38 on S0/S1/own
M13c on S0/S1/own
M13d on S0/S1/own
C_surfel on S0/S1/own
```

### 3.3 必须记录的指标

记录 0.2 节所有字段，并额外记录：

```text
evaluator_ap_core_hash
reportable_config_scan summary
object_dict/prediction alignment mean/min IoU
num_missing_manifest
num_uses_gt_for_prediction
num_diagnostic_only_in_method_table
```

### 3.4 成功标准

```text
py_compile pass
pure-python unittest pass
reportable scan: num_suspicious_configs=0
metric integrity: phase0_pass=true
所有 baseline matrix 行落盘并被 md/csv/json 汇总
```

### 3.5 不满足条件时 Codex 先尝试什么

```text
如果 tests 失败是 optional dependency：把相关 tests 标成 skipUnless 或拆分到 open3d/gpu group。
如果 manifest 缺字段：写 update_config_manifest_fields.py 补齐，并重新跑 scan。
如果 object_dict/pred 不对齐：停止评估，修 export 记录 object_id_to_column 和 point_ids。
如果 evaluator 源码缺失：重新打包，不能继续报新指标。
```

---

## 4. Phase 1：Failure decomposition，不再只看 AP

### 4.1 目标

直接回答当前失败到底来自哪里：

```text
A. 没有候选覆盖 GT；
B. 候选存在但被过滤；
C. 候选存在但 object assignment 错；
D. object assignment 对，但 boundary/support IoU 不够；
E. scoring/ranking 把好候选排低；
F. support/evaluation universe 太小导致 own-support illusion。
```

这是 v14 的第一个关键诊断。没有这个诊断，继续改 solver 或 primitive 都是在猜。

### 4.2 实验设计

对以下 candidate banks 做 GT-only diagnostic，不进入 method table：

```text
B1/O1 clean-core banks
O38 broad-support bank
v13 C_mask
v13 C_regionlet
v13 C_surfel
v13 C_masklet
v13 C_hybrid
M13c/M13d final object slots
```

对每个 GT instance $G_j$，计算候选集合 $\mathcal{C}$ 中 best IoU：

$$
IoU^*(G_j)=\max_{C_i \in \mathcal{C}} IoU(C_i,G_j)
$$

然后把每个 GT 分到以下类别：

```text
no_candidate:      max IoU < 0.10
weak_candidate:    0.10 <= max IoU < 0.25
good_candidate:    0.25 <= max IoU < 0.50
high_candidate:    max IoU >= 0.50
selected_good:     method final prediction 与 GT IoU >= 0.50
filtered_good:     candidate IoU >= 0.50 但 final 没有保留
assignment_error:  final 有 overlap，但抢错 GT 或重复预测
boundary_error:    best candidate 0.25-0.50，中心/identity 对但边界差
```

### 4.3 必须记录的指标

```text
per-scene GT count
per-GT IoU* distribution
no_candidate / weak_candidate / good_candidate / high_candidate counts
filtered_good count
assignment_error count
boundary_error count
best candidate source type
candidate count per GT
duplicate candidates per GT
final selected candidate rank for each GT
oracle AP/AP50/AP25 by candidate source
broad-support oracle AP/AP50/AP25
same-support Stream3D gap
```

### 4.4 可视化

每个 probe5 scene 至少输出 6 张图：

```text
1. GT objects colored by failure category。
2. top missed GT with best candidate overlay。
3. filtered_good examples：被过滤掉但 IoU>=0.5 的候选。
4. assignment_error examples：多个 prediction 抢同一 GT。
5. boundary_error examples：中心对但边界差。
6. support illusion examples：own support 中成功但 S0/S1 失败的同一 object。
```

### 4.5 成功标准

不是要求 AP 提升，而是要得到可决策诊断：

```text
如果 broad-support candidate oracle AP50 < 0.60：primitive 不够好，进入 Phase 2。
如果 broad-support candidate oracle AP50 >= 0.60 但 method AP50 低：solver/selection 错，进入 Phase 3。
如果 filtered_good 很多：过滤策略错，需要 object posterior growth，而不是 harder filter。
如果 no_candidate 很多：measurement density 或 atom coverage 不够，进入 Phase 2A。
```

### 4.6 不满足条件时 Codex 先尝试什么

```text
如果 GT diagnostic 工具太慢：先在 scene0050_00 和 scene0030_00 跑，再扩 probe5。
如果 memory 太大：只保存 top-K best candidates per GT 的 diagnostic，不保存全矩阵。
如果 visualization 失败：先输出 JSON+CSV，图像可延后，但不能跳过 failure category。
```

---

## 5. Phase 2：重做 primitive，从 mask/regionlet 改成 surfel atoms

### 5.1 目标

v13 的 primitive 失败原因是：

```text
full mask 太粗，跨物体；
regionlet direct-output 过碎且冲突；
surfel cluster 很干净但 tiny support；
hybrid support 大但 oracle upper bound 不够。
```

v14 需要构建一个新的中间 primitive：**surfel atom**。atom 不是 object，而是较小、较纯、可组合的 measurement unit。object 是 atoms 的集合。

### 5.2 新 primitive 定义

每个 atom $a_l$ 是一组 D4RT surfels：

$$
a_l = \{s_i\}_{i \in I_l}
$$

atom 必须满足：

```text
1. 内部 trajectory / visibility 一致；
2. 内部 appearance 一致；
3. 在有 mask 的帧中，不显著跨多个 mask；
4. 不跨强 2D boundary / depth discontinuity；
5. 支持 unknown，不强行覆盖无 evidence surfels。
```

### 5.3 Codex 实现

新增：

```text
Stream3D/stream4d/surfel_atom_bank.py
Stream3D/tools/build_v14_surfel_atom_bank.py
Stream3D/tools/diagnose_v14_atom_oracle.py
```

atom bank 必须保存：

```text
atom_id
surfel_indices
source_frames
mask_membership_histogram
frame_visibility_histogram
mean_rgb / appearance descriptor
trajectory descriptor
boundary_safe_ratio
negative_visible_outside_ratio
mask_entropy
trajectory_variance
atom_size
neighbor_atom_ids
```

第一版 atom 构建采用 deterministic training-free 规则：

```text
1. 在每个 source mask 内，用 D4RT surfel 的 trajectory embedding、RGB/appearance、2D connected adjacency 做初始 split。
2. 对每个初始 split，若 mask entropy 高或 trajectory variance 高，递归拆分。
3. 对相邻小 atom，只有在多帧 co-mask + low boundary + low motion variance 下合并。
4. 对没有 mask evidence 的 surfels 保留为 unknown atoms，不进入 method export，只进入 diagnostic。
```

trajectory embedding 可先使用：

$$
\phi_i = [\pi_i(t_1), \pi_i(t_2), ..., \pi_i(t_T), v_i(t_1), ..., v_i(t_T)]
$$

不允许用 GT 监督 atom split。

### 5.4 实验设置

跑以下 atom variants：

```text
A0: source-mask only atom，作为旧方法对照。
A1: source-mask + D4RT trajectory split。
A2: source-mask + trajectory + RGB/appearance split。
A3: source-mask + trajectory + appearance + boundary/depth split。
A4: A3 + conservative atom merge。
```

每个 variant 只导出 candidate bank 和 oracle diagnostic，不直接写 method success。

### 5.5 必须记录的指标

```text
atoms/scene
mean surfels/atom
atom support pre%
atom union%
atom mask entropy mean/p90
atom trajectory variance mean/p90
atom boundary_safe_ratio mean
atom negative_visible_outside_ratio mean
atom oracle AP/AP50/AP25
broad-support atom oracle AP/AP50/AP25
Stream3D on atom support AP/AP50/AP25
atom support GT crop/full
candidate high_candidate count
filtered_good upper bound
```

### 5.6 成功标准

Phase 2 只看 upper bound，不看 final method：

```text
A3/A4 broad-support oracle AP50 >= 0.60
A3/A4 broad-support oracle AP25 >= 0.78
A3/A4 support pre% >= 25%
A3/A4 per-GT best IoU mean >= C_hybrid oracle + 0.08
A3/A4 conflict rate <= C_hybrid unsup conflict
```

如果只有 tiny support oracle 高，例如 pre% < 8%，则不算成功。

### 5.7 不满足条件时 Codex 先尝试什么

```text
如果 A1/A2/A3 atom oracle 都低：说明 source masks 本身太差，尝试引入更多 2D mask frames 或 SAM/CropFormer all-frames cache。
如果 atom 数量爆炸：提高 min surfels/atom 或 merge adjacent low-entropy atoms。
如果 atom purity 高但 support 低：降低 boundary threshold，但必须监控 mask entropy 和 oracle AP50。
如果 support 高但 oracle 不升：primitive 仍跨物体，必须强化 split，不要进入 solver。
```

---

## 6. Phase 3：把 masklet 从 birth-surfels-only 改成 atom-level semantic measurements

### 6.1 目标

v13 masklet 的核心缺陷是只传播 birth surfels，不吸收新 surfels，也没有改善 semantic density。v14 的 masklet 不能再成为 object，也不能只复用 birth group。它应该成为 atom-level measurement：

$$
z_m \rightarrow p(a_l \in O_k)
$$

### 6.2 Codex 实现

新增：

```text
Stream3D/stream4d/atom_measurement_bank.py
Stream3D/tools/build_v14_atom_measurements.py
Stream3D/tools/diagnose_v14_measurement_density.py
```

每个 measurement 保存：

```text
measurement_id
frame_id
mask_id or propagated_masklet_id
covered_atom_ids
atom_coverage_scores
visible_outside_atom_ids
negative_scores
boundary_risk
appearance_agreement
d4rt_cycle_score
source = real_2d_mask | d4rt_propagated_masklet | unknown
```

D4RT propagated masklet 只能产生 measurement likelihood，不直接产生 object support。对于没有真实 2D mask 的帧，masklet 可用作弱 measurement，必须记录 source。

### 6.3 实验设置

```text
B0: only real 2D masks on available frames。
B1: B0 + D4RT propagated masklets over birth atoms。
B2: B0 + propagated masklets evaluated over all nearby atoms。
B3: B2 + appearance/boundary consistency gate。
B4: B3 + visible-outside negative measurements。
```

### 6.4 必须记录的指标

```text
measurements/scene
real vs propagated measurement ratio
semantic frames/atom
positive observations/atom
unobserved atom ratio
visible-outside negative ratio
ambiguous atom ratio
measurement mask entropy
measurement atom coverage distribution
agreement with available real masks
cycle error p50/p90 for propagated measurements
appearance drift
```

### 6.5 成功标准

```text
unobserved atom ratio <= 0.35 for broad-support atoms
positive observations/atom >= 2.5
propagated measurement agreement with available masks >= 0.75
visible-outside negative ratio not trivially zero; it must identify ambiguous support
adding propagated measurements must improve atom oracle or object solver, not just count
```

### 6.6 不满足条件时 Codex 先尝试什么

```text
如果 real masks 只有 2/16 frames：先生成 all-frames CropFormer/SAM masks for probe5，作为 measurement density ablation。
如果 propagated measurements disagree with real masks：tighten cycle/appearance/boundary gates，不要导出 AP。
如果 propagated measurements cover only birth atoms：修改 code，让 measurement votes over all visible atoms near propagated mask envelope。
```

---

## 7. Phase 4：真正的全局 object explanation，不再 source-mask slot filtering

### 7.1 目标

实现一个真正的 global object explanation：object 是 atom set，不是 source mask group。solver 要优化“解释 measurements”的全局目标，而不是只对每个 source mask 生成 slot 后排序。

### 7.2 模型定义

设 atoms 为 $A=\{a_l\}$，measurements 为 $Z=\{z_m\}$，object slots 为 $O=\{O_k\}$。每个 object 是 atom subset：

$$
O_k \subseteq A
$$

目标函数：

$$
E(O) = E_{cover}(O,Z) + E_{neg}(O,Z) + E_{split}(O) + E_{merge}(O) + E_{complexity}(O) + E_{unknown}(O)
$$

其中：

```text
E_cover: object 解释 real/propgated measurements 的收益；
E_neg: visible-outside、cannot-link、mask entropy、motion inconsistency 的惩罚；
E_split: 一个 object 内部 trajectory/appearance 多峰的惩罚；
E_merge: 两个 object 解释同一 measurement 或同一 atom 的冲突惩罚；
E_complexity: object 数量和 atom 数的 MDL cost；
E_unknown: 未解释 atoms/measurements 的可控惩罚，允许 unknown，不强制分配。
```

对象选择必须是全局 set packing / correlation clustering / deterministic split-merge，而不是 connected components。

### 7.3 Codex 实现

新增：

```text
Stream3D/stream4d/global_object_explanation.py
Stream3D/tools/export_v14_global_object_explanation.py
Stream3D/tools/diagnose_v14_object_solver.py
```

solver 最小版本：

```text
1. object birth：从 high-confidence atom groups 生成 object seeds。
2. grow：逐步加入使 total energy 降低的 atoms。
3. split：若 object 内 mask_entropy / trajectory_variance 多峰，则 split。
4. merge：若两个 objects 共同解释多个 measurements 且无 strong negative，则 merge。
5. set packing：object 间 atom overlap 只能通过 unknown/competition 解决，禁止简单 WTA 后补救。
6. unknown：ambiguous atoms 留 unknown，不导出。
```

### 7.4 必须记录的内部指标

```text
object_slots_before/after
atoms_assigned_ratio
atoms_unknown_ratio
measurements_explained_ratio
measurements_unexplained_ratio
negative_violation_rate
object_internal_mask_entropy
object_internal_trajectory_variance
object_internal_appearance_variance
split_events
merge_events
grow_events
rejected_objects_by_negative
set_packing_conflicts
duplicate_predictions_per_GT
filtered_good_recovered_count
```

### 7.5 AP 评估

对每个 solver variant 必须输出：

```text
M own
Stream3D on M
M on S0
M on S1
```

至少跑以下 variants：

```text
S0_old: v13 M13c/M13d baseline frozen。
S1_atom_only: atoms + real masks only。
S2_atom_measurement: atoms + B3 atom measurements。
S3_signed: S2 + negative evidence。
S4_global: S3 + split/merge/set packing。
S5_unknown: S4 + explicit unknown/reject export policy。
```

### 7.6 成功标准

v14 probe5 method gate：

```text
S4/S5 own AP >= 0.25
S4/S5 own AP50 >= 0.48
S4/S5 own AP25 >= 0.75
pre% >= 10%
Stream3D on M - M own AP <= 0.08
M on S0 AP >= 0.06
M on S1 AP >= 0.10
M on S1 AP25 >= 0.35
conflict rate <= 0.10
```

这不是最终目标，只是说明方向有真实推进。若达不到，不允许进入 tune30/final。

### 7.7 不满足条件时 Codex 先尝试什么

```text
如果 own AP 高但 Stream3D on M 更高很多：object assignment 仍差，调 solver energy，不要扩大 support。
如果 own AP 高但 S0/S1 近零：coverage 不够，回 Phase 2/3 增加 atom/measurement density。
如果 conflict 高：说明 set packing 无效，加入 atom-level overlap constraints，而不是导出后 WTA。
如果 AP25 高但 AP/AP50 低：boundary/support 需要 posterior export 改进。
如果 all variants 低于 C_surfel tiny result：global solver 没学到 D4RT signal，检查 D4RT/shuffle control。
```

---

## 8. Phase 5：posterior support export，禁止 full-mask shortcut 成为主结果

### 8.1 目标

v13 M13c full-mask ablation 比 posterior support 好，说明 full-mask backprojection 是捷径，但不是 4D object field。v14 需要验证：posterior support 是否能在不退回 full-mask shortcut 的情况下提高 AP/AP50。

### 8.2 导出策略

每个 object 支持分四类：

```text
core atoms: strong positive, low negative, high posterior
fringe atoms: plausible but uncertain, local to core and no strong negative
unknown atoms: insufficient evidence, not exported
reject atoms: negative evidence, not exported
```

导出规则：

```text
core 必导出；
fringe 只有满足 local adjacency / boundary / appearance / distance to core 才导出；
unknown/reject 不导出；
禁止把整张 owned mask 直接 backproject 作为 main result；
full-mask 可以作为 ablation-only。
```

### 8.3 必须记录的指标

```text
core atoms/object
fringe atoms/object
unknown atoms/object
reject atoms/object
core exported points
fringe candidate points
fringe kept points
fringe kept ratio
boundary risk of exported support
mean distance fringe-to-core
export conflict rate
point owner distribution
full-mask ablation delta
posterior-support vs full-mask AP/AP50/AP25
```

### 8.4 成功标准

```text
posterior-support AP50 >= full-mask AP50 - 0.05
posterior-support conflict <= full-mask conflict / 3
posterior-support S0/S1 not worse than full-mask S0/S1
posterior-support pre% >= 8% in successful variant
```

如果 posterior support 远低于 full-mask，说明 posterior 没有学到可靠 boundary；不能用 full-mask 盖过去，必须回 Phase 4 修 object ownership。

---

## 9. Phase 6：D4RT 几何对齐后给 Stream3D 使用的归因实验

### 9.1 目标

回答一个单独问题：D4RT 几何精度对 Stream3D pipeline 的影响有多大。该实验只用于归因，不是主方法，不进入 method table。

### 9.2 严格边界

该实验会使用 GT/RGB-D/ScanNet mesh anchor 做 Sim3，所以必须写：

```text
uses_gt_for_prediction = false
uses_gt_for_diagnostic = true
is_method_result = false
is_diagnostic_only = true
```

它只能报告 diagnostic 表，不得写成 Stream4D method result。方法内部仍不能用 GT Sim3。

### 9.3 实验设置

比较：

```text
G0: 原版 Stream3D RGB-D/pose geometry。
G1: Stream3D + D4RT raw geometry，使用 D4RT 自身尺度超参。
G2: Stream3D + D4RT scene-level GT/RGB-D Sim3 alignment，density-normalized thresholds。
G3: Stream3D + D4RT window-level GT/RGB-D Sim3 alignment，density-normalized thresholds。
G4: Stream3D + D4RT self-aligned windows，然后仅 evaluation Sim3 到 ScanNet。
G5: Stream3D RGB-D geometry + D4RT ownership evidence diagnostic。
```

注意：G2/G3 使用 GT/RGB-D anchor，所以 diagnostic-only。G4 的 self-aligned windows 不用 GT，可以作为方法内部几何 sanity，但 final AP export 到 ScanNet 仍需要 evaluation adapter。

### 9.4 必须记录的指标

```text
D4RT point spacing q10/q50/q90
Sim3 residual mean/median/p90/p95
scale mean/min/max
NN radius in D4RT units and ScanNet units
projection hit rate
num exported objects
num exported points
conflict rate
Stream3D AP/AP50/AP25 under diagnostic setting
drop from G0 in AP/AP50/AP25
failure reason: geometry miss / threshold miss / manifold threshold / export alignment
```

### 9.5 成功标准

```text
如果 G2/G3 AP drop <= 3 AP and AP50 drop <= 5：D4RT metric geometry after evaluation alignment is not the main bottleneck。
如果 G2/G3 AP drop > 8 AP or AP50 drop > 10：D4RT metric geometry is a major bottleneck for Stream3D-style pipeline。
如果 G2/G3 own AP 非零但 S0/S1 仍 near zero：geometry can generate tiny support but not full segmentation。
```

### 9.6 不满足条件时 Codex 先尝试什么

```text
如果 0 objects：不要复用 RGB-D meter-scale thresholds；用 D4RT point spacing q50/q90 设 NN radius 和 min_points。
如果 residual 高：检查 frame index / coordinate convention / tcam reference / Sim3 anchors。
如果 conflict 高：先做 geometry-only conflict visualization，不要调 segmentation thresholds。
```

---

## 10. Phase 7：D4RT 贡献控制实验

### 10.1 目标

确认 v14 的提升是否来自 D4RT material correspondence，而不是 mask面积、support缩小、随机选择或旧结果融合。

### 10.2 对照实验

对最好的 Phase 4 方法，跑：

```text
D4RT-real: 正常 D4RT surfels/atoms。
D4RT-shuffle: 打乱 surfel-to-mask / atom-to-measurement 对应。
No-temporal: 只用 source frame，不用 target-frame trajectory。
RGB-only: 使用 RGB/appearance split，但不使用 D4RT trajectory。
Mask-area: 按 mask area 选择同数量 candidates。
Random-same-count: 随机选择同数量 candidates，固定 seed。
Stream3D-on-method-support: 同 support baseline。
```

### 10.3 必须记录的指标

```text
AP/AP50/AP25
same-support gap
negative evidence violation
split/merge events
atoms assigned ratio
object count
support ratio
conflict rate
oracle gap
```

### 10.4 成功标准

```text
D4RT-real AP50 >= shuffle AP50 + 0.10
D4RT-real AP >= no-temporal AP + 0.05
D4RT-real reduces duplicate_predictions_per_GT
D4RT-real improves split/merge correctness diagnostics
```

如果真实 D4RT 只在 tiny support 上有用，但不改善 S0/S1 或 same-support gap，则不能把 D4RT 作为 semantic 4D reconstruction 主贡献，只能说它是局部 ownership cue。

---

## 11. Phase 8：Tune30 / final / Dynamic Replica 启动条件

### 11.1 Probe5 gate 通过前禁止启动 full final

只有满足 Phase 4 gate，才允许进入 tune30：

```text
M on S0 AP >= 0.06
M on S1 AP >= 0.10
Stream3D on M - M own AP <= 0.08
pre% >= 10%
conflict <= 0.10
```

Tune30 上只能调一次小范围：

```text
object model cost
negative weight
fringe threshold
unknown penalty
atom merge threshold
```

锁定后 final split 只跑一次。

### 11.2 Dynamic Replica 启动条件

Dynamic Replica 只有在以下条件满足时才能报告 official tracking：

```text
usable_scene_count > 0
instance/object ID GT exists
frame-level object correspondence exists
can_report_official_instance_tracking = true
```

否则只允许报告 qualitative 或 pseudo-consistency，并必须写：

```text
official tracking metrics not available
```

动态指标若可用，记录：

```text
IDF1
ID switches
fragmentation
MOTA/HOTA if supported
4D IoU over time
occlusion reactivation precision/recall
trajectory consistency
text-query object retrieval if semantic labels available
```

---

## 12. Codex 每轮必须提交的最小交付

每轮必须提交：

```text
1. 完整 code audit packet zip + sha256 + filelist + ziptest.log。
2. py_compile log。
3. pure-python unittest log。
4. reportable config scan。
5. metric integrity report。
6. unified eval matrix json/csv/md。
7. failure decomposition json/csv/md。
8. oracle upper-bound diagnostic json/csv/md，若有。
9. method manifests。
10. 每个 method 的 own / Stream3D-on-M / M-on-S0 / M-on-S1 四行结果。
11. 至少 10 张 failure visualization panels。
12. 执行日志与结果复盘。
```

任何没有统一评估矩阵的 AP 结果不得作为决策依据。

---

## 13. v14 最关键的判断标准

v14 不是为了立刻在 full ScanNet 超过 Stream3D，而是为了判断是否找到了正确算法方向。

### 13.1 方向成立的最低条件

```text
1. Atom broad-support oracle 比 C_hybrid oracle 明显提升。
2. Global object explanation 比 v13 M13c/M13d 在 same-support gap 上明显缩小。
3. 方法不再只依赖 pre% < 5% 的 tiny support。
4. D4RT-real 明显优于 shuffle/no-temporal controls。
5. S0/S1 不再接近零。
```

### 13.2 方向失败的判据

```text
1. Atom oracle 仍然低，说明 primitive 仍错。
2. Oracle 高但 method 低，说明 global solver 仍错。
3. D4RT-real 与 shuffle/no-temporal 接近，说明 D4RT 没被正确用上。
4. posterior support 远低于 full-mask ablation，说明 boundary/ownership posterior 不可信。
5. own-support 高但 Stream3D-on-M 更高，说明 object quality 仍不如 Stream3D。
```

如果失败，下一步不能继续调阈值，必须根据 failure category 选择：

```text
primitive failure -> 重做 atom/measurement source；
solver failure -> 重写 object explanation objective；
D4RT contribution failure -> 重新设计 trajectory/identity factor；
support failure -> 增加 real semantic measurement frames 或更强 2D mask source；
geometry failure -> 保持 D4RT geometry diagnostic-only，不作为 ScanNet 主路径。
```

---

## 14. 预期论文级创新点对齐

v14 要推进的顶会级创新不是“更好的 mask filter”，而是：

```text
Training-free object explanation over feed-forward D4RT material fields。
```

相比 Stream3D 的 2D-to-3D mask merging，v14 的研究核心应是：

```text
1. D4RT material surfels 作为 4D field primitive。
2. 2D masks / masklets / regionlets 作为 noisy semantic measurements。
3. Object 是 latent explanation，而不是 mask、regionlet 或 component。
4. Signed evidence：positive + visible-outside negative + cannot-link + motion/appearance/boundary consistency。
5. Unknown/reject 是合法输出，不强行 hallucinate object support。
6. Tracking by construction：object 是 surfel/atom partition，时间上的可见 support 由 D4RT trajectory 给出。
```

只有当 v14 的 atom/oracle/solver 结果显示这条 formulation 真正缩小 same-support gap，并在动态 benchmark 上能展示 identity / occlusion / reactivation 优势，才值得进入论文主线。
