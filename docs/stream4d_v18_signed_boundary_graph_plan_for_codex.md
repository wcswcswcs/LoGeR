# Stream4D v18：Signed Boundary Evidence Graph 实验计划书

面向 Codex 的可执行实验计划。本文档取代 v17 的 `non_gt_object_explanation_solver` 主线。v17 的核心教训是：**从 candidate pool 里做 non-GT union / growth / packing 仍然是在做候选物体选择，而不是在做物体边界推断**。这条路没有把 v16 的 oracle upper bound 转成方法性能，甚至 `M17-real` 低于 random same-count。v18 的目标是把问题重新定义为：

> 在 D4RT material surfels 上构建 signed boundary evidence graph，先判断 surfel 之间哪里该切、哪里该连，再由 surface partition 得到 object。

本文档公式均使用 Typora 友好的 `$...$` 或 `$$...$$`，不使用 display 公式的方括号语法。

---

## 0. 当前结论和 v18 的根本转向

v17 已经证明以下方向不能继续作为主线：

```text
C_hybrid candidate union solver
candidate-level oracle-selected feature AUC -> object solver
candidate growth / packing / support floor
把 C_regionlet / C_surfel tiny anchors 直接加回 broad output
继续调 K、min_region_size、w_conflict、strict packing
```

原因不是这些代码写得不够细，而是**问题层级错了**。C_hybrid oracle 知道每个 candidate 服务哪个 GT object，因此能把多个 measurements union 成一个好 object；非 GT solver 没有 object identity，虽然能选中一些低 conflict、低 boundary-risk 的局部候选，但无法知道这些候选应该和谁合并、应该在哪里切开。v17 的 `M17-real` 结果只有：

```text
M17-real own = 0.016074 / 0.042181 / 0.227249
M17-real S1  = 0.006206 / 0.022883 / 0.215747
```

而 `M17-random-same-count own = 0.073999 / 0.168353 / 0.302491`。这说明 v17 solver 不是“略弱”，而是**组合方式错了**。

v17 最强 repair 是：

```text
repair_cmask own = 0.101653 / 0.248464 / 0.494844
repair_cmask S1  = 0.102883 / 0.242779 / 0.576250
pre% = 60.8353
```

但同一个 support 上 Stream3D 仍明显更强：

```text
Stream3D on repair_cmask support = 0.224924 / 0.401511 / 0.577226
```

这说明 support 里有可用信息，问题在于 Stream4D 不会把这些 surface / mask observations 切成正确 objects。

v18 的根本转向是：

```text
不要再从 object candidates 里选 object。
要在 D4RT surfel graph 上做 signed boundary partition。
```

过去错误路线：

```text
2D mask / regionlet / surfel atom / C_hybrid candidate -> object candidate -> score / WTA / memory
```

v18 正确路线：

```text
D4RT material surfels -> surfel adjacency graph -> signed boundary evidence -> graph partition -> object support
```

一个 2D mask 不再被解释为 object，而是解释为对 surfel adjacency edges 的局部观测：

```text
same-inside + boundary-safe -> weak merge evidence
inside/outside + stable mask boundary -> strong cut evidence
near-boundary / low-confidence / invisible -> uncertain evidence
```

这件事如果成立，才是顶会级别的 formulation：**从 mask merging 改成 material-surface boundary inference**。

---

## 1. 全局硬约束

### 1.1 评估统一协议

每个 reportable method config 都必须输出四行统一评估：

```text
M own:
  prediction = M
  pre_points = M

Stream3D on M:
  prediction = Stream3D baseline
  pre_points = M

M on S0:
  prediction = M
  pre_points = Stream3D original ScanNet support

M on S1:
  prediction = M
  pre_points = historical 32f sparse support
```

如果某个方法是 parent config 的后处理，还必须输出：

```text
M inherit parent:
  prediction = M
  pre_points = parent
```

每行必须记录：

```text
AP
AP50
AP25
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

禁止只报 own/recompute 高分。

### 1.2 GT/Sim3 对齐约束

方法内部不能使用 GT instance、GT semantic、GT mesh labels 或 GT-aligned Sim3 来决定 object selection、edge weights、graph partition、score、memory 或 support export。

允许：

```text
1. 评估 / 测试 / diagnostic 时使用 GT labels 计算 AP、oracle、edge-AUC、failure attribution。
2. D4RT geometry diagnostic 中，用 ScanNet RGB-D / pose / mesh anchors 做 Sim3，对齐后只用于测试几何误差或 Stream3D-D4RT geometry attribution。
3. D4RT 自己和自己对齐，例如 window-to-window D4RT self-alignment、cycle consistency、track stitching；这不是 GT 对齐，但必须在 manifest 中写 alignment_source=d4rt_self_only。
```

禁止：

```text
1. 使用 GT / RGB-D Sim3 参与 v18 edge score。
2. 使用 GT / RGB-D Sim3 参与 v18 graph partition。
3. 根据 AP 选择 Sim3 variant 后再作为 method result。
4. 把 oracle-selected outputs 标为 method result。
```

所有 diagnostic oracle 必须写：

```text
uses_gt_for_prediction = true 或 uses_gt_for_diagnostic = true
is_diagnostic_only = true
forbidden_for_method_table = true
```

### 1.3 每轮代码审计包

Codex 每轮必须提交：

```text
stream4d_v18_<phase>_code_review_packet.zip
stream4d_v18_<phase>_code_review_packet.sha256
stream4d_v18_<phase>_filelist.txt
stream4d_v18_<phase>_ziptest.log
stream4d_v18_<phase>_git_diff.patch
stream4d_v18_<phase>_git_status.txt
```

zip 内必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/*.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_v18_*.sh
Stream3D/scripts/*.json
Stream3D/docs/stream4d_v18_执行日志.md
Stream3D/docs/stream4d_v18_实验结果复盘.md
Stream3D/outputs/audit/v18_*/*.json
Stream3D/outputs/audit/v18_*/*.md
Stream3D/outputs/audit/v18_*/*.csv
Stream3D/data/evaluation/scannet/*_class_agnostic.txt for v18 configs
Stream3D/data/prediction/<v18_configs>_class_agnostic/config_manifest.json
Stream3D/data/TMP/<v18_configs>/config_manifest.json
```

必须通过：

```bash
python -m py_compile evaluation/*.py stream4d/*.py tools/*.py tests/*.py
python -m unittest discover tests
python -m tools.scan_reportable_configs --require-manifest --require-eval-policy
python -m tools.verify_stream4d_metric_integrity --require-manifest
```

如果 optional dependency 如 `open3d` 缺失，不允许影响 pure protocol tests。必须拆成：

```text
pure-python tests
open3d-required tests
gpu-required tests
```

---

## 2. v18 总体目标

v18 不直接承诺超过 Stream3D。v18 的目标是建立一个明确可判定的最小科学闭环：

```text
D4RT surfel graph 是否足以表达 object boundary？
2D masks / depth / appearance 是否能提供有效 signed boundary evidence？
signed boundary graph partition 是否比 candidate union / mask-as-object 明显更接近 Stream3D？
```

v18 的最低成功不是 full ScanNet，而是在 probe5 上证明：

```text
1. edge-level boundary evidence 有可测预测力；
2. graph partition 不是 tiny-support shortcut；
3. method on S0/S1 明显超过 O38 / repair_cmask / carrier-component baselines；
4. real D4RT 明显优于 shuffle / no-temporal controls；
5. 与 Stream3D same-support gap 缩小，而不是只在 own support 漂亮。
```

v18 的算法假设如下：

$H_1$：当前失败的主因是 object boundary / surface partition 错，不是 measurement density 不足。v17 已修正 measurement stats，target positive observation count 很高，因此不能继续把失败归因于 mask frame 不够。

$H_2$：一个 2D mask 的最可靠作用不是生成 object candidate，而是给 surfel adjacency edge 产生 signed evidence。weak positive 不可传递，strong negative 应优先。

$H_3$：D4RT 的价值不在于直接替代 RGB-D geometry，也不在于 candidate scoring，而在于提供 material surfel identity、temporal adjacency 和跨帧 boundary evidence aggregation。

$H_4$：如果 edge evidence 能较好区分 GT cut/merge edges，但 partition AP 仍低，则问题是 graph solver / materialization；如果 edge evidence 本身不能区分 GT boundary，则问题是 evidence source 不足，需要换 mask source、加入 depth/normal/appearance，或重建 surfel adjacency。

---

## 3. Phase 0：冻结基线和统一差距表

### 3.1 目标

建立 v18 的不可变基线表，避免再次被 own-support 高分误导。所有后续实验都必须和该表比较。

### 3.2 输入

固定以下 prediction/support：

```text
S0 = Stream3D original ScanNet support
S1 = historical 32f sparse support
B1 = v8/v9 B1 surfacelet singlemask
O38 = v9/O38 broad memory c055
repair_cmask = v17 best broad repair
P0 = Stream3D baseline prediction
P_v6compact = v6 compact baseline
```

### 3.3 必跑表格

输出：

```text
outputs/audit/v18_phase0/unified_eval_matrix_probe5.json
outputs/audit/v18_phase0/unified_eval_matrix_probe5.csv
outputs/audit/v18_phase0/unified_eval_matrix_probe5.md
outputs/audit/v18_phase0/gap_matrix_heatmap_AP.png
outputs/audit/v18_phase0/gap_matrix_heatmap_AP50.png
outputs/audit/v18_phase0/support_ratio_bar.png
outputs/audit/v18_phase0/method_vs_stream3d_same_support_delta.png
```

至少包含：

```text
P0 on S0
P0 on S1
P0 on B1 support
P0 on O38 support
P0 on repair_cmask support
B1 own / B1 on S0 / B1 on S1
O38 own / O38 on S0 / O38 on S1
repair_cmask own / repair_cmask on S0 / repair_cmask on S1
P_v6compact on S1
```

### 3.4 记录指标

必须记录第 1.1 节全部统一指标，并额外记录：

```text
same_support_gap_to_stream3d_AP
same_support_gap_to_stream3d_AP50
same_support_gap_to_stream3d_AP25
```

### 3.5 判断标准

Phase 0 不追求方法性能，只要求审计通过：

```text
AP core hash unchanged
all configs have manifest
no method config has uses_gt_for_prediction=true
all support source and eval policy fields complete
```

不满足时：

```text
先修 evaluator / manifest / pre_points materialization，不允许继续 Phase 1。
```

---

## 4. Phase 1：构建 D4RT semi-dense surfel adjacency graph

### 4.1 目标

构建 object boundary inference 的基本图。节点是 D4RT material surfel，边是局部邻接关系。这个阶段不输出 object AP，只验证 graph 是否有足够覆盖和连通性。

### 4.2 核心思想

每个 surfel：

$$
s_i = \{\pi_i(t), X_i(t), v_i(t), c_i(t), a_i(t)\}_{t=1}^{T}
$$

其中：

```text
pi_i(t): D4RT prediction projected UV
X_i(t): D4RT 3D coordinate, only used for non-GT/self-consistency or eval bridge diagnostics
a_i(t): local RGB / DINO / CLIP patch feature if available
v_i(t), c_i(t): visibility and confidence
```

构图不是为了生成 candidate，而是为了回答：

```text
edge(i,j) 是否跨过 object boundary？
```

### 4.3 实现文件

新增或重写：

```text
Stream3D/stream4d/signed_surfel_graph.py
Stream3D/tools/build_v18_signed_surfel_graph.py
Stream3D/tools/diagnose_v18_surfel_graph.py
Stream3D/tests/test_v18_signed_surfel_graph.py
```

### 4.4 图节点

基于 v10/v12/v17 fresh D4RT grid surfels：

```text
default grid = 32 x 32 per source frame/window
clip = continuous 16 frames, stride1
use only uv_in_range and visibility/confidence above threshold for edge evidence
retain all surfels as nodes even if not currently assigned to mask, but mark observed/unobserved
```

必须记录：

```text
num_surfels
num_visible_surfels_per_frame
track_length_visible_mean / p10 / p90
uv_in01_rate
self_uv_error_p90
cycle_uv_error_p90
surfel_coverage_2d_per_frame
unobserved_surfel_ratio
ambiguous_surfel_ratio
```

### 4.5 图边

构建多种边，且每种边必须单独记录，不要一开始混成一个分数：

```text
E_2d_knn:
  同一 target frame 中，UV 近邻 kNN edge。

E_2d_grid:
  同一 source grid 中水平/垂直/对角邻接 edge。

E_depth_neighbor:
  RGB-D eval bridge 中 3D 近邻 edge，只用于 ScanNet 静态诊断；不能用 GT label。

E_temporal_same_material:
  同一个 D4RT surfel 跨时间的自边，不用于 partition cut，只用于 tracking consistency。

E_cross_frame_consistency:
  两个 surfels 在多帧中持续近邻且 visibility overlap 高，形成 temporal adjacency。
```

每条 edge 记录：

```text
src_node
dst_node
edge_type
num_visible_together
mean_uv_distance
median_uv_distance
mean_rgb_distance
mean_feature_distance
mean_depth_difference if RGB-D available
mean_normal_difference if computed
trajectory_relative_motion_variance
```

### 4.6 成功标准

probe5 上必须满足：

```text
num_nodes_mean >= 10k
visible_track_length_mean >= 10
uv_in01_rate >= 0.95
cycle_uv_error_p90 <= 5 px
E_2d_knn edge count >= 8 * num_nodes_visible_mean
largest graph component ratio between 0.3 and 0.95
unobserved_surfel_ratio <= 0.05 using target positive observations
```

如果失败：

```text
uv/cycle 失败：检查 D4RT clip 是否连续帧；检查 frame index、normalized UV、margin ratio、query order。
node 覆盖太低：提高 grid density到48，或增加多 source frames；但记录 runtime。
graph 过碎：增加 E_cross_frame_consistency / kNN k。
graph 巨大粘连：降低 kNN k，加入 depth/normal discontinuity pre-cut，不要直接进入 partition。
```

---

## 5. Phase 2：Edge boundary oracle 上界诊断

### 5.1 目标

判断 surfel graph 本身是否能表达 object boundary。这是 GT-only diagnostic，不是方法。

如果 graph oracle 不行，不要继续写 edge solver。

### 5.2 GT edge label

对每条 edge $(i,j)$，在 evaluation/diagnostic 中通过 ScanNet GT instance labels 给出：

```text
same_gt: i 和 j 的 mesh/materialized labels 属于同一 GT instance
cut_gt: i 和 j 属于不同 GT instances
unknown_gt: 任一节点未能可靠映射到 GT / unlabeled / ignored class
```

注意：GT labels 只用于诊断，不允许写进 method artifacts。

### 5.3 诊断一：surfel graph GT coverage

记录：

```text
node_gt_label_coverage
edge_gt_label_coverage
per_GT_num_surfels
per_GT_visible_frame_count
per_GT_internal_edge_count
per_GT_boundary_edge_count
GT instances with >= 20 surfels
GT instances with >= 100 surfels
```

判断标准：

```text
valid GT instances with >=20 surfels >= 60% of GT crop
valid GT instances with >=100 surfels >= 35% of GT crop
node_gt_label_coverage >= 0.70
edge_gt_label_coverage >= 0.60
```

如果失败：

```text
优先修 surfel-to-mesh materialization / sampling density，而不是 solver。
尝试 D4RT grid48 或 multi-source frame union。
检查 D4RT UV->RGB-D point mapping。
```

### 5.4 诊断二：GT edge partition oracle

用 GT edge labels 产生 oracle partition：

```text
删除所有 cut_gt edges。
在 same_gt edges 上求 connected components。
忽略 unknown edges 或低权重处理。
```

导出 oracle prediction，并标为：

```text
uses_gt_for_prediction = true
is_diagnostic_only = true
forbidden_for_method_table = true
```

记录：

```text
Oracle own AP/AP50/AP25
Oracle on S0/S1 diagnostic
pre%
union%
GT crop/full
#components
component purity
component completeness
component oversegmentation per GT
component undersegmentation count
```

### 5.5 判断标准

最小 gate：

```text
edge oracle own AP >= 0.25
edge oracle AP50 >= 0.50
edge oracle AP25 >= 0.70
pre% >= 20
GT crop/full >= 0.50
```

强 gate：

```text
edge oracle own AP >= 0.35
edge oracle AP50 >= 0.60
edge oracle AP25 >= 0.78
pre% >= 35
```

如果 oracle 不过最小 gate：

```text
不要启动非 GT edge solver。
先修 graph coverage、surfel materialization、edge construction。
检查失败 GT 类别：小物体、细长物体、接触物体、墙/地/大平面。
```

如果 oracle 过最小 gate但 method 后面失败：

```text
说明表示空间有上界，问题在 edge evidence 或 graph partition。
```

---

## 6. Phase 3：Non-GT signed boundary evidence 质量评估

### 6.1 目标

在不使用 GT 的情况下，构建每条 edge 的 signed evidence，并用 GT 只做 diagnostic 评估该 evidence 是否能区分 cut/merge edges。

这是 v18 的核心科学问题：

```text
2D mask、D4RT temporal consistency、RGB/depth/normal discontinuity 能否在 edge level 上提供 object boundary 信号？
```

### 6.2 Edge evidence 定义

对每条 edge $(i,j)$ 和 frame $t$，令两个 surfel 的投影为 $
\pi_i(t)$ 和 $\pi_j(t)$。对每个 2D mask $m_t$ 计算：

```text
inside_i = pi_i(t) inside mask
inside_j = pi_j(t) inside mask
safe_i = distance(pi_i(t), mask boundary) > tau_boundary
safe_j = distance(pi_j(t), mask boundary) > tau_boundary
visible_i, visible_j
```

单帧 evidence：

```text
both inside and safe -> merge vote
one inside one outside and boundary between them -> cut vote
both outside -> weak none
near boundary -> uncertain
not both visible -> ignore
```

定义聚合权重：

$$
w_{ij}^{merge} = \sum_t \alpha_t \cdot \mathbf{1}[\text{same-inside-safe}_{ij,t}]
$$

$$
w_{ij}^{cut} = \sum_t \beta_t \cdot \mathbf{1}[\text{inside-outside-boundary}_{ij,t}]
$$

其中 $\alpha_t, \beta_t$ 来自：

```text
D4RT visibility/confidence
cycle consistency
mask stability
boundary distance
RGB/depth/normal discontinuity
```

最终 signed score：

$$
r_{ij} = w_{ij}^{merge} - \lambda w_{ij}^{cut}
$$

也可以记录 cut probability：

$$
p_{ij}^{cut} = \sigma(\lambda_c w_{ij}^{cut} - \lambda_m w_{ij}^{merge} + b)
$$

### 6.3 实现文件

新增：

```text
Stream3D/stream4d/signed_boundary_evidence.py
Stream3D/tools/build_v18_signed_boundary_evidence.py
Stream3D/tools/diagnose_v18_edge_boundary_quality.py
Stream3D/tests/test_v18_signed_boundary_evidence.py
```

### 6.4 实验变体

必须比较以下 non-GT evidence configs：

```text
E0 mask_co_membership_baseline:
  只用 same-mask co-membership，复现旧错误假设。

E1 mask_signed:
  使用 same-inside merge + inside/outside cut。

E2 mask_signed_boundary_safe:
  E1 + boundary distance weighting。

E3 mask_signed_depth_normal:
  E2 + RGB-D depth/normal discontinuity。

E4 mask_signed_d4rt_temporal:
  E2 + D4RT temporal visibility/cycle weighting。

E5 full_signed:
  E2 + depth/normal + D4RT temporal + appearance feature。

E6 shuffle_d4rt:
  打乱 D4RT temporal identity，保留 mask evidence。

E7 no_temporal:
  只用单帧 evidence，不跨帧累积。
```

### 6.5 记录指标

edge-level diagnostic：

```text
edge_cut_AUC
edge_cut_AP
precision_at_top_1_percent_cut_edges
precision_at_top_5_percent_cut_edges
precision_at_top_10_percent_cut_edges
GT_boundary_recall_at_top_10_percent
false_cut_rate_inside_same_GT
false_merge_rate_across_GT
mean_cut_score_same_GT
mean_cut_score_different_GT
score_separation_margin
num_edges_evaluated
unknown_edge_ratio
```

scene-level diagnostic：

```text
edge AUC per scene
worst scene edge AUC
edge score histogram same/different GT
boundary overlay images
```

### 6.6 判断标准

最小 gate：

```text
E5 edge_cut_AUC >= 0.70
E5 edge_cut_AP >= 0.35
precision_at_top_10_percent_cut_edges >= 0.55
false_cut_rate_inside_same_GT <= 0.25
E5 AUC - E0 AUC >= 0.08
E5 AUC - E6 shuffle_d4rt AUC >= 0.03
```

强 gate：

```text
edge_cut_AUC >= 0.78
precision_at_top_10_percent_cut_edges >= 0.70
false_cut_rate_inside_same_GT <= 0.15
```

如果 E5 不过 gate：

```text
如果 E0/E1/E5 都低：2D mask boundary 本身不足，尝试 SAM2/Entity/CropFormer ensemble 或提高 mask frame density。
如果 E1 高但 E5 低：D4RT/depth/appearance weighting 加错，先回退到 mask_signed。
如果 E5 高但 shuffle_d4rt 同样高：D4RT temporal 没贡献，主创新不能写 D4RT tracking advantage；改写为 mask-boundary graph baseline。
如果 false_cut inside same GT 高：boundary threshold 太激进；增加 inside-safe margin和uncertain ignore。
如果 false_merge across GT 高：cut evidence 太弱；加入 depth/normal/color discontinuity和same-frame mutual exclusivity。
```

---

## 7. Phase 4：Signed graph partition method v1

### 7.1 目标

使用 Phase 3 的 non-GT edge weights 进行 graph partition，得到 object components。这个阶段是第一版真正 method result。

### 7.2 Energy formulation

给每个 surfel node 分配 object label $y_i$。优化：

$$
E(Y) = \sum_{(i,j) \in E} w_{ij}^{cut}\mathbf{1}[y_i=y_j] + w_{ij}^{merge}\mathbf{1}[y_i\neq y_j] + \lambda_K K(Y) + \lambda_S S(Y)
$$

其中：

```text
w_cut: 强 cut evidence，同一个 label 会被惩罚
w_merge: 强 merge evidence，不同 label 会被惩罚
K(Y): object 数量复杂度
S(Y): 小碎片 / 巨大 component penalty
```

不要求一开始求全局最优。先实现三个近似：

```text
P1 signed watershed:
  将 cut probability 当边界高度，在 graph 上做 watershed。

P2 agglomerative signed clustering:
  从小 components 开始，按 merge_gain 合并；遇到强 cut edge 停止。

P3 seeded graph partition:
  用 B1 / C_regionlet high-confidence seeds 初始化 object cores，再让 unlabeled surfels 按 lowest cut-cost attach。
```

### 7.3 实现文件

新增：

```text
Stream3D/stream4d/signed_graph_partition.py
Stream3D/tools/export_v18_signed_graph_partition.py
Stream3D/tools/diagnose_v18_partition_quality.py
Stream3D/tests/test_v18_signed_graph_partition.py
```

### 7.4 Export modes

必须同时导出两个版本，避免把 materialization 和 partition 混在一起：

```text
G_core:
  只导出 component 中高置信 surfel hits 映射到 mesh 的点。
  目的：诊断 partition purity。

G_region_fill:
  对每个 object component，在 owned 2D mask region 内填充 mesh support。
  填充条件：mask 内区域和 component seeds 有低 cut-cost path，且不跨强 boundary。
  目的：恢复 support completeness。
```

禁止直接 full-mask backproject。

### 7.5 记录指标

统一评估四行：

```text
M own
Stream3D on M
M on S0
M on S1
M inherit parent if any
```

partition diagnostic：

```text
#components
mean component size
median component size
largest component ratio
tiny component ratio
component purity diagnostic using GT
component completeness diagnostic using GT
oversegmentation per GT
undersegmentation count
edge disagreement rate after partition
mean cut score across object boundary
mean merge score within object
```

AP diagnostic：

```text
AP/AP50/AP25
pre%
union%
GT crop/full
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
duplicate predictions per GT
missed GT count
```

### 7.6 判断标准

最小 gate，probe5：

```text
G_core own AP >= 0.08
G_core AP50 >= 0.18
G_core AP25 >= 0.45
G_core pre% >= 10

G_region_fill own AP >= 0.12
G_region_fill AP50 >= 0.28
G_region_fill AP25 >= 0.55
G_region_fill pre% >= 20

G_region_fill S1 AP >= 0.12
G_region_fill S1 AP50 >= 0.26
G_region_fill S1 AP25 >= 0.55
```

相对基线 gate：

```text
G_region_fill S1 AP > repair_cmask S1 AP by >= 0.03
G_region_fill S1 AP50 > repair_cmask S1 AP50 by >= 0.04
G_region_fill same-support gap to Stream3D reduced by >= 20% relative to repair_cmask
```

强 gate：

```text
G_region_fill S1 AP >= 0.18
G_region_fill S1 AP50 >= 0.35
G_region_fill S1 AP25 >= 0.60
```

如果失败：

```text
edge AUC high but AP low:
  partition solver/materialization failed；检查 component over/under segmentation。
edge AUC low and AP low:
  evidence source insufficient；回到 Phase 3。
G_core high purity but low pre%:
  materialization / region fill 是瓶颈；进入 Phase 5。
G_region_fill AP25 high but AP/AP50 low:
  coarse coverage 有效但边界不准；加强 boundary-safe fill 和 component splitting。
real D4RT 不优于 shuffle/no_temporal:
  D4RT 未成为有效贡献；不要 claim D4RT-native advantage。
```

---

## 8. Phase 5：Mask-internal decomposition and posterior materialization

### 8.1 目标

如果 Phase 4 的 G_core purity 有信号但 support 太小，进入 Phase 5。目标是解决：

```text
从 surfel component 到 mesh/object support 的 materialization 断裂。
```

核心不是 full-mask backproject，而是把每个 2D mask 内部按 graph partition 分解成 object-owned regions。

### 8.2 方法

对于每个 frame 的每个 2D mask，构建 mask 内的 pixel/surfel region graph：

```text
pixels or mesh vertices inside mask = candidates
surfel components project into mask = seeds
mask boundary / RGB edge / depth edge = local cut costs
D4RT signed graph component labels = ownership priors
```

对 mask 内每个 pixel/mesh point $p$，计算它属于 object $k$ 的 cost：

$$
C(p,k) = d_{geo}(p, O_k) + \lambda_b B(p,k) + \lambda_a A(p,k) + \lambda_e E_{cut}(p,k)
$$

只分配满足以下条件的点：

```text
cost below threshold
not crossing strong boundary
not closer to another object by large margin
boundary_safe or component-supported
```

输出每个 mask 的：

```text
owned region for object k
unknown region
reject region
```

### 8.3 实现文件

新增：

```text
Stream3D/stream4d/mask_internal_decomposition.py
Stream3D/tools/export_v18_mask_internal_decomposition.py
Stream3D/tools/diagnose_v18_materialization.py
Stream3D/tests/test_v18_mask_internal_decomposition.py
```

### 8.4 记录指标

per-mask：

```text
num_objects_touching_mask
owned_pixels_ratio
unknown_pixels_ratio
reject_pixels_ratio
boundary_crossing_rate
seed_to_owned_region_distance
owned_region_count
```

per-object：

```text
core_surfel_count
owned_region_area
fringe_area
unknown_area_near_object
contamination_proxy
support_completeness_proxy
```

GT diagnostic：

```text
owned_region_purity
owned_region_completeness
boundary_precision
boundary_recall
per-GT best IoU before/after fill
```

AP：统一四行评估。

### 8.5 判断标准

Phase 5 最小 gate：

```text
G_region_fill AP50 >= G_core AP50 + 0.08
G_region_fill AP25 >= G_core AP25 + 0.08
pre% increases by >= 8 points
GT diagnostic owned_region_purity >= 0.70
unknown_pixels_ratio >= 0.10, 说明没有强行全塞
```

如果失败：

```text
purity低：fill 过激，增加 boundary margin，减少 fringe。
completeness低：seeds 太少，增加 high-confidence regionlet/B1 birth seeds。
AP25升但AP50不升：区域太粗，加入 depth/normal/color boundary。
S0/S1仍崩：support仍太小或 object coverage不足；进入 failure decomposition。
```

---

## 9. Phase 6：D4RT contribution controls

### 9.1 目标

证明 v18 的提升来自 D4RT material correspondence，而不是纯 2D mask boundary / RGB-D depth。

### 9.2 必跑 controls

对 Phase 3/4/5 的 best config 跑：

```text
real_d4rt
shuffle_d4rt_identity
no_temporal_single_frame
mask_only_no_d4rt
depth_normal_only
random_surfel_graph_same_degree
```

### 9.3 指标

```text
edge_cut_AUC
edge_cut_AP
partition own AP/AP50/AP25
partition S1 AP/AP50/AP25
#components
pre%
Stream3D on method support
```

### 9.4 判断标准

最小 D4RT contribution：

```text
real_d4rt edge_cut_AUC - shuffle_d4rt edge_cut_AUC >= 0.03
real_d4rt S1 AP50 - shuffle_d4rt S1 AP50 >= 0.04
real_d4rt S1 AP25 - no_temporal S1 AP25 >= 0.04
```

强 contribution：

```text
real_d4rt S1 AP50 - shuffle >= 0.08
real_d4rt reduces ID fragmentation in dynamic diagnostic if data available
```

如果不满足：

```text
不能写 D4RT-native advantage。
可以保留 signed boundary graph 作为 2D/RGB-D boundary method，但需要重新定位论文贡献。
若 D4RT 只改善 edge AUC 不改善 AP，说明 partition/materialization 没吃到 temporal signal。
```

---

## 10. Phase 7：D4RT geometry aligned -> Stream3D diagnostic

### 10.1 目标

继续保留几何归因实验，用于回答：

```text
D4RT metric geometry 经过 GT/RGB-D anchors Sim3 对齐后，作为 Stream3D 几何源会掉多少？
```

这是 diagnostic-only，不进入 v18 method table。

### 10.2 变体

```text
G0: original Stream3D RGB-D/pose geometry
G1: Stream3D + D4RT raw geometry, no scale adaptation
G2: Stream3D + D4RT scene-level Sim3 aligned geometry
G3: Stream3D + D4RT window-level Sim3 aligned geometry
G4: G2 + density-normalized thresholds
G5: G3 + density-normalized thresholds
G6: RGB-D metric geometry + D4RT image-space correspondence evidence only
```

### 10.3 严格约束

```text
GT/RGB-D Sim3 只在 geometry diagnostic 中使用。
不得将 G2-G5 的 Sim3 或 residual 反馈到 v18 signed graph method。
不得按 AP 选择 Sim3 后写 method。
```

### 10.4 指标

几何：

```text
anchor_count
Sim3 scale
inlier ratio
median residual
p90 residual
p95 residual
uv_in01_rate
cycle_uv_error_p90
D4RT point spacing q25/q50/q75/q90
mask projection hit rate
empty_projected_mask_ratio
```

Stream3D pipeline：

```text
AP/AP50/AP25 own
Stream3D on G_i support
G_i on S0
G_i on S1
pre%
#pred
set-cover selected masks
manifold refine removed points
manifold threshold normalized value
```

### 10.5 判断标准

D4RT geometry 可用标准：

```text
median residual < 0.15m
p90 residual < 0.35m
empty_projected_mask_ratio < 30%
G2/G3 relative AP drop vs G0 <= 3 AP points
G2/G3 relative AP50 drop vs G0 <= 5 AP50 points
```

如果失败：

```text
结论：D4RT metric geometry 当前不能替代 ScanNet RGB-D/pose。
主方法继续使用 D4RT correspondence / boundary evidence，而不是 geometry replacement。
检查 t_cam reference、frame indexing、UV convention、axis convention、density thresholds。
```

---

## 11. Phase 8：Tune30 / final / dynamic gates

### 11.1 Tune30 gate

只有满足以下条件才进入 tune30：

```text
Phase 2 edge oracle pass
Phase 3 edge evidence pass
Phase 4 signed partition minimum gate pass
metric integrity pass
D4RT contribution control pass or claim boundary graph without D4RT advantage
```

Tune30 固定不超过 8 个 configs：

```text
best E/P config from probe5
best materialization config
one ablation without D4RT
one ablation mask-only
one ablation depth/normal-only
one conservative version
one aggressive version
one Stream3D diagnostic row
```

### 11.2 Final gate

只允许 locked config 跑 final 一次。Final 不调阈值。

最低可报告条件：

```text
probe5 and tune30 both show same-support gap reduction
final own-support not just tiny support
final S1 not catastrophically below probe5/tune30
```

### 11.3 Dynamic Replica / Replica-Dynamic

只有在数据有 official instance/object IDs 时报告：

```text
IDF1
IDSW
fragmentation
reactivation after occlusion
4D IoU
trajectory consistency
semantic consistency over time
```

如果没有 object IDs：

```text
只能报告 qualitative / pseudo-consistency diagnostic，不得写 official tracking performance。
```

v18 dynamic 目标不是立刻做主表，而是验证 signed boundary graph 的 4D identity 是否在动态遮挡/重现时优于 static overlap。

---

## 12. 必须生成的可视化

每个 phase 至少输出以下可视化：

```text
1. AP own/cross support matrix heatmap
2. support ratio / GT crop-full bar chart
3. Stream3D same-support delta plot
4. D4RT surfel graph overlay on RGB
5. signed cut/merge edge overlay
6. top false cut edges and top false merge edges
7. graph partition colored by component
8. mask internal decomposition overlay: owned / unknown / reject
9. final prediction vs GT mesh panels
10. missed GT object panels
11. duplicate prediction per GT panels
12. D4RT real vs shuffle edge evidence comparison
13. D4RT geometry Sim3 residual heatmap
```

每个 figure 必须有 JSON sidecar：

```text
scene
method/config
phase
eval_policy
AP/AP50/AP25 if available
pre%
union%
GT crop/full
failure tags
source paths
```

---

## 13. Stop rules

Codex 必须遵守以下 stop rules：

```text
1. Phase 2 edge oracle fail -> 不允许写 non-GT graph partition solver。
2. Phase 3 edge AUC fail -> 不允许继续调 partition；先修 evidence。
3. Phase 4 partition fail 且 edge AUC high -> 只修 partition/materialization，不换 evidence。
4. Phase 4 partition fail 且 edge AUC low -> 回 Phase 3。
5. D4RT controls fail -> 不允许 claim D4RT-native advantage。
6. own-support high but S0/S1 fail -> 不能 claim method success。
7. oracle result cannot enter method table under any name。
8. GT/RGB-D Sim3 alignment cannot be used in method prediction, only diagnostic。
```

---

## 14. v18 最低交付清单

Codex 必须至少交付：

```text
1. 完整 code review packet。
2. Phase 0 unified matrix。
3. Phase 1 surfel graph summary。
4. Phase 2 edge boundary oracle。
5. Phase 3 edge evidence AUC table。
6. Phase 4 signed graph partition method result。
7. D4RT real/shuffle/no-temporal controls。
8. 至少 20 张 failure visualization。
9. 清晰复盘：如果失败，必须指出是 graph coverage、edge evidence、partition、materialization 中哪一层失败。
```

最低 method 结果表必须包含：

```text
M own
Stream3D on M
M on S0
M on S1
M with D4RT shuffle
M no-temporal
M mask-only
M depth-normal-only
```

---

## 15. v18 成功时可以写什么，失败时必须写什么

### 15.1 成功时

如果 signed boundary graph 在 own 和 cross-support 都超过 repair_cmask/O38，并明显缩小 Stream3D same-support gap，可以写：

```text
We find that the bottleneck is not candidate ranking but boundary inference. By reframing 2D masks as signed boundary measurements over D4RT material surfels, the method improves same-support object quality and reduces support-shrinking artifacts relative to previous Stream4D variants.
```

如果 D4RT real 明显优于 shuffle/no-temporal，可以写：

```text
D4RT material correspondence contributes to object boundary inference by aggregating signed mask evidence over time, rather than merely selecting sparse object proposals.
```

### 15.2 失败时

必须写：

```text
v18 did not solve static ScanNet object formation. The failure layer is <graph coverage / edge evidence / partition / materialization>. The result suggests that the current D4RT-surfels + CropFormer masks do not yet provide sufficient non-GT boundary evidence for robust static object partition.
```

不能写：

```text
Stream4D 已接近或超过 Stream3D。
D4RT-native semantic 4D reconstruction 已完成。
Dynamic tracking 已证明。
D4RT geometry replacement 已成立。
```

---

## 16. 一句话总结给 Codex

v18 不再写“更好的候选筛选器”。你要做的是：

```text
构建 D4RT surfel adjacency graph；
把 2D masks 转成 signed cut/merge edge evidence；
先用 edge oracle 验证图是否可切；
再用 non-GT signed evidence 做 graph partition；
最后统一 own/cross-support 和 Stream3D 比较。
```

如果这条线失败，也必须明确失败在哪一层。不要再通过 top-k、NMS、WTA、score sweep、candidate union 或 memory postprocess 继续消耗时间。
