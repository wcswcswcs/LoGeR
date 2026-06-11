# Stream4D v16：停止后处理堆叠，重做 Feed-forward Semantic 4D Object Field 的可执行计划

本文档面向 Codex 执行。它不是继续 v14/v15 的阈值扫参计划，而是一次算法层面的重启。过去几十轮实验已经说明：把 2D mask、regionlet、surfel atom、carrier component、masklet 或 memory slot 直接当作 object primitive 的路线不成立。v16 的目标是验证一个新的、可投稿顶会的问题表述：

**Feed-forward semantic 4D reconstruction and tracking as training-free latent object explanation over D4RT material surfels and noisy semantic measurements.**

本文所有公式使用 Typora 友好的 `$...$` 或 `$$...$$`，不使用 `\[\]`。

---

## 0. 必须先承认的失败事实

过去结果已经足够清楚，不能再继续把失败归因为某个小 bug、某个阈值、某个 score 或某个 NMS 没调好。

### 0.1 tiny-support 高分不是完整 reconstruction

B1/O1/M13c/C_surfel 等结果能在自己的 tiny support 上得到较好 AP/AP50/AP25，但它们的 pre_points 通常只有 2%–4%。例如 B1 own 是：

```text
AP/AP50/AP25 = 0.328439 / 0.629266 / 0.884363
pre% = 3.9861
```

但 B1 放到 Stream3D S0/S1 support 上几乎失效：

```text
B1 on S0 = 0.000635 / 0.004294 / 0.010768
B1 on S1 = 0.016837 / 0.047534 / 0.168162
```

这说明 B1 是 high-precision visible-subset selector，不是 semantic 4D reconstruction。

### 0.2 broad support 后 object quality 仍然不够

O38 c055 own support 已经约 66.68%，但 AP 只有：

```text
O38 own = 0.081038 / 0.219225 / 0.492501
O38 on S0 = 0.033012 / 0.123089 / 0.392066
```

而 Stream3D on S0 是：

```text
0.235730 / 0.414306 / 0.537786
```

这说明 support 不是唯一瓶颈。coverage 上来后，object hypothesis 本身仍然错。

### 0.3 atom-as-object 也失败

v14 试图把 surfel atoms 当作 object candidate，但最好的 A3/A4 target-base oracle 只有：

```text
AP/AP50/AP25 = 0.068627 / 0.117647 / 0.558824
actual pre% ≈ 3.03%
```

补齐 16/16 mask frames 后，raw measurement 覆盖确实大幅上升，但最终 exported support 仍只有约 3%。这说明问题不是只有 mask density，而是 measurement 到 object support 的 materialization / explanation 断裂。

### 0.4 当前最重要的结论

错误不是“D4RT 没用”。D4RT image-space correspondence 多轮显示可用，例如连续帧下 `uv_in01_rate_mean≈0.9858`、`self_uv_error_p90≈1.57px`、`cycle_uv_error_p90≈3.27px`。错误是我们一直把 observation primitive 当成 object primitive。

v16 必须从下面这个错误范式退出：

```text
mask / regionlet / carrier component / surfel atom / masklet / slot -> object
```

改成：

```text
D4RT surfels + masks + regionlets + appearance + negative evidence -> measurements
latent object -> explanation over measurements
```

---

## 1. v16 的总体目标

v16 不以“某个 recompute AP 好看”为成功标准。v16 只回答三个科学问题。

### 问题 A：当前 primitive 是否有足够上界？

如果 broad-support candidate oracle 都低，说明 primitive 错了，不能再写 solver。

### 问题 B：能否从 measurements 推断 latent object，而不是直接导出 measurements？

如果 object explanation 不能同时提升 own-support 和 cross-support，说明 formulation 仍然没有成功。

### 问题 C：D4RT 的贡献是否从 mask selection 升级到 object ownership / split / tracking？

如果真实 D4RT 与 shuffled/no-temporal/no-track controls 差不多，说明 D4RT 没被正确用到。

v16 的最低有效成果不是 full ScanNet 超越，而是：

```text
1. 给出明确 failure decomposition：no candidate / filtered good / wrong assignment / boundary bad / materialization broken。
2. 给出 candidate oracle、slot oracle、materialization oracle 三层上界。
3. 实现一个真正不是 measurement-as-object 的 object explanation prototype。
4. 在 probe5 统一评估矩阵上，至少明显缩小 S0/S1 差距，而不是只提高 own-support。
```

---

## 2. 硬评估协议

每个 method config 必须报告四行，缺一行则不能进入讨论：

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
  pre_points = historical 32f support
```

如果方法是从 parent config 后处理得到，还必须报告：

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
uses_gt_for_prediction
uses_gt_for_diagnostic
```

**GT/RGB-D/ScanNet mesh Sim3 对齐只允许用于 evaluation/testing diagnostic。方法内部禁止使用 GT Sim3。D4RT 自己和自己对齐可以，但 manifest 必须写 `alignment_source=d4rt_self_only`。**

---

## 3. Phase 0：完整代码审计和复盘再现

### 目标

先确认当前包可审计，避免再出现“复盘说测试通过，但人工解压缺依赖无法跑”的情况。

### 必须提交

```text
stream4d_v16_code_review_packet.zip
stream4d_v16_code_review_packet.sha256
stream4d_v16_filelist.txt
stream4d_v16_ziptest.log
stream4d_v16_git_status.txt
stream4d_v16_git_diff.patch
```

zip 内必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/*.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_v16_*.sh
Stream3D/docs/stream4d_v16_执行日志.md
Stream3D/docs/stream4d_v16_实验结果复盘.md
Stream3D/outputs/audit/**/*.json
Stream3D/outputs/audit/**/*.md
Stream3D/outputs/audit/**/*.csv
Stream3D/data/evaluation/scannet/*_class_agnostic.txt
Probe5 所需 prediction/TMP manifests
```

### 检查命令

```bash
python -m py_compile Stream3D/evaluation/*.py Stream3D/stream4d/*.py Stream3D/tools/*.py Stream3D/tests/*.py
python -m unittest discover Stream3D/tests
python -m tools.scan_reportable_configs --require-manifest --require-eval-policy
python -m tools.verify_stream4d_metric_integrity --require-manifest
```

如果因为 `open3d`、torch 或 GPU 缺失导致测试失败，Codex 必须拆分：

```text
pure-python tests
open3d-required tests
gpu-required tests
```

不能让 optional dependency 阻塞所有 protocol tests。

### 成功标准

```text
py_compile pass
pure-python unit tests pass
reportable scan suspicious=0
uses_gt_for_prediction=0
AP core hash consistent
oracle/diagnostic configs forbidden for method table
```

---

## 4. Phase 1：三层 oracle 诊断，先判断到底有没有可解空间

### 目标

过去只做 single candidate oracle 或 candidate source oracle，仍然无法判断失败到底发生在 primitive、solver 还是 materialization。v16 必须拆成三层。

### 三层 oracle

#### 1. Candidate oracle

在每个 candidate primitive 自身 support 上，用 GT 只做 diagnostic selection：

```text
C_mask
C_regionlet
C_surfel_atom
C_masklet
C_hybrid
C_new_measurement_region
```

记录每个 primitive 的 oracle AP/AP50/AP25 和 support。

#### 2. Slot oracle

允许 oracle 从多个 measurements 组成一个 object slot，例如若干 regionlets 或 atoms 的 union。这个 oracle 用来判断“单 primitive 不够，但组合后是否有上界”。

对每个 GT object，允许最多 $K$ 个 measurement units 组成一个 slot：

$$
O_j^{oracle} = \bigcup_{r \in \mathcal{R}_j, |\mathcal{R}_j|\le K} r
$$

建议先跑 $K=2,4,8$。

#### 3. Materialization oracle

给定一个 oracle object slot 的 source measurements，分别用三种 export/materialization 方式导出：

```text
A. surfel-only mesh NN
B. owned-mask-region backprojection
C. core surfel + graph-cut region expansion
```

如果 slot oracle 高但 materialization oracle 低，说明 raw object explanation 有潜力，但 mesh support 导出失败。

### 记录指标

```text
candidate count
measurement units / scene
oracle AP/AP50/AP25
oracle support %
GT crop/full
per-GT best IoU
GT IoU>=0.25 / >=0.50 / >=0.75 count
#units per oracle object
oracle duplicate rate
materialization loss = IoU(slot_raw, exported_support)
```

### 判断标准

如果 broad-support slot oracle 仍低于：

```text
AP50 < 0.60 或 AP25 < 0.80
```

则不要继续优化 solver，说明 primitive/measurement 表达还不够。

如果 slot oracle 高，但 unsupervised method 低，说明下一步聚焦 object explanation solver。

如果 materialization oracle 低，先解决 export/materialization，不要调 object score。

### 不满足条件时 Codex 先尝试

```text
1. 增加 measurement primitive：mask 内按 D4RT surfel seeds + RGB/depth boundary 切 region。
2. 增加 object slot candidate：允许多个 regionlets/atoms union。
3. 修 materialization：从 full mask backproject 改成 object-conditioned graph cut。
4. 如果 D4RT surfel density 不够，提高 grid density 或多 source-frame surfel sampling。
```

---

## 5. Phase 2：构建 measurement bank，不再输出 measurement-as-object

### 目标

构建一个只保存 measurements 的 bank。它不直接产生 object prediction。

每个 D4RT surfel：

$$
s_i = \{X_i(t), \pi_i(t), v_i(t), c_i(t), f_i(t)\}_{t=1}^{T}
$$

每个 semantic measurement：

$$
z_m = (t, R_m, e_m, b_m, q_m)
$$

其中 $R_m$ 是 2D region，$e_m$ 是 appearance/text feature，$b_m$ 是 boundary/depth evidence，$q_m$ 是 mask quality。

### Measurement 类型

```text
mask measurement: 原始 2D mask
region measurement: mask 内按边界/深度/D4RT seeds 切出的 region
surfel measurement: surfel 的 mask membership / outside evidence
masklet measurement: D4RT propagated mask observation
negative measurement: visible but outside / same-frame cannot-link / boundary discontinuity
```

### 必须实现的数据结构

```python
MeasurementBank:
  surfel_id
  frame_id
  region_id
  parent_mask_id
  inside_score
  outside_score
  boundary_distance
  depth_edge_score
  appearance_feature
  d4rt_visibility
  d4rt_confidence
  source_type  # mask, region, masklet, negative
```

### 记录指标

```text
num_surfels
surfel coverage per frame
uv_in01 rate
self/cycle UV error p50/p90
visible track length
num_masks/frame
num_regions/mask
positive observations/surfel
negative observations/surfel
ambiguous surfel ratio
unobserved surfel ratio
boundary-safe ratio
mask frame availability
region purity diagnostic with GT only
```

### 成功标准

```text
uv_in01_rate_mean >= 0.95
cycle_uv_error_p90 <= 5 px
positive observations/surfel >= 2.5 after bank16/masklet repair
unobserved surfel ratio <= 0.05
region conflict rate lower than full mask conflict
```

如果 semantic observation density 不足，Codex 先做：

```text
1. 补齐 predicted mask cache 到连续帧。
2. 用 D4RT propagation 生成 masklet measurements。
3. 降低 mask confidence gate，但只作为 measurement，不直接作为 object。
4. 使用 multiple source frames 生成 surfels。
```

---

## 6. Phase 3：Object-conditioned mask decomposition，不再 full-mask backproject

### 目标

过去失败的核心是 full-mask 当 object 或 full-mask fringe 会污染。v16 必须先分解 mask，再解释 object。

每个 2D mask $M_t$ 不是 object，而是一个待解释区域。给定若干 object seeds / slots，对 mask 内像素做 object-conditioned decomposition：

$$
M_t = R_{t,1} \cup R_{t,2} \cup ... \cup R_{t,n} \cup R_{unknown}
$$

其中每个 $R_{t,k}$ 是某个 object 的 owned region，$R_{unknown}$ 是暂不解释区域。

### 算法要求

对每个 mask：

```text
1. 用 D4RT surfel seeds 作为 foreground/object seeds。
2. 用 visible-outside surfels、depth/normal/RGB edges 作为 negative / boundary。
3. 在 2D mask 内运行 graph cut / watershed / random walker 中的一个 deterministic solver。
4. 输出 owned regions，不输出 full mask。
```

Graph cut energy：

$$
E(L)=\sum_p D_p(L_p)+\lambda\sum_{(p,q)} w_{pq}[L_p\ne L_q]
$$

其中 $D_p$ 来自 D4RT surfel seeds、appearance、distance-to-seed、outside evidence；$w_{pq}$ 来自 RGB/depth/normal boundary。

### 记录指标

```text
num_masks_decomposed
regions/mask
unknown pixels/mask
owned pixels/mask
region conflict rate before/after
WTA removed assignment rate
mask split count
mask no-split count
region boundary adherence
owned region GT purity diagnostic
region recall diagnostic
```

### 成功标准

相比 full-mask/regionlet direct output：

```text
conflict rate 减少至少 50%
S0 AP 至少超过 O38 c055 on S0 = 0.033
S1 AP 至少超过 R1b/O38 historical rows
Stream3D on method-support 与 method own 的 AP gap 缩小
```

如果 graph cut 无提升，Codex 先尝试：

```text
1. 检查 seeds 是否覆盖真实 object，输出 seed/GT diagnostic。
2. 改用更保守 unknown 区域，不强行分配边界区域。
3. 加 appearance feature consistency，而不是只用 geometry。
4. 降低 mask decomposition 范围，只在 high-risk masks 上 split。
```

---

## 7. Phase 4：Global Object Explanation，不再 connected component / per-mask slot

### 目标

真正的 object 是能解释 measurements 的 latent entity，而不是一个 measurement。本阶段实现全局 set packing / split-merge object explanation。

Object slot：

$$
O_k=(A_k, R_k, S_k, U_k, \theta_k)
$$

其中 $A_k$ 是 atom/region set，$R_k$ 是 owned region measurements，$S_k$ 是 surfel set，$U_k$ 是 unknown/reject support，$\theta_k$ 是 appearance/semantic/motion summary。

目标函数：

$$
E(\mathcal{O}) = -\sum_k P(O_k) + \lambda_o\sum_{i<j}Overlap(O_i,O_j) + \lambda_c Complexity(\mathcal{O}) + \lambda_u Unexplained(Z)
$$

其中 $P(O_k)$ 包含：

```text
positive inside evidence
visible-outside negative evidence
appearance consistency
D4RT temporal consistency
boundary consistency
mask explanation quality
```

### 算法要求

先实现 deterministic greedy-MDL：

```text
1. 从 clean births 生成 initial slots。
2. 从 measurement bank 生成 candidate expansions。
3. 每次选择能最大降低 E 的 object/expansion。
4. 强 negative evidence 优先于 weak positive。
5. 允许 unknown，不允许强行分配所有 support。
6. 允许 split 一个 bad mask 给多个 slots 解释。
7. 允许 merge 两个 slots，但只有在没有 negative conflict 时。
```

### 必须记录内部指标

```text
num_initial_births
num_candidate_expansions
num_selected_objects
num_split_masks
num_merged_slots
unexplained_measurement_ratio
unknown_support_ratio
reject_support_ratio
object_overlap_penalty
object_complexity_penalty
positive/negative evidence per object
same-frame cannot-link violations
motion/appearance inconsistency per object
```

### 成功标准

Probe5 gate：

```text
M own AP >= 0.20 with pre% >= 10
M on S0 AP >= 0.08
M on S1 AP >= 0.18
AP50 on S1 >= 0.35
AP25 on S1 >= 0.60
Stream3D-on-M-support minus M-own AP <= 0.08
```

注意：这个 gate 仍低于 Stream3D，但它能证明方向脱离 tiny-support illusion。

如果失败，Codex 根据 failure 分类处理：

```text
no candidate 多：回 Phase 2/3 扩 measurement primitive。
filtered_good 多：降低 birth gate，保留 pending slot，不直接 reject。
wrong_assignment 多：加强 negative evidence 和 split mask。
boundary_bad 多：改 graph cut/materialization。
duplicate 多：加强 set packing overlap penalty。
```

---

## 8. Phase 5：Posterior-controlled materialization，解决 raw coverage 到 exported support 的断裂

### 目标

v14 的最重要断裂是 raw atom-known support 可以很高，但 exported pre% 只有约 3%。v16 必须把 object posterior 真实 materialize 到 ScanNet mesh support。

### 三种 materialization 对照

```text
M_core: 只导出 core surfel NN。
M_region: 导出 object-owned 2D regions backproject。
M_graph: core surfels + object-conditioned graph-cut fringe。
```

禁止默认 full-mask backproject。

### 记录指标

```text
raw object support %
exported pre%
raw-to-export materialization IoU
mesh NN hit rate
owned region pixel count
exported points/object
core/fringe/unknown/reject ratio
GT crop/full
support conflict rate
per-GT best IoU before/after materialization
```

### 成功标准

```text
exported pre% >= 15%
raw-to-export materialization IoU >= 0.60
S0/S1 AP 不比 Phase 4 降低超过 20%
conflict rate <= 0.20
```

如果 exported pre% 仍低，Codex 先尝试：

```text
1. 增大 region backprojection，而不是 surfel NN radius。
2. 使用 depth-valid pixels 补充 object-owned regions。
3. 对 every owned region 做 connected component sanity。
4. 检查 UV-to-depth/pose mapping 是否丢点。
```

---

## 9. Phase 6：D4RT 几何对齐后喂给 Stream3D 的 diagnostic

### 目标

单独判断 D4RT metric geometry 精度对 Stream3D-style pipeline 的影响。此实验只作为 diagnostic，不进入 method table。

### 硬约束

```text
GT/RGB-D/ScanNet mesh Sim3 对齐只能用于 evaluation/testing diagnostic。
不能在 Stream4D method 内部使用该对齐。
manifest 必须写 is_diagnostic_only=true。
```

### 实验设置

比较：

```text
G0: 原版 Stream3D RGB-D/pose geometry
G1: D4RT raw geometry + density-normalized thresholds
G2: D4RT scene-level Sim3 aligned geometry
G3: D4RT window-level Sim3 aligned geometry
G4: D4RT scene-level Sim3 + normalized manifold thresholds
G5: D4RT window-level Sim3 + normalized manifold thresholds
```

关键是不要复用 RGB-D meter-scale 超参。阈值按 D4RT point spacing 分位数设定：

$$
r_{NN}=\alpha \cdot Q_{50}(d_{knn})
$$

$$
\delta_{MR}=\beta \cdot Q_{75}(d_{knn})
$$

### 记录指标

```text
Sim3 scale
Sim3 residual median/p90/p95
D4RT point spacing quantiles
Stream3D AP/AP50/AP25
num exported objects
pre%
projection hit rate
manifold refine rejection rate
mask projection IoU vs RGB-D baseline diagnostic
```

### 判断标准

如果 G2/G4 仍比 G0 AP drop > 5 或 AP50 drop > 8，说明 D4RT metric geometry 不能替代 ScanNet RGB-D/pose 做静态 ScanNet 主几何。

如果 G2/G4 接近 G0，说明之前失败主要是 scale/threshold/adapter 问题，可考虑把 D4RT geometry 作为更强 claim 的 diagnostic support。

---

## 10. Phase 7：tune30 / final / dynamic 进入条件

只有 Phase 4/5 probe5 gate 通过后，才允许跑 tune30。

Tune30 要求：

```text
固定 probe5 选出的参数
own + S0 + S1 + Stream3D-on-M 全部报告
不允许在 tune30 上继续做大量阈值搜索
```

Final 要求：

```text
final split 只跑一次 locked config
不允许 final split 调参
不允许只报 own-support
```

Dynamic Replica 要求：

```text
必须确认 instance/object ID GT 存在，才报告 IDF1/MOTA/4D IoU。
如果只有 RGB/trajectory metadata，没有 instance GT，则只能报告 qualitative 或 D4RT track diagnostic。
不能编造 dynamic semantic AP。
```

---

## 11. v16 的停止条件

如果以下任一条件成立，必须暂停当前路线，不再让 Codex 继续堆工程：

```text
1. broad-support slot oracle AP50 < 0.60。
2. materialization oracle 无法把 exported pre% 提到 15%。
3. real D4RT 与 shuffled/no-temporal controls 在 object explanation 中差距 < 5 AP50 points。
4. probe5 S0 AP 连续两轮仍 < 0.05。
5. Stream3D-on-method-support 持续比 method own 高 > 0.12 AP。
```

这意味着当前 measurement/object explanation formulation 仍错，必须重新定义 primitive，而不是继续调参数。

---

## 12. Codex 最小交付清单

本轮必须至少交付：

```text
1. 完整 code review packet。
2. Phase 1 三层 oracle matrix。
3. Phase 2 measurement bank diagnostics。
4. Phase 3 object-conditioned mask decomposition prototype。
5. Phase 4 global object explanation prototype。
6. Phase 5 materialization diagnostics。
7. Phase 6 D4RT geometry diagnostic。
8. 每个 method config 的 M own / Stream3D on M / M on S0 / M on S1 四行评估。
9. 每个失败 scene 的可视化 panel：GT overlay、Stream3D、method、owned regions、unknown/reject、D4RT surfels。
```

如果时间不够，优先级为：

```text
Phase 1 oracle > Phase 3 decomposition > Phase 4 explanation > Phase 5 materialization > Phase 6 geometry diagnostic
```

原因是：如果 oracle 或 primitive 不成立，后面的 solver/geometry 都是在浪费时间。
