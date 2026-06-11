# Stream4D v19：4D TubeCover、Surfel-Mesh Materialization 与 Tube Manifold Refining 实验计划

面向 Codex 执行。本文档基于 v18 signed boundary graph 结果重新制定。v19 不再继续盲目调 `pre-cut`、`kNN k`、`partition threshold`、`C_hybrid solver`、`WTA` 或 `score sweep`。v18 的关键事实是：signed surfel graph 已经可以构造，非 GT pre-cut 能缓解巨大粘连，但 **GT-only edge oracle 仍没有通过最小 gate**，最佳 bank16 oracle 为 `0.434865 / 0.581342 / 0.664137`，node/edge GT coverage 只有约 `0.45 / 0.41`。因此 v19 的第一任务不是写 non-GT solver，而是回答：**oracle 为什么不过？是 surfel graph 表示不对、surfel-to-mesh materialization 不对，还是 object support 扩张方式不对？**

本文所有公式均使用 Typora 友好的 `$...$` 或 `$$...$$` 格式，不使用 `\[\]`。

---

## 0. 总体判断与 v19 的目标

v18 不能简单判定 signed boundary graph 思路失败。原因有两点。

第一，v18 的 best GT edge oracle AP/AP50 已经不低：`0.434865 / 0.581342`。它在 AP 和 AP50 上已经高于许多历史非 GT 方法，说明 signed graph + GT edge cut 不是完全没有上界。

第二，v18 的硬短板是 coverage：`node_gt_label_coverage≈0.45`、`edge_gt_label_coverage≈0.41`。这意味着近一半 surfel/edge 没有可靠落到可评估 GT instance 上。这个问题会直接压低 edge oracle，因为即使用 GT edge labels，也只能切分已经被 label/materialize 的那部分 graph。

因此 v19 的核心目标是：

```text
先把 v18 的 GT edge oracle failure 分解成 graph coverage / edge evidence / materialization 三个子问题，
再把 Stream3D 的两个强归纳偏置 4D 化：
  1. set-covering noise observation filtering
  2. manifold refining
并在 D4RT material tube field 上实现 4D TubeCover + Tube Manifold Refining。
```

换句话说，v19 不再做：

```text
candidate -> score -> union -> WTA
```

而是做：

```text
D4RT material surfel tubes
-> robust surfel-to-mesh / surfel-to-region materialization
-> 4D set-covering of semantic observations
-> signed tube manifold partition
-> object support export
```

v19 只在以下条件满足后才允许进入 non-GT final method table：

```text
1. GT-only coverage/materialization oracle 过最低 gate。
2. non-GT edge evidence 能以可测方式预测 GT cut/merge edges。
3. signed tube partition 在 S0/S1 上超过 O38 / repair_cmask 等历史 broad-support baselines。
4. 每个 method config 同时汇报 own、Stream3D-on-method-support、method-on-S0、method-on-S1。
```

---

## 1. 评估与审计硬约束

### 1.1 统一四行评估

每个 reportable method config `M` 必须输出以下四行，不允许只报 own/recompute：

```text
M own:
  prediction = M
  pre_points = M

Stream3D on M:
  prediction = Stream3D baseline P0
  pre_points = M

M on S0:
  prediction = M
  pre_points = Stream3D own support S0

M on S1:
  prediction = M
  pre_points = historical sparse support S1
```

如果 `M` 是从 parent config 后处理得到，还必须报告：

```text
M inherit parent:
  prediction = M
  pre_points = parent
```

每行必须记录：

```text
AP / AP50 / AP25
eval_policy
prediction_config
pre_points_config
pre_points %
prediction union %
union in target scene %
union in target pre_points %
GT crop/full
#pred
mean points/object
conflict rate
tiny object ratio <100 points
large object ratio >1000 points
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per matched GT
runtime
manifest_integrity_pass
```

### 1.2 GT/Sim3 对齐硬约束

方法内部不能用 GT/RGB-D/ScanNet mesh 的 Sim3 对齐来生成 prediction、选择 object、打分、更新 memory 或做 partition。

允许的情况只有：

```text
evaluation/testing diagnostic only:
  D4RT geometry -> GT/RGB-D/mesh Sim3 alignment
  GT edge label oracle
  GT materialization oracle
  GT failure decomposition
```

如果某个 artifact 使用 GT/Sim3 diagnostic，manifest 必须写：

```json
{
  "uses_gt_for_prediction": false,
  "uses_gt_for_diagnostic": true,
  "alignment_source": "gt_or_rgbd_eval_only",
  "is_method_result": false,
  "is_diagnostic_only": true,
  "forbidden_for_method_table": true
}
```

方法内部允许 D4RT 自己与自己对齐，例如窗口间 D4RT self-Sim3 / D4RT self-stitching，但 manifest 必须写：

```json
{
  "alignment_source": "d4rt_self_only",
  "uses_gt_for_prediction": false
}
```

### 1.3 每轮代码审计包

Codex 每轮必须提交完整审计包：

```text
stream4d_v19_<phase>_code_review_packet.zip
stream4d_v19_<phase>_code_review_packet.sha256
stream4d_v19_<phase>_filelist.txt
stream4d_v19_<phase>_ziptest.log
stream4d_v19_<phase>_git_diff.patch
stream4d_v19_<phase>_git_status.txt
```

zip 内必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/*.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_v19_*.sh
Stream3D/docs/stream4d_v19_执行日志.md
Stream3D/docs/stream4d_v19_实验结果复盘.md
Stream3D/outputs/audit/v19_*/*.json
Stream3D/outputs/audit/v19_*/*.md
Stream3D/outputs/audit/v19_*/*.csv
Stream3D/data/evaluation/scannet/*_class_agnostic.txt
new method prediction/TMP manifests
probe5 minimal prediction/TMP artifacts sufficient for audit rerun
```

必须通过：

```bash
python -m py_compile evaluation/*.py stream4d/*.py tools/*.py tests/*.py
python -m unittest discover tests
python -m tools.scan_reportable_configs --require-manifest --require-eval-policy
python -m tools.verify_stream4d_metric_integrity --require-manifest
```

如果 `open3d` 或 GPU 缺失，测试必须拆成：

```text
pure-python tests
open3d-required tests
gpu-required tests
```

pure-python protocol tests 不能因为 optional dependency 缺失而失败。

---

## 2. 历史基准和 v19 成功标准

### 2.1 历史基准

v19 必须固定以下 baselines，不允许选择性忽略：

```text
P0 Stream3D on S0:
  0.235730 / 0.414306 / 0.537786

P0 Stream3D on S1:
  0.399213 / 0.597171 / 0.742535

O38 own:
  0.081038 / 0.219225 / 0.492501
  pre% = 66.6809

repair_cmask own:
  0.101653 / 0.248464 / 0.494844
  pre% = 60.8353

repair_cmask on S1:
  0.102883 / 0.242779 / 0.576250

P_v6compact on S1:
  0.284832 / 0.503962 / 0.671915
```

v18 diagnostic reference：

```text
v18 best GT edge oracle:
  0.434865 / 0.581342 / 0.664137
  node coverage ≈ 0.450
  edge coverage ≈ 0.411
  exported objects ≈ 8.6
  exported points ≈ 4848
```

### 2.2 v19 阶段 gate

v19 不要求 probe5 立即超过 Stream3D on S1，但必须证明方向实质前进。

#### Coverage / materialization gate

```text
node_gt_label_coverage >= 0.70
edge_gt_label_coverage >= 0.60
per-GT surfel-covered object ratio >= 0.65
covered mesh vertex ratio >= 0.25
```

如果该 gate 不过，不能进入 non-GT partition method。

#### GT oracle gate

```text
GT edge/materialization oracle:
  AP   >= 0.45
  AP50 >= 0.60
  AP25 >= 0.72
```

或者：

```text
relative to Stream3D S1:
  AP gap <= 0.02
  AP50 gap <= 0.03
  AP25 gap <= 0.05
```

如果该 gate 不过，必须继续做 diagnostic，不允许启动 Phase 5 method table。

#### Non-GT edge evidence gate

```text
edge cut/merge AUC >= 0.70
precision@top10% cut edges >= 0.65
false cut rate inside same GT object <= 0.20
false merge rate across GT object boundary <= 0.25
```

如果不满足，不能运行 final partition solver，只能修 evidence source。

#### Method gate

probe5 method 必须满足：

```text
M own AP   >= 0.16
M own AP50 >= 0.35
M own AP25 >= 0.55
M own pre% >= 25

M on S0 AP   > O38 on S0 AP = 0.033
M on S0 AP50 > O38 on S0 AP50 = 0.123
M on S0 AP25 > O38 on S0 AP25 = 0.392

M on S1 AP   > repair_cmask on S1 AP = 0.103
M on S1 AP50 > repair_cmask on S1 AP50 = 0.243
M on S1 AP25 > repair_cmask on S1 AP25 = 0.576
```

更强 gate：

```text
M on S1 AP >= 0.18
M on S1 AP50 >= 0.35
M on S1 AP25 >= 0.62
```

只有通过 method gate，才进入 tune30。

---

## 3. Phase 0：复现 v18 与补齐统一诊断

### 3.1 目标

建立 v19 的可信起点。确认 v18 的 best graph、best oracle、coverage、exported objects 与 Phase0 baseline 能被复现，并补充 v18 缺少的关键指标。

### 3.2 实验

Codex 需要复现以下 rows：

```text
P0 on S0
P0 on S1
O38 own
repair_cmask own
repair_cmask on S1
P_v6compact on S1
v18 Phase1 pre-cut k16 d0.15 graph
v18 Phase2 bank16 k16 d0.15 GT edge oracle
v18 Phase2 bank16 k8 d0.25 GT edge oracle
v18 Phase2 grid48 k16 d0.15 GT edge oracle
```

并补充：

```text
pre_points % for oracle exports
prediction union % for oracle exports
GT crop/full for oracle exports
per-GT best IoU mean for oracle exports
GT IoU>=0.25 / 0.50 count
missed GT count
component count per scene
largest object component size
per-scene node coverage and edge coverage
```

### 3.3 成功标准

复现值允许微小浮动：

```text
AP absolute error <= 1e-4
coverage absolute error <= 1e-3
```

如果无法复现，Codex 必须先定位：

```text
1. 是否 split 不一致。
2. 是否 carrier/debug root 不一致。
3. 是否 graph pre-cut 参数不一致。
4. 是否 evaluator pre_points config 不一致。
5. 是否 oracle artifact 被覆盖。
```

不能直接进入下一 phase。

---

## 4. Phase 1：v18 GT edge oracle 失败分解

### 4.1 目标

判断 v18 oracle 失败是下面哪个原因造成：

```text
A. surfel node 没覆盖可评估 GT object surface。
B. surfel-to-mesh materialization 不可靠。
C. graph adjacency / edge label 覆盖不足。
D. oracle component 太碎或太小。
E. export 从 surfel component 到 mesh points 的扩张太保守。
```

### 4.2 必做诊断

对每个 scene、每个 GT object 记录：

```text
gt_instance_id
gt_point_count
num_surfels_projected_to_gt
surfel_node_coverage_ratio
num_edges_inside_gt
num_edges_cross_gt_boundary
num_labeled_edges
edge_coverage_ratio
num_connected_components_inside_gt
largest_component_ratio_inside_gt
best_oracle_component_iou
best_oracle_component_precision
best_oracle_component_recall
failure_type
```

`failure_type` 只允许以下之一：

```text
no_surfel_coverage:
  GT object 几乎没有 surfel node。

label_missing:
  surfel 有，但无法稳定映射到 GT mesh label。

fragmented:
  同一 GT object 的 surfels 被分成很多小 component。

underfilled:
  component 纯度高，但 recall 太低。

overmerged:
  component 跨多个 GT object。

export_lost:
  component 内 surfels 有，但导出 mesh points 很少。
```

### 4.3 Oracle 分层

必须实现三个 GT-only oracle，不进入 method table：

#### Oracle A：surfel-label upper bound

直接用 surfel 的 GT label 分组，不使用 graph edge，输出每个 GT label 对应的 surfel node set，再通过当前 export materializer 导出 mesh points。

目的：判断 node/materialization 自身上界。

记录：

```text
AP / AP50 / AP25
node coverage
exported points
per-GT recall
per-GT precision
```

#### Oracle B：edge-cut oracle

复现 v18：使用 GT edge labels 删除跨 GT 边，对 graph component 导出。

目的：判断 graph adjacency + edge cut 的上界。

#### Oracle C：mesh-covered oracle

先把 surfel materialized 到 mesh vertices，得到 covered mesh vertex set，然后直接按 GT instance label 把 covered vertices 分组导出。

目的：剥离 graph partition，只看“covered mesh 上是否能形成高 AP”。

### 4.4 判断标准

如果 Oracle C 很高而 Oracle B 低：

```text
mesh materialization 足够，graph adjacency / cut / merge 出问题。
下一步修 graph construction 和 edge evidence。
```

如果 Oracle A 高但 Oracle C 低：

```text
surfel GT labels 有，但 mesh export/materialization 丢点。
下一步修 surfel-to-mesh export。
```

如果 Oracle A/C 都低：

```text
D4RT surfel sampling / visibility / mesh association 覆盖不足。
不要继续 partition solver。
先修 carrier density、multi-frame association 或使用 evaluation bridge support fill。
```

最低推进条件：

```text
Oracle C AP25 >= 0.72
covered mesh vertex ratio >= 0.25
per-GT covered object ratio >= 0.65
```

否则进入 Phase 2A 修 materialization，不进入 Phase 3/4。

---

## 5. Phase 2A：Surfel-to-Mesh / Support Materialization 修复

### 5.1 目标

v18 只把约 45% node、41% edge 可靠 label 到 GT。Phase 2A 要提高 materialization 覆盖，使 D4RT surfel graph 能真正落到 ScanNet evaluator 的 mesh universe。

### 5.2 Materialization variants

Codex 必须实现并比较以下 export bridge。注意：这些可以使用 RGB-D/mesh/GT 进行 evaluation diagnostic，但 method 内部不得用 GT labels。

#### M0：v18 baseline materialization

保持 v18 原逻辑，作为对照。

#### M1：multi-frame RGB-D hit union

对每个 surfel $s_i$，使用它在所有 visible target frames 的 UV：

$$
\{\pi_i(t) \mid v_i(t)=1\}
$$

对每个 target frame：

```text
1. 从 predicted UV 读取 ScanNet depth。
2. 用 ScanNet camera pose/intrinsics backproject 到 world point。
3. NN 到 ScanNet mesh vertex。
4. 过滤 depth invalid、pose invalid、NN distance outlier。
5. 聚合多个 frame 的 mesh vertex votes。
```

每个 surfel 得到：

```text
mesh_vertex_votes
best_vertex
vote_entropy
median_nn_distance
hit_count
hit_frame_count
```

#### M2：component-supported dilation

对一个 surfel component，不只导出 surfel hit vertices，还在 mesh graph / point KNN 上做受限扩张：

```text
seed = component hit vertices
candidate = nearby mesh vertices within radius r or graph distance g
keep if:
  supported by same 2D mask region in at least n frames, or
  adjacent to high-confidence seed and no strong boundary cut evidence
```

记录：

```text
seed_points
added_points
expansion_ratio
purity diagnostic using GT only
contamination ratio diagnostic using GT only
```

#### M3：mask-region fill with component anchors

每个 component 在每一帧有一组 surfel anchors。用这些 anchors 在对应 2D mask 内找 connected region，只把与 anchors 连通且不跨 strong boundary 的区域 backproject 到 mesh。

这是对 full-mask backprojection 的保守替代：

```text
不是 entire mask -> mesh
而是 anchor-supported region -> mesh
```

### 5.3 指标

每个 variant 必须记录：

```text
node_gt_label_coverage
edge_gt_label_coverage
surfel_hit_rate
mean hit frames per surfel
mesh_vertex_coverage_ratio
covered_gt_instance_count
covered_gt_instance_ratio
per-GT surfel recall
per-GT mesh recall
purity
contamination
AP / AP50 / AP25 of Oracle A/B/C
```

### 5.4 成功标准

Phase 2A 成功条件：

```text
node_gt_label_coverage >= 0.70
edge_gt_label_coverage >= 0.60
Oracle C AP25 >= 0.72
Oracle C AP50 >= 0.60
```

如果 M1 提升明显：

```text
使用 M1 作为后续默认 materialization。
```

如果 M1 不提升、M2/M3 提升：

```text
问题是 sparse surfel hits 不足，需要 component-supported expansion。
后续 partition 必须输出 component anchors + mask-region fill，而不是直接 surfel-only export。
```

如果三者都不提升：

```text
停止 signed graph 主线。
Codex 必须检查 D4RT grid generation、UV scaling、depth sampling、ScanNet frame alignment。
不要写 Phase3 evidence/partition。
```

---

## 6. Phase 2B：4D TubeCover Observation Filtering

### 6.1 目标

借鉴 Stream3D set-cover 的核心思想，但不要直接复用 3D point set-cover。Stream3D 的 set-cover 是为了在多视角 noisy masks 中选出 key masks。v19 的 4D TubeCover 要在 D4RT tube field 上选择可靠 semantic observations，让坏 masks 不再支配 edge/partition。

### 6.2 Universe

定义 4D TubeCover 的 universe：

$$
\mathcal{U}=\mathcal{U}_{node}\cup\mathcal{U}_{edge}\cup\mathcal{U}_{boundary}\cup\mathcal{U}_{visibility}
$$

其中：

```text
U_node:
  visible surfel nodes / tube elements。

U_edge:
  surfel adjacency edges。

U_boundary:
  likely object boundary edges。

U_visibility:
  visible-outside negative observations。
```

一个 2D mask observation $m$ 不再是 object candidate，而是一个 coverage item：

```text
covers interior surfels
covers safe merge edges
covers boundary/cut edges
creates visible-outside negative evidence
has boundary risk
has temporal consistency
```

### 6.3 Greedy objective

实现 training-free greedy set cover：

$$
\Delta(m)=
\alpha C_{node}(m)
+\beta C_{edge}(m)
+\gamma C_{boundary}(m)
-\lambda R_{conflict}(m)
-\mu R_{redundancy}(m)
-\nu R_{temporal}(m)
$$

其中：

```text
C_node: 新覆盖的可靠 surfel nodes。
C_edge: 新覆盖的可靠 same-object merge edges。
C_boundary: 新覆盖的可靠 cut/boundary edges。
R_conflict: 与已选 observations 冲突。
R_redundancy: 对已覆盖 universe 的重复。
R_temporal: 与 D4RT trajectory / visibility 不一致。
```

### 6.4 Controls

必须运行：

```text
all masks no filtering
area top-k
mask-count top-k
random same-count
Stream3D-like point coverage set-cover adapted to surfel nodes only
v19 TubeCover full objective
D4RT shuffle TubeCover
no-temporal TubeCover
```

### 6.5 指标

不只记录 AP，还要记录 coverage/evidence：

```text
selected masks count
selected mask frames
covered nodes %
covered edges %
covered boundary edges %
redundant coverage ratio
conflict rate
edge AUC after selecting observations
AP/AP50/AP25 after downstream oracle partition
```

### 6.6 成功标准

TubeCover 必须满足：

```text
edge AUC improves over all-masks baseline by >= 0.05
boundary precision@top10% improves by >= 0.08
redundant mask count reduced by >= 30%
Oracle B AP25 does not drop more than 0.03 from all-masks oracle
```

如果 TubeCover 不优于 all-masks/no-temporal/shuffle：

```text
不要继续调 partition。
先检查 observation features 是否错误，特别是 mask boundary distance、visibility、UV frame id、mask id 对齐。
```

---

## 7. Phase 3：Non-GT Signed Edge Evidence

### 7.1 目标

在 GT edge oracle 通过后，构建 non-GT signed edge weights，判断它们是否能预测 GT cut/merge labels。此阶段仍不输出 method table，只做 edge classifier diagnostic。

### 7.2 Edge features

每条 edge $(i,j)$ 必须记录：

```text
spatial distance in image
spatial distance in D4RT/world/RGB-D bridge
co-visible frame count
same-mask interior count
inside-outside boundary count
boundary distance min/mean
RGB/color difference
DINO/CLIP/local feature difference if available
depth discontinuity
normal discontinuity
D4RT motion consistency
D4RT cycle consistency if available
visibility disagreement
TubeCover-selected observation support
```

### 7.3 Signed weights

计算：

$$
w_{ij}^{merge}=f_{merge}(e_{ij}),\quad
w_{ij}^{cut}=f_{cut}(e_{ij})
$$

training-free，不能训练 classifier。可以用 fixed rule / calibrated score：

```text
merge if long-term same-mask interior + appearance similar + no depth/normal boundary
cut if repeated inside/outside boundary + depth/normal/color discontinuity + motion inconsistency
unknown if evidence insufficient
```

### 7.4 Metrics

使用 GT edge labels 只做 diagnostic：

```text
edge cut/merge AUC
AP for cut edge ranking
precision@top1%, top5%, top10% cut edges
recall@top10% cut edges
false cut inside same GT object
false merge across GT boundary
edge unknown ratio
per-scene edge AUC
```

### 7.5 成功标准

```text
cut/merge AUC >= 0.70
precision@top10% cut edges >= 0.65
false cut <= 0.20
false merge <= 0.25
D4RT-real AUC - D4RT-shuffle AUC >= 0.04
D4RT-real AUC - no-temporal AUC >= 0.03
```

如果不满足：

```text
1. 如果 depth/normal alone 高，D4RT edge evidence 没贡献：回到 TubeCover / D4RT feature design。
2. 如果 all features 都低：2D mask boundary 不可靠，尝试 SAM/HQ-SAM/HQ boundary or image edge source。
3. 如果 real 不强于 shuffle/no-temporal：D4RT temporal signal 没有被正确接入。
```

---

## 8. Phase 4：Tube Manifold Partition

### 8.1 目标

把 Stream3D 的 3D manifold refining 升级成 4D tube manifold refining。object 不再来自 candidate union，而来自 surfel tube graph partition。

### 8.2 Energy

使用 signed graph partition：

$$
E(Y)=
\sum_{(i,j)\in E}
 w^{cut}_{ij}\mathbf{1}[y_i=y_j]
+
 w^{merge}_{ij}\mathbf{1}[y_i\neq y_j]
+
\lambda C(Y)
+
\eta U(Y)
$$

其中：

```text
C(Y): complexity，惩罚过碎和过大 component。
U(Y): unknown penalty，允许低证据区域不分配。
```

实现可以先用 deterministic approximation：

```text
1. strong cut edges 先删除。
2. strong merge edges 建 initial components。
3. agglomerative merge with signed gain。
4. split components with high internal cut evidence。
5. small components attach only if boundary risk low。
6. unknown components not exported。
```

不要用普通 connected component 作为 final partition。

### 8.3 Export

使用 Phase 2A 中最好的 materialization。

每个 partition component 输出：

```text
core surfels
core mesh vertices
component-supported mask-region fill
fringe mesh vertices
unknown/reject vertices
```

最终 prediction：

```text
object support = core + conservative fringe
unknown/reject 不导出
```

### 8.4 Metrics

统一四行评估，同时记录：

```text
#components
#exported objects
mean component size
largest component ratio
unknown node ratio
small component ratio
per-GT component fragmentation
per-GT overmerge count
component purity
component recall
AP/AP50/AP25 four-row matrix
```

### 8.5 成功标准

最低：

```text
M on S0 AP > 0.033
M on S0 AP50 > 0.123
M on S0 AP25 > 0.392
M on S1 AP > 0.103
M on S1 AP50 > 0.243
M on S1 AP25 > 0.576
```

进阶：

```text
M on S1 AP >= 0.18
M on S1 AP50 >= 0.35
M on S1 AP25 >= 0.62
Stream3D-on-M support AP - M own AP <= 0.10
```

如果 partition own 高但 S0/S1 低：

```text
仍是 tiny/own-support effect；回 Phase 2A 修 materialization。
```

如果 S0/S1 有提升但 own 降：

```text
可接受，说明 broad support 进步；继续调 unknown/fringe，不调 score/NMS。
```

---

## 9. Phase 5：Local-to-Historical Tube Memory

### 9.1 启动条件

只有 Phase 4 method gate 通过，才启动 Phase 5。否则不要做 memory，因为 memory 会把错误 partitions 传播得更远。

### 9.2 目标

把 Stream3D local-to-historical update 4D 化：history 不是 3D mask pool，而是 object tube memory。

每个 object tube memory：

```text
object_id
material surfel ids
temporal visibility span
appearance summary
semantic evidence summary
motion summary
state: active / lost / reappeared / split_suspect / merge_suspect
```

Matching 不用 3D overlap，而用：

```text
shared material tubes
predicted trajectory consistency
appearance consistency
semantic observation consistency
visible-outside negative evidence
strong cannot-link violation
```

### 9.3 Metrics

Static ScanNet 多窗口：

```text
num_windows
num_created
num_matched
num_reactivated
num_split_suspect
num_merge_suspect
object fragmentation rate
object duplication rate
AP/AP50/AP25 own/S0/S1
```

Dynamic / Replica-Dynamic 如果数据满足：

```text
IDF1
ID switches
fragmentation
MOTA-like diagnostic
4D tube IoU over time
occlusion reactivation precision/recall
```

如果没有 instance/object IDs，不能报告 official tracking metrics，只能 qualitative or pseudo consistency。

---

## 10. Phase 6：D4RT 几何对齐后给 Stream3D 使用的 diagnostic

### 10.1 目标

回答：如果把 D4RT geometry 通过 evaluation-only GT/RGB-D Sim3 对齐到 ScanNet geometry 后，喂给 Stream3D-style pipeline，指标会掉多少？这个实验用于判断 D4RT metric geometry 精度对 Stream3D 结果的影响，不是 v19 主方法。

### 10.2 硬约束

```text
GT/RGB-D/mesh Sim3 只能用于 evaluation/testing diagnostic。
不能用于 v19 method 的 object selection、partition、memory、scoring。
输出 config 必须 is_method_result=false。
```

### 10.3 Variants

```text
G0: original Stream3D RGB-D/pose geometry baseline
G1: D4RT raw geometry + density-normalized thresholds
G2: D4RT scene-level Sim3 aligned geometry + density-normalized thresholds
G3: D4RT window-level Sim3 aligned geometry + density-normalized thresholds
G4: D4RT aligned geometry + Stream3D manifold thresholds re-scaled by point spacing quantiles
G5: D4RT aligned geometry + no manifold refining
```

### 10.4 Metrics

```text
Sim3 residual median/p90
point spacing quantiles
projection hit rate
mask backprojection hit rate
#objects
pre%
AP/AP50/AP25
per-GT best IoU
failure scenes
```

### 10.5 判断标准

如果 G2/G3 相对 G0：

```text
AP drop <= 3 points and AP50 drop <= 5 points
```

说明 D4RT geometry 有潜力成为 ScanNet geometry source。

如果 drop 很大：

```text
D4RT metric geometry 不适合静态 ScanNet Stream3D pipeline。
论文叙事中 D4RT 应定位为 material correspondence / temporal identity，而不是 RGB-D geometry replacement。
```

---

## 11. Phase 7：Tune30 / Final / Dynamic

### 11.1 Tune30

只有 Phase 4 method gate 通过才跑 tune30。

Tune30 允许调：

```text
TubeCover weights
edge cut/merge thresholds
unknown/fringe thresholds
component-supported dilation radius
```

不允许调：

```text
基于 AP 单独为某个 scene 定制阈值
使用 GT failure decomposition 修改 method config
final split 上继续调参
```

### 11.2 Final

final split 只允许跑一次 locked config。

必须同时报告：

```text
final M own
final Stream3D on M
final M on S0
final M on S1
```

### 11.3 Dynamic

启动条件：

```text
数据存在 image/depth/trajectory/instance ID/object ID。
```

如果没有 object ID：

```text
不能报告 IDF1/MOTA/official 4D IoU。
```

---

## 12. 可视化要求

每个通过 Phase 2A 以后的方法，必须输出以下可视化：

```text
1. surfel-to-mesh hits overlay。
2. GT coverage failure panels: no_surfel / label_missing / fragmented / underfilled / overmerged / export_lost。
3. edge cut probability map overlay on frames。
4. selected TubeCover masks vs rejected masks。
5. graph partition components colored by object。
6. method prediction vs Stream3D prediction on same support。
7. false merge and false split examples。
8. per-scene AP/support scatter。
```

保存路径：

```text
Stream3D/outputs/audit/v19_visuals/<phase>/<scene>/
```

每张图必须有 JSON manifest，记录：

```text
scene
config
phase
uses_gt_for_visualization
prediction_config
support_config
failure_type
```

---

## 13. Stop Rules

Codex 必须遵守以下停止规则：

```text
1. Phase 1 oracle decomposition 未完成，不得写 new method。
2. node/edge coverage 未过 gate，不得运行 non-GT partition solver。
3. edge evidence AUC 未过 gate，不得导出 method table。
4. partition on S0/S1 不超过 O38/repair_cmask，不得跑 tune30。
5. method 不能强于 D4RT shuffle / no-temporal controls，不得 claim D4RT-native。
6. 任何使用 GT/Sim3 的结果不得进入 method table。
7. 如果三轮 fallback 都不能提升 materialization coverage，停止 signed graph 主线，回到 D4RT surfel generation / mask source quality。
```

---

## 14. v19 期望交付物

Codex 完成 v19 后，必须提交：

```text
1. 完整代码审计包。
2. Phase0 复现表。
3. Phase1 oracle failure decomposition。
4. Phase2A materialization comparison。
5. Phase2B TubeCover diagnostic。
6. Phase3 non-GT edge evidence AUC table。
7. Phase4 partition method four-row matrix。
8. D4RT real/shuffle/no-temporal controls。
9. D4RT-aligned geometry -> Stream3D diagnostic。
10. 可视化 manifest。
11. 复盘文档，明确哪些 gate 通过、哪些未通过。
```

如果最终没有 reportable method result，也必须给出清晰结论：

```text
是 materialization 不足，还是 edge evidence 不足，还是 graph partition 不足。
```

不能再只写“结果未达标，需要继续尝试”。

