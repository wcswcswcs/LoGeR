# Stream4D v15：从 atom-as-object 转向全局 measurement explanation 的可执行实验计划

面向 Codex 的执行计划。本文档基于最新 `stream4d_v14_probe5_code_review_packet.zip` 解压审计、v14 结果复盘、以及历史 v3-v14 的统一评估矩阵写成。本文档的核心立场是：继续调 `top-k / WTA / atom split / score / support dilation` 不会解决问题；下一步必须把算法从“把观测 primitive 当 object”改成“用 latent object 解释 noisy semantic measurements”。

本文所有公式均使用 Typora 友好的 `$...$` 或 `$$...$$`，不使用 `\[...\]`。

---

## 0. 一句话结论

当前 Stream4D 的主问题不是 evaluator 造假，也不是 D4RT correspondence 完全无效，而是 **object primitive 选错了**：2D mask、surfel atom、regionlet、masklet、source-mask birth group 都不是 object。它们只是 measurements。过去的系统反复把这些 measurement 直接导出成 object，因此出现两个坏状态：

```text
small clean subset: own AP 很高，但 support 极小，cross-support 崩；
large noisy subset: support 变大，但 AP/AP50 很低，object boundary 和 one-to-one assignment 崩。
```

v15 的目标是把方法改成：

```text
D4RT material surfels
+ 2D masks / masklets / regionlets / appearance / negative evidence
-> global latent object explanation
-> posterior-controlled dense support export
-> feed-forward semantic 4D reconstruction and tracking
```

---

## 1. 最新代码审计结论

### 1.1 已做审计

我已解压最新包：

```text
/mnt/data/stream4d_v14_probe5_code_review_packet.zip
```

审计目录：

```text
/mnt/data/audit_v14/Stream3D
```

重点检查文件：

```text
Stream3D/stream4d/surfel_atom_bank.py
Stream3D/stream4d/measurement_bank.py
Stream3D/stream4d/object_explanation.py
Stream3D/stream4d/object_explanation_mdl.py
Stream3D/stream4d/export_scannet.py
Stream3D/tools/build_v14_surfel_atom_bank.py
Stream3D/tools/diagnose_v14_atom_oracle.py
Stream3D/tools/diagnose_v14_failure_decomposition.py
Stream3D/tools/scan_reportable_configs.py
Stream3D/evaluation/evaluate.py
Stream3D/tests/*.py
```

### 1.2 指标安全结论

没有发现非 oracle method 直接读取 GT 生成 prediction 的证据。v14 final scan 记录：

```text
num_reportable_method_configs = 4
num_suspicious_configs = 0
num_uses_gt_for_prediction = 0
```

metric integrity 记录：

```text
phase0_pass = True
all_ap_core_equal = True
gt_files_read_by_rescore = False
```

因此当前主要问题不是“虚假指标”。但有两个审计问题必须修正：

1. 在干净解压目录中，`python -m unittest discover tests` 失败，因为 `evaluation/evaluate.py` import 了缺失的 `evaluation.utils_3d`。这说明 v14 审计包仍不是完全 self-contained runnable。下一轮必须把 `evaluation/utils_3d.py` 和所有 evaluator 依赖打包进 zip。
2. oracle diagnostic 输出虽然被标记为 `uses_gt_for_diagnostic=true`、`is_diagnostic_only=true`，但部分 manifest 的 `uses_gt_for_prediction=false` 容易误解。凡是 GT 参与了 candidate selection / oracle output materialization，manifest 必须额外写：

```text
gt_selected_output = true
forbidden_for_method_table = true
```

普通 evaluator 和 summary table 必须拒绝任何 `gt_selected_output=true` 的行进入 method result。

### 1.3 GT / Sim3 对齐硬约束

从 v15 开始，严格区分三类对齐：

```text
1. method-internal self alignment:
   只使用 D4RT 自己的输出进行 clip-to-clip / window-to-window 对齐，允许。

2. evaluation-only GT/RGB-D alignment:
   使用 ScanNet GT mesh、RGB-D depth/pose 或 GT-derived anchors 做 Sim3，只允许在 evaluation / diagnostic / metric attribution 中使用。

3. method-internal GT/RGB-D alignment:
   禁止。任何用于 grouping、object birth、object growth、score、filter、memory 的 GT/RGB-D Sim3 都会导致结果不能作为 method result。
```

manifest 必须写：

```text
alignment_source = none | d4rt_self | gt_eval_only | rgbd_eval_only | forbidden
alignment_used_for_prediction = false
alignment_used_for_diagnostic = true/false
```

---

## 2. 最新实验结果的独立判断

### 2.1 v14 没有达到目标

v14 Phase 2 gate 明确失败。最好的 broad-support target-base A3/A4 oracle 只有：

```text
bank16 target A3/A4 oracle:
AP / AP50 / AP25 = 0.068627 / 0.117647 / 0.558824
actual pre% ≈ 3.03%
```

最宽松 `minpts5` 版本：

```text
pre% = 3.4740%
oracle AP50 = 0.114286
```

而 gate 要求：

```text
A3/A4 broad-support oracle AP50 >= 0.60
A3/A4 broad-support oracle AP25 >= 0.78
actual pre% >= 25%
best IoU >= C_hybrid + 0.08
```

差距不是小阈值能补的。

### 2.2 Stream3D 仍然明显更强

当前 probe5 上基准：

```text
Stream3D on S0:
AP / AP50 / AP25 = 0.235730 / 0.414306 / 0.537786
pre% = 84.6744%

Stream3D on S1:
AP / AP50 / AP25 = 0.399213 / 0.597171 / 0.742535
pre% = 4.5145%
```

历史 high-score rows 仍然是 tiny-support：

```text
B1 own:
0.328439 / 0.629266 / 0.884363
pre% = 3.9861%

M13c own:
0.224575 / 0.419119 / 0.781728
pre% = 4.3855%

C_surfel own:
0.228316 / 0.460285 / 0.778069
pre% = 4.2916%
```

这些不能说明完整静态场景已解决。它们说明当前系统可以找到局部 clean subset，但不能形成完整 object field。

### 2.3 Failure decomposition 揭示两个同时存在的问题

v14 failure decomposition 中，M13c/M13d 的 `pool IoU` 与 `method IoU` 差距很大：

```text
M13c:
pool IoU = 0.3231
method IoU = 0.0270
filtered_good = 52
selected_good = 1

M13d:
pool IoU = 0.3231
method IoU = 0.0100
filtered_good = 56
selected_good = 0
```

这说明 final selection/export 确实过滤掉了很多可解释候选。可是 broad-support oracle 本身也不够强：

```text
O38 oracle AP50 = 0.4444
C_mask oracle AP50 = 0.4533
C_hybrid oracle AP50 = 0.4955
```

所以不能简单说“只是 filter 太狠”。正确结论是：

```text
当前 candidate primitive 上界不足；
当前 selection / ownership 又进一步把已有好候选过滤错。
```

### 2.4 bank16 修复说明 measurement density 不是唯一瓶颈

v14 bank16 把 CropFormer mask frames 从 `2/16` 修到 `16/16`，measurement density 明显改善：

```text
unobserved surfel ratio:
v12 = 0.145764
v14 bank16 = 0.007849

target positive samples:
v12 = 26,461.8
v14 bank16 = 213,839.6

positive observation rate:
v12 = 0.854236
v14 bank16 = 0.992151
```

这说明 semantic measurement 不再严重缺帧。但 source-base atom 没变好，因为 atom birth 仍依赖旧 `src_mask_id`。target-dominant repair 让 raw known atom support 到 `86-89%`，best IoU 到约 `0.303`，但 exported `pre%` 仍只有约 `3%`，oracle AP50 仍只有约 `0.118`。

这暴露了最关键的断裂：

```text
raw surfel/atom measurement coverage 高
并不等于
exported dense object support 高。
```

### 2.5 v14 atom primitive 的根本失败

`surfel_atom_bank.py` 中 A0-A4 的核心仍是按 mask key、trajectory bin、RGB bin、risk bin 分组。也就是说，它在做：

```text
surfel grouping -> atom -> object record -> posterior_support export
```

但 `atom_to_object_record()` 仍把每个 atom 当作 object record。即使 `fringe_from_neighbors` 开启，也只是取邻近 atoms 的 surfels。`export_object_slot_posterior_support()` 最终从 surfel UV 点反投影得到 mesh support。它没有用 mask region 的 dense extent 去 materialize object，也没有把多个 atoms 全局组合成 object。

因此 atom-as-object 有两个结构性失败：

```text
1. 太碎：A3/A4 AP/AP50 接近 0，oracle 也低。
2. 太稀：raw known support 可以高，但 exported mesh pre% 只有 3% 左右。
```

### 2.6 过去路线为什么始终打不过 Stream3D

Stream3D 在 ScanNet 上强，是因为它直接在 RGB-D / pose reconstructed point cloud 上做 2D-to-3D mask projection、set-cover noise filtering、3D manifold refining。这个 inductive bias 非常适合静态 ScanNet。

当前 Stream4D 既没有保留 Stream3D 的 dense RGB-D manifold 优势，也没有真正把 D4RT 的 4D material correspondence 变成 object field。它一直在：

```text
mask / atom / carrier component / masklet -> object
```

这条路已经被 v7-v14 反复证伪。下一步必须变成：

```text
mask / atom / carrier / masklet -> measurement
object -> latent explanation over measurements
```

---

## 3. v15 总体目标

v15 的目标不是再跑一个新后处理，而是回答三个科学问题：

1. **当前观测库有没有足够高的 object upper bound？**
2. **如果有，上限损失来自 materialization、selection 还是 global object explanation？**
3. **能否构造一个 training-free 的 global measurement explanation，使 object 不再等于 mask、atom 或 regionlet？**

v15 不要求直接 full ScanNet 超过 Stream3D，但必须满足以下推进目标：

```text
Goal A: 找到 broad-support candidate pool，其 oracle AP50 >= 0.60, AP25 >= 0.78, pre% >= 25%。
Goal B: 找到 non-GT global solver，使 S0/S1 cross-support 明显超过 O38/M13c/M13d。
Goal C: 证明 D4RT-real evidence 明显优于 shuffle/no-temporal controls，不只是 tiny mask selector。
Goal D: 证明 posterior materialization 不再出现 raw known support 86% 但 exported pre% 3% 的断裂。
```

最低可接受 probe5 gate：

```text
Method own:
  AP >= 0.18
  AP50 >= 0.35
  AP25 >= 0.65
  pre% >= 12%

Method on S0:
  AP >= 0.08
  AP50 >= 0.18
  AP25 >= 0.45

Method on S1:
  AP >= 0.18
  AP50 >= 0.35
  AP25 >= 0.60

Same-support Stream3D gap:
  Stream3D-on-method-support AP - Method-own AP <= 0.08
```

如果这些最低 gate 仍不过，就不启动 tune30/final。

---

## 4. 统一评估协议

每个 method config 必须报告四行：

```text
M own:
  prediction = M
  pre_points = M

Stream3D on M:
  prediction = Stream3D baseline
  pre_points = M

M on S0:
  prediction = M
  pre_points = Stream3D S0

M on S1:
  prediction = M
  pre_points = historical 32f S1
```

如果 M 是从 parent 后处理而来，还必须报告：

```text
M inherit parent:
  prediction = M
  pre_points = parent
```

每一行必须记录：

```text
AP / AP50 / AP25
pre_points %
prediction union %
union in target scene %
union in target pre_points %
GT crop/full
#pred
mean points/object
tiny mask ratio <100 vertices
large mask ratio >1000 vertices
conflict rate
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per matched GT
runtime
manifest_integrity_pass
```

可视化必须至少包含：

```text
1. pre_points ratio bar chart
2. union ratio bar chart
3. GT crop/full bar chart
4. per-GT best IoU histogram
5. prediction-vs-GT 3D overlay for 5 failure cases
6. 2D mask/region ownership overlay for 5 failure cases
```

---

## 5. Phase 0：审计包完整性与复现实验入口

### 5.1 目标

确保下一轮人工审计能在干净目录中运行，不再出现 v14 clean unittest 缺 `evaluation.utils_3d` 的问题。

### 5.2 必须提交

每轮提交：

```text
stream4d_v15_<phase>_code_review_packet.zip
stream4d_v15_<phase>_code_review_packet.sha256
stream4d_v15_<phase>_filelist.txt
stream4d_v15_<phase>_ziptest.log
stream4d_v15_<phase>_git_diff.patch
stream4d_v15_<phase>_git_status.txt
```

zip 内必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/evaluate.py
Stream3D/evaluation/constants.py
Stream3D/evaluation/utils_3d.py
Stream3D/evaluation/__init__.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_v15_*.sh
Stream3D/docs/stream4d_v15_执行日志.md
Stream3D/docs/stream4d_v15_实验结果复盘.md
Stream3D/outputs/audit/**/*.json
Stream3D/outputs/audit/**/*.csv
Stream3D/outputs/audit/**/*.md
Stream3D/data/evaluation/scannet/*_class_agnostic.txt
必要的 probe5 prediction/TMP artifacts
```

### 5.3 运行检查

必须通过：

```bash
python -m py_compile evaluation/*.py stream4d/*.py tools/*.py tests/*.py
python -m unittest discover tests
python -m tools.scan_reportable_configs --require-manifest --require-eval-policy
python -m tools.verify_stream4d_metric_integrity --require-manifest
```

如果 `open3d`、CUDA 或 D4RT checkpoint 缺失，测试必须拆分：

```text
pure-python tests: 必须通过
gpu-required tests: 可以 skip，但必须有 skip reason
open3d-required tests: 可以 skip，但不能阻塞 protocol tests
```

### 5.4 成功标准

```text
py_compile pass
pure unittest pass
manifest scanner suspicious = 0
reportable method configs uses_gt_for_prediction = 0
oracle/diagnostic configs 不进入 method table
clean audit directory can run tests without hidden local dependencies
```

---

## 6. Phase 1：失败归因 v2，先回答“候选不够、过滤错、materialization 错，还是 solver 错”

### 6.1 目标

v14 已经证明：tiny oracle 可以高，broad oracle 不够，final method 还过滤掉 good candidates。v15 必须把失败分解到更细层级，不能再只看 AP。

### 6.2 假设

$H_{1a}$：当前失败来自 candidate primitive 上界低。

$H_{1b}$：当前 candidate primitive 上界足够，但 unsupervised solver/selection 不会选。

$H_{1c}$：candidate surfels/atoms 本身合理，但 mesh support materialization 过窄，导致 raw known support 高、export pre% 低。

### 6.3 实验

对每个 GT object，计算以下 oracle/diagnostic，不进入 method table：

#### Oracle 1：single-candidate oracle

从现有 B1/O38/C_mask/C_regionlet/C_surfel/C_masklet/C_hybrid/A0-A4/A-target candidates 中，每个 GT 只允许匹配一个 candidate。

记录：

```text
oracle AP/AP50/AP25
per-GT best IoU
no_candidate / weak / good / high counts
candidate support%
#candidate per GT at IoU>=0.10/0.25/0.50
```

#### Oracle 2：multi-atom union oracle

允许一个 GT 由多个 atoms 联合解释。对每个 GT，用 greedy set cover 从 atoms 中选最多 $K$ 个 atoms，使 IoU 最大。

设置：

```text
K = 1, 2, 4, 8, 16, 32
support source = atoms before export, exported mesh points after export
```

记录：

```text
oracle_union_AP/AP50/AP25
best IoU vs K curve
selected atom count per GT
atom purity per GT
atom fragmentation count
exported pre% after union
```

判断：

```text
如果 multi-atom union oracle 高，但 single-candidate oracle 低：
  说明 atom primitive 有潜力，global set packing 是关键。

如果 multi-atom union oracle 仍低：
  说明 atom primitive 本身不适合作为 object basis。
```

#### Oracle 3：materialization oracle

对同一组 selected atoms，比较三种导出：

```text
A. surfel-UV point export
B. owned-mask local component export
C. full owned-mask export
```

记录：

```text
raw selected surfel ratio
exported mesh pre%
GT IoU before/after materialization
boundary false-positive ratio
support expansion ratio
```

判断：

```text
如果 raw atoms 能 cover GT，但 surfel-UV export pre% 很低：
  materialization 是瓶颈。

如果 full-mask export 高但 local component export 低：
  region ownership/split 是瓶颈。

如果三者都低：
  candidate primitive 是瓶颈。
```

### 6.4 成功标准

Phase 1 必须输出一张 failure attribution 表：

```text
no primitive candidate
primitive exists but filtered
primitive selected but materialization failed
primitive selected and materialized but boundary bad
duplicate / over-merge
under-merge / fragmentation
```

每类数量必须按 GT object 汇总，并按 scene 可视化。

### 6.5 不满足条件时 Codex 的尝试方向

```text
如果 no primitive candidate > 40%：
  不要写 solver；先改 measurement primitive。

如果 materialization failed > 40%：
  先实现 mask-region local component export，不要继续 atom split。

如果 filtered_good 很多：
  实现 global set packing，不要继续 score threshold。

如果 duplicate/over-merge 多：
  增加强 negative evidence 和 mask split，不要调 NMS。
```

---

## 7. Phase 2：构建 measurement graph，不再把 atom 当 object

### 7.1 目标

构建一个 measurement graph，其中节点不是最终 object，而是可解释观测单位：

```text
surfel atom nodes
mask-region nodes
masklet nodes
appearance/semantic feature nodes
negative evidence nodes
```

object 是之后推断的 latent set，不在本 phase 直接导出。

### 7.2 假设

$H_2$：v14 A3/A4 失败不是因为 D4RT tracking 坏，而是因为 atom 直接导出为 object。若把 atoms 作为 measurement，并加入 mask-region split，可以得到更高的 broad-support oracle upper bound。

### 7.3 新 primitive：mask-region measurement

对每个 2D mask $m_{t,r}$，不要直接作为 object。先在 mask 内生成多个 region measurements：

```text
Input:
  2D mask pixels
  D4RT surfels inside mask
  RGB color boundary
  depth/normal discontinuity if RGB-D exists for ScanNet eval bridge
  distance transform to mask boundary

Output:
  region measurements R_{t,r,j}
```

region split 规则必须 training-free：

```text
1. 以 D4RT surfel seeds 为 anchors；
2. 在 2D mask 内做 watershed / geodesic distance / connected component split；
3. 不能跨过 strong depth or RGB edge；
4. 没有 reliable surfel seed 的区域标为 unknown，不直接进入 object core。
```

记录每个 region：

```text
frame_id
mask_id
region_id
pixel_count
surfel_ids inside
boundary_safe_ratio
visible_outside_negative_count
appearance feature mean
2D bbox
connected_component_count
depth_edge_crossing_score
mask_parent_area_ratio
```

### 7.4 指标

不使用 GT 的 method diagnostics：

```text
regions per mask
regions per frame
region pixel count distribution
surfel count per region
unknown region ratio
ambiguous surfel ratio
mask split count
region overlap conflict rate
```

GT-only diagnostic：

```text
region oracle AP/AP50/AP25
region purity = max GT overlap / region area
region completeness = region overlap / GT area
cross-object contamination ratio
best region IoU per GT
multi-region union oracle curve
```

### 7.5 成功标准

```text
region broad-support oracle AP50 >= 0.60
region AP25 >= 0.78
region support pre% >= 25%
region purity mean >= 0.70
cross-object contamination ratio <= 0.20
```

如果 region broad oracle 达不到上述门槛，不进入 global solver。

### 7.6 不满足条件时 Codex 的尝试方向

```text
如果 region purity 低：
  增强 depth/RGB boundary split，降低 region 跨物体概率。

如果 region completeness 低：
  允许同一 object 由多个 region 解释，不要扩大单 region。

如果 support pre% 低：
  生成 unknown regions，但仅作为 fringe candidate，不进 core。

如果 split 后 regions 过多导致 runtime 爆炸：
  先按 mask area / surfel coverage pruning，再做 per-scene cap；不要直接超时。
```

---

## 8. Phase 3：global object explanation candidate pool

### 8.1 目标

生成 object hypotheses，但 object 不再等于 mask、atom、regionlet。每个 object 是若干 measurements 的解释：

$$
O_k = \{A_i, R_j, M_l\}
$$

其中 $A_i$ 是 surfel atom，$R_j$ 是 mask-region measurement，$M_l$ 是 masklet measurement。

### 8.2 候选生成

生成三类 object proposals：

#### Birth proposals

来自 high-confidence B1/O1 clean core，但只作为 object birth：

```text
single clean mask birth
low conflict core
D4RT-real support > shuffle support
boundary-safe surfels > threshold
```

#### Growth proposals

从 birth 出发吸收 region measurements：

```text
positive evidence:
  shared surfels
  appearance similarity
  repeated mask support
  D4RT temporal consistency

negative evidence:
  visible-outside contradiction
  same-frame mask conflict
  depth/normal boundary crossing
  high mask entropy
  incompatible trajectory mode
```

#### Split proposals

如果一个 mask/region contains multiple trajectory/appearance modes，生成多个 child object hypotheses，而不是把 whole mask 归给一个 object。

### 8.3 Object score

训练自由 score：

$$
S(O) =
\lambda_p E_{pos}(O)
- \lambda_n E_{neg}(O)
+ \lambda_c C_{cover}(O)
- \lambda_o C_{overlap}(O)
- \lambda_b C_{boundary}(O)
- \lambda_s C_{size}(O)
$$

其中：

```text
E_pos: object 内 region/surfel/masklet 的正观测一致性
E_neg: visible-outside / same-frame conflict / motion incompatibility
C_cover: 能解释多少 measurements
C_overlap: 与其他 object 抢同一 surfel/region 的程度
C_boundary: 跨 boundary / high entropy risk
C_size: 过大或过小的先验惩罚
```

### 8.4 必须做的 controls

```text
D4RT-real
D4RT-shuffle
no-temporal source-only
area-only same-count
maskcount-only same-count
region-only without D4RT
```

记录：

```text
AP/AP50/AP25 own/S0/S1
oracle AP/AP50/AP25
same-support Stream3D gap
D4RT-real minus shuffle
D4RT-real minus no-temporal
conflict rate
candidate count
selected object count
measurement explained ratio
unknown ratio
```

### 8.5 成功标准

```text
D4RT-real AP50 >= shuffle + 0.10
D4RT-real AP50 >= no-temporal + 0.10
candidate oracle AP50 >= 0.60
candidate oracle AP25 >= 0.78
method on S0 AP >= 0.08
method on S1 AP >= 0.18
```

### 8.6 不满足条件时 Codex 的尝试方向

```text
如果 D4RT-real 与 shuffle 差不多：
  D4RT evidence 没有进入 object score；检查 surfel-to-region assignment 和 temporal consistency features。

如果 oracle 高但 method 低：
  solver/score 问题，进入 Phase 4 set packing。

如果 oracle 低：
  不要写 solver；回 Phase 2 改 measurement primitive。

如果 own 高但 S0/S1 崩：
  materialization/support export 仍是瓶颈，进入 Phase 5。
```

---

## 9. Phase 4：global set packing / split-merge solver

### 9.1 目标

从 Phase 3 的 proposals 中选出最终 object set。不能用 connected component，不能只用 score top-k。

### 9.2 能量函数

最终选择 object set $\mathcal{O}$，最小化：

$$
E(\mathcal{O}) =
\sum_{O_k \in \mathcal{O}} C_{obj}(O_k)
+ \alpha \sum_{m \in \mathcal{M}} C_{unexplained}(m)
+ \beta \sum_{k \neq l} C_{conflict}(O_k,O_l)
+ \gamma C_{complexity}(\mathcal{O})
$$

解释：

```text
C_obj: 单个 object hypothesis 的负 score
C_unexplained: 未被解释的重要 measurement 惩罚
C_conflict: objects 之间抢同一 region/surfel/mesh point 的惩罚
C_complexity: object 数量和碎片化惩罚
```

### 9.3 实现路线

先实现可控 greedy + local search，不直接上复杂 ILP：

```text
1. 按 object score 初始化候选集合。
2. 逐个加入 object，如果解释收益大于 overlap/complexity cost。
3. 对 selected set 做 split/merge/swap local search。
4. 允许 unknown，不强制解释所有 measurements。
5. 输出 selected/core/fringe/unknown/reject。
```

### 9.4 指标

```text
selected #objects
explained measurement ratio
unexplained high-confidence measurement count
overlap conflict rate
average objects per GT at IoU>=0.25
duplicate predictions per GT
under-merge count
over-merge count
AP/AP50/AP25 own/S0/S1
Stream3D-on-method-support gap
```

### 9.5 成功标准

```text
method AP50 own >= 0.45
method AP25 own >= 0.70
method S0 AP >= 0.08
method S1 AP >= 0.18
same-support Stream3D gap AP <= 0.08
conflict <= 0.10
missed GT count lower than O38 by >= 20%
duplicate predictions per GT <= Stream3D + 0.05
```

### 9.6 不满足条件时 Codex 的尝试方向

```text
如果 selected object count 太少：
  unexplained measurement penalty 太低，增加 C_unexplained。

如果 duplicate 太多：
  conflict cost 太低或 object proposals 过碎，增加 set-packing exclusivity。

如果 AP25 升但 AP/AP50 不升：
  boundary/materialization 问题，进入 Phase 5。

如果 S0/S1 不升但 own 升：
  support 仍太小，不允许进入 tune30。
```

---

## 10. Phase 5：posterior-controlled dense support materialization

### 10.1 目标

解决 v14 的断裂：

```text
raw atom-known support = 86-89%
actual exported pre% = 3%
```

最终 object support 不能只靠 surfel UV 点，也不能 whole-mask backproject。必须由 owned mask regions materialize：

```text
object core = high posterior surfels/regions
object fringe = local connected pixels supported by core and not contradicted
unknown = ambiguous measurements not exported
reject = negative evidence contradicted support
```

### 10.2 三种 materialization 对比

对同一 selected object set，输出三种 config：

```text
E0: surfel-point export only
E1: owned-region export
E2: owned-region + conservative fringe
E3: full-mask upper-bound diagnostic
```

E3 不作为主方法，只是说明 full-mask shortcut 上界。

### 10.3 owned-region export 规则

每个 object $O_k$ 在 frame $t$ 上有一组 owned regions：

$$
R_{k,t} = \{r: p(r \in O_k) > \tau_r, C_{neg}(r,O_k) < \tau_n\}
$$

导出时：

```text
1. 只 backproject owned regions，不 backproject entire parent mask。
2. region 内还要用 boundary distance / depth edge / D4RT seed distance 做保守裁剪。
3. 一个 mesh vertex 如果被多个 objects claim，用 object posterior WTA；不能用 GT。
4. unknown/reject 不导出。
```

### 10.4 指标

```text
raw selected surfel ratio
owned region pixel ratio
exported pre%
mesh union%
core/fringe/unknown/reject ratio
WTA removed ratio
boundary-risk exported ratio
per-GT best IoU
AP/AP50/AP25 own/S0/S1
Stream3D-on-method-support
```

### 10.5 成功标准

```text
exported pre% >= 15% on probe5
AP50 own >= E0 + 0.10
S0 AP >= O38 S0 + 0.04
S1 AP >= O38 S1 + 0.05
WTA removed ratio <= 0.35
boundary-risk exported ratio <= 0.20
```

### 10.6 不满足条件时 Codex 的尝试方向

```text
如果 exported pre% 仍低：
  owned regions 太保守；降低 region posterior threshold，但保留 negative evidence。

如果 AP 掉：
  fringe 太脏；提高 boundary/depth edge gate，或把 fringe 重新标 unknown。

如果 WTA removed > 0.6：
  ownership 在 materialization 前没有解决，回 Phase 4。
```

---

## 11. Phase 6：D4RT geometry aligned -> Stream3D diagnostic

### 11.1 目标

用户要求单独回答：D4RT 几何精度对 Stream3D-style pipeline 的影响到底多大。这个实验是 diagnostic-only，不进入 method table。

### 11.2 硬约束

```text
GT/RGB-D/ScanNet mesh Sim3 alignment 只允许用于 evaluation/testing diagnostic。
不能用于 Stream4D method 内部 object grouping、filtering、memory 或 score。
```

### 11.3 实验设置

比较：

```text
G0: 原版 Stream3D RGB-D/pose geometry
G1: Stream3D + D4RT raw geometry
G2: Stream3D + D4RT scene-level Sim3 aligned geometry, eval-only
G3: Stream3D + D4RT window-level Sim3 aligned geometry, eval-only
G4: G2 + density-normalized manifold thresholds
G5: G3 + density-normalized manifold thresholds
G6: RGB-D geometry + D4RT measurement ownership only
```

所有 D4RT geometry thresholds 必须由 D4RT point spacing quantiles 决定，不能复用 RGB-D meter-scale 超参：

```text
nn_radius = q90(nearest_neighbor_spacing) * r
manifold_delta = q75(object_internal_spacing) * d
min_points_per_object = max(5, percentile_based_density_threshold)
```

### 11.4 指标

```text
Sim3 scale mean/min/max
Sim3 residual median/p90/p95
D4RT point spacing q25/q50/q75/q90
D4RT-vs-RGBD mask IoU
projection hit rate
Stream3D AP/AP50/AP25
#objects
pre%
union%
MR kept/removed ratio
set-cover selected masks
```

### 11.5 成功标准

```text
如果 G2/G4 AP drop <= 3 AP and AP50 drop <= 5 compared with G0:
  D4RT aligned geometry 可以作为 viable geometry diagnostic。

如果 AP drop > 3 or AP50 drop > 5:
  不能把 D4RT metric geometry replacement 作为 ScanNet 主卖点；D4RT 只能作为 correspondence / measurement backbone。
```

### 11.6 不满足条件时 Codex 的尝试方向

```text
如果 0 objects：
  检查 scale-normalized thresholds，降低 min_points，记录 spacing quantiles。

如果 Sim3 residual 高：
  检查 frame stride、D4RT reference frame、source/target/camera index。

如果 residual 低但 AP 低：
  几何本身可用，但 Stream3D manifold assumptions 与 D4RT density 不匹配；改 normalized MR。
```

---

## 12. Phase 7：tune30 / final / dynamic

### 12.1 tune30 启动条件

只有 probe5 满足：

```text
candidate broad oracle pass
method own/S0/S1 minimum gate pass
D4RT-real control > shuffle/no-temporal
materialization no longer collapses raw support to tiny pre%
```

才允许跑 tune30。

### 12.2 final 规则

```text
final split 只跑一次 locked config。
不能在 final 上调阈值。
必须同时报告 own、Stream3D-on-M、M-on-S0、M-on-S1。
```

### 12.3 dynamic 规则

Dynamic Replica / Replica-Dynamic 只有在数据包含 official instance/object IDs 时才能报告：

```text
IDF1
ID switch
MOTA/MOTP if applicable
4D IoU
occlusion reactivation accuracy
track fragmentation
```

如果没有 GT IDs，只允许报告 qualitative 或 pseudo consistency，不能写 official tracking metric。

---

## 13. 并行执行安排

### Lane A：failure/oracle diagnostics

```text
负责人：Codex-A
运行 Phase 1
输出 oracle levels, per-GT failure table, visual panels
预计先完成，不依赖 solver
```

### Lane B：measurement graph / region split

```text
负责人：Codex-B
运行 Phase 2
输出 mask-region measurement bank, region oracle, split visualizations
如果 oracle 不过，不进入 solver
```

### Lane C：global object explanation solver

```text
负责人：Codex-C
等待 Lane B minimum oracle pass
实现 Phase 3/4
输出 method configs and controls
```

### Lane D：materialization

```text
负责人：Codex-D
可与 Lane C 并行，用 known selected objects做 E0-E3 对比
重点解决 raw support vs exported pre% 断裂
```

### Lane E：D4RT geometry diagnostic

```text
负责人：Codex-E
独立运行 Phase 6
输出 G0-G6 对比
严格 diagnostic-only
```

---

## 14. 本轮不再投入的方向

除非作为 ablation，不再作为主线：

```text
top-k / ratio sweep
score mode sweep
NMS / WTA 作为主创新
full-mask fringe + WTA
atom-as-object direct export
source-mask birth group as object
regionlet-as-object direct export
scene memory over already-wrong masks
own-support-only AP 报告
```

这些方向已经被 v3-v14 多轮负结果覆盖。

---

## 15. 预期 deliverables

Codex 下一轮必须交付：

```text
1. 完整 self-contained code audit packet。
2. Phase 1 failure decomposition v2。
3. Phase 2 mask-region measurement bank 和 oracle。
4. Phase 3 object proposal controls。
5. Phase 4 global set packing solver prototype。
6. Phase 5 materialization comparison。
7. Phase 6 D4RT geometry aligned -> Stream3D diagnostic。
8. 每个 method config 的 own/Stream3D-on-M/M-on-S0/M-on-S1 四行评估。
9. 每个 phase 的可视化面板。
10. 清晰的 go/no-go gate 判断。
```

---

## 16. 最终判断标准

v15 如果只得到 own-support 高分，但 S0/S1 仍崩，则判定失败。

v15 如果 only oracle 高，但 method 低，则说明 solver 失败，不能写 method success。

v15 如果 broad oracle 仍低，则说明 measurement primitive 失败，停止 solver。

v15 如果 D4RT-real 与 shuffle/no-temporal 无差异，则说明 D4RT 没有成为 object-field evidence，方法创新不成立。

v15 的真正成功应表现为：

```text
1. broad-support candidate oracle 明显强；
2. training-free solver 能选出接近 oracle 的 object set；
3. posterior materialization 能把 selected evidence 转成 dense support；
4. own 和 cross-support 同时提升；
5. D4RT-real controls 明显优于 shuffle/no-temporal；
6. 没有 GT alignment 泄漏到 method 内部。
```

这才是下一步能确实推进 feed-forward semantic 4D reconstruction and tracking 的路线。
