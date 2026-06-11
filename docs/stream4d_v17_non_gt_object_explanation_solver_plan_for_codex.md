# Stream4D v17 可执行实验计划：从 Oracle 上界走向非 GT Object Explanation Solver

面向 Codex 的执行计划。本文档的目标不是继续做 top-k、WTA、NMS、score sweep、support completion 这类后处理，也不是把某个 observation primitive 继续直接当 object。v16 给出了一个关键转折：**broad-support C_hybrid 在 multi-measurement slot oracle 下已经接近 Stream3D same-support 上界，但 v16 没有 reportable method result**。因此 v17 的核心任务是把这个 oracle 上界转化为一个 **不使用 GT 的 training-free object explanation solver**。

本文所有公式均使用 Typora 友好的 `$...$` 或 `$$...$$`，不使用方括号 display 公式语法。

---

## 0. 结论先行：v17 要解决的不是“再找一个 primitive”，而是“无 GT 地逼近 slot oracle”

v16 的关键事实如下。

```text
C_hybrid single-candidate oracle:
AP / AP50 / AP25 = 0.256256 / 0.495495 / 0.702512
support = 52.8088%

C_hybrid K8 multi-measurement union oracle:
AP / AP50 / AP25 = 0.366996 / 0.634907 / 0.764547
support = 52.8088%
selected union = 50.0378%

C_hybrid K16/K32 stress:
K16 = 0.375373 / 0.643887 / 0.769007
K32 = 0.377319 / 0.643887 / 0.769007
```

对比 Stream3D same-support diagnostic：

```text
Stream3D on S1:
AP / AP50 / AP25 = 0.399213 / 0.597171 / 0.742535
```

这说明：

```text
1. C_hybrid broad-support candidate pool 在 oracle 选择下已经有接近 Stream3D 的上界。
2. AP50/AP25 的 oracle 上界已经超过 Stream3D on S1。
3. AP 仍低于 Stream3D 约 2-3 points，但已经不是之前大 support 完全无望的状态。
4. v16 没有 reportable method result，所以方法层面的差距还没有缩小。
```

因此 v17 不应该继续等待 AP25 gate 到 0.80，也不应该继续造新的 primitive。v17 要启动一个真正的非 GT solver：

```text
Input: C_hybrid broad-support candidates + C_regionlet/C_surfel high-precision anchors + D4RT surfel measurements
Output: non-GT selected multi-measurement object slots
Goal: recover a meaningful fraction of C_hybrid K8 oracle gain
```

---

## 1. v17 的总体目标

v17 的总体目标是验证如下核心假设：

> **H-main：C_hybrid K8 oracle 的增益不是纯 GT 幻觉，而是可以被 D4RT temporal consistency、mask agreement、visible-outside negative evidence、boundary safety 和 appearance consistency 这些非 GT 信号部分恢复。**

换句话说，v17 不再问：

```text
哪个 mask / atom / regionlet 可以直接当 object？
```

而是问：

```text
对于一个 latent object slot，哪些 measurements 应该被它解释？
```

对象表示从：

```text
object = one mask / one regionlet / one surfel atom
```

改成：

```text
object = a set of measurements explained by a latent slot
```

一个 object slot $O_k$ 是一组 candidate measurements 的集合：

$$
O_k = \{z_j \mid a_{j,k}=1\}
$$

每个 measurement $z_j$ 可以来自 mask、regionlet、surfel cluster、masklet 或 hybrid candidate。它们不是 object，只是被 object slot 解释的证据。

v17 的第一阶段不做新的 mesh materializer。**先在已有 C_hybrid candidate union 空间里复现 oracle gain 的一部分**。只有非 GT solver 能在 candidate-union export 上显著提升，才进入 posterior-controlled materialization。这样可以隔离问题：

```text
如果 candidate-union solver 失败：object explanation score 错。
如果 candidate-union solver 成功但 materialization 失败：support materialization 错。
```

---

## 2. 绝对硬约束：GT / Sim3 / evaluation policy

### 2.1 方法内部禁止 GT 对齐

方法内部禁止使用 ScanNet GT instance、GT semantic、GT mesh label 或 GT/RGB-D-to-D4RT Sim3 对齐来决定：

```text
candidate 生成
candidate selection
object slot birth
object slot growth
score
ranking
NMS / set packing
support export
```

只有在评估/测试指标、diagnostic oracle、geometry attribution 时允许与 GT/RGB-D/ScanNet mesh 做 Sim3 或类似对齐。manifest 必须清楚写明用途。

允许：

```text
D4RT self-alignment
D4RT window-to-window self stitching
D4RT internal coordinate normalization
```

前提是 alignment source 只来自 D4RT 自身输出，不来自 GT geometry。

禁止：

```text
在 method result 中使用 GT Sim3 对齐后的几何或 selection。
```

### 2.2 统一评估矩阵必须固定

每个 reportable method config 必须至少输出四行：

```text
M own:
  prediction = M
  pre_points = M

Stream3D on M:
  prediction = Stream3D baseline
  pre_points = M

M on S0:
  prediction = M
  pre_points = Stream3D original S0

M on S1:
  prediction = M
  pre_points = historical 32f sparse support S1
```

如果 M 是从 parent config 后处理得到，还必须输出：

```text
M inherit parent:
  prediction = M
  pre_points = parent
```

所有表必须记录：

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

任何只报 own-support 的结果都不能作为阶段成功结论。

---

## 3. v17 的成功标准

v17 不要求一步 full ScanNet 超越 Stream3D，但必须给出一个非 GT method result，而不是只给 oracle。

### 3.1 Probe5 最小成功标准

设：

```text
C_hybrid unsup own = 0.023515 / 0.066350 / 0.133871
C_hybrid single oracle = 0.256256 / 0.495495 / 0.702512
C_hybrid K8 oracle = 0.366996 / 0.634907 / 0.764547
Stream3D on S1 = 0.399213 / 0.597171 / 0.742535
```

v17 first non-GT solver 在 probe5 必须满足：

```text
Method own:
  AP   >= 0.16
  AP50 >= 0.35
  AP25 >= 0.55
  pre_points % >= 25%

Method on S1:
  AP   >= 0.10
  AP50 >= 0.22
  AP25 >= 0.40

D4RT contribution:
  real-D4RT solver AP50 - shuffled-D4RT solver AP50 >= 0.05
  real-D4RT solver AP25 - no-temporal solver AP25 >= 0.05

No GT leakage:
  uses_gt_for_prediction = false
  is_method_result = true
  diagnostic_only = false
```

这只是最低门槛。它意味着 solver 终于从 oracle diagnostic 走向 method。

### 3.2 Probe5 强成功标准

如果要进入 tune30，必须满足：

```text
Method own:
  AP   >= 0.22
  AP50 >= 0.45
  AP25 >= 0.65
  pre_points % >= 30%

Method on S1:
  AP   >= 0.16
  AP50 >= 0.32
  AP25 >= 0.55

Same-support gap:
  Stream3D-on-M-support AP - M-own AP <= 0.10
  Stream3D-on-M-support AP50 - M-own AP50 <= 0.12
```

### 3.3 Tune30 / final 标准

如果 probe5 强标准通过，固定所有参数进入 tune30。

Tune30 gate：

```text
Method own AP >= 0.18
Method own AP50 >= 0.38
Method own AP25 >= 0.58
Method on S1 AP >= 0.12
No scene catastrophic failure rate > 30%
```

final 只允许 locked config 跑一次。final 不允许再调阈值。final 只报告真实结果，不允许删不好的 scene。

---

## 4. Phase 0：代码审计和评估安全

### 4.1 目标

保证 v17 的 method/oracle/diagnostic artifact 不混淆，避免再次出现“oracle 上界被误当 method”的风险。

### 4.2 必做修改

Codex 必须保证每个 prediction config 和 TMP config 都有 `config_manifest.json`。manifest 必须包含：

```json
{
  "algorithm_name": "...",
  "uses_gt_for_prediction": false,
  "uses_gt_for_diagnostic": false,
  "is_method_result": true,
  "is_diagnostic_only": false,
  "forbidden_for_method_table": false,
  "gt_selected_output": false,
  "eval_policy": "own_recompute_paper_style | cross_fixed_support | inherit_parent_support",
  "prediction_config": "...",
  "pre_points_config": "...",
  "support_source": "own | stream3d_s0 | stream4d_s1 | parent | named_config",
  "alignment_source": "none | d4rt_self | gt_eval_only | rgbd_eval_only",
  "sim3_alignment_used_for_prediction": false,
  "sim3_alignment_used_for_evaluation": false
}
```

Oracle configs 必须写：

```json
{
  "uses_gt_for_prediction": true,
  "uses_gt_for_diagnostic": true,
  "is_method_result": false,
  "is_diagnostic_only": true,
  "forbidden_for_method_table": true,
  "gt_selected_output": true
}
```

普通 report scanner 必须拒绝：

```text
uses_gt_for_prediction = true
is_diagnostic_only = true
forbidden_for_method_table = true
gt_selected_output = true
sim3_alignment_used_for_prediction = true
```

### 4.3 必跑检查

```bash
python -m py_compile evaluation/*.py stream4d/*.py tools/*.py tests/*.py
python -m unittest discover tests
python -m tools.scan_reportable_configs --require-manifest --require-eval-policy --reject-oracle --reject-gt-selected-output
python -m tools.verify_stream4d_metric_integrity --require-manifest
```

### 4.4 记录指标

```text
num_configs
num_reportable_method_configs
num_oracle_configs
num_diagnostic_only_configs
num_suspicious_configs
num_uses_gt_for_prediction
num_gt_selected_output_and_method_result
num_sim3_alignment_used_for_prediction
AP core hash equality
object_dict / prediction alignment mean IoU
pre_points file existence
```

### 4.5 成功标准

```text
py_compile pass
unit tests pass
num_suspicious_configs = 0
num_uses_gt_for_prediction among method configs = 0
num_gt_selected_output_and_method_result = 0
num_sim3_alignment_used_for_prediction = 0
metric integrity phase0_pass = true
```

### 4.6 不满足时 Codex 应先尝试

如果缺 manifest：自动补 manifest 只允许用于 legacy diagnostic，不允许补成 method result。新 method 必须在生成时写 manifest。

如果 unittest 因 optional dependency 失败：拆成 pure-python、open3d-required、gpu-required 三类；protocol tests 必须 pure-python 可运行。

如果 AP core hash 不一致：停止所有新实验，先回滚 evaluator AP core。

---

## 5. Phase 1：修正 measurement bank 统计口径，避免误判瓶颈

### 5.1 目标

v16 暴露出一个统计命名风险：`mean_positive_observations_per_surfel` 实际使用了 `source_positive_propagated`，而不是 `positive_observation`。v17 必须拆开统计口径，避免把 measurement density 误判成瓶颈。

### 5.2 假设

```text
H1: v16 的 mask measurement density 已经不是主瓶颈；主瓶颈是从 target measurements 中做 non-GT object selection。
```

### 5.3 实验实现

修改 `stream4d/measurement_bank.py` 或新增 `tools/diagnose_measurement_bank_v17.py`，输出以下独立统计：

```text
num_surfels
num_frames
num_mask_frames_available
uv_in01_rate
visible_ok_rate
track_length_visible_mean
self_uv_error_p90
cycle_uv_error_p90

positive_observation_count_per_surfel_mean
positive_observation_count_per_surfel_median
positive_observation_count_per_surfel_p10/p90
source_propagated_count_per_surfel_mean
source_propagated_count_per_surfel_median
target_positive_samples_total
source_positive_samples_total
surfel_positive_observation_rate
surfel_source_propagated_rate
unobserved_surfel_ratio
ambiguous_surfel_ratio
boundary_safe_surfel_ratio
```

### 5.4 判断标准

如果：

```text
surfel_positive_observation_rate >= 0.90
positive_observation_count_per_surfel_mean >= 3.0
unobserved_surfel_ratio <= 0.05
```

则 measurement density 不是主瓶颈，直接进入 solver。

如果 `surfel_positive_observation_rate < 0.70` 或 `positive_observation_count_per_surfel_mean < 1.5`，说明 semantic measurement 仍稀疏，需要补 mask cache 或增加 mask propagation。

### 5.5 不满足时 Codex 应先尝试

```text
1. 检查 16/16 mask frames 是否真的存在且被读取。
2. 检查 frame index 是否 stride-1 连续，避免 v8 早期 stride-10 问题。
3. 检查 mask cache path 和 scene/frame mapping。
4. 若 mask cache 只有少数帧，先生成或补齐 predicted masks；不要用 GT masks。
5. 若 D4RT projection 正常但 mask assignment 低，检查 UV normalization 和 image resize/crop。
```

---

## 6. Phase 2：Oracle-selected vs rejected 的非 GT 特征差异分析

### 6.1 目标

v16 证明 C_hybrid K8 oracle 有上界，但没有回答：oracle 选中的 measurements 是否能被非 GT 特征区分。Phase 2 要把 oracle selection 当作 diagnostic label，分析 selected 与 rejected candidate 的非 GT 特征差异。

这一步使用 GT 只做 diagnostic，不输出 method prediction。

### 6.2 假设

```text
H2: Oracle-selected candidates 在非 GT 特征上与 rejected candidates 有可分性。
```

如果 H2 不成立，就说明 C_hybrid K8 upper bound 很可能只能由 GT 才能选出，非 GT solver 很难逼近。

### 6.3 候选特征

对每个 candidate measurement $z_j$ 计算：

```text
area_points
area_pixels
num_surfels
num_frames_supported
num_source_masks
num_target_masks
mean_visibility
mean_confidence
mean_boundary_safe
boundary_risk
ambiguous_surfel_ratio
visible_outside_negative_rate
mask_agreement_score
d4rt_temporal_consistency
cycle_consistency_score
appearance_variance_rgb
appearance_variance_dino_or_clip_if_available
overlap_with_seed_core
incremental_area_ratio
incremental_new_support_ratio
overlap_with_existing_slots
same_frame_conflict_count
candidate_conflict_rate
mask_fragmentation_score
```

其中不允许使用 GT IoU 或 GT category 作为特征。

### 6.4 诊断方法

对 C_hybrid K8/K16 oracle selected candidates 和 non-selected candidates，输出：

```text
feature mean/median/std for selected
feature mean/median/std for rejected
selected/rejected ratio
AUC of each scalar feature
Spearman correlation with oracle marginal gain
simple rank-score top-k recovery using each feature alone
```

可以使用 GT oracle label 计算 AUC，但这只是 diagnostic。生成的任何 classifier/threshold 不得直接进入 method table，除非重新以无 GT 固定规则实现。

### 6.5 成功标准

至少满足一条：

```text
1. 有单个非 GT feature 的 selected-vs-rejected AUC >= 0.70。
2. 有 3 个以上 feature AUC >= 0.62，且方向一致。
3. 非 GT linear score 在 leave-one-scene-out diagnostic 中能恢复 K8 oracle AP50 gain 的 >=30%。
```

### 6.6 不满足时 Codex 应先尝试

如果所有特征 AUC 接近 0.5：

```text
1. 增加 object-slot conditioned features，而不是 candidate-only features。
2. 计算 candidate 加入当前 seed slot 后的 marginal change，而不是候选自身质量。
3. 加入 appearance consistency：RGB histogram、DINO/CLIP mask crop feature，若特征已缓存则优先使用缓存。
4. 加入 mask-level split cues：depth/normal discontinuity、mask boundary distance transform。
5. 若仍无可分性，停止 solver，回到 measurement primitive 设计。
```

---

## 7. Phase 3：非 GT Candidate-Union Object Explanation Solver v1

### 7.1 目标

实现第一个 reportable non-GT solver，直接在 C_hybrid candidate union 空间中选择/组合 measurements，暂时不做新的 mesh materialization。

该阶段的目标不是一步超过 Stream3D，而是证明：

```text
非 GT self-consistency score 可以恢复 C_hybrid K8 oracle gain 的一部分。
```

### 7.2 核心表示

每个 candidate measurement $z_j$ 包含：

```text
support point set V_j
mask frame / mask id
source/target origin
surfel ids
2D region pixels if available
score features
D4RT consistency features
negative evidence features
appearance features
```

每个 object slot $O_k$ 是若干 measurement 的集合。slot 的支持为：

$$
V(O_k)=\bigcup_{z_j \in O_k} V_j
$$

### 7.3 Solver 形式

不要 connected component。使用 seed + beam + global set packing。

#### Step 1：object seed

使用 high-precision anchors 建 seed：

```text
C_regionlet high-score candidates
C_surfel high-confidence tiny objects
B1/O1 clean single-mask cores
```

这些 seed 只用于 birth，不直接作为最终 object。

#### Step 2：candidate neighborhood

对每个 seed $O_k$，候选 neighborhood 包括：

```text
与 seed 有 surfel overlap 的 C_hybrid candidates
与 seed 有 2D mask relation 的 candidates
与 seed 在 3D/RGB-D bridge 上邻近的 candidates
与 seed appearance 一致的 candidates
```

#### Step 3：marginal self-consistency score

加入候选 $z_j$ 的增量分数：

$$
\Delta(z_j \mid O_k)=
\alpha E^+(z_j,O_k)
-\beta E^-(z_j,O_k)
-\gamma R_{boundary}(z_j,O_k)
-\eta R_{overlap}(z_j,O_k)
-\kappa R_{complexity}(O_k \cup z_j)
$$

其中：

```text
E+ = temporal agreement + mask agreement + appearance consistency + new explained support
E- = visible-outside negative + same-frame cannot-link + incompatible motion/geometry
R_boundary = high boundary risk / crossing depth discontinuity
R_overlap = stealing support already explained by another confident slot
R_complexity = too many measurements / too many frames without new support / huge area jump
```

所有项必须是非 GT。

#### Step 4：beam growth

对每个 seed：

```text
beam width = 4 或 8
max measurements per slot = 8 initially
stop if marginal score <= 0
stop if negative evidence exceeds threshold
stop if area expansion ratio too large without enough new agreement
```

#### Step 5：global packing

多个 slots 之间做 set packing：

$$
S(\mathcal{O})=\sum_k S(O_k)-\lambda\sum_{k<l}\mathrm{overlap}(O_k,O_l)-\mu |\mathcal{O}|
$$

不需要训练。可以用 greedy with rollback 或 scipy MILP。如果使用 MILP，必须保持 deterministic，并记录 solver status。

### 7.4 必跑 ablations

```text
M17-real: full solver with real D4RT features
M17-shuffle: shuffle surfel/mask association before score
M17-no-temporal: remove D4RT temporal consistency terms
M17-no-negative: remove visible-outside negative evidence
M17-area-only: rank/grow by area and overlap only
M17-random-same-count: random selected candidate count matched to M17-real
```

### 7.5 记录指标

除了统一四行评估外，还必须记录：

```text
num_seeds
num_candidate_neighbors_per_seed
mean beam size
mean selected measurements per object
slot area expansion ratio
new support ratio per added measurement
positive evidence mean
negative evidence mean
boundary risk mean
appearance consistency mean
D4RT consistency mean
packing overlap penalty
#slots before packing
#slots after packing
#measurements selected
#measurements rejected
selection runtime
```

### 7.6 成功标准

Probe5 minimum pass：

```text
M17-real own AP   >= 0.16
M17-real own AP50 >= 0.35
M17-real own AP25 >= 0.55
M17-real pre% >= 25

M17-real on S1 AP   >= 0.10
M17-real on S1 AP50 >= 0.22
M17-real on S1 AP25 >= 0.40

M17-real AP50 - M17-shuffle AP50 >= 0.05
M17-real AP25 - M17-no-temporal AP25 >= 0.05
```

Strong pass：

```text
M17-real own AP   >= 0.22
M17-real own AP50 >= 0.45
M17-real own AP25 >= 0.65
M17-real on S1 AP >= 0.16
```

### 7.7 不满足时 Codex 应先尝试

如果 M17-real 接近 controls：

```text
1. 输出 oracle-selected vs solver-selected feature diff，检查 score 是否方向错误。
2. 调低正证据权重，调高 visible-outside negative。
3. 把 candidate-level score 改成 slot-conditioned marginal score。
4. 增加 seed purity gate，避免错误 seed 吞噪声。
5. 若 D4RT-real 与 shuffle 仍无差异，说明 D4RT features 没被正确接入，检查 feature join key。
```

如果 own AP 上升但 S1/S0 仍接近 0：

```text
1. support 仍过小，允许 C_hybrid broad candidates 作为 growth source。
2. 检查 selected union % 是否低于 20%。
3. 检查每个 object selected measurement 数是否过少。
4. 不要直接 full-mask backproject；先扩大 candidate-union selection。
```

如果 AP25 上升但 AP/AP50 不升：

```text
1. coarse object 找到了，但边界差。
2. 增强 boundary risk 和 same-frame conflict。
3. 允许一个 broad mask 被 split-explain，不允许整张 mask 直接进 object。
```

---

## 8. Phase 4：Oracle 行为恢复率诊断

### 8.1 目标

把非 GT solver 与 C_hybrid K8 oracle 对齐，量化 solver 到底恢复了多少 oracle gain。

### 8.2 指标定义

对指标 $m \in \{AP, AP50, AP25\}$：

$$
\mathrm{Recovery}_m = \frac{m(M17)-m(B)}{m(O_{K8})-m(B)}
$$

其中：

```text
B = chosen non-GT baseline, e.g. C_hybrid unsup or area-only
O_K8 = C_hybrid K8 oracle
M17 = non-GT solver
```

同时记录：

```text
oracle_selected_count
solver_selected_count
selected_overlap_with_oracle
oracle_recall_by_solver
solver_precision_vs_oracle
```

这些只用于 diagnostic，不进入 method scoring。

### 8.3 成功标准

```text
AP50 recovery >= 0.40
AP25 recovery >= 0.40
oracle selected candidate recall >= 0.35
solver selected candidate precision vs oracle >= 0.30
```

如果 AP 指标没过但 oracle-overlap 指标较高，说明 export/evaluator support 可能是瓶颈；进入 Phase 5 materialization 诊断。

如果 oracle-overlap 很低，说明 solver score 没学到 oracle 行为；回 Phase 2 特征分析。

---

## 9. Phase 5：Posterior-controlled materialization，仅在 solver 成功后启动

### 9.1 启动条件

只有 Phase 3 达到 minimum pass，才启动 Phase 5。否则不要继续写 materializer。

### 9.2 目标

当前 owned-region materialization 很差，v16 里宽松 R0b exported pre 只有约 5.63%，AP50 约 0.000475，purity 也不够。Phase 5 要在 solver 确认可行后，解决 candidate-union 到 posterior mesh support 的导出问题。

### 9.3 方法

对每个 selected slot $O_k$，每个 frame/mask 内部做 object-conditioned decomposition：

```text
core: high posterior surfels / high agreement region
fringe: boundary-safe region connected to core
unknown: mask interior but no object-specific evidence
reject: visible-outside negative or conflicting object evidence
```

最终导出：

```text
core + conservative fringe
```

禁止：

```text
entire full-mask backprojection as final support
```

### 9.4 记录指标

```text
core points
fringe points
unknown points
reject points
core/fringe ratio
mask pixels kept ratio
boundary kept ratio
visible-outside rejection ratio
materialized pre%
materialized union%
purity diagnostic if GT allowed
contamination diagnostic if GT allowed
AP/AP50/AP25 under own/S0/S1
```

### 9.5 成功标准

```text
Compared to candidate-union M17:
  AP50 drop <= 0.05
  AP25 drop <= 0.08
  pre% remains >= 70% of candidate-union pre%
  conflict rate does not increase by > 2x
```

如果 materialization AP 崩：

```text
1. 回滚到 candidate-union export作为 method result。
2. 只把 materialization 作为 diagnostic。
3. 检查 region-to-mesh hit rate、boundary erosion、depth discontinuity。
4. 不要用 GT Sim3 修 method export。
```

---

## 10. Phase 6：D4RT 几何对齐后给 Stream3D 使用的归因实验

### 10.1 目标

单独判断 D4RT metric geometry 精度对 Stream3D-style pipeline 的影响。这个实验是 diagnostic-only，不进入 method table。

### 10.2 GT 对齐规则

只允许在该 diagnostic 中使用 GT/RGB-D/ScanNet geometry 做 Sim3：

```text
allowed: evaluation/testing diagnostic
forbidden: method grouping / object selection / score / reportable method
```

manifest 必须写：

```json
{
  "uses_gt_for_prediction": false,
  "uses_gt_for_diagnostic": true,
  "is_method_result": false,
  "is_diagnostic_only": true,
  "alignment_source": "gt_eval_only",
  "sim3_alignment_used_for_evaluation": true,
  "sim3_alignment_used_for_prediction": false
}
```

### 10.3 实验配置

```text
G0: original Stream3D RGB-D/pose geometry
G1: D4RT raw geometry + density-normalized thresholds
G2: D4RT scene-level Sim3 aligned geometry -> Stream3D geometry adapter
G3: D4RT window-level Sim3 aligned geometry -> Stream3D geometry adapter
G4: D4RT scene Sim3 + normalized manifold threshold
G5: D4RT window Sim3 + normalized manifold threshold
G6: RGB-D geometry + D4RT identity evidence only
```

### 10.4 关键要求

这次必须是真的 Stream3D geometry adapter：

```text
D4RT-aligned per-frame point/depth-like geometry feeds Stream3D local projection / set cover / manifold path.
```

不能只是：

```text
D4RT points aligned后按 2D mask key 投到 mesh 再导出 prediction。
```

### 10.5 记录指标

```text
Sim3 scale mean/min/max
Sim3 residual median/p90/p95
anchor count
D4RT point spacing p10/p50/p90
NN radius in normalized units
projection hit rate
mask-to-geometry support count
Stream3D AP/AP50/AP25 under own/S0/S1
AP drop from G0
AP50 drop from G0
manifold refinement rejection rate
set-cover selected mask count
failure scene list
```

### 10.6 判断标准

如果：

```text
G2/G3 AP drop <= 0.05 absolute
G2/G3 AP50 drop <= 0.08 absolute
```

说明 D4RT aligned geometry 有潜力支撑 Stream3D-style geometry。

如果：

```text
AP drop > 0.10 or AP50 drop > 0.15
```

说明 D4RT metric geometry 仍不能作为 ScanNet 主路径。论文叙事应坚持：

```text
D4RT is correspondence / material identity backbone, not ScanNet RGB-D metric replacement.
```

### 10.7 不满足时 Codex 应先尝试

```text
1. 检查 frame stride 是否连续，不能复用 stride-10 失败缓存。
2. 检查 D4RT normalized UV 与 ScanNet RGB resolution 的映射。
3. 用 D4RT point spacing quantile 设置 NN radius，不复用 RGB-D meter-scale超参。
4. 分 scene-level Sim3 与 window-level Sim3 比较。
5. 检查 Sim3 anchor 是否来自可见、高置信、非边界点。
6. 若 residual 仍高，停止 geometry replacement，不要继续调 Stream3D 阈值。
```

---

## 11. Phase 7：Tune30 / Final / Dynamic Replica

### 11.1 Tune30

启动条件：Phase 3 strong pass。

固定 probe5 参数，在 tune30 跑：

```text
M17-real
M17-shuffle
M17-no-temporal
M17-no-negative
best legacy baseline
Stream3D baseline
```

必须输出统一四行矩阵和 per-scene failure decomposition。

### 11.2 Final

启动条件：tune30 gate 通过。

final 只跑一次 locked config。不能根据 final 结果回调参数。

### 11.3 Dynamic Replica

动态实验只有在数据检查通过时启动。

必须确认：

```text
RGB frames exist
camera trajectory exists
depth exists if required
instance/object ID GT exists for official tracking
semantic GT exists if reporting semantic AP
```

如果没有 object ID GT：

```text
不能报告 IDF1 / MOTA / official 4D IoU。
只能报告 qualitative 或 pseudo-consistency diagnostic。
```

动态指标包括：

```text
IDF1
ID switches
fragmentation
track purity
occlusion reactivation count
object support temporal consistency
D4RT trajectory consistency
open-vocabulary time-sensitive query examples
```

---

## 12. 必做可视化

每个 reportable solver 至少输出以下可视化：

```text
1. oracle selected vs solver selected candidate overlay
2. selected/rejected candidate feature histograms
3. per-object beam growth sequence
4. visible-outside negative evidence heatmap
5. support ownership map: core / fringe / unknown / reject
6. failure GT panels: no candidate / wrong assignment / boundary bad / duplicate
7. D4RT real vs shuffled comparison panels
8. Stream3D prediction vs M17 prediction on the same support
```

每个可视化必须有 manifest：

```json
{
  "uses_gt_for_visualization": true_or_false,
  "diagnostic_only": true_or_false,
  "scene": "...",
  "config": "..."
}
```

使用 GT 的图只能用于 diagnostic，不得放入 method qualitative unless explicitly marked。

---

## 13. v17 Codex 最低交付物

Codex 必须交付：

```text
1. 完整 code audit packet zip + sha256 + filelist + ziptest。
2. Phase 0 manifest / metric integrity 报告。
3. Phase 1 measurement bank fixed statistics。
4. Phase 2 oracle-selected vs rejected feature analysis。
5. Phase 3 first non-GT solver M17-real + controls。
6. Unified eval matrix: M own, Stream3D on M, M on S0, M on S1。
7. Oracle recovery report。
8. D4RT real vs shuffle/no-temporal controls。
9. If solver passes minimum gate: Phase 5 materialization diagnostic。
10. If time permits: Phase 6 D4RT geometry -> Stream3D diagnostic。
```

不得交付：

```text
1. 只有 oracle、没有 method 的结果。
2. 只有 own-support、没有 cross-support 的结果。
3. 用 GT Sim3 参与 method selection 的结果。
4. 没有 manifest 的 prediction/TMP。
5. 没有 failure decomposition 的 AP 表。
```

---

## 14. v17 的真正判断标准

v17 成功不是指“某个 own-support AP 好看”。v17 成功必须满足：

```text
1. 第一次产生 broad-support non-GT object explanation method result。
2. 该 method 明显优于 area/random/shuffle/no-temporal controls。
3. 该 method 能恢复 C_hybrid K8 oracle gain 的有意义部分。
4. 该 method 在 S1/S0 cross-support 下不再接近 0。
5. 所有结果可审计、无 GT leakage、无 oracle 混入。
```

如果这些都不成立，结论必须写成：

```text
C_hybrid oracle upper bound requires GT-like selection and current non-GT evidence is insufficient.
```

这时应停止 solver 堆叠，回到 measurement primitive / feature design，而不是继续调阈值。
