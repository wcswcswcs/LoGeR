# Stream4D v11 深度算法重启计划：统一评估、object primitive 归因、D4RT-native surfel ownership 与 D4RT 几何归因

日期：2026-06-09  
面向：Codex 执行与人工审计  
适用对象：当前 Stream4D v10 之后的下一轮实验  
公式格式：本文只使用 Typora 友好的 `$...$` 和 `$$...$$`，不使用 `\[\]`。

---

## 0. 这份计划为什么必须重写

当前实验没有达到目标。更重要的是，失败不是某个阈值、score、NMS、WTA、regionlet 半径、Sim3 radius 或 `pre_points` policy 的小问题，而是算法表示层面的问题。

过去几轮已经出现了两类互相矛盾但都真实的结果：

```text
small clean support -> own/recompute AP 很高，但换 support 后崩。
large support -> coverage 上来，但 object AP 很低。
```

这说明当前方法没有解决核心问题：**如何从 noisy 2D mask measurements 和 D4RT correspondence 中形成正确的一对一 object instances**。当前很多实现仍然把 2D mask、regionlet 或 carrier component 当成 object primitive，然后靠后处理修补。这个思路已经被 v7-v10 的结果反复否定。

v11 的目标不是继续修补旧 pipeline，而是建立一个能回答下面三个问题的最小闭环：

```text
Q1: 当前候选空间本身有没有足够的上界？
Q2: D4RT surfel/correspondence 对 object ownership 是否真的提供不可替代的信息？
Q3: 若 D4RT 几何通过 GT/RGB-D anchors 对齐后喂给原版 Stream3D，性能到底掉多少？
```

只有回答这三个问题，才能决定后续是继续 D4RT-native semantic 4D field，还是必须引入更强的 2D video masklet prior，或者重新定位论文主张。

---

## 1. 当前实验结果的独立判断

### 1.1 统一评估矩阵已经证明：own-support 好看不等于 object quality 成立

v10 Phase 0 的统一矩阵给出了当前最重要的事实：

```text
P0 Stream3D on S0:
AP / AP50 / AP25 = 0.235730 / 0.414306 / 0.537786
pre% = 84.6744

P0 Stream3D on S1:
AP / AP50 / AP25 = 0.399213 / 0.597171 / 0.742535
pre% = 4.5145

B1 own:
AP / AP50 / AP25 = 0.328439 / 0.629266 / 0.884363
pre% = 3.9861
GT crop/full = 8.20 / 40.60

B1 on S0:
AP / AP50 / AP25 = 0.000635 / 0.004294 / 0.010768

B1 on S1:
AP / AP50 / AP25 = 0.016837 / 0.047534 / 0.168162
```

B1 的 own AP 很强，但它只覆盖了极小的 support，并且只让 8.2/40.6 个 GT instances 进入评价。更关键的是，同一 B1 support 上的 Stream3D 也很强：

```text
P0 Stream3D on S2/B1 support:
AP / AP50 / AP25 = 0.326714 / 0.496778 / 0.726638

B1 own:
AP / AP50 / AP25 = 0.328439 / 0.629266 / 0.884363
```

因此 B1 只能说明：D4RT surfel ownership 对小 support 上的高置信 object subset 有正信号，尤其 AP50/AP25 很强；但它不能说明完整静态场景 segmentation 已经解决。

O38 是另一个反例：

```text
O38 c055 own:
AP / AP50 / AP25 = 0.081038 / 0.219225 / 0.492501
pre% = 66.6809
GT crop/full = 40.2 / 40.6

O38 c055 on S0:
AP / AP50 / AP25 = 0.033012 / 0.123089 / 0.392066
```

O38 的 support 大了，但 AP 仍然远低于 Stream3D。这说明：**问题已经不是单纯 coverage 小，而是 object hypothesis 本身错了。**

### 1.2 Regionlet v10 没有通过：它把 measurement 错当 object

v10 Regionlet Birth 的结果很清楚：

```text
R0 full-mask own:
0.136117 / 0.316625 / 0.426390
conflict = 70.2060%

R1 mask-core own:
0.172123 / 0.291682 / 0.421048
conflict = 67.4841%

R2 depth-split own:
0.000168 / 0.001392 / 0.121278

R3 D4RT-seeded own:
0.142857 / 0.256714 / 0.358752
pre% = 1.5783

R4 combined own:
0.014373 / 0.059587 / 0.168899
```

Regionlet Repair 加 WTA 后把 conflict 清零，但 AP 没有变好：

```text
R0b full-mask 32f WTA own:
0.021122 / 0.098676 / 0.331286

R1b mask-core 32f WTA own:
0.033785 / 0.127959 / 0.395010

R4b combined 32f WTA own:
0.000262 / 0.001621 / 0.104822
```

WTA 能清冲突，但不能把错误 primitive 变成正确 object。regionlet 当前只是把 2D mask 切碎，然后每个 regionlet 仍然被当作 object 或近似 object 导出。真正应该做的是：**regionlet 是 measurement，不是 object。object 是对 regionlets/surfels 的解释。**

### 1.3 D4RT 几何实验说明：correspondence 可用，metric geometry 仍不够，且 v10 几何实验还不是真正 Stream3D geometry replacement

v10 fresh D4RT carriers 的 image-space 指标不错：

```text
uv_in01_rate_mean = 0.985845
track_length_visible_mean = 13.2847
self_uv_error_p90_mean = 1.5708 px
cycle_uv_error_p90_mean = 3.2737 px
surfel_coverage_2d_per_frame_mean = 0.1320
```

这说明 D4RT 连续帧 correspondence 不是主要 blocker。

但 D4RT metric geometry 对 ScanNet mesh 仍然弱。v10 G2/G4 scene-level Sim3 后：

```text
G2/G4 own:
AP / AP50 / AP25 = 0.188825 / 0.364384 / 0.486341
pre% = 1.4370
conflict = 75.1744%

P0 Stream3D on G2/G4 support:
AP / AP50 / AP25 = 0.314343 / 0.413636 / 0.552727

G2/G4 on S0:
AP / AP50 / AP25 ≈ 0.000000 / 0.000000 / 0.000112
```

这个结果说明两个问题：

1. D4RT-aligned geometry 在 tiny support 上有非零 AP，但远弱于 Stream3D 在同 support 上的 object quality。
2. 当前 v10 的 `materialize_d4rt_aligned_geometry_for_stream3d.py` 不是完整“把 D4RT geometry 喂给原版 Stream3D pipeline”。它更像：D4RT points 按 2D mask key 投到 ScanNet mesh，再导出 3D masks。它没有完整替换 Stream3D 内部的 local point cloud、set-cover、manifold refining 和 historical update。

因此 v11 必须补一个真正的归因实验：**把 D4RT-aligned depth/point map 做成 Stream3D 可消费的 geometry source，再跑原版 Stream3D local+historical pipeline**。

---

## 2. v11 的核心研究判断

### 2.1 当前错的假设

当前算法里有五个假设需要废弃或降级：

```text
错误假设 A: 2D mask 是 object primitive。
错误假设 B: regionlet 是 object primitive。
错误假设 C: carrier/surfel co-membership 可以直接连成 object component。
错误假设 D: WTA/exclusivity 可以在最后修复错误 object birth。
错误假设 E: 支持域越大，AP 自然会接近 Stream3D。
```

v10 已经证明这些假设不成立。大 support 的 O38 仍低，regionlet repair 后 conflict 为 0 仍低，B1/O1 tiny support 高但跨 support 崩。

### 2.2 仍然正确的假设

仍然有三个假设是对的，应该保留：

```text
正确假设 A: D4RT image-space correspondence 有用。
证据：连续帧 uv/cycle 指标很好，B1 相比 no-track/shuffle/random/area controls 有明显提升。

正确假设 B: clean mask birth 是有用 object prior。
证据：B1 single-mask ownership 在 tiny support 上 AP50/AP25 很强。

正确假设 C: own/cross-support 同时汇报是必须的。
证据：B1/O1 own 很高但 S0/S1 崩；O38 support 大但 object quality 低。
```

### 2.3 v11 新方向

v11 的主线不是继续做 regionlet object birth，而是：

```text
2D mask / regionlet / D4RT surfel 都只是 measurements。
object 是 latent slot，用来解释这些 measurements。
```

新的算法对象是 object slot $O_k$，不是 mask、regionlet 或 carrier component。每个 surfel/regionlet 对每个 object slot 只有 posterior ownership：

$$
p(y_i=k \mid \mathcal{Z}, \mathcal{G}, \mathcal{A})
$$

其中：

```text
\mathcal{Z}: 2D mask / regionlet / masklet observations
\mathcal{G}: D4RT correspondence, visibility, cycle/self consistency, geometry
\mathcal{A}: appearance evidence, frozen image features, color/texture statistics
```

核心能量函数不是 connected component，而是 observation explanation：

$$
E(Y, O)=
\sum_i \psi_i(y_i)
+\sum_{i,j}\psi_{ij}(y_i,y_j)
+\sum_k \Omega(O_k)
+\sum_m \Phi(m, O)
$$

这里 $Y$ 是 surfel/regionlet ownership，$O$ 是 object slots。$\Phi(m,O)$ 表示一个 2D mask measurement 可以被一个或多个 object slots 解释，不能默认整个 mask 属于一个 object。

---

## 3. 每轮代码审计硬要求

当前 v10 审计包存在严重不足：解压后缺少 `Stream3D/stream4d/*.py`、`Stream3D/data/prediction` 和 `Stream3D/data/TMP`，导致我只能审局部工具脚本和结果 summaries，不能从 zip 中完整复跑。`python -m py_compile evaluation/evaluate.py tools/*.py` 通过，但 `python -m unittest discover tests` 在干净解压目录中因为缺少 `tools.oracle_candidate_upper_bound` 和 `stream4d.*` 模块失败。

v11 每轮必须提交完整审计包：

```text
stream4d_v11_<phase>_code_review_packet.zip
stream4d_v11_<phase>_code_review_packet.sha256
stream4d_v11_<phase>_filelist.txt
stream4d_v11_<phase>_ziptest.log
stream4d_v11_<phase>_git_diff.patch
stream4d_v11_<phase>_git_status.txt
```

zip 内必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/evaluate.py
Stream3D/tests/*.py
Stream3D/scripts/reproduce_v11_*.sh
Stream3D/scripts/*.json
Stream3D/docs/stream4d_v11_执行日志.md
Stream3D/docs/stream4d_v11_实验结果复盘.md
Stream3D/outputs/audit/v11_*/*.json
Stream3D/outputs/audit/v11_*/*.md
Stream3D/outputs/audit/v11_*/*.csv
Stream3D/data/evaluation/scannet/*_class_agnostic.txt
Stream3D/data/prediction/<new_configs>_class_agnostic/config_manifest.json
Stream3D/data/TMP/<new_configs>/config_manifest.json
probe5 prediction/TMP artifacts sufficient for rerun
```

必须在干净解压目录通过：

```text
python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
python -m unittest discover tests
python -m tools.scan_reportable_configs --require-manifest --require-eval-policy
python -m tools.verify_stream4d_metric_integrity --require-manifest
```

如果缺少可选依赖，例如 `open3d`，必须把测试拆成：

```text
pure_python_tests
open3d_required_tests
gpu_required_tests
```

不能让 import 阶段因为可选依赖缺失导致 protocol tests 全部不可运行。

---

## 4. 统一评估协议：以后所有实验必须这样报

每个新方法 $M$ 必须输出以下行，不允许只报 own support：

```text
M own:
  prediction = M
  pre_points = M

Stream3D on M support:
  prediction = Stream3D baseline P0
  pre_points = M

M on Stream3D S0:
  prediction = M
  pre_points = Stream3D baseline S0

M on historical S1:
  prediction = M
  pre_points = S1 historical 32f support

M inherit parent, if applicable:
  prediction = M
  pre_points = parent config
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
conflict rate
tiny mask ratio <100
large mask ratio >1000
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per GT
runtime
manifest_integrity_pass
```

v11 的最低进展门槛不再看 own AP，而看 cross-support：

```text
S0 Gate:
AP   >= 0.08
AP50 >= 0.18
AP25 >= 0.45

S1 Gate:
AP   >= 0.18
AP50 >= 0.35
AP25 >= 0.60

Same-support Gap Gate:
Stream3D on M support - M own AP < 0.08
```

这些门槛仍低于 Stream3D，但它们能证明算法在正确方向上前进。

---

## 5. Phase 1：候选空间上界诊断，先判断 primitive 是否值得继续

### 5.1 目标

在继续设计新算法前，必须知道当前候选空间本身有没有潜力。如果 candidate pool 的 GT oracle 上界都低于 Stream3D，那么后续再调 ownership 没意义；必须回到 2D masklet/regionlet generation。若 oracle 上界很高，而无监督方法低，说明主要问题是 inference/object assignment。

### 5.2 假设

$H_{C1}$：当前 full-mask、regionlet、B1/O1/O38 候选池中，存在足够高 IoU 的候选；失败主要来自无监督 selection/partition。

$H_{C2}$：如果 oracle 上界也低，则当前 object primitive 无法表达 GT object，必须换 primitive。

### 5.3 实验设置

构造候选池：

```text
C0: Stream3D baseline predictions, diagnostic reference
C1: B1/O1 tiny clean candidates
C2: O38 large-support memory candidates
C3: R0/R1/R2/R3/R4 regionlet candidates
C4: R0b/R1b/R4b WTA candidates
C5: union of C1+C2+C3 after duplicate removal
C6: newly generated masklet/regionlet candidates from Phase 3, if available
```

运行两个 diagnostic-only oracle：

```text
Oracle-A: per-GT best candidate IoU upper bound
Oracle-B: set-packing AP upper bound with one prediction per GT and duplicate penalty
```

禁止 oracle 输出进入 method table。manifest 必须写：

```text
uses_gt_for_prediction = true
is_diagnostic_only = true
is_method_result = false
```

### 5.4 必须记录指标

```text
candidate_count
candidate_pool_union %
per-GT best IoU mean
per-GT best IoU median
GT covered at IoU>=0.25
GT covered at IoU>=0.50
GT covered at IoU>=0.75
oracle AP/AP50/AP25 upper bound
#oracle_selected
candidate duplication per GT
candidate conflict rate
candidate area distribution
missed GT object size distribution
missed GT category if labels available only for diagnostic
```

### 5.5 判断标准

如果：

```text
Oracle-B AP50 < Stream3D AP50 - 0.05
```

则当前候选池没有足够表达能力，Codex 不要继续做 selection、score、memory，必须转向 Phase 3 的 masklet/measurement generation。

如果：

```text
Oracle-B AP50 >= Stream3D AP50 + 0.05
method AP50 << Oracle-B AP50
```

则候选池有潜力，主要问题是 ownership inference。Codex 进入 Phase 4。

如果：

```text
Oracle-A 高但 Oracle-B 低
```

说明候选之间重叠/冲突严重，单个 GT 有好候选，但无法组成一致 object set。Codex 必须优先实现 global set-packing / object slot competition，而不是增加候选。

### 5.6 可视化

每个 scene 输出：

```text
missed GT object -> best candidate overlay
GT with multiple duplicate candidates overlay
oracle-selected candidates overlay
false-positive candidate examples
candidate IoU histogram
candidate area vs IoU scatter plot
```

### 5.7 不满足条件时 Codex 先尝试

如果 candidate upper bound 低：

```text
增加 2D mask measurement density。
引入 video masklet propagation。
按 depth/normal boundary 重新切 mask，但不得直接把 regionlet 当 object。
提高 D4RT surfel density。
检查 mask cache 是否只覆盖 2/16 frames。
```

如果 candidate upper bound 高但 method 低：

```text
停止 candidate generation，转 Phase 4 ownership posterior。
实现 object-level set packing。
加入 negative evidence 和 cannot-link。
```

---

## 6. Phase 2：D4RT correspondence 与 semantic measurement density 诊断

### 6.1 目标

当前 v8/v10 显示 D4RT continuous-frame tracks 是可用的，但 semantic masks 很稀疏，历史上有 16 帧仅 2 帧有 CropFormer masks 的情况。v11 必须把“没有足够 semantic measurement”这个问题显式量化。

### 6.2 假设

$H_{M1}$：当前 object formation 失败的重要原因之一是 mask measurement 在时间上太稀疏，导致 D4RT tracks 没有足够语义观测支撑 object posterior。

$H_{M2}$：用 D4RT 传播已有 masks，或补更密的 2D masks，可以显著提高 measurement density；若 object quality 仍不变，则问题在 posterior/inference。

### 6.3 实验设置

对 probe5 每个 scene，固定 D4RT grid32m002 16f stride1 fresh carriers，比较：

```text
M0: 当前 CropFormer mask frames only
M1: framewise available masks, no propagation
M2: D4RT mask propagation from sparse mask frames to all visible frames
M3: frozen video segmentation/masklet propagation if available
M4: no-track source-frame propagation control
M5: shuffled D4RT propagation control
```

M2 的 D4RT propagation 不直接导出 object，只产生 measurement table：

```text
surfel_id, frame_id, region_id, mask_id, inside_prob, boundary_distance, visibility, confidence, cycle_error
```

### 6.4 必须记录指标

```text
num_frames
num_mask_frames_available
mask_frame_density
surfel_positive_observation_rate
mean_positive_observations_per_surfel
median_positive_observations_per_surfel
observation_entropy_per_surfel
visible_but_unobserved_rate
mask propagation self-consistency
cycle error distribution
inside/outside contradiction rate
same-frame competing measurement rate
```

### 6.5 判断标准

如果 M2/M3 相比 M0：

```text
mean_positive_observations_per_surfel 提升 >= 2x
visible_but_unobserved_rate 降低 >= 30%
shuffle/no-track controls 明显更差
```

则 measurement density 是值得解决的方向。

如果 measurement density 提升后 Phase 4 object AP 不提升：

```text
问题不在 measurement 数量，而在 ownership posterior 或 object prior。
```

---

## 7. Phase 3：Masklet / Regionlet measurement generation，不再把 regionlet 当 object

### 7.1 目标

Regionlet v10 失败的根因是把 regionlet 当 object。v11 重新定义：regionlet/masklet 是 measurement primitive，它只提供局部证据，不直接导出为 object。

### 7.2 假设

$H_{R1}$：full 2D mask 经常跨 object，因此需要先拆成局部 measurements。

$H_{R2}$：regionlet 要能被 object posterior 解释，不能直接作为 prediction 输出。

### 7.3 生成方式

每个 2D mask $m_t$ 生成多个 measurement pieces $r_{t,a}$：

```text
source 1: connected component + distance-transform core
source 2: depth / normal discontinuity split
source 3: D4RT surfel seed local support
source 4: appearance edge / color discontinuity
source 5: temporal masklet propagation consistency
```

每个 measurement piece 记录：

```text
frame_id
mask_id
region_id
pixels
backprojected_point_ids for evaluation bridge only
surfel_ids inside
surfel_visibility_stats
boundary_distance_stats
depth_variance
normal_variance
appearance_feature
masklet_track_id if available
```

### 7.4 不直接导出 object

Phase 3 的产物不是 prediction `.npz`，而是 measurement bank：

```text
outputs/v11_measurement_bank/<config>/<scene>/measurements.parquet or jsonl
outputs/v11_measurement_bank/<config>/<scene>/surfel_measurements.npz
```

若为了和历史对比，可以导出 diagnostic prediction，但必须标记：

```text
is_diagnostic_only = true
not_method_result = true
```

### 7.5 指标

```text
num_measurements/frame
num_measurements/mask
measurement area distribution
measurement depth variance
measurement surfel count
measurement boundary distance
measurement-to-GT purity diagnostic only
measurement-to-GT coverage diagnostic only
D4RT surfel support density
masklet temporal consistency
contradiction rate
```

### 7.6 判断标准

Measurement bank 通过的条件：

```text
Oracle-A per-GT best IoU improves over v10 regionlets.
measurement-to-GT purity improves over full-mask baseline.
measurement count does not explode beyond 5x full masks unless oracle upper bound improves substantially.
D4RT shuffle control lower than real D4RT measurement bank.
```

如果 measurement 很纯但 coverage 低：

```text
增加 conservative fringe measurement，不要降低 core purity。
```

如果 measurement coverage 高但 purity 低：

```text
提高 depth/appearance split sensitivity。
引入 temporal masklet consistency。
不要进入 Phase 4。
```

---

## 8. Phase 4：Surfel / measurement ownership posterior

### 8.1 目标

实现真正的新算法核心：object 不是 connected component，而是对 measurements 和 surfels 的解释。D4RT 的作用是提供跨时间 ownership evidence，而不是做 sparse mask selector。

### 8.2 表示

每个 surfel：

$$
s_i=\{X_i(t), \pi_i(t), v_i(t), c_i(t), z_i(t), a_i(t)\}_{t=1}^{T}
$$

每个 measurement：

$$
r_j=\{P_j, S_j, f_j, b_j, d_j, t_j\}
$$

其中 $P_j$ 是像素/点 support，$S_j$ 是包含的 surfels，$f_j$ 是 appearance，$b_j$ 是 boundary confidence，$d_j$ 是 depth/normal statistics。

每个 object slot：

$$
O_k=(C_k, R_k, A_k, \tau_k, q_k)
$$

其中 $C_k$ 是 surfels，$R_k$ 是 measurements，$A_k$ 是 appearance memory，$\tau_k$ 是 temporal lifecycle，$q_k$ 是 quality score。

### 8.3 Energy

$$
E(Y)=
\sum_i \psi_{surfel}(i,y_i)
+\sum_j \psi_{meas}(j,y_j)
+\sum_{i,j}\psi_{link}(i,j,y_i,y_j)
+\sum_{k}\Omega(O_k)
$$

Positive evidence：

```text
surfel repeatedly falls inside measurements of the same object slot
measurement contains many object-core surfels
appearance consistency
D4RT trajectory consistency
visibility continuity
```

Negative evidence：

```text
surfel visible but repeatedly outside object-owned measurement
same-frame cannot-link
two slots claim same surfel
object crosses depth boundary
object contains incompatible motion modes
one measurement is better explained by multiple slots
```

### 8.4 第一版求解方式

Codex 先实现 training-free 近似，不训练新模型：

```text
Step 1: object birth from high-confidence single clean measurement, not full mask.
Step 2: assign surfels to slots using positive/negative evidence.
Step 3: attach compatible measurements to slots.
Step 4: split slots that contain strong negative evidence.
Step 5: merge slots only if appearance + surfel trajectory + measurement consistency all pass.
Step 6: export core/fringe/reject support.
```

不要用普通 connected component。可以用 constrained agglomerative clustering 或 greedy set packing，但每一步必须检查 cannot-link。

### 8.5 实验对照

```text
S0: B1 single-mask baseline
S1: v10 R1/R3 regionlet direct-output baseline
S2: measurement bank + no D4RT posterior
S3: measurement bank + D4RT posterior
S4: measurement bank + shuffled D4RT posterior
S5: measurement bank + no-track posterior
S6: measurement bank + D4RT posterior + appearance
S7: measurement bank + D4RT posterior + appearance + negative evidence
```

### 8.6 指标

除统一 AP 指标外，必须记录：

```text
slot_count
birth_count
split_count
merge_count
rejected_measurement_count
core/fringe/reject point counts
surfel assignment entropy
surfel contradiction rate
visible-outside-owned-region rate
same-frame cannot-link violation count
slot conflict rate
object lifetime length
measurement purity diagnostic only
oracle gap after posterior
```

### 8.7 判断标准

D4RT posterior 成立需要：

```text
S3 > S4 shuffle by at least +0.05 AP50 and +0.05 AP25 on own support.
S3 on S0 > O38 on S0 by at least +0.03 AP.
S3 on S1 > B1/O1 on S1 by at least +0.05 AP.
Stream3D on S3 support - S3 own AP < 0.08.
```

如果 S3 和 S4 接近：

```text
D4RT signal 没有被正确利用；检查 surfel-to-measurement mapping、frame indexing、mask observation sparsity。
```

如果 S3 own 高但 S0/S1 崩：

```text
posterior 仍只是 clean subset selector；增加 verified recall，不要调分数。
```

如果 S3 on S0/S1 升但 own 下降严重：

```text
support recall 有效但 precision 不够；改 core/fringe/reject，不要扩大所有 fringe。
```

---

## 9. Phase 5：Core / Fringe / Reject support export

### 9.1 目标

替代 B1 的 full-mask backprojection 和 R0b/R1b 的 WTA。export 必须尊重 ownership posterior。

### 9.2 表示

每个 object slot 输出三类 support：

```text
core: high posterior surfels/points，少但可靠。
fringe: object-owned measurement 内、靠近 core、无强冲突的区域。
reject: mask 内但有冲突、低置信、跨边界或更适合其他 slot 的区域。
```

最终 prediction：

$$
P_k = Core_k \cup Fringe_k
$$

但 AP 诊断必须单独报告：

```text
Core only
Core + conservative fringe
Core + full fringe diagnostic
```

### 9.3 指标

```text
core union %
fringe union %
reject ratio
core precision diagnostic
fringe contradiction rate
WTA removed assignment rate
conflict before/after export
AP/AP50/AP25 under all policies
```

### 9.4 判断标准

```text
Core-only 应该接近 B1/O1 own quality。
Core+fringe 应该显著提高 S0/S1，不允许 AP50 下降超过 0.05。
WTA removed assignment rate 必须 < 0.35。
若 WTA removed > 0.70，说明又回到先污染再清理。
```

---

## 10. Phase 6：真正的 D4RT geometry aligned -> Stream3D 归因实验

这是用户特别要求的实验，也是 v10 没有完全做到的实验。

### 10.1 目标

把 D4RT geometry 经过 GT/RGB-D anchors 的 Sim3 对齐后，作为几何源喂给原版 Stream3D pipeline，判断 D4RT metric geometry 精度对结果的影响。

注意：这不是主方法，而是 diagnostic-only。它允许使用 ScanNet depth/pose 做 geometry alignment anchors，但禁止使用 GT instance/semantic labels 生成 prediction。

### 10.2 v10 几何实验的不足

v10 `materialize_d4rt_aligned_geometry_for_stream3d.py` 做的是 minimal projection：

```text
D4RT points -> Sim3/raw alignment -> query nearest ScanNet mesh vertex -> group by 2D mask key -> export masks
```

它没有完整替换 Stream3D 的内部几何源。因此 v11 必须实现真正的 Stream3D geometry adapter。

### 10.3 正确实现

Codex 新增：

```text
stream4d/d4rt_stream3d_geometry_adapter.py
tools/run_stream3d_with_d4rt_geometry.py
```

adapter 输出原版 Stream3D 能消费的数据结构：

```text
per-frame local point cloud P^t_D4RT
per-frame 2D mask -> 3D point index mapping
camera/reference metadata
scale-normalized thresholds
geometry_manifest.json
```

比较：

```text
G0: Original Stream3D RGB-D/pose geometry
G1: Stream3D + D4RT raw geometry, no Sim3, negative control
G2: Stream3D + D4RT scene-level Sim3 aligned geometry
G3: Stream3D + D4RT window-level Sim3 aligned geometry
G4: G2 + density-normalized set-cover/manifold thresholds
G5: G3 + density-normalized set-cover/manifold thresholds
G6: RGB-D geometry + D4RT image-space ownership evidence, diagnostic separation
```

### 10.4 Scale-normalized thresholds

不能复用 RGB-D meter-scale 超参。每个 scene/window 计算 D4RT point spacing：

```text
spacing_q25
spacing_q50
spacing_q75
spacing_q90
```

然后设置：

$$
r_{NN}=\alpha \cdot q_{75}(d_{nn})
$$

$$
\delta_{manifold}=\beta \cdot q_{50}(d_{nn})
$$

$$
min\_points = \max(1, \gamma \cdot \text{median projected mask density})
$$

### 10.5 必须记录指标

几何指标：

```text
anchor_count
Sim3 scale
median residual
p90 residual
p95 residual
uv_in01_rate
cycle p90
D4RT point spacing q25/q50/q75/q90
projected mask empty ratio
2D mask RGB-D projection vs D4RT projection IoU
set-cover selected mask count
manifold refine removed ratio
```

AP 指标：

```text
G_i own AP/AP50/AP25
Stream3D original on G_i support
G_i on S0
G_i on S1
pre_points %
union %
GT crop/full
#pred
```

### 10.6 判断标准

D4RT geometry 可替代的最低标准：

```text
G2/G3 relative to G0:
AP drop <= 0.03
AP50 drop <= 0.05
AP25 drop <= 0.07

median residual < 0.15m
p90 residual < 0.35m
empty_projected_mask_ratio < 30%
```

如果 G2/G3 仍然大幅低于 G0：

```text
结论：当前 D4RT metric geometry 不足以替代 ScanNet RGB-D/pose 跑 Stream3D static segmentation。
后续主方法不能 claim metric geometry replacement。
```

如果 G4/G5 明显优于 G2/G3：

```text
结论：过去失败有很大超参尺度因素；所有 D4RT geometry 实验必须 density-normalized。
```

如果 G2/G3 own 高但 G_i on S0 仍崩：

```text
说明 D4RT geometry 只产生 tiny observed support；不是完整 geometry replacement。
```

### 10.7 可视化

每个 scene 输出：

```text
RGB-D point cloud vs D4RT aligned point cloud overlay
Sim3 residual image heatmap
D4RT projected 2D mask vs RGB-D projected 2D mask overlay
Stream3D set-cover selected masks under RGB-D vs D4RT geometry
manifold refining before/after D4RT geometry
failure examples: empty projection, over-merge, fragmented object
```

---

## 11. Phase 7：Tune30 / Final 扩展规则

Probe5 只用于快速判断方向。只有满足以下条件才允许进入 tune30：

```text
S0 Gate passed or S1 Gate passed。
D4RT contribution over shuffle/no-track controls established。
metric integrity pass。
no GT method leakage。
```

Tune30 只调一次参数组，不允许大规模 sweep。Tune30 通过条件：

```text
own-support 不低于 probe5 best 的 80%。
S0/S1 AP 不出现超过 30% 相对下降。
Stream3D-on-method-support gap 不扩大。
```

Final split 只能跑一次 locked config。若 final 失败，记录失败，不能回 tune 后再重跑 final。

---

## 12. v11 必须生成的最终表

### 12.1 Unified ScanNet table

```text
Method | Prediction | Support | Eval policy | AP | AP50 | AP25 | pre% | union% | GT crop/full | #pred | conflict | best IoU | GT>=.25 | GT>=.50 | Stream3D same-support AP | Gap
```

必须包含：

```text
Stream3D P0 on S0
Stream3D P0 on S1
B1
O1
O38
best v10 regionlet
best v11 measurement posterior
best v11 core/fringe/reject
```

### 12.2 Candidate upper-bound table

```text
Candidate pool | #cand | union% | Oracle AP | Oracle AP50 | Oracle AP25 | GT>=.25 | GT>=.50 | duplicate/GT | conclusion
```

### 12.3 D4RT contribution table

```text
Method | real D4RT | shuffle | no-track | random | area control | delta AP | delta AP50 | delta AP25 | conclusion
```

### 12.4 D4RT geometry attribution table

```text
Geometry | Sim3 | density thresholds | median residual | p90 residual | AP | AP50 | AP25 | AP drop vs G0 | empty mask ratio | conclusion
```

---

## 13. 立即停止作为主线的方向

以下方向只能作为 ablation，不能作为主要推进：

```text
adaptive top-k ratio sweep
mask_count / area / logarea score sweep
point NMS / point IoC merge
support completion by nearest object core
full-mask fringe + final WTA
zero-conflict exclusivity
scene object memory over already-wrong masks
regionlet direct-output as object
D4RT raw geometry direct replacement with RGB-D meter-scale thresholds
more windows -> more proposals without measurement posterior
```

---

## 14. v11 的成败判断

v11 成功不要求 full ScanNet 超过 Stream3D，但必须至少证明一个核心方向真的成立：

```text
1. Candidate oracle shows current primitive has upper bound, and v11 posterior closes part of the gap.
2. D4RT real signal beats shuffle/no-track controls under unified support.
3. S0/S1 metrics improve beyond O38/R baselines, not just own-support.
4. D4RT geometry aligned-to-GT diagnostic gives a quantified answer about geometry precision impact.
```

如果这些都失败，必须写清楚：

```text
Current training-free 2D mask measurements + D4RT surfel evidence are insufficient to infer stable object partitions on static ScanNet under the current primitive design. The project must either introduce stronger frozen video masklet priors, use denser semantic observations, or reposition the work around dynamic correspondence where D4RT's advantage is structurally stronger.
```

这不是放弃，而是避免继续在错误层级上消耗时间。
