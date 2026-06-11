# Stream4D v9：统一评估协议、真实差距确认与 D4RT-native Semantic 4D Surfel Field 推进计划

面向 Codex 的执行计划。本文档的核心目标是把评估协议彻底统一，并把方法推进从“后处理调参”转向真正的 **D4RT-native semantic 4D surfel field**。本文档中的公式均使用 Typora 友好的 `$...$` 或 `$$...$$`。

---

## 0. 必须先统一的判断

当前评估确实有混乱风险：历史实验中有时报告 `recompute_pre_points`，有时报告 `inherit_pre_points`，有时又报告 cross-fixed support。以后不能只报其中一个。

**结论必须明确：Stream3D paper-style 主指标对应的是每个方法使用自己的 `TMP/{config}/{scene}_pre_points.npy` 的 cropped-TMP evaluator。**  
对原版 Stream3D，这就是 `scannet` prediction + `scannet` pre_points；对一个正常 materialized 的新方法，也就是该方法自己的 prediction + 该方法自己的 pre_points。对 post-processing 方法来说，若后处理重新写出 prediction 并重新写出自己的 TMP，则这对应 `recompute_pre_points` / own-support 协议。

但是，`recompute_pre_points` 不足以判断和 Stream3D 的真实差距，因为它会让 evaluation universe 随 prediction union 变小。  
所以以后每个方法都必须同时报告：

```text
A. paper-style own-support / recompute_pre_points
B. parent-inherit / inherit_pre_points
C. Stream3D-on-method-support
D. method-on-Stream3D-support
```

只有这样才能回答两个问题：

```text
1. 按 Stream3D paper 的 per-method support 协议，方法有没有超过 baseline？
2. 在同一个 support universe 下，方法本身的 object quality 有没有超过 Stream3D？
```

若一个实验缺少 B/C/D 中任意一项，不能写成“超过 Stream3D”，只能写成 “own-support observed subset result”。

---

## 1. 根据历史数据，当前和 Stream3D 的真实差距

### 1.1 v3 full ScanNet 312 scenes

单位：百分制 AP points。

| 对比 | Stream4D | Stream3D 对照 | 差距 |
|---|---:|---:|---:|
| v3 adaptive recompute vs Stream3D original | `20.3718 / 35.5222 / 55.0649` | `20.1139 / 34.4654 / 50.2268` | `+0.2579 / +1.0568 / +4.8381` |
| v3 adaptive inherit vs Stream3D original | `12.2851 / 23.3147 / 41.6773` | `20.1139 / 34.4654 / 50.2268` | `-7.8288 / -11.1507 / -8.5495` |
| v3 adaptive recompute vs Stream3D on same adaptive support | `20.3718 / 35.5222 / 55.0649` | `32.6669 / 50.1160 / 67.6487` | `-12.2951 / -14.5938 / -12.5838` |
| v3 adaptive on Stream3D support vs Stream3D original | `0.1003 / 0.4225 / 3.2100` | `20.1139 / 34.4654 / 50.2268` | `-20.0136 / -34.0429 / -47.0168` |

解释：v3 在 own-support/recompute 下略超 paper-style baseline，但一旦 fixed/inherit 或同 support 比较，就明显输。尤其 Stream3D 放到 Stream4D adaptive 的小 support 上能到 `32.6669 / 50.1160 / 67.6487`，说明 Stream4D 不是在同一个 support universe 里 object quality 更强。

### 1.2 v4.1 locked final split

单位：百分制 AP points。

| 对比 | Stream4D | Stream3D 对照 | 差距 |
|---|---:|---:|---:|
| v4.1 reliable final recompute vs Stream3D final | `30.2449 / 51.5938 / 67.0619` | `19.4294 / 33.3989 / 49.6361` | `+10.8155 / +18.1949 / +17.4258` |
| v4.1 reliable final on MVP fixed vs Stream3D final | `9.0795 / 18.2119 / 30.2373` | `19.4294 / 33.3989 / 49.6361` | `-10.3499 / -15.1870 / -19.3988` |
| v4.1 reliable final on Stream3D fixed vs Stream3D final | `0.6823 / 2.1181 / 7.3900` | `19.4294 / 33.3989 / 49.6361` | `-18.7471 / -31.2808 / -42.2461` |

解释：v4.1 的 recompute 很强，但 own support 只有约 `7.7%` scene vertices，平均约 `10` 个 objects。它证明了“高精度 observed support proposal”能力，不证明完整 scene support 或统一 support 下超过 Stream3D。

### 1.3 v7 probe5 same-support matrix

v7 的 probe5 表使用原始 0-1 AP。下面统一换成百分制 AP points。

| 对比 | Stream4D / method | Stream3D on same support | 差距 |
|---|---:|---:|---:|
| v6 compact on S1 32f support | `28.4832 / 50.3962 / 67.1915` | `39.9213 / 59.7171 / 74.2535` | `-11.4381 / -9.3209 / -7.0621` |
| v6 scoreunique on S1 32f support | `28.2050 / 49.8231 / 69.1264` | `39.9213 / 59.7171 / 74.2535` | `-11.7163 / -9.8940 / -5.1271` |
| v6 localprop own support | `11.2524 / 28.5390 / 47.9773` | 不可直接同表比较 | 未知 |
| v6 typedv3 own support | `3.3478 / 12.8495 / 39.7753` | 不可直接同表比较 | 未知 |

解释：在真正 same-support 诊断上，Stream4D 仍差 Stream3D 约 `9-12` AP/AP50 points。这个差距不是 evaluator bug，而是 object formation / ownership / high-IoU mask quality 差距。

### 1.4 v8 B1 当前 best 的差距仍未确定

v8 当前 best：

```text
B1 surfacelet_singlemask probe5:
AP / AP50 / AP25 = 32.8439 / 62.9266 / 88.4363
pre_points ratio = 3.9861%
GT crop/full = 8.20 / 40.60
objects/scene = 16.4
```

B1 很有价值，但它缺少最关键的 cross-support 对照：

```text
Stream3D prediction on B1 pre_points
B1 prediction on Stream3D pre_points
B1 prediction on S1 32f support
B1 prediction on parent support
```

因此现在不能确定 B1 和 Stream3D 的真实差距。能说的是：

```text
B1 在 own-support/recompute 上显著强于过去的 Stream4D variants；
但它 support 极小，且只让 8.20/40.60 个 GT instances 进入评价；
在没有 Stream3D-on-B1-support 之前，不能 claim B1 object quality 超过 Stream3D。
```

历史 v3 已经证明，Stream3D 放到 Stream4D 的小 support 上可能更强。因此 B1 必须优先补 cross-support matrix。

---

## 2. 统一评估协议：以后每个实验必须这样报

### 2.1 四类评估行

对任意新方法 `M`，必须生成以下结果。缺任意一项，结果表不得进入主结论。

```text
M_own:
  prediction = M
  pre_points = M
  policy = own_recompute / paper-style

M_inherit_parent:
  prediction = M
  pre_points = parent_config
  policy = parent_inherit

Stream3D_on_M:
  prediction = Stream3D scannet
  pre_points = M
  policy = cross_fixed_method_support

M_on_Stream3D:
  prediction = M
  pre_points = Stream3D scannet
  policy = cross_fixed_stream3d_support
```

如果 `M` 是从 parent config 后处理来的，例如 B1 从 G1 surfel/mask bank 来，parent 必须写清楚：

```text
parent_config
parent_pre_points_path
parent_support_policy
```

如果 `M` 不是后处理而是独立导出，仍要报告 `M_own`、`Stream3D_on_M` 和 `M_on_Stream3D`。

### 2.2 每张表必须包含的列

```text
method
prediction_config
pre_points_config
eval_policy
split
num_scenes
AP
AP50
AP25
pre_points %
prediction union %
union in target % scene
union in target % target
GT crop/full
#pred/scene
objects/scene
points/scene
tiny_mask_ratio <100 vertices
large_mask_ratio >1000 vertices
conflict_rate
uses_gt
is_diagnostic_only
manifest_path
metric_integrity_pass
```

### 2.3 哪个指标对应 Stream3D paper

**Stream3D paper-style 指标 = own-support / per-method TMP evaluator。**

对于原版 Stream3D：

```text
prediction = scannet
pre_points = scannet
```

对于 Stream4D 新方法：

```text
prediction = method_config
pre_points = method_config
```

这就是 `recompute_pre_points` / own-support 结果。它可以作为对齐原版 Stream3D evaluator 的主表，但必须同时附上 same-support diagnostic。  

**`inherit_pre_points` 不是 paper 主指标；它是 top-k/postprocess 的诊断指标。**  
但因为 Stream4D 多次通过缩小 support 获得高 recompute AP，以后它必须和主指标一起报告。

### 2.4 判断 claim 的规则

允许写：

```text
Under Stream3D-style own-support evaluation, M outperforms Stream3D baseline.
```

条件：

```text
M_own > Stream3D_self
metric integrity pass
hyperparameters locked before final split
```

允许写：

```text
M has stronger object quality than Stream3D on the same support.
```

条件：

```text
M_own > Stream3D_on_M
且 M_on_Stream3D 不灾难性低
```

允许写：

```text
M robustly outperforms Stream3D.
```

条件：

```text
M_own > Stream3D_self
M_own >= Stream3D_on_M 或至少 AP/AP50 接近
M_on_Stream3D 不低于 Stream3D_self 太多
pre_points/GT crop 不显著小到只剩 observed subset
```

禁止写：

```text
只凭 M_own/recompute 高，就说全面超过 Stream3D。
只凭 probe5 高，就说 full ScanNet 过关。
只凭 class-agnostic AP，高度概括为 semantic 4D reconstruction。
```

---

## 3. 当前算法问题的重新判断

历史实验给出的根本事实是：

```text
1. recompute 高分容易来自小 support 和少量高精度 object。
2. inherit/fixed 失败说明 support coverage 和 object recall 不够。
3. Stream3D on Stream4D small support 经常更强，说明不是 support 小就天然有利于 Stream4D。
4. B1 当前 best 本质是 D4RT-assisted single-mask ownership selector，而不是完整 semantic 4D surfel field。
5. 多帧 mask history 没有天然变好，single-mask 反而最好，说明当前 temporal evidence 还没建模对。
```

现在真正的问题不是“再调哪个后处理阈值”，而是要解决三个核心问题：

```text
Q1: D4RT semi-dense surfels 是否真的提供了超过 2D mask area/mask_count 的 object evidence？
Q2: 如何从 clean single-mask object birth 过渡到跨时间 surfel ownership field，而不是 full-mask backprojection shortcut？
Q3: 如何在 unified evaluation matrix 下证明方法不是靠小 support 取巧？
```

---

## 4. 相关研究进展给我们的启发

### 4.1 D4RT 的启发

D4RT 的核心不是替代 RGB-D 点云，而是 query-based 4D material correspondence。它用 query $q=(u,v,t_{src},t_{tgt},t_{cam})$ 查询 source point 在目标时间与参考相机下的 3D 位置，并且支持 sparse/dense flexible decoding。D4RT 还提出 dense all-pixel tracking 的 occupancy-grid 策略，用 visible track pixels 标记已访问区域，从而避免朴素 $O(T^2HW)$ 查询。  
对我们来说，D4RT 应该成为 semantic 4D surfel field 的时空骨架，而不是 sparse mask selector。

### 4.2 4D semantic / language field 的启发

近期 4D semantic field 工作集中在两个方向：一类是把 2D foundation model 能力蒸馏或提升到 4D feature field；另一类是把 4D geometry 与 language alignment 做成 feed-forward。Feature4X 强调把 2D vision foundation model 功能扩展到 monocular-video 4D feature field；4DLangVGGT 明确针对 scene-specific 4D Gaussian language field 难以泛化的问题，提出 feed-forward 4D language grounding；DINO_4D 则说明 frozen semantic features 可以作为 structural priors，帮助减少 dynamic tracking 中的 semantic drift。

这些工作给我们的方向是：

```text
语义不能只是最后给 3D mask 贴 label；
语义应该在 surfel/object field inference 阶段参与 object ownership、split、merge、tracking。
```

但是我们仍坚持 training-free：不训练新的 4D semantic model，不用 3D/4D semantic GT，只用 frozen D4RT、2D masks、DINO/CLIP/VLM features 和训练自由优化。

---

## 5. v9 总目标

v9 不是继续优化 `B1 single-mask` 的阈值，而是把 B1 变成一个可验证的中间 primitive：

```text
clean single-mask birth -> D4RT surfel ownership field -> temporal object state -> semantic 4D field
```

v9 的主目标：

```text
1. 统一评估，明确当前与 Stream3D 的真实差距。
2. 验证 B1 的收益是否真的来自 D4RT，而不是 mask area / mask count shortcut。
3. 从 full-mask backprojection 改成 core/fringe/reject 的 surfel ownership export。
4. 补齐 cross-support matrix，禁止只报 recompute。
5. 在 ScanNet 上证明 observed-support result 不再是唯一优势。
6. 在 Dynamic Replica 可用时，启动官方 tracking；不可用时，只做数据 blocker 与非官方 sanity，不伪造指标。
```

---

## 6. Phase 0：Codex 必须提交可审计代码包

每轮实验结束，Codex 必须提交一个 code audit packet。没有这个包，不允许报告新 AP。

### 6.1 包内容

```text
code_review_packet.zip
  Stream3D/stream4d/
  Stream3D/tools/
  Stream3D/evaluation/
  Stream3D/tests/
  Stream3D/scripts/
  docs/实验结果复盘.md
  docs/执行日志.md
  outputs/audit/
  outputs/*summary*.json
  data/evaluation/scannet/*txt
  data/prediction/<new_configs>/*config_manifest.json
  data/TMP/<new_configs>/*_pre_points.npy for probe5
  scripts/reproduce_*.sh
  git_diff.patch
  requirements_or_env.txt
```

### 6.2 必须通过的检查

```text
py_compile pass
unit tests pass
manifest scan pass
metric integrity pass
no uses_gt=true method result
all reportable configs have manifest
all configs declare eval_policy
all AP tables include Stream3D comparison rows
```

不满足时：

```text
不要跑新算法。
先修 manifest/evaluator/report scripts。
```

---

## 7. Phase 1：统一 gap matrix，先把 B1 和 Stream3D 差距跑清楚

### 7.1 目标

回答当前最重要问题：

```text
B1 是否真的比 Stream3D 强，还是只是在小 support 上看起来强？
```

### 7.2 实验配置

必须在相同 probe5 split 上跑以下 matrix。若资源允许，扩展到 tune30。

```text
P0 = Stream3D scannet prediction
P1 = Stream4D v6 compact
P2 = Stream4D v8 B1 surfacelet_singlemask
P3 = B1-no-track control
P4 = B1-shuffle control
P5 = B1-area-matched control

S0 = Stream3D scannet pre_points
S1 = v6 32f support
S2 = B1 own pre_points
S3 = B1 parent/G1 support
S4 = union support of Stream3D and B1
```

至少要跑：

```text
P0 on S0
P0 on S2
P2 on S2
P2 on S0
P2 on S1
P2 on S3
P0 on S3
```

### 7.3 必须记录的指标

```text
AP / AP50 / AP25
pre_points %
prediction union %
union in target % scene
union in target % target
GT crop/full
#pred
objects/scene
points/scene
conflict_rate
tiny_mask_ratio
per-scene AP variance
per-GT best IoU mean
GT IoU>=0.25 count
GT IoU>=0.50 count
missed GT count
duplicate predictions per GT
```

### 7.4 判断标准

B1 可继续作为主方向的最低标准：

```text
B1 own-support AP/AP50/AP25 高于 Stream3D on B1 support 中至少 AP 和 AP50。
B1 on S1 32f support 不低于 v6 compact on S1。
B1 on Stream3D support 不再接近 0；至少 AP25 不能崩到个位数。
```

如果失败：

```text
若 Stream3D on B1 support > B1：
  B1 只是 support/mask selector，不是更强 object quality。
  禁止继续调 B1 top-k；转向 surfel ownership field。

若 B1 on Stream3D support 接近 0：
  说明仍是 tiny-support subset result。
  必须在下一阶段增加 object recall / support field，而不是报告 full ScanNet。

若 B1 on S1 比 v6 compact 差：
  说明 B1 的高分只来自更小 support，不适合作主方法。
```

### 7.5 可视化

每个 scene 输出：

```text
B1 support vs Stream3D support overlay
Stream3D-on-B1 support matched GT mesh
B1-on-B1 support matched GT mesh
B1-on-Stream3D support missed GT instances
per-GT best-IoU bar plot
prediction count vs GT count plot
```

---

## 8. Phase 2：D4RT 贡献判别实验

### 8.1 目标

判断 B1 的收益到底来自 D4RT surfel correspondence，还是来自简单的 mask area / mask count / support shrinking。

### 8.2 实验配置

在 B1 相同 object count / support budget 下跑：

```text
B1-D4RT:
  当前 B1。

B1-no-track:
  不使用 target-time D4RT track，只用 source-frame mask internal carrier count。

B1-shuffle:
  保持每帧 mask 数量和 carrier 数量，打乱 carrier-to-mask assignment。

B1-random-same-count:
  随机选择和 B1 相同数量的 mask-frame proposals，重复 5 seeds。

B1-area-same-count:
  选择相同数量的最大 area masks。

B1-maskcount-same-count:
  选择相同数量的最高 mask_count / observation_count masks。

B1-2D-center:
  用 2D mask centroid / area heuristic 替代 D4RT surfel support。
```

### 8.3 指标

每个配置都必须跑 Phase 1 的四类评估行：

```text
own-support
Stream3D-on-method-support
method-on-Stream3D-support
method-on-parent-support
```

此外记录：

```text
D4RT carrier assignment rate
shuffle degradation
random mean/std
area heuristic gap
object birth precision
object birth recall
mask owner conflict rate
owned mask purity diagnostic
```

### 8.4 判断标准

D4RT 贡献成立：

```text
B1-D4RT 比 B1-shuffle 至少高：
  AP +5 points
  AP50 +8 points
  AP25 +10 points

B1-D4RT 比 area/maskcount control 至少高：
  AP +3 points
  AP50 +5 points

B1-D4RT 的 Stream3D-on-method-support 差距不能反向过大。
```

如果失败：

```text
若 no-track 接近 D4RT：
  当前 D4RT 没有贡献，B1 只是 2D mask selector。
  必须改成真正的 temporal surfel ownership，不继续调 B1。

若 shuffle 仍接近 D4RT：
  carrier-to-mask assignment 不是有效信号，检查 D4RT UV / mask measurement 是否对齐。

若 area control 接近 D4RT：
  必须加入 negative evidence 和 temporal consistency，否则没有新意。
```

---

## 9. Phase 3：D4RT semi-dense surfel field sanity

### 9.1 目标

验证 D4RT-native surfel field 是否有足够几何/时序质量支撑 semantic 4D reconstruction。不能再从 sparse carrier 直接跳到 object AP。

### 9.2 实验配置

在连续帧 clip 上评估：

```text
grid16
grid32
grid48
adaptive occupancy grid
high-confidence-only grid32
margin 0.02 vs 0.05
clip length 16 / 32
```

必须避免 stride-10 temporal-scale 错误，默认使用连续帧。

### 9.3 指标

```text
num_queries
valid_track_ratio
uv_in01_rate
visibility_mean
confidence_mean
track_length_visible_mean
track_length_visible_p10
self_uv_error_mean/p90
cycle_uv_error_mean/p90
surfel_coverage_2d
surfel_coverage_3d
D4RT point spacing quantiles
Sim3 residual median/p90
Sim3 scale
mask measurement frames available
positive observations per surfel
surfel observation entropy
```

### 9.4 判断标准

继续做 surfel field 的最低门槛：

```text
uv_in01_rate >= 0.95
cycle_uv_error_p90 <= 5 px
self_uv_error_p90 <= 3 px
track_length_visible_mean >= 10 for 16f clips
surfel coverage >= 3x sparse carrier support
mask measurement available frames >= 25% of clip frames 或有 mask propagation 补救
```

Sim3 不作为必须过关项，但必须记录：

```text
若 Sim3 residual median > 0.30m:
  不允许 claim D4RT metric geometry 替代 ScanNet RGB-D。
  但可以继续 image-space/surfel-observation-space semantic field。
```

如果失败：

```text
uv/cycle 失败：
  检查 frame_stride、resize/aspect、query normalization、D4RT adapter vs official helper。

coverage 低：
  提升 grid resolution 或用 occupancy-grid adaptive dense tracking。

mask observation 太稀疏：
  增加 2D mask frame sampling frequency。
  或用 D4RT track 把已有 mask observation propagation 到无 mask 帧，但要标记 propagated measurement。
```

---

## 10. Phase 4：从 B1 single-mask birth 到 surfel ownership field

### 10.1 目标

B1 的启发是：clean single-mask object birth 是有效的。但 B1 的问题是它直接把 owned mask full backproject 成最终 support。v9 要把 B1 改成：

```text
single clean mask birth -> object slot -> surfel ownership belief -> core/fringe/reject support -> temporal update
```

### 10.2 方法定义

每个 surfel $s_i$ 有轨迹和观测：

$$
s_i = \{X_i(t), \pi_i(t), v_i(t), c_i(t), z_i(t), f_i(t)\}_{t=1}^{T}
$$

每个 object slot $O_k$ 由一个 clean mask birth 初始化。之后维护 surfel ownership posterior：

$$
p(y_i=k \mid \mathcal{Z}) \propto \exp(
\lambda_m S^{mask}_{ik}
+\lambda_a S^{app}_{ik}
+\lambda_t S^{traj}_{ik}
+\lambda_g S^{geom}_{ik}
-\lambda_x S^{conflict}_{ik}
-\lambda_b S^{boundary}_{ik}
)
$$

其中：

```text
S_mask: surfel 是否稳定落入该 object 的 owned masks
S_app: surfel local feature 是否与 object birth crop 一致
S_traj: trajectory 是否与 object slot 的 motion/visibility 一致
S_geom: 局部 surfel adjacency / point spacing consistency
S_conflict: 是否被同帧其他 object 强支持
S_boundary: 是否长期处在 mask boundary risk 区域
```

将 surfel 分成：

```text
core: p(y_i=k) >= tau_core 且有正证据
fringe: tau_fringe <= p(y_i=k) < tau_core 且无强 conflict
reject: p(y_i=k) < tau_fringe 或 strong conflict
unknown: 无足够 measurement
```

最终导出不能再直接 full-mask backproject，而是：

```text
core surfels + conservative fringe region
```

### 10.3 实验配置

在 probe5 跑：

```text
O0: B1 full-mask backproject 当前 best
O1: B1 surfel-core only
O2: B1 core + local connected fringe
O3: O2 + negative evidence
O4: O3 + temporal propagation to unmasked frames
O5: O4 + object competition
```

每个都必须跑统一四类评估行。

### 10.4 指标

```text
AP/AP50/AP25 under own-support
AP/AP50/AP25 under Stream3D-on-method-support
AP/AP50/AP25 under method-on-Stream3D-support
pre_points %
GT crop/full
surfel core count/object
surfel fringe count/object
core/fringe/reject ratio
ownership entropy
strong conflict rate
boundary-risk ratio
temporal persistence
object birth count
object death/pending count
duplicate predictions per GT
per-GT best IoU mean
```

### 10.5 判断标准

最低成功：

```text
O2/O3 own-support AP50 不低于 B1 超过 3 points。
method support 从 3.986% 提升到 >= 7%。
Stream3D-on-method-support 不显著高于 method own；gap <= 5 AP50 points。
method-on-S1 32f support 高于 v6 compact。
```

强成功：

```text
support >= 10%
AP50 >= B1 AP50 - 3 points
AP >= B1 AP - 2 points
Stream3D-on-method-support <= method own-support
method-on-Stream3D AP25 不低于 20 points
```

如果失败：

```text
core-only AP 高但 support 极小：
  增大 conservative fringe，但必须记录 boundary/conflict。

fringe 让 AP50 崩：
  减小 fringe radius，加入 negative evidence，不要回到 full-mask backproject。

temporal propagation 变差：
  mask measurement 太稀疏或 D4RT track drift；只在 high-confidence frames propagation。

object competition 删除真阳性：
  降低 competition penalty，增加 pending state，不立即 reject。
```

### 10.6 可视化

```text
surfel core/fringe/reject overlay on image
object slot timeline
owned mask history
surfel ownership entropy heatmap
false positive region panels
missed GT panels
Stream3D vs O3 on same support mesh panel
```

---

## 11. Phase 5：mask measurement sparsity 实验

### 11.1 目标

v8 已发现连续 16 帧中只有少数 mask frames，导致 temporal field 没有足够 semantic measurements。必须量化不同 mask sampling rate 对方法的影响。

### 11.2 实验配置

```text
M0: existing mask frames only
M1: use Cropformer every 10 frames
M2: use Cropformer every 5 frames
M3: use Cropformer every 2 frames
M4: use propagated masks via D4RT tracks, no new 2D model
M5: hybrid new masks + propagated masks
```

如果没有现成 masks：

```text
先只在 probe5/scene0050 运行 2D mask model。
若 2D mask model 不可运行，生成 missing-mask report，不得伪造。
```

### 11.3 指标

```text
num_mask_frames_available
num_mask_frames_missing
positive observations per surfel
surfel observation temporal span
object birth recall
object slot reactivation count
AP/AP50/AP25 all eval policies
runtime per frame
2D mask count per frame
mask conflict rate
```

### 11.4 判断标准

```text
如果更高 mask frequency 明显提升 own-support 和 same-support：
  当前主要瓶颈是 semantic measurement sparsity。

如果 mask frequency 提升但 AP 不升：
  问题是 object inference，不是 measurement 数量。

如果 propagated masks 提升 AP25 但 AP/AP50 下降：
  propagation 过粗，不能作为主方法；只作为 fringe evidence。
```

---

## 12. Phase 6：scale-normalized D4RT geometry diagnostic

### 12.1 目标

用户指出直接使用 D4RT 几何时，不能复用 ScanNet meter-scale 超参。这个判断是正确的。v9 必须重新做 scale-normalized diagnostic，回答：

```text
Stream3D-style pipeline 使用 D4RT geometry 到底下降多少？
下降是 geometry 本身，还是尺度超参不适配？
```

### 12.2 实验配置

```text
G0: Stream3D original RGB-D/pose geometry
G1: Stream3D + D4RT point map, old meter-scale params
G2: Stream3D + D4RT point map, Sim3 eval-only alignment
G3: G2 + NN radius from D4RT point spacing quantiles
G4: G3 + normalized manifold threshold
G5: G3 + D4RT track-consistency replacing Euclidean manifold threshold
```

D4RT-normalized thresholds：

```text
nn_radius = q90(nearest_neighbor_distance_D4RT_points)
manifold_delta = alpha * median_local_spacing
min_points_per_object = beta * median_support_density
```

### 12.3 指标

```text
AP/AP50/AP25
num_exported_objects
0-object scenes
D4RT point spacing median/q90/q95
Sim3 residual median/p90
scale factor
mesh NN hit rate
mask backprojection hit rate
manifold rejection rate
fragmentation count
```

### 12.4 判断标准

```text
如果 G3/G4 从 nan/0 AP 恢复到非零：
  证明过去 D4RT geometry failure 部分来自 scale hyperparams。

如果 G3/G4 仍远低于 G0 超过 5 AP/AP50:
  不能把 D4RT geometry replacement 作为 ScanNet 主卖点。

如果 G5 明显优于 G4:
  说明 D4RT track consistency 比 metric Euclidean manifold 更适合该几何源。
```

---

## 13. Phase 7：Dynamic Replica / Replica-Dynamic

### 13.1 目标

ScanNet 是静态 3D segmentation benchmark，不足以证明 semantic 4D reconstruction and tracking。动态场景必须回答：

```text
D4RT-native surfel ownership 是否能减少 ID switch、遮挡断裂、动态物体重现失败？
```

### 13.2 数据检查

先运行：

```text
check_dynamic_replica_env
```

必须记录：

```text
data_root_exists
split_dir_exists
annotation_exists
usable_scene_count
camera fields present
depth available
instance id available
semantic id available
```

如果 `usable_scene_count=0`：

```text
停止 official metrics。
只报告 data blocker，不做伪指标。
```

### 13.3 官方指标

若数据可用，报告：

```text
IDF1
IDSW
Frag
MOTA/MOTP if applicable
4D IoU
track purity
track coverage
object reactivation accuracy
trajectory APD3D / AJ / OA if point-track GT available
```

### 13.4 对照方法

```text
D0: per-frame mask only
D1: Stream3D-style 3D overlap memory
D2: B1 single-mask birth
D3: v9 surfel ownership field
D4: v9 + semantic feature prior
```

### 13.5 成功标准

```text
v9 IDF1 > Stream3D-style memory by at least 5 points
IDSW reduced by at least 20%
Reactivation accuracy improves
No increase in catastrophic over-merge
Qualitative videos show dynamic identity persistence
```

如果失败：

```text
若 IDSW 高：
  增强 negative evidence 和 lost/pending state。

若 over-merge：
  提高 cannot-link priority，不允许 weak positive edge 覆盖 strong negative。

若 tracks drift：
  缩短 D4RT window，使用 high-confidence anchors，检查 camera convention。
```

---

## 14. 并行执行安排

### Day 0：协议与差距

并行：

```text
Lane A: Phase 1 B1 gap matrix
Lane B: Phase 2 D4RT contribution controls
Lane C: Phase 3 surfel field sanity
```

当天必须产出：

```text
B1 vs Stream3D exact gap table
D4RT contribution yes/no
surfel field geometry/measurement sanity
```

### Day 1：方法核心

根据 Day 0 结果：

```text
若 D4RT contribution 成立：
  执行 Phase 4 O1-O3。

若 D4RT contribution 不成立：
  停止 B1 路线，重查 D4RT measurement projection。

若 surfel sanity 失败：
  先修 D4RT continuous clip / query adapter，不跑 AP。
```

### Day 2：扩大验证

```text
probe5 -> tune30
统一四类评估行
生成可视化
只锁定一个 best config
```

### Day 3：final / dynamic

```text
locked final split
Dynamic Replica env check
如果可用，开始 dynamic tracking metrics
```

---

## 15. 最终报告模板

每个实验必须用下面的最小表格，不允许删列。

| method | prediction | pre_points | policy | AP | AP50 | AP25 | pre% | union% | GT crop/full | #pred | conflict | Stream3D same-support AP/AP50/AP25 | gap |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|

每个 method 的主表至少四行：

```text
M own
Stream3D on M support
M on Stream3D support
M inherit parent
```

每个 summary 必须写：

```text
Can claim paper-style improvement? yes/no
Can claim same-support improvement? yes/no
Can claim robust improvement? yes/no
Can claim D4RT-native 4D field? yes/no
```

---

## 16. 当前最重要的执行优先级

第一优先级：

```text
补 B1 cross-support matrix。
```

第二优先级：

```text
做 B1-D4RT / no-track / shuffle / area-matched 对照。
```

第三优先级：

```text
从 B1 full-mask shortcut 改成 surfel core/fringe/reject ownership export。
```

第四优先级：

```text
重新做 scale-normalized D4RT geometry diagnostic，明确不能复用 meter-scale Stream3D 超参。
```

第五优先级：

```text
补 Dynamic Replica 数据与官方 tracking 指标。
```

---

## 17. 现在不能再做的事

```text
不要只报 recompute。
不要继续只调 B1 min_carriers / max_observations / max_masks_per_object。
不要继续用 own-support probe5 高分说“超过 Stream3D”。
不要把 full-mask backprojection 当作 semantic 4D field。
不要把 Stream3D+D4RT 写成方法主线。
不要把 D4RT geometry direct replacement 的旧 0 AP 作为最终结论；必须用 scale-normalized 超参重测。
不要在 Dynamic Replica 数据不可用时报告 official tracking 指标。
```

---

## 18. 结束判断

v9 成功不是看单个 recompute AP 是否高，而是看这三个问题是否被回答：

```text
1. B1 / v9 和 Stream3D 的差距，在同 support 下到底是多少？
2. D4RT surfel correspondence 是否提供了不可由 2D mask heuristic 替代的 object evidence？
3. surfel ownership field 是否能在增加 support 的同时保持 AP/AP50，而不是退化成 tiny-support mask selector？
```

只有这三个问题成立，才能继续写成 D4RT-native semantic 4D reconstruction and tracking。否则，当前路线仍然只是 observed-support object proposal selection。
