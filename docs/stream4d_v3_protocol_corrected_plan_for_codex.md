# Stream4D v3：Stream3D 评估协议确认、方法改进与 ScanNet / Replica-Dynamic 实验计划书

面向 Codex 的执行文件。本文档替代 `stream4d_v2_code_audit_and_experiment_plan.md`。v2 文档里把 `pre_points` 裁剪评价直接定义为“严重指标风险”和“必须改成 fullmesh 主评估”，这个判断需要修正。重新解压并对照原始 `Code_Stream3D.zip` 与审阅包 `stream4d_code_review_packet.zip` 后，结论是：**`evaluation/evaluate.py` 读取 `TMP/{config}/{scene}_pre_points.npy` 并裁剪 GT / prediction 的做法确实来自原版 Stream3D 评估代码，因此它不是虚假指标，也不是我们私自引入的作弊评价。**

但是，当前 Stream4D 的 adaptive top-k 版本还有两个需要继续审计的边界：

1. 原版 Stream3D 的 `pre_points` 语义是该方法通过 2D-to-3D 投影检测到、参与导出的点集；Stream4D 当前 `rescore_scannet.py` 在 top-k 后重新写 `pre_points = pred_masks.any(axis=1)`，会随着后处理筛掉实例而同步缩小评价点集。这个行为可能仍符合“最终输出 support”的 Stream3D-style evaluator，但为了证明 top-k 不是靠缩小评价 universe 获利，必须额外跑一个更严格的 `inherit_pre_points` 实验。
2. 当前 best 配置来自同一个 ScanNet val 312 scenes 上的多轮后处理搜索，因此仍存在 validation overfitting 风险。必须用 tune/final split 或 k-fold scene split 锁定超参后再报告。

本文档使用 Typora 友好的公式格式，只使用 `$...$` 或 `$$...$$`，不使用 display 公式的方括号语法。

---

## 0. 最重要的结论更新

### 0.1 用户判断基本正确

确认结果如下：

```text
原始 Code_Stream3D.zip / Code_Stream3D/evaluation/evaluate.py
assign_instances_for_scan():
  loaded_array = np.load(title + config + '/' + scene_id + '_pre_points.npy')
  pred_info = read_pridiction_npz(pred_file, loaded_array)
  gt_ids = np.loadtxt(gt_file)
  gt_ids = gt_ids[loaded_array]
```

当前审阅包中的 `Stream3D/evaluation/evaluate.py` 保留了这套逻辑，只把硬编码 TMP 路径改成了：

```text
--tmp_root data/TMP
--tmp_config ${CFG}
```

因此：

```text
Stream3D-style cropped evaluator 是原版 Stream3D 的评估协议。
用它和 Stream3D-Cropformer baseline 对齐比较是合理的。
不能把这个 evaluator 本身称为虚假指标。
```

v2 计划中“必须先修 fullmesh evaluator，否则任何结果不可信”的表述需要删除。正确说法是：

```text
主评估应当优先使用与 Stream3D 对齐的 cropped-TMP evaluator。
fullmesh evaluator 只作为额外诊断，不作为替换 Stream3D benchmark 的主协议。
```

### 0.2 仍然必须保留的谨慎边界

虽然 evaluator 是 Stream3D 原版协议，但当前 best 仍不能无限夸大。当前已经记录的事实是：

```text
实际配置：OpenD4RT 32CLIP checkpoint + Cropformer masks + first 32 stride-10 frames + rgbd_eval export adapter
当前 best：adaptive mask_count top-k
结果：20.37 / 35.52 / 55.06
Stream3D-Cropformer baseline：20.11 / 34.47 / 50.23
```

这可以写成：

```text
在本机可复现的 32CLIP + Cropformer 设置下，Stream4D adaptive mask_count top-k 在 Stream3D-style class-agnostic ScanNet evaluator 上超过了原 Stream3D-Cropformer baseline。
```

不能写成：

```text
已经复现 48CLIP + SAM2。
已经证明 RGB-only D4RT geometry 完全替代 ScanNet RGB-D / pose。
已经证明动态 4D tracking。
已经证明完整 4D semantic field reconstruction。
```

### 0.3 本文档对 v2 计划的核心修改

v2 计划的错误优先级：

```text
P0 = 修 fullmesh evaluator，否则指标不可信。
```

v3 计划改为：

```text
P0 = 确认并固定 Stream3D-style evaluator 协议，审计 pre_points 生成策略，防止 postprocess 通过不同 pre_points policy 造成不可解释收益。
P1 = tune/final split，避免在 312-scene val 上反复调参后直接报告。
P2 = 方法改进：减少碎片、提升稳定 support、真正触发多窗口 memory、在 Replica-Dynamic 上验证动态 4D tracking。
P3 = fullmesh 只作为 diagnostic，不作为主 benchmark。
```

---

## 1. 代码审核确认

### 1.1 已解压和对照的代码

Codex 应假设本次审核基于以下目录完成：

```text
/mnt/data/orig_stream3d/Code_Stream3D
/mnt/data/review_packet/Stream3D
/mnt/data/audit_unzip/Stream3D
```

其中：

```text
orig_stream3d/Code_Stream3D       = 原版 Stream3D zip 解压结果
review_packet/Stream3D            = 当前 Stream4D 审阅包源码
audit_unzip/Stream3D              = 审阅包重新解压目录
```

### 1.2 原版 Stream3D evaluator 的关键事实

原版 Stream3D 的 `evaluation/evaluate.py` 中，`assign_instances_for_scan()` 不是在完整 ScanNet mesh universe 上直接评估，而是读取 `TMP/{config}/{scene}_pre_points.npy` 后裁剪 GT 和 prediction。这个逻辑包括三步：

```python
loaded_array = np.load(title + config + '/' + scene_id + '_pre_points.npy')
pred_info = read_pridiction_npz(os.path.join(pred_file), loaded_array)
gt_ids = np.loadtxt(gt_file)
gt_ids = gt_ids[loaded_array]
```

`read_pridiction_npz()` 内部进一步执行：

```python
mask = pred['pred_masks']
mask = mask[idx]
```

因此，Stream3D 原版协议的评价 universe 是 `pre_points`，不是完整 mesh vertices。

### 1.3 原版 Stream3D pre_points 的关键事实

原版 `utils/Stream3D.py` 中，`export_new()` 接收：

```python
export_new(dataset, total_point_ids_list, total_mask_list, detected_points, args)
```

并写出：

```python
array = np.array(flat_unique)
np.save(title + args.config + '/' + scene_id + '_pre_points.npy', array)
```

`flat_unique` 来自 `detected_points`，而调用处传入的是 `all_detected_points`。`all_detected_points` 是由每帧 2D-to-3D mask projection 得到的所有候选点累计而来。原版后续还会把未分类点通过 neighbor point merging 分配给已有 masks。因此，在原版 Stream3D 中，`pre_points` 更接近“该方法检测到并参与最终导出的局部 scene support”，不是全场景 mesh，也不是 GT 支持集。

### 1.4 当前 Stream4D evaluator 的关键事实

当前 `review_packet/Stream3D/evaluation/evaluate.py` 与原版差异主要是把 TMP 路径参数化：

```python
parser.add_argument('--tmp_root', default='data/TMP')
parser.add_argument('--tmp_config', default='')
loaded_array = np.load(os.path.join(opt.tmp_root, tmp_config, scene_id + '_pre_points.npy'))
```

该修改是合理的，因为它允许不同 config 使用不同 TMP，不改变原 evaluator 的数学定义。

### 1.5 当前 Stream4D exporter / rescore 的待审计点

当前 `stream4d/export_scannet.py` 的 `_write_outputs()` 写：

```python
pre_points = np.flatnonzero(point_owner_counts > 0).astype(np.int64)
np.save(tmp_dir / f"{scene}_pre_points.npy", pre_points)
```

当前 `stream4d/rescore_scannet.py` 在筛选 top-k 之后写：

```python
pred_masks = pred['pred_masks'][:, kept_indices]
pre_points = np.flatnonzero(pred_masks.any(axis=1)).astype(np.int64)
np.save(tmp_out_dir / f"{seq_name}_pre_points.npy", pre_points)
```

这就是需要进一步区分的点：

```text
base export 的 pre_points = base method exported support。
rescore 的 pre_points = top-k 后剩余 prediction union。
```

这不等于“虚假指标”，因为原版 evaluator 也使用 method-specific `pre_points`。但为了让结论更稳，Codex 必须实现并比较两个 policy：

```text
recompute_pre_points: 当前策略，pre_points 随 top-k 后 prediction union 改变。
inherit_pre_points: 新增策略，rescore 后沿用 input_config 的 pre_points，不随 top-k 缩小。
```

若 `inherit_pre_points` 仍能超过 baseline，当前结论会非常稳；若只有 `recompute_pre_points` 超过，则要把结果表述为“在 Stream3D-style recomputed observed-support evaluator 下有效”，并继续优化方法本体。

---

## 2. 当前结果的可信表述

### 2.1 可以保留的结果

当前 best 结果可以保留为 Stream3D-style evaluator 下的可复现实验：

```text
Config: stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one
Backbone: Cropformer
D4RT: OpenD4RT_32CLIP_9Dataset_NoAUG
Frames: first 32 stride-10 frames per scene
Export: rgbd_eval
Evaluation: Stream3D-style cropped TMP evaluator
AP / AP50 / AP25: 20.37 / 35.52 / 55.06
Baseline Stream3D-Cropformer: 20.11 / 34.47 / 50.23
```

论文或汇报中可以写：

```text
Under the original Stream3D evaluation protocol, our current Stream4D MVP with adaptive mask-count selection achieves 20.37 / 35.52 / 55.06 on ScanNet class-agnostic evaluation, slightly outperforming the Stream3D-Cropformer baseline 20.11 / 34.47 / 50.23.
```

### 2.2 必须补充的限制

必须同步写清楚：

```text
This is not the paper-exact 48CLIP + SAM2 setting.
This is not D4RT-native geometry export; rgbd_eval still uses ScanNet RGB-D/pose for evaluation bridge.
This does not prove dynamic 4D tracking because ScanNet is static.
This result is currently based on validation-set postprocessing search and must be validated under locked hyperparameters.
```

### 2.3 当前结果的真实含义

当前 best 的主要收益来自：

$$
	ext{many fragmented proposals} \xrightarrow{\text{mask-count top-k}} 
	ext{few stable high-precision proposals}
$$

它不是通过更完整的 scene coverage 获胜。已记录的统计是：

```text
Stream3D-Cropformer union_ratio mean ≈ 0.8702
Stream4D adaptive top-k union_ratio mean ≈ 0.0575
```

在 Stream3D-style cropped evaluator 下，这不是错误；但从方法目标看，这说明当前 Stream4D 仍然更像“高精度候选筛选”，还不是完整 4D semantic reconstruction。

---

## 3. 需要立刻修改的代码任务

### 3.1 P0：新增 pre_points policy，审计 rescore 是否靠缩小 universe 获益

修改文件：

```text
Stream3D/stream4d/rescore_scannet.py
Stream3D/stream4d/reexport_scannet.py
Stream3D/stream4d/export_scannet.py, 如有必要
```

新增参数：

```text
--pre-points-policy recompute
--pre-points-policy inherit
--pre-points-policy fixed_path
--fixed-pre-points-root data/TMP
--fixed-pre-points-config stream4d_scannet_32f_ioc075_fixmem
```

定义：

```text
recompute: 写当前 pred_masks.any(axis=1)，保持现状。
inherit: 直接复制 input_config 的 pre_points 到 output_config。
fixed_path: 从指定 root/config 读取 pre_points，用于统一诊断。
```

验收：

```text
同一个 output_config 目录下 summary.json 必须记录 pre_points_policy。
每个 scene 记录 input_pre_points_count、output_pre_points_count、prediction_union_count。
如果 pre_points_policy=inherit，则 output_pre_points_count 必须等于 input_pre_points_count。
如果 pre_points_policy=recompute，则 output_pre_points_count 必须等于 prediction_union_count。
```

判断：

```text
如果 recompute 超 baseline，inherit 也超 baseline：当前 top-k 结果可信度高。
如果 recompute 超 baseline，inherit 不超 baseline：当前结果仍可作为 Stream3D-style recomputed support result，但主论文 claim 要更谨慎。
如果二者都不超 baseline：继续方法改进，不再围绕旧 best 讲优势。
```

### 3.2 P0：新增 evaluator protocol audit 工具

新增文件：

```text
tools/audit_stream3d_eval_protocol.py
```

功能：

```text
读取原版 Code_Stream3D/evaluation/evaluate.py 和当前 evaluation/evaluate.py。
检查是否都读取 pre_points。
检查当前 evaluator 是否只增加 tmp_root/tmp_config，而没有改变 AP 计算。
读取每个 config 的 TMP 和 prediction，统计 pre_points 与 prediction union 的关系。
输出 metric_protocol_audit.md 和 metric_protocol_audit.json。
```

必须输出字段：

```text
scene_id
num_scene_vertices
num_pre_points
pre_points_ratio
num_prediction_union
prediction_union_ratio
pre_points_equals_prediction_union
prediction_union_subset_of_pre_points
num_pred_instances
num_gt_instances_in_pre_points
num_gt_instances_fullmesh
```

验收标准：

```text
能够自动证明原版 evaluator 与当前 evaluator 都是 cropped-TMP protocol。
能够自动指出 rescore 的 pre_points_policy。
能够生成 table：baseline / MVP / min250 / adaptive top-k 的 pre_points 统计。
```

### 3.3 P0：修复 local set-cover 权重错配 bug

当前 `local_4d_filter.py` 有一个确定 bug：使用 `set` 后再和 `obs.weights` zip，顺序不稳定。必须改为按数组顺序构造 weights。

错误形式：

```python
support = {(frame_id, carrier_id), ...}
for key, weight in zip(support, obs.weights.tolist()):
    weights[key] = max(weights.get(key, 0.0), float(weight))
```

修复形式：

```python
for cid, weight in zip(obs.carrier_ids.tolist(), obs.weights.tolist()):
    key = (int(obs.frame_id), int(cid))
    weights[key] = max(weights.get(key, 0.0), float(weight))
```

新增单元测试：

```text
tests/test_local_4d_filter_weight_alignment.py
```

验收：

```text
构造 3 个 carrier，权重分别为 0.1、0.5、0.9。
打乱 support set 的插入顺序。
修复后 greedy selection 结果稳定。
```

### 3.4 P1：增加 tune/final split，避免 validation overfitting

新增：

```text
tools/make_scannet_stream4d_splits.py
splits/scannet_tune.txt
splits/scannet_final.txt
```

建议：

```text
按 scene id 排序后 deterministic 1:1 split。
只在 tune split 搜索 ratio/min/max/threshold。
最终只在 final split 报一次。
```

所有实验 summary 必须记录：

```text
selected_on = scannet_tune
reported_on = scannet_final
hyperparameters_locked = true
```

验收：

```text
如果在 final split 上 AP 或 AP50 仍超过 Stream3D-Cropformer 对应 split baseline，才可写 stronger claim。
如果 tune 超但 final 不超，必须报告 overfitting。
```

### 3.5 P1：明确 D4RT 几何与 rgbd_eval 的边界

当前 `rgbd_eval` 使用 ScanNet depth/pose 将 UV support 映射到 ScanNet mesh，是 evaluation/export adapter，不是方法内部 D4RT-native geometry。

新增日志字段：

```text
geometry_source_for_grouping = d4rt_uv_tracks
geometry_source_for_export = scannet_rgbd_pose or d4rt_sim3
uses_gt_semantics = false
uses_gt_instance_masks = false
uses_gt_depth_pose_for_grouping = false
uses_gt_depth_pose_for_export = true/false
sim3_used_in_method = false
sim3_used_in_evaluation_export = true/false
```

验收：

```text
任何 report table 必须包含 export column: rgbd_eval / d4rt_sim3_eval。
任何 d4rt_sim3_eval 的 Sim3 只能在 evaluation/export 使用，不能影响 grouping、memory、top-k selection。
```

---

## 4. 方法改进方向：从高精度 top-k 走向真正可超越 Stream3D

当前结果只是“接近/略超 Stream3D-Cropformer”，且靠后处理筛选。下一步要让方法优势更像方法本体，而不是 top-k trick。

### 4.1 核心假设

Stream3D 的优势来自稳定的 2D-to-3D point support 和 3D manifold refinement；Stream4D 的潜在优势应来自 D4RT carrier 的跨时间物理对应。当前 MVP 没有发挥这个优势，原因是：

```text
只看前 32 stride-10 frames，绝大多数 scene 没看完。
每个 scene 只有一个 window，ObjectMemory 没被验证。
carrier support 稀疏，导致 object masks 很小。
naive densification 和 point dilation 会引入冲突。
当前 memory 主要靠 carrier overlap，不能处理动态重现或跨窗口无共享 carrier 的情况。
```

因此 v3 方法目标是：

$$
	ext{sparse but stable carrier evidence}
+ 	ext{reliable object-level densification}
+ 	ext{appearance/motion memory}
\rightarrow
	ext{more complete and less fragmented Stream3D-style support}
$$

### 4.2 改进一：proposal 级别的可靠性评分，而不是纯 top-k

当前 `mask_count` 有效，说明“被多个 2D masks 支持”是强信号。下一版不要只做 adaptive top-k，而是构造可解释 confidence：

$$
q_k = w_1 Q_{mask} + w_2 Q_{carrier} + w_3 Q_{temporal} + w_4 Q_{visibility} + w_5 Q_{area} - w_6 Q_{conflict}
$$

第一版不训练权重，用固定归一化：

```text
Q_mask       = log(1 + unique_mask_count)
Q_carrier    = log(1 + unique_carrier_count)
Q_temporal   = number_of_unique_frames / window_size
Q_visibility = mean D4RT visibility/confidence over support
Q_area       = clipped log area score, avoid tiny fragments and huge stuff leakage
Q_conflict   = point overlap / carrier conflict with other objects
```

实验要对比：

```text
score=one + adaptive top-k, 当前 best
quality score + no top-k
quality score + threshold
quality score + adaptive object budget
mask_count only
area only
carrier_count only
```

判断标准：

```text
quality score 在 tune split 上不低于 mask_count top-k。
final split 上 AP 不下降超过 0.2，AP50 或 AP25 有提升。
object count 不显著爆炸。
```

如果失败：

```text
先保留 mask_count top-k 为 baseline。
查看 quality score 的失败排序，保存 top false positive 和 missed true positive 可视化。
不要用 GT 训练权重；只能用无监督分布归一化和固定规则。
```

### 4.3 改进二：inherit-pre_points 下提升，而不是只靠 recompute-pre_points

真正要证明方法更强，应该让 top-k 不依赖缩小 evaluation support。实现 `inherit_pre_points` 后，优化目标改成：

```text
在 inherit_pre_points evaluator 下仍超过 Stream3D-Cropformer，或至少接近，并在 dynamic tracking 上明显更强。
```

针对 inherit policy，top-k 会保留大 pre_points universe，漏掉的 GT support 会产生更多 false negative。要提分，需要：

```text
减少碎片，但不能过度删 support。
用 object-level densification 补回稳定支持点。
用 memory 合并同一物体的多窗口碎片。
```

### 4.4 改进三：carrier-guided reliable densification

naive dense mask backprojection 已经失败，因为覆盖增加但冲突更大。下一版只做可靠 densification。

对于 object $
\mathcal{O}_k$，定义它支持的 mask observations：

$$
\mathcal{E}_k = \{(t, m, \alpha_{t,m})\}
$$

对 mask 内候选 pixel $p$ 计算：

$$
R_k(p,t)=a_c R_{seed}+a_m R_{mask}+a_b R_{boundary}+a_a R_{appearance}-a_x R_{conflict}
$$

第一版实现不需要复杂模型：

```text
R_seed: pixel 到该 object carrier UV seeds 的距离，越近越高。
R_mask: 该 mask 被多少 carrier / 多少帧支持。
R_boundary: mask 边界附近降权，避免边界污染。
R_appearance: crop feature 与 object feature 的相似度，初版可用 RGB histogram / CLIP crop。
R_conflict: 同一 pixel 或近邻 3D point 被多个 object 支持时降权。
```

具体策略：

```text
每个 object 只 densify top-M mask observations。
每个 mask 只保留与 carrier seeds 同 connected component 的区域。
对大 mask 做 boundary erosion，对小 mask 不 erosion 或弱 erosion。
每个 ScanNet vertex 只分给 reliability 最高的 object。
每个新增点必须来自至少一个高可信 carrier seed 的空间邻域或同 component。
```

必须记录：

```text
num_seed_carriers
num_candidate_pixels
num_kept_pixels
num_backprojected_vertices
num_added_vertices
num_conflict_vertices
reliability_mean
boundary_drop_ratio
component_drop_ratio
```

判断标准：

```text
inherit_pre_points 下 AP50 提升至少 +1.0。
recompute_pre_points 下 AP 不下降超过 0.3。
conflict_rate 不超过 MVP 的 1.5x。
tiny_mask_ratio 下降，median mask area 上升。
```

失败时 Codex 先尝试：

```text
coverage 上升但 AP 下降：提高 R_seed 权重，增加 boundary erosion。
小物体丢失：对 area 小于阈值的 mask 关闭 erosion，降低 min pixels。
大物体仍稀疏：增加 top-M，但要求多帧支持。
conflict 太高：启用 winner-take-all vertex assignment。
速度太慢：mask pixels 按 stride 采样，先在 20 scenes 调通。
```

### 4.5 改进四：ObjectMemory4D-v2

当前 memory 只靠 carrier overlap。下一版 matching score：

$$
S(P_i, O_j)=w_c S_c+w_a S_a+w_m S_m+w_g S_g+w_l S_l-w_x S_x
$$

其中：

```text
S_c: carrier overlap / IoC，只对 overlap windows 强。
S_a: appearance similarity，CLIP/DINO/RGB histogram。
S_m: D4RT trajectory/centroid motion consistency。
S_g: export coordinate 或 D4RT coordinate 下的 centroid proximity。
S_l: open-vocab label distribution consistency，初版可选。
S_x: same-frame exclusivity / point conflict penalty。
```

状态机：

```text
active: 当前窗口可见并匹配。
lost: 最近若干窗口不可见，但可被 reactivated。
reactivated: lost 后重新匹配。
split_candidate: 一个历史 object 匹配多个 proposal。
merge_candidate: 多个历史 object 匹配一个 proposal。
```

必须记录：

```text
num_windows
num_created
num_matched
num_lost
num_reactivated
object_growth_rate
fragmentation_per_gt, if GT available in ScanNet evaluation adapter
merge_error_per_gt, if GT available in offline analysis
match_score_components.csv
```

判断标准：

```text
多窗口 128f scene0050 不再出现 1000+ objects。
full sequence object count 不超过 one-window 的 2.5x。
AP50 不低于 one-window baseline。
num_matched > 0，并能可视化同一 object 跨窗口颜色一致。
```

失败时 Codex 先尝试：

```text
object explosion：提高 creation threshold，增加 proposal pre-merge。
over-merge：提高 same-frame exclusivity penalty，降低 S_g 权重。
reactivation 失败：降低 lost-object threshold，增加 S_a 权重。
appearance 不稳：尝试 DINO/CLIP/RGB histogram ensemble。
```

### 4.6 改进五：D4RT-native Sim3 evaluation export

用户明确说明：D4RT 几何不是真实尺度，需要 Sim3 技巧，**这是评估时的事情**。因此：

```text
Sim3 只用于 evaluation/export adapter。
Sim3 不进入 grouping、memory、top-k、semantic fusion。
```

新增：

```text
stream4d/sim3.py
stream4d/export_scannet_d4rt_sim3.py
```

估计：

$$
T^* = \arg\min_{s,R,t} \sum_i \omega_i \lVert s R x_i^{D4RT} + t - x_i^{ScanNet} \rVert_2^2
$$

anchors：

```text
source: D4RT xyz for high-confidence carriers / dense grid points
target: ScanNet RGB-D/pose backprojected points at same frame/pixel
filter: visibility high, confidence high, finite depth, residual inlier
```

输出：

```text
sim3_scale
sim3_rotation_det
sim3_translation_norm
anchor_count
inlier_ratio
median_residual
p90_residual
export_nn_hit_rate
d4rt_sim3_AP/AP50/AP25 under Stream3D-style evaluator
```

判断标准：

```text
median residual < 0.10m: 可作为主 diagnostic。
AP50 与 rgbd_eval bridge 差距 < 20% relative: D4RT geometry export 有希望。
残差过大：不能 claim D4RT geometry 替代 RGB-D，只报告 carrier grouping + rgbd_eval。
```

失败时 Codex 先尝试：

```text
检查 D4RT camera reference t_cam。
检查坐标轴方向和单位。
改 scene-level Sim3 为 window-level Sim3。
只用低运动 background-like anchors。
只做 scale+translation 与 full Sim3 对比。
检查 frame resize 和 UV 对齐。
```

---

## 5. ScanNet 实验计划

ScanNet 仍然是与 Stream3D 对齐的静态 3D segmentation benchmark。实验重点不是证明动态，而是证明：在原 Stream3D evaluator 下，Stream4D 的 D4RT carrier evidence 能否提升或稳定超过 Stream3D-Cropformer。

### 5.1 ScanNet 总目标

ScanNet 实验要回答：

```text
Q1: 当前 evaluator 是否与原 Stream3D 完全对齐？
Q2: 当前 best 是否在 inherit_pre_points 下仍有效？
Q3: 当前 best 是否在 tune/final split 下仍有效？
Q4: reliable densification 能否提升 support 而不引入过多噪声？
Q5: memory-v2 能否让多窗口不碎片爆炸？
Q6: D4RT-native Sim3 export 与 rgbd_eval bridge 差距多少？
```

### 5.2 S0：Evaluator protocol audit

#### 假设

$H_{S0}$：当前 `evaluation/evaluate.py` 与原版 Stream3D evaluator 在 AP 计算和 `pre_points` 裁剪逻辑上保持一致；因此 cropped-TMP evaluator 可以作为主对齐协议。

#### 实验设置

Codex 运行：

```bash
python -m tools.audit_stream3d_eval_protocol \
  --orig-stream3d-root /mnt/data/orig_stream3d/Code_Stream3D \
  --current-root . \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem,stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one \
  --seq-list splits/scannet.txt \
  --output outputs/audit/eval_protocol_audit.md
```

#### 必须记录

```text
orig_evaluator_reads_pre_points = true
current_evaluator_reads_pre_points = true
current_changes_only_tmp_path = true/false
per_config_pre_points_ratio
per_config_prediction_union_ratio
pre_points_equals_prediction_union
baseline_tmp_config_used
stream4d_tmp_config_used
```

#### 判断标准

```text
如果 current evaluator 只参数化 TMP 路径，没有改变 AP matching，则 evaluator 通过。
如果 current evaluator 改了 min_region_size、overlaps、class handling、void handling，必须列出差异并回滚。
```

#### 可视化和输出

```text
outputs/audit/eval_protocol_audit.md
outputs/audit/eval_protocol_audit.json
outputs/audit/pre_points_ratio_hist.png
outputs/audit/pre_points_vs_pred_union_scatter.png
```

#### 不满足时 Codex 先尝试

```text
若 evaluator diff 发现 AP 逻辑被改，先恢复原版逻辑，只保留 tmp_root/tmp_config 参数。
若 baseline tmp_config 指错，重新用 data/TMP/scannet 评估 baseline。
若某些 scene TMP 缺失，先补生成，不要跳 scene。
```

### 5.3 S1：pre_points policy 实验

#### 假设

$H_{S1}$：如果 adaptive top-k 的收益来自真实候选质量提升，则 `inherit_pre_points` 下仍应保持接近或超过 baseline；如果收益主要来自缩小 evaluation support，则 `inherit_pre_points` 会明显下降。

#### 实验设置

对同一组 candidates 跑：

```text
P0: MVP base, no rescore
P1: adaptive top-k + recompute_pre_points, 当前 best
P2: adaptive top-k + inherit_pre_points from MVP
P3: min250 + recompute_pre_points
P4: min250 + inherit_pre_points from MVP
```

命令形态：

```bash
CFG=stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one_inheritpre
python -m stream4d.rescore_scannet \
  --seq-list splits/scannet.txt \
  --backbone Cropformer \
  --input-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config "$CFG" \
  --score-mode one \
  --select-mode mask_count \
  --filter-max-instances-ratio 0.14 \
  --filter-min-instances 8 \
  --filter-max-instances 18 \
  --pre-points-policy inherit \
  --debug-root outputs/stream4d_rescore_full
```

#### 必须记录

```text
AP / AP50 / AP25 under Stream3D evaluator
num_pre_points
prediction_union_points
pre_points_ratio
prediction_union_ratio
num_kept_instances
num_gt_instances_after_crop
num_gt_instances_full, diagnostic only
```

#### 判断标准

```text
若 P2 AP >= Stream3D baseline AP - 0.3 且 AP50 > baseline AP50，则 top-k 质量确实强。
若 P1 超 baseline 但 P2 明显低于 baseline，说明 recompute policy 对结果影响大，主 claim 要谨慎。
若 P2 也超 baseline，后续可把 inherit policy 作为更强主结果。
```

#### 可视化

```text
P1 vs P2 per-scene AP delta
pre_points shrink ratio vs AP gain scatter
kept object count vs AP gain scatter
Top 10 scenes where recompute improves most: 可视化 pre_points 与 missed GT
```

#### 不满足时 Codex 先尝试

```text
如果 inherit 掉分大，优先做 reliable densification 和 memory-v2，而不是继续调 ratio。
如果某些 scenes inherit 掉分特别大，检查是否 top-k 删除了大物体 support。
如果 recompute/inherit 差异小，则保留 recompute 作为 Stream3D-style default，同时在附录报告 inherit。
```

### 5.4 S2：tune/final split 实验

#### 假设

$H_{S2}$：当前 adaptive top-k 的超参数不是过拟合 312-scene val；在 tune split 选出的参数在 final split 上仍稳定。

#### 实验设置

生成 split：

```bash
python -m tools.make_scannet_stream4d_splits \
  --input splits/scannet.txt \
  --tune-output splits/scannet_tune.txt \
  --final-output splits/scannet_final.txt \
  --seed 20260607
```

在 tune 上搜索：

```text
ratio: 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18
min_instances: 6, 8, 10
max_instances: 14, 16, 18, 20, 24
select_mode: mask_count, mask_area_sqrt, area
score_mode: one only first
pre_points_policy: recompute and inherit separately
```

在 final 上只跑每个 policy 的 best config。

#### 必须记录

```text
tune AP/AP50/AP25
final AP/AP50/AP25
best hyperparameters
rank of selected config on final, if all final grid optionally evaluated later
overfit_gap = tune_metric - final_metric
```

#### 判断标准

```text
overfit_gap(AP) < 0.5 可接受。
final AP 或 AP50 超过 Stream3D-Cropformer final split baseline。
如果只 AP25 超，不能写 overall outperform。
```

#### 可视化

```text
heatmap ratio/max_instances vs AP on tune
same selected point on final
per-scene delta histogram vs baseline
```

#### 不满足时 Codex 先尝试

```text
如果 final 掉分，减少超参数自由度，固定 ratio=0.14 only。
如果 select_mode 过拟合，退回 mask_count。
如果 max_instances 敏感，改成 quality threshold + soft budget。
```

### 5.5 S3：quality score 替代纯 top-k

#### 假设

$H_{S3}$：可解释 quality score 能在不使用 GT 的情况下稳定排序候选，减少 AP 对固定 top-k 的依赖。

#### 实验设置

对比：

```text
Q0: current adaptive mask_count top-k, score=one
Q1: quality score ranking, no top-k
Q2: quality score + threshold
Q3: quality score + adaptive budget
Q4: quality score without conflict
Q5: quality score without temporal span
Q6: quality score without visibility
```

#### 必须记录

```text
AP/AP50/AP25
precision-recall curve points
score distribution
false positive rank distribution
true positive rank distribution, offline analysis only
num_kept_instances
pre_points_ratio
```

#### 判断标准

```text
Q3 在 final split AP 不低于 Q0 - 0.2，并且 AP50/AP25 至少一项提升。
score 与 object quality 的 Spearman correlation 为正，offline diagnostic。
```

#### 可视化

```text
score histogram for TP-like vs FP-like predictions, offline diagnostic
Top 20 high-score false positives
Top 20 low-score missed / fragmented objects
```

#### 不满足时 Codex 先尝试

```text
如果 score ranking 伤 AP，保留 score=one 用于 evaluator，只把 quality 用于 selection。
如果 area 项伤害小物体，改成 clipped area prior。
如果 visibility 噪声大，降低权重或只用于 gating。
```

### 5.6 S4：reliable densification 实验

#### 假设

$H_{S4}$：D4RT carrier-guided densification 比 naive dense mask backprojection 更可靠，能在保留 precision 的同时提高 support。

#### 实验设置

对比：

```text
D0 sparse carrier_uv, 当前 MVP
D1 naive mask_backproject, 已知失败对照
D2 top-M mask observations only
D3 carrier-neighborhood connected component
D4 D3 + boundary erosion
D5 D4 + conflict suppression
D6 D5 + multi-frame support voting
```

分别在：

```text
scene0050_00
20-scene tune subset
scannet_tune
scannet_final
```

逐级推进。

#### 必须记录

```text
AP/AP50/AP25 under recompute and inherit policies
prediction_union_ratio
pre_points_ratio
conflict_rate
median_mask_area
tiny_mask_ratio
export_nn_hit_rate
num_candidate_pixels
num_kept_pixels
runtime_export_seconds
```

#### 判断标准

```text
D5/D6 相比 D0 在 inherit_pre_points AP50 +1.0 以上。
conflict_rate 不超过 D0 的 1.5x。
tiny_mask_ratio 下降。
union_ratio 有增长，但不是靠大范围错误扩张。
```

#### 可视化

```text
RGB + carrier seeds
2D mask component kept/dropped overlay
3D exported vertices on mesh
conflict vertices heatmap
per-object best-IoU histogram
```

#### 不满足时 Codex 先尝试

```text
如果 D1-D6 都下降，先不要全量跑，聚焦 failure scene。
如果大物体补得好但小物体丢，按 area 分层参数。
如果 conflict 高，先 winner-take-all，再尝试 soft suppression。
如果 export 慢，减小 max_pixels 或 stride。
```

### 5.7 S5：multi-window ObjectMemory4D-v2 实验

#### 假设

$H_{S5}$：memory-v2 能解决 naive 多窗口的碎片爆炸，让看更多帧真正提升 coverage 和 AP。

#### 实验设置

对比：

```text
M0 one-window 32f current best
M1 128f old memory
M2 128f memory-v2
M3 full stride-10 old memory
M4 full stride-10 memory-v2
M5 full stride-10 memory-v2 + reliable densification
```

先跑：

```text
scene0050_00
20 scenes tune subset
```

再跑：

```text
scannet_tune
scannet_final
```

#### 必须记录

```text
AP/AP50/AP25
num_windows
num_created
num_matched
num_lost
num_reactivated
final_num_objects
object_growth_rate
fragmentation_per_gt, offline diagnostic
merge_error_per_gt, offline diagnostic
runtime per window
peak GPU memory
```

#### 判断标准

```text
128f memory-v2 AP50 >= 32f current best AP50 - 0.5。
full stride-10 memory-v2 object count 不超过 32f 的 2.5x。
fragmentation 比 old memory 降低 30%。
num_matched > 0，且有跨窗口可视化成功案例。
```

#### 可视化

```text
object timeline: window index vs object id
same object across windows with same color
carrier trajectories colored by object id
merge/split failure panels
```

#### 不满足时 Codex 先尝试

```text
object explosion：提高 matching threshold，增加 pre-merge。
over-merge：加 same-frame exclusivity penalty。
long sequence drift：分 chunks，chunk 内 memory-v2，chunk 间只 appearance+Sim3。
运行太慢：先固定 128f，不跑 full。
```

### 5.8 S6：D4RT Sim3 export 实验

#### 假设

$H_{S6}$：evaluation-only Sim3 能把 D4RT xyz support 对齐到 ScanNet mesh coordinate，使 d4rt_sim3_eval 接近 rgbd_eval bridge。

#### 实验设置

对同一 object memory 导出：

```text
E0 rgbd_eval bridge
E1 d4rt_sim3_scene
E2 d4rt_sim3_window
E3 d4rt_sim3_window + overlap stitching
```

#### 必须记录

```text
AP/AP50/AP25 under Stream3D evaluator
sim3_scale
anchor_count
inlier_ratio
median_residual
p90_residual
export_nn_hit_rate
num_points_outside_radius
```

#### 判断标准

```text
E1/E2 的 AP50 与 E0 差距 < 20% relative，则 D4RT-native export 有希望。
median residual < 0.10m。
若失败，保留 E0 为主 ScanNet bridge，把 E1/E2 作为 diagnostic。
```

#### 可视化

```text
D4RT point cloud after Sim3 vs ScanNet mesh
residual heatmap
failed anchors in image space
per-window scale drift plot
```

#### 不满足时 Codex 先尝试

```text
检查 axis/camera convention。
只用 static/high confidence anchors。
scene-level 改 window-level。
先 scale-only，再 full Sim3。
检查 D4RT query 的 t_cam 选择。
```

---

## 6. Replica-Dynamic / Dynamic Replica 实验计划

动态实验是本工作的真正差异点。ScanNet 只能说明静态 segmentation；Replica-Dynamic / Dynamic Replica 要证明 D4RT carriers 让 Stream4D 支持 4D reconstruction and tracking。

### 6.1 数据路径和命名

Codex 不要硬编码一个名字。实现 path resolver：

```text
candidate roots:
  data/dynamic-replica/v2
  data/replica-dynamic/v2
  /mnt/data/users/chengshun.wang/pjs/sray_plus/data/dynamic-replica/v2
  /mnt/data/users/chengshun.wang/pjs/sray_plus/data/replica-dynamic/v2
```

新增：

```text
tools/check_dynamic_replica_env.py
stream4d/dynamic_replica_stream.py
stream4d/run_dynamic_replica.py
```

环境检查必须输出：

```text
num_sequences
num_frames_per_sequence
has_rgb
has_depth
has_camera
has_trajectory_gt
has_instance_masks
has_semantic_labels
has_object_ids
image_resolution
depth_scale
camera_convention
```

### 6.2 D0：D4RT adapter sanity

#### 假设

$H_{D0}$：当前 `D4RTAdapter` 在 Dynamic Replica 上输出的 UV / XYZ / visibility / confidence 与 OpenD4RT 官方 inference 一致，可作为 carrier backbone。

#### 实验设置

选 10 个 sequences，每个 48 frames，采样：

```text
random visible points
moving foreground points, if available
mask boundary points
background points
```

对比：

```text
D4RTAdapter output
OpenD4RT official infer_track_3d.py output
```

#### 必须记录

```text
adapter_vs_official_xyz_diff_mean
adapter_vs_official_uv_diff_mean
uv_in01_rate
visibility_mean
confidence_mean
track_length_visible
runtime_per_query
```

如果有 GT tracks：

```text
APD3D
AJ
OA
3D EPE
trajectory L1 after evaluation-only scale/Sim3 alignment
```

#### 判断标准

```text
adapter 与 official diff 在数值容差内。
uv_in01_rate 和 visibility 不异常低。
若 D4RT 自身 track 明显失败，先修 D4RT adapter，不进入 semantic tracking。
```

#### 可视化

```text
RGB video with 2D tracks
3D tracks in common coordinate
visibility timeline
confidence timeline
```

#### 不满足时 Codex 先尝试

```text
检查 frame resize/aspect ratio。
检查 RGB channel order 和 normalization。
检查 t_src/t_tgt/t_cam indexing。
检查 confidence/visibility 是否 sigmoid。
检查 camera coordinate convention。
```

### 6.3 D1：动态 object identity tracking

#### 假设

$H_{D1}$：相比 Stream3D-style spatial overlap memory，ObjectMemory4D-v2 使用 D4RT motion + appearance 能显著减少 ID switch 和 fragmentation。

#### 实验设置

方法对比：

```text
B0 per-frame 2D masks without tracking
B1 Stream3D-style overlap memory adapted to dynamic data
B2 Stream4D MVP carrier-overlap memory
B3 Stream4D-v2 motion+appearance memory
B4 B3 without motion term
B5 B3 without appearance term
```

如果有 GT object masks / IDs，使用 GT。若没有，使用 SAM2 video masks 或手工 selected moving object 做 pseudo tracking，但表格必须标注 pseudo。

#### 必须记录

```text
IDF1
ID Precision
ID Recall
ID switches
fragmentation count
track purity
object reactivation success
average track length
lost duration before reactivation
per-frame mask IoU, if GT masks exist
4D IoU, if GT 4D masks exist
```

4D IoU：

$$
IoU_{4D}(P,G)=\frac{\sum_t |P^t \cap G^t|}{\sum_t |P^t \cup G^t|}
$$

Track purity：

$$
Purity(P)=\max_g \frac{\sum_t |P^t \cap G_g^t|}{\sum_t |P^t|}
$$

#### 判断标准

```text
B3 比 B2 IDF1 提升 >= +5.0。
B3 ID switches 降低 >= 30%。
B3 fragmentation 降低 >= 30%。
至少 5 个 occlusion/reappearance 成功案例可视化。
```

#### 可视化

```text
per-frame RGB + object ID overlay
carrier tracks colored by object ID
object state timeline active/lost/reactivated
ID switch marker video
occlusion/reappearance case panel
```

#### 不满足时 Codex 先尝试

```text
ID switch 多：提高 appearance 权重和 same-frame exclusivity penalty。
fragmentation 多：降低 reactivation threshold，引入 temporal decay。
over-merge：提高 motion consistency 权重。
appearance 不稳：DINO/CLIP/RGB histogram ensemble。
```

### 6.4 D2：dynamic semantic 4D field query

#### 假设

$H_{D2}$：把 open-vocabulary semantic evidence 绑定到 object trajectories 后，文本查询比逐帧 CLIP 更稳定，能支持 time-sensitive query。

#### 查询类型

```text
class query: chair, table, person, ball, cup, etc.
motion query: moving object, object being carried, object entering scene
temporal query: same object before occlusion, same object after reappearance
relation query: object near person, object on table, if visible and auditable
```

#### 必须记录

```text
label temporal consistency
text-query top1 stability
CLIP score margin
object-language entropy
query localization IoU, if GT labels/masks exist
false temporal grounding rate, human-audited subset
```

#### 判断标准

```text
object-level label consistency 比 per-frame CLIP 提升 >= 20%。
query videos 中不能出现明显跨 object 漂移。
若无 semantic GT，只能报 qualitative + human-auditable sample，不报 semantic AP。
```

#### 可视化

```text
query text
selected object trajectory
per-frame mask overlay
score over time plot
failure reason tag: CLIP hallucination / wrong mask / wrong track / occlusion
```

#### 不满足时 Codex 先尝试

```text
CLIP hallucination：降低语言权重，只做候选标签。
label 漂移：用 object-level memory feature，不用 per-frame label。
temporal query 失败：先只支持 class query 和 moving/static query。
```

### 6.5 D3：D4RT geometry evaluation with scale / Sim3 alignment

#### 假设

$H_{D3}$：D4RT geometry 经 evaluation-only scale / Sim3 alignment 后，可以支持 3D trajectory evaluation。alignment 不进入方法内部。

#### 实验设置

对 GT trajectories：

```text
A0 no alignment, diagnostic only
A1 scale-only alignment
A2 Sim3 alignment
A3 chunk-level Sim3 alignment
```

#### 必须记录

```text
APD3D
AJ
OA
3D EPE
trajectory L1
scale factor
Sim3 residual
chunk boundary drift
visibility accuracy
```

#### 判断标准

```text
Sim3 后 D4RT track 达到可用水平，才继续做 semantic 4D tracking 主结论。
如果 D4RT track 本身失败，必须拆分为 geometry failure 和 semantic failure。
```

#### 可视化

```text
GT trajectory vs predicted trajectory
Sim3 aligned point cloud
chunk boundary drift plot
per-object motion vectors
```

#### 不满足时 Codex 先尝试

```text
缩短 window。
增加 overlap。
只用 high-confidence anchors。
检查 camera convention 和 depth decode。
分 static/dynamic points 单独看误差。
```

---

## 7. 最终表格与图

### 7.1 ScanNet 主表

主表使用 Stream3D-style evaluator，不再把 fullmesh 作为主协议。

```text
Method | Eval protocol | pre_points policy | Export | AP | AP50 | AP25 | pre_points_ratio | pred_union_ratio | #inst | time/frame
```

必须包含：

```text
Stream3D-Cropformer baseline
Stream4D-MVP sparse carrier
Stream4D adaptive top-k recompute_pre_points
Stream4D adaptive top-k inherit_pre_points
Stream4D-v3 quality score
Stream4D-v3 reliable densification
Stream4D-v3 memory-v2
Stream4D-v3 d4rt_sim3_eval
```

### 7.2 ScanNet 消融表

```text
D4RT carrier | set-cover fix | quality score | densify | memory-v2 | pre_points policy | AP | AP50 | AP25 | conflict | #objects
```

### 7.3 Protocol audit 表

```text
Config | pre_points policy | pre_points ratio | pred union ratio | equals union | AP | AP50 | AP25
```

### 7.4 Dynamic Replica 主表

```text
Method | IDF1 | IDSW | Frag | Track purity | 4D IoU | APD3D | AJ | OA | Reactivation success
```

若无 GT object masks，表名必须写：

```text
Pseudo object-consistency evaluation, not benchmark GT result
```

### 7.5 必须可视化

ScanNet：

```text
predicted instance mesh overlay
pre_points support overlay
prediction union overlay
carrier seeds overlay
densified pixels overlay
false positives / missed GT diagnostic, offline only
Sim3 residual overlay
```

Replica-Dynamic：

```text
2D tracks overlay
3D trajectories
object ID timeline
occlusion reactivation cases
text query to object trajectory videos
```

---

## 8. 执行顺序

### Phase 0：协议修正与审计

```text
1. 实现 pre_points_policy。
2. 实现 audit_stream3d_eval_protocol.py。
3. 重新跑 current best 的 recompute / inherit 两个版本。
4. 更新 summary 和表格命名。
```

### Phase 1：锁定超参和 split

```text
1. 生成 scannet_tune/final。
2. 在 tune 上选择 ratio/min/max。
3. 在 final 上报告 locked result。
```

### Phase 2：修 bug 和最小方法改进

```text
1. 修 local_4d_filter weight bug。
2. 修 carrier_sampler 潜在长度错配。
3. 加 tests。
4. 实现 quality score。
```

### Phase 3：支持提升

```text
1. reliable_densifier.py。
2. 先 scene0050，再 20 scenes，再 split。
3. 不直接全量盲跑。
```

### Phase 4：memory-v2

```text
1. object_memory_v2.py。
2. appearance_memory.py。
3. 先 128f，再 full stride-10。
```

### Phase 5：D4RT Sim3 export

```text
1. sim3.py。
2. export_scannet_d4rt_sim3.py。
3. 输出 residual diagnostics。
```

### Phase 6：Replica-Dynamic

```text
1. check_dynamic_replica_env.py。
2. dynamic_replica_stream.py。
3. run_dynamic_replica.py。
4. evaluate_dynamic_tracks.py。
5. evaluate_dynamic_objects.py。
```

---

## 9. 失败时的决策树

### 9.1 pre_points policy 失败

```text
如果 recompute 超 baseline 但 inherit 不超：
  不否定 evaluator；把 recompute 作为 Stream3D-style observed-support result。
  方法优化优先转向 densification 和 memory。

如果 inherit 也超：
  当前 ScanNet 结论更强，可作为主结果候选。

如果二者都不超：
  不继续围绕 adaptive top-k 调参，进入 method-v3。
```

### 9.2 tune/final 失败

```text
如果 tune 超 final 不超：
  认为 top-k 超参过拟合。
  减少超参空间，固定 mask_count + ratio=0.14 或改 quality threshold。
```

### 9.3 densification 失败

```text
如果 coverage 上升但 AP 降：
  加 boundary erosion、connected component、conflict suppression。

如果小物体丢：
  对小 mask 用弱 erosion，降低 min pixel。

如果速度慢：
  mask stride sampling，先调小场景。
```

### 9.4 memory-v2 失败

```text
如果 object explosion：
  提高 match threshold，加入 pre-merge，减少 full sequence。

如果 over-merge：
  加 same-frame exclusivity 和 appearance gating。

如果 reactivation 失败：
  降低 lost matching threshold，提高 appearance 权重。
```

### 9.5 D4RT Sim3 失败

```text
如果 residual 大：
  检查坐标轴、frame indexing、t_cam、scale、window chunk。

如果 dynamic objects 影响 anchors：
  用 static/high-confidence/background-like anchors。

如果仍失败：
  rgbd_eval 作为 ScanNet bridge，D4RT Sim3 只作为 diagnostic。
```

### 9.6 Replica-Dynamic GT 不完整

```text
如果没有 semantic GT：
  不报 semantic AP。

如果没有 object mask GT：
  报 point tracking + pseudo consistency + qualitative。

如果只有 trajectories：
  先做 D4RT sanity 和 carrier grouping consistency。
```

---

## 10. 最终 claim 安全边界

### 10.1 可以写

```text
我们确认当前 ScanNet evaluator 与原 Stream3D 的 cropped-TMP evaluation protocol 对齐；在该协议下，当前 Stream4D adaptive mask-count 后处理在 32CLIP + Cropformer 设置上达到 20.37 / 35.52 / 55.06，超过 Stream3D-Cropformer baseline 20.11 / 34.47 / 50.23。
```

前提：

```text
明确写 Stream3D-style evaluator。
明确写 32CLIP + Cropformer。
明确写 rgbd_eval export bridge。
明确写 validation-tuned 或 locked split result。
```

### 10.2 暂时不能写

```text
Stream4D 已经全面优于 Stream3D。
Stream4D 已经完成 feed-forward semantic 4D reconstruction and tracking。
D4RT geometry 已经完全替代 ScanNet RGB-D/pose。
ScanNet 证明了动态 tracking。
```

### 10.3 最理想的下一版主线

如果后续实验成立，论文主线应写成：

```text
Stream3D solves streaming 3D mask merging on RGB-D reconstructed point clouds. Stream4D extends this setting by using D4RT carriers as physical 4D semantic support, achieving comparable or better static ScanNet performance under the original Stream3D protocol, and substantially stronger dynamic object tracking on Replica-Dynamic.
```

这条主线比单纯围绕 `+0.26 AP` 更稳，也更符合 D4RT + Stream3D 的真正创新点。
