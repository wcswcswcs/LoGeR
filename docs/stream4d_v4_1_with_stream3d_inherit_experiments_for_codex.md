# Stream4D v4：代码审核、指标审计、方法改进与 ScanNet / Replica-Dynamic 实验计划书

面向 Codex 的执行文件。本文档基于重新解压并审核以下代码包后写成：

```text
/mnt/data/stream4d_v3_code_review_packet.zip
/mnt/data/Code_Stream3D.zip
/mnt/data/Open-d4rt-main.zip
```

审计解压路径：

```text
/mnt/data/audit_v3/Stream3D
/mnt/data/orig_stream3d/Code_Stream3D
/mnt/data/opend4rt/Open-d4rt-main
```

本文档使用 Typora 友好的公式格式，只使用 `$...$` 或 `$$...$$`，不使用 display 公式的方括号语法。

---

## 1. 本轮代码审核结论

### 1.1 我实际审核了什么

本轮审计对象包括：

```text
Stream3D/stream4d/*.py
Stream3D/evaluation/evaluate.py
Stream3D/tools/*.py
Stream3D/tests/test_stream4d_protocol_fixes.py
Stream3D/data/evaluation/scannet/*.txt
Stream3D/outputs/audit/eval_protocol_audit_s1.{md,json}
Stream3D/outputs/stream4d_rescore_v3/*_summary.json
docs/stream4d_v3_codex_plan_scannet_implementation.md
docs/stream4d_v3_实验结果复盘.md
docs/stream4d_v3_protocol_corrected_plan_for_codex.md
```

我还对照了原始 Stream3D 的 `evaluation/evaluate.py`，并检查了 OpenD4RT 的 `infer_track_3d.py`、`src/model/d4rt.py`、`src/model/query_embedding.py`、`src/losses/d4rt_loss.py`、`src/data/dynamic_replica_raw_dataset.py`。

语法和单元测试结果：

```bash
cd /mnt/data/audit_v3/Stream3D
python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 2 tests ... OK
```

注意：当前审阅包没有包含完整 `data/prediction/*`、`data/TMP/*`、`data/scannet/gt/*` 和原始 ScanNet 数据，因此我不能从 zip 包本身重新计算 312-scene AP；我能审核的是源码、结果文件、summary JSON、协议审计工具以及原版 evaluator 对照。Codex 后续必须在完整工程目录重新运行可复现实验。

### 1.2 和 v3 implementation 文件是否相符

结论：v3 implementation 中已经声称完成的工程项，当前代码基本相符。

已经相符的部分：

```text
1. rescore_scannet.py 已新增 --pre-points-policy recompute|inherit|fixed_path。
2. rescore summary 已记录 input_pre_points_count / output_pre_points_count / prediction_union_count / union subset 等审计字段。
3. local_4d_filter.py 已修复 set 顺序导致 carrier 权重错配的问题。
4. carrier_sampler.py 已使用 actual_count 修复采样数量不足时的字段长度错配。
5. tests/test_stream4d_protocol_fixes.py 已覆盖上述两个 bug。
6. audit_stream3d_eval_protocol.py 已能比较原版与当前 evaluator 的 AP 核心函数 hash。
7. make_scannet_stream4d_splits.py 和 materialize_scannet_eval_subset.py 已存在。
```

仍未完成的 v3 / v2 后续项：

```text
1. reliable densification 未实现。
2. ObjectMemory4D-v2 未实现。
3. D4RT-native d4rt_nn / Sim3 export 未实现。
4. Replica-Dynamic / Dynamic Replica 实验未实现。
5. 多窗口 full-sequence memory 没有在 full ScanNet 上稳定验证。
6. 当前 semantic 4D field 仍是 object_dict + representative masks，不是可查询的 4D semantic field。
```

### 1.3 是否存在虚假指标或 GT 泄漏

最重要结论：**没有发现代码层面的 GT 泄漏或直接伪造 AP。**

理由如下：

1. 当前 `evaluation/evaluate.py` 的 AP 核心函数 `evaluate_matches()` 和 `compute_averages()` 与原版 Stream3D by hash 一致。当前代码主要新增 `--tmp_root` 和 `--tmp_config`，用于配置 TMP 路径。
2. 原版 Stream3D evaluator 本身就读取 `TMP/{config}/{scene}_pre_points.npy`，并用该数组裁剪 GT 和 prediction。因此 cropped-TMP evaluator 不是 Stream4D 私自引入的假指标。
3. 当前 `rescore_scannet.py` 使用的是 prediction 和 `object_dict.npy` 中的无监督信号，例如 `area`、`mask_count`、`carrier_count`、`coverage_sum`，未读取 GT 文件，也没有使用 GT instance mask 生成预测。
4. 当前结果文件和 summary 文件互相一致：`recompute` 版本的 `output_pre_points_count` 等于 `prediction_union_count`，`inherit` 版本的 `prediction_union_subset_of_pre_points=True`。

但是，当前指标仍有三个必须写清楚的风险边界：

```text
风险 A：当前 best 只在 Stream3D-style recompute_pre_points evaluator 下略超 baseline。
风险 B：同一 Stream4D prediction 在 inherit_pre_points 诊断下大幅低于 Stream3D original baseline；但目前还缺 `Stream3D prediction + Stream4D inherited/fixed pre_points` 的 cross-support 对照，v4 必须补跑。
风险 C：best 超参来自 ScanNet val 多轮搜索，存在 validation overfitting 风险。
```

因此，当前结果可以作为“原版 Stream3D 评估协议下的可复现结果”，但不能写成“Stream4D 已全面优于 Stream3D”，也不能写成“已经证明 feed-forward semantic 4D reconstruction and tracking”。

### 1.4 当前 v3 结果如何安全表述

当前结果文件显示：

```text
Stream3D-Cropformer baseline:
  20.1139 / 34.4654 / 50.2268

Stream4D MVP:
  12.7594 / 23.6767 / 42.2114

Stream4D adaptive mask-count top-k, recompute_pre_points:
  20.3718 / 35.5222 / 55.0649

Stream4D adaptive mask-count top-k, inherit_pre_points:
  12.2851 / 23.3147 / 41.6773
```

固定配置在 deterministic final split 上：

```text
Stream3D-Cropformer final:
  19.4294 / 33.3989 / 49.6361

Stream4D adaptive recompute final:
  20.2401 / 35.6642 / 54.9907

Stream4D adaptive inherit final:
  11.9313 / 22.9523 / 41.1922
```

可以写：

```text
在原版 Stream3D-style cropped-TMP evaluator 的 recompute_pre_points 协议下，当前 Stream4D adaptive mask-count selection 在 ScanNet class-agnostic AP / AP50 / AP25 上略超本地 Stream3D-Cropformer baseline。
```

必须同时写：

```text
该优势在 inherit_pre_points 诊断下不成立，说明当前方法仍是 high-precision sparse-support proposal selection；要证明方法本体优越性，下一步必须提高 reliable support coverage 和跨窗口 object memory，而不是继续盲调 top-k。
```

不能写：

```text
Stream4D 已全面超过 Stream3D。
Stream4D 已完成 D4RT-native 4D reconstruction。
ScanNet 已证明动态 4D tracking。
inherit/unified support universe 下已经超过 Stream3D。
```


### 1.5 本轮必须补充：Stream3D 在 inherit / fixed-pre_points 下的对照

用户特别要求补充 `inherit_pre_points` 下 Stream3D 的表现。这里必须先澄清术语，否则后续表格会混乱。

对 Stream4D v3 adaptive top-k 而言，`inherit_pre_points` 的含义是：prediction 已经被 top-k 筛小，但 evaluator 的 `pre_points` 不重新等于筛后 prediction union，而是继承输入配置 `stream4d_scannet_32f_ioc075_fixmem` 的 `pre_points`。因此它会把同一 prediction 放回更大的 MVP support universe 中评价。

对原版 Stream3D-Cropformer baseline 而言，没有 top-k 输入配置，因此至少要报告两类对照：

```text
A. Stream3D self-inherit / original:
   prediction = Stream3D-Cropformer baseline `scannet`
   pre_points = Stream3D-Cropformer baseline `scannet`
   预期应与原始 baseline 完全一致：20.1139 / 34.4654 / 50.2268。

B. Stream3D cross-fixed-pre_points:
   prediction = Stream3D-Cropformer baseline `scannet`
   pre_points = 其他 config 的 pre_points，例如 Stream4D MVP 或 Stream4D adaptive recompute。
   这是新的诊断实验，目前没有已有日志结果，Codex 必须实际运行，不能填猜测数字。
```

命名规则必须严格：

```text
self-inherit: prediction 和 inherited pre_points 来自同一个方法/config。
cross-fixed-pre_points: prediction 和 pre_points 来自不同方法/config。
```

不要把 `Stream3D self-inherit` 和 `Stream3D on Stream4D-MVP-pre_points` 都简称为 `Stream3D inherit`。前者应等于原 baseline，是 sanity check；后者才是判断 `pre_points` universe 对不同方法影响的关键 cross-support 诊断。

Codex 必须新增以下 ScanNet baseline rows：

```text
S3D-0  Stream3D-Cropformer original/self-inherit
       pred_config = scannet
       tmp_config  = scannet

S3D-1  Stream3D-Cropformer on Stream4D-MVP support
       pred_config = scannet
       tmp_config  = stream4d_scannet_32f_ioc075_fixmem

S3D-2  Stream3D-Cropformer on Stream4D-v3 adaptive recompute support
       pred_config = scannet
       tmp_config  = stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute

S3D-3  Stream3D-Cropformer on Stream4D-v4 candidate supports
       pred_config = scannet
       tmp_config  = each locked Stream4D-v4 output config

S4D-0  Stream4D-v3 adaptive on Stream3D baseline support
       pred_config = stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute
       tmp_config  = scannet
```

这些实验的目标不是替代原版 Stream3D protocol，而是回答：

```text
1. Stream4D inherit 低，是不是因为 inherited universe 本身更难？
2. 如果把 Stream3D prediction 放到同一个 sparse Stream4D universe 下，Stream3D 是否仍然强？
3. 如果把 Stream4D prediction 放到 Stream3D 的大 universe 下，Stream4D 的 false negative 有多严重？
4. Stream4D v4 的 densification 是否真的缩小了 cross-support 差距？
```

必须先做 prediction shape audit。若 `pred_masks.shape[0] == num_scene_vertices`，可以直接切换 `--tmp_config` 做 cross-fixed evaluation。若 `pred_masks.shape[0] == len(original_pre_points)`，则必须先把 prediction expansion 到 full scene vertex universe：

```text
full_pred_masks = zeros(num_scene_vertices, K)
full_pred_masks[original_pre_points, :] = pred_masks
```

然后才能安全使用其他 config 的 `pre_points`。如果 shape audit 不通过，停止 cross-support AP 计算，不允许 silent indexing。

本节新增结果表格目前应留空，由 Codex 实际运行后填写：

| Row | prediction config | pre_points config | policy name | AP | AP50 | AP25 | pre_points % | union % | GT crop/full | valid? |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| S3D-0 | `scannet` | `scannet` | self-inherit/original | TODO, expected 20.1139 | TODO | TODO | TODO, expected 87.0159 | TODO | TODO | sanity |
| S3D-1 | `scannet` | `stream4d_scannet_32f_ioc075_fixmem` | cross-fixed | TODO | TODO | TODO | TODO | TODO | TODO | required |
| S3D-2 | `scannet` | `stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute` | cross-fixed | TODO | TODO | TODO | TODO | TODO | TODO | required |
| S4D-0 | `stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute` | `scannet` | cross-fixed | TODO | TODO | TODO | TODO | TODO | TODO | required |

解释标准：

```text
如果 S3D-1/S3D-2 也明显低于 Stream3D original，说明 Stream4D sparse universe 本身改变了 benchmark universe，不能把 cross-fixed 数字当主结论，只能当 diagnostic。
如果 S3D-1/S3D-2 仍高于 Stream4D inherit，说明在同一 sparse support 下 Stream3D prediction quality 仍更好，Stream4D 必须继续改 mask quality / densification。
如果 S4D-0 极低，说明 Stream4D 的主要瓶颈仍是 coverage/false negatives，不要只优化 recompute AP。
如果 v4 densification 让 S4D-on-S3D-support 明显提升，同时 recompute 不降，才说明方法从 sparse selector 走向更完整 reconstruction。
```

---

## 2. 代码级问题清单

### 2.1 已修复 bug，保留回归测试

#### 2.1.1 `local_4d_filter.py` carrier 权重错配

已修复版本：

```python
for cid, weight in zip(obs.carrier_ids.tolist(), obs.weights.tolist()):
    key = (int(obs.frame_id), int(cid))
    weights[key] = max(weights.get(key, 0.0), float(weight))
```

该修复正确。Codex 不要回滚。保留测试：

```text
Stream3D/tests/test_stream4d_protocol_fixes.py::test_local_filter_carrier_weights_follow_carrier_id_order
```

#### 2.1.2 `carrier_sampler.py` actual_count 长度错配

已修复版本使用：

```python
actual_count = int(keep.shape[0])
```

并让所有伴随字段使用 `actual_count`。该修复正确。保留测试：

```text
Stream3D/tests/test_stream4d_protocol_fixes.py::test_carrier_sampler_uses_actual_sample_count_for_all_fields
```

### 2.2 仍需修复或增强的代码问题

#### P0-1：`rescore_scannet.py` 缺少 object_dict 与 pred column 的强一致性校验

当前只检查：

```python
if len(object_items) != num_instances:
    raise RuntimeError(...)
```

这只能保证数量相同，不能保证排序后的 `object_dict` 第 $i$ 项真的对应 `pred_masks[:, i]`。当前导出逻辑大概率是对齐的，但后续如果经过 merge、filter、reindex 或外部脚本改动，很容易 silent mismatch。这个问题会直接污染 `mask_count`、`carrier_count` 等选择信号，是指标审计的 P0。

Codex 需要新增校验函数：

```text
verify_object_dict_prediction_alignment(pred_masks, object_items)
```

校验策略：

```text
1. 若 object entry 有 point_ids，则比较 point_ids 与 pred_masks[:, i] 的 true indices。
2. 记录 point_iou、point_ioc、area_object_dict、area_pred_column。
3. 默认要求 mean point_iou > 0.99。
4. 如果 object entry 没有 point_ids，至少记录 cannot_verify_alignment=true。
```

summary 必须新增：

```text
alignment_checked
alignment_mean_iou
alignment_min_iou
alignment_failed_instances
```

不满足条件时：

```text
先停止 rescore，不要输出 AP。
如果历史 object_dict 键没有重排，先生成 object_id_to_column.json 并在每次 export/rescore 时保存。
```

#### P0-2：`fixed_path` pre_points policy 缺少参数检查

当前 `--pre-points-policy fixed_path` 时，如果 `--fixed-pre-points-config` 为空，会读取：

```text
data/TMP/{scene}_pre_points.npy
```

这通常不是合法路径。Codex 需要在 parser 后添加显式校验：

```text
if pre_points_policy == fixed_path and not fixed_pre_points_config:
    raise ValueError("--fixed-pre-points-config is required for fixed_path")
```

#### P1-1：`reexport_scannet.py` 的 `point_dilate` 诊断字段误导

`reexport_scannet.py` 构造 `ScanNetExporter` 时固定：

```python
export_support_mode="mask_backproject"
```

即使实际 `reexport_mode == "point_dilate"`，最终 `_finalize_diag()` 仍会写：

```text
export_support_mode = mask_backproject
```

这会让后续审计误读。Codex 应修改为：

```text
point_dilate -> export_support_mode = reuse_point_ids 或 point_dilate
mask_backproject -> export_support_mode = mask_backproject
```

同时，`export_object_dict_points()` 中的 `export_nn_hit_rate` 不是 NN query 命中率，而是复用 point_ids 的数量比。必须改名或拆分：

```text
reuse_point_count
reuse_point_after_dilation_count
export_nn_hit_rate = NA for reuse_point_ids
```

不满足条件时：

```text
不要在论文表格中使用 point_dilate 的 export_nn_hit_rate。
```

#### P1-2：`export_d4rt_nn()` 仍是 NotImplemented

当前：

```python
raise NotImplementedError("d4rt_nn export requires a scene-coordinate calibration path; rgbd_eval is the MVP default.")
```

这意味着当前 ScanNet 导出仍是 `rgbd_eval` bridge：D4RT 主要提供 carrier UV / visibility / confidence，最终 mesh vertex 映射依赖 ScanNet depth / pose。Codex 不能把当前结果写成 D4RT 几何已替代 GT 3D。

下一步实现见本文第 8 节。Sim3 只允许用于 evaluation/export adapter，不允许进入 method grouping、memory、selection。

#### P1-3：当前 `ObjectMemory4D` 不是动态 tracking memory

当前 matching score：

$$
S_c(P,O)=\frac{|C_P \cap C_O|}{\min(|C_P|,|C_O|)}
$$

它只看 carrier set overlap，没有 appearance、motion、geometry、semantic、lost/reactivation 的真实判别能力。它可以连接 overlapping windows 中共享 carrier 的 proposals，但不能处理动态场景中的遮挡重现、交叉运动、无 overlap re-ID。

Codex 后续实现 `ObjectMemory4D-v2`，详见第 7 节。

#### P1-4：当前 full ScanNet 主要还是 one-window，不是真正 streaming memory 验证

历史结果使用：

```text
--frame-stride 10
--max-frames 32
--window-size 32
--window-stride 16
```

每个 scene 基本只有一个 window，因此：

```text
num_windows = 1
num_matched = 0
object_reactivation_count = 0
```

当前 ScanNet full-val 没有验证跨窗口 memory。Codex 需要在 v4 中优先跑 64/96/128 frames 的小规模多窗口实验，而不是直接全序列 312 scenes。

#### P1-5：当前 top-k 是后处理优势，不是 4D field 优势

`adaptive mask_count top-k` 的实际作用是从 185 个平均 candidates 中保留约 15 个，高 precision 但低 coverage。它可以作为强 baseline，但不能作为方法主贡献。v4 方法必须通过 reliable support expansion 和 memory-v2 证明：

```text
更多帧 + 更可靠 support + 更少碎片 = 更强 ScanNet 和动态 tracking
```

#### P2-1：D4RTAdapter 不支持长视频 chunking

当前 `D4RTAdapter.infer_carriers()` 如果 window 帧数超过 `clip_frames` 会直接报错。OpenD4RT 官方 `infer_track_3d.py` 对长视频有 anchor-clip/chunk 逻辑。v4 若做 Replica-Dynamic 长视频或 ScanNet full sequence，必须：

```text
1. 保持 window_size <= clip_frames；或
2. 把 OpenD4RT 官方 anchor-clip/chunk inference 包装进 adapter。
```

先不要让 Codex 直接把 window_size 提到 64/96/128，除非使用 48CLIP checkpoint 或实现 chunk adapter。

#### P2-2：`MaskEvidenceBuilder` 只在单点命中 mask 时生成 evidence，缺少边界和 cycle consistency

当前 evidence：

$$
\rho = \sigma(visibility) \cdot \sigma(confidence)
$$

然后只看 predicted UV 是否落在某个 mask ID 内。这会受 D4RT track error、mask 边界噪声、薄物体、遮挡影响。v4 应加入 self/cycle consistency 和 boundary-aware weight，详见第 6 节。

---

## 3. v4 的核心目标和研究假设

### 3.1 整体目标

v4 不是继续微调 `ratio=0.14`。v4 的目标是：

```text
把当前 sparse high-precision Stream4D proposal selector，升级为可验证的 feed-forward semantic 4D reconstruction and tracking 框架。
```

在 ScanNet 上，目标是至少稳健地超过 Stream3D-Cropformer，并证明优势不是只来自缩小 `pre_points`。在 Replica-Dynamic / Dynamic Replica 上，目标是证明 D4RT carrier memory 能处理动态 object identity，这才是 Stream3D 静态 point-cloud mask merging 做不到的优势。

### 3.2 核心假设

#### H1：当前 ScanNet 瓶颈是 reliable support coverage，而不是候选排序

当前 best 的 `recompute` AP 超 baseline，但 `inherit` 失败。说明 top-k 提升了 precision，却没有覆盖更大的 support universe。v4 必须让：

$$
\text{coverage} \uparrow, \quad \text{conflict} \not\uparrow\uparrow, \quad \text{fragmentation} \downarrow
$$

#### H2：carrier-guided reliable densification 能比 naive dense backprojection 更稳

naive dense mask backprojection 已经失败，因为它把整个 2D mask 都投到 3D，带来噪声和冲突。v4 只允许在 D4RT carrier seeds 附近、边界安全、跨帧一致的区域扩展 support。

#### H3：ObjectMemory4D-v2 能让更多帧真正带来收益

当前 full sequence 失败是 object explosion。v4 要证明更多窗口不是产生更多碎片，而是通过 motion/appearance/carrier/geometry matching 减少 fragmentation。

#### H4：动态场景上，D4RT carrier identity 应明显优于 Stream3D-style static overlap

Stream3D 的 historical update 依赖 3D overlap 和静态 manifold。动态物体移动后可以没有 overlap。D4RT carrier 的物理轨迹提供了动态 object tracking 的正交优势。

#### H5：D4RT 几何的尺度/坐标问题只在评估阶段用 Sim3 处理

D4RT-native geometry export 要实现，但 Sim3 只能用于 evaluation/export adapter：

$$
T^* = \arg\min_{s,R,t}\sum_i \omega_i \|sRx_i^{D4RT}+t-x_i^{eval}\|_2^2
$$

禁止把 $T^*$ 反馈到 carrier grouping、object memory、mask selection 或 semantic fusion。

---

## 4. Codex 执行总顺序

Codex 按以下顺序执行，不要跳步。

```text
Phase 0  复现与指标守卫：确认 v3 结果、协议、summary、无 GT 泄漏。
Phase 1  修仍存在的 P0/P1 bug：alignment check、fixed_path validation、diagnostic naming。
Phase 2  可靠 carrier evidence：self/cycle consistency、边界权重、mask evidence quality。
Phase 3  reliable densification：seeded component + boundary + conflict + voting。
Phase 4  ObjectMemory4D-v2：appearance/motion/geometry matching，先小规模多窗口。
Phase 5  ScanNet locked final：tune 只选超参，final 报告主结果。
Phase 6  D4RT-native Sim3 export：只做 evaluation/export adapter。
Phase 7  Replica-Dynamic / Dynamic Replica：先数据检查和 D4RT track sanity，再 object tracking。
```

每个 Phase 都必须输出：

```text
1. command log
2. config json
3. result csv/json
4. metrics table
5. failure cases list
6. visualizations
```

---

## 5. Phase 0：复现与指标守卫

### 5.1 目标

确认当前 v3 结果不是由 evaluator 改写、GT 泄漏、object_dict/pred mismatch 或无记录后处理造成。

### 5.2 实验与检查

运行：

```bash
cd Stream3D
python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
python -m unittest tests.test_stream4d_protocol_fixes
```

新增并运行：

```bash
python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /path/to/orig/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet.txt \
  --configs \
    scannet,stream4d_scannet_32f_ioc075_fixmem,stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute,stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit,scannet_self_inherit,scannet_on_stream4d_mvp_prepoints,scannet_on_stream4d_adaptive_prepoints,stream4d_adaptive_on_scannet_prepoints \
  --output outputs/audit/stream4d_v4_metric_integrity.md
```

### 5.3 必须记录的指标

```text
evaluator_ap_core_equal_by_hash
has_pre_points_load_original
has_pre_points_load_current
pre_points_policy
pre_points_equals_prediction_union
prediction_union_subset_of_pre_points
mean_pre_points_ratio
mean_prediction_union_ratio
mean_gt_instances_crop
mean_gt_instances_full
num_pred_instances
object_dict_pred_alignment_mean_iou
object_dict_pred_alignment_min_iou
gt_files_read_by_rescore = False
```

### 5.4 判断标准

Phase 0 通过条件：

```text
1. evaluate_matches / compute_averages 与原版 hash 相同。
2. rescore 不读取 gt_path / data/scannet/gt。
3. object_dict 与 pred columns 强一致。
4. 每个结果 config 的 pre_points policy 清楚可追踪。
5. 所有 result 文件和 summary 文件中的 config 名完全一致。
```

如果不满足：

```text
先停止所有新实验，修复协议和 alignment，不要继续报告 AP。
```

### 5.5 可视化

输出：

```text
outputs/audit/pre_points_ratio_by_config.png
outputs/audit/union_ratio_by_config.png
outputs/audit/gt_crop_full_by_config.png
outputs/audit/object_dict_alignment_iou_hist.png
```

### 5.6 新增：Stream3D inherit / cross-fixed baseline 守卫

Phase 0 必须额外运行 Stream3D baseline 在不同 `pre_points` universe 下的诊断。该实验优先级等同于 metric integrity，因为如果不报告这个对照，`Stream4D inherit` 低分很难被正确解释。

新增工具：

```bash
python -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet.txt \
  --pred-config scannet \
  --pred-suffix _class_agnostic \
  --pre-points-config scannet \
  --output-config scannet_self_inherit \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/scannet_self_inherit_class_agnostic.txt

python -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet.txt \
  --pred-config scannet \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config scannet_on_stream4d_mvp_prepoints \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/scannet_on_stream4d_mvp_prepoints_class_agnostic.txt

python -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet.txt \
  --pred-config scannet \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config scannet_on_stream4d_adaptive_prepoints \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/scannet_on_stream4d_adaptive_prepoints_class_agnostic.txt

python -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet.txt \
  --pred-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --pred-suffix _class_agnostic \
  --pre-points-config scannet \
  --output-config stream4d_adaptive_on_scannet_prepoints \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_adaptive_on_scannet_prepoints_class_agnostic.txt
```

`tools.evaluate_cross_prepoints` 的实现要求：

```text
1. 只改变 evaluator 使用的 pre_points，不改变 prediction masks 本身。
2. 先做 shape audit：prediction mask length、original pre_points length、target pre_points max index、scene vertex count。
3. 若 prediction mask 不是 full scene vertex length，先生成 expanded full-scene prediction，再裁剪 target pre_points。
4. 输出 cross-prepoints summary：每个 scene 的 source_pred_config、source_tmp_config、target_pre_points_config、mask_shape_mode、expanded_or_not、pre_points_ratio、prediction_union_ratio、gt_crop/full。
5. 不读取 GT 生成 prediction；GT 只在 evaluator 中读取。
```

必须记录：

```text
AP / AP50 / AP25
mask_shape_mode = full_scene | cropped_original | invalid
expanded_prediction = true/false
source_pre_points_ratio
target_pre_points_ratio
prediction_union_ratio_under_target_pre_points
num_gt_instances_in_target_pre_points / full
num_pred_instances
cross_support_delta_to_original_stream3d
cross_support_delta_to_stream4d_inherit
```

判断标准：

```text
1. `scannet_self_inherit` 必须和原 `scannet` baseline 差异小于 0.01 AP，否则 evaluator/materialization 有 bug。
2. 所有 cross-fixed rows 必须在表格中单独标注为 diagnostic，不允许混入主 Stream3D protocol 主表数值。
3. 若 `scannet_on_stream4d_mvp_prepoints` 高于 `stream4d_adaptive_inherit` 很多，说明 Stream4D 在同一 sparse support 下 mask/object quality 仍不足。
4. 若 `scannet_on_stream4d_mvp_prepoints` 也明显下降，说明 sparse support universe 本身改变了任务难度，最终论文要把 inherit/fixed 作为 diagnostic 而不是主 benchmark。
```

失败时 Codex 先尝试：

```text
如果 mask shape 不匹配，先写 prediction expansion，不要硬裁剪。
如果 target pre_points index 超出 prediction full length，检查是否用了不同 mesh / processed scene version。
如果 self-inherit 不等于 baseline，检查 symlink/materialize 是否丢 scene、tmp_root 是否错、class_agnostic suffix 是否错。
如果 cross-fixed AP 出现异常高值，检查 GT crop/full 数量是否被错误缩小到几乎无 GT。
```

可视化：

```text
outputs/audit/cross_prepoints_support_matrix.png
outputs/audit/stream3d_vs_stream4d_under_same_prepoints.csv
outputs/audit/cross_prepoints_delta_hist.png
outputs/audit/gt_crop_full_by_cross_prepoints.png
```


---

## 6. Phase 2：可靠 carrier evidence

Phase 1 是 bug fix，不单独作为实验主贡献；Phase 2 开始改方法。

### 6.1 目标

让每个 carrier 的语义证据不只依赖 `visibility * confidence` 和单点落 mask，而是加入 D4RT 自一致性、cycle consistency、mask boundary safety。

### 6.2 方法思路

对 source carrier $c=(u,v,t_s)$，当前 evidence 权重是：

$$
\rho_{c,t}=\sigma(v_{c,t})\sigma(q_{c,t})
$$

v4 改成：

$$
\rho'_{c,t}=\sigma(v_{c,t})\sigma(q_{c,t})\exp(-\lambda_s e^{self}_{c})\exp(-\lambda_{cyc}e^{cycle}_{c,t})B(m,u_{c,t})
$$

其中：

```text
e_self: 查询 t_src=t_tgt=t_cam 时，预测 UV 与 source UV 的误差。
e_cycle: source -> target -> source 的 round-trip UV 误差。
B(m,u): mask boundary safety，离边界越近权重越低。
```

第一版不需要复杂光流。Codex 先实现三个诊断：

```text
self_uv_error
cycle_uv_error
boundary_distance_norm
```

然后在 `MaskEvidenceBuilder` 中支持：

```text
--evidence-mode rho
--evidence-mode rho_self
--evidence-mode rho_self_boundary
--evidence-mode rho_self_cycle_boundary
```

### 6.3 实验设计

在 `scannet_tune.txt` 的 30 个场景先跑，不全量：

```text
E0 current rho
E1 rho + self_uv
E2 rho + self_uv + boundary
E3 rho + self_uv + cycle + boundary
```

保持后处理固定，不重新调 adaptive ratio。然后在 final split 跑 locked best。

### 6.4 必须记录的指标

```text
AP / AP50 / AP25, recompute_pre_points
AP / AP50 / AP25, inherit_pre_points
num_mask_observations_with_carriers
carrier_visibility_rate
carrier_assignment_rate
mean_self_uv_error
p90_self_uv_error
mean_cycle_uv_error
p90_cycle_uv_error
mean_boundary_distance_norm
num_local_proposals
carrier_coverage
local_selected_mask_ratio
```

### 6.5 判断标准

Phase 2 成立条件：

```text
1. recompute AP 不低于 current v3 超过 0.3 AP 以上，或 AP50/AP25 有明显提升。
2. inherit AP 至少提升 +1.0。
3. carrier_assignment_rate 不大幅下降；若下降，AP 必须上升，说明过滤有效。
4. 可视化中低质量 D4RT drift carriers 被过滤。
```

如果失败：

```text
若 evidence 太保守导致 coverage 掉，降低 self/cycle penalty。
若 evidence 无效，检查 D4RTAdapter 与官方 infer_track_3d 输出是否一致。
若 cycle 成本太高，先只用 self_uv + boundary。
```

### 6.6 可视化

每个场景保存：

```text
RGB + carrier UV overlay, color by rho
RGB + carrier UV overlay, color by self error
RGB + carrier UV overlay, color by boundary safety
mask observation before/after evidence filtering
```

---

## 7. Phase 3：carrier-guided reliable densification

### 7.1 目标

解决当前方法最硬的结构性瓶颈：support 太稀疏。目标不是把所有 2D mask dense backproject，而是在 carrier seeds 附近可靠扩展。

### 7.2 方法定义

对 object $O_k$，它有若干 mask observations：

$$
\mathcal{E}_k=\{(t,m,\alpha_{t,m})\}
$$

对 mask $m$ 中的 pixel $p$，定义可靠扩展分数：

$$
R_k(p,t)=w_s\exp\left(-\frac{d(p,S_k^t)^2}{2\sigma^2}\right)+w_bB_m(p)+w_tT_k(t)+w_qQ_k(t)-w_xX_k(p,t)
$$

含义：

```text
S_k^t: object k 在 frame t 的 carrier seed pixels。
d(p,S_k^t): pixel 到最近 seed 的 2D 距离或 distance transform。
B_m(p): boundary safety，mask 内离边界越远越高。
T_k(t): object k 的时间支持强度，例如该 mask 是否来自多帧支持。
Q_k(t): object 的 carrier evidence quality。
X_k(p,t): conflict penalty，该 pixel 是否被其他 object 高置信支持。
```

保守实现顺序：

```text
D1 seeded connected component：只保留包含 carrier seeds 的 connected component。
D2 boundary erosion：去掉 mask 边界附近不稳定像素。
D3 seed distance cap：只保留离 seeds 不超过 r 像素的区域。
D4 winner-take-all conflict：同一 3D point 被多个 object 命中时，只留最高 R。
D5 multi-frame vote：优先扩展被多帧、多 mask_count 支持的 object。
```

### 7.3 ScanNet 实验设置

在 30 tune scenes 上：

```text
A0 sparse carrier_uv export 当前 MVP
A1 naive mask_backproject 失败对照
A2 seeded component only
A3 seeded component + boundary erosion
A4 A3 + seed distance cap
A5 A4 + WTA conflict suppression
A6 A5 + multi-frame vote / top-M observations
```

每个策略同时跑：

```text
recompute_pre_points
inherit_pre_points
```

### 7.4 必须记录的指标

主指标：

```text
AP / AP50 / AP25, recompute
AP / AP50 / AP25, inherit
```

覆盖和冲突：

```text
mean_pre_points_ratio
mean_prediction_union_ratio
mean_gt_instances_in_pre_points / full
prediction_union_count
export_conflict_rate
points_per_object_mean / median
mask_area_mean / median
tiny_mask_ratio < 100 vertices
large_mask_ratio > 1000 vertices
```

2D densification 诊断：

```text
num_seed_carriers_per_mask
mask_pixels_total
mask_pixels_kept
kept_ratio
boundary_removed_ratio
seed_distance_removed_ratio
conflict_removed_ratio
multi_frame_vote_count
```

### 7.5 判断标准

最低成功标准：

```text
1. inherit AP 比 v3 adaptive inherit 提升至少 +3.0。
2. recompute AP 不低于 v3 adaptive recompute 超过 0.5 AP。
3. prediction_union_ratio 至少从 5.75% 提升到 10% 以上。
4. export_conflict_rate 不超过 0.25，或不超过 sparse baseline 的 1.5x。
5. object count 不暴涨超过 2x。
```

强成功标准：

```text
1. recompute AP / AP50 / AP25 全部超过 Stream3D-Cropformer baseline，且 AP 至少 +0.8。
2. inherit AP 接近或超过 Stream3D-Cropformer baseline；若未超过，至少达到 16+ AP 并明显高于 v3 inherit。
3. AP50 提升不是靠 AP25 粗糙扩张，AP 和 AP50 必须同步提升。
```

### 7.6 不满足条件时 Codex 的尝试方向

如果 coverage 上升但 AP 下降：

```text
先增加 boundary erosion，不要加大 densify 半径。
限制每个 object 只用 top-M high-quality observations。
对小物体关闭强 erosion，避免被完全抹掉。
```

如果 conflict 上升：

```text
启用 WTA conflict suppression。
加入 same-frame object exclusivity。
提高 R_k 的 object temporal support 权重。
```

如果小物体漏掉：

```text
按 2D mask area 分 bucket，小物体使用更小 erosion 和更小 min support。
单独报告 small/medium/large object coverage。
```

如果 AP25 上升但 AP/AP50 下降：

```text
说明扩张太粗，不能作为主方法。
回退到 A3/A4，优先保边界。
```

### 7.7 可视化

每个策略保存至少 20 个场景：

```text
RGB frame + original 2D mask
carrier seeds overlay
kept densification pixels overlay
removed boundary/conflict pixels overlay
exported 3D mesh points overlay
per-object best GT IoU histogram
missed GT object panel
false positive / over-expanded object panel
```

---

## 8. Phase 4：ObjectMemory4D-v2

### 8.1 目标

让更多帧带来更完整的 support，而不是产生更多碎片。当前 full sequence scene0050 失败并产生 1015 objects，说明 naive 多窗口不能直接跑。

### 8.2 方法定义

proposal $P_j$ 与历史 object $O_i$ 的 matching score：

$$
S(O_i,P_j)=w_cS_c+w_pS_p+w_aS_a+w_mS_m+w_gS_g+w_lS_l-w_xS_x
$$

定义：

```text
S_c: carrier overlap / IoC。
S_p: exported point overlap / IoC，仅 evaluation bridge 可用时使用，不作为 D4RT-only 必需项。
S_a: appearance similarity，CLIP/DINO/RGB histogram。
S_m: D4RT trajectory / velocity continuity。
S_g: object centroid proximity。
S_l: language / semantic label consistency，可选。
S_x: same-frame conflict / over-merge penalty。
```

第一版实现优先级：

```text
1. S_c + S_a + S_g for ScanNet static。
2. S_c + S_a + S_m for Replica-Dynamic。
3. Hungarian one-to-one matching，避免多个 proposal 贪心合并到同一 object。
4. lost/reactivated lifecycle。
```

### 8.3 必须实现的代码

新增：

```text
stream4d/object_memory_v2.py
stream4d/appearance_memory.py
stream4d/motion_memory.py
stream4d/memory_diagnostics.py
```

`object_memory_v2.py` 输出：

```text
object_id
state = active/lost/reactivated/merged/split_candidate
birth_window
last_seen_window
carrier_ids
frame_support
appearance_feature
centroid_history
velocity_history
match_history
```

### 8.4 ScanNet 多窗口实验

先跑少量场景，不要直接 312 scenes。

配置：

```text
M0 32f one-window current memory
M1 96f old memory
M2 96f memory-v2
M3 128f old memory
M4 128f memory-v2
M5 full stride-10 old memory, only 3 scenes
M6 full stride-10 memory-v2, only 3 scenes
```

建议场景：

```text
scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00, object explosion case
```

### 8.5 必须记录的指标

```text
AP / AP50 / AP25, recompute
AP / AP50 / AP25, inherit
num_windows
num_created
num_matched
num_reactivated
num_lost
final_num_objects
object_growth_rate_per_window
fragmentation_per_gt
merge_error_per_pred
carrier_reuse_rate
appearance_match_score_mean
motion_match_score_mean
same_frame_conflict_rate
runtime_per_window
peak_gpu_memory
```

fragmentation：

$$
Frag(g)=|\{p: IoU(p,g)>0.1\}|-1
$$

merge error：

$$
Merge(p)=|\{g: IoU(p,g)>0.1\}|-1
$$

### 8.6 判断标准

Phase 4 成立：

```text
1. 96f / 128f memory-v2 的 AP50 不低于 32f current，并最好提升。
2. final_num_objects 不超过 one-window 的 2x。
3. fragmentation_per_gt 比 old memory 降低至少 30%。
4. num_matched > 0，且 timeline 可视化显示真实跨窗口合并。
5. 不出现 full sequence object explosion。
```

不满足条件时：

```text
object explosion：提高 matching threshold，增加 pre-merge，增强 same-frame conflict penalty。
over-merge：降低 S_c 权重，增加 S_x，强制 one-to-one Hungarian。
re-ID 失败：提高 appearance 权重，降低 lost object threshold，加入 centroid/motion gate。
appearance 太慢：先用 RGB histogram，记录 feature_type=rgb_histogram，不冒充 CLIP/DINO。
```

### 8.7 可视化

```text
object timeline: window_id vs object_id vs state
match matrix heatmap: proposal x historical object
appearance/motion/carrier score histogram
same object across windows 2D overlay video
fragmentation case panels
merge error case panels
```

---

## 9. Phase 5：ScanNet 最终实验计划

### 9.1 ScanNet 实验总目标

ScanNet 只验证静态室内 3D instance/semantic reconstruction，不证明动态 tracking。ScanNet 的 v4 目标是回答：

```text
1. Stream4D 是否在原版 Stream3D evaluator 下稳定超过 Stream3D-Cropformer。
2. 该优势是否不再只依赖 top-k 后 recompute_pre_points。
3. reliable densification 是否提升 coverage 且不引入大量冲突。
4. memory-v2 是否让更多帧提升而不是导致碎片爆炸。
5. D4RT-native Sim3 export 与 rgbd_eval bridge 的差距在哪里。
```

### 9.2 主评估协议

主协议仍然是：

```text
Stream3D-style cropped-TMP evaluator
```

但每个方法必须同时报告：

```text
recompute_pre_points: 与原版方法特定 observed support 对齐。
inherit_pre_points: 诊断 top-k 是否靠缩小 support 获利。
fixed_pre_points: 统一或交叉 support 诊断。
```

额外要求：Stream3D-Cropformer baseline 也必须报告 self-inherit 和 cross-fixed-pre_points 诊断。`Stream3D self-inherit` 应等于原 baseline；`Stream3D on Stream4D-MVP/adaptive pre_points` 用来判断 sparse pre_points universe 对 baseline 的影响；`Stream4D on Stream3D pre_points` 用来判断 Stream4D coverage/false-negative 缺口。

不要把 fullmesh evaluator 作为主协议替代 Stream3D benchmark；如果实现 fullmesh，只作为 diagnostic。

### 9.3 Tune/final split 规则

必须使用固定 split：

```text
splits/scannet_tune.txt
splits/scannet_final.txt
seed = 20260607
```

规则：

```text
1. 所有阈值、top-M、erosion radius、distance cap、memory weights 只在 tune split 选择。
2. final split 只运行 locked config。
3. 最终主表只报告 final split，同时可补充 full 312 fixed-config diagnostic。
4. 如果想报告 full 312，需要说明超参已在 tune split 锁定，full 312 仅为附加复现。
```

### 9.4 ScanNet S0：v3 复现和协议守卫

假设：

```text
H_S0: 当前 v3 结果可复现，且没有 evaluator fake / GT leakage，但优势主要集中在 recompute_pre_points。
```

实验：

```text
Stream3D-Cropformer baseline original/self-inherit
Stream3D-Cropformer on Stream4D-MVP pre_points, cross-fixed diagnostic
Stream3D-Cropformer on Stream4D-v3 adaptive recompute pre_points, cross-fixed diagnostic
Stream4D MVP
Stream4D v3 adaptive recompute
Stream4D v3 adaptive inherit
Stream4D v3 adaptive on Stream3D baseline pre_points, cross-fixed diagnostic
```

必须补充的命令由第 5.6 节给出。当前已有日志只包含 `Stream3D-Cropformer baseline original/recompute-like`，还没有 `Stream3D on Stream4D pre_points` 的数值；不要在报告里猜这个数字。

指标：

```text
AP/AP50/AP25
mean_pre_points_ratio
mean_union_ratio
mean_gt_crop/full
num_pred_instances
alignment_mean_iou
cross_prepoints_mask_shape_mode
source_pre_points_config
target_pre_points_config
expanded_prediction
cross_support_delta_to_original_stream3d
cross_support_delta_to_stream4d_inherit
```

判断标准：

```text
结果与 v3 文件差异 < 0.05 AP。
metric integrity 全部通过。
`scannet_self_inherit` 与 `scannet` baseline 差异 < 0.01 AP。
所有 cross-fixed rows 有 shape audit 和 target pre_points summary。
```

失败时：

```text
先检查 prediction/TMP/config 路径是否对应。
不要改参数追结果。
```

### 9.5 ScanNet S1：evidence quality ablation

假设：

```text
H_S1: self/cycle/boundary evidence 能减少错误 carrier evidence，提升 selection 稳定性。
```

方法：

```text
rho
rho_self
rho_self_boundary
rho_self_cycle_boundary
```

指标：

```text
AP/AP50/AP25 recompute/inherit
self_uv_error mean/p90
cycle_uv_error mean/p90
carrier_assignment_rate
num_local_proposals
object_count
```

成立标准：

```text
inherit AP +1.0 以上，recompute AP 不降超过 0.3。
```

### 9.6 ScanNet S2：reliable densification ablation

假设：

```text
H_S2: seeded densification 可以提升 support coverage，同时保持 precision。
```

方法：

```text
sparse carrier_uv
naive mask_backproject, 失败对照
seeded component
seeded component + boundary
seeded component + boundary + distance cap
seeded component + boundary + distance cap + WTA conflict
+ multi-frame vote
```

主指标：

```text
AP/AP50/AP25 recompute
AP/AP50/AP25 inherit
```

诊断指标：

```text
union_ratio
pre_points_ratio
gt_crop/full
conflict_rate
points_per_object
small/medium/large object recall proxy
```

成立标准：

```text
inherit AP 至少 +3.0。
recompute AP 至少维持 v3 best，最好 +0.5。
union_ratio 至少 10%。
conflict_rate < 0.25。
```

### 9.7 ScanNet S3：multi-window memory-v2

假设：

```text
H_S3: memory-v2 让 96/128 frames 比 32 frames 更好，而不是 object explosion。
```

方法：

```text
32f current
96f old memory
96f memory-v2
128f old memory
128f memory-v2
```

指标：

```text
AP/AP50/AP25 recompute/inherit
num_windows
num_created
num_matched
final_num_objects
fragmentation_per_gt
merge_error_per_pred
object_growth_rate
```

成立标准：

```text
128f memory-v2 的 AP50 超 32f current。
final_num_objects 不超过 32f current 的 2x。
fragmentation 比 old memory 降低 30%。
```

### 9.8 ScanNet S4：最终 locked comparison

最终表格必须包含：

```text
Method | 2D masks | D4RT | export | eval policy | AP | AP50 | AP25 | pre_points % | union % | GT crop/full | #pred | time/frame
```

方法行：

```text
Stream3D-Cropformer baseline
Stream4D-v3 adaptive top-k
Stream4D-v4 evidence only
Stream4D-v4 densification
Stream4D-v4 memory-v2
Stream4D-v4 densification + memory-v2
```

主 claim 成立标准：

```text
1. final split recompute AP 超 Stream3D-Cropformer 至少 +0.5，AP50 至少 +1.0。
2. inherit AP 相比 v3 inherit 至少 +3.0。
3. pre_points / union 诊断显示 coverage 改善，而不是进一步缩小 universe。
4. 超参 locked，不能 final split 再调。
```

如果只比 baseline 高 +0.26 AP：

```text
写成 slight improvement，不作为主要卖点。
把论文主线转到动态 4D tracking。
```

---

## 10. Phase 6：D4RT-native Sim3 export

### 10.1 目标

实现 `d4rt_nn`，用于评估 D4RT-native geometry 与 rgbd_eval bridge 的差距。注意，这不是第一优先级，不应阻塞 ScanNet v4 方法优化。

### 10.2 方法边界

D4RT 预测点：

$$
x_i^{D4RT}
$$

ScanNet evaluation coordinate anchor：

$$
x_i^{ScanNet}
$$

评估时估计：

$$
T^*=(s,R,t)=\arg\min_{s,R,t}\sum_i \omega_i\|sRx_i^{D4RT}+t-x_i^{ScanNet}\|_2^2
$$

严格禁止：

```text
1. 用 Sim3 后的 ScanNet 坐标参与 object grouping。
2. 用 GT instance / semantic label 选择 anchors。
3. 用 Sim3 residual 调 selection 阈值。
```

允许：

```text
1. evaluation/export adapter 使用 depth/pose anchor 做坐标对齐。
2. 报告 D4RT geometry diagnostic。
3. 与 rgbd_eval bridge 进行误差分解。
```

### 10.3 必须记录的指标

```text
AP/AP50/AP25, d4rt_sim3_eval
AP/AP50/AP25, rgbd_eval_bridge
sim3_scale
sim3_rotation_det
sim3_translation_norm
anchor_count
inlier_ratio
median_residual
p90_residual
nn_hit_rate
points_outside_mesh_radius
D4RT_depth_absrel_after_scale
```

### 10.4 判断标准

```text
d4rt_sim3_eval 与 rgbd_eval_bridge AP50 差距 < 20% relative：可作为主路径候选。
median_residual < 0.10m：几何对齐可接受。
否则只能作为 diagnostic，不作为主 claim。
```

失败时：

```text
检查坐标轴、frame indexing、t_cam reference、scale、chunk alignment。
尝试 scene-level Sim3、window-level Sim3、static-anchor Sim3。
动态物体影响 anchor 时，用 low-motion / high-confidence anchors。
```

### 10.5 可视化

```text
D4RT point cloud after Sim3 vs ScanNet mesh overlay
residual heatmap
anchor inlier/outlier visualization
window-to-window scale drift plot
```

---

## 11. Phase 7：Replica-Dynamic / Dynamic Replica 动态实验

### 11.1 动态实验总目标

动态实验用于证明 Stream4D 的真正优势：

```text
D4RT carrier 将语义绑定到 physical surface tracks，因此比 Stream3D-style static 3D overlap 更适合动态 object reconstruction and tracking。
```

ScanNet 是静态 benchmark；动态优势必须在 Replica-Dynamic / Dynamic Replica 上验证。

### 11.2 数据检查

OpenD4RT 代码中的 raw loader 名称是 `DynamicReplicaRawDataset`，默认结构类似：

```text
data/dynamic-replica/v2/{train,valid,test}/{scene}/images/*.png
data/dynamic-replica/v2/{train,valid,test}/{scene}/depths/*.geometric.png
data/dynamic-replica/v2/{train,valid,test}/{scene}/trajectories/*.pth
frame_annotations_valid.json
```

Codex 先实现：

```text
tools/check_dynamic_replica_env.py
```

检查：

```text
data_root exists
split dirs exist
frame_annotations_valid.json exists
images count
depths count
trajectories count
camera fields R/T/focal_length/principal_point
whether GT semantic labels exist
whether GT instance masks exist
whether object IDs exist
```

数据条件与允许报告：

```text
有 GT instance masks / object IDs：可报告 IDF1 / IDSW / 4D IoU。
有 trajectories：可报告 D4RT point tracking APD3D / AJ / OA / EPE。
只有 RGB/depth/camera：只做 qualitative + consistency，不报 benchmark tracking AP。
有 SAM2/Cropformer masks 但无 GT semantics：只报 pseudo temporal consistency，不报 semantic AP。
```

### 11.3 D0：D4RT track sanity

假设：

```text
H_D0: D4RTAdapter 在 Dynamic Replica 上的输出与 OpenD4RT 官方 inference 对齐，可作为 carrier backbone。
```

实验：

```text
从 valid split 取 10 个 clips，每个 48 frames。
采样 background points、moving points、mask boundary points。
对比 D4RTAdapter 与官方 infer_track_3d.py 的输出。
```

指标：

```text
adapter_vs_official_uv_diff
adapter_vs_official_xyz_diff
visibility_logit_diff
confidence_logit_diff
uv_in01_rate
visibility_rate
confidence_mean
APD3D / AJ / OA / EPE, if GT trajectories exist
```

判断标准：

```text
adapter_vs_official diff 在数值容差内。
若不一致，先修 adapter，不进入 object tracking。
```

失败时：

```text
检查 t_src/t_tgt/t_cam 排列。
检查 query reshape order。
检查 resize 和 aspect_ratio。
检查 confidence 是否 sigmoid。
检查 RGB channel order。
```

### 11.4 D1：动态 object identity tracking

假设：

```text
H_D1: ObjectMemory4D-v2 比 carrier-overlap memory 和 Stream3D-style overlap memory 有更少 ID switch 和 fragmentation。
```

方法对比：

```text
B0 per-frame 2D masks, no tracking
B1 Stream3D-style static 3D overlap memory
B2 Stream4D-v3 carrier-overlap memory
B3 Stream4D-v4 memory-v2
B4 memory-v2 without motion term
B5 memory-v2 without appearance term
```

如果没有 GT instance masks，用 pseudo protocol：

```text
SAM2 video masks 或手动选定 moving object 作为 pseudo GT。
表格标题必须写 pseudo，不得写 benchmark SOTA。
```

指标：

```text
IDF1
ID precision / ID recall
ID switches
fragmentation count
track purity
track coverage
object reactivation success
average track length
lost duration before reactivation
per-frame mask IoU, if GT exists
4D IoU, if GT exists
```

4D IoU：

$$
IoU_{4D}(P,G)=\frac{\sum_t |P^t \cap G^t|}{\sum_t |P^t \cup G^t|}
$$

track purity：

$$
Purity(P)=\max_g\frac{\sum_t |P^t \cap G_g^t|}{\sum_t |P^t|}
$$

判断标准：

```text
IDF1 比 v3 memory 提升至少 +5。
ID switches 降低至少 30%。
fragmentation 降低至少 30%。
至少 5 个遮挡/重现成功案例可视化。
```

失败时：

```text
ID switch 多：增加 appearance 权重和 same-frame exclusivity。
fragmentation 多：降低 lost reactivation threshold。
over-merge 多：增强 motion gate 和 conflict penalty。
appearance 漂移：改用 DINO / multi-crop CLIP / RGB histogram ensemble。
```

可视化：

```text
RGB video with object IDs
carrier tracks colored by object ID
object lifecycle timeline
ID switch markers
occlusion/reappearance panels
3D trajectory visualization
```

### 11.5 D2：dynamic semantic 4D field query

假设：

```text
H_D2: object-level semantic memory 比逐帧 CLIP label 更稳定，能支持 time-sensitive query。
```

查询类型：

```text
class query: chair, person, ball, cup 等可见类别
motion query: moving object, object being carried, object entering scene
temporal query: same object before occlusion, same object after reappearance
relation query: object near person, object on table, if data supports
```

指标：

```text
label_temporal_consistency
text_query_top1_stability
query_localization_iou, if GT exists
false_temporal_grounding_rate
object_language_entropy
CLIP_score_margin
```

判断标准：

```text
label consistency 比 per-frame CLIP 提升至少 20%。
query localization 不出现明显跨 object 漂移。
若有 semantic GT，semantic AP 不低于 Stream3D post-hoc CLIP。
```

若无 semantic GT：

```text
只报 qualitative 和 consistency，不报 semantic AP。
```

### 11.6 D3：D4RT geometry evaluation-only Sim3

假设：

```text
H_D3: D4RT tracks 经 evaluation-only scale/Sim3 alignment 后可以支持动态 3D trajectory evaluation。
```

实验：

```text
scale-only alignment
scene-level Sim3
chunk-level Sim3
```

指标：

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

判断标准：

```text
若 D4RT track 本身失败，必须拆分为 geometry failure，不要归咎于 semantic memory。
```

失败时：

```text
缩短 window。
增加 overlap。
只用 high-confidence anchors。
检查 Dynamic Replica camera convention 和 depth decode。
```

---

## 12. 最终必须输出的表格

### 12.1 ScanNet 主表

```text
Method | 2D masks | D4RT ckpt | frames | memory | densify | export | eval policy | AP | AP50 | AP25 | pre_points % | union % | GT crop/full | #pred | time/frame
```

必须包含：

```text
Stream3D-Cropformer baseline original/self-inherit
Stream3D-Cropformer on Stream4D-MVP pre_points, diagnostic
Stream3D-Cropformer on Stream4D-v3 adaptive pre_points, diagnostic
Stream4D-v3 adaptive top-k recompute
Stream4D-v3 adaptive top-k inherit
Stream4D-v3 adaptive on Stream3D baseline pre_points, diagnostic
Stream4D-v4 evidence
Stream4D-v4 densify
Stream4D-v4 memory-v2
Stream4D-v4 densify + memory-v2
```

每个 Stream4D 方法至少两行：

```text
recompute_pre_points
inherit_pre_points
```

每个最终候选 Stream4D 方法还要追加一行：

```text
fixed_pre_points = scannet baseline pre_points
```

每个 Stream3D baseline 至少三行：

```text
self-inherit / original
fixed_pre_points = Stream4D MVP pre_points
fixed_pre_points = Stream4D locked candidate pre_points
```

### 12.2 ScanNet 消融表

```text
Evidence | Densify | Memory | Sim3 export | AP | AP50 | AP25 | union % | conflict | #objects | Frag | Merge
```

### 12.3 Dynamic Replica 主表

```text
Method | GT type | IDF1 | IDSW | Frag | Track purity | 4D IoU | APD3D | AJ | OA | EPE | Reactivation
```

如果使用 pseudo masks，表名必须含：

```text
Pseudo object-consistency evaluation
```

不能写成 official benchmark result。

### 12.4 必须可视化

ScanNet：

```text
prediction vs GT mesh
prediction union heatmap
missed GT instances
false positives
carrier seeds and densified pixels
object memory timeline
Sim3 residual overlay
per-object IoU histogram
```

Dynamic Replica：

```text
2D tracks overlay
3D trajectories
object ID timeline
ID switch frames
occlusion reactivation cases
query text to object trajectory videos
```

---

## 13. 最终 claim 安全边界

### 13.1 可以 claim 的条件

只有满足以下条件，才能写：

```text
Stream4D-v4 outperforms Stream3D-Cropformer on ScanNet.
```

条件：

```text
1. Stream3D-style evaluator，与 baseline 对齐。
2. locked hyperparameters，final split 成立。
3. recompute AP 至少 +0.5，AP50 至少 +1.0。
4. inherit AP 相比 v3 inherit 明显提升，最好接近或超过 baseline。
5. 必须同时报告 Stream3D self-inherit 和 Stream3D cross-fixed-pre_points；不能只拿 Stream4D inherit 与 Stream3D original 比。
6. Stream4D fixed on Stream3D baseline pre_points 不能灾难性低于 v3 inherit，否则 claim 必须降级为 sparse observed-support improvement。
7. 没有进一步缩小 support 来换 AP。
8. metric integrity audit 通过。
```

### 13.2 如果 ScanNet 仍只是持平

如果 ScanNet 只能做到 comparable，但 Dynamic Replica 明显更好，论文主线应调整为：

```text
Stream3D solves streaming static 3D mask merging; Stream4D solves feed-forward semantic 4D reconstruction and tracking. On static ScanNet it remains comparable or slightly better, while on dynamic scenes it significantly improves identity consistency and 4D tracking.
```

这是更安全也更符合方法本质的叙事。

### 13.3 禁止 claim

```text
禁止说当前 v3 已证明 dynamic semantic 4D reconstruction。
禁止说当前 ScanNet 结果是 D4RT-native geometry。
禁止在没有 GT semantic/object IDs 的 Dynamic Replica 上报 semantic AP / official IDF1。
禁止让 Sim3 进入方法内部。
禁止隐藏 inherit_pre_points 失败。
禁止把 validation grid search 的结果直接当 final。
```

---

## 14. Codex 第一批具体任务清单

Codex 先做这些，完成后再跑大实验。

### Task 1：metric integrity guard

新增：

```text
tools/verify_stream4d_metric_integrity.py
```

功能：

```text
compare evaluator hash
scan rescore source for gt_path reads
check prediction/TMP/object_dict alignment
summarize pre_points policy
write markdown/json
```



### Task 1b：Stream3D inherit / cross-fixed baseline

新增：

```text
tools/evaluate_cross_prepoints.py
```

必须产出：

```text
data/evaluation/scannet/scannet_self_inherit_class_agnostic.txt
data/evaluation/scannet/scannet_on_stream4d_mvp_prepoints_class_agnostic.txt
data/evaluation/scannet/scannet_on_stream4d_adaptive_prepoints_class_agnostic.txt
data/evaluation/scannet/stream4d_adaptive_on_scannet_prepoints_class_agnostic.txt
outputs/audit/cross_prepoints_audit.{md,json,csv}
```

验收：

```text
scannet_self_inherit == scannet original, AP delta < 0.01。
所有 cross-fixed rows 通过 shape audit。
最终 ScanNet 表格中补上 Stream3D 在 inherit/fixed-pre_points 下的结果。
```

### Task 2：object_dict-pred alignment check

修改：

```text
stream4d/rescore_scannet.py
stream4d/reexport_scannet.py
stream4d/export_scannet.py
```

新增：

```text
object_id_to_column.json
alignment diagnostics
```

### Task 3：diagnostic naming fix

修：

```text
reexport point_dilate support mode naming
reuse_point_ids hit rate naming
fixed_path validation
```

### Task 4：evidence quality v1

新增：

```text
stream4d/evidence_quality.py
```

接入：

```text
MaskEvidenceBuilder
run_scannet.py CLI
```

### Task 5：reliable densifier v1

新增：

```text
stream4d/reliable_densifier.py
```

先实现：

```text
seeded component
boundary erosion
seed distance cap
WTA conflict
```

### Task 6：memory-v2 small-scale

新增：

```text
stream4d/object_memory_v2.py
stream4d/appearance_memory.py
```

先支持：

```text
RGB histogram feature
carrier IoC
point IoC when available
Hungarian matching
lifecycle timeline
```

### Task 7：Dynamic Replica env checker

新增：

```text
tools/check_dynamic_replica_env.py
stream4d/dynamic_replica_stream.py
```

只检查数据，不先写指标。

---

## 15. 每次实验完成后必须写的复盘模板

每轮实验都写一个 markdown：

```text
docs/stream4d_v4_{phase}_实验复盘.md
```

模板：

```text
1. 本轮目标
2. 假设
3. 修改了哪些代码
4. 运行命令
5. 完整性检查
6. 主指标
7. 诊断指标
8. 可视化路径
9. 假设是否成立
10. 失败原因
11. 下一步尝试方向
12. 哪些结论不能写
```

不允许只有 AP 表格；必须同时包含 coverage、conflict、object count、pre_points policy、tune/final 标记。
