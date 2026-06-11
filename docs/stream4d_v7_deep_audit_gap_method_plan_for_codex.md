# Stream4D v7 深度代码审核、差距确认与方法改进实验计划书

面向 Codex 执行。本计划基于最新代码审计包：

```text
/mnt/data/stream4d_v6_code_review_packet.zip
```

我已将其解压到：

```text
/mnt/data/audit_v6_latest/Stream3D
```

并对照以下文件和结果复盘进行审核：

```text
docs/stream4d_v6_method_first_audit_and_experiment_plan_for_codex.md
docs/stream4d_v6_实验结果复盘.md
docs/stream4d_v5_deep_audit_and_parallel_experiment_plan_for_codex.md
docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
```

本文档只使用 Typora 友好的公式格式 `$...$` 和 `$$...$$`，不使用其他 display 公式语法。

---

## 0. 给 Codex 的总要求

这轮不要继续做零散后处理，也不要只报告“又跑了一个 config”。目标是三件事：

第一，**确实量化 Stream4D 和 Stream3D 的差距**。必须在相同 support、相同 evaluator、相同 split 下给出差距，不能只看 own recompute。

第二，**验证 D4RT geometry 替换 Stream3D RGB-D/pose geometry 后到底会掉多少**。这不是可选项。必须新增 `Stream3D-D4RT geometry degradation experiment`。

第三，**方法思路必须升级**。当前 v6 的大量结果已经证明，继续调 top-k、fusion、NMS、score mode、radius growth 的收益非常有限。v7 的算法主线必须从“prediction 后处理”转向“D4RT carrier-tracklet 驱动的 object formation”。

每次提交时，Codex 必须提交一个完整 code audit packet，不能只提交实验报告。详见第 2 节。

---

## 1. 当前代码审核结论

### 1.1 审计包完整性

最新 zip 已经比 v5/v4.1 更完整，包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/evaluate.py
Stream3D/tests/test_stream4d_protocol_fixes.py
Stream3D/docs/*
Stream3D/logs/*
Stream3D/data/evaluation/scannet/*.txt
Stream3D/outputs/*/*.json
```

但是它没有包含完整：

```text
Stream3D/data/prediction/*
Stream3D/data/TMP/*
Stream3D/data/scannet/gt/*
ScanNet 原始 RGB-D / mesh 数据
```

因此这次在审计环境里可以做静态代码审核、manifest/GT-leakage 检查、结果文件读取和计划一致性检查；但不能仅靠这个 zip 重新计算完整 AP。下一次提交如果需要我复核 AP，必须至少提供：

```text
1. 结果 config 的 data/evaluation/scannet/*.txt。
2. 对应 outputs/audit/*metric_integrity*.json。
3. 对应 outputs/*_summary.json。
4. 每个 reportable config 的 config_manifest.json。
5. 若要求我复跑某个 probe scene，必须附带该 scene 的 prediction/TMP/object_dict 最小包。
```

### 1.2 语法与测试

我在当前容器里运行：

```bash
cd /mnt/data/audit_v6_latest/Stream3D
python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
```

结果：

```text
py_compile: pass
```

但运行：

```bash
python -m unittest tests.test_stream4d_protocol_fixes
```

在当前容器失败，原因是当前环境缺少 `open3d`：

```text
ModuleNotFoundError: No module named 'open3d'
```

这不是算法代码必然失败，但说明**审计包不是 self-contained runnable**。日志里声称在原环境 unit tests 通过，但下一次提交必须同时提交：

```text
1. 原环境 unit test log。
2. Python/conda environment summary。
3. 若希望第三方审计通过，tests 中不应在 import 阶段强依赖 open3d；需要把 open3d 依赖延迟到具体 test 或提供 mock。
```

### 1.3 虚假指标 / GT 泄漏审计

当前没有发现非 oracle method 脚本直接读取 GT 来生成 prediction 的证据。关键点如下：

```text
1. evaluation/evaluate.py 新增 manifest guard：--require-manifest、uses_gt=true 拒绝、diagnostic-only 拒绝。
2. tools/oracle_candidate_upper_bound.py 明确是 GT-read oracle diagnostic，并强制 output_config 包含 oracle，manifest 写 uses_gt=true 和 is_diagnostic_only=true。
3. tools/scan_reportable_configs.py 能扫描 uses_gt_and_method_result。
4. tools/fuse_prediction_configs.py 在使用 scannet/Stream3D external support 做 overlap suppression 时会把结果标记为 diagnostic-only。
5. v6 复盘记录 reportable configs 的 metric integrity phase0_pass=true、uses_gt=false。
```

但是仍有两个风险需要继续 guard：

**风险 A：retroactive manifest 不能证明历史运行时安全。** 如果一个旧 artifact 后补 manifest，只能说明“报告阶段标注了类型”，不能证明生成时没有问题。Codex 后续报告必须把 retroactive manifest 和 live-run manifest 分开。

**风险 B：evaluator 仍允许不加 `--require-manifest` 运行。** 对内部临时调试可以，但所有 reportable method 结果必须强制使用：

```bash
python -m evaluation.evaluate ... --require-manifest
```

否则不允许进入主表。

### 1.4 和 v6 计划 / implementation 的一致性

已经相符的部分：

```text
1. evaluator manifest guard 已实现。
2. oracle_candidate_upper_bound 有 GT 使用保护。
3. local proposal score-mode bug 已修复：mask_backproject record 写入 observations/area_score/carrier_count/reliability。
4. D4RT geometry diagnostic 已实现，但只是 Sim3 anchor diagnostic，不是 G1 segmentation AP。
5. typed evidence graph v3 minimal candidate 已实现并运行。
6. core/fringe/reject、ownership WTA、object competition、fusion 等后续诊断工具已实现并跑过。
```

不相符或只部分相符的部分：

```text
1. 完整 Typed Evidence Graph v3 没有完成。当前实现主要是 mask-node graph，不是 carrier-tracklet graph。
2. core/fringe/reject 主要是 prediction 后处理，不是 object formation 阶段的一等表示。
3. ObjectMemory4D-v2 仍未成为有效 streaming 4D memory；当前结果显示多窗口没有带来收益。
4. D4RT-native export 仍未实现：stream4d/export_scannet.py 中 export_d4rt_nn 仍是 NotImplementedError。
5. Dynamic Replica 数据缺失，不能报告动态 tracking 指标。
6. Stream3D-D4RT geometry replacement 实验尚未完成。
```

### 1.5 关键代码问题与必须修复项

#### P0：typed evidence graph 的“positive_track”实际上不是 D4RT track-level evidence

`tools/export_typed_evidence_graph_v3.py` 里，node 是 `(frame_id, mask_id)` mask observation，edge 的 shared 依据是 `point_ids` 的 3D backproject overlap。代码虽然把边叫做 `positive_track`，但它并没有直接使用 D4RT carrier ID / trajectory identity 来连接 node。当前逻辑更接近：

```text
2D mask -> RGB-D backprojected 3D point set -> overlap graph
```

而不是计划里想要的：

```text
D4RT carrier trajectory -> carrier co-membership graph -> object hypothesis
```

这很可能是 typed graph 失败的本质原因之一。它把 D4RT 的核心优势丢掉了，又回到了 Stream3D-style 3D mask overlap。v7 必须重写 graph node 和 edge 的定义，详见第 6 节。

#### P0：Stream3D-D4RT geometry replacement 未实现

当前 `export_d4rt_nn()` 仍是：

```text
NotImplementedError("d4rt_nn export requires a scene-coordinate calibration path; rgbd_eval is the MVP default.")
```

所以当前 ScanNet 结果不能写成 D4RT-native geometry。v7 必须实现 geometry degradation experiment，哪怕先只做 5-scene probe。

#### P1：WTA 后 record score 没有随 point_ids 重新计算

`stream4d/reliable_densifier.py::apply_wta_to_records()` 会改变每个 object 的 `point_ids`，但 `area_score`、`reliability`、`dense_quality` 等字段在 WTA 后没有统一重算。若后续使用 `export_score_mode=area/reliability/dense_quality`，score 可能与最终 mask 不一致。

修复要求：

```text
apply_wta_to_records 后统一调用 recompute_record_scores(record)
至少重算 area_score、score、reliability、dense_quality、selection_quality。
```

如果不能重算，所有 WTA score-mode 实验只能标记为 diagnostic。

#### P1：evaluation/evaluate.py 在 import 阶段解析 argparse，导致测试和复用困难

当前 evaluator 在 module 顶层执行：

```python
opt = parser.parse_args()
```

这继承了旧代码风格，但会让单元测试、import smoke 和工具复用变得脆弱。v7 不要求立刻重构 evaluator core，但新增工具不要继续复制这种模式。若要增强 evaluator，应把 argument parsing 放进 `main()`，把 AP core 保持不变。

#### P1：当前 evaluator 强依赖 CUDA

`get_gt_tensor()` 直接 `.cuda()`。这与原项目环境一致，但不利于第三方审计和 CPU smoke。Codex 需要新增一个 tiny-eval CPU unit test，验证 manifest guard 和 toy AP，不改变正式 GPU evaluator。

---

## 2. 下一次必须提交的代码审计包格式

用户明确要求我继续审核代码，所以 Codex 每轮都必须提交 zip，命名格式：

```text
stream4d_v7_code_review_packet_YYYYMMDD_HHMM.zip
```

zip 必须包含：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/*.py
Stream3D/tests/*.py
Stream3D/docs/本轮计划.md
Stream3D/docs/本轮执行日志.md
Stream3D/docs/本轮实验结果复盘.md
Stream3D/logs/*本轮*.log
Stream3D/outputs/audit/*本轮*.json
Stream3D/outputs/audit/*本轮*.md
Stream3D/data/evaluation/scannet/*本轮*.txt
```

如果报告任何 AP，必须同时提交：

```text
1. config_manifest.json 的拷贝或 manifest scanner 输出。
2. metric integrity JSON。
3. eval command log，必须带 --require-manifest。
4. pre_points/union/GT crop/full 统计。
5. 是否使用 external support / Stream3D support / oracle support 的显式字段。
```

如果报告代码修改，必须同时提交：

```text
git diff 或 patch 文件
git status
sha256sum of code packet
py_compile log
unit test log
import smoke log
```

如果结果来自 oracle diagnostic，必须：

```text
output_config 包含 oracle
manifest: uses_gt=true, is_diagnostic_only=true, is_method_result=false
evaluator 若运行必须显式 --allow-oracle-eval
报告表格标题必须写 oracle upper bound，不得进入 method table
```

如果某次只提交报告、没有代码包，我会认为该轮无法完成代码审计。

---

## 3. 当前结果和 Stream3D 的真实差距

### 3.1 必须区分三类比较

**第一类：own recompute。** 每个 prediction 用自己的 `pre_points = prediction union`。这继承 Stream3D-style cropped-TMP evaluator 的配置自评逻辑，但容易奖励 sparse high-precision support。

**第二类：same fixed support。** prediction 不变，但评价 support 固定到同一个 universe，例如 Stream4D 32f support。这用来判断 object quality 差距。

**第三类：Stream3D large support / full support diagnostic。** 把 Stream4D 放回 Stream3D 的大 support，主要暴露 recall / coverage 缺口。

这三类都要报告，否则无法“确实知道差距”。

### 3.2 probe5 上当前最有意义的差距

v6 复盘中的关键对照如下：

| 方法 | AP | AP50 | AP25 | #pred | 说明 |
|---|---:|---:|---:|---:|---|
| `scannet_on_stream4d_32f_probe5` | `0.399213` | `0.597171` | `0.742535` | `128.2` | Stream3D prediction 放在 Stream4D 32f support 上的 same-support diagnostic |
| v4.1 current best | `0.281615` | `0.497583` | `0.690254` | `415.6` | v4.1 strong sparse-support result |
| v6 best AP/AP50: compact-only | `0.284832` | `0.503962` | `0.671915` | `295.6` | #pred 过 gate，但 AP/AP50/AP25 未过 gate |
| v6 best balanced: scoreunique | `0.282050` | `0.498231` | `0.691264` | `295.6` | 三项小幅超 v4.1，但未过 gate |
| v6 localprop score-one | `0.112524` | `0.285390` | `0.479773` | `173.2` | 高 support local proposal，质量失败 |
| v6 typed graph minimal | `0.033478` | `0.128495` | `0.397753` | `57.2` | over-merge / missing split 失败 |

如果以 v6 AP/AP50 最好的 compact-only 和 Stream3D same-support diagnostic 比，差距是：

$$
\Delta AP = 0.284832 - 0.399213 = -0.114380
$$

$$
\Delta AP50 = 0.503962 - 0.597171 = -0.093209
$$

$$
\Delta AP25 = 0.671915 - 0.742535 = -0.070621
$$

换成百分制，就是约：

```text
AP   低 11.44 点
AP50 低 9.32 点
AP25 低 7.06 点
```

如果以 balanced scoreunique 比，差距是：

```text
AP   低约 11.72 点
AP50 低约 9.89 点
AP25 低约 5.13 点
```

这说明当前不是“差一点点”。即使 v6 把 #pred 压到 300 以下，也仍然明显落后 Stream3D 在同一小 support 上的 object quality。

### 3.3 这轮有没有进展

有进展，但进展不是“方法达标”。真实进展有三点：

第一，**metric safety 有进展**。manifest guard、oracle guard、reportable config scanner、metric integrity 基本建立起来了。

第二，**#pred 控制有进展**。v6 no-group/top-k 能把 #pred 从 v4.1 的 `415.6` 降到 `295.6`，并且 AP/AP50 与 v4.1 基本持平或小幅提升。这说明过去一个 blocker：prediction 太多，已经局部缓解。

第三，**负结果帮助定位了问题**。localprop、score-mode、typed graph minimal、core/fringe、radius growth、fusion 基本都失败，说明问题不在某个 score 或 NMS 阈值，而在 object formation 本身。

### 3.4 进度为什么显得慢

慢的根本原因不是工程跑得慢，而是方向多次停留在后处理层：

```text
已有 prediction -> 筛选 / top-k / WTA / fusion / NMS / radius growth -> 评价
```

这些操作只改变最终候选集合或点归属，不能修复“候选对象本身不对”的问题。当前结果已经显示：

```text
1. localprop union 很大，但 AP 低，说明 recall 候选很脏。
2. v4.1 candidate 很干净，但 support 很小，说明 high precision selector 漏召回。
3. fusion 把两者合起来会污染 AP，说明没有可靠的 object-level ownership。
4. typed graph 用的是 backprojected mask point overlap，不是真正 D4RT tracklet graph，没利用 D4RT 最核心优势。
```

因此继续在这些后处理工具之间组合，边际收益会越来越小。

### 3.5 是否在正确道路上

**总体目标方向是对的：D4RT carrier 应该解决 Stream3D 静态 3D overlap 无法解决的 4D correspondence 和 dynamic tracking。**

但**当前实现路线偏离了这个目标**：很多 v6 工具把 D4RT carrier 退化成 sparse support / mask overlap signal，没有把 D4RT 的 material point identity 作为 object formation 的核心。因此在静态 ScanNet 上，它既没有 Stream3D 的 dense RGB-D geometry coverage，又没有充分发挥 D4RT 的 4D tracking 优势。

离目标还差两层：

```text
层 1：ScanNet same-support object quality 仍低 Stream3D 约 9-12 AP/AP50 点。
层 2：Dynamic Replica 4D tracking 还没开始，不能证明 feed-forward semantic 4D reconstruction and tracking。
```

---

## 4. v7 总体实验目标和假设

### 4.1 总体目标

v7 的整体目标是把当前系统从：

```text
sparse D4RT-carrier assisted 3D mask postprocessing
```

推进到：

```text
D4RT carrier-tracklet driven semantic 4D object formation and tracking
```

ScanNet 的目标不是单纯 own recompute 高分，而是确实回答：

```text
在同一 support / 同一 evaluator 下，我们和 Stream3D 差多少？
D4RT geometry 替代 Stream3D RGB-D/pose geometry 会掉多少？
D4RT tracklet object formation 是否能缩小与 Stream3D 的 object quality 差距？
```

Dynamic Replica 的目标是回答：

```text
D4RT carrier identity 是否在动态对象、遮挡重现、跨帧跟踪上超过 Stream3D 静态 3D overlap memory？
```

### 4.2 核心假设

#### H1：当前差距主要来自 object formation，而不是 evaluator 或 GT 泄漏

若 H1 成立，则在同一 support 下，Stream4D 的 per-GT best IoU、duplicate rate、one-to-one matching quality 会明显弱于 Stream3D。v7 需要用 per-GT oracle-free diagnostics 证实。

#### H2：D4RT geometry 直接替代 Stream3D RGB-D/pose geometry 在 ScanNet 上大概率下降

原因是 Stream3D 的 set-cover、manifold refining、3D mask merging 都强依赖 metric RGB-D/pose。D4RT 的优势是 dynamic correspondence，而不是一定能替代 ScanNet 的 RGB-D geometry 做静态 dense 3D projection。

#### H3：真正有效的 D4RT 改进不应是“D4RT geometry 替换 depth”，而应是“D4RT carrier tracklet 替换 3D mask overlap 作为 object identity backbone”

如果 H3 成立，carrier-tracklet graph 在 same-support AP/AP50、duplicate rate、ID consistency 上应优于 current v6 typed mask-node graph。

#### H4：静态 ScanNet 不一定是 D4RT 的最强场景，动态 Replica 才是必须验证的主优势

如果 Dynamic Replica 上 D4RT tracklet memory 显著减少 ID switch / fragmentation，即使 ScanNet 静态 AP 只接近 Stream3D，也可以支撑“4D reconstruction and tracking”动机。

---

## 5. Phase A：确实量化和 Stream3D 的差距

### 5.1 目标

生成一个不可回避的 gap matrix，明确告诉我们：

```text
1. 在 own recompute 下谁强。
2. 在 Stream4D 32f support 下谁强。
3. 在 Stream3D support 下谁强。
4. 在 v6 best support 下 Stream3D 会是多少。
5. Stream4D 放到 Stream3D 大 support 后到底崩到什么程度。
```

### 5.2 必跑配置

Prediction rows：

```text
P0: scannet, 原 Stream3D-Cropformer prediction
P1: stream4d_32f_self_probe5
P2: stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
P3: stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve
P4: stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve
P5: stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075
P6: stream4d_v6_typedv3_probe5_c2_trackonly_ioc060
```

Support columns：

```text
S0: own recompute/self
S1: stream4d_32f_self_probe5 support
S2: v4.1 current best support
S3: v6 compact-only support
S4: v6 scoreunique support
S5: scannet / Stream3D support
S6: union support = union(S1, S3, S4) diagnostic
```

### 5.3 实验命令要求

Codex 新增或复用：

```bash
python -m tools.evaluate_cross_prepoints \
  --pred-config <P> \
  --target-pre-points-config <S> \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config <P>_on_<S> \
  --require-manifest \
  --summary-root outputs/audit/v7_gap_matrix
```

如果工具不支持 union support，则新增 `tools/make_union_prepoints_config.py`：

```text
输入多个 TMP config，输出每 scene 的 union pre_points。
manifest 写 is_diagnostic_only=true，support_policy=union_prepoints。
```

### 5.4 必须记录的指标

每个 cell 必须记录：

```text
AP / AP50 / AP25
mean_pre_points_ratio
mean_prediction_union_ratio
union_in_target_ratio_scene
union_in_target_ratio_target
mean_gt_instances_crop / full
mean_num_pred_instances
mean_pred_area
median_pred_area
duplicate_prediction_rate
per_gt_best_iou_mean
per_gt_best_iou_ge_0.25 / 0.50 / 0.75
per_scene_delta_vs_stream3d
```

新增 `tools/diagnose_prediction_quality.py`，不读 GT 版本记录：

```text
mask overlap distribution
support conflict rate
#pred per scene
area histogram
point ownership conflict
```

读 GT 版本只能叫 diagnostic，不能选模型：

```text
per-GT best IoU
GT coverage by prediction union
duplicate predictions per GT
missed GT instance count
```

### 5.5 判断标准

该阶段不追求改进，只追求确切差距。通过标准：

```text
1. 每个 cell 都有 AP 和 support 统计。
2. Stream3D self-inherit 与原 baseline 一致。
3. 所有 reportable method cell 的 manifest uses_gt=false。
4. 如果使用 GT diagnostic，必须标记 diagnostic-only，不能出现在 method table。
```

如果发现某个结果异常高但 `mean_pre_points_ratio < 1%`，必须标记为 tiny-support diagnostic，不得作为进展。

---

## 6. Phase B：Stream3D 使用 D4RT geometry 会不会下降

### 6.1 目标

用户明确要求：必须知道 Stream3D 使用 D4RT geometry 性能是否会下降。本阶段专门回答这个问题。

### 6.2 实验设计

我们比较以下几条几何路径：

| ID | 几何路径 | 方法内部是否用 GT RGB-D/pose | 评估是否 Sim3 | 目的 |
|---|---|---:|---:|---|
| G0 | 原版 Stream3D RGB-D/pose geometry | 是 | 否 | baseline |
| G1 | Stream3D + D4RT shared-reference point map | 否 | 是，仅 export/eval | 真正测试 D4RT geometry 替换 |
| G2 | Stream3D + D4RT depth + D4RT camera pose | 否 | 是，仅 export/eval | 若 D4RT pose path 可用，测试完整 feed-forward geometry |
| G3 | Stream3D + D4RT geometry + normalized manifold threshold | 否 | 是，仅 export/eval | 测试 MR 是否因尺度不匹配而失败 |
| G4 | Stream3D + D4RT track-consistency MR 替代 Euclidean MR | 否 | 是，仅 export/eval | 测试 D4RT 正确用法 |
| G5 | Stream3D RGB-D geometry + D4RT carrier evidence filtering | 是 | 否 | 分离 geometry 替换与 D4RT correspondence 收益 |

注意：G1-G4 的 Sim3 只允许用于评估导出，不允许进入 Stream3D 的 mask selection、manifold refining、memory matching。内部运行只能使用 D4RT coordinate。

### 6.3 D4RT point map 构建

对每个 frame 的 mask pixel 或采样 pixel，查询：

$$
q=(u,v,t_{src},t_{tgt},t_{cam})
$$

为了在一个共享坐标里生成 point cloud，设置：

$$
t_{src}=t, \quad t_{tgt}=t, \quad t_{cam}=t_{ref}
$$

输出：

$$
X^{D4RT}_{p,t}=D(q,F) \in \mathbb{R}^3
$$

再用 2D mask 生成 D4RT-coordinate 3D mask：

$$
M^{D4RT}_{m,t}=\{X^{D4RT}_{p,t}: p \in m_t\}
$$

Stream3D 的 set-cover 和 merging 在 D4RT coordinate 内执行。

### 6.4 Sim3 evaluation-only adapter

生成最终 prediction 后，为了映射到 ScanNet evaluator mesh，只在 export 阶段拟合：

$$
T^* = \arg\min_{s,R,t}\sum_i \omega_i \left\|sR x_i^{D4RT}+t-x_i^{ScanNet}\right\|_2^2
$$

记录：

```text
sim3_anchor_count
sim3_scale
sim3_residual_mean / median / p90 / p95
inlier_ratio@5cm / 10cm / 20cm
nn_hit_rate_to_mesh
```

不得把 $T^*$ 写回 method cache，也不得用它改变 object grouping。

### 6.5 必须记录的几何指标

对每个 scene/window：

```text
D4RT-vs-RGBD depth AbsRel after scale
D4RT-vs-RGBD depth RMSE after scale
Sim3 residual median / p90
D4RT point map valid ratio
D4RT point map uv_in01_rate
D4RT point cloud density per mask
mask 2D-to-D4RT projection hit rate
D4RT 3D mask area distribution
```

对 segmentation：

```text
AP / AP50 / AP25
union ratio
#pred
MR deletion ratio
set-cover selected mask count
mask projection IoU between RGB-D Stream3D mask and D4RT Stream3D mask
per-scene failure reason
```

### 6.6 判断标准

如果：

```text
G1/G2 相对 G0 AP drop > 3.0 AP 或 AP50 drop > 5.0 AP50
```

则结论必须写成：

```text
D4RT geometry 直接替代 Stream3D RGB-D/pose geometry 在 ScanNet 上显著下降；ScanNet 主结果不应 claim D4RT-native geometry 已经优于 Stream3D geometry。后续应把 D4RT 用作 4D correspondence/object identity，而不是静态 ScanNet metric geometry replacement。
```

如果：

```text
G3 明显优于 G1/G2
```

说明尺度和 manifold threshold 是主要瓶颈，应采用 normalized MR：

$$
\delta = \alpha \cdot \operatorname{median}_{i}\operatorname{NNdist}(p_i)
$$

如果：

```text
G4 明显优于 G3
```

说明 D4RT track-consistency 是正确几何 prior，应减少 Euclidean MR 依赖。

### 6.7 不满足条件时 Codex 先尝试

如果 Sim3 residual 仍像 v6 diagnostic 一样 median 约 0.68m：

```text
1. 检查 D4RT 输出 coordinate 是 camera frame 还是 normalized scene frame。
2. 检查 tcam 是否固定到正确 reference。
3. 检查 src_xy 使用的是原图坐标还是 resize 后坐标。
4. 检查 D4RT preprocessing 的 crop/resize/padding 与 ScanNet pixel coordinate 对齐。
5. 尝试 per-window Sim3、per-scene Sim3、static-anchor Sim3 三种对齐。
6. 可视化 residual heatmap，不要直接启动 segmentation AP。
```

如果 G1 segmentation 无法跑完：

```text
先跑 scene0050_00 / scene0011_00 / scene0030_00 三场景。
先只替换 local projection，不跑 full historical memory。
先关闭 MR，再打开 normalized MR。
```

---

## 7. Phase C：真正的方法改进，carrier-tracklet object formation

### 7.1 为什么必须改算法

当前 v6 typed graph 的 node 是 mask observation，edge 是 backprojected 3D point overlap。这不能充分体现 D4RT 的强项。D4RT 的强项是：一个 source point 可以被 query 到多个 target time 和 camera reference，它天然提供 material point / trajectory identity。

因此 v7 的核心表示要从：

```text
mask node graph
```

改成：

```text
carrier-tracklet graph
```

每个 carrier $c_i$ 是 source pixel anchored physical surface hypothesis：

$$
c_i = (u_i, v_i, t_i^{src})
$$

它在时间上的位置是：

$$
X_i(t)=D(u_i,v_i,t_i^{src},t,t_{ref})
$$

它在每帧的 semantic evidence 是：

$$
E_i(t)=\{m_t, \rho_i(t), b_i(t), a_i(t)\}
$$

其中 $m_t$ 是它落入的 2D mask，$\rho_i(t)$ 是 D4RT visibility/confidence，$b_i(t)$ 是 boundary safety，$a_i(t)$ 是 appearance / mask evidence。

### 7.2 Carrier pair evidence

对于两个 carriers $i,j$，定义正证据：

$$
A^+_{ij}=\frac{\sum_t \mathbb{1}[m_i(t)=m_j(t) \land visible_i(t) \land visible_j(t)] w_i(t)w_j(t)}{\sum_t \mathbb{1}[visible_i(t) \land visible_j(t)] w_i(t)w_j(t)+\epsilon}
$$

定义反证据：

$$
A^-_{ij}=\frac{\sum_t \mathbb{1}[m_i(t)\neq m_j(t) \land visible_i(t) \land visible_j(t)] w_i(t)w_j(t)}{\sum_t \mathbb{1}[visible_i(t) \land visible_j(t)] w_i(t)w_j(t)+\epsilon}
$$

定义 trajectory compatibility：

$$
G_{ij}=\exp\left(-\frac{\operatorname{Var}_{t}\left(\left\|X_i(t)-X_j(t)\right\|_2\right)}{\sigma_g^2}\right)
$$

静态或刚体物体上，pairwise distance 更稳定；非刚体可放宽为 local neighborhood consistency，不强行要求全局刚体。

最终 pair score：

$$
S_{ij}=\lambda_+A^+_{ij}+\lambda_gG_{ij}+\lambda_aA^{app}_{ij}-\lambda_-A^-_{ij}-\lambda_cC_{ij}
$$

其中 $C_{ij}$ 是 same-frame strong cannot-link，比如同一帧落入互斥实例 mask 且长期分离。

### 7.3 Graph partition

使用 training-free constrained clustering，不训练新模型。第一版可用 greedy correlation clustering：

```text
1. 先建立 strong cannot-link components。
2. 按 S_ij 从高到低 union。
3. union 前检查是否违反 cannot-link、same-frame exclusivity、component size/extent。
4. 对弱桥接 edge 只允许 attach 到已有 strong component，不允许连接两个 strong component。
5. 输出 carrier clusters。
```

组件 $C_k$ 是 object core，不是最终 dense mask。

### 7.4 Core / fringe / reject support

对 object carrier cluster $C_k$，每帧从 2D mask 中恢复 support：

```text
core: 包含多个高置信 carrier 的 mask connected component。
fringe: 与 core 连通、边界安全、且不被其他 object 强占的区域。
reject: 同帧 cannot-link、boundary unsafe、trajectory inconsistent 的区域。
```

像素 $p$ 加入 object $k$ 的分数：

$$
R_k(p,t)=\alpha d_k(p,t)+\beta b(p,t)+\gamma v_k(t)+\eta a_k(p,t)-\mu \max_{l\neq k} R_l(p,t)
$$

其中 $d_k$ 是到 object carrier seeds 的距离变换分数，$b$ 是 boundary safety，$v_k$ 是 view reliability，$a_k$ 是 appearance compatibility。

输出策略：

```text
1. core 一定进入。
2. fringe 只有在 R_k 超阈值且无强冲突时进入。
3. reject 永不进入。
4. 每个 3D point / pixel 最终做 ownership WTA。
```

### 7.5 为什么这个方向比继续后处理更合理

当前实验已经证明：

```text
1. localprop 有较高 support 但 AP 极低，说明 raw recall 不等于 object。
2. v4.1/v6 top-k 有高 precision 但 sparse，说明只筛选会漏召回。
3. fusion 会污染，说明候选之间缺少 ownership。
4. mask-node typed graph over-merge，说明 3D point overlap 不是可靠 object identity。
```

carrier-tracklet graph 直接针对这些问题：

```text
1. 用 carrier co-membership 做 identity，而不是 mask overlap。
2. 用 cannot-link 处理相似外观/相邻物体。
3. 用 core/fringe/reject 分层处理 support，而不是把 mask 全部 backproject。
4. 用 ownership WTA 解决重复预测。
```

### 7.6 Phase C 实验设置

先只跑 probe5，不跑 full ScanNet：

```text
scene0011_00
scene0030_00
scene0050_00
scene0081_01
scene0591_00
```

配置：

```text
C0: v6 best compact-only baseline
C1: mask-node typed v3 baseline
C2: carrier-tracklet graph core-only
C3: carrier-tracklet graph core + conservative fringe
C4: carrier-tracklet graph core + fringe + ownership WTA
C5: C4 + high-recall Stream3D/Stream4D proposal bank as fringe only
```

注意：C5 如果使用 Stream3D prediction support 做 method selection，必须标为 hybrid/diagnostic；如果只使用同一 2D masks 和 RGB-D geometry，不使用 Stream3D output，则可作为 method。

### 7.7 必须记录的指标

主指标：

```text
AP / AP50 / AP25
#pred
union ratio
union in target ratio
```

object formation：

```text
num_carriers
num_carrier_clusters
carrier_cluster_size_mean / median
cannot_link_edges
positive_edges
weak_bridge_edges
rejected_unions_by_cannot_link
rejected_unions_by_extent
component_split_count
```

support quality：

```text
core_points
fringe_candidate_points
fringe_kept_points
reject_points
ownership_conflict_points
WTA_removed_assignments
mean_mask_area
median_mask_area
large_mask_ratio
tiny_mask_ratio
```

GT diagnostic，仅用于分析：

```text
per_gt_best_iou_mean
per_gt_best_iou_ge_0.25 / 0.50 / 0.75
duplicate_predictions_per_gt
missed_gt_count
```

### 7.8 成功标准

在 probe5 same 32f fixed support 下：

```text
AP   >= 0.32
AP50 >= 0.53
AP25 >= 0.70
#pred <= 300
union_in_target >= 0.94
```

更重要的是，必须缩小与 Stream3D 的差距：

```text
相对 scannet_on_stream4d_32f_probe5：
AP gap 从约 -0.114 缩小到 > -0.06
AP50 gap 从约 -0.093 缩小到 > -0.04
```

如果 C2/C3/C4 没有超过 v6 compact-only，说明 carrier-tracklet graph 实现仍未正确利用 D4RT，需要先可视化 graph，而不是继续调阈值。

### 7.9 可视化要求

每个场景必须输出：

```text
1. carrier tracks overlay：不同 object cluster 不同颜色。
2. same-frame cannot-link pairs overlay。
3. object core/fringe/reject colored mesh。
4. per-object 2D mask support：core carriers + selected connected component。
5. duplicate GT diagnostic：多个 predictions 对同一 GT 的可视化。
6. failure cases：over-merge、under-merge、boundary contamination、lost small object。
```

---

## 8. Phase D：动态 Replica / Replica-Dynamic 实验

### 8.1 目标

ScanNet 是静态 benchmark，不能证明 4D tracking。Dynamic Replica 必须用来验证：

```text
D4RT carrier-tracklet memory 是否在 moving objects、occlusion、reappearance、crossing objects 上优于 Stream3D-style 3D overlap memory。
```

### 8.2 数据检查

先运行：

```bash
python -m tools.check_dynamic_replica_env \
  --root <dynamic_replica_root> \
  --output outputs/audit/dynamic_replica_env_v7.json
```

如果：

```text
usable_scene_count = 0
annotation missing
camera fields missing
```

则不得报告任何 official dynamic metrics。只能报告 blocker。

### 8.3 对比方法

```text
D0: Stream3D-style static 3D overlap memory
D1: current Stream4D carrier-overlap memory
D2: ObjectMemory4D-v2
D3: carrier-tracklet graph + split/merge memory
D4: D3 without D4RT track edges
D5: D3 without cannot-link
```

### 8.4 指标

如果有 official instance tracking labels：

```text
IDF1
MOTA / MOTP
ID switches
fragmentation
track purity
track recall
4D IoU over time
per-object temporal IoU
occlusion reactivation accuracy
moving-object AP/AP50/AP25
static-object AP/AP50/AP25
```

如果有 3D trajectory GT：

```text
trajectory endpoint error
mean 3D track error
occlusion accuracy
visibility accuracy
```

如果没有 semantic GT：

```text
不能报告 semantic AP。
只能报告 class-agnostic tracking 或 qualitative consistency。
```

### 8.5 成功标准

动态主张成立至少需要：

```text
1. D3 相比 D0/D1 IDF1 提升 >= 10 个百分点，或 ID switch 降低 >= 30%。
2. 遮挡重现 case 中 reactivation precision >= 0.70。
3. moving-object 4D IoU 明显高于 Stream3D static-overlap memory。
4. 定性视频中 object ID 不在交叉/遮挡后频繁切换。
```

如果 D3 在动态上不优于 D0/D1，不能 claim “feed-forward semantic 4D reconstruction and tracking” 已达成。

---

## 9. 并行执行计划

为了加速，不要串行等一个大实验。按以下 lane 并行：

### Lane 0：Audit lane

```text
负责人：一个 Codex session
任务：修 WTA score recompute、补 CPU toy tests、生成完整 code audit packet。
输出：v7_code_review_packet、py_compile、unit tests、manifest scanner。
```

### Lane 1：Gap matrix lane

```text
负责人：一个 Codex session
任务：跑 Phase A cross-support matrix。
输出：v7_gap_matrix.csv/json/md + heatmap。
```

### Lane 2：D4RT geometry degradation lane

```text
负责人：一个 Codex session
任务：实现 G1/G3 最小版本，先 3 scenes，再 probe5。
输出：Sim3 residual + segmentation AP。
```

### Lane 3：Carrier-tracklet graph lane

```text
负责人：一个 Codex session
任务：实现 carrier-level co-membership graph，不能复用 mask-node overlap 当 positive_track。
输出：C2/C3/C4 probe5 AP + graph visualizations。
```

### Lane 4：Dynamic Replica lane

```text
负责人：一个 Codex session
任务：只做数据环境和最小 D4RT track sanity。数据缺失时提交 blocker，不造指标。
输出：env checker + sample visualization。
```

---

## 10. 本轮不能写的结论

在 v7 完成前，不能写：

```text
1. Stream4D 已超过 Stream3D。
2. v6 已达成 method gate。
3. D4RT-native geometry 已在 ScanNet 验证。
4. Dynamic Replica tracking 已验证。
5. typed graph 已证明 D4RT tracklet object formation 有效。
6. own recompute 高 AP 等价于完整 4D semantic reconstruction。
```

目前能安全写的是：

```text
1. v6 建立了较好的指标安全 guard。
2. v6 no-group/top-k 能在 #pred<=300 下基本保持 v4.1 水平。
3. 与 Stream3D same-support diagnostic 相比，当前仍差约 9-12 个 AP/AP50 点。
4. D4RT geometry 目前只有高 residual Sim3 diagnostic，尚不能替代 ScanNet RGB-D/pose。
5. 下一步必须转向 carrier-tracklet object formation 和动态 tracking 验证。
```

---

## 11. 最终提交格式

Codex 完成 v7 后必须提交：

```text
1. stream4d_v7_code_review_packet_*.zip
2. docs/stream4d_v7_执行日志.md
3. docs/stream4d_v7_实验结果复盘.md
4. docs/stream4d_v7_gap_matrix.md
5. outputs/audit/v7_metric_integrity_*.json
6. outputs/audit/v7_gap_matrix.{csv,json,md,png}
7. outputs/audit/v7_d4rt_geometry_degradation.{csv,json,md,png}
8. data/evaluation/scannet/*v7*.txt
9. figures/v7_failure_cases/*
```

报告里必须有一个“失败也记录”的表：

```text
实验名 / 目标 / 是否使用 GT / 是否 diagnostic-only / AP / AP50 / AP25 / #pred / union / 结论 / 下一步
```

没有完整 code packet 和 metric integrity 的结果，不允许作为下一轮主结论。
