# Stream4D v5：代码审计、实验复盘判断与并行实验计划书

面向 Codex 的执行文件。本文基于本轮重新解压并审核以下代码和文档后编写：

```text
/mnt/data/stream4d_v4_1_code_review_packet.zip
/mnt/data/Code_Stream3D.zip
/mnt/data/Open-d4rt-main.zip
/mnt/data/粘贴的 markdown (1)。md(417)
```

本文件使用 Typora 友好的公式格式，只使用 `$...$` 和 `$$...$$`，不使用 display 公式的方括号语法。

---

## 0. 执行摘要：这轮到底有没有进展

这轮有进展，但没有达到最终要求。进展不是来自继续调 top-k 或 score，而是来自一个更正确的算法方向：**evidence graph + same-frame cannot-link + component stability + carrier-seeded connected-component densification**。这个方向第一次在 `scene0050_00` 的多窗口 cached-carrier 实验中把 128f 结果推到强于 32f current 的水平：

```text
128f strict component-densify recompute:
AP/AP50/AP25 = 0.490385 / 0.605769 / 0.810897

32f current:
AP/AP50/AP25 = 0.202698 / 0.445714 / 0.681429
```

但这个结果还不能作为 ScanNet 或方法整体胜利。原因很明确：同一 prediction 放回 32f current support、MVP support、adaptive support 或 `scannet_self_inherit` 大 support 后明显下降；Dynamic Replica 没有跑；D4RT-native Sim3 export 没有跑；full ScanNet final/full split 没有完成。

最重要的判断是：**当前方法已经证明“carrier 可以作为语义 mask 的稳定种子”，但还没有证明“Stream4D 已经重建了完整、可稳定评价的 4D semantic field”。**

当前真正卡住的不是某个小 bug，而是 object-level one-to-one assignment。最新同 support 诊断显示，Stream4D 的 AP25 已经可以超过 Stream3D on same 32f support，但 AP 和 AP50 仍低于 Stream3D；GT-only 诊断显示 Stream4D 有更多预测、更多重复、覆盖到的 IoU≥0.5 GT 反而更少。这说明瓶颈已经从“有没有 coverage”转为“是否能把 coverage 归属到一对一的正确 object”。

因此 v5 的核心方向不是继续后处理拼接，而是：

```text
coverage proposal bank
+ evidence graph precision ranker
+ same-frame exclusivity / cannot-link
+ object-level competition and assignment
+ boundary-aware support refinement
+ dynamic tracking evaluation
```

---

## 1. 本轮代码解压和审计范围

### 1.1 解压路径

本轮必须明确已经解压 zip，而不是只读报告。解压路径如下：

```text
/mnt/data/audit_v4_1/Stream3D
/mnt/data/audit_orig_stream3d/Code_Stream3D
/mnt/data/audit_opend4rt/Open-d4rt-main
```

### 1.2 审计文件范围

v4.1 packet 中重点审核以下文件：

```text
Stream3D/stream4d/d4rt_adapter.py
Stream3D/stream4d/carrier_sampler.py
Stream3D/stream4d/mask_evidence.py
Stream3D/stream4d/local_4d_filter.py
Stream3D/stream4d/object_memory.py
Stream3D/stream4d/object_memory_v2.py
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/reexport_scannet.py
Stream3D/stream4d/replay_memory.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/rescore_scannet.py
Stream3D/stream4d/run_scannet.py
Stream3D/tools/verify_stream4d_metric_integrity.py
Stream3D/tools/evaluate_cross_prepoints.py
Stream3D/tools/fuse_prediction_configs.py
Stream3D/tools/complete_prediction_on_support.py
Stream3D/tools/nms_prediction_masks.py
Stream3D/tools/point_nms_prediction.py
Stream3D/tools/rescore_prediction_scores.py
Stream3D/tools/check_dynamic_replica_env.py
Stream3D/tests/test_stream4d_protocol_fixes.py
```

对照审核对象包括：

```text
Code_Stream3D/evaluation/evaluate.py
Open-d4rt-main/infer_track_3d.py
Open-d4rt-main/src/model/d4rt.py
Open-d4rt-main/src/model/query_embedding.py
Open-d4rt-main/src/data/dynamic_replica_raw_dataset.py
```

### 1.3 语法和单元测试

已执行：

```bash
cd /mnt/data/audit_v4_1/Stream3D
python -m py_compile stream4d/*.py tools/*.py tests/*.py
python -m unittest tests.test_stream4d_protocol_fixes -v
```

结果：

```text
Ran 6 tests in 0.002s
OK
```

通过的测试覆盖：

```text
carrier_sampler actual_count 字段长度一致性
fixed_path validation
local_filter carrier weight order
memory-v2 Hungarian one-to-one
reliable_densifier WTA highest reliability owner
rescore object_dict/pred column mismatch detection
```

这只能证明当前包没有语法错误，并且几个已知协议 bug 有回归测试；它不能证明指标已经达标，也不能证明方法已经完成 feed-forward semantic 4D reconstruction and tracking。

---

## 2. 代码审计结论：是否符合计划、是否有 bug、是否有虚假指标

### 2.1 总结论

没有发现直接伪造 AP 或用 GT 生成 prediction 的证据。`rescore_scannet.py`、`reliable_densifier.py`、`evidence_graph.py`、`fuse_prediction_configs.py` 等主要 prediction-generating 工具使用的是 prediction、carrier、mask observation、point support、pre_points 诊断文件等非 GT 信号。当前报告中也反复区分了 `recompute`、`inherit/fixed` 和 `same-support` 诊断，没有把这些混成一个结论。

但是，有三个不能忽视的指标风险：

```text
1. recompute_pre_points 是原版 Stream3D-style evaluator 下可对齐的协议，但它会让 evaluation support 跟随 prediction union 变化。当前很多强结果仍依赖 own observed support。
2. complete_prediction_on_support.py 和 cross-pre_points/fixed-support 相关工具读取 target support，它们只能作为诊断或 support-conditioned 实验，不能作为主方法。
3. v4.1 zip 没有包含 evaluation/evaluate.py，因此本轮不能从这个 zip 重新 hash evaluator；上一轮 v3 审计已验证 evaluator AP core 和原版一致，v5 仍必须在完整工程目录重新运行 metric guard。
```

因此，当前可以说：

```text
没有发现直接虚假指标。
当前结果在报告层面总体诚实。
但如果把 own-recompute 单场景或 final split 结果写成“全面超过 Stream3D”就是过度 claim。
```

### 2.2 与计划文件/implementation 文件的一致性

已经符合的部分：

```text
1. v4.1 已实现 reliable_densifier.py，并包含 seeded component、boundary erosion、seed distance cap、3D NN backprojection、WTA conflict suppression。
2. v4.1 已实现 object_memory_v2.py、appearance_memory.py、motion_memory.py、memory_diagnostics.py、replay_memory.py。
3. v4.1 已实现 evidence_graph.py、replay_evidence_graph.py。
4. v4.1 已实现 cross-pre_points/fixed-support 诊断工具和 metric guard。
5. v4.1 已开始从后处理转向 evidence graph，这是计划中“不要小修小补”的正确方向。
```

仍不符合或未完成的部分：

```text
1. Dynamic Replica / Replica-Dynamic 仍没有真正跑。
2. D4RT-native d4rt_nn / Sim3 export 仍未完成。
3. run_scannet.py 不能完整暴露最新 exporter 参数，例如 export_enable_wta、densify_* 等；最新强结果主要靠 replay_evidence_graph 路径，不是统一主 runner。
4. mask_evidence.py 仍主要是 visibility/confidence + UV 落 mask，没有真实 self/cycle/boundary evidence。
5. ObjectMemory4D-v2 的 appearance 是 RGB histogram，geometry 是 2D centroid proxy，motion 权重基本未发挥，不能写成强 re-ID memory。
6. semantic 4D field 仍没有形成可查询对象；目前是 object_dict + 代表性 2D mask/feature，而不是统一的 language-aligned 4D field。
```

### 2.3 已修复且应保留的代码修复

这些修复是正确的，不要回滚：

```text
local_4d_filter.py: 修复 set 顺序导致 carrier 权重错配。
carrier_sampler.py: 修复 actual_count 字段长度错配。
rescore_scannet.py: 新增 fixed_path 参数校验和 object_dict/pred column alignment check。
reexport/export diagnostic: reuse_point_ids / point_dilate 不再误报 export_nn_hit_rate。
object_memory_v2.py: Hungarian one-to-one matching 有单元测试。
reliable_densifier.py: WTA owner 选择有单元测试。
```

### 2.4 新发现或仍存在的 bug / 风险

#### P0-1：metric guard 扫描范围不够

`tools/verify_stream4d_metric_integrity.py` 主要检查 `stream4d/rescore_scannet.py` 是否读取 GT，但 v4.1 后 prediction-generating 工具已经扩展到很多文件。下一版必须扫描：

```text
stream4d/reexport_scannet.py
stream4d/replay_evidence_graph.py
stream4d/replay_memory.py
tools/fuse_prediction_configs.py
tools/complete_prediction_on_support.py
tools/nms_prediction_masks.py
tools/point_nms_prediction.py
tools/rescore_prediction_scores.py
```

允许读取 GT 的工具只能是：

```text
evaluation/evaluate.py
tools/evaluate_cross_prepoints.py
tools/audit_stream3d_eval_protocol.py
GT-only diagnostic 脚本
```

判断标准：prediction-generating 工具中出现 `gt_path`、`data/scannet/gt`、`read_gt`、`evaluate_matches`、`valid_class_ids`、`semantic gt` 等关键词时必须报 warning，并人工确认用途。

#### P0-2：support-conditioned completion 不能进入主方法

`tools/complete_prediction_on_support.py` 会读取 target support 的 `pre_points`，把未覆盖区域分配给最近 prediction。这不读取 GT，但读取了 evaluation support universe。如果 target support 来自 Stream3D 或其它 config，它就是 support-conditioned diagnostic，不是方法本体。

规则：

```text
任何读取 --target-support-config 或 fixed target pre_points 的 prediction-generating 工具，输出目录必须含 diagnostic 或 support_oracle 字样。
这些结果只能用于分析上限和失败机制，不能进入主表主方法。
```

#### P0-3：v4.1 packet 缺少 evaluator 文件，不能从包内单独完成 evaluator hash 审计

本轮 zip 没有 `Stream3D/evaluation/evaluate.py`。上一轮 v3 审计确认过 evaluator core hash 和原版一致，但 v5 执行时必须在完整工程目录重新运行：

```bash
python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /path/to/orig/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet.txt \
  --configs ...
```

如果完整工程中 evaluator 不存在或 hash 不一致，停止所有新结果报告。

#### P1-1：run_scannet.py 与 replay_evidence_graph.py 的 exporter 参数不统一

`export_scannet.py` 已支持：

```text
export_enable_wta
export_wta_score_mode
export_wta_min_conflict_owners
densify_boundary_erosion
densify_small_mask_area
densify_seed_distance_px
densify_min_seed_pixels
densify_seed_keep_mode
densify_seed_min_support_views
```

`replay_evidence_graph.py` 会把这些参数传给 `ScanNetExporter`。但 `run_scannet.py` 构造 exporter 时没有传递这些参数，parser 也没有完整暴露。因此最新强结果主要来自 replay/cached path，而不是统一 runner path。

Codex 必须修：

```text
1. 把 replay_evidence_graph.py 的 exporter 参数同步到 run_scannet.py。
2. 每个 summary 写明 runner_type = run_scannet 或 replay_evidence_graph。
3. 同一 config 在 cached replay 和 full runner 上至少对 scene0050_00 做一致性检查。
```

判断标准：同一 carriers/object hypotheses 下，run_scannet 和 replay export 的 prediction union、#objects、AP 差异 < 0.01。

#### P1-2：fuse_prediction_configs.py 的 preserve score 在 select_secondary 模式下有潜在负分 bug

`--preserve-primary-score` 和 `--preserve-secondary-score` 会把 score 参数设为 `-1.0`。`concatenate` 模式通过 `_score_array_or_preserve()` 正确处理，但 `select_secondary` 模式中 `_select_variant_masks()` 直接使用 `primary_score` 和 `secondary_score`，可能写出负的 `pred_score`。

Codex 修复：

```text
1. select_secondary 模式下 preserve score 必须真正从 source pred_score 取值。
2. 如果暂时不支持 preserve，则在 fusion_mode=select_secondary 且 preserve_* 为 true 时直接 raise ValueError。
3. 新增单元测试：select_secondary + preserve 不允许产生负 pred_score。
```

#### P1-3：fuse summary 可复现字段不完整

`fuse_prediction_configs.py` summary 应加入：

```text
drop_secondary_overlap_mode
select_min_primary_ioc
select_max_expansion
secondary_score policy
preserve score flags
output_pre_points policy
```

否则后续结果很难判断是 support 策略、score 策略还是 overlap 策略导致。

#### P1-4：evidence_graph 当前仍是 mask-observation graph，不是真正 carrier-tracklet graph

`evidence_graph.py` 已经比 memory-v2 更接近正确方向，但当前 node 是 `(frame_id, mask_id)`，一个 dirty 2D mask 一旦形成 node，内部 carrier 的歧义没有在 node 前被拆开。

下一步不是继续调 graph IoC，而是把 node 从 “mask observation” 降级成更细的：

```text
carrier tracklet cluster within a mask
```

否则 dirty mask 会继续把不同物体 carrier 包在同一 node 里，后面只能补救。

#### P1-5：ObjectMemory4D-v2 不能继续夸大

`object_memory_v2.py` 的初版有工程价值，但算法上不够强：

```text
S_a = RGB histogram，不是 DINO/CLIP。
S_g = 2D centroid proxy，不是真实 3D centroid。
S_m 基本没有发挥，不是 D4RT trajectory continuity。
```

因此文档和论文里不能写：

```text
我们已解决跨窗口 re-ID。
我们使用强 appearance/motion semantic memory。
```

只能写：

```text
实现了一个 memory-v2 prototype，单场景 96f/128f 直接输出仍未达标，后续 evidence graph 替代了它作为主方向。
```

#### P1-6：D4RTAdapter 缺少 adapter-vs-official sanity test

`d4rt_adapter.py` 调用 `encode_video` / `decode_queries`，但没有和 OpenD4RT 官方 `infer_track_3d.py` 做逐点一致性检查。Dynamic Replica 之前必须先做 D4RT sanity，否则 dynamic failure 无法归因。

Codex 必须新增：

```text
tools/compare_d4rt_adapter_with_official.py
```

记录：

```text
uv_diff_mean / p95
xyz_diff_mean / p95
visibility_diff_mean
confidence_diff_mean
same query ordering check
resize/aspect check
```

---

## 3. 独立分析：当前实验数据说明了什么

### 3.1 v3 的本质：top-k 证明了 precision，不证明完整重建

v3 adaptive top-k 在 Stream3D-style recompute 下略超 baseline：

```text
Stream4D adaptive recompute: 20.3718 / 35.5222 / 55.0649
Stream3D-Cropformer:         20.1139 / 34.4654 / 50.2268
```

但 inherit 后掉到：

```text
12.2851 / 23.3147 / 41.6773
```

这说明 top-k 的收益主要来自删掉大量碎片后让 evaluated support 变小、precision 提高。它不是 fake，因为这是原版 Stream3D-style cropped-TMP evaluator 允许的 per-config support；但它不能证明完整 support 下更强。

### 3.2 reliable densification v1 的本质：own support 强，但 coverage universe 仍不稳

v4.1 reliable densification 在 final split own recompute 很强：

```text
Stream4D v4.1 reliable final recompute: 30.2449 / 51.5938 / 67.0619
Stream3D final baseline:                19.4294 / 33.3989 / 49.6361
```

但同一方法放回固定 support：

```text
MVP fixed:      9.0795 AP
adaptive fixed: 13.0900 AP
Stream3D fixed: 0.6823 AP
```

并且 own support 只有约：

```text
union/pre_points ratio = 7.6984%
#objects = 9.99
```

这说明 reliable densification 生成的是很干净的 observed support 子集，而不是完整 3D/4D reconstruction。

### 3.3 memory-v2 的本质：方向合理，但当前信号太弱

memory-v2 96f/128f 没有达到目标：

```text
32f current AP50 = 0.445714
best 96f memory-v2 direct AP50 = 0.298120
best 96f memory-v2 + top40 AP50 = 0.368304
best 128f memory-v2 direct AP50 = 0.291785
```

这说明更多窗口没有自动带来收益。真正问题是：新窗口带来的 support 同时带来 false positives、duplicates、边界噪声和 object fragmentation。RGB histogram + 2D centroid proxy 无法解决同类椅子、相邻物体和多窗口 re-ID。

### 3.4 evidence graph 是正确方向，但仍不是完整胜利

Evidence graph 首次改变 object 形成方式，而不是简单筛 prediction。它在 `scene0050_00` 96f 上出现了真正的算法收益：

```text
96f evidence graph recompute:
AP/AP50/AP25 = 0.159209 / 0.467980 / 0.689963

32f current:
AP/AP50/AP25 = 0.202698 / 0.445714 / 0.681429
```

它 AP 低于 32f，但 AP50/AP25 超过。诊断也显示 same-frame cannot-link 不是摆设：大量 conflict edges 被拒绝。但 conflict rate 仍高达约 0.3183，说明 point-level ownership 仍然混乱。

后续 mask support/WTA/component densify 进一步证明：carrier 适合做种子，不适合直接当最终边界。最佳 128f strict component-densify own recompute 达到：

```text
0.490385 / 0.605769 / 0.810897
```

这是明确进展。但 fixed-support 又失败：

```text
32f current support:        0.135185 / 0.300000 / 0.300000
scannet self-inherit support: 0.008383 / 0.031532 / 0.072072
```

### 3.5 最新同-support 结果揭示了最深层瓶颈

最新 tiered/containment 方向让 `scene0050_00` 在 32f fixed support 上明显强于 32f current，并且 AP25 已经超过 Stream3D on same support：

```text
new best on 32f support:
AP/AP50/AP25 = 0.253098 / 0.615832 / 0.802912

Stream3D on same 32f support:
AP/AP50/AP25 = 0.391132 / 0.646154 / 0.761538
```

这非常关键：AP25 已经不是主要问题；AP 和 AP50 仍低。GT-only diagnostic 显示：

```text
new best 有更多 IoU≥0.5 prediction，但覆盖到的 GT 更少。
new best 对同一 GT 有接近 2 个重复 prediction。
Stream3D 在同一 support 下更接近一对一。
```

因此，问题本质已经不是“没 coverage”，也不是“score 没调好”，而是：

```text
Stream4D 当前 object hypothesis 没有形成一对一实例解释。
它更像 object proposal bank，而不是最终 instance segmentation / semantic 4D object field。
```

这就是 v5 必须解决的核心。

---

## 4. 当前是否在正确道路上，离目标有多远

### 4.1 正确的部分

当前已从错误路线转向正确路线：

```text
错误路线：top-k、score sweep、point NMS、简单 fusion、简单 seed-all。
正确路线：evidence graph、same-frame cannot-link、component stability、carrier-seeded connected component、strict view quality gate。
```

这和最初方法动机一致：D4RT 的价值不是给 depth，而是提供 carrier/correspondence。实验也证明 carrier 不是最终 mask，但它是定位 2D mask 中稳定 object component 的好种子。

### 4.2 仍然很远的部分

离最终目标还差四个关键环节：

```text
1. ScanNet full/final 多场景验证：目前很多强结果仍是 scene0050_00。
2. Fixed/inherit support 稳健性：大 support 下仍失败。
3. 动态场景验证：Replica-Dynamic/Dynamic Replica 还没跑。
4. D4RT-native geometry：Sim3 evaluation adapter 还没实现，当前仍是 rgbd_eval bridge。
```

若论文目标是“feed-forward semantic 4D reconstruction and tracking”，目前只完成了静态 ScanNet 上的 object-formation prototype，还没有证明 tracking。

### 4.3 为什么进度感觉慢

进度慢不是因为 Codex 没做事，而是因为前半段在优化错误目标：

```text
own recompute AP 提高很快，但它鼓励 high-precision small support。
fixed/inherit 支持失败后，又花了很多轮做 fusion/rescue/NMS，这些都只是后验补救。
真正改变 object 形成机制的 evidence graph 到较后面才开始。
```

此外，实验粒度也不理想：大量时间花在 `scene0050_00` 上，导致很难判断泛化；动态数据一直没跑，导致主 claim 没有被验证。

### 4.4 是否应该继续

应该继续，但必须换推进方式：

```text
不再继续盲调 scene0050 后处理。
不再把 own-recompute 单场景作为主要成功信号。
下一轮必须并行推进：metric guard、多场景 generalization、object-level assignment、dynamic data sanity、D4RT Sim3。
```

---

## 5. v5 总体目标和核心假设

### 5.1 总体目标

v5 的目标不是再把 `scene0050_00` 的 recompute AP 做高，而是验证以下命题：

```text
D4RT carrier evidence can produce temporally consistent object hypotheses;
when combined with object-level competition and support-aware assignment,
Stream4D can outperform Stream3D in settings where dynamic correspondence matters,
while remaining at least competitive on static ScanNet.
```

在 ScanNet 上，v5 目标是：

```text
1. 在 fixed/same-support 或 locked final split 下证明不只是 support shrink。
2. 在多个 scenes 上证明 evidence graph/component densify 不只是 scene0050 过拟合。
3. 把 AP/AP50 的瓶颈从 duplicate/fragmented proposals 转化为更一对一的 object assignment。
```

在 Dynamic Replica 上，v5 目标是：

```text
1. 验证 D4RT carrier 的 track sanity。
2. 验证 Stream4D 在 moving object identity 上优于 Stream3D-style static overlap。
3. 如果 GT 允许，报告 IDF1/IDSW/Frag/4D IoU/APD3D/AJ/OA；如果 GT 不允许，只报告 pseudo/qualitative，禁止伪造 official metric。
```

### 5.2 核心假设

#### H1：当前主要瓶颈是 object-level one-to-one assignment，不是 coverage

证据：AP25 已超过 Stream3D same-support，但 AP/AP50 仍低；new best 有更多 predictions，却覆盖更少 IoU≥0.5 GT，并且重复更多。

目标：把 proposal bank 变成更一对一的 object set。

#### H2：carrier 应该做 seed / evidence，不应该直接做 final support

证据：carrier support AP25 高但 AP 低；maskbp core AP/AP50 高但 AP25 低；simple fusion 失败。

目标：在一个 object 内维护 core/fringe/discard，而不是输出多套 hypotheses。

#### H3：same-frame cannot-link 和 object competition 是关键负证据

当前多数 matching 是正证：overlap、appearance、mask_count。v5 必须强化反证：同帧不同 mask、互斥 2D footprint、过度 containment、边界冲突、同一 support 被多个 object 解释。

#### H4：动态优势不能从 ScanNet 推出，必须跑 Dynamic Replica

ScanNet 是静态场景。即使 ScanNet comparable，也不能证明 4D tracking。Dynamic Replica 是主 claim 的必要实验。

#### H5：D4RT-native geometry 的 Sim3 只能是 evaluation adapter

D4RT geometry 与 ScanNet/Replica coordinate 对齐需要 scale/Sim3，但 Sim3 不允许进入 grouping、selection、memory。它只能用于评估导出和几何误差诊断。

---

## 6. 并行推进总计划

为加速实验，v5 必须并行，而不是串行等待一个方向完全成功。

### Lane A：指标与代码守卫

目标：确保没有 fake metric、support leakage、runner/replay 不一致。

负责人：一个 Codex 线程即可。

输出：`outputs/audit/stream4d_v5_metric_integrity.{md,json}`。

### Lane B：ScanNet object assignment 小场景算法

目标：在 5 个场景上验证 object-level competition 是否解决 duplicate/one-to-one 问题。

建议场景：

```text
scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00
```

输出：每个场景的 recompute、32f fixed、MVP fixed、Stream3D same-support 比较。

### Lane C：ScanNet tune/final 泛化

目标：把 Lane B 成功的 locked config 跑 `scannet_tune30`、`scannet_tune`、`scannet_final`，避免 scene0050 过拟合。

输出：final split 表格和 failure cases。

### Lane D：Dynamic Replica / Replica-Dynamic

目标：先跑数据检查和 D4RT sanity，再跑 object identity tracking。

输出：dynamic data inventory、D4RT track sanity、tracking metrics 或 pseudo qualitative。

### Lane E：D4RT-native Sim3 evaluation adapter

目标：独立实现 Sim3 export，不阻塞 ScanNet semantic algorithm。

输出：Sim3 residual、scale drift、D4RT geometry vs rgbd_eval bridge gap。

并行原则：

```text
Lane A 必须最先通过。
Lane B/C 可以用 cached carriers 并行。
Lane D 不等待 ScanNet 成功。
Lane E 不等待 Dynamic Replica 成功。
任何 Lane 不允许读取 GT 生成 prediction。
```

---

## 7. Lane A：指标、代码和协议守卫

### 7.1 目标

确认当前代码没有虚假指标，并把所有 support-conditioned / diagnostic 工具从主方法工具中隔离出来。

### 7.2 具体任务

#### A1：扩展 metric guard

修改：

```text
tools/verify_stream4d_metric_integrity.py
```

新增扫描范围：

```text
stream4d/*.py
tools/*.py
```

但要区分：

```text
prediction_generating_tools
metric_or_diagnostic_tools
```

输出字段：

```text
evaluator_hash_equal
prediction_tools_gt_read_flags
diagnostic_tools_gt_read_flags
support_conditioned_tools_detected
pre_points_policy_by_config
object_dict_pred_alignment_mean_iou
negative_pred_score_detected
prediction_shape_mismatch_detected
```

判断标准：

```text
prediction-generating 工具 GT-read flags 必须为 False。
所有 support-conditioned 工具输出必须含 diagnostic/support_oracle 标记。
所有 pred_score 必须 finite 且 >= 0。
```

失败尝试方向：

```text
若发现 GT read，先确认是否为 diagnostic；若不是，停止该工具输出。
若发现 negative pred_score，优先修 fuse_prediction_configs.py。
若 evaluator hash 不一致，停止所有 AP 报告。
```

#### A2：修复 run_scannet / replay 参数不一致

把 `replay_evidence_graph.py` 已支持的 exporter 参数同步到 `run_scannet.py`。

新增 CLI：

```text
--export-enable-wta
--export-wta-score-mode
--export-wta-min-conflict-owners
--densify-boundary-erosion
--densify-small-mask-area
--densify-seed-distance-px
--densify-min-seed-pixels
--densify-seed-keep-mode
--densify-seed-min-support-views
```

记录：

```text
runner_type
exporter_args_hash
carrier_cache_hash
object_hypothesis_hash
```

判断标准：

```text
同一 scene0050 cached object 输入下，run_scannet 和 replay export 输出 prediction union、#objects、AP/AP50/AP25 差异 < 0.01。
```

#### A3：修复 fuse preserve score bug

新增测试：

```text
test_fuse_select_secondary_preserve_score_does_not_write_negative_scores
```

判断标准：

```text
所有 fusion 输出 pred_score.min() >= 0。
select_secondary preserve score 要么正确 preserve，要么显式禁止。
```

#### A4：support-conditioned 工具命名保护

修改：

```text
tools/complete_prediction_on_support.py
tools/evaluate_cross_prepoints.py
tools/fuse_prediction_configs.py
```

规则：

```text
如果读取 target pre_points 生成 prediction，output_config 必须包含 diagnostic/support_oracle/support_conditioned。
如果 output_config 不含这些标记，直接 raise ValueError。
```

---

## 8. Lane B：ScanNet 小场景 object-level assignment 算法

### 8.1 总目标

在固定小场景集合上，验证 v5 的核心假设：当前瓶颈是 object-level one-to-one assignment，而不是 coverage。实验要从“更多 proposal”变成“更少重复、更一对一、更边界正确”的 object set。

### 8.2 方法方向：Object Competition Graph

构建一个 proposal-level competition graph。proposal 来源可以包括：

```text
1. evidence graph strict objects
2. component_densify high precision core
3. 32f/current inherited coverage tier
4. wider support recall candidates
```

但输出不再简单拼接，而是通过 competition 选择或分配。

每个 object candidate $o_i$ 有：

```text
core support C_i
fringe support F_i
mask observation set E_i
carrier set T_i
frame support V_i
boundary confidence B_i
component stability S_i
same-frame mask ids M_i
```

定义无监督质量：

$$
Q(o_i)=w_s S_i+w_t \log(1+|V_i|)+w_c \log(1+|T_i|)+w_b B_i-w_x X_i-w_d D_i
$$

其中：

```text
X_i: conflict penalty，包括 point ownership conflict、same-frame overlap conflict。
D_i: duplicate/containment penalty。
```

两个 candidates 的竞争边：

$$
A_{ij}=\alpha \cdot IoC(o_i,o_j)+\beta \cdot SameFrameConflict(o_i,o_j)+\gamma \cdot Containment(o_i,o_j)
$$

输出策略不是 NMS，而是 object assignment：

```text
1. 若两个 candidates 高度解释同一 support，保留高 Q 的 core，低 Q 只允许贡献未覆盖 fringe。
2. 若两个 candidates 同帧互斥且重叠高，强制 cannot-link。
3. 若一个 candidate 被另一个高置信 candidate 包含，删除或只保留不重叠补漏部分。
4. 若一个 low-confidence candidate 只补充已存在 object 的边界，不作为新 object 输出。
```

### 8.3 B1：scene0050 同 support object competition

#### 假设

$H_{B1}$：在 `scene0050_00` 的 32f fixed support 上，object competition 可以减少 duplicate prediction，使 AP/AP50 接近或超过 Stream3D on same support，同时保持 AP25 优势。

#### 对比方法

```text
B1-0 32f current
B1-1 original Stream3D scannet on same 32f support
B1-2 latest tiered/containment best
B1-3 object competition graph v1
B1-4 object competition graph v1 without containment penalty
B1-5 object competition graph v1 without same-frame exclusivity
B1-6 object competition graph v1 without fringe reassignment
```

#### 主指标

```text
AP / AP50 / AP25 on own recompute
AP / AP50 / AP25 on 32f fixed support
AP / AP50 / AP25 on scannet_self_inherit support, diagnostic only
```

#### 必须记录的诊断指标

```text
#pred
prediction union in target support
nonempty prediction count
duplicate_object_rate
containment_suppressed_count
same_frame_conflict_edges
cannot_link_edges
fringe_reassigned_points
point_ownership_conflict_before/after
mean predictions per target component, unsupervised proxy
mean support overlap among kept objects
runtime
```

GT-only diagnostic 只能在 prediction 完成后运行，单独输出到 `outputs/diagnostic_gt_only/`：

```text
#GT covered at IoU>=0.25 / 0.50
#pred with IoU>=0.25 / 0.50
mean predictions per matched GT
best IoU distribution
missed GT panel
merged GT panel
```

GT-only diagnostic 不能用于选择阈值。阈值只在 non-GT proxy 上选择。

#### 判断标准

最低成功：

```text
在 scene0050 32f fixed support 上：
AP50 >= 0.646154 - 0.02
AP >= 0.30
AP25 >= 0.80
重复率比 latest tiered/containment best 降低 30%。
```

强成功：

```text
AP/AP50/AP25 同时超过 Stream3D on same 32f support：
AP > 0.391132
AP50 > 0.646154
AP25 > 0.761538
```

如果不满足：

```text
AP25 高但 AP/AP50 低：加强 boundary-aware core，降低 fringe 进入高分 object。
AP50 高但 AP 低：检查高 IoU 阈值下边界和 duplicate；增加 object-level exclusivity。
#pred 很多：增加 containment suppression，不做简单 top-N。
覆盖到的 support 下降：允许 low-confidence recall tier，但不能作为独立高分 object。
```

#### 可视化

保存：

```text
1. candidate graph 可视化：node size=area，edge color=conflict/containment/same-frame。
2. before/after prediction mesh overlay。
3. 每个 GT-only diagnostic 的 duplicate case panel。
4. core/fringe/discard 三色图。
5. same-frame cannot-link mask overlay。
6. containment suppression 前后 object pair 对比。
```

### 8.4 B2：五场景泛化验证

#### 假设

$H_{B2}$：Object competition 的收益不是 scene0050 过拟合，在不同类型场景上也能减少 duplicate 和 AP/AP50 差距。

#### 场景

```text
scene0050_00: 已知强调参场景，不能单独决定结论。
scene0011_00: 正常室内场景。
scene0030_00: 中等复杂度。
scene0081_01: 高 object 数/碎片化风险。
scene0591_00: object explosion case。
```

#### 对比

```text
32f current
Stream3D on same support
latest tiered/containment best
object competition v1
object competition v1 locked from scene0050
```

#### 指标

```text
per-scene AP/AP50/AP25
mean AP/AP50/AP25
std AP/AP50/AP25
#pred
union in target
duplicate proxy
same-frame conflict rate
point ownership conflict rate
runtime
```

判断标准：

```text
5 scenes mean AP50 比 latest tiered/containment best 提升 >= +0.03。
5 scenes 中至少 4 个 scene AP50 不下降超过 0.02。
#pred 不超过 latest best 的 1.2x。
duplicate proxy 下降 >= 20%。
```

失败处理：

```text
如果 scene0591 object explosion：先启用 per-frame proposal cap 和 same-frame exclusivity。
如果小物体漏掉：按 area bucket 使用不同 containment threshold。
如果只 scene0050 有收益：冻结当前算法，停止 full ScanNet，回到 evidence construction。
```

### 8.5 B3：carrier-tracklet node 替代 mask-observation node

#### 假设

$H_{B3}$：把 evidence graph node 从 `(frame_id, mask_id)` 改成 mask 内 carrier-tracklet cluster，可以在 dirty 2D mask 内分离不同物体，降低 over-merge。

#### 方法

对每个 2D mask observation 内部，根据 carrier 的跨帧 co-membership 建小图：

$$
w(c_i,c_j)=\lambda_1 CoMask(c_i,c_j)+\lambda_2 TrajSim(c_i,c_j)+\lambda_3 GeoSim(c_i,c_j)-\lambda_4 Conflict(c_i,c_j)
$$

其中：

```text
CoMask: 多帧落在同一 mask/component 的次数。
TrajSim: D4RT 2D/3D trajectory similarity。
GeoSim: RGB-D bridge 下 3D centroid/normal/compactness。
Conflict: 同帧被不同 mask 强支持。
```

第一版只做轻量：

```text
同一 mask 内按 carrier target UV 的 connected component / distance / co-mask count cluster。
```

#### 指标

```text
node_count
mean carriers per node
split_dirty_mask_count
AP/AP50/AP25
conflict edge count
overmerge proxy
runtime
```

判断标准：

```text
AP50 提升 >= +0.02 或 duplicate proxy 下降 >= 20%。
如果 node_count 暴涨超过 3x 且 AP 不升，回退。
```

---

## 9. Lane C：ScanNet tune/final 泛化实验

### 9.1 目标

把 Lane B 中成功的 object assignment 算法从小场景扩展到 ScanNet split，证明不是 scene0050 特化。

### 9.2 数据 split

固定使用：

```text
splits/scannet_tune30.txt
splits/scannet_tune.txt
splits/scannet_final.txt
```

规则：

```text
1. scene0050 只用于开发，不用于最终调参判断。
2. tune30 用于快速筛选。
3. tune 用于锁定参数。
4. final 只跑一次，不再根据 final 调参。
```

### 9.3 C1：tune30 快速泛化

#### 假设

$H_{C1}$：object competition 在 tune30 上能提升 fixed/inherit 稳定性，而不是只提升 own recompute。

#### 对比

```text
Stream3D baseline
v3 adaptive recompute/inherit
v4.1 reliable no-seed recompute/fixed
latest tiered/containment
v5 object competition
v5 object competition ablations
```

#### 主表字段

```text
Method | support policy | AP | AP50 | AP25 | union % | target union % | #pred | GT crop/full | conflict | duplicate proxy | runtime
```

#### 判断标准

```text
own recompute AP 不低于 v4.1 reliable no-seed - 2 AP。
MVP/adaptive fixed AP 比 latest tiered/containment 提升 >= +2 AP。
Stream3D fixed 不再灾难性低，至少提升到 > 5 AP。
#pred 不暴涨超过 v4.1 reliable 的 2x。
```

如果不满足：

```text
fixed 仍低：说明 proposal bank 覆盖仍不足，不要继续调 competition，回到 support bank construction。
own recompute 掉太多：说明 competition 删除了高精 core，降低 containment suppression 强度。
#pred 暴涨：提高 duplicate penalty 和 same-frame exclusivity。
```

### 9.4 C2：locked tune/final

#### 假设

$H_{C2}$：locked v5 config 在 final split 上比 v3/v4.1 更稳健，不依赖支持域缩小。

#### 运行

```text
1. 在 tune 上选择一个 config。
2. 保存 configs/stream4d_v5_locked.json。
3. 在 final 上只运行 locked config。
4. 不允许 final 后再修改阈值。
```

#### 成功标准

最低论文可用标准：

```text
final recompute AP >= Stream3D final AP + 0.5
final AP50 >= Stream3D final AP50 + 1.0
final inherit/fixed AP 比 v4.1 reliable fixed 提升 >= +3.0
union % 不低于 v4.1 reliable own support
GT crop/full 不进一步下降
```

强标准：

```text
final same-support AP/AP50/AP25 中至少 AP50/AP25 超过 Stream3D on same support。
或者 Dynamic Replica 大幅超过 Stream3D-style static overlap，ScanNet 只写 comparable。
```

### 9.5 C3：failure case taxonomy

每个 final 失败 scene 必须自动归类：

```text
low D4RT visibility
low carrier assignment
mask evidence too sparse
object duplicate
over-merge
under-merge
support too small
boundary leakage
Sim3/export failure
```

每类至少保存 5 个可视化 case。

---

## 10. Lane D：Replica-Dynamic / Dynamic Replica 实验

### 10.1 总目标

动态实验是方法主线的关键，不应再拖延。ScanNet 只能说明静态 3D instance segmentation；Dynamic Replica 才能回答 Stream4D 是否真正支持 semantic 4D reconstruction and tracking。

### 10.2 D0：数据环境检查

运行：

```bash
python -m tools.check_dynamic_replica_env \
  --root /path/to/dynamic-replica-or-replica-dynamic \
  --output outputs/dynamic_replica/env_check_v5.json
```

必须检查：

```text
frame_annotations_valid.json
images/*.png
depths/*.geometric.png
trajectories/*.pth
camera R/T/focal/principal point
instance masks / object IDs / semantic labels 是否存在
split: train/valid/test
```

判断：

```text
如果有 GT object IDs 和 masks：可做 official-like object tracking。
如果只有 trajectories：只做 point tracking / grouping consistency，不报 semantic AP。
如果只有 RGB/depth/camera：只做 pseudo 和 qualitative，不报 official metric。
```

### 10.3 D1：D4RT adapter sanity

#### 假设

$H_{D1}$：当前 D4RTAdapter 与 OpenD4RT 官方 inference 在相同 video/query 上输出一致。

#### 实验

从 valid 选 10 个视频，每个 48 frames。采样：

```text
random visible points
mask boundary points
moving object points, 若可得
background points
```

对比：

```text
current D4RTAdapter
OpenD4RT infer_track_3d.py official path
```

指标：

```text
uv_diff_mean / p95
xyz_diff_mean / p95
visibility_diff_mean
confidence_diff_mean
uv_in01_rate
visibility_prob_mean
confidence_prob_mean
track_length_visible
runtime_per_query
```

若有 GT trajectories：

```text
APD3D
AJ
OA
3D EPE
trajectory L1
```

判断标准：

```text
adapter-vs-official uv_diff_p95 < 1 px normalized equivalent or documented tolerance。
visibility/confidence ordering 一致。
若不一致，先修 adapter，不跑 semantic tracking。
```

失败方向：

```text
检查 frame indexing。
检查 RGB normalization/channel order。
检查 resize/aspect ratio。
检查 t_src/t_tgt/t_cam order。
检查 confidence/visibility 是否需要 sigmoid。
```

### 10.4 D2：动态 object identity tracking

#### 假设

$H_{D2}$：在动态物体上，D4RT carrier-based Stream4D 比 Stream3D-style static overlap 更少 ID switch 和 fragmentation。

#### 方法对比

```text
D2-0 per-frame 2D masks no tracking
D2-1 Stream3D-style static overlap adapted baseline
D2-2 Stream4D carrier-overlap memory old
D2-3 ObjectMemory4D-v2
D2-4 evidence graph + object competition
D2-5 evidence graph without motion evidence
D2-6 evidence graph without appearance evidence
```

#### 若有 GT object IDs/masks，记录

```text
IDF1
IDSW
Frag
MOTA/MOTP 或 HOTA, 若实现成本可控
4D IoU over time
track purity
track coverage
reactivation success
occlusion gap recovery
per-object lifespan accuracy
```

#### 若只有 trajectories，记录

```text
carrier-track purity
tracklet cluster consistency
moving/static separation quality
same physical point object label consistency
APD3D/AJ/OA for D4RT only
```

#### 若无 GT，只记录 pseudo/qualitative

```text
temporal ID consistency
mask-to-track consistency
object color timeline
manual selected cases
open-vocab query video
```

禁止：

```text
无 GT object ID 时不能报 official IDF1。
无 semantic GT 时不能报 semantic AP。
```

#### 判断标准

官方 GT 条件下：

```text
IDF1 比 Stream3D-style overlap 提升 >= +10%。
IDSW 降低 >= 30%。
Frag 降低 >= 30%。
4D IoU 提升 >= +5%。
至少 5 个 occlusion/reactivation case 可视化成功。
```

Pseudo 条件下：

```text
object ID timeline 少于 baseline 的明显跳变。
同一 carrier 的 object label consistency 提升 >= 20%。
至少 10 个视频 qualitative 成功，且列出失败 cases。
```

### 10.5 D3：time-sensitive semantic query

#### 目标

验证不是只做 3D instance mask，而是能支持 time-sensitive semantic 4D query。

Query 类型：

```text
moving object
object before motion
object after motion
same object across time
object that disappears and reappears
person/object interaction if present
```

指标：

```text
query-to-track consistency
query localization temporal IoU, if GT available
frame interval accuracy
object ID consistency under query
qualitative video panels
```

失败方向：

```text
如果 VLM/CLIP hallucinate，不把语言当真值，只作为 candidate semantic evidence。
如果 semantic 不稳，先做 class-agnostic dynamic tracking，再加 language layer。
```

---

## 11. Lane E：D4RT-native Sim3 evaluation adapter

### 11.1 目标

当前 ScanNet 结果仍主要是 `rgbd_eval` bridge。v5 必须独立实现 D4RT-native geometry evaluation adapter，至少作为 diagnostic，避免论文被质疑“你只是用 D4RT carrier + ScanNet GT geometry bridge”。

### 11.2 方法

D4RT 输出点为 $x_i^{D4RT}$，evaluation coordinate anchor 为 $x_i^{eval}$。评估时估计：

$$
T^*=(s,R,t)=\arg\min_{s,R,t}\sum_i \omega_i\|sRx_i^{D4RT}+t-x_i^{eval}\|_2^2
$$

约束：

```text
Sim3 只能用于 evaluation/export adapter。
不能反馈到 object grouping、evidence graph、selection、memory。
不能使用 GT instance/semantic label 选择 anchors。
```

### 11.3 实验

```text
E0 rgbd_eval bridge
E1 D4RT xyz + scale-only alignment
E2 D4RT xyz + scene-level Sim3
E3 D4RT xyz + window-level Sim3 + overlap stitching
```

### 11.4 指标

```text
AP/AP50/AP25 under d4rt_sim3_eval
AP/AP50/AP25 under rgbd_eval bridge
scale factor
Sim3 residual median/p90
anchor_count
inlier_ratio
window scale drift
mesh NN hit rate
points outside mesh radius
D4RT depth AbsRel after scale, if depth GT available
```

### 11.5 判断标准

```text
median Sim3 residual < 0.10m on ScanNet diagnostic。
D4RT Sim3 AP50 与 rgbd_eval AP50 gap < 20% relative，才考虑主路径。
如果 gap 大，只作为 geometry diagnostic，不作为主 claim。
```

失败方向：

```text
检查 coordinate convention。
检查 t_cam reference。
检查 frame index。
检查 D4RT local vs world coordinate。
尝试 high-confidence static anchors。
尝试 window-level Sim3。
可视化 residual heatmap。
```

---

## 12. 每次实验必须记录的统一指标

### 12.1 主 AP 指标

每个方法至少报告：

```text
AP / AP50 / AP25 recompute_pre_points
AP / AP50 / AP25 inherit_pre_points
AP / AP50 / AP25 fixed_pre_points, if applicable
```

每张表必须写 support policy，不能只写 AP。

### 12.2 Support 和 coverage 指标

```text
pre_points_ratio
prediction_union_ratio
union_in_target_support_scene_ratio
union_in_target_support_target_ratio
GT crop/full, if GT diagnostic allowed
#pred
#nonempty_pred
points_per_object_mean/median
small/medium/large object count
```

### 12.3 Object quality 指标

无 GT 主指标：

```text
same_frame_conflict_rate
point_ownership_conflict_rate
containment_suppression_count
duplicate_proxy_by_support_overlap
component_stability
carrier_support_count
mask_observation_count
boundary_confidence
fringe_ratio
core_ratio
```

GT-only diagnostic：

```text
#GT covered at IoU thresholds
#pred matched at IoU thresholds
mean predictions per matched GT
best IoU histogram
fragmentation_per_gt
merge_error_per_pred
```

### 12.4 Dynamic tracking 指标

```text
IDF1
IDSW
Frag
track purity
track coverage
reactivation count
occlusion gap recovery
4D IoU
APD3D
AJ
OA
3D EPE
trajectory L1
```

如果 GT 不足，明确标注 pseudo，不得伪装 official。

### 12.5 Runtime 和资源

```text
D4RT encode seconds
D4RT decode seconds
carrier count
query count
export seconds
memory update seconds
peak GPU memory
per-frame runtime
per-window runtime
```

---

## 13. 每次实验必须保存的可视化

### 13.1 ScanNet 可视化

```text
prediction vs GT mesh overlay, GT-only diagnostic
prediction union heatmap
missed GT instances, GT-only diagnostic
false positives / duplicate object panels, GT-only diagnostic
carrier seeds overlay on RGB
mask connected component kept/discarded overlay
core/fringe/discard 3-color overlay
object competition graph visualization
same-frame cannot-link edges
point ownership conflict before/after
per-object IoU histogram, GT-only diagnostic
```

### 13.2 Dynamic 可视化

```text
2D tracks overlay video
3D trajectory video
object ID timeline
ID switch frames
occlusion reactivation examples
moving/static object separation
time-sensitive query result videos
failure cases with reason tags
```

---

## 14. v5 最终表格要求

### 14.1 ScanNet 主表

```text
Method | Runner | 2D masks | D4RT ckpt | frames | memory/object formation | support method | eval policy | AP | AP50 | AP25 | union % | target union % | #pred | conflict | runtime
```

必须包含：

```text
Stream3D-Cropformer baseline
Stream4D v3 adaptive
Stream4D v4.1 reliable densification
Stream4D v4.1 evidence graph best, scene-level only if not full
Stream4D v5 object competition
Stream4D v5 object competition + component densify
```

每个 Stream4D 方法至少两行：

```text
own recompute
fixed/inherit diagnostic
```

### 14.2 Same-support 诊断表

```text
Method | target support | AP | AP50 | AP25 | union in target | #pred | duplicate proxy | comments
```

必须有：

```text
Stream3D on Stream4D 32f support
32f current
latest v4.1/v5
```

### 14.3 Dynamic Replica 表

如果 GT 充分：

```text
Method | IDF1 | IDSW | Frag | 4D IoU | APD3D | AJ | OA | EPE | Reactivation | Runtime
```

如果 GT 不充分：

```text
Method | Pseudo track consistency | Label consistency | Reactivation qualitative | Failure count | Runtime
```

表名必须含 `pseudo`。

### 14.4 D4RT Sim3 表

```text
Export | AP | AP50 | AP25 | scale | residual median | residual p90 | inlier ratio | NN hit | notes
```

---

## 15. Claim 安全边界

### 15.1 可以安全写的条件

只有满足以下条件，才可以写：

```text
Stream4D outperforms Stream3D on ScanNet.
```

条件：

```text
1. locked final split，不是 scene0050 或 tune-only。
2. Stream3D-style evaluator 与 baseline 对齐。
3. recompute AP 至少 +0.5，AP50 至少 +1.0。
4. fixed/inherit AP 不再灾难性低，并比 v4.1 fixed 明显提升。
5. 不靠进一步缩小 support 换 AP。
6. metric integrity audit 通过。
```

若 ScanNet 只 comparable，但 Dynamic Replica 明显更好，应写：

```text
Stream4D is comparable on static ScanNet but substantially improves dynamic object identity and semantic 4D tracking.
```

### 15.2 禁止 claim

```text
禁止写 v4.1/v5 已全面超过 Stream3D，除非 full/final/same-support 证据满足标准。
禁止隐藏 own-recompute 与 fixed/inherit 的差异。
禁止把 support-conditioned completion 写成主方法。
禁止在没有 GT object IDs 的 Dynamic Replica 上报 official IDF1。
禁止在没有 semantic GT 时写 semantic AP。
禁止写当前结果是 D4RT-native geometry，除非 d4rt_sim3_eval 跑通并报告 residual。
禁止让 Sim3 进入 method grouping/selection/memory。
```

---

## 16. Codex 下一批具体任务清单

### Task 1：metric guard v5

```text
扩展 tools/verify_stream4d_metric_integrity.py
扫描所有 prediction-generating tools
检查 negative scores
检查 support-conditioned output naming
输出 stream4d_v5_metric_integrity.md/json
```

### Task 2：统一 runner/replay exporter 参数

```text
同步 run_scannet.py 与 replay_evidence_graph.py 的 exporter 参数
新增 runner_type/exporter_args_hash
scene0050 做 replay/run consistency check
```

### Task 3：修 fuse preserve score bug

```text
select_secondary + preserve score 正确处理或禁止
新增单元测试
```

### Task 4：Object Competition Graph v1

新增：

```text
stream4d/object_competition.py
tools/replay_object_competition.py
```

实现：

```text
candidate import from evidence graph/component densify/coverage bank
containment suppression
same-frame exclusivity
fringe reassignment
object quality diagnostics
```

### Task 5：五场景验证

```text
生成 scene0050/0011/0030/0081/0591 的 report
每个 scene 输出 own/fixed/same-support 指标和可视化
```

### Task 6：Dynamic Replica D0/D1

```text
运行 check_dynamic_replica_env.py
实现 compare_d4rt_adapter_with_official.py
跑 10 个短视频 D4RT sanity
根据 GT 可用性决定 official/pseudo metrics
```

### Task 7：D4RT Sim3 export diagnostic

```text
实现 export_scannet_d4rt_sim3.py 或 export_d4rt_nn()
只做 evaluation adapter
记录 residual/scale/inlier/AP gap
```

---

## 17. 每轮实验复盘模板

每次 Codex 跑完必须写：

```text
docs/stream4d_v5_{phase}_实验复盘.md
```

模板：

```text
1. 本轮目标
2. 假设
3. 修改了哪些代码
4. 运行命令
5. 完整性检查
6. 主指标
7. support/coverage 诊断
8. object quality 诊断
9. GT-only diagnostic, if any, clearly separated
10. 可视化路径
11. 假设是否成立
12. 失败原因
13. 下一步尝试方向
14. 哪些结论不能写
```

不允许只有 AP 表格。必须包含：

```text
support policy
union ratio
#pred
conflict/duplicate proxy
runner type
cached/full runner 标记
GT-only diagnostic 标记
```

---

## 18. 最后的判断

当前实验并非完全没进展。真正有价值的进展是：

```text
evidence graph 证明了 carrier-based object formation 比 memory-v2 和后处理更有希望；
component densify 证明 carrier 适合作为 object component seed；
strict relative coverage gate 证明多窗口收益需要视角质量过滤；
同 support 诊断证明 AP25/coverage 已不再是唯一瓶颈，真正瓶颈是一对一实例归属。
```

但当前离目标还差很远。下一步如果还继续在 `scene0050_00` 上调 fusion、score、NMS，就会继续慢。v5 必须把问题改成 object assignment / object competition，并且并行推进 Dynamic Replica 和 D4RT Sim3。只有这样，Stream4D 才可能从“在 observed-support 上高分的 3D proposal selector”变成“training-free feed-forward semantic 4D reconstruction and tracking”方法。
