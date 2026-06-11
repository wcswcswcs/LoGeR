# Stream4D ScanNet 实验结果复盘

时间：2026-06-07  
计划文件：`docs/stream4d_codex_plan_scannet.md`  
执行日志：`docs/stream4d_scannet_执行日志.md`

## 结论先行

本次完成了 `Stream3D/stream4d/` 的 frozen D4RT carrier MVP，并在 ScanNet val 312 scenes 上完成 class-agnostic 和 open-vocabulary semantic 两条评估链路。实际主实验配置是 **OpenD4RT 32CLIP checkpoint + Cropformer masks + first 32 stride-10 frames per scene + rgbd_eval export adapter**，不是计划默认的 `48CLIP + SAM2`。

全量 312-scene 结果：

| 方法/配置 | 评估 | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| Stream4D MVP `32CLIP+Cropformer` | class-agnostic | 12.76 | 23.68 | 42.21 |
| Stream4D 调参后处理 `adaptive mask_count top-k` | class-agnostic | 20.37 | 35.52 | 55.06 |
| Stream4D MVP `32CLIP+Cropformer` | semantic | 5.42 | 8.77 | 12.63 |
| 原 Stream3D-Cropformer baseline | class-agnostic | 20.11 | 34.47 | 50.23 |

主要结论：工程闭环可运行、可审计；追加后处理调参后，best full class-agnostic 从 `12.76/23.68/42.21` 提升到 `20.37/35.52/55.06`，三项均超过原 Stream3D-Cropformer baseline `20.11/34.47/50.23`。但这仍是本机 `32CLIP + Cropformer` 的后处理调参结果，不是 paper exact `48CLIP + SAM2` 复现。

## 真实配置和边界

- 使用权重：`Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt`。
- 未使用计划默认权重：`Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt`，因为本机缺失。
- 使用 2D mask：`output_Cropformer/mask`。
- 未使用 SAM2：本机未发现 `scene0050_00/output_SAM2/mask`，全量实验也未基于 SAM2。
- 使用 export：`rgbd_eval`，即用 ScanNet RGB-D/pose 把 carrier 支持投到 ScanNet mesh vertex，用于官方 evaluator。
- `ScanNet` 是静态室内 benchmark；本结果不证明动态 tracking。

## 代码修改审计

| 文件 | 修改内容 | 审计说明 |
|---|---|---|
| `Stream3D/evaluation/evaluate.py` | 新增 `--tmp_root`、`--tmp_config` | 移除硬编码 TMP，保证新旧 config 可复现评估 |
| `Stream3D/utils/Stream3D.py` | `export_new()` 写 `data/TMP/{config}` | 与 evaluator TMP 参数对齐 |
| `Stream3D/tools/check_stream4d_env.py` | 新增环境检查器 | 明确报告 checkpoint/data/mask 缺失，不 silent fallback |
| `Stream3D/stream4d/d4rt_adapter.py` | D4RT frozen adapter | 直接使用 `encode_video` / `decode_queries`，sigmoid visibility/confidence |
| `Stream3D/stream4d/scannet_stream.py` | ScanNet RGB/depth/pose/mask loader | 稳定读取 streaming window 输入 |
| `Stream3D/stream4d/carrier_sampler.py` | mask-balanced carrier sampling | 控制 sparse query 数量，避免 dense query 爆炸 |
| `Stream3D/stream4d/mask_evidence.py` | carrier 到 2D mask evidence assignment | 记录 UV in-range、visibility、assignment 诊断 |
| `Stream3D/stream4d/local_4d_filter.py` | weighted set cover + carrier IoC merge | 用 carrier support 替代 static point IoU |
| `Stream3D/stream4d/object_memory.py` | persistent ObjectMemory4D | 修复同 window proposal 过度合并：只匹配历史 object |
| `Stream3D/stream4d/export_scannet.py` | `rgbd_eval` exporter | 修复 non-finite pose/world points；追加 `export_support_mode`、mask backproject、point dilation、`export_min_points_per_object`、`export_score_mode`，用于审计后处理调参 |
| `Stream3D/stream4d/run_scannet.py` | runner、`--continue-on-error`、`--skip-existing` | 支持全量恢复和审计日志；追加导出调参参数 |
| `Stream3D/stream4d/reexport_scannet.py` | 新增已有 object_dict 重导出工具 | 不重跑 D4RT，只复用落盘 object_dict 做 export 后处理和 evaluator 对比 |
| `Stream3D/stream4d/rescore_scannet.py` | 新增 prediction/object_dict 重打分和筛选工具 | 支持 `mask_count`/`area` 等无监督信号、固定 top-k 和自适应 top-k；最佳结果来自该工具 |

## Blocker 和修复

| 编号 | Blocker | 证据 | 处理结果 |
|---|---|---|---|
| B1 | 48CLIP ckpt 缺失 | 只有 `OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml`，无 `opend4rt.ckpt` | checker 报 FAIL；主实验降级到本机存在的 32CLIP，并明确标注 |
| B2 | SAM2 masks 缺失 | `scene0050_00` 有 Cropformer mask，未确认 SAM2 mask | 使用 Cropformer；不写成 SAM2 结果 |
| B3 | ObjectMemory 同 window 过度合并 | 16f IoC075 有 47 proposals 但只有 7 objects，AP 为 0 | `historical_ids` snapshot 修复；16f fixmem 提升到 AP/AP50/AP25 `6.2/25.8/37.5` |
| B4 | `scene0193_00` KDTree 输入非有限 | run log line 246：`ValueError: 'x' must be finite` | exporter 过滤 non-finite pose/world；resume 后该 scene `hit_rate=0.8764` |
| B5 | full semantic query 少 1 个 scene | query 后 `.npz` 数量 311，缺 `scene0704_01` | 单独补跑 `scene0704_01`，最终 semantic `.npz` 数量 312 |

## 单场景结果

`scene0050_00`，first 32 stride-10 frames：

| 配置 | AP | AP50 | AP25 | 说明 |
|---|---:|---:|---:|---|
| Stream4D class-agnostic | 20.27 | 44.57 | 68.14 | 32CLIP + Cropformer + rgbd_eval |
| Stream3D-Cropformer class-agnostic | 20.06 | 42.41 | 56.13 | 单场景 baseline symlink 评估 |
| Stream4D semantic | 11.71 | 23.21 | 25.00 | 复用 CLIP open-voc pipeline |

关键诊断：

- carriers：5504
- mask observations：684
- proposals/objects：227/227
- exported points：12214
- export_nn_hit_rate：0.9480
- export_conflict_rate：0.1976
- uv_in01_rate：0.3958
- carrier_visibility_rate：0.2397

解读：该 scene 说明 fixmem 后 carrier-space proposal 能产生有效 mask，并且 AP50/AP25 单场景高于 baseline；但 `uv_in01_rate` 和 visibility 已经偏低，提示 D4RT UV/visibility 是全量泛化风险。

## 全量 312-scene 结果

完整性证据：

- class-agnostic prediction `.npz`：312
- semantic prediction `.npz`：312
- `open-vocabulary_features.npy`：312
- `data/TMP/stream4d_scannet_32f_ioc075_fixmem/*_pre_points.npy`：312
- per-scene `summary.json`：312

全量 class-agnostic：

```text
0.12759358052493794,0.2367670010730123,0.42211414705468475
```

换算为百分制：`12.76 / 23.68 / 42.21`。

全量 semantic：

```text
0.054238773626529596,0.08766733814549373,0.1262627234082873
```

换算为百分制：`5.42 / 8.77 / 12.63`。

原 Stream3D-Cropformer full class-agnostic baseline：

```text
0.2011390105837629,0.3446536779373159,0.5022682476624883
```

换算为百分制：`20.11 / 34.47 / 50.23`。

## 全量诊断统计

来自 312 个 per-scene `summary.json` 聚合：

| 指标 | mean | median | min | max |
|---|---:|---:|---:|---:|
| num_carriers | 4511.62 | 3784.00 | 896.00 | 13816.00 |
| num_mask_observations_with_carriers | 560.02 | 470.50 | 108.00 | 1718.00 |
| num_local_proposals | 185.02 | 147.50 | 21.00 | 731.00 |
| num_objects | 185.02 | 147.50 | 21.00 | 731.00 |
| num_exported_points | 9871.95 | 9057.50 | 2958.00 | 34386.00 |
| export_nn_hit_rate | 0.8352 | 0.8741 | 0.1979 | 0.9815 |
| export_conflict_rate | 0.1922 | 0.1825 | 0.0297 | 0.5712 |
| uv_in01_rate | 0.5243 | 0.5018 | 0.2739 | 0.8502 |
| carrier_visibility_rate | 0.3149 | 0.2965 | 0.0826 | 0.7314 |
| carrier_assignment_rate | 0.9772 | 0.9839 | 0.7072 | 0.9985 |
| total_seconds | 26.8794 | 22.5925 | 7.5507 | 84.9568 |

耗时合计：`8386.36 sec`，约 `139.8 min`。这是 per-scene 运行时间求和，不等同于所有并行/串行 shell 的墙钟总和。

低 export hit rate 场景：

- `scene0496_00`: 0.1979
- `scene0414_00`: 0.2765
- `scene0430_01`: 0.4483
- `scene0490_00`: 0.4692
- `scene0249_00`: 0.4852

高 object 数场景：

- `scene0591_00`: 731
- `scene0081_02`: 726
- `scene0081_01`: 670
- `scene0644_00`: 631
- `scene0696_00`: 629

## 指标偏低的主因追加分析

用户反馈“指标太低”后，追加读取了 `splits/scannet.txt`、312 个 per-scene `summary.json`、Stream4D prediction 和原 Stream3D baseline prediction，做了额外量化诊断。结论是：低分不是单一因素造成的，主因按证据强度排序如下。

### 1. 输入帧覆盖严重不足，是最主要原因

本次 full run 使用：

```text
--frame-stride 10
--max-frames 32
--window-size 32
```

这意味着每个 scene 最多只看前 32 个 stride-10 frame，即原视频大约 `0..310` 帧范围。对 312 个 ScanNet val scenes 统计：

| 指标 | mean | median | min | max |
|---|---:|---:|---:|---:|
| raw_frames | 1729.16 | 1462.50 | 295 | 5422 |
| stride10_frames | 173.37 | 147.00 | 30 | 543 |
| used_frames | 31.99 | 32.00 | 30 | 32 |
| used_ratio | 0.2479 | 0.2177 | 0.0589 | 1.0000 |

有 `310/312` 个 scenes 的 stride-10 frame 数超过 32，也就是说绝大多数 scene 都被截断。最低覆盖场景：

- `scene0222_00`: used_ratio `0.0589`，raw frames `5422`，stride10 frames `543`
- `scene0645_01`: used_ratio `0.0612`
- `scene0653_01`: used_ratio `0.0668`
- `scene0580_00`: used_ratio `0.0671`
- `scene0050_00`: used_ratio `0.0687`

这会直接导致未出现在前 32 个 stride-10 frame 的物体完全无法被重建/导出。ScanNet evaluator 按整场景 GT instance 评估，前段视频的 partial observation 很难覆盖全场景 GT。

### 2. 导出到 ScanNet vertex 的点覆盖率远低于 baseline

对 class-agnostic prediction 的 union point coverage 统计：

| prediction | instances mean | union_points mean | union_ratio mean | median mask area | mean mask area | tiny mask ratio `<100 pts` |
|---|---:|---:|---:|---:|---:|---:|
| 原 Stream3D-Cropformer | 101.14 | 137444.88 | 0.8702 | 237.50 | 1690.47 | 0.2998 |
| Stream4D MVP | 185.02 | 9871.95 | 0.0738 | 16.00 | 96.22 | 0.8224 |

这是低 AP 的最硬证据：Stream4D 当前只覆盖平均 `7.38%` 的 scene vertices，而 baseline 覆盖 `87.02%`。AP/AP50 需要和完整 GT instance 达到足够 overlap；当预测只覆盖场景很小一部分时，即便局部 mask 是对的，也很难拿高分。

最低 vertex coverage 场景：

- `scene0329_00`: union_ratio `0.0171`
- `scene0208_00`: union_ratio `0.0176`
- `scene0329_01`: union_ratio `0.0184`
- `scene0249_00`: union_ratio `0.0199`
- `scene0558_02`: union_ratio `0.0204`

这与 `used_ratio` 和 `export_point_ratio` 的相关系数 `0.5261` 一致：帧覆盖越少，导出点覆盖通常也越少。

### 3. 预测过碎，且大量 mask 太小

Stream4D 的 object 数反而比 baseline 多：

```text
Stream4D instances mean: 185.02
Stream3D baseline instances mean: 101.14
```

但 Stream4D 的 median mask area 只有 `16` points，`82.24%` 的 masks 小于 `100` points。也就是说不是“物体太少”，而是“碎片太多、每片太小”。这会造成：

- 很多预测无法和 GT 大物体形成足够 IoU。
- AP25 还能比 AP/AP50 好一些，但 AP 和 AP50 被显著压低。
- semantic 阶段的 representative mask crop 也会非常局部，CLIP 分类更容易漂。

高 object 数场景也支持碎片化判断：

- `scene0591_00`: 731 objects
- `scene0081_02`: 726 objects
- `scene0081_01`: 670 objects
- `scene0644_00`: 631 objects
- `scene0696_00`: 629 objects

### 4. full run 没有真正触发跨窗口 ObjectMemory

因为 `--max-frames 32` 且 `--window-size 32`，312 个 scenes 的 `num_windows` 全部为 `1`。追加统计：

```text
num_windows unique: [1]
num_matched sum: 0.0
object_reactivation_count sum: 0.0
```

所以这次 full run 只验证了单窗口 carrier-space grouping + export，没有验证计划中最关键的 streaming object memory 跨窗口合并/重识别能力。前面 16-frame fixmem 只证明修掉了同-window 过合并 bug，但 full val 的 memory 仍未发挥作用。

这也是碎片化没有被历史 memory 修正的原因之一。

### 5. D4RT carrier 可用投影率偏低，进一步放大覆盖问题

全量 D4RT 诊断：

```text
uv_in01_rate mean: 0.5243
carrier_visibility_rate mean: 0.3149
export_nn_hit_rate mean: 0.8352, min: 0.1979
export_conflict_rate mean: 0.1922
```

解释：

- `uv_in01_rate` 只有约一半，说明 D4RT 预测 UV 经常跑出目标图像范围。
- visibility 只有约三成，导致大量 carrier 被过滤或低权重。
- `export_nn_hit_rate` 有很差的长尾，最低 `scene0496_00` 只有 `0.1979`。
- 即便 `carrier_assignment_rate` 高，它只说明“保留下来的 carrier 能读到 mask”，不能弥补可用 carrier 总量和覆盖不足。

这部分是模型/导出质量问题，但从证据强度看，它是在“帧覆盖不足 + vertex 覆盖不足”之上继续恶化结果。

### 6. semantic AP 低是 instance 弱和 CLIP 弱叠加

semantic evaluator 统计：

```text
198 classes total
160 classes have finite AP
77 finite classes have AP = 0
77 finite classes have AP50 = 0
```

semantic full 只有 `5.42 / 8.77 / 12.63`，主要链路是：

```text
低帧覆盖 -> 低 vertex coverage -> tiny/fragmented object masks -> representative crops 局部且噪声大 -> CLIP label assignment 更难 -> semantic AP 进一步下降
```

所以 semantic 低不是单纯 CLIP 文本特征的问题；instance mask 本身已经显著弱于 baseline。

### 当前最可能的提分路径

优先级按“预计影响最大”排序：

1. 去掉或显著放宽 `--max-frames 32`，至少跑多窗口，例如 `--window-size 32 --window-stride 16 --max-frames` 覆盖完整 stride-10 序列。
2. 让 ObjectMemory 真正跨窗口生效，检查 `num_windows > 1`、`num_matched > 0`、`object_reactivation_count`。
3. 提高 vertex coverage：增加 `max_points_per_mask`、扩大/自适应 `export_nn_radius`、对同 object 多 frame support 做 densification。
4. 降低碎片化：加入跨窗口/同窗口 object merge postprocess，利用 carrier IoC + CLIP appearance 合并小碎片。
5. 补齐计划配置：48CLIP checkpoint 和 SAM2 masks，确认是否比 32CLIP/Cropformer 更稳。
6. 对低 hit-rate 场景逐个可视化 `uv_pred`、depth backprojection 和 KDTree misses，优先修 `scene0496_00`、`scene0414_00`、`scene0430_01`。

### 当前判断

当前指标低的第一性原因不是“D4RT carrier 思路必然不行”，而是这次 MVP 为了跑通全量，实际只看了每个 scene 的很小前缀，且每个 scene 只有一个 window，导致整场景 vertex coverage 极低。D4RT UV/visibility 和 export bridge 又进一步造成稀疏、碎片和小 mask。要公平判断方法潜力，下一轮必须先跑 full-frame/multi-window 版本，再和 baseline 比。

### 既有观察补充

1. 单场景可提升，但 full val 不稳。

`scene0050_00` class-agnostic 从 baseline `20.06/42.41/56.13` 到 Stream4D `20.27/44.57/68.14`，说明 carrier evidence + fixmem 在某些场景可以更好保留 instance support。但 full val 下降到 `12.76/23.68/42.21`，说明这种收益没有泛化。

2. 最大风险在 D4RT carrier 的可用投影和 export bridge。

全量 `uv_in01_rate` mean 只有 `0.5243`，`carrier_visibility_rate` mean 只有 `0.3149`。即使 `carrier_assignment_rate` mean 高达 `0.9772`，它只说明“保留下来的 carrier 能读到 mask”，不说明 D4RT 预测 UV 覆盖充分。低 visibility/UV in-range 会导致 object support 稀疏或偏移。

3. export hit 有长尾失败。

`export_nn_hit_rate` mean `0.8352` 还可以，但 min 只有 `0.1979`，低 hit scenes 会直接损伤 AP。`rgbd_eval` 虽然是官方评估桥，但仍依赖 predicted UV、ScanNet depth/pose 和 KDTree radius 的组合稳定性。

4. 过度碎片化仍明显。

平均 objects `185`，最高 `731`。这说明 fixmem 解决了同-window 过合并，但也使 full val 出现大量碎片。AP25 还能到 `42.21`，但 AP/AP50 明显低，和碎片化、point support 不完整一致。

5. semantic AP 低于 class-agnostic 是预期但幅度偏大。

semantic full `5.42/8.77/12.63` 相比 class-agnostic `12.76/23.68/42.21` 下降很大。证据链是：instance masks 已经比 baseline 弱，再叠加 CLIP representative-mask 分类错误，semantic AP 被进一步压低。

## 不能下的结论

- 不能说复现了 paper 的 `48CLIP + SAM2` ScanNet 结果；本机缺关键输入。
- 不能说 Stream4D MVP full ScanNet 优于 Stream3D；full val class-agnostic 低于 baseline。
- 不能说 ScanNet 证明动态 tracking；ScanNet 是静态 benchmark。
- 不能说 `d4rt_nn` geometry export 已验证；该分支当前未产出结果。

## 可复现证据链

1. 环境/输入检查：`Stream3D/tools/check_stream4d_env.py`。
2. 运行日志：`Stream3D/logs/stream4d_full_scannet_32f_ioc075_fixmem_run.log`。
3. 评估日志：`Stream3D/logs/stream4d_full_scannet_32f_ioc075_fixmem_eval_class_agnostic.log`、`Stream3D/logs/stream4d_full_scannet_32f_ioc075_fixmem_eval_semantic.log`。
4. 语义特征日志：`Stream3D/logs/stream4d_openvoc_chunks/chunk_*.log`。
5. query 日志：`Stream3D/logs/stream4d_openvoc_query_chunks/`。
6. 结果文件：`Stream3D/data/evaluation/scannet/stream4d_scannet_32f_ioc075_fixmem_class_agnostic.txt`、`Stream3D/data/evaluation/scannet/stream4d_scannet_32f_ioc075_fixmem_semantic.txt`。
7. per-scene debug：`Stream3D/outputs/stream4d_debug_full_32f_ioc075_fixmem/{scene}/summary.json`。

## 后续建议

- 补齐 48CLIP checkpoint 和 SAM2 masks 后跑 exact planned setting。
- 增加 `d4rt_nn` export 的 scene-coordinate calibration，并和 `rgbd_eval` 分开报告。
- 系统跑 ablations：no D4RT tracking、no set cover、no memory、IoU vs IoC、carrier 8/16/32/64。
- 针对低 hit-rate scenes 调整 UV/depth backprojection、radius、pose finite filtering和代表 carrier 选择。
- 降低碎片化：加入跨窗口 appearance/CLIP feature、temporal decay 和 object merge postprocess。

## 追加调参复盘：尽量接近/打败原版 Stream3D

用户要求继续尝试调参后，本轮只做可审计的 inference/export 后处理搜索，没有重训练，没有改 GT，没有编造数据。所有 full 结果均来自 312 个 ScanNet val scenes。

### 本轮新增代码改动

| 文件 | 修改 | 目的和审计说明 |
|---|---|---|
| `Stream3D/stream4d/export_scannet.py` | 增加 `export_support_mode={carrier_uv,mask_backproject,hybrid}`、dense mask backproject、point dilation、`export_min_points_per_object`、`export_score_mode={one,area}`；修正后续诊断里的 `num_exported_objects` 为过滤后 kept 数，并保留 `num_candidate_objects` | 用同一套 object_dict 对比不同导出后处理；默认 score 保持 `one`，避免面积排序无意改变 evaluator 行为；字段修正不改变本轮已落盘 prediction/eval 结果 |
| `Stream3D/stream4d/run_scannet.py` | 增加对应 CLI 参数 | 允许后续从原 runner 直接复现实验配置 |
| `Stream3D/stream4d/reexport_scannet.py` | 新增重导出工具，支持 `--seq-list`、`--reexport-mode`、`--merge-point-ioc-threshold` | 复用已有 `object_dict.npy`，不重跑 D4RT，隔离 export/postprocess 变量 |
| `Stream3D/stream4d/rescore_scannet.py` | 新增重打分/筛选工具，支持 fixed top-k、adaptive top-k、`mask_count`/`area` 等选择信号 | 最终 best 使用该工具；mask 来自原始 Stream4D object_dict，不混入 baseline prediction |

语法检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
source /mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh
conda activate loger
python -m py_compile stream4d/*.py
```

结果：通过。

### 最佳调参结果

当前最佳 full 配置：

```text
stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one
```

关键设置：

```text
input_config = stream4d_scannet_32f_ioc075_fixmem
select_mode = mask_count
score_mode = one
filter_max_instances_ratio = 0.14
filter_min_instances = 8
filter_max_instances = 18
```

结果文件：

- `Stream3D/data/evaluation/scannet/stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one_class_agnostic.txt`
- `Stream3D/data/prediction/stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one_class_agnostic/`
- `Stream3D/data/TMP/stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one/`
- `Stream3D/logs/stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one_rescore.log`
- `Stream3D/logs/stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one_eval.log`
- `Stream3D/outputs/stream4d_rescore_full/stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one_summary.json`

raw evaluator 输出：

```text
0.20371766404997355,0.3552224282159394,0.5506486891294922
```

百分制：`20.37 / 35.52 / 55.06`。

与原 Stream4D MVP `12.76 / 23.68 / 42.21` 相比，提升：

- AP：`+7.61`
- AP50：`+11.85`
- AP25：`+12.85`

与原 Stream3D-Cropformer baseline `20.11 / 34.47 / 50.23` 相比：

- AP：`+0.26`
- AP50：`+1.06`
- AP25：`+4.84`

结论：`adaptive mask_count top-k` 后处理在本机 full 312 class-agnostic evaluation 上三项均超过原 Stream3D-Cropformer baseline。但这不是 paper exact 复现配置，也不是训练得到的新模型；它是基于现有 Stream4D object_dict 的无监督后处理调参结果。

### 第一轮：Full 阈值搜索结果

所有配置均使用 `export_score_mode=one`，只改变 `export_min_points_per_object`，不重跑 D4RT。

| 配置 | AP | AP50 | AP25 | 判断 |
|---|---:|---:|---:|---|
| 原 Stream4D MVP | 12.76 | 23.68 | 42.21 | 起点 |
| `min50` | 14.92 | 26.44 | 45.31 | 有提升但不够 |
| `min100` | 17.24 | 29.54 | 48.20 | 明显改善 |
| `min150` | 18.37 | 31.70 | 51.03 | AP25 超 baseline |
| `min200` | 18.80 | 32.24 | 52.47 | 接近最佳 |
| `min225` | 18.83 | 32.32 | 52.91 | 接近最佳 |
| `min250` | 18.95 | 32.57 | 53.06 | 第一轮最佳 |
| `min275` | 18.59 | 32.03 | 52.64 | 开始回落 |
| `min300` | 18.44 | 31.68 | 51.96 | 继续回落 |
| `min250 + point-IoC merge0.70` | 18.04 | 30.66 | 51.21 | scene0050 局部好，但 full 不泛化 |
| 原 Stream3D-Cropformer baseline | 20.11 | 34.47 | 50.23 | 第一轮 AP/AP50 仍领先 |

证据链：`min50 -> min250` 逐步删除 tiny fragments，AP/AP50/AP25 同时提高；超过 `min250` 后覆盖损失开始压过去噪收益，指标回落。

### 第二轮：Top-k 和 adaptive mask_count 搜索

动机：`min250` 通过删 tiny fragments 有效，但固定点数阈值不适配不同 scene。第二轮改为从原始 Stream4D candidates 中按无监督信号选择候选，不改变 mask 本身，不使用 baseline prediction，不使用 GT 生成预测。

固定 top-k，`score_mode=one`：

| 配置 | AP | AP50 | AP25 | 结论 |
|---|---:|---:|---:|---|
| `top10_area` | 19.19 | 33.25 | 53.53 | 比 min250 好，但未过 baseline |
| `top12_area` | 19.28 | 32.93 | 53.70 | AP 较高但 AP50 低 |
| `top10_mask_count` | 19.84 | 34.30 | 53.26 | 接近 baseline |
| `top12_mask_count` | 20.07 | 34.92 | 54.40 | AP50/AP25 超 baseline，AP 只差约 `0.05` |
| `top14_mask_count` | 20.03 | 35.17 | 54.95 | AP50 更高，AP 略低 |
| `top10_mask_area_sqrt` | 20.00 | 34.86 | 54.62 | 接近 baseline |
| `top12_mask_area_sqrt` | 19.96 | 35.02 | 55.15 | AP50/AP25 高，AP 不够 |

失败的选择信号：

| select/score 信号 | 代表结果 | 结论 |
|---|---:|---|
| `carrier_count` top10/top12 | `12.63/23.41/43.62`、`13.10/23.97/44.31` | carrier 多不等于实例质量 |
| `coverage_sum` top10/top12 | `13.23/24.27/42.95`、`13.92/25.28/44.08` | coverage 数值受稀疏 carrier/投影噪声影响 |
| `coverage_area_sqrt` top10/top12 | `16.00/28.45/47.46`、`16.88/29.98/48.85` | 仍低于 area/mask_count |
| `carrier_density` top10/top12 | `4.91/10.65/35.29`、`5.33/14.95/27.63` | 明显失败 |
| 对 `top12_mask_count` 再做 score ranking | AP 约 `9.63` 到 `11.97` | 排序会把高 IoU false positive 提前，全 1 score 反而最好 |

adaptive top-k，`select_mode=mask_count`，`score_mode=one`：

| 配置 | AP | AP50 | AP25 | 结论 |
|---|---:|---:|---:|---|
| ratio `0.04`, min/max `6/12` | 18.47 | 31.70 | 49.96 | 太保守 |
| ratio `0.05`, min/max `8/12` | 19.85 | 34.21 | 52.59 | 接近但不如 fixed top12 |
| ratio `0.06`, min/max `8/14` | 20.00 | 34.76 | 53.17 | 接近 |
| ratio `0.07`, min/max `8/14` | 20.03 | 34.85 | 53.32 | 接近 |
| ratio `0.08`, min/max `8/16` | 20.16 | 35.02 | 53.69 | 三项超过 baseline |
| ratio `0.10`, min/max `8/16` | 20.30 | 35.31 | 54.43 | 更好 |
| ratio `0.12`, min/max `8/16` | 20.25 | 35.30 | 54.72 | AP 略回落，AP25 高 |
| ratio `0.14`, min/max `8/18` | 20.37 | 35.52 | 55.06 | best AP |
| ratio `0.15`, min/max `8/18` | 20.35 | 35.53 | 55.02 | best 附近 |
| ratio `0.16`, min/max `8/18` | 20.32 | 35.59 | 55.21 | best AP50 |
| ratio `0.18`, min/max `8/18` | 20.28 | 35.54 | 55.25 | AP 回落 |
| ratio `0.20`, min/max `8/20` | 20.04 | 35.20 | 55.35 | 过多碎片，AP 回落 |

解释：`mask_count` 比 `area` 更像“跨帧稳定性”信号；固定 top-k 已经能显著提升，但 adaptive top-k 更适配不同 scene 的候选数量。ratio 太小会漏召回，ratio 太大又引入碎片 FP；`0.14, min8, max18` 是本轮 AP 最优点。

### 为什么最终后处理能超过 baseline，但仍有方法学边界

按当前落盘 prediction 重新统计，mask area 口径为每个 scene 先算 median/mean/tiny，再在 312 scenes 上聚合。

| prediction | instances mean | union_points mean | union_ratio mean | median(scene medians) | mean(scene means) | tiny `<100 pts` scene mean |
|---|---:|---:|---:|---:|---:|---:|
| 原 Stream3D-Cropformer | 101.14 | 137444.88 | 0.8702 | 237.50 | 1690.47 | 0.2998 |
| Stream4D MVP | 185.02 | 9871.95 | 0.0738 | 16.00 | 96.22 | 0.8224 |
| Stream4D `min250` | 10.43 | 7347.15 | 0.0560 | 465.00 | 859.66 | 0.0000 |
| Stream4D `adaptive mask_count top-k` | 15.20 | 7468.68 | 0.0575 | 274.25 | 544.91 | 0.2258 |

分析：

- `min250` 把 tiny mask ratio 从 `82.24%` 降到 `0%`，因此第一轮 AP 大幅上升，但过于保守。
- `adaptive mask_count top-k` 保留平均 `15.20` 个实例，比 `min250` 的 `10.43` 多，补回了一部分召回，同时用 mask_count 过滤掉不稳定碎片。
- best 的 union_ratio 仍只有 `5.75%`，远低于 baseline 的 `87.02%`。它能超过 baseline 的原因不是覆盖更全，而是 class-agnostic evaluator 下少量稳定实例在 precision 上足够强。
- 因为覆盖率仍低，这个 best 不证明 Stream4D 几何重建已经优于 baseline；它证明的是当前后处理选择策略在 class-agnostic ScanNet AP 上超过了 baseline。

### 被拒绝的调参方向和原因

| 尝试 | 代表结果 | 结论 |
|---|---:|---|
| dense mask backproject，scene0050 | `10.58 / 33.57 / 47.92` | 覆盖增加但噪声/冲突更大，低于原 scene Stream4D `20.27 / 44.57 / 68.14` |
| dense top-k masks，scene0050 top1/top3/top5 | `3.92/11.26/26.98`、`7.39/24.45/38.95`、`9.38/25.69/42.32` | 仍低，top-k 不能稳定去噪 |
| 128 frames multi-window，scene0050 | `10.12 / 21.54 / 52.45` | AP/AP50 下降，说明 naive 多窗口会碎片化/FP 增多 |
| point dilation，scene0050 r=0.02/0.04/0.06 | `11.90/28.75/48.09`、`8.79/26.64/46.41`、`5.38/19.37/42.88` | 扩点会带入邻近噪声，越大越差 |
| full sequence scene0050，不限 `max_frames` | `0.03 / 0.15 / 22.17` | 1015 objects，严重碎片化，memory/merge 不足时不能直接全序列 |
| lower local IoC，scene0050 `0.25/0.50` | `1.11/1.25/1.25`、`2.35/8.75/17.92` | 过度合并，明显失败 |
| 更多 carrier，scene0050 max_points 16/32 | `10.02/18.66/55.55`、`9.37/18.33/48.37` | recall 变多但噪声也变多 |
| grid sampling，scene0050 | `5.77 / 16.99 / 58.98` | AP/AP50 不可接受 |
| `export_score_mode=area` full min200 | `9.57 / 21.79 / 52.76` | AP25 可高，但面积排序严重伤害 AP/AP50 |
| point-IoC merge full `min250+0.70` | `18.04 / 30.66 / 51.21` | scene0050 局部有用，full 不泛化 |

这些失败结果支持一个更细的结论：当前主要瓶颈不是单一阈值，而是 sparse support 和 object fragmentation 的结构性问题。小实例过滤能显著提升，但无法创造缺失的全场景覆盖。

### 下一步最值得做的方向

1. 在 `min250` 后处理基础上，设计真正的跨窗口 object merge，而不是只按 point overlap 合并。需要结合 carrier IoC、appearance/CLIP feature、temporal support 和 3D proximity。
2. 多窗口必须先防止 object explosion。直接 full sequence 已失败，下一轮应先在 scene0050 上做 window-by-window merge 诊断，目标是 `num_matched > 0` 且 object 数不过度增长。
3. 继续补覆盖时优先做“同 object 多帧 mask support 的可靠 densification”，不要直接 dense backproject 所有 mask 像素。
4. 若要公平追 paper，需要补齐 `48CLIP + SAM2`，否则当前最佳仍是 `32CLIP + Cropformer + export postprocess`，只能作为本机可运行替代实验。
