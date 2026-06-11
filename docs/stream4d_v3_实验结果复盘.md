# Stream4D v3 实验结果复盘

开始时间：2026-06-07 21:44:25 +08  
计划文件：`docs/stream4d_v3_protocol_corrected_plan_for_codex.md`  
执行日志：`docs/stream4d_v3_执行日志.md`

## 总结论

本轮完成了 v3 计划中的协议修正、S0 evaluator audit、S1 `pre_points` policy 对照，以及一个 locked split 诊断。结论必须分开写：

```text
在 Stream3D-style cropped-TMP evaluator 的 recompute_pre_points 协议下：
  Stream4D adaptive mask-count top-k = 20.3718 / 35.5222 / 55.0649
  Stream3D-Cropformer baseline       = 20.1139 / 34.4654 / 50.2268
  该协议下 AP/AP50/AP25 均超过 baseline。

在 inherit_pre_points 协议下：
  Stream4D adaptive mask-count top-k = 12.2851 / 23.3147 / 41.6773
  明显低于 Stream3D-Cropformer baseline，也略低于 Stream4D MVP。
```

因此不能写“Stream4D 已经全面优于 Stream3D”。更准确的结论是：

```text
当前方法在 observed-support / recompute cropped-TMP universe 中可超过 Stream3D-Cropformer；
但这个优势不稳定，主要依赖缩小 evaluation universe。
要让结果更强，下一步不能继续盲调 top-k ratio，必须提升 support coverage：
  reliable densification
  memory-v2
  更可靠的跨窗口 object support
```

## 当前必须审计的核心问题

上一轮 best：

```text
stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one
AP / AP50 / AP25 = 20.37 / 35.52 / 55.06
```

这个结果使用的是 `rescore_scannet.py` 的 recompute pre_points 行为：

```text
pre_points = pred_masks.any(axis=1)
```

v3 计划要求新增并比较：

- `recompute`：当前行为，评价点集随 top-k 后 prediction union 缩小。
- `inherit`：沿用输入配置 `stream4d_scannet_32f_ioc075_fixmem` 的 pre_points，不随 top-k 缩小。
- `fixed_path`：从指定 TMP root/config 读取 pre_points，作为统一诊断。

只有跑完这些实验后，才能判断当前 best 是否主要依赖缩小 evaluation universe。

## 已做修复与可审计性

### rescore pre_points policy

修改文件：

```text
Stream3D/stream4d/rescore_scannet.py
```

做了什么：

- 新增 `--pre-points-policy recompute|inherit|fixed_path`。
- `recompute`：输出 pre_points = 当前预测 mask 的 union。
- `inherit`：输出 pre_points = 输入配置的 pre_points。
- `fixed_path`：从指定 TMP root/config 读取 pre_points。
- summary 中记录：
  - `pre_points_policy`
  - `input_pre_points_count`
  - `output_pre_points_count`
  - `prediction_union_count`
  - `pre_points_equals_prediction_union`
  - `prediction_union_subset_of_pre_points`

审计意义：

```text
以前只能从结果猜测是否缩小了评价点集。
现在每个 scene 的 pre_points policy 和 union 关系都能从 summary 直接检查。
```

### local_4d_filter 权重错配修复

修改文件：

```text
Stream3D/stream4d/local_4d_filter.py
```

问题：

```python
for key, weight in zip(support, obs.weights.tolist()):
    weights[key] = ...
```

`support` 是 set，迭代顺序无语义保证；`obs.weights` 是和 `obs.carrier_ids` 对齐的数组。把 set 和 weights zip 可能把 carrier A 的权重写到 carrier B 上。

修复：

```python
for cid, weight in zip(obs.carrier_ids.tolist(), obs.weights.tolist()):
    key = (int(obs.frame_id), int(cid))
    weights[key] = max(weights.get(key, 0.0), float(weight))
```

影响：

```text
这个 bug 不直接解释当前 AP 的全部变化，因为 S1 的主要差异来自 pre_points policy；
但它会影响 local set-cover 选择的稳定性，必须修。
```

### carrier_sampler 长度错配修复

修改文件：

```text
Stream3D/stream4d/carrier_sampler.py
```

问题：

当 `min_points_per_mask` 大于实际 mask 像素数时，`keep` 的长度可能小于请求的 `sample_count`，但 `src_frame/src_global/src_mask_id` 仍用 `sample_count` 生成，导致字段长度潜在不一致。

修复：

```text
用 actual_count = len(keep) 生成所有伴随字段。
```

### 测试

新增：

```text
Stream3D/tests/test_stream4d_protocol_fixes.py
```

结果：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
Ran 2 tests ... OK
```

## S0：Evaluator Protocol Audit

审计输出：

```text
Stream3D/outputs/audit/eval_protocol_audit.md
Stream3D/outputs/audit/eval_protocol_audit.json
Stream3D/outputs/audit/eval_protocol_audit_pre_points_ratio.png
Stream3D/outputs/audit/eval_protocol_audit_prediction_union_ratio.png
```

关键事实：

```text
original evaluate.py exists: True
current evaluate.py exists: True
original has pre_points load: True
current has pre_points load: True
AP core functions equal by hash: True
  evaluate_matches: True
  compute_averages: True
```

解释：

```text
当前 evaluator 的 AP 核心逻辑没有改；
当前改动主要是 tmp_root/tmp_config 可配置，以及一些工程/性能相关代码。
所以这轮主指标仍可视为 Stream3D-style cropped-TMP evaluator。
```

S0 protocol 表：

| Config | AP | AP50 | AP25 | OK scenes | mean pre_points % | mean union % | mean #pred | mean GT crop/full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scannet` | 20.1139 | 34.4654 | 50.2268 | 312/312 | 87.0159 | 87.0159 | 101.1410 | 25.5128/25.5353 |
| `stream4d_scannet_32f_ioc075_fixmem` | 12.7594 | 23.6767 | 42.2114 | 312/312 | 7.3808 | 7.3808 | 185.0192 | 14.7532/25.5353 |
| `stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one` | 20.3718 | 35.5222 | 55.0649 | 312/312 | 5.7458 | 5.7458 | 15.1955 | 12.4840/25.5353 |

直接 insight：

```text
Stream3D baseline 的 evaluator universe 很大，平均覆盖 87.02% mesh vertices，几乎覆盖所有 GT instances。
Stream4D MVP 的 support 很稀疏，平均只覆盖 7.38% vertices 和 14.75/25.54 个 GT instances。
adaptive top-k 进一步把 support 压到 5.75% vertices 和 12.48/25.54 个 GT instances。
```

这说明：

```text
当前 AP 提升不能只看最终 AP 数字；
必须同时看 cropped universe 的大小和 GT instance 覆盖。
```

## S1：pre_points policy 对照

S1 主表：

| Method/config | pre_points policy | AP | AP50 | AP25 | mean pre_points % | mean union % | mean #pred | mean GT crop/full | equal union scenes | union subset scenes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stream3D-Cropformer baseline `scannet` | original/recompute-like | 20.1139 | 34.4654 | 50.2268 | 87.0159 | 87.0159 | 101.1410 | 25.5128/25.5353 | 312 | 312 |
| Stream4D MVP `fixmem` | recompute-like | 12.7594 | 23.6767 | 42.2114 | 7.3808 | 7.3808 | 185.0192 | 14.7532/25.5353 | 312 | 312 |
| adaptive top-k | recompute | 20.3718 | 35.5222 | 55.0649 | 5.7458 | 5.7458 | 15.1955 | 12.4840/25.5353 | 312 | 312 |
| adaptive top-k | inherit | 12.2851 | 23.3147 | 41.6773 | 7.3808 | 5.7458 | 15.1955 | 14.7532/25.5353 | 0 | 312 |
| min250 area filter | recompute | 18.9498 | 32.5721 | 53.0585 | 5.6019 | 5.6019 | 10.4263 | 11.9936/25.5353 | 312 | 312 |
| min250 area filter | inherit | 9.9167 | 18.9180 | 36.6473 | 7.3808 | 5.6019 | 10.4263 | 14.7532/25.5353 | 0 | 312 |

### 对比 1：adaptive recompute vs baseline

```text
AP:   20.3718 - 20.1139 = +0.2579
AP50: 35.5222 - 34.4654 = +1.0568
AP25: 55.0649 - 50.2268 = +4.8381
```

在 recompute 协议下，adaptive top-k 确实超过 baseline。

### 对比 2：adaptive inherit vs baseline

```text
AP:   12.2851 - 20.1139 = -7.8288
AP50: 23.3147 - 34.4654 = -11.1507
AP25: 41.6773 - 50.2268 = -8.5495
```

在 inherit 协议下，adaptive top-k 明显失败。

### 对比 3：adaptive inherit vs MVP

```text
AP:   12.2851 - 12.7594 = -0.4743
AP50: 23.3147 - 23.6767 = -0.3620
AP25: 41.6773 - 42.2114 = -0.5341
```

adaptive top-k 在 inherit 下略低于 MVP，说明 top-k 删实例提高 precision 的同时，没有补足继承 universe 中新增 GT/support 的 false negative 问题。

### 对比 4：min250

```text
min250 recompute: 18.9498 / 32.5721 / 53.0585
min250 inherit:    9.9167 / 18.9180 / 36.6473
```

min250 的 recompute 也能显著提升 MVP，但仍低于 adaptive top-k；inherit 更差。这进一步说明：

```text
问题不是某个 top-k 参数独有；
只要预测 union 明显小于 inherited pre_points，评价点集扩大后都会暴露 coverage 不足。
```

## Phase 1 split 诊断

本轮生成了 deterministic split：

```text
seed = 20260607
splits/scannet_tune.txt: 156 scenes
splits/scannet_final.txt: 156 scenes
```

结果：

| Config | Split | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| Stream3D-Cropformer baseline | tune | 20.8394 | 35.5954 | 50.8577 |
| Stream4D MVP | tune | 13.1056 | 24.2241 | 42.6161 |
| adaptive top-k recompute | tune | 20.5105 | 35.3847 | 55.1521 |
| adaptive top-k inherit | tune | 12.6566 | 23.6953 | 42.1901 |
| Stream3D-Cropformer baseline | final | 19.4294 | 33.3989 | 49.6361 |
| Stream4D MVP | final | 12.4298 | 23.1572 | 41.8428 |
| adaptive top-k recompute | final | 20.2401 | 35.6642 | 54.9907 |
| adaptive top-k inherit | final | 11.9313 | 22.9523 | 41.1922 |

split insight：

```text
adaptive recompute tune AP = 20.5105
adaptive recompute final AP = 20.2401
gap = 0.2704 AP
```

这个固定配置的 tune/final gap 不大；但是本轮没有做完整 tune grid search，因此不能写“通过 tune 搜索后 locked final 超越”。可以写：

```text
固定 adaptive config 在 final split 的 recompute 协议下仍超过该 final split baseline：
  AP:   20.2401 vs 19.4294
  AP50: 35.6642 vs 33.3989
  AP25: 54.9907 vs 49.6361

同一固定 config 在 final split 的 inherit 协议下仍失败：
  11.9313 / 22.9523 / 41.1922
```

## 失败原因分析

### 主要失败原因：support coverage 不足

证据链：

```text
Stream3D baseline mean pre_points ratio = 87.0159%
Stream4D MVP mean pre_points ratio      = 7.3808%
adaptive recompute mean pre_points ratio = 5.7458%
adaptive inherit mean pre_points ratio   = 7.3808%, 但 prediction union 仍是 5.7458%
```

解释：

```text
recompute 会把 evaluator 的 GT 和预测都裁到当前 prediction union。
如果 top-k 删除了大量低质量或小 support 的候选，prediction union 变小；
评价 universe 也随之变小，很多没有 support 的 GT 不再进入评价。

inherit 会保留 MVP 的 support universe。
同样的 top-k prediction 需要解释更大的 support universe；
被 top-k 删除或没有覆盖到的 GT 会重新变成 false negatives。
所以 AP 立刻掉回 12.2851。
```

这不是 evaluator 假指标，而是 cropped-TMP 协议本身的性质。S0 已确认当前 evaluator AP core 与原始 Stream3D 一致；问题在于 Stream4D 当前 export/support 太稀疏。

### 次要原因：top-k 提高 precision，但降低 recall/support

证据：

```text
MVP mean #pred = 185.0192
adaptive mean #pred = 15.1955
min250 mean #pred = 10.4263
```

adaptive top-k 把实例数量从 185 压到 15，显著减少 false positives，因此 recompute AP 上升。但没有新的 densification 或 memory support，不能覆盖 inherit universe 中更多 GT。

### min250 的启示

min250 recompute 仍有 18.9498 AP，说明简单删除小对象/小 support 确实能改善 recompute 协议下的 precision。但 min250 inherit 只有 9.9167，说明单靠“删小实例”不能解决 support 召回。

## 可以写与不能写

可以写：

```text
在 Stream3D-style cropped-TMP evaluator 中，使用 recompute_pre_points 的 observed-support 设置，
Stream4D adaptive mask-count top-k 在 ScanNet val 上达到 20.3718 / 35.5222 / 55.0649，
超过本地复现的 Stream3D-Cropformer baseline 20.1139 / 34.4654 / 50.2268。
```

必须同时写：

```text
该优势依赖 recompute_pre_points。
在 inherit_pre_points 对照中，同一 prediction 降至 12.2851 / 23.3147 / 41.6773。
因此当前结果不能作为“全面优于 Stream3D”的证据。
```

不能写：

```text
Stream4D 已全面超过 Stream3D。
Stream4D 的 current top-k 方法在统一/inherit support universe 下超过 Stream3D。
ScanNet 已证明动态 4D tracking。
D4RT geometry 已替代 ScanNet RGB-D/pose bridge。
```

## 下一步建议

按 v3 计划 9.1，本轮落入：

```text
recompute 超 baseline，但 inherit 不超。
```

所以优先级应为：

1. reliable densification：让 prediction union 在不引入大量冲突的情况下接近 inherited support。
2. memory-v2：减少多窗口碎片和 object explosion，让更多帧真正提升 coverage。
3. quality score 只作为 selection 辅助，不应继续盲目扩展 ratio grid。

本轮没有编造 Phase 3-6 的结果，也没有把未跑的 Dynamic Replica 写成实验结论。
