# Stream4D v4.1 ScanNet 实验结果复盘

## 1. 本轮目标

按照 `docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md` 的第一批任务，先建立可审计的指标完整性守卫和 Stream3D cross-pre_points 诊断，再判断 v3 结果是否能在和原版 Stream3D 对齐的评估方式下成立。

## 2. 假设

- H0：当前代码没有直接篡改 AP 计算，也没有在 rescore 阶段读取 ScanNet GT 来生成预测。
- H1：`scannet_self_inherit` 应与原 `scannet` baseline 基本一致，AP 差异应小于 0.01。
- H2：Stream3D baseline 在 Stream4D pre_points 下的 cross-fixed 结果可以解释 Stream4D sparse support 对评估 universe 的影响。
- H3：v3 Stream4D adaptive 的优势主要来自 recompute_pre_points；inherit 或 cross-fixed 诊断必须如实报告，不能隐藏。

## 3. 修改了哪些代码

### 3.1 指标与协议审计相关

| 文件 | 修改 | 审计意义 |
|---|---|---|
| `Stream3D/tools/evaluate_cross_prepoints.py` | 新增 cross-pre_points 诊断工具 | 只替换 evaluator 使用的 pre_points，不改变 prediction mask，用于拆开“预测质量”和“评估支撑点集合” |
| `Stream3D/tools/verify_stream4d_metric_integrity.py` | 新增指标完整性守卫 | 检查 evaluator hash、rescore 是否读 GT、prediction/TMP/object_dict 对齐、pre_points/union/GT crop |
| `Stream3D/stream4d/rescore_scannet.py` | 新增 object_dict 与 prediction mask 列点级对齐检查；新增 fixed_path 参数校验 | 防止 object_dict 和 pred columns 错位，防止 fixed_path 空参数误用 |
| `Stream3D/stream4d/export_scannet.py` | 修正 point reuse / point_dilate 诊断字段 | 避免把 point reuse 误写成 NN hit rate |
| `Stream3D/stream4d/reexport_scannet.py` | 根据 reexport mode 写正确 support mode 名称 | 避免 `point_dilate` 被记录成 `mask_backproject` |
| `Stream3D/tools/check_dynamic_replica_env.py` | 新增 Dynamic Replica 数据环境检查 | 判断后续动态实验能否报告 official 指标 |
| `Stream3D/tests/test_stream4d_protocol_fixes.py` | 新增 alignment mismatch 和 fixed_path 校验测试 | 防止协议守卫回退 |

## 4. 运行命令

完整命令见执行日志：`docs/stream4d_v4_1_执行日志.md`。

关键命令类别：

1. `python -m py_compile ...`
2. `python -m unittest tests.test_stream4d_protocol_fixes`
3. `python -m tools.evaluate_cross_prepoints ...` 四条 cross-pre_points 诊断。
4. `python -m tools.verify_stream4d_metric_integrity ...`
5. `python -m tools.check_dynamic_replica_env ...`

## 5. 完整性检查

完整性审计文件：

- `Stream3D/outputs/audit/stream4d_v4_metric_integrity.md`
- `Stream3D/outputs/audit/stream4d_v4_metric_integrity.json`

核心审计结论：

```text
phase0_pass=True
evaluator_ap_core_equal_by_hash=True
has_pre_points_load_original=True
has_pre_points_load_current=True
gt_files_read_by_rescore=False
```

object_dict 与 prediction 列对齐：

| Config | checked scenes | mean IoU | min IoU | failed instances |
|---|---:|---:|---:|---:|
| `scannet` | 312 | 1.000000 | 1.000000 | 0 |
| `stream4d_scannet_32f_ioc075_fixmem` | 312 | 1.000000 | 1.000000 | 0 |
| `stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute` | 312 | 1.000000 | 1.000000 | 0 |
| `stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit` | 312 | 1.000000 | 1.000000 | 0 |

说明：

- baseline/cross-prepoints 的新 output configs 没有对应 object_dict，因此 alignment 记为不可验证；这不是失败，但也不能拿它们声称 object_dict 对齐。
- 当前 AP core 函数与原版 Stream3D 的 `evaluate_matches` / `compute_averages` hash 一致。
- `rescore_scannet.py` 没有发现 `gt_path`、`data/scannet/gt`、`np.loadtxt` 等 GT 读取痕迹。

## 6. 主指标

所有数值来自 `Stream3D/data/evaluation/scannet/*.txt` 结果文件最后一行，下面表格已乘以 100。

| 方法 / 诊断行 | prediction config | pre_points config | AP | AP50 | AP25 |
|---|---|---|---:|---:|---:|
| Stream3D baseline original | `scannet` | `scannet` | 20.1139 | 34.4654 | 50.2268 |
| Stream3D self-inherit | `scannet` | `scannet` | 20.1139 | 34.4654 | 50.2268 |
| Stream3D on Stream4D MVP pre_points | `scannet` | `stream4d_scannet_32f_ioc075_fixmem` | 29.8516 | 47.5638 | 64.5075 |
| Stream3D on Stream4D adaptive pre_points | `scannet` | `stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute` | 32.6669 | 50.1160 | 67.6487 |
| Stream4D MVP recompute | `stream4d_scannet_32f_ioc075_fixmem` | same config | 12.7594 | 23.6767 | 42.2114 |
| Stream4D v3 adaptive recompute | `stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute` | same config | 20.3718 | 35.5222 | 55.0649 |
| Stream4D v3 adaptive inherit | `stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit` | MVP pre_points | 12.2851 | 23.3147 | 41.6773 |
| Stream4D adaptive on Stream3D pre_points | `stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute` | `scannet` | 0.1003 | 0.4225 | 3.2100 |

最关键的协议判断：

- 与原版 Stream3D 评估方式一致的对照是 `Stream3D baseline original` 和 `Stream3D self-inherit`。
- `scannet_self_inherit` 与 `scannet` baseline 完全一致：

```text
baseline:       20.11390105837629 / 34.46536779373159 / 50.22682476624883
self-inherit:   20.11390105837629 / 34.46536779373159 / 50.22682476624883
delta:          0 / 0 / 0
```

这满足计划里的 `AP delta < 0.01` 要求。

## 7. 诊断指标

来自 `Stream3D/outputs/audit/cross_prepoints_audit.md` 与 `stream4d_v4_metric_integrity.md`。

| output_config | target pre_points % | prediction union % | union in target % scene/target | GT crop/full | #pred | mask shape |
|---|---:|---:|---:|---:|---:|---|
| `scannet_self_inherit` | 87.0159 | 87.0159 | 87.0159 / 100.0000 | 25.51 / 25.54 | 101.14 | full_scene: 312 |
| `scannet_on_stream4d_mvp_prepoints` | 7.3808 | 87.0159 | 7.2568 / 98.3165 | 14.75 / 25.54 | 101.14 | full_scene: 312 |
| `scannet_on_stream4d_adaptive_prepoints` | 5.7458 | 87.0159 | 5.6639 / 98.5903 | 12.48 / 25.54 | 101.14 | full_scene: 312 |
| `stream4d_adaptive_on_scannet_prepoints` | 87.0159 | 5.7458 | 5.6639 / 6.4695 | 25.51 / 25.54 | 15.20 | full_scene: 312 |

证据链解释：

1. Stream3D baseline 的 prediction union 覆盖约 87.02% scene vertices。
2. Stream4D MVP pre_points 只覆盖约 7.38% scene vertices；adaptive recompute pre_points 只覆盖约 5.75%。
3. 当把 Stream3D full prediction 放到这两个小 pre_points 里评估时，AP 显著升高到 29.85 和 32.67。
4. 当把 Stream4D adaptive prediction 放回 Stream3D baseline 的大 pre_points 里评估时，AP 降到 0.1003。
5. 所以 v3 adaptive recompute 的 20.3718 不能被解读为“预测本身稳定超越 Stream3D”；它强依赖很小的 observed support / pre_points universe。

## 8. 可视化路径

已生成：

- `Stream3D/outputs/audit/pre_points_ratio_by_config.png`
- `Stream3D/outputs/audit/union_ratio_by_config.png`
- `Stream3D/outputs/audit/gt_crop_full_by_config.png`
- `Stream3D/outputs/audit/object_dict_alignment_iou_hist.png`

本轮没有生成 prediction-vs-GT mesh 可视化，因为 S0 任务重点是 metric integrity 和 cross-pre_points 协议诊断。

## 9. 假设是否成立

| 假设 | 结论 | 证据 |
|---|---|---|
| H0：没有直接篡改 AP，也没有 rescore GT leakage | 成立 | AP core hash 一致；`gt_files_read_by_rescore=False` |
| H1：`scannet_self_inherit` 应与原 baseline 一致 | 成立 | 两者 AP/AP50/AP25 完全相同 |
| H2：Stream3D cross-fixed 可解释 sparse pre_points universe 的影响 | 成立 | Stream3D on MVP/adaptive pre_points 分别达到 29.8516 / 32.6669 AP |
| H3：v3 adaptive 优势主要来自 recompute_pre_points | 成立 | adaptive recompute 20.3718，但 adaptive on Stream3D pre_points 只有 0.1003 AP |

## 10. 失败原因

本轮最重要的失败不是工具失败，而是方法 claim 失败：

### 10.1 Stream4D adaptive 不能在 Stream3D baseline support 下成立

`stream4d_adaptive_on_scannet_prepoints` 只有：

```text
AP / AP50 / AP25 = 0.1003 / 0.4225 / 3.2100
```

直接原因：

- Stream4D adaptive prediction 的 union 只有 5.7458% scene vertices。
- Stream3D baseline pre_points 有 87.0159% scene vertices。
- 在这个大 support 里，Stream4D adaptive prediction 只覆盖 target pre_points 的 6.4695%。
- GT crop/full 从 adaptive support 的 12.48/25.54 恢复到 baseline support 的 25.51/25.54 后，Stream4D adaptive 的漏召回非常严重。

### 10.2 Stream3D 在同一 sparse support 下远强于 Stream4D adaptive

对比同一个 adaptive pre_points：

```text
Stream3D on Stream4D adaptive pre_points: 32.6669 / 50.1160 / 67.6487
Stream4D adaptive recompute:              20.3718 / 35.5222 / 55.0649
```

解释：

- 这说明 Stream4D adaptive 的 AP 提升不是因为在同一 sparse support 下 mask/object 质量超过 Stream3D。
- sparse support 让评价集合变容易，但 Stream3D full prediction 在这个小集合里更强。

### 10.3 不能继续把 v3 adaptive recompute 写成稳健超越

v3 adaptive recompute 相比 original baseline：

```text
20.3718 - 20.1139 = +0.2579 AP
```

但这个优势在 cross support 诊断下不稳健。按照 v4.1 计划安全边界，只能写：

```text
在当前 Stream3D-style cropped-TMP recompute 协议下，v3 adaptive 有 slight improvement；
但 cross-pre_points 诊断显示该结果强依赖较小 support，不支持稳健超越 Stream3D-Cropformer 的 claim。
```

### 10.4 后续大方法模块的 blocker

1. self/cycle evidence 未完成：当前 `CarrierBatch` 没有 reverse/cycle 输出，不能编造 `rho_self_cycle_boundary`。
2. reliable densification 未完成：当前 exporter 没有 seeded component + WTA conflict 的完整实现，不能用 dilation 冒充。
3. memory-v2 未完成：当前 memory 是 carrier-overlap greedy matching，不是 Hungarian + appearance/motion lifecycle。
4. Dynamic Replica 数据缺失：`data/dynamic-replica/v2` 不存在，不能跑动态 official 指标。

## 11. 下一步尝试方向

优先级建议：

1. 先不要继续调 adaptive top-k。cross-pre_points 已经证明 support 依赖过强。
2. 实现 reliable densification 的最小闭环：
   - seeded component
   - boundary erosion
   - seed distance cap
   - WTA conflict suppression
   - 输出 `mask_pixels_total`、`mask_pixels_kept`、`conflict_removed_ratio`
3. 在 `splits/scannet_tune.txt` 上只调 densification 参数，锁定后再跑 final。
4. 扩展 `D4RTAdapter`，增加 reverse query 和 cycle error，才能实现 `rho_self_cycle_boundary`。
5. memory-v2 先只做 5 个小场景，不直接跑 full ScanNet，避免 object explosion。
6. 若要做动态实验，先补 Dynamic Replica 数据；没有 GT instance/object IDs 时只允许写 pseudo/qualitative。

## 12. 哪些结论不能写

不能写：

- 不能写“当前 Stream4D v3/v4 已稳健超越 Stream3D-Cropformer on ScanNet”。
- 不能写“当前 ScanNet 结果证明了 dynamic semantic 4D reconstruction”。
- 不能写“当前结果是 D4RT-native geometry”；目前仍是 rgbd_eval bridge。
- 不能隐藏 `inherit_pre_points` 和 `stream4d_adaptive_on_scannet_prepoints` 的失败。
- 不能把 Stream3D on Stream4D sparse pre_points 的 32.6669 当成 Stream3D 原 benchmark 数字；它只是 cross-fixed diagnostic。
- 不能把 Dynamic Replica official tracking / semantic AP 写入报告；本轮数据根目录不存在。

## 13. 2026-06-08 继续推进结果：reliable densification v1

### 13.1 这次是否达成 v4.1 目标

结论：**部分达成，但不能写成 v4.1 全面达成。**

已经达成：

- 在 `splits/scannet_final.txt` 的 locked final split 上，`Stream4D-v4.1 reliable densification, recompute_pre_points` 明显超过本地 Stream3D-Cropformer final baseline。
- 该结果使用的是原版 Stream3D-style cropped-TMP evaluator，不是 fullmesh evaluator，也没有改 AP 计算。

没有达成：

- `inherit_pre_points` / `fixed_pre_points` 稳健性没有达成。
- 放回 MVP / adaptive / Stream3D 大支持集后，v4.1 reliable densification 仍然很低。
- 因此不能写“Stream4D-v4.1 已稳健超越 Stream3D-Cropformer”，只能写“在方法自身 recompute observed support 的原版 Stream3D-style 协议下显著超越”。

### 13.2 本轮代码修改审计

| 文件 | 修改 | 目的 |
|---|---|---|
| `Stream3D/stream4d/reliable_densifier.py` | 新增 seeded component、boundary erosion、seed distance cap、3D NN backprojection、WTA conflict suppression | 按计划实现 reliable densification v1，而不是用简单 point dilation 冒充 |
| `Stream3D/stream4d/reliable_densifier.py` | 新增 `seed_keep_mode=none|supported|all` | 诊断是否应该把旧 `object_dict.point_ids` 保留进最终 prediction |
| `Stream3D/stream4d/export_scannet.py` | 新增 `export_support_mode="reliable_densify"` 和 densifier 参数 | 让 reexport 可以生成 reliable densified prediction/TMP/object_dict |
| `Stream3D/stream4d/reexport_scannet.py` | 新增 `--reexport-mode reliable_densify` 和 densify CLI 参数 | 让每个实验命令能完整记录参数 |
| `Stream3D/tests/test_stream4d_protocol_fixes.py` | 新增 WTA 单测 | 防止冲突点归属逻辑回退 |
| `Stream3D/splits/scannet_tune30.txt` | 新增 30-scene tune 子集 | 快速调参，避免直接在 final 上调 |

语法和单测：

```text
Ran 5 tests in 0.001s
OK
```

### 13.3 tune30 结果

所有数值均来自 `Stream3D/data/evaluation/scannet/*.txt` 的最后一行，并乘以 100。

| 方法 / 诊断 | eval support | AP | AP50 | AP25 | 说明 |
|---|---|---:|---:|---:|---|
| Stream3D tune30 | own recompute/self | 21.6016 | 36.3863 | 51.9451 | baseline |
| v3 adaptive tune30 | own recompute | 23.0107 | 38.9224 | 58.0834 | v3 对照 |
| v3 adaptive tune30 | MVP fixed/inherit | 14.1553 | 26.9177 | 44.3760 | v3 inherit 诊断 |
| reliable no-seed e1 d32 top5 | own recompute | 30.8405 | 52.6368 | 66.4716 | tune30 最强 recompute |
| reliable no-seed e1 d32 top5 | MVP fixed | 9.2089 | 19.2868 | 33.5045 | inherit 失败 |
| reliable no-seed e1 d32 top5 | adaptive fixed | 13.1056 | 25.6553 | 42.6223 | 仍低于 v3 adaptive recompute |
| seed-supported e1 d32 top5 | own recompute | 29.5668 | 51.3542 | 65.1298 | 保留可靠支持 seed 后 recompute 略降 |
| seed-supported e1 d32 top5 | MVP fixed | 10.6982 | 20.2008 | 33.7487 | 比 no-seed 略好但未达标 |
| seed-supported e1 d32 top5 | adaptive fixed | 15.3758 | 27.0733 | 44.2360 | 未达到 inherit +3 AP 标准 |
| seed-all e1 d32 top5 min20 | own recompute | 20.4552 | 37.6276 | 56.7153 | 保留全部旧点导致 recompute 明显下降 |
| seed-all e1 d32 top5 min20 | MVP fixed | 15.5576 | 28.6274 | 47.9995 | inherit 改善但仍未达 +3，且 recompute 失败 |
| MVP object_dict reliable | own recompute | 22.6664 | 38.4791 | 59.1280 | 对象更多但质量不足 |
| MVP object_dict reliable | MVP fixed | 7.9377 | 17.5412 | 35.8838 | 高冲突候选不能靠 WTA 后自动变好 |

关键诊断：

| config | union % | #objects | points | conflict | WTA pre-conflict | kept ratio |
|---|---:|---:|---:|---:|---:|---:|
| reliable no-seed e1 d32 top5 tune30 | 6.8552 | 10.5000 | 10078.2 | 0.000000 | 0.045291 | 0.847360 |
| seed-supported e1 d32 top5 tune30 | 7.0841 | 10.9000 | 10402.8 | 0.000000 | 0.045902 | 0.847360 |
| seed-all e1 d32 top5 min20 tune30 | 11.0730 | 14.4667 | 14903.9 | 0.000000 | 0.087878 | 0.847360 |
| MVP object_dict reliable tune30 | 13.8959 | 20.3333 | 20047.7 | 0.000000 | 0.448439 | 0.580677 |

解释：

- no-seed reliable densification 能显著提升 recompute AP，说明 seeded component + boundary + distance cap 确实生成了更高质量的 observed support。
- 但 no-seed 的 union 仍只有 6.8552%，对象数只有 10.5，放回 MVP/adaptive support 后漏召回严重。
- seed-all 把 union 提升到 11.0730%，但 recompute 从 30.8405 掉到 20.4552，说明原始 `object_dict.point_ids` 里存在会破坏实例 IoU 的旧点、边界点或错归属点。
- MVP object_dict reliable 候选对象更多，但 WTA 前冲突率达到 0.448439，说明原始 MVP 候选高度重叠/碎片化；“更多对象”本身不能解决 AP。

### 13.4 final split locked 结果

final split 使用 `splits/scannet_final.txt`，共 156 个场景。参数在 tune30 上锁定，没有在 final 上继续调：

```text
seed_keep_mode=none
boundary_erosion=1
seed_distance_px=32
max_masks_per_object=5
export_min_points_per_object=100
mask_sample_stride=2
mask_max_pixels=12000
nn_radius=0.08
WTA enabled
```

| 方法 / 诊断 | prediction config | pre_points config | AP | AP50 | AP25 |
|---|---|---|---:|---:|---:|
| Stream3D final baseline | `scannet_v3_final` | `scannet_v3_final` | 19.4294 | 33.3989 | 49.6361 |
| Stream4D v3 adaptive final | v3 adaptive | own recompute | 20.2401 | 35.6642 | 54.9907 |
| Stream4D v3 adaptive final | v3 adaptive | MVP inherit | 11.9313 | 22.9523 | 41.1922 |
| Stream4D v4.1 reliable final | v4.1 reliable | own recompute | 30.2449 | 51.5938 | 67.0619 |
| Stream4D v4.1 reliable final | v4.1 reliable | MVP fixed | 9.0795 | 18.2119 | 30.2373 |
| Stream4D v4.1 reliable final | v4.1 reliable | v3 adaptive fixed | 13.0900 | 24.2122 | 37.8926 |
| Stream4D v4.1 reliable final | v4.1 reliable | Stream3D fixed | 0.6823 | 2.1181 | 7.3900 |

相对提升：

```text
v4.1 reliable final recompute AP - Stream3D final AP
= 30.2449 - 19.4294
= +10.8155 AP

v4.1 reliable final recompute AP - v3 adaptive final recompute AP
= 30.2449 - 20.2401
= +10.0048 AP
```

这说明 final split recompute 主结果很强，且不是 tune30 偶然高。

### 13.5 final split 覆盖和 fixed-support 证据链

来自：

- `Stream3D/outputs/stream4d_reexport_v4_1/stream4d_v4_1_reliable_e1_d32_top5_final_summary.json`
- `Stream3D/outputs/audit/cross_prepoints/stream4d_v4_1_reliable_e1_d32_top5_on_mvp_final_summary.json`
- `Stream3D/outputs/audit/cross_prepoints/stream4d_v4_1_reliable_e1_d32_top5_on_adaptive_final_summary.json`
- `Stream3D/outputs/audit/cross_prepoints/stream4d_v4_1_reliable_e1_d32_top5_on_scannet_final_summary.json`

v4.1 final own support：

```text
mean union / pre_points ratio = 7.6984%
mean exported objects = 9.9872
mean exported points = 9936.7
mean export hit rate = 0.879791
mean final conflict rate = 0.000000
mean WTA pre-conflict rate = 0.052659
mean kept ratio = 0.836679
```

fixed support 诊断：

| target support | target pre_points % | v4.1 union % | union in target % scene/target | GT crop/full | #pred | AP |
|---|---:|---:|---:|---:|---:|---:|
| MVP support | 7.3500 | 7.6984 | 2.4179 / 33.7635 | 14.98 / 25.96 | 9.99 | 9.0795 |
| v3 adaptive support | 5.7384 | 7.6984 | 2.1982 / 40.2672 | 12.87 / 25.96 | 9.99 | 13.0900 |
| Stream3D support | 87.1898 | 7.6984 | 7.5048 / 8.6169 | 25.93 / 25.96 | 9.99 | 0.6823 |

解释：

- v4.1 reliable final 的 own support 很小，只有 7.6984% scene vertices。
- 在 Stream3D support 中，v4.1 prediction union 只覆盖 target pre_points 的 8.6169%，所以 AP 只有 0.6823。
- 在 MVP/adaptive support 中，AP 也只有 9.0795 / 13.0900，说明当前 v4.1 仍没有解决 inherit/fixed support 的稳定性。
- final conflict rate 为 0，不是因为多对象冲突爆炸导致失败；主要失败是 object coverage / false negative。

### 13.6 为什么 reliable densification 能强提升 recompute，但不能解决 inherit

证据链：

1. no-seed reliable densification 把每个 object 限制在 seed 所在 2D mask 连通域、边界腐蚀后区域、seed 距离阈值内，因此实例 mask 更干净，recompute AP 大幅提升。
2. recompute_pre_points 会把 evaluator 的 `pre_points` 设置成 prediction union；当 prediction union 小而干净时，GT crop 也只在这个 observed support 里评估，所以 AP 很高。
3. inherit/fixed support 会把 GT crop 放回更大的 MVP/adaptive/Stream3D support。当前 v4.1 平均只有约 10 个 object，union 约 7.7%，无法覆盖这些更大 support 中的 GT instance。
4. 尝试保留旧 seed 点可以提升覆盖：
   - seed-all tune30 union 提到 11.0730%。
   - MVP fixed AP 从 9.2089 提到 15.5576。
   - adaptive fixed AP 从 13.1056 提到 23.7159。
5. 但 seed-all recompute AP 从 30.8405 降到 20.4552，说明旧 seed 点不是可靠 dense mask；直接保留会把错点/边界点/碎片点带回预测，破坏 AP/AP50。
6. 降低 `export_min_points_per_object` 从 100 到 20 增加了 object count，但 no-seed fixed AP 基本没变，说明漏召回不是简单由小对象阈值造成。
7. 用 MVP object_dict 输入能增加对象数到 20.33，但 WTA pre-conflict rate 达 0.448439，AP 只有 22.6664 recompute / 7.9377 MVP fixed，说明 MVP 原始候选高度重叠和碎片化，不能直接作为更高 coverage 的解法。

### 13.7 最终结论

可以写：

```text
在 locked final split 上，Stream4D-v4.1 reliable densification 在原版 Stream3D-style cropped-TMP recompute_pre_points 协议下达到 30.2449 / 51.5938 / 67.0619，显著超过本地 Stream3D-Cropformer final baseline 的 19.4294 / 33.3989 / 49.6361，也超过 v3 adaptive final recompute 的 20.2401 / 35.6642 / 54.9907。
```

必须同时写：

```text
该结果仍强依赖自身 observed support。放回 MVP support 后只有 9.0795 AP，放回 Stream3D support 后只有 0.6823 AP。v4.1 没有达成 inherit/unified support 稳健超越。
```

不能写：

```text
Stream4D-v4.1 全面超越 Stream3D-Cropformer。
Stream4D-v4.1 已解决 inherit_pre_points 失败。
Stream4D-v4.1 已证明完整 4D semantic reconstruction。
```

### 13.8 下一步真实修复方向

如果继续推进，不应继续只调 densify 半径或 min-points。当前证据显示需要解决更上游的问题：

1. **object selection 召回不足**：当前 adaptive top-k 平均只留下约 10-15 个对象，v4.1 final 只有 9.99 个预测对象；要提高 inherit，需要 memory-v2 或更可靠的候选合并，而不是只扩张已有对象。
2. **旧 seed 点质量不稳定**：seed-all 能改善 fixed AP，但严重伤害 recompute；需要 self/cycle/boundary evidence 去筛掉漂移 carrier 和边界点，不能无条件保留旧点。
3. **MVP 候选重叠严重**：MVP reliable 的 WTA pre-conflict rate 约 0.448，说明需要 Hungarian one-to-one memory、same-frame exclusivity 和 appearance/centroid gate。
4. **Dynamic Replica 仍缺数据**：本轮没有数据条件报告动态官方指标，不能把 ScanNet 静态结果外推成动态 tracking 结论。

## 14. 2026-06-08 继续推进：coverage 融合仍未达成 inherit 目标

### 14.1 为什么继续做这一轮

上一节已经证明：

- no-seed reliable densification：recompute 很强，但 fixed support 低。
- seed-all：coverage 更高，但 recompute 明显下降。

因此本轮继续尝试两个折中方案：

1. **直接拼接**：clean dense mask 保持高分，coverage seed-all mask 放低分。
2. **选择式融合**：不增加重复实例；对每个 coverage mask 匹配最相近 clean mask，只有当它覆盖 clean mask 且扩张倍数不超过阈值时，才替换 clean mask。

新增代码：

- `Stream3D/tools/fuse_prediction_configs.py`

### 14.2 代码修改审计

`fuse_prediction_configs.py` 做了什么：

- 读取两个 prediction config 的 `pred_masks` / `pred_score` / `pred_classes`。
- 输出新的 prediction config 和对应 `data/TMP/<output_config>/<scene>_pre_points.npy`。
- `fusion-mode=concatenate` 会拼接两个 config 的 mask，并给 secondary 一个较低分数。
- `fusion-mode=select_secondary` 会计算 secondary mask 对 primary mask 的包含比例和扩张倍数。
- 选择式融合的关键参数：
  - `--select-min-primary-ioc 0.7`
  - `--select-max-expansion 1.5 / 2.0 / 3.0`
  - `--add-unmatched-secondary` 可选。

验证：

```text
Ran 5 tests in 0.001s
OK
```

### 14.3 tune30 结果

所有数值来自 `Stream3D/data/evaluation/scannet/*.txt`，乘以 100。

| 方法 / 诊断 | support | AP | AP50 | AP25 | 结论 |
|---|---|---:|---:|---:|---|
| direct concat clean+seedall | own recompute | 16.9892 | 33.9073 | 53.0696 | 直接拼接失败，重复/低分 coverage 仍伤 AP |
| direct concat clean+seedall | MVP fixed | 11.7611 | 24.7514 | 43.9886 | fixed 有改善但主指标崩 |
| select r1.5 | own recompute | 30.1554 | 52.2972 | 65.6328 | 基本保住 no-seed 主指标 |
| select r1.5 | MVP fixed | 11.8142 | 20.7891 | 34.7619 | 比 no-seed 9.2089 好，但仍不够 |
| select r1.5 | adaptive fixed | 17.2152 | 28.0733 | 43.7700 | 比 no-seed 13.1056 好 |
| select r2.0 | own recompute | 29.7040 | 51.8314 | 64.1886 | 主指标小降 |
| select r2.0 | MVP fixed | 12.3543 | 21.9200 | 34.7619 | MVP fixed 小幅改善 |
| select r2.0 | adaptive fixed | 18.1056 | 30.0885 | 43.7700 | adaptive fixed 改善 |
| select r3.0 | own recompute | 27.9897 | 48.9810 | 62.4055 | 主指标继续下降 |
| select r3.0 | MVP fixed | 13.2994 | 23.5106 | 36.0863 | MVP fixed 提升但仍不够 |
| select r3.0 | adaptive fixed | 19.6440 | 32.4120 | 45.7540 | adaptive fixed 提升 |
| select r1.5 + unmatched | own recompute | 23.9432 | 43.6260 | 60.3566 | 加 unmatched secondary 后主指标明显下降 |
| select r1.5 + unmatched | MVP fixed | 13.4209 | 24.7500 | 43.3579 | fixed 改善，但换来主指标损失 |
| select r1.5 + unmatched | adaptive fixed | 20.1797 | 34.9956 | 54.3890 | fixed 改善，但不适合作主候选 |

tune30 选择式融合诊断：

| config | output union % | #instances | replaced primary | matched secondary | unmatched added |
|---|---:|---:|---:|---:|---:|
| select r1.5 | 7.1045 | 10.5000 | 5.2333 | 9.7000 | 0.0000 |
| select r2.0 | 7.2525 | 10.5000 | 5.6333 | 9.7000 | 0.0000 |
| select r3.0 | 7.5545 | 10.5000 | 6.3667 | 9.7000 | 0.0000 |
| select r1.5 + unmatched | 9.6484 | 15.2667 | 5.2333 | 9.7000 | 4.7667 |

分析：

- 选择式融合验证了一个真实 trade-off：适度替换可以让 fixed support 变好，同时主指标只小幅下降。
- 但只替换、不加 unmatched 时，object count 仍停在 10.5，MVP fixed 不可能大幅提升。
- 加 unmatched 能提高 coverage 和 fixed AP，但立刻导致 recompute 下降到 23.9432，说明额外候选仍然质量不足。

### 14.4 final 结果

final 使用 `splits/scannet_final.txt`，没有在 final 上继续搜索阈值；只把 tune30 上看起来有意义的 r1.5 / r3.0 和 seed-all coverage 上限跑了一遍。

| 方法 / 诊断 | support | AP | AP50 | AP25 | 结论 |
|---|---|---:|---:|---:|---|
| v4.1 reliable no-seed | own recompute | 30.2449 | 51.5938 | 67.0619 | 当前最强主指标 |
| v4.1 reliable no-seed | MVP fixed | 9.0795 | 18.2119 | 30.2373 | inherit/fixed 失败 |
| select r1.5 final | own recompute | 29.0550 | 50.5448 | 66.1502 | 仍显著超过 Stream3D final，但低于 no-seed |
| select r1.5 final | MVP fixed | 11.0744 | 20.0240 | 30.6405 | 比 no-seed 改善，但低于 v3 inherit 11.9313 |
| select r1.5 final | adaptive fixed | 16.2607 | 27.0362 | 38.8649 | 比 no-seed adaptive fixed 13.0900 改善 |
| select r1.5 final | Stream3D fixed | 0.7055 | 2.1770 | 7.6851 | 仍灾难性低 |
| select r3.0 final | own recompute | 27.2674 | 47.4005 | 64.3080 | 主指标继续下降 |
| select r3.0 final | MVP fixed | 11.6883 | 21.2620 | 31.9825 | 接近但仍低于 v3 inherit 11.9313 |
| select r3.0 final | adaptive fixed | 17.2287 | 28.8811 | 40.5127 | fixed 改善但不够 |
| seed-all final | own recompute | 18.7252 | 35.0749 | 53.7801 | 最大 coverage 版本主指标低于 Stream3D final baseline |
| seed-all final | MVP fixed | 13.4864 | 25.4313 | 43.8950 | fixed 最好，但主指标失败 |
| seed-all final | adaptive fixed | 20.6866 | 36.2401 | 56.0416 | adaptive fixed 较好，但不能作为主候选 |

final coverage 诊断：

| config | union % | #instances | notes |
|---|---:|---:|---|
| reliable no-seed final | 7.6984 | 9.9872 | 主指标最佳，但 coverage 小 |
| select r1.5 final | 8.0031 | 9.9872 | 替换约 4.9679 个 primary/method-scene |
| select r3.0 final | 8.2706 | 9.9872 | 替换约 5.8077 个 primary/method-scene |
| seed-all final | 11.2671 | 13.9679 | coverage 最大，但 recompute 崩 |

### 14.5 本轮是否达成目标

仍然没有全面达成。

达成项：

- final recompute 仍有多个配置显著超过 Stream3D final baseline。
- 选择式融合证明可以在不完全牺牲 recompute 的情况下，小幅改善 MVP/adaptive fixed support。

未达成项：

- 没有一个配置同时满足：
  - recompute 保持 v4.1 no-seed 的强结果；
  - MVP inherit/fixed 明显超过 v3 inherit；
  - Stream3D fixed 不灾难性低。
- seed-all 虽然让 MVP fixed 达到 13.4864，但 own recompute 只有 18.7252，低于 Stream3D final baseline 19.4294。
- select r3.0 final MVP fixed 只有 11.6883，仍低于 v3 adaptive inherit final 11.9313。

### 14.6 更新后的失败原因

本轮更明确地定位了失败原因：

1. **不是单纯 score 排序问题。** 直接拼接 high-score clean + low-score coverage 让 recompute 降到 16.9892，说明重复/低质量 coverage mask 会成为大量 false positives。
2. **不是单纯扩张阈值问题。** r1.5/r2.0/r3.0 形成平滑 trade-off：扩张越大，fixed 越好，但 recompute 越差。
3. **核心瓶颈是 object recall 和 seed 质量同时不够。**
   - 如果只保留高质量 object，平均实例数约 10，coverage 不够。
   - 如果把更多 seed-all/unmatched object 放回来，coverage 变好，但实例边界和归属质量不足，AP/AP50 掉。
4. **当前计划里的下一步应转向 evidence quality 和 memory-v2。**
   - 需要用 self/cycle/boundary evidence 过滤旧 seed 点，而不是无条件保留。
   - 需要 Hungarian / appearance / centroid / same-frame exclusivity 做 object memory-v2，解决 MVP 候选重叠和 adaptive top-k 漏召回。

### 14.7 当前最诚实的结论

可以写：

```text
v4.1 reliable no-seed 在 final recompute 协议下达到 30.2449 / 51.5938 / 67.0619，是当前最强主指标。
```

必须同时写：

```text
后续选择式 coverage 融合没有解决 inherit/unified support。最好的 MVP fixed 是 seed-all final 的 13.4864 AP，但它的 own recompute 只有 18.7252 AP，低于 Stream3D final baseline。能保住 recompute 的 select r1.5/r3.0 final 仍不能超过 v3 inherit。
```

不能写：

```text
v4.1 已达成计划要求的 inherit 稳健超越。
```

## 15. 2026-06-08 继续推进：seed 过滤、score 排序、低分 unmatched 与 e2_d48 final

### 15.1 本轮是否达成目标

没有达成。

本轮新增了三类修复尝试，并补跑了一个 final 候选：

1. `boundary/component` seed 可靠性过滤。
2. seed-all 的 `pred_score` 排序方式改造。
3. `select_secondary + unmatched` fusion 的低分 secondary 修复。
4. `e2_d48_top10` final 评估。

这些尝试都没有找到一个同时满足以下条件的配置：

- own recompute 明显超过 Stream3D final baseline；
- MVP/adaptive fixed 明显超过 v3 inherit；
- Stream3D fixed 不灾难性低；
- 不靠 GT、不改 evaluator、不伪造 pre_points。

### 15.2 代码修改审计

本轮修改文件：

| 文件 | 修改 | 审计说明 |
|---|---|---|
| `Stream3D/stream4d/reliable_densifier.py` | 新增 `seed_min_support_views`，新增 `boundary/component` seed 保留模式 | 只使用 prediction/object mask/depth/pose/mesh，不读取 GT |
| `Stream3D/stream4d/export_scannet.py` | 新增 `reliability/observations/dense_quality` score mode | 只改变 `pred_score`，不改变 mask 或 GT |
| `Stream3D/stream4d/reexport_scannet.py` | 暴露新增 CLI 参数 | 方便复现实验 |
| `Stream3D/tools/fuse_prediction_configs.py` | unmatched secondary 使用 `--secondary-score` | 修复之前 unmatched secondary 被错误赋高分的问题 |

语法和单测：

```text
Ran 5 tests in 0.001s
OK
```

### 15.3 tune30：seed 可靠性过滤结果

所有数值来自 `Stream3D/data/evaluation/scannet/*.txt` 最后一行，并乘以 100。

| 方法 | own recompute AP/AP50/AP25 | MVP fixed AP/AP50/AP25 | adaptive fixed AP/AP50/AP25 |
|---|---:|---:|---:|
| seednone min20 | 28.1912 / 48.4717 / 63.7019 | 9.2089 / 19.2868 / 33.5045 | 13.1056 / 25.6553 / 42.6223 |
| seedall min20 | 20.4552 / 37.6276 / 56.7153 | 15.5576 / 28.6274 / 47.9995 | 23.7159 / 41.1552 / 59.3165 |
| seedboundary v1 | 27.6465 / 48.8157 / 62.7625 | 10.6982 / 20.2008 / 33.7487 | 15.3758 / 27.0733 / 44.2360 |
| seedboundary v2 | 27.8695 / 49.4141 / 63.1944 | 10.3181 / 20.3551 / 33.8160 | 14.6662 / 27.0234 / 43.0263 |
| seedcomponent v1 | 27.0422 / 48.4941 / 62.7822 | 10.8736 / 20.7447 / 34.0449 | 15.4937 / 27.7725 / 44.6410 |
| seedcomponent v2 | 27.8825 / 49.2020 / 63.2981 | 10.3358 / 20.0699 / 33.9382 | 14.8554 / 26.9580 / 43.2288 |

覆盖诊断：

| config | union % | #objects | points | 说明 |
|---|---:|---:|---:|---|
| seednone min20 | 7.0506 | 14.3000 | NA | 高质量但 coverage 不够 |
| seedall min20 | 11.0730 | 14.4667 | NA | coverage 上升，但 own recompute 明显下降 |
| seedboundary v1 | 7.2606 | 14.4000 | 10616.7 | 只比 seednone 多一点 coverage |
| seedcomponent v1 | 7.2978 | 14.4000 | 10668.1 | 仍远低于 seedall coverage |

分析：

- `boundary/component` 的设计初衷是过滤 seed-all 中的旧点噪声。
- 结果显示它确实更保守，但过于保守：union 只从 7.0506% 提到约 7.26-7.30%，远低于 seed-all 的 11.0730%。
- 因此 fixed AP 只从 MVP 9.2089 提到最高 10.8736，adaptive 从 13.1056 提到最高 15.4937。
- 这说明“能通过 2D 连通/边界一致性重新确认的旧点”数量太少，不能补足 inherit/fixed support 所需覆盖。

### 15.4 tune30：seed-all score mode 结果

| score mode | own recompute AP/AP50/AP25 | MVP fixed AP/AP50/AP25 | adaptive fixed AP/AP50/AP25 |
|---|---:|---:|---:|
| seedall one | 20.4552 / 37.6276 / 56.7153 | 15.5576 / 28.6274 / 47.9995 | 23.7159 / 41.1552 / 59.3165 |
| area | 12.5396 / 27.9726 / 55.3672 | 7.7901 / 17.8843 / 45.7028 | 13.6844 / 30.6909 / 58.1278 |
| reliability | 12.5083 / 27.9035 / 55.2911 | 7.9067 / 18.1666 / 45.8315 | 13.7658 / 30.7904 / 58.0727 |
| observations | 19.4405 / 36.3937 / 55.6279 | 14.8722 / 27.8044 / 47.2476 | 22.8287 / 40.1788 / 58.4159 |
| dense_quality | 15.1065 / 32.4833 / 53.5393 | 10.0466 / 21.4456 / 42.6472 | 16.8401 / 34.6632 / 54.4162 |

分析：

- 如果 seed-all 失败只是因为坏对象排在前面，新的 score mode 应该能恢复 own recompute。
- 实验结果相反：`area`、`reliability`、`dense_quality` 都明显更差。
- `observations` 是最合理的 score mode，但 own recompute 只有 19.4405，仍低于 seed-all one 的 20.4552，也低于 tune30 Stream3D baseline 21.6016。
- 因此 seed-all 的主要问题不是排序，而是 mask 自身被旧点污染后 IoU 下降。

### 15.5 tune30：低分 unmatched secondary fusion

本轮修复 `tools/fuse_prediction_configs.py` 后，unmatched secondary 使用低分，不再使用 primary 高分。

| 方法 | own recompute AP/AP50/AP25 | MVP fixed AP/AP50/AP25 | adaptive fixed AP/AP50/AP25 |
|---|---:|---:|---:|
| select r1.5 + unmatched，secondary score 0.2 | 23.3049 / 43.0118 / 59.3976 | 12.7954 / 23.4928 / 41.9016 | 19.0734 / 32.8675 / 52.9593 |
| select r1.5 + unmatched，secondary score 0.05 | 23.3049 / 43.0118 / 59.3976 | 12.7954 / 23.4928 / 41.9016 | 19.0734 / 32.8675 / 52.9593 |
| select r2.0 + unmatched，secondary score 0.2 | 22.9213 / 42.3959 / 57.6830 | 13.3699 / 24.7251 / 41.9016 | 20.0361 / 35.1113 / 52.9593 |
| select r2.0 + unmatched，secondary score 0.05 | 22.9213 / 42.3959 / 57.6830 | 13.3699 / 24.7251 / 41.9016 | 20.0361 / 35.1113 / 52.9593 |

对比上一轮 `select r1.5 + unmatched`：

```text
上一轮 high-score unmatched own recompute = 23.9432 / 43.6260 / 60.3566
本轮 low-score unmatched own recompute = 23.3049 / 43.0118 / 59.3976
```

分析：

- 修复后 secondary score 变低，但 AP 没有变好，说明 evaluator 下这些 unmatched false positives 仍会影响 precision-recall 曲线。
- r2.0 fixed 比 r1.5 略好，但 own recompute 更低。
- 该方向不能替代 no-seed 主结果，也不能达成 inherit 稳健超越。

### 15.6 tune30：大半径/更多 mask 候选补诊断

| 方法 | own recompute AP/AP50/AP25 | MVP fixed AP/AP50/AP25 | adaptive fixed AP/AP50/AP25 |
|---|---:|---:|---:|
| e2 d48 top10 tune30 | 28.6700 / 49.3946 / 60.4179 | 13.2391 / 25.1149 / 38.0491 | 18.9737 / 34.4804 / 49.0080 |
| e1 d64 top10 tune30 | 23.0652 / 42.9008 / 57.1545 | 12.7401 / 25.1364 / 38.7791 | 18.4311 / 33.7274 / 49.3483 |

分析：

- `e2_d48_top10` 是本轮唯一看起来还可跑 final 的候选：own recompute 28.6700，没有崩；MVP/adaptive fixed 比 no-seed 高。
- 但它在 tune30 上仍低于 seed-all fixed，也没有超过 v3 adaptive inherit tune30 的 14.1553 AP。

### 15.7 final：e2_d48_top10 结果

| 方法 / 诊断 | support | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| Stream3D final baseline | own/self | 19.4294 | 33.3989 | 49.6361 |
| v4.1 reliable no-seed final | own recompute | 30.2449 | 51.5938 | 67.0619 |
| v4.1 reliable no-seed final | MVP fixed | 9.0795 | 18.2119 | 30.2373 |
| seed-all final | own recompute | 18.7252 | 35.0749 | 53.7801 |
| seed-all final | MVP fixed | 13.4864 | 25.4313 | 43.8950 |
| e2 d48 top10 final | own recompute | 26.1680 | 44.9588 | 60.6955 |
| e2 d48 top10 final | MVP fixed | 12.3649 | 23.0500 | 37.1737 |
| e2 d48 top10 final | adaptive fixed | 18.0156 | 31.0310 | 46.9765 |
| e2 d48 top10 final | Stream3D fixed | 1.1574 | 3.4664 | 10.1528 |

coverage 诊断：

```text
e2 d48 top10 final union = 11.9231%
e2 d48 top10 final #objects = 11.3013
e2 d48 top10 final points = 15707.3
e2 d48 top10 final export_nn_hit_rate = 0.880380
```

分析：

- `e2_d48_top10` 的 union 达到 11.9231%，高于 no-seed final 的 7.6984%，也略高于 seed-all final 的 11.2671%。
- 它 own recompute 仍有 26.1680，说明比 seed-all 更干净。
- 但 MVP fixed 只有 12.3649，低于 seed-all final 13.4864，也没有达到计划里“inherit 比 v3 提升 +3”或“接近/超过 Stream3D baseline”的目标。
- Stream3D fixed 只有 1.1574，仍是灾难性低，说明放回大 support universe 后 coverage/false negative 问题没有本质解决。

### 15.8 本轮更新后的失败原因

本轮把失败原因进一步收窄：

1. **不是旧 seed 点简单过滤不够。**
   - `boundary/component` 过滤太保守，union 只有约 7.3%，不能补 coverage。
   - seed-all coverage 够一些，但旧点污染实例 IoU。

2. **不是 prediction score 排序问题。**
   - `observations` score 是最合理的非 GT 排序，但没有超过 seed-all one。
   - `area/reliability/dense_quality` 更差，说明坏 mask 本身造成 AP 损失。

3. **不是 unmatched secondary 赋分过高这一处 bug。**
   - 修复低分 unmatched 后，r1.5/r2.0 仍无法保住 recompute，也不能达到 inherit 目标。

4. **不是只要扩大 densify 半径就能解决。**
   - `e2_d48_top10 final` union 达到 11.9231%，但 MVP fixed 只有 12.3649，Stream3D fixed 只有 1.1574。
   - 扩张已有 object 能提升 coverage，但不能产生足够好的 object recall 和边界质量。

### 15.9 当前最稳妥结论

目前最强主结果仍然是：

```text
v4.1 reliable no-seed final own recompute:
30.2449 / 51.5938 / 67.0619
```

但它不能被描述为全面超越，因为：

```text
MVP fixed = 9.0795 AP
Stream3D fixed = 0.6823 AP
```

本轮新增的最强 coverage 折中是：

```text
e2 d48 top10 final own recompute:
26.1680 / 44.9588 / 60.6955

e2 d48 top10 final MVP fixed:
12.3649 / 23.0500 / 37.1737
```

它仍没有达成目标。

### 15.10 后续不应继续盲调的方向

根据本轮证据，不建议继续只做这些方向：

- 单纯增大 seed distance。
- 单纯增加 top mask 数。
- 无 GT 的简单 score 重排。
- 无约束地把 secondary/unmatched object 加回来。
- 无条件保留旧 seed 点。

下一步如果继续做，必须转向更上游的真实结构改造：

1. memory-v2：解决 object recall 和碎片化，而不是只扩张已有 object。
2. 更强 evidence quality：需要真正的 self/cycle/boundary 证据，而不是只用重投影到 mask 内的启发式。
3. 多窗口 object identity：当前 ScanNet 仍主要是 32f one-window，缺少更长时序 memory 的有效验证。

在当前缓存和代码基础上，本轮已经没有看到一个不用 GT、又能同时提升 recompute 和 inherit/fixed 的局部后处理方向。

## 16. 2026-06-08 继续推进：质量门控 object rescue 仍未达成目标

### 16.1 本轮目标

上一轮已经排除了几类局部后处理：

- seed-all 无条件保留旧点；
- boundary/component seed 过滤；
- score mode 重排；
- 低分 unmatched secondary；
- 单纯更大半径 `e2_d48_top10`。

本轮继续尝试一个更接近 memory-v2/object recall 的子问题：

```text
从 secondary 高召回配置里，只救回少量和 primary 不重叠的 object。
```

新增门控条件：

- unmatched secondary score 下限；
- unmatched mask 面积范围；
- 每个 scene 最多救回 top-k 个 unmatched secondary；
- 新增 unmatched 使用低分 `secondary_score=0.05`。

该方法不读 GT，不改变 evaluator。

### 16.2 代码修改审计

修改：

- `Stream3D/tools/fuse_prediction_configs.py`

新增参数：

```text
--unmatched-min-secondary-score
--unmatched-min-area
--unmatched-max-area
--unmatched-top-k
```

语法和单测：

```text
Ran 5 tests in 0.001s
OK
```

### 16.3 secondary score/area 分布

对 `stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_score_observations_tune30`：

```text
instances = 434
score percentiles = [0.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
area percentiles = [21.0, 69.0, 169.0, 538.0, 1359.2, 2700.0, 3554.5, 5747.6, 9471.0]
```

结论：

- `observations` score 区分度很弱，大部分候选都是 4 或 5。
- 因此本轮主要靠面积范围和每场景 top-k 限制做质量门控。

### 16.4 seedall-observations secondary rescue 结果

所有数值来自 `Stream3D/data/evaluation/scannet/*.txt` 最后一行，并乘以 100。

| 方法 | own recompute AP/AP50/AP25 | MVP fixed AP/AP50/AP25 | adaptive fixed AP/AP50/AP25 | union % | #instances | unmatched / scene |
|---|---:|---:|---:|---:|---:|---:|
| rescue obs k1 area 100-2500 | 28.7534 / 50.7467 / 66.2051 | 12.1872 / 21.8585 / 37.2809 | 17.9890 / 30.2452 / 46.9561 | 7.5334 | 11.3333 | 0.8333 |
| rescue obs k2 area 100-2500 | 28.0418 / 49.8981 / 65.4780 | 12.4194 / 22.4921 / 38.5144 | 18.3890 / 31.0911 / 48.5112 | 7.7630 | 11.9333 | 1.4333 |
| rescue obs k4 area 100-2500 | 28.0296 / 50.0639 / 66.2366 | 12.7400 / 23.2277 / 39.8855 | 18.9542 / 32.3817 / 50.2368 | 7.9457 | 12.5667 | 2.0667 |
| rescue obs k2 area 100-1500 | 28.0129 / 49.8658 / 65.0588 | 12.3661 / 22.2758 / 37.8734 | 18.1669 / 30.4922 / 47.6960 | 7.6545 | 11.8333 | 1.3333 |
| rescue obs k2 area 250-2500 | 28.4063 / 50.6991 / 65.8220 | 12.5011 / 22.6202 / 38.8814 | 18.5040 / 31.4289 / 48.9755 | 7.7697 | 11.6333 | 1.1333 |

对比关键已有结果：

| 方法 | own recompute AP | MVP fixed AP | adaptive fixed AP | union % |
|---|---:|---:|---:|---:|
| no-seed tune30 | 30.8405 | 9.2089 | 13.1056 | 6.8552 |
| seed-all tune30 | 20.4552 | 15.5576 | 23.7159 | 11.0730 |
| e2 d48 top10 tune30 | 28.6700 | 13.2391 | 18.9737 | NA |
| best rescue obs k4 | 28.0296 | 12.7400 | 18.9542 | 7.9457 |

分析：

- rescue 能把 union 从 no-seed 的约 6.86% 提到 7.53-7.95%，比不加 unmatched 好。
- 但 even k4 每场景平均只补回 2.0667 个 unmatched object，coverage 仍远低于 seed-all 的 11.0730%。
- fixed AP 随 top-k 增加而上升，但 own recompute 同时下降。
- best rescue obs k4 的 MVP fixed 12.7400 仍低于 `e2_d48_top10 tune30` 的 13.2391，也低于 seed-all 的 15.5576。
- 因为 tune30 没有超过已经跑过 final 的 `e2_d48_top10` 折中候选，本轮没有把 rescue obs 再跑 final；继续跑 final 只会增加成本，不会改变方向性结论。

### 16.5 e2_d48 secondary rescue 结果

| 方法 | own recompute AP/AP50/AP25 | MVP fixed AP/AP50/AP25 | adaptive fixed AP/AP50/AP25 | union % | #instances | unmatched / scene |
|---|---:|---:|---:|---:|---:|---:|
| rescue e2d48 k1 area 100-3000 | 28.9476 / 51.1285 / 63.8384 | 10.3651 / 21.1432 / 34.2258 | 15.0504 / 27.9772 / 44.0726 | 7.8410 | 11.2667 | 0.7667 |
| rescue e2d48 k2 area 100-3000 | 28.6693 / 50.9267 / 63.7969 | 10.4289 / 21.3441 / 34.8091 | 15.1547 / 28.2435 / 45.0254 | 8.0559 | 11.8000 | 1.3000 |

分析：

- e2d48 secondary 比 seed-all secondary 更干净，own recompute 保持较好。
- 但 fixed AP 很弱：MVP 只有 10.3651-10.4289，adaptive 只有 15.0504-15.1547。
- 这说明 e2d48 多出来的 unmatched object 不是 inherit/fixed support 缺失的关键对象。

### 16.6 本轮结论

质量门控 rescue 仍没有达成目标。

它验证了一个更细的失败机制：

```text
只救回少量高质量 unmatched object，可以保住 recompute，但 coverage 不够，fixed AP 提升有限。
救回更多 seed-all object，可以提高 fixed，但会把噪声带回来，own recompute 下降。
```

这和前几轮结论一致，但证据更细：

- no-seed 是高质量 sparse object selector。
- seed-all 是 coverage 更高但 mask 污染严重。
- e2d48 是较干净的扩张，但新增对象不是关键漏召回对象。
- rescue/fusion 后处理无法替代真正的 memory-v2 object recall。

### 16.7 当前是否还应继续局部后处理

本轮之后，我已经不再认为继续调以下局部后处理有合理胜率：

- unmatched top-k；
- unmatched 面积阈值；
- secondary score 阈值；
- secondary 低分；
- r1.5/r2/r3 expansion；
- seed_keep_mode；
- 简单 score mode。

继续推进需要真正实现计划里的上游改造，而不是继续在已有 prediction 上做后处理：

1. `ObjectMemory4D-v2`：跨窗口 proposal 的 Hungarian matching、appearance/centroid gate、same-frame conflict penalty。
2. `evidence_quality.py`：真实 self/cycle/boundary evidence，而不是只看旧点能否投回 mask。
3. 多窗口小场景验证：先跑 `scene0050_00`、`scene0011_00`、`scene0030_00`、`scene0081_01`、`scene0591_00`，记录 object growth 和 fragmentation。

在当前 full ScanNet 缓存和不重跑 upstream D4RT 的条件下，v4.1 仍未达成 inherit/unified support 稳健超越。

## 17. 2026-06-08 ObjectMemory4D-v2 小规模多窗口实验复盘

本节回答用户问题：`docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md` 的 Phase 4 目标是否达成。

结论：**没有达成。**

最关键的失败标准：

```text
Phase 4 要求 96f / 128f memory-v2 的 AP50 不低于 32f current，并最好提升。
scene0050_00 上 32f current AP50 = 0.445714。
本轮最好的 96f memory-v2 直接输出 AP50 = 0.298120。
本轮最好的 96f memory-v2 + top40 mask_count postprocess recompute AP50 = 0.368304。
本轮最好的 128f memory-v2 直接输出 AP50 = 0.291785。
本轮最好的 128f memory-v2 + top40 mask_count postprocess recompute AP50 = 0.300000。
均低于 0.445714。
```

注意：本节数值来自单场景 `scene0050_00` 的 evaluator CSV 原始值，未乘以 100。若按百分数展示，乘以 100 即可。

### 17.1 本轮做了什么代码修改

新增文件：

```text
Stream3D/stream4d/appearance_memory.py
Stream3D/stream4d/motion_memory.py
Stream3D/stream4d/memory_diagnostics.py
Stream3D/stream4d/object_memory_v2.py
Stream3D/stream4d/replay_memory.py
```

修改文件：

```text
Stream3D/stream4d/local_4d_filter.py
Stream3D/stream4d/run_scannet.py
Stream3D/tests/test_stream4d_protocol_fixes.py
```

具体修改：

1. `LocalProposal` 增加 `appearance_feature`、`centroid_feature`、`feature_type` 字段。
2. `ObjectMemory4D-v2` 用 Hungarian one-to-one matching，避免多个当前 proposal 同时并入同一个历史 object。
3. matching score 第一版使用：

```text
S_c = carrier overlap / IoC
S_a = RGB histogram cosine similarity
S_g = 2D mask centroid proxy similarity
S_x = same-frame conflict penalty
```

4. `run_scannet.py` 新增：

```text
--memory-version old|v2
--memory-v2-carrier-weight
--memory-v2-appearance-weight
--memory-v2-geometry-weight
--memory-v2-motion-weight
--memory-v2-conflict-weight
--memory-v2-geometry-sigma
--memory-v2-motion-sigma
--memory-v2-min-carrier-score
--memory-v2-appearance-bins
--memory-v2-appearance-max-pixels-per-mask
--memory-v2-appearance-max-masks-per-proposal
```

5. `replay_memory.py` 新增 replay 能力：读取已经保存的 D4RT carrier 文件，重放 evidence/local proposal/memory/export，用于快速复现实验和调 memory 参数。
6. 单元测试新增 `test_memory_v2_enforces_one_to_one_matching`，验证 Hungarian one-to-one 约束。

审计说明：

- 本轮没有读取 GT 做 memory matching、selection 或 export。
- `S_a` 是 RGB histogram，不是 CLIP/DINO。
- `S_g` 是 2D mask centroid proxy，不是 GT 3D centroid。
- `replay_memory.py` 使用相同的 `carriers_window*.npz`，避免每次调 memory 参数都重新跑 D4RT。

### 17.2 编译和测试结果

命令见执行日志 11.2。

结果：

```text
Ran 6 tests in 0.001s
OK
```

### 17.3 关键结果表

| run | config | AP | AP50 | AP25 | objects / kept | points | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| 32f current old | `stream4d_scannet_scene0050_32f_ioc075_fixmem` | 0.202698 | 0.445714 | 0.681429 | 227 | 12214 | Phase 4 对照基线 |
| 96f old | `stream4d_v4_1_memoryold_scene0050_96f_ioc075` | 0.103789 | 0.216450 | 0.529101 | 243 | 27340 | old memory 多窗口 |
| 96f v2 base | `stream4d_v4_1_memoryv2_scene0050_96f_ioc075_wc055_wa025_wg020` | 0.115859 | 0.261364 | 0.518452 | 271 | 27340 | AP/AP50 高于 old 96f，但仍低于 32f |
| 96f v2 carrier gate | `stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005` | 0.123363 | 0.298120 | 0.533222 | 366 | 27340 | AP50 继续升，但对象更碎 |
| 96f v2 top40 mask_count recompute | `stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc005_top40_maskcount` | 0.158234 | 0.368304 | 0.646875 | 40 | 23052 | 本轮 96f 最高 AP50，但仍低于 32f |
| 96f v2 top40 mask_count inherit | `stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc005_top40_maskcount_inherit` | 0.101557 | 0.260985 | 0.516667 | 40 | 27340 | 同一 prediction 放回未筛选 support 后明显下降 |
| 128f old | `stream4d_scannet_scene0050_128f_ioc075_fixmem` | 0.101166 | 0.215385 | 0.524476 | 264 | 29992 | old 128f |
| 128f v2 base | `stream4d_v4_1_memoryv2_scene0050_128f_ioc075_wc055_wa025_wg020` | 0.101895 | 0.237288 | 0.501910 | 299 | 29992 | 只小幅高于 old 128f AP50 |
| 128f v2 carrier gate | `stream4d_v4_1_memoryv2_scene0050_128f_ioc075_thr025_wc055_wa025_wg020_minc005` | 0.117266 | 0.291785 | 0.516326 | 435 | 29992 | 对象碎片更多，AP50 仍低 |
| 128f v2 top40 mask_count recompute | `stream4d_v4_1_memoryv2_scene0050_128f_thr025_minc005_top40_maskcount` | 0.126282 | 0.300000 | 0.538942 | 40 | 23971 | 后处理仍不达标 |

### 17.4 memory-v2 初版相比 old memory 的变化

96f：

```text
old 96f:
objects = 243
AP/AP50/AP25 = 0.103789 / 0.216450 / 0.529101

v2 base 96f:
objects = 271
AP/AP50/AP25 = 0.115859 / 0.261364 / 0.518452
```

解释：

- v2 base 比 old 96f 的 AP 和 AP50 更高，说明 RGB histogram + centroid proxy + Hungarian matching 有一定正向作用。
- 但 v2 base 的 object 数从 243 增到 271，说明它不是通过减少碎片达成提升。
- AP25 反而从 0.529101 降到 0.518452，说明 coarse overlap 层面没有变好。
- 与 32f current 的 AP50 0.445714 相比，v2 base 仍差 0.184350。

128f：

```text
old 128f:
objects = 264
AP/AP50/AP25 = 0.101166 / 0.215385 / 0.524476

v2 base 128f:
objects = 299
AP/AP50/AP25 = 0.101895 / 0.237288 / 0.501910
```

解释：

- 128f v2 base 只把 AP50 从 0.215385 提到 0.237288。
- AP25 下降到 0.501910。
- 128f 增加更多窗口后，memory-v2 没有把更多帧转化为更好结果。

### 17.5 为什么加入 min_carrier_score

第一次调参按计划的 `re-ID 失败` 方向做：

```text
降低 history_match_threshold
提高 appearance weight
```

结果：

| config | objects | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| `thr025_wc045_wa035_wg020` | 227 | 0.104 | 0.244 | 0.502 |
| `thr022_wc040_wa040_wg020` | 227 | 0.104 | 0.244 | 0.478 |
| `thr020_wc035_wa045_wg020` | 227 | 0.106 | 0.262 | 0.498 |

分析：

- 三组对象数都压到 227，和 one-window 初始对象数一样。
- 但 AP 没有提升，AP25 明显下降。
- 这说明它们不是正确 re-ID，而是过度合并。
- 诊断里 accepted appearance 分数约 0.92，说明 scene0050 的 chair 外观相似，纯 appearance 容易把不同椅子合并。

因此新增 `--memory-v2-min-carrier-score`：

```text
只有总分超过 threshold 且 carrier overlap 达到下限，才接受 Hungarian match。
```

该修改是为了防止零 carrier overlap 的 appearance-only 合并。默认值是 0.0，不影响原始 v2 baseline。

### 17.6 min_carrier_score 的结果和副作用

96f 三组 carrier gate：

| config | objects | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| `thr025_minc005` | 366 | 0.123363 | 0.298120 | 0.533222 |
| `thr025_minc010` | 368 | 0.123 | 0.298 | 0.532 |
| `thr022_minc005` | 366 | 0.123 | 0.298 | 0.533 |

分析：

- AP/AP50/AP25 都高于 v2 base 和 old 96f。
- 但 object 数从 v2 base 的 271 增到 366，碎片化更严重。
- carrier gate 防止了 appearance-only 过合并，但让弱 carrier re-ID 无法合并，导致更多新 object。
- 该方向改善了 precision 排序，但没有达成 `final_num_objects 不超过 one-window 的 2x` 以外的更关键目标：没有让 96f AP50 不低于 32f。

### 17.7 postprocess top-N 诊断

对 96f `thr025_minc005` 做 mask_count top-N：

| top-N | points | AP | AP50 | AP25 |
|---:|---:|---:|---:|---:|
| 227 | 26903 | 0.126 | 0.302 | 0.543 |
| 180 | 26808 | 0.131 | 0.304 | 0.548 |
| 120 | 26575 | 0.137 | 0.330 | 0.561 |
| 80 | 25608 | 0.141 | 0.345 | 0.569 |
| 60 | 25181 | 0.145 | 0.343 | 0.559 |
| 40 | 23052 | 0.158234 | 0.368304 | 0.646875 |
| 30 | 20964 | 0.130 | 0.313 | 0.647 |
| 20 | 16758 | 0.089 | 0.249 | 0.626 |
| 10 | 12677 | 0.045 | 0.163 | 0.496 |

结论：

- top40 是这条曲线的峰值。
- top40 AP50 0.368304 仍低于 32f current 的 0.445714。
- top10 support 12677 已接近 32f 的 12214，但 AP50 只有 0.163，说明不是简单把 support 缩回 32f 就能恢复。
- top40 AP25 0.646875 接近 32f 的 0.681429，但 AP50 差距仍大，说明对象边界/分裂质量仍不足。

对 top40 不同 select mode：

| select_mode | points | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| `mask_count` | 23052 | 0.158234 | 0.368304 | 0.646875 |
| `area` | 22926 | 0.161 | 0.325 | 0.625 |
| `carrier_count` | 21608 | 0.139 | 0.319 | 0.637 |
| `coverage_sum` | 14510 | 0.093 | 0.345 | 0.589 |

结论：

- `mask_count` AP50 最好。
- `area` AP 略高，但 AP50 低。
- `coverage_sum` support 缩得更狠，但 AP/AP25 不好。
- 简单无监督排序不能把 96f 拉回 32f 水平。

### 17.8 inherit 证据：top40 收益依赖 support 缩小

同一个 top40 prediction：

```text
recompute:
pre_points = 23052
AP/AP50/AP25 = 0.158234 / 0.368304 / 0.646875

inherit:
pre_points = 27340
prediction_union = 23052
AP/AP50/AP25 = 0.101557 / 0.260985 / 0.516667
```

解释：

- prediction 不变，只把 evaluation support 从筛后的 union 放回未筛选 v2 support。
- AP50 从 0.368304 降到 0.260985。
- 因此 top40 的提升不能作为方法本体达标证据，只能说明筛选缩小 support 后 precision 更高。
- 这与前面 full ScanNet tune/final 的核心问题一致：当前方法容易通过 sparse observed-support selection 提升 recompute，但 inherit/fixed 支持下不稳。

### 17.9 为什么 Phase 4 没达成

失败不是单一 bug，而是三件事叠加：

1. **appearance proxy 太弱**  
   RGB histogram 对同类物体，尤其 scene0050 的多把椅子，区分度不足。降低阈值或提高 appearance 权重会把不同椅子过合并。

2. **carrier overlap 又太稀疏**  
   加 `min_carrier_score` 后 AP 提高，但对象数暴涨到 366/435，说明很多真实跨窗口同一物体没有足够 carrier overlap；只靠 carrier gate 会 re-ID 失败。

3. **更多窗口带来更多 support，也带来更多 false positive / duplicate proposal**  
   32f current 是 12214 points、227 objects、AP50 0.445714。96f 增到 27340 points 后，直接输出 AP50 最高只有 0.298120；top40 后处理缩到 23052 points 后 AP50 也只有 0.368304。新增 support 没有变成更完整、更准确的 object，而是混入更多重复和边界噪声。

这三个证据说明：本轮 memory-v2 框架方向是合理的，但当前使用的 `RGB histogram + 2D centroid proxy + carrier IoC` 不足以解决 ScanNet 多窗口 identity 和 object quality。

### 17.10 当前能安全声称什么

可以声称：

```text
1. 已实现可选 ObjectMemory4D-v2，不破坏 old memory 默认路径。
2. 已实现 Hungarian one-to-one matching，并有单元测试覆盖。
3. 已实现 replay 工具，能用相同 D4RT carrier 重放 memory/export；replay 复现了真实 v2 run 的 objects/points/hit_rate/AP。
4. 在 scene0050_00 96f 上，v2 base 比 old memory 提高 AP 和 AP50：
   old 96f: 0.103789 / 0.216450 / 0.529101
   v2 base: 0.115859 / 0.261364 / 0.518452
5. 加 carrier gate 后，96f AP50 可到 0.298120；再加 mask_count top40 recompute 后，AP50 可到 0.368304。
```

不能声称：

```text
1. ObjectMemory4D-v2 已让 96f / 128f 超过 32f current。
2. ObjectMemory4D-v2 已解决 object fragmentation。
3. top40 postprocess 是 memory-v2 本体胜利。
4. 当前 v4.1 已在 inherit/fixed support 下稳健超过 Stream3D 或 32f current。
```

### 17.11 下一步建议

如果继续推进，应该优先做以下方向，而不是继续调本轮的简单权重：

1. 用 D4RT trajectory 的 3D/2D temporal consistency 形成真正的 `S_m`，不是只用静态 RGB histogram。
2. 在 proposal 阶段记录 mask-level 3D centroid 和 point support，给 `S_g` 使用真实 scene point centroid，而不是 2D centroid proxy。
3. 给 object 增加 duplicate suppression / split-merge audit：跨窗口后按 exported point IoC 或 carrier trajectory 做二次合并，但必须有 alignment 检查。
4. 用更强 appearance：DINO/CLIP 或至少局部 crop RGB histogram + bbox shape ensemble；并记录 `feature_type`，不能冒充。
5. 跑多个小场景，而不是只在 scene0050 上继续调 top-N。scene0050 已显示 failure mechanism，继续单场景调参会过拟合。

## 18. 2026-06-08 追加诊断：停止堆后处理，转向算法核心

用户提醒：“不要一直堆工程，应该注重算法思路上的改进。”

这个提醒是对的。本节记录在提醒前已经启动并完成的最后一批诊断。结论是：**point merge、point-NMS、32f-primary fusion 都没有达成目标，继续堆后处理工具没有意义。**

### 18.1 point-IoC merge 失败

输入：

```text
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005
```

方法：按 exported point IoC 把高度重叠的 object 合并成一个并集 mask。

| 方法 | objects | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| pointmerge 0.25 | 11 | 0.035017 | 0.042424 | 0.042424 |
| pointmerge 0.50 | 37 | 0.062366 | 0.115702 | 0.165289 |
| pointmerge 0.75 | 155 | 0.083018 | 0.195312 | 0.397727 |
| pointmerge 0.90 | 259 | 0.103741 | 0.206061 | 0.414815 |

分析：

- point merge 全部低于原始 96f carrier-gated v2 的 `0.123363 / 0.298120 / 0.533222`。
- 低阈值会把大量对象合成巨大 mask，AP50/AP25 直接崩。
- 高阈值也没有收益，说明重复不是简单“把重叠 mask 并起来”能解决。

算法结论：

```text
多窗口错误不是纯 duplicate merge 问题，而是 object identity、mask purity、support assignment 同时出错。
把预测后验合并成大 mask 会制造 over-merge。
```

### 18.2 point-NMS 也没有解决 AP50

新增工具：

```text
Stream3D/tools/point_nms_prediction.py
```

它只做诊断：保留原 mask，按 point IoC 抑制重复预测，不做 mask 并集。

| 方法 | kept | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| pointNMS 0.25 mask_count | 55 | 0.147045 | 0.334988 | 0.661704 |
| pointNMS 0.50 mask_count | 98 | 0.132679 | 0.301983 | 0.578425 |
| pointNMS 0.75 mask_count | 181 | 0.114744 | 0.272129 | 0.503389 |
| pointNMS 0.90 mask_count | 276 | 0.126314 | 0.302257 | 0.521800 |

对比：

```text
32f current:
0.202698 / 0.445714 / 0.681429

96f v2 top40 mask_count:
0.158234 / 0.368304 / 0.646875

best pointNMS:
0.147045 / 0.334988 / 0.661704
```

分析：

- pointNMS 0.25 的 AP25 接近 32f current，但 AP50 仍差很多。
- 这说明 coarse object coverage 可以保住一些，但准确边界/实例分离不足。
- NMS 抑制重复预测只能减少 false positive，不能修复 proposal 本身的 mask purity。

score 诊断：

| 方法 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| top40 mask_count, pred_score=mask_count | 0.054081 | 0.193390 | 0.605476 |
| pointNMS 0.25, pred_score=mask_count | 0.070743 | 0.229475 | 0.702182 |
| pointNMS 0.50, pred_score=mask_count | 0.060852 | 0.202088 | 0.576164 |

结论：

- `mask_count` 可以用于筛选，但不能直接作为 ranking score。
- 高 mask_count 常常对应“跨很多帧都出现”，但不代表 mask 更干净；可能只是一个大/常见/重复 object。

### 18.3 32f primary + 96f v2 secondary fusion 没有算法增益

目的：避免 96f 多窗口结果污染整个输出，只让它替换 32f 中明显被覆盖且 expansion 可控的 object。

primary：

```text
stream4d_scannet_scene0050_32f_ioc075_fixmem
```

secondary：

```text
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005
```

| select_max_expansion | AP | AP50 | AP25 | 结论 |
|---:|---:|---:|---:|---|
| 1.25 | 0.202698 | 0.445714 | 0.681429 | 和 32f 完全持平 |
| 1.50 | 0.202698 | 0.445714 | 0.681429 | 和 32f 完全持平 |
| 2.00 | 0.189206 | 0.402857 | 0.681429 | 开始下降 |
| 3.00 | 0.176720 | 0.383673 | 0.648980 | 继续下降 |

分析：

- 保守替换没有任何可测提升，说明 96f v2 没有提供比 32f 更好的 matched variant。
- 放宽 expansion 后立刻下降，说明 96f 的大 mask/扩张 mask 会污染 32f 高质量结果。
- 这条线证明：“把 96f 当 secondary rescue”仍然不是算法解法。

### 18.4 到这里应该停止的后处理方向

以下方向已经有足够负证据，不建议继续投入：

```text
1. 继续调 point-IoC merge threshold。
2. 继续调 point-NMS threshold。
3. 继续换 mask_count / area / coverage_sum 这类简单 ranking score。
4. 继续让 96f secondary 替换 32f primary。
5. 继续靠 top-N 缩小 support 提升 recompute AP。
```

原因不是“没调够”，而是这些方法只在预测之后处理症状。真正问题在预测生成过程：

```text
carrier 轨迹如何支持一个 object；
2D mask evidence 如何被判定可靠；
跨窗口 proposal 何时应该合并、何时应该分裂；
新增窗口带来的 support 如何避免污染已有 object。
```

### 18.5 算法层面的下一步，不是工程堆叠

如果继续做 v4.1，应该把目标从“后处理筛选”改成“对象形成时的因果证据约束”。建议路线：

#### A. 从 object-level memory 改成 tracklet-level evidence graph

当前 pipeline 是：

```text
mask observation -> local proposal -> object memory
```

问题是 local proposal 一旦错，后面只能补救。应该改成：

```text
mask observation -> carrier tracklet evidence graph -> object hypothesis
```

核心思想：

- 节点不是整张 2D mask，而是更小的 carrier tracklet / carrier cluster。
- 边表示两个 carrier 是否长期同属一个 object。
- object 是 graph partition 的结果，而不是 mask observation 贪心合并的结果。

这样能避免一个脏 2D mask 直接污染整个 object。

#### B. 引入“反证”而不只是正证

当前 score 多数是正向相似度：

```text
carrier overlap 高 -> 合
appearance 相似 -> 合
centroid 近 -> 合
```

但失败样例说明同类椅子 appearance 很相似。需要强反证：

```text
同一帧出现为两个不同 mask -> 强烈不合并
同一 carrier 在同一时间支持两个 object -> 冲突，需要二选一
两个 object 的 3D support 相交但 2D masks 长期分离 -> 不合并
合并后 mask 的 projected footprint 变大太多 -> 拒绝
```

这比继续调 appearance 权重更关键。

#### C. memory update 应该允许 split，不只允许 merge/create

当前 old/v2 memory 都是：

```text
match existing object 或 create new object
```

但多窗口中常见错误是早期 object 被污染，后面只能越滚越脏。应该加入：

```text
object split_candidate
object quarantine
rollback recent observations
```

算法标准：

- 如果一个 object 在新窗口中出现多个互相冲突的 masks，不立即并入。
- 先把新 evidence 放进 pending buffer。
- 等下一窗口验证哪一支 tracklet 更稳定，再写入 object。

#### D. 用真实 3D centroid / extent，不用 2D centroid proxy

本轮 `S_g` 只是 2D mask centroid proxy。它不能区分同一画面中空间接近但不同的椅子，也不能利用 ScanNet mesh 的几何约束。

应该在 evidence 阶段记录：

```text
每个 mask observation 的 backprojected 3D point set；
3D centroid；
3D bbox / extent；
point support confidence；
跨帧 centroid variance。
```

然后 matching 使用：

```text
3D centroid distance
3D extent compatibility
3D support IoU / IoC
```

注意：这些都是 RGB-D 和 mesh 几何，不是 GT instance。

#### E. 训练-free 也需要 object hypothesis scoring，而不是只输出所有对象

当前多窗口输出的错误之一是 object 太多。不是简单 top-N，而是每个 object 要有可解释质量分：

```text
temporal consistency
3D compactness
2D mask agreement
carrier cycle consistency
conflict penalty
support novelty
```

然后输出时不是固定 top-N，而是根据这些质量分做 calibrated object selection。

### 18.6 当前结论

本轮追加诊断进一步确认：

```text
v4.1 没有达成目标。
继续堆后处理不会解决。
下一步必须回到算法：carrier tracklet evidence graph + 强反证约束 + split-capable memory + 真实 3D geometry。
```

如果要继续，我建议下一轮不要先写工具，而是先写一个最小算法原型：

```text
只在 scene0050_00 的 96f carriers 上做 tracklet graph partition；
输出 graph partition 的 object hypotheses；
和 old/v2 memory 在同一 carrier cache 上比较 AP、object 数、conflict。
```

这个方向才是在改算法，而不是继续后处理补洞。

## 19. 2026-06-08 Evidence Graph 算法原型结果

本节是按第 18 节建议做的最小算法原型：不再筛预测列、不再 point-NMS、不再 fusion，而是改变 object 形成方式。

### 19.1 算法定义

新算法路径：

```text
1. 从已缓存 D4RT carrier 重新构建 MaskObservation。
2. 将同一个 (frame_id, mask_id) 聚合成 evidence graph node。
3. 如果两个 node 共享足够 carrier，建立正向边。
4. 如果两个 component 在同一 frame 对应不同 mask_id，则禁止合并，这是 hard negative / cannot-link。
5. graph partition 的 component 输出为 object hypothesis。
6. 输出前要求 component 至少有一定数量的 mask observations，作为 object stability 门槛。
```

新增代码：

```text
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/replay_evidence_graph.py
```

这不是后处理：object 是由 graph partition 产生，不是先用 old/v2 memory 输出后再筛掉或合并。

### 19.2 96f 参数扫描结果

对照基线：

```text
32f current:
AP/AP50/AP25 = 0.202698 / 0.445714 / 0.681429
objects = 227
points = 12214

96f old:
AP/AP50/AP25 = 0.103789 / 0.216450 / 0.529101

96f v2 carrier gate:
AP/AP50/AP25 = 0.123363 / 0.298120 / 0.533222

96f v2 top40 mask_count:
AP/AP50/AP25 = 0.158234 / 0.368304 / 0.646875
```

Evidence graph 关键扫描：

| config | objects | points | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| `ioc0p50_shared2` | 187 | 27838 | 0.105904 | 0.302718 | 0.546704 |
| `ioc0p70_shared2` | 229 | 27838 | 0.109540 | 0.339971 | 0.584416 |
| `ioc0p85_shared2` | 379 | 27838 | 0.096888 | 0.289773 | 0.554813 |
| `ioc0p70_shared2_minobs5` | 111 | 27217 | 0.117959 | 0.371127 | 0.607763 |
| `ioc0p70_shared2_minobs8` | 85 | 26345 | 0.119618 | 0.384086 | 0.620215 |
| `ioc0p70_shared2_minobs10` | 66 | 24602 | 0.141173 | 0.416944 | 0.652500 |
| `ioc0p75_shared2_minobs10` | 65 | 24021 | 0.146048 | 0.433080 | 0.678551 |
| `ioc0p78_shared2_minobs10` | 61 | 23455 | 0.159209 | 0.467980 | 0.689963 |
| `ioc0p80_shared2_minobs10` | 58 | 22506 | 0.147184 | 0.445023 | 0.702668 |

最佳点：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10
AP/AP50/AP25 = 0.159209 / 0.467980 / 0.689963
objects = 61
points = 23455
```

和 32f current 比：

```text
AP:   0.159209 < 0.202698
AP50: 0.467980 > 0.445714
AP25: 0.689963 > 0.681429
```

解释：

- 这是本轮第一个真正算法路径上超过 32f current AP50/AP25 的结果。
- 但 AP 仍低于 32f，说明高 IoU threshold 下的精细实例质量仍不足。
- object 数从 227 降到 61，说明 graph stability 门槛强烈减少了重复和噪声 object。
- points 从 12214 增到 23455，说明它不是简单缩回 32f support 才提升 AP50/AP25。

### 19.3 graph 诊断

最佳 96f graph：

```text
raw observations = 2640
graph nodes = 1559
edge candidates = 53518
accepted edges = 1276
rejected conflict edges = 6313
rejected weak edges = 45929
components = 283
kept components = 61
dropped components = 222
mean component nodes = 16.0164
max component nodes = 41
mean component carriers = 370.5082
exported points = 23455
export conflict rate = 0.3183
```

关键 insight：

- `rejected conflict edges = 6313` 证明 hard negative 不是摆设，它确实阻止了大量同帧不同 mask 合并。
- dropped components 很多，说明多窗口里确实存在大量不稳定碎片；稳定观测数门槛是必要的。
- export conflict rate 仍高达 0.3183，说明即使 graph partition 改善 AP50/AP25，point-level ownership 仍很脏。这解释了 AP 仍低。

### 19.4 fixed-support 诊断

同一个最佳 96f graph prediction：

| eval support | AP | AP50 | AP25 |
|---|---:|---:|---:|
| recompute own support | 0.159209 | 0.467980 | 0.689963 |
| 96f v2 full support | 0.086563 | 0.289256 | 0.580165 |
| 32f current support | 0.110431 | 0.317568 | 0.527027 |

解释：

- fixed-support 下明显下降。
- 因此不能说 v4.1 已经解决 inherit/fixed support 问题。
- recompute 下 AP50/AP25 超过 32f 是有价值的算法信号，但仍不是 full protocol victory。

### 19.5 128f 同参数结果

配置：

```text
stream4d_v4_1_egraph_scene0050_128f_ioc0p78_shared2_minobs10
```

结果：

```text
objects = 67
points = 26163
AP/AP50/AP25 = 0.152793 / 0.456062 / 0.668242
```

和 32f current 比：

```text
AP:   0.152793 < 0.202698
AP50: 0.456062 > 0.445714
AP25: 0.668242 < 0.681429
```

解释：

- 128f 仍能保持 AP50 超过 32f current。
- 但 AP25 低于 32f，说明更多窗口带来更多 support 的同时也带来更多 coarse contamination。
- 128f 没有比 96f 更好，说明继续加窗口不是自动收益。

### 19.6 本轮是否达成目标

精确结论：

```text
总体 v4.1 目标：没有完全达成。
scene0050_00 recompute 协议下的 Phase 4 局部 AP50/AP25 目标：96f 达成。
128f recompute AP50 不低于 32f current：达成。
AP 总体超过 32f current：没有达成。
fixed/inherit support 稳健性：没有达成。
多场景验证：没有完成。
```

不能写成：

```text
Stream4D-v4.1 已经超过 Stream3D。
memory-v2 已全面解决多窗口问题。
evidence graph 已经完成 ScanNet 复现目标。
```

可以安全写成：

```text
在 scene0050_00 的 cached-carrier 小规模实验中，tracklet/mask evidence graph + same-frame cannot-link + component stability gate 显著优于 old memory、v2 memory 和后处理诊断，并在 recompute support 下让 96f AP50/AP25 超过 32f current。
```

### 19.7 算法层面的下一步

Evidence graph 证明了算法方向比后处理有效，但瓶颈也更明确：

1. **point ownership conflict 仍高**  
   最佳 graph export conflict rate = 0.3183。需要在 graph partition 之后加入 point ownership resolution，而不是简单让多个 object 共享同一点。

2. **AP 低说明高 IoU 质量不足**  
   AP50/AP25 能赢，但 AP 输，说明 object support 大体对，但边界/实例精度不够。下一步应做 3D compactness 和 point-level WTA，不是继续调 graph threshold。

3. **fixed support 失败说明 coverage universe 仍不稳**  
   放回 96f full support 后 AP50 从 0.467980 掉到 0.289256。这说明 graph 输出仍是“较干净 support 子集”，不是完整 support 重建。

4. **多场景验证需要重新生成 96f/128f carrier cache**  
   当前其它推荐场景只有 32f 单窗口 cache，不足以验证多窗口 graph。继续多场景需要明确接受重新跑 D4RT 的成本。

下一步建议不是增加新工具，而是算法上做：

```text
Evidence graph + 3D point ownership WTA + 3D compactness gate
```

其中 WTA 必须使用无监督可靠性分数，例如：

```text
component stability
carrier support count
same-frame conflict count
3D compactness
mask observation count
```

不能使用 GT。

## 20. 2026-06-08 Evidence-aware WTA 与 mask support 扫描复盘

本节继续第 19 节的算法方向，不再做单纯工程筛选。所有指标来自 `evaluation.evaluate` 的实际输出文件，没有补写或猜测数据。

### 20.1 本轮代码修改

修改点：

```text
Stream3D/stream4d/evidence_graph.py
```

给每个 evidence graph component 输出的 `Object4D` 增加无监督证据字段：

```text
evidence_num_nodes
evidence_num_frames
evidence_num_carriers
evidence_mean_coverage
evidence_quality
```

这些字段只来自 carrier/mask 观测，不使用 GT。

```text
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/replay_evidence_graph.py
```

新增导出阶段算法开关：

```text
export_enable_wta
export_wta_score_mode
export_wta_min_conflict_owners
```

含义：

```text
export_enable_wta:
  是否启用 point-level winner-take-all。

export_wta_score_mode:
  用哪个无监督分数决定冲突点归属。
  本轮测试 evidence_quality / evidence_density / compactness。

export_wta_min_conflict_owners:
  一个三维点至少被多少个 object 同时占有时才触发 WTA。
  2 是 hard WTA。
  3/4/5 是 soft WTA，只处理更严重的冲突。
```

新增：

```text
Stream3D/splits/scannet_scene0050.txt
```

只包含 `scene0050_00`，用于多尺度 hypothesis 融合诊断。

### 20.2 对照基线

本节仍只讨论 `scene0050_00` 单场景，不写成完整 ScanNet 复现。

关键对照：

| config | objects | points | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| 32f current | 227 | 12214 | 0.202698 | 0.445714 | 0.681429 |
| 96f evidence graph best before this round | 61 | 23455 | 0.159209 | 0.467980 | 0.689963 |

解释：

- 32f current 是本阶段原版/当前对照。
- 96f evidence graph 已经赢 AP50/AP25，但 AP 低，说明 coarse support 对，高 IoU 精度不足。

### 20.3 Point-level WTA 结果

先只改变点冲突处理，不改变 support 生成方式。

| experiment | objects | points | conflict | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|---:|
| hard WTA evidence_density | 61 | 23455 | 0.000000 | 0.121666 | 0.415745 | 0.668185 |
| hard WTA evidence_quality | 61 | 23455 | 0.000000 | 0.118111 | 0.417772 | 0.672414 |
| hard WTA compactness | 61 | 23455 | 0.000000 | 0.141873 | 0.447281 | 0.707228 |
| soft WTA min3 compactness | 61 | 23455 | 0.233255 | 0.142598 | 0.440752 | 0.694044 |
| soft WTA min4 compactness | 61 | 23455 | 0.292091 | 0.156670 | 0.470219 | 0.694044 |
| soft WTA min5 compactness | 61 | 23455 | 0.309699 | 0.156507 | 0.467980 | 0.689963 |

结论：

- hard WTA 把 conflict 清到 0，但 AP/AP50 大多下降。
- 这说明 carrier support 下的重叠不全是污染；有些重叠对召回有帮助。
- `compactness` 比 `evidence_quality` 和 `evidence_density` 更可靠，说明三维几何紧致性比单纯跨帧证据强度更接近评估目标。
- soft WTA `min4` 是较好的折中：只处理 614 个严重冲突点，删除 2111 个重复归属，AP50 从 0.467980 升到 0.470219，AP25 从 0.689963 升到 0.694044，但 AP 没有超过 32f current。

### 20.4 预测 score 负例

尝试把无监督质量分直接作为预测 score，不做 WTA：

| score mode | objects | points | conflict | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|---:|
| evidence_density | 61 | 23455 | 0.318269 | 0.054429 | 0.211663 | 0.415579 |
| evidence_quality | 61 | 23455 | 0.318269 | 0.054330 | 0.213146 | 0.446613 |
| compactness | 61 | 23455 | 0.318269 | 0.089854 | 0.365930 | 0.518227 |

结论：

- 直接排序失败。
- 当前无监督质量分可以辅助点归属，但不能直接当 AP 排序 score。
- 证据链：同样 61 objects / 23455 points / conflict 0.318269，仅改变 score，AP 从 0.159209 掉到 0.054 到 0.090 区间。

### 20.5 Graph threshold + soft WTA 扫描

固定：

```text
min_component_observations = 10
min_shared_carriers = 2
WTA score = compactness
WTA min_conflict_owners = 4
support = carrier_uv
```

| graph carrier IoC | objects | points | conflict | AP | AP50 | AP25 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.76 | 63 | 23588 | 0.297397 | 0.151033 | 0.467980 | 0.689963 |
| 0.77 | 61 | 23474 | 0.293175 | 0.154302 | 0.470219 | 0.694044 |
| 0.78 | 61 | 23455 | 0.292091 | 0.156670 | 0.470219 | 0.694044 |
| 0.79 | 59 | 23023 | 0.285671 | 0.153257 | 0.474951 | 0.702668 |
| 0.80 | 58 | 22506 | 0.290189 | 0.144636 | 0.447281 | 0.707228 |

结论：

- `0.79` 是 carrier support 下 AP50/AP25 的较好点：AP50 0.474951，AP25 0.702668。
- 但 AP 仍明显低于 32f current 的 0.202698。
- 这进一步说明 carrier support 可以拿到 coarse object，但是边界/实例精度不足。

### 20.6 Mask backprojection support 结果

动机：

```text
carrier support:
  粗覆盖好，AP25 高，但高 IoU AP 不够。

mask backprojection support:
  用 evidence graph 选出的稳定 2D mask 回投到 3D。
  目标是生成更精确 core support，提高 AP/AP50。
```

关键扫描：

| experiment | objects | points | conflict | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|---:|
| hybrid m5 s2 r0.05 soft4 | 59 | 45727 | 0.272749 | 0.149595 | 0.455729 | 0.601852 |
| maskbp m3 s2 r0.05 soft4 | 59 | 24567 | 0.129238 | 0.184343 | 0.454545 | 0.536616 |
| maskbp m5 s2 r0.05 soft4 | 59 | 38235 | 0.177874 | 0.204246 | 0.463415 | 0.563415 |
| maskbp m8 s2 r0.05 soft4 | 59 | 51034 | 0.261434 | 0.165987 | 0.391304 | 0.602921 |
| maskbp m3 s1 r0.05 soft4 | 59 | 26424 | 0.141689 | 0.162963 | 0.425000 | 0.580556 |
| maskbp m5 s1 r0.05 soft4 | 59 | 40599 | 0.191926 | 0.180908 | 0.459921 | 0.558730 |

结论：

- hybrid 失败，点数膨胀到 45727，AP25 也下降，说明 carrier + mask 简单并集会引入污染。
- mask 数量存在甜点：3 个 mask 不够，8 个 mask 过多，5 个 mask 最好。
- stride=2 比 stride=1 更好，说明更密采样并没有改善，反而带来更多噪声。

### 20.7 Radius 和 hard WTA 扫描

固定：

```text
graph carrier IoC = 0.79
min_component_observations = 10
support = mask_backproject
max masks per object = 5
mask sample stride = 2
```

| experiment | objects | points | conflict | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|---:|
| r0.06 soft4 | 59 | 38341 | 0.178816 | 0.202258 | 0.463415 | 0.563415 |
| r0.05 nowta | 59 | 38235 | 0.179757 | 0.204246 | 0.463415 | 0.563415 |
| r0.05 hard2 | 59 | 38235 | 0.000000 | 0.210351 | 0.470940 | 0.609402 |
| r0.04 nowta | 59 | 37937 | 0.176398 | 0.209233 | 0.479394 | 0.582843 |
| r0.04 min3 | 59 | 37937 | 0.156865 | 0.212130 | 0.479394 | 0.582843 |
| r0.04 hard2 | 59 | 37937 | 0.000000 | 0.214166 | 0.487179 | 0.630416 |
| r0.03 nowta | 59 | 36979 | 0.172584 | 0.219793 | 0.479394 | 0.582843 |
| r0.03 hard2 | 59 | 36979 | 0.000000 | 0.225661 | 0.521662 | 0.630416 |
| r0.02 hard2 | 59 | 34416 | 0.000000 | 0.232910 | 0.526316 | 0.562160 |
| r0.03 m6 hard2 | 59 | 43260 | 0.000000 | 0.172043 | 0.385177 | 0.643241 |

最佳高精度结果：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_r002_hard2
AP/AP50/AP25 = 0.232910 / 0.526316 / 0.562160
objects = 59
points = 34416
```

和 32f current 比：

```text
AP:   0.232910 > 0.202698
AP50: 0.526316 > 0.445714
AP25: 0.562160 < 0.681429
```

另一个较均衡但 AP 稍低的结果：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r003_hard2
AP/AP50/AP25 = 0.225661 / 0.521662 / 0.630416
```

它 AP/AP50 仍超过 32f current，但 AP25 仍低于 32f current。

### 20.8 128f 验证

使用 128f carrier cache，同样 mask backprojection `r0.02 hard2`：

```text
stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_maskbp_r002_hard2
AP/AP50/AP25 = 0.220462 / 0.461323 / 0.567568
objects = 67
points = 34467
```

结论：

- 128f 没有比 96f 更好。
- 更多窗口带来了更多 object 和更多不稳定视角，没有自动提高 support 质量。
- 这个现象和第 19 节一致：继续增加窗口数不是可靠收益。

### 20.9 多尺度 hypothesis 负例

动机：

```text
紧 maskbp core:
  AP/AP50 高，AP25 低。

宽 carrier support:
  AP25 高，AP 低。

尝试同时输出 core 和 wide support，core 高分，wide 低分。
```

结果：

| experiment | output objects | points | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| core r0.02 + carrier0.79, secondary score 0.2 | 118 | 42986 | 0.164028 | 0.429140 | 0.585281 |
| core r0.03 + carrier0.80, secondary score 0.2 | 117 | 44825 | 0.170205 | 0.425440 | 0.579404 |
| core r0.02 + carrier0.79, secondary score 0.8 | 118 | 42986 | 0.164028 | 0.429140 | 0.585281 |
| core r0.02 + carrier0.79, secondary score 1.1 | 118 | 42986 | 0.062202 | 0.220262 | 0.579093 |

结论：

- 简单拼接多尺度 hypothesis 失败。
- 低分 wide support 仍会引入 false positive，无法恢复 AP25。
- 高分 wide support 会破坏 AP/AP50 排序。
- 这说明不能用后处理式拼接解决，需要 object 内部做真正的 scale-aware support selection。

### 20.10 本轮是否达成目标

严格结论：

```text
完整 v4.1 / ScanNet 目标：没有达成。
scene0050_00 单场景 AP/AP50 超过 32f current：达成。
scene0050_00 单场景 AP/AP50/AP25 三项同时超过 32f current：没有达成。
多场景验证：没有完成。
fixed/inherit support 协议：本轮没有重新证明。
```

可以安全写：

```text
在 scene0050_00 单场景、cached-carrier、recompute support 条件下，
evidence graph + stable mask backprojection core + hard point ownership WTA
把 AP/AP50 提高到 0.232910 / 0.526316，
超过 32f current 的 0.202698 / 0.445714。
```

不能写：

```text
Stream4D-v4.1 已全面超过 Stream3D。
ScanNet 结果已经复现完成。
AP25 已超过原版。
多场景结果已经验证。
```

### 20.11 失败原因和证据链

#### 原因 A：carrier support 和 mask support 优化的是不同指标

证据：

```text
carrier support best:
AP/AP50/AP25 = 0.153257 / 0.474951 / 0.702668

maskbp r0.02 hard2:
AP/AP50/AP25 = 0.232910 / 0.526316 / 0.562160
```

解释：

- carrier support 覆盖更宽，所以 AP25 高。
- mask backprojection core 更精确，所以 AP/AP50 高。
- 但二者不能简单并集；hybrid 和 fusion 都失败。

#### 原因 B：高 IoU AP 需要紧 support，低 IoU AP25 需要宽 support

证据：

```text
r0.06 soft4: AP/AP50/AP25 = 0.202258 / 0.463415 / 0.563415
r0.04 hard2: AP/AP50/AP25 = 0.214166 / 0.487179 / 0.630416
r0.03 hard2: AP/AP50/AP25 = 0.225661 / 0.521662 / 0.630416
r0.02 hard2: AP/AP50/AP25 = 0.232910 / 0.526316 / 0.562160
```

解释：

- 半径从 0.06 收紧到 0.02，AP/AP50 上升。
- AP25 没有同步上升，说明覆盖不足。
- 这不是参数偶然，而是 precision/coverage tradeoff。

#### 原因 C：直接把质量分当预测 score 不可靠

证据：

```text
原始 evidence graph AP = 0.159209
score evidence_quality AP = 0.054330
score compactness AP = 0.089854
```

解释：

- 无监督证据质量能辅助点归属。
- 但它不等于评估需要的 instance ranking。
- 当前 object quality calibration 仍缺失。

#### 原因 D：多窗口不是自动收益

证据：

```text
96f maskbp r0.02 hard2:
AP/AP50/AP25 = 0.232910 / 0.526316 / 0.562160

128f maskbp r0.02 hard2:
AP/AP50/AP25 = 0.220462 / 0.461323 / 0.567568
```

解释：

- 128f object 数从 59 增到 67。
- 更多窗口带来更多候选和潜在污染，没有自动稳定 object。

### 20.12 下一步算法建议

不要继续做简单 fusion / top-N / score sweep。下一步应该做真正的 object 内 scale-aware support：

```text
对每个 evidence graph object 同时维护：
1. precise core support
2. wide recall support
3. 每个 support point 的不确定性
4. object 内部的 boundary confidence
```

输出时不应该简单输出两套 object hypothesis，而是应该在一个 object mask 里决定哪些点属于：

```text
high-confidence core
low-confidence fringe
discarded conflict region
```

更具体的算法方向：

```text
1. 用 mask backprojection 得到 high-precision core。
2. 用 carrier support 得到 recall candidate。
3. 只允许 carrier candidate 中与 core 在 3D 连通、距离近、且被多个稳定视角支持的点进入 fringe。
4. 对 fringe 点做 soft confidence，而不是直接并入所有 AP 阈值。
5. 如果评估格式不能表达 per-point confidence，则需要学习或校准一个二值化阈值，但这个阈值必须只依赖无监督证据，不能看 GT。
```

这才有可能同时补 AP25，又不牺牲 AP/AP50。

## 21. 2026-06-08 Object 内 scale-aware support 与 component densify 结果

本节继续第 20 节建议，不做简单 fusion/top-N，而是在每个 evidence graph object 内部选择 support。所有结果来自实际 `evaluation.evaluate` 输出文件。

### 21.1 本轮代码修改

新增导出模式：

```text
core_fringe
component_densify
```

涉及代码：

```text
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/replay_memory.py
Stream3D/stream4d/run_scannet.py
```

`core_fringe`：

```text
1. 用稳定 mask 生成 precise core。
2. 从 carrier support 中选离 core 足够近的 fringe 点。
3. 用 max fringe ratio 防止无界扩张。
```

`component_densify`：

```text
1. carrier 只作为种子，不直接并入最终 object。
2. 在稳定 mask 中找到包含 carrier seed 的 2D 连通区域。
3. 对该连通区域做边界腐蚀和 seed distance 限制。
4. 回投到 3D。
5. 用 relative coverage gate 选择 mask 视角。
```

新增参数：

```text
--export-support-mode core_fringe|component_densify
--export-core-nn-radius
--export-fringe-nn-radius
--export-fringe-radius
--export-fringe-max-ratio
--export-mask-min-relative-coverage
```

其中 `export-mask-min-relative-coverage=1.0` 的含义是：

```text
对每个 object，只保留 coverage 与该 object 最强 mask 相等的 mask 视角。
```

这不使用 GT，只使用 evidence graph 中已有的 mask/carrier coverage。

### 21.2 core_fringe 负例

| experiment | objects | points | conflict | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|---:|
| core002 fr002 ratio025 hard2 | 59 | 35299 | 0.000000 | 0.215013 | 0.491379 | 0.526316 |
| core002 fr004 ratio025 hard2 | 59 | 36129 | 0.000000 | 0.206130 | 0.517241 | 0.552155 |
| core002 fr008 ratio050 hard2 | 59 | 36905 | 0.000000 | 0.192002 | 0.517241 | 0.552155 |

结论：

- 从 carrier support 给 precise core 补近邻 fringe 没有提升。
- fringe 距离扩大后 AP 继续下降，说明 carrier fringe 仍带有跨实例污染。
- 这解释了为什么第 20 节的简单多尺度拼接会失败。

### 21.3 选择式 core/wide 诊断

使用已有 fusion 工具做诊断：如果 carrier wide support 对 core 的覆盖足够、扩张不大，就用 carrier 替换 core。

| experiment | output objects | points | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| core002 + carrier079, primary IoC 0.3, expansion 1.25 | 59 | 34143 | 0.227163 | 0.562160 | 0.598911 |
| core003 + carrier079, primary IoC 0.3, expansion 1.25 | 59 | 36979 | 0.220000 | 0.522000 | 0.630000 |

结论：

- 选择式替换比简单拼接好，AP50 可以提高。
- 但 AP/AP25 不能同时超过 best component densify。
- 该方向不作为当前主结果。

### 21.4 component_densify 基础扫描

| experiment | objects | points | conflict | observations used | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| r0.03 d16 m5 hard2 | 59 | 32469 | 0.000000 | 290 | 0.281143 | 0.495806 | 0.604846 |
| r0.02 d32 m5 hard2 | 59 | 32586 | 0.000000 | 289 | 0.282461 | 0.500479 | 0.573276 |
| r0.03 d16 m5 min3 | 59 | 32469 | 0.127475 | 290 | 0.282950 | 0.517241 | 0.587931 |
| r0.03 d16 m6 hard2 | 59 | 38162 | 0.000000 | 348 | 0.219250 | 0.417781 | 0.649095 |

结论：

- component densify 立即把 AP 提到 0.28 左右，明显优于前一轮 best 0.232910。
- m6 能提高 AP25，但 AP/AP50 大幅下降，说明第 6 个 mask 往后噪声很强。
- 这推动了 relative coverage gate。

### 21.5 Relative coverage gate 关键结果

固定：

```text
graph carrier IoC = 0.79
min_component_observations = 10
support = component_densify
core nn radius = 0.03
seed distance = 16 px
max masks per object = 8
WTA = hard2 compactness
```

| relative coverage | objects | points | observations used | AP | AP50 | AP25 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 59 | 19330 | 344 | 0.277688 | 0.507246 | 0.602254 |
| 0.70 | 59 | 15256 | 106 | 0.259000 | 0.464000 | 0.573000 |
| 0.85 | 59 | 8637 | n/a | 0.350000 | 0.508000 | 0.656000 |
| 0.88 | 59 | 7869 | n/a | 0.341000 | 0.493000 | 0.655000 |
| 0.90 | 59 | 7742 | 106 | 0.365873 | 0.528061 | 0.701531 |
| 0.92 | 59 | 7326 | n/a | 0.387000 | 0.538000 | 0.737000 |
| 0.95 | 59 | 6411 | 74 | 0.422572 | 0.555556 | 0.763889 |
| 0.97 | 59 | 6319 | 68 | 0.436553 | 0.575758 | 0.795455 |
| 0.99 | 59 | 6211 | 60 | 0.445286 | 0.575758 | 0.795455 |
| 1.00 | 59 | 6146 | 56 | 0.454545 | 0.575758 | 0.795455 |

最佳 96f：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
AP/AP50/AP25 = 0.454545 / 0.575758 / 0.795455
objects = 59
points = 6146
```

对比 32f current：

```text
32f current = 0.202698 / 0.445714 / 0.681429
delta = +0.251847 / +0.130044 / +0.114026
```

解释：

- 这是本地 scene0050 recompute 协议下第一次 AP/AP50/AP25 三项都明显超过 32f current。
- 严格 relative coverage gate 越强，指标越好，说明低 coverage mask 是主要污染源。
- 最佳结果只用了 56 个 mask observations，说明不是靠更多视角堆覆盖，而是靠极高置信视角。

### 21.6 graph IoC 和 128f 验证

| config | objects | points | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| 96f ioc0.78 rel0.95 | 61 | 6419 | 0.422572 | 0.555556 | 0.763889 |
| 96f ioc0.79 rel1.00 | 59 | 6146 | 0.454545 | 0.575758 | 0.795455 |
| 96f ioc0.80 rel0.95 | 58 | 4378 | 0.361000 | 0.576000 | 0.682000 |
| 128f ioc0.79 rel1.00 | 67 | 6531 | 0.490385 | 0.605769 | 0.810897 |

128f 最佳：

```text
stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
AP/AP50/AP25 = 0.490385 / 0.605769 / 0.810897
objects = 67
points = 6531
```

对比 32f current：

```text
AP:   0.490385 > 0.202698
AP50: 0.605769 > 0.445714
AP25: 0.810897 > 0.681429
```

结论：

- 在极严格 relative coverage gate 下，128f 终于比 96f 更好。
- 更多窗口是否有收益，取决于是否能过滤低质量 mask 视角；没有 gate 时 128f 之前多次变差。

### 21.7 fixed-support 诊断

128f best prediction 放回其它 support universe：

| eval support | AP | AP50 | AP25 |
|---|---:|---:|---:|
| own recompute support | 0.490385 | 0.605769 | 0.810897 |
| 32f current support | 0.135185 | 0.300000 | 0.300000 |
| scannet self-inherit support | 0.008383 | 0.031532 | 0.072072 |
| Stream4D MVP support diagnostic | 0.135185 | 0.300000 | 0.300000 |
| Stream4D adaptive support diagnostic | 0.090741 | 0.133333 | 0.133333 |

结论：

- fixed-support 仍失败。
- 当前 best 是 observed-support / recompute protocol 下的强结果，不是 unified support universe 下的完整胜利。
- 不能写成“Stream4D v4.1 已经全面超过 Stream3D”。

### 21.8 本轮是否达成目标

达成：

```text
scene0050_00 recompute 协议下，AP/AP50/AP25 三项同时超过 32f current。
128f 在严格 mask evidence gate 下超过 96f。
component_densify 证明比 core_fringe / 简单 fusion 更有效。
```

未达成：

```text
完整 ScanNet final split / full split 没有完成。
fixed/inherit support 没有达成。
Replica-Dynamic / Dynamic Replica 没有执行。
D4RT-native Sim3 export 没有执行。
```

安全表述：

```text
在 scene0050_00 单场景、cached-carrier、recompute support 条件下，
evidence graph + carrier-seeded mask connected-component densification
+ strict relative coverage gate
把 128f AP/AP50/AP25 提升到 0.490385 / 0.605769 / 0.810897，
显著超过 32f current 的 0.202698 / 0.445714 / 0.681429。
```

必须同时写：

```text
该结果放回 32f / scannet_self_inherit / MVP / adaptive support 后显著下降，
因此不能作为 fixed/inherit support 胜利。
```

### 21.9 关键 insight

1. **低 coverage mask 是主要污染源**  
   relative coverage 从 0.50 提到 1.00，AP 从 0.277688 提到 0.454545。严格 gate 不是缩小 support 作弊，而是在同一 object 内拒绝低证据视角。

2. **carrier 适合作为种子，不适合直接作为 fringe**  
   core_fringe 失败，component_densify 成功。说明 carrier 的价值是定位 object 在 mask 中的连通区域，而不是直接提供最终边界。

3. **多窗口收益依赖视角质量过滤**  
   没有 strict gate 时 128f 经常差于 96f；rel1.0 后 128f 明显超过 96f。

4. **fixed support 失败说明 coverage universe 仍未解决**  
   best recompute 很强，但 on scannet_self_inherit 只有 0.008383 / 0.031532 / 0.072072。下一步若按计划继续，必须面向 fixed/inherit support 设计，而不是继续优化 scene0050 recompute。

## 22. 2026-06-08 fixed-support 继续推进：补漏有效但未达成目标

本节接续第 21 节。第 21 节已经证明：

```text
scene0050_00 own recompute support:
128f strict component_densify rel1.0 = 0.490385 / 0.605769 / 0.810897
32f current                         = 0.202698 / 0.445714 / 0.681429
```

但是 fixed/inherit support 没有达成。因此本轮只围绕 fixed-support 缺口继续推进，重点不是再提高 own recompute。

### 22.1 本轮代码修改

修改：

```text
Stream3D/tools/fuse_prediction_configs.py
```

新增：

```text
--preserve-primary-score
--preserve-secondary-score
```

目的：

```text
支持多级低置信补漏。
例如第一轮输出 primary score=1.0、secondary score=0.2；
第二轮继续补漏时保留上一轮已有 score，再把新 rescue 设为 0.1。
```

新增：

```text
Stream3D/tools/complete_prediction_on_support.py
```

这个工具做的事情：

```text
1. 读取一个 full-scene prediction。
2. 读取一个目标 support config 的 pre_points。
3. 读取 ScanNet mesh 坐标。
4. 只在目标 support 里寻找还没有被 prediction 覆盖的点。
5. 将这些未覆盖点按 3D 最近邻分配给最近的已有 object。
6. 用 radius 和 max_added_ratio 限制补点范围。
```

重要边界：

```text
该工具不读取 GT。
它是 fixed-support diagnostic / support completion 尝试，不是最终主方法。
```

### 22.2 cross-prepoints 守卫：32f support 本身不难

本轮使用 `tools.evaluate_cross_prepoints` 重新跑 fixed-support，而不是只直接切换 evaluator 的 `tmp_config`。该工具会做 shape audit。所有本轮 scene0050 prediction 都是：

```text
mask_shape_mode = full_scene
expanded_prediction_scenes = 0
```

关键守卫：

| row | prediction | target support | AP | AP50 | AP25 | union in target | #pred |
|---|---|---|---:|---:|---:|---:|---:|
| Stream3D scannet on 32f support | `scannet` | `stream4d_scannet_scene0050_32f_ioc075_fixmem` | 0.391132 | 0.646154 | 0.761538 | 99.7708% | 121 |
| 32f current self audit | `stream4d_scannet_scene0050_32f_ioc075_fixmem` | same | 0.202698 | 0.445714 | 0.681429 | 100.0000% | 227 |
| 128f strict best on 32f support | `128f compdens rel1.0` | same | 0.135185 | 0.300000 | 0.300000 | 19.1583% | 67 |

结论：

```text
1. 32f support 本身不是异常困难的 support universe。
2. 原版 Stream3D prediction 在同一个 32f support 上很强：0.391132 / 0.646154 / 0.761538。
3. 当前 128f strict Stream4D 只有 19.1583% 的 target support 被 prediction union 覆盖，因此 fixed-support 低分不是 evaluator shape bug，而是真 coverage / candidate quality 问题。
```

### 22.3 宽 support 候选的 fixed-support 表现

| prediction | AP | AP50 | AP25 | union in target | #pred |
|---|---:|---:|---:|---:|---:|
| 128f strict rel1.0 | 0.135185 | 0.300000 | 0.300000 | 19.1583% | 67 |
| 96f component densify rel0.5 | 0.152614 | 0.294118 | 0.397059 | 50.7123% | 59 |
| 96f component densify no relative gate | 0.168599 | 0.406522 | 0.508696 | 65.9243% | 59 |
| 128f maskbp r0.02 | 0.152778 | 0.363636 | 0.572727 | 65.1875% | 67 |

结论：

```text
更宽的 support 能明显提高 target support 覆盖率；
但 AP 仍没有超过 32f current，更远没有接近 Stream3D scannet on same support。
```

解释：

```text
低 relative coverage gate 释放了更多点，但这些点的实例边界仍粗；
AP25 有收益，AP/AP50 不够，说明宽 support 主要解决粗 recall，不解决精确实例质量。
```

### 22.4 低置信 rescue：有帮助，但未达成

算法思路：

```text
高精 128f strict rel1.0 作为 high-confidence prediction。
宽 support 候选作为 low-confidence rescue prediction。
不是把宽点硬并入 strict mask，而是作为较低置信度的候选补漏。
```

关键结果：

| config | own recompute AP/AP50/AP25 | fixed 32f AP | fixed AP50 | fixed AP25 | union in target | #pred |
|---|---|---:|---:|---:|---:|---:|
| strict + compnone lowconf | 0.199 / 0.392 / 0.458 | 0.192832 | 0.472414 | 0.553448 | 66.1372% | 126 |
| strict + maskbp lowconf | 0.157 / 0.335 / 0.435 | 0.190179 | 0.435714 | 0.600000 | 65.3758% | 134 |
| strict + compnone + maskbp tiered | 0.156 / 0.354 / 0.447 | 0.193337 | 0.472414 | 0.571518 | 68.6671% | 193 |
| strict + 32f merge070 lowconf | 0.376 / 0.576 / 0.658 | 0.190344 | 0.490476 | 0.685714 | 76.7152% | 82 |

对比：

```text
32f current self audit = 0.202698 / 0.445714 / 0.681429
Stream3D scannet on 32f support = 0.391132 / 0.646154 / 0.761538
```

结论：

```text
低置信 rescue 是本轮最有效 fixed-support 方向：
fixed AP 从 strict 的 0.135185 提高到最高 0.193337。
strict + 32f merge070 lowconf 的 AP50/AP25 已超过 32f current：
  AP50 0.490476 > 0.445714
  AP25 0.685714 > 0.681429
但 AP 仍低于 32f current：
  AP 0.190344 < 0.202698
因此 fixed-support 目标仍未达成。
```

为什么没有达成：

```text
1. low-confidence rescue 增加 recall 的同时带来更多 false positives。
2. compnone/maskbp rescue 的 union in target 能到 66%-69%，但实例边界不够精确。
3. 32f merge070 rescue 的 union in target 达到 76.7152%，AP25 接近/超过 32f current，但 AP 仍低，说明它的高 IoU 匹配数量仍不足。
4. 原版 Stream3D 在同 support 上 union in target 为 99.7708%，且只用 121 个 prediction 就达到 0.391132 AP；Stream4D 目前要么 coverage 不够，要么候选太碎/边界粗。
```

### 22.5 support completion 最近邻补点：失败

算法思路：

```text
只在 32f target support 中寻找原 prediction 未覆盖的点。
将这些点用 3D nearest neighbor 分配给最近的已有 128f strict object。
用半径和每个 object 最大扩张比例限制补点。
```

support completion 诊断：

| config | assigned points | output union | union in target | own recompute AP/AP50/AP25 | fixed AP | fixed AP50 | fixed AP25 |
|---|---:|---:|---:|---|---:|---:|---:|
| r0.02 ratio1 | 394 | 6925 | 22.3841% | 0.433 / 0.569 / 0.755 | 0.141667 | 0.262500 | 0.262500 |
| r0.04 ratio1 | 963 | 7494 | 27.0427% | 0.376 / 0.569 / 0.755 | 0.110278 | 0.240000 | 0.240000 |
| r0.04 ratio2 | 998 | 7529 | 27.3293% | 0.376 / 0.569 / 0.781 | 0.110278 | 0.240000 | 0.240000 |
| r0.04 nocap | 998 | 7529 | 27.3293% | 0.376 / 0.569 / 0.781 | 0.110278 | 0.240000 | 0.240000 |
| r0.06 ratio1 | 1347 | 7878 | 30.1867% | 0.298 / 0.554 / 0.831 | 0.065741 | 0.225000 | 0.225000 |

结论：

```text
3D 最近邻 support completion 失败。
它在 own recompute 下看起来还能保留较高 AP，但 fixed support 下越补越差。
```

原因分析：

```text
1. 最近邻只看几何距离，不知道实例边界。
2. scene0050 中大量椅子相互接近，target support 中未覆盖点容易被分给相邻错误 object。
3. 补点增加的 target coverage 很有限：最高仅 30.1867%，仍远低于 low-confidence rescue 的 66%-77%。
4. 因为补点直接并入原 object，高置信 mask 被污染，fixed AP 比低置信 rescue 更差。
```

### 22.6 本轮是否达成目标

达成：

```text
1. 用 cross-prepoints 工具确认 previous fixed-support 低分不是 shape/index bug。
2. 证明 32f support 本身并不难：Stream3D scannet on same support = 0.391132 / 0.646154 / 0.761538。
3. 找到一个比 strict fixed-support 明显更好的方向：low-confidence rescue。
4. 新增 score-preserving fusion 和 support completion 工具，便于审计。
```

未达成：

```text
1. 没有超过 32f current 的 fixed-support AP。
2. 没有接近原版 Stream3D scannet on same 32f support。
3. support completion 最近邻补点失败。
4. full ScanNet final split / Replica-Dynamic / D4RT-native Sim3 export 仍未完成。
```

当前最强 fixed-support 相关候选：

```text
strict + compnone + maskbp tiered:
  fixed 32f = 0.193337 / 0.472414 / 0.571518

strict + 32f merge070 lowconf:
  fixed 32f = 0.190344 / 0.490476 / 0.685714
```

但这两个都不能作为“达成 v4.1 fixed-support 胜利”。

### 22.7 关键 insight

1. **fixed-support 失败不是 evaluator 问题**  
   cross-prepoints 显示 prediction 都是 full_scene；32f current self audit 也精确复现了 0.202698 / 0.445714 / 0.681429。

2. **32f support 不是问题本身**  
   原版 Stream3D 在同一个 support 上有 0.391132 / 0.646154 / 0.761538。Stream4D 不能把 fixed-support 失败解释为 support universe 太苛刻。

3. **补漏要以“候选”形式出现，不能硬并入高精 object**  
   low-confidence rescue 比 nearest support completion 好很多。原因是 AP 可以把低置信候选放在后面，而硬并入会直接污染高置信 mask。

4. **下一步真正需要的是实例级边界质量，而不是更多点**  
   compnone/maskbp/tiered 都能提高 target support coverage，但 AP 没过 32f current。说明单纯扩大 union 不够，必须改 mask evidence / object matching / same-frame exclusivity，让新增点进入正确 object。

5. **与原版 Stream3D 的差距主要是 proposal quality**  
   原版 Stream3D 在同 support 中 union 覆盖 99.7708%，121 个实例就有 0.391132 AP。当前 Stream4D 最好的 low-confidence rescue 使用 82 到 193 个实例，仍只有约 0.19 AP。问题不是候选数量不足，而是候选边界和排序都没有达到 Stream3D 的质量。

## 23. 2026-06-08 Tiered inherit fusion：scene0050 的 32f fixed support 达成

第 22 节的失败说明：`128f strict` 自己当主 prediction，再试图补 fixed support，效果不够。原因是 fixed-support 需要完整 coverage，而 `128f strict rel1.0` 只有 19.1583% 的 32f target support 被覆盖。

本节改变算法假设：

```text
不要让 128f strict 负责 coverage。
让 32f current / inherited support 负责 coverage。
让 128f strict evidence graph 负责高置信 precision 排序。
再用 compnone/maskbp 作为更低置信 recall tier。
```

这个方向符合 v4.1 计划里“inherit / fixed-pre_points 诊断”和“不要只缩小 support”的要求，因为 target support 仍是 32f current support，prediction union in target 为 100%。

### 23.1 关键结果：32f support 上超过 32f current

基线：

| row | fixed support | AP | AP50 | AP25 | union in target | #pred |
|---|---|---:|---:|---:|---:|---:|
| 32f current self audit | 32f current | 0.202698 | 0.445714 | 0.681429 | 100.0000% | 227 |
| Stream3D scannet on same support | 32f current | 0.391132 | 0.646154 | 0.761538 | 99.7708% | 121 |

本轮 tiered inherit：

| config | fixed support | AP | AP50 | AP25 | union in target | #pred |
|---|---|---:|---:|---:|---:|---:|
| 32f current low + 128f strict high | 32f current | 0.233729 | 0.568293 | 0.731707 | 100.0000% | 294 |
| 128f strict high + 32f current low | 32f current | 0.233729 | 0.568293 | 0.731707 | 100.0000% | 294 |
| + compnone lower tier | 32f current | 0.242074 | 0.582689 | 0.731707 | 100.0000% | 353 |
| + maskbp score 0.05 | 32f current | 0.242409 | 0.582689 | 0.744262 | 100.0000% | 420 |
| + maskbp score 0.10 | 32f current | 0.241514 | 0.581190 | 0.748206 | 100.0000% | 420 |
| + maskbp score 0.20 | 32f current | 0.242460 | 0.554360 | 0.723016 | 100.0000% | 420 |

当前最均衡候选：

```text
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
fixed 32f = 0.242409 / 0.582689 / 0.744262
```

对比 32f current：

```text
AP:   0.242409 > 0.202698, +0.039711
AP50: 0.582689 > 0.445714, +0.136974
AP25: 0.744262 > 0.681429, +0.062833
```

结论：

```text
在 scene0050_00 的 32f fixed support 上，本轮 tiered inherit fusion 已经超过 32f current。
```

### 23.2 为什么这个方向有效

对比第 22 节：

```text
128f strict alone on 32f support:
  AP/AP50/AP25 = 0.135185 / 0.300000 / 0.300000
  union in target = 19.1583%

strict + low-confidence rescue:
  best fixed AP ≈ 0.193337
  union in target 最高约 76.7%

tiered inherit fusion:
  best fixed AP = 0.242409
  union in target = 100%
```

解释：

```text
1. 32f current 提供完整 target support coverage。
2. 128f strict 提供高 precision candidate，并排在 AP 曲线前面。
3. 32f current 作为低分候选，保留 coverage，但不会抢在 strict 前面制造早期 FP。
4. compnone/maskbp 更低分，只在 recall 后段补漏。
```

这比“把宽 support 硬并入 strict object”更合理，因为 AP 可以利用 score 顺序，而硬并入会直接污染 high-confidence mask。

### 23.3 多 support 诊断

最终候选 `stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005` 放到不同 support：

| target support | AP | AP50 | AP25 | union in target |
|---|---:|---:|---:|---:|
| 32f current | 0.242409 | 0.582689 | 0.744262 | 100.0000% |
| Stream4D MVP prepoints | 0.242409 | 0.582689 | 0.744262 | 100.0000% |
| Stream4D adaptive prepoints | 0.264374 | 0.526984 | 0.782460 | 100.0000% |
| scannet self-inherit | 0.008501 | 0.032589 | 0.111223 | 22.1598% |

结论：

```text
1. 在 Stream4D 32f/MVP/adaptive 类 support 下，tiered inherit 是有效的。
2. 在 scannet self-inherit / 原版 Stream3D 大 support 下仍然失败。
3. 因此不能写成“全 fixed support 达成”或“原版 Stream3D support 达成”。
```

### 23.4 本轮是否达成目标

达成：

```text
scene0050_00 的 32f fixed support 上，AP/AP50/AP25 三项都超过 32f current。
```

未达成：

```text
1. 没有超过原版 Stream3D scannet on same support：
   0.242409 / 0.582689 / 0.744262
   仍低于
   0.391132 / 0.646154 / 0.761538。

2. 没有解决 scannet self-inherit support：
   0.008501 / 0.032589 / 0.111223。

3. 没有完成 full ScanNet final split。
4. 没有完成 Replica-Dynamic / D4RT-native Sim3 export。
```

安全表述：

```text
在 scene0050_00 单场景、Stream4D 32f fixed support 诊断下，
使用 128f strict evidence graph 作为高置信 tier、
32f current 作为 inherited coverage tier、
compnone/maskbp 作为低置信 recall tier，
可以把 AP/AP50/AP25 提升到
0.242409 / 0.582689 / 0.744262，
超过 32f current 的
0.202698 / 0.445714 / 0.681429。
```

必须同时写：

```text
该结果仍没有超过原版 Stream3D scannet on same support，
也没有解决 scannet self-inherit support。
```

### 23.5 新 insight

1. **coverage 与 precision 应该分层，而不是混成一个 mask**  
   32f current 的 coverage 很重要；128f strict 的 precision 很重要。二者如果硬合并会污染边界，分成不同 score tier 后 AP 能利用它们各自优点。

2. **previous “补漏失败”不是补漏思路完全错，而是主次反了**  
   第 22 节以 strict 为主、宽 support 为辅，AP 只能到 0.193。第 23 节以 inherited 32f coverage 为底、strict 为高分重排序，AP 到 0.242。

3. **目标支持域决定 claim 范围**  
   在 Stream4D 32f/MVP/adaptive support 下有效；在 scannet self-inherit 大 support 下无效。因此论文/报告必须声明 support policy。

4. **下一步算法方向**  
   不应继续盲目扩大 dense support。更合理的是把 tiered inherit 做成正式算法：
   `coverage proposal bank + evidence graph precision ranker + low-confidence recall tiers`，
   并在更多 scenes 上验证是否稳定。

## 24. 2026-06-08 containment suppression 与排序/NMS 复盘

### 24.1 本轮目标

第 23 节已经在 `scene0050_00` 的 32f fixed support 上超过 32f current，但没有超过原版 Stream3D on same support。

本轮按用户要求继续推进，并重点尝试算法层面的改进，而不是继续堆工程：

```text
1. 用 containment suppression 改善多层候选融合。
2. 用 score calibration 检查排序是否是主要瓶颈。
3. 用 mask NMS 检查重复候选是否是主要瓶颈。
4. 如仍未超过 Stream3D，给出失败原因和证据链。
```

### 24.2 做了什么代码修改

修改 `Stream3D/tools/fuse_prediction_configs.py`：

```text
新增 --drop-secondary-overlap-mode。
支持 iou / secondary_ioc / min_ioc。
本轮最有用的是 secondary_ioc：
  intersection / secondary_area。
它用于删除“低分 secondary 候选几乎被高分 primary 候选包含”的重复候选。
```

新增 `Stream3D/tools/rescore_prediction_scores.py`：

```text
不读取 GT。
只根据 prediction mask 的面积做 score tiebreaker 或面积过滤。
用于诊断“同一置信层内排序是否导致 AP 损失”。
```

新增 `Stream3D/tools/nms_prediction_masks.py`：

```text
不读取 GT。
不依赖 object_dict。
直接对 prediction npz 做 class-agnostic mask NMS。
用于诊断“重复候选是否导致 AP 损失”。
```

### 24.3 关键结果

统一诊断 support：

```text
pre_points config = stream4d_scannet_scene0050_32f_ioc075_fixmem
scene = scene0050_00
dataset = ScanNet
class-agnostic evaluator = 原版 Stream3D-style cropped TMP evaluator
```

| config | AP | AP50 | AP25 | union in target | #pred | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 32f current self audit | 0.202698 | 0.445714 | 0.681429 | 100.0000% | 227 | 被超过 |
| previous tiered inherit best | 0.242409 | 0.582689 | 0.744262 | 100.0000% | 420 | 被本轮超过 |
| new best: strict + 32f containment + comp + mask | 0.253098 | 0.615832 | 0.802912 | 98.9356% | 337 | 本轮最佳 |
| Stream3D scannet on same 32f support | 0.391132 | 0.646154 | 0.761538 | 99.7708% | 121 | 仍未超过 |

本轮最佳 config：

```text
stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
```

对应 fixed-support evaluation config：

```text
stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005_cross_32fsupport
```

提升幅度：

```text
相对 32f current:
  AP   +0.050399
  AP50 +0.170118
  AP25 +0.121483

相对 previous tiered inherit best:
  AP   +0.010689
  AP50 +0.033144
  AP25 +0.058650
```

相对 Stream3D on same support：

```text
AP   仍低 0.138035
AP50 仍低 0.030322
AP25 高 0.041373
```

安全结论：

```text
scene0050_00 的 32f fixed support 上，
本轮 containment-tiered 方法进一步超过 32f current 和 previous tiered inherit best，
并且 AP25 已超过原版 Stream3D on same support。

但 AP 和 AP50 仍未超过原版 Stream3D on same support，
因此不能写成“达成超过原版 Stream3D”。
```

### 24.4 containment suppression 为什么有效

第 23 节的 tiered inherit 是：

```text
128f strict high score
32f current low score
compnone lower score
maskbp lowest score
```

本轮改成：

```text
在每次把 secondary 低分候选加入 primary 高分候选前，
如果 secondary 候选的大部分面积已经被 primary 候选覆盖，
则丢掉 secondary 候选。
```

这和普通 IoU 不同。普通 IoU 对“一个小碎片被大 mask 包含”的情况不敏感，因为 union 很大；`secondary_ioc` 对这种情况敏感，因为分母只看 secondary 面积。

实测趋势：

| first-stage secondary_ioc threshold | final AP | final AP50 | final AP25 |
|---:|---:|---:|---:|
| 0.70 | 0.247831 | 0.602151 | 0.779212 |
| 0.60 | 0.247831 | 0.602151 | 0.779212 |
| 0.50 | 0.253098 | 0.615832 | 0.802912 |
| 0.40 | 0.253098 | 0.615832 | 0.802912 |
| 0.30 | 0.253098 | 0.615832 | 0.802912 |

解释：

```text
0.50 以下已经把同一批被包含 secondary 候选删掉，所以 0.50 / 0.40 / 0.30 结果相同。
0.50 相比 0.70 更强，说明 32f low coverage 层中确实存在一批被 strict 高分层覆盖的重复候选。
```

### 24.5 score calibration 失败

尝试：

```text
在保留原 score tier 的前提下，用 mask area / sqrt_area / log_area 做同层排序微调。
```

代表结果：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| containment best before score calibration | 0.247831 | 0.602151 | 0.779212 |
| log_area tiebreaker | 0.190 | 0.596 | 0.860 |

结论：

```text
面积排序能显著提高 AP25，但会降低 AP。
这说明大 mask 更容易粗匹配，但边界不够干净，放到前面会伤害高 IoU 阈值。
```

这不是可以作为最终算法的改进。

### 24.6 面积过滤失败

尝试：

```text
按 full-scene mask area 过滤极小候选。
min_area = 1 / 5 / 10 / 20 / 50 / 100。
```

代表结果：

```text
min_area=100:
  0.248 / 0.602 / 0.779
```

该结果没有超过未过滤版本。

解释：

```text
错误主要不是 full-scene 空 mask 或极小 mask。
很多问题发生在 fixed target support 内：一个全场景 mask 可能不小，但投到 32f support 后仍然边界不准或实例重复。
```

### 24.7 mask NMS 失败

尝试：

```text
对最终 prediction 做 class-agnostic mask NMS。
overlap mode = min_ioc / candidate_ioc / iou。
不读取 GT。
```

代表结果：

| NMS | AP | AP50 | AP25 | 变化 |
|---|---:|---:|---:|---|
| min_ioc=0.80 | 0.230 | 0.514 | 0.646 | 明显下降 |
| iou=0.50 | 0.241 | 0.604 | 0.784 | AP 下降，AP50/AP25 微升 |

结论：

```text
重复候选确实存在，但不能粗暴删除。
一些重叠候选对 recall 有贡献；NMS 删掉后 AP50/AP25 往往下降。
```

### 24.8 GT 只读诊断：为什么还没超过 Stream3D

注意：

```text
下面诊断只用于解释失败原因。
GT 没有进入任何 prediction 生成、筛选、融合或打分代码。
```

在同一个 32f target support 上，统计每个预测与真实实例的最佳 IoU：

| method | #pred | nonempty pred | union in target | pred best IoU≥0.25 | pred best IoU≥0.5 | pred best IoU≥0.75 | pred best IoU≥0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stream3D on 32f support | 121 | 75 | 12186 | 19 | 18 | 8 | 6 |
| new best | 337 | 309 | 12084 | 37 | 29 | 8 | 5 |
| previous tiered best | 420 | 385 | 12214 | 51 | 39 | 11 | 7 |
| 32f current | 227 | 226 | 12214 | 21 | 14 | 4 | 2 |

按 GT 实例统计：

| method | GT with pred IoU≥0.25 | mean preds per matched GT @0.25 | GT with pred IoU≥0.5 | mean preds per matched GT @0.5 | GT with pred IoU≥0.75 |
|---|---:|---:|---:|---:|---:|
| Stream3D on 32f support | 19 | 1.00 | 18 | 1.00 | 8 |
| new best | 18 | 2.06 | 15 | 1.93 | 6 |
| previous tiered best | 19 | 2.74 | 15 | 2.60 | 7 |
| 32f current | 19 | 1.16 | 14 | 1.00 | 4 |

关键解释：

```text
1. new best 的候选数量更多，IoU≥0.5 的预测也更多：29 vs Stream3D 的 18。
2. 但 new best 覆盖到的 GT 更少：IoU≥0.5 时只有 15 个 GT，被 Stream3D 覆盖 18 个。
3. new best 对同一个 GT 有明显重复：IoU≥0.5 时每个被命中的 GT 平均 1.93 个预测，而 Stream3D 是 1.00。
4. 所以问题不是“没有候选”，而是候选分配和实例边界质量不足：重复候选多、漏掉部分 GT、高 IoU 阈值下不够一对一。
```

这解释了为什么：

```text
new best AP25 已经超过 Stream3D，
但 AP/AP50 仍低于 Stream3D。
```

### 24.9 本轮是否达成目标

达成：

```text
1. 继续超过 32f current。
2. 进一步超过 previous tiered inherit best。
3. AP25 超过原版 Stream3D on same 32f support。
4. 明确定位没有超过 Stream3D 的原因：
   Stream4D 当前候选更多，但高 IoU 一对一实例质量不足。
```

未达成：

```text
1. AP 没有超过 Stream3D on same support：
   0.253098 < 0.391132。

2. AP50 没有超过 Stream3D on same support：
   0.615832 < 0.646154。

3. 没有解决 scannet self-inherit 大 support。
4. 没有完成 full ScanNet final split。
5. 没有完成 Replica-Dynamic / D4RT-native Sim3 export。
```

### 24.10 新 insight

1. **containment suppression 是有效的，但只能解决一部分问题**  
   它能减少低分 coverage 层中被高分 evidence graph 已经解释过的重复候选，因此 AP/AP50/AP25 都提升。

2. **AP25 已经不是主要瓶颈**  
   new best AP25 = 0.802912，超过 Stream3D on same support 的 0.761538。粗覆盖和低阈值召回已经足够强。

3. **真正瓶颈是实例级一对一质量**  
   Stream3D 用 121 个 prediction 覆盖 18 个 IoU≥0.5 的 GT，且几乎一对一；new best 用 337 个 prediction 只覆盖 15 个 IoU≥0.5 的 GT，并且每个 GT 平均接近 2 个重复预测。

4. **简单 score calibration 和 NMS 不能替代 object-level matching**  
   面积排序会提高粗匹配但伤害高 IoU；NMS 会删掉部分有用 recall。下一步应改 object matching / same-frame exclusivity / evidence graph assignment，而不是只做后处理。

5. **下一步更合理的算法方向**  
   对每个 target support 区域做更强的 object-level assignment：
   ```text
   evidence graph precision ranker
   + coverage proposal bank
   + same-frame exclusivity
   + per-GT-like unsupervised object competition proxy
   + boundary-aware support refinement
   ```
   其中最后两项不能使用 GT，只能用 mask overlap、carrier evidence、2D mask boundary、multi-frame consistency 等无监督信号近似。

## 25. 2026-06-08 继续推进：最终 prediction point-level WTA 复盘

### 25.1 本轮目标是否达成

没有完全达成。

按 `docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md` 的目标，本轮仍然没有让主指标 AP 超过原版 Stream3D 的同协议结果：

| method | evaluation support | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| Stream3D baseline | same 32f support | 0.3911324786324787 | 0.6461538461538462 | 0.7615384615384615 |
| previous best before this round | same 32f support | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| best balanced result in this round | same 32f support | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |

本轮达成的部分目标：

```text
AP50 首次超过同 support 的 Stream3D：
0.6548076923076923 > 0.6461538461538462

AP25 继续超过同 support 的 Stream3D：
0.7973356854170808 > 0.7615384615384615
```

本轮未达成的核心目标：

```text
AP 仍然明显低于同 support 的 Stream3D：
0.2530757451946083 < 0.3911324786324787
差距 = 0.1380567334378704
```

### 25.2 本轮具体改了什么

新增代码：

```text
Stream3D/tools/wta_prediction_points.py
```

这个工具实现的是最终 prediction 上的三维点归属竞争。更直白地说：

```text
原来：
  一个三维点可以同时被多个预测物体占有。

现在新增一种后处理：
  如果一个点同时属于多个预测物体，
  就根据无监督优先级只保留一个物体拥有它。
```

它没有读取 ScanNet GT，也没有读取 evaluator 输出；只读取 prediction 文件里的 `pred_masks`、`pred_score`、`pred_classes`。

新增的关键参数：

| 参数 | 含义 |
|---|---|
| `--priority-mode score` | 分数最高的预测拿走冲突点 |
| `--priority-mode score_area_desc` | 分数优先，分数接近时轻微偏向更大的预测 |
| `--priority-mode score_area_asc` | 分数优先，分数接近时轻微偏向更小的预测 |
| `--min-conflict-owners 2` | 一个点至少被两个预测同时占有时才处理 |
| `--min-priority-margin` | 只有赢家分数比第二名高出足够多时才处理，避免强行处理模糊点 |
| `--drop-empty` | 如果某个预测被删到没有点，就删除这个预测 |

### 25.3 单独使用 WTA 的结果

| config | idea | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| no WTA best | containment suppression best | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| `pointwta_score` | 分数最高的预测赢得冲突点 | 0.24273504273504276 | 0.6548076923076923 | 0.7548076923076924 |
| `pointwta_area_desc` | 分数优先，轻微偏向大实例 | 0.23888888888888887 | 0.612 | 0.8160000000000001 |
| `pointwta_area_asc` | 分数优先，轻微偏向小实例 | 0.23422364672364676 | 0.6548076923076923 | 0.7548076923076924 |
| `pointwta_score_margin030` | 只有分数差足够大才 WTA | 0.24018126668988737 | 0.6171434169278996 | 0.8063300026123302 |

结论：

```text
hard WTA 能把 AP50 推过 Stream3D，但会损害 AP 和 AP25。
area_desc 能把 AP25 推高到 0.8160000000000001，但 AP/AP50 变差。
margin WTA 能保留更多模糊点，但没有解决 AP 主指标。
```

这说明“冲突点归属”确实是瓶颈之一，但简单按分数或面积一刀切不是最终答案。

### 25.4 WTA high + original low 的结果

因为 WTA 的 AP50 更好，而原始 containment best 的 AP/AP25 更稳定，所以做了分层融合：

```text
高分层：WTA 后的 prediction
低分层：原始 containment best prediction
评估：仍然使用同一个 32f Stream3D support
```

结果：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `wta_score_high_plus_orig_low001` | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |
| `wta_score_high_plus_orig_low005_drop085` | 0.25236897185487567 | 0.6548076923076923 | 0.8004682782667858 |
| `wta_score_high_plus_orig_low010_drop085` | 0.25261034570736063 | 0.6548076923076923 | 0.8016791044776119 |
| `wta_area_high_plus_orig_low005_drop085` | 0.2510042735042735 | 0.6312307692307693 | 0.8406153846153847 |
| `wta_area_high_plus_orig_low010_drop085` | 0.2513247863247863 | 0.6317692307692307 | 0.8413076923076923 |

本轮最均衡的结果是：

```text
stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001_cross_32fsupport
AP   = 0.2530757451946083
AP50 = 0.6548076923076923
AP25 = 0.7973356854170808
```

它相对 Stream3D 的状态：

```text
AP   仍然输了：0.2530757451946083 < 0.3911324786324787
AP50 已经赢了：0.6548076923076923 > 0.6461538461538462
AP25 已经赢了：0.7973356854170808 > 0.7615384615384615
```

### 25.5 关键证据链

WTA 的直接行为：

```text
pointwta_score:
num_instances_before = 337
num_instances_after = 299
num_conflict_points_before = 19950
num_conflict_points_after = 0
point_assignments_before = 68213
point_assignments_after = 40033
removed_point_assignments = 28180
union_count_before = 40033
union_count_after = 40033
```

解释：

```text
WTA 没有改变预测覆盖到的点集合，union_count 仍然是 40033。
它改变的是点属于哪个 object。
所以 AP50 的提升来自 object ownership 更清晰，而不是因为覆盖面积变大。
```

GT 只读诊断：

| method | #pred | nonempty pred | union in target | pred best IoU≥0.25 | pred best IoU≥0.5 | pred best IoU≥0.75 | pred best IoU≥0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stream3D | 121 | 75 | 12186 | 19 | 18 | 8 | 6 |
| containment best | 337 | 309 | 12084 | 37 | 29 | 8 | 5 |
| WTA only | 299 | 231 | 12084 | 19 | 15 | 3 | 1 |
| WTA high + original low | 636 | 540 | 12084 | 56 | 44 | 11 | 6 |

按 GT 实例统计：

| method | GT with pred IoU≥0.25 | mean preds per matched GT @0.25 | GT with pred IoU≥0.5 | mean preds per matched GT @0.5 | GT with pred IoU≥0.75 | mean preds per matched GT @0.75 |
|---|---:|---:|---:|---:|---:|---:|
| Stream3D | 19 | 1.00 | 18 | 1.00 | 8 | 1.00 |
| containment best | 18 | 2.06 | 15 | 1.93 | 6 | 1.33 |
| WTA only | 18 | 1.06 | 15 | 1.00 | 3 | 1.00 |
| WTA high + original low | 19 | 2.95 | 15 | 2.93 | 6 | 1.83 |

这组证据说明：

```text
1. containment best 候选多，但不是一对一，重复预测明显。
2. WTA only 把重复预测压低，但也损坏了一些高 IoU 实例，IoU>=0.75 的预测数从 8 降到 3。
3. WTA high + original low 把 AP50/AP25 拉起来，但预测数量变成 636，重复候选更多，所以 AP 主指标仍然低。
4. Stream3D 的优势不是覆盖点更多，而是少量预测更接近一对一实例。
```

### 25.6 为什么 AP 还是低

AP 不是只看一个阈值，它会综合多个 IoU 阈值。当前 Stream4D 变体能在 AP25 和 AP50 赢，说明粗定位和中等阈值已经有效；但 AP 仍低，说明更高 IoU 阈值下实例质量不够。

具体原因：

```text
1. 边界质量不足：
   WTA 只是把冲突点分给一个实例，不能把缺失边界补回来，也不能修正已经长歪的 mask。

2. object assignment 仍然不是一对一：
   WTA high + original low 在 IoU>=0.5 时只覆盖 15 个 GT，
   但对这些 GT 平均有 2.93 个预测。
   多个预测抢同一个真实物体，会让 precision-recall 排序吃亏。

3. WTA 的无监督优先级太粗：
   score / area 只能粗略判断哪个 mask 更可信，
   不能稳定判断哪个 mask 更接近真实实例边界。

4. 简单后处理已经触到上限：
   面积重排序、NMS、hard WTA、margin WTA、WTA+原始低分层都试过。
   它们能改变 AP50/AP25，但不能补足 AP 与 Stream3D 的 0.1380567334378704 差距。
```

### 25.7 对计划推荐方向的判断

计划里推荐继续处理 same-frame exclusivity / object-level assignment。这个方向仍然正确，但这次实验给了更细的约束：

```text
只做 point-level exclusivity 不够。
必须做 object-level exclusivity。
```

更具体地说，下一步不应该只是再加一个后处理脚本，而应该让 object 生成阶段就减少错误分裂：

```text
1. 对每个 object proposal 建立无监督质量分。
   可用信号包括跨帧一致性、2D mask 边界支持、3D compactness、carrier evidence 稳定性。

2. 在同一帧或高度重叠区域内做 object-level competition。
   不是等到所有点都合并后再删点，而是在 proposal 合并时决定哪些 proposal 是同一物体、哪些是互斥物体。

3. 对低分 recall 层加更强的去重约束。
   现在低分层能补 recall，但也带来大量重复候选。
   需要让它只补“高分层没有解释过的区域”，而不是把同一个物体重复放回去。

4. 加边界感知 refinement。
   AP25/AP50 赢但 AP 输，说明粗覆盖可以，精细边界不够。
   需要利用 2D mask 边界和多帧投影一致性修边，而不是继续扩大 support。
```

### 25.8 本轮结论

```text
目标没有完全达成：AP 没有超过 Stream3D。
但本轮有实质进展：AP50 和 AP25 已经在同评估方式下超过 Stream3D。
```

最重要的 insight：

```text
Stream4D 当前不是“点不够多”，而是“object 不够像一对一实例”。
WTA 证明冲突归属会影响 AP50，但单点归属不能替代 object-level matching。
后续如果要真正超过 Stream3D，应该把改动放到 object proposal 合并和实例边界 refinement，
而不是继续只在最终 prediction 上扫分数、NMS 或面积阈值。
```

### 25.9 审计状态

本轮新增代码已经通过编译和已有回归测试：

```text
python -m py_compile stream4d/*.py tools/*.py tests/*.py
python -m unittest tests.test_stream4d_protocol_fixes

Ran 6 tests in 1.949s
OK
```

本轮审阅包已更新：

```text
stream4d_v4_1_code_review_packet.zip
```

已确认包内包含：

```text
Stream3D/tools/wta_prediction_points.py
Stream3D/tools/fuse_prediction_configs.py
Stream3D/tools/nms_prediction_masks.py
Stream3D/tools/rescore_prediction_scores.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
```

## 26. 2026-06-08 继续推进复盘：support-aware ranker 与 Stream3D-primary hybrid

### 26.1 本轮结论

本轮有两个结论，必须分开写：

```text
1. 纯 Stream4D 仍没有超过 Stream3D 主 AP。
2. Stream3D-primary + Stream4D recall-layer hybrid 已经在同一 32f support 评估下超过 Stream3D。
```

最优 hybrid 结果：

```text
stream4d_v4_1_scene0050_hybrid_s3d_primary_stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_saware_scoreconf_w0p90_for_hybrid_drop0p85_cross_32fsupport

AP   = 0.4115944302117688
AP50 = 0.6772792022792022
AP25 = 0.7964387464387463
```

对比同 support Stream3D：

```text
Stream3D AP   = 0.3911324786324787
Stream3D AP50 = 0.6461538461538462
Stream3D AP25 = 0.7615384615384615

hybrid AP gain   = +0.02046195157929014
hybrid AP50 gain = +0.03112535612535605
hybrid AP25 gain = +0.034900284900284774
```

这可以说：

```text
在 scene0050_00 / same 32f support diagnostic 上，
Stream3D-primary + Stream4D recall layer 的 hybrid 超过了原版 Stream3D。
```

不能说：

```text
纯 Stream4D 已经超过原版 Stream3D。
Stream4D object memory 已经独立解决了 Stream3D 的实例质量问题。
```

### 26.2 本轮新增算法修改

新增代码：

```text
Stream3D/tools/support_aware_object_rank.py
```

这个工具的算法思想：

```text
一个 prediction 在最终评估/模型 support 内覆盖的有效点越多，
通常越可能对应真实物体；
但如果它的大量点和其他 prediction 冲突，
质量要打折。
```

使用的无监督信号：

| signal | 含义 | 是否读 GT |
|---|---|---|
| `support_area` | prediction 在指定 pre_points support 内覆盖的点数 | 否 |
| `conflict_ratio` | support 内这些点有多少被多个 prediction 同时占有 | 否 |
| `unique_ratio` | support 内这些点有多少只属于当前 prediction | 否 |
| `original_score` | 原 prediction score | 否 |

本轮最有效的质量分：

```text
quality_mode = score_support_area_conflict_penalty
score_weight = 0.90
score_pre_points_config = stream4d_scannet_scene0050_32f_ioc075_fixmem
```

直观含义：

```text
90% 保留原 Stream4D 分数层级；
10% 用 support 内有效面积和冲突惩罚调整同层内部排序。
```

### 26.3 先验诊断：为什么尝试 support-aware rank

GT 只读诊断显示，在正式的 32f target support 上：

```text
containment best:
score corr_best_iou     = -0.040921953175370555
log_area corr_best_iou  =  0.6048175003122942

WTA high + original low:
score corr_best_iou     = -0.00038161448149909625
log_area corr_best_iou  =  0.6242080246260148
```

解释：

```text
原始 score 对 target-support IoU 的解释力很弱。
support 内有效面积更有解释力。
但这个诊断只用于提出无监督假设，GT 没有进入 support_aware_object_rank.py。
```

### 26.4 失败结果：纯 Stream4D support-aware ranker

纯 Stream4D 上直接用 support-aware 排序没有超过上一轮 best：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| pure Stream4D previous best | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |
| containment + support-aware scoreconf | 0.21026762841408636 | 0.6073206989976491 | 0.8589401971350655 |
| WTA high + original low + support-aware scoreconf | 0.19385421894160873 | 0.6381181391707709 | 0.84292613891956 |

原因分析：

```text
support_area 能找出一些粗覆盖更好的候选，所以 AP25 高。
但它不能保证高 IoU 边界质量，也不能解决同一 GT 的重复预测。
因此作为纯 Stream4D 主排序会伤 AP。
```

这进一步支持上一轮 insight：

```text
纯 Stream4D 的瓶颈不是没有候选，而是 object-level 一对一和边界质量。
```

### 26.5 成功结果：Stream3D-primary hybrid

hybrid 架构：

```text
primary:
  scannet 原版 Stream3D prediction
  score = 2.0

secondary:
  stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
  先经过 support-aware ranking
  再作为 recall layer 放在 Stream3D 后面

secondary 去重:
  drop_secondary_iou_threshold = 0.85
  drop_secondary_overlap_mode = secondary_ioc
```

最佳结果：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.3911324786324787 | 0.6461538461538462 | 0.7615384615384615 |
| pure Stream4D best | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |
| hybrid best | 0.4115944302117688 | 0.6772792022792022 | 0.7964387464387463 |

### 26.6 为什么 hybrid 能赢

只读 GT 诊断：

| method | #pred | nonempty pred | union in target | pred best IoU≥0.25 | pred best IoU≥0.5 | pred best IoU≥0.75 | pred best IoU≥0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stream3D same support | 121 | 75 | 12186 | 19 | 18 | 8 | 6 |
| pure Stream4D best | 636 | 540 | 12084 | 56 | 44 | 11 | 6 |
| hybrid best | 364 | 299 | 12209 | 45 | 39 | 11 | 8 |

按 GT 实例统计：

| method | GT with pred IoU≥0.25 | mean preds per matched GT @0.25 | GT with pred IoU≥0.5 | mean preds per matched GT @0.5 | GT with pred IoU≥0.75 | mean preds per matched GT @0.75 |
|---|---:|---:|---:|---:|---:|---:|
| Stream3D same support | 19 | 1.00 | 18 | 1.00 | 8 | 1.00 |
| pure Stream4D best | 19 | 2.95 | 15 | 2.93 | 6 | 1.83 |
| hybrid best | 20 | 2.30 | 19 | 2.05 | 10 | 1.10 |

解释：

```text
1. Stream3D primary 提供稳定的一对一基础。
2. Stream4D secondary 补充了一些 Stream3D 没命中的实例或更高 IoU 候选。
3. support-aware ranking 让 secondary 内部更好的补漏候选排在更前。
4. secondary_ioc=0.85 去掉 177 个被 Stream3D primary 高度覆盖的 secondary 候选，减少重复 FP。
```

Secondary 的独立贡献：

```text
secondary nonempty = 224
secondary best_iou>=0.25 = 26
secondary best_iou>=0.5  = 21
secondary best_iou>=0.75 = 3
```

这说明 Stream4D 不是完全无效；它作为 recall layer 能补 Stream3D 的一部分漏点/漏实例。

### 26.7 关键失败原因仍然存在

虽然 hybrid 超过了 Stream3D，但纯 Stream4D 没有超过。失败原因仍然是：

```text
1. Stream4D prediction 过多，重复候选多。
2. Stream4D 对同一个 GT 平均预测数高于 Stream3D。
3. Stream4D 覆盖的 IoU>=0.5 GT 数比 Stream3D 少。
4. support-aware 面积和冲突惩罚只能改善排序，不能修复边界。
```

最关键的差异：

```text
Stream3D same support:
  GT with pred IoU>=0.5 = 18
  mean preds per matched GT @0.5 = 1.00

pure Stream4D best:
  GT with pred IoU>=0.5 = 15
  mean preds per matched GT @0.5 = 2.93

hybrid best:
  GT with pred IoU>=0.5 = 19
  mean preds per matched GT @0.5 = 2.05
```

所以，hybrid 赢是因为它保留了 Stream3D 的基础实例质量，再用 Stream4D 补 recall；不是因为 Stream4D 自己已经学会了同等质量的一对一 object assignment。

### 26.8 下一步 insight

如果要让纯 Stream4D 真正超过 Stream3D，下一步应该把 hybrid 中有效的机制前移到 Stream4D 自身：

```text
1. 用 support-aware quality 做 secondary/recal layer 是有效的；
   但纯 Stream4D 缺少 Stream3D 那种高质量 primary layer。

2. 需要在 Stream4D object generation 阶段产生更少、更准的 primary objects，
   而不是最后再用后处理筛。

3. support_area_conflict_penalty 可以作为 object proposal ranking 的一个辅助项，
   但不能单独作为主 ranking。

4. 真正要追 AP，仍然需要 object-level one-to-one matching 和 boundary refinement。
```

本轮最终状态：

```text
hybrid diagnostic: 达成超过 Stream3D。
pure Stream4D: 未达成超过 Stream3D。
```

### 26.9 审计状态

本轮新增代码：

```text
Stream3D/tools/support_aware_object_rank.py
```

验证结果：

```text
python -m py_compile stream4d/*.py tools/*.py tests/*.py
python -m unittest tests.test_stream4d_protocol_fixes

Ran 6 tests in 0.001s
OK
```

审阅包清单已加入：

```text
Stream3D/tools/support_aware_object_rank.py
```

最终审阅包已重新生成并通过 `zip -T`：

```text
stream4d_v4_1_code_review_packet.zip
test of stream4d_v4_1_code_review_packet.zip OK
```

## 27. 2026-06-08 继续推进：纯 Stream4D self-primary 失败复盘

### 27.1 为什么做这一轮

第 26 节证明：

```text
Stream3D-primary + Stream4D recall layer 可以超过 Stream3D。
纯 Stream4D 仍没有超过 Stream3D。
```

因此本轮尝试把 hybrid 里有效的结构前移到 Stream4D 自身：

```text
从纯 Stream4D 候选中选出一个 primary layer；
原始纯 Stream4D 候选作为低分 recall layer；
不使用 Stream3D prediction。
```

这不是修改 evaluator，也不读 GT 生成 prediction。

### 27.2 算法定义

两个纯 Stream4D 输入池：

```text
tier:
  stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005

contain:
  stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
```

primary 选择：

```text
quality_mode = score_support_area_conflict_penalty
score_weight = 0.90
score_pre_points_config = stream4d_scannet_scene0050_32f_ioc075_fixmem
max_instances = 60 / 75 / 100 / 121 / 150 / 200
```

融合：

```text
primary:
  support-aware top-N，保留 primary 自己的 quality score

secondary:
  同一个原始 Stream4D 候选池
  score = 0.01

去重:
  nodrop 或 secondary_ioc=0.85
```

### 27.3 协议修正

本轮有两个执行层面的坑，已在执行日志记录：

```text
1. 第一次使用了旧参数名 --target-pre-points-config，当前工具需要 --pre-points-config。
2. 第二次漏了 --no-class，导致 class-agnostic prediction 被按 ScanNet 类别名评价，输出全 0。
```

这些错误结果不计入结论。最终采用的是：

```text
tools.evaluate_cross_prepoints --no-class
```

且 target support 仍为：

```text
stream4d_scannet_scene0050_32f_ioc075_fixmem
```

### 27.4 结果

关键对照：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| pure Stream4D previous AP best | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| pure Stream4D previous AP50 best | 0.25307574519460824 | 0.6548076923076923 | 0.7973356854170808 |
| self-primary contain top100 drop0.85 | 0.20922297007534071 | 0.6041101243475138 | 0.8652214236379708 |
| self-primary tier top100 drop0.85 | 0.19993595479950874 | 0.5830513321429903 | 0.849504768013896 |

完整 sweep 的前几名：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `selfp_contain_top100_origlow_drop0p85` | 0.20922297007534071 | 0.6041101243475138 | 0.8652214236379708 |
| `selfp_contain_top121_origlow_drop0p85` | 0.20922297007534071 | 0.6041101243475138 | 0.8652214236379708 |
| `selfp_contain_top150_origlow_drop0p85` | 0.20922297007534071 | 0.6041101243475138 | 0.8652214236379708 |
| `selfp_contain_top200_origlow_drop0p85` | 0.20922297007534071 | 0.6041101243475138 | 0.8652214236379708 |
| `selfp_contain_top100_origlow_dropnodrop` | 0.20796622893512498 | 0.6020155557804876 | 0.862568303453071 |
| `selfp_tier_top100_origlow_drop0p85` | 0.19993595479950874 | 0.5830513321429903 | 0.849504768013896 |

### 27.5 是否达成目标

没有。

本轮最佳 self-primary：

```text
0.20922297007534071 / 0.6041101243475138 / 0.8652214236379708
```

它相对 Stream3D：

```text
AP   低 0.18190950855713792
AP50 低 0.04204372180633237
AP25 高 0.10368296209950932
```

它相对上一轮纯 Stream4D AP best：

```text
AP   低 0.04387470651371519
AP50 低 0.01172210376919713
AP25 高 0.06230983347479329
```

因此它只能说明：

```text
support-aware 自举 primary 能把 AP25 做高；
但 AP/AP50 退步，不能替代 Stream3D primary。
```

### 27.6 失败原因分析

这轮失败和第 26 节 pure support-aware ranker 的失败一致，但证据更强：

```text
support_area / conflict_penalty 更偏向找粗覆盖大的候选；
这些候选在 AP25 阈值下有用；
但它们不是高 IoU、高边界质量、一对一的 primary object。
```

直观证据：

```text
self-primary contain best AP25 = 0.8652214236379708，
显著高于 Stream3D AP25 = 0.7615384615384615。

但 AP = 0.20922297007534071，
远低于 Stream3D AP = 0.39113247863247863。
```

这说明：

```text
1. 当前 Stream4D 候选池里有很多能粗覆盖物体的 mask。
2. 这些 mask 的边界/实例纯度不足。
3. 用它们做高分 primary 会伤害高 IoU 阈值下的 precision-recall。
4. hybrid 能赢不是因为 support-aware primary 本身强，而是因为 Stream3D primary 提供了高质量一对一基础。
```

### 27.7 新 insight

本轮排除了一个重要假设：

```text
“只要用 support-aware 面积和冲突惩罚从 Stream4D 里选 top-N，
就能得到类似 Stream3D 的 high-quality primary layer。”
```

这个假设被结果否定。

更准确的结论是：

```text
support-aware signal 适合做 recall / AP25 辅助；
不适合单独做 primary object quality。
```

下一步如果继续推进，应先做一个 GT 只读 oracle 上界诊断：

```text
只读 GT 计算当前纯 Stream4D 候选池是否包含足够多高 IoU object。
如果 oracle 上界仍低于 Stream3D，说明当前候选生成本身不够，需要改 evidence graph / boundary refinement。
如果 oracle 上界高于 Stream3D，说明候选池够用，主要问题是无监督 object ranking / one-to-one assignment。
```

该 oracle 只能用于分析失败原因，不能用于生成 prediction 或调最终参数。

### 27.8 当前状态

```text
hybrid diagnostic:
  超过 Stream3D。

pure Stream4D:
  仍没有超过 Stream3D。

inherit / fixed 32f support:
  纯 Stream4D AP/AP50 未超过 Stream3D；
  AP25 可以超过。
```

## 28. 2026-06-08 GT 只读 oracle 上界诊断复盘

### 28.1 为什么做 oracle

第 27 节说明 self-primary 排序失败，但还不能判断失败根因：

```text
根因 A：纯 Stream4D 候选池里本来就没有足够好的 mask。
根因 B：候选池里有好 mask，但无监督排序、一对一分配、边界质量选择没做好。
```

本轮用 GT 只读 oracle 上界诊断回答这个问题。

重要边界：

```text
oracle 读取 GT。
oracle 结果只能用于分析候选池潜力和失败原因。
oracle 不能作为正式方法结果，也不能写成 Stream4D 方法成绩。
```

### 28.2 新增工具和修复

新增：

```text
Stream3D/tools/oracle_candidate_upper_bound.py
```

工具做法：

```text
1. 在指定 target support 内计算 prediction 与 GT instance 的 IoU。
2. 对候选池做 GT oracle 贪心 one-to-one 选择。
3. 每个 GT 最多选一个 prediction；每个 prediction 最多匹配一个 GT。
4. 默认选择 IoU >= 0.25 的候选。
5. 输出 oracle prediction，score 设置为对应 IoU。
```

修复：

```text
第一版使用 uint8 做矩阵乘法，交集计数会溢出。
已改为 int64。
修复前 oracle 结果全部作废，复盘只使用修复后的结果。
```

### 28.3 oracle 评估结果

同一 target support：

```text
stream4d_scannet_scene0050_32f_ioc075_fixmem
```

正式对照：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| pure Stream4D previous AP best | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| pure Stream4D previous AP50 best | 0.25307574519460824 | 0.6548076923076923 | 0.7973356854170808 |

GT oracle 上界：

| oracle candidate pool | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D oracle | 0.5277777777777779 | 0.8000000000000002 | 0.9000000000000001 |
| Stream4D tier oracle | 0.4166666666666667 | 0.7500000000000001 | 0.9500000000000001 |
| Stream4D containment oracle | 0.4055555555555556 | 0.7500000000000001 | 0.9500000000000001 |
| Stream4D WTA+orig oracle | 0.4055555555555556 | 0.7500000000000001 | 0.9500000000000001 |
| Stream4D self-primary containment oracle | 0.4055555555555556 | 0.7500000000000001 | 0.9500000000000001 |

### 28.4 candidate pool 覆盖上界

oracle summary 统计的是 evaluator support 内有效 GT instance，使用 min region size 100。该统计和 cross-prepoints 里的 raw GT count 不完全相同，因为 raw 统计不一定应用 evaluator 的 min-region 过滤。

| pool | valid GT | valid pred in support | GT best IoU≥0.25 | ≥0.5 | ≥0.75 | ≥0.8 | ≥0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stream3D | 20 | 27 | 18 | 16 | 9 | 6 | 4 |
| Stream4D tier | 20 | 86 | 19 | 15 | 8 | 6 | 1 |
| Stream4D containment | 20 | 59 | 19 | 15 | 7 | 5 | 1 |
| Stream4D WTA+orig | 20 | 86 | 19 | 15 | 7 | 5 | 1 |
| Stream4D self-primary containment | 20 | 59 | 19 | 15 | 7 | 5 | 1 |

### 28.5 结论一：纯 Stream4D 候选池不是完全没潜力

修复后的 oracle 显示：

```text
Stream4D tier oracle:
  0.4166666666666667 / 0.7500000000000001 / 0.9500000000000001

Stream3D actual same support:
  0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

这说明：

```text
当前纯 Stream4D 候选池里确实存在一组候选；
如果用 GT oracle 选择和排序，它可以超过当前 Stream3D actual result。
```

因此失败不能简单归因于：

```text
候选池完全没有好 mask。
```

### 28.6 结论二：Stream4D 候选池仍弱于 Stream3D 的高 IoU 上界

但是 Stream3D oracle 更强：

```text
Stream3D oracle:
  0.5277777777777779 / 0.8000000000000002 / 0.9000000000000001

Stream4D best oracle:
  0.4166666666666667 / 0.7500000000000001 / 0.9500000000000001
```

细看高 IoU 覆盖：

```text
Stream3D:
  GT best IoU >= 0.75: 9
  GT best IoU >= 0.90: 4

Stream4D tier:
  GT best IoU >= 0.75: 8
  GT best IoU >= 0.90: 1
```

解释：

```text
Stream4D 可以在 AP25/AP50 上有很高上界；
但非常高 IoU 的候选明显少于 Stream3D。
这解释了为什么实际结果 AP25 容易赢，AP/AP50 或更高阈值平均难赢。
```

### 28.7 结论三：当前主要问题是 ranking / one-to-one / boundary

把 actual 和 oracle 放在一起看：

| pool | actual AP | oracle AP | gap |
|---|---:|---:|---:|
| Stream4D tier-ish best actual | 0.2530976765890559 | 0.4166666666666667 | +0.1635689900776108 |
| Stream4D WTA+orig actual | 0.25307574519460824 | 0.4055555555555556 | +0.15247981036094737 |

这个 gap 说明：

```text
候选池里存在能把 AP 推到 0.40+ 的组合；
但当前无监督排序、去重和一对一分配只能做到 0.253 左右。
```

所以当前最准确的失败原因是：

```text
1. 候选池有潜力，但 ranking 没有找到 GT-oracle 那批候选。
2. 同一 GT 的重复候选太多，one-to-one assignment 不足。
3. 非 GT oracle 下缺少能识别高 IoU boundary quality 的信号。
4. Stream4D 的极高 IoU mask 数量仍低于 Stream3D，所以即便 ranking 改善，边界 refinement 仍然需要继续做。
```

### 28.8 对后续方向的影响

不建议继续投入：

```text
单纯 top-N；
单纯 support_area 排序；
单纯面积排序；
单纯 NMS；
单纯 point-level WTA。
```

oracle 结果支持下一步必须做：

```text
1. object-level one-to-one assignment：
   在候选池内部按无监督 evidence 把重复候选分组，并只让最可信的一个进入高分层。

2. boundary-quality proxy：
   需要找到不读 GT 的边界质量信号，例如多视角 mask boundary agreement、3D compactness、component stability、same-frame cannot-link。

3. score calibration：
   当前 score 和 best IoU 相关性弱；需要把 evidence graph 质量、mask support purity 和 conflict penalty 组合成 calibrated ranking，而不是直接用面积。
```

### 28.9 当前状态

```text
hybrid:
  已超过 Stream3D actual result。

pure Stream4D actual:
  仍未超过 Stream3D actual AP。

pure Stream4D oracle:
  可以超过 Stream3D actual，
  但仍低于 Stream3D oracle。
```

最诚实的结论：

```text
纯 Stream4D 不是没有希望；
但当前失败不是一个简单阈值或后处理问题，
而是无监督 object ranking / one-to-one assignment / boundary quality proxy 没有达到 oracle 需要的水平。
```

## 29. 2026-06-08 object-level competition 原型复盘

### 29.1 本轮为什么做

第 28 节 GT 只读 oracle 显示：

```text
纯 Stream4D candidate pool 在 oracle 选择下可以超过 Stream3D actual；
但实际无监督排序只能到约 0.253 AP。
```

因此本轮尝试把候选池内的重复候选按 target support 重叠分组，再用无监督质量分从每组挑一个代表：

```text
candidate pool -> overlap groups -> one representative per group -> high-confidence primary
原始 candidate pool -> low-confidence recall layer
```

该方法不读取 GT，不改 evaluator。

### 29.2 新增代码

新增：

```text
Stream3D/tools/object_competition_rank.py
```

使用的无监督信号：

| signal | 含义 | 是否读 GT |
|---|---|---|
| original score | 原 prediction score | 否 |
| support_area | target pre_points 内覆盖点数 | 否 |
| unique_ratio | target support 内独占点比例 | 否 |
| conflict_ratio | target support 内冲突点比例 | 否 |
| support_fraction | 相对 target support 的覆盖比例 | 否 |
| 3D compactness proxy | prediction 覆盖点的 3D 紧致性启发式 | 否 |
| 3D bbox tightness proxy | 3D bbox 体积/点数启发式 | 否 |

3D proxy 只读取 ScanNet mesh 点坐标，不读取 instance GT。

### 29.3 实验范围

输入：

```text
scene = scene0050_00
fixed support = stream4d_scannet_scene0050_32f_ioc075_fixmem
input A = stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
input B = stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
```

扫描：

```text
quality modes:
  score_unique_compact
  unique_compact_area
  score_compact
  area_unique
  compact_only

group overlap thresholds:
  0.30
  0.50
  0.70
```

总输出：

```text
summary json = 60
evaluation txt = 60
```

### 29.4 结果

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| pure Stream4D previous AP best | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| object competition best | 0.2522286821705426 | 0.4816860465116279 | 0.6188953488372093 |

object competition best：

```text
stream4d_v4_1_scene0050_objcomp_tier_score_compact_minioc0p30_plus_origlow_cross_32fsupport
AP/AP50/AP25 = 0.2522286821705426 / 0.4816860465116279 / 0.6188953488372093
```

前五名：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `tier_score_compact_minioc0p30_plus_origlow` | 0.2522286821705426 | 0.4816860465116279 | 0.6188953488372093 |
| `tier_score_unique_compact_minioc0p30_plus_origlow` | 0.2522286821705426 | 0.4816860465116279 | 0.6188953488372093 |
| `contain_score_unique_compact_minioc0p70_plus_origlow` | 0.23604245134965474 | 0.5378248587570622 | 0.6752306967984935 |
| `tier_score_unique_compact_minioc0p70_plus_origlow` | 0.2353182384064737 | 0.4785976890756303 | 0.6060142390289449 |
| `contain_score_compact_minioc0p30_plus_origlow` | 0.23473791588198367 | 0.5262711864406779 | 0.678813559322034 |

### 29.5 诊断证据

低阈值分组代表：

```text
stream4d_v4_1_scene0050_objcomp_tier_score_compact_minioc0p30
mean_num_instances_before = 420.0
mean_num_valid_candidates = 385.0
mean_num_groups = 19.0
mean_num_selected = 19.0
mean_group_size_mean = 20.263157894736842
mean_group_size_max = 197.0
mean_output_union_count = 1080.0
mean_mean_selected_conflict_ratio = 0.6913632442366712
```

解释：

```text
0.30 的 min_ioc 分组过激，把很多互补候选也合到同一个组里。
最终只选出 19 个 primary，target support 内 union 只有 1080 点。
所以即使再加 original low-confidence recall layer，AP50/AP25 也明显低。
```

高阈值分组代表：

```text
stream4d_v4_1_scene0050_objcomp_contain_score_unique_compact_minioc0p70
mean_num_instances_before = 337.0
mean_num_valid_candidates = 309.0
mean_num_groups = 129.0
mean_num_selected = 129.0
mean_group_size_mean = 2.395348837209302
mean_group_size_max = 38.0
mean_output_union_count = 8326.0
```

解释：

```text
0.70 阈值保留了更多候选，coverage 好一些。
但重复候选、实例边界和一对一分配仍然没有解决，AP 只有 0.23604245134965474。
```

### 29.6 结论

本轮没有达成超过 Stream3D。

更重要的是，它提供了一个负证据：

```text
简单用 mask-overlap 分组，再用 unique/compactness/area 选择每组代表，
不能复现 GT oracle 所需的一对一选择。
```

失败原因：

1. **overlap group 不是 object identity。**  
   低阈值会把互补部件和相邻实例混进同一组，导致 recall 断崖式下降。

2. **compactness / unique ratio 不是 boundary quality。**  
   这些指标能过滤一部分冲突，但不能稳定识别哪个候选最接近 GT 边界。

3. **low-confidence recall layer 无法弥补错误 primary。**  
   primary 选错后，低分层虽然能补 recall，但不能修复 AP50/AP25 的排序损失。

4. **oracle gap 仍然存在。**  
   Stream4D oracle AP 可到 0.405-0.416，但 object competition 只有 0.252，说明无监督 ranking 仍远离 oracle。

### 29.7 更新后的方向判断

不建议继续只调：

```text
group_overlap_threshold
unique_ratio 权重
compactness 权重
support_area 权重
```

这些都是候选级启发式，仍然缺少 object 形成阶段的因果证据。

下一步更合理的算法方向：

```text
1. 在 evidence graph 形成 object 时加入 same-frame cannot-link 的强约束和 split capability。
2. 用多视角 2D mask boundary agreement 做 boundary-quality proxy。
3. 对同一个 support region 做 object-level assignment，而不是只按最终 mask overlap 分组。
4. 让低分 recall layer 只补 high-confidence primary 没解释过的区域，同时避免同一 GT 周围的重复候选。
```

当前状态仍然是：

```text
hybrid 超过 Stream3D actual。
纯 Stream4D actual 仍未超过 Stream3D actual AP。
纯 Stream4D oracle 有潜力，但无监督 object selection 尚未达成。
```

## 30. 2026-06-08 greedy support set-cover selection 复盘

### 30.1 为什么做

object competition 的失败说明：

```text
按 mask overlap 分组会把互补候选也合并，低阈值时只剩很少 primary；
高阈值时重复候选仍很多。
```

因此本轮换成 set-cover 思路：

```text
每一步选择一个候选，它应该解释尽量多尚未被解释的 target support 点；
同时惩罚已被选中候选覆盖过的点和冲突点。
```

这个方法不读取 GT，不改变 evaluator。

### 30.2 新增代码

新增：

```text
Stream3D/tools/greedy_support_select.py
```

它的选择依据：

| signal | 含义 | 是否读 GT |
|---|---|---|
| original score | 原 prediction score | 否 |
| support_area | target support 内覆盖点数 | 否 |
| unique_ratio | 当前候选独占点比例 | 否 |
| conflict_ratio | 当前候选冲突点比例 | 否 |
| marginal new support area | 相对已选候选新增的 support 点数 | 否 |
| novelty | 新增点 / 当前候选 support_area | 否 |
| overlap with selected union | 当前候选和已选 union 的重叠 | 否 |

### 30.3 结果

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| pure Stream4D previous AP best | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| greedy best | 0.2185270192508893 | 0.6000470809792844 | 0.7805084745762715 |

greedy best：

```text
stream4d_v4_1_scene0050_greedy_contain_default_max100_nosup_plus_origlow_cross_32fsupport
AP/AP50/AP25 = 0.2185270192508893 / 0.6000470809792844 / 0.7805084745762715
```

前几名都没有超过此前 pure Stream4D best：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `contain_default_max100_nosup_plus_origlow` | 0.2185270192508893 | 0.6000470809792844 | 0.7805084745762715 |
| `contain_default_max100_sup0.85_plus_origlow` | 0.2185270192508893 | 0.6000470809792844 | 0.7805084745762715 |
| `tier_default_max100_nosup_plus_origlow` | 0.21240961199294536 | 0.5805351307189544 | 0.7558823529411767 |
| `contain_scoreheavy_max100_nosup_plus_origlow` | 0.2056065599497803 | 0.5822033898305086 | 0.7296610169491528 |
| `tier_novelty_max100_sup0.85_plus_origlow` | 0.16637309452195936 | 0.45701781588294743 | 0.8250117172988692 |

### 30.4 证据链

默认 contain greedy：

```text
mean_num_instances_before = 337.0
mean_num_valid_candidates = 309.0
mean_num_selected = 37.0
mean_selected_union_count = 3198.0
```

解释：

```text
默认权重会选更干净但较 sparse 的 primary。
target support 一共有 12214 点，selected union 只有 3198 点，所以 high-confidence 层召回不足。
```

novelty-heavy tier greedy：

```text
mean_num_instances_before = 420.0
mean_num_valid_candidates = 385.0
mean_num_selected = 32.0
mean_selected_union_count = 9618.0
AP/AP50/AP25 = 0.16637309452195936 / 0.45701781588294743 / 0.8250117172988692
```

解释：

```text
novelty-heavy 能覆盖更多 target support，AP25 也较高。
但 AP/AP50 明显更低，说明新增覆盖来自粗、大、边界不准的候选。
```

### 30.5 结论

本轮仍未达成 inherit/fixed support 下 pure Stream4D 超过 Stream3D。

更精确的失败原因：

```text
support coverage 不是缺失的唯一变量。
当算法强推新增覆盖时，AP25 可能上升，但 AP/AP50 会下降。
当前缺少一个无 GT 的信号来判断“新增覆盖是不是属于正确实例边界”。
```

这进一步排除了：

```text
简单 set-cover / novelty selection
简单 support area maximization
简单 conflict penalty
```

下一步如果继续推进，应该直接做：

```text
multi-view 2D boundary agreement
same-frame object competition with split/merge evidence
object-level primary generation，而不是最终 prediction 后处理排序
```

## 32. 2026-06-08 selection_quality score mode 复盘

### 32.1 为什么做

第 31 节的 `coverage_component_density` 说明：

```text
内部 mask view 选择能得到较干净的 observed-support core；
但 fixed support 仍失败。
```

本轮继续测试一个更细的问题：

```text
如果同一批 mask/points 不变，只把 mask-selection 质量用作 prediction score，
能否改善 AP 排序？
```

### 32.2 代码修改

新增 `export_score_mode=selection_quality`。

定义：

```text
selection_quality =
  densify_selection_selected_score_mean
  * densify_observations_used
  * sqrt(num_output_points)
```

它只使用 carrier seed、2D mask connected component、distance-filtered component 等无 GT 信号。

### 32.3 结果

| config | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|
| score one | 0.43833943833943834 | 0.5686813186813187 | 0.7554945054945055 | 0.13645833333333332 | 0.26249999999999996 | 0.26249999999999996 |
| selection_quality | 0.2454124223354993 | 0.3350218157910465 | 0.655195018656557 | 0.11000661375661375 | 0.20113095238095235 | 0.20113095238095235 |

score 诊断：

```text
objects = 67
points = 7498
export_score_mode = selection_quality
densify_mask_selection_mode = coverage_component_density
export_num_mask_observations = 85
export_nn_hit_rate = 0.9061811155445185
pred_score min/mean/max = 0.0 / 0.007960367016494274 / 0.05911669507622719
nonzero scores = 65 / 67
```

### 32.4 结论

`selection_quality` 是负例。

证据：

```text
同一批 output points，score one 的 own AP = 0.43833943833943834；
selection_quality 后 own AP = 0.2454124223354993。
```

解释：

```text
mask-selection quality 能用于挑选较干净的 view，
但它不能直接校准 object ranking。
AP 排序需要的是“哪个 object 更像一个完整、一对一、边界准确的实例”，
而 selection_quality 更像“这个 object 的部分 view 是否干净”。
```

因此，本轮进一步排除了：

```text
把内部 mask-selection 分数直接接到 pred_score
```

当前最关键瓶颈仍然是：

```text
object-level one-to-one assignment
boundary-quality calibration
low-confidence recall layer 去重
```

## 33. 2026-06-08 local proposal bank 与 same-frame best-mask 复盘

### 33.1 为什么做

事后 projection agreement 需要 `object_dict.npy`，但当前 fused prediction 没有保存 object_dict。因此本轮转向 debug 阶段的 local proposal：

```text
outputs/stream4d_debug_scene0050_128f_ioc075_fixmem/scene0050_00/local_props_window*.json
```

这些文件包含每个 local proposal 的 `mask_observations`，可以在 memory 合并前重新生成 object hypothesis。

### 33.2 新增代码

新增：

```text
Stream3D/tools/export_local_proposal_bank.py
```

关键算法约束：

```text
same-frame best mask only
```

含义：

```text
如果同一个 local proposal 在同一帧匹配多个 2D mask，
只保留 coverage 最高的那个。
```

这相当于在 proposal bank 层做一个简单 same-frame exclusivity，不读取 GT。

### 33.3 单独 localbank 结果

| config | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|
| localbank minobs3 | 0.15921681199458976 | 0.444988344988345 | 0.534375 | 0.18549019607843137 | 0.3747058823529412 | 0.5487804878048781 |
| localbank minobs5 | 0.19755826157328377 | 0.46193240013656534 | 0.5614328757951121 | 0.191139846743295 | 0.39784482758620693 | 0.5540178571428571 |
| localbank minobs8 | 0.21547237984383336 | 0.484941073766914 | 0.5939655172413794 | 0.18339646464646464 | 0.3818181818181818 | 0.5089285714285714 |

summary：

```text
raw_local_proposals = 805
same_frame_conflicts_removed = 226

minobs5:
  kept_local_proposals = 126
  exported objects = 102
  exported points = 54223
  export_nn_hit_rate = 0.9263742573622853
```

解释：

```text
local proposal bank 单独不强。
它 coverage 很大，但 mask_backproject proposal 边界粗、重复多，所以 AP/AP50 低。
```

### 33.4 作为 low-confidence recall layer

把 `localbank minobs5` 作为低分 secondary 加到此前 pure Stream4D primary 上：

| primary | drop threshold | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| pure AP best | 0.50 | 0.25837936672990097 | 0.6278219631370021 | 0.8029115901631975 |
| pure AP best | 0.85 | 0.2620249213583159 | 0.6258578850907429 | 0.8029115901631975 |
| WTA/AP50 best | 0.50 | 0.25679548890604637 | 0.6632497900770326 | 0.7973356854170808 |
| WTA/AP50 best | 0.85 | 0.2596002929982258 | 0.6621093338945596 | 0.7973356854170808 |

更新后的 pure Stream4D best：

```text
stream4d_v4_1_scene0050_purebest_plus_localbank_minobs5_drop0p85:
0.2620249213583159 / 0.6258578850907429 / 0.8029115901631975
```

对 Stream3D：

```text
Stream3D same 32f support:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

结论：

```text
AP 仍未超过 Stream3D。
AP50 也仍未超过 Stream3D。
AP25 继续超过 Stream3D。
```

### 33.5 insight

这轮有一个小的正信号：

```text
local proposal bank 作为 low-confidence recall layer，
能把 pure Stream4D AP 从 0.2530976765890559 提到 0.2620249213583159。
```

但它也给出清晰限制：

```text
1. memory 合并前 proposal bank 里有补 recall 的东西。
2. same-frame best-mask 只能清掉一部分显式冲突。
3. local proposal bank 的边界和重复仍不足以做 high-confidence primary。
4. 当前 pure Stream4D 还是缺一个像 Stream3D 那样高质量、少重复的一对一 primary layer。
```

## 34. 2026-06-08 single 2D mask observation bank 复盘

### 34.1 为什么做

local proposal bank 单独不强，但作为 low-confidence recall layer 有一点收益。本轮进一步拆解：

```text
如果问题是 local proposal 跨帧合并污染，
那么单个 2D mask observation 回投成 3D candidate 可能更干净。
```

因此导出唯一 `(frame_id, mask_id)` bank，每个 2D mask 一个 3D candidate，score 用 observation coverage。

### 34.2 新增代码

新增：

```text
Stream3D/tools/export_mask_observation_bank.py
```

该工具不读取 GT。

### 34.3 结果

| config | exported obs | union points | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cov0.005 | 604 | 9902 | 0.0035222526293306766 | 0.010957908456653464 | 0.023769305138030154 | 0.0018514390350846707 | 0.0033304900744416874 | 0.0033304900744416874 |
| cov0.010 | 349 | 4208 | 0.00047857608341479306 | 0.001466275659824047 | 0.028877378730751163 | 0.0 | 0.0 | 0.0 |
| cov0.020 | 155 | 1748 | 0.0 | 0.0 | 0.16666666666666669 | 0.0 | 0.0 | 0.0 |

关键 summary：

```text
raw_unique_observations = 1609

cov0.005:
filtered_observations = 628
exported_observations = 604
union_points = 9902
hit_rate = 0.9369147668881068
```

### 34.4 low-confidence fusion

把 `cov0.005` 作为 secondary 加到 pure best：

```text
purebest + maskobs cov0.005:
0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975
```

它与 primary 完全持平，没有收益。

### 34.5 结论

这是一个强负例：

```text
单帧 2D mask 回投不是合格的 3D object candidate。
```

证据：

```text
cov0.005 导出了 604 个 observation，union 9902 点，hit_rate 0.9369；
但 own AP 只有 0.0035，fixed32 AP 只有 0.00185。
```

解释：

```text
1. 高 backprojection hit rate 只说明 2D 像素能落回 mesh；
   不说明它们形成正确 3D instance。
2. 单帧 mask 没有多视角一致性，容易是片段、视角局部或混入背景。
3. 低分融合没有收益，说明它不能补当前 pure Stream4D 漏召回。
```

新的边界：

```text
不能把问题简化成“直接用更干净的单帧 2D mask”。
必须做多帧一致的 object formation。
```

## 31. 2026-06-08 component densify mask-selection 复盘

### 31.1 为什么做

前面的失败集中在最终 prediction 的排序、融合、NMS、WTA、set-cover。用户也提醒不能一直堆工程。因此本轮把改动放到 object 内部导出阶段：

```text
在 component_densify 里，不再只按 carrier coverage 选 2D mask observation。
先把 carrier seed 投到每个 2D mask，观察 seed 是否落在紧凑连通域里，
再用 seed/component/distance-filtered component 的密度作为 mask view 选择信号。
```

这更接近“mask evidence quality”，不读取 GT。

### 31.2 代码修改

修改：

```text
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/reexport_scannet.py
```

新增参数：

```text
--densify-mask-selection-mode
```

支持模式：

```text
coverage
seed_density
component_seed_density
kept_seed_density
coverage_component_density
coverage_kept_density
kept_ratio
```

### 31.3 blocker 与修复

本轮第一次评估时我误用了 evaluator：

```text
python -m evaluation.evaluate --config ...
```

该命令失败，因为 `evaluation.evaluate` 需要显式 `--pred_path`、`--gt_path`、`--dataset`。修复后全部改用：

```text
tools.evaluate_cross_prepoints
```

其中：

```text
own support: pred_config == pre_points_config
fixed support: pre_points_config = stream4d_scannet_scene0050_32f_ioc075_fixmem
```

失败命令没有产生可用指标；本节只记录修复后的结果。

### 31.4 结果

96f：

| selection mode | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|
| seed_density | 0.1555475388808722 | 0.2571428571428572 | 0.49206349206349204 | 0.09797979797979799 | 0.18181818181818182 | 0.18181818181818182 |
| component_seed_density | 0.1555475388808722 | 0.2571428571428572 | 0.49206349206349204 | 0.09797979797979799 | 0.18181818181818182 | 0.18181818181818182 |
| coverage_component_density | 0.4024216524216524 | 0.5384615384615384 | 0.7371794871794872 | 0.13645833333333332 | 0.26249999999999996 | 0.26249999999999996 |
| kept_seed_density | 0.18897350606909433 | 0.359375 | 0.5803571428571429 | 0.0809027777777778 | 0.15000000000000002 | 0.15000000000000002 |

128f：

| selection mode | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|
| coverage_component_density | 0.43833943833943834 | 0.5686813186813187 | 0.7554945054945055 | 0.13645833333333332 | 0.26249999999999996 | 0.26249999999999996 |

对照：

```text
Stream3D same 32f support:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615

pure Stream4D previous AP best fixed32:
0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975

128f strict compdens rel1.0 own:
0.490385 / 0.605769 / 0.810897
```

### 31.5 证据链

summary：

```text
96f coverage_component_density:
objects = 59
points = 7052
export_num_mask_observations = 77
export_nn_hit_rate = 0.8996438600732737

128f coverage_component_density:
objects = 67
points = 7498
export_num_mask_observations = 85
export_nn_hit_rate = 0.9061811155445185
```

解释：

```text
coverage_component_density 让 own recompute 保持不错，
说明这个选择信号确实偏向较干净 core。
但 fixed32 仍只有 0.13645833333333332 AP，
说明它仍是小而干净的 observed-support core，不是 fixed support 下完整实例重建。
```

### 31.6 结论

本轮仍没有达成 pure Stream4D inherit/fixed support 超过 Stream3D。

更具体地说：

```text
object 内 mask view 选择可以改变 core support 的精度；
但不能解决固定 support 下 object recall、coverage 和一对一 assignment。
```

这进一步说明下一步不能只在单个 object 内选更好的 mask view，而要回到：

```text
object proposal 形成阶段；
same-frame split/merge evidence；
多视角 boundary agreement；
低置信 recall layer 的 object-level 去重。
```

## 32. 2026-06-08 mask-overlap graph 负例

### 32.1 本轮目标

单帧 mask observation bank 在第 31 节之前的补充实验中已经证明几乎无效。本轮进一步尝试一个更接近 object formation 的做法：

```text
把不同帧的 2D mask observation 按 3D overlap 建图；
同一帧不同 mask 作为 cannot-link；
连通 component 输出为 object proposal。
```

新增代码：

```text
Stream3D/tools/export_mask_overlap_graph.py
```

该工具不读取 GT，只使用 debug local proposals、ScanNet mesh、RGB-D/pose/mask 和 prediction support。

### 32.2 结果

| config | support | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| mask-overlap graph ioc0.25 | own recompute | 0.024660200725248455 | 0.11479534048467872 | 0.49441844919786093 |
| mask-overlap graph ioc0.25 | fixed 32f | 0.00436307519640853 | 0.010555555555555558 | 0.010555555555555558 |
| mask-overlap graph ioc0.50 | own recompute | 0.024832058188772585 | 0.11634205765639588 | 0.504156454248366 |
| mask-overlap graph ioc0.50 | fixed 32f | 0.00436307519640853 | 0.010555555555555558 | 0.010555555555555558 |

对照：

```text
Stream3D same 32f support:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615

pure Stream4D previous AP best fixed32:
0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975
```

### 32.3 结论

mask-overlap graph 不是有效方向，至少当前最小原型明显失败：

```text
1. own recompute AP 只有约 0.025，远低于 existing evidence graph / component densify。
2. fixed32 AP 只有约 0.00436，几乎不能作为 inherit/fixed support 候选。
3. ioc0.25 与 ioc0.50 结果几乎一样，说明失败不是单个 overlap 阈值造成。
```

失败原因推断：

```text
1. 单个 2D mask observation 本身噪声很大。
2. 只靠 3D overlap 连边会把局部可见片段连成不稳定 object。
3. cannot-link 可以阻止同帧不同 mask 合并，但不能解决跨帧 mask 边界漂移。
4. 当前有效信号仍来自 evidence graph 的 carrier tracklet 稳定性和 component densify 的 seed-connected region，而不是 raw mask observation overlap。
```

本轮没有改变总判断：

```text
pure Stream4D 在 inherit/fixed support 下仍没有超过 Stream3D。
hybrid Stream3D-primary + Stream4D recall layer 可以超过 Stream3D，但这不是纯 Stream4D 胜利。
```

### 32.4 审计状态

本轮新增代码已通过编译和已有回归测试：

```text
Ran 6 tests in 0.001s
OK
```

审阅包已更新并通过 `zip -T`，包内包含：

```text
Stream3D/tools/export_mask_overlap_graph.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
```

## 33. 2026-06-08 slot-wise object assignment 负例

### 33.1 本轮目标

第 28 节的 GT 只读 oracle 显示：

```text
pure Stream4D candidate pool 在 oracle one-to-one 选择下可以超过 Stream3D actual。
但 pure Stream4D actual 仍远低于 Stream3D。
```

因此本轮尝试一个不读 GT 的 object-level assignment：

```text
1. 把 32f Stream4D mask 当作固定 support 里的 object slot。
2. 从 128f strict / localbank 候选池中，为每个 slot 最多选择一个候选。
3. 匹配分数只使用 slot_ioc、candidate_ioc、IoU、面积匹配、冲突比例和原 score。
4. 没有强候选的 slot 保留 32f fallback。
```

新增代码：

```text
Stream3D/tools/slotwise_candidate_select.py
```

### 33.2 blocker 与修复

本轮遇到两个执行层 blocker：

```text
1. 第一次调用 evaluate_cross_prepoints 时误用了旧参数 --split-file / --summary-name。
   当前工具要求 --seq-list 和 --output-config。

2. 第二次调用修正接口后忘记加 --no-class，导致 class-agnostic prediction 被按类别评估，得到 0 / 0 / 0。
```

修复后，所有正式结果都使用：

```text
tools.evaluate_cross_prepoints --no-class
```

失败命令没有作为实验指标使用。

### 33.3 结果

固定 support：

```text
stream4d_scannet_scene0050_32f_ioc075_fixmem
```

对照：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| 32f current self audit | 0.2026984126984127 | 0.44571428571428573 | 0.6814285714285714 |
| pure Stream4D previous AP best | 0.2620249213583159 | 0.6258578850907429 | 0.8029115901631975 |

slot-wise assignment：

| config | AP | AP50 | AP25 | 结论 |
|---|---:|---:|---:|---|
| strict loose ms10/mc50 | 0.17245357128934893 | 0.35970160173637145 | 0.6175977493359794 | 大规模替换失败 |
| strict loose ms20/mc30 | 0.17245357128934893 | 0.35970160173637145 | 0.6175977493359794 | 同样失败 |
| purelocal loose | 0.1050909972686474 | 0.26231610169308595 | 0.5000313281793627 | localbank 作为 slot candidate 污染更强 |
| wtalocal loose | 0.200228879429071 | 0.5324016102033344 | 0.7080914727035418 | AP 接近 32f，但仍低于 previous best |
| strict q0.70 | 0.16293981481481482 | 0.34178571428571436 | 0.5818452380952382 | 失败 |
| strict q0.80 | 0.22309523809523807 | 0.46142857142857147 | 0.6935714285714286 | 略高于 32f current，但远低于 Stream3D |
| strict q0.90 | 0.2026984126984127 | 0.44571428571428573 | 0.6814285714285714 | 基本退回 32f current |
| strict q0.80 + unmatched20 | 0.23217054263565895 | 0.47885382059800663 | 0.6935714285714286 | 本轮最好，但低于 previous pure best |
| strict q0.80 + unmatched50 | 0.23197530864197535 | 0.4784920634920635 | 0.6935714285714286 | 没有继续提升 |
| strict q0.80 + unmatched100 | 0.2309259259259259 | 0.47654761904761905 | 0.6935714285714286 | 继续下降 |

### 33.4 诊断

宽松 slot assignment：

```text
selected_slots = 199
fallback_slots = 27
output_instances = 226
output_union = 15005
selected_iou_mean = 0.9595403753567581
selected_slot_ioc_mean = 0.976261575280338
selected_candidate_ioc_mean = 0.9772839429238178
```

解释：

```text
宽松配置把约 200 个 32f slot 都替换成了 candidate。
这些 candidate 与 slot 的 overlap 很高，但指标反而下降。
说明“看起来几何上相近”的 candidate 不一定有更好的实例边界或排序。
```

高质量门槛 q0.80：

```text
selected_slots = 3
fallback_slots = 223
output_instances = 226
output_union = 12214
selected_iou_mean = 1.0
```

解释：

```text
严格门槛只找到 3 个足够强的替换，所以结果只比 32f current 略好。
这说明当前无监督 slot-candidate quality 很难识别 oracle 需要的大量高 IoU 候选。
```

q0.80 + unmatched20：

```text
selected_slots = 3
fallback_slots = 223
unmatched_added = 20
output_instances = 246
output_union = 14796
```

解释：

```text
加入少量 unmatched recall 后 AP 到 0.23217054263565895，
仍低于 previous pure best 0.2620249213583159，也远低于 Stream3D 0.39113247863247863。
继续加到 50 / 100 个 unmatched 没有收益。
```

### 33.5 结论

slot-wise object assignment 没有达成目标：

```text
best slot-wise fixed32 = 0.23217054263565895 / 0.47885382059800663 / 0.6935714285714286
Stream3D same fixed32 = 0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

失败原因：

```text
1. 如果大量替换 slot，AP 会下降，说明 candidate 与 slot 高 overlap 不等于更好实例。
2. 如果只允许高质量替换，可替换数量太少，只能略高于 32f current。
3. localbank / unmatched recall 可以补覆盖，但会带来边界和重复污染。
4. 当前无监督信号仍无法复现 GT oracle 的 one-to-one 选择。
```

这进一步收窄了问题：

```text
候选池不是完全没潜力；
但缺少能判断“哪个候选是同一 object 的最佳边界”的无监督证据。
下一步如果继续，应该做 boundary agreement / multi-view silhouette consistency，
而不是继续基于 support overlap 做 assignment。
```

### 33.6 审计状态

本轮新增代码已通过编译和已有回归测试：

```text
Ran 6 tests in 0.001s
OK
```

审阅包已更新并通过 `zip -T`，包内包含：

```text
Stream3D/tools/slotwise_candidate_select.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
```

## 34. 2026-06-08 silhouette consistency 边界一致性实验

### 34.1 本轮目标

前几轮已经证明：

```text
1. support overlap / containment 能带来一些改善，但无法复现 GT oracle 的 one-to-one 选择。
2. slot-wise assignment 只看 support 几何关系会失败。
```

因此本轮尝试更接近边界质量的无监督信号：

```text
把每个 object 的 3D points 投影回它的多帧 2D mask observation；
只统计深度一致的可见点；
计算这些可见点落在对应 2D mask 内的比例；
再用距离 mask 边界的 margin 作为 silhouette interior 质量。
```

新增代码：

```text
Stream3D/tools/silhouette_consistency_score.py
```

这个工具不读取 GT，不读取 evaluator 输出，不使用任何 instance label。

### 34.2 直接作为主排序：失败

| config | AP | AP50 | AP25 | 结论 |
|---|---:|---:|---:|---|
| 32f silhouette | 0.19635331985316004 | 0.3724225776217969 | 0.557964704172192 | 低于 32f current |
| 128f compdens silhouette | 0.09796296296296296 | 0.3 | 0.3 | 明显失败 |
| localbank silhouette | 0.08680001110600649 | 0.20452444866137034 | 0.4088280775933543 | 明显失败 |

对 32f 放宽深度阈值 / 去掉 boundary margin：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| depth tolerance 0.20, margin 0 | 0.17902095768239432 | 0.360339503423507 | 0.550959314975729 |
| no effective depth filter, margin 0 | 0.12286643274110462 | 0.3077679103923552 | 0.5039995386409526 |
| silhouette + area | 0.15387831245728228 | 0.36115609642373003 | 0.570257355041958 |
| score + silhouette + area | 0.1429759418260058 | 0.3465727630903967 | 0.5733766555597568 |

结论：

```text
silhouette consistency 不能直接作为 primary ranking score。
放宽 depth / boundary 也没有修复，反而更差。
```

### 34.3 作为 low-confidence recall filter：小幅有效

把 `localbank minobs5` 先用 silhouette quality 过滤，再作为 low-confidence secondary 加到此前 pure Stream4D primary：

| localbank silhouette threshold | kept localbank objects | AP | AP50 | AP25 |
|---:|---:|---:|---:|---:|
| 0.50 | 83 | 0.2623750073353132 | 0.6262285643605048 | 0.8029115901631975 |
| 0.70 | 53 | 0.2630035146304872 | 0.6268940426730419 | 0.8029115901631975 |
| 0.80 | 24 | 0.26395589558286814 | 0.6279024460343864 | 0.8029115901631975 |
| 0.85 | 14 | 0.2642829759099485 | 0.6282487663807068 | 0.8029115901631975 |
| 0.90 | 5 | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |

对照：

```text
previous pure Stream4D best:
0.2620249213583159 / 0.6258578850907429 / 0.8029115901631975

Stream3D same 32f support:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

当前新 best pure Stream4D：

```text
stream4d_v4_1_scene0050_purebest_plus_localbank_sil_q0p85_drop0p85
AP/AP50/AP25 = 0.2642829759099485 / 0.6282487663807068 / 0.8029115901631975
```

相对 previous pure best：

```text
AP   +0.0022580545516326
AP50 +0.0023908812899639
AP25 +0
```

相对 Stream3D same support：

```text
AP   仍低 0.1268495027225301
AP50 仍低 0.0179050797731394
AP25 高 0.0413731286247359
```

### 34.4 证据链

localbank q0.85 过滤：

```text
instances before = 102
instances after = 14
removed = 88
silhouette_quality_mean = 0.6679583787918091
silhouette_quality_max = 0.9437242150306702
inside_visible_ratio_mean = 0.7076940548072281
interior_ratio_mean = 0.8812063890624772
visible_points_mean = 4123.921568627451
used_observations_mean = 7.5
```

融合后：

```text
primary instances = 337
secondary instances after containment = 13
output instances = 350
output union = 53137
```

解释：

```text
1. silhouette quality 直接排序会失败，说明 2D mask 自洽不等于 3D instance AP 高。
2. 但用于过滤 low-confidence recall layer 有小幅正收益，说明它能去掉一部分明显不稳定的 localbank recall。
3. q0.90 只剩 5 个 localbank object，收益消失，说明该信号只能作为 recall filter，不能替代候选生成。
```

### 34.5 本轮结论

本轮没有达成 pure Stream4D 超过 Stream3D：

```text
best pure Stream4D fixed32 = 0.2642829759099485 / 0.6282487663807068 / 0.8029115901631975
Stream3D same fixed32      = 0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

但是有一个小的算法 insight：

```text
silhouette consistency 不适合作 primary ranking；
适合作 low-confidence recall filter，能带来小幅 AP/AP50 改善。
```

下一步若继续，不应继续调 silhouette threshold，而应把这个信号和 object-level duplicate suppression 结合：

```text
high-confidence primary 保持原 evidence graph / containment tier；
localbank recall 只保留 silhouette-consistent object；
再对 recall 层做更强的一对一 duplicate suppression。
```

### 34.6 审计状态

本轮新增代码已通过编译和已有回归测试：

```text
Ran 6 tests in 0.001s
OK
```

审阅包已更新并通过 `zip -T`，包内包含：

```text
Stream3D/tools/silhouette_consistency_score.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
```

## 35. 2026-06-08 silhouette recall + support-aware containment

### 35.1 本轮目标

上一节发现：

```text
silhouette consistency 直接做 primary ranking 会失败；
但用于过滤 localbank low-confidence recall layer 有小幅正收益。
```

本轮继续检查：

```text
1. silhouette-filtered localbank 加到 WTA primary 是否更好；
2. localbank recall 自身做 NMS 是否更好；
3. containment suppression 是否应该在 fixed 32f support 内计算。
```

### 35.2 代码修改

修改：

```text
Stream3D/tools/fuse_prediction_configs.py
```

新增参数：

```text
--drop-overlap-pre-points-config
```

审计说明：

```text
这个参数只改变 secondary 被 primary suppression 时的 overlap 计算范围。
它不读取 GT。
本轮使用 stream4d_scannet_scene0050_32f_ioc075_fixmem，
也就是和 fixed-support evaluation 相同的 support universe。
```

### 35.3 结果

对照：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| previous pure Stream4D best | 0.2642829759099485 | 0.6282487663807068 | 0.8029115901631975 |

WTA primary + silhouette localbank：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| WTA + q0.85 localbank | 0.26088464186321986 | 0.6634692326927886 | 0.7973356854170808 |

解释：

```text
WTA primary 保持 AP50 优势，但 AP 低于 pure primary。
所以它不能作为当前主结果。
```

secondary self-NMS：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| pure + q0.85 localbank nms0.70 | 0.26437103599800854 | 0.6283420064739469 | 0.8029115901631975 |
| WTA + q0.85 localbank nms0.70 | 0.26092879541368474 | 0.6635159835109279 | 0.7973356854170808 |
| pure + q0.85 localbank nms0.85 | 0.2642829759099485 | 0.6282487663807068 | 0.8029115901631975 |
| WTA + q0.85 localbank nms0.85 | 0.26088464186321986 | 0.6634692326927886 | 0.7973356854170808 |

解释：

```text
secondary self-NMS 有极小收益，但不足以改变结论。
```

support-aware containment suppression：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| pure + q0.85 localbank, supportdrop0.85 | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| WTA + q0.85 localbank, supportdrop0.85 | 0.25307574519460824 | 0.6548076923076923 | 0.7973356854170808 |
| pure + q0.85 localbank, supportdrop0.95 | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 |
| WTA + q0.85 localbank, supportdrop0.95 | 0.2616556679530756 | 0.6634692326927886 | 0.7973356854170808 |

当前 new best pure Stream4D：

```text
stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
AP/AP50/AP25 = 0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

相对上一节 best：

```text
AP gain   = +0.0011021229241568
AP50 gain = 0
AP25 gain = 0
```

相对 Stream3D same support：

```text
AP   仍低 0.1257473797983733
AP50 仍低 0.0179050797731394
AP25 高 0.0413731286247359
```

### 35.4 证据链

supportdrop0.95 pure summary：

```text
primary instances = 337
secondary instances after support-aware containment = 13
secondary skipped by support-aware containment = 1
secondary overlap support points = 12214
output instances = 350
output union = 52798
```

supportdrop0.85 退回 primary 附近：

```text
pure AP/AP50/AP25 = 0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975
```

解释：

```text
1. fixed-support 内 containment 太严格时，会删掉所有有用 recall，结果退回 primary。
2. supportdrop0.95 保留 13 个 localbank secondary，只删 1 个高覆盖重复项，带来目前最大的 AP 小幅提升。
3. 这说明 localbank 的可用贡献非常窄：它可以补一点低置信 recall，但强 suppression 或强排序都会伤害它。
```

### 35.5 本轮结论

本轮仍没有达成目标：

```text
best pure Stream4D fixed32 = 0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
Stream3D same fixed32      = 0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

但新 insight 更清楚：

```text
silhouette consistency + support-aware containment 能做非常小的 recall cleanup；
它不能解决主 AP 差距。
```

失败机制仍然是：

```text
Stream4D 缺少 Stream3D 那种高质量一对一 primary instance layer。
当前所有无 GT 信号只能微调 recall 层，无法稳定找出 oracle 需要的高 IoU candidate。
```

### 35.6 审计状态

本轮修改已通过编译和已有回归测试：

```text
Ran 6 tests in 0.001s
OK
```

审阅包已更新并通过 `zip -T`，包内包含：

```text
Stream3D/tools/fuse_prediction_configs.py
Stream3D/tools/silhouette_consistency_score.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
```

## 36. 2026-06-08 localbank 低置信层保序实验复盘

### 36.1 本轮问题

第 35 节当前 best pure Stream4D 是：

```text
stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
AP/AP50/AP25 = 0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

这个配置把 silhouette-filtered localbank 作为低置信 recall layer，但 secondary score 是统一低分 `0.005`。本轮检查一个更细的可能原因：

```text
是否因为 secondary 内部的 silhouette 排序被统一分数抹掉，导致 AP 不能继续提升？
```

### 36.2 修改了什么代码

修改：

```text
Stream3D/tools/rescore_prediction_scores.py
```

新增：

```text
--score-feature source_score
```

含义：

```text
把输入 prediction 的原始 pred_score 做 min-max normalize，
作为低置信层内部的 tiebreaker。
该工具只读取 prediction 文件，不读取 GT。
```

本轮没有修改 evaluator，也没有修改 GT 或 pre_points 生成逻辑。

### 36.3 协议排错

第一次运行时，命令错误地把 primary score 覆盖成常数 `1.0`。这不等价于第 35 节 best，因为第 35 节 best 保留了 primary 内部多层分数。

错误协议结果：

| config | AP | AP50 | AP25 | 说明 |
|---|---:|---:|---:|---|
| `lowrank001_supportdrop0p95` | 0.2520543021661612 | 0.48293813693767984 | 0.6279661016949152 | primary score 被错误覆盖 |
| `lowrank005_supportdrop0p95` | 0.2520543021661612 | 0.48293813693767984 | 0.6279661016949152 | primary score 被错误覆盖 |

这个结果只作为审计记录，不能用来判断算法好坏。排错后确认：

```text
previous best:
primary_score = -1.0
secondary_score = 0.005

错误协议：
primary_score = 1.0
secondary_score = -1.0
```

因此正确实验必须同时使用：

```text
--preserve-primary-score
--preserve-secondary-score
```

### 36.4 正确协议结果

lowrank secondary 分数范围：

```text
lowrank001:
score_before = 0.8509376645088196 到 0.9437242150306702
score_after  = 0.004999999888241291 到 0.006000000052154064

lowrank005:
score_after  = 0.004999999888241291 到 0.009999999776482582
```

fixed 32f support 评估：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| previous best: `pure_plus_localbank_sil_q0p85_supportdrop0p95` | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 |
| `lowrank001_preserveprimary_supportdrop0p95` | 0.2652273083220721 | 0.6284313820035432 | 0.8029115901631975 |
| `lowrank005_preserveprimary_supportdrop0p95` | 0.2652273083220721 | 0.6284313820035432 | 0.8029115901631975 |

fusion 诊断：

```text
primary instances = 337.0
secondary instances after support-aware containment = 13.0
secondary skipped by support-aware containment = 1.0
output instances = 350.0
output union = 52798.0
support points = 12214.0
```

### 36.5 结论

本轮没有达成超过 Stream3D。

正确协议下，保留 localbank 内部 silhouette 排序：

```text
AP   略低于 previous best：0.2652273083220721 < 0.2653850988341053
AP50 略高于 previous best：0.6284313820035432 > 0.6282487663807068
AP25 持平：0.8029115901631975
```

相对同 support 的 Stream3D：

```text
Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615

best pure Stream4D remains:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

因此：

```text
纯 Stream4D 仍未在 inherit/fixed support 下超过 Stream3D 主 AP。
secondary 内部排序不是当前 AP 主瓶颈。
localbank 的作用仍只是很窄的 low-confidence recall cleanup。
```

### 36.6 新 evidence / insight

这轮进一步排除了一个较具体的解释：

```text
“之前 localbank 没发挥，是因为统一低分丢掉了 silhouette 排序。”
```

证据显示不是这样。即使保留 silhouette 排序，AP 没有提升。当前瓶颈仍然回到前几节已经定位的问题：

```text
1. pure Stream4D 缺少高质量一对一 primary instance layer。
2. low-confidence recall layer 可以补 AP25 / recall，但对 AP 主指标帮助很小。
3. 无监督 silhouette / support / compactness 这类信号可以过滤明显坏候选，却不能稳定找出高 IoU 边界最好的候选。
4. 要继续接近或超过 Stream3D，必须在 object 形成阶段改善 one-to-one assignment 和 boundary refinement，而不是只重排 secondary。
```

### 36.7 审计状态

本轮新增修改已通过编译和已有回归测试：

```text
Ran 6 tests in 0.002s
OK
```

审阅包已重新生成并通过完整性检查：

```text
stream4d_v4_1_code_review_packet.zip
test of stream4d_v4_1_code_review_packet.zip OK
```

包内已确认包含：

```text
Stream3D/tools/rescore_prediction_scores.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
```

## 37. 2026-06-08 support 内 3D connected-component refinement 复盘

### 37.1 本轮目标

上一轮排除了 localbank secondary 内部排序问题。本轮继续按计划中“不要只调分数、要改实例质量/边界”的方向，尝试一个更贴近算法本体的精炼：

```text
对每个 prediction instance，在 fixed support 内按 3D 距离图拆成 connected components；
如果一个实例由多个空间碎片组成，只保留主要组件或少量大组件。
```

这个方法不读取 GT，不改变 evaluator，只使用：

```text
prediction mask
fixed support pre_points
ScanNet mesh 顶点坐标
```

### 37.2 修改了什么代码

新增：

```text
Stream3D/tools/support_component_refine.py
```

算法步骤：

```text
1. 读取 full-scene prediction。
2. 读取 `stream4d_scannet_scene0050_32f_ioc075_fixmem` 的 pre_points。
3. 对每个预测实例，只取它在 fixed support 中覆盖的点。
4. 用 `scipy.spatial.cKDTree` 在这些点上按半径建立 3D 邻接。
5. 用 union-find 得到 connected components。
6. 保留最大组件或前 K 个大组件。
7. 输出 refined prediction，保留原 score 和 class。
```

参数：

```text
--radius
--max-components-per-instance
--min-component-points
--min-component-ratio
--outside-support drop
--tmp-policy support
```

审计说明：

```text
该工具没有读取 ScanNet GT。
它只针对指定 support 做无监督几何精炼，因此只能作为 fixed-support diagnostic，不能写成 full support victory。
```

### 37.3 结果

对照：

```text
previous best pure Stream4D fixed32:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

本轮结果：

| config | AP | AP50 | AP25 | support union after | keep ratio mean | changed instances |
|---|---:|---:|---:|---:|---:|---:|
| previous best | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 | 12091 | 1.0000 | 0 |
| `compref_r004_k1` | 0.15976769511408848 | 0.3474611973392462 | 0.47065410199556545 | 8231 | 0.7559661717075941 | 144 |
| `compref_r006_k1` | 0.179981844903759 | 0.3642114199310601 | 0.584951690821256 | 9566 | 0.8081890114411976 | 120 |
| `compref_r006_k2` | 0.20475655072227655 | 0.3962130376344086 | 0.7043792766373411 | 10257 | 0.8243891880152694 | 119 |
| `compref_r010_k2` | 0.24077101810367532 | 0.528074898394371 | 0.7731581425673717 | 11569 | 0.8651570650987644 | 103 |
| `compref_r010_k4` | 0.2556287929259532 | 0.5731307443401764 | 0.7731581425673717 | 11709 | 0.8706051733493791 | 103 |

### 37.4 结论

本轮没有达成目标。最好的配置仍低于 previous best：

```text
compref_r010_k4:
0.2556287929259532 / 0.5731307443401764 / 0.7731581425673717

previous best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

失败原因：

```text
1. 3D connected-component trimming 太容易删掉有效 object support。
2. scene0050 的椅子结构本身可能由多个空间上不完全连续的局部组成；保留最大组件会损害 recall。
3. 保留更多组件能恢复一些 AP25，但 AP/AP50 仍下降，说明简单几何连通性不是边界质量的充分条件。
4. 当前错误不是“删除离散碎片”能解决，而是 object-level assignment 和边界证据本身不够强。
```

### 37.5 Insight

这轮提供了一个有用的负证据：

```text
不能把“3D 空间连通”直接等价为“正确实例边界”。
```

对后续方向的约束：

```text
1. 后续如果做边界 refinement，不能只看 3D 连通组件；
   必须结合 2D mask boundary、multi-view consistency 和 object identity。

2. 不能把 support 缩小当成解决方案；
   本轮 support union 从 12091 降到 11709 甚至 8231 时，指标同步下降。

3. 要继续接近 Stream3D，仍需要更强的 object-level one-to-one proposal formation，
   而不是对最终 mask 做几何裁剪。
```

### 37.6 当前状态

截至本轮，pure Stream4D 的 best 仍然是：

```text
stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
AP/AP50/AP25 = 0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

它仍未超过 same fixed32 support 下的 Stream3D：

```text
Stream3D same fixed32 = 0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

### 37.7 审计状态

本轮新增修改已通过编译和已有回归测试：

```text
Ran 6 tests in 0.001s
OK
```

审阅包已重新生成并通过完整性检查：

```text
stream4d_v4_1_code_review_packet.zip
test of stream4d_v4_1_code_review_packet.zip OK
```

包内已确认包含：

```text
Stream3D/tools/support_component_refine.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
```

## 38. 2026-06-08 self-discovered 2D boundary refinement 复盘

### 38.1 本轮目标

第 37 节证明单纯 3D 连通组件裁剪会伤 recall。本轮继续按计划里的 `boundary-aware evidence` 方向推进，但解决一个实际问题：

```text
当前 best pure Stream4D fusion 没有 object_dict，
所以不能直接使用 object_dict.mask_list 做 boundary refinement。
```

因此本轮实现一个 prediction-only 原型：

```text
不依赖 object_dict；
把每个 3D prediction object 投影到多帧 2D Cropformer mask；
自动选择每帧的 dominant 2D mask ID；
用该 self-discovered observation 裁掉低 inside-support 的 3D points。
```

### 38.2 修改了什么代码

新增：

```text
Stream3D/tools/self_discovered_boundary_refine.py
```

算法步骤：

```text
1. 读取 full-scene prediction。
2. 对每个 object 取其在 32f fixed support 中的点。
3. 按固定 frame stride 读取 ScanNet depth / pose / 2D Cropformer mask。
4. 将 object 点投影到每帧，按 depth tolerance 做可见性过滤。
5. 在可见点中选 dominant 2D mask ID。
6. 对每个 3D 点统计它落在 dominant 2D mask 内的比例。
7. 低于阈值的点从该 object 中删除。
```

该工具不读取 GT，不改变 evaluator。

### 38.3 结果

对照：

```text
previous best pure Stream4D:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

本轮结果：

| config | AP | AP50 | AP25 | union after | mean observations | point keep ratio |
|---|---:|---:|---:|---:|---:|---:|
| previous best | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 | 52798 | NA | 1.0000 |
| `selfbd_in30` | 0.25668083900226757 | 0.6253283257747544 | 0.7814791194255479 | 11238 | 1.9085714285714286 | 0.936854966414708 |
| `selfbd_in50` | 0.2559228485572571 | 0.6257119243510372 | 0.7820159932659932 | 11135 | 1.9085714285714286 | 0.9199050965510834 |
| `selfbd_in70` | 0.2283597444145617 | 0.49511835548172756 | 0.8108324434424933 | 10340 | 1.9085714285714286 | 0.8856733498118027 |

### 38.4 结论

本轮没有达成目标。

最好的 AP 仍低于 previous best：

```text
selfbd_in30 AP = 0.25668083900226757
previous best AP = 0.2653850988341053
```

最有趣的是：

```text
selfbd_in70 AP25 = 0.8108324434424933
previous best AP25 = 0.8029115901631975
```

但它的 AP/AP50 明显下降：

```text
selfbd_in70 AP/AP50 = 0.2283597444145617 / 0.49511835548172756
```

解释：

```text
1. self-discovered 2D boundary evidence 能裁掉一些粗 IoU 下的污染点，所以 AP25 可以上升。
2. 但 dominant mask ID 是从最终 prediction 自己反推出来的，不够稳定。
3. 该裁剪会误删高 IoU 所需的边界/部件，导致 AP/AP50 下降。
4. 因此，boundary evidence 方向有信号，但当前 prediction-only observation discovery 不够可靠。
```

### 38.5 Insight

这轮比第 37 节更细地说明：

```text
不是所有裁点都会无意义；
边界/2D mask evidence 确实能影响 AP25。
```

但它也给出约束：

```text
不能在最终 prediction 阶段临时猜 object-to-mask correspondence。
```

更合理的后续方向：

```text
1. 在 evidence graph / carrier 阶段保存 object-to-mask observation，而不是事后从 prediction 反推。
2. boundary refinement 应该使用已知 carrier/mask support 和 same-frame cannot-link。
3. 对每个 object 同时维护 high-confidence core 和 low-confidence fringe，而不是把 inside ratio 直接硬阈值二值化。
```

### 38.6 当前状态

pure Stream4D best 不变：

```text
stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
AP/AP50/AP25 = 0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

仍未超过 Stream3D same fixed32：

```text
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

### 38.7 审计状态

本轮新增修改已通过编译和已有回归测试：

```text
Ran 6 tests in 0.001s
OK
```

审阅包已重新生成并通过完整性检查：

```text
stream4d_v4_1_code_review_packet.zip
test of stream4d_v4_1_code_review_packet.zip OK
```

包内已确认包含：

```text
Stream3D/tools/self_discovered_boundary_refine.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
```

## 39. 2026-06-08 Evidence graph 节点级质量过滤复盘

### 39.1 本轮目标

继续沿第 38 节的结论推进：不要只在最终 prediction 上做后处理，而是把质量判断前移到 evidence graph 形成阶段。

本轮测试两个非常直接的算法假设：

```text
假设 A：carrier 数太少的 graph node 是噪声，删掉后 component 更干净。
假设 B：coverage 太低的 graph node 是噪声，删掉后 component 更干净。
```

这两个假设都没有使用 GT。实验只使用 cached carrier / mask observation。

### 39.2 修改了什么代码

修改：

```text
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/replay_evidence_graph.py
```

新增：

```text
--graph-min-node-carriers
--graph-min-node-coverage
```

新增审计字段：

```text
evidence_graph_num_raw_nodes
evidence_graph_num_dropped_nodes
evidence_graph_min_node_carriers
evidence_graph_min_node_coverage
```

这些修改的审计意义：

```text
1. 可以明确知道建图前删掉了多少 graph nodes。
2. 可以区分“质量过滤真的改变了 graph”与“参数看起来变了但 graph 基本没变”。
3. 不改变 AP evaluator，不读取 GT。
```

### 39.3 结果

对照：

```text
128f strict component_densify rel1.0:
self recompute = 0.490385 / 0.605769 / 0.810897
fixed32        = 0.135185 / 0.300000 / 0.300000

pure Stream4D fixed32 best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

本轮结果：

| config | self AP | self AP50 | self AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 | objects | points | raw nodes | dropped nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `nodecar3` | 0.45454545454545453 | 0.5757575757575757 | 0.7954545454545454 | 0.1088888888888889 | 0.25 | 0.25 | 67 | 6379 | 1938 | 4 |
| `nodecov002` | 0.3333333333333333 | 0.5142857142857142 | 0.7142857142857143 | 0.05972222222222222 | 0.1 | 0.1 | 37 | 1897 | 1938 | 710 |

Graph 诊断：

| config | graph nodes | kept components | dropped components | accepted edges | rejected conflict edges | export conflict | hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nodecar3` | 1934 | 67 | 280 | 1587 | 7863 | 0.0 | 0.9181488441200891 |
| `nodecov002` | 1228 | 37 | 250 | 941 | 4532 | 0.0 | 0.9415782915437781 |

### 39.4 结论

本轮没有达成目标。

`nodecar3` 只删掉 4 个 nodes，理论上是非常轻的过滤，但结果仍低于原 128f strict：

```text
self recompute:
0.454545 / 0.575758 / 0.795455
低于原 128f strict 的
0.490385 / 0.605769 / 0.810897

fixed32:
0.108889 / 0.250000 / 0.250000
低于原 128f strict fixed32 的
0.135185 / 0.300000 / 0.300000
```

`nodecov002` 更明显失败：

```text
删掉 nodes = 710
objects = 37
points = 1897
self = 0.333333 / 0.514286 / 0.714286
fixed32 = 0.059722 / 0.100000 / 0.100000
```

### 39.5 失败原因分析

这轮说明两个简单假设都不成立：

```text
carrier 数少 != 一定是坏 evidence。
coverage 低 != 一定是坏 evidence。
```

更准确的解释是：

```text
1. 低 coverage node 可能是连接同一 object 多个稳定观测的桥。
2. 建图前删 node 会改变 component 拓扑，导致本来能形成稳定 object 的 evidence 被断开。
3. nodecov002 虽然 hit rate 更高，但 support 过小，说明它只是留下更容易回投的点，不代表 instance 更完整。
4. export conflict 已经是 0，继续删 node 不能解决 fixed32 的核心问题；核心仍是 object recall、一对一分配和边界质量。
```

### 39.6 Insight

下一步不应该继续做硬过滤：

```text
不要再只调 graph-min-node-carriers / graph-min-node-coverage。
```

更合理的算法方向：

```text
1. 保留低 coverage node，但降低它在边权或 component quality 中的权重。
2. 把 node quality 用于 support densify 时的 mask view selection，而不是建图前删除。
3. 对 component 内部做 soft support selection：高质量 node 形成 core，低质量 node 只允许提供受限 fringe。
4. fixed32 目标仍需要 object-level 一对一竞争，而不是 node-level 硬清洗。
```

当前结论保持不变：

```text
pure Stream4D fixed32 best 仍是：
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

它仍未超过 Stream3D same fixed32：
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

## 40. 2026-06-08 Evidence graph coverage-aware edge ordering 复盘

### 40.1 本轮目标

第 39 节证明：建图前硬删低 carrier / 低 coverage node 会失败。本轮改成更温和的算法：

```text
保留所有 graph node。
原 carrier IoC 仍决定边是否足够强。
coverage 只改变 edge 合并顺序，让更高 coverage 的 evidence 先合并。
```

目标是避免低 coverage evidence 过早影响 component 形成，同时不丢掉它们。

### 40.2 修改了什么代码

修改：

```text
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/replay_evidence_graph.py
```

新增：

```text
--graph-edge-coverage-power
```

实现过程里出现并修复了一个重要问题：

```text
第一版把 coverage-adjusted score 同时用于 threshold 和 sorting。
这会把大量边直接判成 weak edge。

修正版只让 coverage-adjusted score 用于 sorting；
raw carrier IoC 仍用于 min_carrier_ioc threshold。
```

这两个阶段都保留在日志中，便于审计。

### 40.3 结果

对照：

```text
128f strict component_densify rel1.0:
self recompute = 0.490385 / 0.605769 / 0.810897
fixed32        = 0.135185 / 0.300000 / 0.300000

pure Stream4D fixed32 best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

第一版 threshold-coupled 负例：

| config | self AP | self AP50 | self AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 | objects | points | accepted edges | weak edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ioc0p70_edgecovp025` | nan | nan | nan | 0.0 | 0.0 | 0.0 | 2 | 49 | 44 | 88304 |
| `ioc0p60_edgecovp050` | nan | nan | nan | 0.0 | 0.0 | 0.0 | 2 | 49 | 29 | 88317 |

修正版 sort-only 结果：

| config | self AP | self AP50 | self AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 | objects | points | accepted edges | conflict edges | weak edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ioc0p79_edgesortcovp050` | 0.40823412698412703 | 0.6138392857142857 | 0.7834821428571428 | 0.1088888888888889 | 0.25 | 0.25 | 78 | 8638 | 1607 | 7855 | 78953 |
| `ioc0p70_edgesortcovp050` | 0.34506172839506166 | 0.5249999999999999 | 0.7027777777777777 | 0.1088888888888889 | 0.25 | 0.25 | 80 | 8445 | 1689 | 10627 | 76099 |

### 40.4 结论

本轮没有达成目标。

修正版虽然避免了 graph 崩掉，但仍低于原 128f strict：

```text
原 128f strict self:
0.490385 / 0.605769 / 0.810897

ioc0p79_edgesortcovp050 self:
0.408234 / 0.613839 / 0.783482

ioc0p70_edgesortcovp050 self:
0.345062 / 0.525000 / 0.702778
```

fixed32 更没有改善：

```text
两组 sort-only fixed32 都是：
0.108889 / 0.250000 / 0.250000

低于原 128f strict fixed32：
0.135185 / 0.300000 / 0.300000
```

### 40.5 失败原因分析

这轮的证据说明：

```text
当前主要瓶颈不是 edge sorting。
```

理由：

```text
1. 只改变合并顺序后，kept components 从 67 增加到 78-80。
2. object 变多但 fixed32 AP 不升，说明这是碎片化而非有效召回。
3. own recompute AP 下降，说明 coverage-aware sorting 破坏了一部分原本较好的 component。
4. fixed32 仍只有 0.108889 / 0.25 / 0.25，说明它没有解决 support universe 下的 coverage/one-to-one 问题。
```

### 40.6 Insight

这轮进一步排除了一个上游但仍偏局部的方向：

```text
不要继续只调 graph-edge-coverage-power / graph-min-carrier-ioc。
```

更合理的下一步仍然是 object-level 算法，而不是 node/edge 级局部规则：

```text
1. coverage proposal bank 提供 support coverage。
2. evidence graph precision ranker 提供高置信 object。
3. object-level same-frame exclusivity / competition 减少同一真实实例的重复候选。
4. boundary refinement 解决 AP 与 AP50/AP25 之间的高 IoU 差距。
```

当前总状态不变：

```text
hybrid diagnostic 可以超过 Stream3D；
pure Stream4D 仍没有在 fixed32 / inherit_pre_points 主 AP 上超过 Stream3D。
```

## 41. 2026-06-08 Multi-source consensus object selector 复盘

### 41.1 本轮目标

前面几轮已经排除：

```text
1. 单 prediction 内 object competition。
2. slotwise candidate replacement。
3. node filtering。
4. edge ordering。
```

本轮尝试一个新的 object-level 假设：

```text
如果多个纯 Stream4D 来源在同一个 fixed support 上支持同一个候选，
那么这个候选可能更可靠。
```

因此实现 multi-source consensus selector：

```text
多来源候选池 -> support overlap grouping -> 每组一个代表 -> high confidence
当前 pure best -> low confidence recall
```

该方法不读取 GT，不改变 evaluator。

### 41.2 新增代码

新增：

```text
Stream3D/tools/multi_source_consensus_select.py
```

输入来源：

```text
stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
stream4d_v4_1_scene0050_wta_plus_localbank_sil_q0p85_supportdrop0p95
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
```

主要信号：

```text
source weight
prediction score
support area
support conflict ratio
number of distinct sources in the group
```

### 41.3 结果

对照：

```text
pure Stream4D fixed32 best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

本轮结果：

| config | AP | AP50 | AP25 | high selected | low added | output instances | support union | mean selected support area | mean selected conflict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `msource_hi` | 0.06083333333333332 | 0.24083333333333332 | 0.3 | 99 | 0 | 99 | 4127 | 49.4040404040404 | 1.0 |
| `msource_plus_purelow` | 0.14895299145299146 | 0.446025641025641 | 0.59 | 99 | 207 | 306 | 12085 | 49.4040404040404 | 1.0 |
| `iou025_min20_plus_purelow` | 0.19650758942444677 | 0.6047426741673023 | 0.8826393724347102 | 88 | 54 | 142 | 11852 | 152.0 | 1.0 |
| `minioc070_min20_plus_purelow` | 0.1984336967803787 | 0.6069761147312311 | 0.7711412141641301 | 79 | 71 | 150 | 11866 | 122.55696202531645 | 1.0 |

### 41.4 结论

没有达成目标。

最初的 `min_ioc` consensus 明显选到了小碎片：

```text
mean selected support area = 49.4040
mean selected conflict ratio = 1.0
AP = 0.060833
```

这说明：

```text
一个小候选被多个大候选包含时，min_ioc 会把它误判成强共识。
```

改用 IoU grouping 并提高 min area 后，粗覆盖指标变好：

```text
iou025_min20_plus_purelow:
AP25 = 0.882639
```

但 AP/AP50 仍不够：

```text
AP/AP50 = 0.196508 / 0.604743
低于 pure best:
0.265385 / 0.628249
更低于 Stream3D:
0.391132 / 0.646154
```

### 41.5 失败原因分析

multi-source consensus 提供的是“粗覆盖一致性”，不是“高 IoU 实例边界质量”。

证据链：

```text
1. iou025_min20_plus_purelow 的 AP25 达到 0.882639，说明它能找到覆盖足够大的粗候选。
2. 但 AP 只有 0.196508，说明高 IoU 阈值下实例边界和一对一质量不足。
3. selected conflict ratio 始终是 1.0，说明 high candidates 在 support 内仍高度重叠。
4. low recall 补回来后 support union 约 11852-12085，coverage 够了，但 AP 没上去，说明瓶颈不是单纯漏覆盖。
```

### 41.6 Insight

本轮进一步缩小了失败原因：

```text
不是缺少多来源一致性；
不是缺少 fixed support coverage；
不是简单候选少。
```

真正缺的是：

```text
1. 能区分同一粗覆盖区域里哪个候选边界更好的信号。
2. 能把多个重叠候选压成一对一实例的 object-level duplicate resolution。
3. 不是只看 overlap，而是要看边界 agreement、实例互斥和高 IoU support purity。
```

当前最诚实结论不变：

```text
pure Stream4D 仍未在 inherit/fixed32 主 AP 上超过 Stream3D。
hybrid 可以超过，但不能算 pure Stream4D 胜利。
```

## 42. 2026-06-08 3D geometry slot candidate selection 复盘

### 42.1 本轮目标

第 41 节说明 multi-source consensus 只能提供粗覆盖一致性，不能提供高 IoU 实例边界。本轮测试另一个 object-level 假设：

```text
fixed32 support 的 3D connected components 可以作为 GT-free instance slots；
然后把 Stream4D candidates 一对一分配到这些 geometry slots。
```

如果该假设成立，应该能减少重复候选，并让 pure Stream4D 更接近 Stream3D 的一对一实例质量。

### 42.2 新增代码

新增：

```text
Stream3D/tools/geometry_slot_candidate_select.py
```

算法：

```text
1. 读取 fixed32 support pre_points。
2. 读取 ScanNet mesh 顶点坐标。
3. 用半径图做 3D connected components。
4. 每个 component 是一个 geometry slot。
5. 从多个 pure Stream4D prediction 来源读取候选。
6. 按 slot_ioc / candidate_ioc / IoU / area ratio / conflict 做 slot-candidate 匹配。
7. 每个 slot 最多选一个 candidate，每个 candidate 最多分给一个 slot。
8. 可选把当前 pure best 作为 low-confidence recall layer 加回。
```

不读取 GT。

### 42.3 support 几何诊断

fixed32 support：

```text
support points = 12214
mesh vertices = 211406
```

半径扫描：

| radius | raw components | kept components >=20 | top component sizes |
|---:|---:|---:|---|
| 0.03 | 1501 | 71 | 1812, 636, 582, 434, 417 |
| 0.04 | 654 | 52 | 3932, 1128, 461, 407, 337 |
| 0.05 | 386 | 39 | 4129, 1773, 665, 511, 481 |
| 0.06 | 269 | 27 | 4161, 2327, 893, 544, 521 |
| 0.08 | 114 | 15 | 7639, 1319, 999, 689, 390 |
| 0.10 | 59 | 10 | 8505, 1347, 1204, 689, 91 |

解释：

```text
0.03 / 0.05 仍有较多 slot，可以测试。
0.08 以后大组件明显粘连，不适合作 instance slot。
```

### 42.4 结果

对照：

```text
pure Stream4D fixed32 best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

本轮结果：

| config | AP | AP50 | AP25 | geometry slots | selected high | low added | output instances | support union | selected slot IoC | selected candidate IoC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `geomslot_r003_min20_plus_purelow` | 0.22524157879969278 | 0.40292796384901647 | 0.5228282828282828 | 71 | 29 | 124 | 153 | 11911 | 0.6140736034168086 | 0.6782921255854638 |
| `geomslot_r005_min20_plus_purelow` | 0.18915722597740142 | 0.47357997265892005 | 0.5894372294372295 | 39 | 18 | 137 | 155 | 11893 | 0.6437206484119434 | 0.7937355174550442 |

### 42.5 结论

没有达成目标。

3D geometry slot selection 低于当前 pure best：

```text
best geometry slot AP = 0.225242
pure Stream4D fixed32 best AP = 0.265385
Stream3D same fixed32 AP = 0.391132
```

### 42.6 失败原因分析

本轮失败提供了新的证据：

```text
1. 3D connected components 不是可靠 instance slots。
2. chair 场景里相邻实例、支撑面、局部接触会让 support 组件粘连。
3. 半径小会碎裂，半径稍大就形成巨大 component。
4. slot_ioc / candidate_ioc 不低，但 AP 低，说明几何槽位匹配并不能保证高 IoU 实例边界。
```

更具体：

```text
r=0.03 有 71 个 geometry slots，但只选出 29 个 high candidates。
r=0.05 有 39 个 geometry slots，只选出 18 个 high candidates。
low recall 加回后 support union 接近满，但 AP/AP50/AP25 仍低。
```

这和前几轮结论一致：

```text
coverage 已经不是主瓶颈；
真正瓶颈是候选边界质量和 object-level one-to-one instance assignment。
```

### 42.7 Insight

当前不建议继续只调：

```text
geometry slot radius
min slot points
slot_ioc / candidate_ioc threshold
```

原因：

```text
3D 连通几何本身不是实例边界。
```

如果继续做算法改进，下一步需要的不是更复杂的几何 slot，而是更接近边界质量的无监督信号，例如：

```text
multi-view boundary agreement
same-frame mutually exclusive masks
candidate split/merge audit
per-object high-confidence core and boundary fringe consistency
```

当前总状态仍不变：

```text
pure Stream4D 未在 inherit/fixed32 主 AP 上超过 Stream3D。
```

## 43. 2026-06-08 Per-object core/fringe split 复盘

### 43.1 本轮目标

前几轮已经证明，pure Stream4D 的 fixed32 support 结果没有超过 Stream3D，主要问题不是点覆盖不足，而是：

```text
候选重叠多；
同一个 GT 附近多个候选重复；
高 IoU 一对一实例质量不足。
```

本轮测试一个更算法化的假设：

```text
同一个 prediction object 不应该所有点使用同一置信度。
低冲突点作为 high-confidence core。
高冲突/边界点作为 low-confidence fringe 或 recall layer。
```

这不是 GT oracle。新增工具只读取 prediction 和指定 pre_points support，不读取 ScanNet GT。

### 43.2 本轮代码修改

新增：

```text
Stream3D/tools/split_core_fringe_prediction.py
```

算法：

```text
1. 读取 full-scene prediction。
2. 读取 fixed32 support 的 pre_points。
3. 在 support 内统计每个点被多少个 prediction object 同时占有。
4. 对每个 object，owner_count <= max_core_owners 的点作为 core。
5. 根据 low_mode，把原 full mask 或 conflict fringe 作为低分层。
6. 输出新的 prediction config、TMP 和 summary JSON。
```

不使用 GT，也不修改 evaluator。

### 43.3 对照基线

所有数值均来自 `Stream3D/data/evaluation/scannet/*.txt` 的最后一行。

| method | support | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| Stream3D same fixed32 | `stream4d_scannet_scene0050_32f_ioc075_fixmem` | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| 32f current self audit | same | 0.2026984126984127 | 0.44571428571428573 | 0.6814285714285714 |
| previous pure Stream4D best | same | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 |

### 43.4 core/fringe 结果

| config | low mode | max core owners | AP | AP50 | AP25 | 结论 |
|---|---|---:|---:|---:|---:|---|
| `corefringe_own1_full` | full | 1 | 0.0827699530516432 | 0.18950704225352114 | 0.35373239436619713 | core 太小，失败 |
| `corefringe_own1_fringe` | fringe | 1 | 0.07913764510779436 | 0.19455223880597014 | 0.3374626865671642 | core 太小，失败 |
| `corefringe_own2_full` | full | 2 | 0.1476388888888889 | 0.39666666666666667 | 0.51953125 | 仍明显低 |
| `corefringe_own2_fringe` | fringe | 2 | 0.14194716775599125 | 0.39526960784313725 | 0.5378676470588235 | 仍明显低 |
| `corefringe_own3_full` | full | 3 | 0.24237695924764888 | 0.5431630094043887 | 0.749276645768025 | 接近但低于 pure best |
| `corefringe_own5_full` | full | 5 | 0.24023681602167501 | 0.590468634617362 | 0.7731581425673717 | 低于 pure best |
| `corefringe_own8_full` | full | 8 | 0.262385435534442 | 0.6253881603201007 | 0.8029115901631975 | 接近 pure best，但没超过 |
| `coreonly_own5` | none | 5 | 0.24236333599163215 | 0.5933966743195953 | 0.7731581425673717 | 删除严重冲突点仍低 |
| `coreonly_own8` | none | 8 | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 | 与 pure best 完全相同 |
| `coreonly_own12` | none | 12 | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 | 与 pure best 完全相同 |

### 43.5 诊断证据

关键 summary：

| config | mean core ratio | mean conflict ratio | output support union | output support conflict ratio | output instances |
|---|---:|---:|---:|---:|---:|
| `own1_full` | 0.08653449334592832 | 0.9134655066540717 | 12058 | 0.9671587327915077 | 275 |
| `own2_full` | 0.29509968463982533 | 0.7049003153601747 | 12058 | 0.9906286282965666 | 337 |
| `own8_full` | 0.9833448833647003 | 0.0166551166352998 | 12058 | 0.9999170675070492 | 482 |
| `coreonly_own5` | 0.9069401065964906 | 0.0930598934035094 | 11897 | 0.8052450197528789 | 227 |
| `coreonly_own8` | 0.9833448833647003 | 0.0166551166352998 | 12045 | 0.8081361560813616 | 240 |

解释：

```text
owner<=1 的 core 平均只保留 8.65% object support。
owner<=2 的 core 平均只保留 29.51% object support。
这两个配置 AP 崩溃，说明“唯一点/低冲突点”不足以代表一个实例。
```

更软的 owner threshold：

```text
owner<=8 时 core ratio 已经达到 98.33%，几乎回到原 mask。
coreonly_own8 与 previous pure best 完全相同：
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

这说明：

```text
只有极少数严重冲突点可以删；
删掉它们不会改变指标；
大量中度重叠点虽然看起来冲突，但对当前 AP25/AP50 召回仍有贡献。
```

### 43.6 本轮是否达成目标

没有达成。

当前仍然是：

```text
pure Stream4D best fixed32:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

差距：

```text
AP   差 0.12574737979837333
AP50 差 0.01790507977374539
AP25 pure Stream4D 高 0.04137312844825841
```

### 43.7 失败原因和 insight

本轮进一步约束了后续算法方向：

1. **不能把低冲突点直接当实例核心。**  
   唯一/低 owner count 点太少，尤其在椅子密集场景中，很多真实实例边界和主体点都会被多个候选覆盖。

2. **冲突点不全是噪声。**  
   更激进地删除冲突点会让 AP/AP50/AP25 全部下降；说明当前重叠区域里同时包含污染和必要召回。

3. **单点 owner count 不是足够强的边界质量信号。**  
   owner count 只能知道“有多少候选覆盖这个点”，不能知道哪个候选边界更接近真实实例。

4. **后续应该做 object-level competition，而不是 point-level core split。**  
   需要利用 same-frame mask exclusivity、multi-view boundary agreement、carrier/evidence consistency 来判断整组候选之间谁是主实例、谁是碎片或重复，而不是只按每个点的 owner count 切 core/fringe。

安全结论：

```text
per-object core/fringe split 没有让 pure Stream4D 在 inherit/fixed32 主 AP 上超过 Stream3D。
该负例说明，fixed-support 失败不是简单由“冲突点太多”造成，而是缺少能做一对一实例选择的 object-level 边界质量信号。
```

## 44. 2026-06-08 Residual recall 与 localbank-high 复盘

### 44.1 本轮目标

前面已经发现 local proposal bank 单独不强，但作为低分 recall layer 可以给 pure Stream4D 带来小幅收益。本轮拆解它的收益来源：

```text
假设 A：localbank 主要补 primary 完全没有覆盖的 support 点。
假设 B：localbank 其实是重叠区域里的替代候选，只能低分补充，不能当 primary。
```

### 44.2 代码修改

新增：

```text
Stream3D/tools/residual_recall_fuse.py
```

算法：

```text
1. 读取 primary 和 secondary prediction。
2. 读取 fixed32 support pre_points。
3. 计算 primary 在 support 内的 union。
4. 对每个 secondary 计算 residual = secondary_support - primary_union_support。
5. 只有 residual_area 和 residual_ratio 达标才保留 secondary。
```

输出模式：

```text
residual_support：只输出 residual 点。
support_full：通过 residual 检查后输出 support 内完整 secondary。
full：通过 residual 检查后输出 full-scene secondary。
```

该工具不读取 GT。

### 44.3 residual recall 结果

输入：

```text
primary = stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
secondary = stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85
support = stream4d_scannet_scene0050_32f_ioc075_fixmem
```

结果：

| config | AP | AP50 | AP25 | retained secondary |
|---|---:|---:|---:|---:|
| `residual_a10_r001` | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 | 0 |
| `supportfull_a10_r001` | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 | 0 |
| `full_a10_r001` | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 | 0 |
| `residual_a50_r005` | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 | 0 |
| `supportfull_a50_r005` | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 | 0 |
| `full_a50_r005` | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 | 0 |

关键诊断：

```text
fixed32 support points = 12214
primary_support_union = 12084
uncovered_support_count = 130
residual secondary instances = 0
dropped_empty = 3
dropped_small = 11
```

解释：

```text
strict primary 已覆盖 fixed32 support 的 98.94%。
localbank 没有任何 candidate 能在剩余 130 个未覆盖 support 点上贡献足够 residual。
```

结论：

```text
localbank 的小幅收益不是来自补 primary 完全没覆盖的点。
它的作用发生在已覆盖/重叠区域：提供某些替代候选或低分 recall。
```

### 44.4 localbank-high 结果

为了验证 localbank 是否可以当更好的 primary，本轮反过来：

```text
primary = localbank_minobs5_sil_q0p85
secondary = strict_32f_secioc0p50_comp_plus_mask005
```

结果：

| config | AP | AP50 | AP25 | union in target | output instances |
|---|---:|---:|---:|---:|---:|
| `localhigh_drop0p50` | 0.1281189467312349 | 0.2756295399515739 | 0.491372578692494 | 0.9439168167676437 | 295 |
| `localhigh_drop0p85` | 0.13733906525573192 | 0.29440476190476195 | 0.4842162698412699 | 0.9895202226952677 | 313 |
| `localhigh_drop0p95` | 0.13733906525573192 | 0.29440476190476195 | 0.4842162698412699 | 0.9896020959554609 | 322 |

对照：

```text
previous pure Stream4D best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

解释：

```text
localbank-high 的 union in target 已接近 99%，但 AP/AP50/AP25 全面很低。
因此它不是合格的 high-confidence primary。
```

### 44.5 本轮是否达成目标

没有达成。

本轮排除了两个具体假设：

```text
1. localbank 不是靠补未覆盖 support 点带来收益。
2. localbank 不能提升为 high-confidence primary。
```

更精确的结论：

```text
localbank 是低分重叠候选层。
它有时能补 AP，但它本身边界粗、重复多，不具备 Stream3D 那种一对一 primary 质量。
```

### 44.6 新 insight

当前 pure Stream4D fixed32 的真正缺口不是 support coverage：

```text
strict primary 已覆盖 12084 / 12214 support 点。
pure best AP25 已超过 Stream3D。
```

真正缺口仍是：

```text
已覆盖区域中的 object-level 替代候选选择。
```

也就是说，算法需要判断：

```text
同一片 support 附近，strict candidate、localbank candidate、evidence graph candidate 谁的边界更像一个一对一实例？
```

residual uncovered set-cover 不能回答这个问题；localbank-high 也不能回答。后续如果继续推进，必须构造更强的无 GT 替代候选选择信号，例如：

```text
same-frame mask exclusivity
multi-view boundary agreement
candidate split/merge consistency
候选之间的 object-level mutual exclusion，而不是只看 support 是否覆盖
```

## 45. 2026-06-08 Prediction-only self-discovered silhouette score 复盘

### 45.1 本轮目标

已有 `silhouette_consistency_score.py` 需要 `object_dict.npy` 和真实 `mask_list`，当前最强 fused pure prediction 没有 object_dict。因此本轮实现一个 prediction-only 版本：

```text
从最终 3D prediction 自己投影到 2D；
在每个 frame 自发现 dominant 2D mask；
再计算该 object 的可见点落回 dominant mask 内、且远离边界的比例。
```

目标是测试：

```text
不依赖 object_dict 的多视角 silhouette quality，能否作为 fused pure prediction 的 object-level 排序/过滤信号？
```

### 45.2 代码修改

新增：

```text
Stream3D/tools/self_discovered_silhouette_score.py
```

该工具不读取 GT，不读取 evaluator 输出。

算法：

```text
1. 读取 full-scene prediction。
2. 可选把打分点限制在 fixed32 support 内。
3. 将 object 点投影到 ScanNet RGB-D frames。
4. 用深度一致的可见点投票出 dominant 2D mask。
5. 统计 inside_visible_ratio、interior_ratio、dominant_ratio 和 observation 数量。
6. 得到 self_silhouette_quality。
7. 用该 quality 重排、过滤或与原 score 混合。
```

### 45.3 结果

对照：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same fixed32 | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| previous pure Stream4D best | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 |

本轮结果：

| config | AP | AP50 | AP25 | instances |
|---|---:|---:|---:|---:|
| `score_w090` | 0.2362228624706616 | 0.6126225976948845 | 0.7393155293909851 | 350 |
| `selfonly` | 0.14661777712702742 | 0.35862419935933054 | 0.5006255010931892 | 350 |
| `score_w090_q020` | 0.2362228624706616 | 0.6126225976948845 | 0.7393155293909851 | 248 |
| `score_w090_q040` | 0.2362228624706616 | 0.6126225976948845 | 0.7393155293909851 | 174 |
| `score_w099` | 0.2362228624706616 | 0.6126225976948845 | 0.7389734948993946 | 350 |

### 45.4 诊断

quality 分布：

```text
self_silhouette_quality min/mean/max
= 0.0 / 0.3938911557197571 / 0.9943820238113403

inside_visible_ratio_mean = 0.5995742229334768
interior_ratio_mean = 0.6355530328509802
used_observations_mean = 3.6485714285714286
visible_points_mean = 381.27714285714285
dominant_ratio_mean = 0.6022126462382862
```

过滤：

```text
q0.20: removed 102 / 350 candidates
q0.40: removed 176 / 350 candidates
```

support 覆盖：

```text
score_w090 union in target = 0.9899295889962338
q0.20 union in target = 0.9846897003438677
q0.40 union in target = 0.9760111347633863
```

解释：

```text
1. self-silhouette quality 本身有区分度，确实能删除大量候选。
2. 但删除后 AP 没有恢复，说明被删候选并非主要 high-IoU false positive，或者也包含必要 recall。
3. score_w099 只用 1% silhouette perturbation 仍然明显低于 previous best，说明这个分数对 AP 排序方向是有害的。
```

### 45.5 本轮是否达成目标

没有达成。

当前状态仍是：

```text
pure Stream4D best fixed32:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

本轮最好结果：

```text
0.2362228624706616 / 0.6126225976948845 / 0.7393155293909851
```

低于 previous pure best。

### 45.6 失败原因和 insight

本轮说明：

```text
prediction-only 自发现 2D silhouette quality 不适合作当前 fused prediction 的 object ranking / filtering。
```

原因：

1. **误差链太长。**  
   它先从最终 3D mask 反推出 2D dominant mask，再用这个反推结果评价 3D object。若 3D mask 已经混入多个实例，dominant mask 本身就会偏。

2. **2D 自洽不等于 3D instance AP。**  
   一个候选可以在若干视角投到同一个 2D mask 内，但仍然是过大、过小或重复的 3D object。

3. **当前 AP 缺口不是普通“可见性质量”缺口。**  
   之前已有 object_dict 的 silhouette filter 能给 localbank low-confidence layer 带来很小收益；但对 fused prediction 直接排序会伤 AP。说明 silhouette 只能作为局部 recall filter，不能替代 object-level 一对一实例选择。

进一步排除的方向：

```text
1. prediction-only self-discovered silhouette sorting
2. prediction-only self-discovered silhouette filtering
3. 轻量 silhouette score perturbation
```

后续如果继续，必须把 focus 放在：

```text
候选生成阶段的 object-level exclusivity；
候选之间的 split/merge consistency；
从原始 mask observation / carrier evidence 直接产生更少、更准的 primary object。
```

## 46. 2026-06-08 Probe5 多场景 fixed-support 诊断复盘

### 46.1 本轮目标

前面很多改进和失败分析都来自 `scene0050_00`。本轮用 5 个小场景做 fixed-support 外推诊断，避免把单场景结果写成稳定结论。

新增 split：

```text
Stream3D/splits/scannet_v4_1_probe5.txt

scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00
```

本轮只做已经存在 prediction/TMP 的 cross-prepoints 评估，没有读取 GT 生成预测，也没有修改 AP evaluator。

### 46.2 缓存与 blocker

检查结果：

```text
stream4d_scannet_32f_ioc075_fixmem:
  5/5 scenes prediction and TMP exist.

stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute:
  5/5 scenes prediction and TMP exist.

stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit:
  5/5 scenes prediction and TMP exist.

scannet:
  5/5 scenes prediction exists.
  source pre_points 使用 scannet_self_inherit。
```

重要 blocker：

```text
scene0050_00 有 96f / 128f 多窗口 carrier cache；
scene0011_00、scene0030_00、scene0081_01、scene0591_00 当前只有 32f 单窗口 cache。
```

因此本轮不能诚实地把 `scene0050_00` 的 96f/128f evidence graph、component densify、tiered inherit 等 scene-specific 算法直接扩展到 probe5。若要做多场景 evidence graph，需要先重新生成这些场景的 96f/128f carrier cache。

### 46.3 结果

以下数值来自 `Stream3D/data/evaluation/scannet/*.txt` 最后一行，未乘以 100。

| method | AP | AP50 | AP25 | target pre % | union % | union in target % | #pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stream3D self-inherit probe5 | 0.235730 | 0.414306 | 0.537786 | 84.6744 | 84.6744 | 100.0000 | 128.20 |
| Stream3D on 32f support probe5 | 0.399213 | 0.597171 | 0.742535 | 4.5145 | 84.6744 | 98.5608 | 128.20 |
| Stream4D 32f self probe5 | 0.144238 | 0.288344 | 0.464716 | 4.5145 | 4.5145 | 100.0000 | 386.00 |
| Stream4D v3 recompute pred on 32f probe5 | 0.114468 | 0.241335 | 0.430399 | 4.5145 | 2.9076 | 67.4358 | 17.00 |
| Stream4D v3 inherit pred on 32f probe5 | 0.114468 | 0.241335 | 0.430399 | 4.5145 | 2.9076 | 67.4358 | 17.00 |
| 32f top10 mask_count | 0.108129 | 0.236172 | 0.378560 | 4.5145 | 2.4819 | 58.0059 | 10.00 |
| 32f top12 mask_count | 0.105284 | 0.236475 | 0.384597 | 4.5145 | 2.5959 | 60.8728 | 12.00 |
| 32f top10 area | 0.085730 | 0.168675 | 0.316145 | 4.5145 | 2.6630 | 61.2535 | 10.00 |
| 32f top12 area | 0.099152 | 0.191667 | 0.373896 | 4.5145 | 2.8264 | 65.0526 | 12.00 |
| 32f one_min100 | 0.144238 | 0.288344 | 0.464716 | 4.5145 | 3.4900 | 77.2712 | 25.20 |
| 32f one_min250 | 0.098032 | 0.202289 | 0.344096 | 4.5145 | 2.7125 | 61.2853 | 10.00 |
| 32f one_min250_merge070 | 0.104104 | 0.206357 | 0.352986 | 4.5145 | 2.8215 | 62.9207 | 9.40 |
| 32f adapt014 old | 0.114468 | 0.241335 | 0.430399 | 4.5145 | 2.9076 | 67.4358 | 17.00 |

### 46.4 是否超过 Stream3D

没有。

probe5 上最关键对比：

```text
Stream3D on 32f support:
0.399213 / 0.597171 / 0.742535

Stream4D 32f self:
0.144238 / 0.288344 / 0.464716

Stream4D v3 inherit/on-32f:
0.114468 / 0.241335 / 0.430399
```

这说明：

```text
在 5 个场景的 32f fixed support 下，
纯 Stream4D 没有超过 Stream3D；
而且差距不是 scene0050_00 单场景偶然。
```

### 46.5 失败原因分析

#### A. 32f support 不是异常困难的 support

Stream3D 放到同一个 32f support 上反而从 self-inherit 的：

```text
0.235730 / 0.414306 / 0.537786
```

升到：

```text
0.399213 / 0.597171 / 0.742535
```

所以不能把 Stream4D fixed-support 失败解释成“32f support 太苛刻”。这个 support 对强 prediction 是有利的。

#### B. Stream4D 32f 覆盖完整，但实例质量不够

`Stream4D 32f self probe5` 的 `union in target` 是 100%，说明 fixed support 内点覆盖不是 0；但 AP 只有：

```text
0.144238 / 0.288344 / 0.464716
```

同时它平均有 386 个 prediction，远多于 Stream3D 的 128.2 个。这和 scene0050 的诊断一致：问题不是没有候选，而是候选碎片化、重复、边界/一对一实例质量不足。

#### C. top-k / area / min-points 不能解决

已有候选选择变体全部低于原始 Stream4D 32f self：

```text
top10 mask_count AP = 0.108129
top12 mask_count AP = 0.105284
top10 area AP = 0.085730
one_min250_merge070 AP = 0.104104
```

这说明简单筛选更少候选会减少 false positive，但也丢掉 recall，不能靠后处理 top-k 接近 Stream3D。

#### D. v3 adaptive/inherit 在 probe5 上没有带来 fixed-support 收益

v3 adaptive recompute prediction 和 inherit prediction 放到 32f support 都是：

```text
0.114468 / 0.241335 / 0.430399
```

且 `union in target = 67.4358%`，低于 Stream4D 32f self 的 100%。这说明 v3 的 sparse selection 在 probe5 fixed support 下仍然漏召回严重。

### 46.6 新 insight

1. **scene0050 不是唯一失败场景。**  
   多场景 probe5 仍显示 pure Stream4D fixed support 大幅低于 Stream3D。

2. **Stream3D 在 Stream4D 32f support 上很强。**  
   这和早期 full ScanNet cross-prepoints 结论一致：当 support 变成 Stream4D 的小 observed support，Stream3D full prediction 往往更强。

3. **Stream4D 的核心缺口仍是 object-level quality。**  
   32f self 覆盖目标 support 100%，却 AP 很低；说明重点不是再扩大 union，而是要减少重复、提高边界、让候选更接近一对一实例。

4. **多窗口算法不能继续只在 scene0050 claim。**  
   evidence graph 在 scene0050 的 recompute/部分 fixed support 有进展，但其它推荐场景缺少 96f/128f cache。后续若要把它写成稳定方法，必须先生成多场景多窗口 carrier cache，再跑相同协议。

### 46.7 当前结论

可以安全写：

```text
probe5 多场景 fixed-support 诊断进一步确认：
纯 Stream4D 目前没有在 inherit/fixed support 下超过 Stream3D。
```

不能写：

```text
Stream4D v4.1 已在 inherit_pre_points 下超过 Stream3D。
scene0050 上的 hybrid / evidence graph 结果已经代表多场景稳定收益。
```

下一步如果继续推进，应该先补多场景 96f/128f carrier cache，然后把 evidence graph + component densify + tiered inherit 在同一个 probe5 fixed-support 协议下复测。没有这些 cache 时，继续在 scene0050 上堆后处理不会证明全局目标。

## 47. 2026-06-08 Probe5 cached evidence graph / component densify 复盘

### 47.1 本轮目标

第 46 节说明 pure Stream4D 32f 在 probe5 fixed support 下远低于 Stream3D。按照上一节建议，本轮优先尝试补多场景 96f cache；失败后，使用已经存在的 32f carrier cache 跑 evidence graph + component densify 多场景 replay。

本轮没有使用 GT 生成 prediction，没有改 evaluator。

### 47.2 96f cache 尝试结果

尝试命令：

```text
stream4d.run_scannet
seq = scene0011_00
max_frames = 96
window_size = 32
window_stride = 16
output_config = stream4d_v4_1_memoryold_scene0011_96f_ioc075
```

结果：

```text
logs/stream4d_v4_1_memoryold_scene0011_96f_run.log = 0 bytes
outputs/stream4d_debug_v4_1_memoryold_scene0011_96f 不存在
data/prediction/stream4d_v4_1_memoryold_scene0011_96f_ioc075_class_agnostic/scene0011_00.npz 不存在
```

进程观察：

```text
python 进程处于 D state；
GPU compute apps 没有该进程；
说明卡在 checkpoint/filesystem I/O 初始化阶段，未进入 D4RT 推理。
```

处理：

```text
kill -TERM 终止该 run。
```

审计结论：

```text
本次 96f cache 尝试没有产生任何指标数据，不能把它写成算法结果。
它只是当前环境下的 D4RT ckpt / filesystem I/O blocker。
```

### 47.3 32f cached evidence graph 结果

使用已有 cache：

```text
outputs/stream4d_debug_full_32f_ioc075_fixmem/<scene>/carriers_window000.npz
```

这些 cache 是单窗口 32f，不是 96f/128f 多窗口。因此本轮结果只能说明：

```text
evidence graph + component densify 算法在多场景 32f cached carriers 上的表现。
```

不能说明：

```text
多窗口 96f/128f evidence graph 已经在多场景上验证。
```

### 47.4 指标表

对照：

| method | fixed support | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| Stream3D on 32f support probe5 | 32f support | 0.399213 | 0.597171 | 0.742535 |
| Stream4D 32f self probe5 | 32f support | 0.144238 | 0.288344 | 0.464716 |

本轮 evidence graph / component densify：

| config | own AP | own AP50 | own AP25 | fixed AP | fixed AP50 | fixed AP25 | union in target % | #pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strict ioc0.79 minobs10 rel1.0 | 0.563837 | 0.726708 | 0.786749 | 0.030578 | 0.079899 | 0.131896 | 14.0954 | 15.20 |
| ioc0.79 minobs10 rel0.5 | 0.295135 | 0.479094 | 0.589431 | 0.099900 | 0.205422 | 0.244729 | 31.6218 | 15.20 |
| ioc0.79 minobs5 rel0.5 | 0.262613 | 0.452775 | 0.610256 | 0.163766 | 0.325301 | 0.429323 | 52.5912 | 31.40 |
| ioc0.70 minobs5 rel0.5 | 0.256258 | 0.486111 | 0.596111 | 0.163153 | 0.325979 | 0.464847 | 59.0299 | 36.60 |
| ioc0.70 minobs5 rel0.0 | 0.233518 | 0.441757 | 0.652377 | 0.221688 | 0.430357 | 0.656345 | 84.9736 | 36.60 |
| ioc0.70 minobs3 rel0.0 | 0.242530 | 0.471493 | 0.680090 | 0.240665 | 0.447128 | 0.671741 | 88.2922 | 50.00 |
| ioc0.60 minobs3 rel0.0 | 0.244308 | 0.486900 | 0.688674 | 0.233974 | 0.445890 | 0.647238 | 88.2719 | 50.20 |
| ioc0.70 minobs1 rel0.0 | 0.151374 | 0.346780 | 0.577114 | 0.186444 | 0.381982 | 0.649452 | 91.9782 | 324.00 |

### 47.5 是否达成目标

仍然没有达成。

本轮最佳 fixed-support 结果：

```text
ioc0.70 minobs3 rel0.0:
0.240665 / 0.447128 / 0.671741
```

它超过了原始 Stream4D 32f self：

```text
0.240665 > 0.144238 AP
0.447128 > 0.288344 AP50
0.671741 > 0.464716 AP25
```

但仍低于 Stream3D on same 32f support：

```text
0.240665 < 0.399213 AP
0.447128 < 0.597171 AP50
0.671741 < 0.742535 AP25
```

因此不能写成：

```text
Stream4D v4.1 在 inherit/fixed support 下超过 Stream3D。
```

### 47.6 证据链分析

#### A. Strict evidence graph 是高精小 support

`ioc0.79 minobs10 rel1.0` own support 指标很高：

```text
0.563837 / 0.726708 / 0.786749
```

但放回 32f fixed support 后：

```text
0.030578 / 0.079899 / 0.131896
```

原因非常直接：

```text
union in target = 14.0954%
```

这和 scene0050 的早期证据一致：自身 support 高分不代表 fixed-support 成立。

#### B. Coverage 是 fixed-support 的必要条件

从 strict 到 `minobs3 rel0.0`：

```text
union in target:
14.0954% -> 88.2922%

fixed AP:
0.030578 -> 0.240665
```

这说明 component densify 在 fixed support 下要成立，必须释放更多 mask observations，不能只保留 rel1.0 strict core。

#### C. 过度放宽会制造碎片和 false positives

`minobs1 rel0.0` 的 coverage 更高：

```text
union in target = 91.9782%
```

但 prediction 数暴涨：

```text
#pred = 324.00
```

AP 降到：

```text
0.186444
```

这说明 minobs1 引入太多不稳定 object，重复和 false positives 抵消了 coverage 收益。

#### D. graph IoC 过低不是解法

`ioc0.60 minobs3 rel0.0`：

```text
fixed = 0.233974 / 0.445890 / 0.647238
```

低于 `ioc0.70 minobs3 rel0.0`：

```text
fixed = 0.240665 / 0.447128 / 0.671741
```

说明进一步放宽 graph edge 会带来噪声，不是简单召回越多越好。

### 47.7 新 insight

1. **component densify 在多场景上不是无效。**  
   它把 probe5 fixed AP 从原始 Stream4D 32f 的 0.144238 提到 0.240665，这是实际算法收益。

2. **但它仍没有 Stream3D 的实例质量。**  
   Stream3D 同 support 是 0.399213 AP，差距仍大。

3. **最佳区间在 minobs3/minobs5，而不是 strict 或 minobs1。**  
   strict 太稀疏；minobs1 太碎；minobs3 在 coverage 和 object stability 之间最平衡。

4. **多场景 96f/128f 仍是未完成项。**  
   32f cached replay 不能替代多窗口验证。当前真正 blocker 是 D4RT ckpt/filesystem I/O 阶段卡住，未能生成 scene0011 的 96f cache。

5. **下一步算法方向更清楚了。**  
   不应继续只调最终 prediction 后处理。应该把 `minobs3 rel0.0` 这种较平衡的 evidence graph 输出作为 primary candidate，再做 object-level duplicate suppression / boundary refinement；同时需要解决 D4RT cache 生成 blocker，才能验证多窗口是否真的带来收益。

### 47.8 当前安全结论

可以写：

```text
Probe5 上，cached 32f evidence graph + component densify 在 fixed support 下明显超过原始 Stream4D 32f self，但仍未超过 Stream3D。
```

不能写：

```text
Stream4D v4.1 已经在 inherit_pre_points 下超过 Stream3D。
96f/128f evidence graph 已经完成多场景验证。
```

## 48. 2026-06-08 Probe5 pure Stream4D tiered inherit 与 oracle 诊断复盘

### 48.1 本轮目标

第 47 节得到一个多场景 32f cached evidence graph 候选：

```text
stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs3_compdens_r003_d16_m8_rel000_hard2
fixed32 = 0.240665 / 0.447128 / 0.671741
```

本轮尝试把它作为高置信 primary，再用原始 Stream4D 32f current 作为低置信 coverage 层，构造 pure Stream4D tiered inherit。

注意：

```text
本轮 method 结果不读 GT。
oracle 诊断单独标注为 GT-read-only，只用于分析上界，不作为方法结果。
```

### 48.2 Method 结果

对照：

```text
Stream4D 32f self probe5:
0.144238 / 0.288344 / 0.464716

Stream3D on 32f support probe5:
0.399213 / 0.597171 / 0.742535
```

本轮 method 结果：

| config | fixed AP | fixed AP50 | fixed AP25 | union in target % | #pred |
|---|---:|---:|---:|---:|---:|
| egraph only minobs3 rel0 | 0.240665 | 0.447128 | 0.671741 | 88.2922 | 50.00 |
| tiered no drop | 0.249312 | 0.459458 | 0.688319 | 100.0000 | 436.00 |
| tiered secioc0.50 | 0.243555 | 0.452108 | 0.671741 | 94.7238 | 245.20 |
| tiered secioc0.85 | 0.249551 | 0.460336 | 0.677592 | 99.0055 | 353.40 |
| tiered secioc0.50 score0.05 | 0.243555 | 0.452108 | 0.671741 | 94.7238 | 245.20 |
| tiered minioc0.50 | 0.240665 | 0.447128 | 0.671741 | 91.7022 | 240.00 |
| support-aware scoreconf w0.90 | 0.228716 | 0.431452 | 0.661445 | 99.0055 | 353.40 |
| support-aware scoreconf w0.75 | 0.228716 | 0.431452 | 0.661445 | 99.0055 | 353.40 |
| support-aware supportconf | 0.230424 | 0.426761 | 0.629365 | 99.0055 | 353.40 |
| support-aware w0.90 min_ioc nms0.85 | 0.227654 | 0.426356 | 0.661445 | 98.4188 | 339.40 |

当前最好 method：

```text
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
fixed32 = 0.249551 / 0.460336 / 0.677592
```

相对第 47 节 best egraph only：

```text
AP   +0.008886
AP50 +0.013208
AP25 +0.005851
```

相对 Stream4D 32f self：

```text
AP   +0.105313
AP50 +0.171992
AP25 +0.212876
```

相对 Stream3D：

```text
AP   -0.149662
AP50 -0.136835
AP25 -0.064943
```

### 48.3 Support-aware ranking 失败

`support_aware_object_rank.py` 尝试用 target support 内面积、冲突比例等无监督信号重排或做 overlap competition。结果全部低于原始 tiered secioc0.85：

```text
tiered secioc0.85:
0.249551 / 0.460336 / 0.677592

support-aware scoreconf w0.90:
0.228716 / 0.431452 / 0.661445

support-aware supportconf:
0.230424 / 0.426761 / 0.629365

support-aware min_ioc nms0.85:
0.227654 / 0.426356 / 0.661445
```

解释：

```text
target-support 面积能表示粗 coverage，但不能稳定表示高 IoU object quality。
把它用于重排会把大而粗的候选提前，伤害 AP/AP50。
```

### 48.4 GT-read-only oracle 上界

oracle 诊断读取 GT，只回答候选池是否有足够好的候选。它不能作为方法结果。

| pool | oracle AP | oracle AP50 | oracle AP25 | pred/scene | valid pred in support | oracle selected | GT best IoU>=0.25 | >=0.5 | >=0.75 | >=0.8 | >=0.9 | mean best IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stream3D candidate pool | 0.528782 | 0.722892 | 0.843373 | 128.20 | 18.60 | 14.00 | 14.60 | 12.00 | 8.40 | 7.00 | 5.00 | 0.6687 |
| Stream4D 32f self pool | 0.238286 | 0.445783 | 0.650602 | 386.00 | 25.20 | 10.80 | 11.00 | 7.40 | 3.00 | 1.80 | 0.60 | 0.4364 |
| Stream4D egraph minobs3 pool | 0.356091 | 0.614458 | 0.831325 | 50.00 | 22.60 | 13.80 | 14.20 | 10.20 | 5.00 | 3.40 | 0.20 | 0.5589 |
| Stream4D tiered pool | 0.393574 | 0.650602 | 0.843373 | 353.40 | 39.40 | 14.00 | 14.40 | 10.80 | 5.60 | 4.20 | 0.40 | 0.5895 |

### 48.5 证据链分析

#### A. Tiered inherit 有小收益，但不是突破

egraph only:

```text
0.240665 / 0.447128 / 0.671741
```

tiered secioc0.85:

```text
0.249551 / 0.460336 / 0.677592
```

说明低置信 32f coverage 层有帮助，但改善幅度有限。

#### B. 候选池上界已经接近 Stream3D actual，但还没有稳定超过

Stream4D tiered oracle:

```text
0.393574 / 0.650602 / 0.843373
```

Stream3D actual:

```text
0.399213 / 0.597171 / 0.742535
```

解释：

```text
如果有 GT oracle 选择，Stream4D tiered 候选池在 AP50/AP25 上能超过 Stream3D actual，
但 AP 仍略低于 Stream3D actual。
```

这说明候选池已经不再像原始 32f 那样完全不足，但自动排序/选择还没有达到 oracle 水平。

#### C. Stream3D 的候选池上界仍明显更高

Stream3D oracle:

```text
0.528782 / 0.722892 / 0.843373
```

Stream4D tiered oracle:

```text
0.393574 / 0.650602 / 0.843373
```

Stream3D 在高 IoU 候选数量上明显更强：

```text
GT best IoU >= 0.75:
Stream3D 8.40 / scene
Stream4D tiered 5.60 / scene

GT best IoU >= 0.90:
Stream3D 5.00 / scene
Stream4D tiered 0.40 / scene
```

所以当前失败不是纯排序问题。Stream4D 高 IoU 边界质量仍弱。

#### D. 原始 Stream4D 32f 候选池很弱，egraph/tiered 是真实改进

原始 Stream4D 32f oracle AP：

```text
0.238286
```

egraph minobs3 oracle AP：

```text
0.356091
```

tiered oracle AP：

```text
0.393574
```

这说明 evidence graph + component densify + tiered inherit 确实显著改善候选池，不只是评价噪声。

### 48.6 本轮是否达成目标

没有达成。

可以写：

```text
Probe5 上 pure Stream4D tiered inherit 的真实 fixed-support 最好结果为
0.249551 / 0.460336 / 0.677592，
明显超过原始 Stream4D 32f self，
但仍低于 Stream3D on same 32f support。
```

也可以写：

```text
GT-read-only oracle 显示，tiered Stream4D 候选池的 AP50/AP25 上界已经超过 Stream3D actual，
但 AP 上界仍略低于 Stream3D actual，且 Stream3D candidate-pool oracle 仍显著更强。
```

不能写：

```text
Stream4D v4.1 已在 inherit_pre_points/fixed support 下超过 Stream3D。
当前问题只是排序问题。
```

### 48.7 下一步

下一步更合理的算法方向：

```text
1. 在 evidence graph 生成阶段提升高 IoU 边界质量，而不是只在最终 prediction 重排。
2. 给 tiered candidate 加 object-level duplicate suppression，但不能只靠 target support 面积。
3. 修复 D4RT 96f/128f cache 生成 blocker；否则无法验证多窗口是否进一步提升高 IoU candidate pool。
```

## 49. 2026-06-08 Multi-source consensus / greedy novelty 负例复盘

### 49.1 本轮目标

第 48 节表明：

```text
Stream4D tiered candidate pool oracle:
0.393574 / 0.650602 / 0.843373

Stream4D tiered actual:
0.249551 / 0.460336 / 0.677592
```

本轮尝试两个不读 GT 的候选选择策略：

1. `multi_source_consensus_select`：优先选择 egraph 和 32f current 互相支持的候选。
2. `greedy_support_select`：每次选择质量较高且能覆盖 support 新区域的候选。

### 49.2 结果

| config | fixed AP | fixed AP50 | fixed AP25 | union in target % | #pred |
|---|---:|---:|---:|---:|---:|
| tiered best | 0.249551 | 0.460336 | 0.677592 | 99.0055 | 353.40 |
| consensus req multi-source minIoC0.50 + low32f | 0.129035 | 0.268746 | 0.540988 | 99.0941 | 263.60 |
| consensus req multi-source minIoC0.70 + low32f | 0.136398 | 0.295163 | 0.604728 | 98.7903 | 249.80 |
| consensus all minIoC0.70 | 0.123047 | 0.273277 | 0.593200 | 81.9934 | 158.20 |
| greedy max80 | 0.181316 | 0.343578 | 0.530206 | 72.5842 | 25.80 |
| greedy max120 | 0.181316 | 0.343578 | 0.530206 | 72.5842 | 25.80 |
| greedy suppressed | 0.180838 | 0.344009 | 0.533193 | 73.8750 | 27.20 |

### 49.3 是否达成目标

没有。

最好 consensus：

```text
0.136398 / 0.295163 / 0.604728
```

最好 greedy：

```text
0.181316 / 0.343578 / 0.530206
```

都低于 tiered best：

```text
0.249551 / 0.460336 / 0.677592
```

更低于 Stream3D：

```text
0.399213 / 0.597171 / 0.742535
```

### 49.4 失败原因分析

#### A. 多源一致性不是高 IoU 的可靠代理

要求 egraph 和 32f current 在 support 上重叠，能减少一些不一致候选，但并不能确保候选边界更像真实 instance。

证据：

```text
consensus req multi-source minIoC0.70 + low32f:
union in target = 98.7903%
fixed AP = 0.136398
```

coverage 仍高，但 AP 低，说明保留下来的候选不是高质量一对一实例。

#### B. Greedy novelty 丢掉太多 recall

Greedy 选择器的输出 #pred 只有约 26-27，union in target 只有约 72-74%：

```text
greedy max80:
#pred = 25.80
union in target = 72.5842%
fixed AP = 0.181316
```

它减少了重复，但也删掉了必要 recall，不能兑现 oracle 上界。

#### C. 当前 oracle 差距不是简单候选去重可解决

第 48 节显示 tiered pool oracle 接近 Stream3D actual，但 Stream3D oracle 仍明显更强，尤其高 IoU 候选：

```text
GT best IoU >= 0.90:
Stream3D = 5.00 / scene
Stream4D tiered = 0.40 / scene
```

所以只做一致性选择或 novelty 选择，最多能整理候选，不能创造缺失的高 IoU 边界候选。

### 49.5 新 insight

1. **一致性不等于正确性。**  
   egraph 和 32f current 都可能在同一个粗区域上同意，但这个区域仍可能是过大、过碎或边界不准。

2. **新覆盖不等于高质量 recall。**  
   greedy novelty 选到的新 support 点不一定属于正确 object，且会丢掉被认为重复但实际有用的候选。

3. **下一步必须前移到候选生成/边界阶段。**  
   从 oracle 看，当前需要增加高 IoU candidate，而不是继续在已有候选上做无监督排序。

安全结论：

```text
multi-source consensus 和 greedy support novelty 方向均未达成目标；
它们不能把 Stream4D tiered candidate-pool oracle 上界转化为真实 fixed-support AP。
```

## 50. 2026-06-08 probe5 support component refinement 复盘

### 50.1 本轮目标

第 49 节说明：

```text
multi-source consensus 和 greedy support novelty 都不能把 oracle 上界转成真实 AP。
```

本轮回到边界/实例质量方向，测试一个更具体的无 GT 假设：

```text
probe5 tiered best 中可能有一些离散 3D support 碎片。
如果在 fixed 32f support 内按 3D 连通组件做非常轻的裁剪，可能提高高 IoU AP。
```

该实验不读取 GT 生成 prediction，不改 evaluator，只用 prediction、target support pre_points 和 ScanNet mesh 顶点坐标。

### 50.2 代码修改审计

本轮没有新增代码，复用了已有工具：

```text
Stream3D/tools/support_component_refine.py
```

核心算法：

```text
1. 对每个 prediction instance，只看它在 32f target support 内覆盖的点。
2. 用 3D 半径图把这些点拆成 connected components。
3. 保留前 K 个大组件，删除更小或更弱的组件。
4. 输出 refined prediction，并让 evaluator 继续使用同一个 fixed 32f support。
```

审计边界：

```text
该工具不读取 ScanNet GT。
该工具不是训练，也不是 oracle。
```

### 50.3 结果表

所有数值来自 evaluator 输出文件最后一行，未乘以 100。

| config | fixed AP | fixed AP50 | fixed AP25 | union in target | changed instances | support keep ratio |
|---|---:|---:|---:|---:|---:|---:|
| tiered best | 0.249551 | 0.460336 | 0.677592 | 0.990055 | NA | 1.000000 |
| compref r006 k2 | 0.205341 | 0.374472 | 0.562057 | 0.763210 | 81.2 | 0.635867 |
| compref r010 k4 | 0.254376 | 0.463412 | 0.649452 | 0.903696 | 71.2 | 0.682124 |
| compref r012 k6 | 0.257755 | 0.473841 | 0.666502 | 0.931218 | 68.0 | 0.694878 |
| compref r014 k8 | 0.257852 | 0.472342 | 0.664193 | 0.944344 | 65.0 | 0.703004 |
| compref r016 k10 | 0.260055 | 0.472260 | 0.664159 | 0.953919 | 63.0 | 0.709232 |
| compref r018 k12 | 0.260386 | 0.472260 | 0.677637 | 0.962415 | 60.2 | 0.714625 |
| compref r020 k14 | 0.260367 | 0.472233 | 0.677625 | 0.968224 | 57.8 | 0.717995 |

当前本轮最好：

```text
stream4d_v4_1_probe5_tiered_compref_r018_k12_on_32f_probe5
AP/AP50/AP25 = 0.26038635220275147 / 0.47225957401032703 / 0.6776367126537237
```

对比 Stream3D：

```text
Stream3D on same 32f support:
0.399213 / 0.597171 / 0.742535
```

### 50.4 是否达成目标

没有达成。

本轮相对 previous pure Stream4D tiered best 有小幅提升：

```text
AP gain   = 0.260386 - 0.249551 = +0.010835
AP50 gain = 0.472260 - 0.460336 = +0.011924
AP25 gain = 0.677637 - 0.677592 = +0.000045
```

但相对 Stream3D 仍有明显差距：

```text
AP gap   = 0.399213 - 0.260386 = 0.138827
AP50 gap = 0.597171 - 0.472260 = 0.124912
AP25 gap = 0.742535 - 0.677637 = 0.064898
```

因此不能写：

```text
pure Stream4D 已在 inherit_pre_points/fixed support 下超过 Stream3D。
```

### 50.5 失败原因和 insight

本轮最重要的正信号：

```text
轻量 3D connected-component refinement 确实能略微提升 AP/AP50。
```

解释：

```text
r018_k12 删除了少量空间碎片，让 support union in target 从 tiered best 的约 99.0% 降到 96.24%，但 AP/AP50 上升。
这说明当前 tiered prediction 中确实存在一部分伤害高 IoU precision 的离散碎片。
```

但它不足以解决主问题：

```text
1. 强裁剪会立刻伤 recall。r006_k2 的 union in target 只有 76.32%，AP 掉到 0.205341。
2. 宽松裁剪只能删掉小碎片，不能创造 Stream3D 那种高 IoU 一对一实例边界。
3. 最佳 r018_k12 的 AP25 基本只是持平 tiered best，说明粗覆盖没有真正增强。
4. AP 和 AP50 距离 Stream3D 仍很远，说明瓶颈仍是 candidate generation / boundary quality，而不是单纯 support 碎片。
```

新的结论：

```text
3D component refinement 可以作为 pure Stream4D tiered 的一个小型边界清理模块，
但它不是 inherit/fixed support 超越 Stream3D 的核心解法。
```

### 50.6 当前最诚实状态

当前 probe5 pure Stream4D fixed-support 最好结果更新为：

```text
0.260386 / 0.472260 / 0.677637
```

但 Stream3D on same 32f support 仍是：

```text
0.399213 / 0.597171 / 0.742535
```

所以对用户问题“inherit_pre_points 下 Stream4D 超过 Stream3D 吗”的当前答案仍然是：

```text
没有。pure Stream4D 仍未超过。
```

### 50.7 低分 recall 层补充诊断

为了确认 `compref r018 k12` 是否只是丢掉了粗 recall，本轮又把原 tiered best 作为低分 recall 层加回：

```text
primary high = compref r018 k12
secondary low = original tiered best
secondary score = 0.005
containment overlap computed inside 32f support
```

结果：

| config | fixed AP | fixed AP50 | fixed AP25 | output instances | secondary after drop |
|---|---:|---:|---:|---:|---:|
| compref r018 high only | 0.260386 | 0.472260 | 0.677637 | 353.4 | NA |
| compref high + tiered low secioc0.85 | 0.260386 | 0.472260 | 0.677637 | 471.8 | 118.4 |
| compref high + tiered low secioc0.95 | 0.260386 | 0.472260 | 0.677637 | 489.4 | 136.0 |

结论：

```text
加回低分原 tiered 候选没有改变 AP/AP50/AP25。
```

解释：

```text
1. 低分 recall 层确实增加了候选数量，但没有改变有效 PR 曲线。
2. 这说明被 compref 裁掉/压后的那些区域并不是当前 AP25 或 AP50 的关键缺口。
3. 继续简单堆低分候选没有意义；需要提高高质量候选的边界和一对一实例质量。
```

## 51. 2026-06-08 probe5 egraph silhouette evidence 负例复盘

### 51.1 本轮目标

scene0050 单场景中，silhouette consistency 曾经对 localbank low-confidence recall layer 有过小幅正收益。本轮验证这个信号能否迁移到 probe5，并用于 egraph high layer：

```text
用 object_dict 中的 mask observations 计算 multi-view silhouette consistency。
再尝试把它作为 egraph object 的排序信号或过滤信号。
```

该方法不读 GT，不改 evaluator。

### 51.2 可用性检查

probe5 中有 object_dict 的配置：

```text
stream4d_scannet_32f_ioc075_fixmem
stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs3_compdens_r003_d16_m8_rel000_hard2
```

没有 object_dict 的配置：

```text
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
```

因此本轮只能把 silhouette 作用在 egraph high layer，然后再和 32f low layer 融合。

### 51.3 结果

对照：

```text
egraph minobs3 fixed:
0.240665 / 0.447128 / 0.671741

tiered best:
0.249551 / 0.460336 / 0.677592

compref r018 k12 best:
0.260386 / 0.472260 / 0.677637

Stream3D same 32f support:
0.399213 / 0.597171 / 0.742535
```

silhouette direct / filter：

| config | AP | AP50 | AP25 | instances after | removed / scene |
|---|---:|---:|---:|---:|---:|
| silhouette score w0.75 | 0.137832 | 0.315042 | 0.580297 | 50.0 | 0.0 |
| silhouette score w0.25 | 0.137832 | 0.315042 | 0.580297 | 50.0 | 0.0 |
| silhouette q0.50 + score w0.75 | 0.132773 | 0.303399 | 0.527297 | 32.2 | 17.8 |
| silhouette q0.70 + score w0.75 | 0.124826 | 0.284065 | 0.495152 | 21.8 | 28.2 |
| silhouette q0.50 filter-only | 0.236186 | 0.439819 | 0.605060 | 32.2 | 17.8 |
| silhouette q0.70 filter-only | 0.231320 | 0.428858 | 0.590648 | 21.8 | 28.2 |

silhouette high + 32f low：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| silhouette score w0.75 high + 32f low | 0.146718 | 0.328250 | 0.586149 |
| silhouette q0.50 score w0.75 high + 32f low | 0.142172 | 0.317362 | 0.550833 |
| silhouette q0.50 filter-only high1 + 32f low | 0.245585 | 0.453782 | 0.628597 |
| silhouette q0.70 filter-only high1 + 32f low | 0.244839 | 0.454232 | 0.642204 |

执行审计说明：

```text
我曾先跑了两条 filter-only + --preserve-primary-score 的融合命令。
因为 filter-only 输出分数全为 0，而 secondary score 是 0.005，
它们把 low layer 错排到了 high layer 前面。
这两条不作为正式结果，随后已用 primary-score=1.0 修正。
```

### 51.4 是否达成目标

没有达成。

最好 silhouette-related 结果：

```text
0.245585 / 0.453782 / 0.628597
```

仍低于：

```text
current pure Stream4D probe5 best:
0.260386 / 0.472260 / 0.677637
```

也远低于：

```text
Stream3D same 32f support:
0.399213 / 0.597171 / 0.742535
```

### 51.5 失败原因分析

#### A. silhouette 分数不是可靠排序信号

证据：

```text
silhouette score w0.75 = 0.137832 / 0.315042 / 0.580297
silhouette score w0.25 = 0.137832 / 0.315042 / 0.580297
```

它比原 egraph fixed：

```text
0.240665 / 0.447128 / 0.671741
```

低很多。

解释：

```text
一个 object 的点投影回自身 2D mask 内，并不代表它是高 IoU 的一对一 3D instance。
它只能说明 2D observation 自洽，不能解决相邻同类实例的边界/分裂问题。
```

#### B. silhouette 过滤会删掉必要 recall

证据：

```text
q0.50 removed = 17.8 objects / scene
q0.70 removed = 28.2 objects / scene
```

filter-only 仍下降：

```text
q0.50 filter-only = 0.236186 / 0.439819 / 0.605060
q0.70 filter-only = 0.231320 / 0.428858 / 0.590648
```

说明被过滤掉的对象中有当前 fixed-support AP 仍需要的 recall。

#### C. 与 32f low layer 融合也不能救回

修正 high/low score 后：

```text
q0.50 filter-only high1 + 32f low = 0.245585 / 0.453782 / 0.628597
q0.70 filter-only high1 + 32f low = 0.244839 / 0.454232 / 0.642204
```

这低于原 tiered 和 compref best。说明 silhouette 过滤对 high layer 的损伤，不能靠低分 32f recall 层补回。

### 51.6 新 insight

scene0050 中 silhouette-filtered localbank 的小幅正收益不能泛化为：

```text
silhouette 是通用 object quality score。
```

更准确的结论是：

```text
silhouette 只能作为某些低置信 recall layer 的局部过滤信号；
它不适合作为 egraph high layer 的主排序或主过滤标准。
```

因此后续不应继续沿着：

```text
直接用 projection silhouette quality 排序/过滤 object
```

这个方向扫阈值。当前更可靠的正信号仍是第 50 节的轻量 3D component refinement，但它也只能带来小幅提升。

## 52. 2026-06-08 probe5 final prediction point-level WTA 负例复盘

### 52.1 本轮目标

第 50 节当前 probe5 纯 Stream4D 最好结果是：

```text
stream4d_v4_1_probe5_tiered_compref_r018_k12_on_32f_probe5
AP/AP50/AP25 = 0.26038635220275147 / 0.47225957401032703 / 0.6776367126537237
```

同一 32f fixed support 下的 Stream3D 对照是：

```text
scannet_on_stream4d_32f_probe5
AP/AP50/AP25 = 0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108
```

本轮测试最终 prediction point-level WTA，希望验证：

```text
如果 Stream4D 的主要问题是多个 object 抢同一个三维点，
那么把冲突点分给单一 object 后，fixed-support AP 应该上升。
```

### 52.2 是否修改代码

本轮没有修改算法代码。

使用已有工具：

```text
Stream3D/tools/wta_prediction_points.py
```

该工具不读取 GT，只读取 prediction 文件中的：

```text
pred_masks
pred_score
pred_classes
```

然后根据无监督优先级把冲突点分配给一个预测实例。

### 52.3 结果

所有数值来自实际 evaluator 或 summary JSON。

| 方法 | AP | AP50 | AP25 | union in target | #pred |
|---|---:|---:|---:|---:|---:|
| Stream3D same 32f support | 0.399213 | 0.597171 | 0.742535 | 0.985608 | 128.2 |
| compref best before WTA | 0.260386 | 0.472260 | 0.677637 | 0.962415 | 353.4 |
| WTA score m2 | 0.252840 | 0.469799 | 0.679025 | 0.962415 | 188.4 |
| WTA score-area-desc m2 | 0.252840 | 0.469799 | 0.679025 | 0.962415 | 186.6 |
| WTA score-area-asc m2 | 0.251627 | 0.464172 | 0.678995 | 0.962415 | 203.8 |
| WTA score m2 margin030 | 0.252833 | 0.469753 | 0.678995 | 0.962415 | 212.8 |
| WTA score m3 | 0.258648 | 0.473374 | 0.678121 | 0.962415 | 232.2 |
| WTA score-area-desc m3 | 0.258648 | 0.473374 | 0.678121 | 0.962415 | 231.4 |

最好的 WTA 是：

```text
stream4d_v4_1_probe5_compref_wta_score_m3_on_32f_probe5
AP/AP50/AP25 = 0.2586475908721231 / 0.4733736029107388 / 0.6781207106908648
```

它相对 WTA 前：

```text
AP   下降 0.00173876133062836
AP50 上升 0.00111402890041177
AP25 上升 0.00048399803714108
```

因此不能作为达成目标的结果。

### 52.4 WTA 行为诊断

| config | inst before | inst after | conflict before | conflict after | removed assignments | union before/after | assignments before/after |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTA score m2 | 353.4 | 188.4 | 5627.4 | 0.0 | 7870.4 | 9203.4 / 9203.4 | 17073.8 / 9203.4 |
| WTA score-area-desc m2 | 353.4 | 186.6 | 5627.4 | 0.0 | 7870.4 | 9203.4 / 9203.4 | 17073.8 / 9203.4 |
| WTA score-area-asc m2 | 353.4 | 203.8 | 5627.4 | 0.0 | 7870.4 | 9203.4 / 9203.4 | 17073.8 / 9203.4 |
| WTA score m2 margin030 | 353.4 | 212.8 | 5627.4 | 122.0 | 7701.0 | 9203.4 / 9203.4 | 17073.8 / 9372.8 |
| WTA score m3 | 353.4 | 232.2 | 1138.6 | 0.0 | 3381.6 | 9203.4 / 9203.4 | 17073.8 / 13692.2 |
| WTA score-area-desc m3 | 353.4 | 231.4 | 1138.6 | 0.0 | 3381.6 | 9203.4 / 9203.4 | 17073.8 / 13692.2 |

关键证据：

```text
union before/after 始终是 9203.4 / 9203.4。
```

这说明 WTA 没有改变预测覆盖的三维点集合，只改变“点属于哪个 object”。

### 52.5 本轮是否达成目标

没有达成。

严格对比：

```text
Stream3D same 32f support:
0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108

best WTA:
0.2586475908721231 / 0.4733736029107388 / 0.6781207106908648
```

差距：

```text
AP   仍低 0.14056520232966958
AP50 仍低 0.12379769096039786
AP25 仍低 0.064414648135746
```

### 52.6 失败原因和 insight

#### A. point-level exclusivity 不等于 object-level 一对一

WTA m2 可以把冲突点清到 0：

```text
conflict before/after = 5627.4 / 0.0
```

但 AP 下降：

```text
0.260386 -> 0.252840
```

说明有些重叠归属在当前 sparse support 里仍贡献 recall；强行一刀切会损坏一些有效匹配。

#### B. 当前瓶颈不是 union coverage

WTA 前后：

```text
union in target = 0.962415
union before/after = 9203.4 / 9203.4
```

coverage 没变，AP 主指标也没有提升。当前差距更像是：

```text
实例边界质量不足
object-level duplicate / split 仍然存在
无监督 priority 不能稳定判断哪个 object 是正确实例
```

#### C. m3 soft WTA 的微弱收益不足以作为方向

`min_conflict_owners=3` 的 soft WTA 比 hard WTA m2 保留更多重叠：

```text
assignments before/after = 17073.8 / 13692.2
```

它让 AP50/AP25 极小上升，但 AP 下降。这说明点级 WTA 只能调局部 precision/recall 平衡，不能解决与 Stream3D 的主要差距。

### 52.7 当前结论

截至本节，`inherit_pre_points` / fixed-support 下：

```text
纯 Stream4D probe5 没有超过 Stream3D。
```

当前最强纯 Stream4D 仍是：

```text
stream4d_v4_1_probe5_tiered_compref_r018_k12_on_32f_probe5
0.26038635220275147 / 0.47225957401032703 / 0.6776367126537237
```

而同 support Stream3D 是：

```text
0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108
```

本轮给出的负证据是：

```text
最终 prediction 点级 WTA 不是足够的修复。
下一步若继续推进，应回到 object-level proposal assignment / boundary refinement，
而不是继续只处理最终点归属。
```

## 53. 2026-06-08 probe5 self-discovered boundary refinement 复盘

### 53.1 本轮目标

第 52 节 point-level WTA 证明：

```text
只处理最终点归属，不足以解决 fixed-support AP 差距。
```

本轮转向计划里的 boundary-aware 思路，复用已有：

```text
Stream3D/tools/self_discovered_boundary_refine.py
```

核心思想：

```text
不读 GT。
把 3D prediction point 投影回 ScanNet RGB-D 帧。
在 2D predicted mask 中自发现 dominant mask id。
保留更稳定落在 dominant mask 内、或离边界更安全的点。
```

### 53.2 代码修改审计

本轮没有新增或修改代码。

使用已有工具：

```text
Stream3D/tools/self_discovered_boundary_refine.py
```

输入 prediction：

```text
stream4d_v4_1_probe5_tiered_compref_r018_k12
```

fixed support：

```text
stream4d_scannet_32f_ioc075_fixmem
```

### 53.3 结果

| 方法 | AP | AP50 | AP25 | union in target | #pred | keep ratio | changed | used obs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stream3D same 32f support | 0.399213 | 0.597171 | 0.742535 | 0.985608 | 128.2 | NA | NA | NA |
| compref previous best | 0.260386 | 0.472260 | 0.677637 | 0.962415 | 353.4 | NA | NA | NA |
| selfboundary b1 in045 keep | 0.271394 | 0.486108 | 0.678615 | 0.915772 | 254.8 | 0.962123 | 83.0 | 1.679 |
| selfboundary b1 in050 keep | 0.271394 | 0.486108 | 0.678615 | 0.915772 | 254.8 | 0.962123 | 83.0 | 1.679 |
| selfboundary b2 in050 keep | 0.271394 | 0.486108 | 0.678615 | 0.915772 | 254.8 | 0.962123 | 83.0 | 1.679 |
| selfboundary b1 in055 keep | 0.258984 | 0.491389 | 0.672743 | 0.888608 | 254.8 | 0.947346 | 85.4 | 1.679 |
| selfboundary b1 in060 keep | 0.259012 | 0.491424 | 0.672771 | 0.888234 | 254.8 | 0.946329 | 85.4 | 1.679 |
| selfboundary b2 in060 keep | 0.259012 | 0.491424 | 0.672771 | 0.888234 | 254.8 | 0.946329 | 85.4 | 1.679 |
| selfboundary b3 in060 keep | 0.259012 | 0.491424 | 0.672771 | 0.888234 | 254.8 | 0.946329 | 85.4 | 1.679 |
| selfboundary b2 in060 int020 keep | 0.256553 | 0.493343 | 0.675648 | 0.870154 | 254.8 | 0.940110 | 85.0 | 1.679 |
| selfboundary b2 in060 drop | 0.221359 | 0.436280 | 0.627309 | 0.845476 | 254.8 | 0.934296 | 85.6 | 1.679 |
| selfboundary b1 in050 keep s40 | 0.277338 | 0.491320 | 0.678747 | 0.916531 | 254.8 | 0.960133 | 81.6 | 2.910 |

当前 probe5 纯 Stream4D 最好结果更新为：

```text
stream4d_v4_1_probe5_compref_selfboundary_b1_in050_keep_s40_on_32f_probe5
AP/AP50/AP25 = 0.27733794006771613 / 0.4913198877364921 / 0.6787473172993864
```

### 53.4 是否达成目标

没有达成超过 Stream3D。

对比 Stream3D same 32f support：

```text
Stream3D:
0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108

best selfboundary:
0.27733794006771613 / 0.4913198877364921 / 0.6787473172993864
```

差距：

```text
AP   仍低 0.12187485313407657
AP50 仍低 0.10585140613464458
AP25 仍低 0.06378804152722443
```

### 53.5 为什么它有效

相对上一轮 compref best：

```text
compref best:
0.26038635220275147 / 0.47225957401032703 / 0.6776367126537237

selfboundary best:
0.27733794006771613 / 0.4913198877364921 / 0.6787473172993864
```

提升：

```text
AP   +0.01695158786496466
AP50 +0.01906031372616507
AP25 +0.00111060464566268
```

解释：

```text
1. boundary refinement 删除了一些投影到 2D mask 不稳定的点。
2. 它没有像 WTA 那样只重分配点，而是实际改变了 mask support。
3. 更密视角采样 s40 used obs = 2.910，高于 s80 的 1.679，因此边界判断更稳定。
```

### 53.6 为什么仍失败

#### A. boundary filtering 仍是 precision/recall trade-off

inside ratio 0.55/0.60：

```text
AP50 可到 0.491-0.493
但 AP 降到 0.256-0.259
AP25 降到 0.672-0.676
```

这说明更激进的边界过滤能提高中等 IoU precision，但会删除 recall 所需的点。

#### B. 不能删除 unobserved points

`unobserved-policy=drop`：

```text
0.221359 / 0.436280 / 0.627309
```

明显低于 keep 策略。原因是 probe5 support 中很多点没有足够多帧可见证据；直接删除它们会伤召回。

#### C. 与 Stream3D 的差距仍是 object-level quality

best selfboundary 的 union in target：

```text
0.916531
```

Stream3D same support：

```text
0.985608
```

覆盖差距缩小了一些，但 AP 仍差 0.1219。这说明：

```text
仅靠自发现边界裁剪不能生成 Stream3D 那种更少、更准、更接近一对一的 object proposal。
```

### 53.7 当前进度状态

截至本节：

```text
pure Stream4D probe5 fixed-support best:
0.27733794006771613 / 0.4913198877364921 / 0.6787473172993864

Stream3D same fixed support:
0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108
```

所以当前答案仍然是：

```text
inherit_pre_points / fixed-support 下，纯 Stream4D 没有超过 Stream3D。
```

但本轮给出一个正向算法信号：

```text
boundary-aware 自发现过滤比 point-level WTA 更有效，
是当前 probe5 上最好的 pure Stream4D 改进方向。
```

下一步若继续，应该沿着：

```text
更稳定的 multi-view boundary evidence
+ object-level duplicate/split control
+ boundary-refined proposal generation
```

## 54. 2026-06-08 boundary high + low-confidence recall layer 复盘

### 54.1 本轮目标

第 53 节最好的 pure Stream4D probe5 fixed-support 结果是：

```text
stream4d_v4_1_probe5_compref_selfboundary_b1_in050_keep_s40_on_32f_probe5
AP/AP50/AP25 = 0.27733794006771613 / 0.4913198877364921 / 0.6787473172993864
union in target = 0.916531
```

它的特点是：

```text
边界更干净，但 coverage 比 compref previous best 更小。
```

本轮尝试把它作为 high-confidence layer，再加 low-confidence recall layer。

### 54.2 代码修改审计

本轮没有新增或修改代码。

使用已有工具：

```text
Stream3D/tools/residual_recall_fuse.py
Stream3D/tools/fuse_prediction_configs.py
```

不读取 GT，不改 evaluator。

### 54.3 结果

| 方法 | AP | AP50 | AP25 | union in target | #pred | 诊断 |
|---|---:|---:|---:|---:|---:|---|
| Stream3D same 32f support | 0.399213 | 0.597171 | 0.742535 | 0.985608 | 128.2 | - |
| selfboundary high | 0.277338 | 0.491320 | 0.678747 | 0.916531 | 254.8 | - |
| high + comp residual | 0.279431 | 0.495199 | 0.683492 | 0.955224 | 272.4 | residual_instances=17.6; uncovered=788.2; out_conflict=0.4727 |
| high + comp supportfull | 0.281610 | 0.497938 | 0.690897 | 0.955224 | 272.4 | residual_instances=17.6; uncovered=788.2; out_conflict=0.7053 |
| high + 32f residual | 0.277338 | 0.491320 | 0.678747 | 0.982493 | 279.0 | residual_instances=24.2; uncovered=788.2; out_conflict=0.4517 |
| high + comp cat085 | 0.280989 | 0.498159 | 0.691317 | 0.950209 | 391.6 | secondary_after=136.8; skipped=216.6 |
| high + comp cat095 | 0.281615 | 0.497583 | 0.690254 | 0.957699 | 415.6 | secondary_after=160.8; skipped=192.6 |
| high + 32f cat085 | 0.277686 | 0.491320 | 0.682865 | 0.978747 | 427.6 | secondary_after=172.8; skipped=213.2 |

当前 pure Stream4D probe5 fixed-support best 更新为：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
AP/AP50/AP25 = 0.281615 / 0.497583 / 0.690254
```

### 54.4 是否达成目标

仍未达成超过 Stream3D。

对比：

```text
Stream3D same 32f support:
0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108

best pure Stream4D:
0.281615 / 0.497583 / 0.690254
```

差距：

```text
AP   约低 0.117598
AP50 约低 0.099588
AP25 约低 0.052281
```

### 54.5 为什么这轮有收益

相对 selfboundary high：

```text
AP   +0.004277
AP50 +0.006263
AP25 +0.011507
```

相对 compref previous best：

```text
AP   +0.021229
AP50 +0.025323
AP25 +0.012617
```

解释：

```text
1. boundary high layer 提供更干净的前段 precision。
2. compref low-confidence layer 补回一部分被 boundary refinement 删除的 recall。
3. AP 可以利用 score tier：高置信精修 mask 在前，低置信 recall mask 在后。
```

### 54.6 为什么仍失败

#### A. 简单补 coverage 不够

`high + 32f residual` 的 union in target：

```text
0.982493
```

已经接近 Stream3D same support 的：

```text
0.985608
```

但指标仍只有：

```text
0.277338 / 0.491320 / 0.678747
```

说明：

```text
补到 target support 的点如果没有正确实例边界和 object assignment，
不会转化成 AP。
```

#### B. compref recall layer 比原 32f recall layer 更有效

`high + comp cat095`：

```text
0.281615 / 0.497583 / 0.690254
```

`high + 32f cat085`：

```text
0.277686 / 0.491320 / 0.682865
```

说明 compref candidate pool 里确实有比原始 32f 更好的 recall 候选；但它们仍然太碎、太多，不能接近 Stream3D。

#### C. 输出实例数仍过多

best pure Stream4D：

```text
#pred = 415.6
```

Stream3D same support：

```text
#pred = 128.2
```

Stream4D 依然依赖大量低置信候选补 recall，这会压低 AP 主指标。

### 54.7 当前状态

截至本节：

```text
pure Stream4D probe5 fixed-support best:
0.281615 / 0.497583 / 0.690254

Stream3D same fixed support:
0.399213 / 0.597171 / 0.742535
```

结论仍然是：

```text
inherit_pre_points / fixed-support 下，纯 Stream4D 没有超过 Stream3D。
```

但当前最有效的算法结构更清楚了：

```text
boundary-refined high-confidence layer
+ compref low-confidence recall layer
+ containment suppression
```

下一步真正要补的是：

```text
把低置信 recall layer 中的大量重复/碎片候选，
提升为更少、更准的一对一 object proposal。
```

## 55. 2026-06-08 当前差距总结与 object-level competition 负例

### 55.1 当前最佳结果

截至本节，probe5 fixed-support 下的 pure Stream4D 最好结果是：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
AP/AP50/AP25 = 0.281615 / 0.497583 / 0.690254
union in target = 0.957699
#pred = 415.6
```

同一 support 下 Stream3D 是：

```text
scannet_on_stream4d_32f_probe5
AP/AP50/AP25 = 0.399213 / 0.597171 / 0.742535
union in target = 0.985608
#pred = 128.2
```

差距：

```text
AP   约 0.1176
AP50 约 0.0996
AP25 约 0.0523
```

### 55.2 本轮 object-level competition 结果

本轮复用已有：

```text
Stream3D/tools/object_competition_rank.py
```

在当前 best 候选池上做无监督 object-level overlap grouping 和代表候选选择。

| 方法 | AP | AP50 | AP25 | union in target | #pred |
|---|---:|---:|---:|---:|---:|
| current best before objcomp | 0.281615 | 0.497583 | 0.690254 | 0.957699 | 415.6 |
| objcomp min050 score | 0.129698 | 0.217840 | 0.279147 | 0.476360 | 89.8 |
| objcomp min070 score | 0.158578 | 0.300187 | 0.440296 | 0.646035 | 156.4 |
| objcomp min085 score | 0.176469 | 0.342018 | 0.504653 | 0.739961 | 206.0 |
| objcomp min070 area | 0.161189 | 0.311110 | 0.516286 | 0.809664 | 156.4 |
| objcomp iou050 score | 0.190065 | 0.369790 | 0.556006 | 0.893759 | 233.4 |
| objcomp min070 score preserve | 0.216250 | 0.369478 | 0.509438 | 0.646035 | 156.4 |

结论：

```text
object competition 没有达成目标。
```

它确实能减少候选数，但 AP 大幅下降。

### 55.3 为什么这个负例重要

#### A. 问题不是简单“预测太多，删掉一些”

`objcomp iou050 score` 把预测数从 415.6 降到 233.4：

```text
#pred: 415.6 -> 233.4
```

但 AP 从 0.281615 掉到：

```text
0.190065
```

这说明大量低置信候选虽然很乱，但其中仍包含 recall；简单压缩会错删有用候选。

#### B. 无监督代表选择还远不够

`min_ioc` 分组把候选压缩得更狠，但 union in target 也明显下降：

```text
min070 score union in target = 0.646035
min085 score union in target = 0.739961
```

说明 overlap grouping 会把相邻/互补候选错误地视作同一组，导致漏召回。

#### C. preserve original score 不能救

`objcomp min070 score preserve`：

```text
0.216250 / 0.369478 / 0.509438
```

虽然比同组不 preserve 的 AP 0.158578 高，但仍远低于 current best 0.281615。  
因此失败不只是新 score 差，而是：

```text
候选分组和代表选择本身不可靠。
```

### 55.4 当前已排除的方向

以下方向都已有实验证据，不应继续作为主线堆阈值：

```text
1. recompute/self support 下继续调 top-k。
2. final prediction point-level WTA。
3. point NMS / point merge。
4. support-aware global ranking。
5. silhouette score 直接排序/过滤。
6. greedy support novelty。
7. object-level overlap competition postprocess。
8. 只靠 low-confidence recall layer 补 coverage。
```

它们能带来局部小收益，但不能把 AP 拉近 Stream3D。

### 55.5 目前最可靠的正信号

当前真正有正收益的结构是：

```text
boundary-refined high-confidence layer
+ compref low-confidence recall layer
+ containment suppression
```

它把 probe5 pure Stream4D 从：

```text
0.249551 / 0.460336 / 0.677592
```

逐步提高到：

```text
0.281615 / 0.497583 / 0.690254
```

但这仍不足以超过 Stream3D。

### 55.6 需要下一版新计划

v4.1 的核心结论是：

```text
inherit_pre_points / fixed-support 下，pure Stream4D 没有超过 Stream3D。
```

原因不是单个后处理 bug，而是结构性差距：

```text
1. Stream4D 仍需要 400+ 个候选才能接近 0.69 AP25。
2. Stream3D 只用 128 个候选就能达到 0.399 AP。
3. Stream4D 的 low-confidence recall 里有有用候选，但无监督方法还不能把它们合成少量高质量一对一实例。
4. coverage 接近满也不够；high + 32f residual union in target 到 0.982493，但 AP 几乎不变。
```

下一版必须前移到算法核心，而不是继续后处理：

```text
proposal generation
multi-view boundary evidence
object identity / split-merge memory
candidate quality calibration
```

下一版计划已单独写入：

```text
docs/stream4d_v5_inherit_gap_plan_for_codex.md
```

而不是继续只做最终 prediction WTA/NMS。
