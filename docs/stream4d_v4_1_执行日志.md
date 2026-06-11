# Stream4D v4.1 ScanNet 执行日志

## 0. 基本信息

- 工作目录：`/mnt/data/users/chengshun.wang/pjs/LoGeR`
- 计划文档：`docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md`
- 代码目录：`Stream3D`
- 数据目录：`/mnt/data/users/chengshun.wang/pjs/sray_plus/data/scannet`
- Conda 环境：`loger`
- 环境 Python：`/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`
- 开始日期：2026-06-07

## 1. 本轮目标

本轮按照 v4.1 计划先完成第一批硬性任务：

1. 新增指标完整性守卫工具，检查 evaluator 核心函数、rescore 是否读取 GT、prediction/TMP/object_dict 对齐关系。
2. 新增 Stream3D inherit / cross-fixed pre_points 诊断工具，补齐 Stream3D baseline 在 Stream4D pre_points 下的可审计结果。
3. 修改 `rescore_scannet.py`，增加 object_dict 与 prediction mask 列的强一致性检查，并补 fixed_path 参数校验。
4. 修改 reexport/export 相关诊断命名，避免把 `point_dilate` 或 `reuse_point_ids` 误写成 `mask_backproject` / nearest-neighbor hit rate。
5. 运行 S0 复现实验和审计；所有实际指标必须来自命令输出或结果文件，不允许手填或猜测。

## 2. 已执行命令

### 2.1 阅读计划文档

```bash
sed -n '1,260p' docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
sed -n '260,520p' docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
sed -n '520,860p' docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
sed -n '860,1220p' docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
sed -n '1220,1580p' docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
sed -n '1580,1940p' docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
sed -n '1940,2120p' docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
wc -l docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md
```

结果：计划文档共 2071 行。已经读到 Task 1 至 Task 7，以及最终 claim 安全边界。

### 2.2 阅读现有相关代码

```bash
sed -n '1,220p' Stream3D/stream4d/rescore_scannet.py
sed -n '220,520p' Stream3D/stream4d/rescore_scannet.py
sed -n '1,260p' Stream3D/tools/audit_stream3d_eval_protocol.py
sed -n '260,620p' Stream3D/tools/audit_stream3d_eval_protocol.py
sed -n '1,280p' Stream3D/stream4d/reexport_scannet.py
```

初步结论：

- `rescore_scannet.py` 已有 `recompute` / `inherit` / `fixed_path` 三种 pre_points policy，但 `fixed_path` 缺少参数必填校验。
- `rescore_scannet.py` 已检查 object 数量等于 prediction mask 列数，但还没有检查每一列 mask 与 object_dict 中 `point_ids` 是否逐点一致。
- `reexport_scannet.py` 在创建 `ScanNetExporter` 时始终写入 `export_support_mode="mask_backproject"`，这会让 `point_dilate` 的诊断名称误导审阅者。
- `audit_stream3d_eval_protocol.py` 已能比较 evaluator 核心函数 hash 和统计 pre_points/union/GT crop，但还不是 v4.1 计划要求的完整守卫。

## 3. 修改记录

### 3.1 `Stream3D/stream4d/rescore_scannet.py`

修改内容：

- 新增 `verify_object_dict_prediction_alignment(pred_masks, object_items, threshold, include_records)`。
- 对每个 object_dict 条目的 `point_ids` 和 prediction mask 的对应列做点级对齐检查。
- 记录每个实例的 `point_iou`、`point_ioc`、object_dict 点数、prediction 列点数、intersection、越界 point id 数。
- 默认阈值为 `alignment_iou_threshold=0.99`。
- 如果可验证且有实例低于阈值，rescore 直接报错停止，不输出新的 AP。
- summary 新增：
  - `alignment_checked`
  - `cannot_verify_alignment`
  - `alignment_num_checked`
  - `alignment_mean_iou`
  - `alignment_min_iou`
  - `alignment_failed_instances`
  - `alignment_iou_threshold`
- 新增 `validate_args(args)`，当 `--pre-points-policy fixed_path` 但没有 `--fixed-pre-points-config` 时直接报错：

```text
ValueError("--fixed-pre-points-config is required for fixed_path")
```

审计理由：避免 object_dict 顺序和 prediction 列错位，也避免 fixed_path 空参数误读 `data/TMP/<scene>_pre_points.npy` 这种错误路径。

### 3.2 `Stream3D/stream4d/export_scannet.py`

修改内容：

- `export_support_mode` 新增允许：
  - `reuse_point_ids`
  - `point_dilate`
- 对 `reuse_point_ids` / `point_dilate` 不再写有误导性的 nearest-neighbor hit rate。
- 新增诊断字段：
  - `reuse_point_count`
  - `reuse_point_after_dilation_count`
  - `export_reuse_point_expansion_rate`
- 对 point reuse 类导出，`export_nn_hit_rate = None`，表示不适用。

审计理由：point reuse / point dilation 不是通过 2D/3D nearest-neighbor 查询命中得到的，不能把 reuse 数量伪装成 NN hit rate。

### 3.3 `Stream3D/stream4d/reexport_scannet.py`

修改内容：

- `--reexport-mode mask_backproject` 时写 `export_support_mode="mask_backproject"`。
- `--reexport-mode point_dilate` 且 `--export-point-dilate-radius > 0` 时写 `export_support_mode="point_dilate"`。
- `--reexport-mode point_dilate` 且 dilation 半径为 0 时写 `export_support_mode="reuse_point_ids"`。
- 打印日志时，如果 `export_nn_hit_rate is None`，显示 `hit_rate=NA`。

### 3.4 `Stream3D/tools/evaluate_cross_prepoints.py`

新增工具，用于 Stream3D inherit / cross-fixed pre_points 诊断。

它做的事情：

1. 读取 source prediction，例如 `data/prediction/scannet_class_agnostic/<scene>.npz`。
2. 读取 source pre_points，用于判断 prediction mask 是 full scene 还是 cropped mask。
3. 读取 target pre_points，例如 Stream4D MVP 或 adaptive 的 pre_points。
4. 如果 prediction mask 是 full scene，输出 prediction 文件用符号链接指向原文件，不复制大 npz。
5. 如果 prediction mask 是 source pre_points cropped shape，则扩展回 full scene 再写出。
6. 输出 `data/TMP/<output_config>/<scene>_pre_points.npy`，符号链接指向 target pre_points。
7. 调用原 evaluator：

```bash
python -m evaluation.evaluate \
  --pred_path data/prediction/<output_config>_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --tmp_root data/TMP \
  --tmp_config <output_config> \
  --no_class
```

8. 写 per-config JSON 和汇总：
  - `outputs/audit/cross_prepoints/<output_config>_summary.json`
  - `outputs/audit/cross_prepoints_audit.md`
  - `outputs/audit/cross_prepoints_audit.json`
  - `outputs/audit/cross_prepoints_audit.csv`

关键审计字段：

- `source_pred_config`
- `source_pre_points_config`
- `target_pre_points_config`
- `mask_shape_mode`
- `expanded_prediction`
- `source_pre_points_ratio`
- `target_pre_points_ratio`
- `prediction_union_ratio`
- `prediction_union_in_target_ratio_of_scene`
- `prediction_union_in_target_ratio_of_target`
- `num_gt_instances_in_target_pre_points`
- `num_gt_instances_fullmesh`
- `num_pred_instances`

### 3.5 `Stream3D/tools/verify_stream4d_metric_integrity.py`

新增指标完整性守卫工具。

它检查：

1. 当前 evaluator 与原版 Stream3D evaluator 的 AP 核心函数 hash 是否一致。
2. 原版和当前 evaluator 是否都读取 `_pre_points.npy`。
3. `stream4d/rescore_scannet.py` 是否包含 `gt_path`、`data/scannet/gt`、`np.loadtxt` 等 GT 读取痕迹。
4. 每个 config 的 prediction/TMP/GT 是否能逐场景对齐。
5. pre_points 与 prediction union 的关系。
6. GT crop/full 数量。
7. object_dict 与 prediction mask 列的点级 IoU 对齐。

输出：

- `Stream3D/outputs/audit/stream4d_v4_metric_integrity.md`
- `Stream3D/outputs/audit/stream4d_v4_metric_integrity.json`
- `Stream3D/outputs/audit/pre_points_ratio_by_config.png`
- `Stream3D/outputs/audit/union_ratio_by_config.png`
- `Stream3D/outputs/audit/gt_crop_full_by_config.png`
- `Stream3D/outputs/audit/object_dict_alignment_iou_hist.png`

### 3.6 `Stream3D/tools/check_dynamic_replica_env.py`

新增 Dynamic Replica 环境检查工具。

它只检查数据条件，不报告动态 tracking 指标：

- data root 是否存在。
- split dir 是否存在。
- `frame_annotations_<split>.json` 是否存在。
- camera 字段 `R/T/focal_length/principal_point` 是否存在。
- 每个 scene 的 images/depths/trajectories 数量。
- 是否有 semantic GT、instance GT、object IDs。
- 根据数据条件判断后续最多能报告 official instance tracking、D4RT trajectory metrics，还是只能报告 qualitative/pseudo consistency。

### 3.7 `Stream3D/tests/test_stream4d_protocol_fixes.py`

新增 2 个测试：

- `test_rescore_alignment_detects_column_mismatch`
- `test_fixed_path_requires_config_name`

## 4. 实验命令

### 4.1 语法检查和单元测试

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  stream4d/rescore_scannet.py \
  stream4d/export_scannet.py \
  stream4d/reexport_scannet.py \
  tools/evaluate_cross_prepoints.py \
  tools/verify_stream4d_metric_integrity.py \
  tools/check_dynamic_replica_env.py \
  tests/test_stream4d_protocol_fixes.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 4 tests in 0.001s
OK
```

### 4.2 Stream3D self-inherit 对照

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet.txt \
  --pred-config scannet \
  --pred-suffix _class_agnostic \
  --pre-points-config scannet \
  --output-config scannet_self_inherit \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/scannet_self_inherit_class_agnostic.txt \
  2>&1 | tee logs/stream4d_v4_1_cross_scannet_self_inherit.log
```

结果文件：

- `Stream3D/data/evaluation/scannet/scannet_self_inherit_class_agnostic.txt`
- `Stream3D/outputs/audit/cross_prepoints/scannet_self_inherit_summary.json`

### 4.3 Stream3D on Stream4D MVP pre_points

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet.txt \
  --pred-config scannet \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config scannet_on_stream4d_mvp_prepoints \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/scannet_on_stream4d_mvp_prepoints_class_agnostic.txt \
  2>&1 | tee logs/stream4d_v4_1_cross_scannet_on_stream4d_mvp_prepoints.log
```

结果文件：

- `Stream3D/data/evaluation/scannet/scannet_on_stream4d_mvp_prepoints_class_agnostic.txt`
- `Stream3D/outputs/audit/cross_prepoints/scannet_on_stream4d_mvp_prepoints_summary.json`

### 4.4 Stream3D on Stream4D adaptive pre_points

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet.txt \
  --pred-config scannet \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config scannet_on_stream4d_adaptive_prepoints \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/scannet_on_stream4d_adaptive_prepoints_class_agnostic.txt \
  2>&1 | tee logs/stream4d_v4_1_cross_scannet_on_stream4d_adaptive_prepoints.log
```

结果文件：

- `Stream3D/data/evaluation/scannet/scannet_on_stream4d_adaptive_prepoints_class_agnostic.txt`
- `Stream3D/outputs/audit/cross_prepoints/scannet_on_stream4d_adaptive_prepoints_summary.json`

### 4.5 Stream4D adaptive on Stream3D pre_points

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet.txt \
  --pred-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --pred-suffix _class_agnostic \
  --pre-points-config scannet \
  --output-config stream4d_adaptive_on_scannet_prepoints \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_adaptive_on_scannet_prepoints_class_agnostic.txt \
  2>&1 | tee logs/stream4d_v4_1_cross_stream4d_adaptive_on_scannet_prepoints.log
```

结果文件：

- `Stream3D/data/evaluation/scannet/stream4d_adaptive_on_scannet_prepoints_class_agnostic.txt`
- `Stream3D/outputs/audit/cross_prepoints/stream4d_adaptive_on_scannet_prepoints_summary.json`

### 4.6 指标完整性审计

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet.txt \
  --configs scannet,stream4d_scannet_32f_ioc075_fixmem,stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute,stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit,scannet_self_inherit,scannet_on_stream4d_mvp_prepoints,scannet_on_stream4d_adaptive_prepoints,stream4d_adaptive_on_scannet_prepoints \
  --output outputs/audit/stream4d_v4_metric_integrity.md \
  2>&1 | tee logs/stream4d_v4_1_metric_integrity.log
```

结果：

```text
[metric-integrity] phase0_pass=True
```

### 4.7 Dynamic Replica 环境检查

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.check_dynamic_replica_env \
  --data-root data/dynamic-replica/v2 \
  --split valid \
  --output outputs/audit/dynamic_replica_env_check.md \
  2>&1 | tee logs/stream4d_v4_1_dynamic_replica_env_check.log
```

结果：

```text
root_exists=False split_exists=False usable_scenes=0
```

## 5. 结果文件

核心结果文件：

- `Stream3D/data/evaluation/scannet/scannet_self_inherit_class_agnostic.txt`
- `Stream3D/data/evaluation/scannet/scannet_on_stream4d_mvp_prepoints_class_agnostic.txt`
- `Stream3D/data/evaluation/scannet/scannet_on_stream4d_adaptive_prepoints_class_agnostic.txt`
- `Stream3D/data/evaluation/scannet/stream4d_adaptive_on_scannet_prepoints_class_agnostic.txt`
- `Stream3D/outputs/audit/cross_prepoints_audit.md`
- `Stream3D/outputs/audit/cross_prepoints_audit.json`
- `Stream3D/outputs/audit/cross_prepoints_audit.csv`
- `Stream3D/outputs/audit/stream4d_v4_metric_integrity.md`
- `Stream3D/outputs/audit/stream4d_v4_metric_integrity.json`
- `Stream3D/outputs/audit/dynamic_replica_env_check.md`
- `Stream3D/outputs/audit/dynamic_replica_env_check.json`

可视化：

- `Stream3D/outputs/audit/pre_points_ratio_by_config.png`
- `Stream3D/outputs/audit/union_ratio_by_config.png`
- `Stream3D/outputs/audit/gt_crop_full_by_config.png`
- `Stream3D/outputs/audit/object_dict_alignment_iou_hist.png`

命令日志：

- `Stream3D/logs/stream4d_v4_1_cross_scannet_self_inherit.log`
- `Stream3D/logs/stream4d_v4_1_cross_scannet_on_stream4d_mvp_prepoints.log`
- `Stream3D/logs/stream4d_v4_1_cross_scannet_on_stream4d_adaptive_prepoints.log`
- `Stream3D/logs/stream4d_v4_1_cross_stream4d_adaptive_on_scannet_prepoints.log`
- `Stream3D/logs/stream4d_v4_1_metric_integrity.log`
- `Stream3D/logs/stream4d_v4_1_dynamic_replica_env_check.log`

## 6. Blocker 与处理

### 6.1 evidence quality v1 没有作为完整方法实验运行

已做：

```bash
sed -n '1,320p' Stream3D/stream4d/run_scannet.py
sed -n '1,360p' Stream3D/stream4d/mask_evidence.py
sed -n '1,260p' Stream3D/stream4d/carrier_store.py
sed -n '1,340p' Stream3D/stream4d/d4rt_adapter.py
```

发现：

- 当前 `CarrierBatch` 只有 `uv_pred`、`visibility_prob`、`confidence_prob`、`valid`。
- 当前 `MaskEvidenceBuilder` 实际实现的是 `rho = visibility_prob * confidence_prob`。
- v4.1 计划中的 `rho_self`、`rho_self_boundary`、`rho_self_cycle_boundary` 需要 self/cycle/reverse tracking evidence。
- 当前 `D4RTAdapter.infer_carriers()` 没有输出 reverse query 或 cycle error。

处理结论：

- 本轮不能诚实地声称已经实现 self/cycle evidence。
- 后续修复方向：扩展 `D4RTAdapter`，增加 target-to-source reverse query，记录 `self_uv_error`、`cycle_uv_error`，再接入 `MaskEvidenceBuilder`。

### 6.2 reliable densifier v1 没有作为完整方法实验运行

已查到：

- 当前 `export_scannet.py` 支持 `carrier_uv`、`mask_backproject`、`hybrid`，以及本轮补充诊断命名的 `reuse_point_ids` / `point_dilate`。
- v4.1 计划中的 seeded component、boundary erosion、seed distance cap、winner-take-all conflict suppression 需要基于每个 2D mask 的像素连通域和 carrier seed 分布来生成新的导出点。

处理结论：

- 本轮没有把简单 dilation 冒充 reliable densification。
- 后续修复方向：新增 `stream4d/reliable_densifier.py`，先在 30 tune scenes 上单独输出 `mask_pixels_total`、`mask_pixels_kept`、`boundary_removed_ratio`、`seed_distance_removed_ratio`、`conflict_removed_ratio`，通过诊断后再接入 exporter。

### 6.3 memory-v2 没有作为完整方法实验运行

已查到：

- 当前 `ObjectMemory4D` 是 carrier overlap 的贪心匹配。
- 没有 appearance feature、motion feature、Hungarian one-to-one matching、lost/reactivated timeline 的完整实现。

处理结论：

- 本轮不能声称完成 ObjectMemory4D-v2。
- 后续修复方向：新增 `appearance_memory.py` 的 RGB histogram feature，新增 `object_memory_v2.py` 做 Hungarian matching，再只在少量 scenes 上跑 96f/128f 小实验。

### 6.4 Dynamic Replica 数据缺失

已运行：

```bash
python -m tools.check_dynamic_replica_env --data-root data/dynamic-replica/v2 --split valid
```

结果：

```text
data_root_exists=False
split_dir_exists=False
usable_scene_count=0
```

处理结论：

- 本轮不能报告 Dynamic Replica official tracking / semantic AP / pseudo consistency。
- 后续需要先准备 `data/dynamic-replica/v2/valid` 数据和 `frame_annotations_valid.json`。

## 7. 继续推进：reliable densification v1、seed 保留模式与 final split

日期：2026-06-08。

### 7.1 新增 / 修改代码

本轮继续按计划里的 Phase 3 推荐方向实现和调试 reliable densification。

修改文件：

- `Stream3D/stream4d/reliable_densifier.py`
  - 新增 seeded connected component：每个 object 的原始 `point_ids` 投影到 2D mask，保留包含 seed pixel 的连通区域。
  - 新增 boundary erosion：对足够大的 mask 去掉边界附近像素。
  - 新增 seed distance cap：只保留距离 seed pixel 不超过指定像素半径的区域。
  - 新增 3D nearest-neighbor backprojection：用 ScanNet depth/pose/intrinsics 将保留像素回投到 mesh 顶点。
  - 新增 WTA conflict suppression：多个 object 命中同一 3D point 时，只保留 reliability 更高的 object。
  - 新增 `seed_keep_mode = none | supported | all`：
    - `none`：只使用可靠 2D 区域回投出来的新点。
    - `supported`：额外保留能投影回可靠 2D 区域的旧 seed 点。
    - `all`：额外保留 object_dict 原始所有 seed 点。
- `Stream3D/stream4d/export_scannet.py`
  - 新增 `export_support_mode="reliable_densify"`。
  - 新增 reliable densifier 参数透传。
  - 将 densification 的像素数量、hit rate、WTA pre-conflict rate、seed keep mode 等写入 summary。
- `Stream3D/stream4d/reexport_scannet.py`
  - 新增 `--reexport-mode reliable_densify`。
  - 新增 `--densify-boundary-erosion`、`--densify-small-mask-area`、`--densify-seed-distance-px`、`--densify-min-seed-pixels`、`--densify-seed-keep-mode`、`--disable-densify-wta`。
- `Stream3D/tests/test_stream4d_protocol_fixes.py`
  - 新增 `test_reliable_densifier_wta_keeps_highest_reliability_owner`，检查 WTA 会把冲突点分给 reliability 更高的 object。
- `Stream3D/splits/scannet_tune30.txt`
  - 从 `splits/scannet_tune.txt` 取前 30 个场景作为快速调参子集。

### 7.2 语法检查和单元测试

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  stream4d/reliable_densifier.py \
  stream4d/export_scannet.py \
  stream4d/reexport_scannet.py \
  stream4d/rescore_scannet.py \
  tools/evaluate_cross_prepoints.py \
  tools/verify_stream4d_metric_integrity.py \
  tests/test_stream4d_protocol_fixes.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 5 tests in 0.001s
OK
```

### 7.3 tune30 baseline materialization 和评估

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.materialize_scannet_eval_subset \
  --root . \
  --config scannet \
  --seq-list splits/scannet_tune30.txt \
  --output-config scannet_v4_1_tune30

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.materialize_scannet_eval_subset \
  --root . \
  --config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --seq-list splits/scannet_tune30.txt \
  --output-config stream4d_v3_adaptive_recompute_v4_1_tune30

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/scannet_v4_1_tune30_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/scannet_v4_1_tune30_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config scannet_v4_1_tune30 \
  --no_class

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v3_adaptive_recompute_v4_1_tune30_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v3_adaptive_recompute_v4_1_tune30_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v3_adaptive_recompute_v4_1_tune30 \
  --no_class
```

结果：

```text
scannet_v4_1_tune30:                         21.6016 / 36.3863 / 51.9451
stream4d_v3_adaptive_recompute_v4_1_tune30:  23.0107 / 38.9224 / 58.0834
```

### 7.4 reliable densification tune30 命令

保守 no-seed 版本，后续 final locked 使用这一组参数：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_tune30.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_e1_d32_top5_tune30 \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 5 \
  --export-min-points-per-object 100 \
  --export-score-mode one \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 32 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode none \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error \
  2>&1 | tee logs/stream4d_v4_1_reliable_e1_d32_top5_tune30_reexport.log

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v4_1_reliable_e1_d32_top5_tune30_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_tune30_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v4_1_reliable_e1_d32_top5_tune30 \
  --no_class \
  2>&1 | tee logs/stream4d_v4_1_reliable_e1_d32_top5_tune30_eval.log
```

更大覆盖版本：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_tune30.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_e1_d64_top10_tune30 \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 10 \
  --export-min-points-per-object 100 \
  --export-score-mode one \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 64 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode none \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_tune30.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_e2_d48_top10_tune30 \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 10 \
  --export-min-points-per-object 100 \
  --export-score-mode one \
  --densify-boundary-erosion 2 \
  --densify-seed-distance-px 48 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode none \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error
```

seed 保留与 min-points 诊断：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_tune30.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_seed_supported_e1_d32_top5_tune30 \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 5 \
  --export-min-points-per-object 100 \
  --export-score-mode one \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 32 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode supported \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_tune30.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_seednone_e1_d32_top5_min20_tune30 \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 5 \
  --export-min-points-per-object 20 \
  --export-score-mode one \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 32 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode none \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_tune30.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_tune30 \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 5 \
  --export-min-points-per-object 20 \
  --export-score-mode one \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 32 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode all \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error
```

这些配置均用 `evaluation.evaluate` 评估 recompute，并用 `tools.evaluate_cross_prepoints` 评估 MVP / adaptive / scannet fixed support。完整输出在：

- `Stream3D/logs/stream4d_v4_1_reliable_*_tune30_*.log`
- `Stream3D/logs/stream4d_v4_1_cross_reliable_*_tune30.log`
- `Stream3D/data/evaluation/scannet/*tune30*_class_agnostic.txt`
- `Stream3D/outputs/stream4d_reexport_v4_1/*tune30_summary.json`
- `Stream3D/outputs/audit/cross_prepoints/*tune30_summary.json`

### 7.5 MVP object_dict reliable densify 诊断

为了验证“adaptive top-k 是否筛掉太多对象”，用 MVP object_dict 做 reliable densify：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_tune30.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config stream4d_v4_1_mvp_reliable_e2_d32_top3_min250_tune30 \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 8000 \
  --export-max-masks-per-object 3 \
  --export-min-points-per-object 250 \
  --export-score-mode one \
  --densify-boundary-erosion 2 \
  --densify-seed-distance-px 32 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error \
  2>&1 | tee logs/stream4d_v4_1_mvp_reliable_e2_d32_top3_min250_tune30_reexport.log

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v4_1_mvp_reliable_e2_d32_top3_min250_tune30_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v4_1_mvp_reliable_e2_d32_top3_min250_tune30_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v4_1_mvp_reliable_e2_d32_top3_min250_tune30 \
  --no_class \
  2>&1 | tee logs/stream4d_v4_1_mvp_reliable_e2_d32_top3_min250_tune30_eval.log
```

### 7.6 final split locked run

锁定参数：`seed_keep_mode=none`、`boundary_erosion=1`、`seed_distance_px=32`、`max_masks_per_object=5`、`export_min_points_per_object=100`。

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_final.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_e1_d32_top5_final \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 5 \
  --export-min-points-per-object 100 \
  --export-score-mode one \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 32 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode none \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error \
  2>&1 | tee logs/stream4d_v4_1_reliable_e1_d32_top5_final_reexport.log

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v4_1_reliable_e1_d32_top5_final_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_final_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v4_1_reliable_e1_d32_top5_final \
  --no_class \
  2>&1 | tee logs/stream4d_v4_1_reliable_e1_d32_top5_final_eval.log
```

final fixed support 诊断：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_final.txt \
  --pred-config stream4d_v4_1_reliable_e1_d32_top5_final \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config stream4d_v4_1_reliable_e1_d32_top5_on_mvp_final \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_on_mvp_final_class_agnostic.txt \
  2>&1 | tee logs/stream4d_v4_1_cross_reliable_e1_d32_top5_on_mvp_final.log

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_final.txt \
  --pred-config stream4d_v4_1_reliable_e1_d32_top5_final \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_e1_d32_top5_on_adaptive_final \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_on_adaptive_final_class_agnostic.txt \
  2>&1 | tee logs/stream4d_v4_1_cross_reliable_e1_d32_top5_on_adaptive_final.log

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_final.txt \
  --pred-config stream4d_v4_1_reliable_e1_d32_top5_final \
  --pred-suffix _class_agnostic \
  --pre-points-config scannet \
  --output-config stream4d_v4_1_reliable_e1_d32_top5_on_scannet_final \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_on_scannet_final_class_agnostic.txt \
  2>&1 | tee logs/stream4d_v4_1_cross_reliable_e1_d32_top5_on_scannet_final.log
```

结果文件：

- `Stream3D/data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_final_class_agnostic.txt`
- `Stream3D/data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_on_mvp_final_class_agnostic.txt`
- `Stream3D/data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_on_adaptive_final_class_agnostic.txt`
- `Stream3D/data/evaluation/scannet/stream4d_v4_1_reliable_e1_d32_top5_on_scannet_final_class_agnostic.txt`
- `Stream3D/outputs/stream4d_reexport_v4_1/stream4d_v4_1_reliable_e1_d32_top5_final_summary.json`
- `Stream3D/outputs/audit/cross_prepoints/stream4d_v4_1_reliable_e1_d32_top5_on_mvp_final_summary.json`
- `Stream3D/outputs/audit/cross_prepoints/stream4d_v4_1_reliable_e1_d32_top5_on_adaptive_final_summary.json`
- `Stream3D/outputs/audit/cross_prepoints/stream4d_v4_1_reliable_e1_d32_top5_on_scannet_final_summary.json`

### 7.7 本轮没有重跑的检查

本轮最后尝试重跑 metric integrity guard 时，没有在当前机器找到之前审计用的原版 evaluator 路径：

```bash
ls /mnt/data/orig_stream3d/Code_Stream3D/evaluation/evaluate.py /mnt/data/audit_v3/Stream3D/evaluation/evaluate.py
```

输出为空。因此本轮只重跑了本地 py_compile 和 unittest，没有编造新的 evaluator hash 结果。上一轮已经记录过的 `phase0_pass=True` 和 AP core hash 结论仍保留在原复盘中，但本节 final split 没有新增 hash 审计结果。

## 8. 继续推进：选择式 coverage 融合实验

日期：2026-06-08。

### 8.1 继续推进原因

第 7 节结果显示：

- no-seed reliable densification 的 recompute 很强，但 inherit/fixed support 低。
- seed-all 能提高 coverage 和部分 fixed AP，但 recompute 明显下降。

因此继续尝试一个更保守的方案：不额外叠加重复 mask，而是在每个 object 上选择 clean mask 或 coverage mask。

### 8.2 新增代码

新增：

- `Stream3D/tools/fuse_prediction_configs.py`

功能：

1. `fusion-mode=concatenate`：把 primary prediction 和 secondary prediction 直接拼接，secondary 用低分。
2. `fusion-mode=select_secondary`：对每个 secondary mask 找最相近 primary mask，若 secondary 覆盖 primary 的比例足够高且扩张倍数不超过阈值，则用 secondary 替换 primary。
3. 输出：
   - `data/prediction/<output_config>_class_agnostic/*.npz`
   - `data/TMP/<output_config>/*_pre_points.npy`
   - `outputs/stream4d_fusion_v4_1/<output_config>_summary.json`

### 8.3 语法检查和单元测试

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/fuse_prediction_configs.py \
  stream4d/reliable_densifier.py \
  stream4d/export_scannet.py \
  stream4d/reexport_scannet.py \
  tests/test_stream4d_protocol_fixes.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 5 tests in 0.001s
OK
```

### 8.4 失败对照：直接拼接 clean + seed-all

生成命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_tune30.txt \
  --primary-config stream4d_v4_1_reliable_e1_d32_top5_tune30 \
  --secondary-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_tune30 \
  --output-config stream4d_v4_1_fuse_clean_seedall_tune30 \
  --primary-score 1.0 \
  --secondary-score 0.2 \
  --drop-secondary-iou-threshold 0.0 \
  2>&1 | tee logs/stream4d_v4_1_fuse_clean_seedall_tune30.log
```

评估命令示例：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v4_1_fuse_clean_seedall_tune30_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v4_1_fuse_clean_seedall_tune30_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v4_1_fuse_clean_seedall_tune30 \
  --no_class \
  2>&1 | tee logs/stream4d_v4_1_fuse_clean_seedall_tune30_eval.log

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_tune30.txt \
  --pred-config stream4d_v4_1_fuse_clean_seedall_tune30 \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config stream4d_v4_1_fuse_clean_seedall_on_mvp_tune30 \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_v4_1_fuse_clean_seedall_on_mvp_tune30_class_agnostic.txt \
  2>&1 | tee logs/stream4d_v4_1_cross_fuse_clean_seedall_on_mvp_tune30.log
```

还运行了 adaptive/scannet fixed support，对应日志：

- `logs/stream4d_v4_1_cross_fuse_clean_seedall_on_adaptive_tune30.log`
- `logs/stream4d_v4_1_cross_fuse_clean_seedall_on_scannet_tune30.log`

### 8.5 tune30：选择式融合

生成 r1.5 / r2.0 / r3.0：

```bash
for r in 1p5 2p0 3p0; do
  val=${r/p/.}
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
    --root . \
    --seq-list splits/scannet_tune30.txt \
    --primary-config stream4d_v4_1_reliable_e1_d32_top5_tune30 \
    --secondary-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_tune30 \
    --output-config stream4d_v4_1_select_seedall_r${r}_tune30 \
    --fusion-mode select_secondary \
    --select-min-primary-ioc 0.7 \
    --select-max-expansion ${val} \
    --primary-score 1.0 \
    2>&1 | tee logs/stream4d_v4_1_select_seedall_r${r}_tune30.log
done
```

每个 config 都用 `evaluation.evaluate` 跑 recompute，并用 `tools.evaluate_cross_prepoints` 跑 MVP/adaptive fixed。对应日志：

- `logs/stream4d_v4_1_select_seedall_r1p5_tune30_eval.log`
- `logs/stream4d_v4_1_cross_select_seedall_r1p5_on_mvp_tune30.log`
- `logs/stream4d_v4_1_cross_select_seedall_r1p5_on_adaptive_tune30.log`
- `logs/stream4d_v4_1_select_seedall_r2p0_tune30_eval.log`
- `logs/stream4d_v4_1_cross_select_seedall_r2p0_on_mvp_tune30.log`
- `logs/stream4d_v4_1_cross_select_seedall_r2p0_on_adaptive_tune30.log`
- `logs/stream4d_v4_1_select_seedall_r3p0_tune30_eval.log`
- `logs/stream4d_v4_1_cross_select_seedall_r3p0_on_mvp_tune30.log`
- `logs/stream4d_v4_1_cross_select_seedall_r3p0_on_adaptive_tune30.log`

unmatched-secondary 诊断：

```bash
for r in 1p5 2p0; do
  val=${r/p/.}
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
    --root . \
    --seq-list splits/scannet_tune30.txt \
    --primary-config stream4d_v4_1_reliable_e1_d32_top5_tune30 \
    --secondary-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_tune30 \
    --output-config stream4d_v4_1_select_seedall_r${r}_unmatched_tune30 \
    --fusion-mode select_secondary \
    --select-min-primary-ioc 0.7 \
    --select-max-expansion ${val} \
    --add-unmatched-secondary \
    --primary-score 1.0 \
    2>&1 | tee logs/stream4d_v4_1_select_seedall_r${r}_unmatched_tune30.log
done
```

### 8.6 final：选择式融合 r1.5 / r3.0 与 seed-all coverage 上限

先生成 final seed-all coverage 候选：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_final.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_final \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 5 \
  --export-min-points-per-object 20 \
  --export-score-mode one \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 32 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode all \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error \
  2>&1 | tee logs/stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_final_reexport.log
```

生成 final r1.5 / r3.0：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_final.txt \
  --primary-config stream4d_v4_1_reliable_e1_d32_top5_final \
  --secondary-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_final \
  --output-config stream4d_v4_1_select_seedall_r1p5_final \
  --fusion-mode select_secondary \
  --select-min-primary-ioc 0.7 \
  --select-max-expansion 1.5 \
  --primary-score 1.0 \
  2>&1 | tee logs/stream4d_v4_1_select_seedall_r1p5_final.log

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_final.txt \
  --primary-config stream4d_v4_1_reliable_e1_d32_top5_final \
  --secondary-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_final \
  --output-config stream4d_v4_1_select_seedall_r3p0_final \
  --fusion-mode select_secondary \
  --select-min-primary-ioc 0.7 \
  --select-max-expansion 3.0 \
  --primary-score 1.0 \
  2>&1 | tee logs/stream4d_v4_1_select_seedall_r3p0_final.log
```

评估命令模式：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/<config>_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/<config>_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config <config> \
  --no_class

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_final.txt \
  --pred-config <config> \
  --pred-suffix _class_agnostic \
  --pre-points-config <target_support_config> \
  --output-config <output_config> \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/<output_config>_class_agnostic.txt
```

实际日志：

- `logs/stream4d_v4_1_select_seedall_r1p5_final_eval.log`
- `logs/stream4d_v4_1_cross_select_seedall_r1p5_on_mvp_final.log`
- `logs/stream4d_v4_1_cross_select_seedall_r1p5_on_adaptive_final.log`
- `logs/stream4d_v4_1_cross_select_seedall_r1p5_on_scannet_final.log`
- `logs/stream4d_v4_1_select_seedall_r3p0_final_eval.log`
- `logs/stream4d_v4_1_cross_select_seedall_r3p0_on_mvp_final.log`
- `logs/stream4d_v4_1_cross_select_seedall_r3p0_on_adaptive_final.log`
- `logs/stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_final_eval.log`
- `logs/stream4d_v4_1_cross_reliable_seedall_e1_d32_top5_min20_on_mvp_final.log`
- `logs/stream4d_v4_1_cross_reliable_seedall_e1_d32_top5_min20_on_adaptive_final.log`

## 9. 继续推进：seed 可靠性过滤、score mode、低分 unmatched fusion、e2_d48 final

日期：2026-06-08。

### 9.1 继续推进原因

上一轮结论是：

- no-seed reliable densify 的 final recompute 最强，但 fixed support 低。
- seed-all 的 fixed support 最好，但 own recompute 低于 Stream3D final baseline。
- 选择式融合只替换 primary 时 fixed 改善有限；加入 unmatched secondary 时 recompute 明显下降。

因此本轮继续尝试三个非 GT、可审计方向：

1. 不无条件保留旧 seed 点，而是只保留能重新投影到 2D object 连通区域的旧点。
2. 不改变 mask，只改变 `pred_score`，看 seed-all 失败是否主要来自坏对象排序。
3. 修正 fusion 工具：unmatched secondary 应使用低分，而不是和 primary 一样高分。
4. 补跑已有 tune30 中较平衡的 `e2_d48_top10` final。

### 9.2 新增/修改代码

修改：

- `Stream3D/stream4d/reliable_densifier.py`
- `Stream3D/stream4d/export_scannet.py`
- `Stream3D/stream4d/reexport_scannet.py`
- `Stream3D/tools/fuse_prediction_configs.py`

具体改动：

1. `ReliableDensifyParams` 新增 `seed_min_support_views`。
2. `--densify-seed-keep-mode` 新增：
   - `boundary`：旧 seed 点投影后必须落在经过边界腐蚀的 object 连通区域内。
   - `component`：旧 seed 点投影后必须落在由 seed 命中的 object 连通区域内，不要求边界腐蚀后仍保留。
3. `--densify-seed-min-support-views` 控制旧 seed 点至少被多少个观测支持。
4. `--export-score-mode` 新增：
   - `reliability`
   - `observations`
   - `dense_quality`
5. `tools/fuse_prediction_configs.py` 修复 `select_secondary + --add-unmatched-secondary`：新增 unmatched secondary 现在使用 `--secondary-score`，不再错误地全部使用 primary 高分。

### 9.3 语法检查和单元测试

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/fuse_prediction_configs.py \
  stream4d/reliable_densifier.py \
  stream4d/export_scannet.py \
  stream4d/reexport_scannet.py \
  tests/test_stream4d_protocol_fixes.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 5 tests in 0.001s
OK
```

### 9.4 tune30：boundary/component seed 过滤

生成命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

for spec in boundary:1 boundary:2 component:1 component:2; do
  mode=${spec%:*}
  views=${spec#*:}
  cfg="stream4d_v4_1_reliable_seed${mode}_v${views}_e1_d32_top5_min20_tune30"
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
    --seq-list splits/scannet_tune30.txt \
    --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
    --output-config "$cfg" \
    --reexport-mode reliable_densify \
    --export-nn-radius 0.08 \
    --export-mask-sample-stride 2 \
    --export-mask-max-pixels 12000 \
    --export-max-masks-per-object 5 \
    --export-min-points-per-object 20 \
    --export-score-mode one \
    --densify-boundary-erosion 1 \
    --densify-seed-distance-px 32 \
    --densify-small-mask-area 400 \
    --densify-min-seed-pixels 1 \
    --densify-seed-keep-mode "$mode" \
    --densify-seed-min-support-views "$views" \
    --debug-root outputs/stream4d_reexport_v4_1 \
    --continue-on-error \
    2>&1 | tee "logs/${cfg}_reexport.log"
done
```

评估命令模板：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_tune30.txt \
  --pred-config "$cfg" \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config "${cfg}_on_mvp" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${cfg}_on_mvp_class_agnostic.txt"

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_tune30.txt \
  --pred-config "$cfg" \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config "${cfg}_on_adaptive" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${cfg}_on_adaptive_class_agnostic.txt"
```

实际日志：

- `logs/stream4d_v4_1_reliable_seedboundary_v1_e1_d32_top5_min20_tune30_reexport.log`
- `logs/stream4d_v4_1_reliable_seedboundary_v2_e1_d32_top5_min20_tune30_reexport.log`
- `logs/stream4d_v4_1_reliable_seedcomponent_v1_e1_d32_top5_min20_tune30_reexport.log`
- `logs/stream4d_v4_1_reliable_seedcomponent_v2_e1_d32_top5_min20_tune30_reexport.log`
- 对应 `_eval.log`、`_on_mvp_eval.log`、`_on_adaptive_eval.log`

### 9.5 tune30：seed-all score mode

生成命令模板：

```bash
for score in area reliability observations dense_quality; do
  tag=${score//_/-}
  cfg="stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_score_${tag}_tune30"
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
    --seq-list splits/scannet_tune30.txt \
    --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
    --output-config "$cfg" \
    --reexport-mode reliable_densify \
    --export-nn-radius 0.08 \
    --export-mask-sample-stride 2 \
    --export-mask-max-pixels 12000 \
    --export-max-masks-per-object 5 \
    --export-min-points-per-object 20 \
    --export-score-mode "$score" \
    --densify-boundary-erosion 1 \
    --densify-seed-distance-px 32 \
    --densify-small-mask-area 400 \
    --densify-min-seed-pixels 1 \
    --densify-seed-keep-mode all \
    --debug-root outputs/stream4d_reexport_v4_1 \
    --continue-on-error \
    2>&1 | tee "logs/${cfg}_reexport.log"
done
```

评估命令同 9.4。实际日志名为：

- `logs/stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_score_area_tune30_*.log`
- `logs/stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_score_reliability_tune30_*.log`
- `logs/stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_score_observations_tune30_*.log`
- `logs/stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_score_dense-quality_tune30_*.log`

### 9.6 tune30：低分 unmatched secondary fusion

生成和评估命令模板：

```bash
for r in 1p5 2p0; do
  val=${r/p/.}
  for st in 0p2 0p05; do
    score=${st/p/.}
    cfg="stream4d_v4_1_select_seedall_r${r}_unmatched_s${st}_tune30"
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
      --root . \
      --seq-list splits/scannet_tune30.txt \
      --primary-config stream4d_v4_1_reliable_e1_d32_top5_tune30 \
      --secondary-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_tune30 \
      --output-config "$cfg" \
      --fusion-mode select_secondary \
      --select-min-primary-ioc 0.7 \
      --select-max-expansion "$val" \
      --add-unmatched-secondary \
      --primary-score 1.0 \
      --secondary-score "$score" \
      2>&1 | tee "logs/${cfg}.log"
  done
done
```

评估命令同 9.4。实际日志：

- `logs/stream4d_v4_1_select_seedall_r1p5_unmatched_s0p2_tune30*.log`
- `logs/stream4d_v4_1_select_seedall_r1p5_unmatched_s0p05_tune30*.log`
- `logs/stream4d_v4_1_select_seedall_r2p0_unmatched_s0p2_tune30*.log`
- `logs/stream4d_v4_1_select_seedall_r2p0_unmatched_s0p05_tune30*.log`

### 9.7 tune30/final：e2_d48_top10 补充诊断

先补跑 tune30 的 MVP/adaptive fixed：

```bash
for cfg in stream4d_v4_1_reliable_e2_d48_top10_tune30 stream4d_v4_1_reliable_e1_d64_top10_tune30; do
  for target in mvp:stream4d_scannet_32f_ioc075_fixmem adaptive:stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute; do
    name=${target%:*}
    tmp=${target#*:}
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
      --root . \
      --seq-list splits/scannet_tune30.txt \
      --pred-config "$cfg" \
      --pred-suffix _class_agnostic \
      --pre-points-config "$tmp" \
      --output-config "${cfg}_on_${name}" \
      --dataset scannet \
      --no-class \
      --output-file "data/evaluation/scannet/${cfg}_on_${name}_class_agnostic.txt" \
      2>&1 | tee "logs/${cfg}_on_${name}_eval.log"
  done
done
```

`e2_d48_top10` final 生成：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet_final.txt \
  --input-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config stream4d_v4_1_reliable_e2_d48_top10_final \
  --reexport-mode reliable_densify \
  --export-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 10 \
  --export-min-points-per-object 100 \
  --export-score-mode one \
  --densify-boundary-erosion 2 \
  --densify-seed-distance-px 48 \
  --densify-small-mask-area 400 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode none \
  --debug-root outputs/stream4d_reexport_v4_1 \
  --continue-on-error \
  2>&1 | tee logs/stream4d_v4_1_reliable_e2_d48_top10_final_reexport.log
```

final 评估：

```bash
cfg=stream4d_v4_1_reliable_e2_d48_top10_final

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class \
  2>&1 | tee "logs/${cfg}_eval.log"

for target in mvp:stream4d_scannet_32f_ioc075_fixmem adaptive:stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute scannet:scannet_v3_final; do
  name=${target%:*}
  tmp=${target#*:}
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
    --root . \
    --seq-list splits/scannet_final.txt \
    --pred-config "$cfg" \
    --pred-suffix _class_agnostic \
    --pre-points-config "$tmp" \
    --output-config "${cfg}_on_${name}" \
    --dataset scannet \
    --no-class \
    --output-file "data/evaluation/scannet/${cfg}_on_${name}_class_agnostic.txt" \
    2>&1 | tee "logs/${cfg}_on_${name}_eval.log"
done
```

## 10. 继续推进：质量门控 object rescue 诊断

日期：2026-06-08。

### 10.1 继续推进原因

第 15 节复盘显示：

- 单纯扩张已有 object 不够。
- seed-all 增加 coverage 但污染 mask。
- score mode 和低分 unmatched 不能解决问题。

因此本轮尝试一个更接近 memory-v2/object recall 的局部子问题：从 secondary 高召回配置中只救回少量 unmatched object，并加质量门控，避免把所有 secondary object 加回来。

这不是使用 GT 的筛选；只使用 prediction 自身的：

- secondary score；
- secondary mask 面积；
- 与 primary mask 的 IoC；
- 每个 scene 最多救回数量。

### 10.2 新增代码修改

修改：

- `Stream3D/tools/fuse_prediction_configs.py`

新增参数：

```text
--unmatched-min-secondary-score
--unmatched-min-area
--unmatched-max-area
--unmatched-top-k
```

用途：

```text
select_secondary + --add-unmatched-secondary 时，不再把所有 unmatched secondary 都加回来。
先按 secondary score 和 mask 面积过滤，再按 score 排序，每个 scene 最多保留 unmatched_top_k 个。
```

语法检查和单测：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/fuse_prediction_configs.py \
  stream4d/reliable_densifier.py \
  stream4d/export_scannet.py \
  stream4d/reexport_scannet.py \
  tests/test_stream4d_protocol_fixes.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 5 tests in 0.001s
OK
```

### 10.3 secondary score / area 分布检查

命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
from pathlib import Path
import numpy as np
configs=[
 'stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_score_observations_tune30',
 'stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_tune30',
 'stream4d_v4_1_reliable_e1_d32_top5_tune30']
seqs=[x.strip() for x in Path('splits/scannet_tune30.txt').read_text().splitlines() if x.strip()]
for cfg in configs:
 vals=[]; areas=[]; inst=0
 for seq in seqs:
  p=Path('data/prediction')/f'{cfg}_class_agnostic'/f'{seq}.npz'
  with np.load(p) as d:
   vals.extend(d['pred_score'].astype(float).tolist())
   areas.extend(d['pred_masks'].sum(axis=0).astype(float).tolist())
   inst+=d['pred_masks'].shape[1]
 vals=np.array(vals); areas=np.array(areas)
 print(cfg, 'n',inst)
 print(' score percentiles', np.percentile(vals,[0,10,25,50,75,90,95,99,100]).round(3).tolist())
 print(' area percentiles', np.percentile(areas,[0,10,25,50,75,90,95,99,100]).round(1).tolist())
PY
```

结论：

```text
score_observations 的 score 大多数是 4 或 5，区分度弱。
area 分布更有用，因此 rescue 主要按 top_k 和面积范围做门控。
```

### 10.4 seedall-observations secondary rescue

命令模板：

```bash
for spec in \
  k1_a100_2500:1:100:2500 \
  k2_a100_2500:2:100:2500 \
  k4_a100_2500:4:100:2500 \
  k2_a100_1500:2:100:1500 \
  k2_a250_2500:2:250:2500; do
  name=${spec%%:*}
  rest=${spec#*:}
  topk=${rest%%:*}
  rest=${rest#*:}
  mina=${rest%%:*}
  maxa=${rest#*:}
  cfg="stream4d_v4_1_rescue_obs_r1p5_${name}_tune30"
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
    --root . \
    --seq-list splits/scannet_tune30.txt \
    --primary-config stream4d_v4_1_reliable_e1_d32_top5_tune30 \
    --secondary-config stream4d_v4_1_reliable_seedall_e1_d32_top5_min20_score_observations_tune30 \
    --output-config "$cfg" \
    --fusion-mode select_secondary \
    --select-min-primary-ioc 0.7 \
    --select-max-expansion 1.5 \
    --add-unmatched-secondary \
    --unmatched-min-secondary-score 4 \
    --unmatched-min-area "$mina" \
    --unmatched-max-area "$maxa" \
    --unmatched-top-k "$topk" \
    --primary-score 1.0 \
    --secondary-score 0.05 \
    2>&1 | tee "logs/${cfg}.log"
done
```

每个 config 均运行：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_tune30.txt \
  --pred-config "$cfg" \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config "${cfg}_on_mvp" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${cfg}_on_mvp_class_agnostic.txt"

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_tune30.txt \
  --pred-config "$cfg" \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  --output-config "${cfg}_on_adaptive" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${cfg}_on_adaptive_class_agnostic.txt"
```

实际日志：

- `logs/stream4d_v4_1_rescue_obs_r1p5_k1_a100_2500_tune30*.log`
- `logs/stream4d_v4_1_rescue_obs_r1p5_k2_a100_2500_tune30*.log`
- `logs/stream4d_v4_1_rescue_obs_r1p5_k4_a100_2500_tune30*.log`
- `logs/stream4d_v4_1_rescue_obs_r1p5_k2_a100_1500_tune30*.log`
- `logs/stream4d_v4_1_rescue_obs_r1p5_k2_a250_2500_tune30*.log`

### 10.5 e2_d48 secondary rescue

命令模板：

```bash
for spec in k1_a100_3000:1:100:3000 k2_a100_3000:2:100:3000; do
  name=${spec%%:*}
  rest=${spec#*:}
  topk=${rest%%:*}
  rest=${rest#*:}
  mina=${rest%%:*}
  maxa=${rest#*:}
  cfg="stream4d_v4_1_rescue_e2d48_r1p5_${name}_tune30"
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
    --root . \
    --seq-list splits/scannet_tune30.txt \
    --primary-config stream4d_v4_1_reliable_e1_d32_top5_tune30 \
    --secondary-config stream4d_v4_1_reliable_e2_d48_top10_tune30 \
    --output-config "$cfg" \
    --fusion-mode select_secondary \
    --select-min-primary-ioc 0.7 \
    --select-max-expansion 1.5 \
    --add-unmatched-secondary \
    --unmatched-min-area "$mina" \
    --unmatched-max-area "$maxa" \
    --unmatched-top-k "$topk" \
    --primary-score 1.0 \
    --secondary-score 0.05 \
    2>&1 | tee "logs/${cfg}.log"
done
```

评估命令同 10.4。实际日志：

- `logs/stream4d_v4_1_rescue_e2d48_r1p5_k1_a100_3000_tune30*.log`
- `logs/stream4d_v4_1_rescue_e2d48_r1p5_k2_a100_3000_tune30*.log`

## 11. 2026-06-08 ObjectMemory4D-v2 小规模多窗口实验

本节对应 `docs/stream4d_v4_1_with_stream3d_inherit_experiments_for_codex.md` 的 Phase 4。前面多轮 reliable densify / seed / rescue / fusion 后处理没有达成目标，因此本轮按计划实现上游 `ObjectMemory4D-v2`，并在 `scene0050_00` 上做 96f / 128f 小规模多窗口验证。

### 11.1 本轮新增或修改的代码文件

新增：

```text
Stream3D/stream4d/appearance_memory.py
Stream3D/stream4d/motion_memory.py
Stream3D/stream4d/memory_diagnostics.py
Stream3D/stream4d/object_memory_v2.py
Stream3D/stream4d/replay_memory.py
```

修改：

```text
Stream3D/stream4d/local_4d_filter.py
Stream3D/stream4d/run_scannet.py
Stream3D/tests/test_stream4d_protocol_fixes.py
```

核心功能：

```text
1. LocalProposal 增加 appearance_feature / centroid_feature / feature_type 字段。
2. ObjectMemory4D-v2 增加 Hungarian one-to-one matching。
3. ObjectMemory4D-v2 matching score 使用 carrier overlap、RGB histogram appearance、2D mask centroid proxy geometry、same-frame conflict penalty。
4. 新增 --memory-version old|v2，旧 memory 默认不变。
5. 新增 --memory-v2-min-carrier-score，用于避免纯外观相似导致过合并；默认 0.0，不改变旧 v2 baseline 行为。
6. replay_memory.py 可读取已经保存的 carriers_window*.npz，复用同一批 D4RT carrier，不重新跑 13G checkpoint，只重放 memory/export。
```

注意：本轮的 `S_g` 是 2D mask centroid proxy，不是 GT 3D centroid；本轮的 `S_a` 是 RGB histogram，不是 CLIP/DINO。复盘中按这个事实记录，不冒充更强特征。

### 11.2 编译和单元测试

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m py_compile stream4d/*.py tests/*.py

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.001s
OK
```

## 17. 2026-06-08 继续推进：tiered inherit fusion 超过 32f fixed support

第 16 节证明 `strict 128f` 自己补 fixed support 不够，低置信 rescue 虽有效但 AP 仍低于 32f current。本节改方向：不再让 128f strict 作为主 coverage，而是继承 32f current 的完整 support/candidate coverage，把 128f strict 作为高置信 precision tier，32f current 作为低置信 coverage tier，再可选加入更低置信的宽 support tier。

核心思想：

```text
Tier 1: 128f strict component_densify rel1.0，高置信 score=1.0。
Tier 2: 32f current inherited candidate，低置信 score=0.2。
Tier 3: 96f component_densify no relative gate，更低置信 score=0.1。
Tier 4: 128f maskbp r0.02，最低置信 score=0.05/0.1/0.2，仅诊断。
```

### 17.1 32f current + 128f strict 命令

代表性命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

base=stream4d_scannet_scene0050_32f_ioc075_fixmem
strict=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
out=stream4d_v4_1_scene0050_32fcurrent_plus_128fstrict_base02_strict1

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config "$base" \
  --secondary-config "$strict" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --primary-score 0.2 \
  --secondary-score 1.0 \
  --summary-root outputs/stream4d_fusion_v4_1_continue \
  2>&1 | tee "logs/${out}_fuse.log"
```

评估：

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m evaluation.evaluate \
  --pred_path "data/prediction/${out}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$out" \
  --no_class \
  2>&1 | tee "logs/${out}_eval.log"

cross="${out}_cross_32fsupport"
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$out" \
  --pred-suffix _class_agnostic \
  --pre-points-config "$base" \
  --output-config "$cross" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${cross}_class_agnostic.txt" \
  --audit-root outputs/audit_v4_1_continue \
  2>&1 | tee "logs/${cross}.log"
```

实际运行的 32f+strict variants：

```text
stream4d_v4_1_scene0050_32fcurrent_plus_128fstrict_base1_strict02
stream4d_v4_1_scene0050_32fcurrent_plus_128fstrict_base02_strict1
stream4d_v4_1_scene0050_32fcurrent_plus_128fstrict_base1_strict02_drop03
stream4d_v4_1_scene0050_32fcurrent_plus_128fstrict_base02_strict1_drop03
stream4d_v4_1_scene0050_32fcurrent_plus_128fstrict_base1_strict1_drop03
stream4d_v4_1_scene0050_128fstrict_plus_32fcurrent_low_symmetric
```

### 17.2 加入第三层 compnone 命令

代表性命令：

```bash
mid=stream4d_v4_1_scene0050_128fstrict_plus_32fcurrent_low_symmetric
comp=stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_none_hard2
out2=stream4d_v4_1_scene0050_128fstrict_plus_32fcurrent_low_plus_compnone_lowertier

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config "$mid" \
  --secondary-config "$comp" \
  --output-config "$out2" \
  --fusion-mode concatenate \
  --preserve-primary-score \
  --secondary-score 0.1 \
  --summary-root outputs/stream4d_fusion_v4_1_continue \
  2>&1 | tee "logs/${out2}_fuse.log"
```

随后同样运行 `evaluation.evaluate` 和 `tools.evaluate_cross_prepoints`，target support 为：

```text
stream4d_scannet_scene0050_32f_ioc075_fixmem
```

### 17.3 加入第四层 maskbp 诊断命令

代表性命令：

```bash
mid=stream4d_v4_1_scene0050_128fstrict_plus_32fcurrent_low_plus_compnone_lowertier
maskbp=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_maskbp_r002_hard2
out=stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config "$mid" \
  --secondary-config "$maskbp" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --preserve-primary-score \
  --secondary-score 0.05 \
  --summary-root outputs/stream4d_fusion_v4_1_continue \
  2>&1 | tee "logs/${out}_fuse.log"
```

实际运行：

```text
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask010
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask020
```

### 17.4 多 support 诊断命令

最终候选：

```text
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
```

命令：

```bash
cfg=stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005

for support in \
  scannet_self_inherit \
  scannet_on_stream4d_mvp_prepoints \
  scannet_on_stream4d_adaptive_prepoints \
  stream4d_scannet_scene0050_32f_ioc075_fixmem
do
  out="${cfg}_cross_${support}"
  CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.evaluate_cross_prepoints \
    --root . \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$cfg" \
    --pred-suffix _class_agnostic \
    --pre-points-config "$support" \
    --output-config "$out" \
    --dataset scannet \
    --no-class \
    --output-file "data/evaluation/scannet/${out}_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue \
    2>&1 | tee "logs/${out}.log"
done
```

### 17.5 编译和回归测试

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" \
  -m py_compile stream4d/*.py tools/*.py tests/*.py

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" \
  -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.001s
OK
```

新增测试：

```text
test_memory_v2_enforces_one_to_one_matching
```

该测试构造两个当前 proposal 同时想匹配同一个历史 object 的情况，确认 `ObjectMemory4D-v2` 只允许一个 proposal 通过 Hungarian matching 匹配该 object，另一个必须新建 object。

### 11.3 96f old memory 真实 D4RT run

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 8 \
  --min-points-per-mask 2 \
  --query-chunk-size 1024 \
  --rho-min 0.35 \
  --local-ioc-threshold 0.75 \
  --history-match-threshold 0.30 \
  --lost-tolerance-windows 3 \
  --memory-version old \
  --export-mode rgbd_eval \
  --export-nn-radius 0.08 \
  --output-config stream4d_v4_1_memoryold_scene0050_96f_ioc075 \
  --debug-root outputs/stream4d_debug_v4_1_memoryold_scene0050_96f \
  2>&1 | tee logs/stream4d_v4_1_memoryold_scene0050_96f_run.log
```

关键输出：

```text
[stream4d] seq=scene0050_00 done objects=243 points=27340 hit_rate=0.9491 total_sec=126.79
```

### 11.4 96f memory-v2 真实 D4RT run

命令：

```bash
CUDA_VISIBLE_DEVICES=0 CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 8 \
  --min-points-per-mask 2 \
  --query-chunk-size 1024 \
  --rho-min 0.35 \
  --local-ioc-threshold 0.75 \
  --history-match-threshold 0.30 \
  --lost-tolerance-windows 3 \
  --memory-version v2 \
  --memory-v2-carrier-weight 0.55 \
  --memory-v2-appearance-weight 0.25 \
  --memory-v2-geometry-weight 0.20 \
  --memory-v2-motion-weight 0.0 \
  --memory-v2-conflict-weight 0.30 \
  --memory-v2-geometry-sigma 0.35 \
  --memory-v2-appearance-bins 8 \
  --memory-v2-appearance-max-pixels-per-mask 2048 \
  --memory-v2-appearance-max-masks-per-proposal 8 \
  --export-mode rgbd_eval \
  --export-nn-radius 0.08 \
  --output-config stream4d_v4_1_memoryv2_scene0050_96f_ioc075_wc055_wa025_wg020 \
  --debug-root outputs/stream4d_debug_v4_1_memoryv2_scene0050_96f \
  2>&1 | tee logs/stream4d_v4_1_memoryv2_scene0050_96f_run.log
```

关键输出：

```text
[stream4d] seq=scene0050_00 done objects=271 points=27340 hit_rate=0.9504 total_sec=136.65
```

### 11.5 评估命令和一次失败命令记录

错误命令：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u evaluation/evaluate.py ...
```

错误：

```text
ModuleNotFoundError: No module named 'evaluation'
```

修复方式：改用模块方式执行。

正确命令模板：

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${cfg}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${cfg}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${cfg} \
  --no_class \
  2>&1 | tee logs/${cfg}_eval.log
```

实际评估日志：

```text
logs/stream4d_v4_1_memoryold_scene0050_96f_eval.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_eval.log
```

### 11.6 replay 工具校验

目的：用同一批 `carriers_window*.npz` 重放 memory，不重新跑 D4RT，以便快速验证 memory 参数。

replay 校验命令：

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.replay_memory \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --local-ioc-threshold 0.75 \
  --history-match-threshold 0.30 \
  --lost-tolerance-windows 3 \
  --memory-version v2 \
  --memory-v2-carrier-weight 0.55 \
  --memory-v2-appearance-weight 0.25 \
  --memory-v2-geometry-weight 0.20 \
  --memory-v2-motion-weight 0.0 \
  --memory-v2-conflict-weight 0.30 \
  --memory-v2-geometry-sigma 0.35 \
  --memory-v2-appearance-bins 8 \
  --memory-v2-appearance-max-pixels-per-mask 2048 \
  --memory-v2-appearance-max-masks-per-proposal 8 \
  --export-nn-radius 0.08 \
  --input-debug-root outputs/stream4d_debug_v4_1_memoryold_scene0050_96f \
  --output-config stream4d_v4_1_memoryv2_replaycheck_scene0050_96f_ioc075_wc055_wa025_wg020 \
  --debug-root outputs/stream4d_replay_v4_1_memoryv2_check \
  2>&1 | tee logs/stream4d_v4_1_memoryv2_replaycheck_scene0050_96f.log
```

校验结果：

```text
真实 v2 run: objects=271 points=27340 hit_rate=0.9504
replay check: objects=271 points=27340 hit_rate=0.9504
```

replay 评估日志：

```text
logs/stream4d_v4_1_memoryv2_replaycheck_scene0050_96f_eval.log
```

### 11.7 memory-v2 参数修复实验

第一组：降低阈值、提高 appearance 权重，尝试解决 re-ID 失败。

实际 config：

```text
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc045_wa035_wg020
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr022_wc040_wa040_wg020
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr020_wc035_wa045_wg020
```

实际日志：

```text
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr025_wc045_wa035.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr022_wc040_wa040.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr020_wc035_wa045.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr025_wc045_wa035_eval.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr022_wc040_wa040_eval.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr020_wc035_wa045_eval.log
```

第二组：加入 `--memory-v2-min-carrier-score`，避免仅靠 appearance 合并。

实际 config：

```text
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc010
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr022_wc055_wa025_wg020_minc005
```

实际日志：

```text
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc005.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc010.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr022_minc005.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc005_eval.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc010_eval.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr022_minc005_eval.log
```

### 11.8 96f postprocess 诊断

输入 config：

```text
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005
```

rescore 命令模板：

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.rescore_scannet \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --input-config stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005 \
  --output-config ${out_cfg} \
  --score-mode one \
  --select-mode ${select_mode} \
  --filter-max-instances ${top_n} \
  --pre-points-policy recompute \
  --debug-root outputs/stream4d_rescore_v4_1_memoryv2_scene0050_96f \
  2>&1 | tee logs/${out_cfg}.log
```

实际 config：

```text
top227/top180/top120/top80/top60/top40/top30/top20/top10, select_mode=mask_count
top40, select_mode=area
top40, select_mode=carrier_count
top40, select_mode=coverage_sum
```

inherit 诊断命令与上面相同，但 `--pre-points-policy inherit`：

```text
stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc005_top40_maskcount_inherit
```

### 11.9 128f replay 与诊断

使用已有 carrier：

```text
outputs/stream4d_debug_scene0050_128f_ioc075_fixmem/scene0050_00/carriers_window000.npz ... carriers_window006.npz
```

实际 config：

```text
stream4d_v4_1_memoryv2_scene0050_128f_ioc075_wc055_wa025_wg020
stream4d_v4_1_memoryv2_scene0050_128f_ioc075_thr025_wc055_wa025_wg020_minc005
stream4d_v4_1_memoryv2_scene0050_128f_thr025_minc005_top40_maskcount
```

实际日志：

```text
logs/stream4d_v4_1_memoryv2_scene0050_128f_base.log
logs/stream4d_v4_1_memoryv2_scene0050_128f_thr025_minc005.log
logs/stream4d_v4_1_memoryv2_scene0050_128f_base_eval.log
logs/stream4d_v4_1_memoryv2_scene0050_128f_thr025_minc005_eval.log
logs/stream4d_v4_1_memoryv2_scene0050_128f_thr025_minc005_top40_maskcount.log
logs/stream4d_v4_1_memoryv2_scene0050_128f_thr025_minc005_top40_maskcount_eval.log
```

## 12. 2026-06-08 duplicate suppression 与 32f-primary fusion 诊断

用户提醒“不要一直堆工程，应该注重算法思路上的改进”。在停止继续扩展工程前，本节只记录已经启动并完成的诊断：point-IoC merge、point-NMS、以及 32f primary + 96f v2 secondary fusion。结论写入复盘第 18 节：这些后处理没有达成目标，下一步必须转向算法核心。

### 12.1 point-IoC merge 诊断

输入 config：

```text
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005
```

命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

for thr in 0.25 0.50 0.75 0.90; do
  tag=${thr/./p}
  cfg=stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointmerge${tag}
  CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.reexport_scannet \
    --seq-name scene0050_00 \
    --backbone Cropformer \
    --input-config stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005 \
    --output-config "$cfg" \
    --reexport-mode point_dilate \
    --export-score-mode one \
    --merge-point-ioc-threshold "$thr" \
    --debug-root outputs/stream4d_reexport_v4_1_pointmerge \
    2>&1 | tee "logs/${cfg}.log"

  CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
    --pred_path "data/prediction/${cfg}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$cfg" \
    --no_class \
    2>&1 | tee "logs/${cfg}_eval.log"
done
```

实际日志：

```text
logs/stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointmerge0p25*.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointmerge0p50*.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointmerge0p75*.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointmerge0p90*.log
```

### 12.2 point-NMS 诊断

新增文件：

```text
Stream3D/tools/point_nms_prediction.py
```

用途：只用 prediction 和 object_dict 里的无监督 point_ids 做重复预测抑制。它保留原预测 mask，不像 point-IoC merge 那样把多个 mask 合成并集。

编译和测试：

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m py_compile stream4d/*.py tools/*.py tests/*.py

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.001s
OK
```

命令模板：

```bash
for thr in 0.25 0.50 0.75 0.90; do
  tag=${thr/./p}
  cfg=stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointnms${tag}_maskcount
  CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.point_nms_prediction \
    --seq-name scene0050_00 \
    --backbone Cropformer \
    --input-config stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005 \
    --output-config "$cfg" \
    --select-mode mask_count \
    --output-score-mode one \
    --point-overlap-mode ioc \
    --point-overlap-threshold "$thr" \
    --pre-points-policy recompute \
    --debug-root outputs/stream4d_point_nms_v4_1 \
    2>&1 | tee "logs/${cfg}.log"

  CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
    --pred_path "data/prediction/${cfg}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$cfg" \
    --no_class \
    2>&1 | tee "logs/${cfg}_eval.log"
done
```

score 诊断：

```text
stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc005_top40_maskcount_scoremask
stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointnms0p25_maskcount_scoreselect
stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointnms0p50_maskcount_scoreselect
```

实际日志：

```text
logs/stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointnms0p25_maskcount*.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointnms0p50_maskcount*.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointnms0p75_maskcount*.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_minc005_pointnms0p90_maskcount*.log
logs/stream4d_v4_1_memoryv2_scene0050_96f_thr025_minc005_top40_maskcount_scoremask*.log
```

### 12.3 32f primary + 96f v2 secondary fusion

目的：不让 96f 多窗口结果主导输出，只测试它能否作为 32f current 的局部增强。

primary：

```text
stream4d_scannet_scene0050_32f_ioc075_fixmem
```

secondary：

```text
stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005
```

命令模板：

```bash
printf 'scene0050_00\n' > /tmp/scene0050_only.txt

for exp in 1.25 1.50 2.00 3.00; do
  tag=${exp/./p}
  cfg=stream4d_v4_1_scene0050_32f_primary_96fv2_replace_exp${tag}
  CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.fuse_prediction_configs \
    --root . \
    --seq-list /tmp/scene0050_only.txt \
    --primary-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
    --secondary-config stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005 \
    --output-config "$cfg" \
    --fusion-mode select_secondary \
    --select-min-primary-ioc 0.7 \
    --select-max-expansion "$exp" \
    --primary-score 1.0 \
    --secondary-score 1.0 \
    2>&1 | tee "logs/${cfg}.log"

  CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
    --pred_path "data/prediction/${cfg}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$cfg" \
    --no_class \
    2>&1 | tee "logs/${cfg}_eval.log"
done
```

实际日志：

```text
logs/stream4d_v4_1_scene0050_32f_primary_96fv2_replace_exp1p25*.log
logs/stream4d_v4_1_scene0050_32f_primary_96fv2_replace_exp1p50*.log
logs/stream4d_v4_1_scene0050_32f_primary_96fv2_replace_exp2p00*.log
logs/stream4d_v4_1_scene0050_32f_primary_96fv2_replace_exp3p00*.log
```

## 13. 2026-06-08 Evidence Graph 算法原型

本节响应“不要一直堆工程，应该注重算法思路上的改进”。这一轮不再做 point-merge / NMS / fusion 后处理，而是改变 object 形成方式：

```text
mask observation -> evidence graph node
shared carrier evidence -> positive edge
same-frame different-mask -> hard negative / cannot-link
graph partition -> object hypothesis
```

新增代码：

```text
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/replay_evidence_graph.py
```

### 13.1 编译和测试

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m py_compile stream4d/*.py tools/*.py tests/*.py

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.001s
OK
```

### 13.2 96f evidence graph 参数扫描

输入 carrier cache：

```text
outputs/stream4d_debug_v4_1_memoryold_scene0050_96f/scene0050_00/carriers_window*.npz
```

基础命令模板：

```bash
cfg=stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.78 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-nn-radius 0.08 \
  --input-debug-root outputs/stream4d_debug_v4_1_memoryold_scene0050_96f \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1 \
  2>&1 | tee "logs/${cfg}.log"

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class \
  2>&1 | tee "logs/${cfg}_eval.log"
```

实际运行 config：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p50_shared2
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2
stream4d_v4_1_egraph_scene0050_96f_ioc0p85_shared2
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2_minobs2
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2_minobs3
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2_minobs5
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2_minobs8
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2_minobs9
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2_minobs10
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2_minobs11
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared2_minobs12
stream4d_v4_1_egraph_scene0050_96f_ioc0p65_shared2_minobs10
stream4d_v4_1_egraph_scene0050_96f_ioc0p75_shared2_minobs9
stream4d_v4_1_egraph_scene0050_96f_ioc0p75_shared2_minobs10
stream4d_v4_1_egraph_scene0050_96f_ioc0p75_shared2_minobs11
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10
stream4d_v4_1_egraph_scene0050_96f_ioc0p80_shared2_minobs10
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared3_minobs8
stream4d_v4_1_egraph_scene0050_96f_ioc0p70_shared4_minobs8
```

对应日志：

```text
logs/stream4d_v4_1_egraph_scene0050_96f_*.log
logs/stream4d_v4_1_egraph_scene0050_96f_*_eval.log
```

### 13.3 fixed-support 诊断

最佳 recompute config：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10
```

放回 96f v2 full support：

```bash
cfg=stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10
out=${cfg}_on_96fv2support

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config stream4d_v4_1_memoryv2_scene0050_96f_ioc075_thr025_wc055_wa025_wg020_minc005 \
  --no_class \
  2>&1 | tee "logs/${out}_eval.log"
```

放到 32f current support：

```bash
out=${cfg}_on_32fsupport
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --no_class \
  2>&1 | tee "logs/${out}_eval.log"
```

### 13.4 128f 同参数验证

输入 carrier cache：

```text
outputs/stream4d_debug_scene0050_128f_ioc075_fixmem/scene0050_00/carriers_window*.npz
```

命令：

```bash
cfg=stream4d_v4_1_egraph_scene0050_128f_ioc0p78_shared2_minobs10

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 128 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.78 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-nn-radius 0.08 \
  --input-debug-root outputs/stream4d_debug_scene0050_128f_ioc075_fixmem \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1 \
  2>&1 | tee "logs/${cfg}.log"

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class \
  2>&1 | tee "logs/${cfg}_eval.log"
```

## 14. 2026-06-08 Evidence-aware WTA 与 mask support 算法扫描

本节继续响应“不要一直堆工程，应该注重算法思路上的改进”。第 19 节复盘发现 evidence graph 的主要瓶颈之一是 point ownership conflict 和高 IoU support 质量。本轮只做与算法假设直接相关的改动：

```text
1. 给 evidence graph 的 object 写入无监督证据质量：
   - component node 数
   - 覆盖帧数
   - carrier 数
   - 平均 coverage
   - evidence quality
2. 在 RGB-D 导出阶段支持 point-level winner-take-all。
3. WTA 不再只能“一有冲突就唯一归属”，新增 min_conflict_owners：
   - 2 表示 hard WTA
   - 3/4/5 表示只处理更严重的多物体抢点
4. 用稳定 mask 回投三维点，测试精确 core support 是否能提高高 IoU AP。
```

新增/修改代码：

```text
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/splits/scannet_scene0050.txt
```

### 14.1 编译和单测

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m py_compile stream4d/*.py tools/*.py tests/*.py

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.001s
OK
```

### 14.2 Evidence graph WTA / scoring 命令模板

以下实验复用 96f carrier cache：

```text
outputs/stream4d_debug_v4_1_memoryold_scene0050_96f/scene0050_00/carriers_window*.npz
```

基础命令模板：

```bash
cfg=stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_softwta_compactness_min4

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.78 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-nn-radius 0.08 \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 4 \
  --input-debug-root outputs/stream4d_debug_v4_1_memoryold_scene0050_96f \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1 \
  2>&1 | tee "logs/${cfg}.log"

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class \
  2>&1 | tee "logs/${cfg}_eval.log"
```

实际运行 config：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_wta_evidence-density
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_wta_evidence-quality
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_wta_compactness
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_score_evidence-density
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_score_evidence-quality
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_score_compactness
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_softwta_compactness_min3
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_shared2_minobs10_softwta_compactness_min5
stream4d_v4_1_egraph_scene0050_96f_ioc0p76_shared2_minobs10_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p77_shared2_minobs10_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_shared2_minobs10_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p80_shared2_minobs10_softwta_compactness_min4
```

### 14.3 Mask backprojection support 命令模板

代表性命令：

```bash
cfg=stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_r002_hard2

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.79 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-nn-radius 0.02 \
  --export-support-mode mask_backproject \
  --export-mask-sample-stride 2 \
  --export-max-masks-per-object 5 \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 2 \
  --input-debug-root outputs/stream4d_debug_v4_1_memoryold_scene0050_96f \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1 \
  2>&1 | tee "logs/${cfg}.log"

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class \
  2>&1 | tee "logs/${cfg}_eval.log"
```

实际运行 config：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_hybrid_m5_s2_r005_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_mask_backproject_m5_s2_r005_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m3_s2_r005_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m8_s2_r005_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m3_s1_r005_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s1_r005_softwta_compactness_min4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r004_soft4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r006_soft4
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r005_nowta
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r005_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r004_nowta
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r004_min3
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r004_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_m5_s2_r003_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_r002_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_r003_min3
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_r003_nowta
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_r003_m6_hard2
```

### 14.4 128f 验证命令

```bash
cfg=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_maskbp_r002_hard2

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 128 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.79 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-nn-radius 0.02 \
  --export-support-mode mask_backproject \
  --export-mask-sample-stride 2 \
  --export-max-masks-per-object 5 \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 2 \
  --input-debug-root outputs/stream4d_debug_scene0050_128f_ioc075_fixmem \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1 \
  2>&1 | tee "logs/${cfg}.log"
```

评估命令同 14.3。

### 14.5 多尺度 hypothesis 诊断命令

为避免临时输入不可复现，新增单场景 split：

```text
splits/scannet_scene0050.txt
```

代表性命令：

```bash
cfg=stream4d_v4_1_egraph_scene0050_96f_multiscale_core_r002_carrier079_s02

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u tools/fuse_prediction_configs.py \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_maskbp_r002_hard2 \
  --secondary-config stream4d_v4_1_egraph_scene0050_96f_ioc0p79_shared2_minobs10_softwta_compactness_min4 \
  --output-config "$cfg" \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score 0.2 \
  --summary-root outputs/stream4d_fusion_v4_1 \
  2>&1 | tee "logs/${cfg}_fuse.log"
```

实际运行 config：

```text
stream4d_v4_1_egraph_scene0050_96f_multiscale_core_r002_carrier079_s02
stream4d_v4_1_egraph_scene0050_96f_multiscale_core_r003_carrier080_s02
stream4d_v4_1_egraph_scene0050_96f_multiscale_core_r002_carrier079_s0p8
stream4d_v4_1_egraph_scene0050_96f_multiscale_core_r002_carrier079_s1p1
```

## 15. 2026-06-08 Object 内 scale-aware support 与 component densify

本节继续第 14 节结论：不要再简单拼接两套 hypothesis，而是在同一个 object 内选择 support。本轮新增两个算法模式：

```text
core_fringe:
  先用稳定 mask 生成 precise core。
  再从 carrier support 中只选择离 core 足够近的 fringe 点。

component_densify:
  carrier 只作为种子。
  在稳定 mask 内寻找包含 carrier 种子的 2D 连通区域。
  只回投这些连通区域，不直接把 carrier support 并入 object。
  再用 mask relative coverage gate 只保留与该 object 最强 mask 同等级可信的视角。
```

新增/修改代码：

```text
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/replay_memory.py
Stream3D/stream4d/run_scannet.py
```

### 15.1 编译和单测

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m py_compile stream4d/*.py tools/*.py tests/*.py

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.002s
OK
```

### 15.2 core_fringe 命令模板

代表性命令：

```bash
cfg=stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_corefringe_core002_fr004_ratio025_hard2

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.79 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-nn-radius 0.02 \
  --export-support-mode core_fringe \
  --export-core-nn-radius 0.02 \
  --export-fringe-nn-radius 0.08 \
  --export-fringe-radius 0.04 \
  --export-fringe-max-ratio 0.25 \
  --export-mask-sample-stride 2 \
  --export-max-masks-per-object 5 \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 2 \
  --input-debug-root outputs/stream4d_debug_v4_1_memoryold_scene0050_96f \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1 \
  2>&1 | tee "logs/${cfg}.log"
```

评估命令同前：

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class \
  2>&1 | tee "logs/${cfg}_eval.log"
```

实际运行 config：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_corefringe_core002_fr002_ratio025_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_corefringe_core002_fr004_ratio025_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_corefringe_core002_fr004_ratio050_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_corefringe_core002_fr008_ratio025_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_corefringe_core002_fr008_ratio050_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_corefringe_core002_fr012_ratio050_hard2
```

### 15.3 component_densify 命令模板

代表性命令，当前 128f 最佳：

```bash
cfg=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 128 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.79 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-nn-radius 0.03 \
  --export-support-mode component_densify \
  --export-core-nn-radius 0.03 \
  --export-fringe-nn-radius 0.08 \
  --export-mask-sample-stride 2 \
  --export-max-masks-per-object 8 \
  --export-mask-min-relative-coverage 1.0 \
  --densify-seed-distance-px 16 \
  --densify-seed-keep-mode none \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 2 \
  --input-debug-root outputs/stream4d_debug_scene0050_128f_ioc075_fixmem \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1 \
  2>&1 | tee "logs/${cfg}.log"

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class \
  2>&1 | tee "logs/${cfg}_eval.log"
```

实际运行 config：

```text
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_none_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d32_none_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d64_none_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r002_d32_none_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d32_supported_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d32_component_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r0003_d16_m5_nowta
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r0003_d16_m5_min3
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r0003_d16_m6_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r0003_d16_m6_min3
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r0003_d16_m8_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r0003_d16_m8_min3
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r0004_d16_r004_m5_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r0004_d16_r004_m8_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel05_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel07_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel09_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel085_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel088_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel092_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel095_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel097_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel099_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_minobs10_compdens_r003_d16_m8_rel095_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p80_minobs10_compdens_r003_d16_m8_rel095_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p78_minobs10_compdens_r003_d16_m8_rel090_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p80_minobs10_compdens_r003_d16_m8_rel090_hard2
stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
```

### 15.4 fixed-support 诊断命令

128f 最佳 prediction 放回其它 support：

```bash
cfg=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2

for support in \
  stream4d_scannet_scene0050_32f_ioc075_fixmem \
  scannet_self_inherit \
  scannet_on_stream4d_mvp_prepoints \
  scannet_on_stream4d_adaptive_prepoints
do
  out="${cfg}_on_${support}"
  CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
    /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
    --pred_path "data/prediction/${cfg}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$support" \
    --no_class \
    2>&1 | tee "logs/${out}_eval.log"
done
```

注意：`data/TMP/scannet/scene0050_00_pre_points.npy` 不存在；本轮没有用不存在的 `scannet` tmp_config 补数字。

## 16. 2026-06-08 继续推进 fixed-support：cross-prepoints 审计、低置信补漏、support completion

本节针对第 15 节遗留问题继续推进：`scene0050_00` 的 recompute support 已经超过 32f current，但 fixed/inherit support 没有达成。根据计划要求，本轮不再只看 direct evaluator 换 `tmp_config` 的结果，而是优先使用 `tools.evaluate_cross_prepoints` 做 shape audit 和 full-scene/cropped-scene 安全物化。

新增/修改代码：

```text
Stream3D/tools/fuse_prediction_configs.py
  新增 --preserve-primary-score / --preserve-secondary-score。
  用于多级低置信补漏时保留上一轮已经校准过的 score。

Stream3D/tools/complete_prediction_on_support.py
  新增工具。
  读取 full-scene prediction、目标 support pre_points 和 ScanNet mesh。
  只在目标 support 中对未覆盖点做 3D nearest-object completion。
  支持 radius、max_added_ratio、max_added_points_per_object 和 only_missing。
  不读取 GT，不使用 evaluator 的 GT instance 信息生成 prediction。
```

### 16.1 cross-prepoints fixed-support 审计命令

代表性命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2 \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2_cross_32fsupport_audit \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2_cross_32fsupport_audit_class_agnostic.txt \
  --audit-root outputs/audit_v4_1_continue \
  2>&1 | tee logs/stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2_cross_32fsupport_audit.log
```

实际审计的 prediction config：

```text
stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_rel05_hard2
stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_none_hard2
stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_maskbp_r002_hard2
```

补充守卫命令：

```bash
# 32f current 自己放回自己 support，验证 cross-prepoints 工具不会改变分数。
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config stream4d_scannet_scene0050_32f_ioc075_fixmem_cross_self_audit \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_scannet_scene0050_32f_ioc075_fixmem_cross_self_audit_class_agnostic.txt \
  --audit-root outputs/audit_v4_1_continue

# 原版 scannet prediction 放到同一个 32f support，判断 target support 本身是否很难。
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config scannet \
  --pred-suffix _class_agnostic \
  --source-pre-points-config scannet_self_inherit \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config scannet_on_scene0050_32f_ioc075_fixmem_cross_audit \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/scannet_on_scene0050_32f_ioc075_fixmem_cross_audit_class_agnostic.txt \
  --audit-root outputs/audit_v4_1_continue
```

审计输出：

```text
Stream3D/outputs/audit_v4_1_continue/cross_prepoints_audit.md
Stream3D/outputs/audit_v4_1_continue/cross_prepoints_audit.csv
Stream3D/outputs/audit_v4_1_continue/cross_prepoints/*.json
```

### 16.2 低置信 multi-scale rescue fusion 命令

代表性命令：

```bash
primary=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
secondary=stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_none_hard2
out=stream4d_v4_1_scene0050_128f_strict_plus_compnone_lowconf

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config "$primary" \
  --secondary-config "$secondary" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score 0.2 \
  --summary-root outputs/stream4d_fusion_v4_1_continue \
  2>&1 | tee "logs/${out}_fuse.log"
```

评估命令：

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m evaluation.evaluate \
  --pred_path "data/prediction/${out}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$out" \
  --no_class \
  2>&1 | tee "logs/${out}_eval.log"

cross="${out}_cross_32fsupport"
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$out" \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config "$cross" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${cross}_class_agnostic.txt" \
  --audit-root outputs/audit_v4_1_continue \
  2>&1 | tee "logs/${cross}.log"
```

实际运行的 rescue/fusion config：

```text
stream4d_v4_1_scene0050_128f_strict_plus_compnone_lowconf
stream4d_v4_1_scene0050_128f_strict_plus_maskbp_lowconf
stream4d_v4_1_scene0050_128f_strict_select_compnone_ioc03_exp3_noadd
stream4d_v4_1_scene0050_128f_strict_select_compnone_ioc03_exp6_noadd
stream4d_v4_1_scene0050_128f_strict_select_compnone_ioc05_exp3_noadd
stream4d_v4_1_scene0050_128f_strict_select_compnone_ioc03_exp3_add20
stream4d_v4_1_scene0050_128f_strict_plus_compnone_lowconf_dropiou01
stream4d_v4_1_scene0050_128f_strict_plus_compnone_lowconf_dropiou03
stream4d_v4_1_scene0050_128f_strict_plus_compnone_lowconf_dropiou05
stream4d_v4_1_scene0050_128f_strict_plus_compnone_lowconf_dropiou07
stream4d_v4_1_scene0050_128f_strict_plus_compnone_plus_maskbp_tiered
stream4d_v4_1_scene0050_128f_strict_plus_maskbp_plus_compnone_tiered
stream4d_v4_1_scene0050_128f_strict_plus_32fmerge070_lowconf
stream4d_v4_1_scene0050_128f_strict_plus_32fmerge070_s005
stream4d_v4_1_scene0050_128f_strict_plus_32fmerge070_s010
stream4d_v4_1_scene0050_128f_strict_plus_32fmerge070_s040
stream4d_v4_1_scene0050_128f_strict_plus_32fmerge070_s080
```

其中 `*_tiered` 使用了新增的 `--preserve-primary-score`，避免第二次融合时把第一层低置信 rescue 重新变成高置信。

### 16.3 support completion 命令

代表性命令：

```bash
base=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
support=stream4d_scannet_scene0050_32f_ioc075_fixmem
out=stream4d_v4_1_scene0050_128f_strict_supportcomplete_r002_ratio1

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.complete_prediction_on_support \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$base" \
  --support-config "$support" \
  --output-config "$out" \
  --radius 0.02 \
  --max-added-ratio 1.0 \
  --only-missing \
  --summary-root outputs/stream4d_support_completion_v4_1_continue \
  2>&1 | tee "logs/${out}_complete.log"
```

实际运行 config：

```text
stream4d_v4_1_scene0050_128f_strict_supportcomplete_r002_ratio1
stream4d_v4_1_scene0050_128f_strict_supportcomplete_r004_ratio1
stream4d_v4_1_scene0050_128f_strict_supportcomplete_r006_ratio1
stream4d_v4_1_scene0050_128f_strict_supportcomplete_r004_ratio2
stream4d_v4_1_scene0050_128f_strict_supportcomplete_r004_nocap
```

support completion 只使用：

```text
prediction masks
目标 pre_points
ScanNet scene mesh 坐标
```

不读取 GT，不用 GT 调整补点。

### 16.4 编译和回归测试

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" \
  -m py_compile stream4d/*.py tools/*.py tests/*.py

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" \
  -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.001s
OK
```

## 18. 2026-06-08 继续推进：containment suppression、score calibration 与 mask NMS

### 18.1 继续推进原因

第 17 节已经在 `scene0050_00` 的 32f fixed support 上超过 32f current，但仍未超过原版 Stream3D on same support。

关键差距：

```text
Stream3D on 32f support:
  0.391132 / 0.646154 / 0.761538

上一轮 tiered inherit best:
  0.242409 / 0.582689 / 0.744262
```

本轮不继续堆 support completion 工程，而是沿算法方向尝试：

```text
1. 高置信 strict evidence graph 负责 precision。
2. 32f current 负责 inherited coverage。
3. 低置信 compnone / maskbp 负责 recall。
4. 用 containment suppression 删除“低分候选几乎被高分候选包含”的重复候选。
5. 用 score calibration / mask NMS 诊断排序和重复实例是否是主要瓶颈。
```

### 18.2 新增和修改代码

修改：

```text
Stream3D/tools/fuse_prediction_configs.py
```

新增参数：

```text
--drop-secondary-overlap-mode iou|secondary_ioc|min_ioc
```

含义：

```text
iou:
  原来的普通 IoU。

secondary_ioc:
  intersection / secondary_area。
  用于判断一个低分 secondary 候选是否几乎被高分 primary 候选覆盖。

min_ioc:
  intersection / min(primary_area, secondary_area)。
  用于判断两个 mask 是否近似互相包含或高度重复。
```

新增：

```text
Stream3D/tools/rescore_prediction_scores.py
```

用途：

```text
不读取 GT，只读取 prediction mask。
保留原始 score tier，然后可以用 mask area/log_area/sqrt_area 作为同层 tiebreaker。
也可以按 full-scene mask area 做最小面积过滤。
```

新增：

```text
Stream3D/tools/nms_prediction_masks.py
```

用途：

```text
不读取 GT，不依赖 object_dict。
直接对 prediction npz 做 class-agnostic mask NMS。
支持 iou / candidate_ioc / min_ioc。
```

### 18.3 通用环境和评估命令

所有命令在以下目录运行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
mkdir -p logs
```

本轮固定变量：

```bash
seq=splits/scannet_scene0050.txt
support=stream4d_scannet_scene0050_32f_ioc075_fixmem
strict=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
comp=stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_none_hard2
maskbp=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_maskbp_r002_hard2
complete=stream4d_v4_1_scene0050_128f_strict_supportcomplete_r002_ratio1
```

评估命令模板：

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m evaluation.evaluate \
  --pred_path "data/prediction/${out}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$out" \
  --no_class \
  2>&1 | tee "logs/${out}_eval.log"

cross="${out}_cross_32fsupport"
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list "$seq" \
  --pred-config "$out" \
  --pred-suffix _class_agnostic \
  --pre-points-config "$support" \
  --output-config "$cross" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${cross}_class_agnostic.txt" \
  --audit-root outputs/audit_v4_1_continue \
  2>&1 | tee "logs/${cross}.log"
```

### 18.4 containment suppression fusion 命令

第一层：`strict` 高置信 + `32f current` 低置信 coverage。

```bash
out=stream4d_v4_1_scene0050_strict_plus_32f_low_secioc0p50
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list "$seq" \
  --primary-config "$strict" \
  --secondary-config "$support" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score 0.2 \
  --drop-secondary-iou-threshold 0.50 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/stream4d_fusion_v4_1_continue \
  2>&1 | tee "logs/${out}_fuse.log"
```

第二层：加入 `compnone` 低置信 recall。

```bash
base=stream4d_v4_1_scene0050_strict_plus_32f_low_secioc0p50
out=stream4d_v4_1_scene0050_strict_32f_secioc0p50_plus_comp_low
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list "$seq" \
  --primary-config "$base" \
  --secondary-config "$comp" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score 0.1 \
  --preserve-primary-score \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/stream4d_fusion_v4_1_continue \
  2>&1 | tee "logs/${out}_fuse.log"
```

第三层：加入 `maskbp` 更低置信 recall。

```bash
base=stream4d_v4_1_scene0050_strict_32f_secioc0p50_plus_comp_low
out=stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list "$seq" \
  --primary-config "$base" \
  --secondary-config "$maskbp" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score 0.05 \
  --preserve-primary-score \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/stream4d_fusion_v4_1_continue \
  2>&1 | tee "logs/${out}_fuse.log"
```

实际运行的 containment 阈值：

```text
0.95
0.85
0.70
0.60
0.50
0.40
0.30
```

其中 0.50 / 0.40 / 0.30 的最终三层结果数值相同：

```text
0.253098 / 0.615832 / 0.802912
```

本轮保留 `stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005` 作为代表 config，因为它是第一次达到该 best 的配置。

### 18.5 support-completion 高置信层尝试

尝试把 previous recompute AP 很高的 support-completion 候选作为高置信层：

```bash
out=stream4d_v4_1_scene0050_complete002_plus_32f_low_secioc085
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list "$seq" \
  --primary-config "$complete" \
  --secondary-config "$support" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score 0.2 \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/stream4d_fusion_v4_1_continue \
  2>&1 | tee "logs/${out}_fuse.log"
```

继续加 strict / comp / maskbp 后的代表 config：

```text
stream4d_v4_1_scene0050_complete002_plus_32f_low_secioc085
stream4d_v4_1_scene0050_complete002_32f_secioc085_plus_strict09
stream4d_v4_1_scene0050_complete002_32f_strict09_plus_comp_low
stream4d_v4_1_scene0050_complete002_32f_strict09_comp_plus_mask005
```

结论：没有超过 strict-high 路线。

### 18.6 一次命名错误和修复

中间有一次命令失败：

```text
FileNotFoundError:
Missing prediction:
data/prediction/stream4d_v4_1_scene0050_strict_plus_32f_low_secioc085_class_agnostic/scene0050_00.npz
```

原因：

```text
实际生成 config 名是 secioc0p85。
后续命令误写为 secioc085，少了字符 "0p"。
```

处理：

```text
没有改任何结果文件。
用正确的 stream4d_v4_1_scene0050_strict_plus_32f_low_secioc0p85 继续运行剩余实验。
```

### 18.7 score calibration 命令与失败结果

命令模板：

```bash
inp=stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005
out=${inp}_scorelog_w010
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.rescore_prediction_scores \
  --root . \
  --seq-list "$seq" \
  --input-config "$inp" \
  --output-config "$out" \
  --score-feature log_area \
  --base-score-mode preserve \
  --tiebreaker-weight 0.01 \
  --summary-root outputs/stream4d_score_calibration_v4_1_continue \
  2>&1 | tee "logs/${out}_scorecal.log"
```

实际运行：

```text
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_scorelog_w001
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_scorelog_w010
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_scorelog_w050
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_scoresqrt_w010
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_scorearea_w010
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_scorelog_w010
stream4d_v4_1_scene0050_strict_32f_secioc0p70_plus_comp_low_scorelog_w010
```

代表失败结果：

```text
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_scorelog_w010_cross_32fsupport:
  0.190 / 0.596 / 0.860
```

解释：

```text
area/log_area tiebreaker 能把 AP25 推高，但 AP 明显下降。
说明大 mask 更容易粗匹配，但高 IoU 边界不干净，不能简单提前排序。
```

### 18.8 面积过滤命令与失败结果

命令模板：

```bash
inp=stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005
out=${inp}_minarea100
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.rescore_prediction_scores \
  --root . \
  --seq-list "$seq" \
  --input-config "$inp" \
  --output-config "$out" \
  --score-feature none \
  --base-score-mode preserve \
  --tiebreaker-weight 0.0 \
  --min-area 100 \
  --summary-root outputs/stream4d_score_calibration_v4_1_continue \
  2>&1 | tee "logs/${out}_scorecal.log"
```

实际运行 min-area：

```text
1, 5, 10, 20, 50, 100
```

代表结果：

```text
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_minarea100_cross_32fsupport:
  0.248 / 0.602 / 0.779
```

该结果没有超过未过滤版本，说明 full-scene area 过滤不能解决 fixed-support 中的错误排序和边界问题。

### 18.9 mask NMS 命令与失败结果

命令模板：

```bash
inp=stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005
out=${inp}_nmsminioc080
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" -u -m tools.nms_prediction_masks \
  --root . \
  --seq-list "$seq" \
  --input-config "$inp" \
  --output-config "$out" \
  --overlap-mode min_ioc \
  --overlap-threshold 0.80 \
  --tie-breaker original \
  --summary-root outputs/stream4d_mask_nms_v4_1_continue \
  2>&1 | tee "logs/${out}_nms.log"
```

实际运行：

```text
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_nmsminioc070
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_nmsminioc080
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_nmsminioc090
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_nmscandioc085
stream4d_v4_1_scene0050_strict_32f_secioc0p70_comp_plus_mask005_nmsiou050
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_nmsminioc080
stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_nmscandioc085
```

代表结果：

```text
min_ioc=0.80:
  fixed 32f = 0.230 / 0.514 / 0.646

iou=0.50:
  fixed 32f = 0.241 / 0.604 / 0.784
```

NMS 没有超过 best。它能略微提高部分 AP50/AP25，但会降低 AP，说明很多重叠候选并非纯重复，粗暴删除会损失 recall。

### 18.10 最终编译和回归测试

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" \
  -m py_compile stream4d/*.py tools/*.py tests/*.py

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" \
  -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
Ran 6 tests in 0.001s
OK
```

### 18.11 更新代码审阅压缩包

先更新文件清单：

```text
stream4d_v4_1_code_review_packet_filelist.txt
```

本轮新增纳入审阅包的文件：

```text
Stream3D/tools/nms_prediction_masks.py
Stream3D/tools/rescore_prediction_scores.py
```

## 19. 2026-06-08 继续推进：最终 prediction 上的 point-level WTA

### 19.1 本轮为什么做这个实验

上一轮 `new best` 的 GT 只读诊断说明：

```text
new best 有 337 个预测，明显多于 Stream3D 的 121 个预测。
new best 在 IoU>=0.5 时覆盖到 15 个 GT，Stream3D 覆盖到 18 个 GT。
new best 对同一个 GT 有重复预测，Stream3D 更接近一对一。
```

因此本轮不继续只堆工程，而是尝试一个算法上更直接的改动：

```text
如果一个三维点同时属于多个预测实例，
就让这个点只归属于一个优先级最高的预测实例。
```

这个策略叫 point-level WTA。WTA 是 Winner Takes All 的缩写，意思是“赢家拿走这个点”。这里的“赢家”不使用 GT，只用 prediction 自己的无监督信息，例如预测分数和预测面积。

### 19.2 新增代码

新增文件：

```text
Stream3D/tools/wta_prediction_points.py
```

这个工具做的事情：

```text
输入：
  data/prediction/{input_config}_class_agnostic/{scene}.npz

读取：
  pred_masks
  pred_score
  pred_classes

不读取：
  ScanNet GT
  object_dict
  evaluator 内部结果

处理：
  1. 找到被两个或更多预测同时占有的三维点。
  2. 对每个冲突点，按 priority_mode 选择一个赢家预测。
  3. 把这个点从其他输家预测中删掉。
  4. 可选删除被删空的预测实例。
  5. 复制输入配置的 TMP/pre_points，保证评估 support 仍然按同一套协议走。

输出：
  data/prediction/{output_config}_class_agnostic/{scene}.npz
  data/TMP/{output_config}/{scene}_pre_points.npy
  outputs/stream4d_point_wta_v4_1_continue/{output_config}_summary.json
```

### 19.3 编译检查

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

"$PY" -m py_compile tools/wta_prediction_points.py
```

结果：

```text
通过。
```

### 19.4 hard WTA 命令模板

基础配置：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

seq=splits/scannet_scene0050.txt
support=stream4d_scannet_scene0050_32f_ioc075_fixmem
inp=stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
```

生成 WTA prediction：

```bash
"$PY" -u -m tools.wta_prediction_points \
  --root . \
  --seq-list "$seq" \
  --input-config "$inp" \
  --output-config "$out" \
  --priority-mode "$mode" \
  --min-conflict-owners 2 \
  --drop-empty \
  --summary-root outputs/stream4d_point_wta_v4_1_continue
```

在原版 Stream3D 一致的 32f support 上评估：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list "$seq" \
  --pred-config "$out" \
  --pred-suffix _class_agnostic \
  --pre-points-config "$support" \
  --output-config "${out}_cross_32fsupport" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt" \
  --audit-root outputs/audit_v4_1_continue
```

### 19.5 hard WTA 实验结果

| config | priority mode | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| `stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005` | no WTA | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| `stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005_pointwta_score` | score | 0.24273504273504276 | 0.6548076923076923 | 0.7548076923076924 |
| `stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005_pointwta_area_desc` | score + tiny large-area tie break | 0.23888888888888887 | 0.612 | 0.8160000000000001 |
| `stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005_pointwta_area_asc` | score + tiny small-area tie break | 0.23422364672364676 | 0.6548076923076923 | 0.7548076923076924 |

和原版 Stream3D 对比：

```text
Stream3D on same 32f support:
AP   = 0.3911324786324787
AP50 = 0.6461538461538462
AP25 = 0.7615384615384615
```

本轮 hard WTA 的结论：

```text
score WTA 的 AP50 = 0.6548076923076923，已经超过 Stream3D 的 0.6461538461538462。
area_desc WTA 的 AP25 = 0.8160000000000001，已经超过 Stream3D 的 0.7615384615384615。
但是 hard WTA 的 AP 都下降，没有解决 AP 主指标。
```

### 19.6 WTA margin 实验

为了避免“分数差很小也强行删点”，加入 `--min-priority-margin`：

```bash
"$PY" -u -m tools.wta_prediction_points \
  --root . \
  --seq-list "$seq" \
  --input-config "$inp" \
  --output-config "$out" \
  --priority-mode score \
  --min-conflict-owners 2 \
  --min-priority-margin "$margin" \
  --drop-empty \
  --summary-root outputs/stream4d_point_wta_v4_1_continue
```

结果：

| config suffix | margin | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| `pointwta_score_margin001` | 0.001 | 0.23681353767560667 | 0.596551724137931 | 0.7862068965517243 |
| `pointwta_score_margin005` | 0.05 | 0.23681353767560667 | 0.596551724137931 | 0.7862068965517243 |
| `pointwta_score_margin010` | 0.10 | 0.23681353767560667 | 0.596551724137931 | 0.7862068965517243 |
| `pointwta_score_margin030` | 0.30 | 0.24018126668988737 | 0.6171434169278996 | 0.8063300026123302 |
| `pointwta_score_margin050` | 0.50 | 0.24018126668988737 | 0.6171434169278996 | 0.8063300026123302 |

结论：

```text
margin 能保留一部分冲突点，AP25 变好，但 AP/AP50 没有超过 hard score WTA 或 no-WTA best。
```

### 19.7 WTA high + original low 融合实验

因为 hard WTA 的 AP50 更好，但 AP/AP25 不稳定，所以尝试把 WTA 当作高分精度层，把原始 best 当作低分召回层：

```bash
orig=stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
wta_score=${orig}_pointwta_score

"$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list "$seq" \
  --primary-config "$wta_score" \
  --secondary-config "$orig" \
  --output-config stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001 \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score 0.01 \
  --preserve-primary-score \
  --summary-root outputs/stream4d_fusion_v4_1_continue

"$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list "$seq" \
  --pred-config stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001 \
  --pred-suffix _class_agnostic \
  --pre-points-config "$support" \
  --output-config stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001_cross_32fsupport \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit_v4_1_continue
```

也扫描了 secondary score 0.005/0.01/0.02/0.05，以及 `drop-secondary-iou-threshold 0.85`。

结果：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001` | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |
| `stream4d_v4_1_scene0050_wta_score_high_plus_orig_low0p005` | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |
| `stream4d_v4_1_scene0050_wta_score_high_plus_orig_low0p010` | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |
| `stream4d_v4_1_scene0050_wta_score_high_plus_orig_low0p020` | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |
| `stream4d_v4_1_scene0050_wta_score_high_plus_orig_low0p050` | 0.2530757451946083 | 0.6548076923076923 | 0.7973356854170808 |
| `stream4d_v4_1_scene0050_wta_score_high_plus_orig_low005_drop085` | 0.25236897185487567 | 0.6548076923076923 | 0.8004682782667858 |
| `stream4d_v4_1_scene0050_wta_score_high_plus_orig_low010_drop085` | 0.25261034570736063 | 0.6548076923076923 | 0.8016791044776119 |
| `stream4d_v4_1_scene0050_wta_area_high_plus_orig_low005_drop085` | 0.2510042735042735 | 0.6312307692307693 | 0.8406153846153847 |
| `stream4d_v4_1_scene0050_wta_area_high_plus_orig_low010_drop085` | 0.2513247863247863 | 0.6317692307692307 | 0.8413076923076923 |

### 19.8 WTA summary 证据

`pointwta_score` summary：

```text
num_instances_before = 337
num_instances_after = 299
num_conflict_points_before = 19950
num_conflict_points_after = 0
point_assignments_before = 68213
point_assignments_after = 40033
removed_point_assignments = 28180
union_count_before = 40033
union_count_after = 40033
empty_instances_after_wta = 38
```

`wta_score_high_plus_orig_low001` summary：

```text
num_primary_instances = 299
num_secondary_instances = 337
num_output_instances = 636
output_union_count = 40033
output_union_ratio = 0.18936548631543096
secondary_skipped_by_iou = 0
```

`wta_score_high_plus_orig_low010_drop085` summary：

```text
num_primary_instances = 299
num_secondary_instances = 233
num_output_instances = 532
secondary_skipped_by_iou = 104
secondary_max_overlap_mean = 0.6689417215392041
```

### 19.9 GT 只读诊断命令

下面诊断只读评估 GT，用于解释原因，不进入 prediction 生成：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" - <<'PY'
# 读取 data/prediction、data/TMP、data/scannet/gt，
# 统计每个 prediction 和 GT 的 best IoU，以及每个 GT 被多少 prediction 命中。
PY
```

诊断输出：

```text
Stream3D scannet_on_scene0050_32f_ioc075_fixmem_cross_audit
pred 121 nonempty 75 union_in_target 12186
best_iou>=0.25/0.5/0.75/0.8 19 18 8 6
gt>= 0.25: nonzero 19, mean_nonzero 1.00, max 1
gt>= 0.5: nonzero 18, mean_nonzero 1.00, max 1
gt>= 0.75: nonzero 8, mean_nonzero 1.00, max 1

contain_best stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005_cross_32fsupport
pred 337 nonempty 309 union_in_target 12084
best_iou>=0.25/0.5/0.75/0.8 37 29 8 5
gt>= 0.25: nonzero 18, mean_nonzero 2.06, max 4
gt>= 0.5: nonzero 15, mean_nonzero 1.93, max 4
gt>= 0.75: nonzero 6, mean_nonzero 1.33, max 3

wta_only stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005_pointwta_score_cross_32fsupport
pred 299 nonempty 231 union_in_target 12084
best_iou>=0.25/0.5/0.75/0.8 19 15 3 1
gt>= 0.25: nonzero 18, mean_nonzero 1.06, max 2
gt>= 0.5: nonzero 15, mean_nonzero 1.00, max 1
gt>= 0.75: nonzero 3, mean_nonzero 1.00, max 1

wta_high_orig_low stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001_cross_32fsupport
pred 636 nonempty 540 union_in_target 12084
best_iou>=0.25/0.5/0.75/0.8 56 44 11 6
gt>= 0.25: nonzero 19, mean_nonzero 2.95, max 5
gt>= 0.5: nonzero 15, mean_nonzero 2.93, max 5
gt>= 0.75: nonzero 6, mean_nonzero 1.83, max 4
```

### 19.10 本轮状态

达成：

```text
wta_score_high_plus_orig_low001 的 AP50 = 0.6548076923076923，
超过 Stream3D same support AP50 = 0.6461538461538462。

wta_score_high_plus_orig_low001 的 AP25 = 0.7973356854170808，
超过 Stream3D same support AP25 = 0.7615384615384615。
```

未达成：

```text
wta_score_high_plus_orig_low001 的 AP = 0.2530757451946083，
仍低于 Stream3D same support AP = 0.3911324786324787。
```

## 20. 2026-06-08 最终编译、测试和审阅包更新

### 20.1 编译和回归测试

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 1.949s

OK
```

### 20.2 更新代码审阅包清单

新增纳入审阅包的本轮文件：

```text
Stream3D/tools/wta_prediction_points.py
```

清单文件：

```text
stream4d_v4_1_code_review_packet_filelist.txt
```

### 20.3 重新打包和校验

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR

zip -q -@ stream4d_v4_1_code_review_packet.zip \
  < stream4d_v4_1_code_review_packet_filelist.txt

zip -T stream4d_v4_1_code_review_packet.zip

zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'wta_prediction_points|nms_prediction_masks|rescore_prediction_scores|fuse_prediction_configs|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

校验结果：

```text
test of stream4d_v4_1_code_review_packet.zip OK
Stream3D/tools/fuse_prediction_configs.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/nms_prediction_masks.py
Stream3D/tools/rescore_prediction_scores.py
Stream3D/tools/wta_prediction_points.py
```

## 21. 2026-06-08 继续推进：support-aware object ranker 与 Stream3D-primary hybrid

### 21.1 继续推进原因

上一轮结论：

```text
纯 Stream4D 的 AP 没有超过 Stream3D。
best pure Stream4D = 0.2530757451946083 / 0.6548076923076923 / 0.7973356854170808
Stream3D same 32f support = 0.3911324786324787 / 0.6461538461538462 / 0.7615384615384615
```

所以本轮继续按计划推进，不把 AP50/AP25 超过当成完成。

### 21.2 GT 只读特征诊断

注意：

```text
下面诊断只读 GT，用于分析，不进入任何 prediction 生成脚本。
```

先在 full scene 上算过一次，发现与正式 evaluator 不一致；原因是正式对比使用 `stream4d_scannet_scene0050_32f_ioc075_fixmem` target support 裁剪。随后按同一个 32f support 重新计算无监督特征与 best IoU 的相关性。

命令要点：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" - <<'PY'
# 读取：
#   data/prediction/{config}_class_agnostic/scene0050_00.npz
#   data/TMP/stream4d_scannet_scene0050_32f_ioc075_fixmem/scene0050_00_pre_points.npy
#   data/scannet/gt/scene0050_00.txt
#   data/scannet/processed/scene0050_00/scene0050_00_vh_clean_2.ply
# 只用于分析 score/log_area/compactness/conflict 与 target-support best IoU 的相关性。
PY
```

关键发现：

```text
config = stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
score corr_best_iou     = -0.040921953175370555
log_area corr_best_iou  =  0.6048175003122942
compact corr_best_iou   = -0.09791238225337083
conflict corr_best_iou  = -0.006497471521642615

config = stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001
score corr_best_iou     = -0.00038161448149909625
log_area corr_best_iou  =  0.6242080246260148
compact corr_best_iou   = -0.14989803606214683
```

解释：

```text
原始 score 与 target-support IoU 几乎不相关。
support 内有效面积与 target-support IoU 有较强相关。
因此尝试 support-aware object quality ranker。
```

### 21.3 新增代码

新增文件：

```text
Stream3D/tools/support_aware_object_rank.py
```

## 23. 2026-06-08 继续推进：纯 Stream4D self-primary sweep

### 23.1 本轮目的

上一轮 hybrid diagnostic 已经超过 Stream3D，但它使用了原版 Stream3D prediction 作为 primary，不是纯 Stream4D。用户追问 `inherit_pre_points` 下是否超过 Stream3D 后，本轮继续尝试把 hybrid 中有效的结构前移到 Stream4D 自身：

```text
primary:
  从纯 Stream4D 候选池中，用 support-aware quality 选出 top-N 高置信主层。

secondary:
  同一个纯 Stream4D 候选池以低分 0.01 作为 recall layer。

target support:
  stream4d_scannet_scene0050_32f_ioc075_fixmem
```

这轮不读取 GT 生成 prediction；只使用 prediction mask、score、target support 内面积和冲突比例。

### 23.2 输入候选池

```text
tier:
  stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005

contain:
  stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
```

### 23.3 第一次命令错误和修正

第一次 sweep 的评估命令错误地使用了旧参数名：

```bash
python -m tools.evaluate_cross_prepoints \
  --target-pre-points-config "$SUPPORT" \
  ...
```

当前仓库里的正确参数是：

```bash
python -m tools.evaluate_cross_prepoints \
  --pre-points-config "$SUPPORT" \
  ...
```

该错误在第一条评估前就退出：

```text
evaluate_cross_prepoints.py: error: the following arguments are required: --seq-list, --pre-points-config
```

随后第二次 sweep 虽然参数名修正，但遗漏 `--no-class`，导致 evaluator 按 ScanNet 类别名评价 class-agnostic prediction，出现全 0。该批全 0 文件不作为结果使用，并被下一轮正确 `--no-class` 评估覆盖。

### 23.4 正式 sweep 命令

工作目录：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
SEQ=splits/scannet_scene0050.txt
SUPPORT=stream4d_scannet_scene0050_32f_ioc075_fixmem
SUMMARY=outputs/stream4d_v4_1_self_primary_sweep
```

核心循环：

```bash
run_one() {
  local alias="$1"
  local input="$2"
  local maxn="$3"
  local drop="$4"
  local primary="stream4d_v4_1_scene0050_selfp_${alias}_scoreconf_w90_top${maxn}"
  local out="stream4d_v4_1_scene0050_selfp_${alias}_top${maxn}_origlow_drop${drop}"

  "$PY" -u -m tools.support_aware_object_rank \
    --seq-list "$SEQ" \
    --input-config "$input" \
    --output-config "$primary" \
    --score-pre-points-config "$SUPPORT" \
    --quality-mode score_support_area_conflict_penalty \
    --score-weight 0.90 \
    --preserve-empty-scores \
    --min-support-area 1 \
    --max-instances "$maxn" \
    --summary-root "$SUMMARY"

  if [ "$drop" = "0p85" ]; then
    "$PY" -u -m tools.fuse_prediction_configs \
      --seq-list "$SEQ" \
      --primary-config "$primary" \
      --secondary-config "$input" \
      --output-config "$out" \
      --preserve-primary-score \
      --secondary-score 0.01 \
      --drop-secondary-iou-threshold 0.85 \
      --drop-secondary-overlap-mode secondary_ioc \
      --summary-root "$SUMMARY"
  else
    "$PY" -u -m tools.fuse_prediction_configs \
      --seq-list "$SEQ" \
      --primary-config "$primary" \
      --secondary-config "$input" \
      --output-config "$out" \
      --preserve-primary-score \
      --secondary-score 0.01 \
      --summary-root "$SUMMARY"
  fi

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SEQ" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT" \
    --output-config "${out}_cross_32fsupport" \
    --no-class \
    --output-file "data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints \
    2>&1 | tee "logs/${out}_cross_32fsupport.log"
}

for maxn in 60 75 100 121 150 200; do
  for drop in nodrop 0p85; do
    run_one tier \
      stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005 \
      "$maxn" "$drop"
  done
done

for maxn in 60 75 100 121 150 200; do
  for drop in nodrop 0p85; do
    run_one contain \
      stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005 \
      "$maxn" "$drop"
  done
done
```

### 23.5 输出文件

结果文件：

```text
Stream3D/data/evaluation/scannet/stream4d_v4_1_scene0050_selfp_*_cross_32fsupport_class_agnostic.txt
```

summary：

```text
Stream3D/outputs/stream4d_v4_1_self_primary_sweep/*_summary.json
```

日志：

```text
Stream3D/logs/stream4d_v4_1_scene0050_selfp_*_cross_32fsupport.log
```

### 23.6 结果摘录

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
for f in data/evaluation/scannet/stream4d_v4_1_scene0050_selfp_*_cross_32fsupport_class_agnostic.txt; do
  printf '%s ' "$(basename "$f" _class_agnostic.txt)"
  tail -n 1 "$f"
done | sort -k2,2nr | head -n 30
```

前几名：

```text
stream4d_v4_1_scene0050_selfp_contain_top100_origlow_drop0p85_cross_32fsupport 0.20922297007534071,0.6041101243475138,0.8652214236379708
stream4d_v4_1_scene0050_selfp_contain_top121_origlow_drop0p85_cross_32fsupport 0.20922297007534071,0.6041101243475138,0.8652214236379708
stream4d_v4_1_scene0050_selfp_contain_top150_origlow_drop0p85_cross_32fsupport 0.20922297007534071,0.6041101243475138,0.8652214236379708
stream4d_v4_1_scene0050_selfp_contain_top200_origlow_drop0p85_cross_32fsupport 0.20922297007534071,0.6041101243475138,0.8652214236379708
stream4d_v4_1_scene0050_selfp_contain_top100_origlow_dropnodrop_cross_32fsupport 0.20796622893512498,0.6020155557804876,0.862568303453071
stream4d_v4_1_scene0050_selfp_tier_top100_origlow_drop0p85_cross_32fsupport 0.19993595479950874,0.5830513321429903,0.849504768013896
```

关键对照命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
for cfg in \
  scannet_on_scene0050_32f_ioc075_fixmem_cross_audit \
  stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001_cross_32fsupport \
  stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005_cross_32fsupport \
  stream4d_v4_1_scene0050_selfp_contain_top100_origlow_drop0p85_cross_32fsupport \
  stream4d_v4_1_scene0050_selfp_tier_top100_origlow_drop0p85_cross_32fsupport; do
  f="data/evaluation/scannet/${cfg}_class_agnostic.txt"
  printf '%s,' "$cfg"
  tail -n 1 "$f"
done
```

输出：

```text
scannet_on_scene0050_32f_ioc075_fixmem_cross_audit,0.39113247863247863,0.6461538461538462,0.7615384615384615
stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001_cross_32fsupport,0.25307574519460824,0.6548076923076923,0.7973356854170808
stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005_cross_32fsupport,0.2530976765890559,0.6158322281167109,0.8029115901631975
stream4d_v4_1_scene0050_selfp_contain_top100_origlow_drop0p85_cross_32fsupport,0.20922297007534071,0.6041101243475138,0.8652214236379708
stream4d_v4_1_scene0050_selfp_tier_top100_origlow_drop0p85_cross_32fsupport,0.19993595479950874,0.5830513321429903,0.849504768013896
```

## 24. 2026-06-08 继续推进：GT 只读候选池 oracle 上界诊断

### 24.1 本轮目的

第 23 节 self-primary 失败后，需要判断失败究竟来自哪里：

```text
可能 A：纯 Stream4D 候选池里没有足够好的实例，继续调排序无意义。
可能 B：候选池里有足够好的实例，但无监督排序 / 一对一分配没有选出来。
```

本轮新增一个只读 GT 诊断工具来计算候选池 oracle 上界。该工具生成的 `oracle_*` prediction 不能作为正式方法结果，只能用于失败原因分析。

### 24.2 新增工具

新增文件：

```text
Stream3D/tools/oracle_candidate_upper_bound.py
```

## 26. 2026-06-08 继续推进：object-level competition 原型

### 26.1 本轮目的

第 28 节 oracle 诊断显示：纯 Stream4D 候选池在 GT oracle 选择下有超过 Stream3D actual 的潜力，但当前无监督排序和一对一分配没有选出来。因此本轮不再做全局 top-N，而是尝试一个 object-level competition 原型：

```text
1. 在 target support 内按 prediction mask 重叠把候选分组。
2. 每组只选一个无监督质量分最高的代表。
3. 代表候选作为 high-confidence primary。
4. 原始候选池作为 low-confidence recall layer。
```

该方法不读取 GT，不改变 evaluator。

### 26.2 新增工具

新增：

```text
Stream3D/tools/object_competition_rank.py
```

使用的无监督信号：

```text
original score
support_area
unique_ratio
conflict_ratio
support_fraction
3D compactness proxy
3D bbox tightness proxy
```

其中 3D proxy 只读取 ScanNet mesh 点坐标：

```text
data/scannet/processed/<scene>/<scene>_vh_clean_2.ply
```

不读取 GT。

### 26.3 运行命令

工作目录：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
```

编译：

```bash
$PY -m py_compile tools/object_competition_rank.py
```

核心循环：

```bash
SEQ=splits/scannet_scene0050.txt
SUPPORT=stream4d_scannet_scene0050_32f_ioc075_fixmem
SUMMARY=outputs/stream4d_object_competition_v4_1_continue

run_one() {
  local alias="$1"
  local input="$2"
  local qmode="$3"
  local thr="$4"
  local tagthr=${thr/./p}
  local primary="stream4d_v4_1_scene0050_objcomp_${alias}_${qmode}_minioc${tagthr}"
  local fused="${primary}_plus_origlow"

  "$PY" -u -m tools.object_competition_rank \
    --seq-list "$SEQ" \
    --input-config "$input" \
    --output-config "$primary" \
    --score-pre-points-config "$SUPPORT" \
    --quality-mode "$qmode" \
    --group-overlap-mode min_ioc \
    --group-overlap-threshold "$thr" \
    --min-support-area 1 \
    --summary-root "$SUMMARY"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SEQ" \
    --pred-config "$primary" \
    --pre-points-config "$SUPPORT" \
    --output-config "${primary}_cross_32fsupport" \
    --no-class \
    --output-file "data/evaluation/scannet/${primary}_cross_32fsupport_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints \
    2>&1 | tee "logs/${primary}_cross_32fsupport.log"

  "$PY" -u -m tools.fuse_prediction_configs \
    --seq-list "$SEQ" \
    --primary-config "$primary" \
    --secondary-config "$input" \
    --output-config "$fused" \
    --preserve-primary-score \
    --secondary-score 0.01 \
    --drop-secondary-iou-threshold 0.85 \
    --drop-secondary-overlap-mode secondary_ioc \
    --summary-root "$SUMMARY"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SEQ" \
    --pred-config "$fused" \
    --pre-points-config "$SUPPORT" \
    --output-config "${fused}_cross_32fsupport" \
    --no-class \
    --output-file "data/evaluation/scannet/${fused}_cross_32fsupport_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints \
    2>&1 | tee "logs/${fused}_cross_32fsupport.log"
}
```

实际 sweep：

```bash
for qmode in score_unique_compact unique_compact_area score_compact area_unique compact_only; do
  for thr in 0.30 0.50 0.70; do
    run_one contain \
      stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005 \
      "$qmode" "$thr"
  done
done

for qmode in score_unique_compact unique_compact_area score_compact area_unique compact_only; do
  for thr in 0.30 0.50 0.70; do
    run_one tier \
      stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005 \
      "$qmode" "$thr"
  done
done
```

### 26.4 输出文件

结果文件：

```text
Stream3D/data/evaluation/scannet/stream4d_v4_1_scene0050_objcomp_*_cross_32fsupport_class_agnostic.txt
```

summary：

```text
Stream3D/outputs/stream4d_object_competition_v4_1_continue/*_summary.json
```

日志：

```text
Stream3D/logs/stream4d_v4_1_scene0050_objcomp_*_cross_32fsupport.log
```

### 26.5 结果摘录

排序命令：

```bash
for f in data/evaluation/scannet/stream4d_v4_1_scene0050_objcomp_*_cross_32fsupport_class_agnostic.txt; do
  printf '%s ' "$(basename "$f" _class_agnostic.txt)"
  tail -n 1 "$f"
done | sort -k2,2nr | head -n 40
```

前几名：

```text
stream4d_v4_1_scene0050_objcomp_tier_score_compact_minioc0p30_plus_origlow_cross_32fsupport 0.2522286821705426,0.4816860465116279,0.6188953488372093
stream4d_v4_1_scene0050_objcomp_tier_score_unique_compact_minioc0p30_plus_origlow_cross_32fsupport 0.2522286821705426,0.4816860465116279,0.6188953488372093
stream4d_v4_1_scene0050_objcomp_contain_score_unique_compact_minioc0p70_plus_origlow_cross_32fsupport 0.23604245134965474,0.5378248587570622,0.6752306967984935
stream4d_v4_1_scene0050_objcomp_tier_score_unique_compact_minioc0p70_plus_origlow_cross_32fsupport 0.2353182384064737,0.4785976890756303,0.6060142390289449
stream4d_v4_1_scene0050_objcomp_contain_score_compact_minioc0p30_plus_origlow_cross_32fsupport 0.23473791588198367,0.5262711864406779,0.678813559322034
```

关键 summary：

```text
stream4d_v4_1_scene0050_objcomp_tier_score_compact_minioc0p30:
  num_instances_before = 420
  num_valid_candidates = 385
  num_groups = 19
  num_selected = 19
  group_size_mean = 20.263157894736842
  group_size_max = 197
  output_union_count = 1080

stream4d_v4_1_scene0050_objcomp_contain_score_unique_compact_minioc0p70:
  num_instances_before = 337
  num_valid_candidates = 309
  num_groups = 129
  num_selected = 129
  group_size_mean = 2.395348837209302
  group_size_max = 38
  output_union_count = 8326
```

### 26.6 对照

```text
Stream3D same support:
  0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615

pure Stream4D previous AP best:
  0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975

object competition best:
  0.2522286821705426 / 0.4816860465116279 / 0.6188953488372093
```

功能：

```text
1. 读取 prediction config。
2. 读取 target pre_points support。
3. 读取 ScanNet GT，仅用于计算每个候选和 GT 的 IoU。
4. 用 GT oracle 贪心选择一批候选：
   - 每个 GT 最多选一个 prediction。
   - 每个 prediction 最多匹配一个 GT。
   - 默认只选 IoU >= 0.25 的候选。
5. 写出 oracle diagnostic prediction，score 等于对应 GT IoU。
```

重要边界：

```text
该工具读取 GT，所以输出不能计为方法结果。
它只用于判断候选池上界和失败原因。
```

### 24.3 oracle 工具修复

第一版 oracle 工具的 IoU 矩阵使用了 `uint8` 做矩阵乘法：

```python
gt_int = gt_masks.astype(np.uint8)
pred_int = pred_masks.astype(np.uint8)
intersections = gt_int @ pred_int
```

这会在 ScanNet 点数上发生静默溢出，导致 Stream3D oracle 只统计到 6 个 IoU>=0.5 的 GT，和已有只读诊断矛盾。该结果作废。

修复为：

```python
gt_int = gt_masks.astype(np.int64)
pred_int = pred_masks.astype(np.int64)
```

随后重新运行所有 oracle 结果，并覆盖旧输出。

### 24.4 运行命令

工作目录：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
```

编译：

```bash
$PY -m py_compile tools/oracle_candidate_upper_bound.py
```

核心函数：

```bash
SEQ=splits/scannet_scene0050.txt
SUPPORT=stream4d_scannet_scene0050_32f_ioc075_fixmem
SUMMARY=outputs/oracle_candidate_upper_bound_v4_1_continue

run_oracle() {
  local alias="$1"
  local cfg="$2"
  local out="stream4d_v4_1_oracle_${alias}_scene0050_min025"

  "$PY" -u -m tools.oracle_candidate_upper_bound \
    --seq-list "$SEQ" \
    --pred-config "$cfg" \
    --pre-points-config "$SUPPORT" \
    --output-config "$out" \
    --min-select-iou 0.25 \
    --summary-root "$SUMMARY"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SEQ" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT" \
    --output-config "${out}_cross_32fsupport" \
    --no-class \
    --output-file "data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints \
    2>&1 | tee "logs/${out}_cross_32fsupport.log"
}
```

实际运行：

```bash
run_oracle s3d scannet
run_oracle contain stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
run_oracle wta_orig stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001
run_oracle tier stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
run_oracle selfp_contain stream4d_v4_1_scene0050_selfp_contain_top100_origlow_drop0p85
```

### 24.5 输出文件

summary：

```text
Stream3D/outputs/oracle_candidate_upper_bound_v4_1_continue/*_oracle_upper_bound_summary.json
```

oracle prediction：

```text
Stream3D/data/prediction/stream4d_v4_1_oracle_*_scene0050_min025_class_agnostic/scene0050_00.npz
```

评估结果：

```text
Stream3D/data/evaluation/scannet/stream4d_v4_1_oracle_*_cross_32fsupport_class_agnostic.txt
```

### 24.6 修复后结果摘录

评估结果：

```text
stream4d_v4_1_oracle_s3d_scene0050_min025_cross_32fsupport 0.5277777777777779,0.8000000000000002,0.9000000000000001
stream4d_v4_1_oracle_tier_scene0050_min025_cross_32fsupport 0.4166666666666667,0.7500000000000001,0.9500000000000001
stream4d_v4_1_oracle_contain_scene0050_min025_cross_32fsupport 0.4055555555555556,0.7500000000000001,0.9500000000000001
stream4d_v4_1_oracle_selfp_contain_scene0050_min025_cross_32fsupport 0.4055555555555556,0.7500000000000001,0.9500000000000001
stream4d_v4_1_oracle_wta_orig_scene0050_min025_cross_32fsupport 0.4055555555555556,0.7500000000000001,0.9500000000000001
```

oracle summary 关键字段：

```text
Stream3D oracle:
  gt_best_iou>=0.25 = 18
  gt_best_iou>=0.50 = 16
  gt_best_iou>=0.75 = 9
  gt_best_iou>=0.80 = 6
  gt_best_iou>=0.90 = 4

Stream4D tier oracle:
  gt_best_iou>=0.25 = 19
  gt_best_iou>=0.50 = 15
  gt_best_iou>=0.75 = 8
  gt_best_iou>=0.80 = 6
  gt_best_iou>=0.90 = 1

Stream4D contain / wta_orig / selfp_contain oracle:
  gt_best_iou>=0.25 = 19
  gt_best_iou>=0.50 = 15
  gt_best_iou>=0.75 = 7
  gt_best_iou>=0.80 = 5
  gt_best_iou>=0.90 = 1
```

该工具不读取 GT。它读取：

```text
1. prediction npz:
   data/prediction/{input_config}_class_agnostic/{scene}.npz

2. 用于打分的 support pre_points:
   data/TMP/{score_pre_points_config}/{scene}_pre_points.npy
```

核心逻辑：

```text
support_area = 一个 prediction 在指定 support 内覆盖了多少点
conflict_ratio = 这些 support 点中有多少同时被多个 prediction 占有
unique_ratio = 这些 support 点中有多少只被当前 prediction 占有

quality = score_weight * original_score_norm
          + (1 - score_weight) * support_area_conflict_penalty
```

其中本轮最有效的模式：

```text
quality_mode = score_support_area_conflict_penalty
score_weight = 0.90
```

### 21.4 编译检查

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 "$PY" \
  -m py_compile tools/support_aware_object_rank.py
```

结果：

```text
通过。
```

### 21.5 现有全 scene 面积主排序对照

先用已有 `tools.rescore_prediction_scores` 做全 scene 面积/log_area 主排序，确认不能把“面积相关性”误解成“全局越大越好”。

命令模板：

```bash
"$PY" -u -m tools.rescore_prediction_scores \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --input-config "$inp" \
  --output-config "$out" \
  --score-feature "$feat" \
  --base-score-mode constant \
  --constant-score 0.0 \
  --tiebreaker-weight 1.0 \
  --summary-root outputs/stream4d_score_calibration_v4_1_continue

"$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$out" \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config "${out}_cross_32fsupport" \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt" \
  --audit-root outputs/audit_v4_1_continue
```

结果：

| input | feature | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| containment best | log_area | 0.07621178702317448 | 0.24610383178473408 | 0.5153087099230065 |
| containment best | area | 0.07621178702317448 | 0.24610383178473408 | 0.5153087099230065 |
| containment best | sqrt_area | 0.07621178702317448 | 0.24610383178473408 | 0.5153087099230065 |
| WTA high + original low | log_area | 0.05830335737419485 | 0.19742732328590878 | 0.4360707317665511 |
| WTA high + original low | area | 0.05830335737419485 | 0.19742732328590878 | 0.4360707317665511 |
| WTA high + original low | sqrt_area | 0.05830335737419485 | 0.19742732328590878 | 0.4360707317665511 |

结论：

```text
全 scene 面积主排序失败。
support 内有效面积只能作为局部/tier 内质量信号，不能直接替代原 ranking。
```

### 21.6 纯 Stream4D support-aware ranker 结果

命令模板：

```bash
"$PY" -u -m tools.support_aware_object_rank \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --input-config "$inp" \
  --output-config "$out" \
  --score-pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --quality-mode "$mode" \
  --score-weight "$wt" \
  --preserve-empty-scores \
  --summary-root outputs/stream4d_support_aware_rank_v4_1_continue
```

代表结果：

| input | mode | weight | AP | AP50 | AP25 |
|---|---|---:|---:|---:|---:|
| containment best | `score_support_area_conflict_penalty` | 0.90 | 0.21026762841408636 | 0.6073206989976491 | 0.8589401971350655 |
| containment best | `score_support_area` | 0.90 | 0.19498799453773902 | 0.6040452040649409 | 0.8704656386016244 |
| WTA high + original low | `score_support_area_conflict_penalty` | 0.90 | 0.19385421894160873 | 0.6381181391707709 | 0.84292613891956 |

加 object-level overlap competition 后的代表结果：

| input | mode | setting | AP | AP50 | AP25 |
|---|---|---|---:|---:|---:|
| containment best | scoreconf | min_ioc 0.85 | 0.1918326118326118 | 0.5171879509379509 | 0.6979503829503831 |
| containment best | scoreconf | top320 | 0.21026762841408636 | 0.6073206989976491 | 0.8589401971350655 |
| WTA high + original low | scoreconf | min_ioc 0.85 | 0.18371327973520957 | 0.6381181391707709 | 0.79151740876083 |
| WTA high + original low | scoreconf | top320 | 0.19385421894160873 | 0.6381181391707709 | 0.84292613891956 |

结论：

```text
纯 Stream4D support-aware ranker 没有超过 pure best。
它能提高 AP25，但 AP 主指标仍低，说明无监督面积/冲突 proxy 不能单独解决高 IoU instance quality。
```

### 21.7 Stream3D-primary + Stream4D recall-layer hybrid

根据计划中的 Stream3D inherit / cross-fixed 诊断，本轮额外尝试一个明确标注为 hybrid diagnostic 的算法：

```text
primary: 原版 Stream3D prediction `scannet`
secondary: Stream4D prediction，放在低分 recall layer
评估: 仍然使用 same 32f support
```

这不是纯 Stream4D 结果，不能写成纯 Stream4D 超过 Stream3D。

基础 fusion 命令模板：

```bash
"$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config scannet \
  --secondary-config "$sec" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score "$score" \
  --preserve-primary-score \
  --summary-root outputs/stream4d_fusion_v4_1_continue
```

代表结果：

| secondary | low score | drop secondary IOC | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| containment best | 0.001 | none | 0.40606711915535443 | 0.6665384615384615 | 0.7844343891402714 |
| containment best | 0.001 | 0.85 | 0.40421296296296305 | 0.668621794871795 | 0.7867628205128205 |
| WTA high + original low | 0.001 | none | 0.4051453754578755 | 0.6653331043956044 | 0.7830496936691627 |
| inherit tiered | 0.001 | none | 0.4073183760683761 | 0.6653331043956044 | 0.7830872252747253 |
| inherit tiered | 0.001 | 0.85 | 0.40700854700854705 | 0.6672051282051282 | 0.7851794871794872 |

### 21.8 Stream4D recall layer 内部 support-aware 排序

为了让 secondary recall layer 内部不是全同分，先对 Stream4D secondary 做 support-aware quality 排序：

```bash
ranked=stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_saware_scoreconf_w0p90_for_hybrid

"$PY" -u -m tools.support_aware_object_rank \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --input-config stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005 \
  --output-config "$ranked" \
  --score-pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --quality-mode score_support_area_conflict_penalty \
  --score-weight 0.90 \
  --preserve-empty-scores \
  --summary-root outputs/stream4d_support_aware_rank_v4_1_continue
```

再融合：

```bash
"$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config scannet \
  --secondary-config "$ranked" \
  --output-config stream4d_v4_1_scene0050_hybrid_s3d_primary_${ranked}_drop0p85 \
  --fusion-mode concatenate \
  --primary-score 2.0 \
  --secondary-score 0.01 \
  --preserve-secondary-score \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/stream4d_fusion_v4_1_continue
```

评估：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config stream4d_v4_1_scene0050_hybrid_s3d_primary_stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_saware_scoreconf_w0p90_for_hybrid_drop0p85 \
  --pred-suffix _class_agnostic \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config stream4d_v4_1_scene0050_hybrid_s3d_primary_stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_saware_scoreconf_w0p90_for_hybrid_drop0p85_cross_32fsupport \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_v4_1_scene0050_hybrid_s3d_primary_stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_saware_scoreconf_w0p90_for_hybrid_drop0p85_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit_v4_1_continue
```

权重和 drop 阈值扫描：

| secondary weight | drop secondary IOC | AP | AP50 | AP25 |
|---:|---:|---:|---:|---:|
| 0.10 | 0.70 | 0.4041174874508208 | 0.6693879731379732 | 0.7875900488400488 |
| 0.25 | 0.85 | 0.40994783465608686 | 0.6741136162687886 | 0.7928890362511052 |
| 0.50 | 0.85 | 0.4115513657637674 | 0.6772792022792022 | 0.7964387464387463 |
| 0.75 | 0.85 | 0.4115944302117688 | 0.6772792022792022 | 0.7964387464387463 |
| 0.90 | 0.85 | 0.4115944302117688 | 0.6772792022792022 | 0.7964387464387463 |
| 0.99 | 0.85 | 0.4115944302117688 | 0.6772792022792022 | 0.7964387464387463 |
| 0.90 | 0.95 | 0.41042623242151277 | 0.6741136162687886 | 0.7928890362511052 |

本轮最佳：

```text
stream4d_v4_1_scene0050_hybrid_s3d_primary_stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_saware_scoreconf_w0p90_for_hybrid_drop0p85_cross_32fsupport

AP   = 0.4115944302117688
AP50 = 0.6772792022792022
AP25 = 0.7964387464387463
```

相对 Stream3D same support：

```text
Stream3D AP   = 0.3911324786324787
Stream3D AP50 = 0.6461538461538462
Stream3D AP25 = 0.7615384615384615

hybrid AP gain   = +0.02046195157929014
hybrid AP50 gain = +0.03112535612535605
hybrid AP25 gain = +0.034900284900284774
```

### 21.9 best hybrid summary

Support-aware rank summary：

```text
input_config = stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
output_config = stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_saware_scoreconf_w0p90_for_hybrid
quality_mode = score_support_area_conflict_penalty
score_weight = 0.9
num_instances_before = 420
num_instances_after_competition = 420
num_score_support_points = 12214
support_area_mean = 81.96904761904761
support_area_max = 1465
quality_min = -1.0
quality_mean = 0.13523773849010468
quality_max = 0.9430736899375916
```

Fusion summary：

```text
primary_config = scannet
secondary_config = stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005_saware_scoreconf_w0p90_for_hybrid
drop_secondary_iou_threshold = 0.85
drop_secondary_overlap_mode = secondary_ioc
num_primary_instances = 121
num_secondary_instances_after_drop = 243
num_output_instances = 364
secondary_skipped_by_iou = 177
secondary_max_overlap_mean = 0.7772511508748526
```

Cross-prepoints audit：

```text
target_pre_points_config = stream4d_scannet_scene0050_32f_ioc075_fixmem
target_pre_points_ratio = 0.05777508679980701
prediction_union_in_target_ratio_of_target = 0.9995906336990339
num_gt_instances_in_target_pre_points = 21
num_pred_instances = 364
mask_shape_mode = full_scene
expanded_prediction_scenes = 0
```

### 21.10 GT 只读诊断：hybrid 为什么能超过

只读诊断结果：

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

Secondary 贡献：

```text
hybrid secondary nonempty = 224
secondary best_iou>=0.25 = 26
secondary best_iou>=0.5  = 21
secondary best_iou>=0.75 = 3
```

解释：

```text
hybrid 的提升来自两部分：
1. Stream3D primary 保留高质量一对一基础。
2. Stream4D secondary 补了一部分 Stream3D 没覆盖好的 GT，并在 support-aware 排序后排在 secondary 层更前。

这证明 Stream4D 的 recall layer 有补充价值。
但它不证明纯 Stream4D object memory 已经超过 Stream3D。
```

## 25. 2026-06-08 最新验证与审阅包更新

### 25.1 编译和回归测试

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
```

### 25.2 新增纳入审阅包的文件

```text
Stream3D/tools/support_aware_object_rank.py
Stream3D/tools/oracle_candidate_upper_bound.py
```

## 27. 2026-06-08 审计补充：object-level competition 结果归档

说明：

```text
第 26 节的 object-level competition 命令和结果在本日志中插入到了第 24 节 oracle 诊断中间。
为避免后续复现者误读，本节在文件末尾重新归档该实验的关键命令、输出和结论。
本节没有新增实验数据，所有数值均来自第 26 节已经执行并落盘的结果文件。
```

新增工具：

```text
Stream3D/tools/object_competition_rank.py
```

核心输入：

```text
seq_list = Stream3D/splits/scannet_scene0050.txt
target support = stream4d_scannet_scene0050_32f_ioc075_fixmem
input A = stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
input B = stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
quality modes = score_unique_compact / unique_compact_area / score_compact / area_unique / compact_only
group overlap thresholds = 0.30 / 0.50 / 0.70
```

关键命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

"$PY" -u -m tools.object_competition_rank \
  --seq-list splits/scannet_scene0050.txt \
  --input-config <input_config> \
  --output-config <primary_config> \
  --score-pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --quality-mode <quality_mode> \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold <threshold> \
  --min-support-area 1 \
  --summary-root outputs/stream4d_object_competition_v4_1_continue

"$PY" -u -m tools.fuse_prediction_configs \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config <primary_config> \
  --secondary-config <input_config> \
  --output-config <primary_config>_plus_origlow \
  --preserve-primary-score \
  --secondary-score 0.01 \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/stream4d_object_competition_v4_1_continue

"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config <eval_config> \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config <eval_config>_cross_32fsupport \
  --no-class \
  --output-file data/evaluation/scannet/<eval_config>_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit_v4_1_continue/cross_prepoints
```

结果文件：

```text
Stream3D/data/evaluation/scannet/stream4d_v4_1_scene0050_objcomp_*_cross_32fsupport_class_agnostic.txt
Stream3D/outputs/stream4d_object_competition_v4_1_continue/*_summary.json
Stream3D/logs/stream4d_v4_1_scene0050_objcomp_*_cross_32fsupport.log
```

结果数量：

```text
summary json = 60
evaluation txt = 60
```

前五名结果：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `objcomp_tier_score_compact_minioc0p30_plus_origlow` | 0.2522286821705426 | 0.4816860465116279 | 0.6188953488372093 |
| `objcomp_tier_score_unique_compact_minioc0p30_plus_origlow` | 0.2522286821705426 | 0.4816860465116279 | 0.6188953488372093 |
| `objcomp_contain_score_unique_compact_minioc0p70_plus_origlow` | 0.23604245134965474 | 0.5378248587570622 | 0.6752306967984935 |
| `objcomp_tier_score_unique_compact_minioc0p70_plus_origlow` | 0.2353182384064737 | 0.4785976890756303 | 0.6060142390289449 |
| `objcomp_contain_score_compact_minioc0p30_plus_origlow` | 0.23473791588198367 | 0.5262711864406779 | 0.678813559322034 |

对照：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| pure Stream4D previous AP best | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| object competition best | 0.2522286821705426 | 0.4816860465116279 | 0.6188953488372093 |

关键 summary：

```text
stream4d_v4_1_scene0050_objcomp_tier_score_compact_minioc0p30:
  mean_num_instances_before = 420.0
  mean_num_valid_candidates = 385.0
  mean_num_groups = 19.0
  mean_num_selected = 19.0
  mean_group_size_mean = 20.263157894736842
  mean_group_size_max = 197.0
  mean_output_union_count = 1080.0
  mean_mean_selected_conflict_ratio = 0.6913632442366712

stream4d_v4_1_scene0050_objcomp_contain_score_unique_compact_minioc0p70:
  mean_num_instances_before = 337.0
  mean_num_valid_candidates = 309.0
  mean_num_groups = 129.0
  mean_num_selected = 129.0
  mean_group_size_mean = 2.395348837209302
  mean_group_size_max = 38.0
  mean_output_union_count = 8326.0
```

结论：

```text
object-level competition 没有超过 Stream3D，也没有超过此前纯 Stream4D best。
较低 group threshold 会把大量互补候选合成少数大组，primary recall 严重不足。
较高 threshold 保留更多候选，但重复和边界质量问题仍在。
这说明当前无监督 overlap grouping + compactness/unique quality 还不能替代 oracle 所需的一对一选择。
```

清单文件已更新：

```text
stream4d_v4_1_code_review_packet_filelist.txt
```

### 25.3 重新打包和校验

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR

zip -q -@ stream4d_v4_1_code_review_packet.zip \
  < stream4d_v4_1_code_review_packet_filelist.txt

zip -T stream4d_v4_1_code_review_packet.zip

zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'oracle_candidate_upper_bound|support_aware_object_rank|wta_prediction_points|nms_prediction_masks|rescore_prediction_scores|fuse_prediction_configs|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

校验结果：

```text
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/wta_prediction_points.py
Stream3D/tools/support_aware_object_rank.py
Stream3D/tools/oracle_candidate_upper_bound.py
```

## 28. 2026-06-08 最新编译、测试和审阅包校验

### 28.1 编译和回归测试

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
```

### 28.2 审阅包更新

新增纳入审阅清单：

```text
Stream3D/tools/object_competition_rank.py
```

重新打包：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'object_competition_rank|oracle_candidate_upper_bound|support_aware_object_rank|wta_prediction_points|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

校验结果：

```text
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/wta_prediction_points.py
Stream3D/tools/support_aware_object_rank.py
Stream3D/tools/oracle_candidate_upper_bound.py
Stream3D/tools/object_competition_rank.py
```

## 29. 2026-06-08 继续推进：greedy support set-cover selection 原型

### 29.1 本轮目的

第 28 节 oracle 和第 27 节 object competition 说明：

```text
纯 Stream4D 候选池有 oracle 潜力；
但按 overlap 分组再选代表会过度合并互补候选。
```

因此本轮尝试一个不同的无监督 object-level selection：

```text
每一步选择能给 target support 带来最多新增解释、同时冲突较少的候选。
```

这相当于 set-cover / novelty selection，不读取 GT，不改变 evaluator。

### 29.2 新增工具

新增：

```text
Stream3D/tools/greedy_support_select.py
```

## 33. 2026-06-08 继续推进：selection_quality score mode 负例

### 33.1 目的

第 31 节说明 `coverage_component_density` 能生成较干净 core，但 fixed support 仍低。本轮继续测试：

```text
不改 mask，只把 component_densify 的 mask-selection 质量接入 pred_score。
如果排序是主要瓶颈，selection_quality 应该改善 AP。
```

### 33.2 代码修改

修改：

```text
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/export_scannet.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/reexport_scannet.py
```

新增：

```text
export_score_mode = selection_quality
```

定义：

```text
selection_quality =
  densify_selection_selected_score_mean
  * densify_observations_used
  * sqrt(num_output_points)
```

不读取 GT。

### 33.3 运行命令

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

cfg=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_sel_coverage_component_density_scoreselect_hard2

"$PY" -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 128 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.79 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-support-mode component_densify \
  --export-core-nn-radius 0.03 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 8 \
  --export-mask-min-relative-coverage 0.95 \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 16 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode none \
  --densify-mask-selection-mode coverage_component_density \
  --export-score-mode selection_quality \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 2 \
  --input-debug-root outputs/stream4d_debug_scene0050_128f_ioc075_fixmem \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1

"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$cfg" \
  --pre-points-config "$cfg" \
  --output-config "${cfg}_self_audit" \
  --no-class \
  --output-file "data/evaluation/scannet/${cfg}_self_audit_class_agnostic.txt" \
  --audit-root outputs/audit_v4_1_continue/cross_prepoints

"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$cfg" \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config "${cfg}_cross_32fsupport" \
  --no-class \
  --output-file "data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt" \
  --audit-root outputs/audit_v4_1_continue/cross_prepoints
```

### 33.4 结果

| config | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|
| 128f coverage_component_density, score one | 0.43833943833943834 | 0.5686813186813187 | 0.7554945054945055 | 0.13645833333333332 | 0.26249999999999996 | 0.26249999999999996 |
| 128f coverage_component_density, selection_quality | 0.2454124223354993 | 0.3350218157910465 | 0.655195018656557 | 0.11000661375661375 | 0.20113095238095235 | 0.20113095238095235 |

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

### 33.5 结论

```text
selection_quality 不能作为 AP ranking。
```

解释：

```text
1. mask-selection quality 能辅助选择 mask view，但它不等价于 instance ranking。
2. 同一批 output points 在 selection_quality 排序后 AP 明显下降。
3. 这进一步证明：当前缺少的是 object-level 一对一 assignment 和边界质量校准，而不是简单把内部质量分接到 pred_score。
```

## 34. 2026-06-08 最终编译、测试和审阅包校验

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'reliable_densifier|export_scannet|replay_evidence_graph|reexport_scannet|greedy_support_select|object_competition_rank|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
test of stream4d_v4_1_code_review_packet.zip OK
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/reexport_scannet.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/export_scannet.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/object_competition_rank.py
Stream3D/tools/greedy_support_select.py
```

## 35. 2026-06-08 继续推进：local proposal bank 与 same-frame best-mask 约束

### 35.1 本轮目的

此前 fused prediction 没有保存 `object_dict.npy`，因此无法在事后对每个 fused object 做 projection agreement。继续检查 debug 文件后发现：

```text
outputs/stream4d_debug_scene0050_128f_ioc075_fixmem/scene0050_00/local_props_window*.json
```

包含每个 local proposal 的 `mask_observations`。因此本轮从 memory 合并前的 local proposals 重新生成一个 proposal bank，并加入一个更靠前的算法约束：

```text
same-frame best mask only:
  如果同一个 local proposal 在同一帧关联多个 2D mask，
  只保留 coverage 最高的 mask。
```

该步骤不读取 GT。

### 35.2 新增工具

新增：

```text
Stream3D/tools/export_local_proposal_bank.py
```

## 37. 2026-06-08 继续推进：single 2D mask observation bank 负例

### 37.1 本轮目的

第 35 节 local proposal bank 有轻微 recall 收益，但单独质量不强。为了判断污染来自哪里，本轮继续拆开：

```text
不把多个 2D masks 合成 local proposal；
每个唯一 (frame_id, mask_id) 2D mask observation 直接作为一个 3D candidate。
```

如果单帧 2D mask 本身回投已经足够好，这个 bank 应该有可见 AP；如果它几乎为 0，说明必须依赖多帧/3D object formation，不能直接拿单帧 mask 当 3D instance。

### 37.2 新增工具

新增：

```text
Stream3D/tools/export_mask_observation_bank.py
```

功能：

```text
1. 读取 local_props_window*.json。
2. 去重得到唯一 (frame_id, mask_id)。
3. 用 coverage 作为 pred_score。
4. 将每个 2D mask backproject 成一个 3D candidate。
```

### 37.3 运行命令

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

DEBUG=outputs/stream4d_debug_scene0050_128f_ioc075_fixmem

for cov in 0.005 0.010 0.020; do
  cfg="stream4d_v4_1_scene0050_maskobs128_cov${tag}"
  "$PY" -u -m tools.export_mask_observation_bank \
    --debug-root "$DEBUG" \
    --seq-name scene0050_00 \
    --output-config "$cfg" \
    --min-coverage "$cov" \
    --export-nn-radius 0.05 \
    --min-points-per-mask 100 \
    --summary-root outputs/mask_observation_bank_v4_1_continue

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$cfg" \
    --pre-points-config "$cfg" \
    --output-config "${cfg}_self_audit" \
    --no-class \
    --output-file "data/evaluation/scannet/${cfg}_self_audit_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$cfg" \
    --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
    --output-config "${cfg}_cross_32fsupport" \
    --no-class \
    --output-file "data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints
done
```

### 37.4 结果

| config | exported obs | union points | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cov0.005 | 604 | 9902 | 0.0035222526293306766 | 0.010957908456653464 | 0.023769305138030154 | 0.0018514390350846707 | 0.0033304900744416874 | 0.0033304900744416874 |
| cov0.010 | 349 | 4208 | 0.00047857608341479306 | 0.001466275659824047 | 0.028877378730751163 | 0.0 | 0.0 | 0.0 |
| cov0.020 | 155 | 1748 | 0.0 | 0.0 | 0.16666666666666669 | 0.0 | 0.0 | 0.0 |

summary:

```text
raw_unique_observations = 1609

cov0.005:
  filtered_observations = 628
  exported_observations = 604
  dropped_small = 24
  union_points = 9902
  hit_rate = 0.9369147668881068
  score min/mean/max = 0.005001163110136986 / 0.015430375933647156 / 0.07459677755832672
```

### 37.5 作为 low-confidence recall layer

将 `cov0.005` 加到此前 pure best：

```text
stream4d_v4_1_scene0050_purebest_plus_maskobs_cov0p005_drop0p85:
0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975
```

这与 primary 完全相同，没有收益。

fusion summary：

```text
primary instances = 337
secondary kept instances = 430
secondary skipped by IoU = 174
secondary union count = 8056
output union count = 42716
```

### 37.6 结论

```text
single 2D mask observation bank 是强负例。
```

解释：

```text
1. 虽然 backprojection hit_rate 很高，单帧 mask 的 3D candidate 仍几乎没有 AP。
2. 这说明单帧 2D mask 不是一个合格的 3D instance object。
3. 低分融合也完全没有收益，说明这些单帧候选不能有效补 recall。
4. 必须靠多帧/3D object formation，而不是直接把 2D masks 当作 3D objects。
```

## 38. 2026-06-08 最终编译、测试和审阅包校验

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'export_mask_observation_bank|export_local_proposal_bank|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/export_local_proposal_bank.py
Stream3D/tools/export_mask_observation_bank.py
```

功能：

```text
1. 读取 Stream4D debug local_props_window*.json。
2. 将每个 local proposal 转成一个 object hypothesis。
3. 可选 same-frame best_per_frame 过滤同帧冲突 mask。
4. 用 ScanNetExporter 的 mask_backproject 路径导出 prediction/TMP。
```

### 35.3 单独 localbank 运行命令

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

DEBUG=outputs/stream4d_debug_scene0050_128f_ioc075_fixmem

for minobs in 3 5 8; do
  cfg="stream4d_v4_1_scene0050_localbank128_bestframe_minobs${minobs}"
  "$PY" -u -m tools.export_local_proposal_bank \
    --debug-root "$DEBUG" \
    --seq-name scene0050_00 \
    --output-config "$cfg" \
    --same-frame-policy best_per_frame \
    --min-observations "$minobs" \
    --min-frames "$minobs" \
    --export-nn-radius 0.05 \
    --export-mask-sample-stride 2 \
    --export-mask-max-pixels 12000 \
    --export-max-masks-per-object 5 \
    --export-mask-min-relative-coverage 0.0 \
    --export-min-points-per-object 100 \
    --export-score-mode observations \
    --summary-root outputs/local_proposal_bank_v4_1_continue

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$cfg" \
    --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
    --output-config "${cfg}_cross_32fsupport" \
    --no-class \
    --output-file "data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints
done
```

### 35.4 单独 localbank 结果

| config | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|
| localbank minobs3 | 0.15921681199458976 | 0.444988344988345 | 0.534375 | 0.18549019607843137 | 0.3747058823529412 | 0.5487804878048781 |
| localbank minobs5 | 0.19755826157328377 | 0.46193240013656534 | 0.5614328757951121 | 0.191139846743295 | 0.39784482758620693 | 0.5540178571428571 |
| localbank minobs8 | 0.21547237984383336 | 0.484941073766914 | 0.5939655172413794 | 0.18339646464646464 | 0.3818181818181818 | 0.5089285714285714 |

summary：

```text
minobs3:
  raw_local_proposals = 805
  kept_local_proposals = 194
  dropped_local_proposals = 611
  same_frame_conflicts_removed = 226
  exported objects = 144
  exported points = 58568

minobs5:
  raw_local_proposals = 805
  kept_local_proposals = 126
  dropped_local_proposals = 679
  same_frame_conflicts_removed = 226
  exported objects = 102
  exported points = 54223

minobs8:
  raw_local_proposals = 805
  kept_local_proposals = 92
  dropped_local_proposals = 713
  same_frame_conflicts_removed = 226
  exported objects = 80
  exported points = 48854
```

### 35.5 localbank 作为 low-confidence recall layer

命令模板：

```bash
"$PY" -u -m tools.fuse_prediction_configs \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config <pure_stream4d_primary> \
  --secondary-config stream4d_v4_1_scene0050_localbank128_bestframe_minobs5 \
  --output-config <output_config> \
  --preserve-primary-score \
  --secondary-score 0.005 \
  --drop-secondary-iou-threshold <0.50|0.85> \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/local_proposal_bank_v4_1_continue
```

结果：

| primary | drop | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| pure AP best | 0.50 | 0.25837936672990097 | 0.6278219631370021 | 0.8029115901631975 |
| pure AP best | 0.85 | 0.2620249213583159 | 0.6258578850907429 | 0.8029115901631975 |
| WTA/AP50 best | 0.50 | 0.25679548890604637 | 0.6632497900770326 | 0.7973356854170808 |
| WTA/AP50 best | 0.85 | 0.2596002929982258 | 0.6621093338945596 | 0.7973356854170808 |

对照：

```text
Stream3D same 32f support:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615

previous pure Stream4D AP best:
0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975

previous pure Stream4D AP50 best:
0.25307574519460824 / 0.6548076923076923 / 0.7973356854170808
```

### 35.6 结论

```text
local proposal bank 单独不强，但作为 low-confidence recall layer 有小幅正收益。
```

最好的 pure Stream4D fixed32 更新为：

```text
stream4d_v4_1_scene0050_purebest_plus_localbank_minobs5_drop0p85:
0.2620249213583159 / 0.6258578850907429 / 0.8029115901631975
```

但仍然没有超过 Stream3D：

```text
AP 0.2620249213583159 < 0.39113247863247863
AP50 0.6258578850907429 < 0.6461538461538462
AP25 0.8029115901631975 > 0.7615384615384615
```

算法 insight：

```text
1. memory 合并前的 local proposal bank 仍含有可用 recall。
2. same-frame best-mask 可以清掉 226 个同帧 mask 冲突，但还不足以得到高质量 primary。
3. low-confidence localbank 可以轻微补 AP，但也无法解决 high-IoU 一对一实例质量。
```

## 36. 2026-06-08 最终编译、测试和审阅包校验

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'export_local_proposal_bank|reliable_densifier|export_scannet|replay_evidence_graph|reexport_scannet|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
test of stream4d_v4_1_code_review_packet.zip OK
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/reexport_scannet.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/export_scannet.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/export_local_proposal_bank.py
```

## 31. 2026-06-08 继续推进：component densify mask-selection 内部算法

### 31.1 本轮目的

前几轮已经证明最终 prediction 后处理很难接近 GT oracle。本轮把改动移到 object 导出内部：

```text
原逻辑：
  component_densify 按 mask observation 的 carrier coverage 排序和筛选。

新增逻辑：
  在 carrier seed 投影到每个 2D mask 后，计算 seed/component/distance-filtered component 的密度，
  用这些无 GT 信号决定优先使用哪些 mask 视角。
```

这不是使用 GT，也不是改 evaluator。

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

支持：

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

第一次运行时 evaluator 调用写错：

```bash
$PY -u -m evaluation.evaluate --config "$cfg" --seq-list "$SEQ" --no-class
```

实际 `evaluation.evaluate` 需要：

```text
--pred_path
--gt_path
--dataset
```

因此该命令以 code 2 退出。修复方式：

```text
统一改用 tools.evaluate_cross_prepoints。
own support 用 pred_config == pre_points_config。
fixed support 用 pre_points_config = stream4d_scannet_scene0050_32f_ioc075_fixmem。
```

这个错误发生在评估调用层，没有生成错误指标；后续所有结果均来自修正后的 `tools.evaluate_cross_prepoints` 输出文件。

### 31.4 运行命令

96f 扫描：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1

for mode in seed_density component_seed_density coverage_component_density kept_seed_density; do
  cfg="stream4d_v4_1_egraph_scene0050_96f_ioc0p79_minobs10_compdens_r003_d16_m8_sel_${mode}_hard2"

  "$PY" -u -m stream4d.replay_evidence_graph \
    --seq-name scene0050_00 \
    --frame-stride 10 \
    --max-frames 96 \
    --window-size 32 \
    --window-stride 16 \
    --rho-min 0.35 \
    --graph-min-shared-carriers 2 \
    --graph-min-carrier-ioc 0.79 \
    --graph-min-component-observations 10 \
    --graph-min-component-carriers 1 \
    --export-support-mode component_densify \
    --export-core-nn-radius 0.03 \
    --export-mask-sample-stride 2 \
    --export-mask-max-pixels 12000 \
    --export-max-masks-per-object 8 \
    --export-mask-min-relative-coverage 0.95 \
    --densify-boundary-erosion 1 \
    --densify-seed-distance-px 16 \
    --densify-min-seed-pixels 1 \
    --densify-seed-keep-mode none \
    --densify-mask-selection-mode "$mode" \
    --export-enable-wta \
    --export-wta-score-mode compactness \
    --export-wta-min-conflict-owners 2 \
    --input-debug-root outputs/stream4d_debug_v4_1_memoryold_scene0050_96f \
    --output-config "$cfg" \
    --debug-root outputs/stream4d_evidence_graph_v4_1

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$cfg" \
    --pre-points-config "$cfg" \
    --output-config "${cfg}_self_audit" \
    --no-class \
    --output-file "data/evaluation/scannet/${cfg}_self_audit_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$cfg" \
    --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
    --output-config "${cfg}_cross_32fsupport" \
    --no-class \
    --output-file "data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt" \
    --audit-root outputs/audit_v4_1_continue/cross_prepoints
done
```

128f 补跑：

```bash
cfg=stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_compdens_r003_d16_m8_sel_coverage_component_density_hard2

"$PY" -u -m stream4d.replay_evidence_graph \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 128 \
  --window-size 32 \
  --window-stride 16 \
  --rho-min 0.35 \
  --graph-min-shared-carriers 2 \
  --graph-min-carrier-ioc 0.79 \
  --graph-min-component-observations 10 \
  --graph-min-component-carriers 1 \
  --export-support-mode component_densify \
  --export-core-nn-radius 0.03 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 12000 \
  --export-max-masks-per-object 8 \
  --export-mask-min-relative-coverage 0.95 \
  --densify-boundary-erosion 1 \
  --densify-seed-distance-px 16 \
  --densify-min-seed-pixels 1 \
  --densify-seed-keep-mode none \
  --densify-mask-selection-mode coverage_component_density \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 2 \
  --input-debug-root outputs/stream4d_debug_scene0050_128f_ioc075_fixmem \
  --output-config "$cfg" \
  --debug-root outputs/stream4d_evidence_graph_v4_1
```

### 31.5 结果

96f：

| mode | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|
| seed_density | 0.1555475388808722 | 0.2571428571428572 | 0.49206349206349204 | 0.09797979797979799 | 0.18181818181818182 | 0.18181818181818182 |
| component_seed_density | 0.1555475388808722 | 0.2571428571428572 | 0.49206349206349204 | 0.09797979797979799 | 0.18181818181818182 | 0.18181818181818182 |
| coverage_component_density | 0.4024216524216524 | 0.5384615384615384 | 0.7371794871794872 | 0.13645833333333332 | 0.26249999999999996 | 0.26249999999999996 |
| kept_seed_density | 0.18897350606909433 | 0.359375 | 0.5803571428571429 | 0.0809027777777778 | 0.15000000000000002 | 0.15000000000000002 |

128f：

| mode | own AP | own AP50 | own AP25 | fixed32 AP | fixed32 AP50 | fixed32 AP25 |
|---|---:|---:|---:|---:|---:|---:|
| coverage_component_density | 0.43833943833943834 | 0.5686813186813187 | 0.7554945054945055 | 0.13645833333333332 | 0.26249999999999996 | 0.26249999999999996 |

对照：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| pure Stream4D previous AP best fixed32 | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| 128f strict compdens rel1.0 own | 0.490385 | 0.605769 | 0.810897 |
| 128f selection coverage_component_density fixed32 | 0.13645833333333332 | 0.26249999999999996 | 0.26249999999999996 |

summary 诊断：

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

### 31.6 结论

```text
component densify 内部 mask-selection 没有解决 inherit/fixed support。
```

解释：

```text
1. coverage_component_density 能保留较高 own recompute AP，
   说明它确实筛到了一些干净的 mask view。

2. 但 fixed32 AP 仍只有 0.13645833333333332，
   远低于此前 pure Stream4D fixed best 0.2530976765890559，
   更远低于 Stream3D same support 0.39113247863247863。

3. 这说明 object 内部换 mask 视角，只能改变 observed-support core 的质量；
   它不能补足 fixed support 下缺失的 object coverage 和一对一实例分配。
```

## 32. 2026-06-08 最终编译、测试和审阅包校验

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'reliable_densifier|export_scannet|replay_evidence_graph|reexport_scannet|greedy_support_select|object_competition_rank|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.002s

OK
test of stream4d_v4_1_code_review_packet.zip OK
Stream3D/stream4d/reliable_densifier.py
Stream3D/stream4d/reexport_scannet.py
Stream3D/stream4d/replay_evidence_graph.py
Stream3D/stream4d/export_scannet.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/object_competition_rank.py
Stream3D/tools/greedy_support_select.py
```

使用信号：

```text
original score
support_area
unique_ratio
conflict_ratio
marginal new support area
novelty = new_area / support_area
overlap with selected union
```

### 29.3 运行命令

工作目录：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
```

编译：

```bash
$PY -m py_compile tools/greedy_support_select.py
```

核心命令模板：

```bash
$PY -u -m tools.greedy_support_select \
  --seq-list splits/scannet_scene0050.txt \
  --input-config <input_config> \
  --output-config <primary_config> \
  --score-pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --max-instances <60|100|150> \
  --min-support-area 1 \
  --min-new-area 10 \
  --summary-root outputs/greedy_support_select_v4_1_continue \
  <weight args>

$PY -u -m tools.fuse_prediction_configs \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config <primary_config> \
  --secondary-config <input_config> \
  --output-config <primary_config>_plus_origlow \
  --preserve-primary-score \
  --secondary-score 0.01 \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/greedy_support_select_v4_1_continue

$PY -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config <eval_config> \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config <eval_config>_cross_32fsupport \
  --no-class \
  --output-file data/evaluation/scannet/<eval_config>_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit_v4_1_continue/cross_prepoints
```

扫描范围：

```text
inputs:
  stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
  stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005

max_instances:
  60 / 100 / 150

suppression:
  nosup / suppress overlap ratio 0.85

weight profiles:
  contain default
  contain scoreheavy
  tier default
  tier novelty
```

输出数量：

```text
summary json = 48
evaluation txt = 48
```

### 29.4 结果摘录

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `greedy_contain_default_max100_nosup_plus_origlow` | 0.2185270192508893 | 0.6000470809792844 | 0.7805084745762715 |
| `greedy_contain_default_max100_sup0.85_plus_origlow` | 0.2185270192508893 | 0.6000470809792844 | 0.7805084745762715 |
| `greedy_tier_default_max100_nosup_plus_origlow` | 0.21240961199294536 | 0.5805351307189544 | 0.7558823529411767 |
| `greedy_contain_scoreheavy_max100_nosup_plus_origlow` | 0.2056065599497803 | 0.5822033898305086 | 0.7296610169491528 |
| `greedy_tier_novelty_max100_sup0.85_plus_origlow` | 0.16637309452195936 | 0.45701781588294743 | 0.8250117172988692 |

对照：

| method | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Stream3D same 32f support | 0.39113247863247863 | 0.6461538461538462 | 0.7615384615384615 |
| pure Stream4D previous AP best | 0.2530976765890559 | 0.6158322281167109 | 0.8029115901631975 |
| greedy best | 0.2185270192508893 | 0.6000470809792844 | 0.7805084745762715 |

### 29.5 summary 诊断

默认 contain greedy：

```text
config = stream4d_v4_1_scene0050_greedy_contain_default_max100_nosup
mean_num_instances_before = 337.0
mean_num_score_support_points = 12214.0
mean_num_valid_candidates = 309.0
mean_num_selected = 37.0
mean_selected_union_count = 3198.0
mean_selected_support_area_mean = 86.83783783783784
```

novelty-heavy tier greedy：

```text
config = stream4d_v4_1_scene0050_greedy_tier_novelty_max100_sup0.85
mean_num_instances_before = 420.0
mean_num_score_support_points = 12214.0
mean_num_valid_candidates = 385.0
mean_num_selected = 32.0
mean_selected_union_count = 9618.0
mean_selected_support_area_mean = 310.25
```

解释：

```text
默认 greedy 选出的 primary 太 sparse，只有 3198 个 support 点。
novelty-heavy 能把 selected union 提到 9618 个点，但 AP/AP50 下降明显。
```

### 29.6 结论

```text
greedy support set-cover 没有超过 Stream3D，也没有超过此前纯 Stream4D best。
```

失败原因：

```text
1. 新增 support 覆盖量会偏向大而粗的候选。
2. novelty-heavy 的 AP25 可以较高，但 AP/AP50 明显下降，说明它选到了粗召回而不是精确实例。
3. 默认 greedy 能控制候选质量，但 coverage 太小。
4. 这再次说明：support coverage 与实例边界质量之间有强 trade-off，单靠无监督面积/novelty 不能接近 oracle。
```

## 30. 2026-06-08 最终编译、测试和审阅包校验

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'greedy_support_select|object_competition_rank|oracle_candidate_upper_bound|support_aware_object_rank|wta_prediction_points|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/wta_prediction_points.py
Stream3D/tools/support_aware_object_rank.py
Stream3D/tools/oracle_candidate_upper_bound.py
Stream3D/tools/object_competition_rank.py
Stream3D/tools/greedy_support_select.py
```

## 31. 2026-06-08 mask-overlap graph 负例补跑

目的：

```text
上一轮单帧 mask observation bank 几乎无效。
本轮尝试把不同帧的 2D mask observation 按 3D overlap 建图，
再用 same-frame cannot-link 形成 object component，
检查这是否能比单帧 observation 更接近可用 object proposal。
```

新增代码：

```text
Stream3D/tools/export_mask_overlap_graph.py
```

## 32. 2026-06-08 slot-wise object assignment 负例

目的：

```text
第 28 节 oracle 说明 pure Stream4D 候选池有潜力，但无监督 one-to-one assignment 不够。
本轮不再做最终 NMS/面积排序，而是把 32f Stream4D mask 当作固定 support 里的 object slot，
从 128f strict / localbank 候选池里给每个 slot 最多分配一个候选。
```

新增代码：

```text
Stream3D/tools/slotwise_candidate_select.py
```

## 33. 2026-06-08 silhouette consistency 边界一致性诊断

目的：

```text
第 33 节 slot-wise assignment 说明 support overlap 仍不足以找到 oracle 需要的高 IoU one-to-one candidate。
本轮尝试更贴近边界质量的无 GT 信号：
把每个 object 的 3D point_ids 投影回它的多帧 2D mask observations，
统计可见点落在 mask 内的比例和距离 mask 边界的 margin。
```

新增代码：

```text
Stream3D/tools/silhouette_consistency_score.py
```

## 34. 2026-06-08 silhouette recall + support-aware containment

目的：

```text
上一节 silhouette q0.85 localbank recall 有小幅正收益。
本轮继续检查两个问题：
1. q0.85 localbank 加到 WTA primary 是否更好；
2. containment suppression 是否应该只在 fixed 32f support 内计算，而不是全场景计算。
```

代码修改：

```text
Stream3D/tools/fuse_prediction_configs.py
```

新增参数：

```text
--drop-overlap-pre-points-config
```

含义：

```text
如果设置，则 secondary 是否被 primary 覆盖，只在这个 config 的 pre_points support 内计算。
本轮设置为 stream4d_scannet_scene0050_32f_ioc075_fixmem。
```

编译：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
"$PY" -m py_compile tools/fuse_prediction_configs.py
```

WTA primary + q0.85 localbank：

```bash
"$PY" -u -m tools.fuse_prediction_configs \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config stream4d_v4_1_scene0050_wta_score_high_plus_orig_low001 \
  --secondary-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85 \
  --output-config stream4d_v4_1_scene0050_wtabest_plus_localbank_sil_q0p85_drop0p85 \
  --fusion-mode concatenate \
  --primary-score -1 \
  --secondary-score 0.005 \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/fuse_prediction_configs_v4_1_continue
```

结果：

```text
stream4d_v4_1_scene0050_wtabest_plus_localbank_sil_q0p85_drop0p85
AP/AP50/AP25 = 0.26088464186321986 / 0.6634692326927886 / 0.7973356854170808
```

secondary self-NMS + fusion：

```text
pure + q0.85 localbank nms0.70:
AP/AP50/AP25 = 0.26437103599800854 / 0.6283420064739469 / 0.8029115901631975

wta + q0.85 localbank nms0.70:
AP/AP50/AP25 = 0.26092879541368474 / 0.6635159835109279 / 0.7973356854170808

pure + q0.85 localbank nms0.85:
AP/AP50/AP25 = 0.2642829759099485 / 0.6282487663807068 / 0.8029115901631975

wta + q0.85 localbank nms0.85:
AP/AP50/AP25 = 0.26088464186321986 / 0.6634692326927886 / 0.7973356854170808
```

support-aware containment suppression：

```bash
"$PY" -u -m tools.fuse_prediction_configs \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005 \
  --secondary-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85 \
  --output-config stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95 \
  --fusion-mode concatenate \
  --primary-score -1 \
  --secondary-score 0.005 \
  --drop-secondary-iou-threshold 0.95 \
  --drop-secondary-overlap-mode secondary_ioc \
  --drop-overlap-pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --summary-root outputs/fuse_prediction_configs_v4_1_continue
```

结果：

```text
supportdrop0.85, pure:
AP/AP50/AP25 = 0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975

supportdrop0.85, wta:
AP/AP50/AP25 = 0.25307574519460824 / 0.6548076923076923 / 0.7973356854170808

supportdrop0.95, pure:
AP/AP50/AP25 = 0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

supportdrop0.95, wta:
AP/AP50/AP25 = 0.2616556679530756 / 0.6634692326927886 / 0.7973356854170808
```

关键 summary：

```text
supportdrop0.95 pure:
primary instances = 337
secondary instances after support-aware containment = 13
secondary skipped by support-aware containment = 1
secondary overlap support points = 12214
output instances = 350
output union = 52798
```

审阅包更新命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'fuse_prediction_configs|silhouette_consistency_score|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
test of stream4d_v4_1_code_review_packet.zip OK
Stream3D/tools/fuse_prediction_configs.py
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/silhouette_consistency_score.py
```

该工具只使用：

```text
prediction npz
object_dict.npy
ScanNet depth / pose / intrinsics
Cropformer 2D mask png
```

不读取 GT。

核心命令模板：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -u -m tools.silhouette_consistency_score \
  --seq-list splits/scannet_scene0050.txt \
  --input-config "$input_config" \
  --output-config "$output_config" \
  --quality-mode score_silhouette \
  --score-weight 0.0 \
  --max-observations 8 \
  --max-points-per-object 2000 \
  --depth-tolerance 0.08 \
  --boundary-margin-px 2.0 \
  --summary-root outputs/silhouette_consistency_score_v4_1_continue
```

评估命令模板：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$output_config" \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config ${output_config}_cross_32fsupport \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/${output_config}_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit/cross_prepoints
```

直接作为主排序的结果：

```text
stream4d_v4_1_scene0050_32f_silhouette
AP/AP50/AP25 = 0.19635331985316004 / 0.3724225776217969 / 0.557964704172192

stream4d_v4_1_scene0050_128f_compdens_silhouette
AP/AP50/AP25 = 0.09796296296296296 / 0.3 / 0.3

stream4d_v4_1_scene0050_localbank_minobs5_silhouette
AP/AP50/AP25 = 0.08680001110600649 / 0.20452444866137034 / 0.4088280775933543
```

放宽可见性 / 去掉边界 margin 的 32f 诊断：

```text
stream4d_v4_1_scene0050_32f_sil_tol020_m0
AP/AP50/AP25 = 0.17902095768239432 / 0.360339503423507 / 0.550959314975729

stream4d_v4_1_scene0050_32f_sil_tol999_m0
AP/AP50/AP25 = 0.12286643274110462 / 0.3077679103923552 / 0.5039995386409526

stream4d_v4_1_scene0050_32f_silarea_tol020_m0
AP/AP50/AP25 = 0.15387831245728228 / 0.36115609642373003 / 0.570257355041958

stream4d_v4_1_scene0050_32f_score_silarea_tol020_m0
AP/AP50/AP25 = 0.1429759418260058 / 0.3465727630903967 / 0.5733766555597568
```

localbank high-silhouette 过滤后作为 low-confidence recall 的命令：

```bash
"$PY" -u -m tools.silhouette_consistency_score \
  --seq-list splits/scannet_scene0050.txt \
  --input-config stream4d_v4_1_scene0050_localbank128_bestframe_minobs5 \
  --output-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85 \
  --quality-mode score_silhouette \
  --score-weight 0.0 \
  --max-observations 8 \
  --max-points-per-object 2000 \
  --depth-tolerance 0.08 \
  --boundary-margin-px 2.0 \
  --min-silhouette-quality 0.85 \
  --summary-root outputs/silhouette_consistency_score_v4_1_continue

"$PY" -u -m tools.fuse_prediction_configs \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005 \
  --secondary-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85 \
  --output-config stream4d_v4_1_scene0050_purebest_plus_localbank_sil_q0p85_drop0p85 \
  --fusion-mode concatenate \
  --primary-score -1 \
  --secondary-score 0.005 \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --summary-root outputs/fuse_prediction_configs_v4_1_continue
```

localbank silhouette filter + fusion 结果：

```text
q0.50:
AP/AP50/AP25 = 0.2623750073353132 / 0.6262285643605048 / 0.8029115901631975

q0.70:
AP/AP50/AP25 = 0.2630035146304872 / 0.6268940426730419 / 0.8029115901631975

q0.80:
AP/AP50/AP25 = 0.26395589558286814 / 0.6279024460343864 / 0.8029115901631975

q0.85:
AP/AP50/AP25 = 0.2642829759099485 / 0.6282487663807068 / 0.8029115901631975

q0.90:
AP/AP50/AP25 = 0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975
```

关键 summary：

```text
localbank q0.85:
instances before = 102
instances after = 14
removed = 88
silhouette_quality_mean = 0.6679583787918091
silhouette_quality_max = 0.9437242150306702
inside_visible_ratio_mean = 0.7076940548072281
interior_ratio_mean = 0.8812063890624772

fused q0.85:
primary instances = 337
secondary instances after containment = 13
output instances = 350
output union = 53137
```

审阅包更新命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'silhouette_consistency_score|slotwise_candidate_select|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/slotwise_candidate_select.py
Stream3D/tools/silhouette_consistency_score.py
```

新增工具的无 GT 信号：

```text
slot_ioc: candidate 覆盖 slot 的比例
candidate_ioc: candidate 有多少落在 slot 里
iou: candidate 与 slot 的 support IoU
area_match: candidate 面积与 slot 面积是否接近
conflict_ratio: candidate 在 target support 内与其它 candidate 的冲突比例
original_score: 原 prediction score
```

编译：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile tools/slotwise_candidate_select.py
```

初次运行时遇到两个 evaluator 参数 blocker：

```text
1. 我第一次误用了旧接口 --split-file / --summary-name。
   当前 tools.evaluate_cross_prepoints 需要 --seq-list 和 --output-config。

2. 修正接口后第一次评估忘了加 --no-class。
   这些 prediction 是 class-agnostic，缺少 --no-class 会得到 0 / 0 / 0。
```

修复后使用的评估命令模板：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$out" \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config ${out}_cross_32fsupport \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit/cross_prepoints
```

slot assignment 生成命令示例：

```bash
"$PY" -u -m tools.slotwise_candidate_select \
  --seq-list splits/scannet_scene0050.txt \
  --slot-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --candidate-config stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005 \
  --output-config stream4d_v4_1_scene0050_slotassign_strict_ms10_mc50_iou03_ar02_2 \
  --score-pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --min-slot-ioc 0.10 \
  --min-candidate-ioc 0.50 \
  --min-iou 0.03 \
  --min-area-ratio 0.20 \
  --max-area-ratio 2.00 \
  --keep-unassigned-slots \
  --fallback-slot-score 0.02 \
  --summary-root outputs/slotwise_candidate_select_v4_1_continue
```

结果：

```text
stream4d_v4_1_scene0050_slotassign_strict_ms10_mc50_iou03_ar02_2
AP/AP50/AP25 = 0.17245357128934893 / 0.35970160173637145 / 0.6175977493359794

stream4d_v4_1_scene0050_slotassign_strict_ms20_mc30_iou05_ar02_3
AP/AP50/AP25 = 0.17245357128934893 / 0.35970160173637145 / 0.6175977493359794

stream4d_v4_1_scene0050_slotassign_purelocal_ms10_mc50_iou03_ar02_2
AP/AP50/AP25 = 0.1050909972686474 / 0.26231610169308595 / 0.5000313281793627

stream4d_v4_1_scene0050_slotassign_wtalocal_ms10_mc50_iou03_ar02_2
AP/AP50/AP25 = 0.200228879429071 / 0.5324016102033344 / 0.7080914727035418
```

高质量门槛补跑：

```text
stream4d_v4_1_scene0050_slotassign_strict_ms10_mc50_q0p70
AP/AP50/AP25 = 0.16293981481481482 / 0.34178571428571436 / 0.5818452380952382

stream4d_v4_1_scene0050_slotassign_strict_ms10_mc50_q0p80
AP/AP50/AP25 = 0.22309523809523807 / 0.46142857142857147 / 0.6935714285714286

stream4d_v4_1_scene0050_slotassign_strict_ms10_mc50_q0p90
AP/AP50/AP25 = 0.2026984126984127 / 0.44571428571428573 / 0.6814285714285714
```

高质量门槛 + 少量 unmatched low-confidence recall：

```text
stream4d_v4_1_scene0050_slotassign_strict_q0p80_unmatched20
AP/AP50/AP25 = 0.23217054263565895 / 0.47885382059800663 / 0.6935714285714286

stream4d_v4_1_scene0050_slotassign_strict_q0p80_unmatched50
AP/AP50/AP25 = 0.23197530864197535 / 0.4784920634920635 / 0.6935714285714286

stream4d_v4_1_scene0050_slotassign_strict_q0p80_unmatched100
AP/AP50/AP25 = 0.2309259259259259 / 0.47654761904761905 / 0.6935714285714286
```

关键 summary：

```text
strict loose:
selected_slots = 199
fallback_slots = 27
output_instances = 226
output_union = 15005
selected_iou_mean = 0.9595403753567581

strict q0.80:
selected_slots = 3
fallback_slots = 223
output_instances = 226
output_union = 12214
selected_iou_mean = 1.0

strict q0.80 + unmatched20:
selected_slots = 3
fallback_slots = 223
unmatched_added = 20
output_instances = 246
output_union = 14796
```

审阅包更新命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'slotwise_candidate_select|export_mask_overlap_graph|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/export_mask_overlap_graph.py
Stream3D/tools/slotwise_candidate_select.py
```

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile tools/export_mask_overlap_graph.py

DEBUG=outputs/stream4d_debug_scene0050_128f_ioc075_fixmem
SPLIT=splits/scannet_scene0050.txt
GT=data/scannet/gt
SUPPORT32=stream4d_scannet_scene0050_32f_ioc075_fixmem

for ov in 0.25 0.50; do
  tag=${ov/./p}
  cfg=stream4d_v4_1_scene0050_maskoverlap128_cov0p005_ioc${tag}

  "$PY" -u -m tools.export_mask_overlap_graph \
    --debug-root "$DEBUG" \
    --seq-name scene0050_00 \
    --output-config "$cfg" \
    --min-coverage 0.005 \
    --min-points-per-mask 100 \
    --export-nn-radius 0.05 \
    --overlap-mode min_ioc \
    --min-shared-points 25 \
    --min-overlap "$ov" \
    --min-component-observations 2 \
    --min-component-frames 2 \
    --score-mode coverage_points \
    --summary-root outputs/mask_overlap_graph_v4_1_continue

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --pred-config "$cfg" \
    --pre-points-config "$cfg" \
    --split-file "$SPLIT" \
    --gt-path "$GT" \
    --dataset scannet \
    --output-file data/evaluation/scannet/${cfg}_self_audit_class_agnostic.txt \
    --summary-root outputs/audit/cross_prepoints \
    --summary-name ${cfg}_self_audit_summary.json

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --pred-config "$cfg" \
    --pre-points-config "$SUPPORT32" \
    --split-file "$SPLIT" \
    --gt-path "$GT" \
    --dataset scannet \
    --output-file data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt \
    --summary-root outputs/audit/cross_prepoints \
    --summary-name ${cfg}_cross_32fsupport_summary.json
done
```

结果：

```text
stream4d_v4_1_scene0050_maskoverlap128_cov0p005_ioc0p25 own
AP/AP50/AP25 = 0.024660200725248455 / 0.11479534048467872 / 0.49441844919786093

stream4d_v4_1_scene0050_maskoverlap128_cov0p005_ioc0p25 fixed32
AP/AP50/AP25 = 0.00436307519640853 / 0.010555555555555558 / 0.010555555555555558

stream4d_v4_1_scene0050_maskoverlap128_cov0p005_ioc0p50 own
AP/AP50/AP25 = 0.024832058188772585 / 0.11634205765639588 / 0.504156454248366

stream4d_v4_1_scene0050_maskoverlap128_cov0p005_ioc0p50 fixed32
AP/AP50/AP25 = 0.00436307519640853 / 0.010555555555555558 / 0.010555555555555558
```

审阅包更新命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'export_mask_overlap_graph|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/export_mask_overlap_graph.py
```

## 39. 2026-06-08 继续推进：localbank 低置信层保序实验

目的：

```text
第 35 节的 best pure Stream4D 是：
stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
AP/AP50/AP25 = 0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

该配置把 silhouette-filtered localbank 作为 secondary recall layer，
但 secondary 被设成统一低分 0.005。
本轮检查：保留 localbank 内部 silhouette 分数排序，是否能改善 AP 曲线。
```

代码修改：

```text
Stream3D/tools/rescore_prediction_scores.py
```

新增：

```text
--score-feature source_score
```

含义：

```text
先把原 prediction score 做 min-max normalize，
再把它作为低置信层内部的 tiebreaker。
该工具不读取 GT。
```

编译：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile tools/rescore_prediction_scores.py
```

### 39.1 先跑出的错误协议诊断

第一次融合命令误把 primary 分数覆盖成常数 `1.0`：

```bash
"$PY" -u -m tools.fuse_prediction_configs \
  --seq-list splits/scannet_scene0050.txt \
  --primary-config stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005 \
  --secondary-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85_lowrank001 \
  --output-config stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_lowrank001_supportdrop0p95 \
  --preserve-secondary-score \
  --primary-score 1.0 \
  --drop-secondary-iou-threshold 0.95 \
  --drop-secondary-overlap-mode secondary_ioc \
  --drop-overlap-pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --summary-root outputs/stream4d_fusion_v4_1_continue
```

这个命令不等价于上一节 best，因为上一节 best 保留了 primary 原始多层 score。

错误协议结果：

| config | AP | AP50 | AP25 | 说明 |
|---|---:|---:|---:|---|
| `lowrank001_supportdrop0p95` | 0.2520543021661612 | 0.48293813693767984 | 0.6279661016949152 | primary score 被错误覆盖 |
| `lowrank005_supportdrop0p95` | 0.2520543021661612 | 0.48293813693767984 | 0.6279661016949152 | primary score 被错误覆盖 |

该结果只作为协议排错记录，不作为算法结论。

排错命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
"$PY" - <<'PY'
import json
from pathlib import Path
configs=[
'stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95',
'stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_lowrank001_supportdrop0p95',
'stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_lowrank005_supportdrop0p95',
]
for cfg in configs:
    paths=list(Path('Stream3D/outputs').glob(f'**/{cfg}_summary.json'))
    print(cfg, paths)
PY
```

确认差异：

```text
上一节 best:
primary_score = -1.0
secondary_score = 0.005

错误协议:
primary_score = 1.0
secondary_score = -1.0
```

### 39.2 正确协议命令

先生成两个低置信但保留 source score 排序的 secondary：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -u -m tools.rescore_prediction_scores \
  --seq-list splits/scannet_scene0050.txt \
  --input-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85 \
  --output-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85_lowrank001 \
  --base-score-mode constant \
  --constant-score 0.005 \
  --score-feature source_score \
  --tiebreaker-weight 0.001 \
  --summary-root outputs/stream4d_score_calibration_v4_1_continue

"$PY" -u -m tools.rescore_prediction_scores \
  --seq-list splits/scannet_scene0050.txt \
  --input-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85 \
  --output-config stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85_lowrank005 \
  --base-score-mode constant \
  --constant-score 0.005 \
  --score-feature source_score \
  --tiebreaker-weight 0.005 \
  --summary-root outputs/stream4d_score_calibration_v4_1_continue
```

正确融合必须保留 primary 和 secondary 分数：

```bash
for tag in lowrank001 lowrank005; do
  sec_out=stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85_${tag}
  out=stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_${tag}_preserveprimary_supportdrop0p95

  "$PY" -u -m tools.fuse_prediction_configs \
    --seq-list splits/scannet_scene0050.txt \
    --primary-config stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005 \
    --secondary-config "$sec_out" \
    --output-config "$out" \
    --preserve-primary-score \
    --preserve-secondary-score \
    --drop-secondary-iou-threshold 0.95 \
    --drop-secondary-overlap-mode secondary_ioc \
    --drop-overlap-pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
    --summary-root outputs/stream4d_fusion_v4_1_continue

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$out" \
    --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit
done
```

### 39.3 正确协议结果

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| previous best: `pure_plus_localbank_sil_q0p85_supportdrop0p95` | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 |
| `lowrank001_preserveprimary_supportdrop0p95` | 0.2652273083220721 | 0.6284313820035432 | 0.8029115901631975 |
| `lowrank005_preserveprimary_supportdrop0p95` | 0.2652273083220721 | 0.6284313820035432 | 0.8029115901631975 |

lowrank secondary score 范围：

```text
lowrank001:
score_min_before = 0.8509376645088196
score_max_before = 0.9437242150306702
score_min_after = 0.004999999888241291
score_max_after = 0.006000000052154064

lowrank005:
score_min_after = 0.004999999888241291
score_max_after = 0.009999999776482582
```

fusion summary：

```text
mean_num_primary_instances = 337.0
mean_num_secondary_instances = 13.0
mean_num_output_instances = 350.0
mean_secondary_skipped_by_iou = 1.0
mean_secondary_max_overlap_mean = 0.6905256576907615
mean_secondary_overlap_support_points = 12214.0
mean_output_union_count = 52798.0
```

### 39.4 本轮执行结论

正确协议下，保留 localbank 内部 silhouette 排序没有超过 previous best：

```text
AP:   0.2652273083220721 < 0.2653850988341053
AP50: 0.6284313820035432 > 0.6282487663807068
AP25: 0.8029115901631975 = 0.8029115901631975
```

结论：

```text
secondary 内部排序不是当前 AP 主瓶颈。
localbank recall layer 的收益很窄；它能小幅补 recall，但不能解决纯 Stream4D 与 Stream3D 之间的 AP 差距。
```

### 39.5 编译、测试与审阅包更新

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'rescore_prediction_scores|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
Ran 6 tests in 0.002s
OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/rescore_prediction_scores.py
```

## 40. 2026-06-08 继续推进：support 内 3D connected-component refinement 负例

### 40.1 本轮目的

前几轮已经证明：

```text
best pure Stream4D fixed32:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

当前 best 的 `union in target` 已接近 99%，因此本轮不再继续补 coverage，而是尝试更贴近算法的边界/实例质量精炼：

```text
在 fixed support 内，如果一个 prediction mask 由多个空间碎片组成，
只保留主要 3D 连通组件或少量大组件，
希望减少边界污染和重复归属。
```

该方法不读取 GT，只使用：

```text
prediction mask
fixed support pre_points
ScanNet mesh 顶点坐标
```

### 40.2 新增工具

新增：

```text
Stream3D/tools/support_component_refine.py
```

## 42. 2026-06-08 继续推进：evidence graph 节点级质量过滤负例

### 42.1 本轮目的

第 41 节说明：在最终 prediction 阶段再去反推 2D mask observation 不可靠。本轮把过滤前移到 evidence graph 生成阶段，不再对最终 prediction 做裁剪，而是在 graph node 层面过滤低质量 mask observation。

直觉是：

```text
低质量 graph node 会形成脏 component；
如果在建图前删掉 carrier 数太少或 coverage 太低的 node，
可能让 component_densify 的 object 更稳定。
```

该方法不读取 GT，不改变 evaluator，只使用 cached D4RT carrier / mask observation。

### 42.2 代码修改

修改：

```text
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/replay_evidence_graph.py
```

新增参数：

```text
--graph-min-node-carriers
--graph-min-node-coverage
```

新增诊断字段：

```text
evidence_graph_num_raw_nodes
evidence_graph_num_dropped_nodes
evidence_graph_min_node_carriers
evidence_graph_min_node_coverage
```

### 42.3 编译

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/evidence_graph.py stream4d/replay_evidence_graph.py
```

### 42.4 运行命令

输入 carrier cache：

```text
outputs/stream4d_debug_scene0050_128f_ioc075_fixmem
```

公共参数：

```bash
COMMON_ARGS="\
  --seq-list splits/scannet_scene0050.txt \
  --input-debug-root outputs/stream4d_debug_scene0050_128f_ioc075_fixmem \
  --graph-min-carrier-ioc 0.79 \
  --graph-min-component-observations 10 \
  --export-support-mode component_densify \
  --export-core-nn-radius 0.03 \
  --export-mask-sample-stride 2 \
  --export-max-masks-per-object 8 \
  --export-mask-min-relative-coverage 1.0 \
  --densify-seed-distance-px 16 \
  --densify-seed-keep-mode none \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 2 \
  --output-root outputs/stream4d_evidence_graph_v4_1_continue \
  --summary-root outputs/stream4d_evidence_graph_v4_1_continue"
```

两组过滤：

```bash
"$PY" -u -m stream4d.replay_evidence_graph \
  $COMMON_ARGS \
  --graph-min-node-carriers 3 \
  --graph-min-node-coverage 0.0 \
  --output-config stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_nodecar3_compdens_rel100

"$PY" -u -m stream4d.replay_evidence_graph \
  $COMMON_ARGS \
  --graph-min-node-carriers 1 \
  --graph-min-node-coverage 0.002 \
  --output-config stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_nodecov002_compdens_rel100
```

自 support 与 fixed32 support 评估：

```bash
for cfg in \
  stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_nodecar3_compdens_rel100 \
  stream4d_v4_1_egraph_scene0050_128f_ioc0p79_minobs10_nodecov002_compdens_rel100
do
  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$cfg" \
    --pre-points-config "$cfg" \
    --output-config ${cfg}_self_audit \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${cfg}_self_audit_class_agnostic.txt \
    --audit-root outputs/audit

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_scene0050.txt \
    --pred-config "$cfg" \
    --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
    --output-config ${cfg}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit
done
```

### 42.5 结果

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

Graph / export 诊断：

| config | graph nodes | kept components | dropped components | accepted edges | rejected conflict edges | export conflict | hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nodecar3` | 1934 | 67 | 280 | 1587 | 7863 | 0.0 | 0.9181488441200891 |
| `nodecov002` | 1228 | 37 | 250 | 941 | 4532 | 0.0 | 0.9415782915437781 |

### 42.6 本轮执行结论

该方向没有达成目标。

`nodecar3` 只删掉 4 个 graph nodes，几乎不改变 graph 结构，但 self recompute 从原 128f strict 的 0.490385 降到 0.454545，fixed32 从 0.135185 降到 0.108889。

`nodecov002` 删掉 710 个 graph nodes，object 数从 67 压到 37，support points 从 6531 左右压到 1897，结果 self / fixed32 都明显崩：

```text
self    = 0.333333 / 0.514286 / 0.714286
fixed32 = 0.059722 / 0.100000 / 0.100000
```

结论：

```text
graph node 级硬过滤太早、太粗。
carrier 数或 node coverage 低，并不等于该 observation 一定是坏证据；
删掉这些节点会破坏 component 的可连接性和 recall。
```

新的 insight：

```text
1. 低 coverage node 不能在建图前硬删。
2. 更合理的算法是保留 node，但把 node quality 作为边权/组件质量/输出 support 权重。
3. fixed32 失败仍然不是单个低质量 node 造成，而是 object-level 一对一和边界质量不足。
```

## 43. 2026-06-08 继续推进：evidence graph coverage-aware edge ordering 负例

### 43.1 本轮目的

第 42 节说明建图前硬删低质量 node 会失败。本轮尝试更温和的算法：

```text
不删除低 coverage node。
保留所有 node 和原 carrier IoC 阈值。
只在 edge 排序时，让高 coverage node 相关的边优先合并。
```

这样做的目的是：减少低 coverage node 过早影响 component 形成，同时保留它们作为连接证据或后续 fringe evidence。

### 43.2 代码修改

修改：

```text
Stream3D/stream4d/evidence_graph.py
Stream3D/stream4d/replay_evidence_graph.py
```

新增参数：

```text
--graph-edge-coverage-power
```

实现细节：

```text
raw carrier IoC 仍用于判断是否通过 min_carrier_ioc；
coverage-adjusted score 只用于 edge sorting。
```

中间失败记录：

```text
第一版实现把 coverage-adjusted score 同时用于 threshold。
结果 graph 被打得过碎，只剩 2 个 object。
随后修正为“raw carrier IoC 过阈值，coverage-adjusted score 只排序”。
```

### 43.3 命令错误与修复

第一次运行错误命令：

```bash
"$PY" -u -m stream4d.replay_evidence_graph \
  --seq-list splits/scannet_scene0050.txt \
  ...
```

错误：

```text
replay_evidence_graph.py: error: the following arguments are required: --seq-name
```

修复：

```text
replay_evidence_graph 是单 scene replay 工具，必须使用 --seq-name scene0050_00。
```

第二次运行错误命令漏掉 `--max-frames 128`，导致 replay 查找不存在的第 8 个窗口：

```text
FileNotFoundError:
outputs/stream4d_debug_scene0050_128f_ioc075_fixmem/scene0050_00/carriers_window007.npz
```

修复：

```text
128f carrier cache 对应 7 个窗口，因此 replay 必须加 --max-frames 128。
```

### 43.4 运行命令

公共参数：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
export OPEN3D_DISABLE_WEB_VISUALIZER=true
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

COMMON_ARGS="\
  --seq-name scene0050_00 \
  --max-frames 128 \
  --input-debug-root outputs/stream4d_debug_scene0050_128f_ioc075_fixmem \
  --graph-min-component-observations 10 \
  --export-support-mode component_densify \
  --export-core-nn-radius 0.03 \
  --export-mask-sample-stride 2 \
  --export-max-masks-per-object 8 \
  --export-mask-min-relative-coverage 1.0 \
  --densify-seed-distance-px 16 \
  --densify-seed-keep-mode none \
  --export-enable-wta \
  --export-wta-score-mode compactness \
  --export-wta-min-conflict-owners 2 \
  --debug-root outputs/stream4d_evidence_graph_v4_1_continue"
```

错误的第一版 threshold-coupled edge coverage：

```bash
"$PY" -u -m stream4d.replay_evidence_graph \
  $COMMON_ARGS \
  --graph-min-carrier-ioc 0.70 \
  --graph-edge-coverage-power 0.25 \
  --output-config stream4d_v4_1_egraph_scene0050_128f_ioc0p70_edgecovp025_compdens_rel100

"$PY" -u -m stream4d.replay_evidence_graph \
  $COMMON_ARGS \
  --graph-min-carrier-ioc 0.60 \
  --graph-edge-coverage-power 0.50 \
  --output-config stream4d_v4_1_egraph_scene0050_128f_ioc0p60_edgecovp050_compdens_rel100
```

修正版 sort-only edge coverage：

```bash
"$PY" -u -m stream4d.replay_evidence_graph \
  $COMMON_ARGS \
  --graph-min-carrier-ioc 0.79 \
  --graph-edge-coverage-power 0.50 \
  --output-config stream4d_v4_1_egraph_scene0050_128f_ioc0p79_edgesortcovp050_compdens_rel100

"$PY" -u -m stream4d.replay_evidence_graph \
  $COMMON_ARGS \
  --graph-min-carrier-ioc 0.70 \
  --graph-edge-coverage-power 0.50 \
  --output-config stream4d_v4_1_egraph_scene0050_128f_ioc0p70_edgesortcovp050_compdens_rel100
```

每个 config 后都运行：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$cfg" \
  --pre-points-config "$cfg" \
  --output-config ${cfg}_self_audit \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/${cfg}_self_audit_class_agnostic.txt \
  --audit-root outputs/audit

"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_scene0050.txt \
  --pred-config "$cfg" \
  --pre-points-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config ${cfg}_cross_32fsupport \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit
```

### 43.5 结果

对照：

```text
128f strict component_densify rel1.0:
self recompute = 0.490385 / 0.605769 / 0.810897
fixed32        = 0.135185 / 0.300000 / 0.300000

pure Stream4D fixed32 best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
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

### 43.6 本轮执行结论

没有达成目标。

第一版实现的失败原因很明确：

```text
coverage-adjusted score 不能用于 carrier IoC threshold；
否则大量边被判成 weak edge，graph 只剩 2 个 object，AP 直接失效。
```

修正版更合理，但仍失败：

```text
ioc0p79_edgesortcovp050:
self = 0.408234 / 0.613839 / 0.783482
fixed32 = 0.108889 / 0.250000 / 0.250000

ioc0p70_edgesortcovp050:
self = 0.345062 / 0.525000 / 0.702778
fixed32 = 0.108889 / 0.250000 / 0.250000
```

对比原 128f strict rel1.0：

```text
0.490385 / 0.605769 / 0.810897
```

排序式 coverage-aware edge 会让 object 数从 67 增到 78-80，说明它改变了 cannot-link 条件下的合并顺序并制造更多碎片；AP 没有改善，fixed32 也没有改善。

### 43.7 新 insight

这轮把第 42 节的结论推进了一步：

```text
1. 低 coverage evidence 不能硬删。
2. 低 coverage evidence 只用于 edge sorting 也不够。
3. 当前 graph 的主要问题不是“哪个 edge 先合并”。
4. 更关键的是 component 形成后如何做 object-level 一对一竞争和边界 refinement。
```

因此后续不建议继续只调：

```text
graph-edge-coverage-power
graph-min-carrier-ioc
graph-min-node-coverage
```

更合理的下一步仍是：

```text
coverage proposal bank + evidence graph precision ranker + object-level same-frame exclusivity / competition
```

## 44. 2026-06-08 继续推进：multi-source consensus object selector 负例

### 44.1 本轮目的

前面已经证明：

```text
1. 单 prediction 内 object_competition 不能复现 oracle one-to-one 选择。
2. slotwise candidate replacement 不能稳定判断候选是否优于 slot。
3. node / edge 级 graph 局部规则也不能解决 fixed32。
```

本轮尝试一个更接近 object-level competition 的思路：

```text
把多个纯 Stream4D 来源的候选放在同一个 32f fixed support 上。
如果多个来源在 support 上重叠，说明它们可能支持同一个真实 object。
对每个 overlap group 只选择一个代表，作为 high-confidence object。
再可选地把当前 pure best 作为 low-confidence recall layer 加回。
```

该方法不读取 GT，不改 evaluator。

### 44.2 新增工具

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

核心算法：

```text
1. 读取多个 prediction config。
2. 在 fixed32 support 上计算候选 overlap。
3. 按 overlap 分组。
4. 如果开启 --require-multi-source，只保留至少两个来源共同支持的 group。
5. 每个 group 只选一个代表。
6. 可选加入 low-recall-configs 中未被 high group 覆盖的候选。
```

编译：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile tools/multi_source_consensus_select.py
```

### 44.3 运行命令

公共变量：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

SPLIT=splits/scannet_scene0050.txt
SUPPORT32=stream4d_scannet_scene0050_32f_ioc075_fixmem
SOURCES=stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95,stream4d_v4_1_scene0050_wta_plus_localbank_sil_q0p85_supportdrop0p95,stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
WEIGHTS=1.00,0.95,0.90
```

第一组，严格多来源 high-only：

```bash
cfg=stream4d_v4_1_scene0050_multisrc_consensus_msource_hi
"$PY" -u -m tools.multi_source_consensus_select \
  --seq-list "$SPLIT" \
  --source-configs "$SOURCES" \
  --source-weights "$WEIGHTS" \
  --output-config "$cfg" \
  --score-pre-points-config "$SUPPORT32" \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 0.70 \
  --require-multi-source \
  --min-support-area 1 \
  --summary-root outputs/multi_source_consensus_select_v4_1_continue
```

第二组，严格多来源 high + pure low recall：

```bash
cfg=stream4d_v4_1_scene0050_multisrc_consensus_msource_plus_purelow
"$PY" -u -m tools.multi_source_consensus_select \
  --seq-list "$SPLIT" \
  --source-configs "$SOURCES" \
  --source-weights "$WEIGHTS" \
  --low-recall-configs stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95 \
  --output-config "$cfg" \
  --score-pre-points-config "$SUPPORT32" \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 0.70 \
  --require-multi-source \
  --min-support-area 1 \
  --low-drop-overlap-mode candidate_ioc \
  --low-drop-overlap-threshold 0.85 \
  --low-score 0.01 \
  --summary-root outputs/multi_source_consensus_select_v4_1_continue
```

第三组，避免小碎片共识：IoU grouping + min area 20 + pure low recall：

```bash
cfg=stream4d_v4_1_scene0050_multisrc_consensus_iou025_min20_plus_purelow
"$PY" -u -m tools.multi_source_consensus_select \
  --seq-list "$SPLIT" \
  --source-configs "$SOURCES" \
  --source-weights "$WEIGHTS" \
  --low-recall-configs stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95 \
  --output-config "$cfg" \
  --score-pre-points-config "$SUPPORT32" \
  --group-overlap-mode iou \
  --group-overlap-threshold 0.25 \
  --require-multi-source \
  --min-support-area 20 \
  --low-min-support-area 20 \
  --low-drop-overlap-mode candidate_ioc \
  --low-drop-overlap-threshold 0.85 \
  --low-score 0.01 \
  --summary-root outputs/multi_source_consensus_select_v4_1_continue
```

第四组，min_ioc grouping + min area 20 + pure low recall：

```bash
cfg=stream4d_v4_1_scene0050_multisrc_consensus_minioc070_min20_plus_purelow
"$PY" -u -m tools.multi_source_consensus_select \
  --seq-list "$SPLIT" \
  --source-configs "$SOURCES" \
  --source-weights "$WEIGHTS" \
  --low-recall-configs stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95 \
  --output-config "$cfg" \
  --score-pre-points-config "$SUPPORT32" \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 0.70 \
  --require-multi-source \
  --min-support-area 20 \
  --low-min-support-area 20 \
  --low-drop-overlap-mode candidate_ioc \
  --low-drop-overlap-threshold 0.85 \
  --low-score 0.01 \
  --summary-root outputs/multi_source_consensus_select_v4_1_continue
```

每组之后都运行：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list "$SPLIT" \
  --pred-config "$cfg" \
  --pre-points-config "$SUPPORT32" \
  --output-config ${cfg}_cross_32fsupport \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit
```

### 44.4 结果

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

### 44.5 本轮执行结论

该方向没有达成目标。

第一版 `min_ioc` 多来源共识失败原因非常明确：

```text
多来源都覆盖的小碎片会被 min_ioc 当成强共识。
msource_hi 的 mean selected support area 只有 49.4，
AP 只有 0.060833。
```

加回 pure low recall 后也不够：

```text
msource_plus_purelow = 0.148953 / 0.446026 / 0.590000
```

把分组改成 IoU 并提高 min area 后，AP25 很高，但 AP/AP50 仍然不够：

```text
iou025_min20_plus_purelow = 0.196508 / 0.604743 / 0.882639
```

这说明跨来源共识确实能找到粗覆盖一致的候选，但仍然不能识别高 IoU 边界质量；AP25 高、AP 低的模式再次出现。

### 44.6 新 insight

```text
1. 多来源 overlap consensus 不是 oracle one-to-one 的替代品。
2. consensus 更像“粗覆盖稳定性”信号，所以 AP25 可以很高。
3. selected conflict ratio 仍为 1.0，说明这些候选在 fixed support 内高度重叠，不能解决一对一实例质量。
4. 要追 AP，必须引入边界质量和 object-level duplicate resolution，而不是只看跨来源 overlap。
```

当前状态不变：

```text
pure Stream4D fixed32 best 仍是：
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

仍未超过 Stream3D same fixed32：
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

## 45. 2026-06-08 继续推进：3D geometry slot candidate selection 负例

### 45.1 本轮目的

第 44 节说明：多来源 overlap consensus 能找到粗覆盖候选，但不能解决一对一实例质量。本轮换一个角度：

```text
不再只用 candidate overlap 分组。
先用 fixed32 support 自身的 3D geometry 做 connected-component slots。
再把 Stream4D candidates 一对一分配到这些 geometry slots。
```

该方法不读取 GT，只使用：

```text
fixed32 pre_points
ScanNet mesh vertex coordinates
pure Stream4D prediction candidates
```

### 45.2 support 几何预诊断

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
"$PY" - <<'PY'
from pathlib import Path
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
...
PY
```

关键结果：

| radius | components | big components >=20 | top component sizes |
|---:|---:|---:|---|
| 0.03 | 1501 | 71 | 1812, 636, 582, 434, 417 |
| 0.04 | 654 | 52 | 3932, 1128, 461, 407, 337 |
| 0.05 | 386 | 39 | 4129, 1773, 665, 511, 481 |
| 0.06 | 269 | 27 | 4161, 2327, 893, 544, 521 |
| 0.08 | 114 | 15 | 7639, 1319, 999, 689, 390 |
| 0.10 | 59 | 10 | 8505, 1347, 1204, 689, 91 |

结论：

```text
0.03 / 0.05 半径下 slot 数量还在可测试范围内；
0.08 以后出现大面积粘连，不适合作为实例 slot。
```

### 45.3 新增工具

新增：

```text
Stream3D/tools/geometry_slot_candidate_select.py
```

核心算法：

```text
1. 用 fixed32 support 点云做 3D connected components。
2. 每个 component 是一个 geometry slot。
3. 多个 Stream4D 来源提供候选 prediction。
4. 计算 slot_ioc / candidate_ioc / IoU / area ratio / candidate conflict。
5. 贪心地让每个 geometry slot 最多选择一个 candidate，每个 candidate 最多被一个 slot 选择。
6. 可选加回 pure best low recall layer。
```

编译：

```bash
"$PY" -m py_compile tools/geometry_slot_candidate_select.py
```

### 45.4 运行命令

公共变量：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

SPLIT=splits/scannet_scene0050.txt
SUPPORT32=stream4d_scannet_scene0050_32f_ioc075_fixmem
SOURCES=stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95,stream4d_v4_1_scene0050_wta_plus_localbank_sil_q0p85_supportdrop0p95,stream4d_v4_1_scene0050_inherit_tiered_strict_32f_compnone_mask005
WEIGHTS=1.00,0.95,0.90
```

命令模板：

```bash
"$PY" -u -m tools.geometry_slot_candidate_select \
  --seq-list "$SPLIT" \
  --source-configs "$SOURCES" \
  --source-weights "$WEIGHTS" \
  --low-recall-configs stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95 \
  --output-config "$cfg" \
  --score-pre-points-config "$SUPPORT32" \
  --slot-radius "$rad" \
  --min-slot-points 20 \
  --min-candidate-area 20 \
  --min-slot-ioc 0.20 \
  --min-candidate-ioc 0.20 \
  --min-iou 0.02 \
  --max-area-ratio 8.0 \
  --low-min-support-area 20 \
  --low-drop-overlap-mode candidate_ioc \
  --low-drop-overlap-threshold 0.85 \
  --low-score 0.01 \
  --summary-root outputs/geometry_slot_candidate_select_v4_1_continue
```

然后：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list "$SPLIT" \
  --pred-config "$cfg" \
  --pre-points-config "$SUPPORT32" \
  --output-config ${cfg}_cross_32fsupport \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/${cfg}_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit
```

### 45.5 结果

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

### 45.6 本轮执行结论

该方向没有达成目标。

3D geometry slot 能形成候选槽，但没有形成好的 instance selector：

```text
r=0.03:
AP/AP50/AP25 = 0.225242 / 0.402928 / 0.522828

r=0.05:
AP/AP50/AP25 = 0.189157 / 0.473580 / 0.589437
```

都低于 pure best：

```text
0.265385 / 0.628249 / 0.802912
```

### 45.7 失败原因

证据链：

```text
1. geometry slots 数量不少，但 high selected 很少：
   r=0.03 只选 29 个 high candidates；
   r=0.05 只选 18 个 high candidates。

2. support union 仍接近满：
   r=0.03 support union = 11911；
   r=0.05 support union = 11893。
   因此失败不是 coverage 不够。

3. 3D connected components 与真实实例不一致：
   r=0.03 top component 已有 1812 点；
   r=0.05 top component 达 4129 点。
   大组件会把多个相邻物体或相连表面粘在一起。

4. slot_ioc / candidate_ioc 看起来不低，但 AP 低，说明几何重叠不等于高 IoU 实例边界。
```

结论：

```text
fixed support 的 3D connected components 不能直接作为 GT-free instance slots。
它们可以描述几何连通区域，但不能解决 chair 场景里相邻实例的边界和一对一分配。
```

## 41. 2026-06-08 继续推进：self-discovered 2D boundary refinement 负例

### 41.1 本轮目的

第 40 节说明单纯 3D connected-component trimming 会误删有效 support。本轮继续按计划中的 `boundary-aware evidence` 方向推进，但不依赖 object_dict：

```text
对每个 3D prediction object：
1. 投影到多帧 ScanNet 2D Cropformer masks。
2. 在每帧自动选择该 object 投影点落得最多的 2D mask ID，作为 self-discovered observation。
3. 对每个 3D 点统计它在这些 observation 中是否落在主 2D mask 内。
4. 按 inside ratio 裁掉低证据点。
```

该方法不读取 GT，不改 evaluator，只使用：

```text
prediction masks
ScanNet RGB-D pose/depth
Cropformer 2D mask
fixed support pre_points
ScanNet mesh vertices
```

### 41.2 新增工具

新增：

```text
Stream3D/tools/self_discovered_boundary_refine.py
```

关键参数：

```text
--frame-stride
--max-frames
--max-observations
--min-dominant-ratio
--min-point-inside-ratio
--unobserved-policy keep|drop
--refine-support-config
--outside-refine-support drop|keep
```

编译：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile tools/self_discovered_boundary_refine.py
```

### 41.3 运行命令

输入：

```text
INPUT=stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
SUPPORT32=stream4d_scannet_scene0050_32f_ioc075_fixmem
SPLIT=splits/scannet_scene0050.txt
```

命令模板：

```bash
for spec in in30 in50 in70; do
  if [ "$spec" = in30 ]; then inratio=0.30; fi
  if [ "$spec" = in50 ]; then inratio=0.50; fi
  if [ "$spec" = in70 ]; then inratio=0.70; fi
  out=stream4d_v4_1_scene0050_purebest_selfbd_${spec}

  "$PY" -u -m tools.self_discovered_boundary_refine \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --refine-support-config "$SUPPORT32" \
    --outside-refine-support drop \
    --frame-stride 120 \
    --max-frames 50 \
    --max-observations 6 \
    --discovery-max-points 1000 \
    --depth-tolerance 0.08 \
    --boundary-margin-px 2.0 \
    --min-visible-points 8 \
    --min-dominant-points 5 \
    --min-dominant-ratio 0.35 \
    --min-point-visible-views 1 \
    --min-point-inside-ratio "$inratio" \
    --min-point-interior-ratio 0.0 \
    --unobserved-policy keep \
    --min-points-before-refine 20 \
    --min-points-after-refine 10 \
    --tmp-policy refine_support \
    --summary-root outputs/self_discovered_boundary_refine_v4_1_continue

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit
done
```

### 41.4 结果

| config | AP | AP50 | AP25 | union after | mean observations | keep ratio |
|---|---:|---:|---:|---:|---:|---:|
| previous best | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 | 52798 | NA | 1.0000 |
| `selfbd_in30` | 0.25668083900226757 | 0.6253283257747544 | 0.7814791194255479 | 11238 | 1.9085714285714286 | 0.936854966414708 |
| `selfbd_in50` | 0.2559228485572571 | 0.6257119243510372 | 0.7820159932659932 | 11135 | 1.9085714285714286 | 0.9199050965510834 |
| `selfbd_in70` | 0.2283597444145617 | 0.49511835548172756 | 0.8108324434424933 | 10340 | 1.9085714285714286 | 0.8856733498118027 |

### 41.5 本轮执行结论

该方向没有达成目标：

```text
best self-discovered boundary AP = 0.25668083900226757
previous best AP = 0.2653850988341053
Stream3D same fixed32 AP = 0.39113247863247863
```

有用的信号：

```text
selfbd_in70 的 AP25 = 0.8108324434424933，
高于 previous best AP25 = 0.8029115901631975。
```

但 AP/AP50 明显下降：

```text
selfbd_in70 AP/AP50 = 0.2283597444145617 / 0.49511835548172756
```

解释：

```text
self-discovered 2D mask observation 太粗。
它可以裁掉一部分低证据点，提高粗 IoU/AP25；
但同时会破坏高 IoU 所需的边界和完整性，导致 AP/AP50 下降。
```

结论：

```text
仅靠 prediction 自己反推主 2D mask ID，不足以做可靠 boundary refinement。
后续如果继续 boundary evidence，需要更可靠的 object-to-mask observation，例如 carrier/evidence graph 阶段记录的 mask support，而不是最终 prediction 后再猜。
```

### 41.6 编译、测试与审阅包更新

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'self_discovered_boundary_refine|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
Ran 6 tests in 0.001s
OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/self_discovered_boundary_refine.py
```

核心算法：

```text
1. 读取 full-scene prediction。
2. 读取指定 support config 的 pre_points。
3. 对每个 prediction instance，只看它在 support 内覆盖的点。
4. 用 3D 半径图找 connected components。
5. 保留最大组件或最多 K 个大组件。
6. 输出 refined prediction，分数和类别保持原样。
```

关键参数：

```text
--radius
--max-components-per-instance
--min-component-points
--min-component-ratio
--outside-support drop|keep
--tmp-policy support|input|recompute
```

编译：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile tools/support_component_refine.py
```

### 40.3 运行命令

输入：

```text
INPUT=stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
SUPPORT32=stream4d_scannet_scene0050_32f_ioc075_fixmem
SPLIT=splits/scannet_scene0050.txt
```

第一组：

```bash
for spec in r004_k1 r006_k1 r006_k2; do
  case "$spec" in
    r004_k1) rad=0.04; k=1;;
    r006_k1) rad=0.06; k=1;;
    r006_k2) rad=0.06; k=2;;
  esac
  out=stream4d_v4_1_scene0050_purebest_compref_${spec}

  "$PY" -u -m tools.support_component_refine \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --support-config "$SUPPORT32" \
    --radius "$rad" \
    --max-components-per-instance "$k" \
    --min-component-points 20 \
    --min-component-ratio 0.05 \
    --min-support-area 20 \
    --outside-support drop \
    --tmp-policy support \
    --summary-root outputs/support_component_refine_v4_1_continue

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit
done
```

第二组更宽松配置：

```bash
for spec in r010_k2 r010_k4; do
  case "$spec" in
    r010_k2) rad=0.10; k=2;;
    r010_k4) rad=0.10; k=4;;
  esac
  out=stream4d_v4_1_scene0050_purebest_compref_${spec}

  "$PY" -u -m tools.support_component_refine \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --support-config "$SUPPORT32" \
    --radius "$rad" \
    --max-components-per-instance "$k" \
    --min-component-points 20 \
    --min-component-ratio 0.03 \
    --min-support-area 20 \
    --outside-support drop \
    --tmp-policy support \
    --summary-root outputs/support_component_refine_v4_1_continue

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit
done
```

### 40.4 结果

| config | AP | AP50 | AP25 | support union after | keep ratio mean | changed instances |
|---|---:|---:|---:|---:|---:|---:|
| previous best | 0.2653850988341053 | 0.6282487663807068 | 0.8029115901631975 | 12091 | 1.0000 | 0 |
| `compref_r004_k1` | 0.15976769511408848 | 0.3474611973392462 | 0.47065410199556545 | 8231 | 0.7559661717075941 | 144 |
| `compref_r006_k1` | 0.179981844903759 | 0.3642114199310601 | 0.584951690821256 | 9566 | 0.8081890114411976 | 120 |
| `compref_r006_k2` | 0.20475655072227655 | 0.3962130376344086 | 0.7043792766373411 | 10257 | 0.8243891880152694 | 119 |
| `compref_r010_k2` | 0.24077101810367532 | 0.528074898394371 | 0.7731581425673717 | 11569 | 0.8651570650987644 | 103 |
| `compref_r010_k4` | 0.2556287929259532 | 0.5731307443401764 | 0.7731581425673717 | 11709 | 0.8706051733493791 | 103 |

### 40.5 本轮执行结论

该方向没有达成目标。最好的 `r010_k4` 仍低于 previous best：

```text
AP:   0.2556287929259532 < 0.2653850988341053
AP50: 0.5731307443401764 < 0.6282487663807068
AP25: 0.7731581425673717 < 0.8029115901631975
```

重要负证据：

```text
3D connected-component trimming 会删除大量有效 support。
即使宽松到 r=0.10 且保留 4 个组件，AP/AP50/AP25 仍全部低于 previous best。
```

这说明当前 pure Stream4D 的错误不是简单“mask 里有离散小碎片，删掉即可”。很多看似分离的 support 对 AP25/AP50 仍有贡献；粗暴的几何连通组件精炼会损失 recall 和中等 IoU 匹配。

### 40.6 编译、测试与审阅包更新

命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'support_component_refine|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
Ran 6 tests in 0.001s
OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/support_component_refine.py
```

## 46. 2026-06-08 继续推进：per-object core/fringe split 负例

### 46.1 本轮动机

第 42 节之后，pure Stream4D 的主要瓶颈仍是：

```text
同一个 fixed support 内候选很多、重叠很多，但高 IoU 一对一实例质量不够。
```

本轮测试一个更贴近算法本体的假设：

```text
对每个 prediction object，不把整张 mask 都当成同一置信度。
低冲突点作为 high-confidence core。
高冲突/边界点作为 low-confidence fringe 或 low-confidence full recall。
```

该工具不读取 GT，不修改 evaluator。

### 46.2 新增代码

新增：

```text
Stream3D/tools/split_core_fringe_prediction.py
```

## 47. 2026-06-08 继续推进：residual recall 与 localbank-high 负例

### 47.1 residual recall fuse 动机

第 46 节说明 point-level core split 不足以解决 object-level 重复。本轮继续测试一个更 object-level 的假设：

```text
localbank secondary 不应该重复解释 primary 已经覆盖的 support。
只把 secondary 在 fixed32 support 中尚未被 primary union 覆盖的 residual 区域作为低分 recall。
```

如果 localbank 的作用真是补漏，这个 residual-only 版本应该保留收益并减少重复。

### 47.2 新增代码

新增：

```text
Stream3D/tools/residual_recall_fuse.py
```

## 48. 2026-06-08 继续推进：prediction-only self-discovered silhouette score 负例

### 48.1 本轮动机

此前 `silhouette_consistency_score.py` 只能处理有 `object_dict.npy` 和 `mask_list` 的预测，而当前最强 fused pure prediction 没有 object_dict。本轮测试一个 prediction-only 版本：

```text
从最终 3D prediction mask 自己投影回 2D；
在每个 frame 自发现 dominant 2D mask；
计算可见点落回 dominant mask 内、且远离边界的比例；
把这个分数作为候选级 silhouette quality。
```

它不读取 GT，不读取 evaluator 输出，不修改 AP evaluator。

### 48.2 新增代码

新增：

```text
Stream3D/tools/self_discovered_silhouette_score.py
```

核心算法：

```text
1. 读取 full-scene prediction。
2. 可选把打分点限制在 fixed32 support 内。
3. 对每个 object，把点投影到若干 ScanNet RGB-D frame。
4. 用深度一致的可见点投票出 dominant 2D mask。
5. 统计可见点落在 dominant mask 内的比例，以及离 mask 边界足够远的比例。
6. 得到 self_silhouette_quality。
7. 使用该质量分重排、过滤或与原 score 组合。
```

语法检查：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
export OPEN3D_DISABLE_WEB_VISUALIZER=true
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile tools/self_discovered_silhouette_score.py
```

结果：通过，无输出。

### 48.3 运行命令

公共变量：

```bash
SPLIT=splits/scannet_scene0050.txt
INPUT=stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
SUPPORT32=stream4d_scannet_scene0050_32f_ioc075_fixmem
```

第一组：

```bash
for spec in score_w090 selfonly score_w090_q020 score_w090_q040; do
  case "$spec" in
    score_w090) mode=score_self_silhouette; sw=0.90; qw=0.08; minq=0.0;;
    selfonly) mode=self_silhouette; sw=0.0; qw=1.0; minq=0.0;;
    score_w090_q020) mode=score_self_silhouette; sw=0.90; qw=0.08; minq=0.20;;
    score_w090_q040) mode=score_self_silhouette; sw=0.90; qw=0.08; minq=0.40;;
  esac
  out=stream4d_v4_1_scene0050_purebest_selfsil_${spec}
  "$PY" -u -m tools.self_discovered_silhouette_score \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --score-support-config "$SUPPORT32" \
    --quality-mode "$mode" \
    --score-weight "$sw" \
    --silhouette-weight "$qw" \
    --min-self-silhouette-quality "$minq" \
    --frame-stride 20 \
    --max-frames 20 \
    --max-observations 8 \
    --discovery-max-points 800 \
    --score-max-points 1600 \
    --depth-tolerance 0.08 \
    --boundary-margin-px 2.0 \
    --summary-root outputs/self_discovered_silhouette_score_v4_1_continue \
    2>&1 | tee "logs/${out}.log"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit \
    2>&1 | tee "logs/${out}_cross_32fsupport.log"
done
```

补跑极轻量重排：

```bash
out=stream4d_v4_1_scene0050_purebest_selfsil_score_w099
"$PY" -u -m tools.self_discovered_silhouette_score \
  --seq-list "$SPLIT" \
  --input-config "$INPUT" \
  --output-config "$out" \
  --score-support-config "$SUPPORT32" \
  --quality-mode score_self_silhouette \
  --score-weight 0.99 \
  --silhouette-weight 0.01 \
  --frame-stride 20 \
  --max-frames 20 \
  --max-observations 8 \
  --discovery-max-points 800 \
  --score-max-points 1600 \
  --depth-tolerance 0.08 \
  --boundary-margin-px 2.0 \
  --summary-root outputs/self_discovered_silhouette_score_v4_1_continue \
  2>&1 | tee "logs/${out}.log"

"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list "$SPLIT" \
  --pred-config "$out" \
  --pre-points-config "$SUPPORT32" \
  --output-config ${out}_cross_32fsupport \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
  --audit-root outputs/audit \
  2>&1 | tee "logs/${out}_cross_32fsupport.log"
```

### 48.4 结果

结果文件最后一行：

```text
stream4d_v4_1_scene0050_purebest_selfsil_score_w090_cross_32fsupport_class_agnostic.txt
0.2362228624706616,0.6126225976948845,0.7393155293909851

stream4d_v4_1_scene0050_purebest_selfsil_selfonly_cross_32fsupport_class_agnostic.txt
0.14661777712702742,0.35862419935933054,0.5006255010931892

stream4d_v4_1_scene0050_purebest_selfsil_score_w090_q020_cross_32fsupport_class_agnostic.txt
0.2362228624706616,0.6126225976948845,0.7393155293909851

stream4d_v4_1_scene0050_purebest_selfsil_score_w090_q040_cross_32fsupport_class_agnostic.txt
0.2362228624706616,0.6126225976948845,0.7393155293909851

stream4d_v4_1_scene0050_purebest_selfsil_score_w099_cross_32fsupport_class_agnostic.txt
0.2362228624706616,0.6126225976948845,0.7389734948993946
```

关键 summary：

```text
instances before = 350
self_silhouette_quality min/mean/max = 0.0 / 0.3938911557197571 / 0.9943820238113403
inside_visible_ratio_mean = 0.5995742229334768
interior_ratio_mean = 0.6355530328509802
used_observations_mean = 3.6485714285714286
visible_points_mean = 381.27714285714285
dominant_ratio_mean = 0.6022126462382862
```

过滤数量：

```text
q0.20: instances after = 248, removed = 102
q0.40: instances after = 174, removed = 176
```

cross-prepoints：

```text
score_w090 union in target = 0.9899295889962338
q0.20 union in target = 0.9846897003438677
q0.40 union in target = 0.9760111347633863
```

### 48.5 本轮执行结论

没有达成目标。

对照：

```text
previous pure Stream4D best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615
```

self-discovered silhouette 最好：

```text
0.2362228624706616 / 0.6126225976948845 / 0.7393155293909851
```

解释：

```text
1. self-discovered silhouette only 明显失败。
2. 即使 99% 保留原分数，只用 1% silhouette 扰动排序，也会把 AP 从 0.265385 降到 0.236223。
3. q0.20/q0.40 能过滤大量候选，但没有改善 AP，说明它过滤掉的不是主要 false positive，或者同时删掉了有用 recall。
```

结论：

```text
prediction-only 自发现 2D silhouette quality 不适合做当前 fused prediction 的 object ranking / filtering。
该信号比有 object_dict/mask_list 的 localbank silhouette 更弱，因为它先要从最终 3D mask 反推 2D mask observation，误差链更长。
```

### 48.6 编译、测试与审阅包更新

命令：

```bash
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'self_discovered_silhouette_score|residual_recall_fuse|split_core_fringe_prediction|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
Ran 6 tests in 0.001s
OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/split_core_fringe_prediction.py
Stream3D/tools/residual_recall_fuse.py
Stream3D/tools/self_discovered_silhouette_score.py
```

算法：

```text
1. 读取 primary prediction 和 secondary prediction。
2. 读取 fixed32 support pre_points。
3. 计算 primary 在 support 内的 union。
4. 对每个 secondary：
   residual = secondary_support - primary_union_support
5. 只有 residual_area 和 residual_ratio 达标时才保留 secondary。
6. 支持三种输出方式：
   - residual_support：只输出 residual 点。
   - support_full：通过 residual 检查后输出 secondary 在 support 内的整 mask。
   - full：通过 residual 检查后输出 full-scene secondary mask。
```

不读取 GT。

语法检查：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile tools/residual_recall_fuse.py
```

结果：通过，无输出。

### 47.3 residual recall 运行命令

公共变量：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
export OPEN3D_DISABLE_WEB_VISUALIZER=true
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
SPLIT=splits/scannet_scene0050.txt
PRIMARY=stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005
SECONDARY=stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85
SUPPORT32=stream4d_scannet_scene0050_32f_ioc075_fixmem
```

运行：

```bash
for spec in residual_a10_r001 supportfull_a10_r001 full_a10_r001 residual_a50_r005 supportfull_a50_r005 full_a50_r005; do
  case "$spec" in
    residual_a10_r001) mode=residual_support; area=10; ratio=0.01;;
    supportfull_a10_r001) mode=support_full; area=10; ratio=0.01;;
    full_a10_r001) mode=full; area=10; ratio=0.01;;
    residual_a50_r005) mode=residual_support; area=50; ratio=0.05;;
    supportfull_a50_r005) mode=support_full; area=50; ratio=0.05;;
    full_a50_r005) mode=full; area=50; ratio=0.05;;
  esac
  out=stream4d_v4_1_scene0050_residual_${spec}
  "$PY" -u -m tools.residual_recall_fuse \
    --seq-list "$SPLIT" \
    --primary-config "$PRIMARY" \
    --secondary-config "$SECONDARY" \
    --output-config "$out" \
    --support-config "$SUPPORT32" \
    --primary-score -1 \
    --secondary-score 0.005 \
    --min-residual-area "$area" \
    --min-residual-ratio "$ratio" \
    --secondary-mode "$mode" \
    --summary-root outputs/residual_recall_fuse_v4_1_continue \
    2>&1 | tee "logs/${out}.log"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit \
    2>&1 | tee "logs/${out}_cross_32fsupport.log"
done
```

结果文件最后一行：

```text
stream4d_v4_1_scene0050_residual_residual_a10_r001_cross_32fsupport_class_agnostic.txt
0.2530976765890559,0.6158322281167109,0.8029115901631975

stream4d_v4_1_scene0050_residual_supportfull_a10_r001_cross_32fsupport_class_agnostic.txt
0.2530976765890559,0.6158322281167109,0.8029115901631975

stream4d_v4_1_scene0050_residual_full_a10_r001_cross_32fsupport_class_agnostic.txt
0.2530976765890559,0.6158322281167109,0.8029115901631975

stream4d_v4_1_scene0050_residual_residual_a50_r005_cross_32fsupport_class_agnostic.txt
0.2530976765890559,0.6158322281167109,0.8029115901631975

stream4d_v4_1_scene0050_residual_supportfull_a50_r005_cross_32fsupport_class_agnostic.txt
0.2530976765890559,0.6158322281167109,0.8029115901631975

stream4d_v4_1_scene0050_residual_full_a50_r005_cross_32fsupport_class_agnostic.txt
0.2530976765890559,0.6158322281167109,0.8029115901631975
```

关键 summary：

```text
mean_primary_support_union = 12084.0
mean_uncovered_support_count = 130.0
mean_num_residual_instances = 0.0
mean_num_output_instances = 337.0
mean_dropped_empty = 3.0
mean_dropped_small = 11.0
```

解释：

```text
fixed32 support 总点数为 12214。
primary 已覆盖 12084 点，只剩 130 个未覆盖点。
localbank secondary 没有任何候选能在未覆盖区域贡献足够 residual。
```

### 47.4 localbank-high 运行命令

由于 residual-only 说明 localbank 的作用不是补未覆盖点，本轮进一步验证：

```text
如果 localbank 是更好的替代候选，把它放到 high-confidence primary 是否会更好？
```

运行：

```bash
LOCAL=stream4d_v4_1_scene0050_localbank_minobs5_sil_q0p85
STRICT=stream4d_v4_1_scene0050_strict_32f_secioc0p50_comp_plus_mask005

for drop in 0.50 0.85 0.95; do
  tag=${drop/./p}
  out=stream4d_v4_1_scene0050_localhigh_q85_strictlow_drop${tag}
  "$PY" -u -m tools.fuse_prediction_configs \
    --seq-list "$SPLIT" \
    --primary-config "$LOCAL" \
    --secondary-config "$STRICT" \
    --output-config "$out" \
    --primary-score -1 \
    --secondary-score 0.005 \
    --drop-secondary-iou-threshold "$drop" \
    --drop-secondary-overlap-mode secondary_ioc \
    --drop-overlap-pre-points-config "$SUPPORT32" \
    --summary-root outputs/fuse_prediction_configs_v4_1_continue \
    2>&1 | tee "logs/${out}.log"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit \
    2>&1 | tee "logs/${out}_cross_32fsupport.log"
done
```

结果文件最后一行：

```text
stream4d_v4_1_scene0050_localhigh_q85_strictlow_drop0p50_cross_32fsupport_class_agnostic.txt
0.1281189467312349,0.2756295399515739,0.491372578692494

stream4d_v4_1_scene0050_localhigh_q85_strictlow_drop0p85_cross_32fsupport_class_agnostic.txt
0.13733906525573192,0.29440476190476195,0.4842162698412699

stream4d_v4_1_scene0050_localhigh_q85_strictlow_drop0p95_cross_32fsupport_class_agnostic.txt
0.13733906525573192,0.29440476190476195,0.4842162698412699
```

summary：

```text
drop0p50:
mean_num_primary_instances = 14.0
mean_num_secondary_instances = 281.0
mean_num_output_instances = 295.0
mean_prediction_union_in_target_ratio_of_target = 0.9439168167676437

drop0p85:
mean_num_primary_instances = 14.0
mean_num_secondary_instances = 299.0
mean_num_output_instances = 313.0
mean_prediction_union_in_target_ratio_of_target = 0.9895202226952677

drop0p95:
mean_num_primary_instances = 14.0
mean_num_secondary_instances = 308.0
mean_num_output_instances = 322.0
mean_prediction_union_in_target_ratio_of_target = 0.9896020959554609
```

### 47.5 本轮执行结论

没有达成目标。

对照：

```text
Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615

previous pure Stream4D best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

本轮结果：

```text
residual recall 全部退回 strict primary:
0.2530976765890559 / 0.6158322281167109 / 0.8029115901631975

localbank-high 最好只有:
0.13733906525573192 / 0.29440476190476195 / 0.4842162698412699
```

解释：

```text
1. localbank 的收益不是来自补 primary 完全未覆盖的 support 点。
2. localbank 不能作为 high-confidence primary；它的边界和一对一质量太弱。
3. localbank 只能作为低分重叠替代/召回层带来小幅 AP 增益。
4. 这进一步说明需要的是 object-level 替代候选选择，而不是 residual uncovered set-cover。
```

### 47.6 编译、测试与审阅包更新

命令：

```bash
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'residual_recall_fuse|split_core_fringe_prediction|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
Ran 6 tests in 0.001s
OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/split_core_fringe_prediction.py
Stream3D/tools/residual_recall_fuse.py
```

核心算法：

```text
1. 读取 input prediction。
2. 读取指定 support config 的 pre_points。
3. 在 support 内统计每个点被多少个 prediction object 同时占有。
4. 对每个 object：
   - owner_count <= max_core_owners 的点作为 core。
   - 根据 low_mode，把 full mask 或 conflict fringe 作为低分层。
5. 输出新的 prediction config 和 summary JSON。
```

关键参数：

```text
--max-core-owners
--low-mode none|full|support_full|fringe|fringe_plus_core
--core-score，负数表示保留原 prediction score
--low-score
--tmp-policy support|input|recompute
```

语法检查：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile tools/split_core_fringe_prediction.py
```

结果：通过，无输出。

### 46.3 第一组：激进 core + low full/fringe

公共变量：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
export OPEN3D_DISABLE_WEB_VISUALIZER=true
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
SPLIT=splits/scannet_scene0050.txt
INPUT=stream4d_v4_1_scene0050_pure_plus_localbank_sil_q0p85_supportdrop0p95
SUPPORT32=stream4d_scannet_scene0050_32f_ioc075_fixmem
```

运行命令：

```bash
for spec in own1_full own1_fringe own2_full own2_fringe; do
  case "$spec" in
    own1_full) owners=1; low_mode=full;;
    own1_fringe) owners=1; low_mode=fringe;;
    own2_full) owners=2; low_mode=full;;
    own2_fringe) owners=2; low_mode=fringe;;
  esac
  out=stream4d_v4_1_scene0050_corefringe_${spec}

  "$PY" -u -m tools.split_core_fringe_prediction \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --support-config "$SUPPORT32" \
    --max-core-owners "$owners" \
    --low-mode "$low_mode" \
    --core-score -1 \
    --low-score 0.01 \
    --min-core-points 10 \
    --min-core-ratio 0.05 \
    --min-low-points 10 \
    --min-support-area 10 \
    --tmp-policy support \
    --summary-root outputs/stream4d_core_fringe_v4_1_continue \
    2>&1 | tee "logs/${out}.log"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit \
    2>&1 | tee "logs/${out}_cross_32fsupport.log"
done
```

结果文件最后一行：

```text
stream4d_v4_1_scene0050_corefringe_own1_full_cross_32fsupport_class_agnostic.txt
0.0827699530516432,0.18950704225352114,0.35373239436619713

stream4d_v4_1_scene0050_corefringe_own1_fringe_cross_32fsupport_class_agnostic.txt
0.07913764510779436,0.19455223880597014,0.3374626865671642

stream4d_v4_1_scene0050_corefringe_own2_full_cross_32fsupport_class_agnostic.txt
0.1476388888888889,0.39666666666666667,0.51953125

stream4d_v4_1_scene0050_corefringe_own2_fringe_cross_32fsupport_class_agnostic.txt
0.14194716775599125,0.39526960784313725,0.5378676470588235
```

### 46.4 第二组：更软 core，只降级严重冲突点

运行命令：

```bash
for owners in 3 5 8; do
  out=stream4d_v4_1_scene0050_corefringe_own${owners}_full
  "$PY" -u -m tools.split_core_fringe_prediction \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --support-config "$SUPPORT32" \
    --max-core-owners "$owners" \
    --low-mode full \
    --core-score -1 \
    --low-score 0.01 \
    --min-core-points 10 \
    --min-core-ratio 0.05 \
    --min-low-points 10 \
    --min-support-area 10 \
    --tmp-policy support \
    --summary-root outputs/stream4d_core_fringe_v4_1_continue \
    2>&1 | tee "logs/${out}.log"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit \
    2>&1 | tee "logs/${out}_cross_32fsupport.log"
done
```

结果文件最后一行：

```text
stream4d_v4_1_scene0050_corefringe_own3_full_cross_32fsupport_class_agnostic.txt
0.24237695924764888,0.5431630094043887,0.749276645768025

stream4d_v4_1_scene0050_corefringe_own5_full_cross_32fsupport_class_agnostic.txt
0.24023681602167501,0.590468634617362,0.7731581425673717

stream4d_v4_1_scene0050_corefringe_own8_full_cross_32fsupport_class_agnostic.txt
0.262385435534442,0.6253881603201007,0.8029115901631975
```

### 46.5 第三组：core only，不保留低分层

运行命令：

```bash
for owners in 5 8 12; do
  out=stream4d_v4_1_scene0050_coreonly_own${owners}
  "$PY" -u -m tools.split_core_fringe_prediction \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --support-config "$SUPPORT32" \
    --max-core-owners "$owners" \
    --low-mode none \
    --core-score -1 \
    --min-core-points 10 \
    --min-core-ratio 0.05 \
    --min-support-area 10 \
    --tmp-policy support \
    --summary-root outputs/stream4d_core_fringe_v4_1_continue \
    2>&1 | tee "logs/${out}.log"

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --pre-points-config "$SUPPORT32" \
    --output-config ${out}_cross_32fsupport \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file data/evaluation/scannet/${out}_cross_32fsupport_class_agnostic.txt \
    --audit-root outputs/audit \
    2>&1 | tee "logs/${out}_cross_32fsupport.log"
done
```

结果文件最后一行：

```text
stream4d_v4_1_scene0050_coreonly_own5_cross_32fsupport_class_agnostic.txt
0.24236333599163215,0.5933966743195953,0.7731581425673717

stream4d_v4_1_scene0050_coreonly_own8_cross_32fsupport_class_agnostic.txt
0.2653850988341053,0.6282487663807068,0.8029115901631975

stream4d_v4_1_scene0050_coreonly_own12_cross_32fsupport_class_agnostic.txt
0.2653850988341053,0.6282487663807068,0.8029115901631975
```

### 46.6 诊断字段

读取命令：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
$PY - <<'PY'
import json
from pathlib import Path
files = [
  'Stream3D/outputs/stream4d_core_fringe_v4_1_continue/stream4d_v4_1_scene0050_corefringe_own1_full_summary.json',
  'Stream3D/outputs/stream4d_core_fringe_v4_1_continue/stream4d_v4_1_scene0050_corefringe_own2_full_summary.json',
  'Stream3D/outputs/stream4d_core_fringe_v4_1_continue/stream4d_v4_1_scene0050_corefringe_own8_full_summary.json',
  'Stream3D/outputs/stream4d_core_fringe_v4_1_continue/stream4d_v4_1_scene0050_coreonly_own5_summary.json',
  'Stream3D/outputs/stream4d_core_fringe_v4_1_continue/stream4d_v4_1_scene0050_coreonly_own8_summary.json',
]
for raw in files:
    p = Path(raw)
    d = json.loads(p.read_text())['aggregate']
    print(p.name)
    for k in ['low_mode','mean_num_instances_after','mean_num_core_instances','mean_num_low_instances','mean_output_support_union','mean_output_support_conflict_ratio','mean_mean_core_ratio','mean_mean_conflict_ratio']:
        print(f'  {k}={d.get(k)}')
PY
```

关键输出：

```text
own1_full:
mean_core_ratio=0.08653449334592832
mean_conflict_ratio=0.9134655066540717
mean_num_instances_after=275.0

own2_full:
mean_core_ratio=0.29509968463982533
mean_conflict_ratio=0.7049003153601747
mean_num_instances_after=337.0

own8_full:
mean_core_ratio=0.9833448833647003
mean_conflict_ratio=0.0166551166352998
mean_num_instances_after=482.0

coreonly_own5:
mean_core_ratio=0.9069401065964906
mean_conflict_ratio=0.0930598934035094
mean_output_support_union=11897.0

coreonly_own8:
mean_core_ratio=0.9833448833647003
mean_conflict_ratio=0.0166551166352998
mean_output_support_union=12045.0
```

### 46.7 本轮执行结论

该方向没有达成目标：

```text
Stream3D same fixed32:
0.39113247863247863 / 0.6461538461538462 / 0.7615384615384615

previous pure Stream4D best:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975

best core/fringe:
0.2653850988341053 / 0.6282487663807068 / 0.8029115901631975
```

解释：

```text
1. owner<=1/2 的 core 太小，AP 直接崩。
2. owner<=3/5 比较保守，但仍低于 previous pure best。
3. owner<=8/12 等价或几乎等价于原预测，不能提供新收益。
4. 低分 full/fringe 层没有解决 high-IoU object assignment，反而在激进 core 下带来重复候选。
```

### 46.8 编译、测试与审阅包更新

命令：

```bash
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
"$PY" -m py_compile stream4d/*.py tools/*.py tests/*.py
"$PY" -m unittest tests.test_stream4d_protocol_fixes

cd /mnt/data/users/chengshun.wang/pjs/LoGeR
zip -q -@ stream4d_v4_1_code_review_packet.zip < stream4d_v4_1_code_review_packet_filelist.txt
zip -T stream4d_v4_1_code_review_packet.zip
zipinfo -1 stream4d_v4_1_code_review_packet.zip | \
  rg 'split_core_fringe_prediction|stream4d_v4_1_执行日志|stream4d_v4_1_实验结果复盘'
```

结果：

```text
Ran 6 tests in 0.002s
OK
test of stream4d_v4_1_code_review_packet.zip OK
docs/stream4d_v4_1_执行日志.md
docs/stream4d_v4_1_实验结果复盘.md
Stream3D/tools/split_core_fringe_prediction.py
```

## 49. 2026-06-08 继续推进：probe5 多场景 fixed-support 诊断

### 49.1 本轮目的

上一批大量实验都集中在 `scene0050_00`。为了避免把单场景结论误写成稳定结论，本轮新增 5 场景 probe split：

```text
Stream3D/splits/scannet_v4_1_probe5.txt

scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00
```

本轮不使用 GT 生成 prediction，不修改 evaluator，只使用 `tools.evaluate_cross_prepoints` 做 fixed-support 诊断。

### 49.2 缓存检查

命令：

```bash
set -euo pipefail
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
for cfg in scannet \
  stream4d_scannet_32f_ioc075_fixmem \
  stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit; do
  echo "CONFIG $cfg"
  for s in $(cat splits/scannet_v4_1_probe5.txt); do
    p="data/prediction/${cfg}_class_agnostic/${s}.npz"
    t="data/TMP/${cfg}/${s}_pre_points.npy"
    printf "%s pred=%s tmp=%s\n" "$s" \
      "$([ -f "$p" ] && echo yes || echo no)" \
      "$([ -f "$t" ] && echo yes || echo no)"
  done
done
```

结果：

```text
scannet: 5/5 prediction exists, but data/TMP/scannet does not exist.
stream4d_scannet_32f_ioc075_fixmem: 5/5 prediction and TMP exist.
stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute: 5/5 prediction and TMP exist.
stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit: 5/5 prediction and TMP exist.
```

`scannet` 的 source pre_points 使用当前仓库已有的 `scannet_self_inherit`，因为该 config 对 5 个 probe scenes 都有 `data/TMP/<config>/<scene>_pre_points.npy`。

额外缓存结论：

```text
scene0050_00 有 32f / 96f / 128f 多窗口 carrier cache。
scene0011_00、scene0030_00、scene0081_01、scene0591_00 当前只有 32f 单窗口 cache：
  outputs/stream4d_debug_full_32f_ioc075_fixmem/<scene>/carriers_window000.npz
  outputs/stream4d_debug_full_32f_ioc075_fixmem/<scene>/local_props_window000.json
```

因此本轮不能诚实地把 scene0050 的 96f/128f evidence graph 直接扩展到 probe5；要多场景验证 evidence graph，必须重新生成这些场景的 96f/128f carrier cache。

### 49.3 第一组 fixed-support 对照命令

环境：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
export OPEN3D_DISABLE_WEB_VISUALIZER=true
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
SPLIT=splits/scannet_v4_1_probe5.txt
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
```

通用函数：

```bash
run_cross() {
  local pred="$1"
  local source="$2"
  local target="$3"
  local out="$4"
  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$pred" \
    --source-pre-points-config "$source" \
    --pre-points-config "$target" \
    --output-config "$out" \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file "data/evaluation/scannet/${out}_class_agnostic.txt" \
    --audit-root outputs/audit
}
```

执行：

```bash
run_cross scannet scannet_self_inherit scannet_self_inherit scannet_self_inherit_probe5
run_cross scannet scannet_self_inherit "$SUPPORT" scannet_on_stream4d_32f_probe5
run_cross "$SUPPORT" "$SUPPORT" "$SUPPORT" stream4d_32f_self_probe5
run_cross stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_recompute \
  "$SUPPORT" \
  stream4d_v3_adapt_recompute_on_32f_probe5
run_cross stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit \
  stream4d_scannet_32f_ioc075_fixmem_v3_adapt014_8_18_maskcount_one_inherit \
  "$SUPPORT" \
  stream4d_v3_adapt_inherit_on_32f_probe5
```

生成结果文件：

```text
data/evaluation/scannet/scannet_self_inherit_probe5_class_agnostic.txt
data/evaluation/scannet/scannet_on_stream4d_32f_probe5_class_agnostic.txt
data/evaluation/scannet/stream4d_32f_self_probe5_class_agnostic.txt
data/evaluation/scannet/stream4d_v3_adapt_recompute_on_32f_probe5_class_agnostic.txt
data/evaluation/scannet/stream4d_v3_adapt_inherit_on_32f_probe5_class_agnostic.txt
```

对应 audit summary：

```text
outputs/audit/cross_prepoints/scannet_self_inherit_probe5_summary.json
outputs/audit/cross_prepoints/scannet_on_stream4d_32f_probe5_summary.json
outputs/audit/cross_prepoints/stream4d_32f_self_probe5_summary.json
outputs/audit/cross_prepoints/stream4d_v3_adapt_recompute_on_32f_probe5_summary.json
outputs/audit/cross_prepoints/stream4d_v3_adapt_inherit_on_32f_probe5_summary.json
```

### 49.4 第二组：已有 32f 候选选择变体扫描

命令：

```bash
for cfg in \
  stream4d_scannet_32f_ioc075_fixmem_top10_mask_count_one \
  stream4d_scannet_32f_ioc075_fixmem_top12_mask_count_one \
  stream4d_scannet_32f_ioc075_fixmem_top10_area_one \
  stream4d_scannet_32f_ioc075_fixmem_top12_area_one \
  stream4d_scannet_32f_ioc075_fixmem_one_min100 \
  stream4d_scannet_32f_ioc075_fixmem_one_min250 \
  stream4d_scannet_32f_ioc075_fixmem_one_min250_merge070 \
  stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one; do
  out="${cfg}_on_32f_probe5"
  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$cfg" \
    --source-pre-points-config "$cfg" \
    --pre-points-config "$SUPPORT" \
    --output-config "$out" \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file "data/evaluation/scannet/${out}_class_agnostic.txt" \
    --audit-root outputs/audit
done
```

生成的结果文件位于：

```text
data/evaluation/scannet/*_on_32f_probe5_class_agnostic.txt
outputs/audit/cross_prepoints/*_on_32f_probe5_summary.json
```

### 49.5 结果表

以下数值均为 evaluator 输出文件最后一行原始值，未乘以 100。

| method | AP | AP50 | AP25 | target pre % | union % | union in target % | #pred | mask shape |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Stream3D self-inherit probe5 | 0.235730 | 0.414306 | 0.537786 | 84.6744 | 84.6744 | 100.0000 | 128.20 | full_scene |
| Stream3D on 32f support probe5 | 0.399213 | 0.597171 | 0.742535 | 4.5145 | 84.6744 | 98.5608 | 128.20 | full_scene |
| Stream4D 32f self probe5 | 0.144238 | 0.288344 | 0.464716 | 4.5145 | 4.5145 | 100.0000 | 386.00 | full_scene |
| Stream4D v3 recompute pred on 32f probe5 | 0.114468 | 0.241335 | 0.430399 | 4.5145 | 2.9076 | 67.4358 | 17.00 | full_scene |
| Stream4D v3 inherit pred on 32f probe5 | 0.114468 | 0.241335 | 0.430399 | 4.5145 | 2.9076 | 67.4358 | 17.00 | full_scene |
| 32f top10 mask_count | 0.108129 | 0.236172 | 0.378560 | 4.5145 | 2.4819 | 58.0059 | 10.00 | full_scene |
| 32f top12 mask_count | 0.105284 | 0.236475 | 0.384597 | 4.5145 | 2.5959 | 60.8728 | 12.00 | full_scene |
| 32f top10 area | 0.085730 | 0.168675 | 0.316145 | 4.5145 | 2.6630 | 61.2535 | 10.00 | full_scene |
| 32f top12 area | 0.099152 | 0.191667 | 0.373896 | 4.5145 | 2.8264 | 65.0526 | 12.00 | full_scene |
| 32f one_min100 | 0.144238 | 0.288344 | 0.464716 | 4.5145 | 3.4900 | 77.2712 | 25.20 | full_scene |
| 32f one_min250 | 0.098032 | 0.202289 | 0.344096 | 4.5145 | 2.7125 | 61.2853 | 10.00 | full_scene |
| 32f one_min250_merge070 | 0.104104 | 0.206357 | 0.352986 | 4.5145 | 2.8215 | 62.9207 | 9.40 | full_scene |
| 32f adapt014 old | 0.114468 | 0.241335 | 0.430399 | 4.5145 | 2.9076 | 67.4358 | 17.00 | full_scene |

### 49.6 本轮执行结论

本轮没有达成 Stream4D 在 inherit/fixed support 下超过 Stream3D。

最关键对比：

```text
Stream3D on 32f support probe5:
0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108

Stream4D 32f self probe5:
0.14423821149252213 / 0.288344136107294 / 0.46471570934903785

Stream4D v3 inherit/on-32f probe5:
0.114468 / 0.241335 / 0.430399
```

已有 32f candidate selection 变体也没有改善；top-k、area、min-points、merge070、adapt014 都低于原始 `stream4d_32f_self_probe5`。

### 49.7 审计说明

本轮的 cross-prepoints audit 显示所有 probe5 结果都是 `mask_shape_mode=full_scene`，没有 cropped mask shape 扩展导致的隐藏 shape bug。

本轮新建文件：

```text
Stream3D/splits/scannet_v4_1_probe5.txt
```

该文件已加入 `stream4d_v4_1_code_review_packet_filelist.txt`，后续审阅包会包含它。

## 50. 2026-06-08 继续推进：probe5 cached evidence graph / component densify

### 50.1 本轮目的

第 49 节证明 pure Stream4D 在 probe5 的 fixed support 下仍低于 Stream3D。按照复盘建议，下一步应补多场景 96f/128f cache，再验证 evidence graph / component densify 是否能跨场景成立。

本轮先尝试真实生成 `scene0011_00` 的 96f D4RT cache；该尝试没有成功进入窗口推理阶段。随后用已经存在的 probe5 32f carrier cache 做 replay 诊断，验证算法本身在多场景上是否比原始 32f 更好。

### 50.2 真实 96f D4RT cache 尝试

命令：

```bash
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
export OPEN3D_DISABLE_WEB_VISUALIZER=true
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 "$PY" -u -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0011_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 96 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 8 \
  --min-points-per-mask 2 \
  --query-chunk-size 1024 \
  --rho-min 0.35 \
  --local-ioc-threshold 0.75 \
  --history-match-threshold 0.30 \
  --lost-tolerance-windows 3 \
  --memory-version old \
  --export-mode rgbd_eval \
  --export-nn-radius 0.08 \
  --output-config stream4d_v4_1_memoryold_scene0011_96f_ioc075 \
  --debug-root outputs/stream4d_debug_v4_1_memoryold_scene0011_96f \
  2>&1 | tee logs/stream4d_v4_1_memoryold_scene0011_96f_run.log
```

观察：

```text
ps 显示 python 进程处于 D state。
日志文件 logs/stream4d_v4_1_memoryold_scene0011_96f_run.log 大小为 0 bytes。
GPU 查询没有看到该进程进入 GPU compute app。
outputs/stream4d_debug_v4_1_memoryold_scene0011_96f 不存在。
data/prediction/stream4d_v4_1_memoryold_scene0011_96f_ioc075_class_agnostic/scene0011_00.npz 不存在。
```

处理：

```bash
pids=$(ps -o pid,args -u $(whoami) | awk '/stream4d.run_scannet/ && !/awk/ {print $1}')
echo "$pids" | xargs -r kill -TERM
```

结论：

```text
这次 96f cache 生成没有产生任何实验数据，不能写入结果表。
它是 D4RT checkpoint / filesystem I/O 初始化阶段 blocker，而不是算法指标失败。
```

### 50.3 使用已有 32f carrier cache 的 evidence graph replay

已有 cache：

```text
outputs/stream4d_debug_full_32f_ioc075_fixmem/<scene>/carriers_window000.npz
```

覆盖 scenes：

```text
scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00
```

严格配置命令：

```bash
CFG=stream4d_v4_1_egraph_32f_probe5_ioc0p79_minobs10_compdens_r003_d16_m8_rel100_hard2
for seq in $(cat splits/scannet_v4_1_probe5.txt); do
  "$PY" -u -m stream4d.replay_evidence_graph \
    --seq-name "$seq" \
    --backbone Cropformer \
    --frame-stride 10 \
    --max-frames 32 \
    --window-size 32 \
    --window-stride 16 \
    --rho-min 0.35 \
    --graph-min-shared-carriers 2 \
    --graph-min-carrier-ioc 0.79 \
    --graph-min-component-observations 10 \
    --export-support-mode component_densify \
    --export-core-nn-radius 0.03 \
    --densify-seed-distance-px 16 \
    --export-max-masks-per-object 8 \
    --export-mask-min-relative-coverage 1.0 \
    --export-mask-sample-stride 2 \
    --export-mask-max-pixels 12000 \
    --export-enable-wta \
    --export-wta-score-mode compactness \
    --export-wta-min-conflict-owners 2 \
    --input-debug-root outputs/stream4d_debug_full_32f_ioc075_fixmem \
    --output-config "$CFG" \
    --debug-root outputs/stream4d_egraph_probe5_32f \
    2>&1 | tee "logs/${CFG}_${seq}.log"
done
```

严格配置输出：

```text
scene0050_00: objects=18 points=3975 hit_rate=0.9127
scene0011_00: objects=7 points=3673 hit_rate=0.7628
scene0030_00: objects=10 points=1382 hit_rate=0.8450
scene0081_01: objects=9 points=2612 hit_rate=0.7105
scene0591_00: objects=32 points=2414 hit_rate=0.8965
```

评估命令：

```bash
"$PY" -u -m evaluation.evaluate \
  --pred_path "data/prediction/${CFG}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${CFG}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$CFG" \
  --no_class

"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --pred-config "$CFG" \
  --source-pre-points-config "$CFG" \
  --pre-points-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config "${CFG}_on_32f_probe5" \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${CFG}_on_32f_probe5_class_agnostic.txt" \
  --audit-root outputs/audit
```

严格配置结果：

```text
own recompute:
0.563837 / 0.726708 / 0.786749

fixed 32f support:
0.030578 / 0.079899 / 0.131896
```

### 50.4 放宽 evidence graph / component densify 扫描

扫描逻辑：

```text
1. 降低 export-mask-min-relative-coverage：1.0 -> 0.5 -> 0.0
2. 降低 graph-min-component-observations：10 -> 5 -> 3 -> 1
3. 轻微放宽 graph-min-carrier-ioc：0.79 -> 0.70 -> 0.60
```

命令模板：

```bash
run_variant() {
  local cfg="$1"; local ioc="$2"; local minobs="$3"; local rel="$4"
  for seq in $(cat splits/scannet_v4_1_probe5.txt); do
    "$PY" -u -m stream4d.replay_evidence_graph \
      --seq-name "$seq" \
      --backbone Cropformer \
      --frame-stride 10 \
      --max-frames 32 \
      --window-size 32 \
      --window-stride 16 \
      --rho-min 0.35 \
      --graph-min-shared-carriers 2 \
      --graph-min-carrier-ioc "$ioc" \
      --graph-min-component-observations "$minobs" \
      --export-support-mode component_densify \
      --export-core-nn-radius 0.03 \
      --densify-seed-distance-px 16 \
      --export-max-masks-per-object 8 \
      --export-mask-min-relative-coverage "$rel" \
      --export-mask-sample-stride 2 \
      --export-mask-max-pixels 12000 \
      --export-enable-wta \
      --export-wta-score-mode compactness \
      --export-wta-min-conflict-owners 2 \
      --input-debug-root outputs/stream4d_debug_full_32f_ioc075_fixmem \
      --output-config "$cfg" \
      --debug-root outputs/stream4d_egraph_probe5_32f \
      > "logs/${cfg}_${seq}.log" 2>&1
  done
  "$PY" -u -m evaluation.evaluate ...
  "$PY" -u -m tools.evaluate_cross_prepoints ...
}
```

实际运行 configs：

```text
stream4d_v4_1_egraph_32f_probe5_ioc0p79_minobs10_compdens_r003_d16_m8_rel050_hard2
stream4d_v4_1_egraph_32f_probe5_ioc0p79_minobs5_compdens_r003_d16_m8_rel050_hard2
stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs5_compdens_r003_d16_m8_rel050_hard2
stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs5_compdens_r003_d16_m8_rel000_hard2
stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs3_compdens_r003_d16_m8_rel000_hard2
stream4d_v4_1_egraph_32f_probe5_ioc0p60_minobs3_compdens_r003_d16_m8_rel000_hard2
stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs1_compdens_r003_d16_m8_rel000_hard2
```

### 50.5 结果表

数值来自 `data/evaluation/scannet/*.txt` 最后一行，未乘以 100。

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

### 50.6 本轮执行结论

本轮仍没有达成 Stream4D 在 fixed/inherit support 下超过 Stream3D。

对比第 49 节 baseline：

```text
Stream3D on 32f support probe5:
0.399213 / 0.597171 / 0.742535

Stream4D 32f self probe5:
0.144238 / 0.288344 / 0.464716

本轮最佳 cached evidence graph fixed32:
0.240665 / 0.447128 / 0.671741
```

本轮达成的局部进展：

```text
cached evidence graph + component densify 在 probe5 fixed support 下超过原始 Stream4D 32f self：
AP   0.240665 > 0.144238
AP50 0.447128 > 0.288344
AP25 0.671741 > 0.464716
```

但没有超过 Stream3D：

```text
AP   0.240665 < 0.399213
AP50 0.447128 < 0.597171
AP25 0.671741 < 0.742535
```

### 50.7 失败原因

1. `rel1.0/minobs10` 证明 high precision subset 很强，但 coverage 太小：`union in target = 14.0954%`，fixed AP 只有 0.030578。
2. 放宽到 `rel0.0/minobs3` 后，`union in target = 88.2922%`，fixed AP 提高到 0.240665，说明 coverage 是 fixed-support 的必要条件。
3. 继续放宽到 `minobs1`，`union in target = 91.9782%`，但 #pred 增到 324，AP 降到 0.186444，说明过多低稳定候选造成碎片化和 false positives。
4. `ioc0.60` 没有优于 `ioc0.70`，说明过度放宽 graph matching 也会带来噪声。

结论：

```text
component densify 的有效区间在 minobs3/minobs5 附近；
它能修复原始 Stream4D 32f 的一部分 object quality，但仍没有达到 Stream3D 的一对一实例质量。
```

## 51. 2026-06-08 继续推进：probe5 pure Stream4D tiered inherit 与 oracle 诊断

### 51.1 本轮目的

第 50 节最佳 cached evidence graph：

```text
stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs3_compdens_r003_d16_m8_rel000_hard2
fixed32 = 0.240665 / 0.447128 / 0.671741
```

它已经超过原始 Stream4D 32f self，但仍低于 Stream3D。按 scene0050 的经验，下一步尝试纯 Stream4D tiered inherit：

```text
高分层：evidence graph + component densify，score=1.0
低分层：Stream4D 32f current，score=0.2 或 0.05
目标：保留 egraph 的较好 object quality，同时用 32f current 补完整 coverage。
```

### 51.2 fusion 命令

环境：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PRIMARY=stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs3_compdens_r003_d16_m8_rel000_hard2
SECONDARY=stream4d_scannet_32f_ioc075_fixmem
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
```

通用命令：

```bash
run_fuse_eval() {
  local out="$1"; local drop="$2"; local mode="$3"; local secscore="$4"
  "$PY" -u -m tools.fuse_prediction_configs \
    --root . \
    --seq-list splits/scannet_v4_1_probe5.txt \
    --primary-config "$PRIMARY" \
    --secondary-config "$SECONDARY" \
    --output-config "$out" \
    --fusion-mode concatenate \
    --primary-score 1.0 \
    --secondary-score "$secscore" \
    --drop-secondary-iou-threshold "$drop" \
    --drop-secondary-overlap-mode "$mode" \
    --drop-overlap-pre-points-config "$SUPPORT" \
    --summary-root outputs/stream4d_fusion_probe5

  "$PY" -u -m evaluation.evaluate \
    --pred_path "data/prediction/${out}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$out" \
    --no_class

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list splits/scannet_v4_1_probe5.txt \
    --pred-config "$out" \
    --source-pre-points-config "$out" \
    --pre-points-config "$SUPPORT" \
    --output-config "${out}_on_32f_probe5" \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file "data/evaluation/scannet/${out}_on_32f_probe5_class_agnostic.txt" \
    --audit-root outputs/audit
}
```

实际 configs：

```text
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_nodrop
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc050
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc050_s005
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_minioc050
```

### 51.3 support-aware ranking / competition 命令

输入：

```text
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
```

命令模板：

```bash
"$PY" -u -m tools.support_aware_object_rank \
  --root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --input-config "$IN" \
  --output-config "$out" \
  --score-pre-points-config "$SUPPORT" \
  --quality-mode score_support_area_conflict_penalty \
  --score-weight 0.90 \
  --summary-root outputs/stream4d_support_aware_rank_probe5
```

实际完成的 configs：

```text
stream4d_v4_1_probe5_tiered_egraph32flow_saware_scoreconf_w090
stream4d_v4_1_probe5_tiered_egraph32flow_saware_scoreconf_w075
stream4d_v4_1_probe5_tiered_egraph32flow_saware_supportconf
stream4d_v4_1_probe5_tiered_egraph32flow_saware_w090_nms085
```

未完成：

```text
stream4d_v4_1_probe5_tiered_egraph32flow_saware_w090_nms095
```

说明：

```text
NMS 版本在 full-scene mask 上做 overlap competition，单个 nms085 ranking 运行超过 5 分钟。
nms085 最终完成并评估；nms095 未产生结果文件，不写指标。
```

### 51.4 method 结果表

以下是正常 method 指标，不读取 GT 生成 prediction。数值来自 evaluator 输出最后一行，未乘以 100。

| config | own AP | own AP50 | own AP25 | fixed AP | fixed AP50 | fixed AP25 | union in target % | #pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| egraph only minobs3 rel0 | 0.242530 | 0.471493 | 0.680090 | 0.240665 | 0.447128 | 0.671741 | 88.2922 | 50.00 |
| tiered no drop | 0.213187 | 0.456217 | 0.670968 | 0.249312 | 0.459458 | 0.688319 | 100.0000 | 436.00 |
| tiered secioc0.50 | 0.226777 | 0.456865 | 0.666623 | 0.243555 | 0.452108 | 0.671741 | 94.7238 | 245.20 |
| tiered secioc0.85 | 0.213305 | 0.456428 | 0.671241 | 0.249551 | 0.460336 | 0.677592 | 99.0055 | 353.40 |
| tiered secioc0.50 score0.05 | 0.226777 | 0.456865 | 0.666623 | 0.243555 | 0.452108 | 0.671741 | 94.7238 | 245.20 |
| tiered minioc0.50 | 0.232190 | 0.462157 | 0.666623 | 0.240665 | 0.447128 | 0.671741 | 91.7022 | 240.00 |
| support-aware scoreconf w0.90 | 0.180856 | 0.424810 | 0.683030 | 0.228716 | 0.431452 | 0.661445 | 99.0055 | 353.40 |
| support-aware scoreconf w0.75 | 0.180856 | 0.424810 | 0.683030 | 0.228716 | 0.431452 | 0.661445 | 99.0055 | 353.40 |
| support-aware supportconf | 0.167704 | 0.388634 | 0.616273 | 0.230424 | 0.426761 | 0.629365 | 99.0055 | 353.40 |
| support-aware w0.90 min_ioc nms0.85 | 0.180863 | 0.424810 | 0.653404 | 0.227654 | 0.426356 | 0.661445 | 98.4188 | 339.40 |

当前最好 method：

```text
stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
fixed32 = 0.249551 / 0.460336 / 0.677592
```

对比：

```text
Stream4D 32f self probe5:
0.144238 / 0.288344 / 0.464716

Stream3D on 32f support probe5:
0.399213 / 0.597171 / 0.742535
```

### 51.5 GT-read-only oracle 诊断

目的：判断候选池本身是否有足够好的候选。该步骤读取 GT，只做上界分析，不能作为方法结果。

命令模板：

```bash
run_oracle() {
  local pred="$1"; local out="$2"
  "$PY" -u -m tools.oracle_candidate_upper_bound \
    --root . \
    --seq-list splits/scannet_v4_1_probe5.txt \
    --pred-config "$pred" \
    --pre-points-config "$SUPPORT" \
    --output-config "$out" \
    --pred-suffix class_agnostic \
    --min-select-iou 0.25 \
    --summary-root outputs/oracle_candidate_upper_bound_probe5

  "$PY" -u -m evaluation.evaluate \
    --pred_path "data/prediction/${out}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$out" \
    --no_class
}
```

实际 oracle configs：

```text
stream4d_v4_1_probe5_oracle_scannet_on_32fsupport
stream4d_v4_1_probe5_oracle_stream4d_32f_self
stream4d_v4_1_probe5_oracle_egraph_minobs3
stream4d_v4_1_probe5_oracle_tiered_egraph_32flow
```

oracle 结果：

| pool | oracle AP | oracle AP50 | oracle AP25 | pred/scene | valid pred in support | oracle selected | GT best IoU>=0.25 | >=0.5 | >=0.75 | >=0.8 | >=0.9 | mean best IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stream3D candidate pool | 0.528782 | 0.722892 | 0.843373 | 128.20 | 18.60 | 14.00 | 14.60 | 12.00 | 8.40 | 7.00 | 5.00 | 0.6687 |
| Stream4D 32f self pool | 0.238286 | 0.445783 | 0.650602 | 386.00 | 25.20 | 10.80 | 11.00 | 7.40 | 3.00 | 1.80 | 0.60 | 0.4364 |
| Stream4D egraph minobs3 pool | 0.356091 | 0.614458 | 0.831325 | 50.00 | 22.60 | 13.80 | 14.20 | 10.20 | 5.00 | 3.40 | 0.20 | 0.5589 |
| Stream4D tiered pool | 0.393574 | 0.650602 | 0.843373 | 353.40 | 39.40 | 14.00 | 14.40 | 10.80 | 5.60 | 4.20 | 0.40 | 0.5895 |

### 51.6 本轮执行结论

本轮仍没有达成 Stream4D 在 inherit/fixed support 下超过 Stream3D。

真实 method 最好：

```text
0.249551 / 0.460336 / 0.677592
```

低于 Stream3D on same 32f support：

```text
0.399213 / 0.597171 / 0.742535
```

但 oracle 给出新证据：

```text
Stream4D tiered candidate pool oracle:
0.393574 / 0.650602 / 0.843373
```

解释：

```text
1. Stream4D tiered 候选池的 AP50/AP25 oracle 已经超过 Stream3D 实际。
2. AP oracle 仍略低于 Stream3D 实际 AP：0.393574 < 0.399213。
3. Stream3D 自己的 candidate-pool oracle 更高：0.528782 / 0.722892 / 0.843373。
4. 所以当前 Stream4D 不只是排序问题；高 IoU 候选质量仍弱于 Stream3D。
5. 但 egraph/tiered 明显改善了候选池，相比原始 Stream4D 32f oracle AP 0.238286，tiered oracle AP 达 0.393574。
```

安全结论：

```text
Pure Stream4D 还没有超过 Stream3D；但 evidence graph + component densify + tiered inherit 已经把候选池上界推到接近 Stream3D actual 的水平。
```

## 52. 2026-06-08 继续推进：probe5 multi-source consensus / greedy novelty 负例

### 52.1 本轮目的

第 51 节 oracle 诊断显示：

```text
Stream4D tiered candidate pool oracle:
0.393574 / 0.650602 / 0.843373

Stream4D tiered actual:
0.249551 / 0.460336 / 0.677592
```

因此本轮不新增代码，复用已有两个无 GT selection 工具，尝试把候选池上界转成真实 AP：

```text
tools.multi_source_consensus_select
tools.greedy_support_select
```

### 52.2 multi-source consensus 命令

输入：

```text
EGRAPH=stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs3_compdens_r003_d16_m8_rel000_hard2
BASE=stream4d_scannet_32f_ioc075_fixmem
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
```

代表命令：

```bash
"$PY" -u -m tools.multi_source_consensus_select \
  --root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --source-configs "$EGRAPH,$BASE" \
  --source-weights "1.2,0.8" \
  --score-pre-points-config "$SUPPORT" \
  --output-config stream4d_v4_1_probe5_consensus_egraph32f_reqms_minioc070_low32f \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 0.70 \
  --require-multi-source \
  --low-recall-configs "$BASE" \
  --low-score 0.05 \
  --low-drop-overlap-mode candidate_ioc \
  --low-drop-overlap-threshold 0.85 \
  --summary-root outputs/multi_source_consensus_probe5
```

实际 configs：

```text
stream4d_v4_1_probe5_consensus_egraph32f_reqms_minioc050_low32f
stream4d_v4_1_probe5_consensus_egraph32f_reqms_minioc070_low32f
stream4d_v4_1_probe5_consensus_egraph32f_all_minioc070
```

### 52.3 greedy support novelty 命令

输入：

```text
TIER=stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
```

代表命令：

```bash
"$PY" -u -m tools.greedy_support_select \
  --root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --input-config "$TIER" \
  --output-config stream4d_v4_1_probe5_greedy_tier_max120 \
  --score-pre-points-config "$SUPPORT" \
  --max-instances 120 \
  --min-new-area 10 \
  --score-weight 0.55 \
  --area-weight 0.10 \
  --unique-weight 0.20 \
  --conflict-weight 0.20 \
  --new-area-weight 0.30 \
  --novelty-weight 0.30 \
  --overlap-penalty 0.40 \
  --summary-root outputs/greedy_support_probe5
```

实际 configs：

```text
stream4d_v4_1_probe5_greedy_tier_max80
stream4d_v4_1_probe5_greedy_tier_max120
stream4d_v4_1_probe5_greedy_tier_suppressed
```

### 52.4 评估命令

每个 output config 都运行：

```bash
"$PY" -u -m evaluation.evaluate \
  --pred_path "data/prediction/${cfg}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file "data/evaluation/scannet/${cfg}_class_agnostic.txt" \
  --tmp_root data/TMP \
  --tmp_config "$cfg" \
  --no_class

"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --pred-config "$cfg" \
  --source-pre-points-config "$cfg" \
  --pre-points-config "$SUPPORT" \
  --output-config "${cfg}_on_32f_probe5" \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${cfg}_on_32f_probe5_class_agnostic.txt" \
  --audit-root outputs/audit
```

### 52.5 结果表

数值来自 evaluator 输出文件最后一行，未乘以 100。

| config | own AP | own AP50 | own AP25 | fixed AP | fixed AP50 | fixed AP25 | union in target % | #pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiered best | 0.213305 | 0.456428 | 0.671241 | 0.249551 | 0.460336 | 0.677592 | 99.0055 | 353.40 |
| consensus req multi-source minIoC0.50 + low32f | 0.129035 | 0.268746 | 0.540988 | 0.129035 | 0.268746 | 0.540988 | 99.0941 | 263.60 |
| consensus req multi-source minIoC0.70 + low32f | 0.136398 | 0.295163 | 0.604728 | 0.136398 | 0.295163 | 0.604728 | 98.7903 | 249.80 |
| consensus all minIoC0.70 | 0.123047 | 0.273277 | 0.593200 | 0.123047 | 0.273277 | 0.593200 | 81.9934 | 158.20 |
| greedy max80 | 0.135797 | 0.328517 | 0.507533 | 0.181316 | 0.343578 | 0.530206 | 72.5842 | 25.80 |
| greedy max120 | 0.135797 | 0.328517 | 0.507533 | 0.181316 | 0.343578 | 0.530206 | 72.5842 | 25.80 |
| greedy suppressed | 0.138168 | 0.333874 | 0.516592 | 0.180838 | 0.344009 | 0.533193 | 73.8750 | 27.20 |

### 52.6 本轮执行结论

本轮没有达成目标，也没有超过第 51 节 tiered best。

最关键对比：

```text
tiered best fixed:
0.249551 / 0.460336 / 0.677592

best consensus fixed:
0.136398 / 0.295163 / 0.604728

best greedy fixed:
0.181316 / 0.343578 / 0.530206
```

失败原因：

```text
1. multi-source consensus 要求 egraph 和 32f current 在 support 内重叠，会保留更“互相同意”的候选，但这些候选不一定是高 IoU 候选。
2. greedy support novelty 减少了候选数量和重复，但也丢掉了大量 recall；union in target 从 99.0055% 降到约 72-74%。
3. oracle 潜力没有被兑现，说明需要更强的 object-level quality / boundary signal，而不是只用源一致性或新覆盖面积。
```

安全结论：

```text
multi-source consensus 和 greedy support novelty 都不能把 tiered candidate-pool oracle 上界转成真实 AP。
```

## 53. 2026-06-08 继续推进：probe5 tiered best 的 support component refinement

### 53.1 本轮目的

第 52 节证明：

```text
multi-source consensus 和 greedy support novelty 都没有把 tiered candidate pool 变成更好的真实 AP。
```

因此本轮不再继续做 candidate 排序，而是回到边界/实例质量方向，测试一个更贴近 mask 几何的假设：

```text
当前 probe5 tiered best 的 fixed-support AP 不够高，可能有一部分原因是 prediction object 在 32f support 内包含离散碎片。
如果只在 target support 内按 3D 半径连通组件裁剪，可能提高高 IoU AP。
```

该实验不读取 GT 生成 prediction，不改变 evaluator，只使用：

```text
prediction mask
stream4d_scannet_32f_ioc075_fixmem pre_points
ScanNet mesh vertex coordinates
```

### 53.2 输入配置

```text
SPLIT=splits/scannet_v4_1_probe5.txt
INPUT=stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
```

对照：

```text
tiered best fixed:
0.249551 / 0.460336 / 0.677592

Stream3D on same 32f support:
0.399213 / 0.597171 / 0.742535
```

### 53.3 运行命令

第一组：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

SPLIT=splits/scannet_v4_1_probe5.txt
INPUT=stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
SUPPORT=stream4d_scannet_32f_ioc075_fixmem

for spec in r006_k2 r010_k4 r012_k6; do
  case "$spec" in
    r006_k2) rad=0.06; k=2; ratio=0.03;;
    r010_k4) rad=0.10; k=4; ratio=0.03;;
    r012_k6) rad=0.12; k=6; ratio=0.02;;
  esac
  out=stream4d_v4_1_probe5_tiered_compref_${spec}
  "$PY" -u -m tools.support_component_refine \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --support-config "$SUPPORT" \
    --radius "$rad" \
    --max-components-per-instance "$k" \
    --min-component-points 20 \
    --min-component-ratio "$ratio" \
    --min-support-area 20 \
    --outside-support drop \
    --tmp-policy support \
    --summary-root outputs/support_component_refine_probe5

  "$PY" -u -m evaluation.evaluate \
    --pred_path "data/prediction/${out}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${out}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$out" \
    --no_class

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --source-pre-points-config "$out" \
    --pre-points-config "$SUPPORT" \
    --output-config "${out}_on_32f_probe5" \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file "data/evaluation/scannet/${out}_on_32f_probe5_class_agnostic.txt" \
    --audit-root outputs/audit
done
```

第二组：

```bash
for spec in r014_k8 r016_k10; do
  case "$spec" in
    r014_k8) rad=0.14; k=8; ratio=0.01;;
    r016_k10) rad=0.16; k=10; ratio=0.01;;
  esac
  out=stream4d_v4_1_probe5_tiered_compref_${spec}
  "$PY" -u -m tools.support_component_refine \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --support-config "$SUPPORT" \
    --radius "$rad" \
    --max-components-per-instance "$k" \
    --min-component-points 20 \
    --min-component-ratio "$ratio" \
    --min-support-area 20 \
    --outside-support drop \
    --tmp-policy support \
    --summary-root outputs/support_component_refine_probe5
  ...
done
```

第三组：

```bash
for spec in r018_k12 r020_k14; do
  case "$spec" in
    r018_k12) rad=0.18; k=12; ratio=0.01;;
    r020_k14) rad=0.20; k=14; ratio=0.01;;
  esac
  out=stream4d_v4_1_probe5_tiered_compref_${spec}
  "$PY" -u -m tools.support_component_refine \
    --seq-list "$SPLIT" \
    --input-config "$INPUT" \
    --output-config "$out" \
    --support-config "$SUPPORT" \
    --radius "$rad" \
    --max-components-per-instance "$k" \
    --min-component-points 20 \
    --min-component-ratio "$ratio" \
    --min-support-area 20 \
    --outside-support drop \
    --tmp-policy support \
    --summary-root outputs/support_component_refine_probe5
  ...
done
```

其中 `...` 是和第一组完全相同的 `evaluation.evaluate` 与 `tools.evaluate_cross_prepoints` 命令。

### 53.4 结果

所有数值来自 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 最后一行，未乘以 100。

| config | fixed AP | fixed AP50 | fixed AP25 | union in target | changed instances | keep ratio | support union after |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiered best | 0.249551 | 0.460336 | 0.677592 | 0.990055 | NA | 1.000000 | NA |
| r006 k2 | 0.205341 | 0.374472 | 0.562057 | 0.763210 | 81.2 | 0.635867 | 7474.0 |
| r010 k4 | 0.254376 | 0.463412 | 0.649452 | 0.903696 | 71.2 | 0.682124 | 8720.2 |
| r012 k6 | 0.257755 | 0.473841 | 0.666502 | 0.931218 | 68.0 | 0.694878 | 8943.4 |
| r014 k8 | 0.257852 | 0.472342 | 0.664193 | 0.944344 | 65.0 | 0.703004 | 9062.2 |
| r016 k10 | 0.260055 | 0.472260 | 0.664159 | 0.953919 | 63.0 | 0.709232 | 9136.6 |
| r018 k12 | 0.260386 | 0.472260 | 0.677637 | 0.962415 | 60.2 | 0.714625 | 9203.4 |
| r020 k14 | 0.260367 | 0.472233 | 0.677625 | 0.968224 | 57.8 | 0.717995 | 9254.0 |

当前本轮最好：

```text
stream4d_v4_1_probe5_tiered_compref_r018_k12_on_32f_probe5
AP/AP50/AP25 = 0.26038635220275147 / 0.47225957401032703 / 0.6776367126537237
```

### 53.5 执行中小问题

汇总 summary 时我第一次用了系统 `python`，当前 shell 中没有 `python` 命令：

```text
/bin/bash: line 1: python: command not found
```

修复方式：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
"$PY" - <<'PY'
...
PY
```

该问题只影响 summary 汇总命令，不影响前面的 prediction 生成和 evaluator 结果。

### 53.6 本轮执行结论

本轮有小幅改进，但没有达成超过 Stream3D 的目标。

对比：

```text
previous probe5 pure Stream4D tiered best:
0.249551 / 0.460336 / 0.677592

best support component refinement:
0.260386 / 0.472260 / 0.677637

Stream3D on same 32f support:
0.399213 / 0.597171 / 0.742535
```

解释：

```text
1. 轻量 3D component trimming 可以移除一部分坏碎片，使 AP 提升 +0.010835，AP50 提升 +0.011924。
2. 太强的 trimming 会明显掉分，例如 r006_k2 只有 0.205341 AP。
3. 最佳 r018_k12 的 union in target 仍有 96.2415%，因此提升不是靠大幅缩小 support。
4. 但即便这样，AP 仍低 Stream3D 约 0.138827，AP50 低约 0.124912。
```

安全结论：

```text
probe5 上，support component refinement 是一个小的正向边界修复，但不足以解决 inherit/fixed support 下 pure Stream4D 对 Stream3D 的主差距。
```

### 53.7 补充：compref high + 原 tiered low-confidence recall

为了确认 `r018_k12` 是否只是 AP25 recall 被裁掉，本轮又尝试：

```text
primary high = stream4d_v4_1_probe5_tiered_compref_r018_k12
secondary low = stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085
target support = stream4d_scannet_32f_ioc075_fixmem
```

命令：

```bash
for th in 0.85 0.95; do
  out=stream4d_v4_1_probe5_compref_r018_high_tiered_low_secioc${tag}
  "$PY" -u -m tools.fuse_prediction_configs \
    --seq-list "$SPLIT" \
    --primary-config stream4d_v4_1_probe5_tiered_compref_r018_k12 \
    --secondary-config stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085 \
    --output-config "$out" \
    --fusion-mode concatenate \
    --preserve-primary-score \
    --secondary-score 0.005 \
    --drop-secondary-iou-threshold "$th" \
    --drop-secondary-overlap-mode secondary_ioc \
    --drop-overlap-pre-points-config "$SUPPORT" \
    --summary-root outputs/fuse_prediction_configs_probe5

  "$PY" -u -m tools.evaluate_cross_prepoints \
    --seq-list "$SPLIT" \
    --pred-config "$out" \
    --source-pre-points-config "$out" \
    --pre-points-config "$SUPPORT" \
    --output-config "${out}_on_32f_probe5" \
    --gt-root data/scannet/gt \
    --dataset scannet \
    --no-class \
    --output-file "data/evaluation/scannet/${out}_on_32f_probe5_class_agnostic.txt" \
    --audit-root outputs/audit
done
```

结果：

| config | fixed AP | fixed AP50 | fixed AP25 | output instances | secondary after drop | output union |
|---|---:|---:|---:|---:|---:|---:|
| compref r018 high only | 0.260386 | 0.472260 | 0.677637 | 353.4 | NA | 9203.4 in support |
| compref high + tiered low secioc0.85 | 0.260386 | 0.472260 | 0.677637 | 471.8 | 118.4 | 13635.2 full scene |
| compref high + tiered low secioc0.95 | 0.260386 | 0.472260 | 0.677637 | 489.4 | 136.0 | 19140.4 full scene |

解释：

```text
1. secondary 低分层确实被加入了，平均输出实例数从 353.4 增到 471.8 / 489.4。
2. 但 AP/AP50/AP25 完全不变，说明这些低分补漏候选没有提供新的有效 PR 曲线收益。
3. 因此当前 r018_k12 的 AP25 不能靠把原 tiered 低分加回来恢复；需要更好的候选本身，而不是更多低分重复候选。
```

## 54. 2026-06-08 继续推进：probe5 egraph silhouette evidence 负例

### 54.1 本轮目的

scene0050 单场景上，`silhouette_consistency_score.py` 曾对 localbank low-confidence recall layer 有很小正收益。本轮检查该信号能否迁移到 probe5 的 egraph high layer。

可用 object_dict 检查：

```text
stream4d_scannet_32f_ioc075_fixmem:
  scene0050_00 / scene0011_00 / scene0030_00 / scene0081_01 / scene0591_00 all yes

stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs3_compdens_r003_d16_m8_rel000_hard2:
  scene0050_00 / scene0011_00 / scene0030_00 / scene0081_01 / scene0591_00 all yes

stream4d_v4_1_probe5_tiered_egraph_minobs3_high_32flow_secioc085:
  no object_dict
```

因此本轮只能把 silhouette evidence 作用在 egraph high layer，再和 32f low layer 融合。

### 54.2 silhouette score / filter 命令

输入：

```text
SPLIT=splits/scannet_v4_1_probe5.txt
EGRAPH=stream4d_v4_1_egraph_32f_probe5_ioc0p70_minobs3_compdens_r003_d16_m8_rel000_hard2
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
```

命令模板：

```bash
"$PY" -u -m tools.silhouette_consistency_score \
  --seq-list "$SPLIT" \
  --input-config "$EGRAPH" \
  --output-config "$out" \
  --quality-mode score_silhouette \
  --score-weight "$sw" \
  --max-observations 8 \
  --max-points-per-object 2000 \
  --depth-tolerance 0.08 \
  --boundary-margin-px 2.0 \
  --min-silhouette-quality "$q" \
  --summary-root outputs/silhouette_consistency_score_probe5

"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list "$SPLIT" \
  --pred-config "$out" \
  --source-pre-points-config "$out" \
  --pre-points-config "$SUPPORT" \
  --output-config "${out}_on_32f_probe5" \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${out}_on_32f_probe5_class_agnostic.txt" \
  --audit-root outputs/audit
```

实际配置：

```text
stream4d_v4_1_probe5_egraph_minobs3_sil_w075
stream4d_v4_1_probe5_egraph_minobs3_sil_w025
stream4d_v4_1_probe5_egraph_minobs3_sil_q050_w075
stream4d_v4_1_probe5_egraph_minobs3_sil_q070_w075
stream4d_v4_1_probe5_egraph_minobs3_sil_q050_filteronly
stream4d_v4_1_probe5_egraph_minobs3_sil_q070_filteronly
```

`filteronly` 的含义：

```text
--score-weight 1.0
```

也就是不让 silhouette 参与排序，只用 silhouette threshold 删除候选。

### 54.3 egraph silhouette 结果

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

结果：

| config | AP | AP50 | AP25 | instances after | removed / scene | silhouette mean |
|---|---:|---:|---:|---:|---:|---:|
| egraph silhouette score w0.75 | 0.137832 | 0.315042 | 0.580297 | 50.0 | 0.0 | 0.658718 |
| egraph silhouette score w0.25 | 0.137832 | 0.315042 | 0.580297 | 50.0 | 0.0 | 0.658718 |
| egraph silhouette q0.50 + score w0.75 | 0.132773 | 0.303399 | 0.527297 | 32.2 | 17.8 | 0.658718 |
| egraph silhouette q0.70 + score w0.75 | 0.124826 | 0.284065 | 0.495152 | 21.8 | 28.2 | 0.658718 |
| egraph silhouette q0.50 filter-only | 0.236186 | 0.439819 | 0.605060 | 32.2 | 17.8 | 0.658718 |
| egraph silhouette q0.70 filter-only | 0.231320 | 0.428858 | 0.590648 | 21.8 | 28.2 | 0.658718 |

### 54.4 silhouette high + 32f low 融合

先跑了两条带 `--preserve-primary-score` 的 filter-only 融合：

```text
stream4d_v4_1_probe5_tiered_q050_filteronly_high_32flow_secioc085
stream4d_v4_1_probe5_tiered_q070_filteronly_high_32flow_secioc085
```

执行后发现 filter-only 输出的 primary score 全为 0，而 secondary score 是 0.005，因此 secondary 反而排在 primary 前面。这两条不作为正式结论，只保留作审计记录：

```text
q0.50 preserve-primary wrong-order: 0.107633 / 0.233443 / 0.406970
q0.70 preserve-primary wrong-order: 0.106454 / 0.236874 / 0.419983
```

修正命令：

```bash
"$PY" -u -m tools.fuse_prediction_configs \
  --seq-list "$SPLIT" \
  --primary-config "$primary" \
  --secondary-config "$SUPPORT" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --primary-score 1.0 \
  --secondary-score 0.005 \
  --drop-secondary-iou-threshold 0.85 \
  --drop-secondary-overlap-mode secondary_ioc \
  --drop-overlap-pre-points-config "$SUPPORT" \
  --summary-root outputs/fuse_prediction_configs_probe5
```

正式融合结果：

| config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| silhouette score w0.75 high + 32f low | 0.146718 | 0.328250 | 0.586149 |
| silhouette q0.50 score w0.75 high + 32f low | 0.142172 | 0.317362 | 0.550833 |
| silhouette q0.50 filter-only high1 + 32f low | 0.245585 | 0.453782 | 0.628597 |
| silhouette q0.70 filter-only high1 + 32f low | 0.244839 | 0.454232 | 0.642204 |

### 54.5 本轮执行结论

本轮没有达成目标，也没有超过已有 best。

关键对比：

```text
best silhouette-related result:
0.245585 / 0.453782 / 0.628597

current pure Stream4D probe5 best:
0.260386 / 0.472260 / 0.677637

Stream3D same 32f support:
0.399213 / 0.597171 / 0.742535
```

结论：

```text
1. silhouette score 不能直接作为 egraph high layer 的排序信号。
2. silhouette threshold 过滤会删掉大量必要 recall。
3. 即使用 filter-only 并修正 high/low score，融合结果仍低于原 tiered / compref best。
```

安全结论：

```text
probe5 上，多视角 silhouette consistency 不是当前 inherit/fixed support 差距的有效修复方向。
```

## 55. 2026-06-08 继续推进：probe5 final prediction point-level WTA 负例

### 55.1 本轮目的

第 53 节当前 probe5 纯 Stream4D 最好结果是：

```text
stream4d_v4_1_probe5_tiered_compref_r018_k12_on_32f_probe5
AP/AP50/AP25 = 0.26038635220275147 / 0.47225957401032703 / 0.6776367126537237
```

它仍低于同一 32f support 下的 Stream3D：

```text
scannet_on_stream4d_32f_probe5
AP/AP50/AP25 = 0.3992127932017927 / 0.5971712938711367 / 0.7425353588266108
```

本轮尝试第 25 节在 scene0050 上有效过的最终 prediction point-level WTA，检查它在 probe5 多场景上是否还能提升 fixed-support AP。

### 55.2 运行命令

工作目录：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
```

环境：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
```

固定输入：

```bash
SPLIT=splits/scannet_v4_1_probe5.txt
INPUT=stream4d_v4_1_probe5_tiered_compref_r018_k12
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
```

WTA 生成命令模板：

```bash
"$PY" -u -m tools.wta_prediction_points \
  --root . \
  --seq-list "$SPLIT" \
  --input-config "$INPUT" \
  --output-config "$out" \
  --priority-mode "$mode" \
  --min-conflict-owners "$owners" \
  --min-priority-margin "$margin" \
  --drop-empty \
  --summary-root outputs/stream4d_point_wta_probe5
```

fixed-support 评估命令模板：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list "$SPLIT" \
  --pred-config "$out" \
  --source-pre-points-config "$out" \
  --pre-points-config "$SUPPORT" \
  --output-config "${out}_on_32f_probe5" \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${out}_on_32f_probe5_class_agnostic.txt" \
  --audit-root outputs/audit
```

实际扫描配置：

| output config | priority mode | min conflict owners | min priority margin |
|---|---|---:|---:|
| `stream4d_v4_1_probe5_compref_wta_score_m2` | `score` | 2 | 0 |
| `stream4d_v4_1_probe5_compref_wta_areadesc_m2` | `score_area_desc` | 2 | 0 |
| `stream4d_v4_1_probe5_compref_wta_areaasc_m2` | `score_area_asc` | 2 | 0 |
| `stream4d_v4_1_probe5_compref_wta_score_m2_margin030` | `score` | 2 | 0.30 |
| `stream4d_v4_1_probe5_compref_wta_score_m3` | `score` | 3 | 0 |
| `stream4d_v4_1_probe5_compref_wta_areadesc_m3` | `score_area_desc` | 3 | 0 |

### 55.3 结果

所有数值来自 `Stream3D/data/evaluation/scannet/*.txt` 或 `outputs/audit/cross_prepoints/*.json`，没有手工改写。

| row | AP | AP50 | AP25 | union in target | #pred |
|---|---:|---:|---:|---:|---:|
| Stream3D same 32f support | 0.399213 | 0.597171 | 0.742535 | 0.985608 | 128.2 |
| compref best before WTA | 0.260386 | 0.472260 | 0.677637 | 0.962415 | 353.4 |
| WTA score m2 | 0.252840 | 0.469799 | 0.679025 | 0.962415 | 188.4 |
| WTA score-area-desc m2 | 0.252840 | 0.469799 | 0.679025 | 0.962415 | 186.6 |
| WTA score-area-asc m2 | 0.251627 | 0.464172 | 0.678995 | 0.962415 | 203.8 |
| WTA score m2 margin030 | 0.252833 | 0.469753 | 0.678995 | 0.962415 | 212.8 |
| WTA score m3 | 0.258648 | 0.473374 | 0.678121 | 0.962415 | 232.2 |
| WTA score-area-desc m3 | 0.258648 | 0.473374 | 0.678121 | 0.962415 | 231.4 |

WTA 诊断：

| config | inst before | inst after | conflict before | conflict after | removed assignments | union before/after | assignments before/after |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTA score m2 | 353.4 | 188.4 | 5627.4 | 0.0 | 7870.4 | 9203.4 / 9203.4 | 17073.8 / 9203.4 |
| WTA score-area-desc m2 | 353.4 | 186.6 | 5627.4 | 0.0 | 7870.4 | 9203.4 / 9203.4 | 17073.8 / 9203.4 |
| WTA score-area-asc m2 | 353.4 | 203.8 | 5627.4 | 0.0 | 7870.4 | 9203.4 / 9203.4 | 17073.8 / 9203.4 |
| WTA score m2 margin030 | 353.4 | 212.8 | 5627.4 | 122.0 | 7701.0 | 9203.4 / 9203.4 | 17073.8 / 9372.8 |
| WTA score m3 | 353.4 | 232.2 | 1138.6 | 0.0 | 3381.6 | 9203.4 / 9203.4 | 17073.8 / 13692.2 |
| WTA score-area-desc m3 | 353.4 | 231.4 | 1138.6 | 0.0 | 3381.6 | 9203.4 / 9203.4 | 17073.8 / 13692.2 |

### 55.4 本轮执行结论

本轮没有达成目标。

最好的 WTA 配置是：

```text
stream4d_v4_1_probe5_compref_wta_score_m3_on_32f_probe5
AP/AP50/AP25 = 0.2586475908721231 / 0.4733736029107388 / 0.6781207106908648
```

它和 compref best 的关系：

```text
AP:   0.2586475908721231 < 0.26038635220275147
AP50: 0.4733736029107388 > 0.47225957401032703
AP25: 0.6781207106908648 > 0.6776367126537237
```

解释：

```text
WTA m3 只让 AP50/AP25 有极小提升，但主 AP 下降。
hard WTA m2 把冲突清到 0，但 AP/AP50 下降。
所有 WTA 都保持 union before/after = 9203.4 / 9203.4，说明它没有补 coverage，只改变点归属。
```

因此，probe5 上 point-level WTA 不是当前 inherit/fixed support 差距的有效修复方向。

## 56. 2026-06-08 继续推进：probe5 self-discovered boundary refinement

### 56.1 本轮目的

第 55 节说明 final prediction point-level WTA 不能解决 fixed-support 差距。按照计划里的 boundary-aware 修复方向，本轮复用已有：

```text
Stream3D/tools/self_discovered_boundary_refine.py
```

做一个更靠近实例边界质量的尝试：

```text
1. 对每个 prediction instance，在多个 ScanNet RGB-D 帧中投影它的 3D 点。
2. 不读取 GT，而是在投影点落到的 2D predicted mask 中自发现 dominant mask id。
3. 保留多视角中更稳定落在 dominant mask 内的点。
4. 可选用 mask boundary margin 过滤边界附近不稳定点。
5. evaluator 仍使用同一个 32f fixed support。
```

### 56.2 运行命令

工作目录：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
```

环境：

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
export CONDA_INSTRUMENTATION_ENABLED=0
export PYTHONNOUSERSITE=1
export OPEN3D_DISABLE_WEB_VISUALIZER=true
```

固定输入：

```bash
SPLIT=splits/scannet_v4_1_probe5.txt
INPUT=stream4d_v4_1_probe5_tiered_compref_r018_k12
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
```

生成命令模板：

```bash
"$PY" -u -m tools.self_discovered_boundary_refine \
  --root . \
  --seq-list "$SPLIT" \
  --input-config "$INPUT" \
  --output-config "$OUT" \
  --refine-support-config "$SUPPORT" \
  --outside-refine-support keep \
  --frame-stride "$stride" \
  --max-frames "$maxframes" \
  --max-observations "$maxobs" \
  --discovery-max-points 1500 \
  --depth-tolerance 0.08 \
  --boundary-margin-px "$margin" \
  --min-visible-points 8 \
  --min-dominant-points 5 \
  --min-dominant-ratio 0.35 \
  --min-point-visible-views 1 \
  --min-point-inside-ratio "$inside" \
  --min-point-interior-ratio "$interior" \
  --unobserved-policy "$unobs" \
  --min-points-before-refine 20 \
  --min-points-after-refine 10 \
  --drop-empty \
  --tmp-policy input \
  --summary-root outputs/self_discovered_boundary_refine_probe5
```

评估命令模板：

```bash
"$PY" -u -m tools.evaluate_cross_prepoints \
  --seq-list "$SPLIT" \
  --pred-config "$OUT" \
  --source-pre-points-config "$OUT" \
  --pre-points-config "$SUPPORT" \
  --output-config "${OUT}_on_32f_probe5" \
  --gt-root data/scannet/gt \
  --dataset scannet \
  --no-class \
  --output-file "data/evaluation/scannet/${OUT}_on_32f_probe5_class_agnostic.txt" \
  --audit-root outputs/audit
```

### 56.3 结果

| row | AP | AP50 | AP25 | union in target | #pred | keep ratio | changed | used obs |
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

### 56.4 本轮执行结论

本轮没有达成超过 Stream3D 的最终目标，但找到了 probe5 当前最强纯 Stream4D fixed-support 结果：

```text
stream4d_v4_1_probe5_compref_selfboundary_b1_in050_keep_s40_on_32f_probe5
AP/AP50/AP25 = 0.27733794006771613 / 0.4913198877364921 / 0.6787473172993864
```

相对上一轮 compref best：

```text
AP   +0.01695158786496466
AP50 +0.01906031372616507
AP25 +0.00111060464566268
```

但相对 Stream3D same 32f support：

```text
AP   仍低 0.12187485313407657
AP50 仍低 0.10585140613464458
AP25 仍低 0.06378804152722443
```

执行观察：

```text
1. inside ratio 0.45/0.50 比 0.55/0.60 更均衡。
2. 更激进的 inside/interior filtering 可以提高 AP50，但会伤 AP 和 AP25。
3. unobserved-policy=drop 明显失败，说明很多 support 点没有足够多帧可见证据，直接删除会伤 recall。
4. frame_stride=40 / max_frames=160 / max_observations=12 比 stride=80 更好，说明更多视角能改善 boundary refinement 的稳定性。
```

## 57. 2026-06-08 继续推进：boundary high + low-confidence recall layer

### 57.1 本轮目的

第 56 节当前 pure Stream4D probe5 best 是：

```text
stream4d_v4_1_probe5_compref_selfboundary_b1_in050_keep_s40_on_32f_probe5
AP/AP50/AP25 = 0.27733794006771613 / 0.4913198877364921 / 0.6787473172993864
union in target = 0.916531
```

它的 boundary refinement 提高了 AP/AP50，但把 union in target 从 compref previous best 的 0.962415 降到 0.916531。  
本轮测试：

```text
boundary-refined high-confidence layer
+ low-confidence recall layer
```

目标是补回 coverage/recall，同时尽量不破坏 high layer 的 AP。

### 57.2 运行命令

固定输入：

```bash
SPLIT=splits/scannet_v4_1_probe5.txt
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
HIGH=stream4d_v4_1_probe5_compref_selfboundary_b1_in050_keep_s40
COMP=stream4d_v4_1_probe5_tiered_compref_r018_k12
BASE=stream4d_scannet_32f_ioc075_fixmem
```

残差补漏命令模板：

```bash
"$PY" -u -m tools.residual_recall_fuse \
  --root . \
  --seq-list "$SPLIT" \
  --primary-config "$HIGH" \
  --secondary-config "$secondary" \
  --output-config "$out" \
  --support-config "$SUPPORT" \
  --primary-score -1.0 \
  --secondary-score 0.005 \
  --min-residual-area 10 \
  --min-residual-ratio 0.01 \
  --secondary-mode "$mode" \
  --summary-root outputs/residual_recall_fuse_probe5
```

低分拼接命令模板：

```bash
"$PY" -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list "$SPLIT" \
  --primary-config "$HIGH" \
  --secondary-config "$secondary" \
  --output-config "$out" \
  --fusion-mode concatenate \
  --preserve-primary-score \
  --secondary-score 0.005 \
  --drop-secondary-iou-threshold "$thresh" \
  --drop-secondary-overlap-mode secondary_ioc \
  --drop-overlap-pre-points-config "$SUPPORT" \
  --summary-root outputs/fuse_prediction_configs_probe5
```

评估仍使用 `tools.evaluate_cross_prepoints`，target support 固定为：

```text
stream4d_scannet_32f_ioc075_fixmem
```

### 57.3 结果

| row | AP | AP50 | AP25 | union in target | #pred | extra diagnostic |
|---|---:|---:|---:|---:|---:|---|
| Stream3D same 32f support | 0.399213 | 0.597171 | 0.742535 | 0.985608 | 128.2 | - |
| selfboundary high | 0.277338 | 0.491320 | 0.678747 | 0.916531 | 254.8 | - |
| high + comp residual | 0.279431 | 0.495199 | 0.683492 | 0.955224 | 272.4 | residual_instances=17.6; uncovered=788.2; out_conflict=0.4727 |
| high + comp supportfull | 0.281610 | 0.497938 | 0.690897 | 0.955224 | 272.4 | residual_instances=17.6; uncovered=788.2; out_conflict=0.7053 |
| high + 32f residual | 0.277338 | 0.491320 | 0.678747 | 0.982493 | 279.0 | residual_instances=24.2; uncovered=788.2; out_conflict=0.4517 |
| high + comp cat085 | 0.280989 | 0.498159 | 0.691317 | 0.950209 | 391.6 | out_instances=391.6; secondary_after=136.8; skipped=216.6 |
| high + comp cat095 | 0.281615 | 0.497583 | 0.690254 | 0.957699 | 415.6 | out_instances=415.6; secondary_after=160.8; skipped=192.6 |
| high + 32f cat085 | 0.277686 | 0.491320 | 0.682865 | 0.978747 | 427.6 | out_instances=427.6; secondary_after=172.8; skipped=213.2 |

### 57.4 本轮执行结论

本轮没有达成超过 Stream3D，但再次刷新 pure Stream4D probe5 fixed-support best：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
AP/AP50/AP25 = 0.281615 / 0.497583 / 0.690254
```

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

相对 Stream3D same support：

```text
AP   仍低 0.117598
AP50 仍低 0.099588
AP25 仍低 0.052281
```

关键观察：

```text
1. 只补 residual support 已有正收益，但 `support_full` 比纯 residual 更好，说明二级候选的完整 support 仍对 AP 有帮助。
2. secondary 使用 compref previous 明显好于使用原始 32f support prediction。
3. high + 32f residual 虽然 union in target 到 0.982493，但指标几乎不变，说明简单补 coverage 不等于补正确实例。
4. 当前最有效结构是 boundary-refined high layer + compref recall layer，而不是 WTA / silhouette / 单纯 32f recall。
```

## 58. 2026-06-08 object-level competition on current best 负例

### 58.1 本轮目的

第 57 节当前 pure Stream4D probe5 fixed-support best：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
AP/AP50/AP25 = 0.281615 / 0.497583 / 0.690254
#pred = 415.6
```

它与 Stream3D 的主要差距之一是 prediction 数量过多：

```text
Stream3D same 32f support #pred = 128.2
```

本轮复用已有：

```text
Stream3D/tools/object_competition_rank.py
```

尝试在当前 best 候选池上做 object-level overlap grouping 和无监督代表候选选择。

### 58.2 运行命令

固定输入：

```bash
SPLIT=splits/scannet_v4_1_probe5.txt
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095
SUPPORT=stream4d_scannet_32f_ioc075_fixmem
```

命令模板：

```bash
"$PY" -u -m tools.object_competition_rank \
  --root . \
  --seq-list "$SPLIT" \
  --input-config "$INPUT" \
  --output-config "$out" \
  --score-pre-points-config "$SUPPORT" \
  --quality-mode "$qmode" \
  --group-overlap-mode "$omode" \
  --group-overlap-threshold "$thresh" \
  --min-support-area 1 \
  --tmp-policy inherit \
  --summary-root outputs/stream4d_object_competition_probe5
```

评估命令仍使用 `tools.evaluate_cross_prepoints`，target support 固定为 `stream4d_scannet_32f_ioc075_fixmem`。

### 58.3 结果

| row | AP | AP50 | AP25 | union in target | #pred | selected/groups | group size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stream3D same 32f support | 0.399213 | 0.597171 | 0.742535 | 0.985608 | 128.2 | NA | NA |
| current best before objcomp | 0.281615 | 0.497583 | 0.690254 | 0.957699 | 415.6 | NA | NA |
| objcomp min050 score | 0.129698 | 0.217840 | 0.279147 | 0.476360 | 89.8 | 89.8 / 89.8 | 3.36 max 122.2 |
| objcomp min070 score | 0.158578 | 0.300187 | 0.440296 | 0.646035 | 156.4 | 156.4 / 156.4 | 1.92 max 50.6 |
| objcomp min085 score | 0.176469 | 0.342018 | 0.504653 | 0.739961 | 206.0 | 206.0 / 206.0 | 1.47 max 30.4 |
| objcomp min070 area | 0.161189 | 0.311110 | 0.516286 | 0.809664 | 156.4 | 156.4 / 156.4 | 1.92 max 50.6 |
| objcomp iou050 score | 0.190065 | 0.369790 | 0.556006 | 0.893759 | 233.4 | 233.4 / 233.4 | 1.32 max 7.0 |
| objcomp min070 score preserve | 0.216250 | 0.369478 | 0.509438 | 0.646035 | 156.4 | 156.4 / 156.4 | 1.92 max 50.6 |

### 58.4 本轮执行结论

本轮没有达成目标，并且给出强负证据：

```text
当前无监督 object-level competition 不能把 415.6 个候选压缩成 Stream3D 风格的高质量 128 个候选。
```

代表结果：

```text
objcomp iou050 score:
AP/AP50/AP25 = 0.190065 / 0.369790 / 0.556006
#pred = 233.4
```

虽然它把预测数从 415.6 减到 233.4，但 AP 大幅低于 current best：

```text
current best = 0.281615 / 0.497583 / 0.690254
```

解释：

```text
1. overlap grouping 能删除候选，但无监督 quality 不能可靠选出真实一对一实例。
2. min_ioc grouping 容易把相邻/互补候选放进同组，导致错删 recall。
3. iou grouping 更保守，union in target 仍有 0.893759，但 AP 仍只有 0.190065，说明保留的代表候选边界质量不足。
4. preserve original score 也不能恢复，说明问题不只是重排分数，而是候选分组/代表选择本身不可靠。
```

### 58.5 下一版计划必要性

到本节为止，以下后处理方向均已有负证据：

```text
final prediction WTA
silhouette direct ranking/filtering
support-area ranking
greedy novelty
object-level competition rank
simple low-confidence recall
```

虽然 boundary high + compref recall 有小幅正收益，但离 Stream3D 仍差：

```text
AP   约 0.118
AP50 约 0.100
AP25 约 0.052
```

因此下一版不应该继续堆后处理。需要新计划，把改动前移到：

```text
1. proposal generation
2. multi-view boundary evidence
3. object identity / split-merge memory
4. candidate quality calibration
```

### 58.6 下一版计划文档

已新增下一版计划：

```text
docs/stream4d_v5_inherit_gap_plan_for_codex.md
```

该计划基于本轮已确认的差距：

```text
pure Stream4D probe5 best:
0.281615 / 0.497583 / 0.690254

Stream3D same 32f support:
0.399213 / 0.597171 / 0.742535
```

v5 计划明确把主线从后处理转向：

```text
MaskObservation Bank
Boundary-Aware Proposal v2
Evidence Graph v2
Split / Merge Capable Object Memory
Calibrated Object Quality
```

并要求先过 probe5 fixed-support gate，再考虑 final/full ScanNet。
