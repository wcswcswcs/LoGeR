# Stream4D v6 执行日志

日期：2026-06-08（Asia/Singapore）  
计划文件：`docs/stream4d_v6_method_first_audit_and_experiment_plan_for_codex.md`  
结果复盘：`docs/stream4d_v6_实验结果复盘.md`  
代码审计包：`stream4d_v6_code_review_packet_20260608_2136.zip`

本日志只记录真实执行过的命令、文件和结果路径。没有运行的 phase 不写成完成。

## 0. 执行边界

v6 计划的核心是 method-first：先修 P0 审计和 metric safety，再验证 score-mode blocker 与 D4RT geometry blocker。用户追问后，本轮继续补做了一个 minimal Typed Evidence Graph v3 candidate；它不是完整 core/fringe/reject 或 split/merge memory，但已经作为真实 method candidate 跑了 scene0050 与 probe5，并记录失败证据。

工作目录：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
```

Python：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
```

Probe5 split：

```text
Stream3D/splits/scannet_v6_probe5.txt
scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00
```

## 1. 代码修改记录

### 1.1 evaluator manifest guard

文件：

```text
Stream3D/evaluation/evaluate.py
```

修改：

```text
1. 新增 --tmp_root、--tmp_config、--allow-oracle-eval、--require-manifest。
2. 从 pred_path/TMP 读取 config_manifest.json。
3. --require-manifest 时缺 manifest 直接拒绝。
4. manifest uses_gt=true 时，除非 --allow-oracle-eval，否则拒绝。
5. manifest is_diagnostic_only=true 时，除非 --allow-oracle-eval，否则拒绝。
6. 保留 oracle-name guard。
7. TMP pre_points 路径改为 opt.tmp_root / opt.tmp_config / scene_pre_points.npy。
```

目的：解决 v6 P0-2，避免 GT-read diagnostic 只因不含 `oracle` 字符串而进入普通 evaluator。

### 1.2 mask_backproject score-mode 修复

文件：

```text
Stream3D/stream4d/export_scannet.py
Stream3D/tools/export_local_proposal_bank.py
```

修改：

```text
1. 新增 score_export_record(record, export_score_mode)。
2. mask_backproject object record 写入：
   observations
   area_score
   carrier_count
   reliability = observations * sqrt(max(area_score, 1.0))
3. export_local_proposal_bank 的 --export-score-mode choices 增加 reliability。
```

目的：解决 v6 P0-3。v5 local proposal 默认 `observations` score 实际缺字段，导致分数语义不符合计划。

### 1.3 fusion manifest support 记录

文件：

```text
Stream3D/tools/fuse_prediction_configs.py
```

修改：

```text
1. 新增 _external_support_signal(config)。
2. manifest 写入：
   drop_overlap_pre_points_config
   drop_overlap_pre_points_external_support
   drop_secondary_iou_threshold
   drop_secondary_overlap_mode
3. 若 drop_overlap_pre_points_config 是 scannet / Stream3D support，则 manifest 标记 diagnostic-only。
```

目的：解决 v6 P0-4，防止 Stream3D/scannet support 作为 selection signal 被误报为 pure Stream4D method。

### 1.4 D4RT geometry diagnostic

文件：

```text
Stream3D/tools/d4rt_geometry_diagnostic.py
Stream3D/tests/test_stream4d_protocol_fixes.py
```

修改：

```text
1. 新增 fit_sim3_umeyama(source, target)。
2. 从 D4RT carriers_window*.npz 读取同像素 carrier anchors。
3. 用 ScanNet depth/pose/intrinsics 得到 RGB-D world anchors。
4. 拟合 Sim3 并输出 window/scene residual。
5. 单测覆盖已知 Sim3 变换。
```

注意：这是几何诊断，不是 `G1 Stream3D-D4RT internal geometry` segmentation AP。

### 1.5 审计包补充

新增：

```text
Stream3D/splits/scannet_v6_probe5.txt
Stream3D/tests/fixtures/tiny_scene_prediction.npz
Stream3D/tests/fixtures/tiny_pre_points.npy
Stream3D/tests/fixtures/tiny_gt.txt
```

目的：满足 v6 计划中审计包复现入口和 toy fixture 要求。

### 1.6 minimal Typed Evidence Graph v3 candidate

文件：

```text
Stream3D/tools/export_typed_evidence_graph_v3.py
```

修改：

```text
1. 从 outputs/stream4d_v5_cache_96f_probe5/<scene>/local_props_window*.json 读取 mask observations。
2. 通过 ScanNetExporter._backproject_mask 将每个 observation 映射到 ScanNet mesh points。
3. 构建 typed graph node：frame_id、mask_id、coverage、point_ids、centroid、bbox、weak flag。
4. 构建 typed edges：
   positive_track：跨帧 point overlap。
   positive_complement：几何邻近补全。
   negative_conflict：同帧 overlap cannot-link。
   weak_bridge：弱节点 attach-only。
5. 用 DSU 合并 graph component，并阻止 same-frame conflict / cannot-link 合并。
6. 输出 prediction npz、TMP pre_points、object_dict.npy、config_manifest.json、summary.json。
```

审计边界：

```text
这是 minimal graph candidate。
它没有完整实现 v6 计划中的 core/fringe/reject support、negative_ownership、component split audit、object competition 或 split/merge lifecycle。
因此若结果失败，不能写成完整 v6 Typed Evidence Graph v3 已失败；只能写 minimal candidate 失败。
```

## 2. 验证命令

### 2.1 py_compile

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile evaluation/evaluate.py stream4d/*.py tools/*.py tests/*.py 2>&1 | tee logs/stream4d_v6_py_compile.log
```

结果：

```text
exit=0
```

### 2.2 unit tests

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests 2>&1 | tee logs/stream4d_v6_unit_tests.log
```

结果：

```text
Ran 12 tests in 1.368s
OK
```

### 2.3 import smoke

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY' 2>&1 | tee logs/stream4d_v6_import_smoke.log
import sys
sys.path.insert(0, '.')
mods = [
    'stream4d.run_scannet',
    'stream4d.object_memory_v2',
    'stream4d.export_scannet',
    'stream4d.reliable_densifier',
    'tools.verify_stream4d_metric_integrity',
    'tools.scan_reportable_configs',
    'tools.d4rt_geometry_diagnostic',
]
for mod in mods:
    __import__(mod)
    print(f'{mod} OK')
print('import smoke OK')
PY
```

结果：

```text
stream4d.run_scannet OK
stream4d.object_memory_v2 OK
stream4d.export_scannet OK
stream4d.reliable_densifier OK
tools.verify_stream4d_metric_integrity OK
tools.scan_reportable_configs OK
tools.d4rt_geometry_diagnostic OK
import smoke OK
```

## 3. v6 local proposal score-mode 重跑

目的：验证 v5 local proposal 失败是否只是 score-field bug。

输入 cache：

```text
Stream3D/outputs/stream4d_v5_cache_96f_probe5/
```

公共参数：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
DEBUG=outputs/stream4d_v5_cache_96f_probe5
SUMMARY=outputs/local_proposal_bank_v6
SCENES="scene0050_00 scene0011_00 scene0030_00 scene0081_01 scene0591_00"
```

导出命令模板：

```bash
for SCENE in $SCENES; do
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.export_local_proposal_bank \
    --debug-root "$DEBUG" \
    --seq-name "$SCENE" \
    --output-config "$CONFIG" \
    --backbone Cropformer \
    --same-frame-policy best_per_frame \
    --min-observations 2 \
    --min-frames 2 \
    --export-nn-radius 0.05 \
    --export-mask-sample-stride 2 \
    --export-mask-max-pixels 12000 \
    --export-max-masks-per-object 5 \
    --export-mask-min-relative-coverage 0.0 \
    --export-min-points-per-object 100 \
    --export-score-mode "$MODE" \
    --summary-root "$SUMMARY"
done 2>&1 | tee logs/stream4d_v6_localprop_score_${MODE}_export.log
```

运行的 modes/configs：

```text
MODE=one
CONFIG=stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075

MODE=observations
CONFIG=stream4d_v6_localprop_96f_probe5_min2_bestframe_score_observations_ioc075

MODE=area
CONFIG=stream4d_v6_localprop_96f_probe5_min2_bestframe_score_area_ioc075

MODE=reliability
CONFIG=stream4d_v6_localprop_96f_probe5_min2_bestframe_score_reliability_ioc075
```

第一次运行 `reliability` 暴露 CLI blocker：

```text
export_local_proposal_bank.py 的 --export-score-mode choices 没有 reliability。
```

修复后重新运行 `reliability`，导出成功。

评估命令模板：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_localprop_score_${MODE}_eval.log
```

结果文件：

```text
Stream3D/data/evaluation/scannet/stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_localprop_96f_probe5_min2_bestframe_score_observations_ioc075_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_localprop_96f_probe5_min2_bestframe_score_area_ioc075_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v6_localprop_96f_probe5_min2_bestframe_score_reliability_ioc075_class_agnostic.txt
```

汇总文件：

```text
Stream3D/outputs/local_proposal_bank_v6/stream4d_v6_localprop_score_modes_summary.json
```

## 4. v6 manifest / metric integrity

Reportable scanner：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIGS=stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075,stream4d_v6_localprop_96f_probe5_min2_bestframe_score_observations_ioc075,stream4d_v6_localprop_96f_probe5_min2_bestframe_score_area_ioc075,stream4d_v6_localprop_96f_probe5_min2_bestframe_score_reliability_ioc075
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --configs "$CONFIGS" \
  --output outputs/audit/v6_reportable_config_scan_localprop_score_modes.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_reportable_config_scan_localprop_score_modes.log
```

Metric integrity：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIGS=stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075,stream4d_v6_localprop_96f_probe5_min2_bestframe_score_observations_ioc075,stream4d_v6_localprop_96f_probe5_min2_bestframe_score_area_ioc075,stream4d_v6_localprop_96f_probe5_min2_bestframe_score_reliability_ioc075
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --configs "$CONFIGS" \
  --output outputs/audit/v6_metric_integrity_localprop_score_modes_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_metric_integrity_localprop_score_modes_probe5.log
```

## 5. D4RT geometry diagnostic

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v6_d4rt_geometry_probe5 \
  --max-anchors-per-window 2000 \
  2>&1 | tee logs/stream4d_v6_d4rt_geometry_probe5.log
```

输出：

```text
Stream3D/outputs/audit/v6_d4rt_geometry_probe5.csv
Stream3D/outputs/audit/v6_d4rt_geometry_probe5.json
Stream3D/outputs/audit/v6_d4rt_geometry_probe5.md
```

## 6. Dynamic Replica env check

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.check_dynamic_replica_env \
  --root data/dynamic-replica/v2 \
  --split valid \
  --output outputs/audit/dynamic_replica_env_v6.md \
  2>&1 | tee logs/stream4d_v6_dynamic_replica_env.log
```

输出：

```text
Stream3D/outputs/audit/dynamic_replica_env_v6.md
Stream3D/outputs/audit/dynamic_replica_env_v6.json
```

## 7. 代码审计包

打包目标：

```text
stream4d_v6_code_review_packet_20260608_2036.zip
stream4d_v6_code_review_packet_20260608_2036.sha256
stream4d_v6_code_review_packet_20260608_2036_filelist.txt
```

打包内容包括：

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/evaluate.py
Stream3D/evaluation/constants.py
Stream3D/evaluation/utils_3d.py
Stream3D/tests/*.py
Stream3D/tests/fixtures/*
Stream3D/configs/stream4d*.json
Stream3D/splits/scannet_v6_probe5.txt
Stream3D/splits/scannet_tune.txt
Stream3D/splits/scannet_final.txt
docs/stream4d_v6_method_first_audit_and_experiment_plan_for_codex.md
docs/stream4d_v6_执行日志.md
docs/stream4d_v6_实验结果复盘.md
git_status.txt
git_diff.patch
filelist.txt
import_smoke.log
unit_tests.log
metric_integrity.log
```

不包含 large prediction npz、ScanNet raw data、D4RT cache、checkpoints。

打包后干净解包 import smoke：

```bash
ROOT=/mnt/data/users/chengshun.wang/pjs/LoGeR
PKG=stream4d_v6_code_review_packet_20260608_2036
TEST=/tmp/${PKG}_clean_test
rm -rf "$TEST"
mkdir -p "$TEST"
unzip -q "$ROOT/${PKG}.zip" -d "$TEST"
cd "$TEST/$PKG/Stream3D"
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY' 2>&1 | tee "$ROOT/Stream3D/logs/stream4d_v6_packet_2036_clean_import_smoke.log"
import sys
sys.path.insert(0, '.')
mods = [
    'stream4d.run_scannet',
    'stream4d.object_memory_v2',
    'stream4d.export_scannet',
    'stream4d.reliable_densifier',
    'tools.verify_stream4d_metric_integrity',
    'tools.scan_reportable_configs',
    'tools.d4rt_geometry_diagnostic',
    'tools.export_typed_evidence_graph_v3',
]
for mod in mods:
    __import__(mod)
    print(f'{mod} OK')
print('clean packet import smoke OK')
PY
```

结果：

```text
tools.export_typed_evidence_graph_v3 OK
clean packet import smoke OK
```

## 8. 用户追问后的继续执行：minimal Typed Evidence Graph v3

用户指出“不继续解决问题”后，本轮继续补做 typed evidence graph 方向，而不是停在 localprop score-mode blocker。

### 8.1 继续后的验证

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/export_typed_evidence_graph_v3.py tests/test_stream4d_protocol_fixes.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests 2>&1 | tee logs/stream4d_v6_continue_unit_tests.log
```

结果：

```text
py_compile exit=0
Ran 12 tests in 1.377s
OK
```

### 8.2 scene0050 探针矩阵

公共输入：

```text
debug_root = outputs/stream4d_v5_cache_96f_probe5
seq_name = scene0050_00
summary_root = outputs/typed_evidence_graph_v6
```

minimal weak/complement 探针：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.export_typed_evidence_graph_v3 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v6_typedv3_scene0050_minimal \
  --backbone Cropformer \
  --min-coverage 0.001 \
  --weak-coverage 0.004 \
  --min-points-per-mask 80 \
  --min-points-per-object 100 \
  --export-nn-radius 0.05 \
  --min-track-shared 30 \
  --min-track-ioc 0.35 \
  --min-weak-shared 20 \
  --min-weak-ioc 0.20 \
  --min-conflict-shared 25 \
  --min-conflict-ioc 0.15 \
  --complement-max-centroid 0.20 \
  --complement-max-bbox-gap 0.05 \
  --min-component-observations 2 \
  --min-component-frames 2 \
  --score-mode quality \
  --summary-root outputs/typed_evidence_graph_v6 \
  2>&1 | tee logs/stream4d_v6_typedv3_scene0050_minimal_export.log
```

track-only ioc060 探针：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.export_typed_evidence_graph_v3 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v6_typedv3_scene0050_c2_trackonly_ioc060 \
  --backbone Cropformer \
  --min-coverage 0.001 \
  --weak-coverage 0.0 \
  --min-points-per-mask 80 \
  --min-points-per-object 100 \
  --export-nn-radius 0.05 \
  --min-track-shared 30 \
  --min-track-ioc 0.60 \
  --min-weak-shared 999999 \
  --min-weak-ioc 1.0 \
  --min-conflict-shared 25 \
  --min-conflict-ioc 0.15 \
  --complement-max-centroid 0.0 \
  --complement-max-bbox-gap 0.0 \
  --min-component-observations 2 \
  --min-component-frames 2 \
  --score-mode quality \
  --summary-root outputs/typed_evidence_graph_v6 \
  2>&1 | tee logs/stream4d_v6_typedv3_scene0050_c2_trackonly_ioc060_export.log
```

另外运行：

```text
stream4d_v6_typedv3_scene0050_c2_trackonly_ioc079
stream4d_v6_typedv3_scene0050_c3_weak_limited_ioc060
```

对应日志：

```text
Stream3D/logs/stream4d_v6_typedv3_scene0050_c2_trackonly_ioc079_export.log
Stream3D/logs/stream4d_v6_typedv3_scene0050_c3_weak_limited_ioc060_export.log
```

评估命令模板：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CONFIG}_eval.log
```

scene0050 结果：

| Config | AP | AP50 | AP25 | summary |
|---|---:|---:|---:|---|
| `stream4d_v6_typedv3_scene0050_minimal` | `0.011495007248902506` | `0.04739194753157756` | `0.4607985515043228` | `Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_scene0050_minimal_scene0050_00_summary.json` |
| `stream4d_v6_typedv3_scene0050_c2_trackonly_ioc060` | `0.046778612740200544` | `0.234317709001236` | `0.4910354817322768` | `Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_scene0050_c2_trackonly_ioc060_scene0050_00_summary.json` |
| `stream4d_v6_typedv3_scene0050_c2_trackonly_ioc079` | `0.04040711787907115` | `0.2108040535136109` | `0.48872778226274444` | `Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_scene0050_c2_trackonly_ioc079_scene0050_00_summary.json` |
| `stream4d_v6_typedv3_scene0050_c3_weak_limited_ioc060` | `0.03985265384110804` | `0.19815002337241225` | `0.4408752842946256` | `Stream3D/outputs/typed_evidence_graph_v6/stream4d_v6_typedv3_scene0050_c3_weak_limited_ioc060_scene0050_00_summary.json` |

scene0050 上最好的探针是 track-only ioc060，因此推到 probe5。

### 8.3 probe5 track-only ioc060

导出命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIG=stream4d_v6_typedv3_probe5_c2_trackonly_ioc060
for SCENE in scene0011_00 scene0030_00 scene0050_00 scene0081_01 scene0591_00; do
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.export_typed_evidence_graph_v3 \
    --debug-root outputs/stream4d_v5_cache_96f_probe5 \
    --seq-name "$SCENE" \
    --output-config "$CONFIG" \
    --backbone Cropformer \
    --min-coverage 0.001 \
    --weak-coverage 0.0 \
    --min-points-per-mask 80 \
    --min-points-per-object 100 \
    --export-nn-radius 0.05 \
    --min-track-shared 30 \
    --min-track-ioc 0.60 \
    --min-weak-shared 999999 \
    --min-weak-ioc 1.0 \
    --min-conflict-shared 25 \
    --min-conflict-ioc 0.15 \
    --complement-max-centroid 0.0 \
    --complement-max-bbox-gap 0.0 \
    --min-component-observations 2 \
    --min-component-frames 2 \
    --score-mode quality \
    --summary-root outputs/typed_evidence_graph_v6 \
    2>&1 | tee logs/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_${SCENE}_export.log
done
```

实际日志：

```text
Stream3D/logs/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0011_export.log
Stream3D/logs/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0030_export.log
Stream3D/logs/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0050_export.log
Stream3D/logs/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0081_export.log
Stream3D/logs/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_scene0591_export.log
```

评估命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v6_typedv3_probe5_c2_trackonly_ioc060 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_eval.log
```

结果：

```text
Stream3D/data/evaluation/scannet/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_class_agnostic.txt
AP/AP50/AP25 = 0.03347846361047295 / 0.12849540654750258 / 0.3977530181719252
```

per-scene summary：

| Scene | nodes | positive_track_edges | negative_conflict_edges | accepted_track | raw_components | kept_components | dropped_components | union_points | backproject_hit_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene0011_00` | `255` | `935` | `21` | `176` | `79` | `19` | `60` | `29325` | `0.6416344583701575` |
| `scene0030_00` | `1167` | `6856` | `154` | `1019` | `148` | `98` | `50` | `90459` | `0.8386020550620544` |
| `scene0050_00` | `539` | `3373` | `66` | `467` | `72` | `49` | `23` | `34242` | `0.925039969291374` |
| `scene0081_01` | `238` | `570` | `8` | `162` | `76` | `29` | `47` | `24311` | `0.47566825228504306` |
| `scene0591_00` | `1155` | `6696` | `230` | `996` | `159` | `91` | `68` | `84318` | `0.7687900159074992` |

Metric integrity：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIGS=stream4d_v6_typedv3_probe5_c2_trackonly_ioc060
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --configs "$CONFIGS" \
  --output outputs/audit/v6_metric_integrity_typedv3_probe5_c2_trackonly_ioc060.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_metric_integrity_typedv3_probe5_c2_trackonly_ioc060.log
```

结果：

```text
phase0_pass=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=1
num_suspicious_configs=0
alignment mean/min = 1.0 / 1.0
alignment failed = 0
pre % = 24.1733
union % = 24.1733
#pred = 57.20
```

证据：

```text
Stream3D/outputs/audit/v6_metric_integrity_typedv3_probe5_c2_trackonly_ioc060.md
Stream3D/outputs/audit/v6_metric_integrity_typedv3_probe5_c2_trackonly_ioc060.json
Stream3D/data/prediction/stream4d_v6_typedv3_probe5_c2_trackonly_ioc060_class_agnostic/config_manifest.json
```

结论：

```text
minimal typed v3 已经继续实现并真实跑完 probe5，但失败。
失败不是 manifest、oracle 或 object_dict alignment 问题；metric integrity 通过。
主要现象是 graph component 输出过少：#pred=57.20，远低于 localprop 173.20，也远低于 v4.1 current best 415.6。
它将多 observation 聚合成较少 component 后，AP 从 localprop best 0.1125 降到 0.0335，说明当前 typed edge 合并策略过度合并/错合并，缺少 core/fringe/reject、negative_ownership 和 split audit。
```
## 2026-06-08 continuation：Phase D core/fringe/reject 与 ownership-aware WTA

触发原因：

```text
用户追问“达成目标了吗，没有请继续”。
截至上一版 v6 复盘，best reportable method 仍未达到 probe5 gate。
本轮按计划中 Phase D 推荐方向继续：core/fringe/reject、ownership-aware WTA，并避免继续简单 score sweep。
```

### 代码修改

修改文件：

```text
Stream3D/tools/export_typed_evidence_graph_v3.py
Stream3D/tools/split_core_fringe_prediction.py
```

修改摘要：

```text
1. export_typed_evidence_graph_v3.py：
   新增 --support-mode union/core/core_connected_fringe。
   新增 --min-core-frames、--min-core-observations、--fringe-radius。
   summary 中写 component_union/core/fringe/reject 点数与比例。
   默认 support-mode=union，保持旧结果可复现。

2. split_core_fringe_prediction.py：
   补写 config_manifest.json，避免后处理 prediction 无 manifest。
   新增 --assignment-mode low_conflict/wta。
   新增 --wta-priority score/small_area/large_area/score_over_sqrt_area。
   low_conflict 为旧行为；wta 为 Phase D ownership-aware WTA 尝试。
```

验证命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/split_core_fringe_prediction.py 2>&1 | tee logs/stream4d_v6_wta_py_compile.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests 2>&1 | tee logs/stream4d_v6_wta_unit_tests.log
```

验证结果：

```text
py_compile exit=0
unit tests: Ran 12 tests in 1.347s, OK
```

### Localprop score-one core-only probe5

scene0050 smoke 命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIG=stream4d_v6_localprop_scene0050_coreonly_scoreone_ioc075
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.split_core_fringe_prediction \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --input-config stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075 \
  --output-config "$CONFIG" \
  --support-config stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075 \
  --max-core-owners 1 \
  --min-core-points 20 \
  --min-core-ratio 0.05 \
  --min-low-points 20 \
  --min-support-area 20 \
  --low-mode none \
  --tmp-policy recompute \
  --summary-root outputs/core_fringe_v6 \
  2>&1 | tee logs/stream4d_v6_localprop_scene0050_coreonly_scoreone_ioc075_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_localprop_scene0050_coreonly_scoreone_ioc075_eval.log
```

scene0050 结果：

```text
AP/AP50/AP25 = 0.18047138047138048 / 0.40909090909090906 / 0.55078125
```

probe5 命令同上，替换：

```text
CONFIG=stream4d_v6_localprop_probe5_coreonly_scoreone_ioc075
--seq-list splits/scannet_v6_probe5.txt
log:
  logs/stream4d_v6_localprop_probe5_coreonly_scoreone_ioc075_export.log
  logs/stream4d_v6_localprop_probe5_coreonly_scoreone_ioc075_eval.log
```

probe5 结果：

```text
AP/AP50/AP25 = 0.1326388888888889 / 0.25392156862745097 / 0.5254354960868468
```

metric integrity：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIGS=stream4d_v6_localprop_probe5_coreonly_scoreone_ioc075
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --configs "$CONFIGS" \
  --output outputs/audit/v6_metric_integrity_localprop_probe5_coreonly_scoreone_ioc075.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_metric_integrity_localprop_probe5_coreonly_scoreone_ioc075.log
```

metric integrity 结果：

```text
phase0_pass=True
mean_pre_points_ratio=0.1313102002615043
mean_prediction_union_ratio=0.1313102002615043
mean_num_pred_instances=76.0
```

### Localprop ownership variants

scene0050 分叉：

```text
stream4d_v6_localprop_scene0050_coreown2_scoreone_ioc075:
  max_core_owners=2, min_core_ratio=0.05
  AP/AP50/AP25 = 0.193... / 0.371... / 0.593...

stream4d_v6_localprop_scene0050_coreown2_ratio010_scoreone_ioc075:
  max_core_owners=2, min_core_ratio=0.10
  AP/AP50/AP25 = 0.194... / 0.373... / 0.597...

stream4d_v6_localprop_scene0050_coreown2_ratio020_scoreone_ioc075:
  max_core_owners=2, min_core_ratio=0.20
  AP/AP50/AP25 = 0.22676107480029048 / 0.39351851851851855 / 0.6013071895424836

stream4d_v6_localprop_scene0050_coreown2_ratio030_scoreone_ioc075:
  max_core_owners=2, min_core_ratio=0.30
  AP/AP50/AP25 = 0.193... / 0.347... / 0.587...

stream4d_v6_localprop_scene0050_coreown2_ratio040_scoreone_ioc075:
  max_core_owners=2, min_core_ratio=0.40
  AP/AP50/AP25 = 0.212... / 0.355... / 0.645...
```

失败的 fringe/WTA scene0050 分叉：

```text
stream4d_v6_localprop_scene0050_corefringe_low001_scoreone_ioc075:
  AP/AP50/AP25 = 0.05037695279245847 / 0.1475545900178253 / 0.2781808035714286

stream4d_v6_localprop_scene0050_corefringeplus_low001_scoreone_ioc075:
  AP/AP50/AP25 = 0.05823640715060206 / 0.17542613636363635 / 0.29265625

stream4d_v6_localprop_scene0050_wta_smallarea_scoreone_ioc075:
  AP/AP50/AP25 = 0.0813218279736137 / 0.1911670918367347 / 0.5151417525773196

stream4d_v6_localprop_scene0050_wta_scoresqrtarea_scoreone_ioc075:
  AP/AP50/AP25 = 0.0813218279736137 / 0.1911670918367347 / 0.5151417525773196

stream4d_v6_localprop_scene0050_wta_largearea_scoreone_ioc075:
  AP/AP50/AP25 = 0.13270049283154123 / 0.33845766129032256 / 0.5507172131147541
```

最佳 scene0050 ownership 分叉推到 probe5：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIG=stream4d_v6_localprop_probe5_coreown2_ratio020_scoreone_ioc075
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.split_core_fringe_prediction \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075 \
  --output-config "$CONFIG" \
  --support-config stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075 \
  --max-core-owners 2 \
  --min-core-points 20 \
  --min-core-ratio 0.20 \
  --min-low-points 20 \
  --min-support-area 20 \
  --low-mode none \
  --tmp-policy recompute \
  --summary-root outputs/core_fringe_v6 \
  2>&1 | tee logs/stream4d_v6_localprop_probe5_coreown2_ratio020_scoreone_ioc075_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_localprop_probe5_coreown2_ratio020_scoreone_ioc075_eval.log
```

probe5 结果：

```text
AP/AP50/AP25 = 0.13568501383679288 / 0.27218506615709775 / 0.520180250783699
```

metric integrity：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIGS=stream4d_v6_localprop_probe5_coreown2_ratio020_scoreone_ioc075
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --configs "$CONFIGS" \
  --output outputs/audit/v6_metric_integrity_localprop_probe5_coreown2_ratio020_scoreone_ioc075.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_metric_integrity_localprop_probe5_coreown2_ratio020_scoreone_ioc075.log
```

metric integrity 结果：

```text
phase0_pass=True
mean_pre_points_ratio=0.21967539878907916
mean_prediction_union_ratio=0.21967539878907916
mean_num_pred_instances=98.0
```

### v4.1 current-best core-only 审计

目的：

```text
v4.1 current-best AP 更接近 gate，但 #pred=415.6 过多。
本轮尝试用同一 core-only 后处理压 #pred，检查是否能变成可报告 v6 method。
```

输入：

```text
stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
原 probe5 AP/AP50/AP25 = 0.2816154378895367 / 0.4975830336133305 / 0.6902541954477854
```

tiny-pre_points recompute 版本：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIG=stream4d_v6_v41self_probe5_coreonly
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.split_core_fringe_prediction \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config "$INPUT" \
  --output-config "$CONFIG" \
  --support-config "$INPUT" \
  --assignment-mode low_conflict \
  --max-core-owners 1 \
  --min-core-points 20 \
  --min-core-ratio 0.05 \
  --min-low-points 20 \
  --min-support-area 20 \
  --low-mode none \
  --tmp-policy recompute \
  --summary-root outputs/core_fringe_v6 \
  2>&1 | tee logs/stream4d_v6_v41self_probe5_coreonly_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_v41self_probe5_coreonly_eval.log
```

结果：

```text
AP/AP50/AP25 = 0.6062679879644165 / 0.7970712323390894 / 0.906634288330717
mean_pre_points_ratio = 0.00925595426581326
mean_prediction_union_ratio = 0.00925595426581326
mean_num_pred_instances = 11.2
```

审计结论：

```text
这是 tiny-pre_points 高 AP，不满足覆盖 gate，不能作为成功。
```

support-policy 对照版本：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIG=stream4d_v6_v41self_probe5_coreonly_on_v41support
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.split_core_fringe_prediction \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config "$INPUT" \
  --output-config "$CONFIG" \
  --support-config "$INPUT" \
  --assignment-mode low_conflict \
  --max-core-owners 1 \
  --min-core-points 20 \
  --min-core-ratio 0.05 \
  --min-low-points 20 \
  --min-support-area 20 \
  --low-mode none \
  --tmp-policy support \
  --summary-root outputs/core_fringe_v6 \
  2>&1 | tee logs/stream4d_v6_v41self_probe5_coreonly_on_v41support_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_v41self_probe5_coreonly_on_v41support_eval.log
```

结果：

```text
AP/AP50/AP25 = 0.06277817955517258 / 0.1449568388496515 / 0.18177814707104278
mean_pre_points_ratio = 0.04514451433782776
mean_prediction_union_ratio = 0.00925595426581326
mean_num_pred_instances = 11.2
phase0_pass = true
```

证据：

```text
Stream3D/outputs/audit/v6_metric_integrity_v41self_probe5_coreonly.json
Stream3D/outputs/audit/v6_metric_integrity_v41self_probe5_coreonly_on_v41support.json
Stream3D/outputs/core_fringe_v6/stream4d_v6_v41self_probe5_coreonly_summary.json
Stream3D/outputs/core_fringe_v6/stream4d_v6_v41self_probe5_coreonly_on_v41support_summary.json
```

### Boundary-growth 近似：core + low-score full mask

目的：

```text
core-only AP/AP25 有小幅正信号，但召回和边界不足。
尝试用 high-score core seed + low-score original full mask 恢复 support。
这是 Phase D boundary growth 的低成本近似，不读 GT。
```

scene0050 命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.split_core_fringe_prediction \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --input-config stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075 \
  --output-config <CONFIG> \
  --support-config stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075 \
  --assignment-mode low_conflict \
  --max-core-owners <1_or_2> \
  --min-core-points 20 \
  --min-core-ratio <0.05_or_0.20> \
  --min-low-points 20 \
  --min-support-area 20 \
  --low-mode full \
  --low-score 0.001 \
  --tmp-policy recompute \
  --summary-root outputs/core_fringe_v6 \
  2>&1 | tee logs/<CONFIG>_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/<CONFIG>_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/<CONFIG>_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config <CONFIG> \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/<CONFIG>_eval.log
```

运行：

```text
stream4d_v6_localprop_scene0050_corefull_low001_scoreone_ioc075:
  max_core_owners=1, min_core_ratio=0.05
  AP/AP50/AP25 = 0.05823640715060206 / 0.17542613636363635 / 0.29265625

stream4d_v6_localprop_scene0050_coreown2_ratio020_full_low001_scoreone_ioc075:
  max_core_owners=2, min_core_ratio=0.20
  AP/AP50/AP25 = 0.09026673251270163 / 0.23627857541120714 / 0.4650996654586005
```

结论：

```text
core + low-score full mask 没有恢复召回，反而明显低于 core-only。
该分支不推 probe5。
```

### 更新代码审计包

打包前审计命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile evaluation/evaluate.py stream4d/*.py tools/*.py tests/*.py 2>&1 | tee logs/stream4d_v6_package_py_compile.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY' 2>&1 | tee logs/stream4d_v6_package_import_smoke.log
import stream4d.run_scannet
import stream4d.object_memory_v2
import stream4d.export_scannet
import stream4d.reliable_densifier
import tools.verify_stream4d_metric_integrity
import tools.scan_reportable_configs
print('import smoke OK')
PY
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests 2>&1 | tee logs/stream4d_v6_package_unit_tests.log
```

结果：

```text
py_compile exit=0
import smoke OK
unit tests: Ran 12 tests in 1.423s, OK
```

打包产物：

```text
stream4d_v6_code_review_packet_20260608_2102.zip
stream4d_v6_code_review_packet_20260608_2102.sha256
stream4d_v6_code_review_packet_20260608_2102_filelist.txt
sha256 = 5b212f17e79c1b095cff0b2fe12c16d00f7888ab209ef65a7aee335e4de32726
```

说明：

```text
该包包含本轮改动后的 Stream3D/stream4d/*.py、Stream3D/tools/*.py、evaluation、tests/fixtures、v6 文档、关键 evaluation txt、summary json、metric integrity md/json、命令日志、git_status.txt、git_diff.patch 和 filelist.txt。
不包含 raw ScanNet 数据、prediction npz 或 checkpoint。
```

## 12. 用户追问后的继续执行：geometry-aware radius growth

动机：

```text
Phase D 的 core-only / coreown2 结果显示：
1. core seed 更纯，但 support/union 过小；
2. low-score full mask 会把污染带回来；
3. 因此按计划尝试 geometry-aware boundary growth，而不是继续直接扩大 fringe。
```

代码修改：

```text
文件：Stream3D/tools/split_core_fringe_prediction.py
修改：
  1. 增加 open3d mesh points 读取和 scipy cKDTree radius growth。
  2. 新增 CLI：
       --growth-mode none/radius
       --growth-candidate-mode support/full
       --growth-radius
       --growth-max-owners
       --backbone
  3. summary 新增：
       growth_candidate_points
       growth_kept_points
       growth_added_points
  4. manifest support_policy 记录 growth 参数。

审计边界：
  只使用预测 mask、support pre_points 与 ScanNet mesh 几何。
  不读取 GT，不改变 evaluator。
```

验证命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/split_core_fringe_prediction.py \
  2>&1 | tee logs/stream4d_v6_radius_growth_py_compile.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests \
  2>&1 | tee logs/stream4d_v6_radius_growth_unit_tests.log
```

验证结果：

```text
py_compile exit=0
unit tests: Ran 12 tests in 1.433s, OK
```

scene0050 命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIG=<CONFIG>
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.split_core_fringe_prediction \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --input-config stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075 \
  --output-config "$CONFIG" \
  --support-config stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075 \
  --assignment-mode low_conflict \
  --max-core-owners <1_or_2> \
  --min-core-points 20 \
  --min-core-ratio <0.05_or_0.20> \
  --growth-mode radius \
  --growth-candidate-mode <support_or_full> \
  --growth-radius <0.03_or_0.05> \
  --growth-max-owners <0_or_2> \
  --low-mode none \
  --tmp-policy recompute \
  --summary-root outputs/core_fringe_v6 \
  2>&1 | tee logs/${CONFIG}_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CONFIG}_eval.log
```

结果：

| Config | seed | growth | added points | AP | AP50 | AP25 |
|---|---|---|---:|---:|---:|---:|
| `stream4d_v6_localprop_scene0050_coreown2_rgrow_support003_g2_scoreone_ioc075` | owner<=2, ratio>=0.20 | support r=0.03 max_owner=2 | `0` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` |
| `stream4d_v6_localprop_scene0050_coreown2_rgrow_support005_g2_scoreone_ioc075` | owner<=2, ratio>=0.20 | support r=0.05 max_owner=2 | `0` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` |
| `stream4d_v6_localprop_scene0050_coreown2_rgrow_full003_g2_scoreone_ioc075` | owner<=2, ratio>=0.20 | full r=0.03 max_owner=2 | `0` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` |
| `stream4d_v6_localprop_scene0050_coreown2_rgrow_full005_g2_scoreone_ioc075` | owner<=2, ratio>=0.20 | full r=0.05 max_owner=2 | `0` | `0.22676107480029048` | `0.39351851851851855` | `0.6013071895424836` |
| `stream4d_v6_localprop_scene0050_core1_rgrow_support003_g2_scoreone_ioc075` | owner<=1, ratio>=0.05 | support r=0.03 max_owner=2 | `7204` | `0.17589366369854176` | `0.33217189314750295` | `0.5296167247386759` |
| `stream4d_v6_localprop_scene0050_core1_rgrow_support005_g0_scoreone_ioc075` | owner<=1, ratio>=0.05 | support r=0.05 no owner cap | `19016` | `0.1395385980479148` | `0.35758651286601595` | `0.49778172138420584` |
| `stream4d_v6_localprop_scene0050_core1_rgrow_full003_g2_scoreone_ioc075` | owner<=1, ratio>=0.05 | full r=0.03 max_owner=2 | `7204` | `0.17589366369854176` | `0.33217189314750295` | `0.5296167247386759` |
| `stream4d_v6_localprop_scene0050_core1_rgrow_full005_g0_scoreone_ioc075` | owner<=1, ratio>=0.05 | full r=0.05 no owner cap | `19016` | `0.1395385980479148` | `0.35758651286601595` | `0.49778172138420584` |

结论：

```text
1. owner<=2 seed 再用 owner<=2 growth 是 no-op，added_points=0。
2. 从纯 core seed 做 radius growth 确实增加了 7204/19016 点，但 AP/AP50/AP25 均低于 scene0050 core-only 和 coreown2_ratio020。
3. 该分支不推 probe5。
4. 负结果说明简单几何半径补点仍会引入边界污染；需要回到 object candidate / competition 方向。
```

## 13. 用户追问后的继续执行：Phase E object competition small-object rescue

动机：

```text
v4.1 current best 接近 AP/AP50/AP25 gate，但 #pred=415.6 超过 <=300。
已有 pure suppression / greedy tier 历史结果会明显降 AP。
按 Phase E 失败预案，尝试 small-object rescue：压重复时保留少量名额给“support 小、提供新点、不过度重叠”的候选。
```

代码修改：

```text
文件：Stream3D/tools/object_competition_rank.py
修改：
  1. 输出 prediction/TMP manifest，标明 uses_gt=false。
  2. 增加 small rescue 参数：
       --small-rescue-reserve
       --small-rescue-min-support-area
       --small-rescue-max-support-area
       --small-rescue-min-novel-points
       --small-rescue-overlap-threshold
       --small-rescue-overlap-mode
  3. summary 新增：
       num_selected_base
       num_selected_small_rescue
       small_rescue_rejected_area/novelty/overlap

实现语义：
  max_instances 中预留 reserve 个槽位给 small rescue，不允许超过 #pred gate。
  rescue 只看预测 mask 与 support pre_points，不读 GT。
```

验证命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/object_competition_rank.py \
  2>&1 | tee logs/stream4d_v6_object_comp_small_rescue_py_compile.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests \
  2>&1 | tee logs/stream4d_v6_object_comp_small_rescue_unit_tests.log
```

验证结果：

```text
py_compile exit=0
unit tests: Ran 12 tests in 1.643s, OK
```

scene0050 基线 view：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
# 将 v4.1 probe5 prediction/TMP 中的 scene0050 拷贝为 one-scene view，只用于单场景对照。
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5_scene0050_view_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5_scene0050_view_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5_scene0050_view \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v6_e4_scene0050_baseline_view_eval.log
```

scene0050 基线结果：

```text
AP/AP50/AP25 = 0.28717796840958604 / 0.47205882352941175 / 0.7758333333333333
```

scene0050 small rescue 命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
CONFIG=<CONFIG>
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.object_competition_rank \
  --root . \
  --seq-list splits/scannet_scene0050.txt \
  --input-config "$INPUT" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$INPUT" \
  --quality-mode <score_unique_compact_or_area_unique> \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 0.85 \
  --min-support-area 1 \
  --max-instances 300 \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve <0_or_40_or_80> \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area <0_or_1500_or_2500> \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/object_competition_v6 \
  2>&1 | tee logs/${CONFIG}_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CONFIG}_eval.log
```

scene0050 结果：

| Config | AP | AP50 | AP25 | selected | small rescue | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e4_scene0050_objcomp_m300_g085_scoreunique_preserve` | `0.21159477124183007` | `0.3223529411764706` | `0.546` | `183` | `0` | pure competition 明显低于 v4.1 scene0050 |
| `stream4d_v6_e4_scene0050_objcomp_m300_g085_sr40_a1500_n80_preserve` | `0.23919455169455173` | `0.400965250965251` | `0.6830357142857142` | `186` | `3` | small rescue 有正信号但仍低于 v4.1 |
| `stream4d_v6_e4_scene0050_objcomp_m300_g085_sr80_a2500_n80_preserve` | `0.23919455169455173` | `0.400965250965251` | `0.6830357142857142` | `186` | `3` | 与 sr40 相同 |
| `stream4d_v6_e4_scene0050_objcomp_m300_g085_sr80_a2500_n80_areaunique_preserve` | `0.23524904214559386` | `0.37931034482758624` | `0.7293103448275862` | `185` | `2` | AP25 较高，但 AP/AP50 更低 |

probe5 推进命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
CONFIG=stream4d_v6_e4_probe5_objcomp_m300_g085_sr40_a1500_n80_preserve
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.object_competition_rank \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config "$INPUT" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$INPUT" \
  --quality-mode score_unique_compact \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 0.85 \
  --min-support-area 1 \
  --max-instances 300 \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve 40 \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area 1500 \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/object_competition_v6 \
  2>&1 | tee logs/${CONFIG}_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CONFIG}_eval.log
```

probe5 结果：

```text
AP/AP50/AP25 = 0.26253062027752616 / 0.47629926485603435 / 0.6499047035600027
mean_num_pred_instances = 173.0
mean_pre_points_ratio = 0.04514451433782776
mean_prediction_union_ratio = 0.03537121456885783
mean_num_selected_base = 170.0
mean_num_selected_small_rescue = 3.0
```

metric integrity 命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIG=stream4d_v6_e4_probe5_objcomp_m300_g085_sr40_a1500_n80_preserve
# 第一次少传 --orig-stream3d-root，日志为参数错误；随后用下面命令重跑成功。
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --configs "$CONFIG" \
  --seq-list splits/scannet_v6_probe5.txt \
  --backbone Cropformer \
  --output outputs/audit/v6_metric_integrity_${CONFIG}.md \
  --require-manifest \
  2>&1 | tee logs/v6_metric_integrity_${CONFIG}_rerun.log
```

metric integrity 结果：

```text
phase0_pass = true
manifest_exists = true
uses_gt = false
num_configs_missing_manifest = 0
num_uses_gt_and_method_result = 0
```

结论：

```text
1. small rescue 比 pure competition 有真实正信号，但仍低于 v4.1 current best。
2. probe5 虽然 #pred=173 <=300，但 AP/AP50/AP25 仍低于 gate，也低于 v4.1 baseline。
3. 这说明当前 object competition 的盲压缩仍误删 recall；重复 prediction 不是唯一主因。
4. Phase E 未通过，不启动 full ScanNet final。
```

### 更新代码审计包 r2

打包前审计命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile evaluation/evaluate.py stream4d/*.py tools/*.py tests/*.py \
  2>&1 | tee logs/stream4d_v6_package_py_compile_r2.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY' 2>&1 | tee logs/stream4d_v6_package_import_smoke_r2.log
import stream4d.run_scannet
import stream4d.object_memory_v2
import stream4d.export_scannet
import stream4d.reliable_densifier
import tools.verify_stream4d_metric_integrity
import tools.scan_reportable_configs
import tools.object_competition_rank
import tools.split_core_fringe_prediction
print('import smoke OK')
PY
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests \
  2>&1 | tee logs/stream4d_v6_package_unit_tests_r2.log
```

结果：

```text
py_compile exit=0
import smoke OK
unit tests: Ran 12 tests in 1.391s, OK
```

打包产物：

```text
stream4d_v6_code_review_packet_20260608_2121.zip
stream4d_v6_code_review_packet_20260608_2121.sha256
stream4d_v6_code_review_packet_20260608_2121_filelist.txt
```

说明：

```text
该包包含最新 Stream3D/tools/object_competition_rank.py、split_core_fringe_prediction.py、typed graph 工具、manifest/metric 工具、Stream3D/stream4d/*.py、evaluation、tests、v6 文档、radius-growth / E4 small rescue 的 evaluation txt、summary json、metric integrity json/md、关键命令日志、git_status 和 git_diff。
不包含 raw ScanNet 数据、prediction npz 或 checkpoint。
最终 sha256 以同名 .sha256 文件和最终回复为准；不在包内文档中写死，避免 package self-hash 循环。
```

## 12. continuation：object competition no-group / top-k 压缩

### 12.1 执行原因

用户追问“达成目标了吗，没有请继续”。截至 E4 small-rescue：

```text
stream4d_v6_e4_probe5_objcomp_m300_g085_sr40_a1500_n80_preserve
AP/AP50/AP25 = 0.26253062027752616 / 0.47629926485603435 / 0.6499047035600027
#pred = 173.0
```

该结果 #pred pass 但 AP/AP50/AP25 下降，说明 g085 object competition 过度压缩/误删 recall。scene0050 上 g098 已显示更少 grouping 会恢复 AP25，因此继续跑 g098 和 no-group/top-k 反事实。

注意：

```text
本节实验均基于已有 prediction/TMP 做 CPU postprocess + evaluator + metric integrity。
没有重新跑 D4RT/cache/model forward，因此不会调用 GPU；GPU 4/5/6/7 保留给后续需要模型推理或 cache 生成的实验。
```

### 12.2 scene0050 g095/g098 探针

命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
SPLIT=splits/scannet_scene0050.txt
CONFIG=<CONFIG>
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.object_competition_rank \
  --root . \
  --seq-list "$SPLIT" \
  --input-config "$INPUT" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$INPUT" \
  --quality-mode score_unique_compact \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold <0.95_or_0.98> \
  --min-support-area 1 \
  --max-instances 300 \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve <0_or_40> \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area <0_or_1500> \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/object_competition_v6 \
  2>&1 | tee logs/${CONFIG}_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CONFIG}_eval.log
```

结果：

| Config | AP | AP50 | AP25 | selected | small rescue | output_union |
|---|---:|---:|---:|---:|---:|---:|
| `stream4d_v6_e4_scene0050_objcomp_m300_g095_scoreunique_preserve` | `0.2448429198429199` | `0.36320166320166325` | `0.5913461538461539` | `207` | `0` | `8974` |
| `stream4d_v6_e4_scene0050_objcomp_m300_g098_scoreunique_preserve` | `0.26998737373737375` | `0.43738636363636363` | `0.7758333333333333` | `216` | `0` | `10606` |
| `stream4d_v6_e4_scene0050_objcomp_m300_g095_sr40_a1500_n80_preserve` | `0.2503586691086691` | `0.40054945054945057` | `0.6830357142857142` | `209` | `2` | `9806` |
| `stream4d_v6_e4_scene0050_objcomp_m300_g098_sr40_a1500_n80_preserve` | `0.26998737373737375` | `0.43738636363636363` | `0.7758333333333333` | `216` | `0` | `10606` |

结论：

```text
g098 明显优于 g085/g095，说明 earlier competition 误删 recall。
g098 下 small rescue 没再新增实例，说明当前 rescue 条件不是主要瓶颈。
```

### 12.3 g098 推到 probe5

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
CONFIG=stream4d_v6_e4_probe5_objcomp_m300_g098_scoreunique_preserve
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.object_competition_rank \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config "$INPUT" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$INPUT" \
  --quality-mode score_unique_compact \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 0.98 \
  --min-support-area 1 \
  --max-instances 300 \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve 0 \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area 0 \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/object_competition_v6 \
  2>&1 | tee logs/${CONFIG}_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CONFIG}_eval.log
```

结果：

```text
AP/AP50/AP25 = 0.26605892239144985 / 0.4726395846667555 / 0.6724933696462123
mean_num_selected = 190.6
mean_prediction_union_ratio = 0.03612109191439577
metric integrity phase0_pass = true
```

### 12.4 no-group/top-k 反事实

原因：

```text
g098 仍将 mean_num_selected 压到 190.6，低于 #pred gate 300 很多。
为确认 grouping 是否仍是主损伤，设置 group_overlap_threshold=1.01，使候选基本不被合并，只做 top-k 选择。
```

命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
CONFIG=<CONFIG>
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.object_competition_rank \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config "$INPUT" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$INPUT" \
  --quality-mode <score_unique_compact_or_compact_only> \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 1.01 \
  --min-support-area 1 \
  --max-instances <300_or_500_or_650_or_670> \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve 0 \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area 0 \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/object_competition_v6 \
  2>&1 | tee logs/${CONFIG}_export.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CONFIG}_eval.log
```

结果：

| Config | AP | AP50 | AP25 | mean #pred | union % | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e4_probe5_objcomp_m300_g101_scoreunique_preserve` | `0.2785629906546174` | `0.49121769019013883` | `0.6894759991330806` | `219.0` | `4.0756%` | 接近 v4.1，#pred pass |
| `stream4d_v6_e4_probe5_objcomp_m500_g101_scoreunique_preserve` | `0.28063532510475836` | `0.4957469672224599` | `0.6881410431792294` | `265.6` | n/a | 更接近 v4.1 |
| `stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve` | `0.28204965102921625` | `0.4982312805326885` | `0.6912642582412274` | `295.6` | `4.2507%` | AP/AP50/AP25 小幅超过 v4.1，#pred pass |
| `stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve` | `0.28483247256897415` | `0.5039622641509434` | `0.6719147248897401` | `295.6` | `4.1578%` | 当前最高 AP/AP50，但 AP25 退化 |
| `stream4d_v6_e4_probe5_objcomp_m670_g101_score_unique_compact_preserve` | `0.28203394183547537` | `0.49820855659464575` | `0.691223277348379` | `299.6` | n/a | 与 m650 基本相同 |
| `stream4d_v6_e4_probe5_objcomp_m670_g101_compact_only_preserve` | `0.28382830955380606` | `0.5019341996506697` | `0.6832167422177096` | `299.6` | n/a | 不如 m650 compact |

metric integrity：

```text
stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve:
  phase0_pass = true
  uses_gt = false
  mean_num_pred_instances = 295.6
  mean_prediction_union_ratio = 0.042507400047896333

stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve:
  phase0_pass = true
  uses_gt = false
  mean_num_pred_instances = 295.6
  mean_prediction_union_ratio = 0.0415780150820649
```

结论：

```text
1. no-group/top-k 证明 grouping 是前一轮 E4 的主要损伤来源。
2. m650 score_unique 在 #pred<=300 下小幅超过 v4.1 current best：
   0.2820496510 / 0.4982312805 / 0.6912642582
   vs v4.1 0.2816154379 / 0.4975830336 / 0.6902541954。
3. m650 compact_only 进一步提高 AP/AP50 到 0.284832/0.503962，但 AP25 降到 0.671915。
4. 两者仍没有达到 v6 gate：
   AP < 0.32, AP50 < 0.53, AP25 < 0.70, union << 0.94。
5. 因此本轮得到的是“压缩版 v4.1 小幅正信号”，不是 v6 method success。
```

### 12.5 更新代码审计包 r3

打包前审计命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile evaluation/evaluate.py stream4d/*.py tools/*.py tests/*.py \
  2>&1 | tee logs/stream4d_v6_package_py_compile_r3.log
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY' 2>&1 | tee logs/stream4d_v6_package_import_smoke_r3.log
import stream4d.run_scannet
import stream4d.object_memory_v2
import stream4d.export_scannet
import stream4d.reliable_densifier
import tools.verify_stream4d_metric_integrity
import tools.scan_reportable_configs
import tools.object_competition_rank
import tools.split_core_fringe_prediction
print('import smoke OK')
PY
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests \
  2>&1 | tee logs/stream4d_v6_package_unit_tests_r3.log
```

结果：

```text
py_compile exit=0
import smoke OK
unit tests: Ran 12 tests in 1.389s, OK
```

打包命令摘要：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
PKG=stream4d_v6_code_review_packet_20260608_2136.zip
LIST=stream4d_v6_code_review_packet_20260608_2136_filelist.txt
SHA=stream4d_v6_code_review_packet_20260608_2136.sha256
# file list includes v6 docs, Stream3D source/tests/splits/configs, key evaluation txt,
# object_competition/core_fringe/typed summaries, audit json/md/png, logs, git status/diff.
zip -q "$PKG" -@ < "$TMP_LIST"
sha256sum "$PKG" | tee "$SHA"
```

打包产物：

```text
stream4d_v6_code_review_packet_20260608_2136.zip
stream4d_v6_code_review_packet_20260608_2136.sha256
stream4d_v6_code_review_packet_20260608_2136_filelist.txt
filelist entries = 427
```

说明：

```text
该包包含最新 g098/g101/m650/m670 no-group/top-k 的 evaluation、summary、metric integrity 和命令日志。
不包含 raw ScanNet 数据、prediction npz 或 checkpoint。
```

## 13. continuation：ownership-aware WTA 反事实

### 13.1 执行原因

no-group/top-k 说明 ranking/NMS 能在 #pred<=300 下小幅超过 v4.1，但仍不过 gate。计划推荐方向是 object formation / ownership-aware support，因此继续利用已有 `tools.split_core_fringe_prediction` 的 `assignment-mode=wta` 做最小反事实：

```text
如果把 support conflict point 分配给单一 owner，是否能提升 AP/AP50 或减少 over-merge？
```

同样说明：

```text
本节仍是 CPU 后处理 + evaluator + metric integrity，不调用 GPU。
```

### 13.2 执行命令

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
for PRIOR in score_over_sqrt_area score large_area; do
  CONFIG=stream4d_v6_e5_probe5_v41_wta_${PRIOR}_core005_preserve
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.split_core_fringe_prediction \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --input-config "$INPUT" \
    --support-config "$INPUT" \
    --output-config "$CONFIG" \
    --tmp-policy support \
    --assignment-mode wta \
    --wta-priority "$PRIOR" \
    --min-support-area 1 \
    --min-core-points 5 \
    --min-core-ratio 0.05 \
    --low-mode none \
    --core-score -1 \
    --summary-root outputs/core_fringe_v6 \
    2>&1 | tee logs/${CONFIG}_export.log
  /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m evaluation.evaluate \
    --pred_path data/prediction/${CONFIG}_class_agnostic \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
    --tmp_root data/TMP \
    --tmp_config ${CONFIG} \
    --no_class \
    --require-manifest \
    2>&1 | tee logs/${CONFIG}_eval.log
done
```

best WTA metric integrity：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIG=stream4d_v6_e5_probe5_v41_wta_score_core005_preserve
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --configs "$CONFIG" \
  --seq-list splits/scannet_v6_probe5.txt \
  --backbone Cropformer \
  --output outputs/audit/v6_metric_integrity_${CONFIG}.md \
  --require-manifest \
  2>&1 | tee logs/v6_metric_integrity_${CONFIG}.log
```

### 13.3 结果

| Config | AP | AP50 | AP25 | #pred | union % | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e5_probe5_v41_wta_score_over_sqrt_area_core005_preserve` | `0.2504209406842788` | `0.4830729675575163` | `0.6671145539377294` | not_audited | not_audited | not_run |
| `stream4d_v6_e5_probe5_v41_wta_score_core005_preserve` | `0.2713964926079926` | `0.48931285917848677` | `0.6812376191185117` | `103.8` | `4.1271%` | pass |
| `stream4d_v6_e5_probe5_v41_wta_large_area_core005_preserve` | `0.1888086582238025` | `0.36780109986986276` | `0.5223900140738225` | not_audited | not_audited | not_run |

best WTA summary:

```text
mean_num_instances_before = 415.6
mean_num_instances_after = 103.8
mean_output_support_conflict_ratio = 0.0
mean_mean_core_ratio = 0.46751355660911537
mean_mean_conflict_ratio = 0.5324864433908847
metric integrity phase0_pass = true
uses_gt = false
```

结论：

```text
1. WTA 确实消除了 output support conflict，但 #pred 被压到 103.8，AP/AP50/AP25 都低于 no-group/top-k 和 v4.1。
2. 这说明简单 point-level owner assignment 会丢掉大量必要 support，不是当前解法。
3. 计划里的 split-aware graph 不能只靠 WTA；需要先形成更高质量候选/component，再做 split/merge lifecycle。
4. 当前已有后处理路线（grouping、small rescue、no-group/top-k、WTA）已经给出足够证据：继续微调后处理不太可能跨过 AP>=0.32/AP50>=0.53/AP25>=0.70。
```

## 14. continuation：pure Stream4D candidate fusion 反事实

### 14.1 执行原因

用户继续要求未达标则推进。上一轮 no-group/top-k 的最好结果是：

```text
stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve
AP/AP50/AP25 = 0.28483247256897415 / 0.5039622641509434 / 0.6719147248897401

stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve
AP/AP50/AP25 = 0.28204965102921625 / 0.4982312805326885 / 0.6912642582412274
```

目标没有达成，因此继续检查一个更具体的问题：

```text
v4.1 高分候选能否和 v6 local/core proposal 互补，在不使用 Stream3D/scannet support 的情况下提升 AP/AP50/AP25？
```

本节仍是 CPU 后处理 + evaluator，不调用 GPU。

### 14.2 E6 concat fusion

命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
PRIMARY=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5

# SEC_TAG=localprop:
SECONDARY=stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075
# SEC_TAG=coreonly:
SECONDARY=stream4d_v6_localprop_probe5_coreonly_scoreone_ioc075

FUSED=stream4d_v6_e6_probe5_fuse_v41_${SEC_TAG}_concat_p1_s005
$PY -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --primary-config "$PRIMARY" \
  --secondary-config "$SECONDARY" \
  --output-config "$FUSED" \
  --preserve-primary-score \
  --secondary-score 0.05 \
  --fusion-mode concatenate \
  --drop-secondary-iou-threshold 0.0 \
  --summary-root outputs/fusion_v6 \
  2>&1 | tee logs/${FUSED}_fuse.log
```

随后分别用 `score_unique_compact` 和 `compact_only` 做 object competition。第一轮使用 fused union 作为 score support：

```bash
CONFIG=stream4d_v6_e6_probe5_fuse_v41_${SEC_TAG}_m300_g101_${MODE_TAG}_preserve
$PY -u -m tools.object_competition_rank \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config "$FUSED" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$FUSED" \
  --quality-mode "$MODE" \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 1.01 \
  --min-support-area 1 \
  --max-instances 300 \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve 0 \
  --summary-root outputs/object_competition_v6 \
  2>&1 | tee logs/${CONFIG}_export.log

$PY -u -m evaluation.evaluate \
  --pred_path data/prediction/${CONFIG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CONFIG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config ${CONFIG} \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CONFIG}_eval.log
```

metric integrity：

```bash
CONFIGS=stream4d_v6_e6_probe5_fuse_v41_localprop_m300_g101_scoreunique_preserve,stream4d_v6_e6_probe5_fuse_v41_localprop_m300_g101_compact_preserve,stream4d_v6_e6_probe5_fuse_v41_coreonly_m300_g101_scoreunique_preserve,stream4d_v6_e6_probe5_fuse_v41_coreonly_m300_g101_compact_preserve
$PY -u -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root /mnt/data/users/chengshun.wang/pjs/orig_stream3d/Code_Stream3D \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --backbone Cropformer \
  --output outputs/audit/v6_metric_integrity_stream4d_v6_e6_fusion_probe5.md \
  --require-manifest \
  2>&1 | tee logs/v6_metric_integrity_stream4d_v6_e6_fusion_probe5.log
```

结果：

| Config | AP | AP50 | AP25 | #pred | union % | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e6_probe5_fuse_v41_localprop_m300_g101_scoreunique_preserve` | `0.010218108812631676` | `0.04115012746934571` | `0.17617853353466817` | `287.8` | `25.5904%` | pass |
| `stream4d_v6_e6_probe5_fuse_v41_localprop_m300_g101_compact_preserve` | `0.011710341031375226` | `0.044383533492114685` | `0.18216000373994912` | `287.8` | `25.6962%` | pass |
| `stream4d_v6_e6_probe5_fuse_v41_coreonly_m300_g101_scoreunique_preserve` | `0.0405954551412587` | `0.1204511962249542` | `0.33605775521707504` | `256.4` | `16.0945%` | pass |
| `stream4d_v6_e6_probe5_fuse_v41_coreonly_m300_g101_compact_preserve` | `0.0377836588858184` | `0.11856998329568297` | `0.3440586154109523` | `256.4` | `14.7493%` | pass |

fusion summary：

```text
localprop concat:
  mean_primary_union_ratio = 0.043406033305299525
  mean_secondary_union_ratio = 0.35797330322212584
  mean_output_union_ratio = 0.36263520796432436

coreonly concat:
  mean_primary_union_ratio = 0.043406033305299525
  mean_secondary_union_ratio = 0.1313102002615043
  mean_output_union_ratio = 0.16420905875918435
```

### 14.3 E6 primary-support 保守重跑

直接用 fused union 做 scoring support 后 AP 崩，因此重跑保守变体：候选仍为 fused candidates，但 score/pre_points support 固定为 v4.1 Stream4D primary support。

命令差异：

```bash
CONFIG=stream4d_v6_e6_probe5_fuse_v41_${SEC_TAG}_m300_g101_${MODE_TAG}_primarysupp_preserve
$PY -u -m tools.object_competition_rank \
  --input-config "$FUSED" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$PRIMARY" \
  ... \
  2>&1 | tee logs/${CONFIG}_export.log
```

metric integrity：

```text
outputs/audit/v6_metric_integrity_stream4d_v6_e6_fusion_primarysupp_probe5.md
outputs/audit/v6_metric_integrity_stream4d_v6_e6_fusion_primarysupp_probe5.json
phase0_pass = true
```

结果：

| Config | AP | AP50 | AP25 | #pred | union % | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e6_probe5_fuse_v41_localprop_m300_g101_scoreunique_primarysupp_preserve` | `0.27979015467712104` | `0.48746989600281426` | `0.6968854618036644` | `263.2` | `13.8754%` | pass |
| `stream4d_v6_e6_probe5_fuse_v41_localprop_m300_g101_compact_primarysupp_preserve` | `0.22856043273657833` | `0.3864847517795402` | `0.5071203008901843` | `263.2` | `16.4226%` | pass |
| `stream4d_v6_e6_probe5_fuse_v41_coreonly_m300_g101_scoreunique_primarysupp_preserve` | `0.27801811145138666` | `0.4910369268480345` | `0.6889750798092922` | `238.2` | `7.9644%` | pass |
| `stream4d_v6_e6_probe5_fuse_v41_coreonly_m300_g101_compact_primarysupp_preserve` | `0.2754073604133401` | `0.45256450717681596` | `0.5913662748158626` | `238.2` | `7.2392%` | pass |

审计注意：

```text
primary-support 版本 phase0_pass=true 且 uses_gt=false。
但 metric integrity 中 union subset policy 多数为 inconsistent_union_not_subset。
因此这类结果只能作为保守诊断，不能当作 clean final policy。
```

## 15. continuation：select_secondary 分数保留修复与替换式融合

### 15.1 工具修复

文件：

```text
Stream3D/tools/fuse_prediction_configs.py
```

问题：

```text
select_secondary 分支中 --preserve-primary-score 会把 primary_score 改成 -1.0，
但 _select_variant_masks 直接把 -1.0 写入 selected_scores，
没有真正保留 primary 原始分数。
```

修改：

```text
1. _select_variant_masks 新增 primary_scores 参数。
2. primary/secondary empty case 使用 _score_array_or_preserve。
3. selected_scores 初始化为 primary 原始分数或指定常量。
4. unmatched secondary 在 secondary_score<0 时保留 secondary 原始分数。
```

验证：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/fuse_prediction_configs.py
```

结果：

```text
exit=0
```

### 15.2 E7 select_secondary 实验

命令模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
PRIMARY=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
FUSED=stream4d_v6_e7_probe5_select_v41_${SEC_TAG}_ioc050_exp150

$PY -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --primary-config "$PRIMARY" \
  --secondary-config "$SECONDARY" \
  --output-config "$FUSED" \
  --preserve-primary-score \
  --secondary-score 0.05 \
  --fusion-mode select_secondary \
  --select-min-primary-ioc 0.50 \
  --select-max-expansion 1.50 \
  --summary-root outputs/fusion_v6 \
  2>&1 | tee logs/${FUSED}_fuse.log

CONFIG=stream4d_v6_e7_probe5_select_v41_${SEC_TAG}_m650_g101_${MODE_TAG}_primarysupp_preserve
$PY -u -m tools.object_competition_rank \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config "$FUSED" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$PRIMARY" \
  --quality-mode "$MODE" \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 1.01 \
  --min-support-area 1 \
  --max-instances 650 \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve 0 \
  --summary-root outputs/object_competition_v6 \
  2>&1 | tee logs/${CONFIG}_export.log
```

metric integrity：

```text
outputs/audit/v6_metric_integrity_stream4d_v6_e7_select_probe5.md
outputs/audit/v6_metric_integrity_stream4d_v6_e7_select_probe5.json
phase0_pass = true
```

结果：

| Config | AP | AP50 | AP25 | #pred | output union | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e7_probe5_select_v41_localprop_m650_g101_scoreunique_primarysupp_preserve` | `0.27360533351987215` | `0.48736235779772474` | `0.6857581778604103` | `295.6` | `9084.2` | pass |
| `stream4d_v6_e7_probe5_select_v41_localprop_m650_g101_compact_primarysupp_preserve` | `0.2762186735105981` | `0.4928908044100521` | `0.6665564862285539` | `295.6` | `8919.6` | pass |
| `stream4d_v6_e7_probe5_select_v41_coreonly_m650_g101_scoreunique_primarysupp_preserve` | `0.27348684186262195` | `0.4982312805326885` | `0.6912642582412274` | `295.6` | `9051.2` | pass |
| `stream4d_v6_e7_probe5_select_v41_coreonly_m650_g101_compact_primarysupp_preserve` | `0.2770380429028228` | `0.5058237524138866` | `0.6746102478165856` | `295.6` | `8869.6` | pass |

fusion summary：

```text
localprop select:
  mean_output_instances = 415.6
  mean_output_union_ratio = 0.04373242088002579

coreonly select:
  mean_output_instances = 415.6
  mean_output_union_ratio = 0.04362121829325931
```

## 16. continuation：dual-quality compact/scoreunique fusion

### 16.1 执行原因

已有两个互补正信号：

```text
m650 compact_only: AP/AP50 较高，AP25 退化。
m650 scoreunique: AP25 较高，AP/AP50 较低。
```

因此尝试 compact 作为 primary、scoreunique 作为 secondary，删除几乎重复的 secondary，再竞争筛选到 300。

### 16.2 命令

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
PRIMARY=stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve
SECONDARY=stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve
FUSED=stream4d_v6_e8_probe5_dualquality_compact_primary_scoreunique_secondary_drop098

$PY -u -m tools.fuse_prediction_configs \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --primary-config "$PRIMARY" \
  --secondary-config "$SECONDARY" \
  --output-config "$FUSED" \
  --preserve-primary-score \
  --preserve-secondary-score \
  --fusion-mode concatenate \
  --drop-secondary-iou-threshold 0.98 \
  --drop-secondary-overlap-mode min_ioc \
  --summary-root outputs/fusion_v6 \
  2>&1 | tee logs/${FUSED}_fuse.log

CONFIG=stream4d_v6_e8_probe5_dualquality_m300_g085_${MODE_TAG}_preserve
$PY -u -m tools.object_competition_rank \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --input-config "$FUSED" \
  --output-config "$CONFIG" \
  --score-pre-points-config "$FUSED" \
  --quality-mode "$MODE" \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 0.85 \
  --min-support-area 1 \
  --max-instances 300 \
  --preserve-original-score \
  --tmp-policy score_support \
  --summary-root outputs/object_competition_v6 \
  2>&1 | tee logs/${CONFIG}_export.log
```

metric integrity：

```text
outputs/audit/v6_metric_integrity_stream4d_v6_e8_dualquality_probe5.md
outputs/audit/v6_metric_integrity_stream4d_v6_e8_dualquality_probe5.json
phase0_pass = true
```

结果：

| Config | AP | AP50 | AP25 | #pred | output union | phase0 |
|---|---:|---:|---:|---:|---:|---|
| `stream4d_v6_e8_probe5_dualquality_m300_g085_scoreunique_preserve` | `0.27031150512181795` | `0.4332201554423777` | `0.5934316489872046` | `178.0` | `7016.6` | pass |
| `stream4d_v6_e8_probe5_dualquality_m300_g085_compact_preserve` | `0.1993965865242845` | `0.3115810996763378` | `0.43117816927340735` | `178.0` | `6420.0` | pass |

fusion summary：

```text
mean_primary_instances = 295.6
mean_secondary_instances after drop = 13.6
mean_output_instances = 309.2
mean_secondary_skipped_by_iou = 282.0
mean_output_union_ratio = 0.0426531859248895
```

### 16.3 当前执行结论

```text
E6 direct concat: secondary union 噪声主导，AP 崩到 0.01-0.04。
E6 primary-support concat: 恢复到 0.278-0.280，但没有超过已有 best，且 union policy 不干净。
E7 select_secondary: 替换 primary 形状后仍低于已有 best。
E8 dual-quality: compact/scoreunique 两个正信号没有互补成功，反而被 0.85 grouping 过度压缩。
```

当前未达成：

```text
AP >= 0.32: false
AP50 >= 0.53: false
AP25 >= 0.70: false
#pred <= 300: 部分 pass
union target >= 0.94: false
```

## 17. r4 验证与审计包更新

### 17.1 打包前验证

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile evaluation/evaluate.py stream4d/*.py tools/*.py tests/*.py \
  2>&1 | tee logs/stream4d_v6_package_py_compile_r4.log

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY' \
  2>&1 | tee logs/stream4d_v6_package_import_smoke_r4.log
import stream4d.run_scannet
import stream4d.object_memory_v2
import stream4d.export_scannet
import stream4d.reliable_densifier
import tools.verify_stream4d_metric_integrity
import tools.scan_reportable_configs
import tools.object_competition_rank
import tools.split_core_fringe_prediction
import tools.fuse_prediction_configs
print('import smoke OK')
PY

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests \
  2>&1 | tee logs/stream4d_v6_package_unit_tests_r4.log
```

结果：

```text
py_compile exit=0
import smoke OK
unit tests: Ran 12 tests in 1.396s, OK
```

### 17.2 审计包

命令摘要：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
git status --short > Stream3D/outputs/audit/git_status_v6_r4.txt
git diff -- > Stream3D/outputs/audit/git_diff_v6_r4.patch
# filelist includes v6 docs, Stream3D source/tests/splits/configs, key evaluation txt,
# fusion/object_competition/core_fringe/typed summaries, audit json/md/png, logs, git status/diff.
zip -q stream4d_v6_code_review_packet_20260608_2207.zip -@ < stream4d_v6_code_review_packet_20260608_2207_filelist.txt
sha256sum stream4d_v6_code_review_packet_20260608_2207.zip | tee stream4d_v6_code_review_packet_20260608_2207.sha256
```

结果：

```text
stream4d_v6_code_review_packet_20260608_2207.zip
stream4d_v6_code_review_packet_20260608_2207.sha256
stream4d_v6_code_review_packet_20260608_2207_filelist.txt
filelist entries = 492
sha256 = 4705daf7f30c75e8a815cf7efa13865438b9f5d9cb7386a5bfee9e5bb2aa3099
```

说明：

```text
包内不含 raw ScanNet 数据、checkpoint 或大型 prediction npz。
包内包含 E6/E7/E8 最新命令日志、evaluation txt、summary json、metric integrity 报告，以及 r4 验证日志。
由于记录本节本身会再次修改文档，最终审计包名和 sha256 以最终回复及同名 .sha256 文件为准。
```
