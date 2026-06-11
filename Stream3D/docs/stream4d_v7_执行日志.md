# Stream4D v7 深度审计与实验执行日志

日期：2026-06-08 至 2026-06-09（Asia/Singapore）  
工作目录：`/mnt/data/users/chengshun.wang/pjs/LoGeR`  
主要实验目录：`Stream3D/`  
计划文件：`docs/stream4d_v7_deep_audit_gap_method_plan_for_codex.md`

本日志只记录真实执行过的命令、产物和 blocker 处理。没有落盘的指标不补写。

## 环境

```text
可用 Python：/mnt/data/users/chengshun.wang/miniconda3/bin/python
系统 python3：/usr/bin/python3
conda Python：有 numpy/scipy/open3d/cv2，无 torch，无 pytest
最终单测入口：python -m unittest -v tests.test_stream4d_protocol_fixes
```

## 代码修改记录

| 文件 | 修改 | 原因 | 验证 |
|---|---|---|---|
| `Stream3D/stream4d/reliable_densifier.py` | 新增 `recompute_record_scores()`，`apply_wta_to_records()` 在 WTA 后重算 `area_score/score/reliability/dense_quality/selection_quality` | 修复计划 P1：WTA 改变 `point_ids` 后 score 不能沿用旧面积 | `stream4d_v7_unit_tests_final_unittest.log` 13 tests pass |
| `Stream3D/tests/test_stream4d_protocol_fixes.py` | 新增 WTA 后面积相关 score 重算测试 | 防止 WTA score-mode 审计缺口复发 | `test_reliable_densifier_wta_recomputes_area_sensitive_scores` pass |
| `Stream3D/evaluation/evaluate.py` | torch import 改为 optional；无 torch 时用 numpy CPU fallback；AP core 函数未改 | 当前环境无 torch，初始 unit test/import 被 blocker 卡住；同时保留 AP core hash 一致 | `v7_metric_integrity_carrier_probe5.json` 中 `all_ap_core_equal=True` |
| `Stream3D/tools/evaluate_cross_prepoints.py` | 写 diagnostic manifest，新增 `--require-manifest` 和 `--allow-diagnostic-eval` | gap matrix 是 diagnostic cross-support eval，必须显式 guard | `stream4d_v7_gap_matrix.log` |
| `Stream3D/tools/make_union_prepoints_config.py` | 新增 union support 生成工具 | Phase A 的 S6 union support | `stream4d_v7_union_prepoints.log` |
| `Stream3D/tools/diagnose_prediction_quality.py` | 新增 prediction quality diagnostic | Phase A 证据链：mask overlap、GT diagnostic、duplicate、missed GT | `stream4d_v7_prediction_quality_diagnostic.log` |
| `Stream3D/tools/export_d4rt_geometry_degradation_v7.py` | 新增 D4RT geometry degradation diagnostic/export | Phase B：D4RT geometry replacement 与 Sim3 residual | `stream4d_v7_d4rt_geometry_*.log` |
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | 新增 carrier-tracklet object formation，支持 C2/C3/C4 | Phase C：用 D4RT carrier co-membership 替代 mask-node point overlap | `stream4d_v7_carrier_*.log` |
| `Stream3D/tools/summarize_v7_gap_matrix.py` | 新增 gap matrix 汇总工具 | 生成 JSON/CSV/MD/PNG 总表 | `stream4d_v7_gap_matrix_summary.log` |
| `Stream3D/tools/diagnose_trackbucket_suppression_v7.py` | 新增 track-bucket suppression GT diagnostic；落盘 `diagnostic_only=True, uses_gt=True` | 收尾检查 C11/C12/C13 删除的 candidates 是否为有用 GT 覆盖；不进入 method result | `stream4d_v7_c14_trackbucket_diagnostic_py_compile.log` pass |
| `Stream3D/tools/export_carrier_tracklet_graph_v7.py` | `object_dict.npy` 写入 processed 目录失败时 fallback 到 `outputs/v7_carrier_tracklet_graph/object_dicts/...`，并在 summary 写入 fallback path/error | 当前 sandbox 中 `data/scannet/processed/.../object/<config>` 只读；prediction/TMP 可写但 object_dict 审计文件不能丢 | `stream4d_v7_c14_object_dict_fallback_py_compile.log` pass；C14/C15 summaries 写出 fallback |
| `Stream3D/tools/verify_stream4d_metric_integrity.py` | object_dict alignment 优先查 processed 路径；缺失时读取 exporter summary 的 `object_dict_write_path` fallback | 让 C14/C15 的 object_dict/prediction alignment 能被审计，不因只读 fallback 路径显示 skipped | `stream4d_v7_r6_metric_integrity_fallback_py_compile.log` pass；C14/C15 r6 alignment mean/min=1.0/1.0 |

## 初始验证与 blocker 修复

初始语法检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m py_compile ...
```

日志：

```text
logs/stream4d_v7_py_compile_initial.log
```

初始单测失败：

```text
logs/stream4d_v7_unit_tests_initial.log
ModuleNotFoundError: No module named 'torch'
```

处理：

```text
按计划 P1 的 CPU smoke / import 方向修复 evaluator import 阶段 torch 依赖。
AP core 函数 evaluate_matches / compute_averages 未改。
```

修复后验证：

```text
logs/stream4d_v7_py_compile_r2.log
logs/stream4d_v7_unit_tests_r2.log
logs/stream4d_v7_py_compile_r3.log
logs/stream4d_v7_unit_tests_r3.log
logs/stream4d_v7_import_smoke.log
```

## Phase A：gap matrix

配置：

```text
Prediction rows:
P0=scannet
P1=stream4d_32f_self_probe5
P2=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
P3=stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve
P4=stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve
P5=stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075
P6=stream4d_v6_typedv3_probe5_c2_trackonly_ioc060

Support columns:
S0=own
S1=stream4d_32f_self_probe5
S2=P2 support
S3=P3 support
S4=P4 support
S5=scannet
S6=stream4d_v7_union_s1_s3_s4_probe5
```

生成 union support：

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.make_union_prepoints_config \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v7_union_s1_s3_s4_probe5 \
  --input-configs stream4d_32f_self_probe5,stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve,stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve
```

日志与产物：

```text
logs/stream4d_v7_union_prepoints.log
outputs/audit/v7_gap_matrix/stream4d_v7_union_s1_s3_s4_probe5.{json,csv,md}
```

运行 cross-prepoints matrix：

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.evaluate_cross_prepoints \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config <P> \
  --source-pre-points-config <P_source_support> \
  --pre-points-config <S> \
  --output-config stream4d_v7_gap_p<i>_on_s<j> \
  --dataset scannet \
  --no-class \
  --output-file data/evaluation/scannet/stream4d_v7_gap_p<i>_on_s<j>_class_agnostic.txt \
  --audit-root outputs/audit/v7_gap_matrix \
  --require-manifest \
  --allow-diagnostic-eval
```

日志与产物：

```text
logs/stream4d_v7_gap_matrix.log
outputs/audit/v7_gap_matrix/cross_prepoints_audit.{json,csv,md}
outputs/audit/v7_gap_matrix/cross_prepoints/stream4d_v7_gap_p*_on_s*_summary.json
data/evaluation/scannet/stream4d_v7_gap_p*_on_s*_class_agnostic.txt
```

汇总：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.summarize_v7_gap_matrix \
  --input outputs/audit/v7_gap_matrix/cross_prepoints_audit.json \
  --output-prefix outputs/audit/v7_gap_matrix
```

日志与产物：

```text
logs/stream4d_v7_gap_matrix_summary.log
outputs/audit/v7_gap_matrix.{json,csv,md,png}
```

Prediction quality diagnostic：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.diagnose_prediction_quality \
  --configs scannet,stream4d_32f_self_probe5,stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5,stream4d_v6_e4_probe5_objcomp_m650_g101_compact_only_preserve,stream4d_v6_e4_probe5_objcomp_m650_g101_scoreunique_preserve,stream4d_v6_localprop_96f_probe5_min2_bestframe_score_one_ioc075,stream4d_v6_typedv3_probe5_c2_trackonly_ioc060 \
  --seq-list splits/scannet_v6_probe5.txt \
  --gt-root data/scannet/gt \
  --output outputs/audit/v7_prediction_quality_diagnostic.md
```

日志与产物：

```text
logs/stream4d_v7_prediction_quality_diagnostic.log
outputs/audit/v7_prediction_quality_diagnostic.{json,csv,md}
```

## Phase B：D4RT geometry degradation

Sim3 residual diagnostic：

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_d4rt_geometry_degradation_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_d4rt_geometry_degradation.md \
  --skip-export
```

日志与产物：

```text
logs/stream4d_v7_d4rt_geometry_residual.log
outputs/audit/v7_d4rt_geometry_degradation.{json,csv,md}
```

G1/G3 segmentation diagnostic：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_d4rt_geometry_degradation_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --g1-output-config stream4d_v7_g1_d4rt_geometry_probe5 \
  --g3-output-config stream4d_v7_g3_d4rt_geometry_norm_probe5
```

日志：

```text
logs/stream4d_v7_d4rt_geometry_segmentation.log
```

Blocker 处理：

```text
默认 min_points=30 时 G1/G3 导出 0 objects，eval 为 nan。
按计划修复方向尝试增大 mesh NN radius 到 0.50：
  logs/stream4d_v7_d4rt_geometry_segmentation_r050.log
仍为 0 objects。
继续做极限 diagnostic：mesh_nn_radius=0.50, min_points=1：
  logs/stream4d_v7_d4rt_geometry_segmentation_r050_min1.log
该设置导出大量碎片，但 AP/AP50/AP25=0.0/0.0/0.0。
```

相关 eval 文件：

```text
data/evaluation/scannet/stream4d_v7_g1_d4rt_geometry_probe5_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_g3_d4rt_geometry_norm_probe5_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_g1_d4rt_geometry_probe5_r050_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_g3_d4rt_geometry_norm_probe5_r050_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_g1_d4rt_geometry_probe5_r050_min1_class_agnostic.txt
```

## Phase C：carrier-tracklet object formation

Scene0050 smoke：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v7_c2_carrier_core_scene0050 \
  --support-mode core
```

日志：

```text
logs/stream4d_v7_carrier_c2_scene0050_smoke.log
logs/stream4d_v7_carrier_c3_c4_scene0050_smoke.log
logs/stream4d_v7_carrier_scene0050_eval.log
```

Full probe5 C2/C3/C4：

```bash
cd Stream3D
for scene in $(cat splits/scannet_v6_probe5.txt); do
  /mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
    --debug-root outputs/stream4d_v5_cache_96f_probe5 \
    --seq-name "$scene" \
    --output-config stream4d_v7_c2_carrier_core_probe5 \
    --support-mode core
done

for scene in $(cat splits/scannet_v6_probe5.txt); do
  /mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
    --debug-root outputs/stream4d_v5_cache_96f_probe5 \
    --seq-name "$scene" \
    --output-config stream4d_v7_c3_carrier_corefringe_probe5 \
    --support-mode core_fringe
done

for scene in $(cat splits/scannet_v6_probe5.txt); do
  /mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
    --debug-root outputs/stream4d_v5_cache_96f_probe5 \
    --seq-name "$scene" \
    --output-config stream4d_v7_c4_carrier_corefringe_wta_probe5 \
    --support-mode core_fringe_wta
done
```

日志与 summary：

```text
logs/stream4d_v7_carrier_c2_probe5.log
logs/stream4d_v7_carrier_c3_probe5.log
logs/stream4d_v7_carrier_c4_probe5.log
outputs/v7_carrier_tracklet_graph/stream4d_v7_c*_carrier_*_probe5_scene*_summary.json
```

Evaluator：

```bash
for config in \
  stream4d_v7_c2_carrier_core_probe5 \
  stream4d_v7_c3_carrier_corefringe_probe5 \
  stream4d_v7_c4_carrier_corefringe_wta_probe5; do
  /mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
    --pred_path "data/prediction/${config}_class_agnostic" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${config}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$config" \
    --no_class \
    --require-manifest
done
```

日志与 eval：

```text
logs/stream4d_v7_carrier_probe5_eval.log
data/evaluation/scannet/stream4d_v7_c2_carrier_core_probe5_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_c3_carrier_corefringe_probe5_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_c4_carrier_corefringe_wta_probe5_class_agnostic.txt
```

## Phase D：Dynamic Replica 环境检查

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.check_dynamic_replica_env \
  --root data/dynamic-replica/v2 \
  --split valid \
  --output outputs/audit/dynamic_replica_env_v7.md
```

日志与产物：

```text
logs/stream4d_v7_dynamic_replica_env.log
outputs/audit/dynamic_replica_env_v7.{json,md}
```

结果边界：

```text
root_exists=False
split_exists=False
usable_scenes=0
不能报告 Dynamic Replica 官方指标。
```

## 最终验证

Metric integrity：

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs stream4d_v7_c2_carrier_core_probe5,stream4d_v7_c3_carrier_corefringe_probe5,stream4d_v7_c4_carrier_corefringe_wta_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_carrier_probe5.md \
  --require-manifest
```

结果：

```text
logs/stream4d_v7_metric_integrity_carrier_probe5.log
phase0_pass=True
```

Reportable config scan：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v7_c2_carrier_core_probe5,stream4d_v7_c3_carrier_corefringe_probe5,stream4d_v7_c4_carrier_corefringe_wta_probe5 \
  --output outputs/audit/v7_reportable_config_scan_carrier_probe5.md \
  --require-manifest
```

结果：

```text
logs/stream4d_v7_reportable_config_scan_carrier_probe5.log
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=3
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## 追加执行：C8 scene-level carrier track linking（2026-06-09）

背景：

```text
C7 pre-expansion mask ownership 是正向，但仍未建模跨 window 的 object-level temporal track。
C8 继续按复盘建议尝试 scene-level temporal object linking：
  先检查 D4RT carrier_id 是否跨 window 稳定；
  再用 shared carrier IDs 将跨 window records 合并成 scene-level tracks。
```

carrier_id overlap 检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR

/mnt/data/users/chengshun.wang/miniconda3/bin/python - <<'PY'
from pathlib import Path
import numpy as np
for scene in [x.strip() for x in Path('Stream3D/splits/scannet_v6_probe5.txt').read_text().splitlines() if x.strip()]:
    root=Path('Stream3D/outputs/stream4d_v5_cache_96f_probe5')/scene
    sets=[]
    for p in sorted(root.glob('carriers_window*.npz')):
        with np.load(p) as d:
            sets.append(set(np.asarray(d['carrier_id'], dtype=np.int64).tolist()))
    inter=[len(sets[i]&sets[i+1]) for i in range(len(sets)-1)]
    print(scene, 'windows', len(sets), 'adj_inter', inter, 'ids_total_unique', len(set().union(*sets)))
PY
```

结果：

```text
scene0050_00 windows 5 adj_inter [3736, 1872, 1304, 1808] ids_total_unique 12472
scene0011_00 windows 5 adj_inter [1416, 1328, 1720, 2064] ids_total_unique 9888
scene0030_00 windows 5 adj_inter [1656, 3176, 3864, 3824] ids_total_unique 15904
scene0081_01 windows 5 adj_inter [3544, 2648, 1008, 2360] ids_total_unique 15272
scene0591_00 windows 5 adj_inter [5240, 3512, 3736, 3088] ids_total_unique 23720
```

C8 代码修改：

```text
文件：
  Stream3D/tools/export_carrier_tracklet_graph_v7.py

新增 support modes：
  core_owned_track_fringe
  core_owned_track_fringe_wta
  core_owned_fringe_wta_posttrack

新增 scene-link 参数：
  --scene-link-min-shared-carriers
  --scene-link-min-overlap-ratio
  --scene-link-max-window-gap
  --scene-link-max-masks-per-frame

新增行为：
  pre-WTA track merge:
    core_owned_track_fringe_wta
    先合并跨 window records，再 WTA。

  post-WTA track merge:
    core_owned_fringe_wta_posttrack
    先按 C7 做 WTA，再合并跨 window records。

新增 diagnostics：
  scene_link_candidate_pairs_raw
  scene_link_candidate_pairs
  scene_link_accepted_pairs
  scene_link_rejected_same_group
  scene_link_rejected_frame_mask_conflict
  scene_link_input_records
  scene_link_output_records
  scene_link_merged_groups
  scene_link_max_group_size
  scene_link_mean_group_size
```

语法和 import smoke：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m py_compile \
  tools/export_carrier_tracklet_graph_v7.py \
  2>&1 | tee logs/stream4d_v7_c8_py_compile.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python - <<'PY' \
  2>&1 | tee logs/stream4d_v7_c8_import_smoke.log
import importlib
importlib.import_module('tools.export_carrier_tracklet_graph_v7')
print('tools.export_carrier_tracklet_graph_v7 OK')
PY
```

结果：

```text
logs/stream4d_v7_c8_py_compile.log
logs/stream4d_v7_c8_import_smoke.log
均 pass。
```

### C8 / C8S / C8P scene0050 smoke

C8 pre-WTA default：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v7_c8_track_owned_wta_scene0050 \
  --support-mode core_owned_track_fringe_wta \
  --min-shared-frames 3 \
  --min-positive-ratio 0.65 \
  --max-pair-distance-variance 0.005 \
  --max-component-carriers 180 \
  --min-mask-carriers 6 \
  --min-frame-mask-ratio 0.65 \
  --scene-link-min-shared-carriers 10 \
  --scene-link-min-overlap-ratio 0.15 \
  --scene-link-max-window-gap 1 \
  --scene-link-max-masks-per-frame 1 \
  2>&1 | tee logs/stream4d_v7_c8_track_owned_wta_scene0050_smoke.log
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c8_track_owned_wta_scene0050_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c8_track_owned_wta_scene0050_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c8_track_owned_wta_scene0050 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c8_track_owned_wta_scene0050_eval.log
```

C8S pre-WTA strict：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v7_c8s_track_owned_wta_scene0050 \
  --support-mode core_owned_track_fringe_wta \
  --min-shared-frames 3 \
  --min-positive-ratio 0.65 \
  --max-pair-distance-variance 0.005 \
  --max-component-carriers 180 \
  --min-mask-carriers 6 \
  --min-frame-mask-ratio 0.65 \
  --scene-link-min-shared-carriers 32 \
  --scene-link-min-overlap-ratio 0.50 \
  --scene-link-max-window-gap 1 \
  --scene-link-max-masks-per-frame 1 \
  2>&1 | tee logs/stream4d_v7_c8s_track_owned_wta_scene0050_smoke.log
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c8s_track_owned_wta_scene0050_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c8s_track_owned_wta_scene0050_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c8s_track_owned_wta_scene0050 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c8s_track_owned_wta_scene0050_eval.log
```

C8P post-WTA default：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v7_c8p_posttrack_owned_wta_scene0050 \
  --support-mode core_owned_fringe_wta_posttrack \
  --min-shared-frames 3 \
  --min-positive-ratio 0.65 \
  --max-pair-distance-variance 0.005 \
  --max-component-carriers 180 \
  --min-mask-carriers 6 \
  --min-frame-mask-ratio 0.65 \
  --scene-link-min-shared-carriers 10 \
  --scene-link-min-overlap-ratio 0.15 \
  --scene-link-max-window-gap 1 \
  --scene-link-max-masks-per-frame 1 \
  2>&1 | tee logs/stream4d_v7_c8p_posttrack_owned_wta_scene0050_smoke.log
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c8p_posttrack_owned_wta_scene0050_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c8p_posttrack_owned_wta_scene0050_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c8p_posttrack_owned_wta_scene0050 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c8p_posttrack_owned_wta_scene0050_eval.log
```

### C8P full probe5

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
DEBUG_ROOT=outputs/stream4d_v5_cache_96f_probe5
SEQ_LIST=splits/scannet_v6_probe5.txt
LOG=logs/stream4d_v7_c8p_posttrack_owned_wta_probe5.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  printf '### C8P %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m tools.export_carrier_tracklet_graph_v7 \
    --debug-root "$DEBUG_ROOT" \
    --seq-name "$scene" \
    --output-config stream4d_v7_c8p_posttrack_owned_wta_probe5 \
    --support-mode core_owned_fringe_wta_posttrack \
    --min-shared-frames 3 \
    --min-positive-ratio 0.65 \
    --max-pair-distance-variance 0.005 \
    --max-component-carriers 180 \
    --min-mask-carriers 6 \
    --min-frame-mask-ratio 0.65 \
    --scene-link-min-shared-carriers 10 \
    --scene-link-min-overlap-ratio 0.15 \
    --scene-link-max-window-gap 1 \
    --scene-link-max-masks-per-frame 1 2>&1 | tee -a "$LOG"
done < "$SEQ_LIST"
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c8p_posttrack_owned_wta_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c8p_posttrack_owned_wta_probe5_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c8p_posttrack_owned_wta_probe5 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c8p_posttrack_owned_wta_probe5_eval.log
```

per-scene 评估：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
SRC_PRED=data/prediction/stream4d_v7_c8p_posttrack_owned_wta_probe5_class_agnostic
TMP_CONFIG=stream4d_v7_c8p_posttrack_owned_wta_probe5
LOG=logs/stream4d_v7_c8p_posttrack_owned_wta_probe5_per_scene_eval.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  SCENE_DIR="data/prediction/stream4d_v7_c8p_posttrack_owned_wta_probe5_${scene}_class_agnostic"
  mkdir -p "$SCENE_DIR"
  ln -sf "$(realpath "$SRC_PRED/config_manifest.json")" "$SCENE_DIR/config_manifest.json"
  ln -sf "$(realpath "$SRC_PRED/${scene}.npz")" "$SCENE_DIR/${scene}.npz"
  printf '### C8P per-scene eval %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m evaluation.evaluate \
    --pred_path "$SCENE_DIR" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/stream4d_v7_c8p_posttrack_owned_wta_probe5_${scene}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$TMP_CONFIG" \
    --no_class \
    --require-manifest 2>&1 | tee -a "$LOG"
done < splits/scannet_v6_probe5.txt
```

### C8P 验证与审计

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m py_compile \
  tools/export_carrier_tracklet_graph_v7.py \
  stream4d/reliable_densifier.py \
  tests/test_stream4d_protocol_fixes.py \
  2>&1 | tee logs/stream4d_v7_c8p_py_compile_final.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m unittest -v tests.test_stream4d_protocol_fixes \
  2>&1 | tee logs/stream4d_v7_c8p_unit_tests.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python - <<'PY' \
  2>&1 | tee logs/stream4d_v7_c8p_import_smoke_final.log
import importlib
for name in ['tools.export_carrier_tracklet_graph_v7','stream4d.reliable_densifier']:
    importlib.import_module(name)
    print(f'{name} OK')
PY
```

结果：

```text
logs/stream4d_v7_c8p_py_compile_final.log
exit code 0，日志为空为正常。

logs/stream4d_v7_c8p_import_smoke_final.log
tools.export_carrier_tracklet_graph_v7 OK
stream4d.reliable_densifier OK

logs/stream4d_v7_c8p_unit_tests.log
Ran 13 tests in 0.079s
OK
```

metric integrity：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs stream4d_v7_c8p_posttrack_owned_wta_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c8p_posttrack_owned_wta_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c8p_posttrack_owned_wta_probe5.log
```

结果：

```text
outputs/audit/v7_metric_integrity_c8p_posttrack_owned_wta_probe5.md
phase0_pass=True
evaluator_ap_core_equal_by_hash=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=1
num_suspicious_configs=0
alignment mean/min=1.000000/1.000000
```

reportable config scan：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v7_c8p_posttrack_owned_wta_probe5 \
  --output outputs/audit/v7_reportable_config_scan_c8p_posttrack_owned_wta_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c8p_posttrack_owned_wta_probe5.log
```

结果：

```text
outputs/audit/v7_reportable_config_scan_c8p_posttrack_owned_wta_probe5.json
num_configs=1
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=1
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

Final py_compile：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m py_compile \
  evaluation/evaluate.py \
  stream4d/reliable_densifier.py \
  tools/evaluate_cross_prepoints.py \
  tools/make_union_prepoints_config.py \
  tools/diagnose_prediction_quality.py \
  tools/export_carrier_tracklet_graph_v7.py \
  tools/export_d4rt_geometry_degradation_v7.py \
  tools/summarize_v7_gap_matrix.py \
  tools/check_dynamic_replica_env.py \
  tools/verify_stream4d_metric_integrity.py \
  tools/scan_reportable_configs.py \
  tests/test_stream4d_protocol_fixes.py
```

结果：

```text
logs/stream4d_v7_py_compile_final.log
exit code 0，日志为空为正常。
```

Final import smoke：

```text
logs/stream4d_v7_import_smoke_final.log
全部目标 module import OK。
```

Final unit test：

```text
logs/stream4d_v7_unit_tests_final.log
conda Python 无 pytest，pytest 入口失败：No module named pytest。

logs/stream4d_v7_unit_tests_final_unittest.log
python -m unittest -v tests.test_stream4d_protocol_fixes
Ran 13 tests in 0.081s
OK
```

## 待打包清单

```text
Stream3D/stream4d/*.py
Stream3D/tools/*.py
Stream3D/evaluation/*.py
Stream3D/tests/*.py
Stream3D/docs/stream4d_v7_*.md
Stream3D/logs/*v7*.log
Stream3D/outputs/audit/*v7*.json
Stream3D/outputs/audit/*v7*.md
Stream3D/outputs/audit/*v7*.csv
Stream3D/outputs/audit/*v7*.png
Stream3D/outputs/v7_carrier_tracklet_graph/*v7*.json
Stream3D/data/evaluation/scannet/*v7*.txt
Stream3D/data/prediction/*v7*/config_manifest.json
git diff/status
sha256sum sidecar
```

## 追加执行：C5/C6 carrier formation 修复尝试（2026-06-09）

背景：

```text
v7 初版 C4 仍远低于 Stream3D/v6 object-component baseline。
按复盘里的下一步方向继续尝试：
1. 提前做 carrier trajectory consistency split。
2. 在 component merge 前加强 per-frame mask purity / exclusivity。
3. 保持不使用 GT、不使用 mask-node point overlap。
```

代码修改：

```text
文件：
  Stream3D/tools/export_carrier_tracklet_graph_v7.py

新增 support mode：
  seeded_fringe
  seeded_fringe_wta

新增 seeded-mask 参数：
  --seeded-max-masks-per-object
  --seeded-mask-min-relative-coverage
  --seeded-mask-sample-stride
  --seeded-mask-max-pixels
  --seeded-boundary-erosion
  --seeded-small-mask-area
  --seeded-distance-px
  --seeded-min-seed-pixels
  --seeded-seed-keep-mode
  --seeded-seed-min-support-views
  --seeded-mask-selection-mode

实现意图：
  seeded_fringe 只从 carrier core seeds 出发，在 RGB-D mask 内回投 seed-connected / seed-near 支持区域；
  seeded_fringe_wta 再执行 WTA，避免多个 object 重复占用同一点；
  C6 不使用 seeded mode，而是用更严格的轨迹与 mask purity 参数测试提前 split/merge 约束。
```

语法与 import smoke：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m py_compile \
  tools/export_carrier_tracklet_graph_v7.py \
  stream4d/reliable_densifier.py \
  tests/test_stream4d_protocol_fixes.py \
  2>&1 | tee logs/stream4d_v7_c5_py_compile.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python - <<'PY' \
  2>&1 | tee logs/stream4d_v7_c5_import_smoke.log
import importlib
for name in ['tools.export_carrier_tracklet_graph_v7', 'stream4d.reliable_densifier']:
    importlib.import_module(name)
    print(f'{name} OK')
PY
```

结果：

```text
logs/stream4d_v7_c5_py_compile.log
exit code 0，日志为空为正常。

logs/stream4d_v7_c5_import_smoke.log
tools.export_carrier_tracklet_graph_v7 OK
stream4d.reliable_densifier OK
```

### C5 seeded_fringe_wta scene0050 smoke

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v7_c5_seeded_fringe_wta_scene0050 \
  --support-mode seeded_fringe_wta \
  2>&1 | tee logs/stream4d_v7_c5_seeded_fringe_wta_scene0050_smoke.log
```

说明：

```text
未显式指定的 graph/seeded 参数使用工具默认值。
完整实际参数已落盘：
  outputs/v7_carrier_tracklet_graph/stream4d_v7_c5_seeded_fringe_wta_scene0050_scene0050_00_summary.json
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c5_seeded_fringe_wta_scene0050_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c5_seeded_fringe_wta_scene0050_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c5_seeded_fringe_wta_scene0050 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c5_seeded_fringe_wta_scene0050_eval.log
```

结果文件：

```text
data/evaluation/scannet/stream4d_v7_c5_seeded_fringe_wta_scene0050_class_agnostic.txt
outputs/v7_carrier_tracklet_graph/stream4d_v7_c5_seeded_fringe_wta_scene0050_scene0050_00_summary.json
```

### C5W seeded_fringe_wta wide scene0050 smoke

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v7_c5w_seeded_fringe_wta_scene0050 \
  --support-mode seeded_fringe_wta \
  --seeded-max-masks-per-object 12 \
  --seeded-distance-px 64 \
  --seeded-mask-selection-mode coverage \
  2>&1 | tee logs/stream4d_v7_c5w_seeded_fringe_wta_scene0050_smoke.log
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c5w_seeded_fringe_wta_scene0050_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c5w_seeded_fringe_wta_scene0050_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c5w_seeded_fringe_wta_scene0050 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c5w_seeded_fringe_wta_scene0050_eval.log
```

结果文件：

```text
data/evaluation/scannet/stream4d_v7_c5w_seeded_fringe_wta_scene0050_class_agnostic.txt
outputs/v7_carrier_tracklet_graph/stream4d_v7_c5w_seeded_fringe_wta_scene0050_scene0050_00_summary.json
```

### C6 strict-track WTA scene0050 smoke

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v7_c6_strict_track_wta_scene0050 \
  --support-mode core_fringe_wta \
  --min-shared-frames 3 \
  --min-positive-ratio 0.65 \
  --max-pair-distance-variance 0.005 \
  --max-component-carriers 180 \
  --min-mask-carriers 6 \
  --min-frame-mask-ratio 0.65 \
  2>&1 | tee logs/stream4d_v7_c6_strict_track_wta_scene0050_smoke.log
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c6_strict_track_wta_scene0050_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c6_strict_track_wta_scene0050_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c6_strict_track_wta_scene0050 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c6_strict_track_wta_scene0050_eval.log
```

结果文件：

```text
data/evaluation/scannet/stream4d_v7_c6_strict_track_wta_scene0050_class_agnostic.txt
outputs/v7_carrier_tracklet_graph/stream4d_v7_c6_strict_track_wta_scene0050_scene0050_00_summary.json
```

### C6 strict-track WTA full probe5

导出 full probe5：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
DEBUG_ROOT=outputs/stream4d_v5_cache_96f_probe5
SEQ_LIST=splits/scannet_v6_probe5.txt
LOG=logs/stream4d_v7_c6_strict_track_wta_probe5.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  printf '### C6 %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m tools.export_carrier_tracklet_graph_v7 \
    --debug-root "$DEBUG_ROOT" \
    --seq-name "$scene" \
    --output-config stream4d_v7_c6_strict_track_wta_probe5 \
    --support-mode core_fringe_wta \
    --min-shared-frames 3 \
    --min-positive-ratio 0.65 \
    --max-pair-distance-variance 0.005 \
    --max-component-carriers 180 \
    --min-mask-carriers 6 \
    --min-frame-mask-ratio 0.65 2>&1 | tee -a "$LOG"
done < "$SEQ_LIST"
```

评估 full probe5：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c6_strict_track_wta_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c6_strict_track_wta_probe5_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c6_strict_track_wta_probe5 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c6_strict_track_wta_probe5_eval.log
```

per-scene 评估（C6）：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
SRC_PRED=data/prediction/stream4d_v7_c6_strict_track_wta_probe5_class_agnostic
TMP_CONFIG=stream4d_v7_c6_strict_track_wta_probe5
LOG=logs/stream4d_v7_c6_strict_track_wta_probe5_per_scene_eval.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  SCENE_DIR="data/prediction/stream4d_v7_c6_strict_track_wta_probe5_${scene}_class_agnostic"
  mkdir -p "$SCENE_DIR"
  ln -sf "$(realpath "$SRC_PRED/config_manifest.json")" "$SCENE_DIR/config_manifest.json"
  ln -sf "$(realpath "$SRC_PRED/${scene}.npz")" "$SCENE_DIR/${scene}.npz"
  printf '### C6 per-scene eval %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m evaluation.evaluate \
    --pred_path "$SCENE_DIR" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/stream4d_v7_c6_strict_track_wta_probe5_${scene}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$TMP_CONFIG" \
    --no_class \
    --require-manifest 2>&1 | tee -a "$LOG"
done < splits/scannet_v6_probe5.txt
```

per-scene 对照评估（C4）：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
SRC_PRED=data/prediction/stream4d_v7_c4_carrier_corefringe_wta_probe5_class_agnostic
TMP_CONFIG=stream4d_v7_c4_carrier_corefringe_wta_probe5
LOG=logs/stream4d_v7_c4_carrier_corefringe_wta_probe5_per_scene_eval.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  SCENE_DIR="data/prediction/stream4d_v7_c4_carrier_corefringe_wta_probe5_${scene}_class_agnostic"
  mkdir -p "$SCENE_DIR"
  ln -sf "$(realpath "$SRC_PRED/config_manifest.json")" "$SCENE_DIR/config_manifest.json"
  ln -sf "$(realpath "$SRC_PRED/${scene}.npz")" "$SCENE_DIR/${scene}.npz"
  printf '### C4 per-scene eval %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m evaluation.evaluate \
    --pred_path "$SCENE_DIR" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/stream4d_v7_c4_carrier_corefringe_wta_probe5_${scene}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$TMP_CONFIG" \
    --no_class \
    --require-manifest 2>&1 | tee -a "$LOG"
done < splits/scannet_v6_probe5.txt
```

### C6 验证与审计

语法、import、单测：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m py_compile \
  tools/export_carrier_tracklet_graph_v7.py \
  stream4d/reliable_densifier.py \
  tests/test_stream4d_protocol_fixes.py \
  2>&1 | tee logs/stream4d_v7_c6_py_compile.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m unittest -v tests.test_stream4d_protocol_fixes \
  2>&1 | tee logs/stream4d_v7_c6_unit_tests.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python - <<'PY' \
  2>&1 | tee logs/stream4d_v7_c6_import_smoke.log
import importlib
for name in ['tools.export_carrier_tracklet_graph_v7','stream4d.reliable_densifier']:
    importlib.import_module(name)
    print(f'{name} OK')
PY
```

结果：

```text
logs/stream4d_v7_c6_py_compile.log
exit code 0，日志为空为正常。

logs/stream4d_v7_c6_import_smoke.log
tools.export_carrier_tracklet_graph_v7 OK
stream4d.reliable_densifier OK

logs/stream4d_v7_c6_unit_tests.log
Ran 13 tests in 0.088s
OK
```

metric integrity：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs stream4d_v7_c6_strict_track_wta_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c6_strict_track_wta_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c6_strict_track_wta_probe5.log
```

结果：

```text
outputs/audit/v7_metric_integrity_c6_strict_track_wta_probe5.md
phase0_pass=True
evaluator_ap_core_equal_by_hash=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=1
num_suspicious_configs=0
alignment mean/min=1.000000/1.000000
```

reportable config scan：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v7_c6_strict_track_wta_probe5 \
  --output outputs/audit/v7_reportable_config_scan_c6_strict_track_wta_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c6_strict_track_wta_probe5.log
```

结果：

```text
outputs/audit/v7_reportable_config_scan_c6_strict_track_wta_probe5.json
num_configs=1
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=1
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## 追加执行：C7 pre-expansion mask ownership（2026-06-09）

背景：

```text
C6 strict-track WTA 相比 C4 有小幅提升，但仍保留很高的 WTA pre-conflict rate。
继续按复盘 Insight 尝试：
  per-frame one-to-one mask assignment before support expansion。
```

代码修改：

```text
文件：
  Stream3D/tools/export_carrier_tracklet_graph_v7.py

新增 support mode：
  core_owned_fringe
  core_owned_fringe_wta

核心行为：
  在每个 window 内，先对所有 candidate component 的 selected masks 做 ownership 分配；
  每个 (frame_id, mask_id) 只给得分最高的一个 component；
  full-mask backprojection 只使用 owned masks；
  最后仍可执行 WTA。

ownership score：
  count * ratio * sqrt(component_size) * sqrt(frame_count)

新增 diagnostics：
  support_ownership_candidate_mask_claims
  support_ownership_unique_masks
  support_ownership_competing_masks
  support_ownership_mask_claims_kept
  support_ownership_mask_claims_dropped
```

语法和 import smoke：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m py_compile \
  tools/export_carrier_tracklet_graph_v7.py \
  2>&1 | tee logs/stream4d_v7_c7_py_compile.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python - <<'PY' \
  2>&1 | tee logs/stream4d_v7_c7_import_smoke.log
import importlib
importlib.import_module('tools.export_carrier_tracklet_graph_v7')
print('tools.export_carrier_tracklet_graph_v7 OK')
PY
```

结果：

```text
logs/stream4d_v7_c7_py_compile.log
exit code 0，日志为空为正常。

logs/stream4d_v7_c7_import_smoke.log
tools.export_carrier_tracklet_graph_v7 OK
```

### C7 scene0050 smoke

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.export_carrier_tracklet_graph_v7 \
  --debug-root outputs/stream4d_v5_cache_96f_probe5 \
  --seq-name scene0050_00 \
  --output-config stream4d_v7_c7_owned_fringe_wta_scene0050 \
  --support-mode core_owned_fringe_wta \
  --min-shared-frames 3 \
  --min-positive-ratio 0.65 \
  --max-pair-distance-variance 0.005 \
  --max-component-carriers 180 \
  --min-mask-carriers 6 \
  --min-frame-mask-ratio 0.65 \
  2>&1 | tee logs/stream4d_v7_c7_owned_fringe_wta_scene0050_smoke.log
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c7_owned_fringe_wta_scene0050_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c7_owned_fringe_wta_scene0050_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c7_owned_fringe_wta_scene0050 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c7_owned_fringe_wta_scene0050_eval.log
```

结果文件：

```text
data/evaluation/scannet/stream4d_v7_c7_owned_fringe_wta_scene0050_class_agnostic.txt
outputs/v7_carrier_tracklet_graph/stream4d_v7_c7_owned_fringe_wta_scene0050_scene0050_00_summary.json
```

### C7 full probe5

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
DEBUG_ROOT=outputs/stream4d_v5_cache_96f_probe5
SEQ_LIST=splits/scannet_v6_probe5.txt
LOG=logs/stream4d_v7_c7_owned_fringe_wta_probe5.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  printf '### C7 %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m tools.export_carrier_tracklet_graph_v7 \
    --debug-root "$DEBUG_ROOT" \
    --seq-name "$scene" \
    --output-config stream4d_v7_c7_owned_fringe_wta_probe5 \
    --support-mode core_owned_fringe_wta \
    --min-shared-frames 3 \
    --min-positive-ratio 0.65 \
    --max-pair-distance-variance 0.005 \
    --max-component-carriers 180 \
    --min-mask-carriers 6 \
    --min-frame-mask-ratio 0.65 2>&1 | tee -a "$LOG"
done < "$SEQ_LIST"
```

评估：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v7_c7_owned_fringe_wta_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/stream4d_v7_c7_owned_fringe_wta_probe5_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config stream4d_v7_c7_owned_fringe_wta_probe5 \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_c7_owned_fringe_wta_probe5_eval.log
```

per-scene 评估：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
SRC_PRED=data/prediction/stream4d_v7_c7_owned_fringe_wta_probe5_class_agnostic
TMP_CONFIG=stream4d_v7_c7_owned_fringe_wta_probe5
LOG=logs/stream4d_v7_c7_owned_fringe_wta_probe5_per_scene_eval.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  SCENE_DIR="data/prediction/stream4d_v7_c7_owned_fringe_wta_probe5_${scene}_class_agnostic"
  mkdir -p "$SCENE_DIR"
  ln -sf "$(realpath "$SRC_PRED/config_manifest.json")" "$SCENE_DIR/config_manifest.json"
  ln -sf "$(realpath "$SRC_PRED/${scene}.npz")" "$SCENE_DIR/${scene}.npz"
  printf '### C7 per-scene eval %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m evaluation.evaluate \
    --pred_path "$SCENE_DIR" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/stream4d_v7_c7_owned_fringe_wta_probe5_${scene}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$TMP_CONFIG" \
    --no_class \
    --require-manifest 2>&1 | tee -a "$LOG"
done < splits/scannet_v6_probe5.txt
```

### C7 验证与审计

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m py_compile \
  tools/export_carrier_tracklet_graph_v7.py \
  stream4d/reliable_densifier.py \
  tests/test_stream4d_protocol_fixes.py \
  2>&1 | tee logs/stream4d_v7_c7_py_compile_final.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m unittest -v tests.test_stream4d_protocol_fixes \
  2>&1 | tee logs/stream4d_v7_c7_unit_tests.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python - <<'PY' \
  2>&1 | tee logs/stream4d_v7_c7_import_smoke_final.log
import importlib
for name in ['tools.export_carrier_tracklet_graph_v7','stream4d.reliable_densifier']:
    importlib.import_module(name)
    print(f'{name} OK')
PY
```

结果：

```text
logs/stream4d_v7_c7_py_compile_final.log
exit code 0，日志为空为正常。

logs/stream4d_v7_c7_import_smoke_final.log
tools.export_carrier_tracklet_graph_v7 OK
stream4d.reliable_densifier OK

logs/stream4d_v7_c7_unit_tests.log
Ran 13 tests in 0.095s
OK
```

metric integrity：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs stream4d_v7_c7_owned_fringe_wta_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c7_owned_fringe_wta_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c7_owned_fringe_wta_probe5.log
```

结果：

```text
outputs/audit/v7_metric_integrity_c7_owned_fringe_wta_probe5.md
phase0_pass=True
evaluator_ap_core_equal_by_hash=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=1
num_suspicious_configs=0
alignment mean/min=1.000000/1.000000
```

reportable config scan：

```bash
/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v7_c7_owned_fringe_wta_probe5 \
  --output outputs/audit/v7_reportable_config_scan_c7_owned_fringe_wta_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c7_owned_fringe_wta_probe5.log
```

结果：

```text
outputs/audit/v7_reportable_config_scan_c7_owned_fringe_wta_probe5.json
num_configs=1
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=1
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## 追加执行：C9 dense object-component + C8P carrier support（2026-06-09）

背景：

```text
C8P post-WTA scene track linking 是 carrier-tracklet 分支当前最佳，
但 AP=0.04249776562268287，仍远低于 v6 object-component baseline。

按 C8P 复盘的下一步方向，C9 测试：
  RGB-D/object-component 作为 dense geometry 主体；
  C8P carrier track 作为 identity/support signal。

本轮不改代码，复用已有工具：
  tools.fuse_prediction_configs
  tools.object_competition_rank
```

关键输入：

```text
primary dense object:
  stream4d_v6_e4_probe5_objcomp_m670_g101_compact_only_preserve

secondary / support carrier track:
  stream4d_v7_c8p_posttrack_owned_wta_probe5

candidate bank:
  stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5

seq list:
  splits/scannet_v6_probe5.txt
```

### C9A/C9B：dense primary + low-score C8P secondary fusion

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
PRIMARY=stream4d_v6_e4_probe5_objcomp_m670_g101_compact_only_preserve
SECONDARY=stream4d_v7_c8p_posttrack_owned_wta_probe5
SEQ=splits/scannet_v6_probe5.txt

$PY -m tools.fuse_prediction_configs \
  --root . \
  --seq-list "$SEQ" \
  --primary-config "$PRIMARY" \
  --secondary-config "$SECONDARY" \
  --output-config stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050 \
  --fusion-mode concatenate \
  --preserve-primary-score \
  --secondary-score 0.005 \
  --drop-secondary-iou-threshold 0.50 \
  --drop-secondary-overlap-mode min_ioc \
  --summary-root outputs/stream4d_v7_c9_fusion \
  2>&1 | tee logs/stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050_fuse.log

$PY -m tools.fuse_prediction_configs \
  --root . \
  --seq-list "$SEQ" \
  --primary-config "$PRIMARY" \
  --secondary-config "$SECONDARY" \
  --output-config stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030 \
  --fusion-mode concatenate \
  --preserve-primary-score \
  --secondary-score 0.005 \
  --drop-secondary-iou-threshold 0.30 \
  --drop-secondary-overlap-mode min_ioc \
  --summary-root outputs/stream4d_v7_c9_fusion \
  2>&1 | tee logs/stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030_fuse.log
```

评估：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

for CFG in \
  stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050 \
  stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030
do
  $PY -m evaluation.evaluate \
    --pred_path data/prediction/${CFG}_class_agnostic \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
    --tmp_root data/TMP \
    --tmp_config $CFG \
    --no_class \
    --require-manifest \
    2>&1 | tee logs/${CFG}_eval.log
done
```

结果文件：

```text
data/evaluation/scannet/stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030_class_agnostic.txt
outputs/stream4d_v7_c9_fusion/stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050_summary.json
outputs/stream4d_v7_c9_fusion/stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030_summary.json
```

### C9C/C9D：C8P support 替换 TMP 的错误尝试

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
SUPPORT=stream4d_v7_c8p_posttrack_owned_wta_probe5
SEQ=splits/scannet_v6_probe5.txt

$PY -m tools.object_competition_rank \
  --root . \
  --seq-list "$SEQ" \
  --input-config "$INPUT" \
  --output-config stream4d_v7_c9c_objcomp_c8p_support_compact_m670_g101 \
  --score-pre-points-config "$SUPPORT" \
  --quality-mode compact_only \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 1.01 \
  --min-support-area 1 \
  --max-instances 670 \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve 0 \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area 0 \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/stream4d_v7_c9_object_competition \
  2>&1 | tee logs/stream4d_v7_c9c_objcomp_c8p_support_compact_m670_g101.log

$PY -m tools.object_competition_rank \
  --root . \
  --seq-list "$SEQ" \
  --input-config "$INPUT" \
  --output-config stream4d_v7_c9d_objcomp_c8p_support_scoreunique_m670_g101 \
  --score-pre-points-config "$SUPPORT" \
  --quality-mode score_unique_compact \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 1.01 \
  --min-support-area 1 \
  --max-instances 670 \
  --preserve-original-score \
  --tmp-policy score_support \
  --small-rescue-reserve 0 \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area 0 \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/stream4d_v7_c9_object_competition \
  2>&1 | tee logs/stream4d_v7_c9d_objcomp_c8p_support_scoreunique_m670_g101.log
```

评估：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

for CFG in \
  stream4d_v7_c9c_objcomp_c8p_support_compact_m670_g101 \
  stream4d_v7_c9d_objcomp_c8p_support_scoreunique_m670_g101
do
  $PY -m evaluation.evaluate \
    --pred_path data/prediction/${CFG}_class_agnostic \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
    --tmp_root data/TMP \
    --tmp_config $CFG \
    --no_class \
    --require-manifest \
    2>&1 | tee logs/${CFG}_eval.log
done
```

说明：

```text
C9C/C9D 的结果极差后，审计显示 policy={"inconsistent_union_not_subset": 5}。
这是因为 tmp-policy=score_support 把 evaluation/TMP support 也换成 C8P track union，
不再符合“RGB-D dense object-component 主体”的设计。
因此继续按修复方向重跑 C9E/C9F：C8P 只影响 ranking，TMP 继承 dense input。
```

### C9E/C9F：修复 C9C/C9D，inherit dense TMP

导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
INPUT=stream4d_v4_1_probe5_selfboundary_high_comp_cat095_on_32f_probe5
SUPPORT=stream4d_v7_c8p_posttrack_owned_wta_probe5
SEQ=splits/scannet_v6_probe5.txt

$PY -m tools.object_competition_rank \
  --root . \
  --seq-list "$SEQ" \
  --input-config "$INPUT" \
  --output-config stream4d_v7_c9e_objcomp_c8p_rank_inherit_compact_m670_g101 \
  --score-pre-points-config "$SUPPORT" \
  --quality-mode compact_only \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 1.01 \
  --min-support-area 1 \
  --max-instances 670 \
  --preserve-original-score \
  --tmp-policy inherit \
  --small-rescue-reserve 0 \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area 0 \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/stream4d_v7_c9_object_competition \
  2>&1 | tee logs/stream4d_v7_c9e_objcomp_c8p_rank_inherit_compact_m670_g101.log

$PY -m tools.object_competition_rank \
  --root . \
  --seq-list "$SEQ" \
  --input-config "$INPUT" \
  --output-config stream4d_v7_c9f_objcomp_c8p_rank_inherit_scoreunique_m670_g101 \
  --score-pre-points-config "$SUPPORT" \
  --quality-mode score_unique_compact \
  --group-overlap-mode min_ioc \
  --group-overlap-threshold 1.01 \
  --min-support-area 1 \
  --max-instances 670 \
  --preserve-original-score \
  --tmp-policy inherit \
  --small-rescue-reserve 0 \
  --small-rescue-min-support-area 1 \
  --small-rescue-max-support-area 0 \
  --small-rescue-min-novel-points 80 \
  --small-rescue-overlap-threshold 0.50 \
  --small-rescue-overlap-mode min_ioc \
  --summary-root outputs/stream4d_v7_c9_object_competition \
  2>&1 | tee logs/stream4d_v7_c9f_objcomp_c8p_rank_inherit_scoreunique_m670_g101.log
```

评估：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

for CFG in \
  stream4d_v7_c9e_objcomp_c8p_rank_inherit_compact_m670_g101 \
  stream4d_v7_c9f_objcomp_c8p_rank_inherit_scoreunique_m670_g101
do
  $PY -m evaluation.evaluate \
    --pred_path data/prediction/${CFG}_class_agnostic \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
    --tmp_root data/TMP \
    --tmp_config $CFG \
    --no_class \
    --require-manifest \
    2>&1 | tee logs/${CFG}_eval.log
done
```

### C9 审计

第一次审计命令错误：

```text
logs/stream4d_v7_metric_integrity_c9_probe5.log
error: unrecognized arguments ...
原因：--configs 需要逗号分隔字符串，不能传多个 positional configs。
```

修复版：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
CONFIGS=stream4d_v7_c9a_dense_primary_c8p_low_s005_minioc050,stream4d_v7_c9b_dense_primary_c8p_low_s005_minioc030,stream4d_v7_c9c_objcomp_c8p_support_compact_m670_g101,stream4d_v7_c9d_objcomp_c8p_support_scoreunique_m670_g101,stream4d_v7_c9e_objcomp_c8p_rank_inherit_compact_m670_g101,stream4d_v7_c9f_objcomp_c8p_rank_inherit_scoreunique_m670_g101

$PY -m py_compile \
  tools/fuse_prediction_configs.py \
  tools/object_competition_rank.py \
  tools/export_carrier_tracklet_graph_v7.py \
  tools/prediction_manifest.py \
  2>&1 | tee logs/stream4d_v7_c9_py_compile_fixed.log

$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c9_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c9_probe5_fixed.log

$PY -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v7_reportable_config_scan_c9_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c9_probe5_fixed.log
```

审计结果：

```text
logs/stream4d_v7_c9_py_compile_fixed.log
exit code 0，日志为空为正常。

outputs/audit/v7_metric_integrity_c9_probe5.md
phase0_pass=True
evaluator_ap_core_equal_by_hash=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=6
num_suspicious_configs=0

outputs/audit/v7_reportable_config_scan_c9_probe5.json
num_configs=6
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=6
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## 追加执行：C10 dense masks + C8P track-overlap score-only rescore（2026-06-09）

背景：

```text
C9 说明：
1. 直接输出 C8P coarse masks 会引入低分 FP；
2. C8P union 作为 support/ranking 几乎不能超过 v6 dense baseline。

C10 继续尝试更保守方向：
  dense masks 不变；
  dense TMP/pre_points 不变；
  只用 dense object 与 C8P track mask 的 overlap feature 重排 score。

本轮不改源码，生成 prediction artifact。
```

生成逻辑：

```text
primary_config = stream4d_v6_e4_probe5_objcomp_m670_g101_compact_only_preserve
track_config   = stream4d_v7_c8p_posttrack_owned_wta_probe5

对每个 scene：
  primary_area = dense_mask.sum
  track_area = c8p_track_mask.sum
  inter = dense_mask.T @ c8p_track_mask

feature best_primary_ioc:
  max_track inter / primary_area

feature best_track_ioc:
  max_track inter / track_area

输出 masks/classes/TMP 全部继承 primary；
只改 pred_score:
  new_score = original_score + weight * feature
```

生成命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY - <<'PY' 2>&1 | tee logs/stream4d_v7_c10_dense_track_overlap_rescore_generate.log
# Python heredoc 生成四个 config：
#   stream4d_v7_c10a_dense_compact_track_pioc_w001
#     feature=best_primary_ioc, weight=0.01
#   stream4d_v7_c10b_dense_compact_track_pioc_w010
#     feature=best_primary_ioc, weight=0.10
#   stream4d_v7_c10c_dense_compact_track_tioc_w010
#     feature=best_track_ioc, weight=0.10
#   stream4d_v7_c10d_dense_compact_track_tioc_w050
#     feature=best_track_ioc, weight=0.50
#
# 生成内容：
#   data/prediction/${CFG}_class_agnostic/*.npz
#   data/TMP/${CFG}/*_pre_points.npy
#   data/prediction/${CFG}_class_agnostic/config_manifest.json
#   outputs/stream4d_v7_c10_track_rescore/${CFG}_summary.json
#
# 完整生成逻辑见本节“生成逻辑”和 summary/manifest。
PY
```

实际生成日志：

```text
logs/stream4d_v7_c10_dense_track_overlap_rescore_generate.log
[c10-generate] wrote stream4d_v7_c10a_dense_compact_track_pioc_w001
[c10-generate] wrote stream4d_v7_c10b_dense_compact_track_pioc_w010
[c10-generate] wrote stream4d_v7_c10c_dense_compact_track_tioc_w010
[c10-generate] wrote stream4d_v7_c10d_dense_compact_track_tioc_w050
```

评估：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
set -euo pipefail
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

for CFG in \
  stream4d_v7_c10a_dense_compact_track_pioc_w001 \
  stream4d_v7_c10b_dense_compact_track_pioc_w010 \
  stream4d_v7_c10c_dense_compact_track_tioc_w010 \
  stream4d_v7_c10d_dense_compact_track_tioc_w050
do
  $PY -m evaluation.evaluate \
    --pred_path data/prediction/${CFG}_class_agnostic \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
    --tmp_root data/TMP \
    --tmp_config $CFG \
    --no_class \
    --require-manifest \
    2>&1 | tee logs/${CFG}_eval.log
done
```

结果文件：

```text
data/evaluation/scannet/stream4d_v7_c10a_dense_compact_track_pioc_w001_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_c10b_dense_compact_track_pioc_w010_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_c10c_dense_compact_track_tioc_w010_class_agnostic.txt
data/evaluation/scannet/stream4d_v7_c10d_dense_compact_track_tioc_w050_class_agnostic.txt
outputs/stream4d_v7_c10_track_rescore/stream4d_v7_c10*_summary.json
```

C10 审计：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CONFIGS=stream4d_v7_c10a_dense_compact_track_pioc_w001,stream4d_v7_c10b_dense_compact_track_pioc_w010,stream4d_v7_c10c_dense_compact_track_tioc_w010,stream4d_v7_c10d_dense_compact_track_tioc_w050

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c10_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c10_probe5.log

/mnt/data/users/chengshun.wang/miniconda3/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v7_reportable_config_scan_c10_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c10_probe5.log
```

审计结果：

```text
outputs/audit/v7_metric_integrity_c10_probe5.md
phase0_pass=True
evaluator_ap_core_equal_by_hash=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=4
num_suspicious_configs=0

outputs/audit/v7_reportable_config_scan_c10_probe5.json
num_configs=4
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=4
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## r4 code-review packet（2026-06-09）

生成状态与 diff：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR

git status --short > Stream3D/outputs/audit/git_status_stream4d_v7_packet_r4.txt

git diff --no-ext-diff -- \
  Stream3D/evaluation/evaluate.py \
  Stream3D/stream4d/reliable_densifier.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/make_union_prepoints_config.py \
  Stream3D/tools/diagnose_prediction_quality.py \
  Stream3D/tools/export_carrier_tracklet_graph_v7.py \
  Stream3D/tools/export_d4rt_geometry_degradation_v7.py \
  Stream3D/tools/summarize_v7_gap_matrix.py \
  Stream3D/tests/test_stream4d_protocol_fixes.py \
  docs/stream4d_v7_执行日志.md \
  docs/stream4d_v7_实验结果复盘.md \
  > Stream3D/outputs/audit/git_diff_stream4d_v7_packet_r4.patch
```

打包：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR

PACKET=stream4d_v7_code_review_packet_20260609_015450_r4
FILELIST=${PACKET}_filelist.txt

# filelist 收录：
#   root docs/stream4d_v7*
#   Stream3D/docs/stream4d_v7*
#   Stream3D/evaluation/*.py
#   Stream3D/stream4d/*.py
#   Stream3D/tools/*.py
#   Stream3D/tests/*.py
#   Stream3D/logs/*v7*.log
#   Stream3D/outputs/audit/*v7* / git packet status-diff
#   Stream3D/outputs/v7_carrier_tracklet_graph/*v7*.json
#   Stream3D/outputs/stream4d_v7_c9_*/*.json
#   Stream3D/outputs/stream4d_v7_c10_track_rescore/*.json
#   Stream3D/outputs/v7_d4rt_geometry_degradation/*v7*.json
#   Stream3D/data/evaluation/scannet/*v7*.txt
#   Stream3D/data/prediction/stream4d_v7*/config_manifest.json

zip -q "$PACKET.zip" -@ < "$FILELIST"
sha256sum "$PACKET.zip" > "$PACKET.sha256"
printf '%s.zip\n' "$PACKET" > stream4d_v7_latest_packet_name.txt
unzip -t "$PACKET.zip" | tail -n 5
sha256sum "$PACKET.zip"
wc -l "$FILELIST"
ls -lh "$PACKET.zip"
```

结果：

```text
packet:
  stream4d_v7_code_review_packet_20260609_015450_r4.zip

sha256:
  384f998c45a2691c30044b6a3bd48f80edec3939c782b36d27d5df7b22b96bd2

filelist:
  stream4d_v7_code_review_packet_20260609_015450_r4_filelist.txt
  534 files

zip test:
  No errors detected in compressed data of stream4d_v7_code_review_packet_20260609_015450_r4.zip.

size:
  1.1M
```

## 追加执行：C11 dense primary + C8P track-bucket NMS（2026-06-09）

背景：

```text
C10 证明 C8P track overlap 不适合直接改 dense score ordering。
继续按 C10 insight 尝试更结构化但仍保守的 object-level assignment：
  dense primary masks/TMP 不变；
  不输出 C8P masks；
  不使用 C8P union；
  只把 C8P track 当作 dense candidate bucket，用于 bucket 内 duplicate suppression。
```

首次生成尝试：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY - <<'PY' 2>&1 | tee logs/stream4d_v7_c11_track_bucket_nms_generate.log
# Python heredoc：dense primary + C8P track bucket NMS
PY
```

结果：

```text
命令运行过慢，手动终止。
logs/stream4d_v7_c11_track_bucket_nms_generate.log 为空。
未生成可报告结果。
```

第二次生成尝试：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY - <<'PY' 2>&1 | tee logs/stream4d_v7_c11_track_bucket_nms_generate_fast.log
# Python heredoc：full-scene dense matrix 版本
PY
```

结果：

```text
进程被 kill，exit 143。
logs/stream4d_v7_c11_track_bucket_nms_generate_fast.log 为空。
原因：full dense matrix over scene 太重。
```

修复方向：

```text
按 blocker 修复，不放弃：
  改成 sparse primary-union overlap；
  只在 dense primary union / C8P bucket 范围内计算；
  TMP/pre_points 继承 primary；
  输出仅包含 dense primary masks/classes/scores 的 subset。
```

成功生成命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY - <<'PY' 2>&1 | tee logs/stream4d_v7_c11_track_bucket_nms_generate_sparse.log
# Python heredoc 生成五个 config：
#   stream4d_v7_c11a_compact_trackbucket_minioc090_a050
#   stream4d_v7_c11b_compact_trackbucket_minioc075_a050
#   stream4d_v7_c11c_compact_trackbucket_iou050_a050
#   stream4d_v7_c11d_scoreunique_trackbucket_minioc090_a050
#   stream4d_v7_c11e_scoreunique_trackbucket_minioc075_a050
#
# 生成内容：
#   data/prediction/${CFG}_class_agnostic/*.npz
#   data/TMP/${CFG}/*_pre_points.npy
#   data/prediction/${CFG}_class_agnostic/config_manifest.json
#   outputs/stream4d_v7_c11_track_bucket_nms/${CFG}_summary.json
PY
```

实际生成日志：

```text
[c11-generate-sparse] wrote stream4d_v7_c11a_compact_trackbucket_minioc090_a050
[c11-generate-sparse] wrote stream4d_v7_c11b_compact_trackbucket_minioc075_a050
[c11-generate-sparse] wrote stream4d_v7_c11c_compact_trackbucket_iou050_a050
[c11-generate-sparse] wrote stream4d_v7_c11d_scoreunique_trackbucket_minioc090_a050
[c11-generate-sparse] wrote stream4d_v7_c11e_scoreunique_trackbucket_minioc075_a050
```

C11 评估命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

for CFG in \
  stream4d_v7_c11a_compact_trackbucket_minioc090_a050 \
  stream4d_v7_c11b_compact_trackbucket_minioc075_a050 \
  stream4d_v7_c11c_compact_trackbucket_iou050_a050 \
  stream4d_v7_c11d_scoreunique_trackbucket_minioc090_a050 \
  stream4d_v7_c11e_scoreunique_trackbucket_minioc075_a050
do
  $PY -m evaluation.evaluate \
    --pred_path data/prediction/${CFG}_class_agnostic \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
    --tmp_root data/TMP \
    --tmp_config $CFG \
    --no_class \
    --require-manifest \
    2>&1 | tee logs/${CFG}_eval.log
done
```

C11 结果文件：

```text
stream4d_v7_c11a_compact_trackbucket_minioc090_a050:
  0.2704320873317211 / 0.47988173584030724 / 0.6695362536371277
stream4d_v7_c11b_compact_trackbucket_minioc075_a050:
  0.2686413052853578 / 0.47453179052845046 / 0.6644916609507336
stream4d_v7_c11c_compact_trackbucket_iou050_a050:
  0.27236752539720466 / 0.47994690773425164 / 0.6696705657993772
stream4d_v7_c11d_scoreunique_trackbucket_minioc090_a050:
  0.26886596867628404 / 0.47656996919717703 / 0.6780442110000419
stream4d_v7_c11e_scoreunique_trackbucket_minioc075_a050:
  0.2671394709133604 / 0.4714001637618435 / 0.6731127314002505
```

C11 审计命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
CONFIGS=stream4d_v7_c11a_compact_trackbucket_minioc090_a050,stream4d_v7_c11b_compact_trackbucket_minioc075_a050,stream4d_v7_c11c_compact_trackbucket_iou050_a050,stream4d_v7_c11d_scoreunique_trackbucket_minioc090_a050,stream4d_v7_c11e_scoreunique_trackbucket_minioc075_a050

$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c11_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c11_probe5.log

$PY -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v7_reportable_config_scan_c11_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c11_probe5.log
```

C11 审计结果：

```text
outputs/audit/v7_metric_integrity_c11_probe5.md
phase0_pass=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=5
num_suspicious_configs=0

outputs/audit/v7_reportable_config_scan_c11_probe5.json
num_configs=5
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=5
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## 追加执行：C12 conservative track-bucket suppression（2026-06-09）

背景：

```text
C11 删除 38.8 到 64.2 个 instances/scene，明显 over-suppression。
C12 提高阈值，只删几乎确定的 duplicate：
  assign_primary_ioc_min = 0.90 或 0.95
  min_ioc@0.99 或 iou@0.85
```

生成命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY - <<'PY' 2>&1 | tee logs/stream4d_v7_c12_conservative_track_bucket_generate.log
# Python heredoc 生成五个 config：
#   stream4d_v7_c12a_compact_trackbucket_minioc099_a090
#   stream4d_v7_c12b_compact_trackbucket_iou085_a090
#   stream4d_v7_c12c_compact_trackbucket_minioc099_a095
#   stream4d_v7_c12d_scoreunique_trackbucket_minioc099_a090
#   stream4d_v7_c12e_scoreunique_trackbucket_iou085_a090
PY
```

实际生成日志：

```text
[c12-generate] wrote stream4d_v7_c12a_compact_trackbucket_minioc099_a090
[c12-generate] wrote stream4d_v7_c12b_compact_trackbucket_iou085_a090
[c12-generate] wrote stream4d_v7_c12c_compact_trackbucket_minioc099_a095
[c12-generate] wrote stream4d_v7_c12d_scoreunique_trackbucket_minioc099_a090
[c12-generate] wrote stream4d_v7_c12e_scoreunique_trackbucket_iou085_a090
```

C12 评估命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

for CFG in \
  stream4d_v7_c12a_compact_trackbucket_minioc099_a090 \
  stream4d_v7_c12b_compact_trackbucket_iou085_a090 \
  stream4d_v7_c12c_compact_trackbucket_minioc099_a095 \
  stream4d_v7_c12d_scoreunique_trackbucket_minioc099_a090 \
  stream4d_v7_c12e_scoreunique_trackbucket_iou085_a090
do
  $PY -m evaluation.evaluate \
    --pred_path data/prediction/${CFG}_class_agnostic \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
    --tmp_root data/TMP \
    --tmp_config $CFG \
    --no_class \
    --require-manifest \
    2>&1 | tee logs/${CFG}_eval.log
done
```

C12 结果：

```text
stream4d_v7_c12a_compact_trackbucket_minioc099_a090:
  0.2800747423738896 / 0.49685112752837807 / 0.6829500187596522
stream4d_v7_c12b_compact_trackbucket_iou085_a090:
  0.283520751520795 / 0.5023288260789226 / 0.6829700373610922
stream4d_v7_c12c_compact_trackbucket_minioc099_a095:
  0.2799994121344577 / 0.49673714634563515 / 0.6827419329888822
stream4d_v7_c12d_scoreunique_trackbucket_minioc099_a090:
  0.27834384843765725 / 0.4932120531494701 / 0.6916215209066441
stream4d_v7_c12e_scoreunique_trackbucket_iou085_a090:
  0.28169453388969407 / 0.49853735125740806 / 0.6916715496834056
```

C12 审计命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
CONFIGS=stream4d_v7_c12a_compact_trackbucket_minioc099_a090,stream4d_v7_c12b_compact_trackbucket_iou085_a090,stream4d_v7_c12c_compact_trackbucket_minioc099_a095,stream4d_v7_c12d_scoreunique_trackbucket_minioc099_a090,stream4d_v7_c12e_scoreunique_trackbucket_iou085_a090

$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c12_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c12_probe5.log

$PY -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v7_reportable_config_scan_c12_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c12_probe5.log
```

C12 审计结果：

```text
outputs/audit/v7_metric_integrity_c12_probe5.md
phase0_pass=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=5
num_suspicious_configs=0

outputs/audit/v7_reportable_config_scan_c12_probe5.json
num_configs=5
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=5
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## 追加执行：C13 conservative IoU grid（2026-06-09）

背景：

```text
C12 best local behavior 来自 iou@0.85，但 AP 仍下降。
C13 继续缩小网格，只测更保守的 iou@0.90 / iou@0.95。
```

生成命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY - <<'PY' 2>&1 | tee logs/stream4d_v7_c13_conservative_iou_grid_generate.log
# Python heredoc 生成六个 config：
#   stream4d_v7_c13a_compact_trackbucket_iou090_a090
#   stream4d_v7_c13b_compact_trackbucket_iou095_a090
#   stream4d_v7_c13c_compact_trackbucket_iou090_a095
#   stream4d_v7_c13d_compact_trackbucket_iou095_a095
#   stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090
#   stream4d_v7_c13f_scoreunique_trackbucket_iou095_a090
PY
```

实际生成日志：

```text
[c13-generate] wrote stream4d_v7_c13a_compact_trackbucket_iou090_a090
[c13-generate] wrote stream4d_v7_c13b_compact_trackbucket_iou095_a090
[c13-generate] wrote stream4d_v7_c13c_compact_trackbucket_iou090_a095
[c13-generate] wrote stream4d_v7_c13d_compact_trackbucket_iou095_a095
[c13-generate] wrote stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090
[c13-generate] wrote stream4d_v7_c13f_scoreunique_trackbucket_iou095_a090
```

C13 评估命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

for CFG in \
  stream4d_v7_c13a_compact_trackbucket_iou090_a090 \
  stream4d_v7_c13b_compact_trackbucket_iou095_a090 \
  stream4d_v7_c13c_compact_trackbucket_iou090_a095 \
  stream4d_v7_c13d_compact_trackbucket_iou095_a095 \
  stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090 \
  stream4d_v7_c13f_scoreunique_trackbucket_iou095_a090
do
  $PY -m evaluation.evaluate \
    --pred_path data/prediction/${CFG}_class_agnostic \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
    --tmp_root data/TMP \
    --tmp_config $CFG \
    --no_class \
    --require-manifest \
    2>&1 | tee logs/${CFG}_eval.log
done
```

C13 结果：

```text
stream4d_v7_c13a_compact_trackbucket_iou090_a090:
  0.2837834686112302 / 0.5021096487158756 / 0.6827009115465468
stream4d_v7_c13b_compact_trackbucket_iou095_a090:
  0.28382830955380606 / 0.5019341996506697 / 0.6824298780157181
stream4d_v7_c13c_compact_trackbucket_iou090_a095:
  0.28375652456257155 / 0.5020692354906476 / 0.682628369001877
stream4d_v7_c13d_compact_trackbucket_iou095_a095:
  0.28382830955380606 / 0.5019341996506697 / 0.6824298780157181
stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090:
  0.28195599230250884 / 0.49833569328009064 / 0.6914188730867814
stream4d_v7_c13f_scoreunique_trackbucket_iou095_a090:
  0.28203394183547537 / 0.49820855659464575 / 0.691223277348379
```

C13 审计命令，第一次失败：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
CONFIGS=stream4d_v7_c13a_compact_trackbucket_iou090_a090,stream4d_v7_c13b_compact_trackbucket_iou095_a090,stream4d_v7_c13c_compact_trackbucket_iou090_a095,stream4d_v7_c13d_compact_trackbucket_iou095_a095,stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090,stream4d_v7_c13f_scoreunique_trackbucket_iou095_a090

$PY -m tools.verify_stream4d_metric_integrity \
  --configs "$CONFIGS" \
  --output outputs/audit/v7_metric_integrity_c13_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c13_probe5.log
```

失败原因：

```text
verify_stream4d_metric_integrity.py 要求 --orig-stream3d-root 和 --seq-list。
第一次命令只产生参数错误，不作为有效审计结果。
```

C13 审计命令，修正后成功：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
CONFIGS=stream4d_v7_c13a_compact_trackbucket_iou090_a090,stream4d_v7_c13b_compact_trackbucket_iou095_a090,stream4d_v7_c13c_compact_trackbucket_iou090_a095,stream4d_v7_c13d_compact_trackbucket_iou095_a095,stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090,stream4d_v7_c13f_scoreunique_trackbucket_iou095_a090

$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c13_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c13_probe5.log

$PY -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v7_reportable_config_scan_c13_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c13_probe5.log
```

C13 审计结果：

```text
outputs/audit/v7_metric_integrity_c13_probe5.md
phase0_pass=True
num_configs_missing_manifest=0
num_oracle_configs=0
num_reportable_method_configs=6
num_suspicious_configs=0

outputs/audit/v7_reportable_config_scan_c13_probe5.json
num_configs=6
num_configs_missing_manifest=0
num_diagnostic_only_configs=0
num_oracle_configs=0
num_reportable_method_configs=6
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## 追加执行：C11/C12/C13 复现脚本固化（2026-06-09）

文件：

```text
Stream3D/tools/export_trackbucket_dense_variants_v7.py
```

说明：

```text
该文件在 C11/C12/C13 结果已经落盘后新增。
它固化本轮 dense-primary track-bucket suppression 的 group/config 表和生成逻辑，
用于后续复现，不倒填为原始运行命令。
```

验证命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY -m py_compile tools/export_trackbucket_dense_variants_v7.py \
  2>&1 | tee logs/stream4d_v7_trackbucket_repro_py_compile.log

$PY - <<'PY' 2>&1 | tee logs/stream4d_v7_trackbucket_repro_import_smoke.log
import tools.export_trackbucket_dense_variants_v7 as mod
print('tools.export_trackbucket_dense_variants_v7 OK')
print(','.join(sorted(mod.VARIANTS)))
PY
```

验证结果：

```text
py_compile: pass
import smoke:
  tools.export_trackbucket_dense_variants_v7 OK
  c11,c12,c13
```

## r5 code-review packet（2026-06-09）

r5 验证：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py \
  2>&1 | tee logs/stream4d_v7_r5_py_compile.log

$PY -m unittest tests.test_stream4d_protocol_fixes \
  2>&1 | tee logs/stream4d_v7_r5_unit_tests.log

$PY - <<'PY' 2>&1 | tee logs/stream4d_v7_r5_import_smoke.log
import stream4d.reliable_densifier
import tools.export_carrier_tracklet_graph_v7
import tools.export_d4rt_geometry_degradation_v7
import tools.export_trackbucket_dense_variants_v7
import tools.verify_stream4d_metric_integrity
import tools.scan_reportable_configs
print('r5 import smoke OK')
PY
```

验证结果：

```text
py_compile: pass
unit tests:
  Ran 13 tests in 0.099s
  OK
import smoke:
  r5 import smoke OK
```

注意：

```text
第一次 r5 import smoke 曾直接 import evaluation.evaluate，触发旧 evaluator 顶层 argparse：
  error: the following arguments are required: --pred_path, --gt_path, --dataset
这不是 AP core 变更；随后改为不触发 argparse 的模块集合，import smoke 通过。
最终有效日志是 logs/stream4d_v7_r5_import_smoke.log。
```

同步文档：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
cp docs/stream4d_v7_执行日志.md Stream3D/docs/stream4d_v7_执行日志.md
cp docs/stream4d_v7_实验结果复盘.md Stream3D/docs/stream4d_v7_实验结果复盘.md
```

打包命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
PACKET=stream4d_v7_code_review_packet_20260609_022358_r5
STATUS=Stream3D/outputs/audit/git_status_stream4d_v7_packet_r5.txt
DIFF=Stream3D/outputs/audit/git_diff_stream4d_v7_packet_r5.patch

git status --short > "$STATUS"
: > "$DIFF"

git diff --no-ext-diff -- \
  Stream3D/evaluation/evaluate.py \
  Stream3D/stream4d/reliable_densifier.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/make_union_prepoints_config.py \
  Stream3D/tools/diagnose_prediction_quality.py \
  Stream3D/tools/export_carrier_tracklet_graph_v7.py \
  Stream3D/tools/export_d4rt_geometry_degradation_v7.py \
  Stream3D/tools/export_trackbucket_dense_variants_v7.py \
  Stream3D/tools/summarize_v7_gap_matrix.py \
  Stream3D/tests/test_stream4d_protocol_fixes.py \
  docs/stream4d_v7_执行日志.md \
  docs/stream4d_v7_实验结果复盘.md \
  >> "$DIFF"

git diff --no-index -- /dev/null Stream3D/tools/export_trackbucket_dense_variants_v7.py \
  >> "$DIFF" 2>/dev/null || true

{
  find docs -maxdepth 1 -type f -name 'stream4d_v7*'
  find Stream3D/docs -maxdepth 1 -type f -name 'stream4d_v7*'
  find Stream3D/evaluation -maxdepth 1 -type f -name '*.py'
  find Stream3D/stream4d -maxdepth 1 -type f -name '*.py'
  find Stream3D/tools -maxdepth 1 -type f -name '*.py'
  find Stream3D/tests -maxdepth 1 -type f -name '*.py'
  find Stream3D/logs -maxdepth 1 -type f -name '*v7*.log'
  find Stream3D/outputs/audit -maxdepth 1 -type f \( -name '*v7*' -o -name 'git_*stream4d_v7_packet_r5*' \)
  find Stream3D/outputs -maxdepth 2 -type f -path '*/stream4d_v7_*/*' -name '*.json'
  find Stream3D/outputs -maxdepth 2 -type f -path '*/v7_*/*' -name '*.json'
  find Stream3D/data/evaluation/scannet -maxdepth 1 -type f \( -name '*v7*.txt' -o -name 'stream4d_v6_e4_probe5_objcomp_m670_g101*_class_agnostic.txt' \)
  find Stream3D/data/prediction -maxdepth 2 -type f -path '*/stream4d_v7*' -name 'config_manifest.json'
} | sort -u > ${PACKET}_filelist.txt

zip -q ${PACKET}.zip -@ < ${PACKET}_filelist.txt
sha256sum ${PACKET}.zip > ${PACKET}.sha256
printf '%s.zip\n' "$PACKET" > stream4d_v7_latest_packet_name.txt
unzip -t ${PACKET}.zip | tail -n 5
sha256sum ${PACKET}.zip
wc -l ${PACKET}_filelist.txt
ls -lh ${PACKET}.zip
```

打包结果：

```text
packet:
  stream4d_v7_code_review_packet_20260609_022358_r5.zip

sha256:
  see stream4d_v7_code_review_packet_20260609_022358_r5.sha256

filelist:
  stream4d_v7_code_review_packet_20260609_022358_r5_filelist.txt

zip test:
  No errors detected in compressed data of stream4d_v7_code_review_packet_20260609_022358_r5.zip.
```

## 收尾执行：C14/C15 与 GT diagnostic（2026-06-09）

收尾目标：

```text
C11/C12/C13 track-bucket duplicate suppression 没有稳定提升。
按复盘 insight 做最后两类确认：
1. 用 GT-only diagnostic 检查被 suppression 删除的 candidates 是否真伤害 GT 覆盖。
2. 检查 C8P/C14/C15 posttrack merge 是否只是过合并问题。

约束：
GT diagnostic 必须写明 diagnostic_only=True, uses_gt=True，不能进入 method result。
C14/C15 method eval 必须使用 --require-manifest，并跑 metric integrity / reportable scan。
```

### C14：track-bucket suppression GT diagnostic

新增 diagnostic 工具：

```text
Stream3D/tools/diagnose_trackbucket_suppression_v7.py
```

语法检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY -m py_compile tools/diagnose_trackbucket_suppression_v7.py \
  2>&1 | tee logs/stream4d_v7_c14_trackbucket_diagnostic_py_compile.log
```

第一次尝试全量 C11/C12/C13 diagnostic：

```bash
$PY -m tools.diagnose_trackbucket_suppression_v7 \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --groups c11 c12 c13 \
  --output-prefix outputs/audit/v7_trackbucket_suppression_gt_diagnostic_c11_c13 \
  2>&1 | tee logs/stream4d_v7_trackbucket_suppression_gt_diagnostic_c11_c13.log
```

结果：

```text
该全量 diagnostic 运行超过约 90s 仍无输出，手动 Ctrl-C 终止。
未落盘指标，不写入结果表。
随后按 blocker 方向给 diagnostic 工具增加 --configs allowlist 和 progress print，改跑代表配置。
```

代表配置 diagnostic：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
CONFIGS=stream4d_v7_c11a_compact_trackbucket_minioc090_a050,stream4d_v7_c12b_compact_trackbucket_iou085_a090,stream4d_v7_c13a_compact_trackbucket_iou090_a090,stream4d_v7_c13e_scoreunique_trackbucket_iou090_a090

$PY -m py_compile tools/diagnose_trackbucket_suppression_v7.py \
  2>&1 | tee logs/stream4d_v7_c14_trackbucket_diagnostic_py_compile.log

$PY -m tools.diagnose_trackbucket_suppression_v7 \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --groups c11 c12 c13 \
  --configs "$CONFIGS" \
  --output-prefix outputs/audit/v7_trackbucket_suppression_gt_diagnostic_representative \
  2>&1 | tee logs/stream4d_v7_trackbucket_suppression_gt_diagnostic_representative.log
```

产物：

```text
outputs/audit/v7_trackbucket_suppression_gt_diagnostic_representative.json
outputs/audit/v7_trackbucket_suppression_gt_diagnostic_representative.csv
outputs/audit/v7_trackbucket_suppression_gt_diagnostic_representative.md
logs/stream4d_v7_trackbucket_suppression_gt_diagnostic_representative.log
```

### C14：strict-owned-posttrack reproduction

第一次 C14 导出遇到 object_dict 写入 blocker：

```text
OSError: [Errno 30] Read-only file system:
  data/scannet/processed/<scene>/output_Cropformer/object/stream4d_v7_c14_strict_owned_posttrack_probe5
```

处理：

```text
修改 Stream3D/tools/export_carrier_tracklet_graph_v7.py。
prediction npz 和 TMP pre_points 仍写原位置；
object_dict.npy 若不能写 processed object dir，则 fallback 到：
  outputs/v7_carrier_tracklet_graph/object_dicts/<config>/<scene>/object_dict.npy
summary 写入：
  object_dict_write_fallback
  object_dict_write_path
  object_dict_write_error
```

语法检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY -m py_compile tools/export_carrier_tracklet_graph_v7.py \
  2>&1 | tee logs/stream4d_v7_c14_object_dict_fallback_py_compile.log
```

C14 fixed full probe5 导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
DEBUG_ROOT=outputs/stream4d_v5_cache_96f_probe5
SEQ_LIST=splits/scannet_v6_probe5.txt
LOG=logs/stream4d_v7_c14_strict_owned_posttrack_probe5_fixed.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  printf '### C14 fixed %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m tools.export_carrier_tracklet_graph_v7 \
    --debug-root "$DEBUG_ROOT" \
    --seq-name "$scene" \
    --output-config stream4d_v7_c14_strict_owned_posttrack_probe5 \
    --support-mode core_owned_fringe_wta_posttrack \
    --min-shared-frames 3 \
    --min-positive-ratio 0.65 \
    --max-pair-distance-variance 0.005 \
    --max-component-carriers 180 \
    --min-mask-carriers 6 \
    --min-frame-mask-ratio 0.65 \
    2>&1 | tee -a "$LOG"
done < "$SEQ_LIST"
```

C14 evaluator：

```bash
CFG=stream4d_v7_c14_strict_owned_posttrack_probe5
$PY -m evaluation.evaluate \
  --pred_path data/prediction/${CFG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config $CFG \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CFG}_eval.log
```

C14 audit：

```bash
$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CFG" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c14_strict_owned_posttrack_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c14_strict_owned_posttrack_probe5.log

$PY -m tools.scan_reportable_configs \
  --root . \
  --configs "$CFG" \
  --output outputs/audit/v7_reportable_config_scan_c14_strict_owned_posttrack_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c14_strict_owned_posttrack_probe5.log
```

C14 结果：

```text
AP/AP50/AP25 =
  0.04249776562268287 / 0.17857097183073392 / 0.5404251478272147

phase0_pass=True
reportable scan:
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

注意：

```text
C14 与 C8P 是同参数复现。
5 个 scene npz sha256 与 C8P 完全一致，说明 C14 不是新方法收益。
```

### C15：strict posttrack merge

目的：

```text
测试 C8P/C14 是否主要被 posttrack over-merge 伤害。
C15 保持 C8P/C14 window formation 不变，只收紧 scene-link merge：
  --scene-link-min-shared-carriers 32
  --scene-link-min-overlap-ratio 0.50
```

C15 full probe5 导出：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python
DEBUG_ROOT=outputs/stream4d_v5_cache_96f_probe5
SEQ_LIST=splits/scannet_v6_probe5.txt
LOG=logs/stream4d_v7_c15_strict_posttrack_merge_probe5.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  printf '### C15 %s\n' "$scene" | tee -a "$LOG"
  "$PY" -m tools.export_carrier_tracklet_graph_v7 \
    --debug-root "$DEBUG_ROOT" \
    --seq-name "$scene" \
    --output-config stream4d_v7_c15_strict_posttrack_merge_probe5 \
    --support-mode core_owned_fringe_wta_posttrack \
    --min-shared-frames 3 \
    --min-positive-ratio 0.65 \
    --max-pair-distance-variance 0.005 \
    --max-component-carriers 180 \
    --min-mask-carriers 6 \
    --min-frame-mask-ratio 0.65 \
    --scene-link-min-shared-carriers 32 \
    --scene-link-min-overlap-ratio 0.50 \
    2>&1 | tee -a "$LOG"
done < "$SEQ_LIST"
```

C15 evaluator：

```bash
CFG=stream4d_v7_c15_strict_posttrack_merge_probe5
$PY -m evaluation.evaluate \
  --pred_path data/prediction/${CFG}_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --output_file data/evaluation/scannet/${CFG}_class_agnostic.txt \
  --tmp_root data/TMP \
  --tmp_config $CFG \
  --no_class \
  --require-manifest \
  2>&1 | tee logs/${CFG}_eval.log
```

C15 audit：

```bash
$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CFG" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c15_strict_posttrack_merge_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c15_strict_posttrack_merge_probe5.log

$PY -m tools.scan_reportable_configs \
  --root . \
  --configs "$CFG" \
  --output outputs/audit/v7_reportable_config_scan_c15_strict_posttrack_merge_probe5.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_reportable_config_scan_c15_strict_posttrack_merge_probe5.log
```

C15 结果：

```text
AP/AP50/AP25 =
  0.03494669836186661 / 0.14738590648397554 / 0.5024238943440352

phase0_pass=True
reportable scan:
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

### r6 audit fallback verification

修复 metric integrity fallback 后重新审计 C14/C15：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY -m py_compile tools/verify_stream4d_metric_integrity.py \
  2>&1 | tee logs/stream4d_v7_r6_metric_integrity_fallback_py_compile.log

CFG=stream4d_v7_c14_strict_owned_posttrack_probe5
$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CFG" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c14_strict_owned_posttrack_probe5_r6.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c14_strict_owned_posttrack_probe5_r6.log

CFG=stream4d_v7_c15_strict_posttrack_merge_probe5
$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root ../stream4d_v6_code_review_packet_20260608_2121/Stream3D \
  --current-root . \
  --configs "$CFG" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v7_metric_integrity_c15_strict_posttrack_merge_probe5_r6.md \
  --require-manifest \
  2>&1 | tee logs/stream4d_v7_metric_integrity_c15_strict_posttrack_merge_probe5_r6.log
```

r6 audit fallback 结果：

```text
C14:
  phase0_pass=True
  alignment_checked_scenes=5
  object_dict_path_source=summary_fallback for 5/5 scenes
  object_dict_pred_alignment_mean_iou=1.0
  object_dict_pred_alignment_min_iou=1.0
  object_dict_pred_alignment_failed_instances=0

C15:
  phase0_pass=True
  alignment_checked_scenes=5
  object_dict_path_source=summary_fallback for 5/5 scenes
  object_dict_pred_alignment_mean_iou=1.0
  object_dict_pred_alignment_min_iou=1.0
  object_dict_pred_alignment_failed_instances=0
```

## r6 最终验证与打包（2026-06-09）

最终验证：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/bin/python

$PY -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py \
  2>&1 | tee logs/stream4d_v7_r6_py_compile.log

$PY -m unittest tests.test_stream4d_protocol_fixes \
  2>&1 | tee logs/stream4d_v7_r6_unit_tests.log

$PY -c "import importlib; mods=['stream4d.reliable_densifier','tools.export_carrier_tracklet_graph_v7','tools.export_d4rt_geometry_degradation_v7','tools.export_trackbucket_dense_variants_v7','tools.diagnose_trackbucket_suppression_v7','tools.verify_stream4d_metric_integrity','tools.scan_reportable_configs']; [importlib.import_module(m) for m in mods]; print('r6 import smoke OK')" \
  2>&1 | tee logs/stream4d_v7_r6_import_smoke.log
```

验证结果：

```text
py_compile: pass
unit tests:
  Ran 13 tests in 0.107s
  OK
import smoke:
  r6 import smoke OK
```

同步文档：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
cp docs/stream4d_v7_执行日志.md Stream3D/docs/stream4d_v7_执行日志.md
cp docs/stream4d_v7_实验结果复盘.md Stream3D/docs/stream4d_v7_实验结果复盘.md
```

打包命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
PACKET=stream4d_v7_code_review_packet_20260609_030044_r6
STATUS=Stream3D/outputs/audit/git_status_stream4d_v7_packet_r6.txt
DIFF=Stream3D/outputs/audit/git_diff_stream4d_v7_packet_r6.patch

git status --short > "$STATUS"
: > "$DIFF"

git diff --no-ext-diff -- \
  Stream3D/evaluation/evaluate.py \
  Stream3D/stream4d/reliable_densifier.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/make_union_prepoints_config.py \
  Stream3D/tools/diagnose_prediction_quality.py \
  Stream3D/tools/export_carrier_tracklet_graph_v7.py \
  Stream3D/tools/export_d4rt_geometry_degradation_v7.py \
  Stream3D/tools/export_trackbucket_dense_variants_v7.py \
  Stream3D/tools/diagnose_trackbucket_suppression_v7.py \
  Stream3D/tools/summarize_v7_gap_matrix.py \
  Stream3D/tools/verify_stream4d_metric_integrity.py \
  Stream3D/tests/test_stream4d_protocol_fixes.py \
  docs/stream4d_v7_执行日志.md \
  docs/stream4d_v7_实验结果复盘.md \
  >> "$DIFF"

git diff --no-index -- /dev/null Stream3D/tools/export_trackbucket_dense_variants_v7.py \
  >> "$DIFF" 2>/dev/null || true
git diff --no-index -- /dev/null Stream3D/tools/diagnose_trackbucket_suppression_v7.py \
  >> "$DIFF" 2>/dev/null || true

{
  find docs -maxdepth 1 -type f -name 'stream4d_v7*'
  find Stream3D/docs -maxdepth 1 -type f -name 'stream4d_v7*'
  find Stream3D/evaluation -maxdepth 1 -type f -name '*.py'
  find Stream3D/stream4d -maxdepth 1 -type f -name '*.py'
  find Stream3D/tools -maxdepth 1 -type f -name '*.py'
  find Stream3D/tests -maxdepth 1 -type f -name '*.py'
  find Stream3D/logs -maxdepth 1 -type f -name '*v7*.log'
  find Stream3D/outputs/audit -maxdepth 1 -type f \( -name '*v7*' -o -name 'git_*stream4d_v7_packet_r6*' \)
  find Stream3D/outputs -maxdepth 2 -type f -path '*/stream4d_v7_*/*' -name '*.json'
  find Stream3D/outputs -maxdepth 2 -type f -path '*/v7_*/*' -name '*.json'
  find Stream3D/outputs/v7_carrier_tracklet_graph/object_dicts -type f \( -path '*stream4d_v7_c14_strict_owned_posttrack_probe5*' -o -path '*stream4d_v7_c15_strict_posttrack_merge_probe5*' \) -name '*.npy'
  find Stream3D/data/evaluation/scannet -maxdepth 1 -type f \( -name '*v7*.txt' -o -name 'stream4d_v6_e4_probe5_objcomp_m670_g101*_class_agnostic.txt' \)
  find Stream3D/data/prediction -maxdepth 2 -type f -path '*/stream4d_v7*' -name 'config_manifest.json'
  find Stream3D/data/prediction -maxdepth 2 -type f \( -path '*/stream4d_v7_c8p_posttrack_owned_wta_probe5_class_agnostic/*' -o -path '*/stream4d_v7_c14_strict_owned_posttrack_probe5_class_agnostic/*' -o -path '*/stream4d_v7_c15_strict_posttrack_merge_probe5_class_agnostic/*' \) -name '*.npz'
  find Stream3D/data/TMP -maxdepth 2 -type f \( -path '*/stream4d_v7_c14_strict_owned_posttrack_probe5/*' -o -path '*/stream4d_v7_c15_strict_posttrack_merge_probe5/*' \) -name '*pre_points.npy'
} | sort -u > ${PACKET}_filelist.txt

zip -q ${PACKET}.zip -@ < ${PACKET}_filelist.txt
sha256sum ${PACKET}.zip > ${PACKET}.sha256
printf '%s.zip\n' "$PACKET" > stream4d_v7_latest_packet_name.txt
unzip -t ${PACKET}.zip | tail -n 5
sha256sum ${PACKET}.zip
wc -l ${PACKET}_filelist.txt
ls -lh ${PACKET}.zip
```

打包结果：

```text
packet:
  stream4d_v7_code_review_packet_20260609_030044_r6.zip

sha256:
  see stream4d_v7_code_review_packet_20260609_030044_r6.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

filelist:
  stream4d_v7_code_review_packet_20260609_030044_r6_filelist.txt

zip test:
  No errors detected in compressed data of stream4d_v7_code_review_packet_20260609_030044_r6.zip.

file count:
  669
```
