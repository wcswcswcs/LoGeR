# Stream4D v9 unified eval and method 执行日志

日期：2026-06-09（Asia/Singapore）  
计划文件：`docs/stream4d_v9_unified_eval_and_method_plan_for_codex.md`  
结果复盘：`docs/stream4d_v9_实验结果复盘.md`

本日志只记录本轮实际执行过的命令、修改和输出位置。没有跑出的实验不补写成结果；失败和修复按实际日志记录。

## 环境

```text
repo root: /mnt/data/users/chengshun.wang/pjs/LoGeR
Stream3D root: /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
python: /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CUDA_VISIBLE_DEVICES: 6,7
MPLCONFIGDIR: /mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache
split: Stream3D/splits/scannet_v6_probe5.txt
```

## 本轮代码和脚本修改

| 文件 | 修改 | 原因 |
|---|---|---|
| `Stream3D/tools/evaluate_cross_prepoints.py` | 新增 `--eval-policy`，并把 `eval_policy`、`prediction_config`、`pre_points_config` 写入 manifest | v9 unified eval 需要明确每个 cross-support row 的评估策略，避免 own-support 与 diagnostic 混淆 |
| `Stream3D/tools/scan_reportable_configs.py` | 扫描并汇总 `eval_policy`；新增 `--require-eval-policy` | reportable scan 必须能发现缺失评估策略的 method artifact |
| `Stream3D/tools/summarize_v9_unified_eval.py` | 新增 unified matrix 汇总工具，读取官方 evaluator AP 和 support/IoU diagnostic | v9 需要同一表中比较 own support、Stream3D on method support、method on Stream3D support |
| `Stream3D/tools/export_v9_b1_controls.py` | 新增 B1 control exporter：no-track、shuffle、random、area、maskcount | Phase2 需要验证 D4RT/track ownership 是否优于简单面积/数量/随机控制 |
| `Stream3D/tools/split_core_fringe_prediction.py` | 新增 `--eval-policy` 并写入 manifest | Phase4 初跑发现 O1/O2/O3 缺 `eval_policy`，修复后重跑 |
| `Stream3D/scripts/v9_day0_matrix_probe5.json` | 新增 Day0 unified matrix row spec | 复现 Day0/S0/S1/S2/controls 统一评估 |
| `Stream3D/scripts/reproduce_v9_day0.sh` | 新增 Day0 复现脚本 | 统一执行 B1 gap matrix、controls、audit |
| `Stream3D/scripts/v9_phase4_matrix_probe5.json` | 新增 Phase4 matrix row spec | 复现 O1/O2/O3 core/fringe 统一评估 |
| `Stream3D/scripts/reproduce_v9_phase4.sh` | 新增 Phase4 复现脚本 | 统一执行 core-only / radius fringe / WTA-negative fringe |
| `Stream3D/data/prediction/stream4d_v8_b1_surfacelet_singlemask_probe5_class_agnostic/config_manifest.json` | 补充 `eval_policy=own_recompute_paper_style` | 既有 B1 artifact 进入 v9 reportable scan 时缺 eval policy |
| `Stream3D/data/TMP/stream4d_v8_b1_surfacelet_singlemask_probe5/config_manifest.json` | 同上 | 让 pred/TMP manifest 一致 |

## Day0 unified eval

执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
./scripts/reproduce_v9_day0.sh
```

脚本内主要步骤：

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/evaluate_cross_prepoints.py \
  tools/export_v9_b1_controls.py \
  tools/summarize_v9_unified_eval.py \
  tools/scan_reportable_configs.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -c \
  "import importlib; mods=['tools.evaluate_cross_prepoints','tools.export_v9_b1_controls','tools.summarize_v9_unified_eval','tools.scan_reportable_configs']; [importlib.import_module(m) for m in mods]; print('v9 import smoke OK')"

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

验证结果：

```text
py_compile: pass
import smoke: v9 import smoke OK
unittest: Ran 13 tests in 0.121s, OK
```

Day0 输出：

```text
Stream3D/outputs/audit/v9_day0/unified_eval_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_day0/reportable_config_scan_controls_probe5_r2.{json,csv,md}
Stream3D/outputs/audit/v9_day0/metric_integrity_controls_probe5.{json,md}
Stream3D/outputs/v9_b1_controls/*.json
Stream3D/data/evaluation/scannet/stream4d_v9_*_probe5_class_agnostic.txt
Stream3D/logs/stream4d_v9_*.log
```

Day0 初始 reportable scan 发现 B1 既有 manifest 缺 `eval_policy`。修复 B1 pred/TMP manifest 后重跑：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v9_b1_no_track_probe5,stream4d_v9_b1_shuffle_probe5,stream4d_v9_b1_random_same_count_s0_probe5,stream4d_v9_b1_area_same_count_probe5,stream4d_v9_b1_maskcount_same_count_probe5 \
  --output outputs/audit/v9_day0/reportable_config_scan_controls_probe5_r2.md \
  --require-manifest \
  --require-eval-policy \
  > logs/stream4d_v9_reportable_scan_controls_probe5_r2.log 2>&1
```

重跑结果：

```text
num_configs=6
num_configs_missing_manifest=0
num_configs_missing_eval_policy=0
num_oracle_configs=0
num_reportable_method_configs=6
num_suspicious_configs=0
num_uses_gt_and_method_result=0
```

## Dynamic Replica v9 env check

执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.check_dynamic_replica_env \
  --data-root data/dynamic-replica/v2 \
  --split valid \
  --output outputs/audit/v9_day0/dynamic_replica_env_v9.md \
  > logs/stream4d_v9_dynamic_replica_env.log 2>&1
```

输出：

```text
Stream3D/outputs/audit/v9_day0/dynamic_replica_env_v9.{json,md}
usable_scene_count=0
can_report_official_instance_tracking=False
can_report_d4rt_trajectory_metrics=False
can_report_only_qualitative_consistency=False
```

## Phase4 core/fringe

语法检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
chmod +x Stream3D/scripts/reproduce_v9_phase4.sh
bash -n Stream3D/scripts/reproduce_v9_phase4.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/split_core_fringe_prediction.py
```

第一次执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
./scripts/reproduce_v9_phase4.sh
```

第一次结果：

```text
评估和 matrix 已产出，但脚本最后 reportable scan 退出码 4。
原因：O1/O2/O3 manifest 均缺 eval_policy。
日志：Stream3D/logs/stream4d_v9_phase4_reportable_scan.log
scan summary: num_configs_missing_eval_policy=3
```

修复：

```text
给 tools/split_core_fringe_prediction.py 增加 --eval-policy，并把该字段写入 manifest。
给 scripts/reproduce_v9_phase4.sh 的 O1/O2/O3 export 命令分别传入：
  own_recompute_core_only
  own_recompute_core_radius_fringe
  own_recompute_wta_negative_fringe
```

修复后重跑：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
./scripts/reproduce_v9_phase4.sh
```

结果：

```text
v9 Phase4 done
reportable scan:
  num_configs=3
  num_configs_missing_manifest=0
  num_configs_missing_eval_policy=0
  num_oracle_configs=0
  num_reportable_method_configs=3
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
```

Phase4 追加 integrity audit：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs stream4d_v9_o1_b1_core_only_probe5,stream4d_v9_o2_b1_core_radius_fringe_probe5,stream4d_v9_o3_b1_wta_negative_fringe_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v9_phase4/metric_integrity_phase4_probe5.md \
  --require-manifest \
  > logs/stream4d_v9_phase4_metric_integrity_probe5.log 2>&1
```

输出：

```text
Stream3D/outputs/audit/v9_phase4/phase4_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_phase4/reportable_config_scan_phase4_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_phase4/metric_integrity_phase4_probe5.{json,md}
Stream3D/outputs/v9_core_fringe/*_summary.json
Stream3D/data/evaluation/scannet/stream4d_v9_o*_probe5_class_agnostic.txt
```

## S3 / G1 parent support blocker

计划中的 S3 = B1 parent/G1 support 不能直接进入当前 ScanNet mesh evaluator。检查 G1 carrier artifact：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
from pathlib import Path
import numpy as np
p = Path('Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger/scene0050_00/carriers_window000.npz')
with np.load(p) as data:
    print('keys', sorted(data.files))
    for k in sorted(data.files):
        arr = data[k]
        print(k, arr.shape, arr.dtype)
PY
```

实际字段：

```text
carrier_id (16384,) int64
confidence_prob (16, 16384) float32
src_frame (16384,) int64
src_frame_global (16384,) int64
src_mask_id (16384,) int64
src_uv (16384, 2) float32
src_xy (16384, 2) int64
uv_pred (16, 16384, 2) float32
valid (16, 16384) bool
visibility_prob (16, 16384) float32
xyz_ref (16, 16384, 3) float32
```

结论：

```text
G1 carrier 是 D4RT surfel/query artifact，不是 data/TMP/<config>/<scene>_pre_points.npy 格式的 ScanNet mesh vertex support。
本轮不伪造 P0 on S3 / B1 on S3 AP。
后续若要跑 S3，需要新增 surfel-to-mesh support materializer，并单独审计 Sim3/nearest-neighbor 投影误差。
```

## 复现入口

```bash
# Day0 unified eval + controls
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
./scripts/reproduce_v9_day0.sh

# Phase4 core/fringe
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
./scripts/reproduce_v9_phase4.sh

# Dynamic Replica env check
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.check_dynamic_replica_env \
  --data-root data/dynamic-replica/v2 \
  --split valid \
  --output outputs/audit/v9_day0/dynamic_replica_env_v9.md
```

## 主要日志文件

```text
Stream3D/logs/stream4d_v9_py_compile.log
Stream3D/logs/stream4d_v9_import_smoke.log
Stream3D/logs/stream4d_v9_unittest.log
Stream3D/logs/stream4d_v9_unified_eval_matrix_probe5.log
Stream3D/logs/stream4d_v9_reportable_scan_controls_probe5.log
Stream3D/logs/stream4d_v9_reportable_scan_controls_probe5_r2.log
Stream3D/logs/stream4d_v9_metric_integrity_controls_probe5.log
Stream3D/logs/stream4d_v9_dynamic_replica_env.log
Stream3D/logs/stream4d_v9_phase4_py_compile.log
Stream3D/logs/stream4d_v9_phase4_matrix_probe5.log
Stream3D/logs/stream4d_v9_phase4_reportable_scan.log
Stream3D/logs/stream4d_v9_phase4_metric_integrity_probe5.log
Stream3D/logs/stream4d_v9_o1_core_only_export.log
Stream3D/logs/stream4d_v9_o2_core_radius_fringe_export.log
Stream3D/logs/stream4d_v9_o3_wta_negative_fringe_export.log
```

## 环境快照

```text
Stream3D/pip_freeze_v9_loger.txt
```

## 追加执行：cross-support top priority

用户追加要求：

```text
优先解决 cross-support，这是 top priority。
```

执行原则：

```text
不把 own-support 高分当成 cross-support 成功。
每个新增方法/诊断都要跑 unified matrix，至少包含 method own、Stream3D-on-method-support、method-on-S0/S1。
诊断使用 Stream3D/scannet candidate 时必须标为 diagnostic-only，不作为 method result。
```

### X1：S4 / Phase4b support union 与 scene-fringe

目的：

```text
检查 S4 = union(S0 Stream3D support, S2 B1 support) 是否能缓解 support mismatch。
检查 O4 scene-wide fringe 是否能在扩大 support 时保住 AP/AP50。
```

首次误操作：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
chmod Stream3D/scripts/reproduce_v9_s4_phase4b.sh
```

结果：

```text
失败，路径写错：
chmod: cannot access 'Stream3D/scripts/reproduce_v9_s4_phase4b.sh': No such file or directory
```

正确执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
chmod +x scripts/reproduce_v9_s4_phase4b.sh
./scripts/reproduce_v9_s4_phase4b.sh
```

输出：

```text
v9 S4/Phase4b done
Stream3D/outputs/audit/v9_s4_phase4b/s4_phase4b_matrix_probe5.{json,csv,md}
Stream3D/outputs/v9_support/stream4d_v9_s4_union_s0_b1_probe5_summary.{json,csv,md}
Stream3D/outputs/v9_core_fringe/stream4d_v9_o4_b1_scene_fringe_r002_probe5_summary.json
Stream3D/outputs/v9_core_fringe/stream4d_v9_o4_b1_scene_fringe_r005_probe5_summary.json
```

### X2：CropFormer higher-frequency mask blocker

目的：

```text
按 Phase5 推荐方向尝试增加 mask frequency，先检查本地 CropFormer 是否可运行。
```

命令 1：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python third_party/Cropformer.py --help \
  > logs/stream4d_v9_phase5_cropformer_help.log 2>&1 || true
```

结果：

```text
ModuleNotFoundError: No module named 'mask2former'
```

命令 2：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PYTHONPATH=third_party/detectron2/projects/CropFormer:$PYTHONPATH \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python third_party/Cropformer.py --help \
  > logs/stream4d_v9_phase5_cropformer_help_py_path.log 2>&1 || true
```

结果：

```text
ImportError: cannot import name '_C' from 'detectron2'
```

结论：

```text
本地 detectron2 C extension 不可用，本轮不能可靠生成更高频 CropFormer masks。
不伪造新增 mask frames。
```

### X3：D4RT mask propagation diagnostic

目的：

```text
在不能新增 2D masks 的情况下，按计划尝试 D4RT propagation 方向；
该实验只诊断 temporal measurement coverage，不产 AP。
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v9_d4rt_mask_propagation \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --carrier-run stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --visibility-threshold 0.5 \
  --output-prefix outputs/audit/v9_phase5/d4rt_mask_propagation_probe5_vis05 \
  > logs/stream4d_v9_phase5_d4rt_mask_propagation_probe5_vis05.log 2>&1
```

输出：

```text
Stream3D/outputs/audit/v9_phase5/d4rt_mask_propagation_probe5_vis05.{json,csv,md}
```

### X4：O5 D4RT propagated slots cross-support

目的：

```text
把 propagated mask slots materialize 成 method prototype，直接跑 cross-support matrix。
```

执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
bash -n scripts/reproduce_v9_phase5_o5_cross_support.sh
chmod +x scripts/reproduce_v9_phase5_o5_cross_support.sh
./scripts/reproduce_v9_phase5_o5_cross_support.sh
```

输出：

```text
v9 Phase5 O5 cross-support done
Stream3D/outputs/audit/v9_phase5/o5_cross_support_matrix_probe5.{json,csv,md}
Stream3D/outputs/v9_propagated_slot_field/stream4d_v9_o5_d4rt_propagated_slot_probe5_summary.{json,csv,md}
Stream3D/outputs/audit/v9_phase5/reportable_config_scan_o5_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_phase5/metric_integrity_o5_probe5.{json,md}
```

### X5：O6 support completion

目的：

```text
直接修复 method-on-S0 崩溃：以 O1 clean core 为输入，把 target support S0 点分配给最近 object core。
测试 r=0.02/0.05/0.10 和 all。
```

新增文件：

```text
Stream3D/tools/complete_prediction_to_support.py
Stream3D/scripts/reproduce_v9_o6_support_completion_cross_support.sh
Stream3D/scripts/v9_o6_support_completion_matrix_probe5.json
```

执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile Stream3D/tools/complete_prediction_to_support.py
bash -n Stream3D/scripts/reproduce_v9_o6_support_completion_cross_support.sh
cat Stream3D/scripts/v9_o6_support_completion_matrix_probe5.json | python3 -m json.tool >/tmp/v9_o6_matrix_json_check.txt

cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
chmod +x scripts/reproduce_v9_o6_support_completion_cross_support.sh
./scripts/reproduce_v9_o6_support_completion_cross_support.sh
```

输出：

```text
v9 O6 support-completion cross-support done
Stream3D/outputs/audit/v9_o6_support_completion/o6_support_completion_matrix_probe5.{json,csv,md}
Stream3D/outputs/v9_support_completion/stream4d_v9_o6_o1_complete_s0_*_probe5_summary.{json,csv,md}
Stream3D/outputs/audit/v9_o6_support_completion/reportable_config_scan_o6_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o6_support_completion/metric_integrity_o6_probe5.{json,md}
```

### X6：O7 object birth recall

目的：

```text
O6 显示点级补全不够，转向 object recall。
降低 B1 min_carriers 到 8/4/2，测试更多 object slots 是否能修复 S0/S1。
```

修改：

```text
Stream3D/tools/export_v8_surfel_object_field.py 新增 --eval-policy，并写入 manifest。
```

执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/export_v8_surfel_object_field.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py
bash -n Stream3D/scripts/reproduce_v9_o7_birth_recall_cross_support.sh
cat Stream3D/scripts/v9_o7_birth_recall_matrix_probe5.json | python3 -m json.tool >/tmp/v9_o7_matrix_json_check.txt

cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
chmod +x scripts/reproduce_v9_o7_birth_recall_cross_support.sh
./scripts/reproduce_v9_o7_birth_recall_cross_support.sh
```

输出：

```text
v9 O7 birth-recall cross-support done
Stream3D/outputs/audit/v9_o7_birth_recall/o7_birth_recall_matrix_probe5.{json,csv,md}
Stream3D/outputs/v9_birth_recall/stream4d_v9_o7_b1_recall_mc*_probe5_summary.{json,csv,md}
Stream3D/outputs/audit/v9_o7_birth_recall/reportable_config_scan_o7_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o7_birth_recall/metric_integrity_o7_probe5.{json,md}
```

### X7：O8 slot-guided full-scene candidate diagnostic / obs-bank attempt

目的：

```text
诊断 cross-support 是否主要缺 full-scene candidates。
scannet candidate upper bound 必须标 diagnostic-only。
obs-bank variants 使用非 GT v5 observation-bank candidates，测试是否能替代 Stream3D candidates。
```

修改：

```text
Stream3D/tools/slotwise_candidate_select.py 新增 manifest、--eval-policy、--diagnostic-only。
```

第一次执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
chmod +x scripts/reproduce_v9_o8_slot_candidate_cross_support.sh
./scripts/reproduce_v9_o8_slot_candidate_cross_support.sh
```

失败：

```text
logs/stream4d_v9_o8_stream4d_v9_o8_scannet_slot_upper_probe5_eval.log:
ValueError: Refusing to evaluate diagnostic-only manifest without --allow-oracle-eval.
```

修复：

```text
在 reproduce_v9_o8_slot_candidate_cross_support.sh 的 eval_own 中添加 --allow-oracle-eval。
原因：scannet candidate upper bound 是 diagnostic-only，必须显式允许 evaluator 评估。
```

重新执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
./scripts/reproduce_v9_o8_slot_candidate_cross_support.sh
```

输出：

```text
v9 O8 slot-candidate cross-support done
Stream3D/outputs/audit/v9_o8_slot_candidate/o8_slot_candidate_matrix_probe5.{json,csv,md}
Stream3D/outputs/v9_slot_candidate/stream4d_v9_o8_*_probe5_summary.json
Stream3D/outputs/audit/v9_o8_slot_candidate/reportable_config_scan_o8_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o8_slot_candidate/metric_integrity_o8_probe5.{json,md}
```

### X8：O9 multi-window D4RT attempt

目的：

```text
O7/O8 说明单 16f clip 覆盖太小。
按 Phase5/Phase4 思路跑真正 multi-window D4RT：grid16，连续帧，前 96 帧，每 20 帧一个 16f window。
这不是 full scan，只是验证更多 D4RT windows/mask frames 是否改善 cross-support。
```

新增：

```text
Stream3D/scripts/reproduce_v9_o9_multwindow_cross_support.sh
Stream3D/scripts/v9_o9_multwindow_matrix_probe5.json
```

执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/export_d4rt_grid_surfel_field_v8.py \
  Stream3D/tools/export_v8_surfel_object_field.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py
bash -n Stream3D/scripts/reproduce_v9_o9_multwindow_cross_support.sh
cat Stream3D/scripts/v9_o9_multwindow_matrix_probe5.json | python3 -m json.tool >/tmp/v9_o9_matrix_json_check.txt

cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
chmod +x scripts/reproduce_v9_o9_multwindow_cross_support.sh
./scripts/reproduce_v9_o9_multwindow_cross_support.sh
```

输出：

```text
v9 O9 multi-window cross-support done
Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v9_g1_grid16m002_probe5_96f_stride20_loger/summary.{json,csv,md}
Stream3D/outputs/v9_multwindow/stream4d_v9_o9_b1_multwin96_grid16_mc08_probe5_summary.{json,csv,md}
Stream3D/outputs/audit/v9_o9_multwindow/o9_multwindow_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o9_multwindow/reportable_config_scan_o9_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o9_multwindow/metric_integrity_o9_probe5.{json,md}
```

O9 过程中 OpenCV 对连续帧缺失 mask 输出大量 warning，例如：

```text
imread_('data/scannet/processed/<scene>/output_Cropformer/mask/1.png'): can't open/read file
```

解释：

```text
这些 warning 符合预期；本轮使用 --allow-missing-masks，缺失连续帧 mask 用 0 mask 占位。
真正可用的 mask frames 仍是 0/10/20/.../90。
```

### X9：O9 多窗口 rate summary 修复

问题：

```text
export_v8_surfel_object_field.py 在多窗口场景中把 positive_mask_sample_rate 按窗口相加，
导致 O9 summary 出现 >1 的无意义 rate。
AP/evaluator 不受该字段影响，但复盘不能保留误导性诊断。
```

修复：

```text
将 scene-level positive_mask_sample_rate 改为 sum(positive_mask_samples) / sum(valid_visible_samples_on_mask_frames)。
```

重跑命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
O9_CONFIG=stream4d_v9_o9_b1_multwin96_grid16_mc08_probe5
G1_RUN=stream4d_v9_g1_grid16m002_probe5_96f_stride20_loger

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/export_v8_surfel_object_field.py \
  > logs/stream4d_v9_o9_export_rate_fix_py_compile.log 2>&1

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.export_v8_surfel_object_field \
  --debug-root outputs/v8_d4rt_grid_surfel_field/${G1_RUN} \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config $O9_CONFIG \
  --prototype-direction B_surfacelet_singlemask \
  --min-observations 1 \
  --max-observations 1 \
  --min-carriers 8 \
  --min-owned-masks 1 \
  --max-masks-per-object 1 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 50000 \
  --min-points-per-object 20 \
  --summary-root outputs/v9_multwindow \
  --eval-policy own_recompute_d4rt_multwindow96_grid16_mc08 \
  > logs/stream4d_v9_o9_export_rate_fix.log 2>&1

# 后续只重跑 O9 own/cross eval、matrix、reportable scan、metric integrity；
# 详见 logs/stream4d_v9_o9_*_rate_fix.log。
```

修复后检查：

```text
O9 positive_mask_sample_rate_mean = 0.987537692406441
O9 AP/AP50/AP25 未改变：0.040167 / 0.124304 / 0.294343
```

### X10：最终检查

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/complete_prediction_to_support.py \
  Stream3D/tools/export_v8_surfel_object_field.py \
  Stream3D/tools/slotwise_candidate_select.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/export_v9_propagated_slot_field.py \
  Stream3D/tools/diagnose_v9_d4rt_mask_propagation.py \
  Stream3D/tools/build_union_prepoints.py \
  Stream3D/tools/split_core_fringe_prediction.py

cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
bash -n \
  scripts/reproduce_v9_o6_support_completion_cross_support.sh \
  scripts/reproduce_v9_o7_birth_recall_cross_support.sh \
  scripts/reproduce_v9_o8_slot_candidate_cross_support.sh \
  scripts/reproduce_v9_o9_multwindow_cross_support.sh \
  scripts/reproduce_v9_s4_phase4b.sh \
  scripts/reproduce_v9_phase5_o5_cross_support.sh

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests
```

结果：

```text
py_compile: pass
bash -n: pass
unit tests: Ran 13 tests in 0.124s, OK
```

### X11：cross-support 审计包

命令摘要：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
BASE=stream4d_v9_code_audit_packet_20260609_0652_cross_support
git status --short > ${BASE}_git_status.txt
git diff -- Stream3D/stream4d Stream3D/tools Stream3D/evaluation Stream3D/tests Stream3D/scripts \
  docs/stream4d_v9_执行日志.md docs/stream4d_v9_实验结果复盘.md > ${BASE}_git_diff.patch
# filelist 包含代码、scripts、docs、outputs/audit、summary、evaluation txt、manifest、TMP、logs、env。
zip -q -r ${BASE}.zip -@ < ${BASE}_filelist.txt
sha256sum ${BASE}.zip > ${BASE}.sha256
unzip -t ${BASE}.zip > ${BASE}_ziptest.log
```

结果：

```text
packet: stream4d_v9_code_audit_packet_20260609_0652_cross_support.zip
sha256: 6e842bdd773097ffd9fc04800e65f9ff68a7a4d69c183ac0a492b128401a9b76
filelist: stream4d_v9_code_audit_packet_20260609_0652_cross_support_filelist.txt
file count: 1007
zip test: No errors detected in compressed data
```

### X20：最终索引

```text
latest executed repair attempts:
  O11 obs-bank overlap suppression
  O12 O10/O11 fusion + overlap suppression
  O13 O12 support completion to S0

latest final checks:
  Stream3D/logs/stream4d_v9_o13_final_py_compile.log
  Stream3D/logs/stream4d_v9_o13_final_bash_n.log
  Stream3D/logs/stream4d_v9_o13_final_unit_tests.log

latest final packet:
  stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.zip

latest final sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.sha256

latest final file count:
  1250

latest final conclusion:
  cross-support top priority not solved.
  strongest own-support method row = O12 fused overlap 0.50, AP/AP50/AP25 = 0.177930 / 0.380577 / 0.643669.
  S0 remains failed: O12 on S0 = 0.000301 / 0.001498 / 0.026504.
```

### X33：真实文件末尾最新索引（O28 final）

```text
latest executed repair attempts after O13:
  O14 full-span grid8 ws100
  O15/O16/O18 overlap merge threshold sweeps
  O17 object competition/ranking rescue
  O19 score calibration
  O20/O22 full-span window-density ws50/ws25
  O24 mask-aware query ws50
  O25 self-discovered boundary refine
  O26 boundary-refine score calibration
  O27 self-discovered silhouette score
  O28 target-support-aware diagnostic rank

latest final checks:
  Stream3D/logs/stream4d_v9_o28_final_py_compile.log
  Stream3D/logs/stream4d_v9_o28_final_bash_n.log
  Stream3D/logs/stream4d_v9_o28_final_unit_tests.log

latest final packet:
  stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.zip

latest final sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.sha256

latest final file count:
  1331

latest final conclusion:
  cross-support top priority still not solved.
  strongest reportable method-on-S0 AP/AP50/AP25 = O26 inside050 logarea on S0:
    0.021517 / 0.100351 / 0.291694
  strongest reportable AP25 on S0 = O25 inside050 on S0:
    0.021388 / 0.100300 / 0.292631
  strongest diagnostic-only S0-aware row = O28 S0-aware rank diagnostic on S0:
    0.023637 / 0.095270 / 0.267036
  P0 Stream3D on S0 reference:
    0.235730 / 0.414306 / 0.537786
```

### X44：真实 EOF 最新索引（O38 final）

```text
latest executed repair attempts after O28:
  O29/O30 scene object memory + score-exclusive/refine
  O31 memory exclusivity ablation
  O32 score calibration
  O33 boundary refine
  O34 O33 score calibration
  O35 memory update-mode ablation
  O36 O35 boundary refine
  O37 O35/O36 score calibration
  O38 memory threshold sweep + logarea
  O39 attempted then aborted before valid result

latest best reportable method-on-S0:
  O37 O35 new-points logarea on S0
  AP/AP50/AP25 = 0.032908 / 0.126690 / 0.418266

P0 Stream3D on S0 reference:
  AP/AP50/AP25 = 0.235730 / 0.414306 / 0.537786

latest final checks:
  Stream3D/logs/stream4d_v9_o38_final_py_compile.log
  Stream3D/logs/stream4d_v9_o38_final_bash_n.log
  Stream3D/logs/stream4d_v9_o38_final_unit_tests.log

latest final packet:
  stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final.zip

latest final sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final.sha256

latest final filelist:
  stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final_filelist.txt

latest final zip test:
  see sibling file stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final_ziptest.log

latest final file count:
  371

latest final conclusion:
  cross-support top priority still not solved.
  O37 improved over O26, but remains far below P0 Stream3D on S0.
```

### X35：O29-O30 scene object memory

目的：

```text
继续解决 cross-support top priority。
按 O28 后复盘建议，尝试 scene-level persistent object memory，
不再只做 support completion 或 target-support-aware diagnostic ranking。
```

核心新增文件：

```text
Stream3D/tools/scene_object_memory_from_predictions.py
Stream3D/scripts/reproduce_v9_o29_scene_memory_cross_support.sh
Stream3D/scripts/v9_o29_scene_memory_matrix_probe5.json
```

执行命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o29_scene_memory_cross_support.sh
```

输出：

```text
Stream3D/outputs/audit/v9_o29_scene_memory/o29_scene_memory_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o29_scene_memory/reportable_config_scan_o29_o30_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o29_scene_memory/metric_integrity_o29_o30_probe5.{json,md}
Stream3D/outputs/v9_scene_object_memory/stream4d_v9_o29_*_summary.json
Stream3D/data/evaluation/scannet/stream4d_v9_o29*_class_agnostic.txt
Stream3D/data/evaluation/scannet/stream4d_v9_o30*_class_agnostic.txt
```

结果摘要：

```text
reportable scan: 4 configs clean
metric integrity: phase0_pass=True
O30 c065 refine own = 0.092722 / 0.223007 / 0.512540
O30 c065 refine on S0 = 0.017962 / 0.079300 / 0.373374
结论：own/S1 有正信号，但 score-exclusive memory 对 S0 不够。
```

### X36：O31 memory exclusivity ablation

执行命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o31_memory_exclusive_cross_support.sh
```

输出：

```text
Stream3D/outputs/audit/v9_o31_memory_exclusive/o31_memory_exclusive_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o31_memory_exclusive/reportable_config_scan_o31_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o31_memory_exclusive/metric_integrity_o31_probe5.{json,md}
```

结果摘要：

```text
O31 c065 no-exclusive own = 0.065042 / 0.198975 / 0.500077
O31 c065 no-exclusive on S0 = 0.023980 / 0.102236 / 0.423419
O31 c065 no-exclusive on S1 = 0.075444 / 0.204144 / 0.491821
O31 c065 small-area-exclusive on S0 = 0.023499 / 0.096988 / 0.373138
结论：不做 exclusivity 反而更利于 S0，zero-conflict 不是目标。
```

### X37：O32/O33/O34 score + boundary refine

执行命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o32_o31_score_calibration_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o33_o31_boundary_refine_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o34_o33_score_calibration_cross_support.sh
```

输出：

```text
Stream3D/outputs/audit/v9_o32_o31_score_calibration/
Stream3D/outputs/audit/v9_o33_o31_boundary_refine/
Stream3D/outputs/audit/v9_o34_o33_score_calibration/
Stream3D/outputs/v9_score_calibration/stream4d_v9_o32*_summary.json
Stream3D/outputs/v9_score_calibration/stream4d_v9_o34*_summary.json
Stream3D/outputs/v9_boundary_refine/stream4d_v9_o33*_summary.json
```

结果摘要：

```text
O32 no-exclusive logarea on S0 = 0.023991 / 0.102364 / 0.424218
O33 inside035 on S0 = 0.025754 / 0.111061 / 0.412295
O34 inside035 logarea on S0 = 0.026925 / 0.117447 / 0.413656
结论：boundary + logarea 小幅提升 AP/AP50，但 AP25 不如 O32。
```

### X38：O35 memory update-mode ablation

执行命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o35_memory_update_mode_cross_support.sh
```

输出：

```text
Stream3D/outputs/audit/v9_o35_memory_update_mode/o35_memory_update_mode_matrix_probe5.{json,csv,md}
Stream3D/outputs/v9_scene_object_memory/stream4d_v9_o35*_summary.json
```

结果摘要：

```text
O35 keep-slot on S0 = 0.002759 / 0.015338 / 0.294378
O35 new-points-only own = 0.086196 / 0.228637 / 0.504178
O35 new-points-only on S0 = 0.031794 / 0.122303 / 0.413561
O35 new-points-only on S1 = 0.119207 / 0.272558 / 0.558640
结论：new_points_only 是本轮最清楚的 reportable 正向修复；keep_slot 为负结果。
```

### X39：O36/O37 O35 follow-up

执行命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o36_o35_boundary_refine_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o37_o35_o36_score_calibration_cross_support.sh
```

输出：

```text
Stream3D/outputs/audit/v9_o36_o35_boundary_refine/
Stream3D/outputs/audit/v9_o37_o35_o36_score_calibration/
Stream3D/outputs/v9_boundary_refine/stream4d_v9_o36*_summary.json
Stream3D/outputs/v9_score_calibration/stream4d_v9_o37*_summary.json
```

结果摘要：

```text
O36 inside050 own = 0.112869 / 0.273510 / 0.525436
O36 inside050 on S0 = 0.029340 / 0.113117 / 0.398909
O36 inside035 on S0 = 0.029107 / 0.118791 / 0.398909
O37 O35 new-points logarea own = 0.088577 / 0.239542 / 0.511839
O37 O35 new-points logarea on S0 = 0.032908 / 0.126690 / 0.418266
O37 O35 new-points logarea on S1 = 0.108012 / 0.260753 / 0.570863
结论：O37 是本轮 best reportable S0；O36 boundary refine own-support 变强但 S0 降。
```

### X40：O38 threshold sweep + O39 abort

执行命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o38_memory_threshold_logarea_cross_support.sh
```

输出：

```text
Stream3D/outputs/audit/v9_o38_memory_threshold_logarea/
Stream3D/outputs/v9_scene_object_memory/stream4d_v9_o38*_summary.json
Stream3D/outputs/v9_score_calibration/stream4d_v9_o38*_summary.json
```

结果摘要：

```text
O38 c055 logarea on S0 = 0.033012 / 0.123089 / 0.392066
O38 c075split logarea on S0 = 0.025503 / 0.106879 / 0.376944
O38 c075split logarea on S1 = 0.126225 / 0.305517 / 0.559456
结论：c055 只微升 AP，不升 AP50/AP25；c075split 的 S1 和 best-IoU diagnostics 好，但 S0 AP 变差。
```

O39 attempted but aborted by user-requested wrap-up:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o39_c075split_exclusive_cross_support.sh
```

状态：

```text
O39 终止在 scene_object_memory_from_predictions export 阶段。
log: Stream3D/logs/stream4d_v9_o39_stream4d_v9_o39_o22_memory_c075split_smallarea_probe5_export.log
该 log size=0，未写出 summary/eval/matrix。
O39 不进入有效结果表。
```

终止命令：

```bash
pgrep -af 'stream4d_v9_o39|tools.scene_object_memory_from_predictions.*o39|tools.rescore_prediction_scores.*o39|evaluate_cross_prepoints.*o39|reproduce_v9_o39'
kill 2459455 2459446 2>/dev/null || true
```

### X41：O29-O38 final checks

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -m py_compile \
  tools/scene_object_memory_from_predictions.py \
  tools/rescore_prediction_scores.py \
  tools/self_discovered_boundary_refine.py \
  tools/summarize_v9_unified_eval.py \
  tools/evaluate_cross_prepoints.py \
  tools/scan_reportable_configs.py \
  tools/verify_stream4d_metric_integrity.py \
  > logs/stream4d_v9_o38_final_py_compile.log 2>&1

bash -n \
  scripts/reproduce_v9_o29_scene_memory_cross_support.sh \
  scripts/reproduce_v9_o31_memory_exclusive_cross_support.sh \
  scripts/reproduce_v9_o32_o31_score_calibration_cross_support.sh \
  scripts/reproduce_v9_o33_o31_boundary_refine_cross_support.sh \
  scripts/reproduce_v9_o34_o33_score_calibration_cross_support.sh \
  scripts/reproduce_v9_o35_memory_update_mode_cross_support.sh \
  scripts/reproduce_v9_o36_o35_boundary_refine_cross_support.sh \
  scripts/reproduce_v9_o37_o35_o36_score_calibration_cross_support.sh \
  scripts/reproduce_v9_o38_memory_threshold_logarea_cross_support.sh \
  scripts/reproduce_v9_o39_c075split_exclusive_cross_support.sh \
  > logs/stream4d_v9_o38_final_bash_n.log 2>&1

$PY -m unittest discover -s tests -p 'test_*.py' \
  > logs/stream4d_v9_o38_final_unit_tests.log 2>&1
```

结果：

```text
py_compile: pass
bash -n: pass
unit tests: Ran 13 tests in 0.124s, OK
```

### X42：EOF 最新索引（O38 final）

```text
latest executed repair attempts after O28:
  O29/O30 scene object memory + score-exclusive/refine
  O31 memory exclusivity ablation
  O32 score calibration
  O33 boundary refine
  O34 O33 score calibration
  O35 memory update-mode ablation
  O36 O35 boundary refine
  O37 O35/O36 score calibration
  O38 memory threshold sweep + logarea
  O39 attempted then aborted before valid result

latest final checks:
  Stream3D/logs/stream4d_v9_o38_final_py_compile.log
  Stream3D/logs/stream4d_v9_o38_final_bash_n.log
  Stream3D/logs/stream4d_v9_o38_final_unit_tests.log

latest final conclusion:
  cross-support top priority still not solved.
  strongest reportable method-on-S0 AP/AP50/AP25 = O37 O35 new-points logarea on S0:
    0.032908 / 0.126690 / 0.418266
  P0 Stream3D on S0 reference:
    0.235730 / 0.414306 / 0.537786
```

### X43：O29-O38 final 审计包

命令摘要：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
BASE=stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final
git status --short > ${BASE}_git_status.txt
git diff -- Stream3D/tools Stream3D/scripts \
  docs/stream4d_v9_执行日志.md docs/stream4d_v9_实验结果复盘.md \
  > ${BASE}_git_diff.patch
# filelist 包含 O29-O38 code/scripts/logs/audit outputs/prediction/TMP/evaluation/docs，
# O39 只包含 attempted script/log，不包含有效 result。
zip -q -r ${BASE}.zip -@ < ${BASE}_filelist.txt
sha256sum ${BASE}.zip > ${BASE}.sha256
unzip -t ${BASE}.zip > ${BASE}_ziptest.log
```

结果：

```text
packet: stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final.zip
sha256: see sibling file stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final.sha256
filelist: stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final_filelist.txt
zip test: No errors detected in compressed data
file count: 371
```

## 追加执行：O14-O28 cross-support 继续修复

时间：2026-06-09 Asia/Singapore  
目标：用户要求优先解决 cross-support。O13 后目标没有达成，因此继续按计划推荐方向推进：

```text
继续原则：
  不把 own-support 高分当成 cross-support 成功。
  每次新方法/诊断都跑 unified matrix 或明确标注 diagnostic-only。
  不读 GT 做方法构造；GT 只由 official evaluator / integrity diagnostic 使用。
  失败、命名修复、负结果都写入日志和复盘。
```

### X22：O14 full-span grid8 ws100

目的：

```text
O13 说明局部 window / support completion 不能解决 S0。
O14 改为覆盖 probe5 全场景的 grid8 D4RT windows，测试更多 mask frames/full-span candidates 是否改善 cross-support。
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o14_fullspan_grid8_cross_support.sh
```

输出：

```text
Stream3D/scripts/reproduce_v9_o14_fullspan_grid8_cross_support.sh
Stream3D/scripts/v9_o14_fullspan_grid8_matrix_probe5.json
Stream3D/outputs/audit/v9_o14_fullspan_grid8/o14_fullspan_grid8_matrix_probe5.{json,csv,md}
Stream3D/outputs/v8_d4rt_grid_surfel_field/stream4d_v9_o14_g1_grid8_fullspan_ws100_probe5/summary.{json,csv,md}
Stream3D/outputs/v8_surfel_object_field/stream4d_v9_o14_fullspan_grid8_raw_probe5_summary.{json,csv,md}
Stream3D/logs/stream4d_v9_o14_*.log
```

### X23：O15-O18 overlap merge / threshold / ranking sweep

目的：

```text
O14 raw full-span candidates duplicate/conflict 很高。
继续尝试 non-GT overlap merge、threshold sweep、object competition/ranking rescue。
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o15_o14_overlap_merge_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o16_o14_overlap_merge_threshold_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o17_o14_competition_rescue_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o18_o14_merge_fine_threshold_cross_support.sh
```

新增/修改：

```text
Stream3D/tools/merge_overlapping_prediction_masks.py
  新增：按 prediction mask min-IoC 连接组件合并，不读 GT，写 manifest。

Stream3D/tools/object_competition_rank.py
  新增：--eval-policy，并把 eval_policy 写入 manifest。
```

输出：

```text
Stream3D/scripts/v9_o15_o14_overlap_merge_matrix_probe5.json
Stream3D/scripts/v9_o16_o14_overlap_merge_threshold_matrix_probe5.json
Stream3D/scripts/v9_o17_o14_competition_rescue_matrix_probe5.json
Stream3D/scripts/v9_o18_o14_merge_fine_threshold_matrix_probe5.json
Stream3D/outputs/audit/v9_o15_o14_overlap_merge/
Stream3D/outputs/audit/v9_o16_o14_merge_threshold/
Stream3D/outputs/audit/v9_o17_o14_competition_rescue/
Stream3D/outputs/audit/v9_o18_o14_merge_fine_threshold/
Stream3D/outputs/v9_mask_merge/
Stream3D/outputs/v9_object_competition/
Stream3D/logs/stream4d_v9_o15_*.log
Stream3D/logs/stream4d_v9_o16_*.log
Stream3D/logs/stream4d_v9_o17_*.log
Stream3D/logs/stream4d_v9_o18_*.log
```

### X24：O19 score calibration

目的：

```text
测试 overlap merge 后是否主要是 score/order 问题。
在不读 GT 前提下，用 log_area / inverse_area 等 score feature 重新排序。
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o19_score_calibration_cross_support.sh
```

blocker 和修复：

```text
问题：
  首次 O19 eval 已完成，但 matrix 汇总失败。
  原因是 matrix JSON 里的 cross output_config 少了 input config 自带的 `_probe5` 片段。

修复：
  修正 Stream3D/scripts/v9_o19_score_calibration_matrix_probe5.json 的 output_config 名称。
  不重跑已经成功的 eval，只重跑 matrix/reportable scan/metric integrity。
```

修复后命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -m tools.summarize_v9_unified_eval \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --matrix-json scripts/v9_o19_score_calibration_matrix_probe5.json \
  --output-prefix outputs/audit/v9_o19_score_calibration/o19_score_calibration_matrix_probe5 \
  --dataset scannet \
  --stream3d-config scannet \
  > logs/stream4d_v9_o19_score_calibration_matrix_probe5_r2.log 2>&1
$PY -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v9_o19_o16_g030_logarea_probe5,stream4d_v9_o19_o16_g030_invarea_probe5,stream4d_v9_o19_o18_g040_logarea_probe5,stream4d_v9_o19_o18_g040_invarea_probe5 \
  --output outputs/audit/v9_o19_score_calibration/reportable_config_scan_o19_probe5.md \
  --require-manifest \
  --require-eval-policy \
  > logs/stream4d_v9_o19_score_calibration_reportable_scan_r2.log 2>&1
$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs stream4d_v9_o19_o16_g030_logarea_probe5,stream4d_v9_o19_o16_g030_invarea_probe5,stream4d_v9_o19_o18_g040_logarea_probe5,stream4d_v9_o19_o18_g040_invarea_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v9_o19_score_calibration/metric_integrity_o19_probe5.md \
  --require-manifest \
  > logs/stream4d_v9_o19_score_calibration_metric_integrity_r2.log 2>&1
```

新增/修改：

```text
Stream3D/tools/rescore_prediction_scores.py
  新增 manifest 写入、--eval-policy。
  新增 inverse_area / inverse_sqrt_area / inverse_log_area score features。
```

### X25：O20-O24 full-span window density / mask-aware query

目的：

```text
O14 full-span ws100 仍不够。
继续增加窗口密度 ws50/ws25，测试是否改善 S0；
再测试 mask-aware query densification 是否比固定 grid8 更好。
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o20_fullspan_grid8_ws50_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o21_o20_merge_threshold_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o22_fullspan_grid8_ws25_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o23_o22_merge_threshold_cross_support.sh
bash Stream3D/scripts/reproduce_v9_o24_maskaware_ws50_cross_support.sh
```

输出：

```text
Stream3D/scripts/v9_o20_fullspan_grid8_ws50_matrix_probe5.json
Stream3D/scripts/v9_o21_o20_merge_threshold_matrix_probe5.json
Stream3D/scripts/v9_o22_fullspan_grid8_ws25_matrix_probe5.json
Stream3D/scripts/v9_o23_o22_merge_threshold_matrix_probe5.json
Stream3D/scripts/v9_o24_maskaware_ws50_matrix_probe5.json
Stream3D/outputs/audit/v9_o20_fullspan_grid8_ws50/
Stream3D/outputs/audit/v9_o21_o20_merge_threshold/
Stream3D/outputs/audit/v9_o22_fullspan_grid8_ws25/
Stream3D/outputs/audit/v9_o23_o22_merge_threshold/
Stream3D/outputs/audit/v9_o24_maskaware_ws50/
Stream3D/logs/stream4d_v9_o20_*.log
Stream3D/logs/stream4d_v9_o21_*.log
Stream3D/logs/stream4d_v9_o22_*.log
Stream3D/logs/stream4d_v9_o23_*.log
Stream3D/logs/stream4d_v9_o24_*.log
```

### X26：O25 self-discovered boundary refine

目的：

```text
O23 说明 window density 能提高 S0，但仍差。
O25 用 RGB-D + non-GT 2D masks 做 self-discovered negative evidence / boundary refine，
测试是否减少 full-scene masks 的外溢和 conflict。
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o25_boundary_refine_cross_support.sh
```

blocker 和修复：

```text
问题：
  首次 O25 eval 已完成，但 matrix 汇总失败。
  原因同 O19：matrix JSON 的 cross output_config 命名缺少 `_probe5` 片段。

修复：
  修正 Stream3D/scripts/v9_o25_boundary_refine_matrix_probe5.json。
  不重跑已成功 eval，只重跑 matrix/reportable scan/metric integrity。
```

修复后命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -m tools.summarize_v9_unified_eval \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --matrix-json scripts/v9_o25_boundary_refine_matrix_probe5.json \
  --output-prefix outputs/audit/v9_o25_boundary_refine/o25_boundary_refine_matrix_probe5 \
  --dataset scannet \
  --stream3d-config scannet \
  > logs/stream4d_v9_o25_boundary_refine_matrix_probe5_r2.log 2>&1
$PY -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v9_o25_o23_refine_inside050_probe5,stream4d_v9_o25_o23_refine_inside070int010_probe5 \
  --output outputs/audit/v9_o25_boundary_refine/reportable_config_scan_o25_probe5.md \
  --require-manifest \
  --require-eval-policy \
  > logs/stream4d_v9_o25_boundary_refine_reportable_scan_r2.log 2>&1
$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs stream4d_v9_o25_o23_refine_inside050_probe5,stream4d_v9_o25_o23_refine_inside070int010_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v9_o25_boundary_refine/metric_integrity_o25_probe5.md \
  --require-manifest \
  > logs/stream4d_v9_o25_boundary_refine_metric_integrity_r2.log 2>&1
```

新增/修改：

```text
Stream3D/tools/self_discovered_boundary_refine.py
  新增 manifest 写入和 --eval-policy。
  使用 RGB-D visibility 与 non-GT 2D mask 内外一致性裁剪 object masks；不读 GT。
```

### X27：O26 boundary refine 后 score calibration

目的：

```text
O25 成为当前最强 S0 method 修复。
O26 检查 O25 后是否还有简单 score/order 提升空间。
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o26_boundary_refine_rescore_cross_support.sh
```

blocker 和修复：

```text
问题：
  首次 O26 eval 已完成，但 matrix 汇总失败。
  原因仍是 cross output_config 命名缺少 `_probe5` 片段。

修复：
  修正 Stream3D/scripts/v9_o26_boundary_refine_rescore_matrix_probe5.json。
  不重跑已成功 eval，只重跑 matrix/reportable scan/metric integrity。
```

修复后命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -m tools.summarize_v9_unified_eval \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --matrix-json scripts/v9_o26_boundary_refine_rescore_matrix_probe5.json \
  --output-prefix outputs/audit/v9_o26_boundary_refine_rescore/o26_boundary_refine_rescore_matrix_probe5 \
  --dataset scannet \
  --stream3d-config scannet \
  > logs/stream4d_v9_o26_boundary_refine_rescore_matrix_probe5_r2.log 2>&1
$PY -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v9_o26_o25_inside050_logarea_probe5,stream4d_v9_o26_o25_inside070int010_logarea_probe5 \
  --output outputs/audit/v9_o26_boundary_refine_rescore/reportable_config_scan_o26_probe5.md \
  --require-manifest \
  --require-eval-policy \
  > logs/stream4d_v9_o26_boundary_refine_rescore_reportable_scan_r2.log 2>&1
$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs stream4d_v9_o26_o25_inside050_logarea_probe5,stream4d_v9_o26_o25_inside070int010_logarea_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v9_o26_boundary_refine_rescore/metric_integrity_o26_probe5.md \
  --require-manifest \
  > logs/stream4d_v9_o26_boundary_refine_rescore_metric_integrity_r2.log 2>&1
```

### X28：O27 self-discovered silhouette score

目的：

```text
O26 只有极小提升。
O27 尝试用 non-GT 多帧 silhouette agreement 重新评分，验证剩余问题是否主要是排序。
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o27_silhouette_score_cross_support.sh
```

新增/修改：

```text
Stream3D/tools/self_discovered_silhouette_score.py
  新增 manifest 写入、--eval-policy、--diagnostic-only。
  使用 RGB-D visibility 与 self-discovered 2D silhouette agreement 评分；不读 GT。
```

输出：

```text
Stream3D/scripts/reproduce_v9_o27_silhouette_score_cross_support.sh
Stream3D/scripts/v9_o27_silhouette_score_matrix_probe5.json
Stream3D/outputs/audit/v9_o27_silhouette_score/
Stream3D/outputs/v9_silhouette_score/
Stream3D/logs/stream4d_v9_o27_*.log
```

### X29：O28 target-support-aware rank diagnostic

目的：

```text
O27 负结果后，进一步诊断：
如果允许用 target support 统计做 ranking/suppression，当前 O25 masks 是否能被救回来？
该实验标记 diagnostic-only，不作为 method claim。
```

首次命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o28_support_aware_rank_cross_support.sh
```

blocker 和修复：

```text
问题：
  support_aware_object_rank 直接读取 data/TMP/{score_pre_points_config}。
  `scannet` 是 evaluator 特殊别名，本地 data/TMP/scannet 只有 manifest，没有 scene pre_points。

修复：
  将 S0-aware ranking 的 score support 从 `scannet`
  改为已经 materialized 的 `stream4d_v9_p0_on_s0_scannet_probe5`。
  cross-eval 仍按 `scannet` S0 protocol 评估。
```

修复后命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash Stream3D/scripts/reproduce_v9_o28_support_aware_rank_cross_support.sh
```

输出：

```text
Stream3D/scripts/reproduce_v9_o28_support_aware_rank_cross_support.sh
Stream3D/scripts/v9_o28_support_aware_rank_matrix_probe5.json
Stream3D/outputs/audit/v9_o28_support_aware_rank/
Stream3D/outputs/v9_support_aware_rank/
Stream3D/logs/stream4d_v9_o28_*.log
```

### X30：最终验证

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
$PY -m py_compile \
  tools/merge_overlapping_prediction_masks.py \
  tools/object_competition_rank.py \
  tools/rescore_prediction_scores.py \
  tools/self_discovered_boundary_refine.py \
  tools/self_discovered_silhouette_score.py \
  tools/support_aware_object_rank.py \
  tools/summarize_v9_unified_eval.py \
  tools/evaluate_cross_prepoints.py \
  tools/scan_reportable_configs.py \
  tools/verify_stream4d_metric_integrity.py \
  > logs/stream4d_v9_o28_final_py_compile.log 2>&1
bash -n \
  scripts/reproduce_v9_o14_fullspan_grid8_cross_support.sh \
  scripts/reproduce_v9_o15_o14_overlap_merge_cross_support.sh \
  scripts/reproduce_v9_o16_o14_overlap_merge_threshold_cross_support.sh \
  scripts/reproduce_v9_o17_o14_competition_rescue_cross_support.sh \
  scripts/reproduce_v9_o18_o14_merge_fine_threshold_cross_support.sh \
  scripts/reproduce_v9_o19_score_calibration_cross_support.sh \
  scripts/reproduce_v9_o20_fullspan_grid8_ws50_cross_support.sh \
  scripts/reproduce_v9_o21_o20_merge_threshold_cross_support.sh \
  scripts/reproduce_v9_o22_fullspan_grid8_ws25_cross_support.sh \
  scripts/reproduce_v9_o23_o22_merge_threshold_cross_support.sh \
  scripts/reproduce_v9_o24_maskaware_ws50_cross_support.sh \
  scripts/reproduce_v9_o25_boundary_refine_cross_support.sh \
  scripts/reproduce_v9_o26_boundary_refine_rescore_cross_support.sh \
  scripts/reproduce_v9_o27_silhouette_score_cross_support.sh \
  scripts/reproduce_v9_o28_support_aware_rank_cross_support.sh \
  > logs/stream4d_v9_o28_final_bash_n.log 2>&1
$PY -m unittest discover -s tests -p 'test_*.py' \
  > logs/stream4d_v9_o28_final_unit_tests.log 2>&1
```

结果：

```text
py_compile: pass
bash -n: pass
unit tests: Ran 13 tests in 0.119s, OK
```

### X31：O14-O28 执行结论索引

```text
cross-support top priority 仍未达成。

当前最强 reportable method-on-S0：
  O26 inside050 logarea on S0 = 0.021517 / 0.100351 / 0.291694

当前最强 reportable AP25 on S0：
  O25 inside050 on S0 = 0.021388 / 0.100300 / 0.292631

当前最强 diagnostic-only S0-aware row：
  O28 S0-aware rank diagnostic on S0 = 0.023637 / 0.095270 / 0.267036

P0 Stream3D on S0 reference:
  0.235730 / 0.414306 / 0.537786

因此仍不能 claim cross-support solved。
```

### X32：O14-O28 final 审计包

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
BASE=stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final

# 生成 filelist：包含 v9 计划/执行/复盘文档、O14-O28 scripts/matrix、相关 tools、
# O14-O28 logs、outputs/audit、summary outputs、evaluation txt、prediction/TMP 文件。
wc -l ${BASE}_filelist.txt

git status --short > ${BASE}_git_status.txt
git diff -- Stream3D/tools Stream3D/scripts \
  docs/stream4d_v9_执行日志.md docs/stream4d_v9_实验结果复盘.md \
  > ${BASE}_git_diff.patch

zip -q -r ${BASE}.zip -@ < ${BASE}_filelist.txt
sha256sum ${BASE}.zip > ${BASE}.sha256
unzip -t ${BASE}.zip > ${BASE}_ziptest.log
```

预期/实际文件数：

```text
file count: 1331
```

审计包：

```text
packet: stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.zip
sha256: see sibling file stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.sha256
filelist: stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final_filelist.txt
zip test: see sibling file stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final_ziptest.log
```

### X15：O11 obs-bank overlap suppression

目的：

```text
O10 证明 overlap suppression 能修 O9 duplicate/conflict，但 O10 support 仍太小。
O11 将同一 suppression 思路迁移到 O8 非 GT obs-bank slot+top80 candidates，
验证更大 support 的 obs-bank candidates 是否能在去重后改善 cross-support。
```

新增：

```text
Stream3D/scripts/reproduce_v9_o11_obsbank_overlap_cross_support.sh
Stream3D/scripts/v9_o11_obsbank_overlap_matrix_probe5.json
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash -n Stream3D/scripts/reproduce_v9_o11_obsbank_overlap_cross_support.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/support_aware_object_rank.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/scan_reportable_configs.py \
  Stream3D/tools/verify_stream4d_metric_integrity.py
python3 -m json.tool Stream3D/scripts/v9_o11_obsbank_overlap_matrix_probe5.json >/tmp/v9_o11_matrix_json_check.txt

cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
chmod +x scripts/reproduce_v9_o11_obsbank_overlap_cross_support.sh
./scripts/reproduce_v9_o11_obsbank_overlap_cross_support.sh
```

输出：

```text
v9 O11 obs-bank overlap cross-support done
Stream3D/outputs/audit/v9_o11_obsbank_overlap/o11_obsbank_overlap_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o11_obsbank_overlap/reportable_config_scan_o11_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o11_obsbank_overlap/metric_integrity_o11_probe5.{json,md}
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o11_obsbank_overlap_mioc050_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o11_obsbank_overlap_mioc070_probe5_summary.json
```

结果摘要：

```text
O8 obs-bank slot+top80 own:
  AP/AP50/AP25 = 0.000975 / 0.004807 / 0.039085
  pre ratio = 0.552481
  GT crop/full = 37.80/40.60
  #pred = 96.40
  conflict = 0.704240

O11 minIoC 0.50 own:
  AP/AP50/AP25 = 0.136121 / 0.296384 / 0.654162
  pre ratio = 0.071923
  GT crop/full = 11.60/40.60
  #pred = 20.40
  conflict = 0.179091
  O11 on S0 = 0.000429 / 0.002271 / 0.020027
  O11 on S1 = 0.023517 / 0.063834 / 0.146866

O11 minIoC 0.70 own:
  AP/AP50/AP25 = 0.108074 / 0.223622 / 0.618033
  pre ratio = 0.081563
  #pred = 33.40
  conflict = 0.405934
  O11 on S0 = 0.000429 / 0.002271 / 0.018948
  O11 on S1 = 0.022953 / 0.064411 / 0.144321

reportable scan: pass, num_reportable_method_configs=2
metric integrity: phase0_pass=True
```

判断：

```text
O11 说明 obs-bank top80 的大 support 主要被 duplicate/conflict 毁掉；
去重后 own AP 从约 0.001 提到 0.136。
但 O11 on S0/S1 仍很低，不能解决 cross-support。
```

### X16：O12 fused O10/O11 candidates

目的：

```text
O10 D4RT-clean candidates 与 O11 obs-bank-clean candidates 可能覆盖不同对象。
O12 先 concat O10 0.50 和 O11 0.50，再做一次 support-aware min-IoC 0.50 suppression。
```

新增/修改：

```text
Stream3D/scripts/reproduce_v9_o12_fused_o10_o11_cross_support.sh
Stream3D/scripts/v9_o12_fused_o10_o11_matrix_probe5.json
Stream3D/tools/fuse_prediction_configs.py 新增 --eval-policy，并把 eval_policy 写入 manifest extra。
```

第一次执行失败：

```text
reportable scan 失败：
  num_configs_missing_eval_policy=1

原因：
  fuse_prediction_configs.py 生成的 union config manifest 没有 eval_policy。

修复：
  fuse_prediction_configs.py 新增 --eval-policy；
  O12 union export 添加 --eval-policy own_recompute_o10_o11_union。
```

重跑命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash -n Stream3D/scripts/reproduce_v9_o12_fused_o10_o11_cross_support.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/fuse_prediction_configs.py \
  Stream3D/tools/support_aware_object_rank.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/scan_reportable_configs.py \
  Stream3D/tools/verify_stream4d_metric_integrity.py
python3 -m json.tool Stream3D/scripts/v9_o12_fused_o10_o11_matrix_probe5.json >/tmp/v9_o12_matrix_json_check.txt

cd Stream3D
./scripts/reproduce_v9_o12_fused_o10_o11_cross_support.sh
```

输出：

```text
v9 O12 fused O10/O11 cross-support done
Stream3D/outputs/audit/v9_o12_fused_o10_o11/o12_fused_o10_o11_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o12_fused_o10_o11/reportable_config_scan_o12_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o12_fused_o10_o11/metric_integrity_o12_probe5.{json,md}
Stream3D/outputs/v9_fusion/stream4d_v9_o12_o10_o11_union_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o12_o10_o11_union_overlap_mioc050_probe5_summary.json
```

结果摘要：

```text
O12 union own:
  AP/AP50/AP25 = 0.079668 / 0.217072 / 0.543820
  pre ratio = 0.091712
  GT crop/full = 13.00/40.60
  #pred = 45.00
  conflict = 0.577206

O12 overlap 0.50 own:
  AP/AP50/AP25 = 0.177930 / 0.380577 / 0.643669
  pre ratio = 0.085801
  GT crop/full = 12.60/40.60
  #pred = 26.00
  conflict = 0.192014
  P0 Stream3D on O12 support = 0.359558 / 0.567568 / 0.682199
  O12 on S0 = 0.000301 / 0.001498 / 0.026504
  O12 on S1 = 0.045242 / 0.126188 / 0.232727

reportable scan: pass, num_reportable_method_configs=2
metric integrity: phase0_pass=True
```

判断：

```text
O12 是当前 own support 最强 cross-support repair prototype：
  own AP = 0.177930，高于 O10 0.157425 和 O11 0.136121。
  S1 AP/AP25 也升到 0.045242 / 0.232727。
但 S0 仍只有 0.000301 / 0.001498 / 0.026504，cross-support 仍失败。
```

### X17：O13 O12 support completion to S0

目的：

```text
O12 的主要问题仍是 sparse support。
O13 用 complete_prediction_to_support 将 O12 core 扩到 S0 support，
测试 coverage 不足能否通过最近邻 completion 修复。
只测 r0.10 和 all 两个强度。
```

新增：

```text
Stream3D/scripts/reproduce_v9_o13_o12_support_completion_cross_support.sh
Stream3D/scripts/v9_o13_o12_support_completion_matrix_probe5.json
```

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash -n Stream3D/scripts/reproduce_v9_o13_o12_support_completion_cross_support.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/complete_prediction_to_support.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/scan_reportable_configs.py \
  Stream3D/tools/verify_stream4d_metric_integrity.py
python3 -m json.tool Stream3D/scripts/v9_o13_o12_support_completion_matrix_probe5.json >/tmp/v9_o13_matrix_json_check.txt

cd Stream3D
chmod +x scripts/reproduce_v9_o13_o12_support_completion_cross_support.sh
./scripts/reproduce_v9_o13_o12_support_completion_cross_support.sh
```

输出：

```text
v9 O13 O12 support-completion cross-support done
Stream3D/outputs/audit/v9_o13_o12_support_completion/o13_o12_support_completion_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o13_o12_support_completion/reportable_config_scan_o13_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o13_o12_support_completion/metric_integrity_o13_probe5.{json,md}
Stream3D/outputs/v9_support_completion/stream4d_v9_o13_o12_complete_s0_r010_probe5_summary.{json,csv,md}
Stream3D/outputs/v9_support_completion/stream4d_v9_o13_o12_complete_s0_all_probe5_summary.{json,csv,md}
```

结果摘要：

```text
O13 r0.10 own:
  AP/AP50/AP25 = 0.064846 / 0.167195 / 0.450757
  pre ratio = 0.166961
  GT crop/full = 14.00/40.60
  #pred = 26.00
  conflict = 0.100829
  O13 r0.10 on S1 = 0.054274 / 0.146604 / 0.309673

O13 all own:
  AP/AP50/AP25 = 0.000692 / 0.001785 / 0.065276
  pre ratio = 0.848995
  GT crop/full = 40.60/40.60
  #pred = 26.00
  conflict = 0.022499
  O13 all on S1 = 0.042918 / 0.114857 / 0.352031

reportable scan: pass, num_reportable_method_configs=2
metric integrity: phase0_pass=True
```

判断：

```text
O13 all 覆盖几乎完整 S0，但 AP 崩为 0.000692 / 0.001785 / 0.065276。
这证明最近邻 support completion 不能替代 object boundary / object identity。
O13 r0.10 增加 support 后 own AP 也低于 O12。
因此 support completion 路线也不能解决 cross-support。
```

### X18：O13 后最终检查

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/fuse_prediction_configs.py \
  Stream3D/tools/support_aware_object_rank.py \
  Stream3D/tools/complete_prediction_to_support.py \
  Stream3D/tools/export_v8_surfel_object_field.py \
  Stream3D/tools/slotwise_candidate_select.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/export_v9_propagated_slot_field.py \
  Stream3D/tools/diagnose_v9_d4rt_mask_propagation.py \
  Stream3D/tools/build_union_prepoints.py \
  Stream3D/tools/split_core_fringe_prediction.py \
  > Stream3D/logs/stream4d_v9_o13_final_py_compile.log 2>&1

cd Stream3D
bash -n \
  scripts/reproduce_v9_o13_o12_support_completion_cross_support.sh \
  scripts/reproduce_v9_o12_fused_o10_o11_cross_support.sh \
  scripts/reproduce_v9_o11_obsbank_overlap_cross_support.sh \
  scripts/reproduce_v9_o10_overlap_suppression_cross_support.sh \
  scripts/reproduce_v9_o9_multwindow_cross_support.sh \
  scripts/reproduce_v9_o8_slot_candidate_cross_support.sh \
  scripts/reproduce_v9_o7_birth_recall_cross_support.sh \
  scripts/reproduce_v9_o6_support_completion_cross_support.sh \
  scripts/reproduce_v9_phase5_o5_cross_support.sh \
  scripts/reproduce_v9_s4_phase4b.sh \
  > logs/stream4d_v9_o13_final_bash_n.log 2>&1

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests \
  > logs/stream4d_v9_o13_final_unit_tests.log 2>&1
```

结果：

```text
py_compile: pass
bash -n: pass
unit tests: Ran 13 tests in 0.123s, OK
```

### X19：O13 final cross-support 审计包

命令摘要：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
BASE=stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final
PREV=stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10_filelist.txt
git status --short > ${BASE}_git_status.txt
git diff -- Stream3D/stream4d Stream3D/tools Stream3D/evaluation Stream3D/tests Stream3D/scripts \
  docs/stream4d_v9_执行日志.md docs/stream4d_v9_实验结果复盘.md > ${BASE}_git_diff.patch
# filelist 以 O10 包为基底，追加 O11/O12/O13 outputs/prediction/TMP/eval/logs/docs。
zip -q -r ${BASE}.zip -@ < ${BASE}_filelist.txt
sha256sum ${BASE}.zip > ${BASE}.sha256
unzip -t ${BASE}.zip > ${BASE}_ziptest.log
```

结果：

```text
packet: stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.zip
sha256: see sibling file stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.sha256
filelist: stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final_filelist.txt
file count: 1250
zip test: see sibling file stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final_ziptest.log
```

### X12：O10 overlap suppression for O9 multi-window

目的：

```text
用户明确要求优先解决 cross-support。
O9 已证明 multi-window 可以增加 coverage，但 per-window export 造成严重重复/conflict。
O10 按推荐方向做第一版跨窗口 duplicate suppression：
  从 O9 predictions 出发；
  按 support-aware quality 排序；
  用 min-IoC overlap threshold 竞争保留 object；
  TMP 使用 recompute，确保 own support 与输出 union 一致；
  分别测试 threshold 0.50 / 0.70 / 0.85。
```

修改：

```text
Stream3D/tools/support_aware_object_rank.py
  新增 --tmp-policy {input,recompute}
  新增 --eval-policy
  新增 --diagnostic-only
  输出 config_manifest.json，便于 reportable scan / metric integrity 审计。

Stream3D/scripts/reproduce_v9_o10_overlap_suppression_cross_support.sh
  新增 O10 一键复现入口。

Stream3D/scripts/v9_o10_overlap_suppression_matrix_probe5.json
  新增 O10 own / P0-on-O10 / O10-on-S0 / O10-on-S1 统一矩阵。
```

预检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
bash -n Stream3D/scripts/reproduce_v9_o10_overlap_suppression_cross_support.sh
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/support_aware_object_rank.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py
cat Stream3D/scripts/v9_o10_overlap_suppression_matrix_probe5.json | python3 -m json.tool >/tmp/v9_o10_matrix_json_check.txt
```

执行：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
chmod +x scripts/reproduce_v9_o10_overlap_suppression_cross_support.sh
./scripts/reproduce_v9_o10_overlap_suppression_cross_support.sh
```

输出：

```text
v9 O10 overlap-suppression cross-support done

Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o10_o9_overlap_mioc050_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o10_o9_overlap_mioc070_probe5_summary.json
Stream3D/outputs/v9_overlap_suppression/stream4d_v9_o10_o9_overlap_mioc085_probe5_summary.json
Stream3D/outputs/audit/v9_o10_overlap_suppression/o10_overlap_suppression_matrix_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o10_overlap_suppression/reportable_config_scan_o10_probe5.{json,csv,md}
Stream3D/outputs/audit/v9_o10_overlap_suppression/metric_integrity_o10_probe5.{json,md}
```

主要日志：

```text
Stream3D/logs/stream4d_v9_o10_overlap_suppression_py_compile.log
Stream3D/logs/stream4d_v9_o10_o9_overlap_mioc050_export.log
Stream3D/logs/stream4d_v9_o10_o9_overlap_mioc070_export.log
Stream3D/logs/stream4d_v9_o10_o9_overlap_mioc085_export.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc050_probe5_eval.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc070_probe5_eval.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc085_probe5_eval.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc050_on_s0_probe5.log
Stream3D/logs/stream4d_v9_o10_stream4d_v9_o10_o9_overlap_mioc050_on_s1_probe5.log
Stream3D/logs/stream4d_v9_o10_overlap_suppression_matrix_probe5.log
Stream3D/logs/stream4d_v9_o10_overlap_suppression_reportable_scan.log
Stream3D/logs/stream4d_v9_o10_overlap_suppression_metric_integrity.log
```

结果摘要：

```text
O10 minIoC 0.50 own:
  AP/AP50/AP25 = 0.157425 / 0.376554 / 0.686405
  pre ratio = 0.070985
  GT crop/full = 12.60/40.60
  #pred = 24.60
  conflict = 0.153755
  P0 Stream3D on O10 support = 0.368456 / 0.556364 / 0.664697
  O10 on S0 = 0.000223 / 0.001530 / 0.012000
  O10 on S1 = 0.029484 / 0.089440 / 0.202250

O10 minIoC 0.70 own:
  AP/AP50/AP25 = 0.101223 / 0.265244 / 0.562828
  pre ratio = 0.076435
  #pred = 35.00
  conflict = 0.300720
  O10 on S0 = 0.000182 / 0.001250 / 0.009998
  O10 on S1 = 0.025296 / 0.076849 / 0.167189

O10 minIoC 0.85 own:
  AP/AP50/AP25 = 0.068145 / 0.201929 / 0.443341
  pre ratio = 0.081425
  #pred = 48.60
  conflict = 0.550321
  O10 on S0 = 0.000150 / 0.001041 / 0.007994
  O10 on S1 = 0.023424 / 0.075299 / 0.130966
```

O10 summary check：

```text
minIoC 0.50:
  mean_num_instances_before=63.4
  mean_num_instances_after_competition=24.6
  mean_num_suppressed_by_overlap=38.8
  mean_output_union_count=15067.4
  overlap_mode=min_ioc
  overlap_threshold=0.5

reportable scan:
  num_configs=3
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=3
  num_diagnostic_only_configs=0
  num_uses_gt_and_method_result=0
  num_configs_missing_eval_policy=0

metric integrity:
  phase0_pass=True
  num_reportable_method_configs=3
  gt_files_read_by_rescore=False
```

判断：

```text
O10 明显修复了 O9 的 duplicate/conflict：
  O9 own: 0.040167 / 0.124304 / 0.294343, conflict 0.684667, #pred 63.40
  O10 0.50 own: 0.157425 / 0.376554 / 0.686405, conflict 0.153755, #pred 24.60

但 cross-support 目标仍未达成：
  O10 0.50 on S0 只有 0.000223 / 0.001530 / 0.012000
  O10 0.50 on S1 只有 0.029484 / 0.089440 / 0.202250

因此 O10 是有效修复尝试，但不是最终解。
```

### X13：O10 后最终检查

命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  Stream3D/tools/support_aware_object_rank.py \
  Stream3D/tools/complete_prediction_to_support.py \
  Stream3D/tools/export_v8_surfel_object_field.py \
  Stream3D/tools/slotwise_candidate_select.py \
  Stream3D/tools/summarize_v9_unified_eval.py \
  Stream3D/tools/evaluate_cross_prepoints.py \
  Stream3D/tools/export_v9_propagated_slot_field.py \
  Stream3D/tools/diagnose_v9_d4rt_mask_propagation.py \
  Stream3D/tools/build_union_prepoints.py \
  Stream3D/tools/split_core_fringe_prediction.py \
  > Stream3D/logs/stream4d_v9_o10_final_py_compile.log 2>&1

cd Stream3D
bash -n \
  scripts/reproduce_v9_o10_overlap_suppression_cross_support.sh \
  scripts/reproduce_v9_o9_multwindow_cross_support.sh \
  scripts/reproduce_v9_o8_slot_candidate_cross_support.sh \
  scripts/reproduce_v9_o7_birth_recall_cross_support.sh \
  scripts/reproduce_v9_o6_support_completion_cross_support.sh \
  scripts/reproduce_v9_phase5_o5_cross_support.sh \
  scripts/reproduce_v9_s4_phase4b.sh \
  > logs/stream4d_v9_o10_final_bash_n.log 2>&1

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover -s tests \
  > logs/stream4d_v9_o10_final_unit_tests.log 2>&1
```

结果：

```text
py_compile: pass
bash -n: pass
unit tests: Ran 13 tests in 0.121s, OK
```

### X14：O10 cross-support 审计包

命令摘要：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
BASE=stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10
PREV=stream4d_v9_code_audit_packet_20260609_0652_cross_support_filelist.txt
git status --short > ${BASE}_git_status.txt
git diff -- Stream3D/stream4d Stream3D/tools Stream3D/evaluation Stream3D/tests Stream3D/scripts \
  docs/stream4d_v9_执行日志.md docs/stream4d_v9_实验结果复盘.md > ${BASE}_git_diff.patch
# filelist 以 0652 cross-support 包为基底，追加 O10 outputs/prediction/TMP/eval/logs/docs。
zip -q -r ${BASE}.zip -@ < ${BASE}_filelist.txt
sha256sum ${BASE}.zip > ${BASE}.sha256
unzip -t ${BASE}.zip > ${BASE}_ziptest.log
```

结果：

```text
packet: stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10.zip
sha256: see sibling file stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10.sha256
filelist: stream4d_v9_code_audit_packet_20260609_0702_cross_support_o10_filelist.txt
file count: 1091
zip test: No errors detected in compressed data
```

### X21：真实文件末尾最终索引

```text
latest executed repair attempts:
  O11 obs-bank overlap suppression
  O12 O10/O11 fusion + overlap suppression
  O13 O12 support completion to S0

latest final checks:
  Stream3D/logs/stream4d_v9_o13_final_py_compile.log
  Stream3D/logs/stream4d_v9_o13_final_bash_n.log
  Stream3D/logs/stream4d_v9_o13_final_unit_tests.log

latest final packet:
  stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.zip

latest final sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_0712_cross_support_o13_final.sha256

latest final file count:
  1250

latest final conclusion:
  cross-support top priority not solved.
  strongest own-support method row = O12 fused overlap 0.50, AP/AP50/AP25 = 0.177930 / 0.380577 / 0.643669.
  S0 remains failed: O12 on S0 = 0.000301 / 0.001498 / 0.026504.
```
### X34：EOF 最新索引（O28 final）

```text
latest executed repair attempts after O13:
  O14 full-span grid8 ws100
  O15/O16/O18 overlap merge threshold sweeps
  O17 object competition/ranking rescue
  O19 score calibration
  O20/O22 full-span window-density ws50/ws25
  O24 mask-aware query ws50
  O25 self-discovered boundary refine
  O26 boundary-refine score calibration
  O27 self-discovered silhouette score
  O28 target-support-aware diagnostic rank

latest final checks:
  Stream3D/logs/stream4d_v9_o28_final_py_compile.log
  Stream3D/logs/stream4d_v9_o28_final_bash_n.log
  Stream3D/logs/stream4d_v9_o28_final_unit_tests.log

latest final packet:
  stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.zip

latest final sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_0909_cross_support_o28_final.sha256

latest final file count:
  1331

latest final conclusion:
  cross-support top priority still not solved.
  strongest reportable method-on-S0 AP/AP50/AP25 = O26 inside050 logarea on S0:
    0.021517 / 0.100351 / 0.291694
  strongest reportable AP25 on S0 = O25 inside050 on S0:
    0.021388 / 0.100300 / 0.292631
  strongest diagnostic-only S0-aware row = O28 S0-aware rank diagnostic on S0:
    0.023637 / 0.095270 / 0.267036
  P0 Stream3D on S0 reference:
    0.235730 / 0.414306 / 0.537786
```

### X44：真实 EOF 最新索引（O38 final）

```text
latest executed repair attempts after O28:
  O29/O30 scene object memory + score-exclusive/refine
  O31 memory exclusivity ablation
  O32 score calibration
  O33 boundary refine
  O34 O33 score calibration
  O35 memory update-mode ablation
  O36 O35 boundary refine
  O37 O35/O36 score calibration
  O38 memory threshold sweep + logarea
  O39 attempted then aborted before valid result

latest best reportable method-on-S0:
  O37 O35 new-points logarea on S0
  AP/AP50/AP25 = 0.032908 / 0.126690 / 0.418266

P0 Stream3D on S0 reference:
  AP/AP50/AP25 = 0.235730 / 0.414306 / 0.537786

latest final checks:
  Stream3D/logs/stream4d_v9_o38_final_py_compile.log
  Stream3D/logs/stream4d_v9_o38_final_bash_n.log
  Stream3D/logs/stream4d_v9_o38_final_unit_tests.log

latest final packet:
  stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final.zip

latest final sha256:
  see sibling file stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final.sha256

latest final filelist:
  stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final_filelist.txt

latest final zip test:
  see sibling file stream4d_v9_code_audit_packet_20260609_1000_cross_support_o38_final_ziptest.log

latest final file count:
  371

latest final conclusion:
  cross-support top priority still not solved.
  O37 improved over O26, but remains far below P0 Stream3D on S0.
```
