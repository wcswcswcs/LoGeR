# Stream4D v10 执行日志

日期: 2026-06-09  
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`  
计划文档: `docs/stream4d_v10_unified_eval_regionlet_d4rt_geometry_plan_for_codex.md`  
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`  
GPU 约定: `CUDA_VISIBLE_DEVICES=6,7`

## 执行原则

- 不伪造数据。所有 AP/AP50/AP25 和诊断字段来自本次生成的 evaluator 输出、matrix JSON 或 summary JSON/MD。
- method result 必须满足 `uses_gt_for_prediction=false`、`uses_gt_for_diagnostic=false`、`is_method_result=true`、`is_diagnostic_only=false`。
- 读取 GT 做诊断的配置必须标为 diagnostic-only，不能进入 method table。
- own-support 与 cross-support 同时报告，避免只报 cropped TMP support 的高分。
- D4RT carrier 不继续使用 v8 缓存作为最终证据。用户指出 GPU/缓存风险后，已改为 fresh v10 D4RT carrier 重新生成。

## 代码修改

1. `Stream3D/tools/prediction_manifest.py`
   - manifest 新增 `uses_gt_for_prediction`、`uses_gt_for_diagnostic`、`eval_policy`、`support_source`、`geometry_source`。
   - 默认保持兼容，非 GT method 默认 `uses_gt_for_prediction=false`。

2. `Stream3D/tools/scan_reportable_configs.py`
   - 扫描 v10 GT 字段。
   - method table 拒绝 `uses_gt_for_prediction=true`，拒绝 diagnostic GT method result。
   - 输出 `num_uses_gt_for_prediction`、`num_uses_gt_for_diagnostic_and_method_result` 等计数。

3. `Stream3D/evaluation/evaluate.py`
   - 增加 `--require-manifest`、`--allow-oracle-eval`、`--tmp-root`、`--tmp-config`。
   - 默认拒绝 oracle/diagnostic-only/GT diagnostic 配置被当作正式评测。

4. `Stream3D/tools/verify_stream4d_metric_integrity.py`
   - 记录 v10 manifest GT 字段。
   - `phase0_pass` 同时检查 evaluator core、manifest、support/prediction alignment。

5. `Stream3D/tools/summarize_v10_unified_eval.py`
   - 生成 v10 unified matrix 的 JSON/CSV/MD 和图。
   - 字段包括 AP、support/union、GT crop/full、conflict、best IoU、manifest gate、method table gate。

6. `Stream3D/tools/export_v10_regionlet_birth.py`
   - 新增 R0-R4 training-free regionlet exporter。
   - 修复 RGB/mask 尺寸不一致导致可视化崩溃的问题。
   - 增加 `available_masks` frame source、`--max-mask-frames`、`--enable-point-wta`。

7. `Stream3D/tools/materialize_d4rt_aligned_geometry_for_stream3d.py`
   - 新增 G1-G5 D4RT geometry diagnostic exporter。
   - 输出 diagnostic-only prediction/TMP/geometry manifest。

8. `Stream3D/tools/update_config_manifest_fields.py`
   - 用于评测后 patch manifest 元数据，不改 prediction/TMP 数组。

9. `Stream3D/tools/summarize_v9_unified_eval.py`
   - 修复空预测诊断崩溃: `gt>0, pred=0` 时 `per_gt_best_iou` 写 0，而不是对空 IoU 矩阵做 `max`。

10. `Stream3D/tests/test_stream4d_protocol_fixes.py`
    - 增加 v10 manifest 默认字段和 scanner 拒绝 diagnostic GT method result 的测试。

## 新增脚本

- `Stream3D/scripts/reproduce_v10_phase0.sh`
- `Stream3D/scripts/v10_phase0_matrix_probe5.json`
- `Stream3D/scripts/reproduce_v10_fresh_d4rt_carriers_probe5.sh`
- `Stream3D/scripts/reproduce_v10_regionlet_probe5.sh`
- `Stream3D/scripts/v10_regionlet_matrix_probe5.json`
- `Stream3D/scripts/reproduce_v10_regionlet_repair_probe5.sh`
- `Stream3D/scripts/v10_regionlet_repair_matrix_probe5.json`
- `Stream3D/scripts/reproduce_v10_d4rt_geometry_probe5.sh`
- `Stream3D/scripts/v10_d4rt_geometry_matrix_probe5.json`

## Phase 0 统一评估

执行:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
bash scripts/reproduce_v10_phase0.sh
```

主要输出:

- `outputs/audit/v10_phase0/unified_eval_matrix_probe5.json`
- `outputs/audit/v10_phase0/unified_eval_matrix_probe5.csv`
- `outputs/audit/v10_phase0/unified_eval_matrix_probe5.md`
- `outputs/audit/v10_phase0/reportable_config_scan_phase0_methods.json`
- `outputs/audit/v10_phase0/metric_integrity_phase0_methods.json`

Blocker 与修复:

- 初始脚本直接对完整 `scannet_class_agnostic` 做 own eval，但 probe5 的 TMP 不包含完整 312 scene，触发缺失 `data/TMP/scannet/scene0011_00_pre_points.npy`。
- 修复方向: 不把 full scannet own eval 强行混入 probe5；改为在 probe5 上 materialize cross-prepoints row。
- 修复结果: Phase 0 重新完成，reportable scan 和 metric integrity 均通过。

关键验证:

- reportable scan: `num_configs=6`，`num_reportable_method_configs=6`，`num_suspicious_configs=0`，`num_uses_gt_for_prediction=0`。
- metric integrity: `phase0_pass=True`。

## Fresh D4RT Carrier

执行:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
bash scripts/reproduce_v10_fresh_d4rt_carriers_probe5.sh > logs/stream4d_v10_fresh_d4rt_carriers_probe5.log 2>&1
```

实际 D4RT 命令记录在:

- `Stream3D/outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67/summary.md`

关键参数:

- `--d4rt-config /mnt/data/users/chengshun.wang/pjs/LoGeR/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml`
- `--d4rt-ckpt /mnt/data/users/chengshun.wang/pjs/LoGeR/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt`
- `--device cuda:0`
- `CUDA_VISIBLE_DEVICES=6,7`
- `--seq-list splits/scannet_v6_probe5.txt`
- `--max-frames 16 --window-size 16 --grid-size 32 --query-chunk-size 2048`

GPU 说明:

- `cuda:0` 在 `CUDA_VISIBLE_DEVICES=6,7` 下映射到物理 GPU 6。
- D4RT fresh run 期间用 `nvidia-smi` 观察到物理 GPU 6 约 6816 MiB 显存、99% util。物理 GPU 7 未被该命令显著使用。
- 后续 Regionlet export、matrix 汇总、scanner、evaluator 大多是 CPU/IO 密集，不能代表没有执行 GPU D4RT。

Blocker 与修复:

- 初始 fresh D4RT 使用 `Open-d4rt/configs/model_effective.yaml`，缺少 `checkpoints/VideoMAE2/weights/mae-g/vit_g_hybrid_pt_1200e.pth`。
- 修复方向: 改用 checkpoint 自带 `OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml`，其中 `pretrained.enabled=false`。
- 修复结果: fresh carrier 成功，`num_ok_windows=5`，`num_failed_windows=0`。

Fresh D4RT summary:

- `num_windows=5`
- `num_ok_windows=5`
- `num_failed_windows=0`
- `uv_in01_rate_mean=0.9858451843261719`
- `track_length_visible_mean_mean=13.284716796875`
- `self_uv_error_p90_mean=1.5708253622055055`
- `cycle_uv_error_p90_mean=3.2737268686294554`
- `surfel_coverage_2d_per_frame_mean_mean=0.13198280334472656`

每 scene:

| scene | uv in01 | visible len mean | self p90 px | cycle p90 px | coverage mean | seconds |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.990829 | 15.7715 | 1.38513 | 2.31369 | 0.161015 | 21.18 |
| scene0011_00 | 0.959251 | 9.6402 | 1.55750 | 4.13698 | 0.133202 | 21.07 |
| scene0030_00 | 0.999809 | 15.9829 | 1.61023 | 2.81068 | 0.130849 | 21.05 |
| scene0081_01 | 0.982063 | 11.5381 | 1.65579 | 3.89239 | 0.134019 | 21.16 |
| scene0591_00 | 0.997272 | 13.4908 | 1.64548 | 3.21488 | 0.100830 | 21.23 |

缓存修正:

- 早期 R3/R4 曾经引用 v8 D4RT debug root。
- 用户指出 GPU/缓存风险后，停止把旧 cache 当最终证据。
- 已把 Regionlet、repair、geometry 脚本的 `DEBUG_ROOT` 统一改为:

```text
outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67
```

## Regionlet Lane

执行:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
bash scripts/reproduce_v10_regionlet_probe5.sh
```

主要输出:

- `outputs/audit/v10_regionlet/regionlet_matrix_probe5.json`
- `outputs/audit/v10_regionlet/regionlet_matrix_probe5.csv`
- `outputs/audit/v10_regionlet/regionlet_matrix_probe5.md`
- `outputs/audit/v10_regionlet/reportable_config_scan_regionlet_probe5.json`

Blocker 与修复:

- R0 可视化阶段 RGB 与 mask 尺寸不一致，报 `IndexError: boolean index did not match indexed array`。
- 修复方向: overlay 前把 RGB resize 到 region/mask shape。
- 修复结果: R0-R4 成功导出并完成 matrix。

关键验证:

- reportable scan: `num_configs=5`，`num_reportable_method_configs=5`，`num_suspicious_configs=0`。

## Regionlet Repair

执行:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
bash scripts/reproduce_v10_regionlet_repair_probe5.sh
```

主要输出:

- `outputs/audit/v10_regionlet_repair/regionlet_repair_matrix_probe5.json`
- `outputs/audit/v10_regionlet_repair/regionlet_repair_matrix_probe5.csv`
- `outputs/audit/v10_regionlet_repair/regionlet_repair_matrix_probe5.md`
- `outputs/audit/v10_regionlet_repair/reportable_config_scan_regionlet_repair_probe5.json`

Repair 动作:

- `R0b/R1b/R4b`
- `frame-source=available_masks`
- `max-mask-frames=32`
- `enable-point-wta`

关键验证:

- reportable scan: `num_configs=3`，`num_reportable_method_configs=3`，`num_suspicious_configs=0`。
- WTA 后 conflict rate 全部变为 `0.0`，但 AP/cross-support 没有达到继续扩展门槛。

## D4RT Geometry Diagnostic

执行:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
bash scripts/reproduce_v10_d4rt_geometry_probe5.sh
```

脚本已完成 G1-G5 export、own eval、cross eval。最后 matrix 汇总最初失败，原因和修复如下。

Blocker 与修复:

- matrix 汇总阶段遇到 `ValueError: zero-size array to reduction operation maximum which has no identity`。
- 根因: G1 raw diagnostic 有 scene 没有有效预测对象，形成 `gt>0, pred=0` 的 IoU 矩阵；旧汇总器直接做 `iou.max(axis=1)`。
- 修复方向: 空预测时 best IoU 按 0 记录，保留 AP 文件中的 NA 或 0，不改任何 prediction/TMP/evaluator 输出。
- 修复文件: `Stream3D/tools/summarize_v9_unified_eval.py`。
- 修复后命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.summarize_v10_unified_eval \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --matrix-json scripts/v10_d4rt_geometry_matrix_probe5.json \
  --output-prefix outputs/audit/v10_d4rt_geometry/d4rt_geometry_matrix_probe5 \
  --plot-dir outputs/audit/v10_d4rt_geometry \
  --dataset scannet \
  --stream3d-config scannet
```

主要输出:

- `outputs/audit/v10_d4rt_geometry/d4rt_geometry_matrix_probe5.json`
- `outputs/audit/v10_d4rt_geometry/d4rt_geometry_matrix_probe5.csv`
- `outputs/audit/v10_d4rt_geometry/d4rt_geometry_matrix_probe5.md`
- `outputs/audit/v10_d4rt_geometry/reportable_config_scan_d4rt_geometry_probe5.json`
- `outputs/v10_d4rt_geometry/*_summary.md`

关键验证:

- reportable scan: `num_configs=5`，`num_diagnostic_only_configs=5`，`num_reportable_method_configs=0`。
- G1-G5 都是 diagnostic-only，原因是 Sim3 对齐使用 RGB-D/depth/pose diagnostic anchor，manifest 标记 `uses_gt_for_diagnostic=true`。

## 最终验证命令

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/summarize_v9_unified_eval.py \
  tools/summarize_v10_unified_eval.py
```

结果: 通过。

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
...............
----------------------------------------------------------------------
Ran 15 tests in 0.131s

OK
```

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v10_g1_d4rt_raw_probe5,stream4d_v10_g2_d4rt_scene_sim3_probe5,stream4d_v10_g3_d4rt_window_sim3_probe5,stream4d_v10_g4_d4rt_scene_sim3_density_probe5,stream4d_v10_g5_d4rt_window_sim3_density_probe5 \
  --output outputs/audit/v10_d4rt_geometry/reportable_config_scan_d4rt_geometry_probe5.md \
  --require-manifest \
  --require-eval-policy
```

结果:

```text
num_configs=5
num_configs_missing_eval_policy=0
num_configs_missing_manifest=0
num_diagnostic_only_configs=5
num_reportable_method_configs=0
num_suspicious_configs=0
num_uses_gt_for_prediction=0
```

## 未执行项

- 没有启动 tune30/full ScanNet 扩展。
- 原因: v10 probe5 gates 未通过。Regionlet repair 后仍未超过基线，D4RT geometry 只得到 diagnostic-only 结果，cross-support 近零。

## 审计包

已生成:

- `stream4d_v10_code_audit_packet_20260609_1134_probe5_final.zip`
- `stream4d_v10_code_audit_packet_20260609_1134_probe5_final.sha256`
- `stream4d_v10_code_audit_packet_20260609_1134_probe5_final_filelist.txt`
- `stream4d_v10_code_audit_packet_20260609_1134_probe5_final_git_status.txt`
- `stream4d_v10_code_audit_packet_20260609_1134_probe5_final_git_diff.patch`
- `stream4d_v10_code_audit_packet_20260609_1134_probe5_final_ziptest.log`

SHA256 与 zip test 分别见同名 `.sha256` 和 `_ziptest.log` 文件。没有把最终 SHA256 写死在本日志中，因为本日志本身会被打入 zip，写死 hash 会造成自引用。
