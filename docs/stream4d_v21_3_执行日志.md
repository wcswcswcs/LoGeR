# Stream4D v21.3 执行日志

日期: 2026-06-10
计划文档: `docs/stream4d_v21_3_gt_guard_opend4rt_geometry_diagnostic_plan_for_codex.md`
结果根目录: `Stream3D/outputs/audit`
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`

本日志只记录本次实际执行过的命令、修改文件和落盘 artifact。不把未运行内容写成已完成。

## 0. 环境说明

- 工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`
- Stream3D 命令工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D`
- `python` 不在 PATH；系统 `python3` 缺 numpy。因此后续使用 `loger` conda 环境。
- sandbox 内 `torch.cuda.is_available()` 为 `False`；非 sandbox GPU 检查为 `True`。
- `nvidia-smi` 显示 GPU 6/7 空闲；本次只在 GPU 6 上跑了一个 4-frame/4-query smoke，不跑完整 D4RT probe5。

## 1. 代码修改记录

新增 D4RT-native 代码:

- `Stream3D/stream4d_native/__init__.py`
- `Stream3D/stream4d_native/chunk_alignment.py`
- `Stream3D/stream4d_native/sim3.py`
- `Stream3D/stream4d_native/opend4rt_long_video.py`
- `Stream3D/stream4d_native/occupancy_state.py`
- `Stream3D/stream4d_native/occupancy_dense_tracker.py`
- `Stream3D/stream4d_native/d4rt_scene_builder.py`
- `Stream3D/stream4d_native/OPEND4RT_SOURCE_NOTES.md`

新增 GeometryProvider 抽象:

- `Stream3D/geometry_provider/base.py`
- `Stream3D/geometry_provider/rgbd_provider.py`
- `Stream3D/geometry_provider/d4rt_raw_provider.py`
- `Stream3D/geometry_provider/d4rt_self_stitched_provider.py`
- `Stream3D/geometry_provider/d4rt_eval_sim3_provider.py`
- 修改 `Stream3D/utils/mask_backprojection.py`，在 `frame_backprojection` 入口添加 `args.geometry_provider` hook。

新增审计/诊断工具:

- `Stream3D/tools/scan_native_manifests.py`
- `Stream3D/tools/audit_opend4rt_source_alignment.py`
- `Stream3D/tools/native_geometry_diagnostics.py`

Manifest / guard 修复:

- 修改 `Stream3D/tools/prediction_manifest.py`，新增 v21.3 guard 字段:
  - `uses_rgbd_for_prediction`
  - `uses_pose_for_prediction`
  - `uses_scannet_mesh_for_prediction`
  - `uses_gt_sim3_for_prediction`
  - `uses_d4rt_self_sim3`
  - `uses_rgbd_for_evaluation`
  - `chunking_policy`
  - `opend4rt_reference_policy`
- 修改 `Stream3D/stream4d/run_scannet.py`，`export_mode=rgbd_eval` 输出改为 diagnostic-only / forbidden-for-method-table。
- 修改 `Stream3D/stream4d/export_scannet.py`，`export_rgbd_eval` 写 diagnostic-only manifest。

新增测试:

- `Stream3D/tests/test_native_chunking_and_sim3.py`
- `Stream3D/tests/test_native_occupancy_and_builder.py`

## 2. Phase A 编译和测试

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  stream4d_native/*.py \
  tools/scan_native_manifests.py \
  tools/audit_opend4rt_source_alignment.py \
  tests/test_native_*.py
```

结果: pass，无输出。

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests -p 'test_native_*.py'
```

结果:

```text
Ran 14 tests in 0.036s
OK
```

覆盖点:

- checkpoint 32/48 clip length 读取。
- sliding window 不超过 checkpoint，且 full-scene 默认有 overlap。
- anchor clip 同时包含 source/target。
- overlap Sim3 recover known transform / low-inlier reject。
- Sim3 apply / compose / invert。
- occupancy marks all target frames。
- occupancy reduces duplicate source queries。
- native scene builder 不导入 ScanNet depth/pose/mesh/GT 路径。

## 3. Phase A guard 审计

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_native_manifests --require-no-gt-prediction
```

输出:

```json
{
  "forbidden_import_count": 0,
  "manifest_files_scanned": 0,
  "manifest_glob": "data/prediction/*native*/*config_manifest.json",
  "method_path_forbidden_imports_count": 0,
  "num_method_configs_with_gt_or_rgbd_geometry": 0,
  "source_files_scanned": 7,
  "stream3d_root": "/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D"
}
```

Artifact:

- `Stream3D/outputs/audit/v21_3_phaseA/native_manifest_scan.md`
- `Stream3D/outputs/audit/v21_3_phaseA/native_manifest_scan.json`
- `Stream3D/outputs/audit/v21_3_phaseA/native_manifest_scan.csv`

注意: 本 scanner 默认只扫 native-pattern manifest: `data/prediction/*native*/*config_manifest.json`。本次没有新的 native method manifest，所以 manifest count 为 0；source guard 扫了 7 个 native source 文件。

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.audit_opend4rt_source_alignment --opend4rt-root Open-d4rt
```

输出:

```json
{
  "chunk_size_policy_pass": true,
  "occupancy_primary_path_present": true,
  "opend4rt_helpers_reused_or_ported": 6,
  "opend4rt_required_files_present": true,
  "opend4rt_required_helpers_present": true,
  "opend4rt_root": "/mnt/data/users/chengshun.wang/pjs/LoGeR/Open-d4rt",
  "opend4rt_source_notes_mentions_helpers": true,
  "opend4rt_source_notes_present": true,
  "requested_root_missing_used_fallback": false
}
```

Artifact:

- `Stream3D/outputs/audit/v21_3_phaseA/opend4rt_source_alignment.md`
- `Stream3D/outputs/audit/v21_3_phaseA/opend4rt_source_alignment.json`

全量测试命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 50 tests in 1.773s
OK (skipped=1)
```

## 4. GPU native occupancy smoke

目的: 只验证新增 native occupancy path 能加载真实 OpenD4RT 32CLIP checkpoint 并调用 D4RT，不作为 dense/probe5 实验结果。

sandbox 内 CUDA check:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

sandbox 结果:

```text
2.6.0+cu124
12.4
False
```

非 sandbox 结果:

```text
2.6.0+cu124
12.4
True
```

GPU smoke 命令使用 GPU 6，scene0050_00 前 4 帧，query budget = 4。

结果:

```json
{
  "scene": "scene0050_00",
  "frame_ids": [0, 10, 20, 30],
  "gpu_visible": "6",
  "num_chunks": 1,
  "chunk_policy": {
    "clip_frames": 32,
    "temporal_chunk_size": 32,
    "temporal_chunk_stride": 16,
    "temporal_chunk_overlap": 16
  },
  "chunk_diagnostics": [
    {
      "uses_spatiotemporal_occupancy": true,
      "naive_source_query_count": 5018112,
      "actual_source_query_count": 4,
      "num_output_tubes": 2,
      "pixel_occupancy_coverage_mean": 0.00007811702887460464,
      "query_budget_hit": true,
      "total_d4rt_time_sec": 0.5092999935150146
    }
  ]
}
```

Artifact:

- `Stream3D/outputs/audit/v21_3_geometry/native_occupancy_gpu_smoke.json`

解释: 该 smoke 的 `adaptive_speedup_vs_naive` 很大只因 query budget 极小，不能作为 dense tracking speedup 结论。`query_budget_hit=True`，所以它不是完整 occupancy dense 实验。

## 5. Phase B/C/D cached geometry diagnostic

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics
```

输入:

- Phase B probe5 carrier cache: `Stream3D/outputs/stream4d_debug_full_32f_ioc075_fixmem`
- Phase C multi-window cache: `Stream3D/outputs/stream4d_debug_scene0050_128f_ioc075_fixmem`
- split: `Stream3D/splits/scannet_v6_probe5.txt`
- available old geometry matrix: `Stream3D/outputs/audit/v11_d4rt_geometry/d4rt_geometry_matrix_probe5.json`

输出摘要:

```json
{
  "phase_b": {
    "uv_in01_rate": 0.6336186710030136,
    "visible_track_length_mean": 0.1969606629696502,
    "self_uv_error_p90": 0.014398351926356556,
    "mask_interior_coverage_mean": 0.9910409591491313,
    "mask_boundary_coverage_mean": 0.6023544556601523,
    "local_neighbor_outlier_rate": 0.01435546875,
    "local_neighbor_stretch_p90": 0.23479760206350808
  },
  "phase_c": {
    "num_pairs": 6,
    "alignment_fail_count": 0,
    "overlap_anchor_count_mean": 9164.5,
    "self_sim3_inlier_ratio_mean": 0.8999520507004428,
    "self_sim3_residual_p90_mean": 1.239926672771048,
    "self_sim3_scale_std": 0.2159472010915834,
    "accumulated_scale_drift": 0.7899097455772878
  },
  "phase_d_blocked": true
}
```

Artifact:

- `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics.md`
- `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics.json`
- `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_phase_b_rows.csv`
- `Stream3D/outputs/audit/v21_3_geometry/native_geometry_diagnostics_phase_c_rows.csv`

Phase B eval-only Sim3 mean:

```text
anchor_candidates: 189491.2
anchor_valid: 31550.2
anchor_count: 6233.8
sim3_scale: 0.22376774567487293
median_residual: 0.6438608175007301
p90_residual: 1.057818893724281
p95_residual: 1.1827370324056061
inlier_ratio: 0.8999961897504287
translation_norm: 5.015109679382578
```

Phase C pair rows:

| pair | anchors | scale | inlier | p90 residual |
|---|---:|---:|---:|---:|
| w000 -> w001 | 7592 | 0.931534 | 0.899895 | 0.995011 |
| w001 -> w002 | 6877 | 1.083819 | 0.899956 | 0.943936 |
| w002 -> w003 | 7362 | 0.989746 | 0.899891 | 1.878365 |
| w003 -> w004 | 10691 | 0.601226 | 0.900009 | 0.963557 |
| w004 -> w005 | 12695 | 0.482517 | 0.899961 | 1.401789 |
| w005 -> w006 | 9770 | 0.724731 | 0.900000 | 1.256902 |

## 6. Phase D provider replacement status

注意: 本节记录第一次收尾时的状态。用户要求继续推进后，后续 `## 7. 继续执行` 已经完成 G0-G6 provider replacement diagnostic rerun，并 supersede 本节的 blocker 判定。

已完成:

- 新增 `GeometryProvider` 抽象。
- RGB-D provider 明确 diagnostic-only。
- D4RT raw / self-stitched / eval-Sim3 provider scaffold 已有。
- `frame_backprojection` 已可通过 `args.geometry_provider` 从 provider 获取每帧 mask->point projection。

未完成 / blocker:

- 没有完成真正 G0-G6 full Stream3D provider rerun。
- 旧 v10/v11 artifact 是 minimal projection / adapter diagnostic，不是完整 set-cover、manifold refining、historical merge 的 provider 替换。
- 当前 D4RT provider 仍缺稳定的 per-mask D4RT point ownership 到 Stream3D point universe 的贯通；因此不能诚实生成 v21.3 要求的完整 G0-G6 表。

按计划处理:

- 没有启动 Phase E object formation。
- 没有启动 Phase F ScanNet method export。
- 没有把任何 diagnostic-only artifact 写成 method result。

## 7. 继续执行: Phase D provider replacement 修复与 G0-G6 rerun

用户要求确认目标未达成则继续推进。本阶段按计划推荐方向继续修复 Phase D。

新增 / 修改文件:

- 新增 `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 从 D4RT carrier cache 直接构建 `mask_id -> scene point ids`。
  - 使用 `src_frame_global` 恢复真实 ScanNet frame id，避免旧 helper 把 local frame `0..31` 错当真实帧号。
  - 支持 `raw` / `self_stitched` / `eval_sim3` / `eval_sim3_density` / `self_stitched_density`。
  - 记录 `projection_hit_rate`、`mask_projection_empty_rate`、`mean_points_per_2d_mask`、Sim3 residual 等 provider diagnostics。
- 新增 `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - 通过 `Stream3D main -> mask_graph_construction -> frame_backprojection -> Stream3D/post_process` 路径真实 rerun provider replacement。
  - 每个 variant 输出 diagnostic-only manifest，并生成 `D4RT_geometry_replacement_stream3d_probe5.{md,csv,json}`。
- 新增 `Stream3D/tests/test_v21_3_geometry_provider.py`
  - 覆盖 `src_frame_global` frame-id 恢复和 D4RT carrier mask ownership。
- 修改 `Stream3D/main.py`
  - 新增 `args.frame_id_allowlist`，用于让 G0-G6 都限制到 D4RT debug cache 覆盖的同一组真实帧。
- 修改 `Stream3D/utils/Stream3D.py`
  - `export_class_agnostic_mask` 支持 0-object 输出 `(N, 0)`，避免 diagnostic 空结果崩溃。
  - `export_new` 用 `dataset.seq_name` 推断 TMP scene id，修复新 config 名下 `scene_id` 未定义的问题。
- 修改 `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - eval-Sim3 variants 的 manifest 明确 `uses_gt_for_prediction=true`、`is_diagnostic_only=true`、`forbidden_for_method_table=true`，避免把 GT/RGB-D Sim3 输出生成伪装成普通 method。

编译与单测:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  main.py \
  geometry_provider/d4rt_carrier_provider.py \
  tools/run_v21_3_stream3d_provider_replacement.py \
  utils/Stream3D.py \
  tests/test_v21_3_geometry_provider.py
```

结果: pass。

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v21_3_geometry_provider
```

结果:

```text
Ran 1 test in 0.006s
OK
```

第一次 G1 smoke 在 sandbox 内运行:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_full_32f_ioc075_fixmem \
  --output-prefix stream4d_v21_3_provider_smoke_scene0050 \
  --audit-root outputs/audit/v21_3_phaseD_smoke \
  --variants G1
```

结果: runner 正常启动但 scene 失败，错误为 `RuntimeError: No CUDA GPUs are available`。该失败 row 已写入:

- `Stream3D/outputs/audit/v21_3_phaseD_smoke/D4RT_geometry_replacement_stream3d_probe5.*`

非 sandbox GPU 6 重跑同一 smoke:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_full_32f_ioc075_fixmem \
  --output-prefix stream4d_v21_3_provider_smoke_scene0050 \
  --audit-root outputs/audit/v21_3_phaseD_smoke \
  --variants G1
```

结果: pass，但暴露 frame 范围不匹配问题:

- `num_projected_frames=466`
- `source_windows_mean=0.068670`
- 原因: Stream3D 默认扫完整 scene，而 D4RT cache 只有 32 frames。

修复后用 `frame_id_allowlist` 限制到 debug cache 覆盖帧，重跑 smoke:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_full_32f_ioc075_fixmem \
  --output-prefix stream4d_v21_3_provider_smoke_scene0050_framefix \
  --audit-root outputs/audit/v21_3_phaseD_smoke_framefix \
  --variants G1
```

结果:

```json
{
  "status": "ok",
  "ap": 0.0,
  "ap50": 0.0,
  "ap25": 0.5,
  "num_projected_frames": 32,
  "source_windows_mean": 1.0,
  "projection_hit_rate_mean": 0.05601053626065913,
  "pre_points_ratio": 0.001925205528698334
}
```

正式 Phase D probe5 G0-G6:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_v6_probe5.txt \
  --debug-root outputs/stream4d_debug_full_32f_ioc075_fixmem \
  --output-prefix stream4d_v21_3_provider_probe5_r1 \
  --audit-root outputs/audit/v21_3_phaseD \
  --variants G0,G1,G2,G3,G4,G5,G6
```

输出 artifact:

- `Stream3D/outputs/audit/v21_3_phaseD/D4RT_geometry_replacement_stream3d_probe5.md`
- `Stream3D/outputs/audit/v21_3_phaseD/D4RT_geometry_replacement_stream3d_probe5.csv`
- `Stream3D/outputs/audit/v21_3_phaseD/D4RT_geometry_replacement_stream3d_probe5.json`
- `Stream3D/outputs/audit/v21_3_phaseD/stream4d_v21_3_provider_probe5_r1_g*_provider_diagnostics.json`

关键结果:

| variant | AP | AP50 | AP25 | pre_points_ratio | projection_hit_rate | status |
|---|---:|---:|---:|---:|---:|---|
| G0 RGBD baseline | 0.324948 | 0.497839 | 0.650992 | 0.229350 | NA | ok |
| G1 D4RT raw | 0.000000 | 0.000000 | 0.500000 | 0.000607 | 0.014528 | ok |
| G2 D4RT self-stitched | 0.000000 | 0.000000 | 0.500000 | 0.000607 | 0.014528 | ok |
| G3 D4RT eval-Sim3 | 0.067447 | 0.170455 | 0.581163 | 0.018251 | 0.676159 | ok |
| G4 eval-Sim3 + outlier filter | 0.085961 | 0.227257 | 0.616071 | 0.016314 | 0.607961 | ok |
| G5 eval-Sim3 + density | 0.051376 | 0.126974 | 0.388286 | 0.022382 | 0.840654 | ok |
| G6 self-stitched + density | 0.072531 | 0.222222 | 1.000000 | 0.001005 | 0.033196 | ok |

Deltas:

- `delta_d4rt_eval_sim3 = 0.257501`
- `delta_self_stitch = 0.067447`
- `delta_outlier = 0.018514`
- `delta_density_threshold = -0.016071`

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 51 tests in 1.754s
OK (skipped=1)
```

Reportable scan 首次运行:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v21_3_provider_probe5_r1_g0,stream4d_v21_3_provider_probe5_r1_g1,stream4d_v21_3_provider_probe5_r1_g2,stream4d_v21_3_provider_probe5_r1_g3,stream4d_v21_3_provider_probe5_r1_g4,stream4d_v21_3_provider_probe5_r1_g5,stream4d_v21_3_provider_probe5_r1_g6 \
  --output outputs/audit/v21_3_phaseD/reportable_config_scan_provider_probe5.md \
  --require-manifest \
  --require-eval-policy
```

结果: 返回码 `9`，写出 scan 文件。摘要:

```json
{
  "num_configs": 7,
  "num_reportable_method_configs": 0,
  "num_diagnostic_only_configs": 7,
  "num_suspicious_configs": 3,
  "num_uses_gt_for_prediction": 0,
  "num_alignment_used_for_prediction": 3
}
```

解释: G3/G4/G5 的 eval-Sim3 manifest 初始没有把 `uses_gt_for_prediction` 置为 true，但 `alignment_used_for_prediction=true`，因此被 scanner 标 suspicious。为避免审计语义过软，修正 runner 并补丁已落盘 manifest。

第一次 metadata patch 命令因为 `--reason` 未加引号失败，argparse 报:

```text
error: unrecognized arguments: eval-Sim3 provider diagnostic uses GT/RGB-D Sim3 in output generation
```

随后用无空格 reason 成功更新 G3/G4/G5:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.update_config_manifest_fields --root . --config stream4d_v21_3_provider_probe5_r1_g3 --uses-gt-for-prediction true --uses-gt-for-diagnostic true --is-method-result false --is-diagnostic-only true --forbidden-for-method-table true --reason v21_3_eval_sim3_diagnostic_gt_sim3_used
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.update_config_manifest_fields --root . --config stream4d_v21_3_provider_probe5_r1_g4 --uses-gt-for-prediction true --uses-gt-for-diagnostic true --is-method-result false --is-diagnostic-only true --forbidden-for-method-table true --reason v21_3_eval_sim3_diagnostic_gt_sim3_used
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.update_config_manifest_fields --root . --config stream4d_v21_3_provider_probe5_r1_g5 --uses-gt-for-prediction true --uses-gt-for-diagnostic true --is-method-result false --is-diagnostic-only true --forbidden-for-method-table true --reason v21_3_eval_sim3_diagnostic_gt_sim3_used
```

Reportable scan 修正后:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v21_3_provider_probe5_r1_g0,stream4d_v21_3_provider_probe5_r1_g1,stream4d_v21_3_provider_probe5_r1_g2,stream4d_v21_3_provider_probe5_r1_g3,stream4d_v21_3_provider_probe5_r1_g4,stream4d_v21_3_provider_probe5_r1_g5,stream4d_v21_3_provider_probe5_r1_g6 \
  --output outputs/audit/v21_3_phaseD/reportable_config_scan_provider_probe5_after_manifest_fix.md \
  --require-manifest \
  --require-eval-policy
```

结果: 返回码 `5`，符合预期，因为 G3/G4/G5 使用 GT/RGB-D Sim3 生成 diagnostic output。摘要:

```json
{
  "num_configs": 7,
  "num_reportable_method_configs": 0,
  "num_diagnostic_only_configs": 7,
  "num_suspicious_configs": 3,
  "num_uses_gt_for_prediction": 3,
  "num_uses_gt_for_prediction_and_method_result": 0,
  "num_alignment_used_for_prediction": 3,
  "num_configs_missing_manifest": 0,
  "num_configs_missing_eval_policy": 0
}
```

结论: Phase D diagnostic rerun 已完成；所有 G0-G6 artifact 都是 diagnostic-only / forbidden-for-method-table，没有可报告 method result。

## 8. 继续执行: dense128-grid carrier support repair

用户指出 GPU 0/1/2/3/4/5/7 空闲后，本轮继续按 Phase D 后续修复方向推进: 优先测试 D4RT carrier/object-surface support 是否因每 mask carrier 太稀而失败。没有启动 Phase E/F，没有把任何 eval-Sim3 row 当 method。

新增临时 split 文件:

- `Stream3D/splits/scannet_v21_3_dense128_scene0030.txt`
- `Stream3D/splits/scannet_v21_3_dense128_scene0081.txt`
- `Stream3D/splits/scannet_v21_3_dense128_scene0591.txt`

### 8.1 scene0050 dense128-grid smoke

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet_scene0050.txt \
  --output-config stream4d_v21_3_dense128_grid_smoke_scene0050 \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_smoke_scene0050 \
  --frame-stride 10 \
  --max-frames 32 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 128 \
  --min-points-per-mask 8 \
  --sampling-strategy grid_inside_mask \
  --query-chunk-size 2048 \
  --export-mode rgbd_eval \
  --export-support-mode carrier_uv \
  --export-nn-radius 0.05 \
  --continue-on-error
```

结果:

```text
scene0050_00: carriers=88064, props=17, objects=17, window_sec=355.33, total_sec=358.38, export_points=34756, export_hit_rate=0.9285
```

随后先在 sandbox 内错误运行 provider smoke:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_smoke_scene0050 \
  --output-prefix stream4d_v21_3_dense128_grid_smoke_provider_scene0050_r1 \
  --audit-root outputs/audit/v21_3_phaseD_dense128_grid_smoke \
  --variants G0,G1,G3,G4,G5
```

结果: 失败，所有 row 报 `RuntimeError: No CUDA GPUs are available`。该失败是 sandbox CUDA 限制，不作为实验结论。

GPU 6 重跑:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_smoke_scene0050 \
  --output-prefix stream4d_v21_3_dense128_grid_smoke_provider_scene0050_r1 \
  --audit-root outputs/audit/v21_3_phaseD_dense128_grid_smoke \
  --variants G0,G1,G3,G4,G5
```

scene0050 dense smoke provider 结果:

| variant | AP | AP50 | AP25 | pre% | projection hit | status |
|---|---:|---:|---:|---:|---:|---|
| G0 | 0.266232 | 0.477454 | 0.672414 | 0.232373 | NA | ok |
| G1 | 0.061728 | 0.092593 | 0.481481 | 0.011745 | 0.052490 | ok |
| G3 | 0.126090 | 0.312500 | 0.705062 | 0.109864 | 0.748805 | ok |
| G4 | 0.123737 | 0.315476 | 0.670996 | 0.104997 | 0.673906 | ok |
| G5 | 0.196495 | 0.368049 | 0.679474 | 0.120257 | 0.952276 | ok |

解释: dense128-grid 对 scene0050 有明显覆盖改善，G5 单 scene AP 提升到 `0.196495`，但仍低于 G0 `0.266232`，且 G5 使用 eval-Sim3，仍是 diagnostic-only。

### 8.2 probe5 dense128-grid cache 生成

先启动串行 probe5:

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v21_3_dense128_grid_probe5_r1 \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_probe5_r1 \
  --frame-stride 10 \
  --max-frames 32 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 128 \
  --min-points-per-mask 8 \
  --sampling-strategy grid_inside_mask \
  --query-chunk-size 2048 \
  --export-mode rgbd_eval \
  --export-support-mode carrier_uv \
  --export-nn-radius 0.05 \
  --continue-on-error
```

串行已完成:

```text
scene0050_00: carriers=88064, objects=17, export_hit_rate=0.9285, total_sec=357.49
scene0011_00: carriers=49664, objects=86, export_hit_rate=0.6839, total_sec=179.00
```

用户提示其它 GPU 空闲后，为避免同目录并发写入，先用 `kill -TERM 1238218` 停止已进入 scene0030 的串行进程。随后用独立 debug root 并行补跑剩余 scenes:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet_v21_3_dense128_scene0030.txt \
  --output-config stream4d_v21_3_dense128_grid_scene0030_r1 \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_scene0030_r1 \
  --frame-stride 10 --max-frames 32 --window-size 32 --window-stride 16 \
  --max-points-per-mask 128 --min-points-per-mask 8 --sampling-strategy grid_inside_mask \
  --query-chunk-size 2048 --export-mode rgbd_eval --export-support-mode carrier_uv --export-nn-radius 0.05 --continue-on-error
```

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet_v21_3_dense128_scene0081.txt \
  --output-config stream4d_v21_3_dense128_grid_scene0081_r1 \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_scene0081_r1 \
  --frame-stride 10 --max-frames 32 --window-size 32 --window-stride 16 \
  --max-points-per-mask 128 --min-points-per-mask 8 --sampling-strategy grid_inside_mask \
  --query-chunk-size 2048 --export-mode rgbd_eval --export-support-mode carrier_uv --export-nn-radius 0.05 --continue-on-error
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet_v21_3_dense128_scene0591.txt \
  --output-config stream4d_v21_3_dense128_grid_scene0591_r1 \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_scene0591_r1 \
  --frame-stride 10 --max-frames 32 --window-size 32 --window-stride 16 \
  --max-points-per-mask 128 --min-points-per-mask 8 --sampling-strategy grid_inside_mask \
  --query-chunk-size 2048 --export-mode rgbd_eval --export-support-mode carrier_uv --export-nn-radius 0.05 --continue-on-error
```

并行结果:

| scene | carriers | props | objects | window sec | total sec | export points | export hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | 49664 | 86 | 86 | 177.535 | 179.000 | 25874 | 0.683929 |
| scene0030_00 | 44160 | 12 | 12 | 183.032 | 186.893 | 32290 | 0.835200 |
| scene0050_00 | 88064 | 17 | 17 | 354.324 | 357.486 | 34756 | 0.928501 |
| scene0081_01 | 106624 | 492 | 492 | 375.701 | 380.236 | 33581 | 0.590275 |
| scene0591_00 | 185216 | 271 | 271 | 911.200 | 918.571 | 33426 | 0.847446 |

合并到完整 probe5 debug root:

```bash
mkdir -p outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1
cp -a outputs/stream4d_debug_v21_3_dense128_grid_probe5_r1/scene0050_00 outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1/
cp -a outputs/stream4d_debug_v21_3_dense128_grid_probe5_r1/scene0011_00 outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1/
cp -a outputs/stream4d_debug_v21_3_dense128_grid_scene0030_r1/scene0030_00 outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1/
cp -a outputs/stream4d_debug_v21_3_dense128_grid_scene0081_r1/scene0081_01 outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1/
cp -a outputs/stream4d_debug_v21_3_dense128_grid_scene0591_r1/scene0591_00 outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1/
```

校验:

```text
5 个 scenes 都有 carriers_window000.npz 和 summary.json
merged root size: 347M
```

### 8.3 dense128-grid probe5 provider replacement

G0/G1/G3/G4/G5:

```bash
CUDA_VISIBLE_DEVICES=7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_v6_probe5.txt \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1 \
  --output-prefix stream4d_v21_3_dense128_grid_provider_probe5_r1 \
  --audit-root outputs/audit/v21_3_phaseD_dense128_grid_probe5 \
  --variants G0,G1,G3,G4,G5
```

G2/G6 单独补跑:

```bash
CUDA_VISIBLE_DEVICES=7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_v6_probe5.txt \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1 \
  --output-prefix stream4d_v21_3_dense128_grid_provider_probe5_self_r1 \
  --audit-root outputs/audit/v21_3_phaseD_dense128_grid_probe5_self \
  --variants G2,G6
```

结果:

| variant | AP | AP50 | AP25 | pre% | projection hit | #pred | status |
|---|---:|---:|---:|---:|---:|---:|---|
| G0 | 0.324948 | 0.497839 | 0.650992 | 0.229350 | NA | 57.8 | ok |
| G1 | 0.086785 | 0.201183 | 0.497041 | 0.003953 | 0.013630 | 6.8 | ok |
| G2 | 0.086785 | 0.201183 | 0.497041 | 0.003953 | 0.013630 | 6.8 | ok |
| G3 | 0.140696 | 0.289128 | 0.537907 | 0.090106 | 0.676236 | 70.0 | ok |
| G4 | 0.136916 | 0.281781 | 0.556021 | 0.082324 | 0.608571 | 67.8 | ok |
| G5 | 0.116184 | 0.234043 | 0.454701 | 0.111051 | 0.898392 | 87.8 | ok |
| G6 | 0.079138 | 0.169082 | 0.543860 | 0.006601 | 0.037451 | 9.0 | ok |

对比旧 Phase D sparse carrier:

| variant | old AP | dense AP | old pre% | dense pre% | old hit | dense hit |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 0.000000 | 0.086785 | 0.000607 | 0.003953 | 0.014528 | 0.013630 |
| G2 | 0.000000 | 0.086785 | 0.000607 | 0.003953 | 0.014528 | 0.013630 |
| G3 | 0.067447 | 0.140696 | 0.018251 | 0.090106 | 0.676159 | 0.676236 |
| G4 | 0.085961 | 0.136916 | 0.016314 | 0.082324 | 0.607961 | 0.608571 |
| G5 | 0.051376 | 0.116184 | 0.022382 | 0.111051 | 0.840654 | 0.898392 |
| G6 | 0.072531 | 0.079138 | 0.001005 | 0.006601 | 0.033196 | 0.037451 |

Dense deltas:

```text
delta_d4rt_eval_sim3 = AP(G0) - AP(G3) = 0.184252
delta_self_stitch = AP(G3) - AP(G2) = 0.053911
delta_outlier = AP(G4) - AP(G3) = -0.003780
delta_density_threshold = AP(G5) - AP(G3) = -0.024511
```

Artifact:

- `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5/D4RT_geometry_replacement_stream3d_probe5.*`
- `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5_self/D4RT_geometry_replacement_stream3d_probe5.*`
- `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5/*provider_diagnostics.json`
- `Stream3D/outputs/audit/v21_3_phaseD_dense128_grid_probe5_self/*provider_diagnostics.json`

### 8.4 dense128 audit / tests

第一次 reportable scan 命令写错，误用了不存在的 `--glob`:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --glob 'data/prediction/stream4d_v21_3_dense128_grid_provider_probe5*_class_agnostic/*config_manifest.json' \
  --output outputs/audit/v21_3_phaseD_dense128_grid_probe5/reportable_config_scan_dense128_provider_probe5.md \
  --fail-on-suspicious
```

结果: 失败，argparse 报 `the following arguments are required: --configs`。未用于审计结论。

修正后:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v21_3_dense128_grid_provider_probe5_r1_g0,stream4d_v21_3_dense128_grid_provider_probe5_r1_g1,stream4d_v21_3_dense128_grid_provider_probe5_self_r1_g2,stream4d_v21_3_dense128_grid_provider_probe5_r1_g3,stream4d_v21_3_dense128_grid_provider_probe5_r1_g4,stream4d_v21_3_dense128_grid_provider_probe5_r1_g5,stream4d_v21_3_dense128_grid_provider_probe5_self_r1_g6 \
  --pred-suffix class_agnostic \
  --output outputs/audit/v21_3_phaseD_dense128_grid_probe5/reportable_config_scan_dense128_provider_probe5.md \
  --require-manifest \
  --require-eval-policy
```

结果: 返回码 `5`，符合预期。摘要:

```json
{
  "num_configs": 7,
  "num_reportable_method_configs": 0,
  "num_diagnostic_only_configs": 7,
  "num_suspicious_configs": 3,
  "num_uses_gt_for_prediction": 3,
  "num_uses_gt_for_prediction_and_method_result": 0,
  "num_alignment_used_for_prediction": 3,
  "num_configs_missing_manifest": 0,
  "num_configs_missing_eval_policy": 0
}
```

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 51 tests in 1.756s
OK (skipped=1)
```

本轮 dense128-grid repair 结论: 增加 carrier density 能明显改善 eval-Sim3 diagnostic coverage 和 AP，但仍没有达到可报告方法成功；G3 dense AP `0.140696` 仍低于 G0 `0.324948`，且 raw/self D4RT 仍低覆盖。继续启动 Phase E/F 仍不合理。

## 9. 继续执行: mask-interior provider repair

dense128-grid 后继续尝试减少半径/边界污染。新增一个只影响 diagnostic provider 的 2D mask interior gate:

修改文件:

- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 新增 `min_mask_interior_px` 参数。
  - 对 carrier 当前 UV 落入的 2D mask 计算 `distance_transform_edt(mask == mask_id)`。
  - 当 `min_mask_interior_px > 0` 时过滤 mask 边界附近 carrier。
  - 记录 `interior_filtered_point_count` 和 `min_mask_interior_px`。
- `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - 新增 CLI 参数 `--min-mask-interior-px`。
  - 新增 G7/G8 diagnostic variants:
    - G7: `eval_sim3 + min_mask_interior_px=2.0`
    - G8: `eval_sim3_density + min_mask_interior_px=2.0`
- `Stream3D/tests/test_v21_3_geometry_provider.py`
  - 新增边界过滤单测，确认边界点被过滤、内部点保留。

编译与 provider 单测:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  geometry_provider/d4rt_carrier_provider.py \
  tools/run_v21_3_stream3d_provider_replacement.py \
  tests/test_v21_3_geometry_provider.py
```

结果: pass。

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v21_3_geometry_provider
```

结果:

```text
Ran 2 tests in 0.009s
OK
```

G7/G8 provider diagnostic:

```bash
CUDA_VISIBLE_DEVICES=7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_v6_probe5.txt \
  --debug-root outputs/stream4d_debug_v21_3_dense128_grid_probe5_merged_r1 \
  --output-prefix stream4d_v21_3_dense128_grid_provider_probe5_interior_r1 \
  --audit-root outputs/audit/v21_3_phaseD_dense128_grid_probe5_interior \
  --variants G7,G8
```

结果:

| variant | AP | AP50 | AP25 | pre% | projection hit | interior filtered | #pred | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G7 eval-Sim3 + interior gate | 0.144135 | 0.286865 | 0.563043 | 0.086586 | 0.614394 | 2131.131 | 70.2 | ok |
| G8 eval-Sim3 + density + interior gate | 0.122724 | 0.240363 | 0.493827 | 0.106406 | 0.813372 | 2131.131 | 86.0 | ok |

对照:

- G7 vs G3: AP `0.144135` vs `0.140696`，小幅 +`0.003439`；pre% `0.086586` vs `0.090106`，略降。
- G8 vs G5: AP `0.122724` vs `0.116184`，小幅 +`0.006540`；仍低于 G3/G7。

Reportable scan:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v21_3_dense128_grid_provider_probe5_interior_r1_g7,stream4d_v21_3_dense128_grid_provider_probe5_interior_r1_g8 \
  --pred-suffix class_agnostic \
  --output outputs/audit/v21_3_phaseD_dense128_grid_probe5_interior/reportable_config_scan_dense128_interior_provider_probe5.md \
  --require-manifest \
  --require-eval-policy
```

结果: 返回码 `5`，符合预期。摘要:

```json
{
  "num_configs": 2,
  "num_reportable_method_configs": 0,
  "num_diagnostic_only_configs": 2,
  "num_suspicious_configs": 2,
  "num_uses_gt_for_prediction": 2,
  "num_uses_gt_for_prediction_and_method_result": 0,
  "num_configs_missing_manifest": 0,
  "num_configs_missing_eval_policy": 0
}
```

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 52 tests in 1.723s
OK (skipped=1)
```

本轮 mask-interior repair 结论: 2D interior gate 可以带来很小 AP 改善，说明边界污染存在；但改善幅度不足以改变主结论。G7 `0.144135` 仍远低于 G0 `0.324948`，G8 仍说明 density/radius 扩张污染没有根治。继续不启动 Phase E/F。

## 10. Phase C corrected overlap matching 与 gate sweep

继续检查 Phase C self-Sim3 blocker 时发现 `tools/native_geometry_diagnostics.py` 的 `_matched_overlap` 旧逻辑按相同 local frame index 匹配相邻 window。对 32/16 sliding window 来说这是错误的: window0 local 16..31 才与 window1 local 0..15 对应同一批真实 ScanNet frame。旧 Phase C 报告因此混入了非 overlap frame 的 UV/XYZ 近邻匹配。

代码修复:

- `Stream3D/tools/native_geometry_diagnostics.py`
  - `_matched_overlap` 新增 `prev_frame_ids/curr_frame_ids`。
  - 使用 `_frame_ids_from_summary` 从 `summary.json` 恢复每个 window 的真实 frame id。
  - 只在相邻 window 的共同真实 frame 上做 UV 近邻 matching。
  - Phase C row 新增 `overlap_frame_count`。
  - 新增 `--phase-c-max-matches-per-frame`。
- `Stream3D/tests/test_v21_3_geometry_provider.py`
  - 新增 regression test，确认 Phase C overlap matching 使用真实 global frame ids，而不是相同 local index。

编译:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/native_geometry_diagnostics.py
```

结果: pass。

Corrected overlap baseline:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --output outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap.md
```

结果:

```json
{
  "num_pairs": 6,
  "alignment_fail_count": 0,
  "overlap_frame_count_mean": 16.0,
  "overlap_anchor_count_mean": 7568.0,
  "self_sim3_inlier_ratio_mean": 0.8999319360959953,
  "self_sim3_residual_p90_mean": 0.11185275637680007,
  "self_sim3_scale_std": 0.0475873474643657,
  "accumulated_scale_drift": 0.26968406729811234
}
```

Corrected pair scales:

| pair | overlap frames | anchors | scale | p90 residual |
|---|---:|---:|---:|---:|
| w000 -> w001 | 16 | 7160 | 0.945616 | 0.148605 |
| w001 -> w002 | 16 | 5657 | 1.012924 | 0.132771 |
| w002 -> w003 | 16 | 8015 | 0.901958 | 0.114902 |
| w003 -> w004 | 16 | 8192 | 1.002363 | 0.090686 |
| w004 -> w005 | 16 | 8192 | 0.955421 | 0.096526 |
| w005 -> w006 | 16 | 8192 | 0.882699 | 0.087626 |

Gate sweep:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --overlap-uv-radius 0.005 \
  --output outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005.md
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --overlap-uv-radius 0.0025 \
  --output outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv0025.md
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --overlap-uv-radius 0.005 \
  --min-visibility 0.7 \
  --min-confidence 0.7 \
  --output outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005_vc07.md
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --overlap-uv-radius 0.005 \
  --min-visibility 0.9 \
  --min-confidence 0.9 \
  --output outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005_vc09.md
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --phase-c-max-matches-per-frame 2048 \
  --output outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_m2048.md
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --overlap-uv-radius 0.005 \
  --phase-c-max-matches-per-frame 2048 \
  --output outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_uv005_m2048.md
```

Sweep 摘要:

| setting | anchors mean | residual p90 | scale std | accumulated drift | gate |
|---|---:|---:|---:|---:|---|
| uv0.010 m512 | 7568.0 | 0.111853 | 0.047587 | 0.269684 | False |
| uv0.005 m512 | 6159.667 | 0.107457 | 0.048867 | 0.255112 | False |
| uv0.0025 m512 | 3787.667 | 0.104421 | 0.050291 | 0.246115 | False |
| uv0.005 vis/conf0.7 | 5156.333 | 0.098947 | 0.049296 | 0.254580 | False |
| uv0.005 vis/conf0.9 | 1883.833 | 0.093877 | 0.048916 | 0.266656 | False |
| uv0.010 m2048 | 15373.667 | 0.111948 | 0.048123 | 0.266380 | False |
| uv0.005 m2048 | 9108.833 | 0.107804 | 0.049220 | 0.253983 | False |

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 53 tests in 1.733s
OK (skipped=1)
```

Corrected Phase C 结论:

- 旧 Phase C 的 `self_sim3_scale_std=0.215947` / `accumulated_scale_drift=0.789910` 被 corrected-overlap 结果 supersede；旧结果来自 local-index overlap 匹配 bug，不应继续作为最终证据。
- 修正后相邻 pair self-Sim3 质量显著改善: `scale_std=0.047587 <= 0.10`，`residual_p90=0.111853`，`inlier=0.899932`。
- 但 chain accumulated drift 仍 `0.246115~0.269684 > 0.20`，gate 仍未完全通过。
- 简单收紧 UV / visibility / confidence / match cap 不能解决 residual drift；下一步若继续 Phase C，需要 pairwise-to-reference 或 local bundle self-Sim3，而不是继续阈值硬调。

## 11. Phase C scale-normalized local-bundle diagnostic

继续沿 Phase C 推荐修复方向实现一个非 GT 的 scale-normalized local-bundle diagnostic。该诊断不使用 RGB-D/pose/mesh/GT，只对 corrected pairwise self-Sim3 的 scale 做共同偏置分摊:

```text
scale_bias = geometric_mean(pair_scales)
normalized_pair_scale_i = pair_scale_i / scale_bias
```

这样让 pair scale product 回到 1，用于测试 chain accumulated drift 是否主要来自全局 scale bias。该结果仍是 diagnostic，不是 frozen method output。

代码修改:

- `Stream3D/tools/native_geometry_diagnostics.py`
  - 新增 `_fixed_scale_residual_summary`。
  - Phase C 输出新增 `scale_normalized_bundle` summary。
  - 新增 `*_phase_c_scale_normalized_rows.csv`。

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --output outputs/audit/v21_3_geometry/native_geometry_diagnostics_corrected_overlap_bundle.md
```

结果:

```json
{
  "num_pairs": 6,
  "overlap_anchor_count_mean": 7568.0,
  "self_sim3_scale_std": 0.0475873474643657,
  "accumulated_scale_drift": 0.26968406729811234,
  "scale_normalized_bundle": {
    "enabled": true,
    "scale_bias_removed": 0.9489684981977856,
    "normalized_scale_std": 0.05014639322036529,
    "normalized_accumulated_scale_drift": 2.220446049250313e-16,
    "normalized_residual_median_mean": 0.09621417140707446,
    "normalized_residual_p90_mean": 0.17959963150481661
  }
}
```

Scale-normalized pair rows:

| pair | original scale | normalized scale | normalized p90 residual |
|---|---:|---:|---:|
| w000 -> w001 | 0.945616 | 0.996467 | 0.219877 |
| w001 -> w002 | 1.012924 | 1.067395 | 0.215902 |
| w002 -> w003 | 0.901958 | 0.950462 | 0.182004 |
| w003 -> w004 | 1.002363 | 1.056266 | 0.148682 |
| w004 -> w005 | 0.955421 | 1.006800 | 0.160016 |
| w005 -> w006 | 0.882699 | 0.930167 | 0.151116 |

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 53 tests in 1.665s
OK
```

结论:

- scale-normalized bundle 能把 accumulated scale drift 从 `0.269684` 降到约 `0`，且 normalized residual p90 mean `0.179600` 没有爆炸。
- 这说明 corrected Phase C 的剩余 drift 很可能可由 D4RT-only scale prior / local bundle 缓解。
- 但这还只是 self-Sim3 diagnostic；Phase B image-space gate 和 Phase D provider/materialization gap 仍未过，因此仍不能启动 Phase E/F。

## 12. Provider-level scale-normalized self-stitch smoke

继续把 scale-normalized bundle 从纯 Phase C 表格诊断接入 `D4RTCarrierProjectionProvider`，用 scene0050 128-frame multi-window cache 做 provider-level smoke。该实验仍是 diagnostic-only，不使用 GT/RGB-D Sim3。

代码修改:

- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 新增 `self_stitched_scale_normalized`。
  - 新增 `self_stitched_scale_normalized_density`。
  - 先估计相邻窗口 curr->prev 的 D4RT-only local pair Sim3。
  - 用 pair scales 的几何均值作为 `scale_bias` 分摊掉共同偏置。
  - 将 normalized pair transforms compose 到 canonical frame。
- `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - 新增 G9/G10:
    - G9: self-stitched scale-normalized bundle
    - G10: self-stitched scale-normalized density

编译与单测:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  geometry_provider/d4rt_carrier_provider.py \
  tools/run_v21_3_stream3d_provider_replacement.py
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v21_3_geometry_provider
```

结果:

```text
Ran 3 tests in 0.009s
OK
```

Provider smoke 命令:

```bash
CUDA_VISIBLE_DEVICES=7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_scene0050_128f_ioc075_fixmem \
  --output-prefix stream4d_v21_3_scene0050_128f_bundle_provider_r1 \
  --audit-root outputs/audit/v21_3_phaseD_scene0050_128f_bundle \
  --variants G0,G2,G6,G9,G10
```

结果:

| variant | AP | AP50 | AP25 | pre% | hit | #pred | stitch drift | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| G0 RGBD baseline | 0.251972 | 0.492424 | 0.650866 | 0.436771 | NA | 84 | NA | ok |
| G2 self-stitched | 0.000000 | 0.000000 | 0.666667 | 0.002852 | 0.013647 | 5 | NA | ok |
| G6 self-stitched density | 0.000000 | 0.000000 | 0.656250 | 0.005426 | 0.040477 | 8 | NA | ok |
| G9 scale-normalized | 0.000000 | 0.000000 | 0.333333 | 0.003108 | 0.013806 | 5 | 0.000000 | ok |
| G10 scale-normalized density | 0.000000 | 0.000000 | 0.720000 | 0.006797 | 0.047125 | 11 | 0.000000 | ok |

G9/G10 diagnostics:

```text
self_stitch_scale_bias_removed_mean = 0.950389
self_stitch_accumulated_scale_drift_mean = 2.22e-16
self_stitch_residual_p90_mean_mean = 0.162034
```

Reportable scan:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v21_3_scene0050_128f_bundle_provider_r1_g0,stream4d_v21_3_scene0050_128f_bundle_provider_r1_g2,stream4d_v21_3_scene0050_128f_bundle_provider_r1_g6,stream4d_v21_3_scene0050_128f_bundle_provider_r1_g9,stream4d_v21_3_scene0050_128f_bundle_provider_r1_g10 \
  --pred-suffix class_agnostic \
  --output outputs/audit/v21_3_phaseD_scene0050_128f_bundle/reportable_config_scan_scene0050_128f_bundle_provider.md \
  --require-manifest \
  --require-eval-policy
```

结果:

```json
{
  "num_configs": 5,
  "num_reportable_method_configs": 0,
  "num_diagnostic_only_configs": 5,
  "num_suspicious_configs": 0,
  "num_uses_gt_for_prediction": 0,
  "num_uses_gt_for_prediction_and_method_result": 0
}
```

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 53 tests in 1.692s
OK
```

结论:

- G9/G10 证明 scale-normalized bundle 可以在 provider 层把 self-stitch accumulated drift 归零。
- 但 scene0050 128f 的 AP 仍为 `0.0`，pre% 仍只有 `0.003108/0.006797`，说明 raw D4RT self-stitched support 的主问题不是 chain scale drift，而是 carrier 落点/coverage 极低。
- 因此即使 Phase C drift 可被 local bundle 缓解，Phase B/D 的 geometry/materialization blocker 仍未解除，继续不启动 Phase E/F。

## 13. 继续执行: scene0050 32f native occupancy ablation smoke

用户提示 GPU 0/1/2/3/4/5/7 可用后，继续补计划 3.7 的 occupancy 对照。该实验调用真实 D4RT checkpoint，但只跑 `scene0050_00` 32-frame window 的 D0/D2/D3/D4 smoke，不是 probe5 full D0-D5，也不是 method result。

代码修改:

- `Stream3D/stream4d_native/occupancy_state.py`
  - 新增 mask-aware source priority: mask interior -> boundary -> uniform。
  - 新增 mask interior / boundary coverage summary。
- `Stream3D/tools/run_v21_3_native_occupancy_ablation.py`
  - 新增真实 `D4RTAdapter` occupancy ablation runner。
  - 支持 D0 fixed full-grid、D2 mask-aware fixed、D3 occupancy uniform、D4 occupancy mask-aware。
- `Stream3D/tests/test_native_occupancy_and_builder.py`
  - 新增 mask-aware sampling 和 mask coverage summary 单测。

编译和单测:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  stream4d_native/occupancy_state.py \
  stream4d_native/occupancy_dense_tracker.py \
  tools/run_v21_3_native_occupancy_ablation.py \
  tests/test_native_occupancy_and_builder.py
```

结果: pass。

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_occupancy_and_builder
```

结果:

```text
Ran 7 tests in 0.003s
OK
```

第一次 r1 命令:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D0,D2,D3,D4 \
  --query-budget 4096 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --output-prefix outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r1
```

r1 结果说明:

- 命令成功落盘。
- 但 r1 暴露 runner 汇总字段 bug: fixed D0/D2 的 coverage summary 覆盖了 `uses_spatiotemporal_occupancy`、`adaptive_speedup_vs_naive` 和 `total_d4rt_time_sec`。
- 已修复 runner；r1 不作为最终数据。

修复后重新编译/单测:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/run_v21_3_native_occupancy_ablation.py \
  stream4d_native/occupancy_state.py \
  tests/test_native_occupancy_and_builder.py
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_occupancy_and_builder
```

结果:

```text
Ran 7 tests in 0.003s
OK
```

r2 命令:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D0,D2,D3,D4 \
  --query-budget 4096 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --output-prefix outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r2
```

r2 stdout progress:

```text
[occupancy_ablation] start variant=D0
[occupancy_ablation] done variant=D0
[occupancy_ablation] start variant=D2
[occupancy_ablation] done variant=D2
[occupancy_ablation] start variant=D3
[occupancy_ablation] done variant=D3
[occupancy_ablation] start variant=D4
[occupancy_ablation] done variant=D4
```

r2 结果:

| variant | queries | occupancy | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| D0 fixed full-grid32 | 32768 | False | 0.145756 | NA | NA | 0.410795 | 0.244301 | False | 109.647530 |
| D2 mask-aware fixed32 | 22016 | False | 0.074613 | 0.072570 | 0.156897 | 0.393523 | 0.215321 | False | 73.819152 |
| D3 occupancy uniform | 4096 | True | 0.019867 | NA | NA | NA | NA | True | 25.670812 |
| D4 occupancy mask-aware | 4096 | True | 0.019924 | 0.020016 | 0.015978 | NA | NA | True | 25.517930 |

随后补充 D3/D4 的 UV/visibility 统计。代码修改:

- `Stream3D/tools/run_v21_3_native_occupancy_ablation.py`
  - 对 occupancy tubes 增加 `uv_in01_rate` 与 `visible_track_length_mean` 汇总。

编译和单测:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/run_v21_3_native_occupancy_ablation.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_occupancy_and_builder
```

结果:

```text
Ran 7 tests in 0.003s
OK
```

r3 D3/D4 命令:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D3,D4 \
  --query-budget 4096 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --output-prefix outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r3_d3d4
```

r3 结果:

| variant | queries | occupancy | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| D3 occupancy uniform | 4096 | True | 0.019867 | NA | NA | 0.248396 | 0.248818 | True | 25.849630 |
| D4 occupancy mask-aware | 4096 | True | 0.019924 | 0.020016 | 0.015978 | 0.248865 | 0.249287 | True | 25.472391 |

Artifact:

- `Stream3D/outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r2.*`
- `Stream3D/outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r3_d3d4.*`

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 55 tests in 1.682s
OK
```

结论:

- D3/D4 使用真实 D4RT checkpoint，不再只是 fake-model unit test。
- D3/D4 只用 `4096` queries，query-count speedup 明显，但都 `query_budget_hit=True`。
- D4 mask-aware occupancy coverage 低于 D2 fixed mask-aware: mask interior `0.020016` vs `0.072570`，mask boundary `0.015978` vs `0.156897`。
- 当前 occupancy priority / boundary sampling / budget policy 仍是 blocker；不能启动 Phase E/F。

## 14. 继续执行: occupancy priority-balanced / radius / budget repair

根据 `## 13` 的 blocker，继续修 D4 mask-aware occupancy。问题定位:

- r3 D4 的 priority 是贪心式 `mask interior -> boundary -> uniform`，4096 budget 基本被 interior 消耗，boundary coverage 只有 `0.015978`。
- 计划排查方向包括 `mark_radius_px`、boundary priority 和 budget policy，因此继续按这些方向修。

代码修改:

- `Stream3D/stream4d_native/occupancy_state.py`
  - 将 mask-aware priority 从 greedy 改成 priority-balanced sampling。
  - 第一版 balanced sampler 把 `uniform_unvisited` 也放入前置 quota，导致构建 full-image candidate 过慢。
  - 随后修复为先平衡非 uniform priority，例如 interior/boundary，只有剩余配额才构建 uniform。
- `Stream3D/tests/test_native_occupancy_and_builder.py`
  - 将 mask-aware sampling 单测改成匹配新语义: 测试 mask-priority 内采样，不要求包含 uniform fallback 时全部落在 mask 内。

编译与单测:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  stream4d_native/occupancy_state.py \
  tools/run_v21_3_native_occupancy_ablation.py \
  tests/test_native_occupancy_and_builder.py
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_occupancy_and_builder
```

结果:

```text
Ran 7 tests in 0.003s
OK
```

失败的 r4 第一次运行:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D4 \
  --query-budget 4096 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --output-prefix outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r4_d4_priority_balanced
```

结果: 运行超过 2 分钟时 GPU util 为 0、CPU 高，判定卡在 candidate generation。用 `kill -TERM 1454533` 终止。该次未作为实验结果。

修复 uniform candidate 构建后，同名 r4 重跑:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D4 \
  --query-budget 4096 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --output-prefix outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r4_d4_priority_balanced
```

r4 结果:

| variant | queries | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| D4 priority-balanced | 4096 | 0.017740 | 0.017121 | 0.052089 | 0.226539 | 0.226877 | True | 31.612875 |

r5: priority-balanced + `mark_radius_px=4`:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D4 \
  --query-budget 4096 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-prefix outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r5_d4_priority_balanced_radius4
```

r5 结果:

| variant | queries | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| D4 priority-balanced radius4 | 4096 | 0.055019 | 0.053288 | 0.149907 | 0.223559 | 0.223820 | True | 31.467228 |

r6: priority-balanced + `mark_radius_px=4` + budget 8192:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D4 \
  --query-budget 8192 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-prefix outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r6_d4_priority_balanced_radius4_budget8192
```

r6 结果:

| variant | queries | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| D4 priority-balanced radius4 budget8192 | 8192 | 0.103342 | 0.100476 | 0.261895 | 0.221797 | 0.222070 | True | 61.887372 |

r7: priority-balanced + `mark_radius_px=4` + budget 7168:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D4 \
  --query-budget 7168 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-prefix outputs/audit/v21_3_occupancy/native_occupancy_ablation_scene0050_32f_r7_d4_priority_balanced_radius4_budget7168
```

r7 结果:

| variant | queries | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| D4 priority-balanced radius4 budget7168 | 7168 | 0.091768 | 0.089131 | 0.237754 | 0.222065 | 0.222354 | True | 54.388076 |

对照 D2 fixed mask-aware:

| variant | queries | pixel cov | mask interior | mask boundary | time sec |
|---|---:|---:|---:|---:|---:|
| D2 fixed mask-aware | 22016 | 0.074613 | 0.072570 | 0.156897 | 73.819152 |
| D4 r7 priority-balanced radius4 budget7168 | 7168 | 0.091768 | 0.089131 | 0.237754 | 54.388076 |

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 55 tests in 1.638s
OK
```

结论:

- priority-balanced sampling 修复了 boundary starvation；r4 boundary coverage `0.052089` 高于 r3 `0.015978`。
- `mark_radius_px=4` 是有效修复；r5 在相同 4096 queries 下把 boundary coverage 提到 `0.149907`，接近 D2 `0.156897`。
- r7 用 `7168` queries 超过 D2 的 mask coverage: interior `0.089131 > 0.072570`，boundary `0.237754 > 0.156897`，且 queries 少 3.07x、time 少约 19.43s。
- 但 r7 仍 `query_budget_hit=True`，且只在 scene0050 32f smoke 上验证；完整 probe5 D0-D5 仍未完成，不能启动 Phase E/F。

## 15. 继续执行: probe5 D4 r7 generalization 与 D2 baseline

为验证 scene0050 上的 D4 r7 policy 是否泛化，继续在 probe5 五个 scene 上并行跑 D4 r7，并补跑 D2 fixed mask-aware baseline 做公平对照。仍然是 occupancy diagnostic，不是 method result。

D4 r7 policy:

```text
variants: D4
query_budget: 7168
source_points_per_round: 512
mark_radius_px: 4
query_chunk_size: 2048
frame_stride: 10
max_frames: 32
```

D4 r7 并行命令模板，分别设置 `CUDA_VISIBLE_DEVICES=0..4` 和不同 `--seq-name`:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name <scene> \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D4 \
  --query-budget 7168 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-prefix outputs/audit/v21_3_occupancy_probe5_d4_r7/<scene>_d4_r7
```

实际 scene/GPU 映射:

| scene | GPU |
|---|---:|
| scene0050_00 | 0 |
| scene0011_00 | 1 |
| scene0030_00 | 2 |
| scene0081_01 | 3 |
| scene0591_00 | 4 |

D4 r7 probe5 结果:

| scene | queries | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| scene0011_00 | 7168 | 0.039917 | 0.038749 | 0.087170 | 0.093873 | 0.094372 | True | 56.298534 |
| scene0030_00 | 7168 | 0.143084 | 0.141894 | 0.210721 | 0.356110 | 0.356742 | True | 53.311596 |
| scene0050_00 | 7168 | 0.091768 | 0.089131 | 0.237754 | 0.222065 | 0.222354 | True | 54.234831 |
| scene0081_01 | 7168 | 0.042046 | 0.041652 | 0.060915 | 0.100243 | 0.100556 | True | 61.744026 |
| scene0591_00 | 7168 | 0.083232 | 0.083650 | 0.131802 | 0.204136 | 0.204446 | True | 62.272944 |

D4 r7 mean:

```text
pixel_occupancy_coverage_mean = 0.080009
mask_interior_coverage_mean = 0.079015
mask_boundary_coverage_mean = 0.145672
uv_in01_rate = 0.195286
visible_track_length_mean = 0.195694
total_d4rt_time_sec = 57.572386
semantic_adaptive_speedup = 5490.115374
all_query_budget_hit = True
```

D2 fixed baseline 并行命令模板，分别设置 `CUDA_VISIBLE_DEVICES=0..4` 和不同 `--seq-name`:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name <scene> \
  --frame-stride 10 \
  --max-frames 32 \
  --variants D2 \
  --query-chunk-size 2048 \
  --output-prefix outputs/audit/v21_3_occupancy_probe5_d2_fixed/<scene>_d2_fixed
```

D2 vs D4 r7:

| scene | D2 q | D4 q | q reduction | D2 interior | D4 interior | D2 boundary | D4 boundary | D2 time | D4 time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | 12416 | 7168 | 1.732 | 0.017890 | 0.038749 | 0.035615 | 0.087170 | 41.323 | 56.299 |
| scene0030_00 | 11040 | 7168 | 1.540 | 0.069768 | 0.141894 | 0.077989 | 0.210721 | 36.663 | 53.312 |
| scene0050_00 | 22016 | 7168 | 3.071 | 0.072570 | 0.089131 | 0.156897 | 0.237754 | 74.224 | 54.235 |
| scene0081_01 | 26656 | 7168 | 3.719 | 0.027582 | 0.041652 | 0.052161 | 0.060915 | 86.751 | 61.744 |
| scene0591_00 | 46304 | 7168 | 6.460 | 0.122358 | 0.083650 | 0.205452 | 0.131802 | 150.849 | 62.273 |

Probe5 means:

| variant | queries mean | pixel cov | mask interior | mask boundary | uv_in01 | visible len | time sec | all budget hit |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D2 fixed | 23686.4 | 0.062799 | 0.062034 | 0.105623 | 0.631871 | 0.194399 | 77.962186 | False |
| D4 r7 | 7168.0 | 0.080009 | 0.079015 | 0.145672 | 0.195286 | 0.195694 | 57.572386 | True |

Wins:

```text
D4 r7 > D2 interior coverage: 4 / 5 scenes
D4 r7 > D2 boundary coverage: 4 / 5 scenes
```

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 55 tests in 1.467s
OK
```

结论:

- D4 r7 在 probe5 mean 上超过 D2 fixed mask-aware coverage: interior `0.079015 > 0.062034`，boundary `0.145672 > 0.105623`，并且 queries/time 更低。
- 但 D4 r7 的 uv_in01 mean 只有 `0.195286`，明显低于 D2 的 `0.631871`；它提升的是 occupancy coverage，不代表 D4RT geometry 已可用于 Stream3D point universe。
- D4 r7 在 scene0591 上低于 D2，说明不是全 scene 稳定胜出。
- 所有 D4 r7 rows 仍 `query_budget_hit=True`；probe5 D0/D1/D3/D5 未跑；未接入 provider/object AP。因此仍不能启动 Phase E/F。

## 16. 继续执行: D2 radius4 公平对照、D0/D1/D3 补全

Section 15 的 D2 baseline 使用默认 `mark_radius_px=2`，而 D4 r7 使用 `mark_radius_px=4`。因此“D4 r7 coverage 高于 D2”的结论只对默认半径成立，不是公平半径对照。本节补跑/补记同半径 D2r4，以及 D0/D1/D3 probe5 artifact，用于完整 D0-D4 occupancy diagnostic。

代码修改:

- `Stream3D/tools/run_v21_3_native_occupancy_ablation.py`
  - 新增 D1: `D1_fixed_full_grid48`。
  - runner 只加载 RGB/mask，不再通过 `ScanNetStream.load_window` 顺手加载 depth/pose。
  - 新增 `--window-size` / `--window-stride`，供 D5 overlap warmstart 使用。
- `Stream3D/stream4d_native/occupancy_dense_tracker.py`
  - 新增 `warmstart_tracks` 参数，用于 D5 在 overlap frame 上先 mark 上一 window 的可见 track。
- `Stream3D/tests/test_native_occupancy_and_builder.py`
  - 新增 warmstart mark 单测。

D0/D1/D2r4/D3 并行命令模板，分别设置 `CUDA_VISIBLE_DEVICES=0..4` 和不同 `--seq-name`:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name <scene> \
  --frame-stride 10 \
  --max-frames 32 \
  --variants <D0|D1|D2|D3> \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-prefix outputs/audit/<output_dir>/<scene>_<suffix>
```

额外 D3 参数:

```text
--query-budget 7168 --source-points-per-round 512
```

输出目录:

| variant | output dir |
|---|---|
| D0 | `Stream3D/outputs/audit/v21_3_occupancy_probe5_d0_grid32_radius4` |
| D1 | `Stream3D/outputs/audit/v21_3_occupancy_probe5_d1_grid48_radius4` |
| D2r4 | `Stream3D/outputs/audit/v21_3_occupancy_probe5_d2_fixed_radius4` |
| D3r7 | `Stream3D/outputs/audit/v21_3_occupancy_probe5_d3_uniform_r7` |

scene/GPU 映射:

| scene | GPU |
|---|---:|
| scene0050_00 | 0 |
| scene0011_00 | 1 |
| scene0030_00 | 2 |
| scene0081_01 | 3 |
| scene0591_00 | 4 |

Probe5 means:

| variant | query mean | pixel cov | mask interior | mask boundary | uv_in01 | visible len | time sec | budget hit |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D0 full-grid32 radius4 | 32768.0 | 0.329996 | NA | NA | 0.583287 | 0.224036 | 107.196207 | False |
| D1 full-grid48 radius4 | 73728.0 | 0.537018 | NA | NA | 0.585857 | 0.228456 | 240.800609 | False |
| D2 fixed mask-aware radius4 | 23686.4 | 0.150638 | 0.149194 | 0.220562 | 0.631871 | 0.194399 | 77.563186 | False |
| D3 occupancy uniform r7 | 1945.6 | 0.023628 | NA | NA | 0.235452 | 0.235616 | 12.372525 | False |
| D4 r7 priority-balanced radius4 | 7168.0 | 0.080009 | 0.079015 | 0.145672 | 0.195286 | 0.195694 | 57.572386 | True |

D2r4 vs D4 r7:

| scene | D2r4 q | D4 q | D2r4 interior | D4 interior | D2r4 boundary | D4 boundary | D2r4 uv | D4 uv | D2r4 time | D4 time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | 12416 | 7168 | 0.053335 | 0.038749 | 0.097173 | 0.087170 | 0.615436 | 0.093873 | 41.129 | 56.299 |
| scene0030_00 | 11040 | 7168 | 0.186100 | 0.141894 | 0.184685 | 0.210721 | 0.515614 | 0.356110 | 36.236 | 53.312 |
| scene0050_00 | 22016 | 7168 | 0.180171 | 0.089131 | 0.324923 | 0.237754 | 0.393523 | 0.222065 | 73.586 | 54.235 |
| scene0081_01 | 26656 | 7168 | 0.080427 | 0.041652 | 0.143080 | 0.060915 | 0.797363 | 0.100243 | 86.291 | 61.744 |
| scene0591_00 | 46304 | 7168 | 0.245935 | 0.083650 | 0.352951 | 0.131802 | 0.837422 | 0.204136 | 150.574 | 62.273 |

Wins:

```text
D4 r7 > D2r4 interior coverage: 0 / 5 scenes
D4 r7 > D2r4 boundary coverage: 1 / 5 scenes
```

结论:

- Section 15 的“D4 r7 mean 超过 D2”只对默认半径 D2 成立；同半径 D2r4 是更公平对照，并且 D2r4 明显更强。
- D4 r7 保留 query/time 优势，但 coverage 与 uv_in01 都不足。
- D0/D1 full-grid 提升 pixel coverage，但 query/time 过大，且没有 mask interior/boundary 语义；D3 uniform 便宜但 coverage 很低。

## 17. 继续执行: D5 overlap warmstart probe5

D5 目的: 按计划继续尝试 overlap window warmstart，使相邻 32-frame window 的 shared frames 能继承上一 window 的 D4RT track occupancy mark。该实验不使用 RGB-D/pose/mesh/GT，只使用 D4RT 输出的 `uv/visibility/confidence/valid` 和 frame overlap。

先跑 scene0050 smoke:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 64 \
  --window-size 32 \
  --window-stride 16 \
  --variants D5 \
  --query-budget 7168 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-prefix outputs/audit/v21_3_occupancy_d5_smoke/scene0050_00_d5_64f_r1
```

Smoke 结果:

```text
queries=21504
num_windows=3
warmstart_track_count=9062
pixel_occupancy_coverage_mean=0.131435
mask_interior_coverage_mean=0.127957
mask_boundary_coverage_mean=0.352909
uv_in01_rate=0.255265
visible_track_length_mean=0.255636
query_budget_hit=True
total_d4rt_time_sec=156.237864
```

D5 probe5 并行命令模板:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_native_occupancy_ablation \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name <scene> \
  --frame-stride 10 \
  --max-frames 64 \
  --window-size 32 \
  --window-stride 16 \
  --variants D5 \
  --query-budget 7168 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-prefix outputs/audit/v21_3_occupancy_probe5_d5_warmstart64/<scene>_d5_warmstart64
```

实际 scene/GPU 映射:

| scene | GPU |
|---|---:|
| scene0050_00 | 0 |
| scene0011_00 | 1 |
| scene0030_00 | 2 |
| scene0081_01 | 3 |
| scene0591_00 | 4 |

Probe5 D5 结果:

| scene | queries | windows | warmstart | pixel cov | mask interior | mask boundary | uv_in01 | visible len | budget hit | time sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| scene0011_00 | 21504 | 3 | 8823 | 0.105969 | 0.103476 | 0.221116 | 0.219482 | 0.220039 | True | 161.481327 |
| scene0030_00 | 21504 | 3 | 10602 | 0.156739 | 0.156122 | 0.221561 | 0.296525 | 0.297231 | True | 165.621760 |
| scene0050_00 | 21504 | 3 | 9062 | 0.131435 | 0.127957 | 0.352909 | 0.255265 | 0.255636 | True | 156.868990 |
| scene0081_01 | 21504 | 3 | 7302 | 0.058262 | 0.057717 | 0.085110 | 0.107751 | 0.107971 | True | 175.380746 |
| scene0591_00 | 21504 | 3 | 8673 | 0.114603 | 0.114685 | 0.154577 | 0.218374 | 0.218793 | True | 179.182308 |

D5 mean:

```text
actual_source_query_count = 21504.000000
warmstart_track_count = 8892.400000
pixel_occupancy_coverage_mean = 0.113402
mask_interior_coverage_mean = 0.111991
mask_boundary_coverage_mean = 0.207054
uv_in01_rate = 0.219479
visible_track_length_mean = 0.219934
total_d4rt_time_sec = 167.707026
semantic_adaptive_speedup = 5496.993843
adaptive_speedup_vs_naive = 5600.571429
all_query_budget_hit = True
```

D5 window progression:

| scene | w0 interior | w1 interior | w2 interior | w0 boundary | w1 boundary | w2 boundary | w1 warm | w2 warm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | 0.038749 | 0.120824 | 0.150855 | 0.087170 | 0.276466 | 0.299711 | 3588 | 5235 |
| scene0030_00 | 0.141894 | 0.166736 | 0.159738 | 0.210721 | 0.230689 | 0.223273 | 5722 | 4880 |
| scene0050_00 | 0.089131 | 0.131195 | 0.163544 | 0.237754 | 0.317191 | 0.503781 | 4780 | 4282 |
| scene0081_01 | 0.041652 | 0.064146 | 0.067355 | 0.060915 | 0.096085 | 0.098331 | 3699 | 3603 |
| scene0591_00 | 0.083650 | 0.126842 | 0.133563 | 0.131802 | 0.158567 | 0.173361 | 4585 | 4088 |

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 56 tests in 1.623s
OK
```

结论:

- D5 warmstart 真正生效: 每个 scene 都有非零 `warmstart_track_count`，且 w1/w2 coverage 通常高于 w0。
- D5 mean mask interior/boundary `0.111991/0.207054` 高于 D4 r7 `0.079015/0.145672`，但仍低于 D2r4 `0.149194/0.220562`。
- D5 仍然 5/5 scenes `query_budget_hit=True`，且未接入 provider/object AP。
- 因此 D0-D5 diagnostic 已完成，但仍不是可报告方法成功，继续不启动 Phase E/F。

## 18. 继续执行: occupancy carrier cache 接入 provider/AP diagnostic

上一节 D0-D5 只验证 occupancy coverage，还没有验证这些 support 能否转成 Stream3D provider/AP。本节新增 exporter，把 D2r4/D5 的 D4RT carrier/tube 输出落成 `D4RTCarrierProjectionProvider` 可读取的 `carriers_window*.npz` cache，然后继续走 Phase D provider replacement。

新增文件:

- `Stream3D/tools/export_v21_3_occupancy_carrier_cache.py`

新增工具职责:

- D2: 导出 mask-aware fixed D4RT carrier cache。
- D5: 导出 overlap warmstart 后每个 window 的 accepted D4RT tubes。
- 每个 `carriers_window*.npz` 同步写 manifest，标明 diagnostic-only、无 GT/RGB-D/pose/mesh prediction。

代码验证:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/export_v21_3_occupancy_carrier_cache.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_occupancy_and_builder tests.test_v21_3_geometry_provider
```

结果:

```text
Ran 11 tests in 0.010s
OK
```

### 18.1 D2r4 cache export

scene0050 smoke:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.export_v21_3_occupancy_carrier_cache \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --variant D2 \
  --frame-stride 10 \
  --max-frames 32 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-root outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1 \
  --summary-root outputs/audit/v21_3_occupancy_provider_cache_d2r4
```

结果:

```text
actual_source_query_count=22016
num_carriers_saved=22016
total_d4rt_time_sec=73.781950
```

剩余 probe5 scenes 并行导出，命令同上，只改 `CUDA_VISIBLE_DEVICES` 和 `--seq-name`:

| scene | GPU | queries | carriers saved | time sec |
|---|---:|---:|---:|---:|
| scene0011_00 | 1 | 12416 | 12416 | 41.044640 |
| scene0030_00 | 2 | 11040 | 11040 | 36.392366 |
| scene0081_01 | 3 | 26656 | 26656 | 86.348844 |
| scene0591_00 | 4 | 46304 | 46304 | 150.342746 |

### 18.2 D2r4 provider smoke/probe5

scene0050 provider smoke:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1 \
  --output-prefix stream4d_v21_3_occupancy_d2r4_provider_scene0050_r1 \
  --audit-root outputs/audit/v21_3_phaseD_occupancy_d2r4_provider_scene0050 \
  --variants G3,G5 \
  --nn-radius 0.05 \
  --density-alpha 2.0
```

Smoke 结果:

| row | AP | AP50 | AP25 | pre% | hit |
|---|---:|---:|---:|---:|---:|
| scene0050 D2r4 G3 | 0.111274 | 0.161491 | 0.639234 | 0.062297 | 0.749542 |
| scene0050 D2r4 G5 | 0.115044 | 0.252857 | 0.545190 | 0.069577 | 0.957777 |

Probe5 D2r4 G3/G5:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1 \
  --output-prefix stream4d_v21_3_occupancy_d2r4_provider_probe5_r1 \
  --audit-root outputs/audit/v21_3_phaseD_occupancy_d2r4_provider_probe5 \
  --variants G3,G5 \
  --nn-radius 0.05 \
  --density-alpha 2.0
```

结果:

| row | AP | AP50 | AP25 | pre% | hit | empty | #pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| D2r4 G3 | 0.104381 | 0.247014 | 0.552210 | 0.044965 | 0.679178 | 0.099930 | 53.6 |
| D2r4 G5 | 0.086880 | 0.183997 | 0.462698 | 0.056575 | 0.900515 | 0.053667 | 77.2 |

结论: D2r4 support 能进入 provider/AP，但直接 G3/G5 不够，且 density G5 是负修复。

### 18.3 D5 cache/provider

D5 probe5 cache export 并行命令模板:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.export_v21_3_occupancy_carrier_cache \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name <scene> \
  --variant D5 \
  --frame-stride 10 \
  --max-frames 64 \
  --window-size 32 \
  --window-stride 16 \
  --query-budget 7168 \
  --source-points-per-round 512 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-root outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1 \
  --summary-root outputs/audit/v21_3_occupancy_provider_cache_d5_warmstart64
```

D5 cache export 结果:

| scene | GPU | queries | carriers saved | warmstart | time sec |
|---|---:|---:|---:|---:|---:|
| scene0050_00 | 0 | 21504 | 21362 | 9062 | 157.112847 |
| scene0011_00 | 1 | 21504 | 21291 | 8823 | 161.125576 |
| scene0030_00 | 2 | 21504 | 21355 | 10602 | 165.903678 |
| scene0081_01 | 3 | 21504 | 21284 | 7302 | 174.443926 |
| scene0591_00 | 4 | 21504 | 21220 | 8673 | 180.483020 |

D5 provider:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1 \
  --output-prefix stream4d_v21_3_occupancy_d5_provider_probe5_r1 \
  --audit-root outputs/audit/v21_3_phaseD_occupancy_d5_provider_probe5 \
  --variants G3,G5 \
  --nn-radius 0.05 \
  --density-alpha 2.0
```

结果:

| row | AP | AP50 | AP25 | pre% | hit | empty | sim3 p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D5 G3 | 0.025905 | 0.054767 | 0.465900 | 0.013054 | 0.166395 | 0.458584 | 1.696078 |
| D5 G5 | 0.017455 | 0.042783 | 0.270536 | 0.024586 | 0.372644 | 0.316338 | 1.696078 |

结论: D5 warmstart 在 occupancy coverage 上有效，但 provider/AP 是负修复；accepted tubes 的 metric consistency 很差。

### 18.4 D2r4 interior/outlier provider sweep

命令模板:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1 \
  --output-prefix <output_prefix> \
  --audit-root <audit_root> \
  --variants <G3|G4|G7|G8> \
  --nn-radius <radius> \
  --density-alpha 2.0 \
  --min-mask-interior-px <px>
```

结果:

| row | AP | AP50 | AP25 | pre% | hit | empty | #pred | interior px | nn radius |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2r4 G7 i2 | 0.131037 | 0.275046 | 0.588235 | 0.042806 | 0.615016 | 0.121368 | 50.0 | 2.0 | 0.050 |
| D2r4 G8 i2 | 0.100882 | 0.244156 | 0.441371 | 0.053493 | 0.812637 | 0.068842 | 75.2 | 2.0 | 0.124 |
| D2r4 G3 i4 | 0.158133 | 0.274194 | 0.577545 | 0.038769 | 0.518655 | 0.197842 | 46.0 | 4.0 | 0.050 |
| D2r4 G3 i5 | 0.178432 | 0.320269 | 0.557541 | 0.037032 | 0.483039 | 0.232110 | 43.4 | 5.0 | 0.050 |
| D2r4 G3 i6 | 0.188584 | 0.293836 | 0.556523 | 0.034907 | 0.443836 | 0.273008 | 43.6 | 6.0 | 0.050 |
| D2r4 G3 i7 | 0.165676 | 0.329428 | 0.610881 | 0.033233 | 0.415857 | 0.304924 | 39.2 | 7.0 | 0.050 |
| D2r4 G3 i8 | 0.176619 | 0.318841 | 0.578814 | 0.031443 | 0.388774 | 0.331040 | 38.8 | 8.0 | 0.050 |
| D2r4 G3 i6 r0.03 | 0.148719 | 0.313457 | 0.522748 | 0.026865 | 0.310994 | 0.317438 | 34.6 | 6.0 | 0.030 |
| D2r4 G3 i6 r0.07 | 0.148766 | 0.285707 | 0.566323 | 0.039151 | 0.520232 | 0.253598 | 47.6 | 6.0 | 0.070 |
| D2r4 G4 i5 | 0.162343 | 0.273845 | 0.542824 | 0.033586 | 0.434551 | 0.254694 | 43.2 | 5.0 | 0.050 |
| D2r4 G4 i6 | 0.198368 | 0.315906 | 0.628571 | 0.031738 | 0.399259 | 0.296387 | 40.6 | 6.0 | 0.050 |
| D2r4 G4 i7 | 0.198176 | 0.357224 | 0.585234 | 0.030273 | 0.374129 | 0.326661 | 40.2 | 7.0 | 0.050 |

Best:

```text
D2r4 G4 interior6 AP/AP50/AP25 = 0.198368 / 0.315906 / 0.628571
delta AP vs G0 = -0.126580
delta AP vs dense128+interior G7 = +0.054233
```

### 18.5 D2r4 raw/self no-eval-Sim3 provider check

目的: 检查 D2r4 occupancy carrier cache 在不使用 eval-Sim3 GT/RGB-D 对齐时是否已经能作为 method 候选。该实验只跑 G1/G2/G6，不使用 `uses_gt_sim3_for_prediction`。

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_v4_1_probe5.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d2r4_probe5_r1 \
  --output-prefix stream4d_v21_3_occupancy_d2r4_rawself_provider_probe5_r1 \
  --audit-root outputs/audit/v21_3_phaseD_occupancy_d2r4_rawself_provider_probe5 \
  --variants G1,G2,G6 \
  --nn-radius 0.05 \
  --density-alpha 2.0
```

结果:

| row | status | AP | AP50 | AP25 | pre% | hit | empty | #pred | error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| D2r4 G1 raw | ok | 0.032407 | 0.072917 | 0.625000 | 0.001710 | 0.013862 | 0.894686 | 3.2 | NA |
| D2r4 G2 self-stitched | ok | 0.032407 | 0.072917 | 0.625000 | 0.001710 | 0.013862 | 0.894686 | 3.2 | NA |
| D2r4 G6 self-density | failed | NA | NA | NA | 0.003370 | 0.042824 | 0.856939 | 8.25 | `scene0591_00: IndexError: index -1 is out of bounds for axis 0 with size 0` |

结论:

- G1/G2 在 no-eval-Sim3 下仍远低于 G0，AP/AP50 只有 `0.032407/0.072917`，pre% 只有 `0.001710`。
- G2 与 G1 完全一致，因为 probe5 D2r4 cache 每 scene 只有 1 个 window，`self_stitch_pair_count=0`。
- G6 density 路径在 scene0591 失败，且 partial stats 仍只有 pre% `0.003370`、hit `0.042824`。
- 因此 D2r4 G4 interior6 的 `0.198368` 仍是 eval-Sim3 diagnostic improvement，不是 method success。

Artifact:

- `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_rawself_provider_probe5/D4RT_geometry_replacement_stream3d_probe5.*`
- `Stream3D/outputs/audit/v21_3_phaseD_occupancy_d2r4_rawself_provider_probe5/*provider_diagnostics.json`

### 18.6 audit scan and final tests

Reportable scan:

```bash
CONFIGS=$(find data/prediction -maxdepth 1 -type d -name 'stream4d_v21_3_occupancy*_provider*_class_agnostic' -printf '%f\n' | sed 's/_class_agnostic$//' | sort | paste -sd, -)
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v21_3_phaseD_occupancy_provider_reportable_scan.md \
  --require-manifest \
  --require-eval-policy
```

结果:

```text
exit code = 5
num_configs = 21
num_diagnostic_only_configs = 21
num_reportable_method_configs = 0
num_configs_missing_manifest = 0
num_configs_missing_eval_policy = 0
num_suspicious_configs = 18
num_uses_gt_for_prediction = 18
num_uses_gt_for_prediction_and_method_result = 0
num_alignment_used_for_prediction = 18
```

解释: exit code 5 是预期审计结果，因为 18 个 eval-Sim3 diagnostic 必须显式标 `uses_gt_for_prediction=true`。新增 3 个 raw/self config 不使用 GT Sim3，但也都是 diagnostic-only / forbidden-for-method-table。关键是 `method_result=0`，没有 GT method 泄漏。

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 56 tests in 1.638s
OK
```

最终判定:

- 本轮得到一个更强 diagnostic repair: D2r4 G4 interior6 `0.198368/0.315906/0.628571`。
- 但它仍依赖 eval-Sim3 GT/RGB-D 对齐，且低于 G0 `0.324948/0.497839/0.650992`。
- D2r4 raw/self no-eval-Sim3 仍失败: G1/G2 AP `0.032407`，G6 scene0591 失败。
- D5 warmstart provider 是负修复。
- 继续不启动 Phase E/F，不生成 method table。

## 19. D2r4 overlap-window self-stitch smoke

动机: 18.5 发现 probe5 D2r4 raw/self 中 G2 与 G1 完全一致，因为每个 scene 只有 1 个 D2 window，`self_stitch_pair_count=0`。按计划中的 self-Sim3 / overlap 修复方向，本轮排查“D2 无 overlap 是否掩盖了 self-stitch 的效果”。

代码修改:

- `Stream3D/tools/export_v21_3_occupancy_carrier_cache.py`
  - D2 export 从单 window 改成当 `max_frames > window_size` 时使用 `window_size/window_stride` 生成 sliding windows。
  - D2 每个 window 继续用 `CarrierSampler` 的 stable carrier id；该 id 基于 global frame/x/y，overlap frame 的同一 source pixel 会在跨窗口共享 carrier id。
  - 输出 summary 新增 `num_windows/window_size/window_stride`。
- `Stream3D/tests/test_native_occupancy_and_builder.py`
  - 新增 `test_d2_export_uses_overlap_windows_with_stable_ids`，用 fake stream/adapter 验证 6 帧、window 4/stride 2 导出 2 个 windows，且 overlap carrier id 有交集。

验证:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/export_v21_3_occupancy_carrier_cache.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_occupancy_and_builder
```

结果:

```text
py_compile OK
Ran 9 tests in 0.008s
OK
```

### 19.1 scene0050 D2r4 64f overlap-window cache

命令:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.export_v21_3_occupancy_carrier_cache \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --variant D2 \
  --frame-stride 10 \
  --max-frames 64 \
  --window-size 32 \
  --window-stride 16 \
  --query-chunk-size 2048 \
  --mark-radius-px 4 \
  --output-root outputs/stream4d_debug_v21_3_occupancy_d2r4_win64_scene0050_r1 \
  --summary-root outputs/audit/v21_3_occupancy_provider_cache_d2r4_win64_scene0050
```

结果:

| scene | windows | queries/carriers | total D4RT sec | window0 uv | window1 uv | window2 uv |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 3 | 57152 | 191.969105 | 0.393523 | 0.373964 | 0.331015 |

窗口:

| window | frame start | frame end | carriers | sec |
|---:|---:|---:|---:|---:|
| 0 | 0 | 310 | 22016 | 73.693349 |
| 1 | 160 | 470 | 22432 | 75.240081 |
| 2 | 320 | 630 | 12704 | 43.035674 |

### 19.2 scene0050 provider smoke

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d2r4_win64_scene0050_r1 \
  --output-prefix stream4d_v21_3_occupancy_d2r4_win64_scene0050_provider_r1 \
  --audit-root outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_provider \
  --variants G0,G1,G2,G3,G4,G6,G9,G10 \
  --nn-radius 0.05 \
  --density-alpha 2.0
```

结果:

| row | AP | AP50 | AP25 | pre% | hit | empty | #pred | self pairs | self fail | residual p90 | scale std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 RGB-D baseline | 0.263124 | 0.462112 | 0.669151 | 0.336807 | NA | NA | 86.0 | NA | NA | NA | NA |
| G1 raw | 0.000000 | 0.000000 | 0.225000 | 0.017568 | 0.050440 | 0.577493 | 32.0 | NA | NA | NA | NA |
| G2 self-stitched | 0.000000 | 0.000000 | 0.555556 | 0.007043 | 0.025262 | 0.858742 | 15.0 | 2.0 | 0.0 | 0.143379 | 0.008392 |
| G3 eval-Sim3 | 0.064815 | 0.194444 | 0.750000 | 0.006490 | 0.232136 | 0.350870 | 14.0 | NA | NA | 2.185261 | NA |
| G4 eval-Sim3 outlier | 0.000000 | 0.000000 | 0.500000 | 0.006211 | 0.208822 | 0.378279 | 15.0 | NA | NA | 2.185261 | NA |
| G6 self-density | 0.058201 | 0.214286 | 0.357143 | 0.012242 | 0.077385 | 0.828692 | 14.0 | 2.0 | 0.0 | 0.143379 | 0.008392 |
| G9 scale-normalized | 0.027778 | 0.250000 | 0.250000 | 0.007474 | 0.025398 | 0.858742 | 13.0 | 2.0 | 0.0 | 0.146869 | 0.034045 |
| G10 scale-normalized density | 0.030093 | 0.090278 | 0.454545 | 0.013240 | 0.080160 | 0.823519 | 14.0 | 2.0 | 0.0 | 0.146869 | 0.034045 |

判定:

- 修复确实让 D2 有 3 个 windows 和 2 个 self-stitch pairs，`self_stitch_fail_count=0`。
- 但 G2 AP 仍为 `0.0`，G6 只有 `0.058201`，G9/G10 只有 `0.027778/0.030093`，没有接近 G0 `0.263124`。
- eval-Sim3 G3 也只有 `0.064815`，G4 为 `0.0`；64f overlap-window 没有比 32f D2r4 scene0050 diagnostic 更好。
- 因此本 repair 是负证据，不扩到 probe5；继续不启动 Phase E/F。

### 19.3 final scan/tests after overlap-window smoke

Reportable scan:

```bash
CONFIGS=$(find data/prediction -maxdepth 1 -type d -name 'stream4d_v21_3_occupancy*_provider*_class_agnostic' -printf '%f\n' | sed 's/_class_agnostic$//' | sort | paste -sd, -)
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v21_3_phaseD_occupancy_provider_reportable_scan.md \
  --require-manifest \
  --require-eval-policy
```

结果:

```text
exit code = 5
num_configs = 29
num_diagnostic_only_configs = 29
num_reportable_method_configs = 0
num_configs_missing_manifest = 0
num_configs_missing_eval_policy = 0
num_suspicious_configs = 20
num_uses_gt_for_prediction = 20
num_uses_gt_for_prediction_and_method_result = 0
num_alignment_used_for_prediction = 20
```

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 57 tests in 1.654s
OK
```

本轮最终判定:

- D2 overlap-window exporter 修复合理且有测试覆盖。
- scene0050 smoke 表明 self-stitch pair 充足时仍无法形成 method-level support。
- 当前无可报告方法成功；Phase E/F 仍按 stop rules 不运行。

## 20. 用户质疑后的 self-stitch 实现自查与尺度超参 sweep

用户指出: self-stitch 没道理会更差，可能存在实现 bug；同时尺度不一样时超参也要跟着变化。本轮据此做两类自查:

1. 实现自查: 检查 self-stitch transform 方向、overlap frame 多窗口 union 是否污染。
2. 尺度自查: 对 fixed `nn_radius` 和 density spacing-aware radius 做 sweep。

代码修改:

- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 新增 `overlap_policy={"all","best_confidence"}`。
  - G2/G6/G9/G10 使用 `best_confidence` 后，对 overlap frame 只选择一个质量最高窗口，避免多个窗口重复投影污染 set-cover。
  - diagnostics 新增 `candidate_source_windows` 和 `overlap_policy`。
- `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - 新增 `--overlap-policy`。
  - G2/G6/G9/G10 默认 `overlap_policy=best_confidence`。
- `Stream3D/tests/test_v21_3_geometry_provider.py`
  - 新增 `test_carrier_provider_best_confidence_overlap_policy_selects_one_window`。
  - 新增 `test_self_stitch_transform_maps_later_window_to_canonical_points`，用已知 scale/translation 验证 self-stitch 方向。

验证:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile geometry_provider/d4rt_carrier_provider.py tools/run_v21_3_stream3d_provider_replacement.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v21_3_geometry_provider
```

结果:

```text
py_compile OK
Ran 5 tests in 0.015s
OK
```

### 20.1 best-window rerun

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d2r4_win64_scene0050_r1 \
  --output-prefix stream4d_v21_3_occupancy_d2r4_win64_scene0050_provider_bestwin_r1 \
  --audit-root outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_provider_bestwin \
  --variants G0,G1,G2,G3,G4,G6,G9,G10 \
  --nn-radius 0.05 \
  --density-alpha 2.0
```

结果:

| row | AP | AP50 | AP25 | pre% | hit | source windows | candidate windows | nn radius |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 RGB-D baseline | 0.263124 | 0.462112 | 0.669151 | 0.336807 | NA | NA | NA | NA |
| G1 raw all-window | 0.000000 | 0.000000 | 0.225000 | 0.017568 | 0.050440 | 1.5 | 1.5 | 0.050000 |
| G2 self best-window | 0.000000 | 0.000000 | 0.133333 | 0.005236 | 0.022957 | 1.0 | 1.5 | 0.050000 |
| G3 eval-Sim3 | 0.064815 | 0.194444 | 0.750000 | 0.006490 | 0.232136 | 1.5 | 1.5 | 0.050000 |
| G4 eval-Sim3 outlier | 0.000000 | 0.000000 | 0.500000 | 0.006211 | 0.208822 | 1.5 | 1.5 | 0.050000 |
| G6 self-density best-window | 0.000000 | 0.000000 | 0.375000 | 0.009673 | 0.071769 | 1.0 | 1.5 | 0.158941 |
| G9 scale-normalized best-window | 0.000000 | 0.000000 | 0.500000 | 0.005610 | 0.023278 | 1.0 | 1.5 | 0.050000 |
| G10 scale-normalized density best-window | 0.031250 | 0.093750 | 0.343750 | 0.010520 | 0.075393 | 1.0 | 1.5 | 0.163794 |

结论: best-window 修掉了 all-window overlap union 的污染风险，但没有带来 AP 改善。

### 20.2 fixed radius sweep

命令模板:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d2r4_win64_scene0050_r1 \
  --output-prefix stream4d_v21_3_occupancy_d2r4_win64_scene0050_self_r<radius>_r1 \
  --audit-root outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_r<radius> \
  --variants G2,G9 \
  --nn-radius <0.02|0.05|0.10|0.20> \
  --density-alpha 2.0
```

结果:

| radius | G2 AP/AP50/AP25 | G2 pre% | G2 hit | G9 AP/AP50/AP25 | G9 pre% | G9 hit |
|---:|---|---:|---:|---|---:|---:|
| 0.02 | 0.000000 / 0.000000 / 0.250000 | 0.002791 | 0.008265 | 0.000000 / 0.000000 / 0.250000 | 0.002867 | 0.008073 |
| 0.05 | 0.000000 / 0.000000 / 0.133333 | 0.005236 | 0.022957 | 0.000000 / 0.000000 / 0.500000 | 0.005610 | 0.023278 |
| 0.10 | 0.000000 / 0.000000 / 0.400000 | 0.007578 | 0.045393 | 0.032407 / 0.097222 / 0.233333 | 0.008155 | 0.045900 |
| 0.20 | 0.000000 / 0.000000 / 0.375000 | 0.010828 | 0.089884 | 0.000000 / 0.000000 / 0.481481 | 0.011494 | 0.092293 |

### 20.3 density-alpha scale-aware sweep

命令模板:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list splits/scannet_scene0050.txt \
  --debug-root outputs/stream4d_debug_v21_3_occupancy_d2r4_win64_scene0050_r1 \
  --output-prefix stream4d_v21_3_occupancy_d2r4_win64_scene0050_self_a<alpha>_r1 \
  --audit-root outputs/audit/v21_3_phaseD_occupancy_d2r4_win64_scene0050_self_a<alpha> \
  --variants G6,G10 \
  --nn-radius 0.02 \
  --density-alpha <0.5|1.0|2.0|4.0>
```

结果:

| alpha | G6 radius | G6 AP/AP50/AP25 | G6 pre% | G6 hit | G10 radius | G10 AP/AP50/AP25 | G10 pre% | G10 hit |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 0.5 | 0.039735 | 0.000000 / 0.000000 / 0.187500 | 0.004593 | 0.017848 | 0.040948 | 0.037037 / 0.166667 / 0.250000 | 0.004891 | 0.018758 |
| 1.0 | 0.079470 | 0.000000 / 0.000000 / 0.357143 | 0.006731 | 0.036271 | 0.081897 | 0.000000 / 0.000000 / 0.400000 | 0.007393 | 0.037971 |
| 2.0 | 0.158941 | 0.000000 / 0.000000 / 0.375000 | 0.009673 | 0.071769 | 0.163794 | 0.031250 / 0.093750 / 0.343750 | 0.010520 | 0.075393 |
| 4.0 | 0.317882 | 0.000000 / 0.000000 / 0.196970 | 0.012422 | 0.116439 | 0.327587 | 0.000000 / 0.000000 / 0.576923 | 0.013197 | 0.119539 |

结论:

- 尺度超参确实关键: hit/pre 随 radius/alpha 增大而上升。
- 但 AP 不单调，过大半径继续污染。
- 最佳 no-GT self-stitch row 是 G10 alpha0.5 `0.037037/0.166667/0.250000`，仍远低于 G0 `0.263124/0.462112/0.669151`。
- 因此本轮没有证明 self-stitch 本身无效；只证明当前实现修掉 obvious overlap selection 风险并做尺度 sweep 后，D2r4 self-aligned support 仍不足。

### 20.4 final scan/tests after self-stitch audit

注意: 19.3 的 scan glob 只覆盖名字里带 `provider` 的 configs，漏掉了 `self_r*` / `self_a*` sweep。这里改用完整 occupancy glob。

```bash
CONFIGS=$(find data/prediction -maxdepth 1 -type d -name 'stream4d_v21_3_occupancy*_class_agnostic' -printf '%f\n' | sed 's/_class_agnostic$//' | sort | paste -sd, -)
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v21_3_phaseD_occupancy_all_reportable_scan.md \
  --require-manifest \
  --require-eval-policy
```

结果:

```text
exit code = 5
num_configs = 53
num_diagnostic_only_configs = 53
num_reportable_method_configs = 0
num_configs_missing_manifest = 0
num_configs_missing_eval_policy = 0
num_suspicious_configs = 22
num_uses_gt_for_prediction = 22
num_uses_gt_for_prediction_and_method_result = 0
num_alignment_used_for_prediction = 22
```

完整测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 59 tests in 1.642s
OK
```

本轮最终判定:

- 用户质疑是合理的: 固定 `nn_radius=0.05` 不能作为 self-stitch 失败的唯一证据。
- 修正 overlap-window selection 并做 scale-aware sweep 后，仍没有得到 method-level AP。
- 当前继续不启动 Phase E/F。

## 21. v21.3 代码审核包打包

目的: 按用户要求，把 v21.3 相关源码、runner、tests、split 与计划/执行/复盘文档打成窄范围代码审核包。包内不包含 `Stream3D/outputs`、`Stream3D/data`、`Open-d4rt` checkpoint/source tree、`__pycache__` 或大实验 artifact。

打包范围:

- v21.3 文档:
  - `docs/stream4d_v21_3_gt_guard_opend4rt_geometry_diagnostic_plan_for_codex.md`
  - `docs/stream4d_v21_3_执行日志.md`
  - `docs/stream4d_v21_3_实验结果复盘.md`
- Stream3D provider / native core:
  - `Stream3D/stream4d_native/*.py`
  - `Stream3D/stream4d_native/OPEND4RT_SOURCE_NOTES.md`
  - `Stream3D/geometry_provider/*.py`
  - `Stream3D/main.py`
  - `Stream3D/graph/construction.py`
  - `Stream3D/utils/config.py`
  - `Stream3D/utils/mask_backprojection.py`
  - `Stream3D/utils/Stream3D.py`
  - `Stream3D/evaluation/evaluate.py`
  - `Stream3D/configs/*.json`
- Stream4D helper code used by v21.3 runner/cache/provider path:
  - `Stream3D/stream4d/*.py`
- v21.3 runner/audit tools:
  - `Stream3D/tools/audit_opend4rt_source_alignment.py`
  - `Stream3D/tools/audit_stream3d_eval_protocol.py`
  - `Stream3D/tools/check_stream4d_env.py`
  - `Stream3D/tools/export_v21_3_occupancy_carrier_cache.py`
  - `Stream3D/tools/native_geometry_diagnostics.py`
  - `Stream3D/tools/prediction_manifest.py`
  - `Stream3D/tools/run_v21_3_native_occupancy_ablation.py`
  - `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - `Stream3D/tools/scan_native_manifests.py`
  - `Stream3D/tools/scan_reportable_configs.py`
  - `Stream3D/tools/update_config_manifest_fields.py`
  - `Stream3D/tools/verify_stream4d_metric_integrity.py`
- tests/splits:
  - `Stream3D/tests/*.py`
  - `Stream3D/splits/*.txt`

输出文件:

- `stream4d_v21_3_code_review_packet_20260610.zip`
- `stream4d_v21_3_code_review_packet_20260610.sha256`
- `stream4d_v21_3_code_review_packet_20260610_filelist.txt`
- `stream4d_v21_3_code_review_packet_20260610_git_status.txt`
- `stream4d_v21_3_code_review_packet_20260610_git_diff.patch`
- `stream4d_v21_3_code_review_packet_20260610_ziptest.log`

执行命令摘要:

```bash
git status --short > stream4d_v21_3_code_review_packet_20260610_git_status.txt
git diff -- Stream3D/configs Stream3D/evaluation Stream3D/graph Stream3D/main.py Stream3D/utils > stream4d_v21_3_code_review_packet_20260610_git_diff.patch
zip -r stream4d_v21_3_code_review_packet_20260610.zip stream4d_v21_3_code_review_packet_20260610
zip -T stream4d_v21_3_code_review_packet_20260610.zip
sha256sum stream4d_v21_3_code_review_packet_20260610.zip > stream4d_v21_3_code_review_packet_20260610.sha256
```
