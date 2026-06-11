# Stream4D v13 执行日志

日期: 2026-06-09  
计划文档: `docs/stream4d_v13_object_explanation_reboot_experiment_plan_for_codex.md`  
结果复盘: `docs/stream4d_v13_实验结果复盘.md`  
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`  
Stream3D 目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D`  
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`  
GPU: `CUDA_VISIBLE_DEVICES=6,7`

## 执行原则

- 不编造数据，不把 `NA`、`nan`、空预测、GT oracle 或 diagnostic-only 结果写成 method success。
- 所有 AP/AP50/AP25 引用 `Stream3D/data/evaluation/scannet/*_class_agnostic.txt` 或统一评估汇总脚本从这些文件读取的结果。
- GT oracle 只作为 diagnostic upper bound，manifest 必须写明 `uses_gt_for_prediction=true`、`uses_gt_for_diagnostic=true`、`is_diagnostic_only=true`、`is_method_result=false`。
- Reportable method config 必须无 GT prediction，且 manifest/eval policy 扫描通过。
- Probe5 Phase D/E gate 未通过时，不启动 tune30/final。

## 初始代码状态

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
git status --short
```

观察:

- 工作树开始时已有大量既有修改和未跟踪文件，包括 v12 Stream3D 代码、数据、输出、审计包和其他 ACL2 文档。
- 本轮只追加/修改 v13 相关代码、脚本、测试、日志与必要的 Stream4D support code，不回滚已有用户/历史产物。

## 计划阅读摘要

v13 目标不是继续调 v12 M5，而是重启 object explanation 路线:

```text
D4RT surfels + sparse/densified semantic measurements
-> video masklet measurement density
-> MDL-style global object explanation / set packing
-> posterior surfel ownership support export
-> unified own/S0/S1/Stream3D-on-M evaluation
```

最低交付:

```text
Phase A code audit + pure tests
Phase B candidate/oracle failure attribution matrix
Phase C video masklet density diagnostic
Phase D MDL object explanation prototype
Phase E posterior support export prototype
Phase F D4RT geometry aligned -> Stream3D diagnostic
unified own/S0/S1/Stream3D-on-M table
visualizations with JSON sidecars where feasible
```

## Phase A 代码审计与实现

### A.1 lazy open3d import

修改文件:

- `Stream3D/stream4d/export_scannet.py`

修改内容:

- 将 `import open3d as o3d` 从 module top-level 移到 `ScanNetExporter.__init__()` 内部。
- 目的: 让 `score_export_record` 等 pure-python helper 在没有 `open3d` 的环境中也能导入。
- 若实际构造 `ScanNetExporter` 时缺少 `open3d`，会给出明确错误。

### A.2 posterior support export

修改文件:

- `Stream3D/stream4d/export_scannet.py`

修改内容:

- 新增 `export_support_mode="posterior_support"`。
- 新增 `export_object_slot_posterior_support(input_object_dict, bank)`，从 slot posterior 的 core/fringe surfels 反投影到可见帧点，unknown/reject surfels 不导出。
- 记录诊断字段: `posterior_core_surfels`、`posterior_fringe_surfels`、`posterior_unknown_surfels_not_exported`、`posterior_reject_surfels_not_exported`、`posterior_core_exported_points`、`posterior_fringe_candidate_points`、`posterior_fringe_kept_points`、`posterior_connected_component_count`。
- 后续 M13d 修复中给 posterior export 加入 WTA: 同一 mesh point 若被多个 object 占用，只保留 score 更高的 object。

### A.3 新增 v13 模块/工具

新增文件:

```text
Stream3D/stream4d/video_masklet.py
Stream3D/stream4d/object_explanation_mdl.py
Stream3D/tools/build_v13_video_masklet_measurements.py
Stream3D/tools/export_v13_masklet_candidates.py
Stream3D/tools/select_v13_unsupervised_candidate_pool.py
Stream3D/tools/export_v13_object_explanation_mdl.py
Stream3D/tools/summarize_v13_geometry_diagnostic.py
Stream3D/tools/make_v13_failure_visuals.py
Stream3D/scripts/reproduce_v13_masklet_density_probe5.sh
Stream3D/scripts/reproduce_v13_candidate_attribution_probe5.sh
Stream3D/scripts/reproduce_v13_object_mdl_probe5.sh
Stream3D/scripts/reproduce_v13_object_mdl_repairs_probe5.sh
Stream3D/scripts/reproduce_v13_geometry_diagnostic_probe5.sh
Stream3D/scripts/reproduce_v13_visuals_probe5.sh
Stream3D/scripts/v13_candidate_unsup_matrix_probe5.json
Stream3D/scripts/v13_candidate_oracle_matrix_probe5.json
Stream3D/scripts/v13_object_mdl_matrix_probe5.json
```

### A.4 新增测试

新增测试文件:

```text
Stream3D/tests/test_protocol_pure_python.py
Stream3D/tests/test_manifest_and_eval_policy.py
Stream3D/tests/test_object_explanation_pure.py
Stream3D/tests/test_export_scannet_open3d.py
Stream3D/tests/test_d4rt_gpu_optional.py
```

覆盖内容:

- `MeasurementBank` save/load roundtrip。
- `birth_groups` 按 frame/mask 拆分。
- `posterior_for_group` negative evidence reject 行为。
- `measurement_votes` deterministic + dedup 行为。
- manifest scanner 拒绝 `uses_gt_for_prediction` 的 method result。
- method-table policy 拒绝 diagnostic-only method。
- `open3d` 和 GPU 测试作为 optional dependency 测试；环境不可用时 skip。

验证命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile: pass
unittest discover tests: Ran 27 tests in 1.416s ... OK
```

## Phase C Video Masklet Measurement Density

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v13_masklet_density_probe5.sh
```

关键输出:

```text
outputs/v13_masklet_measurements/C1/*/masklets.npz
outputs/v13_masklet_measurements/C2/*/masklets.npz
outputs/v13_masklet_measurements/C3/*/masklets.npz
outputs/audit/v13_masklet_density/masklet_density_probe5.json
outputs/audit/v13_masklet_density/masklet_density_probe5.csv
outputs/audit/v13_masklet_density/masklet_density_probe5.md
logs/stream4d_v13_masklet_density_probe5.log
```

摘要:

- C0 semantic frames/surfel: `1.6151`，unobserved: `0.1458`。
- C1/C2/C3 semantic frames/surfel: `1.5619`，masklets: `247.80`，frames/birth: `14.4660`，agreement: `0.9925`，unobserved: `0.8832`。
- C1/C2/C3 在 probe5 上相同，说明当前 gate 没有进一步过滤出不同有效集合。

执行判断:

- Masklet 产生了多帧 birth 轨迹，但没有增加 broad surfel coverage；unobserved surfel ratio 反而高于 C0。
- 这是 v13 的第一处真实负结果，不写成 density 成功。

## Phase B Candidate/Oracle 归因矩阵

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v13_candidate_attribution_probe5.sh
```

关键输出:

```text
outputs/audit/v13_candidate_attribution/candidate_unsup_matrix_probe5.json
outputs/audit/v13_candidate_attribution/candidate_unsup_matrix_probe5.csv
outputs/audit/v13_candidate_attribution/candidate_unsup_matrix_probe5.md
outputs/audit/v13_candidate_attribution/candidate_oracle_matrix_probe5.json
outputs/audit/v13_candidate_attribution/candidate_oracle_matrix_probe5.csv
outputs/audit/v13_candidate_attribution/candidate_oracle_matrix_probe5.md
outputs/audit/v13_candidate_attribution/stream4d_v13_oracle_*_upper_bound.{json,csv,md}
outputs/v13_candidate_unsupervised/*
outputs/v13_masklet_candidates/*
logs/stream4d_v13_candidate_*.log
```

摘要:

- 最强 unsupervised tiny support primitive 是 `C_surfel unsup own`: `AP/AP50/AP25 = 0.228316 / 0.460285 / 0.778069`，support `4.2916%`。
- `C_masklet unsup own`: `0.062802 / 0.267185 / 0.517357`，support `2.1988%`。
- `C_hybrid unsup own`: `0.023515 / 0.066350 / 0.133871`，support `52.8088%`。
- `C_masklet oracle`: `0.183908 / 0.551724 / 0.926056`，support `2.1168%`。
- `C_hybrid oracle`: `0.256256 / 0.495495 / 0.702512`，support `52.8088%`，仍不是足够强的 broad-support upper bound。

执行判断:

- Oracle 使用 GT selection，只作 diagnostic upper bound。
- Masklet oracle 是 tiny/high AP25，上界不足以说明完整 reconstruction 能成功。
- Hybrid broad support 没有超过 v12 边界，说明 primitive attribution 和 coverage 仍是瓶颈。

## Phase D/E MDL Object Explanation + Posterior Support

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v13_object_mdl_probe5.sh
```

关键输出:

```text
outputs/audit/v13_object_explanation_mdl/object_mdl_matrix_probe5.json
outputs/audit/v13_object_explanation_mdl/object_mdl_matrix_probe5.csv
outputs/audit/v13_object_explanation_mdl/object_mdl_matrix_probe5.md
outputs/v13_object_explanation_mdl/stream4d_v13_m13a_mdl_c3_posterior_probe5_summary.{json,csv,md}
outputs/v13_object_explanation_mdl/stream4d_v13_m13b_mdl_c3_strict_probe5_summary.{json,csv,md}
outputs/v13_object_explanation_mdl/stream4d_v13_m13c_mdl_c3_fullmask_probe5_summary.{json,csv,md}
logs/stream4d_v13_mdl_*.log
```

原型结果:

- M13a posterior own: `0.145332 / 0.417796 / 0.698669`，support `2.0784%`，conflict `26.4328%`。
- M13b strict own: `0.159829 / 0.400813 / 0.678327`，support `2.0401%`，conflict `26.2442%`。
- M13c full-mask ablation own: `0.224575 / 0.419119 / 0.781728`，support `4.3855%`，conflict `55.8958%`。
- S0/S1 cross-support 均明显低于 gate。

Blocker:

- Posterior support 更干净但太小，full-mask support 稍大但冲突很高；Phase D/E gate 未通过。

按计划修复方向尝试:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v13_object_mdl_repairs_probe5.sh
```

修复修改:

- 在 `export_scannet.py` 的 posterior support export 中启用 WTA point assignment。
- 新增 M13d: `stream4d_v13_m13d_mdl_c3_posterior_wta_probe5`。

修复输出:

```text
outputs/v13_object_explanation_mdl/stream4d_v13_m13d_mdl_c3_posterior_wta_probe5_summary.{json,csv,md}
logs/stream4d_v13_mdl_repair_*.log
```

修复结果:

- M13d posterior WTA own: `0.161109 / 0.427857 / 0.793144`。
- conflict 从 M13a/M13b 的约 `26%` 降到 `0.0000%`。
- support 仍只有 `2.0522%`，S0 `0.000000 / 0.000000 / 0.000766`，S1 `0.000084 / 0.000377 / 0.023845`。

执行判断:

- WTA 修复了重叠冲突，但没有解决 tiny support 和 cross-support 崩溃。
- Gate 仍未通过，所以不启动 tune30/final。

## Phase F D4RT Geometry Diagnostic

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v13_geometry_diagnostic_probe5.sh
```

关键输出:

```text
outputs/audit/v13_geometry_diagnostic/d4rt_sim3_residual_probe5.json
outputs/audit/v13_geometry_diagnostic/d4rt_sim3_residual_probe5.csv
outputs/audit/v13_geometry_diagnostic/d4rt_sim3_residual_probe5.md
outputs/audit/v13_geometry_diagnostic/geometry_diagnostic_probe5.json
outputs/audit/v13_geometry_diagnostic/geometry_diagnostic_probe5.md
logs/stream4d_v13_geometry_*.log
```

摘要:

- G0 scannet: `AP/AP50/AP25 = 0.201139 / 0.344654 / 0.502268`。
- `stream4d_v10_g2` own: `0.188825 / 0.364384 / 0.486341`，on S0 `0.000000 / 0.000000 / 0.000112`，on S1 `0.000000 / 0.000000 / 0.004523`。
- 新 Sim3 residual diagnostic: `num_ok_windows=5/5`，`uv_in01_rate_mean=0.985845`，`visibility_mean=0.999855`，`confidence_mean=0.999931`，`sim3_residual_median_mean=0.468208`，`sim3_residual_p90_mean=0.859581`。

执行判断:

- D4RT carrier image-space quality 仍好，但 metric geometry replacement/cross-support 不好。
- v13 不把 D4RT geometry 写成直接改进。

## Visualizations

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=6,7 bash scripts/reproduce_v13_visuals_probe5.sh
```

输出:

```text
outputs/audit/v13_visuals/v13_failure_panel_00.png ... v13_failure_panel_19.png
outputs/audit/v13_visuals/v13_failure_panel_00.json ... v13_failure_panel_19.json
outputs/audit/v13_visuals/v13_failure_visuals_manifest.json
logs/stream4d_v13_visuals.log
```

说明:

- 共生成 20 个 PNG + 20 个 JSON sidecar + 1 个 manifest。
- 这些是 failure metric panel，不是逐 scene 3D mesh overlay；不把它们夸大成完整可视化审计。

## Final Audit

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
CONFIGS=stream4d_v13_c_mask_unsup_probe5,stream4d_v13_c_regionlet_unsup_probe5,stream4d_v13_c_surfel_unsup_probe5,stream4d_v13_c3_masklet_candidate_probe5,stream4d_v13_c_hybrid_unsup_probe5,stream4d_v13_m13a_mdl_c3_posterior_probe5,stream4d_v13_m13b_mdl_c3_strict_probe5,stream4d_v13_m13c_mdl_c3_fullmask_probe5,stream4d_v13_m13d_mdl_c3_posterior_wta_probe5
$PY -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py > logs/stream4d_v13_final_py_compile.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 $PY -m unittest discover tests > logs/stream4d_v13_final_unittest.log 2>&1
$PY -m tools.scan_reportable_configs --root . --configs "$CONFIGS" --output outputs/audit/v13_final/reportable_config_scan_v13_methods_probe5.md --require-manifest --require-eval-policy > logs/stream4d_v13_final_reportable_scan.log 2>&1
$PY -m tools.verify_stream4d_metric_integrity --orig-stream3d-root . --current-root . --configs "$CONFIGS" --seq-list splits/scannet_v6_probe5.txt --output outputs/audit/v13_final/metric_integrity_v13_methods_probe5.md --require-manifest > logs/stream4d_v13_final_metric_integrity.log 2>&1
```

结果:

```text
py_compile: pass
unittest discover tests: Ran 27 tests in 1.416s ... OK
reportable scan: num_reportable_method_configs=9, num_suspicious_configs=0, num_uses_gt_for_prediction=0
metric integrity: phase0_pass=True, evaluator_ap_core_equal_by_hash=True, gt_files_read_by_rescore=False
```

输出:

```text
outputs/audit/v13_final/reportable_config_scan_v13_methods_probe5.{json,csv,md}
outputs/audit/v13_final/metric_integrity_v13_methods_probe5.{json,csv,md}
outputs/audit/v13_final/pre_points_ratio_by_config.png
outputs/audit/v13_final/union_ratio_by_config.png
outputs/audit/v13_final/gt_crop_full_by_config.png
outputs/audit/v13_final/object_dict_alignment_iou_hist.png
```

## Tune30 / Final

未启动。

原因:

- 最强 M13 method-result 的 probe5 gate 未通过。
- `M13d` own AP/AP50/AP25 为 `0.161109 / 0.427857 / 0.793144`，own AP 与 AP50 未达 `0.30 / 0.55`。
- `M13d on S0` 和 `M13d on S1` 接近零，远低于 cross-support gate。
- 按计划，probe5 gate 失败时不进入 tune30/final。

## 审计材料

生成审计包:

```text
stream4d_v13_probe5_code_review_packet.zip
stream4d_v13_probe5_code_review_packet.sha256
stream4d_v13_probe5_filelist.txt
stream4d_v13_probe5_ziptest.log
stream4d_v13_probe5_git_diff.patch
stream4d_v13_probe5_git_status.txt
```

审计包包含:

- v13 执行日志与复盘日志。
- v13 新增/修改代码、工具、测试、复现实验脚本。
- v13 audit outputs、method prediction/TMP/evaluation artifacts、summary files、visual panel sidecars。
- `git diff` 与 `git status` 快照。
