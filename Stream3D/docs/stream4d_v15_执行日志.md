# Stream4D v15 执行日志

日期: 2026-06-09  
计划文档: `docs/stream4d_v15_executable_global_measurement_explanation_plan_for_codex.md`  
复盘文档: `docs/stream4d_v15_实验结果复盘.md`  
GPU: 用户说明显卡 6/7 可用；本轮命令默认使用 `CUDA_VISIBLE_DEVICES=6,7`。  
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`

## 0. 执行边界

- 不使用 GT 生成 reportable method prediction。
- GT 只用于 oracle selection、failure attribution、region purity/completeness diagnostic 和 standard evaluation。
- 本轮所有 v15 prediction configs 均为 diagnostic-only；没有 reportable method result。
- Oracle 输出 manifest 显式写入 `gt_selected_output=true`、`forbidden_for_method_table=true`。
- alignment 字段显式写入，且没有 method prediction 使用 GT/RGB-D alignment。
- Phase 1/2 gate 不通过，因此没有启动 Phase 3/4 global solver、Phase 5 method materialization、tune30/final。

## 1. 初始盘点

工作目录:

```bash
pwd
```

结果:

```text
/mnt/data/users/chengshun.wang/pjs/LoGeR
```

固定 probe5 split:

```bash
cat Stream3D/splits/scannet_v6_probe5.txt
```

结果:

```text
scene0050_00
scene0011_00
scene0030_00
scene0081_01
scene0591_00
```

环境确认:

```bash
python
```

结果:

```text
python: command not found
```

修复: 与 v14 一致，使用 loger conda env:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
```

关键输入已存在:

- `Stream3D/outputs/v14_measurement_bank_bank16_cropformer/*/measurement_bank.npz`
- `Stream3D/outputs/v14_surfel_atom_bank_bank16_target/A3/*/atom_bank.npz`
- `Stream3D/outputs/v14_surfel_atom_bank_bank16_target/A4/*/atom_bank.npz`
- `Stream3D/data/evaluation/scannet/stream4d_v14_*_class_agnostic.txt`

## 2. 代码修改

审计字段修复:

- `Stream3D/tools/prediction_manifest.py`
  - 新增 `gt_selected_output`
  - 新增 `forbidden_for_method_table`
  - 新增 `alignment_source`
  - 新增 `alignment_used_for_prediction`
  - 新增 `alignment_used_for_diagnostic`
- `Stream3D/tools/scan_reportable_configs.py`
  - 扫描并拦截 `gt_selected_output && is_method_result`
  - 扫描并拦截 `forbidden_for_method_table && is_method_result`
  - 扫描并拦截 `alignment_used_for_prediction`
- `Stream3D/tools/oracle_candidate_upper_bound.py`
  - oracle 输出 manifest 明确标记 `gt_selected_output=true` 和 `forbidden_for_method_table=true`

新增 v15 工具:

- `Stream3D/tools/diagnose_v15_union_oracle.py`
  - GT-read-only multi-measurement union oracle。
  - 对每个 GT 用 greedy set cover 从候选中选最多 `K=1,2,4,8,16,32` 个 measurement。
  - 输出 oracle diagnostic prediction，并用标准 evaluator 计算 AP/AP50/AP25。
- `Stream3D/tools/build_v15_mask_region_measurements.py`
  - 不读 GT 生成 mask-region measurement candidate。
  - 支持 `component`、`seed_voronoi`、`boundary_core` 三种 split/materialization 变体。
  - 可选 GT diagnostic 只在输出后计算 purity/completeness/contamination。
- `Stream3D/tests/test_v15_pure.py`
  - 覆盖 union greedy set cover。
  - 覆盖 v15 manifest forbidden oracle 字段。

新增复现脚本:

- `Stream3D/scripts/reproduce_v15_phase0_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase1_union_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase2_regions_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase6_geometry_probe5.sh`

## 3. Phase 0 审计

命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py > outputs/audit/v15_logs/py_compile_final.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests > outputs/audit/v15_logs/unittest_all_final.log 2>&1
```

结果:

```text
py_compile_final.log: empty, exit 0
unittest_all_final.log: Ran 30 tests in 1.570s ... OK
```

生成 v15 config 列表:

```bash
find data/prediction -maxdepth 1 -type d -name 'stream4d_v15*_class_agnostic' -printf '%f\n' \
  | sed 's/_class_agnostic$//' \
  | sort \
  > outputs/audit/v15_phase0_configs.txt
```

结果:

```text
40 configs
36 oracle diagnostic configs
4 measurement candidate diagnostic configs
```

Reportable scan:

```bash
CONFIGS=$(paste -sd, outputs/audit/v15_phase0_configs.txt)
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v15_phase0/reportable_config_scan_v15_probe5.md \
  --require-manifest \
  --require-eval-policy \
  > outputs/audit/v15_logs/v15_reportable_scan.log 2>&1
```

结果:

```text
num_configs=40
num_configs_missing_manifest=0
num_configs_missing_eval_policy=0
num_diagnostic_only_configs=40
num_oracle_configs=36
num_reportable_method_configs=0
num_suspicious_configs=0
num_uses_gt_for_prediction=0
num_gt_selected_output_and_method_result=0
num_forbidden_for_method_table_and_method_result=0
num_alignment_used_for_prediction=0
```

Metric integrity:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v15_phase0/metric_integrity_v15_probe5.md \
  --require-manifest \
  > outputs/audit/v15_logs/v15_metric_integrity.log 2>&1
```

结果:

```text
phase0_pass=True
gt_files_read_by_rescore=False
num_suspicious_configs=0
num_uses_gt_for_prediction=0
```

核心输出:

- `Stream3D/outputs/audit/v15_phase0/reportable_config_scan_v15_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase0/metric_integrity_v15_probe5.{json,md}`
- `Stream3D/outputs/audit/v15_phase0/pre_points_ratio_by_config.png`
- `Stream3D/outputs/audit/v15_phase0/union_ratio_by_config.png`
- `Stream3D/outputs/audit/v15_phase0/gt_crop_full_by_config.png`

## 4. Phase 1 Multi-Atom Union Oracle

目标: 验证 atom 作为 measurement basis 时，多个 atoms 联合解释一个 object 的 upper bound。

命令入口:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v14_a3_bank16_target_atom_candidate_probe5 \
  --pre-points-config stream4d_v14_a3_bank16_target_atom_candidate_probe5 \
  --output-config-prefix stream4d_v15_oracle_a3t16_union_probe5 \
  --summary-prefix outputs/audit/v15_phase1/a3t16_union_oracle_probe5 \
  --eval-support candidate \
  > outputs/audit/v15_logs/v15_phase1_a3t16_union_oracle.log 2>&1
```

同样运行:

- `stream4d_v14_a4_bank16_target_atom_candidate_probe5`
- `stream4d_v14_a4_bank16_target_minpts5_atom_candidate_probe5`

输出:

- `Stream3D/outputs/audit/v15_phase1/a3t16_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase1/a4t16_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase1/a4t16mp5_union_oracle_probe5.{json,csv,md}`
- `Stream3D/data/evaluation/scannet/stream4d_v15_oracle_a*t16*_union_probe5_k*_class_agnostic.txt`

关键结果:

| source | best K | oracle AP | AP50 | AP25 | candidate pre% | selected union pre% | mean best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A3 target atom | 8 | 0.297386 | 0.529412 | 0.764706 | 3.0295 | 2.0379 | 0.500310 | False |
| A4 target atom | 8 | 0.294118 | 0.529412 | 0.764706 | 3.0469 | 2.0379 | 0.499494 | False |
| A4 target minpts5 atom | 8 | 0.238095 | 0.485714 | 0.742857 | 3.4740 | 1.9971 | 0.458516 | False |

结论:

- 好消息: atom-as-measurement 有明显 upper-bound 信号。A3/A4 从 v14 single-candidate oracle AP50 `0.117647` 提升到 union oracle AP50 `0.529412`。
- 坏消息: candidate pre% 仍只有 `3.03-3.47%`，远低于 `25%` gate；AP50 也低于 `0.60`，AP25 对 A3/A4 为 `0.764706`，略低于 `0.78`。

## 5. Phase 2 Mask-Region Measurement Candidate

目标: 构建不把 mask/atom 直接当 object 的 measurement graph primitive，检查 region upper bound。

### R0 component

命令:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.build_v15_mask_region_measurements \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --output-config stream4d_v15_r0_component_region_probe5 \
  --region-root outputs/v15_mask_region_measurements \
  --summary-prefix outputs/audit/v15_phase2/r0_component_region_probe5 \
  --visual-dir outputs/audit/v15_phase2/visuals_r0 \
  --mode component \
  --min-surfels 5 \
  --min-region-pixels 80 \
  --max-regions-per-scene 600 \
  --pixel-stride 3 \
  --max-pixels-per-region 12000 \
  --export-nn-radius 0.05 \
  --min-export-points 20 \
  --gt-diagnostic \
  > outputs/audit/v15_logs/v15_phase2_r0_component_region.log 2>&1
```

结果:

```text
candidate AP/AP50/AP25 = 0.000132 / 0.000478 / 0.036858
exported pre% = 5.4552
export NN hit rate = 0.075273
area-weighted purity = 0.585791
contamination = 0.414209
best region IoU mean = 0.016472
gate=False
```

### R0b materialization radius repair

动机: R0 backproject hit rate 很低，按 Phase 5/materialization blocker 方向尝试将 `export_nn_radius` 从 `0.05` 放宽到 `0.10`。

命令差异:

```text
--output-config stream4d_v15_r0b_component_region_r010_probe5
--summary-prefix outputs/audit/v15_phase2/r0b_component_region_r010_probe5
--export-nn-radius 0.10
```

结果:

```text
candidate AP/AP50/AP25 = 0.000092 / 0.000475 / 0.036856
exported pre% = 5.6308
export NN hit rate = 0.077466
area-weighted purity = 0.575856
contamination = 0.424144
best region IoU mean = 0.016797
gate=False
```

结论: 放宽 radius 只把 pre% 从 `5.4552%` 提到 `5.6308%`，没有改善 AP/AP50，也略降低 purity。

### R1 seed_voronoi split repair

动机: R0 purity 低且 contamination 高，按文档方向增强 region split，使用 D4RT surfel seed 的 grid/Voronoi split。

命令差异:

```text
--output-config stream4d_v15_r1_seed_voronoi_region_probe5
--summary-prefix outputs/audit/v15_phase2/r1_seed_voronoi_region_probe5
--mode seed_voronoi
--split-grid 2
```

结果:

```text
candidate AP/AP50/AP25 = 0.000146 / 0.001065 / 0.025048
exported pre% = 4.5435
export NN hit rate = 0.055488
area-weighted purity = 0.697447
purity mean = 0.781139
contamination = 0.302553
completeness = 0.043924
best region IoU mean = 0.016551
gate=False
```

结论: split 修复确实提高 purity、降低 contamination，但 support 和 completeness 下降；直接 candidate AP 仍接近 0。

### R2 boundary_core repair

动机: 继续按 purity blocker 方向尝试保守边界 erosion。

命令差异:

```text
--output-config stream4d_v15_r2_boundary_core_region_probe5
--summary-prefix outputs/audit/v15_phase2/r2_boundary_core_region_probe5
--mode boundary_core
--erode-px 2
```

结果:

```text
candidate AP/AP50/AP25 = 0.000137 / 0.000495 / 0.036890
exported pre% = 5.3673
export NN hit rate = 0.074881
area-weighted purity = 0.587792
contamination = 0.412208
best region IoU mean = 0.016070
gate=False
```

结论: 简单 boundary erosion 没有解决 contamination，也没有改善 AP。

## 6. Phase 2 Region Union Oracle

命令入口:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v15_r1_seed_voronoi_region_probe5 \
  --pre-points-config stream4d_v15_r1_seed_voronoi_region_probe5 \
  --output-config-prefix stream4d_v15_oracle_r1_seed_voronoi_region_union_probe5 \
  --summary-prefix outputs/audit/v15_phase2/r1_seed_voronoi_region_union_oracle_probe5 \
  --eval-support candidate \
  --max-candidates-per-gt 256 \
  > outputs/audit/v15_logs/v15_phase2_r1_seed_voronoi_region_union_oracle.log 2>&1
```

同样对 R0/R2 运行。

结果:

| source | best K | oracle AP | AP50 | AP25 | candidate pre% | selected union pre% | mean best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0 component | 32 | 0.054374 | 0.170213 | 0.539435 | 5.4552 | 4.9739 | 0.296611 | False |
| R1 seed_voronoi | 8 | 0.130556 | 0.450000 | 0.800000 | 4.5435 | 3.8349 | 0.447398 | False |
| R2 boundary_core | 32 | 0.055270 | 0.171347 | 0.551510 | 5.3673 | 4.8855 | 0.298155 | False |

结论:

- R1 seed_voronoi 是最好的 mask-region variant，AP25 达到 `0.800000`，说明 split 后多 region 联合可解释一部分 object。
- 但 AP50 只有 `0.450000`，candidate pre% 只有 `4.5435%`，仍低于 `0.60 AP50 / 25% support` gate。

## 7. Phase 6 D4RT Geometry Diagnostic

命令:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v15_phase6/d4rt_sim3_residual_probe5 \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --max-anchors-per-window 2000 \
  --min-visibility 0.0 \
  --min-confidence 0.0 \
  > outputs/audit/v15_logs/v15_phase6_d4rt_sim3_residual.log 2>&1
```

输出:

- `Stream3D/outputs/audit/v15_phase6/d4rt_sim3_residual_probe5.{json,csv,md}`

结果:

```text
num_windows=5
num_ok_windows=5
num_failed_windows=0
sim3_anchor_count_mean=431.2
sim3_scale_mean=0.5601005412377911
sim3_scale_min=0.19451511179585823
sim3_scale_max=0.9788719831060647
sim3_residual_median_mean=0.46820795089862405
sim3_residual_p90_mean=0.8595805074130748
sim3_residual_p95_mean=1.0775159603094677
uv_in01_rate_mean=0.9858451843261719
```

结论: D4RT correspondence/UV 可用性高，但 Sim3 后 metric residual 仍大，不支持把 D4RT geometry 直接当 ScanNet RGB-D geometry replacement 主卖点。

## 8. Gate 判断

v15 probe5 最好结果:

```text
atom union oracle: A3/A4 target K=8, AP/AP50/AP25 = 0.297386 / 0.529412 / 0.764706, pre% ≈ 3.03%
region union oracle: R1 seed_voronoi K=8, AP/AP50/AP25 = 0.130556 / 0.450000 / 0.800000, pre% ≈ 4.54%
```

最低 broad-support gate 需要:

```text
oracle AP50 >= 0.60
oracle AP25 >= 0.78
pre% >= 25%
```

判断:

```text
Phase 1/2 broad-support gate: False
Phase 3 global object explanation solver: not started
Phase 4 set packing solver: not started
Phase 5 method materialization: not started
tune30/final: not started
```

## 9. 审计材料和复现入口

核心输出:

- `Stream3D/outputs/audit/v15_phase0/reportable_config_scan_v15_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase0/metric_integrity_v15_probe5.{json,md}`
- `Stream3D/outputs/audit/v15_phase1/*union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase2/*region_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase2/*union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v15_phase6/d4rt_sim3_residual_probe5.{json,csv,md}`

复现脚本:

- `Stream3D/scripts/reproduce_v15_phase0_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase1_union_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase2_regions_probe5.sh`
- `Stream3D/scripts/reproduce_v15_phase6_geometry_probe5.sh`
