# Stream4D v16 执行日志

日期: 2026-06-09  
计划文档: `docs/stream4d_v16_decisive_algorithmic_plan_for_codex.md`  
结果复盘: `docs/stream4d_v16_实验结果复盘.md`  
工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`  
Stream3D 目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D`  
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`  
GPU: `CUDA_VISIBLE_DEVICES=6,7`

## 执行边界

- v16 按计划先做三层 oracle / measurement / materialization 诊断，先判断是否有可解空间。
- 所有 `oracle` 输出都只允许作为 GT-read diagnostic upper bound，不允许进入 method table。
- 本轮没有把任何 `oracle`、`GT-selected`、`diagnostic-only` 输出改写为 method result。
- 因 official broad-support slot oracle 未过 v16 gate，Phase 3/4/5 solver/materialization method prototype 未启动。

## 代码和脚本修改

1. 修改 `Stream3D/tools/diagnose_v15_union_oracle.py`
   - 新增 `--algorithm-name` 参数，使 v16 输出 manifest/summary 可以写 `v16_union_oracle`、`v16_union_oracle_stress`、`v16_union_oracle_min50`。
   - 修复 `--k` 参数默认值: 显式传 `--k 2 --k 4 --k 8` 时不再和默认 `[1,2,4,8,16,32]` 混合。
   - 算法逻辑未改变，仍是 GT-read-only greedy union oracle diagnostic。

2. 新增 `Stream3D/tools/summarize_v16_decisive_diagnostics.py`
   - 只读取已有 JSON/eval 文件，不重算 AP。
   - 汇总 candidate oracle、slot oracle、owned-region materialization、Phase2 measurement bank、Phase6 D4RT geometry。
   - 输出 v16 stop gate 判定: `outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5.{json,csv,md}`。

3. 新增复现脚本
   - `Stream3D/scripts/reproduce_v16_phase0_audit_probe5.sh`
   - `Stream3D/scripts/reproduce_v16_phase1_decisive_oracle_probe5.sh`
   - `Stream3D/scripts/reproduce_v16_phase2_measurement_bank_probe5.sh`
   - `Stream3D/scripts/reproduce_v16_phase6_geometry_probe5.sh`

## Phase 1: Three-Layer Oracle

### Official broad/support slot oracle

命令: `Stream3D/scripts/reproduce_v16_phase1_decisive_oracle_probe5.sh`

核心实际命令如下。

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
export CUDA_VISIBLE_DEVICES=6,7
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

$PY -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_mask_unsup_probe5 \
  --pre-points-config stream4d_v13_c_mask_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_mask_union_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_mask_union_oracle_probe5 \
  --algorithm-name v16_union_oracle \
  --k 2 --k 4 --k 8 \
  --max-candidates-per-gt 512

$PY -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_hybrid_unsup_probe5 \
  --pre-points-config stream4d_v13_c_hybrid_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_hybrid_union_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_hybrid_union_oracle_probe5 \
  --algorithm-name v16_union_oracle \
  --k 2 --k 4 --k 8 \
  --max-candidates-per-gt 512

$PY -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_regionlet_unsup_probe5 \
  --pre-points-config stream4d_v13_c_regionlet_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_regionlet_union_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_regionlet_union_oracle_probe5 \
  --algorithm-name v16_union_oracle \
  --k 2 --k 4 --k 8 \
  --max-candidates-per-gt 512
```

输出:

- `Stream3D/outputs/audit/v16_phase1/c_mask_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase1/c_hybrid_union_oracle_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase1/c_regionlet_union_oracle_probe5.{json,csv,md}`
- `Stream3D/data/evaluation/scannet/stream4d_v16_oracle_*_class_agnostic.txt`
- logs: `Stream3D/outputs/audit/v16_logs/v16_phase1_*_union_oracle.log`

结果摘要:

| primitive | K | AP | AP50 | AP25 | support% | selected union% | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| C_mask | 8 | 0.276531 | 0.528889 | 0.687852 | 60.8842 | 60.4630 | False |
| C_hybrid | 8 | 0.366996 | 0.634907 | 0.764547 | 52.8088 | 50.0378 | False |
| C_regionlet | 8 | 0.595554 | 0.877258 | 0.979039 | 18.5455 | 16.0752 | False |

### Blocker repair: C_hybrid slot budget stress

动机: official C_hybrid K8 的 AP50 已过 `0.60`，但 AP25 `0.764547` 未过 v16 `0.80` gate。按计划的 slot candidate union 方向继续尝试，不直接放弃。

命令:

```bash
$PY -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_hybrid_unsup_probe5 \
  --pre-points-config stream4d_v13_c_hybrid_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_hybrid_union_stress_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_hybrid_union_stress_probe5 \
  --algorithm-name v16_union_oracle_stress \
  --k 16 --k 32 \
  --max-candidates-per-gt 768
```

结果:

| variant | K | AP | AP50 | AP25 | support% | selected union% | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| C_hybrid stress | 16 | 0.375373 | 0.643887 | 0.769007 | 52.8088 | 51.6155 | False |
| C_hybrid stress | 32 | 0.377319 | 0.643887 | 0.769007 | 52.8088 | 52.2386 | False |

结论: 扩大 slot budget 只把 AP25 从 `0.764547` 提到 `0.769007`，仍低于 `0.80`。

### Blocker repair: C_hybrid smaller min region

动机: 检查 `min_region_size=100` 是否过滤了有用小 measurement。

命令:

```bash
$PY -m tools.diagnose_v15_union_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --pred-config stream4d_v13_c_hybrid_unsup_probe5 \
  --pre-points-config stream4d_v13_c_hybrid_unsup_probe5 \
  --output-config-prefix stream4d_v16_oracle_c_hybrid_union_min50_probe5 \
  --summary-prefix outputs/audit/v16_phase1/c_hybrid_union_min50_probe5 \
  --algorithm-name v16_union_oracle_min50 \
  --k 2 --k 4 --k 8 \
  --min-region-size 50 \
  --max-candidates-per-gt 768
```

结果: K2/K4/K8 数值与默认 min-region-size 100 相同，best 仍为 `0.366996 / 0.634907 / 0.764547`。说明该瓶颈不是 50-100 点小候选被过滤导致。

### Decisive summary

命令:

```bash
$PY -m tools.summarize_v16_decisive_diagnostics \
  --root . \
  --output-prefix outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5
```

输出:

- `Stream3D/outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5.json`
- `Stream3D/outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5.csv`
- `Stream3D/outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5.md`

Gate summary:

```text
official_broad_slot_gate_pass=False
stress_broad_slot_gate_pass=False
measurement_bank_gate_pass=False
stop_before_solver=True
stop_reason=official broad-support slot oracle misses AP25>=0.80; C_hybrid stress K16/K32 also misses AP25>=0.80
```

## Phase 2: Measurement Bank Diagnostic

命令: `Stream3D/scripts/reproduce_v16_phase2_measurement_bank_probe5.sh`

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

$PY -m tools.diagnose_v12_measurement_bank \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v16_phase2/measurement_bank_bank16_probe5
```

输出:

- `Stream3D/outputs/audit/v16_phase2/measurement_bank_bank16_probe5.{json,csv,md}`
- log: `Stream3D/outputs/audit/v16_logs/v16_phase2_measurement_bank_bank16.log`

关键结果:

| metric | value |
|---|---:|
| num_mask_frames_available | 16.0 |
| num_mask_frames_missing | 0.0 |
| uv_in01_rate | 0.985845 |
| cycle_uv_error_p90 | 3.273727 |
| self_uv_error_p90 | 1.570825 |
| unobserved_surfel_ratio | 0.007849 |
| mean_positive_observations_per_surfel | 1.632495 |
| surfel_positive_observation_rate | 0.992151 |

结论: geometry/mask-frame density 通过关键稳定性条件，但 `mean_positive_observations_per_surfel=1.632495` 未达到 v16 计划中的 `>=2.5`。

## Phase 3/4/5: Not Started By Gate

原因: Phase1 official broad-support slot oracle 未过 v16 gate。最强 official broad row 是 C_hybrid K8:

```text
AP/AP50/AP25 = 0.366996 / 0.634907 / 0.764547
support% = 52.8088
```

AP50 已过 `0.60`，但 AP25 未过 `0.80`。额外 K16/K32 stress repair 仍未过:

```text
AP/AP50/AP25 = 0.375373 / 0.643887 / 0.769007
```

因此按计划停止，不启动 object explanation solver、posterior materialization method、四行 method evaluation。

## Phase 6: D4RT Geometry Diagnostic

命令: `Stream3D/scripts/reproduce_v16_phase6_geometry_probe5.sh`

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
export CUDA_VISIBLE_DEVICES=6,7
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

$PY -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v16_phase6/d4rt_sim3_residual_probe5 \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --max-anchors-per-window 2000 \
  --min-visibility 0.0 \
  --min-confidence 0.0
```

输出:

- `Stream3D/outputs/audit/v16_phase6/d4rt_sim3_residual_probe5.{json,csv,md}`
- log: `Stream3D/outputs/audit/v16_logs/v16_phase6_d4rt_sim3_residual.log`

关键结果:

| metric | value |
|---|---:|
| num_ok_windows | 5 |
| num_failed_windows | 0 |
| sim3_anchor_count_mean | 431.2 |
| sim3_scale_mean/min/max | 0.560101 / 0.194515 / 0.978872 |
| sim3_residual_median_mean | 0.468208 |
| sim3_residual_p90_mean | 0.859581 |
| sim3_residual_p95_mean | 1.077516 |
| uv_in01_rate_mean | 0.985845 |

## Phase 0 Final Audit

命令: `Stream3D/scripts/reproduce_v16_phase0_audit_probe5.sh`

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

$PY -m py_compile evaluation/*.py stream4d/*.py tools/*.py tests/*.py
$PY -m unittest discover tests

find data/prediction -maxdepth 1 -type d -name 'stream4d_v16*_class_agnostic' -printf '%f\n' \
  | sed 's/_class_agnostic$//' \
  | sort \
  > outputs/audit/v16_phase0_configs.txt

CONFIGS="$(paste -sd, outputs/audit/v16_phase0_configs.txt)"

$PY -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v16_phase0/reportable_config_scan_v16_probe5.md \
  --require-manifest \
  --require-eval-policy

$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v16_phase0/metric_integrity_v16_probe5.md \
  --require-manifest
```

结果:

```text
py_compile: pass
unittest: Ran 30 tests in 1.474s OK
reportable scan:
  num_configs=14
  num_oracle_configs=14
  num_reportable_method_configs=0
  num_diagnostic_only_configs=14
  num_suspicious_configs=0
  num_uses_gt_for_prediction=0
  num_gt_selected_output_and_method_result=0
  num_forbidden_for_method_table_and_method_result=0
  num_alignment_used_for_prediction=0
metric integrity:
  phase0_pass=True
  gt_files_read_by_rescore=False
```

输出:

- `Stream3D/outputs/audit/v16_phase0/reportable_config_scan_v16_probe5.{json,csv,md}`
- `Stream3D/outputs/audit/v16_phase0/metric_integrity_v16_probe5.{json,md}`
- `Stream3D/outputs/audit/v16_phase0_configs.txt`

## 审计包

审计包生成在 repo 根目录:

- `stream4d_v16_code_review_packet.zip`
- `stream4d_v16_code_review_packet.sha256`
- `stream4d_v16_filelist.txt`
- `stream4d_v16_ziptest.log`
- `stream4d_v16_git_status.txt`
- `stream4d_v16_git_diff.patch`

