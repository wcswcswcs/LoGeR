# Stream4D v18 Signed Boundary Graph 执行日志

日期: 2026-06-09
计划文档: `docs/stream4d_v18_signed_boundary_graph_plan_for_codex.md`
结果根目录: `Stream3D/outputs/audit`
GPU: `CUDA_VISIBLE_DEVICES=6,7`
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`

说明: 本日志只记录本次实际执行过的命令、落盘 artifact、代码修改与 blocker repair。没有运行的 phase 明确标为未运行；不补写不存在的数据。

## 0. 代码实现与自检

新增/修改的 v18 相关文件:

- `Stream3D/stream4d/signed_surfel_graph.py`: signed surfel graph 数据结构、source-grid/2D-kNN/cross-frame 边、edge metrics、Phase1 summary；blocker repair 中加入非 GT `precut_keep`，用于断开明显 mask-disagreement/source RGB/UV discontinuity 边，同时保留 raw edges 便于审计。
- `Stream3D/stream4d/signed_boundary_evidence.py`: E0-E7 non-GT edge evidence 变体与 summary。
- `Stream3D/stream4d/signed_graph_partition.py`: signed partition v1 与 export object dict；partition 默认只用 `precut_keep=True` 的边合并，可用 `--disable-graph-precut` 对照。
- `Stream3D/tools/build_v18_signed_surfel_graph.py`: Phase1 graph builder。
- `Stream3D/tools/diagnose_v18_edge_boundary_quality.py`: Phase2 GT-only oracle 与 Phase3 evidence diagnostic；修复点包括 per-scene KDTree 复用、oracle export 半径参数化、Phase2 gate 拆成 AP gate 与 graph coverage gate。
- `Stream3D/tools/build_v18_signed_boundary_evidence.py`
- `Stream3D/tools/export_v18_signed_graph_partition.py`
- `Stream3D/tools/diagnose_v18_partition_quality.py`
- `Stream3D/tools/summarize_v18_unified_eval_matrix.py`
- `Stream3D/tests/test_v18_signed_boundary_graph.py`
- `Stream3D/scripts/reproduce_v18_*.sh`

自检命令:

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  Stream3D/scripts/reproduce_v18_audit_probe5.sh
```

结果:

- `py_compile`: pass。
- `unittest tests.test_v18_signed_boundary_graph`: `Ran 3 tests in 0.201s OK`。
- 日志:
  - `Stream3D/outputs/audit/v18_logs/audit_py_compile.log`
  - `Stream3D/outputs/audit/v18_logs/audit_unittest_v18.log`

## 1. Phase0 Unified Eval Matrix

命令:

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  Stream3D/scripts/reproduce_v18_phase0_probe5.sh
```

输出:

- `Stream3D/outputs/audit/v18_phase0/unified_eval_matrix_probe5.json`
- `Stream3D/outputs/audit/v18_phase0/unified_eval_matrix_probe5.csv`
- `Stream3D/outputs/audit/v18_phase0/unified_eval_matrix_probe5.md`
- figures: `Stream3D/outputs/audit/v18_phase0/figures`
- log: `Stream3D/outputs/audit/v18_logs/phase0_unified_eval_matrix.log`

stdout 摘要: `num_rows=15`。

## 2. Phase1 Signed Surfel Graph

### 2.1 初始 graph

命令:

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  Stream3D/scripts/reproduce_v18_phase1_graph_probe5.sh
```

输出:

- `Stream3D/outputs/audit/v18_phase1/signed_surfel_graph_probe5.*`
- per-scene graph: `Stream3D/outputs/audit/v18_phase1/*/signed_surfel_graph.npz`

结果: fail。aggregate `largest_graph_component_ratio=1.0`，属于计划中的 graph 巨大粘连 blocker。

### 2.2 Blocker repair: 降 k / 去 cross-frame

命令:

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
OUTPUT_ROOT=outputs/audit/v18_phase1_repair_k4_nocross \
OUTPUT_PREFIX=outputs/audit/v18_phase1_repair_k4_nocross/signed_surfel_graph_probe5 \
KNN_K=4 CROSS_FRAME_NEIGHBORS=0 \
Stream3D/scripts/reproduce_v18_phase1_graph_probe5.sh
```

结果: fail。aggregate `largest_graph_component_ratio=1.0`，说明仅降低 kNN/cross-frame 不足以修复巨大粘连。

### 2.3 Blocker repair: 非 GT pre-cut

实现改动:

- 在 `SignedSurfelGraph` 中新增 `precut_keep`。
- `summarize_signed_surfel_graph` 同时记录 raw component 与 pre-cut 后 component。
- pre-cut 依据只使用预测/几何信息: target mask disagreement、source RGB discontinuity、UV discontinuity；不读 GT。

命令:

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
OUTPUT_ROOT=outputs/audit/v18_phase1_repair_precut_k8 \
OUTPUT_PREFIX=outputs/audit/v18_phase1_repair_precut_k8/signed_surfel_graph_probe5 \
KNN_K=8 CROSS_FRAME_NEIGHBORS=4 \
Stream3D/scripts/reproduce_v18_phase1_graph_probe5.sh
```

结果: aggregate pass without edge-count caveat。`raw_largest_graph_component_ratio=1.0`，pre-cut 后 `largest_graph_component_ratio=0.828466796875`。

继续补 kNN edge count:

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
OUTPUT_ROOT=outputs/audit/v18_phase1_repair_precut_k16_d015 \
OUTPUT_PREFIX=outputs/audit/v18_phase1_repair_precut_k16_d015/signed_surfel_graph_probe5 \
KNN_K=16 CROSS_FRAME_NEIGHBORS=4 PRECUT_MASK_DISAGREEMENT_RATIO=0.15 \
Stream3D/scripts/reproduce_v18_phase1_graph_probe5.sh
```

结果: aggregate pass。关键均值:

- `num_nodes=16384.0`
- `num_edges=271744.2`
- `track_length_visible_mean=13.284716796875`
- `uv_in01_rate=0.9858451843261719`
- `cycle_uv_error_p90=3.2737268686294554`
- `raw_largest_graph_component_ratio=1.0`
- `largest_graph_component_ratio=0.6813232421875`
- `precut_removed_edge_ratio=0.2144799646695895`

逐 scene residual:

- `scene0011_00`: track mean `9.64019775390625`，largest `0.95269775390625`，row gate false。
- `scene0030_00`: largest `0.25335693359375`，row gate false。
- `scene0081_01`: largest `0.9617919921875`，row gate false。
- `scene0050_00` / `scene0591_00`: row gate true。

按计划使用 probe5 aggregate gate 进入 Phase2，但复盘保留逐 scene residual。

## 3. Phase2 GT-only Edge Boundary Oracle

### 3.1 Main oracle: bank16 k16 d0.15

命令:

```bash
cd Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
CUDA_VISIBLE_DEVICES=6,7 \
BANK_ROOT=outputs/v14_measurement_bank_bank16_cropformer \
GRAPH_ROOT=outputs/audit/v18_phase1_repair_precut_k16_d015 \
OUTPUT_PREFIX=outputs/audit/v18_phase2_precut_k16_d015/edge_oracle_probe5 \
ORACLE_CONFIG=stream4d_v18_edge_oracle_precut_k16_d015_probe5 \
LOG_DIR=outputs/audit/v18_logs \
bash scripts/reproduce_v18_phase2_oracle_probe5.sh
```

输出:

- `Stream3D/outputs/audit/v18_phase2_precut_k16_d015/edge_oracle_probe5.*`
- metric: `Stream3D/data/evaluation/scannet/stream4d_v18_edge_oracle_precut_k16_d015_probe5_class_agnostic.txt`

结果:

- AP/AP50/AP25: `0.43486488661121014 / 0.581341911764706 / 0.6641366223908919`
- `node_gt_label_coverage=0.4501220703125`
- `edge_gt_label_coverage=0.4107389459487833`
- `phase2_min_gate=False`

失败原因: AP25 未达到 `0.70`；coverage 也低于计划 `0.70/0.60`。

### 3.2 Blocker repair: GT label nn radius

命令:

```bash
cd Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
CUDA_VISIBLE_DEVICES=6,7 \
BANK_ROOT=outputs/v14_measurement_bank_bank16_cropformer \
GRAPH_ROOT=outputs/audit/v18_phase1_repair_precut_k16_d015 \
OUTPUT_PREFIX=outputs/audit/v18_phase2_repair_nn008/edge_oracle_probe5 \
ORACLE_CONFIG=stream4d_v18_edge_oracle_precut_k16_d015_nn008_probe5 \
NN_RADIUS=0.08 LOG_DIR=outputs/audit/v18_logs \
bash scripts/reproduce_v18_phase2_oracle_probe5.sh
```

结果:

- AP/AP50/AP25: `0.4202611284412755 / 0.581341911764706 / 0.6641366223908919`
- `node_gt_label_coverage=0.467236328125`
- `edge_gt_label_coverage=0.4290656697147191`
- gate false。

结论: label nn 半径略增 coverage，但没有修复 AP25。

### 3.3 Blocker repair: oracle export radius

代码改动: `diagnose_v18_edge_boundary_quality.py` 新增 `--oracle-export-core-nn-radius` 等参数，复现脚本同步记录。

命令:

```bash
cd Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
CUDA_VISIBLE_DEVICES=6,7 \
BANK_ROOT=outputs/v14_measurement_bank_bank16_cropformer \
GRAPH_ROOT=outputs/audit/v18_phase1_repair_precut_k16_d015 \
OUTPUT_PREFIX=outputs/audit/v18_phase2_repair_export008/edge_oracle_probe5 \
ORACLE_CONFIG=stream4d_v18_edge_oracle_precut_k16_d015_export008_probe5 \
ORACLE_EXPORT_CORE_NN_RADIUS=0.08 LOG_DIR=outputs/audit/v18_logs \
bash scripts/reproduce_v18_phase2_oracle_probe5.sh
```

结果:

- AP/AP50/AP25: `0.43179497177658943 / 0.581341911764706 / 0.6641366223908919`
- `node_gt_label_coverage=0.4501220703125`
- `edge_gt_label_coverage=0.4107389459487833`
- gate false。

结论: export radius 不是主因。

### 3.4 Blocker repair: oracle min surfels

命令:

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m tools.diagnose_v18_edge_boundary_quality \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --graph-root outputs/audit/v18_phase1_repair_precut_k16_d015 \
  --seq-list splits/scannet_v6_probe5.txt \
  --mode oracle \
  --output-prefix outputs/audit/v18_phase2_repair_min5/edge_oracle_probe5 \
  --oracle-output-config stream4d_v18_edge_oracle_precut_k16_d015_min5_probe5 \
  --oracle-min-surfels 5
```

结果:

- AP/AP50/AP25: `0.43486488661121014 / 0.581341911764706 / 0.6641366223908919`
- gate false。

结论: 小组件过滤不是主因。

### 3.5 Blocker repair: grid48 density

D4RT grid48 carriers:

```bash
cd Stream3D
mkdir -p outputs/audit/v18_logs outputs/v18_d4rt_grid_surfel_field
CUDA_VISIBLE_DEVICES=6,7 \
MPLCONFIGDIR=/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache \
PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m tools.export_d4rt_grid_surfel_field_v8 \
  --d4rt-root /mnt/data/users/chengshun.wang/pjs/LoGeR/Open-d4rt \
  --d4rt-config /mnt/data/users/chengshun.wang/pjs/LoGeR/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt /mnt/data/users/chengshun.wang/pjs/LoGeR/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --device cuda:0 \
  --seq-list splits/scannet_v6_probe5.txt \
  --frame-stride 1 \
  --max-frames 16 \
  --window-size 16 \
  --window-stride 16 \
  --grid-size 48 \
  --grid-margin-ratio 0.02 \
  --visible-min-visibility 0.5 \
  --visible-min-confidence 0.5 \
  --query-chunk-size 2048 \
  --cycle-max-tracks 256 \
  --cycle-source-local 0 \
  --cycle-target-local -1 \
  --output-root outputs/v18_d4rt_grid_surfel_field \
  --run-name stream4d_v18_g1_grid48m002_probe5_16f_stride1_gpu67 \
  --allow-missing-masks \
  --continue-on-error \
  2>&1 | tee outputs/audit/v18_logs/phase2_repair_grid48_carriers.log
```

carriers 结果:

- `num_ok_windows=5`
- `num_surfel_tracks_mean=36864.0`
- `cycle_uv_error_p90_mean=3.284617304801941`

grid48 measurement bank:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m tools.build_v12_measurement_bank \
  --debug-root outputs/v18_d4rt_grid_surfel_field/stream4d_v18_g1_grid48m002_probe5_16f_stride1_gpu67 \
  --seq-list splits/scannet_v6_probe5.txt \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --output-root outputs/v18_measurement_bank_grid48_cropformer \
  --audit-prefix outputs/audit/v18_measurement_bank_grid48/measurement_bank_probe5 \
  2>&1 | tee outputs/audit/v18_logs/phase2_repair_grid48_measurement_bank.log
```

bank 结果:

- `num_surfels=36864.0`
- `track_length_visible_mean=13.341786024305557`
- `uv_in01_rate=0.9882273356119791`
- `unobserved_surfel_ratio=0.00754123263888889`

grid48 graph:

```bash
cd Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
CUDA_VISIBLE_DEVICES=6,7 \
BANK_ROOT=outputs/v18_measurement_bank_grid48_cropformer \
OUTPUT_ROOT=outputs/audit/v18_phase1_grid48_precut_k16_d015 \
OUTPUT_PREFIX=outputs/audit/v18_phase1_grid48_precut_k16_d015/signed_surfel_graph_probe5 \
KNN_K=16 CROSS_FRAME_NEIGHBORS=4 PRECUT_MASK_DISAGREEMENT_RATIO=0.15 \
LOG_DIR=outputs/audit/v18_logs \
bash scripts/reproduce_v18_phase1_graph_probe5.sh
```

graph 结果:

- `num_nodes=36864.0`
- `num_edges=625660.8`
- `largest_graph_component_ratio=0.8130967881944444`
- `precut_removed_edge_ratio=0.18820201366254616`

grid48 oracle:

```bash
cd Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
CUDA_VISIBLE_DEVICES=6,7 \
BANK_ROOT=outputs/v18_measurement_bank_grid48_cropformer \
GRAPH_ROOT=outputs/audit/v18_phase1_grid48_precut_k16_d015 \
OUTPUT_PREFIX=outputs/audit/v18_phase2_grid48_precut_k16_d015/edge_oracle_probe5 \
ORACLE_CONFIG=stream4d_v18_edge_oracle_grid48_precut_k16_d015_probe5 \
LOG_DIR=outputs/audit/v18_logs \
bash scripts/reproduce_v18_phase2_oracle_probe5.sh
```

结果:

- AP/AP50/AP25: `0.39600233234657134 / 0.504017531044558 / 0.6385135135135135`
- `node_gt_label_coverage=0.4540852864583334`
- `edge_gt_label_coverage=0.4255603199622774`
- gate false。

结论: grid48 增密未修复 oracle；AP25 反而下降。

### 3.6 Blocker repair 对照: bank16 k8 d0.25 oracle

命令:

```bash
cd Stream3D
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
CUDA_VISIBLE_DEVICES=6,7 \
BANK_ROOT=outputs/v14_measurement_bank_bank16_cropformer \
GRAPH_ROOT=outputs/audit/v18_phase1_repair_precut_k8 \
OUTPUT_PREFIX=outputs/audit/v18_phase2_repair_bank16_k8_d025/edge_oracle_probe5 \
ORACLE_CONFIG=stream4d_v18_edge_oracle_bank16_k8_d025_probe5 \
LOG_DIR=outputs/audit/v18_logs \
bash scripts/reproduce_v18_phase2_oracle_probe5.sh
```

结果:

- AP/AP50/AP25: `0.43486488661121014 / 0.581341911764706 / 0.6641366223908919`
- `node_gt_label_coverage=0.4501220703125`
- `edge_gt_label_coverage=0.4144560233498689`
- gate false。

## 4. Phase3/Phase4

未运行。原因: Phase2 GT-only edge oracle 未达到最小 gate；按计划不能启动 non-GT edge evidence/partition solver，也不能产出 reportable method table。

## 5. Manifest / Reportable Scan

oracle configs:

- `stream4d_v18_edge_oracle_precut_k16_d015_probe5`
- `stream4d_v18_edge_oracle_precut_k16_d015_nn008_probe5`
- `stream4d_v18_edge_oracle_precut_k16_d015_export008_probe5`
- `stream4d_v18_edge_oracle_precut_k16_d015_min5_probe5`
- `stream4d_v18_edge_oracle_grid48_precut_k16_d015_probe5`
- `stream4d_v18_edge_oracle_bank16_k8_d025_probe5`

manifest 检查:

- 全部 `is_method_result=False`
- 全部 `is_diagnostic_only=True`
- 全部 `uses_gt=True`, `uses_gt_for_prediction=True`
- 全部 `forbidden_for_method_table=True`

reportable scan:

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m tools.scan_reportable_configs \
  --root . \
  --configs stream4d_v18_edge_oracle_precut_k16_d015_probe5,stream4d_v18_edge_oracle_precut_k16_d015_nn008_probe5,stream4d_v18_edge_oracle_precut_k16_d015_export008_probe5,stream4d_v18_edge_oracle_precut_k16_d015_min5_probe5,stream4d_v18_edge_oracle_grid48_precut_k16_d015_probe5,stream4d_v18_edge_oracle_bank16_k8_d025_probe5 \
  --output outputs/audit/v18_final/oracle_reportable_scan.md
```

scan exit code: `5`，预期原因是所有 oracle diagnostic 都 `uses_gt_for_prediction=True`，因此不可报告。重要安全项:

- `num_uses_gt_and_method_result=0`
- `num_uses_gt_for_prediction_and_method_result=0`
- `num_gt_selected_output_and_method_result=0`
- `num_forbidden_for_method_table_and_method_result=0`
- `num_reportable_method_configs=0`

## 6. 最终停止点

停止在 Phase2。没有运行 Phase3 evidence variants、Phase4 partition、cross-support controls 或 final method package。后续若继续，应先修 surfel-to-mesh / object support materialization coverage，而不是绕过 oracle gate。
