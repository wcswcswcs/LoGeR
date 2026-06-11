# Stream4D v19 4D TubeCover / Materialization 执行日志

日期: 2026-06-09
计划文档: `docs/stream4d_v19_4d_tubecover_materialization_plan_for_codex.md`
结果根目录: `Stream3D/outputs/audit`
GPU: `CUDA_VISIBLE_DEVICES=6,7`
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`

说明: 本日志只记录本次实际执行过的命令、代码修改、落盘 artifact 与 blocker repair。未运行 phase 明确标为未运行；不补写不存在的数据。

## 0. 代码实现与自检

新增/修改的 v19 文件:

- `Stream3D/tools/diagnose_v19_materialization.py`: v19 Phase1/Phase2A 诊断主工具。实现 surfel->mesh RGB-D hit union、Oracle A/B/C、M0/M1/M2/M3 materialization variants、per-GT failure decomposition、diagnostic-only manifest 与 evaluator 调用。
- `Stream3D/tools/summarize_v19_phase0.py`: 汇总 v18 baseline / graph / oracle artifacts，生成 v19 Phase0 复现表。
- `Stream3D/scripts/reproduce_v19_phase0_probe5.sh`
- `Stream3D/scripts/reproduce_v19_materialization_variant_probe5.sh`
- `Stream3D/scripts/reproduce_v19_phase2a_probe5.sh`
- `Stream3D/scripts/reproduce_v19_audit_probe5.sh`
- `Stream3D/tests/test_v19_materialization.py`

关键实现审计点:

- GT-only oracle artifacts 按 v19 计划标记为 `is_method_result=false`、`is_diagnostic_only=true`、`forbidden_for_method_table=true`、`uses_gt_for_prediction=false`、`uses_gt_for_diagnostic=true`。
- `gt_selected_output=true` 保留在 manifest 中，说明这些 oracle prediction 使用 GT diagnostic 选择，禁止进入 method table。
- M2/M3 初版发现 `mesh_vertex_coverage_ratio` 仍引用 raw UV hit coverage；已修正为 variant materialized coverage，并额外保留 `raw_mesh_hit_vertex_coverage_ratio`。

自检命令:

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m py_compile tools/diagnose_v19_materialization.py tools/summarize_v19_phase0.py tests/test_v19_materialization.py

/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m unittest tests.test_v19_materialization
```

结果:

- `py_compile`: pass。
- `tests.test_v19_materialization`: `Ran 3 tests ... OK`。

## 1. Smoke

目的: 单 scene 验证 open3d/ScanNet 路径、surfel hit、Oracle export 与 failure decomposition 能落盘。该 smoke 不作为最终结论。

命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m tools.diagnose_v19_materialization \
  --variant M1 \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --graph-root outputs/audit/v18_phase1_repair_precut_k16_d015 \
  --seq-list splits/scannet_scene0050.txt \
  --output-prefix outputs/audit/v19_smoke_m1/materialization_probe1 \
  --output-config-prefix stream4d_v19_smoke \
  --skip-eval
```

输出:

- `Stream3D/outputs/audit/v19_smoke_m1/materialization_probe1.*`

## 2. Phase0: v18 复现与 v19 起点汇总

命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
bash scripts/reproduce_v19_phase0_probe5.sh
```

输出:

- `Stream3D/outputs/audit/v18_phase0/unified_eval_matrix_probe5.*`
- `Stream3D/outputs/audit/v19_phase0/phase0_reproduction_probe5.*`
- logs:
  - `Stream3D/outputs/audit/v19_logs/phase0_v18_unified_eval_matrix.log`
  - `Stream3D/outputs/audit/v19_logs/phase0_v19_reproduction_summary.log`

结果:

- `num_rows=12`
- `num_missing=0`
- `all_reference_baselines_reproduced=True`

## 3. Phase1/Phase2A: materialization M0-M3

主命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
bash scripts/reproduce_v19_phase2a_probe5.sh
```

输出:

- `Stream3D/outputs/audit/v19_phase2a_M0/materialization_probe5.*`
- `Stream3D/outputs/audit/v19_phase2a_M1/materialization_probe5.*`
- `Stream3D/outputs/audit/v19_phase2a_M2/materialization_probe5.*`
- `Stream3D/outputs/audit/v19_phase2a_M3/materialization_probe5.*`
- per-scene hit/decomposition:
  - `Stream3D/outputs/audit/v19_phase2a_<VARIANT>/<scene>/surfel_mesh_hits.npz`
  - `Stream3D/outputs/audit/v19_phase2a_<VARIANT>/<scene>/failure_decomposition.json`
- evaluator metric files:
  - `Stream3D/data/evaluation/scannet/stream4d_v19_m{0,1,2,3}_oracle_{a,b,c}_probe5_class_agnostic.txt`

主结果摘要:

| variant | node cov | edge cov | materialized mesh cov | covered GT inst ratio | Oracle A AP/AP50/AP25 | Oracle B AP/AP50/AP25 | Oracle C AP/AP50/AP25 | gate |
|---|---:|---:|---:|---:|---|---|---|---|
| M0 v18 posterior | 0.450122 | 0.410739 | 0.019828 | 0.217783 | 0.531747/0.691765/0.735294 | 0.434865/0.581342/0.664137 | 1.000000/1.000000/1.000000 | False |
| M1 multi-frame hit union | 0.450122 | 0.410739 | 0.019867 | 0.276430 | 0.491969/0.696217/0.735294 | 0.444403/0.647368/0.723408 | 1.000000/1.000000/1.000000 | False |
| M2 dilation r0.03 | 0.450122 | 0.410739 | 0.038123 | 0.283573 | 0.346451/0.601698/0.711111 | 0.288492/0.512341/0.656624 | 1.000000/1.000000/1.000000 | False |
| M3 anchor mask-region fill | 0.450122 | 0.410739 | 0.032380 | 0.274550 | 0.328369/0.590742/0.757120 | 0.284319/0.503402/0.698622 | 1.000000/1.000000/1.000000 | False |

## 4. Blocker repair attempts

### 4.1 manifest/metric field repair

现象:

- 初版 diagnostic manifest 沿用 v18 语义，写成 `uses_gt_for_prediction=true`。
- 初版 M2/M3 `mesh_vertex_coverage_ratio` 仍报 raw hit coverage，而不是 variant materialized coverage。

修复:

- `diagnose_v19_materialization.py` 中把 v19 oracle manifest 改为 `uses_gt_for_prediction=false`、`uses_gt_for_diagnostic=true`，并保留 `gt_selected_output=true` 与 `forbidden_for_method_table=true`。
- scene/aggregate coverage 字段改为:
  - `mesh_vertex_coverage_ratio`: variant materialized GT-labeled mesh vertex coverage。
  - `raw_mesh_hit_vertex_coverage_ratio`: raw UV->depth->mesh hit coverage。

重写 artifacts:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
SKIP_EVAL=1 \
bash scripts/reproduce_v19_phase2a_probe5.sh
```

说明: `SKIP_EVAL=1` 只重写 JSON/manifest/object artifacts，并从已存在 metric file 读取 AP；不重新计算 AP。

### 4.2 NN radius repair

命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m tools.diagnose_v19_materialization \
  --variant M1 \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --graph-root outputs/audit/v18_phase1_repair_precut_k16_d015 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v19_phase2a_repair_m1_nn008/materialization_probe5 \
  --output-config-prefix stream4d_v19_nn008 \
  --nn-radius 0.08 \
  --max-frames 16
```

输出:

- `Stream3D/outputs/audit/v19_phase2a_repair_m1_nn008/materialization_probe5.*`
- log: `Stream3D/outputs/audit/v19_logs/phase2a_repair_m1_nn008.log`

结果: node/edge `0.467236/0.429066`，materialized mesh coverage `0.020490`，Oracle B `0.412439/0.683826/0.718042`，gate false。

### 4.3 grid48 density repair

命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m tools.diagnose_v19_materialization \
  --variant M1 \
  --bank-root outputs/v18_measurement_bank_grid48_cropformer \
  --graph-root outputs/audit/v18_phase1_grid48_precut_k16_d015 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v19_phase2a_repair_grid48_m1/materialization_probe5 \
  --output-config-prefix stream4d_v19_grid48 \
  --nn-radius 0.05 \
  --max-frames 16
```

输出:

- `Stream3D/outputs/audit/v19_phase2a_repair_grid48_m1/materialization_probe5.*`
- log: `Stream3D/outputs/audit/v19_logs/phase2a_repair_grid48_m1.log`

结果: node/edge `0.454085/0.425560`，materialized mesh coverage `0.026224`，Oracle B `0.375282/0.502732/0.690234`，gate false。

### 4.4 aggressive dilation repair

命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
  -m tools.diagnose_v19_materialization \
  --variant M2 \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --graph-root outputs/audit/v18_phase1_repair_precut_k16_d015 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v19_phase2a_repair_m2_r008/materialization_probe5 \
  --output-config-prefix stream4d_v19_m2r008 \
  --nn-radius 0.05 \
  --max-frames 16 \
  --m2-dilation-radius 0.08
```

输出:

- `Stream3D/outputs/audit/v19_phase2a_repair_m2_r008/materialization_probe5.*`
- log: `Stream3D/outputs/audit/v19_logs/phase2a_repair_m2_r008.log`

结果: materialized mesh coverage `0.060089`，covered GT inst ratio `0.290716`，但 Oracle B 降到 `0.141086/0.371698/0.593569`，gate false。

## 5. Audit

命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 \
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
bash scripts/reproduce_v19_audit_probe5.sh
```

输出:

- `Stream3D/outputs/audit/v19_logs/audit_py_compile.log`
- `Stream3D/outputs/audit/v19_logs/audit_unittest_discover.log`
- `Stream3D/outputs/audit/v19_logs/audit_scan_reportable_configs.log`
- `Stream3D/outputs/audit/v19_logs/audit_metric_integrity.log`
- `Stream3D/outputs/audit/v19_final/oracle_reportable_scan.*`
- `Stream3D/outputs/audit/v19_final/metric_integrity.*`

结果:

- `py_compile`: pass。
- `unittest discover tests`: `Ran 36 tests in 1.454s OK`。
- reportable scan: 21 configs, 21 diagnostic-only, 0 method configs, 0 suspicious configs, 0 missing manifest, 0 missing eval policy。
- metric integrity: `phase0_pass=True`。

## 6. 未运行项目

- Phase2B TubeCover: 未运行。
- Phase3 non-GT signed edge evidence: 未运行。
- Phase4 Tube Manifold Partition: 未运行。
- Phase5 memory / Phase7 tune30/final/dynamic: 未运行。

原因: Phase2A coverage/materialization gate 未通过。按 stop rules，不能启动 non-GT evidence、partition solver 或 method table。

## 7. Code Review Packet

生成命令摘要:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
PACK=stream4d_v19_code_review_packet_20260609
# 生成 ${PACK}_filelist.txt / ${PACK}_git_status.txt / ${PACK}_git_diff.patch
zip -q "${PACK}.zip" -@ < "${PACK}_filelist.txt"
sha256sum "${PACK}.zip" > "${PACK}.sha256"
unzip -t "${PACK}.zip" > "${PACK}_ziptest.log"
```

审计包:

- `stream4d_v19_code_review_packet_20260609.zip`
- `stream4d_v19_code_review_packet_20260609.sha256`
- `stream4d_v19_code_review_packet_20260609_filelist.txt`
- `stream4d_v19_code_review_packet_20260609_git_diff.patch`
- `stream4d_v19_code_review_packet_20260609_git_status.txt`
- `stream4d_v19_code_review_packet_20260609_ziptest.log`

sha256: 见 `stream4d_v19_code_review_packet_20260609.sha256`。
