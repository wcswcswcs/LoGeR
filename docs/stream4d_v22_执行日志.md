# Stream4D v22 执行日志

日期: 2026-06-10
计划文档: `docs/stream4d_v22_rectification_plan.md`
结果根目录: `Stream3D/outputs/audit`
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`

本日志只记录本轮实际执行过的命令、代码修改、artifact 和真实输出摘要。不把未运行内容写成已完成。

## 0. 初始环境与约束

- 工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR`
- Stream3D 命令工作目录: `/mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D`
- 开始时间: `2026-06-10 23:56:10 +08 +0800`
- Python: `Python 3.11.15`
- GPU: 0-7 可用，均为 `NVIDIA RTX A5000`，`23028 MiB`，初始 `memory.used=1 MiB`，`utilization.gpu=0%`。
- 初始 `git status --short` 显示大量 v21.3 相关未提交文件和若干既有修改；本轮不回滚既有改动，只追加或修改 v22 需要的文件。

初始检查命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
git status --short
sed -n '1,260p' docs/stream4d_v22_rectification_plan.md
sed -n '261,620p' docs/stream4d_v22_rectification_plan.md
sed -n '621,980p' docs/stream4d_v22_rectification_plan.md
sed -n '981,1320p' docs/stream4d_v22_rectification_plan.md
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python --version
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

读取结论:

- v22 计划要求先完成 Phase A 代码包/测试可信度，再完成 Phase B self-stitch 指标、matching、persistent tube id 和 synthetic tests。
- Stop Rules 明确要求: 如果 Phase A 或 Phase B 不可信，不能继续宣称 provider/method 结论。
- Phase C 必须新增 D4RT-on-ScanNet direct reconstruction benchmark，直接报告 depth / point cloud / pose / track / stitching / coverage，而不是只用 Stream3D provider AP 间接判断。

## 1. 待执行阶段

- Phase A: 代码包与核心依赖审计。
- Phase B: self-stitch 指标、matching、persistent tube id 与 synthetic tests 整改。
- Phase C: direct D4RT reconstruction benchmark。
- Phase D/E: 只有在 A/B/C gate 可信后才继续 provider replacement 和 occupancy correction。

## 2. Phase A 代码包与核心依赖审计

### 2.1 核心文件存在性

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
for f in Stream3D/tools/d4rt_geometry_diagnostic.py Stream3D/tools/materialize_d4rt_aligned_geometry_for_stream3d.py Stream3D/stream4d_native/sim3.py Stream3D/stream4d_native/chunk_alignment.py Stream3D/geometry_provider/d4rt_carrier_provider.py Stream3D/tools/native_geometry_diagnostics.py Stream3D/tools/export_v21_3_occupancy_carrier_cache.py Stream3D/tools/run_v21_3_stream3d_provider_replacement.py Stream3D/tests/test_v21_3_geometry_provider.py Stream3D/tests/test_native_chunking_and_sim3.py; do test -f "$f" || echo "MISSING $f"; done
```

结果: 无输出，说明计划列出的 10 个核心文件均存在。

### 2.2 helper 归属修复

发现:

```bash
rg -n "fit_sim3|_backproject_xy_world|persistent_tube|inlier_ratio|match_source|overlap_policy|best_confidence|lowest_residual|newest_window|all_window" Stream3D/geometry_provider Stream3D/stream4d_native Stream3D/tools Stream3D/tests
```

关键发现:

- `Stream3D/geometry_provider/d4rt_carrier_provider.py` 从 `tools.d4rt_geometry_diagnostic` import `fit_sim3_umeyama`。
- `Stream3D/geometry_provider/d4rt_carrier_provider.py` 从 `tools.materialize_d4rt_aligned_geometry_for_stream3d` import `_backproject_xy_world` 和 `_fit_transform`。

代码修改:

- `Stream3D/stream4d_native/sim3.py`
  - 新增 public `fit_sim3_umeyama`，保留 v21.x 工具脚本需要的 dict key: `scale`、`rotation`、`translation`、`residual`、`anchor_count`。
  - 修复 residual 计算为 float64，避免通过 `apply_sim3_to_xyz` 转成 float32 后损失精度。
- `Stream3D/geometry_provider/common.py`
  - 新增 `backproject_xy_world`。
  - 新增 `fit_transform` robust trim helper。
- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - 改为从 `stream4d_native.sim3` 和 `geometry_provider.common` import，不再依赖 `tools.*`。
- `Stream3D/tools/d4rt_geometry_diagnostic.py`
  - 改为复用 `stream4d_native.sim3.fit_sim3_umeyama`。
- `Stream3D/tools/materialize_d4rt_aligned_geometry_for_stream3d.py`
  - 保留旧内部函数名作为兼容 wrapper，但实现调用 `geometry_provider.common`。

复查命令:

```bash
rg -n "from tools\.(d4rt_geometry_diagnostic|materialize_d4rt_aligned_geometry_for_stream3d)" Stream3D/geometry_provider Stream3D/stream4d_native
```

结果: 无输出。`rg` exit code 为 1，表示没有匹配项。

### 2.3 编译与测试

编译命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile evaluation/*.py stream4d/*.py stream4d_native/*.py geometry_provider/*.py tools/*.py tests/*.py
```

结果: pass，无输出。

指定测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v21_3_geometry_provider
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_chunking_and_sim3
```

结果:

```text
Ran 5 tests in 0.016s
OK

Ran 9 tests in 0.021s
OK
```

首次全量测试:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 59 tests in 1.679s
FAILED (failures=1)
```

失败点:

```text
test_d4rt_geometry_sim3_fit_recovers_known_transform
AssertionError: 2.2324505258646623e-07 not less than 1e-09
```

原因: 新公共 Sim3 helper 初版用 `apply_sim3_to_xyz` 计算 residual，而该函数为了 batch 输出会返回 float32，导致高精度单测失败。

修复后重跑:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_stream4d_protocol_fixes.Stream4DProtocolFixTests.test_d4rt_geometry_sim3_fit_recovers_known_transform
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile stream4d_native/sim3.py geometry_provider/common.py geometry_provider/d4rt_carrier_provider.py tools/d4rt_geometry_diagnostic.py tools/materialize_d4rt_aligned_geometry_for_stream3d.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

最终结果:

```text
Ran 1 test in 0.001s
OK

Ran 59 tests in 1.438s
OK
```

Phase A 指标:

| metric | value |
|---|---:|
| missing_core_file_count | 0 |
| provider_imports_resolved | true |
| sim3_helper_single_source_of_truth | true |
| full_unittest_count | 59 |
| full_unittest_pass | true |
| clean_packet_unittest_pass | true |

## 3. Phase B Self-Stitch 实现整改

### 3.1 代码修改

新增/修改:

- `Stream3D/stream4d_native/self_stitch.py`
  - 新增 `residual_diagnostics`，报告:
    - `inlier_ratio_abs005`
    - `inlier_ratio_abs010`
    - `inlier_ratio_rel001`
    - `inlier_ratio_rel002`
    - `inlier_ratio_mad`
    - `residual_median`
    - `residual_p90`
    - `residual_p95`
    - `residual_mad`
  - 新增 `match_overlap_carriers`，统一 provider / diagnostic 的 overlap matching:
    - primary: `persistent_tube_id`，如不存在则用 stable `carrier_id`
    - secondary: same `src_frame_global + src_xy`
    - fallback: mutual UV nearest-neighbor + cycle consistency
  - 记录 `match_source_stable_id_count`、`match_source_same_source_pixel_count`、`match_source_mutual_uv_count`、`match_source_rejected_count`、`stable_id_match_ratio`、`mutual_uv_match_ratio`、`cycle_consistency_pass_ratio`。
  - `appearance_consistency_pass_ratio` 字段存在，但当前 cached carrier/provider 没有 RGB patch 输入，记录为 `None` / `appearance_consistency_available=false`，不伪造通过。
- `Stream3D/stream4d_native/sim3.py`
  - `estimate_overlap_sim3` 不再把 p90 quantile residual 当 inlier ratio。
  - backward-compatible `inlier_ratio` 现在等于 `inlier_ratio_abs010`。
- `Stream3D/geometry_provider/d4rt_carrier_provider.py`
  - self-stitch 改为调用 `match_overlap_carriers`。
  - provider diagnostics 新增 true inlier ratio 与 matching source 汇总。
  - `_WindowData` 新增 `persistent_tube_id`、`src_frame_global`、`src_xy`。
  - overlap policy 扩展为 `all_window_union`、`best_confidence`、`lowest_residual`、`newest_window`，旧 `all` 作为 alias。
  - frame diagnostics 新增 `selected_source_windows`、`duplicate_window_hit_rate`。
- `Stream3D/stream4d/carrier_store.py`
  - `CarrierBatch` 新增可选字段:
    - `persistent_tube_id`
    - `parent_tube_id`
    - `warmstart_source_chunk`
    - `warmstart_source_frame`
    - `is_warmstarted`
  - `save_npz` 会在字段存在时写入 cache。
- `Stream3D/tools/export_v21_3_occupancy_carrier_cache.py`
  - D5 warmstart 新增 persistent identity propagation。
  - 新增 overlap UV 一对一匹配的 identity retention:
    - `persistent_tube_retention_count`
    - `persistent_tube_retention_rate`
    - `warmstart_tube_acceptance_rate`
  - 新增 CLI: `--persistent-uv-radius`。
- `Stream3D/tools/native_geometry_diagnostics.py`
  - Phase C diagnostic 改为复用 `match_overlap_carriers`。
  - Phase C rows 新增 stable/source-pixel/mutual-UV match counts 和 true inlier ratios。
- `Stream3D/tools/run_v21_3_stream3d_provider_replacement.py`
  - `--overlap-policy` choices 扩展到 `all_window_union`、`best_confidence`、`lowest_residual`、`newest_window`。

新增测试:

- `Stream3D/tests/test_native_chunking_and_sim3.py`
  - `test_residual_diagnostics_are_not_quantile_defined`
  - `test_overlap_matching_prefers_persistent_tube_id`
  - `test_estimate_overlap_sim3_recovers_known_transform` 增加 true inlier ratio 字段断言。
- `Stream3D/tests/test_v21_3_geometry_provider.py`
  - `test_self_stitch_three_window_nonidentity_sim3_chain`
  - overlap policy 断言: `best_confidence` vs `all_window_union`。
- `Stream3D/tests/test_native_occupancy_and_builder.py`
  - `test_d5_identity_assignment_writes_persistent_fields`

### 3.2 Phase B 测试

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_chunking_and_sim3
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v21_3_geometry_provider
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_native_occupancy_and_builder
```

结果:

```text
Ran 11 tests in 0.031s
OK

Ran 6 tests in 0.028s
OK

Ran 10 tests in 0.009s
OK
```

全量命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile evaluation/*.py stream4d/*.py stream4d_native/*.py geometry_provider/*.py tools/*.py tests/*.py
```

结果:

```text
Ran 63 tests in 1.536s
OK
```

`py_compile` pass，无输出。

### 3.3 Cached self-stitch diagnostic rerun

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.native_geometry_diagnostics \
  --output outputs/audit/v22_phaseB/native_geometry_diagnostics_v22_phaseB.md
```

stdout 摘要:

```json
{
  "phase_b": {
    "uv_in01_rate": 0.6336186710030136,
    "visible_track_length_mean": 0.1969606629696502,
    "mask_interior_coverage_mean": 0.9910409591491313,
    "mask_boundary_coverage_mean": 0.6023544556601523
  },
  "phase_c": {
    "num_pairs": 6,
    "alignment_fail_count": 0,
    "overlap_frame_count_mean": 16.0,
    "overlap_anchor_count_mean": 15891.666666666666,
    "self_sim3_inlier_ratio_abs005_mean": 0.3955584891741292,
    "self_sim3_inlier_ratio_abs010_mean": 0.8846424683950961,
    "self_sim3_inlier_ratio_rel001_mean": 0.27543888522582016,
    "self_sim3_inlier_ratio_rel002_mean": 0.7904198920960077,
    "self_sim3_inlier_ratio_mad_mean": 0.9681738394181285,
    "self_sim3_residual_p90_mean": 0.10195049235911836,
    "self_sim3_scale_std": 0.04806286727574159,
    "accumulated_scale_drift": 0.2659042892460197
  }
}
```

落盘 artifact:

- `Stream3D/outputs/audit/v22_phaseB/native_geometry_diagnostics_v22_phaseB.md`
- `Stream3D/outputs/audit/v22_phaseB/native_geometry_diagnostics_v22_phaseB.json`
- `Stream3D/outputs/audit/v22_phaseB/native_geometry_diagnostics_v22_phaseB_phase_b_rows.csv`
- `Stream3D/outputs/audit/v22_phaseB/native_geometry_diagnostics_v22_phaseB_phase_c_rows.csv`
- `Stream3D/outputs/audit/v22_phaseB/native_geometry_diagnostics_v22_phaseB_phase_c_scale_normalized_rows.csv`

首个 pair row 核查:

```text
match_source_stable_id_count = 21168
match_source_same_source_pixel_count = 0
match_source_mutual_uv_count = 79
stable_id_match_ratio = 0.9962818280227796
mutual_uv_match_ratio = 0.0037181719772203135
cycle_consistency_pass_ratio = 0.9186046511627907
appearance_consistency_available = false
appearance_consistency_pass_ratio = None
```

Phase B 判定:

- true inlier ratios 已可用，不再由 p90 quantile artifact 定义。
- diagnostic/provider matching 统一为 `match_overlap_carriers`。
- D5 cache schema 已支持 persistent tube identity，并有单测验证 `save_npz` 写字段。
- 三窗口 non-identity Sim3 provider test 已通过。
- overlap policy 支持四种计划要求的策略；policy AP ablation 暂未运行，留到 Phase D provider rerun。

## 4. Phase C: direct D4RT-on-ScanNet reconstruction benchmark

目标: 按 v22 计划新增 direct reconstruction benchmark，直接评估 D4RT 输出点云/深度在 ScanNet 坐标系下的质量。该 benchmark 使用 ScanNet depth/pose/instance 作为 evaluation-only GT；raw/self rows 不使用 GT Sim3，eval-Sim3 rows 明确是 diagnostic-only，不能进入 method table。

新增代码:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`
  - 从已有 carrier/tube cache 读取 D4RT `xyz_ref/uv/valid/confidence`。
  - 输出 point-to-point residual、F@5/F@10、completeness@20、outlier@20、depth median/LS alignment 指标。
  - 输出 per-scene residual histogram 和 variant F@10 bar plot。
  - 支持 `--variants`、`--max-scenes`、`--max-windows-per-scene`、`--debug-progress`，便于遇到慢路径时做小规模复现。

### 4.1 初始 blocker 与修复

初始直接跑 scene0050 全 R0-R9 时，runner 在 eval-Sim3 anchor 构建阶段过慢，未及时产出。处理方式:

```bash
pkill -f 'tools.run_v22_direct_reconstruction_benchmark' || true
```

修复:

- 给 runner 增加 `--variants` / `--max-scenes` / `--max-windows-per-scene` / `--debug-progress`。
- eval-Sim3 fitting 先按 `--max-anchors` 采样 carrier anchors，再读取/反投影对应 GT depth，避免在 provider 内一次收集所有 anchor。
- self variants 先 raw load，再按 `--max-windows-per-scene` 截断窗口，避免无意义地 stitch 全量窗口 smoke。

### 4.2 scene0050 R0/R4/R5 smoke

临时 split:

```bash
printf 'scene0050_00\n' > /tmp/stream4d_v22_scene0050.txt
```

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list /tmp/stream4d_v22_scene0050.txt \
  --audit-root outputs/audit/v22_direct_reconstruction_scene0050_r0r4r5_smoke \
  --variants R0,R4,R5 \
  --depth-sample-stride 64 \
  --max-gt-points-per-scene 3000 \
  --max-pred-points-per-frame 60 \
  --max-windows-per-scene 1 \
  --max-anchors 512 \
  --debug-progress
```

结果:

| variant | F@10 | F@5 | completeness@20 | Chamfer-L1 | outlier@20 | depth delta1 median | depth AbsRel median |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 raw single chunk | 0.040025 | 0.010371 | 0.155163 | 2.124020 | 0.907292 | 0.334201 | 0.372214 |
| R4 eval-only scene Sim3 | 0.774526 | 0.322281 | 0.944473 | 0.152623 | 0.005729 | 0.981522 | 0.041555 |
| R5 eval-only per-chunk Sim3 | 0.774526 | 0.322281 | 0.944473 | 0.152623 | 0.005729 | 0.981522 | 0.041555 |

Artifact:

- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r0r4r5_smoke/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r0r4r5_smoke/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r0r4r5_smoke/*_residual_hist.png`
- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r0r4r5_smoke/direct_reconstruction_f10_by_variant.png`

### 4.3 scene0050 R1/R2/R3 self-stitch smoke

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list /tmp/stream4d_v22_scene0050.txt \
  --audit-root outputs/audit/v22_direct_reconstruction_scene0050_r1r2r3 \
  --variants R1,R2,R3 \
  --depth-sample-stride 64 \
  --max-gt-points-per-scene 3000 \
  --max-pred-points-per-frame 60 \
  --max-windows-per-scene 2 \
  --max-anchors 512 \
  --debug-progress
```

结果:

| variant | F@10 | F@5 | completeness@20 | Chamfer-L1 | outlier@20 | depth delta1 median | depth AbsRel median |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 sliding raw | 0.035874 | 0.008452 | 0.128601 | 2.428985 | 0.895667 | 0.218472 | 0.452097 |
| R2 self-Sim3 | 0.028903 | 0.007580 | 0.109396 | 2.866639 | 0.932667 | 0.172436 | 0.644672 |
| R3 scale-normalized self-Sim3 | 0.029448 | 0.006247 | 0.120370 | 2.869251 | 0.927667 | 0.169182 | 0.646745 |

判定: self-stitch/scale-normalized 在 direct reconstruction 上没有改善 raw geometry，反而降低 F@10/completeness；这和 v21.3 provider-level G9/G10 的负证据一致。

Artifact:

- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r1r2r3/*`

### 4.4 scene0050 R6/R7/R8/R9 occupancy/dense raw smoke

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list /tmp/stream4d_v22_scene0050.txt \
  --audit-root outputs/audit/v22_direct_reconstruction_scene0050_r6r7r8r9 \
  --variants R6,R7,R8,R9 \
  --depth-sample-stride 64 \
  --max-gt-points-per-scene 3000 \
  --max-pred-points-per-frame 60 \
  --max-windows-per-scene 2 \
  --max-anchors 512 \
  --debug-progress
```

结果:

| variant | F@10 | F@5 | completeness@20 | Chamfer-L1 | outlier@20 | depth delta1 median | depth AbsRel median |
|---|---:|---:|---:|---:|---:|---:|---:|
| R6 D5 warmstart raw | 0.036785 | 0.006062 | 0.156036 | 2.423119 | 0.893667 | 0.324388 | 0.377857 |
| R7 dense128 raw | 0.025877 | 0.006947 | 0.180592 | 2.116803 | 0.921875 | 0.338104 | 0.361807 |
| R8 D2r4 raw | 0.031051 | 0.007459 | 0.176440 | 2.107375 | 0.914583 | 0.370338 | 0.367863 |
| R9 D5 self-stitch | 0.037612 | 0.005571 | 0.143004 | 2.793762 | 0.917000 | 0.200434 | 0.482259 |

注意: R9 读取的是旧 v21.3 D5 cache，该 cache 没有 v22 新 persistent IDs，因此 self matching 主要退回 mutual-UV fallback。后面单独补了 v22 D5 persistent smoke。

Artifact:

- `Stream3D/outputs/audit/v22_direct_reconstruction_scene0050_r6r7r8r9/*`

### 4.5 probe5 R4 eval-Sim3 diagnostic

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --audit-root outputs/audit/v22_direct_reconstruction_probe5_r4_eval_scene \
  --variants R4 \
  --depth-sample-stride 80 \
  --max-gt-points-per-scene 2000 \
  --max-pred-points-per-frame 40 \
  --max-windows-per-scene 1 \
  --max-anchors 384 \
  --debug-progress
```

Probe5 mean:

| variant | scenes | F@10 | F@5 | completeness@20 | Chamfer-L1 | outlier@20 | depth delta1 median | depth AbsRel median | covered GT inst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 eval-only scene Sim3 | 5 | 0.613669 | 0.249488 | 0.856899 | 0.252397 | 0.112500 | 0.969155 | 0.045506 | 0.892631 |

Per-scene F@10 / completeness@20 / depth delta1:

| scene | F@10 | completeness@20 | depth delta1 |
|---|---:|---:|---:|
| scene0050_00 | 0.659442 | 0.917692 | 0.991042 |
| scene0011_00 | 0.654141 | 0.927520 | 0.976721 |
| scene0030_00 | 0.660749 | 0.869323 | 0.971098 |
| scene0081_01 | 0.403453 | 0.787542 | 0.919913 |
| scene0591_00 | 0.690561 | 0.782418 | 0.987002 |

Artifact:

- `Stream3D/outputs/audit/v22_direct_reconstruction_probe5_r4_eval_scene/*`

### 4.6 probe5 R0/R8 raw diagnostic

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --audit-root outputs/audit/v22_direct_reconstruction_probe5_r0_r8_raw \
  --variants R0,R8 \
  --depth-sample-stride 80 \
  --max-gt-points-per-scene 2000 \
  --max-pred-points-per-frame 40 \
  --max-windows-per-scene 1 \
  --max-anchors 384 \
  --debug-progress
```

Probe5 mean:

| variant | scenes | F@10 | F@5 | completeness@20 | Chamfer-L1 | outlier@20 | depth delta1 median | depth AbsRel median | covered GT inst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 raw single chunk | 5 | 0.005726 | 0.001442 | 0.028761 | 8.002120 | 0.987969 | 0.614228 | 0.241052 | 0.892631 |
| R8 D2r4 raw | 5 | 0.005211 | 0.000685 | 0.027247 | 7.962239 | 0.984375 | 0.596471 | 0.247281 | 0.891272 |

Per-scene F@10 / completeness@20 / depth delta1:

| scene | R0 F@10 | R0 comp@20 | R0 delta1 | R8 F@10 | R8 comp@20 | R8 delta1 |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.028630 | 0.143803 | 0.397574 | 0.026054 | 0.136235 | 0.284427 |
| scene0011_00 | 0.000000 | 0.000000 | 0.800607 | 0.000000 | 0.000000 | 0.800626 |
| scene0030_00 | 0.000000 | 0.000000 | 0.762180 | 0.000000 | 0.000000 | 0.757877 |
| scene0081_01 | 0.000000 | 0.000000 | 0.412338 | 0.000000 | 0.000000 | 0.430617 |
| scene0591_00 | 0.000000 | 0.000000 | 0.698440 | 0.000000 | 0.000000 | 0.708806 |

Artifact:

- `Stream3D/outputs/audit/v22_direct_reconstruction_probe5_r0_r8_raw/*`

Phase C 判定:

- Stop C 不触发: eval-Sim3 R4 在 probe5 上 F@10 `0.613669`、completeness@20 `0.856899`、depth delta1 `0.969155`，明显高于计划最低线。
- Stop D 触发: raw/self D4RT 在 probe5 上 F@10 只有 `0.005726/0.005211`，scene0050 self-stitch R2/R3 也只有 `0.028903/0.029448`。eval-Sim3 direct reconstruction 很好但 raw/self 极差，说明下一步应继续修 canonical scale/self-alignment，而不是启动 semantic method table。

## 5. v22 D5 persistent-ID real checkpoint smoke

目的: Phase B 已改 D5 exporter schema，但 v21.3 旧 D5 cache 没有 persistent ID。这里补一个真实 checkpoint 小规模 smoke，验证 `persistent_tube_id` 等字段能在 D5 real path 落盘。该 smoke 使用小 query budget，只用于身份传播/字段审计，不用于 AP/coverage 结论。

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.export_v21_3_occupancy_carrier_cache \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --variant D5 \
  --frame-stride 10 \
  --max-frames 48 \
  --window-size 32 \
  --window-stride 16 \
  --query-budget 512 \
  --source-points-per-round 256 \
  --query-chunk-size 1024 \
  --mark-radius-px 4 \
  --persistent-uv-radius 0.01 \
  --output-root outputs/stream4d_debug_v22_occupancy_d5_persistent_scene0050_smoke \
  --summary-root outputs/audit/v22_occupancy_d5_persistent_scene0050_smoke \
  --device cuda
```

stdout row:

```json
{
  "actual_source_query_count": 1024,
  "mark_radius_px": 4,
  "num_carriers_saved": 1013,
  "num_windows": 2,
  "output_scene_dir": "outputs/stream4d_debug_v22_occupancy_d5_persistent_scene0050_smoke/scene0050_00",
  "persistent_tube_retention_count": 6,
  "persistent_tube_retention_rate": 0.005923000987166831,
  "scene": "scene0050_00",
  "status": "ok",
  "total_d4rt_time_sec": 13.120853185653687,
  "variant": "D5_occupancy_dense_overlap_warmstart",
  "warmstart_track_count": 353
}
```

Window summary:

| window | carriers | persistent tubes | warmstarted carriers | queries | warmstart tracks | retention count | retention rate | acceptance rate | uv_in01 | time sec |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 507 | 507 | 0 | 512 | 0 | 0 | 0.000000 | 0.000000 | 0.218997 | 6.722571 |
| 1 | 506 | 506 | 6 | 512 | 353 | 6 | 0.011858 | 0.011834 | 0.235425 | 6.398282 |

NPZ 字段检查:

```text
carriers_window000.npz
has_fields {'persistent_tube_id': True, 'parent_tube_id': True, 'warmstart_source_chunk': True, 'warmstart_source_frame': True, 'is_warmstarted': True, 'src_frame_global': True}
num_carriers 507 unique_persistent 507 warmstarted 0

carriers_window001.npz
has_fields {'persistent_tube_id': True, 'parent_tube_id': True, 'warmstart_source_chunk': True, 'warmstart_source_frame': True, 'is_warmstarted': True, 'src_frame_global': True}
num_carriers 506 unique_persistent 506 warmstarted 6
```

Artifact:

- `Stream3D/outputs/stream4d_debug_v22_occupancy_d5_persistent_scene0050_smoke/scene0050_00/carriers_window*.npz`
- `Stream3D/outputs/audit/v22_occupancy_d5_persistent_scene0050_smoke/scene0050_00/summary.*`

### 5.1 D5 persistent provider smoke: overlap policy ablation

目的: 在同一个小 cache 上确认 provider 能读到 persistent IDs 并产出 match-source diagnostics。因为这是 512-budget smoke，AP 不是有效实验目标。

all-window 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list /tmp/stream4d_v22_scene0050.txt \
  --debug-root outputs/stream4d_debug_v22_occupancy_d5_persistent_scene0050_smoke \
  --output-prefix stream4d_v22_d5_persistent_smoke_allwin \
  --audit-root outputs/audit/v22_d5_persistent_provider_smoke_allwin \
  --variants G2,G6 \
  --nn-radius 0.05 \
  --density-alpha 2.0 \
  --overlap-policy all_window_union
```

best-confidence 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v21_3_stream3d_provider_replacement \
  --seq-list /tmp/stream4d_v22_scene0050.txt \
  --debug-root outputs/stream4d_debug_v22_occupancy_d5_persistent_scene0050_smoke \
  --output-prefix stream4d_v22_d5_persistent_smoke_bestwin \
  --audit-root outputs/audit/v22_d5_persistent_provider_smoke_bestwin \
  --variants G2,G6 \
  --nn-radius 0.05 \
  --density-alpha 2.0 \
  --overlap-policy best_confidence
```

结果:

| policy | variant | status | AP | pre% | hit | #pred | self pairs | stable-id matches | mutual-UV matches | abs010 inlier | residual p90 | candidate windows | duplicate hit |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_window_union | G2 | ok | NA | 0.000378 | 0.037554 | 1.0 | 1.0 | 31.0 | 76.0 | 0.504673 | 0.175374 | 1.333333 | 0.166667 |
| all_window_union | G6 | ok | NA | 0.000378 | 0.037554 | 1.0 | 1.0 | 31.0 | 76.0 | 0.504673 | 0.175374 | 1.333333 | 0.166667 |
| best_confidence | G2 | ok | NA | 0.000378 | 0.037554 | 1.0 | 1.0 | 31.0 | 76.0 | 0.504673 | 0.175374 | 1.333333 | 0.166667 |
| best_confidence | G6 | ok | NA | 0.000378 | 0.037554 | 1.0 | 1.0 | 31.0 | 76.0 | 0.504673 | 0.175374 | 1.333333 | 0.166667 |

说明:

- AP 为 NA，是因为该小 cache 只形成 1 个 prediction，evaluator 对空类别均值返回 `nan`；不能解读为 method 成功或失败。
- 两种 overlap policy 数值相同，因为最终 `source_windows_mean=1.0`，policy 没有足够重复窗口选择空间。
- 关键正向证据是 provider self-stitch 现在确实读到 stable/persistent ID: `self_stitch_match_source_stable_id_count_mean=31.0`，不再完全依赖 mutual UV fallback。
- 关键负向证据是 retention 很低: exporter window1 retention 只有 `6/506`，provider pair anchor 只有 `107`，说明小预算 D5 persistent smoke 还远不足以支持语义或 AP 结论。

Artifact:

- `Stream3D/outputs/audit/v22_d5_persistent_provider_smoke_allwin/*`
- `Stream3D/outputs/audit/v22_d5_persistent_provider_smoke_bestwin/*`

## 6. 最终验证与 stop decision

最终命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile evaluation/*.py stream4d/*.py stream4d_native/*.py geometry_provider/*.py tools/*.py tests/*.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass，无输出。

Ran 63 tests in 1.677s
OK
```

最终执行判定:

- Phase A: 完成并通过。
- Phase B: 完成并通过；true inlier / unified matching / persistent schema / three-window tests / overlap policy 支持均已落地。
- Phase C: 完成 direct reconstruction benchmark；Stop C 不触发，Stop D 触发。
- Phase D/E: 未启动 full semantic/provider method rerun；只做了 v22 D5 persistent-ID 小规模真实 smoke 和 provider policy smoke。原因是 Phase C 已明确 raw/self canonical alignment 失败，按计划 Stop D 应继续修 scale/stitching，不应启动 semantic method table。
- Phase F/method table: 未运行/未生成。

## 7. 2026-06-11 继续执行: camera-space direct diagnostic 与 UV+Z rectification smoke

用户要求“继续下一步”后，沿 v22 Stop D 继续，不启动 method table。目标是把 raw/self 失败进一步拆开:

```text
world-space F@10 低，是因为缺 ScanNet world pose/canonical anchoring？
还是 D4RT raw xyz 在 camera-local space 内也不符合 GT depth 点云？
```

### 7.1 代码修改

文件:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`

修改:

1. 新增 per-frame camera-space point metrics:
   - `camera_space_fscore@5cm/10cm/20cm`
   - `camera_space_chamfer_l1`
   - `camera_space_outlier_rate_20cm`
   - `camera_space_pred_to_gt_median/p90`
   - `camera_space_frame_count`
2. 新增 `VariantSpec.point_mode`。
3. 新增 diagnostic-only variants:
   - `R10`: single-chunk `uv_pred + z` 通过 ScanNet intrinsics 回投 camera-space points。
   - `R11`: D2r4 `uv_pred + z` 通过 ScanNet intrinsics 回投 camera-space points。

重要说明:

- R10/R11 不使用 ScanNet depth/pose/mesh 做 prediction；只用 D4RT `uv_pred` 和 D4RT `z` 加 ScanNet intrinsics 重建 camera-space xyz。
- ScanNet depth/pose 仍只用于 evaluation metrics。
- R10/R11 的 world-space F-score 不可解释为 method，因为它们是 camera-space points，未进入 ScanNet world/canonical frame；只看 `camera_space_*` 指标。

编译:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/run_v22_direct_reconstruction_benchmark.py
```

结果: pass，无输出。

### 7.2 scene0050 camera-space raw/self diagnostic

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list /tmp/stream4d_v22_scene0050.txt \
  --audit-root outputs/audit/v22_1_camera_space_scene0050_r0r4r8 \
  --variants R0,R4,R8 \
  --depth-sample-stride 64 \
  --max-gt-points-per-scene 3000 \
  --max-pred-points-per-frame 60 \
  --max-windows-per-scene 1 \
  --max-anchors 512 \
  --debug-progress
```

结果:

| variant | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 | depth LS delta1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 raw | 0.040025 | 0.001529 | 0.009415 | 4.537961 | 0.991667 | 0.334201 | 0.450543 |
| R4 eval-Sim3 | 0.774526 | 0.308946 | 0.765074 | 0.302274 | 0.088542 | 0.981522 | 0.982609 |
| R8 D2r4 raw | 0.031051 | 0.000508 | 0.008848 | 4.539411 | 0.991146 | 0.370338 | 0.448649 |

Artifact:

- `Stream3D/outputs/audit/v22_1_camera_space_scene0050_r0r4r8/*`

### 7.3 scene0050 camera-space self-stitch diagnostic

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list /tmp/stream4d_v22_scene0050.txt \
  --audit-root outputs/audit/v22_1_camera_space_scene0050_r1r2r3 \
  --variants R1,R2,R3 \
  --depth-sample-stride 64 \
  --max-gt-points-per-scene 3000 \
  --max-pred-points-per-frame 60 \
  --max-windows-per-scene 2 \
  --max-anchors 512 \
  --debug-progress
```

结果:

| variant | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 | depth LS delta1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 sliding raw | 0.035874 | 0.001025 | 0.009551 | 4.185458 | 0.992014 | 0.218472 | 0.520197 |
| R2 self-Sim3 | 0.028903 | 0.001025 | 0.006287 | 4.925231 | 0.994444 | 0.172436 | 0.599891 |
| R3 scale-normalized self-Sim3 | 0.029448 | 0.001025 | 0.006287 | 4.945916 | 0.994444 | 0.169182 | 0.609989 |

Artifact:

- `Stream3D/outputs/audit/v22_1_camera_space_scene0050_r1r2r3/*`

### 7.4 probe5 camera-space R0/R4/R8 diagnostic

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --audit-root outputs/audit/v22_1_camera_space_probe5_r0r4r8 \
  --variants R0,R4,R8 \
  --depth-sample-stride 80 \
  --max-gt-points-per-scene 2000 \
  --max-pred-points-per-frame 40 \
  --max-windows-per-scene 1 \
  --max-anchors 384 \
  --debug-progress
```

Probe5 mean:

| variant | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 | depth LS delta1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 raw | 0.005726 | 0.005737 | 0.035425 | 3.187777 | 0.960469 | 0.614228 | 0.652708 |
| R4 eval-Sim3 | 0.613669 | 0.173169 | 0.490943 | 0.612816 | 0.443438 | 0.969155 | 0.971001 |
| R8 D2r4 raw | 0.005211 | 0.004404 | 0.033213 | 3.180623 | 0.962031 | 0.596471 | 0.661383 |

Artifact:

- `Stream3D/outputs/audit/v22_1_camera_space_probe5_r0r4r8/*`

### 7.5 UV+Z camera-space rectification smoke

scene0050 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list /tmp/stream4d_v22_scene0050.txt \
  --audit-root outputs/audit/v22_1_uvz_camera_scene0050_r10r11 \
  --variants R10,R11 \
  --depth-sample-stride 64 \
  --max-gt-points-per-scene 3000 \
  --max-pred-points-per-frame 60 \
  --max-windows-per-scene 1 \
  --max-anchors 512 \
  --debug-progress
```

scene0050 结果:

| variant | point mode | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 | depth LS delta1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R10 single UV+Z | uvz_camera | 0.000000 | 0.020943 | 0.095642 | 2.464201 | 0.889063 | 0.334201 | 0.450543 |
| R11 D2r4 UV+Z | uvz_camera | 0.000000 | 0.022487 | 0.091912 | 2.460551 | 0.891667 | 0.370338 | 0.448649 |

probe5 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --audit-root outputs/audit/v22_1_uvz_camera_probe5_r10r11 \
  --variants R10,R11 \
  --depth-sample-stride 80 \
  --max-gt-points-per-scene 2000 \
  --max-pred-points-per-frame 40 \
  --max-windows-per-scene 1 \
  --max-anchors 384 \
  --debug-progress
```

Probe5 mean:

| variant | point mode | world F@10 | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth median delta1 | depth LS delta1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R10 single UV+Z | uvz_camera | 0.000000 | 0.013578 | 0.060304 | 1.863811 | 0.934531 | 0.614228 | 0.652708 |
| R11 D2r4 UV+Z | uvz_camera | 0.000000 | 0.010180 | 0.060192 | 1.857475 | 0.936250 | 0.596471 | 0.661383 |

Artifact:

- `Stream3D/outputs/audit/v22_1_uvz_camera_scene0050_r10r11/*`
- `Stream3D/outputs/audit/v22_1_uvz_camera_probe5_r10r11/*`

### 7.6 最终验证

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/run_v22_direct_reconstruction_benchmark.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass，无输出。

Ran 63 tests in 1.656s
OK
```

本轮 continuation 判定:

- camera-space 指标确认 raw/self 不只是缺 world pose。R0/R8 probe5 camera F@10 只有 `0.005737/0.004404`，R4 eval-Sim3 转回 camera-space 后为 `0.173169`。
- self-stitch 没有修复 camera-local geometry。scene0050 R1/R2/R3 camera F@10 都约 `0.001025`，R2/R3 camera outlier@20 仍约 `0.994444`。
- UV+Z 回投是小幅正修复但远不够。probe5 R10/R11 camera F@10 为 `0.013578/0.010180`，比 R0/R8 raw 高，但仍远低于 R4 camera F@10 `0.173169`，且 camera outlier@20 仍约 `0.934531/0.936250`。
- 因此下一步不应只修 world pose/canonical anchoring，也不应只替换 x/y 为 pinhole UV+Z；需要继续查 D4RT `z` scale/shift、uv-z correspondence、valid/visibility filtering 和 OpenD4RT 输出坐标约定。

## 8. Continuation: z/uv/filter/depth-calibration diagnostic

目的: 接 7.6 的 blocker 继续拆解。上一轮已经排除“只差 world pose”和“只用 UV+Z 回投就能修好”。本轮专门验证:

- D4RT `z` 是否只是需要线性 scale/shift。
- 非 GT 的正深度过滤是否能显著提升 camera-space direct reconstruction。
- `xyz` 与 `uv` 是否可能只是轴顺序/符号约定读错。
- 这些修复是否能把 raw/self 结果推进到可进入 provider/AP 的 gate。

### 8.1 代码修改

文件:

- `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`

修改:

- `VariantSpec` 新增 `depth_calibration` 字段。
- 新增 R12/R13/R14/R15:
  - R12: single-chunk UV+Z + eval-only median depth calibration。
  - R13: single-chunk UV+Z + eval-only linear depth calibration。
  - R14: D2r4 UV+Z + eval-only median depth calibration。
  - R15: D2r4 UV+Z + eval-only linear depth calibration。
- 新增 `_fit_depth_calibration`，用 ScanNet depth 在当前 scene/window 内拟合 D4RT z 到 GT depth 的 median scale 与 linear scale/shift，并写出 `depth_calibration_*` 诊断字段。
- 新增 `--min-pred-z`，用于非 GT 正深度过滤对照。
- 新增 `_raw_uvz_signal_metrics`:
  - `raw_uvz_positive_z_rate`
  - `raw_uvz_reproj_error_px_*`
  - `raw_uvz_best_reproj_*`
- `raw_uvz_best_reproj_*` 会遍历 xyz 轴排列与符号，找最小 UV 重投影误差，用于排查 OpenD4RT 输出坐标解释是否只是轴/符号误读。
- 修复 `_instance_coverage` 性能 bug: 以前每个预测点重复 `np.unique(inst)`，现改为每帧缓存 instance id set。指标定义不变，只减少重复计算。

解释器注意:

```bash
python -m py_compile Stream3D/tools/run_v22_direct_reconstruction_benchmark.py
```

结果: `/bin/bash: python: command not found`。

```bash
python3 -m tools.run_v22_direct_reconstruction_benchmark ...
```

结果: `ModuleNotFoundError: No module named 'matplotlib'`。

因此继续沿用项目环境:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
```

### 8.2 scene0050 GT-only depth calibration upper-bound

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/run_v22_direct_reconstruction_benchmark.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_scene0050.txt \
  --audit-root outputs/audit/v22_2_depth_calibrated_scene0050_r10_r15 \
  --variants R10,R11,R12,R13,R14,R15 \
  --max-anchors 8000 \
  --max-pred-points-per-frame 1500 \
  --depth-sample-stride 12 \
  --max-gt-points-per-scene 60000 \
  --debug-progress
```

结果:

| variant | calibration | pred pts | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth raw delta1 | depth LS delta1 | calib anchors | calib linear scale | calib linear shift | raw z+ rate | raw uv reproj median px |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R10 single UV+Z | none | 33752 | 0.061262 | 0.155613 | 2.324445 | 0.846409 | 0.212504 | 0.410729 | NA | NA | NA | 0.616250 | 649.548112 |
| R11 D2r4 UV+Z | none | 48000 | 0.067654 | 0.159721 | 2.312221 | 0.848750 | 0.207035 | 0.469525 | NA | NA | NA | 0.613375 | 642.845727 |
| R12 single median | eval-only median | 33752 | 0.078704 | 0.188637 | 2.144429 | 0.825701 | 0.356152 | 0.410729 | 4805 | 0.201875 | 1.362088 | 0.616250 | 649.548112 |
| R13 single linear | eval-only linear | 33752 | 0.185883 | 0.435729 | 0.602599 | 0.393403 | 0.439775 | 0.705642 | 4805 | 0.201875 | 1.362088 | 0.616250 | 649.548112 |
| R14 D2r4 median | eval-only median | 48000 | 0.085671 | 0.198526 | 2.128403 | 0.822729 | 0.349138 | 0.469525 | 4777 | 0.197679 | 1.372338 | 0.613375 | 642.845727 |
| R15 D2r4 linear | eval-only linear | 48000 | 0.200641 | 0.443457 | 0.590846 | 0.387583 | 0.477873 | 0.713825 | 4777 | 0.197679 | 1.372338 | 0.613375 | 642.845727 |

Artifact:

- `Stream3D/outputs/audit/v22_2_depth_calibrated_scene0050_r10_r15/*`

### 8.3 non-GT positive-z filter

scene0050 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_scene0050.txt \
  --audit-root outputs/audit/v22_2_uvz_positive_z_scene0050_r10_r11 \
  --variants R10,R11 \
  --max-anchors 8000 \
  --max-pred-points-per-frame 1500 \
  --depth-sample-stride 12 \
  --max-gt-points-per-scene 60000 \
  --min-pred-z 0.001 \
  --debug-progress
```

probe5 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --audit-root outputs/audit/v22_2_uvz_positive_z_probe5_r10_r11 \
  --variants R10,R11 \
  --max-anchors 8000 \
  --max-pred-points-per-frame 1500 \
  --depth-sample-stride 12 \
  --max-gt-points-per-scene 60000 \
  --min-pred-z 0.001
```

结果:

| scope | variant | pred pts | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | per-inst covered |
|---|---|---:|---:|---:|---:|---:|---:|
| scene0050 | R10 positive-z | 17985 | 0.077346 | 0.196634 | 1.274156 | 0.800362 | 0.541667 |
| scene0050 | R11 positive-z | 29364 | 0.085877 | 0.202954 | 1.262718 | 0.802104 | 0.583333 |
| probe5 | R10 positive-z | 23798.0 | 0.062131 | 0.172887 | 1.258419 | 0.790796 | 0.866835 |
| probe5 | R11 positive-z | 40103.0 | 0.071781 | 0.189676 | 1.245578 | 0.793083 | 0.883502 |

对照:

- probe5 R10 no filter camera F@10 `0.058913` -> positive-z `0.062131`。
- probe5 R11 no filter camera F@10 `0.068136` -> positive-z `0.071781`。
- scene0050 正深度过滤提升更明显，但覆盖实例比降到 `0.541667/0.583333`，不是完整 support 修复。

Artifact:

- `Stream3D/outputs/audit/v22_2_uvz_positive_z_scene0050_r10_r11/*`
- `Stream3D/outputs/audit/v22_2_uvz_positive_z_probe5_r10_r11/*`

### 8.4 probe5 GT-only depth calibration upper-bound

命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --audit-root outputs/audit/v22_2_depth_calibrated_probe5_r10_r15 \
  --variants R10,R11,R12,R13,R14,R15 \
  --max-anchors 8000 \
  --max-pred-points-per-frame 1500 \
  --depth-sample-stride 12 \
  --max-gt-points-per-scene 60000
```

Probe5 mean:

| variant | calibration | pred pts | camera F@10 | camera F@20 | camera Chamfer-L1 | camera outlier@20 | depth raw delta1 | depth LS delta1 | calib anchors | calib linear scale | calib linear shift | raw uv reproj median px |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R10 single UV+Z | none | 26951.6 | 0.058913 | 0.164682 | 1.468482 | 0.800008 | 0.212910 | 0.641025 | NA | NA | NA | 377.689976 |
| R11 D2r4 UV+Z | none | 43830.2 | 0.068136 | 0.181029 | 1.455479 | 0.802413 | 0.203974 | 0.655746 | NA | NA | NA | 371.276760 |
| R12 single median | eval-only median | 26951.6 | 0.196149 | 0.427154 | 0.892196 | 0.519310 | 0.608358 | 0.641025 | 6157.6 | 0.433842 | 0.786679 | 377.689976 |
| R13 single linear | eval-only linear | 26951.6 | 0.215022 | 0.477449 | 0.563770 | 0.397737 | 0.650033 | 0.700008 | 6157.6 | 0.433842 | 0.786679 | 377.689976 |
| R14 D2r4 median | eval-only median | 43830.2 | 0.221848 | 0.450228 | 0.877473 | 0.522725 | 0.609444 | 0.655746 | 6280.4 | 0.435936 | 0.786795 | 371.276760 |
| R15 D2r4 linear | eval-only linear | 43830.2 | 0.244729 | 0.499092 | 0.547921 | 0.397922 | 0.659358 | 0.704606 | 6280.4 | 0.435936 | 0.786795 | 371.276760 |

R13/R15 per-scene:

| variant | scene | camera F@10 | camera F@20 | Chamfer-L1 | outlier@20 | linear scale | linear shift | raw uv reproj median px |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R13 | scene0050_00 | 0.185883 | 0.435729 | 0.602599 | 0.393403 | 0.201875 | 1.362088 | 649.548112 |
| R13 | scene0011_00 | 0.240686 | 0.575633 | 0.418378 | 0.373952 | 0.468610 | 0.564484 | 388.115456 |
| R13 | scene0030_00 | 0.272826 | 0.558141 | 0.458361 | 0.351679 | 0.393457 | 1.119505 | 254.947141 |
| R13 | scene0081_01 | 0.112065 | 0.292987 | 0.813087 | 0.609512 | 0.675702 | 0.458969 | 433.496462 |
| R13 | scene0591_00 | 0.263649 | 0.524752 | 0.526425 | 0.260139 | 0.429564 | 0.428349 | 162.342710 |
| R15 | scene0050_00 | 0.200641 | 0.443457 | 0.590846 | 0.387583 | 0.197679 | 1.372338 | 642.845727 |
| R15 | scene0011_00 | 0.331264 | 0.630486 | 0.386262 | 0.374421 | 0.477355 | 0.544455 | 368.040013 |
| R15 | scene0030_00 | 0.283783 | 0.565026 | 0.452771 | 0.347250 | 0.391967 | 1.132190 | 255.101827 |
| R15 | scene0081_01 | 0.144180 | 0.330075 | 0.790911 | 0.619251 | 0.679095 | 0.466488 | 428.298275 |
| R15 | scene0591_00 | 0.263775 | 0.526418 | 0.518816 | 0.261104 | 0.433586 | 0.418503 | 162.097957 |

Artifact:

- `Stream3D/outputs/audit/v22_2_depth_calibrated_probe5_r10_r15/*`

### 8.5 axis/sign reprojection convention diagnostic

scene0050 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/run_v22_direct_reconstruction_benchmark.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_scene0050.txt \
  --audit-root outputs/audit/v22_2_reproj_convention_scene0050_r10_r11 \
  --variants R10,R11 \
  --max-anchors 8000 \
  --max-pred-points-per-frame 1500 \
  --depth-sample-stride 12 \
  --max-gt-points-per-scene 60000
```

probe5 命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --audit-root outputs/audit/v22_2_reproj_convention_probe5_r10_r11 \
  --variants R10,R11 \
  --max-anchors 8000 \
  --max-pred-points-per-frame 1500 \
  --depth-sample-stride 12 \
  --max-gt-points-per-scene 60000
```

Best axis/sign convention result:

| scope | variant | best convention | best median px | best p90 px | default median px | default p90 px |
|---|---|---|---:|---:|---:|---:|
| scene0050 | R10 | `+1,+0,+2` | 641.390188 | 5837.675830 | 649.548112 | 5845.670934 |
| scene0050 | R11 | `+1,+0,+2` | 627.465555 | 6346.574253 | 642.845727 | 6381.275413 |
| probe5 mean | R10 | mixed, mostly `+0,+1,+2` | 376.058392 | 1630.224503 | 377.689976 | 1631.823524 |
| probe5 mean | R11 | mixed, mostly `+0,+1,+2` | 368.200725 | 1731.522253 | 371.276760 | 1738.462485 |

Per-scene notes:

- scene0050 best swaps x/y (`+1,+0,+2`) but median error remains `> 600px`。
- scene0011/0030/0081/0591 的 best convention 基本就是 default `+0,+1,+2`。
- 轴/符号搜索不能把 uv reprojection error 降到可用级别，因此当前问题不是简单的 axis/sign convention typo。

Artifact:

- `Stream3D/outputs/audit/v22_2_reproj_convention_scene0050_r10_r11/*`
- `Stream3D/outputs/audit/v22_2_reproj_convention_probe5_r10_r11/*`

### 8.6 最终验证

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/run_v22_direct_reconstruction_benchmark.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass，无输出。

Ran 63 tests in 1.674s
OK
```

本轮 continuation 判定:

- GT-only linear depth calibration 是强上界修复: probe5 R15 camera F@10/F@20 到 `0.244729/0.499092`，明显高于 R11 no-calibration `0.068136/0.181029`。
- 但该修复使用 ScanNet depth 拟合 z scale/shift，明确 diagnostic-only / forbidden for method。
- 非 GT positive-z filter 只是小修: probe5 R11 camera F@10 `0.068136 -> 0.071781`，不足以通过 geometry gate。
- best axis/sign search 无法救回 UV correspondence: probe5 R11 median reproj error `371.276760px -> 368.200725px`，仍是数百像素级。
- depth linear scale/shift 跨 scene 变化很大，例如 R15 scale/shift 从 scene0050 `0.197679/1.372338` 到 scene0081 `0.679095/0.466488`，说明不能用一个固定全局常数修正。
- 因此下一步应继续查 D4RT `z` 生成/归一化反变换和 uv-z correspondence，而不是启动 provider/AP 或 method table。

## 9. v22.3 D4RT `xyz_local` vs `xyz_ref0` coordinate branch diagnostic

用户继续要求“继续下一步”后，沿 8.6 的 blocker 继续查 D4RT 输出坐标约定。本轮重点验证一个具体怀疑: 旧 cache 中 `uv_pred` 与 `xyz_ref` 可能来自 OpenD4RT 的不同 decode branch。

### 9.1 代码阅读结论

读取文件:

```bash
sed -n '1,280p' Stream3D/stream4d/d4rt_adapter.py
sed -n '1,220p' Open-d4rt/infer_track_3d.py
```

确认:

- 本仓库旧 `D4RTAdapter.track_window` 同时调用:
  - `pred_local = decode(query_local, t_cam=t_tgt)`
  - `pred_ref = decode(query_ref, t_cam=0)`
- 旧 cache 保存 `uv_pred = pred_local["uv_2d"]`，但只保存 `xyz_ref = pred_ref["xyz_3d"]`。
- OpenD4RT 官方 `infer_track_3d.py` 同时返回 `tracks_xyz_local = pred_local["xyz_3d"]` 和 `tracks_xyz_ref0 = pred_ref["xyz_3d"]`。
- 因此 v22.2 的 R10/R11 很可能把 target-frame `uv_pred` 与 ref0/canonical `xyz_ref` 的 z 混用，导致 UV-Z correspondence 诊断偏坏。

### 9.2 代码修改

修改文件:

- `Stream3D/stream4d/carrier_store.py`
  - `CarrierBatch` 新增可选字段 `xyz_local`。
  - `save_npz` 在字段存在时写入 `xyz_local`。
- `Stream3D/stream4d/d4rt_adapter.py`
  - 从 `pred_local["xyz_3d"]` 提取 `xyz_local`。
  - finite valid mask 同时检查 `uv_pred` / `xyz_ref` / `xyz_local`。
  - empty batch 也写入空 `xyz_local`。
- `Stream3D/stream4d/replay_memory.py`
  - 读取 carrier cache 时兼容可选 `xyz_local`。
- `Stream3D/tools/diagnose_v22_d4rt_local_vs_ref.py`
  - 新增 real-checkpoint diagnostic runner，直接比较 `xyz_ref0` 与 `xyz_local`。
  - 指标包括 target UV reprojection、raw/median/linear depth fit、per-frame camera-space point F-score、UV+Z raw/median/linear camera F-score、source self UV/depth。
  - 可选 `--cache-root` 写出包含 `xyz_local` 的 `carriers_window*.npz`，用于后续 provider/direct runner。
  - 修正 camera metric 为 per-frame 计算，避免把不同 frame camera-space 点云池化后得到虚高结果。
  - `_uvz_points_from_depth` 增加 depth/uv/frame 数组长度检查。

### 9.3 validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  stream4d/carrier_store.py \
  stream4d/d4rt_adapter.py \
  stream4d/replay_memory.py \
  tools/diagnose_v22_d4rt_local_vs_ref.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest \
  tests.test_native_occupancy_and_builder \
  tests.test_v21_3_geometry_provider
```

结果:

```text
py_compile pass，无输出。

Ran 16 tests in 0.035s
OK
```

### 9.4 scene0050 small real-checkpoint diagnostic

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v22_d4rt_local_vs_ref \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --frame-stride 10 \
  --max-frames 16 \
  --max-source-points 1024 \
  --max-points-per-mask 16 \
  --min-points-per-mask 4 \
  --query-chunk-size 1024 \
  --output-prefix outputs/audit/v22_3_local_vs_ref_scene0050/local_vs_ref_scene0050 \
  --cache-root outputs/stream4d_debug_v22_local_xyz_scene0050_r1
```

scene0050 结果:

| branch | target reproj median px | target reproj p90 px | raw depth delta1 | linear depth delta1 | xyz camera F@10 | xyz camera F@20 | raw UVZ F@10 | raw UVZ F@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xyz_ref0` | 395.720598 | 1017.236081 | 0.207089 | 0.778596 | 0.000524 | 0.003142 | 0.098752 | 0.240623 |
| `xyz_local` | 36.824630 | 68.256919 | 0.022706 | 0.930795 | 0.001000 | 0.035169 | 0.016902 | 0.081275 |

说明: 该 standalone scene0050 smoke 是 calibrated UVZ 字段加入前的 artifact；eval-only calibrated UVZ 上界见 9.5 probe5 rerun 中的 scene0050 row。

Artifact:

- `Stream3D/outputs/audit/v22_3_local_vs_ref_scene0050/local_vs_ref_scene0050.*`
- `Stream3D/outputs/stream4d_debug_v22_local_xyz_scene0050_r1/scene0050_00/carriers_window000.npz`

### 9.5 probe5 `xyz_local` vs `xyz_ref0` diagnostic

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v22_d4rt_local_vs_ref \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet_v6_probe5.txt \
  --frame-stride 10 \
  --max-frames 16 \
  --max-source-points 512 \
  --max-points-per-mask 16 \
  --min-points-per-mask 4 \
  --query-chunk-size 1024 \
  --output-prefix outputs/audit/v22_3_local_vs_ref_probe5/local_vs_ref_probe5 \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1
```

Probe5 mean:

| branch | target reproj median px | target reproj p90 px | raw depth delta1 | median depth delta1 | linear depth delta1 | xyz camera F@10 | xyz camera F@20 | xyz outlier@20 | raw UVZ F@10 | raw UVZ F@20 | linear UVZ F@10 | linear UVZ F@20 | source self UV median px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xyz_ref0` | 226.747204 | 438.900232 | 0.217184 | 0.810198 | 0.819182 | 0.016254 | 0.071730 | 0.874549 | 0.040446 | 0.129410 | 0.229843 | 0.562896 | 4.379759 |
| `xyz_local` | 30.753682 | 52.095698 | 0.197125 | 0.904115 | 0.903552 | 0.014329 | 0.082448 | 0.837975 | 0.022713 | 0.107783 | 0.355244 | 0.693759 | 4.379759 |

Probe5 `xyz_local` per-scene highlights:

| scene | reproj median px | raw depth delta1 | linear depth delta1 | xyz F@10 | raw UVZ F@10 | linear UVZ F@10 | linear UVZ F@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 36.882055 | 0.019104 | 0.938281 | 0.000789 | 0.012700 | 0.602850 | 0.923622 |
| scene0011_00 | 19.019359 | 0.034646 | 0.902362 | 0.001315 | 0.001567 | 0.177168 | 0.617651 |
| scene0030_00 | 26.833669 | 0.071085 | 0.916229 | 0.010321 | 0.007297 | 0.504020 | 0.888708 |
| scene0081_01 | 32.554647 | 0.770416 | 0.858243 | 0.051153 | 0.080385 | 0.201751 | 0.504564 |
| scene0591_00 | 38.478680 | 0.090375 | 0.902645 | 0.008068 | 0.011618 | 0.290428 | 0.534251 |

Artifact:

- `Stream3D/outputs/audit/v22_3_local_vs_ref_probe5/local_vs_ref_probe5.*`
- `Stream3D/outputs/stream4d_debug_v22_local_xyz_probe5_r1/*/carriers_window000.npz`

### 9.6 final validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  stream4d/carrier_store.py \
  stream4d/d4rt_adapter.py \
  stream4d/replay_memory.py \
  tools/diagnose_v22_d4rt_local_vs_ref.py
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass，无输出。

Ran 63 tests in 1.681s
OK
```

Cache 字段抽查命令:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
from pathlib import Path
import numpy as np
paths = [
    Path('Stream3D/outputs/stream4d_debug_v22_local_xyz_scene0050_r1/scene0050_00/carriers_window000.npz'),
    Path('Stream3D/outputs/stream4d_debug_v22_local_xyz_probe5_r1/scene0050_00/carriers_window000.npz'),
]
for path in paths:
    with np.load(path, allow_pickle=True) as data:
        print(path)
        print('xyz_local_present', 'xyz_local' in data.files, 'shape', data['xyz_local'].shape if 'xyz_local' in data.files else None)
        print('xyz_ref_shape', data['xyz_ref'].shape, 'uv_pred_shape', data['uv_pred'].shape)
PY
```

结果:

```text
Stream3D/outputs/stream4d_debug_v22_local_xyz_scene0050_r1/scene0050_00/carriers_window000.npz
xyz_local_present True shape (16, 1024, 3)
xyz_ref_shape (16, 1024, 3) uv_pred_shape (16, 1024, 2)
Stream3D/outputs/stream4d_debug_v22_local_xyz_probe5_r1/scene0050_00/carriers_window000.npz
xyz_local_present True shape (16, 512, 3)
xyz_ref_shape (16, 512, 3) uv_pred_shape (16, 512, 2)
```

本轮 continuation 判定:

- 找到并验证了一个实质坐标分支问题: `uv_pred` 更应与 target-frame `xyz_local` 对齐，而不是与 ref0/canonical `xyz_ref0` 混用。
- `xyz_local` 明显修复 UV/XYZ correspondence: probe5 target reproj median `226.747204px -> 30.753682px`，p90 `438.900232px -> 52.095698px`。
- 但 `xyz_local` raw geometry 仍不是 method success: probe5 raw `xyz_local` camera F@10/F@20 只有 `0.014329/0.082448`，raw UVZ F@10/F@20 只有 `0.022713/0.107783`，outlier@20 仍 `0.837975/0.801467`。
- eval-only depth calibration 的上界更强: `xyz_local` linear UVZ F@10/F@20 到 `0.355244/0.693759`，scene0050 到 `0.602850/0.923622`，但这使用 ScanNet depth 拟合，不能作为 method。
- Source self UV median `4.379759px`，说明 source query identity 基本可用；source raw depth delta1 仍低，`xyz_local` mean 只有 `0.091690`，说明 source depth/scale 也未自然正确。
- 因此下一步不启动 Phase F/method table；应把后续 direct/provider diagnostic 改为显式区分 `xyz_local` 和 `xyz_ref0`，并继续寻找非 GT 的 per-window depth scale/shift 或 D4RT-native 反归一化信号。

## 10. v22.4 OpenD4RT loss-space xyz transform diagnostic

目的: v22.3 确认 `xyz_local` 是正确 target-frame branch 后，继续追查为什么 raw `xyz_local` 仍不能形成 metric camera geometry。重点检查 OpenD4RT 训练 loss 是否对 `xyz_3d` 使用了归一化或非线性变换，并验证这些变换能否作为非 GT diagnostic signal 改善 raw camera metrics。

### 10.1 source/config inspection

读取位置:

- `Open-d4rt/src/losses/d4rt_loss.py`
- `Open-d4rt/configs/train_effective.yaml`
- `Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml`

关键发现:

```python
if normalize_depth:
    depth = out[..., 2].abs()
    scale = masked_mean_per_sample(depth, mask).clamp_min(1e-6).unsqueeze(-1)
    out = out / scale
if transform_log:
    out = torch.sign(out) * torch.log1p(out.abs())
```

训练配置里 `xyz_3d` loss 使用:

```yaml
normalize_by_mean_depth: true
value_transform: sign_x_log1p_abs_x
```

checkpoint 旁的 `model.yaml` 只包含 model 配置，没有 loss 配置。因此本轮只验证 `raw`、`signed_log1p`、`signed_expm1` 三个输出解释 hypothesis，不把任何一个直接当作 method。

### 10.2 code changes

修改 `Stream3D/tools/diagnose_v22_d4rt_local_vs_ref.py`:

- 新增 `--xyz-transform-modes`，默认 `raw`，可传 `raw,signed_log1p,signed_expm1`。
- 新增 `_transform_xyz_hypothesis`:
  - `raw`: 原始 `xyz`。
  - `signed_log1p`: `sign(x) * log1p(abs(x))`。
  - `signed_expm1`: `sign(x) * expm1(abs(x))`，为避免爆炸对 `abs(x)` clip 到 `20`。
- 输出 CSV/JSON/MD 新增 `xyz_transform` 字段，并按 branch + transform 写表。

### 10.3 validation after patch

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_d4rt_local_vs_ref.py
```

结果:

```text
py_compile pass，无输出。
```

### 10.4 failed checkpoint path reproduction note

第一次 scene0050 命令错误使用了不存在的 checkpoint:

```text
../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/step_200000.pth
```

结果:

```text
FileNotFoundError: D4RT checkpoint does not exist: .../step_200000.pth
```

修正后使用真实存在的:

```text
../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
```

### 10.5 scene0050 transform diagnostic

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=1 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v22_d4rt_local_vs_ref \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 16 \
  --max-points-per-mask 16 \
  --min-points-per-mask 4 \
  --sampling-strategy grid_inside_mask \
  --max-source-points 1024 \
  --query-chunk-size 1024 \
  --min-visibility 0.5 \
  --min-confidence 0.5 \
  --max-anchors 8000 \
  --gt-depth-stride 12 \
  --max-gt-points-per-frame 1500 \
  --device cuda \
  --xyz-transform-modes raw,signed_log1p,signed_expm1 \
  --output-prefix outputs/audit/v22_4_xyz_transform_scene0050/xyz_transform_scene0050
```

Artifact:

- `Stream3D/outputs/audit/v22_4_xyz_transform_scene0050/xyz_transform_scene0050.{json,csv,md}`

结果:

| branch | transform | reproj med | raw d1 | linear d1 | xyz F@10 | xyz F@20 | UVZ F@10 | UVZ F@20 | UVZ linear F@10 | source UV med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xyz_ref0` | raw | 395.720598 | 0.207089 | 0.778596 | 0.000524 | 0.003142 | 0.098752 | 0.240623 | 0.303684 | 4.234780 |
| `xyz_ref0` | signed_log1p | 474.891820 | 0.290525 | 0.790593 | 0.062859 | 0.184096 | 0.107195 | 0.313788 | 0.277527 | 4.234780 |
| `xyz_ref0` | signed_expm1 | 257.255319 | 0.015951 | 0.735515 | 0.000000 | 0.000000 | 0.014155 | 0.036359 | 0.281551 | 4.234780 |
| `xyz_local` | raw | 36.824630 | 0.022706 | 0.930795 | 0.001000 | 0.035169 | 0.016902 | 0.081275 | 0.678660 | 4.234780 |
| `xyz_local` | signed_log1p | 148.769374 | 0.389531 | 0.929980 | 0.060170 | 0.331659 | 0.171066 | 0.397636 | 0.704962 | 4.234780 |
| `xyz_local` | signed_expm1 | 110.442446 | 0.000000 | 0.809789 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.310947 | 4.234780 |

Scene0050 判定:

- `xyz_local + signed_log1p` 是明显正向信号: xyz camera F@10/F@20 从 `0.001000/0.035169` 提升到 `0.060170/0.331659`，raw UVZ F@10/F@20 从 `0.016902/0.081275` 提升到 `0.171066/0.397636`。
- 但 reprojection median 从 `36.824630px` 恶化到 `148.769374px`，说明它不是一个干净的几何反变换。
- `signed_expm1` 基本为负修复。

### 10.6 probe5 transform diagnostic

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=1 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v22_d4rt_local_vs_ref \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet_v6_probe5.txt \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 16 \
  --max-points-per-mask 16 \
  --min-points-per-mask 4 \
  --sampling-strategy grid_inside_mask \
  --max-source-points 512 \
  --query-chunk-size 1024 \
  --min-visibility 0.5 \
  --min-confidence 0.5 \
  --max-anchors 8000 \
  --gt-depth-stride 12 \
  --max-gt-points-per-frame 1500 \
  --device cuda \
  --xyz-transform-modes raw,signed_log1p,signed_expm1 \
  --output-prefix outputs/audit/v22_4_xyz_transform_probe5/xyz_transform_probe5
```

Artifact:

- `Stream3D/outputs/audit/v22_4_xyz_transform_probe5/xyz_transform_probe5.{json,csv,md}`

Probe5 mean:

| branch | transform | n | reproj med | reproj p90 | raw d1 | linear d1 | xyz F@10 | xyz F@20 | outlier@20 | UVZ F@10 | UVZ F@20 | UVZ linear F@10 | source UV med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xyz_local` | raw | 5 | 30.753682 | 52.095698 | 0.197125 | 0.903552 | 0.014329 | 0.082448 | 0.837975 | 0.022713 | 0.107783 | 0.355244 | 4.379759 |
| `xyz_local` | signed_expm1 | 5 | 126.714840 | 209.529425 | 0.000308 | 0.674063 | 0.000093 | 0.001296 | 0.996692 | 0.000204 | 0.002161 | 0.152485 | 4.379759 |
| `xyz_local` | signed_log1p | 5 | 129.863558 | 179.404753 | 0.262363 | 0.884994 | 0.063919 | 0.233917 | 0.652049 | 0.080513 | 0.271661 | 0.362463 | 4.379759 |
| `xyz_ref0` | raw | 5 | 226.747204 | 438.900232 | 0.217184 | 0.819182 | 0.016254 | 0.071730 | 0.874549 | 0.040446 | 0.129410 | 0.229843 | 4.379759 |
| `xyz_ref0` | signed_expm1 | 5 | 185.013147 | 710.472078 | 0.003144 | 0.679512 | 0.000198 | 0.001237 | 0.997716 | 0.001695 | 0.008510 | 0.140930 | 4.379759 |
| `xyz_ref0` | signed_log1p | 5 | 284.717411 | 454.359420 | 0.241030 | 0.812735 | 0.043142 | 0.137393 | 0.796829 | 0.070060 | 0.253282 | 0.233035 | 4.379759 |

Probe5 `xyz_local + signed_log1p` per-scene:

| scene | reproj med | raw d1 | xyz F@10 | xyz F@20 | UVZ F@10 | UVZ F@20 |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 148.381158 | 0.382807 | 0.047929 | 0.294691 | 0.148922 | 0.388211 |
| scene0011_00 | 106.745672 | 0.513386 | 0.110171 | 0.372452 | 0.140946 | 0.415392 |
| scene0030_00 | 119.040356 | 0.108049 | 0.081834 | 0.184626 | 0.044609 | 0.174926 |
| scene0081_01 | 142.523909 | 0.049307 | 0.019721 | 0.075887 | 0.020652 | 0.142802 |
| scene0591_00 | 132.626694 | 0.258266 | 0.059939 | 0.241927 | 0.047437 | 0.236974 |

Probe5 判定:

- `xyz_local + signed_log1p` 是目前最强非 GT raw-camera diagnostic signal: xyz F@10/F@20 从 `0.014329/0.082448` 提升到 `0.063919/0.233917`，outlier@20 从 `0.837975` 降到 `0.652049`；UVZ F@10/F@20 从 `0.022713/0.107783` 提升到 `0.080513/0.271661`。
- 但它仍远低于 eval-only calibrated upper bound，且 target reprojection median 恶化到 `129.863558px`。
- 因此这是一个新的正向 blocker 线索，不是 method success。

### 10.7 visibility/confidence sweep

补跑目的: 验证 `signed_log1p` 是否还需要更强 visibility/confidence gate 才能稳定。注意: 这两个 sweep 在用户后续限制 GPU 前已经完成；用户随后先更新为仅 GPU 0 可用，当前策略为仅 GPU 0/1 可用。

命令核心参数:

```bash
# vc07
CUDA_VISIBLE_DEVICES=2 ... tools.diagnose_v22_d4rt_local_vs_ref \
  --seq-list splits/scannet_v6_probe5.txt \
  --min-visibility 0.7 --min-confidence 0.7 \
  --xyz-transform-modes raw,signed_log1p \
  --output-prefix outputs/audit/v22_4_xyz_transform_probe5_vc07/xyz_transform_probe5_vc07

# vc09
CUDA_VISIBLE_DEVICES=3 ... tools.diagnose_v22_d4rt_local_vs_ref \
  --seq-list splits/scannet_v6_probe5.txt \
  --min-visibility 0.9 --min-confidence 0.9 \
  --xyz-transform-modes raw,signed_log1p \
  --output-prefix outputs/audit/v22_4_xyz_transform_probe5_vc09/xyz_transform_probe5_vc09
```

Artifacts:

- `Stream3D/outputs/audit/v22_4_xyz_transform_probe5_vc07/xyz_transform_probe5_vc07.{json,csv,md}`
- `Stream3D/outputs/audit/v22_4_xyz_transform_probe5_vc09/xyz_transform_probe5_vc09.{json,csv,md}`

Sweep summary:

| run | branch | transform | n | anchors | reproj med | raw d1 | xyz F@10 | xyz F@20 | outlier@20 | UVZ F@10 | UVZ F@20 | UVZ linear F@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vc05 | `xyz_local` | raw | 5 | 2776.000000 | 30.753682 | 0.197125 | 0.014329 | 0.082448 | 0.837975 | 0.022713 | 0.107783 | 0.355244 |
| vc05 | `xyz_local` | signed_log1p | 5 | 2776.000000 | 129.863558 | 0.262363 | 0.063919 | 0.233917 | 0.652049 | 0.080513 | 0.271661 | 0.362463 |
| vc07 | `xyz_local` | raw | 5 | 2278.400000 | 32.105646 | 0.192814 | 0.010022 | 0.065964 | 0.859145 | 0.018374 | 0.091269 | 0.317662 |
| vc07 | `xyz_local` | signed_log1p | 5 | 2278.400000 | 134.293386 | 0.237951 | 0.046500 | 0.194865 | 0.678586 | 0.063260 | 0.242120 | 0.326364 |
| vc09 | `xyz_local` | raw | 5 | 1455.000000 | 32.081337 | 0.175393 | 0.003971 | 0.039990 | 0.894544 | 0.010101 | 0.061590 | 0.266589 |
| vc09 | `xyz_local` | signed_log1p | 5 | 1455.000000 | 136.735741 | 0.221335 | 0.040848 | 0.174127 | 0.704281 | 0.057031 | 0.224134 | 0.277090 |
| vc05 | `xyz_ref0` | raw | 5 | 2776.000000 | 226.747204 | 0.217184 | 0.016254 | 0.071730 | 0.874549 | 0.040446 | 0.129410 | 0.229843 |
| vc05 | `xyz_ref0` | signed_log1p | 5 | 2776.000000 | 284.717411 | 0.241030 | 0.043142 | 0.137393 | 0.796829 | 0.070060 | 0.253282 | 0.233035 |
| vc07 | `xyz_ref0` | raw | 5 | 2278.400000 | 219.528430 | 0.216055 | 0.012417 | 0.059534 | 0.877909 | 0.034609 | 0.112966 | 0.197598 |
| vc07 | `xyz_ref0` | signed_log1p | 5 | 2278.400000 | 279.849486 | 0.223145 | 0.037410 | 0.121676 | 0.802228 | 0.056282 | 0.224992 | 0.200071 |
| vc09 | `xyz_ref0` | raw | 5 | 1455.000000 | 214.585077 | 0.205112 | 0.007279 | 0.042252 | 0.898854 | 0.024214 | 0.087084 | 0.170810 |
| vc09 | `xyz_ref0` | signed_log1p | 5 | 1455.000000 | 278.526625 | 0.204933 | 0.034374 | 0.109467 | 0.811614 | 0.049444 | 0.205211 | 0.174038 |

Sweep 判定:

- 收紧 visibility/confidence 是负修复或不足修复。`xyz_local + signed_log1p` 的 F@20 从 vc05 `0.233917` 下降到 vc07 `0.194865`、vc09 `0.174127`。
- anchor mean 从 `2776.0` 降到 `2278.4/1455.0`，但 reprojection 没改善，geometry 也下降。
- 后续不应把简单 visibility/confidence threshold 当主 repair。

### 10.8 GPU policy note

本节前半的 scene/probe transform diagnostic 使用 GPU 1；vc07/vc09 sweep 在用户更新 GPU 限制前已经完成，分别使用 GPU 2/3。用户随后更新策略为 GPU 0 可用、其他暂时不用；最后更新为 GPU 0/1 可用、其他暂时不用。后续所有 GPU 命令只使用 `CUDA_VISIBLE_DEVICES=0` 或 `CUDA_VISIBLE_DEVICES=1`。

### 10.9 v22.4 conclusion

- OpenD4RT loss-space `signed_log1p` 不是无关细节，它是目前第一个能明显改善非 GT raw camera geometry 的信号。
- 最佳非 GT signal 仍是 `xyz_local + signed_log1p`: probe5 xyz camera F@10/F@20 `0.063919/0.233917`，UVZ F@10/F@20 `0.080513/0.271661`。
- 但它仍不满足进入 method/provider AP 主线的 gate: reprojection median 恶化到 `129.863558px`，F@10 仍远低于 eval-only calibrated upper bound，且没有形成稳定 canonical/provider support。
- `signed_expm1` 基本失败；简单 visibility/confidence filtering 也失败。
- Stop D 继续，不启动 Phase F/method table。下一步可考虑把 `xyz_local + signed_log1p` 作为 diagnostic branch 接入 direct/provider runner，但必须继续保持 diagnostic-only，不能报告 method。

### 10.10 final validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_d4rt_local_vs_ref.py
```

结果:

```text
py_compile pass，无输出。
```

第一次 unittest 命令:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
FAILED (failures=1)
test_gpu_visibility_includes_requested_devices_when_set
AssertionError: False is not true
Ran 63 tests in 1.933s
```

解释: 这是已有 optional-GPU 测试的环境假设，它只接受 `CUDA_VISIBLE_DEVICES` 为空或包含 `6/7`；本次为了遵守用户 GPU 限制设置成 `0`，触发了该测试断言。随后用隐藏 GPU 的 CPU-only 环境重跑。

重跑命令:

```bash
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 63 tests in 1.697s
OK (skipped=1)
```

## 11. v22.5 `xyz_local + signed_log1p` direct reconstruction branch

目的: 将 v22.4 找到的 `xyz_local + signed_log1p` 正向信号接入 v22 direct reconstruction benchmark，确认它在固定 direct runner 指标体系下的表现。该实验只读取 v22.3 已落盘 cache，不重新调用 D4RT checkpoint，不使用 GT 做 prediction，也不生成 method table。

### 11.1 code changes

修改 `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`:

- `VariantSpec` 新增:
  - `xyz_field`
  - `xyz_transform`
- 新增 `_transform_xyz_hypothesis`:
  - `raw`
  - `signed_log1p`
  - `signed_expm1`
- 新增 `_apply_variant_xyz_to_windows`，从 `carriers_window*.npz` 选择 `xyz_ref` 或 `xyz_local` 并按 hypothesis transform 替换 window xyz。
- 新增 R16-R19:
  - R16: `xyz_local raw`
  - R17: `xyz_local signed_log1p`
  - R18: `xyz_local raw UV+Z camera backprojection`
  - R19: `xyz_local signed_log1p UV+Z camera backprojection`

新增测试:

- `Stream3D/tests/test_v22_direct_reconstruction.py`
  - 覆盖 `signed_log1p` 数学形式。
  - 覆盖从 NPZ 读取 `xyz_local` 并替换 window xyz。

### 11.2 compile/test before run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/run_v22_direct_reconstruction_benchmark.py \
  tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass，无输出。

Ran 65 tests in 1.602s
OK (skipped=1)
```

### 11.3 probe5 R16-R19 direct benchmark

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --variants R16,R17,R18,R19 \
  --audit-root outputs/audit/v22_5_direct_xyz_local_transform_probe5 \
  --debug-progress
```

说明: 该 runner 只读 cache，不需要 GPU；这里用 `CUDA_VISIBLE_DEVICES=""` 隐藏 GPU，避免占用用户保留的 GPU 0/1。

Artifacts:

- `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_5_direct_xyz_local_transform_probe5/direct_reconstruction_f10_by_variant.png`

Probe5 mean:

| variant | label | point mode | xyz transform | world F@10 | camera F@10 | camera F@20 | camera outlier@20 | reproj med | raw d1 | covered GT inst |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| R16 | D4RT xyz_local raw | xyz | raw | 0.000000 | 0.014329 | 0.082525 | 0.836184 | 30.753682 | 0.197125 | 0.937831 |
| R17 | D4RT xyz_local signed-log1p | xyz | signed_log1p | 0.000000 | 0.064293 | 0.234276 | 0.650592 | 129.863558 | 0.262363 | 0.937831 |
| R18 | D4RT xyz_local UV+Z camera backprojection | uvz_camera | raw | 0.000000 | 0.022764 | 0.107910 | 0.799749 | 30.753682 | 0.197125 | 0.937831 |
| R19 | D4RT xyz_local signed-log1p UV+Z camera backprojection | uvz_camera | signed_log1p | 0.000000 | 0.080797 | 0.272468 | 0.535702 | 129.863558 | 0.262363 | 0.937831 |

R17/R19 per-scene:

| variant | scene | point mode | camera F@10 | camera F@20 | camera outlier@20 | reproj med | raw d1 |
|---|---|---|---:|---:|---:|---:|---:|
| R17 | scene0050_00 | xyz | 0.049094 | 0.295915 | 0.639142 | 148.381158 | 0.382807 |
| R17 | scene0011_00 | xyz | 0.109763 | 0.372371 | 0.515004 | 106.745672 | 0.513386 |
| R17 | scene0030_00 | xyz | 0.082547 | 0.185224 | 0.719965 | 119.040356 | 0.108049 |
| R17 | scene0081_01 | xyz | 0.020025 | 0.076008 | 0.857713 | 142.523909 | 0.049307 |
| R17 | scene0591_00 | xyz | 0.060038 | 0.241864 | 0.521135 | 132.626694 | 0.258266 |
| R19 | scene0050_00 | uvz_camera | 0.149372 | 0.389861 | 0.357096 | 148.381158 | 0.382807 |
| R19 | scene0011_00 | uvz_camera | 0.141079 | 0.416296 | 0.400661 | 106.745672 | 0.513386 |
| R19 | scene0030_00 | uvz_camera | 0.045075 | 0.175641 | 0.712989 | 119.040356 | 0.108049 |
| R19 | scene0081_01 | uvz_camera | 0.020589 | 0.143235 | 0.679641 | 142.523909 | 0.049307 |
| R19 | scene0591_00 | uvz_camera | 0.047872 | 0.237309 | 0.528120 | 132.626694 | 0.258266 |

### 11.4 v22.5 conclusion

- Direct runner 复现并强化了 v22.4 结论: `signed_log1p` 是正向 camera-space signal。R17 相比 R16 把 camera F@10/F@20 从 `0.014329/0.082525` 提到 `0.064293/0.234276`。
- `UV+Z + signed_log1p` 更强: R19 camera F@10/F@20 到 `0.080797/0.272468`，camera outlier@20 降到 `0.535702`。
- 但 world F@10 仍为 `0.0`，说明没有非 GT world/canonical anchor 时，该 branch 还不能形成 method geometry。
- `raw_uvz_reproj_error_px_median=129.863558` 仍高，说明 signed-log1p 改善 metric proximity 的同时破坏了 raw UV/XYZ reprojection consistency。
- 因此 Stop D 继续；下一步若继续，应尝试 D4RT-native scale/canonical anchor 或 eval-only ablation 来定位缺失的世界对齐因素，但不能启动 Phase F/method table。

## 12. v22.6 `xyz_local` eval-Sim3 upper-bound diagnostic

目的: v22.5 证明 `xyz_local + signed_log1p` 有 camera-space 正信号，但 raw world F@10 全为 `0.0`。本轮新增 eval-only scene Sim3 诊断上界，定位问题是“缺非 GT world/canonical anchor”还是“`xyz_local` 本身即使 GT 对齐也弱”。该实验只读取 v22.3 cache，不重新调用 D4RT checkpoint；R20/R21 使用 ScanNet depth/pose 拟合 scene Sim3，明确是 diagnostic-only / forbidden-for-method-table。

用户 GPU 限制: 只可用 GPU 0/1，其它 GPU 暂时不可用。本轮 benchmark 和测试都使用 `CUDA_VISIBLE_DEVICES=""` CPU-only 隐藏 GPU，未占用其它卡。

### 12.1 code changes

修改 `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`:

- 新增 R20: `D4RT xyz_local eval-only scene Sim3`
- 新增 R21: `D4RT xyz_local signed-log1p eval-only scene Sim3`

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 R20/R21 variant registry 测试，确认二者都为 `eval_sim3` 且读取 `xyz_local`。

### 12.2 compile/unit test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/run_v22_direct_reconstruction_benchmark.py \
  tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
py_compile pass，无输出。

Ran 3 tests in 0.003s
OK
```

### 12.3 probe5 R20/R21 direct benchmark

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --variants R20,R21 \
  --audit-root outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5 \
  --debug-progress
```

Artifacts:

- `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_6_eval_sim3_xyz_local_transform_probe5/direct_reconstruction_f10_by_variant.png`

Probe5 mean:

| variant | label | transform | world F@10 | world F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | reproj med |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R4 old | D4RT eval-only scene Sim3 | old xyz_ref | 0.613669 | 0.868878 | 0.856899 | 0.887500 | 0.112500 | NA | NA | 0.971796 | NA |
| R20 | D4RT xyz_local eval-only scene Sim3 | raw | 0.408784 | 0.679166 | 0.650008 | 0.729472 | 0.270528 | 0.118600 | 0.360883 | 0.788298 | 30.753682 |
| R21 | D4RT xyz_local signed-log1p eval-only scene Sim3 | signed_log1p | 0.281703 | 0.533511 | 0.447521 | 0.711899 | 0.288101 | 0.083132 | 0.279111 | 0.678162 | 129.863558 |

R20/R21 per-scene:

| variant | scene | F@10 | F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | reproj med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R20 | scene0050_00 | 0.348323 | 0.646027 | 0.515179 | 0.865972 | 0.134028 | 0.096239 | 0.356310 | 0.741122 | 36.882055 |
| R20 | scene0011_00 | 0.466501 | 0.763437 | 0.779927 | 0.747630 | 0.252370 | 0.077049 | 0.341575 | 0.848819 | 19.019359 |
| R20 | scene0030_00 | 0.555682 | 0.817010 | 0.813265 | 0.820790 | 0.179210 | 0.237734 | 0.605065 | 0.913605 | 26.833669 |
| R20 | scene0081_01 | 0.320899 | 0.570688 | 0.599693 | 0.544359 | 0.455641 | 0.057141 | 0.197598 | 0.556240 | 32.554647 |
| R20 | scene0591_00 | 0.352517 | 0.598668 | 0.541974 | 0.668608 | 0.331392 | 0.124838 | 0.303866 | 0.881705 | 38.478680 |
| R21 | scene0050_00 | 0.221050 | 0.511257 | 0.357854 | 0.894861 | 0.105139 | 0.093335 | 0.304126 | 0.596620 | 148.381158 |
| R21 | scene0011_00 | 0.316169 | 0.620123 | 0.557964 | 0.697867 | 0.302133 | 0.062856 | 0.259059 | 0.714961 | 106.745672 |
| R21 | scene0030_00 | 0.351252 | 0.576722 | 0.485485 | 0.710187 | 0.289813 | 0.128640 | 0.386270 | 0.808399 | 119.040356 |
| R21 | scene0081_01 | 0.287594 | 0.580192 | 0.571393 | 0.589266 | 0.410734 | 0.054087 | 0.215541 | 0.525424 | 142.523909 |
| R21 | scene0591_00 | 0.232450 | 0.379260 | 0.264909 | 0.667314 | 0.332686 | 0.076742 | 0.230561 | 0.745408 | 132.626694 |

### 12.4 final test

命令:

```bash
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 66 tests in 1.758s
OK (skipped=1)
```

### 12.5 v22.6 conclusion

- `xyz_local raw` 在 eval-only scene Sim3 下可形成中等质量 world 点云: R20 F@10/F@20 为 `0.408784/0.679166`，comp@20 `0.650008`。这证明 `xyz_local` 不是完全无结构，raw world F@10 为 `0.0` 的大头确实来自缺少非 GT world/canonical anchoring。
- 但 R20 仍明显低于旧 R4 `xyz_ref` eval-Sim3 上界: R4 F@10/F@20/comp@20 为 `0.613669/0.868878/0.856899`。因此 `pred_ref`/canonical 分支仍携带更强的可对齐几何；不能简单把主线改成 `xyz_local` 后推进 provider/AP。
- R21 说明 `signed_log1p` 的正向作用主要在 raw camera-space，不适合直接再做 scene Sim3 上界: R21 F@10/F@20 降到 `0.281703/0.533511`，低于 R20。
- 结论更新: 下一步不应只追 `signed_log1p`；应重点恢复非 GT canonical/world anchor，尤其要理解 `pred_ref` 如何从 ref0/canonical 转成 ScanNet world，同时用 `xyz_local` 作为 UV/target-frame consistency 诊断辅助。Stop D 继续，不启动 Phase F/method table。

## 13. v22.7 `xyz_ref0` ref0-pose / scale diagnostic

目的: v22.6 说明 `xyz_ref`/canonical 分支仍有更高 eval-aligned 上界，但 raw world 完全失败。本轮按 OpenD4RT worldtrack/demo 代码线索新增两个 diagnostic:

- R22: 将 `xyz_ref0` 直接通过窗口第 0 帧 ScanNet pose 映射到 world。
- R23: 固定 ref0 pose，只用 ScanNet depth/pose anchors 拟合一个 eval-only scale，再映射到 world。

这两个 row 都使用 ScanNet pose；R23 还使用 ScanNet depth/pose anchors 拟合 scale，因此明确是 diagnostic-only / forbidden-for-method-table。用户 GPU 限制为只可用 GPU 0/1，本轮 benchmark 和测试均为 CPU-only，使用 `CUDA_VISIBLE_DEVICES=""`，没有占用任何 GPU。

### 13.1 code changes

修改 `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`:

- 新增 R22: `D4RT xyz_ref0 + ScanNet ref0 pose`
- 新增 R23: `D4RT xyz_ref0 + ScanNet ref0 pose + eval-only scale`
- 新增 `_fit_ref0_pose_scale`，把 GT world anchors 通过 ref0 pose 反变换到 ref0 camera 坐标，只求一个正 scale。
- 新增 `_spec_outputs_world`，让 R22/R23 在 depth/camera-space metric 中按 world-space 输出处理。
- scene row 新增 `ref0_pose_*` / `ref0_pose_scale_*` 诊断字段。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 R22/R23 registry 测试，并确认二者被识别为 world-space output。

### 13.2 compile/unit test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/run_v22_direct_reconstruction_benchmark.py \
  tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
py_compile pass，无输出。

Ran 4 tests in 0.002s
OK
```

### 13.3 probe5 R22/R23 direct benchmark

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --variants R22,R23 \
  --audit-root outputs/audit/v22_7_ref0_pose_scale_probe5 \
  --debug-progress
```

Artifacts:

- `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_7_ref0_pose_scale_probe5/direct_reconstruction_f10_by_variant.png`

Probe5 mean comparison:

| variant | label | world F@10 | world F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | ref0 scale | scale residual med | scale residual p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | D4RT single-chunk raw | 0.005726 | 0.016965 | 0.028761 | 0.012031 | 0.987969 | NA | NA | 0.205612 | 4.346847 | 6.385360 | NA | NA | NA |
| R4 | D4RT eval-only scene Sim3 | 0.613669 | 0.868878 | 0.856899 | 0.887500 | 0.112500 | NA | NA | 0.971796 | 0.074398 | 0.321411 | NA | NA | NA |
| R22 | xyz_ref0 + ScanNet ref0 pose | 0.108860 | 0.280340 | 0.254315 | 0.327160 | 0.672840 | 0.036096 | 0.143050 | 0.111773 | 0.530517 | 1.261981 | NA | NA | NA |
| R23 | xyz_ref0 + ScanNet ref0 pose + eval-only scale | 0.622259 | 0.902864 | 0.888330 | 0.920589 | 0.079411 | 0.492137 | 0.809117 | 0.966551 | 0.063193 | 0.176206 | 0.744080 | 0.159111 | 0.357298 |

R22/R23 per-scene:

| variant | scene | F@10 | F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | ref0 scale | scale residual med | scale residual p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R22 | scene0050_00 | 0.047368 | 0.113140 | 0.098933 | 0.132111 | 0.867889 | 0.026468 | 0.081354 | 0.003399 | 0.824056 | 1.422727 | NA | NA | NA |
| R22 | scene0011_00 | 0.058508 | 0.168045 | 0.182245 | 0.155897 | 0.844103 | 0.020139 | 0.093082 | 0.018418 | 1.137038 | 1.903489 | NA | NA | NA |
| R22 | scene0030_00 | 0.082610 | 0.303141 | 0.257733 | 0.367970 | 0.632030 | 0.037876 | 0.197605 | 0.058940 | 0.313623 | 1.343151 | NA | NA | NA |
| R22 | scene0081_01 | 0.158993 | 0.416243 | 0.423340 | 0.409381 | 0.590619 | 0.061865 | 0.217120 | 0.469498 | 0.218331 | 0.440812 | NA | NA | NA |
| R22 | scene0591_00 | 0.196823 | 0.401132 | 0.309324 | 0.570439 | 0.429561 | 0.034130 | 0.126089 | 0.008608 | 0.159536 | 1.199727 | NA | NA | NA |
| R23 | scene0050_00 | 0.600588 | 0.965901 | 0.941033 | 0.992119 | 0.007881 | 0.500525 | 0.936842 | 0.975650 | 0.063385 | 0.138309 | 0.618160 | 0.200635 | 0.316476 |
| R23 | scene0011_00 | 0.751893 | 0.968949 | 0.976211 | 0.961795 | 0.038205 | 0.519124 | 0.869600 | 0.978951 | 0.040801 | 0.117973 | 0.648489 | 0.085059 | 0.181278 |
| R23 | scene0030_00 | 0.773478 | 0.939515 | 0.898917 | 0.983953 | 0.016047 | 0.693516 | 0.913275 | 0.960954 | 0.034564 | 0.111192 | 0.671717 | 0.123179 | 0.272399 |
| R23 | scene0081_01 | 0.197514 | 0.721343 | 0.759940 | 0.686477 | 0.313523 | 0.082684 | 0.481774 | 0.929328 | 0.158568 | 0.439288 | 1.086632 | 0.293151 | 0.802004 |
| R23 | scene0591_00 | 0.787823 | 0.918611 | 0.865551 | 0.978601 | 0.021399 | 0.664836 | 0.844096 | 0.987873 | 0.018648 | 0.074266 | 0.695402 | 0.093529 | 0.214334 |

### 13.4 v22.7 conclusion

- R22 只用 ref0 pose 仍明显失败: world F@10/F@20 只有 `0.108860/0.280340`，raw depth delta1 只有 `0.111773`。这说明 raw `xyz_ref0` 的主要问题不是单纯“缺 pose”，还缺尺度。
- R23 固定 ref0 pose 后只拟合 eval-only scale，world F@10/F@20/comp@20 达到 `0.622259/0.902864/0.888330`，略高于旧 R4 full eval-Sim3 的 `0.613669/0.868878/0.856899`。这强烈说明 `xyz_ref` 的坐标系更接近 OpenD4RT ref0 convention，而不是任意 canonical；合适的 ref0 pose + scale 比自由 Sim3 更稳定。
- R23 camera-space 也明显变好: camera F@10/F@20 为 `0.492137/0.809117`，raw depth delta1 为 `0.966551`。这说明 ref0-pose+scale 对齐不是只在 world Chamfer 上碰巧有效。
- scale 不是常数: per-scene ref0 scale 从 `0.618160` 到 `1.086632`，scene0081 residual p90 `0.802004` 明显偏高。后续 method 修复不能硬编码一个全局 scale，而要找非 GT per-scene/per-window scale anchor。
- 该结果是很强的 upper-bound 证据，不是 method success。R22/R23 使用 ScanNet pose，R23 还使用 ScanNet depth/pose anchors 拟合 scale，因此 Stop D 继续，不启动 Phase F/method table。下一步应做非 GT ref0 pose/scale anchor: 用 D4RT ref0 convention、source/target consistency、OpenD4RT scale normalization 或 monocular/camera trajectory scale clue 逼近 R23。

### 13.5 final test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 67 tests in 1.725s
OK (skipped=1)
```

## 14. v22.8 non-GT local/ref scale proxy diagnostic

目的: v22.7 证明 `xyz_ref0 + ScanNet ref0 pose + eval-only scale` 是强 upper-bound，但 R23 的 scale 来自 ScanNet depth/pose anchors，不能进入 method。本轮尝试不用 ScanNet depth 拟合 scale，只从 D4RT cache 内部的 `xyz_local` / `xyz_ref` 统计估计 scale，再接 ScanNet ref0 pose 做 direct reconstruction diagnostic。

边界:

- R24/R25/R26 都仍使用 ScanNet ref0 pose，因此仍是 diagnostic-only / forbidden-for-method-table。
- R24/R25/R26 的 scale 不使用 ScanNet depth/pose anchors，只使用 D4RT 输出的 `xyz_local` / `xyz_ref` 统计。
- 用户当前 GPU 限制为只可用 GPU 0/1；本轮 benchmark 和测试均为 CPU-only，使用 `CUDA_VISIBLE_DEVICES=""`，未占用 GPU。

### 14.1 code changes

修改 `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`:

- 新增 R24: `D4RT xyz_ref0 + ref0 pose + local/ref median-norm scale`。
- 新增 R25: `D4RT xyz_ref0 + ref0 pose + local/ref RMS-norm scale`。
- 新增 R26: `D4RT xyz_ref0 + ref0 pose + source-frame z scale`。
- 新增 `_estimate_ref0_local_scale`:
  - `local_median_norm`: `median(||xyz_local||) / median(||xyz_ref||)`。
  - `local_rms_norm`: RMS norm ratio。
  - `source_z`: 同 source frame carrier 上 `abs(z_local) / abs(z_ref)` 的 median ratio。
- R24/R25/R26 使用 `outputs/stream4d_debug_v22_local_xyz_probe5_r1`，因为该 cache 同时有 `xyz_local` 和 `xyz_ref`。
- `_spec_outputs_world` 扩展到所有 `ref0_pose*` variants。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 R24/R25/R26 registry/world-output 单测。
- 新增 synthetic local/ref scale estimator 单测，确认三个 scale proxy 在已知 `xyz_local = 2 * xyz_ref` 时返回 `2.0`。

### 14.2 compile/unit test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/run_v22_direct_reconstruction_benchmark.py \
  tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
py_compile pass，无输出。

Ran 6 tests in 0.004s
OK
```

### 14.3 probe5 R24/R25/R26 direct benchmark

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --variants R24,R25,R26 \
  --audit-root outputs/audit/v22_8_ref0_local_scale_probe5 \
  --debug-progress
```

Artifacts:

- `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_8_ref0_local_scale_probe5/direct_reconstruction_f10_by_variant.png`

Probe5 mean:

| variant | scale proxy | world F@10 | world F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | estimated scale | anchors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R23 upper-bound | GT depth/pose scale | 0.622259 | 0.902864 | 0.888330 | 0.920589 | 0.079411 | 0.492137 | 0.809117 | 0.966551 | 0.063193 | 0.176206 | 0.744080 | NA |
| R24 | local/ref median-norm | 0.051524 | 0.186488 | 0.149141 | 0.283856 | 0.716144 | 0.025446 | 0.117937 | 0.197789 | 0.594960 | 1.484469 | 0.977360 | 2788.0 |
| R25 | local/ref RMS-norm | 0.056787 | 0.185555 | 0.149540 | 0.282856 | 0.717144 | 0.030268 | 0.116029 | 0.203527 | 0.594858 | 1.458447 | 0.976609 | 2788.0 |
| R26 | source-frame z | 0.027983 | 0.116290 | 0.081328 | 0.208961 | 0.791039 | 0.008791 | 0.052977 | 0.168462 | 0.754244 | 1.672021 | 1.057880 | 505.4 |

R24/R25/R26 per-scene estimated scale:

| scene | R24 scale | R24 F@10/F@20 | R25 scale | R25 F@10/F@20 | R26 scale | R26 F@10/F@20 |
|---|---:|---|---:|---|---:|---|
| scene0050_00 | 0.916684 | 0.010952 / 0.089711 | 0.947803 | 0.009401 / 0.064824 | 1.162987 | 0.000000 / 0.000000 |
| scene0011_00 | 1.053518 | 0.020654 / 0.076539 | 1.021793 | 0.020219 / 0.080914 | 1.017139 | 0.019581 / 0.082416 |
| scene0030_00 | 1.017340 | 0.010479 / 0.053080 | 1.017801 | 0.010556 / 0.053020 | 1.036978 | 0.010902 / 0.052890 |
| scene0081_01 | 0.867741 | 0.196239 / 0.583838 | 0.882035 | 0.220488 / 0.597902 | 1.072823 | 0.084079 / 0.312270 |
| scene0591_00 | 1.031516 | 0.019296 / 0.129270 | 1.013613 | 0.023269 / 0.131114 | 0.999472 | 0.025352 / 0.133875 |

### 14.4 v22.8 conclusion

- R24/R25/R26 都没有逼近 R23 upper-bound。最好的 R25 world F@10/F@20 只有 `0.056787/0.185555`，远低于 R23 的 `0.622259/0.902864`，也低于 R22 的 F@20 `0.280340`。
- 简单 D4RT-internal norm/z scale proxy 基本估出接近 `1.0` 的 scale: R24/R25 mean `0.977360/0.976609`，R26 mean `1.057880`。这与 R23 eval-only mean scale `0.744080` 和 per-scene范围 `0.618160..1.086632` 不匹配。
- scene0081 在 R24/R25 上仍相对最好，但这不是整体解决；R24/R25 其它 4 个 scene 的 F@10 都在 `0.009401..0.023269` 附近。
- 结论: 非 GT scale anchor 不能用简单 `xyz_local` / `xyz_ref` norm ratio 或 source-frame z ratio 近似。下一步需要更强的 D4RT-native scale clue，例如模型 normalization 反变换、ref0/camera trajectory consistency、source-target reprojection+depth consistency、或 object/mask scale prior。Stop D 继续，不启动 provider/AP 或 method table。

### 14.5 final test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 69 tests in 2.443s
OK (skipped=1)
```

## 15. v22.9 pose-trajectory scale diagnostic

目的: v22.8 证明简单 `xyz_local` / `xyz_ref` norm/z ratio 不能近似 R23 的 ref0 metric scale。本轮尝试更强的 diagnostic 线索: 对每个 target frame，用 D4RT 内部 `xyz_ref -> xyz_local` carrier 对拟合无尺度 rigid transform，取其 translation norm；再用 ScanNet pose 中 ref0 到 target 的 camera-center baseline 除以该 D4RT translation norm，得到 trajectory scale。该 scale 不使用 ScanNet depth，但使用 ScanNet pose trajectory，因此仍是 diagnostic-only / forbidden-for-method-table。

边界:

- R27 使用 ScanNet ref0 pose 和 target-frame ScanNet poses。
- R27 不使用 ScanNet depth anchors 拟合 scale。
- R27 不是 method result；它只验证 pose trajectory baseline 是否能提供比 v22.8 更强的 scale clue。
- 本轮 CPU-only 运行，使用 `CUDA_VISIBLE_DEVICES=""`，未占用 GPU。

### 15.1 code changes

修改 `Stream3D/tools/run_v22_direct_reconstruction_benchmark.py`:

- 新增 R27: `D4RT xyz_ref0 + ref0 pose + pose-trajectory scale`。
- 新增 `_fit_rigid_no_scale`，对 `xyz_ref -> xyz_local` carrier 对拟合 SE3 no-scale transform。
- 新增 `_estimate_ref0_pose_trajectory_scale`:
  - 每个 target frame 计算 D4RT rigid translation norm。
  - 每个 target frame 计算 ScanNet pose baseline norm。
  - 取 `pose_baseline / d4rt_translation` 的 median 作为 ref0 scale。
  - 记录 frame count、ratio mean/std/min/max、D4RT translation median、pose translation median、rigid residual p90 median。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- R27 registry/world-output 单测。
- synthetic trajectory-scale 单测，构造 D4RT translation `1/2` 与 pose baseline `2/4`，确认 scale 为 `2.0`。

### 15.2 compile/unit test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
  tools/run_v22_direct_reconstruction_benchmark.py \
  tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
Ran 7 tests in 0.007s
OK
```

### 15.3 probe5 R27 direct benchmark

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.run_v22_direct_reconstruction_benchmark \
  --seq-list splits/scannet_v6_probe5.txt \
  --variants R27 \
  --audit-root outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5 \
  --debug-progress
```

Artifacts:

- `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_summary.*`
- `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_scene_rows.*`
- `Stream3D/outputs/audit/v22_9_ref0_pose_trajectory_scale_probe5/direct_reconstruction_f10_by_variant.png`

Probe5 mean comparison:

| variant | scale source | world F@10 | world F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | scale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R22 | ref0 pose only | 0.108860 | 0.280340 | 0.254315 | 0.327160 | 0.672840 | 0.036096 | 0.143050 | 0.111773 | 0.530517 | 1.261981 | NA |
| R23 | eval-only depth/pose scale | 0.622259 | 0.902864 | 0.888330 | 0.920589 | 0.079411 | 0.492137 | 0.809117 | 0.966551 | 0.063193 | 0.176206 | 0.744080 |
| R25 | local/ref RMS-norm scale | 0.056787 | 0.185555 | 0.149540 | 0.282856 | 0.717144 | 0.030268 | 0.116029 | 0.203527 | 0.594858 | 1.458447 | 0.976609 |
| R27 | pose trajectory scale | 0.218665 | 0.556108 | 0.474452 | 0.701966 | 0.298034 | 0.122417 | 0.415634 | 0.650079 | 0.189421 | 0.517978 | 0.693379 |

R27 per-scene:

| scene | F@10 | F@20 | comp@20 | precision@20 | outlier@20 | camera F@10 | camera F@20 | raw d1 | pred->GT med | pred->GT p90 | traj scale | scale mean/std | frames | D4RT trans med | pose trans med | rigid p90 med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| scene0050_00 | 0.175587 | 0.670721 | 0.524977 | 0.928487 | 0.071513 | 0.134783 | 0.607808 | 0.852804 | 0.137554 | 0.192348 | 0.493855 | 0.467927 / 0.087287 | 15 | 0.426625 | 0.217465 | 0.047783 |
| scene0011_00 | 0.550465 | 0.862890 | 0.834426 | 0.893365 | 0.106635 | 0.266822 | 0.642778 | 0.900787 | 0.055773 | 0.211131 | 0.572213 | 0.695970 / 0.226055 | 15 | 0.546049 | 0.371217 | 0.061000 |
| scene0030_00 | 0.017300 | 0.129462 | 0.092278 | 0.216840 | 0.783160 | 0.012128 | 0.086723 | 0.118110 | 0.492710 | 1.223101 | 0.926332 | 0.946675 / 0.269850 | 15 | 0.236630 | 0.192321 | 0.029415 |
| scene0081_01 | 0.234320 | 0.605460 | 0.540027 | 0.688938 | 0.311062 | 0.139921 | 0.416576 | 0.839753 | 0.102437 | 0.661338 | 0.889791 | 0.798835 / 0.220658 | 15 | 0.724507 | 0.585620 | 0.066886 |
| scene0591_00 | 0.115653 | 0.512004 | 0.380550 | 0.782201 | 0.217799 | 0.058430 | 0.324285 | 0.538942 | 0.158632 | 0.301971 | 0.584703 | 0.596980 / 0.154907 | 15 | 0.167304 | 0.089479 | 0.039773 |

### 15.4 v22.9 conclusion

- R27 是本轮第一个不用 ScanNet depth scale fitting、明显改善 ref0 direct reconstruction 的 diagnostic: world F@10/F@20 从 R22 `0.108860/0.280340` 提到 `0.218665/0.556108`，raw depth delta1 从 `0.111773` 提到 `0.650079`。
- R27 明显强于 v22.8 的简单 local/ref scale proxy: R25 F@10/F@20 只有 `0.056787/0.185555`。
- 但 R27 仍远低于 R23 upper-bound `0.622259/0.902864`，且使用 ScanNet pose trajectory，不能作为 method。
- R27 的主要负例是 scene0030: trajectory scale `0.926332`，明显高于 R23 scale `0.671717`，F@10/F@20 只有 `0.017300/0.129462`。这说明 D4RT ref/local rigid translation baseline 并不总是稳定的 metric scale clue。
- 正向结论: camera/pose trajectory consistency 是比 norm/z proxy 更强的 scale 线索，后续可考虑用非 GT camera trajectory / VO / model-internal motion scale 逼近它。
- 负向结论: 仅凭 D4RT ref/local rigid translation + pose baseline 仍不够稳定，不能启动 provider/AP 或 method table。Stop D 继续。

### 15.5 final test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 70 tests in 2.456s
OK (skipped=1)
```

## 16. v22.10 ref0 trajectory consistency diagnostic

目的: v22.9 R27 的 scene0030 负例很明显。R27 只报告了 trajectory scale 的 aggregate，无法判断失败来自 transform 方向、rotation、translation direction，还是 D4RT translation magnitude 漂移。本轮新增 per-frame diagnostic，不新增方法 row。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮诊断和测试都是 CPU-only，显式使用 `CUDA_VISIBLE_DEVICES=""`，未占用 GPU。

### 16.1 code changes

新增 `Stream3D/tools/diagnose_v22_ref0_trajectory_scale.py`:

- 从 `outputs/stream4d_debug_v22_local_xyz_probe5_r1` 读取 `xyz_ref` / `xyz_local` carrier cache。
- 对每个 target frame 拟合 D4RT `xyz_ref -> xyz_local` no-scale rigid transform。
- 与 ScanNet relative pose 的 ref-to-target rotation / translation direction 比较。
- 每帧记录:
  - `trajectory_scale_ratio = pose_translation_norm / d4rt_translation_norm`
  - `eval_ref0_depth_scale`，即 R23 eval-only ref0 scale，用作 diagnostic reference
  - ratio absrel vs R23 eval scale
  - rotation error、translation-direction error、rigid residual
- 输出 frame/window/scene/candidate-scale 四类 CSV/JSON 和一个 Markdown summary。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 rotation/translation direction helper 单测。

### 16.2 targeted validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_ref0_trajectory_scale.py
```

结果:

```text
Ran 8 tests
OK
py_compile pass
```

备注: 初始用 base conda `/mnt/data/users/chengshun.wang/miniconda3/bin/python` 跑 full discover 时失败，因为 base conda 缺 `torch`；这是环境问题。后续使用 canonical `envs/loger` 环境，完整测试通过。

### 16.3 probe5 diagnostic run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_ref0_trajectory_scale.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1 \
  --audit-root outputs/audit/v22_10_ref0_trajectory_consistency_probe5 \
  --max-anchors 8000 \
  --robust-trim-percentile 90
```

结果:

```text
Wrote v22.10 ref0 trajectory diagnostic to outputs/audit/v22_10_ref0_trajectory_consistency_probe5
```

Artifacts:

- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_frame_rows.*`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_window_rows.*`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_scene_summary.*`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_candidate_scale_errors.*`
- `Stream3D/outputs/audit/v22_10_ref0_trajectory_consistency_probe5/ref0_trajectory_consistency.md`

Probe5 scene summary:

| scene | R23 eval scale | median ratio | q25 | q75 | median absrel | rot err med | trans dir err med | rigid p90 med | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.561127 | 0.493855 | 0.410124 | 0.552539 | 0.119887 | 1.717264 | 10.175581 | 0.047783 | 15 |
| scene0011_00 | 0.561406 | 0.572213 | 0.523307 | 0.847068 | 0.019250 | 2.363984 | 8.807973 | 0.061000 | 15 |
| scene0030_00 | 0.656649 | 0.926332 | 0.771188 | 1.042536 | 0.410697 | 1.930591 | 15.677520 | 0.029415 | 15 |
| scene0081_01 | 0.862760 | 0.889791 | 0.598418 | 0.932503 | 0.031330 | 2.315460 | 11.082123 | 0.066886 | 15 |
| scene0591_00 | 0.707645 | 0.584703 | 0.508000 | 0.623795 | 0.173734 | 1.902544 | 10.212947 | 0.039773 | 15 |

Candidate scale error vs R23 eval-only scale:

| candidate | mean absrel | median absrel | max absrel |
|---|---:|---:|---:|
| ratio_median | 0.150980 | 0.119887 | 0.410697 |
| ratio_residual_weighted_median | 0.169472 | 0.173734 | 0.368032 |
| ratio_low_residual_median | 0.183642 | 0.151174 | 0.306391 |
| ratio_low_direction_median | 0.191472 | 0.141591 | 0.554731 |
| ratio_mean | 0.215588 | 0.166095 | 0.441677 |
| ratio_q25 | 0.219984 | 0.269107 | 0.306391 |
| ratio_q75 | 0.262226 | 0.118492 | 0.587662 |
| ratio_q10 | 0.279964 | 0.383812 | 0.457045 |
| ratio_min | 0.334162 | 0.445036 | 0.492834 |

scene0030 frame-level highlights:

| frame | ratio | absrel vs R23 scale | rot err | trans dir err | rigid p90 | D4RT trans | pose trans |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.607577 | 0.074730 | 0.132418 | 85.049825 | 0.026309 | 0.022755 | 0.013825 |
| 60 | 0.665212 | 0.013042 | 0.674470 | 8.785631 | 0.020088 | 0.289112 | 0.192321 |
| 80 | 0.926332 | 0.410697 | 1.475769 | 15.677520 | 0.023233 | 0.176657 | 0.163643 |
| 120 | 1.716449 | 1.613954 | 4.772149 | 12.395835 | 0.035809 | 0.236630 | 0.406164 |
| 150 | 0.776038 | 0.181816 | 6.395605 | 25.863996 | 0.047475 | 0.272615 | 0.211559 |

### 16.4 conclusion

- v22.10 排除了一个简单实现错误: D4RT ref-to-local rigid rotation 与 ScanNet ref-to-target relative pose 的方向大体一致。probe5 median rotation error 约 `1.72..2.36°`，scene0030 为 `1.930591°`。
- scene0030 的失败更像 translation magnitude/ratio instability，而不是 rotation 反了。它在 frame60 ratio `0.665212` 几乎贴近 R23 eval scale `0.656649`，但 frame120 ratio 到 `1.716449`，把 median 推高到 `0.926332`。
- 简单替代 estimator 没有明确胜出。`ratio_median` 的 probe5 mean absrel `0.150980` 仍是候选里最低；`ratio_low_residual_median` 降低 max absrel 到 `0.306391`，但 mean absrel 变差到 `0.183642`。
- 因此本轮不新增 R28 benchmark，也不启动 provider/AP 或 method table。下一步应查 D4RT ref/local translation magnitude 为什么随时间漂移。

### 16.5 final test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 71 tests in 1.731s
OK (skipped=1)
```

## 17. v22.11 ref/local trajectory anchor-policy sweep

目的: v22.10 已把 R27 失败定位到 translation magnitude / ratio instability，尤其是 scene0030 frame120。用户指出 self-stitch/scale 超参和 carrier policy 可能影响结论，因此本轮新增 anchor-policy sweep，验证 ratio 漂移是否来自低置信度 anchor、source-frame 混合、或 residual 外点。该轮不新增 benchmark row，也不生成 method result。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮诊断和测试都是 CPU-only，显式使用 `CUDA_VISIBLE_DEVICES=""`，未占用 GPU。

### 17.1 code changes

新增 `Stream3D/tools/diagnose_v22_ref0_trajectory_policy_sweep.py`:

- 复用 v22.10 的 ref0 trajectory rigid-fit 诊断逻辑。
- 对同一 probe5 cache 跑多组 anchor policy:
  - `vc05_all`
  - `vc07_all`
  - `vc09_all`
  - `vc05_ref_source`
  - `vc05_target_source`
  - `vc05_nonref_source`
  - `vc05_trim90`
  - `vc05_trim80`
- 输出 frame/window/scene/policy-error CSV/JSON 与 Markdown summary。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 `_apply_source_policy` 单测，验证 ref/target/nonref source-frame mask 行为。

### 17.2 targeted validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_ref0_trajectory_policy_sweep.py tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
py_compile pass
Ran 9 tests in 0.008s
OK
```

### 17.3 probe5 diagnostic run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_ref0_trajectory_policy_sweep.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1 \
  --audit-root outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5 \
  --max-anchors 8000 \
  --robust-trim-percentile 90
```

结果:

```text
Wrote v22.11 trajectory policy sweep to outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5
```

Artifacts:

- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_frame_rows.*`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_window_rows.*`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_scene_summary.*`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_errors.*`
- `Stream3D/outputs/audit/v22_11_ref0_trajectory_policy_sweep_probe5/ref0_trajectory_policy_sweep.md`

Policy error vs R23 eval-only scale:

| policy | mean absrel | median absrel | max absrel | mean frames | mean anchors |
|---|---:|---:|---:|---:|---:|
| vc05_trim90 | 0.139010 | 0.088908 | 0.415662 | 15.0 | 158.4 |
| vc05_trim80 | 0.144020 | 0.098948 | 0.422611 | 15.0 | 140.8 |
| vc05_all | 0.150980 | 0.119887 | 0.410697 | 15.0 | 176.2 |
| vc05_ref_source | 0.152608 | 0.097989 | 0.438942 | 9.6 | 12.6 |
| vc05_nonref_source | 0.153684 | 0.122569 | 0.408695 | 15.0 | 163.6 |
| vc09_all | 0.159319 | 0.106471 | 0.388400 | 15.0 | 93.8 |
| vc05_target_source | 0.164263 | 0.149934 | 0.390012 | 15.0 | 30.6 |
| vc07_all | 0.164732 | 0.111906 | 0.408655 | 15.0 | 145.2 |

scene0030 policy summary:

| policy | median ratio | absrel vs R23 | ratio std | anchors | rot err | trans dir err | residual p90 | frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vc05_all | 0.926332 | 0.410697 | 0.269850 | 310 | 1.930591 | 15.677520 | 0.029415 | 15 |
| vc07_all | 0.924992 | 0.408655 | 0.261723 | 267 | 1.870768 | 15.783989 | 0.027896 | 15 |
| vc09_all | 0.911691 | 0.388400 | 0.262760 | 136 | 1.945094 | 16.413512 | 0.022731 | 15 |
| vc05_ref_source | 0.944879 | 0.438942 | 0.302727 | 20 | 1.732618 | 34.093272 | 0.018555 | 13 |
| vc05_target_source | 0.912749 | 0.390012 | 0.257374 | 31 | 1.326005 | 16.327894 | 0.019055 | 15 |
| vc05_nonref_source | 0.925018 | 0.408695 | 0.270615 | 289 | 1.934525 | 15.693827 | 0.030298 | 15 |
| vc05_trim90 | 0.929592 | 0.415662 | 0.274519 | 279 | 1.935720 | 15.665980 | 0.024988 | 15 |
| vc05_trim80 | 0.934155 | 0.422611 | 0.281771 | 248 | 1.942183 | 15.651227 | 0.022771 | 15 |

scene0030 hard frame120:

| policy | ratio | absrel vs R23 | D4RT trans | pose trans | anchors | residual p90 |
|---|---:|---:|---:|---:|---:|---:|
| vc05_all | 1.716449 | 1.613954 | 0.236630 | 0.406164 | 302 | 0.035809 |
| vc07_all | 1.714694 | 1.611280 | 0.236872 | 0.406164 | 247 | 0.034161 |
| vc09_all | 1.686167 | 1.567837 | 0.240880 | 0.406164 | 132 | 0.027030 |
| vc05_ref_source | 1.763000 | 1.684846 | 0.230383 | 0.406164 | 21 | 0.021210 |
| vc05_target_source | 1.701453 | 1.591117 | 0.238710 | 0.406164 | 29 | 0.019811 |
| vc05_nonref_source | 1.713780 | 1.609889 | 0.236998 | 0.406164 | 281 | 0.036950 |
| vc05_trim90 | 1.715394 | 1.612348 | 0.236776 | 0.406164 | 272 | 0.030407 |
| vc05_trim80 | 1.729790 | 1.634271 | 0.234805 | 0.406164 | 242 | 0.027059 |

### 17.4 conclusion

- `vc05_trim90` 是 probe5 mean absrel 最低的 policy: `0.139010`，比 default `vc05_all=0.150980` 小幅改善。
- 但 `trim90` 没有修 scene0030，反而从 `0.410697` 变到 `0.415662`；max absrel 仍高。
- `vc09_all` 和 `vc05_target_source` 能让 scene0030 小幅下降到 `0.388400/0.390012`，但 probe5 mean 变差到 `0.159319/0.164263`。
- `vc05_ref_source` 太稀疏，mean frames 只有 `9.6`，mean anchors 只有 `12.6`，且 scene0030 变差到 `0.438942`。
- scene0030 frame120 在所有 policy 下 ratio 仍约 `1.686..1.763`，核心现象不随 confidence/source-frame/residual-trim 改变。
- 因此 R27/v22.10 的失败不是简单 anchor policy 或 outlier trimming 问题，更像 D4RT ref/local translation magnitude compression、per-frame scale convention 或 normalization 反变换问题。本轮不新增 R28 benchmark，不启动 provider/AP 或 method table。

### 17.5 final test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 72 tests in 1.841s
OK (skipped=1)
```

## 18. v22.12 ref0 scale-convention diagnostic

目的: v22.10/v22.11 已排除 transform 方向错误、简单 carrier policy、confidence/source-frame/residual trimming。根据 OpenD4RT loss 源码，`xyz_3d` loss 使用 `normalize_by_mean_depth=true`，且 normalization 对 pred/GT 各自按 z 均值执行，绝对尺度可能天然弱约束。本轮直接检查 ScanNet target/source depth 与 D4RT predicted local/ref z 是否能解释 R23 eval-only ref0 scale。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮诊断和测试都是 CPU-only，显式使用 `CUDA_VISIBLE_DEVICES=""`，未占用 GPU。

### 18.1 source inspection

本轮读取到的关键源码:

- `Open-d4rt/src/losses/d4rt_loss.py`
  - `_xyz_preprocess` 在 `normalize_depth=True` 时用 `depth = out[..., 2].abs()` 和 `masked_mean_per_sample(depth, mask)` 得到 scale，然后 `out = out / scale`。
  - `pred` 与 `gt` 都各自走 `_xyz_preprocess`，因此 loss 对绝对尺度有弱约束。
- `Open-d4rt/configs/train_effective.yaml`
  - `loss.xyz_3d.normalize_by_mean_depth: true`
  - `loss.xyz_3d.value_transform: sign_x_log1p_abs_x`
- `Stream3D/stream4d/d4rt_adapter.py`
  - inference 只 resize RGB 到模型输入，并保存 `pred_local["xyz_3d"]` 为 `xyz_local`、`pred_ref["xyz_3d"]` 为 `xyz_ref`。
  - cache 没有保存可直接反归一化的 mean-depth metadata。

### 18.2 code changes

新增 `Stream3D/tools/diagnose_v22_ref0_scale_convention.py`:

- 对每个 target frame 拟合 D4RT `xyz_ref -> xyz_local` rigid motion，保留 R27 trajectory ratio。
- 同时采样 ScanNet target depth、source depth，统计 predicted `xyz_local/xyz_ref` 的 z/norm。
- 生成多个候选 scale:
  - `target_depth_over_local_z_median`
  - `target_depth_over_ref_z_median`
  - `source_depth_over_ref_z_median`
  - `source_depth_over_local_z_median`
  - `target_depth_over_local_z_mean`
  - `source_depth_over_ref_z_mean`
  - `trajectory_scale_ratio`
- 对所有候选与 R23 eval-only ref0 scale 比 absrel。
- 输出 frame/window/scene/candidate-error CSV+JSON+MD。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 `_median_positive` / `_safe_ratio` helper 单测。

### 18.3 targeted validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_ref0_scale_convention.py tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
py_compile pass
Ran 10 tests in 0.008s
OK
```

### 18.4 probe5 diagnostic run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_ref0_scale_convention.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1 \
  --audit-root outputs/audit/v22_12_ref0_scale_convention_probe5 \
  --max-anchors 8000 \
  --robust-trim-percentile 90
```

结果:

```text
Wrote v22.12 ref0 scale-convention diagnostic to outputs/audit/v22_12_ref0_scale_convention_probe5
```

Artifacts:

- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_frame_rows.*`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_window_rows.*`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_scene_summary.*`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention_candidate_errors.*`
- `Stream3D/outputs/audit/v22_12_ref0_scale_convention_probe5/ref0_scale_convention.md`

Candidate error vs R23 eval-only scale:

| candidate | mean absrel | median absrel | max absrel | mean candidate |
|---|---:|---:|---:|---:|
| target_depth_over_local_z_median | 0.046781 | 0.058547 | 0.082377 | 0.670878 |
| target_depth_over_local_z_mean | 0.054944 | 0.034633 | 0.161532 | 0.647083 |
| source_depth_over_ref_z_mean | 0.081876 | 0.067390 | 0.141542 | 0.656299 |
| source_depth_over_local_z_median | 0.091145 | 0.091139 | 0.169265 | 0.640300 |
| target_depth_over_ref_z_median | 0.113576 | 0.112007 | 0.207063 | 0.705249 |
| source_depth_over_ref_z_median | 0.128608 | 0.144181 | 0.207063 | 0.676243 |
| trajectory_scale_ratio | 0.150980 | 0.119887 | 0.410697 | 0.693379 |

Per-scene:

| scene | R23 eval scale | trajectory ratio | target/local z | source/ref z | target/ref z | source/local z | local/ref z | target/source depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.561127 | 0.493855 | 0.593979 | 0.677315 | 0.677315 | 0.586915 | 1.127166 | 1.012288 |
| scene0011_00 | 0.561406 | 0.572213 | 0.552688 | 0.584672 | 0.607600 | 0.533094 | 1.093874 | 0.978468 |
| scene0030_00 | 0.656649 | 0.926332 | 0.696137 | 0.751325 | 0.745886 | 0.721613 | 1.030412 | 0.980433 |
| scene0081_01 | 0.862760 | 0.889791 | 0.791689 | 0.721808 | 0.766125 | 0.716725 | 1.007092 | 0.972939 |
| scene0591_00 | 0.707645 | 0.584703 | 0.719900 | 0.646096 | 0.729319 | 0.643151 | 0.996841 | 1.120000 |

scene0030 hard-frame check:

| frame | trajectory ratio | traj absrel | target/local z | target/local absrel | source/ref z | source/ref absrel | target depth | local z | D4RT trans | pose trans |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.926332 | 0.410697 | 0.704815 | 0.073352 | 0.769977 | 0.172586 | 2.366000 | 3.356908 | 0.176657 | 0.163643 |
| 120 | 1.716449 | 1.613954 | 0.749492 | 0.141389 | 0.748628 | 0.140074 | 2.531000 | 3.376956 | 0.236630 | 0.406164 |
| 150 | 0.776038 | 0.181816 | 0.695032 | 0.058454 | 0.743469 | 0.132217 | 2.792000 | 4.017079 | 0.272615 | 0.211559 |

### 18.5 conclusion

- `target_depth_over_local_z_median` 是当前最强 scale attribution 线索: probe5 mean absrel `0.046781`，明显低于 R27/v22.10 trajectory ratio 的 `0.150980`。
- scene0030 负例被大幅解释: scene median absrel 从 `0.410697` 降到 `0.060137`；frame120 从 `1.613954` 降到 `0.141389`。
- `local_over_ref_z_median` 大多约 `1.0`，说明 `xyz_local` 和 `xyz_ref` 共享同一内部尺度，解释了 v22.8 local/ref norm proxy 失败。
- target/source depth ratio 也大多约 `1.0`，说明 scene0030 不是简单 source/target 深度分布差异。
- 因此 blocker 更像 D4RT 输出缺少 target-depth / mean-depth scale anchor，或者训练 loss 的 mean-depth normalization 反归一化不可从当前 cache 直接恢复。
- 但本轮使用 ScanNet depth，仍是 upper-bound/attribution diagnostic，不是 method result。不新增 method table，不启动 provider/AP。

### 18.6 final test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 73 tests in 1.687s
OK (skipped=1)
```

## 19. v22.13 ref0 intrinsics proxy diagnostic

目的: v22.12 找到 `target_depth_over_local_z_median` 是最强 scale attribution 线索，但该线索使用 ScanNet depth。本轮检查 OpenD4RT 官方 `intrinsics_from_queries` / `_estimate_intrinsics_params_from_predictions` 公式是否能从 D4RT predicted `xyz_local/xyz_ref` + `uv` 提供非 depth 的 metric scale proxy。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮诊断和测试都是 CPU-only，显式使用 `CUDA_VISIBLE_DEVICES=""`，未占用 GPU。

### 19.1 source inspection

读取到的关键源码:

- `Open-d4rt/src/eval/tasks.py`
  - `_estimate_intrinsics_params_from_predictions(pred_tracks, pred_uv_norm, image_hw)` 使用:
    - `fx_vals = z * abs(u_px - cx) / max(abs(x), 1e-6)`
    - `fy_vals = z * abs(v_px - cy) / max(abs(y), 1e-6)`
    - 最终取 median `fx/fy`。
- `Open-d4rt/configs/model_effective.yaml`
  - `geometry_decoding.extrinsics_from_queries.method: umeyama`
  - `geometry_decoding.intrinsics_from_queries.method: median_from_fx_fy_estimates`
  - 该模型配置没有独立 metric depth/camera scale head；query heads 主要是 `xyz_3d/uv_2d/visibility/displacement/normal/confidence`。

实现判断:

- 如果 predicted `x/y/z` 共享同一个 uniform internal scale，`z * (u-cx) / x` 形式的 focal estimator 对该尺度不敏感。
- 因此 v22.13 主要验证它是否有意外的 metric-scale 信号；预期它更像 `x/z` 投影一致性诊断，而不是绝对尺度恢复。

### 19.2 code changes

新增 `Stream3D/tools/diagnose_v22_ref0_intrinsics_proxy.py`:

- 对每个 frame 用 D4RT `xyz_local` + `uv` 估计 query-derived `fx/fy`。
- 同时用 `xyz_ref` + `uv` 做 ref branch 对照。
- 输出候选:
  - `local_fx_over_scannet_fx`
  - `local_fxy_over_scannet_fxy`
  - `scannet_fxy_over_local_fxy`
  - `ref_fxy_over_scannet_fxy`
  - `local_fxy_over_ref_fxy`
- 输出 local/ref 的 ScanNet-intrinsics reprojection error，以及 local query-derived intrinsics reprojection error。
- 与 R23 eval-only ref0 scale 比 absrel，生成 frame/window/scene/candidate-error CSV+JSON+MD。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 `_estimate_intrinsics_params_from_query_geometry` 的 uniform scale invariance 单测。

### 19.3 targeted validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_ref0_intrinsics_proxy.py tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
py_compile pass
Ran 11 tests in 0.008s
OK
```

### 19.4 probe5 diagnostic run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_ref0_intrinsics_proxy.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1 \
  --audit-root outputs/audit/v22_13_ref0_intrinsics_proxy_probe5 \
  --max-anchors 8000 \
  --robust-trim-percentile 90
```

结果:

```text
Wrote v22.13 ref0 intrinsics proxy diagnostic to outputs/audit/v22_13_ref0_intrinsics_proxy_probe5
```

Artifacts:

- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_frame_rows.*`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_window_rows.*`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_scene_summary.*`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy_candidate_errors.*`
- `Stream3D/outputs/audit/v22_13_ref0_intrinsics_proxy_probe5/ref0_intrinsics_proxy.md`

Candidate error vs R23 eval-only scale:

| candidate | mean absrel | median absrel | max absrel | mean candidate |
|---|---:|---:|---:|---:|
| local_fx_over_scannet_fx | 0.364440 | 0.378509 | 0.767552 | 0.883376 |
| local_fxy_over_scannet_fxy | 0.428334 | 0.504193 | 0.898481 | 0.926279 |
| local_fy_over_scannet_fy | 0.507383 | 0.517418 | 1.052878 | 0.975513 |
| ref_fxy_over_scannet_fxy | 0.566822 | 0.508222 | 0.855233 | 0.835982 |
| ref_fxy_over_local_fxy | 0.582795 | 0.423599 | 1.234153 | 0.914841 |

Per-scene:

| scene | R23 eval scale | ScanNet/local fxy | local/ScanNet fxy | local fxy | ScanNet fxy | local ScanNet reproj p90 | local pred-intr p90 | ref ScanNet reproj p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.561127 | 1.184775 | 0.844043 | 487.747422 | 577.870605 | 58.464093 | 7.347940 | 780.728892 |
| scene0011_00 | 0.561406 | 0.938246 | 1.065819 | 616.213695 | 578.159967 | 33.711921 | 10.507235 | 268.463152 |
| scene0030_00 | 0.656649 | 0.975672 | 1.024936 | 590.424947 | 576.060247 | 36.645796 | 7.086878 | 131.319775 |
| scene0081_01 | 0.862760 | 1.158068 | 0.863507 | 498.995252 | 577.870605 | 54.505246 | 10.439631 | 492.437685 |
| scene0591_00 | 0.707645 | 1.200353 | 0.833088 | 481.658274 | 578.159967 | 60.792617 | 6.002721 | 158.252875 |

scene0030 hard-frame check:

| frame | R23 scale | local/ScanNet fxy | local fx/ScanNet fx | ScanNet/local fxy | local ScanNet p90 | local pred-intr p90 | ref ScanNet p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.656649 | 1.023584 | 0.915802 | 0.976959 | 33.771021 | 8.695748 | 128.015353 |
| 120 | 0.656649 | 1.005367 | 0.883582 | 0.994662 | 47.593972 | 28.959668 | 162.399082 |
| 150 | 0.656649 | 1.047867 | 0.936088 | 0.954319 | 40.536904 | 21.173376 | 361.735417 |

### 19.5 conclusion

- OpenD4RT query-derived intrinsics 不能近似 R23 metric scale。最佳候选 `local_fx_over_scannet_fx` mean absrel `0.364440`，明显差于 v22.12 `target_depth_over_local_z_median` 的 `0.046781`。
- scene0030 hard frame120 也失败: `local_fxy_over_scannet_fxy=1.005367`，R23 eval scale 是 `0.656649`。
- local branch 用 query-derived intrinsics 的 reprojection p90 通常只有 `6..29px`，显著好于 ref branch 的 ScanNet reprojection p90。这说明 `xyz_local` 的 `x/z` 投影关系内部较一致。
- 但 focal estimation 对 uniform `xyz` scale 不敏感，不能提供 absolute depth / mean-depth scale。
- 因此下一步不应把 `intrinsics_from_queries` 当成 metric-scale 修复主线；仍应追非 GT target-depth proxy、训练 normalization 反归一化、monocular/pseudo-depth prior 或 object/mask/scene scale prior。

### 19.6 final test

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 74 tests in 1.976s
OK (skipped=1)
```

## 20. v22.14 LoGeR geometry scale-proxy diagnostic

目的: v22.12 指出 `target_depth_over_local_z_median` 是最强 scale attribution 线索，但该线索使用 ScanNet depth，不能作为 method。本轮尝试用本仓库 LoGeR geometry backbone 输出的 local pointmap 作为非 GT pseudo-depth / pointmap scale proxy，检查它能否近似 R23 eval-only ref0 depth scale。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- LoGeR 真实 checkpoint inference 只使用 GPU `0`，显式 `CUDA_VISIBLE_DEVICES=0`。
- 诊断汇总和测试均 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

### 20.1 source inspection

读取到的关键入口:

- `run_geometry_backbone_inference.py`
  - 加载 `ckpts/LoGeR/latest.pt` 和 `ckpts/LoGeR/original_config.yaml`。
  - 输出 `.pt` 包含 `local_points`、`world_points`、`camera_poses`、`confidence`。
- ScanNet RGB 输入:
  - `Stream3D/data/scannet/processed/<scene>/color`
- D4RT cache:
  - `Stream3D/outputs/stream4d_debug_v22_local_xyz_probe5_r1`

判断:

- LoGeR pointmap 可以提供一个不使用 ScanNet depth 的 per-pixel local geometry signal。
- 本轮只把它作为 scale proxy diagnostic，不把 LoGeR 输出写成 Stream4D method result。

### 20.2 code changes

新增 `Stream3D/tools/diagnose_v22_loger_scale_proxy.py`:

- 读取预先保存的 LoGeR `.pt`。
- 在 D4RT carrier target UV 位置采样 LoGeR local pointmap。
- 输出非 GT 候选:
  - `loger_z_over_d4rt_local_z_median`
  - `loger_z_over_d4rt_ref_z_median`
  - `loger_norm_over_d4rt_local_norm_median`
  - `loger_norm_over_d4rt_ref_norm_median`
  - `d4rt_local_z_over_loger_z_median`
- 输出 GT positive controls:
  - `scannet_depth_over_loger_z_median`
  - `scannet_depth_over_d4rt_local_z_median`
- 所有结果与 R23 eval-only ref0 scale 比 absrel，并写 frame/window/scene/candidate-error CSV+JSON+MD。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 LoGeR UV sampler normalized-coordinate 单测。
- 新增 candidate rows 标记 GT-control 的单测。

### 20.3 scene0050 LoGeR 4-frame smoke

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR
CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_geometry_backbone_inference.py \
  --input Stream3D/data/scannet/processed/scene0050_00/color \
  --config ckpts/LoGeR/original_config.yaml \
  --checkpoint ckpts/LoGeR/latest.pt \
  --start_frame 0 --end_frame 40 --stride 10 \
  --resolution 504 378 \
  --window_size 4 --overlap_size 1 \
  --output Stream3D/outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke/loger_scene0050_4f.pt
```

结果:

```text
Collected 4 images from Stream3D/data/scannet/processed/scene0050_00/color
Image tensor shape: torch.Size([4, 3, 378, 504])
Model loaded in 12.2s
Inference completed in 6.09s (0.7 FPS)
Output saved to Stream3D/outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke/loger_scene0050_4f.pt
```

保存内容:

- `local_points`: `(4, 378, 504, 3)`
- `world_points`: present
- `camera_poses`: present
- `confidence`: present

### 20.4 probe5 LoGeR inference

继续用 GPU `0` 顺序生成其余 4 个 scene 的 4-frame LoGeR pointmap:

```bash
for scene in scene0011_00 scene0030_00 scene0081_01 scene0591_00; do
  short=${scene%%_*}
  CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_geometry_backbone_inference.py \
    --input Stream3D/data/scannet/processed/${scene}/color \
    --config ckpts/LoGeR/original_config.yaml \
    --checkpoint ckpts/LoGeR/latest.pt \
    --start_frame 0 --end_frame 40 --stride 10 \
    --resolution 504 378 \
    --window_size 4 --overlap_size 1 \
    --output Stream3D/outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke/loger_${short}_4f.pt \
    > Stream3D/outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke/loger_${short}_4f_inference.log 2>&1
done
```

Per-scene logs:

| scene | model load | inference | output |
|---|---:|---:|---|
| scene0011_00 | 11.5s | 2.77s | saved |
| scene0030_00 | 11.5s | 2.72s | saved |
| scene0081_01 | 11.7s | 2.65s | saved |
| scene0591_00 | 11.3s | 2.72s | saved |

### 20.5 probe5 scale-proxy diagnostic

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_loger_scale_proxy.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1 \
  --loger-output-root outputs/audit/v22_14_loger_scale_proxy_scene0050_smoke \
  --loger-template 'loger_{scene_short}_4f.pt' \
  --audit-root outputs/audit/v22_14_loger_scale_proxy_probe5_4f \
  --max-frames 4 \
  --frame-stride 10 \
  --max-windows-per-scene 1
```

结果:

```text
Wrote v22.14 LoGeR scale-proxy diagnostic to outputs/audit/v22_14_loger_scale_proxy_probe5_4f
```

Artifacts:

- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy.md`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_candidate_errors.*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_scene_summary.*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_frame_rows.*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_window_rows.*`
- `Stream3D/outputs/audit/v22_14_loger_scale_proxy_probe5_4f/loger_scale_proxy_metadata.json`

Candidate error vs R23 eval-only scale:

| candidate | uses ScanNet depth for proxy | mean absrel | median absrel | max absrel | mean candidate | scenes |
|---|---:|---:|---:|---:|---:|---:|
| loger_z_over_d4rt_ref_z_median | False | 0.730324 | 0.693623 | 0.884752 | 0.173463 | 5 |
| loger_norm_over_d4rt_ref_norm_median | False | 0.732677 | 0.692137 | 0.885836 | 0.171971 | 5 |
| loger_z_over_d4rt_local_z_median | False | 0.734930 | 0.691965 | 0.887910 | 0.170685 | 5 |
| loger_norm_over_d4rt_local_norm_median | False | 0.735012 | 0.690007 | 0.889936 | 0.170475 | 5 |
| d4rt_local_z_over_loger_z_median | False | 8.510500 | 8.955088 | 10.985711 | 6.388830 | 5 |
| scannet_depth_over_d4rt_local_z_median | True | 0.044534 | 0.035141 | 0.075514 | 0.681455 | 5 |
| scannet_depth_over_loger_z_median | True | 5.469452 | 4.859856 | 9.703976 | 4.507986 | 5 |

Per-scene:

| scene | R23 eval scale | loger/local z | loger/ref z | local/loger z | GT/local z | GT/loger z | frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 0.561127 | 0.179017 | 0.181363 | 5.586065 | 0.591627 | 3.288122 | 4 |
| scene0011_00 | 0.561406 | 0.151206 | 0.160306 | 6.613573 | 0.519012 | 3.426504 | 4 |
| scene0030_00 | 0.656649 | 0.202271 | 0.201182 | 4.943874 | 0.676458 | 3.343461 | 4 |
| scene0081_01 | 0.862760 | 0.096707 | 0.099431 | 10.340792 | 0.893079 | 9.234962 | 4 |
| scene0591_00 | 0.707645 | 0.224223 | 0.225030 | 4.459844 | 0.727100 | 3.246880 | 4 |

### 20.6 conclusion

- LoGeR pointmap 不能直接作为 R23 metric scale proxy。最好的非 GT 候选 `loger_z_over_d4rt_ref_z_median` mean absrel 仍为 `0.730324`，远差于 v22.12 GT-depth attribution 的 `0.046781`。
- LoGeR/D4RT scale ratio 基本落在 `0.17` 左右，而 R23 eval scale 落在 `0.56..0.86`；这不是一个小偏差。
- GT positive control `scannet_depth_over_d4rt_local_z_median` 仍然很强，mean absrel `0.044534`，再次确认 v22.12 的结论: `target_depth / D4RT local_z` 是真正接近 R23 scale 的线索。
- `scannet_depth_over_loger_z_median` 很差，mean absrel `5.469452`，说明 LoGeR pointmap 自身也不在 ScanNet metric depth 尺度上。
- scene0081 是硬负例: `loger_z_over_d4rt_local_z_median=0.096707`，R23 scale `0.862760`，`GT/loger z=9.234962`。
- 本轮不生成 method row，不启动 Phase F。

### 20.7 final validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_loger_scale_proxy.py tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass
Ran 13 tests in 0.008s
OK
Ran 76 tests in 1.775s
OK (skipped=1)
```

## 21. v22.15 OpenD4RT xyz loss scale-invariance diagnostic

目的: v22.12/v22.14 都指向 “target depth / D4RT local z” 是最强 scale attribution 线索，但非 GT proxy 失败。本轮继续向下检查 OpenD4RT 训练 loss 本身是否约束 metric scale，验证 `xyz_3d` head 是否天然丢掉 uniform scale。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮无 GPU inference，全部 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

### 21.1 source inspection

读取 OpenD4RT source:

- `Open-d4rt/src/losses/d4rt_loss.py`
- `Open-d4rt/configs/train_effective.yaml`
- `Open-d4rt/configs/model_effective.yaml`
- `Open-d4rt/infer_track_3d.py`

关键代码:

```python
def _xyz_preprocess(self, xyz, mask, normalize_depth, transform_log):
    out = xyz
    if normalize_depth:
        depth = out[..., 2].abs()
        scale = masked_mean_per_sample(depth, mask).clamp_min(1e-6).unsqueeze(-1)
        out = out / scale
    if transform_log:
        out = torch.sign(out) * torch.log1p(out.abs())
    return out
```

训练配置:

```yaml
loss:
  xyz_3d:
    normalize_by_mean_depth: true
    value_transform: sign_x_log1p_abs_x
```

判断:

- pred 和 GT 在 loss 里分别按各自 mean abs-z 归一化。
- 这会让 uniform pred scale 在 loss space 中完全抵消。
- `model_effective.yaml` 只看到 `xyz_3d` / `uv_2d` / `visibility` / `displacement` / `normal` / `confidence` 等 heads，没有显式 scale 或 mean-depth head。
- `infer_track_3d.py` 直接写出 `pred_local["xyz_3d"]` / `pred_ref["xyz_3d"]`，没有看到对应 inverse-normalization metadata。

### 21.2 code changes

新增 `Stream3D/tools/diagnose_v22_loss_scale_invariance.py`:

- 读取 `outputs/stream4d_debug_v22_local_xyz_probe5_r1` 的 `xyz_local` / `uv_pred`。
- 在 target UV 位置用 ScanNet depth + intrinsics 构造 GT camera-space XYZ。
- 对 pred scale `0.25/0.5/1.0/2.0/4.0` 做 sweep。
- 同时记录:
  - metric-space `metric_l1`
  - metric-space `metric_z_absrel`
  - OpenD4RT loss-space `loss_l1_signed_log`
  - normalized-only `loss_l1_normalized`
  - no-normalization raw L1 control
- 输出 frame/sweep/scene/by-scale CSV+JSON+MD。
- Metadata 显式标记 `is_diagnostic_only=true`、`uses_scannet_depth_for_gt_xyz=true`、`forbidden_for_method_table=true`。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 uniform pred scale 下 loss-space L1 不变的单测。
- 新增 metric-space L1 会随 uniform pred scale 改变的单测。

### 21.3 validation before run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_loss_scale_invariance.py tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
py_compile pass
Ran 15 tests in 0.008s
OK
```

### 21.4 probe5 loss scale-invariance diagnostic

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_loss_scale_invariance.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1 \
  --audit-root outputs/audit/v22_15_loss_scale_invariance_probe5 \
  --max-windows-per-scene 1 \
  --max-anchors 8000 \
  --pred-scales 0.25 0.5 1.0 2.0 4.0
```

结果:

```text
Wrote v22.15 loss scale-invariance diagnostic to outputs/audit/v22_15_loss_scale_invariance_probe5
```

Metadata:

| item | value |
|---|---:|
| frame rows | 80 |
| sweep rows | 400 |
| pred scales | `0.25,0.5,1.0,2.0,4.0` |
| loss signed-log range across pred scales mean | 0.000000 |
| loss signed-log range across pred scales max | 0.000000 |
| normalized loss range across pred scales mean | 0.000000 |
| normalized loss range across pred scales max | 0.000000 |
| metric L1 range across pred scales mean | 4.750760 |
| metric L1 range across pred scales max | 6.158282 |
| metric z absrel range across pred scales mean | 4.864648 |
| metric z absrel range across pred scales max | 6.728604 |

Scale sweep:

| pred scale | frames | loss L1 mean | normalized loss L1 mean | metric L1 mean | z absrel mean |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 80 | 0.034302 | 0.056277 | 0.606992 | 0.620367 |
| 0.50 | 80 | 0.034302 | 0.056277 | 0.253184 | 0.256145 |
| 1.00 | 80 | 0.034302 | 0.056277 | 0.520732 | 0.526143 |
| 2.00 | 80 | 0.034302 | 0.056277 | 1.995299 | 2.037693 |
| 4.00 | 80 | 0.034302 | 0.056277 | 4.967838 | 5.075384 |

Per-scene:

| scene | frames | anchors mean | loss range mean | metric range mean | metric best scale median | GT/pred mean-z median |
|---|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 16 | 255.187500 | 0.000000 | 4.963678 | 0.500000 | 0.578260 |
| scene0011_00 | 16 | 39.687500 | 0.000000 | 5.480789 | 0.500000 | 0.566638 |
| scene0030_00 | 16 | 285.750000 | 0.000000 | 4.918066 | 0.500000 | 0.677070 |
| scene0081_01 | 16 | 40.562500 | 0.000000 | 4.375589 | 1.000000 | 0.867373 |
| scene0591_00 | 16 | 170.125000 | 0.000000 | 4.015680 | 0.500000 | 0.727257 |

### 21.5 conclusion

- OpenD4RT `xyz_3d` loss 确认对 uniform pred scale 不敏感。真实 probe5 diagnostic 中，`0.25..4.0` pred scale sweep 的 `loss_l1_signed_log_range_across_pred_scales_mean/max` 都是 `0.0`。
- metric-space 明显随 scale 改变: `metric_l1_range_across_pred_scales_mean=4.750760`，`metric_z_absrel_range_across_pred_scales_mean=4.864648`。
- 这解释了 v22.12 的现象: `target_depth/local_z` 是缺失的 inference-time scale attribution，而不是一个已经由 `xyz_3d` loss 唯一约束出来的尺度。
- 这也解释了 v22.13/v22.14 的负结果: query-derived intrinsics 对 uniform scale 不敏感；LoGeR pointmap 没有提供可直接匹配 R23 的 metric scale。
- 本轮不新增 method row，不启动 Phase F。

### 21.6 final validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_loss_scale_invariance.py tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass
Ran 15 tests in 0.008s
OK
Ran 78 tests in 2.038s
OK (skipped=1)
```

## 22. v22.16 target-scale observability diagnostic

目的: v22.12 找到 `target_depth / D4RT local_z` 是强 GT-only scale attribution，v22.13/v22.14/v22.15 又排除了 query intrinsics、LoGeR pointmap 和 OpenD4RT loss 自带 metric scale。本轮继续问: 这个 GT-only target-scale label 是否能被 D4RT 内部可观测统计用简单模型预测出来，还是基本不可观测。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮无 GPU inference，全部 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

### 22.1 code changes

新增 `Stream3D/tools/diagnose_v22_target_scale_observability.py`:

- 读取 `outputs/stream4d_debug_v22_local_xyz_probe5_r1`。
- 逐 frame 生成两个 diagnostic label:
  - `target_depth_over_local_z_median`: ScanNet target depth / D4RT `xyz_local` abs-z median，使用 ScanNet depth。
  - `eval_ref0_depth_scale`: R23 eval-only ref0 scale，使用 ScanNet depth/pose。
- 提取 27 个 D4RT-internal feature:
  - visibility/confidence
  - UV spread / bbox area
  - predicted local/ref z and norm statistics
  - local/ref ratios
  - D4RT ref->local rigid translation/residual
  - source-frame spread
- 另做一个 `D4RT + ScanNet pose` diagnostic control，加入 `pose_translation_norm` 和 `trajectory_scale_ratio`，明确标记为 pose-control，不是 method。
- 对 `global_median_loo`、`linear_loo_d4rt_internal`、`linear_loo_d4rt_plus_pose_diagnostic`、`oracle_scene_median` 做 leave-one-scene-out 对照。
- 额外做所有单特征 LOO linear sweep，避免全特征小样本过拟合掩盖弱信号。
- 输出 frame/window/prediction/correlation/univariate/scene CSV+JSON+MD。
- Metadata 显式标记 `is_diagnostic_only=true`、`forbidden_for_method_table=true`。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 absrel summary 单测。
- 新增 synthetic leave-one-scene-out linear scale prediction 单测。
- 新增 Spearman tie handling 单测。

### 22.2 validation before run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_target_scale_observability.py tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
```

结果:

```text
py_compile pass
Ran 18 tests in 0.010s
OK
```

### 22.3 probe5 observability diagnostic

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_target_scale_observability.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1 \
  --audit-root outputs/audit/v22_16_target_scale_observability_probe5 \
  --max-anchors 8000
```

结果:

```text
Wrote v22.16 target-scale observability diagnostic to outputs/audit/v22_16_target_scale_observability_probe5
```

Metadata:

| item | value |
|---|---:|
| frame rows | 75 |
| window rows | 5 |
| D4RT internal features | 27 |
| pose diagnostic features | 29 |
| best D4RT univariate for target/local label | `rigid_residual_median` |
| best D4RT univariate target/local mean absrel | 0.129793 |
| best pose-control target/local mean absrel | 0.136832 |
| best D4RT univariate for R23 eval scale | `uv_x_std` |
| best D4RT univariate R23 eval mean absrel | 0.157355 |
| best pose-control R23 eval mean absrel | 0.192556 |

Predictor summary:

| predictor | label | mean absrel | median absrel | max absrel | uses pose feature |
|---|---|---:|---:|---:|---|
| global median LOO | eval ref0 depth scale | 0.179578 | 0.215069 | 0.294094 | False |
| linear LOO D4RT internal | eval ref0 depth scale | 0.316544 | 0.217751 | 1.212866 | False |
| linear LOO D4RT + pose diagnostic | eval ref0 depth scale | 0.374789 | 0.259574 | 1.793648 | True |
| oracle scene median | eval ref0 depth scale | 0.000000 | 0.000000 | 0.000000 | False |
| oracle scene median | target depth / local z | 0.056326 | 0.029889 | 0.451290 | False |
| global median LOO | target depth / local z | 0.160601 | 0.148082 | 0.452184 | False |
| linear LOO D4RT internal | target depth / local z | 0.466158 | 0.364515 | 2.336099 | False |
| linear LOO D4RT + pose diagnostic | target depth / local z | 0.619112 | 0.446028 | 2.756365 | True |

Best univariate LOO:

| label | feature | mean absrel | median absrel | max absrel | uses pose feature |
|---|---|---:|---:|---:|---|
| eval ref0 depth scale | `uv_x_std` | 0.157355 | 0.224958 | 0.318771 | False |
| eval ref0 depth scale | `pred_local_abs_z_median` | 0.158963 | 0.216945 | 0.346765 | False |
| eval ref0 depth scale | `local_over_ref_z_median` | 0.161262 | 0.206724 | 0.382397 | False |
| eval ref0 depth scale | `trajectory_scale_ratio` | 0.192556 | 0.223558 | 0.382371 | True |
| target depth / local z | `rigid_residual_median` | 0.129793 | 0.098340 | 0.403580 | False |
| target depth / local z | `local_over_ref_z_median` | 0.130900 | 0.094871 | 0.492785 | False |
| target depth / local z | `pred_ref_norm_median` | 0.139480 | 0.131980 | 0.425850 | False |
| target depth / local z | `trajectory_scale_ratio` | 0.136832 | 0.130683 | 0.459729 | True |

Per-scene:

| scene | frames | target/local scale median | target/local scale std | eval scale median | D4RT all-feature target absrel | pose-control target absrel | D4RT all-feature eval absrel |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0011_00 | 15 | 0.552688 | 0.059716 | 0.561406 | 0.666638 | 0.638850 | 0.457999 |
| scene0030_00 | 15 | 0.696137 | 0.025571 | 0.656649 | 0.288845 | 0.400978 | 0.188945 |
| scene0050_00 | 15 | 0.593979 | 0.011217 | 0.561127 | 0.281819 | 0.888966 | 0.536035 |
| scene0081_01 | 15 | 0.791689 | 0.099616 | 0.862760 | 0.600985 | 0.704292 | 0.154847 |
| scene0591_00 | 15 | 0.719900 | 0.050971 | 0.707645 | 0.492503 | 0.462477 | 0.244892 |

Strongest feature correlations:

| label | feature | spearman | uses pose feature |
|---|---|---:|---|
| target depth / local z | `uv_y_std` | -0.507340 | False |
| eval ref0 depth scale | `local_over_ref_z_median` | -0.492555 | False |
| eval ref0 depth scale | `uv_bbox_area` | -0.479925 | False |
| eval ref0 depth scale | `confidence_mean` | -0.445530 | False |
| eval ref0 depth scale | `trajectory_scale_ratio` | 0.334467 | True |
| target depth / local z | `trajectory_scale_ratio` | 0.278065 | True |

### 22.4 conclusion

- 有弱可观测信号，但没有形成可靠 scale anchor。最佳 D4RT 单特征对 `target_depth/local_z` 的 mean absrel 是 `0.129793`，比 global median `0.160601` 好，但仍远差于 scene-oracle median `0.056326`，且 max absrel `0.403580`。
- 对 R23 eval scale，最佳 D4RT 单特征 `uv_x_std` mean absrel `0.157355`，只比 global median `0.179578` 小幅好，max absrel `0.318771`。
- 27 维 D4RT-internal linear LOO 过拟合明显: `target_depth/local_z` mean absrel `0.466158`，R23 eval scale mean absrel `0.316544`，都差于 global median。
- 加入 ScanNet pose control 后也没有改善，说明 R27 类 trajectory signal 在这个简单 LOO regression 里不是足够稳定的补充。
- 当前最合理解释: `target_depth/local_z` 在 scene 内相对稳定，但不能从这些简单 D4RT 内部统计中可靠恢复。下一步不能只靠线性组合 visibility/confidence/uv spread/local-ref ratio；需要更强 scale anchor，比如保留训练 normalization scale、引入真正 metric pseudo-depth/VO、或做自监督 target-depth consistency。
- 本轮不新增 method row，不启动 Phase F。

### 22.5 final validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_target_scale_observability.py tests/test_v22_direct_reconstruction.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass
Ran 18 tests in 0.010s
OK
Ran 81 tests in 1.773s
OK (skipped=1)
```

## 23. v22.17 OpenD4RT scale metadata audit

### 23.1 purpose

v22.15 已确认 OpenD4RT `xyz_3d` loss 对 uniform metric scale 不敏感；v22.16 又确认简单 D4RT 内部统计不能可靠预测 `target_depth/local_z` scale。本轮继续按 Stop D 检查一个更直接的问题: 当前 OpenD4RT 训练/推理/cache 路径里是否已经有可反归一化的 mean-depth / metric-scale side channel，只是之前没有读出来。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮只做源码/config/cache audit 和 CPU 单测；所有命令显式 `CUDA_VISIBLE_DEVICES=""`。

### 23.2 code changes

新增 `Stream3D/tools/diagnose_v22_opend4rt_scale_metadata.py`:

- 检查 `Open-d4rt/configs/train_effective.yaml` 中 `loss.xyz_3d.normalize_by_mean_depth`。
- 检查 `Open-d4rt/src/losses/d4rt_loss.py` 是否对 pred/GT 各自调用 `_xyz_preprocess`，并各自除以 mean abs-z。
- 解析 `Open-d4rt/src/model/heads.py` 输出 key，检查是否存在 scale/depth head。
- 解析 `Open-d4rt/infer_track_3d.py` return key，检查 inference 是否返回 scale metadata。
- 检查 `Open-d4rt/docs/data_schema.md` 是否有 mean-depth / depth-scale / metric-scale 字段。
- 扫描 `outputs/stream4d_debug_v22_local_xyz_probe5_r1/*/carriers_window*.npz` 的 keys。
- 输出 source evidence / cache keys / metadata 的 CSV+JSON+MD。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 scale-key detector 单测，确认 `xyz_3d` / `tracks_xyz_local` / `tracks_uv_norm` 不会误报为 scale key。
- 新增 explicit scale head detector 单测。
- 新增 independent loss normalization detector 单测。

### 23.3 validation and audit run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_opend4rt_scale_metadata.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_opend4rt_scale_metadata.py \
  --audit-root outputs/audit/v22_17_opend4rt_scale_metadata \
  --max-cache-files 16
```

结果:

```text
py_compile pass
Ran 21 tests in 0.010s
OK
Wrote v22.17 OpenD4RT scale metadata audit to outputs/audit/v22_17_opend4rt_scale_metadata
```

Metadata:

| item | value |
|---|---:|
| loss_config_enables_mean_depth_normalization | True |
| loss_normalizes_pred_and_gt_independently | True |
| model output keys | `confidence, displacement, normal, uv_2d, visibility, xyz_3d` |
| model_has_explicit_scale_or_depth_head | False |
| inference return keys | `clip_frames, stitch_diagnostics, t_cam, t_src, t_tgt, tracks_confidence, tracks_uv_norm, tracks_visibility, tracks_visibility_logits, tracks_xyz_local, tracks_xyz_ref0, u, v` |
| inference_return_has_scale_metadata | False |
| schema_has_metric_xyz | True |
| schema_has_mean_depth_scale_field | False |
| cache_files_scanned | 5 |
| cache_files_with_scale_like_keys | 0 |
| cache_files_with_depth_like_keys | 0 |
| method_result | False |
| diagnostic_only | True |

### 23.4 final validation

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
Ran 84 tests in 1.855s
OK (skipped=1)
```

### 23.5 conclusion

- OpenD4RT 当前训练目标是 metric xyz，但启用的 `xyz_3d` loss 会对 pred/GT 各自移除 mean-depth scale。
- 模型没有显式 scale/depth head；官方 inference return 没有 scale metadata；v22 local carrier cache 也没有 mean-depth / depth-scale / metric-scale key。
- 因此没有“已落盘但未使用”的隐藏反归一化尺度可以直接挖出来。
- 下一步如果继续修 method，需要新增/学习/引入真正的 scale anchor，例如显式保留训练 normalization scale、引入 metric pseudo-depth/VO、或做 self-supervised target-depth consistency。
- 本轮不新增 method row，不启动 Phase F。

## 24. v22.18 self-supervised scale-sensitivity diagnostic

### 24.1 purpose

v22.17 确认当前 OpenD4RT inference/cache 没有可直接复用的 scale metadata。本轮继续按 Stop D 检查一个自然后续问题: 如果不使用 GT depth，只用现有 D4RT `xyz_local` / `uv` / relative-depth shape / depth order 这类 self-supervised consistency，是否能对 uniform metric scale 产生可观测约束。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮只跑 CPU diagnostic 和 CPU tests；所有命令显式 `CUDA_VISIBLE_DEVICES=""`。

### 24.2 code changes

新增 `Stream3D/tools/diagnose_v22_self_supervised_scale_sensitivity.py`:

- 对 probe5 v22 local carrier cache 的 `xyz_local` 做 uniform scale sweep: `0.25/0.5/1.0/2.0/4.0`。
- 对每个 frame 统计:
  - D4RT `uv` 与 scaled `xyz_local` 的 pinhole reprojection error。
  - normalized positive z shape L1。
  - predicted z 与 target depth 的 rank/Spearman consistency。
  - GT depth AbsRel positive control。
- GT depth 只作为 positive control，metadata 标记 `uses_scannet_depth_for_positive_control=true`、`diagnostic_only=true`、`method_result=false`。
- 输出 frame/sweep/scene summary 的 CSV+JSON+MD。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 synthetic scale-sweep 单测，确认 UV reprojection / normalized z / depth rank 对 uniform scale 不敏感，而 GT-depth AbsRel 能选出正确 scale。

### 24.3 validation and audit run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_self_supervised_scale_sensitivity.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_self_supervised_scale_sensitivity.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --cache-root outputs/stream4d_debug_v22_local_xyz_probe5_r1 \
  --audit-root outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5 \
  --max-anchors 8000
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass
Ran 22 tests in 0.011s
OK
Wrote v22.18 self-supervised scale-sensitivity diagnostic to outputs/audit/v22_18_self_supervised_scale_sensitivity_probe5
Ran 85 tests in 1.845s
OK (skipped=1)
```

Aggregate:

| item | value |
|---|---:|
| scene_count | 5 |
| frame_count | 80 |
| pred_scales | `0.25,0.5,1.0,2.0,4.0` |
| uv_reprojection_median_px_range_mean | 0.000000 |
| normalized_z_l1_range_mean | 0.000000 |
| depth_rank_spearman_range_mean | 0.000000 |
| gt_depth_absrel_range_mean | 4.864648 |
| gt_depth_absrel_min_mean | 0.210736 |
| gt_depth_absrel_at_scale_1_mean | 0.526143 |
| method_result | False |

Per-scene:

| scene | frames | uv range mean | normalized-z range mean | rank range mean | GT absrel range mean | GT absrel min mean | GT absrel scale1 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0050_00 | 16 | 0.000000 | 0.000000 | 0.000000 | 5.736074 | 0.162162 | 0.724559 |
| scene0011_00 | 16 | 0.000000 | 0.000000 | 0.000000 | 5.958859 | 0.134865 | 0.777020 |
| scene0030_00 | 16 | 0.000000 | 0.000000 | 0.000000 | 4.643735 | 0.266497 | 0.480311 |
| scene0081_01 | 16 | 0.000000 | 0.000000 | 0.000000 | 3.484639 | 0.193569 | 0.193569 |
| scene0591_00 | 16 | 0.000000 | 0.000000 | 0.000000 | 4.499931 | 0.296587 | 0.455255 |

### 24.4 conclusion

- 当前已有 D4RT `uv` + `xyz_local` 的 pinhole reprojection consistency 对 uniform scale 完全不敏感，range mean/max 都是 `0.0`。
- normalized z shape 和 depth rank consistency 也对 uniform scale 完全不敏感，range mean/max 都是 `0.0`。
- GT depth positive control 对同一 scale sweep 强敏感，`gt_depth_absrel_range_mean=4.864648`，说明诊断本身能看见 metric scale，只是无 GT/self-supervised 约束看不见。
- 因此“只用现有 D4RT UV/relative-shape/depth-order 做 target-depth consistency”不能恢复 metric scale；需要外部 metric anchor、显式 learned scale/depth head、或训练/推理保留 normalization scale。
- 本轮不新增 method row，不启动 Phase F。

## 25. v22.19 scale-anchor tolerance diagnostic

### 25.1 purpose

v22.18 已证明现有 D4RT UV / relative-shape / depth-rank self-supervised consistency 对 uniform metric scale 不敏感。本轮继续按 Stop D 做一个更直接的容忍度诊断: 从 R23 `xyz_ref0 + ScanNet ref0 pose + eval-only scale` upper bound 出发，只扰动 fitted scale，量化未来非 GT scale anchor 需要多准确才能保住 R23 点云质量。

该实验仍是 diagnostic-only:

- 使用 ScanNet ref0 pose 和 ScanNet depth/pose anchors 形成 R23 oracle scale。
- 后续只对 oracle scale 乘以固定 multiplier。
- 不生成 reconstruction/provider/AP method row。

GPU policy:

- 用户更新: GPU `0,1` 可用，其它 GPU 暂时不用。
- 本轮诊断和测试均 CPU-only，显式 `CUDA_VISIBLE_DEVICES=""`。

### 25.2 code changes

新增 `Stream3D/tools/diagnose_v22_scale_anchor_tolerance.py`:

- 复用 v22 direct reconstruction benchmark 的 R23 point collection、ref0 pose scale fitting、point/depth metrics。
- 对 R23 fitted scale 做 multiplier sweep: `0.5/0.75/0.9/1.0/1.1/1.25/1.5`。
- 输出 scene rows / summary / metadata 的 CSV+JSON+MD。
- metadata 标记 `diagnostic_only=true`、`method_result=false`、`uses_gt_depth_or_pose_for_oracle_scale=true`。

修改 `Stream3D/tests/test_v22_direct_reconstruction.py`:

- 新增 scale-anchor tolerance helper 单测，确认 scale multiplier 只修改 copied fit，不 mutation 原始 fit。

### 25.3 validation and audit run

命令:

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile tools/diagnose_v22_scale_anchor_tolerance.py
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest tests.test_v22_direct_reconstruction
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/diagnose_v22_scale_anchor_tolerance.py \
  --seq-list splits/scannet_v6_probe5.txt \
  --audit-root outputs/audit/v22_19_scale_anchor_tolerance_probe5 \
  --scale-multipliers 0.5 0.75 0.9 1.0 1.1 1.25 1.5 \
  --max-anchors 8000
CUDA_VISIBLE_DEVICES="" /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests
```

结果:

```text
py_compile pass
Ran 23 tests in 0.011s
OK
Wrote v22.19 scale-anchor tolerance diagnostic to outputs/audit/v22_19_scale_anchor_tolerance_probe5
Ran 86 tests in 1.556s
OK (skipped=1)
```

Aggregate:

| item | value |
|---|---:|
| scene_count | 5 |
| scale_multipliers | `0.5,0.75,0.9,1.0,1.1,1.25,1.5` |
| oracle_fscore10_mean | 0.622259 |
| oracle_completeness20_mean | 0.888330 |
| oracle_depth_delta1_mean | 0.968216 |
| method_result | False |

Scale sweep:

| multiplier | rel scale error | F@10 mean | F@10 retention | comp@20 mean | outlier@20 mean | depth delta1 mean | depth AbsRel mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.500000 | 0.067882 | 0.109089 | 0.154312 | 0.826163 | 0.644346 | 0.227499 |
| 0.75 | 0.250000 | 0.113740 | 0.182785 | 0.333507 | 0.635708 | 0.872429 | 0.109861 |
| 0.9 | 0.100000 | 0.320440 | 0.514962 | 0.591724 | 0.259553 | 0.946160 | 0.067317 |
| 1.0 | 0.000000 | 0.622259 | 1.000000 | 0.888330 | 0.079411 | 0.968216 | 0.059158 |
| 1.1 | 0.100000 | 0.464646 | 0.746708 | 0.760754 | 0.285874 | 0.967814 | 0.064413 |
| 1.25 | 0.250000 | 0.183494 | 0.294884 | 0.502301 | 0.557986 | 0.960305 | 0.078098 |
| 1.5 | 0.500000 | 0.098092 | 0.157638 | 0.245881 | 0.697236 | 0.928020 | 0.097342 |

### 25.4 conclusion

- R23 upper-bound 对 scale anchor 误差非常敏感。10% under-scale 时 F@10 retention 只有 `0.514962`，10% over-scale 为 `0.746708`。
- 25% scale error 已基本破坏 R23 geometry: F@10 retention 只有 `0.182785/0.294884`，completeness@20 也降到 `0.333507/0.502301`。
- 这说明 v22.16 那种约 `0.13` mean AbsRel 的弱 D4RT-internal scale predictor 仍可能不够，尤其在 under-scale 或 hard-scene 上会把 point F-score 打掉很多。
- 下一步若要 method 化，scale anchor 目标不应只是“比 global median 略好”，而应尽量接近 R23/R12 GT target-depth attribution 的 `~0.05` mean AbsRel 量级，或通过 joint optimization 抵消 scale residual。
- 本轮不新增 method row，不启动 Phase F。
