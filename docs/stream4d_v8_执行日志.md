# Stream4D v8 执行日志

日期：2026-06-09（Asia/Singapore）  
计划文件：`docs/stream4d_v8_d4rt_native_surfel_experiment_plan_for_codex.md`  
复盘文件：`docs/stream4d_v8_实验结果复盘.md`

## 运行约束

```text
不允许编造数据。
方法结果不得读取 GT；GT 只能用于 diagnostic_only=true, uses_gt=true 的诊断。
本轮优先执行 v8 Lane 0 / Lane 1 / Lane 2 coverage diagnostic；若 D4RT semi-dense surfel field 或 mask measurement 不成立，不继续伪造 Lane 3 AP。
用户声明 GPU 6/7 可用；本轮 D4RT 运行默认使用 CUDA_VISIBLE_DEVICES=6。
```

## 环境探测

```text
pwd = /mnt/data/users/chengshun.wang/pjs/LoGeR
系统 python3 = /usr/bin/python3，未安装 torch。
base conda python = /mnt/data/users/chengshun.wang/miniconda3/bin/python，未安装 torch。
正确 Stream4D/D4RT 环境 = /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
loger env torch = 2.6.0+cu124，CUDA 可用。
GPU 6/7 均为 NVIDIA RTX A5000，nvidia-smi 显示空闲。
D4RT checkpoint = Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
```

环境纠正记录：

```text
最初误用系统 python3/base conda 做 smoke，二者均无 torch，不能跑 D4RT。
随后又临时用 4D env 跑了 G1 diagnostic；用户指出历史 Stream4D 环境应为 loger。
4D env 输出保留为错误环境 diagnostic，不进入 v8 有效结果表。
后续 D4RT/v8 Lane 1 固定使用：
  PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
  PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
  CUDA_VISIBLE_DEVICES=6
base conda 仅可用于无 torch 的 numpy/open3d/evaluator 类审计，不作为 D4RT adapter 环境。
```

## 修改记录

| 时间 | 文件 | 修改 | 审计理由 |
|---|---|---|---|
| 2026-06-09 | `Stream3D/tools/export_d4rt_grid_surfel_field_v8.py` | 新增 G1 grid/semi-dense surfel field diagnostic 工具 | 计划 Lane 1 要验证 D4RT dense/semi-dense surfel field，而不是继续 sparse carrier component |
| 2026-06-09 | `Stream3D/tools/export_d4rt_grid_surfel_field_v8.py` | 新增 `--grid-margin-ratio` | 按 OpenD4RT helper `_grid_query_points(..., margin_ratio=0.02)` 修复方向排查边缘 uv/preprocessing 问题 |
| 2026-06-09 | `Stream3D/tools/compare_d4rt_adapter_official_v8.py` | 新增 adapter-vs-official helper 等价性诊断 | 按计划 Lane 1 blocker 修复方向检查 D4RTAdapter 是否偏离 OpenD4RT helper |
| 2026-06-09 | `Stream3D/stream4d/scannet_stream.py` | `load_window(..., require_masks=False)` 支持缺 mask 帧用 0 mask 占位 | 连续帧 D4RT geometry diagnostic 不应依赖 stride-10 2D mask 是否存在；默认仍要求 mask |
| 2026-06-09 | `Stream3D/tools/export_d4rt_grid_surfel_field_v8.py` | 新增 `--allow-missing-masks` | 允许 Lane 1 纯几何/轨迹诊断使用连续 ScanNet frames |
| 2026-06-09 | `Stream3D/tools/diagnose_v8_mask_measurement_coverage.py` | 新增 Lane 2 非 GT mask measurement coverage diagnostic | 量化连续 D4RT clip 中可用 2D mask observation 覆盖，不读 GT、不产 AP |
| 2026-06-09 | `Stream3D/tools/export_v8_surfel_object_field.py` | 新增 Lane3 surfel object field exporter，支持 A/B/C lightweight prototypes | 执行计划第 7 节要求并行比较 signed clustering / surfacelet ownership / core-fringe-reject 原型 |
| 2026-06-09 | `Stream3D/scripts/reproduce_v8_lane1.sh` | 新增 Lane 1 scene0050 16f 复现脚本 | 方便后续按同一命令复现 G1 诊断 |
| 2026-06-09 | `Stream3D/scripts/reproduce_v8_lane3.sh` | 新增 Lane3 B1 最小复现脚本 | 审计包中提供当前 best method result 的复现入口 |
| 2026-06-09 | `docs/stream4d_v8_执行日志.md`、`docs/stream4d_v8_实验结果复盘.md` | 新增 v8 执行和复盘日志 | 满足本轮可审计记录要求 |

## 命令记录

### E0：GPU / Python 环境

```bash
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
```

结果摘要：

```text
GPU 6: NVIDIA RTX A5000, 22712 MiB free / 23028 MiB total
GPU 7: NVIDIA RTX A5000, 22712 MiB free / 23028 MiB total
```

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/4D/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

结果摘要：

```text
torch 2.4.0+cu118
CUDA 可用
```

### E1：Stream4D/D4RT 环境检查

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
CUDA_VISIBLE_DEVICES=6 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.check_stream4d_env \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer
```

日志：

```text
Stream4D environment check passed.
```

### E2：错误环境 G1 diagnostic（不进入有效结果表）

```text
曾用 /mnt/data/users/chengshun.wang/miniconda3/envs/4D/bin/python 跑过：
  stream4d_v8_g1_grid16_scene0050_16f
  stream4d_v8_g1_grid32_scene0050_16f

这些结果是真实落盘，但环境不符合历史 Stream4D 约定。
处理：保留文件用于排查，不写入 v8 method/sanity gate，有效 G1 需要用 loger env 重跑。
```

### E3：loger 环境基础验证

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
CUDA_VISIBLE_DEVICES=6 python - <<'PY'
import torch, sys
print(sys.executable)
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
PY
```

结果：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
torch 2.6.0+cu124
cuda=True
visible device count=1
```

```bash
python -m py_compile \
  tools/export_d4rt_grid_surfel_field_v8.py \
  stream4d/d4rt_adapter.py stream4d/scannet_stream.py \
  tools/d4rt_geometry_diagnostic.py tools/check_dynamic_replica_env.py \
  tests/test_stream4d_protocol_fixes.py \
  > logs/stream4d_v8_g1_loger_py_compile.log 2>&1

python - <<'PY' > logs/stream4d_v8_g1_loger_import_smoke.log 2>&1
import tools.export_d4rt_grid_surfel_field_v8
import stream4d.d4rt_adapter
import tools.d4rt_geometry_diagnostic
import tools.check_dynamic_replica_env
print('v8 loger import smoke OK')
PY
```

结果：

```text
py_compile: pass
import smoke: v8 loger import smoke OK
```

### E4：G1 grid16 / grid32，scene0050_00 16f，loger 环境

grid16：

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
export CUDA_VISIBLE_DEVICES=6
export MPLCONFIGDIR=/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache
python -m tools.export_d4rt_grid_surfel_field_v8 \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 16 \
  --window-size 16 \
  --window-stride 16 \
  --grid-size 16 \
  --cycle-max-tracks 128 \
  --query-chunk-size 2048 \
  --save-overlays \
  --run-name stream4d_v8_g1_grid16_scene0050_16f_loger \
  > logs/stream4d_v8_g1_grid16_scene0050_16f_loger.log 2>&1
```

grid32：

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
export CUDA_VISIBLE_DEVICES=7
export MPLCONFIGDIR=/mnt/data/users/chengshun.wang/tmp_torch/matplotlib_cache
python -m tools.export_d4rt_grid_surfel_field_v8 \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 16 \
  --window-size 16 \
  --window-stride 16 \
  --grid-size 32 \
  --cycle-max-tracks 128 \
  --query-chunk-size 4096 \
  --save-overlays \
  --run-name stream4d_v8_g1_grid32_scene0050_16f_loger \
  > logs/stream4d_v8_g1_grid32_scene0050_16f_loger.log 2>&1
```

结果摘要：

| Run | queries | valid tracks | visible obs | uv in01 | track len mean | self p90 px | cycle p90 px | 2D coverage mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `grid16_loger` | `4096` | `3979` | `34209` | `0.593902587890625` | `8.351806640625` | `6.634120941162109` | `34.77774353027344` | `0.032052040100097656` |
| `grid32_loger` | `16384` | `16150` | `141300` | `0.6092605590820312` | `8.624267578125` | `6.3304579257965115` | `38.27891311645508` | `0.12543678283691406` |

### E5：G1 Sim3 geometry diagnostic，loger 输出

```bash
python -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid16_scene0050_16f_loger \
  --seq-list splits/scannet_scene0050.txt \
  --output-prefix outputs/audit/v8_g1_grid16_scene0050_16f_loger_geometry

python -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32_scene0050_16f_loger \
  --seq-list splits/scannet_scene0050.txt \
  --output-prefix outputs/audit/v8_g1_grid32_scene0050_16f_loger_geometry
```

结果摘要：

| Run | anchors | Sim3 scale | residual median | residual p90 | residual p95 |
|---|---:|---:|---:|---:|---:|
| `grid16_loger` | `372` | `0.5183166008465534` | `0.45260965177544044` | `0.7828105450682501` | `0.8919001312807748` |
| `grid32_loger` | `432` | `0.5366248786728798` | `0.45915942109306296` | `0.7715883894991051` | `0.9256885432288109` |

### E6：OpenD4RT grid margin 修复尝试

原因：

```text
OpenD4RT/infer_track_3d.py 的 `_grid_query_points` 使用 margin_ratio=0.02。
初始 G1 grid 命中图像边缘，uv_in01_rate 低可能与边缘 query/preprocessing 有关。
因此新增 `--grid-margin-ratio` 并重跑 grid32 margin 0.02。
```

命令：

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
export CUDA_VISIBLE_DEVICES=6
python -m tools.export_d4rt_grid_surfel_field_v8 \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 16 \
  --window-size 16 \
  --window-stride 16 \
  --grid-size 32 \
  --grid-margin-ratio 0.02 \
  --cycle-max-tracks 128 \
  --query-chunk-size 4096 \
  --save-overlays \
  --run-name stream4d_v8_g1_grid32m002_scene0050_16f_loger \
  > logs/stream4d_v8_g1_grid32m002_scene0050_16f_loger.log 2>&1

python -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_scene0050_16f_loger \
  --seq-list splits/scannet_scene0050.txt \
  --output-prefix outputs/audit/v8_g1_grid32m002_scene0050_16f_loger_geometry
```

结果：

```text
uv_in01_rate: 0.6092605590820312 -> 0.6279640197753906
track_length_visible_mean: 8.624267578125 -> 8.94482421875
self_uv_error_p90: 6.3304579257965115 -> 6.181506586074829
cycle_uv_error_p90: 38.27891311645508 -> 39.24992179870606
Sim3 residual median: 0.45915942109306296 -> 0.4602868233077916
```

判断：

```text
margin 修复对 uv/track length/self error 有小幅正向，但仍未达到 Lane 1 gate。
cycle error 未改善，Sim3 residual 仍高于 0.30m。
```

### E7：官方 infer_track_3d.py 入口检查

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
python ../Open-d4rt/infer_track_3d.py --help > logs/stream4d_v8_official_infer_track_help_loger.log 2>&1 || true
```

结果：

```text
logs/stream4d_v8_official_infer_track_help_loger.log size = 0 bytes。
`infer_track_3d.py` 没有 argparse/CLI main；它是 helper module。
后续若继续 adapter-vs-official sanity，需要按其 helper API 写对比脚本。
```

### E8：Dynamic Replica 环境检查，loger

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
python -m tools.check_dynamic_replica_env \
  --root data/dynamic-replica/v2 \
  --output outputs/audit/dynamic_replica_env_v8_loger.md \
  > logs/stream4d_v8_dynamic_replica_env_loger.log 2>&1
```

结果：

```text
data_root_exists=True
split_dir_exists=True
annotation_exists=True
scene_count=20
usable_scene_count=0
can_report_official_instance_tracking=False
can_report_d4rt_trajectory_metrics=False
```

### E9：loger unittest

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
python -m unittest -v tests.test_stream4d_protocol_fixes \
  > logs/stream4d_v8_loger_unit_tests_unittest.log 2>&1
```

结果：

```text
Ran 13 tests in 0.125s
OK
```

### E10：adapter-vs-official helper 对比，loger

目的：

```text
排查 Lane 1 初始 stride-10 失败是否来自 D4RTAdapter query 顺序 / helper API 偏差。
该诊断不读 GT，不报告 AP，manifest 字段：
diagnostic_only=True, uses_gt=False, is_method_result=False。
```

命令：

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
export CUDA_VISIBLE_DEVICES=6
cd Stream3D
python -m py_compile \
  tools/compare_d4rt_adapter_official_v8.py \
  tools/export_d4rt_grid_surfel_field_v8.py \
  stream4d/d4rt_adapter.py \
  > logs/stream4d_v8_adapter_official_compare_py_compile.log 2>&1

python -m tools.compare_d4rt_adapter_official_v8 \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 16 \
  --grid-size 4 \
  --grid-margin-ratio 0.02 \
  --query-chunk-size 1024 \
  --output-prefix outputs/audit/v8_adapter_vs_official_scene0050_grid4_loger \
  > logs/stream4d_v8_adapter_vs_official_scene0050_grid4_loger.log 2>&1
```

结果：

| Diff | mean abs | p90 abs | max abs |
|---|---:|---:|---:|
| `uv` | `3.855219983961433e-05` | `8.618831634521484e-05` | `0.0006253048777580261` |
| `xyz_ref0` | `0.00034643031540326774` | `0.0007100462913513185` | `0.002689838409423828` |
| `visibility_prob` | `0.00015662208897992969` | `0.00039833784103393555` | `0.0204317569732666` |
| `confidence_prob` | `3.018067218363285e-08` | `1.1920928955078125e-07` | `1.5497207641601562e-06` |

判断：

```text
D4RTAdapter 和 OpenD4RT helper 在同一 clip / query 上几乎一致。
Lane 1 初始失败不是 adapter helper API/order 差异导致。
```

### E11：连续 16 帧修复尝试，scene0050

原因：

```text
初始 G1 使用 frame_stride=10，把 ScanNet frames 0..150 压成 16-frame clip。
D4RT 是 video correspondence model，stride-10 可能破坏 temporal scale。
连续帧缺 Cropformer mask，因此新增 --allow-missing-masks；
缺失 mask 只在 Lane 1 source metadata 中置 0，不用于 GT 或 AP。
```

命令：

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
export CUDA_VISIBLE_DEVICES=6
cd Stream3D
python -m py_compile \
  stream4d/scannet_stream.py \
  tools/export_d4rt_grid_surfel_field_v8.py \
  tools/compare_d4rt_adapter_official_v8.py \
  > logs/stream4d_v8_stride1_missing_mask_fix_py_compile.log 2>&1

python -m tools.export_d4rt_grid_surfel_field_v8 \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 1 \
  --max-frames 16 \
  --window-size 16 \
  --window-stride 16 \
  --grid-size 32 \
  --grid-margin-ratio 0.02 \
  --allow-missing-masks \
  --cycle-max-tracks 128 \
  --query-chunk-size 4096 \
  --save-overlays \
  --run-name stream4d_v8_g1_grid32m002_scene0050_16f_stride1_loger \
  > logs/stream4d_v8_g1_grid32m002_scene0050_16f_stride1_loger.log 2>&1

python -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_scene0050_16f_stride1_loger \
  --seq-list splits/scannet_scene0050.txt \
  --output-prefix outputs/audit/v8_g1_grid32m002_scene0050_16f_stride1_loger_geometry \
  > logs/stream4d_v8_g1_grid32m002_scene0050_16f_stride1_loger_geometry.log 2>&1
```

结果：

| Run | uv in01 | track len mean | self p90 px | cycle p90 px | coverage mean | Sim3 median | Sim3 p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `grid32m002 scene0050 stride10` | `0.6279640197753906` | `8.94482421875` | `6.181506586074829` | `39.24992179870606` | `0.12944412231445312` | `0.4602868233077916` | `0.7802417144670429` |
| `grid32m002 scene0050 stride1` | `0.9908294677734375` | `15.771484375` | `1.3851011872291565` | `2.381803274154663` | `0.16101455688476562` | `0.2619269225708374` | `0.4944150956084864` |

判断：

```text
连续帧修复显著解决 uv/cycle blocker。
初始 stride-10 G1 不能作为 D4RT surfel field 失败证据。
```

### E12：连续 16 帧 G1 扩到 probe5

命令：

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
export CUDA_VISIBLE_DEVICES=6
cd Stream3D
python -m tools.export_d4rt_grid_surfel_field_v8 \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet_v6_probe5.txt \
  --backbone Cropformer \
  --frame-stride 1 \
  --max-frames 16 \
  --window-size 16 \
  --window-stride 16 \
  --grid-size 32 \
  --grid-margin-ratio 0.02 \
  --allow-missing-masks \
  --cycle-max-tracks 128 \
  --query-chunk-size 4096 \
  --save-overlays \
  --run-name stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  > logs/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger.log 2>&1

python -m tools.d4rt_geometry_diagnostic \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-prefix outputs/audit/v8_g1_grid32m002_probe5_16f_stride1_loger_geometry \
  > logs/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger_geometry.log 2>&1
```

结果摘要：

| Metric | mean | min | max |
|---|---:|---:|---:|
| `uv_in01_rate` | `0.9858451843261719` | `0.9592514038085938` | `0.9998092651367188` |
| `track_length_visible_mean` | `13.284716796875` | `9.64019775390625` | `15.98291015625` |
| `track_length_visible_p10` | `7.4` | `1.0` | `16.0` |
| `self_uv_error_p90` | `1.57081866979599` | `1.3851011872291565` | `1.6558191418647772` |
| `cycle_uv_error_p90` | `3.2913891315460204` | `2.381803274154663` | `4.192905950546264` |
| `coverage mean` | `0.13198394775390626` | `0.1008310317993164` | `0.16101455688476562` |
| `Sim3 residual median` | `0.46820781478265117` | `0.2619269225708374` | `0.6804238543343015` |
| `Sim3 residual p90` | `0.8595804531797114` | `0.4944150956084864` | `1.1890475588672236` |

判断：

```text
probe5 连续帧 G1 通过 uv/cycle/track-length 最低 gate。
但 Sim3 median mean=0.46820781478265117，仍大于 0.30m；
ScanNet mesh AP 只能作为 diagnostic，不可作为 D4RT metric geometry claim。
```

### E13：Lane 2 mask measurement coverage diagnostic

命令：

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
cd Stream3D
python -m py_compile \
  tools/diagnose_v8_mask_measurement_coverage.py \
  stream4d/mask_evidence.py \
  stream4d/carrier_store.py \
  > logs/stream4d_v8_mask_measurement_coverage_py_compile.log 2>&1

python -m tools.diagnose_v8_mask_measurement_coverage \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --backbone Cropformer \
  --rho-min 0.35 \
  --output-prefix outputs/audit/v8_mask_measurement_coverage_probe5_stride1_loger \
  > logs/stream4d_v8_mask_measurement_coverage_probe5_stride1_loger.log 2>&1
```

结果摘要：

```text
diagnostic_only=True
uses_gt=False
is_method_result=False
num_ok_windows=5
num_mask_frames_available_mean=2.0
num_mask_frames_missing_mean=14.0
carrier_assignment_rate_all_frames_mean=0.12105091419117497
carrier_assignment_rate_available_mask_frames_mean=0.9806140838500657
surfel_positive_observation_rate_mean=0.91259765625
mean_positive_observations_per_surfel_mean=1.72508544921875
```

判断：

```text
连续 16 帧 D4RT clip 中，Cropformer mask 只在 frame 0 和 10 可用。
有 mask 的帧 assignment 很高，但 observation 在时间上很稀疏。
Lane 3 若继续，必须按 sparse mask measurement 设计，不能假装有 dense 16-frame mask stream。
```

### E14：最终验证与审计包

最终验证：

```bash
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
cd Stream3D
python -m py_compile \
  stream4d/scannet_stream.py \
  tools/export_d4rt_grid_surfel_field_v8.py \
  tools/compare_d4rt_adapter_official_v8.py \
  tools/diagnose_v8_mask_measurement_coverage.py \
  stream4d/d4rt_adapter.py \
  stream4d/mask_evidence.py \
  tests/test_stream4d_protocol_fixes.py \
  > logs/stream4d_v8_final_py_compile_loger.log 2>&1

bash -n scripts/reproduce_v8_lane1.sh \
  > logs/stream4d_v8_reproduce_script_bash_n.log 2>&1

python -m unittest -v tests.test_stream4d_protocol_fixes \
  > logs/stream4d_v8_final_unit_tests_loger.log 2>&1
```

结果：

```text
final py_compile loger: pass
reproduce script bash -n: pass
unittest loger: Ran 13 tests in 0.121s, OK
```

审计包：

```text
packet:
  stream4d_v8_code_audit_packet_20260609_0440_lane1_final.zip

sha256:
  see sibling file stream4d_v8_code_audit_packet_20260609_0440_lane1_final.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

filelist:
  stream4d_v8_code_audit_packet_20260609_0440_lane1_final_filelist.txt

zip test:
  see stream4d_v8_code_audit_packet_20260609_0440_lane1_final_ziptest.log

file count:
  216
```

### E15：Lane3 A/B/C surfel object field prototype

纠正记录：

```text
此前日志/复盘只完成 Lane1/Lane2，没有继续执行计划第 7 节 Lane3。
用户指出后，继续实现并运行 Lane3 lightweight prototypes。
所有命令继续固定使用 loger 环境：
  PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python
  PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
```

新增工具验证：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
export PATH=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin:$PATH
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python

$PY -m py_compile tools/export_v8_surfel_object_field.py \
  > logs/stream4d_v8_lane3a_py_compile.log 2>&1

$PY - <<'PY' > logs/stream4d_v8_lane3a_import_smoke.log 2>&1
import tools.export_v8_surfel_object_field
print('v8 lane3a import smoke OK')
PY
```

结果：

```text
py_compile: pass
import smoke: v8 lane3a import smoke OK
```

Scene0050 A1 smoke：

```bash
export CUDA_VISIBLE_DEVICES=6
$PY -m tools.export_v8_surfel_object_field \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_scene0050.txt \
  --output-config stream4d_v8_a1_signed_history_m2_scene0050 \
  --min-observations 2 \
  --min-carriers 16 \
  --min-owned-masks 1 \
  --max-masks-per-object 2 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 50000 \
  --min-points-per-object 20 \
  --summary-root outputs/v8_surfel_object_field \
  > logs/stream4d_v8_a1_signed_history_m2_scene0050_export.log 2>&1

$PY -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v8_a1_signed_history_m2_scene0050_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --output_file data/evaluation/scannet/stream4d_v8_a1_signed_history_m2_scene0050_class_agnostic.txt \
  --require-manifest \
  > logs/stream4d_v8_a1_signed_history_m2_scene0050_eval.log 2>&1
```

结果：

```text
AP/AP50/AP25 = 0.3835978835978835 / 0.7142857142857142 / 0.7142857142857142
objects = 15
pre_points = 9654
export_conflict_rate = 0.08856432566811684
ownership_competing_masks = 32
ownership_dropped_claims = 93
```

Probe5 A1：

```bash
$PY -m tools.export_v8_surfel_object_field \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v8_a1_signed_history_m2_probe5 \
  --min-observations 2 \
  --min-carriers 16 \
  --min-owned-masks 1 \
  --max-masks-per-object 2 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 50000 \
  --min-points-per-object 20 \
  --summary-root outputs/v8_surfel_object_field \
  > logs/stream4d_v8_a1_signed_history_m2_probe5_export.log 2>&1

$PY -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v8_a1_signed_history_m2_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --output_file data/evaluation/scannet/stream4d_v8_a1_signed_history_m2_probe5_class_agnostic.txt \
  --require-manifest \
  > logs/stream4d_v8_a1_signed_history_m2_probe5_eval.log 2>&1
```

Probe5 A2：

```bash
$PY -m tools.export_v8_surfel_object_field \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v8_a2_signed_history_m1_probe5 \
  --min-observations 1 \
  --min-carriers 16 \
  --min-owned-masks 1 \
  --max-masks-per-object 2 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 50000 \
  --min-points-per-object 20 \
  --summary-root outputs/v8_surfel_object_field \
  > logs/stream4d_v8_a2_signed_history_m1_probe5_export.log 2>&1

$PY -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v8_a2_signed_history_m1_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --output_file data/evaluation/scannet/stream4d_v8_a2_signed_history_m1_probe5_class_agnostic.txt \
  --require-manifest \
  > logs/stream4d_v8_a2_signed_history_m1_probe5_eval.log 2>&1
```

Probe5 A1 stride1 export check：

```bash
$PY -m tools.export_v8_surfel_object_field \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v8_a1s1_signed_history_m2_probe5 \
  --min-observations 2 \
  --min-carriers 16 \
  --min-owned-masks 1 \
  --max-masks-per-object 2 \
  --export-mask-sample-stride 1 \
  --export-mask-max-pixels 0 \
  --min-points-per-object 20 \
  --summary-root outputs/v8_surfel_object_field \
  > logs/stream4d_v8_a1s1_signed_history_m2_probe5_export.log 2>&1

$PY -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v8_a1s1_signed_history_m2_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --output_file data/evaluation/scannet/stream4d_v8_a1s1_signed_history_m2_probe5_class_agnostic.txt \
  --require-manifest \
  > logs/stream4d_v8_a1s1_signed_history_m2_probe5_eval.log 2>&1
```

工具标记修复：

```text
问题：
  A/B/C 共用 export_v8_surfel_object_field.py。
  初版 summary/manifest 对所有 configs 都硬编码为 Lane3A，容易误导审计。

修复：
  新增 --prototype-direction：
    A_signed_history
    B_surfacelet_singlemask
    C_core_fringe_reject

验证：
  logs/stream4d_v8_surfel_object_field_py_compile_r2.log: pass
  logs/stream4d_v8_surfel_object_field_import_smoke_r2.log: pass
```

Probe5 B1 / C1 并行执行：

```bash
# B1 on GPU6
export CUDA_VISIBLE_DEVICES=6
$PY -m tools.export_v8_surfel_object_field \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v8_b1_surfacelet_singlemask_probe5 \
  --prototype-direction B_surfacelet_singlemask \
  --min-observations 1 \
  --max-observations 1 \
  --min-carriers 16 \
  --min-owned-masks 1 \
  --max-masks-per-object 1 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 50000 \
  --min-points-per-object 20 \
  --summary-root outputs/v8_surfel_object_field \
  > logs/stream4d_v8_b1_surfacelet_singlemask_probe5_export.log 2>&1

$PY -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v8_b1_surfacelet_singlemask_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --output_file data/evaluation/scannet/stream4d_v8_b1_surfacelet_singlemask_probe5_class_agnostic.txt \
  --require-manifest \
  > logs/stream4d_v8_b1_surfacelet_singlemask_probe5_eval.log 2>&1

# C1 on GPU7
export CUDA_VISIBLE_DEVICES=7
$PY -m tools.export_v8_surfel_object_field \
  --debug-root outputs/v8_d4rt_grid_surfel_field/stream4d_v8_g1_grid32m002_probe5_16f_stride1_loger \
  --seq-list splits/scannet_v6_probe5.txt \
  --output-config stream4d_v8_c1_core_owned2_probe5 \
  --prototype-direction C_core_fringe_reject \
  --min-observations 2 \
  --min-carriers 16 \
  --min-owned-masks 2 \
  --max-masks-per-object 2 \
  --export-mask-sample-stride 2 \
  --export-mask-max-pixels 50000 \
  --min-points-per-object 20 \
  --summary-root outputs/v8_surfel_object_field \
  > logs/stream4d_v8_c1_core_owned2_probe5_export.log 2>&1

$PY -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_v8_c1_core_owned2_probe5_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --output_file data/evaluation/scannet/stream4d_v8_c1_core_owned2_probe5_class_agnostic.txt \
  --require-manifest \
  > logs/stream4d_v8_c1_core_owned2_probe5_eval.log 2>&1
```

Lane3 probe5 results：

| Config | AP | AP50 | AP25 | objects/scene | pre_points/scene | conflict |
|---|---:|---:|---:|---:|---:|---:|
| `stream4d_v8_a1_signed_history_m2_probe5` | `0.31800136425080006` | `0.5499198184432726` | `0.8688954479508236` | `15.2` | `9482.8` | `0.09091808474220602` |
| `stream4d_v8_a2_signed_history_m1_probe5` | `0.3060221611896139` | `0.5496301734726751` | `0.888005513856092` | `17.2` | `9733.8` | `0.11135387397243113` |
| `stream4d_v8_b1_surfacelet_singlemask_probe5` | `0.32843947812986807` | `0.6292662056580957` | `0.8843628978668244` | `16.4` | `8513.2` | `0.08430650606572185` |
| `stream4d_v8_c1_core_owned2_probe5` | `0.31583318558955015` | `0.5295522312291147` | `0.8468851431724259` | `13.6` | `9218.4` | `0.0820415825059075` |
| `stream4d_v8_a1s1_signed_history_m2_probe5` | `0.29073598846418347` | `0.5581574610318252` | `0.8679909541120018` | `15.2` | `10302.6` | `0.11975014288387284` |

B1 per-scene eval：

```bash
CFG=stream4d_v8_b1_surfacelet_singlemask_probe5
SRC_PRED=data/prediction/${CFG}_class_agnostic
LOG=logs/stream4d_v8_b1_surfacelet_singlemask_probe5_per_scene_eval.log
: > "$LOG"
while IFS= read -r scene; do
  [ -n "$scene" ] || continue
  SCENE_DIR="data/prediction/${CFG}_${scene}_class_agnostic"
  mkdir -p "$SCENE_DIR"
  ln -sf "$(realpath "$SRC_PRED/config_manifest.json")" "$SCENE_DIR/config_manifest.json"
  ln -sf "$(realpath "$SRC_PRED/${scene}.npz")" "$SCENE_DIR/${scene}.npz"
  "$PY" -m evaluation.evaluate \
    --pred_path "$SCENE_DIR" \
    --gt_path data/scannet/gt \
    --dataset scannet \
    --output_file "data/evaluation/scannet/${CFG}_${scene}_class_agnostic.txt" \
    --tmp_root data/TMP \
    --tmp_config "$CFG" \
    --no_class \
    --require-manifest 2>&1 | tee -a "$LOG"
done < splits/scannet_v6_probe5.txt
```

Per-scene result：

| Scene | AP | AP50 | AP25 | objects | pre_points | conflict |
|---|---:|---:|---:|---:|---:|---:|
| `scene0050_00` | `0.40079365079365076` | `0.7142857142857142` | `0.7142857142857142` | `14` | `8593` | `0.06237635284533923` |
| `scene0011_00` | `0.2904170675004008` | `0.7850108225108225` | `0.9052015692640693` | `12` | `5980` | `0.1588628762541806` |
| `scene0030_00` | `0.5142857142857142` | `0.5547619047619048` | `0.857142857142857` | `7` | `10790` | `0.026042632066728452` |
| `scene0081_01` | `0.2457010582010582` | `0.7331349206349207` | `0.9603174603174605` | `10` | `8912` | `0.1099640933572711` |
| `scene0591_00` | `0.2875921917588584` | `0.48261002886002885` | `0.972020202020202` | `39` | `8291` | `0.06428657580508985` |

Metric audit：

```bash
CONFIGS=stream4d_v8_a1_signed_history_m2_probe5,stream4d_v8_a2_signed_history_m1_probe5,stream4d_v8_b1_surfacelet_singlemask_probe5,stream4d_v8_c1_core_owned2_probe5,stream4d_v8_a1s1_signed_history_m2_probe5

$PY -m tools.scan_reportable_configs \
  --configs "$CONFIGS" \
  --require-manifest \
  --output outputs/audit/v8_reportable_config_scan_lane3abc_probe5.md \
  > logs/stream4d_v8_reportable_config_scan_lane3abc_probe5_r2.log 2>&1

$PY -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs stream4d_v8_b1_surfacelet_singlemask_probe5 \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v8_metric_integrity_b1_surfacelet_singlemask_probe5.md \
  --backbone Cropformer \
  --require-manifest \
  > logs/stream4d_v8_metric_integrity_b1_surfacelet_singlemask_probe5_r2.log 2>&1
```

注意：

```text
第一次误把 --output 设成 .json。
scan_reportable_configs/verify_stream4d_metric_integrity 会把 Markdown 写到 --output，
JSON 写到 output.with_suffix(".json")；当 --output 已经是 .json 时会被 Markdown 覆盖。
已用 .md output 重跑，最终 .md/.json/.csv 均存在且 JSON 可解析。
```

审计结果：

```text
reportable scan:
  num_configs=5
  num_configs_missing_manifest=0
  num_diagnostic_only_configs=0
  num_oracle_configs=0
  num_reportable_method_configs=5
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0

B1 metric integrity:
  phase0_pass=True
  evaluator_ap_core_equal_by_hash=True
  gt_files_read_by_rescore=False
  num_configs_missing_manifest=0
  num_oracle_configs=0
  num_reportable_method_configs=1
  num_suspicious_configs=0
  num_uses_gt_and_method_result=0
  mean_pre_points_ratio=0.03986074960713631
  mean_prediction_union_ratio=0.03986074960713631
  pre_points_policy={"recompute_like": 5}
  object_dict_pred_alignment_mean_iou=1.0
  object_dict_pred_alignment_min_iou=1.0
  object_dict_pred_alignment_failed_instances=0
```

最终验证：

```bash
$PY -m py_compile \
  tools/export_d4rt_grid_surfel_field_v8.py \
  tools/compare_d4rt_adapter_official_v8.py \
  tools/diagnose_v8_mask_measurement_coverage.py \
  tools/export_v8_surfel_object_field.py \
  stream4d/scannet_stream.py \
  stream4d/export_scannet.py \
  tools/prediction_manifest.py \
  tools/verify_stream4d_metric_integrity.py \
  tools/scan_reportable_configs.py \
  evaluation/evaluate.py \
  tests/test_stream4d_protocol_fixes.py \
  > logs/stream4d_v8_lane3abc_final_py_compile.log 2>&1

$PY - <<'PY' > logs/stream4d_v8_lane3abc_final_import_smoke.log 2>&1
import importlib
for name in [
    'tools.export_d4rt_grid_surfel_field_v8',
    'tools.compare_d4rt_adapter_official_v8',
    'tools.diagnose_v8_mask_measurement_coverage',
    'tools.export_v8_surfel_object_field',
    'stream4d.scannet_stream',
    'stream4d.export_scannet',
]:
    importlib.import_module(name)
    print(f'{name} OK')
PY

$PY -m unittest tests.test_stream4d_protocol_fixes \
  > logs/stream4d_v8_lane3abc_final_unit_tests.log 2>&1
```

结果：

```text
final py_compile: pass
final import smoke: pass
final unittest: Ran 13 tests in 0.129s, OK
reproduce_v8_lane3.sh bash -n: pass
```

Lane3 审计包：

```text
packet:
  stream4d_v8_code_audit_packet_20260609_0503_lane3_probe5_final.zip

sha256:
  see sibling file stream4d_v8_code_audit_packet_20260609_0503_lane3_probe5_final.sha256
  （不嵌入本文档，避免 zip 自引用 hash）

filelist:
  stream4d_v8_code_audit_packet_20260609_0503_lane3_probe5_final_filelist.txt

zip test:
  No errors detected in compressed data of stream4d_v8_code_audit_packet_20260609_0503_lane3_probe5_final.zip.

file count:
  273
```
