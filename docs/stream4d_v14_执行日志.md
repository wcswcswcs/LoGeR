# Stream4D v14 执行日志

日期: 2026-06-09  
计划文档: `docs/stream4d_v14_global_object_explanation_plan_for_codex.md`  
复盘文档: `docs/stream4d_v14_实验结果复盘.md`  
GPU: 用户说明显卡 6/7 可用；本轮命令默认使用 `CUDA_VISIBLE_DEVICES=6,7`。  
Python: `/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python`

## 0. 执行边界

- 不使用 GT 生成 prediction。
- GT 只用于 evaluation / diagnostic / oracle upper bound。
- oracle、failure decomposition、cross-support diagnostic 不进入 method table。
- 若 Phase 2 broad-support atom oracle gate 不通过，不进入 Phase 3/4 solver method claim。
- 不改写 `nan`、空预测或 oracle 上界为有利数字。
- 初始 worktree 已经很脏，本轮不回滚用户/历史改动。

## 1. 环境和初始盘点

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

环境排错:

```bash
python -V
python3 -m unittest discover tests
```

结果:

```text
python: command not found
system python3 unittest failed: ModuleNotFoundError: numpy
```

修复: 使用 loger conda env:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -V
```

结果:

```text
Python 3.11.15
numpy 1.26.4
scipy 1.17.1
open3d 0.19.0
```

输入 artifact 盘点:

```bash
find Stream3D/outputs/v12_measurement_bank -maxdepth 2 -name measurement_bank.npz | wc -l
find Stream3D/outputs/v13_masklet_measurements/C3 -maxdepth 2 -name masklets.npz
```

结果:

```text
v12 measurement_bank.npz: 5 scenes
v13 C3 masklets.npz: 5 scenes
```

## 2. Phase 0 审计与基线矩阵

最终测试命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile stream4d/*.py tools/*.py evaluation/*.py tests/*.py > outputs/audit/v14_logs/py_compile_final.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests -p '*pure*.py' > outputs/audit/v14_logs/unittest_pure_final.log 2>&1
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m unittest discover tests > outputs/audit/v14_logs/unittest_all_final.log 2>&1
```

结果:

```text
py_compile_final.log: empty, exit 0
pure unittest: Ran 7 tests in 0.006s ... OK
all unittest: Ran 27 tests in 1.426s ... OK
```

Phase 0 baseline matrix:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.summarize_v9_unified_eval \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --matrix-json scripts/v14_phase0_baseline_matrix_probe5.json \
  --output-prefix outputs/audit/v14_phase0/baseline_matrix_probe5 \
  --stream3d-config scannet
```

输出:

- `Stream3D/outputs/audit/v14_phase0/baseline_matrix_probe5.json`
- `Stream3D/outputs/audit/v14_phase0/baseline_matrix_probe5.csv`
- `Stream3D/outputs/audit/v14_phase0/baseline_matrix_probe5.md`

关键 baseline:

| row | AP/AP50/AP25 | pre% | conflict | best IoU |
|---|---:|---:|---:|---:|
| P0 Stream3D on S0 | 0.235730/0.414306/0.537786 | 84.6744 | 0.2213 | 0.6308 |
| P0 Stream3D on S1 | 0.399213/0.597171/0.742535 | 4.5145 | 0.2213 | 0.7370 |
| B1 own | 0.328439/0.629266/0.884363 | 3.9861 | 8.4307 | 0.6914 |
| O38 own | 0.081038/0.219225/0.492501 | 66.6809 | 3.9025 | 0.4611 |
| M13c own | 0.224575/0.419119/0.781728 | 4.3855 | 55.8958 | 0.6576 |
| M13d own | 0.161109/0.427857/0.793144 | 2.0522 | 0.0000 | 0.5991 |
| C_surfel own | 0.228316/0.460285/0.778069 | 4.2916 | 52.7632 | 0.6522 |

Final reportable scan:

```bash
cd Stream3D
CONFIGS=$(paste -sd, outputs/audit/v14_final/configs_v14_final.txt)
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.scan_reportable_configs \
  --root . \
  --configs "$CONFIGS" \
  --output outputs/audit/v14_final/reportable_config_scan_v14_final.md \
  --require-manifest \
  --require-eval-policy
```

结果:

```text
num_configs=61
num_configs_missing_eval_policy=0
num_configs_missing_manifest=0
num_diagnostic_only_configs=57
num_oracle_configs=13
num_reportable_method_configs=4
num_suspicious_configs=0
num_uses_gt_for_prediction=0
num_uses_gt_and_method_result=0
```

Final metric integrity:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.verify_stream4d_metric_integrity \
  --orig-stream3d-root . \
  --current-root . \
  --configs "$CONFIGS" \
  --seq-list splits/scannet_v6_probe5.txt \
  --output outputs/audit/v14_final/metric_integrity_v14_final.md \
  --require-manifest
```

结果:

```text
phase0_pass=True
all_ap_core_equal=True
gt_files_read_by_rescore=False
configs=61
```

## 3. Phase 1 Failure Decomposition

命令:

```bash
cd Stream3D
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v14_failure_decomposition \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --source B1:stream4d_v8_b1_surfacelet_singlemask_probe5 \
  --source O38:stream4d_v9_o38_o22_memory_c055_newpts_logarea_probe5:stream4d_v11_oracle_c2_o38_memory_probe5 \
  --source C_mask:stream4d_v13_c_mask_unsup_probe5:stream4d_v13_oracle_c_mask_probe5 \
  --source C_regionlet:stream4d_v13_c_regionlet_unsup_probe5:stream4d_v13_oracle_c_regionlet_probe5 \
  --source C_surfel:stream4d_v13_c_surfel_unsup_probe5:stream4d_v13_oracle_c_surfel_probe5 \
  --source C_masklet:stream4d_v13_c3_masklet_candidate_probe5:stream4d_v13_oracle_c_masklet_probe5 \
  --source C_hybrid:stream4d_v13_c_hybrid_unsup_probe5:stream4d_v13_oracle_c_hybrid_probe5 \
  --source M13c:stream4d_v13_m13c_mdl_c3_fullmask_probe5 \
  --source M13d:stream4d_v13_m13d_mdl_c3_posterior_wta_probe5 \
  --method M13c:stream4d_v13_m13c_mdl_c3_fullmask_probe5 \
  --method M13d:stream4d_v13_m13d_mdl_c3_posterior_wta_probe5 \
  --output-prefix outputs/audit/v14_failure_decomposition/failure_decomposition_probe5 \
  --visual-dir outputs/audit/v14_failure_decomposition/visuals \
  --visual-limit 30 \
  > outputs/audit/v14_logs/v14_failure_decomposition.log 2>&1
```

输出:

- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5.json`
- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5.md`
- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5_source_gt.csv`
- `Stream3D/outputs/audit/v14_failure_decomposition/failure_decomposition_probe5_method_gt.csv`
- `Stream3D/outputs/audit/v14_failure_decomposition/visuals/failure_visuals_manifest.json`
- 30 个 visual PNG/JSON sidecar

关键结果:

| source | AP/AP50/AP25 | oracle AP/AP50/AP25 | support% | best IoU | no/weak/good/high |
|---|---:|---:|---:|---:|---:|
| B1 | 0.328439/0.629266/0.884363 | NA | 3.9861 | 0.0260 | 217/12/6/1 |
| O38 | 0.081038/0.219225/0.492501 | 0.223210/0.444444/0.635556 | 66.6809 | 0.3000 | 71/35/73/57 |
| C_mask | 0.058281/0.161357/0.343526 | 0.224691/0.453333/0.648889 | 60.8842 | 0.2921 | 73/35/75/53 |
| C_regionlet | 0.045679/0.122830/0.266596 | 0.338574/0.613208/0.829643 | 18.5455 | 0.0959 | 162/33/37/4 |
| C_surfel | 0.228316/0.460285/0.778069 | 0.395062/0.750000/0.993360 | 4.2916 | 0.0265 | 217/10/8/1 |
| C_masklet | 0.062802/0.267185/0.517357 | 0.183908/0.551724/0.926056 | 2.1988 | 0.0104 | 228/7/1/0 |
| C_hybrid | 0.023515/0.066350/0.133871 | 0.256256/0.495495/0.702512 | 52.8088 | 0.2756 | 67/47/79/43 |
| M13c | 0.224575/0.419119/0.781728 | NA | 4.3855 | 0.0270 | 217/10/8/1 |
| M13d | 0.161109/0.427857/0.793144 | NA | 2.0522 | 0.0100 | 228/8/0/0 |

Final method attribution:

| method | support% | pool IoU | method IoU | selected_good | filtered_good | assignment | boundary | weak | no |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M13c | 4.3855 | 0.3231 | 0.0270 | 1 | 52 | 7 | 83 | 37 | 56 |
| M13d | 2.0522 | 0.3231 | 0.0100 | 0 | 56 | 4 | 83 | 37 | 56 |

Pool best source counts:

```text
O38: 416
C_regionlet: 50
M13c: 2
B1: 4
```

## 4. 代码修改

新增/修改:

- `Stream3D/tools/diagnose_v14_failure_decomposition.py`
  - GT-read-only diagnostic。
  - 写明 `uses_gt_for_prediction=false`、`uses_gt_for_diagnostic=true`、`is_method_result=false`、`is_diagnostic_only=true`。
- `Stream3D/stream4d/surfel_atom_bank.py`
  - 新增 `SurfelAtomBank`。
  - 支持 A0-A4 atom split。
  - 保存 atom id、surfel indices、mask/frame hist、RGB、trajectory descriptor、boundary/negative/entropy/variance、neighbors。
  - 追加 `base_mode={source,source_or_target,target_dominant}`，用于 blocker repair，只使用 predicted 2D masks，不读 GT。
- `Stream3D/tools/build_v14_surfel_atom_bank.py`
  - 构建 atom bank 并导出 diagnostic candidate prediction。
  - manifest 标记为 diagnostic-only。
- `Stream3D/tools/diagnose_v14_atom_oracle.py`
  - 调用 GT-only oracle upper bound。
  - 修复 support 度量: gate 使用 actual `data/TMP/<config>/*_pre_points.npy` 占 scene vertices 的 pre%，raw atom-known support 单独记录。
  - 修复 repair variant summary lookup，可跨 `atom-summary-root/*/<candidate>_summary.json` 查找。
  - 修复 high-candidate 字段读取 `mean_gt_best_iou_ge_0p5`。
- `Stream3D/scripts/reproduce_v14_*.sh`
  - 新增 Phase0、Phase1、Phase2、Final audit 复现脚本。

## 5. Phase 2 默认 Atom Primitive

默认 A0-A4 命令模板:

```bash
cd Stream3D
for V in A0 A1 A2 A3 A4; do
  L=$(printf '%s' "$V" | tr 'A-Z' 'a-z')
  CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.build_v14_surfel_atom_bank \
    --seq-list splits/scannet_v6_probe5.txt \
    --variant "$V" \
    --output-config "stream4d_v14_${L}_atom_candidate_probe5" \
    --atom-root outputs/v14_surfel_atom_bank \
    --summary-root outputs/v14_surfel_atom_bank \
    --min-surfels 4 \
    --min-export-surfels 4 \
    --min-export-points-per-object 20 \
    --export-enable-wta
done
```

Oracle:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.diagnose_v14_atom_oracle \
  --root . \
  --seq-list splits/scannet_v6_probe5.txt \
  --variant A0:stream4d_v14_a0_atom_candidate_probe5 \
  --variant A1:stream4d_v14_a1_atom_candidate_probe5 \
  --variant A2:stream4d_v14_a2_atom_candidate_probe5 \
  --variant A3:stream4d_v14_a3_atom_candidate_probe5 \
  --variant A4:stream4d_v14_a4_atom_candidate_probe5 \
  --summary-root outputs/audit/v14_atom_oracle \
  --atom-summary-root outputs/v14_surfel_atom_bank
```

Stream3D-on-atom support:

```bash
for V in A0 A1 A2 A3 A4; do
  L=$(printf '%s' "$V" | tr 'A-Z' 'a-z')
  CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.evaluate_cross_prepoints \
    --root . \
    --seq-list splits/scannet_v6_probe5.txt \
    --pred-config scannet \
    --source-pre-points-config scannet \
    --pre-points-config "stream4d_v14_${L}_atom_candidate_probe5" \
    --output-config "stream4d_v14_p0_on_${L}_atom_candidate_probe5" \
    --dataset scannet \
    --gt-root data/scannet/gt \
    --no-class \
    --output-file "data/evaluation/scannet/stream4d_v14_p0_on_${L}_atom_candidate_probe5_class_agnostic.txt" \
    --audit-root outputs/audit/v14_atom_support \
    --require-manifest \
    --allow-diagnostic-eval \
    --eval-policy stream3d_on_v14_atom_support
done
```

默认结果:

| row | candidate AP/AP50/AP25 | oracle AP/AP50/AP25 | Stream3D-on-support | pre% | atom known% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| A0 | 0.055958/0.259918/0.621966 | 0.120915/0.411765/0.852941 | 0.316840/0.445312/0.693015 | 2.2934 | 12.0898 | 0.459438 | False |
| A1 | 0.000288/0.001294/0.089091 | 0.015326/0.068966/0.482759 | 0.376710/0.591133/0.722291 | 2.0994 | 11.4001 | 0.243462 | False |
| A2 | 0.000000/0.000000/0.032591 | 0.000000/0.000000/0.200000 | 0.472356/0.772800/0.827200 | 1.7472 | 9.6252 | 0.102932 | False |
| A3 | 0.000000/0.000000/0.077154 | 0.000000/0.000000/0.200000 | 0.496667/0.843333/0.843333 | 1.6450 | 9.0344 | 0.104866 | False |
| A4 | 0.000000/0.000000/0.077154 | 0.000000/0.000000/0.200000 | 0.473067/0.827200/0.827200 | 1.7203 | 9.5300 | 0.102010 | False |

## 6. Blocker 修复尝试

### 6.1 Shell 命令失误

初次 A0 命令使用:

```bash
PY=/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python CUDA_VISIBLE_DEVICES=6,7 $PY -m tools.build_v14_surfel_atom_bank ...
```

结果:

```text
exit code 127
```

原因: 同一 shell simple command 中 `$PY` 在变量赋值生效前展开为空。修复: 使用绝对 Python 路径。

### 6.2 Unknown atom export repair

过宽版本:

```bash
--min-surfels 1 --min-export-surfels 1 --merge-small-surfels 64 --boundary-safe-px 0.0 --trajectory-bins 2 --rgb-bins 2 --fringe-from-neighbors --export-unknown-atoms --min-export-points-per-object 5
```

结果: 运行超过 3 分钟仍在 build/export，日志未落出；按计划判定为 atom/export 数量过大 blocker，终止 PID `3549200/3549196`。

保守版本:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.build_v14_surfel_atom_bank \
  --seq-list splits/scannet_v6_probe5.txt \
  --variant A4 \
  --output-config stream4d_v14_a4_unknown_merge_atom_candidate_probe5 \
  --atom-root outputs/v14_surfel_atom_bank \
  --summary-root outputs/v14_surfel_atom_bank \
  --min-surfels 8 \
  --min-export-surfels 8 \
  --merge-small-surfels 128 \
  --boundary-safe-px 0.0 \
  --trajectory-bins 2 \
  --rgb-bins 2 \
  --max-mask-votes 12 \
  --export-unknown-atoms \
  --min-export-points-per-object 20 \
  --export-enable-wta
```

结果:

| row | candidate | oracle | Stream3D-on-support | pre% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---|
| A4 unknown merge | 0.000132/0.001190/0.140602 | 0.003175/0.028571/0.400000 | 0.306765/0.450000/0.700340 | 3.1863 | 0.212128 | False |

### 6.3 CropFormer bank16 density repair

发现: v12 measurement bank 只用了 16 个 D4RT local frames，但每个 scene 只有 `0.png` 和 `10.png` 两帧有 CropFormer mask。

检查命令:

```bash
cd Stream3D
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python - <<'PY'
from pathlib import Path
from stream4d.measurement_bank import MeasurementBank
from stream4d.scannet_stream import ScanNetStream
for scene in Path('splits/scannet_v6_probe5.txt').read_text().splitlines():
    b=MeasurementBank.load(Path('outputs/v12_measurement_bank')/scene/'measurement_bank.npz')
    s=ScanNetStream(scene)
    frame_ids=[int(x) for x in b.frame_ids.tolist()]
    avail=[fid for fid in frame_ids if (s.mask_dir/f'{fid}.png').exists()]
    missing=[fid for fid in frame_ids if not (s.mask_dir/f'{fid}.png').exists()]
    print(scene, 'bank_frames', frame_ids, 'mask_available', len(avail), 'missing', missing)
PY
```

结果: 每个 scene `mask_available=2`，missing 为 `1..9,11..15`，共 70 张。

CropFormer 环境排错:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python third_party/Cropformer.py --help
```

结果:

```text
ModuleNotFoundError: No module named 'mask2former'
```

错误 PYTHONPATH 尝试:

```bash
PYTHONPATH=third_party/detectron2/projects/CropFormer:third_party/detectron2/projects/CropFormer/demo_cropformer:third_party/detectron2 ...
```

结果:

```text
ImportError: cannot import name '_C' from 'detectron2'
```

修复: 不把本地未编译 `third_party/detectron2` 放入 `PYTHONPATH`，只加入 CropFormer project/demo，让 conda env 中已编译 detectron2 生效。

可用命令:

```bash
PYTHONPATH=third_party/detectron2/projects/CropFormer:third_party/detectron2/projects/CropFormer/demo_cropformer \
CUDA_VISIBLE_DEVICES=6,7 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python third_party/Cropformer.py --help
```

结果: exit 0。

1-image smoke:

```bash
rm -rf outputs/audit/v14_cropformer_smoke
mkdir -p outputs/audit/v14_cropformer_smoke/scene0050_00/color
ln -s /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D/data/scannet/processed/scene0050_00/color/690.jpg outputs/audit/v14_cropformer_smoke/scene0050_00/color/690.jpg
PYTHONPATH=third_party/detectron2/projects/CropFormer:third_party/detectron2/projects/CropFormer/demo_cropformer \
CUDA_VISIBLE_DEVICES=6,7 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python third_party/Cropformer.py \
  --config-file third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml \
  --seq_name_list scene0050_00 \
  --root outputs/audit/v14_cropformer_smoke \
  --image_path_pattern 'color/*.jpg' \
  --dataset scannet \
  --confidence-threshold 0.5 \
  --opts MODEL.WEIGHTS third_party/seg_models/Mask2Former_hornet_3x_576d0b.pth MODEL.DEVICE cuda
```

结果: exit 0，输出 `outputs/audit/v14_cropformer_smoke/scene0050_00/output_Cropformer/mask/690.png`。

生成 70 张缺失 bank-frame masks:

```bash
rm -rf outputs/audit/v14_cropformer_bank16_missing_input
mkdir -p outputs/audit/v14_cropformer_bank16_missing_input
for scene in scene0050_00 scene0011_00 scene0030_00 scene0081_01 scene0591_00; do
  mkdir -p outputs/audit/v14_cropformer_bank16_missing_input/$scene/color
  for fid in 1 2 3 4 5 6 7 8 9 11 12 13 14 15; do
    ln -s /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D/data/scannet/processed/$scene/color/${fid}.jpg outputs/audit/v14_cropformer_bank16_missing_input/$scene/color/${fid}.jpg
  done
done
PYTHONPATH=third_party/detectron2/projects/CropFormer:third_party/detectron2/projects/CropFormer/demo_cropformer \
CUDA_VISIBLE_DEVICES=6,7 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python third_party/Cropformer.py \
  --config-file third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml \
  --seq_name_list scene0050_00+scene0011_00+scene0030_00+scene0081_01+scene0591_00 \
  --root outputs/audit/v14_cropformer_bank16_missing_input \
  --image_path_pattern 'color/*.jpg' \
  --dataset scannet \
  --confidence-threshold 0.5 \
  --opts MODEL.WEIGHTS third_party/seg_models/Mask2Former_hornet_3x_576d0b.pth MODEL.DEVICE cuda
```

结果:

```text
generated masks: 70
log: outputs/audit/v14_logs/v14_cropformer_bank16_missing.log
copied filelist: outputs/audit/v14_cropformer_bank16_missing_copied_filelist.txt
copied count: 70
```

重建 measurement bank:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.build_v12_measurement_bank \
  --debug-root outputs/v10_d4rt_grid_surfel_field/stream4d_v10_g1_grid32m002_probe5_16f_stride1_fresh_gpu67 \
  --seq-list splits/scannet_v6_probe5.txt \
  --scannet-root data/scannet/processed \
  --backbone Cropformer \
  --output-root outputs/v14_measurement_bank_bank16_cropformer \
  --audit-prefix outputs/audit/v14_measurement_bank_bank16/measurement_bank_probe5
```

Density 对比:

| bank | mask frames available | mask frames missing | unobserved surfel ratio | target positive samples | positive obs rate | negative obs rate | ambiguous |
|---|---:|---:|---:|---:|---:|---:|---:|
| v12 | 2.0 | 14.0 | 0.145764 | 26461.8 | 0.854236 | 0.001477 | 0.012196 |
| v14 bank16 | 16.0 | 0.0 | 0.007849 | 213839.6 | 0.992151 | 0.005811 | 0.047914 |

bank16 atom 结果:

| row | candidate | oracle | Stream3D-on-support | pre% | atom known% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| bank16 A0 | 0.056986/0.262359/0.627404 | 0.120915/0.411765/0.852941 | 0.316840/0.445312/0.693015 | 2.2936 | 12.0898 | 0.458642 | False |
| bank16 A3 | 0.000000/0.000000/0.057673 | 0.000000/0.000000/0.240000 | 0.491204/0.843333/0.843333 | 1.6761 | 9.0515 | 0.124489 | False |
| bank16 A4 | 0.000000/0.000000/0.057673 | 0.000000/0.000000/0.240000 | 0.473067/0.827200/0.827200 | 1.7172 | 9.5251 | 0.123183 | False |
| bank16 A4 unknown merge | 0.000257/0.001157/0.152601 | 0.006173/0.027778/0.416667 | 0.323428/0.461886/0.707364 | 3.2367 | 10.2539 | 0.213122 | False |

### 6.4 Target-dominant atom base repair

原因: bank16 增加 target masks 后，default atom 仍使用旧 carrier `src_mask_id` 作为 atom base，因此 raw source-known support 基本不变。修复: `base_mode=target_dominant`，每个 surfel 使用第一个 positive target mask `(frame_id, mask_id)` 作为 atom birth key。该修复只使用 predicted 2D mask，不读 GT。

命令示例:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.build_v14_surfel_atom_bank \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --seq-list splits/scannet_v6_probe5.txt \
  --variant A4 \
  --base-mode target_dominant \
  --output-config stream4d_v14_a4_bank16_target_atom_candidate_probe5 \
  --atom-root outputs/v14_surfel_atom_bank_bank16_target \
  --summary-root outputs/v14_surfel_atom_bank_bank16_target \
  --min-surfels 4 \
  --min-export-surfels 4 \
  --min-export-points-per-object 20 \
  --export-enable-wta
```

结果:

| row | candidate | oracle | Stream3D-on-support | pre% | atom known% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| bank16 target A3 | 0.002005/0.004992/0.165599 | 0.068627/0.117647/0.558824 | 0.342857/0.494118/0.741176 | 3.0295 | 86.2830 | 0.303492 | False |
| bank16 target A4 | 0.002005/0.004992/0.165599 | 0.068627/0.117647/0.558824 | 0.336866/0.484102/0.760731 | 3.0469 | 88.8538 | 0.302721 | False |

过宽 target-loose:

```text
min_surfels=1, min_export_surfels=1, fringe_from_neighbors, min_export_points=5
```

结果: 2 分 35 秒仍在 build/export，日志未落出；按 atom/export 规模 blocker 终止 PID `3609883/3609879`。

保守 target minpts5:

```bash
CUDA_VISIBLE_DEVICES=6,7 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m tools.build_v14_surfel_atom_bank \
  --bank-root outputs/v14_measurement_bank_bank16_cropformer \
  --seq-list splits/scannet_v6_probe5.txt \
  --variant A4 \
  --base-mode target_dominant \
  --output-config stream4d_v14_a4_bank16_target_minpts5_atom_candidate_probe5 \
  --atom-root outputs/v14_surfel_atom_bank_bank16_target \
  --summary-root outputs/v14_surfel_atom_bank_bank16_target \
  --min-surfels 4 \
  --min-export-surfels 4 \
  --min-export-points-per-object 5 \
  --export-enable-wta
```

结果:

| row | candidate | oracle | Stream3D-on-support | pre% | atom known% | best IoU | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| bank16 target A4 minpts5 | 0.001693/0.004850/0.149635 | 0.060317/0.114286/0.542857 | 0.313843/0.475083/0.727575 | 3.4740 | 88.8538 | 0.276845 | False |

## 7. Phase 2 Summary

汇总文件:

- `Stream3D/outputs/audit/v14_phase2_summary/phase2_atom_repair_matrix_probe5.json`
- `Stream3D/outputs/audit/v14_phase2_summary/phase2_atom_repair_matrix_probe5.csv`
- `Stream3D/outputs/audit/v14_phase2_summary/phase2_atom_repair_matrix_probe5.md`

Gate:

```text
required A3/A4 broad-support oracle AP50 >= 0.60
required A3/A4 broad-support oracle AP25 >= 0.78
required pre_points_ratio >= 25%
required best IoU >= C_hybrid + 0.08 = 0.355611
any_gate_pass = False
```

Best rows:

```text
best oracle AP50: default A0 = 0.411765, but pre%=2.2934 and A0 is not A3/A4 broad-support.
best actual pre%: bank16 target A4 minpts5 = 3.4740%, still far below 25%.
best target-base A3/A4 oracle AP50: 0.117647.
best target-base A3/A4 best IoU: 0.303492, still below C_hybrid+0.08.
```

Decision:

```text
Phase 2 primitive gate failed.
Phase 3/4 global solver not started.
tune30/final not started.
```

## 8. 审计包

复现脚本:

- `Stream3D/scripts/reproduce_v14_phase0_probe5.sh`
- `Stream3D/scripts/reproduce_v14_phase1_failure_decomposition_probe5.sh`
- `Stream3D/scripts/reproduce_v14_phase2_atoms_probe5.sh`
- `Stream3D/scripts/reproduce_v14_final_audit_probe5.sh`

最终审计包文件:

- `stream4d_v14_probe5_code_review_packet.zip`
- `stream4d_v14_probe5_code_review_packet.sha256`
- `stream4d_v14_probe5_filelist.txt`
- `stream4d_v14_probe5_git_diff.patch`
- `stream4d_v14_probe5_git_status.txt`
- `stream4d_v14_probe5_ziptest.log`

审计包包含:

- `Stream3D/evaluation/evaluate.py`
- `Stream3D/evaluation/constants.py`
- `Stream3D/evaluation/__init__.py`
- `Stream3D/stream4d/*.py`
- `Stream3D/tools/*.py`
- `Stream3D/tests/*.py`
- `Stream3D/scripts/reproduce_v14_*.sh`
- `Stream3D/splits/scannet_v6_probe5.txt`
- `Stream3D/data/evaluation/scannet/*v14*class_agnostic.txt`
- `Stream3D/data/prediction/*v14*/config_manifest.json`
- `Stream3D/data/TMP/*v14*/config_manifest.json`
- `Stream3D/outputs/audit/v14_*`
- `Stream3D/outputs/v14_*/*summary*`
- `docs/stream4d_v14_*.md`

## 9. 最终状态

```text
py_compile: pass
pure unittest: 7 tests OK
all unittest: 27 tests OK
reportable scan: suspicious=0, uses_gt_for_prediction=0
metric integrity: phase0_pass=True
Phase 2 gate: fail
Phase 3/4/tune30/final: not started by gate
```
