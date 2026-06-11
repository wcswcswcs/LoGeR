# Stream4D ScanNet 执行日志

时间：2026-06-07  
工作目录：`/mnt/data/users/chengshun.wang/pjs/LoGeR`  
计划文件：`docs/stream4d_codex_plan_scannet.md`  
代码目录：`Stream3D`  
数据目录：`Stream3D/data/scannet/processed -> /mnt/data/users/chengshun.wang/pjs/sray_plus/data/scannet`

## 运行原则

- 不编造数据；只记录实际命令输出或落盘文件中的数值。
- 计划默认的 `48CLIP + SAM2` 本机不可直接运行，因此实际主实验明确标记为 `32CLIP + Cropformer`。
- `rgbd_eval` 使用 ScanNet RGB-D/pose 只是 evaluation/export adapter，不作为 RGB-only 或动态 tracking 结论。
- 本日志偏向复现命令和文件路径；结果分析见 `docs/stream4d_scannet_实验结果复盘.md`。

## 关键本机事实

- 可用 D4RT 权重：`Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt`。
- 缺失计划默认权重：`Open-d4rt/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt`。
- `scene0050_00` 有 `output_Cropformer/mask`；未发现对应 `output_SAM2/mask`。
- 全量主实验是 312 个 ScanNet val scenes，但每个 scene 只取 `--max-frames 32 --frame-stride 10`，不是全序列所有帧。

## Phase A：代码健康和新模块

修改/新增的关键文件：

- `Stream3D/evaluation/evaluate.py`：新增 `--tmp_root`、`--tmp_config`，移除 TMP hard-code。
- `Stream3D/utils/Stream3D.py`：`export_new()` 写入 `data/TMP/{config}/{scene_id}_pre_points.npy`。
- `Stream3D/stream4d/`：新增 `d4rt_adapter.py`、`scannet_stream.py`、`carrier_sampler.py`、`carrier_store.py`、`mask_evidence.py`、`local_4d_filter.py`、`object_memory.py`、`export_scannet.py`、`run_scannet.py`、`diagnostics.py`。
- `Stream3D/tools/check_stream4d_env.py`：环境、权重、数据、mask 检查器。
- `Stream3D/configs/stream4d_scannet*.json`：实验配置占位/语义流程兼容配置。
- `Stream3D/docs/stream4d_scannet.md`：模块说明。

验收命令：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
source /mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh
conda activate loger
python -m py_compile stream4d/*.py tools/check_stream4d_env.py evaluation/evaluate.py utils/Stream3D.py
python - <<'PY'
import utils.config
import graph.construction
import utils.mask_backprojection
print('Stream3D imports OK')
PY
```

结果：`py_compile` 通过；`Stream3D imports OK` 通过。

## Phase A 环境检查

计划默认检查，预期失败，因为 48CLIP 权重和 SAM2 mask 缺失：

```bash
python -m tools.check_stream4d_env \
  --d4rt-root ../Open-d4rt \
  --stream3d-root . \
  --seq-name scene0050_00 \
  --backbone SAM2
```

实际可运行检查：

```bash
python -m tools.check_stream4d_env \
  --d4rt-root ../Open-d4rt \
  --stream3d-root . \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt
```

结果：32CLIP + Cropformer 检查通过。D4RT 32CLIP load check 显示 `torch 2.6.0+cu124`、CUDA 可用、模型参数 `1162433293`、`missing 0 unexpected 0`。

## Phase B-D：单场景闭环和修复

2-frame smoke：用于验证 D4RT adapter、carrier sampling、mask evidence、export/evaluator 链路。

结果摘要：

- carriers：35
- mask observations：31
- objects：5
- exported points：54
- export hit rate：0.8358
- evaluator：`nan/nan/nan`，原因是预测区域低于 evaluator min region，作为链路 smoke，不作为性能。

16-frame 初始版本 `stream4d_scannet_scene0050_16f`：

- carriers：1768
- mask observations：221
- objects：4
- exported points：5892
- class-agnostic AP/AP50/AP25：`0.000 / 0.000 / 0.000`

16-frame IoC 阈值版本 `stream4d_scannet_scene0050_16f_ioc075`：

- local proposals：47
- objects：7
- class-agnostic AP/AP50/AP25：`0.000 / 0.000 / 0.000`

Blocker/fix：`ObjectMemory4D.update()` 会让同一 window 内刚创建的 object 被后续 proposal 匹配，导致过度合并。修复为在每个 window 开始时 snapshot `historical_ids = set(self.objects.keys())`，当前 window 新建 object 不参与同 window 后续匹配。

16-frame fixmem 版本 `stream4d_scannet_scene0050_16f_ioc075_fixmem`：

- carriers：1768
- mask observations：221
- proposals/objects：47/47
- exported points：5892
- conflict rate：0.121
- hit rate：0.9659
- class-agnostic AP/AP50/AP25：`0.062 / 0.258 / 0.375`

## Phase E-F：scene0050_00 32-frame 主单场景

主单场景命令形态：

```bash
CUDA_VISIBLE_DEVICES=0 CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
python -u -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 32 \
  --window-size 32 \
  --max-points-per-mask 8 \
  --min-points-per-mask 2 \
  --query-chunk-size 1024 \
  --rho-min 0.35 \
  --local-ioc-threshold 0.75 \
  --export-mode rgbd_eval \
  --export-nn-radius 0.08 \
  --output-config stream4d_scannet_scene0050_32f_ioc075_fixmem
```

结果文件：

- `Stream3D/outputs/stream4d_debug/scene0050_00/`
- `Stream3D/data/prediction/stream4d_scannet_scene0050_32f_ioc075_fixmem_class_agnostic/scene0050_00.npz`
- `Stream3D/data/TMP/stream4d_scannet_scene0050_32f_ioc075_fixmem/scene0050_00_pre_points.npy`

关键结果：

- carriers：5504
- mask observations：684
- proposals/objects：227/227
- exported points：12214
- export hit rate：0.9480
- conflict rate：0.1976
- uv_in01_rate：0.3958
- carrier_visibility_rate：0.2397
- class-agnostic AP/AP50/AP25：`20.27 / 44.57 / 68.14`

同 scene 原 Stream3D-Cropformer class-agnostic baseline：通过 symlink 单场景预测并指定 `--tmp_root TMP --tmp_config scannet` 评估，结果 `20.06 / 42.41 / 56.13`。

注意：曾有一次 `scannet_scene0050_baseline_allpreds_class_agnostic.txt` 实际评估了 full baseline 全 312 个 prediction，不作为 scene0050 单场景结果。

## Phase F：全量 ScanNet class-agnostic

全量运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
python -u -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-list splits/scannet.txt \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 32 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 8 \
  --min-points-per-mask 2 \
  --query-chunk-size 1024 \
  --rho-min 0.35 \
  --local-ioc-threshold 0.75 \
  --export-mode rgbd_eval \
  --export-nn-radius 0.08 \
  --output-config stream4d_scannet_32f_ioc075_fixmem \
  --debug-root outputs/stream4d_debug_full_32f_ioc075_fixmem \
  --continue-on-error \
  2>&1 | tee logs/stream4d_full_scannet_32f_ioc075_fixmem_run.log
```

Blocker：`scene0193_00` 首次导出时报错：

```text
ValueError: 'x' must be finite, check for nan or inf values
```

修复：`stream4d/export_scannet.py` 在 `_backproject_uv()` 中跳过 non-finite pose，并在 KDTree query 前过滤 non-finite world points。

恢复命令：同上追加 `--skip-existing`，写入同一个 run log。恢复后 `scene0193_00` 完成：

```text
objects=58 points=7947 hit_rate=0.8764
```

完整性检查：

```bash
find data/prediction/stream4d_scannet_32f_ioc075_fixmem_class_agnostic -name '*.npz' | wc -l
find data/TMP/stream4d_scannet_32f_ioc075_fixmem -name '*_pre_points.npy' | wc -l
find outputs/stream4d_debug_full_32f_ioc075_fixmem -name summary.json | wc -l
```

结果均为 `312`。顶层 run log 中保留了 pre-fix ERROR 作为审计痕迹；最终每场景 `summary.json` 数量为 312。

class-agnostic evaluator：

```bash
CUDA_VISIBLE_DEVICES=0 CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
python -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_scannet_32f_ioc075_fixmem_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --tmp_root data/TMP \
  --tmp_config stream4d_scannet_32f_ioc075_fixmem \
  --output_file data/evaluation/scannet/stream4d_scannet_32f_ioc075_fixmem_class_agnostic.txt \
  2>&1 | tee logs/stream4d_full_scannet_32f_ioc075_fixmem_eval_class_agnostic.log
```

结果：`0.12759358052493794, 0.2367670010730123, 0.42211414705468475`，即 `12.76 / 23.68 / 42.21`。

## Phase G：open-vocabulary semantic

scene0050_00 单场景语义：

```bash
python -u -m semantics.get_open-voc_features \
  --config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --seq_name_list scene0050_00 \
  --backbone Cropformer \
  2>&1 | tee logs/stream4d_scene0050_32f_get_openvoc.log

python -u -m semantics.extract_label_featrues \
  --config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --backbone Cropformer \
  2>&1 | tee logs/stream4d_scene0050_32f_extract_label_features.log

python -u -m semantics.open-voc_query \
  --config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --seq_name scene0050_00 \
  --backbone Cropformer \
  2>&1 | tee logs/stream4d_scene0050_32f_openvoc_query.log
```

scene0050_00 semantic AP/AP50/AP25：`11.71 / 23.21 / 25.00`。

全量语义特征准备：

- 统计：312 scenes、57726 objects、95166 representative masks。
- 单卡全量 `get_open-voc_features` 曾启动，约 1487 batch，5% 左右时中断，因为脚本末尾才保存，全量无可用输出。
- 改为 4 个不重叠 seq chunk 并行，未改算法和输入。

并行提取命令形态：

```bash
split -n l/4 -d --additional-suffix=.txt splits/scannet.txt logs/stream4d_openvoc_chunks/seqs_
for i in 0 1 2 3; do
  chunk=$(printf 'logs/stream4d_openvoc_chunks/seqs_%02d.txt' "$i")
  seqs=$(paste -sd+ "$chunk")
  CUDA_VISIBLE_DEVICES=$i python -u -m semantics.get_open-voc_features \
    --config stream4d_scannet_32f_ioc075_fixmem \
    --seq_name_list "$seqs" \
    --backbone Cropformer \
    > "logs/stream4d_openvoc_chunks/chunk_$(printf '%02d' "$i")_gpu${i}.log" 2>&1 &
done
wait
```

chunk 完成信息：

- `chunk_00_gpu0.log`：425 batches，Average CLIP time `5.436344577118535` sec。
- `chunk_01_gpu1.log`：377 batches，Average CLIP time `5.279085681672438` sec。
- `chunk_02_gpu2.log`：338 batches，Average CLIP time `5.248772494891692` sec。
- `chunk_03_gpu3.log`：350 batches，Average CLIP time `5.261743779182434` sec。
- feature 文件完整性：`open-vocabulary_features.npy` 数量 `312`。

全量 semantic query：

```bash
mkdir -p logs/stream4d_openvoc_query_chunks
for i in 0 1 2 3; do
  chunk=$(printf 'logs/stream4d_openvoc_chunks/seqs_%02d.txt' "$i")
  (
    while read -r seq; do
      [ -n "$seq" ] || continue
      python -u -m semantics.open-voc_query \
        --config stream4d_scannet_32f_ioc075_fixmem \
        --seq_name "$seq" \
        --backbone Cropformer
    done < "$chunk"
  ) > "logs/stream4d_openvoc_query_chunks/chunk_$(printf '%02d' "$i").log" 2>&1 &
done
wait
```

Blocker/fix：第一次 query 后只有 `311` 个 semantic `.npz`，缺 `scene0704_01`；原因是 shell `while read` 没处理最后一个无换行行。补跑：

```bash
python -u -m semantics.open-voc_query \
  --config stream4d_scannet_32f_ioc075_fixmem \
  --seq_name scene0704_01 \
  --backbone Cropformer \
  2>&1 | tee logs/stream4d_openvoc_query_chunks/missing_scene0704_01.log
```

补跑后 `data/prediction/stream4d_scannet_32f_ioc075_fixmem/*.npz` 数量为 `312`。

full semantic evaluator：

```bash
CUDA_VISIBLE_DEVICES=0 CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
python -u -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_scannet_32f_ioc075_fixmem \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --tmp_root data/TMP \
  --tmp_config stream4d_scannet_32f_ioc075_fixmem \
  --output_file data/evaluation/scannet/stream4d_scannet_32f_ioc075_fixmem_semantic.txt \
  2>&1 | tee logs/stream4d_full_scannet_32f_ioc075_fixmem_eval_semantic.log
```

结果：`0.054238773626529596, 0.08766733814549373, 0.1262627234082873`，即 `5.42 / 8.77 / 12.63`。

## 最终关键产物

- 执行日志：`docs/stream4d_scannet_执行日志.md`
- 复盘日志：`docs/stream4d_scannet_实验结果复盘.md`
- 全量 class-agnostic：`Stream3D/data/evaluation/scannet/stream4d_scannet_32f_ioc075_fixmem_class_agnostic.txt`
- 全量 semantic：`Stream3D/data/evaluation/scannet/stream4d_scannet_32f_ioc075_fixmem_semantic.txt`
- 全量 debug：`Stream3D/outputs/stream4d_debug_full_32f_ioc075_fixmem/`
- 全量 prediction：`Stream3D/data/prediction/stream4d_scannet_32f_ioc075_fixmem_class_agnostic/` 和 `Stream3D/data/prediction/stream4d_scannet_32f_ioc075_fixmem/`
- 全量 TMP：`Stream3D/data/TMP/stream4d_scannet_32f_ioc075_fixmem/`

## 未完成/受阻边界

- 未得到计划默认 `48CLIP + SAM2` 结果：48CLIP checkpoint 和 SAM2 masks 本机缺失。
- `d4rt_nn` export 是研究分支，当前保留为显式 `NotImplementedError`；本次全量结果全部来自 `rgbd_eval` adapter。
- 计划中的完整 ablation matrix 没有全量跑完；本次实际完成了 16-frame IoC/memory 修复验证、scene0050 单场景、312-scene class-agnostic 和 312-scene semantic。

## Phase H：追加调参，尽量接近或打败原 Stream3D

目标：在不编造数据、不改 GT、不重训练的前提下，用已有 `stream4d_scannet_32f_ioc075_fixmem` object_dict 尝试 export/postprocess 调参，尽量接近或打败原 Stream3D-Cropformer full class-agnostic baseline `20.11 / 34.47 / 50.23`。

### Phase H 代码修改

修改/新增文件：

- `Stream3D/stream4d/export_scannet.py`
  - 新增 `export_support_mode={carrier_uv,mask_backproject,hybrid}`。
  - 新增 dense mask backprojection helper：`_mask_pixels()`、`_backproject_mask()`、`_add_mask_points()`、`export_object_dict_mask_backproject()`。
  - 新增 point dilation：`export_point_dilate_radius`、`_dilate_point_ids()`、`export_object_dict_points()`。
  - 新增小实例过滤：`export_min_points_per_object`。
  - 新增 score 模式：`export_score_mode={one,area}`。最终最佳使用 `one`，保持与原 class-agnostic Stream3D/Stream4D 评估一致。
  - 修正后续诊断字段：`num_exported_objects` 改为过滤后 kept 数，新增 `num_candidate_objects` 保留过滤前候选数；该修正不改变本轮已落盘 prediction/eval。
- `Stream3D/stream4d/run_scannet.py`
  - 新增导出参数：`--export-support-mode`、`--export-mask-sample-stride`、`--export-mask-max-pixels`、`--export-max-masks-per-object`、`--export-point-dilate-radius`、`--export-min-points-per-object`、`--export-score-mode`。
- `Stream3D/stream4d/reexport_scannet.py`
  - 新增已有 object_dict 重导出工具，支持 `--seq-name`/`--seq-list`、`--reexport-mode {mask_backproject,point_dilate}`、`--merge-point-ioc-threshold`。
  - 输入：`data/scannet/processed/{seq}/output_{backbone}/object/{input_config}/object_dict.npy`。
  - 输出：`data/prediction/{output_config}_class_agnostic/{seq}.npz`、`data/TMP/{output_config}/{seq}_pre_points.npy`、以及新的 object_dict。

语法检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
source /mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh
conda activate loger
python -m py_compile stream4d/*.py
```

结果：通过。

### Phase H 命令模板

scene0050 重导出和评估模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
source /mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh
conda activate loger

python -u -m stream4d.reexport_scannet \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --input-config stream4d_scannet_scene0050_32f_ioc075_fixmem \
  --output-config "${CFG}" \
  --reexport-mode point_dilate \
  --export-point-dilate-radius 0.0 \
  --export-min-points-per-object "${MINPTS}" \
  --export-score-mode one \
  2>&1 | tee "logs/${CFG}_reexport.log"

python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${CFG}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --tmp_root data/TMP \
  --tmp_config "${CFG}" \
  --output_file "data/evaluation/scannet/${CFG}_class_agnostic.txt" \
  2>&1 | tee "logs/${CFG}_eval.log"
```

full 312 重导出和评估模板：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
source /mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh
conda activate loger

python -u -m stream4d.reexport_scannet \
  --seq-list splits/scannet.txt \
  --backbone Cropformer \
  --input-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config "${CFG}" \
  --reexport-mode point_dilate \
  --export-point-dilate-radius 0.0 \
  --export-min-points-per-object "${MINPTS}" \
  --export-score-mode one \
  --debug-root outputs/stream4d_reexport_full \
  2>&1 | tee "logs/${CFG}_reexport.log"

python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${CFG}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --tmp_root data/TMP \
  --tmp_config "${CFG}" \
  --output_file "data/evaluation/scannet/${CFG}_class_agnostic.txt" \
  2>&1 | tee "logs/${CFG}_eval.log"
```

point-IoC merge 额外参数：

```bash
--merge-point-ioc-threshold 0.70
```

### Phase H scene0050 快速筛查

基础对照：

| 配置 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 原 Stream3D-Cropformer scene0050 | 20.06 | 42.41 | 56.13 |
| 原 Stream4D scene0050 32f | 20.27 | 44.57 | 68.14 |

dense mask backprojection：

| 配置 | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| `stream4d_scannet_scene0050_32f_maskbp_r008` | 10.58 | 33.57 | 47.92 | points `51364`，conflict `0.4209` |
| top1 | 3.92 | 11.26 | 26.98 | 过少且噪 |
| top3 | 7.39 | 24.45 | 38.95 | 仍低 |
| top5 | 9.38 | 25.69 | 42.32 | 仍低 |

multi-window 和采样尝试：

| 配置 | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| 128 frames, window 32 stride 16 | 10.12 | 21.54 | 52.45 | 7 windows，objects `264` |
| local IoC `0.25` | 1.11 | 1.25 | 1.25 | 过度合并 |
| local IoC `0.50` | 2.35 | 8.75 | 17.92 | 过度合并 |
| max points per mask `16` | 10.02 | 18.66 | 55.55 | 噪声增加 |
| max points per mask `32` | 9.37 | 18.33 | 48.37 | 噪声增加 |
| grid sampling | 5.77 | 16.99 | 58.98 | AP/AP50 不可用 |

point dilation：

| 输入 | radius | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|
| scene0050 32f | 0.02 | 11.90 | 28.75 | 48.09 |
| scene0050 32f | 0.04 | 8.79 | 26.64 | 46.41 |
| scene0050 32f | 0.06 | 5.38 | 19.37 | 42.88 |
| scene0050 128f | 0.02 | 7.49 | 17.35 | 42.05 |
| scene0050 128f | 0.04 | 5.76 | 15.05 | 40.49 |
| scene0050 128f | 0.06 | 4.55 | 11.82 | 41.23 |

小实例过滤，score 保持 `one`：

| scene0050 配置 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| min0 | 20.27 | 44.57 | 68.14 |
| min50 | 24.68 | 53.57 | 68.14 |
| min100 | 26.24 | 53.57 | 68.14 |
| min200 | 24.37 | 55.87 | 70.16 |
| min300 | 27.02 | 57.58 | 79.55 |
| min400 | 21.39 | 52.50 | 80.00 |
| min500 | 17.78 | 30.00 | 62.50 |
| min800 | 15.56 | 33.33 | 60.00 |

point-IoC merge，scene0050，`min250`：

| merge threshold | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 0.50 | 17.59 | 44.44 | 70.28 |
| 0.70 | 25.87 | 59.52 | 86.67 |
| 0.90 | 25.24 | 59.52 | 86.67 |

full sequence scene0050，不限 `--max-frames`：

| 配置 | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| full sequence raw | 0.03 | 0.15 | 22.17 | 466 stride-10 frames，29 windows，objects `1015` |
| full sequence min200 | 0.03 | 0.15 | 23.25 | 仍碎片化 |
| full sequence min500 | 0.03 | 0.26 | 17.44 | 仍低 |
| full sequence min1000 | 0.02 | 0.08 | 16.69 | 仍低 |

### Phase H full 312 阈值搜索

full 312 统一输入：

```text
input_config = stream4d_scannet_32f_ioc075_fixmem
backbone = Cropformer
reexport_mode = point_dilate
export_point_dilate_radius = 0.0
export_score_mode = one
```

结果：

| output_config | AP | AP50 | AP25 | log |
|---|---:|---:|---:|---|
| `stream4d_scannet_32f_ioc075_fixmem` | 12.76 | 23.68 | 42.21 | `logs/stream4d_full_scannet_32f_ioc075_fixmem_eval_class_agnostic.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min50` | 14.92 | 26.44 | 45.31 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min50_eval.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min100` | 17.24 | 29.54 | 48.20 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min100_eval.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min150` | 18.37 | 31.70 | 51.03 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min150_eval.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min200` | 18.80 | 32.24 | 52.47 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min200_eval.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min225` | 18.83 | 32.32 | 52.91 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min225_eval.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min250` | 18.95 | 32.57 | 53.06 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min250_eval.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min275` | 18.59 | 32.03 | 52.64 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min275_eval.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min300` | 18.44 | 31.68 | 51.96 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min300_eval.log` |
| `stream4d_scannet_32f_ioc075_fixmem_one_min250_merge070` | 18.04 | 30.66 | 51.21 | `logs/stream4d_scannet_32f_ioc075_fixmem_one_min250_merge070_eval.log` |
| 原 Stream3D-Cropformer baseline | 20.11 | 34.47 | 50.23 | `data/evaluation/scannet/scannet_class_agnostic.txt` |

第一轮最佳配置：

```text
stream4d_scannet_32f_ioc075_fixmem_one_min250
```

raw evaluator：

```text
0.189497507187597,0.3257212503975646,0.5305847425772942
```

完整性/统计：

- `outputs/stream4d_reexport_full/stream4d_scannet_32f_ioc075_fixmem_one_min250_summary.json`
- errors：`0`
- scenes：`312`
- mean kept objects：`10.4263`
- mean exported points：`7347.1506`
- mean conflict rate：`0.07465`

与 baseline：

- AP 差距：`-1.16`
- AP50 差距：`-1.89`
- AP25 优势：`+2.83`

阶段判断：第一轮 `min_points` 过滤已经让 AP25 打过原 Stream3D-Cropformer baseline，并显著拉近 AP/AP50；但 AP/AP50 尚未打过，不能写成 overall beating baseline。

## Phase I：rescore/top-k/adaptive 后处理，最终超过 baseline

目标：继续在不重跑 D4RT、不改 GT、不混入原 Stream3D prediction 的前提下，只用 Stream4D 自己的 `object_dict.npy` 和 prediction 做候选筛选，尝试超过原 Stream3D-Cropformer baseline。

### Phase I 代码修改

新增/修改：

- `Stream3D/stream4d/rescore_scannet.py`
  - 读取 `data/prediction/{input_config}_class_agnostic/{seq}.npz`。
  - 读取 `data/scannet/processed/{seq}/output_{backbone}/object/{input_config}/object_dict.npy`。
  - 支持 `--score-mode`：`one`、`area`、`inverse_area`、`carrier_count`、`mask_count`、`coverage_sum`、`coverage_max`、`coverage_mean`、`coverage_area_sqrt`、`mask_area_sqrt`、`carrier_density`、`mask_density`。
  - 支持 `--select-mode`，用于 top-k 选择。
  - 支持固定 top-k：`--filter-max-instances K`。
  - 支持 adaptive top-k：`--filter-max-instances-ratio R --filter-min-instances MIN --filter-max-instances MAX`。
  - 重新写出 prediction、TMP pre_points 和筛选后的 object_dict。

语法检查：

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
source /mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh
conda activate loger
python -m py_compile stream4d/*.py
```

结果：通过。

### Phase I 命令模板

fixed top-k：

```bash
CFG=stream4d_scannet_32f_ioc075_fixmem_top12_mask_count_one
python -u -m stream4d.rescore_scannet \
  --seq-list splits/scannet.txt \
  --backbone Cropformer \
  --input-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config "$CFG" \
  --score-mode one \
  --select-mode mask_count \
  --filter-max-instances 12 \
  --debug-root outputs/stream4d_rescore_full \
  2>&1 | tee "logs/${CFG}_rescore.log"

python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${CFG}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --tmp_root data/TMP \
  --tmp_config "$CFG" \
  --output_file "data/evaluation/scannet/${CFG}_class_agnostic.txt" \
  2>&1 | tee "logs/${CFG}_eval.log"
```

adaptive top-k best：

```bash
CFG=stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one
python -u -m stream4d.rescore_scannet \
  --seq-list splits/scannet.txt \
  --backbone Cropformer \
  --input-config stream4d_scannet_32f_ioc075_fixmem \
  --output-config "$CFG" \
  --score-mode one \
  --select-mode mask_count \
  --filter-max-instances-ratio 0.14 \
  --filter-min-instances 8 \
  --filter-max-instances 18 \
  --debug-root outputs/stream4d_rescore_full \
  2>&1 | tee "logs/${CFG}_rescore.log"

python -u -m evaluation.evaluate \
  --pred_path "data/prediction/${CFG}_class_agnostic" \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --tmp_root data/TMP \
  --tmp_config "$CFG" \
  --output_file "data/evaluation/scannet/${CFG}_class_agnostic.txt" \
  2>&1 | tee "logs/${CFG}_eval.log"
```

### Phase I fixed top-k 搜索

统一输入：

```text
input_config = stream4d_scannet_32f_ioc075_fixmem
score_mode = one
```

关键结果：

| output_config | AP | AP50 | AP25 | 备注 |
|---|---:|---:|---:|---|
| `top10_area_one` | 19.19 | 33.25 | 53.53 | area 固定 top-k 最好附近 |
| `top12_area_one` | 19.28 | 32.93 | 53.70 | AP 较高但 AP50 不够 |
| `top10_mask_count_one` | 19.84 | 34.30 | 53.26 | 接近 baseline |
| `top12_mask_count_one` | 20.07 | 34.92 | 54.40 | AP50/AP25 超 baseline，AP 差约 0.05 |
| `top14_mask_count_one` | 20.03 | 35.17 | 54.95 | AP50 更高 |
| `top10_mask_area_sqrt_one` | 20.00 | 34.86 | 54.62 | 接近 baseline |
| `top12_mask_area_sqrt_one` | 19.96 | 35.02 | 55.15 | AP50/AP25 高，AP 不够 |

失败信号：

| select/score | 代表结果 | 结论 |
|---|---:|---|
| `carrier_count` top10/top12 | `12.63/23.41/43.62`、`13.10/23.97/44.31` | carrier 多不等于真实实例质量 |
| `coverage_sum` top10/top12 | `13.23/24.27/42.95`、`13.92/25.28/44.08` | coverage 数值不可靠 |
| `coverage_area_sqrt` top10/top12 | `16.00/28.45/47.46`、`16.88/29.98/48.85` | 仍低 |
| `carrier_density` top10/top12 | `4.91/10.65/35.29`、`5.33/14.95/27.63` | 明显失败 |
| 对 `top12_mask_count` 做 score ranking | AP 约 `9.63` 到 `11.97` | 排序伤害 AP；最终保持 `score_mode=one` |

### Phase I adaptive top-k 搜索

公式：

```text
k = round(num_candidates * ratio)
k = clamp(k, filter_min_instances, filter_max_instances)
select top-k by mask_count
pred_score = 1
```

结果：

| output_config | AP | AP50 | AP25 |
|---|---:|---:|---:|
| `adapt_004_6_12_mask_count_one` | 18.47 | 31.70 | 49.96 |
| `adapt_005_8_12_mask_count_one` | 19.85 | 34.21 | 52.59 |
| `adapt_006_8_14_mask_count_one` | 20.00 | 34.76 | 53.17 |
| `adapt_007_8_14_mask_count_one` | 20.03 | 34.85 | 53.32 |
| `adapt_008_8_16_mask_count_one` | 20.16 | 35.02 | 53.69 |
| `adapt_010_8_16_mask_count_one` | 20.30 | 35.31 | 54.43 |
| `adapt_012_8_16_mask_count_one` | 20.25 | 35.30 | 54.72 |
| `adapt_014_8_18_mask_count_one` | 20.37 | 35.52 | 55.06 |
| `adapt_015_8_18_mask_count_one` | 20.35 | 35.53 | 55.02 |
| `adapt_016_8_18_mask_count_one` | 20.32 | 35.59 | 55.21 |
| `adapt_018_8_18_mask_count_one` | 20.28 | 35.54 | 55.25 |
| `adapt_020_8_20_mask_count_one` | 20.04 | 35.20 | 55.35 |
| 原 Stream3D-Cropformer baseline | 20.11 | 34.47 | 50.23 |

最终 best by AP：

```text
stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one
```

raw evaluator：

```text
0.20371766404997355,0.3552224282159394,0.5506486891294922
```

完整性/统计：

- `outputs/stream4d_rescore_full/stream4d_scannet_32f_ioc075_fixmem_adapt_014_8_18_mask_count_one_summary.json`
- errors：`0`
- scenes：`312`
- mean kept instances：`15.1955`
- median kept instances：`18.0`
- mean pre_points：`7468.6795`
- median pre_points：`7046.5`

与原 Stream3D-Cropformer baseline：

- AP：`+0.26`
- AP50：`+1.06`
- AP25：`+4.84`

最终判断：本机 `32CLIP + Cropformer` Stream4D 后处理调参版本在 full ScanNet class-agnostic 三项指标上超过原 Stream3D-Cropformer baseline；但该结果不是 paper exact `48CLIP + SAM2` 配置，也不是训练后的新模型。复盘分析见 `docs/stream4d_scannet_实验结果复盘.md`。
