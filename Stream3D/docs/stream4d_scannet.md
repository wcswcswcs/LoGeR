# Stream4D ScanNet MVP

This directory contains a training-free Stream4D prototype that keeps Stream3D's ScanNet assets and evaluator, but changes the intermediate unit from static 3D point masks to D4RT carriers.

The default ScanNet exporter is `rgbd_eval`. It uses ScanNet depth and poses only as an evaluation bridge to write official mesh-vertex masks. Do not describe this export path as RGB-only 4D reconstruction.

## Environment Check

```bash
cd /mnt/data/users/chengshun.wang/pjs/LoGeR/Stream3D
source /mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh
conda activate loger
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
python -m tools.check_stream4d_env \
  --d4rt-root ../Open-d4rt \
  --stream3d-root . \
  --seq-name scene0050_00 \
  --backbone Cropformer
```

## Single-Scene Smoke

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 OPEN3D_DISABLE_WEB_VISUALIZER=true \
python -m stream4d.run_scannet \
  --d4rt-root ../Open-d4rt \
  --d4rt-config ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --d4rt-ckpt ../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --seq-name scene0050_00 \
  --backbone Cropformer \
  --frame-stride 10 \
  --max-frames 32 \
  --window-size 32 \
  --window-stride 16 \
  --max-points-per-mask 16 \
  --export-mode rgbd_eval \
  --output-config stream4d_scannet \
  --save-overlays
```

Then evaluate:

```bash
CONDA_INSTRUMENTATION_ENABLED=0 PYTHONNOUSERSITE=1 \
python -m evaluation.evaluate \
  --pred_path data/prediction/stream4d_scannet_class_agnostic \
  --gt_path data/scannet/gt \
  --dataset scannet \
  --no_class \
  --tmp_root data/TMP \
  --tmp_config stream4d_scannet \
  --output_file data/evaluation/scannet/stream4d_scannet_class_agnostic.txt
```
