# KITTI 04 Table 2 Pipeline Reproduction

This note records the verified pipeline commands for reproducing the
LoGeR / LoGeR* KITTI sequence 04 numbers from Table 2 of the LoGeR paper.

Reference: https://loger-project.github.io/files/loger_paper.pdf

## Goal

Reproduce KITTI seq 04 ATE with the pipeline while keeping the semantic
pipeline disabled. This is a geometry-only reproduction:

- Stage A: LoGeR Geometry Backbone
- Stage B/C/D: skipped via `--geometry_eval_mode`
- Stage E: `TTTWriteController` native write-through via
  `--native_write_through_controller`

Do not run the Video Masklet Front-end for Table 2 reproduction.

## Dataset

This run used the local KITTI odometry data:

```bash
/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/04/image_2
/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/04.txt
```

Seq 04 has 271 RGB frames and 271 GT poses.

## LoGeR

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/04/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --geometry_eval_mode \
  --ttt_write_mode native \
  --native_write_through_controller \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --end_frame 10000 \
  --output_txt results/pipeline_kitti04_native_controller/LoGeR/04.txt \
  --output_pt results/pipeline_kitti04_native_controller/LoGeR/04.pt 2>&1 | tee results/pipeline_kitti04_native_controller/LoGeR/04.log
```

## LoGeR*

`--se3` is required for LoGeR* in the pipeline command. The config enables
SE(3) in the backbone, but the external exact merge path also reads
`args.se3`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/04/image_2 \
  --checkpoint ckpts/LoGeR_star/latest.pt \
  --config ckpts/LoGeR_star/original_config.yaml \
  --geometry_eval_mode \
  --ttt_write_mode native \
  --native_write_through_controller \
  --geometry_edge_rtol 0.0 \
  --chunk_size 64 \
  --chunk_overlap 3 \
  --window_size 64 \
  --overlap_size 3 \
  --reset_every 5 \
  --se3 \
  --end_frame 10000 \
  --output_txt results/pipeline_kitti04_native_controller/LoGeR_star/04.txt \
  --output_pt results/pipeline_kitti04_native_controller/LoGeR_star/04.pt 2>&1 | tee results/pipeline_kitti04_native_controller/LoGeR_star/04.log
```

## KITTI Benchmark

The local `kitti_benchmark` executable expects the estimated trajectory file
directly under the method directory, so run it once per method directory:

```bash
cd eval/long_eval_script

./kitti_benchmark \
  /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses \
  ../../results/pipeline_kitti04_native_controller/LoGeR \
  --plot 2>&1 | tee ../../results/pipeline_kitti04_native_controller/LoGeR/kitti_benchmark.log

./kitti_benchmark \
  /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses \
  ../../results/pipeline_kitti04_native_controller/LoGeR_star \
  --plot 2>&1 | tee ../../results/pipeline_kitti04_native_controller/LoGeR_star/kitti_benchmark.log
```

Do not pass the parent directory `results/pipeline_kitti04_native_controller`
to this benchmark; this version does not recurse into `LoGeR/` and
`LoGeR_star/`.

## Verified Results

Table 2 reports Absolute Trajectory Error (ATE, meters) on KITTI.

| Method | Seq 04 Table 2 ATE | Pipeline ATE RMSE | Status |
| --- | ---: | ---: | --- |
| LoGeR | 1.82 m | 1.8155 m | reproduced |
| LoGeR* | 1.95 m | 1.9545 m | reproduced |

The benchmark also prints RPE, but Table 2 is ATE. Use the `ATE RMSE stats`
block for this reproduction.

## Parity Check Against demo_viser.py

The pipeline trajectory with `--native_write_through_controller` was
byte-identical to the original `demo_viser.py` trajectory on seq 04:

```bash
cmp -s results/pipeline_kitti04_demo/LoGeR/04.txt \
       results/pipeline_kitti04_native_controller/LoGeR/04.txt; echo $?
# 0

cmp -s results/pipeline_kitti04_demo/LoGeR_star/04.txt \
       results/pipeline_kitti04_native_controller/LoGeR_star/04.txt; echo $?
# 0
```

Observed md5:

```bash
d39e906ce3feb34d87737fee25271575  results/pipeline_kitti04_native_controller/LoGeR/04.txt
b6e1b629f5892643026af0830203080f  results/pipeline_kitti04_native_controller/LoGeR_star/04.txt
```

## Native Controller Implementation Note

The native controller path is intentionally lightweight. Native write-through
does not need replay primitives (`q/k/v/lr`); Stage A has already computed the
provisional fast weights. The pipeline therefore builds a small native
`WriteCacheOutput` from `backbone.get_ttt_state()` and passes it to
`TTTWriteController(write_mode="native")`.

This keeps Stage E explicit while preserving the low-memory behavior and the
exact `demo_viser.py` trajectory.

The old implementation used `cache_ttt_primitives=True` for native controller
mode, which cached full replay primitives and OOMed on 23GB GPUs. That is not
needed for native mode; full replay primitives are only needed for
`semantic` / `unity_replay` write modes.
