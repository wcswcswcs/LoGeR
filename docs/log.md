# Run Log

## 2026-04-23

### KITTI 00 verification

Original `LoGeR` full-sequence baseline via `demo_viser.py`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python demo_viser.py \
  --input data/kitti/dataset/sequences/00/image_2 \
  --config ckpts/LoGeR/original_config.yaml \
  --model_name ckpts/LoGeR/latest.pt \
  --window_size 32 \
  --end_frame 10000 \
  --skip_viser \
  --output_txt results/verify_kitti00_demo/LoGeR/00.txt \
  --reset_every 5
```

Pipeline external-exact full-sequence `LoGeR`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_pipeline_abc.py \
  --input data/kitti/dataset/sequences/00/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --geometry_eval_mode \
  --ttt_write_mode native \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --output_txt results/verify_kitti00_pipeline/LoGeR_external_exact/00.txt \
  --output_pt results/verify_kitti00_pipeline/LoGeR_external_exact/00.pt
```

Pipeline external-exact full-sequence `LoGeR*`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_pipeline_abc.py \
  --input data/kitti/dataset/sequences/00/image_2 \
  --checkpoint ckpts/LoGeR_star/latest.pt \
  --config ckpts/LoGeR_star/original_config.yaml \
  --geometry_eval_mode \
  --ttt_write_mode native \
  --geometry_edge_rtol 0.0 \
  --chunk_size 64 \
  --chunk_overlap 3 \
  --window_size 64 \
  --overlap_size 3 \
  --reset_every 5 \
  --se3 \
  --output_txt results/verify_kitti00_pipeline/LoGeR_star_external_exact/00.txt \
  --output_pt results/verify_kitti00_pipeline/LoGeR_star_external_exact/00.pt
```

Original `LoGeR` prefix-64 check via `demo_viser.py`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python demo_viser.py \
  --input data/kitti/dataset/sequences/00/image_2 \
  --config ckpts/LoGeR/original_config.yaml \
  --model_name ckpts/LoGeR/latest.pt \
  --window_size 32 \
  --end_frame 64 \
  --skip_viser \
  --output_txt results/verify_kitti00_demo/LoGeR_prefix64/00.txt \
  --reset_every 5
```

Pipeline external-exact prefix-64 `LoGeR`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_pipeline_abc.py \
  --input data/kitti/dataset/sequences/00/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --geometry_eval_mode \
  --ttt_write_mode native \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --end_frame 64 \
  --output_txt results/verify_kitti00_pipeline/LoGeR_external_exact_prefix64/00.txt \
  --output_pt results/verify_kitti00_pipeline/LoGeR_external_exact_prefix64/00.pt
```

Original `LoGeR*` prefix-64 check via `demo_viser.py`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python demo_viser.py \
  --input data/kitti/dataset/sequences/00/image_2 \
  --config ckpts/LoGeR_star/original_config.yaml \
  --model_name ckpts/LoGeR_star/latest.pt \
  --window_size 64 \
  --end_frame 64 \
  --skip_viser \
  --output_txt results/verify_kitti00_demo/LoGeR_star_prefix64/00.txt \
  --reset_every 5 \
  --se3
```

Pipeline external-exact prefix-64 `LoGeR*`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_pipeline_abc.py \
  --input data/kitti/dataset/sequences/00/image_2 \
  --checkpoint ckpts/LoGeR_star/latest.pt \
  --config ckpts/LoGeR_star/original_config.yaml \
  --geometry_eval_mode \
  --ttt_write_mode native \
  --geometry_edge_rtol 0.0 \
  --chunk_size 64 \
  --chunk_overlap 3 \
  --window_size 64 \
  --overlap_size 3 \
  --reset_every 5 \
  --se3 \
  --end_frame 64 \
  --output_txt results/verify_kitti00_pipeline/LoGeR_star_external_exact_prefix64/00.txt \
  --output_pt results/verify_kitti00_pipeline/LoGeR_star_external_exact_prefix64/00.pt
```

KITTI benchmark commands:

```bash
cd eval/long_eval_script
./kitti_benchmark ../../data/kitti/dataset/poses ../../results/verify_kitti00_demo/LoGeR
./kitti_benchmark ../../data/kitti/dataset/poses ../../results/verify_kitti00_pipeline/LoGeR_external_exact
./kitti_benchmark ../../data/kitti/dataset/poses ../../results/verify_kitti00_pipeline/LoGeR_star_external_exact
```

Exact output comparison commands:

```bash
cmp -s results/verify_kitti00_demo/LoGeR/00.txt results/verify_kitti00_pipeline/LoGeR_external_exact/00.txt; echo $?
cmp -s results/verify_kitti00_demo/LoGeR_star_prefix64/00.txt results/verify_kitti00_pipeline/LoGeR_star_external_exact_prefix64/00.txt; echo $?
md5sum results/verify_kitti00_demo/LoGeR/00.txt results/verify_kitti00_pipeline/LoGeR_external_exact/00.txt
md5sum results/verify_kitti00_demo/LoGeR_star_prefix64/00.txt results/verify_kitti00_pipeline/LoGeR_star_external_exact_prefix64/00.txt
```

Attempted original `LoGeR*` full-sequence `demo_viser.py` run on seq00 (still OOM on current A5000):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LOGER_TTT_DISABLE_COMPILE=1 CUDA_VISIBLE_DEVICES=2 \
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python demo_viser.py \
  --input data/kitti/dataset/sequences/00/image_2 \
  --config ckpts/LoGeR_star/original_config.yaml \
  --model_name ckpts/LoGeR_star/latest.pt \
  --window_size 64 \
  --end_frame 10000 \
  --skip_viser \
  --output_txt results/verify_kitti00_demo/LoGeR_star/00.txt \
  --reset_every 5 \
  --se3
```

### 2026-04-24

Low-memory exact `demo_viser.py` run for full KITTI 00 `LoGeR*` (no need to save visual outputs, but still saved default output folder):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python demo_viser.py   --input data/kitti/dataset/sequences/00/image_2   --config ckpts/LoGeR_star/original_config.yaml   --model_name ckpts/LoGeR_star/latest.pt   --window_size 64   --end_frame 10000   --skip_viser   --external_exact_windows   --output_txt results/verify_kitti00_demo/LoGeR_star/00.txt   --reset_every 5   --se3
```

Prefix-64 exact check for `demo_viser.py --external_exact_windows`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python demo_viser.py   --input data/kitti/dataset/sequences/00/image_2   --config ckpts/LoGeR_star/original_config.yaml   --model_name ckpts/LoGeR_star/latest.pt   --window_size 64   --end_frame 64   --skip_viser   --external_exact_windows   --output_txt results/verify_kitti00_demo/LoGeR_star_prefix64_extdemo/00.txt   --reset_every 5   --se3
```

EfficientSAM3 frontend smoke run on `taylor.mp4` (first failed because `conda` shell function was unavailable, then succeeded with the env python directly):

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n loger python run_efficient_video_masklet_front_end.py \
  --input data/examples/taylor.mp4 \
  --end_frame 8 \
  --chunk_size 8 \
  --chunk_overlap 2 \
  --device cuda \
  --yoloe_model yoloe-11l-seg.pt \
  --output_video results/taylor_efficientsam3_smoke.mp4 \
  --output_pt results/taylor_efficientsam3_smoke.pt

CUDA_VISIBLE_DEVICES=0 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_efficient_video_masklet_front_end.py \
  --input data/examples/taylor.mp4 \
  --end_frame 8 \
  --chunk_size 8 \
  --chunk_overlap 2 \
  --device cuda \
  --yoloe_model yoloe-11l-seg.pt \
  --output_video results/taylor_efficientsam3_smoke.mp4 \
  --output_pt results/taylor_efficientsam3_smoke.pt
```

Longer EfficientSAM3 validation moved off `GPU0` and started on `GPU1`:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python run_efficient_video_masklet_front_end.py \
  --input data/examples/taylor.mp4 \
  --end_frame 32 \
  --chunk_size 32 \
  --chunk_overlap 4 \
  --device cuda \
  --yoloe_model yoloe-11l-seg.pt \
  --output_video results/taylor_efficientsam3_32f_gpu1.mp4 \
  --output_pt results/taylor_efficientsam3_32f_gpu1.pt
```
