# KITTI01 GSL-WC Experiments

Date: 2026-04-29

Plan: `docs/KITTI01_Geometry_First_Write_Control_Experiment_Plan.md`

Wandb project: `loger-kitti01-gslwc`

Note: the wandb API key is intentionally not stored in this file.

## Hardware Policy

- GPU 0 is kept free for the user.
- Experiments use GPU 1, GPU 2, and GPU 3.

## Code State

- `run_pipeline_abc.py` supports `--native_write_through_controller` for the exact geometry path.
- `run_pipeline_abc.py` now supports `--ttt_write_mode unity_replay` in `--geometry_eval_mode`, so replay parity can be checked before semantic/write-control sweeps.
- `loger/models/pi3.py` offloads cached TTT replay primitives to CPU immediately. This prevents full-sequence replay cache OOM while preserving replay data.
- `run_pipeline_abc.py` now uses Pi3's native reset-block-aware window merge for the ordinary ABC route too. This is required because the old ABC fallback only aligned adjacent chunks by overlap poses and diverged from exact geometry evaluation on long KITTI runs.

## Gate 0: Native LoGeR Reproduction on KITTI 01

Goal: verify the current pipeline still reproduces the known KITTI 01 baselines before tuning GSL-WC.

Pass criterion from the plan: native LoGeR should be within 2 m of the expected Table 2 value.

### LoGeR

Command summary:

```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
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
  --output_txt results/kitti01_gslwc/gate0_native_controller/LoGeR/01.txt \
  --output_pt results/kitti01_gslwc/gate0_native_controller/LoGeR/01.pt
```

Eval command:

```bash
cd eval/long_eval_script
./kitti_benchmark /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses \
  ../../results/kitti01_gslwc/gate0_native_controller/LoGeR --plot
```

Result:

- Frames: 1101 / 1101 matched
- Sequence length: 2453.2 m
- ATE RMSE: 41.7502 m
- ATE rotation RMSE: 8.9928 deg
- RPE translation: 92.3961 %
- RPE rotation: 0.0084 deg / 100 m
- Wall time: 353.87 s
- Expected Table 2 LoGeR on KITTI 01: 41.64 m
- Difference vs expected: +0.1102 m
- Verdict: PASS

Artifacts:

- `results/kitti01_gslwc/gate0_native_controller/LoGeR/01.txt`
- `results/kitti01_gslwc/gate0_native_controller/LoGeR/01.pt`
- `results/kitti01_gslwc/gate0_native_controller/LoGeR/01.log`
- `results/kitti01_gslwc/gate0_native_controller/LoGeR/kitti_benchmark.log`

### LoGeR*

Command summary:

```bash
CUDA_VISIBLE_DEVICES=2 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
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
  --output_txt results/kitti01_gslwc/gate0_native_controller/LoGeR_star/01.txt \
  --output_pt results/kitti01_gslwc/gate0_native_controller/LoGeR_star/01.pt
```

Eval command:

```bash
cd eval/long_eval_script
./kitti_benchmark /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses \
  ../../results/kitti01_gslwc/gate0_native_controller/LoGeR_star --plot
```

Result:

- Frames: 1101 / 1101 matched
- Sequence length: 2453.2 m
- ATE RMSE: 47.9793 m
- ATE rotation RMSE: 5.8502 deg
- RPE translation: 90.7286 %
- RPE rotation: 0.0075 deg / 100 m
- Wall time: 426.33 s
- Expected Table 2 LoGeR* on KITTI 01: 47.91 m
- Difference vs expected: +0.0693 m
- Verdict: PASS

Artifacts:

- `results/kitti01_gslwc/gate0_native_controller/LoGeR_star/01.txt`
- `results/kitti01_gslwc/gate0_native_controller/LoGeR_star/01.pt`
- `results/kitti01_gslwc/gate0_native_controller/LoGeR_star/01.log`
- `results/kitti01_gslwc/gate0_native_controller/LoGeR_star/kitti_benchmark.log`

## Gate 1: Unity Replay Parity

Goal: verify replay write-back is equivalent to native write-back before semantic-light write-control sweeps. If this gate fails, prior tuning is meaningless because token/prior alignment may be wrong.

### Prefix Smoke Test, First 64 Frames

Native command summary:

```bash
CUDA_VISIBLE_DEVICES=3 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
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
  --end_frame 64 \
  --output_txt results/kitti01_gslwc/gate1_unity_smoke/LoGeR_native64/01.txt \
  --output_pt results/kitti01_gslwc/gate1_unity_smoke/LoGeR_native64/01.pt
```

Unity replay command summary:

```bash
CUDA_VISIBLE_DEVICES=3 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --geometry_eval_mode \
  --ttt_write_mode unity_replay \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --end_frame 64 \
  --output_txt results/kitti01_gslwc/gate1_unity_smoke/LoGeR_unity64/01.txt \
  --output_pt results/kitti01_gslwc/gate1_unity_smoke/LoGeR_unity64/01.pt
```

Result:

- Native prefix ATE RMSE: 0.8915 m
- Native prefix ATE rotation RMSE: 2.7618 deg
- Unity prefix ATE RMSE: 0.8904 m
- Unity prefix ATE rotation RMSE: 2.7537 deg
- ATE difference: -0.0011 m
- Verdict: PASS for smoke parity

Artifacts:

- `results/kitti01_gslwc/gate1_unity_smoke/LoGeR_native64/01.txt`
- `results/kitti01_gslwc/gate1_unity_smoke/LoGeR_native64/kitti_benchmark.log`
- `results/kitti01_gslwc/gate1_unity_smoke/LoGeR_unity64/01.txt`
- `results/kitti01_gslwc/gate1_unity_smoke/LoGeR_unity64/kitti_benchmark.log`

### Full KITTI 01 Unity Replay

Status: completed on GPU 3.

Command summary:

```bash
CUDA_VISIBLE_DEVICES=3 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --geometry_eval_mode \
  --ttt_write_mode unity_replay \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --end_frame 10000 \
  --output_txt results/kitti01_gslwc/gate1_unity_replay/LoGeR/01.txt \
  --output_pt results/kitti01_gslwc/gate1_unity_replay/LoGeR/01.pt
```

Eval command:

```bash
cd eval/long_eval_script
./kitti_benchmark /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses \
  ../../results/kitti01_gslwc/gate1_unity_replay/LoGeR --plot
```

Result:

- Frames: 1101 / 1101 matched
- Sequence length: 2453.2 m
- ATE RMSE: 41.6193 m
- ATE rotation RMSE: 8.9508 deg
- RPE translation: 92.3968 %
- RPE rotation: 0.0083 deg / 100 m
- Wall time: 506.58 s
- Native LoGeR reference ATE: 41.7502 m
- Difference vs native: -0.1309 m
- Gate criterion: absolute ATE difference < 3 m
- Verdict: PASS

Trajectory sanity check vs native LoGeR:

- File shape: 1101 x 8 trajectory rows
- Timestamp equality: true
- Translation diff mean: 0.0406305 m
- Translation diff max: 0.0577467 m
- Quaternion diff mean: 0.000634608
- Quaternion diff max: 0.00159634

Artifacts:

- `results/kitti01_gslwc/gate1_unity_replay/LoGeR/01.txt`
- `results/kitti01_gslwc/gate1_unity_replay/LoGeR/01.pt`
- `results/kitti01_gslwc/gate1_unity_replay/LoGeR/01.log`
- `results/kitti01_gslwc/gate1_unity_replay/LoGeR/kitti_benchmark.log`

## Wandb Logging

Status: uploaded.

Runs:

- `gate0_native_controller_loger_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/ou6woam8
- `gate0_native_controller_loger_star_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/0gg5dtms
- `gate1_unity_smoke_loger64_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/if7nw45n
- `gate1_unity_replay_loger_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/q2d0fabg
- `fusion_F2_explicit_orig_strength_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/xrj9ji0v
- `fusion_F4_max_orig_strength_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/ts1fw4r9
- `fusion_F8_calib050_orig_strength_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/luec9749
- `budget_F4_max_lam1_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/e35tpgb7
- `budget_F4_max_lam1_floor07_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/wljbg7zb
- `budget_F8_calib050_lam1_floor07_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/91i1yldl
- `debug_semantic_all_one_64_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/hkhiujbv
- `debug_abc_unity_64_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/neqr13w2
- `debug_abc_unity_full_native_merge_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/qbpe8035
- `fixed_F4_max_orig_strength_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/zg8agwg0
- `fixed_F8_calib050_orig_strength_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/svsqfwth
- `fixed_F2_explicit_orig_strength_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/drzgom5q
- `fixed_F4_max_lam1_floor07_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/gouklx3a
- `fixed_F8_calib050_lam1_floor07_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/jt1whull
- `support_GS_S3_F8_k5_floor07_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/nfse8f31
- `support_GS_G2_F8_k5_floor07_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/l7wmj173
- `support_GS_G3_F8_k5_floor07_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/546amsbd
- `floor_G3_F8_k5_floor05_lam1_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/vbmz9t1k
- `floor_G3_F8_k5_nofloor_lam1_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/50zwlvpf
- `floor_G3_F8_k5_floor07_lam015_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/3jwe40uv
- `phase0_patch_only_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/vwlzkr81
- `phase0_special_only_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/btrsz0xa
- `phase0_frame_ramp_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/gwb8lh5k
- `phase0_reverse_frame_ramp_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/ocfdln8a
- `phase0_g3_unrolled_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/iwxa9yfk
- `phase0_g3_roll_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/k5l6vo9p
- `phase1_MP01_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/0rgqui8f
- `phase1_MP02_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/cyv81wbh
- `phase1_MP03_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/yzz0258d
- `phase1_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/l8u0c8cj
- `phase1_MP02_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/taz5u6be
- `phase1_MP03_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/wrj9zdry
- `phase4_B301_reproj_MP02_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/w8w3msgu
- `phase4_B302_reproj_nonocc_MP02_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/t8sv2ihk
- `phase4_B301_reproj_MP02_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/0kngu91k
- `phase2_SD01_anchor_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/9v66i9eg
- `phase2_SD02_positive_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/vw2te93y
- `phase2_SD03_dyn_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/969sw30j
- `phase2_SD04_unc_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/8ymvuc5j
- `phase2_SD05_occ_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/bwlfqtzq
- `phase2_SD06_anchor_minus_dyn_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/ewwnpqck
- `phase2_unity_ref_and_segment_diag_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/ipoccp96
- `phase2_BL01_branch0_dyn_MP01_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/3fw2ucgv
- `phase2_BL02_branch1_dyn_MP01_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/u0im282y
- `phase2_BL03_branch2_dyn_MP01_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/ihed8t8p
- `phase2_BL04_all_late_dyn_MP01_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/679z0l01
- `phase2_BL05_all_early_dyn_MP01_128_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/7egal17c
- `phase2_BL01_branch0_dyn_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/uh61jm5e
- `phase2_BL02_branch1_dyn_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/uaal67d6
- `phase2_BL03_branch2_dyn_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/azwtu9q0
- `phase2_BL06_branch0_late_dyn_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/e7e2j52f
- `phase2_BL07_branch0_early_dyn_MP01_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/3hvvnkgu
- `phase2_BL08_branch0_dyn_MP02_full_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/x7obeofk
- `phase2_BL_segment_diag_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/tbnzpiif

## Current Verdict

- Gate 0 native reproduction passed for both LoGeR and LoGeR* on KITTI 01.
- Gate 1 unity replay parity passed on both prefix smoke and full KITTI 01.
- Fixed-merge GSL-WC sweeps are now the only valid model-selection results. Earlier `fusion_sweep/` and `budget_sweep/` numbers are retained only as debugging records because they used the old ordinary-ABC merge.
- Best valid GSL-WC result so far is `support_GS_G3_F8_k5_floor07`: ATE 41.5765 m, which is 0.0428 m better than the unity replay baseline. This is still a tiny gain and far from the plan target of ATE < 33 m.
- Next-plan Phase 0 variable-prior alignment passed; there is no obvious token-type/frame-order prior misalignment.
- Next-plan Phase 1 eta-mean-preserving reweighting failed to improve ATE. MP-01/02/03 preserve lr-weighted write mass correctly, but all are worse than unity replay and the current best. Stronger reweighting improves rotation while hurting ATE, so the current geometry ranking/proxy is the likely bottleneck.
- Next-plan Phase 4 first-pass reprojection proxy also failed to improve ATE. B3-01 full is 41.8684 m, close to MP-02 and worse than unity/best. A simple fitted-pinhole reprojection replacement is not enough.
- Phase 2 diagnostic reliability changed the best result: SD-03 identified `-rank(C_dyn)` as the best score, and BL-01 found that applying it only to TTT branch 0 across all layers reaches 41.3665 m ATE. This is now the best KITTI 01 GSL-WC result so far, improving by 0.2528 m over unity replay.

## Stage-C None / Geometry-Only Smoke

Implementation added:

- `--stage_c_mode none`: skips full-res image loading and does not load SAM/YOLOE.
- Stage C returns an empty `MaskletOutput`, so Stage D degenerates to geometry-only write prior.
- Stage B now exposes `--dyn_fusion_mode` with `explicit`, `implicit`, `max`, `soft_or`, `avg`, `addclip`, and `calibrated_soft_or`.
- Stage D now exposes `--a_token_floor` for light write suppression experiments.

Smoke command summary:

```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --ttt_write_mode semantic \
  --stage_c_mode none \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --end_frame 64 \
  --dyn_fusion_mode max \
  --lambda_min 0.15 \
  --lambda_max 1.0 \
  --a_min_special 0.5 \
  --output_txt results/kitti01_gslwc/gate2_geo_only_smoke/F4_max64/01.txt \
  --output_pt results/kitti01_gslwc/gate2_geo_only_smoke/F4_max64/01.pt
```

Result:

- Pipeline status: PASS, no Stage-C model loaded.
- Prefix ATE RMSE: 0.9951 m
- Prefix ATE rotation RMSE: 2.6415 deg
- Wall time: 60.62 s

Artifacts:

- `results/kitti01_gslwc/gate2_geo_only_smoke/F4_max64/01.txt`
- `results/kitti01_gslwc/gate2_geo_only_smoke/F4_max64/01.pt`
- `results/kitti01_gslwc/gate2_geo_only_smoke/F4_max64/01.log`
- `results/kitti01_gslwc/gate2_geo_only_smoke/F4_max64/kitti_benchmark.log`

## Fusion Sweep 1: Geometry-Only, Original Write Strength

Important: these three runs were produced before the ordinary ABC route was fixed to use Pi3's native reset-block-aware merge. They are retained as a debugging record and are uploaded to wandb, but they should not be treated as final GSL-WC metrics.

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--lambda_min 0.15`
- `--lambda_max 1.0`
- `--a_min_special 0.5`
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`

Results:

| Run | Fusion | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | mean C_dyn | mean B_chunk_geo | mean A_tok | mean lambda | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F2 | explicit | 55.5787 | 12.1337 | 92.3592 | 0.0118 | 833.73 | 0.374679 | 0.443876 | 0.445203 | 0.527300 | FAIL |
| F4 | raw max | 55.4161 | 12.0884 | 92.3623 | 0.0118 | 833.73 | 0.614518 | 0.390295 | 0.391753 | 0.481754 | FAIL |
| F8 | calibrated soft-or, w=0.50 | 55.4521 | 12.0526 | 92.3593 | 0.0118 | 868.83 | 0.438442 | 0.431542 | 0.432897 | 0.516811 | FAIL |

Interpretation:

- All geometry-only candidates are much worse than native LoGeR / unity replay (~41.6-41.8 m).
- The main failure mode is under-writing: `A_tok` mean is only ~0.39-0.45 and Stage E lambda is only ~0.48-0.53, so the effective update is roughly 0.19-0.24 of native in many chunks.
- Fusion mode differences are not decisive yet; write strength must be fixed before interpreting explicit vs implicit cue quality.

Next running tests:

- `results/kitti01_gslwc/budget_sweep/F4_max_lam1`: raw max, `lambda_min=lambda_max=1.0`, no token floor.
- `results/kitti01_gslwc/budget_sweep/F4_max_lam1_floor07`: raw max, `lambda=1.0`, `a_token_floor=0.7`.
- `results/kitti01_gslwc/budget_sweep/F8_calib050_lam1_floor07`: calibrated soft-or, `lambda=1.0`, `a_token_floor=0.7`.

## Budget Sweep 1: Geometry-Only, Stronger Write

Important: these three runs were also produced before the ordinary ABC route was fixed to use Pi3's native reset-block-aware merge. They are retained as a debugging record only.

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--lambda_min 1.0`
- `--lambda_max 1.0`
- `--a_min_special 0.5`
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`

Results:

| Run | Fusion | Extra write floor | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | mean C_dyn | mean B_chunk_geo | mean A_tok | mean lambda | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| F4-lam1 | raw max | none | 55.4400 | 12.0680 | 92.3628 | 0.0118 | 908.06 | 0.614516 | 0.390337 | 0.391787 | 1.000000 | FAIL |
| F4-lam1-floor07 | raw max | `A_tok >= 0.7` | 55.6178 | 12.2765 | 92.3530 | 0.0118 | 906.67 | 0.614288 | 0.390529 | 0.700034 | 1.000000 | FAIL |
| F8-lam1-floor07 | calibrated soft-or, w=0.50 | `A_tok >= 0.7` | 55.4687 | 12.2421 | 92.3535 | 0.0118 | 910.68 | 0.438069 | 0.431839 | 0.702205 | 1.000000 | FAIL |

Interpretation:

- Increasing the chunk write gain to 1.0 did not recover the native / unity replay baseline.
- Clamping token prior to at least 0.7 also did not recover performance.
- Therefore the first failure is not simply "lambda too low" or "mean `A_tok` too low".
- The results suggest that either the ABC non-geometry route has a different baseline from the exact geometry-eval route, or the spatially varying prior is suppressing a small but critical set of tokens.

Artifacts:

- `results/kitti01_gslwc/budget_sweep/F4_max_lam1/`
- `results/kitti01_gslwc/budget_sweep/F4_max_lam1_floor07/`
- `results/kitti01_gslwc/budget_sweep/F8_calib050_lam1_floor07/`

## Replay Diagnostics

### ABC Semantic All-One Prior, 64 Frames

Purpose: verify whether the semantic controller path itself breaks replay when `A_tok=1` and `lambda=1`.

Command summary:

```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --ttt_write_mode semantic \
  --stage_c_mode none \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --end_frame 64 \
  --dyn_fusion_mode max \
  --lambda_min 1.0 \
  --lambda_max 1.0 \
  --a_min_special 1.0 \
  --a_token_floor 1.0 \
  --output_txt results/kitti01_gslwc/debug_replay/semantic_ones64/01.txt \
  --output_pt results/kitti01_gslwc/debug_replay/semantic_ones64/01.pt
```

Result:

- Prefix ATE RMSE: 0.9997 m
- Prefix ATE rotation RMSE: 2.6308 deg
- Wall time: 59.01 s
- Verdict: replay controller path is not catastrophically broken on the prefix.

### ABC Unity Replay, 64 Frames

Purpose: compare the ordinary ABC route against semantic all-one replay.

Command summary:

```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline_abc.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --ttt_write_mode unity_replay \
  --stage_c_mode none \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --window_size 32 \
  --overlap_size 3 \
  --reset_every 5 \
  --end_frame 64 \
  --output_txt results/kitti01_gslwc/debug_replay/abc_unity64/01.txt \
  --output_pt results/kitti01_gslwc/debug_replay/abc_unity64/01.pt
```

Result:

- Prefix ATE RMSE: 0.9997 m
- Prefix ATE rotation RMSE: 2.6308 deg
- Wall time: 58.59 s
- Difference vs semantic all-one prefix: 0.0000 m
- Difference vs exact geometry-eval unity prefix: +0.1093 m
- Verdict: semantic all-one and ABC unity are equivalent on the prefix; the remaining difference is between the ordinary ABC route and the exact geometry-eval route.

### ABC Unity Replay, 64 Frames, Native Merge Fix

After replacing the ordinary ABC fallback merge with Pi3's native reset-block-aware window merge, the same ABC unity prefix was rerun.

Result:

- Prefix ATE RMSE: 0.8904 m
- Prefix ATE rotation RMSE: 2.7537 deg
- Wall time: 60.01 s
- Difference vs exact geometry-eval unity prefix: 0.0000 m
- Verdict: PASS. The ordinary ABC route is now aligned with the exact geometry-eval route on the prefix.

Next diagnostic:

- Run full KITTI 01 with ordinary ABC `unity_replay + stage_c none` after the native merge fix.
- If full ABC unity lands near ~41.6 m, the fixed ABC route is valid and GSL-WC sweeps must be rerun under the fixed merge.

### ABC Unity Replay, Full KITTI 01, Native Merge Fix

Result:

- Frames: 1101 / 1101 matched
- Sequence length: 2453.2 m
- ATE RMSE: 41.6193 m
- ATE rotation RMSE: 8.9508 deg
- RPE translation: 92.3968 %
- RPE rotation: 0.0083 deg / 100 m
- Wall time: 663.82 s
- Difference vs exact geometry-eval unity full: 0.0000 m
- Verdict: PASS. The ordinary ABC route is now a valid baseline for GSL-WC experiments.

Artifacts:

- `results/kitti01_gslwc/debug_replay/abc_unity_full_native_merge/01.txt`
- `results/kitti01_gslwc/debug_replay/abc_unity_full_native_merge/01.pt`
- `results/kitti01_gslwc/debug_replay/abc_unity_full_native_merge/01.log`
- `results/kitti01_gslwc/debug_replay/abc_unity_full_native_merge/kitti_benchmark.log`

## Fixed-Merge Sweep 1: Geometry-Only, Original Write Strength

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--lambda_min 0.15`
- `--lambda_max 1.0`
- `--a_min_special 0.5`
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`
- Pi3 native reset-block-aware output merge enabled.

Results:

| Run | Fusion | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| ABC unity baseline | all-one replay | 41.6193 | 8.9508 | 92.3968 | 0.0083 | 663.82 | PASS baseline |
| F4 | raw max | 42.1406 | 8.0618 | 92.4135 | 0.0080 | 693.40 | WORSE ATE |
| F8 | calibrated soft-or, w=0.50 | 42.0657 | 7.9017 | 92.4095 | 0.0081 | 734.93 | WORSE ATE |

Interpretation:

- The old ~55 m failures were caused by the old ordinary-ABC merge and are invalid for model selection.
- Under the fixed merge, geometry-only suppression no longer catastrophically fails, but both tested priors are slightly worse than unity on ATE.
- Both F4/F8 improve rotation RMSE, especially F8, while hurting translational ATE. This suggests the prior is reducing some rotational noise but suppressing too much translation-useful memory.

## Fixed-Merge Sweep 2: Explicit and Token-Floor Variants

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`
- Pi3 native reset-block-aware output merge enabled.

Results:

| Run | Fusion | Lambda | Token Floor | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | Delta ATE vs Unity (m) | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| ABC unity baseline | all-one replay | n/a | n/a | 41.6193 | 8.9508 | 92.3968 | 0.0083 | 663.82 | 0.0000 | PASS baseline |
| F2 | explicit only | 0.15-1.0 | none | 42.0243 | 7.9976 | 92.4070 | 0.0081 | 877.19 | +0.4050 | WORSE ATE, better rotation |
| F4 | raw max | 1.0 | `A_tok >= 0.7` | 41.6394 | 9.1002 | 92.3940 | 0.0084 | 879.34 | +0.0201 | near baseline, slightly worse |
| F8 | calibrated soft-or, w=0.50 | 1.0 | `A_tok >= 0.7` | 41.6148 | 9.0234 | 92.3954 | 0.0084 | 878.17 | -0.0045 | best so far, tiny gain |

Interpretation:

- `F8_calib050_lam1_floor07` is the first fixed-merge GSL-WC run that beats unity replay on ATE, but the margin is only 4.5 mm over a 2.45 km sequence. This should be considered statistically fragile until repeated on more sequences or with adjacent seeds.
- The token floor prevents catastrophic suppression and brings F4/F8 back near unity behavior, but it also weakens the controller enough that gains are very small.
- Explicit-only F2 improves rotation relative to unity but hurts translational ATE by 0.405 m, so explicit geometric dynamic cues alone are not sufficient for the KITTI 01 objective.
- All fixed-merge semantic runs are slower than unity replay because they still compute cue maps and semantic write-control budgets in every window; the current best run takes 878.17 s versus 663.82 s for unity replay.

Artifacts:

- `results/kitti01_gslwc/fixed_merge_sweep/F2_explicit_orig_strength/`
- `results/kitti01_gslwc/fixed_merge_sweep/F4_max_lam1_floor07/`
- `results/kitti01_gslwc/fixed_merge_sweep/F8_calib050_lam1_floor07/`

## Support Horizon / G-Weight Sweep: F8 + Token Floor

Purpose: continue the plan after fixed-merge parity by testing whether a longer support horizon and stronger geometry eligibility weights can move beyond the near-unity behavior. These runs use the best previous direction, `calibrated_soft_or` with `implicit_weight=0.50`, `implicit_gate_floor=0.25`, full write scale, and `A_tok >= 0.7`.

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--dyn_fusion_mode calibrated_soft_or`
- `--implicit_weight 0.50`
- `--implicit_gate_floor 0.25`
- `--k_intra 5`
- `--sigma_pt 0.25`
- `--lambda_min 1.0`
- `--lambda_max 1.0`
- `--a_min_special 0.5`
- `--a_token_floor 0.7`
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`

Results:

| Run | G Weights | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | Delta ATE vs Unity (m) | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ABC unity baseline | n/a | 41.6193 | 8.9508 | 92.3968 | 0.0083 | 663.82 | 0.0000 | PASS baseline |
| Previous best F8 floor | default G, `k_intra=3` | 41.6148 | 9.0234 | 92.3954 | 0.0084 | 878.17 | -0.0045 | previous best |
| S3 | default G, `k_intra=5` | 41.6008 | 9.0060 | 92.3946 | 0.0084 | 1011.26 | -0.0185 | small gain |
| G2 | `lambda_s=1.0, lambda_a=0.8, lambda_d=1.4, lambda_o=0.5, lambda_u=0.6` | 41.7184 | 9.0230 | 92.3942 | 0.0084 | 1018.35 | +0.0991 | worse |
| G3 | `lambda_s=1.2, lambda_a=0.8, lambda_d=1.2, lambda_o=0.3, lambda_u=0.3` | 41.5765 | 8.9914 | 92.3961 | 0.0084 | 1019.42 | -0.0428 | best so far |

Interpretation:

- Increasing `k_intra` from 3 to 5 helps slightly, so the support horizon direction is valid but weak.
- G2's stronger dynamic/uncertainty suppression worsens ATE, which argues against aggressive write suppression on KITTI 01 under the current prior design.
- G3 is the best valid run so far. It reduces uncertainty penalty and keeps static/anchor terms stronger, suggesting that KITTI 01 needs continued stable-structure adaptation more than aggressive dynamic filtering.
- The gain is still only 4.28 cm over a 2.45 km sequence, so the current GSL-WC mechanism has not yet produced the plan's intended effect.

Artifacts:

- `results/kitti01_gslwc/support_g_sweep/S3_F8_k5_floor07/`
- `results/kitti01_gslwc/support_g_sweep/G2_F8_k5_floor07/`
- `results/kitti01_gslwc/support_g_sweep/G3_F8_k5_floor07/`

Next experiment direction:

- Test whether the `A_tok >= 0.7` floor is masking useful signal. Run G3 with lower/no floor and compare against this best run.
- If lower floors collapse, the current cue/prior is not strong enough for the target; the next code change should add better diagnostics or patch/risk pooling rather than more semantic prompts.

## Floor / Chunk-Budget Sweep: G3 + F8 + k5

Purpose: test whether the small gain from `G3_F8_k5_floor07` is limited by an overly conservative token floor, and whether the chunk-level write budget should be restored.

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--dyn_fusion_mode calibrated_soft_or`
- `--implicit_weight 0.50`
- `--implicit_gate_floor 0.25`
- `--k_intra 5`
- `--sigma_pt 0.25`
- `--lambda_s 1.2`
- `--lambda_a 0.8`
- `--lambda_d 1.2`
- `--lambda_o 0.3`
- `--lambda_u 0.3`
- `--a_min_special 0.5`
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`

Results:

| Run | Token Floor | Lambda | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | Delta ATE vs Best (m) | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Best so far G3/F8/k5 | 0.7 | 1.0 | 41.5765 | 8.9914 | 92.3961 | 0.0084 | 1019.42 | 0.0000 | current best |
| Floor05 | 0.5 | 1.0 | 41.9613 | 8.8948 | 92.4157 | 0.0083 | 1018.78 | +0.3848 | worse ATE |
| NoFloor | 0.0 | 1.0 | 43.5147 | 7.8645 | 92.4639 | 0.0084 | 1015.02 | +1.9382 | ATE collapse, rotation better |
| Floor07Budget | 0.7 | 0.15-1.0 | 41.6924 | 9.0262 | 92.3954 | 0.0084 | 1017.42 | +0.1159 | worse than full lambda |

Interpretation:

- Lowering `A_tok` does not improve KITTI 01 ATE. It improves rotation in the aggressive `NoFloor` case, but translational ATE worsens by 1.94 m versus the best valid run.
- The chunk-level budget also hurts here. KITTI 01 appears to need steady write-through adaptation; suppressing whole-window write scale loses translation accuracy.
- The current dynamic/geometry prior is not accurate enough to replace native LoGeR memory writes. The only safe region is close to unity write behavior (`A_tok >= 0.7`, `lambda=1.0`), which explains why gains are small.
- This falsifies the simple "less write is better" hypothesis for KITTI 01. The next meaningful path is not more scalar tuning; it should improve token ranking itself, e.g. patch risk pooling, segment-level diagnostics, or better motion/static proxy quality.

Artifacts:

- `results/kitti01_gslwc/floor_budget_sweep/G3_F8_k5_floor05_lam1/`
- `results/kitti01_gslwc/floor_budget_sweep/G3_F8_k5_nofloor_lam1/`
- `results/kitti01_gslwc/floor_budget_sweep/G3_F8_k5_floor07_lam015/`

Operational note:

- These full-sequence runs initially used `tee`, and the large stdout volume could back-pressure the process if the session output was not consumed. Future long KITTI sweeps should redirect directly to `01.log` with `> 01.log 2>&1` and avoid streaming every chunk summary to the terminal.

## Next Plan Phase 0: Variable Prior Alignment Gate

Plan source: `docs/KITTI01_GSLWC_Next_Experiment_Plan_Typora.md`.

Implementation added:

- `--debug_prior_mode none|patch_only|special_only|frame_ramp|reverse_frame_ramp|checkerboard|roll`.
- `--prior_debug_jsonl` writes per-chunk/per-layer prior statistics, including `cache_l`, `L_tok`, token-type prefix counts, first prior values, patch/special means, and lr-weighted prior ratios.
- Stage E now receives `geo.token_type` for diagnostics. The actual replay alignment is still the current prefix alignment; this phase tests whether that assumption is sane before changing the policy.

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--lambda_min 1.0`
- `--lambda_max 1.0`
- `--a_min_special 1.0`
- `--a_token_floor 1.0` for synthetic D1-D3; synthetic prior overrides the floor after Stage D.
- `--end_frame 128`
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`

Results:

| Run | Purpose | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | Mean prior | Patch mean | Special mean | Mean eta lr0/lr1/lr2 | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `patch_only_128` | patch tokens set to 0.7, special to 1.0 | 0.9225 | 1.7624 | 95.0764 | 0.0133 | 145.91 | 0.701428 | 0.700000 | 1.000000 | 0.724567 / 0.731543 / 0.732469 | PASS diagnostics |
| `special_only_128` | special tokens set to 0.7, patch to 1.0 | 1.1804 | 1.7692 | 94.9950 | 0.0143 | 145.99 | 0.998571 | 1.000000 | 0.700000 | 0.975433 / 0.968456 / 0.967532 | PASS diagnostics |
| `frame_ramp_128` | frame prior ramps 0.7 to 1.0 | 1.1325 | 1.7346 | 95.0009 | 0.0148 | 144.23 | 0.850000 | 0.850000 | 0.850000 | 0.841992 / 0.848947 / 0.846298 | PASS diagnostics |
| `reverse_frame_ramp_128` | frame prior ramps 1.0 to 0.7 | 0.8776 | 1.8493 | 95.2148 | 0.0150 | 140.09 | 0.850000 | 0.850000 | 0.850000 | 0.858134 / 0.851229 / 0.853710 | PASS diagnostics |
| `g3_unrolled_128` | original G3/F8/k5 no-floor geometry prior | 0.9220 | 2.1265 | 95.0855 | 0.0207 | 153.03 | 0.567335 | 0.565265 | 1.000000 | 0.619382 / 0.554669 / 0.595825 | reference |
| `g3_roll_128` | circularly shifted G3/F8/k5 patch prior | 0.9150 | 1.8788 | 95.0614 | 0.0155 | 151.11 | 0.568354 | 0.566289 | 1.000000 | 0.619803 / 0.566355 / 0.599328 | no suspicious win |

Interpretation:

- `cache_l` matches full-token prefix lengths (`15120` or `40320`) and `L_tok`, not patch-only cache. The recorded token-type prefix includes special and patch tokens in the expected order.
- Patch/special synthetic priors land on the intended token types: patch-only gives patch mean `0.7` and special mean `1.0`; special-only gives patch mean `1.0` and special mean `0.7`.
- Frame ramp and reverse ramp both preserve the expected mean, and their first-token values follow the intended temporal direction.
- The roll prior does not systematically outperform the unshifted prior on the 128-frame smoke. This does not prove ranking quality, but it removes the obvious “spatial prior is totally misaligned” failure.
- Phase 0 is treated as PASS, so the next valid step is eta-weighted mean-preserving reweighting, not more suppressive floor/lambda sweep.

Artifacts:

- `results/kitti01_gslwc/phase0_alignment/patch_only_128/`
- `results/kitti01_gslwc/phase0_alignment/special_only_128/`
- `results/kitti01_gslwc/phase0_alignment/frame_ramp_128/`
- `results/kitti01_gslwc/phase0_alignment/reverse_frame_ramp_128/`
- `results/kitti01_gslwc/phase0_alignment/g3_unrolled_128/`
- `results/kitti01_gslwc/phase0_alignment/g3_roll_128/`

## Next Plan Phase 1: Eta-Weighted Mean-Preserving Reweighting

Implementation added:

- `--prior_policy suppressive|eta_mean_preserving`.
- `--mp_alpha`, `--mp_min`, `--mp_max`, and `--mp_score_source`.
- For `eta_mean_preserving`, Stage D converts patch geometry scores to percentile ranks, maps them to multipliers around `1.0`, fixes special tokens at `1.0`, and disables chunk-level down-scaling by setting `B_chunk_geo=1.0`.
- Stage E can now eta-normalize each branch (`lr0`, `lr1`, `lr2`) separately before replay, while still passing token multipliers to `fast_weight_replay_update`.

Status:

- 128-frame smoke runs for MP-01, MP-02, and MP-03 completed and passed the eta-normalization sanity check.
- Full KITTI 01 runs for MP-01, MP-02, and MP-03 are running on GPU 1, GPU 2, and GPU 3.

### 128-Frame Smoke

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--dyn_fusion_mode calibrated_soft_or`
- `--implicit_weight 0.50`
- `--implicit_gate_floor 0.25`
- `--k_intra 5`
- `--sigma_pt 0.25`
- `--lambda_s 1.2`
- `--lambda_a 0.8`
- `--lambda_d 1.2`
- `--lambda_o 0.3`
- `--lambda_u 0.3`
- `--lambda_min 1.0`
- `--lambda_max 1.0`
- `--a_min_special 1.0`
- `--a_token_floor 0.0`
- `--prior_policy eta_mean_preserving`
- `--mp_score_source e_patch`
- `--end_frame 128`

Results:

| Run | Alpha | Range | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Mean prior | Pre eta lr0 | Post eta lr0/lr1/lr2 | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| MP-01 | 0.1 | `[0.8,1.2]` | 1.0171 | 1.7753 | 95.0444 | 0.0142 | 1.000000 | 1.008000 | 1.000000 / 1.000000 / 1.000000 | PASS smoke |
| MP-02 | 0.2 | `[0.7,1.3]` | 1.0108 | 1.7712 | 95.0407 | 0.0145 | 1.000000 | 1.016001 | 1.000000 / 1.000000 / 1.000000 | PASS smoke |
| MP-03 | 0.4 | `[0.7,1.4]` | 1.0494 | 1.7954 | 95.0360 | 0.0161 | 1.006222 | 1.035747 | 1.000000 / 1.000000 / 1.000000 | PASS smoke, stronger perturbation |

Interpretation:

- Eta normalization is numerically working: post-normalized lr-weighted write ratio stays at `1.0` with about `1e-7` error on all three branches.
- The 128-frame trajectories do not collapse. Prefix ATE is not better than the best suppressive smoke, so the full-sequence result is the real decision point.
- MP-02 remains the main candidate because it is the planned middle strength and has slightly better 128-frame ATE than MP-01/MP-03.

Artifacts:

- `results/kitti01_gslwc/phase1_mean_preserving/MP01_G3F8_k5_etaNorm_range08_12_special1_128/`
- `results/kitti01_gslwc/phase1_mean_preserving/MP02_G3F8_k5_etaNorm_range07_13_special1_128/`
- `results/kitti01_gslwc/phase1_mean_preserving/MP03_G3F8_k5_etaNorm_range07_14_special1_128/`

### Full KITTI 01

Important storage note:

- The filesystem reached 100% during final `.pt` serialization. MP-01 and MP-02 had already saved `01.txt` and `prior_debug.jsonl`, but their `.pt` writes failed with `PytorchStreamWriter failed writing file`.
- The just-generated full-run `.pt` files for MP-01/02/03 were removed to free space. KITTI metrics below are from the preserved `01.txt` trajectory files.

Results:

| Run | Alpha | Range | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | Mean prior | Pre eta lr0 | Post eta lr0/lr1/lr2 | Delta vs Unity (m) | Delta vs Best (m) | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| Unity replay baseline | n/a | n/a | 41.6193 | 8.9508 | 92.3968 | 0.0083 | 663.82 | n/a | n/a | n/a | 0.0000 | +0.0428 | baseline |
| Best suppressive G3/F8/k5 floor07 | n/a | `[0.7,1.0]` | 41.5765 | 8.9914 | 92.3961 | 0.0084 | 1019.42 | n/a | n/a | n/a | -0.0428 | 0.0000 | previous best |
| MP-01 | 0.1 | `[0.8,1.2]` | 41.7479 | 8.9708 | 92.3905 | 0.0084 | 1119.15 | 1.000000 | 1.004662 | 1.000000 / 1.000000 / 1.000000 | +0.1286 | +0.1714 | FAIL |
| MP-02 | 0.2 | `[0.7,1.3]` | 41.8596 | 8.7987 | 92.3875 | 0.0083 | 1119.95 | 1.000000 | 1.009331 | 1.000000 / 1.000000 / 1.000000 | +0.2403 | +0.2831 | FAIL |
| MP-03 | 0.4 | `[0.7,1.4]` | 41.9070 | 8.5595 | 92.3852 | 0.0082 | 1115.24 | 1.006221 | 1.021985 | 1.000000 / 1.000000 / 1.000000 | +0.2877 | +0.3305 | FAIL |

Interpretation:

- Eta-weighted normalization worked as designed: all post-normalized branch ratios stayed at `1.0` up to numerical error.
- Despite preserved effective write mass, ATE worsened as reweighting strength increased. Rotation improved monotonically with stronger MP, but translation/ATE degraded.
- This means the previous failure was not only caused by total under-writing. The current geometry ranking is still not good enough: it changes relative token dominance in a way that helps rotation but hurts translation-useful memory.
- Phase 2 patch-risk ranking should not be run yet because the required MP base policy did not meet the minimum criterion. The next useful step is diagnostic-driven proxy upgrade, especially Stage-B reprojection consistency and better dynamic/occlusion/uncertainty separation.

Artifacts:

- `results/kitti01_gslwc/phase1_mean_preserving/MP01_G3F8_k5_etaNorm_range08_12_special1_full/`
- `results/kitti01_gslwc/phase1_mean_preserving/MP02_G3F8_k5_etaNorm_range07_13_special1_full/`
- `results/kitti01_gslwc/phase1_mean_preserving/MP03_G3F8_k5_etaNorm_range07_14_special1_full/`

## Next Plan Phase 3: Segment Diagnostics

Purpose: after MP full runs failed the full-sequence ATE criterion, diagnose whether the new policy is uniformly bad or only fails in specific sequence regions. The diagnostic computes Sim(3)-aligned translation ATE on overlapping segments and compares each candidate against unity replay.

Implementation:

- Generated `results/kitti01_gslwc/phase3_segment_diagnostics/segment_ate_sim3.csv`.
- Generated `results/kitti01_gslwc/phase3_segment_diagnostics/segment_summary.json`.
- Segment grids: `[200, stride 100]` and `[100, stride 50]`.

Summary versus unity replay:

| Run | 200-frame mean delta (m) | 200-frame improved / total | 200-frame min delta (m) | 200-frame max delta (m) | 100-frame mean delta (m) | 100-frame improved / total | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Best G3 floor07 | -0.0130 | 8 / 10 | -0.1063 | +0.1041 | +0.0009 | 10 / 21 | Mostly near unity, small robust-ish gains |
| MP-01 | -0.0008 | 4 / 10 | -0.1622 | +0.1816 | +0.0269 | 5 / 21 | Mixed; safer but not useful |
| MP-02 | +0.0155 | 4 / 10 | -0.3339 | +0.2106 | +0.0580 | 6 / 21 | Some strong local wins but worse overall |
| MP-03 | +0.0054 | 5 / 10 | -0.6326 | +0.3012 | +0.0908 | 5 / 21 | Stronger local wins and stronger local failures |

Largest 200-frame deltas:

| Run | Worst segments vs unity | Best segments vs unity |
|---|---|---|
| Best G3 floor07 | `[400,600) +0.1041`, `[600,800) +0.0507`, `[800,1000) -0.0012` | `[500,700) -0.1063`, `[300,500) -0.0544`, `[200,400) -0.0513` |
| MP-01 | `[400,600) +0.1816`, `[0,200) +0.0959`, `[600,800) +0.0894` | `[300,500) -0.1622`, `[100,300) -0.1583`, `[500,700) -0.0999` |
| MP-02 | `[400,600) +0.2106`, `[0,200) +0.1697`, `[600,800) +0.1442` | `[300,500) -0.3339`, `[100,300) -0.1591`, `[700,900) -0.0462` |
| MP-03 | `[400,600) +0.3012`, `[0,200) +0.2544`, `[800,1000) +0.2004` | `[300,500) -0.6326`, `[100,300) -0.1585`, `[700,900) -0.1215` |

Interpretation:

- MP is not uniformly harmful. It can substantially improve the `[300,500)` segment, and the improvement grows with alpha.
- The same alpha increase also amplifies failures in `[0,200)`, `[400,600)`, and later segments. This is why full-sequence ATE worsens despite local wins.
- The current geometry ranking therefore has signal, but it is not reliable enough globally. A scalar policy sweep is unlikely to solve this because the same stronger reweighting both helps and hurts different segments.
- This supports the plan's next move: improve Stage B proxy quality, especially reprojection consistency and dynamic/occlusion/uncertainty separation, before attempting Phase 2 patch-risk ranking or semantic reference.

## Next Plan Phase 4: Stage-B v3 Proxy Smoke

Implementation added:

- `--stageb_proxy_mode same_pixel|reprojection`.
- `reprojection` fits a lightweight per-frame pinhole projection from LoGeR's predicted camera-space pointmap, projects current world points into support frames, and bilinear-samples support-frame world/depth/confidence maps before computing point residuals.
- `--stageb_nonocc_dynamic 1` optionally replaces selected dynamic cue with `C_dyn * (1-C_occ) * (1-C_unc)` before constructing `C_anchor` and `G_write_geo`.
- Defaults remain unchanged: `same_pixel` and `stageb_nonocc_dynamic=0`.

### 128-Frame Smoke

| Run | Stage-B proxy | Nonocc dynamic | Base policy | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| MP-02 smoke reference | same-pixel | 0 | MP-02 | 1.0108 | 1.7712 | 95.0407 | 0.0145 | n/a | reference |
| B3-01 smoke | reprojection | 0 | MP-02 | 1.0140 | 1.7868 | 95.0258 | 0.0149 | 107.88 | PASS smoke, not better |
| B3-02 smoke | reprojection | 1 | MP-02 | 1.0205 | 1.7686 | 95.0321 | 0.0147 | 124.25 | PASS smoke, slightly worse |

### Full KITTI 01

| Run | Stage-B proxy | Nonocc dynamic | Base policy | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Debug rows | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| Unity replay baseline | same-pixel | 0 | unity | 41.6193 | 8.9508 | 92.3968 | 0.0083 | n/a | baseline |
| Best G3 floor07 | same-pixel | 0 | suppressive floor07 | 41.5765 | 8.9914 | 92.3961 | 0.0084 | n/a | current best |
| MP-02 full reference | same-pixel | 0 | MP-02 | 41.8596 | 8.7987 | 92.3875 | 0.0083 | 684 | failed |
| B3-01 full | reprojection | 0 | MP-02 | 41.8684 | 8.8177 | 92.3889 | 0.0083 | 306 | failed |

Notes:

- B3-01 full produced a complete `01.txt` and valid KITTI benchmark. Its `prior_debug.jsonl` stopped at 306 rows, likely due to the earlier filesystem/log-file issue during this phase, so only the trajectory metric is treated as authoritative.
- Reprojection consistency in this first implementation did not improve full ATE over MP-02; it is effectively neutral-to-slightly-worse.
- B3-02 was not run full because its 128-frame smoke was worse than B3-01 and MP-02, and B3-01 already failed the full-sequence criterion.
- The current evidence says the bottleneck is not fixed by simply replacing same-pixel residual with a fitted-pinhole reprojection residual. The next meaningful Stage-B work would need better calibration/validation of the projection model or a stronger external motion cue, not another scalar sweep.

Artifacts:

- `results/kitti01_gslwc/phase4_stageb_v3/B301_reproj_MP02_range07_13_special1_full/`

## Phase 2 Diagnostic Reliability: Score Decomposition

Plan source: `docs/KITTI01_GSLWC_Phase2_Diagnostic_Reliability_Plan_Typora.md`.

Purpose: decompose the geometry ranking into simpler cue scores before doing any more scalar tuning. All runs use the gentle eta-mean-preserving MP-01 policy, with `alpha=0.1`, range `[0.8, 1.2]`, `a_min_special=1.0`, and branch-wise eta normalization enabled. Stage C is disabled.

Implementation added:

- `--mp_score_source` now supports `anchor`, `positive`, `dyn`, `unc`, `occ`, and `anchor_minus_dyn` in addition to previous patch-prior sources.
- Score sources are built from Stage-B cue channels before percentile-rank normalization. Risk cues use negative rank so larger final score still means "write more here".
- `prior_debug.jsonl` records mean prior and branch-wise pre/post eta write ratios for every chunk/layer.

Shared config:

- `--ttt_write_mode semantic`
- `--stage_c_mode none`
- `--dyn_fusion_mode calibrated_soft_or`
- `--implicit_weight 0.50`
- `--implicit_gate_floor 0.25`
- `--k_intra 5`
- `--sigma_pt 0.25`
- `--lambda_s 1.2`
- `--lambda_a 0.8`
- `--lambda_d 1.2`
- `--lambda_o 0.3`
- `--lambda_u 0.3`
- `--lambda_min 1.0`
- `--lambda_max 1.0`
- `--a_min_special 1.0`
- `--a_token_floor 0.0`
- `--prior_policy eta_mean_preserving`
- `--mp_alpha 0.1`
- `--mp_min 0.8`
- `--mp_max 1.2`
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`

Results:

| Run | Score source | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Wall (s) | Mean prior | Pre eta lr0 | Post eta lr0 | Delta vs Unity (m) | Delta vs Previous Best (m) | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Unity replay baseline | n/a | 41.6193 | 8.9508 | 92.3968 | 0.0083 | 663.82 | n/a | n/a | n/a | 0.0000 | +0.0428 | baseline |
| Previous best G3/F8/k5 floor07 | mixed suppressive | 41.5765 | 8.9914 | 92.3961 | 0.0084 | 1019.42 | n/a | n/a | n/a | -0.0428 | 0.0000 | previous best |
| SD-01 | `anchor` | 41.6542 | 8.8492 | 92.3878 | 0.0083 | 1021.89 | 1.000000 | 1.001928 | 1.000000 | +0.0349 | +0.0777 | worse ATE, better rotation |
| SD-02 | `positive` | 41.6109 | 8.9345 | 92.3896 | 0.0083 | 1020.49 | 1.000000 | 1.000779 | 1.000000 | -0.0084 | +0.0344 | slightly beats unity, not previous best |
| SD-03 | `dyn` | 41.4742 | 8.9224 | 92.3924 | 0.0083 | 1021.23 | 1.000000 | 1.004202 | 1.000000 | -0.1451 | -0.1023 | new best, model useful |
| SD-04 | `unc` | 41.7294 | 8.9350 | 92.3913 | 0.0083 | 1031.97 | 1.000000 | 1.001455 | 1.000000 | +0.1101 | +0.1529 | worse |
| SD-05 | `occ` | 41.7078 | 9.0263 | 92.3946 | 0.0084 | 1031.17 | 1.000000 | 1.004390 | 1.000000 | +0.0885 | +0.1313 | worse |
| SD-06 | `anchor_minus_dyn` | 41.6191 | 8.8649 | 92.3915 | 0.0083 | 1031.10 | 1.000000 | 1.004029 | 1.000000 | -0.0002 | +0.0426 | essentially unity |

Interpretation:

- SD-03 is the first diagnostic-reliability run that beats the previous best by a nontrivial margin: ATE improves by 0.1023 m over `G3_F8_k5_floor07` and 0.1451 m over unity replay.
- The dynamic cue is not merely a rotation-denoising signal. Under mild eta-mean-preserving reweighting, `-rank(C_dyn)` improves both ATE and rotation versus unity.
- Positive-only SD-02 is safe but weak. It slightly beats unity but does not beat the previous best, so it is less attractive than SD-03 for the next branch/layer selective phase.
- Uncertainty-only SD-04 and occlusion-only SD-05 are not safe token-level write rankings on KITTI 01. They should be treated as reliability/gating signals later, not as direct per-token boost/suppression scores.
- SD-06 does not improve on SD-03. Mixing anchor back into the dynamic score cancels most of the ATE gain, so the next phase should use SD-03 as the candidate score unless segment diagnostics contradict it.

Status:

- Full KITTI benchmark is complete for SD-01 through SD-06.
- Unity replay reference trajectory was regenerated under `results/kitti01_gslwc/phase2_references/unity_replay_full/` because the older `01.txt` artifact had been removed during storage cleanup.
- Non-overlap segment diagnostics have been generated for 100-frame and 50-frame segments.

Non-overlap segment diagnostics versus regenerated unity reference:

| Run | 100-frame mean delta (m) | 100-frame improved / total | 100-frame min delta (m) | 100-frame max delta (m) | 50-frame mean delta (m) | 50-frame improved / total | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| SD-01 anchor | +0.0193 | 4 / 11 | -0.0748 | +0.1749 | +0.0200 | 9 / 22 | local wins but worse full ATE |
| SD-02 positive | +0.0225 | 3 / 11 | -0.1069 | +0.1697 | +0.0233 | 7 / 22 | safe full, but segment mean not better |
| SD-03 dyn | +0.0249 | 3 / 11 | -0.0499 | +0.1759 | +0.0171 | 8 / 22 | best full ATE despite weak segment mean |
| SD-04 unc | +0.0231 | 5 / 11 | -0.0777 | +0.1568 | +0.0245 | 9 / 22 | hurts full ATE |
| SD-05 occ | -0.0177 | 9 / 11 | -0.1001 | +0.0410 | -0.0106 | 15 / 22 | best local Sim(3) mean but bad full ATE |
| SD-06 anchor-minus-dyn | +0.0232 | 4 / 11 | -0.0754 | +0.1883 | +0.0258 | 10 / 22 | essentially unity in full ATE |

Segment interpretation:

- The segment diagnostic is useful, but it is not the same objective as full-sequence KITTI ATE. SD-05 looks best on local Sim(3) non-overlap segments but is worse in full ATE, which means local per-segment shape alignment can hide global trajectory/scale/continuity damage.
- SD-03 remains the model-selection winner because full ATE is the primary objective. The segment table says its gain is probably not coming from broad local Sim(3) improvements; it may be preserving global trajectory consistency better than the other cue decompositions.
- Because SD-05 has many local wins but poor full ATE, occlusion should be considered for a reliability gate or local/manual schedule later, not as the main full-sequence token ranking.

Artifacts:

- `results/kitti01_gslwc/phase2_score_decomposition/SD01_anchor_MP01_full/`
- `results/kitti01_gslwc/phase2_score_decomposition/SD02_positive_MP01_full/`
- `results/kitti01_gslwc/phase2_score_decomposition/SD03_dyn_MP01_full/`
- `results/kitti01_gslwc/phase2_score_decomposition/SD04_unc_MP01_full/`
- `results/kitti01_gslwc/phase2_score_decomposition/SD05_occ_MP01_full/`
- `results/kitti01_gslwc/phase2_score_decomposition/SD06_anchor_minus_dyn_MP01_full/`
- `results/kitti01_gslwc/phase2_score_decomposition/segment_ate_sim3_nonoverlap_100.csv`
- `results/kitti01_gslwc/phase2_score_decomposition/segment_ate_sim3_nonoverlap_50.csv`
- `results/kitti01_gslwc/phase2_score_decomposition/segment_delta_summary.json`
- `results/kitti01_gslwc/phase2_references/unity_replay_full/`

## Phase 2 Diagnostic Reliability: Branch / Layer Selective Prior

Purpose: test whether the useful SD-03 dynamic score should act on all TTT branches/layers or only selected update paths. This follows the plan's hypothesis that earlier gains/losses may be caused by branch/layer coupling rather than the score alone.

Implementation added:

- `--prior_branch_mask`: comma-separated TTT branches that receive the token prior. Unselected branches use unity prior and unity lambda, so they are not affected by token reweighting.
- `--prior_layer_mode all|early|late|middle|single`: selects which layers receive the token prior. Unselected layers use unity replay.
- `fast_weight_replay_update` now accepts optional branch-specific token priors while preserving the previous shared-prior API.
- Branch-wise eta normalization is only applied to branches that actually receive the prior.

Shared score/policy:

- Score source: SD-03 `dyn`, i.e. `-rank(C_dyn)`.
- Policy: MP-01 eta-mean-preserving, `alpha=0.1`, range `[0.8,1.2]`, special tokens fixed at `1.0`.
- Stage C disabled.

### 128-Frame Smoke

| Run | Branch mask | Layer mode | ATE RMSE (m) | Rot RMSE (deg) | Layer prior enabled mean | Branch enabled mean | Verdict |
|---|---|---|---:|---:|---:|---|---|
| BL-01 smoke | `0` | all | 1.0171 | 1.7603 | 1.00 | b0=1.00, b1=0.00, b2=0.00 | PASS smoke |
| BL-02 smoke | `1` | all | 1.0022 | 1.7738 | 1.00 | b0=0.00, b1=1.00, b2=0.00 | PASS smoke, best prefix |
| BL-03 smoke | `2` | all | 1.0057 | 1.7624 | 1.00 | b0=0.00, b1=0.00, b2=1.00 | PASS smoke |
| BL-04 smoke | `0,1,2` | late | 1.0245 | 1.7772 | 0.50 | b0=0.50, b1=0.50, b2=0.50 | PASS smoke, weak prefix |
| BL-05 smoke | `0,1,2` | early | 1.0109 | 1.7733 | 0.50 | b0=0.50, b1=0.50, b2=0.50 | PASS smoke |

Smoke interpretation:

- Branch/layer masks are numerically working: debug rows show only the intended branch/layer subsets are enabled.
- Branch 1 has the best 128-frame ATE, followed by branch 2, but earlier experiments showed prefix behavior is not predictive enough for full KITTI 01.
- The prefix result was misleading: branch 1 was best on 128 frames but fails on the full sequence.

### Full KITTI 01

| Run | Branch mask | Layer mode | Strength | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Delta vs Unity (m) | Delta vs Previous Best (m) | Delta vs SD-03 (m) | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Unity replay baseline | n/a | n/a | n/a | 41.6193 | 8.9508 | 92.3968 | 0.0083 | 0.0000 | +0.0428 | +0.1451 | baseline |
| Previous best G3/F8/k5 floor07 | all | all | suppressive | 41.5765 | 8.9914 | 92.3961 | 0.0084 | -0.0428 | 0.0000 | +0.1023 | previous best |
| SD-03 all-branch dynamic | `0,1,2` | all | MP-01 | 41.4742 | 8.9224 | 92.3924 | 0.0083 | -0.1451 | -0.1023 | 0.0000 | score-decomp best |
| BL-01 | `0` | all | MP-01 | 41.3665 | 8.9490 | 92.3947 | 0.0083 | -0.2528 | -0.2100 | -0.1077 | new best, near strong candidate |
| BL-02 | `1` | all | MP-01 | 41.7513 | 9.0153 | 92.3948 | 0.0084 | +0.1320 | +0.1748 | +0.2771 | fails full despite best prefix |
| BL-03 | `2` | all | MP-01 | 41.6043 | 9.0248 | 92.3958 | 0.0084 | -0.0150 | +0.0278 | +0.1301 | weak gain only |
| BL-06 | `0` | late | MP-01 | 41.5252 | 8.9363 | 92.3959 | 0.0083 | -0.0941 | -0.0513 | +0.0510 | useful but worse than all-layer branch0 |
| BL-07 | `0` | early | MP-01 | 41.5829 | 9.0084 | 92.3962 | 0.0084 | -0.0364 | +0.0064 | +0.1087 | near previous best only |
| BL-08 | `0` | all | MP-02 | 41.6310 | 8.9841 | 92.3948 | 0.0084 | +0.0117 | +0.0545 | +0.1568 | stronger alpha fails |

Full-run interpretation:

- The branch/layer hypothesis is confirmed. The same SD-03 dynamic score is useful when applied to branch 0, but branch 1 damages full ATE and branch 2 only gives a weak gain.
- BL-01 is the new best valid GSL-WC result on KITTI 01 so far: 41.3665 m. It improves by 0.2528 m over unity replay, 0.2100 m over the previous suppressive best, and 0.1077 m over SD-03 all-branch MP-01.
- BL-01 is close to the plan's "strong candidate" bar of 41.3 m, but does not cross it. It is still the first result that looks like a real model direction rather than noise-level scalar tuning.
- Layer splitting does not help. Branch0 late-only and early-only are both worse than branch0 all-layer, so the best current setting is `--prior_branch_mask 0 --prior_layer_mode all`.
- Increasing strength to MP-02 breaks the gain. Branch0 needs mild reweighting; stronger alpha reintroduces trajectory damage.

Non-overlap segment diagnostics versus unity:

| Run | 100-frame mean delta (m) | 100-frame improved / total | 100-frame min delta (m) | 100-frame max delta (m) | 50-frame mean delta (m) | 50-frame improved / total | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| BL-01 branch0 MP-01 | -0.0098 | 6 / 11 | -0.1177 | +0.0427 | -0.0040 | 13 / 22 | best full result, also best 50-frame coverage |
| BL-02 branch1 MP-01 | +0.0193 | 3 / 11 | -0.0256 | +0.1395 | +0.0166 | 9 / 22 | local and full both bad |
| BL-03 branch2 MP-01 | +0.0033 | 5 / 11 | -0.0102 | +0.0313 | +0.0008 | 8 / 22 | weak, mostly near unity |
| BL-06 branch0 late MP-01 | -0.0068 | 5 / 11 | -0.0895 | +0.0433 | +0.0045 | 9 / 22 | useful but less broad than BL-01 |
| BL-07 branch0 early MP-01 | +0.0044 | 5 / 11 | -0.0409 | +0.0442 | +0.0003 | 12 / 22 | weak |
| BL-08 branch0 MP-02 | -0.0009 | 5 / 11 | -0.0782 | +0.0404 | -0.0023 | 8 / 22 | stronger local wins do not translate to full ATE |

Segment interpretation:

- BL-01 has the best full ATE and the best 50-frame improved coverage, so unlike SD-05 it is not merely a local Sim(3) artifact.
- The worst BL-01 100-frame segment is only +0.0427 m versus unity, much smaller than earlier MP failures. This is encouraging for reliability gating because the bad-region tail is already controlled.
- Since BL-01 is near but not under 41.3 m, the next planned step should be a reliability/manual-schedule experiment using branch0 dynamic MP-01 as the base. The likely upside is modest but now grounded in a reliable branch choice.

Artifacts:

- `results/kitti01_gslwc/phase2_branch_layer/BL01_branch0_dyn_MP01_128/`
- `results/kitti01_gslwc/phase2_branch_layer/BL02_branch1_dyn_MP01_128/`
- `results/kitti01_gslwc/phase2_branch_layer/BL03_branch2_dyn_MP01_128/`
- `results/kitti01_gslwc/phase2_branch_layer/BL04_all_late_dyn_MP01_128/`
- `results/kitti01_gslwc/phase2_branch_layer/BL05_all_early_dyn_MP01_128/`
- `results/kitti01_gslwc/phase2_branch_layer/BL01_branch0_dyn_MP01_full/`
- `results/kitti01_gslwc/phase2_branch_layer/BL02_branch1_dyn_MP01_full/`
- `results/kitti01_gslwc/phase2_branch_layer/BL03_branch2_dyn_MP01_full/`
- `results/kitti01_gslwc/phase2_branch_layer/BL06_branch0_late_dyn_MP01_full/`
- `results/kitti01_gslwc/phase2_branch_layer/BL07_branch0_early_dyn_MP01_full/`
- `results/kitti01_gslwc/phase2_branch_layer/BL08_branch0_dyn_MP02_full/`
- `results/kitti01_gslwc/phase2_branch_layer/segment_ate_sim3_nonoverlap_100.csv`
- `results/kitti01_gslwc/phase2_branch_layer/segment_ate_sim3_nonoverlap_50.csv`
- `results/kitti01_gslwc/phase2_branch_layer/segment_delta_summary.json`
