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

## Trajectory Diagnostics Utility

Implementation added:

- `tools/kitti_trajectory_diagnostics.py` compares KITTI GT poses against one or more LoGeR/TUM-style experiment trajectories.
- It saves GT, raw predictions, and Sim(3)-aligned predictions as TUM text plus `camera_trajectories.npz`.
- It generates top-down trajectory plots, per-frame aligned error plots, axis-error plots, segment ATE bar charts, `per_frame_errors.csv`, `segment_errors.csv`, `summary.json`, and `diagnosis.md`.
- It now supports chunk-aware diagnostics via `--chunk_size`, `--chunk_overlap`, and `--focus_run`. This adds `chunk_errors.csv`, a trajectory map colored by focus-run chunk RMSE, GT-to-pred error arrows, worst-frame annotations, and a chunk error timeline.

Example command:

```bash
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python tools/kitti_trajectory_diagnostics.py \
  --gt /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses/01.txt \
  --pred Unity=results/kitti01_gslwc/phase2_references/unity_replay_full/01.txt \
  --pred SD03_dyn=results/kitti01_gslwc/phase2_score_decomposition/SD03_dyn_MP01_full/01.txt \
  --pred BL01_branch0=results/kitti01_gslwc/phase2_branch_layer/BL01_branch0_dyn_MP01_full/01.txt \
  --out_dir results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01 \
  --seq_name KITTI01_Phase2 \
  --axes xz \
  --align sim3 \
  --segment_lengths 50 100 200 \
  --chunk_size 32 \
  --chunk_overlap 3 \
  --focus_run BL01_branch0 \
  --arrow_stride 58 \
  --top_error_count 12 \
  --chunk_label_every 5
```

Sample output:

- `results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01/trajectory_xz_sim3.png`
- `results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01/trajectory_chunk_errors_BL01_branch0_xz_sim3.png`
- `results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01/chunk_error_timeline_BL01_branch0.png`
- `results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01/chunk_errors.csv`
- `results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01/aligned_error_over_frame.png`
- `results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01/aligned_axis_errors.png`
- `results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01/camera_trajectories.npz`
- `results/kitti01_gslwc/trajectory_diagnostics/phase2_unity_sd03_bl01/diagnosis.md`

Sample diagnostic summary:

| Run | Aligned ATE (m) | Sim(3) scale | Largest aligned-error axis | Final error (m) |
|---|---:|---:|---|---:|
| Unity | 41.6193 | 30.787316 | x | 3.681 |
| SD03_dyn | 41.4742 | 30.745089 | x | 4.612 |
| BL01_branch0 | 41.3665 | 30.766306 | x | 3.590 |

Interpretation note:

- The large Sim(3) scale reflects raw-coordinate scale mismatch and is absorbed by Sim(3) alignment; the aligned ATE and aligned axis/segment errors are the relevant diagnostics for KITTI benchmark differences.
- In this sample, BL01 improves aligned ATE over Unity and SD03. The largest residual component is still x-axis error, and the worst global-aligned segments are around frames `[200, 400)`.
- Chunk-aware rerun with `chunk_size=32`, `chunk_overlap=3`, and `focus_run=BL01_branch0` shows the worst BL01 chunks are `c8 [232,264)` RMSE `87.53 m`, `c9 [261,293)` RMSE `74.29 m`, `c7 [203,235)` RMSE `70.62 m`, `c16 [464,496)` RMSE `67.41 m`, and `c15 [435,467)` RMSE `66.63 m`. This makes the main failure windows visible directly on the trajectory rather than only in CSV metrics.

## Pipeline v2 / HMC Correctness Gates

Plan source: `docs/pipeline2/Pipelinev2_Experiment_Plan_Typora.md`.

Purpose: validate the new two-pass Hybrid Memory Controller pipeline before any read-path model experiments. All formal trajectories below are generated by `run_pipeline_abc_v2.py`, not by the old `run_pipeline_abc.py --geometry_eval_mode` shortcut.

Implementation / logging additions:

- `run_pipeline_abc_v2.py` now writes `hmc_config.yaml`, `hmc_state_hash.jsonl`, `hmc_probe_summary.jsonl`, `hmc_control_summary.jsonl`, and `hmc_hook_identity_check.json`.
- `hmc_state_hash.jsonl` records `hash_H_m_before_probe`, `hash_H_m_after_probe`, `hash_H_m_before_pass2`, `hash_H_m_after_commit`, and `hash_H_next`.
- Per-chunk pass1/pass2 geometry parity is logged: pose translation diff, pose matrix diff, local/world pointmap diff, and confidence diff.
- Real identity hook sites are now wired through Pi3 frame attention, SWA read, TTT apply residual, and chunk/global attention. Identity mode records hook coverage while leaving bias/gates as no-op values.
- `tools/upload_hmc_v2_wandb.py` can upload lightweight HMC v2 result/dashboard directories to wandb without uploading `.pt` files.

### A0/A1: 64-Frame Two-Pass No-Control Smoke

Command summary:

```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline_abc_v2.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --geometry_eval_mode \
  --hybrid_memory_mode unity_replay \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 --chunk_overlap 3 \
  --window_size 32 --overlap_size 3 \
  --reset_every 5 \
  --end_frame 64 \
  --output_txt results/kitti01_hmc_v2/A0A1_unity64_v2/01.txt \
  --hybrid_debug_jsonl results/kitti01_hmc_v2/A0A1_unity64_v2/hmc_state_hash.jsonl
```

Result:

| Gate | Check | Result | Verdict |
|---|---|---:|---|
| G0 | `hash_H_m_before_probe == hash_H_m_after_probe` | true for 3 / 3 chunks | PASS |
| G1 | `state_double_write_safe` | true for 3 / 3 chunks | PASS |
| G1 | max pass1/pass2 pose translation diff | 0.0 m | PASS |
| G1 | max pass1/pass2 pose matrix abs diff | 0.0 | PASS |
| G1 | max local/world/confidence diff | 0.0 | PASS |

Artifacts:

- `results/kitti01_hmc_v2/A0A1_unity64_v2/01.txt`
- `results/kitti01_hmc_v2/A0A1_unity64_v2/01.log`
- `results/kitti01_hmc_v2/A0A1_unity64_v2/hmc_state_hash.jsonl`
- `results/kitti01_hmc_v2/A0A1_unity64_v2/hmc_probe_summary.jsonl`
- `results/kitti01_hmc_v2/A0A1_unity64_v2/hmc_control_summary.jsonl`
- `results/kitti01_hmc_v2/A0A1_unity64_v2/hmc_hook_identity_check.json`

### G2: Identity Hook Parity

Command summary:

```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline_abc_v2.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --geometry_eval_mode \
  --hybrid_memory_mode identity_hooks \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 --chunk_overlap 3 \
  --window_size 32 --overlap_size 3 \
  --reset_every 5 \
  --end_frame 10000 \
  --output_txt results/kitti01_hmc_v2/G2_identity_hooks_full_loger/01.txt \
  --hybrid_debug_jsonl results/kitti01_hmc_v2/G2_identity_hooks_full_loger/hmc_state_hash.jsonl
```

KITTI benchmark:

| Run | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | A3 no-control reference | Delta | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| G2 HMC v2 identity hooks | 41.7502 | 8.9928 | 92.3961 | 0.0084 | 41.7502 | 0.0000 | PASS |

State/hook diagnostics:

- `probe_no_commit_hash_equal`: true for 38 / 38 chunks.
- `state_double_write_safe`: true for 38 / 38 chunks.
- max pass1/pass2 pose translation diff: 0.0 m.
- max pass1/pass2 pose matrix abs diff: 0.0.
- max local/world/confidence diff: 0.0.
- Output trajectory is byte-identical to A3 no-control `01.txt` (`max_abs_diff=0.0` over 1101 x 8 rows).
- Hook trace coverage is present for every chunk: frame attention 18 records, SWA read 4 records, TTT apply 18 records, and chunk/global attention 18 records.
- `hmc_hook_identity_check.json` marks frame attention, SWA read, TTT apply, and chunk attention as `implemented_identity_passthrough`.

Artifacts:

- `results/kitti01_hmc_v2/G2_identity_hooks_full_loger/01.txt`
- `results/kitti01_hmc_v2/G2_identity_hooks_full_loger/01.log`
- `results/kitti01_hmc_v2/G2_identity_hooks_full_loger/kitti_benchmark.log`
- `results/kitti01_hmc_v2/G2_identity_hooks_full_loger/hmc_state_hash.jsonl`
- `results/kitti01_hmc_v2/G2_identity_hooks_full_loger/hmc_probe_summary.jsonl`
- `results/kitti01_hmc_v2/G2_identity_hooks_full_loger/hmc_control_summary.jsonl`
- `results/kitti01_hmc_v2/G2_identity_hooks_full_loger/hmc_hook_identity_check.json`

### A3: HMC v2 No-Control LoGeR Reproduction

Command summary:

```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline_abc_v2.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --geometry_eval_mode \
  --hybrid_memory_mode native \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 --chunk_overlap 3 \
  --window_size 32 --overlap_size 3 \
  --reset_every 5 \
  --end_frame 10000 \
  --output_txt results/kitti01_hmc_v2/A3_no_control_full_loger/01.txt \
  --hybrid_debug_jsonl results/kitti01_hmc_v2/A3_no_control_full_loger/hmc_state_hash.jsonl
```

KITTI benchmark:

| Run | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Reference | Delta | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A3 HMC v2 LoGeR native | 41.7502 | 8.9928 | 92.3961 | 0.0084 | 41.7502 | 0.0000 | PASS |

State/parity diagnostics:

- `probe_no_commit_hash_equal`: true for 38 / 38 chunks.
- `state_double_write_safe`: true for 38 / 38 chunks.
- max pass1/pass2 pose translation diff: 0.0 m.
- max pass1/pass2 pose matrix abs diff: 0.0.

Artifacts:

- `results/kitti01_hmc_v2/A3_no_control_full_loger/01.txt`
- `results/kitti01_hmc_v2/A3_no_control_full_loger/01.log`
- `results/kitti01_hmc_v2/A3_no_control_full_loger/kitti_benchmark.log`
- `results/kitti01_hmc_v2/A3_no_control_full_loger/hmc_state_hash.jsonl`
- `results/kitti01_hmc_v2/A3_no_control_full_loger/hmc_probe_summary.jsonl`
- `results/kitti01_hmc_v2/A3_no_control_full_loger/hmc_control_summary.jsonl`

### A4: HMC v2 No-Control LoGeR* Reproduction

Command summary:

```bash
CUDA_VISIBLE_DEVICES=2 python run_pipeline_abc_v2.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR_star/latest.pt \
  --config ckpts/LoGeR_star/original_config.yaml \
  --geometry_eval_mode \
  --hybrid_memory_mode native \
  --geometry_edge_rtol 0.0 \
  --chunk_size 64 --chunk_overlap 3 \
  --window_size 64 --overlap_size 3 \
  --reset_every 5 \
  --se3 \
  --end_frame 10000 \
  --output_txt results/kitti01_hmc_v2/A4_no_control_full_loger_star/01.txt \
  --hybrid_debug_jsonl results/kitti01_hmc_v2/A4_no_control_full_loger_star/hmc_state_hash.jsonl
```

KITTI benchmark:

| Run | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Reference | Delta | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A4 HMC v2 LoGeR* native | 47.9793 | 5.8502 | 90.7286 | 0.0075 | 47.9793 | 0.0000 | PASS |

State/parity diagnostics:

- `probe_no_commit_hash_equal`: true for 18 / 18 chunks.
- `state_double_write_safe`: true for 18 / 18 chunks.
- max pass1/pass2 pose translation diff: 0.0 m.
- max pass1/pass2 pose matrix abs diff: 0.0.

Artifacts:

- `results/kitti01_hmc_v2/A4_no_control_full_loger_star/01.txt`
- `results/kitti01_hmc_v2/A4_no_control_full_loger_star/01.log`
- `results/kitti01_hmc_v2/A4_no_control_full_loger_star/kitti_benchmark.log`
- `results/kitti01_hmc_v2/A4_no_control_full_loger_star/hmc_state_hash.jsonl`
- `results/kitti01_hmc_v2/A4_no_control_full_loger_star/hmc_probe_summary.jsonl`
- `results/kitti01_hmc_v2/A4_no_control_full_loger_star/hmc_control_summary.jsonl`

### A5: HMC TTT-Only BL01 Reproduction

Command summary:

```bash
CUDA_VISIBLE_DEVICES=3 python run_pipeline_abc_v2.py \
  --input /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2 \
  --checkpoint ckpts/LoGeR/latest.pt \
  --config ckpts/LoGeR/original_config.yaml \
  --stage_c_mode none \
  --hybrid_memory_mode ttt_write_only \
  --geometry_edge_rtol 0.0 \
  --chunk_size 32 --chunk_overlap 3 \
  --window_size 32 --overlap_size 3 \
  --reset_every 5 \
  --dyn_fusion_mode calibrated_soft_or \
  --implicit_weight 0.50 \
  --implicit_gate_floor 0.25 \
  --k_intra 5 \
  --sigma_pt 0.25 \
  --lambda_s 1.2 --lambda_a 0.8 --lambda_d 1.2 --lambda_o 0.3 --lambda_u 0.3 \
  --lambda_min 1.0 --lambda_max 1.0 \
  --a_min_special 1.0 --a_token_floor 0.0 \
  --prior_policy eta_mean_preserving \
  --mp_alpha 0.1 --mp_min 0.8 --mp_max 1.2 \
  --mp_score_source dyn \
  --prior_branch_mask 0 \
  --prior_layer_mode all \
  --end_frame 10000 \
  --output_txt results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/01.txt \
  --hybrid_debug_jsonl results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/hmc_state_hash.jsonl \
  --prior_debug_jsonl results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/prior_debug.jsonl
```

KITTI benchmark:

| Run | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Reference BL01 | Delta | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A5 HMC v2 TTT branch0 BL01 | 41.3665 | 8.9490 | 92.3947 | 0.0083 | 41.3665 | 0.0000 | PASS |

State/parity diagnostics:

- `probe_no_commit_hash_equal`: true for 38 / 38 chunks.
- `state_double_write_safe`: true for 38 / 38 chunks.
- max pass1/pass2 pose translation diff: 0.0 m.
- max pass1/pass2 pose matrix abs diff: 0.0.
- `prior_debug.jsonl` confirms branch-wise eta normalization around 1.0 for enabled branch0.

Artifacts:

- `results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/01.txt`
- `results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/01.log`
- `results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/kitti_benchmark.log`
- `results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/hmc_state_hash.jsonl`
- `results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/hmc_probe_summary.jsonl`
- `results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/hmc_control_summary.jsonl`
- `results/kitti01_hmc_v2/A5_ttt_branch0_bl01_repro_full/prior_debug.jsonl`

### Current HMC v2 Verdict

- G0/G1 passed on the 64-frame smoke and on all full runs.
- G2 passed: real identity hooks reproduce A3 no-control exactly at 41.7502 m.
- G3 passed: HMC v2 no-control/native reproduces LoGeR exactly at 41.7502 m.
- G4 passed: HMC v2 no-control/native reproduces LoGeR* exactly at 47.9793 m.
- G5 passed: HMC v2 TTT-only branch0 reproduces old BL01 exactly at 41.3665 m.
- The previous hook-coverage blocker is resolved. Phase C can now start from valid identity hook sites, but non-identity read controllers still need conservative single-path validation.

Wandb runs:

- `hmc_v2_A0A1_unity64_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/67y61o39
- `hmc_v2_A3_no_control_loger_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/zre6fkcz
- `hmc_v2_A4_no_control_loger_star_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/qvxez08g
- `hmc_v2_A5_ttt_branch0_bl01_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/tdqqvrge
- `hmc_v2_G2_identity_hooks_full_loger_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/dm4o303y

## Pipeline v2 / HMC Phase B Probe Trace Dashboards

Plan source: `docs/pipeline2/Pipelinev2_Experiment_Plan_Typora.md`.

Purpose: inspect whether Pass 1 can expose interpretable hybrid-memory signals before running read-path controller experiments. This phase does not have an ATE target; it is a signal-quality and hook-coverage gate for Phase C.

Implementation added:

- `tools/hmc_phase_b_dashboard.py` generates segment dashboards from the HMC v2 two-pass runner.
- The dashboard overlays Stage-B geometry cues, internal proxy maps, branch0 TTT update proxy, and final BL01-style `A_prior`.
- The script also writes `phase_b_config.json`, `phase_b_trace_summary.jsonl`, `phase_b_chunk_summary.csv`, and `phase_b_trace_availability.json`.
- Important limitation: these dashboards were generated before the real G2 hook implementation, so they remain proxy visualizations. The G2 artifacts now provide real identity hook coverage and TTT-apply residual summaries, but the Phase B dashboard panels have not yet been regenerated from those lower-level traces.

Segments:

| Dashboard | Purpose | Chunks summarized | Frames visualized |
|---|---|---:|---:|
| `HMC_B_dashboard_segments_0_200` | old MP hurt segment | 7 | 8 |
| `HMC_B_dashboard_segments_300_500_400_600` | old MP win/hurt comparison | 21 | 16 |
| `HMC_B_dashboard_segments_800_1000` | late drift / hurt segment | 35 | 8 |

Chunk-level summary:

| Segment group | Mean `C_dyn` | Mean `C_dyn` p90 | Mean `C_unc` | Mean `C_anchor` | Mean `G_write` | Mean `attn_dyn` | Mean `ttt_update_proxy` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[0,200)` | 0.2846 | 0.7866 | 0.6359 | 0.2237 | 0.5522 | 0.4996 | 0.3726 |
| `[300,600)` sampled | 0.4310 | 0.8086 | 0.7715 | 0.1405 | 0.4520 | 0.4994 | 0.3483 |
| `[800,1000)` | 0.4347 | 0.8098 | 0.7866 | 0.1372 | 0.4472 | 0.4994 | 0.3513 |

Visual inspection:

- `G_write` and final `A_prior` are interpretable: they consistently favor road/lower-image stable structure and reduce writes on sky/horizon/uncertain regions. This is consistent with why BL01 branch0 TTT write control is safe.
- `C_dyn` is not a clean moving-object detector. It is often high on sky/horizon/background structure and does not isolate vehicles strongly enough to serve as a direct read-path dynamic key mask.
- The current `attn_dyn`, `dyn4d`, and `ttt_update_proxy` panels are useful as rough probes, but they are noisy/proxy-like rather than decisive read-path evidence. `attn_dyn` is especially close to a constant mean around `0.499`, so it should not be used yet as a controller source.
- Representative visualized frames include `frame_0113_chunk_003`, `frame_0485_chunk_016`, and `frame_0913_chunk_031`. They all support the same interpretation: the write prior is plausible, but the internal read-path traces are not yet strong enough.

Artifacts:

- `results/kitti01_hmc_v2/HMC_B_dashboard_segments_0_200/`
- `results/kitti01_hmc_v2/HMC_B_dashboard_segments_300_500_400_600/`
- `results/kitti01_hmc_v2/HMC_B_dashboard_segments_800_1000/`

Phase B verdict:

- Available-signal dashboard generation: PASS.
- Phase C readiness after G2 hook implementation: PASS for identity-hook coverage, still cautious for signal quality.
- Reason: G2 now proves real frame-attention, SWA-read, TTT-apply residual, and chunk-attention hook sites preserve LoGeR exactly in identity mode. The Phase B dashboards still warn that the available read-path signals are noisy, so Phase C should begin with single-path identity-to-control tests rather than combined controllers.

Next required step:

- Begin Phase C read-path-only experiments from the now-validated identity hook baseline.
- Keep TTT update unity/native unless the specific Phase C setting says otherwise, so read-path changes remain attributable.

Wandb backfill:

- `hmc_v2_PhaseB_probe_dashboards_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/v7s2pc8u

## Pipeline v2 / HMC Phase C Read-Path Single-Path Validation

Plan source: `docs/pipeline2/Pipelinev2_Experiment_Plan_Typora.md`.

Purpose: after G2 identity-hook parity passed, test whether individual read-path controls can improve KITTI 01 before combining them with TTT write control. Because Phase B still showed noisy read-path signals, Phase C was limited to single-path diagnostics rather than multi-controller combinations.

Phase B decision:

- Engineering readiness: PASS. Real identity hooks for frame attention, SWA read, TTT apply, and chunk attention preserve the A3 no-control trajectory exactly.
- Signal quality: CAUTIOUS PASS for single-path Phase C only. The old dashboards remain useful but noisy, so Phase C should identify one reliable control source before Phase D.

Implementation added:

- `--read_layer_mode all|early|middle|late|single` and `--read_single_layer` for read-path controls.
- Frame-attention control now applies only on selected decoder layers and records `layer_enabled` in trace summaries.
- TTT-apply control now applies only on selected decoder layers and records real residual/gate summaries at hook sites.
- Phase B dashboard now includes a patch-token TTT-apply residual proxy instead of only the old branch0 update proxy.
- Wandb upload helper now prefixes HMC metrics per result directory when a bundle contains multiple experiments.

Shared Phase C config:

- `--hybrid_memory_mode read_path_only`
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
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`

64-frame smoke:

| Run | Control | Strength | Layer mode | ATE RMSE (m) | Rot RMSE (deg) | Verdict |
|---|---|---:|---|---:|---:|---|
| RFR-100 smoke | frame attention | beta=1.0 | early | 1.0058 | 2.5517 | PASS smoke, non-identity path active |
| R-TTTA-030 smoke | TTT apply | rho=0.3 | all | 0.7987 | 2.9697 | PASS smoke, non-identity path active |

Full KITTI 01:

| Run | Control | Strength | Layer mode | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Delta vs A3 native (m) | Delta vs BL01 (m) | Verdict |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| A3 HMC v2 native | none | n/a | n/a | 41.7502 | 8.9928 | 92.3961 | 0.0084 | 0.0000 | +0.3837 | reference |
| BL01 TTT write | branch0 dyn MP-01 | n/a | all | 41.3665 | 8.9490 | 92.3947 | 0.0083 | -0.3837 | 0.0000 | previous best |
| RFR-025 | frame attention | beta=0.25 | early | 41.1524 | 9.1499 | 92.3839 | 0.0085 | -0.5978 | -0.2141 | useful |
| RFR-050 | frame attention | beta=0.50 | early | 41.1323 | 9.1113 | 92.3808 | 0.0086 | -0.6179 | -0.2342 | useful |
| RFR-100 | frame attention | beta=1.00 | early | 41.0733 | 9.0158 | 92.3818 | 0.0085 | -0.6769 | -0.2932 | best Phase C, near threshold |
| RFR-125 | frame attention | beta=1.25 | early | 41.2165 | 9.0406 | 92.3866 | 0.0086 | -0.5337 | -0.1500 | over-strength starts hurting |
| RFR-150 | frame attention | beta=1.50 | early | 41.2125 | 9.0046 | 92.3853 | 0.0086 | -0.5377 | -0.1540 | worse than beta=1.0 |
| RFR-200 | frame attention | beta=2.00 | early | 41.5707 | 8.9167 | 92.3880 | 0.0085 | -0.1795 | +0.2042 | too strong |
| R-TTTA-010 | TTT apply | rho=0.10 | all | 42.6125 | 8.6284 | 92.4558 | 0.0081 | +0.8623 | +1.2460 | bad ATE, better rotation |
| R-TTTA-020 | TTT apply | rho=0.20 | all | 43.5418 | 8.0282 | 92.5090 | 0.0083 | +1.7916 | +2.1753 | bad ATE |
| R-TTTA-030 | TTT apply | rho=0.30 | all | 44.8062 | 7.7155 | 92.5640 | 0.0091 | +3.0560 | +3.4397 | fails |

Interpretation:

- Frame-attention control has real signal. RFR-100 improves ATE by 0.6769 m over A3 native and 0.2932 m over BL01.
- RFR-100 does not cross the plan's Phase C debug success bar of `<41.0 m`; it misses by 0.0733 m. This is close, but not enough to justify Phase D combinations yet.
- The stronger beta sweep turns around after beta=1.0, so the immediate failure mode is not under-strength. More aggressive frame attention starts damaging the trajectory.
- TTT-apply control is not safe as currently defined. It improves rotation strongly but introduces large translation/ATE damage, which suggests the residual gate is suppressing useful adaptation or changing pose scale/continuity.
- SWA-read and chunk-attention non-identity controls are not yet treated as valid Phase C result rows because dense read-importance and chunk dynamic-mass control maps still need stronger source signals.

Phase C verdict:

- Phase C started successfully from the validated Phase B/G2 hook baseline.
- Phase C has not passed the plan's read-path-only success criterion because no run reached `<41.0 m` ATE.
- Do not start Phase D yet. The next optimization should refine the frame-attention cue/layer schedule or protect reference/low-risk regions, then rerun a smaller Phase C gate around beta=1.0.

Artifacts:

- `results/kitti01_hmc_v2/phaseC_read_path/RFR025_frame_early_full/`
- `results/kitti01_hmc_v2/phaseC_read_path/RFR050_frame_early_full/`
- `results/kitti01_hmc_v2/phaseC_read_path/RFR100_frame_early_64/`
- `results/kitti01_hmc_v2/phaseC_read_path/RFR100_frame_early_full/`
- `results/kitti01_hmc_v2/phaseC_read_path/RFR125_frame_early_full/`
- `results/kitti01_hmc_v2/phaseC_read_path/RFR150_frame_early_full/`
- `results/kitti01_hmc_v2/phaseC_read_path/RFR200_frame_early_full/`
- `results/kitti01_hmc_v2/phaseC_read_path/RTTTA010_all_full/`
- `results/kitti01_hmc_v2/phaseC_read_path/RTTTA020_all_full/`
- `results/kitti01_hmc_v2/phaseC_read_path/RTTTA030_all_64/`
- `results/kitti01_hmc_v2/phaseC_read_path/RTTTA030_all_full/`

Wandb:

- `hmc_v2_PhaseC_read_path_single_path_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/062k2sj9

## Pipeline v2 / HMC Phase C v2 FineGate

Plan source: `docs/pipeline2/KITTI01_Pipelinev2__PhaseC_v2_FineGate_Experiment_Plan_Typora.md`.

Purpose: refine the only useful Phase C v1 route, frame-attention read control, by splitting cue source, bias direction, dynamic mass cap, static/reference protection, and early-layer schedule. The goal was to beat the RFR-100 read-path-only reference (`41.0733 m`) and ideally cross the Phase C v2 pass bar (`<=40.8 m`) before any Phase D combinations.

Implementation added:

- `--read_cue_source dyn|dyn_reliable|internal_attn|key_cosine_avg|key_cosine_shallow|key_cosine_deep`.
- `--frame_bias_mode pair|protected_pair|key|query`.
- `--read_topk_frac` for per-frame top-k dynamic mass caps.
- `--read_protect_static`, `--read_static_anchor_thr`, and `--read_static_dyn_thr` for high-anchor / low-dynamic static patch protection.
- `--read_layer_mode early_quarter|early_half` in addition to the existing read layer schedules.
- TTT-apply read gates now also support `--ttt_apply_min_gate`, although TTT-apply was not promoted in this FineGate run because Phase C v1 already showed it was unsafe.

Shared config:

- `--hybrid_memory_mode read_path_only`
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
- `--chunk_size 32`
- `--chunk_overlap 3`
- `--window_size 32`
- `--overlap_size 3`
- `--reset_every 5`

### 64-Frame Smoke

| Run | Cue | Bias | Layer mode | Strength | ATE RMSE (m) | Rot RMSE (deg) | Verdict |
|---|---|---|---|---:|---:|---:|---|
| F64-01 | `dyn` | pair | early | beta=1.0 | 1.0058 | 2.5517 | reproduces RFR-100 smoke |
| F64-02 | `dyn` | key | early | beta=0.5 | 0.9548 | 2.6367 | useful smoke, added to 256 |
| F64-03 | `dyn` | protected_pair | early | beta=1.0 | 1.0074 | 2.5487 | protection alone not enough |
| F64-04 | `dyn_reliable` | protected_pair | early | beta=1.0 | 0.9122 | 2.7108 | best 64-frame |
| F64-05 | `dyn` top10 | protected_pair | early | beta=1.0 | 0.9420 | 2.6910 | useful smoke |
| F64-06 | `internal_attn` | protected_pair | early | beta=1.0 | 1.0302 | 2.5487 | weakest smoke, dropped before 256 |
| F64-07 | `internal_attn` top10 | protected_pair | early_quarter | beta=1.0 | 0.9233 | 2.6854 | useful conservative internal candidate |

64-frame interpretation:

- Reliability-filtering and top-k mass caps both improved the short smoke versus RFR-100.
- Dense internal attention was weak, so it was not worth spending a 256-frame slot on it.
- Key-only beta=0.5 was not in the original suggested 256 matrix, but it passed smoke better than dense internal, so it was promoted as a replacement candidate.

### 256-Frame Mid-Run

| Run | Cue | Bias | Layer mode | Strength | ATE RMSE (m) | Rot RMSE (deg) | Verdict |
|---|---|---|---|---:|---:|---:|---|
| F256-01 | `dyn` | pair | early | beta=1.0 | 3.8666 | 4.2789 | RFR reference |
| F256-02 | `dyn_reliable` | protected_pair | early | beta=1.0 | 3.7941 | 3.7843 | best 256-frame |
| F256-03 | `dyn` top10 | protected_pair | early | beta=1.0 | 3.9533 | 3.8526 | worse than reference |
| F256-04 | `dyn` top10 | protected_pair | early_quarter | beta=1.0 | 3.9240 | 3.8663 | worse than reference |
| F256-06 | `internal_attn` top10 | protected_pair | early_quarter | beta=1.0 | 3.8854 | 3.8226 | near reference, not better |
| F256-07 | `dyn` | key | early | beta=0.5 | 3.8467 | 4.1613 | second-best ATE, promoted to full |

256-frame interpretation:

- F256-02 looked promising: it improved both ATE and rotation over the RFR reference.
- The top10 dynamic variants did not carry their 64-frame gain into 256 frames.
- Key-only beta=0.5 remained competitive, so it was included as an exploratory full candidate despite not being in the original suggested FC2 list.

### Full KITTI 01

| Run | Cue | Bias | Layer mode | Strength | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Delta vs RFR-100 (m) | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| FC2-01 | `dyn` | pair | early | beta=1.0 | 41.0733 | 9.0158 | 92.3818 | 0.0085 | 0.0000 | reference reproduced |
| FC2-02 | `dyn_reliable` | protected_pair | early | beta=1.0 | 41.4679 | 9.0387 | 92.3965 | 0.0084 | +0.3946 | failed full despite best 256 |
| FC2-03 | `dyn` | key | early | beta=0.5 | 41.2003 | 9.1682 | 92.3689 | 0.0087 | +0.1270 | better than BL01, worse than RFR-100 |
| FC2-04 | `internal_attn` top10 | protected_pair | early_quarter | beta=1.0 | 41.8235 | 9.0160 | 92.3882 | 0.0084 | +0.7502 | failed |

Full-run interpretation:

- Phase C v2-A did not improve over RFR-100. The best FineGate full result remains the RFR-100 reference at `41.0733 m`.
- The most important negative result is F256-02 to FC2-02: reliability-filtered protected-pair looked best on 256 frames but degraded full ATE to `41.4679 m`. This confirms that 256-frame wins are not sufficient for full-sequence model selection.
- Key-only beta=0.5 was more robust than the reliability-filtered protected-pair full run, but still worse than RFR-100 and worse in rotation. It should not replace the current best read-path candidate.
- Internal top10 early-quarter was conservative in short runs but failed badly on the full sequence, so current `internal_attn` is not a reliable motion cue for Phase C.

Phase C v2 verdict:

- Engineering and experiment execution: PASS. The new knobs ran through F64, F256, and full KITTI 01 with real HMC hooks and wandb logging.
- Read-path-only success: NOT PASS. No FineGate candidate beat RFR-100, and no run crossed `<41.0 m`, `<=40.8 m`, or the strong `<=40.5 m` target.
- Phase D readiness: NOT READY. Do not combine these read-path controls with TTT branch0 yet; the full-sequence evidence says the current frame-attention variants are at or below the RFR-100 ceiling.

Recommended next step:

- Stop sweeping frame-attention beta/protection around the current `C_dyn` family. The useful signal exists, but the current cue is not reliable enough globally.
- If continuing Phase C, prioritize a new motion cue source before more combinations: stronger Gram-lite Q/K statistics, semantic-protected dynamic maps, or optical-flow / epipolar consistency. SWA read should only be tested if its dashboard clearly shows protected previous-key suppression without hurting overlap/reference tokens.

Artifacts:

- `results/kitti01_hmc_v2/phaseC_v2_finegate/F64_01_dyn_pair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F64_02_dyn_key_early_b050/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F64_03_dyn_protpair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F64_04_dynrel_protpair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F64_05_dyn_top10_protpair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F64_06_internal_protpair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F64_07_internal_top10_protpair_earlyq_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F256_01_dyn_pair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F256_02_dynrel_protpair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F256_03_dyn_top10_protpair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F256_04_dyn_top10_protpair_earlyq_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F256_06_internal_top10_protpair_earlyq_b100/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/F256_07_dyn_key_early_b050/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/FC2_01_dyn_pair_early_b100_full/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/FC2_02_dynrel_protpair_early_b100_full/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/FC2_03_dyn_key_early_b050_full/`
- `results/kitti01_hmc_v2/phaseC_v2_finegate/FC2_04_internal_top10_protpair_earlyq_b100_full/`

Wandb:

- `hmc_v2_PhaseC_v2_finegate_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/hy8t88au

## Pipeline v2 / HMC Phase C v3 SignalGate

Plan source: `docs/pipeline2/KITTI01_Pipelinev2_PhaseC_v3_SignalGate_Experiment_Plan_Typora.md`.

Purpose: stop tuning the old `C_dyn` family and test whether a better read-path motion cue can survive a stricter sequence of gates: HMC correctness, cue-source smoke, 256-frame check, stateful long-memory slices, then at most one full KITTI 01 candidate.

Implementation added:

- `--read_cue_source` now supports `old_dyn`, `gram_lite`, `gram4d`, `entropy`, `flow`, `flow_sem_veto`, `random`, and `inverted_dyn` in addition to previous Phase C sources.
- `--read_path frame|swa|chunk|ttt_apply` is a single-path selector that maps to the existing read-control booleans.
- Added Phase C v3 bookkeeping args: `--flow_model`, `--flow_pair_stride`, `--flow_fb_thr`, `--flow_residual_thr`, `--gram_layer_groups`, `--chunk_bias_mode`, `--swa_bias_mode`, `--stateful_slice_mode`, `--stateful_slice_starts`, `--stateful_slice_len`, `--save_hmc_states`, and `--load_hmc_state_at_chunk`.
- `run_pipeline_abc_v2.py` now writes `cue_quality_per_chunk.jsonl`, `cue_quality_summary.json`, `hook_effect_summary.jsonl`, and `hmc_correctness_summary.json`.
- HMC committed states can now be saved per chunk and loaded for stateful continuation slices.
- Slice trajectory timestamps now preserve KITTI global frame ids parsed from image filenames, so middle-sequence slices align against the correct GT frames.

Important limitation:

- `flow_sem_veto` currently uses a patch-match / Stage-B reprojection proxy (`--stageb_proxy_mode reprojection --flow_model patch_match`), not RAFT or GMFlow. It is a stronger internal proxy than old same-pixel `C_dyn`, but it is not yet a true external optical-flow residual.
- SWA and chunk-attention dense non-identity controls remain deferred. The current valid full candidate is frame-attention key suppression only.

### C3-0 Correctness / Sanity

| Run | Purpose | ATE RMSE (m) | Rot RMSE (deg) | Result |
|---|---|---:|---:|---|
| C3-0A no-control LoGeR stateful full | save committed HMC states and revalidate native | 41.7502 | 8.9928 | PASS, exact A3 reproduction |
| C3-0D random frame-key 64 | non-identity hook effect sanity | 1.0443 | 2.6048 | output changes, hook active |
| C3-0D inverted-dyn frame-key 64 | sign/effect sanity | 1.0020 | 2.8423 | output changes, hook active |

Correctness diagnostics:

- `probe_no_commit_hash_equal_all`: true for 38 / 38 chunks.
- `state_double_write_safe_all`: true for 38 / 38 chunks.
- max no-control pass1/pass2 pose translation diff: `0.0 m`.
- full no-control hook trace totals: frame attention `684`, SWA read `152`, TTT apply `684`, chunk attention `684`, all with zero bias.

### C3-1 / C3-3 Short Gates

| Run | Cue | Bias | Layer | Strength | Frames | ATE RMSE (m) | Rot RMSE (deg) | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| RFR-100 reference | old `C_dyn` | pair | early | beta=1.0 | 64 | 1.0058 | 2.5517 | previous reference |
| C31B | `gram_lite` | key | early | beta=1.0 | 64 | 0.9796 | 2.7205 | useful smoke |
| C31C | `gram4d` | key | early | beta=1.0 | 64 | 0.9386 | 2.6826 | strong smoke |
| C31D | `entropy` proxy | key | early | beta=1.0 | 64 | 1.0438 | 2.5926 | dropped |
| C31F | `flow_sem_veto` proxy | key | early | beta=1.0 | 64 | 0.9163 | 2.7308 | best smoke |
| RFR-100 reference | old `C_dyn` | pair | early | beta=1.0 | 256 | 3.8666 | 4.2789 | previous reference |
| C31B | `gram_lite` | key | early | beta=1.0 | 256 | 3.8648 | 3.7181 | near reference |
| C31C | `gram4d` | key | early | beta=1.0 | 256 | 3.8534 | 3.8378 | useful |
| C31F | `flow_sem_veto` proxy | key | early | beta=1.0 | 256 | 3.8000 | 3.9007 | best 256 |
| C31F | `flow_sem_veto` proxy | key | early | beta=0.75 | 256 | 3.8230 | 3.8338 | weaker than beta=1.0 |
| C31C | `gram4d` | key | early | beta=0.75 | 256 | 3.8756 | 3.8507 | weaker |

Cue-quality note:

- `flow_sem_veto` had the best 64/256 ATE, but full-run cue quality later showed mean dynamic mass only `0.010` over all chunks. It is therefore a sparse proxy and should be treated as a cautious promote, not a clean Phase C cue pass.

### C3-2 Stateful Slice Gate

Candidate: `flow_sem_veto`, frame-attention key suppression, early layers, beta `1.0`.

| Chunk | Frames | No-control ATE (m) | Candidate ATE (m) | Delta ATE (m) | Improved |
|---:|---|---:|---:|---:|---|
| 0 | `[0,128)` | 1.0204 | 1.0490 | +0.0286 | no |
| 5 | `[145,273)` | 41.0849 | 41.0653 | -0.0196 | yes |
| 10 | `[290,418)` | 29.6698 | 29.4409 | -0.2289 | yes |
| 15 | `[435,563)` | 16.6805 | 16.6076 | -0.0729 | yes |
| 20 | `[580,708)` | 34.2306 | 34.0583 | -0.1723 | yes |
| 25 | `[725,853)` | 35.9329 | 35.8356 | -0.0973 | yes |
| 30 | `[870,998)` | 4.8915 | 4.8408 | -0.0507 | yes |

Slice verdict:

- PASS by the planned numeric gate: `6 / 7` slices improved, and the only worse slice was `+0.0286 m`, below the `+0.15 m` bad-tail threshold.
- However, the full run below shows this slice gate is still too permissive. It catches broad local continuation wins but not full-sequence global cue sparsity / trajectory coupling.

### C3-4 Full Candidate

| Run | Cue | Path | Bias | Layer | Strength | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Delta vs RFR-100 (m) | Verdict |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| RFR-100 reference | old `C_dyn` | frame | pair | early | 1.0 | 41.0733 | 9.0158 | 92.3818 | 0.0085 | 0.0000 | reference |
| FC3-02 | `flow_sem_veto` proxy | frame | key | early | 1.0 | 41.4333 | 9.1252 | 92.3907 | 0.0085 | +0.3600 | failed full |

Full-run interpretation:

- Phase C v3 did not pass. The best new cue passed 64-frame, 256-frame, and the stateful slice gate, but failed full KITTI 01 by `+0.3600 m` versus RFR-100.
- This is a stronger negative result than Phase C v2: even stateful slices can be false-positive when the cue is too sparse or when local Sim(3)-aligned continuation wins do not preserve full-sequence trajectory consistency.
- The full `flow_sem_veto` cue has mean dynamic mass `0.010`, below the plan's desired `0.03-0.25` range, and mean `Corr(D,Conf)=0.680`. It is not just a dynamic-object detector; it is partly a high-confidence sparse residual selector.
- Do not start Phase D. Read-path-only still has not crossed `<41.0 m`, let alone the formal `<=40.8 m` Phase C v3 pass bar.

Recommended next step:

- Tighten C3-2 before any more full runs: require full-run-like cue quality over all chunks, especially mean dynamic mass inside `0.03-0.25`, and add global end-drift / scale-continuity checks, not only local slice ATE.
- If continuing Phase C, implement a true external optical-flow residual (`RAFT` or `GMFlow`) or a learned semantic/motion reliability gate. The current patch-match/reprojection proxy is informative but insufficient.
- Keep SWA/chunk-attention non-identity experiments blocked until their bias can be applied sparsely without dense all-token masks and their dashboards show protected previous/reference keys are not being suppressed.

Artifacts:

- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C30A_no_control_loger_stateful_full/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C30D_random_frame_key_b100_64/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C30D_invdyn_frame_key_b100_64/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31B_gramlite_frame_key_b100_64/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31C_gram4d_frame_key_b100_64/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31D_entropy_frame_key_b100_64/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31F_flowsem_frame_key_b100_64/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31B_gramlite_frame_key_b100_256/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31C_gram4d_frame_key_b100_256/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31F_flowsem_frame_key_b100_256/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31F_flowsem_frame_key_b075_256/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/C31C_gram4d_frame_key_b075_256/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/slices/`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/stateful_slice_summary.csv`
- `results/kitti01_hmc_v2/phaseC_v3_signalgate/FC3_02_flowsem_frame_key_b100_full/`

Wandb:

- `hmc_v2_PhaseC_v3_signalgate_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/l5l7w8qd

## Pipeline v2 / HMC Phase C v4 Global-Safe Cue Gate

Plan source: `docs/pipeline2/KITTI01_Pipelinev2_PhaseC_v4_GlobalSafeCueGate_Experiment_Plan_Typora.md`.

Purpose: tighten the Phase C gate after v3 showed that short-sequence and local Sim(3) slice wins can be false positives. Phase C v4 first audits full-sequence cue quality, then only allows globally safe cues into controlled/slice/full experiments.

Implementation added:

- `--read_cue_source` now supports `flow_proxy`, `flow_sem_veto_proxy`, `flow_proxy_calib`, `flow_sem_veto_calib`, `old_dyn_plus_flow_proxy`, `old_dyn_plus_flow_sem_veto`, `old_dyn_switch_flow_proxy`, and `old_dyn_switch_flow_sem_veto`.
- New calibration/fallback knobs: `--read_calib_mode`, `--read_target_mass`, `--read_calib_tau`, `--read_blend_lambda`, `--read_quality_mass_min`, `--read_quality_mass_max`, `--read_quality_anchor_max`, and `--read_quality_frag_max`.
- `cue_quality_summary.json` now records dynamic-mass coverage, anchor collision, fragmentation, old-dyn IoU/coverage/recall, cue quality pass fraction, cue gate, and fallback rate.
- `probe_only` now runs Stage-B cue extraction and commits the probe native provisional state between chunks, so full-run cue audit is stateful without doing Pass 2.
- Added `tools/hmc_global_slice_eval.py` for the planned global-fixed slice gate. It was not promoted to use here because no new cue passed C4-1.

Important limitation:

- The flow cue is still the Stage-B reprojection / patch-match proxy (`--stageb_proxy_mode reprojection --flow_model patch_match`), not true RAFT/GMFlow optical flow.

### C4-1 Full-Run Stateful Probe Cue Audit

| Run | Cue | Target / blend | Mean mass `D>0.5` | Chunk coverage | Anchor collision | Fragmentation | Corr(D,Conf) | IoU old_dyn | Fallback | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| P-A | `old_dyn` | native | 0.4650 | 1.0000 | 0.0328 | 0.0141 | 0.0004 | 1.0000 | 0.0000 | reference, dense above v4 mass band |
| P-B | `flow_proxy_calib` | q=0.06/tau=0.05 | 0.0601 | 1.0000 | 0.1060 | 0.2363 | 0.2071 | 0.1241 | 0.0000 | fails fragmentation |
| P-C | `old_dyn_plus_flow_proxy` | lambda=0.50, q=0.06 | 0.4434 | 1.0000 | 0.0412 | 0.0180 | 0.0520 | 0.8537 | 0.0000 | too dense, mostly old_dyn |
| P-D | `old_dyn_switch_flow_proxy` | q=0.06 | 0.1362 | 0.5789 | 0.0642 | 0.1673 | 0.2745 | 0.3136 | 0.1324 | fails coverage/fragmentation |
| P-E | `flow_proxy_calib` | q=0.10/tau=0.10 | 0.0993 | 1.0000 | 0.1149 | 0.1774 | 0.2063 | 0.1857 | 0.0000 | safer mass, still fragmented |
| P-F | `old_dyn_switch_flow_proxy` | q=0.10/tau=0.10 | 0.1463 | 0.6053 | 0.0600 | 0.1638 | 0.2949 | 0.3548 | 0.1720 | fails coverage/fragmentation |

Audit interpretation:

- Calibration fixed the v3 sparsity problem, but did not produce a full-sequence-safe cue. `flow_proxy_calib q=0.06` hits the target mass almost exactly and has full chunk coverage, but fragmentation is too high.
- Increasing to `q=0.10/tau=0.10` reduces the controlled 256-frame risk somewhat but still leaves full-run fragmentation above the v4 threshold.
- Blend and switch variants do not solve the global gate. Blend stays too close to dense old_dyn; switch has low effective coverage because the quality gate often falls back or produces near-empty chunks.

### C4-2 / C4-4 256-Frame Controlled Gate

| Run | Cue | Bias | Layer | Strength | ATE RMSE (m) | Rot RMSE (deg) | Mean mass | Fragmentation | Delta vs RFR-256 | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| G256-00 | `old_dyn` | pair | early | beta=1.0 | 3.8666 | 4.2789 | 0.3009 | 0.0246 | 0.0000 | RFR reference |
| G256-01 | `flow_proxy_calib` q=0.06 | pair | early | beta=0.75 | 3.7706 | 4.0398 | 0.0603 | 0.1709 | -0.0960 | best 256, fails full cue audit |
| G256-02 | `flow_proxy_calib` q=0.06 | pair | early | beta=1.0 | 3.8251 | 4.1699 | 0.0603 | 0.1705 | -0.0415 | useful 256, fails audit |
| G256-03 | `old_dyn+0.25 flow_proxy` | pair | early | beta=1.0 | 3.9111 | 4.2356 | 0.2990 | 0.0241 | +0.0445 | worse |
| G256-04 | `old_dyn+0.50 flow_proxy` | pair | early | beta=1.0 | 3.8497 | 4.1199 | 0.2234 | 0.0365 | -0.0169 | tiny gain, mass high |
| G256-05 | `old_dyn switch flow_proxy` q=0.06 | pair | early | beta=1.0 | 3.8587 | 4.1123 | 0.1026 | 0.1197 | -0.0079 | tiny gain, audit coverage fails |
| G256-06 | `flow_proxy_calib` q=0.10/tau=0.10 | pair | early | beta=0.75 | 3.8616 | 4.1451 | 0.0992 | 0.1307 | -0.0050 | safer but no useful gain |
| G256-07 | `flow_sem_veto_calib` q=0.10/tau=0.10 | pair | early | beta=0.75 | 3.8670 | 4.1988 | 0.1005 | 0.1312 | +0.0004 | no gain |
| G256-08 | `old_dyn switch flow_proxy` q=0.10/tau=0.10 | pair | early | beta=1.0 | 3.8716 | 4.1894 | 0.1431 | 0.0847 | +0.0050 | no gain |

256 interpretation:

- The best short result is again not globally safe: G256-01 improves by `0.0960 m` but the same cue fails full-run fragmentation.
- The safer q=0.10 variants remove most of the 256-frame advantage. This is useful negative evidence: the apparent gain comes from a sharper/specklier residual selector, not a stable full-sequence cue.
- No candidate satisfied both C4-1 cue quality and a meaningful 256-frame improvement.

Phase C v4 verdict:

- Engineering / logging: PASS. New calibration, fallback, stateful probe audit, and global-slice evaluation tooling are implemented and logged to wandb.
- C4-1 cue gate: NOT PASS. No new cue satisfies full-sequence-safe mass/coverage/fragmentation jointly.
- C4-3 / full candidates: NOT RUN by design. Since no cue passed the global-safe audit, running stateful slices or full KITTI 01 candidates would violate the v4 plan and likely recreate a v3-style false positive.
- Phase C read-path-only remains NOT PASS. RFR-100 (`41.0733 m`) remains the best read-path-only result, and Phase D should not be started from these new proxy cues.

Recommended next step:

- Stop hand-tuning patch-match/reprojection flow-proxy read controls. The proxy has short-window signal, but after calibration it either becomes fragmented or loses its ATE gain.
- If Phase C continues, implement true external flow residual (RAFT/GMFlow) or a learned reliability gate. Otherwise move to a clearly labeled Phase D combination experiment using RFR-100 as a diagnostic read-path reference, not as a passed Phase C component.

Artifacts:

- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_00_olddyn_pair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_01_flowcalib_q006_pair_early_b075/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_02_flowcalib_q006_pair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_03_olddynplusflow_l025_q006_pair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_04_olddynplusflow_l050_q006_pair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_05_olddynswitchflow_q006_pair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_06_flowcalib_q010t010_pair_early_b075/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_07_flowsemcalib_q010t010_pair_early_b075/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/G256_08_olddynswitchflow_q010t010_pair_early_b100/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/P_A_stateful_olddyn_probe_full/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/P_B_stateful_flowcalib_probe_full/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/P_C_stateful_olddynplusflow_l050_probe_full/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/P_D_stateful_olddynswitchflow_probe_full/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/P_E_stateful_flowcalib_q010t010_probe_full/`
- `results/kitti01_hmc_v2/phaseC_v4_globalsafe/P_F_stateful_olddynswitchflow_q010t010_probe_full/`

Wandb:

- `hmc_v2_PhaseC_v4_globalsafe_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/9fgyoecl

## Pipeline v2 / HMC Phase C v5 Commit-Safe Read-Path

Plan source: `docs/pipeline2/KITTI01_Pipelinev2_PhaseC_v5_CommitSafe_ReadPath_Experiment_Plan_Typora.md`.

Purpose: test whether the best Phase C read-path control, RFR-100, was limited by the cue itself or by committing controlled read-path side effects into future HMC memory. This phase stops tuning patch-match flow proxies and instead isolates controlled output from memory commit.

Implementation added:

- `--hmc_commit_mode controlled|probe_native|split_ttt_native`.
- `controlled` commits the Pass-2 controlled state, matching the previous RFR-100 behavior.
- `probe_native` keeps the controlled geometry output but commits the Pass-1 native provisional state.
- `split_ttt_native` keeps the controlled geometry output but commits native/probe TTT fast weights. In the current HMC state, this is numerically equivalent to `probe_native` because SWA/ref states are not yet separate committed tensors.
- `hmc_state_hash.jsonl` now records selected commit-state hashes and memory side-effect summaries between controlled and probe/native states.
- Memory side-effect logging records TTT state mean/max relative diff plus branch-wise `w0/w1/w2` diffs.
- Bias-energy normalization knobs were added for the optional BE gate, but BE runs were not executed because the commit-isolation gate already crossed the main-candidate threshold.

### C5-0 Correctness Reproduction

| Run | Model | Commit mode | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | HMC correctness | Verdict |
|---|---|---|---:|---:|---:|---:|---|---|
| C50A | LoGeR | controlled / no-control | 41.7502 | 8.9928 | 92.3961 | 0.0084 | 38 / 38 probe-safe, double-write-safe | PASS, exact A3 |
| C50B | LoGeR* | controlled / no-control | 47.9793 | 5.8502 | 90.7286 | 0.0075 | 18 / 18 probe-safe, double-write-safe | PASS, exact A4 |

Correctness diagnostics:

- No-control LoGeR hook totals: frame attention `684`, SWA read `152`, TTT apply `684`, chunk attention `684`, all with max bias `0.0`.
- No-control LoGeR* hook totals: frame attention `324`, SWA read `72`, TTT apply `324`, chunk attention `324`, all with max bias `0.0`.
- No-control pass1/pass2 pose translation and pose-matrix diffs remain exactly `0.0`.

### C5-1 Commit-Mode Isolation

All rows use RFR-100 read-path control: `old_dyn`, frame-attention `pair` bias, early layers, `beta=1.0`, read-path-only mode.

| Run | Commit mode | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Delta vs CM01 (m) | Commit hash check | Mean TTT side-effect | Verdict |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| CM01 | controlled | 41.0733 | 9.0158 | 92.3818 | 0.0085 | 0.0000 | `hash_H_next == controlled` for 38 / 38 | 0.018061 | RFR-100 reproduced |
| CM02 | probe_native | 39.7820 | 9.7417 | 92.3953 | 0.0096 | -1.2913 | `hash_H_next == probe_native` for 38 / 38 | 0.018118 | strong pass, main candidate |
| CM03 | split_ttt_native | 39.7820 | 9.7417 | 92.3953 | 0.0096 | -1.2913 | `hash_H_next == probe_native` for 38 / 38 | 0.018118 | same as CM02 |
| CM02R | probe_native repeat | 39.7820 | 9.7417 | 92.3953 | 0.0096 | -1.2913 | `hash_H_next == probe_native` for 38 / 38 | 0.018118 | deterministic repeat |

Memory side-effect interpretation:

- RFR-100 controlled output changes the next TTT memory state: max TTT relative diff reaches `0.111240`, and average mean relative diff is about `0.0181`.
- CM01 commits that altered memory, while CM02/CM03 use the same controlled output but commit native/probe memory instead.
- CM02 and CM03 matching exactly indicates the current long-term side effect is carried by TTT fast weights; there is no separate SWA/ref committed tensor path changing the result in this setup.

Trajectory diagnostics:

| Run | Aligned ATE (m) | Sim(3) scale | Largest aligned-error axis | Final error (m) | 200-frame mean ATE (m) |
|---|---:|---:|---|---:|---:|
| RFR100 controlled | 41.0733 | 30.659544 | x | 2.907 | 36.868 |
| RFR100 probe_native | 39.7820 | 30.762480 | x | 5.589 | 35.299 |
| RFR100 split_ttt_native | 39.7820 | 30.762480 | x | 5.589 | 35.299 |

Interpretation:

- Phase C v5 confirms the key hypothesis: RFR-100's read-path signal is much stronger when controlled read output is not allowed to contaminate future memory commits.
- CM02 improves by `1.2913 m` over RFR-100 controlled, `1.5845 m` over BL01 TTT-write best (`41.3665 m`), and `1.9682 m` over A3 native (`41.7502 m`).
- CM02 crosses all Phase C v5 success bars: `<41.0 m`, `<=40.8 m`, `<=40.5 m`, and the `<40.0 m` main-candidate precondition.
- The tradeoff is clear: rotation worsens from `9.0158 deg` to `9.7417 deg`, and final aligned error increases. The ATE gain appears to come from better full-sequence trajectory shape / mid-sequence alignment, not from uniformly better orientation.
- BE and true-flow rescue experiments were skipped intentionally: FC5-01 already met the strongest v5 gate, so the plan's stop condition favors moving to Phase D instead of adding more read-path-only full runs.

Phase C v5 verdict:

- Engineering / correctness: PASS.
- Commit-isolation hypothesis: PASS.
- Phase C read-path-only: STRONG PASS, with best ATE `39.7820 m`.
- Phase D readiness: PASS, but Phase D must preserve the v5 isolation pattern: controlled read-path output with native/probe TTT memory commit, then combine with branch0 TTT write control carefully.

Artifacts:

- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/SMOKE64_controlled/`
- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/SMOKE64_probe_native/`
- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/SMOKE64_split_ttt_native/`
- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/C50A_no_control_loger_full/`
- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/C50B_no_control_loger_star_full/`
- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/CM01_rfr100_controlled_full/`
- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/CM02_rfr100_probe_native_full/`
- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/CM03_rfr100_split_ttt_native_full/`
- `results/kitti01_hmc_v2/phaseC_v5_commitsafe/CM02R_rfr100_probe_native_full_repeat/`
- `results/kitti01_hmc_v2/trajectory_diagnostics/phaseC_v5_commit_modes/`

Wandb:

- `hmc_v2_PhaseC_v5_commitsafe_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/xc71nyfb

## Pipeline v2 / HMC Phase D v5 Commit-Safe Read + Branch0 Write

Plan source: Phase D combination rule from `docs/pipeline2/Pipelinev2_Experiment_Plan_Typora.md`, using the Phase C v5 commit-safe result from `docs/pipeline2/KITTI01_Pipelinev2_PhaseC_v5_CommitSafe_ReadPath_Experiment_Plan_Typora.md`.

Purpose: combine the Phase C v5 read-path controller with the BL01 branch0 TTT write controller without reintroducing controlled read-path memory contamination. The critical design is to use controlled Pass-2 geometry as output while committing a native/probe TTT state updated by the branch0 semantic write prior.

Implementation added:

- `--hmc_commit_mode probe_ttt_write`.
- `probe_ttt_write` keeps the controlled read-path output but builds the next committed HMC state from the Pass-1 native/probe TTT cache plus the selected TTT write prior.
- `hybrid` mode now reports both `ttt_update` and active read hook paths in `control_trace.implemented_paths`.
- Smoke check confirmed `hash_H_next` is distinct from both pure controlled-output state and pure probe-native state, while `prior_ttt_write_present=true`.

Shared read/write config:

- Read path: `old_dyn`, frame-attention `pair` bias, early decoder layers.
- Write path: BL01 branch0 dynamic MP-01, `--prior_branch_mask 0`, `--prior_layer_mode all`.
- Stage C disabled; Stage-B cue settings match Phase C v5 / BL01.
- Main commit mode: `probe_ttt_write`.

Full KITTI 01:

| Run | Mode | Commit | Beta policy | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Delta vs BL01 (m) | Delta vs CM02 read-only (m) | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| BL01 TTT write | TTT branch0 | controlled | n/a | 41.3665 | 8.9490 | 92.3947 | 0.0083 | 0.0000 | +1.5845 | write baseline |
| D5-02 | hybrid | controlled | beta=1.00 | 41.0221 | 9.0510 | 92.3842 | 0.0085 | -0.3444 | +1.2401 | naive commit still contaminated |
| D5-03 | hybrid | probe_native | beta=1.00 | 39.7820 | 9.7417 | 92.3953 | 0.0096 | -1.5845 | 0.0000 | equals read-only, TTT write discarded |
| D5-04 | hybrid | probe_ttt_write | beta=1.00 | 39.5127 | 9.7345 | 92.3937 | 0.0096 | -1.8538 | -0.2693 | strong combo |
| D5-05 | hybrid | probe_ttt_write | beta=0.75 | 39.5861 | 9.6165 | 92.3932 | 0.0094 | -1.7804 | -0.1959 | more rotation-safe |
| D5-06 | hybrid | probe_ttt_write | beta=0.50 | 39.7739 | 9.4839 | 92.3934 | 0.0092 | -1.5926 | -0.0081 | safer rotation, little combo gain |
| D5-07 | hybrid | probe_ttt_write | beta=1.25 | 39.4903 | 9.8299 | 92.3937 | 0.0097 | -1.8762 | -0.2917 | best ATE |
| D5-07R | hybrid | probe_ttt_write | beta=1.25 | 39.4903 | 9.8299 | 92.3937 | 0.0097 | -1.8762 | -0.2917 | deterministic repeat |
| D5-08 | hybrid | probe_ttt_write | beta=1.50 | 39.5279 | 9.9083 | 92.3940 | 0.0098 | -1.8386 | -0.2541 | too much rotation damage |
| BE-02 | read only | controlled | bias-energy norm | 40.9629 | 9.0031 | 92.3816 | 0.0085 | -0.4036 | +1.1809 | small BE gain only |
| BE-03 | read only | probe_native | bias-energy norm | 39.7711 | 9.7802 | 92.3953 | 0.0096 | -1.5954 | -0.0109 | tiny ATE gain, worse rotation |
| BE-04 | read only | controlled | BE early-quarter | 40.9217 | 9.1589 | 92.3678 | 0.0087 | -0.4448 | +1.1397 | diagnostic only |
| BE-H | hybrid | probe_ttt_write | bias-energy norm | 39.5251 | 9.7667 | 92.3940 | 0.0097 | -1.8414 | -0.2569 | worse than fixed beta best |

Trajectory diagnostics:

| Run | Aligned ATE (m) | Final error (m) | 50-frame mean ATE (m) | 100-frame mean ATE (m) | 200-frame mean ATE (m) |
|---|---:|---:|---:|---:|---:|
| BL01 | 41.3665 | 3.590 | 36.704 | 37.170 | 37.177 |
| ReadOnly_CM02 | 39.7820 | 5.589 | 34.887 | 35.380 | 35.299 |
| D5_b100 | 39.5127 | 6.058 | 34.599 | 35.094 | 35.042 |
| D5_b125 | 39.4903 | 6.311 | 34.590 | 35.081 | 35.043 |
| D5_BEH | 39.5251 | 6.192 | 34.604 | 35.101 | 35.048 |

Worst D5-b125 chunks:

| Chunk | Frames | RMSE (m) | Worst frame |
|---:|---|---:|---:|
| 8 | `[232,264)` | 87.73 | 255 |
| 9 | `[261,293)` | 74.47 | 264 |
| 7 | `[203,235)` | 70.92 | 228 |
| 15 | `[435,467)` | 65.06 | 463 |
| 16 | `[464,496)` | 63.54 | 464 |

Interpretation:

- Phase D v5 passes the combination gate. The correct commit isolation pattern is `probe_ttt_write`, not naive `controlled` and not pure `probe_native`.
- D5-07 is the best ATE result so far at `39.4903 m`, improving by `1.8762 m` over BL01 and `0.2917 m` over the Phase C v5 read-only CM02 result.
- D5-04 beta `1.00` is nearly as good in ATE (`+0.0224 m` versus D5-07) and has better rotation (`9.7345` vs `9.8299`). It is the more balanced candidate if rotation is weighted.
- Bias-energy normalization is not the missing piece. It helps controlled read-only slightly, but does not improve the best commit-safe hybrid result.
- The tradeoff remains: ATE and segment means improve, but final aligned error and rotation worsen versus BL01. The next phase should add rotation/reference protection rather than more beta sweeps.
- True-flow rescue remains BLOCKED, not skipped for lack of interest: the current implementation has only patch-match / Stage-B flow proxy logic. The v5 plan requires real RAFT/GMFlow residual and explicitly disallows proxy flow as a substitute for FC5-03.

Phase D v5 verdict:

- Engineering / commit isolation: PASS.
- Read + write complementarity: PASS.
- Best ATE candidate: D5-07 beta `1.25`, `39.4903 m`.
- Balanced candidate: D5-04 beta `1.00`, `39.5127 m` with less rotation damage.
- Main-candidate bar from the original combination plan (`<38.0 m`) is not reached, so this is a strong Phase D result but not a final main model.

Artifacts:

- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/SMOKE64_hybrid_probe_ttt_write/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_00_ttt_branch0_bl01_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_02_hybrid_controlled_b100_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_03_hybrid_probe_native_b100_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_04_hybrid_probe_ttt_write_b100_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_05_hybrid_probe_ttt_write_b075_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_06_hybrid_probe_ttt_write_b050_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_07_hybrid_probe_ttt_write_b125_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_07R_hybrid_probe_ttt_write_b125_full_repeat/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_08_hybrid_probe_ttt_write_b150_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_BE02_rfr100_BE_controlled_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_BE03_rfr100_BE_probe_native_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_BE04_rfr100_BE_earlyq_controlled_full/`
- `results/kitti01_hmc_v2/phaseD_v5_commit_safe/D5_BEH_hybrid_probe_ttt_write_BE_full/`
- `results/kitti01_hmc_v2/trajectory_diagnostics/phaseD_v5_commit_safe/`

Wandb:

- `hmc_v2_PhaseD_v5_commit_safe_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/xlqfsr7j

## Pipeline v2 / HMC Phase E Read/Write Co-Design Rotation-Safe

Plan source: `docs/pipeline2/KITTI01_Pipelinev2_PhaseE_ReadWrite_CoDesign_RotationSafe_Plan_Typora.md`.

Purpose: keep the Phase D v5 commit-safe read+write ATE gain while trying to reduce rotation and endpoint damage. Phase E tested read-side reference protection, read cue modulation, update-needed TTT write from probe-cache residuals, sparse exact-preserve write, and a small read/write pair search. All valid full runs keep the Phase D commit rule: controlled Pass-2 geometry output with `probe_ttt_write` commit.

Implementation added:

- `--read_protection_mode none|overlap|anchor|high_anchor|static|reset|ref|attention|attn|combined_light|combined_strong`.
- `--read_ref_strength`, `--read_overlap_frames`, `--read_reset_frames`, and `--read_attention_q` for Phase E protected read maps.
- `--hmc_write_score_source stage_d|old_dyn|ttt_residual|residual_reliability|alignment_confidence` plus `--hmc_write_sparse_ratio` and `--hmc_write_sparse_mode`.
- `old_dyn_gram_lite_agree`, `old_dyn_gram4d_agree`, and `old_dyn_key_static_rescue` read cue sources for E2 modulation.
- Probe-cache TTT residual write scores are computed from `apply_output_raw` versus `v`, then combined with read reliability / uncertainty / occlusion / reference protection when requested.

Shared config:

- `--hybrid_memory_mode hybrid`
- `--hmc_commit_mode probe_ttt_write`
- `--stage_c_mode none`
- Read path: frame attention, `old_dyn` unless otherwise noted, `pair` bias, early decoder layers.
- Write path baseline: BL01 branch0 dynamic MP-01, `--prior_branch_mask 0`, `--prior_layer_mode all`.
- Stage-B cue settings match Phase D v5.

Full KITTI 01:

| Run | Purpose | Read / write variant | Beta | ATE RMSE (m) | Rot RMSE (deg) | RPE t (%) | RPE r (deg/100m) | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| E0-D5-04R | balanced repeat | D5 old_dyn + BL01 write | 1.00 | 39.5127 | 9.7345 | 92.3937 | 0.0096 | exact repeat |
| E0-D5-07R | best ATE repeat | D5 old_dyn + BL01 write | 1.25 | 39.4903 | 9.8299 | 92.3937 | 0.0097 | exact repeat |
| E1-overlap | read protection | overlap protected read + BL01 write | 1.00 | 40.5264 | 9.2121 | 92.3874 | 0.0086 | rotation improves, ATE fails |
| E1-anchor | read protection | anchor protected read + BL01 write | 1.00 | 39.5291 | 9.7454 | 92.3936 | 0.0096 | near D5-04, no gain |
| E1-combined-light | read protection | overlap+anchor+attention protected read + BL01 write | 1.00 | 41.4207 | 8.9605 | 92.3946 | 0.0084 | rotation-safe but too much ATE loss |
| E3-residual-reliability | update-needed write | old_dyn read + residual x reliability write | 1.00 | 39.5209 | 9.7076 | 92.3949 | 0.0096 | slight rotation help, worse ATE |
| E4-oldDyn-sparse95 | sparse write | old_dyn read + oldDyn sparse95 write | 1.00 | 39.5307 | 9.6327 | 92.3934 | 0.0095 | rotation help, worse ATE |
| E4-resrel-sparse95 | sparse write | old_dyn read + residual sparse95 write | 1.00 | 39.6552 | 9.4967 | 92.3947 | 0.0093 | balanced-ish but ATE too high |
| E5-anchor-sparse95 | pair search | anchor protected read + oldDyn sparse95 write | 1.00 | 39.5111 | 9.6396 | 92.3935 | 0.0095 | best balanced-ish Phase E |
| E5-anchor-resrel | pair search | anchor protected read + residual write | 1.00 | 39.7182 | 9.7850 | 92.3958 | 0.0096 | fails |
| E5-resrel-b125 | pair search | old_dyn read + residual write | 1.25 | 39.4881 | 9.7984 | 92.3950 | 0.0097 | tiny new best |
| E5-resrel-b125 repeat | repeat | old_dyn read + residual write | 1.25 | 39.4881 | 9.7984 | 92.3950 | 0.0097 | deterministic repeat |
| E5-oldDyn-sparse95-b125 | pair search | old_dyn read + oldDyn sparse95 write | 1.25 | 39.5445 | 9.7486 | 92.3937 | 0.0097 | worse ATE |
| E2-oldDyn-GramLite | modulation | oldDyn x Gram-lite agreement + BL01 write | 1.00 | 39.8441 | 9.6976 | 92.3944 | 0.0095 | fails |
| E2-oldDyn-Gram4D | modulation | oldDyn x Gram4D agreement + BL01 write | 1.00 | 39.8048 | 9.6467 | 92.3927 | 0.0094 | fails |
| E2-key-static-rescue | modulation | key/static rescue read + BL01 write | 1.00 | 39.5127 | 9.7345 | 92.3937 | 0.0096 | identity-equivalent to D5-04 |

Trajectory diagnostics:

| Run | Aligned ATE (m) | Final error (m) | 50-frame mean ATE (m) | 100-frame mean ATE (m) | 200-frame mean ATE (m) |
|---|---:|---:|---:|---:|---:|
| BL01 | 41.3665 | 3.590 | 36.704 | 37.170 | 37.177 |
| D5_b100 | 39.5127 | 6.058 | 34.599 | 35.094 | 35.042 |
| D5_b125 | 39.4903 | 6.311 | 34.590 | 35.081 | 35.043 |
| E5_resrel_b125 | 39.4881 | 6.309 | 34.597 | 35.088 | 35.037 |
| E5_anchor_sparse_b100 | 39.5111 | 5.967 | 34.578 | 35.072 | 34.996 |

Worst E5-resrel-b125 chunks:

| Chunk | Frames | RMSE (m) | Worst frame |
|---:|---|---:|---:|
| 8 | `[232,264)` | 87.64 | 255 |
| 9 | `[261,293)` | 74.32 | 264 |
| 7 | `[203,235)` | 70.80 | 228 |
| 15 | `[435,467)` | 65.06 | 463 |
| 16 | `[464,496)` | 63.63 | 464 |

Interpretation:

- Phase E produced a deterministic but tiny ATE improvement: E5-resrel-b125 reaches `39.4881 m`, only `0.0022 m` better than D5-07. This is a valid numerical best but not a meaningful model leap.
- Rotation-safe protection behaves as expected but trades away ATE. Overlap and combined-light protection improve rotation substantially, but full ATE degrades to `40.5264 m` and `41.4207 m`.
- Sparse write helps rotation/final modestly but does not improve ATE. The best balanced-ish result is E5-anchor-sparse95 at `39.5111 m`, `9.6396 deg`, final error `5.967 m`.
- Update-needed residual write is mildly useful at beta `1.25`, but not enough to cross the `39.0 m` Phase E strong target or the `<38.0 m` main-candidate target.
- Gram modulation does not help; both Gram-lite and Gram4D variants fall back toward `39.8 m`.
- True-flow remains BLOCKED: this repo still only has patch-match / Stage-B proxy flow, not a real RAFT/GMFlow residual. Per the Phase E plan, proxy flow was not used as a full candidate.

Phase E verdict:

- Engineering / execution: PASS.
- Baseline correctness: PASS, D5-04 and D5-07 reproduce exactly under Phase E code.
- Rotation-safe co-design: PARTIAL. Some variants reduce rotation, but the ATE cost is too large.
- Best ATE: E5-resrel-b125, `39.4881 m`, deterministic repeat.
- Best balanced candidate: E5-anchor-sparse95, `39.5111 m`, lower rotation than D5-04/D5-07 and lower final error than D5 variants, but not below the main D5 ATE.
- Stop condition: HIT. E1-E5 did not reach `<39.0 m`, did not reach `<38.0 m`, and did not produce a clear balanced pass (`Rot <= 9.5` and `FinalErr <= 5.5` with ATE near 39.5). Do not continue hand-rule sweeps on KITTI 01 without a new cue source.

Recommended next step:

- Keep D5-07 as the best strong baseline and E5-resrel-b125 as the tiny numerical best.
- If optimizing further, implement true external optical-flow / epipolar residual or a learned reliability gate before more Phase E-style hand rules.
- If robustness matters more than tiny ATE, consider E5-anchor-sparse95 as the balanced diagnostic, but it is not a main model candidate.

Artifacts:

- `results/kitti01_hmc_v2/phaseE_readwrite_rotation_safe/`
- `results/kitti01_hmc_v2/trajectory_diagnostics/phaseE_readwrite_rotation_safe/`

Wandb:

- `hmc_v2_PhaseE_readwrite_rotation_safe_seq01`: https://wandb.ai/edward20121127/loger-kitti01-gslwc/runs/kny29fd1
