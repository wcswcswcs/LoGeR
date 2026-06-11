# ACL2 v29B 实验复盘：SemanticKITTI 3D Projection + VideoMasklet SemanticPrior AllMemory

日期：2026-05-23（Asia/Singapore）  
计划文件：`docs/ACL2_v29B_SemanticKITTI3DProjection_VideoMasklet_SemanticPrior_AllMemory_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v29b_semantickitti3dprojection_videomasklet_semanticprior_allmemory/`

本轮原则：只记录实际落盘结果；不把 video-masklet predicted semantic 写成 projected 3D semantic；不把 projection audit、blocked gate、或未启动矩阵写成 deployable online success。SemanticKITTI projection frame gate 没有通过时，不生成 masklet-3D alignment，不启动 no-op smoke、memory candidate、selector 或 full online validation。

---

## 0. 当前结论

v29B 合法停止在 Phase 0 SemanticKITTI projection hard gate，未达到 Target-25。

已完成并落盘：

```text
1. 阅读 v29B 计划，确认本轮硬前置是：
       SemanticKITTI velodyne/*.bin
       SemanticKITTI labels/*.label
       KITTI image_2
       calib.txt with P2 and Tr/Tr_velo_to_cam
   必须同时覆盖目标 frames。

2. 新增 v29B projection audit:
       tools/v29b_semantickitti_projection_audit.py

3. 执行 Phase 0 projection prerequisite scan:
       chunks = 6,10,16
       horizons = 10,15
       unique frames expected = 757

4. Phase 0 gate = fail:
       projection_frame_hit_rate = 0.0
       projection_frames_hit = 0 / 757
       sequence_has_image_2 = true
       sequence_has_calib = true
       sequence_has_velodyne = false
       sequence_has_labels = false
       no_predicted_fallback_flag = true
```

最终边界：

```text
1. No SemanticKITTI projection cache was generated.
2. No projected 3D semantic pixel/token tensors were generated.
3. No masklet_3d_alignment.csv was generated.
4. No masklet trust calibration was generated.
5. No no-op semantic smoke was launched.
6. No Phase 2 memory candidate was launched.
7. No masklet causal bank was launched.
8. No no-GT selector was launched.
9. No full online validation was launched.
10. No online Target-25 result was produced in v29B.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v29B 计划允许使用 SemanticKITTI projected 3D labels 作为 sparse GT diagnostic，
但本地 KITTI01 sequence 01 不具备投影所需 velodyne/labels/calib transform。

因此本轮不能把 video-masklet predicted semantic fallback 伪装成 projected 3D semantic。
这不是 semantic route 的性能负结果，而是 projection data hard gate 失败。
```

---

## 1. 工程修改

新增：

```text
tools/v29b_semantickitti_projection_audit.py:
    strict Phase 0 audit for v29B SemanticKITTI projection prerequisites.

    Checks:
        image_2 existence
        calib.txt existence
        P2 existence
        Tr / Tr_velo_to_cam existence
        velodyne directory existence
        labels directory existence
        matching velodyne/*.bin + labels/*.label for selected frames
        projection_frame_hit_rate
        no predicted fallback

    Writes:
        implementation_audit/codex_self_check_report.md
        implementation_audit/codex_self_check_summary.json
        implementation_audit/codex_self_check_failures.jsonl
        implementation_audit/projection_layout_scan.csv
        implementation_audit/projection_frame_hit_audit.csv
        implementation_audit/noop_parity_metrics.csv
        implementation_audit/masklet_3d_alignment_status.csv
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v29b_semantickitti_projection_audit.py

PASS
```

Audit command：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v29b_semantickitti_projection_audit.py \
    --repo-root . \
    --sequence-root /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01 \
    --sequence 01 \
    --results-root results/kitti01_hmc_v2/acl2_v29b_semantickitti3dprojection_videomasklet_semanticprior_allmemory \
    --chunks 6,10,16 \
    --horizons 10,15
```

Exit status：

```text
1
```

Reason：

```text
Phase 0 gate failed as intended because projection prerequisites are missing.
```

---

## 2. Phase 0：SemanticKITTI Projection Hard Gate

输出：

```text
implementation_audit/
```

Audit summary：

| Metric | Value |
|---|---:|
| `phase0_gate_pass` | `false` |
| `projection_available` | `false` |
| `projection_frame_hit_rate` | `0.0` |
| `projection_frames_expected` | `757` |
| `projection_frames_hit` | `0` |
| `sequence_has_image_2` | `true` |
| `sequence_has_calib` | `true` |
| `sequence_has_velodyne` | `false` |
| `sequence_has_labels` | `false` |
| `image_width` | `1241` |
| `image_height` | `376` |
| `no_predicted_fallback_flag` | `true` |
| `masklet_3d_alignment_allowed` | `false` |
| `phase2_rollout_allowed` | `false` |
| `selector_allowed` | `false` |
| `full_online_validation_allowed` | `false` |

Selected frame windows：

| Chunk | Horizon | Start frame | End exclusive | Frames expected | Frames hit | Hit rate |
|---:|---:|---:|---:|---:|---:|---:|
| `6` | `10` | `174` | `496` | `322` | `0` | `0.0` |
| `6` | `15` | `174` | `641` | `467` | `0` | `0.0` |
| `10` | `10` | `290` | `612` | `322` | `0` | `0.0` |
| `10` | `15` | `290` | `757` | `467` | `0` | `0.0` |
| `16` | `10` | `464` | `786` | `322` | `0` | `0.0` |
| `16` | `15` | `464` | `931` | `467` | `0` | `0.0` |

Local sequence root：

```text
/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01
```

Observed local files：

```text
image_2/ exists
image_3/ exists
calib.txt exists
times.txt exists
velodyne/ missing
labels/ missing
```

Calibration note：

```text
calib.txt contains P0/P1/P2/P3.
The audit did not find Tr / Tr_velo_to_cam in this sequence-local calib.txt.
Even if velodyne/labels existed, this calibration file would still need a
valid velodyne-to-camera transform or an external SemanticKITTI-compatible
calibration source before projection could be trusted.
```

Decision：

```text
Phase 0 projection hard gate = fail.

Required:
    projection_frame_hit_rate = 1.0
    matching velodyne/*.bin and labels/*.label
    calib with P2 and Tr/Tr_velo_to_cam
    no predicted fallback

Observed:
    projection_frame_hit_rate = 0.0
    velodyne/ missing
    labels/ missing
    Tr/Tr_velo_to_cam missing from local calib scan
    no predicted fallback used
```

---

## 3. Blocker 与修复记录

### Blocker 1：KITTI01 local sequence lacks SemanticKITTI projection prerequisites

现象：

```text
projection_layout_scan.csv:
    semantic_kitti:sequence_point_projection
    image_dir_exists = true
    velodyne_dir_exists = false
    point_label_dir_exists = false
    unique_frames_hit = 0
    hit_rate = 0.0
    first_missing_frame = 174
```

原因：

```text
本地 KITTI Odometry sequence 01 只有 image_2/image_3/calib/times，
没有 velodyne/*.bin 和 labels/*.label。
```

按计划尝试的修复方向：

```text
1. 检查 sequence-local layout:
       /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01

2. 检查是否存在 explicit SemanticKITTI root:
       V29B_SEMANTICKITTI_ROOT is empty / not configured.

3. 扫描本地候选 layout:
       sequence-local SemanticKITTI point projection
       parent-root SemanticKITTI sequences/01 point projection

4. 检查 image resolution:
       first image = 000000.png
       width = 1241
       height = 376

5. 检查 calibration:
       P2 exists
       Tr / Tr_velo_to_cam not found in local calib.txt
```

结果：

```text
No valid SemanticKITTI point projection layout was found.
No projection cache can be generated without fabricating velodyne/label/calib data.
```

### Blocker 2：不能用 predicted video-masklet semantic fallback

风险：

```text
已有 Stage-C video-masklet cache 可用，但 v29B 明确要求 projected 3D semantic
不能由 predicted semantic fallback 伪装。
```

处理：

```text
Audit summary records:
    no_predicted_fallback_flag = true

No masklet_3d_alignment was generated.
No video-masklet trust calibration was generated.
No semantic rollout was launched.
```

结果：

```text
The experiment stops at Phase 0 instead of producing invalid projected-3D claims.
```

---

## 4. Not Started

以下阶段未启动，原因均为 Phase 0 projection hard gate 失败：

```text
1. Projected semantic cache:
       L_3d_fine_pix.pt
       G_3d_coarse_pix.pt
       V_3d_pix.pt
       Z_3d_pix.pt
       L_3d_fine_tok.pt
       G_3d_coarse_tok.pt
       V_3d_tok.pt

2. Masklet-3D semantic alignment:
       masklet_3d_alignment.csv
       masklet_trust_summary.csv
       masklet_temporal_consistency.csv
       per_label_agreement_summary.csv
       video_vs_projected_confusion_matrix.csv

3. Action distinguishability gate.

4. Single-path memory candidates:
       FRAME_VIDEO_ONLY
       FRAME_3DPROJ_ONLY
       FRAME_VIDEO_3DTRUST
       GLOBAL_VIDEO_ONLY
       GLOBAL_3DPROJ_ONLY
       GLOBAL_VIDEO_3DTRUST
       SWA_VIDEO_ONLY
       SWA_3DPROJ_ONLY
       SWA_VIDEO_3DTRUST
       TTT_VIDEO_ONLY
       TTT_3DPROJ_ONLY
       TTT_VIDEO_3DTRUST

5. SWA projected-anchor diagnostic.

6. Masklet causal bank.

7. Durability / washout attribution.

8. No-GT selector and full online validation.
```

---

## 5. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Phase 0 projection prerequisite scan | fail | `projection_frame_hit_rate = 0.0`; no `velodyne/`; no `labels/` |
| Projection cache generation | not started | projection prerequisites missing |
| No-op parity smoke | not started | forbidden before projection cache gate |
| Masklet-3D alignment | not started | no projected 3D semantic cache |
| Trust-gated Semantic Prior | not started | no masklet-3D trust signal |
| Action distinguishability gate | not started | no 3D/video/trust action tensors |
| Phase 2 single-path rollout | not started | H0 projection gate failed |
| Masklet causal bank | not started | no masklet-3D alignment |
| No-GT selector | not started | gate failed |
| Full online validation | not started | gate failed |

Boundary：

```text
No v29B result counts as deployable online TTT write success.
No projected 3D semantic was available.
No predicted semantic fallback was used.
No full online validation was launched.
No online Target-25 result was produced in v29B.
```

---

## 6. Final Decision

v29B 的真实成功点：

```text
1. Added a strict v29B-specific SemanticKITTI projection audit.
2. Verified the local KITTI01 image_2/calib files exist.
3. Verified the required SemanticKITTI velodyne/labels projection data is absent.
4. Prevented invalid fallback from video-masklet predicted semantic to projected 3D semantic.
5. Produced audit artifacts documenting why Phase 0 cannot pass.
```

v29B 的关键负结果：

```text
The experiment cannot test the planned SemanticKITTI 3D projection route
with the current local data copy.

projection_frame_hit_rate = 0.0
projection_frames_hit = 0 / 757
sequence_has_velodyne = false
sequence_has_labels = false
Tr / Tr_velo_to_cam not found in local calib scan
```

Interpretation：

```text
This is not evidence that projected 3D semantic is weak.
It is a hard data availability failure.

The plan's core scientific questions remain unanswered:
    whether projected 3D semantic is stronger than video semantic,
    whether video masklet label quality is the semantic bottleneck,
    whether a masklet-level semantic causal oracle has an upper bound.

Answering those questions requires a local SemanticKITTI-compatible sequence 01
layout with matching:
    image_2/
    velodyne/
    labels/
    calib.txt containing P2 and Tr/Tr_velo_to_cam
```

Conclusion type：

```text
v29B cannot proceed on the current filesystem without valid SemanticKITTI
point-label projection inputs. The correct action is to stop at Phase 0,
not to replace missing projected 3D semantic with predicted video-masklet labels.
```

Next required direction：

```text
Provide or mount a SemanticKITTI/KITTI-Odometry-compatible sequence 01 root:
    sequences/01/image_2/*.png
    sequences/01/velodyne/*.bin
    sequences/01/labels/*.label
    sequences/01/calib.txt with P2 and Tr/Tr_velo_to_cam

Then rerun:
    tools/v29b_semantickitti_projection_audit.py

Only if projection_frame_hit_rate = 1.0 should v29B continue to:
    projection cache generation,
    masklet_3d_alignment,
    action distinguishability,
    single-path projected semantic memory candidates.
```
