# ACL2 v25 实验复盘：GT SemanticPrior AllMemory Parallel

日期：2026-05-22（Asia/Singapore）  
计划文件：`docs/ACL2_v25_GT_SemanticPrior_AllMemory_Parallel_Experiment_Plan.md`  
主结果目录：`results/kitti01_hmc_v2/acl2_v25_gt_semanticprior_allmemory_parallel/`

本轮原则：只记录实际落盘结果；不把 GT semantic 缺失时的 predicted Stage-C / Mask2Former fallback 写成 GT semantic 实验；不把 implementation audit、blocked smoke、missing-data gate、或未启动矩阵写成 deployable online success。没有通过 GT semantic Phase 0 hard gate 时，不启动 passive GT attribution、single-path candidate、pairwise/all-memory、selector 或 full online validation。

---

## 0. 当前结论

v25 已按计划执行到 Phase 0 GT semantic implementation hard gate，并在该 hard gate 处停止。

停止原因：

```text
本地 KITTI Odometry sequence 01 副本没有可覆盖目标 frame 的 GT semantic。

已支持的 GT 输入形式：
    KITTI semantic benchmark dense PNG
    KITTI-STEP panoptic PNG
    KITTI-360 dense 2D semantic PNG
    SemanticKITTI point label + velodyne 投影到 image_2

实际本地命中：
    best_dense_frames_hit = 0
    best_projection_frames_hit = 0
    GT semantic cache hit rate = 0.0
    Phase 0 gate = fail
```

最终边界：

```text
1. No GT semantic candidate rollout was launched.
2. No predicted semantic fallback was used.
3. No passive GT semantic attribution was launched.
4. No Phase 2 single-path GT semantic matrix was launched.
5. No pairwise/all-memory GT semantic validation was launched.
6. No no-GT selector was started.
7. No full online validation was launched.
8. No online Target-25 result was produced.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

重要解释：

```text
v25 当前不是“GT semantic 弱”的科学结论。
当前只能得出工程/数据结论：
    GT semantic data unavailable for local KITTI01 frames,
    so v25 semantic upper-bound experiment cannot legally start.

不能用 v24 predicted Stage-C labels 替代 GT semantic。
不能把 predicted/coarse semantic 结果重命名为 GT semantic result。
```

---

## 1. 外部 repo / 数据格式自查

用户要求重点看 KITTI 2D semantic repo 的做法。已查到的关键事实：

```text
KITTI semantic segmentation benchmark:
    官方 benchmark 是 200 train + 200 test image-level semantic labels。
    数据格式与 Cityscapes conform。
    这不是 KITTI Odometry sequence 01 自带目录。

PointPainting / BiSeNetv2 KITTI training repo:
    使用 KITTI semantic dataset。
    目录结构：
        BiSeNetv2/data/KITTI/training/image_2
        BiSeNetv2/data/KITTI/training/semantic
        BiSeNetv2/data/KITTI/training/semantic_rgb

KITTI-STEP / DeepLab2:
    使用 video-level pixel annotation。
    panoptic map 目录结构：
        panoptic_maps/{train,val}/{sequence_id}/{frame_id}.png
    PNG 编码：
        R = semantic_id
        G = instance_id // 256
        B = instance_id % 256

KITTI-360:
    2D semantic 目录结构：
        data_2d_semantics/train/.../image_{00|01}/semantic/{frame:010d}.png
        data_2d_semantics/train/.../image_{00|01}/semantic_rgb/{frame:010d}.png
    这是 KITTI-family dense 2D semantic，但不是当前 KITTI Odometry sequence 01。
```

参考链接：

```text
KITTI semantic benchmark:
    https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics

PointPainting repo README:
    https://github.com/AmrElsersy/PointPainting

KITTI-STEP / DeepLab2 setup:
    https://huggingface.co/spaces/akhaliq/deeplab2/blob/main/g3doc/setup/kitti_step.md

KITTI-360 documentation:
    https://www.cvlibs.net/datasets/kitti-360/documentation.php
```

结论：

```text
这些 repo 的“KITTI 2D semantic”一般不是从本地 odometry sequence/01 自动长出来的。
它们依赖单独下载的 KITTI semantic / KITTI-STEP / KITTI-360 标注包。

因此 v25 需要支持这些实际布局，但如果本地没有对应标注包，
不能伪造为 GT semantic cache hit。
```

---

## 2. 工程修改

新增：

```text
loger/pipeline/gt_semantic_provider.py:
    GT semantic provider / layout discovery / frame loader
    never falls back to predicted Stage-C / Mask2Former cache

    supported dense 2D layouts:
        sequence-local semantic/*.png
        sequence-local semantics/*.png
        sequence-local semantic_labels/*.png
        sequence-local image_2_semantic/*.png
        KITTI semantic benchmark training/semantic/*.png
        KITTI semantic benchmark training/semantic_rgb/*.png
        KITTI semantic benchmark Stereo2015-style *_10.png
        KITTI-STEP panoptic_maps/{train,val}/{seq}/{frame}.png
        KITTI-360 data_2d_semantics/.../semantic/*.png

    supported projection layout:
        SemanticKITTI / KITTI odometry point labels:
            sequences/01/velodyne/{frame:06d}.bin
            sequences/01/labels/{frame:06d}.label
            sequences/01/calib.txt with P2 and Tr / Tr_velo_to_cam

    projection behavior:
        reads velodyne float32 XYZI points
        reads SemanticKITTI uint32 labels
        uses lower 16 bits as semantic id
        projects with P2 * R0_rect * Tr_velo_to_cam
        z-buffer keeps nearest projected point per pixel
        outputs image-aligned semantic map with ignore label for uncovered pixels

tools/v25_gt_semantic_audit.py:
    strict Phase 0 GT semantic cache / layout audit
    now uses loger.pipeline.gt_semantic_provider
    scans practical KITTI 2D semantic training repo layouts
    scans KITTI-STEP / KITTI-360 layouts
    scans SemanticKITTI point-label projection prerequisites
    records dense hit count and projection hit count separately
    writes required Phase 0 audit artifacts:
        codex_self_check_report.md
        codex_self_check_summary.json
        codex_self_check_failures.jsonl
        gt_semantic_layout_scan.csv
        gt_semantic_cache_audit.csv
        semantic_role_alignment_audit.csv
        path_consumption_audit.csv
        noop_parity_metrics.csv
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    loger/pipeline/gt_semantic_provider.py \
    tools/v25_gt_semantic_audit.py

PASS
```

---

## 3. Phase 0：GT Semantic Implementation Hard Gate

输出：

```text
results/kitti01_hmc_v2/acl2_v25_gt_semanticprior_allmemory_parallel/
    implementation_audit/codex_self_check_report.md
    implementation_audit/codex_self_check_summary.json
    implementation_audit/codex_self_check_failures.jsonl
    implementation_audit/gt_semantic_layout_scan.csv
    implementation_audit/gt_semantic_cache_audit.csv
    implementation_audit/semantic_role_alignment_audit.csv
    implementation_audit/path_consumption_audit.csv
    implementation_audit/noop_parity_metrics.csv
```

命令：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python \
    tools/v25_gt_semantic_audit.py \
    --repo-root . \
    --sequence-root /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01 \
    --sequence 01 \
    --results-root results/kitti01_hmc_v2/acl2_v25_gt_semanticprior_allmemory_parallel
```

Audit exit status：

```text
AUDIT_EXIT_STATUS = 1
```

该非零状态是 expected hard-gate fail，不是可忽略错误。

Gate summary：

| Metric | Value |
|---|---:|
| `phase0_gate_pass` | `false` |
| `gt_semantic_available` | `false` |
| `gt_cache_hit_rate` | `0.0` |
| `gt_frames_expected` | `757` |
| `gt_frames_hit` | `0` |
| `best_dense_frames_hit` | `0` |
| `best_projection_frames_hit` | `0` |
| `no_predicted_fallback_flag` | `true` |
| `selector_allowed` | `false` |
| `full_online_validation_allowed` | `false` |
| `counts_as_deployable_online_success` | `false` |

Selected audit scope：

```text
sequence = 01
chunks = 6, 10, 16
horizons = 10, 15
first RGB image = /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2/000000.png
image resolution = 1241 x 376
unique selected frames = 757
```

Local data actually present：

```text
/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/
    calib.txt
    image_2/
    image_3/
    times.txt
```

Local data missing：

```text
dense 2D semantic:
    semantic/
    semantics/
    semantic_labels/
    image_2_semantic/
    training/semantic/
    training/semantic_rgb/
    data_semantics/training/semantic/
    data_semantics/training/semantic_rgb/
    panoptic_maps/

projection prerequisites:
    velodyne/
    labels/
    calib Tr / Tr_velo_to_cam
```

Calib status:

```text
calib.txt contains P0/P1/P2/P3.
calib_has_p2 = true
calib_has_tr_velo = false

Therefore even if point labels appeared alone, current local calib is insufficient
for audited Velodyne-to-image projection unless Tr / Tr_velo_to_cam is supplied
from the matching KITTI odometry calibration package.
```

Failure record：

```text
gate = phase0_gt_cache
failure = gt_semantic_cache_missing
detail:
    No GT semantic layout has labels for all selected KITTI01 frames.
    Predicted Stage-C semantic fallback is forbidden by the v25 plan.
attempted_fix:
    Implemented GT provider support for KITTI semantic benchmark dense PNG layouts,
    KITTI-STEP panoptic PNG layouts, KITTI-360 dense 2D semantic layouts,
    and SemanticKITTI point-label projection from velodyne/*.bin + labels/*.label.
    The local KITTI01 odometry copy still has only calib.txt, image_2, image_3,
    and times.txt; no dense 2D labels, velodyne scans, or point labels were found.
```

Decision：

```text
Phase 0 gate = fail.

Per v25 plan:
    if GT semantic cache missing:
        fix GTSemanticProvider / frame-id mapping;
        do not fallback predicted;
        do not run candidate.

What was attempted:
    searched external KITTI 2D semantic training repo layouts;
    implemented dense PNG GT provider;
    implemented KITTI-STEP panoptic PNG reader;
    implemented SemanticKITTI point-label projection provider;
    reran Phase 0 audit;
    confirmed no supported local GT layout covers KITTI01 selected frames.

Therefore:
    Phase 1 passive GT attribution is not allowed.
    Phase 2 GT single-path screening is not allowed.
    Pairwise/all-memory GT semantic experiments are not allowed.
    Selector/full online validation is not allowed.
```

---

## 4. Blocker 与修复方向

### Blocker 1：本地 KITTI01 没有 GT semantic 标注包

现象：

```text
GT_cache_hit_rate = 0.0
gt_frames_hit = 0
gt_frames_expected = 757
best_dense_frames_hit = 0
best_projection_frames_hit = 0
phase0_gate_pass = false
```

原因：

```text
当前本地数据目录只有 KITTI Odometry RGB / calib / times。
未发现 2D dense semantic 标签包。
未发现 SemanticKITTI velodyne + labels 点云标签组合。
```

已完成修复尝试：

```text
1. 不再只扫 sequence-local dense PNG。
2. 增加 KITTI semantic benchmark 常见 repo 布局。
3. 增加 KITTI-STEP panoptic map 布局。
4. 增加 KITTI-360 dense 2D semantic 布局。
5. 增加 SemanticKITTI point label -> image_2 投影实现。
6. 重跑 Phase 0 audit。
```

结果：

```text
仍无可用 GT semantic。
该 blocker 不能通过 predicted fallback 绕过。
也不能把 KITTI semantic benchmark 的 200 张 stereo2015 标签强行映射到 KITTI Odometry sequence 01。
```

后续如果提供 GT semantic：

```text
可设置：
    V25_GT_SEMANTIC_ROOT=/path/to/gt_semantic_root

或使用：
    tools/v25_gt_semantic_audit.py --gt-root /path/to/gt_semantic_root

可接受形式之一：
    1. dense image-aligned semantic PNG，frame id 与 KITTI01 image_2 对齐；
    2. KITTI-STEP panoptic PNG，并且 sequence/frame 对齐；
    3. SemanticKITTI point labels + matching velodyne scans + matching calib Tr / Tr_velo_to_cam。

只有 audit 达到：
    GT_cache_hit_rate = 1.0
    no_predicted_fallback_flag = true
才允许启动 GT no-op parity smoke 和后续候选。
```

---

## 5. Downstream Phase Decision

| Stage | Status | Reason |
|---|---|---|
| Phase 0 GT cache audit | fail | no supported GT semantic layout covers KITTI01 selected frames |
| GT no-op parity smoke | not started | forbidden until GT cache hit rate = 1.0 |
| Runtime GT role projection | implemented provider, not run | no GT labels to load |
| Passive GT attribution | not started | Phase 0 hard gate failed |
| Phase 2 GT single-path | not started | Phase 0 hard gate failed |
| Phase 3 pairwise | not started | no Phase 2 candidate |
| Phase 4 all-memory | not started | no Phase 3 candidate |
| Phase 5 durability attribution | not started | no h10 strong / h15 weak GT candidate |
| Phase 6 cross-sequence | not started | no stable GT candidate |
| No-GT selector | not started | forbidden |
| Full online validation | not started | forbidden |

Boundary：

```text
No v25 result counts as GT semantic diagnostic success.
No v25 result counts as deployable online TTT write success.
No predicted semantic fallback was used.
No GT candidate metric exists.
No online Target-25 result was produced in v25.
```

---

## 6. Final Decision

v25 的真实成功点：

```text
1. The v25 plan was parsed and the GT-only boundary was enforced.
2. KITTI 2D semantic repo/data layouts were checked before coding assumptions.
3. A strict Phase 0 GT semantic audit script was implemented and rerun.
4. A GT semantic provider now supports dense 2D PNG, KITTI-STEP panoptic PNG,
   KITTI-360 dense semantic PNG, and SemanticKITTI point-label projection.
5. The script writes the required audit artifacts.
6. The script verifies no predicted semantic fallback is used.
7. The local KITTI01 data blocker was detected before launching any invalid candidate rollout.
```

v25 的关键负结果：

```text
No supported GT semantic labels are available for local KITTI01 selected frames.
GT cache hit rate = 0.0.
Phase 0 gate failed.
```

Interpretation：

```text
v25 still cannot answer whether GT semantic is stronger than predicted semantic.
The missing-data hard gate prevents any valid upper-bound experiment.

This is the correct stop behavior:
    using predicted Stage-C labels would violate the v25 plan;
    treating unrelated KITTI semantic benchmark frames as KITTI01 labels would be invalid;
    treating SemanticKITTI point labels as image labels without matching velodyne/calib would be invalid.
```

Next required direction：

```text
Provide GT semantic aligned to KITTI01 image_2 frame ids,
or provide SemanticKITTI point labels plus matching velodyne scans and calibration.

Only after:
    GT_cache_hit_rate = 1.0
    no_predicted_fallback_flag = true
    GT no-op parity exact
    runtime GT fine-label projection available
may v25 Phase 1 / Phase 2 start.
```
