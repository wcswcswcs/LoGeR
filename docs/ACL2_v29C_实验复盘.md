# ACL2 v29C 实验复盘：SemanticKITTI Download + Projection + VideoMasklet

日期：2026-05-23（Asia/Singapore）  
计划文件：`docs/ACL2_v29C_SemanticKITTI_Download_Projection_VideoMasklet_Plan.md`  
数据根目录：`/mnt/data/users/chengshun.wang/data/semantickitti_odometry/`  
主结果目录：`results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet/`

本轮原则：只记录实际落盘结果；不把下载 gate、HTML/license 页面、projection audit、offline distinguishability、blocked gate 或未启动矩阵写成 deployable online success。Projected 3D semantic 只来自 official KITTI/SemanticKITTI point-label projection；不使用 video-masklet predicted semantic fallback 冒充 3D semantic。

---

## 0. 当前结论

v29C 没有停在 v29B 的数据缺失状态：已继续补齐 official KITTI/SemanticKITTI sequence 01 投影数据，并完成 sparse projection cache、masklet-3D alignment、offline action distinguishability，以及一个最小 runtime masklet-level causal-bank h10 screen。

但 v29C 尚未达到 Target-25，也尚未产生 deployable online result。Phase 5 h10 causal oracle 未过 gate。

已完成并落盘：

```text
1. 创建独立 SemanticKITTI/KITTI Odometry 数据根目录：
       /mnt/data/users/chengshun.wang/data/semantickitti_odometry/

2. 新增 strict download/data gate audit：
       tools/v29c_semantickitti_download_and_projection_audit.py

3. 首次 cvlibs calib URL 返回 HTML email/license 页面：
       data_odometry_calib.zip = 1089 bytes, HTML document
   按计划写入：
       /mnt/data/users/chengshun.wang/data/semantickitti_odometry/MANUAL_DOWNLOAD_REQUEST.md

4. 继续补救，探测 KITTI/AVG official S3 object endpoint：
       https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
       https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
   HEAD returned:
       HTTP 200 OK
       Content-Type: application/zip
       Accept-Ranges: bytes

5. 新增 parallel range downloader：
       tools/v29c_parallel_range_download.py
   用于同一 official S3 object 的 resumable HTTP Range 下载。

6. 下载并验证 official files：
       data_odometry_calib.zip      = 600238 bytes, Zip archive data
       data_odometry_labels.zip     = 179298418 bytes, Zip archive data
       data_odometry_velodyne.zip   = 84786535790 bytes, Zip archive data
   sequence 01 zip tests:
       calib:    1 file tested, no errors
       labels:   1102 files tested, no errors
       velodyne: 1102 files tested, no errors

7. 解压 sequence 01 only：
       dataset/sequences/01/calib.txt
       dataset/sequences/01/labels/*.label
       dataset/sequences/01/velodyne/*.bin
       dataset/sequences/01/image_2 -> symlink to local KITTI01 image_2

8. Phase 0 data gate 通过：
       downloads_gate_pass = true
       phase0_data_gate_pass = true
       projection_frame_hit_rate = 1.0
       projection_frames_hit = 757 / 757
       calib_parse_pass = true
       point_label_count_gate_pass = true

9. 新增 sparse projection cache generator：
       tools/v29c_semantickitti_projection_cache.py
   Phase 1 projection cache gate 通过：
       frames_projected = 757 / 757
       median_unique_projected_pixels = 15978
       mean_projected_pixel_coverage = 0.0340871176
       focus_200_300_projected_coverage = 0.0323367180

10. 新增 masklet-3D alignment：
       tools/v29c_masklet_3d_alignment.py
   Masklet trust/alignment gate 通过：
       chunks_aligned = 28
       masklets_total = 182
       supported_supportable_ratio = 0.9610389610
       supported_structure_ground_agreement_ratio = 0.8352941176

11. 新增 offline action distinguishability：
       tools/v29c_action_distinguishability.py
   Offline action distinguishability gate 通过：
       action_distinguishability_gate_pass = true

12. 继续按计划推进 runtime masklet-level causal bank：
       implemented v29c_masklet_override role hook
       h10 rows completed = 6/6
       repair row completed = 1/1
       oracle_gate_pass = false

13. 按后续审计要求补做 projected 2D semantic visual audit：
       generated montage/overlay/depth/valid-mask images for 10 frames
       manually inspected representative frames = 5
       visual_alignment_gate = pass for inspected frames
```

当前边界：

```text
1. Sparse projected 3D semantic cache exists and passed projection cache gate.
2. Masklet-3D alignment exists and passed trust/alignment gate.
3. Projected 2D semantic visual audit exists and passes for inspected frames.
4. Offline action distinguishability exists and passed.
5. A small h10 masklet-level causal-bank rollout was launched and failed oracle gate.
6. No learned quality-aware runtime role router was launched.
7. No h15 semantic candidate rollout was launched from v29C.
8. No no-GT selector was launched.
9. No full online validation was launched.
10. No online Target-25 result was produced in v29C.
```

当前 best deployable online TTT write 仍然是：

```text
C9_P0_R2
ATE = 33.7629421029m
```

解释：

```text
v29C 已经解决 v29B 的本地 projection data 缺失问题。

Projected semantic source:
    KITTI Odometry velodyne + calib
    SemanticKITTI point labels
    LiDAR-to-image sparse projection

uses_gt_projected_3d_semantic = true for sparse anchors
uses_video_masklet_semantic_as_gt = false

目前通过的是 data/projection/visual/alignment/offline action gate。
Phase 5 h10 trajectory causal bank 已补跑，但没有通过 oracle gate，
不是 deployable online validation。
```

---

## 1. 工程修改

新增：

```text
tools/v29c_semantickitti_download_and_projection_audit.py:
    strict download / projection data gate audit.
    Writes:
        implementation_audit/download_audit_summary.json
        implementation_audit/download_audit_report.md
        implementation_audit/download_file_type_audit.csv
        implementation_audit/sequence01_file_counts.csv
        implementation_audit/calib_parse_report.json
        implementation_audit/frame_hit_audit.csv
        implementation_audit/frame_window_hit_audit.csv
        implementation_audit/label_velodyne_point_count_audit.csv

tools/v29c_parallel_range_download.py:
    resumable HTTP Range downloader for large official KITTI zip objects.
    Used because aria2c was unavailable and single-connection wget was slow.

tools/v29c_semantickitti_projection_cache.py:
    generates sparse projected semantic cache:
        *_sem_sparse.npy
        *_inst_sparse.npy
        *_depth_sparse.npy
        *_valid_mask.npy
        *_meta.json
    Pixel convention:
        unprojected semantic = 255
        valid_mask = 0

tools/v29c_projection_visual_audit.py:
    generates human-inspectable projected 2D semantic visual audit images:
        RGB
        projected semantic overlay
        projected semantic points only
        projected depth overlay
        valid mask
        per-frame montage with label legend
    Writes:
        projection_visual_audit/projection_visual_audit_summary.csv
        projection_visual_audit/projection_visual_audit_summary.json
        projection_visual_audit/frame_*_montage.png
        projection_visual_audit/frame_*_semantic_overlay.png

tools/v29c_masklet_3d_alignment.py:
    aligns VideoMasklet masklets with sparse projected SemanticKITTI anchors.
    Writes:
        masklet_alignment.csv
        masklet_alignment.jsonl
        per_chunk_alignment_summary.csv
        per_label_agreement_summary.csv
        masklet_trust_summary.json

tools/v29c_action_distinguishability.py:
    offline masklet-level action distinguishability preview.
    Writes:
        phase4_action_distinguishability/action_distinguishability.csv
        phase4_action_distinguishability/action_distinguishability_summary.json

run_pipeline_abc_v2.py:
    added v29C runtime masklet intervention hook:
        --v29c_masklet_alignment_csv
        --v29c_masklet_intervention_policy
        --v29c_masklet_intervention_path
        --v29c_masklet_intervention_action
    Added semantic_role_policy = v29c_masklet_override.
    The hook selects audited masklets from masklet_alignment.csv and overwrites
    path role tensors only; it does not modify projected 3D labels or predicted
    video-masklet semantic labels.

loger/pipeline/hybrid_memory_controller.py:
    v29c_masklet_override consumes incoming path-specific role tensors instead
    of recomputing fine/causal roles from label thresholds.

tools/run_attention_cue_experiment.sh:
    forwards V29C_MASKLET_* environment variables to run_pipeline_abc_v2.py.

tools/run_v24_candidate_rollout.sh:
    added v29C causal-bank aliases:
        V29C_BASE_H9_REFERENCE
        V29C_CAUSAL_FRAME_SKIP_TOP
        V29C_CAUSAL_GLOBAL_SKIP_TOP
        V29C_CAUSAL_SWA_ANCHOR_TOP
        V29C_CAUSAL_SWA_REMOVE_TOP
        V29C_CAUSAL_TTT_POS_TOP
        V29C_CAUSAL_TTT_NEG_TOP

tools/v29c_masklet_causal_bank_report.py:
    aggregates landed h10 causal-bank trajectory deltas and runtime intervention
    metadata into:
        masklet_causal_bank/causal_effects.csv
        masklet_causal_bank/causal_effects_by_label.csv
        masklet_causal_bank/causal_effects_by_path.csv
        masklet_causal_bank/top_positive_masklets.csv
        masklet_causal_bank/top_negative_masklets.csv
        masklet_causal_bank/causal_bank_summary.json
```

验证：

```text
/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python -m py_compile \
    tools/v29c_semantickitti_download_and_projection_audit.py \
    tools/v29c_parallel_range_download.py \
    tools/v29c_semantickitti_projection_cache.py \
    tools/v29c_projection_visual_audit.py \
    tools/v29c_masklet_3d_alignment.py \
    tools/v29c_action_distinguishability.py \
    tools/v29c_masklet_causal_bank_report.py \
    run_pipeline_abc_v2.py \
    loger/pipeline/hybrid_memory_controller.py \
    tools/v24_candidate_bank_report.py

PASS

bash -n \
    tools/run_v24_candidate_rollout.sh \
    tools/run_attention_cue_experiment.sh

PASS
```

---

## 2. Phase 0A：Download / Data Gate

Disk gate：

| Metric | Value |
|---|---:|
| initial `/mnt/data` available | `242G` |
| after downloads/extract/cache available | `161G` |
| required before download | `>= 180G` |
| disk gate before download | `pass` |

Downloader tools：

| Tool | Status |
|---|---|
| `aria2c` | not found |
| `wget` | `/usr/bin/wget` |
| `unzip` | `/usr/bin/unzip` |

Download attempts：

| File | Source | Result | File type | Size |
|---|---|---|---|---:|
| `data_odometry_calib.zip` | cvlibs form URL | blocked | `HTML document, ASCII text` | `1089` |
| `data_odometry_calib.zip` | KITTI/AVG S3 | success | `Zip archive data` | `600238` |
| `data_odometry_labels.zip` | SemanticKITTI | success | `Zip archive data` | `179298418` |
| `data_odometry_velodyne.zip` | KITTI/AVG S3 | success | `Zip archive data` | `84786535790` |

Zip integrity tests：

| Zip | Tested sequence01 files | Result |
|---|---:|---|
| `data_odometry_calib.zip` | `1` | no errors |
| `data_odometry_labels.zip` | `1102` | no errors |
| `data_odometry_velodyne.zip` | `1102` | no errors |

Sequence01 layout after extraction：

```text
/mnt/data/users/chengshun.wang/data/semantickitti_odometry/dataset/sequences/01/
    image_2 -> /mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences/01/image_2
    calib.txt
    labels/*.label
    velodyne/*.bin
```

Phase 0 audit summary：

| Metric | Value |
|---|---:|
| `downloads_gate_pass` | `true` |
| `phase0_data_gate_pass` | `true` |
| `selected_unique_frames_expected` | `757` |
| `projection_frame_hit_rate` | `1.0` |
| `projection_frames_hit` | `757` |
| `velodyne_hit_rate` | `1.0` |
| `labels_hit_rate` | `1.0` |
| `image_hit_rate` | `1.0` |
| `calib_parse_pass` | `true` |
| `calib_has_p2` | `true` |
| `calib_has_tr_velo` | `true` |
| `tr_key` | `Tr` |
| `R0_rect` | missing, identity logged explicitly |
| `point_label_count_gate_pass` | `true` |
| `point_label_frames_checked` | `757` |
| `point_label_count_mismatch_rate` | `0.0` |

Decision：

```text
Phase 0 data gate = pass.
Projection cache generation is allowed.
No trajectory candidate is allowed yet.
```

---

## 3. Phase 1：Sparse Projection Cache

输出：

```text
projection_cache/seq01/
phase1_projection_cache/
```

Projection cache summary：

| Metric | Value |
|---|---:|
| `projection_cache_gate_pass` | `true` |
| `frames_expected` | `757` |
| `frames_projected` | `757` |
| `frame_hit_rate` | `1.0` |
| `median_unique_projected_pixels` | `15978` |
| `mean_unique_projected_pixels` | `15905.5944517834` |
| `mean_projected_pixel_coverage` | `0.0340871176` |
| `p10_projected_pixel_coverage` | `0.0294460541` |
| `focus_200_300_projected_coverage` | `0.0323367180` |
| `calib_tr_key` | `Tr` |
| `r0_rect_missing_identity_used` | `true` |

Example frame metrics：

| Frame | Points | Unique projected pixels | Coverage | Top labels |
|---:|---:|---:|---:|---|
| `174` | `124415` | `19543` | `0.0418824044` | road / vegetation / terrain / fence |
| `930` | `111359` | `16410` | `0.0351681040` | road / fence / terrain / vegetation |

Decision：

```text
Phase 1 projection cache gate = pass.
Masklet-3D alignment is allowed.
```

---

## 3A. Projected 2D Semantic Visual Audit

触发原因：

```text
用户明确要求不能只靠 numerical gate：
    必须通过可视化证明 projected 2D semantic 是对的，
    并由 Codex 自己打开图像检查。
```

输出：

```text
projection_visual_audit/
    projection_visual_audit_summary.csv
    projection_visual_audit_summary.json
    visual_inspection_notes.md
    frame_000174_montage.png
    frame_000220_montage.png
    frame_000290_montage.png
    frame_000350_montage.png
    frame_000464_montage.png
    frame_000550_montage.png
    frame_000650_montage.png
    frame_000757_montage.png
    frame_000850_montage.png
    frame_000925_montage.png
    and per-frame RGB / semantic_overlay / semantic_points / depth_overlay / valid_mask PNGs
```

设置：

```text
frames generated = 10/10
frames generated:
    174, 220, 290, 350, 464, 550, 650, 757, 850, 925

frames manually inspected by Codex in image viewer:
    174, 290, 464, 650, 925
```

Visual summary：

| Frame | Valid projected pixels | Projected coverage | Top labels |
|---:|---:|---:|---|
| `174` | `19543` | `0.0418824044` | `road / vegetation / terrain / fence` |
| `290` | `13607` | `0.0291610232` | `road / fence / terrain / vegetation` |
| `464` | `14504` | `0.0310833748` | `road / fence / moving_car / terrain` |
| `650` | `18664` | `0.0399986284` | `road / vegetation / fence` |
| `925` | `17013` | `0.0364603871` | `road / terrain / vegetation / fence` |

Manual inspection notes：

```text
1. Road labels are concentrated on the visible drivable road surface in the
   lower image region.
2. Vegetation / terrain / fence labels appear on roadside vegetation, shoulders,
   and guardrail-side regions.
3. In frame 000464, moving_car labels align with the visible left-lane car body.
4. Projected depth overlays show plausible near-to-far image gradients.
5. No global left/right flip, vertical flip, major translation, or obviously
   wrong camera projection was observed in the inspected frames.
6. Sky is intentionally mostly unlabeled because this is sparse LiDAR projection;
   no projection does not mean sky.
```

Decision：

```text
Projected 2D semantic visual audit = pass for inspected representative frames.

This supports using the projected sparse 2D semantic anchors as diagnostic GT
projection evidence in v29C.

Boundary:
    This is still sparse projected GT, not dense 2D semantic GT.
    It does not exhaustively prove every frame by human inspection.
```

---

## 4. Phase 2：VideoMasklet vs Projected 3D Alignment

输出：

```text
masklet_3d_alignment/
```

设置：

```text
stage_c_cache_dir = results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full
projection_cache_dir = results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet/projection_cache/seq01
selected frames = 757
N_min = 50 projected pixels
```

Masklet alignment summary：

| Metric | Value |
|---|---:|
| `masklet_3d_alignment_gate_pass` | `true` |
| `chunks_aligned` | `28` |
| `masklets_total` | `182` |
| `supportable_non_sky_non_unknown_masklets` | `154` |
| `supported_supportable_masklets` | `148` |
| `supported_supportable_ratio` | `0.9610389610` |
| `supported_structure_ground_masklets` | `85` |
| `supported_structure_ground_agreement_ratio` | `0.8352941176` |

Per-label summary：

| Fine label | Masklets | Supported | Mean support | Known agreement ratio |
|---|---:|---:|---:|---:|
| `road` | `28` | `28` | `306801.2143` | `1.0` |
| `vegetation` | `28` | `28` | `89451.25` | `0.7037037037` |
| `grass` | `27` | `27` | `44827.5556` | `1.0` |
| `fence` | `25` | `25` | `21207.96` | `0.84` |
| `wall` | `18` | `18` | `6519.1111` | `0.7222222222` |
| `building` | `22` | `16` | `2285.1818` | `0.625` |
| `sidewalk` | `6` | `6` | `3071.8333` | `0.6666666667` |
| `sky` | `28` | `17` | `204.7143` | `0.0` |

Boundary note：

```text
Sky agreement ratio is not used as a hard negative,
because LiDAR sparse projection cannot validate sky reliably.
```

Decision：

```text
Masklet-3D alignment gate = pass.
Action distinguishability diagnostic is allowed.
```

---

## 5. Phase 4：Offline Action Distinguishability

输出：

```text
phase4_action_distinguishability/
```

Important boundary：

```text
This is an offline masklet-level action preview.
No trajectory rollout is claimed.
No ATE / Target-25 result is claimed.
```

Action distinguishability summary：

| Metric | Value |
|---|---:|
| `action_distinguishability_gate_pass` | `true` |
| `masklets` | `182` |

Selected comparisons：

| Pair | Action | Jaccard | Ratio diff | Gate |
|---|---|---:|---:|---|
| `SEM_ONLY` vs `QUALITY_AWARE_SEM` | `swa_anchor_keep` | `0.5555555556` | `0.2417582418` | pass |
| `SEM_ONLY` vs `QUALITY_AWARE_SEM` | `ttt_positive` | `0.5555555556` | `0.2417582418` | pass |
| `SEM_ONLY` vs `QUALITY_AWARE_SEM` | `ttt_no_write` | `0.0` | `0.2692307692` | pass |
| `QUALITY_AWARE_SEM` vs `RISK_ONLY` | `source_skip` | `0.2439024390` | `0.3406593407` | pass |
| `RISK_ONLY` vs `QUALITY_AWARE_SEM_RISK` | `ttt_negative` | `0.0` | `0.1208791209` | pass |

Decision：

```text
Offline action distinguishability gate = pass.

This only proves the projected-3D trust signal can produce different intended
masklet-level actions. It does not prove runtime hook intervention or trajectory
improvement.
```

---

## 6. Blocker 与修复记录

### Blocker 1：cvlibs direct URL returned manual email/license HTML

现象：

```text
data_odometry_calib.zip from cvlibs:
    file type = HTML document, ASCII text
    size = 1089 bytes
```

处理：

```text
Wrote MANUAL_DOWNLOAD_REQUEST.md.
Then checked KITTI/AVG official S3 object endpoint with HEAD only.
Because it returned HTTP 200 and application/zip for the required official files,
continued download from that official storage endpoint.
```

### Blocker 2：single-connection velodyne download too slow

现象：

```text
wget single connection estimated roughly 90+ minutes and had only downloaded ~4.1G.
```

修复：

```text
Added tools/v29c_parallel_range_download.py.
Used HTTP Range requests against the same KITTI/AVG S3 object.
Downloaded:
    bytes = 84786535790
    ranges = 158
    workers = 8
    logical_mib_per_second ≈ 49.351
```

Result：

```text
data_odometry_velodyne.zip = Zip archive data
sequence 01 unzip test passed for 1102 files.
```

### Blocker 3：v29C originally lacked projection cache and alignment tools

处理：

```text
Added:
    tools/v29c_semantickitti_projection_cache.py
    tools/v29c_masklet_3d_alignment.py
    tools/v29c_action_distinguishability.py
```

Result：

```text
Projection cache gate passed.
Masklet-3D alignment gate passed.
Offline action distinguishability gate passed.
```

### Blocker 4：offline action preview 不能替代 trajectory causal bank

现象：

```text
phase4_action_distinguishability 只证明不同 policy 的 intended masklet action
集合不同，但没有 ATE / trajectory effect。
这不能回答 v29C Phase 5 oracle gate。
```

修复：

```text
Added runtime masklet-level intervention hook:
    semantic_role_policy = v29c_masklet_override

The hook:
    reads masklet_3d_alignment/masklet_alignment.csv
    selects top projected-support masklet per runtime chunk
    projects that masklet to patch/token roles
    overwrites only the requested path role stream:
        frame/global/swa/ttt
    records v29c_masklet_intervention in hmc_state_hash.jsonl
```

Result：

```text
Phase 5 h10 causal-bank rows were launched and completed.
Trajectory deltas are now available for the tested top-support road masklet
interventions.
```

### Blocker 5：SWA anchor/remove actions collapsed to identical trajectory

现象：

```text
V29C_CAUSAL_SWA_ANCHOR_TOP and V29C_CAUSAL_SWA_REMOVE_TOP produced identical
h10 trajectory deltas:
    ATE delta = -0.3641983213m
    [200,300) delta = -2.3655037795m
```

Audit：

```text
The runtime role streams were not identical:
    SWA anchor uses role = 2
    SWA remove uses role = 3
    v29c_masklet_intervention.enabled = true

But the hook_effect_summary stayed effectively identical:
    implemented_paths = ['ttt_update', 'swa_read']
    mean_swa_gate ~= 0.9825
    source gate applied = 1 active SWA layer
```

Repair attempt：

```text
Reran V29C_CAUSAL_SWA_REMOVE_TOP with:
    SEMANTIC_ROLE_SWA_NEGATIVE_SCALE = 1.0
    run_prefix = V29C_P5_CAUSAL_H10_REPAIR1
```

Repair result：

```text
The hard-remove repair row remained identical:
    ATE delta = -0.3641983213m
    [200,300) delta = -2.3655037795m

Interpretation:
    the current SWA read hook is dominated by shared SWA source/gate behavior,
    not by the anchor/remove role identity. This must not be counted as a
    differentiated SWA masklet causal action.
```

---

## 7. Phase 5：Masklet-Level Causal Bank h10

输出：

```text
phase5_causal_bank_h10_report_R1/
masklet_causal_bank/
rollouts/V29C_P5_CAUSAL_H10_R1_*
```

设置：

```text
chunk = 10
horizon = 10
selection = top projected-support masklet per chunk
dominant selected masklet = road / projected road
selected masklet count = 11 chunks
mean selected token count = 13295.2727
mean projected support count = 345279.1818
rows completed = 6/6
failures = 0
```

Rows：

| Candidate | Path/action | h10 ATE delta | h10 `[200,300)` delta | h10 `[400,600)` delta | Oracle pass |
|---|---|---:|---:|---:|---|
| `V29C_CAUSAL_FRAME_SKIP_TOP` | frame / source_skip | `0.0` | `0.0` | `0.0` | `false` |
| `V29C_CAUSAL_GLOBAL_SKIP_TOP` | global / source_skip | `-0.3934658463` | `-2.3381097022` | `-0.3988510683` | `false` |
| `V29C_CAUSAL_SWA_ANCHOR_TOP` | SWA / anchor_keep | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` | `false` |
| `V29C_CAUSAL_SWA_REMOVE_TOP` | SWA / remove | `-0.3641983213` | `-2.3655037795` | `-0.3077443114` | `false` |
| `V29C_CAUSAL_TTT_POS_TOP` | TTT / positive | `+0.0711710095` | `+0.1368956294` | `+0.0747473821` | `false` |
| `V29C_CAUSAL_TTT_NEG_TOP` | TTT / negative | `-0.1188432371` | `-0.0887012416` | `-0.1600059993` | `false` |

Oracle gate：

```text
Required:
    h10 ATE delta <= -3m
    or h10 [200,300) delta <= -5m

Observed best:
    best ATE delta = -0.3934658463m
    best [200,300) delta = -2.3655037795m

oracle_gate_pass = false
```

By path：

| Path | Rows | Best ATE effect | Best `[200,300)` effect |
|---|---:|---:|---:|
| frame | `1` | `0.0` | `0.0` |
| global | `1` | `0.3934658463` | `2.3381097022` |
| SWA | `2` | `0.3641983213` | `2.3655037795` |
| TTT | `2` | `0.1188432371` | `0.0887012416` |

Decision：

```text
Phase 5 masklet causal oracle = fail.

The trusted top-support road masklets do have a local h10 effect through global
and SWA paths, but the effect is below oracle threshold and not enough to justify
learning a quality-aware router or starting full semantic matrices.

No h15 confirmation is allowed because the h10 oracle gate failed.
No learned router, no no-GT selector, and no full online validation are allowed.
```

---

## 8. Not Started / Not Claimed

```text
1. No learned quality-aware runtime role router was trained or evaluated.
2. No h15 causal-bank confirmation was launched.
3. No Phase 3/Phase 4 all-memory rollout matrix was launched.
4. No no-GT selector was launched.
5. No full online validation was launched.
6. No online Target-25 result was produced.
```

Reason：

```text
The current landed work reaches a real h10 masklet-level causal bank,
but the causal oracle gate failed.
```

---

## 9. Downstream Decision

| Stage | Status | Reason |
|---|---|---|
| Disk budget gate | pass | initial free disk `242G` |
| Official data acquisition | pass | required zips available as valid zip data after KITTI/AVG S3 fallback |
| Sequence01 extraction | pass | `image_2`, `calib.txt`, `velodyne`, `labels` present |
| Phase 0 data audit | pass | frame hit `757/757`, calib parse pass, point/label count match |
| Sparse projection cache | pass | median projected pixels `15978`, focus coverage `0.0323` |
| Projected 2D semantic visual audit | pass | 10 frame montages generated; 5 representative frames manually inspected with no major misalignment |
| Masklet-3D alignment | pass | support ratio `0.9610`, structure/ground agreement `0.8353` |
| Offline action distinguishability | pass | intended action masks differ from semantic-only/risk-only |
| Masklet causal bank h10 | fail | best ATE effect `0.3935m`, best `[200,300)` effect `2.3655m`; oracle requires `>=3m` or `>=5m` |
| SWA hard-remove repair | fail | `SEMANTIC_ROLE_SWA_NEGATIVE_SCALE=1.0` remained identical to SWA anchor/remove |
| h15 rollout | not started | causal oracle gate failed |
| No-GT selector | not started | no trajectory candidate gate |
| Full online validation | not started | no selector/full-run entry |

Boundary：

```text
No v29C result counts as deployable online TTT write success.
Only short h10 causal-bank trajectory deltas are claimed from v29C.
No full online validation was launched.
No online Target-25 result was produced in v29C.
```

---

## 10. Final Decision

v29C 的真实成功点：

```text
1. Solved the v29B projection data blocker by obtaining official KITTI/SemanticKITTI sequence01 projection inputs.
2. Generated a valid sparse 3D-to-2D projected semantic cache for 757 selected frames.
3. Generated visual projection audit images and manually inspected representative
   projected 2D semantic overlays.
4. Verified sparse projection coverage is sufficient for diagnostics.
5. Aligned VideoMasklet masklets with projected 3D anchors.
6. Verified most non-sky/non-unknown masklets have sparse projected support.
7. Verified supported structure/ground masklets agree with projected semantic group above gate.
8. Verified projected-3D trust can produce distinguishable intended actions offline.
9. Implemented and executed a real h10 runtime masklet-level causal-bank screen.
```

关键尚未完成：

```text
The trusted top-support road masklets produce some h10 trajectory improvement
through global/SWA paths, but not enough for the oracle gate.

Best h10 causal-bank result:
    V29C_CAUSAL_GLOBAL_SKIP_TOP
    ATE delta = -0.3934658463m
    [200,300) delta = -2.3381097022m

Best h10 local result:
    V29C_CAUSAL_SWA_ANCHOR_TOP / V29C_CAUSAL_SWA_REMOVE_TOP
    [200,300) delta = -2.3655037795m

But oracle requires:
    h10 ATE delta <= -3m
    or h10 [200,300) delta <= -5m

No quality-aware runtime role router ATE exists yet.
No h10/h15 durability result exists yet.
```

Interpretation：

```text
v29C overturns the previous data-only blocker:
    projected 3D semantic anchors are now available and useful for masklet trust calibration.

The added visual audit supports the geometric projection:
    road / terrain / vegetation / fence / moving_car overlays are visually
    plausible in inspected frames,
    and no major projection flip or translation was observed.

But the scientific/Target-25 question has a negative first answer:
    the current top-support trusted-masklet interventions are far below the
    masklet causal oracle threshold.

The SWA result is especially bounded:
    SWA anchor and SWA remove produce identical trajectories even after a
    hard-remove scale repair, so the current SWA hook is not yet a clean
    differentiated masklet causal action.
```

Conclusion type：

```text
v29C has valid data/projection/alignment infrastructure.
It is not a successful Target-25 correction mechanism with the tested
top-support masklet causal actions.
```

Next required direction：

```text
Do not train a v29C learned router, do not launch h15, and do not start selector
or full online validation from the current v29C causal-bank rows.

If continuing semantic projection work, repair the SWA hook so anchor/remove
actions produce auditable distinct hook effects, then test a more selective
masklet bank:
    high disagreement masklets
    high D_g / high conflict masklets
    boundary-proximal masklets
    non-road structure masklets

Otherwise Target-25 mainline should return to non-semantic trajectory-state,
scale-state, lifecycle, or merge/gauge-aware correction.
```
