#!/usr/bin/env python3
"""Run a v107 scheduled G3 reactivation smoke inside the v106 rolling stream.

This is a bridge from the standalone live-state probe toward the Phase8
rolling scheduler: it uses the real v106 rolling SAM2 inference_state, removes
preselected objects from that state, then re-adds them with LingBot-projected
positive prompts and an online sibling-negative selector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
GSAM2_ROOT = Path(os.environ.get("GSAM2_ROOT", str(REPO_ROOT / "Grounded-SAM-2"))).resolve()
for item in (GSAM2_ROOT, STREAM3D_ROOT, REPO_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import tools.audit_v105_baseline_x_sam2_twostage_tracking as base  # noqa: E402
from tools.audit_v105_4dpm_style_per_frame_segmentors import parse_frame_ids, sha256_file  # noqa: E402
from tools.run_v106_stateful_sam2_scene_stream import _format_seconds, _write_side_by_side_video  # noqa: E402
from tools.run_v107_phase7_lingbot_sam2_prompt_benchmark import (  # noqa: E402
    autocast_kwargs as image_autocast_kwargs,
    build_sam2_predictor,
    jsonable,
    load_points,
    load_reference_records,
    map_lingbot_xy_to_original,
    mask_metrics,
    read_json,
    rel,
    resolve_path,
    write_json,
)
from tools.run_v107_phase5_prompt_capsule_visibility_probe import (  # noqa: E402
    bbox_distance,
    bbox_from_mask,
    load_lingbot_geometry as load_raw_lingbot_geometry,
    resize_label_to_shape,
    sample_mask_points_spread,
    visibility_project,
)
from tools.run_v107_phase8_sam2_live_state_reactivation_probe import (  # noqa: E402
    add_points_after_tracking_started,
    autocast_for as probe_autocast_for,
    reconsolidate_stream_state_outputs as probe_reconsolidate_stream_state_outputs,
    draw_zoom_overlay,
    event_frame_label,
    infer_lingbot_hw,
    point_arrays,
    points_for_event,
    prompt_point_rates,
    rgb_frame,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/v106/v106_stateful_sam2_rolling_scene_stream_area20k_preprune6_maxvis45_labelcompact_noempty.yaml"
DEFAULT_PROMPT_ROOT = REPO_ROOT / "Stream3D/outputs/audit/v107_phase8_lingbot_prompt_capsule_alllag_confirm_20260713_2308"
DEFAULT_PROBE_ROOT = (
    REPO_ROOT
    / "Stream3D/outputs/audit/v107_phase8_sam2_live_state_reactivation_probe24_confirm_reprompt_g3selector_vis3_20260714_0046"
)
DEFAULT_REFERENCE_ROOT = (
    REPO_ROOT
    / "Stream3D/outputs/audit/v106_stateful_sam2_rolling_scene0050_area20k_e1_preprune6_maxvis45_labelcompact_noempty_full90_gpu6_20260713_1505/v106_stateful_sam2_rolling_scene_stream"
)
VARIANT_ID = "v107_phase8_g3_rolling_scheduler_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--rgb-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed/scene0050_00")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompt-probe-root", default=str(DEFAULT_PROMPT_ROOT))
    parser.add_argument("--probe-root", default=str(DEFAULT_PROBE_ROOT))
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument(
        "--events",
        default="0,12,18,20",
        help="Comma-separated event indices, or 'auto' to select one lifecycle event per object from the setup.",
    )
    parser.add_argument("--visual-events", default="0,12,18,20")
    parser.add_argument("--frame-start", type=int, default=4450)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=10)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--seed", type=int, default=107)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--model-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16", "bf16"])
    parser.add_argument("--runtime-num-maskmem", type=int, default=3)
    parser.add_argument("--runtime-max-obj-ptrs-in-encoder", type=int, default=8)
    parser.add_argument("--runtime-max-cond-frames-in-attn", type=int, default=4)
    parser.add_argument("--stream-keep-noncond-frames", type=int, default=8)
    parser.add_argument("--stream-prune-invisible-after-frames", type=int, default=18)
    parser.add_argument("--stream-prune-min-visible-area", type=int, default=256)
    parser.add_argument(
        "--stream-prune-protect-min-ever-area",
        type=int,
        default=0,
        help=(
            "Default-off invisible-prune guard: objects whose historical max visible area reaches "
            "this threshold remain in the SAM2 stream state instead of being removed after a short "
            "temporary disappearance."
        ),
    )
    parser.add_argument(
        "--stream-prune-protect-max-objects",
        type=int,
        default=0,
        help="When the invisible-prune protection is enabled, protect at most this many expired objects per frame.",
    )
    parser.add_argument("--stream-prune-max-visible-area", type=int, default=0)
    parser.add_argument("--stream-prune-max-visible-area-ratio", type=float, default=0.45)
    parser.add_argument(
        "--stream-disjoin-policy",
        default="keep_order",
        choices=["keep_order", "small_first", "recent_overlap_first", "small_first_recent_tiebreak"],
        help=(
            "Ownership resolver for propagated active masks. small_first lets smaller active objects "
            "claim pixels before large planes; recent_overlap_first first prefers slots that overlap "
            "their own recent output; small_first_recent_tiebreak keeps small objects before large "
            "planes and only uses recent overlap inside coarse area buckets."
        ),
    )
    parser.add_argument(
        "--stream-disjoin-claim-kept-only",
        action="store_true",
        default=False,
        help=(
            "For small_first stream disjoin, masks rejected by the residual-area threshold do not claim "
            "pixels from later masks. Default-off preserves older runs."
        ),
    )
    parser.add_argument(
        "--stream-disjoin-min-area-px",
        type=int,
        default=0,
        help=(
            "Default 0 keeps the historical empty-ratio threshold for propagated active masks. "
            "Positive values use this absolute residual-area threshold only for stream disjoin, "
            "so small tracked residuals can keep their IDs without changing SAM gap segmentation."
        ),
    )
    parser.add_argument(
        "--stream-disjoin-recent-min-iou",
        type=float,
        default=0.0,
        help=(
            "For recent_overlap_first, treat a propagated slot as stable only when its current mask "
            "has at least this IoU with its own recent output mask."
        ),
    )
    parser.add_argument(
        "--stream-disjoin-recent-max-area-growth",
        type=float,
        default=0.0,
        help=(
            "For recent_overlap_first, ignore recent-overlap priority when the current mask area is "
            "larger than recent_area * this value. Default 0 disables the growth guard."
        ),
    )
    parser.add_argument(
        "--stream-oversized-prune-action",
        default="prune",
        choices=["prune", "alert_only", "suppress_output"],
        help=(
            "Action when an active propagated mask exceeds the max visible area. alert_only records the "
            "growth event without deleting the object from SAM2 state or current output."
        ),
    )
    parser.add_argument(
        "--stream-growth-prune-ratio",
        type=float,
        default=0.0,
        help="Default-off active-object growth guard: prune if current area exceeds recent median by this ratio.",
    )
    parser.add_argument(
        "--stream-growth-prune-min-area",
        type=int,
        default=0,
        help="Minimum visible area before the active-object growth guard can prune. 0 disables with ratio 0.",
    )
    parser.add_argument("--stream-growth-prune-history", type=int, default=5)
    parser.add_argument("--stream-growth-prune-warmup", type=int, default=3)
    parser.add_argument(
        "--stream-growth-prune-action",
        default="prune",
        choices=["prune", "alert_only", "suppress_output"],
        help="Action for the history-based growth guard; alert_only turns the guard into a lifecycle diagnostic.",
    )
    parser.add_argument(
        "--stream-growth-prune-max-history-median-area",
        type=int,
        default=0,
        help=(
            "Default-off guard for growth prune: when >0, only prune objects whose recent median visible "
            "area is at most this value. This targets tiny-support runaway masks while preserving normal "
            "large objects."
        ),
    )
    parser.add_argument("--stream-empty-cache-every", type=int, default=8)
    parser.add_argument("--stream-empty-cache-on-prune", action="store_true", default=True)
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=False)
    parser.add_argument("--offload-state-to-cpu", action="store_true", default=False)
    parser.add_argument("--gap-max-points", type=int, default=64)
    parser.add_argument("--gap-min-component-area", type=int, default=4096)
    parser.add_argument("--gap-area-per-extra-point", type=int, default=80000)
    parser.add_argument("--gap-max-points-per-component", type=int, default=3)
    parser.add_argument(
        "--gap-iou-threshold",
        type=float,
        default=None,
        help="Override the config gap pred_iou threshold used for strict gap mask selection.",
    )
    parser.add_argument(
        "--gap-stability-threshold",
        type=float,
        default=None,
        help="Override the config gap stability threshold used for strict gap mask selection.",
    )
    parser.add_argument(
        "--gap-relaxed-min-uncovered-ratio",
        type=float,
        default=0.0,
        help=(
            "Default-off gap fallback: when current uncovered ratio is at least this value, prompts "
            "with no strict-good mask may admit a lower-threshold candidate clipped to uncovered."
        ),
    )
    parser.add_argument(
        "--gap-relaxed-iou-threshold",
        type=float,
        default=0.0,
        help="Predicted-IoU threshold for the large-uncovered relaxed gap fallback; 0 disables it.",
    )
    parser.add_argument(
        "--gap-relaxed-stability-threshold",
        type=float,
        default=0.0,
        help="Stability threshold for the large-uncovered relaxed gap fallback; 0 disables it.",
    )
    parser.add_argument(
        "--gap-relaxed-min-clipped-area",
        type=int,
        default=0,
        help="Minimum uncovered-clipped candidate area for the relaxed gap fallback.",
    )
    parser.add_argument(
        "--gap-small-mask-max-area",
        type=int,
        default=0,
        help=(
            "Default-off gap birth quality gate: when >0, selected gap masks at or below this "
            "uncovered-clipped area are dropped before ID assignment/tracking if pred_iou is low."
        ),
    )
    parser.add_argument(
        "--gap-small-mask-min-pred-iou",
        type=float,
        default=0.0,
        help="Minimum pred_iou for small gap masks controlled by --gap-small-mask-max-area; 0 disables.",
    )
    parser.add_argument(
        "--gap-output-min-pred-iou",
        type=float,
        default=0.0,
        help="Default-off gap output quality gate: drop current-frame gap masks below this pred_iou.",
    )
    parser.add_argument(
        "--gap-output-min-stability",
        type=float,
        default=0.0,
        help="Default-off gap output quality gate: drop current-frame gap masks below this stability score.",
    )
    parser.add_argument(
        "--gap-output-disallow-relaxed",
        action="store_true",
        default=False,
        help="Drop current-frame gap masks that were selected only by relaxed fallback thresholds.",
    )
    parser.add_argument(
        "--gap-delayed-admission",
        action="store_true",
        default=False,
        help=(
            "Keep gap masks in the current output plane but only admit the selected durable subset to "
            "SAM2 stream memory. This is default-off to preserve older runs."
        ),
    )
    parser.add_argument(
        "--gap-admission-min-pred-iou",
        type=float,
        default=0.0,
        help="When --gap-delayed-admission is enabled, require this pred_iou for durable SAM2 admission.",
    )
    parser.add_argument(
        "--gap-admission-min-stability",
        type=float,
        default=0.0,
        help="When --gap-delayed-admission is enabled, require this stability score for durable SAM2 admission.",
    )
    parser.add_argument(
        "--gap-admission-disallow-relaxed",
        action="store_true",
        default=False,
        help="When delayed admission is enabled, do not durably admit masks selected only by relaxed gap thresholds.",
    )
    parser.add_argument(
        "--gap-reuse-recent-id-window",
        type=int,
        default=0,
        help=(
            "Default-off gap ID continuity repair: when >0, a gap birth may reuse a non-visible "
            "recent object id if mask IoU with the object's latest output is high enough."
        ),
    )
    parser.add_argument(
        "--gap-reuse-recent-id-iou",
        type=float,
        default=0.0,
        help="IoU threshold for reusing a recently visible object id during gap birth; 0 disables reuse.",
    )
    parser.add_argument(
        "--gap-reuse-recent-id-min-area",
        type=int,
        default=4096,
        help="Minimum previous and current mask area for the recent-ID gap reuse test.",
    )
    parser.add_argument(
        "--gap-large-reuse-recent-id-iou",
        type=float,
        default=0.0,
        help=(
            "Default-off large-mask temporal reuse lane for gap births. When >0, large masks may "
            "reuse a recent id at this lower IoU if the area-ratio constraint also holds."
        ),
    )
    parser.add_argument(
        "--gap-large-reuse-min-area",
        type=int,
        default=0,
        help="Minimum current and previous area for --gap-large-reuse-recent-id-iou.",
    )
    parser.add_argument(
        "--gap-large-reuse-max-area-ratio",
        type=float,
        default=0.0,
        help="Maximum max(area_a/area_b, area_b/area_a) for large gap reuse; 0 disables this constraint.",
    )
    parser.add_argument(
        "--gap-anti-merge-core-window-frames",
        type=int,
        default=0,
        help=(
            "Default-off durable gap admission guard: compare each gap birth with recent output cores "
            "from this many frames and keep multi-core candidates output-only."
        ),
    )
    parser.add_argument(
        "--gap-anti-merge-core-erode-px",
        type=int,
        default=0,
        help="Erode recent output masks by this many pixels before anti-merge core overlap checks.",
    )
    parser.add_argument(
        "--gap-anti-merge-core-min-area",
        type=int,
        default=0,
        help="Minimum recent output mask area to contribute an anti-merge ownership core.",
    )
    parser.add_argument(
        "--gap-anti-merge-core-min-overlap-px",
        type=int,
        default=0,
        help="Minimum candidate/core intersection pixels counted by the gap anti-merge guard.",
    )
    parser.add_argument(
        "--gap-anti-merge-core-min-overlap-ratio",
        type=float,
        default=0.0,
        help="Minimum fraction of an ownership core covered by a gap candidate for anti-merge counting.",
    )
    parser.add_argument(
        "--gap-anti-merge-max-overlap-objects",
        type=int,
        default=1,
        help="Durably reject gap candidates overlapping more than this many recent ownership cores.",
    )
    parser.add_argument(
        "--output-reuse-recent-id-window",
        type=int,
        default=0,
        help=(
            "Default-off output-plane ID continuity repair: high-IoU current output masks may be "
            "canonicalized to a recent lower object id before label/video export."
        ),
    )
    parser.add_argument(
        "--output-reuse-recent-id-iou",
        type=float,
        default=0.0,
        help="IoU threshold for output-plane recent-ID canonicalization; 0 disables it.",
    )
    parser.add_argument(
        "--output-reuse-recent-id-min-area",
        type=int,
        default=4096,
        help="Minimum previous and current mask area for output-plane recent-ID canonicalization.",
    )
    parser.add_argument(
        "--output-large-reuse-recent-id-iou",
        type=float,
        default=0.0,
        help=(
            "Default-off large-mask temporal reuse lane for output canonicalization. When >0, large "
            "masks may canonicalize to a recent id at this lower IoU if the area-ratio constraint holds."
        ),
    )
    parser.add_argument(
        "--output-large-reuse-min-area",
        type=int,
        default=0,
        help="Minimum current and previous area for --output-large-reuse-recent-id-iou.",
    )
    parser.add_argument(
        "--output-large-reuse-max-area-ratio",
        type=float,
        default=0.0,
        help="Maximum max(area_a/area_b, area_b/area_a) for large output reuse; 0 disables this constraint.",
    )
    parser.add_argument(
        "--output-reuse-recent-id-preference",
        default="lower_id",
        choices=["lower_id", "recent_id"],
        help=(
            "Canonical id preference for output-plane recent-ID matches. lower_id favors older/lower "
            "client ids; recent_id favors immediate temporal continuity with the previous output."
        ),
    )
    parser.add_argument(
        "--output-reuse-prevent-collision-union",
        action="store_true",
        default=False,
        help=(
            "When output recent-ID canonicalization maps multiple current masks to one canonical id, "
            "keep only the strongest owner and leave the others on their original ids instead of unioning."
        ),
    )
    parser.add_argument(
        "--output-fragment-max-area",
        type=int,
        default=0,
        help=(
            "Default-off output-only cleanup: disconnected components up to this area may be merged into "
            "a touching large neighbor before label/video export."
        ),
    )
    parser.add_argument(
        "--output-fragment-suppress-max-area",
        type=int,
        default=0,
        help="Default-off output-only cleanup: isolated components up to this area may be removed.",
    )
    parser.add_argument(
        "--output-fragment-merge-dilate-px",
        type=int,
        default=2,
        help="Dilation radius used to find touching neighbors for output fragment merge.",
    )
    parser.add_argument(
        "--output-fragment-merge-min-touch-px",
        type=int,
        default=16,
        help="Minimum dilated-border pixels touching a large neighbor before a fragment can be merged.",
    )
    parser.add_argument(
        "--output-fragment-merge-min-touch-ratio",
        type=float,
        default=0.02,
        help="Minimum touch_px / fragment_area before a fragment can be merged.",
    )
    parser.add_argument(
        "--output-fragment-merge-min-neighbor-area",
        type=int,
        default=20000,
        help="Minimum current-frame area of the target neighbor for output fragment merge.",
    )
    parser.add_argument("--disable-gap-birth", action="store_true", default=False)
    parser.add_argument(
        "--disable-output-plane",
        action="store_true",
        default=False,
        help=(
            "Keep reactivation diagnostics and memory-plane operations, but do not merge shadow/probation/"
            "confirm masks into the current-frame returned output mask set."
        ),
    )
    parser.add_argument(
        "--gap-min-image-edge-distance-px",
        type=int,
        default=0,
        help=(
            "Default-off component-adaptive gap sampler guard: when >0, require eligible gap prompt "
            "points at least this many pixels from the image border when possible, and rank candidates "
            "by min(component-distance, image-edge-distance) so touching full-image components do not "
            "select the image border as a false interior. Components with no eligible pixels fall back "
            "and record edge_margin_fallback."
        ),
    )
    parser.add_argument(
        "--gap-output-max-bbox-frac",
        type=float,
        default=0.0,
        help="Default-off gap output gate: drop current-frame gap masks whose bbox/image area exceeds this value.",
    )
    parser.add_argument(
        "--gap-output-max-edge-touch-count",
        type=int,
        default=-1,
        help="Default-off gap output gate: drop current-frame gap masks touching more than this many image edges.",
    )
    parser.add_argument(
        "--gap-output-min-extent",
        type=float,
        default=0.0,
        help="Default-off gap output gate: drop current-frame gap masks with area/bbox area below this value.",
    )
    parser.add_argument(
        "--gap-output-min-core-area-px",
        type=int,
        default=0,
        help="Default-off gap output gate: drop current-frame gap masks with too few pixels at distance >=16 px from boundary.",
    )
    parser.add_argument(
        "--gap-output-shape-min-uncovered-ratio",
        type=float,
        default=0.0,
        help=(
            "Only activate gap output shape gates when the current uncovered image ratio is at least this value. "
            "0 means active whenever any gap output shape threshold is enabled."
        ),
    )
    parser.add_argument(
        "--gap-output-min-input-mask-count",
        type=int,
        default=0,
        help="Only activate gap output shape gates when a gap call returns at least this many masks. 0 disables this extra condition.",
    )
    parser.add_argument("--birth-admission-min-area", type=int, default=20000)
    parser.add_argument("--birth-admission-max-area", type=int, default=0)
    parser.add_argument(
        "--birth-admission-max-uncovered-ratio",
        type=float,
        default=0.0,
        help=(
            "Reject post-start birth masks whose mask area is larger than this fraction of the current "
            "uncovered image area. 0 disables. This is a non-oracle guard for single gap prompts that "
            "grow into wall/floor-scale masks."
        ),
    )
    parser.add_argument(
        "--birth-admission-max-bbox-frac",
        type=float,
        default=0.0,
        help="Default-off birth anchor gate: reject post-start birth masks whose bbox/image area exceeds this value.",
    )
    parser.add_argument(
        "--birth-admission-max-edge-touch-count",
        type=int,
        default=-1,
        help="Default-off birth anchor gate: reject post-start birth masks touching more than this many image edges.",
    )
    parser.add_argument(
        "--birth-admission-min-extent",
        type=float,
        default=0.0,
        help="Default-off birth anchor gate: reject post-start birth masks with area/bbox area below this value.",
    )
    parser.add_argument(
        "--birth-admission-min-core-area-px",
        type=int,
        default=0,
        help="Default-off birth anchor gate: reject post-start birth masks with too few pixels at distance >=16 px from boundary.",
    )
    parser.add_argument(
        "--birth-admission-shape-min-uncovered-ratio",
        type=float,
        default=0.0,
        help=(
            "Only activate birth shape gates when current uncovered image ratio is at least this value. "
            "0 means shape gates are active whenever any shape threshold is enabled."
        ),
    )
    parser.add_argument("--birth-admission-every", type=int, default=1)
    parser.add_argument("--birth-admission-max-per-frame", type=int, default=6)
    parser.add_argument("--birth-admission-persistence-iou", type=float, default=0.0)
    parser.add_argument("--birth-admission-persistence-hits", type=int, default=0)
    parser.add_argument("--birth-admission-pending-ttl", type=int, default=0)
    parser.add_argument("--birth-admission-persistence-min-area", type=int, default=0)
    parser.add_argument("--birth-admission-persistence-max-per-frame", type=int, default=0)
    parser.add_argument("--birth-admission-immediate-area", type=int, default=0)
    parser.add_argument("--birth-admission-rescue-min-visible-count", type=int, default=0)
    parser.add_argument("--birth-admission-rescue-min-foreground-ratio", type=float, default=0.0)
    parser.add_argument("--birth-admission-appearance-enabled", action="store_true", default=False)
    parser.add_argument("--disable-birth-admission-appearance", action="store_true", default=False)
    parser.add_argument("--birth-admission-appearance-min-iou", type=float, default=0.02)
    parser.add_argument("--birth-admission-appearance-max-color-distance", type=float, default=0.16)
    parser.add_argument("--birth-admission-appearance-max-centroid-distance", type=float, default=96.0)
    parser.add_argument("--birth-admission-appearance-max-area-ratio", type=float, default=4.0)
    parser.add_argument("--birth-transaction-enabled", action="store_true", default=True)
    parser.add_argument("--disable-birth-transaction", action="store_true", default=False)
    parser.add_argument("--birth-transaction-min-pending", type=int, default=2)
    parser.add_argument("--birth-transaction-max-delay-frames", type=int, default=2)
    parser.add_argument("--birth-transaction-immediate-area", type=int, default=0)
    parser.add_argument("--birth-transaction-min-total-area", type=int, default=0)
    parser.add_argument("--birth-recon-prune-keep-frames", type=int, default=6)
    parser.add_argument("--online-select-neg-conflict-threshold", type=float, default=0.25)
    parser.add_argument("--online-select-min-g2-positive-support", type=float, default=0.50)
    parser.add_argument(
        "--image-g3-selector-g2-eval-policy",
        default="conflict_only",
        choices=["conflict_only", "always_if_negatives"],
        help=(
            "When image_g3_selector is active, conflict_only evaluates G2 only after G1 overlaps "
            "negative prompts; always_if_negatives also evaluates G2 whenever retained negative prompts "
            "exist, so co-visible negative evidence can actively constrain reactivation."
        ),
    )
    parser.add_argument(
        "--image-g3-selector-g2-select-policy",
        default="strict_improvement",
        choices=["strict_improvement", "strict_improvement_unless_g1_conflict", "not_worse"],
        help=(
            "How to choose G2 after it is evaluated. strict_improvement requires G2 to strictly reduce "
            "negative prompt conflict. strict_improvement_unless_g1_conflict keeps old not-worse "
            "behavior when G1 is already above the negative-conflict threshold."
        ),
    )
    parser.add_argument(
        "--image-g3-selector-g2-min-neg-conflict-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum reduction in negative prompt conflict required by "
            "strict_improvement_unless_g1_conflict when G1 is not above threshold. 0 still requires a "
            "strict reduction."
        ),
    )
    parser.add_argument("--min-source-mapping-iou", type=float, default=0.30)
    parser.add_argument("--unmapped-source-policy", default="skip", choices=["skip", "prompt_new_object"])
    parser.add_argument("--auto-source-lags", default="")
    parser.add_argument("--auto-min-target-source-area", type=int, default=0)
    parser.add_argument("--auto-min-positive-points", type=int, default=1)
    parser.add_argument("--auto-min-confirm-positive-points", type=int, default=1)
    parser.add_argument("--auto-max-events", type=int, default=0)
    parser.add_argument("--auto-max-events-per-object", type=int, default=1)
    parser.add_argument("--auto-selection-policy", default="default", choices=["default", "random", "area_only"])
    parser.add_argument("--long-term-min-source-area", type=int, default=0)
    parser.add_argument("--long-term-min-positive-points", type=int, default=1)
    parser.add_argument("--long-term-min-confirm-positive-points", type=int, default=1)
    parser.add_argument("--long-term-max-events", type=int, default=0)
    parser.add_argument(
        "--long-term-anchor-max-area-frac",
        type=float,
        default=0.0,
        help="Default-off physical-anchor gate: reject long-term memory candidates whose live mask area/image area exceeds this value.",
    )
    parser.add_argument(
        "--long-term-anchor-max-bbox-frac",
        type=float,
        default=0.0,
        help="Default-off physical-anchor gate: reject long-term memory candidates whose live bbox area/image area exceeds this value.",
    )
    parser.add_argument(
        "--long-term-anchor-max-edge-touch-count",
        type=int,
        default=-1,
        help="Default-off physical-anchor gate: reject long-term memory candidates touching more than this many image edges.",
    )
    parser.add_argument(
        "--long-term-anchor-min-extent",
        type=float,
        default=0.0,
        help="Default-off physical-anchor gate: reject long-term memory candidates whose live mask area/bbox area is below this value.",
    )
    parser.add_argument(
        "--long-term-anchor-min-core-area-px",
        type=int,
        default=0,
        help="Default-off physical-anchor gate: reject long-term memory candidates with too few pixels at distance >=16 px from mask boundary.",
    )
    parser.add_argument("--recoverability-mode", default="enabled", choices=["enabled", "disabled"])
    parser.add_argument(
        "--reactivation-prompt-mode",
        default="lingbot_geometry",
        choices=["lingbot_geometry", "no_geometry", "appearance_only", "appearance_geometry_filter", "random_geometry"],
        help=(
            "lingbot_geometry uses LingBot-Map projected visible prompts; no_geometry keeps lifecycle "
            "events but skips geometry-prompt reactivation; appearance_only uses current-frame 2D masks "
            "and RGB/shape descriptors without LingBot prompts; appearance_geometry_filter keeps the "
            "current-frame 2D mask selector but filters candidates with LingBot visible positive/negative points; "
            "random_geometry preserves prompt counts/roles but replaces projected coordinates with deterministic "
            "random image coordinates."
        ),
    )
    parser.add_argument("--appearance-only-min-score", type=float, default=0.55)
    parser.add_argument("--appearance-only-min-margin", type=float, default=0.02)
    parser.add_argument("--appearance-only-color-scale", type=float, default=0.35)
    parser.add_argument("--appearance-geometry-min-positive-support", type=float, default=0.50)
    parser.add_argument("--appearance-geometry-max-negative-conflict", type=float, default=0.05)
    parser.add_argument("--appearance-geometry-appearance-weight", type=float, default=0.55)
    parser.add_argument("--appearance-geometry-positive-weight", type=float, default=0.35)
    parser.add_argument(
        "--prompt-core-min-source-mask-distance-px",
        type=float,
        default=0.0,
        help=(
            "Drop positive and negative prompt points whose source-frame pixel is closer than this many "
            "pixels to the source object mask boundary. This avoids boundary samples that can drift onto "
            "neighboring objects after LingBot projection. 0 disables the filter."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-positive-points",
        type=int,
        default=0,
        help=(
            "Add this many extra positive prompt candidates per event/frame by sampling the historical "
            "source object mask core at LingBot resolution, projecting with LingBot raw geometry, and "
            "then letting the normal target filters decide. 0 disables supplementation."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-trigger-max-positive-points",
        type=int,
        default=0,
        help=(
            "When >0, add source-core supplement points only if the pre-supplement positive prompt count "
            "is at or below this limit. This keeps already well-supported large objects from becoming "
            "over-prompted."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-min-distance-px",
        type=float,
        default=0.0,
        help=(
            "Minimum LingBot-resolution distance from the historical source mask boundary for "
            "source-core supplement candidates. 0 samples the whole source mask."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-depth-abs-tolerance",
        type=float,
        default=0.12,
        help="LingBot projection absolute depth tolerance for source-core supplement points.",
    )
    parser.add_argument(
        "--prompt-source-core-supplement-depth-rel-tolerance",
        type=float,
        default=0.08,
        help="LingBot projection relative depth tolerance for source-core supplement points.",
    )
    parser.add_argument(
        "--prompt-source-core-supplement-min-depth-conf",
        type=float,
        default=0.0,
        help="Minimum LingBot source/target depth confidence for source-core supplement projection.",
    )
    parser.add_argument(
        "--prompt-source-core-supplement-duplicate-radius-px",
        type=float,
        default=2.0,
        help="Skip source-core supplement samples within this LingBot-pixel radius of an existing positive source point.",
    )
    parser.add_argument(
        "--prompt-source-core-supplement-negative-points",
        type=int,
        default=0,
        help=(
            "Add this many extra negative prompt candidates by sampling nearby co-visible source-frame "
            "object mask cores, projecting them with LingBot raw geometry, and then applying the normal "
            "source-core and target visibility filters. 0 disables the negative supplement."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-negative-trigger-max-negative-points",
        type=int,
        default=0,
        help=(
            "When >0, add co-visible negative supplement points only if the pre-supplement negative "
            "prompt count is at or below this limit."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-negative-min-distance-px",
        type=float,
        default=0.0,
        help=(
            "Minimum LingBot-resolution distance from the source negative object's mask boundary for "
            "co-visible negative supplement candidates. 0 samples the whole negative object mask."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-negative-max-neighbor-bbox-distance-px",
        type=float,
        default=0.0,
        help=(
            "Maximum LingBot-resolution bbox distance from the target source object to a candidate "
            "co-visible negative object. 0 allows any same-view source object."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-negative-target-border-margin-px",
        type=float,
        default=0.0,
        help=(
            "Drop projected co-visible negative supplement points closer than this many LingBot pixels "
            "to the target image border. This keeps audit visuals readable and avoids barely visible edge points."
        ),
    )
    parser.add_argument(
        "--prompt-source-core-supplement-negative-min-area-px",
        type=int,
        default=64,
        help="Minimum LingBot-resolution area for a co-visible source negative object candidate.",
    )
    parser.add_argument(
        "--prompt-source-core-supplement-negative-max-objects",
        type=int,
        default=4,
        help="Maximum number of nearest co-visible source objects to sample for negative supplementation. 0 means no cap.",
    )
    parser.add_argument(
        "--prompt-target-stability-depth-radius-px",
        type=int,
        default=0,
        help=(
            "When >0, evaluate LingBot target-depth local stability in this radius around each projected "
            "prompt point. Used with --prompt-target-stability-max-local-depth-range-m."
        ),
    )
    parser.add_argument(
        "--prompt-target-stability-max-local-depth-range-m",
        type=float,
        default=0.0,
        help=(
            "Drop projected prompt points whose LingBot target-depth patch max-min range exceeds this "
            "many meters. 0 disables the local depth-edge filter."
        ),
    )
    parser.add_argument(
        "--prompt-target-stability-max-depth-abs-error",
        type=float,
        default=0.0,
        help="Drop prompt rows whose LingBot projection depth_abs_error exceeds this value. 0 disables.",
    )
    parser.add_argument(
        "--prompt-target-stability-min-depth-conf",
        type=float,
        default=0.0,
        help="Drop prompt rows whose LingBot target_depth_conf is below this value. 0 disables.",
    )
    parser.add_argument(
        "--prompt-target-stability-min-valid-depth-count",
        type=int,
        default=4,
        help="Minimum valid LingBot depth pixels required inside the target-depth stability patch.",
    )
    parser.add_argument(
        "--prompt-anchor-conflict-negative-radius-px",
        type=float,
        default=0.0,
        help=(
            "Default-off physical-anchor prompt filter: drop positive projected points that fall within "
            "this LingBot-pixel radius of visible sibling-negative prompt points, while keeping the "
            "minimum positive-point floor."
        ),
    )
    parser.add_argument(
        "--prompt-anchor-conflict-positive-cluster-radius-px",
        type=float,
        default=0.0,
        help=(
            "Default-off physical-anchor prompt filter: keep the largest positive projected-point "
            "consensus component under this LingBot-pixel radius and drop isolated positive outliers."
        ),
    )
    parser.add_argument(
        "--prompt-anchor-conflict-min-positive-points",
        type=int,
        default=2,
        help="Minimum positive projected points retained by the physical-anchor conflict filter.",
    )
    parser.add_argument(
        "--prompt-target-mask-core-min-distance-px",
        type=float,
        default=0.0,
        help=(
            "After an online SAM2 image-prompt candidate is produced, drop positive prompt points whose "
            "target-frame pixel is closer than this many pixels to that candidate mask boundary. This is "
            "an online target-side edge guard and does not use reference labels. 0 disables the filter."
        ),
    )
    parser.add_argument(
        "--prompt-target-mask-core-min-positive-points",
        type=int,
        default=2,
        help=(
            "Minimum positive prompt points required after target-mask-core filtering; if the requested "
            "edge distance leaves too few positives, keep the farthest positive top-k and record the "
            "effective adaptive distance. The default keeps multiple interior points without forcing a "
            "third point back toward the target boundary."
        ),
    )
    parser.add_argument(
        "--reactivation-probation-mode",
        default="video_attempt_commit",
        choices=["video_attempt_commit", "shadow_attempt_confirm_commit"],
    )
    parser.add_argument("--probation-output-mode", default="image_g1", choices=["image_g1", "image_g3_selector"])
    parser.add_argument("--probation-min-positive-support", type=float, default=0.50)
    parser.add_argument("--probation-visual-events", default="")
    parser.add_argument("--shadow-output-mode", default="none", choices=["none", "image_g1", "image_g3_selector"])
    parser.add_argument("--shadow-min-source-area", type=int, default=20000)
    parser.add_argument("--shadow-min-positive-support", type=float, default=0.50)
    parser.add_argument("--shadow-max-events-per-frame", type=int, default=4)
    parser.add_argument("--shadow-visual-events", default="")
    parser.add_argument("--shadow-visual-frame-ids", default="")
    parser.add_argument("--sam2-checkpoint", default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--visual-pad", type=int, default=90)
    parser.add_argument("--visual-scale", type=int, default=2)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--skip-visual-export", action="store_true", default=False)
    parser.add_argument("--label-only-visual-export", action="store_true", default=True)
    parser.add_argument("--compact-visual-video", action="store_true", default=True)
    return parser.parse_args()


def as_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else REPO_ROOT / path


def model_dtype_for_probe(value: str) -> str:
    value_l = str(value).lower()
    if value_l in {"bfloat16", "bf16"}:
        return "bf16"
    if value_l in {"float16", "fp16"}:
        return "float16"
    return "float32"


def extract_mask(ids: list[int] | np.ndarray, masks: np.ndarray, obj_id: int, shape: tuple[int, int]) -> tuple[bool, np.ndarray]:
    ids_i = [int(v) for v in list(ids)]
    if int(obj_id) not in ids_i:
        return False, np.zeros(shape, dtype=bool)
    return True, np.asarray(masks[ids_i.index(int(obj_id))]).astype(bool)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = int(np.count_nonzero(a & b))
    if inter <= 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(inter / max(union, 1))


def mask_descriptor_rgb_shape(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    h, w = mask_b.shape[:2]
    area = int(np.count_nonzero(mask_b))
    if area <= 0:
        return {
            "area_px": 0,
            "mean_rgb": [0.0, 0.0, 0.0],
            "std_rgb": [0.0, 0.0, 0.0],
            "centroid_xy_norm": [0.0, 0.0],
            "bbox_wh_norm": [0.0, 0.0],
            "aspect_ratio": 0.0,
            "extent": 0.0,
        }
    pixels = np.asarray(rgb, dtype=np.float32)[mask_b] / 255.0
    ys, xs = np.where(mask_b)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    return {
        "area_px": int(area),
        "mean_rgb": [float(v) for v in pixels.mean(axis=0).tolist()],
        "std_rgb": [float(v) for v in pixels.std(axis=0).tolist()],
        "centroid_xy_norm": [
            float(xs.mean() / max(1, w - 1)),
            float(ys.mean() / max(1, h - 1)),
        ],
        "bbox_wh_norm": [float(bw / max(1, w)), float(bh / max(1, h))],
        "aspect_ratio": float(bw / max(1, bh)),
        "extent": float(area / max(1, bw * bh)),
    }


def mask_physical_anchor_stats(mask: np.ndarray) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    h, w = mask_b.shape[:2]
    image_area = max(1, int(h) * int(w))
    area = int(np.count_nonzero(mask_b))
    if area <= 0:
        return {
            "anchor_area_px": 0,
            "anchor_area_frac": 0.0,
            "anchor_bbox_area_frac": 0.0,
            "anchor_extent": 0.0,
            "anchor_bbox_xyxy": [],
            "anchor_bbox_wh": [0, 0],
            "anchor_edge_touch_count": 0,
            "anchor_touches_left": False,
            "anchor_touches_right": False,
            "anchor_touches_top": False,
            "anchor_touches_bottom": False,
            "anchor_core8_area_px": 0,
            "anchor_core16_area_px": 0,
        }
    ys, xs = np.where(mask_b)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    touches = {
        "left": bool(x0 == 0),
        "right": bool(x1 == w - 1),
        "top": bool(y0 == 0),
        "bottom": bool(y1 == h - 1),
    }
    dist = cv2.distanceTransform(mask_b.astype(np.uint8), cv2.DIST_L2, 3)
    return {
        "anchor_area_px": int(area),
        "anchor_area_frac": float(area / image_area),
        "anchor_bbox_area_frac": float((bw * bh) / image_area),
        "anchor_extent": float(area / max(1, bw * bh)),
        "anchor_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "anchor_bbox_wh": [int(bw), int(bh)],
        "anchor_edge_touch_count": int(sum(touches.values())),
        "anchor_touches_left": bool(touches["left"]),
        "anchor_touches_right": bool(touches["right"]),
        "anchor_touches_top": bool(touches["top"]),
        "anchor_touches_bottom": bool(touches["bottom"]),
        "anchor_core8_area_px": int(np.count_nonzero(dist >= 8.0)),
        "anchor_core16_area_px": int(np.count_nonzero(dist >= 16.0)),
    }


def physical_anchor_skip_reasons(stats: dict[str, Any], cli: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    max_area_frac = float(getattr(cli, "long_term_anchor_max_area_frac", 0.0))
    max_bbox_frac = float(getattr(cli, "long_term_anchor_max_bbox_frac", 0.0))
    max_edge_touch_count = int(getattr(cli, "long_term_anchor_max_edge_touch_count", -1))
    min_extent = float(getattr(cli, "long_term_anchor_min_extent", 0.0))
    min_core_area = int(getattr(cli, "long_term_anchor_min_core_area_px", 0))
    if max_area_frac > 0.0 and float(stats.get("anchor_area_frac", 0.0)) > max_area_frac:
        reasons.append("anchor_area_frac_above_max")
    if max_bbox_frac > 0.0 and float(stats.get("anchor_bbox_area_frac", 0.0)) > max_bbox_frac:
        reasons.append("anchor_bbox_frac_above_max")
    if max_edge_touch_count >= 0 and int(stats.get("anchor_edge_touch_count", 0)) > max_edge_touch_count:
        reasons.append("anchor_edge_touch_count_above_max")
    if min_extent > 0.0 and float(stats.get("anchor_extent", 0.0)) < min_extent:
        reasons.append("anchor_extent_below_min")
    if min_core_area > 0 and int(stats.get("anchor_core16_area_px", 0)) < min_core_area:
        reasons.append("anchor_core16_area_below_min")
    return reasons


def physical_anchor_readiness(event: dict[str, Any], cli: argparse.Namespace) -> dict[str, Any]:
    """Return non-reference physical evidence for durable-memory eligibility."""

    min_pos = max(1, int(getattr(cli, "prompt_anchor_conflict_min_positive_points", 2)))
    reasons: list[str] = []

    if not bool(event.get("geometry_prompts_enabled", False)):
        reasons.append("physical_anchor_geometry_prompts_disabled")
    if not bool(event.get("lingbot_prompt_points_available", False)):
        reasons.append("physical_anchor_lingbot_prompt_points_unavailable")
    if bool(event.get("random_geometry_prompts_enabled", False)):
        reasons.append("physical_anchor_random_geometry_provenance_invalid")
    if not bool(event.get("prompt_core_filter_enabled", False)):
        reasons.append("physical_anchor_source_core_filter_disabled")
    if not bool(event.get("prompt_target_stability_filter_enabled", False)):
        reasons.append("physical_anchor_target_depth_stability_filter_disabled")
    if float(event.get("prompt_target_stability_max_depth_abs_error", 0.0)) <= 0.0:
        reasons.append("physical_anchor_target_depth_abs_error_gate_disabled")
    if not bool(event.get("attempt_prompt_anchor_conflict_enabled", False)) or not bool(
        event.get("confirm_prompt_anchor_conflict_enabled", False)
    ):
        reasons.append("physical_anchor_conflict_filter_disabled")

    attempt_pos = int(event.get("attempt_positive_prompt_count_after_anchor_conflict", 0))
    confirm_pos = int(event.get("confirm_positive_prompt_count_after_anchor_conflict", 0))
    attempt_neg = int(event.get("attempt_negative_prompt_count_after_target_stability", 0))
    confirm_neg = int(event.get("confirm_negative_prompt_count_after_target_stability", 0))
    if attempt_pos < min_pos:
        reasons.append("physical_anchor_attempt_positive_count_below_min")
    if confirm_pos < min_pos:
        reasons.append("physical_anchor_confirm_positive_count_below_min")
    if attempt_neg <= 0:
        reasons.append("physical_anchor_attempt_negative_count_zero")
    if confirm_neg <= 0:
        reasons.append("physical_anchor_confirm_negative_count_zero")

    core_threshold = float(event.get("prompt_core_min_source_mask_distance_px", 0.0))
    if core_threshold > 0.0:
        attempt_core = float(event.get("attempt_prompt_core_min_retained_source_mask_distance_px", -1.0))
        confirm_core = float(event.get("confirm_prompt_core_min_retained_source_mask_distance_px", -1.0))
        if attempt_core < core_threshold:
            reasons.append("physical_anchor_attempt_source_core_distance_below_min")
        if confirm_core < core_threshold:
            reasons.append("physical_anchor_confirm_source_core_distance_below_min")

    return {
        "physical_anchor_ready": bool(not reasons),
        "physical_anchor_readiness_reasons": reasons,
        "physical_anchor_readiness_min_positive_points": int(min_pos),
        "physical_anchor_attempt_positive_after_conflict": int(attempt_pos),
        "physical_anchor_confirm_positive_after_conflict": int(confirm_pos),
        "physical_anchor_attempt_negative_after_stability": int(attempt_neg),
        "physical_anchor_confirm_negative_after_stability": int(confirm_neg),
        "physical_anchor_uses_lingbot_geometry": bool(event.get("geometry_prompts_enabled", False))
        and not bool(event.get("random_geometry_prompts_enabled", False)),
        "physical_anchor_requires_visual_confirmation": True,
        "physical_anchor_metrics_are_diagnostic_only": True,
    }


def ratio_score(a: float, b: float) -> float:
    a_f = max(float(a), 1e-6)
    b_f = max(float(b), 1e-6)
    return float(min(a_f, b_f) / max(a_f, b_f))


def appearance_descriptor_score(
    candidate: dict[str, Any],
    source: dict[str, Any],
    *,
    color_scale: float,
) -> dict[str, float]:
    cand_mean = np.asarray(candidate.get("mean_rgb", [0.0, 0.0, 0.0]), dtype=np.float32)
    src_mean = np.asarray(source.get("mean_rgb", [0.0, 0.0, 0.0]), dtype=np.float32)
    color_dist = float(np.linalg.norm(cand_mean - src_mean))
    color_score = float(np.exp(-color_dist / max(float(color_scale), 1e-6)))
    area = ratio_score(float(candidate.get("area_px", 0)), float(source.get("area_px", 0)))
    aspect = ratio_score(float(candidate.get("aspect_ratio", 0.0)), float(source.get("aspect_ratio", 0.0)))
    extent = ratio_score(float(candidate.get("extent", 0.0)), float(source.get("extent", 0.0)))
    shape = float(0.45 * area + 0.35 * aspect + 0.20 * extent)
    score = float(0.65 * color_score + 0.35 * shape)
    return {
        "appearance_score": score,
        "appearance_color_score": color_score,
        "appearance_color_l2": color_dist,
        "appearance_shape_score": shape,
        "appearance_area_score": area,
        "appearance_aspect_score": aspect,
        "appearance_extent_score": extent,
    }


def load_event_rows(probe_root: Path, event_indices: set[int] | None) -> list[dict[str, Any]]:
    payload = read_json(probe_root / "live_state_reactivation_event_setup.json")
    rows_all = [dict(row) for row in payload.get("events", [])]
    if event_indices is None:
        return sorted(rows_all, key=lambda row: int(row["event_index"]))
    rows = [row for row in rows_all if int(row["event_index"]) in event_indices]
    missing = sorted(event_indices - {int(row["event_index"]) for row in rows})
    if missing:
        raise RuntimeError(f"missing event setup rows in {probe_root}: {missing}")
    return sorted(rows, key=lambda row: int(row["event_index"]))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def event_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in {"", None}:
        return int(default)
    return int(value)


def choose_auto_events(
    events: list[dict[str, Any]],
    *,
    frame_ids: list[int],
    source_lags: set[int],
    min_target_source_area: int,
    min_positive_points: int,
    min_confirm_positive_points: int,
    max_events: int,
    max_events_per_object: int,
    selection_policy: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame_id_set = {int(v) for v in frame_ids}
    candidates: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}

    for row in sorted(events, key=lambda item: int(item["event_index"])):
        reason = ""
        required_frames = [
            event_int(row, "source_frame_id"),
            event_int(row, "attempt_frame_id"),
            event_int(row, "confirm_frame_id"),
        ]
        if any(frame_id not in frame_id_set for frame_id in required_frames):
            reason = "frame_outside_rolling_window"
        elif source_lags and event_int(row, "source_lag") not in source_lags:
            reason = "source_lag_not_requested"
        elif not truthy(row.get("usable_positive_negative_prompt", True)):
            reason = "not_usable_positive_negative_prompt"
        elif event_int(row, "target_source_area_px") < int(min_target_source_area):
            reason = "target_source_area_below_auto_min"
        elif event_int(row, "positive_point_count") < int(min_positive_points):
            reason = "positive_prompt_count_below_auto_min"
        elif event_int(row, "confirm_positive_point_count") < int(min_confirm_positive_points):
            reason = "confirm_positive_prompt_count_below_auto_min"

        selection_row = {
            "event_index": event_int(row, "event_index"),
            "global_id": event_int(row, "global_id"),
            "source_lag": event_int(row, "source_lag"),
            "source_frame_id": event_int(row, "source_frame_id"),
            "attempt_frame_id": event_int(row, "attempt_frame_id"),
            "confirm_frame_id": event_int(row, "confirm_frame_id"),
            "target_source_area_px": event_int(row, "target_source_area_px"),
            "positive_point_count": event_int(row, "positive_point_count"),
            "negative_point_count": event_int(row, "negative_point_count"),
            "confirm_positive_point_count": event_int(row, "confirm_positive_point_count"),
            "confirm_negative_point_count": event_int(row, "confirm_negative_point_count"),
            "selected": False,
            "skip_reason": reason,
        }
        if reason:
            skip_counts[reason] = int(skip_counts.get(reason, 0)) + 1
            selection_rows.append(selection_row)
            continue
        candidates.append(row)
        selection_rows.append(selection_row)

    ranked_by_object: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        gid = event_int(row, "global_id")
        ranked_by_object.setdefault(gid, []).append(row)

    per_object_cap = max(1, int(max_events_per_object))

    def stable_random_key(row: dict[str, Any]) -> float:
        payload = ":".join(
            [
                str(int(seed)),
                str(event_int(row, "event_index")),
                str(event_int(row, "global_id")),
                str(event_int(row, "source_frame_id")),
                str(event_int(row, "attempt_frame_id")),
                str(event_int(row, "confirm_frame_id")),
            ]
        ).encode("utf-8")
        raw = hashlib.sha256(payload).digest()[:8]
        return int.from_bytes(raw, "big") / float(2**64 - 1)

    def per_object_rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
        policy = str(selection_policy)
        if policy == "random":
            return (stable_random_key(item), event_int(item, "event_index"))
        if policy == "area_only":
            return (-event_int(item, "target_source_area_px"), event_int(item, "event_index"))
        return (
            -event_int(item, "source_lag"),
            -event_int(item, "target_source_area_px"),
            -(event_int(item, "positive_point_count") + event_int(item, "confirm_positive_point_count")),
            event_int(item, "event_index"),
        )

    def global_rank_key(item: dict[str, Any]) -> tuple[Any, ...]:
        policy = str(selection_policy)
        if policy == "random":
            return (stable_random_key(item), event_int(item, "event_index"))
        if policy == "area_only":
            return (-event_int(item, "target_source_area_px"), event_int(item, "event_index"))
        return (-event_int(item, "source_lag"), -event_int(item, "target_source_area_px"), event_int(item, "event_index"))

    selected_per_object: dict[int, list[dict[str, Any]]] = {}
    for gid, rows in ranked_by_object.items():
        selected_per_object[gid] = sorted(rows, key=per_object_rank_key)[:per_object_cap]

    selected = sorted(
        [row for rows in selected_per_object.values() for row in rows],
        key=global_rank_key,
    )
    if int(max_events) > 0 and len(selected) > int(max_events):
        selected_ids = {event_int(row, "event_index") for row in selected[: int(max_events)]}
        budget_skips = [row for row in selected[int(max_events) :]]
        selected = selected[: int(max_events)]
        for row in budget_skips:
            skip_counts["auto_max_events_budget_exhausted"] = int(
                skip_counts.get("auto_max_events_budget_exhausted", 0)
            ) + 1
            for selection_row in selection_rows:
                if event_int(selection_row, "event_index") == event_int(row, "event_index"):
                    selection_row["skip_reason"] = "auto_max_events_budget_exhausted"
                    break
    else:
        selected_ids = {event_int(row, "event_index") for row in selected}

    duplicate_selected_ids = selected_ids
    for row in candidates:
        if event_int(row, "event_index") in duplicate_selected_ids:
            continue
        if row not in selected_per_object.get(event_int(row, "global_id"), []):
            skip_counts["duplicate_object_lower_priority"] = int(skip_counts.get("duplicate_object_lower_priority", 0)) + 1
            for selection_row in selection_rows:
                if event_int(selection_row, "event_index") == event_int(row, "event_index"):
                    selection_row["skip_reason"] = "duplicate_object_lower_priority"
                    break

    for selection_row in selection_rows:
        selected_now = event_int(selection_row, "event_index") in selected_ids
        selection_row["selected"] = bool(selected_now)
        if selected_now:
            selection_row["skip_reason"] = ""

    summary = {
        "mode": "auto",
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "selected_event_indices": [event_int(row, "event_index") for row in selected],
        "skipped_count": int(sum(1 for row in selection_rows if not row["selected"])),
        "skip_reasons": skip_counts,
        "source_lags": sorted(source_lags),
        "auto_min_target_source_area": int(min_target_source_area),
        "auto_min_positive_points": int(min_positive_points),
        "auto_min_confirm_positive_points": int(min_confirm_positive_points),
        "auto_max_events": int(max_events),
        "auto_max_events_per_object": int(max_events_per_object),
        "auto_selection_policy": str(selection_policy),
        "auto_selection_seed": int(seed),
        "uses_reference_labels_for_final_acceptance": False,
        "event_setup_target_area_used_for_auto_smoke_candidate_generation_only": True,
    }
    return selected, selection_rows, summary


def load_prompt_summary(prompt_root: Path) -> dict[str, Any]:
    summary = read_json(prompt_root / "prompt_capsule_visibility_probe_summary.json")
    if "selected_visible_point_records" not in summary:
        summary["selected_visible_point_records"] = "prompt_capsule_visible_point_records_direct_as_c2w.json"
    if "selected_rows_csv" not in summary:
        summary["selected_rows_csv"] = "prompt_capsule_visibility_rows_direct_as_c2w.csv"
    return summary


def selected_prompt_file(prompt_root: Path, summary: dict[str, Any], key: str, fallback: str) -> Path:
    return resolve_path(str(summary.get(key) or fallback), prompt_root)


def load_prompt_target_depth_maps(
    prompt_root: Path,
    prompt_summary: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], Path]:
    raw = prompt_summary.get("raw_lingbot_geometry", {})
    npz_text = str(raw.get("npz_path", "")).strip()
    if not npz_text:
        raise RuntimeError("prompt target stability requires raw_lingbot_geometry.npz_path in prompt summary")
    npz_path = resolve_path(npz_text, prompt_root)
    with np.load(npz_path) as payload:
        required = {"frame_ids", "depth", "depth_conf"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError({"missing_lingbot_depth_keys": missing, "npz": str(npz_path)})
        frame_ids = [int(v) for v in np.asarray(payload["frame_ids"]).tolist()]
        depth = np.asarray(payload["depth"]).copy()
        depth_conf = np.asarray(payload["depth_conf"]).copy()
    if depth.shape[0] != len(frame_ids) or depth_conf.shape[0] != len(frame_ids):
        raise RuntimeError(
            {
                "invalid_lingbot_depth_shape": list(depth.shape),
                "invalid_lingbot_depth_conf_shape": list(depth_conf.shape),
                "frame_count": len(frame_ids),
            }
        )
    return (
        {int(frame_id): np.asarray(depth[idx]) for idx, frame_id in enumerate(frame_ids)},
        {int(frame_id): np.asarray(depth_conf[idx]) for idx, frame_id in enumerate(frame_ids)},
        npz_path,
    )


def prompt_source_core_supplement_enabled(config: dict[str, Any]) -> bool:
    return bool(int(config.get("positive_points", 0)) > 0)


def prompt_source_core_negative_supplement_enabled(config: dict[str, Any]) -> bool:
    return bool(int(config.get("negative_points", 0)) > 0)


def load_prompt_raw_geometry(prompt_root: Path, prompt_summary: dict[str, Any]) -> tuple[dict[str, np.ndarray], Path]:
    raw = prompt_summary.get("raw_lingbot_geometry", {})
    npz_text = str(raw.get("npz_path", "")).strip()
    if not npz_text:
        raise RuntimeError("source-core supplement requires raw_lingbot_geometry.npz_path in prompt summary")
    npz_path = resolve_path(npz_text, prompt_root)
    geometry = load_raw_lingbot_geometry(npz_path)
    required = {"frame_ids", "depth", "depth_conf", "intrinsics", "poses_direct", "poses_inverted"}
    missing = sorted(required - set(geometry.keys()))
    if missing:
        raise RuntimeError({"missing_lingbot_raw_geometry_keys_for_source_core_supplement": missing, "npz": str(npz_path)})
    return geometry, npz_path


def supplement_prompt_records_from_source_core(
    point_records: list[dict[str, Any]],
    *,
    source_label: np.ndarray,
    lingbot_hw: tuple[int, int],
    target_obj_id: int,
    source_frame_id: int,
    target_frame_id: int,
    pose_mode: str,
    geometry: dict[str, np.ndarray] | None,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_counts = prompt_record_role_counts(point_records)
    stats: dict[str, Any] = {
        "enabled": bool(prompt_source_core_supplement_enabled(config)),
        "positive_points_requested": int(config.get("positive_points", 0)),
        "trigger_max_positive_points": int(config.get("trigger_max_positive_points", 0)),
        "min_source_distance_px": float(config.get("min_source_distance_px", 0.0)),
        "depth_abs_tolerance": float(config.get("depth_abs_tolerance", 0.0)),
        "depth_rel_tolerance": float(config.get("depth_rel_tolerance", 0.0)),
        "min_depth_conf": float(config.get("min_depth_conf", 0.0)),
        "duplicate_radius_px": float(config.get("duplicate_radius_px", 0.0)),
        "record_count_before": int(before_counts["total"]),
        "positive_count_before": int(before_counts["positive"]),
        "negative_count_before": int(before_counts["negative"]),
        "record_count_after": int(before_counts["total"]),
        "positive_count_after": int(before_counts["positive"]),
        "negative_count_after": int(before_counts["negative"]),
        "candidate_source_pixel_count": 0,
        "sampled_count": 0,
        "projected_count": 0,
        "added_positive_count": 0,
        "duplicate_skipped_count": 0,
        "projection_rejection_reasons": {},
        "min_added_source_distance_px": -1.0,
        "max_added_source_distance_px": -1.0,
        "fallback_reason": "",
    }
    if not stats["enabled"]:
        return list(point_records), stats
    trigger_max_positive = int(config.get("trigger_max_positive_points", 0))
    if trigger_max_positive > 0 and int(before_counts["positive"]) > trigger_max_positive:
        stats["fallback_reason"] = "source_core_supplement_trigger_positive_count_above_limit"
        return list(point_records), stats
    if geometry is None:
        raise RuntimeError("source-core supplement enabled but LingBot raw geometry was not loaded")

    frame_ids = [int(v) for v in np.asarray(geometry["frame_ids"]).tolist()]
    frame_to_lingbot_idx = {int(frame_id): int(idx) for idx, frame_id in enumerate(frame_ids)}
    if int(source_frame_id) not in frame_to_lingbot_idx or int(target_frame_id) not in frame_to_lingbot_idx:
        stats["fallback_reason"] = "frame_id_missing_from_lingbot_geometry"
        return list(point_records), stats

    source_label_lingbot = resize_label_to_shape(source_label, lingbot_hw)
    source_mask = source_label_lingbot == int(target_obj_id)
    if not np.any(source_mask):
        stats["fallback_reason"] = "empty_source_object_mask"
        return list(point_records), stats

    dist = cv2.distanceTransform(source_mask.astype(np.uint8), cv2.DIST_L2, 3)
    min_distance = float(config.get("min_source_distance_px", 0.0))
    candidate_mask = source_mask if min_distance <= 0.0 else (dist >= min_distance)
    stats["candidate_source_pixel_count"] = int(np.count_nonzero(candidate_mask))
    if not np.any(candidate_mask):
        stats["fallback_reason"] = "empty_source_core_after_distance_filter"
        return list(point_records), stats

    duplicate_radius = max(0.0, float(config.get("duplicate_radius_px", 0.0)))
    duplicate_radius2 = duplicate_radius * duplicate_radius
    existing_pos_xy = [
        (float(row.get("source_x", -1.0)), float(row.get("source_y", -1.0)))
        for row in point_records
        if str(row.get("role", "")) == "positive"
    ]

    requested = int(config.get("positive_points", 0))
    sample_count = max(requested * 4, requested + len(existing_pos_xy), requested)
    sampled = sample_mask_points_spread(candidate_mask, count=sample_count, seed=int(seed))
    stats["sampled_count"] = int(len(sampled))
    pose_c2w = geometry["poses_direct"] if str(pose_mode) == "direct_as_c2w" else geometry["poses_inverted"]
    source_idx = int(frame_to_lingbot_idx[int(source_frame_id)])
    target_idx = int(frame_to_lingbot_idx[int(target_frame_id)])
    out = list(point_records)
    added_distances: list[float] = []
    projection_rejections: dict[str, int] = {}
    for sample_idx, (y, x) in enumerate(sampled):
        if len(added_distances) >= requested:
            break
        if duplicate_radius2 > 0.0:
            duplicate = any((float(x) - ex) ** 2 + (float(y) - ey) ** 2 <= duplicate_radius2 for ex, ey in existing_pos_xy)
            if duplicate:
                stats["duplicate_skipped_count"] = int(stats["duplicate_skipped_count"] + 1)
                continue
        projected, status, _meta = visibility_project(
            source_xy=(int(y), int(x)),
            source_index=source_idx,
            target_index=target_idx,
            geometry=geometry,
            pose_c2w=pose_c2w,
            depth_abs_tolerance=float(config.get("depth_abs_tolerance", 0.0)),
            depth_rel_tolerance=float(config.get("depth_rel_tolerance", 0.0)),
            min_depth_conf=float(config.get("min_depth_conf", 0.0)),
        )
        if projected is None:
            projection_rejections[str(status)] = int(projection_rejections.get(str(status), 0) + 1)
            continue
        source_distance = float(dist[int(y), int(x)])
        row = dict(projected)
        row.update(
            {
                "role": "positive",
                "pose_mode": str(pose_mode),
                "source_lag": int(target_idx - source_idx),
                "source_obj_id": int(target_obj_id),
                "target_obj_id": int(target_obj_id),
                "source_frame_index": int(source_idx),
                "source_frame_id": int(source_frame_id),
                "target_frame_index": int(target_idx),
                "target_frame_id": int(target_frame_id),
                "point_index": int(100000 + sample_idx),
                "prompt_source_core_supplement": True,
                "source_core_supplement_distance_px": source_distance,
            }
        )
        out.append(row)
        existing_pos_xy.append((float(x), float(y)))
        added_distances.append(source_distance)
    stats["projection_rejection_reasons"] = projection_rejections
    stats["projected_count"] = int(len(added_distances))
    stats["added_positive_count"] = int(len(added_distances))
    after_counts = prompt_record_role_counts(out)
    stats["record_count_after"] = int(after_counts["total"])
    stats["positive_count_after"] = int(after_counts["positive"])
    stats["negative_count_after"] = int(after_counts["negative"])
    stats["min_added_source_distance_px"] = float(min(added_distances)) if added_distances else -1.0
    stats["max_added_source_distance_px"] = float(max(added_distances)) if added_distances else -1.0
    return out, stats


def supplement_negative_prompt_records_from_coview_source_core(
    point_records: list[dict[str, Any]],
    *,
    source_label: np.ndarray,
    lingbot_hw: tuple[int, int],
    target_obj_id: int,
    source_frame_id: int,
    target_frame_id: int,
    pose_mode: str,
    geometry: dict[str, np.ndarray] | None,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_counts = prompt_record_role_counts(point_records)
    stats: dict[str, Any] = {
        "enabled": bool(prompt_source_core_negative_supplement_enabled(config)),
        "negative_points_requested": int(config.get("negative_points", 0)),
        "trigger_max_negative_points": int(config.get("trigger_max_negative_points", 0)),
        "min_source_distance_px": float(config.get("min_source_distance_px", 0.0)),
        "max_neighbor_bbox_distance_px": float(config.get("max_neighbor_bbox_distance_px", 0.0)),
        "target_border_margin_px": float(config.get("target_border_margin_px", 0.0)),
        "min_object_area_px": int(config.get("min_object_area_px", 0)),
        "max_objects": int(config.get("max_objects", 0)),
        "depth_abs_tolerance": float(config.get("depth_abs_tolerance", 0.0)),
        "depth_rel_tolerance": float(config.get("depth_rel_tolerance", 0.0)),
        "min_depth_conf": float(config.get("min_depth_conf", 0.0)),
        "duplicate_radius_px": float(config.get("duplicate_radius_px", 0.0)),
        "record_count_before": int(before_counts["total"]),
        "positive_count_before": int(before_counts["positive"]),
        "negative_count_before": int(before_counts["negative"]),
        "record_count_after": int(before_counts["total"]),
        "positive_count_after": int(before_counts["positive"]),
        "negative_count_after": int(before_counts["negative"]),
        "candidate_object_count": 0,
        "selected_object_ids": [],
        "candidate_source_pixel_count": 0,
        "sampled_count": 0,
        "projected_count": 0,
        "added_negative_count": 0,
        "duplicate_skipped_count": 0,
        "projection_rejection_reasons": {},
        "min_added_source_distance_px": -1.0,
        "max_added_source_distance_px": -1.0,
        "min_selected_neighbor_bbox_distance_px": -1.0,
        "max_selected_neighbor_bbox_distance_px": -1.0,
        "fallback_reason": "",
    }
    if not stats["enabled"]:
        return list(point_records), stats
    if int(before_counts["positive"]) <= 0:
        stats["fallback_reason"] = "source_core_negative_supplement_no_positive_prompt_context"
        return list(point_records), stats
    trigger_max_negative = int(config.get("trigger_max_negative_points", 0))
    if trigger_max_negative > 0 and int(before_counts["negative"]) > trigger_max_negative:
        stats["fallback_reason"] = "source_core_negative_supplement_trigger_negative_count_above_limit"
        return list(point_records), stats
    if geometry is None:
        raise RuntimeError("source-core negative supplement enabled but LingBot raw geometry was not loaded")

    frame_ids = [int(v) for v in np.asarray(geometry["frame_ids"]).tolist()]
    frame_to_lingbot_idx = {int(frame_id): int(idx) for idx, frame_id in enumerate(frame_ids)}
    if int(source_frame_id) not in frame_to_lingbot_idx or int(target_frame_id) not in frame_to_lingbot_idx:
        stats["fallback_reason"] = "frame_id_missing_from_lingbot_geometry"
        return list(point_records), stats

    source_label_lingbot = resize_label_to_shape(source_label, lingbot_hw)
    target_mask = source_label_lingbot == int(target_obj_id)
    target_bbox = bbox_from_mask(target_mask)
    if target_bbox is None:
        stats["fallback_reason"] = "empty_source_target_object_mask"
        return list(point_records), stats

    min_distance = float(config.get("min_source_distance_px", 0.0))
    max_neighbor_distance = float(config.get("max_neighbor_bbox_distance_px", 0.0))
    min_object_area = max(0, int(config.get("min_object_area_px", 0)))
    candidates: list[dict[str, Any]] = []
    for obj_id in sorted(int(v) for v in np.unique(source_label_lingbot).tolist()):
        if obj_id in {0, int(target_obj_id)}:
            continue
        obj_mask = source_label_lingbot == int(obj_id)
        area = int(np.count_nonzero(obj_mask))
        if area < min_object_area:
            continue
        obj_bbox = bbox_from_mask(obj_mask)
        if obj_bbox is None:
            continue
        neighbor_distance = float(bbox_distance(target_bbox, obj_bbox))
        if max_neighbor_distance > 0.0 and neighbor_distance > max_neighbor_distance:
            continue
        dist = cv2.distanceTransform(obj_mask.astype(np.uint8), cv2.DIST_L2, 3)
        candidate_mask = obj_mask if min_distance <= 0.0 else (dist >= min_distance)
        candidate_pixels = int(np.count_nonzero(candidate_mask))
        if candidate_pixels <= 0:
            continue
        candidates.append(
            {
                "obj_id": int(obj_id),
                "area": int(area),
                "bbox_distance": float(neighbor_distance),
                "candidate_pixels": int(candidate_pixels),
                "candidate_mask": candidate_mask,
                "dist": dist,
            }
        )
    candidates.sort(key=lambda row: (float(row["bbox_distance"]), -int(row["candidate_pixels"]), int(row["obj_id"])))
    max_objects = int(config.get("max_objects", 0))
    if max_objects > 0:
        candidates = candidates[:max_objects]
    stats["candidate_object_count"] = int(len(candidates))
    stats["selected_object_ids"] = [int(row["obj_id"]) for row in candidates]
    stats["candidate_source_pixel_count"] = int(sum(int(row["candidate_pixels"]) for row in candidates))
    if not candidates:
        stats["fallback_reason"] = "no_coview_negative_source_core_candidates"
        return list(point_records), stats

    duplicate_radius = max(0.0, float(config.get("duplicate_radius_px", 0.0)))
    duplicate_radius2 = duplicate_radius * duplicate_radius
    existing_neg_xy = [
        (float(row.get("source_x", -1.0)), float(row.get("source_y", -1.0)))
        for row in point_records
        if str(row.get("role", "")) == "negative"
    ]
    requested = int(config.get("negative_points", 0))
    samples: list[tuple[float, int, int, int, float]] = []
    for cand_index, candidate in enumerate(candidates):
        per_obj_count = max(1, requested * 2)
        obj_samples = sample_mask_points_spread(
            np.asarray(candidate["candidate_mask"], dtype=bool),
            count=per_obj_count,
            seed=int(seed) + cand_index * 9973 + int(candidate["obj_id"]),
        )
        for y, x in obj_samples:
            samples.append(
                (
                    float(candidate["bbox_distance"]),
                    int(candidate["obj_id"]),
                    int(y),
                    int(x),
                    float(candidate["dist"][int(y), int(x)]),
                )
            )
    samples.sort(key=lambda row: (row[0], row[1], -row[4], row[2], row[3]))
    stats["sampled_count"] = int(len(samples))

    pose_c2w = geometry["poses_direct"] if str(pose_mode) == "direct_as_c2w" else geometry["poses_inverted"]
    source_idx = int(frame_to_lingbot_idx[int(source_frame_id)])
    target_idx = int(frame_to_lingbot_idx[int(target_frame_id)])
    out = list(point_records)
    added_distances: list[float] = []
    selected_neighbor_distances: list[float] = []
    projection_rejections: dict[str, int] = {}
    for sample_idx, (neighbor_distance, obj_id, y, x, source_distance) in enumerate(samples):
        if len(added_distances) >= requested:
            break
        if duplicate_radius2 > 0.0:
            duplicate = any((float(x) - ex) ** 2 + (float(y) - ey) ** 2 <= duplicate_radius2 for ex, ey in existing_neg_xy)
            if duplicate:
                stats["duplicate_skipped_count"] = int(stats["duplicate_skipped_count"] + 1)
                continue
        projected, status, _meta = visibility_project(
            source_xy=(int(y), int(x)),
            source_index=source_idx,
            target_index=target_idx,
            geometry=geometry,
            pose_c2w=pose_c2w,
            depth_abs_tolerance=float(config.get("depth_abs_tolerance", 0.0)),
            depth_rel_tolerance=float(config.get("depth_rel_tolerance", 0.0)),
            min_depth_conf=float(config.get("min_depth_conf", 0.0)),
        )
        if projected is None:
            projection_rejections[str(status)] = int(projection_rejections.get(str(status), 0) + 1)
            continue
        target_border_margin = float(config.get("target_border_margin_px", 0.0))
        if target_border_margin > 0.0:
            target_depth = geometry["depth"][target_idx]
            target_h, target_w = target_depth.shape[:2]
            target_x = float(projected.get("target_x", -1.0))
            target_y = float(projected.get("target_y", -1.0))
            if (
                target_x < target_border_margin
                or target_y < target_border_margin
                or target_x > float(target_w - 1) - target_border_margin
                or target_y > float(target_h - 1) - target_border_margin
            ):
                projection_rejections["target_border_margin"] = int(
                    projection_rejections.get("target_border_margin", 0) + 1
                )
                continue
        row = dict(projected)
        row.update(
            {
                "role": "negative",
                "pose_mode": str(pose_mode),
                "source_lag": int(target_idx - source_idx),
                "source_obj_id": int(obj_id),
                "target_obj_id": int(target_obj_id),
                "source_frame_index": int(source_idx),
                "source_frame_id": int(source_frame_id),
                "target_frame_index": int(target_idx),
                "target_frame_id": int(target_frame_id),
                "point_index": int(200000 + sample_idx),
                "prompt_source_core_negative_supplement": True,
                "source_core_negative_supplement_distance_px": float(source_distance),
                "source_core_negative_supplement_neighbor_bbox_distance_px": float(neighbor_distance),
            }
        )
        out.append(row)
        existing_neg_xy.append((float(x), float(y)))
        added_distances.append(float(source_distance))
        selected_neighbor_distances.append(float(neighbor_distance))
    stats["projection_rejection_reasons"] = projection_rejections
    stats["projected_count"] = int(len(added_distances))
    stats["added_negative_count"] = int(len(added_distances))
    after_counts = prompt_record_role_counts(out)
    stats["record_count_after"] = int(after_counts["total"])
    stats["positive_count_after"] = int(after_counts["positive"])
    stats["negative_count_after"] = int(after_counts["negative"])
    stats["min_added_source_distance_px"] = float(min(added_distances)) if added_distances else -1.0
    stats["max_added_source_distance_px"] = float(max(added_distances)) if added_distances else -1.0
    stats["min_selected_neighbor_bbox_distance_px"] = (
        float(min(selected_neighbor_distances)) if selected_neighbor_distances else -1.0
    )
    stats["max_selected_neighbor_bbox_distance_px"] = (
        float(max(selected_neighbor_distances)) if selected_neighbor_distances else -1.0
    )
    return out, stats


def parse_int_set(text: str) -> set[int]:
    return {int(v) for v in str(text).split(",") if str(v).strip()}


def prompt_record_role_counts(point_records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": int(len(point_records)),
        "positive": int(sum(1 for row in point_records if str(row.get("role", "")) == "positive")),
        "negative": int(sum(1 for row in point_records if str(row.get("role", "")) == "negative")),
    }


def _distance_stat(values: list[float]) -> float:
    return float(min(values)) if values else -1.0


def filter_prompt_records_by_source_mask_core(
    point_records: list[dict[str, Any]],
    *,
    source_label: np.ndarray,
    lingbot_hw: tuple[int, int],
    min_source_mask_distance_px: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_counts = prompt_record_role_counts(point_records)
    threshold = float(min_source_mask_distance_px)
    stats: dict[str, Any] = {
        "enabled": bool(threshold > 0.0),
        "min_source_mask_distance_px": threshold,
        "record_count_before": int(before_counts["total"]),
        "positive_count_before": int(before_counts["positive"]),
        "negative_count_before": int(before_counts["negative"]),
        "record_count_after": int(before_counts["total"]),
        "positive_count_after": int(before_counts["positive"]),
        "negative_count_after": int(before_counts["negative"]),
        "dropped_record_count": 0,
        "min_retained_source_mask_distance_px": -1.0,
        "min_dropped_source_mask_distance_px": -1.0,
    }
    if threshold <= 0.0:
        return list(point_records), stats

    dist_maps: dict[int, np.ndarray] = {}
    retained: list[dict[str, Any]] = []
    retained_distances: list[float] = []
    dropped_distances: list[float] = []
    source_h, source_w = source_label.shape[:2]
    for row in point_records:
        role = str(row.get("role", ""))
        if role not in {"positive", "negative"}:
            retained.append(row)
            continue
        source_obj_id = int(row.get("source_obj_id", row.get("target_obj_id", 0)))
        if source_obj_id not in dist_maps:
            source_mask = (source_label == int(source_obj_id)).astype(np.uint8)
            dist_maps[source_obj_id] = (
                cv2.distanceTransform(source_mask, cv2.DIST_L2, 3)
                if np.any(source_mask)
                else np.zeros(source_label.shape[:2], dtype=np.float32)
            )
        sx, sy = map_lingbot_xy_to_original(
            float(row["source_x"]),
            float(row["source_y"]),
            lingbot_hw=lingbot_hw,
            orig_hw=source_label.shape[:2],
        )
        x = int(round(float(sx)))
        y = int(round(float(sy)))
        if 0 <= x < source_w and 0 <= y < source_h:
            distance_px = float(dist_maps[source_obj_id][y, x])
        else:
            distance_px = 0.0
        enriched = dict(row)
        enriched["source_mask_distance_px"] = float(distance_px)
        enriched["source_mask_core_min_distance_px"] = float(threshold)
        if distance_px >= threshold:
            retained.append(enriched)
            retained_distances.append(distance_px)
        else:
            dropped_distances.append(distance_px)

    after_counts = prompt_record_role_counts(retained)
    stats.update(
        {
            "record_count_after": int(after_counts["total"]),
            "positive_count_after": int(after_counts["positive"]),
            "negative_count_after": int(after_counts["negative"]),
            "dropped_record_count": int(before_counts["total"] - after_counts["total"]),
            "min_retained_source_mask_distance_px": _distance_stat(retained_distances),
            "min_dropped_source_mask_distance_px": _distance_stat(dropped_distances),
        }
    )
    return retained, stats


def prompt_target_stability_enabled(config: dict[str, Any]) -> bool:
    return bool(
        (int(config.get("depth_radius_px", 0)) > 0 and float(config.get("max_local_depth_range_m", 0.0)) > 0.0)
        or float(config.get("max_depth_abs_error", 0.0)) > 0.0
        or float(config.get("min_depth_conf", 0.0)) > 0.0
    )


def target_depth_local_stats(
    row: dict[str, Any],
    *,
    depth_by_frame_id: dict[int, np.ndarray],
    depth_conf_by_frame_id: dict[int, np.ndarray],
    radius_px: int,
) -> tuple[dict[str, Any], str]:
    frame_id = int(row["target_frame_id"])
    if frame_id not in depth_by_frame_id:
        return {}, "missing_target_depth_frame"
    depth = depth_by_frame_id[frame_id]
    depth_conf = depth_conf_by_frame_id.get(frame_id)
    h, w = depth.shape[:2]
    x = int(round(float(row.get("target_depth_x", row.get("target_x", -1)))))
    y = int(round(float(row.get("target_depth_y", row.get("target_y", -1)))))
    if x < 0 or y < 0 or x >= w or y >= h:
        return {
            "target_depth_stability_x": int(x),
            "target_depth_stability_y": int(y),
        }, "target_depth_stability_offscreen"
    radius = max(0, int(radius_px))
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    patch = depth[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0.0)
    values = patch[valid]
    center_depth = float(depth[y, x])
    center_conf = (
        float(depth_conf[y, x])
        if depth_conf is not None and depth_conf.shape == depth.shape
        else float(row.get("target_depth_conf", 1.0))
    )
    stats = {
        "target_depth_stability_x": int(x),
        "target_depth_stability_y": int(y),
        "target_depth_stability_radius_px": int(radius),
        "target_depth_stability_valid_count": int(values.size),
        "target_depth_center_m": float(center_depth),
        "target_depth_conf": float(center_conf),
        "target_depth_abs_error": float(row.get("depth_abs_error", -1.0)),
        "target_depth_local_range_m": -1.0,
        "target_depth_local_max_delta_m": -1.0,
    }
    if values.size == 0 or not np.isfinite(center_depth) or center_depth <= 0.0:
        return stats, "target_depth_stability_no_valid_depth"
    local_range = float(np.max(values) - np.min(values))
    max_delta = float(np.max(np.abs(values - center_depth)))
    stats.update(
        {
            "target_depth_local_range_m": local_range,
            "target_depth_local_max_delta_m": max_delta,
        }
    )
    return stats, "ok"


def filter_prompt_records_by_target_depth_stability(
    point_records: list[dict[str, Any]],
    *,
    depth_by_frame_id: dict[int, np.ndarray],
    depth_conf_by_frame_id: dict[int, np.ndarray],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_counts = prompt_record_role_counts(point_records)
    enabled = prompt_target_stability_enabled(config)
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "depth_radius_px": int(config.get("depth_radius_px", 0)),
        "max_local_depth_range_m": float(config.get("max_local_depth_range_m", 0.0)),
        "max_depth_abs_error": float(config.get("max_depth_abs_error", 0.0)),
        "min_depth_conf": float(config.get("min_depth_conf", 0.0)),
        "min_valid_depth_count": int(config.get("min_valid_depth_count", 0)),
        "record_count_before": int(before_counts["total"]),
        "positive_count_before": int(before_counts["positive"]),
        "negative_count_before": int(before_counts["negative"]),
        "record_count_after": int(before_counts["total"]),
        "positive_count_after": int(before_counts["positive"]),
        "negative_count_after": int(before_counts["negative"]),
        "dropped_record_count": 0,
        "drop_reasons": {},
        "max_retained_target_depth_local_range_m": -1.0,
        "min_dropped_target_depth_local_range_m": -1.0,
        "max_retained_target_depth_abs_error": -1.0,
        "min_dropped_target_depth_abs_error": -1.0,
    }
    if not enabled:
        return list(point_records), stats

    if not depth_by_frame_id:
        raise RuntimeError("target-depth-stability filter enabled but LingBot target depth maps were not loaded")

    retained: list[dict[str, Any]] = []
    retained_ranges: list[float] = []
    dropped_ranges: list[float] = []
    retained_errors: list[float] = []
    dropped_errors: list[float] = []
    drop_reasons: dict[str, int] = {}
    radius_px = int(config.get("depth_radius_px", 0))
    max_local_range = float(config.get("max_local_depth_range_m", 0.0))
    max_depth_abs_error = float(config.get("max_depth_abs_error", 0.0))
    min_depth_conf = float(config.get("min_depth_conf", 0.0))
    min_valid_depth_count = int(config.get("min_valid_depth_count", 0))

    for row in point_records:
        role = str(row.get("role", ""))
        if role not in {"positive", "negative"}:
            retained.append(row)
            continue
        local_stats, status = target_depth_local_stats(
            row,
            depth_by_frame_id=depth_by_frame_id,
            depth_conf_by_frame_id=depth_conf_by_frame_id,
            radius_px=radius_px,
        )
        enriched = dict(row)
        enriched.update(local_stats)
        reasons: list[str] = []
        depth_abs_error = float(enriched.get("target_depth_abs_error", row.get("depth_abs_error", -1.0)))
        target_conf = float(enriched.get("target_depth_conf", row.get("target_depth_conf", 1.0)))
        local_range = float(enriched.get("target_depth_local_range_m", -1.0))
        valid_count = int(enriched.get("target_depth_stability_valid_count", 0))
        if status != "ok":
            reasons.append(status)
        if max_depth_abs_error > 0.0 and (not np.isfinite(depth_abs_error) or depth_abs_error > max_depth_abs_error):
            reasons.append("target_depth_abs_error_above_threshold")
        if min_depth_conf > 0.0 and (not np.isfinite(target_conf) or target_conf < min_depth_conf):
            reasons.append("target_depth_conf_below_threshold")
        if radius_px > 0 and max_local_range > 0.0:
            if valid_count < min_valid_depth_count:
                reasons.append("target_depth_valid_count_below_threshold")
            if not np.isfinite(local_range) or local_range < 0.0:
                reasons.append("target_depth_local_range_invalid")
            elif local_range > max_local_range:
                reasons.append("target_depth_local_range_above_threshold")
        if reasons:
            reason_key = ";".join(sorted(set(reasons)))
            drop_reasons[reason_key] = int(drop_reasons.get(reason_key, 0) + 1)
            if local_range >= 0.0:
                dropped_ranges.append(local_range)
            if depth_abs_error >= 0.0:
                dropped_errors.append(depth_abs_error)
            continue
        retained.append(enriched)
        if local_range >= 0.0:
            retained_ranges.append(local_range)
        if depth_abs_error >= 0.0:
            retained_errors.append(depth_abs_error)

    after_counts = prompt_record_role_counts(retained)
    stats.update(
        {
            "record_count_after": int(after_counts["total"]),
            "positive_count_after": int(after_counts["positive"]),
            "negative_count_after": int(after_counts["negative"]),
            "dropped_record_count": int(before_counts["total"] - after_counts["total"]),
            "drop_reasons": drop_reasons,
            "max_retained_target_depth_local_range_m": float(max(retained_ranges)) if retained_ranges else -1.0,
            "min_dropped_target_depth_local_range_m": _distance_stat(dropped_ranges),
            "max_retained_target_depth_abs_error": float(max(retained_errors)) if retained_errors else -1.0,
            "min_dropped_target_depth_abs_error": _distance_stat(dropped_errors),
        }
    )
    return retained, stats


def prompt_anchor_conflict_enabled(config: dict[str, Any]) -> bool:
    return bool(
        float(config.get("negative_radius_px", 0.0)) > 0.0
        or float(config.get("positive_cluster_radius_px", 0.0)) > 0.0
    )


def _record_target_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    try:
        x = float(row.get("target_x"))
        y = float(row.get("target_y"))
    except Exception:
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def _largest_radius_component(coords: np.ndarray, radius_px: float) -> tuple[set[int], int]:
    if coords.shape[0] == 0:
        return set(), 0
    radius2 = float(radius_px) * float(radius_px)
    visited: set[int] = set()
    best: set[int] = set()
    component_count = 0
    for start in range(int(coords.shape[0])):
        if start in visited:
            continue
        component_count += 1
        stack = [int(start)]
        current: set[int] = set()
        visited.add(int(start))
        while stack:
            idx = stack.pop()
            current.add(int(idx))
            delta = coords - coords[int(idx)]
            neighbors = np.where(np.sum(delta * delta, axis=1) <= radius2)[0]
            for neighbor in neighbors.tolist():
                neighbor_i = int(neighbor)
                if neighbor_i not in visited:
                    visited.add(neighbor_i)
                    stack.append(neighbor_i)
        if len(current) > len(best):
            best = current
    return best, int(component_count)


def filter_prompt_records_by_anchor_conflict(
    point_records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_counts = prompt_record_role_counts(point_records)
    negative_radius = float(config.get("negative_radius_px", 0.0))
    cluster_radius = float(config.get("positive_cluster_radius_px", 0.0))
    min_positive_points = max(1, int(config.get("min_positive_points", 1)))
    enabled = prompt_anchor_conflict_enabled(config)
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "negative_radius_px": float(negative_radius),
        "positive_cluster_radius_px": float(cluster_radius),
        "min_positive_points": int(min_positive_points),
        "record_count_before": int(before_counts["total"]),
        "positive_count_before": int(before_counts["positive"]),
        "negative_count_before": int(before_counts["negative"]),
        "record_count_after": int(before_counts["total"]),
        "positive_count_after": int(before_counts["positive"]),
        "negative_count_after": int(before_counts["negative"]),
        "dropped_positive_count": 0,
        "dropped_positive_negative_conflict_count": 0,
        "dropped_positive_cluster_outlier_count": 0,
        "positive_supply_floor_kept_conflict_count": 0,
        "positive_cluster_component_count": 0,
        "positive_cluster_largest_size": 0,
        "min_dropped_negative_distance_px": -1.0,
        "min_retained_negative_distance_px": -1.0,
        "fallback_reason": "",
    }
    if not enabled:
        return list(point_records), stats

    records = [dict(row) for row in point_records]
    positive_items: list[tuple[int, tuple[float, float]]] = []
    negative_coords: list[tuple[float, float]] = []
    for idx, row in enumerate(records):
        xy = _record_target_xy(row)
        if xy is None:
            continue
        role = str(row.get("role", ""))
        if role == "positive":
            positive_items.append((int(idx), xy))
        elif role == "negative":
            negative_coords.append(xy)

    if len(positive_items) <= min_positive_points:
        stats["fallback_reason"] = "positive_count_at_or_below_min"
        return records, stats

    keep_indices = set(range(len(records)))
    retained_positive_indices = [idx for idx, _xy in positive_items]
    retained_negative_distances: list[float] = []
    dropped_negative_distances: list[float] = []

    if negative_radius > 0.0 and negative_coords:
        negative_arr = np.asarray(negative_coords, dtype=np.float32).reshape(-1, 2)
        conflict_candidates: list[tuple[float, int]] = []
        for idx, xy in positive_items:
            pos = np.asarray(xy, dtype=np.float32).reshape(1, 2)
            nearest = float(np.sqrt(np.min(np.sum((negative_arr - pos) ** 2, axis=1))))
            records[idx]["anchor_conflict_nearest_negative_distance_px"] = nearest
            if nearest <= negative_radius:
                conflict_candidates.append((nearest, int(idx)))
            else:
                retained_negative_distances.append(nearest)
        conflict_candidates.sort(key=lambda item: (item[0], item[1]))
        for nearest, idx in conflict_candidates:
            if len(retained_positive_indices) <= min_positive_points:
                stats["positive_supply_floor_kept_conflict_count"] = int(
                    stats["positive_supply_floor_kept_conflict_count"]
                ) + 1
                retained_negative_distances.append(float(nearest))
                continue
            keep_indices.discard(int(idx))
            retained_positive_indices.remove(int(idx))
            dropped_negative_distances.append(float(nearest))
            stats["dropped_positive_negative_conflict_count"] = int(
                stats["dropped_positive_negative_conflict_count"]
            ) + 1

    if cluster_radius > 0.0 and len(retained_positive_indices) > min_positive_points:
        cluster_items: list[tuple[int, tuple[float, float]]] = []
        for idx, xy in positive_items:
            if int(idx) in keep_indices:
                cluster_items.append((int(idx), xy))
        coords = np.asarray([xy for _idx, xy in cluster_items], dtype=np.float32).reshape(-1, 2)
        largest_component, component_count = _largest_radius_component(coords, cluster_radius)
        stats["positive_cluster_component_count"] = int(component_count)
        stats["positive_cluster_largest_size"] = int(len(largest_component))
        if len(largest_component) >= min_positive_points:
            keep_component_indices = {int(cluster_items[local_idx][0]) for local_idx in largest_component}
            for idx, _xy in cluster_items:
                if int(idx) not in keep_component_indices:
                    keep_indices.discard(int(idx))
                    stats["dropped_positive_cluster_outlier_count"] = int(
                        stats["dropped_positive_cluster_outlier_count"]
                    ) + 1
        else:
            stats["fallback_reason"] = "largest_positive_cluster_below_min"

    retained = [row for idx, row in enumerate(records) if int(idx) in keep_indices]
    after_counts = prompt_record_role_counts(retained)
    stats.update(
        {
            "record_count_after": int(after_counts["total"]),
            "positive_count_after": int(after_counts["positive"]),
            "negative_count_after": int(after_counts["negative"]),
            "dropped_positive_count": int(before_counts["positive"] - after_counts["positive"]),
            "min_dropped_negative_distance_px": _distance_stat(dropped_negative_distances),
            "min_retained_negative_distance_px": _distance_stat(retained_negative_distances),
        }
    )
    return retained, stats


def prompt_target_mask_core_enabled(min_distance_px: float) -> bool:
    return bool(float(min_distance_px) > 0.0)


def filter_prompt_coords_by_target_mask_core(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    mask: np.ndarray,
    min_distance_px: float,
    min_positive_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    coords_arr = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    if coords_arr.shape[0] != labels_arr.shape[0]:
        raise RuntimeError(
            {
                "invalid_prompt_coord_label_shape_for_target_mask_core": [
                    list(coords_arr.shape),
                    list(labels_arr.shape),
                ]
            }
        )
    enabled = prompt_target_mask_core_enabled(float(min_distance_px))
    positive_before = int(np.count_nonzero(labels_arr == 1))
    negative_before = int(np.count_nonzero(labels_arr == 0))
    requested_min_positive_points = max(0, int(min_positive_points))
    effective_min_positive_points = (
        min(requested_min_positive_points, positive_before) if requested_min_positive_points > 0 else 0
    )
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "filter_used": False,
        "filter_applied": False,
        "adaptive_topk_used": False,
        "fallback_reason": "",
        "min_distance_px": float(min_distance_px),
        "effective_min_distance_px": float(min_distance_px),
        "min_positive_points": int(requested_min_positive_points),
        "effective_min_positive_points": int(effective_min_positive_points),
        "positive_supply_below_min": bool(positive_before < requested_min_positive_points),
        "positive_count_before": positive_before,
        "positive_count_after_filter": positive_before,
        "positive_count_after": positive_before,
        "negative_count_before": negative_before,
        "negative_count_after": negative_before,
        "dropped_positive_count": 0,
        "min_retained_target_mask_distance_px": -1.0,
        "min_dropped_target_mask_distance_px": -1.0,
    }
    if not enabled:
        return coords_arr.copy(), labels_arr.copy(), stats
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim != 2 or mask_bool.size == 0 or not np.any(mask_bool):
        stats["fallback_reason"] = "empty_target_mask_candidate"
        return coords_arr.copy(), labels_arr.copy(), stats

    dist = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 3)
    h, w = dist.shape[:2]
    distances: list[float] = []
    for x_f, y_f in coords_arr:
        x = int(round(float(x_f)))
        y = int(round(float(y_f)))
        distances.append(float(dist[y, x]) if 0 <= x < w and 0 <= y < h else 0.0)

    keep: list[bool] = []
    for distance_px, label in zip(distances, labels_arr, strict=False):
        if int(label) != 1:
            keep.append(True)
            continue
        keep.append(bool(distance_px >= float(min_distance_px)))

    keep_arr = np.asarray(keep, dtype=bool)
    positive_after_requested = int(np.count_nonzero((labels_arr == 1) & keep_arr))
    if positive_after_requested < int(effective_min_positive_points):
        positive_distances = sorted(
            [float(distance) for distance, label in zip(distances, labels_arr, strict=False) if int(label) == 1],
            reverse=True,
        )
        cutoff = float(positive_distances[int(effective_min_positive_points) - 1])
        keep_arr = np.asarray(
            [
                True if int(label) != 1 else bool(float(distance) >= cutoff)
                for distance, label in zip(distances, labels_arr, strict=False)
            ],
            dtype=bool,
        )
        stats["adaptive_topk_used"] = True
        stats["effective_min_distance_px"] = cutoff
    elif positive_after_requested < int(requested_min_positive_points) and positive_before <= 0:
        stats["fallback_reason"] = "target_mask_core_no_positive_points_before_filter"
        stats["positive_count_after"] = positive_before
        stats["negative_count_after"] = negative_before
        return coords_arr.copy(), labels_arr.copy(), stats

    filtered_coords = coords_arr[keep_arr]
    filtered_labels = labels_arr[keep_arr]
    positive_after_filter = int(np.count_nonzero(filtered_labels == 1))
    retained_distances = [
        float(distance)
        for distance, label, keep_flag in zip(distances, labels_arr, keep_arr, strict=False)
        if int(label) == 1 and bool(keep_flag)
    ]
    dropped_distances = [
        float(distance)
        for distance, label, keep_flag in zip(distances, labels_arr, keep_arr, strict=False)
        if int(label) == 1 and not bool(keep_flag)
    ]
    stats.update(
        {
            "positive_count_after_filter": positive_after_filter,
            "negative_count_after": int(np.count_nonzero(filtered_labels == 0)),
            "dropped_positive_count": int(positive_before - positive_after_filter),
            "min_retained_target_mask_distance_px": _distance_stat(retained_distances),
            "min_dropped_target_mask_distance_px": _distance_stat(dropped_distances),
        }
    )
    stats["filter_used"] = True
    stats["filter_applied"] = bool(stats["dropped_positive_count"] > 0)
    stats["positive_count_after"] = positive_after_filter
    return filtered_coords, filtered_labels, stats


def split_positive_prompt_coords(coords: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords_arr = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    pos = labels_arr == 1
    return coords_arr[pos], labels_arr[pos]


def deterministic_random_prompt_coords(
    *,
    count: int,
    shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    n = max(0, int(count))
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    h, w = int(shape[0]), int(shape[1])
    rng = np.random.default_rng(int(seed))
    xs = rng.integers(0, max(1, w), size=n, endpoint=False)
    ys = rng.integers(0, max(1, h), size=n, endpoint=False)
    return np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)


def randomize_prompt_set_geometry(
    *,
    coords: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
    rand_coords = deterministic_random_prompt_coords(count=int(labels_arr.shape[0]), shape=shape, seed=int(seed))
    return rand_coords, labels_arr


def flatten_prompt_target_mask_core_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_prompt_target_mask_core_enabled": bool(stats.get("enabled", False)),
        f"{prefix}_prompt_target_mask_core_filter_used": bool(stats.get("filter_used", False)),
        f"{prefix}_prompt_target_mask_core_filter_applied": bool(stats.get("filter_applied", False)),
        f"{prefix}_prompt_target_mask_core_adaptive_topk_used": bool(stats.get("adaptive_topk_used", False)),
        f"{prefix}_prompt_target_mask_core_fallback_reason": str(stats.get("fallback_reason", "")),
        f"{prefix}_prompt_target_mask_core_min_distance_px": float(stats.get("min_distance_px", 0.0)),
        f"{prefix}_prompt_target_mask_core_effective_min_distance_px": float(
            stats.get("effective_min_distance_px", stats.get("min_distance_px", 0.0))
        ),
        f"{prefix}_prompt_target_mask_core_min_positive_points": int(stats.get("min_positive_points", 0)),
        f"{prefix}_prompt_target_mask_core_effective_min_positive_points": int(
            stats.get("effective_min_positive_points", stats.get("min_positive_points", 0))
        ),
        f"{prefix}_prompt_target_mask_core_positive_supply_below_min": bool(
            stats.get("positive_supply_below_min", False)
        ),
        f"{prefix}_positive_prompt_count_before_target_mask_core": int(stats.get("positive_count_before", 0)),
        f"{prefix}_positive_prompt_count_after_target_mask_core_filter": int(
            stats.get("positive_count_after_filter", 0)
        ),
        f"{prefix}_positive_prompt_count_after_target_mask_core": int(stats.get("positive_count_after", 0)),
        f"{prefix}_negative_prompt_count_before_target_mask_core": int(stats.get("negative_count_before", 0)),
        f"{prefix}_negative_prompt_count_after_target_mask_core": int(stats.get("negative_count_after", 0)),
        f"{prefix}_prompt_target_mask_core_dropped_positive_count": int(stats.get("dropped_positive_count", 0)),
        f"{prefix}_prompt_target_mask_core_min_retained_distance_px": float(
            stats.get("min_retained_target_mask_distance_px", -1.0)
        ),
        f"{prefix}_prompt_target_mask_core_min_dropped_distance_px": float(
            stats.get("min_dropped_target_mask_distance_px", -1.0)
        ),
    }


def flatten_prompt_source_core_supplement_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_prompt_source_core_supplement_enabled": bool(stats.get("enabled", False)),
        f"{prefix}_prompt_source_core_supplement_positive_points_requested": int(
            stats.get("positive_points_requested", 0)
        ),
        f"{prefix}_prompt_source_core_supplement_trigger_max_positive_points": int(
            stats.get("trigger_max_positive_points", 0)
        ),
        f"{prefix}_prompt_source_core_supplement_min_source_distance_px": float(
            stats.get("min_source_distance_px", 0.0)
        ),
        f"{prefix}_prompt_source_core_supplement_record_count_before": int(stats.get("record_count_before", 0)),
        f"{prefix}_prompt_source_core_supplement_record_count_after": int(stats.get("record_count_after", 0)),
        f"{prefix}_positive_prompt_count_before_source_core_supplement": int(stats.get("positive_count_before", 0)),
        f"{prefix}_positive_prompt_count_after_source_core_supplement": int(stats.get("positive_count_after", 0)),
        f"{prefix}_prompt_source_core_supplement_candidate_source_pixel_count": int(
            stats.get("candidate_source_pixel_count", 0)
        ),
        f"{prefix}_prompt_source_core_supplement_sampled_count": int(stats.get("sampled_count", 0)),
        f"{prefix}_prompt_source_core_supplement_projected_count": int(stats.get("projected_count", 0)),
        f"{prefix}_prompt_source_core_supplement_added_positive_count": int(
            stats.get("added_positive_count", 0)
        ),
        f"{prefix}_prompt_source_core_supplement_duplicate_skipped_count": int(
            stats.get("duplicate_skipped_count", 0)
        ),
        f"{prefix}_prompt_source_core_supplement_projection_rejection_reasons": dict(
            stats.get("projection_rejection_reasons", {})
        ),
        f"{prefix}_prompt_source_core_supplement_min_added_source_distance_px": float(
            stats.get("min_added_source_distance_px", -1.0)
        ),
        f"{prefix}_prompt_source_core_supplement_max_added_source_distance_px": float(
            stats.get("max_added_source_distance_px", -1.0)
        ),
        f"{prefix}_prompt_source_core_supplement_fallback_reason": str(stats.get("fallback_reason", "")),
    }


def flatten_prompt_source_core_negative_supplement_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_prompt_source_core_negative_supplement_enabled": bool(stats.get("enabled", False)),
        f"{prefix}_prompt_source_core_negative_supplement_negative_points_requested": int(
            stats.get("negative_points_requested", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_trigger_max_negative_points": int(
            stats.get("trigger_max_negative_points", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_min_source_distance_px": float(
            stats.get("min_source_distance_px", 0.0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_max_neighbor_bbox_distance_px": float(
            stats.get("max_neighbor_bbox_distance_px", 0.0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_target_border_margin_px": float(
            stats.get("target_border_margin_px", 0.0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_min_object_area_px": int(
            stats.get("min_object_area_px", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_max_objects": int(stats.get("max_objects", 0)),
        f"{prefix}_prompt_source_core_negative_supplement_record_count_before": int(
            stats.get("record_count_before", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_record_count_after": int(
            stats.get("record_count_after", 0)
        ),
        f"{prefix}_negative_prompt_count_before_source_core_negative_supplement": int(
            stats.get("negative_count_before", 0)
        ),
        f"{prefix}_negative_prompt_count_after_source_core_negative_supplement": int(
            stats.get("negative_count_after", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_candidate_object_count": int(
            stats.get("candidate_object_count", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_selected_object_ids": list(
            stats.get("selected_object_ids", [])
        ),
        f"{prefix}_prompt_source_core_negative_supplement_candidate_source_pixel_count": int(
            stats.get("candidate_source_pixel_count", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_sampled_count": int(stats.get("sampled_count", 0)),
        f"{prefix}_prompt_source_core_negative_supplement_projected_count": int(stats.get("projected_count", 0)),
        f"{prefix}_prompt_source_core_negative_supplement_added_negative_count": int(
            stats.get("added_negative_count", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_duplicate_skipped_count": int(
            stats.get("duplicate_skipped_count", 0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_projection_rejection_reasons": dict(
            stats.get("projection_rejection_reasons", {})
        ),
        f"{prefix}_prompt_source_core_negative_supplement_min_added_source_distance_px": float(
            stats.get("min_added_source_distance_px", -1.0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_max_added_source_distance_px": float(
            stats.get("max_added_source_distance_px", -1.0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_min_selected_neighbor_bbox_distance_px": float(
            stats.get("min_selected_neighbor_bbox_distance_px", -1.0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_max_selected_neighbor_bbox_distance_px": float(
            stats.get("max_selected_neighbor_bbox_distance_px", -1.0)
        ),
        f"{prefix}_prompt_source_core_negative_supplement_fallback_reason": str(stats.get("fallback_reason", "")),
    }


def flatten_prompt_core_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_prompt_record_count_before_core_filter": int(stats.get("record_count_before", 0)),
        f"{prefix}_prompt_record_count_after_core_filter": int(stats.get("record_count_after", 0)),
        f"{prefix}_positive_prompt_count_before_core_filter": int(stats.get("positive_count_before", 0)),
        f"{prefix}_positive_prompt_count_after_core_filter": int(stats.get("positive_count_after", 0)),
        f"{prefix}_negative_prompt_count_before_core_filter": int(stats.get("negative_count_before", 0)),
        f"{prefix}_negative_prompt_count_after_core_filter": int(stats.get("negative_count_after", 0)),
        f"{prefix}_prompt_core_dropped_record_count": int(stats.get("dropped_record_count", 0)),
        f"{prefix}_prompt_core_min_retained_source_mask_distance_px": float(
            stats.get("min_retained_source_mask_distance_px", -1.0)
        ),
        f"{prefix}_prompt_core_min_dropped_source_mask_distance_px": float(
            stats.get("min_dropped_source_mask_distance_px", -1.0)
        ),
    }


def flatten_prompt_target_stability_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_prompt_target_stability_enabled": bool(stats.get("enabled", False)),
        f"{prefix}_prompt_target_stability_record_count_before": int(stats.get("record_count_before", 0)),
        f"{prefix}_prompt_target_stability_record_count_after": int(stats.get("record_count_after", 0)),
        f"{prefix}_positive_prompt_count_before_target_stability": int(stats.get("positive_count_before", 0)),
        f"{prefix}_positive_prompt_count_after_target_stability": int(stats.get("positive_count_after", 0)),
        f"{prefix}_negative_prompt_count_before_target_stability": int(stats.get("negative_count_before", 0)),
        f"{prefix}_negative_prompt_count_after_target_stability": int(stats.get("negative_count_after", 0)),
        f"{prefix}_prompt_target_stability_dropped_record_count": int(stats.get("dropped_record_count", 0)),
        f"{prefix}_prompt_target_stability_drop_reasons": dict(stats.get("drop_reasons", {})),
        f"{prefix}_prompt_target_stability_max_retained_local_depth_range_m": float(
            stats.get("max_retained_target_depth_local_range_m", -1.0)
        ),
        f"{prefix}_prompt_target_stability_min_dropped_local_depth_range_m": float(
            stats.get("min_dropped_target_depth_local_range_m", -1.0)
        ),
        f"{prefix}_prompt_target_stability_max_retained_depth_abs_error": float(
            stats.get("max_retained_target_depth_abs_error", -1.0)
        ),
        f"{prefix}_prompt_target_stability_min_dropped_depth_abs_error": float(
            stats.get("min_dropped_target_depth_abs_error", -1.0)
        ),
    }


def flatten_prompt_anchor_conflict_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_prompt_anchor_conflict_enabled": bool(stats.get("enabled", False)),
        f"{prefix}_prompt_anchor_conflict_negative_radius_px": float(stats.get("negative_radius_px", 0.0)),
        f"{prefix}_prompt_anchor_conflict_positive_cluster_radius_px": float(
            stats.get("positive_cluster_radius_px", 0.0)
        ),
        f"{prefix}_prompt_anchor_conflict_min_positive_points": int(stats.get("min_positive_points", 0)),
        f"{prefix}_prompt_anchor_conflict_record_count_before": int(stats.get("record_count_before", 0)),
        f"{prefix}_prompt_anchor_conflict_record_count_after": int(stats.get("record_count_after", 0)),
        f"{prefix}_positive_prompt_count_before_anchor_conflict": int(stats.get("positive_count_before", 0)),
        f"{prefix}_positive_prompt_count_after_anchor_conflict": int(stats.get("positive_count_after", 0)),
        f"{prefix}_negative_prompt_count_before_anchor_conflict": int(stats.get("negative_count_before", 0)),
        f"{prefix}_negative_prompt_count_after_anchor_conflict": int(stats.get("negative_count_after", 0)),
        f"{prefix}_prompt_anchor_conflict_dropped_positive_count": int(stats.get("dropped_positive_count", 0)),
        f"{prefix}_prompt_anchor_conflict_dropped_positive_negative_conflict_count": int(
            stats.get("dropped_positive_negative_conflict_count", 0)
        ),
        f"{prefix}_prompt_anchor_conflict_dropped_positive_cluster_outlier_count": int(
            stats.get("dropped_positive_cluster_outlier_count", 0)
        ),
        f"{prefix}_prompt_anchor_conflict_positive_supply_floor_kept_conflict_count": int(
            stats.get("positive_supply_floor_kept_conflict_count", 0)
        ),
        f"{prefix}_prompt_anchor_conflict_positive_cluster_component_count": int(
            stats.get("positive_cluster_component_count", 0)
        ),
        f"{prefix}_prompt_anchor_conflict_positive_cluster_largest_size": int(
            stats.get("positive_cluster_largest_size", 0)
        ),
        f"{prefix}_prompt_anchor_conflict_min_dropped_negative_distance_px": float(
            stats.get("min_dropped_negative_distance_px", -1.0)
        ),
        f"{prefix}_prompt_anchor_conflict_min_retained_negative_distance_px": float(
            stats.get("min_retained_negative_distance_px", -1.0)
        ),
        f"{prefix}_prompt_anchor_conflict_fallback_reason": str(stats.get("fallback_reason", "")),
    }


def event_prompt_core_fields(event: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "prompt_core_filter_enabled": bool(event.get("prompt_core_filter_enabled", False)),
        "prompt_core_min_source_mask_distance_px": float(event.get("prompt_core_min_source_mask_distance_px", 0.0)),
        "prompt_target_stability_filter_enabled": bool(event.get("prompt_target_stability_filter_enabled", False)),
        "prompt_target_stability_depth_radius_px": int(event.get("prompt_target_stability_depth_radius_px", 0)),
        "prompt_target_stability_max_local_depth_range_m": float(
            event.get("prompt_target_stability_max_local_depth_range_m", 0.0)
        ),
        "prompt_target_stability_max_depth_abs_error": float(
            event.get("prompt_target_stability_max_depth_abs_error", 0.0)
        ),
        "prompt_target_stability_min_depth_conf": float(event.get("prompt_target_stability_min_depth_conf", 0.0)),
        "attempt_prompt_record_count_before_core_filter": int(
            event.get("attempt_prompt_record_count_before_core_filter", 0)
        ),
        "attempt_prompt_record_count_after_core_filter": int(
            event.get("attempt_prompt_record_count_after_core_filter", 0)
        ),
        "attempt_positive_prompt_count_before_core_filter": int(
            event.get("attempt_positive_prompt_count_before_core_filter", 0)
        ),
        "attempt_positive_prompt_count_after_core_filter": int(
            event.get("attempt_positive_prompt_count_after_core_filter", 0)
        ),
        "attempt_negative_prompt_count_before_core_filter": int(
            event.get("attempt_negative_prompt_count_before_core_filter", 0)
        ),
        "attempt_negative_prompt_count_after_core_filter": int(
            event.get("attempt_negative_prompt_count_after_core_filter", 0)
        ),
        "attempt_prompt_core_dropped_record_count": int(event.get("attempt_prompt_core_dropped_record_count", 0)),
        "attempt_prompt_core_min_retained_source_mask_distance_px": float(
            event.get("attempt_prompt_core_min_retained_source_mask_distance_px", -1.0)
        ),
        "attempt_prompt_core_min_dropped_source_mask_distance_px": float(
            event.get("attempt_prompt_core_min_dropped_source_mask_distance_px", -1.0)
        ),
        "confirm_prompt_record_count_before_core_filter": int(
            event.get("confirm_prompt_record_count_before_core_filter", 0)
        ),
        "confirm_prompt_record_count_after_core_filter": int(
            event.get("confirm_prompt_record_count_after_core_filter", 0)
        ),
        "confirm_positive_prompt_count_before_core_filter": int(
            event.get("confirm_positive_prompt_count_before_core_filter", 0)
        ),
        "confirm_positive_prompt_count_after_core_filter": int(
            event.get("confirm_positive_prompt_count_after_core_filter", 0)
        ),
        "confirm_negative_prompt_count_before_core_filter": int(
            event.get("confirm_negative_prompt_count_before_core_filter", 0)
        ),
        "confirm_negative_prompt_count_after_core_filter": int(
            event.get("confirm_negative_prompt_count_after_core_filter", 0)
        ),
        "confirm_prompt_core_dropped_record_count": int(event.get("confirm_prompt_core_dropped_record_count", 0)),
        "confirm_prompt_core_min_retained_source_mask_distance_px": float(
            event.get("confirm_prompt_core_min_retained_source_mask_distance_px", -1.0)
        ),
        "confirm_prompt_core_min_dropped_source_mask_distance_px": float(
            event.get("confirm_prompt_core_min_dropped_source_mask_distance_px", -1.0)
        ),
        "attempt_prompt_target_stability_record_count_before": int(
            event.get("attempt_prompt_target_stability_record_count_before", 0)
        ),
        "attempt_prompt_target_stability_record_count_after": int(
            event.get("attempt_prompt_target_stability_record_count_after", 0)
        ),
        "attempt_positive_prompt_count_before_target_stability": int(
            event.get("attempt_positive_prompt_count_before_target_stability", 0)
        ),
        "attempt_positive_prompt_count_after_target_stability": int(
            event.get("attempt_positive_prompt_count_after_target_stability", 0)
        ),
        "attempt_negative_prompt_count_before_target_stability": int(
            event.get("attempt_negative_prompt_count_before_target_stability", 0)
        ),
        "attempt_negative_prompt_count_after_target_stability": int(
            event.get("attempt_negative_prompt_count_after_target_stability", 0)
        ),
        "attempt_prompt_target_stability_dropped_record_count": int(
            event.get("attempt_prompt_target_stability_dropped_record_count", 0)
        ),
        "attempt_prompt_target_stability_drop_reasons": dict(
            event.get("attempt_prompt_target_stability_drop_reasons", {})
        ),
        "attempt_prompt_target_stability_max_retained_local_depth_range_m": float(
            event.get("attempt_prompt_target_stability_max_retained_local_depth_range_m", -1.0)
        ),
        "attempt_prompt_target_stability_min_dropped_local_depth_range_m": float(
            event.get("attempt_prompt_target_stability_min_dropped_local_depth_range_m", -1.0)
        ),
        "confirm_prompt_target_stability_record_count_before": int(
            event.get("confirm_prompt_target_stability_record_count_before", 0)
        ),
        "confirm_prompt_target_stability_record_count_after": int(
            event.get("confirm_prompt_target_stability_record_count_after", 0)
        ),
        "confirm_positive_prompt_count_before_target_stability": int(
            event.get("confirm_positive_prompt_count_before_target_stability", 0)
        ),
        "confirm_positive_prompt_count_after_target_stability": int(
            event.get("confirm_positive_prompt_count_after_target_stability", 0)
        ),
        "confirm_negative_prompt_count_before_target_stability": int(
            event.get("confirm_negative_prompt_count_before_target_stability", 0)
        ),
        "confirm_negative_prompt_count_after_target_stability": int(
            event.get("confirm_negative_prompt_count_after_target_stability", 0)
        ),
        "confirm_prompt_target_stability_dropped_record_count": int(
            event.get("confirm_prompt_target_stability_dropped_record_count", 0)
        ),
        "confirm_prompt_target_stability_drop_reasons": dict(
            event.get("confirm_prompt_target_stability_drop_reasons", {})
        ),
        "confirm_prompt_target_stability_max_retained_local_depth_range_m": float(
            event.get("confirm_prompt_target_stability_max_retained_local_depth_range_m", -1.0)
        ),
        "confirm_prompt_target_stability_min_dropped_local_depth_range_m": float(
            event.get("confirm_prompt_target_stability_min_dropped_local_depth_range_m", -1.0)
        ),
    }
    for key, value in event.items():
        if (
            "prompt_source_core_supplement" in str(key)
            or "source_core_supplement" in str(key)
            or "prompt_source_core_negative_supplement" in str(key)
            or "source_core_negative_supplement" in str(key)
            or "prompt_anchor_conflict" in str(key)
            or "anchor_conflict" in str(key)
        ):
            fields[str(key)] = value
    return fields


def build_schedule(
    *,
    events: list[dict[str, Any]],
    frame_ids: list[int],
    prompt_root: Path,
    prompt_summary: dict[str, Any],
    reference_records: dict[int, dict[str, Any]],
    lingbot_hw: tuple[int, int],
    points_by_case: dict[tuple[int, int, int], list[dict[str, Any]]],
    visual_event_indices: set[int],
    geometry_prompts_enabled: bool = True,
    prompt_points_enabled: bool | None = None,
    random_geometry_prompts_enabled: bool = False,
    prompt_core_min_source_mask_distance_px: float = 0.0,
    prompt_source_core_supplement_config: dict[str, Any] | None = None,
    prompt_source_core_negative_supplement_config: dict[str, Any] | None = None,
    prompt_source_core_supplement_geometry: dict[str, np.ndarray] | None = None,
    prompt_target_stability_config: dict[str, Any] | None = None,
    prompt_anchor_conflict_config: dict[str, Any] | None = None,
    target_depth_by_frame_id: dict[int, np.ndarray] | None = None,
    target_depth_conf_by_frame_id: dict[int, np.ndarray] | None = None,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    frame_to_idx = {int(frame_id): int(idx) for idx, frame_id in enumerate(frame_ids)}
    schedule: dict[int, dict[str, list[dict[str, Any]]]] = {}
    selected_pose = str(prompt_summary.get("selected_pose_mode", "direct_as_c2w"))
    if prompt_points_enabled is None:
        prompt_points_enabled = bool(geometry_prompts_enabled)
    prompt_source_core_supplement_config = dict(prompt_source_core_supplement_config or {})
    prompt_source_core_negative_supplement_config = dict(prompt_source_core_negative_supplement_config or {})
    prompt_target_stability_config = dict(prompt_target_stability_config or {})
    prompt_anchor_conflict_config = dict(prompt_anchor_conflict_config or {})
    target_depth_by_frame_id = dict(target_depth_by_frame_id or {})
    target_depth_conf_by_frame_id = dict(target_depth_conf_by_frame_id or {})
    target_stability_enabled = prompt_target_stability_enabled(prompt_target_stability_config)
    for event in events:
        event_index = int(event["event_index"])
        target_obj_id = int(event["global_id"])
        source_frame_id = int(event["source_frame_id"])
        attempt_frame_id = int(event["attempt_frame_id"])
        confirm_frame_id = int(event["confirm_frame_id"])
        for frame_id in (source_frame_id, attempt_frame_id, confirm_frame_id):
            if frame_id not in frame_to_idx:
                raise RuntimeError(f"event {event_index} frame {frame_id} not in rolling frame ids")
        attempt_label = event_frame_label(reference_records, attempt_frame_id)
        confirm_label = event_frame_label(reference_records, confirm_frame_id)
        source_label = event_frame_label(reference_records, source_frame_id)
        if prompt_points_enabled:
            attempt_points = [
                row
                for row in points_for_event(
                    points_by_case,
                    source_frame_index=None,
                    source_frame_id=source_frame_id,
                    target_frame_id=attempt_frame_id,
                    target_obj_id=target_obj_id,
                )
                if str(row.get("pose_mode", selected_pose)) == selected_pose
            ]
            confirm_points = [
                row
                for row in points_for_event(
                    points_by_case,
                    source_frame_index=None,
                    source_frame_id=source_frame_id,
                    target_frame_id=confirm_frame_id,
                    target_obj_id=target_obj_id,
                )
                if str(row.get("pose_mode", selected_pose)) == selected_pose
            ]
            if not attempt_points:
                raise RuntimeError(f"event {event_index} has no attempt prompt points")
            if not confirm_points:
                raise RuntimeError(f"event {event_index} has no confirm prompt points")
            attempt_points, attempt_supplement_stats = supplement_prompt_records_from_source_core(
                attempt_points,
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                target_obj_id=target_obj_id,
                source_frame_id=source_frame_id,
                target_frame_id=attempt_frame_id,
                pose_mode=selected_pose,
                geometry=prompt_source_core_supplement_geometry,
                config=prompt_source_core_supplement_config,
                seed=int(event_index) * 100003 + int(attempt_frame_id) * 9176 + int(target_obj_id),
            )
            confirm_points, confirm_supplement_stats = supplement_prompt_records_from_source_core(
                confirm_points,
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                target_obj_id=target_obj_id,
                source_frame_id=source_frame_id,
                target_frame_id=confirm_frame_id,
                pose_mode=selected_pose,
                geometry=prompt_source_core_supplement_geometry,
                config=prompt_source_core_supplement_config,
                seed=int(event_index) * 100003 + int(confirm_frame_id) * 9176 + int(target_obj_id),
            )
            attempt_points, attempt_negative_supplement_stats = (
                supplement_negative_prompt_records_from_coview_source_core(
                    attempt_points,
                    source_label=source_label,
                    lingbot_hw=lingbot_hw,
                    target_obj_id=target_obj_id,
                    source_frame_id=source_frame_id,
                    target_frame_id=attempt_frame_id,
                    pose_mode=selected_pose,
                    geometry=prompt_source_core_supplement_geometry,
                    config=prompt_source_core_negative_supplement_config,
                    seed=int(event_index) * 110017 + int(attempt_frame_id) * 9176 + int(target_obj_id),
                )
            )
            confirm_points, confirm_negative_supplement_stats = (
                supplement_negative_prompt_records_from_coview_source_core(
                    confirm_points,
                    source_label=source_label,
                    lingbot_hw=lingbot_hw,
                    target_obj_id=target_obj_id,
                    source_frame_id=source_frame_id,
                    target_frame_id=confirm_frame_id,
                    pose_mode=selected_pose,
                    geometry=prompt_source_core_supplement_geometry,
                    config=prompt_source_core_negative_supplement_config,
                    seed=int(event_index) * 110017 + int(confirm_frame_id) * 9176 + int(target_obj_id),
                )
            )
            attempt_points, attempt_core_stats = filter_prompt_records_by_source_mask_core(
                attempt_points,
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                min_source_mask_distance_px=float(prompt_core_min_source_mask_distance_px),
            )
            confirm_points, confirm_core_stats = filter_prompt_records_by_source_mask_core(
                confirm_points,
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                min_source_mask_distance_px=float(prompt_core_min_source_mask_distance_px),
            )
            attempt_points, attempt_target_stats = filter_prompt_records_by_target_depth_stability(
                attempt_points,
                depth_by_frame_id=target_depth_by_frame_id,
                depth_conf_by_frame_id=target_depth_conf_by_frame_id,
                config=prompt_target_stability_config,
            )
            confirm_points, confirm_target_stats = filter_prompt_records_by_target_depth_stability(
                confirm_points,
                depth_by_frame_id=target_depth_by_frame_id,
                depth_conf_by_frame_id=target_depth_conf_by_frame_id,
                config=prompt_target_stability_config,
            )
            attempt_points, attempt_anchor_conflict_stats = filter_prompt_records_by_anchor_conflict(
                attempt_points,
                config=prompt_anchor_conflict_config,
            )
            confirm_points, confirm_anchor_conflict_stats = filter_prompt_records_by_anchor_conflict(
                confirm_points,
                config=prompt_anchor_conflict_config,
            )
            attempt_pos_coords, attempt_pos_labels, _ = point_arrays(
                attempt_points,
                lingbot_hw=lingbot_hw,
                orig_hw=attempt_label.shape[:2],
                include_negative=False,
            )
            attempt_all_coords, attempt_all_labels, attempt_neg_ids = point_arrays(
                attempt_points,
                lingbot_hw=lingbot_hw,
                orig_hw=attempt_label.shape[:2],
                include_negative=True,
            )
            confirm_pos_coords, confirm_pos_labels, _ = point_arrays(
                confirm_points,
                lingbot_hw=lingbot_hw,
                orig_hw=confirm_label.shape[:2],
                include_negative=False,
            )
            confirm_all_coords, confirm_all_labels, confirm_neg_ids = point_arrays(
                confirm_points,
                lingbot_hw=lingbot_hw,
                orig_hw=confirm_label.shape[:2],
                include_negative=True,
            )
            if random_geometry_prompts_enabled:
                attempt_all_coords, attempt_all_labels = randomize_prompt_set_geometry(
                    coords=attempt_all_coords,
                    labels=attempt_all_labels,
                    shape=attempt_label.shape[:2],
                    seed=int(event_index) * 130003 + int(attempt_frame_id) * 9176 + int(target_obj_id),
                )
                confirm_all_coords, confirm_all_labels = randomize_prompt_set_geometry(
                    coords=confirm_all_coords,
                    labels=confirm_all_labels,
                    shape=confirm_label.shape[:2],
                    seed=int(event_index) * 130003 + int(confirm_frame_id) * 9176 + int(target_obj_id),
                )
                attempt_pos_coords, attempt_pos_labels = split_positive_prompt_coords(
                    attempt_all_coords,
                    attempt_all_labels,
                )
                confirm_pos_coords, confirm_pos_labels = split_positive_prompt_coords(
                    confirm_all_coords,
                    confirm_all_labels,
                )
            if attempt_pos_coords.size == 0 or confirm_pos_coords.size == 0:
                raise RuntimeError(
                    f"event {event_index} missing positive prompt coords after prompt filters; "
                    f"source_core_threshold_px={float(prompt_core_min_source_mask_distance_px)}; "
                    f"target_stability_enabled={bool(target_stability_enabled)}"
                )
        else:
            attempt_pos_coords = np.zeros((0, 2), dtype=np.float32)
            attempt_pos_labels = np.zeros((0,), dtype=np.int32)
            attempt_all_coords = np.zeros((0, 2), dtype=np.float32)
            attempt_all_labels = np.zeros((0,), dtype=np.int32)
            attempt_neg_ids: list[int] = []
            confirm_pos_coords = np.zeros((0, 2), dtype=np.float32)
            confirm_pos_labels = np.zeros((0,), dtype=np.int32)
            confirm_all_coords = np.zeros((0, 2), dtype=np.float32)
            confirm_all_labels = np.zeros((0,), dtype=np.int32)
            confirm_neg_ids: list[int] = []
            attempt_core_stats = filter_prompt_records_by_source_mask_core(
                [],
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                min_source_mask_distance_px=float(prompt_core_min_source_mask_distance_px),
            )[1]
            confirm_core_stats = filter_prompt_records_by_source_mask_core(
                [],
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                min_source_mask_distance_px=float(prompt_core_min_source_mask_distance_px),
            )[1]
            attempt_target_stats = filter_prompt_records_by_target_depth_stability(
                [],
                depth_by_frame_id=target_depth_by_frame_id,
                depth_conf_by_frame_id=target_depth_conf_by_frame_id,
                config=prompt_target_stability_config,
            )[1]
            confirm_target_stats = filter_prompt_records_by_target_depth_stability(
                [],
                depth_by_frame_id=target_depth_by_frame_id,
                depth_conf_by_frame_id=target_depth_conf_by_frame_id,
                config=prompt_target_stability_config,
            )[1]
            attempt_anchor_conflict_stats = filter_prompt_records_by_anchor_conflict(
                [],
                config=prompt_anchor_conflict_config,
            )[1]
            confirm_anchor_conflict_stats = filter_prompt_records_by_anchor_conflict(
                [],
                config=prompt_anchor_conflict_config,
            )[1]
            attempt_supplement_stats = supplement_prompt_records_from_source_core(
                [],
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                target_obj_id=target_obj_id,
                source_frame_id=source_frame_id,
                target_frame_id=attempt_frame_id,
                pose_mode=selected_pose,
                geometry=prompt_source_core_supplement_geometry,
                config=prompt_source_core_supplement_config,
                seed=int(event_index) * 100003 + int(attempt_frame_id) * 9176 + int(target_obj_id),
            )[1]
            confirm_supplement_stats = supplement_prompt_records_from_source_core(
                [],
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                target_obj_id=target_obj_id,
                source_frame_id=source_frame_id,
                target_frame_id=confirm_frame_id,
                pose_mode=selected_pose,
                geometry=prompt_source_core_supplement_geometry,
                config=prompt_source_core_supplement_config,
                seed=int(event_index) * 100003 + int(confirm_frame_id) * 9176 + int(target_obj_id),
            )[1]
            attempt_negative_supplement_stats = supplement_negative_prompt_records_from_coview_source_core(
                [],
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                target_obj_id=target_obj_id,
                source_frame_id=source_frame_id,
                target_frame_id=attempt_frame_id,
                pose_mode=selected_pose,
                geometry=prompt_source_core_supplement_geometry,
                config=prompt_source_core_negative_supplement_config,
                seed=int(event_index) * 110017 + int(attempt_frame_id) * 9176 + int(target_obj_id),
            )[1]
            confirm_negative_supplement_stats = supplement_negative_prompt_records_from_coview_source_core(
                [],
                source_label=source_label,
                lingbot_hw=lingbot_hw,
                target_obj_id=target_obj_id,
                source_frame_id=source_frame_id,
                target_frame_id=confirm_frame_id,
                pose_mode=selected_pose,
                geometry=prompt_source_core_supplement_geometry,
                config=prompt_source_core_negative_supplement_config,
                seed=int(event_index) * 110017 + int(confirm_frame_id) * 9176 + int(target_obj_id),
            )[1]
        item = {
            "event_index": event_index,
            "global_id": target_obj_id,
            "reference_global_id": target_obj_id,
            "live_obj_id": None,
            "source_label": source_label,
            "source_lag": int(event["source_lag"]),
            "source_frame_id": source_frame_id,
            "attempt_frame_id": attempt_frame_id,
            "confirm_frame_id": confirm_frame_id,
            "target_source_area_px": event_int(event, "target_source_area_px"),
            "target_attempt_area_px": event_int(event, "target_attempt_area_px"),
            "target_confirm_area_px": event_int(event, "target_confirm_area_px"),
            "positive_point_count": event_int(event, "positive_point_count"),
            "negative_point_count": event_int(event, "negative_point_count"),
            "confirm_positive_point_count": event_int(event, "confirm_positive_point_count"),
            "confirm_negative_point_count": event_int(event, "confirm_negative_point_count"),
            "source_idx": frame_to_idx[source_frame_id],
            "attempt_idx": frame_to_idx[attempt_frame_id],
            "confirm_idx": frame_to_idx[confirm_frame_id],
            "attempt_pos_coords": attempt_pos_coords,
            "attempt_pos_labels": attempt_pos_labels,
            "attempt_all_coords": attempt_all_coords,
            "attempt_all_labels": attempt_all_labels,
            "attempt_neg_ids": sorted(set(int(v) for v in attempt_neg_ids)),
            "confirm_pos_coords": confirm_pos_coords,
            "confirm_pos_labels": confirm_pos_labels,
            "confirm_all_coords": confirm_all_coords,
            "confirm_all_labels": confirm_all_labels,
            "confirm_neg_ids": sorted(set(int(v) for v in confirm_neg_ids)),
            "attempt_label": attempt_label,
            "confirm_label": confirm_label,
            "write_visuals": event_index in visual_event_indices,
            "selected_variant": "",
            "source_live_area_px": 0,
            "shadow_by_idx": {},
            "geometry_prompts_enabled": bool(geometry_prompts_enabled),
            "lingbot_prompt_points_available": bool(prompt_points_enabled),
            "random_geometry_prompts_enabled": bool(random_geometry_prompts_enabled),
            "prompt_core_filter_enabled": bool(float(prompt_core_min_source_mask_distance_px) > 0.0),
            "prompt_core_min_source_mask_distance_px": float(prompt_core_min_source_mask_distance_px),
            "prompt_target_stability_filter_enabled": bool(target_stability_enabled),
            "prompt_target_stability_depth_radius_px": int(prompt_target_stability_config.get("depth_radius_px", 0)),
            "prompt_target_stability_max_local_depth_range_m": float(
                prompt_target_stability_config.get("max_local_depth_range_m", 0.0)
            ),
            "prompt_target_stability_max_depth_abs_error": float(
                prompt_target_stability_config.get("max_depth_abs_error", 0.0)
            ),
            "prompt_target_stability_min_depth_conf": float(prompt_target_stability_config.get("min_depth_conf", 0.0)),
            **flatten_prompt_core_stats("attempt", attempt_core_stats),
            **flatten_prompt_core_stats("confirm", confirm_core_stats),
            **flatten_prompt_source_core_supplement_stats("attempt", attempt_supplement_stats),
            **flatten_prompt_source_core_supplement_stats("confirm", confirm_supplement_stats),
            **flatten_prompt_source_core_negative_supplement_stats("attempt", attempt_negative_supplement_stats),
            **flatten_prompt_source_core_negative_supplement_stats("confirm", confirm_negative_supplement_stats),
            **flatten_prompt_target_stability_stats("attempt", attempt_target_stats),
            **flatten_prompt_target_stability_stats("confirm", confirm_target_stats),
            **flatten_prompt_anchor_conflict_stats("attempt", attempt_anchor_conflict_stats),
            **flatten_prompt_anchor_conflict_stats("confirm", confirm_anchor_conflict_stats),
        }
        if geometry_prompts_enabled:
            for shadow_frame_id in frame_ids:
                if not (int(source_frame_id) < int(shadow_frame_id) < int(attempt_frame_id)):
                    continue
                shadow_points = [
                    row
                    for row in points_for_event(
                        points_by_case,
                        source_frame_index=None,
                        source_frame_id=source_frame_id,
                        target_frame_id=int(shadow_frame_id),
                        target_obj_id=target_obj_id,
                    )
                    if str(row.get("pose_mode", selected_pose)) == selected_pose
                ]
                if not shadow_points:
                    continue
                shadow_points, shadow_supplement_stats = supplement_prompt_records_from_source_core(
                    shadow_points,
                    source_label=source_label,
                    lingbot_hw=lingbot_hw,
                    target_obj_id=target_obj_id,
                    source_frame_id=source_frame_id,
                    target_frame_id=int(shadow_frame_id),
                    pose_mode=selected_pose,
                    geometry=prompt_source_core_supplement_geometry,
                    config=prompt_source_core_supplement_config,
                    seed=int(event_index) * 100003 + int(shadow_frame_id) * 9176 + int(target_obj_id),
                )
                shadow_points, shadow_negative_supplement_stats = (
                    supplement_negative_prompt_records_from_coview_source_core(
                        shadow_points,
                        source_label=source_label,
                        lingbot_hw=lingbot_hw,
                        target_obj_id=target_obj_id,
                        source_frame_id=source_frame_id,
                        target_frame_id=int(shadow_frame_id),
                        pose_mode=selected_pose,
                        geometry=prompt_source_core_supplement_geometry,
                        config=prompt_source_core_negative_supplement_config,
                        seed=int(event_index) * 110017 + int(shadow_frame_id) * 9176 + int(target_obj_id),
                    )
                )
                shadow_points, shadow_core_stats = filter_prompt_records_by_source_mask_core(
                    shadow_points,
                    source_label=source_label,
                    lingbot_hw=lingbot_hw,
                    min_source_mask_distance_px=float(prompt_core_min_source_mask_distance_px),
                )
                shadow_points, shadow_target_stats = filter_prompt_records_by_target_depth_stability(
                    shadow_points,
                    depth_by_frame_id=target_depth_by_frame_id,
                    depth_conf_by_frame_id=target_depth_conf_by_frame_id,
                    config=prompt_target_stability_config,
                )
                shadow_points, shadow_anchor_conflict_stats = filter_prompt_records_by_anchor_conflict(
                    shadow_points,
                    config=prompt_anchor_conflict_config,
                )
                shadow_label = event_frame_label(reference_records, int(shadow_frame_id))
                shadow_pos_coords, shadow_pos_labels, _ = point_arrays(
                    shadow_points,
                    lingbot_hw=lingbot_hw,
                    orig_hw=shadow_label.shape[:2],
                    include_negative=False,
                )
                shadow_all_coords, shadow_all_labels, shadow_neg_ids = point_arrays(
                    shadow_points,
                    lingbot_hw=lingbot_hw,
                    orig_hw=shadow_label.shape[:2],
                    include_negative=True,
                )
                if shadow_pos_coords.size == 0:
                    continue
                shadow_payload = {
                    "frame_id": int(shadow_frame_id),
                    "frame_idx": int(frame_to_idx[int(shadow_frame_id)]),
                    "pos_coords": shadow_pos_coords,
                    "pos_labels": shadow_pos_labels,
                    "all_coords": shadow_all_coords,
                    "all_labels": shadow_all_labels,
                    "neg_ids": sorted(set(int(v) for v in shadow_neg_ids)),
                    "label": shadow_label,
                    "prompt_core_filter_enabled": bool(float(prompt_core_min_source_mask_distance_px) > 0.0),
                    "prompt_core_min_source_mask_distance_px": float(prompt_core_min_source_mask_distance_px),
                    "prompt_target_stability_filter_enabled": bool(target_stability_enabled),
                    **flatten_prompt_core_stats("shadow", shadow_core_stats),
                    **flatten_prompt_source_core_supplement_stats("shadow", shadow_supplement_stats),
                    **flatten_prompt_source_core_negative_supplement_stats(
                        "shadow", shadow_negative_supplement_stats
                    ),
                    **flatten_prompt_target_stability_stats("shadow", shadow_target_stats),
                    **flatten_prompt_anchor_conflict_stats("shadow", shadow_anchor_conflict_stats),
                }
                item["shadow_by_idx"][int(frame_to_idx[int(shadow_frame_id)])] = shadow_payload
                schedule.setdefault(frame_to_idx[int(shadow_frame_id)], {}).setdefault("shadow", []).append(item)
        schedule.setdefault(frame_to_idx[source_frame_id], {}).setdefault("demote", []).append(item)
        schedule.setdefault(frame_to_idx[attempt_frame_id], {}).setdefault("attempt", []).append(item)
        schedule.setdefault(frame_to_idx[confirm_frame_id], {}).setdefault("confirm", []).append(item)
    return schedule


def add_points(
    predictor: Any,
    state: dict[str, Any],
    *,
    frame_idx: int,
    obj_id: int,
    coords: np.ndarray,
    labels: np.ndarray,
    probe_args: SimpleNamespace,
) -> tuple[list[int], np.ndarray, dict[str, Any], float]:
    started = time.time()
    ids, masks, state_record = add_points_after_tracking_started(
        predictor,
        state,
        frame_idx=int(frame_idx),
        obj_id=int(obj_id),
        point_coords=coords,
        point_labels=labels,
        args=probe_args,
    )
    with torch.inference_mode(), probe_autocast_for(probe_args):
        probe_reconsolidate_stream_state_outputs(predictor, state)
    return ids, masks, state_record, float(time.time() - started)


def select_mask_by_sam_score(masks: Any, scores: Any, shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    mask_arr = np.asarray(masks)
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]
    score_arr = np.asarray(scores if scores is not None else np.zeros((mask_arr.shape[0],), dtype=np.float32)).reshape(-1)
    if mask_arr.size == 0:
        return np.zeros(shape, dtype=bool), {"candidate_index": -1, "sam2_score": 0.0, "candidate_count": 0}
    best_idx = int(np.argmax(score_arr)) if score_arr.size else 0
    best_idx = max(0, min(best_idx, int(mask_arr.shape[0]) - 1))
    score = float(score_arr[best_idx]) if best_idx < int(score_arr.size) else 0.0
    return (np.squeeze(mask_arr[best_idx]) > 0).astype(bool), {
        "candidate_index": int(best_idx),
        "sam2_score": score,
        "candidate_count": int(mask_arr.shape[0]),
    }


def merge_object_mask(
    ids: np.ndarray | list[int],
    masks: np.ndarray,
    *,
    obj_id: int,
    mask: np.ndarray,
    prefer_new: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    ids_i = [int(v) for v in list(ids)]
    masks_i = np.asarray(masks).astype(bool)
    keep_indices = [idx for idx, value in enumerate(ids_i) if int(value) != int(obj_id)]
    kept_masks = masks_i[keep_indices] if masks_i.size and keep_indices else np.zeros((0, *mask.shape), dtype=bool)
    kept_ids = [ids_i[idx] for idx in keep_indices]
    new_mask = np.asarray(mask, dtype=bool)[None, ...]
    if prefer_new:
        merged_ids = np.asarray([int(obj_id), *kept_ids], dtype=np.int64)
        merged_masks = np.concatenate([new_mask, kept_masks], axis=0)
    else:
        merged_ids = np.asarray([*kept_ids, int(obj_id)], dtype=np.int64)
        merged_masks = np.concatenate([kept_masks, new_mask], axis=0)
    return merged_ids, merged_masks


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(jsonable(row))


def run() -> int:
    started = time.time()
    cli = parse_args()
    if str(cli.gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from tools.audit_v106_sam2_rolling_state import get_rolling_stats, load_config, make_args, run as run_rolling

    output_root = as_path(str(cli.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    config_path = as_path(str(cli.config))
    config = load_config(config_path)
    prompt_root = as_path(str(cli.prompt_probe_root))
    probe_root = as_path(str(cli.probe_root))
    reference_root = as_path(str(cli.reference_run_root))
    scene_root = as_path(str(cli.scene_root))
    events_arg = str(cli.events).strip()
    recoverability_enabled = str(cli.recoverability_mode) != "disabled"
    reactivation_prompt_mode = str(cli.reactivation_prompt_mode)
    appearance_only_enabled = reactivation_prompt_mode == "appearance_only"
    appearance_geometry_filter_enabled = reactivation_prompt_mode == "appearance_geometry_filter"
    random_geometry_prompts_enabled = reactivation_prompt_mode == "random_geometry"
    appearance_selector_enabled = bool(appearance_only_enabled or appearance_geometry_filter_enabled)
    geometry_prompts_enabled = reactivation_prompt_mode in {"lingbot_geometry", "random_geometry"}
    prompt_points_enabled = bool(geometry_prompts_enabled or appearance_geometry_filter_enabled)
    output_plane_enabled = not bool(cli.disable_output_plane)
    visual_event_indices = {int(v) for v in str(cli.visual_events).split(",") if str(v).strip()}
    shadow_visual_event_indices = parse_int_set(str(cli.shadow_visual_events))
    shadow_visual_frame_ids = parse_int_set(str(cli.shadow_visual_frame_ids))
    probation_visual_event_indices = parse_int_set(str(cli.probation_visual_events))
    frame_ids = parse_frame_ids(str(cli.frame_ids), int(cli.frame_start), int(cli.frame_stride), int(cli.frame_count))
    auto_source_lags = parse_int_set(str(cli.auto_source_lags))

    prompt_summary = load_prompt_summary(prompt_root)
    points_path = selected_prompt_file(
        prompt_root,
        prompt_summary,
        "selected_visible_point_records",
        "prompt_capsule_visible_point_records_direct_as_c2w.json",
    )
    points_by_case = load_points(points_path) if prompt_points_enabled else {}
    lingbot_hw = infer_lingbot_hw(prompt_root, prompt_summary)
    prompt_source_core_supplement_config = {
        "positive_points": int(cli.prompt_source_core_supplement_positive_points),
        "trigger_max_positive_points": int(cli.prompt_source_core_supplement_trigger_max_positive_points),
        "min_source_distance_px": float(cli.prompt_source_core_supplement_min_distance_px),
        "depth_abs_tolerance": float(cli.prompt_source_core_supplement_depth_abs_tolerance),
        "depth_rel_tolerance": float(cli.prompt_source_core_supplement_depth_rel_tolerance),
        "min_depth_conf": float(cli.prompt_source_core_supplement_min_depth_conf),
        "duplicate_radius_px": float(cli.prompt_source_core_supplement_duplicate_radius_px),
    }
    source_core_supplement_enabled = prompt_source_core_supplement_enabled(prompt_source_core_supplement_config)
    prompt_source_core_negative_supplement_config = {
        "negative_points": int(cli.prompt_source_core_supplement_negative_points),
        "trigger_max_negative_points": int(cli.prompt_source_core_supplement_negative_trigger_max_negative_points),
        "min_source_distance_px": float(cli.prompt_source_core_supplement_negative_min_distance_px),
        "max_neighbor_bbox_distance_px": float(
            cli.prompt_source_core_supplement_negative_max_neighbor_bbox_distance_px
        ),
        "target_border_margin_px": float(cli.prompt_source_core_supplement_negative_target_border_margin_px),
        "min_object_area_px": int(cli.prompt_source_core_supplement_negative_min_area_px),
        "max_objects": int(cli.prompt_source_core_supplement_negative_max_objects),
        "depth_abs_tolerance": float(cli.prompt_source_core_supplement_depth_abs_tolerance),
        "depth_rel_tolerance": float(cli.prompt_source_core_supplement_depth_rel_tolerance),
        "min_depth_conf": float(cli.prompt_source_core_supplement_min_depth_conf),
        "duplicate_radius_px": float(cli.prompt_source_core_supplement_duplicate_radius_px),
    }
    source_core_negative_supplement_enabled = prompt_source_core_negative_supplement_enabled(
        prompt_source_core_negative_supplement_config
    )
    prompt_target_stability_config = {
        "depth_radius_px": int(cli.prompt_target_stability_depth_radius_px),
        "max_local_depth_range_m": float(cli.prompt_target_stability_max_local_depth_range_m),
        "max_depth_abs_error": float(cli.prompt_target_stability_max_depth_abs_error),
        "min_depth_conf": float(cli.prompt_target_stability_min_depth_conf),
        "min_valid_depth_count": int(cli.prompt_target_stability_min_valid_depth_count),
    }
    prompt_anchor_conflict_config = {
        "negative_radius_px": float(cli.prompt_anchor_conflict_negative_radius_px),
        "positive_cluster_radius_px": float(cli.prompt_anchor_conflict_positive_cluster_radius_px),
        "min_positive_points": int(cli.prompt_anchor_conflict_min_positive_points),
    }
    target_stability_enabled = prompt_target_stability_enabled(prompt_target_stability_config)
    anchor_conflict_enabled = prompt_anchor_conflict_enabled(prompt_anchor_conflict_config)
    target_depth_by_frame_id: dict[int, np.ndarray] = {}
    target_depth_conf_by_frame_id: dict[int, np.ndarray] = {}
    target_depth_npz_path: Path | None = None
    source_core_supplement_geometry: dict[str, np.ndarray] | None = None
    source_core_supplement_npz_path: Path | None = None
    if target_stability_enabled:
        target_depth_by_frame_id, target_depth_conf_by_frame_id, target_depth_npz_path = load_prompt_target_depth_maps(
            prompt_root,
            prompt_summary,
        )
    if source_core_supplement_enabled or source_core_negative_supplement_enabled:
        source_core_supplement_geometry, source_core_supplement_npz_path = load_prompt_raw_geometry(
            prompt_root,
            prompt_summary,
        )
    reference_records = load_reference_records(reference_root)
    event_selection_rows: list[dict[str, Any]] = []
    event_selection_summary: dict[str, Any] = {"mode": "manual"}
    if events_arg.lower() == "auto":
        all_events = load_event_rows(probe_root, None)
        events, event_selection_rows, event_selection_summary = choose_auto_events(
            all_events,
            frame_ids=frame_ids,
            source_lags=auto_source_lags,
            min_target_source_area=int(cli.auto_min_target_source_area),
            min_positive_points=int(cli.auto_min_positive_points),
            min_confirm_positive_points=int(cli.auto_min_confirm_positive_points),
            max_events=int(cli.auto_max_events),
            max_events_per_object=int(cli.auto_max_events_per_object),
            selection_policy=str(cli.auto_selection_policy),
            seed=int(cli.seed),
        )
        event_indices = {event_int(row, "event_index") for row in events}
    else:
        event_indices = {int(v) for v in events_arg.split(",") if str(v).strip()}
        events = load_event_rows(probe_root, event_indices)
        event_selection_rows = [
            {
                "event_index": event_int(row, "event_index"),
                "global_id": event_int(row, "global_id"),
                "source_lag": event_int(row, "source_lag"),
                "source_frame_id": event_int(row, "source_frame_id"),
                "attempt_frame_id": event_int(row, "attempt_frame_id"),
                "confirm_frame_id": event_int(row, "confirm_frame_id"),
                "target_source_area_px": event_int(row, "target_source_area_px"),
                "positive_point_count": event_int(row, "positive_point_count"),
                "confirm_positive_point_count": event_int(row, "confirm_positive_point_count"),
                "selected": True,
                "skip_reason": "",
            }
            for row in events
        ]
        event_selection_summary = {
            "mode": "manual",
            "candidate_count": int(len(events)),
            "selected_count": int(len(events)),
            "selected_event_indices": sorted(event_indices),
            "skipped_count": 0,
            "skip_reasons": {},
        }
    event_selection_json = output_root / "event_selection_records.json"
    event_selection_csv = output_root / "event_selection_records.csv"
    write_json(
        event_selection_json,
        {
            "schema_version": "stream4d_v107_phase8_event_selection_records_v1",
            "events_arg": events_arg,
            "summary": event_selection_summary,
            "rows": event_selection_rows,
        },
    )
    write_rows_csv(event_selection_csv, event_selection_rows)
    schedule = build_schedule(
        events=events,
        frame_ids=frame_ids,
        prompt_root=prompt_root,
        prompt_summary=prompt_summary,
        reference_records=reference_records,
        lingbot_hw=lingbot_hw,
        points_by_case=points_by_case,
        visual_event_indices=visual_event_indices,
        geometry_prompts_enabled=geometry_prompts_enabled,
        prompt_points_enabled=prompt_points_enabled,
        random_geometry_prompts_enabled=random_geometry_prompts_enabled,
        prompt_core_min_source_mask_distance_px=float(cli.prompt_core_min_source_mask_distance_px),
        prompt_source_core_supplement_config=prompt_source_core_supplement_config,
        prompt_source_core_negative_supplement_config=prompt_source_core_negative_supplement_config,
        prompt_source_core_supplement_geometry=source_core_supplement_geometry,
        prompt_target_stability_config=prompt_target_stability_config,
        prompt_anchor_conflict_config=prompt_anchor_conflict_config,
        target_depth_by_frame_id=target_depth_by_frame_id,
        target_depth_conf_by_frame_id=target_depth_conf_by_frame_id,
    )

    probe_args = SimpleNamespace(
        device="cuda" if str(cli.gpu).strip() or torch.cuda.is_available() else "cpu",
        model_dtype=model_dtype_for_probe(str(cli.model_dtype)),
        visual_pad=int(cli.visual_pad),
        visual_scale=int(cli.visual_scale),
    )
    image_args = SimpleNamespace(
        sam2_checkpoint=str(as_path(str(cli.sam2_checkpoint))),
        sam2_model_cfg=str(cli.sam2_model_cfg),
        device=str(probe_args.device),
        model_dtype=str(probe_args.model_dtype),
    )
    adapter_records: list[dict[str, Any]] = []
    visual_paths: list[Path] = []
    shadow_visual_paths: list[Path] = []
    probation_visual_paths: list[Path] = []
    demoted_source_frames: set[int] = set()
    source_mapping_done_frames: set[int] = set()
    shadow_predictor: Any | None = None
    shadow_checkpoint: Path | None = None
    shadow_current_frame_id: int | None = None
    shadow_build_runtime_sec = 0.0
    shadow_set_image_runtime_sec = 0.0
    shadow_predict_runtime_sec = 0.0
    prompt_new_object_ids_by_ref: dict[int, int] = {}

    original_base_infer = base.infer_stream_frame
    original_base_add_masks = base.add_masks_to_stream_state
    original_cv2_imwrite_for_v107 = cv2.imwrite

    def ensure_shadow_image(frame_id: int) -> Any:
        nonlocal shadow_predictor, shadow_checkpoint, shadow_current_frame_id
        nonlocal shadow_build_runtime_sec, shadow_set_image_runtime_sec
        if shadow_predictor is None:
            build_t0 = time.time()
            shadow_predictor, shadow_checkpoint = build_sam2_predictor(image_args)
            shadow_build_runtime_sec += float(time.time() - build_t0)
        if shadow_current_frame_id != int(frame_id):
            set_t0 = time.time()
            rgb = rgb_frame(scene_root, int(frame_id))
            with torch.inference_mode(), torch.autocast(**image_autocast_kwargs(image_args)):
                shadow_predictor.set_image(rgb)
            shadow_set_image_runtime_sec += float(time.time() - set_t0)
            shadow_current_frame_id = int(frame_id)
        return shadow_predictor

    def predict_image_prompt_mask(
        *,
        frame_id: int,
        shape: tuple[int, int],
        coords: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any], float]:
        nonlocal shadow_predict_runtime_sec
        if coords.size == 0:
            return np.zeros(shape, dtype=bool), {
                "candidate_index": -1,
                "sam2_score": 0.0,
                "candidate_count": 0,
            }, 0.0
        predictor = ensure_shadow_image(int(frame_id))
        pred_t0 = time.time()
        with torch.inference_mode(), torch.autocast(**image_autocast_kwargs(image_args)):
            masks, scores, _logits = predictor.predict(
                point_coords=np.asarray(coords, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=True,
            )
        runtime = float(time.time() - pred_t0)
        shadow_predict_runtime_sec += runtime
        mask, select_record = select_mask_by_sam_score(masks, scores, shape)
        return mask, select_record, runtime

    def target_mask_core_filter_prompt_set(
        *,
        frame_id: int,
        shape: tuple[int, int],
        coords: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        candidate_mask, candidate_select, candidate_sec = predict_image_prompt_mask(
            frame_id=int(frame_id),
            shape=shape,
            coords=coords,
            labels=labels,
        )
        filtered_coords, filtered_labels, stats = filter_prompt_coords_by_target_mask_core(
            coords,
            labels,
            mask=candidate_mask,
            min_distance_px=float(cli.prompt_target_mask_core_min_distance_px),
            min_positive_points=int(cli.prompt_target_mask_core_min_positive_points),
        )
        stats["candidate_sam2_score"] = float(candidate_select.get("sam2_score", 0.0))
        stats["candidate_count"] = int(candidate_select.get("candidate_count", 0))
        stats["candidate_runtime_sec"] = float(candidate_sec)
        return filtered_coords, filtered_labels, stats

    def image_g3_g2_eval_decision(
        *,
        output_mode: str,
        g1_negative_conflict_rate: float,
        negative_prompt_count: int,
    ) -> dict[str, Any]:
        policy = str(cli.image_g3_selector_g2_eval_policy)
        threshold = float(cli.online_select_neg_conflict_threshold)
        if str(output_mode) != "image_g3_selector":
            return {
                "policy": policy,
                "evaluate": False,
                "reason": "output_mode_not_image_g3_selector",
                "threshold": threshold,
            }
        if int(negative_prompt_count) <= 0:
            return {
                "policy": policy,
                "evaluate": False,
                "reason": "no_retained_negative_prompts",
                "threshold": threshold,
            }
        if policy == "always_if_negatives":
            return {
                "policy": policy,
                "evaluate": True,
                "reason": "retained_negative_prompts_present",
                "threshold": threshold,
            }
        evaluate = bool(float(g1_negative_conflict_rate) > threshold)
        return {
            "policy": policy,
            "evaluate": evaluate,
            "reason": "g1_negative_conflict_above_threshold"
            if evaluate
            else "g1_negative_conflict_at_or_below_threshold",
            "threshold": threshold,
        }

    def image_g3_g2_conflict_selection(
        *,
        g1_negative_conflict_rate: float,
        g2_negative_conflict_rate: float,
    ) -> dict[str, Any]:
        policy = str(cli.image_g3_selector_g2_select_policy)
        threshold = float(cli.online_select_neg_conflict_threshold)
        min_improvement = max(0.0, float(cli.image_g3_selector_g2_min_neg_conflict_improvement))
        g1_conflict = float(g1_negative_conflict_rate)
        g2_conflict = float(g2_negative_conflict_rate)
        improvement = float(g1_conflict - g2_conflict)
        g1_above_threshold = bool(g1_conflict > threshold)
        if policy == "not_worse" or (policy == "strict_improvement_unless_g1_conflict" and g1_above_threshold):
            conflict_ok = bool(g2_conflict <= g1_conflict)
            reason = (
                "g2_negative_conflict_not_worse"
                if conflict_ok
                else "g2_negative_conflict_worse_than_g1"
            )
        else:
            conflict_ok = bool(improvement > min_improvement)
            reason = (
                "g2_negative_conflict_strictly_improved"
                if conflict_ok
                else "g2_negative_conflict_not_improved_enough"
            )
        return {
            "policy": policy,
            "conflict_ok": conflict_ok,
            "reason": reason,
            "g1_above_threshold": g1_above_threshold,
            "threshold": threshold,
            "min_neg_conflict_improvement": min_improvement,
            "neg_conflict_improvement": improvement,
        }

    def decode_image_prompt_mask(
        event: dict[str, Any],
        payload: dict[str, Any],
        *,
        frame_idx: int,
        frame_id: int,
        record_type: str,
        variant_prefix: str,
        output_mode: str,
        min_positive_support: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        shape = payload["label"].shape[:2]
        initial_g1_mask, initial_g1_select, initial_g1_sec = predict_image_prompt_mask(
            frame_id=int(frame_id),
            shape=shape,
            coords=payload["pos_coords"],
            labels=payload["pos_labels"],
        )
        eval_all_coords, eval_all_labels, target_mask_core_stats = filter_prompt_coords_by_target_mask_core(
            payload["all_coords"],
            payload["all_labels"],
            mask=initial_g1_mask,
            min_distance_px=float(cli.prompt_target_mask_core_min_distance_px),
            min_positive_points=int(cli.prompt_target_mask_core_min_positive_points),
        )
        target_mask_core_stats["candidate_sam2_score"] = float(initial_g1_select.get("sam2_score", 0.0))
        target_mask_core_stats["candidate_count"] = int(initial_g1_select.get("candidate_count", 0))
        target_mask_core_stats["candidate_runtime_sec"] = float(initial_g1_sec)
        eval_pos_coords, eval_pos_labels = split_positive_prompt_coords(eval_all_coords, eval_all_labels)
        if bool(target_mask_core_stats.get("filter_applied", False)):
            g1_mask, g1_select, g1_sec = predict_image_prompt_mask(
                frame_id=int(frame_id),
                shape=shape,
                coords=eval_pos_coords,
                labels=eval_pos_labels,
            )
            g1_sec = float(initial_g1_sec + g1_sec)
        else:
            g1_mask, g1_select, g1_sec = initial_g1_mask, initial_g1_select, initial_g1_sec
        payload["visual_coords"] = eval_all_coords
        payload["visual_labels"] = eval_all_labels
        g1_rates = prompt_point_rates(g1_mask, eval_all_coords, eval_all_labels)
        selected_mask = g1_mask
        selected_variant = f"{variant_prefix}_G1_pos"
        selected_select = g1_select
        selected_runtime = g1_sec
        g2_rates: dict[str, Any] = {}
        g2_select: dict[str, Any] = {}
        g2_sec = 0.0
        g2_mask = np.zeros_like(g1_mask, dtype=bool)
        negative_prompt_count = int(np.count_nonzero(eval_all_labels == 0))
        g2_eval = image_g3_g2_eval_decision(
            output_mode=str(output_mode),
            g1_negative_conflict_rate=float(g1_rates["candidate_negative_point_conflict_rate"]),
            negative_prompt_count=negative_prompt_count,
        )
        g2_choose_reason = "not_evaluated"
        if bool(g2_eval["evaluate"]):
            g2_mask, g2_select, g2_sec = predict_image_prompt_mask(
                frame_id=int(frame_id),
                shape=shape,
                coords=eval_all_coords,
                labels=eval_all_labels,
            )
            g2_rates = prompt_point_rates(g2_mask, eval_all_coords, eval_all_labels)
            g2_positive_support_ok = bool(
                float(g2_rates["positive_point_support_rate"]) >= float(min_positive_support)
            )
            g2_conflict_selection = image_g3_g2_conflict_selection(
                g1_negative_conflict_rate=float(g1_rates["candidate_negative_point_conflict_rate"]),
                g2_negative_conflict_rate=float(g2_rates["candidate_negative_point_conflict_rate"]),
            )
            choose_g2 = bool(g2_positive_support_ok and bool(g2_conflict_selection["conflict_ok"]))
            if not g2_positive_support_ok:
                g2_choose_reason = "g2_positive_support_below_min"
            elif not bool(g2_conflict_selection["conflict_ok"]):
                g2_choose_reason = str(g2_conflict_selection["reason"])
            else:
                g2_choose_reason = f"g2_positive_support_ok_and_{g2_conflict_selection['reason']}"
            if choose_g2:
                selected_mask = g2_mask
                selected_variant = f"{variant_prefix}_G2_pos_neg"
                selected_select = g2_select
                selected_runtime = g2_sec
        selected_rates = prompt_point_rates(selected_mask, eval_all_coords, eval_all_labels)
        metrics = mask_metrics(
            selected_mask,
            payload["label"] == int(event["reference_global_id"]),
            payload["label"],
            set(int(v) for v in payload["neg_ids"]),
        )
        support_ok = float(selected_rates["positive_point_support_rate"]) >= float(min_positive_support)
        if record_type == "probation_attempt":
            metric_prefix = "probation"
        elif record_type == "shadow_output":
            metric_prefix = "shadow"
        else:
            metric_prefix = "confirm"
        record = {
            "record_type": record_type,
            "event_index": int(event["event_index"]),
            "source_lag": int(event["source_lag"]),
            "frame_idx": int(frame_idx),
            "frame_id": int(frame_id),
            "reference_global_id": int(event["reference_global_id"]),
            "live_obj_id": int(event["live_obj_id"]),
            "image_prompt_output_mode": str(output_mode),
            "selected_variant": selected_variant,
            "output_mask": bool(support_ok),
            "skip_reason": "" if support_ok else f"below_{variant_prefix}_min_positive_support",
            "online_selection_uses_reference_iou": False,
            "source_live_area_px": int(event.get("source_live_area_px", 0)),
            "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
            "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
            "prompt_only_unmapped_source_reactivation": bool(
                event.get("prompt_only_unmapped_source_reactivation", False)
            ),
            "prompt_only_new_object_id": int(event.get("prompt_only_new_object_id", -1)),
            "positive_prompt_count": int(np.count_nonzero(eval_all_labels == 1)),
            "negative_prompt_count": int(np.count_nonzero(eval_all_labels == 0)),
            "positive_point_support_rate": float(selected_rates["positive_point_support_rate"]),
            "candidate_negative_point_conflict_rate": float(selected_rates["candidate_negative_point_conflict_rate"]),
            "g1_positive_point_support_rate": float(g1_rates["positive_point_support_rate"]),
            "g1_candidate_negative_point_conflict_rate": float(g1_rates["candidate_negative_point_conflict_rate"]),
            "g1_sam2_score": float(g1_select.get("sam2_score", 0.0)),
            "g2_eval_policy": str(g2_eval["policy"]),
            "g2_evaluated": bool(g2_eval["evaluate"]),
            "g2_eval_reason": str(g2_eval["reason"]),
            "g2_eval_neg_conflict_threshold": float(g2_eval["threshold"]),
            "g2_chosen": bool(selected_variant.endswith("G2_pos_neg")),
            "g2_choose_reason": str(g2_choose_reason),
            "g2_select_policy": str(cli.image_g3_selector_g2_select_policy),
            "g2_min_neg_conflict_improvement": float(
                cli.image_g3_selector_g2_min_neg_conflict_improvement
            ),
            "g2_positive_point_support_rate": float(g2_rates.get("positive_point_support_rate", -1.0)),
            "g2_candidate_negative_point_conflict_rate": float(g2_rates.get("candidate_negative_point_conflict_rate", -1.0)),
            "g2_sam2_score": float(g2_select.get("sam2_score", -1.0)),
            "selected_sam2_score": float(selected_select.get("sam2_score", 0.0)),
            "runtime_sec": float(selected_runtime),
            **event_prompt_core_fields(event),
            **flatten_prompt_target_mask_core_stats(metric_prefix, target_mask_core_stats),
            **{f"{variant_prefix}_{key}": value for key, value in metrics.items()},
        }
        return (selected_mask if support_ok else np.zeros_like(selected_mask, dtype=bool)), record

    def decode_shadow_mask(
        event: dict[str, Any],
        payload: dict[str, Any],
        *,
        frame_idx: int,
        frame_id: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        mask, record = decode_image_prompt_mask(
            event,
            payload,
            frame_idx=frame_idx,
            frame_id=frame_id,
            record_type="shadow_output",
            variant_prefix="shadow",
            output_mode=str(cli.shadow_output_mode),
            min_positive_support=float(cli.shadow_min_positive_support),
        )
        record["shadow_output_mode"] = str(cli.shadow_output_mode)
        return mask, record

    def select_appearance_only_mask(
        event: dict[str, Any],
        *,
        frame_idx: int,
        frame_id: int,
        record_type: str,
        label: np.ndarray,
        obj_ids: np.ndarray,
        masks: np.ndarray,
        use_lingbot_point_filter: bool = False,
        all_coords: np.ndarray | None = None,
        all_labels: np.ndarray | None = None,
        neg_ids: list[int] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        source_desc = event.get("appearance_source_descriptor")
        ref_id = int(event["reference_global_id"])
        obj_id = int(event["live_obj_id"])
        base_ids = [int(v) for v in np.asarray(obj_ids).tolist()]
        base_masks = np.asarray(masks).astype(bool)
        coords_arr = (
            np.asarray(all_coords, dtype=np.float32)
            if all_coords is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
        labels_arr = (
            np.asarray(all_labels, dtype=np.int32)
            if all_labels is not None
            else np.zeros((0,), dtype=np.int32)
        )
        neg_id_set = set(int(v) for v in (neg_ids or []))
        pos_prompt_count = int(np.count_nonzero(labels_arr == 1))
        neg_prompt_count = int(np.count_nonzero(labels_arr == 0))
        selector_variant = "appearance_geometry_filter" if use_lingbot_point_filter else "appearance_only_rgb_shape"
        if not isinstance(source_desc, dict) or int(source_desc.get("area_px", 0)) <= 0:
            return np.zeros(label.shape[:2], dtype=bool), {
                "record_type": record_type,
                "event_index": int(event["event_index"]),
                "source_lag": int(event["source_lag"]),
                "frame_idx": int(frame_idx),
                "frame_id": int(frame_id),
                "reference_global_id": ref_id,
                "live_obj_id": obj_id,
                "selected_variant": "SKIPPED_APPEARANCE_NO_SOURCE_DESCRIPTOR",
                "skip_reason": "appearance_no_source_descriptor",
                "output_mask": False,
                "appearance_only_control": bool(not use_lingbot_point_filter),
                "appearance_geometry_filter_control": bool(use_lingbot_point_filter),
                "geometry_prompts_enabled": False,
                "uses_lingbot_prompt_points_for_reactivation": bool(use_lingbot_point_filter),
                "sam2_add_new_points_or_box_called": False,
                "online_selection_uses_reference_iou": False,
                **event_prompt_core_fields(event),
            }
        if use_lingbot_point_filter and pos_prompt_count <= 0:
            return np.zeros(label.shape[:2], dtype=bool), {
                "record_type": record_type,
                "event_index": int(event["event_index"]),
                "source_lag": int(event["source_lag"]),
                "frame_idx": int(frame_idx),
                "frame_id": int(frame_id),
                "reference_global_id": ref_id,
                "live_obj_id": obj_id,
                "selected_variant": "SKIPPED_APPEARANCE_GEOMETRY_NO_POSITIVE_POINTS",
                "skip_reason": "appearance_geometry_no_positive_prompt_points",
                "output_mask": False,
                "appearance_only_control": False,
                "appearance_geometry_filter_control": True,
                "geometry_prompts_enabled": False,
                "uses_lingbot_prompt_points_for_reactivation": True,
                "sam2_add_new_points_or_box_called": False,
                "online_selection_uses_reference_iou": False,
                "appearance_geometry_positive_prompt_count": int(pos_prompt_count),
                "appearance_geometry_negative_prompt_count": int(neg_prompt_count),
                **event_prompt_core_fields(event),
            }
        rgb = rgb_frame(scene_root, int(frame_id))
        candidates: list[dict[str, Any]] = []
        for cand_id, cand_mask in zip(base_ids, base_masks, strict=False):
            area = int(np.count_nonzero(cand_mask))
            if area <= 0:
                continue
            desc = mask_descriptor_rgb_shape(rgb, cand_mask)
            score = appearance_descriptor_score(
                desc,
                source_desc,
                color_scale=float(cli.appearance_only_color_scale),
            )
            point_rates = (
                prompt_point_rates(cand_mask, coords_arr, labels_arr)
                if use_lingbot_point_filter
                else {
                    "positive_point_support_rate": 0.0,
                    "candidate_negative_point_conflict_rate": 0.0,
                    "positive_point_count_for_online_rate": 0,
                    "candidate_negative_point_count_for_online_rate": 0,
                }
            )
            pos_support = float(point_rates.get("positive_point_support_rate", 0.0))
            neg_conflict = float(point_rates.get("candidate_negative_point_conflict_rate", 0.0))
            geometry_filter_ok = bool(
                (not use_lingbot_point_filter)
                or (
                    pos_support >= float(cli.appearance_geometry_min_positive_support)
                    and neg_conflict <= float(cli.appearance_geometry_max_negative_conflict)
                )
            )
            appearance_weight = float(cli.appearance_geometry_appearance_weight)
            positive_weight = float(cli.appearance_geometry_positive_weight)
            negative_weight = max(0.0, 1.0 - appearance_weight - positive_weight)
            combined_score = float(
                appearance_weight * float(score["appearance_score"])
                + positive_weight * pos_support
                + negative_weight * (1.0 - neg_conflict)
            )
            candidates.append(
                {
                    "candidate_live_obj_id": int(cand_id),
                    "candidate_area_px": int(area),
                    "candidate_descriptor": desc,
                    "appearance_geometry_positive_support_rate": pos_support,
                    "appearance_geometry_negative_conflict_rate": neg_conflict,
                    "appearance_geometry_filter_ok": bool(geometry_filter_ok),
                    "appearance_geometry_combined_score": combined_score,
                    "appearance_geometry_positive_point_count_for_online_rate": int(
                        point_rates.get("positive_point_count_for_online_rate", 0)
                    ),
                    "appearance_geometry_negative_point_count_for_online_rate": int(
                        point_rates.get("candidate_negative_point_count_for_online_rate", 0)
                    ),
                    **score,
                }
            )
        ranking_key = "appearance_geometry_combined_score" if use_lingbot_point_filter else "appearance_score"
        candidates.sort(key=lambda item: float(item[ranking_key]), reverse=True)
        if use_lingbot_point_filter:
            valid_candidates = [item for item in candidates if bool(item.get("appearance_geometry_filter_ok", False))]
            ranked_candidates = valid_candidates if valid_candidates else candidates
        else:
            valid_candidates = candidates
            ranked_candidates = candidates
        if not candidates:
            selected = {
                "candidate_live_obj_id": -1,
                "appearance_score": 0.0,
                "appearance_color_score": 0.0,
                "appearance_color_l2": 0.0,
                "appearance_shape_score": 0.0,
                "appearance_geometry_combined_score": 0.0,
                "appearance_geometry_positive_support_rate": 0.0,
                "appearance_geometry_negative_conflict_rate": 0.0,
                "appearance_geometry_filter_ok": False,
            }
            selected_mask = np.zeros(label.shape[:2], dtype=bool)
            second_score = -1.0
        else:
            selected = ranked_candidates[0]
            selected_mask = base_masks[base_ids.index(int(selected["candidate_live_obj_id"]))]
            second_score = float(ranked_candidates[1][ranking_key]) if len(ranked_candidates) > 1 else -1.0
        selected_rank_score = float(selected.get(ranking_key, selected.get("appearance_score", 0.0)))
        margin = float(selected_rank_score - second_score)
        output_ok = bool(
            candidates
            and float(selected["appearance_score"]) >= float(cli.appearance_only_min_score)
            and margin >= float(cli.appearance_only_min_margin)
            and (not use_lingbot_point_filter or bool(selected.get("appearance_geometry_filter_ok", False)))
        )
        metrics = mask_metrics(
            selected_mask if output_ok else np.zeros_like(selected_mask, dtype=bool),
            label == ref_id,
            label,
            neg_id_set if use_lingbot_point_filter else set(),
        )
        skip_reasons: list[str] = []
        if not output_ok:
            if not candidates:
                skip_reasons.append("appearance_no_candidates")
            if float(selected.get("appearance_score", 0.0)) < float(cli.appearance_only_min_score):
                skip_reasons.append("appearance_score_below_threshold")
            if margin < float(cli.appearance_only_min_margin):
                skip_reasons.append("appearance_margin_below_threshold")
            if use_lingbot_point_filter and not bool(selected.get("appearance_geometry_filter_ok", False)):
                skip_reasons.append("appearance_geometry_support_or_conflict_failed")
        metric_prefix = "probation" if str(record_type) == "probation_attempt" else "confirm"
        record = {
            "record_type": record_type,
            "event_index": int(event["event_index"]),
            "source_lag": int(event["source_lag"]),
            "frame_idx": int(frame_idx),
            "frame_id": int(frame_id),
            "reference_global_id": ref_id,
            "live_obj_id": obj_id,
            "source_live_area_px": int(event.get("source_live_area_px", 0)),
            "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
            "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
            "prompt_only_unmapped_source_reactivation": False,
            "prompt_only_new_object_id": -1,
            "selected_variant": selector_variant,
            "output_mask": bool(output_ok),
            "skip_reason": "" if output_ok else ";".join(skip_reasons) or "appearance_score_below_threshold_or_margin",
            "target_present": bool(output_ok and int(np.count_nonzero(selected_mask)) > 0),
            "appearance_only_control": bool(not use_lingbot_point_filter),
            "appearance_geometry_filter_control": bool(use_lingbot_point_filter),
            "appearance_candidate_live_obj_id": int(selected.get("candidate_live_obj_id", -1)),
            "appearance_candidate_count": int(len(candidates)),
            "appearance_geometry_valid_candidate_count": int(len(valid_candidates)),
            "appearance_top2_score": float(second_score),
            "appearance_margin": float(margin),
            "appearance_ranking_score_key": ranking_key,
            "appearance_selected_ranking_score": float(selected_rank_score),
            "appearance_min_score": float(cli.appearance_only_min_score),
            "appearance_min_margin": float(cli.appearance_only_min_margin),
            "appearance_only_color_scale": float(cli.appearance_only_color_scale),
            "appearance_source_area_px": int(source_desc.get("area_px", 0)),
            "geometry_prompts_enabled": False,
            "uses_lingbot_prompt_points_for_reactivation": bool(use_lingbot_point_filter),
            "uses_lingbot_prompt_points_for_candidate_filter": bool(use_lingbot_point_filter),
            "appearance_geometry_min_positive_support": float(cli.appearance_geometry_min_positive_support),
            "appearance_geometry_max_negative_conflict": float(cli.appearance_geometry_max_negative_conflict),
            "appearance_geometry_positive_prompt_count": int(pos_prompt_count),
            "appearance_geometry_negative_prompt_count": int(neg_prompt_count),
            "sam2_add_new_points_or_box_called": False,
            "online_selection_uses_reference_iou": False,
            "online_gate_uses_reference_iou": False,
            "reactivation_prompt_mode": str(cli.reactivation_prompt_mode),
            "reactivation_committed_to_sam2_video_state": False,
            "top_appearance_candidates": candidates[:5],
            **event_prompt_core_fields(event),
            **{key: value for key, value in selected.items() if key != "candidate_descriptor"},
            **{f"appearance_{key}": value for key, value in metrics.items()},
            **{f"{metric_prefix}_{key}": value for key, value in metrics.items()},
        }
        return (selected_mask if output_ok else np.zeros_like(selected_mask, dtype=bool)), record

    def draw_zoom_overlay_with_real_imwrite(**kwargs: Any) -> None:
        current_imwrite = cv2.imwrite
        cv2.imwrite = original_cv2_imwrite_for_v107
        try:
            draw_zoom_overlay(**kwargs)
        finally:
            cv2.imwrite = current_imwrite

    def demote_rows_for_frame_idx(frame_idx_i: int) -> list[dict[str, Any]]:
        return sorted(schedule.get(int(frame_idx_i), {}).get("demote", []), key=lambda row: int(row["event_index"]))

    def process_source_mappings(
        *,
        source_frame_idx: int,
        source_frame_id: int,
        mapped_at_frame_idx: int,
        mapped_at_frame_id: int,
        source_ids: np.ndarray,
        source_masks: np.ndarray,
        mask_source: str,
    ) -> None:
        nonlocal source_mapping_done_frames
        source_frame_idx_i = int(source_frame_idx)
        demote_rows_for_frame = demote_rows_for_frame_idx(source_frame_idx_i)
        if not demote_rows_for_frame or source_frame_idx_i in source_mapping_done_frames:
            return
        source_ids_i = [int(v) for v in np.asarray(source_ids).tolist()]
        source_masks_b = np.asarray(source_masks).astype(bool)
        source_rgb = rgb_frame(scene_root, int(source_frame_id)) if appearance_selector_enabled else None
        mapping_candidates: list[tuple[float, int, dict[str, Any]]] = []
        for event in demote_rows_for_frame:
            ref_id = int(event["reference_global_id"])
            ref_mask = np.asarray(event["source_label"] == ref_id)
            if appearance_selector_enabled and source_rgb is not None:
                event["appearance_source_descriptor"] = mask_descriptor_rgb_shape(source_rgb, ref_mask)
            candidates = []
            source_anchor_stats_by_live_id: dict[int, dict[str, Any]] = {}
            for live_id, live_mask in zip(source_ids_i, source_masks_b, strict=False):
                source_anchor_stats_by_live_id[int(live_id)] = mask_physical_anchor_stats(live_mask)
                candidates.append((mask_iou(live_mask, ref_mask), int(live_id), int(np.count_nonzero(live_mask))))
            candidates.sort(reverse=True)
            best_iou, best_live_id, best_live_area = candidates[0] if candidates else (0.0, -1, 0)
            event["source_mapping_iou"] = float(best_iou)
            event["source_mapping_candidate_live_id"] = int(best_live_id)
            event["source_mapping_candidate_live_area_px"] = int(best_live_area)
            event["source_anchor_stats"] = source_anchor_stats_by_live_id.get(
                int(best_live_id),
                mask_physical_anchor_stats(np.zeros(ref_mask.shape[:2], dtype=bool)),
            )
            event["source_mapping_found"] = False
            event["source_mapping_skip_reason"] = ""
            event["long_term_memory_admitted"] = False
            event["long_term_admission_skip_reason"] = ""
            if best_iou >= float(cli.min_source_mapping_iou):
                event["source_mapping_found"] = True
                mapping_candidates.append((float(best_iou), int(best_live_id), event))
            else:
                event["live_obj_id"] = None
                event["source_mapping_skip_reason"] = "below_min_source_mapping_iou"
        claimed: dict[int, int] = {}
        for _best_iou, best_live_id, event in sorted(mapping_candidates, key=lambda item: item[0], reverse=True):
            ref_id = int(event["reference_global_id"])
            claimed_ref_id = claimed.get(int(best_live_id))
            if claimed_ref_id is not None and int(claimed_ref_id) != ref_id:
                event["live_obj_id"] = None
                event["source_mapping_skip_reason"] = "duplicate_live_obj_mapping_lower_iou"
                continue
            claimed[int(best_live_id)] = ref_id
            event["live_obj_id"] = int(best_live_id)
            event["source_live_area_px"] = int(event.get("source_mapping_candidate_live_area_px", 0))
        long_term_candidates = [
            event for _by_iou, _best_live_id, event in mapping_candidates if event.get("live_obj_id") is not None
        ]
        long_term_order = sorted(
            long_term_candidates,
            key=lambda row: (-int(row.get("source_live_area_px", 0)), int(row["event_index"])),
        )
        long_term_budget_ids = (
            {int(row["event_index"]) for row in long_term_order[: int(cli.long_term_max_events)]}
            if int(cli.long_term_max_events) > 0
            else {int(row["event_index"]) for row in long_term_order}
        )
        for event in long_term_candidates:
            admission_reasons: list[str] = []
            readiness = physical_anchor_readiness(event, cli)
            event["physical_anchor_readiness"] = readiness
            if int(event.get("source_live_area_px", 0)) < int(cli.long_term_min_source_area):
                admission_reasons.append("source_area_below_long_term_min_source_area")
            if event_int(event, "positive_point_count") < int(cli.long_term_min_positive_points):
                admission_reasons.append("positive_prompt_count_below_long_term_min")
            if event_int(event, "confirm_positive_point_count") < int(cli.long_term_min_confirm_positive_points):
                admission_reasons.append("confirm_positive_prompt_count_below_long_term_min")
            if int(event["event_index"]) not in long_term_budget_ids:
                admission_reasons.append("long_term_budget_exhausted")
            admission_reasons.extend(physical_anchor_skip_reasons(dict(event.get("source_anchor_stats", {})), cli))
            if not bool(readiness["physical_anchor_ready"]):
                admission_reasons.extend(list(readiness["physical_anchor_readiness_reasons"]))
            if admission_reasons:
                event["long_term_admission_skip_reason"] = ";".join(admission_reasons)
                event["source_mapping_skip_reason"] = event["long_term_admission_skip_reason"]
                event["live_obj_id"] = None
                event["long_term_memory_admitted"] = False
            else:
                event["long_term_memory_admitted"] = True
        alignment = (
            "declared_source_frame"
            if int(source_frame_idx) == int(mapped_at_frame_idx) and int(source_frame_id) == int(mapped_at_frame_id)
            else "delayed_available_frame"
        )
        for event in demote_rows_for_frame:
            ref_id = int(event["reference_global_id"])
            adapter_records.append(
                {
                    "record_type": "source_identity_mapping",
                    "frame_idx": int(source_frame_idx),
                    "frame_id": int(source_frame_id),
                    "source_frame_idx": int(source_frame_idx),
                    "source_frame_id": int(source_frame_id),
                    "mapped_at_frame_idx": int(mapped_at_frame_idx),
                    "mapped_at_frame_id": int(mapped_at_frame_id),
                    "source_mapping_mask_source": str(mask_source),
                    "source_mapping_frame_alignment": alignment,
                    "event_index": int(event["event_index"]),
                    "reference_global_id": ref_id,
                    "live_obj_id": event["live_obj_id"],
                    "source_mapping_candidate_live_id": int(event["source_mapping_candidate_live_id"]),
                    "source_mapping_iou": float(event["source_mapping_iou"]),
                    "source_mapping_candidate_live_area_px": int(event.get("source_mapping_candidate_live_area_px", 0)),
                    "source_live_area_px": int(event.get("source_live_area_px", 0)),
                    **{f"source_{key}": value for key, value in dict(event.get("source_anchor_stats", {})).items()},
                    "target_source_area_px": event_int(event, "target_source_area_px"),
                    "target_attempt_area_px": event_int(event, "target_attempt_area_px"),
                    "target_confirm_area_px": event_int(event, "target_confirm_area_px"),
                    "positive_point_count": event_int(event, "positive_point_count"),
                    "negative_point_count": event_int(event, "negative_point_count"),
                    "confirm_positive_point_count": event_int(event, "confirm_positive_point_count"),
                    "confirm_negative_point_count": event_int(event, "confirm_negative_point_count"),
                    "min_source_mapping_iou": float(cli.min_source_mapping_iou),
                    "source_mapping_found": bool(event.get("source_mapping_found", False)),
                    "source_mapping_accepted": event.get("live_obj_id") is not None,
                    "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                    "long_term_min_source_area": int(cli.long_term_min_source_area),
                    "long_term_min_positive_points": int(cli.long_term_min_positive_points),
                    "long_term_min_confirm_positive_points": int(cli.long_term_min_confirm_positive_points),
                    "long_term_max_events": int(cli.long_term_max_events),
                    "long_term_anchor_max_area_frac": float(cli.long_term_anchor_max_area_frac),
                    "long_term_anchor_max_bbox_frac": float(cli.long_term_anchor_max_bbox_frac),
                    "long_term_anchor_max_edge_touch_count": int(cli.long_term_anchor_max_edge_touch_count),
                    "long_term_anchor_min_extent": float(cli.long_term_anchor_min_extent),
                    "long_term_anchor_min_core_area_px": int(cli.long_term_anchor_min_core_area_px),
                    **dict(event.get("physical_anchor_readiness", physical_anchor_readiness(event, cli))),
                    "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                    "source_mapping_skip_reason": str(event.get("source_mapping_skip_reason", "")),
                    "candidate_live_obj_count": int(len(source_ids_i)),
                }
            )
        source_mapping_done_frames.add(source_frame_idx_i)

    def process_source_demotions(
        predictor: Any,
        state: dict[str, Any],
        *,
        source_frame_idx: int,
        source_frame_id: int,
        demoted_at_frame_idx: int,
        demoted_at_frame_id: int,
        demotion_source: str,
    ) -> None:
        nonlocal demoted_source_frames
        source_frame_idx_i = int(source_frame_idx)
        demote_rows_for_frame = demote_rows_for_frame_idx(source_frame_idx_i)
        if not demote_rows_for_frame or source_frame_idx_i in demoted_source_frames:
            return
        seen: set[int] = set()
        for event in demote_rows_for_frame:
            ref_id = int(event["reference_global_id"])
            obj_id = event.get("live_obj_id")
            if obj_id is None:
                skip_reason = str(
                    event.get("long_term_admission_skip_reason")
                    or event.get("source_mapping_skip_reason")
                    or "no_source_live_id_mapping"
                )
                adapter_records.append(
                    {
                        "record_type": "demotion",
                        "event_index": int(event["event_index"]),
                        "frame_idx": int(source_frame_idx),
                        "frame_id": int(source_frame_id),
                        "source_frame_idx": int(source_frame_idx),
                        "source_frame_id": int(source_frame_id),
                        "demoted_at_frame_idx": int(demoted_at_frame_idx),
                        "demoted_at_frame_id": int(demoted_at_frame_id),
                        "demotion_source": str(demotion_source),
                        "reference_global_id": ref_id,
                        "live_obj_id": None,
                        "source_live_area_px": int(event.get("source_live_area_px", 0)),
                        "removed": False,
                        "skip_reason": skip_reason,
                        "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                        "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                        "state_obj_ids_before": [int(v) for v in state.get("obj_ids", [])],
                        "state_obj_ids_after": [int(v) for v in state.get("obj_ids", [])],
                        "runtime_sec": 0.0,
                    }
                )
                continue
            obj_id = int(obj_id)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            before = [int(v) for v in state.get("obj_ids", [])]
            ok = obj_id in before and hasattr(predictor, "remove_object")
            runtime = 0.0
            if ok:
                t0 = time.time()
                with torch.inference_mode():
                    predictor.remove_object(state, obj_id, strict=False, need_output=False)
                runtime = float(time.time() - t0)
            adapter_records.append(
                {
                    "record_type": "demotion",
                    "event_index": int(event["event_index"]),
                    "frame_idx": int(source_frame_idx),
                    "frame_id": int(source_frame_id),
                    "source_frame_idx": int(source_frame_idx),
                    "source_frame_id": int(source_frame_id),
                    "demoted_at_frame_idx": int(demoted_at_frame_idx),
                    "demoted_at_frame_id": int(demoted_at_frame_id),
                    "demotion_source": str(demotion_source),
                    "reference_global_id": ref_id,
                    "live_obj_id": obj_id,
                    "source_live_area_px": int(event.get("source_live_area_px", 0)),
                    "removed": bool(ok),
                    "skip_reason": "",
                    "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                    "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                    "state_obj_ids_before": before,
                    "state_obj_ids_after": [int(v) for v in state.get("obj_ids", [])],
                    "runtime_sec": runtime,
                }
            )
        demoted_source_frames.add(source_frame_idx_i)

    def maybe_assign_prompt_new_object(
        event: dict[str, Any],
        state: dict[str, Any],
        *,
        frame_idx: int,
        frame_id: int,
        trigger: str,
    ) -> bool:
        nonlocal prompt_new_object_ids_by_ref
        if event.get("live_obj_id") is not None:
            return True
        if not recoverability_enabled:
            event["prompt_new_object_skip_reason"] = "recoverability_disabled"
            return False
        if not geometry_prompts_enabled:
            event["prompt_new_object_skip_reason"] = "geometry_disabled"
            return False
        if str(cli.unmapped_source_policy) != "prompt_new_object":
            return False
        if str(event.get("source_mapping_skip_reason", "")) != "below_min_source_mapping_iou":
            return False
        skip_reasons: list[str] = []
        if event_int(event, "target_source_area_px") < int(cli.long_term_min_source_area):
            skip_reasons.append("source_area_below_prompt_new_object_min_source_area")
        if event_int(event, "positive_point_count") < int(cli.long_term_min_positive_points):
            skip_reasons.append("positive_prompt_count_below_prompt_new_object_min")
        if event_int(event, "confirm_positive_point_count") < int(cli.long_term_min_confirm_positive_points):
            skip_reasons.append("confirm_positive_prompt_count_below_prompt_new_object_min")
        if skip_reasons:
            event["prompt_new_object_skip_reason"] = ";".join(skip_reasons)
            return False
        ref_id = int(event["reference_global_id"])
        if ref_id not in prompt_new_object_ids_by_ref:
            used_ids = {int(v) for v in state.get("obj_ids", [])}
            used_ids.update(int(v) for v in prompt_new_object_ids_by_ref.values())
            prompt_new_object_ids_by_ref[ref_id] = max({59999, *used_ids}) + 1
        obj_id = int(prompt_new_object_ids_by_ref[ref_id])
        event["live_obj_id"] = obj_id
        event["source_live_area_px"] = event_int(event, "target_source_area_px")
        event["prompt_only_unmapped_source_reactivation"] = True
        event["prompt_only_new_object_id"] = obj_id
        event["prompt_new_object_trigger"] = str(trigger)
        adapter_records.append(
            {
                "record_type": "prompt_new_object_assignment",
                "event_index": int(event["event_index"]),
                "source_lag": int(event["source_lag"]),
                "frame_idx": int(frame_idx),
                "frame_id": int(frame_id),
                "source_frame_idx": int(event["source_idx"]),
                "source_frame_id": int(event["source_frame_id"]),
                "reference_global_id": ref_id,
                "live_obj_id": obj_id,
                "prompt_new_object_trigger": str(trigger),
                "source_mapping_skip_reason": str(event.get("source_mapping_skip_reason", "")),
                "target_source_area_px": event_int(event, "target_source_area_px"),
                "positive_point_count": event_int(event, "positive_point_count"),
                "confirm_positive_point_count": event_int(event, "confirm_positive_point_count"),
                "long_term_memory_admitted": False,
                "prompt_only_unmapped_source_reactivation": True,
                "state_obj_ids_at_assignment": [int(v) for v in state.get("obj_ids", [])],
            }
        )
        return True

    def hooked_add_masks_to_stream_state(
        predictor: Any,
        state: dict[str, Any],
        *,
        tracker: str,
        frame_idx: int,
        obj_ids: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        original_base_add_masks(
            predictor,
            state,
            tracker=tracker,
            frame_idx=int(frame_idx),
            obj_ids=obj_ids,
            masks=masks,
        )
        frame_idx_i = int(frame_idx)
        if frame_idx_i != 0:
            return
        if not demote_rows_for_frame_idx(frame_idx_i):
            return
        frame_id = int(frame_ids[frame_idx_i])
        process_source_mappings(
            source_frame_idx=frame_idx_i,
            source_frame_id=frame_id,
            mapped_at_frame_idx=frame_idx_i,
            mapped_at_frame_id=frame_id,
            source_ids=obj_ids,
            source_masks=masks,
            mask_source="stream_initial_add_masks",
        )
        process_source_demotions(
            predictor,
            state,
            source_frame_idx=frame_idx_i,
            source_frame_id=frame_id,
            demoted_at_frame_idx=frame_idx_i,
            demoted_at_frame_id=frame_id,
            demotion_source="stream_initial_add_masks",
        )

    def hooked_infer_stream_frame(predictor: Any, state: dict[str, Any], *, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal demoted_source_frames, source_mapping_done_frames
        frame_idx_i = int(frame_idx)
        frame_id = int(frame_ids[frame_idx_i])
        frame_events = schedule.get(frame_idx_i, {})
        return_ids: np.ndarray | None = None
        return_masks: np.ndarray | None = None

        def ensure_return_base() -> tuple[np.ndarray, np.ndarray]:
            nonlocal return_ids, return_masks
            if return_ids is None or return_masks is None:
                return_ids, return_masks = original_base_infer(predictor, state, frame_idx=frame_idx_i)
            return return_ids, return_masks
        demote_rows_for_frame = sorted(frame_events.get("demote", []), key=lambda row: int(row["event_index"]))
        if demote_rows_for_frame and frame_idx_i not in source_mapping_done_frames:
            source_ids, source_masks = ensure_return_base()
            process_source_mappings(
                source_frame_idx=frame_idx_i,
                source_frame_id=frame_id,
                mapped_at_frame_idx=frame_idx_i,
                mapped_at_frame_id=frame_id,
                source_ids=source_ids,
                source_masks=source_masks,
                mask_source="stream_base_infer",
            )
        if demote_rows_for_frame and frame_idx_i not in demoted_source_frames:
            process_source_demotions(
                predictor,
                state,
                source_frame_idx=frame_idx_i,
                source_frame_id=frame_id,
                demoted_at_frame_idx=frame_idx_i,
                demoted_at_frame_id=frame_id,
                demotion_source="stream_base_infer",
            )

        if recoverability_enabled and str(cli.shadow_output_mode) != "none" and frame_events.get("shadow"):
            emitted_this_frame = 0
            for event in sorted(
                frame_events.get("shadow", []),
                key=lambda row: (-int(row.get("source_live_area_px", 0)), int(row["event_index"])),
            ):
                payload = event.get("shadow_by_idx", {}).get(frame_idx_i)
                ref_id = int(event["reference_global_id"])
                obj_id_raw = event.get("live_obj_id")
                base_record = {
                    "record_type": "shadow_output",
                    "event_index": int(event["event_index"]),
                    "source_lag": int(event["source_lag"]),
                    "frame_idx": frame_idx_i,
                    "frame_id": frame_id,
                    "reference_global_id": ref_id,
                    "live_obj_id": obj_id_raw,
                    "shadow_output_mode": str(cli.shadow_output_mode),
                    "output_mask": False,
                    "online_selection_uses_reference_iou": False,
                    "source_live_area_px": int(event.get("source_live_area_px", 0)),
                    "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                    "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                    "prompt_only_unmapped_source_reactivation": bool(
                        event.get("prompt_only_unmapped_source_reactivation", False)
                    ),
                    "prompt_only_new_object_id": int(event.get("prompt_only_new_object_id", -1)),
                }
                if obj_id_raw is None:
                    maybe_assign_prompt_new_object(
                        event,
                        state,
                        frame_idx=frame_idx_i,
                        frame_id=frame_id,
                        trigger="shadow",
                    )
                    obj_id_raw = event.get("live_obj_id")
                    base_record["live_obj_id"] = obj_id_raw
                    base_record["source_live_area_px"] = int(event.get("source_live_area_px", 0))
                    base_record["prompt_only_unmapped_source_reactivation"] = bool(
                        event.get("prompt_only_unmapped_source_reactivation", False)
                    )
                    base_record["prompt_only_new_object_id"] = int(event.get("prompt_only_new_object_id", -1))
                if obj_id_raw is None:
                    adapter_records.append(
                        {
                            **base_record,
                            "skip_reason": str(
                                event.get("prompt_new_object_skip_reason")
                                or event.get("long_term_admission_skip_reason")
                                or event.get("source_mapping_skip_reason")
                                or "no_source_live_id_mapping"
                            ),
                        }
                    )
                    continue
                obj_id = int(obj_id_raw)
                if payload is None:
                    adapter_records.append({**base_record, "live_obj_id": obj_id, "skip_reason": "no_shadow_prompt_points"})
                    continue
                if int(event.get("source_live_area_px", 0)) < int(cli.shadow_min_source_area):
                    adapter_records.append(
                        {
                            **base_record,
                            "live_obj_id": obj_id,
                            "skip_reason": "source_area_below_shadow_min_source_area",
                            "shadow_min_source_area": int(cli.shadow_min_source_area),
                        }
                    )
                    continue
                if emitted_this_frame >= int(cli.shadow_max_events_per_frame):
                    adapter_records.append(
                        {
                            **base_record,
                            "live_obj_id": obj_id,
                            "skip_reason": "frame_shadow_budget_exhausted",
                            "shadow_max_events_per_frame": int(cli.shadow_max_events_per_frame),
                        }
                    )
                    continue
                ensure_return_base()
                shadow_mask, record = decode_shadow_mask(event, payload, frame_idx=frame_idx_i, frame_id=frame_id)
                if output_plane_enabled and bool(record.get("output_mask")) and int(np.count_nonzero(shadow_mask)) > 0:
                    assert return_ids is not None and return_masks is not None
                    return_ids, return_masks = merge_object_mask(
                        return_ids,
                        return_masks,
                        obj_id=obj_id,
                        mask=shadow_mask,
                        prefer_new=True,
                    )
                    emitted_this_frame += 1
                if (
                    int(event["event_index"]) in shadow_visual_event_indices
                    and (not shadow_visual_frame_ids or frame_id in shadow_visual_frame_ids)
                ):
                    rgb = rgb_frame(scene_root, frame_id)
                    path = (
                        output_root
                        / "highres_shadow_visuals"
                        / f"event{int(event['event_index']):03d}_{record.get('selected_variant', 'shadow')}_f{frame_id}_ref{ref_id}_live{obj_id}.jpg"
                    )
                    draw_zoom_overlay_with_real_imwrite(
                        rgb=rgb,
                        ref_mask=payload["label"] == ref_id,
                        pred_mask=shadow_mask,
                        points=payload.get("visual_coords", payload["all_coords"]),
                        labels=payload.get("visual_labels", payload["all_labels"]),
                        title=(
                            f"shadow event{int(event['event_index']):03d} {record.get('selected_variant')} "
                            f"f{frame_id} ref{ref_id} live{obj_id}"
                        ),
                        output_path=path,
                        pad=int(cli.visual_pad),
                        scale=int(cli.visual_scale),
                        color=(60, 255, 120),
                    )
                    shadow_visual_paths.append(path)
                    record["visual_path"] = rel(path)
                    record["visual_sha256"] = sha256_file(path)
                adapter_records.append(record)

        for event in sorted(frame_events.get("attempt", []), key=lambda row: int(row["event_index"])):
            ref_id = int(event["reference_global_id"])
            if not recoverability_enabled:
                adapter_records.append(
                    {
                        "record_type": "attempt",
                        "event_index": int(event["event_index"]),
                        "source_lag": int(event["source_lag"]),
                        "frame_idx": frame_idx_i,
                        "frame_id": frame_id,
                        "reference_global_id": ref_id,
                        "live_obj_id": event.get("live_obj_id"),
                        "selected_variant": "SKIPPED_RECOVERABILITY_DISABLED",
                        "skip_reason": "recoverability_disabled",
                        "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                        "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                        "prompt_only_unmapped_source_reactivation": False,
                        "prompt_only_new_object_id": -1,
                        "online_gate_uses_reference_iou": False,
                        "recoverability_mode": str(cli.recoverability_mode),
                    }
                )
                continue
            if not geometry_prompts_enabled and not appearance_selector_enabled:
                adapter_records.append(
                    {
                        "record_type": "attempt",
                        "event_index": int(event["event_index"]),
                        "source_lag": int(event["source_lag"]),
                        "frame_idx": frame_idx_i,
                        "frame_id": frame_id,
                        "reference_global_id": ref_id,
                        "live_obj_id": event.get("live_obj_id"),
                        "source_live_area_px": int(event.get("source_live_area_px", 0)),
                        "selected_variant": "SKIPPED_GEOMETRY_DISABLED",
                        "skip_reason": "geometry_disabled",
                        "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                        "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                        "prompt_only_unmapped_source_reactivation": False,
                        "prompt_only_new_object_id": -1,
                        "online_gate_uses_reference_iou": False,
                        "reactivation_prompt_mode": str(cli.reactivation_prompt_mode),
                        "geometry_prompts_enabled": False,
                    }
                )
                continue
            if event.get("live_obj_id") is None:
                maybe_assign_prompt_new_object(
                    event,
                    state,
                    frame_idx=frame_idx_i,
                    frame_id=frame_id,
                    trigger="attempt",
                )
            if event.get("live_obj_id") is None:
                adapter_records.append(
                    {
                        "record_type": "attempt",
                        "event_index": int(event["event_index"]),
                        "source_lag": int(event["source_lag"]),
                        "frame_idx": frame_idx_i,
                        "frame_id": frame_id,
                        "reference_global_id": ref_id,
                        "live_obj_id": None,
                        "selected_variant": "SKIPPED_LONG_TERM_POLICY"
                        if str(event.get("long_term_admission_skip_reason", ""))
                        else "SKIPPED_NO_SOURCE_MAPPING",
                        "skip_reason": str(
                            event.get("prompt_new_object_skip_reason")
                            or event.get("long_term_admission_skip_reason")
                            or event.get("source_mapping_skip_reason")
                            or "no_source_live_id_mapping"
                        ),
                        "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                        "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                        "prompt_only_unmapped_source_reactivation": bool(
                            event.get("prompt_only_unmapped_source_reactivation", False)
                        ),
                        "prompt_only_new_object_id": int(event.get("prompt_only_new_object_id", -1)),
                        "online_gate_uses_reference_iou": False,
                    }
                )
                continue
            obj_id = int(event["live_obj_id"])
            if appearance_selector_enabled:
                ensure_return_base()
                assert return_ids is not None and return_masks is not None
                appearance_mask, record = select_appearance_only_mask(
                    event,
                    frame_idx=frame_idx_i,
                    frame_id=frame_id,
                    record_type="probation_attempt",
                    label=event["attempt_label"],
                    obj_ids=return_ids,
                    masks=return_masks,
                    use_lingbot_point_filter=bool(appearance_geometry_filter_enabled),
                    all_coords=event["attempt_all_coords"],
                    all_labels=event["attempt_all_labels"],
                    neg_ids=event["attempt_neg_ids"],
                )
                event["selected_variant"] = str(record.get("selected_variant", "appearance_only_rgb_shape"))
                event["probation_output_mask"] = bool(record.get("output_mask"))
                event["probation_skip_reason"] = str(record.get("skip_reason", ""))
                if output_plane_enabled and bool(record.get("output_mask")) and int(np.count_nonzero(appearance_mask)) > 0:
                    return_ids, return_masks = merge_object_mask(
                        return_ids,
                        return_masks,
                        obj_id=obj_id,
                        mask=appearance_mask,
                        prefer_new=True,
                    )
                if int(event["event_index"]) in probation_visual_event_indices:
                    rgb = rgb_frame(scene_root, frame_id)
                    path = (
                        output_root
                        / "highres_probation_visuals"
                        / f"event{int(event['event_index']):03d}_{record.get('selected_variant', 'appearance_only')}_f{frame_id}_ref{ref_id}_live{obj_id}.jpg"
                    )
                    draw_zoom_overlay_with_real_imwrite(
                        rgb=rgb,
                        ref_mask=event["attempt_label"] == ref_id,
                        pred_mask=appearance_mask,
                        points=event["attempt_all_coords"] if appearance_geometry_filter_enabled else None,
                        labels=event["attempt_all_labels"] if appearance_geometry_filter_enabled else None,
                        title=f"appearance-only event{int(event['event_index']):03d} attempt f{frame_id} ref{ref_id}",
                        output_path=path,
                        pad=int(cli.visual_pad),
                        scale=int(cli.visual_scale),
                        color=(255, 190, 40),
                    )
                    probation_visual_paths.append(path)
                    record["visual_path"] = rel(path)
                    record["visual_sha256"] = sha256_file(path)
                adapter_records.append(record)
                continue
            if str(cli.reactivation_probation_mode) == "shadow_attempt_confirm_commit":
                payload = {
                    "frame_id": int(frame_id),
                    "frame_idx": int(frame_idx_i),
                    "pos_coords": event["attempt_pos_coords"],
                    "pos_labels": event["attempt_pos_labels"],
                    "all_coords": event["attempt_all_coords"],
                    "all_labels": event["attempt_all_labels"],
                    "neg_ids": event["attempt_neg_ids"],
                    "label": event["attempt_label"],
                }
                probation_mask, record = decode_image_prompt_mask(
                    event,
                    payload,
                    frame_idx=frame_idx_i,
                    frame_id=frame_id,
                    record_type="probation_attempt",
                    variant_prefix="probation",
                    output_mode=str(cli.probation_output_mode),
                    min_positive_support=float(cli.probation_min_positive_support),
                )
                selected_variant_raw = str(record.get("selected_variant", "probation_G1_pos"))
                event["selected_variant"] = "G2_pos_neg" if selected_variant_raw.endswith("G2_pos_neg") else "G1_pos"
                event["probation_output_mask"] = bool(record.get("output_mask"))
                event["probation_skip_reason"] = str(record.get("skip_reason", ""))
                record["reactivation_probation_mode"] = str(cli.reactivation_probation_mode)
                record["probation_output_mode"] = str(cli.probation_output_mode)
                record["probation_min_positive_support"] = float(cli.probation_min_positive_support)
                record["will_commit_in_sam2_video_state_at_confirm"] = bool(record.get("output_mask"))
                if output_plane_enabled and bool(record.get("output_mask")) and int(np.count_nonzero(probation_mask)) > 0:
                    ensure_return_base()
                    assert return_ids is not None and return_masks is not None
                    return_ids, return_masks = merge_object_mask(
                        return_ids,
                        return_masks,
                        obj_id=obj_id,
                        mask=probation_mask,
                        prefer_new=True,
                    )
                if int(event["event_index"]) in probation_visual_event_indices:
                    rgb = rgb_frame(scene_root, frame_id)
                    path = (
                        output_root
                        / "highres_probation_visuals"
                        / f"event{int(event['event_index']):03d}_{selected_variant_raw}_f{frame_id}_ref{ref_id}_live{obj_id}.jpg"
                    )
                    draw_zoom_overlay_with_real_imwrite(
                        rgb=rgb,
                        ref_mask=event["attempt_label"] == ref_id,
                        pred_mask=probation_mask,
                        points=payload.get("visual_coords", payload["all_coords"]),
                        labels=payload.get("visual_labels", payload["all_labels"]),
                        title=(
                            f"probation event{int(event['event_index']):03d} {selected_variant_raw} "
                            f"f{frame_id} ref{ref_id} live{obj_id}"
                        ),
                        output_path=path,
                        pad=int(cli.visual_pad),
                        scale=int(cli.visual_scale),
                        color=(255, 190, 40),
                    )
                    probation_visual_paths.append(path)
                    record["visual_path"] = rel(path)
                    record["visual_sha256"] = sha256_file(path)
                adapter_records.append(record)
                continue

            ensure_return_base()
            g1_ids, g1_masks, g1_state, g1_sec = add_points(
                predictor,
                state,
                frame_idx=frame_idx_i,
                obj_id=obj_id,
                coords=event["attempt_pos_coords"],
                labels=event["attempt_pos_labels"],
                probe_args=probe_args,
            )
            g1_present, g1_mask = extract_mask(g1_ids, g1_masks, obj_id, event["attempt_label"].shape[:2])
            g1_rates = prompt_point_rates(g1_mask, event["attempt_all_coords"], event["attempt_all_labels"])
            negative_prompt_count = int(np.count_nonzero(np.asarray(event["attempt_all_labels"], dtype=np.int32) == 0))
            g2_eval = image_g3_g2_eval_decision(
                output_mode="image_g3_selector",
                g1_negative_conflict_rate=float(g1_rates["candidate_negative_point_conflict_rate"]),
                negative_prompt_count=negative_prompt_count,
            )
            g2_rates: dict[str, Any] = {}
            selected_ids, selected_masks = g1_ids, g1_masks
            selected_mask = g1_mask
            selected_variant = "G1_pos"
            selected_state = g1_state
            selected_runtime = g1_sec
            g2_present = False
            g2_choose_reason = "not_evaluated"
            if bool(g2_eval["evaluate"]):
                g2_ids, g2_masks, g2_state, g2_sec = add_points(
                    predictor,
                    state,
                    frame_idx=frame_idx_i,
                    obj_id=obj_id,
                    coords=event["attempt_all_coords"],
                    labels=event["attempt_all_labels"],
                    probe_args=probe_args,
                )
                g2_present, g2_mask = extract_mask(g2_ids, g2_masks, obj_id, event["attempt_label"].shape[:2])
                g2_rates = prompt_point_rates(g2_mask, event["attempt_all_coords"], event["attempt_all_labels"])
                g2_positive_support_ok = bool(
                    float(g2_rates["positive_point_support_rate"]) >= float(cli.online_select_min_g2_positive_support)
                )
                g2_conflict_selection = image_g3_g2_conflict_selection(
                    g1_negative_conflict_rate=float(g1_rates["candidate_negative_point_conflict_rate"]),
                    g2_negative_conflict_rate=float(g2_rates["candidate_negative_point_conflict_rate"]),
                )
                choose_g2 = bool(g2_present and g2_positive_support_ok and bool(g2_conflict_selection["conflict_ok"]))
                if not g2_present:
                    g2_choose_reason = "g2_object_not_present"
                elif not g2_positive_support_ok:
                    g2_choose_reason = "g2_positive_support_below_min"
                elif not bool(g2_conflict_selection["conflict_ok"]):
                    g2_choose_reason = str(g2_conflict_selection["reason"])
                else:
                    g2_choose_reason = f"g2_positive_support_ok_and_{g2_conflict_selection['reason']}"
                if choose_g2:
                    selected_ids, selected_masks = g2_ids, g2_masks
                    selected_mask = g2_mask
                    selected_variant = "G2_pos_neg"
                    selected_state = g2_state
                    selected_runtime = g2_sec
            event["selected_variant"] = selected_variant
            metrics = mask_metrics(
                selected_mask,
                event["attempt_label"] == ref_id,
                event["attempt_label"],
                set(int(v) for v in event["attempt_neg_ids"]),
            )
            adapter_records.append(
                {
                    "record_type": "attempt",
                    "event_index": int(event["event_index"]),
                    "source_lag": int(event["source_lag"]),
                    "frame_idx": frame_idx_i,
                    "frame_id": frame_id,
                    "reference_global_id": ref_id,
                    "live_obj_id": obj_id,
                    "source_live_area_px": int(event.get("source_live_area_px", 0)),
                    "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                    "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                    "prompt_only_unmapped_source_reactivation": bool(
                        event.get("prompt_only_unmapped_source_reactivation", False)
                    ),
                    "prompt_only_new_object_id": int(event.get("prompt_only_new_object_id", -1)),
                    "selected_variant": selected_variant,
                    "g1_present": bool(g1_present),
                    "g2_present": bool(g2_present),
                    "g1_candidate_negative_point_conflict_rate": float(
                        g1_rates["candidate_negative_point_conflict_rate"]
                    ),
                    "g2_candidate_negative_point_conflict_rate": float(
                        g2_rates.get("candidate_negative_point_conflict_rate", -1.0)
                    ),
                    "g2_positive_point_support_rate": float(g2_rates.get("positive_point_support_rate", -1.0)),
                    "g2_eval_policy": str(g2_eval["policy"]),
                    "g2_evaluated": bool(g2_eval["evaluate"]),
                    "g2_eval_reason": str(g2_eval["reason"]),
                    "g2_eval_neg_conflict_threshold": float(g2_eval["threshold"]),
                    "g2_chosen": bool(selected_variant == "G2_pos_neg"),
                    "g2_choose_reason": str(g2_choose_reason),
                    "g2_select_policy": str(cli.image_g3_selector_g2_select_policy),
                    "g2_min_neg_conflict_improvement": float(
                        cli.image_g3_selector_g2_min_neg_conflict_improvement
                    ),
                    "online_gate_uses_reference_iou": False,
                    "runtime_sec": float(selected_runtime),
                    "state_obj_ids_before_readd": selected_state.get("state_obj_ids_before_readd", []),
                    "state_obj_ids_after_readd": selected_state.get("state_obj_ids_after_readd", []),
                    **event_prompt_core_fields(event),
                    **{f"attempt_{key}": value for key, value in metrics.items()},
                }
            )
            if (
                output_plane_enabled
                and selected_ids is not None
                and selected_masks is not None
                and return_ids is not None
                and return_masks is not None
            ):
                return_ids, return_masks = merge_object_mask(
                    return_ids,
                    return_masks,
                    obj_id=obj_id,
                    mask=selected_mask,
                    prefer_new=True,
                )

        for event in sorted(frame_events.get("confirm", []), key=lambda row: int(row["event_index"])):
            ref_id = int(event["reference_global_id"])
            if not recoverability_enabled:
                adapter_records.append(
                    {
                        "record_type": "confirm",
                        "event_index": int(event["event_index"]),
                        "source_lag": int(event["source_lag"]),
                        "frame_idx": frame_idx_i,
                        "frame_id": frame_id,
                        "reference_global_id": ref_id,
                        "live_obj_id": event.get("live_obj_id"),
                        "source_live_area_px": int(event.get("source_live_area_px", 0)),
                        "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                        "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                        "prompt_only_unmapped_source_reactivation": False,
                        "prompt_only_new_object_id": -1,
                        "selected_variant": "SKIPPED_RECOVERABILITY_DISABLED",
                        "skip_reason": "recoverability_disabled",
                        "target_present": False,
                        "online_gate_uses_reference_iou": False,
                        "reactivation_probation_mode": str(cli.reactivation_probation_mode),
                        "reactivation_committed_to_sam2_video_state": False,
                        "recoverability_mode": str(cli.recoverability_mode),
                        **event_prompt_core_fields(event),
                    }
                )
                continue
            if not geometry_prompts_enabled and not appearance_selector_enabled:
                adapter_records.append(
                    {
                        "record_type": "confirm",
                        "event_index": int(event["event_index"]),
                        "source_lag": int(event["source_lag"]),
                        "frame_idx": frame_idx_i,
                        "frame_id": frame_id,
                        "reference_global_id": ref_id,
                        "live_obj_id": event.get("live_obj_id"),
                        "source_live_area_px": int(event.get("source_live_area_px", 0)),
                        "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                        "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                        "prompt_only_unmapped_source_reactivation": False,
                        "prompt_only_new_object_id": -1,
                        "selected_variant": "SKIPPED_GEOMETRY_DISABLED",
                        "skip_reason": "geometry_disabled",
                        "target_present": False,
                        "online_gate_uses_reference_iou": False,
                        "reactivation_probation_mode": str(cli.reactivation_probation_mode),
                        "reactivation_committed_to_sam2_video_state": False,
                        "reactivation_prompt_mode": str(cli.reactivation_prompt_mode),
                        "geometry_prompts_enabled": False,
                        **event_prompt_core_fields(event),
                    }
                )
                continue
            if event.get("live_obj_id") is None:
                adapter_records.append(
                    {
                        "record_type": "confirm",
                        "event_index": int(event["event_index"]),
                        "source_lag": int(event["source_lag"]),
                        "frame_idx": frame_idx_i,
                        "frame_id": frame_id,
                        "reference_global_id": ref_id,
                        "live_obj_id": None,
                        "selected_variant": "SKIPPED_LONG_TERM_POLICY"
                        if str(event.get("long_term_admission_skip_reason", ""))
                        else "SKIPPED_NO_SOURCE_MAPPING",
                        "skip_reason": str(
                            event.get("prompt_new_object_skip_reason")
                            or event.get("long_term_admission_skip_reason")
                            or event.get("source_mapping_skip_reason")
                            or "no_source_live_id_mapping"
                        ),
                        "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                        "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                        "prompt_only_unmapped_source_reactivation": bool(
                            event.get("prompt_only_unmapped_source_reactivation", False)
                        ),
                        "prompt_only_new_object_id": int(event.get("prompt_only_new_object_id", -1)),
                        "target_present": False,
                        "online_gate_uses_reference_iou": False,
                        **event_prompt_core_fields(event),
                    }
                )
                continue
            obj_id = int(event["live_obj_id"])
            if str(cli.reactivation_probation_mode) == "shadow_attempt_confirm_commit" and not bool(
                event.get("probation_output_mask", False)
            ):
                adapter_records.append(
                    {
                        "record_type": "confirm",
                        "event_index": int(event["event_index"]),
                        "source_lag": int(event["source_lag"]),
                        "frame_idx": frame_idx_i,
                        "frame_id": frame_id,
                        "reference_global_id": ref_id,
                        "live_obj_id": obj_id,
                        "source_live_area_px": int(event.get("source_live_area_px", 0)),
                        "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                        "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                        "prompt_only_unmapped_source_reactivation": bool(
                            event.get("prompt_only_unmapped_source_reactivation", False)
                        ),
                        "prompt_only_new_object_id": int(event.get("prompt_only_new_object_id", -1)),
                        "selected_variant": "SKIPPED_PROBATION_FAILED",
                        "skip_reason": str(event.get("probation_skip_reason", "probation_output_mask_false")),
                        "target_present": False,
                        "online_gate_uses_reference_iou": False,
                        "reactivation_probation_mode": str(cli.reactivation_probation_mode),
                        "reactivation_committed_to_sam2_video_state": False,
                        **event_prompt_core_fields(event),
                    }
                )
                continue
            if appearance_selector_enabled:
                ensure_return_base()
                assert return_ids is not None and return_masks is not None
                appearance_mask, record = select_appearance_only_mask(
                    event,
                    frame_idx=frame_idx_i,
                    frame_id=frame_id,
                    record_type="confirm",
                    label=event["confirm_label"],
                    obj_ids=return_ids,
                    masks=return_masks,
                    use_lingbot_point_filter=bool(appearance_geometry_filter_enabled),
                    all_coords=event["confirm_all_coords"],
                    all_labels=event["confirm_all_labels"],
                    neg_ids=event["confirm_neg_ids"],
                )
                record["reactivation_probation_mode"] = str(cli.reactivation_probation_mode)
                record["reactivation_committed_to_sam2_video_state"] = False
                record["confirm_reactivation_is_output_selection_only"] = True
                if output_plane_enabled and bool(record.get("output_mask")) and int(np.count_nonzero(appearance_mask)) > 0:
                    return_ids, return_masks = merge_object_mask(
                        return_ids,
                        return_masks,
                        obj_id=obj_id,
                        mask=appearance_mask,
                        prefer_new=True,
                    )
                if bool(event["write_visuals"]):
                    rgb = rgb_frame(scene_root, frame_id)
                    path = (
                        output_root
                        / "highres_event_visuals"
                        / f"event{int(event['event_index']):03d}_{record.get('selected_variant', 'appearance_only')}_confirm_f{frame_id}_ref{ref_id}_live{obj_id}.jpg"
                    )
                    draw_zoom_overlay_with_real_imwrite(
                        rgb=rgb,
                        ref_mask=event["confirm_label"] == ref_id,
                        pred_mask=appearance_mask,
                        points=event["confirm_all_coords"] if appearance_geometry_filter_enabled else None,
                        labels=event["confirm_all_labels"] if appearance_geometry_filter_enabled else None,
                        title=f"appearance-only event{int(event['event_index']):03d} confirm f{frame_id} ref{ref_id}",
                        output_path=path,
                        pad=int(cli.visual_pad),
                        scale=int(cli.visual_scale),
                        color=(70, 180, 255),
                    )
                    visual_paths.append(path)
                    record["visual_path"] = rel(path)
                    record["visual_sha256"] = sha256_file(path)
                adapter_records.append(record)
                continue
            ensure_return_base()
            selected_variant = str(event.get("selected_variant") or "G1_pos")
            coords = event["confirm_all_coords"] if selected_variant == "G2_pos_neg" else event["confirm_pos_coords"]
            labels = event["confirm_all_labels"] if selected_variant == "G2_pos_neg" else event["confirm_pos_labels"]
            coords, labels, confirm_target_mask_core_stats = target_mask_core_filter_prompt_set(
                frame_id=int(frame_id),
                shape=event["confirm_label"].shape[:2],
                coords=coords,
                labels=labels,
            )
            ids, masks, state_record, runtime = add_points(
                predictor,
                state,
                frame_idx=frame_idx_i,
                obj_id=obj_id,
                coords=coords,
                labels=labels,
                probe_args=probe_args,
            )
            reset_obj_ids = state.setdefault("v107_growth_history_reset_obj_ids", [])
            if isinstance(reset_obj_ids, list) and obj_id not in reset_obj_ids:
                reset_obj_ids.append(int(obj_id))
            present, mask = extract_mask(ids, masks, obj_id, event["confirm_label"].shape[:2])
            point_rates = prompt_point_rates(mask, event["confirm_all_coords"], event["confirm_all_labels"])
            metrics = mask_metrics(
                mask,
                event["confirm_label"] == ref_id,
                event["confirm_label"],
                set(int(v) for v in event["confirm_neg_ids"]),
            )
            record: dict[str, Any] = {
                "record_type": "confirm",
                "event_index": int(event["event_index"]),
                "source_lag": int(event["source_lag"]),
                "frame_idx": frame_idx_i,
                "frame_id": frame_id,
                "reference_global_id": ref_id,
                "live_obj_id": obj_id,
                "source_live_area_px": int(event.get("source_live_area_px", 0)),
                "long_term_memory_admitted": bool(event.get("long_term_memory_admitted", False)),
                "long_term_admission_skip_reason": str(event.get("long_term_admission_skip_reason", "")),
                "prompt_only_unmapped_source_reactivation": bool(
                    event.get("prompt_only_unmapped_source_reactivation", False)
                ),
                "prompt_only_new_object_id": int(event.get("prompt_only_new_object_id", -1)),
                "selected_variant": selected_variant,
                "target_present": bool(present),
                "reactivation_probation_mode": str(cli.reactivation_probation_mode),
                "reactivation_committed_to_sam2_video_state": True,
                "confirm_positive_point_support_rate": float(point_rates["positive_point_support_rate"]),
                "confirm_candidate_negative_point_conflict_rate": float(
                    point_rates["candidate_negative_point_conflict_rate"]
                ),
                "online_gate_uses_reference_iou": False,
                "runtime_sec": float(runtime),
                "state_obj_ids_before_readd": state_record.get("state_obj_ids_before_readd", []),
                "state_obj_ids_after_readd": state_record.get("state_obj_ids_after_readd", []),
                **event_prompt_core_fields(event),
                **flatten_prompt_target_mask_core_stats("confirm", confirm_target_mask_core_stats),
                **{f"confirm_{key}": value for key, value in metrics.items()},
            }
            if bool(event["write_visuals"]):
                rgb = rgb_frame(scene_root, frame_id)
                color = (255, 70, 170) if selected_variant == "G2_pos_neg" else (40, 220, 255)
                path = (
                    output_root
                    / "highres_event_visuals"
                    / f"event{int(event['event_index']):03d}_{selected_variant}_confirm_f{frame_id}_ref{ref_id}_live{obj_id}.jpg"
                )
                draw_zoom_overlay_with_real_imwrite(
                    rgb=rgb,
                    ref_mask=event["confirm_label"] == ref_id,
                    pred_mask=mask,
                    points=coords,
                    labels=labels,
                    title=(
                        f"rolling event{int(event['event_index']):03d} {selected_variant} "
                        f"confirm f{frame_id} ref{ref_id} live{obj_id}"
                    ),
                    output_path=path,
                    pad=int(cli.visual_pad),
                    scale=int(cli.visual_scale),
                    color=color,
                )
                visual_paths.append(path)
                record["visual_path"] = rel(path)
                record["visual_sha256"] = sha256_file(path)
                if np.count_nonzero(event["confirm_all_labels"] == 0) > 0:
                    neg_mask = np.asarray(event["confirm_all_labels"], dtype=np.int32).reshape(-1) == 0
                    audit_neg_coords = np.asarray(event["confirm_all_coords"], dtype=np.float32).reshape(-1, 2)[neg_mask]
                    audit_neg_labels = np.asarray(event["confirm_all_labels"], dtype=np.int32).reshape(-1)[neg_mask]
                    audit_coords = np.concatenate([np.asarray(coords, dtype=np.float32).reshape(-1, 2), audit_neg_coords], axis=0)
                    audit_labels = np.concatenate([np.asarray(labels, dtype=np.int32).reshape(-1), audit_neg_labels], axis=0)
                    audit_path = (
                        output_root
                        / "highres_event_visuals"
                        / (
                            f"event{int(event['event_index']):03d}_{selected_variant}_all_prompts_confirm_"
                            f"f{frame_id}_ref{ref_id}_live{obj_id}.jpg"
                        )
                    )
                    draw_zoom_overlay_with_real_imwrite(
                        rgb=rgb,
                        ref_mask=event["confirm_label"] == ref_id,
                        pred_mask=mask,
                        points=audit_coords,
                        labels=audit_labels,
                        title=(
                            f"rolling event{int(event['event_index']):03d} {selected_variant} all-prompts "
                            f"confirm f{frame_id} ref{ref_id} live{obj_id}"
                        ),
                        output_path=audit_path,
                        pad=int(cli.visual_pad),
                        scale=int(cli.visual_scale),
                        color=color,
                    )
                    visual_paths.append(audit_path)
                    record["all_prompt_visual_path"] = rel(audit_path)
                    record["all_prompt_visual_sha256"] = sha256_file(audit_path)
            adapter_records.append(record)
            if output_plane_enabled and present and return_ids is not None and return_masks is not None:
                return_ids, return_masks = merge_object_mask(
                    return_ids,
                    return_masks,
                    obj_id=obj_id,
                    mask=mask,
                    prefer_new=True,
                )

        if return_ids is not None and return_masks is not None:
            return return_ids, return_masks
        return original_base_infer(predictor, state, frame_idx=frame_idx_i)

    baseline_cli = SimpleNamespace(
        config=str(config_path),
        scene_id=str(cli.scene_id),
        rgb_root=str(cli.rgb_root),
        frame_start=int(cli.frame_start),
        frame_stride=int(cli.frame_stride),
        frame_count=int(cli.frame_count),
        frame_ids=str(cli.frame_ids or ""),
        output_root=str(output_root),
        seed=int(cli.seed),
        birth_dump_dir="",
    )
    args = make_args(config, baseline_cli)
    args.output_root = str(output_root)
    args.variant_id = VARIANT_ID
    args.baseline_id = "v107-phase8-g3-rolling-scheduler-smoke"
    args.model_provider = "sam2"
    args.propagation_mode = "streaming_state"
    args.model_dtype = "bfloat16" if str(cli.model_dtype).lower() == "bf16" else str(cli.model_dtype)
    args.runtime_num_maskmem = int(cli.runtime_num_maskmem)
    args.runtime_max_obj_ptrs_in_encoder = int(cli.runtime_max_obj_ptrs_in_encoder)
    args.runtime_max_cond_frames_in_attn = int(cli.runtime_max_cond_frames_in_attn)
    args.stream_keep_noncond_frames = int(cli.stream_keep_noncond_frames)
    args.stream_prune_invisible_after_frames = int(cli.stream_prune_invisible_after_frames)
    args.stream_prune_min_visible_area = int(cli.stream_prune_min_visible_area)
    args.stream_prune_protect_min_ever_area = int(cli.stream_prune_protect_min_ever_area)
    args.stream_prune_protect_max_objects = int(cli.stream_prune_protect_max_objects)
    args.stream_prune_max_visible_area = int(cli.stream_prune_max_visible_area)
    args.stream_prune_max_visible_area_ratio = float(cli.stream_prune_max_visible_area_ratio)
    args.stream_disjoin_policy = str(cli.stream_disjoin_policy)
    args.stream_disjoin_claim_dropped = not bool(cli.stream_disjoin_claim_kept_only)
    args.stream_disjoin_min_area_px = int(cli.stream_disjoin_min_area_px)
    args.stream_disjoin_recent_min_iou = float(cli.stream_disjoin_recent_min_iou)
    args.stream_disjoin_recent_max_area_growth = float(cli.stream_disjoin_recent_max_area_growth)
    args.stream_oversized_prune_action = str(cli.stream_oversized_prune_action)
    args.stream_growth_prune_ratio = float(cli.stream_growth_prune_ratio)
    args.stream_growth_prune_min_area = int(cli.stream_growth_prune_min_area)
    args.stream_growth_prune_history = int(cli.stream_growth_prune_history)
    args.stream_growth_prune_warmup = int(cli.stream_growth_prune_warmup)
    args.stream_growth_prune_action = str(cli.stream_growth_prune_action)
    args.stream_growth_prune_max_history_median_area = int(
        cli.stream_growth_prune_max_history_median_area
    )
    args.stream_empty_cache_every = int(cli.stream_empty_cache_every)
    args.stream_empty_cache_on_prune = bool(cli.stream_empty_cache_on_prune)
    args.offload_video_to_cpu = bool(cli.offload_video_to_cpu)
    args.offload_state_to_cpu = bool(cli.offload_state_to_cpu)
    args.gap_sampler = "component_adaptive"
    args.gap_max_points = int(cli.gap_max_points)
    args.gap_min_component_area = int(cli.gap_min_component_area)
    args.gap_area_per_extra_point = int(cli.gap_area_per_extra_point)
    args.gap_max_points_per_component = int(cli.gap_max_points_per_component)
    args.gap_min_image_edge_distance_px = int(cli.gap_min_image_edge_distance_px)
    if cli.gap_iou_threshold is not None:
        args.gap_iou_threshold = float(cli.gap_iou_threshold)
    if cli.gap_stability_threshold is not None:
        args.gap_stability_threshold = float(cli.gap_stability_threshold)
    args.gap_relaxed_min_uncovered_ratio = float(cli.gap_relaxed_min_uncovered_ratio)
    args.gap_relaxed_iou_threshold = float(cli.gap_relaxed_iou_threshold)
    args.gap_relaxed_stability_threshold = float(cli.gap_relaxed_stability_threshold)
    args.gap_relaxed_min_clipped_area = int(cli.gap_relaxed_min_clipped_area)
    args.gap_small_mask_max_area = int(cli.gap_small_mask_max_area)
    args.gap_small_mask_min_pred_iou = float(cli.gap_small_mask_min_pred_iou)
    args.gap_output_min_pred_iou = float(cli.gap_output_min_pred_iou)
    args.gap_output_min_stability = float(cli.gap_output_min_stability)
    args.gap_output_allow_relaxed = not bool(cli.gap_output_disallow_relaxed)
    args.gap_delayed_admission_enabled = bool(cli.gap_delayed_admission)
    args.gap_admission_min_pred_iou = float(cli.gap_admission_min_pred_iou)
    args.gap_admission_min_stability = float(cli.gap_admission_min_stability)
    args.gap_admission_allow_relaxed = not bool(cli.gap_admission_disallow_relaxed)
    args.gap_reuse_recent_id_window = int(cli.gap_reuse_recent_id_window)
    args.gap_reuse_recent_id_iou = float(cli.gap_reuse_recent_id_iou)
    args.gap_reuse_recent_id_min_area = int(cli.gap_reuse_recent_id_min_area)
    args.gap_large_reuse_recent_id_iou = float(cli.gap_large_reuse_recent_id_iou)
    args.gap_large_reuse_min_area = int(cli.gap_large_reuse_min_area)
    args.gap_large_reuse_max_area_ratio = float(cli.gap_large_reuse_max_area_ratio)
    args.gap_anti_merge_core_window_frames = int(cli.gap_anti_merge_core_window_frames)
    args.gap_anti_merge_core_erode_px = int(cli.gap_anti_merge_core_erode_px)
    args.gap_anti_merge_core_min_area = int(cli.gap_anti_merge_core_min_area)
    args.gap_anti_merge_core_min_overlap_px = int(cli.gap_anti_merge_core_min_overlap_px)
    args.gap_anti_merge_core_min_overlap_ratio = float(cli.gap_anti_merge_core_min_overlap_ratio)
    args.gap_anti_merge_max_overlap_objects = int(cli.gap_anti_merge_max_overlap_objects)
    args.output_reuse_recent_id_window = int(cli.output_reuse_recent_id_window)
    args.output_reuse_recent_id_iou = float(cli.output_reuse_recent_id_iou)
    args.output_reuse_recent_id_min_area = int(cli.output_reuse_recent_id_min_area)
    args.output_large_reuse_recent_id_iou = float(cli.output_large_reuse_recent_id_iou)
    args.output_large_reuse_min_area = int(cli.output_large_reuse_min_area)
    args.output_large_reuse_max_area_ratio = float(cli.output_large_reuse_max_area_ratio)
    args.output_reuse_recent_id_preference = str(cli.output_reuse_recent_id_preference)
    args.output_reuse_prevent_collision_union = bool(cli.output_reuse_prevent_collision_union)
    args.output_fragment_max_area = int(cli.output_fragment_max_area)
    args.output_fragment_suppress_max_area = int(cli.output_fragment_suppress_max_area)
    args.output_fragment_merge_dilate_px = int(cli.output_fragment_merge_dilate_px)
    args.output_fragment_merge_min_touch_px = int(cli.output_fragment_merge_min_touch_px)
    args.output_fragment_merge_min_touch_ratio = float(cli.output_fragment_merge_min_touch_ratio)
    args.output_fragment_merge_min_neighbor_area = int(cli.output_fragment_merge_min_neighbor_area)
    args.disable_gap_birth = bool(cli.disable_gap_birth)
    args.gap_output_max_bbox_frac = float(cli.gap_output_max_bbox_frac)
    args.gap_output_max_edge_touch_count = int(cli.gap_output_max_edge_touch_count)
    args.gap_output_min_extent = float(cli.gap_output_min_extent)
    args.gap_output_min_core_area_px = int(cli.gap_output_min_core_area_px)
    args.gap_output_shape_min_uncovered_ratio = float(cli.gap_output_shape_min_uncovered_ratio)
    args.gap_output_min_input_mask_count = int(cli.gap_output_min_input_mask_count)
    args.skip_visual_export = bool(cli.skip_visual_export)
    args.lean_visual_export = False
    args.label_only_visual_export = bool(cli.label_only_visual_export)
    args.compact_visual_video = bool(cli.compact_visual_video)
    args.birth_admission_min_area = int(cli.birth_admission_min_area)
    args.birth_admission_max_area = int(cli.birth_admission_max_area)
    args.birth_admission_max_uncovered_ratio = float(cli.birth_admission_max_uncovered_ratio)
    args.birth_admission_max_bbox_frac = float(cli.birth_admission_max_bbox_frac)
    args.birth_admission_max_edge_touch_count = int(cli.birth_admission_max_edge_touch_count)
    args.birth_admission_min_extent = float(cli.birth_admission_min_extent)
    args.birth_admission_min_core_area_px = int(cli.birth_admission_min_core_area_px)
    args.birth_admission_shape_min_uncovered_ratio = float(cli.birth_admission_shape_min_uncovered_ratio)
    args.birth_admission_every = int(cli.birth_admission_every)
    args.birth_admission_max_per_frame = int(cli.birth_admission_max_per_frame)
    args.birth_admission_persistence_iou = float(cli.birth_admission_persistence_iou)
    args.birth_admission_persistence_hits = int(cli.birth_admission_persistence_hits)
    args.birth_admission_pending_ttl = int(cli.birth_admission_pending_ttl)
    args.birth_admission_persistence_min_area = int(cli.birth_admission_persistence_min_area)
    args.birth_admission_persistence_max_per_frame = int(cli.birth_admission_persistence_max_per_frame)
    args.birth_admission_immediate_area = int(cli.birth_admission_immediate_area)
    args.birth_admission_rescue_min_visible_count = int(cli.birth_admission_rescue_min_visible_count)
    args.birth_admission_rescue_min_foreground_ratio = float(cli.birth_admission_rescue_min_foreground_ratio)
    args.birth_admission_appearance_enabled = bool(cli.birth_admission_appearance_enabled) and not bool(
        cli.disable_birth_admission_appearance
    )
    args.birth_admission_appearance_min_iou = float(cli.birth_admission_appearance_min_iou)
    args.birth_admission_appearance_max_color_distance = float(cli.birth_admission_appearance_max_color_distance)
    args.birth_admission_appearance_max_centroid_distance = float(cli.birth_admission_appearance_max_centroid_distance)
    args.birth_admission_appearance_max_area_ratio = float(cli.birth_admission_appearance_max_area_ratio)
    args.birth_transaction_enabled = bool(cli.birth_transaction_enabled) and not bool(cli.disable_birth_transaction)
    args.birth_transaction_min_pending = int(cli.birth_transaction_min_pending)
    args.birth_transaction_max_delay_frames = int(cli.birth_transaction_max_delay_frames)
    args.birth_transaction_immediate_area = int(cli.birth_transaction_immediate_area)
    args.birth_transaction_min_total_area = int(cli.birth_transaction_min_total_area)
    args.birth_recon_prune_keep_frames = int(cli.birth_recon_prune_keep_frames)
    args.fps = float(cli.fps)

    base.add_masks_to_stream_state = hooked_add_masks_to_stream_state
    base.infer_stream_frame = hooked_infer_stream_frame
    run_error = ""
    try:
        t0 = time.time()
        run_rolling(args)
        rolling_wall_sec = float(time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        rolling_wall_sec = float(time.time() - started)
        run_error = repr(exc)
        raise
    finally:
        base.infer_stream_frame = original_base_infer
        base.add_masks_to_stream_state = original_base_add_masks

    run_root = output_root / VARIANT_ID
    summary_path = run_root / "summary.json"
    summary = read_json(summary_path)
    rolling_stats = get_rolling_stats()
    visual_started = time.time()
    if bool(cli.skip_visual_export):
        visual = {
            "schema_version": "stream4d_v107_g3_rolling_visual_v1",
            "path": "",
            "sha256": "",
            "frame_count": 0,
            "skipped": True,
            "reason": "skip_visual_export",
        }
    elif bool(cli.label_only_visual_export):
        from tools.run_v106_stateful_sam2_rolling_scene_stream import _write_side_by_side_video_from_labels

        visual = _write_side_by_side_video_from_labels(
            summary=summary,
            output_root=run_root,
            variant_id=VARIANT_ID,
            fps=float(cli.fps),
            compact_video=bool(cli.compact_visual_video),
        )
        visual["skipped"] = False
        visual["layout"] = "RGB frame | v107 scheduled G3 rolling overlay from labels"
    else:
        visual = _write_side_by_side_video(
            summary=summary,
            output_root=run_root,
            variant_id=VARIANT_ID,
            fps=float(cli.fps),
        )
        visual["skipped"] = False
        visual["layout"] = "RGB frame | v107 scheduled G3 rolling overlay"
    visual_export_sec = float(time.time() - visual_started)

    records_jsonl = output_root / "g3_scheduler_records.jsonl"
    with records_jsonl.open("w", encoding="utf-8") as handle:
        for row in adapter_records:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
    records_csv = output_root / "g3_scheduler_records.csv"
    write_rows_csv(records_csv, adapter_records)

    confirm_rows = [row for row in adapter_records if row.get("record_type") == "confirm"]
    attempt_rows = [row for row in adapter_records if row.get("record_type") == "attempt"]
    probation_rows = [row for row in adapter_records if row.get("record_type") == "probation_attempt"]
    shadow_rows = [row for row in adapter_records if row.get("record_type") == "shadow_output"]
    selected_g2 = [
        row
        for row in [*attempt_rows, *probation_rows]
        if str(row.get("selected_variant", "")).endswith("G2_pos_neg")
    ]
    probation_output_rows = [row for row in probation_rows if bool(row.get("output_mask"))]
    shadow_output_rows = [row for row in shadow_rows if bool(row.get("output_mask"))]
    probation_skip_reasons: dict[str, int] = {}
    for row in probation_rows:
        reason = str(row.get("skip_reason", ""))
        if reason:
            probation_skip_reasons[reason] = int(probation_skip_reasons.get(reason, 0)) + 1
    shadow_skip_reasons: dict[str, int] = {}
    for row in shadow_rows:
        reason = str(row.get("skip_reason", ""))
        if reason:
            shadow_skip_reasons[reason] = int(shadow_skip_reasons.get(reason, 0)) + 1
    mapping_rows = [row for row in adapter_records if row.get("record_type") == "source_identity_mapping"]
    admitted_mapping_rows = [row for row in mapping_rows if bool(row.get("long_term_memory_admitted"))]
    prompt_new_assignment_rows = [
        row for row in adapter_records if row.get("record_type") == "prompt_new_object_assignment"
    ]
    prompt_only_output_rows = [
        row
        for row in [*shadow_rows, *probation_rows, *confirm_rows]
        if bool(row.get("prompt_only_unmapped_source_reactivation", False))
    ]
    actual_video_readd_rows = [
        row
        for row in [*attempt_rows, *confirm_rows]
        if row.get("live_obj_id") not in {None, ""}
        and not str(row.get("selected_variant", "")).startswith("SKIPPED")
        and str(row.get("skip_reason", "")) == ""
        and not bool(row.get("appearance_only_control", False))
        and not bool(row.get("appearance_geometry_filter_control", False))
    ]
    appearance_only_rows = [row for row in [*probation_rows, *confirm_rows] if bool(row.get("appearance_only_control", False))]
    appearance_only_output_rows = [
        row for row in appearance_only_rows if bool(row.get("output_mask") or row.get("target_present"))
    ]
    appearance_geometry_rows = [
        row for row in [*probation_rows, *confirm_rows] if bool(row.get("appearance_geometry_filter_control", False))
    ]
    appearance_geometry_output_rows = [
        row for row in appearance_geometry_rows if bool(row.get("output_mask") or row.get("target_present"))
    ]
    recoverability_disabled_rows = [
        row for row in adapter_records if str(row.get("skip_reason", "")) == "recoverability_disabled"
    ]
    long_term_skip_reasons: dict[str, int] = {}
    source_mapping_skip_reasons: dict[str, int] = {}
    for row in mapping_rows:
        reason = str(row.get("source_mapping_skip_reason", ""))
        if reason:
            source_mapping_skip_reasons[reason] = int(source_mapping_skip_reasons.get(reason, 0)) + 1
        admission_reason = str(row.get("long_term_admission_skip_reason", ""))
        if admission_reason:
            for part in admission_reason.split(";"):
                long_term_skip_reasons[part] = int(long_term_skip_reasons.get(part, 0)) + 1

    def mean(rows: list[dict[str, Any]], key: str) -> float:
        vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    def rate(rows: list[dict[str, Any]], key: str, threshold: float, op: str = "ge") -> float:
        vals = [float(row[key]) for row in rows if key in row and row[key] is not None]
        if not vals:
            return 0.0
        if op == "gt":
            return float(sum(v > threshold for v in vals) / len(vals))
        return float(sum(v >= threshold for v in vals) / len(vals))

    adapter_summary = {
        "schema_version": "stream4d_v107_phase8_g3_rolling_scheduler_smoke_summary_v1",
        "status": "SCHEDULED_G3_ROLLING_SMOKE_NOT_FULL_PHASE8_POLICY",
        "runtime_sec": float(time.time() - started),
        "rolling_wall_sec": float(rolling_wall_sec),
        "run_error": run_error,
        "variant_id": VARIANT_ID,
        "scene_id": str(cli.scene_id),
        "frame_ids": [int(v) for v in frame_ids],
        "event_indices": sorted(event_indices),
        "visual_event_indices": sorted(visual_event_indices),
        "prompt_probe_root": rel(prompt_root),
        "prompt_points_json": rel(points_path),
        "prompt_points_json_sha256": sha256_file(points_path),
        "prompt_points_json_loaded": bool(prompt_points_enabled),
        "prompt_points_json_used_for_reactivation": bool(geometry_prompts_enabled),
        "prompt_points_json_used_for_candidate_filter": bool(appearance_geometry_filter_enabled),
        "prompt_core_filter_enabled": bool(float(cli.prompt_core_min_source_mask_distance_px) > 0.0),
        "prompt_core_min_source_mask_distance_px": float(cli.prompt_core_min_source_mask_distance_px),
        "prompt_core_filter_descriptor": (
            "source_frame_object_mask_distance_transform_core_points_only"
            if float(cli.prompt_core_min_source_mask_distance_px) > 0.0
            else ""
        ),
        "prompt_source_core_supplement_enabled": bool(source_core_supplement_enabled),
        "prompt_source_core_supplement_positive_points": int(cli.prompt_source_core_supplement_positive_points),
        "prompt_source_core_supplement_trigger_max_positive_points": int(
            cli.prompt_source_core_supplement_trigger_max_positive_points
        ),
        "prompt_source_core_supplement_min_distance_px": float(
            cli.prompt_source_core_supplement_min_distance_px
        ),
        "prompt_source_core_supplement_depth_abs_tolerance": float(
            cli.prompt_source_core_supplement_depth_abs_tolerance
        ),
        "prompt_source_core_supplement_depth_rel_tolerance": float(
            cli.prompt_source_core_supplement_depth_rel_tolerance
        ),
        "prompt_source_core_supplement_min_depth_conf": float(cli.prompt_source_core_supplement_min_depth_conf),
        "prompt_source_core_supplement_duplicate_radius_px": float(
            cli.prompt_source_core_supplement_duplicate_radius_px
        ),
        "prompt_source_core_supplement_descriptor": (
            "lingbot_source_mask_core_extra_positive_candidates_projected_with_raw_lingbot_geometry"
            if bool(source_core_supplement_enabled)
            else ""
        ),
        "prompt_source_core_supplement_npz": rel(source_core_supplement_npz_path)
        if source_core_supplement_npz_path is not None
        else "",
        "prompt_source_core_supplement_npz_sha256": sha256_file(source_core_supplement_npz_path)
        if source_core_supplement_npz_path is not None
        else "",
        "prompt_source_core_negative_supplement_enabled": bool(source_core_negative_supplement_enabled),
        "prompt_source_core_negative_supplement_negative_points": int(
            cli.prompt_source_core_supplement_negative_points
        ),
        "prompt_source_core_negative_supplement_trigger_max_negative_points": int(
            cli.prompt_source_core_supplement_negative_trigger_max_negative_points
        ),
        "prompt_source_core_negative_supplement_min_distance_px": float(
            cli.prompt_source_core_supplement_negative_min_distance_px
        ),
        "prompt_source_core_negative_supplement_max_neighbor_bbox_distance_px": float(
            cli.prompt_source_core_supplement_negative_max_neighbor_bbox_distance_px
        ),
        "prompt_source_core_negative_supplement_target_border_margin_px": float(
            cli.prompt_source_core_supplement_negative_target_border_margin_px
        ),
        "prompt_source_core_negative_supplement_min_area_px": int(
            cli.prompt_source_core_supplement_negative_min_area_px
        ),
        "prompt_source_core_negative_supplement_max_objects": int(
            cli.prompt_source_core_supplement_negative_max_objects
        ),
        "prompt_source_core_negative_supplement_descriptor": (
            "nearby_coview_source_object_mask_core_extra_negative_candidates_projected_with_raw_lingbot_geometry"
            if bool(source_core_negative_supplement_enabled)
            else ""
        ),
        "prompt_source_core_negative_supplement_npz": rel(source_core_supplement_npz_path)
        if source_core_supplement_npz_path is not None and bool(source_core_negative_supplement_enabled)
        else "",
        "prompt_source_core_negative_supplement_npz_sha256": sha256_file(source_core_supplement_npz_path)
        if source_core_supplement_npz_path is not None and bool(source_core_negative_supplement_enabled)
        else "",
        "prompt_target_stability_filter_enabled": bool(target_stability_enabled),
        "prompt_target_stability_depth_radius_px": int(cli.prompt_target_stability_depth_radius_px),
        "prompt_target_stability_max_local_depth_range_m": float(
            cli.prompt_target_stability_max_local_depth_range_m
        ),
        "prompt_target_stability_max_depth_abs_error": float(cli.prompt_target_stability_max_depth_abs_error),
        "prompt_target_stability_min_depth_conf": float(cli.prompt_target_stability_min_depth_conf),
        "prompt_target_stability_min_valid_depth_count": int(cli.prompt_target_stability_min_valid_depth_count),
        "prompt_target_stability_descriptor": (
            "lingbot_target_depth_local_range_no_reference_label_gate"
            if bool(target_stability_enabled)
            else ""
        ),
        "prompt_target_stability_depth_npz": rel(target_depth_npz_path) if target_depth_npz_path is not None else "",
        "prompt_target_stability_depth_npz_sha256": sha256_file(target_depth_npz_path)
        if target_depth_npz_path is not None
        else "",
        "prompt_anchor_conflict_filter_enabled": bool(anchor_conflict_enabled),
        "prompt_anchor_conflict_negative_radius_px": float(cli.prompt_anchor_conflict_negative_radius_px),
        "prompt_anchor_conflict_positive_cluster_radius_px": float(
            cli.prompt_anchor_conflict_positive_cluster_radius_px
        ),
        "prompt_anchor_conflict_min_positive_points": int(cli.prompt_anchor_conflict_min_positive_points),
        "prompt_anchor_conflict_descriptor": (
            "visible_projected_positive_points_filtered_by_negative_prompt_conflict_and_positive_consensus_cluster"
            if bool(anchor_conflict_enabled)
            else ""
        ),
        "prompt_target_mask_core_filter_enabled": bool(
            prompt_target_mask_core_enabled(float(cli.prompt_target_mask_core_min_distance_px))
        ),
        "prompt_target_mask_core_min_distance_px": float(cli.prompt_target_mask_core_min_distance_px),
        "prompt_target_mask_core_min_positive_points": int(cli.prompt_target_mask_core_min_positive_points),
        "prompt_target_mask_core_descriptor": (
            "online_sam2_candidate_mask_distance_transform_positive_points_only_no_reference_label_gate"
            if prompt_target_mask_core_enabled(float(cli.prompt_target_mask_core_min_distance_px))
            else ""
        ),
        "online_select_neg_conflict_threshold": float(cli.online_select_neg_conflict_threshold),
        "online_select_min_g2_positive_support": float(cli.online_select_min_g2_positive_support),
        "image_g3_selector_g2_eval_policy": str(cli.image_g3_selector_g2_eval_policy),
        "image_g3_selector_g2_select_policy": str(cli.image_g3_selector_g2_select_policy),
        "image_g3_selector_g2_min_neg_conflict_improvement": float(
            cli.image_g3_selector_g2_min_neg_conflict_improvement
        ),
        "probe_root": rel(probe_root),
        "reference_run_root": rel(reference_root),
        "uses_scannet_pose_or_depth_for_projection": False,
        "projection_geometry_source": prompt_summary.get("projection_geometry_source", "LingBot-Map"),
        "selected_pose_mode": prompt_summary.get("selected_pose_mode", "direct_as_c2w"),
        "reactivation_prompt_mode": str(cli.reactivation_prompt_mode),
        "geometry_prompts_enabled": bool(geometry_prompts_enabled),
        "random_geometry_prompts_enabled": bool(random_geometry_prompts_enabled),
        "random_geometry_descriptor": (
            "deterministic_uniform_image_coordinates_preserving_prompt_role_counts"
            if bool(random_geometry_prompts_enabled)
            else ""
        ),
        "appearance_only_reactivation_implemented": bool(appearance_only_enabled),
        "appearance_geometry_filter_enabled": bool(appearance_geometry_filter_enabled),
        "appearance_only_uses_lingbot_prompt_points": False,
        "appearance_geometry_filter_uses_lingbot_prompt_points": bool(appearance_geometry_filter_enabled),
        "appearance_only_descriptor": "rgb_mean_std_shape_from_source_mask_vs_current_base_masks"
        if appearance_only_enabled
        else "",
        "appearance_geometry_filter_descriptor": (
            "rgb_mean_std_shape_current_base_masks_plus_lingbot_visible_positive_negative_point_filter"
            if appearance_geometry_filter_enabled
            else ""
        ),
        "appearance_only_min_score": float(cli.appearance_only_min_score),
        "appearance_only_min_margin": float(cli.appearance_only_min_margin),
        "appearance_only_color_scale": float(cli.appearance_only_color_scale),
        "appearance_geometry_min_positive_support": float(cli.appearance_geometry_min_positive_support),
        "appearance_geometry_max_negative_conflict": float(cli.appearance_geometry_max_negative_conflict),
        "appearance_geometry_appearance_weight": float(cli.appearance_geometry_appearance_weight),
        "appearance_geometry_positive_weight": float(cli.appearance_geometry_positive_weight),
        "geometry_disabled_skip_reason": "geometry_disabled" if str(cli.reactivation_prompt_mode) == "no_geometry" else "",
        "output_plane_enabled": bool(output_plane_enabled),
        "disable_output_plane": bool(cli.disable_output_plane),
        "online_gate_uses_reference_iou": False,
        "reference_labels_used_for_offline_evaluation_only": True,
        "reference_metrics_are_diagnostic_only": True,
        "acceptance_gate_uses_diagnostic_reference_metrics": False,
        "visual_review_status": "USER_VISUAL_REVIEW_PENDING",
        "small_objects_may_skip_long_term_memory": True,
        "recoverability_mode": str(cli.recoverability_mode),
        "recoverability_enabled": bool(recoverability_enabled),
        "recoverability_disabled_record_count": int(len(recoverability_disabled_rows)),
        "auto_selection_policy": str(cli.auto_selection_policy),
        "event_selection": event_selection_summary,
        "event_selection_records_json": rel(event_selection_json),
        "event_selection_records_json_sha256": sha256_file(event_selection_json),
        "event_selection_records_csv": rel(event_selection_csv),
        "event_selection_records_csv_sha256": sha256_file(event_selection_csv),
        "long_term_min_source_area": int(cli.long_term_min_source_area),
        "long_term_min_positive_points": int(cli.long_term_min_positive_points),
        "long_term_min_confirm_positive_points": int(cli.long_term_min_confirm_positive_points),
        "long_term_max_events": int(cli.long_term_max_events),
        "long_term_anchor_max_area_frac": float(cli.long_term_anchor_max_area_frac),
        "long_term_anchor_max_bbox_frac": float(cli.long_term_anchor_max_bbox_frac),
        "long_term_anchor_max_edge_touch_count": int(cli.long_term_anchor_max_edge_touch_count),
        "long_term_anchor_min_extent": float(cli.long_term_anchor_min_extent),
        "long_term_anchor_min_core_area_px": int(cli.long_term_anchor_min_core_area_px),
        "long_term_anchor_gate_enabled": bool(
            float(cli.long_term_anchor_max_area_frac) > 0.0
            or float(cli.long_term_anchor_max_bbox_frac) > 0.0
            or int(cli.long_term_anchor_max_edge_touch_count) >= 0
            or float(cli.long_term_anchor_min_extent) > 0.0
            or int(cli.long_term_anchor_min_core_area_px) > 0
        ),
        "unmapped_source_policy": str(cli.unmapped_source_policy),
        "prompt_new_object_assignment_count": int(len(prompt_new_assignment_rows)),
        "prompt_new_object_reference_ids": sorted(
            {int(row["reference_global_id"]) for row in prompt_new_assignment_rows}
        ),
        "prompt_new_object_ids": sorted({int(row["live_obj_id"]) for row in prompt_new_assignment_rows}),
        "prompt_only_unmapped_source_output_record_count": int(len(prompt_only_output_rows)),
        "source_mapping_record_count": int(len(mapping_rows)),
        "source_mapping_accepted_count": int(sum(bool(row.get("source_mapping_accepted")) for row in mapping_rows)),
        "source_mapping_skip_reasons": source_mapping_skip_reasons,
        "long_term_memory_admitted_count": int(len(admitted_mapping_rows)),
        "long_term_admission_skip_reasons": long_term_skip_reasons,
        "physical_anchor_readiness_gate_enabled": True,
        "physical_anchor_ready_source_mapping_count": int(
            sum(bool(row.get("physical_anchor_ready", False)) for row in mapping_rows)
        ),
        "physical_anchor_readiness_requires_lingbot_provenance": True,
        "physical_anchor_readiness_rejects_random_geometry": True,
        "physical_anchor_readiness_requires_negative_prompts": True,
        "physical_anchor_readiness_min_positive_points": int(cli.prompt_anchor_conflict_min_positive_points),
        "shadow_output_mode": str(cli.shadow_output_mode),
        "shadow_online_selection_uses_reference_iou": False,
        "shadow_min_source_area": int(cli.shadow_min_source_area),
        "shadow_min_positive_support": float(cli.shadow_min_positive_support),
        "shadow_max_events_per_frame": int(cli.shadow_max_events_per_frame),
        "shadow_image_predictor_built": shadow_predictor is not None,
        "shadow_sam2_checkpoint": rel(shadow_checkpoint) if shadow_checkpoint is not None else "",
        "shadow_sam2_checkpoint_sha256": sha256_file(shadow_checkpoint) if shadow_checkpoint is not None else "",
        "shadow_build_runtime_sec": float(shadow_build_runtime_sec),
        "shadow_set_image_runtime_sec": float(shadow_set_image_runtime_sec),
        "shadow_predict_runtime_sec": float(shadow_predict_runtime_sec),
        "gap_output_max_bbox_frac": float(cli.gap_output_max_bbox_frac),
        "disable_gap_birth": bool(cli.disable_gap_birth),
        "gap_max_points": int(cli.gap_max_points),
        "gap_max_points_per_component": int(cli.gap_max_points_per_component),
        "gap_area_per_extra_point": int(cli.gap_area_per_extra_point),
        "gap_min_image_edge_distance_px": int(cli.gap_min_image_edge_distance_px),
        "gap_iou_threshold_override": None if cli.gap_iou_threshold is None else float(cli.gap_iou_threshold),
        "gap_stability_threshold_override": None
        if cli.gap_stability_threshold is None
        else float(cli.gap_stability_threshold),
        "gap_relaxed_min_uncovered_ratio": float(cli.gap_relaxed_min_uncovered_ratio),
        "gap_relaxed_iou_threshold": float(cli.gap_relaxed_iou_threshold),
        "gap_relaxed_stability_threshold": float(cli.gap_relaxed_stability_threshold),
        "gap_relaxed_min_clipped_area": int(cli.gap_relaxed_min_clipped_area),
        "gap_small_mask_max_area": int(cli.gap_small_mask_max_area),
        "gap_small_mask_min_pred_iou": float(cli.gap_small_mask_min_pred_iou),
        "gap_output_min_pred_iou": float(cli.gap_output_min_pred_iou),
        "gap_output_min_stability": float(cli.gap_output_min_stability),
        "gap_output_allow_relaxed": not bool(cli.gap_output_disallow_relaxed),
        "gap_delayed_admission": bool(cli.gap_delayed_admission),
        "gap_admission_min_pred_iou": float(cli.gap_admission_min_pred_iou),
        "gap_admission_min_stability": float(cli.gap_admission_min_stability),
        "gap_admission_allow_relaxed": not bool(cli.gap_admission_disallow_relaxed),
        "gap_reuse_recent_id_window": int(cli.gap_reuse_recent_id_window),
        "gap_reuse_recent_id_iou": float(cli.gap_reuse_recent_id_iou),
        "gap_reuse_recent_id_min_area": int(cli.gap_reuse_recent_id_min_area),
        "gap_large_reuse_recent_id_iou": float(cli.gap_large_reuse_recent_id_iou),
        "gap_large_reuse_min_area": int(cli.gap_large_reuse_min_area),
        "gap_large_reuse_max_area_ratio": float(cli.gap_large_reuse_max_area_ratio),
        "gap_anti_merge_core_window_frames": int(cli.gap_anti_merge_core_window_frames),
        "gap_anti_merge_core_erode_px": int(cli.gap_anti_merge_core_erode_px),
        "gap_anti_merge_core_min_area": int(cli.gap_anti_merge_core_min_area),
        "gap_anti_merge_core_min_overlap_px": int(cli.gap_anti_merge_core_min_overlap_px),
        "gap_anti_merge_core_min_overlap_ratio": float(cli.gap_anti_merge_core_min_overlap_ratio),
        "gap_anti_merge_max_overlap_objects": int(cli.gap_anti_merge_max_overlap_objects),
        "output_reuse_recent_id_window": int(cli.output_reuse_recent_id_window),
        "output_reuse_recent_id_iou": float(cli.output_reuse_recent_id_iou),
        "output_reuse_recent_id_min_area": int(cli.output_reuse_recent_id_min_area),
        "output_large_reuse_recent_id_iou": float(cli.output_large_reuse_recent_id_iou),
        "output_large_reuse_min_area": int(cli.output_large_reuse_min_area),
        "output_large_reuse_max_area_ratio": float(cli.output_large_reuse_max_area_ratio),
        "output_reuse_recent_id_preference": str(cli.output_reuse_recent_id_preference),
        "output_reuse_prevent_collision_union": bool(cli.output_reuse_prevent_collision_union),
        "output_fragment_max_area": int(cli.output_fragment_max_area),
        "output_fragment_suppress_max_area": int(cli.output_fragment_suppress_max_area),
        "output_fragment_merge_dilate_px": int(cli.output_fragment_merge_dilate_px),
        "output_fragment_merge_min_touch_px": int(cli.output_fragment_merge_min_touch_px),
        "output_fragment_merge_min_touch_ratio": float(cli.output_fragment_merge_min_touch_ratio),
        "output_fragment_merge_min_neighbor_area": int(cli.output_fragment_merge_min_neighbor_area),
        "gap_output_max_edge_touch_count": int(cli.gap_output_max_edge_touch_count),
        "gap_output_min_extent": float(cli.gap_output_min_extent),
        "gap_output_min_core_area_px": int(cli.gap_output_min_core_area_px),
        "gap_output_shape_min_uncovered_ratio": float(cli.gap_output_shape_min_uncovered_ratio),
        "gap_output_min_input_mask_count": int(cli.gap_output_min_input_mask_count),
        "gap_output_shape_gate_enabled": bool(
            float(cli.gap_output_max_bbox_frac) > 0.0
            or int(cli.gap_output_max_edge_touch_count) >= 0
            or float(cli.gap_output_min_extent) > 0.0
            or int(cli.gap_output_min_core_area_px) > 0
        ),
        "birth_transaction_enabled": bool(args.birth_transaction_enabled),
        "birth_admission_appearance_enabled": bool(args.birth_admission_appearance_enabled),
        "disable_birth_admission_appearance": bool(cli.disable_birth_admission_appearance),
        "birth_admission_max_uncovered_ratio": float(cli.birth_admission_max_uncovered_ratio),
        "birth_admission_max_bbox_frac": float(cli.birth_admission_max_bbox_frac),
        "birth_admission_max_edge_touch_count": int(cli.birth_admission_max_edge_touch_count),
        "birth_admission_min_extent": float(cli.birth_admission_min_extent),
        "birth_admission_min_core_area_px": int(cli.birth_admission_min_core_area_px),
        "birth_admission_shape_min_uncovered_ratio": float(cli.birth_admission_shape_min_uncovered_ratio),
        "birth_admission_shape_gate_enabled": bool(
            float(cli.birth_admission_max_bbox_frac) > 0.0
            or int(cli.birth_admission_max_edge_touch_count) >= 0
            or float(cli.birth_admission_min_extent) > 0.0
            or int(cli.birth_admission_min_core_area_px) > 0
        ),
        "stream_growth_prune_ratio": float(cli.stream_growth_prune_ratio),
        "stream_growth_prune_min_area": int(cli.stream_growth_prune_min_area),
        "stream_growth_prune_history": int(cli.stream_growth_prune_history),
        "stream_growth_prune_warmup": int(cli.stream_growth_prune_warmup),
        "stream_prune_protect_min_ever_area": int(cli.stream_prune_protect_min_ever_area),
        "stream_prune_protect_max_objects": int(cli.stream_prune_protect_max_objects),
        "stream_disjoin_policy": str(cli.stream_disjoin_policy),
        "stream_disjoin_claim_dropped": not bool(cli.stream_disjoin_claim_kept_only),
        "stream_disjoin_min_area_px": int(cli.stream_disjoin_min_area_px),
        "stream_disjoin_recent_min_iou": float(cli.stream_disjoin_recent_min_iou),
        "stream_disjoin_recent_max_area_growth": float(cli.stream_disjoin_recent_max_area_growth),
        "stream_oversized_prune_action": str(cli.stream_oversized_prune_action),
        "stream_growth_prune_action": str(cli.stream_growth_prune_action),
        "stream_growth_prune_max_history_median_area": int(
            cli.stream_growth_prune_max_history_median_area
        ),
        "disable_birth_transaction": bool(cli.disable_birth_transaction),
        "birth_transaction_min_pending": int(cli.birth_transaction_min_pending),
        "birth_transaction_max_delay_frames": int(cli.birth_transaction_max_delay_frames),
        "birth_transaction_immediate_area": int(cli.birth_transaction_immediate_area),
        "birth_transaction_min_total_area": int(cli.birth_transaction_min_total_area),
        "reactivation_probation_mode": str(cli.reactivation_probation_mode),
        "probation_output_mode": str(cli.probation_output_mode),
        "probation_min_positive_support": float(cli.probation_min_positive_support),
        "sam2_remove_object_called": any(row.get("record_type") == "demotion" and row.get("removed") for row in adapter_records),
        "sam2_add_new_points_or_box_called": bool(actual_video_readd_rows),
        "same_client_obj_id_readd_attempted": bool(actual_video_readd_rows),
        "actual_video_readd_record_count": int(len(actual_video_readd_rows)),
        "appearance_only_record_count": int(len(appearance_only_rows)),
        "appearance_only_output_record_count": int(len(appearance_only_output_rows)),
        "appearance_only_mean_score": mean(appearance_only_output_rows, "appearance_score"),
        "appearance_only_mean_margin": mean(appearance_only_output_rows, "appearance_margin"),
        "appearance_geometry_filter_record_count": int(len(appearance_geometry_rows)),
        "appearance_geometry_filter_output_record_count": int(len(appearance_geometry_output_rows)),
        "appearance_geometry_filter_mean_score": mean(appearance_geometry_output_rows, "appearance_score"),
        "appearance_geometry_filter_mean_margin": mean(appearance_geometry_output_rows, "appearance_margin"),
        "appearance_geometry_filter_mean_positive_support": mean(
            appearance_geometry_output_rows, "appearance_geometry_positive_support_rate"
        ),
        "appearance_geometry_filter_mean_negative_conflict": mean(
            appearance_geometry_output_rows, "appearance_geometry_negative_conflict_rate"
        ),
        "confirm_reprompt_requested": True,
        "event_count": int(len(events)),
        "probation_record_count": int(len(probation_rows)),
        "probation_output_mask_count": int(len(probation_output_rows)),
        "probation_skip_reasons": probation_skip_reasons,
        "probation_mean_iou": mean(probation_output_rows, "probation_iou_to_reference"),
        "diagnostic_probation_mean_iou_to_reference": mean(probation_output_rows, "probation_iou_to_reference"),
        "probation_iou_ge_0_5_rate": rate(probation_output_rows, "probation_iou_to_reference", 0.5),
        "probation_iou_ge_0_7_rate": rate(probation_output_rows, "probation_iou_to_reference", 0.7),
        "probation_mean_positive_point_support": mean(probation_output_rows, "positive_point_support_rate"),
        "probation_negative_sibling_overlap_rate_mean": mean(
            probation_output_rows, "probation_negative_sibling_overlap_rate"
        ),
        "shadow_record_count": int(len(shadow_rows)),
        "shadow_output_mask_count": int(len(shadow_output_rows)),
        "shadow_skip_reasons": shadow_skip_reasons,
        "shadow_mean_iou": mean(shadow_output_rows, "shadow_iou_to_reference"),
        "diagnostic_shadow_mean_iou_to_reference": mean(shadow_output_rows, "shadow_iou_to_reference"),
        "shadow_iou_ge_0_5_rate": rate(shadow_output_rows, "shadow_iou_to_reference", 0.5),
        "shadow_iou_ge_0_7_rate": rate(shadow_output_rows, "shadow_iou_to_reference", 0.7),
        "shadow_mean_positive_point_support": mean(shadow_output_rows, "positive_point_support_rate"),
        "shadow_negative_sibling_overlap_rate_mean": mean(shadow_output_rows, "shadow_negative_sibling_overlap_rate"),
        "attempt_record_count": int(len(attempt_rows)),
        "confirm_record_count": int(len(confirm_rows)),
        "selected_g2_count": int(len(selected_g2)),
        "selected_g2_rate": float(len(selected_g2) / max(len([*attempt_rows, *probation_rows]), 1)),
        "confirm_mean_iou": mean(confirm_rows, "confirm_iou_to_reference"),
        "diagnostic_confirm_mean_iou_to_reference": mean(confirm_rows, "confirm_iou_to_reference"),
        "confirm_iou_ge_0_5_rate": rate(confirm_rows, "confirm_iou_to_reference", 0.5),
        "confirm_iou_ge_0_7_rate": rate(confirm_rows, "confirm_iou_to_reference", 0.7),
        "confirm_negative_sibling_overlap_rate_mean": mean(confirm_rows, "confirm_negative_sibling_overlap_rate"),
        "confirm_negative_sibling_overlap_gt_0_1_rate": rate(
            confirm_rows, "confirm_negative_sibling_overlap_rate", 0.1, op="gt"
        ),
        "records_jsonl": rel(records_jsonl),
        "records_jsonl_sha256": sha256_file(records_jsonl),
        "records_csv": rel(records_csv),
        "records_csv_sha256": sha256_file(records_csv),
        "rolling_summary": rel(summary_path),
        "rolling_summary_sha256_before_adapter_update": sha256_file(summary_path),
        "rolling_stats": rolling_stats,
        "visual": visual,
        "visual_export_sec": visual_export_sec,
        "highres_visual_count": int(len(visual_paths)),
        "highres_visuals": [{"path": rel(path), "sha256": sha256_file(path)} for path in visual_paths],
        "highres_shadow_visual_count": int(len(shadow_visual_paths)),
        "highres_shadow_visuals": [
            {"path": rel(path), "sha256": sha256_file(path)} for path in shadow_visual_paths
        ],
        "highres_probation_visual_count": int(len(probation_visual_paths)),
        "highres_probation_visuals": [
            {"path": rel(path), "sha256": sha256_file(path)} for path in probation_visual_paths
        ],
        "audit_note": (
            "This run injects scheduled demotion/reactivation events into one real v106 rolling SAM2 stream. "
            "It is a scheduler integration smoke, not the final recoverability-aware full-scene policy or holdout."
        ),
    }
    adapter_summary_path = output_root / "g3_scheduler_summary.json"
    write_json(adapter_summary_path, adapter_summary)

    summary["v107_g3_scheduler_smoke"] = adapter_summary
    summary["v107_visual_confirmation_video"] = visual
    summary["visual_review_video_path"] = str(visual.get("path", ""))
    summary["visual_review_video_sha256"] = str(visual.get("sha256", ""))
    summary["visual_review_video_skipped"] = bool(visual.get("skipped", False))
    summary["v107_visual_export_runtime_sec"] = visual_export_sec
    summary["wrapper_total_with_v107_g3_wall_time_sec"] = float(time.time() - started)
    summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    adapter_summary["rolling_summary_sha256_after_adapter_update"] = sha256_file(summary_path)
    write_json(adapter_summary_path, adapter_summary)

    print(
        json.dumps(
            {
                "summary": str(adapter_summary_path),
                "rolling_summary": str(summary_path),
                "status": adapter_summary["status"],
                "event_count": adapter_summary["event_count"],
                "probation_output_mask_count": adapter_summary["probation_output_mask_count"],
                "shadow_output_mask_count": adapter_summary["shadow_output_mask_count"],
                "selected_g2_count": adapter_summary["selected_g2_count"],
                "confirm_mean_iou": adapter_summary["confirm_mean_iou"],
                "shadow_mean_iou": adapter_summary["shadow_mean_iou"],
                "confirm_negative_sibling_overlap_rate_mean": adapter_summary[
                    "confirm_negative_sibling_overlap_rate_mean"
                ],
                "wall_time": _format_seconds(time.time() - started),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
