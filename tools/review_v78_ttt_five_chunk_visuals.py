#!/usr/bin/env python3
"""Write a focused visual review for v78 TTT five-chunk bad/reference panels.

This is a diagnostic review over existing bad-vs-reference panels. It does not
claim a method gate, and it explicitly records that these panels are not a full
Phase8 PCA/action-vs-random overlay set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "bad_good_case_contrast/v2_unique_scenes_top5"
)
DEFAULT_REGIME_ROWS = DEFAULT_ROOT / "ttt_five_chunk_regime_signal_v1/ttt_five_chunk_regime_signal_rows.csv"
DEFAULT_VISUAL_MANIFEST = DEFAULT_ROOT / "visual_artifact_manifest.csv"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "ttt_five_chunk_visual_review_v1"


REVIEW_NOTES = {
    "5": {
        "review_status": "confirmed",
        "visual_pattern_observed": (
            "Bad window shows the clearest long-window regime shift: narrow curving corridor, "
            "strong shadow/exposure changes, road cracks, and wall/tree/road-edge changes "
            "across the sampled five chunks. Reference still has urban objects, but the "
            "corridor and visibility regime are more stable."
        ),
        "semantic_alignment": (
            "Not a single semantic-category cue; road, wall/tree boundary, shadow, and "
            "surface texture jointly define the pattern."
        ),
        "geometry_alignment": (
            "Consistent with high road-center/road-edge regime score and 8/9 expected-direction votes."
        ),
        "failure_alignment": (
            "Failure is not the largest RMSE case, but it is the strongest match to the "
            "TTT long-window regime-shift hypothesis."
        ),
        "reviewer_note": (
            "Manual visual review 2026-06-21: prioritize this as TTT regime-shift-top probe."
        ),
    },
    "2": {
        "review_status": "ambiguous",
        "visual_pattern_observed": (
            "Bad window shows strong shadow/exposure variation plus wall/tree boundary and "
            "road-surface crack changes; corridor shift is present but less decisive than rank5."
        ),
        "semantic_alignment": "Multi-factor road/corridor context; not a single object class.",
        "geometry_alignment": "Moderate positive TTT regime score with 6/9 expected-direction votes.",
        "failure_alignment": "High RMSE and visible regime change agree, but score is weaker than rank5.",
        "reviewer_note": "Manual visual review 2026-06-21: keep as second-priority TTT probe.",
    },
    "3": {
        "review_status": "ambiguous",
        "visual_pattern_observed": (
            "Bad window continues the curved-road/wall boundary sequence, but some illumination "
            "features are mixed and the signal is weaker than rank5/rank2."
        ),
        "semantic_alignment": "Road, wall/tree boundary, and surface texture matter jointly.",
        "geometry_alignment": "Positive but modest TTT regime score with 7/9 expected-direction votes.",
        "failure_alignment": "Supports regime-shift hypothesis weakly; should not be promoted alone.",
        "reviewer_note": "Manual visual review 2026-06-21: keep as supporting but not primary evidence.",
    },
    "1": {
        "review_status": "ambiguous",
        "visual_pattern_observed": (
            "Largest RMSE case shows bend/wall/vegetation boundary changes, but the long-window "
            "shadow/exposure/corridor-shift combination is less concentrated than rank5."
        ),
        "semantic_alignment": "Not explained by an individual semantic class.",
        "geometry_alignment": "Only weak positive TTT regime score with 5/9 expected-direction votes.",
        "failure_alignment": (
            "Important high-error case, but it may include non-regime failure factors."
        ),
        "reviewer_note": (
            "Manual visual review 2026-06-21: include as error-top control against overfitting the regime score."
        ),
    },
    "4": {
        "review_status": "rejected",
        "visual_pattern_observed": (
            "Bad window has stronger vehicles/urban objects and local occlusions, but not the "
            "same continuous narrow-corridor shadow/exposure regime shift as rank5."
        ),
        "semantic_alignment": (
            "This is a useful negative example: urban objects alone are not the TTT regime-shift cue."
        ),
        "geometry_alignment": "Negative TTT regime score with 4/9 expected-direction votes.",
        "failure_alignment": (
            "High error remains real, but this visual pattern should not define the TTT regime-shift action."
        ),
        "reviewer_note": (
            "Manual visual review 2026-06-21: use as non-regime high-error contrast."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime-rows", type=Path, default=DEFAULT_REGIME_ROWS)
    parser.add_argument("--visual-manifest", type=Path, default=DEFAULT_VISUAL_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_stats(path: Path) -> dict[str, Any]:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    return {
        "width": int(image.width),
        "height": int(image.height),
        "image_intensity_std": float(arr.std()),
        "nonempty_image": bool(arr.std() > 1.0),
    }


def _manifest_by_rank(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out = {}
    for row in rows:
        if row.get("family") == "five_chunk":
            out[row.get("contrast_rank", "")] = row
    return out


def _review_rows(
    regime_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_by_rank = _manifest_by_rank(manifest_rows)
    reviews: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for row in regime_rows:
        rank = row.get("contrast_rank", "")
        note = REVIEW_NOTES.get(rank, REVIEW_NOTES["1"])
        manifest = manifest_by_rank.get(rank, {})
        visual_file = manifest.get("visual_file") or row.get("visual_file", "")
        visual_path = Path(visual_file)
        actual_sha = _sha256(visual_path) if visual_path.exists() else ""
        stats = _image_stats(visual_path) if visual_path.exists() else {
            "width": 0,
            "height": 0,
            "image_intensity_std": 0.0,
            "nonempty_image": False,
        }
        artifact = {
            "visual_file": visual_file,
            "family": "five_chunk",
            "contrast_rank": rank,
            "bad_case": row.get("bad_case", ""),
            "reference_case": row.get("reference_case", ""),
            "bad_metric_value": row.get("bad_metric_value", ""),
            "reference_metric_value": row.get("reference_metric_value", ""),
            "ttt_regime_shift_score": row.get("ttt_regime_shift_score", ""),
            "expected_direction_vote_count": row.get("expected_direction_vote_count", ""),
            "expected_direction_vote_rate": row.get("expected_direction_vote_rate", ""),
            "sha256": actual_sha,
            "manifest_sha256": manifest.get("sha256", ""),
            "rgb_overlay_present": manifest.get("rgb_overlay_present", ""),
            "semantic_overlay_present": manifest.get("semantic_overlay_present", ""),
            "confidence_overlay_present": manifest.get("confidence_overlay_present", ""),
            "trajectory_metric_present": manifest.get("trajectory_metric_present", ""),
            "pca_overlay_present": "False",
            "action_mask_overlay_present": "False",
            "same_mass_random_overlay_present": "False",
            "group_stratified_random_overlay_present": "False",
            **stats,
        }
        artifacts.append(artifact)
        hypothesis_id = (
            "HYP-PCA-REDISC-TTT-002"
            if note["review_status"] == "rejected"
            else "HYP-PCA-REDISC-TTT-001"
        )
        reviews.append(
            {
                "visual_file": visual_file,
                "chunk_id": row.get("bad_case", ""),
                "frame_id": "five_chunk_window",
                "tap": "bad_good_case_contrast",
                "layer": "NA",
                "memory_body": "TTT_long_window_regime",
                "overlay_types": "RGB;semantic;confidence;trajectory_metric",
                "review_status": note["review_status"],
                "visual_pattern_observed": note["visual_pattern_observed"],
                "semantic_alignment": note["semantic_alignment"],
                "geometry_alignment": note["geometry_alignment"],
                "failure_alignment": note["failure_alignment"],
                "action_mask_alignment": (
                    "not available in these bad-good contrast panels; must generate action-vs-random "
                    "or TTT write-score overlays before claiming an action rule"
                ),
                "random_mask_difference": (
                    "not available in these bad-good contrast panels; this review cannot claim "
                    "actual-vs-random action-mask separation"
                ),
                "reviewer_note": note["reviewer_note"],
                "new_hypothesis_id": hypothesis_id,
            }
        )
    return reviews, artifacts


def _write_insight(path: Path, reviews: list[dict[str, Any]]) -> None:
    status_counts = {
        status: sum(1 for row in reviews if row["review_status"] == status)
        for status in sorted({row["review_status"] for row in reviews})
    }
    text = [
        "# v78 TTT Five-Chunk Visual Review v1",
        "",
        "Status: diagnostic-only. This review does not claim a method gate.",
        "",
        "Scope: existing five-chunk bad-vs-reference panels from `v2_unique_scenes_top5`, reviewed against the user-highlighted long-window shadow/exposure/corridor/road-edge regime-shift hypothesis.",
        "",
        f"Review status counts: {json.dumps(status_counts, sort_keys=True)}",
        "",
        "Key findings:",
        "",
        "- Rank5 / chunks 64-65-66-67-68 is the clearest visual match to the long-window regime-shift hypothesis, despite being only fifth by RMSE.",
        "- Rank1 / chunks 63-64-65-66-67 remains the largest-error case, but its visual/regime score is weaker; keep it as an error-top control.",
        "- Rank4 / KITTI00 chunks 138-142 is a useful negative contrast: more vehicles and urban objects do not by themselves define the TTT regime-shift cue.",
        "- References can contain urban objects, walls, buildings, or vehicles; the visual difference is corridor/road-edge/visibility stability, not object presence.",
        "",
        "Limitations:",
        "",
        "- These panels contain RGB, semantic, confidence, and trajectory metric views.",
        "- They do not contain PCA overlays, TTT write-score maps, action selected masks, same-mass random masks, or group-stratified random masks.",
        "- Therefore this is not a full Phase8 visual gate pass and must not be promoted into a runtime TTT action.",
        "",
        "Next hypothesis:",
        "",
        "TTT should first detect five-chunk shadow/exposure plus road-center/corridor plus road-edge-confidence regime shift, then decide whether to update, freeze, or protect stable memory. Any action must be validated on held-out five-chunk windows with controls.",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _write_hypothesis_bank(path: Path) -> None:
    text = """# v78 TTT Five-Chunk Hypothesis Bank v1

Status: diagnostic-only. No method gate is claimed.

## HYP-PCA-REDISC-TTT-001

Memory body: TTT write/update memory.

Tap/layer: not yet a layer claim; current evidence is bad-vs-reference visual/regime contrast.

Visual evidence files:

- `bad_good_case_panels/five_chunk_5_02_kitti02_c3_v4full_64-65-66-67-68.png`
- `bad_good_case_panels/five_chunk_2_02_kitti02_c3_v4full_62-63-64-65-66.png`
- `bad_good_case_panels/five_chunk_1_02_kitti02_c3_v4full_63-64-65-66-67.png`

Observed visual pattern:

Five-chunk bad windows can show continuous shadow/exposure change, narrow or curved road corridor, road cracks, wall/tree boundaries, and road-edge changes. Reference windows may still contain urban objects but have more stable corridor/visibility.

Why previous action failed:

Previous SWA route-bias actions found weak local/boundary signals but did not amplify to phase9 key metrics. The five-chunk failure pattern looks like long-window regime shift rather than a single semantic-category issue.

New action point:

TTT should test an update/freeze/protect decision conditioned on no-GT long-window regime shift, not a class-specific write strength.

Expected mechanism metric:

`window5_joint_sim3_rmse_m` and downstream future consistency should improve on held-out five-chunk windows without harming adjacent/overlap controls.

Required controls:

Error-top windows, regime-score-top windows, low-error references, shuffled temporal order if available, and no chunk-id-specific runtime policy.

Required visual outputs:

TTT write-score map, candidate-native cosine map, update-term/final-output panels, and same-mass/random contrast for selected update regions.

Stop rule:

If held-out five-chunk windows do not improve or visual review fails to show any TTT-specific write/update pattern, keep this as diagnostic-only.

## HYP-PCA-REDISC-TTT-002

Memory body: TTT / failure taxonomy.

Tap/layer: non-regime high-error contrast.

Visual evidence files:

- `bad_good_case_panels/five_chunk_4_00_kitti00_c3_v4full_138-139-140-141-142.png`

Observed visual pattern:

Vehicles and urban objects can be present in high-error windows without matching the shadow/exposure/corridor regime-shift signature.

Why previous action failed:

A semantic-class-only TTT rule risks overfitting object presence rather than long-window geometry/visibility stability.

New action point:

Use non-regime high-error windows as negative controls for any TTT regime detector.

Expected mechanism metric:

A correct detector should not over-trigger on object presence alone.

Required controls:

Low-error urban references and non-regime high-error windows.

Required visual outputs:

Same as HYP-PCA-REDISC-TTT-001.

Stop rule:

If the detector cannot separate regime-score-top from object-heavy non-regime cases, do not promote to runtime.
"""
    path.write_text(text, encoding="utf-8")


def _write_failed_questions(path: Path) -> None:
    rows = [
        {
            "failed_phase": "phase4_ttt_write_update_and_phase9_swa_followup",
            "failed_candidate": "prior TTT semantic/write actions and SWA route-bias candidates",
            "failure_reason": "weak local/boundary signals did not become method gate; five-chunk failures suggest long-window regime shift",
            "old_layer": "NA",
            "old_tap": "semantic/category write strength or SWA route mass",
            "old_action": "class/route amplification",
            "old_visual_evidence_file": "bad_good_case_contrast/v2_unique_scenes_top5/five_chunk_ttt_visual_focus_index.md",
            "what_visual_evidence_was_missing": "TTT write-score/update-term/final-output overlays and action-vs-random masks",
            "new_visual_question": "Does TTT write/update react to five-chunk shadow/exposure/corridor/road-edge regime shift rather than object class?",
            "new_tap_or_layer_to_dump": "TTT operator_output;update_term;final_output;post_zp_delta;write_score_map;candidate_native_cosine_map",
            "new_overlay_required": "RGB;semantic;confidence;road-edge/corridor;TTT write-score;actual selected update;same-mass random;group-stratified random",
            "new_candidate_hypothesis": "HYP-PCA-REDISC-TTT-001",
        }
    ]
    _write_csv(path, rows, list(rows[0].keys()))


def _write_integrity(path: Path, artifacts: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> None:
    all_exist = all(Path(row["visual_file"]).exists() for row in artifacts)
    sha_match = all(row["sha256"] and row["sha256"] == row["manifest_sha256"] for row in artifacts)
    all_nonempty = all(bool(row["nonempty_image"]) for row in artifacts)
    existing_overlay_pass = all(
        str(row[col]).lower() == "true"
        for row in artifacts
        for col in [
            "rgb_overlay_present",
            "semantic_overlay_present",
            "confidence_overlay_present",
            "trajectory_metric_present",
        ]
    )
    missing_full_phase8 = [
        "pca_overlay_present",
        "action_mask_overlay_present",
        "same_mass_random_overlay_present",
        "group_stratified_random_overlay_present",
    ]
    summary = {
        "schema": "acl2_v78_ttt_five_chunk_visual_review_integrity_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "num_visual_files": len(artifacts),
        "num_review_rows": len(reviews),
        "all_visual_files_exist": all_exist,
        "sha256_matches_source_manifest": sha_match,
        "all_nonempty": all_nonempty,
        "existing_bad_good_panel_overlay_gate_pass": existing_overlay_pass,
        "full_phase8_pca_action_overlay_gate_pass": False,
        "full_phase8_missing_overlay_types": missing_full_phase8,
        "review_status_counts": {
            status: sum(1 for row in reviews if row["review_status"] == status)
            for status in sorted({row["review_status"] for row in reviews})
        },
        "existing_bad_good_panel_integrity_pass": bool(
            all_exist and sha_match and all_nonempty and existing_overlay_pass
        ),
        "gate_pass": False,
        "promotion_allowed": False,
        "promotion_blocker": (
            "existing panels lack PCA/action-vs-random/TTT write-score overlays; "
            "review is diagnostic-only"
        ),
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    regime_rows = _read_csv(args.regime_rows)
    manifest_rows = _read_csv(args.visual_manifest)
    reviews, artifacts = _review_rows(regime_rows, manifest_rows)

    artifact_fields = list(artifacts[0].keys())
    review_fields = list(reviews[0].keys())
    _write_csv(args.out_dir / "visual_artifact_manifest.csv", artifacts, artifact_fields)
    _write_csv(args.out_dir / "visual_review.csv", reviews, review_fields)
    _write_insight(args.out_dir / "visual_insight.md", reviews)
    _write_hypothesis_bank(args.out_dir / "new_hypothesis_bank.md")
    _write_failed_questions(args.out_dir / "failed_action_to_visual_question.csv")
    _write_integrity(args.out_dir / "visual_integrity_audit.json", artifacts, reviews)

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "visual_review": str(args.out_dir / "visual_review.csv"),
                "visual_integrity_audit": str(args.out_dir / "visual_integrity_audit.json"),
                "diagnostic_only": True,
                "method_gate_claimed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
