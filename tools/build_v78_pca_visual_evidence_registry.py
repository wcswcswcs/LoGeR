#!/usr/bin/env python3
"""Build the v78 PCA visual evidence registry.

This tool deliberately separates two ideas:

1. A layer/tap can have a real, nonempty PCA visual clue.
2. That clue is not action-ready unless the v78 overlays and controls exist.

The registry therefore preserves useful visual observations while marking
missing D_geo/future/action/random overlays as gate failures instead of
silently promoting old PCA screenshots into method evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


DEFAULT_PCA_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v76tf_c9_informed_semantic_tri_replay_memory_control/"
    "report_final/phase8_layer_feature_pca_visual_audit/full_qkv_smoke96_pca_rgb4views"
)
DEFAULT_TTT_DELTA_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v76tf_c9_informed_semantic_tri_replay_memory_control/"
    "report_final/phase12_ttt_write_delta_visual_audit_large"
)


CLUES: list[dict[str, Any]] = [
    {
        "clue_id": "V78-CLUE-GLOBAL-K-L07",
        "memory_body": "short_term_global_attention",
        "attention_type": "global_attn",
        "tap": "pca_attn_global_k_layers",
        "layer": 7,
        "component": "K",
        "review_status": "confirmed",
        "visual_pattern_observed": "clear road/sky/right-vegetation layout partition across sampled frames",
        "semantic_alignment": "road/sky/stuff regions broadly align with PCA bands",
        "geometry_alignment": "layout-like clue only; no D_geo overlay in source visual",
        "failure_alignment": "not evaluated because future/head-tail/scale overlays are missing",
        "action_mask_alignment": "not evaluated; no actual action mask panel",
        "random_mask_difference": "not evaluated; no same-mass/group-stratified random panel",
        "reviewer_note": (
            "Opened contact sheet with RGB/semantic/trust/PCA. It supports L07 K as layout selector/"
            "key-side prior, but not direct action or dynamic-object evidence."
        ),
        "allowed_actions": [
            "layout_region_select",
            "stable_harm_context_mask_construction",
            "key_side_trust_prior",
            "selector_for_L13_value_action",
        ],
        "forbidden_actions": [
            "direct_action_layer",
            "stable_dynamic_object_contour_claim",
            "direct_704F_promotion",
        ],
    },
    {
        "clue_id": "V78-CLUE-GLOBAL-V-L13",
        "memory_body": "short_term_global_attention",
        "attention_type": "global_attn",
        "tap": "pca_attn_global_v_layers",
        "layer": 13,
        "component": "V",
        "review_status": "confirmed",
        "visual_pattern_observed": "value-side road/structure/horizontal partition, noisier than L07 K",
        "semantic_alignment": "broad stuff/layout alignment; object contours are not stable",
        "geometry_alignment": "plausible value-propagation clue only; no D_geo overlay in source visual",
        "failure_alignment": "not evaluated because future/head-tail/scale overlays are missing",
        "action_mask_alignment": "not evaluated; no actual action mask panel",
        "random_mask_difference": "not evaluated; no same-mass/group-stratified random panel",
        "reviewer_note": (
            "Opened contact sheet with RGB/semantic/trust/PCA. It can motivate L13 source-side "
            "negative damp/stable protect tests, but numeric controls are mandatory."
        ),
        "allowed_actions": [
            "source_side_negative_damp",
            "stable_protect",
            "L07_guided_value_action",
            "short_term_semantic_memory_action",
        ],
        "forbidden_actions": [
            "success_claim_from_96F_or_256F_ATE_only",
            "704F_without_group_stratified_controls",
        ],
    },
    {
        "clue_id": "V78-CLUE-GLOBAL-K-L17",
        "memory_body": "short_term_global_attention",
        "attention_type": "global_attn",
        "tap": "pca_attn_global_k_layers",
        "layer": 17,
        "component": "K",
        "review_status": "confirmed",
        "visual_pattern_observed": "layout partition remains visible but is more fragmented than L07",
        "semantic_alignment": "coarse stuff alignment; dynamic/object boundaries are unstable",
        "geometry_alignment": "secondary validation clue only; no D_geo overlay in source visual",
        "failure_alignment": "not evaluated because future/head-tail/scale overlays are missing",
        "action_mask_alignment": "not evaluated; no actual action mask panel",
        "random_mask_difference": "not evaluated; no same-mass/group-stratified random panel",
        "reviewer_note": "Opened contact sheet. Use as secondary ablation, not primary semantic action layer.",
        "allowed_actions": ["secondary_validation", "layer_ablation"],
        "forbidden_actions": ["primary_semantic_action_layer"],
    },
    {
        "clue_id": "V78-CLUE-FRAME-V-L18",
        "memory_body": "short_term_frame_attention",
        "attention_type": "frame_attn",
        "tap": "pca_attn_frame_v_layers",
        "layer": 18,
        "component": "V",
        "review_status": "confirmed",
        "visual_pattern_observed": "local texture and horizontal/road-direction bands",
        "semantic_alignment": "weaker than global attention; broad stuff alignment only",
        "geometry_alignment": "tail-stabilization hypothesis only; no scale overlay in source visual",
        "failure_alignment": "not evaluated because head/tail scale overlay is missing",
        "action_mask_alignment": "not evaluated; no actual action mask panel",
        "random_mask_difference": "not evaluated; no same-mass random panel",
        "reviewer_note": "Opened contact sheet. This is not a main semantic path; only tail rebalance tests are allowed.",
        "allowed_actions": ["tail_stabilization", "head_to_tail_rebalance"],
        "forbidden_actions": ["mix_with_global_attention_write", "primary_semantic_path"],
    },
    {
        "clue_id": "V78-CLUE-SWA-CURRENT-Q-L18",
        "memory_body": "mid_term_swa",
        "attention_type": "swa_current",
        "tap": "pca_swa_current_q_layers",
        "layer": 18,
        "component": "Q",
        "review_status": "confirmed",
        "visual_pattern_observed": "coarse layout and geometry-corridor structure",
        "semantic_alignment": "road/right structure alignment visible; object boundaries not clean",
        "geometry_alignment": "route/handoff hypothesis only; no overlap residual overlay in source visual",
        "failure_alignment": "not evaluated because future_after_overlap overlay is missing",
        "action_mask_alignment": "not evaluated; no route mask panel",
        "random_mask_difference": "not evaluated; no same-route-mass random panel",
        "reviewer_note": (
            "Opened contact sheet. Supports mass-preserving SWA route/carry-over hypotheses, "
            "not hard remove or semantic success claims."
        ),
        "allowed_actions": [
            "mass_preserving_route",
            "stable_harm_context_carry_over",
            "context_floor",
        ],
        "forbidden_actions": [
            "hard_remove",
            "semantic_success_if_actual_loses_to_random",
        ],
    },
    {
        "clue_id": "V78-CLUE-SWA-CACHE-V-L18",
        "memory_body": "mid_term_swa",
        "attention_type": "swa_cache",
        "tap": "pca_swa_cache_v_layers",
        "layer": 18,
        "component": "V",
        "review_status": "ambiguous",
        "visual_pattern_observed": "prior artifact exists but not freshly opened in this Phase 0 pass",
        "semantic_alignment": "not reviewed in this pass",
        "geometry_alignment": "not reviewed in this pass",
        "failure_alignment": "not evaluated",
        "action_mask_alignment": "not evaluated",
        "random_mask_difference": "not evaluated",
        "reviewer_note": "Kept as registry candidate but not confirmed until a fresh visual review is performed.",
        "allowed_actions": ["mass_preserving_route", "stable_harm_context_carry_over"],
        "forbidden_actions": ["hard_remove"],
    },
    {
        "clue_id": "V78-CLUE-TTT-UPDATE-L18",
        "memory_body": "long_term_ttt",
        "attention_type": "ttt",
        "tap": "pca_ttt_update_term_layers",
        "layer": 18,
        "component": "update_term",
        "review_status": "confirmed",
        "visual_pattern_observed": "patchy update-term structure; not stable object contours",
        "semantic_alignment": "weak object-level alignment; coarse bands sometimes present",
        "geometry_alignment": "write/update strength audit only; no post-zp delta overlay in source visual",
        "failure_alignment": "not evaluated because future/head-tail/scale overlays are missing",
        "action_mask_alignment": "not evaluated; no write/update selected role panel",
        "random_mask_difference": "not evaluated; no same-write-mass random role panel",
        "reviewer_note": (
            "Opened contact sheet. TTT operator/update/final must stay separated. This does not support "
            "spatial semantic replacement."
        ),
        "allowed_actions": [
            "write_update_strength_modulation",
            "post_zp_delta_audit",
            "role_mass_audit",
        ],
        "forbidden_actions": [
            "TTT_spatial_semantic_replacement",
            "final_output_as_update_write_control_evidence",
        ],
    },
    {
        "clue_id": "V78-CLUE-TTT-OPERATOR-L18",
        "memory_body": "long_term_ttt",
        "attention_type": "ttt",
        "tap": "pca_ttt_operator_output_layers",
        "layer": 18,
        "component": "operator_output",
        "review_status": "ambiguous",
        "visual_pattern_observed": "prior separated artifact exists but not freshly opened in this Phase 0 pass",
        "semantic_alignment": "not reviewed in this pass",
        "geometry_alignment": "not reviewed in this pass",
        "failure_alignment": "not evaluated",
        "action_mask_alignment": "not evaluated",
        "random_mask_difference": "not evaluated",
        "reviewer_note": "Required for TTT output separation; keep as candidate but not confirmed yet.",
        "allowed_actions": ["operator_delta_gate", "output_separation_audit"],
        "forbidden_actions": ["spatial_semantic_replacement"],
    },
    {
        "clue_id": "V78-CLUE-TTT-FINAL-L18",
        "memory_body": "long_term_ttt",
        "attention_type": "ttt",
        "tap": "pca_ttt_final_output_layers",
        "layer": 18,
        "component": "final_output",
        "review_status": "ambiguous",
        "visual_pattern_observed": "prior separated artifact exists but final output is not a write/update control point",
        "semantic_alignment": "not reviewed in this pass",
        "geometry_alignment": "not reviewed in this pass",
        "failure_alignment": "not evaluated",
        "action_mask_alignment": "not evaluated",
        "random_mask_difference": "not evaluated",
        "reviewer_note": "Use only to prove output separation; do not treat as write/update control evidence.",
        "allowed_actions": ["output_separation_audit"],
        "forbidden_actions": ["final_output_as_update_write_control_evidence"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca-root", type=Path, default=DEFAULT_PCA_ROOT)
    parser.add_argument("--ttt-delta-root", type=Path, default=DEFAULT_TTT_DELTA_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "sha256": "",
            "width": 0,
            "height": 0,
            "image_intensity_std": 0.0,
            "nonempty_image": False,
        }
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    std = float(arr.std())
    return {
        "exists": True,
        "sha256": _sha256(path),
        "width": int(img.width),
        "height": int(img.height),
        "image_intensity_std": std,
        "nonempty_image": bool(img.width >= 512 and img.height >= 256 and std > 1.0),
    }


def _read_fit_tokens(pca_root: Path) -> dict[tuple[str, int], int]:
    metrics_path = pca_root / "auto_layer_metrics.csv"
    out: dict[tuple[str, int], int] = {}
    if not metrics_path.exists():
        return out
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[(row["tap"], int(row["layer"]))] = int(float(row.get("fit_tokens") or 0))
            except Exception:
                continue
    return out


def _contact_path(root: Path, clue: dict[str, Any]) -> Path:
    return root / "contact_sheets" / f"{clue['tap']}_L{int(clue['layer']):02d}_contact.png"


def _filmstrip_path(root: Path, clue: dict[str, Any]) -> Path:
    return root / "filmstrips" / f"{clue['tap']}_L{int(clue['layer']):02d}_filmstrip.png"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _manifest_rows(pca_root: Path, ttt_delta_root: Path, clue: dict[str, Any], fit_tokens: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    visual_specs = [
        ("contact_sheet", _contact_path(pca_root, clue), True, True, True),
        ("temporal_filmstrip", _filmstrip_path(pca_root, clue), False, False, False),
    ]
    if clue["clue_id"] == "V78-CLUE-TTT-UPDATE-L18":
        visual_specs.extend(
            [
                ("ttt_write_delta_projection_f000000", ttt_delta_root / "ttt_write_delta_pca_f000000.png", True, False, False),
                ("ttt_write_delta_projection_f000031", ttt_delta_root / "ttt_write_delta_pca_f000031.png", True, False, False),
                ("ttt_write_delta_projection_f000060", ttt_delta_root / "ttt_write_delta_pca_f000060.png", True, False, False),
                ("ttt_write_delta_projection_f000089", ttt_delta_root / "ttt_write_delta_pca_f000089.png", True, False, False),
                ("ttt_write_delta_projection_f000095", ttt_delta_root / "ttt_write_delta_pca_f000095.png", True, False, False),
            ]
        )
    for visual_type, path, has_rgb, has_sem, has_conf in visual_specs:
        stats = _image_stats(path)
        missing = []
        if not has_rgb:
            missing.append("RGB")
        if not has_sem:
            missing.append("semantic")
        if not has_conf:
            missing.append("confidence")
        missing.extend(["D_geo", "future", "action_mask", "same_mass_random", "group_stratified_random"])
        rows.append(
            {
                "clue_id": clue["clue_id"],
                "visual_type": visual_type,
                "visual_file": str(path),
                "sha256": stats["sha256"],
                "width": stats["width"],
                "height": stats["height"],
                "chunk_id": "multi",
                "frame_id": "multi",
                "tap": clue["tap"],
                "layer": clue["layer"],
                "pca_fit_sample_count": fit_tokens,
                "nonempty_image": stats["nonempty_image"],
                "image_intensity_std": f"{stats['image_intensity_std']:.6f}",
                "semantic_overlay_present": has_sem,
                "confidence_overlay_present": has_conf,
                "D_geo_overlay_present": False,
                "future_overlay_present": False,
                "action_mask_overlay_present": False,
                "random_mask_overlay_present": False,
                "missing_overlay_types": ";".join(missing),
                "review_status": clue["review_status"],
                "reviewer_note": clue["reviewer_note"],
            }
        )
    return rows


def _registry_row(clue: dict[str, Any], pca_root: Path, fit_tokens: int, manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    files_exist = all(Path(r["visual_file"]).exists() for r in manifest_rows)
    nonempty = all(str(r["nonempty_image"]) == "True" or r["nonempty_image"] is True for r in manifest_rows)
    overlays_complete = all(not r["missing_overlay_types"] for r in manifest_rows)
    confirmed = clue["review_status"] == "confirmed"
    return {
        "clue_id": clue["clue_id"],
        "memory_body": clue["memory_body"],
        "attention_type": clue["attention_type"],
        "tap": clue["tap"],
        "layer": clue["layer"],
        "component": clue["component"],
        "representative_contact_sheet": str(_contact_path(pca_root, clue)),
        "representative_filmstrip": str(_filmstrip_path(pca_root, clue)),
        "pca_fit_sample_count": fit_tokens,
        "review_status": clue["review_status"],
        "confirmed_layer_tap_visual_clue": bool(confirmed and files_exist and nonempty),
        "action_ready_under_v78": bool(confirmed and files_exist and nonempty and overlays_complete),
        "registry_status": (
            "action_ready"
            if confirmed and files_exist and nonempty and overlays_complete
            else "visual_clue_confirmed_but_v78_action_overlays_missing"
            if confirmed and files_exist and nonempty
            else "not_confirmed_or_missing_visual"
        ),
        "visual_pattern_observed": clue["visual_pattern_observed"],
        "allowed_actions": ";".join(clue["allowed_actions"]),
        "forbidden_actions": ";".join(clue["forbidden_actions"]),
    }


def _review_rows(clue: dict[str, Any], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in manifest_rows:
        rows.append(
            {
                "visual_file": r["visual_file"],
                "chunk_id": r["chunk_id"],
                "frame_id": r["frame_id"],
                "tap": clue["tap"],
                "layer": clue["layer"],
                "memory_body": clue["memory_body"],
                "overlay_types": "RGB;semantic;confidence;PCA"
                if r["semantic_overlay_present"]
                else "PCA_only_or_partial_projection",
                "review_status": clue["review_status"],
                "visual_pattern_observed": clue["visual_pattern_observed"],
                "semantic_alignment": clue["semantic_alignment"],
                "geometry_alignment": clue["geometry_alignment"],
                "failure_alignment": clue["failure_alignment"],
                "action_mask_alignment": clue["action_mask_alignment"],
                "random_mask_difference": clue["random_mask_difference"],
                "reviewer_note": clue["reviewer_note"],
                "new_hypothesis_id": "",
            }
        )
    return rows


def _write_markdown(out_dir: Path, registry_rows: list[dict[str, Any]], integrity: dict[str, Any]) -> None:
    allowed_lines = [
        "# v78 Allowed Actions By Layer",
        "",
        f"Phase 0 gate pass: `{integrity['gate_pass']}`",
        "",
        "Important: `action_ready_under_v78=false` means the layer/tap clue can guide new visual/action experiments,",
        "but it cannot be used to claim method success until required overlays and controls exist.",
        "",
    ]
    forbidden_lines = ["# v78 Forbidden Actions By Layer", ""]
    for row in registry_rows:
        allowed_lines.extend(
            [
                f"## {row['clue_id']}",
                "",
                f"- Memory body: `{row['memory_body']}`",
                f"- Tap/layer: `{row['tap']}` / L{int(row['layer']):02d}",
                f"- Review status: `{row['review_status']}`",
                f"- Registry status: `{row['registry_status']}`",
                f"- Action-ready under v78: `{row['action_ready_under_v78']}`",
                f"- Allowed actions: `{row['allowed_actions']}`",
                "",
            ]
        )
        forbidden_lines.extend(
            [
                f"## {row['clue_id']}",
                "",
                f"- Tap/layer: `{row['tap']}` / L{int(row['layer']):02d}",
                f"- Forbidden actions: `{row['forbidden_actions']}`",
                "",
            ]
        )
    (out_dir / "allowed_actions_by_layer.md").write_text("\n".join(allowed_lines), encoding="utf-8")
    (out_dir / "forbidden_actions_by_layer.md").write_text("\n".join(forbidden_lines), encoding="utf-8")


def _write_insight(out_dir: Path, registry_rows: list[dict[str, Any]], integrity: dict[str, Any]) -> None:
    confirmed = [r for r in registry_rows if r["confirmed_layer_tap_visual_clue"]]
    action_ready = [r for r in registry_rows if r["action_ready_under_v78"]]
    lines = [
        "# v78 Phase 0 Visual Insight",
        "",
        "## Summary",
        "",
        f"- Confirmed layer/tap visual clues: {len(confirmed)}",
        f"- Action-ready clues under full v78 overlay requirements: {len(action_ready)}",
        f"- Phase 0 gate pass: `{integrity['gate_pass']}`",
        "",
        "The existing v76/v68 visual artifacts are useful for rediscovering layer/tap hypotheses, but they do not yet",
        "contain the full v78 action/failure overlays. Therefore they may guide Phase 1 visual/action generation,",
        "but they cannot by themselves prove a memory-control candidate.",
        "",
        "## Confirmed Visual Patterns",
        "",
    ]
    for row in confirmed:
        lines.extend(
            [
                f"### {row['clue_id']}",
                "",
                f"- Tap/layer: `{row['tap']}` / L{int(row['layer']):02d}",
                f"- Pattern: {row['visual_pattern_observed']}",
                f"- Registry status: `{row['registry_status']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Required Repair Before Action Interpretation",
            "",
            "- Generate v78 overlay panels with D_geo / future or head-tail-scale failure / action mask / same-mass random / group-stratified random.",
            "- Keep frame/global/SWA/TTT taps separate; do not merge frame attention and global attention evidence.",
            "- Keep TTT operator_output, update_term, and final_output separate.",
            "- Re-run visual integrity audit after overlay generation.",
        ]
    )
    (out_dir / "visual_insight.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit_tokens_by_key = _read_fit_tokens(args.pca_root)

    manifest_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    for clue in CLUES:
        fit_tokens = fit_tokens_by_key.get((clue["tap"], int(clue["layer"])), 0)
        clue_manifest = _manifest_rows(args.pca_root, args.ttt_delta_root, clue, fit_tokens)
        manifest_rows.extend(clue_manifest)
        review_rows.extend(_review_rows(clue, clue_manifest))
        registry_rows.append(_registry_row(clue, args.pca_root, fit_tokens, clue_manifest))

    sha_values = [r["sha256"] for r in manifest_rows if r["sha256"]]
    duplicate_count = len(sha_values) - len(set(sha_values))
    all_files_exist = all(Path(r["visual_file"]).exists() for r in manifest_rows)
    all_sha = all(bool(r["sha256"]) for r in manifest_rows)
    all_dims = all(int(r["width"]) > 0 and int(r["height"]) > 0 for r in manifest_rows)
    all_nonempty = all(bool(r["nonempty_image"]) for r in manifest_rows)
    missing_overlay_count = sum(1 for r in manifest_rows if r["missing_overlay_types"])
    invalid_visual_count = sum(1 for r in manifest_rows if not bool(r["nonempty_image"]))
    review_coverage = (len(review_rows) / max(1, len(manifest_rows)))
    gate_pass = bool(
        all_files_exist
        and all_sha
        and all_dims
        and all_nonempty
        and duplicate_count == 0
        and review_coverage >= 0.8
        and missing_overlay_count == 0
        and any(r["action_ready_under_v78"] for r in registry_rows)
    )
    integrity = {
        "schema": "acl2_v78_visual_integrity_audit_v1",
        "num_visual_files": len(manifest_rows),
        "num_manifest_rows": len(manifest_rows),
        "num_review_rows": len(review_rows),
        "review_coverage": review_coverage,
        "all_files_exist": all_files_exist,
        "all_sha256_present": all_sha,
        "all_dimensions_present": all_dims,
        "all_nonempty": all_nonempty,
        "duplicate_image_count": duplicate_count,
        "missing_overlay_count": missing_overlay_count,
        "invalid_visual_count": invalid_visual_count,
        "gate_pass": gate_pass,
        "gate_status": "pass" if gate_pass else "invalid_visual_confirmation_requires_v78_overlays",
        "pca_root": str(args.pca_root),
        "ttt_delta_root": str(args.ttt_delta_root),
    }

    manifest_fields = [
        "clue_id",
        "visual_type",
        "visual_file",
        "sha256",
        "width",
        "height",
        "chunk_id",
        "frame_id",
        "tap",
        "layer",
        "pca_fit_sample_count",
        "nonempty_image",
        "image_intensity_std",
        "semantic_overlay_present",
        "confidence_overlay_present",
        "D_geo_overlay_present",
        "future_overlay_present",
        "action_mask_overlay_present",
        "random_mask_overlay_present",
        "missing_overlay_types",
        "review_status",
        "reviewer_note",
    ]
    registry_fields = [
        "clue_id",
        "memory_body",
        "attention_type",
        "tap",
        "layer",
        "component",
        "representative_contact_sheet",
        "representative_filmstrip",
        "pca_fit_sample_count",
        "review_status",
        "confirmed_layer_tap_visual_clue",
        "action_ready_under_v78",
        "registry_status",
        "visual_pattern_observed",
        "allowed_actions",
        "forbidden_actions",
    ]
    review_fields = [
        "visual_file",
        "chunk_id",
        "frame_id",
        "tap",
        "layer",
        "memory_body",
        "overlay_types",
        "review_status",
        "visual_pattern_observed",
        "semantic_alignment",
        "geometry_alignment",
        "failure_alignment",
        "action_mask_alignment",
        "random_mask_difference",
        "reviewer_note",
        "new_hypothesis_id",
    ]
    _write_csv(args.out_dir / "visual_file_manifest.csv", manifest_rows, manifest_fields)
    _write_csv(args.out_dir / "pca_visual_registry.csv", registry_rows, registry_fields)
    _write_csv(args.out_dir / "visual_review.csv", review_rows, review_fields)
    (args.out_dir / "pca_visual_registry.json").write_text(
        json.dumps(
            {
                "schema": "acl2_v78_pca_visual_registry_v1",
                "integrity": integrity,
                "registry": registry_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "visual_integrity_audit.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(args.out_dir, registry_rows, integrity)
    _write_insight(args.out_dir, registry_rows, integrity)
    print(json.dumps(integrity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
