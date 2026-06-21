#!/usr/bin/env python3
"""Audit whether current v78 artifacts contain direct Q/K/V alignment tensors.

This is an availability check, not a model run.  It distinguishes between:

* direct tensor artifacts that could support numeric Q/K/V or cache-current
  alignment,
* SWA overlap feature maps, which are useful but explicitly not direct Q/K/V,
* phase8 PCA visual clues, which are PNG/CSV review artifacts rather than
  tensors for the current Phase9 windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final"
)
PHASE9_ROOTS = [
    REPORT_ROOT / "phase9_swa_cache_value_carryover",
    Path(
        "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
        "phase9_swa_cache_value_carryover"
    ),
]
DEFAULT_OUT_DIR = REPORT_ROOT / "phase9_swa_cache_value_carryover/qkv_artifact_availability_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _tensor_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for suffix in ("*.pt", "*.pth", "*.npz"):
        out.extend(root.rglob(suffix))
    return sorted(out)


def _is_overlap_feature(path: Path) -> bool:
    text = str(path)
    return "/swa_overlap_feature_maps/" in text


def _is_materialized_mask(path: Path) -> bool:
    return "/selected_mask_materialization_v1/" in str(path)


def _phase8_roots() -> list[Path]:
    return sorted(path for path in REPORT_ROOT.glob("phase8_pca_rediscovery*") if path.is_dir())


def _phase8_visual_review_matches() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _phase8_roots():
        path = root / "visual_review.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                joined = " ".join(str(value) for value in row.values())
                if (
                    "pca_swa_current" in joined
                    or "pca_swa_cache" in joined
                    or "K/V" in joined
                    or "current/cache" in joined
                ):
                    rows.append(
                        {
                            "phase8_root": str(root),
                            "hypothesis_id": row.get("new_hypothesis_id", "")
                            or row.get("hypothesis_id", ""),
                            "feature_family": row.get("tap", "")
                            or row.get("feature_family", ""),
                            "layer": row.get("layer", ""),
                            "decision": row.get("review_status", "")
                            or row.get("decision", ""),
                            "review_note": (
                                row.get("reviewer_note", "")
                                or row.get("review_note", "")
                                or row.get("visual_pattern_observed", "")
                            )[:240],
                        }
                    )
    return rows


def _code_capabilities() -> dict[str, Any]:
    path = Path("run_pipeline_abc_v2.py")
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {
        "run_pipeline_abc_v2_exists": path.is_file(),
        "has_v68_export_full_pca_debug_flag": "--v68_export_full_pca_debug" in text,
        "mentions_swa_current_cache_qkv": "SWA current/cache q/k/v" in text,
        "has_swa_overlap_source_replace_target_kv": "--swa_overlap_source_replace_target" in text
        and '"kv"' in text,
        "has_swa_write_cache_blend_target": "--swa_write_cache_blend_target" in text,
    }


def main() -> None:
    args = parse_args()
    phase9_rows: list[dict[str, Any]] = []
    all_direct_candidates: list[Path] = []
    all_overlap_features: list[Path] = []
    all_materialized_masks: list[Path] = []
    for root in PHASE9_ROOTS:
        files = _tensor_files(root)
        overlap = [path for path in files if _is_overlap_feature(path)]
        masks = [path for path in files if _is_materialized_mask(path)]
        direct = [
            path
            for path in files
            if not _is_overlap_feature(path) and not _is_materialized_mask(path)
        ]
        all_overlap_features.extend(overlap)
        all_materialized_masks.extend(masks)
        all_direct_candidates.extend(direct)
        phase9_rows.append(
            {
                "root": str(root),
                "tensor_file_count": len(files),
                "swa_overlap_feature_tensor_count": len(overlap),
                "materialized_mask_tensor_count": len(masks),
                "direct_qkv_tensor_candidate_count": len(direct),
            }
        )

    phase8_tensor_files = [path for root in _phase8_roots() for path in _tensor_files(root)]
    visual_matches = _phase8_visual_review_matches()
    rows_csv = args.out_dir / "qkv_artifact_availability_rows.csv"
    visual_csv = args.out_dir / "phase8_visual_kv_clue_rows.csv"
    summary_json = args.out_dir / "qkv_artifact_availability_summary.json"
    _write_csv(rows_csv, phase9_rows)
    _write_csv(visual_csv, visual_matches)
    direct_available = bool(all_direct_candidates)
    _write_json(
        summary_json,
        {
            "schema": "acl2_v78_qkv_artifact_availability_v1",
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "phase9_roots": [str(root) for root in PHASE9_ROOTS],
            "phase9_direct_qkv_tensor_candidate_count": len(all_direct_candidates),
            "phase9_swa_overlap_feature_tensor_count": len(all_overlap_features),
            "phase9_materialized_mask_tensor_count": len(all_materialized_masks),
            "phase8_roots": [str(root) for root in _phase8_roots()],
            "phase8_tensor_file_count": len(phase8_tensor_files),
            "phase8_visual_kv_clue_count": len(visual_matches),
            "code_capabilities": _code_capabilities(),
            "rows_csv": str(rows_csv),
            "visual_kv_clue_rows_csv": str(visual_csv),
            "interpretation": [
                "Current Phase9 artifacts expose SWA overlap feature maps and reconstructed masks.",
                (
                    "Direct compact Q/K/V tensor candidates were found under current Phase9 roots."
                    if direct_available
                    else "No direct Q/K/V tensor candidate was found under current Phase9 roots."
                ),
                "Phase8 rediscovery contains visual K/V clues, but no tensor files in those directories.",
                (
                    "The code path can export broader Q/K/V PCA debug artifacts, and the current run now includes compact tiny-smoke dumps."
                    if direct_available
                    else "The code path can export broader Q/K/V PCA debug artifacts, but this was not present for the current Phase9 windows."
                ),
            ],
            "next_required_evidence": [
                (
                    "Summarize compact Q/K/V cross-overlap statistics across the tiny-smoke dumps."
                    if direct_available
                    else "Rerun a tiny selector smoke with v68_export_full_pca_debug enabled if direct numeric Q/K/V alignment is required."
                ),
                "Persist selected-mask-conditioned K/V statistics only after the token-grid mismatch is resolved.",
            ],
        },
    )
    print(json.dumps({"rows": str(rows_csv), "visual": str(visual_csv), "summary": str(summary_json)}, indent=2))


if __name__ == "__main__":
    main()
