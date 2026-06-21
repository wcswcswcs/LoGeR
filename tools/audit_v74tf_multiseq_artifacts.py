#!/usr/bin/env python3
"""Phase 0 multi-sequence artifact audit for ACL2 v74-TF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v73_semantic_memory_common import utc_now, write_csv, write_json
from v74tf_common import (
    DEFAULT_SEQS,
    PREPROCESS_ROOT,
    REPORT_ROOT,
    boolish,
    directory_file_count,
    discover_geometry_for_seq,
    discover_radio_dirs,
    label_role_coverage,
    parse_seqs,
    read_semantic_metrics,
    read_stage_summary,
    seq_preprocess_dir,
    stage_cache_dir,
    stage_chunk_dirs,
)


def _seq_row(preprocess_root: Path, seq: str) -> dict[str, Any]:
    seq_root = seq_preprocess_dir(preprocess_root, seq)
    sparse_pt = seq_root / "sparse_masklets_with_semantic.pt"
    metrics = read_semantic_metrics(preprocess_root, seq)
    stage_dir = stage_cache_dir(preprocess_root, seq)
    stage_summary = read_stage_summary(preprocess_root, seq)
    chunk_dirs = stage_chunk_dirs(stage_dir)
    radio_dirs = discover_radio_dirs(preprocess_root, seq)
    radio_file_count = sum(directory_file_count(path) for path in radio_dirs)
    geom = discover_geometry_for_seq(seq)
    has_label_maps = boolish(metrics.get("semantic_format")) or boolish(stage_summary.get("has_semantic_segmentation"))
    has_confidence = boolish(metrics.get("has_confidence_maps")) or boolish(stage_summary.get("has_semantic_confidence"))
    stage_cache_pass = stage_dir.exists() and boolish(stage_summary.get("has_semantic_segmentation")) and len(chunk_dirs) > 0
    baseline_pass = bool(geom.get("baseline_trajectory_available"))
    per_chunk_geometry_pass = (
        bool(geom.get("merge_trace_available"))
        or bool(geom.get("per_chunk_geometry_available"))
        or bool(geom.get("overlap_pairs_available"))
    )
    semantic_action_allowed = bool(has_label_maps and stage_cache_pass and baseline_pass and per_chunk_geometry_pass)
    radio_available = bool(radio_dirs and radio_file_count > 0)
    role = label_role_coverage(metrics)
    row = {
        "seq": seq,
        "seq_root": str(seq_root),
        "seq_root_exists": seq_root.exists(),
        "sparse_semantic_pt": str(sparse_pt),
        "sparse_semantic_pt_exists": sparse_pt.exists(),
        "semantic_label_maps_available": has_label_maps,
        "semantic_confidence_available": has_confidence,
        "semantic_num_labels": metrics.get("num_labels", stage_summary.get("semantic_num_labels")),
        "semantic_num_frames": metrics.get("num_frames", stage_summary.get("num_frames")),
        "stage_c_cache_dir": str(stage_dir),
        "stage_c_cache_exists": stage_dir.exists(),
        "stage_c_chunk_dirs": len(chunk_dirs),
        "stage_c_num_chunks_summary": stage_summary.get("num_chunks"),
        "stage_c_has_semantic_segmentation": boolish(stage_summary.get("has_semantic_segmentation")),
        "stage_c_has_semantic_confidence": boolish(stage_summary.get("has_semantic_confidence")),
        "stage_c_cache_pass": stage_cache_pass,
        "radio_sidecar_dirs": ";".join(str(path) for path in radio_dirs),
        "radio_sidecar_dir_count": len(radio_dirs),
        "radio_sidecar_file_count": radio_file_count,
        "radio_sidecar_available": radio_available,
        "radio_specific_action_allowed": radio_available and semantic_action_allowed,
        "ttt_post_zp_spatial_delta_available": False,
        "ttt_action_allowed": False,
        "semantic_conditioned_action_allowed": semantic_action_allowed,
        "seq_status": "semantic_conditioned_action_allowed" if semantic_action_allowed else "diagnostic_only_or_preprocessing_needed",
        **role,
        **geom,
    }
    if not has_confidence:
        row["confidence_note"] = "seq_confidence_unavailable; confidence-aware semantic main candidate disallowed"
    elif not semantic_action_allowed:
        row["confidence_note"] = "confidence exists but sequence lacks required geometry/action artifacts"
    else:
        row["confidence_note"] = "confidence-aware diagnostics allowed"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqs", default=",".join(DEFAULT_SEQS))
    parser.add_argument("--preprocess-root", type=Path, default=PREPROCESS_ROOT)
    parser.add_argument("--results-root", type=Path, default=Path("results/kitti01_hmc_v2"))
    parser.add_argument("--out-dir", type=Path, default=REPORT_ROOT / "phase0_multiseq_artifact_audit")
    args = parser.parse_args()

    seqs = parse_seqs(args.seqs)
    rows = [_seq_row(args.preprocess_root, seq) for seq in seqs]
    semantic_rows = [
        {
            key: row.get(key)
            for key in (
                "seq",
                "sparse_semantic_pt_exists",
                "semantic_label_maps_available",
                "semantic_confidence_available",
                "semantic_num_frames",
                "semantic_num_labels",
                "stage_c_cache_pass",
                "stage_c_chunk_dirs",
                "confidence_note",
            )
        }
        for row in rows
    ]
    radio_rows = [
        {
            key: row.get(key)
            for key in (
                "seq",
                "radio_sidecar_available",
                "radio_sidecar_dir_count",
                "radio_sidecar_file_count",
                "radio_specific_action_allowed",
                "radio_sidecar_dirs",
            )
        }
        for row in rows
    ]
    geometry_rows = [
        {
            key: row.get(key)
            for key in (
                "seq",
                "gt_exists",
                "baseline_trajectory_available",
                "merge_trace_available",
                "per_chunk_geometry_available",
                "per_chunk_geometry_count",
                "per_chunk_geometry_dir",
                "overlap_pairs_available",
                "overlap_pair_count",
                "geometry_artifact_status",
                "baseline_run_dir",
                "baseline_txt",
            )
        }
        for row in rows
    ]
    allowed_paths = {
        row["seq"]: {
            "semantic_conditioned_action_allowed": bool(row.get("semantic_conditioned_action_allowed")),
            "radio_specific_action_allowed": bool(row.get("radio_specific_action_allowed")),
            "ttt_action_allowed": bool(row.get("ttt_action_allowed")),
            "seq_status": row.get("seq_status"),
            "required_blockers": [
                name
                for name, ok in (
                    ("semantic_label_maps_available", row.get("semantic_label_maps_available")),
                    ("stage_c_cache_pass", row.get("stage_c_cache_pass")),
                    ("baseline_trajectory_available", row.get("baseline_trajectory_available")),
                    (
                        "per_chunk_geometry_or_overlap_available",
                        row.get("merge_trace_available")
                        or row.get("per_chunk_geometry_available")
                        or row.get("overlap_pairs_available"),
                    ),
                )
                if not ok
            ],
        }
        for row in rows
    }
    summary = {
        "schema": "acl2_v74tf_phase0_multiseq_artifact_audit_v1",
        "created_at": utc_now(),
        "seqs": seqs,
        "preprocess_root": str(args.preprocess_root),
        "results_root": str(args.results_root),
        "semantic_conditioned_action_allowed_seqs": [row["seq"] for row in rows if row.get("semantic_conditioned_action_allowed")],
        "radio_specific_action_allowed_seqs": [row["seq"] for row in rows if row.get("radio_specific_action_allowed")],
        "ttt_action_allowed_seqs": [row["seq"] for row in rows if row.get("ttt_action_allowed")],
        "phase0_gate_pass": "01" in [row["seq"] for row in rows if row.get("semantic_conditioned_action_allowed")],
        "cross_seq_blocker": {
            row["seq"]: allowed_paths[row["seq"]]["required_blockers"]
            for row in rows
            if allowed_paths[row["seq"]]["required_blockers"]
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "artifact_availability_by_seq.csv", rows)
    write_csv(args.out_dir / "semantic_cache_status_by_seq.csv", semantic_rows)
    write_csv(args.out_dir / "radio_sidecar_status_by_seq.csv", radio_rows)
    write_csv(args.out_dir / "geometry_artifact_status_by_seq.csv", geometry_rows)
    write_json(args.out_dir / "allowed_paths_by_seq.json", allowed_paths)
    write_json(args.out_dir / "artifact_audit_summary.json", summary)
    print(
        {
            "out_dir": str(args.out_dir),
            "phase0_gate_pass": summary["phase0_gate_pass"],
            "semantic_conditioned_action_allowed_seqs": summary["semantic_conditioned_action_allowed_seqs"],
            "cross_seq_blocker": summary["cross_seq_blocker"],
        }
    )


if __name__ == "__main__":
    main()
