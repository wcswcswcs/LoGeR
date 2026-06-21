#!/usr/bin/env python3
"""Build a Phase8-style review for v78 TTT five-chunk visual probe panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_PROBE_ROOT = Path(
    "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase4_ttt_five_chunk_regime_visual_probe_v1"
)

CHUNK_NOTES: Dict[int, Dict[str, str]] = {
    64: {
        "status": "confirmed",
        "pattern": (
            "Five-chunk head shows narrow bright road corridor, hard shadow/exposure transition, "
            "road cracks, and wall/road-edge structure. TTT panels are nonblank and aligned to the "
            "same long-window regime-shift case."
        ),
        "geometry": "D_tok emphasizes upper road/corridor and wall/edge bands while the RGB shows strong illumination shift.",
        "random": "The diagnostic selected write/update role is structured along road/corridor bands; same-mass random is scattered.",
    },
    65: {
        "status": "confirmed",
        "pattern": (
            "Curved corridor and wall boundary continue, with road-surface cracks and shadow changes "
            "visible across sampled frames."
        ),
        "geometry": "D_tok and P_ttt_write remain spatially structured over road/corridor and boundary bands.",
        "random": "The diagnostic role mask remains contiguous around road/corridor structure, unlike the random mask.",
    },
    66: {
        "status": "confirmed",
        "pattern": (
            "Middle of the five-chunk window preserves the bend plus wall/vegetation boundary pattern; "
            "the TTT write prior and D_tok are not blank and vary with corridor geometry."
        ),
        "geometry": "D_tok highlights the road/corridor transition zone and boundary around the curve.",
        "random": "The diagnostic role mask follows road/edge structure more than the same-mass random scatter.",
    },
    67: {
        "status": "confirmed",
        "pattern": (
            "Late-window frames show stronger vegetation/wall-side occlusion and exposure variation; "
            "the diagnostic write role still concentrates on corridor/road-edge bands."
        ),
        "geometry": "D_tok remains structured along the road strip and right-side boundary/vegetation transition.",
        "random": "Same-mass random lacks the contiguous road/edge pattern seen in the diagnostic role mask.",
    },
    68: {
        "status": "confirmed",
        "pattern": (
            "Window tail transitions toward a more open bend/intersection-like view with continuing "
            "shadow and road-edge geometry shift; update/final panels remain separated and nonblank."
        ),
        "geometry": "D_tok shifts with the tail corridor/road-edge geometry and the update/final PCA panels remain spatially structured.",
        "random": "Write-role random contrast is available in the chunk068 write-role panel; update/final panels are not action masks.",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _chunk_from_visual_file(path: str) -> int:
    name = Path(path).name
    # chunk_064_TTT_update_term_L18.png
    return int(name.split("_")[1])


def _review_row(row: Dict[str, str]) -> Dict[str, Any]:
    visual_file = row["visual_file"]
    chunk = _chunk_from_visual_file(visual_file)
    note = CHUNK_NOTES.get(chunk, CHUNK_NOTES[64])
    visual_kind = row.get("visual_kind", "")
    output_name = row.get("output_name", "")
    if visual_kind == "TTT_write_role_mass_panel":
        action_mask_alignment = (
            "diagnostic write/update role mask is present; it is not a promoted runtime action mask"
        )
        random_difference = note["random"]
    elif visual_kind == "TTT_post_zp_delta_overlay":
        action_mask_alignment = "post-zp delta projection is present; not a raw fast-weight gradient or action mask"
        random_difference = "same-mass random contrast is carried by the paired write-role panel for this chunk"
    else:
        action_mask_alignment = "not an action-mask panel; operator/update/final output is separated for inspection"
        random_difference = "not applicable to this separated-output panel; use paired write-role panel"
    layer = row.get("layers", "")
    return {
        "visual_file": visual_file,
        "chunk_id": str(chunk),
        "frame_id": row.get("global_frames", ""),
        "tap": output_name,
        "layer": layer,
        "memory_body": "TTT_write_update_memory",
        "overlay_types": (
            "RGB;semantic;D_tok;TTT_operator/update/final/post_delta/write_role;"
            "same_write_mass_random_for_write_role"
        ),
        "review_status": note["status"],
        "visual_pattern_observed": f"{note['pattern']} Panel kind: {visual_kind}.",
        "semantic_alignment": (
            "Not a single semantic-category cue; road, wall/vegetation boundary, cracks, shadow, "
            "and corridor geometry jointly define the observed regime."
        ),
        "geometry_alignment": note["geometry"],
        "failure_alignment": (
            "Aligned with the user-prioritized rank5 five-chunk bad window; diagnostic only, "
            "with no window5_joint_sim3 improvement claim."
        ),
        "action_mask_alignment": action_mask_alignment,
        "random_mask_difference": random_difference,
        "reviewer_note": (
            "Manual/semiautomatic review 2026-06-21: use these panels to design a no-GT "
            "TTT update/freeze/protect hook, not as proof that the hook already works."
        ),
        "new_hypothesis_id": "HYP-PCA-REDISC-TTT-001",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, default=DEFAULT_PROBE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    probe_root = args.probe_root
    out_dir = args.out_dir or (probe_root / "five_chunk_probe_review_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = probe_root / "ttt_five_chunk_visual_probe_run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    manifest_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    for chunk in summary.get("chunks", []):
        manifest_path = probe_root / "visual_output_separated" / f"chunk{int(chunk):03d}" / "visual_artifact_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        for row in _read_csv(manifest_path):
            visual_path = Path(row["visual_file"])
            row = dict(row)
            row["chunk_id"] = str(chunk)
            row["sha256_recomputed"] = _sha256(visual_path) if visual_path.exists() else ""
            row["file_exists"] = bool(visual_path.exists())
            manifest_rows.append(row)
            review_rows.append(_review_row(row))

    _write_csv(
        out_dir / "visual_artifact_manifest.csv",
        manifest_rows,
        [
            "chunk_id",
            "visual_file",
            "visual_kind",
            "output_name",
            "layers",
            "global_frames",
            "operator_update_final_separated",
            "semantic_label_present",
            "D_geo_present",
            "D_geo_source",
            "post_zp_delta_present",
            "write_update_selected_role_present",
            "same_write_mass_random_present",
            "width",
            "height",
            "sha256",
            "sha256_recomputed",
            "image_intensity_std",
            "nonempty_image",
            "file_exists",
        ],
    )
    _write_csv(
        out_dir / "visual_review.csv",
        review_rows,
        [
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
        ],
    )

    all_files_exist = all(bool(row["file_exists"]) for row in manifest_rows)
    all_sha_match = all(row.get("sha256") == row.get("sha256_recomputed") for row in manifest_rows)
    all_nonempty = all(str(row.get("nonempty_image")).lower() == "true" for row in manifest_rows)
    all_visual_gate = bool(summary.get("all_visual_gates_pass"))
    audit = {
        "schema": "acl2_v78_ttt_five_chunk_visual_probe_review_integrity_v1",
        "probe_root": str(probe_root),
        "run_summary": str(summary_path),
        "seq": summary.get("seq"),
        "chunks": summary.get("chunks", []),
        "num_chunks": len(summary.get("chunks", [])),
        "num_visual_files": len(manifest_rows),
        "num_review_rows": len(review_rows),
        "all_files_exist": bool(all_files_exist),
        "all_sha256_match": bool(all_sha_match),
        "all_nonempty": bool(all_nonempty),
        "all_pipeline_jobs_ok": bool(summary.get("all_pipeline_jobs_ok")),
        "all_visual_gates_pass": bool(all_visual_gate),
        "review_status_counts": {
            status: sum(1 for row in review_rows if row["review_status"] == status)
            for status in sorted({row["review_status"] for row in review_rows})
        },
        "operator_update_final_separated_all": all(
            str(row.get("operator_update_final_separated")).lower() == "true" for row in manifest_rows
        ),
        "post_zp_delta_present_all": all(str(row.get("post_zp_delta_present")).lower() == "true" for row in manifest_rows),
        "same_write_mass_random_present_all": all(
            str(row.get("same_write_mass_random_present")).lower() == "true" for row in manifest_rows
        ),
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "runtime_promotion_blocker": (
            "visual probe shows TTT-specific artifacts but no implemented no-GT update/freeze/protect hook "
            "and no held-out window5_joint_sim3/downstream improvement gate"
        ),
        "visual_review_gate_pass": bool(
            all_files_exist
            and all_sha_match
            and all_nonempty
            and bool(summary.get("all_pipeline_jobs_ok"))
            and all_visual_gate
            and len(review_rows) >= len(manifest_rows)
        ),
    }
    (out_dir / "visual_integrity_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    insight = [
        "# v78 TTT Five-Chunk Visual Probe Review v1",
        "",
        "Status: diagnostic-only. This review does not claim a method gate or runtime promotion.",
        "",
        "Scope: KITTI02 rank5 five-chunk bad window, chunks 64-68.",
        "",
        "What changed relative to the earlier bad-good visual review:",
        "",
        "- The earlier review confirmed a long-window shadow/exposure/corridor/road-edge regime-shift pattern.",
        "- This probe now adds TTT-specific visual artifacts for that same window:",
        "  operator output, update term, final output, post-zp delta projection, and write-role/random panels.",
        "- All five chunks have separated TTT outputs and nonempty panels.",
        "",
        "Key observation:",
        "",
        "The TTT artifacts are spatially structured over the same corridor/road-edge regime-shift window, but this is still only evidence for where to design the next hook. It is not evidence that a hook improves trajectory metrics.",
        "",
        "Design implication:",
        "",
        "A next TTT candidate should detect five-chunk shadow/exposure plus road-center/corridor plus road-edge-confidence regime shift, then choose update / freeze / protect stable memory. It should not be a single semantic-class write-strength rule.",
        "",
        "Required before promotion:",
        "",
        "- Implement a no-GT long-window TTT hook or freeze/protect policy.",
        "- Compare against error-top, regime-score-top, low-error reference, same-write-mass random, and temporal-order controls.",
        "- Show held-out improvement in `window5_joint_sim3` or downstream future consistency.",
        "",
        f"Visual review gate pass: `{audit['visual_review_gate_pass']}`.",
        "Runtime promotion allowed: `False`.",
    ]
    (out_dir / "visual_insight.md").write_text("\n".join(insight) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "audit": audit}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
