#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_v105_fullscene_local2history_stitch import _boundary_matches, _read_label  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _local_to_output_global(curr_local: np.ndarray, curr_global: np.ndarray) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for local_id in [int(v) for v in np.unique(curr_local) if int(v) > 0]:
        mask = curr_local == int(local_id)
        global_vals, counts = np.unique(curr_global[mask], return_counts=True)
        pairs = [
            (int(global_id), int(count))
            for global_id, count in zip(global_vals.tolist(), counts.tolist())
            if int(global_id) > 0
        ]
        pairs.sort(key=lambda item: item[1], reverse=True)
        total = int(np.count_nonzero(mask))
        top_global = int(pairs[0][0]) if pairs else 0
        top_count = int(pairs[0][1]) if pairs else 0
        out[int(local_id)] = {
            "curr_local_id": int(local_id),
            "output_global_id": top_global,
            "local_area": total,
            "top_global_pixel_count": top_count,
            "top_global_pixel_ratio": float(top_count / total) if total > 0 else 0.0,
            "global_id_count_under_local": len(pairs),
            "global_ids_first10": [{"global_id": int(g), "pixels": int(c)} for g, c in pairs[:10]],
        }
    return out


def _row_by_local(rows: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict) or row.get("curr_local_id") is None:
            continue
        out[int(row["curr_local_id"])] = row
    return out


def _inheritance_sources(boundary: dict[str, Any]) -> dict[int, str]:
    sources: dict[int, str] = {}
    for row in boundary.get("accepted_matches_first40", []) or []:
        if isinstance(row, dict) and row.get("curr_local_id") is not None:
            sources[int(row["curr_local_id"])] = "overlap_continuation"
    audit = boundary.get("appearance_audit", {})
    if not isinstance(audit, dict):
        return sources
    for key, source in [
        ("accepted_matches_first40", "appearance_reappearance"),
        ("weak_overlap_overrides_first40", "weak_overlap_override"),
        ("part_merges_first40", "duplicate_part_merge"),
        ("tiny_lock_expansions_first40", "tiny_lock_expansion"),
    ]:
        for row in audit.get(key, []) or []:
            if isinstance(row, dict) and row.get("curr_local_id") is not None:
                sources[int(row["curr_local_id"])] = source
    return sources


def build(summary_path: Path, output_root: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    scene_rows = summary.get("scene_rows", [])
    if not isinstance(scene_rows, list):
        scene_rows = []

    for scene in scene_rows:
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id"))
        source_mask_dir = _resolve(str(scene["source_mask_dir"]))
        output_mask_dir = _resolve(str(scene["mask_dir"]))
        for boundary in scene.get("boundary_records", []) or []:
            if not isinstance(boundary, dict):
                continue
            prev_frame_id = int(boundary["prev_frame_id"])
            curr_frame_id = int(boundary["curr_frame_id"])
            prev_global = _read_label(output_mask_dir / f"{prev_frame_id}.png")
            curr_local = _read_label(source_mask_dir / f"{curr_frame_id}.png")
            curr_global = _read_label(output_mask_dir / f"{curr_frame_id}.png")
            _overlap_mapping, overlap_audit = _boundary_matches(
                prev_global=prev_global,
                curr_local=curr_local,
                min_iou=float(boundary.get("min_iou", 0.02)),
                min_overlap_min=float(boundary.get("min_overlap_min", 0.08)),
            )
            prev_frame_global_ids = set(int(v) for v in np.unique(prev_global) if int(v) > 0)
            local_to_global = _local_to_output_global(curr_local, curr_global)
            inheritance_sources = _inheritance_sources(boundary)
            large_rows = overlap_audit.get("large_unmatched_curr_first20", [])
            large_rows_full = [
                {"curr_local_id": int(row["curr_local_id"]), "area": int(row["area"])}
                for row in large_rows
                if isinstance(row, dict)
            ]
            inherited_large: list[dict[str, Any]] = []
            prev_frame_inherited_large: list[dict[str, Any]] = []
            new_large: list[dict[str, Any]] = []
            split_large: list[dict[str, Any]] = []
            for row in large_rows_full:
                local_id = int(row["curr_local_id"])
                mapping = local_to_global.get(local_id, {})
                out_global = int(mapping.get("output_global_id", 0) or 0)
                inheritance_source = inheritance_sources.get(local_id, "new_birth_or_untracked_source")
                inherited_from_prev_frame = bool(out_global in prev_frame_global_ids)
                inherited_from_history = bool(inherited_from_prev_frame or inheritance_source != "new_birth_or_untracked_source")
                enriched = {
                    **row,
                    "output_global_id": out_global,
                    "inherited_from_previous_global": inherited_from_prev_frame,
                    "inherited_from_previous_frame_global": inherited_from_prev_frame,
                    "inherited_from_history_global": inherited_from_history,
                    "inheritance_source": inheritance_source,
                    "top_global_pixel_ratio": mapping.get("top_global_pixel_ratio"),
                    "global_id_count_under_local": mapping.get("global_id_count_under_local"),
                }
                if int(mapping.get("global_id_count_under_local", 0) or 0) > 1:
                    split_large.append(enriched)
                if inherited_from_prev_frame:
                    prev_frame_inherited_large.append(enriched)
                if inherited_from_history:
                    inherited_large.append(enriched)
                else:
                    new_large.append(enriched)

            records.append(
                {
                    "scene_id": scene_id,
                    "prev_chunk_index": int(boundary["prev_chunk_index"]),
                    "curr_chunk_index": int(boundary["curr_chunk_index"]),
                    "prev_frame_id": prev_frame_id,
                    "curr_frame_id": curr_frame_id,
                    "overlap_large_unmatched_curr_count": int(overlap_audit.get("large_unmatched_curr_count", 0)),
                    "overlap_large_unmatched_curr_first20": large_rows_full,
                    "post_l2h_large_inherited_count_first20": len(inherited_large),
                    "post_l2h_large_prev_frame_inherited_count_first20": len(prev_frame_inherited_large),
                    "post_l2h_large_new_birth_count_first20": len(new_large),
                    "post_l2h_large_split_count_first20": len(split_large),
                    "post_l2h_weak_boundary_first20": bool(new_large or split_large),
                    "inherited_large_first20": inherited_large[:20],
                    "prev_frame_inherited_large_first20": prev_frame_inherited_large[:20],
                    "new_large_first20": new_large[:20],
                    "split_large_first20": split_large[:20],
                    "total_mapping_count": int(boundary.get("total_mapping_count", 0)),
                    "mapping_source_counts": boundary.get("mapping_source_counts", {}),
                    "appearance_audit_counts": {
                        "accepted_count": int((boundary.get("appearance_audit", {}) or {}).get("accepted_count", 0)),
                        "weak_overlap_override_count": int(
                            (boundary.get("appearance_audit", {}) or {}).get("weak_overlap_override_count", 0)
                        ),
                        "part_merge_count": int((boundary.get("appearance_audit", {}) or {}).get("part_merge_count", 0)),
                        "tiny_lock_expansion_count": int(
                            (boundary.get("appearance_audit", {}) or {}).get("tiny_lock_expansion_count", 0)
                        ),
                        "rejected_part_merge_by_object_witness_count": int(
                            (boundary.get("appearance_audit", {}) or {}).get(
                                "rejected_part_merge_by_object_witness_count", 0
                            )
                        ),
                        "rejected_tiny_lock_expansion_by_object_witness_count": int(
                            (boundary.get("appearance_audit", {}) or {}).get(
                                "rejected_tiny_lock_expansion_by_object_witness_count", 0
                            )
                        ),
                    },
                }
            )

    post_weak = [row for row in records if row["post_l2h_weak_boundary_first20"]]
    payload = {
        "schema_version": "stream4d_v105_post_l2h_boundary_identity_diagnostic_v2",
        "summary_path": _rel(summary_path),
        "summary_sha256": _sha256(summary_path),
        "variant_id": summary.get("variant_id"),
        "scene_count": len(scene_rows),
        "boundary_count": len(records),
        "post_l2h_weak_boundary_count_first20": len(post_weak),
        "post_l2h_nonweak_boundary_count_first20": len(records) - len(post_weak),
        "total_overlap_large_unmatched_curr_first20": int(
            sum(int(row["overlap_large_unmatched_curr_count"]) for row in records)
        ),
        "total_post_l2h_large_inherited_first20": int(
            sum(int(row["post_l2h_large_inherited_count_first20"]) for row in records)
        ),
        "total_post_l2h_large_prev_frame_inherited_first20": int(
            sum(int(row["post_l2h_large_prev_frame_inherited_count_first20"]) for row in records)
        ),
        "total_post_l2h_large_new_birth_first20": int(
            sum(int(row["post_l2h_large_new_birth_count_first20"]) for row in records)
        ),
        "total_post_l2h_large_split_first20": int(sum(int(row["post_l2h_large_split_count_first20"]) for row in records)),
        "records_json": _rel(output_root / "post_l2h_boundary_identity_records.json"),
        "claim_boundary": (
            "Non-visual diagnostic only. It proves which overlap-large-unmatched first-frame local IDs inherited a "
            "previous-frame or local2history reappearance global ID after remapping; it does not claim visual identity "
            "correctness or user final confirmation."
        ),
        "records": records,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v105 post-L2H boundary identity diagnostic.")
    parser.add_argument("--l2h-summary", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    summary_path = _resolve(args.l2h_summary)
    output_root = _resolve(args.output_root)
    payload = build(summary_path, output_root)
    records = payload.pop("records")
    _write_json(output_root / "post_l2h_boundary_identity_records.json", records)
    payload["records_sha256"] = _sha256(output_root / "post_l2h_boundary_identity_records.json")
    _write_json(output_root / "post_l2h_boundary_identity_summary.json", payload)
    _write_json(
        output_root / "hashes.json",
        {
            "summary_sha256": _sha256(output_root / "post_l2h_boundary_identity_summary.json"),
            "records_sha256": _sha256(output_root / "post_l2h_boundary_identity_records.json"),
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
