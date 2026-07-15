#!/usr/bin/env python3
"""Diagnose temporal identity for v106 Phase7 child/residual admissions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_VARIANTS = [
    "S2_h2_area025_child_parentminus_p16_cap1_parent13_lcc075",
    "S2_h2_area025_child_replace_p16_cap1_parent13_lcc075",
    "S2_h2_area025_child_parentminus_p16_cap1_parent13_lcc090_reject",
]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    return label


def _best_reference_for_mask(mask: np.ndarray, reference_label: np.ndarray) -> Dict[str, Any]:
    area = int(mask.sum())
    if area <= 0:
        return {
            "present": False,
            "pred_area": 0,
            "best_reference_obj_id": None,
            "best_intersection": 0,
            "best_overlap_fraction_of_pred": 0.0,
            "best_reference_coverage": 0.0,
        }

    values, counts = np.unique(reference_label[mask], return_counts=True)
    pairs = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != 0]
    if not pairs:
        return {
            "present": True,
            "pred_area": area,
            "best_reference_obj_id": None,
            "best_intersection": 0,
            "best_overlap_fraction_of_pred": 0.0,
            "best_reference_coverage": 0.0,
        }

    best_label_value, best_intersection = max(pairs, key=lambda item: item[1])
    reference_area = int((reference_label == best_label_value).sum())
    return {
        "present": True,
        "pred_area": area,
        "best_reference_obj_id": int(best_label_value - 1),
        "best_reference_label_value": int(best_label_value),
        "best_intersection": int(best_intersection),
        "best_overlap_fraction_of_pred": float(best_intersection / area) if area else 0.0,
        "best_reference_coverage": float(best_intersection / reference_area) if reference_area else 0.0,
        "reference_area": reference_area,
    }


def _role_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = []
    for row in rows:
        role = row.get("frame0_child_split_role")
        if role and role not in {"untouched", "not_applicable"}:
            selected.append(row)
    selected.sort(key=lambda r: (int(r.get("frame_id", 0)), int(r.get("obj_id", -1))))
    return selected


def _record_key(row: Dict[str, Any]) -> str:
    role = row.get("frame0_child_split_role", "unknown")
    obj_id = row.get("obj_id", "na")
    parent = row.get("frame0_child_parent_raw_mask_index", "na")
    return f"{role}_obj{obj_id}_parent{parent}"


def _summarize_track(per_frame: List[Dict[str, Any]]) -> Dict[str, Any]:
    present = [row for row in per_frame if row["present"]]
    best_ids = [
        row["best_reference_obj_id"]
        for row in present
        if row.get("best_reference_obj_id") is not None
    ]
    counts = Counter(best_ids)
    dominant_id: Optional[int] = None
    dominant_count = 0
    if counts:
        dominant_id, dominant_count = counts.most_common(1)[0]
    mean_best = (
        float(np.mean([row["best_overlap_fraction_of_pred"] for row in present]))
        if present
        else 0.0
    )
    mean_ref_cov = (
        float(np.mean([row["best_reference_coverage"] for row in present]))
        if present
        else 0.0
    )
    return {
        "frame_count": len(per_frame),
        "present_frame_count": len(present),
        "first_present_frame": int(present[0]["frame_id"]) if present else None,
        "last_present_frame": int(present[-1]["frame_id"]) if present else None,
        "dominant_reference_obj_id": dominant_id,
        "dominant_reference_frame_count": int(dominant_count),
        "dominant_reference_present_fraction": float(dominant_count / len(present))
        if present
        else 0.0,
        "best_reference_obj_id_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "mean_best_overlap_fraction_of_pred": mean_best,
        "mean_best_reference_coverage": mean_ref_cov,
    }


def _metric_record_by_frame(metric_records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(row["frame_id"]): row for row in metric_records}


def _metric_sources_split_by_pair(
    metric_row: Dict[str, Any], child_obj_id: int, residual_obj_id: int
) -> List[int]:
    out = []
    source_map = metric_row.get("source_overlap_ids_thresholded") or {}
    for source_id, target_ids in source_map.items():
        target_set = {int(v) for v in target_ids}
        if child_obj_id in target_set and residual_obj_id in target_set:
            out.append(int(source_id))
    return sorted(out)


def _pair_groups(role_rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, Any], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in role_rows:
        parent = row.get("frame0_child_parent_raw_mask_index")
        stage = row.get("frame0_child_parent_stage")
        role = row.get("frame0_child_split_role")
        grouped[(parent, stage)][role] = row

    pairs = []
    for roles in grouped.values():
        child = roles.get("child")
        residual = roles.get("parent_residual")
        if child is not None and residual is not None:
            pairs.append((child, residual))
    pairs.sort(key=lambda item: int(item[0].get("frame0_child_parent_raw_mask_index", -1)))
    return pairs


def _diagnose_variant(repo_root: Path, variant_dir: Path) -> Dict[str, Any]:
    birth_path = variant_dir / "residual_birth_bank" / "residual_gap_birth_records.json"
    replay_path = variant_dir / "residual_replay" / "phase5_frozen_birth_replay_summary.json"
    metric_path = variant_dir / "metric_records_vs_reference.json"
    birth = _read_json(birth_path)
    replay = _read_json(replay_path)
    metric_records = _read_json(metric_path)
    metric_by_frame = _metric_record_by_frame(metric_records)
    role_rows = _role_rows(birth.get("rows", []))

    target_tracks = {}
    per_obj_frame_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for replay_row in replay.get("records", []):
        frame_id = int(replay_row["frame_id"])
        pred_label = _load_label(_resolve(repo_root, replay_row["label_path"]))
        reference_label = _load_label(_resolve(repo_root, replay_row["reference_label_path"]))
        for role_row in role_rows:
            obj_id = int(role_row["obj_id"])
            mask = pred_label == obj_id + 1
            record = _best_reference_for_mask(mask, reference_label)
            record.update({"frame_id": frame_id, "obj_id": obj_id})
            per_obj_frame_records[obj_id].append(record)

    for role_row in role_rows:
        obj_id = int(role_row["obj_id"])
        per_frame = per_obj_frame_records[obj_id]
        target_tracks[str(obj_id)] = {
            "birth_row": role_row,
            "track_summary": _summarize_track(per_frame),
            "per_frame": per_frame,
        }

    pair_records = []
    for child_row, residual_row in _pair_groups(role_rows):
        child_obj_id = int(child_row["obj_id"])
        residual_obj_id = int(residual_row["obj_id"])
        child_frames = {int(row["frame_id"]): row for row in per_obj_frame_records[child_obj_id]}
        residual_frames = {
            int(row["frame_id"]): row for row in per_obj_frame_records[residual_obj_id]
        }
        frame_records = []
        for frame_id in sorted(set(child_frames) & set(residual_frames)):
            child_frame = child_frames[frame_id]
            residual_frame = residual_frames[frame_id]
            both_present = bool(child_frame["present"] and residual_frame["present"])
            same_best_reference = (
                both_present
                and child_frame.get("best_reference_obj_id") is not None
                and child_frame.get("best_reference_obj_id")
                == residual_frame.get("best_reference_obj_id")
            )
            metric_row = metric_by_frame.get(frame_id, {})
            split_sources = _metric_sources_split_by_pair(
                metric_row, child_obj_id, residual_obj_id
            )
            frame_records.append(
                {
                    "frame_id": frame_id,
                    "both_present": both_present,
                    "child_best_reference_obj_id": child_frame.get("best_reference_obj_id"),
                    "child_pred_area": child_frame.get("pred_area"),
                    "child_best_overlap_fraction_of_pred": child_frame.get(
                        "best_overlap_fraction_of_pred"
                    ),
                    "residual_best_reference_obj_id": residual_frame.get(
                        "best_reference_obj_id"
                    ),
                    "residual_pred_area": residual_frame.get("pred_area"),
                    "residual_best_overlap_fraction_of_pred": residual_frame.get(
                        "best_overlap_fraction_of_pred"
                    ),
                    "same_best_reference": same_best_reference,
                    "metric_sources_split_by_child_and_residual": split_sources,
                    "CFR": metric_row.get("CFR"),
                    "CMR": metric_row.get("CMR"),
                }
            )
        both = [row for row in frame_records if row["both_present"]]
        same = [row for row in frame_records if row["same_best_reference"]]
        metric_split = [
            row
            for row in frame_records
            if row["metric_sources_split_by_child_and_residual"]
        ]
        pair_records.append(
            {
                "child_obj_id": child_obj_id,
                "residual_obj_id": residual_obj_id,
                "parent_raw_mask_index": child_row.get("frame0_child_parent_raw_mask_index"),
                "parent_stage": child_row.get("frame0_child_parent_stage"),
                "summary": {
                    "frame_count": len(frame_records),
                    "both_present_frame_count": len(both),
                    "same_best_reference_frame_count": len(same),
                    "metric_split_source_frame_count": len(metric_split),
                    "first_metric_split_frame": int(metric_split[0]["frame_id"])
                    if metric_split
                    else None,
                },
                "metric_split_examples": metric_split[:8],
                "same_best_reference_examples": same[:8],
                "per_frame": frame_records,
            }
        )

    top_cfr = max(metric_records, key=lambda row: float(row.get("CFR", 0.0)))
    top_cmr = max(metric_records, key=lambda row: float(row.get("CMR", 0.0)))
    return {
        "variant_dir": str(variant_dir),
        "input_files": {
            "birth_records": str(birth_path),
            "birth_records_sha256": _sha256_file(birth_path),
            "replay_summary": str(replay_path),
            "replay_summary_sha256": _sha256_file(replay_path),
            "metric_records": str(metric_path),
            "metric_records_sha256": _sha256_file(metric_path),
        },
        "role_row_count": len(role_rows),
        "target_tracks": target_tracks,
        "child_residual_pairs": pair_records,
        "metric_top_frames": {
            "top_cfr": {
                "frame_id": int(top_cfr["frame_id"]),
                "CFR": top_cfr.get("CFR"),
                "CMR": top_cfr.get("CMR"),
                "source_overlap_ids_thresholded": top_cfr.get(
                    "source_overlap_ids_thresholded"
                ),
                "target_overlap_ids_thresholded": top_cfr.get(
                    "target_overlap_ids_thresholded"
                ),
            },
            "top_cmr": {
                "frame_id": int(top_cmr["frame_id"]),
                "CFR": top_cmr.get("CFR"),
                "CMR": top_cmr.get("CMR"),
                "source_overlap_ids_thresholded": top_cmr.get(
                    "source_overlap_ids_thresholded"
                ),
                "target_overlap_ids_thresholded": top_cmr.get(
                    "target_overlap_ids_thresholded"
                ),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether v106 Phase7 frame0 child/residual admissions persist "
            "as independent identities or split the same reference object."
        )
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--root", required=True, help="Phase7 repair root.")
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant directory name under --root. Defaults to parent13 LCC controls.",
    )
    parser.add_argument("--output", required=True, help="Output diagnostic JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    root = _resolve(repo_root, args.root)
    variants = args.variant or DEFAULT_VARIANTS
    payload = {
        "schema_version": "stream4d_v106_phase7_temporal_identity_diagnostic_v1",
        "repo_root": str(repo_root),
        "root": str(root),
        "diagnostic_limits": [
            "Uses reference labels for attribution only; not used for prediction or tuning.",
            "Does not replace gate metrics; it explains CFR/CMR failure modes.",
            "Targets frame0 child/residual/fallback rows recorded in birth records.",
        ],
        "variants": {},
    }
    for variant in variants:
        payload["variants"][variant] = _diagnose_variant(repo_root, root / variant)
    _write_json(_resolve(repo_root, args.output), payload)

    for variant, result in payload["variants"].items():
        print(f"variant={variant} role_rows={result['role_row_count']}")
        for pair in result["child_residual_pairs"]:
            print(
                "  pair child={child} residual={residual} both={both} "
                "same_best_ref={same} metric_split={metric_split}".format(
                    child=pair["child_obj_id"],
                    residual=pair["residual_obj_id"],
                    both=pair["summary"]["both_present_frame_count"],
                    same=pair["summary"]["same_best_reference_frame_count"],
                    metric_split=pair["summary"]["metric_split_source_frame_count"],
                )
            )
        for obj_id, track in result["target_tracks"].items():
            summary = track["track_summary"]
            print(
                "  obj={obj} role={role} present={present}/{frames} dom_ref={dom} "
                "dom_frac={frac:.3f} mean_pred_overlap={overlap:.3f}".format(
                    obj=obj_id,
                    role=track["birth_row"].get("frame0_child_split_role"),
                    present=summary["present_frame_count"],
                    frames=summary["frame_count"],
                    dom=summary["dominant_reference_obj_id"],
                    frac=summary["dominant_reference_present_fraction"],
                    overlap=summary["mean_best_overlap_fraction_of_pred"],
                )
            )


if __name__ == "__main__":
    main()
