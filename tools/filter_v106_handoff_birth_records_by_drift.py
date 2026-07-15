#!/usr/bin/env python3
"""Filter v106 handoff birth records using non-oracle early replay drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def _resolve(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64, copy=False)


def _label_map(repo_root: Path, replay_summary: dict[str, Any]) -> dict[int, Path]:
    labels: dict[int, Path] = {}
    for record in replay_summary.get("records", []):
        if "frame_id" not in record or "label_path" not in record:
            continue
        labels[int(record["frame_id"])] = _resolve(repo_root, str(record["label_path"]))
    return labels


def _source_area_by_obj(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_obj: dict[int, dict[str, Any]] = {}
    for row in rows:
        obj_id = int(row["obj_id"])
        area = int(row.get("mask_area", 0) or 0)
        entry = by_obj.setdefault(obj_id, {"source_max_area": 0, "source_rows": []})
        entry["source_max_area"] = max(int(entry["source_max_area"]), area)
        entry["source_rows"].append(
            {
                "frame_id": int(row.get("frame_id", -1)),
                "chunk_frame_index": int(row.get("chunk_frame_index", -1)),
                "source_overlap_index": row.get("source_overlap_index"),
                "mask_area": area,
            }
        )
    return by_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop inherited handoff objects that grow abruptly in the first "
            "non-overlap frames of their own replay. This uses only predicted "
            "handoff replay labels and the source handoff masks, not reference "
            "labels or GT."
        )
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--birth-records", required=True, help="Input handoff birth_records.json.")
    parser.add_argument("--handoff-replay-summary", required=True, help="Replay summary for the unfiltered handoff.")
    parser.add_argument("--output", required=True, help="Filtered birth_records.json.")
    parser.add_argument("--overlap", type=int, default=3, help="Number of overlap frames at chunk start.")
    parser.add_argument(
        "--probe-frame-count",
        type=int,
        default=2,
        help="How many non-overlap frames after overlap to inspect.",
    )
    parser.add_argument(
        "--growth-threshold",
        type=float,
        default=1.35,
        help="Drop when max probe area / max source handoff area is at least this value.",
    )
    parser.add_argument(
        "--min-probe-area",
        type=int,
        default=20000,
        help="Drop only if the max early non-overlap area is at least this many pixels.",
    )
    parser.add_argument(
        "--audit-output",
        default="",
        help="Optional separate audit JSON path. Defaults to output sibling with .audit.json suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    birth_path = _resolve(repo_root, args.birth_records)
    replay_path = _resolve(repo_root, args.handoff_replay_summary)
    output_path = _resolve(repo_root, args.output)
    audit_path = (
        _resolve(repo_root, args.audit_output)
        if str(args.audit_output).strip()
        else output_path.with_suffix(output_path.suffix + ".drift_audit.json")
    )

    birth = _read_json(birth_path)
    replay = _read_json(replay_path)
    rows = [dict(row) for row in birth.get("rows", [])]
    frame_ids = [int(v) for v in birth.get("frame_ids", [])]
    if not frame_ids:
        frame_ids = [int(v) for v in replay.get("frame_ids", [])]
    if len(frame_ids) <= int(args.overlap):
        raise ValueError("frame_ids must include non-overlap frames")
    probe_frame_ids = frame_ids[int(args.overlap) : int(args.overlap) + int(args.probe_frame_count)]
    if not probe_frame_ids:
        raise ValueError("probe_frame_count selected no frames")

    labels = _label_map(repo_root, replay)
    missing_probe_labels = [int(fid) for fid in probe_frame_ids if int(fid) not in labels]
    if missing_probe_labels:
        raise FileNotFoundError(f"missing replay labels for probe frames: {missing_probe_labels}")

    source_by_obj = _source_area_by_obj(rows)
    probe_labels = {int(fid): _load_label(labels[int(fid)]) for fid in probe_frame_ids}
    drift_records = []
    drop_obj_ids: set[int] = set()
    for obj_id, source_entry in sorted(source_by_obj.items()):
        source_max_area = int(source_entry["source_max_area"])
        probe_areas = {
            int(frame_id): int(np.count_nonzero(label == int(obj_id) + 1))
            for frame_id, label in probe_labels.items()
        }
        max_probe_area = max(probe_areas.values()) if probe_areas else 0
        growth_ratio = float(max_probe_area / max(1, source_max_area))
        drop = bool(
            max_probe_area >= int(args.min_probe_area)
            and growth_ratio >= float(args.growth_threshold)
        )
        if drop:
            drop_obj_ids.add(int(obj_id))
        drift_records.append(
            {
                "obj_id": int(obj_id),
                "source_max_area": int(source_max_area),
                "source_rows": source_entry["source_rows"],
                "probe_frame_ids": [int(v) for v in probe_frame_ids],
                "probe_areas": {str(k): int(v) for k, v in sorted(probe_areas.items())},
                "max_probe_area": int(max_probe_area),
                "growth_ratio_vs_source_max": float(growth_ratio),
                "dropped": bool(drop),
                "reason": (
                    "early_nonoverlap_area_growth_exceeds_threshold"
                    if drop
                    else "kept"
                ),
            }
        )

    kept_rows = [row for row in rows if int(row["obj_id"]) not in drop_obj_ids]
    payload = dict(birth)
    payload["schema_version"] = str(birth.get("schema_version", "unknown")) + "+handoff_drift_filter_v1"
    payload["source_birth_records"] = str(birth_path)
    payload["source_birth_records_sha256"] = _sha256_file(birth_path)
    payload["source_handoff_replay_summary"] = str(replay_path)
    payload["source_handoff_replay_summary_sha256"] = _sha256_file(replay_path)
    payload["handoff_drift_filter_policy"] = {
        "overlap": int(args.overlap),
        "probe_frame_count": int(args.probe_frame_count),
        "probe_frame_ids": [int(v) for v in probe_frame_ids],
        "growth_threshold": float(args.growth_threshold),
        "min_probe_area": int(args.min_probe_area),
        "uses_reference_labels": False,
        "uses_ground_truth": False,
        "uses_predicted_handoff_replay_labels": True,
    }
    payload["handoff_drift_filter_records"] = drift_records
    payload["handoff_drift_filter_dropped_obj_ids"] = [int(v) for v in sorted(drop_obj_ids)]
    payload["handoff_drift_filter_dropped_count"] = int(len(drop_obj_ids))
    payload["filtered_from_row_count"] = int(len(rows))
    payload["row_count"] = int(len(kept_rows))
    payload["rows"] = kept_rows

    audit = {
        "schema_version": "stream4d_v106_handoff_drift_filter_audit_v1",
        "birth_records": str(birth_path),
        "birth_records_sha256": _sha256_file(birth_path),
        "handoff_replay_summary": str(replay_path),
        "handoff_replay_summary_sha256": _sha256_file(replay_path),
        "output": str(output_path),
        "policy": payload["handoff_drift_filter_policy"],
        "row_count_before": int(len(rows)),
        "row_count_after": int(len(kept_rows)),
        "dropped_obj_ids": [int(v) for v in sorted(drop_obj_ids)],
        "records": drift_records,
    }
    _write_json(output_path, payload)
    _write_json(audit_path, audit)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": _sha256_file(output_path),
                "audit": str(audit_path),
                "audit_sha256": _sha256_file(audit_path),
                "row_count_before": int(len(rows)),
                "row_count_after": int(len(kept_rows)),
                "dropped_obj_ids": [int(v) for v in sorted(drop_obj_ids)],
                "policy": payload["handoff_drift_filter_policy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
