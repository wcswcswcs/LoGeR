#!/usr/bin/env python3
"""Filter v106 birth records using non-oracle replay presence diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

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
    return label


def _presence_by_obj(repo_root: Path, replay_summary: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    presence: Dict[int, Dict[str, Any]] = {}
    for record in replay_summary.get("records", []):
        frame_id = int(record["frame_id"])
        label = _load_label(_resolve(repo_root, record["label_path"]))
        values, counts = np.unique(label, return_counts=True)
        for value, count in zip(values, counts):
            label_value = int(value)
            if label_value <= 0:
                continue
            obj_id = int(label_value - 1)
            entry = presence.setdefault(
                obj_id,
                {
                    "present_frame_ids": [],
                    "areas": [],
                },
            )
            entry["present_frame_ids"].append(frame_id)
            entry["areas"].append(int(count))
    for obj_id, entry in presence.items():
        frames: List[int] = [int(v) for v in entry["present_frame_ids"]]
        areas: List[int] = [int(v) for v in entry["areas"]]
        entry["present_frame_count"] = int(len(frames))
        entry["first_present_frame"] = int(frames[0]) if frames else None
        entry["last_present_frame"] = int(frames[-1]) if frames else None
        entry["mean_area"] = float(np.mean(areas)) if areas else 0.0
        entry["max_area"] = int(max(areas)) if areas else 0
    return presence


def _should_consider(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if str(row.get("phase5_role")) != "birth_new":
        return False
    if args.anchor_chunk_index is not None and int(row.get("chunk_frame_index", -1)) != int(args.anchor_chunk_index):
        return False
    if args.protect_frame0_parent_original and str(row.get("frame0_child_split_role")) in {
        "parent_original",
        "parent_original_no_child_fallback",
    }:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a filtered birth-record bank by dropping short-lived new births "
            "according to an existing replay's predicted labels. This is a control "
            "for tentative/defer policies, not a GT-based filter."
        )
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--birth-records", required=True, help="Input residual birth records JSON.")
    parser.add_argument("--replay-summary", required=True, help="Replay summary used to measure presence.")
    parser.add_argument("--output", required=True, help="Filtered birth records output JSON.")
    parser.add_argument("--min-present-frames", type=int, required=True)
    parser.add_argument(
        "--anchor-chunk-index",
        type=int,
        default=0,
        help="Only filter birth_new rows anchored at this chunk index. Use -1 for all anchors.",
    )
    parser.add_argument(
        "--protect-frame0-parent-original",
        action="store_true",
        default=False,
        help="Never drop parent_original or parent_original_no_child_fallback rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    birth_path = _resolve(repo_root, args.birth_records)
    replay_path = _resolve(repo_root, args.replay_summary)
    output_path = _resolve(repo_root, args.output)
    if int(args.anchor_chunk_index) < 0:
        args.anchor_chunk_index = None

    birth = _read_json(birth_path)
    replay = _read_json(replay_path)
    presence = _presence_by_obj(repo_root, replay)

    kept_rows = []
    filter_records = []
    for row in birth.get("rows", []):
        obj_id = int(row["obj_id"])
        entry = presence.get(
            obj_id,
            {
                "present_frame_ids": [],
                "present_frame_count": 0,
                "first_present_frame": None,
                "last_present_frame": None,
                "mean_area": 0.0,
                "max_area": 0,
            },
        )
        considered = _should_consider(row, args)
        drop = bool(considered and int(entry["present_frame_count"]) < int(args.min_present_frames))
        record = {
            "obj_id": obj_id,
            "frame_id": int(row.get("frame_id", -1)),
            "chunk_frame_index": int(row.get("chunk_frame_index", -1)),
            "phase5_role": str(row.get("phase5_role")),
            "frame0_child_split_role": row.get("frame0_child_split_role"),
            "considered": bool(considered),
            "dropped": bool(drop),
            "present_frame_count": int(entry["present_frame_count"]),
            "first_present_frame": entry["first_present_frame"],
            "last_present_frame": entry["last_present_frame"],
            "mean_area": float(entry["mean_area"]),
            "max_area": int(entry["max_area"]),
            "reason": "present_frame_count_lt_min" if drop else "kept",
        }
        filter_records.append(record)
        if not drop:
            kept_rows.append(row)

    payload = dict(birth)
    payload["schema_version"] = str(birth.get("schema_version", "unknown")) + "+presence_filter_v1"
    payload["source_birth_records"] = str(birth_path)
    payload["source_birth_records_sha256"] = _sha256_file(birth_path)
    payload["source_replay_summary"] = str(replay_path)
    payload["source_replay_summary_sha256"] = _sha256_file(replay_path)
    payload["presence_filter_policy"] = {
        "min_present_frames": int(args.min_present_frames),
        "anchor_chunk_index": args.anchor_chunk_index,
        "protect_frame0_parent_original": bool(args.protect_frame0_parent_original),
        "uses_reference_labels": False,
        "uses_predicted_replay_labels": True,
        "control_only_note": (
            "This filtered bank validates a tentative/defer direction. "
            "A promoted method must replace this full replay probe with a cheaper "
            "within-chunk persistence test."
        ),
    }
    payload["presence_filter_records"] = filter_records
    payload["presence_filter_dropped_obj_ids"] = [
        int(row["obj_id"]) for row in filter_records if row["dropped"]
    ]
    payload["presence_filter_dropped_count"] = int(len(payload["presence_filter_dropped_obj_ids"]))
    payload["filtered_from_row_count"] = int(len(birth.get("rows", [])))
    payload["row_count"] = int(len(kept_rows))
    payload["rows"] = kept_rows

    _write_json(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "row_count_before": int(len(birth.get("rows", []))),
                "row_count_after": int(len(kept_rows)),
                "dropped_obj_ids": payload["presence_filter_dropped_obj_ids"],
                "output_sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
