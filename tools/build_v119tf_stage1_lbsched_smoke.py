#!/usr/bin/env python3
"""Build a lightweight ACL2 v119-TF LB-SCHED schedule smoke artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
OUT = RESULT_ROOT / "stage1_lbsched"

SEQ_LENGTHS = {"00": 4541, "02": 4661}
SCALE_FRAMES = 8
AUTO_KEYFRAME_THRESHOLD = 320
ANCHOR_SETS = {
    "default_first8": {
        "00": list(range(8)),
        "02": list(range(8)),
    },
    "delayed_b1_like": {
        "00": [0, 1, 2, 3, 668, 683, 3113, 3128],
        "02": [0, 1, 2, 3, 2813, 2843, 3818, 3833],
    },
    "spread_seed0": {
        "00": [0, 180, 540, 900, 1440, 2160, 3240, 4320],
        "02": [0, 240, 720, 1200, 1800, 2520, 3600, 4440],
    },
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def keyframe_interval(num_frames: int) -> int:
    return (num_frames + AUTO_KEYFRAME_THRESHOLD - 1) // AUTO_KEYFRAME_THRESHOLD


def sha_indices(indices: set[int]) -> str:
    payload = ",".join(str(idx) for idx in sorted(indices))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def legacy_observed(num_frames: int, anchors: list[int], interval: int) -> set[int]:
    selected = set(anchors)
    stream = [idx for idx in range(num_frames) if idx not in selected]
    return {idx for pos, idx in enumerate(stream) if interval <= 1 or pos % interval == 0}


def global_frozen_observed(num_frames: int, anchors: list[int], frozen: set[int]) -> set[int]:
    selected = set(anchors)
    stream = [idx for idx in range(num_frames) if idx not in selected]
    return {idx for idx in stream if idx in frozen}


def main() -> None:
    rows: list[dict[str, Any]] = []
    for seq, num_frames in SEQ_LENGTHS.items():
        interval = keyframe_interval(num_frames)
        default_anchors = list(range(SCALE_FRAMES))
        frozen = legacy_observed(num_frames, default_anchors, interval)
        frozen_hash = sha_indices(frozen)
        for anchor_name, by_seq in ANCHOR_SETS.items():
            anchors = sorted({int(idx) for idx in by_seq[seq]})
            legacy = legacy_observed(num_frames, anchors, interval)
            frozen_obs = global_frozen_observed(num_frames, anchors, frozen)
            expected_downstream = frozen - set(anchors)
            rows.append(
                {
                    "schema": "acl2_v119tf_stage1_lbsched_smoke_row_v1",
                    "seq": seq,
                    "num_frames": num_frames,
                    "keyframe_interval": interval,
                    "anchor_set": anchor_name,
                    "selected_anchor_frames": ";".join(str(idx) for idx in anchors),
                    "frozen_keyframe_set_hash": frozen_hash,
                    "frozen_keyframe_count": len(frozen),
                    "selected_anchor_overlap_frozen_count": len(set(anchors) & frozen),
                    "legacy_downstream_keyframe_count": len(legacy),
                    "global_frozen_downstream_keyframe_count": len(frozen_obs),
                    "expected_downstream_keyframe_count": len(expected_downstream),
                    "legacy_matches_default_downstream": legacy == expected_downstream,
                    "global_frozen_matches_expected_downstream": frozen_obs == expected_downstream,
                    "legacy_extra_count": len(legacy - expected_downstream),
                    "legacy_missing_count": len(expected_downstream - legacy),
                    "global_extra_count": len(frozen_obs - expected_downstream),
                    "global_missing_count": len(expected_downstream - frozen_obs),
                }
            )
    write_csv(OUT / "lbsched_schedule_smoke_rows.csv", rows)
    summary = {
        "schema": "acl2_v119tf_stage1_lbsched_smoke_summary_v1",
        "rows": rel(OUT / "lbsched_schedule_smoke_rows.csv"),
        "row_count": len(rows),
        "all_global_frozen_rows_match_expected_downstream": all(
            bool(row["global_frozen_matches_expected_downstream"]) for row in rows
        ),
        "legacy_mismatch_rows": sum(1 for row in rows if not bool(row["legacy_matches_default_downstream"])),
        "scope": "schedule resolver smoke only; no pose/depth parity or full runtime pass is claimed",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lbsched_schedule_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
