from __future__ import annotations

from collections import defaultdict
from typing import Any

from .v47_common import parse_int


def shared_observation_rows(carrier_rows: list[dict[str, Any]], track_by_carrier: dict[tuple[str, int], str]) -> list[dict[str, Any]]:
    tracks_by_mask: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in carrier_rows:
        mask_id = parse_int(row.get("observed_mask_id"), 0)
        if mask_id <= 0:
            continue
        key = (str(row["scene"]), parse_int(row["frame_id"]), mask_id)
        track_id = track_by_carrier.get((str(row["scene"]), parse_int(row["carrier_id"])))
        if track_id:
            tracks_by_mask[key].add(track_id)
    return [
        {
            "scene": scene,
            "frame_id": frame_id,
            "mask_id": mask_id,
            "shared_observation_track_count": len(track_ids),
            "shared_observation": len(track_ids) > 1,
            "can_create_identity_merge_edge": False,
        }
        for (scene, frame_id, mask_id), track_ids in sorted(tracks_by_mask.items())
        if len(track_ids) > 1
    ]


__all__ = ["shared_observation_rows"]

