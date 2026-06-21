from __future__ import annotations

from collections import defaultdict
from typing import Any


def export_object_fields_from_tracklets(tracklet_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_tracklet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tracklet_rows:
        by_tracklet[str(row["tracklet_id"])].append(row)
    fields: list[dict[str, Any]] = []
    for idx, (tracklet_id, rows) in enumerate(sorted(by_tracklet.items())):
        fields.append(
            {
                "object_id": f"v47_obj_{idx:05d}",
                "tracklet_id": tracklet_id,
                "mask_observation_ids": [f"{row.get('scene')}:{row.get('frame_id')}:{row.get('mask_id')}" for row in rows],
                "mask_atom_ids": [],
                "carrier_ids": [],
                "tracklet_ids": [tracklet_id],
                "temporal_span": len({row.get("frame_id") for row in rows}),
                "confidence": 1.0,
                "unknown_evidence": [],
                "has_mask_or_mask_atom_evidence": True,
                "birth_from_d4rt_tube": False,
            }
        )
    return fields


__all__ = ["export_object_fields_from_tracklets"]

