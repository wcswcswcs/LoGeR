from __future__ import annotations

from typing import Any


def local_graph_refinement_noop(track_payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(track_payload)
    out["local_graph_refinement_status"] = "not_run_until_temporal_flow_signal"
    return out


__all__ = ["local_graph_refinement_noop"]

