from __future__ import annotations

from typing import Any

from .v47_tracklet_builder import build_tracklets


def build_sparse_temporal_flow(**kwargs: Any) -> dict[str, Any]:
    payload = build_tracklets(edge_types={"adjacent", "skip", "short_reactivation"}, **kwargs)
    payload["summary"]["phase"] = "v47_sparse_temporal_flow_greedy_min_cost_proxy"
    payload["summary"]["solver_note"] = "Greedy one-predecessor/one-successor min-cost proxy; no GT used for prediction."
    return payload


__all__ = ["build_sparse_temporal_flow"]

