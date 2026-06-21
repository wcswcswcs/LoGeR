from __future__ import annotations

from typing import Any

from .v47_common import parse_bool, parse_float


def build_failure_summary(
    edge_summary: dict[str, Any] | None,
    tracklet_summary: dict[str, Any] | None,
    tracklet_control_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    edge_gate = (edge_summary or {}).get("gate", {})
    tracklet_gate = (tracklet_summary or {}).get("gate", {})
    control_gate = (tracklet_control_summary or {}).get("gate", {})
    if edge_summary and not edge_gate.get("pass", False):
        label = "NO_GO_ADJACENT_EDGE"
        blocker = "Adjacent temporal edge gate failed."
    elif tracklet_control_summary and not control_gate.get("pass", False):
        label = "NO_GO_TRACKLET_CONTROL"
        blocker = "Short tracklet real variant did not beat shuffled/no-temporal controls."
    elif tracklet_summary and not tracklet_gate.get("pass", False):
        label = "NO_GO_TRACKLET_PURIFY"
        blocker = "High-purity short tracklet gate failed."
    else:
        label = "PARTIAL_TRACKLET_SIGNAL"
        blocker = "Early gates did not provide enough evidence for full Stage-1 closure."
    return {
        "phase": "v47_failure_autopsy",
        "failure_label": label,
        "blocker": blocker,
        "edge_gate": edge_gate,
        "tracklet_gate": tracklet_gate,
        "tracklet_control_gate": control_gate,
        "tracklet_control_best_row": (tracklet_control_summary or {}).get("best_row"),
        "tracklet_control_result_rows": (tracklet_control_summary or {}).get("result_rows"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


__all__ = ["build_failure_summary"]
