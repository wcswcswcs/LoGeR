from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ReviewStatus(str, Enum):
    USER_REVIEW_PENDING = "USER_REVIEW_PENDING"
    RECORDED_NOT_ACCEPTED = "RECORDED_NOT_ACCEPTED"
    REJECTED_BY_EVIDENCE = "REJECTED_BY_EVIDENCE"


@dataclass(frozen=True)
class DiagnosticMetricPolicy:
    """Policy object that prevents metrics from becoming acceptance gates."""

    metrics_are_diagnostic_only: bool = True
    user_review_status: ReviewStatus = ReviewStatus.USER_REVIEW_PENDING

    def label_metric(self, name: str, value: Any, source: str) -> dict[str, Any]:
        return {
            "metric_name": name,
            "metric_value": value,
            "source": source,
            "diagnostic_only": self.metrics_are_diagnostic_only,
            "review_status": self.user_review_status.value,
            "may_set_acceptance": False,
        }

    def assert_no_auto_acceptance(self, payload: dict[str, Any]) -> None:
        forbidden = {"VISUAL_PASS", "FINAL_ACCEPTED", "USER_VISUAL_ACCEPTED"}
        status = str(payload.get("status", ""))
        decision = str(payload.get("decision", ""))
        if status in forbidden or decision in forbidden:
            raise ValueError(f"automated v108 code cannot emit acceptance status {status or decision!r}")


REFERENCE_FIDELITY_METRICS = (
    "foreground_recall",
    "foreground_precision",
    "foreground_iou",
    "temporal_objectlet_recall",
    "reference_fragmentation_k80",
    "reference_merge_error",
)

GT_DIAGNOSTIC_METRICS = (
    "MV_AP_window",
    "MV_AP_scene",
    "MV_AP50",
    "MV_AP25",
    "Temporal_Track_Purity",
    "ASA",
    "GT_K80",
)
