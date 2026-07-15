from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class LifecycleState(str, Enum):
    ACTIVE = "ACTIVE"
    ACTIVE_UNCERTAIN = "ACTIVE_UNCERTAIN"
    OUTPUT_PROBATION = "OUTPUT_PROBATION"
    MEMORY_PROBATION = "MEMORY_PROBATION"
    DORMANT = "DORMANT"
    REACTIVATING = "REACTIVATING"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class LifecycleEvent:
    frame_id: int
    global_object_id: int
    from_state: LifecycleState
    to_state: LifecycleState
    reason: str
    diagnostic_only: bool = True


@dataclass(frozen=True)
class AdmissionDiagnostic:
    frame_id: int
    global_object_id: int
    output_state: LifecycleState
    output_allowed: bool
    durable_memory_allowed: bool
    reasons: tuple[str, ...]
    visual_review_status: str = "USER_REVIEW_PENDING"
    user_attestation_verified: bool = False
    visual_review_required: bool = True
    diagnostic_only: bool = True


class LifecycleStateMachine:
    def transition(
        self,
        frame_id: int,
        global_object_id: int,
        from_state: LifecycleState,
        to_state: LifecycleState,
        reason: str,
    ) -> LifecycleEvent:
        if to_state is LifecycleState.ACTIVE and "user_review" in reason.lower():
            raise ValueError("user review cannot be inferred by automated lifecycle code")
        return LifecycleEvent(frame_id, global_object_id, from_state, to_state, reason)


class DelayedAdmissionPolicy:
    """Shadow policy for separating output-plane probation from durable memory."""

    def __init__(
        self,
        *,
        max_durable_edge_touch_count: int = 1,
        max_durable_bbox_area_frac: float = 0.35,
        max_durable_area_frac: float = 0.25,
        min_durable_extent: float = 0.25,
        min_watcher_visible_frames: int = 2,
        min_watcher_mean_iou: float = 0.40,
        min_visible_positive_anchors: int = 1,
        max_anchor_conflict_count: int = 0,
        max_positive_anchor_outlier_count: int = 0,
        min_anchor_depth_valid_fraction: float = 0.01,
    ) -> None:
        self.max_durable_edge_touch_count = int(max_durable_edge_touch_count)
        self.max_durable_bbox_area_frac = float(max_durable_bbox_area_frac)
        self.max_durable_area_frac = float(max_durable_area_frac)
        self.min_durable_extent = float(min_durable_extent)
        self.min_watcher_visible_frames = int(min_watcher_visible_frames)
        self.min_watcher_mean_iou = float(min_watcher_mean_iou)
        self.min_visible_positive_anchors = int(min_visible_positive_anchors)
        self.max_anchor_conflict_count = int(max_anchor_conflict_count)
        self.max_positive_anchor_outlier_count = int(max_positive_anchor_outlier_count)
        self.min_anchor_depth_valid_fraction = float(min_anchor_depth_valid_fraction)

    def evaluate(
        self,
        *,
        frame_id: int,
        global_object_id: int,
        component_stats: dict[str, Any],
        watcher_stats: dict[str, Any] | None = None,
        physical_support_stats: dict[str, Any] | None = None,
        visual_review_status: str = "USER_REVIEW_PENDING",
        user_attestation_verified: bool = False,
    ) -> AdmissionDiagnostic:
        watcher_stats = dict(watcher_stats or {})
        physical_support_stats = dict(physical_support_stats or {}) if physical_support_stats is not None else None
        reasons: list[str] = []
        area_px = int(component_stats.get("area_px", 0))
        if area_px <= 0:
            return AdmissionDiagnostic(
                frame_id=int(frame_id),
                global_object_id=int(global_object_id),
                output_state=LifecycleState.REJECTED,
                output_allowed=False,
                durable_memory_allowed=False,
                reasons=("empty_component",),
                visual_review_status=str(visual_review_status),
                user_attestation_verified=bool(user_attestation_verified),
            )

        if int(component_stats.get("edge_touch_count", 0)) > self.max_durable_edge_touch_count:
            reasons.append("boundary_cropped_or_edge_touching")
        if float(component_stats.get("bbox_area_frac", 0.0)) > self.max_durable_bbox_area_frac:
            reasons.append("bbox_too_large_for_durable_identity")
        if float(component_stats.get("area_frac", 0.0)) > self.max_durable_area_frac:
            reasons.append("area_too_large_for_durable_identity")
        if float(component_stats.get("bbox_extent", 1.0)) < self.min_durable_extent:
            reasons.append("component_extent_too_low")

        visible_frames = int(watcher_stats.get("visible_frame_count", 0))
        mean_iou = float(watcher_stats.get("mean_iou_to_previous_visible", -1.0))
        if visible_frames < self.min_watcher_visible_frames:
            reasons.append("insufficient_watcher_persistence")
        if mean_iou >= 0.0 and mean_iou < self.min_watcher_mean_iou:
            reasons.append("watcher_temporal_overlap_low")

        if physical_support_stats is not None:
            if physical_support_stats.get("physical_anchor_ready") is False:
                readiness_reasons = physical_support_stats.get("physical_anchor_readiness_reasons") or []
                if readiness_reasons:
                    reasons.extend(f"physical_anchor_not_ready:{reason}" for reason in readiness_reasons)
                else:
                    reasons.append("physical_anchor_not_ready")
            if physical_support_stats.get("geometry_available") is False:
                reasons.append("physical_geometry_unavailable")
            projected_positive_count = int(physical_support_stats.get("projected_positive_count", 0))
            if projected_positive_count < self.min_visible_positive_anchors:
                reasons.append("insufficient_visible_positive_anchors")
            conflict = dict(physical_support_stats.get("conflict_diagnostics") or {})
            if int(conflict.get("positive_negative_conflict_count", 0)) > self.max_anchor_conflict_count:
                reasons.append("anchor_conflict_with_negative_points")
            if int(conflict.get("positive_cluster_outlier_count", 0)) > self.max_positive_anchor_outlier_count:
                reasons.append("positive_anchor_outlier_points")
            target_support = dict(physical_support_stats.get("target_support") or {})
            depth_fraction = float(target_support.get("core_depth_valid_fraction", target_support.get("depth_valid_fraction", 1.0)))
            if depth_fraction < self.min_anchor_depth_valid_fraction:
                reasons.append("anchor_depth_support_too_low")

        if str(visual_review_status) != "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY":
            reasons.append("visual_review_not_accepted_for_durable_memory")
        elif not bool(user_attestation_verified):
            reasons.append("explicit_user_attestation_not_verified_for_durable_memory")

        durable_allowed = len(reasons) == 0
        return AdmissionDiagnostic(
            frame_id=int(frame_id),
            global_object_id=int(global_object_id),
            output_state=LifecycleState.ACTIVE if durable_allowed else LifecycleState.OUTPUT_PROBATION,
            output_allowed=True,
            durable_memory_allowed=bool(durable_allowed),
            reasons=tuple(reasons),
            visual_review_status=str(visual_review_status),
            user_attestation_verified=bool(user_attestation_verified),
            visual_review_required=True,
        )
