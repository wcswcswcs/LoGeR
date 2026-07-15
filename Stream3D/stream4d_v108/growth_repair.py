from __future__ import annotations

from dataclasses import dataclass

from .masklet_watcher import GrowthAlert


@dataclass(frozen=True)
class GrowthRepairSuggestion:
    frame_id: int
    global_object_id: int
    action: str
    reason: str
    output_plane_only: bool = True
    durable_memory_allowed: bool = False
    user_attestation_verified: bool = False


class GrowthRepairPlanner:
    def suggest(self, alert: GrowthAlert, has_anchor_conflict: bool) -> GrowthRepairSuggestion:
        if has_anchor_conflict:
            return GrowthRepairSuggestion(
                frame_id=alert.frame_id,
                global_object_id=alert.global_object_id,
                action="clip_or_shadow_conflicted_growth",
                reason="growth overlaps existing physical anchor support",
            )
        return GrowthRepairSuggestion(
            frame_id=alert.frame_id,
            global_object_id=alert.global_object_id,
            action="keep_output_probation_and_collect_more_evidence",
            reason="sudden growth is diagnostic until geometry/appearance support agrees",
        )

    def suggest_from_shadow_stats(
        self,
        *,
        frame_id: int,
        global_object_id: int,
        visible: bool,
        edge_touch_count: int,
        area_ratio_to_history: float,
        bbox_area_fraction: float,
        visual_review_status: str = "USER_REVIEW_PENDING",
        user_attestation_verified: bool = False,
    ) -> GrowthRepairSuggestion:
        if not visible:
            return GrowthRepairSuggestion(
                frame_id=int(frame_id),
                global_object_id=int(global_object_id),
                action="watch_or_dormant_capsule_no_mask",
                reason="object is not visible in the current diagnostic frame",
            )
        if int(edge_touch_count) > 0 and float(bbox_area_fraction) > 0.20:
            return GrowthRepairSuggestion(
                frame_id=int(frame_id),
                global_object_id=int(global_object_id),
                action="mark_active_uncertain_and_demote_from_durable_memory",
                reason="boundary-touching large mask is unsafe for durable memory",
            )
        if float(area_ratio_to_history) >= 2.0:
            return GrowthRepairSuggestion(
                frame_id=int(frame_id),
                global_object_id=int(global_object_id),
                action="mark_active_uncertain_and_request_repair_candidate",
                reason="relative growth against object history is high",
            )
        if str(visual_review_status) != "VISUALLY_ACCEPTED_FOR_DURABLE_MEMORY":
            return GrowthRepairSuggestion(
                frame_id=int(frame_id),
                global_object_id=int(global_object_id),
                action="keep_output_probation_until_visual_review",
                reason="durable memory is blocked without explicit visual acceptance",
            )
        if not bool(user_attestation_verified):
            return GrowthRepairSuggestion(
                frame_id=int(frame_id),
                global_object_id=int(global_object_id),
                action="keep_output_probation_until_user_attestation",
                reason="durable memory is blocked without verified explicit user attestation",
                user_attestation_verified=False,
            )
        return GrowthRepairSuggestion(
            frame_id=int(frame_id),
            global_object_id=int(global_object_id),
            action="no_growth_repair_needed",
            reason="shadow diagnostics did not find unsafe growth",
            user_attestation_verified=True,
        )
