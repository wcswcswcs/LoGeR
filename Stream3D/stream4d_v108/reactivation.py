from __future__ import annotations

from dataclasses import dataclass

from .geometry_capsule import ProjectedAnchor


@dataclass(frozen=True)
class PromptPoint:
    uv: tuple[float, float]
    label: int
    source: str
    global_object_id: int | None


@dataclass(frozen=True)
class ReactivationPromptSet:
    frame_id: int
    target_global_object_id: int
    points: tuple[PromptPoint, ...]
    box_xyxy: tuple[float, float, float, float] | None = None
    source: str = "geometry"

    @property
    def positive_count(self) -> int:
        return sum(1 for point in self.points if point.label == 1)

    @property
    def negative_count(self) -> int:
        return sum(1 for point in self.points if point.label == 0)


def prompt_set_from_projected_anchors(
    frame_id: int,
    target_global_object_id: int,
    positive_anchors: list[ProjectedAnchor],
    negative_anchors: list[ProjectedAnchor],
) -> ReactivationPromptSet:
    points: list[PromptPoint] = []
    for anchor in positive_anchors:
        if anchor.usable_for_prompt:
            points.append(PromptPoint(anchor.target_uv, 1, "visible_target_anchor", target_global_object_id))
    for anchor in negative_anchors:
        if anchor.usable_for_prompt:
            points.append(PromptPoint(anchor.target_uv, 0, "visible_coview_negative_anchor", anchor.anchor.global_object_id))
    return ReactivationPromptSet(frame_id, target_global_object_id, tuple(points), source="geometry")


@dataclass(frozen=True)
class TwoDReactivationCandidate:
    frame_id: int
    target_global_object_id: int
    variant: str
    prompt_set: ReactivationPromptSet
    candidate_area_px: int
    diagnostic_only: bool = True
    output_plane_only: bool = True
    durable_memory_allowed: bool = False


def prompt_set_from_2d_capsule(
    *,
    frame_id: int,
    target_global_object_id: int,
    positive_uv: list[tuple[float, float]],
    box_xyxy: tuple[float, float, float, float] | None,
    source: str,
) -> ReactivationPromptSet:
    points = tuple(PromptPoint(tuple(point), 1, str(source), target_global_object_id) for point in positive_uv)
    return ReactivationPromptSet(
        frame_id=int(frame_id),
        target_global_object_id=int(target_global_object_id),
        points=points,
        box_xyxy=box_xyxy,
        source=str(source),
    )
