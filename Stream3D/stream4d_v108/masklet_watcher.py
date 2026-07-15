from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskletObservation:
    frame_id: int
    global_object_id: int
    area_px: int
    bbox_area_fraction: float
    visible: bool
    touches_image_edge: bool


@dataclass(frozen=True)
class GrowthAlert:
    frame_id: int
    global_object_id: int
    previous_area_px: int
    current_area_px: int
    ratio: float
    diagnostic_only: bool = True


class MaskletWatcher:
    def __init__(self, growth_ratio_alert: float = 2.0) -> None:
        self.growth_ratio_alert = growth_ratio_alert
        self._last_by_object: dict[int, MaskletObservation] = {}

    def observe(self, obs: MaskletObservation) -> GrowthAlert | None:
        last = self._last_by_object.get(obs.global_object_id)
        self._last_by_object[obs.global_object_id] = obs
        if last is None or last.area_px <= 0:
            return None
        ratio = obs.area_px / last.area_px
        if ratio >= self.growth_ratio_alert:
            return GrowthAlert(obs.frame_id, obs.global_object_id, last.area_px, obs.area_px, ratio)
        return None
