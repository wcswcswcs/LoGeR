from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GapDecision:
    residual_area: int
    action: str
    reason: str


def classify_gap(residual_area: int, inherited_overlap_area: int, min_birth_area: int = 1) -> GapDecision:
    if residual_area < min_birth_area:
        return GapDecision(residual_area, "defer", "residual area below birth threshold")
    if inherited_overlap_area > 0:
        return GapDecision(residual_area, "repair_existing", "residual touches inherited objectlet")
    return GapDecision(residual_area, "exact_birth", "unexplained foreground residual")

