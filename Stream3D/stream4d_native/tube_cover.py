from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .measurement_bank import MaskMeasurement


@dataclass
class TubeCoverResult:
    selected_measurements: list[MaskMeasurement]
    covered_tube_ids: set[int]
    diagnostics: dict[str, Any]


def select_tube_cover(
    measurements: list[MaskMeasurement],
    *,
    strategy: str = "all_measurements",
    top_k: int | None = None,
) -> TubeCoverResult:
    """Select image-space measurements for downstream tube graph construction."""

    if strategy not in {"all_measurements", "area_topk", "greedy_tube_cover"}:
        raise ValueError(f"unsupported tube cover strategy: {strategy}")
    if strategy == "area_topk" and top_k is not None:
        selected = sorted(measurements, key=lambda m: len(m.inside_tube_ids), reverse=True)[: int(top_k)]
    elif strategy == "greedy_tube_cover":
        selected = []
        covered: set[int] = set()
        for meas in sorted(measurements, key=lambda m: len(set(m.inside_tube_ids) - covered), reverse=True):
            gain = set(meas.inside_tube_ids) - covered
            if not gain:
                continue
            selected.append(meas)
            covered.update(gain)
            if top_k is not None and len(selected) >= int(top_k):
                break
    else:
        selected = list(measurements)
    covered_ids = {int(tube_id) for meas in selected for tube_id in meas.inside_tube_ids}
    return TubeCoverResult(
        selected_measurements=selected,
        covered_tube_ids=covered_ids,
        diagnostics={
            "strategy": strategy,
            "input_measurement_count": int(len(measurements)),
            "selected_measurement_count": int(len(selected)),
            "covered_tube_count": int(len(covered_ids)),
            "uses_metric_geometry": False,
        },
    )
