from __future__ import annotations

from typing import Any

import numpy as np

from .semantic_occupancy import MaterialQuery, SemanticOccupancyState


def schedule_material_queries(
    masks: np.ndarray,
    *,
    variant: str,
    budget: int,
    overlap_frame_ranks: list[int] | None = None,
    disagreement: np.ndarray | None = None,
) -> tuple[list[MaterialQuery], dict[str, Any]]:
    state = SemanticOccupancyState(masks, overlap_frame_ranks=overlap_frame_ranks, disagreement=disagreement)
    queries = state.schedule(variant=variant, budget=budget)
    return queries, state.coverage_metrics(queries)

