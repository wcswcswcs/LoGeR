from __future__ import annotations

from .semantic_masklet_inference import MaskletMeasurement, infer_semantic_masklets


def split_mixed_same_frame_masks(measurements: list[MaskletMeasurement]) -> dict[tuple[int, int], list[list[int]]]:
    groups: dict[tuple[int, int], dict[str, list[int]]] = {}
    for measurement in measurements:
        key = (int(measurement.frame_rank), int(measurement.mask_id))
        groups.setdefault(key, {}).setdefault(measurement.d4rt_support_key, []).append(int(measurement.measurement_id))
    return {key: list(parts.values()) for key, parts in groups.items() if len(parts) > 1}


def safe_merge_fragments(measurements: list[MaskletMeasurement], *, visual_threshold: float = 0.90) -> dict[int, int]:
    return infer_semantic_masklets(
        measurements,
        use_visual=True,
        use_d4rt=True,
        max_rank_delta=1,
        visual_threshold=float(visual_threshold),
    )

