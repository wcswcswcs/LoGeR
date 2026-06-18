from __future__ import annotations

from typing import Any

from .object_field import ObjectFieldCandidate


def _jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    sa, sb = set(int(v) for v in a), set(int(v) for v in b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(len(sa | sb), 1))


def select_compact_object_fields(
    candidates: list[ObjectFieldCandidate],
    *,
    max_fields: int = 300,
    duplicate_support_jaccard: float = 0.75,
    duplicate_material_jaccard: float = 1.01,
) -> tuple[list[ObjectFieldCandidate], dict[str, Any]]:
    valid: list[ObjectFieldCandidate] = []
    forbidden_birth_count = 0
    for candidate in candidates:
        try:
            candidate.validate_birth()
        except Exception:
            forbidden_birth_count += 1
            continue
        valid.append(candidate)
    ordered = sorted(valid, key=lambda c: (-float(c.score), int(c.candidate_id)))
    selected: list[ObjectFieldCandidate] = []
    duplicate_drop_count = 0
    material_duplicate_drop_count = 0
    for candidate in ordered:
        if len(selected) >= int(max_fields):
            break
        if any(_jaccard(candidate.semantic_masklet_ids, prev.semantic_masklet_ids) >= float(duplicate_support_jaccard) for prev in selected):
            duplicate_drop_count += 1
            continue
        if float(duplicate_material_jaccard) <= 1.0 and any(
            _jaccard(candidate.material_tube_ids, prev.material_tube_ids) >= float(duplicate_material_jaccard)
            for prev in selected
        ):
            material_duplicate_drop_count += 1
            continue
        selected.append(candidate)
    diagnostics = {
        "input_candidate_count": int(len(candidates)),
        "valid_semantic_birth_candidate_count": int(len(valid)),
        "forbidden_birth_count": int(forbidden_birth_count),
        "selected_object_count": int(len(selected)),
        "duplicate_drop_count": int(duplicate_drop_count),
        "material_duplicate_drop_count": int(material_duplicate_drop_count),
        "duplicate_rate": float((duplicate_drop_count + material_duplicate_drop_count) / max(len(valid), 1)),
    }
    return selected, diagnostics
