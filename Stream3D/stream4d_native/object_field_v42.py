from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class V42ObjectFieldConstraintError(ValueError):
    pass


@dataclass(frozen=True)
class V42ObjectField:
    object_id: int
    primary_field_id: int
    semantic_masklet_ids: tuple[int, ...]
    attached_tube_ids: tuple[int, ...]
    confidence: float
    birth_source: str = "semantic_part_graph"
    static_scene_update_weight: float = 0.0

    def validate(self) -> None:
        if self.birth_source != "semantic_part_graph":
            raise V42ObjectFieldConstraintError(
                f"object {self.object_id} has forbidden birth_source={self.birth_source}"
            )
        if not self.semantic_masklet_ids:
            raise V42ObjectFieldConstraintError(f"object {self.object_id} has no semantic masklet support")


def _parse_json_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    import json

    payload = json.loads(text)
    return [int(v) for v in payload]


def _selected(row: dict[str, Any], selected_column: str) -> bool:
    value = row.get(selected_column, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(int(value), int(value))
        if self.parent[int(value)] != int(value):
            self.parent[int(value)] = self.find(self.parent[int(value)])
        return self.parent[int(value)]

    def union(self, left: int, right: int) -> None:
        root_l = self.find(left)
        root_r = self.find(right)
        if root_l != root_r:
            self.parent[root_r] = root_l


def build_v42_object_fields_from_alignment_rows(
    rows: list[dict[str, Any]],
    *,
    selected_column: str = "selected_O4_semantic_part_gated_robust_trim",
    safe_merge_semantic_affinity: float | None = None,
    safe_merge_object_affinity: float | None = None,
    safe_merge_max_visible_outside_conflict: float = 0.35,
    max_material_union_count: int | None = None,
    max_fields: int = 300,
) -> list[V42ObjectField]:
    uf = _UnionFind()
    selected_rows: list[dict[str, Any]] = []
    for row in rows:
        selected = _selected(row, selected_column)
        if (
            not selected
            and safe_merge_semantic_affinity is not None
            and safe_merge_object_affinity is not None
        ):
            selected = (
                str(row.get("same_frame_cannot_link", "")).strip().lower() not in {"true", "1"}
                and str(row.get("role_conflict", "")).strip().lower() not in {"true", "1"}
                and float(row.get("semantic_affinity", 0.0) or 0.0) >= float(safe_merge_semantic_affinity)
                and float(row.get("object_affinity", 0.0) or 0.0) >= float(safe_merge_object_affinity)
                and float(row.get("visible_outside_conflict_ratio", 0.0) or 0.0)
                <= float(safe_merge_max_visible_outside_conflict)
            )
        if not selected:
            continue
        material_union_count = int(float(row.get("material_union_count", 0.0) or 0.0))
        if max_material_union_count is not None and material_union_count > int(max_material_union_count):
            continue
        token_i = int(row["token_i"])
        token_j = int(row["token_j"])
        uf.union(token_i, token_j)
        selected_rows.append(row)
    by_root: dict[int, list[dict[str, Any]]] = {}
    for row in selected_rows:
        root = uf.find(int(row["token_i"]))
        by_root.setdefault(root, []).append(row)
    fields: list[V42ObjectField] = []
    for primary_id, (_root, group) in enumerate(
        sorted(by_root.items(), key=lambda item: (-len(item[1]), int(item[0])))[: int(max_fields)]
    ):
        tokens: set[int] = set()
        tubes: set[int] = set()
        scores: list[float] = []
        for row in group:
            tokens.add(int(row["token_i"]))
            tokens.add(int(row["token_j"]))
            tubes.update(_parse_json_list(row.get("shared_tube_ids", "")))
            residual = float(row.get("residual_proxy", 0.0) or 0.0)
            scores.append(max(0.0, 1.0 - residual))
        field = V42ObjectField(
            object_id=int(primary_id),
            primary_field_id=int(primary_id),
            semantic_masklet_ids=tuple(sorted(tokens)),
            attached_tube_ids=tuple(sorted(tubes)),
            confidence=float(np.mean(np.asarray(scores, dtype=np.float64))) if scores else 0.0,
            birth_source="semantic_part_graph",
            static_scene_update_weight=0.0,
        )
        field.validate()
        fields.append(field)
    return fields


def object_field_rows(fields: list[V42ObjectField], *, scene: str, variant: str, source: str) -> list[dict[str, Any]]:
    return [
        {
            "scene": scene,
            "variant": variant,
            "source": source,
            "object_id": int(field.object_id),
            "primary_field_id": int(field.primary_field_id),
            "semantic_masklet_ids": list(field.semantic_masklet_ids),
            "attached_tube_ids": list(field.attached_tube_ids),
            "semantic_masklet_count": int(len(field.semantic_masklet_ids)),
            "attached_tube_count": int(len(field.attached_tube_ids)),
            "confidence": float(field.confidence),
            "birth_source": field.birth_source,
            "static_scene_update_weight": float(field.static_scene_update_weight),
        }
        for field in fields
    ]
