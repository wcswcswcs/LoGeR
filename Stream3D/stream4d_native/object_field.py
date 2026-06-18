from __future__ import annotations

from dataclasses import dataclass, field


class ObjectBirthConstraintError(ValueError):
    pass


@dataclass(frozen=True)
class ObjectFieldCandidate:
    candidate_id: int
    semantic_masklet_ids: tuple[int, ...]
    material_tube_ids: tuple[int, ...]
    score: float
    birth_source: str = "semantic_masklet"

    def validate_birth(self) -> None:
        if self.birth_source != "semantic_masklet":
            raise ObjectBirthConstraintError(
                f"object field candidate {self.candidate_id} has forbidden birth_source={self.birth_source}"
            )
        if not self.semantic_masklet_ids:
            raise ObjectBirthConstraintError(
                f"object field candidate {self.candidate_id} has no semantic masklet support"
            )


@dataclass
class ObjectField:
    object_id: int
    primary_field_id: int
    semantic_masklet_ids: list[int] = field(default_factory=list)
    attached_tube_ids: list[int] = field(default_factory=list)
    confidence: float = 0.0
    birth_state: str = "active"

