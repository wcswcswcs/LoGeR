from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OwnershipClaim:
    frame_id: int
    global_object_id: int
    mask_id: str
    support_score: float
    source: str


@dataclass(frozen=True)
class OwnershipConflict:
    frame_id: int
    mask_id: str
    global_object_ids: tuple[int, ...]
    reason: str


def detect_support_conflicts(claims: list[OwnershipClaim], min_score: float = 0.0) -> list[OwnershipConflict]:
    by_mask: dict[str, list[OwnershipClaim]] = {}
    for claim in claims:
        if claim.support_score >= min_score:
            by_mask.setdefault(claim.mask_id, []).append(claim)
    conflicts: list[OwnershipConflict] = []
    for mask_id, mask_claims in by_mask.items():
        object_ids = sorted({claim.global_object_id for claim in mask_claims})
        if len(object_ids) > 1:
            conflicts.append(
                OwnershipConflict(
                    frame_id=mask_claims[0].frame_id,
                    mask_id=mask_id,
                    global_object_ids=tuple(object_ids),
                    reason="multiple objects claim the same mask support",
                )
            )
    return conflicts
