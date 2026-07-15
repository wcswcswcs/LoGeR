from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ReconciliationResult:
    accepted_global_id: int
    rejected_duplicate_ids: tuple[int, ...]
    coverage_before: float
    coverage_after: float


def choose_coverage_preserving_identity(
    candidate_coverages: Dict[int, float], min_coverage_drop: float = -0.01
) -> ReconciliationResult:
    if not candidate_coverages:
        raise ValueError("candidate_coverages must not be empty")
    accepted = max(candidate_coverages.items(), key=lambda item: (item[1], -item[0]))[0]
    before = max(candidate_coverages.values())
    after = candidate_coverages[accepted]
    if after + 1e-9 < before + min_coverage_drop:
        raise AssertionError("reconciliation reduced coverage beyond allowed tolerance")
    rejected = tuple(sorted(gid for gid in candidate_coverages if gid != accepted))
    return ReconciliationResult(accepted, rejected, before, after)

