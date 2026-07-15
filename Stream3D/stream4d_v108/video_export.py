from __future__ import annotations

from dataclasses import dataclass

from .diagnostics import ReviewStatus


@dataclass(frozen=True)
class CasebookItem:
    case_id: str
    frame_id: int
    category: str
    artifact_paths: tuple[str, ...]
    visible_facts: tuple[str, ...]
    possible_error_classes: tuple[str, ...]
    review_status: ReviewStatus = ReviewStatus.USER_REVIEW_PENDING

    def assert_pending(self) -> None:
        if self.review_status is not ReviewStatus.USER_REVIEW_PENDING:
            raise ValueError("casebook items must stay pending until user review")
