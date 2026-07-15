from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureBankContract:
    shared_frame_feature_bank: bool
    storage: str
    hot_window: int

    def validate(self) -> None:
        if self.storage not in {"cuda", "cpu"}:
            raise ValueError("feature bank storage must be cuda or cpu")
        if self.hot_window <= 0:
            raise ValueError("feature bank hot window must be positive")

