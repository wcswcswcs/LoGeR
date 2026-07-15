from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectletCoverage:
    foreground_pixels_previous: int
    foreground_pixels_current: int
    retained_pixels: int

    @property
    def coverage_ratio(self) -> float:
        if self.foreground_pixels_previous == 0:
            return 1.0
        return self.foreground_pixels_current / self.foreground_pixels_previous

    @property
    def history_coverage_retention(self) -> float:
        if self.foreground_pixels_previous == 0:
            return 1.0
        return self.retained_pixels / self.foreground_pixels_previous

