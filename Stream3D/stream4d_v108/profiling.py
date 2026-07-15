from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuntimeSpan:
    name: str
    start_sec: float
    end_sec: float

    @property
    def elapsed_sec(self) -> float:
        return self.end_sec - self.start_sec


class StageTimer:
    def __init__(self) -> None:
        self._active: dict[str, float] = {}
        self.spans: list[RuntimeSpan] = []

    def start(self, name: str) -> None:
        self._active[name] = time.perf_counter()

    def stop(self, name: str) -> RuntimeSpan:
        start = self._active.pop(name)
        span = RuntimeSpan(name=name, start_sec=start, end_sec=time.perf_counter())
        self.spans.append(span)
        return span

    def total(self, name: Optional[str] = None) -> float:
        return sum(span.elapsed_sec for span in self.spans if name is None or span.name == name)
