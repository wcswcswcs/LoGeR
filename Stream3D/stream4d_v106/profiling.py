from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TimingRecord:
    name: str
    seconds: float


class StageTimer:
    def __init__(self) -> None:
        self.records: List[TimingRecord] = []

    def measure(self, name: str):
        timer = self

        class _Context:
            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                elapsed = time.perf_counter() - self.start
                timer.records.append(TimingRecord(name, elapsed))
                return False

        return _Context()

    def summary(self) -> Dict[str, float]:
        return {record.name: record.seconds for record in self.records}

