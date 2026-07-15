from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheKey:
    version: str
    scene_id: str
    chunk_index: int
    artifact_kind: str

    def path(self, root: Path) -> Path:
        return root / self.version / self.scene_id / f"chunk_{self.chunk_index:04d}" / self.artifact_kind

