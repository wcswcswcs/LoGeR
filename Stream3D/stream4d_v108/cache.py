from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheKey:
    namespace: str
    config_hash: str
    source_hash: str

    @property
    def digest(self) -> str:
        raw = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class CacheContract:
    def __init__(self, root: Path):
        self.root = Path(root)

    def metadata_path(self, key: CacheKey) -> Path:
        return self.root / key.namespace / f"{key.digest}.json"

    def write_metadata(self, key: CacheKey, payload: dict[str, Any]) -> Path:
        path = self.metadata_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"cache_key": key.__dict__, "payload": payload}
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
