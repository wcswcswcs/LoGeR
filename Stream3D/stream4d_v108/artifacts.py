from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    schema_version: str
    sha256: str | None
    byte_size: int | None
    created_at_utc: str


class ArtifactWriter:
    """Small writer that records hashes for v108 audit artifacts."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.records: list[ArtifactRecord] = []

    def _record(self, path: Path, schema_version: str) -> ArtifactRecord:
        record = ArtifactRecord(
            path=path.as_posix(),
            schema_version=schema_version,
            sha256=sha256_file(path) if path.exists() else None,
            byte_size=path.stat().st_size if path.exists() else None,
            created_at_utc=now_utc(),
        )
        self.records.append(record)
        return record

    def write_json(self, rel_path: str, payload: Any, schema_version: str) -> ArtifactRecord:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self._record(path, schema_version)

    def write_jsonl(self, rel_path: str, rows: Iterable[Any], schema_version: str) -> ArtifactRecord:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")
        return self._record(path, schema_version)

    def write_csv(self, rel_path: str, rows: Iterable[Any], schema_version: str) -> ArtifactRecord:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        rows_list = []
        for row in rows:
            ready = json_ready(row)
            if not isinstance(ready, Mapping):
                raise TypeError(f"CSV rows must become mappings after json_ready, got {type(ready).__name__}")
            rows_list.append(dict(ready))
        fieldnames = sorted({key for row in rows_list for key in row})
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_list:
                writer.writerow({key: json_ready(row.get(key)) for key in fieldnames})
        return self._record(path, schema_version)

    def record_existing(self, rel_path: str, schema_version: str) -> ArtifactRecord:
        path = self.root / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        return self._record(path, schema_version)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "stream4d_v108_artifact_manifest_v1",
            "created_at_utc": now_utc(),
            "artifact_count": len(self.records),
            "artifacts": [asdict(record) for record in self.records],
        }
