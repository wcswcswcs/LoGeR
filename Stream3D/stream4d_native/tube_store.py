from __future__ import annotations

from pathlib import Path

from .object_tube_io import TubeRecord, read_tube_records_jsonl, write_tube_records_jsonl


def save_tube_records(path: str | Path, tubes: list[TubeRecord]) -> None:
    write_tube_records_jsonl(path, tubes)


def load_tube_records(path: str | Path) -> list[TubeRecord]:
    return read_tube_records_jsonl(path)
