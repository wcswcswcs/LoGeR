from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_MODULE_SECTIONS = (
    "Responsibilities",
    "Inputs",
    "Outputs",
    "State",
    "Forbidden Actions",
    "Artifact Schema",
    "Failure Mode",
    "Unit Tests",
    "Integration Tests",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(repo_root: Path, rel_path: str, required: bool = True) -> Dict[str, Any]:
    path = repo_root / rel_path
    record: Dict[str, Any] = {
        "path": rel_path,
        "required": required,
        "exists": path.exists(),
        "sha256": None,
        "byte_size": None,
        "line_count": None,
    }
    if path.exists() and path.is_file():
        record["sha256"] = sha256_file(path)
        record["byte_size"] = path.stat().st_size
        text_suffixes = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".csv", ".toml", ".ini"}
        if path.suffix.lower() in text_suffixes:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                record["line_count"] = sum(1 for _ in f)
    return record


def scan_markdown_sections(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    headings = set(re.findall(r"^## (.+)$", text, flags=re.MULTILINE))
    missing = [name for name in REQUIRED_MODULE_SECTIONS if name not in headings]
    return {
        "path": str(path),
        "exists": True,
        "required_sections": list(REQUIRED_MODULE_SECTIONS),
        "present_required_section_count": len(REQUIRED_MODULE_SECTIONS) - len(missing),
        "missing_required_sections": missing,
        "passes": not missing,
    }


def module_doc_status(repo_root: Path, module_doc_paths: Iterable[str]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for rel_path in module_doc_paths:
        path = repo_root / rel_path
        if not path.exists():
            records.append(
                {
                    "path": rel_path,
                    "exists": False,
                    "required_sections": list(REQUIRED_MODULE_SECTIONS),
                    "present_required_section_count": 0,
                    "missing_required_sections": list(REQUIRED_MODULE_SECTIONS),
                    "passes": False,
                }
            )
            continue
        rec = scan_markdown_sections(path)
        rec["path"] = rel_path
        records.append(rec)
    return {
        "module_doc_count": len(records),
        "required_section_count_per_doc": len(REQUIRED_MODULE_SECTIONS),
        "all_exist": all(r["exists"] for r in records),
        "all_pass": all(r["passes"] for r in records),
        "records": records,
    }


def source_import_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(stripped)
    return lines
