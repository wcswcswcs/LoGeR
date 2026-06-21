#!/usr/bin/env python3
"""Shared helpers for ACL2 v76-TF artifact audits."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results/kitti01_hmc_v2"
PREPROCESS_ROOT = REPO_ROOT / "results/kitti_preprocess"
V76_ROOT = RESULT_ROOT / "acl2_v76tf_c9_informed_semantic_tri_replay_memory_control/report_final"
V45_ROOT = RESULT_ROOT / "acl2_v45_codeaudit_c9clean_attribution_c23_adaptive_trireplay"
V46B_ROOT = RESULT_ROOT / "acl2_v46b_component_attribution_frame_ttt_swa/phase2_factorial/report_R1"
V52_ROOT = RESULT_ROOT / "acl2_v52_c9clean_adaptive_ttt_semantic_geometry/phase1_c9_attribution"
V64_ROOT = RESULT_ROOT / "acl2_v64_ttt_scale_mechanism_attribution/report_final"
V65_ROOT = RESULT_ROOT / "acl2_v65_c9_h35_transition_swap_merge_gauge_attribution/report_final"
V74_ROOT = RESULT_ROOT / "acl2_v74tf_training_free_semantic_memory_control"

C9_REPEAT_DIR = V45_ROOT / "phase0_hard_gate/rollouts/V45_P0_C9_REPEAT"
V46B_REGISTRY = V46B_ROOT / "phase2_factorial_registry.csv"
V45_CLEAN_REGISTRY = V45_ROOT / "phase1_c9_clean/report_R1/full_metrics/full_online_registry.csv"
V45_LEDGER = V45_ROOT / "final_reports/v45_component_contribution_ledger.csv"
V45_INTERACTION = V45_ROOT / "final_reports/v45_component_interaction_matrix.csv"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    out_rows = [dict(row) for row in rows]
    if fields is None:
        ordered: List[str] = []
        for row in out_rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)


def clean_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> Optional[int]:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    return out


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "done", "pass"}


def mean(values: Iterable[Any]) -> Optional[float]:
    vals = [num for num in (safe_float(value) for value in values) if num is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def min_max(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float]]:
    vals = [num for num in (safe_float(value) for value in values) if num is not None]
    if not vals:
        return None, None
    return min(vals), max(vals)


def first_row(rows: Iterable[Mapping[str, Any]], key: str, value: str) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row.get(key)) == value:
            return row
    return None


def walk_json(value: Any, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            yield child_path, child
            yield from walk_json(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_path = path + (str(idx),)
            yield child_path, child
            yield from walk_json(child, child_path)


def collect_numeric_by_key(rows: Iterable[Mapping[str, Any]], suffixes: Sequence[str]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for row in rows:
        for path, value in walk_json(row):
            key = path[-1] if path else ""
            if not any(key.endswith(suffix) for suffix in suffixes):
                continue
            num = safe_float(value)
            if num is not None:
                out.setdefault(key, []).append(num)
    return out


def count_nonzero_csv(path: Path) -> Dict[str, Any]:
    rows = read_csv(path)
    numeric_values: List[float] = []
    for row in rows:
        for value in row.values():
            num = safe_float(value)
            if num is not None:
                numeric_values.append(abs(num))
    nonzero = sum(1 for value in numeric_values if value > 0.0)
    return {
        "path": rel(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "numeric_value_count": len(numeric_values),
        "nonzero_numeric_value_count": nonzero,
        "numeric_mean_abs": mean(numeric_values),
    }


def artifact_status(path: Path, kind: str = "file") -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": rel(path),
        "kind": kind,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
    }
