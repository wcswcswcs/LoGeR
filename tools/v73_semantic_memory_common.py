#!/usr/bin/env python3
"""Shared helpers for ACL2 v73 semantic-memory diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TARGET_CHUNKS = [6, 7, 8, 10, 12, 19, 20, 29, 30, 31, 32]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_chunks(text: str | Sequence[int] | None) -> list[int]:
    if text is None:
        return list(TARGET_CHUNKS)
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fields})


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def finite_values(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        val = safe_float(value)
        if val is not None:
            out.append(float(val))
    return out


def finite_mean(values: Iterable[Any]) -> float | None:
    vals = finite_values(values)
    return float(np.mean(vals)) if vals else None


def finite_median(values: Iterable[Any]) -> float | None:
    vals = finite_values(values)
    return float(np.median(vals)) if vals else None


def finite_quantile(values: Iterable[Any], q: float) -> float | None:
    vals = finite_values(values)
    return float(np.quantile(vals, q)) if vals else None


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(xs: Sequence[Any], ys: Sequence[Any]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        xf = safe_float(x)
        yf = safe_float(y)
        if xf is not None and yf is not None:
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None
    x_arr = rankdata([p[0] for p in pairs])
    y_arr = rankdata([p[1] for p in pairs])
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def auc_binary(scores: Sequence[Any], labels: Sequence[Any]) -> float | None:
    pairs: list[tuple[float, int]] = []
    for score, label in zip(scores, labels):
        sf = safe_float(score)
        if sf is None or label is None or label == "":
            continue
        try:
            lf = int(bool(int(label))) if str(label).strip() in {"0", "1"} else int(bool(label))
        except (TypeError, ValueError):
            lf = 1 if bool(label) else 0
        pairs.append((sf, lf))
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    ranks = rankdata([s for s, _ in pairs])
    pos_rank_sum = float(sum(rank for rank, (_, y) in zip(ranks, pairs) if y == 1))
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def best_auc(scores: Sequence[Any], labels: Sequence[Any]) -> dict[str, Any]:
    auc = auc_binary(scores, labels)
    if auc is None:
        return {"auc": None, "best_auc": None, "direction": None}
    if auc >= 0.5:
        return {"auc": float(auc), "best_auc": float(auc), "direction": "higher_is_positive"}
    return {"auc": float(auc), "best_auc": float(1.0 - auc), "direction": "lower_is_positive"}


def topk_precision(scores: Sequence[Any], labels: Sequence[Any], k: int = 5, higher_is_positive: bool = True) -> float | None:
    pairs: list[tuple[float, int]] = []
    for score, label in zip(scores, labels):
        sf = safe_float(score)
        if sf is None or label is None or label == "":
            continue
        pairs.append((sf, int(bool(label))))
    if not pairs:
        return None
    pairs.sort(key=lambda item: item[0], reverse=bool(higher_is_positive))
    top = pairs[: max(1, min(k, len(pairs)))]
    return float(sum(y for _, y in top) / len(top))


def zscore(values: Sequence[Any]) -> list[float | None]:
    vals = [safe_float(v) for v in values]
    finite = [v for v in vals if v is not None]
    if not finite:
        return [None for _ in vals]
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std <= 1e-12:
        std = 1.0
    return [None if v is None else float((v - mean) / std) for v in vals]


def find_chunk_dir(root: Path, chunk_id: int) -> Path | None:
    candidates = sorted(root.glob(f"chunk_{int(chunk_id):03d}_*"))
    return candidates[0] if candidates else None


def nested_get(obj: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def rotation_deg_from_trace(rot_trace: Any) -> float | None:
    val = safe_float(rot_trace)
    if val is None:
        return None
    x = max(-1.0, min(1.0, (val - 1.0) / 2.0))
    return float(math.degrees(math.acos(x)))
