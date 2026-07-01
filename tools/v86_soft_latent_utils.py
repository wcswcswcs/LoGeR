#!/usr/bin/env python3
"""Shared helpers for ACL2 v86 soft latent gauge transport audits."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        out_fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in out_fields:
                    out_fields.append(key)
    else:
        out_fields = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in out_fields})


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if not math.isfinite(v):
            return ""
        return f"{v:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def clamp01(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def seq_norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return f"{int(float(text)):02d}"
    except ValueError:
        return text.zfill(2)


def pair_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    seq = seq_norm(row.get("seq"))
    prev_chunk = safe_int(row.get("prev_chunk"))
    curr_chunk = safe_int(row.get("curr_chunk"))
    return seq, int(prev_chunk or 0), int(curr_chunk or 0)


def stable_hash_float(*parts: Any) -> float:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float((w.sum() ** 2) / max(float((w * w).sum()), 1e-12))


def weighted_rank(features: np.ndarray, weights: np.ndarray, rel_tol: float = 1e-5) -> int:
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(w) & (w > 0)
    if x.ndim != 2 or not mask.any():
        return 0
    x = x[mask]
    w = np.sqrt(w[mask])[:, None]
    xw = (x - np.average(x, axis=0, weights=(w[:, 0] ** 2))) * w
    if xw.shape[0] < 2:
        return 0
    try:
        s = np.linalg.svd(xw, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0
    if s.size == 0:
        return 0
    threshold = max(float(s[0]) * rel_tol, 1e-10)
    return int(np.sum(s > threshold))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.sum(v[mask] * w[mask]) / max(float(np.sum(w[mask])), 1e-12))


def weighted_residual(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    dist2 = np.sum(diff * diff, axis=1)
    return weighted_mean(dist2, weights)


def spearman_rho(xs: Sequence[Any], ys: Sequence[Any]) -> float | None:
    x = np.asarray([safe_float(v) for v in xs], dtype=object)
    y = np.asarray([safe_float(v) for v in ys], dtype=object)
    mask = np.array([(a is not None) and (b is not None) for a, b in zip(x, y)], dtype=bool)
    if int(mask.sum()) < 3:
        return None
    xv = np.asarray([float(v) for v in x[mask]], dtype=np.float64)
    yv = np.asarray([float(v) for v in y[mask]], dtype=np.float64)
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return None
    rx = _rankdata(xv)
    ry = _rankdata(yv)
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr) if math.isfinite(float(corr)) else None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def finite_median(values: Iterable[Any]) -> float | None:
    vals = [safe_float(v) for v in values]
    arr = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def finite_quantile(values: Iterable[Any], q: float) -> float | None:
    vals = [safe_float(v) for v in values]
    arr = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))
