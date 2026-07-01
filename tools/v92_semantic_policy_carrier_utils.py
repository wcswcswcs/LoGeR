#!/usr/bin/env python3
"""Shared paths and helpers for ACL2 v92 semantic policy carrier audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from v86_soft_latent_utils import safe_float, safe_int


ROOT = Path("results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery")
V91_ROOT = Path("results/acl2_v91tf_semantic_topology_regime_adaptive_memory_control")
V91_PHASE5 = V91_ROOT / "phase5_memory_update_policy"
V91_PHASE7 = V91_ROOT / "phase7_carrier_attribution_or_blocked"
RISK_STATES = {"RESET_RISK", "DELAY", "REJECT"}
HOLD_STATES = {"HOLD", "ABSTAIN"}


def bool_text(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def seq_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return f"{int(float(text)):02d}"
    except ValueError:
        return text.zfill(2)


def int0(value: Any) -> int:
    return int(safe_int(value) or 0)


def pair_id(seq: Any, prev: Any, curr: Any) -> str:
    return f"{seq_text(seq)}_{int0(prev):03d}_{int0(curr):03d}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def nseries(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(default, index=df.index)), errors="coerce").fillna(default)


def state_counts(rows: Iterable[dict[str, Any]], col: str = "policy_state") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(col, "") or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def mean_or_none(values: Iterable[Any]) -> float | None:
    vals = [safe_float(v) for v in values]
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))
