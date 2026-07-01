#!/usr/bin/env python3
"""Shared paths and helpers for ACL2 v93 semantic object identity audits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier")
V92_ROOT = Path("results/acl2_v92tf_semantic_policy_carrier_merge_gauge_boundary_discovery")
V91_ROOT = Path("results/acl2_v91tf_semantic_topology_regime_adaptive_memory_control")


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
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def pair_id(seq: Any, prev: Any, curr: Any) -> str:
    return f"{seq_text(seq)}_{int0(prev):03d}_{int0(curr):03d}"


def bool_text(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
