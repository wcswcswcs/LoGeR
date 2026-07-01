#!/usr/bin/env python3
"""Shared helpers for ACL2 v91 semantic topology regime audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import safe_float, safe_int, seq_norm, stable_hash_float


ROOT = Path("results/acl2_v91tf_semantic_topology_regime_adaptive_memory_control")
V90_ROOT = Path("results/acl2_v90tf_semantic_object_topology_scale_mode_memory_control")
V90_SOURCE = V90_ROOT / "phase1_semantic_topology_source"
V90_LEDGER = V90_ROOT / "phase2_semantic_topology_scale_mode_ledger"
V90_RELEVANCE = V90_ROOT / "phase3_semantic_topology_relevance"
V90_POLICY = V90_ROOT / "phase4_semantic_topology_observability_policy"
V90_FEATURE = V90_ROOT / "phase5_feature_match_topology_ruler"


def num(value: Any, default: float = 0.0) -> float:
    out = safe_float(value)
    return default if out is None else float(out)


def int0(value: Any) -> int:
    return int(safe_int(value) or 0)


def pair_id(seq: Any, prev: Any, curr: Any) -> str:
    return f"{seq_norm(seq)}_{int0(prev):03d}_{int0(curr):03d}"


def normalize_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["seq"] = out["seq"].astype(str).str.zfill(2)
    out["prev_chunk"] = pd.to_numeric(out["prev_chunk"], errors="coerce").fillna(0).astype(int)
    out["curr_chunk"] = pd.to_numeric(out["curr_chunk"], errors="coerce").fillna(0).astype(int)
    out["pair_id"] = [pair_id(s, p, c) for s, p, c in zip(out["seq"], out["prev_chunk"], out["curr_chunk"])]
    return out


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def nseries(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(default, index=df.index)), errors="coerce").fillna(default)


def stable_shuffle(values: pd.Series, salt: str) -> pd.Series:
    arr = values.to_numpy(copy=True)
    if len(arr) <= 1:
        return pd.Series(arr, index=values.index)
    order = sorted(range(len(arr)), key=lambda i: stable_hash_float(salt, i))
    shuffled = arr[order].copy()
    shuffled = np.roll(shuffled, 1)
    out = arr.copy()
    for dst, value in zip(order, shuffled):
        out[dst] = value
    return pd.Series(out, index=values.index)


def policy_metric(df: pd.DataFrame, score: pd.Series, signal: str, state: pd.Series | None = None) -> dict[str, Any]:
    from v86_soft_latent_utils import spearman_rho

    labelled = df[pd.to_numeric(df["abs_log_scale_jump_gt"], errors="coerce").notna()].copy()
    if len(labelled) == 0:
        return {"signal": signal, "available_rows": 0, "sequence_coverage": 0}
    v = pd.to_numeric(score.loc[labelled.index], errors="coerce").fillna(0.0)
    y = pd.to_numeric(labelled["abs_log_scale_jump_gt"], errors="coerce")
    high = y >= float(y.quantile(0.75))
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    if state is None:
        threshold = float(v.quantile(0.75))
        flags = v >= threshold
    else:
        s = state.loc[labelled.index].astype(str)
        flags = s.isin(["UPDATE", "REJECT", "DELAY", "RESET_RISK", "COMMIT_UPDATE", "COMMIT_RISK"])
    bad_or_high = bad | high
    rho = spearman_rho(v.tolist(), y.tolist())
    return {
        "signal": signal,
        "available_rows": int(len(labelled)),
        "sequence_coverage": int(labelled["seq"].astype(str).str.zfill(2).nunique()),
        "spearman_rho_abs_log_scale_jump": rho,
        "bad_recall": float((flags & bad_or_high).sum() / max(int(bad_or_high.sum()), 1)),
        "good_FPR": float((flags & good_low).sum() / max(int(good_low.sum()), 1)),
        "good_any_FPR": float((flags & labelled["base_case_type"].astype(str).eq("good")).sum() / max(int(labelled["base_case_type"].astype(str).eq("good").sum()), 1)),
    }


def positive_loso_folds(df: pd.DataFrame, score: pd.Series, baseline: pd.Series | None = None) -> int:
    from v86_soft_latent_utils import spearman_rho

    labelled = df[pd.to_numeric(df["abs_log_scale_jump_gt"], errors="coerce").notna()].copy()
    out = 0
    for seq, group in labelled.groupby(labelled["seq"].astype(str).str.zfill(2)):
        if len(group) < 3:
            continue
        y = pd.to_numeric(group["abs_log_scale_jump_gt"], errors="coerce")
        rho = spearman_rho(pd.to_numeric(score.loc[group.index], errors="coerce").fillna(0.0).tolist(), y.tolist())
        base_rho = None
        if baseline is not None:
            base_rho = spearman_rho(pd.to_numeric(baseline.loc[group.index], errors="coerce").fillna(0.0).tolist(), y.tolist())
        if rho is not None and (base_rho is None or rho >= base_rho + 0.01):
            out += 1
    return out
