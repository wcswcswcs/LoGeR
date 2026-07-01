#!/usr/bin/env python3
"""Build ACL2 v86 Phase1 soft pair universe from v85 anchor rows and Q/K features."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from v86_soft_latent_utils import (
    clamp01,
    effective_sample_size,
    finite_quantile,
    pair_key,
    parse_bool,
    safe_float,
    safe_int,
    seq_norm,
    weighted_rank,
    write_csv,
    write_json,
)


DEFAULT_ANCHOR_ROWS = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_rows.csv"
)
DEFAULT_FEATURE_PT = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank/qk_anchor_features.pt"
)
DEFAULT_OUT = Path("results/acl2_v86tf_robust_soft_latent_gauge_transport/phase1_soft_pair_universe")

ROW_FIELDS = [
    "seq",
    "prev_chunk",
    "curr_chunk",
    "case_label",
    "quality_label",
    "prev_patch_id",
    "curr_patch_id",
    "prev_frame_id",
    "curr_frame_id",
    "q_feature_available",
    "k_feature_available",
    "q_feature_source",
    "k_feature_source",
    "semantic_compatibility",
    "same_label",
    "same_role",
    "cross_boundary_flag",
    "dynamic_flag",
    "zero_conf_flag",
    "raw_overlap_residual",
    "confidence_weighted_residual",
    "local_shape_residual",
    "pairwise_distance_ratio_residual",
    "parallax_score",
    "local_3d_spread_prev",
    "local_3d_spread_curr",
    "read_usage_current",
    "swa_qk_proxy",
    "true_route_available",
    "true_route_mass_if_available",
    "w_conf",
    "w_sem",
    "w_shape",
    "w_overlap",
    "w_parallax",
    "w_memory",
    "w_risk",
    "w_fit",
    "support_class",
    "support_state_candidate",
    "risk_reason",
    "source_anchor_support_class",
    "raw_coord_available",
    "raw_lookup_status",
    "feature_source_path",
    "anchor_row_index",
    "pair_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-rows", type=Path, default=DEFAULT_ANCHOR_ROWS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-dim", type=int, default=8)
    return parser.parse_args()


def _bool_from_row(row: Mapping[str, Any], key: str) -> bool:
    return parse_bool(row.get(key))


def _same_label(row: Mapping[str, Any]) -> bool:
    text = str(row.get("same_label") or "").strip()
    if text:
        return parse_bool(text)
    return str(row.get("prev_sem_label")) == str(row.get("curr_sem_label"))


def _semantic_compatibility(row: Mapping[str, Any], same_label: bool) -> float:
    if same_label:
        return 1.0
    prev_label = str(row.get("prev_sem_label") or "").strip()
    curr_label = str(row.get("curr_sem_label") or "").strip()
    if not prev_label or not curr_label:
        return 0.5
    return 0.2


def _exp_score(value: float | None, scale: float | None, missing_value: float) -> float:
    if value is None or scale is None or scale <= 0:
        return missing_value
    return clamp01(math.exp(-max(0.0, value) / scale))


def _classify_support(row: Mapping[str, Any], *, same_label: bool, raw_q50: float, parallax_q25: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    source = str(row.get("anchor_support_class") or "")
    quality = str(row.get("quality_label") or "")
    if "low_conf_stress" in quality or source.startswith("A_STRESS"):
        reasons.append("stress_or_low_conf_source")
        return "A_STRESS", reasons
    if _bool_from_row(row, "zero_conf_flag"):
        reasons.append("zero_conf")
        return "A_STRESS", reasons
    if _bool_from_row(row, "dynamic_risk_flag"):
        reasons.append("dynamic_risk")
        return "A_RISK", reasons
    if _bool_from_row(row, "cross_boundary_flag"):
        reasons.append("cross_boundary")
        return "A_RISK", reasons
    raw = safe_float(row.get("raw_overlap_residual"))
    parallax = safe_float(row.get("parallax_score"))
    if source in {"A_STRONG_MATURE", "A_STRONG_BOOTSTRAP"}:
        return source, reasons
    if not same_label:
        reasons.append("semantic_conflict")
        return "A_RISK", reasons
    raw_ok = raw is not None and raw <= raw_q50
    parallax_ok = parallax is not None and parallax >= parallax_q25
    raw_coord_ok = _bool_from_row(row, "raw_coord_available")
    if raw_ok and (parallax_ok or raw_coord_ok):
        reasons.append("soft_geometry_reliable")
        return "A_WEAK_RELIABLE", reasons
    if source == "A_RISK":
        reasons.append("source_a_risk_low_soft_reliability")
        return "A_RISK", reasons
    reasons.append("context_or_low_observability")
    return "A_CONTEXT_DEGENERATE", reasons


def _risk_multiplier(support_class: str, row: Mapping[str, Any]) -> float:
    if support_class == "A_STRESS":
        return 0.0
    if support_class == "A_RISK":
        return 0.08
    if support_class == "A_CONTEXT_DEGENERATE":
        return 0.25
    if support_class == "A_WEAK_RELIABLE":
        return 0.80
    return 1.0


def _support_state(row: Mapping[str, Any], w_fit: float, support_class: str) -> str:
    if support_class in {"A_STRONG_MATURE", "A_STRONG_BOOTSTRAP", "A_WEAK_RELIABLE"} and w_fit > 0:
        return "UPDATE_CANDIDATE"
    if support_class == "A_CONTEXT_DEGENERATE":
        return "HOLD_OR_ABSTAIN_CANDIDATE"
    if support_class == "A_RISK":
        return "RESET_RISK_CANDIDATE"
    return "ABSTAIN_CANDIDATE"


def _load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["q_features"].detach().cpu().float().numpy(), payload["k_features"].detach().cpu().float().numpy()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.anchor_rows)
    q_features, k_features = _load_features(args.features)
    if len(df) != q_features.shape[0] or len(df) != k_features.shape[0]:
        raise ValueError(f"row/features mismatch: rows={len(df)} q={q_features.shape} k={k_features.shape}")

    raw_q50 = finite_quantile(df["raw_overlap_residual"].tolist(), 0.50) or 0.15
    raw_q75 = finite_quantile(df["raw_overlap_residual"].tolist(), 0.75) or raw_q50
    conf_resid_q75 = finite_quantile(df["confidence_weighted_residual"].tolist(), 0.75) or raw_q75
    shape_q75 = finite_quantile(df["local_shape_residual"].tolist(), 0.75) or raw_q75
    parallax_q25 = finite_quantile(df["parallax_score"].tolist(), 0.25) or 0.0
    parallax_q75 = finite_quantile(df["parallax_score"].tolist(), 0.75) or 1.0

    rows: list[dict[str, Any]] = []
    for idx, row_s in df.iterrows():
        row = row_s.to_dict()
        same = _same_label(row)
        support_class, reasons = _classify_support(row, same_label=same, raw_q50=raw_q50, parallax_q25=parallax_q25)
        prev_conf = safe_float(row.get("prev_sem_conf"))
        curr_conf = safe_float(row.get("curr_sem_conf"))
        w_conf = math.sqrt(clamp01(prev_conf) * clamp01(curr_conf)) if prev_conf is not None and curr_conf is not None else 0.0
        w_sem = _semantic_compatibility(row, same)
        raw = safe_float(row.get("raw_overlap_residual"))
        conf_resid = safe_float(row.get("confidence_weighted_residual"))
        shape = safe_float(row.get("local_shape_residual"))
        parallax = safe_float(row.get("parallax_score"))
        w_shape = _exp_score(shape, shape_q75, 0.50)
        w_overlap = _exp_score(conf_resid if conf_resid is not None else raw, conf_resid_q75, 0.0)
        w_parallax = clamp01((parallax or 0.0) / max(parallax_q75, 1e-12))
        q_avail = _bool_from_row(row, "feature_q_available")
        k_avail = _bool_from_row(row, "feature_k_available")
        w_memory = 1.0 if q_avail and k_avail else 0.0
        w_risk = _risk_multiplier(support_class, row)
        w_fit = w_conf * w_sem * w_shape * w_overlap * w_parallax * w_memory * w_risk
        if support_class in {"A_RISK", "A_STRESS", "A_CONTEXT_DEGENERATE"}:
            # Keep risk/context visible for absence audit but prevent them from becoming strong positives.
            w_fit = min(w_fit, {"A_RISK": 0.02, "A_STRESS": 0.0, "A_CONTEXT_DEGENERATE": 0.08}[support_class])
        out = {
            "seq": seq_norm(row.get("seq")),
            "prev_chunk": safe_int(row.get("prev_chunk")),
            "curr_chunk": safe_int(row.get("curr_chunk")),
            "case_label": row.get("case_label"),
            "quality_label": row.get("quality_label"),
            "prev_patch_id": safe_int(row.get("prev_patch_id")),
            "curr_patch_id": safe_int(row.get("curr_patch_id")),
            "prev_frame_id": safe_int(row.get("prev_frame_id")),
            "curr_frame_id": safe_int(row.get("curr_frame_id")),
            "q_feature_available": q_avail,
            "k_feature_available": k_avail,
            "q_feature_source": "v85_direct_pca_swa_current_q",
            "k_feature_source": "v85_direct_pca_swa_cache_k",
            "semantic_compatibility": w_sem,
            "same_label": same,
            "same_role": parse_bool(row.get("same_role")),
            "cross_boundary_flag": parse_bool(row.get("cross_boundary_flag")),
            "dynamic_flag": parse_bool(row.get("dynamic_risk_flag")),
            "zero_conf_flag": parse_bool(row.get("zero_conf_flag")),
            "raw_overlap_residual": raw,
            "confidence_weighted_residual": conf_resid,
            "local_shape_residual": shape,
            "pairwise_distance_ratio_residual": safe_float(row.get("pairwise_distance_ratio_residual")),
            "parallax_score": parallax,
            "local_3d_spread_prev": safe_float(row.get("local_3d_spread_prev")),
            "local_3d_spread_curr": safe_float(row.get("local_3d_spread_curr")),
            "read_usage_current": safe_float(row.get("read_usage_current")),
            "swa_qk_proxy": safe_float(row.get("swa_qk_proxy")),
            "true_route_available": parse_bool(row.get("true_route_available")),
            "true_route_mass_if_available": safe_float(row.get("true_route_mass")),
            "w_conf": w_conf,
            "w_sem": w_sem,
            "w_shape": w_shape,
            "w_overlap": w_overlap,
            "w_parallax": w_parallax,
            "w_memory": w_memory,
            "w_risk": w_risk,
            "w_fit": w_fit,
            "support_class": support_class,
            "support_state_candidate": _support_state(row, w_fit, support_class),
            "risk_reason": ";".join([str(row.get("risk_reason") or ""), *reasons]).strip(";"),
            "source_anchor_support_class": row.get("anchor_support_class"),
            "raw_coord_available": parse_bool(row.get("raw_coord_available")),
            "raw_lookup_status": row.get("raw_lookup_status"),
            "feature_source_path": row.get("feature_source_path"),
            "anchor_row_index": int(idx),
            "pair_id": row.get("pair_id"),
        }
        rows.append(out)

    by_pair: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[pair_key(row)].append(row)
    for key, items in sorted(grouped.items()):
        idxs = np.asarray([int(item["anchor_row_index"]) for item in items], dtype=np.int64)
        weights = np.asarray([float(item["w_fit"] or 0.0) for item in items], dtype=np.float64)
        case_labels = Counter(str(item.get("case_label") or "") for item in items)
        quality_labels = Counter(str(item.get("quality_label") or "") for item in items)
        support_counts = Counter(str(item.get("support_class") or "") for item in items)
        nonzero = int(np.sum(weights > 0.0))
        neff = effective_sample_size(weights)
        rank_q = weighted_rank(q_features[idxs, : args.feature_dim], weights)
        rank_k = weighted_rank(k_features[idxs, : args.feature_dim], weights)
        support_dim8 = bool(neff >= 3 * args.feature_dim and rank_q >= int(0.75 * args.feature_dim) and rank_k >= int(0.75 * args.feature_dim))
        support_dim4 = bool(neff >= 12 and rank_q >= 3 and rank_k >= 3)
        risk_values = np.asarray([1.0 - float(item["w_risk"] or 0.0) for item in items], dtype=np.float64)
        parallax_values = np.asarray([float(item["w_parallax"] or 0.0) for item in items], dtype=np.float64)
        absence_score = (1.0 - min(1.0, neff / max(3 * args.feature_dim, 1))) * (1.0 - float(np.mean(parallax_values))) * (
            1.0 + float(np.mean(risk_values))
        )
        if support_dim8:
            prelim = "UPDATE"
        elif absence_score >= 0.35:
            prelim = "HOLD_OR_ABSTAIN"
        elif support_counts.get("A_RISK", 0) > len(items) * 0.5:
            prelim = "RESET_RISK"
        else:
            prelim = "ABSTAIN"
        by_pair.append(
            {
                "seq": key[0],
                "prev_chunk": key[1],
                "curr_chunk": key[2],
                "case_label": case_labels.most_common(1)[0][0],
                "quality_label": quality_labels.most_common(1)[0][0],
                "pair_row_count": len(items),
                "nonzero_weight_count": nonzero,
                "effective_sample_size": neff,
                "weighted_rank_q": rank_q,
                "weighted_rank_k": rank_k,
                "strong_count": support_counts.get("A_STRONG_MATURE", 0) + support_counts.get("A_STRONG_BOOTSTRAP", 0),
                "weak_reliable_count": support_counts.get("A_WEAK_RELIABLE", 0),
                "context_count": support_counts.get("A_CONTEXT_DEGENERATE", 0),
                "risk_count": support_counts.get("A_RISK", 0),
                "stress_count": support_counts.get("A_STRESS", 0),
                "mean_w_fit": float(np.mean(weights)) if weights.size else 0.0,
                "q_feature_availability": float(np.mean([bool(item["q_feature_available"]) for item in items])),
                "k_feature_availability": float(np.mean([bool(item["k_feature_available"]) for item in items])),
                "support_sufficient_for_dim8": support_dim8,
                "support_sufficient_for_dim4": support_dim4,
                "support_state_preliminary": prelim,
                "anchor_absence_score": absence_score,
                "mean_parallax_weight": float(np.mean(parallax_values)) if parallax_values.size else 0.0,
                "mean_risk_score": float(np.mean(risk_values)) if risk_values.size else 0.0,
                "raw_coord_available_ratio": float(np.mean([bool(item["raw_coord_available"]) for item in items])),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "soft_pair_rows.csv", rows, fields=ROW_FIELDS)
    write_csv(args.out_dir / "soft_pair_by_seq_chunk.csv", by_pair)
    summary = {
        "phase": "Phase1_soft_pair_universe_build",
        "row_count": len(rows),
        "pair_count": len(by_pair),
        "feature_dim": args.feature_dim,
        "weight_quantiles": {
            "raw_overlap_q50": raw_q50,
            "raw_overlap_q75": raw_q75,
            "confidence_weighted_residual_q75": conf_resid_q75,
            "shape_residual_q75": shape_q75,
            "parallax_q25": parallax_q25,
            "parallax_q75": parallax_q75,
        },
        "support_class_counts": dict(Counter(row["support_class"] for row in rows)),
        "support_state_counts": dict(Counter(row["support_state_candidate"] for row in rows)),
        "pair_support_state_counts": dict(Counter(row["support_state_preliminary"] for row in by_pair)),
        "notes": [
            "Soft weights are derived from v85 rows and direct PCA Q/cache-K features.",
            "Risk/context/stress rows are retained for absence audit but capped or zeroed for C-fit support.",
            "No QK compatibility proxy is used as a replacement for missing Q/K features.",
        ],
    }
    write_json(args.out_dir / "build_summary.json", summary)
    print(f"row_count={len(rows)}")
    print(f"pair_count={len(by_pair)}")
    print(f"support_dim8_pairs={sum(1 for row in by_pair if row['support_sufficient_for_dim8'])}")
    print(f"bad_support_dim8_pairs={sum(1 for row in by_pair if row['case_label']=='bad' and row['support_sufficient_for_dim8'])}")


if __name__ == "__main__":
    main()
