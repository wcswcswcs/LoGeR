#!/usr/bin/env python3
"""ACL2 v70 RADIO sidecar SWA offline residual oracle.

This diagnostic tests whether RADIO/RADSeg sidecar fields produce better
overlap K/V source weights than geometry or shuffled controls. It does not
modify an HMC trajectory, so its residual-proxy gate is separated from the real
R5 future/head-tail gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from diagnose_v67_offline_scale_controller import DEFAULT_GT, _load_kitti_gt, _load_postmerge_trajectory, _load_trace
    from diagnose_v67_overlap_pair_action_oracle import _load_pair, _parse_source
    from diagnose_v69_centered_overlap_pair_action_oracle import _make_baseline_row
    from diagnose_v70_radio_merge_oracle import (
        _feature_cosine,
        _finite_mean,
        _finite_median,
        _component_match_mask,
        _index_sidecars,
        _load_sidecar,
        _np,
        _pair_array,
        _sample_sidecar,
        _safe_tag,
        _write_csv,
    )
    from v70_radio_sidecar_common import parse_chunks, utc_now
except ImportError:  # pragma: no cover
    from tools.diagnose_v67_offline_scale_controller import DEFAULT_GT, _load_kitti_gt, _load_postmerge_trajectory, _load_trace
    from tools.diagnose_v67_overlap_pair_action_oracle import _load_pair, _parse_source
    from tools.diagnose_v69_centered_overlap_pair_action_oracle import _make_baseline_row
    from tools.diagnose_v70_radio_merge_oracle import (
        _feature_cosine,
        _finite_mean,
        _finite_median,
        _component_match_mask,
        _index_sidecars,
        _load_sidecar,
        _np,
        _pair_array,
        _sample_sidecar,
        _safe_tag,
        _write_csv,
    )
    from tools.v70_radio_sidecar_common import parse_chunks, utc_now


RADIO_CANDIDATES = {
    "swa_radio_kv_protect",
    "swa_radio_risky_cross_object_gate",
    "swa_radio_proxy_replace",
    "swa_radio_geometry_blend",
}
CONTROL_CANDIDATES = {
    "swa_current_label_trust",
    "swa_current_label_shuffle",
    "swa_current_confidence_shuffle",
    "swa_radio_component_shuffle",
    "swa_radio_feature_shuffle",
    "swa_radio_confidence_temporal_shuffle",
    "swa_same_entropy_random_proxy",
    "swa_same_count_random_components",
}
BASELINE_CANDIDATES = {"swa_geometry_only"}


def _float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _family(candidate_type: str) -> str:
    if candidate_type in RADIO_CANDIDATES:
        return "radio"
    if candidate_type in CONTROL_CANDIDATES:
        return "control"
    if candidate_type in BASELINE_CANDIDATES:
        return "baseline"
    return "unknown"


def _clip_weight(weights: np.ndarray, base: np.ndarray) -> np.ndarray:
    out = np.asarray(weights, dtype=np.float64).copy()
    out[~base] = 0.0
    out[~np.isfinite(out)] = 0.0
    out[out < 0.0] = 0.0
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _weighted_rmse(residual: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(max(_weighted_mean(residual * residual, weights), 0.0))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    vv = values[valid]
    ww = weights[valid]
    order = np.argsort(vv)
    vv = vv[order]
    ww = ww[order]
    cdf = np.cumsum(ww)
    cutoff = float(q) * float(cdf[-1])
    return float(vv[min(int(np.searchsorted(cdf, cutoff, side="left")), vv.size - 1)])


def _effective_sample_ratio(weights: np.ndarray, base: np.ndarray) -> float:
    valid = base & np.isfinite(weights) & (weights > 0.0)
    n = int(np.sum(base))
    if n <= 0 or not np.any(valid):
        return 0.0
    ww = weights[valid].astype(np.float64)
    eff = float((np.sum(ww) ** 2) / max(np.sum(ww * ww), 1e-12))
    return float(eff / max(n, 1))


def _entropy_ratio(weights: np.ndarray, base: np.ndarray) -> float:
    valid = base & np.isfinite(weights) & (weights > 0.0)
    n = int(np.sum(base))
    if n <= 0 or not np.any(valid):
        return 0.0
    ww = weights[valid].astype(np.float64)
    prob = ww / max(float(np.sum(ww)), 1e-12)
    entropy = -float(np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0))))
    return float(math.exp(entropy) / max(n, 1))


def _improvement(base_value: float, candidate_value: float) -> float:
    if not math.isfinite(base_value) or not math.isfinite(candidate_value) or abs(base_value) < 1e-12:
        return float("nan")
    return float((base_value - candidate_value) / abs(base_value))


def _random_same_count(base: np.ndarray, keep_count: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(base, dtype=np.float64)
    idx = np.where(base)[0]
    if idx.size == 0:
        return out
    keep = min(int(keep_count), int(idx.size))
    if keep <= 0:
        return out
    out[rng.choice(idx, size=keep, replace=False)] = 1.0
    return out


def _pair_features(
    pair: Mapping[str, Any],
    prev_s: Mapping[str, np.ndarray],
    curr_s: Mapping[str, np.ndarray],
    seed: int,
    curr_chunk: int,
    *,
    semantic_min_conf: float,
    min_feature_cos: float,
    component_match_mode: str,
) -> Dict[str, np.ndarray]:
    n = int(_np(pair["prev_overlap_points"]).shape[0])
    prev_conf = _pair_array(pair, "prev_conf", n, 1.0)
    curr_conf = _pair_array(pair, "curr_conf", n, 1.0)
    geom = np.nan_to_num(np.minimum(prev_conf, curr_conf), nan=0.0, posinf=0.0, neginf=0.0)
    base = prev_s["valid"] & curr_s["valid"] & (geom > 0.0)

    same_comp_id = (prev_s["component"] == curr_s["component"]) & (prev_s["component"] >= 0)
    radio_conf = np.nan_to_num(np.minimum(prev_s["confidence"], curr_s["confidence"]), nan=0.0)
    radio_stability = np.nan_to_num(np.minimum(prev_s["stability"], curr_s["stability"]), nan=0.0)
    radio_risk = np.nan_to_num(np.maximum(prev_s["risk"], curr_s["risk"]), nan=1.0)
    radio_interior = np.nan_to_num(np.minimum(prev_s["interior"], curr_s["interior"]), nan=0.0)
    feature_cos = np.nan_to_num(_feature_cosine(prev_s["feat"], curr_s["feat"]), nan=-1.0)
    static = np.clip(radio_conf * radio_stability * radio_interior * (1.0 - radio_risk), 0.0, 1.0)
    radio_score = np.clip(
        0.30 * radio_conf
        + 0.25 * radio_stability
        + 0.20 * radio_interior
        + 0.15 * np.clip(1.0 - radio_risk, 0.0, 1.0)
        + 0.10 * np.clip((feature_cos + 1.0) / 2.0, 0.0, 1.0),
        0.0,
        1.0,
    )

    prev_label = _pair_array(pair, "prev_semantic_labels", n, -1).astype(np.int64)
    curr_label = _pair_array(pair, "curr_semantic_labels", n, -2).astype(np.int64)
    prev_sem_conf = _pair_array(pair, "prev_semantic_conf", n, 0.0)
    curr_sem_conf = _pair_array(pair, "curr_semantic_conf", n, 0.0)
    same_label = (prev_label == curr_label) & (prev_label >= 0)
    sem_conf = np.nan_to_num(np.minimum(prev_sem_conf, curr_sem_conf), nan=0.0)
    same_feature = feature_cos >= float(min_feature_cos)
    same_feature_label = same_label & (sem_conf >= float(semantic_min_conf)) & same_feature
    same_comp = _component_match_mask(
        pair,
        prev_s,
        curr_s,
        semantic_min_conf=float(semantic_min_conf),
        min_feature_cos=float(min_feature_cos),
        component_match_mode=str(component_match_mode),
    )

    rng = np.random.default_rng(int(seed) + int(curr_chunk) * 2003)
    perm_comp = rng.permutation(n)
    perm_feat = rng.permutation(n)
    perm_conf = rng.permutation(n)
    perm_label = rng.permutation(n)
    same_comp_id_shuffle = (prev_s["component"] == curr_s["component"][perm_comp]) & (prev_s["component"] >= 0)
    feature_cos_shuffle = np.nan_to_num(_feature_cosine(prev_s["feat"], curr_s["feat"][perm_feat]), nan=-1.0)
    conf_shuffle = np.nan_to_num(np.minimum(prev_s["confidence"], curr_s["confidence"][perm_conf]), nan=0.0)
    stability_shuffle = np.nan_to_num(np.minimum(prev_s["stability"], curr_s["stability"][perm_conf]), nan=0.0)
    risk_shuffle = np.nan_to_num(np.maximum(prev_s["risk"], curr_s["risk"][perm_conf]), nan=1.0)
    static_shuffle = np.clip(conf_shuffle * stability_shuffle * radio_interior * (1.0 - risk_shuffle), 0.0, 1.0)
    label_shuffle = (prev_label == curr_label[perm_label]) & (prev_label >= 0)
    sem_conf_shuffle = np.nan_to_num(np.minimum(prev_sem_conf, curr_sem_conf[perm_conf]), nan=0.0)
    same_feature_shuffle = feature_cos_shuffle >= float(min_feature_cos)
    same_feature_label_shuffle = label_shuffle & (sem_conf_shuffle >= float(semantic_min_conf)) & same_feature_shuffle
    if component_match_mode == "id":
        same_comp_shuffle = same_comp_id_shuffle
    elif component_match_mode == "feature":
        same_comp_shuffle = same_feature_shuffle
    elif component_match_mode == "feature_label":
        same_comp_shuffle = same_feature_label_shuffle
    else:
        raise ValueError(f"unsupported component_match_mode={component_match_mode!r}")

    return {
        "base": base,
        "geom": geom,
        "same_comp": same_comp.astype(np.float64),
        "same_comp_bool": same_comp,
        "same_component_id_count": int(np.sum(base & same_comp_id)),
        "same_feature_match_count": int(np.sum(base & same_feature)),
        "same_feature_label_match_count": int(np.sum(base & same_feature_label)),
        "same_component_count": int(np.sum(base & same_comp)),
        "component_match_mode": str(component_match_mode),
        "same_comp_shuffle": same_comp_shuffle.astype(np.float64),
        "radio_conf": radio_conf,
        "radio_stability": radio_stability,
        "radio_risk": radio_risk,
        "radio_interior": radio_interior,
        "feature_cos": feature_cos,
        "feature_cos_shuffle": feature_cos_shuffle,
        "static": static,
        "static_shuffle": static_shuffle,
        "radio_score": radio_score,
        "same_label": same_label.astype(np.float64),
        "label_shuffle": label_shuffle.astype(np.float64),
        "sem_conf": sem_conf,
        "sem_conf_shuffle": sem_conf_shuffle,
    }


def _candidate_weights(feat: Mapping[str, np.ndarray], *, alpha: float, beta: float, seed: int, curr_chunk: int) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    base = feat["base"]
    geom = feat["geom"]
    cross = 1.0 - feat["same_comp"]
    risk = feat["radio_risk"]
    rng = np.random.default_rng(int(seed) + int(curr_chunk) * 3001 + int(alpha * 1000) + int(beta * 1000))

    radio_kv = geom * (1.0 + float(alpha) * feat["static"] * (1.0 + 0.5 * feat["same_comp"])) * np.exp(-float(beta) * risk * (0.5 + cross))
    radio_gate = geom * np.exp(-float(beta) * (risk + 0.5 * cross))
    radio_replace = feat["radio_score"] * (0.25 + 0.75 * geom)
    radio_blend = geom * (1.0 + float(alpha) * feat["static"]) * np.exp(-float(beta) * risk)

    label = geom * (1.0 + float(alpha) * feat["same_label"] * feat["sem_conf"])
    label_shuffle = geom * (1.0 + float(alpha) * feat["label_shuffle"] * feat["sem_conf"])
    conf_shuffle = geom * (1.0 + float(alpha) * feat["same_label"] * feat["sem_conf_shuffle"])

    comp_shuffle = geom * (1.0 + float(alpha) * feat["static"] * (1.0 + 0.5 * feat["same_comp_shuffle"])) * np.exp(-float(beta) * risk)
    feat_shuffle = geom * (
        1.0
        + float(alpha)
        * (0.30 * feat["radio_conf"] + 0.25 * feat["radio_stability"] + 0.20 * feat["radio_interior"] + 0.25 * np.clip((feat["feature_cos_shuffle"] + 1.0) / 2.0, 0.0, 1.0))
    ) * np.exp(-float(beta) * risk)
    conf_temp_shuffle = geom * (1.0 + float(alpha) * feat["static_shuffle"]) * np.exp(-float(beta) * feat["radio_risk"])
    same_count = _random_same_count(base, int(np.sum(base & feat["same_comp_bool"])), rng)
    same_entropy = _random_same_count(base, int(max(1, np.sum(base) * _effective_sample_ratio(radio_blend, base))), rng)

    return {
        "swa_geometry_only": (geom, {"mode": "geometry_only"}),
        "swa_current_label_trust": (label, {"mode": "current_label_trust"}),
        "swa_current_label_shuffle": (label_shuffle, {"mode": "current_label_shuffle"}),
        "swa_current_confidence_shuffle": (conf_shuffle, {"mode": "current_confidence_shuffle"}),
        "swa_radio_kv_protect": (radio_kv, {"mode": "radio_overlap_kv_protect"}),
        "swa_radio_risky_cross_object_gate": (radio_gate, {"mode": "radio_risky_cross_object_gate"}),
        "swa_radio_proxy_replace": (radio_replace, {"mode": "radio_proxy_replace"}),
        "swa_radio_geometry_blend": (radio_blend, {"mode": "radio_geometry_blend"}),
        "swa_radio_component_shuffle": (comp_shuffle, {"mode": "radio_component_shuffle"}),
        "swa_radio_feature_shuffle": (feat_shuffle, {"mode": "radio_feature_shuffle"}),
        "swa_radio_confidence_temporal_shuffle": (conf_temp_shuffle, {"mode": "radio_confidence_temporal_shuffle"}),
        "swa_same_count_random_components": (geom * same_count, {"mode": "same_count_random_components"}),
        "swa_same_entropy_random_proxy": (geom * same_entropy, {"mode": "same_entropy_random_proxy"}),
    }


def _row_for_candidate(
    *,
    source_label: str,
    run_dir: Path,
    pair_file: Path,
    prev_chunk: int,
    curr_chunk: int,
    residual: np.ndarray,
    base: np.ndarray,
    same_comp: np.ndarray,
    weights: np.ndarray,
    geometry_rmse: float,
    geometry_mean: float,
    candidate_type: str,
    alpha: float,
    beta: float,
    meta: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    weights = _clip_weight(weights, base)
    fit_point_count = int(np.sum(weights > 0.0))
    rmse = _weighted_rmse(residual, weights)
    mean = _weighted_mean(residual, weights)
    p90 = _weighted_quantile(residual, weights, 0.90)
    improvement = _improvement(geometry_rmse, rmse)
    mean_improvement = _improvement(geometry_mean, mean)
    entropy_ratio = _entropy_ratio(weights, base)
    eff_ratio = _effective_sample_ratio(weights, base)
    internal = same_comp & base
    cross = (~same_comp) & base
    return {
        "source_label": source_label,
        "source_run": str(run_dir),
        "overlap_pair_file": str(pair_file),
        "prev_chunk": int(prev_chunk),
        "curr_chunk": int(curr_chunk),
        "candidate_type": candidate_type,
        "candidate_family": _family(candidate_type),
        "alpha": float(alpha),
        "beta": float(beta),
        "fit_point_count": fit_point_count,
        "base_valid_count": int(np.sum(base)),
        "weighted_overlap_rmse_m": rmse,
        "weighted_overlap_mean_m": mean,
        "weighted_overlap_p90_m": p90,
        "geometry_overlap_rmse_m": geometry_rmse,
        "geometry_overlap_mean_m": geometry_mean,
        "swa_overlap_rmse_improvement_vs_geometry": improvement,
        "swa_overlap_mean_improvement_vs_geometry": mean_improvement,
        "object_internal_weighted_residual_m": _weighted_rmse(residual, weights * internal.astype(np.float64)),
        "object_cross_weighted_residual_m": _weighted_rmse(residual, weights * cross.astype(np.float64)),
        "same_object_weight_mass_ratio": _weighted_mean(internal.astype(np.float64), weights),
        "effective_sample_ratio": eff_ratio,
        "attention_entropy_ratio": entropy_ratio,
        "empty_attention_rows": int(fit_point_count == 0),
        "future_after_overlap_mean_baseline": baseline.get("future_after_overlap_mean"),
        "head_to_tail_transfer_ratio_mean_baseline": baseline.get("head_to_tail_transfer_ratio_mean"),
        "intra_scale_variance_mean_baseline": baseline.get("intra_scale_variance_mean"),
        "future_after_overlap_improvement_vs_baseline": 0.0,
        "future_metric_source": "baseline_unchanged_no_online_swa",
        "future_metric_available": False,
        **dict(meta),
    }


def _best_by_chunk(rows: Sequence[Dict[str, Any]], family: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if row.get("candidate_family") != family:
            continue
        chunk = int(row.get("curr_chunk"))
        cur = out.get(chunk)
        if cur is None or _float(row.get("swa_overlap_rmse_improvement_vs_geometry"), -1e9) > _float(cur.get("swa_overlap_rmse_improvement_vs_geometry"), -1e9):
            out[chunk] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=_parse_source, required=True)
    parser.add_argument("--radio-sidecar-dir", type=Path, action="append", required=True)
    parser.add_argument("--overlap-pairs-dir", type=Path, default=None)
    parser.add_argument("--target-chunks", default="6,7,8,10,12,19,20,29,30,31,32")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--head-len", type=int, default=10)
    parser.add_argument("--alpha", type=float, action="append", default=None)
    parser.add_argument("--beta", type=float, action="append", default=None)
    parser.add_argument("--semantic-min-conf", type=float, default=0.5)
    parser.add_argument("--min-feature-cos", type=float, default=-0.10)
    parser.add_argument(
        "--component-match-mode",
        choices=["id", "feature", "feature_label"],
        default="id",
        help="Proxy used for cross-side RADIO/RADSeg object correspondence.",
    )
    parser.add_argument("--min-proxy-improvement-ratio", type=float, default=0.05)
    parser.add_argument("--min-entropy-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7071)
    args = parser.parse_args()

    source_label, run_dir = args.source
    pairs_dir = args.overlap_pairs_dir or (run_dir / "overlap_pairs")
    target_chunks = set(parse_chunks(args.target_chunks))
    pair_files = [
        path for path in sorted(pairs_dir.glob("chunk_*_*.pt"))
        if int(path.stem.split("_")[-1]) in target_chunks
    ]
    if not pair_files:
        raise FileNotFoundError(f"No target overlap pair files in {pairs_dir}")

    trace = _load_trace(run_dir / "merge_state_trace.jsonl")
    frames, poses, _ = _load_postmerge_trajectory(run_dir / "postmerge_global_pose.jsonl")
    _, _, gt_pos = _load_kitti_gt(args.gt)
    baseline = _make_baseline_row(frames, poses, gt_pos, trace, args.chunk_size, args.chunk_overlap, args.head_len)
    sidecar_index = _index_sidecars(args.radio_sidecar_dir)
    sidecar_cache: Dict[int, Dict[str, Any]] = {}
    alphas = args.alpha if args.alpha is not None else [0.05, 0.10, 0.20]
    betas = args.beta if args.beta is not None else [0.5, 1.0, 2.0]

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for pair_file in pair_files:
        pair = _load_pair(pair_file)
        curr_chunk = int(pair.get("curr_chunk"))
        prev_chunk = int(pair.get("prev_chunk", curr_chunk - 1))
        try:
            prev_sidecar = _load_sidecar(sidecar_index, prev_chunk, sidecar_cache)
            curr_sidecar = _load_sidecar(sidecar_index, curr_chunk, sidecar_cache)
            prev_s, prev_order = _sample_sidecar(pair, prev_sidecar, "prev")
            curr_s, curr_order = _sample_sidecar(pair, curr_sidecar, "curr")
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "pair_file": str(pair_file),
                "prev_chunk": prev_chunk,
                "curr_chunk": curr_chunk,
                "failure": f"input_error:{type(exc).__name__}:{exc}",
            })
            continue

        prev = _np(pair["prev_overlap_points"], dtype=np.float64)
        curr = _np(pair["curr_overlap_points"], dtype=np.float64)
        residual = np.linalg.norm(prev - curr, axis=1)
        feat = _pair_features(
            pair,
            prev_s,
            curr_s,
            int(args.seed),
            curr_chunk,
            semantic_min_conf=float(args.semantic_min_conf),
            min_feature_cos=float(args.min_feature_cos),
            component_match_mode=str(args.component_match_mode),
        )
        base = feat["base"]
        geometry_weights = _clip_weight(feat["geom"], base)
        geometry_rmse = _weighted_rmse(residual, geometry_weights)
        geometry_mean = _weighted_mean(residual, geometry_weights)
        for alpha in alphas:
            for beta in betas:
                for candidate_type, (weights, meta) in _candidate_weights(
                    feat,
                    alpha=float(alpha),
                    beta=float(beta),
                    seed=int(args.seed),
                    curr_chunk=curr_chunk,
                ).items():
                    row = _row_for_candidate(
                        source_label=source_label,
                        run_dir=run_dir,
                        pair_file=pair_file,
                        prev_chunk=prev_chunk,
                        curr_chunk=curr_chunk,
                        residual=residual,
                        base=base,
                        same_comp=feat["same_comp_bool"],
                        weights=weights,
                        geometry_rmse=geometry_rmse,
                        geometry_mean=geometry_mean,
                        candidate_type=candidate_type,
                        alpha=float(alpha),
                        beta=float(beta),
                        meta=meta,
                        baseline=baseline,
                    )
                    row.update({
                        "prev_pixel_coord_order": prev_order,
                        "curr_pixel_coord_order": curr_order,
                        "component_match_mode": str(args.component_match_mode),
                        "same_component_count": int(feat.get("same_component_count", 0)),
                        "same_component_id_count": int(feat.get("same_component_id_count", 0)),
                        "same_feature_match_count": int(feat.get("same_feature_match_count", 0)),
                        "same_feature_label_match_count": int(feat.get("same_feature_label_match_count", 0)),
                        "proxy_support_pass": bool(int(row["fit_point_count"]) > 0),
                        "proxy_entropy_pass": bool(_float(row["attention_entropy_ratio"]) >= float(args.min_entropy_ratio)),
                        "proxy_improvement_pass": bool(
                            _float(row["swa_overlap_rmse_improvement_vs_geometry"]) >= float(args.min_proxy_improvement_ratio)
                        ),
                    })
                    row["swa_residual_proxy_gate_pass"] = bool(
                        row["proxy_support_pass"]
                        and row["proxy_entropy_pass"]
                        and row["proxy_improvement_pass"]
                    )
                    row["r5_swa_oracle_gate_pass"] = False
                    rows.append(row)

    best_radio = _best_by_chunk(rows, "radio")
    best_control = _best_by_chunk(rows, "control")
    best_baseline = _best_by_chunk(rows, "baseline")
    radio_beats_controls_chunks: List[int] = []
    for chunk, row in best_radio.items():
        radio_imp = _float(row.get("swa_overlap_rmse_improvement_vs_geometry"), -1e9)
        control_imp = _float(best_control.get(chunk, {}).get("swa_overlap_rmse_improvement_vs_geometry"), -1e9)
        baseline_imp = _float(best_baseline.get(chunk, {}).get("swa_overlap_rmse_improvement_vs_geometry"), -1e9)
        if bool(row.get("swa_residual_proxy_gate_pass")) and radio_imp > max(control_imp, baseline_imp):
            radio_beats_controls_chunks.append(chunk)

    best_radio_proxy = [
        row
        for row in best_radio.values()
        if bool(row.get("swa_residual_proxy_gate_pass"))
    ]
    best_radio_proxy_improvements = [row.get("swa_overlap_rmse_improvement_vs_geometry") for row in best_radio_proxy]
    residual_proxy_gate = bool(
        len(radio_beats_controls_chunks) >= 4
        and (_finite_median(best_radio_proxy_improvements) or float("-inf")) >= float(args.min_proxy_improvement_ratio)
    )
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        key = str(row.get("candidate_type"))
        counts.setdefault(key, {"rows": 0, "proxy_gate_pass": 0})
        counts[key]["rows"] += 1
        counts[key]["proxy_gate_pass"] += int(bool(row.get("swa_residual_proxy_gate_pass")))

    rows.sort(key=lambda row: (
        not bool(row.get("swa_residual_proxy_gate_pass")),
        -_float(row.get("swa_overlap_rmse_improvement_vs_geometry"), -1e9),
        -_float(row.get("attention_entropy_ratio"), -1e9),
        str(row.get("candidate_type")),
    ))
    radio_rows = [row for row in rows if row.get("candidate_family") == "radio"]
    control_rows = [row for row in rows if row.get("candidate_family") == "control"]
    summary = {
        "schema": "acl2_v70_radio_swa_oracle_summary_v1",
        "created_at": utc_now(),
        "source_label": source_label,
        "source_run": str(run_dir),
        "overlap_pairs_dir": str(pairs_dir),
        "radio_sidecar_dirs": [str(x) for x in args.radio_sidecar_dir],
        "target_chunks": sorted(target_chunks),
        "pair_files": len(pair_files),
        "rows": len(rows),
        "failures": failures,
        "candidate_counts": counts,
        "baseline_future_after_overlap_mean": baseline.get("future_after_overlap_mean"),
        "baseline_head_to_tail_transfer_ratio_mean": baseline.get("head_to_tail_transfer_ratio_mean"),
        "baseline_intra_scale_variance_mean": baseline.get("intra_scale_variance_mean"),
        "radio_proxy_gate_rows": sum(bool(row.get("swa_residual_proxy_gate_pass")) for row in radio_rows),
        "control_proxy_gate_rows": sum(bool(row.get("swa_residual_proxy_gate_pass")) for row in control_rows),
        "radio_proxy_gate_chunks": sorted({
            int(row.get("curr_chunk")) for row in radio_rows if bool(row.get("swa_residual_proxy_gate_pass"))
        }),
        "radio_beats_controls_chunks": sorted(radio_beats_controls_chunks),
        "median_best_radio_proxy_improvement": _finite_median(best_radio_proxy_improvements),
        "mean_radio_proxy_improvement": _finite_mean(row.get("swa_overlap_rmse_improvement_vs_geometry") for row in radio_rows),
        "mean_control_proxy_improvement": _finite_mean(row.get("swa_overlap_rmse_improvement_vs_geometry") for row in control_rows),
        "swa_residual_proxy_gate_pass": residual_proxy_gate,
        "r5_swa_oracle_gate_pass": False,
        "r6_online_allowed_by_this_oracle": False,
        "decision": "diagnostic_only_proxy_pass_no_online" if residual_proxy_gate else "no_go_r6_continue_r5_repair",
        "gate_rule": {
            "min_proxy_improvement_ratio": float(args.min_proxy_improvement_ratio),
            "min_entropy_ratio": float(args.min_entropy_ratio),
            "semantic_min_conf": float(args.semantic_min_conf),
            "min_feature_cos": float(args.min_feature_cos),
            "component_match_mode": str(args.component_match_mode),
            "alphas": [float(x) for x in alphas],
            "betas": [float(x) for x in betas],
            "proxy_pass_rule": ">=4 chunks where best RADIO residual proxy beats best control/baseline and median best RADIO proxy improvement >= threshold",
            "r5_gate_note": "False because this oracle does not alter trajectory or real SWA K/V taps; future_after_overlap remains baseline.",
        },
        "best_row": rows[0] if rows else {},
        "best_radio_row": next((row for row in rows if row.get("candidate_family") == "radio"), {}),
        "best_control_row": next((row for row in rows if row.get("candidate_family") == "control"), {}),
        "note": (
            "Offline SWA residual-weight oracle. It tests overlap K/V coherence only; "
            "future/head-tail improvements are not claimed without online SWA taps."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "radio_swa_oracle_results.csv", rows)
    (args.out_dir / "radio_swa_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# v70 RADIO SWA Oracle",
        "",
        f"- rows: `{summary['rows']}`",
        f"- swa_residual_proxy_gate_pass: `{summary['swa_residual_proxy_gate_pass']}`",
        f"- r5_swa_oracle_gate_pass: `{summary['r5_swa_oracle_gate_pass']}`",
        f"- radio_proxy_gate_rows: `{summary['radio_proxy_gate_rows']}`",
        f"- control_proxy_gate_rows: `{summary['control_proxy_gate_rows']}`",
        f"- radio_beats_controls_chunks: `{summary['radio_beats_controls_chunks']}`",
        "",
        "This is not an online SWA result; trajectory future/head-tail metrics are unchanged baseline context.",
    ]
    (args.out_dir / "radio_swa_oracle_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
