from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


V37_BASELINE = {
    "4D_ARI": 0.42599481039581194,
    "4D_purity": 0.8673519940549913,
    "4D_completeness": 0.5056972999752292,
    "temporal_span_mean": 1.702673104336451,
    "scene0081_ARI": 0.20186794681675915,
}

V43_MINIMUM_GATE = {
    "4D_ARI": 0.485,
    "4D_purity": 0.875,
    "4D_completeness": 0.555,
    "temporal_span_mean": 1.70,
    "scene0081_ARI": 0.270,
}

V43_STRONG_GATE = {
    "4D_ARI": 0.520,
    "4D_purity": 0.880,
    "4D_completeness": 0.580,
    "temporal_span_mean": 1.75,
    "scene0081_ARI": 0.300,
}


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    cand = as_float(candidate.get(key))
    base = as_float(baseline.get(key))
    if cand is None or base is None:
        return None
    return float(cand - base)


def threshold_gate(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key, threshold in thresholds.items():
        value = as_float(metrics.get(key))
        checks[f"{key}_pass"] = bool(value is not None and value >= float(threshold))
        checks[f"{key}_value"] = value
        checks[f"{key}_threshold"] = float(threshold)
    checks["pass"] = bool(all(value for key, value in checks.items() if key.endswith("_pass")))
    return checks


def minimum_significant_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    return threshold_gate(metrics, V43_MINIMUM_GATE)


def strong_conference_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    return threshold_gate(metrics, V43_STRONG_GATE)


def compactness_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "birth_from_d4rt_tube_count_pass": int(metrics.get("birth_from_d4rt_tube_count") or 0) == 0,
        "mean_predictions_per_scene_pass": (
            as_float(metrics.get("mean_predictions_per_scene")) is not None
            and float(metrics["mean_predictions_per_scene"]) <= 150.0
        ),
        "duplicate_rate_pass": as_float(metrics.get("duplicate_rate")) is not None
        and float(metrics["duplicate_rate"]) <= 0.05,
        "conflict_rate_pass": as_float(metrics.get("conflict_rate")) is not None
        and float(metrics["conflict_rate"]) <= 0.10,
        "unknown_tube_ratio_pass": as_float(metrics.get("unknown_tube_ratio")) is not None
        and float(metrics["unknown_tube_ratio"]) <= 0.35,
        "changed_object_ratio_pass": as_float(metrics.get("changed_object_ratio")) is not None
        and float(metrics["changed_object_ratio"]) <= 0.30,
    }
    checks["pass"] = bool(all(checks.values()))
    return checks


def control_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "real_minus_shuffled_pass": as_float(metrics.get("real_minus_shuffled")) is not None
        and float(metrics["real_minus_shuffled"]) >= 0.30,
        "real_minus_no_temporal_pass": as_float(metrics.get("real_minus_no_temporal")) is not None
        and float(metrics["real_minus_no_temporal"]) >= 0.25,
        "real_minus_mask_only_pass": as_float(metrics.get("real_minus_mask_only")) is not None
        and float(metrics["real_minus_mask_only"]) >= 0.25,
    }
    checks["pass"] = bool(all(checks.values()))
    return checks


def scene_delta_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    metric: str,
    scene_key: str = "scene",
) -> list[dict[str, Any]]:
    base_by_scene = {str(row.get(scene_key)): row for row in baseline_rows if row.get(scene_key)}
    cand_by_scene = {str(row.get(scene_key)): row for row in candidate_rows if row.get(scene_key)}
    rows = []
    for scene in sorted(set(base_by_scene) & set(cand_by_scene)):
        base = as_float(base_by_scene[scene].get(metric))
        cand = as_float(cand_by_scene[scene].get(metric))
        if base is None or cand is None:
            continue
        rows.append({"scene": scene, "metric": metric, "baseline": base, "candidate": cand, "delta": cand - base})
    return rows


def bootstrap_lower_bound(deltas: list[float], *, seed: int = 4322, samples: int = 2000, alpha: float = 0.05) -> float | None:
    values = np.asarray([float(v) for v in deltas if math.isfinite(float(v))], dtype=np.float64)
    if values.size == 0:
        return None
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for idx in range(int(samples)):
        draw = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(draw))
    return float(np.quantile(means, float(alpha / 2.0)))


def significance_summary(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    seed: int = 4322,
    samples: int = 2000,
) -> dict[str, Any]:
    ari_rows = scene_delta_rows(baseline_rows, candidate_rows, metric="4D_ARI")
    comp_rows = scene_delta_rows(baseline_rows, candidate_rows, metric="4D_completeness")
    ari_deltas = [float(row["delta"]) for row in ari_rows]
    comp_deltas = [float(row["delta"]) for row in comp_rows]
    checks = {
        "scene_count": int(len(ari_rows)),
        "ari_delta_lower95": bootstrap_lower_bound(ari_deltas, seed=seed, samples=samples),
        "completeness_delta_lower95": bootstrap_lower_bound(comp_deltas, seed=seed + 17, samples=samples),
        "median_scene_delta_ARI": float(np.median(ari_deltas)) if ari_deltas else None,
        "scene_delta_ARI_ge_002_count": int(sum(1 for value in ari_deltas if value >= 0.02)),
        "scene_delta_ARI_ge_004_count": int(sum(1 for value in ari_deltas if value >= 0.04)),
    }
    checks["lower95_delta_ARI_pass"] = checks["ari_delta_lower95"] is not None and checks["ari_delta_lower95"] >= 0.025
    checks["lower95_delta_completeness_pass"] = (
        checks["completeness_delta_lower95"] is not None and checks["completeness_delta_lower95"] >= 0.020
    )
    checks["median_scene_delta_ARI_pass"] = checks["median_scene_delta_ARI"] is not None and checks["median_scene_delta_ARI"] > 0.0
    checks["four_of_five_scene_delta_ARI_ge_002_pass"] = checks["scene_delta_ARI_ge_002_count"] >= 4
    checks["three_of_five_scene_delta_ARI_ge_004_pass"] = checks["scene_delta_ARI_ge_004_count"] >= 3
    checks["object_level_bootstrap_status"] = "not_available_from_current_aggregate_artifacts"
    checks["tube_level_bootstrap_status"] = "not_available_from_current_aggregate_artifacts"
    checks["pass"] = bool(
        checks["lower95_delta_ARI_pass"]
        and checks["lower95_delta_completeness_pass"]
        and checks["median_scene_delta_ARI_pass"]
        and checks["four_of_five_scene_delta_ARI_ge_002_pass"]
        and checks["three_of_five_scene_delta_ARI_ge_004_pass"]
    )
    return {"checks": checks, "ari_scene_delta_rows": ari_rows, "completeness_scene_delta_rows": comp_rows}


@dataclass(frozen=True)
class StageDecision:
    label: str
    status: str
    metrics: dict[str, Any]
    reason: str


def stage1_decision(metrics: dict[str, Any], significance: dict[str, Any]) -> StageDecision:
    min_gate = minimum_significant_gate(metrics)
    compact_gate = compactness_gate(metrics)
    control = control_gate(metrics)
    sig_pass = bool(significance.get("checks", {}).get("pass"))
    if min_gate["pass"] and compact_gate["pass"] and control["pass"] and sig_pass:
        return StageDecision("GO_STAGE1_SIGNIFICANT_MATCHING_BREAKTHROUGH", "pass", metrics, "all minimum gates passed")
    return StageDecision("NO_GO_MATCHING_NOT_SIGNIFICANT", "fail", metrics, "one or more Stage-1 minimum gates failed")
