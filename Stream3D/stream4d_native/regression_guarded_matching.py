from __future__ import annotations

from typing import Any

from stream4d_native.matching_significance import as_float, delta


def regression_guard(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    min_delta_ari: float = 0.0,
    min_delta_completeness: float = 0.0,
    max_purity_drop: float = 0.003,
    max_temporal_span_drop: float = 0.03,
) -> dict[str, Any]:
    d_ari = delta(candidate, baseline, "4D_ARI")
    d_comp = delta(candidate, baseline, "4D_completeness")
    d_purity = delta(candidate, baseline, "4D_purity")
    d_span = delta(candidate, baseline, "temporal_span_mean")
    checks = {
        "delta_4D_ARI": d_ari,
        "delta_4D_completeness": d_comp,
        "delta_4D_purity": d_purity,
        "delta_temporal_span_mean": d_span,
        "ari_pass": d_ari is not None and d_ari >= float(min_delta_ari),
        "completeness_pass": d_comp is not None and d_comp >= float(min_delta_completeness),
        "purity_pass": d_purity is not None and d_purity >= -float(max_purity_drop),
        "temporal_span_pass": d_span is not None and d_span >= -float(max_temporal_span_drop),
    }
    checks["pass"] = bool(checks["ari_pass"] and checks["completeness_pass"] and checks["purity_pass"] and checks["temporal_span_pass"])
    return checks


def phase_d_semantic_gate(candidate: dict[str, Any], baseline: dict[str, Any], hard_scene_delta_ari: float | None) -> dict[str, Any]:
    guard = regression_guard(candidate, baseline, min_delta_ari=0.035, min_delta_completeness=0.015)
    guard["hard_scene_delta_ari"] = hard_scene_delta_ari
    guard["hard_scene_pass"] = hard_scene_delta_ari is not None and float(hard_scene_delta_ari) >= 0.050
    guard["changed_object_ratio_pass"] = as_float(candidate.get("changed_object_ratio")) is not None and float(candidate["changed_object_ratio"]) <= 0.20
    guard["pass"] = bool(guard["pass"] and guard["hard_scene_pass"] and guard["changed_object_ratio_pass"])
    return guard
