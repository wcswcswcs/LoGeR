from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v90_dev_risk_gate_repair as phase7e  # noqa: E402


OUT = ROOT / "outputs/audit/v90_phase7f_dev_scene_balanced_score_repair"
PHASE4_ROOT = ROOT / "outputs/audit/v90_phase4_geo_semantic_witness_cover"
PHASE7B = ROOT / "outputs/audit/v90_phase7b_dev_score_control_audit"
PHASE7E = ROOT / "outputs/audit/v90_phase7e_dev_risk_gate_repair"
BASE_VARIANT = "W9a_risk_balanced_p135_plus_carving"
REFERENCE_VARIANT = "R_W9a_h9_drop_broad_low_h9_5"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _rank01(values: list[float]) -> list[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0 for _ in values]
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        out[idx] = float(rank / denom)
    return out


def _rank_by_scene(rows: list[dict[str, Any]], values: list[float]) -> list[float]:
    out = [0.0 for _ in values]
    by_scene: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_scene.setdefault(str(row.get("scene_id", "")), []).append(idx)
    for indices in by_scene.values():
        ranks = _rank01([values[i] for i in indices])
        for idx, rank in zip(indices, ranks):
            out[idx] = rank
    return out


def _control_threshold() -> dict[str, Any]:
    path = PHASE7B / "best_variant_summary.json"
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("best_control_metrics", {}))


def _scene_reference_rows() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path, variants in [
        (PHASE4_ROOT / "mv_metric_rows.csv", {BASE_VARIANT: "W9a_original"}),
        (PHASE7E / "variant_metric_rows.csv", {REFERENCE_VARIANT: "phase7e_reference"}),
    ]:
        for row in _read_csv(path):
            label = variants.get(str(row.get("variant_id") or row.get("variant") or ""))
            if not label:
                continue
            scene = str(row.get("scene_id", ""))
            out.setdefault(scene, {})[f"{label}_MV_AP_window"] = _num(row.get("MV_AP_window"), _num(row.get("MV_AP")))
            out.setdefault(scene, {})[f"{label}_MV_AP50_window"] = _num(row.get("MV_AP50_window"), _num(row.get("MV_AP50")))
    return out


def _scene_stats(rows: list[dict[str, Any]], feature_map: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scene = str(row.get("scene_id", ""))
        key = (scene, _int(row.get("frame_id"), -1), _int(row.get("mask_id"), -1))
        feat = feature_map.get(key, {})
        by_scene.setdefault(scene, []).append(feat)
    out: dict[str, dict[str, float]] = {}
    for scene, feats in by_scene.items():
        out[scene] = {
            "broad_risk_rate": _mean([1.0 if _bool(feat.get("broad_background_risk")) else 0.0 for feat in feats]),
            "observed_density_mean": _mean([_num(feat.get("observed_density_mean")) for feat in feats]),
            "support_count_mean": _mean([_num(feat.get("support_count")) for feat in feats]),
            "area_ratio_mean": _mean([_num(feat.get("area_ratio")) for feat in feats]),
            "feature_row_count": float(len(feats)),
        }
    return out


def _apply_risk_filter(
    base_rows: list[dict[str, str]],
    feature_map: dict[tuple[str, int, int], dict[str, Any]],
    mode: str,
) -> list[dict[str, str]]:
    keep = phase7e._keep_flags(base_rows, feature_map, mode)
    return [row for row, flag in zip(base_rows, keep) if flag]


def _scores(
    rows: list[dict[str, str]],
    feature_map: dict[tuple[str, int, int], dict[str, Any]],
    scene_stats: dict[str, dict[str, float]],
    score_mode: str,
) -> tuple[list[float], list[float], list[float]]:
    original_raw = [_num(row.get("object_score"), _num(row.get("frame_mask_score"), 1.0)) for row in rows]
    h9_raw = [phase7e._raw_h9(dict(row), feature_map) for row in rows]
    original_rank = _rank_by_scene([dict(row) for row in rows], original_raw)
    h9_rank = _rank_by_scene([dict(row) for row in rows], h9_raw)
    out: list[float] = []
    for row, orig, h9 in zip(rows, original_rank, h9_rank):
        scene = str(row.get("scene_id", ""))
        broad_rate = scene_stats.get(scene, {}).get("broad_risk_rate", 0.0)
        if score_mode == "h9_rank":
            score = h9
        elif score_mode == "original_rank":
            score = orig
        elif score_mode == "blend_o25_h75":
            score = 0.25 * orig + 0.75 * h9
        elif score_mode == "blend_o50_h50":
            score = 0.50 * orig + 0.50 * h9
        elif score_mode == "blend_o75_h25":
            score = 0.75 * orig + 0.25 * h9
        elif score_mode == "broad_scene_orig_ge060":
            score = orig if broad_rate >= 0.60 else h9
        elif score_mode == "broad_scene_blend50_ge060":
            score = (0.50 * orig + 0.50 * h9) if broad_rate >= 0.60 else h9
        elif score_mode == "broad_scene_orig_ge070":
            score = orig if broad_rate >= 0.70 else h9
        else:
            raise ValueError(score_mode)
        out.append(float(score))
    return out, original_rank, h9_rank


def _variant_rows(
    rows: list[dict[str, str]],
    feature_map: dict[tuple[str, int, int], dict[str, Any]],
    scene_stats: dict[str, dict[str, float]],
    risk_mode: str,
    score_mode: str,
    variant_id: str,
) -> list[dict[str, Any]]:
    score, original_rank, h9_rank = _scores(rows, feature_map, scene_stats, score_mode)
    out: list[dict[str, Any]] = []
    for row, final_score, orig, h9 in zip(rows, score, original_rank, h9_rank):
        obj = str(row.get("mv_object_id", ""))
        if obj.startswith(f"{BASE_VARIANT}:"):
            obj = obj.replace(f"{BASE_VARIANT}:", f"{variant_id}:", 1)
        scene = str(row.get("scene_id", ""))
        out.append(
            {
                **row,
                "variant": variant_id,
                "source_variant": variant_id,
                "mv_object_id": obj,
                "frame_mask_score": float(final_score),
                "object_score": float(final_score),
                "original_rank_score": float(orig),
                "h9_rank_score": float(h9),
                "score_mode_detail": score_mode,
                "risk_filter_mode": risk_mode,
                "scene_broad_risk_rate": float(scene_stats.get(scene, {}).get("broad_risk_rate", 0.0)),
                "base_extent_variant": BASE_VARIANT,
                "selection_reason": f"phase7f_scene_balanced_{score_mode}_{risk_mode}_from_{BASE_VARIANT}",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    control = _control_threshold()
    control_threshold = _num(control.get("mean_MV_AP_window"))
    reference_by_scene = _scene_reference_rows()
    feature_map = phase7d._load_feature_rows(BASE_VARIANT)
    base_rows = [row for row in _read_csv(PHASE4_ROOT / "mv_object_frame_mask_rows.csv") if row.get("variant") == BASE_VARIANT]
    b0_risk = _num(json.loads((PHASE4_ROOT / "summary.json").read_text(encoding="utf-8")).get("B0_risk_penalty_mean_proxy"), 1.0)
    risk_modes = ["drop_broad_low_h9_2", "drop_broad_low_h9_5"]
    score_modes = [
        "h9_rank",
        "original_rank",
        "blend_o25_h75",
        "blend_o50_h50",
        "blend_o75_h25",
        "broad_scene_orig_ge060",
        "broad_scene_blend50_ge060",
        "broad_scene_orig_ge070",
    ]
    config_rows: list[dict[str, Any]] = []
    frame_mask_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    scene_stat_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for risk_mode in risk_modes:
        kept_rows = _apply_risk_filter(base_rows, feature_map, risk_mode)
        stats = _scene_stats([dict(row) for row in kept_rows], feature_map)
        for scene, values in sorted(stats.items()):
            scene_stat_rows.append({"risk_filter_mode": risk_mode, "scene_id": scene, **values})
        risks = [phase7e._risk(row, feature_map) for row in kept_rows]
        risk_mean = _mean(risks)
        for score_mode in score_modes:
            variant_id = f"F_W9a_{risk_mode}_{score_mode}"
            rows = _variant_rows(kept_rows, feature_map, stats, risk_mode, score_mode, variant_id)
            frame_mask_rows.extend(rows)
            metrics, cases = phase7d._evaluate_variant(BASE_VARIANT, variant_id, rows)
            metric_rows.extend(metrics)
            case_rows.extend({**row, "variant_id": variant_id} for row in cases)
            config_rows.append(
                {
                    "variant_id": variant_id,
                    "parent_variant_id": BASE_VARIANT,
                    "risk_filter_mode": risk_mode,
                    "score_mode": score_mode,
                    "changed_parameters": f"risk_filter_mode={risk_mode}; score_mode={score_mode}",
                    "changed_module": "risk_safe_W9a_extent_plus_scene_balanced_object_score",
                    "reason_for_change": "scene0011 regression under H9 score; broad-risk scene fallback is GT-free and feature-derived",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                    "dev_only_or_holdout": "dev_only",
                    "expected_blocker": "SCENE_SPECIFIC_RANKING_BLOCKER+CONTROL_BIAS_BLOCKER",
                }
            )
            risk_rows.append(
                {
                    "variant_id": variant_id,
                    "risk_filter_mode": risk_mode,
                    "score_mode": score_mode,
                    "selected_rows": len(kept_rows),
                    "dropped_rows": len(base_rows) - len(kept_rows),
                    "risk_penalty_mean": risk_mean,
                    "B0_risk_penalty_mean_proxy": b0_risk,
                    "risk_safe_vs_B0_proxy": risk_mean <= b0_risk + 1e-12,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    aggregate_rows = phase7d._aggregate(metric_rows)
    by_variant = {row["variant_id"]: row for row in aggregate_rows}
    risk_by_variant = {row["variant_id"]: row for row in risk_rows}
    scene_metrics: dict[tuple[str, str], dict[str, Any]] = {
        (str(row.get("variant_id", "")), str(row.get("scene_id", ""))): row for row in metric_rows
    }
    ref_scene0011 = reference_by_scene.get("scene0011_00", {})
    ref_phase7e_scene0011 = ref_scene0011.get("phase7e_reference_MV_AP_window", 0.0)
    ref_w9a_scene0011 = ref_scene0011.get("W9a_original_MV_AP_window", 0.0)
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for config in config_rows:
        variant_id = str(config["variant_id"])
        metric = by_variant.get(variant_id, {})
        risk = risk_by_variant.get(variant_id, {})
        scene0011 = scene_metrics.get((variant_id, "scene0011_00"), {})
        scene0050 = scene_metrics.get((variant_id, "scene0050_00"), {})
        scene0011_mv_ap = _num(scene0011.get("MV_AP_window"), _num(scene0011.get("MV_AP")))
        scene0050_mv_ap = _num(scene0050.get("MV_AP_window"), _num(scene0050.get("MV_AP")))
        control_gap = _num(metric.get("mean_MV_AP_window")) - control_threshold
        scene0011_delta_vs_phase7e = scene0011_mv_ap - ref_phase7e_scene0011
        scene0011_delta_vs_w9a_original = scene0011_mv_ap - ref_w9a_scene0011
        risk_safe = bool(risk.get("risk_safe_vs_B0_proxy"))
        gate_pass = (
            risk_safe
            and control_gap > 0.005
            and scene0011_delta_vs_phase7e > 0.002
            and scene0011_delta_vs_w9a_original >= -0.0005
            and not _bool(metric.get("uses_gt_for_prediction"))
            and not _bool(metric.get("uses_future"))
        )
        gate = {
            **config,
            "actual_blocker": "NEEDS_HOLDOUT_IN_NEXT_VERSION" if gate_pass else "SCENE_SPECIFIC_RANKING_OR_CONTROL_BLOCKER",
            "MV_AP_window": metric.get("mean_MV_AP_window", ""),
            "MV_AP50_window": metric.get("mean_MV_AP50_window", ""),
            "MV_AP25_window": metric.get("mean_MV_AP25_window", ""),
            "best_control_gap": control_gap,
            "scene0011_MV_AP_window": scene0011_mv_ap,
            "scene0050_MV_AP_window": scene0050_mv_ap,
            "scene0011_delta_vs_phase7e_reference": scene0011_delta_vs_phase7e,
            "scene0011_delta_vs_W9a_original": scene0011_delta_vs_w9a_original,
            "risk_penalty_mean": risk.get("risk_penalty_mean", ""),
            "risk_safe_vs_B0_proxy": risk_safe,
            "selected_rows": risk.get("selected_rows", ""),
            "dropped_rows": risk.get("dropped_rows", ""),
            "control_threshold_variant": control.get("variant_id", ""),
            "control_threshold_MV_AP_window": control_threshold,
            "same_frame_collision_count": metric.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": metric.get("missing_mask_raster_count", ""),
            "gate_pass_scene_balanced_dev_repair": gate_pass,
        }
        gate_rows.append(gate)
        if not gate_pass:
            failure_rows.append(
                {
                    "variant_id": variant_id,
                    "failure_reason": "risk/control/scene0011_repair_gate_not_satisfied",
                    "MV_AP_window": metric.get("mean_MV_AP_window", ""),
                    "best_control_gap": control_gap,
                    "scene0011_MV_AP_window": scene0011_mv_ap,
                    "scene0011_delta_vs_phase7e_reference": scene0011_delta_vs_phase7e,
                    "scene0011_delta_vs_W9a_original": scene0011_delta_vs_w9a_original,
                    "risk_penalty_mean": risk.get("risk_penalty_mean", ""),
                    "risk_safe_vs_B0_proxy": risk_safe,
                }
            )
    passing = [row for row in gate_rows if row.get("gate_pass_scene_balanced_dev_repair")]
    best = max(passing or gate_rows, key=lambda row: _num(row.get("MV_AP_window")), default={})
    summary = {
        "phase": "v90_phase7f_dev_scene_balanced_score_repair",
        "schema": "stream4d_v90_phase7f_dev_scene_balanced_score_repair_v1",
        "repair_scope": "dev_only_scene0011_ranking_repair_for_phase7e_risk_safe_seed",
        "base_extent_variant": BASE_VARIANT,
        "reference_variant": REFERENCE_VARIANT,
        "B0_risk_penalty_mean_proxy": b0_risk,
        "control_threshold_variant": control.get("variant_id", ""),
        "control_threshold_metrics": control,
        "scene_reference_metrics": reference_by_scene,
        "best_variant_id": best.get("variant_id", ""),
        "best_variant_gate": best,
        "any_scene_balanced_dev_pass": bool(passing),
        "decision": "DEV_SCENE_BALANCED_SCORE_CANDIDATE_FOUND" if passing else "SCENE_SPECIFIC_RANKING_BLOCKER_REMAINS",
        "holdout_policy": "No v90 holdout rerun is allowed; this dev-only loop tests a GT-free score fallback for v91 pre-registration.",
        "row_counts": {
            "base_frame_mask_rows": len(base_rows),
            "variant_frame_mask_rows": len(frame_mask_rows),
            "metric_rows": len(metric_rows),
            "case_rows": len(case_rows),
            "scene_stat_rows": len(scene_stat_rows),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    outputs = {
        "variant_config_rows": OUT / "variant_config_rows.csv",
        "variant_frame_mask_rows": OUT / "variant_frame_mask_rows.csv",
        "variant_metric_rows": OUT / "variant_metric_rows.csv",
        "variant_metric_aggregate_rows": OUT / "variant_metric_aggregate_rows.csv",
        "variant_gate_rows": OUT / "variant_gate_rows.csv",
        "variant_failure_rows": OUT / "variant_failure_rows.csv",
        "variant_risk_rows": OUT / "variant_risk_rows.csv",
        "variant_scene_stat_rows": OUT / "variant_scene_stat_rows.csv",
        "variant_case_rows": OUT / "variant_case_rows.csv",
        "best_variant_summary": OUT / "best_variant_summary.json",
    }
    _write_csv(outputs["variant_config_rows"], config_rows)
    _write_csv(outputs["variant_frame_mask_rows"], frame_mask_rows)
    _write_csv(outputs["variant_metric_rows"], metric_rows)
    _write_csv(outputs["variant_metric_aggregate_rows"], aggregate_rows)
    _write_csv(outputs["variant_gate_rows"], gate_rows)
    _write_csv(outputs["variant_failure_rows"], failure_rows)
    _write_csv(outputs["variant_risk_rows"], risk_rows)
    _write_csv(outputs["variant_scene_stat_rows"], scene_stat_rows)
    _write_csv(outputs["variant_case_rows"], case_rows)
    _write_json(outputs["best_variant_summary"], summary)
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs.values() if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
