#!/usr/bin/env python3
"""Post-final v99 Phase10B holdout autopsy and DA3 role repair sweep.

This script is diagnostic. It is run after Phase10 used holdout feedback, so it
does not turn a passing row into a formal method claim. A passing row here would
only identify a candidate that needs a fresh frozen holdout.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v98_1_canonical_holdout_metrics as holdout  # noqa: E402
from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10_holdout_final_decision as phase10  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10b_holdout_repair_autopsy"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE4_DIR = AUDIT_ROOT / "v99_phase4_f2_da3_link_verifier"
PHASE10_DIR = AUDIT_ROOT / "v99_phase10_holdout_final_decision"
HOLDOUT_SOURCE_ROWS = AUDIT_ROOT / "v98_phase13_holdout/source_container_rows.csv"
HOLDOUT_FIXED_ROWS = PHASE2_DIR / "holdout_mv_object_frame_mask_rows.csv"


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _norm_metric(values: dict[str, float]) -> dict[str, float]:
    vals = list(values.values())
    if not vals:
        return {}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def _metric_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _phase2_dev_parent_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads((PHASE2_DIR / "best_variant_summary.json").read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"no Phase2 dev parent rows for {variant}")
    return variant, rows


def _dev_obj_metric() -> dict[str, dict[str, float]]:
    rows = [row for row in _read_csv(PHASE4_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == "P4_B1_DA3_boost_only"]
    metric: dict[str, dict[str, float]] = {}
    for row in rows:
        oid = str(row["mv_object_id"])
        if oid in metric:
            continue
        metric[oid] = {
            "mean_geometry_consistency": _num(row.get("da3_geometry_consistency")),
            "confident_link_count": _num(row.get("da3_confident_link_count")),
            "conflict_count": _num(row.get("da3_conflict_count")),
        }
    return metric


def _score_fn(
    mode: str,
    eps: float,
    geom_norm: dict[str, float],
    conflict_norm: dict[str, float],
    geom_raw: dict[str, float],
    tau: float,
) -> Callable[[float, str], float]:
    def fn(score: float, oid: str) -> float:
        g = geom_norm.get(oid, 0.0)
        c = conflict_norm.get(oid, 0.0)
        raw = geom_raw.get(oid, 0.0)
        if mode == "parent":
            return score
        if mode == "boost":
            return score + eps * g
        if mode == "veto":
            return score - eps * c
        if mode == "boost_veto":
            return score + eps * g - eps * c
        if mode == "centered":
            return score + eps * (2.0 * g - 1.0)
        if mode == "confidence_gated_boost":
            return score + (eps * g if raw >= tau else 0.0)
        raise ValueError(f"unknown mode {mode}")

    return fn


def _make_rows(
    parent_rows: list[dict[str, Any]],
    *,
    variant_id: str,
    score_policy: str,
    obj_metric: dict[str, dict[str, float]],
    mode: str,
    eps: float,
    tau: float,
    split: str,
) -> list[dict[str, Any]]:
    geom_raw = {oid: vals.get("mean_geometry_consistency", 0.0) for oid, vals in obj_metric.items()}
    geom_norm = _norm_metric(geom_raw)
    conflict_norm = _norm_metric({oid: vals.get("conflict_count", 0.0) for oid, vals in obj_metric.items()})
    fn = _score_fn(mode, eps, geom_norm, conflict_norm, geom_raw, tau)
    out: list[dict[str, Any]] = []
    for row in parent_rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["score"] = float(fn(_num(row.get("score")), oid))
        new["score_policy"] = score_policy
        new["phase10b_parent_variant_id"] = row.get("variant_id")
        new["phase10b_split"] = split
        new["phase10b_mode"] = mode
        new["phase10b_eps"] = eps
        new["phase10b_tau"] = tau if mode == "confidence_gated_boost" else ""
        new["da3_geometry_consistency"] = geom_raw.get(oid, 0.0)
        new["da3_geometry_consistency_norm"] = geom_norm.get(oid, 0.0)
        new["da3_conflict_count"] = obj_metric.get(oid, {}).get("conflict_count", 0.0)
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _score_delta_rows(parent_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]], *, split: str, variant_id: str) -> list[dict[str, Any]]:
    parent_score: dict[tuple[str, str, str, str], float] = {}
    for row in parent_rows:
        key = (
            str(row.get("scene_id")),
            str(row.get("frame_id")),
            str(row.get("mask_id", row.get("selected_mask_id"))),
            str(row.get("mv_object_id")),
        )
        parent_score[key] = _num(row.get("score"))
    deltas: list[float] = []
    geos: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in variant_rows:
        key = (
            str(row.get("scene_id")),
            str(row.get("frame_id")),
            str(row.get("mask_id", row.get("selected_mask_id"))),
            str(row.get("mv_object_id")),
        )
        if key not in parent_score:
            continue
        delta = _num(row.get("score")) - parent_score[key]
        deltas.append(delta)
        geos.append(_num(row.get("da3_geometry_consistency")))
    stats = _metric_percentiles(deltas)
    geo_stats = _metric_percentiles(geos)
    row = {
        "schema_version": "stream4d_v99_phase10b_score_delta_diag_v1",
        "phase_id": "v99_phase10b_holdout_repair_autopsy",
        "split": split,
        "variant_id": variant_id,
        "row_count": len(deltas),
    }
    for key, value in stats.items():
        row[f"score_delta_{key}"] = value
    for key, value in geo_stats.items():
        row[f"da3_geometry_consistency_{key}"] = value
    rows.append(row)
    return rows


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    source_scope = holdout._load_source_scope(HOLDOUT_SOURCE_ROWS)
    scope = p1._load_source_scope()
    dev_parent_variant, dev_parent_rows = _phase2_dev_parent_rows()
    holdout_parent_rows = [dict(row) for row in _read_csv(HOLDOUT_FIXED_ROWS)]
    if not holdout_parent_rows:
        raise RuntimeError(f"missing holdout parent rows: {HOLDOUT_FIXED_ROWS}")

    dev_obj = _dev_obj_metric()
    holdout_link_rows, holdout_obj = phase10._build_holdout_link_metrics(holdout_parent_rows, source_scope)

    configs = [
        ("T0_parent_no_aux", "parent", 0.0, 0.0),
        ("T1_boost_eps1e-4", "boost", 1e-4, 0.0),
        ("T2_veto_eps1e-4", "veto", 1e-4, 0.0),
        ("T3_boost_veto_eps1e-4", "boost_veto", 1e-4, 0.0),
        ("T4_boost_eps3e-5", "boost", 3e-5, 0.0),
        ("T5_boost_eps3e-4", "boost", 3e-4, 0.0),
        ("T6_boost_eps1e-3", "boost", 1e-3, 0.0),
        ("T7_centered_eps1e-4", "centered", 1e-4, 0.0),
        ("T8_gated_boost_eps1e-4_tau0p95", "confidence_gated_boost", 1e-4, 0.95),
    ]

    config_rows: list[dict[str, Any]] = []
    all_dev_rows: list[dict[str, Any]] = []
    all_holdout_rows: list[dict[str, Any]] = []
    dev_scene_rows: list[dict[str, Any]] = []
    dev_frame_rows: list[dict[str, Any]] = []
    holdout_metric_rows: list[dict[str, Any]] = []
    holdout_case_rows: list[dict[str, Any]] = []
    holdout_top_rows: list[dict[str, Any]] = []
    score_diag_rows: list[dict[str, Any]] = []

    for name, mode, eps, tau in configs:
        dev_variant = f"V99P10B_dev_{name}"
        holdout_variant = f"V99P10B_holdout_{name}"
        policy = f"phase10b_da3_{mode}_eps{eps:g}" + (f"_tau{tau:g}" if mode == "confidence_gated_boost" else "")
        dev_rows = _make_rows(
            dev_parent_rows,
            variant_id=dev_variant,
            score_policy=policy,
            obj_metric=dev_obj,
            mode=mode,
            eps=eps,
            tau=tau,
            split="full_dev",
        )
        holdout_rows = _make_rows(
            holdout_parent_rows,
            variant_id=holdout_variant,
            score_policy=policy,
            obj_metric=holdout_obj,
            mode=mode,
            eps=eps,
            tau=tau,
            split="same_scene_temporal_holdout",
        )
        dev_metrics, dev_frames = p1._evaluate_variant(dev_variant, dev_rows, scope)
        hold_metrics, hold_cases, hold_tops = holdout._evaluate_variant(holdout_variant, holdout_rows, source_scope)
        all_dev_rows.extend(dev_rows)
        all_holdout_rows.extend(holdout_rows)
        dev_scene_rows.extend(dev_metrics)
        dev_frame_rows.extend(dev_frames)
        holdout_metric_rows.extend(hold_metrics)
        holdout_case_rows.extend(hold_cases)
        holdout_top_rows.extend(hold_tops)
        score_diag_rows.extend(_score_delta_rows(dev_parent_rows, dev_rows, split="full_dev", variant_id=dev_variant))
        score_diag_rows.extend(_score_delta_rows(holdout_parent_rows, holdout_rows, split="same_scene_temporal_holdout", variant_id=holdout_variant))
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10b_variant_config_v1",
                "phase_id": "v99_phase10b_holdout_repair_autopsy",
                "family": "post_final_diagnostic_not_formal_claim",
                "name": name,
                "mode": mode,
                "eps": eps,
                "tau": tau if mode == "confidence_gated_boost" else "",
                "dev_variant_id": dev_variant,
                "holdout_variant_id": holdout_variant,
                "score_policy": policy,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    dev_agg = p1._aggregate_metrics(dev_scene_rows)
    holdout_agg = holdout._aggregate(holdout_metric_rows, family="v99_phase10b_post_final_da3_repair")
    holdout_by_name = {row["variant_id"].replace("V99P10B_holdout_", ""): row for row in holdout_agg}
    dev_by_name = {row["variant_id"].replace("V99P10B_dev_", ""): row for row in dev_agg}

    f2_dev_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    f2_dev_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"])
    f2_hold_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_hold_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    paired_rows: list[dict[str, Any]] = []
    for name, _mode, _eps, _tau in configs:
        d = dev_by_name[f"{name}"]
        h = holdout_by_name[f"{name}"]
        dev_gate = _num(d.get("MV_AP_window")) >= f2_dev_window + 0.005 and _num(d.get("MV_AP50_window")) >= f2_dev_ap50 + 0.010
        hold_gate = _num(h.get("mean_MV_AP_window")) >= f2_hold_window + 0.005 and _num(h.get("mean_MV_AP50_window")) >= f2_hold_ap50 + 0.010
        paired_rows.append(
            {
                "schema_version": "stream4d_v99_phase10b_paired_metric_v1",
                "phase_id": "v99_phase10b_holdout_repair_autopsy",
                "name": name,
                "dev_variant_id": d["variant_id"],
                "holdout_variant_id": h["variant_id"],
                "dev_MV_AP_window": d.get("MV_AP_window"),
                "dev_MV_AP50_window": d.get("MV_AP50_window"),
                "dev_MV_AP_scene": d.get("MV_AP_scene"),
                "holdout_MV_AP_window": h.get("mean_MV_AP_window"),
                "holdout_MV_AP50_window": h.get("mean_MV_AP50_window"),
                "dev_delta_vs_F2_base_window": _num(d.get("MV_AP_window")) - f2_dev_window,
                "holdout_delta_vs_F2_base_window": _num(h.get("mean_MV_AP_window")) - f2_hold_window,
                "dev_gate_pass": dev_gate,
                "holdout_gate_pass": hold_gate,
                "formal_claim_allowed": False,
                "formal_claim_blocker": "post_final_holdout_feedback_sweep_requires_fresh_holdout_even_if_any_row_passes",
            }
        )

    best_holdout = max(paired_rows, key=lambda row: (_num(row["holdout_MV_AP_window"]), _num(row["holdout_MV_AP50_window"])))
    best_dev = max(paired_rows, key=lambda row: (_num(row["dev_MV_AP_window"]), _num(row["dev_MV_AP50_window"])))
    any_both = any(bool(row["dev_gate_pass"]) and bool(row["holdout_gate_pass"]) for row in paired_rows)
    gate_rows = [
        {
            "gate_id": "post_final_sweep_any_variant_passes_dev_and_holdout_strict_gates",
            "pass": any_both,
            "expected": "some pre-registered repair row passes dev local gate and holdout strict gate",
            "observed": f"best_dev={best_dev['name']} dev={best_dev['dev_MV_AP_window']}; best_holdout={best_holdout['name']} holdout={best_holdout['holdout_MV_AP_window']}",
            "severity": "diagnostic",
        },
        {
            "gate_id": "formal_claim_allowed_after_holdout_feedback",
            "pass": False,
            "expected": "fresh frozen holdout required after post-final repair sweep",
            "observed": "this script is explicitly post-final diagnostic and uses holdout feedback",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If a diagnostic row looks promising, freeze it from dev-only evidence and require a fresh holdout; otherwise keep F2_base and treat DA3 score role as non-generalizing.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10b_holdout_repair_autopsy_summary_v1",
        "phase_id": "v99_phase10b_holdout_repair_autopsy",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "NO_GO_POST_FINAL_REPAIR_NO_FORMAL_CLAIM" if not any_both else "DIAGNOSTIC_CANDIDATE_REQUIRES_FRESH_HOLDOUT",
        "formal_claim_allowed": False,
        "variant_count": len(configs),
        "best_dev_name": best_dev["name"],
        "best_dev_MV_AP_window": float(_num(best_dev["dev_MV_AP_window"])),
        "best_dev_MV_AP50_window": float(_num(best_dev["dev_MV_AP50_window"])),
        "best_holdout_name": best_holdout["name"],
        "best_holdout_MV_AP_window": float(_num(best_holdout["holdout_MV_AP_window"])),
        "best_holdout_MV_AP50_window": float(_num(best_holdout["holdout_MV_AP50_window"])),
        "best_holdout_delta_vs_F2_base_window": float(_num(best_holdout["holdout_delta_vs_F2_base_window"])),
        "any_variant_passes_dev_and_holdout_strict_gates": any_both,
        "holdout_da3_link_count": len(holdout_link_rows),
        "holdout_da3_confident_link_count": sum(1 for row in holdout_link_rows if bool(row.get("da3_confident"))),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "dev_metric_rows": _rel(OUT_DIR / "dev_metric_rows.csv"),
            "dev_metric_scene_rows": _rel(OUT_DIR / "dev_metric_scene_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "score_delta_diagnostic_rows": _rel(OUT_DIR / "score_delta_diagnostic_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "paired_metric_rows.csv", paired_rows)
    _write_csv(OUT_DIR / "dev_metric_rows.csv", dev_agg)
    _write_csv(OUT_DIR / "dev_metric_scene_rows.csv", dev_scene_rows)
    _write_csv(OUT_DIR / "dev_frame_rows.csv", dev_frame_rows)
    _write_csv(OUT_DIR / "dev_mv_object_frame_mask_rows.csv", all_dev_rows)
    _write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", holdout_agg)
    _write_csv(OUT_DIR / "holdout_metric_rows.csv", holdout_metric_rows)
    _write_csv(OUT_DIR / "holdout_case_rows.csv", holdout_case_rows)
    _write_csv(OUT_DIR / "holdout_top_iou_rows.csv", holdout_top_rows)
    _write_csv(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv", all_holdout_rows)
    _write_csv(OUT_DIR / "holdout_da3_link_rows.csv", holdout_link_rows)
    _write_csv(OUT_DIR / "score_delta_diagnostic_rows.csv", score_diag_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if any_both else 2


if __name__ == "__main__":
    raise SystemExit(main())
