#!/usr/bin/env python3
"""Post-final v99 Phase10D GT-free score/area repair autopsy.

This is diagnostic only. It runs after the Phase10 holdout decision, so a
passing row here is not a formal method claim. The purpose is to test whether a
common non-GT score signal available on both full-dev and holdout can close the
ranking gap exposed by the Phase10C GT-score oracle upper bound.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v98_1_canonical_holdout_metrics as holdout  # noqa: E402
from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10d_score_area_repair_autopsy"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
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


def _norm_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    vals = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not vals:
        return {key: 0.0 for key in values}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (float(value) - lo) / (hi - lo) for key, value in values.items()}


def _mask_key(row: dict[str, Any]) -> int:
    return int(_num(row.get("selected_mask_id", row.get("mask_id", row.get("mask_id_or_generated_id", -1))), -1))


def _label_area_ratio(
    *,
    scene: str,
    frame: int,
    mask_id: int,
    mask_path_by_frame: dict[tuple[str, int], Path],
    cache: dict[tuple[str, int], np.ndarray],
) -> float:
    if mask_id <= 0:
        return 0.0
    key = (scene, int(frame))
    label = cache.get(key)
    if label is None:
        path = mask_path_by_frame.get(key)
        if path is None or not path.exists():
            return 0.0
        label = p1._read_label(path)
        cache[key] = label
    if label.size <= 0:
        return 0.0
    return float(np.count_nonzero(label == int(mask_id)) / float(label.size))


def _object_features(rows: list[dict[str, Any]], mask_path_by_frame: dict[tuple[str, int], Path]) -> dict[str, dict[str, float]]:
    cache: dict[tuple[str, int], np.ndarray] = {}
    by_obj: dict[str, dict[str, Any]] = defaultdict(lambda: {"scores": [], "areas": [], "frames": set()})
    for row in rows:
        oid = str(row.get("mv_object_id", ""))
        scene = str(row.get("scene_id", ""))
        frame = int(_num(row.get("frame_id"), -1))
        if not oid or not scene or frame < 0:
            continue
        area = _label_area_ratio(
            scene=scene,
            frame=frame,
            mask_id=_mask_key(row),
            mask_path_by_frame=mask_path_by_frame,
            cache=cache,
        )
        by_obj[oid]["scores"].append(_num(row.get("score")))
        by_obj[oid]["areas"].append(area)
        by_obj[oid]["frames"].add((scene, frame))

    parent_max = {oid: max(vals["scores"]) if vals["scores"] else 0.0 for oid, vals in by_obj.items()}
    area_mean = {oid: float(np.mean(vals["areas"])) if vals["areas"] else 0.0 for oid, vals in by_obj.items()}
    area_max = {oid: max(vals["areas"]) if vals["areas"] else 0.0 for oid, vals in by_obj.items()}
    area_std = {oid: float(np.std(vals["areas"])) if vals["areas"] else 0.0 for oid, vals in by_obj.items()}
    frame_count = {oid: float(len(vals["frames"])) for oid, vals in by_obj.items()}

    parent_norm = _norm_map(parent_max)
    mean_area_norm = _norm_map(area_mean)
    max_area_norm = _norm_map(area_max)
    area_std_norm = _norm_map(area_std)
    frame_count_norm = _norm_map(frame_count)
    out: dict[str, dict[str, float]] = {}
    for oid in by_obj:
        out[oid] = {
            "parent_max_score": parent_max[oid],
            "parent_score_norm": parent_norm.get(oid, 0.0),
            "area_mean_ratio": area_mean[oid],
            "area_mean_norm": mean_area_norm.get(oid, 0.0),
            "area_max_ratio": area_max[oid],
            "area_max_norm": max_area_norm.get(oid, 0.0),
            "area_std_ratio": area_std[oid],
            "area_std_norm": area_std_norm.get(oid, 0.0),
            "frame_count": frame_count[oid],
            "frame_count_norm": frame_count_norm.get(oid, 0.0),
        }
    return out


def _score_for(config: dict[str, Any], base: float, f: dict[str, float]) -> float:
    mode = str(config["mode"])
    eps = float(config.get("eps", 0.0))
    if mode == "parent":
        return base
    if mode == "object_parent_max":
        return f["parent_max_score"]
    if mode == "mean_area_boost":
        return f["parent_max_score"] + eps * f["area_mean_norm"]
    if mode == "mean_area_penalty":
        return f["parent_max_score"] - eps * f["area_mean_norm"]
    if mode == "max_area_penalty":
        return f["parent_max_score"] - eps * f["area_max_norm"]
    if mode == "area_std_penalty":
        return f["parent_max_score"] - eps * f["area_std_norm"]
    if mode == "blend_parent_mean_area":
        w = float(config["parent_weight"])
        return w * f["parent_score_norm"] + (1.0 - w) * f["area_mean_norm"]
    if mode == "blend_parent_inverse_mean_area":
        w = float(config["parent_weight"])
        return w * f["parent_score_norm"] + (1.0 - w) * (1.0 - f["area_mean_norm"])
    if mode == "blend_parent_inverse_max_area":
        w = float(config["parent_weight"])
        return w * f["parent_score_norm"] + (1.0 - w) * (1.0 - f["area_max_norm"])
    raise ValueError(f"unknown score mode: {mode}")


def _make_rows(
    parent_rows: list[dict[str, Any]],
    *,
    variant_id: str,
    split: str,
    config: dict[str, Any],
    features: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in parent_rows:
        oid = str(row.get("mv_object_id", ""))
        f = features.get(oid)
        if f is None:
            continue
        new = dict(row)
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["score"] = float(_score_for(config, _num(row.get("score")), f))
        new["score_policy"] = str(config["score_policy"])
        new["phase10d_parent_variant_id"] = row.get("variant_id", "")
        new["phase10d_split"] = split
        new["phase10d_mode"] = config["mode"]
        new["phase10d_parent_weight"] = config.get("parent_weight", "")
        new["phase10d_eps"] = config.get("eps", "")
        new["phase10d_area_mean_ratio"] = f["area_mean_ratio"]
        new["phase10d_area_max_ratio"] = f["area_max_ratio"]
        new["phase10d_area_std_ratio"] = f["area_std_ratio"]
        new["phase10d_frame_count"] = f["frame_count"]
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _phase2_dev_parent_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads((PHASE2_DIR / "best_variant_summary.json").read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"no Phase2 dev parent rows for {variant}")
    return variant, rows


def _paired_rows(
    *,
    configs: list[dict[str, Any]],
    dev_agg: list[dict[str, Any]],
    holdout_agg: list[dict[str, Any]],
    f2_dev_window: float,
    f2_dev_ap50: float,
    f2_hold_window: float,
    f2_hold_ap50: float,
) -> list[dict[str, Any]]:
    dev_by_name = {str(row["variant_id"]).replace("V99P10D_dev_", ""): row for row in dev_agg}
    hold_by_name = {str(row["variant_id"]).replace("V99P10D_holdout_", ""): row for row in holdout_agg}
    rows: list[dict[str, Any]] = []
    for config in configs:
        name = str(config["name"])
        d = dev_by_name[name]
        h = hold_by_name[name]
        dev_gate = _num(d.get("MV_AP_window")) >= f2_dev_window + 0.005 and _num(d.get("MV_AP50_window")) >= f2_dev_ap50 + 0.010
        hold_gate = _num(h.get("mean_MV_AP_window")) >= f2_hold_window + 0.005 and _num(h.get("mean_MV_AP50_window")) >= f2_hold_ap50 + 0.010
        rows.append(
            {
                "schema_version": "stream4d_v99_phase10d_paired_metric_v1",
                "phase_id": "v99_phase10d_score_area_repair_autopsy",
                "name": name,
                "mode": config["mode"],
                "score_policy": config["score_policy"],
                "dev_variant_id": d["variant_id"],
                "holdout_variant_id": h["variant_id"],
                "dev_MV_AP_window": d.get("MV_AP_window"),
                "dev_MV_AP50_window": d.get("MV_AP50_window"),
                "dev_MV_AP_scene": d.get("MV_AP_scene"),
                "holdout_MV_AP_window": h.get("mean_MV_AP_window"),
                "holdout_MV_AP50_window": h.get("mean_MV_AP50_window"),
                "holdout_MV_AP25_window": h.get("mean_MV_AP25_window"),
                "dev_delta_vs_F2_base_window": _num(d.get("MV_AP_window")) - f2_dev_window,
                "holdout_delta_vs_F2_base_window": _num(h.get("mean_MV_AP_window")) - f2_hold_window,
                "dev_gate_pass": dev_gate,
                "holdout_gate_pass": hold_gate,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
                "formal_claim_blocker": "post_final_holdout_feedback_sweep_requires_fresh_holdout_even_if_any_row_passes",
            }
        )
    return rows


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    dev_scope = p1._load_source_scope()
    hold_scope = holdout._load_source_scope(HOLDOUT_SOURCE_ROWS)
    dev_parent_variant, dev_parent_rows = _phase2_dev_parent_rows()
    hold_parent_rows = [dict(row) for row in _read_csv(HOLDOUT_FIXED_ROWS)]
    if not hold_parent_rows:
        raise RuntimeError(f"missing holdout parent rows: {HOLDOUT_FIXED_ROWS}")

    dev_features = _object_features(dev_parent_rows, dev_scope["mask_path_by_frame"])
    hold_features = _object_features(hold_parent_rows, hold_scope["mask_path_by_frame"])
    configs = [
        {"name": "A0_parent", "mode": "parent", "score_policy": "phase10d_parent_score_replay"},
        {"name": "A1_object_parent_max", "mode": "object_parent_max", "score_policy": "phase10d_object_parent_max_score"},
        {"name": "A2_mean_area_boost_eps1e-4", "mode": "mean_area_boost", "eps": 1e-4, "score_policy": "phase10d_parent_plus_1e-4_mean_mask_area"},
        {"name": "A3_mean_area_penalty_eps1e-4", "mode": "mean_area_penalty", "eps": 1e-4, "score_policy": "phase10d_parent_minus_1e-4_mean_mask_area"},
        {"name": "A4_mean_area_penalty_eps1e-3", "mode": "mean_area_penalty", "eps": 1e-3, "score_policy": "phase10d_parent_minus_1e-3_mean_mask_area"},
        {"name": "A5_max_area_penalty_eps1e-3", "mode": "max_area_penalty", "eps": 1e-3, "score_policy": "phase10d_parent_minus_1e-3_max_mask_area"},
        {"name": "A6_area_std_penalty_eps1e-3", "mode": "area_std_penalty", "eps": 1e-3, "score_policy": "phase10d_parent_minus_1e-3_mask_area_std"},
        {"name": "A7_blend_parent_area_90_10", "mode": "blend_parent_mean_area", "parent_weight": 0.90, "score_policy": "phase10d_0p90_parent_0p10_mean_mask_area"},
        {"name": "A8_blend_parent_inverse_area_90_10", "mode": "blend_parent_inverse_mean_area", "parent_weight": 0.90, "score_policy": "phase10d_0p90_parent_0p10_inverse_mean_mask_area"},
        {"name": "A9_blend_parent_inverse_max_area_90_10", "mode": "blend_parent_inverse_max_area", "parent_weight": 0.90, "score_policy": "phase10d_0p90_parent_0p10_inverse_max_mask_area"},
    ]

    config_rows: list[dict[str, Any]] = []
    all_dev_rows: list[dict[str, Any]] = []
    all_hold_rows: list[dict[str, Any]] = []
    dev_scene_rows: list[dict[str, Any]] = []
    dev_frame_rows: list[dict[str, Any]] = []
    hold_metric_rows: list[dict[str, Any]] = []
    hold_case_rows: list[dict[str, Any]] = []
    hold_top_rows: list[dict[str, Any]] = []
    for config in configs:
        name = str(config["name"])
        dev_variant = f"V99P10D_dev_{name}"
        hold_variant = f"V99P10D_holdout_{name}"
        dev_rows = _make_rows(dev_parent_rows, variant_id=dev_variant, split="full_dev", config=config, features=dev_features)
        hold_rows = _make_rows(hold_parent_rows, variant_id=hold_variant, split="same_scene_temporal_holdout", config=config, features=hold_features)
        d_metrics, d_frames = p1._evaluate_variant(dev_variant, dev_rows, dev_scope)
        h_metrics, h_cases, h_tops = holdout._evaluate_variant(hold_variant, hold_rows, hold_scope)
        all_dev_rows.extend(dev_rows)
        all_hold_rows.extend(hold_rows)
        dev_scene_rows.extend(d_metrics)
        dev_frame_rows.extend(d_frames)
        hold_metric_rows.extend(h_metrics)
        hold_case_rows.extend(h_cases)
        hold_top_rows.extend(h_tops)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10d_variant_config_v1",
                "phase_id": "v99_phase10d_score_area_repair_autopsy",
                "family": "post_final_gt_free_common_score_area_repair",
                "name": name,
                "mode": config["mode"],
                "eps": config.get("eps", ""),
                "parent_weight": config.get("parent_weight", ""),
                "dev_variant_id": dev_variant,
                "holdout_variant_id": hold_variant,
                "score_policy": config["score_policy"],
                "dev_parent_variant_id": dev_parent_variant,
                "holdout_parent_rows": _rel(HOLDOUT_FIXED_ROWS),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )

    dev_agg = p1._aggregate_metrics(dev_scene_rows)
    hold_agg = holdout._aggregate(hold_metric_rows, family="v99_phase10d_post_final_score_area_repair")
    f2_dev_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    f2_dev_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"])
    f2_hold_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_hold_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    paired = _paired_rows(
        configs=configs,
        dev_agg=dev_agg,
        holdout_agg=hold_agg,
        f2_dev_window=f2_dev_window,
        f2_dev_ap50=f2_dev_ap50,
        f2_hold_window=f2_hold_window,
        f2_hold_ap50=f2_hold_ap50,
    )
    best_dev = max(paired, key=lambda row: (_num(row["dev_MV_AP_window"]), _num(row["dev_MV_AP50_window"])))
    best_hold = max(paired, key=lambda row: (_num(row["holdout_MV_AP_window"]), _num(row["holdout_MV_AP50_window"])))
    best_dev_hold = next(row for row in paired if row["name"] == best_dev["name"])
    any_both = any(bool(row["dev_gate_pass"]) and bool(row["holdout_gate_pass"]) for row in paired)
    gate_rows = [
        {
            "gate_id": "post_final_score_area_any_variant_passes_dev_and_holdout_strict_gates",
            "pass": any_both,
            "expected": "some common non-GT score/area repair passes both dev and holdout strict gates",
            "observed": f"best_dev={best_dev['name']} dev={best_dev['dev_MV_AP_window']}; best_holdout={best_hold['name']} holdout={best_hold['holdout_MV_AP_window']}",
            "severity": "diagnostic",
        },
        {
            "gate_id": "formal_claim_allowed_after_holdout_feedback",
            "pass": False,
            "expected": "fresh frozen holdout required after post-final repair sweep",
            "observed": "this script is explicitly post-final diagnostic",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If a row passes here, freeze from dev-only evidence and require a fresh holdout; otherwise common score/area repair is insufficient and ranking needs a stronger non-GT quality estimator.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10d_score_area_repair_autopsy_summary_v1",
        "phase_id": "v99_phase10d_score_area_repair_autopsy",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "DIAGNOSTIC_CANDIDATE_REQUIRES_FRESH_HOLDOUT" if any_both else "NO_GO_POST_FINAL_SCORE_AREA_REPAIR_NO_FORMAL_CLAIM",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "variant_count": len(configs),
        "dev_object_count": len(dev_features),
        "holdout_object_count": len(hold_features),
        "best_dev_name": best_dev["name"],
        "best_dev_MV_AP_window": float(_num(best_dev["dev_MV_AP_window"])),
        "best_dev_MV_AP50_window": float(_num(best_dev["dev_MV_AP50_window"])),
        "best_dev_holdout_MV_AP_window": float(_num(best_dev_hold["holdout_MV_AP_window"])),
        "best_dev_holdout_MV_AP50_window": float(_num(best_dev_hold["holdout_MV_AP50_window"])),
        "best_holdout_name": best_hold["name"],
        "best_holdout_MV_AP_window": float(_num(best_hold["holdout_MV_AP_window"])),
        "best_holdout_MV_AP50_window": float(_num(best_hold["holdout_MV_AP50_window"])),
        "best_holdout_delta_vs_F2_base_window": float(_num(best_hold["holdout_delta_vs_F2_base_window"])),
        "any_variant_passes_dev_and_holdout_strict_gates": any_both,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "dev_metric_rows": _rel(OUT_DIR / "dev_metric_rows.csv"),
            "dev_metric_scene_rows": _rel(OUT_DIR / "dev_metric_scene_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "paired_metric_rows.csv", paired)
    _write_csv(OUT_DIR / "dev_metric_rows.csv", dev_agg)
    _write_csv(OUT_DIR / "dev_metric_scene_rows.csv", dev_scene_rows)
    _write_csv(OUT_DIR / "dev_frame_rows.csv", dev_frame_rows)
    _write_csv(OUT_DIR / "dev_mv_object_frame_mask_rows.csv", all_dev_rows)
    _write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", hold_agg)
    _write_csv(OUT_DIR / "holdout_metric_rows.csv", hold_metric_rows)
    _write_csv(OUT_DIR / "holdout_case_rows.csv", hold_case_rows)
    _write_csv(OUT_DIR / "holdout_top_iou_rows.csv", hold_top_rows)
    _write_csv(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv", all_hold_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if any_both else 2


if __name__ == "__main__":
    raise SystemExit(main())
