#!/usr/bin/env python3
"""Post-final v99 Phase10J pre-registered Phase2 support protocol holdout.

Phase10H/10I used stronger post-final support-score diagnostics. This script
steps back to the Phase2 pre-registered score policies and projects the
support-family variants to the same-scene temporal holdout with the repaired
same-schema support_surfel_count field. It does not introduce new weights.
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
from tools import build_v99_phase2_f2_strengthening_holdout as p2h  # noqa: E402
from tools import build_v99_phase10h_same_schema_support_quality_holdout as p10h  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10j_preregistered_support_protocol_holdout"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
EPS = 1e-4


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


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


def _norm(values: dict[str, float]) -> dict[str, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {key: 0.0 for key in values}
    lo = min(finite)
    hi = max(finite)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (float(value) - lo) / (hi - lo) for key, value in values.items()}


def _feature_table(joined_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    frames: dict[str, set[tuple[str, str]]] = defaultdict(set)
    support_sum: dict[str, float] = defaultdict(float)
    support_n: dict[str, int] = defaultdict(int)
    for row in joined_rows:
        oid = str(row["mv_object_id"])
        frames[oid].add((str(row["scene_id"]), str(row["frame_id"])))
        support_sum[oid] += _num(row.get("support_surfel_count"))
        support_n[oid] += 1

    frame_count = {oid: float(len(vals)) for oid, vals in frames.items()}
    max_frame_count = max(frame_count.values(), default=1.0)
    support_mean = {oid: support_sum[oid] / max(1, support_n[oid]) for oid in support_n}
    support_norm = _norm(support_mean)
    semantic_norm = p2h._semantic_consistency_by_object(joined_rows)

    out: dict[str, dict[str, float]] = {}
    for oid in sorted(frames):
        out[oid] = {
            "frame_count": frame_count[oid],
            "frame_count_norm": frame_count[oid] / max(1.0, max_frame_count),
            "support_surfel_count_mean": support_mean.get(oid, 0.0),
            "support_surfel_count_norm": support_norm.get(oid, 0.0),
            "semantic_norm": float(semantic_norm.get(oid, 0.0)),
        }
    return out


def _score(config: dict[str, Any], feature: dict[str, float]) -> float:
    mode = str(config["mode"])
    if mode == "frame":
        return feature["frame_count_norm"]
    if mode == "support_tiebreak":
        return feature["frame_count_norm"] + EPS * feature["support_surfel_count_norm"]
    if mode == "semantic_tiebreak":
        return feature["frame_count_norm"] + EPS * feature["semantic_norm"]
    if mode == "support_semantic_tiebreak":
        return feature["frame_count_norm"] + EPS * (
            0.5 * feature["support_surfel_count_norm"] + 0.5 * feature["semantic_norm"]
        )
    raise ValueError(f"unknown mode {mode}")


def _make_rows(
    parent_rows: list[dict[str, Any]],
    features: dict[str, dict[str, float]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    variant_id = f"{config['variant_id']}__holdout_same_schema_support_protocol"
    out: list[dict[str, Any]] = []
    for row in parent_rows:
        oid = str(row["mv_object_id"])
        feature = features[oid]
        new = dict(row)
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["score"] = float(_score(config, feature))
        new["score_policy"] = config["score_policy"]
        new["fixed_dev_variant_id"] = config["variant_id"]
        new["phase10j_parent_variant_id"] = row.get("variant_id", "")
        new["phase10j_mode"] = config["mode"]
        new["phase10j_frame_count"] = feature["frame_count"]
        new["phase10j_frame_count_norm"] = feature["frame_count_norm"]
        new["phase10j_support_surfel_count_mean"] = feature["support_surfel_count_mean"]
        new["phase10j_support_surfel_count_norm"] = feature["support_surfel_count_norm"]
        new["phase10j_semantic_norm"] = feature["semantic_norm"]
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _phase2_dev_rows() -> list[dict[str, Any]]:
    rows = _read_csv(PHASE2_DIR / "variant_metric_rows.csv")
    wanted = {
        "P2_B0_phase1_main_score_replay",
        "P2_D1_frame_count_plus_support_tiebreak",
        "P2_D2_frame_count_plus_semantic_tiebreak",
        "P2_D4_frame_count_plus_support_semantic_tiebreak",
    }
    return [row for row in rows if row.get("variant_id") in wanted]


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    scope = holdout._load_source_scope(p10h.HOLDOUT_SOURCE_ROWS)
    parent_rows = [dict(row) for row in _read_csv(PHASE2_DIR / "holdout_mv_object_frame_mask_rows.csv")]
    joined_rows, join_diag = p10h._join_support_surfel_count(parent_rows)
    features = _feature_table(joined_rows)

    configs = [
        {
            "variant_id": "P2_B0_phase1_main_score_replay",
            "mode": "frame",
            "score_policy": "phase10j_replay_preregistered_frame_count_holdout",
            "phase2_family": "baseline",
        },
        {
            "variant_id": "P2_D1_frame_count_plus_support_tiebreak",
            "mode": "support_tiebreak",
            "score_policy": "phase10j_preregistered_current_chunk_frame_count_plus_1e-4_surfel_support_tiebreak",
            "phase2_family": "F2-D_score_policy",
        },
        {
            "variant_id": "P2_D2_frame_count_plus_semantic_tiebreak",
            "mode": "semantic_tiebreak",
            "score_policy": "phase10j_replay_preregistered_current_chunk_frame_count_plus_1e-4_semantic_tiebreak",
            "phase2_family": "F2-D_score_policy",
        },
        {
            "variant_id": "P2_D4_frame_count_plus_support_semantic_tiebreak",
            "mode": "support_semantic_tiebreak",
            "score_policy": "phase10j_preregistered_current_chunk_frame_count_plus_1e-4_support_semantic_tiebreak",
            "phase2_family": "F2-A_F2-D_reliability_score",
        },
    ]
    all_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for config in configs:
        rows = _make_rows(joined_rows, features, config)
        variant_id = f"{config['variant_id']}__holdout_same_schema_support_protocol"
        metrics, cases, tops = holdout._evaluate_variant(variant_id, rows, scope)
        all_rows.extend(rows)
        metric_rows.extend(metrics)
        case_rows.extend(cases)
        top_rows.extend(tops)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10j_variant_config_v1",
                "phase_id": "v99_phase10j_preregistered_support_protocol_holdout",
                "variant_id": variant_id,
                "fixed_dev_variant_id": config["variant_id"],
                "phase2_family": config["phase2_family"],
                "mode": config["mode"],
                "score_policy": config["score_policy"],
                "support_source": p10h._rel(p10h.HOLDOUT_SUPPORT_ROWS),
                "support_schema": "support_surfel_count joined by object_id/scene/frame/mask",
                "new_weight_sweep": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )

    aggregate_rows = holdout._aggregate(
        metric_rows,
        family="v99_phase10j_preregistered_support_protocol_holdout",
    )
    f2_hold_window = float(phase0["F2_base_holdout_MV_AP_window"])
    f2_hold_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"])
    paired: list[dict[str, Any]] = []
    for row in aggregate_rows:
        variant_id = str(row["variant_id"])
        fixed = variant_id.replace("__holdout_same_schema_support_protocol", "")
        window = _num(row.get("mean_MV_AP_window"))
        ap50 = _num(row.get("mean_MV_AP50_window"))
        paired.append(
            {
                "schema_version": "stream4d_v99_phase10j_holdout_metric_v1",
                "phase_id": "v99_phase10j_preregistered_support_protocol_holdout",
                "fixed_dev_variant_id": fixed,
                "holdout_variant_id": variant_id,
                "holdout_MV_AP_window": row.get("mean_MV_AP_window"),
                "holdout_MV_AP50_window": row.get("mean_MV_AP50_window"),
                "holdout_MV_AP25_window": row.get("mean_MV_AP25_window"),
                "holdout_delta_vs_F2_base_window": window - f2_hold_window,
                "strict_local_holdout_gate_pass": window >= f2_hold_window + 0.005 and ap50 >= f2_hold_ap50 + 0.010,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "formal_claim_allowed": False,
            }
        )

    repair_rows = [row for row in paired if row["fixed_dev_variant_id"] != "P2_B0_phase1_main_score_replay"]
    best = max(repair_rows, key=lambda row: (_num(row["holdout_MV_AP_window"]), _num(row["holdout_MV_AP50_window"])))
    any_pass = any(bool(row["strict_local_holdout_gate_pass"]) for row in repair_rows)
    dev_rows = _phase2_dev_rows()
    gate_rows = [
        {
            "gate_id": "same_schema_support_join_complete",
            "pass": int(join_diag["missing_parent_row_count"]) == 0,
            "expected": "0 missing fixed holdout rows after support_surfel_count join",
            "observed": join_diag["missing_parent_row_count"],
            "severity": "required_data_contract",
        },
        {
            "gate_id": "preregistered_support_protocol_strict_holdout_gate",
            "pass": any_pass,
            "expected": f"MV_AP_window>={f2_hold_window + 0.005} and MV_AP50_window>={f2_hold_ap50 + 0.010}",
            "observed": f"best={best['fixed_dev_variant_id']} MV_AP_window={best['holdout_MV_AP_window']} MV_AP50_window={best['holdout_MV_AP50_window']}",
            "severity": "method_gate",
        },
        {
            "gate_id": "formal_claim_allowed_after_post_final_diagnostic",
            "pass": False,
            "expected": "fresh frozen holdout",
            "observed": "post-final diagnostic after prior holdout feedback",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "Pre-registered Phase2 support protocol does not pass holdout; stop support-score protocol on this fixed universe and move only with a new object-birth/candidate-generation plan.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    feature_rows = [
        {
            "schema_version": "stream4d_v99_phase10j_object_feature_v1",
            "phase_id": "v99_phase10j_preregistered_support_protocol_holdout",
            "mv_object_id": oid,
            **values,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for oid, values in sorted(features.items())
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10j_preregistered_support_protocol_holdout_summary_v1",
        "phase_id": "v99_phase10j_preregistered_support_protocol_holdout",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "NO_GO_PREREGISTERED_SUPPORT_PROTOCOL_HOLDOUT" if not any_pass else "DIAGNOSTIC_PREREGISTERED_SUPPORT_PROTOCOL_HOLDOUT_PASS_REQUIRES_FRESH_HOLDOUT",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "new_weight_sweep": False,
        "variant_count": len(configs),
        "object_count": len(features),
        "join_diag": join_diag,
        "phase2_dev_rows": dev_rows,
        "best_holdout_fixed_dev_variant_id": best["fixed_dev_variant_id"],
        "best_holdout_MV_AP_window": float(_num(best["holdout_MV_AP_window"])),
        "best_holdout_MV_AP50_window": float(_num(best["holdout_MV_AP50_window"])),
        "best_holdout_delta_vs_F2_base_window": float(_num(best["holdout_delta_vs_F2_base_window"])),
        "any_preregistered_support_variant_passes_strict_gate": any_pass,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "object_feature_rows": _rel(OUT_DIR / "object_feature_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
        },
    }

    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "paired_metric_rows.csv", paired)
    _write_csv(OUT_DIR / "holdout_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "holdout_case_rows.csv", case_rows)
    _write_csv(OUT_DIR / "holdout_top_iou_rows.csv", top_rows)
    _write_csv(OUT_DIR / "holdout_mv_object_frame_mask_rows.csv", all_rows)
    _write_csv(OUT_DIR / "object_feature_rows.csv", feature_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
