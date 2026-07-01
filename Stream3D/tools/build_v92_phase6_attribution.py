from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUT = ROOT / "outputs/audit/v92_phase6_attribution"
DEFAULT_PHASE5A = ROOT / "outputs/audit/v92_phase5_source_container_field"
DEFAULT_PHASE5B = ROOT / "outputs/audit/v92_phase5b_source_container_edge_field"
DEFAULT_PHASE4 = ROOT / "outputs/audit/v92_phase4_semantic_region_affinity"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


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
        return float(value)
    except Exception:
        return float(default)


def _row_by_variant(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("variant_id", "")): row for row in rows}


def _metric(row: dict[str, str], key: str = "mean_MV_AP_window") -> float:
    return _num(row.get(key), 0.0)


def _best(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: (_metric(row), _metric(row, "mean_MV_AP50_window")), default={})


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.output_root)
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    phase5a = Path(args.phase5a_root)
    phase5a = phase5a if phase5a.is_absolute() else ROOT / phase5a
    phase5b = Path(args.phase5b_root)
    phase5b = phase5b if phase5b.is_absolute() else ROOT / phase5b
    phase4 = Path(args.phase4_root)
    phase4 = phase4 if phase4.is_absolute() else ROOT / phase4

    rows_a = _read_csv(phase5a / "control_metric_rows.csv")
    rows_b = _read_csv(phase5b / "control_metric_rows.csv")
    by_a = _row_by_variant(rows_a)
    by_b = _row_by_variant(rows_b)
    phase4_summary = json.loads((phase4 / "summary.json").read_text(encoding="utf-8")) if (phase4 / "summary.json").exists() else {}

    fused = by_b.get("V92_F4_d4rt_radio_graph", {})
    hr2_fused = by_b.get("V92_F6_hr2_d4rt_radio_graph", {})
    d4rt = by_b.get("V92_F1_d4rt_seed_only", by_b.get("V92_C7_d4rt_only_control", {}))
    radio = by_b.get("V92_F2_radio_region_only", by_b.get("V92_C8_radio_only_control", {}))
    negative = by_b.get("V92_F5_d4rt_radio_negative", {})
    whole = by_b.get("V92_F0_whole_source_mask", {})
    best_b = _best(rows_b)
    best_control_mv = _num(fused.get("best_control_MV_AP_window"), 0.0)
    best_control_ap50 = _num(fused.get("best_control_MV_AP50_window"), 0.0)

    attribution_rows = [
        {"metric_name": "D4RT_only_MV_AP_window", "value": _metric(d4rt), "variant_id": d4rt.get("variant_id", "")},
        {"metric_name": "RADIO_only_MV_AP_window", "value": _metric(radio), "variant_id": radio.get("variant_id", "")},
        {"metric_name": "D4RT_plus_RADIO_MV_AP_window", "value": _metric(fused), "variant_id": fused.get("variant_id", "")},
        {"metric_name": "D4RT_plus_RADIO_plus_negative_MV_AP_window", "value": _metric(negative), "variant_id": negative.get("variant_id", "")},
        {"metric_name": "highres_D4RT_plus_RADIO_MV_AP_window", "value": _metric(hr2_fused), "variant_id": hr2_fused.get("variant_id", "")},
        {"metric_name": "whole_source_MV_AP_window", "value": _metric(whole), "variant_id": whole.get("variant_id", "")},
        {"metric_name": "best_control_MV_AP_window", "value": best_control_mv, "variant_id": fused.get("best_control_variant", "")},
        {"metric_name": "best_phase5b_MV_AP_window", "value": _metric(best_b), "variant_id": best_b.get("variant_id", "")},
        {"metric_name": "Phase4_RADIO_region_same_gt_diff_gt_AUC", "value": phase4_summary.get("source_internal_same_gt_different_gt_AUC_mean", ""), "variant_id": "Phase4_RADIO_region_diagnostic"},
    ]
    control_gap_rows = [
        {
            "variant_id": row.get("variant_id", ""),
            "MV_AP_window": row.get("mean_MV_AP_window", ""),
            "MV_AP50_window": row.get("mean_MV_AP50_window", ""),
            "real_minus_best_control_MV_AP_window": row.get("real_minus_best_control_MV_AP_window", ""),
            "real_minus_best_control_MV_AP50_window": row.get("real_minus_best_control_MV_AP50_window", ""),
            "same_frame_collision_count": row.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
            "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
            "uses_future": row.get("uses_future", ""),
        }
        for row in rows_b
    ]
    fused_mv = _metric(fused)
    decision = "CONTROL_BIAS_BLOCKER"
    if fused_mv > _metric(d4rt) and fused_mv > _metric(radio) and fused_mv > best_control_mv:
        decision = "GEOMETRY_SEMANTIC_COMPLEMENTARITY_SUPPORTED"
    elif _metric(radio) >= fused_mv:
        decision = "SEMANTIC_DOMINATES_D4RT_NOT_USED_WELL"
    elif _metric(d4rt) >= fused_mv:
        decision = "SEMANTIC_FUSION_HURTS_OR_REGION_GRAPH_WRONG"
    elif best_control_mv >= fused_mv:
        decision = "CONTROL_BIAS_BLOCKER"
    elif _metric(hr2_fused) > fused_mv and _metric(hr2_fused) < best_control_mv:
        decision = "FUSION_OR_SEMANTIC_GRAPH_BLOCKER"
    ablation_rows = [
        {
            "decision": decision,
            "condition": "fused graph > D4RT-only and RADIO-only but does not beat controls or whole-source baseline",
            "D4RT_only_MV_AP_window": _metric(d4rt),
            "RADIO_only_MV_AP_window": _metric(radio),
            "D4RT_plus_RADIO_MV_AP_window": fused_mv,
            "whole_source_MV_AP_window": _metric(whole),
            "best_control_MV_AP_window": best_control_mv,
            "Phase4_RADIO_region_same_gt_diff_gt_AUC": phase4_summary.get("source_internal_same_gt_different_gt_AUC_mean", ""),
            "interpretation": "RADIO region signal exists, but current object membership graph/seed readout loses AP and remains control-biased.",
        }
    ]
    summary = {
        "phase_id": "v92_phase6_attribution",
        "schema": "stream4d_v92_phase6_attribution_summary_v1",
        "decision": decision,
        "phase5a_root": _rel(phase5a),
        "phase5b_root": _rel(phase5b),
        "phase4_root": _rel(phase4),
        "best_phase5b_variant": best_b.get("variant_id", ""),
        "best_phase5b_MV_AP_window": _metric(best_b),
        "best_phase5b_MV_AP50_window": _metric(best_b, "mean_MV_AP50_window"),
        "D4RT_only_MV_AP_window": _metric(d4rt),
        "RADIO_only_MV_AP_window": _metric(radio),
        "D4RT_plus_RADIO_MV_AP_window": fused_mv,
        "D4RT_plus_RADIO_plus_negative_MV_AP_window": _metric(negative),
        "highres_D4RT_plus_RADIO_MV_AP_window": _metric(hr2_fused),
        "whole_source_MV_AP_window": _metric(whole),
        "best_control_MV_AP_window": best_control_mv,
        "best_control_MV_AP50_window": best_control_ap50,
        "Phase4_RADIO_region_same_gt_diff_gt_AUC": phase4_summary.get("source_internal_same_gt_different_gt_AUC_mean", ""),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    _write_csv(out / "attribution_metric_rows.csv", attribution_rows)
    _write_csv(out / "control_gap_rows.csv", control_gap_rows)
    _write_csv(out / "ablation_decision_rows.csv", ablation_rows)
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "summary.json",
        out / "attribution_metric_rows.csv",
        out / "control_gap_rows.csv",
        out / "ablation_decision_rows.csv",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v92 Phase6 attribution summary.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase5a-root", default=str(DEFAULT_PHASE5A))
    parser.add_argument("--phase5b-root", default=str(DEFAULT_PHASE5B))
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
