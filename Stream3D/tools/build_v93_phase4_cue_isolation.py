from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ID = "v93_phase4_cue_isolation"
RUN_ID = "v93_phase4_cue_isolation_from_locked_v92_readouts"
OUT = ROOT / "outputs/audit/v93_phase4_cue_isolation"

V92_PHASE5B = ROOT / "outputs/audit/v92_phase5b_source_container_edge_field"
V92_PHASE6 = ROOT / "outputs/audit/v92_phase6_attribution"
V93_PHASE3 = ROOT / "outputs/audit/v93_phase3_region_edge_graph"
V93_EDGE = ROOT / "outputs/audit/v93_phase4_edge_only_materialization"


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


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(ROOT.parent))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _num(value: Any) -> float | str:
    try:
        if value is None or value == "":
            return ""
        return float(value)
    except Exception:
        return ""


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _row_by_variant(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("variant_id", ""): row for row in rows}


def _metric_row(cue_id: str, source_variant: str, cue_family: str, row: dict[str, str], source_artifact: Path, notes: str, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v93_phase4_variant_metric_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": cue_id,
        "source_variant_id": source_variant,
        "cue_family": cue_family,
        "scene_id": "ALL_DEV",
        "split": "dev",
        "window_id": "ALL_WINDOWS",
        "MV_AP_window": _num(row.get("mean_MV_AP_window")),
        "MV_AP50_window": _num(row.get("mean_MV_AP50_window")),
        "MV_AP25_window": _num(row.get("mean_MV_AP25_window")),
        "ScoreFreeMatch50_window": _num(row.get("mean_score_free_Match50_window")),
        "mean_generated_area_ratio": "",
        "undercoverage_proxy": "",
        "overcoverage_proxy": "",
        "same_frame_collision_count": _num(row.get("same_frame_collision_count")),
        "missing_mask_raster_count": _num(row.get("missing_mask_raster_count")),
        "control_gap_vs_random_edge": "",
        "control_gap_vs_whole_source": "",
        "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
        "uses_future": _bool(row.get("uses_future")),
        "source_artifact": _rel(source_artifact),
        "source_artifact_sha256": _sha256(source_artifact) if source_artifact.exists() else "",
        "notes": notes,
        "created_at": created_at,
    }


def _edge_metric_row(row: dict[str, str], source_artifact: Path, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v93_phase4_variant_metric_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": row.get("variant_id", ""),
        "source_variant_id": row.get("variant_id", ""),
        "cue_family": "edge_only" if not str(row.get("variant_id", "")).startswith("R") else "edge_control",
        "scene_id": "ALL_DEV",
        "split": "dev",
        "window_id": "ALL_WINDOWS",
        "MV_AP_window": _num(row.get("mean_MV_AP_window")),
        "MV_AP50_window": _num(row.get("mean_MV_AP50_window")),
        "MV_AP25_window": _num(row.get("mean_MV_AP25_window")),
        "ScoreFreeMatch50_window": _num(row.get("mean_score_free_Match50_window")),
        "mean_generated_area_ratio": _num(row.get("mean_generated_area_ratio")),
        "undercoverage_proxy": _num(row.get("undercoverage_proxy")),
        "overcoverage_proxy": _num(row.get("overcoverage_proxy")),
        "same_frame_collision_count": _num(row.get("same_frame_collision_count")),
        "missing_mask_raster_count": _num(row.get("missing_mask_raster_count")),
        "control_gap_vs_random_edge": _num(row.get("control_gap_vs_random_edge")),
        "control_gap_vs_whole_source": _num(row.get("control_gap_vs_whole_source")),
        "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
        "uses_future": _bool(row.get("uses_future")),
        "source_artifact": _rel(source_artifact),
        "source_artifact_sha256": _sha256(source_artifact) if source_artifact.exists() else "",
        "notes": "edge-only/control metric from v93 Phase4 edge-only materialization; proxies are source-area only, not GT",
        "created_at": created_at,
    }


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    phase5b_rows = _row_by_variant(_read_csv(V92_PHASE5B / "variant_metric_rows.csv"))
    phase6 = _read_json(V92_PHASE6 / "summary.json")
    phase3 = _read_json(V93_PHASE3 / "summary.json")

    source_artifact = V92_PHASE5B / "variant_metric_rows.csv"
    mapping = [
        ("E0_whole_source", "V92_F0_whole_source_mask", "whole_source", "whole source baseline from locked v92 Phase5B readout"),
        ("D0_D4RT_witness_only", "V92_F1_d4rt_seed_only", "d4rt_only", "D4RT witness-only readout from locked v92 Phase5B"),
        ("S0_RADIO_region_only", "V92_F2_radio_region_only", "radio_region_only", "RADIO region-only readout from locked v92 Phase5B"),
        ("F_proxy_D4RT_RADIO_unary", "V92_F3_d4rt_plus_radio_unary", "fusion_proxy", "D4RT+RADIO unary proxy, diagnostic only for cue comparison"),
        ("F_proxy_D4RT_RADIO_graph", "V92_F4_d4rt_radio_graph", "fusion_proxy", "D4RT+RADIO graph proxy, diagnostic only for cue comparison"),
        ("R0_random_region_seed_control", "V92_C5_random_region_seed_control", "random_control", "random region seed control from locked v92 Phase5B"),
    ]
    metric_rows = [
        _metric_row(cue_id, src_id, family, phase5b_rows.get(src_id, {}), source_artifact, notes, created_at)
        for cue_id, src_id, family, notes in mapping
    ]
    edge_metric_path = V93_EDGE / "variant_metric_rows.csv"
    edge_metric_rows = _read_csv(edge_metric_path)
    metric_rows.extend(_edge_metric_row(row, edge_metric_path, created_at) for row in edge_metric_rows)
    _write_csv(OUT / "variant_metric_rows.csv", metric_rows)

    cue_rows = []
    for row in metric_rows:
        cue_rows.append(
            {
                "schema_version": "stream4d_v93_phase4_cue_diagnostic_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": row["variant_id"],
                "cue_family": row["cue_family"],
                "MV_AP_window": row["MV_AP_window"],
                "MV_AP50_window": row["MV_AP50_window"],
                "interpretation": "locked_existing_readout_not_new_v93_claim",
                "uses_gt_for_prediction": row["uses_gt_for_prediction"],
                "uses_future": row["uses_future"],
                "created_at": created_at,
            }
        )
    _write_csv(OUT / "cue_diagnostic_rows.csv", cue_rows)

    expected_edge_variants = [
        "E1_outer_edge_only",
        "E2_nested_overlap_edge",
        "E3_competing_edge",
        "E4_repeated_multiview_edge",
        "R0_random_edge_control",
        "R1_shuffled_edge_control",
    ]
    edge_metric_ids = {row.get("variant_id", "") for row in edge_metric_rows}
    edge_failure_rows = _read_csv(V93_EDGE / "variant_failure_rows.csv")
    edge_unsupported_ids = {
        row.get("variant_id", "")
        for row in edge_failure_rows
        if row.get("failure_type") == "NO_REPEATED_MULTIVIEW_EDGE_SUPPORT"
    }
    missing_edge_variants = [
        variant_id for variant_id in expected_edge_variants if variant_id not in edge_metric_ids and variant_id not in edge_unsupported_ids
    ]
    failure_rows: list[dict[str, Any]] = []
    if missing_edge_variants:
        failure_rows.extend(
            {
                "schema_version": "stream4d_v93_phase4_variant_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "failure_type": "EDGE_ONLY_READOUT_NOT_MATERIALIZED",
                "repair_direction": "implement edge-only materialization from v93 Phase1 mask_edge_hypothesis_rows and evaluate with v65 MV_AP before claiming full Phase4 cue isolation",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
            for variant_id in missing_edge_variants
        )
    for row in edge_failure_rows:
        failure_rows.append(
            {
                "schema_version": "stream4d_v93_phase4_variant_failure_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": row.get("variant_id", ""),
                "failure_type": row.get("failure_type", ""),
                "repair_direction": row.get("repair_direction", ""),
                "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                "uses_future": _bool(row.get("uses_future")),
                "created_at": created_at,
            }
        )
    _write_csv(OUT / "variant_failure_rows.csv", failure_rows)

    # The readout rows include locked v92 Phase5B cue rows and v93 edge-only rows so reviewers can trace evaluated masks.
    readout_row_counts: dict[str, int] = {}
    for name in ["mv_object_rows.csv", "mv_object_frame_mask_rows.csv"]:
        transformed = []
        for source_root, source_name in [(V92_PHASE5B, "locked_v92_phase5b"), (V93_EDGE, "v93_edge_only_materialization")]:
            rows = _read_csv(source_root / name)
            readout_row_counts[f"{source_name}_{name.replace('.csv', '')}"] = len(rows)
            for row in rows:
                out = dict(row)
                out["schema_version"] = "stream4d_v93_phase4_" + name.replace(".csv", "") + "_v1"
                out["phase_id"] = PHASE_ID
                out["run_id"] = RUN_ID
                out["phase4_readout_source"] = source_name
                out["created_at"] = created_at
                transformed.append(out)
        _write_csv(OUT / name, transformed)

    available_ids = {row["variant_id"] for row in metric_rows if row.get("MV_AP_window") != ""}
    materialized_edge_ids = sorted(edge_metric_ids)
    unsupported_edge_ids = sorted(edge_unsupported_ids)
    phase4_complete = len(missing_edge_variants) == 0
    edge_summary = _read_json(V93_EDGE / "summary.json")
    best_edge = max(
        [row for row in metric_rows if row.get("variant_id") in edge_metric_ids],
        key=lambda row: float(row.get("MV_AP_window") or -999.0),
        default={},
    )
    summary = {
        "schema": "stream4d_v93_phase4_cue_isolation_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": (
            "BLOCK_V93_PHASE4_EDGE_ONLY_READOUTS_MISSING"
            if missing_edge_variants
            else "PASS_V93_PHASE4_CUE_ISOLATION_WITH_E4_UNSUPPORTED"
            if unsupported_edge_ids
            else "PASS_V93_PHASE4_CUE_ISOLATION"
        ),
        "phase4_complete": phase4_complete,
        "phase4_complete_scope": "all materializable cue variants; unsupported variants are explicitly recorded",
        "available_cue_variant_count": len(available_ids),
        "missing_edge_only_variant_count": len(failure_rows),
        "missing_materializable_edge_only_variant_count": len(missing_edge_variants),
        "materialized_edge_only_variant_count": len(materialized_edge_ids),
        "materialized_edge_only_variants": materialized_edge_ids,
        "unsupported_edge_only_variant_count": len(unsupported_edge_ids),
        "unsupported_edge_only_variants": unsupported_edge_ids,
        "best_edge_only_variant_id": best_edge.get("variant_id", edge_summary.get("best_edge_variant_id", "")),
        "best_edge_only_MV_AP_window": best_edge.get("MV_AP_window", edge_summary.get("best_edge_MV_AP_window", "")),
        "best_edge_only_MV_AP50_window": best_edge.get("MV_AP50_window", edge_summary.get("best_edge_MV_AP50_window", "")),
        "best_edge_only_control_gap_vs_random_edge": best_edge.get("control_gap_vs_random_edge", edge_summary.get("best_edge_control_gap_vs_random_edge", "")),
        "best_edge_only_control_gap_vs_whole_source": best_edge.get("control_gap_vs_whole_source", edge_summary.get("best_edge_control_gap_vs_whole_source", "")),
        "row_counts": {
            "variant_metric_rows": len(metric_rows),
            "cue_diagnostic_rows": len(cue_rows),
            "variant_failure_rows": len(failure_rows),
            **readout_row_counts,
        },
        "whole_source_MV_AP_window": phase6.get("whole_source_MV_AP_window", ""),
        "D4RT_only_MV_AP_window": phase6.get("D4RT_only_MV_AP_window", ""),
        "RADIO_only_MV_AP_window": phase6.get("RADIO_only_MV_AP_window", ""),
        "D4RT_plus_RADIO_MV_AP_window": phase6.get("D4RT_plus_RADIO_MV_AP_window", ""),
        "best_control_MV_AP_window": phase6.get("best_control_MV_AP_window", ""),
        "Phase3_region_feature_available_rate": phase3.get("region_feature_available_rate", ""),
        "Phase3_source_internal_AUC_diagnostic": phase3.get("source_internal_same_gt_different_gt_AUC_diagnostic", ""),
        "uses_gt_for_prediction_count": sum(1 for row in metric_rows if _bool(row.get("uses_gt_for_prediction"))),
        "uses_future_count": sum(1 for row in metric_rows if _bool(row.get("uses_future"))),
        "notes": (
            "Cue isolation includes locked v92 D4RT/RADIO readouts plus v93 edge-only materialization for E1/E2/E3/R0/R1. "
            "E4 is explicitly unsupported because repeated_multiview edge rows are absent; it is not fabricated."
        ),
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                V92_PHASE5B / "variant_metric_rows.csv",
                V92_PHASE5B / "mv_object_rows.csv",
                V92_PHASE5B / "mv_object_frame_mask_rows.csv",
                V93_EDGE / "mv_object_rows.csv",
                V93_EDGE / "mv_object_frame_mask_rows.csv",
                V92_PHASE6 / "summary.json",
                V93_PHASE3 / "summary.json",
                V93_EDGE / "summary.json",
                V93_EDGE / "variant_metric_rows.csv",
            ]
            if path.exists()
        },
        "duration_sec": time.time() - started,
        "created_at": created_at,
    }
    _write_json(OUT / "summary.json", summary)
    sha_rows = {path.name: _sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    _write_json(OUT / "SHA256SUMS.json", sha_rows)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
