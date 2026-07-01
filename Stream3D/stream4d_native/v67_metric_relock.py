from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _write_json  # noqa: E402
from tools.run_v66_metric_lock import run as run_v66_metric_lock  # noqa: E402


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _project(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metric_selfcheck_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("case") or ""): row for row in rows}


def _mean(rows: list[float]) -> float | None:
    return float(np.mean(rows)) if rows else None


def _scene_mv_ap_support(v66_mv_ap_root: Path) -> dict[str, Any]:
    rows_path = v66_mv_ap_root / "mv_ap_rows.csv"
    summary_path = v66_mv_ap_root / "mv_ap_summary.json"
    if not rows_path.exists() or not summary_path.exists():
        return {
            "available": False,
            "rows_csv": _rel(rows_path),
            "summary_json": _rel(summary_path),
            "stream3d_score_free_match50_recall_mean": None,
            "stream3d_rendering_diagnostic_only": None,
            "soma_prediction_uses_gt": None,
        }
    rows = _read_csv(rows_path)
    summary = _read_json(summary_path)
    stream3d_sf50 = [
        value
        for value in (
            _float_or_none(row.get("score_free_match50_recall"))
            for row in rows
            if row.get("method") == "Stream3D_constant" and str(row.get("stride")) == "5"
        )
        if value is not None
    ]
    soma_uses_gt_flags = [
        str(row.get("uses_gt_for_prediction")).lower() == "true"
        for row in rows
        if str(row.get("method") or "").startswith("SOMA_current")
    ]
    stream3d_diag_flags = [
        str(row.get("diagnostic_only")).lower() == "true"
        and str(row.get("forbidden_for_method_table")).lower() == "true"
        for row in rows
        if row.get("method") == "Stream3D_constant"
    ]
    return {
        "available": True,
        "rows_csv": _rel(rows_path),
        "summary_json": _rel(summary_path),
        "summary_gate": summary.get("gate", {}),
        "stream3d_score_free_match50_recall_mean": _mean(stream3d_sf50),
        "stream3d_rendering_diagnostic_only": bool(stream3d_diag_flags) and all(stream3d_diag_flags),
        "soma_prediction_uses_gt": any(soma_uses_gt_flags) if soma_uses_gt_flags else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    v66_args = argparse.Namespace(
        output_root=str(output_root),
        v65_constant_aggregate=args.v65_constant_aggregate,
        v65_predarea_aggregate=args.v65_predarea_aggregate,
        v65_constant_sha256sums=args.v65_constant_sha256sums,
        v65_predarea_sha256sums=args.v65_predarea_sha256sums,
    )
    v66_payload = run_v66_metric_lock(v66_args)
    selfcheck_rows = _read_csv(output_root / "metric_selfcheck_rows.csv")
    cases = _metric_selfcheck_lookup(selfcheck_rows)
    mv_support = _scene_mv_ap_support(_project(args.v66_mv_ap_root))

    record = {
        "synthetic_perfect_AP": _float_or_none(cases.get("perfect", {}).get("AP")),
        "synthetic_all_background_AP": _float_or_none(cases.get("all_background_prediction", {}).get("AP")),
        "synthetic_split_object_AP": _float_or_none(cases.get("split_one_object", {}).get("AP")),
        "synthetic_merge_object_AP": _float_or_none(cases.get("merge_two_objects", {}).get("AP")),
        "stream3d_stride5_10_delta_AP": v66_payload.get("stream3d_stride_delta_abs_AP"),
        "stream3d_score_free_match50_recall_mean": mv_support.get("stream3d_score_free_match50_recall_mean"),
        "soma_prediction_uses_gt": mv_support.get("soma_prediction_uses_gt"),
        "stream3d_rendering_diagnostic_only": mv_support.get("stream3d_rendering_diagnostic_only"),
    }
    gate = {
        "v66_metric_lock_gate_pass": bool(v66_payload.get("gate", {}).get("pass")),
        "synthetic_perfect_AP_eq_1": record["synthetic_perfect_AP"] == 1.0,
        "synthetic_all_background_AP_eq_0": record["synthetic_all_background_AP"] == 0.0,
        "stream3d_stride5_10_delta_AP_le_0p02": (
            record["stream3d_stride5_10_delta_AP"] is not None
            and float(record["stream3d_stride5_10_delta_AP"]) <= 0.02
        ),
        "soma_prediction_uses_gt_false": record["soma_prediction_uses_gt"] is False,
        "stream3d_rendering_diagnostic_only_true": record["stream3d_rendering_diagnostic_only"] is True,
        "score_free_metric_present": record["stream3d_score_free_match50_recall_mean"] is not None,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    payload = {
        "phase": "v67_phase0_metric_relock",
        "diagnostic_only": True,
        "metric_name": "scene_level_multi_view_2d_AP",
        "recorded_fields": record,
        "gate": gate,
        "inputs": {
            "v65_constant_aggregate": _rel(args.v65_constant_aggregate),
            "v65_predarea_aggregate": _rel(args.v65_predarea_aggregate),
            "v65_constant_sha256sums": _rel(args.v65_constant_sha256sums),
            "v65_predarea_sha256sums": _rel(args.v65_predarea_sha256sums),
            "v66_mv_ap_root": _rel(args.v66_mv_ap_root),
        },
        "v66_metric_lock_summary_json": _rel(output_root / "metric_lock_summary.json"),
        "metric_selfcheck_rows_csv": _rel(output_root / "metric_selfcheck_rows.csv"),
        "code_audit_rows_csv": _rel(output_root / "code_audit_rows.csv"),
        "summary_contract_rows_csv": _rel(output_root / "summary_contract_rows.csv"),
        "sha256_sidecar_rows_csv": _rel(output_root / "sha256_sidecar_rows.csv"),
        "provenance_note": (
            "This v67 relock re-runs the v66 metric-lock checks into the v67 output root, "
            "then records the extra v67 required fields from current on-disk v66 MV-AP support artifacts."
        ),
    }
    _write_json(output_root / "metric_relock_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v67 Phase 0 metric relock.")
    parser.add_argument("--output-root", default="outputs/audit/v67_phase0_metric_relock")
    parser.add_argument(
        "--v65-constant-aggregate",
        default="outputs/audit/v65_scene_multiview_2d_ap_scene0050_full_v4_scoreaudit/aggregate_summary.json",
    )
    parser.add_argument(
        "--v65-predarea-aggregate",
        default="outputs/audit/v65_scene_multiview_2d_ap_scene0050_predarea_diagnostic_v2_scoreaudit/aggregate_summary.json",
    )
    parser.add_argument(
        "--v65-constant-sha256sums",
        default="outputs/audit/v65_scene_multiview_2d_ap_scene0050_full_v4_scoreaudit/SHA256SUMS.txt",
    )
    parser.add_argument(
        "--v65-predarea-sha256sums",
        default="outputs/audit/v65_scene_multiview_2d_ap_scene0050_predarea_diagnostic_v2_scoreaudit/SHA256SUMS.txt",
    )
    parser.add_argument("--v66-mv-ap-root", default="outputs/audit/v66_scene_mv_ap_probe5_full")
    return parser.parse_args()
