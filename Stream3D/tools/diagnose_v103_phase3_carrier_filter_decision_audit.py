#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
PHASE_ID = "v103_phase3_carrier_filter_decision_audit"


DEFAULT_PHASE3_ROOT = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_q5c_objlike16384_competing_repair5"
DEFAULT_GT_DIAG_ROOT = AUDIT_ROOT / "v103_phase3_reliable_carrier_gt_diagnostic_r11_positive_core_phase4r12"
DEFAULT_ATTR_ROOT = AUDIT_ROOT / "v103_phase3_filter_failure_attribution_r2_repair5"
DEFAULT_PHASE4_AUDIT_ROOT = AUDIT_ROOT / "v103_phase4_affinity_correctness_r4_positive_core_r12"
DEFAULT_PHASE4_ROOT = AUDIT_ROOT / "v103_phase4_positive_core_affinity_q5c_repair5_r12_dual_role"
DEFAULT_PHASE5_ROOT = AUDIT_ROOT / "v103_phase5_positive_core_pooling_q5c_repair5_r1"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase3_carrier_filter_decision_audit_r1"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _scene_variant_map(gt_diag_summary: dict[str, Any], phase3_summary: dict[str, Any]) -> dict[str, str]:
    selected = dict(phase3_summary.get("selected_variant_by_scene", {}))
    selected.update(dict(gt_diag_summary.get("selected_variant_override_by_scene", {})))
    selected.update(dict(gt_diag_summary.get("selected_variant_by_scene", {})))
    return {str(k): str(v) for k, v in selected.items()}


def _decision_for_filter(row: pd.Series) -> tuple[str, str]:
    multi_rate = float(row["retained_multi_gt_rate"])
    clean_recall = float(row["clean_retention_rate"])
    multi_vs_clean = float(row["multi_over_clean_retention_ratio"])
    purity_delta = float(row.get("purity_delta_vs_unfiltered", 0.0))
    multi_reduction = float(row.get("multi_GT_relative_reduction_vs_unfiltered", 0.0))

    if multi_rate >= 0.30 and clean_recall < 0.10:
        return (
            "FILTER_BAD_LEAKAGE_PLUS_COVERAGE_LOSS",
            "retained multi-GT remains high while clean same-GT recall is below 10%; not safe as a positive witness pool",
        )
    if multi_vs_clean >= 0.95 and purity_delta < 0.005 and multi_reduction < 0.05:
        return (
            "FILTER_DOES_NOT_SEPARATE_CLEAN_FROM_MULTI_GT",
            "clean and multi-GT carriers are retained at almost the same rate; semantic/geometric scores do not separate enough",
        )
    if multi_rate <= 0.15 and clean_recall < 0.15:
        return (
            "RELATIVELY_CLEAN_BUT_COVERAGE_LIMITED",
            "retained pool is comparatively clean, but clean carrier recall is still low; usable as sparse anchor rather than full coverage",
        )
    if multi_rate <= 0.20:
        return (
            "USABLE_PRECISION_ANCHOR_NEEDS_COVERAGE_PROVIDER",
            "retained pool is clean enough for anchor use, but coverage should be supplemented before object birth",
        )
    return (
        "FILTER_PARTIAL_NOT_POSITIVE_WITNESS",
        "filter improved some GT-free gates but retained pool is not reliable enough to act as full positive primitive evidence",
    )


def _build_filter_rows(
    phase3_root: Path,
    gt_diag_root: Path,
    attr_root: Path,
    selected_by_scene: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric = pd.read_csv(gt_diag_root / "carrier_gt_metric_rows.csv")
    retention = pd.read_csv(attr_root / "retention_attribution_rows.csv")
    score = pd.read_csv(attr_root / "score_separation_rows.csv")
    source = pd.read_csv(attr_root / "source_retention_rows.csv")
    phase3_metric = pd.read_csv(phase3_root / "carrier_filter_metric_rows.csv")

    filter_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    key_metrics = {
        "reliability_s2",
        "r_geo",
        "r_mask",
        "r_sem",
        "semantic_contradiction_rate",
        "competing_mask_conflict_rate",
        "source_risk_score",
        "broad_mask_participation_rate",
        "normalized_jitter",
        "object_like_mask_rate",
        "visibility_rate",
        "in_image_rate",
    }

    for scene, variant in selected_by_scene.items():
        ret = retention[(retention["scene_id"] == scene) & (retention["variant_id"] == variant)]
        met = metric[(metric["scene_id"] == scene) & (metric["phase3_variant_id"] == variant)]
        p3 = phase3_metric[(phase3_metric["scene_id"] == scene) & (phase3_metric["variant_id"] == variant)]
        if ret.empty or met.empty or p3.empty:
            filter_rows.append(
                {
                    "schema_version": "stream4d_v103_phase3_carrier_filter_decision_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "selected_variant_id": variant,
                    "decision": "MISSING_REQUIRED_INPUT_ROWS",
                    "evidence": f"retention={len(ret)} metric={len(met)} phase3_metric={len(p3)}",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic": True,
                    "diagnostic_only": True,
                }
            )
            continue
        ret_s = ret.iloc[0]
        met_s = met.iloc[0]
        p3_s = p3.iloc[0]
        merged = ret_s.to_dict()
        merged.update(met_s.to_dict())
        decision, interpretation = _decision_for_filter(pd.Series(merged))
        filter_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_carrier_filter_decision_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "selected_variant_id": variant,
                "decision": decision,
                "interpretation": interpretation,
                "total_carrier_count": int(ret_s["total_carrier_count"]),
                "retained_carrier_count": int(ret_s["retained_carrier_count"]),
                "retained_carrier_rate": float(p3_s["retained_carrier_rate"]),
                "clean_single_gt_total": int(ret_s["clean_single_gt_total"]),
                "multi_gt_total": int(ret_s["multi_gt_total"]),
                "retained_clean_single_gt_count": int(ret_s["retained_clean_single_gt_count"]),
                "retained_multi_gt_count": int(ret_s["retained_multi_gt_count"]),
                "clean_retention_rate": float(ret_s["clean_retention_rate"]),
                "multi_gt_retention_rate": float(ret_s["multi_gt_retention_rate"]),
                "multi_over_clean_retention_ratio": float(ret_s["multi_over_clean_retention_ratio"]),
                "retained_clean_precision": float(ret_s["retained_clean_precision"]),
                "retained_multi_gt_rate": float(ret_s["retained_multi_gt_rate"]),
                "dominant_gt_purity_mean": float(met_s["dominant_gt_purity_mean"]),
                "dominant_gt_purity_p10": float(met_s["dominant_gt_purity_p10"]),
                "purity_delta_vs_unfiltered": float(met_s["purity_delta_vs_unfiltered"]),
                "multi_GT_relative_reduction_vs_unfiltered": float(met_s["multi_GT_relative_reduction_vs_unfiltered"]),
                "selected_clean_gt_instance_count": int(met_s["selected_clean_gt_instance_count"]),
                "selected_clean_carriers_per_gt_p10": float(met_s["selected_clean_carriers_per_gt_p10"]),
                "object_like_mask_support_p10": float(p3_s["object_like_mask_support_p10"]),
                "boundary_band_support_p10": float(p3_s["boundary_band_support_p10"]),
                "broad_relative_reduction": float(p3_s["broad_relative_reduction"]),
                "semantic_relative_reduction": float(p3_s["semantic_relative_reduction"]),
                "jitter_relative_reduction": float(p3_s["jitter_relative_reduction"]),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
                "diagnostic_only": True,
            }
        )

        sub_score = score[
            (score["scene_id"] == scene)
            & (score["variant_id"] == variant)
            & (score["metric_key"].isin(key_metrics))
        ]
        for _, row in sub_score.iterrows():
            clean_direction = str(row["clean_direction"])
            if clean_direction == "higher":
                separation_ok = float(row["clean_minus_multi_good_direction_p50"]) > 0
            else:
                separation_ok = float(row["clean_minus_multi_good_direction_p50"]) > 0
            score_rows.append(
                {
                    "schema_version": "stream4d_v103_phase3_score_separation_decision_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "variant_id": variant,
                    "metric_key": str(row["metric_key"]),
                    "clean_direction": clean_direction,
                    "clean_p50": float(row["clean_p50"]),
                    "multi_gt_p50": float(row["multi_gt_p50"]),
                    "clean_minus_multi_good_direction_p50": float(row["clean_minus_multi_good_direction_p50"]),
                    "retained_clean_p50": float(row["retained_clean_p50"]),
                    "retained_multi_gt_p50": float(row["retained_multi_gt_p50"]),
                    "separates_clean_from_multi_gt_at_p50": bool(separation_ok),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic": True,
                    "diagnostic_only": True,
                }
            )

        sub_source = source[(source["scene_id"] == scene) & (source["variant_id"] == variant)]
        for _, row in sub_source.iterrows():
            retained_total = int(row["source_retained_total"])
            multi_rate = float(row["source_retained_multi_gt_rate"])
            if retained_total == 0:
                source_decision = "SOURCE_VETOED_OR_UNUSED"
            elif multi_rate >= 0.50:
                source_decision = "SOURCE_HIGH_BAD_LEAKAGE"
            elif multi_rate <= 0.15:
                source_decision = "SOURCE_RELATIVELY_CLEAN"
            else:
                source_decision = "SOURCE_MIXED"
            source_rows.append(
                {
                    "schema_version": "stream4d_v103_phase3_source_retention_decision_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "variant_id": variant,
                    "query_source": str(row["query_source"]),
                    "source_total": int(row["source_total"]),
                    "source_retained_total": retained_total,
                    "source_retained_clean_count": int(row["source_retained_clean_count"]),
                    "source_retained_multi_gt_count": int(row["source_retained_multi_gt_count"]),
                    "source_clean_retention_rate": float(row["source_clean_retention_rate"]),
                    "source_multi_gt_retention_rate": float(row["source_multi_gt_retention_rate"]),
                    "source_retained_multi_gt_rate": multi_rate,
                    "source_decision": source_decision,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic": True,
                    "diagnostic_only": True,
                }
            )
    return filter_rows, score_rows, source_rows


def _build_geometry_rows(
    phase4_audit_root: Path,
    phase4_root: Path,
    phase5_root: Path,
) -> tuple[list[dict[str, Any]], str]:
    correctness = pd.read_csv(phase4_audit_root / "affinity_correctness_rows.csv")
    phase4_gate = pd.read_csv(phase4_root / "gate_rows.csv")
    phase4_metric = pd.read_csv(phase4_root / "primitive_feature_metric_rows.csv")
    phase5_gate = pd.read_csv(phase5_root / "gate_rows.csv")
    phase5_control = pd.read_csv(phase5_root / "mask_feature_control_rows.csv")
    rows: list[dict[str, Any]] = []
    any_bug = False
    for _, corr in correctness.iterrows():
        scene = str(corr["scene_id"])
        gates = phase4_gate[phase4_gate["scene_id"] == scene]
        met = phase4_metric[phase4_metric["scene_id"] == scene].iloc[0]
        p5_gates = phase5_gate[phase5_gate["scene_id"] == scene]
        p5_control = phase5_control[phase5_control["scene_id"] == scene]
        arithmetic_ok = (
            _boolish(corr["carrier_id_exact_match_phase3_phase4"])
            and float(corr["B_ia_max_abs_error_vs_recomputed"]) <= 1e-6
            and int(corr["feature_zero_norm_count"]) == 0
            and float(corr["incidence_label_match_rate"]) == 1.0
        )
        phase4_gates_pass = bool(len(gates) > 0 and gates["pass"].map(_boolish).all())
        phase5_gates_pass = bool(len(p5_gates) > 0 and p5_gates["pass"].map(_boolish).all())
        l2o_pass = bool(len(p5_control) > 0 and p5_control["hard_negative_separation_pass"].map(_boolish).all())
        bug = not (arithmetic_ok and phase4_gates_pass and phase5_gates_pass and l2o_pass)
        any_bug = any_bug or bug
        rows.append(
            {
                "schema_version": "stream4d_v103_geometry_affinity_decision_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "geometry_affinity_decision": "NO_ARITHMETIC_OR_L2O_BUG_EVIDENCE" if not bug else "GEOMETRY_AFFINITY_BUG_OR_CONTROL_FAILURE",
                "carrier_id_exact_match_phase3_phase4": bool(_boolish(corr["carrier_id_exact_match_phase3_phase4"])),
                "B_ia_max_abs_error_vs_recomputed": float(corr["B_ia_max_abs_error_vs_recomputed"]),
                "B_ia_p95_abs_error_vs_recomputed": float(corr["B_ia_p95_abs_error_vs_recomputed"]),
                "incidence_label_match_rate": float(corr["incidence_label_match_rate"]),
                "feature_zero_norm_count": int(corr["feature_zero_norm_count"]),
                "feature_valid_rate": float(met["feature_valid_rate"]),
                "exact_vs_sketch_cosine_p95_error": float(met["exact_vs_sketch_cosine_p95_error"]),
                "broad_mask_feature_contribution_ratio": float(met["broad_mask_feature_contribution_ratio"]),
                "object_like_mask_feature_contribution_ratio": float(met["object_like_mask_feature_contribution_ratio"]),
                "phase4_gates_pass": phase4_gates_pass,
                "phase5_gates_pass": phase5_gates_pass,
                "leave_one_out_control_pass": l2o_pass,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": False,
                "diagnostic_only": False,
            }
        )
    return rows, "GEOMETRY_AFFINITY_BUG_SUSPECTED" if any_bug else "NO_GEOMETRY_AFFINITY_BUG_EVIDENCE_IN_EXISTING_AUDITS"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate v103 carrier-filter and geometry-affinity diagnostic decisions.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--gt-diagnostic-root", default=str(DEFAULT_GT_DIAG_ROOT))
    parser.add_argument("--failure-attribution-root", default=str(DEFAULT_ATTR_ROOT))
    parser.add_argument("--phase4-correctness-root", default=str(DEFAULT_PHASE4_AUDIT_ROOT))
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4_ROOT))
    parser.add_argument("--phase5-root", default=str(DEFAULT_PHASE5_ROOT))
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phase3_root = _project(args.phase3_root)
    gt_diag_root = _project(args.gt_diagnostic_root)
    attr_root = _project(args.failure_attribution_root)
    phase4_audit_root = _project(args.phase4_correctness_root)
    phase4_root = _project(args.phase4_root)
    phase5_root = _project(args.phase5_root)

    phase3_summary = _read_json(phase3_root / "summary.json")
    gt_diag_summary = _read_json(gt_diag_root / "summary.json")
    selected_by_scene = _scene_variant_map(gt_diag_summary, phase3_summary)

    filter_rows, score_rows, source_rows = _build_filter_rows(phase3_root, gt_diag_root, attr_root, selected_by_scene)
    geometry_rows, geometry_decision = _build_geometry_rows(phase4_audit_root, phase4_root, phase5_root)

    _write_csv(out / "carrier_filter_decision_rows.csv", filter_rows)
    _write_csv(out / "score_separation_decision_rows.csv", score_rows)
    _write_csv(out / "source_retention_decision_rows.csv", source_rows)
    _write_csv(out / "geometry_affinity_decision_rows.csv", geometry_rows)

    scene_decision_by_scene = {str(row["scene_id"]): str(row["decision"]) for row in filter_rows}
    if any("BAD_LEAKAGE" in d for d in scene_decision_by_scene.values()):
        overall_filter_decision = "D4RT_FILTER_NOT_SAFE_AS_FULL_POSITIVE_WITNESS"
    elif any("COVERAGE_LIMITED" in d for d in scene_decision_by_scene.values()):
        overall_filter_decision = "D4RT_RELATIVELY_CLEAN_BUT_COVERAGE_LIMITED"
    else:
        overall_filter_decision = "D4RT_FILTER_USABLE_AS_PRECISION_ANCHOR_ONLY"

    recommended_next = (
        "Use D4RT E5/E1 retained carriers only as sparse precision anchors or veto/risk evidence; "
        "do not treat the full retained pool as object birth coverage. "
        "If adding DA3, keep the v103 primitive affinity formula and use DA3 as coverage provider, "
        "with E3/source-risk evidence only as hard veto or downweight."
    )
    summary = {
        "schema_version": "stream4d_v103_phase3_carrier_filter_decision_audit_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": f"{overall_filter_decision}__{geometry_decision}",
        "overall_filter_decision": overall_filter_decision,
        "geometry_decision": geometry_decision,
        "scene_decision_by_scene": scene_decision_by_scene,
        "selected_variant_by_scene": selected_by_scene,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "diagnostic_only": True,
        "truthfulness_note": "This consolidates existing GT-free gates and GT-only diagnostics. GT is used only to explain filter failure after preselected variants, not for thresholds or prediction.",
        "recommended_next": recommended_next,
        "inputs": {
            "phase3_root": _rel(phase3_root),
            "gt_diagnostic_root": _rel(gt_diag_root),
            "failure_attribution_root": _rel(attr_root),
            "phase4_correctness_root": _rel(phase4_audit_root),
            "phase4_root": _rel(phase4_root),
            "phase5_root": _rel(phase5_root),
        },
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "carrier_filter_decision_rows": _rel(out / "carrier_filter_decision_rows.csv"),
            "score_separation_decision_rows": _rel(out / "score_separation_decision_rows.csv"),
            "source_retention_decision_rows": _rel(out / "source_retention_decision_rows.csv"),
            "geometry_affinity_decision_rows": _rel(out / "geometry_affinity_decision_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
