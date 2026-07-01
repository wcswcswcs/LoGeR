from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _rel  # noqa: E402


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _metric_row(metric: str, value: Any, source: str, pass_value: bool | None = None, notes: str = "") -> dict[str, Any]:
    return {"metric": metric, "value": value, "pass": pass_value, "source": source, "notes": notes}


def _best_stream3d_sf50(rows: list[dict[str, Any]]) -> float | None:
    vals = [
        _float(row.get("score_free_match50_recall"))
        for row in rows
        if str(row.get("method") or "").startswith("Stream3D") and str(row.get("scene_id") or "")
    ]
    return _mean(vals)


def _soma_current_sf50(rows: list[dict[str, Any]]) -> float | None:
    vals = [
        _float(row.get("score_free_match50_recall"))
        for row in rows
        if "SOMA" in str(row.get("method") or "") and str(row.get("scene_id") or "")
    ]
    return _mean(vals)


def _row_by_name(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get(key) or "") == value:
            return row
    return {}


def _row_by_any_name(rows: list[dict[str, Any]], key: str, values: list[str]) -> dict[str, Any]:
    wanted = set(values)
    for row in rows:
        if str(row.get(key) or "") in wanted:
            return row
    return {}


def _code_audit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = [
        "stream4d_native/v71_fact_lock.py",
        "stream4d_native/v71_candidate_bank.py",
        "stream4d_native/v71_d4rt_atoms.py",
        "stream4d_native/v71_semantic_features.py",
        "stream4d_native/v71_key_atoms.py",
        "stream4d_native/v71_representative_setcover.py",
        "stream4d_native/v71_local_birth.py",
        "stream4d_native/v71_controls.py",
        "stream4d_native/v71_scene_mv_ap.py",
        "stream4d_native/v71_visualization.py",
        "stream4d_native/v71_final_eval.py",
    ]
    for rel_path in files:
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rows.append(
            {
                "path": rel_path,
                "source_available": path.exists(),
                "line_count": len(text.splitlines()) if text else 0,
                "mentions_gt": "GT" in text or "gt" in text,
                "mentions_uses_gt_for_prediction": "uses_gt_for_prediction" in text,
                "mentions_forbidden_for_method_table": "forbidden_for_method_table" in text,
                "mentions_diagnostic_only": "diagnostic_only" in text,
                "notes": "static token scan only; absence does not prove runtime behavior",
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "v66_metric_lock": args.v66_metric_lock,
        "v66_mv_ap": args.v66_mv_ap,
        "v67_mask_universe": args.v67_mask_universe,
        "v68_candidate_bank": args.v68_candidate_bank,
        "v68_edge_audit": args.v68_edge_audit,
        "v68_local_solver": args.v68_local_solver,
        "v70_final": args.v70_final,
        "v70_casebook": args.v70_casebook,
    }
    missing = [{"name": name, "path": path} for name, path in paths.items() if not _rooted(path).exists()]
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v71_phase0_fact_lock",
            "decision": "FAIL_MISSING_INPUTS",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_inputs": missing,
        }
        _write_json(output_root / "fact_lock_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    v66_metric = _load_json(_rooted(args.v66_metric_lock))
    v66_mv = _load_json(_rooted(args.v66_mv_ap))
    v66_rows = _load_csv_rows(_rooted(str(v66_mv.get("rows_csv") or "")))
    v67 = _load_json(_rooted(args.v67_mask_universe))
    v67_rows = _load_csv_rows(_rooted(str((v67.get("rows") or {}).get("universe_metric_rows_csv") or "")))
    v68_cb = _load_json(_rooted(args.v68_candidate_bank))
    v68_edge = _load_json(_rooted(args.v68_edge_audit))
    v68_solver = _load_json(_rooted(args.v68_local_solver))
    v70 = _load_json(_rooted(args.v70_final))
    v70_casebook = _load_json(_rooted(args.v70_casebook))

    u0 = _row_by_name(v67_rows, "universe_name", "U0_current_selected_masks")
    u2 = _row_by_name(v67_rows, "universe_name", "U2_all_representative_masks")
    u3 = _row_by_name(v67_rows, "universe_name", "U3_all_CropFormer_masks_stride5")
    u4 = _row_by_any_name(v67_rows, "universe_name", ["U4_high_quality_raw_masks", "U4_high_quality_CropFormer_masks"])
    best_cb = v68_cb.get("best_CB") or {}
    combined = v68_edge.get("combined_metrics") or {}
    best_solver = v68_solver.get("best_S") or {}
    v70_metrics = v70.get("key_metrics") or {}
    metric_selfcheck_pass = bool((v66_metric.get("gate") or {}).get("pass"))
    no_gt_for_method_prediction_pass = all(
        [
            bool((v66_metric.get("gate") or {}).get("no_gt_for_soma_prediction", True)),
            not bool(v68_edge.get("uses_gt_for_prediction")),
            not bool(v68_solver.get("uses_gt_for_prediction")),
            v70.get("required_answers", {}).get("v69r2_failed_due_to_proxy_closure") is not None,
        ]
    )
    rows = [
        _metric_row("v66_stream3d_mean_SF50", v66_mv.get("stream3d_mean_score_free_match50_recall") or _best_stream3d_sf50(v66_rows), args.v66_mv_ap),
        _metric_row("v66_soma_current_mean_SF50", _soma_current_sf50(v66_rows), args.v66_mv_ap, None, "null if no SOMA method rows were present in mv_ap_rows.csv"),
        _metric_row("v66_oracle_selected_masks_SF50", None, args.v66_mv_ap, None, "not present as a named field in v66 mv_ap summary; v67 U0 records current selected oracle"),
        _metric_row("v66_decision", "STREAM3D_DIAGNOSTIC_PASS_SOMA_LOCAL_GAP" if (v66_mv.get("gate") or {}).get("stream3d_pass") else "unknown", args.v66_mv_ap),
        _metric_row("v67_U0_current_selected_oracle_SF50", _float(u0.get("oracle_SF50_mean")), args.v67_mask_universe),
        _metric_row("v67_U2_representative_oracle_SF50", _float(u2.get("oracle_SF50_mean")), args.v67_mask_universe),
        _metric_row("v67_U3_raw_oracle_SF50", _float(u3.get("oracle_SF50_mean")), args.v67_mask_universe),
        _metric_row(
            "v67_U4_high_quality_oracle_SF50",
            _first_float(u4, "oracle_SF50_mean", "oracle_score_free_match50_recall_mean"),
            args.v67_mask_universe,
            None,
            "supports legacy and current U4 row names",
        ),
        _metric_row("v67_decision", v67.get("decision"), args.v67_mask_universe),
        _metric_row(
            "v68_candidate_bank_oracle_SF50",
            _first_float(best_cb, "local_SF50", "local_SF50_mean", "local_score_free_match50_recall_mean", "oracle_SF50"),
            args.v68_candidate_bank,
            None,
            "supports v68 best_CB metric aliases",
        ),
        _metric_row(
            "v68_candidate_bank_oracle_AP50",
            _first_float(best_cb, "local_AP50", "local_AP50_mean", "oracle_AP50"),
            args.v68_candidate_bank,
            None,
            "supports v68 best_CB metric aliases",
        ),
        _metric_row(
            "v68_candidate_bank_oracle_GT_best_IoU_mean",
            _first_float(best_cb, "GT_best_IoU_mean", "local_GT_best_IoU_mean_mean", "oracle_GT_best_IoU_mean"),
            args.v68_candidate_bank,
            None,
            "supports v68 best_CB metric aliases",
        ),
        _metric_row("v68_DINO_edge_AUC", combined.get("edge_AUC"), args.v68_edge_audit),
        _metric_row("v68_DINO_top1_precision", combined.get("top1_precision"), args.v68_edge_audit),
        _metric_row(
            "v68_best_local_solver_SF50",
            _first_float(best_solver, "local_SF50", "local_SF50_mean", "local_score_free_match50_recall_mean"),
            args.v68_local_solver,
            None,
            "supports v68 best_S metric aliases",
        ),
        _metric_row(
            "v68_best_local_solver_single_frame_rate",
            _first_float(best_solver, "single_frame_object_rate", "single_frame_object_rate_mean"),
            args.v68_local_solver,
            None,
            "supports v68 best_S metric aliases",
        ),
        _metric_row("v68_decision", v68_solver.get("decision"), args.v68_local_solver),
        _metric_row("v70_anchor_with_carrier_witness_rate", v70_metrics.get("phase1_anchor_with_carrier_witness_rate"), args.v70_final),
        _metric_row("v70_true_material_best_SF50", v70_metrics.get("phase2_best_SF50"), args.v70_final),
        _metric_row("v70_true_material_real_minus_no_temporal_SF50", v70_metrics.get("phase2_real_minus_no_temporal_SF50"), args.v70_final),
        _metric_row("v70_object_capsule_best_SF50", v70_metrics.get("phase4_best_SF50"), args.v70_final),
        _metric_row("v70_underseg_false_bridge_rate", v70_metrics.get("phase4_best_underseg_false_bridge_rate"), args.v70_final),
        _metric_row("v70_decision", v70.get("decision"), args.v70_final),
        _metric_row("metric_selfcheck_pass", metric_selfcheck_pass, args.v66_metric_lock, metric_selfcheck_pass),
        _metric_row("no_gt_for_method_prediction_pass", no_gt_for_method_prediction_pass, "summary_gate_static_audit", no_gt_for_method_prediction_pass),
        _metric_row("can_enter_local2history_initial", False, args.v70_final, True),
        _metric_row("v70_casebook_supports_conclusion", bool((v70_casebook.get("gate") or {}).get("pass")), args.v70_casebook),
    ]
    value_by_metric = {row["metric"]: row["value"] for row in rows}
    raw_or_rep = max(
        _float(value_by_metric.get("v67_U2_representative_oracle_SF50")) or 0.0,
        _float(value_by_metric.get("v67_U3_raw_oracle_SF50")) or 0.0,
    )
    gate = {
        "all_inputs_present": True,
        "metric_selfcheck_pass": metric_selfcheck_pass,
        "no_gt_for_method_prediction_pass": no_gt_for_method_prediction_pass,
        "v67_raw_or_representative_oracle_SF50_ge_0p50": raw_or_rep >= 0.50,
        "v68_candidate_bank_oracle_SF50_ge_0p50": (_float(value_by_metric.get("v68_candidate_bank_oracle_SF50")) or 0.0) >= 0.50,
        "v68_DINO_edge_AUC_ge_0p80": (_float(value_by_metric.get("v68_DINO_edge_AUC")) or 0.0) >= 0.80,
        "v70_true_material_best_SF50_lt_0p10": (_float(value_by_metric.get("v70_true_material_best_SF50")) or 999.0) < 0.10,
        "v70_object_capsule_best_SF50_below_local_gate": (_float(value_by_metric.get("v70_object_capsule_best_SF50")) or 999.0) < 0.10,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    decision = "PASS_V71_FACT_LOCK" if gate["pass"] else "NO_GO_PHASE0_FACT_LOCK"
    code_rows = _code_audit_rows()
    _write_csv(output_root / "fact_metric_rows.csv", rows)
    _write_csv(output_root / "code_audit_rows.csv", code_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing)
    summary = {
        "phase": "v71_phase0_fact_lock",
        "decision": decision,
        "diagnostic_only": True,
        "gate": gate,
        "key_metrics": value_by_metric,
        "inputs": paths,
        "rows": {
            "fact_metric_rows_csv": _rel(output_root / "fact_metric_rows.csv"),
            "code_audit_rows_csv": _rel(output_root / "code_audit_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "Phase 0 only locks prior facts and no-GT boundaries; it is not a v71 method result.",
            "Null metric values are retained when a named field is absent from the current artifacts.",
            "GT-derived oracle metrics are diagnostic-only and are not method predictions.",
        ],
    }
    _write_json(output_root / "fact_lock_summary.json", summary)
    sha_rows = []
    for path in [output_root / "fact_lock_summary.json", output_root / "fact_metric_rows.csv", output_root / "code_audit_rows.csv", output_root / "missing_input_rows.csv"]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v71 Phase 0 fact and metric lock.")
    parser.add_argument("--output-root", default="outputs/audit/v71_phase0_fact_lock")
    parser.add_argument("--v66-metric-lock", default="outputs/audit/v66_phase0_metric_lock/metric_lock_summary.json")
    parser.add_argument("--v66-mv-ap", default="outputs/audit/v66_scene_mv_ap_probe5_full/mv_ap_summary.json")
    parser.add_argument("--v67-mask-universe", default="outputs/audit/v67_mask_universe/mask_universe_summary.json")
    parser.add_argument("--v68-candidate-bank", default="outputs/audit/v68_candidate_bank/candidate_bank_summary.json")
    parser.add_argument("--v68-edge-audit", default="outputs/audit/v68_edge_audit_dinov2/edge_audit_summary.json")
    parser.add_argument("--v68-local-solver", default="outputs/audit/v68_local_graph_solver/local_solver_summary.json")
    parser.add_argument("--v70-final", default="outputs/audit/v70_final_decision_continued/final_decision.json")
    parser.add_argument("--v70-casebook", default="outputs/audit/v70_casebook/casebook_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
