from __future__ import annotations

import argparse
import importlib.util
from typing import Any

from stream4d_native.v47_common import write_csv, write_json
from stream4d_native.v48_data_contract import load_optional_json, project_path, rel, utc_now


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(value: Any, base: Any) -> float | None:
    left = _num(value)
    right = _num(base)
    return None if left is None or right is None else float(left - right)


def _row(
    *,
    variant: str,
    source: str,
    row: dict[str, Any],
    raw: dict[str, Any],
    real_minus_shuffled: Any = None,
    real_minus_no_temporal: Any = None,
    status: str = "ok",
    solver_type: str = "imported_summary",
    note: str = "",
) -> dict[str, Any]:
    ari = row.get("ARI")
    purity = row.get("purity")
    completeness = row.get("completeness")
    delta_ari = _delta(ari, raw.get("ARI"))
    delta_completeness = _delta(completeness, raw.get("completeness"))
    purity_drop = _delta(raw.get("purity"), purity)
    gate = {
        "delta_ARI_pass": delta_ari is not None and delta_ari >= 0.04,
        "delta_completeness_pass": delta_completeness is not None and delta_completeness >= 0.08,
        "purity_pass": _num(purity) is not None and _num(purity) >= 0.875,
        "purity_drop_pass": purity_drop is not None and purity_drop <= 0.02,
        "real_minus_shuffled_pass": _num(real_minus_shuffled) is not None and _num(real_minus_shuffled) >= 0.20,
        "real_minus_no_temporal_pass": _num(real_minus_no_temporal) is not None and _num(real_minus_no_temporal) >= 0.10,
    }
    gate["partial_object_completion_pass"] = bool(
        gate["delta_ARI_pass"]
        and gate["delta_completeness_pass"]
        and gate["purity_pass"]
        and gate["real_minus_shuffled_pass"]
        and gate["real_minus_no_temporal_pass"]
    )
    return {
        "variant": variant,
        "status": status,
        "solver_type": solver_type,
        "source": rel(source),
        "selected_object_count": row.get("cluster_count") or row.get("component_count"),
        "mean_components_per_object": None,
        "shared_observation_count": None,
        "uncovered_component_ratio": None,
        "duplicate_component_ratio": None,
        "conflict_rate": None,
        "unknown_component_ratio": None,
        "merge_count": row.get("merge_count") or row.get("selected_pair_count"),
        "split_count": row.get("split_count"),
        "energy_total": None,
        "energy_uncovered": None,
        "energy_duplicate": None,
        "energy_conflict": None,
        "energy_support": None,
        "ARI": ari,
        "purity": purity,
        "completeness": completeness,
        "delta_ARI_vs_raw": delta_ari,
        "delta_completeness_vs_raw": delta_completeness,
        "purity_drop_vs_raw": purity_drop,
        "temporal_span_mean": row.get("temporal_span_mean"),
        "scene0081_ARI": row.get("scene0081_ARI"),
        "scene0591_completeness": row.get("scene0591_completeness"),
        "real_minus_shuffled_ARI": real_minus_shuffled,
        "real_minus_no_temporal_ARI": real_minus_no_temporal,
        "birth_from_d4rt_tube_count": row.get("birth_from_d4rt_tube_count"),
        "maskless_object_count": row.get("maskless_object_count"),
        "uses_gt_for_prediction": row.get("uses_gt_for_prediction", False),
        "uses_gt_for_diagnostic_labels": row.get("uses_gt_for_diagnostic_labels", True),
        "gate": gate,
        "gate_pass": gate["partial_object_completion_pass"],
        "note": note,
    }


def _solver_availability() -> dict[str, Any]:
    return {
        "ortools_available": importlib.util.find_spec("ortools") is not None,
        "pulp_available": importlib.util.find_spec("pulp") is not None,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    primitive = load_optional_json(args.primitive_root + "/primitive_audit_summary.json")
    carrier_mdl = load_optional_json(args.carrier_mdl_root + "/carrier_component_mdl_semantic_summary.json")
    constrained = load_optional_json(args.component_constrained_root + "/component_constrained_merge_summary.json")
    coarse = load_optional_json(args.coarse_mdl_root + "/carrier_coarse_to_fine_mdl_summary.json")
    raw = dict(primitive.get("primary_primitive") or {})
    raw["ARI"] = raw.get("ARI")
    raw["purity"] = raw.get("primitive_purity_mean")
    raw["completeness"] = raw.get("primitive_completeness_mean")
    raw["component_count"] = raw.get("primitive_count")
    raw["uses_gt_for_prediction"] = False
    raw["uses_gt_for_diagnostic_labels"] = True

    rows = [
        _row(
            variant="A0_raw_carrier_component",
            source=args.primitive_root + "/primitive_audit_summary.json",
            row=raw,
            raw=raw,
            real_minus_shuffled=raw.get("real_minus_shuffled_ARI"),
            status="baseline",
            solver_type="raw_component",
            note="Phase 1 primary primitive baseline.",
        ),
        _row(
            variant="A5_greedy_MDL_semantic_temporal_conflict",
            source=args.carrier_mdl_root + "/carrier_component_mdl_semantic_summary.json",
            row=carrier_mdl.get("best_real_row", {}),
            raw=raw,
            real_minus_shuffled=carrier_mdl.get("best_real_minus_best_shuffled_d4rt_ARI"),
            real_minus_no_temporal=carrier_mdl.get("best_real_minus_best_no_temporal_ARI"),
            solver_type="greedy_MDL_local_search",
            note="Imported v47 carrier-component MDL semantic-complete-link summary.",
        ),
        _row(
            variant="A8_no_temporal_control",
            source=args.carrier_mdl_root + "/carrier_component_mdl_semantic_summary.json",
            row=carrier_mdl.get("best_no_temporal_row", {}),
            raw=raw,
            solver_type="control",
            status="control",
            note="No-temporal control from the same carrier MDL scan.",
        ),
        _row(
            variant="A7_shuffled_D4RT_control",
            source=args.carrier_mdl_root + "/carrier_component_mdl_semantic_summary.json",
            row=carrier_mdl.get("best_shuffled_d4rt_row", {}),
            raw=raw,
            solver_type="control",
            status="control",
            note="Shuffled-D4RT control from the same carrier MDL scan.",
        ),
        _row(
            variant="A9_semantic_shuffled_control",
            source=args.carrier_mdl_root + "/carrier_component_mdl_semantic_summary.json",
            row=carrier_mdl.get("best_shuffled_semantic_row", {}),
            raw=raw,
            solver_type="control",
            status="control",
            note="Shuffled semantic control from the same carrier MDL scan.",
        ),
        _row(
            variant="A6_constrained_merge_proxy_not_exact_ILP",
            source=args.component_constrained_root + "/component_constrained_merge_summary.json",
            row=constrained.get("best_real_row", {}),
            raw=raw,
            real_minus_shuffled=constrained.get("best_real_minus_best_shuffled_ARI"),
            real_minus_no_temporal=constrained.get("best_real_minus_best_no_temporal_ARI"),
            solver_type="greedy_constrained_pair_selection_proxy",
            note="Existing constrained component merge proxy; not exact ILP/CP-SAT.",
        ),
        _row(
            variant="A6_no_temporal_proxy_best_but_invalid_purity",
            source=args.component_constrained_root + "/component_constrained_merge_summary.json",
            row=constrained.get("best_no_temporal_row", {}),
            raw=raw,
            solver_type="control",
            status="control",
            note="Best constrained proxy row is a no-temporal control and purity is below gate.",
        ),
        _row(
            variant="A_coarse_to_fine_repair",
            source=args.coarse_mdl_root + "/carrier_coarse_to_fine_mdl_summary.json",
            row=coarse.get("best_real_row", {}),
            raw=raw,
            real_minus_shuffled=coarse.get("best_real_minus_best_shuffled_semantic_ARI"),
            solver_type="coarse_to_fine_repair",
            note="Plan-directed repair attempt to improve completeness using coarse carrier evidence.",
        ),
    ]

    availability = _solver_availability()
    exact_status = "not_run_solver_unavailable"
    if availability["ortools_available"] or availability["pulp_available"]:
        exact_status = "not_run_no_v48_exact_model_implemented"
    exact_row = {
        "variant": "A6_small_scene_exact_ILP",
        "status": exact_status,
        "solver_type": "exact_ILP_diagnostic",
        "source": "",
        "ARI": None,
        "purity": None,
        "completeness": None,
        "gate_pass": False,
        "note": "OR-Tools/PuLP exact solver not available in current environment; no exact ILP result is claimed.",
    }
    rows.append(exact_row)

    best_real = max(
        [row for row in rows if row.get("status") not in {"control", "baseline", exact_status} and row.get("ARI") is not None],
        key=lambda row: float(row.get("ARI") or -1.0),
        default=None,
    )
    any_pass = any(bool(row.get("gate_pass")) for row in rows if row.get("status") != "control")
    gate = {
        "pass": any_pass,
        "best_real_variant": None if best_real is None else best_real.get("variant"),
        "best_real_ARI": None if best_real is None else best_real.get("ARI"),
        "raw_ARI": raw.get("ARI"),
        "exact_solver_available": bool(availability["ortools_available"] or availability["pulp_available"]),
        "ortools_available": availability["ortools_available"],
        "pulp_available": availability["pulp_available"],
        "exact_ILP_claimed": False,
        "failure_label": None if any_pass else "NO_GO_COMPONENT_COMPLETION",
    }
    return {
        "phase": "v48_object_explanation_ilp",
        "created_at": utc_now(),
        "scheme": "A",
        "rows": rows,
        "gate": gate,
        "thresholds": {
            "delta_ARI_vs_raw": 0.04,
            "delta_completeness_vs_raw": 0.08,
            "purity_min": 0.875,
            "real_minus_shuffled_ARI": 0.20,
            "real_minus_no_temporal_ARI": 0.10,
        },
        "repair_decision": "A5/A6-proxy do not pass; per plan continue to Scheme B/C/D/F rather than tuning semantic thresholds.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v48 Scheme A object explanation/ILP audit.")
    parser.add_argument("--primitive-root", default="outputs/audit/v48_primitive_audit")
    parser.add_argument("--carrier-mdl-root", default="outputs/audit/v47_carrier_component_mdl_semantic_continued19")
    parser.add_argument("--component-constrained-root", default="outputs/audit/v47_component_constrained_merge_union32_gap2_narrow")
    parser.add_argument("--coarse-mdl-root", default="outputs/audit/v47_carrier_coarse_to_fine_mdl_continued20")
    parser.add_argument("--output-root", default="outputs/audit/v48_object_explanation_ilp")
    args = parser.parse_args()

    payload = build_summary(args)
    out = project_path(args.output_root)
    write_json(out / "object_explanation_summary.json", payload)
    write_csv(out / "object_explanation_rows.csv", payload["rows"])
    print({"summary": str(out / "object_explanation_summary.json"), "gate": payload["gate"]})


if __name__ == "__main__":
    main()

