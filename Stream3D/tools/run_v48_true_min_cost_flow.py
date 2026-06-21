from __future__ import annotations

import argparse
from typing import Any

from stream4d_native.v47_common import ROOT, parse_float, parse_int, read_csv, write_csv, write_json
from stream4d_native.v48_true_min_cost_flow import evaluate_tracks, select_min_cost_circulation_edges


def _parse_csv_values(spec: str) -> list[str]:
    return [item.strip() for item in str(spec).split(",") if item.strip()]


def _parse_float_values(spec: str) -> list[float]:
    return [float(item.strip()) for item in str(spec).split(",") if item.strip()]


def _signature(row: dict[str, Any]) -> tuple[str, str, float, float, int]:
    return (
        str(row.get("variant")),
        str(row.get("score_key")),
        parse_float(row.get("min_score")),
        parse_float(row.get("max_visible_outside")),
        parse_int(row.get("min_visible_carriers")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v48 true sparse temporal min-cost flow.")
    parser.add_argument("--observation-root", default="outputs/audit/v47_observation_tables_metricfix")
    parser.add_argument("--edge-table", default="outputs/audit/v47_adjacent_edges_gap2_only/temporal_candidate_edge_table.csv")
    parser.add_argument("--edge-type-sets", default="B1_adjacent:adjacent;B2_adjacent_skip:adjacent,skip")
    parser.add_argument("--real-score-keys", default="A5_d4rt_semantic_confirmation,A4_d4rt_visible_veto")
    parser.add_argument("--control-score-keys", default="A8_no_temporal_control,A7_shuffled_D4RT,A0_bbox_overlap")
    parser.add_argument("--min-scores", default="0.30,0.70,0.97")
    parser.add_argument("--max-visible-outside-values", default="1.0")
    parser.add_argument("--min-visible-carrier-values", default="0")
    parser.add_argument("--respect-edge-accept-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--proxy-root", default="outputs/audit/v47_matching_flow_gap2_global_proxy")
    parser.add_argument("--output-root", default="outputs/audit/v48_true_flow")
    args = parser.parse_args()

    mask_rows = read_csv(ROOT / str(args.observation_root) / "mask_observation_table.csv")
    edge_rows = read_csv(ROOT / str(args.edge_table))
    score_keys = _parse_csv_values(args.real_score_keys) + _parse_csv_values(args.control_score_keys)
    min_scores = _parse_float_values(args.min_scores)
    max_visible_values = _parse_float_values(args.max_visible_outside_values)
    min_carrier_values = [int(value) for value in _parse_float_values(args.min_visible_carrier_values)]

    edge_type_sets: list[tuple[str, set[str]]] = []
    for item in str(args.edge_type_sets).split(";"):
        if not item.strip():
            continue
        name, values = item.split(":", 1)
        edge_type_sets.append((name.strip(), set(_parse_csv_values(values))))

    scan_rows: list[dict[str, Any]] = []
    selected_by_signature: dict[tuple[str, str, float, float, int], list[dict[str, Any]]] = {}
    track_by_signature: dict[tuple[str, str, float, float, int], list[dict[str, Any]]] = {}
    solver_rows: list[dict[str, Any]] = []
    for variant, edge_types in edge_type_sets:
        for score_key in score_keys:
            for min_score in min_scores:
                for max_visible in max_visible_values:
                    for min_carriers in min_carrier_values:
                        selected, solver_info = select_min_cost_circulation_edges(
                            edge_rows=edge_rows,
                            score_key=score_key,
                            min_score=float(min_score),
                            edge_types=edge_types,
                            max_visible_outside=float(max_visible),
                            min_visible_carriers=int(min_carriers),
                            respect_edge_accept_candidate=bool(args.respect_edge_accept_candidate),
                        )
                        evaluated = evaluate_tracks(mask_rows, selected)
                        row = {
                            "variant": variant,
                            "score_key": score_key,
                            "min_score": float(min_score),
                            "edge_types": ",".join(sorted(edge_types)),
                            "max_visible_outside": float(max_visible),
                            "min_visible_carriers": int(min_carriers),
                            **solver_info,
                            **evaluated["metrics"],
                        }
                        scan_rows.append(row)
                        sig = _signature(row)
                        selected_by_signature[sig] = selected
                        track_by_signature[sig] = evaluated["track_rows"]
                        solver_rows.append(
                            {
                                "variant": variant,
                                "score_key": score_key,
                                "min_score": float(min_score),
                                "edge_types": ",".join(sorted(edge_types)),
                                **solver_info,
                            }
                        )

    scan_rows.sort(key=lambda row: (parse_float(row.get("ARI")), parse_float(row.get("purity"))), reverse=True)
    real_keys = set(_parse_csv_values(args.real_score_keys))
    real_rows = [row for row in scan_rows if row["score_key"] in real_keys]
    no_temporal_rows = [row for row in scan_rows if row["score_key"] == "A8_no_temporal_control"]
    shuffled_rows = [row for row in scan_rows if row["score_key"] == "A7_shuffled_D4RT"]
    mask_only_rows = [row for row in scan_rows if row["score_key"] == "A0_bbox_overlap"]
    best_real = real_rows[0] if real_rows else {}
    best_no_temporal = no_temporal_rows[0] if no_temporal_rows else {}
    best_shuffled = shuffled_rows[0] if shuffled_rows else {}
    best_mask_only = mask_only_rows[0] if mask_only_rows else {}

    proxy = {}
    proxy_path = ROOT / str(args.proxy_root) / "matching_flow_summary.json"
    if proxy_path.exists():
        import json

        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    proxy_best = proxy.get("best_real_row", {})
    delta_vs_proxy_ari = parse_float(best_real.get("ARI")) - parse_float(proxy_best.get("ARI"))
    delta_vs_proxy_completeness = parse_float(best_real.get("completeness")) - parse_float(proxy_best.get("completeness"))
    real_minus_shuffled = parse_float(best_real.get("ARI")) - parse_float(best_shuffled.get("ARI"))
    real_minus_no_temporal = parse_float(best_real.get("ARI")) - parse_float(best_no_temporal.get("ARI"))
    real_minus_mask_only = parse_float(best_real.get("ARI")) - parse_float(best_mask_only.get("ARI"))
    gate = {
        "beats_proxy_ARI_pass": delta_vs_proxy_ari >= 0.03,
        "beats_proxy_completeness_pass": delta_vs_proxy_completeness >= 0.05,
        "purity_drop_vs_proxy_pass": (parse_float(proxy_best.get("purity")) - parse_float(best_real.get("purity"))) <= 0.01,
        "partial_ARI_pass": parse_float(best_real.get("ARI")) >= 0.45,
        "partial_purity_pass": parse_float(best_real.get("purity")) >= 0.87,
        "partial_completeness_pass": parse_float(best_real.get("completeness")) >= 0.50,
        "real_minus_shuffled_pass": real_minus_shuffled >= 0.20,
        "real_minus_no_temporal_pass": real_minus_no_temporal >= 0.10,
        "real_minus_mask_only_pass": real_minus_mask_only >= 0.10,
    }
    gate["pass"] = bool(
        gate["beats_proxy_ARI_pass"]
        and gate["beats_proxy_completeness_pass"]
        and gate["purity_drop_vs_proxy_pass"]
        and gate["partial_ARI_pass"]
        and gate["partial_purity_pass"]
        and gate["partial_completeness_pass"]
    )
    summary = {
        "phase": "v48_true_min_cost_flow",
        "solver_type": "networkx_min_cost_circulation",
        "solver_note": "Sparse min-cost circulation over temporal edge candidates; selected edges are negative-cost cycles under node capacity 1, not per-edge greedy.",
        "observation_root": str(ROOT / str(args.observation_root)),
        "edge_table": str(ROOT / str(args.edge_table)),
        "rows": len(scan_rows),
        "best_real_row": best_real,
        "best_proxy_row": proxy_best,
        "best_no_temporal_row": best_no_temporal,
        "best_shuffled_row": best_shuffled,
        "best_mask_only_row": best_mask_only,
        "delta_vs_proxy_ARI": delta_vs_proxy_ari,
        "delta_vs_proxy_completeness": delta_vs_proxy_completeness,
        "real_minus_shuffled_ARI": real_minus_shuffled,
        "real_minus_no_temporal_ARI": real_minus_no_temporal,
        "real_minus_mask_only_ARI": real_minus_mask_only,
        "gate": gate,
        "failure_label": None if gate["pass"] else "NO_GO_FLOW",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }

    out = ROOT / str(args.output_root)
    write_csv(out / "true_flow_scan_rows.csv", scan_rows)
    write_csv(out / "true_flow_solver_rows.csv", solver_rows)
    write_json(out / "true_flow_summary.json", summary)
    for name, row in [
        ("best_real", best_real),
        ("best_no_temporal", best_no_temporal),
        ("best_shuffled", best_shuffled),
        ("best_mask_only", best_mask_only),
    ]:
        if row:
            sig = _signature(row)
            write_csv(out / f"true_flow_{name}_selected_edges.csv", selected_by_signature.get(sig, []))
            write_csv(out / f"true_flow_{name}_track_rows.csv", track_by_signature.get(sig, []))
    print({"summary": str(out / "true_flow_summary.json"), "gate": gate, "failure_label": summary["failure_label"]})


if __name__ == "__main__":
    main()

