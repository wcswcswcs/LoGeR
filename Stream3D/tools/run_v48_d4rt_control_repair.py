from __future__ import annotations

import argparse
from typing import Any

from stream4d_native.v47_common import ROOT, parse_bool, parse_float, read_csv, write_csv, write_json
from stream4d_native.v48_data_contract import utc_now


def _key(row: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in keys)


def _bool_text(value: Any) -> str:
    return "True" if parse_bool(value) else "False"


def _mdl_rows(rows: list[dict[str, Any]], raw_ari: float, raw_completeness: float) -> list[dict[str, Any]]:
    keys = ["threshold", "min_d4rt", "max_visible_outside", "allow_frame_overlap"]
    by_variant: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant")), {})[_key(row, keys)] = row
    out: list[dict[str, Any]] = []
    for key, real in by_variant.get("M2_d4rt_confirmed_complete_link", {}).items():
        no_temporal = by_variant.get("M3_no_temporal_confirmed_control", {}).get(key)
        shuffled = by_variant.get("M4_shuffled_d4rt_confirmed_control", {}).get(key)
        if not no_temporal or not shuffled:
            continue
        out.append(
            _repair_row(
                family="MDL_complete_link",
                variant="M2_d4rt_confirmed_complete_link",
                real=real,
                no_temporal=no_temporal,
                shuffled=shuffled,
                raw_ari=raw_ari,
                raw_completeness=raw_completeness,
                param_keys=keys,
            )
        )
    return out


def _constrained_rows(rows: list[dict[str, Any]], raw_ari: float, raw_completeness: float) -> list[dict[str, Any]]:
    keys = [
        "score_threshold",
        "min_pair_edge_count",
        "max_visible_outside",
        "min_visible_carriers",
        "max_selected_pairs",
        "enforce_cluster_frame_exclusion",
        "forbid_pair_same_frame_conflict",
    ]
    by_key: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault(str(row.get("score_key")), {})[_key(row, keys)] = row
    out: list[dict[str, Any]] = []
    for score_key in ["A5_d4rt_semantic_confirmation", "A4_d4rt_visible_veto"]:
        for key, real in by_key.get(score_key, {}).items():
            no_temporal = by_key.get("A8_no_temporal_control", {}).get(key)
            shuffled = by_key.get("A7_shuffled_D4RT", {}).get(key)
            if not no_temporal or not shuffled:
                continue
            out.append(
                _repair_row(
                    family="constrained_component_merge",
                    variant=score_key,
                    real=real,
                    no_temporal=no_temporal,
                    shuffled=shuffled,
                    raw_ari=raw_ari,
                    raw_completeness=raw_completeness,
                    param_keys=keys,
                )
            )
    return out


def _repair_row(
    *,
    family: str,
    variant: str,
    real: dict[str, Any],
    no_temporal: dict[str, Any],
    shuffled: dict[str, Any],
    raw_ari: float,
    raw_completeness: float,
    param_keys: list[str],
) -> dict[str, Any]:
    real_ari = parse_float(real.get("ARI"))
    real_purity = parse_float(real.get("purity"))
    real_completeness = parse_float(real.get("completeness"))
    no_temporal_ari = parse_float(no_temporal.get("ARI"))
    shuffled_ari = parse_float(shuffled.get("ARI"))
    delta_ari = real_ari - raw_ari
    delta_completeness = real_completeness - raw_completeness
    real_minus_no_temporal = real_ari - no_temporal_ari
    real_minus_shuffled = real_ari - shuffled_ari
    gate = {
        "delta_ARI_pass": delta_ari >= 0.04,
        "delta_completeness_pass": delta_completeness >= 0.08,
        "purity_pass": real_purity >= 0.875,
        "real_minus_no_temporal_pass": real_minus_no_temporal >= 0.10,
        "real_minus_shuffled_d4rt_pass": real_minus_shuffled >= 0.20,
        "stage1_ARI_pass": real_ari >= 0.485,
        "stage1_completeness_pass": real_completeness >= 0.555,
    }
    gate["partial_repair_pass"] = bool(
        gate["delta_ARI_pass"]
        and gate["delta_completeness_pass"]
        and gate["purity_pass"]
        and gate["real_minus_no_temporal_pass"]
        and gate["real_minus_shuffled_d4rt_pass"]
    )
    row = {
        "family": family,
        "variant": variant,
        "ARI": real_ari,
        "purity": real_purity,
        "completeness": real_completeness,
        "delta_ARI_vs_raw": delta_ari,
        "delta_completeness_vs_raw": delta_completeness,
        "no_temporal_ARI_same_params": no_temporal_ari,
        "shuffled_d4rt_ARI_same_params": shuffled_ari,
        "real_minus_no_temporal_ARI_same_params": real_minus_no_temporal,
        "real_minus_shuffled_d4rt_ARI_same_params": real_minus_shuffled,
        "candidate_pair_count_after_filter": real.get("candidate_pair_count_after_filter"),
        "merge_count": real.get("merge_count") or real.get("selected_pair_count"),
        "selected_pair_count": real.get("selected_pair_count") or real.get("merge_count"),
        "scene0081_ARI": real.get("scene0081_ARI"),
        "scene0011_purity": real.get("scene0011_purity"),
        "scene0050_purity": real.get("scene0050_purity"),
        "scene0591_completeness": real.get("scene0591_completeness"),
        "mean_predictions_per_scene": real.get("mean_predictions_per_scene"),
        "temporal_span_mean": real.get("temporal_span_mean"),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        **{key: _bool_text(real.get(key)) if key.startswith(("allow_", "enforce_", "forbid_")) else real.get(key) for key in param_keys},
        **{f"gate_{name}": value for name, value in gate.items()},
    }
    row["gate_pass"] = gate["partial_repair_pass"]
    return row


def _rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(parse_bool(row.get("gate_pass"))),
        int(parse_bool(row.get("gate_purity_pass"))),
        int(parse_bool(row.get("gate_real_minus_no_temporal_pass"))),
        int(parse_bool(row.get("gate_real_minus_shuffled_d4rt_pass"))),
        parse_float(row.get("ARI")),
        parse_float(row.get("completeness")),
        parse_float(row.get("real_minus_no_temporal_ARI_same_params")),
        parse_float(row.get("real_minus_shuffled_d4rt_ARI_same_params")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v48 D4RT-required component completion with paired controls.")
    parser.add_argument("--mdl-scan", default="outputs/audit/v47_carrier_component_mdl_semantic_continued19/carrier_component_mdl_semantic_scan_rows.csv")
    parser.add_argument("--constrained-scan", default="outputs/audit/v47_component_constrained_merge_union32_gap2_narrow/component_constrained_merge_scan_rows.csv")
    parser.add_argument("--raw-ari", type=float, default=0.4247026471350924)
    parser.add_argument("--raw-completeness", type=float, default=0.41711229946524064)
    parser.add_argument("--output-root", default="outputs/audit/v48_d4rt_control_repair")
    args = parser.parse_args()

    mdl_scan = read_csv(ROOT / str(args.mdl_scan))
    constrained_scan = read_csv(ROOT / str(args.constrained_scan))
    rows = _mdl_rows(mdl_scan, args.raw_ari, args.raw_completeness)
    rows.extend(_constrained_rows(constrained_scan, args.raw_ari, args.raw_completeness))
    rows.sort(key=_rank, reverse=True)
    best = rows[0] if rows else {}
    passing = [row for row in rows if parse_bool(row.get("gate_pass"))]
    summary = {
        "phase": "v48_d4rt_control_repair",
        "created_at": utc_now(),
        "row_count": len(rows),
        "passing_row_count": len(passing),
        "best_row": best,
        "gate": {
            "pass": bool(passing),
            "failure_label": None if passing else "NO_GO_D4RT_CONTROL_REPAIR",
            "best_variant": best.get("variant"),
            "best_family": best.get("family"),
            "best_ARI": best.get("ARI"),
            "best_purity": best.get("purity"),
            "best_completeness": best.get("completeness"),
            "best_real_minus_no_temporal_ARI_same_params": best.get("real_minus_no_temporal_ARI_same_params"),
            "best_real_minus_shuffled_d4rt_ARI_same_params": best.get("real_minus_shuffled_d4rt_ARI_same_params"),
        },
        "thresholds": {
            "partial_delta_ARI_vs_raw": 0.04,
            "partial_delta_completeness_vs_raw": 0.08,
            "purity": 0.875,
            "real_minus_no_temporal_ARI_same_params": 0.10,
            "real_minus_shuffled_d4rt_ARI_same_params": 0.20,
            "stage1_ARI": 0.485,
            "stage1_completeness": 0.555,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    out = ROOT / str(args.output_root)
    write_json(out / "d4rt_control_repair_summary.json", summary)
    write_csv(out / "d4rt_control_repair_rows.csv", rows)
    write_csv(out / "d4rt_control_repair_passing_rows.csv", passing)
    print({"summary": str(out / "d4rt_control_repair_summary.json"), "gate": summary["gate"]})


if __name__ == "__main__":
    main()
