"""Summarize Stream4D v16 decisive oracle/materialization diagnostics.

This script intentionally does not recompute AP. It collects already-produced
diagnostic JSON/evaluation outputs into one auditable matrix and evaluates the
v16 stop gates from the experiment plan.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_json(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data["_path"] = str(path)
    return data


def _fmt(value: Any, *, scale: float = 1.0, digits: int = 6) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value) * float(scale):.{digits}f}"
    except Exception:
        return str(value)


def _best(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row.get(key) or 0.0))


def _add_v13_candidate_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    data = _load_json(root, "outputs/audit/v13_candidate_attribution/candidate_oracle_matrix_probe5.json")
    for row in data.get("rows", []):
        primitive = str(row.get("method", "")).replace(" oracle", "")
        rows.append(
            {
                "layer": "candidate_oracle",
                "primitive": primitive,
                "variant": "v13_single_candidate",
                "k": "",
                "ap": row.get("ap"),
                "ap50": row.get("ap50"),
                "ap25": row.get("ap25"),
                "support_pre_ratio": row.get("pre_points_ratio"),
                "selected_union_pre_ratio": "",
                "mean_best_iou": row.get("mean_best_iou_per_gt"),
                "exported_pre_ratio": "",
                "purity": "",
                "contamination": "",
                "uses_gt_for_diagnostic": True,
                "uses_gt_for_prediction": False,
                "forbidden_for_method_table": True,
                "source_file": data.get("_path"),
            }
        )


def _add_v14_atom_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    data = _load_json(root, "outputs/audit/v14_phase2_summary/phase2_atom_repair_matrix_probe5.json")
    keep = {"bank16 target A3", "bank16 target A4", "bank16 target A4 minpts5"}
    for row in data.get("rows", []):
        label = str(row.get("label", ""))
        if label not in keep:
            continue
        rows.append(
            {
                "layer": "candidate_oracle",
                "primitive": "C_surfel_atom",
                "variant": label,
                "k": "",
                "ap": row.get("oracle_ap"),
                "ap50": row.get("oracle_ap50"),
                "ap25": row.get("oracle_ap25"),
                "candidate_ap": row.get("candidate_ap"),
                "candidate_ap50": row.get("candidate_ap50"),
                "candidate_ap25": row.get("candidate_ap25"),
                "support_pre_ratio": row.get("pre_points_ratio"),
                "atom_known_support_ratio": row.get("atom_known_support_ratio"),
                "selected_union_pre_ratio": "",
                "mean_best_iou": row.get("mean_best_iou_per_gt"),
                "exported_pre_ratio": "",
                "purity": "",
                "contamination": "",
                "uses_gt_for_diagnostic": True,
                "uses_gt_for_prediction": False,
                "forbidden_for_method_table": True,
                "source_file": data.get("_path"),
            }
        )


def _add_union_rows(root: Path, rows: list[dict[str, Any]], rel: str, primitive: str, variant: str) -> None:
    data = _load_json(root, rel)
    summary = data.get("summary", {})
    for row in data.get("rows", []):
        rows.append(
            {
                "layer": "slot_oracle",
                "primitive": primitive,
                "variant": variant,
                "k": row.get("k"),
                "ap": row.get("ap"),
                "ap50": row.get("ap50"),
                "ap25": row.get("ap25"),
                "support_pre_ratio": row.get("candidate_support_pre_ratio"),
                "selected_union_pre_ratio": row.get("selected_union_pre_ratio"),
                "mean_best_iou": row.get("mean_best_iou"),
                "mean_selected_count": row.get("mean_selected_count"),
                "exported_pre_ratio": "",
                "purity": "",
                "contamination": "",
                "uses_gt_for_diagnostic": bool(summary.get("uses_gt_for_diagnostic", True)),
                "uses_gt_for_prediction": bool(summary.get("uses_gt_for_prediction", False)),
                "gt_selected_output": bool(summary.get("gt_selected_output", True)),
                "forbidden_for_method_table": bool(summary.get("forbidden_for_method_table", True)),
                "source_file": data.get("_path"),
            }
        )


def _add_region_materialization_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    for rel, primitive, variant in [
        ("outputs/audit/v15_phase2/r0_component_region_probe5.json", "C_new_measurement_region", "R0 component"),
        ("outputs/audit/v15_phase2/r0b_component_region_r010_probe5.json", "C_new_measurement_region", "R0b component radius0.10"),
        ("outputs/audit/v15_phase2/r1_seed_voronoi_region_probe5.json", "C_new_measurement_region", "R1 seed_voronoi"),
        ("outputs/audit/v15_phase2/r2_boundary_core_region_probe5.json", "C_new_measurement_region", "R2 boundary_core"),
    ]:
        data = _load_json(root, rel)
        summary = data.get("summary", {})
        metric = summary.get("candidate_metric", {})
        mean = summary.get("numeric_mean", {})
        rows.append(
            {
                "layer": "materialization_direct",
                "primitive": primitive,
                "variant": variant,
                "k": "",
                "ap": metric.get("ap"),
                "ap50": metric.get("ap50"),
                "ap25": metric.get("ap25"),
                "support_pre_ratio": mean.get("exported_pre_ratio"),
                "selected_union_pre_ratio": "",
                "mean_best_iou": mean.get("best_region_iou_per_gt_mean"),
                "exported_pre_ratio": mean.get("exported_pre_ratio"),
                "purity": mean.get("region_purity_area_weighted"),
                "contamination": mean.get("cross_object_contamination_ratio"),
                "export_nn_hit_rate": mean.get("export_nn_hit_rate"),
                "region_completeness_mean": mean.get("region_completeness_mean"),
                "uses_gt_for_diagnostic": bool(summary.get("uses_gt_for_diagnostic", False)),
                "uses_gt_for_prediction": bool(summary.get("uses_gt_for_prediction", False)),
                "forbidden_for_method_table": True,
                "source_file": data.get("_path"),
            }
        )


def _write_outputs(prefix: Path, payload: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    rows = payload["rows"]
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = payload["summary"]
    lines = [
        "# Stream4D v16 Decisive Diagnostic Matrix",
        "",
        "All oracle rows are GT-read diagnostic upper bounds and are forbidden for method tables.",
        "",
        "## Stop Gate",
        "",
        f"- stop_before_solver: `{summary.get('stop_before_solver')}`",
        f"- stop_reason: `{summary.get('stop_reason')}`",
        f"- official broad slot gate pass: `{summary.get('official_broad_slot_gate_pass')}`",
        f"- stress broad slot gate pass: `{summary.get('stress_broad_slot_gate_pass')}`",
        f"- measurement bank gate pass: `{summary.get('measurement_bank_gate_pass')}`",
        "",
        "## Key Rows",
        "",
        "| layer | primitive | variant | K | AP/AP50/AP25 | support% | selected union% | best IoU | notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    key_names = [
        "best_official_broad_slot",
        "best_stress_broad_slot",
        "best_non_broad_slot",
        "best_materialization_direct",
    ]
    for name in key_names:
        row = summary.get(name) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("layer", "")),
                    str(row.get("primitive", "")),
                    str(row.get("variant", "")),
                    str(row.get("k", "")),
                    f"{_fmt(row.get('ap'))}/{_fmt(row.get('ap50'))}/{_fmt(row.get('ap25'))}",
                    _fmt(row.get("support_pre_ratio"), scale=100.0),
                    _fmt(row.get("selected_union_pre_ratio"), scale=100.0),
                    _fmt(row.get("mean_best_iou")),
                    name,
                ]
            )
            + " |"
        )
    bank = payload.get("measurement_bank", {}).get("aggregate", {}).get("numeric_mean", {})
    lines.extend(
        [
            "",
            "## Phase 2 Measurement Bank",
            "",
            f"- uv_in01_rate: `{_fmt(bank.get('uv_in01_rate'))}`",
            f"- cycle_uv_error_p90: `{_fmt(bank.get('cycle_uv_error_p90'))}`",
            f"- mean_positive_observations_per_surfel: `{_fmt(bank.get('mean_positive_observations_per_surfel'))}`",
            f"- unobserved_surfel_ratio: `{_fmt(bank.get('unobserved_surfel_ratio'))}`",
            "",
            "## Full Matrix",
            "",
            "| layer | primitive | variant | K | AP/AP50/AP25 | support% | selected union% | exported pre% | purity | contamination |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("layer", "")),
                    str(row.get("primitive", "")),
                    str(row.get("variant", "")),
                    str(row.get("k", "")),
                    f"{_fmt(row.get('ap'))}/{_fmt(row.get('ap50'))}/{_fmt(row.get('ap25'))}",
                    _fmt(row.get("support_pre_ratio"), scale=100.0),
                    _fmt(row.get("selected_union_pre_ratio"), scale=100.0),
                    _fmt(row.get("exported_pre_ratio"), scale=100.0),
                    _fmt(row.get("purity")),
                    _fmt(row.get("contamination")),
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-prefix", default="outputs/audit/v16_phase1/three_layer_oracle_matrix_probe5")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows: list[dict[str, Any]] = []
    _add_v13_candidate_rows(root, rows)
    _add_v14_atom_rows(root, rows)

    for rel, primitive, variant in [
        ("outputs/audit/v16_phase1/c_mask_union_oracle_probe5.json", "C_mask", "v16 official K2/4/8"),
        ("outputs/audit/v16_phase1/c_hybrid_union_oracle_probe5.json", "C_hybrid", "v16 official K2/4/8"),
        ("outputs/audit/v16_phase1/c_regionlet_union_oracle_probe5.json", "C_regionlet", "v16 official K2/4/8"),
        ("outputs/audit/v16_phase1/c_hybrid_union_stress_probe5.json", "C_hybrid", "v16 stress K16/32"),
        ("outputs/audit/v16_phase1/c_hybrid_union_min50_probe5.json", "C_hybrid", "v16 min_region_size50"),
        ("outputs/audit/v15_phase1/a3t16_union_oracle_probe5.json", "C_surfel_atom", "v15 A3 target atom union"),
        ("outputs/audit/v15_phase1/a4t16_union_oracle_probe5.json", "C_surfel_atom", "v15 A4 target atom union"),
        ("outputs/audit/v15_phase1/a4t16mp5_union_oracle_probe5.json", "C_surfel_atom", "v15 A4 target minpts5 union"),
        ("outputs/audit/v15_phase2/r0_component_region_union_oracle_probe5.json", "C_new_measurement_region", "R0 component union"),
        ("outputs/audit/v15_phase2/r1_seed_voronoi_region_union_oracle_probe5.json", "C_new_measurement_region", "R1 seed_voronoi union"),
        ("outputs/audit/v15_phase2/r2_boundary_core_region_union_oracle_probe5.json", "C_new_measurement_region", "R2 boundary_core union"),
    ]:
        _add_union_rows(root, rows, rel, primitive, variant)
    _add_region_materialization_rows(root, rows)

    thresholds = {
        "broad_support_pre_ratio": 0.25,
        "slot_ap50": 0.60,
        "slot_ap25": 0.80,
        "materialization_exported_pre_ratio": 0.15,
        "measurement_uv_in01_rate": 0.95,
        "measurement_cycle_uv_error_p90": 5.0,
        "measurement_mean_positive_observations_per_surfel": 2.5,
        "measurement_unobserved_surfel_ratio": 0.05,
    }
    official_slots = [
        row
        for row in rows
        if row.get("layer") == "slot_oracle"
        and "v16 official" in str(row.get("variant", ""))
        and float(row.get("support_pre_ratio") or 0.0) >= thresholds["broad_support_pre_ratio"]
    ]
    stress_slots = [
        row
        for row in rows
        if row.get("layer") == "slot_oracle"
        and row.get("primitive") == "C_hybrid"
        and float(row.get("support_pre_ratio") or 0.0) >= thresholds["broad_support_pre_ratio"]
    ]
    non_broad_slots = [
        row
        for row in rows
        if row.get("layer") == "slot_oracle"
        and float(row.get("support_pre_ratio") or 0.0) < thresholds["broad_support_pre_ratio"]
    ]
    material_rows = [row for row in rows if row.get("layer") == "materialization_direct"]
    best_official = _best(official_slots, "ap50") or {}
    best_stress = _best(stress_slots, "ap50") or {}
    best_non_broad = _best(non_broad_slots, "ap50") or {}
    best_material = _best(material_rows, "exported_pre_ratio") or {}

    official_gate = bool(
        best_official
        and float(best_official.get("ap50") or 0.0) >= thresholds["slot_ap50"]
        and float(best_official.get("ap25") or 0.0) >= thresholds["slot_ap25"]
    )
    stress_gate = bool(
        best_stress
        and float(best_stress.get("ap50") or 0.0) >= thresholds["slot_ap50"]
        and float(best_stress.get("ap25") or 0.0) >= thresholds["slot_ap25"]
    )

    measurement_bank = _load_json(root, "outputs/audit/v16_phase2/measurement_bank_bank16_probe5.json")
    bank_mean = measurement_bank.get("aggregate", {}).get("numeric_mean", {})
    bank_gate = bool(
        float(bank_mean.get("uv_in01_rate") or 0.0) >= thresholds["measurement_uv_in01_rate"]
        and float(bank_mean.get("cycle_uv_error_p90") or 1e9) <= thresholds["measurement_cycle_uv_error_p90"]
        and float(bank_mean.get("mean_positive_observations_per_surfel") or 0.0)
        >= thresholds["measurement_mean_positive_observations_per_surfel"]
        and float(bank_mean.get("unobserved_surfel_ratio") or 1.0) <= thresholds["measurement_unobserved_surfel_ratio"]
    )

    geometry = _load_json(root, "outputs/audit/v16_phase6/d4rt_sim3_residual_probe5.json")
    stop_reason = (
        "official broad-support slot oracle misses AP25>=0.80; C_hybrid stress K16/K32 also misses AP25>=0.80"
    )
    if official_gate and not bank_gate:
        stop_reason = "measurement bank gate misses mean_positive_observations_per_surfel>=2.5"
    if official_gate and bank_gate:
        stop_reason = "no v16 stop gate triggered by oracle/bank summary"

    payload = {
        "summary": {
            "thresholds": thresholds,
            "best_official_broad_slot": best_official,
            "best_stress_broad_slot": best_stress,
            "best_non_broad_slot": best_non_broad,
            "best_materialization_direct": best_material,
            "official_broad_slot_gate_pass": official_gate,
            "stress_broad_slot_gate_pass": stress_gate,
            "measurement_bank_gate_pass": bank_gate,
            "stop_before_solver": bool(not official_gate),
            "stop_reason": stop_reason,
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
            "is_method_result": False,
            "is_diagnostic_only": True,
        },
        "rows": rows,
        "measurement_bank": measurement_bank,
        "geometry": geometry,
    }
    _write_outputs(root / args.output_prefix, payload)
    print(json.dumps(_json_safe(payload["summary"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
