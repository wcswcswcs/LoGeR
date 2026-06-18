from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _delta(metrics: dict[str, Any], reference: dict[str, Any], key: str, reference_key: str) -> float | None:
    value = metrics.get(key)
    ref_value = reference.get(reference_key)
    if value is None or ref_value is None:
        return None
    return float(value) - float(ref_value)


def summarize_sweeps(root: Path, baseline: Path | None) -> dict[str, Any]:
    summary_paths = sorted(root.glob("*/native_support_metrics_summary.json"))
    if baseline and baseline.exists():
        summary_paths.insert(0, baseline)

    rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = _read_json(path)
        name = path.parent.name
        metrics = payload.get("aggregate_metrics", {})
        gate = payload.get("gate", {})
        no_birth_pass = gate.get("no_d4rt_tube_birth_pass")
        if no_birth_pass is None and metrics.get("birth_from_d4rt_tube_count_sum") is not None:
            no_birth_pass = int(metrics.get("birth_from_d4rt_tube_count_sum") or 0) == 0
        pass_gate = bool(gate.get("pass_native_support_metric_gate") is True and no_birth_pass is True)
        v37_4d = payload.get("v37_reference", {}).get("v37_4d_decision", {}).get("best_metrics", {})
        rows.append(
            {
                "config": name,
                "recorded_pass_gate": gate.get("pass_native_support_metric_gate"),
                "pass_gate": pass_gate,
                "ari_pass": gate.get("ari_pass"),
                "purity_pass": gate.get("purity_pass"),
                "completeness_pass": gate.get("completeness_pass"),
                "unknown_pass": gate.get("unknown_pass"),
                "prediction_count_pass": gate.get("prediction_count_pass"),
                "duplicate_rate_pass": gate.get("duplicate_rate_pass"),
                "conflict_rate_pass": gate.get("conflict_rate_pass"),
                "no_d4rt_tube_birth_pass": no_birth_pass,
                "tube_birth_negative_control_pass": gate.get("tube_birth_negative_control_pass"),
                "no_forbidden_prediction_source": gate.get("no_forbidden_prediction_source"),
                "ARI": metrics.get("4D_ARI"),
                "purity": metrics.get("4D_purity"),
                "completeness": metrics.get("4D_completeness"),
                "unknown_labeled": metrics.get("unknown_tube_ratio_labeled"),
                "mean_predictions_per_scene": metrics.get("mean_predictions_per_scene"),
                "duplicate_rate_mean": metrics.get("duplicate_rate_mean"),
                "conflict_rate_mean": metrics.get("conflict_rate_mean"),
                "native_points": metrics.get("native_point_count_sum"),
                "exported_tubes": metrics.get("exported_tube_count_sum"),
                "birth_from_d4rt_tube_count_sum": metrics.get("birth_from_d4rt_tube_count_sum"),
                "rejected_forbidden_birth_candidate_count_sum": metrics.get(
                    "rejected_forbidden_birth_candidate_count_sum"
                ),
                "selected_fields_mean": metrics.get("selected_object_field_count_mean"),
                "delta_v37_4d_ARI": _delta(metrics, v37_4d, "4D_ARI", "4D_ARI"),
                "delta_v37_4d_purity": _delta(metrics, v37_4d, "4D_purity", "4D_purity"),
                "delta_v37_4d_completeness": _delta(metrics, v37_4d, "4D_completeness", "4D_completeness"),
            }
        )
        for scene_row in payload.get("scene_rows", []):
            scene_rows.append({"config": name, **scene_row})

    pass_rows = [row for row in rows if row.get("pass_gate") is True]
    ranked_by_ari = sorted(
        rows,
        key=lambda row: (
            row.get("ARI") is not None,
            row.get("ARI") if row.get("ARI") is not None else -1.0,
            row.get("purity") if row.get("purity") is not None else -1.0,
            row.get("completeness") if row.get("completeness") is not None else -1.0,
        ),
        reverse=True,
    )
    ranked_balanced = sorted(
        rows,
        key=lambda row: (
            bool(row.get("ari_pass")) and bool(row.get("completeness_pass")),
            row.get("purity") if row.get("purity") is not None else -1.0,
            row.get("ARI") if row.get("ARI") is not None else -1.0,
            row.get("completeness") if row.get("completeness") is not None else -1.0,
        ),
        reverse=True,
    )
    return {
        "config_count": len(rows),
        "pass_gate_count": len(pass_rows),
        "pass_gate_configs": [str(row["config"]) for row in pass_rows],
        "best_by_ari": ranked_by_ari[0] if ranked_by_ari else None,
        "best_balanced": ranked_balanced[0] if ranked_balanced else None,
        "rows": rows,
        "scene_rows": scene_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize v41.1 native-support metric sweep outputs.")
    parser.add_argument("--sweep-root", default="outputs/audit/v41_1_native_support_metrics_probe5_sweep")
    parser.add_argument(
        "--baseline-summary",
        default="outputs/audit/v41_1_native_support_metrics_probe5/native_support_metrics_summary.json",
    )
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-scene-csv", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sweep_root = Path(args.sweep_root)
    baseline = Path(args.baseline_summary) if args.baseline_summary else None
    summary = summarize_sweeps(sweep_root, baseline)
    output_csv = Path(args.output_csv) if args.output_csv else sweep_root / "probe5_sweep_summary.csv"
    output_json = Path(args.output_json) if args.output_json else sweep_root / "probe5_sweep_summary.json"
    output_scene_csv = Path(args.output_scene_csv) if args.output_scene_csv else sweep_root / "probe5_sweep_scene_rows.csv"
    _write_csv(output_csv, summary["rows"])
    _write_csv(output_scene_csv, summary["scene_rows"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("config_count", "pass_gate_count", "pass_gate_configs", "best_by_ari", "best_balanced")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
