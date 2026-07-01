from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OUT = ROOT / "outputs/audit/v93_phase7_density_readout_gap"
HR2_ROOT = ROOT / "outputs/audit/v92_phase3_hr2_same_readout_adaptive_materialization"
A512_ROOT = ROOT / "outputs/audit/v93_phase7_A512_same_readout_adaptive_materialization"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def _median(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else 0.0


def _summarize_generated(root: Path, label: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(root / "generated_mask_rows.csv"):
        groups[str(row.get("variant_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(groups.items()):
        source_area = [_num(row.get("source_mask_area"), 0.0) for row in rows]
        support_area = [_num(row.get("support_area"), 0.0) for row in rows]
        generated_area = [_num(row.get("generated_mask_area"), 0.0) for row in rows]
        support_counts = [_num(row.get("carrier_support_count"), 0.0) for row in rows]
        out.append(
            {
                "run_label": label,
                "variant_id": variant,
                "generated_rows": len(rows),
                "support_count_mean": _mean(support_counts),
                "support_count_median": _median(support_counts),
                "support_to_source_area_mean": _mean([s / max(1.0, a) for s, a in zip(support_area, source_area)]),
                "generated_to_source_area_mean": _mean([g / max(1.0, a) for g, a in zip(generated_area, source_area)]),
                "generated_to_support_area_mean": _mean([g / max(1.0, s) for g, s in zip(generated_area, support_area)]),
                "object_score_mean": _mean([_num(row.get("object_score"), 0.0) for row in rows]),
                "adaptive_radius_mean": _mean([_num(row.get("adaptive_radius"), 0.0) for row in rows]),
                "adaptive_jitter_px_mean": _mean([_num(row.get("adaptive_jitter_px"), 0.0) for row in rows]),
                "uses_gt_for_prediction": any(_bool(row.get("uses_gt_for_prediction")) for row in rows),
                "uses_future": any(_bool(row.get("uses_future")) for row in rows),
            }
        )
    return out


def _summarize_quality(root: Path, label: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(root / "support_quality_rows.csv"):
        groups[str(row.get("variant_id", ""))].append(row)
    out: list[dict[str, Any]] = []
    for variant, rows in sorted(groups.items()):
        out.append(
            {
                "run_label": label,
                "variant_id": variant,
                "support_quality_rows": len(rows),
                "support_carrier_count_mean": _mean([_num(row.get("support_carrier_count"), 0.0) for row in rows]),
                "support_heatmap_area_mean": _mean([_num(row.get("support_heatmap_area"), 0.0) for row in rows]),
                "selected_mask_area_mean": _mean([_num(row.get("selected_mask_area"), 0.0) for row in rows]),
                "generated_mask_area_mean": _mean([_num(row.get("generated_mask_area"), 0.0) for row in rows]),
                "support_to_mask_ratio_mean": _mean([_num(row.get("support_to_mask_ratio"), 0.0) for row in rows]),
                "mask_to_support_ratio_mean": _mean([_num(row.get("mask_to_support_ratio"), 0.0) for row in rows]),
                "support_coverage_mean": _mean([_num(row.get("support_coverage"), 0.0) for row in rows]),
                "support_density_mean": _mean([_num(row.get("support_density"), 0.0) for row in rows]),
                "broad_risk_rate": _mean([1.0 if _bool(row.get("broad_risk")) else 0.0 for row in rows]),
                "hard_negative_density_mean": _mean([_num(row.get("hard_negative_density"), 0.0) for row in rows]),
            }
        )
    return out


def _comparison_rows(hr2: list[dict[str, Any]], a512: list[dict[str, Any]], metric_prefix: str) -> list[dict[str, Any]]:
    by_hr2 = {row["variant_id"]: row for row in hr2}
    by_a512 = {row["variant_id"]: row for row in a512}
    numeric_keys = sorted(
        {
            key
            for row in [*hr2, *a512]
            for key, value in row.items()
            if key not in {"run_label", "variant_id"} and isinstance(value, (int, float))
        }
    )
    out: list[dict[str, Any]] = []
    for variant in sorted(set(by_hr2) | set(by_a512)):
        row: dict[str, Any] = {"metric_group": metric_prefix, "variant_id": variant}
        h = by_hr2.get(variant, {})
        a = by_a512.get(variant, {})
        for key in numeric_keys:
            hv = _num(h.get(key), 0.0)
            av = _num(a.get(key), 0.0)
            row[f"HR2_{key}"] = hv
            row[f"A512_{key}"] = av
            row[f"delta_{key}"] = av - hv
        out.append(row)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    gen_hr2 = _summarize_generated(args.hr2_root, "HR2_G16_uniform")
    gen_a512 = _summarize_generated(args.a512_root, "A512_adaptive")
    qual_hr2 = _summarize_quality(args.hr2_root, "HR2_G16_uniform")
    qual_a512 = _summarize_quality(args.a512_root, "A512_adaptive")
    generated_compare = _comparison_rows(gen_hr2, gen_a512, "generated")
    quality_compare = _comparison_rows(qual_hr2, qual_a512, "support_quality")
    _write_csv(out / "generated_summary_rows.csv", gen_hr2 + gen_a512)
    _write_csv(out / "support_quality_summary_rows.csv", qual_hr2 + qual_a512)
    _write_csv(out / "density_readout_comparison_rows.csv", generated_compare + quality_compare)
    best_variant = "V91_AD1_sr2_adapt_sig4_b05_j05_r16"
    best_gen = next((row for row in generated_compare if row["metric_group"] == "generated" and row["variant_id"] == best_variant), {})
    best_quality = next((row for row in quality_compare if row["metric_group"] == "support_quality" and row["variant_id"] == best_variant), {})
    summary = {
        "schema": "stream4d_v93_phase7_density_readout_gap_v1",
        "decision": "DIAGNOSE_V93_PHASE7_DENSITY_INCREASE_NOT_READOUT_GAIN",
        "best_variant_id": best_variant,
        "best_variant_delta_support_count_mean": best_gen.get("delta_support_count_mean", ""),
        "best_variant_delta_support_to_source_area_mean": best_gen.get("delta_support_to_source_area_mean", ""),
        "best_variant_delta_generated_to_source_area_mean": best_gen.get("delta_generated_to_source_area_mean", ""),
        "best_variant_delta_object_score_mean": best_gen.get("delta_object_score_mean", ""),
        "best_variant_delta_support_to_mask_ratio_mean": best_quality.get("delta_support_to_mask_ratio_mean", ""),
        "best_variant_delta_mask_to_support_ratio_mean": best_quality.get("delta_mask_to_support_ratio_mean", ""),
        "best_variant_delta_support_density_mean": best_quality.get("delta_support_density_mean", ""),
        "interpretation": "A512 increases witness/support density, but generated masks remain near source extent and score/ranking does not clear control gates.",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose why v93 Phase7 adaptive density did not improve readout AP.")
    parser.add_argument("--hr2-root", type=Path, default=HR2_ROOT)
    parser.add_argument("--a512-root", type=Path, default=A512_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
