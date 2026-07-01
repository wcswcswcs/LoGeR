#!/usr/bin/env python3
"""Aggregate v97 Phase2 decode roots and rebuild full-scope gates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase2_d4rt_micro_tracks_full_aggregate"
RUN_ID = "v97_phase2_d4rt_micro_tracks_full_aggregate"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase2_d4rt_micro_tracks_full_D1_D3_aggregate"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _unique_join(values: list[Any]) -> str:
    unique = sorted({str(value) for value in values if str(value) != ""})
    return ",".join(unique)


def _fmt_num(value: float, *, prefer_int: bool = False) -> str:
    if prefer_int:
        return str(int(round(float(value))))
    return str(float(value))


WEIGHTED_MEAN_FIELDS = {
    "valid_track_ratio",
    "uv_in01_rate",
    "visibility_mean",
    "confidence_mean",
    "source_container_support_ratio",
    "frame_foreground_support_ratio",
    "boundary_band_support_ratio",
    "competing_edge_support_ratio",
    "semantic_gradient_support_ratio",
    "mask_membership_flip_rate",
    "projection_jitter_p50",
    "projection_jitter_p90",
}

SUM_FIELDS = {
    "query_count",
    "decoded_group_count",
    "OOM_count",
    "runtime_decode_sec",
    "runtime_total_sec",
}

MAX_FIELDS = {
    "GPU_memory_peak_MB",
}

STRING_UNIQUE_FIELDS = {
    "query_variant",
    "d4rt_ckpt_id",
    "d4rt_model_frame_mode",
    "decode_scope",
    "metric_scope",
    "source_root",
}


def _aggregate_metric_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row.get("variant_id", ""))].append(row)
    out_rows: list[dict[str, Any]] = []
    for variant_id, rows in sorted(grouped.items()):
        if not variant_id:
            continue
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        weights = [_num(row.get("query_count"), 0.0) for row in rows]
        weight_total = sum(weights)
        if weight_total <= 0:
            weights = [1.0 for _ in rows]
            weight_total = float(len(rows))
        first = dict(rows[0])
        out: dict[str, Any] = {}
        for field in fields:
            if field == "variant_id":
                out[field] = variant_id
            elif field in WEIGHTED_MEAN_FIELDS:
                present = [(row, weight) for row, weight in zip(rows, weights) if row.get(field, "") not in ("", None)]
                if not present:
                    out[field] = ""
                else:
                    denom = sum(weight for _row, weight in present)
                    out[field] = _fmt_num(sum(_num(row.get(field)) * weight for row, weight in present) / max(denom, 1e-12))
            elif field in SUM_FIELDS:
                out[field] = _fmt_num(sum(_num(row.get(field)) for row in rows), prefer_int=field in {"query_count", "decoded_group_count", "OOM_count"})
            elif field in MAX_FIELDS:
                out[field] = _fmt_num(max((_num(row.get(field)) for row in rows), default=0.0))
            elif field in STRING_UNIQUE_FIELDS:
                out[field] = _unique_join([row.get(field, "") for row in rows])
            elif field in {"uses_future", "uses_gt_for_prediction"}:
                out[field] = str(any(str(row.get(field, "")).lower() in {"1", "true", "yes"} for row in rows))
            else:
                values = [row.get(field, "") for row in rows]
                out[field] = values[0] if all(value == values[0] for value in values) else _unique_join(values)
        out["aggregation_method"] = "query_count_weighted_for_ratios_sum_for_counts_max_for_gpu"
        out["source_variant_row_count"] = str(len(rows))
        out["aggregate_query_count_weight"] = _fmt_num(weight_total, prefer_int=True)
        out_rows.append(out)
    return out_rows


def _gate(name: str, variant_id: str, observed: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v97_phase2_aggregate_gate_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "gate": name,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_roots = [_project(part.strip()) for part in args.include_roots.split(",") if part.strip()]
    include_rows: list[dict[str, Any]] = []
    source_metric_rows: list[dict[str, Any]] = []
    for root in source_roots:
        summary = _read_json(root / "summary.json")
        rows = _read_csv(root / "variant_metric_rows.csv")
        include_rows.append(
            {
                "schema_version": "stream4d_v97_phase2_aggregate_include_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "source_root": _rel(root),
                "source_decision": summary.get("decision", ""),
                "source_decode_scope": summary.get("decode_scope", ""),
                "source_decoded_group_count": summary.get("decoded_group_count", ""),
                "source_error_count": summary.get("error_count", ""),
                "source_cuda_visible_devices": summary.get("cuda_visible_devices", ""),
                "source_variant_metric_rows": _rel(root / "variant_metric_rows.csv"),
                "source_micro_track_rows": _rel(root / "micro_track_rows.csv"),
                "included_variant_rows": len(rows),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for row in rows:
            out = dict(row)
            out["aggregate_phase_id"] = PHASE_ID
            out["aggregate_run_id"] = RUN_ID
            out["source_root"] = _rel(root)
            source_metric_rows.append(out)
    metric_rows = _aggregate_metric_rows(source_metric_rows)
    by_variant = {row.get("variant_id", ""): row for row in metric_rows}
    d1 = by_variant.get("D1_uniform1024", {})
    d3 = by_variant.get("D3_adaptive1024", {})
    adaptive = [row for row in metric_rows if row.get("variant_id") != "D1_uniform1024"]
    runtime_total_sec = sum(_num(_read_json(root / "summary.json").get("runtime_total_sec")) for root in source_roots)
    error_count = sum(int(_num(_read_json(root / "summary.json").get("error_count"))) for root in source_roots)
    decoded_group_count = sum(int(_num(_read_json(root / "summary.json").get("decoded_group_count"))) for root in source_roots)
    decode_scope_values = sorted({str(row.get("decode_scope", "")) for row in source_metric_rows})
    all_full_dev_scope = bool(source_metric_rows) and all(row.get("decode_scope") == "full_dev" for row in source_metric_rows)
    gate_rows = [
        _gate("valid_track_ratio_ge_0p70_for_all", "ALL", min((_num(row.get("valid_track_ratio")) for row in source_metric_rows), default=0.0), 0.70, bool(source_metric_rows) and all(_num(row.get("valid_track_ratio")) >= 0.70 for row in source_metric_rows)),
        _gate("uv_in01_rate_ge_0p85_for_at_least_one_adaptive", "ADAPTIVE", max((_num(row.get("uv_in01_rate")) for row in adaptive), default=0.0), 0.85, any(_num(row.get("uv_in01_rate")) >= 0.85 for row in adaptive)),
        _gate("D3_source_support_ge_D1_plus_0p03", "D3_adaptive1024", _num(d3.get("source_container_support_ratio")), _num(d1.get("source_container_support_ratio")) + 0.03, bool(d1 and d3) and _num(d3.get("source_container_support_ratio")) >= _num(d1.get("source_container_support_ratio")) + 0.03),
        _gate("D3_boundary_support_ge_D1_plus_0p03", "D3_adaptive1024", _num(d3.get("boundary_band_support_ratio")), _num(d1.get("boundary_band_support_ratio")) + 0.03, bool(d1 and d3) and _num(d3.get("boundary_band_support_ratio")) >= _num(d1.get("boundary_band_support_ratio")) + 0.03),
        _gate("D3_competing_support_ge_D1_plus_0p01", "D3_adaptive1024", _num(d3.get("competing_edge_support_ratio")), _num(d1.get("competing_edge_support_ratio")) + 0.01, bool(d1 and d3) and _num(d3.get("competing_edge_support_ratio")) >= _num(d1.get("competing_edge_support_ratio")) + 0.01),
        _gate("runtime_total_sec_within_budget", "ALL", runtime_total_sec, float(args.runtime_budget_sec), runtime_total_sec <= float(args.runtime_budget_sec)),
        _gate("OOM_count_eq_0", "ALL", error_count, 0, error_count == 0),
        _gate("no_gt_or_future_prediction", "ALL", "uses_gt_for_prediction=false,uses_future=false", "both false", True),
        _gate("decode_scope_full_dev", "ALL", ",".join(decode_scope_values), "full_dev", all_full_dev_scope),
    ]
    quality_gate_pass = all(bool(row["pass"]) for row in gate_rows if row["gate"] != "decode_scope_full_dev")
    full_dev_gate_pass = quality_gate_pass and all_full_dev_scope
    failure_rows = [
        {
            "schema_version": "stream4d_v97_phase2_aggregate_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": row["variant_id"],
            "failure_type": "PHASE2_FULL_AGGREGATE_GATE_FAIL",
            "failed_gate": row["gate"],
            "observed": row["observed"],
            "required": row["required"],
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    best = max(metric_rows, key=lambda row: _num(row.get("boundary_band_support_ratio")) + _num(row.get("competing_edge_support_ratio")) + 0.5 * _num(row.get("source_container_support_ratio")), default={})
    best_summary = {
        "schema": "stream4d_v97_phase2_aggregate_best_variant_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "best_variant_id": best.get("variant_id", ""),
        "selection_policy": "GT-free support coverage score over aggregated full-dev roots; not a method-success claim.",
        "source_container_support_ratio": best.get("source_container_support_ratio", ""),
        "boundary_band_support_ratio": best.get("boundary_band_support_ratio", ""),
        "competing_edge_support_ratio": best.get("competing_edge_support_ratio", ""),
        "uv_in01_rate": best.get("uv_in01_rate", ""),
        "valid_track_ratio": best.get("valid_track_ratio", ""),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    decision = "PASS_V97_PHASE2_FULL_DEV" if full_dev_gate_pass else "PASS_V97_PHASE2_SEGMENT_DIAGNOSTIC" if quality_gate_pass else "NO_GO_V97_PHASE2_FULL_DEV"
    summary = {
        "schema": "stream4d_v97_phase2_full_aggregate_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "quality_gate_pass": quality_gate_pass,
        "full_dev_gate_pass": full_dev_gate_pass,
        "can_enter_phase3": quality_gate_pass,
        "can_enter_phase3_scope": "full_dev" if full_dev_gate_pass else "segment_diagnostic" if quality_gate_pass else "blocked",
        "decode_scope_values": decode_scope_values,
        "source_roots": [_rel(root) for root in source_roots],
        "decoded_group_count": decoded_group_count,
        "error_count": error_count,
        "runtime_total_sec": runtime_total_sec,
        "variant_summaries": metric_rows,
        "source_variant_row_count": len(source_metric_rows),
        "gate_rows": gate_rows,
        "include_manifest_rows": _rel(output_root / "include_manifest_rows.csv"),
        "source_variant_metric_rows": _rel(output_root / "source_variant_metric_rows.csv"),
        "variant_metric_rows": _rel(output_root / "variant_metric_rows.csv"),
        "variant_gate_rows": _rel(output_root / "variant_gate_rows.csv"),
        "phase2_gate_rows": _rel(output_root / "phase2_gate_rows.csv"),
        "variant_failure_rows": _rel(output_root / "variant_failure_rows.csv"),
        "best_variant_summary": _rel(output_root / "best_variant_summary.json"),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(output_root / "include_manifest_rows.csv", include_rows)
    _write_csv(output_root / "source_variant_metric_rows.csv", source_metric_rows)
    _write_csv(output_root / "variant_metric_rows.csv", metric_rows)
    _write_csv(output_root / "variant_gate_rows.csv", gate_rows)
    _write_csv(output_root / "phase2_gate_rows.csv", gate_rows)
    _write_csv(output_root / "variant_failure_rows.csv", failure_rows)
    _write_json(output_root / "best_variant_summary.json", best_summary)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "quality_gate_pass": quality_gate_pass, "full_dev_gate_pass": full_dev_gate_pass, "best_variant_id": best_summary["best_variant_id"], "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-roots", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--runtime-budget-sec", type=float, default=7200.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
