#!/usr/bin/env python3
"""Aggregate v96 Phase2 segmented D4RT decode quality rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v96_phase2_d4rt_micro_tracks_segment_aggregate"
RUN_ID = "v96_phase2_d4rt_micro_tracks_segment_aggregate"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase2_d4rt_micro_tracks_segment_aggregate"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
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
        return _rel(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _parse_include(raw: str) -> tuple[Path, set[str]]:
    parts = raw.split("::")
    root = _project(parts[0])
    scenes: set[str] = set()
    for part in parts[1:]:
        if part.startswith("scene="):
            scenes.add(part.split("=", 1)[1])
        elif part:
            raise ValueError(f"unknown include filter {part!r}; use ::scene=<scene_id>")
    return root, scenes


def _load_included_rows(includes: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for raw in includes:
        root, scenes = _parse_include(raw)
        quality_path = root / "micro_track_quality_rows.csv"
        if not quality_path.exists():
            raise FileNotFoundError(f"missing micro_track_quality_rows.csv under {root}")
        source_rows = _read_csv(quality_path)
        kept = 0
        for row in source_rows:
            scene = row.get("scene_id", "")
            if scenes and scene not in scenes:
                continue
            key = (
                row.get("decode_variant", ""),
                row.get("query_variant", ""),
                scene,
                row.get("window_id", ""),
                row.get("frame_ids", ""),
            )
            if key in seen:
                raise ValueError(f"duplicate segment key encountered: {key}")
            seen.add(key)
            out = dict(row)
            out["source_phase2_root"] = _rel(root)
            rows.append(out)
            kept += 1
        manifest_rows.append(
            {
                "include_arg": raw,
                "source_root": _rel(root),
                "scene_filter": ",".join(sorted(scenes)) if scenes else "*",
                "source_quality_rows": len(source_rows),
                "kept_quality_rows": kept,
            }
        )
    return rows, manifest_rows


def _aggregate_variant_rows(quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        by_variant[str(row.get("decode_variant", ""))].append(row)
    variant_rows: list[dict[str, Any]] = []
    for variant, rows in sorted(by_variant.items()):
        weights = np.asarray([int(_num(row.get("query_count"))) for row in rows], dtype=np.float64)

        def wmean(key: str) -> float:
            vals = np.asarray([_num(row.get(key)) for row in rows], dtype=np.float64)
            return float(np.average(vals, weights=weights)) if weights.sum() > 0 else 0.0

        variant_rows.append(
            {
                "decode_variant": variant,
                "query_variant": rows[0].get("query_variant", ""),
                "segment_quality_row_count": len(rows),
                "query_count": int(weights.sum()),
                "target_frame_count_sum": int(sum(int(_num(row.get("target_frame_count"))) for row in rows)),
                "valid_track_ratio": wmean("valid_track_ratio"),
                "uv_in01_rate": wmean("uv_in01_rate"),
                "visibility_mean": wmean("visibility_mean"),
                "confidence_mean": wmean("confidence_mean"),
                "track_length_visible_mean": wmean("track_length_visible_mean"),
                "track_length_visible_p10_mean": wmean("track_length_visible_p10"),
                "projection_jitter_mean": wmean("projection_jitter_mean"),
                "projection_jitter_p90_mean": wmean("projection_jitter_p90"),
                "mask_membership_flip_rate": wmean("mask_membership_flip_rate"),
                "source_support_area_ratio": wmean("source_support_area_ratio"),
                "boundary_band_support_ratio": wmean("boundary_band_support_ratio"),
                "competing_edge_support_ratio": wmean("competing_edge_support_ratio"),
                "semantic_gradient_support_ratio": wmean("semantic_gradient_support_ratio"),
                "runtime_decode_sec": float(sum(_num(row.get("runtime_decode_sec")) for row in rows)),
                "GPU_memory_peak_MB": float(max((_num(row.get("GPU_memory_peak_MB")) for row in rows), default=0.0)),
            }
        )
    return variant_rows


def _gate_rows(variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {row["decode_variant"]: row for row in variant_rows}
    d1 = by_variant.get("D1_uniform1024", {})
    d3 = by_variant.get("D3_adaptive1024", {})
    rows = [
        {
            "schema_version": "stream4d_v96_phase2_segment_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "D3_boundary_support_ge_uniform1024_plus_0p05_or_abs_0p08",
            "pass": bool(
                d3
                and (
                    _num(d3.get("boundary_band_support_ratio")) >= _num(d1.get("boundary_band_support_ratio")) + 0.05
                    or _num(d3.get("boundary_band_support_ratio")) >= 0.08
                )
            ),
            "observed": d3.get("boundary_band_support_ratio", ""),
            "uniform1024": d1.get("boundary_band_support_ratio", ""),
            "required": "D3 >= D1 + 0.05 or D3 >= 0.08",
        },
        {
            "schema_version": "stream4d_v96_phase2_segment_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "D3_source_support_ge_uniform1024_plus_0p05_or_abs_0p12",
            "pass": bool(
                d3
                and (
                    _num(d3.get("source_support_area_ratio")) >= _num(d1.get("source_support_area_ratio")) + 0.05
                    or _num(d3.get("source_support_area_ratio")) >= 0.12
                )
            ),
            "observed": d3.get("source_support_area_ratio", ""),
            "uniform1024": d1.get("source_support_area_ratio", ""),
            "required": "D3 >= D1 + 0.05 or D3 >= 0.12",
        },
        {
            "schema_version": "stream4d_v96_phase2_segment_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "D3_uv_in01_ge_0p90",
            "pass": bool(d3 and _num(d3.get("uv_in01_rate")) >= 0.90),
            "observed": d3.get("uv_in01_rate", ""),
            "required": 0.90,
        },
        {
            "schema_version": "stream4d_v96_phase2_segment_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "D3_valid_track_ratio_ge_0p50",
            "pass": bool(d3 and _num(d3.get("valid_track_ratio")) >= 0.50),
            "observed": d3.get("valid_track_ratio", ""),
            "required": 0.50,
        },
        {
            "schema_version": "stream4d_v96_phase2_segment_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "runtime_recorded_no_oom",
            "pass": all(_num(row.get("runtime_decode_sec")) > 0 for row in variant_rows),
            "observed": f"variants={len(variant_rows)}",
            "required": "runtime>0 for each variant",
        },
    ]
    for row in rows:
        row["uses_gt_for_prediction"] = False
        row["uses_future"] = False
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_rows, manifest_rows = _load_included_rows(args.include_root)
    if not quality_rows:
        raise RuntimeError("No quality rows included for segment aggregate.")
    variant_rows = _aggregate_variant_rows(quality_rows)
    gates = _gate_rows(variant_rows)
    decision = "PASS_V96_PHASE2_SEGMENT_AGGREGATE" if all(bool(row.get("pass")) for row in gates) else "NO_GO_V96_PHASE2_SEGMENT_AGGREGATE"
    summary = {
        "schema": "stream4d_v96_phase2_segment_aggregate_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": decision,
        "output_root": _rel(output_root),
        "include_manifest_rows": _rel(output_root / "include_manifest_rows.csv"),
        "included_quality_rows": _rel(output_root / "included_quality_rows.csv"),
        "variant_summary_rows": _rel(output_root / "variant_summary_rows.csv"),
        "phase2_gate_rows": _rel(output_root / "phase2_gate_rows.csv"),
        "included_quality_row_count": len(quality_rows),
        "variant_summaries": variant_rows,
        "gate_rows": gates,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(output_root / "include_manifest_rows.csv", manifest_rows)
    _write_csv(output_root / "included_quality_rows.csv", quality_rows)
    _write_csv(output_root / "variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "phase2_gate_rows.csv", gates)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": decision, "included_quality_row_count": len(quality_rows), "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate v96 Phase2 segmented quality rows.")
    parser.add_argument("--include-root", action="append", required=True, help="Root, optionally with ::scene=<scene_id> filter.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
