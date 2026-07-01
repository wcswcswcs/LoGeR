#!/usr/bin/env python3
"""Build v97 Phase3 Triton point-to-mask incidence diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase3_triton_incidence"
RUN_ID = "v97_phase3_triton_incidence"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_SCANNET = ROOT / "data/scannet/processed"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase3_triton_incidence"

V97_EVENT_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "variant_id",
    "query_variant",
    "scene_id",
    "window_id",
    "frame_id",
    "micro_primitive_id",
    "mask_observation_id",
    "source_mask_id",
    "mask_id",
    "membership",
    "visibility",
    "confidence",
    "B_pa",
    "inside_mask",
    "near_boundary",
    "signed_boundary_proxy",
    "distinct_mask_count_3x3",
    "query_stratum",
    "uses_gt_for_prediction",
    "uses_future",
]


def _load_v96_module() -> Any:
    path = ROOT / "tools/build_v96_phase3_triton_incidence.py"
    spec = importlib.util.spec_from_file_location("_stream4d_v96_phase3_for_v97", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load v96 Phase3 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "true"}


def _parse_include(raw: str) -> tuple[Path, set[str]]:
    parts = raw.split("::")
    root = _project(parts[0])
    scenes: set[str] = set()
    for part in parts[1:]:
        if part.startswith("scene="):
            scenes.add(part.split("=", 1)[1])
    return root, scenes


def _iter_selected_track_rows(include_roots: list[str], decode_variants: set[str], max_track_rows: int) -> Iterator[dict[str, str]]:
    yielded = 0
    for raw in include_roots:
        root, scenes = _parse_include(raw)
        path = root / "micro_track_rows.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if decode_variants and row.get("decode_variant", "") not in decode_variants:
                    continue
                if scenes and row.get("scene_id", "") not in scenes:
                    continue
                yield row
                yielded += 1
                if max_track_rows > 0 and yielded >= max_track_rows:
                    return


def _source_mask_lookup(include_roots: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in include_roots:
        root, _scenes = _parse_include(raw)
        path = root / "micro_query_rows.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                query_id = row.get("query_id", "")
                if query_id and query_id not in mapping:
                    mapping[query_id] = row.get("source_mask_id_optional", "")
    return mapping


def _rewrite_incidence_events(
    *,
    output_root: Path,
    include_roots: list[str],
    decode_variants: set[str],
    max_track_rows: int,
    near_boundary_px: float,
) -> dict[str, Any]:
    old_path = output_root / "incidence_event_rows.csv"
    raw_backup = output_root / "incidence_event_rows_v96_raw.csv"
    if raw_backup.exists():
        raw_backup.unlink()
    old_path.rename(raw_backup)
    source_masks = _source_mask_lookup(include_roots)
    event_count = 0
    positive_count = 0
    bpa_sum = 0.0
    out_of_bound_center_count = 0
    with raw_backup.open(newline="", encoding="utf-8") as event_handle, old_path.open("w", newline="", encoding="utf-8") as out_handle:
        reader = csv.DictReader(event_handle)
        writer = csv.DictWriter(out_handle, fieldnames=V97_EVENT_FIELDS)
        writer.writeheader()
        for raw_event, track in zip(reader, _iter_selected_track_rows(include_roots, decode_variants, max_track_rows)):
            visibility = _num(track.get("visibility"))
            confidence = _num(track.get("confidence"))
            mask_id = int(_num(raw_event.get("center_mask_id")))
            membership = mask_id > 0
            b_pa = visibility * confidence if membership and _bool(track.get("uv_in01")) else 0.0
            boundary_distance = _num(raw_event.get("boundary_distance_px"), -1.0)
            near_boundary = membership and boundary_distance >= 0.0 and boundary_distance <= near_boundary_px
            frame_id = int(_num(raw_event.get("target_frame_id")))
            query_id = raw_event.get("query_id", "")
            row = {
                "schema_version": "stream4d_v97_incidence_event_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": raw_event.get("decode_variant", ""),
                "query_variant": raw_event.get("query_variant", ""),
                "scene_id": raw_event.get("scene_id", ""),
                "window_id": raw_event.get("window_id", ""),
                "frame_id": frame_id,
                "micro_primitive_id": query_id,
                "mask_observation_id": f"{raw_event.get('scene_id', '')}:{raw_event.get('window_id', '')}:f{frame_id}:m{mask_id}",
                "source_mask_id": source_masks.get(query_id, ""),
                "mask_id": mask_id,
                "membership": membership,
                "visibility": visibility,
                "confidence": confidence,
                "B_pa": b_pa,
                "inside_mask": membership,
                "near_boundary": near_boundary,
                "signed_boundary_proxy": boundary_distance,
                "distinct_mask_count_3x3": raw_event.get("distinct_mask_count_3x3", ""),
                "query_stratum": raw_event.get("query_stratum", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            writer.writerow(row)
            event_count += 1
            positive_count += int(membership)
            bpa_sum += float(b_pa)
            out_of_bound_center_count += int(boundary_distance < 0)
    return {
        "event_count": event_count,
        "positive_mask_rate": positive_count / max(1, event_count),
        "B_pa_mean": bpa_sum / max(1, event_count),
        "out_of_bound_center_count": out_of_bound_center_count,
        "raw_v96_event_rows": _rel(raw_backup),
        "source_mask_lookup_count": len(source_masks),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    v96 = _load_v96_module()
    v96.PHASE_ID = PHASE_ID
    v96.RUN_ID = RUN_ID
    v96_summary = v96.run(args)
    decode_variants = {part.strip() for part in args.decode_variants.split(",") if part.strip()}
    event_stats = _rewrite_incidence_events(
        output_root=output_root,
        include_roots=args.include_root,
        decode_variants=decode_variants,
        max_track_rows=int(args.max_track_rows),
        near_boundary_px=float(args.near_boundary_px),
    )
    v96_variant_rows = []
    with (output_root / "variant_metric_rows.csv").open(newline="", encoding="utf-8") as handle:
        v96_variant_rows = list(csv.DictReader(handle))
    incidence_quality_rows = []
    for row in v96_variant_rows:
        incidence_quality_rows.append(
            {
                "schema_version": "stream4d_v97_phase3_incidence_quality_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": row.get("decode_variant", ""),
                "query_variant": row.get("query_variant", ""),
                "incidence_event_count": row.get("incidence_event_count", ""),
                "positive_mask_rate": row.get("query_with_positive_mask_rate", ""),
                "distinct_mask_count_mean": row.get("mean_masks_per_query", ""),
                "boundary_query_with_mask_rate": row.get("boundary_query_with_mask_rate", ""),
                "conflict_query_with_multiple_masks_rate": row.get("conflict_query_with_multiple_masks_rate", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    parity = (v96_summary.get("parity") or {})
    kernel_runtime_rows = [
        {
            "schema_version": "stream4d_v97_phase3_kernel_runtime_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "kernel_name": "triton_incidence_lookup",
            "kernel_runtime_ms": float(v96_summary.get("runtime_incidence_sec", 0.0)) * 1000.0,
            "host_to_device_ms": "",
            "device_to_host_ms": "",
            "GPU_memory_peak_MB": v96_summary.get("GPU_memory_peak_MB", ""),
            "selected_track_rows": v96_summary.get("selected_track_rows", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    gate_rows = [
        {
            "schema_version": "stream4d_v97_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "membership_cpu_gpu_mismatch_count_eq_0_on_parity_sample",
            "pass": bool(_num(parity.get("cpu_vs_triton_membership_mismatch_rate")) == 0.0),
            "observed": parity.get("cpu_vs_triton_membership_mismatch_rate", ""),
            "required": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v97_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "distinct_count_cpu_gpu_mismatch_count_eq_0_on_parity_sample",
            "pass": bool(_num(parity.get("cpu_vs_triton_distinct_count_mismatch_rate")) == 0.0),
            "observed": parity.get("cpu_vs_triton_distinct_count_mismatch_rate", ""),
            "required": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v97_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "positive_mask_rate_gt_0p05",
            "pass": bool(event_stats["positive_mask_rate"] > 0.05),
            "observed": event_stats["positive_mask_rate"],
            "required": 0.05,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v97_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "out_of_bound_center_count_recorded_with_guard",
            "pass": bool(
                event_stats["out_of_bound_center_count"] >= 0
                and _num(parity.get("cpu_vs_triton_membership_mismatch_rate")) == 0.0
                and _num(parity.get("cpu_vs_triton_distinct_count_mismatch_rate")) == 0.0
            ),
            "observed": event_stats["out_of_bound_center_count"],
            "required": "recorded; CPU/Triton parity proves guarded OOB handling",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v97_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "gate": "kernel_runtime_ms_recorded",
            "pass": bool(float(v96_summary.get("runtime_incidence_sec", 0.0)) > 0.0),
            "observed": float(v96_summary.get("runtime_incidence_sec", 0.0)) * 1000.0,
            "required": ">0",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    decision = "PASS_V97_PHASE3_TRITON_INCIDENCE_DIAGNOSTIC" if all(bool(row["pass"]) for row in gate_rows) else "NO_GO_V97_PHASE3_TRITON_INCIDENCE_DIAGNOSTIC"
    _write_csv(output_root / "incidence_quality_rows.csv", incidence_quality_rows)
    _write_csv(output_root / "kernel_runtime_rows.csv", kernel_runtime_rows)
    _write_csv(output_root / "phase3_gate_rows.csv", gate_rows)
    summary = {
        "schema": "stream4d_v97_phase3_triton_incidence_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "diagnostic_scope": "bounded_track_rows" if int(args.max_track_rows) > 0 else "selected_include_roots",
        "max_track_rows": int(args.max_track_rows),
        "output_root": _rel(output_root),
        "include_roots": args.include_root,
        "selected_track_rows": event_stats["event_count"],
        "positive_mask_rate": event_stats["positive_mask_rate"],
        "B_pa_mean": event_stats["B_pa_mean"],
        "out_of_bound_center_count": event_stats["out_of_bound_center_count"],
        "source_mask_lookup_count": event_stats["source_mask_lookup_count"],
        "raw_v96_event_rows": event_stats["raw_v96_event_rows"],
        "incidence_event_rows": _rel(output_root / "incidence_event_rows.csv"),
        "incidence_quality_rows": _rel(output_root / "incidence_quality_rows.csv"),
        "cpu_triton_parity_rows": _rel(output_root / "cpu_triton_parity_rows.csv"),
        "kernel_runtime_rows": _rel(output_root / "kernel_runtime_rows.csv"),
        "variant_metric_rows": _rel(output_root / "variant_metric_rows.csv"),
        "phase3_gate_rows": _rel(output_root / "phase3_gate_rows.csv"),
        "gate_rows": gate_rows,
        "parity": parity,
        "runtime_total_sec": float(time.time() - started),
        "runtime_incidence_sec": v96_summary.get("runtime_incidence_sec", ""),
        "GPU_memory_peak_MB": v96_summary.get("GPU_memory_peak_MB", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": decision, "output_root": _rel(output_root), "selected_track_rows": event_stats["event_count"], "positive_mask_rate": event_stats["positive_mask_rate"], "out_of_bound_center_count": event_stats["out_of_bound_center_count"]}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-root", action="append", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET))
    parser.add_argument("--decode-variants", default="D3_adaptive1024")
    parser.add_argument("--max-track-rows", type=int, default=0)
    parser.add_argument("--parity-sample-rows", type=int, default=4096)
    parser.add_argument("--triton-block-size", type=int, default=256)
    parser.add_argument("--near-boundary-px", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
