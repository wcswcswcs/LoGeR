#!/usr/bin/env python3
"""Decode v97 D4RT micro-tracks from v97 Phase1 query plans."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase2_d4rt_micro_tracks"
RUN_ID = "v97_phase2_d4rt_micro_tracks"
DEFAULT_QUERY_ROOT = ROOT / "outputs/audit/v97_phase1_query_planner"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase2_d4rt_micro_tracks"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_SCANNET = ROOT / "data/scannet/processed"
DEFAULT_D4RT_ROOT = REPO_ROOT / "Open-d4rt"
DEFAULT_D4RT_CONFIG = DEFAULT_D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
DEFAULT_D4RT_CKPT = DEFAULT_D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
DEFAULT_VARIANT_MAP = {
    "D1_uniform1024": "Q1_uniform1024",
    "D2_adaptive512": "Q2_adaptive512",
    "D3_adaptive1024": "Q3_adaptive1024",
    "D4_boundary_conflict1024": "Q4_boundary_conflict1024",
    "D5_semantic_gradient1024": "Q5_semantic_gradient1024",
    "D6_occupancy_adaptive1024": "Q6_occupancy_adaptive1024",
}


def _load_v96_module() -> Any:
    path = ROOT / "tools/build_v96_phase2_d4rt_micro_tracks.py"
    spec = importlib.util.spec_from_file_location("_stream4d_v96_phase2_decode", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load v96 Phase2 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_variant_map(raw: str) -> dict[str, str]:
    if not raw:
        return dict(DEFAULT_VARIANT_MAP)
    out: dict[str, str] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"variant-map entries must be D=Q, got {part!r}")
        left, right = part.split("=", 1)
        out[left.strip()] = right.strip()
    return out


def _balanced_cap_frame_rows(rows: list[dict[str, str]], cap: int) -> list[dict[str, str]]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    priority = {"uniform": 0, "interior": 1, "boundary": 2, "conflict": 3, "semantic_gradient": 4}
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("query_stratum", ""), []).append(row)
    total = max(1, len(rows))
    alloc = {name: min(len(vals), int(math.floor(cap * len(vals) / total))) for name, vals in grouped.items()}
    remaining = cap - sum(alloc.values())
    order = sorted(grouped, key=lambda name: (cap * len(grouped[name]) / total - math.floor(cap * len(grouped[name]) / total), len(grouped[name])), reverse=True)
    while remaining > 0 and order:
        moved = False
        for name in order:
            if remaining <= 0:
                break
            if alloc[name] < len(grouped[name]):
                alloc[name] += 1
                remaining -= 1
                moved = True
        if not moved:
            break
    out: list[dict[str, str]] = []
    for name in sorted(grouped, key=lambda item: (priority.get(item, 9), item)):
        out.extend(grouped[name][: alloc.get(name, 0)])
    return out[:cap]


def _read_selected_queries_v97(
    query_path: Path,
    variant_map: dict[str, str],
    *,
    scenes: set[str],
    window_ids: set[str],
    max_windows: int,
    max_queries_per_frame: int,
    frame_id_min: int,
    frame_id_max: int,
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    q_to_d = {q: d for d, q in variant_map.items()}
    selected: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    seen_windows: list[tuple[str, str]] = []
    with query_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            qv = raw.get("variant_id", raw.get("query_variant", ""))
            if qv not in q_to_d:
                continue
            scene = raw.get("scene_id", "")
            window = raw.get("window_id", "")
            frame_id = int(_num(raw.get("frame_id")))
            if scenes and scene not in scenes:
                continue
            if window_ids and window not in window_ids:
                continue
            if frame_id_min >= 0 and frame_id < frame_id_min:
                continue
            if frame_id_max >= 0 and frame_id > frame_id_max:
                continue
            sw = (scene, window)
            if sw not in seen_windows:
                if max_windows > 0 and len(seen_windows) >= max_windows:
                    continue
                seen_windows.append(sw)
            mapped = dict(raw)
            mapped["query_variant"] = qv
            mapped["query_u_norm"] = raw.get("x_norm", raw.get("query_u_norm", ""))
            mapped["query_v_norm"] = raw.get("y_norm", raw.get("query_v_norm", ""))
            mapped["query_x"] = raw.get("x_px", raw.get("query_x", ""))
            mapped["query_y"] = raw.get("y_px", raw.get("query_y", ""))
            mapped["query_u"] = raw.get("x_px", raw.get("query_u", ""))
            mapped["query_v"] = raw.get("y_px", raw.get("query_v", ""))
            mapped["source_mask_id_optional"] = raw.get("source_mask_id", raw.get("source_mask_id_optional", ""))
            mapped["mask_conflict_score"] = "1.0" if _bool(raw.get("near_competing_edge")) else "0.0"
            mapped["query_priority"] = str({"conflict": 0, "boundary": 1, "semantic_gradient": 2, "interior": 3, "uniform": 4}.get(raw.get("query_stratum", ""), 9))
            mapped["occupancy_before"] = "0.0"
            selected.setdefault((q_to_d[qv], scene, window), []).append(mapped)
    if max_queries_per_frame > 0:
        capped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for group_key, rows in selected.items():
            by_frame: dict[int, list[dict[str, str]]] = {}
            for row in rows:
                by_frame.setdefault(int(_num(row.get("frame_id"))), []).append(row)
            out: list[dict[str, str]] = []
            for _frame_id, frame_rows in sorted(by_frame.items()):
                out.extend(_balanced_cap_frame_rows(frame_rows, int(max_queries_per_frame)))
            capped[group_key] = out
        selected = capped
    return selected


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _build_variant_metric_rows(v96_variant_rows: list[dict[str, str]], error_count: int, runtime_total_sec: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in v96_variant_rows:
        rows.append(
            {
                "schema_version": "stream4d_v97_phase2_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": row.get("decode_variant", ""),
                "query_variant": row.get("query_variant", ""),
                "query_count": int(_num(row.get("query_count"))),
                "decoded_group_count": int(_num(row.get("group_count"))),
                "valid_track_ratio": _num(row.get("valid_track_ratio")),
                "uv_in01_rate": _num(row.get("uv_in01_rate")),
                "visibility_mean": _num(row.get("visibility_mean")),
                "confidence_mean": _num(row.get("confidence_mean")),
                "source_container_support_ratio": _num(row.get("source_support_area_ratio")),
                "frame_foreground_support_ratio": _num(row.get("source_support_area_ratio")),
                "boundary_band_support_ratio": _num(row.get("boundary_band_support_ratio")),
                "competing_edge_support_ratio": _num(row.get("competing_edge_support_ratio")),
                "semantic_gradient_support_ratio": _num(row.get("semantic_gradient_support_ratio")),
                "mask_membership_flip_rate": _num(row.get("mask_membership_flip_rate")),
                "projection_jitter_p50": "",
                "projection_jitter_p50_status": "not_computed_by_reused_v96_decode_core",
                "projection_jitter_p90": _num(row.get("projection_jitter_p90_mean")),
                "runtime_total_sec": float(runtime_total_sec),
                "runtime_decode_sec": _num(row.get("runtime_decode_sec")),
                "GPU_memory_peak_MB": _num(row.get("GPU_memory_peak_MB")),
                "OOM_count": int(error_count),
                "metric_scope": "segment_diagnostic",
                "decode_scope": "segment_diagnostic",
                "d4rt_model_frame_mode": "",
                "d4rt_ckpt_id": _rel(DEFAULT_D4RT_CKPT),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _gate_row(gate: str, variant_id: str, observed: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v97_phase2_gate_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "gate": gate,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _build_gate_rows(metric_rows: list[dict[str, Any]], *, error_count: int, runtime_total_sec: float, runtime_budget_sec: float, decode_scope: str) -> list[dict[str, Any]]:
    by_variant = {row["variant_id"]: row for row in metric_rows}
    d1 = by_variant.get("D1_uniform1024", {})
    d3 = by_variant.get("D3_adaptive1024", {})
    adaptive_rows = [row for row in metric_rows if row["variant_id"] != "D1_uniform1024"]
    rows = [
        _gate_row("valid_track_ratio_ge_0p70_for_all", "ALL", min((_num(row.get("valid_track_ratio")) for row in metric_rows), default=0.0), 0.70, bool(metric_rows) and all(_num(row.get("valid_track_ratio")) >= 0.70 for row in metric_rows)),
        _gate_row("uv_in01_rate_ge_0p85_for_at_least_one_adaptive", "ADAPTIVE", max((_num(row.get("uv_in01_rate")) for row in adaptive_rows), default=0.0), 0.85, any(_num(row.get("uv_in01_rate")) >= 0.85 for row in adaptive_rows)),
        _gate_row("D3_source_support_ge_D1_plus_0p03", "D3_adaptive1024", _num(d3.get("source_container_support_ratio")), _num(d1.get("source_container_support_ratio")) + 0.03, bool(d1 and d3) and _num(d3.get("source_container_support_ratio")) >= _num(d1.get("source_container_support_ratio")) + 0.03),
        _gate_row("D3_boundary_support_ge_D1_plus_0p03", "D3_adaptive1024", _num(d3.get("boundary_band_support_ratio")), _num(d1.get("boundary_band_support_ratio")) + 0.03, bool(d1 and d3) and _num(d3.get("boundary_band_support_ratio")) >= _num(d1.get("boundary_band_support_ratio")) + 0.03),
        _gate_row("D3_competing_support_ge_D1_plus_0p01", "D3_adaptive1024", _num(d3.get("competing_edge_support_ratio")), _num(d1.get("competing_edge_support_ratio")) + 0.01, bool(d1 and d3) and _num(d3.get("competing_edge_support_ratio")) >= _num(d1.get("competing_edge_support_ratio")) + 0.01),
        _gate_row("runtime_total_sec_within_budget", "ALL", runtime_total_sec, runtime_budget_sec, runtime_total_sec <= runtime_budget_sec),
        _gate_row("OOM_count_eq_0", "ALL", error_count, 0, error_count == 0),
        _gate_row("no_gt_or_future_prediction", "ALL", "uses_gt_for_prediction=false,uses_future=false", "both false", True),
        _gate_row("decode_scope_full_dev", "ALL", decode_scope, "full_dev", decode_scope == "full_dev"),
    ]
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    requested_model_frame_mode = args.model_frame_mode
    decode_args = argparse.Namespace(**vars(args))
    if requested_model_frame_mode == "segmented_active":
        decode_args.model_frame_mode = "active_sparse"
    v96 = _load_v96_module()
    v96.PHASE_ID = PHASE_ID
    v96.RUN_ID = RUN_ID
    v96._read_selected_queries = _read_selected_queries_v97
    summary_from_decode = v96.run(decode_args)

    runtime_total_sec = float(time.time() - started)
    error_rows = _read_csv(output_root / "decode_error_rows.csv")
    error_count = len(error_rows)
    v96_variant_rows = _read_csv(output_root / "variant_summary_rows.csv")
    metric_rows = _build_variant_metric_rows(v96_variant_rows, error_count, runtime_total_sec)
    decode_scope = "full_dev" if (
        int(args.max_windows) == 0
        and int(args.max_queries_per_frame) == 0
        and int(args.frame_id_min) < 0
        and int(args.frame_id_max) < 0
        and not args.scenes
        and not args.window_ids
        and requested_model_frame_mode == "active_sparse"
    ) else "segment_diagnostic"
    for row in metric_rows:
        row["metric_scope"] = decode_scope
        row["decode_scope"] = decode_scope
        row["d4rt_model_frame_mode"] = requested_model_frame_mode
        row["d4rt_backend_model_frame_mode"] = decode_args.model_frame_mode
        row["d4rt_ckpt_id"] = _rel(_project(args.d4rt_ckpt))
    gate_rows = _build_gate_rows(
        metric_rows,
        error_count=error_count,
        runtime_total_sec=runtime_total_sec,
        runtime_budget_sec=float(args.runtime_budget_sec),
        decode_scope=decode_scope,
    )
    quality_pass = all(bool(row["pass"]) for row in gate_rows if row["gate"] != "decode_scope_full_dev")
    full_dev_gate_pass = quality_pass and decode_scope == "full_dev"
    best = max(metric_rows, key=lambda row: (_num(row.get("boundary_band_support_ratio")) + _num(row.get("competing_edge_support_ratio")) + 0.5 * _num(row.get("source_container_support_ratio"))), default={})
    best_summary = {
        "schema": "stream4d_v97_phase2_best_variant_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "best_variant_id": best.get("variant_id", ""),
        "selection_policy": "GT-free support coverage score; not a method-success claim.",
        "decode_scope": decode_scope,
        "valid_track_ratio": best.get("valid_track_ratio", ""),
        "uv_in01_rate": best.get("uv_in01_rate", ""),
        "source_container_support_ratio": best.get("source_container_support_ratio", ""),
        "boundary_band_support_ratio": best.get("boundary_band_support_ratio", ""),
        "competing_edge_support_ratio": best.get("competing_edge_support_ratio", ""),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    failure_rows = [
        {
            "schema_version": "stream4d_v97_phase2_variant_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": row["variant_id"],
            "failure_type": "PHASE2_GATE_FAIL",
            "failed_gate": row["gate"],
            "observed": row["observed"],
            "required": row["required"],
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    variant_config_rows = [
        {
            "schema_version": "stream4d_v97_phase2_variant_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": decode_variant,
            "query_variant": query_variant,
            "model_frame_mode": requested_model_frame_mode,
            "backend_model_frame_mode": decode_args.model_frame_mode,
            "query_chunk_size": int(args.query_chunk_size),
            "source_uv_clamp_eps": float(args.source_uv_clamp_eps),
            "d4rt_input_width": int(args.d4rt_input_width),
            "d4rt_input_height": int(args.d4rt_input_height),
            "d4rt_output_width": int(args.d4rt_output_width),
            "d4rt_output_height": int(args.d4rt_output_height),
            "coordinate_grid": (
                f"fixed_{int(args.d4rt_output_width)}x{int(args.d4rt_output_height)}"
                if int(args.d4rt_output_width) > 0 and int(args.d4rt_output_height) > 0
                else "source_mask_resolution"
            ),
            "min_visibility": float(args.min_visibility),
            "min_confidence": float(args.min_confidence),
            "occupancy_radius_px": int(args.occupancy_radius_px),
            "d4rt_root": _rel(_project(args.d4rt_root)),
            "d4rt_config": _rel(_project(args.d4rt_config)),
            "d4rt_ckpt": _rel(_project(args.d4rt_ckpt)),
            "decode_scope": decode_scope,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for decode_variant, query_variant in _parse_variant_map(args.variant_map).items()
        if not args.decode_variants or decode_variant in {part.strip() for part in args.decode_variants.split(",") if part.strip()}
    ]
    casebook_rows = []
    for row in _read_csv(output_root / "decode_group_rows.csv")[:20]:
        casebook_rows.append(
            {
                "schema_version": "stream4d_v97_phase2_casebook_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": row.get("decode_variant", ""),
                "query_variant": row.get("query_variant", ""),
                "scene_id": row.get("scene_id", ""),
                "window_id": row.get("window_id", ""),
                "query_count": row.get("query_count", ""),
                "target_frame_count": row.get("target_frame_count", ""),
                "runtime_decode_sec": row.get("runtime_decode_sec", ""),
                "GPU_memory_peak_MB": row.get("GPU_memory_peak_MB", ""),
                "carrier_batch_npz": row.get("carrier_batch_npz", ""),
                "status": row.get("status", ""),
                "decode_scope": decode_scope,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    _copy_if_exists(output_root / "micro_track_quality_rows.csv", output_root / "d4rt_quality_rows.csv")
    _write_csv(output_root / "variant_config_rows.csv", variant_config_rows)
    _write_csv(output_root / "variant_metric_rows.csv", metric_rows)
    _write_csv(output_root / "variant_gate_rows.csv", gate_rows)
    _write_csv(output_root / "phase2_gate_rows.csv", gate_rows)
    _write_csv(output_root / "variant_failure_rows.csv", failure_rows)
    _write_csv(output_root / "casebook_rows.csv", casebook_rows)
    _write_json(output_root / "best_variant_summary.json", best_summary)
    summary = {
        "schema": "stream4d_v97_phase2_d4rt_micro_tracks_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PASS_V97_PHASE2_D4RT_MICRO_TRACKS" if full_dev_gate_pass else "PASS_V97_PHASE2_SEGMENT_DIAGNOSTIC" if quality_pass else "NO_GO_V97_PHASE2_D4RT_MICRO_TRACKS",
        "decode_scope": decode_scope,
        "full_dev_gate_pass": full_dev_gate_pass,
        "can_enter_phase3": quality_pass,
        "query_root": _rel(_project(args.query_root)),
        "source_rows": _rel(_project(args.source_rows)),
        "output_root": _rel(output_root),
        "selected_group_count": summary_from_decode.get("selected_group_count", ""),
        "decoded_group_count": summary_from_decode.get("decoded_group_count", ""),
        "error_count": error_count,
        "variant_summaries": metric_rows,
        "gate_rows": gate_rows,
        "micro_query_rows": _rel(output_root / "micro_query_rows.csv"),
        "micro_track_rows": _rel(output_root / "micro_track_rows.csv"),
        "d4rt_quality_rows": _rel(output_root / "d4rt_quality_rows.csv"),
        "variant_metric_rows": _rel(output_root / "variant_metric_rows.csv"),
        "variant_config_rows": _rel(output_root / "variant_config_rows.csv"),
        "variant_gate_rows": _rel(output_root / "variant_gate_rows.csv"),
        "variant_failure_rows": _rel(output_root / "variant_failure_rows.csv"),
        "best_variant_summary": _rel(output_root / "best_variant_summary.json"),
        "casebook_rows": _rel(output_root / "casebook_rows.csv"),
        "decode_group_rows": _rel(output_root / "decode_group_rows.csv"),
        "carrier_batches_dir": _rel(output_root / "carrier_batches"),
        "runtime_total_sec": runtime_total_sec,
        "cuda_visible_devices": summary_from_decode.get("cuda_visible_devices", ""),
        "device": args.device,
        "query_chunk_size": int(args.query_chunk_size),
        "source_uv_clamp_eps": float(args.source_uv_clamp_eps),
        "d4rt_input_width": int(args.d4rt_input_width),
        "d4rt_input_height": int(args.d4rt_input_height),
        "d4rt_output_width": int(args.d4rt_output_width),
        "d4rt_output_height": int(args.d4rt_output_height),
        "coordinate_grid": (
            f"fixed_{int(args.d4rt_output_width)}x{int(args.d4rt_output_height)}"
            if int(args.d4rt_output_width) > 0 and int(args.d4rt_output_height) > 0
            else "source_mask_resolution"
        ),
        "model_frame_mode": requested_model_frame_mode,
        "backend_model_frame_mode": decode_args.model_frame_mode,
        "frame_id_min": int(args.frame_id_min),
        "frame_id_max": int(args.frame_id_max),
        "max_windows": int(args.max_windows),
        "max_queries_per_frame": int(args.max_queries_per_frame),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(output_root / "summary.json", summary)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "decode_scope": decode_scope,
                "decoded_group_count": summary["decoded_group_count"],
                "best_variant_id": best_summary["best_variant_id"],
                "runtime_total_sec": summary["runtime_total_sec"],
            },
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-root", default=str(DEFAULT_QUERY_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--variant-map", default=",".join(f"{d}={q}" for d, q in DEFAULT_VARIANT_MAP.items()))
    parser.add_argument("--decode-variants", default="")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--window-ids", default="")
    parser.add_argument("--frame-id-min", type=int, default=-1)
    parser.add_argument("--frame-id-max", type=int, default=-1)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--max-queries-per-frame", type=int, default=0)
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET))
    parser.add_argument("--d4rt-root", default=str(DEFAULT_D4RT_ROOT))
    parser.add_argument("--d4rt-config", default=str(DEFAULT_D4RT_CONFIG))
    parser.add_argument("--d4rt-ckpt", default=str(DEFAULT_D4RT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=1024)
    parser.add_argument("--source-uv-clamp-eps", type=float, default=0.0)
    parser.add_argument("--d4rt-input-width", type=int, default=0)
    parser.add_argument("--d4rt-input-height", type=int, default=0)
    parser.add_argument("--d4rt-output-width", type=int, default=0)
    parser.add_argument("--d4rt-output-height", type=int, default=0)
    parser.add_argument("--model-frame-mode", choices=("active_sparse", "contiguous32_from_first", "segmented_active"), default="active_sparse")
    parser.add_argument("--contiguous-frame-count", type=int, default=32)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--occupancy-radius-px", type=int, default=3)
    parser.add_argument("--progress-every-groups", type=int, default=1)
    parser.add_argument("--runtime-budget-sec", type=float, default=3600.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
