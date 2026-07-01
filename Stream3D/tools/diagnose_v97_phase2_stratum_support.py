#!/usr/bin/env python3
"""Diagnose v97 Phase2 support by query stratum from saved D4RT carrier batches."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v97_phase2_stratum_support_diagnostic"
RUN_ID = "v97_phase2_stratum_support_diagnostic"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase2_stratum_support_relseg_size7_D1_D3"


def _load_v96_module() -> Any:
    path = ROOT / "tools/build_v96_phase2_d4rt_micro_tracks.py"
    spec = importlib.util.spec_from_file_location("_stream4d_v96_phase2_decode_for_v97_stratum", path)
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
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _carrier_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "outputs":
        return ROOT / p
    return _project(p)


def _load_query_strata(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    path = root / "micro_query_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[row.get("query_id", "")] = row.get("query_stratum", "")
    return mapping


def _init_accumulator() -> dict[str, Any]:
    return {
        "query_instance_count": 0,
        "track_cell_count": 0,
        "valid_cell_count": 0,
        "uv_in01_count": 0,
        "accepted_count": 0,
        "visibility_sum": 0.0,
        "visibility_count": 0,
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "covered_pixel_count": 0,
        "source_foreground_pixel_count": 0,
        "covered_source_pixel_count": 0,
        "boundary_pixel_count": 0,
        "covered_boundary_pixel_count": 0,
    }


def _add_track_stats(acc: dict[str, Any], valid: np.ndarray, in01: np.ndarray, accepted: np.ndarray, visibility: np.ndarray, confidence: np.ndarray) -> None:
    acc["track_cell_count"] += int(valid.size)
    acc["valid_cell_count"] += int(np.count_nonzero(valid))
    acc["uv_in01_count"] += int(np.count_nonzero(in01))
    acc["accepted_count"] += int(np.count_nonzero(accepted))
    if np.any(valid):
        acc["visibility_sum"] += float(np.sum(visibility[valid]))
        acc["visibility_count"] += int(np.count_nonzero(valid))
        acc["confidence_sum"] += float(np.sum(confidence[valid]))
        acc["confidence_count"] += int(np.count_nonzero(valid))


def _row_from_acc(
    *,
    variant_id: str,
    query_variant: str,
    query_stratum: str,
    segment_source_root_count: int,
    group_count: int,
    acc: dict[str, Any],
    occupancy_radius_px: int,
    missing_mask_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v97_phase2_stratum_support_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "query_variant": query_variant,
        "query_stratum": query_stratum,
        "segment_source_root_count": segment_source_root_count,
        "decode_group_count": group_count,
        "query_instance_count": acc["query_instance_count"],
        "track_cell_count": acc["track_cell_count"],
        "valid_track_ratio": acc["valid_cell_count"] / max(1, acc["track_cell_count"]),
        "uv_in01_rate": acc["uv_in01_count"] / max(1, acc["track_cell_count"]),
        "accepted_track_rate": acc["accepted_count"] / max(1, acc["track_cell_count"]),
        "visibility_mean": acc["visibility_sum"] / max(1, acc["visibility_count"]),
        "confidence_mean": acc["confidence_sum"] / max(1, acc["confidence_count"]),
        "covered_pixel_count": acc["covered_pixel_count"],
        "source_foreground_pixel_count": acc["source_foreground_pixel_count"],
        "covered_source_pixel_count": acc["covered_source_pixel_count"],
        "source_support_area_ratio": acc["covered_source_pixel_count"] / max(1, acc["source_foreground_pixel_count"]),
        "boundary_pixel_count": acc["boundary_pixel_count"],
        "covered_boundary_pixel_count": acc["covered_boundary_pixel_count"],
        "boundary_band_support_ratio": acc["covered_boundary_pixel_count"] / max(1, acc["boundary_pixel_count"]),
        "occupancy_radius_px": occupancy_radius_px,
        "missing_mask_count": missing_mask_count,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    v96 = _load_v96_module()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_roots = [_project(part.strip()) for part in args.include_roots.split(",") if part.strip()]
    mask_lookup = v96._mask_path_lookup(_project(args.source_rows))
    radius = int(args.occupancy_radius_px)
    min_visibility = float(args.min_visibility)
    min_confidence = float(args.min_confidence)

    include_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    summary_acc: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(_init_accumulator)
    summary_group_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    variant_root_counts: dict[str, set[str]] = defaultdict(set)
    missing_mask_count = 0
    processed_group_count = 0
    processed_carrier_count = 0

    for root in source_roots:
        summary = _read_json(root / "summary.json")
        query_strata = _load_query_strata(root)
        quality_rows = _read_csv(root / "micro_track_quality_rows.csv")
        include_rows.append(
            {
                "schema_version": "stream4d_v97_phase2_stratum_support_include_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "source_root": _rel(root),
                "source_decision": summary.get("decision", ""),
                "source_decode_scope": summary.get("decode_scope", ""),
                "source_model_frame_mode": summary.get("model_frame_mode", ""),
                "source_backend_model_frame_mode": summary.get("backend_model_frame_mode", ""),
                "quality_row_count": len(quality_rows),
                "micro_query_rows": _rel(root / "micro_query_rows.csv"),
                "micro_track_quality_rows": _rel(root / "micro_track_quality_rows.csv"),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for quality in quality_rows:
            variant = quality.get("decode_variant", "")
            query_variant = quality.get("query_variant", "")
            scene = quality.get("scene_id", "")
            window = quality.get("window_id", "")
            carrier = _carrier_path(quality.get("carrier_batch_npz", ""))
            if not carrier.exists():
                missing_mask_count += 1
                continue
            with np.load(carrier, allow_pickle=True) as data:
                frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
                query_ids = np.asarray(data["query_id"]).astype(str)
                uv = np.asarray(data["uv_pred"], dtype=np.float32)[: len(frame_ids)]
                valid = np.asarray(data["valid"], dtype=bool)[: len(frame_ids)]
                visibility = np.asarray(data["visibility_prob"], dtype=np.float32)[: len(frame_ids)]
                confidence = np.asarray(data["confidence_prob"], dtype=np.float32)[: len(frame_ids)]
            in01 = valid & (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) & (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
            accepted = in01 & (visibility >= min_visibility) & (confidence >= min_confidence)
            strata_by_query = np.asarray([query_strata.get(str(qid), "unknown") for qid in query_ids], dtype=object)
            strata = sorted({str(value) for value in strata_by_query})
            masks_by_stratum: dict[str, np.ndarray] = {name: strata_by_query == name for name in strata}
            masks_by_stratum["__all__"] = np.ones((len(query_ids),), dtype=bool)
            group_rows: list[dict[str, Any]] = []
            for stratum, qmask in masks_by_stratum.items():
                acc = _init_accumulator()
                acc["query_instance_count"] = int(np.count_nonzero(qmask))
                if acc["query_instance_count"] <= 0:
                    continue
                _add_track_stats(acc, valid[:, qmask], in01[:, qmask], accepted[:, qmask], visibility[:, qmask], confidence[:, qmask])
                for local_idx, frame_id in enumerate(frame_ids):
                    mask_path = mask_lookup.get((scene, window, int(frame_id)))
                    if mask_path is None or not mask_path.exists():
                        missing_mask_count += 1
                        continue
                    label = v96._load_label(mask_path).astype(np.int64, copy=False)
                    boundary = v96._label_boundary(label)
                    h, w = label.shape
                    cover = np.zeros((h, w), dtype=np.uint8)
                    local_accept = accepted[local_idx] & qmask
                    for qidx in np.flatnonzero(local_accept):
                        x = int(np.clip(round(float(uv[local_idx, qidx, 0]) * (w - 1)), 0, w - 1))
                        y = int(np.clip(round(float(uv[local_idx, qidx, 1]) * (h - 1)), 0, h - 1))
                        v96._mark_disk(cover, y, x, radius)
                    cover_bool = cover.astype(bool)
                    foreground = label > 0
                    acc["covered_pixel_count"] += int(np.count_nonzero(cover_bool))
                    acc["source_foreground_pixel_count"] += int(np.count_nonzero(foreground))
                    acc["covered_source_pixel_count"] += int(np.count_nonzero(cover_bool & foreground))
                    acc["boundary_pixel_count"] += int(np.count_nonzero(boundary))
                    acc["covered_boundary_pixel_count"] += int(np.count_nonzero(cover_bool & boundary))
                key = (variant, query_variant, stratum)
                for acc_key, value in acc.items():
                    summary_acc[key][acc_key] += value
                summary_group_counts[key] += 1
                group_rows.append(
                    _row_from_acc(
                        variant_id=variant,
                        query_variant=query_variant,
                        query_stratum=stratum,
                        segment_source_root_count=1,
                        group_count=1,
                        acc=acc,
                        occupancy_radius_px=radius,
                        missing_mask_count=missing_mask_count,
                    )
                    | {
                        "scene_id": scene,
                        "window_id": window,
                        "source_root": _rel(root),
                        "carrier_batch_npz": _rel(carrier),
                    }
                )
            support_rows.extend(group_rows)
            variant_root_counts[variant].add(_rel(root))
            processed_group_count += 1
            processed_carrier_count += 1

    summary_rows = []
    for key, acc in sorted(summary_acc.items()):
        variant, query_variant, stratum = key
        summary_rows.append(
            _row_from_acc(
                variant_id=variant,
                query_variant=query_variant,
                query_stratum=stratum,
                segment_source_root_count=len(variant_root_counts.get(variant, set())),
                group_count=summary_group_counts[key],
                acc=acc,
                occupancy_radius_px=radius,
                missing_mask_count=missing_mask_count,
            )
        )
    by_variant_all = {row["variant_id"]: row for row in summary_rows if row["query_stratum"] == "__all__"}
    d1 = by_variant_all.get("D1_uniform1024", {})
    d3 = by_variant_all.get("D3_adaptive1024", {})
    source_delta_d3_minus_d1 = _num(d3.get("source_support_area_ratio")) - _num(d1.get("source_support_area_ratio"))
    boundary_delta_d3_minus_d1 = _num(d3.get("boundary_band_support_ratio")) - _num(d1.get("boundary_band_support_ratio"))
    summary = {
        "schema": "stream4d_v97_phase2_stratum_support_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_roots": [_rel(root) for root in source_roots],
        "processed_group_count": processed_group_count,
        "processed_carrier_count": processed_carrier_count,
        "occupancy_radius_px": radius,
        "min_visibility": min_visibility,
        "min_confidence": min_confidence,
        "missing_mask_count": missing_mask_count,
        "source_delta_d3_minus_d1": source_delta_d3_minus_d1,
        "boundary_delta_d3_minus_d1": boundary_delta_d3_minus_d1,
        "stratum_support_rows": _rel(output_root / "stratum_support_rows.csv"),
        "variant_stratum_summary_rows": _rel(output_root / "variant_stratum_summary_rows.csv"),
        "include_manifest_rows": _rel(output_root / "include_manifest_rows.csv"),
        "runtime_total_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(output_root / "include_manifest_rows.csv", include_rows)
    _write_csv(output_root / "stratum_support_rows.csv", support_rows)
    _write_csv(output_root / "variant_stratum_summary_rows.csv", summary_rows)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"output_root": _rel(output_root), "processed_group_count": processed_group_count, "source_delta_d3_minus_d1": source_delta_d3_minus_d1, "boundary_delta_d3_minus_d1": boundary_delta_d3_minus_d1}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-roots", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--occupancy-radius-px", type=int, default=3)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
