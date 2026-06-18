from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


RAW_REQUIRED_FIELDS = (
    "carrier_id",
    "src_frame",
    "src_frame_global",
    "src_xy",
    "src_uv",
    "src_mask_id",
    "uv_pred",
    "visibility_prob",
    "confidence_prob",
    "valid",
    "xyz_ref",
    "xyz_local",
    "persistent_tube_id",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _field_present_for_tubes(data: np.lib.npyio.NpzFile, field: str, tube_count: int) -> np.ndarray:
    if field not in data:
        return np.zeros((tube_count,), dtype=bool)
    arr = np.asarray(data[field])
    if field in {"uv_pred", "visibility_prob", "confidence_prob", "valid", "xyz_ref", "xyz_local"}:
        if arr.ndim < 2 or int(arr.shape[1]) != int(tube_count):
            return np.zeros((tube_count,), dtype=bool)
        finite = np.isfinite(arr) if arr.dtype.kind in {"f", "c"} else np.ones(arr.shape, dtype=bool)
        axes = tuple(axis for axis in range(finite.ndim) if axis != 1)
        return np.any(finite, axis=axes)
    if arr.shape[0] != int(tube_count):
        return np.zeros((tube_count,), dtype=bool)
    if arr.dtype.kind in {"f", "c"}:
        finite = np.isfinite(arr)
        if finite.ndim > 1:
            return np.all(finite, axis=tuple(range(1, finite.ndim)))
        return finite
    return np.ones((tube_count,), dtype=bool)


def _source_xy_valid(data: np.lib.npyio.NpzFile, tube_count: int) -> np.ndarray:
    if "src_xy" not in data:
        return np.zeros((tube_count,), dtype=bool)
    xy = np.asarray(data["src_xy"])
    if xy.shape != (tube_count, 2):
        return np.zeros((tube_count,), dtype=bool)
    return np.all(xy >= 0, axis=1)


def _default_range_id_detected(carrier_id: np.ndarray) -> bool:
    ids = np.asarray(carrier_id, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return False
    if np.array_equal(ids, np.arange(ids.size, dtype=np.int64)):
        return True
    if np.array_equal(ids, np.arange(1, ids.size + 1, dtype=np.int64)):
        return True
    diffs = np.diff(np.sort(ids))
    return bool(ids.size > 8 and diffs.size and np.all(diffs == 1) and int(ids.max() - ids.min()) == ids.size - 1)


def _audit_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        tube_count = int(data["uv_pred"].shape[1]) if "uv_pred" in data and np.asarray(data["uv_pred"]).ndim >= 2 else 0
        row: dict[str, Any] = {
            "scene": path.parent.name,
            "file": path.name,
            "tube_count": int(tube_count),
            "frame_count": int(data["uv_pred"].shape[0]) if "uv_pred" in data and np.asarray(data["uv_pred"]).ndim >= 1 else 0,
            "field_missing_reason": "",
            "derived_from": "",
            "is_placeholder": False,
        }
        for field in RAW_REQUIRED_FIELDS:
            present = _field_present_for_tubes(data, field, tube_count)
            row[f"{field}_present_ratio"] = float(np.mean(present)) if present.size else 0.0
        row["src_xy_valid_ratio"] = float(np.mean(_source_xy_valid(data, tube_count))) if tube_count else 0.0
        if "carrier_id" in data and tube_count:
            ids = np.asarray(data["carrier_id"], dtype=np.int64).reshape(-1)
            row["carrier_id_collision_count"] = int(ids.size - np.unique(ids).size)
            row["default_range_id_detected"] = _default_range_id_detected(ids)
        else:
            row["carrier_id_collision_count"] = int(tube_count)
            row["default_range_id_detected"] = True
        missing = [field for field in RAW_REQUIRED_FIELDS if float(row[f"{field}_present_ratio"]) < 1.0]
        if missing:
            row["field_missing_reason"] = "missing_or_malformed:" + ",".join(missing)
        return row


def audit_raw_cache(cache_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_dir in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        for npz_path in sorted(scene_dir.glob("carriers_window*.npz")):
            rows.append(_audit_npz(npz_path))
    total_tubes = int(sum(int(row["tube_count"]) for row in rows))
    summary: dict[str, Any] = {
        "cache_root": str(cache_root),
        "scene_count": int(len({row["scene"] for row in rows})),
        "window_count": int(len(rows)),
        "tube_count": int(total_tubes),
        "carrier_id_collision_count": int(sum(int(row["carrier_id_collision_count"]) for row in rows)),
        "default_range_id_detected": bool(any(bool(row["default_range_id_detected"]) for row in rows)),
    }
    for field in RAW_REQUIRED_FIELDS:
        if total_tubes:
            summary[f"{field}_present_ratio"] = float(
                sum(float(row[f"{field}_present_ratio"]) * int(row["tube_count"]) for row in rows) / total_tubes
            )
        else:
            summary[f"{field}_present_ratio"] = 0.0
    summary["src_xy_valid_ratio"] = (
        float(sum(float(row["src_xy_valid_ratio"]) * int(row["tube_count"]) for row in rows) / total_tubes)
        if total_tubes
        else 0.0
    )
    summary["phase_a_raw_pass"] = bool(
        summary["src_xy_valid_ratio"] >= 0.99
        and summary["xyz_local_present_ratio"] >= 0.99
        and summary["carrier_id_collision_count"] == 0
        and not summary["default_range_id_detected"]
    )
    return rows, summary


def _audit_tube_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    has_xyz_canonical = payload.get("xyz_canonical") is not None
    alignment_quality = payload.get("alignment_quality") or {}
    return {
        "scene": path.parent.name,
        "file": path.name,
        "tube_id": int(payload.get("tube_id", -1)),
        "has_xyz_canonical": bool(has_xyz_canonical),
        "allow_metric_merge": bool(payload.get("allow_metric_merge", False)),
        "coordinate_frame": str(payload.get("coordinate_frame", "unknown")),
        "alignment_source": str(payload.get("alignment_source", "unknown")),
        "alignment_pass_gate": bool(alignment_quality.get("pass_gate", False)),
        "has_transform": payload.get("T_chunk_to_canonical") is not None,
        "has_source_xy": payload.get("source_xy") is not None,
        "has_xyz_local": payload.get("xyz_local") is not None,
        "has_xyz_ref0": payload.get("xyz_ref0") is not None,
    }


def audit_tube_records(record_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(record_root.rglob("*tube_records*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(_audit_tube_record(path, json.loads(line)))
    total = int(len(rows))
    summary = {
        "record_root": str(record_root),
        "tube_record_count": int(total),
        "xyz_canonical_present_ratio": float(np.mean([row["has_xyz_canonical"] for row in rows])) if rows else 0.0,
        "allow_metric_merge_ratio": float(np.mean([row["allow_metric_merge"] for row in rows])) if rows else 0.0,
        "alignment_pass_gate_ratio": float(np.mean([row["alignment_pass_gate"] for row in rows])) if rows else 0.0,
        "unknown_coordinate_frame_count": int(sum(row["coordinate_frame"] == "unknown" for row in rows)),
        "eval_alignment_allowed_count": int(
            sum(row["alignment_source"] == "eval_gt_sim3" and row["allow_metric_merge"] for row in rows)
        ),
    }
    summary["phase_a_record_pass"] = bool(
        summary["xyz_canonical_present_ratio"] >= 0.95
        and summary["unknown_coordinate_frame_count"] == 0
        and summary["eval_alignment_allowed_count"] == 0
    )
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_outputs(output_root: Path, prefix: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / f"{prefix}_rows.csv", rows)
    (output_root / f"{prefix}_rows.json").write_text(json.dumps(_json_safe(rows), indent=2, sort_keys=True), encoding="utf-8")
    (output_root / f"{prefix}_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit v26 raw carrier cache and optional canonical TubeRecord completeness.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tube-record-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", default="v26_cache")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    raw_rows, raw_summary = audit_raw_cache(Path(args.cache_root))
    _write_outputs(output_root, f"{args.label}_raw", raw_rows, raw_summary)
    payload: dict[str, Any] = {"raw": raw_summary}
    if args.tube_record_root:
        record_rows, record_summary = audit_tube_records(Path(args.tube_record_root))
        _write_outputs(output_root, f"{args.label}_records", record_rows, record_summary)
        payload["records"] = record_summary
    (output_root / f"{args.label}_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
