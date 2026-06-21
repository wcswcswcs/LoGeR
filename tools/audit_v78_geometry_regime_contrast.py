#!/usr/bin/env python3
"""Quantify geometry/visibility regime differences for bad-vs-reference cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _ids_containing(label_names: list[str], words: tuple[str, ...]) -> set[int]:
    lowered = [str(x).lower() for x in label_names]
    return {idx for idx, name in enumerate(lowered) if any(word in name for word in words)}


def _mean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(np.mean(arr[finite]))


def _std_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(np.std(arr[finite]))


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    edge = np.zeros_like(m, dtype=bool)
    edge[:, 1:] |= m[:, 1:] != m[:, :-1]
    edge[:, :-1] |= m[:, 1:] != m[:, :-1]
    edge[1:, :] |= m[1:, :] != m[:-1, :]
    edge[:-1, :] |= m[1:, :] != m[:-1, :]
    return edge


def _neighbor_abs_diffs(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    diffs: list[np.ndarray] = []
    dh = np.abs(x[:, 1:] - x[:, :-1])
    dv = np.abs(x[1:, :] - x[:-1, :])
    if mask is None:
        diffs.extend([dh.reshape(-1), dv.reshape(-1)])
    else:
        m = np.asarray(mask, dtype=bool)
        mh = (m[:, 1:] != m[:, :-1]).reshape(-1)
        mv = (m[1:, :] != m[:-1, :]).reshape(-1)
        diffs.extend([dh.reshape(-1)[mh], dv.reshape(-1)[mv]])
    if not diffs:
        return np.asarray([], dtype=np.float32)
    return np.concatenate([d for d in diffs if d.size > 0]) if any(d.size > 0 for d in diffs) else np.asarray([], dtype=np.float32)


def _road_shape_features(road: np.ndarray) -> dict[str, float]:
    h, w = road.shape
    ys = np.arange(h, dtype=np.float32)
    keep = ys >= float(h) * 0.35
    centers: list[float] = []
    widths: list[float] = []
    for y in np.where(keep)[0].tolist():
        xs = np.where(road[int(y)])[0]
        if xs.size < max(3, int(0.03 * w)):
            continue
        centers.append(float(np.mean(xs) / max(1, w - 1)))
        widths.append(float(xs.size / max(1, w)))
    if len(centers) < 5:
        return {
            "road_center_range": float("nan"),
            "road_center_second_diff_std": float("nan"),
            "road_width_mean": _mean_or_nan(np.asarray(widths)),
            "road_width_std": _std_or_nan(np.asarray(widths)),
        }
    centers_arr = np.asarray(centers, dtype=np.float32)
    widths_arr = np.asarray(widths, dtype=np.float32)
    second = np.diff(centers_arr, n=2) if centers_arr.size >= 3 else np.asarray([], dtype=np.float32)
    return {
        "road_center_range": float(np.max(centers_arr) - np.min(centers_arr)),
        "road_center_second_diff_std": _std_or_nan(second),
        "road_width_mean": _mean_or_nan(widths_arr),
        "road_width_std": _std_or_nan(widths_arr),
    }


def _luminance(rgb_path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(rgb_path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _frame_features(asset: dict[str, Any], chunk_cache: dict[Path, Any]) -> dict[str, Any]:
    semantic_path = Path(asset.get("path", ""))
    rgb_path = Path(asset.get("rgb_path", ""))
    local_frame = int(asset.get("local_frame", 0))
    frame = int(asset.get("frame", -1))
    if semantic_path not in chunk_cache:
        chunk_cache[semantic_path] = _torch_load(semantic_path)
    payload = chunk_cache[semantic_path]
    sem = payload.get("semantic_segmentation", {}) if isinstance(payload, dict) else {}
    label_maps = sem.get("label_maps")
    conf_maps = sem.get("confidence_maps")
    if not torch.is_tensor(label_maps) or not rgb_path.exists():
        return {"frame": frame, "valid": False}
    local_frame = max(0, min(local_frame, int(label_maps.shape[0]) - 1))
    labels = label_maps[local_frame].detach().cpu().long().numpy()
    conf = (
        conf_maps[local_frame].detach().cpu().float().numpy()
        if torch.is_tensor(conf_maps)
        else np.ones_like(labels, dtype=np.float32)
    )
    label_names = [str(x) for x in sem.get("label_names", [])]
    road_ids = _ids_containing(label_names, ("road", "ground"))
    dynamic_ids = _ids_containing(label_names, ("person", "car", "truck", "bus", "bicycle", "motorcycle"))
    vegetation_ids = _ids_containing(label_names, ("grass", "tree", "vegetation", "mountain"))
    static_ids = _ids_containing(label_names, ("wall", "fence", "pole", "building", "house", "bridge", "construction"))
    road = np.isin(labels, list(road_ids)) if road_ids else np.zeros_like(labels, dtype=bool)
    dynamic = np.isin(labels, list(dynamic_ids)) if dynamic_ids else np.zeros_like(labels, dtype=bool)
    vegetation = np.isin(labels, list(vegetation_ids)) if vegetation_ids else np.zeros_like(labels, dtype=bool)
    static = np.isin(labels, list(static_ids)) if static_ids else np.zeros_like(labels, dtype=bool)
    road_edge = _edge_mask(road)
    semantic_edge = _edge_mask(labels)
    lum = _luminance(rgb_path, (labels.shape[1], labels.shape[0]))
    road_conf_diffs = _neighbor_abs_diffs(conf, road)
    lum_road_edge_diffs = _neighbor_abs_diffs(lum, road)
    out = {
        "frame": frame,
        "valid": True,
        "road_frac": float(np.mean(road)),
        "dynamic_frac": float(np.mean(dynamic)),
        "vegetation_frac": float(np.mean(vegetation)),
        "static_frac": float(np.mean(static)),
        "road_edge_density": float(np.mean(road_edge)),
        "semantic_boundary_density": float(np.mean(semantic_edge)),
        "road_edge_confidence_mean": _mean_or_nan(conf[road_edge]),
        "road_edge_confidence_std": _std_or_nan(conf[road_edge]),
        "road_boundary_confidence_discontinuity": _mean_or_nan(road_conf_diffs),
        "road_boundary_luminance_discontinuity": _mean_or_nan(lum_road_edge_diffs),
        "luminance_mean": float(np.mean(lum)),
        "luminance_std": float(np.std(lum)),
        "dark_frac": float(np.mean(lum < 0.25)),
        "bright_frac": float(np.mean(lum > 0.85)),
        "confidence_mean": float(np.mean(conf)),
        "confidence_std": float(np.std(conf)),
        "low_conf_frac": float(np.mean(conf < 0.55)),
    }
    out.update(_road_shape_features(road))
    return out


def _aggregate_frame_features(frame_rows: list[dict[str, Any]]) -> dict[str, float]:
    valid = [row for row in frame_rows if row.get("valid")]
    metadata_keys = {
        "frame",
        "valid",
        "family",
        "contrast_rank",
        "role",
        "run",
        "sequence",
        "case_id",
    }
    keys = sorted(
        {
            key
            for row in valid
            for key, value in row.items()
            if key not in metadata_keys and _finite(value) is not None
        }
    )
    out: dict[str, float] = {"valid_frame_count": float(len(valid))}
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in valid if _finite(row.get(key)) is not None],
            dtype=np.float64,
        )
        out[f"{key}_mean"] = _mean_or_nan(values)
        out[f"{key}_std"] = _std_or_nan(values)
        if values.size:
            out[f"{key}_temporal_range"] = float(np.nanmax(values) - np.nanmin(values))
        else:
            out[f"{key}_temporal_range"] = float("nan")
    return out


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contrast_rows = _read_csv(args.contrast_csv)
    chunk_cache: dict[Path, Any] = {}
    frame_feature_rows: list[dict[str, Any]] = []
    case_feature_rows: list[dict[str, Any]] = []

    for row in contrast_rows:
        try:
            assets = json.loads(row.get("source_assets", "[]"))
        except json.JSONDecodeError:
            assets = []
        frame_rows = []
        for asset in assets:
            features = _frame_features(asset, chunk_cache)
            features.update(
                {
                    "family": row.get("family", ""),
                    "contrast_rank": row.get("contrast_rank", ""),
                    "role": row.get("role", ""),
                    "run": row.get("run", ""),
                    "sequence": row.get("sequence", ""),
                    "case_id": row.get("case_id", ""),
                }
            )
            frame_feature_rows.append(features)
            frame_rows.append(features)
        agg = _aggregate_frame_features(frame_rows)
        out = {
            "family": row.get("family", ""),
            "contrast_rank": row.get("contrast_rank", ""),
            "role": row.get("role", ""),
            "run": row.get("run", ""),
            "sequence": row.get("sequence", ""),
            "case_id": row.get("case_id", ""),
            "metric": row.get("metric", ""),
            "case_metric_value": row.get("bad_metric_value") if row.get("role") == "bad" else row.get("reference_metric_value"),
            "reference_strategy": row.get("reference_strategy", ""),
        }
        out.update(agg)
        case_feature_rows.append(out)

    deltas: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in case_feature_rows:
        by_pair.setdefault((row["family"], str(row["contrast_rank"])), {})[row["role"]] = row
    for (family, rank), pair in sorted(by_pair.items()):
        bad = pair.get("bad")
        ref = pair.get("reference")
        if not bad or not ref:
            continue
        out = {
            "family": family,
            "contrast_rank": rank,
            "bad_case": bad.get("case_id", ""),
            "reference_case": ref.get("case_id", ""),
            "metric": bad.get("metric", ""),
            "bad_metric_value": bad.get("case_metric_value", ""),
            "reference_metric_value": ref.get("case_metric_value", ""),
            "reference_strategy": bad.get("reference_strategy", ""),
        }
        for key in sorted(k for k in bad if k.endswith("_mean") or k.endswith("_std") or k.endswith("_temporal_range")):
            b = _finite(bad.get(key))
            r = _finite(ref.get(key))
            if b is not None and r is not None:
                out[f"delta_{key}"] = b - r
        deltas.append(out)

    _write_csv(args.out_dir / "geometry_regime_frame_features.csv", frame_feature_rows)
    _write_csv(args.out_dir / "geometry_regime_case_features.csv", case_feature_rows)
    _write_csv(args.out_dir / "geometry_regime_feature_deltas.csv", deltas)

    feature_votes: dict[str, dict[str, int]] = {}
    for row in deltas:
        for key, value in row.items():
            if not key.startswith("delta_"):
                continue
            val = _finite(value)
            if val is None:
                continue
            stats = feature_votes.setdefault(key, {"positive": 0, "negative": 0, "zero": 0})
            if val > 0:
                stats["positive"] += 1
            elif val < 0:
                stats["negative"] += 1
            else:
                stats["zero"] += 1
    summary = {
        "schema": "acl2_v78_geometry_regime_contrast_v1",
        "diagnostic_only": True,
        "contrast_csv": str(args.contrast_csv),
        "out_dir": str(args.out_dir),
        "num_contrast_case_rows": len(contrast_rows),
        "num_frame_feature_rows": len(frame_feature_rows),
        "num_case_feature_rows": len(case_feature_rows),
        "num_delta_rows": len(deltas),
        "feature_direction_votes": feature_votes,
        "note": "Positive delta means bad case has a larger feature value than its reference case. These are correlations, not causal proof.",
    }
    _write_json(args.out_dir / "geometry_regime_contrast_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
