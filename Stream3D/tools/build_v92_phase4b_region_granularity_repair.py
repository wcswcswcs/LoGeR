from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_IN = ROOT / "outputs/audit/v92_phase4_semantic_region_affinity"
DEFAULT_OUT = ROOT / "outputs/audit/v92_phase4b_region_granularity_coarse2"


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _group_key(row: dict[str, str], factor: int) -> tuple[str, str, str, int, int, int, int]:
    return (
        str(row.get("scene_id", "")),
        str(row.get("split", "dev")),
        str(row.get("window_id", "")),
        _int(row.get("frame_id"), -1),
        _int(row.get("source_mask_id"), -1),
        _int(row.get("feature_y"), 0) // max(1, int(factor)),
        _int(row.get("feature_x"), 0) // max(1, int(factor)),
    )


def _new_region_id(key: tuple[str, str, str, int, int, int, int]) -> str:
    scene, _split, _window, frame_id, mask_id, gy, gx = key
    return f"{scene}:{frame_id}:{mask_id}:g{gy:04d}_{gx:04d}"


def _build_nodes(input_root: Path, out: Path, factor: int) -> tuple[dict[str, str], dict[str, tuple[str, str, str, int, int, int, int]], int]:
    accum: dict[tuple[str, str, str, int, int, int, int], dict[str, Any]] = {}
    old_to_new: dict[str, str] = {}
    new_key_by_id: dict[str, tuple[str, str, str, int, int, int, int]] = {}
    input_rows = 0
    for row in _read_csv(input_root / "region_node_rows.csv"):
        input_rows += 1
        key = _group_key(row, factor)
        new_id = _new_region_id(key)
        old_to_new[str(row.get("region_id", ""))] = new_id
        new_key_by_id[new_id] = key
        area = max(1.0, _num(row.get("pixel_count"), 1.0))
        item = accum.setdefault(
            key,
            {
                "schema_version": "stream4d_v92_phase4b_coarse_region_node_v1",
                "phase_id": "v92_phase4b_region_granularity_repair",
                "run_id": f"v92_phase4b_coarse{factor}",
                "scene_id": key[0],
                "split": key[1],
                "window_id": key[2],
                "frame_id": key[3],
                "source_mask_id": key[4],
                "region_id": new_id,
                "feature_y": key[5],
                "feature_x": key[6],
                "pixel_count": 0.0,
                "bbox_x0": 10**9,
                "bbox_y0": 10**9,
                "bbox_x1": -1,
                "bbox_y1": -1,
                "centroid_x_sum": 0.0,
                "centroid_y_sum": 0.0,
                "mean_rgb_r_sum": 0.0,
                "mean_rgb_g_sum": 0.0,
                "mean_rgb_b_sum": 0.0,
                "source_mean_cosine_sum": 0.0,
                "radio_feature_norm_sum": 0.0,
                "center_distance_norm_sum": 0.0,
                "boundary_token": False,
                "diagnostic_only_uses_gt": _bool(row.get("diagnostic_only_uses_gt")),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            },
        )
        item["pixel_count"] += area
        item["bbox_x0"] = min(_int(item["bbox_x0"]), _int(row.get("bbox_x0"), 0))
        item["bbox_y0"] = min(_int(item["bbox_y0"]), _int(row.get("bbox_y0"), 0))
        item["bbox_x1"] = max(_int(item["bbox_x1"]), _int(row.get("bbox_x1"), 0))
        item["bbox_y1"] = max(_int(item["bbox_y1"]), _int(row.get("bbox_y1"), 0))
        item["centroid_x_sum"] += _num(row.get("centroid_x"), 0.0) * area
        item["centroid_y_sum"] += _num(row.get("centroid_y"), 0.0) * area
        item["mean_rgb_r_sum"] += _num(row.get("mean_rgb_r"), 0.0) * area
        item["mean_rgb_g_sum"] += _num(row.get("mean_rgb_g"), 0.0) * area
        item["mean_rgb_b_sum"] += _num(row.get("mean_rgb_b"), 0.0) * area
        item["source_mean_cosine_sum"] += _num(row.get("source_mean_cosine"), 0.0) * area
        item["radio_feature_norm_sum"] += _num(row.get("radio_feature_norm"), 0.0) * area
        item["center_distance_norm_sum"] += _num(row.get("center_distance_norm"), 0.0) * area
        item["boundary_token"] = bool(item["boundary_token"]) or _bool(row.get("boundary_token"))

    rows: list[dict[str, Any]] = []
    for idx, (key, item) in enumerate(sorted(accum.items())):
        area = max(1.0, _num(item.get("pixel_count"), 1.0))
        row = {
            "schema_version": item["schema_version"],
            "phase_id": item["phase_id"],
            "run_id": item["run_id"],
            "scene_id": item["scene_id"],
            "split": item["split"],
            "window_id": item["window_id"],
            "frame_id": item["frame_id"],
            "source_mask_id": item["source_mask_id"],
            "region_id": item["region_id"],
            "region_index": idx,
            "feature_y": item["feature_y"],
            "feature_x": item["feature_x"],
            "pixel_count": int(area),
            "bbox_x0": item["bbox_x0"],
            "bbox_y0": item["bbox_y0"],
            "bbox_x1": item["bbox_x1"],
            "bbox_y1": item["bbox_y1"],
            "centroid_x": item["centroid_x_sum"] / area,
            "centroid_y": item["centroid_y_sum"] / area,
            "radio_feature_hash": hashlib.sha256(str(item["region_id"]).encode("utf-8")).hexdigest(),
            "radio_feature_norm": item["radio_feature_norm_sum"] / area,
            "dino_feature_hash": "",
            "dino_feature_norm": "",
            "mean_rgb_r": item["mean_rgb_r_sum"] / area,
            "mean_rgb_g": item["mean_rgb_g_sum"] / area,
            "mean_rgb_b": item["mean_rgb_b_sum"] / area,
            "source_mean_cosine": item["source_mean_cosine_sum"] / area,
            "boundary_token": bool(item["boundary_token"]),
            "center_distance_norm": item["center_distance_norm_sum"] / area,
            "diagnostic_gt_id": "",
            "diagnostic_gt_fraction": "",
            "diagnostic_only_uses_gt": bool(item["diagnostic_only_uses_gt"]),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        rows.append(row)
    _write_csv(out / "region_node_rows.csv", rows)
    return old_to_new, new_key_by_id, input_rows


def _build_edges(input_root: Path, out: Path, old_to_new: dict[str, str], new_key_by_id: dict[str, tuple[str, str, str, int, int, int, int]]) -> tuple[int, int]:
    edge_accum: dict[tuple[str, str], list[float]] = defaultdict(list)
    input_edges = 0
    for row in _read_csv(input_root / "region_edge_rows.csv"):
        input_edges += 1
        a = old_to_new.get(str(row.get("region_id_a", "")))
        b = old_to_new.get(str(row.get("region_id_b", "")))
        if not a or not b or a == b:
            continue
        key = tuple(sorted((a, b)))
        edge_accum[key].append(_num(row.get("radio_cosine"), 0.0))
    rows: list[dict[str, Any]] = []
    for (a, b), vals in sorted(edge_accum.items()):
        ka = new_key_by_id[a]
        kb = new_key_by_id[b]
        if ka[:5] != kb[:5]:
            continue
        rows.append(
            {
                "schema_version": "stream4d_v92_phase4b_coarse_region_edge_v1",
                "phase_id": "v92_phase4b_region_granularity_repair",
                "run_id": "v92_phase4b_coarse2",
                "scene_id": ka[0],
                "split": ka[1],
                "window_id": ka[2],
                "frame_id": ka[3],
                "source_mask_id": ka[4],
                "region_id_a": a,
                "region_id_b": b,
                "edge_kind": "coarsened_original_radio_edge",
                "radio_cosine": sum(vals) / max(1, len(vals)),
                "radio_contrast": 1.0 - (sum(vals) / max(1, len(vals))),
                "same_diagnostic_gt": "",
                "both_foreground_gt": "",
                "diagnostic_only_uses_gt": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    _write_csv(out / "region_edge_rows.csv", rows)
    return input_edges, len(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    input_root = _resolve(args.input_root)
    out = _resolve(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    factor = max(1, int(args.factor))
    old_to_new, new_key_by_id, input_nodes = _build_nodes(input_root, out, factor)
    input_edges, output_edges = _build_edges(input_root, out, old_to_new, new_key_by_id)
    node_rows = sum(1 for _ in (out / "region_node_rows.csv").open(encoding="utf-8")) - 1
    summary = {
        "phase_id": "v92_phase4b_region_granularity_repair",
        "schema": "stream4d_v92_phase4b_region_granularity_summary_v1",
        "run_id": f"v92_phase4b_coarse{factor}",
        "input_root": _rel(input_root),
        "coarsen_factor": factor,
        "input_region_node_rows": input_nodes,
        "output_region_node_rows": node_rows,
        "input_region_edge_rows": input_edges,
        "output_region_edge_rows": output_edges,
        "region_node_reduction_ratio": float(node_rows / max(1, input_nodes)),
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    _write_json(out / "summary.json", summary)
    _write_csv(out / "semantic_failure_rows.csv", [])
    outputs = [out / "region_node_rows.csv", out / "region_edge_rows.csv", out / "semantic_failure_rows.csv", out / "summary.json"]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v92 Phase4B coarser region graph for granularity repair.")
    parser.add_argument("--input-root", default=str(DEFAULT_IN))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--factor", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
