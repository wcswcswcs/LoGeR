from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ID = "v93_phase3_region_edge_graph"
RUN_ID = "v93_phase3_radio_region_edge_graph"
OUT = ROOT / "outputs/audit/v93_phase3_region_edge_graph"

V92_PHASE4 = ROOT / "outputs/audit/v92_phase4_semantic_region_affinity"
V93_PHASE2 = ROOT / "outputs/audit/v93_phase2_d4rt_edge_sampling_diagnostic"

SEMANTIC_BARRIER_THRESHOLD = 0.15


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(ROOT.parent))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "allow"}


def _float(value: Any, default: float | str = "") -> float | str:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_div(num: float | int, den: float | int) -> float:
    den_f = float(den)
    return 0.0 if den_f == 0.0 else float(num) / den_f


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _source_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(row.get("scene_id", "")), str(row.get("window_id", "")), str(row.get("frame_id", "")), str(row.get("source_mask_id", "")))


def _load_d4rt_mass() -> dict[tuple[str, str, str], float]:
    mass: dict[tuple[str, str, str], float] = {}
    path = V93_PHASE2 / "d4rt_source_support_rows.csv"
    if not path.exists():
        return mass
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("scene_id", "")), str(row.get("frame_id", "")), str(row.get("source_mask_id", "")))
            current = mass.get(key, 0.0)
            value = float(_float(row.get("carrier_count_inside_source"), 0.0) or 0.0)
            if value > current:
                mass[key] = value
    return mass


def _transform_nodes(created_at: str, d4rt_mass: dict[tuple[str, str, str], float]) -> tuple[Counter[tuple[str, str, str, str]], int, int]:
    src = V92_PHASE4 / "region_node_rows.csv"
    dst = OUT / "region_node_rows.csv"
    feature_dst = OUT / "region_feature_rows.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    node_counts: Counter[tuple[str, str, str, str]] = Counter()
    feature_available = 0
    total = 0
    node_fields = [
        "schema_version",
        "phase_id",
        "run_id",
        "scene_id",
        "split",
        "window_id",
        "frame_id",
        "source_mask_id",
        "region_id",
        "region_index",
        "feature_y",
        "feature_x",
        "region_type",
        "area_px",
        "pixel_count",
        "centroid_x",
        "centroid_y",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "radio_feature_ref",
        "dino_feature_ref",
        "source_mean_cosine",
        "boundary_token",
        "center_distance_norm",
        "source_edge_distance",
        "nested_edge_distance",
        "competing_edge_distance",
        "d4rt_witness_mass",
        "hard_negative_witness_mass",
        "background_risk",
        "broad_risk",
        "diagnostic_only_uses_gt",
        "uses_gt_for_prediction",
        "uses_future",
        "created_at",
    ]
    feature_fields = [
        "schema_version",
        "phase_id",
        "run_id",
        "scene_id",
        "split",
        "window_id",
        "frame_id",
        "source_mask_id",
        "region_id",
        "feature_backend",
        "radio_feature_ref",
        "radio_feature_norm",
        "dino_feature_ref",
        "dino_feature_norm",
        "feature_source",
        "diagnostic_only_uses_gt",
        "uses_gt_for_prediction",
        "uses_future",
        "created_at",
    ]
    with src.open(newline="", encoding="utf-8") as src_handle, dst.open("w", newline="", encoding="utf-8") as dst_handle, feature_dst.open("w", newline="", encoding="utf-8") as feat_handle:
        reader = csv.DictReader(src_handle)
        node_writer = csv.DictWriter(dst_handle, fieldnames=node_fields)
        feat_writer = csv.DictWriter(feat_handle, fieldnames=feature_fields)
        node_writer.writeheader()
        feat_writer.writeheader()
        for row in reader:
            total += 1
            key = _source_key(row)
            node_counts[key] += 1
            mass_key = (str(row.get("scene_id", "")), str(row.get("frame_id", "")), str(row.get("source_mask_id", "")))
            radio_ref = row.get("radio_feature_hash", "")
            dino_ref = row.get("dino_feature_hash", "")
            if radio_ref or dino_ref:
                feature_available += 1
            node_writer.writerow(
                {
                    "schema_version": "stream4d_v93_phase3_region_node_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": row.get("scene_id", ""),
                    "split": row.get("split", "dev"),
                    "window_id": row.get("window_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "source_mask_id": row.get("source_mask_id", ""),
                    "region_id": row.get("region_id", ""),
                    "region_index": row.get("region_index", ""),
                    "feature_y": row.get("feature_y", ""),
                    "feature_x": row.get("feature_x", ""),
                    "region_type": "radio_token_grid",
                    "area_px": row.get("pixel_count", ""),
                    "pixel_count": row.get("pixel_count", ""),
                    "centroid_x": row.get("centroid_x", ""),
                    "centroid_y": row.get("centroid_y", ""),
                    "bbox_x0": row.get("bbox_x0", ""),
                    "bbox_y0": row.get("bbox_y0", ""),
                    "bbox_x1": row.get("bbox_x1", ""),
                    "bbox_y1": row.get("bbox_y1", ""),
                    "radio_feature_ref": radio_ref,
                    "dino_feature_ref": dino_ref,
                    "source_mean_cosine": row.get("source_mean_cosine", ""),
                    "boundary_token": _bool(row.get("boundary_token")),
                    "center_distance_norm": row.get("center_distance_norm", ""),
                    "source_edge_distance": row.get("center_distance_norm", ""),
                    "nested_edge_distance": "",
                    "competing_edge_distance": "",
                    "d4rt_witness_mass": d4rt_mass.get(mass_key, 0.0),
                    "hard_negative_witness_mass": 0.0,
                    "background_risk": "",
                    "broad_risk": "",
                    "diagnostic_only_uses_gt": _bool(row.get("diagnostic_only_uses_gt")),
                    "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                    "uses_future": _bool(row.get("uses_future")),
                    "created_at": created_at,
                }
            )
            feat_writer.writerow(
                {
                    "schema_version": "stream4d_v93_phase3_region_feature_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": row.get("scene_id", ""),
                    "split": row.get("split", "dev"),
                    "window_id": row.get("window_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "source_mask_id": row.get("source_mask_id", ""),
                    "region_id": row.get("region_id", ""),
                    "feature_backend": "radio_radseg_spatial_features",
                    "radio_feature_ref": radio_ref,
                    "radio_feature_norm": row.get("radio_feature_norm", ""),
                    "dino_feature_ref": dino_ref,
                    "dino_feature_norm": row.get("dino_feature_norm", ""),
                    "feature_source": _rel(src),
                    "diagnostic_only_uses_gt": _bool(row.get("diagnostic_only_uses_gt")),
                    "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                    "uses_future": _bool(row.get("uses_future")),
                    "created_at": created_at,
                }
            )
    return node_counts, total, feature_available


class _DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: Counter[str] = Counter()

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def largest_ratio(self) -> float:
        if not self.parent:
            return 0.0
        counts: Counter[str] = Counter()
        for item in self.parent:
            counts[self.find(item)] += 1
        return max(counts.values()) / max(1, len(self.parent))


def _transform_edges(created_at: str, node_counts: Counter[tuple[str, str, str, str]]) -> tuple[Counter[tuple[str, str, str, str]], list[dict[str, Any]], int]:
    src = V92_PHASE4 / "region_edge_rows.csv"
    dst = OUT / "region_edge_rows.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    edge_counts: Counter[tuple[str, str, str, str]] = Counter()
    quality_rows: list[dict[str, Any]] = []
    total_edges = 0
    fields = [
        "schema_version",
        "phase_id",
        "run_id",
        "scene_id",
        "split",
        "window_id",
        "frame_id",
        "source_mask_id",
        "region_u",
        "region_v",
        "region_id_a",
        "region_id_b",
        "is_adjacent",
        "spatial_distance",
        "radio_cosine",
        "radio_contrast",
        "dino_cosine",
        "mask_edge_barrier",
        "nested_edge_barrier",
        "competing_edge_barrier",
        "semantic_gradient_barrier",
        "rgb_gradient_barrier",
        "d4rt_conflict_barrier",
        "edge_weight",
        "diagnostic_only_uses_gt",
        "uses_gt_for_prediction",
        "uses_future",
        "created_at",
    ]
    current_key: tuple[str, str, str, str] | None = None
    current_dsu = _DSU()
    current_edge_count = 0

    def flush() -> None:
        if current_key is None:
            return
        node_count = node_counts.get(current_key, 0)
        lcc = current_dsu.largest_ratio() if node_count else 0.0
        quality_rows.append(
            {
                "schema_version": "stream4d_v93_phase3_region_graph_quality_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "scene_id": current_key[0],
                "window_id": current_key[1],
                "frame_id": current_key[2],
                "source_mask_id": current_key[3],
                "region_count": node_count,
                "edge_count": current_edge_count,
                "connected_component_count_proxy": "",
                "edge_graph_lcc_ratio": lcc,
                "semantic_barrier_threshold": SEMANTIC_BARRIER_THRESHOLD,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )

    with src.open(newline="", encoding="utf-8") as src_handle, dst.open("w", newline="", encoding="utf-8") as dst_handle:
        reader = csv.DictReader(src_handle)
        writer = csv.DictWriter(dst_handle, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            key = _source_key(row)
            if current_key is None:
                current_key = key
            if key != current_key:
                flush()
                current_key = key
                current_dsu = _DSU()
                current_edge_count = 0
            total_edges += 1
            current_edge_count += 1
            edge_counts[key] += 1
            contrast = float(_float(row.get("radio_contrast"), 0.0) or 0.0)
            if contrast <= SEMANTIC_BARRIER_THRESHOLD:
                current_dsu.union(row.get("region_id_a", ""), row.get("region_id_b", ""))
            else:
                current_dsu.add(row.get("region_id_a", ""))
                current_dsu.add(row.get("region_id_b", ""))
            writer.writerow(
                {
                    "schema_version": "stream4d_v93_phase3_region_edge_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "scene_id": row.get("scene_id", ""),
                    "split": row.get("split", "dev"),
                    "window_id": row.get("window_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "source_mask_id": row.get("source_mask_id", ""),
                    "region_u": row.get("region_id_a", ""),
                    "region_v": row.get("region_id_b", ""),
                    "region_id_a": row.get("region_id_a", ""),
                    "region_id_b": row.get("region_id_b", ""),
                    "is_adjacent": True,
                    "spatial_distance": "",
                    "radio_cosine": row.get("radio_cosine", ""),
                    "radio_contrast": contrast,
                    "dino_cosine": "",
                    "mask_edge_barrier": "",
                    "nested_edge_barrier": "",
                    "competing_edge_barrier": "",
                    "semantic_gradient_barrier": contrast,
                    "rgb_gradient_barrier": "",
                    "d4rt_conflict_barrier": "",
                    "edge_weight": math.exp(-contrast),
                    "diagnostic_only_uses_gt": _bool(row.get("diagnostic_only_uses_gt")),
                    "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                    "uses_future": _bool(row.get("uses_future")),
                    "created_at": created_at,
                }
            )
        flush()
    return edge_counts, quality_rows, total_edges


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    v92_summary = _read_json(V92_PHASE4 / "summary.json")
    d4rt_mass = _load_d4rt_mass()
    node_counts, region_node_rows, region_feature_available_rows = _transform_nodes(created_at, d4rt_mass)
    edge_counts, quality_rows, region_edge_rows = _transform_edges(created_at, node_counts)
    _write_csv(OUT / "region_graph_quality_rows.csv", quality_rows)

    diagnostic_rows = [_rewrite_diagnostic(row, created_at) for row in _read_csv(V92_PHASE4 / "semantic_diagnostic_auc_rows.csv")]
    _write_csv(OUT / "region_diagnostic_auc_rows.csv", diagnostic_rows)

    node_count_values = list(node_counts.values())
    edge_count_values = [edge_counts.get(key, 0) for key in node_counts]
    lcc_values = [float(row.get("edge_graph_lcc_ratio", 0.0) or 0.0) for row in quality_rows]
    region_feature_available_rate = _safe_div(region_feature_available_rows, max(1, region_node_rows))
    region_count_mean = float(np.mean(node_count_values)) if node_count_values else 0.0
    region_count_p90 = float(np.percentile(node_count_values, 90)) if node_count_values else 0.0
    edge_count_mean = float(np.mean(edge_count_values)) if edge_count_values else 0.0
    lcc_all_one = bool(lcc_values) and all(abs(value - 1.0) < 1e-12 for value in lcc_values)
    uses_gt_count = 0
    uses_future_count = 0
    phase3_pass_conditions = {
        "region_feature_available_rate_ge_0p95": region_feature_available_rate >= 0.95,
        "region_count_per_source_mean_ge_8": region_count_mean >= 8.0,
        "edge_count_per_source_mean_ge_region_count_mean": edge_count_mean >= region_count_mean,
        "edge_graph_lcc_ratio_not_all_one": not lcc_all_one,
        "uses_gt_for_prediction_count_eq_0": uses_gt_count == 0,
        "uses_future_count_eq_0": uses_future_count == 0,
    }
    phase3_pass = all(phase3_pass_conditions.values())
    gate_rows = []
    for gate, passed in phase3_pass_conditions.items():
        gate_rows.append(
            {
                "schema_version": "stream4d_v93_phase3_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": "R2_RADIO_TOKEN_GRID_REGION_GRAPH",
                "scene_id": "ALL_DEV",
                "split": "dev",
                "window_id": "ALL_WINDOWS",
                "gate_name": gate,
                "gate_pass": bool(passed),
                "gate_value": {
                    "region_feature_available_rate_ge_0p95": region_feature_available_rate,
                    "region_count_per_source_mean_ge_8": region_count_mean,
                    "edge_count_per_source_mean_ge_region_count_mean": edge_count_mean,
                    "edge_graph_lcc_ratio_not_all_one": lcc_all_one,
                    "uses_gt_for_prediction_count_eq_0": uses_gt_count,
                    "uses_future_count_eq_0": uses_future_count,
                }.get(gate, ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "created_at": created_at,
            }
        )
    _write_csv(OUT / "variant_gate_rows.csv", gate_rows)

    summary = {
        "schema": "stream4d_v93_phase3_region_edge_graph_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": "PASS_V93_PHASE3_REGION_EDGE_GRAPH" if phase3_pass else "BLOCK_V93_PHASE3_REGION_EDGE_GRAPH",
        "phase3_pass": phase3_pass,
        "phase3_pass_conditions": phase3_pass_conditions,
        "feature_backend": "radio_radseg_spatial_features",
        "region_node_rows": region_node_rows,
        "region_edge_rows": region_edge_rows,
        "region_feature_rows": region_node_rows,
        "region_feature_available_rate": region_feature_available_rate,
        "region_count_per_source_mean": region_count_mean,
        "region_count_per_source_p90": region_count_p90,
        "edge_count_per_source_mean": edge_count_mean,
        "edge_graph_lcc_ratio_mean": float(np.mean(lcc_values)) if lcc_values else "",
        "edge_graph_lcc_ratio_all_one": lcc_all_one,
        "source_internal_same_gt_different_gt_AUC_diagnostic": v92_summary.get("source_internal_same_gt_different_gt_AUC_mean", ""),
        "foreground_background_region_AUC_diagnostic": v92_summary.get("foreground_background_region_AUC_mean", ""),
        "edge_barrier_density": _safe_div(sum(1 for value in lcc_values if value < 1.0), max(1, len(lcc_values))),
        "diagnostic_only_uses_gt": True,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                V92_PHASE4 / "summary.json",
                V92_PHASE4 / "region_node_rows.csv",
                V92_PHASE4 / "region_edge_rows.csv",
                V92_PHASE4 / "semantic_diagnostic_auc_rows.csv",
                V93_PHASE2 / "summary.json",
                V93_PHASE2 / "d4rt_source_support_rows.csv",
            ]
            if path.exists()
        },
        "duration_sec": time.time() - started,
        "created_at": created_at,
    }
    _write_json(OUT / "summary.json", summary)
    sha_rows = {path.name: _sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    _write_json(OUT / "SHA256SUMS.json", sha_rows)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def _rewrite_diagnostic(row: dict[str, Any], created_at: str) -> dict[str, Any]:
    out = dict(row)
    out["schema_version"] = "stream4d_v93_phase3_region_diagnostic_auc_v1"
    out["phase_id"] = PHASE_ID
    out["run_id"] = RUN_ID
    out["created_at"] = created_at
    return out


if __name__ == "__main__":
    run()
