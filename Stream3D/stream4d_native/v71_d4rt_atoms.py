from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _parse_csv_list  # noqa: E402


ATOM_FIELDS = [
    "scene_id",
    "chunk_id",
    "atom_id",
    "atom_type",
    "carrier_id",
    "source_frame",
    "visible_frame_count",
    "visible_frame_ratio",
    "frame_span",
    "uv_valid_count",
    "confidence_mean",
    "visibility_mean",
    "D4RT_position_mean",
    "D4RT_position_std",
    "D4RT_position_descriptor_type",
    "D4RT_relative_position_descriptor",
    "D4RT_trajectory_descriptor",
    "D4RT_motion_magnitude",
    "D4RT_temporal_smoothness",
    "D4RT_cycle_consistency_available",
    "D4RT_cycle_consistency_score",
    "D4RT_local_neighbor_stability_available",
    "D4RT_local_neighbor_stability",
    "mask_membership_count",
    "mask_membership_entropy",
    "mask_membership_mode",
    "mask_membership_mode_frame",
    "atom_mask_observation_id",
    "semantic_feature_available",
    "semantic_feature_mean",
    "semantic_feature_std",
    "semantic_descriptor_type",
    "semantic_backend",
    "semantic_prototype_id",
    "semantic_entropy_mean",
    "semantic_feature_observation_count",
    "non_gt_reliability_score",
    "reliability_components",
    "cluster_size",
    "cluster_method",
    "cluster_D4RT_radius",
    "cluster_semantic_radius",
    "cluster_trajectory_variance",
    "cluster_semantic_variance",
    "cluster_visibility_union",
    "cluster_reliability_mean",
    "cluster_reliability_min",
    "uses_gt_for_prediction",
    "diagnostic_only",
    "forbidden_for_method_table",
]


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return float(sum(valid) / len(valid)) if valid else None


def _variance(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    mu = float(sum(valid) / len(valid))
    return float(sum((v - mu) ** 2 for v in valid) / len(valid))


def _percentile(values: list[float | None], q: float) -> float | None:
    valid = sorted(float(v) for v in values if v is not None)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    idx = int(round((len(valid) - 1) * q))
    idx = max(0, min(len(valid) - 1, idx))
    return valid[idx]


def _entropy(counter: Counter[Any]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0 or len(counter) <= 1:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = float(count) / total
        value -= p * math.log(max(p, 1e-12))
    return float(value / max(1e-12, math.log(len(counter))))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def _chunk_key(scene: str, chunk_id: Any) -> str:
    text = str(chunk_id or "")
    if text.startswith(scene + ":chunk"):
        return text
    num = _int(text, -1)
    if num >= 0:
        return f"{scene}:chunk{num:03d}"
    return text


def _load_semantic_index(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("scene_id") or ""), _int(row.get("frame_id")), _int(row.get("mask_id")))
            out[key] = row
    return out


def _load_pipeline_roots(path: Path, scenes: list[str]) -> dict[str, Path]:
    summary = _load_json(path)
    raw = summary.get("pipeline_roots") or {}
    out: dict[str, Path] = {}
    for scene in scenes:
        value = raw.get(scene)
        if value:
            out[scene] = _rooted(value)
    return out


def _load_gt_count_mean(path: Path, variant: str) -> tuple[float | None, int]:
    values: list[float] = []
    if not path.exists():
        return None, 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("variant")) != variant:
                continue
            value = _float(row.get("local_gt_count"))
            if value is not None:
                values.append(float(value))
    return _mean(values), len(values)


@dataclass(slots=True)
class AtomAgg:
    scene: str
    chunk_id: str
    carrier_id: str
    total_rows: int = 0
    accepted_rows: int = 0
    uv_valid_count: int = 0
    visible_count: int = 0
    min_frame: int | None = None
    max_frame: int | None = None
    sum_confidence: float = 0.0
    sum_visibility: float = 0.0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    step_sum: float = 0.0
    step_sq_sum: float = 0.0
    step_count: int = 0
    last_frame: int | None = None
    last_x: float | None = None
    last_y: float | None = None
    mask_counts: Counter[int] = field(default_factory=Counter)
    mask_first_frame: dict[int, int] = field(default_factory=dict)
    semantic_available_count: int = 0
    semantic_observation_count: int = 0
    semantic_norm_sum: float = 0.0
    semantic_norm_sq_sum: float = 0.0
    semantic_entropy_sum: float = 0.0
    semantic_entropy_sq_sum: float = 0.0
    semantic_backend_counts: Counter[str] = field(default_factory=Counter)
    semantic_prototype_counts: Counter[str] = field(default_factory=Counter)

    def update_total(self, row: dict[str, str]) -> None:
        self.total_rows += 1
        if _bool(row.get("valid_uv")):
            self.uv_valid_count += 1

    def update_accepted(self, row: dict[str, str], semantic_row: dict[str, str] | None) -> None:
        frame = _int(row.get("frame_id"))
        x = float(_float(row.get("uv_x"), 0.0) or 0.0)
        y = float(_float(row.get("uv_y"), 0.0) or 0.0)
        conf = float(_float(row.get("confidence"), 0.0) or 0.0)
        vis = float(_float(row.get("visibility_prob"), 0.0) or 0.0)
        self.accepted_rows += 1
        self.visible_count += 1
        self.min_frame = frame if self.min_frame is None else min(self.min_frame, frame)
        self.max_frame = frame if self.max_frame is None else max(self.max_frame, frame)
        self.sum_confidence += conf
        self.sum_visibility += vis
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        if self.last_frame is not None and self.last_x is not None and self.last_y is not None and frame > self.last_frame:
            dt = max(1, frame - self.last_frame)
            step = math.hypot(x - self.last_x, y - self.last_y) / float(dt)
            self.step_sum += step
            self.step_sq_sum += step * step
            self.step_count += 1
        if self.last_frame is None or frame >= self.last_frame:
            self.last_frame = frame
            self.last_x = x
            self.last_y = y
        mask_id = _int(row.get("observed_mask_id"), 0)
        if mask_id > 0:
            self.mask_counts[mask_id] += 1
            self.mask_first_frame.setdefault(mask_id, frame)
        if semantic_row:
            self.semantic_observation_count += 1
            if _bool(semantic_row.get("feature_available")):
                self.semantic_available_count += 1
                norm = float(_float(semantic_row.get("feature_norm"), 0.0) or 0.0)
                entropy = float(_float(semantic_row.get("semantic_entropy"), 0.0) or 0.0)
                self.semantic_norm_sum += norm
                self.semantic_norm_sq_sum += norm * norm
                self.semantic_entropy_sum += entropy
                self.semantic_entropy_sq_sum += entropy * entropy
                backend = str(semantic_row.get("semantic_backend") or "unknown")
                proto = str(semantic_row.get("semantic_prototype_id") or "unknown")
                self.semantic_backend_counts[backend] += 1
                self.semantic_prototype_counts[proto] += 1

    def to_row(self, atom_index: int, min_visible_frames: int) -> dict[str, Any] | None:
        if self.accepted_rows < int(min_visible_frames):
            return None
        n = float(max(1, self.accepted_rows))
        x_mean = self.sum_x / n
        y_mean = self.sum_y / n
        x_var = max(0.0, self.sum_x2 / n - x_mean * x_mean)
        y_var = max(0.0, self.sum_y2 / n - y_mean * y_mean)
        conf_mean = self.sum_confidence / n
        vis_mean = self.sum_visibility / n
        valid_uv_ratio = float(self.uv_valid_count) / float(max(1, self.total_rows))
        visible_ratio = float(self.visible_count) / float(max(1, self.total_rows))
        motion = self.step_sum / float(max(1, self.step_count)) if self.step_count else 0.0
        temporal_smoothness = math.exp(-5.0 * motion)
        membership_entropy = _entropy(self.mask_counts)
        mode_mask = self.mask_counts.most_common(1)[0][0] if self.mask_counts else ""
        membership_stability = 1.0 - membership_entropy
        semantic_available = self.semantic_available_count > 0
        semantic_ratio = float(self.semantic_available_count) / float(max(1, self.semantic_observation_count or self.accepted_rows))
        semantic_norm_mean = self.semantic_norm_sum / float(max(1, self.semantic_available_count)) if semantic_available else None
        semantic_norm_var = (
            max(0.0, self.semantic_norm_sq_sum / float(max(1, self.semantic_available_count)) - float(semantic_norm_mean or 0.0) ** 2)
            if semantic_available
            else None
        )
        semantic_entropy_mean = self.semantic_entropy_sum / float(max(1, self.semantic_available_count)) if semantic_available else None
        semantic_entropy_var = (
            max(0.0, self.semantic_entropy_sq_sum / float(max(1, self.semantic_available_count)) - float(semantic_entropy_mean or 0.0) ** 2)
            if semantic_available
            else None
        )
        backend = self.semantic_backend_counts.most_common(1)[0][0] if self.semantic_backend_counts else "unavailable"
        prototype = self.semantic_prototype_counts.most_common(1)[0][0] if self.semantic_prototype_counts else ""
        visibility_valid_uv_score = min(vis_mean, valid_uv_ratio)
        reliability_components = {
            "confidence_mean": conf_mean,
            "visibility_mean": vis_mean,
            "valid_uv_ratio": valid_uv_ratio,
            "visibility_valid_uv_score": visibility_valid_uv_score,
            "temporal_smoothness": temporal_smoothness,
            "mask_membership_stability": membership_stability,
            "mask_membership_stability_used_in_score": False,
            "semantic_feature_available_ratio": semantic_ratio,
            "semantic_feature_available_ratio_used_in_score": False,
            "cycle_consistency": {"available": False, "used_in_score": False},
            "local_neighbor_stability": {"available": False, "used_in_score": False},
            "score_formula": "confidence*min(visibility_mean,valid_uv_ratio)*temporal_smoothness",
        }
        reliability = conf_mean * visibility_valid_uv_score * temporal_smoothness
        frame_span = int((self.max_frame or 0) - (self.min_frame or 0) + 1)
        source_frame = int(self.min_frame or 0)
        mode_frame = self.mask_first_frame.get(int(mode_mask), source_frame) if mode_mask != "" else ""
        return {
            "scene_id": self.scene,
            "chunk_id": self.chunk_id,
            "atom_id": f"{self.scene}:A0:{atom_index:08d}",
            "atom_type": "A0_single_carrier_atom",
            "carrier_id": self.carrier_id,
            "source_frame": source_frame,
            "visible_frame_count": int(self.visible_count),
            "visible_frame_ratio": visible_ratio,
            "frame_span": frame_span,
            "uv_valid_count": int(self.uv_valid_count),
            "confidence_mean": conf_mean,
            "visibility_mean": vis_mean,
            "D4RT_position_mean": _json_dump({"uv_x": x_mean, "uv_y": y_mean}),
            "D4RT_position_std": _json_dump({"uv_x": math.sqrt(x_var), "uv_y": math.sqrt(y_var)}),
            "D4RT_position_descriptor_type": "uv_projection_no_3d_position",
            "D4RT_relative_position_descriptor": _json_dump(
                {
                    "chunk_local_uv_mean": [x_mean, y_mean],
                    "chunk_local_uv_std": [math.sqrt(x_var), math.sqrt(y_var)],
                    "physical_3d_position_available": False,
                }
            ),
            "D4RT_trajectory_descriptor": _json_dump(
                {
                    "mean_normalized_uv_step_per_frame": motion,
                    "step_count": int(self.step_count),
                    "trajectory_source": "carrier_observation_table_uv",
                }
            ),
            "D4RT_motion_magnitude": motion,
            "D4RT_temporal_smoothness": temporal_smoothness,
            "D4RT_cycle_consistency_available": False,
            "D4RT_cycle_consistency_score": "",
            "D4RT_local_neighbor_stability_available": False,
            "D4RT_local_neighbor_stability": "",
            "mask_membership_count": int(len(self.mask_counts)),
            "mask_membership_entropy": membership_entropy,
            "mask_membership_mode": mode_mask,
            "mask_membership_mode_frame": mode_frame,
            "atom_mask_observation_id": f"{self.scene}:{int(mode_frame)}:{int(mode_mask)}" if mode_mask != "" and mode_frame != "" else "",
            "semantic_feature_available": semantic_available,
            "semantic_feature_mean": semantic_norm_mean if semantic_norm_mean is not None else "",
            "semantic_feature_std": math.sqrt(semantic_norm_var) if semantic_norm_var is not None else "",
            "semantic_descriptor_type": "feature_norm_entropy_prototype_no_dense_vector",
            "semantic_backend": backend,
            "semantic_prototype_id": prototype,
            "semantic_entropy_mean": semantic_entropy_mean if semantic_entropy_mean is not None else "",
            "semantic_feature_observation_count": int(self.semantic_available_count),
            "non_gt_reliability_score": reliability,
            "reliability_components": _json_dump(reliability_components),
            "cluster_size": "",
            "cluster_method": "",
            "cluster_D4RT_radius": "",
            "cluster_semantic_radius": "",
            "cluster_trajectory_variance": "",
            "cluster_semantic_variance": semantic_entropy_var if semantic_entropy_var is not None else "",
            "cluster_visibility_union": "",
            "cluster_reliability_mean": "",
            "cluster_reliability_min": "",
            "uses_gt_for_prediction": False,
            "diagnostic_only": False,
            "forbidden_for_method_table": False,
        }


def _build_single_atoms(
    *,
    scenes: list[str],
    pipeline_roots: dict[str, Path],
    semantic_index: dict[tuple[str, int, int], dict[str, str]],
    min_confidence: float,
    min_visibility: float,
    min_visible_frames: int,
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    atom_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    total_seen = 0
    for scene in scenes:
        root = pipeline_roots.get(scene)
        if root is None:
            missing_rows.append({"scene_id": scene, "missing": "pipeline_root"})
            continue
        table = root / "observation_tables" / "carrier_observation_table.csv"
        summary_path = root / "observation_tables" / "observation_table_summary.json"
        source_rows.append(
            {
                "scene_id": scene,
                "pipeline_root": _rel(root),
                "carrier_observation_table": _rel(table),
                "observation_summary": _rel(summary_path),
                "carrier_table_exists": table.exists(),
            }
        )
        if not table.exists():
            missing_rows.append({"scene_id": scene, "missing": _rel(table)})
            continue
        aggs: dict[tuple[str, str], AtomAgg] = {}
        with table.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total_seen += 1
                if max_rows is not None and total_seen > int(max_rows):
                    break
                row_scene = str(row.get("scene") or scene)
                chunk = _chunk_key(row_scene, row.get("chunk_id"))
                carrier = str(row.get("carrier_global_id") or row.get("carrier_id") or "")
                if not carrier:
                    continue
                key = (chunk, carrier)
                agg = aggs.get(key)
                if agg is None:
                    agg = AtomAgg(scene=row_scene, chunk_id=chunk, carrier_id=carrier)
                    aggs[key] = agg
                agg.update_total(row)
                if not (_bool(row.get("valid")) and _bool(row.get("valid_uv")) and _bool(row.get("visible"))):
                    continue
                conf = float(_float(row.get("confidence"), 0.0) or 0.0)
                vis = float(_float(row.get("visibility_prob"), 0.0) or 0.0)
                if conf < float(min_confidence) or vis < float(min_visibility):
                    continue
                mask_id = _int(row.get("observed_mask_id"), 0)
                frame_id = _int(row.get("frame_id"))
                semantic_row = semantic_index.get((row_scene, frame_id, mask_id))
                agg.update_accepted(row, semantic_row)
            rows_before_filter = len(aggs)
            for agg in aggs.values():
                atom_index = len(atom_rows)
                atom = agg.to_row(atom_index, min_visible_frames=min_visible_frames)
                if atom is not None:
                    atom_rows.append(atom)
            source_rows[-1]["single_carrier_group_count_before_min_visible_filter"] = rows_before_filter
            source_rows[-1]["single_carrier_atom_count_after_filter"] = len([r for r in atom_rows if r["scene_id"] == scene])
        if max_rows is not None and total_seen > int(max_rows):
            break
    return atom_rows, source_rows, missing_rows


def _build_cluster_atoms(single_rows: list[dict[str, Any]], min_cluster_size: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in single_rows:
        mean_desc = json.loads(str(row.get("D4RT_position_mean") or "{}"))
        ux = float(mean_desc.get("uv_x") or 0.0)
        uy = float(mean_desc.get("uv_y") or 0.0)
        ux_bin = int(max(0, min(3, math.floor(ux * 4.0))))
        uy_bin = int(max(0, min(3, math.floor(uy * 4.0))))
        proto = str(row.get("semantic_prototype_id") or "no_semantic")
        groups[(str(row["scene_id"]), str(row["chunk_id"]), proto, ux_bin, uy_bin)].append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        if len(rows) < int(min_cluster_size):
            continue
        scene, chunk, proto, ux_bin, uy_bin = key
        positions = [json.loads(str(r.get("D4RT_position_mean") or "{}")) for r in rows]
        xs = [float(p.get("uv_x") or 0.0) for p in positions]
        ys = [float(p.get("uv_y") or 0.0) for p in positions]
        mx = float(sum(xs) / len(xs))
        my = float(sum(ys) / len(ys))
        radius = float(max(math.hypot(x - mx, y - my) for x, y in zip(xs, ys))) if xs else 0.0
        reliabilities = [float(r.get("non_gt_reliability_score") or 0.0) for r in rows]
        motions = [float(r.get("D4RT_motion_magnitude") or 0.0) for r in rows]
        sems = [_float(r.get("semantic_entropy_mean")) for r in rows]
        visible_counts = [float(r.get("visible_frame_count") or 0.0) for r in rows]
        frame_spans = [float(r.get("frame_span") or 0.0) for r in rows]
        cluster_size = len(rows)
        reliability_mean = float(sum(reliabilities) / cluster_size)
        atom_index = len(out)
        out.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "atom_id": f"{scene}:A1:{atom_index:08d}",
                "atom_type": "A1_local_carrier_cluster_atom",
                "carrier_id": "",
                "source_frame": min(_int(r.get("source_frame")) for r in rows),
                "visible_frame_count": _mean(visible_counts),
                "visible_frame_ratio": _mean([_float(r.get("visible_frame_ratio")) for r in rows]),
                "frame_span": _mean(frame_spans),
                "uv_valid_count": int(sum(_int(r.get("uv_valid_count")) for r in rows)),
                "confidence_mean": _mean([_float(r.get("confidence_mean")) for r in rows]),
                "visibility_mean": _mean([_float(r.get("visibility_mean")) for r in rows]),
                "D4RT_position_mean": _json_dump({"uv_x": mx, "uv_y": my}),
                "D4RT_position_std": _json_dump({"uv_x": math.sqrt(_variance(xs) or 0.0), "uv_y": math.sqrt(_variance(ys) or 0.0)}),
                "D4RT_position_descriptor_type": "uv_projection_no_3d_position_cluster",
                "D4RT_relative_position_descriptor": _json_dump(
                    {
                        "cluster_uv_bin": [ux_bin, uy_bin],
                        "cluster_uv_mean": [mx, my],
                        "physical_3d_position_available": False,
                    }
                ),
                "D4RT_trajectory_descriptor": _json_dump({"member_motion_magnitude_mean": _mean(motions), "member_count": cluster_size}),
                "D4RT_motion_magnitude": _mean(motions),
                "D4RT_temporal_smoothness": math.exp(-5.0 * float(_mean(motions) or 0.0)),
                "D4RT_cycle_consistency_available": False,
                "D4RT_cycle_consistency_score": "",
                "D4RT_local_neighbor_stability_available": False,
                "D4RT_local_neighbor_stability": "",
                "mask_membership_count": _mean([_float(r.get("mask_membership_count")) for r in rows]),
                "mask_membership_entropy": _mean([_float(r.get("mask_membership_entropy")) for r in rows]),
                "mask_membership_mode": "",
                "mask_membership_mode_frame": "",
                "atom_mask_observation_id": "",
                "semantic_feature_available": any(_bool(r.get("semantic_feature_available")) for r in rows),
                "semantic_feature_mean": _mean([_float(r.get("semantic_feature_mean")) for r in rows]),
                "semantic_feature_std": _mean([_float(r.get("semantic_feature_std")) for r in rows]),
                "semantic_descriptor_type": "prototype_uv4x4_cluster",
                "semantic_backend": Counter(str(r.get("semantic_backend") or "") for r in rows).most_common(1)[0][0],
                "semantic_prototype_id": proto,
                "semantic_entropy_mean": _mean(sems),
                "semantic_feature_observation_count": int(sum(_int(r.get("semantic_feature_observation_count")) for r in rows)),
                "non_gt_reliability_score": reliability_mean,
                "reliability_components": _json_dump(
                    {
                        "source": "mean_member_reliability",
                        "cluster_size": cluster_size,
                        "member_reliability_min": min(reliabilities) if reliabilities else None,
                    }
                ),
                "cluster_size": cluster_size,
                "cluster_method": "semantic_prototype_uv4x4_bin",
                "cluster_D4RT_radius": radius,
                "cluster_semantic_radius": math.sqrt(_variance(sems) or 0.0) if [v for v in sems if v is not None] else "",
                "cluster_trajectory_variance": _variance(motions),
                "cluster_semantic_variance": _variance(sems),
                "cluster_visibility_union": max(visible_counts) if visible_counts else "",
                "cluster_reliability_mean": reliability_mean,
                "cluster_reliability_min": min(reliabilities) if reliabilities else "",
                "uses_gt_for_prediction": False,
                "diagnostic_only": False,
                "forbidden_for_method_table": False,
            }
        )
    return out


def _load_geometry_rows(path: Path, scenes: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = _load_json(path)
    scene_set = set(scenes)
    rows = []
    for row in data.get("scene_rows") or []:
        scene = str(row.get("scene") or "")
        if scene and scene not in scene_set:
            continue
        rows.append(dict(row))
    summary = {
        "source_path": _rel(path),
        "is_diagnostic_only": bool(data.get("is_diagnostic_only", True)),
        "metric_note": data.get("metric_note", ""),
        "row_count": len(rows),
    }
    return rows, summary


def _geometry_diagnostic_rows(geometry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in geometry_rows:
        out.append(
            {
                "scene_id": row.get("scene", ""),
                "variant": row.get("variant", ""),
                "diagnostic_D4RT_depth_absrel": "",
                "diagnostic_D4RT_depth_absrel_available": False,
                "diagnostic_D4RT_point_l1_aligned": row.get("used_frame_support_chamfer_l1", ""),
                "diagnostic_D4RT_point_l1_aligned_metric": "used_frame_support_chamfer_l1",
                "diagnostic_D4RT_scale_only_error": "",
                "diagnostic_D4RT_scale_only_error_available": False,
                "diagnostic_D4RT_scale_shift_error": row.get("scene_fit_sim3_residual_median", ""),
                "diagnostic_D4RT_scale_shift_error_metric": "scene_fit_sim3_residual_median",
                "diagnostic_relative_neighbor_overlap_at_k": row.get("support_support_nn_hit_rate", ""),
                "diagnostic_rank_correlation_distance": "",
                "diagnostic_rank_correlation_distance_available": False,
                "diagnostic_carrier_inside_GT_instance_purity": "",
                "diagnostic_carrier_inside_GT_instance_purity_available": False,
                "diagnostic_scope": "scene_level_existing_v64r2_probe5_GT_depth_pose_mesh_diagnostic",
                "uses_gt_for_prediction": True,
                "diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        )
    return out


def _geometry_summary(geometry_rows: list[dict[str, Any]], preferred_variant: str) -> dict[str, Any]:
    if not geometry_rows:
        return {"available": False, "reason": "geometry_summary_missing_or_empty"}
    preferred = [r for r in geometry_rows if str(r.get("variant")) == preferred_variant] or geometry_rows
    return {
        "available": True,
        "preferred_variant": preferred_variant if any(str(r.get("variant")) == preferred_variant for r in geometry_rows) else "fallback_all_rows",
        "scene_count": len({str(r.get("scene")) for r in preferred}),
        "used_frame_support_chamfer_l1_mean": _mean([_float(r.get("used_frame_support_chamfer_l1")) for r in preferred]),
        "used_frame_support_gt_to_pred_median_mean": _mean([_float(r.get("used_frame_support_gt_to_pred_median")) for r in preferred]),
        "used_frame_support_pred_to_gt_median_mean": _mean([_float(r.get("used_frame_support_pred_to_gt_median")) for r in preferred]),
        "scene_fit_sim3_residual_median_mean": _mean([_float(r.get("scene_fit_sim3_residual_median")) for r in preferred]),
        "scene_fit_sim3_residual_p90_mean": _mean([_float(r.get("scene_fit_sim3_residual_p90")) for r in preferred]),
        "support_nn_hit_rate_mean": _mean([_float(r.get("support_support_nn_hit_rate")) for r in preferred]),
        "full_mesh_chamfer_l1_mean": _mean([_float(r.get("full_mesh_chamfer_l1")) for r in preferred]),
        "metric_source_note": "GT/depth/pose/mesh diagnostic only; not used by method reliability.",
    }


def _chunk_rows(atom_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in atom_rows:
        grouped[(str(row["scene_id"]), str(row["chunk_id"]))].append(row)
    out = []
    for (scene, chunk), rows in sorted(grouped.items()):
        out.append(
            {
                "scene_id": scene,
                "chunk_id": chunk,
                "atom_count": len(rows),
                "single_carrier_atom_count": sum(1 for r in rows if r.get("atom_type") == "A0_single_carrier_atom"),
                "cluster_atom_count": sum(1 for r in rows if r.get("atom_type") == "A1_local_carrier_cluster_atom"),
                "visible_frame_count_mean": _mean([_float(r.get("visible_frame_count")) for r in rows]),
                "frame_span_mean": _mean([_float(r.get("frame_span")) for r in rows]),
                "confidence_mean": _mean([_float(r.get("confidence_mean")) for r in rows]),
                "reliability_mean": _mean([_float(r.get("non_gt_reliability_score")) for r in rows]),
                "semantic_feature_success_rate": _mean([1.0 if _bool(r.get("semantic_feature_available")) else 0.0 for r in rows]),
                "mask_membership_entropy_mean": _mean([_float(r.get("mask_membership_entropy")) for r in rows]),
                "uses_gt_for_prediction": False,
            }
        )
    return out


def _write_histogram_png(values: list[float], path: Path, title: str, bins: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.ones((420, 720, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    valid = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float32)
    if valid.size == 0:
        cv2.putText(canvas, "no values", (260, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), canvas)
        return
    hist, edges = np.histogram(valid, bins=bins, range=(float(valid.min()), float(valid.max()) if valid.max() > valid.min() else float(valid.min() + 1.0)))
    max_count = max(1, int(hist.max()))
    x0, y0 = 60, 360
    width, height = 600, 280
    cv2.rectangle(canvas, (x0, y0 - height), (x0 + width, y0), (220, 220, 220), 1)
    for i, count in enumerate(hist):
        x_a = x0 + int(i * width / bins)
        x_b = x0 + int((i + 1) * width / bins) - 2
        bar_h = int(height * int(count) / max_count)
        cv2.rectangle(canvas, (x_a, y0 - bar_h), (x_b, y0), (52, 105, 168), -1)
    cv2.putText(canvas, f"n={valid.size} min={valid.min():.4f} mean={valid.mean():.4f} max={valid.max():.4f}", (24, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _write_visuals(atom_rows: list[dict[str, Any]], chunk_rows: list[dict[str, Any]], visual_root: Path) -> list[dict[str, Any]]:
    visual_root.mkdir(parents=True, exist_ok=True)
    files = []
    rel_values = [float(r.get("non_gt_reliability_score") or 0.0) for r in atom_rows]
    vis_values = [float(r.get("visible_frame_count") or 0.0) for r in atom_rows]
    chunk_counts = [float(r.get("atom_count") or 0.0) for r in chunk_rows]
    specs = [
        (rel_values, visual_root / "atom_reliability_hist.png", "v71 atom non-GT reliability"),
        (vis_values, visual_root / "atom_visible_frame_count_hist.png", "v71 atom visible frame count"),
        (chunk_counts, visual_root / "atom_count_per_chunk_hist.png", "v71 atom count per chunk"),
    ]
    for values, path, title in specs:
        _write_histogram_png(values, path, title)
        files.append({"visual_path": _rel(path), "kind": "histogram", "title": title})
    return files


def build_atoms(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _parse_csv_list(args.scenes) if args.scenes else list(DEFAULT_SCENES)
    output_root = _rooted(args.output_root)
    visual_root = _rooted(args.visual_root)
    semantic_index = _load_semantic_index(_rooted(args.semantic_feature_rows))
    pipeline_roots = _load_pipeline_roots(_rooted(args.v70_witness_summary), scenes)
    single_rows, source_rows, missing_rows = _build_single_atoms(
        scenes=scenes,
        pipeline_roots=pipeline_roots,
        semantic_index=semantic_index,
        min_confidence=float(args.min_confidence),
        min_visibility=float(args.min_visibility),
        min_visible_frames=int(args.min_visible_frames),
        max_rows=args.max_rows,
    )
    cluster_rows = _build_cluster_atoms(single_rows, min_cluster_size=int(args.min_cluster_size))
    atom_rows = single_rows + cluster_rows
    chunk_rows = _chunk_rows(atom_rows)
    geometry_rows, geometry_source_summary = _load_geometry_rows(_rooted(args.geometry_summary), scenes)
    diagnostic_rows = _geometry_diagnostic_rows(geometry_rows)
    geom_summary = _geometry_summary(geometry_rows, preferred_variant=str(args.preferred_geometry_variant))
    visual_rows = _write_visuals(atom_rows, chunk_rows, visual_root)
    gt_count_mean, gt_count_chunks = _load_gt_count_mean(_rooted(args.candidate_metric_rows), str(args.gt_count_variant))
    reliability_values = [_float(r.get("non_gt_reliability_score")) for r in atom_rows]
    single_reliability_by_scene: dict[str, list[float]] = defaultdict(list)
    for row in single_rows:
        value = _float(row.get("non_gt_reliability_score"))
        if value is not None:
            single_reliability_by_scene[str(row["scene_id"])].append(value)
    preferred_geom = [r for r in geometry_rows if str(r.get("variant")) == str(args.preferred_geometry_variant)]
    corr_x: list[float] = []
    corr_y: list[float] = []
    for row in preferred_geom:
        scene = str(row.get("scene") or "")
        rel = _mean(single_reliability_by_scene.get(scene, []))
        err = _float(row.get("used_frame_support_chamfer_l1"))
        if rel is not None and err is not None:
            corr_x.append(float(rel))
            corr_y.append(float(err))
    corr = _pearson(corr_x, corr_y)
    semantic_success = _mean([1.0 if _bool(r.get("semantic_feature_available")) else 0.0 for r in atom_rows])
    atom_count_per_chunk_mean = _mean([_float(r.get("atom_count")) for r in chunk_rows])
    non_gt_exists_rate = _mean([1.0 if _float(r.get("non_gt_reliability_score")) is not None else 0.0 for r in atom_rows])
    gate = {
        "all_inputs_present": not missing_rows,
        "atom_count_per_chunk_mean_ge_5x_diagnostic_GT_count_per_chunk_mean": bool(
            atom_count_per_chunk_mean is not None
            and gt_count_mean is not None
            and atom_count_per_chunk_mean >= 5.0 * gt_count_mean
        ),
        "atom_visible_frame_count_mean_ge_2": bool((_mean([_float(r.get("visible_frame_count")) for r in atom_rows]) or 0.0) >= 2.0),
        "atom_reliability_mean_ge_0p30": bool((_mean(reliability_values) or 0.0) >= 0.30),
        "atom_semantic_feature_success_rate_ge_0p90": bool((semantic_success or 0.0) >= 0.90),
        "D4RT_diagnostic_geometry_gap_recorded_not_ignored": bool(diagnostic_rows),
        "non_gt_reliability_score_exists_for_ge_0p90_atoms": bool((non_gt_exists_rate or 0.0) >= 0.90),
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    summary = {
        "phase": "v71_d4rt_atoms",
        "decision": "PASS_V71_D4RT_ATOMS" if gate["pass"] else "NO_GO_PHASE2_D4RT_ATOM_UNIVERSE",
        "config": {
            "scenes": scenes,
            "min_confidence": float(args.min_confidence),
            "min_visibility": float(args.min_visibility),
            "min_visible_frames": int(args.min_visible_frames),
            "min_cluster_size": int(args.min_cluster_size),
            "max_rows": args.max_rows,
            "geometry_descriptor": "uv_projection_no_3d_position",
            "reliability_is_non_gt": True,
        },
        "source_artifacts": {
            "v70_witness_summary": _rel(_rooted(args.v70_witness_summary)),
            "semantic_feature_rows": _rel(_rooted(args.semantic_feature_rows)),
            "candidate_metric_rows": _rel(_rooted(args.candidate_metric_rows)),
            "geometry_summary": _rel(_rooted(args.geometry_summary)),
            "pipeline_roots": {scene: _rel(path) for scene, path in pipeline_roots.items()},
        },
        "source_rows": source_rows,
        "geometry_source_summary": geometry_source_summary,
        "key_metrics": {
            "atom_count_total": len(atom_rows),
            "atom_count_per_chunk_mean": atom_count_per_chunk_mean,
            "diagnostic_GT_count_per_chunk_mean": gt_count_mean,
            "diagnostic_GT_count_chunk_count": gt_count_chunks,
            "single_carrier_atom_count": len(single_rows),
            "cluster_atom_count": len(cluster_rows),
            "atom_visible_frame_count_mean": _mean([_float(r.get("visible_frame_count")) for r in atom_rows]),
            "atom_frame_span_mean": _mean([_float(r.get("frame_span")) for r in atom_rows]),
            "atom_confidence_mean": _mean([_float(r.get("confidence_mean")) for r in atom_rows]),
            "atom_reliability_mean": _mean(reliability_values),
            "atom_reliability_p10": _percentile(reliability_values, 0.10),
            "atom_reliability_p90": _percentile(reliability_values, 0.90),
            "atom_mask_membership_entropy_mean": _mean([_float(r.get("mask_membership_entropy")) for r in atom_rows]),
            "atom_semantic_feature_success_rate": semantic_success,
            "atom_cluster_size_mean": _mean([_float(r.get("cluster_size")) for r in cluster_rows]),
            "atom_cluster_semantic_variance_mean": _mean([_float(r.get("cluster_semantic_variance")) for r in cluster_rows]),
            "atom_cluster_trajectory_variance_mean": _mean([_float(r.get("cluster_trajectory_variance")) for r in cluster_rows]),
            "D4RT_diagnostic_geometry_gap_summary": geom_summary,
            "D4RT_reliability_vs_GT_error_correlation_diagnostic": {
                "available": corr is not None,
                "scope": "scene_level_proxy_preferred_geometry_variant",
                "preferred_geometry_variant": str(args.preferred_geometry_variant),
                "n_scenes": len(corr_x),
                "x": "mean_non_gt_single_atom_reliability_by_scene",
                "y": "used_frame_support_chamfer_l1_GT_depth_pose_mesh_diagnostic",
                "pearson_r": corr,
                "uses_gt_for_prediction": True,
                "diagnostic_only": True,
                "forbidden_for_method_table": True,
            },
            "non_gt_reliability_score_exists_rate": non_gt_exists_rate,
        },
        "gate": gate,
        "notes": [
            "Carrier observation tables expose UV projection, visibility, confidence, and observed_mask_id, not physical 3D coordinates; D4RT_position_* fields are explicitly uv_projection_no_3d_position descriptors.",
            "non_gt_reliability_score uses only confidence, visibility, valid_uv ratio, UV temporal smoothness, mask membership stability, and semantic feature availability.",
            "Cycle consistency and local neighbor stability were not available in the carrier observation table; rows mark those components available=false.",
            "D4RT diagnostic geometry gap is imported from an existing GT/depth/pose/mesh diagnostic summary and is diagnostic-only; it does not modify method atom reliability.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "atom_rows.csv", atom_rows)
    _write_csv(output_root / "atom_chunk_rows.csv", chunk_rows)
    _write_csv(output_root / "d4rt_geometry_diagnostic_rows.csv", diagnostic_rows)
    _write_csv(output_root / "source_rows.csv", source_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    _write_csv(output_root / "visualization_rows.csv", visual_rows)
    _write_json(output_root / "atom_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "atom_rows.csv",
        output_root / "atom_chunk_rows.csv",
        output_root / "d4rt_geometry_diagnostic_rows.csv",
        output_root / "source_rows.csv",
        output_root / "missing_input_rows.csv",
        output_root / "visualization_rows.csv",
        output_root / "atom_summary.json",
    ]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": path.stat().st_size})
    for row in visual_rows:
        path = _rooted(row["visual_path"])
        if path.exists():
            sha_rows.append({"path": row["visual_path"], "sha256": _sha256(path), "bytes": path.stat().st_size})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v71 D4RT 4D atom universe and reliability diagnostics.")
    parser.add_argument("--output-root", default="outputs/audit/v71_d4rt_atoms")
    parser.add_argument("--visual-root", default="outputs/audit/v71_visualizations/d4rt_atoms")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--v70-witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--semantic-feature-rows", default="outputs/audit/v71_semantic_features/mask_feature_rows.csv")
    parser.add_argument("--candidate-metric-rows", default="outputs/audit/v71_candidate_bank/candidate_metric_rows.csv")
    parser.add_argument("--geometry-summary", default="outputs/audit/v64r2_scene_level_d4rt_geometry_probe5/scene_level_d4rt_geometry_summary.json")
    parser.add_argument("--preferred-geometry-variant", default="D5_SCALE_STITCH_EVAL_SIM3_DENSITY")
    parser.add_argument("--gt-count-variant", default="C6_union_candidate_bank")
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-visible-frames", type=int, default=2)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    summary = build_atoms(args)
    print(json.dumps({"decision": summary["decision"], "gate": summary["gate"], "key_metrics": summary["key_metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(parse_args())
