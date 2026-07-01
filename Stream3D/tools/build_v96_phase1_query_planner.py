#!/usr/bin/env python3
"""Build Stream4D v96 Phase1 micro-primitive query plans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v96_phase1_query_planner"
RUN_ID = "v96_phase1_query_planner"

DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_EDGE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/mask_edge_rows.csv"
DEFAULT_REGION_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/region_node_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v96_phase1_query_planner"

VARIANT_BUDGETS: dict[str, dict[str, int]] = {
    "Q0_uniform256": {"uniform": 256},
    "Q1_uniform1024": {"uniform": 1024},
    "Q2_adaptive512": {"uniform": 128, "interior": 128, "boundary": 128, "conflict": 64, "semgrad": 64},
    "Q3_adaptive1024": {"uniform": 256, "interior": 256, "boundary": 256, "conflict": 128, "semgrad": 128},
    "Q4_adaptive2048_stress": {"uniform": 512, "interior": 512, "boundary": 512, "conflict": 256, "semgrad": 256},
    "Q5_occupancy_adaptive1024": {"uniform": 128, "interior": 256, "boundary": 320, "conflict": 192, "semgrad": 128},
}
STRATUM_PRIORITY = {"conflict": 0, "boundary": 1, "semgrad": 2, "interior": 3, "uniform": 4}
QUERY_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "scene_id",
    "split",
    "window_id",
    "frame_id",
    "query_variant",
    "query_id",
    "query_u",
    "query_v",
    "query_x",
    "query_y",
    "query_u_norm",
    "query_v_norm",
    "query_stratum",
    "query_priority",
    "source_mask_id_optional",
    "source_edge_id_optional",
    "semantic_gradient_score",
    "mask_conflict_score",
    "is_mask_interior",
    "is_mask_boundary",
    "is_conflict_region",
    "is_semantic_gradient",
    "occupancy_before",
    "sampler_status",
    "uses_gt_for_query_selection",
    "uses_gt_for_prediction",
    "uses_future",
]


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
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
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) & 0xFFFFFFFF


def _load_label(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask label image: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _label_boundary(label: np.ndarray, source_ids: set[int], band_px: int) -> np.ndarray:
    positive = np.isin(label, np.asarray(sorted(source_ids), dtype=np.int64)) if source_ids else label > 0
    edge = np.zeros(label.shape, dtype=np.uint8)
    diff_x = label[:, 1:] != label[:, :-1]
    diff_y = label[1:, :] != label[:-1, :]
    edge[:, 1:] |= diff_x
    edge[:, :-1] |= diff_x
    edge[1:, :] |= diff_y
    edge[:-1, :] |= diff_y
    edge &= positive.astype(np.uint8)
    kernel = np.ones((max(1, int(band_px)) * 2 + 1, max(1, int(band_px)) * 2 + 1), dtype=np.uint8)
    return cv2.dilate(edge, kernel, iterations=1).astype(bool) & positive


def _disk_points(mask: np.ndarray, centers: list[tuple[float, float]], radius: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.uint8)
    h, w = mask.shape
    for cx, cy in centers:
        x = int(round(cx))
        y = int(round(cy))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), int(radius), 1, thickness=-1)
    return out.astype(bool) & mask


def _uniform_points(height: int, width: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2), dtype=np.int32)
    cols = int(math.ceil(math.sqrt(count * width / max(height, 1))))
    rows = int(math.ceil(count / max(cols, 1)))
    xs = ((np.arange(cols, dtype=np.float64) + 0.5) * width / max(cols, 1)).clip(0, width - 1)
    ys = ((np.arange(rows, dtype=np.float64) + 0.5) * height / max(rows, 1)).clip(0, height - 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pts = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)[:count]
    return np.rint(pts).astype(np.int32)


def _sample_mask(mask: np.ndarray, count: int, *, seed: int, fallback: np.ndarray) -> tuple[np.ndarray, str]:
    if count <= 0:
        return np.zeros((0, 2), dtype=np.int32), "empty_budget"
    ys, xs = np.nonzero(mask)
    status = "primary"
    if ys.size == 0:
        ys, xs = np.nonzero(fallback)
        status = "fallback_foreground"
    if ys.size == 0:
        h, w = mask.shape
        return _uniform_points(h, w, count), "fallback_uniform_empty"
    rng = np.random.default_rng(seed)
    replace = ys.size < count
    idx = rng.choice(ys.size, size=count, replace=replace)
    if replace:
        status += "_with_replacement"
    return np.stack([ys[idx], xs[idx]], axis=1).astype(np.int32), status


def _frame_key(row: dict[str, str]) -> tuple[str, str, int, str]:
    return (row["scene_id"], row["window_id"], int(float(row["frame_id"])), row["mask_path"])


def _collect_conflict_ids(edge_rows: list[dict[str, str]], source_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], set[int]]:
    out: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for row in edge_rows:
        if row.get("edge_type") == "competing":
            key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
            a = int(_num(row.get("edge_mask_id_a"), _num(row.get("source_mask_id"))))
            b = int(_num(row.get("edge_mask_id_b")))
            if a > 0:
                out[key].add(a)
            if b > 0:
                out[key].add(b)
    for row in source_rows:
        if int(_num(row.get("object_candidate_count"))) >= 2:
            key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
            mid = int(_num(row.get("source_mask_id")))
            if mid > 0:
                out[key].add(mid)
    return out


def _collect_semgrad_centers(region_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], list[tuple[float, float]]]:
    out: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    per_source: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in region_rows:
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("source_mask_id"))))
        per_source[key].append(row)
    for (scene, window, frame, _mask_id), rows in per_source.items():
        ranked = sorted(
            rows,
            key=lambda r: (0 if _bool(r.get("boundary_token")) else 1, _num(r.get("source_mean_cosine"), 1.0)),
        )
        for row in ranked[: min(12, len(ranked))]:
            out[(scene, window, frame)].append((_num(row.get("centroid_x")), _num(row.get("centroid_y"))))
    return out


class MetricAgg:
    def __init__(self) -> None:
        self.count = 0
        self.unique_pixels: set[tuple[str, int, int]] = set()
        self.strata: Counter[str] = Counter()
        self.frames: set[tuple[str, str, int]] = set()
        self.interior = 0
        self.boundary = 0
        self.conflict = 0
        self.semgrad = 0
        self.uses_gt = 0
        self.uses_future = 0

    def add(self, row: dict[str, Any]) -> None:
        self.count += 1
        self.strata[str(row["query_stratum"])] += 1
        self.frames.add((str(row["scene_id"]), str(row["window_id"]), int(row["frame_id"])))
        self.unique_pixels.add((str(row["scene_id"]), int(row["frame_id"]), int(row["query_y"]) * 10000 + int(row["query_x"])))
        self.interior += int(bool(row["is_mask_interior"]))
        self.boundary += int(bool(row["is_mask_boundary"]))
        self.conflict += int(bool(row["is_conflict_region"]))
        self.semgrad += int(bool(row["is_semantic_gradient"]))
        self.uses_gt += int(bool(row["uses_gt_for_query_selection"]) or bool(row["uses_gt_for_prediction"]))
        self.uses_future += int(bool(row["uses_future"]))

    def summary(self, variant: str, image_area_mean: float, runtime_sec: float) -> dict[str, Any]:
        denom = max(1, self.count)
        frame_count = max(1, len(self.frames))
        return {
            "query_variant": variant,
            "query_count": int(self.count),
            "frame_count": int(len(self.frames)),
            "query_points_per_frame": float(self.count / frame_count),
            "stratum_count_fraction": json.dumps({k: v / denom for k, v in sorted(self.strata.items())}, sort_keys=True),
            "mask_interior_query_rate": float(self.interior / denom),
            "mask_boundary_query_rate": float(self.boundary / denom),
            "conflict_region_query_rate": float(self.conflict / denom),
            "semantic_gradient_query_rate": float(self.semgrad / denom),
            "unvisited_pixel_rate_after_sampling_estimate": float(1.0 - min(1.0, len(self.unique_pixels) / max(image_area_mean * frame_count, 1.0))),
            "uses_gt_for_query_selection_count": int(self.uses_gt),
            "uses_future_count": int(self.uses_future),
            "runtime_plan_sec": float(runtime_sec),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_rows = [row for row in _read_csv(_project(args.source_rows)) if row.get("split", "dev") == "dev"]
    edge_rows = _read_csv(_project(args.edge_rows))
    region_rows = _read_csv(_project(args.region_rows))
    if args.scenes:
        scenes = {part.strip() for part in args.scenes.split(",") if part.strip()}
        source_rows = [row for row in source_rows if row.get("scene_id") in scenes]
        edge_rows = [row for row in edge_rows if row.get("scene_id") in scenes]
        region_rows = [row for row in region_rows if row.get("scene_id") in scenes]

    grouped: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        grouped[_frame_key(row)].append(row)
    frame_items = sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]))
    if args.max_frames > 0:
        frame_items = frame_items[: int(args.max_frames)]

    conflict_ids = _collect_conflict_ids(edge_rows, source_rows)
    semgrad_centers = _collect_semgrad_centers(region_rows)
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    if not variants:
        variants = list(VARIANT_BUDGETS)
    for variant in variants:
        if variant not in VARIANT_BUDGETS:
            raise ValueError(f"unknown query variant: {variant}")

    query_path = output_root / "query_plan_rows.csv"
    query_file = query_path.open("w", newline="", encoding="utf-8")
    query_writer = csv.DictWriter(query_file, fieldnames=QUERY_FIELDS)
    query_writer.writeheader()

    variant_aggs = {variant: MetricAgg() for variant in variants}
    stratum_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    image_area_values: list[int] = []
    query_index_by_variant: Counter[str] = Counter()
    missing_mask_frames: list[dict[str, Any]] = []

    try:
        for frame_idx, ((scene, window, frame_id, mask_path_raw), rows) in enumerate(frame_items):
            mask_path = _project(mask_path_raw)
            if not mask_path.exists():
                missing_mask_frames.append({"scene_id": scene, "window_id": window, "frame_id": frame_id, "mask_path": _rel(mask_path)})
                continue
            label = _load_label(mask_path)
            height, width = label.shape
            image_area_values.append(int(height * width))
            source_ids = {int(_num(row.get("source_mask_id"))) for row in rows if int(_num(row.get("source_mask_id"))) > 0}
            source_ids_arr = np.asarray(sorted(source_ids), dtype=np.int64)
            foreground = np.isin(label, source_ids_arr) if source_ids else label > 0
            boundary = _label_boundary(label, source_ids, int(args.boundary_band_px))
            interior = foreground & ~boundary
            frame_conflict_ids = conflict_ids.get((scene, window, frame_id), set())
            conflict_mask = (
                np.isin(label, np.asarray(sorted(frame_conflict_ids), dtype=np.int64))
                if frame_conflict_ids
                else np.zeros(label.shape, dtype=bool)
            )
            # Conflict samples should target hard-negative / competing-edge bands.
            # Marking whole broad masks as conflict made uniform sampling nearly all
            # conflict and invalidated the Q0-relative gate.
            conflict = boundary & conflict_mask
            if frame_conflict_ids and not np.any(conflict):
                conflict = boundary & foreground
            semgrad = _disk_points(foreground, semgrad_centers.get((scene, window, frame_id), []), radius=int(args.semgrad_radius_px))
            if not np.any(semgrad):
                semgrad = boundary
            fallback = foreground if np.any(foreground) else np.ones(label.shape, dtype=bool)
            maps = {
                "interior": interior,
                "boundary": boundary,
                "conflict": conflict,
                "semgrad": semgrad,
                "uniform": np.ones(label.shape, dtype=bool),
            }
            candidate_counts = {name: int(np.count_nonzero(mask)) for name, mask in maps.items()}
            for variant in variants:
                variant_t0 = time.time()
                budgets = VARIANT_BUDGETS[variant]
                per_variant_rows: list[dict[str, Any]] = []
                for stratum, budget in budgets.items():
                    if stratum == "uniform":
                        points = _uniform_points(height, width, int(budget))
                        status = "grid"
                    else:
                        points, status = _sample_mask(
                            maps[stratum],
                            int(budget),
                            seed=_stable_seed(RUN_ID, scene, window, frame_id, variant, stratum),
                            fallback=fallback,
                        )
                    for y, x in points:
                        y_i = int(np.clip(y, 0, height - 1))
                        x_i = int(np.clip(x, 0, width - 1))
                        mask_id = int(label[y_i, x_i])
                        is_fg = bool(foreground[y_i, x_i])
                        row = {
                            "schema_version": "stream4d_v96_query_plan_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "scene_id": scene,
                            "split": "dev",
                            "window_id": window,
                            "frame_id": int(frame_id),
                            "query_variant": variant,
                            "query_id": f"{variant}:{scene}:{window}:f{frame_id:06d}:q{query_index_by_variant[variant]:09d}",
                            "query_u": float(x_i),
                            "query_v": float(y_i),
                            "query_x": int(x_i),
                            "query_y": int(y_i),
                            "query_u_norm": float(x_i / max(width - 1, 1)),
                            "query_v_norm": float(y_i / max(height - 1, 1)),
                            "query_stratum": stratum,
                            "query_priority": int(STRATUM_PRIORITY.get(stratum, 9)),
                            "source_mask_id_optional": mask_id if is_fg else "",
                            "source_edge_id_optional": "",
                            "semantic_gradient_score": float(1.0 if semgrad[y_i, x_i] else 0.0),
                            "mask_conflict_score": float(1.0 if conflict[y_i, x_i] else 0.0),
                            "is_mask_interior": bool(interior[y_i, x_i]),
                            "is_mask_boundary": bool(boundary[y_i, x_i]),
                            "is_conflict_region": bool(conflict[y_i, x_i]),
                            "is_semantic_gradient": bool(semgrad[y_i, x_i]),
                            "occupancy_before": 0.0,
                            "sampler_status": status,
                            "uses_gt_for_query_selection": False,
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                        query_index_by_variant[variant] += 1
                        per_variant_rows.append(row)
                        variant_aggs[variant].add(row)
                for row in per_variant_rows:
                    query_writer.writerow({key: _jsonable(row.get(key, "")) for key in QUERY_FIELDS})
                stratum_counts = Counter(row["query_stratum"] for row in per_variant_rows)
                for stratum, count in sorted(stratum_counts.items()):
                    stratum_rows.append(
                        {
                            "schema_version": "stream4d_v96_query_stratum_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "scene_id": scene,
                            "split": "dev",
                            "window_id": window,
                            "frame_id": int(frame_id),
                            "query_variant": variant,
                            "query_stratum": stratum,
                            "query_count": int(count),
                            "query_fraction": float(count / max(1, len(per_variant_rows))),
                            "candidate_pixel_count": int(candidate_counts.get(stratum, height * width)),
                            "source_mask_count": int(len(source_ids)),
                            "conflict_mask_count": int(len(frame_conflict_ids)),
                            "runtime_frame_variant_sec": float(time.time() - variant_t0),
                            "uses_gt_for_query_selection": False,
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
            if frame_idx < int(args.manifest_frames):
                manifest_rows.append(
                    {
                        "schema_version": "stream4d_v96_query_visualization_manifest_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "scene_id": scene,
                        "window_id": window,
                        "frame_id": int(frame_id),
                        "mask_path": _rel(mask_path),
                        "query_plan_rows": _rel(query_path),
                        "visualization_status": "manifest_only",
                        "note": "Overlay can be reproduced from query_plan_rows filtered by scene/window/frame.",
                        "uses_gt_for_query_selection": False,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
            if args.progress_every_frames > 0 and (frame_idx + 1) % int(args.progress_every_frames) == 0:
                print({"phase": PHASE_ID, "processed_frames": frame_idx + 1, "total_frames": len(frame_items)})
    finally:
        query_file.close()

    image_area_mean = float(np.mean(image_area_values)) if image_area_values else 0.0
    elapsed = float(time.time() - started)
    variant_rows = [variant_aggs[variant].summary(variant, image_area_mean, elapsed) for variant in variants]
    by_variant = {row["query_variant"]: row for row in variant_rows}
    q0_boundary = float(by_variant.get("Q0_uniform256", {}).get("mask_boundary_query_rate", 0.0))
    q0_conflict = float(by_variant.get("Q0_uniform256", {}).get("conflict_region_query_rate", 0.0))
    gate_rows = []
    for variant in ["Q2_adaptive512", "Q3_adaptive1024"]:
        row = by_variant.get(variant, {})
        gate_rows.append(
            {
                "schema_version": "stream4d_v96_phase1_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "query_variant": variant,
                "gate": "boundary_query_rate_ge_Q0_plus_0p15",
                "pass": float(row.get("mask_boundary_query_rate", 0.0)) >= q0_boundary + 0.15,
                "observed": row.get("mask_boundary_query_rate", 0.0),
                "required": q0_boundary + 0.15,
                "uses_gt_for_query_selection": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        gate_rows.append(
            {
                "schema_version": "stream4d_v96_phase1_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "query_variant": variant,
                "gate": "conflict_region_query_rate_ge_Q0_plus_0p05",
                "pass": float(row.get("conflict_region_query_rate", 0.0)) >= q0_conflict + 0.05,
                "observed": row.get("conflict_region_query_rate", 0.0),
                "required": q0_conflict + 0.05,
                "uses_gt_for_query_selection": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    gate_rows.append(
        {
            "schema_version": "stream4d_v96_phase1_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "query_variant": "ALL",
            "gate": "no_gt_or_future_query_selection",
            "pass": all(row["uses_gt_for_query_selection_count"] == 0 and row["uses_future_count"] == 0 for row in variant_rows),
            "observed": json.dumps({row["query_variant"]: [row["uses_gt_for_query_selection_count"], row["uses_future_count"]] for row in variant_rows}, sort_keys=True),
            "required": "all zero",
            "uses_gt_for_query_selection": False,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    )
    phase1_pass = all(bool(row["pass"]) for row in gate_rows)

    summary = {
        "schema": "stream4d_v96_phase1_query_planner_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V96_PHASE1_QUERY_PLANNER" if phase1_pass else "NO_GO_V96_PHASE1_QUERY_PLANNER",
        "source_container_rows": _rel(_project(args.source_rows)),
        "mask_edge_rows": _rel(_project(args.edge_rows)),
        "region_node_rows": _rel(_project(args.region_rows)),
        "source_row_count": int(len(source_rows)),
        "processed_frame_count": int(len(image_area_values)),
        "missing_mask_frame_count": int(len(missing_mask_frames)),
        "query_variant_rows": variant_rows,
        "gate_rows": gate_rows,
        "query_plan_rows": _rel(query_path),
        "query_plan_rows_sha256": _sha256(query_path),
        "query_stratum_rows": _rel(output_root / "query_stratum_rows.csv"),
        "query_visualization_manifest": _rel(output_root / "query_visualization_manifest.csv"),
        "variant_metric_rows": _rel(output_root / "variant_metric_rows.csv"),
        "phase1_gate_rows": _rel(output_root / "phase1_gate_rows.csv"),
        "missing_mask_rows": _rel(output_root / "missing_mask_rows.csv"),
        "boundary_band_px": int(args.boundary_band_px),
        "semgrad_radius_px": int(args.semgrad_radius_px),
        "runtime_plan_sec": elapsed,
        "uses_gt_for_query_selection": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    _write_csv(output_root / "query_stratum_rows.csv", stratum_rows)
    _write_csv(output_root / "query_visualization_manifest.csv", manifest_rows)
    _write_csv(output_root / "variant_metric_rows.csv", variant_rows)
    _write_csv(output_root / "phase1_gate_rows.csv", gate_rows)
    _write_csv(output_root / "missing_mask_rows.csv", missing_mask_frames)
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v96 micro-primitive query plan rows.")
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--edge-rows", default=str(DEFAULT_EDGE_ROWS))
    parser.add_argument("--region-rows", default=str(DEFAULT_REGION_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--variants", default=",".join(VARIANT_BUDGETS))
    parser.add_argument("--scenes", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--boundary-band-px", type=int, default=4)
    parser.add_argument("--semgrad-radius-px", type=int, default=10)
    parser.add_argument("--manifest-frames", type=int, default=8)
    parser.add_argument("--progress-every-frames", type=int, default=32)
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "processed_frame_count": summary["processed_frame_count"],
                "query_plan_rows": summary["query_plan_rows"],
                "runtime_plan_sec": summary["runtime_plan_sec"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
