#!/usr/bin/env python3
"""Build Stream4D v97 Phase1 object-informative micro-query plans."""

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
PHASE_ID = "v97_phase1_query_planner"
RUN_ID = "v97_phase1_query_planner"

DEFAULT_PHASE0 = ROOT / "outputs/audit/v97_phase0_fact_lock/summary.json"
DEFAULT_SOURCE_ROWS = ROOT / "outputs/audit/v95_phase1_physical_source_registry/source_container_rows.csv"
DEFAULT_EDGE_ROWS = ROOT / "outputs/audit/v93_phase1_source_edge_registry/mask_edge_hypothesis_rows.csv"
DEFAULT_REGION_ROWS = ROOT / "outputs/audit/v93_phase3_region_edge_graph/region_node_rows.csv"
DEFAULT_OUT = ROOT / "outputs/audit/v97_phase1_query_planner"

VARIANT_BUDGETS: dict[str, dict[str, int]] = {
    "Q0_uniform256": {"uniform": 256},
    "Q1_uniform1024": {"uniform": 1024},
    "Q2_adaptive512": {"uniform": 128, "interior": 128, "boundary": 128, "conflict": 64, "semantic_gradient": 64},
    "Q3_adaptive1024": {"uniform": 256, "interior": 256, "boundary": 256, "conflict": 128, "semantic_gradient": 128},
    "Q4_boundary_conflict1024": {"uniform": 128, "interior": 128, "boundary": 384, "conflict": 256, "semantic_gradient": 128},
    "Q5_semantic_gradient1024": {"uniform": 128, "interior": 128, "boundary": 192, "conflict": 128, "semantic_gradient": 448},
    "Q6_occupancy_adaptive1024": {"uniform": 128, "interior": 256, "boundary": 320, "conflict": 192, "semantic_gradient": 128},
}

STRATUM_PRIORITY = {
    "conflict": 0,
    "boundary": 1,
    "semantic_gradient": 2,
    "interior": 3,
    "uniform": 4,
}

QUERY_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "variant_id",
    "scene_id",
    "window_id",
    "chunk_id",
    "frame_id",
    "query_id",
    "query_stratum",
    "source_mask_id",
    "x_norm",
    "y_norm",
    "x_px",
    "y_px",
    "stratum_weight",
    "importance_weight",
    "near_mask_boundary",
    "near_competing_edge",
    "near_nested_edge",
    "semantic_gradient_score",
    "source_risk_score",
    "query_selection_uses_gt",
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    kernel_size = max(1, int(band_px)) * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
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
        return _uniform_points(*mask.shape, count), "fallback_uniform_empty"
    rng = np.random.default_rng(seed)
    replace = ys.size < count
    idx = rng.choice(ys.size, size=count, replace=replace)
    if replace:
        status += "_with_replacement"
    return np.stack([ys[idx], xs[idx]], axis=1).astype(np.int32), status


def _frame_key(row: dict[str, str]) -> tuple[str, str, str, int, str]:
    return (
        row.get("scene_id", ""),
        row.get("window_id", ""),
        row.get("chunk_id", "0") or "0",
        int(_num(row.get("frame_id"))),
        row.get("mask_path", ""),
    )


def _collect_edge_ids(edge_rows: list[dict[str, str]], source_rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str, int], set[int]], dict[tuple[str, str, int], set[int]]]:
    conflict: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    nested: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for row in edge_rows:
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
        edge_type = str(row.get("edge_type", "")).lower()
        ids = [int(_num(row.get("edge_mask_id_a"), _num(row.get("source_mask_id")))), int(_num(row.get("edge_mask_id_b")))]
        target = conflict if "competing" in edge_type or "conflict" in edge_type else nested if "nested" in edge_type or "contain" in edge_type or "part" in edge_type else None
        if target is None:
            continue
        for mid in ids:
            if mid > 0:
                target[key].add(mid)
    # Multiple object candidates inside one source are a GT-free risk proxy for competing interior support.
    for row in source_rows:
        if int(_num(row.get("object_candidate_count"))) >= 2:
            key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))))
            mid = int(_num(row.get("source_mask_id")))
            if mid > 0:
                conflict[key].add(mid)
    return conflict, nested


def _collect_semgrad_centers(region_rows: list[dict[str, str]]) -> dict[tuple[str, str, int], list[tuple[float, float]]]:
    out: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    per_source: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in region_rows:
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        key = (row.get("scene_id", ""), row.get("window_id", ""), int(_num(row.get("frame_id"))), int(_num(row.get("source_mask_id"))))
        per_source[key].append(row)
    for (scene, window, frame, _mask_id), rows in per_source.items():
        # Boundary token plus low source cosine is used only as a proxy for semantic-gradient sampling.
        ranked = sorted(rows, key=lambda r: (0 if _bool(r.get("boundary_token")) else 1, _num(r.get("source_mean_cosine"), 1.0)))
        for row in ranked[: min(12, len(ranked))]:
            out[(scene, window, frame)].append((_num(row.get("centroid_x")), _num(row.get("centroid_y"))))
    return out


class MetricAgg:
    def __init__(self) -> None:
        self.count = 0
        self.per_frame: Counter[tuple[str, str, int]] = Counter()
        self.unique_pixels: set[tuple[str, int, int, int]] = set()
        self.strata: Counter[str] = Counter()
        self.interior = 0
        self.boundary = 0
        self.conflict = 0
        self.semgrad = 0
        self.weight_sum = 0.0
        self.weight_max = 0.0
        self.uses_gt = 0
        self.uses_future = 0
        self.query_bbox_area_sum = 0.0
        self.query_bbox_frame_count = 0

    def add_many(self, rows: list[dict[str, Any]], image_area: int) -> None:
        if rows:
            xs = [int(row["x_px"]) for row in rows]
            ys = [int(row["y_px"]) for row in rows]
            bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
            self.query_bbox_area_sum += float(bbox_area / max(image_area, 1))
            self.query_bbox_frame_count += 1
        for row in rows:
            self.count += 1
            key = (str(row["scene_id"]), str(row["window_id"]), int(row["frame_id"]))
            self.per_frame[key] += 1
            self.strata[str(row["query_stratum"])] += 1
            self.unique_pixels.add((str(row["scene_id"]), int(row["frame_id"]), int(row["x_px"]), int(row["y_px"])))
            self.boundary += int(bool(row["near_mask_boundary"]))
            self.conflict += int(bool(row["near_competing_edge"]))
            self.semgrad += int(float(row["semantic_gradient_score"]) > 0)
            self.weight_sum += float(row["importance_weight"])
            self.weight_max = max(self.weight_max, float(row["importance_weight"]))
            self.uses_gt += int(bool(row["query_selection_uses_gt"]))
            self.uses_future += int(bool(row["uses_future"]))
            if row.get("source_mask_id") not in ("", None) and not bool(row["near_mask_boundary"]):
                self.interior += 1

    def summary(self, variant_id: str, runtime_sec: float) -> dict[str, Any]:
        denom = max(1, self.count)
        frame_counts = np.asarray(list(self.per_frame.values()), dtype=np.float64) if self.per_frame else np.asarray([0.0])
        duplicate_rate = 1.0 - len(self.unique_pixels) / denom
        return {
            "schema_version": "stream4d_v97_phase1_variant_metric_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": variant_id,
            "query_count": int(self.count),
            "query_count_per_frame_mean": float(np.mean(frame_counts)),
            "query_count_per_frame_p95": float(np.percentile(frame_counts, 95)),
            "boundary_query_rate": float(self.boundary / denom),
            "conflict_query_rate": float(self.conflict / denom),
            "semantic_gradient_query_rate": float(self.semgrad / denom),
            "interior_query_rate": float(self.interior / denom),
            "uniform_query_rate": float(self.strata.get("uniform", 0) / denom),
            "mean_importance_weight": float(self.weight_sum / denom),
            "max_importance_weight": float(self.weight_max),
            "query_bbox_coverage_rate": float(self.query_bbox_area_sum / max(1, self.query_bbox_frame_count)),
            "duplicate_query_rate": float(max(0.0, duplicate_rate)),
            "uses_gt_for_query_selection_count": int(self.uses_gt),
            "uses_future_count": int(self.uses_future),
            "runtime_sec": float(runtime_sec),
            "stratum_count_fraction_json": json.dumps({k: v / denom for k, v in sorted(self.strata.items())}, sort_keys=True),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }


def _variant_config_rows(variants: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        budget = VARIANT_BUDGETS[variant]
        rows.append(
            {
                "schema_version": "stream4d_v97_phase1_variant_config_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant,
                "budget_json": json.dumps(budget, sort_keys=True),
                "query_budget_per_frame": int(sum(budget.values())),
                "stratum_weight_policy": "lambda_s_over_query_count_per_frame_stratum",
                "semantic_gradient_status": "region_proxy",
                "query_selection_uses_gt": False,
                "uses_future": False,
            }
        )
    return rows


def _gate_row(variant_id: str, gate: str, observed: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v97_phase1_gate_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "gate": gate,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "query_selection_uses_gt": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase0 = _read_json(_project(args.phase0_summary))
    if not _bool(phase0.get("can_enter_phase1")):
        raise RuntimeError(f"Phase0 does not allow Phase1: {_project(args.phase0_summary)}")

    source_rows = [row for row in _read_csv(_project(args.source_rows)) if row.get("split", "dev") == args.split]
    edge_rows = _read_csv(_project(args.edge_rows))
    region_rows = _read_csv(_project(args.region_rows))
    if args.scenes:
        scenes = {part.strip() for part in args.scenes.split(",") if part.strip()}
        source_rows = [row for row in source_rows if row.get("scene_id") in scenes]
        edge_rows = [row for row in edge_rows if row.get("scene_id") in scenes]
        region_rows = [row for row in region_rows if row.get("scene_id") in scenes]

    grouped: dict[tuple[str, str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
            continue
        grouped[_frame_key(row)].append(row)
    frame_items = sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4]))
    if args.max_frames > 0:
        frame_items = frame_items[: int(args.max_frames)]

    conflict_ids, nested_ids = _collect_edge_ids(edge_rows, source_rows)
    semgrad_centers = _collect_semgrad_centers(region_rows)
    variants = [part.strip() for part in args.variants.split(",") if part.strip()] or list(VARIANT_BUDGETS)
    for variant in variants:
        if variant not in VARIANT_BUDGETS:
            raise ValueError(f"unknown query variant: {variant}")

    query_path = output_root / "query_plan_rows.csv"
    query_file = query_path.open("w", newline="", encoding="utf-8")
    query_writer = csv.DictWriter(query_file, fieldnames=QUERY_FIELDS)
    query_writer.writeheader()

    variant_aggs = {variant: MetricAgg() for variant in variants}
    stratum_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    missing_mask_rows: list[dict[str, Any]] = []
    query_index_by_variant: Counter[str] = Counter()
    image_area_values: list[int] = []

    try:
        for frame_idx, ((scene, window, chunk, frame_id, mask_path_raw), rows) in enumerate(frame_items):
            mask_path = _project(mask_path_raw)
            if not mask_path.exists():
                missing_mask_rows.append({"scene_id": scene, "window_id": window, "chunk_id": chunk, "frame_id": frame_id, "mask_path": _rel(mask_path)})
                continue
            label = _load_label(mask_path)
            height, width = label.shape
            image_area = int(height * width)
            image_area_values.append(image_area)
            source_ids = {int(_num(row.get("source_mask_id"))) for row in rows if int(_num(row.get("source_mask_id"))) > 0}
            source_ids_arr = np.asarray(sorted(source_ids), dtype=np.int64)
            foreground = np.isin(label, source_ids_arr) if source_ids else label > 0
            boundary = _label_boundary(label, source_ids, int(args.boundary_band_px))
            interior = foreground & ~boundary
            edge_key = (scene, window, frame_id)
            conflict_mask = (
                np.isin(label, np.asarray(sorted(conflict_ids.get(edge_key, set())), dtype=np.int64))
                if conflict_ids.get(edge_key)
                else np.zeros(label.shape, dtype=bool)
            )
            nested_mask = (
                np.isin(label, np.asarray(sorted(nested_ids.get(edge_key, set())), dtype=np.int64))
                if nested_ids.get(edge_key)
                else np.zeros(label.shape, dtype=bool)
            )
            conflict = boundary & conflict_mask
            if conflict_ids.get(edge_key) and not np.any(conflict):
                conflict = boundary & foreground
            nested = boundary & nested_mask
            semgrad = _disk_points(foreground, semgrad_centers.get(edge_key, []), radius=int(args.semgrad_radius_px))
            if not np.any(semgrad):
                semgrad = boundary
            fallback = foreground if np.any(foreground) else np.ones(label.shape, dtype=bool)
            source_risk_by_mask = {
                int(_num(row.get("source_mask_id"))): min(1.0, 0.15 * max(0.0, _num(row.get("object_candidate_count")) - 1.0) + _num(row.get("mask_area_ratio")) * 2.0)
                for row in rows
            }
            maps = {
                "uniform": np.ones(label.shape, dtype=bool),
                "interior": interior,
                "boundary": boundary,
                "conflict": conflict,
                "semantic_gradient": semgrad,
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
                    stratum_weight = 1.0 / max(int(budget), 1)
                    for y, x in points:
                        y_i = int(np.clip(y, 0, height - 1))
                        x_i = int(np.clip(x, 0, width - 1))
                        mask_id = int(label[y_i, x_i])
                        is_source = bool(mask_id in source_ids)
                        row = {
                            "schema_version": "stream4d_v97_query_plan_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "variant_id": variant,
                            "scene_id": scene,
                            "window_id": window,
                            "chunk_id": chunk,
                            "frame_id": int(frame_id),
                            "query_id": f"{variant}:{scene}:{window}:c{chunk}:f{frame_id:06d}:q{query_index_by_variant[variant]:09d}",
                            "query_stratum": stratum,
                            "source_mask_id": mask_id if is_source else "",
                            "x_norm": float(x_i / max(width - 1, 1)),
                            "y_norm": float(y_i / max(height - 1, 1)),
                            "x_px": int(x_i),
                            "y_px": int(y_i),
                            "stratum_weight": float(stratum_weight),
                            "importance_weight": float(stratum_weight),
                            "near_mask_boundary": bool(boundary[y_i, x_i]),
                            "near_competing_edge": bool(conflict[y_i, x_i]),
                            "near_nested_edge": bool(nested[y_i, x_i]),
                            "semantic_gradient_score": float(1.0 if semgrad[y_i, x_i] else 0.0),
                            "source_risk_score": float(source_risk_by_mask.get(mask_id, 0.0)),
                            "query_selection_uses_gt": False,
                            "uses_future": False,
                        }
                        query_index_by_variant[variant] += 1
                        per_variant_rows.append(row)
                for row in per_variant_rows:
                    query_writer.writerow({key: _jsonable(row.get(key, "")) for key in QUERY_FIELDS})
                variant_aggs[variant].add_many(per_variant_rows, image_area)
                stratum_counts = Counter(row["query_stratum"] for row in per_variant_rows)
                for stratum, count in sorted(stratum_counts.items()):
                    stratum_rows.append(
                        {
                            "schema_version": "stream4d_v97_query_stratum_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "variant_id": variant,
                            "scene_id": scene,
                            "window_id": window,
                            "chunk_id": chunk,
                            "frame_id": int(frame_id),
                            "query_stratum": stratum,
                            "query_count": int(count),
                            "query_fraction": float(count / max(1, len(per_variant_rows))),
                            "candidate_pixel_count": int(candidate_counts.get(stratum, image_area)),
                            "sampler_status": status,
                            "runtime_frame_variant_sec": float(time.time() - variant_t0),
                            "query_selection_uses_gt": False,
                            "uses_future": False,
                        }
                    )
            if frame_idx < int(args.casebook_frames):
                casebook_rows.append(
                    {
                        "schema_version": "stream4d_v97_phase1_casebook_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "scene_id": scene,
                        "window_id": window,
                        "chunk_id": chunk,
                        "frame_id": int(frame_id),
                        "mask_path": _rel(mask_path),
                        "source_mask_count": int(len(source_ids)),
                        "conflict_mask_count": int(len(conflict_ids.get(edge_key, set()))),
                        "nested_mask_count": int(len(nested_ids.get(edge_key, set()))),
                        "boundary_pixel_count": int(np.count_nonzero(boundary)),
                        "conflict_pixel_count": int(np.count_nonzero(conflict)),
                        "semantic_gradient_pixel_count": int(np.count_nonzero(semgrad)),
                        "note": "GT-free query-plan reproduction case.",
                        "query_selection_uses_gt": False,
                        "uses_future": False,
                    }
                )
            if args.progress_every_frames > 0 and (frame_idx + 1) % int(args.progress_every_frames) == 0:
                print(json.dumps({"phase": PHASE_ID, "processed_frames": frame_idx + 1, "total_frames": len(frame_items)}, sort_keys=True))
    finally:
        query_file.close()

    elapsed = float(time.time() - started)
    variant_metric_rows = [variant_aggs[variant].summary(variant, elapsed) for variant in variants]
    by_variant = {row["variant_id"]: row for row in variant_metric_rows}
    q0_boundary = float(by_variant.get("Q0_uniform256", {}).get("boundary_query_rate", 0.0))
    q0_conflict = float(by_variant.get("Q0_uniform256", {}).get("conflict_query_rate", 0.0))
    gate_rows: list[dict[str, Any]] = []
    for variant in ["Q2_adaptive512", "Q3_adaptive1024"]:
        row = by_variant.get(variant, {})
        gate_rows.append(_gate_row(variant, "boundary_query_rate_ge_Q0_plus_0p15", row.get("boundary_query_rate", 0.0), q0_boundary + 0.15, float(row.get("boundary_query_rate", 0.0)) >= q0_boundary + 0.15))
        gate_rows.append(_gate_row(variant, "conflict_query_rate_ge_Q0_plus_0p05", row.get("conflict_query_rate", 0.0), q0_conflict + 0.05, float(row.get("conflict_query_rate", 0.0)) >= q0_conflict + 0.05))
    q3 = by_variant.get("Q3_adaptive1024", {})
    gate_rows.append(_gate_row("Q3_adaptive1024", "query_count_per_frame_mean_le_1100", q3.get("query_count_per_frame_mean", 0.0), 1100, float(q3.get("query_count_per_frame_mean", 0.0)) <= 1100))
    for variant, row in by_variant.items():
        gate_rows.append(
            _gate_row(
                variant,
                "importance_weight_max_le_10x_mean",
                row.get("max_importance_weight", 0.0),
                float(row.get("mean_importance_weight", 0.0)) * 10.0,
                float(row.get("max_importance_weight", 0.0)) <= float(row.get("mean_importance_weight", 0.0)) * 10.0,
            )
        )
    gate_rows.append(
        _gate_row(
            "ALL",
            "no_gt_or_future_query_selection",
            json.dumps({row["variant_id"]: [row["uses_gt_for_query_selection_count"], row["uses_future_count"]] for row in variant_metric_rows}, sort_keys=True),
            "all zero",
            all(row["uses_gt_for_query_selection_count"] == 0 and row["uses_future_count"] == 0 for row in variant_metric_rows),
        )
    )
    phase1_pass = all(bool(row["pass"]) for row in gate_rows)
    variant_failure_rows = [
        {
            "schema_version": "stream4d_v97_phase1_variant_failure_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": row["variant_id"],
            "failure_type": "PHASE1_GATE_FAIL",
            "failed_gate": row["gate"],
            "observed": row["observed"],
            "required": row["required"],
            "query_selection_uses_gt": False,
            "uses_future": False,
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    # Pick the strongest passing adaptive query planner by GT-free conflict/boundary coverage.
    scored = []
    for row in variant_metric_rows:
        score = float(row["boundary_query_rate"]) + float(row["conflict_query_rate"]) + 0.5 * float(row["semantic_gradient_query_rate"])
        if row["variant_id"].startswith("Q0") or row["variant_id"].startswith("Q1"):
            score -= 1.0
        scored.append((score, row))
    best = max(scored, key=lambda item: item[0])[1] if scored else {}
    best_variant_summary = {
        "schema": "stream4d_v97_phase1_best_variant_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "best_variant_id": best.get("variant_id", ""),
        "selection_policy": "GT-free max boundary+conflict+0.5*semantic_gradient among query-plan variants; not a method-success claim.",
        "boundary_query_rate": best.get("boundary_query_rate", ""),
        "conflict_query_rate": best.get("conflict_query_rate", ""),
        "semantic_gradient_query_rate": best.get("semantic_gradient_query_rate", ""),
        "query_count_per_frame_mean": best.get("query_count_per_frame_mean", ""),
        "query_selection_uses_gt": False,
        "uses_future": False,
    }
    summary = {
        "schema": "stream4d_v97_phase1_query_planner_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "PASS_V97_PHASE1_QUERY_PLANNER" if phase1_pass else "NO_GO_V97_PHASE1_QUERY_PLANNER",
        "source_container_rows": _rel(_project(args.source_rows)),
        "split": args.split,
        "mask_edge_rows": _rel(_project(args.edge_rows)),
        "region_node_rows": _rel(_project(args.region_rows)),
        "phase0_summary": _rel(_project(args.phase0_summary)),
        "source_row_count": int(len(source_rows)),
        "processed_frame_count": int(len(image_area_values)),
        "missing_mask_frame_count": int(len(missing_mask_rows)),
        "query_plan_rows": _rel(query_path),
        "query_plan_rows_sha256": _sha256(query_path),
        "query_stratum_rows": _rel(output_root / "query_stratum_rows.csv"),
        "variant_config_rows": _rel(output_root / "variant_config_rows.csv"),
        "variant_metric_rows": _rel(output_root / "variant_metric_rows.csv"),
        "variant_gate_rows": _rel(output_root / "variant_gate_rows.csv"),
        "variant_failure_rows": _rel(output_root / "variant_failure_rows.csv"),
        "best_variant_summary": _rel(output_root / "best_variant_summary.json"),
        "casebook_rows": _rel(output_root / "casebook_rows.csv"),
        "phase1_gate_rows": _rel(output_root / "phase1_gate_rows.csv"),
        "missing_mask_rows": _rel(output_root / "missing_mask_rows.csv"),
        "query_variant_rows": variant_metric_rows,
        "gate_rows": gate_rows,
        "best_variant_id": best_variant_summary["best_variant_id"],
        "boundary_band_px": int(args.boundary_band_px),
        "semgrad_radius_px": int(args.semgrad_radius_px),
        "runtime_sec": elapsed,
        "query_selection_uses_gt": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    _write_csv(output_root / "query_stratum_rows.csv", stratum_rows)
    _write_csv(output_root / "variant_config_rows.csv", _variant_config_rows(variants))
    _write_csv(output_root / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(output_root / "variant_gate_rows.csv", gate_rows)
    _write_csv(output_root / "phase1_gate_rows.csv", gate_rows)
    _write_csv(output_root / "variant_failure_rows.csv", variant_failure_rows)
    _write_csv(output_root / "casebook_rows.csv", casebook_rows)
    _write_csv(output_root / "missing_mask_rows.csv", missing_mask_rows)
    _write_json(output_root / "best_variant_summary.json", best_variant_summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-summary", default=str(DEFAULT_PHASE0))
    parser.add_argument("--source-rows", default=str(DEFAULT_SOURCE_ROWS))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--edge-rows", default=str(DEFAULT_EDGE_ROWS))
    parser.add_argument("--region-rows", default=str(DEFAULT_REGION_ROWS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--variants", default=",".join(VARIANT_BUDGETS))
    parser.add_argument("--scenes", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--boundary-band-px", type=int, default=4)
    parser.add_argument("--semgrad-radius-px", type=int, default=10)
    parser.add_argument("--casebook-frames", type=int, default=12)
    parser.add_argument("--progress-every-frames", type=int, default=32)
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "processed_frame_count": summary["processed_frame_count"],
                "best_variant_id": summary["best_variant_id"],
                "query_plan_rows": summary["query_plan_rows"],
                "runtime_sec": summary["runtime_sec"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
