#!/usr/bin/env python3
"""Materialize a Phase3A smoke from v94 object-axis unary shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v92_phase5_source_container_field as v92field  # noqa: E402
from tools import build_v93_phase5_boundary_affinity_field as phase5  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


PHASE_ID = "v94_phase3A_object_axis_smoke"
RUN_ID = "v94_phase3A_object_axis_smoke"
OUT = ROOT / "outputs/audit/v94_phase3A_object_axis_smoke"
DEFAULT_FIELD_ROOT = ROOT / "outputs/audit/v94_phase7c_object_axis_field_smoke"
V94_PHASE0 = ROOT / "outputs/audit/v94_phase0_fact_lock"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return WORKSPACE_ROOT / path
    return ROOT / path


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(WORKSPACE_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
    return float(out) if math.isfinite(out) else float(default)


def _int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _mean(values: list[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _source_key_parts(raw: str) -> tuple[str, str, int, int]:
    parts = str(raw).split("|")
    if len(parts) == 4:
        return parts[0], parts[1], int(parts[2]), int(parts[3])
    if len(parts) == 3:
        return parts[0], "", int(parts[1]), int(parts[2])
    raise ValueError(f"bad source key: {raw}")


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "OA0_whole_source_per_object_control",
            "family": "control",
            "mode": "whole_source",
            "description": "Assign every object the whole source container; frame WTA keeps only the highest scoring overlap.",
        },
        {
            "variant_id": "OA1_argmax_all_regions",
            "family": "real",
            "mode": "argmax",
            "min_margin": -999.0,
            "min_top_score": -999.0,
            "description": "Multi-object region WTA: every source region goes to the object with highest RADIO prototype cosine.",
        },
        {
            "variant_id": "OA2_argmax_margin_0p02",
            "family": "real",
            "mode": "argmax",
            "min_margin": 0.02,
            "min_top_score": -999.0,
            "description": "Only materialize WTA regions whose top-object margin is at least 0.02.",
        },
        {
            "variant_id": "OA3_argmax_margin_0p05",
            "family": "real",
            "mode": "argmax",
            "min_margin": 0.05,
            "min_top_score": -999.0,
            "description": "Stricter object-axis WTA margin smoke.",
        },
        {
            "variant_id": "OA4_source_local_top70",
            "family": "real",
            "mode": "top_quantile",
            "top_quantile": 0.70,
            "description": "Keep only regions whose winning unary is in the source-local top 70 percent.",
        },
        {
            "variant_id": "OA5_object_score_top50_no_wta",
            "family": "real",
            "mode": "object_quantile",
            "score_quantile": 0.50,
            "description": "Competition-softened repair: each object keeps regions in its own top 50 percent unary scores, without region-label WTA.",
        },
        {
            "variant_id": "OA6_object_score_top35_no_wta",
            "family": "real",
            "mode": "object_quantile",
            "score_quantile": 0.65,
            "description": "Tighter competition-softened repair: each object keeps regions in its own top 35 percent unary scores, without region-label WTA.",
        },
        {
            "variant_id": "OA7_argmax_plus_object_top50_floor",
            "family": "real",
            "mode": "argmax_plus_object_quantile",
            "score_quantile": 0.50,
            "min_object_margin": -0.02,
            "description": "WTA repair with a per-object top-50 coverage floor, allowing mildly losing regions to reduce undercoverage.",
        },
        {
            "variant_id": "OA8_argmax_plus_object_top35_floor",
            "family": "real",
            "mode": "argmax_plus_object_quantile",
            "score_quantile": 0.65,
            "min_object_margin": -0.02,
            "description": "WTA repair with a tighter per-object top-35 coverage floor.",
        },
        {
            "variant_id": "OA_CTRL_shuffled_object_axis",
            "family": "control",
            "mode": "shuffle_argmax",
            "description": "Deterministically permute the object label axis inside each source before materialization.",
        },
        {
            "variant_id": "OA_CTRL_shuffled_object_score_top50",
            "family": "control",
            "mode": "shuffled_object_quantile",
            "score_quantile": 0.50,
            "description": "Control for object-score top-fraction variants: use a deterministic wrong object score row for each object.",
        },
    ]


def _load_region_nodes(path: Path, source_keys: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], dict[int, dict[str, Any]]]:
    out: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("scene_id", "")), _int(row.get("frame_id"), -1), _int(row.get("source_mask_id"), -1))
            if key not in source_keys:
                continue
            idx = _int(row.get("region_index"), -1)
            if idx >= 0:
                out[key][idx] = dict(row)
    return out


def _selected_for_variant(
    spec: dict[str, Any],
    object_local_index: int,
    scores: np.ndarray,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
    margins: np.ndarray,
    rng_seed_text: str,
) -> set[int]:
    mode = str(spec.get("mode", "argmax"))
    if mode == "whole_source":
        return set(range(scores.shape[1]))
    assigned = np.asarray(top_idx, dtype=np.int32)
    score_object_index = int(object_local_index)
    if mode == "shuffled_object_quantile" and scores.shape[0] > 1:
        rng = np.random.default_rng(_stable_seed(rng_seed_text))
        perm = rng.permutation(scores.shape[0])
        score_object_index = int(perm[int(object_local_index)])
    object_scores = scores[score_object_index, :]
    if scores.shape[0] >= 2:
        other_scores = np.delete(scores, score_object_index, axis=0)
        object_margins = object_scores - np.max(other_scores, axis=0)
    else:
        object_margins = np.zeros_like(object_scores)
    if mode == "shuffle_argmax" and scores.shape[0] > 1:
        rng = np.random.default_rng(_stable_seed(rng_seed_text))
        perm = rng.permutation(scores.shape[0])
        assigned = perm[assigned]
    keep = assigned == int(object_local_index)
    if mode == "argmax":
        keep &= margins >= float(spec.get("min_margin", -999.0))
        keep &= top_scores >= float(spec.get("min_top_score", -999.0))
    elif mode == "top_quantile":
        threshold = float(np.quantile(top_scores, float(spec.get("top_quantile", 0.70)))) if top_scores.size else float("inf")
        keep &= top_scores >= threshold
    elif mode in {"object_quantile", "shuffled_object_quantile"}:
        threshold = float(np.quantile(object_scores, float(spec.get("score_quantile", 0.50)))) if object_scores.size else float("inf")
        keep = object_scores >= threshold
    elif mode == "argmax_plus_object_quantile":
        threshold = float(np.quantile(object_scores, float(spec.get("score_quantile", 0.50)))) if object_scores.size else float("inf")
        floor = (object_scores >= threshold) & (object_margins >= float(spec.get("min_object_margin", -999.0)))
        keep |= floor
    return {int(i) for i in np.nonzero(keep)[0].tolist()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = _resolve(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    field_root = _resolve(args.field_root)
    shard_paths = sorted((field_root / "field_shards").glob("object_axis_unary_shard_*.npz"))
    if int(args.max_shards) > 0:
        shard_paths = shard_paths[: int(args.max_shards)]
    if not shard_paths:
        raise RuntimeError(f"No object-axis unary shards found under {field_root / 'field_shards'}")

    specs = _variant_specs()
    variant_ids = [spec["variant_id"] for spec in specs]
    created_at = _created_at()
    source_meta = v92field._load_source_meta(_resolve(args.source_container_rows))
    source_key_texts: list[str] = []
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as data:
            source_key_texts.extend(str(value) for value in data["source_keys"].tolist())
    physical_source_keys = {(scene, frame_id, mask_id) for scene, _window, frame_id, mask_id in (_source_key_parts(raw) for raw in source_key_texts)}
    node_maps = _load_region_nodes(_resolve(args.region_node_rows), physical_source_keys)

    writer = phase5.ScoreWTAFrameWriter(out, variant_ids)
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    generated_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    source_summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    config_rows = [
        {
            "schema_version": "stream4d_v94_phase3A_object_axis_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": spec["variant_id"],
            "family": spec["family"],
            "mode": spec["mode"],
            "description": spec["description"],
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for spec in specs
    ]

    processed_source_count = 0
    materialized_object_count = 0
    score_protocol_counts: Counter[str] = Counter()
    source_object_counts: list[float] = []
    source_region_counts: list[float] = []

    for shard_i, shard_path in enumerate(shard_paths):
        with np.load(shard_path, allow_pickle=False) as data:
            source_keys = [str(value) for value in data["source_keys"].tolist()]
            object_keys = [str(value) for value in data["object_keys"].tolist()]
            object_source_index = data["object_source_index"].astype(np.int32)
            object_local_index = data["object_local_index"].astype(np.int32)
            region_source_index = data["region_source_index"].astype(np.int32)
            region_indices = data["region_indices"].astype(np.int32)
            unary_source_index = data["unary_source_index"].astype(np.int32)
            unary_object_local_index = data["unary_object_local_index"].astype(np.int32)
            unary_region_local_index = data["unary_region_local_index"].astype(np.int32)
            unary_cosine = data["unary_cosine"].astype(np.float32)

            for source_idx, raw_source_key in enumerate(source_keys):
                scene, window_id, frame_id, mask_id = _source_key_parts(raw_source_key)
                physical_key = (scene, frame_id, mask_id)
                meta = source_meta.get(physical_key)
                node_map = node_maps.get(physical_key, {})
                if not meta or not node_map:
                    failure_rows.append(
                        {
                            "schema_version": "stream4d_v94_phase3A_object_axis_failure_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_key": raw_source_key,
                            "failure_type": "missing_source_meta_or_region_nodes",
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                    continue
                region_mask = region_source_index == source_idx
                source_region_indices = region_indices[region_mask]
                nodes = [node_map.get(int(region_index)) for region_index in source_region_indices.tolist()]
                if any(node is None for node in nodes):
                    failure_rows.append(
                        {
                            "schema_version": "stream4d_v94_phase3A_object_axis_failure_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_key": raw_source_key,
                            "failure_type": "missing_region_node_for_shard_index",
                            "missing_count": sum(1 for node in nodes if node is None),
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                    continue
                nodes_typed: list[dict[str, Any]] = [node for node in nodes if node is not None]
                object_mask = object_source_index == source_idx
                object_pairs = sorted(
                    [(int(local_idx), object_keys[obj_pos]) for obj_pos, local_idx in enumerate(object_local_index) if bool(object_mask[obj_pos])],
                    key=lambda item: item[0],
                )
                if not object_pairs or not nodes_typed:
                    continue
                object_by_local = {local_idx: object_key for local_idx, object_key in object_pairs}
                k_objects = max(object_by_local) + 1
                r_regions = len(nodes_typed)
                scores = np.full((k_objects, r_regions), -1.0e9, dtype=np.float32)
                source_unary_mask = unary_source_index == source_idx
                for obj_idx, region_idx, value in zip(
                    unary_object_local_index[source_unary_mask],
                    unary_region_local_index[source_unary_mask],
                    unary_cosine[source_unary_mask],
                    strict=False,
                ):
                    if 0 <= int(obj_idx) < k_objects and 0 <= int(region_idx) < r_regions:
                        scores[int(obj_idx), int(region_idx)] = float(value)
                finite_cols = np.isfinite(scores).all(axis=0) & (np.max(scores, axis=0) > -1.0e8)
                if not np.any(finite_cols):
                    continue
                top_idx = np.argmax(scores, axis=0).astype(np.int32)
                sorted_scores = np.sort(scores, axis=0)
                top_scores = sorted_scores[-1, :]
                second_scores = sorted_scores[-2, :] if k_objects >= 2 else np.zeros_like(top_scores)
                margins = top_scores - second_scores

                mask_path = _resolve(str(meta.get("mask_path", "")))
                frame_key = (scene, frame_id)
                if frame_key not in label_cache:
                    label_cache[frame_key] = v92field._read_label(mask_path)
                label = label_cache[frame_key]
                source_mask = label == int(mask_id)
                if not np.any(source_mask):
                    continue
                writer.ensure_frame(scene, frame_id, label.shape)
                source_area = int(np.count_nonzero(source_mask))
                processed_source_count += 1
                source_object_counts.append(float(len(object_pairs)))
                source_region_counts.append(float(r_regions))

                for spec in specs:
                    variant_id = str(spec["variant_id"])
                    for object_local_idx, object_key in object_pairs:
                        selected = _selected_for_variant(
                            spec,
                            object_local_idx,
                            scores,
                            top_idx,
                            top_scores,
                            margins,
                            f"{raw_source_key}|{object_key}|{variant_id}",
                        )
                        if not selected:
                            continue
                        mask = v92field._node_mask(nodes_typed, selected, source_mask)
                        if not np.any(mask):
                            continue
                        selected_scores = [float(scores[object_local_idx, idx]) for idx in selected]
                        if scores.shape[0] >= 2:
                            other_scores = np.delete(scores, int(object_local_idx), axis=0)
                            object_margins = scores[int(object_local_idx), :] - np.max(other_scores, axis=0)
                        else:
                            object_margins = np.zeros(scores.shape[1], dtype=np.float32)
                        selected_margins = [float(object_margins[idx]) for idx in selected]
                        area = int(np.count_nonzero(mask))
                        area_ratio = float(area / max(1, source_area))
                        object_score = float(0.75 * _mean(selected_scores) + 0.20 * _mean(selected_margins) + 0.05 * min(1.0, area_ratio))
                        score_protocol = "object_axis_radio_proto_cosine_margin_area"
                        score_protocol_counts[score_protocol] += 1
                        generated_row = {
                            "schema_version": "stream4d_v94_phase3A_object_axis_generated_mask_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "variant_id": variant_id,
                            "scene_id": scene,
                            "split": "dev",
                            "window_id": window_id,
                            "frame_id": frame_id,
                            "source_mask_id": mask_id,
                            "new_mask_id": "",
                            "object_hypothesis_id": object_key,
                            "generated_mask_path": _rel(out / "generated_masks" / variant_id / scene / "mask" / f"{int(frame_id)}.png"),
                            "source_mask_area": source_area,
                            "generated_mask_area_before_frame_wta": area,
                            "generated_mask_area": area,
                            "generated_mask_area_ratio": area_ratio,
                            "selected_region_count": len(selected),
                            "total_region_count": r_regions,
                            "mean_selected_unary_cosine": _mean(selected_scores),
                            "mean_selected_margin": _mean(selected_margins),
                            "score_protocol": score_protocol,
                            "solver_backend": "object_axis_region_wta",
                            "gpu_device": "phase7c_cuda_unary_input",
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                        mv_row = {
                            "split": "dev",
                            "scene_id": scene,
                            "source_variant": variant_id,
                            "variant": variant_id,
                            "mv_object_id": f"{variant_id}:{object_key}",
                            "frame_id": frame_id,
                            "mask_id": "",
                            "frame_mask_score": object_score,
                            "object_score": object_score,
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                            "uses_rgbd_pose_mesh": False,
                            "materializable": True,
                            "selection_reason": f"v94_object_axis_{spec['mode']}_{score_protocol}",
                        }
                        writer.add(variant_id, mask, object_score, generated_row, mv_row)
                        materialized_object_count += 1
                        assignment_rows.append(
                            {
                                "schema_version": "stream4d_v94_phase3A_object_axis_assignment_summary_v1",
                                "phase_id": PHASE_ID,
                                "run_id": RUN_ID,
                                "variant_id": variant_id,
                                "scene_id": scene,
                                "window_id": window_id,
                                "frame_id": frame_id,
                                "source_mask_id": mask_id,
                                "canonical_object_key": object_key,
                                "object_local_index": object_local_idx,
                                "selected_region_count": len(selected),
                                "total_region_count": r_regions,
                                "selected_region_fraction": float(len(selected) / max(1, r_regions)),
                                "generated_mask_area_ratio": area_ratio,
                                "mean_selected_unary_cosine": _mean(selected_scores),
                                "mean_selected_margin": _mean(selected_margins),
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                            }
                        )
                source_summary_rows.append(
                    {
                        "schema_version": "stream4d_v94_phase3A_object_axis_source_summary_v1",
                        "phase_id": PHASE_ID,
                        "run_id": RUN_ID,
                        "source_key": raw_source_key,
                        "canonical_object_count": len(object_pairs),
                        "region_count": r_regions,
                        "top_unary_mean": float(np.mean(top_scores)),
                        "unary_margin_mean": float(np.mean(margins)),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
        if (shard_i + 1) % max(1, int(args.progress_every_shards)) == 0:
            print(
                json.dumps(
                    {
                        "phase": PHASE_ID,
                        "processed_shards": shard_i + 1,
                        "processed_source_count": processed_source_count,
                        "materialized_object_count_before_frame_wta": materialized_object_count,
                        "elapsed_sec": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    writer.flush()
    generated_rows = writer.generated_rows
    mv_rows = writer.mv_rows
    object_rows = phase5._object_rows_from_mv(mv_rows)
    radius_sweep.OUT = out
    metric_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    for variant_id in variant_ids:
        rows = [row for row in mv_rows if row.get("variant") == variant_id]
        metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
        metric_rows.extend(metrics)
        casebook_rows.extend({**row, "variant_id": variant_id} for row in cases)
    aggregate_rows = phase7d._aggregate(metric_rows)
    assignment_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignment_rows:
        assignment_by_variant[str(row.get("variant_id", ""))].append(row)
    variant_metric_rows: list[dict[str, Any]] = []
    for row in aggregate_rows:
        variant_id = str(row.get("variant_id", ""))
        group = assignment_by_variant.get(variant_id, [])
        variant_metric_rows.append(
            {
                "schema_version": "stream4d_v94_phase3A_object_axis_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "scene_id": "ALL_DEV_SMOKE",
                "split": "dev_smoke",
                "source_artifact": _rel(out / "mv_metric_aggregate_rows.csv"),
                "created_at": created_at,
                **row,
                "mean_generated_area_ratio": _mean([_num(item.get("generated_mask_area_ratio")) for item in group]),
                "object_region_count_mean": _mean([_num(item.get("selected_region_count")) for item in group]),
                "selected_region_fraction_mean": _mean([_num(item.get("selected_region_fraction")) for item in group]),
                "mean_selected_unary_cosine": _mean([_num(item.get("mean_selected_unary_cosine")) for item in group]),
                "mean_selected_margin": _mean([_num(item.get("mean_selected_margin")) for item in group]),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    phase0 = _read_json(V94_PHASE0 / "summary.json")
    required_ap = _num(phase0.get("required_MV_AP_window"))
    required_ap50 = _num(phase0.get("required_MV_AP50_window"))
    gate_rows: list[dict[str, Any]] = []
    failure_rows_out: list[dict[str, Any]] = []
    any_smoke_materialized = len(mv_rows) > 0 and not failure_rows
    full_dev_gate_evaluated = bool(args.full_dev_eval)
    for row in variant_metric_rows:
        variant_id = str(row.get("variant_id", ""))
        mv_ap = _num(row.get("mean_MV_AP_window"))
        mv_ap50 = _num(row.get("mean_MV_AP50_window"))
        missing_mask_raster_count = int(_num(row.get("missing_mask_raster_count")))
        same_frame_collision_count = int(_num(row.get("same_frame_collision_count")))
        dev_gate_pass = bool(
            full_dev_gate_evaluated
            and any_smoke_materialized
            and mv_ap >= required_ap
            and mv_ap50 >= required_ap50
            and missing_mask_raster_count == 0
            and same_frame_collision_count == 0
        )
        gate_rows.append(
            {
                "schema_version": "stream4d_v94_phase3A_object_axis_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "smoke_materialization_gate_pass": any_smoke_materialized,
                "full_dev_gate_evaluated": full_dev_gate_evaluated,
                "dev_progress_gate_pass": dev_gate_pass,
                "MV_AP_window_smoke": mv_ap,
                "MV_AP50_window_smoke": mv_ap50,
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "same_frame_collision_count": same_frame_collision_count,
                "missing_mask_raster_count": missing_mask_raster_count,
                "required_MV_AP_window_full_dev": required_ap,
                "required_MV_AP50_window_full_dev": required_ap50,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if not dev_gate_pass:
            failure_rows_out.append(
                {
                    "schema_version": "stream4d_v94_phase3A_object_axis_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "FULL_DEV_GATE_FAILED" if full_dev_gate_evaluated else "SMOKE_METRIC_BELOW_FULL_DEV_GATE_OR_NOT_COMPARABLE",
                    "MV_AP_window_smoke": mv_ap,
                    "MV_AP50_window_smoke": mv_ap50,
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "same_frame_collision_count": same_frame_collision_count,
                    "missing_mask_raster_count": missing_mask_raster_count,
                    "repair_direction": "Inspect assignment extent, missing-raster coverage, object ranking, and stronger pairwise/barrier variants." if full_dev_gate_evaluated else "Run full-dev object-axis materialization before drawing performance conclusions; if full-dev remains low, inspect assignment extent and object ranking.",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )

    best_real = max(
        [row for row in variant_metric_rows if str(row.get("variant_id", "")).startswith("OA") and "CTRL" not in str(row.get("variant_id", "")) and "OA0" not in str(row.get("variant_id", ""))],
        key=lambda row: (_num(row.get("mean_MV_AP_window"), -999.0), _num(row.get("mean_MV_AP50_window"), -999.0)),
        default={},
    )
    best_real_dev_gate_pass = bool(
        full_dev_gate_evaluated
        and best_real
        and _num(best_real.get("mean_MV_AP_window")) >= required_ap
        and _num(best_real.get("mean_MV_AP50_window")) >= required_ap50
        and int(_num(best_real.get("missing_mask_raster_count"))) == 0
        and int(_num(best_real.get("same_frame_collision_count"))) == 0
    )
    _write_csv(out / "variant_config_rows.csv", config_rows)
    _write_csv(out / "generated_mask_rows.csv", generated_rows)
    _write_csv(out / "mv_object_rows.csv", object_rows)
    _write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    _write_csv(out / "assignment_summary_rows.csv", assignment_rows)
    _write_csv(out / "source_summary_rows.csv", source_summary_rows)
    _write_csv(out / "source_failure_rows.csv", failure_rows)
    _write_csv(out / "mv_metric_rows.csv", metric_rows)
    _write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_csv(out / "variant_failure_rows.csv", failure_rows_out)
    _write_csv(out / "casebook_rows.csv", casebook_rows)

    summary = {
        "schema": "stream4d_v94_phase3A_object_axis_smoke_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": (
            "PASS_V94_PHASE3A_OBJECT_AXIS_FULL_DEV_GATE"
            if best_real_dev_gate_pass
            else ("NO_GO_V94_PHASE3A_OBJECT_AXIS_FULL_DEV_GATE" if full_dev_gate_evaluated else ("PASS_V94_PHASE3A_OBJECT_AXIS_SMOKE_MATERIALIZED" if any_smoke_materialized else "NO_GO_V94_PHASE3A_OBJECT_AXIS_SMOKE_FAILED"))
        ),
        "created_at": created_at,
        "duration_sec": float(time.time() - started),
        "field_root": _rel(field_root),
        "field_shard_count": len(shard_paths),
        "processed_source_count": processed_source_count,
        "materialized_object_count_before_frame_wta": materialized_object_count,
        "generated_mask_rows_after_frame_wta": len(generated_rows),
        "mv_object_frame_mask_rows": len(mv_rows),
        "variant_count": len(specs),
        "score_protocol_counts": dict(score_protocol_counts),
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window_smoke": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window_smoke": best_real.get("mean_MV_AP50_window", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "dev_progress_gate_pass": best_real_dev_gate_pass,
        "full_dev_gate_evaluated": full_dev_gate_evaluated,
        "smoke_only_not_dev_success": not full_dev_gate_evaluated,
        "required_MV_AP_window_full_dev": required_ap,
        "required_MV_AP50_window_full_dev": required_ap50,
        "source_object_count_mean": _mean(source_object_counts),
        "source_region_count_mean": _mean(source_region_counts),
        "failure_count": len(failure_rows),
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_rows": len(object_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "assignment_summary_rows": len(assignment_rows),
            "source_summary_rows": len(source_summary_rows),
            "source_failure_rows": len(failure_rows),
            "mv_metric_rows": len(metric_rows),
            "variant_metric_rows": len(variant_metric_rows),
            "variant_gate_rows": len(gate_rows),
            "variant_failure_rows": len(failure_rows_out),
            "casebook_rows": len(casebook_rows),
        },
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                field_root / "summary.json",
                *shard_paths,
                _resolve(args.source_container_rows),
                _resolve(args.region_node_rows),
            ]
            if path.exists()
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "summary.json",
        out / "variant_config_rows.csv",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "assignment_summary_rows.csv",
        out / "source_summary_rows.csv",
        out / "source_failure_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "variant_metric_rows.csv",
        out / "variant_gate_rows.csv",
        out / "variant_failure_rows.csv",
        out / "casebook_rows.csv",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--field-root", default=str(DEFAULT_FIELD_ROOT))
    parser.add_argument("--source-container-rows", default=str(ROOT / "outputs/audit/v94_phase1_canonical_graph/container_rows.csv"))
    parser.add_argument("--region-node-rows", default=str(ROOT / "outputs/audit/v94_phase1_canonical_graph/region_node_rows.csv"))
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--progress-every-shards", type=int, default=1)
    parser.add_argument("--full-dev-eval", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
