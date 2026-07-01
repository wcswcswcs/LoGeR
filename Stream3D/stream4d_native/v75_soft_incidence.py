from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SCENE_ROOTS = {
    "scene0011_00": "outputs/audit/v66_soma_fullscene_pipeline_scene0011_00_stride5_conf02_integrated_d4rt",
    "scene0050_00": "outputs/audit/v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt",
    "scene0030_00": "outputs/audit/v66_soma_fullscene_pipeline_scene0030_00_stride5_conf02_integrated_d4rt",
    "scene0081_01": "outputs/audit/v66_soma_fullscene_pipeline_scene0081_01_stride5_conf02_integrated_d4rt",
    "scene0591_00": "outputs/audit/v66_soma_fullscene_pipeline_scene0591_00_stride5_conf02_integrated_d4rt",
}

VARIANT_ORDER = [
    "I0_observed_mask_id_hard",
    "I1_uv_soft_sigma_fixed",
    "I2_uv_soft_confidence_sigma",
    "I3_uv_soft_confidence_jitter_sigma",
    "I4_uv_soft_with_boundary_band",
]

ROW_DEFAULTS = {
    "uses_gt_for_prediction": False,
    "diagnostic_only": False,
    "forbidden_for_method_table": False,
    "method_prediction_safe": True,
}


@dataclass(slots=True)
class Obs:
    scene_id: str
    chunk_id: int
    frame_id: int
    carrier_id: str
    carrier_global_id: str
    uv_x: float
    uv_y: float
    visible: bool
    confidence: float
    visibility_prob: float
    valid_uv: bool
    mask_id: int
    mask_area: int
    support_density: float
    is_boundary_region: bool
    uses_gt_for_prediction: bool


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _sigmoid(value: float) -> float:
    if value >= 50:
        return 1.0
    if value <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask_dir_from_pipeline(pipeline_root: Path) -> Path:
    summary = _load_json(pipeline_root / "pipeline_summary.json")
    mask_dir = ((summary.get("mask_frame_coverage") or {}).get("mask_dir") or "").strip()
    if not mask_dir:
        return pipeline_root / "cropformer_masks"
    return _rooted(mask_dir)


def _read_label_and_boundary(mask_dir: Path, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    path = mask_dir / f"{int(frame_id)}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask label png: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    label = np.asarray(image, dtype=np.int32)
    boundary = np.zeros(label.shape, dtype=bool)
    boundary[:-1, :] |= label[:-1, :] != label[1:, :]
    boundary[1:, :] |= label[:-1, :] != label[1:, :]
    boundary[:, :-1] |= label[:, :-1] != label[:, 1:]
    boundary[:, 1:] |= label[:, :-1] != label[:, 1:]
    dist_input = (~boundary).astype(np.uint8)
    dist = cv2.distanceTransform(dist_input, cv2.DIST_L2, 3)
    return label, dist


def _load_chunk_defs(pipeline_root: Path, max_chunk_exclusive: int) -> dict[int, dict[str, Any]]:
    path = pipeline_root / "chunk_universe" / "chunk_rows.csv"
    chunks: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            chunk = _int(row.get("chunk_index"), -1)
            if 0 <= chunk < max_chunk_exclusive:
                chunks[chunk] = row
    return chunks


def _load_mask_info(pipeline_root: Path, chunks: dict[int, dict[str, Any]]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, float]]:
    path = pipeline_root / "observation_tables" / "mask_observation_table.csv"
    mask_info: dict[tuple[int, int], dict[str, Any]] = {}
    gt_by_chunk: dict[int, set[int]] = {chunk: set() for chunk in chunks}
    frame_to_chunks: dict[int, list[int]] = defaultdict(list)
    for chunk, row in chunks.items():
        start = _int(row.get("raw_frame_start"), 0)
        end = _int(row.get("raw_frame_end"), -1)
        for frame in range(start, end + 1, 5):
            frame_to_chunks[frame].append(chunk)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = _int(row.get("frame_id"), -1)
            mask_id = _int(row.get("mask_id"), 0)
            if frame not in frame_to_chunks or mask_id <= 0:
                continue
            key = (frame, mask_id)
            mask_info[key] = {
                "mask_observation_id": row.get("mask_observation_id") or f"{row.get('scene_id')}:{frame}:{mask_id}",
                "mask_area": _int(row.get("mask_area"), 0),
                "support_density": _float(row.get("support_density"), 0.0),
                "uses_gt_for_prediction": _bool(row.get("uses_gt_for_prediction")),
                "uses_gt_for_diagnostic_labels": _bool(row.get("uses_gt_for_diagnostic_labels")),
            }
            gt_instance = _int(row.get("diagnostic_gt_instance"), 0)
            if gt_instance > 0:
                for chunk in frame_to_chunks[frame]:
                    gt_by_chunk[chunk].add(gt_instance)
    return mask_info, {chunk: float(len(values)) for chunk, values in gt_by_chunk.items()}


def _load_semantic_entropy(scenes: set[str], max_chunk_exclusive: int) -> dict[tuple[str, int, int], float]:
    path = ROOT / "outputs/audit/v71_semantic_features/mask_feature_rows.csv"
    out: dict[tuple[str, int, int], float] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or row.get("scene") or "")
            if scene not in scenes:
                continue
            chunk_text = str(row.get("chunk_id") or "")
            if ":chunk" in chunk_text:
                chunk = _int(chunk_text.split(":chunk")[-1], -1)
                if not (0 <= chunk < max_chunk_exclusive):
                    continue
            frame = _int(row.get("frame_id"), -1)
            mask_id = _int(row.get("mask_id"), 0)
            if frame >= 0 and mask_id > 0:
                out[(scene, frame, mask_id)] = _float(row.get("semantic_entropy"), 0.0)
    return out


def _read_observations(
    scene_id: str,
    pipeline_root: Path,
    max_chunk_exclusive: int,
) -> tuple[list[Obs], dict[int, set[str]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    table = pipeline_root / "observation_tables" / "carrier_observation_table.csv"
    observations: list[Obs] = []
    total_carriers_by_chunk: dict[int, set[str]] = defaultdict(set)
    raw_stats: dict[int, dict[str, Any]] = defaultdict(lambda: {"visible_frame_by_carrier": defaultdict(set), "confidence_sum": 0.0, "confidence_count": 0})
    gt_violation_rows: list[dict[str, Any]] = []
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            chunk = _int(row.get("chunk_id"), -1)
            if not (0 <= chunk < max_chunk_exclusive):
                continue
            carrier_global = str(row.get("carrier_global_id") or row.get("carrier_id") or "")
            if carrier_global:
                total_carriers_by_chunk[chunk].add(carrier_global)
            visible = _bool(row.get("visible"))
            valid = _bool(row.get("valid"))
            valid_uv = _bool(row.get("valid_uv"))
            confidence = _float(row.get("confidence"), 0.0)
            if visible and valid and valid_uv and carrier_global:
                raw_stats[chunk]["visible_frame_by_carrier"][carrier_global].add(_int(row.get("frame_id"), -1))
                raw_stats[chunk]["confidence_sum"] += confidence
                raw_stats[chunk]["confidence_count"] += 1
            uses_gt = _bool(row.get("uses_gt_for_prediction"))
            if uses_gt:
                gt_violation_rows.append(
                    {
                        "scene_id": scene_id,
                        "chunk_id": chunk,
                        "frame_id": row.get("frame_id"),
                        "carrier_id": row.get("carrier_id"),
                        "notes": "carrier_observation_table row has uses_gt_for_prediction=true",
                    }
                )
            mask_id = _int(row.get("observed_mask_id"), 0)
            if not (visible and valid and valid_uv and _bool(row.get("mask_label_available")) and mask_id > 0):
                continue
            observations.append(
                Obs(
                    scene_id=scene_id,
                    chunk_id=chunk,
                    frame_id=_int(row.get("frame_id"), -1),
                    carrier_id=str(row.get("carrier_id") or ""),
                    carrier_global_id=carrier_global,
                    uv_x=_float(row.get("uv_x"), 0.0),
                    uv_y=_float(row.get("uv_y"), 0.0),
                    visible=visible,
                    confidence=confidence,
                    visibility_prob=_float(row.get("visibility_prob"), 0.0),
                    valid_uv=valid_uv,
                    mask_id=mask_id,
                    mask_area=_int(row.get("observed_mask_area"), 0),
                    support_density=_float(row.get("observed_mask_support_density"), 0.0),
                    is_boundary_region=_bool(row.get("is_boundary_region")),
                    uses_gt_for_prediction=uses_gt,
                )
            )
    return observations, total_carriers_by_chunk, raw_stats, gt_violation_rows


def _compute_jitter_px(observations: list[Obs], image_hw_by_scene: dict[str, tuple[int, int]]) -> dict[tuple[str, int, str, int], float]:
    grouped: dict[tuple[str, int, str], list[tuple[int, float, float]]] = defaultdict(list)
    for obs in observations:
        h, w = image_hw_by_scene[obs.scene_id]
        grouped[(obs.scene_id, obs.chunk_id, obs.carrier_global_id)].append((obs.frame_id, obs.uv_x * (w - 1), obs.uv_y * (h - 1)))
    jitter: dict[tuple[str, int, str, int], float] = {}
    for key, values in grouped.items():
        values.sort(key=lambda item: item[0])
        for index, (frame, x, y) in enumerate(values):
            lo = max(0, index - 1)
            hi = min(len(values), index + 2)
            med_x = float(np.median([item[1] for item in values[lo:hi]]))
            med_y = float(np.median([item[2] for item in values[lo:hi]]))
            jitter[(key[0], key[1], key[2], frame)] = math.hypot(x - med_x, y - med_y)
    return jitter


def _specificity_weight(mask_area_ratio: float) -> float:
    if mask_area_ratio <= 0:
        return 1.0
    # Smooth area suppression: huge masks keep only a weak same-level channel.
    return max(0.05, min(1.0, -math.log(mask_area_ratio + 1e-9) / math.log(1296.0 * 968.0)))


def _variant_membership(
    variant: str,
    confidence: float,
    visibility_prob: float,
    signed_distance_px: float,
    jitter_px: float,
    is_boundary_region: bool,
    *,
    sigma0: float,
    beta: float,
    jitter_lambda: float,
    fixed_sigma: float,
    boundary_penalty: float,
) -> tuple[float, float]:
    evidence = max(0.0, min(1.0, confidence)) * max(0.0, min(1.0, visibility_prob))
    if variant == "I0_observed_mask_id_hard":
        return 0.0, evidence
    if variant == "I1_uv_soft_sigma_fixed":
        sigma = fixed_sigma
    elif variant == "I2_uv_soft_confidence_sigma":
        sigma = max(1.0, sigma0 * math.exp(-beta * max(0.0, min(1.0, confidence))))
    else:
        sigma = max(1.0, sigma0 * math.exp(-beta * max(0.0, min(1.0, confidence))) * (1.0 + jitter_lambda * jitter_px))
    membership = evidence * _sigmoid(signed_distance_px / (sigma + 1e-6))
    if variant == "I4_uv_soft_with_boundary_band" and (is_boundary_region or signed_distance_px <= max(2.0, sigma)):
        membership *= boundary_penalty
    return sigma, membership


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.time()
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    max_chunk_exclusive = int(args.max_chunks)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    for variant in variants:
        if variant not in VARIANT_ORDER:
            raise ValueError(f"unknown incidence variant: {variant}")

    scene_roots = {scene: _rooted(path) for scene, path in DEFAULT_SCENE_ROOTS.items() if scene in set(args.scenes.split(","))}
    missing: list[dict[str, Any]] = []
    for scene, root in scene_roots.items():
        for rel_path in [
            "pipeline_summary.json",
            "chunk_universe/chunk_rows.csv",
            "observation_tables/mask_observation_table.csv",
            "observation_tables/carrier_observation_table.csv",
        ]:
            path = root / rel_path
            if not path.exists():
                missing.append({"scene_id": scene, "missing": rel_path, "path": _rel(path)})
    if not scene_roots:
        missing.append({"scene_id": "", "missing": "scene_roots", "path": args.scenes})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v75_phase1_soft_incidence",
            "schema": "stream4d_v75_phase1_soft_incidence_v1",
            "decision": "NO_GO_PHASE1_MISSING_INPUT",
            "missing_input_count": len(missing),
            "gate": {"pass": False, "all_inputs_present": False},
        }
        _write_json(output_root / "incidence_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary

    mask_dirs = {scene: _mask_dir_from_pipeline(root) for scene, root in scene_roots.items()}
    image_hw_by_scene: dict[str, tuple[int, int]] = {}
    for scene, mask_dir in mask_dirs.items():
        first = sorted(mask_dir.glob("*.png"))[0]
        image = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"failed to read first mask for {scene}: {first}")
        image_hw_by_scene[scene] = image.shape[:2]

    semantic_entropy = _load_semantic_entropy(set(scene_roots), max_chunk_exclusive)
    all_obs: list[Obs] = []
    total_carriers_by_scene_chunk: dict[tuple[str, int], set[str]] = defaultdict(set)
    raw_stats_by_scene_chunk: dict[tuple[str, int], dict[str, Any]] = {}
    diagnostic_gt_counts: dict[tuple[str, int], float] = {}
    chunk_defs_by_scene: dict[str, dict[int, dict[str, Any]]] = {}
    mask_info_by_scene: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    gt_violation_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for scene, root in scene_roots.items():
        chunk_defs = _load_chunk_defs(root, max_chunk_exclusive)
        chunk_defs_by_scene[scene] = chunk_defs
        mask_info, gt_counts = _load_mask_info(root, chunk_defs)
        mask_info_by_scene[scene] = mask_info
        for chunk, count in gt_counts.items():
            diagnostic_gt_counts[(scene, chunk)] = count
        obs, carrier_sets, raw_stats, violations = _read_observations(scene, root, max_chunk_exclusive)
        all_obs.extend(obs)
        gt_violation_rows.extend(violations)
        for chunk, carriers in carrier_sets.items():
            total_carriers_by_scene_chunk[(scene, chunk)].update(carriers)
        for chunk, stats in raw_stats.items():
            raw_stats_by_scene_chunk[(scene, chunk)] = stats
        for rel_path in [
            "pipeline_summary.json",
            "chunk_universe/chunk_rows.csv",
            "observation_tables/mask_observation_table.csv",
            "observation_tables/carrier_observation_table.csv",
        ]:
            path = root / rel_path
            source_rows.append(
                {
                    **ROW_DEFAULTS,
                    "scene_id": scene,
                    "phase": "v75_phase1_soft_incidence",
                    "metric": "input_presence",
                    "source_artifact": _rel(path),
                    "value": True,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    all_obs.sort(key=lambda obs: (obs.scene_id, obs.frame_id, obs.chunk_id, obs.carrier_global_id, obs.mask_id))
    jitter_by_obs = _compute_jitter_px(all_obs, image_hw_by_scene)

    incidence_path = output_root / "incidence_rows.csv"
    chunk_acc: dict[tuple[str, int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "nnz": 0,
            "membership_sum": 0.0,
            "membership_carriers": set(),
            "boundary_uncertain": 0,
            "raw_mask_mass": defaultdict(float),
            "weighted_mask_mass": defaultdict(float),
            "large_hyperedges": set(),
            "jitter_sum": 0.0,
            "jitter_count": 0,
        }
    )
    frame_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    cache_order: list[tuple[str, int]] = []

    fields = [
        "scene_id",
        "chunk_id",
        "frame_id",
        "carrier_id",
        "carrier_global_id",
        "mask_observation_id",
        "mask_id",
        "uv_x",
        "uv_y",
        "visible",
        "confidence",
        "valid_uv",
        "sigma",
        "signed_distance_to_mask",
        "soft_membership",
        "membership_variant",
        "source_observed_mask_id",
        "semantic_entropy_of_mask",
        "mask_area_ratio",
        "mask_carrier_mass",
        "carrier_jitter_px",
        "support_density",
        "uses_gt_for_prediction",
    ]
    with incidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for obs in all_obs:
            cache_key = (obs.scene_id, obs.frame_id)
            if cache_key not in frame_cache:
                label, dist = _read_label_and_boundary(mask_dirs[obs.scene_id], obs.frame_id)
                frame_cache[cache_key] = (label, dist)
                cache_order.append(cache_key)
                while len(cache_order) > int(args.frame_cache_size):
                    old = cache_order.pop(0)
                    frame_cache.pop(old, None)
            label, dist = frame_cache[cache_key]
            h, w = label.shape[:2]
            x = min(max(int(round(obs.uv_x * (w - 1))), 0), w - 1)
            y = min(max(int(round(obs.uv_y * (h - 1))), 0), h - 1)
            label_at_uv = int(label[y, x])
            signed_distance = float(dist[y, x])
            if label_at_uv != obs.mask_id:
                signed_distance = -signed_distance
            image_area = float(h * w)
            mask_area = obs.mask_area or int((label == obs.mask_id).sum())
            mask_area_ratio = float(mask_area) / image_area if image_area > 0 else 0.0
            specificity = _specificity_weight(mask_area_ratio)
            jitter_px = jitter_by_obs.get((obs.scene_id, obs.chunk_id, obs.carrier_global_id, obs.frame_id), 0.0)
            sem_entropy = semantic_entropy.get((obs.scene_id, obs.frame_id, obs.mask_id))
            mask_obs = mask_info_by_scene[obs.scene_id].get((obs.frame_id, obs.mask_id), {})
            mask_observation_id = mask_obs.get("mask_observation_id") or f"{obs.scene_id}:{obs.frame_id}:{obs.mask_id}"
            for variant in variants:
                sigma, membership = _variant_membership(
                    variant,
                    obs.confidence,
                    obs.visibility_prob,
                    signed_distance,
                    jitter_px,
                    obs.is_boundary_region,
                    sigma0=float(args.sigma0),
                    beta=float(args.beta),
                    jitter_lambda=float(args.jitter_lambda),
                    fixed_sigma=float(args.fixed_sigma),
                    boundary_penalty=float(args.boundary_penalty),
                )
                if membership <= float(args.min_membership):
                    continue
                key = (obs.scene_id, obs.chunk_id, variant)
                acc = chunk_acc[key]
                acc["nnz"] += 1
                acc["membership_sum"] += membership
                acc["membership_carriers"].add(obs.carrier_global_id)
                acc["jitter_sum"] += jitter_px
                acc["jitter_count"] += 1
                if obs.is_boundary_region or abs(signed_distance) <= max(2.0, sigma):
                    acc["boundary_uncertain"] += 1
                edge_key = f"{obs.frame_id}:{obs.mask_id}"
                acc["raw_mask_mass"][edge_key] += membership
                acc["weighted_mask_mass"][edge_key] += membership * specificity
                if mask_area_ratio >= float(args.large_mask_area_ratio):
                    acc["large_hyperedges"].add(edge_key)
                writer.writerow(
                    {
                        "scene_id": obs.scene_id,
                        "chunk_id": obs.chunk_id,
                        "frame_id": obs.frame_id,
                        "carrier_id": obs.carrier_id,
                        "carrier_global_id": obs.carrier_global_id,
                        "mask_observation_id": mask_observation_id,
                        "mask_id": obs.mask_id,
                        "uv_x": obs.uv_x,
                        "uv_y": obs.uv_y,
                        "visible": obs.visible,
                        "confidence": obs.confidence,
                        "valid_uv": obs.valid_uv,
                        "sigma": sigma,
                        "signed_distance_to_mask": signed_distance,
                        "soft_membership": membership,
                        "membership_variant": variant,
                        "source_observed_mask_id": obs.mask_id,
                        "semantic_entropy_of_mask": sem_entropy,
                        "mask_area_ratio": mask_area_ratio,
                        "mask_carrier_mass": membership,
                        "carrier_jitter_px": jitter_px,
                        "support_density": obs.support_density,
                        "uses_gt_for_prediction": False,
                    }
                )

    chunk_rows: list[dict[str, Any]] = []
    variant_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    chunks_seen = sorted(total_carriers_by_scene_chunk)
    for scene, chunk in chunks_seen:
        total_carriers = len(total_carriers_by_scene_chunk[(scene, chunk)])
        raw_stats = raw_stats_by_scene_chunk.get((scene, chunk), {})
        visible_counts = [len(frames) for frames in (raw_stats.get("visible_frame_by_carrier") or {}).values()]
        visible_frame_count_mean = float(sum(visible_counts) / len(visible_counts)) if visible_counts else 0.0
        confidence_count = int(raw_stats.get("confidence_count") or 0)
        confidence_mean = float(raw_stats.get("confidence_sum") or 0.0) / confidence_count if confidence_count else 0.0
        for variant in variants:
            acc = chunk_acc.get((scene, chunk, variant))
            if acc is None:
                acc = {
                    "nnz": 0,
                    "membership_sum": 0.0,
                    "membership_carriers": set(),
                    "boundary_uncertain": 0,
                    "raw_mask_mass": {},
                    "weighted_mask_mass": {},
                    "large_hyperedges": set(),
                    "jitter_sum": 0.0,
                    "jitter_count": 0,
                }
            raw_total = float(sum(acc["raw_mask_mass"].values()))
            weighted_total = float(sum(acc["weighted_mask_mass"].values()))
            largest_raw = max(acc["raw_mask_mass"].values()) / raw_total if raw_total > 0 else 0.0
            largest_weighted = max(acc["weighted_mask_mass"].values()) / weighted_total if weighted_total > 0 else 0.0
            nnz = int(acc["nnz"])
            row = {
                **ROW_DEFAULTS,
                "scene_id": scene,
                "chunk_id": chunk,
                "variant": variant,
                "carrier_count": total_carriers,
                "diagnostic_GT_count": diagnostic_gt_counts.get((scene, chunk), 0.0),
                "mask_observation_count": len(acc["raw_mask_mass"]),
                "soft_incidence_nnz": nnz,
                "carrier_with_mask_membership_rate": (len(acc["membership_carriers"]) / total_carriers) if total_carriers else 0.0,
                "mean_membership_per_carrier": (float(acc["membership_sum"]) / total_carriers) if total_carriers else 0.0,
                "visible_frame_count_mean": visible_frame_count_mean,
                "carrier_confidence_mean": confidence_mean,
                "carrier_jitter_mean": (float(acc["jitter_sum"]) / int(acc["jitter_count"])) if int(acc["jitter_count"]) else 0.0,
                "boundary_uncertain_membership_rate": (int(acc["boundary_uncertain"]) / nnz) if nnz else 0.0,
                "largest_hyperedge_mass_ratio_raw": largest_raw,
                "largest_hyperedge_mass_ratio_after_specificity": largest_weighted,
                "large_hyperedge_count": len(acc["large_hyperedges"]),
                "runtime_sec": None,
                "peak_memory_gb": None,
                "uses_gt_for_prediction": False,
            }
            chunk_rows.append(row)
            for metric in [
                "carrier_count",
                "diagnostic_GT_count",
                "soft_incidence_nnz",
                "carrier_with_mask_membership_rate",
                "visible_frame_count_mean",
                "boundary_uncertain_membership_rate",
                "largest_hyperedge_mass_ratio_after_specificity",
            ]:
                variant_values[variant][metric].append(float(row[metric]))

    runtime_sec = time.time() - start_time
    peak_memory_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    for row in chunk_rows:
        row["runtime_sec"] = runtime_sec / max(1, len(chunks_seen))
        row["peak_memory_gb"] = peak_memory_gb

    variant_rows: list[dict[str, Any]] = []
    for variant in variants:
        vals = variant_values[variant]
        carrier_mean = _mean(vals["carrier_count"])
        gt_mean = _mean(vals["diagnostic_GT_count"])
        summary_row = {
            **ROW_DEFAULTS,
            "scene_id": "aggregate",
            "chunk_id": "aggregate",
            "phase": "v75_phase1_soft_incidence",
            "variant": variant,
            "carrier_count_per_chunk_mean": carrier_mean,
            "diagnostic_GT_count_per_chunk_mean": gt_mean,
            "carrier_vs_diagnostic_GT_ratio": (carrier_mean / gt_mean) if gt_mean > 0 else None,
            "soft_incidence_nnz_per_chunk_mean": _mean(vals["soft_incidence_nnz"]),
            "carrier_with_mask_membership_rate_mean": _mean(vals["carrier_with_mask_membership_rate"]),
            "visible_frame_count_mean": _mean(vals["visible_frame_count_mean"]),
            "boundary_uncertain_membership_rate_mean": _mean(vals["boundary_uncertain_membership_rate"]),
            "largest_hyperedge_mass_ratio_after_specificity_mean": _mean(vals["largest_hyperedge_mass_ratio_after_specificity"]),
            "runtime_per_chunk_sec": runtime_sec / max(1, len(chunks_seen)),
            "peak_memory_gb": peak_memory_gb,
        }
        summary_row["gate_pass"] = _phase1_gate_pass(summary_row)
        variant_rows.append(summary_row)

    main_variant = args.main_variant
    main_row = next((row for row in variant_rows if row["variant"] == main_variant), variant_rows[-1])
    best_pass = next((row for row in variant_rows if row.get("gate_pass")), None)
    decision = "PASS_V75_PHASE1_SOFT_INCIDENCE" if main_row.get("gate_pass") else "NO_GO_PHASE1_SOFT_INCIDENCE_GATE"
    if not main_row.get("gate_pass") and best_pass is not None:
        decision = "PARTIAL_PHASE1_REPAIR_VARIANT_PASS_MAIN_FAIL"

    summary = {
        "phase": "v75_phase1_soft_incidence",
        "schema": "stream4d_v75_phase1_soft_incidence_v1",
        "decision": decision,
        "main_variant": main_variant,
        "best_gate_variant": best_pass.get("variant") if best_pass else None,
        "row_variants": variants,
        "scenes": sorted(scene_roots),
        "chunk_count": len(chunks_seen),
        "incidence_row_count": sum(1 for _ in incidence_path.open(encoding="utf-8")) - 1,
        "method_prediction_uses_gt_anywhere": bool(gt_violation_rows),
        "gt_boundary_violation_count": len(gt_violation_rows),
        "gate": {
            "pass": bool(main_row.get("gate_pass")),
            "main_variant": main_variant,
            "carrier_count_per_chunk_mean_ge_5x_diagnostic_GT_count_per_chunk_mean": (
                (main_row["carrier_vs_diagnostic_GT_ratio"] or 0) >= 5.0
            ),
            "carrier_with_mask_membership_rate_ge_0p70": main_row["carrier_with_mask_membership_rate_mean"] >= 0.70,
            "visible_frame_count_mean_ge_3": main_row["visible_frame_count_mean"] >= 3.0,
            "soft_incidence_nnz_per_chunk_mean_gt_0": main_row["soft_incidence_nnz_per_chunk_mean"] > 0,
            "boundary_uncertain_membership_rate_le_0p45": main_row["boundary_uncertain_membership_rate_mean"] <= 0.45,
            "largest_hyperedge_mass_ratio_after_specificity_le_0p35": main_row[
                "largest_hyperedge_mass_ratio_after_specificity_mean"
            ]
            <= 0.35,
            "runtime_per_chunk_le_30s": main_row["runtime_per_chunk_sec"] <= 30.0,
            "peak_memory_gb_le_16": main_row["peak_memory_gb"] <= 16.0,
            "uses_gt_for_prediction_false": not gt_violation_rows,
        },
        "main_variant_metrics": main_row,
        "variant_metrics": variant_rows,
        "inputs": {scene: _rel(root) for scene, root in scene_roots.items()},
        "mask_dirs": {scene: _rel(mask_dir) for scene, mask_dir in mask_dirs.items()},
        "notes": [
            "Signed distance is computed from real CropFormer label PNGs as distance to the nearest label boundary; no GT/mesh/RGB-D geometry is used for prediction.",
            "diagnostic_GT_count is used only for the Phase1 density gate denominator and is not used in incidence membership.",
            "I4 is the plan repair variant with boundary-band penalty; local2history remains blocked regardless of Phase1.",
        ],
    }

    _write_csv(output_root / "incidence_chunk_rows.csv", chunk_rows)
    _write_csv(output_root / "incidence_variant_summary_rows.csv", variant_rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_csv(output_root / "source_rows.csv", source_rows)
    _write_csv(output_root / "gt_boundary_rows.csv", gt_violation_rows)
    _write_json(output_root / "incidence_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _write_sha_rows(output_root, scene_roots, incidence_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _phase1_gate_pass(row: dict[str, Any]) -> bool:
    ratio = row.get("carrier_vs_diagnostic_GT_ratio") or 0.0
    return bool(
        ratio >= 5.0
        and row.get("carrier_with_mask_membership_rate_mean", 0.0) >= 0.70
        and row.get("visible_frame_count_mean", 0.0) >= 3.0
        and row.get("soft_incidence_nnz_per_chunk_mean", 0.0) > 0.0
        and row.get("boundary_uncertain_membership_rate_mean", 1.0) <= 0.45
        and row.get("largest_hyperedge_mass_ratio_after_specificity_mean", 1.0) <= 0.35
        and row.get("runtime_per_chunk_sec", 999.0) <= 30.0
        and row.get("peak_memory_gb", 999.0) <= 16.0
    )


def _write_sha_rows(output_root: Path, scene_roots: dict[str, Path], incidence_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for scene, root in scene_roots.items():
        for rel_path in [
            "pipeline_summary.json",
            "chunk_universe/chunk_rows.csv",
            "observation_tables/mask_observation_table.csv",
            "observation_tables/carrier_observation_table.csv",
        ]:
            path = root / rel_path
            if path.exists():
                rows.append(
                    {
                        **ROW_DEFAULTS,
                        "scene_id": scene,
                        "chunk_id": "aggregate",
                        "phase": "v75_phase1_soft_incidence",
                        "name": f"input:{scene}:{rel_path}",
                        "source_artifact": _rel(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            rows.append(
                {
                    **ROW_DEFAULTS,
                    "scene_id": "aggregate",
                    "chunk_id": "aggregate",
                    "phase": "v75_phase1_soft_incidence",
                    "name": f"output:{path.name}",
                    "source_artifact": _rel(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_csv(output_root / "sha256_rows.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v75 Phase1 D4RT carrier-mask soft incidence builder.")
    parser.add_argument("--output-root", default="outputs/audit/v75_phase1_soft_incidence")
    parser.add_argument("--scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--variants", default=",".join(VARIANT_ORDER))
    parser.add_argument("--main-variant", default="I3_uv_soft_confidence_jitter_sigma")
    parser.add_argument("--sigma0", type=float, default=12.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--jitter-lambda", type=float, default=0.02)
    parser.add_argument("--fixed-sigma", type=float, default=8.0)
    parser.add_argument("--boundary-penalty", type=float, default=0.5)
    parser.add_argument("--min-membership", type=float, default=0.0)
    parser.add_argument("--large-mask-area-ratio", type=float, default=0.25)
    parser.add_argument("--frame-cache-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
