from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def frame_ids_for_carrier_file(carrier_path: Path, num_frames: int) -> list[int]:
    summary_path = carrier_path.with_name(carrier_path.name.replace("carriers_", "").replace(".npz", "_summary.json"))
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            frame_ids = [int(value) for value in payload.get("frame_ids", [])]
            if len(frame_ids) == num_frames:
                return frame_ids
        except Exception:
            pass
    return list(range(num_frames))


def load_window_summary(carrier_path: Path) -> dict[str, Any]:
    summary_path = carrier_path.with_name(carrier_path.name.replace("carriers_", "").replace(".npz", "_summary.json"))
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_mask(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64, copy=False)


def load_rgb(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sample_mask(mask: np.ndarray, uv_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    x = np.rint(uv_norm[:, 0] * float(max(width - 1, 1))).astype(np.int64)
    y = np.rint(uv_norm[:, 1] * float(max(height - 1, 1))).astype(np.int64)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    out = np.zeros((uv_norm.shape[0],), dtype=np.int64)
    if np.any(in_bounds):
        out[in_bounds] = mask[y[in_bounds], x[in_bounds]]
    return out, in_bounds, np.stack([x, y], axis=1)


def sample_rgb(rgb: np.ndarray, xy: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    x = np.clip(xy[:, 0].astype(np.int64), 0, max(width - 1, 0))
    y = np.clip(xy[:, 1].astype(np.int64), 0, max(height - 1, 0))
    return rgb[y, x].astype(np.float32) / 255.0


def visible_ok(
    uv_pred: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    *,
    min_visibility: float,
    min_confidence: float,
) -> np.ndarray:
    return (
        valid
        & np.isfinite(uv_pred).all(axis=1)
        & (uv_pred[:, 0] >= 0.0)
        & (uv_pred[:, 0] <= 1.0)
        & (uv_pred[:, 1] >= 0.0)
        & (uv_pred[:, 1] <= 1.0)
        & (visibility >= float(min_visibility))
        & (confidence >= float(min_confidence))
    )


def boundary_distance_for_samples(mask: np.ndarray, sampled_ids: np.ndarray, xy: np.ndarray) -> np.ndarray:
    distances = np.zeros((sampled_ids.shape[0],), dtype=np.float32)
    positive_ids = sorted(int(v) for v in np.unique(sampled_ids) if int(v) > 0)
    if not positive_ids:
        return distances
    height, width = mask.shape[:2]
    x = np.clip(xy[:, 0].astype(np.int64), 0, max(width - 1, 0))
    y = np.clip(xy[:, 1].astype(np.int64), 0, max(height - 1, 0))
    for mask_id in positive_ids:
        member = sampled_ids == int(mask_id)
        if not np.any(member):
            continue
        binary = (mask == int(mask_id)).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            continue
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        distances[member] = dist[y[member], x[member]].astype(np.float32)
    return distances


def quantiles(values: list[float] | np.ndarray, qs: tuple[int, ...] = (10, 50, 90)) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {f"q{q}": 0.0 for q in qs}
    return {f"q{q}": float(np.percentile(arr, q)) for q in qs}


@dataclass
class MeasurementBank:
    scene: str
    frame_ids: np.ndarray
    carrier_id: np.ndarray
    uv_pred: np.ndarray
    valid: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    src_frame_global: np.ndarray
    src_mask_id: np.ndarray
    src_xy: np.ndarray
    src_rgb: np.ndarray
    target_mask_id: np.ndarray
    target_in_bounds: np.ndarray
    visible_ok: np.ndarray
    boundary_distance: np.ndarray
    source_boundary_distance: np.ndarray
    mask_frame_available: np.ndarray
    positive_observation: np.ndarray
    negative_observation: np.ndarray
    source_positive_propagated: np.ndarray
    meta: dict[str, Any]

    @property
    def num_surfels(self) -> int:
        return int(self.carrier_id.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene=np.asarray(self.scene),
            frame_ids=self.frame_ids,
            carrier_id=self.carrier_id,
            uv_pred=self.uv_pred,
            valid=self.valid,
            visibility=self.visibility,
            confidence=self.confidence,
            src_frame_global=self.src_frame_global,
            src_mask_id=self.src_mask_id,
            src_xy=self.src_xy,
            src_rgb=self.src_rgb,
            target_mask_id=self.target_mask_id,
            target_in_bounds=self.target_in_bounds,
            visible_ok=self.visible_ok,
            boundary_distance=self.boundary_distance,
            source_boundary_distance=self.source_boundary_distance,
            mask_frame_available=self.mask_frame_available,
            positive_observation=self.positive_observation,
            negative_observation=self.negative_observation,
            source_positive_propagated=self.source_positive_propagated,
            meta_json=np.asarray(json.dumps(json_safe(self.meta), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> "MeasurementBank":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"].item()))
            return cls(
                scene=str(data["scene"].item()),
                frame_ids=np.asarray(data["frame_ids"], dtype=np.int64),
                carrier_id=np.asarray(data["carrier_id"], dtype=np.int64),
                uv_pred=np.asarray(data["uv_pred"], dtype=np.float32),
                valid=np.asarray(data["valid"], dtype=bool),
                visibility=np.asarray(data["visibility"], dtype=np.float32),
                confidence=np.asarray(data["confidence"], dtype=np.float32),
                src_frame_global=np.asarray(data["src_frame_global"], dtype=np.int64),
                src_mask_id=np.asarray(data["src_mask_id"], dtype=np.int64),
                src_xy=np.asarray(data["src_xy"], dtype=np.int64),
                src_rgb=np.asarray(data["src_rgb"], dtype=np.float32),
                target_mask_id=np.asarray(data["target_mask_id"], dtype=np.int64),
                target_in_bounds=np.asarray(data["target_in_bounds"], dtype=bool),
                visible_ok=np.asarray(data["visible_ok"], dtype=bool),
                boundary_distance=np.asarray(data["boundary_distance"], dtype=np.float32),
                source_boundary_distance=np.asarray(data["source_boundary_distance"], dtype=np.float32),
                mask_frame_available=np.asarray(data["mask_frame_available"], dtype=bool),
                positive_observation=np.asarray(data["positive_observation"], dtype=bool),
                negative_observation=np.asarray(data["negative_observation"], dtype=bool),
                source_positive_propagated=np.asarray(data["source_positive_propagated"], dtype=bool),
                meta=meta,
            )


def _concat_carrier_windows(carrier_paths: list[Path]) -> tuple[dict[str, np.ndarray], list[int], dict[str, Any]]:
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    frame_ids_ref: list[int] | None = None
    summaries: list[dict[str, Any]] = []
    for carrier_path in carrier_paths:
        with np.load(carrier_path) as data:
            frame_ids = frame_ids_for_carrier_file(carrier_path, np.asarray(data["uv_pred"]).shape[0])
            if frame_ids_ref is None:
                frame_ids_ref = frame_ids
            elif frame_ids != frame_ids_ref:
                raise ValueError(
                    "v12 measurement bank currently expects carrier windows with identical frame ids; "
                    f"{carrier_path} has {frame_ids}, expected {frame_ids_ref}"
                )
            for key in (
                "carrier_id",
                "uv_pred",
                "visibility_prob",
                "confidence_prob",
                "valid",
                "src_frame_global",
                "src_xy",
                "src_mask_id",
            ):
                arrays[key].append(np.asarray(data[key]))
        summaries.append(load_window_summary(carrier_path))
    if frame_ids_ref is None:
        raise ValueError("No carrier windows were provided")
    merged: dict[str, np.ndarray] = {}
    for key, parts in arrays.items():
        if key in {"uv_pred", "visibility_prob", "confidence_prob", "valid"}:
            merged[key] = np.concatenate(parts, axis=1)
        else:
            merged[key] = np.concatenate(parts, axis=0)
    numeric = {}
    for key in ("uv_in01_rate", "self_uv_error_p90", "cycle_uv_error_p90", "track_length_visible_mean"):
        vals = [float(item[key]) for item in summaries if item.get(key) is not None]
        if vals:
            numeric[f"{key}_mean"] = float(np.mean(vals))
    return merged, frame_ids_ref, {"carrier_windows": [str(p) for p in carrier_paths], **numeric}


def build_scene_measurement_bank(
    *,
    debug_root: Path,
    scannet_root: Path,
    scene: str,
    backbone: str = "Cropformer",
    min_visibility: float = 0.5,
    min_confidence: float = 0.5,
    boundary_safe_px: float = 3.0,
) -> tuple[MeasurementBank, dict[str, Any]]:
    scene_debug = debug_root / scene
    carrier_paths = sorted(scene_debug.glob("carriers_window*.npz"))
    if not carrier_paths:
        raise FileNotFoundError(f"No carrier windows under {scene_debug}")
    merged, frame_ids, meta = _concat_carrier_windows(carrier_paths)
    uv_pred = np.asarray(merged["uv_pred"], dtype=np.float32)
    valid = np.asarray(merged["valid"], dtype=bool)
    visibility = np.asarray(merged["visibility_prob"], dtype=np.float32)
    confidence = np.asarray(merged["confidence_prob"], dtype=np.float32)
    src_frame_global = np.asarray(merged["src_frame_global"], dtype=np.int64)
    src_mask_id = np.asarray(merged["src_mask_id"], dtype=np.int64)
    src_xy = np.asarray(merged["src_xy"], dtype=np.int64)
    carrier_id = np.asarray(merged["carrier_id"], dtype=np.int64)
    num_frames, num_surfels = int(uv_pred.shape[0]), int(uv_pred.shape[1])
    scene_root = scannet_root / scene
    mask_dir = scene_root / f"output_{backbone}" / "mask"
    color_dir = scene_root / "color"

    target_mask_id = np.zeros((num_frames, num_surfels), dtype=np.int64)
    target_in_bounds = np.zeros((num_frames, num_surfels), dtype=bool)
    boundary_distance = np.zeros((num_frames, num_surfels), dtype=np.float32)
    mask_frame_available = np.zeros((num_frames,), dtype=bool)
    visible = np.zeros((num_frames, num_surfels), dtype=bool)
    src_rgb = np.zeros((num_surfels, 3), dtype=np.float32)
    source_boundary_distance = np.zeros((num_surfels,), dtype=np.float32)

    for local_idx, frame_id in enumerate(frame_ids):
        visible[local_idx] = visible_ok(
            uv_pred[local_idx],
            valid[local_idx],
            visibility[local_idx],
            confidence[local_idx],
            min_visibility=min_visibility,
            min_confidence=min_confidence,
        )
        mask = load_mask(mask_dir / f"{int(frame_id)}.png")
        if mask is None:
            continue
        mask_frame_available[local_idx] = True
        sampled, in_bounds, xy = sample_mask(mask, uv_pred[local_idx])
        target_mask_id[local_idx] = sampled
        target_in_bounds[local_idx] = in_bounds
        boundary_distance[local_idx] = boundary_distance_for_samples(mask, sampled, xy)

    rgb_cache: dict[int, np.ndarray] = {}
    mask_cache: dict[int, np.ndarray] = {}
    for frame_id in sorted(set(int(v) for v in src_frame_global.tolist())):
        rgb = load_rgb(color_dir / f"{frame_id}.jpg")
        if rgb is not None:
            rgb_cache[frame_id] = rgb
        mask = load_mask(mask_dir / f"{frame_id}.png")
        if mask is not None:
            mask_cache[frame_id] = mask

    for frame_id, indices in _indices_by_value(src_frame_global).items():
        if frame_id in rgb_cache:
            src_rgb[indices] = sample_rgb(rgb_cache[frame_id], src_xy[indices])
        if frame_id in mask_cache:
            mask = mask_cache[frame_id]
            sampled = np.zeros((indices.shape[0],), dtype=np.int64)
            h, w = mask.shape[:2]
            x = np.clip(src_xy[indices, 0], 0, max(w - 1, 0))
            y = np.clip(src_xy[indices, 1], 0, max(h - 1, 0))
            sampled[:] = mask[y, x]
            source_boundary_distance[indices] = boundary_distance_for_samples(mask, sampled, src_xy[indices])

    positive_observation = visible & mask_frame_available[:, None] & (target_mask_id > 0)
    source_positive = src_mask_id > 0
    source_positive_propagated = visible & source_positive[None, :]
    negative_observation = visible & mask_frame_available[:, None] & source_positive[None, :] & (target_mask_id <= 0)

    bank = MeasurementBank(
        scene=scene,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        carrier_id=carrier_id,
        uv_pred=uv_pred,
        valid=valid,
        visibility=visibility,
        confidence=confidence,
        src_frame_global=src_frame_global,
        src_mask_id=src_mask_id,
        src_xy=src_xy,
        src_rgb=src_rgb,
        target_mask_id=target_mask_id,
        target_in_bounds=target_in_bounds,
        visible_ok=visible,
        boundary_distance=boundary_distance,
        source_boundary_distance=source_boundary_distance,
        mask_frame_available=mask_frame_available,
        positive_observation=positive_observation,
        negative_observation=negative_observation,
        source_positive_propagated=source_positive_propagated,
        meta={
            "debug_root": str(debug_root),
            "scannet_root": str(scannet_root),
            "backbone": backbone,
            "min_visibility": float(min_visibility),
            "min_confidence": float(min_confidence),
            "boundary_safe_px": float(boundary_safe_px),
            **meta,
        },
    )
    return bank, summarize_bank(bank, boundary_safe_px=boundary_safe_px)


def _indices_by_value(values: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, list[int]] = defaultdict(list)
    for idx, value in enumerate(values.tolist()):
        out[int(value)].append(int(idx))
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in out.items()}


def summarize_bank(bank: MeasurementBank, *, boundary_safe_px: float = 3.0) -> dict[str, Any]:
    visible_counts = bank.visible_ok.sum(axis=0).astype(np.float64)
    positive_counts = bank.positive_observation.sum(axis=0).astype(np.float64)
    propagated_counts = bank.source_positive_propagated.sum(axis=0).astype(np.float64)
    negative_counts = bank.negative_observation.sum(axis=0).astype(np.float64)
    source_positive = bank.src_mask_id > 0
    target_counts: Counter[tuple[int, int]] = Counter()
    for frame_idx, frame_id in enumerate(bank.frame_ids.tolist()):
        ids, counts = np.unique(bank.target_mask_id[frame_idx][bank.positive_observation[frame_idx]], return_counts=True)
        for mask_id, count in zip(ids.tolist(), counts.tolist()):
            if int(mask_id) > 0:
                target_counts[(int(frame_id), int(mask_id))] += int(count)
    mask_counts = np.asarray(list(target_counts.values()), dtype=np.float64)
    boundary_safe = bank.positive_observation & (bank.boundary_distance >= float(boundary_safe_px))
    ambiguous_or_contradicted = source_positive & (negative_counts > 0)
    uv_in01 = (
        bank.valid
        & np.isfinite(bank.uv_pred).all(axis=2)
        & (bank.uv_pred[:, :, 0] >= 0.0)
        & (bank.uv_pred[:, :, 0] <= 1.0)
        & (bank.uv_pred[:, :, 1] >= 0.0)
        & (bank.uv_pred[:, :, 1] <= 1.0)
    )
    summary = {
        "scene": bank.scene,
        "status": "ok",
        "num_frames": int(bank.frame_ids.shape[0]),
        "num_surfels": int(bank.num_surfels),
        "num_valid_tracks": int(np.count_nonzero(visible_counts > 0)),
        "track_length_visible_mean": float(np.mean(visible_counts)) if visible_counts.size else 0.0,
        "uv_in01_rate": float(np.count_nonzero(uv_in01) / max(int(uv_in01.size), 1)),
        "visible_ok_rate": float(np.count_nonzero(bank.visible_ok) / max(int(bank.visible_ok.size), 1)),
        "self_uv_error_p90": bank.meta.get("self_uv_error_p90_mean"),
        "cycle_uv_error_p90": bank.meta.get("cycle_uv_error_p90_mean"),
        "surfel_2d_coverage_per_frame": float(np.mean(bank.visible_ok.sum(axis=1) / max(bank.num_surfels, 1))),
        "num_mask_frames_available": int(np.count_nonzero(bank.mask_frame_available)),
        "num_mask_frames_missing": int(bank.mask_frame_available.shape[0] - np.count_nonzero(bank.mask_frame_available)),
        "surfel_positive_observation_rate": float(np.count_nonzero(positive_counts > 0) / max(bank.num_surfels, 1)),
        "mean_positive_observations_per_surfel": float(np.mean(propagated_counts)) if propagated_counts.size else 0.0,
        "median_positive_observations_per_surfel": float(np.median(propagated_counts)) if propagated_counts.size else 0.0,
        "surfel_negative_observation_rate": float(np.count_nonzero(negative_counts > 0) / max(bank.num_surfels, 1)),
        "mask_to_surfel_count_mean": float(np.mean(mask_counts)) if mask_counts.size else 0.0,
        "mask_to_surfel_count_p10": float(np.percentile(mask_counts, 10)) if mask_counts.size else 0.0,
        "mask_to_surfel_count_p50": float(np.percentile(mask_counts, 50)) if mask_counts.size else 0.0,
        "mask_to_surfel_count_p90": float(np.percentile(mask_counts, 90)) if mask_counts.size else 0.0,
        "regionlet_to_surfel_count_mean": None,
        "regionlet_note": "No separate v12 regionlet measurement was materialized in this bank; mask measurements are recorded directly.",
        "boundary_safe_surfel_ratio": float(np.count_nonzero(boundary_safe.any(axis=0)) / max(bank.num_surfels, 1)),
        "ambiguous_surfel_ratio": float(np.count_nonzero(ambiguous_or_contradicted) / max(np.count_nonzero(source_positive), 1)),
        "unobserved_surfel_ratio": float(np.count_nonzero(positive_counts == 0) / max(bank.num_surfels, 1)),
        "visible_samples": int(np.count_nonzero(bank.visible_ok)),
        "target_positive_samples": int(np.count_nonzero(bank.positive_observation)),
        "source_propagated_positive_samples": int(np.count_nonzero(bank.source_positive_propagated)),
        "negative_samples": int(np.count_nonzero(bank.negative_observation)),
        "source_positive_surfel_ratio": float(np.count_nonzero(source_positive) / max(bank.num_surfels, 1)),
    }
    return summary


def write_summary_bundle(prefix: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.generic)) and value is not None and not isinstance(value, bool)
        }
    )
    aggregate = {
        "diagnostic_only": True,
        "uses_gt": False,
        "is_method_result": False,
        "num_scenes": int(len(rows)),
        "num_ok_scenes": int(sum(1 for row in rows if row.get("status") == "ok")),
        "numeric_mean": {
            key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
            for key in numeric_keys
            if any(row.get(key) is not None for row in rows)
        },
    }
    payload = {"aggregate": json_safe(aggregate), "scenes": json_safe(rows)}
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        import csv

        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Stream4D v12 Measurement Bank Diagnostic",
        "",
        "This diagnostic does not read GT labels and does not report AP.",
        "",
        "## Aggregate",
        "",
    ]
    for key in (
        "num_surfels",
        "uv_in01_rate",
        "track_length_visible_mean",
        "self_uv_error_p90",
        "cycle_uv_error_p90",
        "mean_positive_observations_per_surfel",
        "mask_to_surfel_count_mean",
        "boundary_safe_surfel_ratio",
        "ambiguous_surfel_ratio",
        "unobserved_surfel_ratio",
    ):
        lines.append(f"- {key}_mean: `{aggregate['numeric_mean'].get(key)}`")
    lines.extend(
        [
            "",
            "## Scenes",
            "",
            "| scene | surfels | uv in01 | visible len | self p90 | cycle p90 | obs/surfel | mask->surfel | boundary safe | ambiguous | unobserved |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scene")),
                    str(row.get("num_surfels")),
                    f"{float(row.get('uv_in01_rate') or 0.0):.6f}",
                    f"{float(row.get('track_length_visible_mean') or 0.0):.4f}",
                    str(row.get("self_uv_error_p90")),
                    str(row.get("cycle_uv_error_p90")),
                    f"{float(row.get('mean_positive_observations_per_surfel') or 0.0):.4f}",
                    f"{float(row.get('mask_to_surfel_count_mean') or 0.0):.4f}",
                    f"{float(row.get('boundary_safe_surfel_ratio') or 0.0):.4f}",
                    f"{float(row.get('ambiguous_surfel_ratio') or 0.0):.4f}",
                    f"{float(row.get('unobserved_surfel_ratio') or 0.0):.4f}",
                ]
            )
            + " |"
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate
