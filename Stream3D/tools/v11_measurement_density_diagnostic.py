from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _frame_ids_for_carrier_file(carrier_path: Path, num_frames: int) -> list[int]:
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


def _load_window_summary(carrier_path: Path) -> dict[str, Any]:
    summary_path = carrier_path.with_name(carrier_path.name.replace("carriers_", "").replace(".npz", "_summary.json"))
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_mask(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64, copy=False)


def _load_rgb_hw(scene_root: Path, frame_id: int) -> tuple[int, int]:
    path = scene_root / "color" / f"{int(frame_id)}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return int(image.shape[0]), int(image.shape[1])


def _sample_mask(mask: np.ndarray, uv_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    x = np.rint(uv_norm[:, 0] * float(max(width - 1, 1))).astype(np.int64)
    y = np.rint(uv_norm[:, 1] * float(max(height - 1, 1))).astype(np.int64)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    out = np.zeros((uv_norm.shape[0],), dtype=np.int64)
    if np.any(in_bounds):
        out[in_bounds] = mask[y[in_bounds], x[in_bounds]]
    return out, in_bounds


def _entropy_from_counter(counter: Counter[int]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    probs = np.asarray([float(v) / total for v in counter.values()], dtype=np.float64)
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log2(probs)))


def _summarize_counts(
    *,
    mode: str,
    scene: str,
    window: int,
    frame_ids: list[int],
    available: list[bool],
    num_surfels: int,
    positive_counts: np.ndarray,
    observation_ids: dict[int, Counter[int]],
    visible_samples: int,
    positive_samples: int,
    target_available_visible_samples: int,
    target_positive_samples: int,
    contradiction_samples: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    positive_counts = np.asarray(positive_counts, dtype=np.float64)
    surfels_with_positive = positive_counts > 0.0
    entropies = [_entropy_from_counter(counter) for counter in observation_ids.values() if counter]
    row = {
        "status": "ok",
        "mode": mode,
        "scene": scene,
        "window": int(window),
        "num_frames": int(len(frame_ids)),
        "num_mask_frames_available": int(sum(available)),
        "num_mask_frames_missing": int(len(frame_ids) - sum(available)),
        "mask_frame_density": float(sum(available) / max(len(frame_ids), 1)),
        "num_surfels": int(num_surfels),
        "visible_samples": int(visible_samples),
        "positive_samples": int(positive_samples),
        "visible_but_unobserved_rate": float(1.0 - positive_samples / max(visible_samples, 1)),
        "surfel_positive_observation_rate": float(np.count_nonzero(surfels_with_positive) / max(num_surfels, 1)),
        "mean_positive_observations_per_surfel": float(np.mean(positive_counts)) if positive_counts.size else 0.0,
        "median_positive_observations_per_surfel": float(np.median(positive_counts)) if positive_counts.size else 0.0,
        "p90_positive_observations_per_surfel": float(np.percentile(positive_counts, 90))
        if positive_counts.size
        else 0.0,
        "observation_entropy_per_surfel_mean": float(np.mean(entropies)) if entropies else 0.0,
        "target_available_visible_samples": int(target_available_visible_samples),
        "target_positive_samples": int(target_positive_samples),
        "mask_propagation_self_consistency": float(
            target_positive_samples / max(target_available_visible_samples, 1)
        ),
        "inside_outside_contradiction_rate": float(
            contradiction_samples / max(target_available_visible_samples, 1)
        ),
        "same_frame_competing_measurement_rate": 0.0,
        **extra,
    }
    return row


def _visible_ok(
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


def _target_mask_sampling_mode(
    *,
    mode: str,
    scene: str,
    window: int,
    frame_ids: list[int],
    available: list[bool],
    masks: list[np.ndarray | None],
    uv_pred: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    min_visibility: float,
    min_confidence: float,
    rng: np.random.Generator,
    shuffle_uv: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    num_surfels = int(uv_pred.shape[1])
    positive_counts = np.zeros((num_surfels,), dtype=np.float64)
    observation_ids: dict[int, Counter[int]] = defaultdict(Counter)
    visible_samples = 0
    positive_samples = 0
    target_available_visible_samples = 0
    target_positive_samples = 0
    for local_idx, frame_id in enumerate(frame_ids):
        ok = _visible_ok(
            uv_pred[local_idx],
            valid[local_idx],
            visibility[local_idx],
            confidence[local_idx],
            min_visibility=min_visibility,
            min_confidence=min_confidence,
        )
        visible_samples += int(np.count_nonzero(ok))
        if not available[local_idx] or masks[local_idx] is None:
            continue
        uv = uv_pred[local_idx].copy()
        if shuffle_uv and uv.shape[0] > 1:
            uv = uv[rng.permutation(uv.shape[0])]
        sampled, in_bounds = _sample_mask(masks[local_idx], uv)
        use = ok & in_bounds
        target_available_visible_samples += int(np.count_nonzero(use))
        positive = use & (sampled > 0)
        target_positive_samples += int(np.count_nonzero(positive))
        positive_samples += int(np.count_nonzero(positive))
        for carrier_idx in np.flatnonzero(positive).tolist():
            positive_counts[int(carrier_idx)] += 1.0
            observation_ids[int(carrier_idx)][int(sampled[carrier_idx])] += 1
    return _summarize_counts(
        mode=mode,
        scene=scene,
        window=window,
        frame_ids=frame_ids,
        available=available,
        num_surfels=num_surfels,
        positive_counts=positive_counts,
        observation_ids=observation_ids,
        visible_samples=visible_samples,
        positive_samples=positive_samples,
        target_available_visible_samples=target_available_visible_samples,
        target_positive_samples=target_positive_samples,
        contradiction_samples=target_available_visible_samples - target_positive_samples,
        extra=extra,
    )


def _source_propagation_mode(
    *,
    mode: str,
    scene: str,
    window: int,
    frame_ids: list[int],
    available: list[bool],
    masks: list[np.ndarray | None],
    uv_pred: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    src_frame_global: np.ndarray,
    src_mask_id: np.ndarray,
    min_visibility: float,
    min_confidence: float,
    rng: np.random.Generator,
    shuffle_source: bool,
    no_track: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    num_surfels = int(uv_pred.shape[1])
    source_frames = np.asarray(src_frame_global, dtype=np.int64).copy()
    source_masks = np.asarray(src_mask_id, dtype=np.int64).copy()
    if shuffle_source and num_surfels > 1:
        order = rng.permutation(num_surfels)
        source_frames = source_frames[order]
        source_masks = source_masks[order]
    source_positive = source_masks > 0
    positive_counts = np.zeros((num_surfels,), dtype=np.float64)
    observation_ids: dict[int, Counter[int]] = defaultdict(Counter)
    visible_samples = 0
    positive_samples = 0
    target_available_visible_samples = 0
    target_positive_samples = 0
    contradiction_samples = 0

    if no_track:
        visible_samples = int(num_surfels)
        positive_samples = int(np.count_nonzero(source_positive))
        positive_counts[source_positive] = 1.0
        for carrier_idx in np.flatnonzero(source_positive).tolist():
            key = int(source_frames[carrier_idx]) * 1_000_000 + int(source_masks[carrier_idx])
            observation_ids[int(carrier_idx)][key] += 1
        return _summarize_counts(
            mode=mode,
            scene=scene,
            window=window,
            frame_ids=frame_ids,
            available=available,
            num_surfels=num_surfels,
            positive_counts=positive_counts,
            observation_ids=observation_ids,
            visible_samples=visible_samples,
            positive_samples=positive_samples,
            target_available_visible_samples=0,
            target_positive_samples=0,
            contradiction_samples=0,
            extra=extra,
        )

    for local_idx, frame_id in enumerate(frame_ids):
        ok = _visible_ok(
            uv_pred[local_idx],
            valid[local_idx],
            visibility[local_idx],
            confidence[local_idx],
            min_visibility=min_visibility,
            min_confidence=min_confidence,
        )
        visible_samples += int(np.count_nonzero(ok))
        propagated = ok & source_positive
        positive_samples += int(np.count_nonzero(propagated))
        for carrier_idx in np.flatnonzero(propagated).tolist():
            positive_counts[int(carrier_idx)] += 1.0
            key = int(source_frames[carrier_idx]) * 1_000_000 + int(source_masks[carrier_idx])
            observation_ids[int(carrier_idx)][key] += 1
        if available[local_idx] and masks[local_idx] is not None:
            sampled, in_bounds = _sample_mask(masks[local_idx], uv_pred[local_idx])
            comparable = propagated & in_bounds
            target_available_visible_samples += int(np.count_nonzero(comparable))
            target_positive = comparable & (sampled > 0)
            target_positive_samples += int(np.count_nonzero(target_positive))
            contradiction_samples += int(np.count_nonzero(comparable & (sampled <= 0)))

    return _summarize_counts(
        mode=mode,
        scene=scene,
        window=window,
        frame_ids=frame_ids,
        available=available,
        num_surfels=num_surfels,
        positive_counts=positive_counts,
        observation_ids=observation_ids,
        visible_samples=visible_samples,
        positive_samples=positive_samples,
        target_available_visible_samples=target_available_visible_samples,
        target_positive_samples=target_positive_samples,
        contradiction_samples=contradiction_samples,
        extra=extra,
    )


def _process_window(args: argparse.Namespace, scene: str, carrier_path: Path, rng: np.random.Generator) -> list[dict[str, Any]]:
    with np.load(carrier_path) as data:
        uv_pred = np.asarray(data["uv_pred"], dtype=np.float32)
        visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
        confidence = np.asarray(data["confidence_prob"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool)
        src_frame_global = np.asarray(data["src_frame_global"], dtype=np.int64)
        src_mask_id = np.asarray(data["src_mask_id"], dtype=np.int64)
    frame_ids = _frame_ids_for_carrier_file(carrier_path, uv_pred.shape[0])
    window = int(carrier_path.stem.replace("carriers_window", ""))
    scene_root = Path(args.scannet_root) / scene
    mask_dir = scene_root / f"output_{args.backbone}" / "mask"
    _load_rgb_hw(scene_root, frame_ids[0])
    masks: list[np.ndarray | None] = []
    available: list[bool] = []
    for frame_id in frame_ids:
        mask = _load_mask(mask_dir / f"{int(frame_id)}.png")
        masks.append(mask)
        available.append(mask is not None)
    summary = _load_window_summary(carrier_path)
    shared_extra = {
        "carrier_file": str(carrier_path),
        "cycle_uv_error_p90": summary.get("cycle_uv_error_p90"),
        "self_uv_error_p90": summary.get("self_uv_error_p90"),
        "track_length_visible_mean": summary.get("track_length_visible_mean"),
        "uv_in01_rate": summary.get("uv_in01_rate"),
    }
    rows = [
        _target_mask_sampling_mode(
            mode="M0_cropformer_available_frames",
            scene=scene,
            window=window,
            frame_ids=frame_ids,
            available=available,
            masks=masks,
            uv_pred=uv_pred,
            valid=valid,
            visibility=visibility,
            confidence=confidence,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            rng=rng,
            shuffle_uv=False,
            extra={**shared_extra, "mode_note": "current available CropFormer masks sampled at D4RT uv"},
        ),
        _target_mask_sampling_mode(
            mode="M1_framewise_available_no_propagation",
            scene=scene,
            window=window,
            frame_ids=frame_ids,
            available=available,
            masks=masks,
            uv_pred=uv_pred,
            valid=valid,
            visibility=visibility,
            confidence=confidence,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            rng=rng,
            shuffle_uv=False,
            extra={**shared_extra, "mode_note": "same local evidence as M0 because no denser framewise mask cache is available"},
        ),
        _source_propagation_mode(
            mode="M2_d4rt_source_mask_propagation",
            scene=scene,
            window=window,
            frame_ids=frame_ids,
            available=available,
            masks=masks,
            uv_pred=uv_pred,
            valid=valid,
            visibility=visibility,
            confidence=confidence,
            src_frame_global=src_frame_global,
            src_mask_id=src_mask_id,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            rng=rng,
            shuffle_source=False,
            no_track=False,
            extra={**shared_extra, "mode_note": "source mask id propagated along real D4RT tracks"},
        ),
        _source_propagation_mode(
            mode="M4_no_track_source_frame_only",
            scene=scene,
            window=window,
            frame_ids=frame_ids,
            available=available,
            masks=masks,
            uv_pred=uv_pred,
            valid=valid,
            visibility=visibility,
            confidence=confidence,
            src_frame_global=src_frame_global,
            src_mask_id=src_mask_id,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            rng=rng,
            shuffle_source=False,
            no_track=True,
            extra={**shared_extra, "mode_note": "source-frame mask only, no temporal propagation"},
        ),
        _source_propagation_mode(
            mode="M5_shuffled_source_mask_propagation",
            scene=scene,
            window=window,
            frame_ids=frame_ids,
            available=available,
            masks=masks,
            uv_pred=uv_pred,
            valid=valid,
            visibility=visibility,
            confidence=confidence,
            src_frame_global=src_frame_global,
            src_mask_id=src_mask_id,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            rng=rng,
            shuffle_source=True,
            no_track=False,
            extra={**shared_extra, "mode_note": "source mask identities shuffled across tracks before propagation"},
        ),
        _target_mask_sampling_mode(
            mode="M5b_shuffled_uv_target_mask_control",
            scene=scene,
            window=window,
            frame_ids=frame_ids,
            available=available,
            masks=masks,
            uv_pred=uv_pred,
            valid=valid,
            visibility=visibility,
            confidence=confidence,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            rng=rng,
            shuffle_uv=True,
            extra={**shared_extra, "mode_note": "target-frame mask sampling after shuffling uv across tracks"},
        ),
    ]
    rows.append(
        {
            "status": "not_available",
            "mode": "M3_frozen_video_masklet",
            "scene": scene,
            "window": window,
            "num_frames": int(len(frame_ids)),
            "num_mask_frames_available": int(sum(available)),
            "num_mask_frames_missing": int(len(frame_ids) - sum(available)),
            "mask_frame_density": float(sum(available) / max(len(frame_ids), 1)),
            "failure_reason": "No frozen video segmentation or masklet propagation cache was found in this workspace.",
            **shared_extra,
        }
    )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("mode", ""))].append(row)
    mode_summary: dict[str, dict[str, Any]] = {}
    numeric_keys = [
        "num_frames",
        "num_mask_frames_available",
        "mask_frame_density",
        "visible_samples",
        "positive_samples",
        "visible_but_unobserved_rate",
        "surfel_positive_observation_rate",
        "mean_positive_observations_per_surfel",
        "median_positive_observations_per_surfel",
        "p90_positive_observations_per_surfel",
        "observation_entropy_per_surfel_mean",
        "mask_propagation_self_consistency",
        "inside_outside_contradiction_rate",
        "cycle_uv_error_p90",
        "self_uv_error_p90",
        "track_length_visible_mean",
        "uv_in01_rate",
    ]
    for mode, items in sorted(by_mode.items()):
        ok = [row for row in items if row.get("status") == "ok"]
        summary: dict[str, Any] = {
            "num_rows": int(len(items)),
            "num_ok_rows": int(len(ok)),
            "num_not_available_rows": int(sum(1 for row in items if row.get("status") == "not_available")),
        }
        for key in numeric_keys:
            vals = [float(row[key]) for row in ok if row.get(key) is not None]
            if vals:
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_min"] = float(np.min(vals))
                summary[f"{key}_max"] = float(np.max(vals))
        mode_summary[mode] = summary
    return {
        "diagnostic_only": True,
        "uses_gt": False,
        "is_method_result": False,
        "num_rows": int(len(rows)),
        "num_ok_rows": int(sum(1 for row in rows if row.get("status") == "ok")),
        "mode_summary": mode_summary,
    }


def _write_outputs(prefix: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": _json_safe(summary), "rows": _json_safe(rows)}
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# Stream4D v11 Measurement Density Diagnostic",
        "",
        "This diagnostic does not read GT labels and does not report AP.",
        "",
        "## Mode Summary",
        "",
        "| mode | ok rows | mask density | surfel obs rate | obs/surfel | visible unobserved | self consistency | contradiction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, item in summary["mode_summary"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    mode,
                    f"{item.get('num_ok_rows', 0)}/{item.get('num_rows', 0)}",
                    f"{float(item.get('mask_frame_density_mean', 0.0)):.6f}"
                    if "mask_frame_density_mean" in item
                    else "NA",
                    f"{float(item.get('surfel_positive_observation_rate_mean', 0.0)):.6f}"
                    if "surfel_positive_observation_rate_mean" in item
                    else "NA",
                    f"{float(item.get('mean_positive_observations_per_surfel_mean', 0.0)):.6f}"
                    if "mean_positive_observations_per_surfel_mean" in item
                    else "NA",
                    f"{float(item.get('visible_but_unobserved_rate_mean', 0.0)):.6f}"
                    if "visible_but_unobserved_rate_mean" in item
                    else "NA",
                    f"{float(item.get('mask_propagation_self_consistency_mean', 0.0)):.6f}"
                    if "mask_propagation_self_consistency_mean" in item
                    else "NA",
                    f"{float(item.get('inside_outside_contradiction_rate_mean', 0.0)):.6f}"
                    if "inside_outside_contradiction_rate_mean" in item
                    else "NA",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- M0 and M1 are identical in this workspace because no denser framewise mask cache beyond available CropFormer masks was found.",
            "- M3 is marked not available because no frozen video masklet cache was found.",
            "- M2/M5 source-mask propagation measure temporal observation density; they are diagnostic measurement tables, not object predictions.",
        ]
    )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    for scene in _read_seq_list(Path(args.seq_list)):
        scene_dir = Path(args.debug_root) / scene
        carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
        if not carrier_paths:
            rows.append({"status": "failed", "scene": scene, "mode": "all", "failure_reason": "missing carriers"})
            continue
        for carrier_path in carrier_paths:
            try:
                rows.extend(_process_window(args, scene, carrier_path, rng))
            except Exception as exc:
                rows.append(
                    {
                        "status": "failed",
                        "scene": scene,
                        "mode": "all",
                        "carrier_file": str(carrier_path),
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    summary = _aggregate(rows)
    _write_outputs(Path(args.output_prefix), rows, summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
