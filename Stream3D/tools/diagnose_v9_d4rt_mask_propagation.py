from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_seq_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _rgb_shape(root: Path, scene_id: str, frame_id: int) -> tuple[int, int]:
    path = root / "data" / "scannet" / "processed" / scene_id / "color" / f"{int(frame_id)}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return int(height), int(width)


def _carrier_path(root: Path, run_name: str, scene_id: str, window_idx: int) -> Path:
    return (
        root
        / "outputs"
        / "v8_d4rt_grid_surfel_field"
        / run_name
        / scene_id
        / f"carriers_window{int(window_idx):03d}.npz"
    )


def process_scene(args: argparse.Namespace, root: Path, scene_id: str) -> dict[str, Any]:
    path = _carrier_path(root, args.carrier_run, scene_id, args.window_idx)
    row: dict[str, Any] = {
        "scene_id": scene_id,
        "carrier_path": str(path),
        "ok": False,
        "error": "",
    }
    if not path.exists():
        row["error"] = f"missing carrier npz: {path}"
        return row
    with np.load(path) as data:
        src_frame = data["src_frame"].astype(np.int64)
        src_frame_global = data["src_frame_global"].astype(np.int64)
        src_mask_id = data["src_mask_id"].astype(np.int64)
        uv_pred = data["uv_pred"].astype(np.float32)
        valid = data["valid"].astype(bool)
        visibility_prob = data["visibility_prob"].astype(np.float32)

    num_frames, num_carriers = valid.shape
    if uv_pred.shape[:2] != valid.shape:
        row["error"] = f"uv/valid shape mismatch: uv={uv_pred.shape}, valid={valid.shape}"
        return row
    frame_ids = [int(item) for item in sorted(set(src_frame_global.tolist()))]
    if len(frame_ids) != num_frames:
        # G1 stores one source frame per carrier; reconstruct the 16f window from local indices.
        frame_ids = list(range(num_frames))
    height, width = _rgb_shape(root, scene_id, int(src_frame_global.min()) if src_frame_global.size else 0)

    positive = src_mask_id > 0
    visible = valid & (visibility_prob >= float(args.visibility_threshold))
    uv_in = (
        (uv_pred[..., 0] >= 0.0)
        & (uv_pred[..., 0] <= 1.0)
        & (uv_pred[..., 1] >= 0.0)
        & (uv_pred[..., 1] <= 1.0)
    )
    propagated = visible & uv_in & positive[None, :]
    propagated_counts = propagated.sum(axis=1).astype(np.int64)
    frames_with_propagation = propagated_counts > 0

    per_carrier_obs = propagated.sum(axis=0).astype(np.int64)
    positive_obs = per_carrier_obs[positive]

    collision_rates: list[float] = []
    conflicting_pixel_rates: list[float] = []
    unique_slot_counts: list[int] = []
    for frame_idx in range(num_frames):
        keep = propagated[frame_idx]
        if not np.any(keep):
            collision_rates.append(0.0)
            conflicting_pixel_rates.append(0.0)
            unique_slot_counts.append(0)
            continue
        uv = uv_pred[frame_idx, keep]
        xs = np.clip(np.rint(uv[:, 0] * (width - 1)).astype(np.int64), 0, width - 1)
        ys = np.clip(np.rint(uv[:, 1] * (height - 1)).astype(np.int64), 0, height - 1)
        pixels = ys * width + xs
        slots = np.stack([src_frame_global[keep], src_mask_id[keep]], axis=1)
        order = np.argsort(pixels)
        pixels_sorted = pixels[order]
        slots_sorted = slots[order]
        total_points = int(pixels_sorted.shape[0])
        unique_pixels, starts, counts = np.unique(pixels_sorted, return_index=True, return_counts=True)
        del unique_pixels
        collision_points = int(np.sum(counts[counts > 1]))
        conflict_pixels = 0
        for start, count in zip(starts, counts):
            if count <= 1:
                continue
            group = slots_sorted[start : start + count]
            if np.unique(group, axis=0).shape[0] > 1:
                conflict_pixels += 1
        collision_rates.append(float(collision_points / max(total_points, 1)))
        conflicting_pixel_rates.append(float(conflict_pixels / max(counts.shape[0], 1)))
        unique_slot_counts.append(int(np.unique(slots, axis=0).shape[0]))

    row.update(
        {
            "ok": True,
            "num_frames": int(num_frames),
            "num_carriers": int(num_carriers),
            "num_positive_source_carriers": int(np.count_nonzero(positive)),
            "positive_source_carrier_rate": float(np.count_nonzero(positive) / max(num_carriers, 1)),
            "num_positive_source_slots": int(np.unique(np.stack([src_frame_global[positive], src_mask_id[positive]], axis=1), axis=0).shape[0])
            if np.any(positive)
            else 0,
            "frames_with_propagated_measurement": int(np.count_nonzero(frames_with_propagation)),
            "propagated_frame_rate": float(np.count_nonzero(frames_with_propagation) / max(num_frames, 1)),
            "propagated_positive_carriers_per_frame_mean": float(np.mean(propagated_counts)),
            "propagated_positive_carriers_per_frame_min": int(np.min(propagated_counts)) if propagated_counts.size else 0,
            "propagated_positive_carriers_per_frame_max": int(np.max(propagated_counts)) if propagated_counts.size else 0,
            "positive_carrier_observations_mean": float(np.mean(positive_obs)) if positive_obs.size else 0.0,
            "positive_carrier_observations_p10": float(np.percentile(positive_obs, 10)) if positive_obs.size else 0.0,
            "positive_carrier_observations_p90": float(np.percentile(positive_obs, 90)) if positive_obs.size else 0.0,
            "collision_point_rate_mean": float(np.mean(collision_rates)) if collision_rates else 0.0,
            "conflicting_pixel_rate_mean": float(np.mean(conflicting_pixel_rates)) if conflicting_pixel_rates else 0.0,
            "unique_propagated_slots_per_frame_mean": float(np.mean(unique_slot_counts)) if unique_slot_counts else 0.0,
            "visibility_threshold": float(args.visibility_threshold),
        }
    )
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get("ok") and row.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    keys = [
        "num_frames",
        "num_carriers",
        "num_positive_source_carriers",
        "positive_source_carrier_rate",
        "num_positive_source_slots",
        "frames_with_propagated_measurement",
        "propagated_frame_rate",
        "propagated_positive_carriers_per_frame_mean",
        "positive_carrier_observations_mean",
        "positive_carrier_observations_p10",
        "positive_carrier_observations_p90",
        "collision_point_rate_mean",
        "conflicting_pixel_rate_mean",
        "unique_propagated_slots_per_frame_mean",
    ]
    return {
        "diagnostic_only": True,
        "uses_gt": False,
        "is_method_result": False,
        "carrier_run": args.carrier_run,
        "num_scenes": len(rows),
        "num_ok_scenes": len(ok_rows),
        "num_failed_scenes": len(rows) - len(ok_rows),
        "visibility_threshold": float(args.visibility_threshold),
        **{f"{key}_mean": _mean(ok_rows, key) for key in keys},
    }


def write_outputs(output_prefix: Path, payload: dict[str, Any]) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rows = payload["rows"]
    with output_prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "scene_id",
            "ok",
            "error",
            "num_positive_source_carriers",
            "positive_source_carrier_rate",
            "num_positive_source_slots",
            "frames_with_propagated_measurement",
            "propagated_frame_rate",
            "propagated_positive_carriers_per_frame_mean",
            "positive_carrier_observations_mean",
            "collision_point_rate_mean",
            "conflicting_pixel_rate_mean",
            "unique_propagated_slots_per_frame_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    lines = [
        "# v9 D4RT Mask Propagation Diagnostic",
        "",
        "This diagnostic uses existing D4RT carriers and source-frame 2D mask ids. It does not run a new 2D model, does not read GT, and does not report AP.",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| scene | positive carriers | source slots | propagated frames | frame rate | obs/carrier | collision | conflict pixels | slots/frame |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['scene_id']} | {row.get('num_positive_source_carriers')} | "
            f"{row.get('num_positive_source_slots')} | {row.get('frames_with_propagated_measurement')} | "
            f"{float(row.get('propagated_frame_rate', 0.0)):.6f} | "
            f"{float(row.get('positive_carrier_observations_mean', 0.0)):.6f} | "
            f"{float(row.get('collision_point_rate_mean', 0.0)):.6f} | "
            f"{float(row.get('conflicting_pixel_rate_mean', 0.0)):.6f} | "
            f"{float(row.get('unique_propagated_slots_per_frame_mean', 0.0)):.6f} |"
        )
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose D4RT propagation of existing 2D mask measurements.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--carrier-run", required=True)
    parser.add_argument("--window-idx", type=int, default=0)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--output-prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    scene_ids = _read_seq_list(root / args.seq_list)
    rows = [process_scene(args, root, scene_id) for scene_id in scene_ids]
    payload = {"args": vars(args), "aggregate": aggregate(rows, args), "rows": rows}
    write_outputs(root / args.output_prefix, payload)
    print(f"[v9-mask-propagation] wrote {root / args.output_prefix}.md")


if __name__ == "__main__":
    main()
