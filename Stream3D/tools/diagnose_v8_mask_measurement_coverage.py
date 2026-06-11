from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.carrier_store import CarrierBatch
from stream4d.mask_evidence import MaskEvidenceBuilder


def _read_seq_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_carrier_batch(path: Path) -> CarrierBatch:
    data = np.load(path, allow_pickle=False)
    return CarrierBatch(
        carrier_id=np.asarray(data["carrier_id"], dtype=np.int64),
        src_frame=np.asarray(data["src_frame"], dtype=np.int64),
        src_uv=np.asarray(data["src_uv"], dtype=np.float32),
        xyz_ref=np.asarray(data["xyz_ref"], dtype=np.float32),
        uv_pred=np.asarray(data["uv_pred"], dtype=np.float32),
        visibility_prob=np.asarray(data["visibility_prob"], dtype=np.float32),
        confidence_prob=np.asarray(data["confidence_prob"], dtype=np.float32),
        valid=np.asarray(data["valid"], dtype=bool),
        src_frame_global=np.asarray(data["src_frame_global"], dtype=np.int64)
        if "src_frame_global" in data
        else None,
        src_xy=np.asarray(data["src_xy"], dtype=np.int64) if "src_xy" in data else None,
        src_mask_id=np.asarray(data["src_mask_id"], dtype=np.int64) if "src_mask_id" in data else None,
    )


def _load_mask_or_zero(mask_dir: Path, frame_id: int, fallback_hw: tuple[int, int]) -> tuple[np.ndarray, bool]:
    path = mask_dir / f"{int(frame_id)}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return np.zeros(fallback_hw, dtype=np.int32), False
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int32, copy=False), True


def _load_rgb_hw(scene_root: Path, frame_id: int) -> tuple[int, int]:
    path = scene_root / "color" / f"{int(frame_id)}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB frame for mask fallback shape: {path}")
    return int(image.shape[0]), int(image.shape[1])


def _load_window_summary(scene_dir: Path, window_idx: int) -> dict[str, Any]:
    path = scene_dir / f"window{int(window_idx):03d}_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_outputs(prefix: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": _json_safe(summary), "rows": _json_safe(rows)}
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    lines = [
        "# v8 Mask Measurement Coverage Diagnostic",
        "",
        "This diagnostic does not read GT labels and does not report AP.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Scene | Window | Mask frames | Missing frames | Assign rate all | Assign rate mask-only | Surfels observed | Obs/surfel |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {scene} | {window} | {mask_frames} | {missing} | {assign_all} | {assign_avail} | {surfels} | {obs} |".format(
                scene=row["scene"],
                window=row["window"],
                mask_frames=row["num_mask_frames_available"],
                missing=row["num_mask_frames_missing"],
                assign_all=f"{row['carrier_assignment_rate_all_frames']:.6g}",
                assign_avail=f"{row['carrier_assignment_rate_available_mask_frames']:.6g}",
                surfels=f"{row['surfel_positive_observation_rate']:.6g}",
                obs=f"{row['mean_positive_observations_per_surfel']:.6g}",
            )
        )
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "diagnostic_only": True,
        "uses_gt": False,
        "is_method_result": False,
        "num_windows": int(len(rows)),
        "num_ok_windows": int(sum(1 for row in rows if row.get("status") == "ok")),
        "num_failed_windows": int(sum(1 for row in rows if row.get("status") != "ok")),
    }
    for key in (
        "num_frames",
        "num_mask_frames_available",
        "num_mask_frames_missing",
        "num_raw_mask_observations_all_frames",
        "num_mask_observations_with_carriers_all_frames",
        "num_raw_mask_observations_available_mask_frames",
        "num_mask_observations_with_carriers_available_mask_frames",
        "carrier_visibility_rate_all_frames",
        "carrier_assignment_rate_all_frames",
        "carrier_visibility_rate_available_mask_frames",
        "carrier_assignment_rate_available_mask_frames",
        "surfel_positive_observation_rate",
        "mean_positive_observations_per_surfel",
    ):
        values = [float(row[key]) for row in rows if row.get("status") == "ok" and row.get(key) is not None]
        if values:
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_min"] = float(np.min(values))
            out[f"{key}_max"] = float(np.max(values))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--rho-min", type=float, default=0.35)
    parser.add_argument("--output-prefix", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    debug_root = Path(args.debug_root)
    scenes = _read_seq_list(Path(args.seq_list))
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        scene_dir = debug_root / scene
        carrier_files = sorted(scene_dir.glob("carriers_window*.npz"))
        if not carrier_files:
            rows.append({"status": "failed", "scene": scene, "window": "", "failure_reason": "missing carriers"})
            continue
        scene_root = Path(args.scannet_root) / scene
        mask_dir = scene_root / f"output_{args.backbone}" / "mask"
        for carrier_file in carrier_files:
            window = carrier_file.stem.replace("carriers_window", "")
            try:
                batch = _load_carrier_batch(carrier_file)
                window_summary = _load_window_summary(scene_dir, int(window))
                frame_ids = [int(v) for v in window_summary["frame_ids"]]
                fallback_hw = _load_rgb_hw(scene_root, frame_ids[0])
                masks: list[np.ndarray] = []
                available: list[bool] = []
                for frame_id in frame_ids:
                    mask, ok = _load_mask_or_zero(mask_dir, frame_id, fallback_hw)
                    masks.append(mask)
                    available.append(bool(ok))
                masks_arr = np.stack(masks, axis=0)
                builder = MaskEvidenceBuilder(rho_min=float(args.rho_min))
                observations, diag_all = builder.build(batch, masks_arr, frame_ids)

                available_idx = np.flatnonzero(np.asarray(available, dtype=bool))
                if available_idx.size:
                    sub_batch = CarrierBatch(
                        carrier_id=batch.carrier_id,
                        src_frame=batch.src_frame,
                        src_uv=batch.src_uv,
                        xyz_ref=batch.xyz_ref[available_idx],
                        uv_pred=batch.uv_pred[available_idx],
                        visibility_prob=batch.visibility_prob[available_idx],
                        confidence_prob=batch.confidence_prob[available_idx],
                        valid=batch.valid[available_idx],
                        src_frame_global=batch.src_frame_global,
                        src_xy=batch.src_xy,
                        src_mask_id=batch.src_mask_id,
                    )
                    _, diag_available = builder.build(sub_batch, masks_arr[available_idx], [frame_ids[i] for i in available_idx])
                else:
                    diag_available = {
                        "num_raw_mask_observations": 0.0,
                        "num_mask_observations_with_carriers": 0.0,
                        "carrier_visibility_rate": 0.0,
                        "carrier_assignment_rate": 0.0,
                        "mean_mask_carrier_count": 0.0,
                    }

                surfel_obs_counts: dict[int, int] = {}
                for obs in observations:
                    for carrier_id in obs.carrier_ids.tolist():
                        key = int(carrier_id)
                        surfel_obs_counts[key] = surfel_obs_counts.get(key, 0) + 1
                num_surfels = int(batch.carrier_id.shape[0])
                row = {
                    "status": "ok",
                    "scene": scene,
                    "window": window,
                    "num_frames": int(len(frame_ids)),
                    "num_mask_frames_available": int(sum(available)),
                    "num_mask_frames_missing": int(len(available) - sum(available)),
                    "available_frame_ids": [int(frame_ids[i]) for i, ok in enumerate(available) if ok],
                    "missing_frame_ids": [int(frame_ids[i]) for i, ok in enumerate(available) if not ok],
                    "num_surfels": num_surfels,
                    "num_surfels_with_positive_observation": int(len(surfel_obs_counts)),
                    "surfel_positive_observation_rate": float(len(surfel_obs_counts) / max(num_surfels, 1)),
                    "mean_positive_observations_per_surfel": float(
                        sum(surfel_obs_counts.values()) / max(num_surfels, 1)
                    ),
                    "num_raw_mask_observations_all_frames": int(diag_all["num_raw_mask_observations"]),
                    "num_mask_observations_with_carriers_all_frames": int(
                        diag_all["num_mask_observations_with_carriers"]
                    ),
                    "carrier_visibility_rate_all_frames": float(diag_all["carrier_visibility_rate"]),
                    "carrier_assignment_rate_all_frames": float(diag_all["carrier_assignment_rate"]),
                    "mean_mask_carrier_count_all_frames": float(diag_all["mean_mask_carrier_count"]),
                    "num_raw_mask_observations_available_mask_frames": int(
                        diag_available["num_raw_mask_observations"]
                    ),
                    "num_mask_observations_with_carriers_available_mask_frames": int(
                        diag_available["num_mask_observations_with_carriers"]
                    ),
                    "carrier_visibility_rate_available_mask_frames": float(
                        diag_available["carrier_visibility_rate"]
                    ),
                    "carrier_assignment_rate_available_mask_frames": float(
                        diag_available["carrier_assignment_rate"]
                    ),
                    "mean_mask_carrier_count_available_mask_frames": float(
                        diag_available["mean_mask_carrier_count"]
                    ),
                    "failure_reason": "",
                }
            except Exception as exc:
                row = {
                    "status": "failed",
                    "scene": scene,
                    "window": window,
                    "failure_reason": repr(exc),
                }
            rows.append(row)
    summary = _aggregate(rows)
    _write_outputs(Path(args.output_prefix), rows, summary)
    print(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
