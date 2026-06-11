from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream


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


def _read_mask_observations(raw: list[dict[str, Any]], same_frame_policy: str) -> list[tuple[int, int, float]]:
    observations = [
        (int(item["frame_id"]), int(item["mask_id"]), float(item.get("coverage", 0.0)))
        for item in raw
    ]
    if same_frame_policy == "all":
        return sorted(observations, key=lambda item: (item[0], -item[2], item[1]))
    if same_frame_policy != "best_per_frame":
        raise ValueError(f"Unsupported same_frame_policy: {same_frame_policy}")
    best: dict[int, tuple[int, int, float]] = {}
    for obs in observations:
        frame_id = int(obs[0])
        prev = best.get(frame_id)
        if prev is None or float(obs[2]) > float(prev[2]):
            best[frame_id] = obs
    return sorted(best.values(), key=lambda item: (item[0], -item[2], item[1]))


def _load_local_proposals(
    debug_root: Path,
    seq_name: str,
    same_frame_policy: str,
    min_observations: int,
    min_frames: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    seq_dir = debug_root / seq_name
    if not seq_dir.exists():
        raise FileNotFoundError(seq_dir)
    object_dict: dict[int, dict[str, Any]] = {}
    raw_props = 0
    dropped = 0
    conflicts_removed = 0
    for window_path in sorted(seq_dir.glob("local_props_window*.json")):
        with window_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        window_idx = int(window_path.stem.replace("local_props_window", ""))
        for prop in payload.get("proposals", []):
            raw_obs = list(prop.get("mask_observations", []))
            raw_props += 1
            observations = _read_mask_observations(raw_obs, same_frame_policy)
            conflicts_removed += max(len(raw_obs) - len(observations), 0)
            frames = {int(item[0]) for item in observations}
            if len(observations) < int(min_observations) or len(frames) < int(min_frames):
                dropped += 1
                continue
            object_id = len(object_dict)
            object_dict[object_id] = {
                "point_ids": np.empty((0,), dtype=np.int64),
                "mask_list": observations,
                "repre_mask_list": sorted(observations, key=lambda item: item[2], reverse=True)[:3],
                "carrier_ids": np.empty((0,), dtype=np.int64),
                "source_window": int(window_idx),
                "source_proposal_id": int(prop.get("proposal_id", -1)),
                "num_carriers": int(prop.get("num_carriers", 0)),
            }
    return object_dict, {
        "raw_local_proposals": float(raw_props),
        "kept_local_proposals": float(len(object_dict)),
        "dropped_local_proposals": float(dropped),
        "same_frame_conflicts_removed": float(conflicts_removed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local proposal bank from Stream4D debug local_props.")
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--same-frame-policy", default="best_per_frame", choices=["all", "best_per_frame"])
    parser.add_argument("--min-observations", type=int, default=1)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=12000)
    parser.add_argument("--export-max-masks-per-object", type=int, default=5)
    parser.add_argument("--export-mask-min-relative-coverage", type=float, default=0.0)
    parser.add_argument("--export-min-points-per-object", type=int, default=100)
    parser.add_argument("--export-score-mode", default="observations", choices=["one", "area", "observations", "reliability"])
    parser.add_argument("--summary-root", default="outputs/local_proposal_bank")
    args = parser.parse_args()

    object_dict, local_diag = _load_local_proposals(
        debug_root=Path(args.debug_root),
        seq_name=args.seq_name,
        same_frame_policy=args.same_frame_policy,
        min_observations=args.min_observations,
        min_frames=args.min_frames,
    )
    stream = ScanNetStream(seq_name=args.seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=args.export_nn_radius,
        export_support_mode="mask_backproject",
        export_mask_sample_stride=args.export_mask_sample_stride,
        export_mask_max_pixels=args.export_mask_max_pixels,
        export_max_masks_per_object=args.export_max_masks_per_object,
        export_mask_min_relative_coverage=args.export_mask_min_relative_coverage,
        export_min_points_per_object=args.export_min_points_per_object,
        export_score_mode=args.export_score_mode,
    )
    export_diag = exporter.export_object_dict_mask_backproject(object_dict)
    summary = {
        "args": vars(args),
        "local": local_diag,
        "export": export_diag,
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.output_config}_{args.seq_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    with (out_dir / f"{args.output_config}_latest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
