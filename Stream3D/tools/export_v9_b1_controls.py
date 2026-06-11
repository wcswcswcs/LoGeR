from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.export_scannet import ScanNetExporter
from stream4d.scannet_stream import ScanNetStream
from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_seq_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64, copy=False)


def _sample_mask_ids(mask: np.ndarray, uv_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    x = np.rint(uv_norm[:, 0] * float(max(width - 1, 1))).astype(np.int64)
    y = np.rint(uv_norm[:, 1] * float(max(height - 1, 1))).astype(np.int64)
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    out = np.zeros((uv_norm.shape[0],), dtype=np.int64)
    if np.any(in_bounds):
        out[in_bounds] = mask[y[in_bounds], x[in_bounds]]
    return out, in_bounds


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


def _target_count_from_summary(summary_root: Path, match_count_config: str, scene_id: str) -> int | None:
    for name in (
        f"{match_count_config}_{scene_id}_summary.json",
        f"{match_count_config}_summary.json",
    ):
        path = summary_root / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("seq_name") == scene_id and payload.get("num_exported_objects") is not None:
            return int(round(float(payload["num_exported_objects"])))
        for row in payload.get("scenes", []):
            if row.get("seq_name") == scene_id and row.get("num_exported_objects") is not None:
                return int(round(float(row["num_exported_objects"])))
    return None


def _available_mask_observations(stream: ScanNetStream, frame_ids: list[int]) -> list[tuple[int, int, int]]:
    observations: list[tuple[int, int, int]] = []
    for frame_id in sorted(set(int(v) for v in frame_ids)):
        mask = _load_mask(stream.mask_dir / f"{frame_id}.png")
        if mask is None:
            continue
        ids, counts = np.unique(mask[mask > 0], return_counts=True)
        for mask_id, count in zip(ids.tolist(), counts.tolist()):
            observations.append((int(frame_id), int(mask_id), int(count)))
    return observations


def _d4rt_mask_counts(
    *,
    stream: ScanNetStream,
    carrier_paths: list[Path],
    min_visibility: float,
    min_confidence: float,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[Counter[tuple[int, int]], dict[str, float]]:
    counts: Counter[tuple[int, int]] = Counter()
    total_valid = 0
    total_positive = 0
    available_frames = set()
    for carrier_path in carrier_paths:
        with np.load(carrier_path) as data:
            uv_pred = np.asarray(data["uv_pred"], dtype=np.float32)
            visibility = np.asarray(data["visibility_prob"], dtype=np.float32)
            confidence = np.asarray(data["confidence_prob"], dtype=np.float32)
        frame_ids = _frame_ids_for_carrier_file(carrier_path, uv_pred.shape[0])
        for local_idx, frame_id in enumerate(frame_ids):
            mask = _load_mask(stream.mask_dir / f"{int(frame_id)}.png")
            if mask is None:
                continue
            available_frames.add(int(frame_id))
            valid = (visibility[local_idx] >= float(min_visibility)) & (
                confidence[local_idx] >= float(min_confidence)
            )
            mask_ids, in_bounds = _sample_mask_ids(mask, uv_pred[local_idx])
            valid &= in_bounds
            valid_idx = np.flatnonzero(valid)
            total_valid += int(valid_idx.shape[0])
            if valid_idx.size == 0:
                continue
            values = mask_ids[valid_idx].astype(np.int64, copy=True)
            if shuffle:
                positive_pool = values[values > 0]
                if positive_pool.size:
                    values = rng.choice(positive_pool, size=values.shape[0], replace=True)
            positive = values > 0
            total_positive += int(np.count_nonzero(positive))
            for mask_id in values[positive].tolist():
                counts[(int(frame_id), int(mask_id))] += 1
    return counts, {
        "num_available_mask_frames": float(len(available_frames)),
        "d4rt_valid_samples": float(total_valid),
        "d4rt_positive_samples": float(total_positive),
        "d4rt_positive_sample_rate": float(total_positive / max(total_valid, 1)),
    }


def _no_track_counts(carrier_paths: list[Path]) -> tuple[Counter[tuple[int, int]], dict[str, float]]:
    counts: Counter[tuple[int, int]] = Counter()
    total = 0
    positive = 0
    for carrier_path in carrier_paths:
        with np.load(carrier_path) as data:
            frames = np.asarray(data["src_frame_global"], dtype=np.int64)
            masks = np.asarray(data["src_mask_id"], dtype=np.int64)
        total += int(masks.shape[0])
        keep = masks > 0
        positive += int(np.count_nonzero(keep))
        for frame_id, mask_id in zip(frames[keep].tolist(), masks[keep].tolist()):
            counts[(int(frame_id), int(mask_id))] += 1
    return counts, {
        "no_track_source_samples": float(total),
        "no_track_positive_samples": float(positive),
        "no_track_positive_sample_rate": float(positive / max(total, 1)),
    }


def _select_observations(
    *,
    mode: str,
    stream: ScanNetStream,
    carrier_paths: list[Path],
    frame_ids: list[int],
    target_count: int,
    min_visibility: float,
    min_confidence: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, float]], dict[str, float]]:
    all_obs = _available_mask_observations(stream, frame_ids)
    if target_count <= 0 or not all_obs:
        return [], {"target_count": float(target_count), "candidate_observations": float(len(all_obs))}

    if mode == "area_same_count":
        ranked = sorted(all_obs, key=lambda item: (-int(item[2]), int(item[0]), int(item[1])))
        selected = [(frame_id, mask_id, float(area)) for frame_id, mask_id, area in ranked[:target_count]]
        return selected, {"candidate_observations": float(len(all_obs)), "target_count": float(target_count)}

    if mode == "random_same_count":
        order = rng.permutation(len(all_obs))[:target_count]
        selected = [(all_obs[int(idx)][0], all_obs[int(idx)][1], float(all_obs[int(idx)][2])) for idx in order]
        return selected, {"candidate_observations": float(len(all_obs)), "target_count": float(target_count)}

    if mode == "no_track":
        counts, diag = _no_track_counts(carrier_paths)
    elif mode == "maskcount_same_count":
        counts, diag = _d4rt_mask_counts(
            stream=stream,
            carrier_paths=carrier_paths,
            min_visibility=min_visibility,
            min_confidence=min_confidence,
            rng=rng,
            shuffle=False,
        )
    elif mode == "shuffle":
        counts, diag = _d4rt_mask_counts(
            stream=stream,
            carrier_paths=carrier_paths,
            min_visibility=min_visibility,
            min_confidence=min_confidence,
            rng=rng,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unsupported control mode: {mode}")

    ranked_counts = sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0][0]), int(item[0][1])))
    selected = [(frame_id, mask_id, float(count)) for (frame_id, mask_id), count in ranked_counts[:target_count]]
    diag.update({"candidate_observations": float(len(counts)), "target_count": float(target_count)})
    return selected, diag


def _export_scene(args: argparse.Namespace, seq_name: str, rng: np.random.Generator) -> dict[str, Any]:
    stream = ScanNetStream(seq_name=seq_name, backbone=args.backbone)
    errors = stream.validate(require_masks=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    scene_dir = Path(args.debug_root) / seq_name
    carrier_paths = sorted(scene_dir.glob("carriers_window*.npz"))
    if not carrier_paths:
        raise FileNotFoundError(f"No carrier windows under {scene_dir}")

    frame_ids: list[int] = []
    for carrier_path in carrier_paths:
        with np.load(carrier_path) as data:
            frame_ids.extend(_frame_ids_for_carrier_file(carrier_path, np.asarray(data["uv_pred"]).shape[0]))

    target_count = _target_count_from_summary(Path(args.match_count_summary_root), args.match_count_config, seq_name)
    if target_count is None:
        target_count = int(args.fallback_target_count)
    selected, diag = _select_observations(
        mode=args.control_mode,
        stream=stream,
        carrier_paths=carrier_paths,
        frame_ids=frame_ids,
        target_count=int(target_count),
        min_visibility=float(args.min_visibility),
        min_confidence=float(args.min_confidence),
        rng=rng,
    )
    object_dict: dict[int, dict[str, Any]] = {}
    for object_id, (frame_id, mask_id, weight) in enumerate(selected):
        object_dict[int(object_id)] = {
            "mask_list": [(int(frame_id), int(mask_id), float(weight))],
            "carrier_ids": np.empty((0,), dtype=np.int64),
            "surfel_member_count": 0,
        }

    exporter = ScanNetExporter(
        stream,
        output_config=args.output_config,
        export_nn_radius=float(args.export_nn_radius),
        export_support_mode="mask_backproject",
        export_mask_sample_stride=int(args.export_mask_sample_stride),
        export_mask_max_pixels=int(args.export_mask_max_pixels),
        export_min_points_per_object=int(args.min_points_per_object),
        export_score_mode=args.export_score_mode,
    )
    export_diag = exporter.export_object_dict_mask_backproject(object_dict)
    summary = {
        "seq_name": seq_name,
        "algorithm": "v9_b1_control",
        "control_mode": args.control_mode,
        "uses_gt": False,
        "is_method_result": True,
        "target_count": int(target_count),
        "num_selected_observations": int(len(selected)),
        "selected_observations": selected,
        **diag,
        **export_diag,
    }
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.output_config}_{seq_name}_summary.json"
    summary["summary_path"] = str(path)
    path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _write_aggregate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    out_dir = Path(args.summary_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    numeric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool)
    )
    numeric_mean = {
        key: float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in numeric_keys
        if any(key in row for row in rows)
    }
    payload = {
        "args": vars(args),
        "algorithm": "v9_b1_control",
        "control_mode": args.control_mode,
        "uses_gt": False,
        "is_method_result": True,
        "num_scenes": len(rows),
        "numeric_mean": numeric_mean,
        "scenes": rows,
    }
    json_path = out_dir / f"{args.output_config}_summary.json"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    csv_path = out_dir / f"{args.output_config}_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seq_name"] + numeric_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in ["seq_name"] + numeric_keys})
    md_path = out_dir / f"{args.output_config}_summary.md"
    lines = [
        f"# {args.output_config}",
        "",
        "| scene | selected | objects | points | conflict | candidates |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("seq_name")),
                    str(row.get("num_selected_observations")),
                    f"{float(row.get('num_exported_objects', 0.0)):.0f}",
                    f"{float(row.get('num_exported_points', 0.0)):.0f}",
                    f"{float(row.get('export_conflict_rate', 0.0)):.6f}",
                    f"{float(row.get('candidate_observations', 0.0)):.0f}",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = build_prediction_manifest(
        root=".",
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[str(args.debug_root), args.match_count_config],
        pre_points_policy="recompute",
        support_policy=f"v9_b1_control:{args.control_mode}:mask_backproject",
        notes=(
            f"v9 B1 control {args.control_mode}. Prediction is generated from existing D4RT carrier/mask "
            "measurements or 2D mask heuristics only; GT is not read."
        ),
        extra={
            "algorithm": "v9_b1_control",
            "control_mode": args.control_mode,
            "eval_policy": "own_recompute_control",
            "summary_path": str(json_path),
            "seq_list": str(args.seq_list),
        },
    )
    write_prediction_manifest(args.output_config, manifest, root=".")
    print(f"[v9-b1-controls] wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-root", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--control-mode",
        required=True,
        choices=["no_track", "shuffle", "random_same_count", "area_same_count", "maskcount_same_count"],
    )
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--summary-root", default="outputs/v9_b1_controls")
    parser.add_argument("--match-count-config", default="stream4d_v8_b1_surfacelet_singlemask_probe5")
    parser.add_argument("--match-count-summary-root", default="outputs/v8_surfel_object_field")
    parser.add_argument("--fallback-target-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--export-nn-radius", type=float, default=0.05)
    parser.add_argument("--export-mask-sample-stride", type=int, default=2)
    parser.add_argument("--export-mask-max-pixels", type=int, default=50000)
    parser.add_argument("--min-points-per-object", type=int, default=20)
    parser.add_argument(
        "--export-score-mode",
        choices=["one", "area", "reliability", "observations", "dense_quality", "selection_quality"],
        default="reliability",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    rows = [_export_scene(args, seq_name, rng) for seq_name in _read_seq_list(Path(args.seq_list))]
    _write_aggregate(args, rows)


if __name__ == "__main__":
    main()
