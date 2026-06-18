from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stream4d.scannet_stream import ScanNetStream


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
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_ids_for_window(npz_path: Path, data: np.lib.npyio.NpzFile) -> list[int]:
    manifest = _load_json(npz_path.with_name(npz_path.stem + "_manifest.json"))
    values = manifest.get("frame_ids") or manifest.get("raw_frame_ids") or manifest.get("frame_indices")
    if values:
        return [int(v) for v in values]
    return list(range(int(data["uv_pred"].shape[0])))


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_gt_instance(scene_root: Path, frame_id: int) -> np.ndarray | None:
    zip_path = scene_root / f"{scene_root.name}_2d-instance.zip"
    if not zip_path.exists():
        return None
    member = f"instance/{int(frame_id)}.png"
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if member not in set(zf.namelist()):
                return None
            data = np.frombuffer(zf.read(member), dtype=np.uint8)
    except (KeyError, zipfile.BadZipFile):
        return None
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int64)


def _tube_xy(
    uv: np.ndarray,
    valid: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    *,
    width: int,
    height: int,
    min_visibility: float,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, dtype=np.float32)
    ok = (
        np.asarray(valid, dtype=bool)
        & (np.asarray(visibility, dtype=np.float32) >= float(min_visibility))
        & (np.asarray(confidence, dtype=np.float32) >= float(min_confidence))
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= 1.0)
    )
    xy = np.zeros((uv.shape[0], 2), dtype=np.int64)
    xy[:, 0] = np.clip(np.rint(uv[:, 0] * float(max(width - 1, 0))), 0, max(width - 1, 0)).astype(np.int64)
    xy[:, 1] = np.clip(np.rint(uv[:, 1] * float(max(height - 1, 0))), 0, max(height - 1, 0)).astype(np.int64)
    return xy, ok


def _boundary_band(mask: np.ndarray, radius_px: float) -> np.ndarray:
    labels = np.asarray(mask)
    edge = np.zeros(labels.shape, dtype=np.uint8)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    edge &= (labels > 0).astype(np.uint8)
    radius = max(1, int(np.ceil(float(radius_px))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(edge, kernel, iterations=1).astype(bool) & (labels > 0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _p10(values: list[float]) -> float:
    return float(np.percentile(values, 10)) if values else 0.0


def _overlay_frame(
    *,
    rgb: np.ndarray,
    xy: np.ndarray,
    ok: np.ndarray,
    mask_at_xy: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    image = np.asarray(rgb, dtype=np.uint8).copy()
    if image.ndim != 3:
        return
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    inside = ok & (mask_at_xy > 0)
    outside = ok & (mask_at_xy <= 0)
    for x, y in xy[outside][:: max(1, int(max(np.count_nonzero(outside), 1) / 500))]:
        cv2.circle(image_bgr, (int(x), int(y)), 1, (80, 80, 255), -1)
    for x, y in xy[inside][:: max(1, int(max(np.count_nonzero(inside), 1) / 1000))]:
        cv2.circle(image_bgr, (int(x), int(y)), 1, (0, 220, 0), -1)
    cv2.putText(image_bgr, title[:100], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image_bgr)


def run_scene(
    *,
    scene: str,
    cache_root: Path,
    output_root: Path,
    backbone: str,
    min_visibility: float,
    min_confidence: float,
    large_mask_area_min: int,
    boundary_band_px: float,
    max_overlay_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stream = ScanNetStream(seq_name=scene, backbone=backbone)
    scene_dir = cache_root / scene
    per_mask_rows: list[dict[str, Any]] = []
    interior_scores: list[float] = []
    boundary_scores: list[float] = []
    inside_counts: list[int] = []
    boundary_counts: list[int] = []
    large_mask_count = 0
    covered_large_masks = 0
    covered_boundary_masks = 0
    visible_tube_counts: list[int] = []
    gt_instances: set[tuple[int, int]] = set()
    covered_gt_instances: set[tuple[int, int]] = set()
    visible_tubes_with_gt = 0
    visible_tubes_total = 0
    overlay_written = 0

    for npz_path in sorted(scene_dir.glob("carriers_window*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            frame_ids = _frame_ids_for_window(npz_path, data)
            tube_ids = (
                np.asarray(data["persistent_tube_id"], dtype=np.int64)
                if "persistent_tube_id" in data
                else np.asarray(data["carrier_id"], dtype=np.int64)
            )
            for local_t, frame_id in enumerate(frame_ids):
                mask = stream.load_mask(int(frame_id)).astype(np.int64)
                height, width = mask.shape[:2]
                xy, ok = _tube_xy(
                    data["uv_pred"][local_t],
                    data["valid"][local_t],
                    data["visibility_prob"][local_t],
                    data["confidence_prob"][local_t],
                    width=width,
                    height=height,
                    min_visibility=min_visibility,
                    min_confidence=min_confidence,
                )
                mask_at_xy = mask[xy[:, 1], xy[:, 0]]
                boundary_band = _boundary_band(mask, float(boundary_band_px))
                visible_ids = set(int(v) for v in tube_ids[ok].tolist())
                visible_tube_counts.append(len(visible_ids))
                visible_tubes_total += int(np.count_nonzero(ok))
                gt = _load_gt_instance(stream.root, int(frame_id))
                if gt is not None and gt.shape == mask.shape:
                    gt_values = gt[xy[:, 1], xy[:, 0]]
                    visible_tubes_with_gt += int(np.count_nonzero(ok & (gt_values > 0)))
                    for gt_id in np.unique(gt):
                        if int(gt_id) > 0:
                            gt_instances.add((int(frame_id), int(gt_id)))
                    for gt_id in np.unique(gt_values[ok & (gt_values > 0)]):
                        covered_gt_instances.add((int(frame_id), int(gt_id)))
                if overlay_written < int(max_overlay_frames):
                    rgb = stream.load_rgb(int(frame_id))
                    _overlay_frame(
                        rgb=rgb,
                        xy=xy,
                        ok=ok,
                        mask_at_xy=mask_at_xy,
                        out_path=output_root / "overlays" / scene / f"{int(frame_id):06d}_{npz_path.stem}.png",
                        title=f"{scene} frame {frame_id} visible={len(visible_ids)}",
                    )
                    overlay_written += 1
                for mask_id in np.unique(mask):
                    if int(mask_id) <= 0:
                        continue
                    area = int(np.count_nonzero(mask == int(mask_id)))
                    if area < int(large_mask_area_min):
                        continue
                    large_mask_count += 1
                    inside = ok & (mask_at_xy == int(mask_id))
                    inside_ids = set(int(v) for v in tube_ids[inside].tolist())
                    boundary = inside & boundary_band[xy[:, 1], xy[:, 0]]
                    boundary_ids = set(int(v) for v in tube_ids[boundary].tolist())
                    outside_visible = ok & (mask_at_xy != int(mask_id))
                    inside_count = int(len(inside_ids))
                    boundary_count = int(len(boundary_ids))
                    inside_counts.append(inside_count)
                    boundary_counts.append(boundary_count)
                    interior_score = min(float(inside_count) / 16.0, 1.0)
                    boundary_score = min(float(boundary_count) / 8.0, 1.0)
                    interior_scores.append(interior_score)
                    boundary_scores.append(boundary_score)
                    covered_large_masks += int(inside_count >= 16)
                    covered_boundary_masks += int(boundary_count >= 8)
                    if inside_count >= 16 and boundary_count >= 8:
                        status = "good"
                    elif inside_count == 0:
                        status = "uncovered"
                    elif inside_count < 16:
                        status = "weak_interior"
                    else:
                        status = "weak_boundary"
                    per_mask_rows.append(
                        {
                            "scene": scene,
                            "window_file": npz_path.name,
                            "frame_id": int(frame_id),
                            "mask_id": int(mask_id),
                            "mask_area": int(area),
                            "num_inside_tubes": inside_count,
                            "num_boundary_tubes": boundary_count,
                            "num_visible_outside_tubes": int(np.count_nonzero(outside_visible)),
                            "inside_tube_density": float(inside_count / max(area, 1)),
                            "boundary_tube_density": float(boundary_count / max(area, 1)),
                            "coverage_status": status,
                        }
                    )
    summary = {
        "scene": scene,
        "num_tubes": int(sum(row["tube_count"] for row in _scene_npz_rows(scene_dir))),
        "num_visible_tubes_mean": _mean([float(v) for v in visible_tube_counts]),
        "num_masks": int(len(per_mask_rows)),
        "large_mask_count": int(large_mask_count),
        "mask_interior_coverage_mean": _mean(interior_scores),
        "mask_interior_coverage_p10": _p10(interior_scores),
        "mask_boundary_coverage_mean": _mean(boundary_scores),
        "mask_boundary_coverage_p10": _p10(boundary_scores),
        "large_masks_with_ge16_interior_tubes_ratio": float(covered_large_masks / max(large_mask_count, 1)),
        "large_masks_with_ge8_boundary_tubes_ratio": float(covered_boundary_masks / max(large_mask_count, 1)),
        "visible_tube_coverage_proxy": _mean([float(v) for v in visible_tube_counts]),
        "uncovered_large_mask_count": int(sum(1 for v in inside_counts if v == 0)),
        "uncovered_boundary_mask_count": int(sum(1 for v in boundary_counts if v < 8)),
        "covered_GT_instance_ratio": float(len(covered_gt_instances) / max(len(gt_instances), 1)) if gt_instances else None,
        "node_gt_label_coverage": float(visible_tubes_with_gt / max(visible_tubes_total, 1)) if visible_tubes_total else None,
        "GT_instances_with_no_tubes_count": int(len(gt_instances - covered_gt_instances)) if gt_instances else None,
        "overlay_count": int(overlay_written),
    }
    return per_mask_rows, summary


def _scene_npz_rows(scene_dir: Path) -> list[dict[str, int]]:
    rows = []
    for npz_path in sorted(scene_dir.glob("carriers_window*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            rows.append({"tube_count": int(data["uv_pred"].shape[1])})
    return rows


def aggregate_scene_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = {}
    for key in rows[0].keys() if rows else []:
        vals = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and row.get(key) is not None]
        if vals:
            numeric[f"{key}_mean"] = float(np.mean(vals))
    pass_flag = bool(
        rows
        and np.mean([float(row["large_masks_with_ge16_interior_tubes_ratio"]) for row in rows]) >= 0.75
        and np.mean([float(row["large_masks_with_ge8_boundary_tubes_ratio"]) for row in rows]) >= 0.50
        and np.min([float(row["mask_interior_coverage_p10"]) for row in rows]) > 0.0
        and all(
            row.get("covered_GT_instance_ratio") is None or float(row["covered_GT_instance_ratio"]) >= 0.50
            for row in rows
        )
        and all(row.get("node_gt_label_coverage") is None or float(row["node_gt_label_coverage"]) >= 0.50 for row in rows)
    )
    return {
        "scene_count": int(len(rows)),
        "phase_c_pass": pass_flag,
        **numeric,
        "method_result": False,
        "is_diagnostic_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v26 tube coverage and measurement-readiness diagnostics.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--backbone", default="Cropformer")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--large-mask-area-min", type=int, default=1024)
    parser.add_argument("--boundary-band-px", type=float, default=3.0)
    parser.add_argument("--max-overlay-frames", type=int, default=10)
    parser.add_argument("--label", default="v26_coverage")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    scenes = _read_split(Path(args.split))
    all_mask_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for scene in scenes:
        if not (Path(args.cache_root) / scene).exists():
            continue
        mask_rows, summary = run_scene(
            scene=scene,
            cache_root=Path(args.cache_root),
            output_root=output_root,
            backbone=args.backbone,
            min_visibility=float(args.min_visibility),
            min_confidence=float(args.min_confidence),
            large_mask_area_min=int(args.large_mask_area_min),
            boundary_band_px=float(args.boundary_band_px),
            max_overlay_frames=int(args.max_overlay_frames),
        )
        all_mask_rows.extend(mask_rows)
        scene_rows.append(summary)
    aggregate = aggregate_scene_rows(scene_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / f"{args.label}_per_mask.csv", all_mask_rows)
    _write_csv(output_root / f"{args.label}_scene_rows.csv", scene_rows)
    (output_root / f"{args.label}_per_mask.json").write_text(
        json.dumps(_json_safe(all_mask_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / f"{args.label}_scene_rows.json").write_text(
        json.dumps(_json_safe(scene_rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / f"{args.label}_summary.json").write_text(
        json.dumps(_json_safe(aggregate), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(aggregate), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
