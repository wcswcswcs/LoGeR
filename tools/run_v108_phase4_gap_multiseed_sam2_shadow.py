#!/usr/bin/env python3
"""Run a visual-first SAM2 multiseed shadow probe for v108 gap components.

This diagnostic treats selected current-output label components as gap
components. It samples multiple interior points away from component edges,
runs SAM2 image predictor in shadow mode, and writes high-resolution visual
overlays for manual review. It does not mutate Stream4D output, lifecycle
state, or SAM2 video memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = ROOT / "Stream3D"
GSAM2_ROOT = Path(os.environ.get("GSAM2_ROOT", str(ROOT / "Grounded-SAM-2"))).resolve()
TOOLS_ROOT = ROOT / "tools"
for item in (TOOLS_ROOT, ROOT, STREAM3D_ROOT, GSAM2_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from Stream3D.stream4d_v108.geometry_capsule import (  # noqa: E402
    bbox_from_mask,
    mask_depth_support,
    mask_edge_distance,
    sample_interior_points,
)
from run_v107_phase5_prompt_capsule_visibility_probe import (  # noqa: E402
    load_lingbot_geometry,
    resize_label_to_shape,
)


DEFAULT_REFERENCE_ROOT = (
    ROOT
    / "Stream3D/outputs/audit/v108_phase1_reference_scene0050_90f_labelonly_20260714_1442"
    / "v106_stateful_sam2_rolling_scene_stream"
)
DEFAULT_LINGBOT_NPZ = (
    ROOT
    / "Stream3D/outputs/audit/v107_phase6_lingbot_prompt_capsule_delta32_20260713_2145"
    / "lingbot_raw_geometry_outputs.npz"
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fields})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def load_rgb(scene_root: Path, scene_id: str, frame_id: int) -> np.ndarray:
    path = scene_root / scene_id / "color" / f"{int(frame_id)}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_reference_records(reference_root: Path) -> dict[int, dict[str, Any]]:
    summary_path = reference_root / "summary.json"
    if not summary_path.exists():
        nested = reference_root / "v106_stateful_sam2_rolling_scene_stream" / "summary.json"
        if nested.exists():
            summary_path = nested
    summary = read_json(summary_path)
    records: dict[int, dict[str, Any]] = {}
    for row in summary.get("records", []):
        item = dict(row)
        item["label_path"] = resolve_path(str(row["label_path"]), summary_path.parent)
        records[int(row["frame_id"])] = item
    return records


def parse_ids(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def select_gap_object_ids(
    target_label: np.ndarray,
    *,
    source_label: np.ndarray | None,
    max_objects: int,
    min_area_px: int,
) -> list[int]:
    rows: list[tuple[int, int]] = []
    for obj_id in sorted(int(v) for v in np.unique(target_label).tolist() if int(v) > 0):
        area = int(np.count_nonzero(target_label == int(obj_id)))
        if area < int(min_area_px):
            continue
        if source_label is not None and int(np.count_nonzero(source_label == int(obj_id))) > 0:
            continue
        rows.append((area, int(obj_id)))
    rows.sort(reverse=True)
    return [obj_id for _area, obj_id in rows[: int(max_objects)]]


def deepest_seed(mask: np.ndarray) -> tuple[int, int, float] | None:
    mask_b = np.asarray(mask).astype(bool)
    if not np.any(mask_b):
        return None
    dist = mask_edge_distance(mask_b)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(y), int(x), float(dist[int(y), int(x)])


def mask_stats(mask: np.ndarray, *, image_hw: tuple[int, int]) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    area = int(np.count_nonzero(mask_b))
    h, w = int(image_hw[0]), int(image_hw[1])
    bbox = bbox_from_mask(mask_b)
    if bbox is None:
        return {
            "area_px": 0,
            "area_frac": 0.0,
            "bbox_xyxy": [],
            "bbox_area_frac": 0.0,
            "bbox_extent": 0.0,
            "edge_touch_count": 0,
        }
    x0, y0, x1, y1 = bbox
    bw = max(1, int(x1 - x0 + 1))
    bh = max(1, int(y1 - y0 + 1))
    touches = [x0 <= 0, y0 <= 0, x1 >= w - 1, y1 >= h - 1]
    return {
        "area_px": int(area),
        "area_frac": float(area / max(1, h * w)),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "bbox_area_frac": float((bw * bh) / max(1, h * w)),
        "bbox_extent": float(area / max(1, bw * bh)),
        "edge_touch_count": int(sum(bool(v) for v in touches)),
    }


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = rgb.copy()
    mask_b = np.asarray(mask).astype(bool)
    if np.any(mask_b):
        c = np.asarray(color, dtype=np.float32)
        out[mask_b] = ((1.0 - float(alpha)) * out[mask_b].astype(np.float32) + float(alpha) * c).clip(0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2, lineType=cv2.LINE_AA)
    return out


def draw_seed(image: np.ndarray, xy: tuple[float, float], text: str, *, color: tuple[int, int, int]) -> None:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    cv2.circle(image, (x, y), 8, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), 10, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(image, text[:10], (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    header = 34
    out = np.zeros((image.shape[0] + header, image.shape[1], 3), dtype=np.uint8)
    out[:] = 12
    out[header:] = image
    cv2.putText(out, text[:180], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_sam2_predictor(args: argparse.Namespace):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = resolve_path(str(args.sam2_checkpoint), ROOT)
    model = build_sam2(str(args.sam2_model_cfg), str(checkpoint), device=str(args.device))
    dtype_name = str(args.model_dtype).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        model.to(dtype=torch.bfloat16)
    elif dtype_name in {"fp16", "float16"}:
        model.to(dtype=torch.float16)
    model.eval()
    return SAM2ImagePredictor(model), checkpoint


def autocast_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if not str(args.device).startswith("cuda"):
        return {"enabled": False}
    dtype_name = str(args.model_dtype).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}
    if dtype_name in {"fp16", "float16"}:
        return {"device_type": "cuda", "dtype": torch.float16, "enabled": True}
    return {"enabled": False}


def run_object_probe(
    *,
    obj_id: int,
    rgb: np.ndarray,
    target_label: np.ndarray,
    predictor: Any,
    geometry_support: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    case_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current_mask = target_label == int(obj_id)
    image_hw = rgb.shape[:2]
    deepest = deepest_seed(current_mask)
    seeds: list[tuple[str, int, int, float]] = []
    if deepest is not None:
        y, x, dist = deepest
        seeds.append(("G0_deepest", int(y), int(x), float(dist)))
    sampled, sample_stats = sample_interior_points(
        current_mask,
        count=int(args.num_seeds),
        min_distance_px=float(args.seed_min_distance_px),
        seed=int(args.seed) + int(obj_id) * 101 + int(case_index),
    )
    for idx, (y, x, dist) in enumerate(sampled):
        seeds.append((f"G1_fps_{idx}", int(y), int(x), float(dist)))
    unique: list[tuple[str, int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for name, y, x, dist in seeds:
        key = (int(y), int(x))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, y, x, dist))
    seeds = unique[: max(1, int(args.num_seeds) + 1)]

    seed_overlay = overlay_mask(rgb, current_mask, color=(40, 220, 255), alpha=0.36)
    for seed_idx, (name, y, x, _dist) in enumerate(seeds):
        draw_seed(seed_overlay, (float(x), float(y)), f"s{seed_idx}", color=(30, 245, 70))
    seed_vis = add_header(seed_overlay, f"target frame {args.target_frame_id} obj {obj_id}; current gap component with interior seeds")
    seed_path = output_root / "visual_checks" / f"case_{case_index:02d}_obj{obj_id:04d}_seed_overlay.png"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(seed_path), cv2.cvtColor(seed_vis, cv2.COLOR_RGB2BGR))

    candidate_rows: list[dict[str, Any]] = []
    candidate_visual_paths: list[Path] = [seed_path]
    current_stats = mask_stats(current_mask, image_hw=image_hw)
    for seed_idx, (name, y, x, dist) in enumerate(seeds):
        point_coords = np.asarray([[float(x), float(y)]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int32)
        with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
            masks, scores, _logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=bool(args.multimask_output),
            )
        mask_arr = np.asarray(masks)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, ...]
        score_arr = np.asarray(scores if scores is not None else np.zeros((mask_arr.shape[0],), dtype=np.float32)).reshape(-1)
        for cand_idx, raw in enumerate(mask_arr):
            candidate_mask = np.squeeze(raw) > 0
            stats = mask_stats(candidate_mask, image_hw=image_hw)
            inter = int(np.count_nonzero(candidate_mask & current_mask))
            union = int(np.count_nonzero(candidate_mask | current_mask))
            row = {
                "case_index": int(case_index),
                "target_obj_id": int(obj_id),
                "seed_index": int(seed_idx),
                "seed_name": str(name),
                "seed_x": float(x),
                "seed_y": float(y),
                "seed_distance_to_component_edge_px": float(dist),
                "candidate_index": int(cand_idx),
                "sam2_score": float(score_arr[cand_idx]) if cand_idx < len(score_arr) else 0.0,
                "current_component_area_px": int(current_stats["area_px"]),
                "candidate_to_current_iou_diagnostic": float(inter / max(union, 1)),
                **{f"candidate_{key}": val for key, val in stats.items()},
            }
            candidate_rows.append(row)
    candidate_rows.sort(key=lambda row: (int(row["case_index"]), -float(row["sam2_score"]), int(row["seed_index"]), int(row["candidate_index"])))

    for visual_idx, row in enumerate(candidate_rows[: int(args.visual_topk_per_object)]):
        seed_x = float(row["seed_x"])
        seed_y = float(row["seed_y"])
        seed_idx = int(row["seed_index"])
        cand_idx = int(row["candidate_index"])
        point_coords = np.asarray([[seed_x, seed_y]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int32)
        with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
            masks, scores, _logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=bool(args.multimask_output),
            )
        mask_arr = np.asarray(masks)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, ...]
        candidate_mask = np.squeeze(mask_arr[cand_idx]) > 0
        panel = overlay_mask(rgb, current_mask, color=(40, 220, 255), alpha=0.24)
        panel = overlay_mask(panel, candidate_mask, color=(255, 70, 190), alpha=0.38)
        draw_seed(panel, (seed_x, seed_y), f"s{seed_idx}", color=(30, 245, 70))
        title = (
            f"obj {obj_id}; SAM2 candidate seed={seed_idx} cand={cand_idx}; "
            f"score={float(row['sam2_score']):.3f}; visual review required"
        )
        vis = add_header(panel, title)
        path = output_root / "visual_checks" / (
            f"case_{case_index:02d}_obj{obj_id:04d}_candidate_{visual_idx:02d}_seed{seed_idx}_cand{cand_idx}.png"
        )
        cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        candidate_visual_paths.append(path)

    case_summary = {
        "case_index": int(case_index),
        "target_obj_id": int(obj_id),
        "current_component_stats": current_stats,
        "geometry_support": geometry_support,
        "seed_sample_stats": sample_stats,
        "seed_count": int(len(seeds)),
        "candidate_count": int(len(candidate_rows)),
        "seed_overlay": rel(seed_path),
        "seed_overlay_sha256": sha256_file(seed_path),
        "visual_paths": [rel(path) for path in candidate_visual_paths],
        "visual_sha256": {rel(path): sha256_file(path) for path in candidate_visual_paths},
        "visual_review_required": True,
        "acceptance_rule": "Metrics are diagnostic only; quality must be judged by high-resolution visual review.",
    }
    return case_summary, candidate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed")
    parser.add_argument("--reference-run-root", default=str(DEFAULT_REFERENCE_ROOT))
    parser.add_argument("--lingbot-geometry-npz", default=str(DEFAULT_LINGBOT_NPZ))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-frame-id", type=int, default=4495)
    parser.add_argument("--target-frame-id", type=int, default=4500)
    parser.add_argument("--object-ids", default="")
    parser.add_argument("--max-objects", type=int, default=2)
    parser.add_argument("--min-gap-area-px", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--seed-min-distance-px", type=float, default=16.0)
    parser.add_argument("--multimask-output", type=int, default=1)
    parser.add_argument("--visual-topk-per-object", type=int, default=3)
    parser.add_argument("--sam2-checkpoint", default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bf16", choices=["float32", "bf16", "float16"])
    parser.add_argument("--min-depth-conf", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1085)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_root = resolve_path(str(args.output_root), ROOT)
    scene_root = resolve_path(str(args.scene_root), ROOT)
    reference_root = resolve_path(str(args.reference_run_root), ROOT)
    npz_path = resolve_path(str(args.lingbot_geometry_npz), ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    records_by_frame = load_reference_records(reference_root)
    if int(args.target_frame_id) not in records_by_frame:
        raise RuntimeError({"missing_target_frame": int(args.target_frame_id), "reference_root": rel(reference_root)})
    target_label = load_label(Path(records_by_frame[int(args.target_frame_id)]["label_path"]))
    source_label = None
    if int(args.source_frame_id) in records_by_frame:
        source_label = load_label(Path(records_by_frame[int(args.source_frame_id)]["label_path"]))
    rgb = load_rgb(scene_root, str(args.scene_id), int(args.target_frame_id))

    object_ids = parse_ids(str(args.object_ids))
    if not object_ids:
        object_ids = select_gap_object_ids(
            target_label,
            source_label=source_label,
            max_objects=int(args.max_objects),
            min_area_px=int(args.min_gap_area_px),
        )
    if not object_ids:
        raise RuntimeError("no gap object ids selected")

    geometry_available = False
    geometry_support_by_obj: dict[int, dict[str, Any]] = {}
    geometry_npz_sha = ""
    lingbot_frame_range: list[int] = []
    if npz_path.exists():
        geometry_npz_sha = sha256_file(npz_path)
        geometry = load_lingbot_geometry(npz_path)
        frame_ids = [int(v) for v in np.asarray(geometry["frame_ids"]).tolist()]
        lingbot_frame_range = [int(frame_ids[0]), int(frame_ids[-1])] if frame_ids else []
        frame_to_index = {int(frame_id): int(idx) for idx, frame_id in enumerate(frame_ids)}
        if int(args.target_frame_id) in frame_to_index:
            geometry_available = True
            idx = int(frame_to_index[int(args.target_frame_id)])
            target_label_lingbot = resize_label_to_shape(target_label, tuple(int(v) for v in geometry["depth"].shape[1:3]))
            for obj_id in object_ids:
                geometry_support_by_obj[int(obj_id)] = mask_depth_support(
                    target_label_lingbot == int(obj_id),
                    depth=geometry["depth"][idx],
                    depth_conf=geometry["depth_conf"][idx],
                    min_depth_conf=float(args.min_depth_conf),
                    core_min_distance_px=float(args.seed_min_distance_px),
                )

    predictor, checkpoint = build_sam2_predictor(args)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
        predictor.set_image(rgb)

    case_summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for case_idx, obj_id in enumerate(object_ids):
        case_summary, rows = run_object_probe(
            obj_id=int(obj_id),
            rgb=rgb,
            target_label=target_label,
            predictor=predictor,
            geometry_support=geometry_support_by_obj.get(int(obj_id), {"geometry_available": bool(geometry_available)}),
            args=args,
            output_root=output_root,
            case_index=int(case_idx),
        )
        case_summaries.append(case_summary)
        all_rows.extend(rows)

    rows_csv = output_root / "sam2_gap_candidate_rows.csv"
    write_csv(rows_csv, all_rows)
    rows_jsonl = output_root / "sam2_gap_candidate_rows.jsonl"
    rows_jsonl.write_text("".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in all_rows), encoding="utf-8")
    case_summary_path = output_root / "case_summaries.json"
    write_json(case_summary_path, {"cases": case_summaries})
    peak_cuda_mb = 0.0
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_cuda_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    summary_path = output_root / "phase4_gap_multiseed_sam2_shadow_summary.json"
    summary = {
        "schema_version": "stream4d_v108_phase4_gap_multiseed_sam2_shadow_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "scene_id": str(args.scene_id),
        "source_frame_id": int(args.source_frame_id),
        "target_frame_id": int(args.target_frame_id),
        "object_ids": [int(v) for v in object_ids],
        "case_count": int(len(case_summaries)),
        "candidate_row_count": int(len(all_rows)),
        "sam2_checkpoint": rel(checkpoint),
        "sam2_checkpoint_sha256": sha256_file(checkpoint),
        "sam2_model_cfg": str(args.sam2_model_cfg),
        "device": str(args.device),
        "model_dtype": str(args.model_dtype),
        "peak_cuda_allocated_mb": peak_cuda_mb,
        "geometry_available_for_target": bool(geometry_available),
        "geometry_source": "LingBot-Map decoded pose_enc + depth + depth_conf + intrinsics" if geometry_available else "not_available_for_target_frame",
        "uses_scannet_pose_or_depth_for_projection": False,
        "lingbot_geometry_npz": rel(npz_path) if npz_path.exists() else "",
        "lingbot_geometry_npz_sha256": geometry_npz_sha,
        "lingbot_frame_id_first_last": lingbot_frame_range,
        "rows_csv": rel(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
        "rows_jsonl": rel(rows_jsonl),
        "rows_jsonl_sha256": sha256_file(rows_jsonl),
        "case_summaries": rel(case_summary_path),
        "case_summaries_sha256": sha256_file(case_summary_path),
        "visual_paths": [path for case in case_summaries for path in case["visual_paths"]],
        "acceptance_rule": "Metrics are diagnostic only; quality must be judged by high-resolution visual review.",
        "shadow_only": True,
    }
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "case_count": len(case_summaries), "candidate_row_count": len(all_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
