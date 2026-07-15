#!/usr/bin/env python3
"""Run a live-order v107 reactivation pilot from LingBot-projected prompt packets.

This is intentionally a Phase8 pilot, not the final rolling SAM2 lifecycle
integration. It replays dormant->reactivating events in chronological order,
uses LingBot-visible prompt packets for SAM2 image reactivation, applies an
online prompt-consistency/two-frame confirmation gate, and writes lifecycle
records that mirror the v107 design docs.
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GSAM2_ROOT = Path(os.environ.get("GSAM2_ROOT", str(ROOT / "Grounded-SAM-2"))).resolve()
for item in (GSAM2_ROOT, ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


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
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def resolve_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def load_reference_records(reference_root: Path) -> dict[int, dict[str, Any]]:
    summary = read_json(reference_root / "summary.json")
    records: dict[int, dict[str, Any]] = {}
    for row in summary.get("records", []):
        frame_id = int(row["frame_id"])
        item = dict(row)
        item["label_path"] = resolve_path(str(row["label_path"]), reference_root)
        item["rgb_path"] = resolve_path(str(row.get("rgb_path", "")), reference_root) if row.get("rgb_path") else None
        records[frame_id] = item
    return records


def load_points(points_json: Path) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    payload = read_json(points_json)
    out: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("rows", []):
        key = (int(row["target_frame_index"]), int(row["target_obj_id"]), int(row["source_frame_index"]))
        out[key].append(row)
    return out


def load_rows(rows_csv: Path) -> list[dict[str, Any]]:
    with rows_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int_set(text: str) -> set[int]:
    out: set[int] = set()
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def map_lingbot_xy_to_original(
    x: float,
    y: float,
    *,
    lingbot_hw: tuple[int, int],
    orig_hw: tuple[int, int],
) -> tuple[float, float]:
    lh, lw = int(lingbot_hw[0]), int(lingbot_hw[1])
    oh, ow = int(orig_hw[0]), int(orig_hw[1])
    ox = float(x) * float(max(ow - 1, 1)) / float(max(lw - 1, 1))
    oy = float(y) * float(max(oh - 1, 1)) / float(max(lh - 1, 1))
    return ox, oy


def mask_metrics(mask: np.ndarray, ref: np.ndarray, target_label: np.ndarray, neg_ids: set[int]) -> dict[str, Any]:
    pred = np.asarray(mask).astype(bool)
    refb = np.asarray(ref).astype(bool)
    inter = int(np.count_nonzero(pred & refb))
    pred_area = int(np.count_nonzero(pred))
    ref_area = int(np.count_nonzero(refb))
    union = int(np.count_nonzero(pred | refb))
    neg_mask = np.isin(target_label, list(neg_ids)) if neg_ids else np.zeros_like(target_label, dtype=bool)
    neg_overlap = int(np.count_nonzero(pred & neg_mask))
    return {
        "mask_area_px": pred_area,
        "reference_area_px": ref_area,
        "intersection_px": inter,
        "iou_to_reference": float(inter / max(union, 1)),
        "precision_to_reference": float(inter / max(pred_area, 1)),
        "recall_to_reference": float(inter / max(ref_area, 1)),
        "negative_sibling_overlap_px": neg_overlap,
        "negative_sibling_overlap_rate": float(neg_overlap / max(pred_area, 1)),
    }


def point_hit_rate(mask: np.ndarray, coords: list[tuple[float, float]]) -> float:
    if not coords:
        return 0.0
    h, w = mask.shape[:2]
    hit = 0
    for x, y in coords:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < w and 0 <= yi < h and bool(mask[yi, xi]):
            hit += 1
    return float(hit / max(len(coords), 1))


def choose_cases(
    rows: list[dict[str, Any]],
    points_by_case: dict[tuple[int, int, int], list[dict[str, Any]]],
    *,
    attempt_lags: set[int],
    confirm_step: int,
    max_events: int,
    per_lag: int,
) -> list[dict[str, Any]]:
    usable = [r for r in rows if str(r.get("usable_positive_negative_prompt", "")).lower() == "true"]
    selected: list[dict[str, Any]] = []
    lag_counts: dict[int, int] = defaultdict(int)
    usable.sort(
        key=lambda r: (
            int(r["target_frame_index"]),
            int(r.get("source_lag", 0)),
            -int(r.get("positive_visible_count", 0)),
            int(r.get("negative_reference_hits_target_obj_count", 0)),
            -int(r.get("negative_visible_count", 0)),
            int(r["target_obj_id"]),
        )
    )
    row_by_confirm_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    for r in usable:
        row_by_confirm_key[(int(r["target_frame_index"]), int(r["target_obj_id"]), int(r["source_frame_index"]))] = r
    for row in usable:
        lag = int(row.get("source_lag", 0))
        if attempt_lags and lag not in attempt_lags:
            continue
        if lag_counts[lag] >= int(per_lag):
            continue
        key = (int(row["target_frame_index"]), int(row["target_obj_id"]), int(row["source_frame_index"]))
        confirm_key = (int(row["target_frame_index"]) + int(confirm_step), int(row["target_obj_id"]), int(row["source_frame_index"]))
        if key not in points_by_case or confirm_key not in points_by_case or confirm_key not in row_by_confirm_key:
            continue
        row = dict(row)
        row["_confirm_row"] = row_by_confirm_key[confirm_key]
        selected.append(row)
        lag_counts[lag] += 1
        if len(selected) >= int(max_events):
            break
    return selected


def build_sam2_predictor(args: argparse.Namespace):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = Path(args.sam2_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
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
    dtype = torch.float32
    enabled = False
    if dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
        enabled = True
    elif dtype_name in {"fp16", "float16"}:
        dtype = torch.float16
        enabled = True
    return {"device_type": "cuda", "dtype": dtype, "enabled": enabled}


def select_mask_by_score(
    masks: Any,
    scores: Any,
    ref_mask: np.ndarray,
    target_label: np.ndarray,
    neg_ids: set[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    mask_arr = np.asarray(masks)
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]
    score_arr = np.asarray(scores if scores is not None else np.zeros((mask_arr.shape[0],), dtype=np.float32)).reshape(-1)
    candidates = []
    for idx, raw in enumerate(mask_arr):
        mask = np.squeeze(raw) > 0
        metrics = mask_metrics(mask, ref_mask, target_label, neg_ids)
        score = float(score_arr[idx]) if idx < len(score_arr) else 0.0
        metrics.update({"candidate_index": int(idx), "sam2_score": score})
        candidates.append((score, idx, mask, metrics))
    if not candidates:
        empty = np.zeros_like(ref_mask, dtype=bool)
        return empty, {"candidate_index": -1, "sam2_score": 0.0, **mask_metrics(empty, ref_mask, target_label, neg_ids)}
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, _idx, mask, metrics = candidates[0]
    oracle = max((item[3]["iou_to_reference"] for item in candidates), default=0.0)
    metrics["oracle_best_iou_among_multimask"] = float(oracle)
    return mask.astype(bool), metrics


def contact_sheet(paths: list[Path], out_path: Path, cols: int = 1, pad: int = 6) -> Path | None:
    images = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not images:
        return None
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), dtype=np.uint8)
    canvas[:] = 24
    for idx, image in enumerate(images):
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        y = (idx // cols) * (h + pad)
        x = (idx % cols) * (w + pad)
        canvas[y : y + h, x : x + w] = image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_path


def overlay_panel(rgb: np.ndarray, ref_mask: np.ndarray, mask: np.ndarray | None, label: str, color: tuple[int, int, int]) -> np.ndarray:
    panel = rgb.copy()
    if mask is not None:
        color_arr = np.asarray(color, dtype=np.float32)
        m = np.asarray(mask).astype(bool)
        panel[m] = (0.45 * panel[m].astype(np.float32) + 0.55 * color_arr).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(ref_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(panel, label[:72], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def prompt_panel(rgb: np.ndarray, ref_mask: np.ndarray, coords: np.ndarray, labels: np.ndarray, label: str) -> np.ndarray:
    panel = overlay_panel(rgb, ref_mask, None, label, (255, 255, 255))
    for (x, y), point_label in zip(coords, labels):
        color = (25, 240, 60) if int(point_label) == 1 else (240, 50, 45)
        cv2.circle(panel, (int(round(x)), int(round(y))), 7, color, -1, lineType=cv2.LINE_AA)
    return panel


def draw_event(
    *,
    event_id: int,
    obj_id: int,
    lag: int,
    attempt_frame_id: int,
    confirm_frame_id: int,
    attempt_rgb: np.ndarray,
    confirm_rgb: np.ndarray,
    attempt_ref: np.ndarray,
    confirm_ref: np.ndarray,
    attempt_coords: np.ndarray,
    attempt_labels: np.ndarray,
    confirm_coords: np.ndarray,
    confirm_labels: np.ndarray,
    attempt_masks: dict[str, np.ndarray],
    confirm_masks: dict[str, np.ndarray],
    attempt_iou: dict[str, float],
    confirm_iou: dict[str, float],
) -> np.ndarray:
    colors = {"G1_pos": (40, 220, 255), "G2_pos_neg": (255, 70, 170)}
    row_a = [
        prompt_panel(attempt_rgb, attempt_ref, attempt_coords, attempt_labels, f"attempt f={attempt_frame_id} obj={obj_id}"),
    ]
    row_c = [
        prompt_panel(confirm_rgb, confirm_ref, confirm_coords, confirm_labels, f"confirm f={confirm_frame_id} obj={obj_id}"),
    ]
    for variant in ("G1_pos", "G2_pos_neg"):
        row_a.append(
            overlay_panel(
                attempt_rgb,
                attempt_ref,
                attempt_masks[variant],
                f"{variant} iou={attempt_iou.get(variant, 0.0):.3f}",
                colors[variant],
            )
        )
        row_c.append(
            overlay_panel(
                confirm_rgb,
                confirm_ref,
                confirm_masks[variant],
                f"{variant} iou={confirm_iou.get(variant, 0.0):.3f}",
                colors[variant],
            )
        )
    top = np.concatenate(row_a, axis=1)
    bottom = np.concatenate(row_c, axis=1)
    merged = np.concatenate([top, bottom], axis=0)
    header = 32
    out = np.zeros((merged.shape[0] + header, merged.shape[1], 3), dtype=np.uint8)
    out[:] = 12
    out[header:] = merged
    title = f"phase8 pilot event={event_id} obj={obj_id} lag={lag} two-frame reactivation"
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def flatten_variant(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-probe-root", required=True)
    parser.add_argument("--reference-run-root", required=True)
    parser.add_argument("--scene-root", default="Stream3D/data/scannet/processed/scene0050_00")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rows-csv", default="prompt_capsule_visibility_rows.csv")
    parser.add_argument("--points-json", default="prompt_capsule_visible_point_records.json")
    parser.add_argument("--attempt-source-lags", default="1,2,4,8")
    parser.add_argument("--confirm-frame-step", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=24)
    parser.add_argument("--max-events-per-lag", type=int, default=6)
    parser.add_argument("--min-sam2-score", type=float, default=0.50)
    parser.add_argument("--min-positive-support-rate", type=float, default=0.75)
    parser.add_argument("--max-negative-conflict-rate", type=float, default=0.12)
    parser.add_argument("--min-area-ratio-to-source", type=float, default=0.20)
    parser.add_argument("--max-area-ratio-to-source", type=float, default=4.50)
    parser.add_argument("--min-temporal-area-ratio", type=float, default=0.25)
    parser.add_argument("--max-temporal-area-ratio", type=float, default=4.00)
    parser.add_argument("--merge-overlap-threshold", type=float, default=0.10)
    parser.add_argument("--sam2-checkpoint", default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bf16", choices=["float32", "bf16", "float16"])
    parser.add_argument("--multimask-output", type=int, default=1)
    parser.add_argument("--max-visual-events", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    prompt_root = Path(args.prompt_probe_root)
    if not prompt_root.is_absolute():
        prompt_root = ROOT / prompt_root
    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root
    scene_root = Path(args.scene_root)
    if not scene_root.is_absolute():
        scene_root = ROOT / scene_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )

    probe_summary = read_json(prompt_root / "prompt_capsule_visibility_probe_summary.json")
    if bool(probe_summary.get("uses_scannet_pose_or_depth_for_projection", True)):
        raise RuntimeError("prompt probe summary is not LingBot-only geometry")
    lingbot_hw = tuple(int(v) for v in probe_summary["raw_lingbot_geometry"].get("image_shape", [392, 518]))
    rows = load_rows(prompt_root / args.rows_csv)
    points_by_case = load_points(prompt_root / args.points_json)
    cases = choose_cases(
        rows,
        points_by_case,
        attempt_lags=parse_int_set(str(args.attempt_source_lags)),
        confirm_step=int(args.confirm_frame_step),
        max_events=int(args.max_events),
        per_lag=int(args.max_events_per_lag),
    )
    reference_records = load_reference_records(reference_root)
    predictor, sam2_checkpoint = build_sam2_predictor(args)

    labels_cache: dict[int, np.ndarray] = {}
    rgb_cache: dict[int, np.ndarray] = {}
    current_frame_id: int | None = None

    def get_label(frame_id: int) -> np.ndarray:
        if frame_id not in labels_cache:
            labels_cache[frame_id] = load_label(Path(reference_records[frame_id]["label_path"]))
        return labels_cache[frame_id]

    def get_rgb(frame_id: int) -> np.ndarray:
        if frame_id not in rgb_cache:
            rgb_bgr = cv2.imread(str(scene_root / "color" / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise FileNotFoundError(scene_root / "color" / f"{frame_id}.jpg")
            rgb_cache[frame_id] = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        return rgb_cache[frame_id]

    def set_image(frame_id: int) -> np.ndarray:
        nonlocal current_frame_id
        rgb = get_rgb(frame_id)
        if current_frame_id != frame_id:
            with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
                predictor.set_image(rgb)
            current_frame_id = frame_id
        return rgb

    def build_prompt(row: dict[str, Any], rgb: np.ndarray) -> dict[str, Any]:
        key = (int(row["target_frame_index"]), int(row["target_obj_id"]), int(row["source_frame_index"]))
        point_records = points_by_case[key]
        positives = [p for p in point_records if p.get("role") == "positive"]
        negatives = [p for p in point_records if p.get("role") == "negative"]
        orig_hw = rgb.shape[:2]
        coords_pos = [
            map_lingbot_xy_to_original(
                float(p["target_x"]),
                float(p["target_y"]),
                lingbot_hw=lingbot_hw,
                orig_hw=orig_hw,
            )
            for p in positives
        ]
        coords_neg = [
            map_lingbot_xy_to_original(
                float(p["target_x"]),
                float(p["target_y"]),
                lingbot_hw=lingbot_hw,
                orig_hw=orig_hw,
            )
            for p in negatives
        ]
        return {
            "coords_pos": coords_pos,
            "coords_neg": coords_neg,
            "neg_ids": {int(p["source_obj_id"]) for p in negatives},
            "positive_count": int(len(coords_pos)),
            "negative_count": int(len(coords_neg)),
        }

    def predict_variant(
        *,
        variant: str,
        prompt: dict[str, Any],
        ref_mask: np.ndarray,
        target_label: np.ndarray,
        source_area: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        coords_pos = list(prompt["coords_pos"])
        coords_neg = list(prompt["coords_neg"])
        if variant == "G1_pos":
            coords = coords_pos
            labels = [1] * len(coords_pos)
        elif variant == "G2_pos_neg":
            coords = coords_pos + coords_neg
            labels = [1] * len(coords_pos) + [0] * len(coords_neg)
        else:
            raise ValueError(variant)
        if not coords:
            mask = np.zeros_like(ref_mask, dtype=bool)
            metrics = {"candidate_index": -1, "sam2_score": 0.0, **mask_metrics(mask, ref_mask, target_label, prompt["neg_ids"])}
        else:
            point_coords = np.asarray(coords, dtype=np.float32)
            point_labels = np.asarray(labels, dtype=np.int32)
            with torch.inference_mode(), torch.autocast(**autocast_kwargs(args)):
                masks, scores, _logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=bool(args.multimask_output),
                )
            mask, metrics = select_mask_by_score(masks, scores, ref_mask, target_label, prompt["neg_ids"])
        pos_support = point_hit_rate(mask, coords_pos)
        neg_conflict = point_hit_rate(mask, coords_neg)
        area_ratio = float(metrics["mask_area_px"] / max(int(source_area), 1))
        online_gate_pass = bool(
            int(metrics["mask_area_px"]) > 0
            and float(metrics["sam2_score"]) >= float(args.min_sam2_score)
            and pos_support >= float(args.min_positive_support_rate)
            and neg_conflict <= float(args.max_negative_conflict_rate)
            and float(args.min_area_ratio_to_source) <= area_ratio <= float(args.max_area_ratio_to_source)
        )
        metrics.update(
            {
                "positive_support_rate": pos_support,
                "negative_prompt_conflict_rate": neg_conflict,
                "area_ratio_to_source_snapshot": area_ratio,
                "online_gate_pass": online_gate_pass,
                "online_gate_note": (
                    "score+positive-support+cannot-link-point-conflict+source-area-ratio; no reference IoU used"
                ),
            }
        )
        return mask, metrics

    transition_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    visual_paths: list[Path] = []
    inference_start = time.time()

    for event_idx, row in enumerate(cases):
        confirm_row = dict(row["_confirm_row"])
        obj_id = int(row["target_obj_id"])
        source_frame_id = int(row["source_frame_id"])
        attempt_frame_id = int(row["target_frame_id"])
        confirm_frame_id = int(confirm_row["target_frame_id"])
        lag = int(row["source_lag"])
        source_label = get_label(source_frame_id)
        source_area = int(np.count_nonzero(source_label == obj_id))
        attempt_label = get_label(attempt_frame_id)
        confirm_label = get_label(confirm_frame_id)
        attempt_ref = attempt_label == obj_id
        confirm_ref = confirm_label == obj_id

        for variant in ("G1_pos", "G2_pos_neg"):
            transition_rows.append(
                {
                    "schema_version": "stream4d_v107_lifecycle_transition_v1",
                    "event_index": event_idx,
                    "prompt_variant": variant,
                    "global_id": obj_id,
                    "from_state": "ACTIVE_SAM",
                    "to_state": "DORMANT_GEO",
                    "frame_id": source_frame_id,
                    "reason": "phase8_pilot_forced_demotion_after_capsule_snapshot",
                    "sam2_runtime_id_removed": True,
                }
            )
            transition_rows.append(
                {
                    "schema_version": "stream4d_v107_lifecycle_transition_v1",
                    "event_index": event_idx,
                    "prompt_variant": variant,
                    "global_id": obj_id,
                    "from_state": "DORMANT_GEO",
                    "to_state": "REACTIVATING",
                    "frame_id": attempt_frame_id,
                    "reason": "lingbot_visible_prompt_packet_available",
                    "sam2_runtime_id_removed": False,
                }
            )

        attempt_rgb = set_image(attempt_frame_id)
        attempt_prompt = build_prompt(row, attempt_rgb)
        attempt_masks: dict[str, np.ndarray] = {}
        attempt_metrics: dict[str, dict[str, Any]] = {}
        for variant in ("G1_pos", "G2_pos_neg"):
            mask, metrics = predict_variant(
                variant=variant,
                prompt=attempt_prompt,
                ref_mask=attempt_ref,
                target_label=attempt_label,
                source_area=source_area,
            )
            attempt_masks[variant] = mask
            attempt_metrics[variant] = metrics

        confirm_rgb = set_image(confirm_frame_id)
        confirm_prompt = build_prompt(confirm_row, confirm_rgb)
        confirm_masks: dict[str, np.ndarray] = {}
        confirm_metrics: dict[str, dict[str, Any]] = {}
        for variant in ("G1_pos", "G2_pos_neg"):
            mask, metrics = predict_variant(
                variant=variant,
                prompt=confirm_prompt,
                ref_mask=confirm_ref,
                target_label=confirm_label,
                source_area=source_area,
            )
            confirm_masks[variant] = mask
            confirm_metrics[variant] = metrics

        for variant in ("G1_pos", "G2_pos_neg"):
            attempt_area = int(attempt_metrics[variant]["mask_area_px"])
            confirm_area = int(confirm_metrics[variant]["mask_area_px"])
            temporal_area_ratio = float(min(attempt_area, confirm_area) / max(max(attempt_area, confirm_area), 1))
            two_frame_confirmed = bool(
                attempt_metrics[variant]["online_gate_pass"]
                and confirm_metrics[variant]["online_gate_pass"]
                and float(args.min_temporal_area_ratio) <= temporal_area_ratio <= float(args.max_temporal_area_ratio)
            )
            temporal_iou_mean = float(
                (float(attempt_metrics[variant]["iou_to_reference"]) + float(confirm_metrics[variant]["iou_to_reference"]))
                / 2.0
            )
            merge_like = bool(
                max(
                    float(attempt_metrics[variant]["negative_sibling_overlap_rate"]),
                    float(confirm_metrics[variant]["negative_sibling_overlap_rate"]),
                )
                > float(args.merge_overlap_threshold)
            )
            transition_rows.append(
                {
                    "schema_version": "stream4d_v107_lifecycle_transition_v1",
                    "event_index": event_idx,
                    "prompt_variant": variant,
                    "global_id": obj_id,
                    "from_state": "REACTIVATING",
                    "to_state": "ACTIVE_SAM" if two_frame_confirmed else "DORMANT_GEO",
                    "frame_id": confirm_frame_id,
                    "reason": "two_frame_online_confirmation_pass" if two_frame_confirmed else "two_frame_online_confirmation_fail",
                    "sam2_runtime_id_removed": not two_frame_confirmed,
                }
            )
            attempt_payload = {
                "schema_version": "stream4d_v107_reactivation_attempt_v1",
                "event_index": event_idx,
                "global_id": obj_id,
                "source_frame_id": source_frame_id,
                "frame_id": attempt_frame_id,
                "source_lag": lag,
                "prompt_variant": variant,
                "positive_points": int(attempt_prompt["positive_count"]),
                "negative_points": int(attempt_prompt["negative_count"] if variant == "G2_pos_neg" else 0),
                "negative_candidate_points_available": int(attempt_prompt["negative_count"]),
                "first_frame_iou_to_reference": float(attempt_metrics[variant]["iou_to_reference"]),
                "confirmed_after_two_frames": two_frame_confirmed,
                **flatten_variant("attempt", attempt_metrics[variant]),
            }
            confirm_payload = {
                "schema_version": "stream4d_v107_reactivation_confirmation_v1",
                "event_index": event_idx,
                "global_id": obj_id,
                "source_frame_id": source_frame_id,
                "attempt_frame_id": attempt_frame_id,
                "confirm_frame_id": confirm_frame_id,
                "source_lag": lag,
                "prompt_variant": variant,
                "confirm_iou_to_reference": float(confirm_metrics[variant]["iou_to_reference"]),
                "temporal_iou_mean": temporal_iou_mean,
                "temporal_area_ratio_min_over_max": temporal_area_ratio,
                "confirmed_after_two_frames": two_frame_confirmed,
                "merge_like_negative_overlap": merge_like,
                **flatten_variant("confirm", confirm_metrics[variant]),
            }
            metric_payload = {
                "schema_version": "stream4d_v107_reactivation_metric_record_v1",
                "event_index": event_idx,
                "global_id": obj_id,
                "source_lag": lag,
                "prompt_variant": variant,
                "first_frame_iou_to_reference": float(attempt_metrics[variant]["iou_to_reference"]),
                "confirm_frame_iou_to_reference": float(confirm_metrics[variant]["iou_to_reference"]),
                "temporal_iou_mean": temporal_iou_mean,
                "reactivation_success_iou_0_3": bool(two_frame_confirmed and temporal_iou_mean >= 0.3),
                "reactivation_success_iou_0_5": bool(two_frame_confirmed and temporal_iou_mean >= 0.5),
                "reactivation_success_iou_0_7": bool(two_frame_confirmed and temporal_iou_mean >= 0.7),
                "wrong_sibling_activation": merge_like,
                "global_id_retained_by_online_gate": two_frame_confirmed,
                "uses_reference_iou_for_online_gate": False,
            }
            attempt_rows.append(attempt_payload)
            confirmation_rows.append(confirm_payload)
            metric_rows.append(metric_payload)

        if len(visual_paths) < int(args.max_visual_events):
            attempt_coords = np.asarray(attempt_prompt["coords_pos"] + attempt_prompt["coords_neg"], dtype=np.float32)
            attempt_labels = np.asarray(
                [1] * int(attempt_prompt["positive_count"]) + [0] * int(attempt_prompt["negative_count"]),
                dtype=np.int32,
            )
            confirm_coords = np.asarray(confirm_prompt["coords_pos"] + confirm_prompt["coords_neg"], dtype=np.float32)
            confirm_labels = np.asarray(
                [1] * int(confirm_prompt["positive_count"]) + [0] * int(confirm_prompt["negative_count"]),
                dtype=np.int32,
            )
            vis = draw_event(
                event_id=event_idx,
                obj_id=obj_id,
                lag=lag,
                attempt_frame_id=attempt_frame_id,
                confirm_frame_id=confirm_frame_id,
                attempt_rgb=attempt_rgb,
                confirm_rgb=confirm_rgb,
                attempt_ref=attempt_ref,
                confirm_ref=confirm_ref,
                attempt_coords=attempt_coords,
                attempt_labels=attempt_labels,
                confirm_coords=confirm_coords,
                confirm_labels=confirm_labels,
                attempt_masks=attempt_masks,
                confirm_masks=confirm_masks,
                attempt_iou={k: float(v["iou_to_reference"]) for k, v in attempt_metrics.items()},
                confirm_iou={k: float(v["iou_to_reference"]) for k, v in confirm_metrics.items()},
            )
            vis_path = output_root / "visual_overlays" / (
                f"event{event_idx:03d}_obj{obj_id:04d}_lag{lag:02d}_f{attempt_frame_id:06d}_to_{confirm_frame_id:06d}.jpg"
            )
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            visual_paths.append(vis_path)

    inference_runtime = float(time.time() - inference_start)
    attempt_path = output_root / "reactivation_attempt_records.jsonl"
    confirmation_path = output_root / "reactivation_confirmation_records.jsonl"
    transition_path = output_root / "lifecycle_transition_records.jsonl"
    metric_path = output_root / "reactivation_metric_records.jsonl"
    metric_csv = output_root / "reactivation_metric_records.csv"
    write_jsonl(attempt_path, attempt_rows)
    write_jsonl(confirmation_path, confirmation_rows)
    write_jsonl(transition_path, transition_rows)
    write_jsonl(metric_path, metric_rows)
    write_csv(metric_csv, metric_rows)
    sheet = contact_sheet(visual_paths, output_root / "phase8_live_reactivation_pilot_contact_sheet.jpg", cols=1)

    def aggregate(rows_in: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for variant in ("G1_pos", "G2_pos_neg"):
            vals = [r for r in rows_in if r["prompt_variant"] == variant]
            if not vals:
                out[variant] = {"event_count": 0}
                continue
            confirmed = [r for r in vals if bool(r["global_id_retained_by_online_gate"])]
            confirmed_iou50 = [r for r in vals if bool(r["reactivation_success_iou_0_5"])]
            confirmed_iou70 = [r for r in vals if bool(r["reactivation_success_iou_0_7"])]
            out[variant] = {
                "event_count": int(len(vals)),
                "online_confirmed_count": int(len(confirmed)),
                "online_confirmation_rate": float(len(confirmed) / max(len(vals), 1)),
                "reactivation_success_iou_0_3_rate": float(
                    sum(bool(r["reactivation_success_iou_0_3"]) for r in vals) / max(len(vals), 1)
                ),
                "reactivation_success_iou_0_5_rate": float(len(confirmed_iou50) / max(len(vals), 1)),
                "reactivation_success_iou_0_7_rate": float(len(confirmed_iou70) / max(len(vals), 1)),
                "confirmed_iou_0_5_precision": float(len(confirmed_iou50) / max(len(confirmed), 1)),
                "mean_first_frame_iou": float(np.mean([float(r["first_frame_iou_to_reference"]) for r in vals])),
                "mean_confirm_frame_iou": float(np.mean([float(r["confirm_frame_iou_to_reference"]) for r in vals])),
                "mean_temporal_iou": float(np.mean([float(r["temporal_iou_mean"]) for r in vals])),
                "wrong_sibling_activation_rate": float(
                    sum(bool(r["wrong_sibling_activation"]) for r in vals) / max(len(vals), 1)
                ),
            }
        return out

    by_lag: dict[str, Any] = {}
    for lag in sorted({int(r["source_lag"]) for r in metric_rows}):
        by_lag[str(lag)] = aggregate([r for r in metric_rows if int(r["source_lag"]) == lag])
    aggregate_summary = aggregate(metric_rows)
    g1 = aggregate_summary.get("G1_pos", {})
    g2 = aggregate_summary.get("G2_pos_neg", {})
    summary = {
        "schema_version": "stream4d_v107_phase8_live_reactivation_pilot_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "sam2_inference_runtime_sec": inference_runtime,
        "prompt_probe_root": rel(prompt_root),
        "reference_run_root": rel(reference_root),
        "scene_root": rel(scene_root),
        "event_count": int(len(cases)),
        "variant_count": 2,
        "prompt_probe_summary_sha256": sha256_file(prompt_root / "prompt_capsule_visibility_probe_summary.json"),
        "projection_geometry_source": str(probe_summary.get("projection_geometry_source", "")),
        "uses_scannet_pose_or_depth_for_projection": False,
        "online_gate_uses_reference_iou": False,
        "evaluation_uses_reference_labels": True,
        "phase8_status": "PILOT_NOT_FULL_LIVE_SAM2_MEMORY_INTEGRATION",
        "attempt_source_lags": sorted(parse_int_set(str(args.attempt_source_lags))),
        "confirm_frame_step": int(args.confirm_frame_step),
        "sam2_checkpoint": rel(sam2_checkpoint),
        "sam2_checkpoint_sha256": sha256_file(sam2_checkpoint),
        "sam2_model_cfg": str(args.sam2_model_cfg),
        "device": str(args.device),
        "model_dtype": str(args.model_dtype),
        "thresholds": {
            "min_sam2_score": float(args.min_sam2_score),
            "min_positive_support_rate": float(args.min_positive_support_rate),
            "max_negative_conflict_rate": float(args.max_negative_conflict_rate),
            "min_area_ratio_to_source": float(args.min_area_ratio_to_source),
            "max_area_ratio_to_source": float(args.max_area_ratio_to_source),
            "min_temporal_area_ratio": float(args.min_temporal_area_ratio),
            "max_temporal_area_ratio": float(args.max_temporal_area_ratio),
            "merge_overlap_threshold": float(args.merge_overlap_threshold),
        },
        "aggregate": aggregate_summary,
        "by_source_lag": by_lag,
        "G2_minus_G1": {
            "mean_temporal_iou": float(g2.get("mean_temporal_iou", 0.0) - g1.get("mean_temporal_iou", 0.0)),
            "online_confirmation_rate": float(
                g2.get("online_confirmation_rate", 0.0) - g1.get("online_confirmation_rate", 0.0)
            ),
            "reactivation_success_iou_0_5_rate": float(
                g2.get("reactivation_success_iou_0_5_rate", 0.0) - g1.get("reactivation_success_iou_0_5_rate", 0.0)
            ),
            "wrong_sibling_activation_rate": float(
                g2.get("wrong_sibling_activation_rate", 0.0) - g1.get("wrong_sibling_activation_rate", 0.0)
            ),
        },
        "artifacts": {
            "lifecycle_transition_records": rel(transition_path),
            "lifecycle_transition_records_sha256": sha256_file(transition_path),
            "reactivation_attempt_records": rel(attempt_path),
            "reactivation_attempt_records_sha256": sha256_file(attempt_path),
            "reactivation_confirmation_records": rel(confirmation_path),
            "reactivation_confirmation_records_sha256": sha256_file(confirmation_path),
            "reactivation_metric_records": rel(metric_path),
            "reactivation_metric_records_sha256": sha256_file(metric_path),
            "reactivation_metric_records_csv": rel(metric_csv),
            "reactivation_metric_records_csv_sha256": sha256_file(metric_csv),
            "visual_overlay_count": int(len(visual_paths)),
            "visual_overlays": [rel(path) for path in visual_paths],
            "contact_sheet": rel(sheet) if sheet is not None else "",
            "contact_sheet_sha256": sha256_file(sheet) if sheet is not None else "",
        },
        "audit_note": (
            "This runner replays live-order reactivation attempts with online prompt-consistency gates. "
            "It does not mutate the SAM2 video inference_state or prove final Phase8 L4 scheduler success."
        ),
    }
    summary_path = output_root / "lifecycle_metric_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": rel(summary_path), "event_count": len(cases)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
