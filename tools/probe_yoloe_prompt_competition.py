#!/usr/bin/env python3
"""Probe YOLOE prompt competition on selected frames."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_video_masklet_front_end import collect_image_paths, prepare_processing_image_paths  # noqa: E402
from loger.pipeline.video_masklet_frontend import YOLOEDetector, canonicalize_label  # noqa: E402


DEFAULT_PROFILES = {
    "base_car_person": "car,person",
    "abstract_else": "car,person,else,other object,background",
    "concrete_static": (
        "car,person,traffic sign,signpost,pole,bollard,delineator post,"
        "roadside post,guardrail,barrier,traffic cone"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe YOLOE prompt competition on selected frames.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--profiles", default="")
    parser.add_argument("--yoloe_model", default="yoloe-26l-seg.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yoloe_batch_size", type=int, default=4)
    parser.add_argument("--yoloe_imgsz", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--min_report_conf", type=float, default=0.15)
    parser.add_argument("--draw", type=int, default=1)
    return parser.parse_args()


def _parse_frames(raw: str) -> List[int]:
    frames: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if part:
            frames.append(int(part))
    return sorted(set(frames))


def _parse_profiles(raw: str) -> Dict[str, List[str]]:
    if not str(raw or "").strip():
        return {name: [item.strip() for item in prompts.split(",") if item.strip()] for name, prompts in DEFAULT_PROFILES.items()}
    out: Dict[str, List[str]] = {}
    for chunk in str(raw).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid profile chunk {chunk!r}; expected name=prompt,prompt")
        name, prompts = chunk.split("=", 1)
        out[name.strip()] = [item.strip() for item in prompts.split(",") if item.strip()]
    if not out:
        raise ValueError("No profiles parsed.")
    return out


def _load_frames(input_video: str, processing_max_side: int, frames: Sequence[int]) -> Tuple[List[str], List[str], List[int]]:
    image_paths, temp_dir = collect_image_paths(input_video, 0, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    image_paths, resize_tmp, _orig_shape, _proc_shape = prepare_processing_image_paths(image_paths, int(processing_max_side))
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    selected = [idx for idx in frames if 0 <= idx < len(image_paths)]
    return [image_paths[idx] for idx in selected], temp_dirs, selected


def _box_area(box: Any) -> float:
    arr = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
    if arr.size < 4:
        return 0.0
    return float(max(0.0, arr[2] - arr[0]) * max(0.0, arr[3] - arr[1]))


def _draw_sheet(
    output_path: Path,
    image_paths: Sequence[str],
    frame_indices: Sequence[int],
    rows: Sequence[Dict[str, Any]],
    profile_names: Sequence[str],
) -> None:
    panels: List[np.ndarray] = []
    row_map: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        row_map.setdefault((str(row["profile"]), int(row["frame_idx"])), []).append(row)
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]
    for profile in profile_names:
        for image_path, frame_idx in zip(image_paths, frame_indices):
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            panel = rgb.copy()
            dets = row_map.get((profile, int(frame_idx)), [])
            for det_idx, det in enumerate(dets):
                x1, y1, x2, y2 = [int(round(float(x))) for x in json.loads(det["box_json"])]
                colour = palette[det_idx % len(palette)]
                cv2.rectangle(panel, (x1, y1), (x2, y2), colour, 2)
                text = f"{det['raw_label']} {float(det['confidence']):.2f}"
                cv2.rectangle(panel, (x1, max(0, y1 - 14)), (min(panel.shape[1] - 1, x1 + 125), y1), (0, 0, 0), -1)
                cv2.putText(panel, text, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            title = f"{profile} f{frame_idx}"
            cv2.putText(panel, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(panel)
    if not panels:
        return
    cols = max(1, len(frame_indices))
    rows_n = int(np.ceil(len(panels) / cols))
    H, W = panels[0].shape[:2]
    sheet = np.zeros((rows_n * H, cols * W, 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        sheet[r * H : (r + 1) * H, c * W : (c + 1) * W] = panel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _parse_frames(args.frames)
    profiles = _parse_profiles(args.profiles)
    image_paths, temp_dirs, selected_frames = _load_frames(args.input_video, int(args.processing_max_side), frames)

    detector = YOLOEDetector(
        model_path=str(args.yoloe_model),
        device=str(args.device),
        batch_size=int(args.yoloe_batch_size),
        imgsz=int(args.yoloe_imgsz),
    )
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"frames": selected_frames, "profiles": {}}
    try:
        for profile_name, prompts in profiles.items():
            detections = detector.detect_batch(
                image_paths,
                thing_prompts=prompts,
                stuff_prompts=[],
                box_threshold=float(args.box_threshold),
                text_threshold=float(args.text_threshold),
            )
            label_counts: Dict[str, int] = {}
            car_person_count = 0
            static_competitor_count = 0
            total_count = 0
            for frame_idx, frame_dets in zip(selected_frames, detections):
                for det in frame_dets:
                    conf = float(det.get("confidence", 0.0))
                    if conf < float(args.min_report_conf):
                        continue
                    raw_label = str(det.get("label", ""))
                    label = canonicalize_label(raw_label)
                    label_counts[label] = label_counts.get(label, 0) + 1
                    if label in {"car", "person"}:
                        car_person_count += 1
                    if label not in {"car", "person", "else", "background", "other object"}:
                        static_competitor_count += 1
                    total_count += 1
                    box = np.asarray(det.get("box"), dtype=np.float32).reshape(-1)[:4].tolist()
                    rows.append(
                        {
                            "profile": profile_name,
                            "frame_idx": int(frame_idx),
                            "raw_label": raw_label,
                            "canonical_label": label,
                            "confidence": conf,
                            "box_area": _box_area(box),
                            "box_json": json.dumps([float(x) for x in box]),
                        }
                    )
            summary["profiles"][profile_name] = {
                "prompts": prompts,
                "reported_detection_count": int(total_count),
                "car_person_count": int(car_person_count),
                "static_competitor_count": int(static_competitor_count),
                "label_counts": label_counts,
            }
    finally:
        detector.release_gpu()
        for temp_dir in temp_dirs:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    fieldnames = ["profile", "frame_idx", "raw_label", "canonical_label", "confidence", "box_area", "box_json"]
    with (out_dir / "detections.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if int(args.draw):
        image_paths, temp_dirs, selected_frames = _load_frames(args.input_video, int(args.processing_max_side), frames)
        try:
            _draw_sheet(out_dir / "prompt_competition_sheet.jpg", image_paths, selected_frames, rows, list(profiles.keys()))
        finally:
            for temp_dir in temp_dirs:
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
