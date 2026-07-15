#!/usr/bin/env python3
"""Summarize one-scene v105 baseline chunk table runs.

The script reads per-run JSON records emitted by run_v105_sgq_stream_pipeline,
verifies each selected visualization video decodes to the expected 32 input
frames, and writes a compact JSON/Markdown table plus full-frame sheets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DISPLAY_NAMES = {
    "sam2": "sam2",
    "sam31": "sam3.1",
    "edgetam": "EdgeTAM",
    "cropformer": "Cropformer",
    "fastsam": "FastSAM-x",
    "sam2_video": "sam2",
    "sam31_multiplex": "sam3.1",
    "edgetam_video": "EdgeTAM",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def row_by_variant(record_path: Path, variant_id: str) -> dict[str, Any]:
    table = load_json(record_path)
    rows = table.get("rows", [])
    matches = [r for r in rows if r.get("variant_id") == variant_id]
    if len(matches) != 1:
        raise RuntimeError(f"{record_path} has {len(matches)} rows for {variant_id}")
    return matches[0]


def video_row(record_path: Path, variant_id: str, scene_id: str) -> dict[str, Any]:
    table = load_json(record_path)
    rows = table.get("rows", [])
    matches = [
        r
        for r in rows
        if r.get("variant_id") == variant_id and r.get("scene_id") == scene_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{record_path} has {len(matches)} video rows for {variant_id} {scene_id}"
        )
    return matches[0]


def decode_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def make_sheet(
    frames: list[np.ndarray],
    output_path: Path,
    *,
    title: str,
    cols: int = 8,
    cell_w: int = 300,
    cell_h: int = 210,
) -> None:
    if not frames:
        raise RuntimeError(f"no frames for {output_path}")
    rows = int(np.ceil(len(frames) / cols))
    title_h = 36
    sheet = np.full((rows * cell_h + title_h, cols * cell_w, 3), 245, dtype=np.uint8)
    cv2.putText(
        sheet,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    for idx, frame in enumerate(frames):
        resized = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        y = title_h + (idx // cols) * cell_h
        x = (idx % cols) * cell_w
        sheet[y : y + cell_h, x : x + cell_w] = resized
        cv2.rectangle(sheet, (x, y), (x + cell_w - 1, y + cell_h - 1), (30, 30, 30), 1)
        cv2.putText(
            sheet,
            f"{idx:02d}",
            (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"{idx:02d}",
            (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), sheet)
    if not ok:
        raise RuntimeError(f"failed to write {output_path}")


def format_mb(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f} MB"


def format_sec(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f} s"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(Path(args.manifest))
    output_root = Path(args.output_root)
    sheets_dir = output_root / "frame_sheets"
    scene_id = args.scene_id
    expected_frame_count = int(args.expected_frame_count)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(manifest, start=1):
        variant_id = item["variant_id"]
        run_root = Path(item["output_root"])
        baselines_root = run_root / "baselines"

        latency = row_by_variant(baselines_root / "baseline_latency_records.json", variant_id)
        memory = row_by_variant(baselines_root / "baseline_memory_records.json", variant_id)
        video = video_row(baselines_root / "video_index_records.json", variant_id, scene_id)
        video_path = Path(video["video_path"])
        if not video_path.exists():
            raise RuntimeError(f"missing video for {variant_id}: {video_path}")

        frames, fps = decode_video(video_path)
        decoded_frame_count = len(frames)
        frame_count_ok = decoded_frame_count == expected_frame_count
        sheet_path = sheets_dir / f"{variant_id}_{scene_id}_all32.jpg"
        make_sheet(
            frames,
            sheet_path,
            title=f"baseline-{index} {variant_id} {scene_id} decoded={decoded_frame_count}",
        )

        row = {
            "baseline": f"baseline-{index}",
            "variant_id": variant_id,
            "scene_id": scene_id,
            "chunk": f"stride5_input_frames_0_to_155_{expected_frame_count}f",
            "segmentor": DISPLAY_NAMES.get(item["segmentor"], item["segmentor"]),
            "tracker": DISPLAY_NAMES.get(item["tracker"], item["tracker"]),
            "memory_mb": float(memory["peak_gpu_memory_mb"]),
            "latency_sec": float(latency["latency_sec"]),
            "latency_stage": latency.get("stage"),
            "video_path": str(video_path),
            "video_exists": bool(video.get("video_exists")) and video_path.exists(),
            "decoded_frame_count": decoded_frame_count,
            "expected_frame_count": expected_frame_count,
            "frame_count_ok": frame_count_ok,
            "fps": fps,
            "full_frame_sheet": str(sheet_path),
            "run_root": str(run_root),
        }
        rows.append(row)

    summary = {
        "schema_version": "stream4d_v105_baseline_chunk_table_summary_v1",
        "scene_id": scene_id,
        "expected_frame_count": expected_frame_count,
        "all_runs_completed": True,
        "all_video_frame_counts_ok": all(r["frame_count_ok"] for r in rows),
        "row_count": len(rows),
        "rows": rows,
    }
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v105 Baseline Chunk Table",
        "",
        f"- scene: `{summary['scene_id']}`",
        f"- chunk: `stride=5`, input frames `0,5,...,155`, expected decoded video frames `{summary['expected_frame_count']}`",
        f"- all_video_frame_counts_ok: `{summary['all_video_frame_counts_ok']}`",
        "",
        "|            | segmentor  | tracker | Mem  | Latency (1 chunk) | video path | full-frame sheet |",
        "| ---------- | ---------- | ------- | ---- | ----------------- | ---------- | ---------------- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {baseline} | {segmentor} | {tracker} | {mem} | {latency} | `{video}` | `{sheet}` |".format(
                baseline=row["baseline"],
                segmentor=row["segmentor"],
                tracker=row["tracker"],
                mem=format_mb(row["memory_mb"]),
                latency=format_sec(row["latency_sec"]),
                video=row["video_path"],
                sheet=row["full_frame_sheet"],
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Latency and memory are read from each run's own `baseline_latency_records.json` and `baseline_memory_records.json` row for the listed variant.",
            "- For FastSAM-x rows, the selected row already includes `fastsam_segmentor_plus_*_tracker_plus_v65_eval_plus_video`; `B1_fastsam_only` is not added again.",
            "- Videos were reopened with OpenCV and decoded frame counts were checked against 32 input frames.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--expected-frame-count", type=int, default=32)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary = build_summary(args)
    write_json(output_root / "baseline_chunk_table_summary.json", summary)
    write_markdown(output_root / "baseline_chunk_table.md", summary)
    print(json.dumps({
        "output_root": str(output_root),
        "row_count": summary["row_count"],
        "all_video_frame_counts_ok": summary["all_video_frame_counts_ok"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
