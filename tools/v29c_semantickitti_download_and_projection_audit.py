#!/usr/bin/env python3
"""ACL2 v29C SemanticKITTI download/projection data gate audit.

This audit is deliberately conservative.  It checks only official
SemanticKITTI/KITTI-Odometry style files already present on disk and never
falls back to predicted VideoMasklet semantic outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.gt_semantic_provider import (  # noqa: E402
    SEMANTIC_KITTI_ID_TO_NAME,
    read_kitti_calib,
)


CHUNK_STARTS = {
    6: 174,
    10: 290,
    16: 464,
}

REQUIRED_DOWNLOADS = {
    "data_odometry_velodyne.zip": "https://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip",
    "data_odometry_calib.zip": "https://www.cvlibs.net/download.php?file=data_odometry_calib.zip",
    "data_odometry_labels.zip": "https://semantic-kitti.org/assets/data_odometry_labels.zip",
}


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    text = (text or "").strip()
    if not text:
        return list(default)
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _selected_frame_rows(chunks: Sequence[int], horizons: Sequence[int]) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    for chunk in chunks:
        if chunk not in CHUNK_STARTS:
            raise ValueError(f"Unsupported v29C chunk: {chunk}")
        start = CHUNK_STARTS[chunk]
        for horizon in horizons:
            end = start + 32 + horizon * 29
            for frame in range(start, end):
                rows.append({"chunk": int(chunk), "horizon": int(horizon), "frame": int(frame)})
    return rows


def _file_type(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        result = subprocess.run(
            ["file", "-b", str(path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"file_probe_error: {exc}"


def _download_rows(download_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, url in REQUIRED_DOWNLOADS.items():
        path = download_dir / name
        ftype = _file_type(path)
        rows.append(
            {
                "file": name,
                "url": url,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "file_type": ftype,
                "is_zip_archive": "Zip archive data" in ftype,
                "looks_like_html_or_text": any(token in ftype.lower() for token in ("html", "text", "ascii")),
            }
        )
    return rows


def _image_resolution(image_dir: Path) -> Tuple[Optional[int], Optional[int], str]:
    first = next(iter(sorted(image_dir.glob("*.png"))), None)
    if first is None:
        return None, None, ""
    try:
        from PIL import Image  # type: ignore

        with Image.open(first) as img:
            width, height = img.size
        return int(width), int(height), str(first)
    except Exception:
        return None, None, str(first)


def _calib_parse(calib_path: Path) -> Dict[str, object]:
    status: Dict[str, object] = {
        "calib_path": str(calib_path),
        "calib_exists": calib_path.exists(),
        "calib_parse_pass": False,
        "calib_has_p2": False,
        "calib_has_tr_velo": False,
        "calib_has_r0_rect": False,
        "p2_shape": "",
        "tr_key": "",
        "tr_shape": "",
        "r0_rect_shape": "",
        "r0_rect_assumption": "",
        "calib_error": "",
    }
    if not calib_path.exists():
        status["calib_error"] = "missing_calib"
        return status
    try:
        calib = read_kitti_calib(calib_path)
        p2 = calib.get("P2")
        status["calib_has_p2"] = p2 is not None and tuple(p2.shape) == (3, 4)
        status["p2_shape"] = "x".join(str(x) for x in p2.shape) if p2 is not None else ""
        tr = None
        tr_key = ""
        for key in ("Tr", "Tr_velo_to_cam", "Tr_velo_cam"):
            if key in calib:
                tr = calib[key]
                tr_key = key
                break
        status["tr_key"] = tr_key
        status["calib_has_tr_velo"] = tr is not None and tuple(tr.shape) in {(3, 4), (4, 4)}
        status["tr_shape"] = "x".join(str(x) for x in tr.shape) if tr is not None else ""
        r0 = calib.get("R0_rect")
        status["calib_has_r0_rect"] = r0 is not None
        status["r0_rect_shape"] = "x".join(str(x) for x in r0.shape) if r0 is not None else ""
        status["r0_rect_assumption"] = "" if r0 is not None else "identity_logged_explicitly"
        status["calib_parse_pass"] = bool(status["calib_has_p2"] and status["calib_has_tr_velo"])
    except Exception as exc:
        status["calib_error"] = str(exc)
    return status


def _count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern)) if path.exists() else 0


def _frame_hit_rows(seq_dir: Path, frames: Sequence[int]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    image_dir = seq_dir / "image_2"
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    rows: List[Dict[str, object]] = []
    image_hits = velodyne_hits = label_hits = all_hits = 0
    for frame in frames:
        image = image_dir / f"{frame:06d}.png"
        velo = velodyne_dir / f"{frame:06d}.bin"
        label = labels_dir / f"{frame:06d}.label"
        has_image = image.exists()
        has_velo = velo.exists()
        has_label = label.exists()
        image_hits += int(has_image)
        velodyne_hits += int(has_velo)
        label_hits += int(has_label)
        hit = has_image and has_velo and has_label
        all_hits += int(hit)
        rows.append(
            {
                "frame": int(frame),
                "image_exists": has_image,
                "velodyne_exists": has_velo,
                "label_exists": has_label,
                "projection_frame_hit": hit,
                "image_path": str(image),
                "velodyne_path": str(velo),
                "label_path": str(label),
            }
        )
    total = max(1, len(frames))
    summary = {
        "selected_frames_expected": len(frames),
        "image_frames_hit": image_hits,
        "velodyne_frames_hit": velodyne_hits,
        "label_frames_hit": label_hits,
        "projection_frames_hit": all_hits,
        "image_hit_rate": image_hits / total,
        "velodyne_hit_rate": velodyne_hits / total,
        "labels_hit_rate": label_hits / total,
        "projection_frame_hit_rate": all_hits / total,
    }
    return rows, summary


def _point_count_rows(seq_dir: Path, frames: Sequence[int], *, max_rows: int = 0) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    rows: List[Dict[str, object]] = []
    checked = 0
    mismatched = 0
    available = 0
    frame_iter = list(frames)
    if max_rows > 0:
        frame_iter = frame_iter[:max_rows]
    for frame in frame_iter:
        velo = velodyne_dir / f"{frame:06d}.bin"
        label = labels_dir / f"{frame:06d}.label"
        row: Dict[str, object] = {
            "frame": int(frame),
            "velodyne_exists": velo.exists(),
            "label_exists": label.exists(),
            "num_points": "",
            "num_labels": "",
            "count_match": False,
            "unique_semantic_ids": "",
            "unique_semantic_names": "",
            "error": "",
        }
        if not velo.exists() or not label.exists():
            row["error"] = "missing_velodyne_or_label"
            rows.append(row)
            continue
        available += 1
        try:
            raw_points = np.fromfile(velo, dtype=np.float32)
            if raw_points.size % 4 != 0:
                raise ValueError(f"velodyne_float_count_not_divisible_by_4:{raw_points.size}")
            points = raw_points.reshape(-1, 4)
            labels = np.fromfile(label, dtype=np.uint32)
            semantic_ids = (labels & np.uint32(0xFFFF)).astype(np.int64)
            unique_ids = sorted(int(x) for x in np.unique(semantic_ids))
            row["num_points"] = int(points.shape[0])
            row["num_labels"] = int(labels.shape[0])
            row["count_match"] = bool(points.shape[0] == labels.shape[0])
            row["unique_semantic_ids"] = " ".join(str(x) for x in unique_ids)
            row["unique_semantic_names"] = " ".join(SEMANTIC_KITTI_ID_TO_NAME.get(x, f"id{x}") for x in unique_ids)
            checked += 1
            mismatched += int(not row["count_match"])
        except Exception as exc:
            row["error"] = str(exc)
            mismatched += 1
        rows.append(row)
    mismatch_rate = mismatched / max(1, checked)
    summary = {
        "point_label_frames_available": available,
        "point_label_frames_checked": checked,
        "point_label_count_mismatch_frames": mismatched,
        "point_label_count_mismatch_rate": mismatch_rate,
        "point_label_count_gate_pass": bool(checked > 0 and mismatch_rate <= 0.01),
        "point_label_count_audit_limited_to_rows": max_rows if max_rows > 0 else "all",
    }
    return rows, summary


def _chunk_window_rows(
    frame_rows: Sequence[Dict[str, int]],
    frame_hits: Dict[int, Dict[str, object]],
    chunks: Sequence[int],
    horizons: Sequence[int],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for chunk in chunks:
        for horizon in horizons:
            frames = [
                int(row["frame"])
                for row in frame_rows
                if int(row["chunk"]) == int(chunk) and int(row["horizon"]) == int(horizon)
            ]
            hits = [frame_hits[f] for f in frames]
            total = max(1, len(hits))
            out.append(
                {
                    "chunk": int(chunk),
                    "horizon": int(horizon),
                    "start_frame": min(frames),
                    "end_frame_exclusive": max(frames) + 1,
                    "frames_expected": len(frames),
                    "image_frames_hit": sum(int(bool(row["image_exists"])) for row in hits),
                    "velodyne_frames_hit": sum(int(bool(row["velodyne_exists"])) for row in hits),
                    "label_frames_hit": sum(int(bool(row["label_exists"])) for row in hits),
                    "projection_frames_hit": sum(int(bool(row["projection_frame_hit"])) for row in hits),
                    "projection_frame_hit_rate": sum(int(bool(row["projection_frame_hit"])) for row in hits) / total,
                    "first_missing_frame": next(
                        (int(row["frame"]) for row in hits if not bool(row["projection_frame_hit"])),
                        "",
                    ),
                }
            )
    return out


def _write_manual_download_request(root: Path, download_rows: Sequence[Dict[str, object]], reason: str) -> Path:
    path = root / "MANUAL_DOWNLOAD_REQUEST.md"
    lines = [
        "# ACL2 v29C Manual Download Request",
        "",
        f"Reason: `{reason}`",
        "",
        "Please manually download the official files below and place them in:",
        "",
        f"`{root / 'downloads'}`",
        "",
        "| File | Official URL | Current status |",
        "|---|---|---|",
    ]
    for row in download_rows:
        status = "present zip" if row.get("is_zip_archive") else ("present non-zip" if row.get("exists") else "missing")
        lines.append(f"| `{row['file']}` | `{row['url']}` | `{status}` |")
    lines.extend(
        [
            "",
            "After placing the files, run:",
            "",
            "```bash",
            "file /mnt/data/users/chengshun.wang/data/semantickitti_odometry/downloads/data_odometry_velodyne.zip",
            "file /mnt/data/users/chengshun.wang/data/semantickitti_odometry/downloads/data_odometry_calib.zip",
            "file /mnt/data/users/chengshun.wang/data/semantickitti_odometry/downloads/data_odometry_labels.zip",
            "```",
            "",
            "All three must report `Zip archive data` before v29C projection may continue.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--semkitti-root", default=os.environ.get("SEMKITTI_ROOT", "/mnt/data/users/chengshun.wang/data/semantickitti_odometry"))
    parser.add_argument("--sequence", default="01")
    parser.add_argument("--results-root", default="results/kitti01_hmc_v2/acl2_v29c_semantickitti_download_projection_videomasklet")
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizons", default="10,15")
    parser.add_argument("--point-count-max-rows", type=int, default=0)
    parser.add_argument("--write-manual-request", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sem_root = Path(args.semkitti_root).resolve()
    results = Path(args.results_root)
    if not results.is_absolute():
        results = repo / results
    out_dir = results / "implementation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    download_dir = sem_root / "downloads"
    seq_dir = sem_root / "dataset" / "sequences" / str(args.sequence)
    image_dir = seq_dir / "image_2"
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"
    calib_path = seq_dir / "calib.txt"

    chunks = _parse_int_list(args.chunks, [6, 10, 16])
    horizons = _parse_int_list(args.horizons, [10, 15])
    selected_rows = _selected_frame_rows(chunks, horizons)
    unique_frames = sorted({int(row["frame"]) for row in selected_rows})

    download_rows = _download_rows(download_dir)
    width, height, first_image = _image_resolution(image_dir)
    calib_status = _calib_parse(calib_path)
    frame_rows, frame_summary = _frame_hit_rows(seq_dir, unique_frames)
    frame_hit_by_id = {int(row["frame"]): row for row in frame_rows}
    chunk_rows = _chunk_window_rows(selected_rows, frame_hit_by_id, chunks, horizons)
    point_rows, point_summary = _point_count_rows(seq_dir, unique_frames, max_rows=max(0, int(args.point_count_max_rows)))

    file_count_rows = [
        {
            "sequence": str(args.sequence),
            "sequence_dir": str(seq_dir),
            "image_2_exists": image_dir.exists(),
            "velodyne_exists": velodyne_dir.exists(),
            "labels_exists": labels_dir.exists(),
            "calib_exists": calib_path.exists(),
            "image_2_count": _count_files(image_dir, "*.png"),
            "velodyne_count": _count_files(velodyne_dir, "*.bin"),
            "labels_count": _count_files(labels_dir, "*.label"),
            "image_width": width or "",
            "image_height": height or "",
            "first_image": first_image,
        }
    ]

    downloads_gate_pass = all(bool(row.get("is_zip_archive")) for row in download_rows)
    data_gate_pass = bool(
        frame_summary["projection_frame_hit_rate"] >= 0.95
        and frame_summary["velodyne_hit_rate"] >= 0.95
        and frame_summary["labels_hit_rate"] >= 0.95
        and bool(calib_status["calib_parse_pass"])
        and bool(point_summary["point_label_count_gate_pass"])
    )

    blocker_reason = ""
    if not downloads_gate_pass:
        blocker_reason = "official_download_zips_missing_or_non_zip"
    elif not bool(calib_status["calib_parse_pass"]):
        blocker_reason = "calib_missing_p2_or_tr_velo"
    elif frame_summary["projection_frame_hit_rate"] < 0.95:
        blocker_reason = "sequence01_selected_frames_incomplete"
    elif not bool(point_summary["point_label_count_gate_pass"]):
        blocker_reason = "point_label_count_mismatch_or_unchecked"

    if args.write_manual_request and not downloads_gate_pass:
        manual_path = _write_manual_download_request(sem_root, download_rows, blocker_reason)
    else:
        manual_path = None

    _write_csv(out_dir / "download_file_type_audit.csv", download_rows)
    _write_csv(out_dir / "sequence01_file_counts.csv", file_count_rows)
    _write_json(out_dir / "calib_parse_report.json", calib_status)
    _write_csv(out_dir / "frame_hit_audit.csv", frame_rows)
    _write_csv(out_dir / "frame_window_hit_audit.csv", chunk_rows)
    _write_csv(out_dir / "label_velodyne_point_count_audit.csv", point_rows)

    summary: Dict[str, object] = {
        "phase": "v29c_phase0_download_projection_data_gate",
        "semkitti_root": str(sem_root),
        "download_dir": str(download_dir),
        "sequence_dir": str(seq_dir),
        "sequence": str(args.sequence),
        "chunks": chunks,
        "horizons": horizons,
        "selected_unique_frames_expected": len(unique_frames),
        "downloads_gate_pass": downloads_gate_pass,
        "phase0_data_gate_pass": data_gate_pass,
        "projection_frame_hit_rate": frame_summary["projection_frame_hit_rate"],
        "projection_frames_hit": frame_summary["projection_frames_hit"],
        "velodyne_hit_rate": frame_summary["velodyne_hit_rate"],
        "labels_hit_rate": frame_summary["labels_hit_rate"],
        "image_hit_rate": frame_summary["image_hit_rate"],
        "calib_parse_pass": calib_status["calib_parse_pass"],
        "calib_has_p2": calib_status["calib_has_p2"],
        "calib_has_tr_velo": calib_status["calib_has_tr_velo"],
        "point_label_count_gate_pass": point_summary["point_label_count_gate_pass"],
        "point_label_frames_checked": point_summary["point_label_frames_checked"],
        "point_label_count_mismatch_rate": point_summary["point_label_count_mismatch_rate"],
        "image_width": width,
        "image_height": height,
        "no_predicted_fallback_flag": True,
        "projection_cache_allowed": data_gate_pass,
        "masklet_3d_alignment_allowed": data_gate_pass,
        "phase2_rollout_allowed": False,
        "selector_allowed": False,
        "full_online_validation_allowed": False,
        "counts_as_deployable_online_success": False,
        "blocked_reason": blocker_reason,
        "manual_download_request": str(manual_path or ""),
        "required_downloads": REQUIRED_DOWNLOADS,
    }
    _write_json(out_dir / "download_audit_summary.json", summary)

    report_lines = [
        "# ACL2 v29C SemanticKITTI Download / Projection Data Gate",
        "",
        f"semkitti_root: `{sem_root}`",
        f"sequence_dir: `{seq_dir}`",
        f"selected_unique_frames_expected: `{len(unique_frames)}`",
        "",
        "## Gate",
        "",
        f"downloads_gate_pass = `{str(downloads_gate_pass).lower()}`",
        f"phase0_data_gate_pass = `{str(data_gate_pass).lower()}`",
        f"projection_frame_hit_rate = `{frame_summary['projection_frame_hit_rate']:.10f}`",
        f"velodyne_hit_rate = `{frame_summary['velodyne_hit_rate']:.10f}`",
        f"labels_hit_rate = `{frame_summary['labels_hit_rate']:.10f}`",
        f"calib_parse_pass = `{str(bool(calib_status['calib_parse_pass'])).lower()}`",
        f"point_label_count_gate_pass = `{str(bool(point_summary['point_label_count_gate_pass'])).lower()}`",
        f"blocked_reason = `{blocker_reason}`",
        "",
        "## Download Files",
        "",
        "| File | Exists | Zip archive | Size bytes | File type |",
        "|---|---:|---:|---:|---|",
    ]
    for row in download_rows:
        report_lines.append(
            f"| `{row['file']}` | `{str(bool(row['exists'])).lower()}` | "
            f"`{str(bool(row['is_zip_archive'])).lower()}` | `{row['size_bytes']}` | `{row['file_type']}` |"
        )
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if data_gate_pass:
        report_lines.append("Phase 0 data gate passed. Projection cache generation is allowed next.")
    else:
        report_lines.append(
            "Phase 0 data gate failed. No projection cache, masklet-3D alignment, candidate rollout, selector, or full online validation is allowed."
        )
        if manual_path is not None:
            report_lines.append(f"Manual download request written to `{manual_path}`.")
    (out_dir / "download_audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return 0 if data_gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
