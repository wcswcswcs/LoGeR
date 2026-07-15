from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


FRAME_RE = re.compile(r"frame_(\d+)\.(?:png|jpg|jpeg)$")


@dataclass(frozen=True)
class LabelParityRow:
    frame_id: int
    candidate_label_path: str
    reference_label_path: str
    same_shape: bool
    label_equal: bool
    foreground_equal: bool
    candidate_fg_area_px: int
    reference_fg_area_px: int
    intersection_px: int
    union_px: int
    foreground_iou: float


def frame_id_from_path(path: Path) -> int | None:
    match = FRAME_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def label_files_by_frame(label_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(Path(label_dir).glob("frame_*.png")):
        frame_id = frame_id_from_path(path)
        if frame_id is not None:
            files[frame_id] = path
    return files


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def compare_label_pair(candidate_path: Path, reference_path: Path, frame_id: int) -> LabelParityRow:
    cand = read_label(candidate_path)
    ref = read_label(reference_path)
    same_shape = cand.shape == ref.shape
    if not same_shape:
        raise ValueError(f"shape mismatch for frame {frame_id}: {cand.shape} vs {ref.shape}")
    cand_fg = cand > 0
    ref_fg = ref > 0
    intersection = int(np.logical_and(cand_fg, ref_fg).sum())
    union = int(np.logical_or(cand_fg, ref_fg).sum())
    return LabelParityRow(
        frame_id=int(frame_id),
        candidate_label_path=candidate_path.as_posix(),
        reference_label_path=reference_path.as_posix(),
        same_shape=True,
        label_equal=bool(np.array_equal(cand, ref)),
        foreground_equal=bool(np.array_equal(cand_fg, ref_fg)),
        candidate_fg_area_px=int(cand_fg.sum()),
        reference_fg_area_px=int(ref_fg.sum()),
        intersection_px=intersection,
        union_px=union,
        foreground_iou=float(intersection / union) if union else 1.0,
    )


def compare_label_dirs(
    candidate_dir: Path,
    reference_dir: Path,
    frame_ids: Iterable[int] | None = None,
) -> list[LabelParityRow]:
    cand_files = label_files_by_frame(Path(candidate_dir))
    ref_files = label_files_by_frame(Path(reference_dir))
    if frame_ids is None:
        selected = sorted(set(cand_files) & set(ref_files))
    else:
        selected = [int(frame_id) for frame_id in frame_ids]
    rows: list[LabelParityRow] = []
    for frame_id in selected:
        if frame_id not in cand_files:
            raise FileNotFoundError(f"candidate label missing for frame {frame_id}")
        if frame_id not in ref_files:
            raise FileNotFoundError(f"reference label missing for frame {frame_id}")
        rows.append(compare_label_pair(cand_files[frame_id], ref_files[frame_id], frame_id))
    return rows


def summarize_parity(rows: list[LabelParityRow]) -> dict[str, object]:
    if not rows:
        return {
            "schema_version": "stream4d_v108_phase1_label_parity_summary_v1",
            "frame_count": 0,
            "all_label_equal": False,
            "all_foreground_equal": False,
        }
    return {
        "schema_version": "stream4d_v108_phase1_label_parity_summary_v1",
        "frame_count": len(rows),
        "frame_id_min": min(row.frame_id for row in rows),
        "frame_id_max": max(row.frame_id for row in rows),
        "all_label_equal": all(row.label_equal for row in rows),
        "all_foreground_equal": all(row.foreground_equal for row in rows),
        "foreground_iou_min": min(row.foreground_iou for row in rows),
        "foreground_iou_mean": sum(row.foreground_iou for row in rows) / len(rows),
        "label_mismatch_count": sum(1 for row in rows if not row.label_equal),
        "foreground_mismatch_count": sum(1 for row in rows if not row.foreground_equal),
    }
