#!/usr/bin/env python3
"""Audit foreground fidelity of a v107 G3 rolling smoke against v106 reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v107_phase7_lingbot_sam2_prompt_benchmark import jsonable, rel, sha256_file, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return ROOT / path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int32, copy=False)


def record_map(summary: dict[str, Any], base: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in summary.get("records", []):
        item = dict(row)
        item["label_path_resolved"] = resolve(str(row["label_path"]), base)
        out[int(row["frame_id"])] = item
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(jsonable(row))


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    args = parse_args()
    candidate_summary = Path(args.candidate_summary)
    if not candidate_summary.is_absolute():
        candidate_summary = ROOT / candidate_summary
    reference_summary = Path(args.reference_summary)
    if not reference_summary.is_absolute():
        reference_summary = ROOT / reference_summary
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    cand = read_json(candidate_summary)
    ref = read_json(reference_summary)
    cand_records = record_map(cand, candidate_summary.parent)
    ref_records = record_map(ref, reference_summary.parent)
    frame_ids = sorted(set(cand_records) & set(ref_records))
    if not frame_ids:
        raise RuntimeError("no overlapping frame ids between candidate and reference")

    rows: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        c_label = load_label(Path(cand_records[frame_id]["label_path_resolved"]))
        r_label = load_label(Path(ref_records[frame_id]["label_path_resolved"]))
        if c_label.shape != r_label.shape:
            c_label = cv2.resize(c_label, (r_label.shape[1], r_label.shape[0]), interpolation=cv2.INTER_NEAREST)
        cand_fg = c_label > 0
        ref_fg = r_label > 0
        inter = int(np.count_nonzero(cand_fg & ref_fg))
        cand_area = int(np.count_nonzero(cand_fg))
        ref_area = int(np.count_nonzero(ref_fg))
        union = int(np.count_nonzero(cand_fg | ref_fg))
        rows.append(
            {
                "frame_id": int(frame_id),
                "candidate_label_path": rel(Path(cand_records[frame_id]["label_path_resolved"])),
                "reference_label_path": rel(Path(ref_records[frame_id]["label_path_resolved"])),
                "candidate_fg_area_px": cand_area,
                "reference_fg_area_px": ref_area,
                "intersection_px": inter,
                "union_px": union,
                "foreground_recall": float(inter / max(ref_area, 1)),
                "foreground_precision": float(inter / max(cand_area, 1)),
                "foreground_iou": float(inter / max(union, 1)),
                "candidate_visible_id_count": int(cand_records[frame_id].get("visible_id_count", 0)),
                "reference_visible_id_count": int(ref_records[frame_id].get("visible_id_count", 0)),
            }
        )

    csv_path = output_root / "foreground_fidelity_rows.csv"
    write_csv(csv_path, rows)
    summary = {
        "schema_version": "stream4d_v107_phase8_g3_rolling_fidelity_audit_v1",
        "candidate_summary": rel(candidate_summary),
        "candidate_summary_sha256": sha256_file(candidate_summary),
        "reference_summary": rel(reference_summary),
        "reference_summary_sha256": sha256_file(reference_summary),
        "frame_count": int(len(rows)),
        "frame_ids": [int(row["frame_id"]) for row in rows],
        "foreground_recall_mean": finite_mean([float(row["foreground_recall"]) for row in rows]),
        "foreground_recall_min": float(min(float(row["foreground_recall"]) for row in rows)),
        "foreground_precision_mean": finite_mean([float(row["foreground_precision"]) for row in rows]),
        "foreground_precision_min": float(min(float(row["foreground_precision"]) for row in rows)),
        "foreground_iou_mean": finite_mean([float(row["foreground_iou"]) for row in rows]),
        "foreground_iou_min": float(min(float(row["foreground_iou"]) for row in rows)),
        "candidate_visible_id_count_mean": finite_mean([float(row["candidate_visible_id_count"]) for row in rows]),
        "reference_visible_id_count_mean": finite_mean([float(row["reference_visible_id_count"]) for row in rows]),
        "rows_csv": rel(csv_path),
        "rows_csv_sha256": sha256_file(csv_path),
        "audit_note": (
            "Foreground fidelity is a broad safety check only. It does not prove object identity preservation, "
            "but it prevents target-only reactivation metrics from hiding scene-level label loss."
        ),
    }
    summary_path = output_root / "foreground_fidelity_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), **summary}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
