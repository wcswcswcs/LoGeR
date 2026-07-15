#!/usr/bin/env python3
"""Compare two Stream4D label directories through their summary.json records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return ROOT / path


def imread_label(path: Path):
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-root", required=True)
    parser.add_argument("--candidate-run-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    reference_root = Path(args.reference_run_root)
    if not reference_root.is_absolute():
        reference_root = ROOT / reference_root
    candidate_root = Path(args.candidate_run_root)
    if not candidate_root.is_absolute():
        candidate_root = ROOT / candidate_root
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = ROOT / output_json

    ref_summary_path = reference_root / "summary.json"
    cand_summary_path = candidate_root / "summary.json"
    ref_summary = read_json(ref_summary_path)
    cand_summary = read_json(cand_summary_path)
    ref_records = {int(row["frame_id"]): row for row in ref_summary.get("records", [])}
    cand_records = {int(row["frame_id"]): row for row in cand_summary.get("records", [])}

    rows = []
    pixel_mismatch_count = 0
    missing_frames = []
    extra_frames = sorted(set(cand_records) - set(ref_records))
    for frame_id in sorted(ref_records):
        if frame_id not in cand_records:
            missing_frames.append(int(frame_id))
            continue
        ref_label_path = resolve(str(ref_records[frame_id]["label_path"]), reference_root)
        cand_label_path = resolve(str(cand_records[frame_id]["label_path"]), candidate_root)
        ref_label = imread_label(ref_label_path)
        cand_label = imread_label(cand_label_path)
        same_shape = ref_label.shape == cand_label.shape
        exact = bool(same_shape and (ref_label == cand_label).all())
        mismatch = 0 if exact else int((ref_label != cand_label).sum()) if same_shape else -1
        if mismatch > 0:
            pixel_mismatch_count += mismatch
        rows.append(
            {
                "frame_id": int(frame_id),
                "reference_label": rel(ref_label_path),
                "candidate_label": rel(cand_label_path),
                "same_shape": same_shape,
                "pixel_exact_equal": exact,
                "pixel_mismatch_count": mismatch,
                "reference_sha256": sha256_file(ref_label_path),
                "candidate_sha256": sha256_file(cand_label_path),
            }
        )

    exact_frame_count = int(sum(1 for row in rows if row["pixel_exact_equal"]))
    result = {
        "schema_version": "stream4d_v107_label_parity_compare_v1",
        "reference_summary": {"path": rel(ref_summary_path), "sha256": sha256_file(ref_summary_path)},
        "candidate_summary": {"path": rel(cand_summary_path), "sha256": sha256_file(cand_summary_path)},
        "frame_count": int(len(ref_records)),
        "compared_frame_count": int(len(rows)),
        "exact_frame_count": exact_frame_count,
        "missing_frames": missing_frames,
        "extra_frames": [int(v) for v in extra_frames],
        "pixel_mismatch_count": int(pixel_mismatch_count),
        "label_exact_parity_pass": exact_frame_count == len(ref_records)
        and not missing_frames
        and not extra_frames
        and pixel_mismatch_count == 0,
        "bad_frame_count": int(sum(1 for row in rows if not row["pixel_exact_equal"])),
        "rows": rows,
    }
    write_json(output_json, result)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "label_exact_parity_pass": result["label_exact_parity_pass"],
                "bad_frame_count": result["bad_frame_count"],
                "pixel_mismatch_count": result["pixel_mismatch_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["label_exact_parity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
