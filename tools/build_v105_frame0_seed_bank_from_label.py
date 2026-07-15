#!/usr/bin/env python3
"""Build a Phase6 frame0 birth bank from an existing numeric label PNG."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64, copy=False)


def obj_id_for_label(label_id: int, ordinal: int, mode: str) -> int:
    if mode == "zero_based_order":
        return int(ordinal)
    if mode == "label_value":
        return int(label_id)
    if mode == "label_value_minus_one":
        return int(label_id) - 1
    raise ValueError(f"unsupported obj_id_mode={mode}")


def build_bank(
    *,
    scene_id: str,
    frame_id: int,
    label_path: Path,
    output_root: Path,
    obj_id_mode: str,
) -> dict[str, Any]:
    label = read_label(label_path)
    mask_dir = output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    positive_label_ids = sorted(int(v) for v in np.unique(label) if int(v) > 0)
    for ordinal, label_id in enumerate(positive_label_ids):
        obj_id = obj_id_for_label(int(label_id), int(ordinal), str(obj_id_mode))
        if obj_id < 0:
            raise ValueError(f"obj_id_mode={obj_id_mode} produced negative obj_id={obj_id} for label_id={label_id}")
        mask = label == int(label_id)
        area = int(np.count_nonzero(mask))
        if area <= 0:
            continue
        mask_path = mask_dir / f"frame_{int(frame_id):06d}_obj_{int(obj_id):06d}_frame0_seed.png"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        rows.append(
            {
                "scene_id": str(scene_id),
                "chunk_frame_index": 0,
                "frame_id": int(frame_id),
                "obj_id": int(obj_id),
                "source": "frame0_seed",
                "source_label_id": int(label_id),
                "mask_path": str(mask_path),
                "mask_area": int(area),
            }
        )
    if not rows:
        raise ValueError(f"no positive ids found in {label_path}")

    birth_records_path = output_root / "birth_records.json"
    payload = {
        "schema_version": "stream4d_v105_frame0_seed_bank_from_label_v1",
        "scene_id": str(scene_id),
        "frame_ids": [int(frame_id)],
        "source_label_path": str(label_path),
        "source_label_sha256": sha256_file(label_path),
        "obj_id_mode": str(obj_id_mode),
        "row_count": int(len(rows)),
        "rows": rows,
    }
    birth_records_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "schema_version": "stream4d_v105_frame0_seed_bank_from_label_summary_v1",
        "scene_id": str(scene_id),
        "frame_id": int(frame_id),
        "label_path": str(label_path),
        "label_sha256": sha256_file(label_path),
        "obj_id_mode": str(obj_id_mode),
        "birth_records_path": str(birth_records_path),
        "birth_records_sha256": sha256_file(birth_records_path),
        "mask_dir": str(mask_dir),
        "object_count": int(len(rows)),
        "foreground_area": int(np.count_nonzero(label > 0)),
        "object_ids": [int(row["obj_id"]) for row in rows],
        "source_label_ids": [int(row["source_label_id"]) for row in rows],
    }
    summary_path = output_root / "frame0_seed_bank_from_label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--obj-id-mode",
        choices=["zero_based_order", "label_value", "label_value_minus_one"],
        default="zero_based_order",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_bank(
        scene_id=str(args.scene_id),
        frame_id=int(args.frame_id),
        label_path=resolve_path(args.label_path),
        output_root=resolve_path(args.output_root),
        obj_id_mode=str(args.obj_id_mode),
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
