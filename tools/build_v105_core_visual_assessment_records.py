#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _image_stats(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"path": _rel(path), "exists": path.exists(), "readable": False}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "path": _rel(path),
        "exists": True,
        "readable": True,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "blank_like": bool(np.std(gray) < 3.0),
    }


def _make_inspection_sheet(frame_paths: list[Path], out_path: Path, thumb_w: int, thumb_h: int, title: str) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumbs: list[np.ndarray] = []
    frame_stats: list[dict[str, Any]] = []
    for idx, path in enumerate(frame_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        frame_stats.append(_image_stats(path))
        if image is None:
            thumb = np.full((thumb_h, thumb_w, 3), 230, dtype=np.uint8)
            cv2.putText(thumb, "missing", (20, thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2, cv2.LINE_AA)
        else:
            thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (170, 34), (0, 0, 0), -1)
        cv2.putText(thumb, f"frame {idx:02d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    cols = 4
    rows = 2
    header_h = 44
    sheet = np.full((header_h + rows * thumb_h, cols * thumb_w, 3), 255, dtype=np.uint8)
    cv2.rectangle(sheet, (0, 0), (cols * thumb_w, header_h), (20, 20, 20), -1)
    cv2.putText(sheet, title[:140], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    for idx, thumb in enumerate(thumbs):
        y = header_h + (idx // cols) * thumb_h
        x = (idx % cols) * thumb_w
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb
    cv2.imwrite(str(out_path), sheet)
    return {
        "inspection_sheet_path": _rel(out_path),
        "inspection_sheet_height": int(sheet.shape[0]),
        "inspection_sheet_width": int(sheet.shape[1]),
        "frame_stats": frame_stats,
        "blank_frame_present_auto": any(bool(s.get("blank_like")) for s in frame_stats),
        "all_frames_readable": all(bool(s.get("readable")) for s in frame_stats),
    }


def _source_rows(root: Path, rel: str, source_kind: str) -> list[dict[str, Any]]:
    audit = _read_json(root / rel)
    rows = audit.get("rows", [])
    if not isinstance(rows, list):
        return []
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        variant = str(row.get("variant_id", ""))
        if source_kind == "local2history" and not variant.endswith("_history_v1"):
            continue
        selected.append(row)
    return selected


def _build_records_for_root(split_name: str, root: Path, output_root: Path, thumb_w: int, thumb_h: int) -> list[dict[str, Any]]:
    specs = [
        ("sgq_local", "sgq_local/full_frame_visual_audit/full_frame_visual_audit.json"),
        ("local2history", "local2history/full_frame_visual_audit/full_frame_visual_audit.json"),
    ]
    records: list[dict[str, Any]] = []
    for source_kind, rel in specs:
        for row in _source_rows(root, rel, source_kind):
            scene_id = str(row.get("scene_id"))
            variant_id = str(row.get("variant_id"))
            frame_dir = Path(str(row.get("frame_dir", "")))
            if not frame_dir.is_absolute():
                frame_dir = Path.cwd() / frame_dir
            groups = row.get("sheet_groups", [])
            if not isinstance(groups, list):
                groups = []
            for group in groups:
                if not isinstance(group, dict):
                    continue
                group_index = int(group.get("sheet_group_index", len(records)))
                start = int(group.get("frame_start_index", 0))
                end = int(group.get("frame_end_index", start))
                frame_paths = [frame_dir / f"frame_{idx:03d}.jpg" for idx in range(start, end + 1)]
                out_path = output_root / split_name / source_kind / scene_id / variant_id / f"frames_{start:03d}_{end:03d}_inspection_4x2.jpg"
                title = f"{split_name} {source_kind} {scene_id} {variant_id} frames {start:03d}-{end:03d}"
                sheet_info = _make_inspection_sheet(frame_paths, out_path, thumb_w, thumb_h, title)
                records.append(
                    {
                        "schema_version": "stream4d_v105_visual_assessment_record_v1",
                        "split_name": split_name,
                        "source_kind": source_kind,
                        "scene_id": scene_id,
                        "variant_id": variant_id,
                        "sheet_group_index": group_index,
                        "frames": list(range(start, end + 1)),
                        "source_sheet_path": group.get("sheet_path"),
                        **sheet_info,
                        "blank_frame_present": bool(sheet_info["blank_frame_present_auto"]),
                        "major_object_disappearance": "manual_pending",
                        "major_id_switch": "manual_pending",
                        "same_object_duplicate_ids": "manual_pending",
                        "wrong_large_background_mask": "manual_pending",
                        "small_or_thin_structure_loss": "manual_pending",
                        "underseg_event": "manual_pending",
                        "overseg_noise_event": "manual_pending",
                        "late_new_object_missed": "manual_pending",
                        "birth_too_early_or_duplicate": "manual_pending",
                        "reconciliation_coverage_loss": "manual_pending",
                        "notes": "Generated larger 4x2 inspection sheet; manual visual verdict pending.",
                        "verdict": "PENDING_MANUAL_REVIEW",
                    }
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build core v105 visual assessment records and readable 4x2 inspection sheets.")
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--holdout-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--thumb-width", type=int, default=480)
    parser.add_argument("--thumb-height", type=int, default=360)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    records = []
    records.extend(_build_records_for_root("dev", Path(args.dev_root), output_root / "inspection_sheets", args.thumb_width, args.thumb_height))
    records.extend(_build_records_for_root("holdout", Path(args.holdout_root), output_root / "inspection_sheets", args.thumb_width, args.thumb_height))
    pending = sum(1 for row in records if row.get("verdict") == "PENDING_MANUAL_REVIEW")
    summary = {
        "schema_version": "stream4d_v105_core_visual_assessment_summary_v1",
        "output_root": _rel(output_root),
        "record_count": len(records),
        "pending_manual_review_count": pending,
        "reviewed_count": len(records) - pending,
        "core_scope": "dev/holdout sgq_local and final local2history only; controls and baselines are excluded from this core manual assessment worklist.",
        "thumb_width": int(args.thumb_width),
        "thumb_height": int(args.thumb_height),
        "records_json": _rel(output_root / "visual_assessment_records.json"),
        "inspection_sheet_root": _rel(output_root / "inspection_sheets"),
    }
    _write_json(output_root / "visual_assessment_records.json", records)
    _write_json(output_root / "visual_assessment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
