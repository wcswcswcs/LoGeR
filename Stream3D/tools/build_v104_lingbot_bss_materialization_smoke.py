#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v104_lingbot_map_only_phase2_bss_materialization_smoke"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_BSS_ROOT = (
    REPO_ROOT
    / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
    / "stage1_lingbot_baseline/workspace/kitti_v105_00_01_02_05/00/lingbot_map_stream_default"
)


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_frames(value: str) -> list[int]:
    frames: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        frames.append(int(part))
    return frames


def build(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(STREAM3D_ROOT))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    from stream4d.lingbot_map_stream3d_geometry_adapter import LingBotMapStream3DGeometryAdapter

    t0 = time.time()
    out = Path(args.output_root)
    if not out.is_absolute():
        out = STREAM3D_ROOT / out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    frames = _parse_frames(args.frames)
    materialized_root = out / "materialized_lingbot_geometry"
    adapter = LingBotMapStream3DGeometryAdapter(
        lingbot_root=args.lingbot_root,
        output_root=materialized_root,
        max_points_per_frame=int(args.max_points_per_frame),
        min_confidence=args.min_confidence,
    )
    manifest = adapter.materialize_frames(frames)
    frame_rows = []
    for row in manifest.get("frames", []):
        frame_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_bss_materialization_frame_row_v1",
                "phase_id": PHASE_ID,
                **row,
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
            }
        )
    materialization_pass = bool(frame_rows) and all(int(row.get("num_points", 0)) > 0 for row in frame_rows)
    summary = {
        "schema_version": "stream4d_v104_lingbot_bss_materialization_summary_v1",
        "phase_id": PHASE_ID,
        "materialization_pass": materialization_pass,
        "taxonomy": "LINGBOT_BSS_FRAME_POINTS_MATERIALIZED" if materialization_pass else "LINGBOT_BSS_FRAME_POINTS_NOT_MATERIALIZED",
        "lingbot_root": _rel(Path(args.lingbot_root)),
        "frame_ids": frames,
        "num_frames_materialized": int(manifest.get("num_frames_materialized", 0)),
        "total_points": int(manifest.get("total_points", 0)),
        "stream4d_metric_ready": False,
        "stream4d_metric_note": "BSS frame points materialized only; no mask support or AP/MV_AP metric is produced.",
        "outputs": {
            "geometry_manifest": _rel(materialized_root / "geometry_manifest.json"),
            "frame_rows": _rel(out / "frame_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_csv(out / "frame_rows.csv", frame_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a few saved LingBot BSS frames as Stream4D-ready point files.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--lingbot-root", default=str(DEFAULT_BSS_ROOT))
    parser.add_argument("--frames", default="0,1")
    parser.add_argument("--max-points-per-frame", type=int, default=32)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
