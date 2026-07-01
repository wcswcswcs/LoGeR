#!/usr/bin/env python3
"""Materialize a v96 no-temporal Phase5-style control root."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v96_phase7_materialize_no_temporal_control"
RUN_ID = "v96_phase7_materialize_no_temporal_control"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "Open-d4rt", "docs"}:
        return REPO_ROOT / p
    return ROOT / p


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    phase5_root = _project(args.phase5_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in _read_csv(phase5_root / "selected_masklet_rows.csv")
        if row.get("family") == args.source_family
    ]
    selected_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        object_id = f"{args.output_family}_obj_{idx:06d}"
        group_id = (
            f"C3_no_temporal:{row.get('scene_id','')}:{row.get('window_id','')}:"
            f"f{row.get('frame_id','')}:m{row.get('selected_mask_id','')}:src{row.get('object_id','')}"
        )
        masklet_score = float(row.get("masklet_score") or 0.0)
        selected_rows.append(
            {
                **row,
                "family": args.output_family,
                "object_id": object_id,
                "group_id": group_id,
                "selection_status": "control_no_temporal_frame_independent",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        object_rows.append(
            {
                "schema_version": "stream4d_v96_object_candidate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": args.output_family,
                "object_id": object_id,
                "group_id": group_id,
                "micro_query_count": row.get("object_query_count", ""),
                "selected_frame_count_before_collision_resolution": 1,
                "masklet_support_query_count_sum": row.get("masklet_support_query_count", ""),
                "object_score": masklet_score,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "selected_frame_count": 1,
            }
        )
    summary = {
        "schema": "stream4d_v96_no_temporal_control_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "MATERIALIZED_V96_C3_NO_TEMPORAL_CONTROL",
        "source_phase5_root": _rel(phase5_root),
        "output_root": _rel(output_root),
        "source_family": args.source_family,
        "output_family": args.output_family,
        "selected_masklet_count": len(selected_rows),
        "object_count": len(object_rows),
        "control_definition": "Each selected object-frame masklet becomes its own object; no cross-frame temporal identity is preserved.",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(output_root / "selected_masklet_rows.csv", selected_rows)
    _write_csv(output_root / "object_candidate_rows.csv", object_rows)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "object_count": len(object_rows), "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize v96 C3 no-temporal control as a Phase5-style root.")
    parser.add_argument("--phase5-root", default=str(ROOT / "outputs/audit/v96_phase5_object_birth_w0020_segmented_r4_D3_repair5_overlap090_sceneoffset"))
    parser.add_argument("--source-family", default="C_hybrid_cover_cluster")
    parser.add_argument("--output-family", default="C3_no_temporal")
    parser.add_argument("--output-root", default=str(ROOT / "outputs/audit/v96_phase5_control_C3_no_temporal_sceneoffset"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
