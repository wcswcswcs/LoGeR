#!/usr/bin/env python3
"""Materialize a v96 shuffled-D4RT identity Phase5-style control root."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PHASE_ID = "v96_phase7_materialize_shuffled_d4rt_control"
RUN_ID = "v96_phase7_materialize_shuffled_d4rt_control"


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
    rng = random.Random(int(args.seed))
    phase5_root = _project(args.phase5_root)
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_rows = [
        row
        for row in _read_csv(phase5_root / "selected_masklet_rows.csv")
        if row.get("family") == args.source_family
    ]
    source_objects = sorted({row.get("object_id", "") for row in source_rows if row.get("object_id", "")})
    object_map = {
        source_id: f"{args.output_family}_obj_{idx:06d}"
        for idx, source_id in enumerate(source_objects, start=1)
    }
    output_objects = list(object_map.values())
    by_frame: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_frame[(row.get("scene_id", ""), row.get("window_id", ""), int(float(row.get("frame_id") or 0)))].append(row)

    selected_rows: list[dict[str, Any]] = []
    assigned_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame_key, rows in sorted(by_frame.items()):
        if len(rows) > len(output_objects):
            raise ValueError(f"frame {frame_key} has more rows than available shuffled objects")
        shuffled_objects = output_objects[:]
        rng.shuffle(shuffled_objects)
        for row, shuffled_object_id in zip(sorted(rows, key=lambda item: (item.get("object_id", ""), item.get("selected_mask_id", ""))), shuffled_objects):
            out = {
                **row,
                "family": args.output_family,
                "object_id": shuffled_object_id,
                "group_id": f"C2_shuffled_D4RT:seed{int(args.seed)}:from:{row.get('object_id','')}",
                "selection_status": "control_shuffled_d4rt_identity_per_frame",
                "shuffle_seed": int(args.seed),
                "source_object_id": row.get("object_id", ""),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            selected_rows.append(out)
            assigned_by_object[shuffled_object_id].append(out)

    object_rows: list[dict[str, Any]] = []
    for object_id, rows in sorted(assigned_by_object.items()):
        support_sum = sum(int(float(row.get("masklet_support_query_count") or 0)) for row in rows)
        q_sum = sum(int(float(row.get("object_query_count") or 0)) for row in rows)
        score_vals = [float(row.get("masklet_score") or 0.0) for row in rows]
        object_rows.append(
            {
                "schema_version": "stream4d_v96_object_candidate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "family": args.output_family,
                "object_id": object_id,
                "group_id": f"C2_shuffled_D4RT:seed{int(args.seed)}:{object_id}",
                "micro_query_count": q_sum,
                "selected_frame_count_before_collision_resolution": len(rows),
                "masklet_support_query_count_sum": support_sum,
                "object_score": sum(score_vals) / max(1, len(score_vals)),
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "selected_frame_count": len(rows),
            }
        )
    summary = {
        "schema": "stream4d_v96_shuffled_d4rt_control_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": _created_at(),
        "decision": "MATERIALIZED_V96_C2_SHUFFLED_D4RT_CONTROL",
        "source_phase5_root": _rel(phase5_root),
        "output_root": _rel(output_root),
        "source_family": args.source_family,
        "output_family": args.output_family,
        "shuffle_seed": int(args.seed),
        "selected_masklet_count": len(selected_rows),
        "object_count": len(object_rows),
        "control_definition": "Object identities are shuffled independently per frame while preserving per-frame one-object-per-masklet assignment.",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }
    _write_csv(output_root / "selected_masklet_rows.csv", selected_rows)
    _write_csv(output_root / "object_candidate_rows.csv", object_rows)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps({"decision": summary["decision"], "object_count": len(object_rows), "output_root": _rel(output_root)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize v96 C2 shuffled-D4RT control as a Phase5-style root.")
    parser.add_argument("--phase5-root", default=str(ROOT / "outputs/audit/v96_phase5_object_birth_w0020_segmented_r4_D3_repair5_overlap090_sceneoffset"))
    parser.add_argument("--source-family", default="C_hybrid_cover_cluster")
    parser.add_argument("--output-family", default="C2_shuffled_D4RT")
    parser.add_argument("--seed", type=int, default=9602)
    parser.add_argument("--output-root", default=str(ROOT / "outputs/audit/v96_phase5_control_C2_shuffled_D4RT_sceneoffset"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
