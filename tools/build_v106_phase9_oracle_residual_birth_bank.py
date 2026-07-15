#!/usr/bin/env python3
"""Build an oracle residual birth bank for diagnosing v106 handoff coverage loss.

This is not a valid v106 method artifact: it copies residual births from an
independent reference run to test whether the foreground loss is caused by an
insufficient inherited birth bank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Stream3D.stream4d_v106.artifacts import sha256_file, write_json  # noqa: E402


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _rel(path: str | Path) -> str:
    path = _resolve(path)
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inherited-birth-records", required=True)
    parser.add_argument("--reference-birth-records", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference-obj-id-offset", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inherited_path = _resolve(args.inherited_birth_records)
    reference_path = _resolve(args.reference_birth_records)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    inherited = _read_json(inherited_path)
    reference = _read_json(reference_path)
    inherited_rows = [dict(row) for row in inherited.get("rows", [])]
    reference_rows = [dict(row) for row in reference.get("rows", [])]
    if inherited.get("frame_ids") != reference.get("frame_ids"):
        raise ValueError(
            f"frame_ids mismatch: inherited={inherited.get('frame_ids')} reference={reference.get('frame_ids')}"
        )
    scene_id = str(inherited.get("scene_id"))
    if str(reference.get("scene_id")) != scene_id:
        raise ValueError(f"scene mismatch: inherited={scene_id} reference={reference.get('scene_id')}")

    max_obj_id = max([int(row["obj_id"]) for row in inherited_rows], default=-1)
    offset = int(args.reference_obj_id_offset)
    if offset < 0:
        offset = max_obj_id + 1

    merged_rows: list[dict[str, Any]] = []
    for row in inherited_rows:
        copied = dict(row)
        copied["source"] = str(copied.get("source", "inherited")) + "|diagnostic_inherited_component"
        copied["phase5_role"] = "inherited"
        merged_rows.append(copied)
    for row in reference_rows:
        copied = dict(row)
        original_obj_id = int(copied["obj_id"])
        copied["obj_id"] = int(offset + original_obj_id)
        copied["global_id"] = int(copied["obj_id"]) + 1
        copied["source"] = str(copied.get("source", "reference_birth")) + "|diagnostic_reference_residual_oracle"
        copied["phase5_role"] = "birth_new"
        copied["diagnostic_original_reference_obj_id"] = int(original_obj_id)
        copied["diagnostic_reference_birth_records"] = _rel(reference_path)
        merged_rows.append(copied)

    merged_rows.sort(key=lambda row: (int(row["chunk_frame_index"]), int(row["obj_id"])))
    payload = {
        "schema_version": "stream4d_v106_phase9_oracle_residual_birth_bank_v1",
        "valid_v106_method": False,
        "diagnostic_only": True,
        "diagnostic_purpose": "upper-bound test for inherited handoff coverage loss",
        "scene_id": scene_id,
        "frame_ids": [int(v) for v in inherited.get("frame_ids", [])],
        "inherited_birth_records": _rel(inherited_path),
        "inherited_birth_records_sha256": sha256_file(inherited_path),
        "reference_birth_records": _rel(reference_path),
        "reference_birth_records_sha256": sha256_file(reference_path),
        "reference_obj_id_offset": int(offset),
        "inherited_row_count": len(inherited_rows),
        "reference_residual_row_count": len(reference_rows),
        "row_count": len(merged_rows),
        "rows": merged_rows,
    }
    out_path = output_root / "oracle_residual_birth_records.json"
    write_json(out_path, payload)
    summary = {
        "schema_version": "stream4d_v106_phase9_oracle_residual_birth_bank_summary_v1",
        "valid_v106_method": False,
        "birth_records": _rel(out_path),
        "birth_records_sha256": sha256_file(out_path),
        "scene_id": scene_id,
        "frame_count": len(payload["frame_ids"]),
        "inherited_row_count": len(inherited_rows),
        "reference_residual_row_count": len(reference_rows),
        "row_count": len(merged_rows),
        "reference_obj_id_offset": int(offset),
    }
    write_json(output_root / "oracle_residual_birth_bank_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
