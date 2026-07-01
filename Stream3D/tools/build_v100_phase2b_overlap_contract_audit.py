#!/usr/bin/env python3
"""Audit whether v100 Phase2 materialized rows actually satisfy overlap=3."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
PHASE2_DIR = AUDIT_ROOT / "v100_phase2_f2_local_final"
OUT_DIR = AUDIT_ROOT / "v100_phase2b_overlap_contract_audit"


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
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
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _chunk_index(chunk_id: str) -> int:
    text = str(chunk_id)
    if text.startswith("c"):
        return int(text[1:])
    return int(float(text))


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2 = json.loads((PHASE2_DIR / "summary.json").read_text(encoding="utf-8"))
    expected_overlap = int(phase2.get("method_contract", {}).get("overlap", 3))
    df = pd.read_parquet(PHASE2_DIR / "mv_object_frame_mask_rows.parquet")
    chunk_frames: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in df[["dataset_split", "scene_id", "chunk_id", "frame_id"]].drop_duplicates().itertuples(index=False):
        chunk_frames[(str(row.dataset_split), str(row.scene_id), str(row.chunk_id))].add(int(row.frame_id))

    transition_rows: list[dict[str, Any]] = []
    by_split_scene: dict[tuple[str, str], list[tuple[str, set[int]]]] = defaultdict(list)
    for (split, scene, chunk), frames in chunk_frames.items():
        by_split_scene[(split, scene)].append((chunk, frames))
    for (split, scene), chunks in sorted(by_split_scene.items()):
        chunks_sorted = sorted(chunks, key=lambda item: _chunk_index(item[0]))
        for (chunk_a, frames_a), (chunk_b, frames_b) in zip(chunks_sorted, chunks_sorted[1:]):
            shared = sorted(frames_a & frames_b)
            transition_rows.append(
                {
                    "schema_version": "stream4d_v100_phase2b_overlap_transition_row_v1",
                    "phase_id": "v100_phase2b_overlap_contract_audit",
                    "dataset_split": split,
                    "scene_id": scene,
                    "chunk_a": chunk_a,
                    "chunk_b": chunk_b,
                    "chunk_a_frame_count": len(frames_a),
                    "chunk_b_frame_count": len(frames_b),
                    "expected_overlap_frame_count": expected_overlap,
                    "observed_overlap_frame_count": len(shared),
                    "shared_frames": ";".join(str(v) for v in shared),
                    "chunk_a_tail_frames": ";".join(str(v) for v in sorted(frames_a)[-5:]),
                    "chunk_b_head_frames": ";".join(str(v) for v in sorted(frames_b)[:5]),
                    "overlap_contract_pass": len(shared) >= expected_overlap,
                }
            )
    failed = [row for row in transition_rows if not bool(row["overlap_contract_pass"])]
    gate_rows = [
        {
            "schema_version": "stream4d_v100_phase2b_gate_row_v1",
            "phase_id": "v100_phase2b_overlap_contract_audit",
            "gate_id": "materialized_adjacent_chunk_overlap_ge_contract",
            "pass": len(failed) == 0,
            "expected": f">={expected_overlap} shared stride-5 frames for every adjacent chunk pair",
            "observed": f"failed_transitions={len(failed)}/{len(transition_rows)} min_overlap={min([row['observed_overlap_frame_count'] for row in transition_rows] or [0])}",
            "severity": "formal_contract_required",
        }
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v100_phase2b_failure_row_v1",
            "phase_id": "v100_phase2b_overlap_contract_audit",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "Rebuild Phase2 object birth with true overlap-aware chunk membership, or update method claim to overlap=0. Do not claim overlap=3 from current materialized rows.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    pass_gate = not failure_rows
    transition_csv = OUT_DIR / "overlap_transition_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    summary_json = OUT_DIR / "summary.json"
    _write_csv(transition_csv, transition_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    _write_csv(
        artifact_csv,
        [
            {
                "schema_version": "stream4d_v100_phase2b_artifact_manifest_row_v1",
                "phase_id": "v100_phase2b_overlap_contract_audit",
                "artifact_path": _rel(path),
                "artifact_type": kind,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "note": note,
            }
            for path, kind, note in [
                (transition_csv, "csv", "adjacent chunk overlap evidence"),
                (gate_csv, "csv", "overlap contract gate"),
                (failure_csv, "csv", "overlap contract failures"),
            ]
        ],
    )
    summary = {
        "schema_version": "stream4d_v100_phase2b_overlap_contract_audit_summary_v1",
        "phase_id": "v100_phase2b_overlap_contract_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_OVERLAP3_CONTRACT" if pass_gate else "BLOCK_OVERLAP3_CONTRACT_NOT_MATERIALIZED",
        "phase2b_pass": pass_gate,
        "failure_count": len(failure_rows),
        "expected_overlap": expected_overlap,
        "transition_count": len(transition_rows),
        "failed_transition_count": len(failed),
        "min_observed_overlap": min([row["observed_overlap_frame_count"] for row in transition_rows] or [0]),
        "max_observed_overlap": max([row["observed_overlap_frame_count"] for row in transition_rows] or [0]),
        "formal_claim_allowed": False,
        "outputs": {
            "summary": _rel(summary_json),
            "overlap_transition_rows": _rel(transition_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(summary_json, summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
