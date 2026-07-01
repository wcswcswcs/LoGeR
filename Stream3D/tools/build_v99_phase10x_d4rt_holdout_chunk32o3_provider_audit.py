#!/usr/bin/env python3
"""Audit v99 holdout D4RT chunk32/overlap3 decode and self-stitch artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10x_d4rt_holdout_chunk32o3_provider_audit"

SCENES = {
    "scene0011_00": {
        "query": AUDIT_ROOT / "v99_phase10u_d4rt_holdout_chunk32o3_query_scene0011",
        "decode": AUDIT_ROOT / "v99_phase10v_d4rt_holdout_chunk32o3_decode_scene0011_gpu6",
        "stitch": AUDIT_ROOT / "v99_phase10w_d4rt_holdout_chunk32o3_stitched_scene0011",
        "decode_log": AUDIT_ROOT / "v99_phase10v_d4rt_holdout_chunk32o3_decode_scene0011_gpu6.log",
        "stitch_log": AUDIT_ROOT / "v99_phase10w_d4rt_holdout_chunk32o3_stitched_scene0011.log",
    },
    "scene0050_00": {
        "query": AUDIT_ROOT / "v99_phase10u_d4rt_holdout_chunk32o3_query_scene0050",
        "decode": AUDIT_ROOT / "v99_phase10v_d4rt_holdout_chunk32o3_decode_scene0050_gpu7",
        "stitch": AUDIT_ROOT / "v99_phase10w_d4rt_holdout_chunk32o3_stitched_scene0050",
        "decode_log": AUDIT_ROOT / "v99_phase10v_d4rt_holdout_chunk32o3_decode_scene0050_gpu7.log",
        "stitch_log": AUDIT_ROOT / "v99_phase10w_d4rt_holdout_chunk32o3_stitched_scene0050.log",
    },
}


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
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _line_count(path: Path) -> int:
    if not path.exists():
        return -1
    count = 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _audit_scene(scene: str, spec: dict[str, Path]) -> dict[str, Any]:
    query = _read_json(spec["query"] / "summary.json")
    decode = _read_json(spec["decode"] / "summary.json")
    stitch = _read_json(spec["stitch"] / "summary.json")
    stitch_rows = _read_csv(spec["stitch"] / "overlap_stitch_rows.csv")
    quality_rows = _read_csv(spec["decode"] / "d4rt_quality_rows.csv")
    kept = [_num(row.get("fit_kept_anchor_count")) for row in stitch_rows]
    scales = [_num(row.get("transform_scale_to_method")) for row in stitch_rows]
    uv_rates = [_num(row.get("uv_in01_rate")) for row in quality_rows]
    visibility = [_num(row.get("visibility_mean")) for row in quality_rows]
    confidence = [_num(row.get("confidence_mean")) for row in quality_rows]
    chunk_gate = int(query.get("chunk_size", -1)) == 32 and int(query.get("overlap_frames", -1)) == 3
    decode_gate = int(decode.get("decoded_group_count") or 0) == int(query.get("chunk_count") or -1) and int(decode.get("error_count") or 0) == 0
    stitch_gate = (
        bool(stitch.get("d4rt_applies_overlap_stitch"))
        and _int(stitch.get("overlap_stitch_edge_count"), -1) == _int(stitch.get("required_overlap_stitch_edge_count"), -2)
        and _int(stitch.get("missing_transform_track_row_count"), -1) == 0
        and all(v >= _int(stitch.get("min_overlap_anchors"), 16) for v in kept)
    )
    provider_gate = chunk_gate and decode_gate and stitch_gate and not bool(stitch.get("uses_gt_for_prediction")) and not bool(stitch.get("uses_future"))
    return {
        "schema_version": "stream4d_v99_phase10x_d4rt_holdout_provider_row_v1",
        "phase_id": "v99_phase10x_d4rt_holdout_chunk32o3_provider_audit",
        "provider": "D4RT",
        "scene_id": scene,
        "query_root": _rel(spec["query"]),
        "decode_root": _rel(spec["decode"]),
        "stitch_root": _rel(spec["stitch"]),
        "decode_log": _rel(spec["decode_log"]),
        "stitch_log": _rel(spec["stitch_log"]),
        "chunk_size": int(query.get("chunk_size", -1)),
        "overlap": int(query.get("overlap_frames", -1)),
        "chunk_count": int(query.get("chunk_count", -1)),
        "unique_input_frame_count": int(query.get("unique_input_frame_count", -1)),
        "query_row_count": int(query.get("query_row_count", -1)),
        "source_row_count": int(query.get("source_row_count", -1)),
        "decode_decision": decode.get("decision", ""),
        "decoded_group_count": int(decode.get("decoded_group_count") or 0),
        "decode_error_count": int(decode.get("error_count") or 0),
        "decode_runtime_total_sec": float(decode.get("runtime_total_sec") or 0.0),
        "query_chunk_size": int(decode.get("query_chunk_size") or 0),
        "query_chunk_size_note": "query batching count, not D4RT temporal model chunk length",
        "model_frame_mode": decode.get("model_frame_mode", ""),
        "micro_query_row_count": max(0, _line_count(spec["decode"] / "micro_query_rows.csv") - 1),
        "micro_track_row_count": max(0, _line_count(spec["decode"] / "micro_track_rows.csv") - 1),
        "stitch_decision": stitch.get("decision", ""),
        "can_enter_phase3_from_legacy_gate": bool(stitch.get("can_enter_phase3", False)),
        "d4rt_applies_overlap_stitch": bool(stitch.get("d4rt_applies_overlap_stitch")),
        "d4rt_applies_final_gt_sim3": bool(stitch.get("d4rt_applies_final_gt_sim3")),
        "geometry_coordinate_mode": stitch.get("geometry_coordinate_mode", ""),
        "overlap_stitch_edge_count": int(stitch.get("overlap_stitch_edge_count") or 0),
        "required_overlap_stitch_edge_count": int(stitch.get("required_overlap_stitch_edge_count") or 0),
        "min_overlap_anchors": int(stitch.get("min_overlap_anchors") or 0),
        "fit_kept_anchor_count_min": float(min(kept)) if kept else 0.0,
        "fit_kept_anchor_count_mean": float(np.mean(kept)) if kept else 0.0,
        "fit_kept_anchor_count_max": float(max(kept)) if kept else 0.0,
        "stitch_scale_min": float(min(scales)) if scales else 0.0,
        "stitch_scale_mean": float(np.mean(scales)) if scales else 0.0,
        "stitch_scale_max": float(max(scales)) if scales else 0.0,
        "stitched_track_row_count": int(stitch.get("stitched_track_row_count") or 0),
        "missing_transform_track_row_count": int(stitch.get("missing_transform_track_row_count") or 0),
        "runtime_overlap_stitch_sec": float(stitch.get("runtime_overlap_stitch_sec") or 0.0),
        "uv_in01_rate_mean": float(np.mean(uv_rates)) if uv_rates else 0.0,
        "visibility_mean": float(np.mean(visibility)) if visibility else 0.0,
        "confidence_mean": float(np.mean(confidence)) if confidence else 0.0,
        "scale_ambiguous": True,
        "cross_model_sim3_required_before_da3_d4rt_comparison": True,
        "uses_gt_for_prediction": bool(query.get("uses_gt_for_prediction")) or bool(decode.get("uses_gt_for_prediction")) or bool(stitch.get("uses_gt_for_prediction")),
        "uses_future": bool(query.get("uses_future")) or bool(decode.get("uses_future")) or bool(stitch.get("uses_future")),
        "chunk32_overlap3_self_stitch_provider_gate_pass": provider_gate,
        "formal_ap_claim_allowed": False,
        "formal_ap_claim_blocker": "provider only; D4RT reliable anchor association and MV_AP_scene evaluation not rerun",
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_audit_scene(scene, spec) for scene, spec in SCENES.items()]
    provider_gate = all(bool(row["chunk32_overlap3_self_stitch_provider_gate_pass"]) for row in rows)
    legacy_quality_pass = all(bool(row["can_enter_phase3_from_legacy_gate"]) for row in rows)
    summary = {
        "schema_version": "stream4d_v99_phase10x_d4rt_holdout_provider_summary_v1",
        "phase_id": "v99_phase10x_d4rt_holdout_chunk32o3_provider_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "PASS_D4RT_HOLDOUT_CHUNK32O3_SELF_STITCH_PROVIDER_AUDIT" if provider_gate else "NO_GO_D4RT_HOLDOUT_CHUNK32O3_SELF_STITCH_PROVIDER_AUDIT",
        "provider_gate_pass": provider_gate,
        "legacy_decode_quality_gate_pass": legacy_quality_pass,
        "formal_ap_claim_allowed": False,
        "formal_ap_claim_blocker": "D4RT provider audit only; reliable anchor association, DA3<->D4RT Sim3 alignment, and MV_AP_scene evaluation are still required.",
        "required_chunk_size": 32,
        "required_overlap": 3,
        "scale_alignment_contract": [
            "D4RT is scale ambiguous and must self-overlap stitch first.",
            "DA3 is scale ambiguous and must self-overlap stitch first.",
            "DA3 and D4RT must be Sim3/scale aligned before cross-model geometric comparison.",
        ],
        "uses_gt_for_prediction": any(bool(row["uses_gt_for_prediction"]) for row in rows),
        "uses_future": any(bool(row["uses_future"]) for row in rows),
        "scene_rows": rows,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "provider_rows": _rel(OUT_DIR / "provider_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "provider_rows.csv", rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if provider_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
