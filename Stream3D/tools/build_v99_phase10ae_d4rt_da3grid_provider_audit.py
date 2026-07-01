#!/usr/bin/env python3
"""Audit v99 D4RT provider artifacts decoded on the DA3 output grid."""

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
OUT_DIR = AUDIT_ROOT / "v99_phase10ae_d4rt_da3grid_provider_audit"

SCENES = {
    "scene0011_00": {
        "decode": AUDIT_ROOT / "v99_phase10ac_d4rt_da3grid_decode_scene0011_gpu6",
        "stitch": AUDIT_ROOT / "v99_phase10ad_d4rt_da3grid_stitched_scene0011",
        "decode_log": AUDIT_ROOT / "v99_phase10ac_d4rt_da3grid_decode_scene0011_gpu6.log",
        "stitch_log": AUDIT_ROOT / "v99_phase10ad_d4rt_da3grid_stitched_scene0011.log",
    },
    "scene0050_00": {
        "decode": AUDIT_ROOT / "v99_phase10ac_d4rt_da3grid_decode_scene0050_gpu7",
        "stitch": AUDIT_ROOT / "v99_phase10ad_d4rt_da3grid_stitched_scene0050",
        "decode_log": AUDIT_ROOT / "v99_phase10ac_d4rt_da3grid_decode_scene0050_gpu7.log",
        "stitch_log": AUDIT_ROOT / "v99_phase10ad_d4rt_da3grid_stitched_scene0050.log",
    },
}

REQUIRED_WIDTH = 504
REQUIRED_HEIGHT = 378


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _scan_track_grid(path: Path) -> dict[str, Any]:
    row_count = 0
    in01_count = 0
    bad_grid = 0
    src_min = np.asarray([np.inf, np.inf], dtype=np.float64)
    src_max = np.asarray([-np.inf, -np.inf], dtype=np.float64)
    tgt_min = np.asarray([np.inf, np.inf], dtype=np.float64)
    tgt_max = np.asarray([-np.inf, -np.inf], dtype=np.float64)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            if row.get("coordinate_grid") != "fixed_504x378":
                bad_grid += 1
            src = np.asarray([_num(row.get("u_src")), _num(row.get("v_src"))], dtype=np.float64)
            src_min = np.minimum(src_min, src)
            src_max = np.maximum(src_max, src)
            if _bool(row.get("uv_in01")):
                in01_count += 1
                tgt = np.asarray([_num(row.get("u_tgt")), _num(row.get("v_tgt"))], dtype=np.float64)
                tgt_min = np.minimum(tgt_min, tgt)
                tgt_max = np.maximum(tgt_max, tgt)
    return {
        "iterated_track_rows": int(row_count),
        "uv_in01_count": int(in01_count),
        "bad_coordinate_grid_rows": int(bad_grid),
        "u_src_min": float(src_min[0]),
        "u_src_max": float(src_max[0]),
        "v_src_min": float(src_min[1]),
        "v_src_max": float(src_max[1]),
        "u_tgt_in01_min": float(tgt_min[0]) if in01_count else "",
        "u_tgt_in01_max": float(tgt_max[0]) if in01_count else "",
        "v_tgt_in01_min": float(tgt_min[1]) if in01_count else "",
        "v_tgt_in01_max": float(tgt_max[1]) if in01_count else "",
    }


def _stitch_scale_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows: list[dict[str, str]]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, Any] = {"stitch_edge_count_from_rows": len(rows)}
    for field in ["fit_kept_anchor_count", "scale_curr_to_prev", "transform_scale_to_method", "residual_p90_curr_to_prev"]:
        values = [_num(row.get(field), np.nan) for row in rows]
        values = [value for value in values if np.isfinite(value)]
        if values:
            out[f"{field}_min"] = float(min(values))
            out[f"{field}_mean"] = float(sum(values) / len(values))
            out[f"{field}_max"] = float(max(values))
    return out


def _audit_scene(scene_id: str, spec: dict[str, Path]) -> dict[str, Any]:
    decode = _read_json(spec["decode"] / "summary.json")
    stitch = _read_json(spec["stitch"] / "summary.json")
    track_path = spec["stitch"] / "micro_track_rows.csv"
    grid = _scan_track_grid(track_path)
    pass_grid = (
        grid["bad_coordinate_grid_rows"] == 0
        and grid["u_src_min"] >= 0.0
        and grid["u_src_max"] <= REQUIRED_WIDTH - 1
        and grid["v_src_min"] >= 0.0
        and grid["v_src_max"] <= REQUIRED_HEIGHT - 1
        and float(grid["u_tgt_in01_min"]) >= 0.0
        and float(grid["u_tgt_in01_max"]) <= REQUIRED_WIDTH - 1
        and float(grid["v_tgt_in01_min"]) >= 0.0
        and float(grid["v_tgt_in01_max"]) <= REQUIRED_HEIGHT - 1
    )
    pass_stitch = (
        stitch.get("decision") == "PASS_V97_PHASE2_OVERLAP_STITCH_MICRO_TRACKS"
        and _int(stitch.get("missing_transform_track_row_count"), -1) == 0
        and _int(stitch.get("overlap_stitch_edge_count"), -1) == _int(stitch.get("required_overlap_stitch_edge_count"), -2)
    )
    return {
        "schema_version": "stream4d_v99_phase10ae_d4rt_da3grid_provider_row_v1",
        "phase_id": "v99_phase10ae_d4rt_da3grid_provider_audit",
        "scene_id": scene_id,
        "decode_root": _rel(spec["decode"]),
        "stitch_root": _rel(spec["stitch"]),
        "decode_log": _rel(spec["decode_log"]),
        "stitch_log": _rel(spec["stitch_log"]),
        "decode_decision": decode.get("decision", ""),
        "decoded_group_count": decode.get("decoded_group_count", ""),
        "decode_error_count": decode.get("error_count", ""),
        "decode_runtime_total_sec": decode.get("runtime_total_sec", ""),
        "decode_coordinate_grid": decode.get("coordinate_grid", ""),
        "decode_d4rt_input_width": decode.get("d4rt_input_width", ""),
        "decode_d4rt_input_height": decode.get("d4rt_input_height", ""),
        "decode_d4rt_output_width": decode.get("d4rt_output_width", ""),
        "decode_d4rt_output_height": decode.get("d4rt_output_height", ""),
        "stitch_decision": stitch.get("decision", ""),
        "overlap_stitch_edge_count": stitch.get("overlap_stitch_edge_count", ""),
        "required_overlap_stitch_edge_count": stitch.get("required_overlap_stitch_edge_count", ""),
        "missing_transform_track_row_count": stitch.get("missing_transform_track_row_count", ""),
        "micro_track_row_count": stitch.get("micro_track_row_count", ""),
        "stitched_track_row_count": stitch.get("stitched_track_row_count", ""),
        "stitch_runtime_sec": stitch.get("runtime_overlap_stitch_sec", ""),
        "uses_gt_for_prediction": bool(decode.get("uses_gt_for_prediction")) or bool(stitch.get("uses_gt_for_prediction")),
        "uses_future": bool(decode.get("uses_future")) or bool(stitch.get("uses_future")),
        "da3_grid_contract_pass": bool(pass_grid),
        "d4rt_self_overlap_stitch_contract_pass": bool(pass_stitch),
        "provider_gate_pass": bool(pass_grid and pass_stitch),
        **grid,
        **_stitch_scale_stats(spec["stitch"] / "overlap_stitch_rows.csv"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_audit_scene(scene_id, spec) for scene_id, spec in SCENES.items()]
    provider_gate = all(bool(row["provider_gate_pass"]) for row in rows)
    summary = {
        "schema_version": "stream4d_v99_phase10ae_d4rt_da3grid_provider_summary_v1",
        "phase_id": "v99_phase10ae_d4rt_da3grid_provider_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "PASS_D4RT_DA3GRID_CHUNK32O3_SELF_STITCH_PROVIDER_AUDIT" if provider_gate else "NO_GO_D4RT_DA3GRID_PROVIDER_AUDIT",
        "provider_gate_pass": bool(provider_gate),
        "required_d4rt_external_grid_width": REQUIRED_WIDTH,
        "required_d4rt_external_grid_height": REQUIRED_HEIGHT,
        "note": "D4RT checkpoint still internally resizes to its model.input.image_size; this audit locks the external D4RT RGB load/output pixel grid to DA3 results_output resolution.",
        "formal_ap_claim_allowed": False,
        "formal_ap_claim_blocker": "provider audit only; chunk-causal DA3-D4RT Sim3 anchor association and MV_AP_scene evaluation still required.",
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
