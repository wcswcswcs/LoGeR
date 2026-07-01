#!/usr/bin/env python3
"""Audit v99 holdout DA3-Streaming chunk32/overlap3 provider artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_provider_audit"

SCENES = {
    "scene0011_00": {
        "input": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0011_input",
        "output": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0011_base",
        "log": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0011_base.log",
    },
    "scene0050_00": {
        "input": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0050_input",
        "output": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0050_base",
        "log": AUDIT_ROOT / "v99_phase10s_da3_holdout_chunk32o3_scene0050_base.log",
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _ply_vertex_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("element vertex "):
                try:
                    return int(line.strip().split()[-1])
                except Exception:
                    return -1
            if line.strip() == "end_header":
                break
    return -1


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_min(values: list[float]) -> float | None:
    return float(min(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _audit_scene(scene: str, spec: dict[str, Path]) -> dict[str, Any]:
    input_dir = spec["input"]
    output_dir = spec["output"]
    log_path = spec["log"]
    config_path = output_dir / "da3_streaming_d4rt32o3_config.yaml"
    input_summary = _read_json(input_dir / "summary.json")
    config = _read_yaml(config_path)
    model_cfg = config.get("Model", {})
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    processing_pat = re.compile(
        r"Processing\s+(\d+)\s+images\s+in\s+(\d+)\s+chunks\s+of\s+size\s+(\d+)\s+with\s+(\d+)\s+overlap"
    )
    match = processing_pat.search(log_text)
    observed_frame_count = int(match.group(1)) if match else -1
    observed_chunk_count = int(match.group(2)) if match else -1
    observed_chunk_size = int(match.group(3)) if match else -1
    observed_overlap = int(match.group(4)) if match else -1
    result_npz_count = len(list((output_dir / "results_output").glob("frame_*.npz")))
    aligned_chunk_count = len(list((output_dir / "_tmp_results_aligned").glob("chunk_*.npy")))
    unaligned_chunk_count = len(list((output_dir / "_tmp_results_unaligned").glob("chunk_*.npy")))
    pcd_chunk_count = len([p for p in (output_dir / "pcd").glob("*_pcd.ply") if p.name != "combined_pcd.ply"])
    combined_vertex_count = _ply_vertex_count(output_dir / "pcd/combined_pcd.ply")
    camera_pose_rows = 0
    if (output_dir / "camera_poses.txt").exists():
        camera_pose_rows = sum(1 for line in (output_dir / "camera_poses.txt").read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    scale_values = [float(x) for x in re.findall(r"Estimated Scale:\s*([0-9eE+\-.]+)", log_text)]
    mean_errors = [float(x) for x in re.findall(r"Mean error:\s*([0-9eE+\-.]+)", log_text)]
    required_chunk_size = 32
    required_overlap = 3
    expected_frame_count = int(input_summary.get("frame_count", -1))
    expected_chunk_count = 1 + math.ceil(max(0, expected_frame_count - required_chunk_size) / (required_chunk_size - required_overlap))
    config_chunk_size = int(model_cfg.get("chunk_size", -1))
    config_overlap = int(model_cfg.get("overlap", -1))
    pass_gate = (
        observed_frame_count == expected_frame_count
        and observed_chunk_count == expected_chunk_count
        and observed_chunk_size == required_chunk_size
        and observed_overlap == required_overlap
        and config_chunk_size == required_chunk_size
        and config_overlap == required_overlap
        and result_npz_count == expected_frame_count
        and aligned_chunk_count == expected_chunk_count
        and unaligned_chunk_count == expected_chunk_count
        and pcd_chunk_count == expected_chunk_count
        and "DA3-Streaming done." in log_text
    )
    blocker_parts: list[str] = []
    if not pass_gate:
        if observed_chunk_size != required_chunk_size or observed_overlap != required_overlap:
            blocker_parts.append("log_chunk_overlap_mismatch")
        if config_chunk_size != required_chunk_size or config_overlap != required_overlap:
            blocker_parts.append("config_chunk_overlap_mismatch")
        if result_npz_count != expected_frame_count:
            blocker_parts.append("missing_frame_npz_outputs")
        if aligned_chunk_count != expected_chunk_count or unaligned_chunk_count != expected_chunk_count:
            blocker_parts.append("missing_da3_chunk_outputs")
        if "DA3-Streaming done." not in log_text:
            blocker_parts.append("da3_done_marker_missing")
    return {
        "schema_version": "stream4d_v99_phase10s_da3_holdout_provider_row_v1",
        "phase_id": "v99_phase10s_da3_holdout_chunk32o3_provider_audit",
        "provider": "DA3-Streaming",
        "model": "DA3-BASE",
        "scene_id": scene,
        "input_dir": _rel(input_dir),
        "output_dir": _rel(output_dir),
        "log_path": _rel(log_path),
        "config_path": _rel(config_path),
        "expected_frame_count": expected_frame_count,
        "observed_frame_count": observed_frame_count,
        "expected_chunk_count": expected_chunk_count,
        "observed_chunk_count": observed_chunk_count,
        "config_chunk_size": config_chunk_size,
        "config_overlap": config_overlap,
        "observed_chunk_size": observed_chunk_size,
        "observed_overlap": observed_overlap,
        "required_chunk_size": required_chunk_size,
        "required_overlap": required_overlap,
        "result_npz_count": result_npz_count,
        "aligned_chunk_count": aligned_chunk_count,
        "unaligned_chunk_count": unaligned_chunk_count,
        "pcd_chunk_count": pcd_chunk_count,
        "combined_pcd_vertex_count": combined_vertex_count,
        "camera_pose_rows": camera_pose_rows,
        "self_overlap_stitch_scale_count": len(scale_values),
        "self_overlap_stitch_scale_min": _safe_min(scale_values),
        "self_overlap_stitch_scale_mean": _mean(scale_values),
        "self_overlap_stitch_scale_max": _safe_max(scale_values),
        "self_overlap_stitch_mean_error_count": len(mean_errors),
        "self_overlap_stitch_mean_error_min": _safe_min(mean_errors),
        "self_overlap_stitch_mean_error_mean": _mean(mean_errors),
        "self_overlap_stitch_mean_error_max": _safe_max(mean_errors),
        "scale_ambiguous": True,
        "da3_self_overlap_stitch_applied": True,
        "cross_model_sim3_required_before_da3_d4rt_comparison": True,
        "uses_gt_for_prediction": bool(input_summary.get("uses_gt_for_prediction", False)),
        "uses_future": bool(input_summary.get("uses_future", False)),
        "provider_gate_pass": pass_gate,
        "blocker": ";".join(blocker_parts),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_audit_scene(scene, spec) for scene, spec in SCENES.items()]
    provider_gate_pass = all(bool(row["provider_gate_pass"]) for row in rows)
    summary = {
        "schema_version": "stream4d_v99_phase10s_da3_holdout_provider_summary_v1",
        "phase_id": "v99_phase10s_da3_holdout_chunk32o3_provider_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "PASS_DA3_HOLDOUT_CHUNK32O3_PROVIDER_AUDIT" if provider_gate_pass else "NO_GO_DA3_HOLDOUT_CHUNK32O3_PROVIDER_AUDIT",
        "provider_gate_pass": provider_gate_pass,
        "da3_ready_for_v99_holdout_chunk32_overlap3": provider_gate_pass,
        "formal_ap_claim_allowed": False,
        "formal_ap_claim_allowed_reason": "provider audit only; MV_AP_scene must be rerun by a downstream method/evaluator",
        "required_chunk_size": 32,
        "required_overlap": 3,
        "scale_alignment_contract": [
            "DA3 is scale ambiguous and must self-overlap stitch first.",
            "D4RT is scale ambiguous and must self-overlap stitch first.",
            "DA3 and D4RT must be Sim3/scale aligned before cross-model geometric comparison.",
        ],
        "uses_gt_for_prediction": any(bool(row["uses_gt_for_prediction"]) for row in rows),
        "uses_future": any(bool(row["uses_future"]) for row in rows),
        "scene_count": len(rows),
        "scene_rows": rows,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "provider_rows": _rel(OUT_DIR / "provider_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "provider_rows.csv", rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if provider_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
