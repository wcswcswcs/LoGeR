#!/usr/bin/env python3
"""Audit geometry-provider availability for v99 overlap3 scene repair."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10r_geometry_provider_contract_audit"

DA3_HOLDOUT = {
    "scene0011_00": {
        "input": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0011_input",
        "output": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0011",
        "log": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0011.log",
    },
    "scene0050_00": {
        "input": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0050_input",
        "output": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0050",
        "log": AUDIT_ROOT / "v98_phase1_provider_contract/da3_streaming_holdout_scene0050.log",
    },
}
D4RT_HOLDOUT_PHASE4 = AUDIT_ROOT / "v98_phase13_holdout_phase4_d4rt_anchor_alignment"
D4RT_HOLDOUT_SURFEL_OBS = AUDIT_ROOT / "v98_phase13_holdout_phase5_fused_surfel/surfel_observation_rows.csv"
D4RT_PROVIDER_CANDIDATES = {
    "v99_phase5_full_dev_legacy_anchor_source": AUDIT_ROOT / "v97_phase2_d4rt_micro_tracks_full_D3_gpu7",
    "v97_overlap48_self_stitched_legacy": AUDIT_ROOT / "v97_phase2_d4rt_micro_tracks_overlap48_48clip_all4_q512_stitched",
    "v98_phase13_holdout_anchor_alignment": D4RT_HOLDOUT_PHASE4,
}


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] in {"Stream3D", "docs", "Open-d4rt"}:
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _da3_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pat = re.compile(r"Processing\s+(\d+)\s+images\s+in\s+(\d+)\s+chunks\s+of\s+size\s+(\d+)\s+with\s+(\d+)\s+overlap")
    for scene, spec in DA3_HOLDOUT.items():
        input_summary = _read_json(Path(spec["input"]) / "summary.json") if (Path(spec["input"]) / "summary.json").exists() else {}
        log_text = Path(spec["log"]).read_text(encoding="utf-8", errors="replace") if Path(spec["log"]).exists() else ""
        match = pat.search(log_text)
        chunk_size = int(match.group(3)) if match else -1
        overlap = int(match.group(4)) if match else -1
        rows.append(
            {
                "schema_version": "stream4d_v99_phase10r_da3_provider_contract_v1",
                "phase_id": "v99_phase10r_geometry_provider_contract_audit",
                "provider": "DA3-Streaming",
                "scene_id": scene,
                "input_dir": _rel(spec["input"]),
                "output_dir": _rel(spec["output"]),
                "log_path": _rel(spec["log"]),
                "frame_count": input_summary.get("frame_count", ""),
                "observed_chunk_size": chunk_size,
                "observed_overlap": overlap,
                "required_chunk_size": 32,
                "required_overlap": 3,
                "config_matches_v99_chunk32_overlap3": chunk_size == 32 and overlap == 3,
                "uses_gt_for_prediction": bool(input_summary.get("uses_gt_for_prediction", False)),
                "uses_future": bool(input_summary.get("uses_future", False)),
                "self_stitch_required": True,
                "scale_ambiguous": True,
                "cross_model_sim3_required_before_da3_d4rt_comparison": True,
                "formal_scene_stitch_evidence_allowed": False,
                "blocker": "existing holdout DA3 artifact is not chunk32 overlap3" if not (chunk_size == 32 and overlap == 3) else "",
            }
        )
    return rows


def _d4rt_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nonempty_anchor_obs = 0
    total_obs = 0
    if D4RT_HOLDOUT_SURFEL_OBS.exists():
        for row in _read_csv(D4RT_HOLDOUT_SURFEL_OBS):
            total_obs += 1
            if str(row.get("d4rt_anchor_ids_nearby", "")).strip():
                nonempty_anchor_obs += 1

    for candidate_id, root in D4RT_PROVIDER_CANDIDATES.items():
        summary_path = root / "summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        query_root_summary: dict[str, Any] = {}
        query_root = str(summary.get("query_root", "") or "")
        if query_root:
            qroot = _project(query_root)
            if (qroot / "summary.json").exists():
                query_root_summary = _read_json(qroot / "summary.json")

        # D4RT query_chunk_size is a query batching count, not the model temporal
        # chunk length. Only explicit chunk_size/overlap_frames evidence is used
        # for the v99 chunk32/overlap3 contract.
        observed_chunk_size = summary.get("chunk_size", query_root_summary.get("chunk_size", -1))
        observed_overlap = summary.get(
            "overlap",
            summary.get("overlap_frames", query_root_summary.get("overlap_frames", -1)),
        )
        try:
            observed_chunk_size_int = int(observed_chunk_size)
        except Exception:
            observed_chunk_size_int = -1
        try:
            observed_overlap_int = int(observed_overlap)
        except Exception:
            observed_overlap_int = -1
        chunk_contract_pass = observed_chunk_size_int == 32 and observed_overlap_int == 3

        anchor_row_count = int(summary.get("anchor_row_count") or 0)
        if candidate_id == "v98_phase13_holdout_anchor_alignment":
            surfel_anchor_count = nonempty_anchor_obs
            surfel_row_count = total_obs
        else:
            surfel_anchor_count = ""
            surfel_row_count = ""

        self_stitch_applied = bool(summary.get("d4rt_applies_overlap_stitch", False))
        blocker_parts: list[str] = []
        if not chunk_contract_pass:
            blocker_parts.append("missing_or_wrong_d4rt_chunk32_overlap3_contract")
        if not self_stitch_applied:
            blocker_parts.append("d4rt_self_overlap_stitch_not_verified")
        if candidate_id == "v98_phase13_holdout_anchor_alignment" and (anchor_row_count <= 0 or nonempty_anchor_obs <= 0):
            blocker_parts.append("holdout_d4rt_anchor_alignment_has_zero_anchors")

        rows.append(
            {
            "schema_version": "stream4d_v99_phase10r_d4rt_provider_contract_v1",
            "phase_id": "v99_phase10r_geometry_provider_contract_audit",
            "provider": "D4RT",
                "candidate_id": candidate_id,
                "root": _rel(root),
                "summary_path": _rel(summary_path),
                "decision": summary.get("decision", ""),
                "decode_scope": summary.get("decode_scope", ""),
                "query_root": _rel(_project(query_root)) if query_root else "",
                "query_root_summary_path": _rel(_project(query_root) / "summary.json") if query_root else "",
                "observed_chunk_size": observed_chunk_size_int,
                "observed_overlap": observed_overlap_int,
                "required_chunk_size": 32,
                "required_overlap": 3,
                "config_matches_v99_chunk32_overlap3": chunk_contract_pass,
                "query_chunk_size": summary.get("query_chunk_size", ""),
                "query_chunk_size_note": "query batching count, not D4RT temporal model chunk length",
                "model_frame_mode": summary.get("model_frame_mode", ""),
                "backend_model_frame_mode": summary.get("backend_model_frame_mode", ""),
                "d4rt_self_overlap_stitch_applied": self_stitch_applied,
                "anchor_row_count": anchor_row_count,
                "surfel_observation_row_count": surfel_row_count,
                "surfel_observation_with_anchor_count": surfel_anchor_count,
            "formal_scene_stitch_evidence_allowed": False,
            "self_stitch_required": True,
            "scale_ambiguous": True,
            "cross_model_sim3_required_before_da3_d4rt_comparison": True,
                "uses_gt_for_prediction": bool(summary.get("uses_gt_for_prediction", False)),
                "uses_future": bool(summary.get("uses_future", False)),
                "blocker": ";".join(blocker_parts),
            }
        )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    da3_rows = _da3_rows()
    d4rt_rows = _d4rt_rows()
    da3_ready = all(bool(row["config_matches_v99_chunk32_overlap3"]) for row in da3_rows)
    d4rt_holdout_rows = [row for row in d4rt_rows if row.get("candidate_id") == "v98_phase13_holdout_anchor_alignment"]
    d4rt_ready = any(
        bool(row.get("config_matches_v99_chunk32_overlap3"))
        and bool(row.get("d4rt_self_overlap_stitch_applied"))
        and int(row.get("anchor_row_count") or 0) > 0
        and int(row.get("surfel_observation_with_anchor_count") or 0) > 0
        for row in d4rt_holdout_rows
    )
    summary = {
        "schema_version": "stream4d_v99_phase10r_geometry_provider_contract_audit_summary_v1",
        "phase_id": "v99_phase10r_geometry_provider_contract_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "GEOMETRY_PROVIDER_NOT_READY_FOR_HOLDOUT_SCENE_STITCH",
        "formal_claim_allowed": False,
        "da3_ready_for_v99_chunk32_overlap3": da3_ready,
        "d4rt_ready_for_holdout_scene_stitch": d4rt_ready,
        "d4rt_chunk32_overlap3_contract_required": True,
        "da3_d4rt_cross_model_scale_alignment_required": True,
        "scale_alignment_contract": [
            "DA3 must be self-overlap stitched first.",
            "D4RT must be self-overlap stitched first.",
            "DA3 and D4RT must then be Sim3/scale aligned before any cross-model geometric comparison.",
            "Raw centroid/distance comparisons across DA3 and D4RT coordinates are not valid evidence.",
            "Both DA3 and D4RT provider artifacts must explicitly use chunk_size=32 with overlap=3.",
        ],
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "da3_provider_rows": _rel(OUT_DIR / "da3_provider_rows.csv"),
            "d4rt_provider_rows": _rel(OUT_DIR / "d4rt_provider_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "da3_provider_rows.csv", da3_rows)
    _write_csv(OUT_DIR / "d4rt_provider_rows.csv", d4rt_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
