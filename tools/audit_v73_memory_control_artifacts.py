#!/usr/bin/env python3
"""Phase 0 artifact audit for ACL2 v73 semantic-memory control."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v73_semantic_memory_common import (
    TARGET_CHUNKS,
    find_chunk_dir,
    load_json,
    parse_chunks,
    safe_float,
    torch_load,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_OUT = Path("results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final/phase0_artifact_audit")
DEFAULT_STAGE = Path("results/kitti_preprocess/01/stage_c_cache_semantic_chunks")
DEFAULT_RADIO = Path("results/kitti_preprocess/01/radseg_sidecar_chunks_slide336_stride224")
DEFAULT_H35 = Path(
    "results/kitti01_hmc_v2/acl2_v67_dense_semantic_reconstruction/"
    "phaseO2_h35_trace_geom_merge_full/rollouts/V67S_H35_TRACE_GEOM_MERGE_FULL_H35_PARITY"
)
DEFAULT_FEATURES = Path("results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/phaseC_target_feature_dumps/features")
DEFAULT_V70 = Path("results/kitti01_hmc_v2/acl2_v70_geometry_first_semantic_trust/report_final")
DEFAULT_V69 = Path("results/kitti01_hmc_v2/acl2_v69_semantic_anchor_scale/report_final")


def _semantic_status(stage_dir: Path, chunk_id: int) -> dict[str, Any]:
    chunk_dir = find_chunk_dir(stage_dir, chunk_id)
    row: dict[str, Any] = {
        "chunk_id": chunk_id,
        "stage_chunk_dir": str(chunk_dir) if chunk_dir else "",
        "stage_masklet_exists": False,
        "dense_label_maps": False,
        "dense_confidence_maps": False,
        "thingstuff_source_type": False,
        "semantic_frames": None,
        "semantic_source": "",
        "semantic_failure": "",
    }
    if chunk_dir is None:
        row["semantic_failure"] = "missing_stage_chunk_dir"
        return row
    masklet = chunk_dir / "masklet.pt"
    row["stage_masklet_exists"] = masklet.exists()
    if not masklet.exists():
        row["semantic_failure"] = "missing_masklet_pt"
        return row
    try:
        payload = torch_load(masklet)
    except Exception as exc:
        row["semantic_failure"] = f"load_error:{type(exc).__name__}"
        return row
    sem = payload.get("semantic_segmentation") if isinstance(payload, dict) else None
    row["thingstuff_source_type"] = isinstance(payload, dict) and isinstance(payload.get("source_type"), list)
    if not isinstance(sem, dict):
        row["semantic_failure"] = "missing_semantic_segmentation"
        return row
    labels = sem.get("label_maps")
    confidence = sem.get("confidence_maps")
    row["dense_label_maps"] = hasattr(labels, "shape")
    row["dense_confidence_maps"] = hasattr(confidence, "shape")
    row["semantic_frames"] = int(labels.shape[0]) if hasattr(labels, "shape") else None
    row["semantic_source"] = str(sem.get("source", ""))
    if not row["dense_label_maps"]:
        row["semantic_failure"] = "missing_label_maps"
    elif not row["dense_confidence_maps"]:
        row["semantic_failure"] = "missing_confidence_maps"
    return row


def _radio_status(radio_dir: Path, chunk_id: int) -> dict[str, Any]:
    chunk_dir = find_chunk_dir(radio_dir, chunk_id)
    row: dict[str, Any] = {
        "chunk_id": chunk_id,
        "radio_chunk_dir": str(chunk_dir) if chunk_dir else "",
        "radio_sidecar_exists": False,
        "radio_component_id": False,
        "radio_boundary": False,
        "radio_interior": False,
        "radio_temporal_stability": False,
        "radio_shape": "",
        "radio_failure": "",
    }
    if chunk_dir is None:
        row["radio_failure"] = "missing_radio_chunk_dir"
        return row
    sidecar = chunk_dir / "radio_sidecar.pt"
    row["radio_sidecar_exists"] = sidecar.exists()
    if not sidecar.exists():
        row["radio_failure"] = "missing_radio_sidecar_pt"
        return row
    try:
        payload = torch_load(sidecar)
    except Exception as exc:
        row["radio_failure"] = f"load_error:{type(exc).__name__}"
        return row
    for key, field in (
        ("object_component_id", "radio_component_id"),
        ("object_boundary_score", "radio_boundary"),
        ("object_interior_score", "radio_interior"),
        ("temporal_stability", "radio_temporal_stability"),
    ):
        value = payload.get(key) if isinstance(payload, dict) else None
        row[field] = hasattr(value, "shape")
        if key == "object_component_id" and hasattr(value, "shape"):
            row["radio_shape"] = str(list(value.shape))
    if not all(bool(row[k]) for k in ("radio_component_id", "radio_boundary", "radio_interior", "radio_temporal_stability")):
        row["radio_failure"] = "missing_required_radio_fields"
    return row


def _feature_status(feature_dir: Path, path_taps: dict[str, Any] | None, chunk_id: int) -> dict[str, Any]:
    feature_path = feature_dir / f"chunk_{chunk_id:03d}.pt"
    row: dict[str, Any] = {
        "chunk_id": chunk_id,
        "v68_feature_path": str(feature_path),
        "v68_feature_exists": feature_path.exists(),
        "global_k_layer5_7_available": False,
        "path_tap_manifest_available": False,
        "feature_failure": "",
    }
    if feature_path.exists():
        try:
            payload = torch_load(feature_path)
            tensor = payload.get("tap::global_k_raw_patchvec_layers") if isinstance(payload, dict) else None
            row["global_k_layer5_7_available"] = hasattr(tensor, "shape") and int(tensor.shape[1]) >= 2
        except Exception as exc:
            row["feature_failure"] = f"load_error:{type(exc).__name__}"
    entries = path_taps.get("entries", []) if isinstance(path_taps, dict) else []
    for entry in entries:
        if int(entry.get("chunk_id", -1)) == int(chunk_id):
            row["path_tap_manifest_available"] = bool(entry.get("available"))
            break
    if not row["v68_feature_exists"]:
        row["feature_failure"] = "missing_v68_feature_dump"
    elif not row["global_k_layer5_7_available"]:
        row["feature_failure"] = row["feature_failure"] or "missing_global_k_layer5_7"
    return row


def _h35_status(run_dir: Path, chunk_id: int) -> dict[str, Any]:
    geom = run_dir / "per_chunk_geometry" / f"chunk_{chunk_id:03d}.pt"
    pair = run_dir / "overlap_pairs" / f"chunk_{chunk_id - 1:03d}_{chunk_id:03d}.pt"
    merge_state = run_dir / "merge_states" / f"chunk_{chunk_id:03d}_transform.json"
    return {
        "chunk_id": chunk_id,
        "h35_geometry_exists": geom.exists(),
        "h35_overlap_pair_exists": pair.exists() if chunk_id > 0 else False,
        "h35_merge_state_exists": merge_state.exists(),
        "h35_run_dir": str(run_dir),
    }


def _previous_locks(v69_root: Path, v70_root: Path) -> dict[str, Any]:
    v69 = load_json(v69_root / "phaseC_centered_action_retarget_h35" / "oracle_target_retarget_summary.json") or {}
    v70_mech = load_json(v70_root / "phaseR5_radio_mechanism_input_audit_after_swa" / "radio_mechanism_input_audit.json") or {}
    v70_gate = load_json(v70_root / "v70_plan_gate_status_audit.json") or {}
    return {
        "v69_centered_retarget": {
            "path": str(v69_root / "phaseC_centered_action_retarget_h35" / "oracle_target_retarget_summary.json"),
            "exists": bool(v69),
            "mechanism_target_gate_pass_any": v69.get("mechanism_target_gate_pass_any"),
            "existing_oracle_gate_chunks": v69.get("existing_oracle_gate_chunks"),
            "future_pass_chunks": next(
                (row.get("positive_chunks") for row in v69.get("target_summaries", []) if row.get("target") == "future_after_overlap"),
                None,
            ),
            "headtail_pass_chunks": next(
                (row.get("positive_chunks") for row in v69.get("target_summaries", []) if row.get("target") == "head_to_tail"),
                None,
            ),
        },
        "v70_mechanism_audit": {
            "path": str(v70_root / "phaseR5_radio_mechanism_input_audit_after_swa" / "radio_mechanism_input_audit.json"),
            "exists": bool(v70_mech),
            "decision": v70_mech.get("decision"),
            "mechanism_gate_pass": v70_mech.get("mechanism_gate_pass"),
            "r6_online_allowed": v70_mech.get("r6_online_allowed"),
            "blockers": v70_mech.get("blockers", []),
        },
        "v70_plan_gate_status": {
            "path": str(v70_root / "v70_plan_gate_status_audit.json"),
            "exists": bool(v70_gate),
            "aggregated_decision": v70_gate.get("aggregated_decision"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-chunks", default=",".join(map(str, TARGET_CHUNKS)))
    parser.add_argument("--stage-c-cache", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--radio-sidecar-dir", type=Path, default=DEFAULT_RADIO)
    parser.add_argument("--h35-run-dir", type=Path, default=DEFAULT_H35)
    parser.add_argument("--v68-feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--v70-report-root", type=Path, default=DEFAULT_V70)
    parser.add_argument("--v69-report-root", type=Path, default=DEFAULT_V69)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    chunks = parse_chunks(args.target_chunks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path_taps = load_json(args.v70_report_root / "phase1_path_taps" / "path_taps_manifest.json") or {}
    rows: list[dict[str, Any]] = []
    for chunk_id in chunks:
        row: dict[str, Any] = {"chunk_id": chunk_id}
        row.update(_semantic_status(args.stage_c_cache, chunk_id))
        row.update(_radio_status(args.radio_sidecar_dir, chunk_id))
        row.update(_feature_status(args.v68_feature_dir, path_taps, chunk_id))
        row.update(_h35_status(args.h35_run_dir, chunk_id))
        row["target_chunk_artifact_pass"] = all(
            bool(row.get(key))
            for key in (
                "dense_label_maps",
                "dense_confidence_maps",
                "thingstuff_source_type",
                "radio_sidecar_exists",
                "radio_component_id",
                "global_k_layer5_7_available",
                "h35_geometry_exists",
                "h35_overlap_pair_exists",
                "h35_merge_state_exists",
            )
        )
        rows.append(row)

    v68_audit = load_json(
        Path("results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/phaseA_cache_confidence_guard/dense_semantic_source_audit.json")
    ) or {}
    locks = _previous_locks(args.v69_report_root, args.v70_report_root)
    merge_trace = args.h35_run_dir / "merge_state_trace.jsonl"
    hmc_state = args.h35_run_dir / "hmc_state_hash.jsonl"
    ttt_spatial = args.v70_report_root / "phaseR5_radio_ttt_spatial_delta_diagnostic_v1" / "radio_ttt_spatial_delta_summary.json"
    ttt_summary = load_json(ttt_spatial) or {}
    sidecar_count = len(list(args.radio_sidecar_dir.glob("chunk_*/radio_sidecar.pt")))
    stage_count = len(list(args.stage_c_cache.glob("chunk_*/masklet.pt")))
    summary = {
        "schema": "acl2_v73_phase0_artifact_audit_v1",
        "created_at": utc_now(),
        "target_chunks": chunks,
        "stage_c_cache": str(args.stage_c_cache),
        "stage_chunk_count": stage_count,
        "radio_sidecar_dir": str(args.radio_sidecar_dir),
        "radio_sidecar_count": sidecar_count,
        "h35_run_dir": str(args.h35_run_dir),
        "target_rows": len(rows),
        "target_artifact_pass_count": sum(bool(row.get("target_chunk_artifact_pass")) for row in rows),
        "all_target_artifacts_pass": all(bool(row.get("target_chunk_artifact_pass")) for row in rows),
        "dense_semantic_v68_audit": {
            "path": "results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/phaseA_cache_confidence_guard/dense_semantic_source_audit.json",
            "all_chunks_have_confidence_maps": v68_audit.get("all_chunks_have_confidence_maps"),
            "confidence_maps_chunks": v68_audit.get("confidence_maps_chunks"),
            "semantic_source": v68_audit.get("semantic_source"),
            "confidence_normalized_to_0_1": v68_audit.get("confidence_normalized_to_0_1"),
        },
        "path_taps_available_entries": sum(1 for entry in path_taps.get("entries", []) if entry.get("available")) if isinstance(path_taps, dict) else 0,
        "merge_trace_exists": merge_trace.exists(),
        "hmc_state_hash_exists": hmc_state.exists(),
        "ttt_spatial_delta_summary_exists": ttt_spatial.exists(),
        "ttt_spatial_delta_decision": ttt_summary.get("decision"),
        "previous_no_go_locks": locks,
    }
    allowed_paths = {
        "wave0_allowed": bool(summary["all_target_artifacts_pass"] and summary["merge_trace_exists"] and summary["hmc_state_hash_exists"]),
        "short_term_read_action": {
            "status": "diagnostic_proxy_only",
            "reason": "v68/v70 path taps are global_k feature proxies; online attention mass taps are not sufficient by themselves.",
        },
        "mid_term_swa_merge_action": {
            "status": "offline_oracle_and_smoke_allowed",
            "reason": "H35 overlap pairs, merge trace, RADIO sidecars, and prior v70 SWA/MERGE oracle artifacts exist; online promotion still requires v73 gate.",
        },
        "long_term_ttt_action": {
            "status": "diagnostic_only" if ttt_spatial.exists() else "blocked_missing_spatial_delta_summary",
            "reason": "TTT spatial delta is available as diagnostic summary only; future-scale impact must be separately proven.",
        },
        "phase10_704_full": {
            "status": "blocked_until_single_path_gate",
            "reason": "Plan forbids 704F/full without Wave0 and single-path action gate evidence.",
        },
    }
    write_csv(args.out_dir / "artifact_availability.csv", rows)
    write_json(args.out_dir / "artifact_availability.json", {"summary": summary, "rows": rows})
    write_json(args.out_dir / "allowed_paths.json", allowed_paths)
    lock_lines = [
        "# v73 Previous No-Go Lock",
        "",
        f"- v69 centered retarget summary exists: {locks['v69_centered_retarget']['exists']}",
        f"- v69 mechanism_target_gate_pass_any: {locks['v69_centered_retarget']['mechanism_target_gate_pass_any']}",
        f"- v69 existing_oracle_gate_chunks: {locks['v69_centered_retarget']['existing_oracle_gate_chunks']}",
        f"- v69 future_pass_chunks: {locks['v69_centered_retarget']['future_pass_chunks']}",
        f"- v69 headtail_pass_chunks: {locks['v69_centered_retarget']['headtail_pass_chunks']}",
        f"- v70 mechanism audit decision: {locks['v70_mechanism_audit']['decision']}",
        f"- v70 mechanism_gate_pass: {locks['v70_mechanism_audit']['mechanism_gate_pass']}",
        f"- v70 r6_online_allowed: {locks['v70_mechanism_audit']['r6_online_allowed']}",
        "",
        "Interpretation: these are locks against reusing old proxy/oracle positives as v73 method success. They do not replace v73 Wave 0 or action gates.",
        "",
    ]
    write_text(args.out_dir / "previous_no_go_lock.md", "\n".join(lock_lines))
    print({"out_dir": str(args.out_dir), "all_target_artifacts_pass": summary["all_target_artifacts_pass"], "target_rows": len(rows)})


if __name__ == "__main__":
    main()
