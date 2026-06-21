#!/usr/bin/env python3
"""Phase 0 state lock and artifact audit for ACL2 v74."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v73_semantic_memory_common import (
    TARGET_CHUNKS,
    load_json,
    parse_chunks,
    read_csv,
    safe_float,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_V73_ROOT = Path("results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final")
DEFAULT_OUT = Path(
    "results/kitti01_hmc_v2/acl2_v74_diagnostic_to_action_semantic_memory_control/"
    "report_final/phase0_state_lock"
)


OLD_FAILED_ACTIONS = {
    "mt3": "qscale hold/damp",
    "mt4": "semantic-conditioned overlap support",
    "mt5": "overlap support plus qscale",
    "mt7": "scale-side-state",
    "mt10": "RADIO component handoff",
    "mt11": "RADIO qscale handoff",
    "mt12": "thing/stuff state handoff",
    "mt13": "thing/stuff plus RADIO qscale",
    "mt14": "bounded scale clamp",
}


def _rows_by_chunk(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            out[int(row.get("chunk_id", -1))] = row
        except (TypeError, ValueError):
            continue
    return out


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _artifact_rows(v73_root: Path, chunks: list[int]) -> list[dict[str, Any]]:
    rows = read_csv(v73_root / "phase0_artifact_audit" / "artifact_availability.csv")
    by_chunk = _rows_by_chunk(rows)
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        row = dict(by_chunk.get(chunk, {"chunk_id": chunk}))
        dense_pass = _bool(row.get("dense_label_maps")) and _bool(row.get("dense_confidence_maps"))
        radio_pass = (
            _bool(row.get("radio_sidecar_exists"))
            and _bool(row.get("radio_component_id"))
            and _bool(row.get("radio_boundary"))
            and _bool(row.get("radio_interior"))
            and _bool(row.get("radio_temporal_stability"))
        )
        merge_swa_pass = (
            _bool(row.get("h35_geometry_exists"))
            and _bool(row.get("h35_overlap_pair_exists"))
            and _bool(row.get("h35_merge_state_exists"))
        )
        feature_pass = _bool(row.get("v68_feature_exists")) and _bool(row.get("global_k_layer5_7_available"))
        row.update(
            {
                "v74_dense_semantic_pass": dense_pass,
                "v74_radio_sidecar_pass": radio_pass,
                "v74_merge_swa_artifact_pass": merge_swa_pass,
                "v74_v68_feature_pass": feature_pass,
                "v74_artifact_pass": dense_pass and radio_pass and merge_swa_pass and feature_pass,
            }
        )
        out.append(row)
    return out


def _action_failure_matrix(v73_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase5 = v73_root / "phase5_mid_term"
    for path in sorted(phase5.glob("*/phaseE_multichunk_summary*.json")):
        payload = load_json(path) or {}
        if not isinstance(payload, dict):
            continue
        run_id = path.parent.name
        action_key = next((key for key in OLD_FAILED_ACTIONS if run_id.startswith(key)), "")
        head_count = payload.get("head_tail_pass_count")
        overlap_count = payload.get("overlap_pass_count")
        head_median = safe_float(payload.get("head_tail_median_improvement_vs_baseline_ratio"))
        overlap_median = safe_float(payload.get("overlap_median_improvement_vs_baseline_ratio"))
        row = {
            "run_id": run_id,
            "summary_file": str(path),
            "old_action_key": action_key,
            "planned_action_family": OLD_FAILED_ACTIONS.get(action_key, ""),
            "candidate": payload.get("candidate"),
            "is_control_summary": str(payload.get("candidate", "")).lower() in {"geometry_only", "random", "shuffled"},
            "chunks": ",".join(str(x) for x in payload.get("chunks", [])),
            "chunk_count": len(payload.get("chunks", [])),
            "phaseE_gate_pass": bool(payload.get("phaseE_gate_pass")),
            "head_tail_pass_count": head_count,
            "overlap_pass_count": overlap_count,
            "head_tail_median_improvement_vs_baseline_ratio": head_median,
            "overlap_median_improvement_vs_baseline_ratio": overlap_median,
            "missing_count": len(payload.get("missing", [])),
            "missing": ",".join(str(x) for x in payload.get("missing", [])),
            "promotion_status": "no_promotion_old_failed_action",
            "promotion_reason": "v73 summary phaseE_gate_pass is false; v74 may use these only as prior failure/action-fidelity evidence.",
        }
        rows.append(row)
    return rows


def _old_action_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, dict[str, Any]] = {}
    for key, desc in OLD_FAILED_ACTIONS.items():
        key_rows = [row for row in rows if row.get("old_action_key") == key]
        if not key_rows:
            by_key[key] = {"description": desc, "found": False, "promotion_allowed": False}
            continue
        candidate_rows = [row for row in key_rows if not bool(row.get("is_control_summary"))] or key_rows
        max_head = max(int(row.get("head_tail_pass_count") or 0) for row in candidate_rows)
        max_overlap = max(int(row.get("overlap_pass_count") or 0) for row in candidate_rows)
        best_head_median = max((safe_float(row.get("head_tail_median_improvement_vs_baseline_ratio")) or -999.0) for row in candidate_rows)
        best_overlap_median = max((safe_float(row.get("overlap_median_improvement_vs_baseline_ratio")) or -999.0) for row in candidate_rows)
        by_key[key] = {
            "description": desc,
            "found": True,
            "summary_count": len(key_rows),
            "candidate_summary_count": len(candidate_rows),
            "control_summary_count": len(key_rows) - len(candidate_rows),
            "any_phaseE_gate_pass": any(bool(row.get("phaseE_gate_pass")) for row in candidate_rows),
            "max_head_tail_pass_count": max_head,
            "max_overlap_pass_count": max_overlap,
            "best_head_tail_median_improvement_vs_baseline_ratio": best_head_median,
            "best_overlap_median_improvement_vs_baseline_ratio": best_overlap_median,
            "promotion_allowed": False,
        }
    return by_key


def _write_markdown(path: Path, state: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v74 Phase 0 State Lock",
        "",
        f"- created_at: `{state['created_at']}`",
        f"- v73_status: `{state['v73_status']}`",
        f"- v74_main_path_allowed: `{state['v74_main_path_allowed']}`",
        f"- dense_semantic_gate_pass: `{state['dense_semantic_gate_pass']}`",
        f"- merge_swa_artifact_gate_pass: `{state['merge_swa_artifact_gate_pass']}`",
        f"- radio_sidecar_gate_pass: `{state['radio_sidecar_gate_pass']}`",
        "",
        "## Path Status",
        "",
    ]
    for key, value in state["path_status"].items():
        lines.append(f"- {key}: `{value['status']}`. {value['reason']}")
    lines += [
        "",
        "## Old Action Lock",
        "",
        "- MT3, MT4, MT5, MT7, MT10, MT11, MT12, MT13, MT14 are prior failed actions.",
        "- They are usable as failure/action-fidelity evidence only.",
        "- They are not promoted into v74 online without new Phase 3/4 gate evidence.",
        "",
        "## Local Positives",
        "",
        "- chunk8, chunk29, chunk10 remain diagnostic/local prior positives only.",
        "- No v73 local positive is treated as v74 full11 or official H35 success.",
        "",
    ]
    write_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v73-root", type=Path, default=DEFAULT_V73_ROOT)
    parser.add_argument("--target-chunks", default=",".join(map(str, TARGET_CHUNKS)))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    chunks = parse_chunks(args.target_chunks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    v73_summary = load_json(args.v73_root / "v73_summary.json") or {}
    v73_allowed = load_json(args.v73_root / "phase0_artifact_audit" / "allowed_paths.json") or {}
    v73_artifact_summary = load_json(args.v73_root / "phase0_artifact_audit" / "artifact_availability.json") or {}
    action_rows = _action_failure_matrix(args.v73_root)
    artifact_rows = _artifact_rows(args.v73_root, chunks)

    dense_pass = bool(artifact_rows) and all(bool(row.get("v74_dense_semantic_pass")) for row in artifact_rows)
    radio_pass = bool(artifact_rows) and all(bool(row.get("v74_radio_sidecar_pass")) for row in artifact_rows)
    merge_pass = bool(artifact_rows) and all(bool(row.get("v74_merge_swa_artifact_pass")) for row in artifact_rows)
    feature_pass = bool(artifact_rows) and all(bool(row.get("v74_v68_feature_pass")) for row in artifact_rows)
    v73_phase0 = v73_summary.get("phase0_artifact_audit", {}) if isinstance(v73_summary, dict) else {}
    y_long = (v73_summary.get("phase1_scale_drift_ledger", {}) if isinstance(v73_summary, dict) else {}).get("Y_long")

    path_status = {
        "READ": {
            "status": "diagnostic_only",
            "reason": "v73/v70 locks show source-attention evidence is not sufficient for promotion; v74 READ can re-enter only under Phase 7 criteria.",
        },
        "SWA_MERGE": {
            "status": "primary_path_allowed_for_offline_oracle",
            "reason": "H35 geometry, overlap pairs, merge states, dense semantic features, and RADIO sidecars are available for target chunks.",
        },
        "TTT": {
            "status": "diagnostic_only",
            "reason": f"v73 Y_long is {y_long}; no current future-scale TTT label or online promotion gate is available.",
        },
        "704_full": {
            "status": "blocked_until_online_gate",
            "reason": "v74 plan forbids 704F/full before Phase 3/4 and online gates pass.",
        },
    }
    state = {
        "schema": "acl2_v74_phase0_state_lock_v1",
        "created_at": utc_now(),
        "v73_root": str(args.v73_root),
        "target_chunks": chunks,
        "v73_status": v73_summary.get("status"),
        "v73_final_interpretation": v73_summary.get("final_interpretation"),
        "v73_phase0_artifact_audit": v73_phase0,
        "v73_allowed_paths": v73_allowed,
        "v73_artifact_summary_keys": list(v73_artifact_summary.keys()) if isinstance(v73_artifact_summary, dict) else [],
        "dense_semantic_gate_pass": dense_pass,
        "radio_sidecar_gate_pass": radio_pass,
        "merge_swa_artifact_gate_pass": merge_pass,
        "v68_feature_gate_pass": feature_pass,
        "v74_main_path_allowed": dense_pass and merge_pass and feature_pass,
        "semantic_conditioned_tests_allowed": dense_pass,
        "radio_conditioned_tests_allowed": radio_pass,
        "old_action_summary": _old_action_summary(action_rows),
        "path_status": path_status,
        "current_local_positives_prior_only": [8, 29, 10],
        "gate_decision": (
            "proceed_to_phase1_phase2_phase3"
            if dense_pass and merge_pass and feature_pass
            else "stop_v74_main_path_missing_required_artifacts"
        ),
    }
    allowed_paths = {
        "wave0_allowed": state["v74_main_path_allowed"],
        "phase1_frozen_validation": {
            "status": "allowed" if state["v74_main_path_allowed"] else "blocked",
            "reason": "Uses v73 phase1/2/3 locked ledgers and does not require online promotion.",
        },
        "phase2_failure_mode_classifier": {
            "status": "allowed" if state["v74_main_path_allowed"] else "blocked",
            "reason": "Uses frozen ledger features; classifier is diagnostic until action gates pass.",
        },
        "phase3_action_family_oracle": {
            "status": "allowed" if state["v74_main_path_allowed"] else "blocked",
            "reason": "Offline oracle may run on saved geometry/semantic/merge traces.",
        },
        "phase5_online": {
            "status": "blocked_until_phase3_or_phase4_gate",
            "reason": "Plan forbids online implementation from diagnostic formula alone.",
        },
        "phase6_selector": {
            "status": "blocked_until_positive_count_ge_4",
            "reason": "Do not train selector from failed v73 action labels.",
        },
        "phase8_704_full": {
            "status": "blocked_until_online_gate",
            "reason": "No v74 online candidate exists yet.",
        },
    }

    write_json(args.out_dir / "v73_state_lock.json", state)
    write_csv(args.out_dir / "v73_action_failure_matrix.csv", action_rows)
    write_csv(args.out_dir / "v74_artifact_availability.csv", artifact_rows)
    write_json(args.out_dir / "allowed_paths.json", allowed_paths)
    _write_markdown(args.out_dir / "v74_state_lock.md", state)
    print(
        {
            "out_dir": str(args.out_dir),
            "v74_main_path_allowed": state["v74_main_path_allowed"],
            "dense_semantic_gate_pass": dense_pass,
            "merge_swa_artifact_gate_pass": merge_pass,
            "radio_sidecar_gate_pass": radio_pass,
            "old_action_rows": len(action_rows),
        }
    )


if __name__ == "__main__":
    main()
