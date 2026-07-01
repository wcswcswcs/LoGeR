from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUTS = {
    "v75_plan": "../docs/stream4d_v75_cmap_local_l2h_experiment_plan.md",
    "v70_carrier_witness": "outputs/audit/v70_carrier_witness/witness_summary.json",
    "v71_d4rt_atoms": "outputs/audit/v71_d4rt_atoms/atom_summary.json",
    "v71_candidate_bank": "outputs/audit/v71_candidate_bank/candidate_bank_summary.json",
    "v71_semantic_features": "outputs/audit/v71_semantic_features/semantic_summary.json",
    "v72_phase5_setcover_area": "outputs/audit/v72_phase5_proposal_setcover_area_bin1_uvmember_uvcoverage/proposal_setcover_summary.json",
    "v72_phase5_setcover_no_area": "outputs/audit/v72_phase5_proposal_setcover_no_area_floor_uvmember_uvcoverage/proposal_setcover_summary.json",
    "v73_phase3_d4rt": "outputs/audit/v73_phase3_d4rt_proposal_verification/d4rt_proposal_summary.json",
    "v73_phase4_local": "outputs/audit/v73_phase4_local_slot_birth/local_slot_summary.json",
    "v73_phase5_controls": "outputs/audit/v73_phase5_local_controls/local_control_summary.json",
    "v73_final_decision": "outputs/audit/v73_final_decision/final_decision.json",
}


ROW_CONTRACT_DEFAULTS = {
    "scene_id": "aggregate",
    "chunk_id": "aggregate",
    "phase": "v75_phase0_fact_lock",
    "variant": "fact_lock",
    "uses_gt_for_prediction": False,
    "uses_gt_for_evaluation": False,
    "diagnostic_only": False,
    "forbidden_for_method_table": False,
    "method_prediction_safe": True,
    "score_mode": "not_applicable",
    "support_scope": "prior_artifact_summary",
}


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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
        writer.writerows(rows)


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return {"text_file": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _metric_row(
    metric: str,
    value: Any,
    source: str,
    expected: str = "",
    passed: bool | None = None,
    notes: str = "",
) -> dict[str, Any]:
    row = dict(ROW_CONTRACT_DEFAULTS)
    row.update(
        {
            "metric": metric,
            "value": value,
            "source_artifact": source,
            "expected": expected,
            "pass": passed,
            "notes": notes,
        }
    )
    return row


def _source_row(name: str, path: Path, present: bool) -> dict[str, Any]:
    row = dict(ROW_CONTRACT_DEFAULTS)
    row.update(
        {
            "metric": "input_presence",
            "input_name": name,
            "value": present,
            "source_artifact": _rel(path) if present else str(path),
            "expected": "true",
            "pass": present,
            "bytes": path.stat().st_size if present else None,
            "sha256": _sha256(path) if present else None,
        }
    )
    return row


def _gt_row(source: str, json_path: str, payload: dict[str, Any], allowed: bool, notes: str) -> dict[str, Any]:
    row = dict(ROW_CONTRACT_DEFAULTS)
    row.update(
        {
            "phase": "v75_phase0_gt_boundary",
            "variant": "gt_boundary_scan",
            "uses_gt_for_prediction": payload.get("uses_gt_for_prediction"),
            "uses_gt_for_evaluation": payload.get("uses_gt_for_evaluation"),
            "diagnostic_only": payload.get("diagnostic_only"),
            "forbidden_for_method_table": payload.get("forbidden_for_method_table"),
            "method_prediction_safe": allowed,
            "source_artifact": source,
            "json_path": json_path,
            "gt_boundary_allowed": allowed,
            "notes": notes,
        }
    )
    return row


def _collect_gt_boundary(payload: Any, source: str, path: str = "$") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        has_boundary_fields = any(
            key in payload
            for key in (
                "uses_gt_for_prediction",
                "uses_gt_for_evaluation",
                "diagnostic_only",
                "forbidden_for_method_table",
                "method_uses_gt_anywhere",
                "uses_gt_for_method_prediction",
            )
        )
        if has_boundary_fields:
            uses_gt_for_prediction = _bool(payload.get("uses_gt_for_prediction")) or _bool(
                payload.get("method_uses_gt_anywhere")
            ) or _bool(payload.get("uses_gt_for_method_prediction"))
            diagnostic = _bool(payload.get("diagnostic_only"))
            forbidden = _bool(payload.get("forbidden_for_method_table"))
            allowed = (not uses_gt_for_prediction) or (diagnostic and forbidden)
            notes = "allowed diagnostic/oracle GT row" if allowed and uses_gt_for_prediction else "method-safe/no GT prediction"
            if not allowed:
                notes = "GT prediction is not marked diagnostic_only=true and forbidden_for_method_table=true"
            rows.append(_gt_row(source, path, payload, allowed, notes))
        for key, value in payload.items():
            rows.extend(_collect_gt_boundary(value, source, f"{path}.{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            rows.extend(_collect_gt_boundary(value, source, f"{path}[{index}]"))
    return rows


def _write_sha_rows(output_root: Path, inputs: dict[str, str]) -> None:
    rows: list[dict[str, Any]] = []
    for name, rel_path in inputs.items():
        path = _rooted(rel_path)
        if path.exists():
            row = dict(ROW_CONTRACT_DEFAULTS)
            row.update({"name": name, "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
            rows.append(row)
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            row = dict(ROW_CONTRACT_DEFAULTS)
            row.update({"name": f"output:{path.name}", "source_artifact": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
            rows.append(row)
    _write_csv(output_root / "sha256_rows.csv", rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    missing: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}
    source_rows: list[dict[str, Any]] = []
    for name, rel_path in DEFAULT_INPUTS.items():
        path = _rooted(rel_path)
        present = path.exists()
        source_rows.append(_source_row(name, path, present))
        if not present:
            missing.append({"input_name": name, "path": rel_path, "resolved_path": str(path)})
            continue
        loaded[name] = _load_payload(path)

    if missing:
        _write_csv(output_root / "main_rows.csv", source_rows)
        _write_csv(output_root / "fact_metric_rows.csv", [])
        _write_csv(output_root / "metric_rows.csv", [])
        _write_csv(output_root / "gt_boundary_rows.csv", [])
        _write_csv(output_root / "variant_summary_rows.csv", [])
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v75_phase0_fact_lock",
            "schema": "stream4d_v75_phase0_fact_lock_v1",
            "decision": "NO_GO_PHASE0_MISSING_INPUT",
            "all_required_inputs_present": False,
            "missing_input_count": len(missing),
            "can_enter_v75_local": False,
            "can_enter_local2history": False,
            "method_prediction_uses_gt_anywhere": None,
        }
        _write_json(output_root / "summary.json", summary)
        _write_json(output_root / "fact_lock_summary.json", summary)
        _write_sha_rows(output_root, DEFAULT_INPUTS)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    v70_witness = loaded["v70_carrier_witness"]
    v71_d4rt = loaded["v71_d4rt_atoms"]
    v71_candidate = loaded["v71_candidate_bank"]
    v71_semantic = loaded["v71_semantic_features"]
    v72_area = loaded["v72_phase5_setcover_area"]
    v72_no_area = loaded["v72_phase5_setcover_no_area"]
    v73_d4rt = loaded["v73_phase3_d4rt"]
    v73_local = loaded["v73_phase4_local"]
    v73_controls = loaded["v73_phase5_controls"]
    v73_final = loaded["v73_final_decision"]

    gt_rows: list[dict[str, Any]] = []
    for name, payload in loaded.items():
        if isinstance(payload, dict):
            gt_rows.extend(_collect_gt_boundary(payload, DEFAULT_INPUTS[name]))
    gt_violation_count = sum(1 for row in gt_rows if not _bool(row.get("gt_boundary_allowed")))

    v73_final_decision = v73_final.get("final_decision") or v73_final.get("decision")
    v73_local_gate_pass = _bool(_nested(v73_controls, "gate", "local_gate_pass")) or _bool(_nested(v73_local, "gate", "pass"))
    v73_controls_pass = _bool(_nested(v73_controls, "gate", "pass"))
    v73_controls_fail = v73_local_gate_pass and not v73_controls_pass
    d4rt_direct_contribution_not_proven = not _bool(v73_d4rt.get("D4RT_contribution_proven")) and not _bool(
        v73_final.get("D4RT_contribution_proven")
    )
    carrier_observations_available = _bool(_nested(v70_witness, "gate", "carrier_observation_tables_available")) and (
        v70_witness.get("decision") == "PASS_CARRIER_WITNESS_TABLE"
    )
    method_prediction_uses_gt_anywhere = gt_violation_count > 0 or _bool(v73_final.get("method_uses_gt_anywhere"))

    key_metrics = {
        "phase0_decision": None,
        "can_enter_v75_local": None,
        "method_prediction_uses_gt_anywhere": method_prediction_uses_gt_anywhere,
        "gt_boundary_violation_count": gt_violation_count,
        "v73_final_decision": v73_final_decision,
        "v73_local_SF50": _nested(v73_final, "key_metrics", "phase4_local_SF50"),
        "v73_area_only_control_SF50": _nested(v73_controls, "gate", "C6_area_only_control_SF50"),
        "v73_lattice_only_control_SF50": _nested(v73_controls, "gate", "C2_boundary_or_mask_lattice_only_SF50"),
        "v73_D4RT_real_minus_shuffled": v73_d4rt.get("real_minus_shuffled_SF50"),
        "v73_D4RT_real_minus_no_temporal": v73_d4rt.get("real_minus_no_temporal_SF50"),
        "v71_D4RT_atom_count_per_chunk_mean": _nested(v71_d4rt, "key_metrics", "atom_count_per_chunk_mean"),
        "v71_D4RT_atom_visible_frame_count_mean": _nested(v71_d4rt, "key_metrics", "atom_visible_frame_count_mean"),
        "v71_semantic_feature_success_rate": _nested(v71_semantic, "key_metrics", "semantic_feature_success_rate"),
        "v71_candidate_bank_oracle_SF50": _nested(v71_candidate, "key_metrics", "C6_union_candidate_bank_oracle_SF50"),
        "v72_area_phase5_decision": v72_area.get("decision"),
        "v72_no_area_phase5_decision": v72_no_area.get("decision"),
        "carrier_observations_available": carrier_observations_available,
        "D4RT_direct_contribution_not_proven": d4rt_direct_contribution_not_proven,
        "v75_requires_new_relational_lifting": True,
    }

    v73_final_expected = v73_final_decision == "NO_GO_LOCAL_CONTROLS_AREA_LATTICE_BIAS"
    all_required_inputs_present = True
    can_enter_v75_local = (
        all_required_inputs_present
        and not method_prediction_uses_gt_anywhere
        and v73_final_expected
        and v73_controls_fail
        and d4rt_direct_contribution_not_proven
        and carrier_observations_available
    )
    key_metrics["can_enter_v75_local"] = can_enter_v75_local
    key_metrics["phase0_decision"] = "PASS_V75_PHASE0_FACT_LOCK" if can_enter_v75_local else "NO_GO_PHASE0_FACT_MISMATCH"

    metric_rows = [
        _metric_row("all_required_inputs_present", all_required_inputs_present, "input scan", "true", all_required_inputs_present),
        _metric_row("method_prediction_uses_gt_anywhere", method_prediction_uses_gt_anywhere, "recursive summary scan + v73 final", "false", not method_prediction_uses_gt_anywhere),
        _metric_row("gt_boundary_violation_count", gt_violation_count, "recursive summary scan", "0", gt_violation_count == 0),
        _metric_row("v73_final_decision", v73_final_decision, DEFAULT_INPUTS["v73_final_decision"], "NO_GO_LOCAL_CONTROLS_AREA_LATTICE_BIAS", v73_final_expected),
        _metric_row("v73_local_gate_pass", v73_local_gate_pass, DEFAULT_INPUTS["v73_phase5_controls"], "true", v73_local_gate_pass),
        _metric_row("v73_controls_pass", v73_controls_pass, DEFAULT_INPUTS["v73_phase5_controls"], "false", not v73_controls_pass),
        _metric_row("v73_local_SF50", key_metrics["v73_local_SF50"], DEFAULT_INPUTS["v73_final_decision"], "record only"),
        _metric_row("v73_area_only_control_SF50", key_metrics["v73_area_only_control_SF50"], DEFAULT_INPUTS["v73_phase5_controls"], "record only"),
        _metric_row("v73_lattice_only_control_SF50", key_metrics["v73_lattice_only_control_SF50"], DEFAULT_INPUTS["v73_phase5_controls"], "record C2 boundary_or_mask_lattice control"),
        _metric_row("v73_D4RT_real_minus_shuffled", key_metrics["v73_D4RT_real_minus_shuffled"], DEFAULT_INPUTS["v73_phase3_d4rt"], "record old direct check"),
        _metric_row("D4RT_direct_contribution_not_proven", d4rt_direct_contribution_not_proven, DEFAULT_INPUTS["v73_phase3_d4rt"], "true", d4rt_direct_contribution_not_proven),
        _metric_row("v71_D4RT_atom_count_per_chunk_mean", key_metrics["v71_D4RT_atom_count_per_chunk_mean"], DEFAULT_INPUTS["v71_d4rt_atoms"], "record only"),
        _metric_row("v71_D4RT_atom_visible_frame_count_mean", key_metrics["v71_D4RT_atom_visible_frame_count_mean"], DEFAULT_INPUTS["v71_d4rt_atoms"], "record only"),
        _metric_row("v71_semantic_feature_success_rate", key_metrics["v71_semantic_feature_success_rate"], DEFAULT_INPUTS["v71_semantic_features"], "record only"),
        _metric_row("v71_candidate_bank_oracle_SF50", key_metrics["v71_candidate_bank_oracle_SF50"], DEFAULT_INPUTS["v71_candidate_bank"], "diagnostic headroom only"),
        _metric_row("carrier_observations_available", carrier_observations_available, DEFAULT_INPUTS["v70_carrier_witness"], "true", carrier_observations_available),
        _metric_row("v75_requires_new_relational_lifting", True, DEFAULT_INPUTS["v75_plan"], "true", True, "Phase0 confirms v75 is not a v73 area/lattice rescue."),
        _metric_row("can_enter_v75_local", can_enter_v75_local, "phase0_gate", "true", can_enter_v75_local),
        _metric_row("can_enter_local2history", False, "phase0_gate", "false until Phase6 local attribution pass", True),
    ]

    summary = {
        "phase": "v75_phase0_fact_lock",
        "schema": "stream4d_v75_phase0_fact_lock_v1",
        "decision": key_metrics["phase0_decision"],
        "phase0_decision": key_metrics["phase0_decision"],
        "inputs": DEFAULT_INPUTS,
        "key_metrics": key_metrics,
        "gate": {
            "all_required_inputs_present": all_required_inputs_present,
            "method_prediction_uses_gt_anywhere": method_prediction_uses_gt_anywhere,
            "gt_boundary_violation_count": gt_violation_count,
            "v73_final_controls_fail": v73_final_expected,
            "v73_local_gate_pass": v73_local_gate_pass,
            "v73_controls_fail": v73_controls_fail,
            "D4RT_direct_contribution_not_proven": d4rt_direct_contribution_not_proven,
            "carrier_observations_available": carrier_observations_available,
            "pass": can_enter_v75_local,
        },
        "can_enter_v75_local": can_enter_v75_local,
        "can_enter_local2history": False,
        "local2history_decision": "BLOCKED_UNTIL_PHASE6_LOCAL_ATTRIBUTION_PASS",
        "method_prediction_uses_gt_anywhere": method_prediction_uses_gt_anywhere,
        "gt_boundary_row_count": len(gt_rows),
        "gt_boundary_violation_count": gt_violation_count,
        "notes": [
            "Phase0 reads prior summaries and plan text only; it does not generate method predictions.",
            "v75 can start local CMAP only as a new relational lifting experiment, not as a v73 area/lattice rescue.",
            "local2history remains blocked until v75 Phase6 proves local method attribution.",
        ],
    }

    variant_summary_rows = [
        _metric_row("v75_phase0_fact_lock_decision", summary["decision"], "phase0_summary", "PASS_V75_PHASE0_FACT_LOCK", can_enter_v75_local),
    ]

    _write_csv(output_root / "main_rows.csv", source_rows)
    _write_csv(output_root / "metric_rows.csv", metric_rows)
    _write_csv(output_root / "fact_metric_rows.csv", metric_rows)
    _write_csv(output_root / "gt_boundary_rows.csv", gt_rows)
    _write_csv(output_root / "variant_summary_rows.csv", variant_summary_rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "fact_lock_summary.json", summary)
    _write_sha_rows(output_root, DEFAULT_INPUTS)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v75 Phase0 fact/protocol lock over v70-v73 Stream4D artifacts.")
    parser.add_argument("--output-root", default="outputs/audit/v75_phase0_fact_lock")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
