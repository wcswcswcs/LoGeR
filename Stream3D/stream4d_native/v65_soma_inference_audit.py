from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .soma_inference_policy import (
    bool_value,
    gt_geometry_eval_or_export_reasons,
    gt_geometry_inference_reasons,
    policy_violation_reasons,
)
from .v47_common import ROOT, read_json, utc_now, write_csv, write_json


DEFAULT_OUTPUT_ROOT = "outputs/audit/v65_soma_inference_policy_audit"


@dataclass(frozen=True)
class V65SOMAInferenceArtifact:
    name: str
    path: str | Path
    kind: str
    required: bool = False


DEFAULT_ARTIFACTS = (
    V65SOMAInferenceArtifact(
        "soma_object_bank_summary",
        "outputs/audit/v65_soma_object_bank/soma_object_bank_summary.json",
        "json",
        required=True,
    ),
    V65SOMAInferenceArtifact(
        "soma_object_bank_rows",
        "outputs/audit/v65_soma_object_bank/soma_object_bank_rows.csv",
        "csv",
        required=True,
    ),
    V65SOMAInferenceArtifact(
        "soma_object_material_rows",
        "outputs/audit/v65_soma_object_bank/soma_object_material_rows.csv",
        "csv",
        required=True,
    ),
    V65SOMAInferenceArtifact(
        "soma_object_support_rows",
        "outputs/audit/v65_soma_object_bank/soma_object_support_rows.csv",
        "csv",
        required=True,
    ),
    V65SOMAInferenceArtifact(
        "v65_ap_contract_summary",
        "outputs/audit/v65_ap_contract/ap_contract_summary.json",
        "json",
        required=True,
    ),
    V65SOMAInferenceArtifact(
        "v65_ap_contract_rows",
        "outputs/audit/v65_ap_contract/ap_contract_rows.csv",
        "csv",
        required=True,
    ),
    V65SOMAInferenceArtifact(
        "v65_soma_non_ap_repro_summary",
        "outputs/audit/v65_soma_non_ap_repro/soma_non_ap_repro_summary.json",
        "json",
        required=False,
    ),
    V65SOMAInferenceArtifact(
        "soma_eval_adapter_summary",
        "outputs/audit/v65_soma_eval_adapter/soma_eval_adapter_summary.json",
        "json",
        required=False,
    ),
    V65SOMAInferenceArtifact(
        "soma_eval_adapter_scene_rows",
        "outputs/audit/v65_soma_eval_adapter/soma_eval_adapter_scene_rows.csv",
        "csv",
        required=False,
    ),
)


def build_v65_soma_inference_audit(
    artifacts: tuple[V65SOMAInferenceArtifact, ...] = DEFAULT_ARTIFACTS,
) -> dict[str, Any]:
    scanned_rows: list[dict[str, Any]] = []
    violation_rows: list[dict[str, Any]] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []

    for artifact in artifacts:
        path = _project(artifact.path)
        if not path.exists():
            if artifact.required:
                missing_required.append(_rel(path))
            else:
                missing_optional.append(_rel(path))
            continue
        records = _load_records(path, artifact.kind)
        for idx, record in enumerate(records):
            audit_row = _audit_record(artifact, path, idx, record)
            scanned_rows.append(audit_row)
            if audit_row["violations"]:
                violation_rows.append(audit_row)

    method_rows = [row for row in scanned_rows if bool_value(row.get("is_method_result"))]
    gt_inference_rows = [row for row in scanned_rows if row.get("gt_geometry_inference_reasons")]
    method_inference_gt_rows = [
        row for row in method_rows if row.get("gt_geometry_inference_reasons")
    ]
    gt_inference_unmarked = [
        row
        for row in gt_inference_rows
        if bool_value(row.get("is_method_result"))
        or not bool_value(row.get("is_diagnostic_only"))
        or not bool_value(row.get("forbidden_for_method_table"))
    ]
    eval_or_export_rows = [
        row for row in scanned_rows if row.get("gt_geometry_eval_or_export_reasons")
    ]
    eval_or_export_unmarked = [
        row
        for row in eval_or_export_rows
        if not bool_value(row.get("is_diagnostic_only")) or not bool_value(row.get("forbidden_for_method_table"))
    ]
    gate = {
        "required_artifacts_present": not missing_required,
        "no_policy_violations": not violation_rows,
        "no_method_inference_gt_geometry": not method_inference_gt_rows,
        "gt_inference_records_are_forbidden_diagnostic_only": not gt_inference_unmarked,
        "gt_eval_or_export_records_are_diagnostic_only": not eval_or_export_unmarked,
        "method_records_scanned": bool(method_rows),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v65_soma_inference_policy_audit",
        "created_at": utc_now(),
        "scope": "Audit SOMA-related artifacts for the policy: no GT geometry in method inference; GT geometry only allowed for evaluation/diagnostic artifacts.",
        "artifact_count": len(artifacts),
        "record_count": len(scanned_rows),
        "method_record_count": len(method_rows),
        "gt_inference_record_count": len(gt_inference_rows),
        "gt_inference_record_ids": [
            {
                "artifact_name": row.get("artifact_name"),
                "row_id": row.get("row_id"),
                "reasons": row.get("gt_geometry_inference_reasons"),
                "is_method_result": row.get("is_method_result"),
                "is_diagnostic_only": row.get("is_diagnostic_only"),
                "forbidden_for_method_table": row.get("forbidden_for_method_table"),
            }
            for row in gt_inference_rows
        ],
        "method_inference_gt_geometry_record_count": len(method_inference_gt_rows),
        "gt_inference_unmarked_record_count": len(gt_inference_unmarked),
        "gt_eval_or_export_record_count": len(eval_or_export_rows),
        "gt_eval_or_export_unmarked_record_count": len(eval_or_export_unmarked),
        "policy_violation_count": len(violation_rows),
        "missing_required_artifacts": missing_required,
        "missing_optional_artifacts": missing_optional,
        "gate": gate,
    }
    return {
        "summary": summary,
        "scanned_rows": scanned_rows,
        "violation_rows": violation_rows,
    }


def write_v65_soma_inference_audit(output_root: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    root = _project(output_root)
    paths = {
        "summary": root / "soma_inference_policy_audit_summary.json",
        "scanned_rows": root / "soma_inference_policy_scanned_rows.csv",
        "violation_rows": root / "soma_inference_policy_violations.csv",
    }
    write_json(paths["summary"], payload["summary"])
    write_csv(paths["scanned_rows"], payload["scanned_rows"])
    write_csv(paths["violation_rows"], payload["violation_rows"])
    return {name: _rel(path) for name, path in paths.items()}


def _audit_record(
    artifact: V65SOMAInferenceArtifact,
    path: Path,
    row_index: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    normalized = _normalize_record(record)
    inference_reasons = gt_geometry_inference_reasons(normalized)
    eval_or_export_reasons = gt_geometry_eval_or_export_reasons(normalized)
    violations = policy_violation_reasons(normalized)
    if bool_value(normalized.get("is_method_result")) and inference_reasons:
        violations.append("method_inference_record_uses_gt_geometry")
    if bool_value(normalized.get("is_method_result")) and eval_or_export_reasons:
        violations.append("method_result_uses_eval_or_export_gt_geometry")
    if eval_or_export_reasons and not bool_value(normalized.get("is_diagnostic_only")):
        violations.append("gt_eval_or_export_not_diagnostic_only")
    violations = sorted(set(violations))
    return {
        "artifact_name": artifact.name,
        "artifact_path": _rel(path),
        "artifact_kind": artifact.kind,
        "row_index": row_index,
        "row_id": normalized.get("row_id") or normalized.get("object_id") or normalized.get("history_id") or "",
        "phase": normalized.get("phase") or "",
        "method_result_type": normalized.get("method_result_type") or "",
        "is_method_result": bool_value(normalized.get("is_method_result")),
        "method_safe_inference_artifact": bool_value(normalized.get("method_safe_inference_artifact")),
        "is_diagnostic_only": bool_value(normalized.get("is_diagnostic_only")),
        "forbidden_for_method_table": bool_value(normalized.get("forbidden_for_method_table")),
        "uses_gt_for_prediction": bool_value(normalized.get("uses_gt_for_prediction")),
        "uses_gt_geometry_for_inference": bool_value(normalized.get("uses_gt_geometry_for_inference")),
        "uses_rgbd_pose_mesh_for_export": bool_value(normalized.get("uses_rgbd_pose_mesh_for_export")),
        "gt_geometry_inference_reasons": ";".join(inference_reasons),
        "gt_geometry_eval_or_export_reasons": ";".join(eval_or_export_reasons),
        "violations": ";".join(violations),
    }


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    ground_truth_usage = out.get("ground_truth_usage")
    if isinstance(ground_truth_usage, dict):
        for key, value in ground_truth_usage.items():
            out.setdefault(key, value)
    return out


def _load_records(path: Path, kind: str) -> list[dict[str, Any]]:
    if kind == "json":
        payload = read_json(path)
        return [payload] if isinstance(payload, dict) else []
    if kind == "csv":
        return _read_csv_large(path)
    raise ValueError(f"Unsupported artifact kind: {kind}")


def _read_csv_large(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


__all__ = [
    "DEFAULT_ARTIFACTS",
    "DEFAULT_OUTPUT_ROOT",
    "V65SOMAInferenceArtifact",
    "build_v65_soma_inference_audit",
    "write_v65_soma_inference_audit",
]
