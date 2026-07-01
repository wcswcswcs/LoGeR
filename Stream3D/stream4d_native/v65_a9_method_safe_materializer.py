from __future__ import annotations

from pathlib import Path
from typing import Any

from .v53_native_carrier_materialization import (
    build_native_carrier_materialization,
    write_native_carrier_materialization,
)
from .v65_common import load_dict, project, rel, sha256_file, write_standard_outputs
from .v65_soma_object_bank import (
    V65SOMAObjectBankConfig,
    build_v65_soma_object_bank,
    write_v65_soma_object_bank,
)


A9_ROOT = "outputs/audit/v65_a9_method_safe_materializer"

AP_JOIN_KEYS = {
    "mesh_vertex_id",
    "mesh_vertex_ids",
    "scene_point_id",
    "scene_point_ids",
    "scannet_vertex_id",
    "scannet_vertex_ids",
    "point_id",
    "point_ids",
    "pre_points",
    "pre_point",
}

NATIVE_INPUTS = {
    "v62_native_summary": "outputs/audit/v62_native_field/native_field_summary.json",
    "v64r2_native_contract": "outputs/audit/v64r2_native_contract/native_contract_summary.json",
    "v64r2_material_state_rows": "outputs/audit/v64r2_native_contract/material_state_rows.csv",
    "v64r2_object_field_rows": "outputs/audit/v64r2_native_contract/object_field_rows.csv",
    "v53_native_carrier_summary": "outputs/audit/v53_native_carrier_materialization/native_carrier_summary.json",
    "v53_native_carrier_rows": "outputs/audit/v53_native_carrier_materialization/objectlet_native_carrier_rows.csv",
    "v47_carrier_observation_table": "outputs/audit/v47_observation_tables_metricfix/carrier_observation_table.csv",
}


def build_v65_a9_method_safe_materializer(
    *,
    output_root: str | Path = A9_ROOT,
    rerun_native_carrier: bool = True,
) -> dict[str, Any]:
    """Attempt A9 method-safe native AP materialization and record exact blockers.

    This does not convert native carrier support with RGB-D/pose/mesh. If the
    existing native artifacts do not contain a method-safe ScanNet AP-mask join
    key, A9 stays blocked with evidence instead of being silently skipped.
    """

    root = project(output_root)
    native_attempt: dict[str, Any] = {}
    if rerun_native_carrier:
        native_payload = build_native_carrier_materialization()
        write_native_carrier_materialization(root / "native_carrier_attempt", native_payload)
        native_attempt = native_payload.get("summary", {})
    else:
        native_attempt = load_dict("outputs/audit/v53_native_carrier_materialization/native_carrier_summary.json")

    object_bank_root = root / "soma_object_bank_attempt"
    object_bank_payload = build_v65_soma_object_bank(V65SOMAObjectBankConfig(output_root=object_bank_root))
    object_bank_paths = write_v65_soma_object_bank(object_bank_payload, object_bank_root)
    object_bank_summary = object_bank_payload["summary"]

    evidence_rows = _artifact_evidence_rows(root)
    join_key_rows = [row for row in evidence_rows if row.get("present_ap_join_keys")]
    native_contract = load_dict(NATIVE_INPUTS["v64r2_native_contract"])
    v62_summary = load_dict(NATIVE_INPUTS["v62_native_summary"])
    source_hashes = {
        name: sha256_file(path)
        for name, path in NATIVE_INPUTS.items()
        if project(path).exists()
    }
    attempt_summary_path = root / "native_carrier_attempt" / "native_carrier_summary.json"
    if attempt_summary_path.exists():
        source_hashes["v65_native_carrier_attempt"] = sha256_file(attempt_summary_path)
    for name, path in object_bank_paths.items():
        source_hashes[f"v65_soma_object_bank_{name}"] = sha256_file(path)

    object_bank_available = bool(object_bank_summary.get("object_bank_available"))
    object_bank_support_available = int(object_bank_summary.get("object_support_row_count") or 0) > 0
    method_safe_support_available = bool(
        native_attempt.get("method_safe_native_support_available")
        or load_dict(NATIVE_INPUTS["v53_native_carrier_summary"]).get("method_safe_native_support_available")
        or object_bank_support_available
    )
    has_ap_join_key = bool(join_key_rows)
    method_safe_ap_available = bool(method_safe_support_available and has_ap_join_key)
    if method_safe_ap_available:
        status = "ready_for_scannet_ap_export"
        blocker = ""
    elif object_bank_available and object_bank_support_available:
        status = "object_bank_ready_no_scannet_ap_mask_export"
        if int(object_bank_summary.get("native_carrier_support_row_count") or 0) > 0:
            blocker = (
                "Method-safe SOMA object bank exists and contains native carrier/view-mask support, but current artifacts "
                "still do not contain mesh_vertex_id/scene_point_id/pre_points or an equivalent ScanNet AP mask join key. "
                "The object bank is the method inference output; ScanNet AP export still needs a method-safe native "
                "carrier-to-evaluator adapter, not RGB-D/pose/mesh projection."
            )
        else:
            blocker = (
                "Method-safe SOMA object bank exists and contains verified view-mask support, but no verified native "
                "carrier/3D support or ScanNet AP mask join key is available. The object bank is the method inference "
                "output; ScanNet AP export still needs a method-safe view-mask-to-native-3D/evaluator adapter, not "
                "RGB-D/pose/mesh projection in the method path."
            )
    elif object_bank_available:
        status = "object_bank_ready_missing_verified_object_support"
        blocker = (
            "Method-safe SOMA object bank exists, but no verified object-to-view-mask or object-to-native-carrier "
            "support rows are available. Same-text component_id overlaps between v64r2 L11 object/material rows and "
            "v53 L6 objectlet carrier rows are recorded only as unverified candidates, not as support."
        )
    else:
        status = "blocked_missing_native_component_to_scannet_ap_mask_join_key"
        blocker = (
            "Native component/carrier support exists, but current method-safe artifacts do not contain "
            "mesh_vertex_id/scene_point_id/pre_points or an equivalent ScanNet AP mask join key. "
            "Using RGB-D/pose/mesh projection would be diagnostic-only under the v65 plan."
        )

    summary = {
        "phase": "v65_a9_method_safe_materializer",
        "status": status,
        "soma_object_bank_available": object_bank_available,
        "soma_object_count": object_bank_summary.get("object_count"),
        "soma_object_material_assignment_count": object_bank_summary.get("material_assignment_count"),
        "soma_object_support_row_count": object_bank_summary.get("object_support_row_count"),
        "soma_objects_with_any_support_count": object_bank_summary.get("objects_with_any_support_count"),
        "soma_native_carrier_support_row_count": object_bank_summary.get("native_carrier_support_row_count"),
        "soma_object_support_coverage_ratio": object_bank_summary.get("object_support_coverage_ratio"),
        "method_safe_native_support_available": method_safe_support_available,
        "method_safe_ap_available": method_safe_ap_available,
        "native_component_field_available": bool(native_contract.get("component_level_available")),
        "native_carrier_level_available": bool(native_contract.get("carrier_level_available")),
        "component_to_carrier_mapping_available": bool(native_contract.get("component_to_carrier_mapping_available")),
        "v62_component_level_field_available": bool(v62_summary.get("component_level_field_available")),
        "v62_carrier_level_field_available": bool(v62_summary.get("carrier_level_field_available")),
        "scan_ap_join_key_available": has_ap_join_key,
        "present_ap_join_key_artifacts": [row["artifact"] for row in join_key_rows],
        "forbidden_route_not_used": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation": False,
        "uses_rgbd_pose_mesh_for_export": False,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "support_scope": "NATIVE_CARRIER_SUPPORT_NO_SCANNET_AP_MASK",
        "support_policy": "method-safe native D4RT carrier/component support; no ScanNet AP mask materializer",
        "input_frame_policy": "v47/v53 native carrier local 32-frame window support; no Stream3D-comparable AP row generated",
        "blocker": blocker,
        "repair_direction_attempted": [
            "Rebuilt v53 native carrier materialization from v47 carrier/mask tables.",
            "Exported v65 SOMA method-safe object bank as object/history rows with material assignments and native support rows.",
            "Audited v62/v64r2 native contract and v53 carrier rows for ScanNet AP mask join keys.",
            "Rejected RGB-D/pose/mesh bridge as method AP because v65 forbids it for method tables.",
        ],
        "required_future_change": (
            "Add a method-safe native carrier/object-bank to AP-mask adapter that emits ScanNet vertex ids, method-safe "
            "D4RT 3D points, or a native carrier evaluator; do not use GT depth/pose/mesh or eval-Sim3 for method "
            "prediction/export."
        ),
        "source_hashes": source_hashes,
        "native_carrier_attempt_summary": rel(attempt_summary_path) if attempt_summary_path.exists() else "",
        "native_carrier_attempt_summary_sha256": sha256_file(attempt_summary_path),
        "soma_object_bank_summary": object_bank_paths["summary"],
        "soma_object_bank_rows": object_bank_paths["object_bank_rows"],
        "soma_object_material_rows": object_bank_paths["object_material_rows"],
        "soma_object_support_rows": object_bank_paths["object_support_rows"],
        "evidence_rows_file": rel(root / "a9_materializer_evidence_rows.csv"),
        "summary_file": rel(root / "a9_materializer_summary.json"),
        "gate": {
            "soma_object_bank_available": object_bank_available,
            "soma_object_support_available": object_bank_support_available,
            "native_component_field_available": bool(native_contract.get("component_level_available")),
            "method_safe_native_support_available": method_safe_support_available,
            "scan_ap_join_key_available": has_ap_join_key,
            "does_not_use_forbidden_rgbd_pose_mesh": True,
            "method_safe_ap_available": method_safe_ap_available,
        },
    }
    summary["gate"]["pass"] = method_safe_ap_available
    return {"summary": summary, "evidence_rows": evidence_rows}


def write_v65_a9_method_safe_materializer(output_root: str | Path, payload: dict[str, Any]) -> None:
    write_standard_outputs(
        output_root,
        {
            "a9_materializer_summary.json": payload["summary"],
            "a9_materializer_evidence_rows.csv": payload["evidence_rows"],
        },
    )


def _artifact_evidence_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_paths = dict(NATIVE_INPUTS)
    candidate_paths["v65_native_carrier_attempt_rows"] = str(root / "native_carrier_attempt" / "objectlet_native_carrier_rows.csv")
    candidate_paths["v65_soma_object_bank_summary"] = str(root / "soma_object_bank_attempt" / "soma_object_bank_summary.json")
    candidate_paths["v65_soma_object_bank_rows"] = str(root / "soma_object_bank_attempt" / "soma_object_bank_rows.csv")
    candidate_paths["v65_soma_object_material_rows"] = str(root / "soma_object_bank_attempt" / "soma_object_material_rows.csv")
    candidate_paths["v65_soma_object_support_rows"] = str(root / "soma_object_bank_attempt" / "soma_object_support_rows.csv")
    for name, path_text in candidate_paths.items():
        path = project(path_text)
        if not path.exists():
            rows.append(
                {
                    "artifact": name,
                    "path": rel(path),
                    "exists": False,
                    "row_count": 0,
                    "columns": [],
                    "present_ap_join_keys": [],
                    "sha256": "",
                }
            )
            continue
        columns: list[str] = []
        row_count = 0
        if path.suffix == ".csv":
            columns = _csv_header(path)
            row_count = _line_count(path) - 1 if columns else 0
        elif path.suffix == ".json":
            payload = load_dict(path)
            columns = sorted(str(key) for key in payload.keys())
            row_count = 1 if payload else 0
        else:
            columns = []
        present = sorted(AP_JOIN_KEYS.intersection(columns))
        rows.append(
            {
                "artifact": name,
                "path": rel(path),
                "exists": True,
                "row_count": row_count,
                "columns": columns,
                "present_ap_join_keys": present,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
    return first.split(",") if first else []


def _line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _line in handle:
            count += 1
    return count


__all__ = [
    "A9_ROOT",
    "build_v65_a9_method_safe_materializer",
    "write_v65_a9_method_safe_materializer",
]
