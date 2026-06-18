from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PHASE_C_GATE = {
    "purity": 0.85,
    "coverage_at_010": 0.75,
    "same_frame_cannot_link_violation": 0.05,
    "scene0081_purity": 0.80,
}
OBJECT_GATE = {
    "ARI": 0.40,
    "purity": 0.85,
    "completeness": 0.50,
    "scene0081_ARI": 0.20,
    "unknown_tube_ratio_max": 0.40,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _phase_c_pass(row: dict[str, Any]) -> bool:
    return bool(
        (row.get("purity") or 0.0) >= PHASE_C_GATE["purity"]
        and (row.get("coverage_at_010") or 0.0) >= PHASE_C_GATE["coverage_at_010"]
        and (row.get("same_frame_cannot_link_violation") or 0.0) <= PHASE_C_GATE["same_frame_cannot_link_violation"]
        and (row.get("scene0081_purity") or 0.0) >= PHASE_C_GATE["scene0081_purity"]
    )


def _object_gate_pass(row: dict[str, Any]) -> bool:
    return bool(
        (row.get("ARI") or 0.0) >= OBJECT_GATE["ARI"]
        and (row.get("purity") or 0.0) >= OBJECT_GATE["purity"]
        and (row.get("completeness") or 0.0) >= OBJECT_GATE["completeness"]
        and (row.get("scene0081_ARI") or 0.0) >= OBJECT_GATE["scene0081_ARI"]
        and (row.get("unknown_tube_ratio") or 1.0) <= OBJECT_GATE["unknown_tube_ratio_max"]
    )


def _is_forbidden_oracle_row(row: dict[str, Any]) -> bool:
    label = str(row.get("label") or "").lower()
    status = str(row.get("status") or "").lower()
    return bool(
        row.get("is_diagnostic_only")
        or "oracle" in label
        or "forbidden" in label
        or "oracle" in status
        or "forbidden" in status
    )


def _source_row_from_v36(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "family": "v36_external_mask_source",
            "label": label,
            "path": str(path),
            "status": "missing",
            "phaseC_gate_pass": False,
        }
    data = _read_json(path)
    mixed = data.get("mixed_region_rate")
    row = {
        "family": "v36_external_mask_source",
        "label": label,
        "path": str(path),
        "status": "ok" if data.get("integration_pass") else str(data.get("failure_stage") or "not_integrated"),
        "purity": None if mixed is None else 1.0 - float(mixed),
        "coverage_at_010": data.get("GT_object_coverage@0.10"),
        "coverage_at_025": data.get("GT_object_coverage@0.25"),
        "scene0081_purity": None if data.get("scene") != "scene0081_01" or mixed is None else 1.0 - float(mixed),
        "same_frame_cannot_link_violation": 0.0,
        "source_available": bool(data.get("integration_pass")),
        "note": data.get("note"),
    }
    row["phaseC_gate_pass"] = _phase_c_pass(row)
    return row


def _source_rows_from_same_frame(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return [
            {
                "family": "v37_same_frame_objectlets",
                "label": label,
                "path": str(path),
                "status": "missing",
                "phaseC_gate_pass": False,
            }
        ]
    rows = []
    for raw in _read_csv(path):
        row = {
            "family": "v37_same_frame_objectlets",
            "label": f"{label}:{raw.get('variant')}",
            "path": str(path),
            "status": raw.get("status"),
            "purity": _float(raw, "purity"),
            "coverage_at_010": _float(raw, "GT_cov@0.10"),
            "coverage_at_005": _float(raw, "GT_cov@0.05"),
            "mixed_rate": _float(raw, "mixed_rate"),
            "scene0081_purity": _float(raw, "scene0081_seed_purity"),
            "scene0081_coverage": _float(raw, "scene0081"),
            "same_frame_cannot_link_violation": 0.0,
            "source_available": raw.get("status", "").startswith("ok") or raw.get("status", "").startswith("proxy"),
            "note": "Imported stronger-source same-frame seed summary.",
        }
        row["phaseC_gate_pass"] = _phase_c_pass(row)
        rows.append(row)
    return rows


def _object_rows_from_tube_assignment(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return [
            {
                "family": "v37_tube_assignment",
                "label": label,
                "path": str(path),
                "status": "missing",
                "object_gate_pass": False,
            }
        ]
    rows = []
    for raw in _read_csv(path):
        row = {
            "family": "v37_tube_assignment",
            "label": f"{label}:{raw.get('stage')}",
            "path": str(path),
            "stage": raw.get("stage"),
            "ARI": _float(raw, "ARI"),
            "purity": _float(raw, "purity"),
            "completeness": _float(raw, "completeness"),
            "scene0081_ARI": _float(raw, "scene0081_ARI"),
            "unknown_tube_ratio": _float(raw, "unknown_tube_ratio"),
            "same_frame_cannot_link_violations": int(float(raw.get("same_frame_cannot_link_violations") or 0)),
            "status": "imported",
            "note": "Imported stronger-source tube/object gate summary.",
        }
        row["object_gate_pass"] = _object_gate_pass(row)
        rows.append(row)
    return rows


def _oracle_rows_from_v35(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return [
            {
                "family": "v35_mask_source_oracle",
                "label": "v35_mask_source_oracle",
                "path": str(path),
                "status": "missing",
                "object_gate_pass": False,
            }
        ]
    rows = []
    for raw in _read_json(path):
        if raw.get("scene") != "ALL":
            continue
        row = {
            "family": "v35_mask_source_oracle",
            "label": f"v35:{raw.get('pool')}",
            "path": str(path),
            "status": "imported_diagnostic_oracle",
            "ARI": raw.get("oracle_ARI"),
            "purity": raw.get("oracle_purity"),
            "completeness": raw.get("oracle_completeness"),
            "scene0081_ARI": raw.get("scene0081_oracle_ARI"),
            "GT_with_best_IoU_ge_050": raw.get("GT_with_best_IoU_ge_050"),
            "proposal_count": raw.get("proposal_count"),
            "is_diagnostic_only": True,
            "note": "GT oracle proposal pool diagnostic; forbidden as method result.",
        }
        row["object_gate_pass"] = bool(
            (row.get("ARI") or 0.0) >= OBJECT_GATE["ARI"]
            and (row.get("purity") or 0.0) >= OBJECT_GATE["purity"]
            and (row.get("completeness") or 0.0) >= OBJECT_GATE["completeness"]
            and (row.get("scene0081_ARI") or 0.0) >= OBJECT_GATE["scene0081_ARI"]
        )
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    source_rows = [
        _source_row_from_v36(
            root / "outputs/audit/v36_external_mask_source/dinov2_maskcut/sample8/summary.json",
            "v36_dinov2_maskcut_sample8_scene0081",
        ),
        _source_row_from_v36(
            root / "outputs/audit/v36_external_mask_source/efficientsam3/sample8/summary.json",
            "v36_efficientsam3_sample8_scene0081",
        ),
        _source_row_from_v36(
            root / "outputs/audit/v36_external_mask_source/sam3/frame0/summary.json",
            "v36_sam3_frame0_scene0081",
        ),
        _source_row_from_v36(
            root / "outputs/audit/v36_external_mask_source/watershed/probe5_full32/summary.json",
            "v36_watershed_probe5_full32",
        ),
    ]
    source_specs = [
        (
            "v37_dino_compact060_probe5",
            "outputs/audit/v37_selfcheck_dino_compact060_probe5/v37_same_frame_objectlets/same_frame_seed_summary.csv",
            "outputs/audit/v37_selfcheck_dino_compact060_probe5/v37_tube_assignment/tube_assignment_summary.csv",
        ),
        (
            "v37_dino_compact060_small_unknown_probe5",
            "outputs/audit/v37_selfcheck_dino_compact060_small_unknown_probe5/v37_same_frame_objectlets/same_frame_seed_summary.csv",
            "outputs/audit/v37_selfcheck_dino_compact060_small_unknown_probe5/v37_tube_assignment/tube_assignment_summary.csv",
        ),
        (
            "v37_dinov2_sample8_probe5",
            "outputs/audit/v37_selfcheck_dinov2_sample8_probe5/v37_same_frame_objectlets/same_frame_seed_summary.csv",
            "outputs/audit/v37_selfcheck_dinov2_sample8_probe5/v37_tube_assignment/tube_assignment_summary.csv",
        ),
        (
            "v37_efficientsam3_sample32_scene0081",
            "outputs/audit/v37_selfcheck_efficientsam3_sample32_scene0081/v37_same_frame_objectlets/same_frame_seed_summary.csv",
            "outputs/audit/v37_selfcheck_efficientsam3_sample32_scene0081/v37_tube_assignment/tube_assignment_summary.csv",
        ),
    ]
    object_rows = []
    for label, same_frame, tube_assignment in source_specs:
        source_rows.extend(_source_rows_from_same_frame(root / same_frame, label))
        object_rows.extend(_object_rows_from_tube_assignment(root / tube_assignment, label))
    object_rows.extend(
        _oracle_rows_from_v35(
            root / "outputs/audit/v35_mask_source_audit/proposal_rebuild_conda/v35_mask_source_rebuild_conda_oracle_summary.json"
        )
    )

    best_phase_c = max(source_rows, key=lambda row: float(row.get("purity") or -1.0), default={})
    best_phase_c_balanced = max(
        source_rows,
        key=lambda row: min(float(row.get("purity") or 0.0), float(row.get("coverage_at_010") or 0.0), float(row.get("scene0081_purity") or 0.0)),
        default={},
    )
    non_oracle_object_rows = [row for row in object_rows if not _is_forbidden_oracle_row(row)]
    best_object = max(non_oracle_object_rows, key=lambda row: float(row.get("ARI") or -1.0), default={})
    best_oracle_object = max(
        (row for row in object_rows if _is_forbidden_oracle_row(row)),
        key=lambda row: float(row.get("ARI") or -1.0),
        default={},
    )
    gate = {
        "any_phaseC_source_gate_pass": any(bool(row.get("phaseC_gate_pass")) for row in source_rows),
        "any_object_gate_pass_all_rows": any(bool(row.get("object_gate_pass")) for row in object_rows),
        "any_non_oracle_object_gate_pass": any(
            bool(row.get("object_gate_pass")) and not _is_forbidden_oracle_row(row) for row in object_rows
        ),
        "best_phaseC_by_purity": best_phase_c.get("label"),
        "best_phaseC_purity": best_phase_c.get("purity"),
        "best_phaseC_coverage_at_010": best_phase_c.get("coverage_at_010"),
        "best_phaseC_scene0081_purity": best_phase_c.get("scene0081_purity"),
        "best_phaseC_balanced": best_phase_c_balanced.get("label"),
        "best_phaseC_balanced_purity": best_phase_c_balanced.get("purity"),
        "best_phaseC_balanced_coverage_at_010": best_phase_c_balanced.get("coverage_at_010"),
        "best_phaseC_balanced_scene0081_purity": best_phase_c_balanced.get("scene0081_purity"),
        "best_object_by_ARI": best_object.get("label"),
        "best_object_ARI": best_object.get("ARI"),
        "best_object_purity": best_object.get("purity"),
        "best_object_completeness": best_object.get("completeness"),
        "best_object_scene0081_ARI": best_object.get("scene0081_ARI"),
        "best_forbidden_oracle_object_by_ARI": best_oracle_object.get("label"),
        "best_forbidden_oracle_object_ARI": best_oracle_object.get("ARI"),
        "best_forbidden_oracle_object_purity": best_oracle_object.get("purity"),
        "best_forbidden_oracle_object_completeness": best_oracle_object.get("completeness"),
        "best_forbidden_oracle_object_scene0081_ARI": best_oracle_object.get("scene0081_ARI"),
    }
    final_status = (
        "GO_STRONGER_SOURCE_FOUND"
        if gate["any_phaseC_source_gate_pass"] or gate["any_non_oracle_object_gate_pass"]
        else "NO_GO_STRONGER_MASK_SOURCE_AUDIT_FAILED_GATES"
    )
    manifest = {
        "phase": "v39_stronger_mask_source_audit",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "uses_frozen_visual_backbone": True,
        "visual_backbone_name": "DINO/DINOv2/SAM/EfficientSAM diagnostics imported when available",
        "mask_source": "existing v35/v36/v37 stronger-source artifacts",
        "object_birth_source": "diagnostic import only",
        "d4rt_role": "support/diagnostic only",
        "geometry_field": "none_for_method_prediction",
        "coordinate_frame": "artifact-native summaries",
        "alignment_source": "existing audit artifacts with GT diagnostic labels",
    }
    summary = {
        **manifest,
        "final_status": final_status,
        "stronger_source_gate": gate,
        "source_rows": source_rows,
        "object_rows": object_rows,
        "notes": [
            "This audit follows v39 Stop 2/Stop 6 recommendations by checking stronger mask/source artifacts already present in the workspace.",
            "No AP/export or held-out method success is claimed from these diagnostic imports.",
            "Forbidden GT/oracle upper-bound rows are reported separately and cannot trigger GO status.",
            "High-purity DINOv2/EfficientSAM rows collapse coverage/unknown; DINO compact rows are close but still miss purity.",
        ],
    }
    _write_json(output_root / "stronger_mask_source_manifest.json", manifest)
    _write_json(output_root / "stronger_mask_source_summary.json", summary)
    _write_csv(output_root / "stronger_mask_source_matrix.csv", source_rows)
    _write_csv(output_root / "stronger_object_gate_matrix.csv", object_rows)
    md = [
        "# Stream4D v39 Stronger Mask Source Audit",
        "",
        f"`final_status={final_status}`",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key, value in gate.items():
        md.append(f"| {key} | {value} |")
    md.append("")
    md.append("Diagnostic-only import; no v39 method success is claimed.")
    (output_root / "stronger_mask_source_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit existing stronger mask/source artifacts for v39 Phase C retry evidence.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--output-root", default="outputs/audit/v39_stronger_mask_source")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
