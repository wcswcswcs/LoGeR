from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_matrix(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["variant"]: row for row in csv.DictReader(handle)}


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _diff_actual(summary: dict[str, Any], group: str, key: str) -> float:
    return float(summary[group][key]["actual"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    repo_root = root.parent
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    phase_a = _read_json(root / "outputs/audit/v38_phaseA_freeze/phaseA_freeze_summary.json")
    export_trace = _read_json(root / "outputs/audit/v38_export_trace/export_trace_summary.json")
    oracle_matrix = _read_matrix(root / "outputs/audit/v38_oracle_attribution/oracle_attribution_matrix.csv")

    required_docs = {
        "v38_report": repo_root / "docs/stream4d_v38_object_materialization_report.md",
        "v38_recap": repo_root / "docs/stream4d_v38_实验结果复盘.md",
        "v37_report": repo_root / "docs/stream4d_v37_temporal_curriculum_masklet_report.md",
    }
    doc_checks = {name: {"path": str(path), "exists": path.exists()} for name, path in required_docs.items()}

    c5 = oracle_matrix["C5_oracle_vertex_owner"]
    c6 = oracle_matrix["C6_oracle_gt_mask_by_object"]
    c7 = oracle_matrix["C7_same_support_stream3d"]

    facts = {
        "v37_3d_gate_pass": phase_a["f31_status"] == "GO_3D_TEMPORAL_CURRICULUM",
        "v37_4d_gate_pass": phase_a["i4_status"] == "GO_4D_MEMORY",
        "v37_stage": phase_a["f31_stage"],
        "v37_i4_variant": phase_a["i4_variant"],
        "v37_ARI": _diff_actual(phase_a, "f31_diff", "ARI"),
        "v37_purity": _diff_actual(phase_a, "f31_diff", "purity"),
        "v37_completeness": _diff_actual(phase_a, "f31_diff", "completeness"),
        "v37_unknown_tube_ratio": _diff_actual(phase_a, "f31_diff", "unknown_tube_ratio"),
        "v37_scene0081_ARI": _diff_actual(phase_a, "f31_diff", "scene0081_ARI"),
        "v37_4D_ARI": _diff_actual(phase_a, "i4_diff", "4D_ARI"),
        "v37_4D_purity": _diff_actual(phase_a, "i4_diff", "4D_purity"),
        "v37_4D_completeness": _diff_actual(phase_a, "i4_diff", "4D_completeness"),
        "v37_4D_temporal_span_mean": _diff_actual(phase_a, "i4_diff", "temporal_span_mean"),
        "v38_raw_AP": _diff_actual(phase_a, "ap_diff", "raw_AP"),
        "v38_raw_AP50": _diff_actual(phase_a, "ap_diff", "raw_AP50"),
        "v38_raw_AP25": _diff_actual(phase_a, "ap_diff", "raw_AP25"),
        "v38_best_postprocess_AP": _diff_actual(phase_a, "ap_diff", "best_postprocess_AP"),
        "v38_best_postprocess_AP50": _diff_actual(phase_a, "ap_diff", "best_postprocess_AP50"),
        "v38_best_postprocess_AP25": _diff_actual(phase_a, "ap_diff", "best_postprocess_AP25"),
        "v38_C5_oracle_vertex_owner_AP": _float(c5, "AP"),
        "v38_C5_oracle_vertex_owner_AP50": _float(c5, "AP50"),
        "v38_C5_oracle_vertex_owner_AP25": _float(c5, "AP25"),
        "v38_C6_oracle_gt_mask_by_object_AP": _float(c6, "AP"),
        "v38_C6_oracle_gt_mask_by_object_AP50": _float(c6, "AP50"),
        "v38_C6_oracle_gt_mask_by_object_AP25": _float(c6, "AP25"),
        "same_support_stream3d_AP": _float(c7, "AP"),
        "same_support_stream3d_AP50": _float(c7, "AP50"),
        "same_support_stream3d_AP25": _float(c7, "AP25"),
        "mean_predictions_per_scene": float(export_trace["mean_num_predictions"]),
        "mean_export_conflict_rate": float(export_trace["mean_export_conflict_rate"]),
        "prediction_trace_row_count": int(export_trace["prediction_trace_row_count"]),
        "exported_prediction_count": int(export_trace["exported_prediction_count"]),
        "v38_export_trace_AP": float(export_trace["eval_result"]["metrics"]["AP"]),
    }

    retirement_reasons = [
        "mean_predictions_per_scene is far above the v39 <=300 object identity gate",
        "mean_export_conflict_rate is far above the v39 <=0.10 conflict gate",
        "v38 best postprocess AP remains far below same-support Stream3D",
        "v38 oracle object/candidate rows remain below same-support Stream3D, indicating candidate multiplicity/representation gap",
    ]
    manifest = {
        "phase": "v39_phaseA_failure_lock",
        "candidate_first_route_retired": True,
        "retired_reason": "object_candidate_multiplicity_and_AP_upper_bound_gap",
        "forbidden_to_continue_as_main_route": True,
        "retirement_reasons": retirement_reasons,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": False,
        "uses_frozen_visual_backbone": False,
        "visual_backbone_name": None,
        "mask_source": "none_phaseA_lock_only",
        "object_birth_source": "none_phaseA_lock_only",
        "d4rt_role": "none_phaseA_lock_only",
        "geometry_field": "none_phaseA_lock_only",
        "coordinate_frame": "none_phaseA_lock_only",
        "alignment_source": "existing_v37_v38_audit_artifacts",
        "source_artifacts": [
            "docs/stream4d_v38_object_materialization_report.md",
            "docs/stream4d_v38_实验结果复盘.md",
            "docs/stream4d_v37_temporal_curriculum_masklet_report.md",
            "outputs/audit/v38_phaseA_freeze/phaseA_freeze_summary.json",
            "outputs/audit/v38_export_trace/export_trace_summary.json",
            "outputs/audit/v38_oracle_attribution/oracle_attribution_matrix.csv",
        ],
    }
    phase_pass = (
        bool(facts["v37_3d_gate_pass"])
        and bool(facts["v37_4d_gate_pass"])
        and all(item["exists"] for item in doc_checks.values())
        and manifest["candidate_first_route_retired"]
        and manifest["forbidden_to_continue_as_main_route"]
    )
    summary = {
        "phase": "v39_phaseA_failure_lock",
        "phaseA_pass": phase_pass,
        "facts": facts,
        "doc_checks": doc_checks,
        "old_route_retired_manifest": "outputs/audit/v39_phaseA_failure_lock/old_route_retired_manifest.json",
        "notes": [
            "This phase locks previously measured v37/v38 facts; it does not generate predictions.",
            "GT/oracle values loaded here are diagnostic-only and forbidden for method tables.",
        ],
    }
    _write_json(output_root / "failure_lock.json", summary)
    _write_json(output_root / "old_route_retired_manifest.json", manifest)
    md_lines = [
        "# Stream4D v39 Phase A Failure Lock",
        "",
        "| item | value |",
        "|---|---:|",
    ]
    for key, value in facts.items():
        md_lines.append(f"| {key} | {value} |")
    md_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`phaseA_pass={phase_pass}`",
            "",
            "`candidate_first_route_retired=true`",
            "",
            "`forbidden_to_continue_as_main_route=true`",
        ]
    )
    (output_root / "failure_lock.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v39 Phase A failure lock for candidate-first route retirement.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--output-root", default="outputs/audit/v39_phaseA_failure_lock")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
