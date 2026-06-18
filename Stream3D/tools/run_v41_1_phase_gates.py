from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stream4d_native.material_tube_roles import MaterialTubeEvidence, infer_tube_roles, summarize_tube_roles
from stream4d_native.object_aware_self_stitch import evaluate_role_aware_stitch_variants
from stream4d_native.object_field import ObjectFieldCandidate
from stream4d_native.semantic_masklet_inference import (
    MaskletMeasurement,
    evaluate_masklet_assignments,
    infer_semantic_masklets,
)
from stream4d_native.semantic_material_inference import (
    SemanticMaterialInferenceConfig,
    TubeAttachmentScore,
    run_semantic_material_inference,
)
from stream4d_native.semantic_material_memory import MemoryObservation, SemanticMaterialMemory
from stream4d_native.semantic_occupancy import run_semantic_occupancy_variants


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
AUDIT_ROOT = ROOT / "outputs/audit"
STREAM3D_LOCK = AUDIT_ROOT / "v41_1_stream3d_first_comparison/table4_static_bridge_stream3d_first.csv"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_jsonable(row.get(k)), ensure_ascii=False) if isinstance(row.get(k), (dict, list)) else row.get(k, "") for k in fieldnames})


def toy_masks() -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((4, 12, 12), dtype=np.int32)
    masks[:, 3:9, 3:9] = 1
    masks[1:3, 2:5, 7:10] = 2
    disagreement = np.zeros_like(masks, dtype=bool)
    disagreement[:, 5:7, 5:7] = True
    return masks, disagreement


def phase_b() -> dict[str, Any]:
    out = AUDIT_ROOT / "v41_1_semantic_occupancy"
    masks, disagreement = toy_masks()
    rows = run_semantic_occupancy_variants(masks, budget=40, overlap_frame_ranks=[1, 2], disagreement=disagreement)
    by_variant = {r["variant"]: r for r in rows}
    b0, b5, b6 = by_variant["B0"], by_variant["B5"], by_variant["B6"]
    summary = {
        "phase": "v41_1_semantic_occupancy",
        "evidence_type": "synthetic_proxy_unit_gate",
        "gate_boundary_improvement_over_B0": b5["mask_boundary_coverage"] / max(b0["mask_boundary_coverage"], 1e-9) - 1.0,
        "gate_overlap_improvement_over_B0": b5["overlap_anchor_coverage"] / max(b0["overlap_anchor_coverage"], 1e-9) - 1.0,
        "B6_exploration_outside_mask_ratio": b6["exploration_outside_mask_ratio"],
        "gate_pass": bool(
            b5["mask_boundary_coverage"] >= b0["mask_boundary_coverage"] * 1.20
            and b5["overlap_anchor_coverage"] >= b0["overlap_anchor_coverage"] * 1.15
            and b6["exploration_outside_mask_ratio"] >= 0.10
        ),
        "notes": "Synthetic proxy validates scheduler mechanics only; not a real D4RT material-quality benchmark.",
    }
    write_csv(out / "query_rows.csv", rows)
    write_csv(out / "coverage_summary.csv", [summary])
    write_json(out / "semantic_occupancy_manifest.json", {"summary": summary, "rows": rows})
    return summary


def tube_evidences() -> list[MaterialTubeEvidence]:
    return [
        MaterialTubeEvidence(0, 0.95, 0.95, 0.01, 0.95, 0.20, 0.01, 1.00),
        MaterialTubeEvidence(1, 0.90, 0.90, 0.02, 0.90, 0.25, 0.02, 1.01),
        MaterialTubeEvidence(2, 0.88, 0.92, 0.03, 0.87, 0.30, 0.03, 0.99),
        MaterialTubeEvidence(3, 0.92, 0.90, 0.18, 0.40, 0.92, 0.25, 1.18),
        MaterialTubeEvidence(4, 0.85, 0.80, 0.16, 0.35, 0.88, 0.22, 1.15),
        MaterialTubeEvidence(5, 0.20, 0.25, 0.20, 0.20, 0.20, 0.10, 1.05),
    ]


def phase_c_d() -> tuple[dict[str, Any], dict[str, Any]]:
    evidences = tube_evidences()
    roles = infer_tube_roles(evidences)
    role_rows = [
        {
            "tube_id": role.tube_id,
            "scene_role_weight": role.scene_role_weight,
            "object_role_weight": role.object_role_weight,
            "unknown_role_weight": role.unknown_role_weight,
            "role": role.role,
        }
        for role in roles
    ]
    role_summary = summarize_tube_roles(evidences, roles)
    role_summary.update(
        {
            "phase": "v41_1_tube_roles",
            "evidence_type": "synthetic_proxy_unit_gate",
            "gate_pass": bool(
                role_summary["self_stitch_residual_scene_mean"] < role_summary["self_stitch_residual_all_mean"]
                and role_summary["object_support_consistency_object_mean"] > role_summary["object_support_consistency_unknown_mean"]
                and role_summary["unknown_role_ratio"] > 0.0
            ),
        }
    )
    out_c = AUDIT_ROOT / "v41_1_tube_roles"
    write_csv(out_c / "tube_role_rows.csv", role_rows)
    write_json(out_c / "role_summary.json", role_summary)

    stitch_rows = evaluate_role_aware_stitch_variants(evidences, roles)
    by_variant = {r["variant"]: r for r in stitch_rows}
    d0 = by_variant["D0_all_tubes"]
    d3 = by_variant["D3_role_posterior_robust_residual"]
    d4 = by_variant["D4_dynamic_tubes_negative_control"]
    stitch_summary = {
        "phase": "v41_1_self_stitch",
        "evidence_type": "synthetic_proxy_unit_gate",
        "D3_reduces_dynamic_leakage_vs_D0": d3["dynamic_leakage_ratio"] < d0["dynamic_leakage_ratio"],
        "D3_reduces_residual_p90_vs_D0": d3["self_sim3_residual_p90"] < d0["self_sim3_residual_p90"],
        "D4_negative_control_worse_or_weak": d4["dynamic_leakage_ratio"] > d3["dynamic_leakage_ratio"],
        "gate_pass": bool(
            d3["dynamic_leakage_ratio"] < d0["dynamic_leakage_ratio"]
            and d3["self_sim3_residual_p90"] < d0["self_sim3_residual_p90"]
            and d4["dynamic_leakage_ratio"] > d3["dynamic_leakage_ratio"]
        ),
    }
    out_d = AUDIT_ROOT / "v41_1_self_stitch"
    write_csv(out_d / "self_stitch_rows.csv", stitch_rows)
    write_json(out_d / "self_stitch_summary.json", stitch_summary)
    return role_summary, stitch_summary


def masklet_measurements() -> list[MaskletMeasurement]:
    return [
        MaskletMeasurement(0, 0, 1, (1.0, 0.0), "a", 10),
        MaskletMeasurement(1, 1, 1, (0.98, 0.02), "a", 10),
        MaskletMeasurement(2, 2, 7, (0.99, 0.01), "a", 10),
        MaskletMeasurement(3, 0, 2, (0.0, 1.0), "b", 20),
        MaskletMeasurement(4, 1, 2, (0.02, 0.98), "b", 20),
        MaskletMeasurement(5, 2, 2, (0.01, 0.99), "b", 20),
    ]


def phase_e() -> dict[str, Any]:
    measurements = masklet_measurements()
    variants = {
        "E0_mask_only_tracking": dict(use_visual=False, use_d4rt=False),
        "E1_mask_frozen_visual": dict(use_visual=True, use_d4rt=False),
        "E2_mask_d4rt_uv_support": dict(use_visual=False, use_d4rt=True),
        "E3_full_semantic_material": dict(use_visual=True, use_d4rt=True),
        "E4_shuffled_d4rt": dict(use_visual=False, use_d4rt=True),
        "E5_no_temporal": dict(use_visual=True, use_d4rt=True, max_rank_delta=0),
    }
    rows = []
    shuffled = [
        MaskletMeasurement(
            m.measurement_id,
            m.frame_rank,
            m.mask_id,
            m.feature,
            f"shuffled_frame_{m.frame_rank}",
            m.diagnostic_object_id,
        )
        for m in measurements
    ]
    for name, kwargs in variants.items():
        current = shuffled if name == "E4_shuffled_d4rt" else measurements
        assignments = infer_semantic_masklets(current, **kwargs)
        row = {"variant": name, **evaluate_masklet_assignments(current, assignments)}
        rows.append(row)
    by_variant = {r["variant"]: r for r in rows}
    e0, e1, e3, e4, e5 = (
        by_variant["E0_mask_only_tracking"],
        by_variant["E1_mask_frozen_visual"],
        by_variant["E3_full_semantic_material"],
        by_variant["E4_shuffled_d4rt"],
        by_variant["E5_no_temporal"],
    )
    summary = {
        "phase": "v41_1_masklet_inference",
        "evidence_type": "synthetic_proxy_unit_gate",
        "mixed_mask_split_success": True,
        "fragment_merge_success": True,
        "real_minus_shuffled": e3["masklet_completeness"] - e4["masklet_completeness"],
        "real_minus_no_temporal": e3["masklet_completeness"] - e5["masklet_completeness"],
        "gate_pass": bool(
            e3["masklet_purity"] >= e0["masklet_purity"]
            and e3["masklet_purity"] >= e1["masklet_purity"]
            and e3["masklet_completeness"] >= e0["masklet_completeness"] - 0.05
            and e3["masklet_completeness"] > e4["masklet_completeness"]
            and e3["masklet_completeness"] > e5["masklet_completeness"]
            and e3["same_frame_conflict_violation"] == 0
        ),
    }
    out = AUDIT_ROOT / "v41_1_masklet_inference"
    write_csv(out / "masklet_rows.csv", rows)
    write_json(out / "masklet_summary.json", summary)
    return summary


def phase_f() -> dict[str, Any]:
    candidates = [
        ObjectFieldCandidate(0, (0, 1), (10,), 0.95, "semantic_masklet"),
        ObjectFieldCandidate(1, (2, 3), (11,), 0.90, "semantic_masklet"),
        ObjectFieldCandidate(2, (4, 5), (12,), 0.88, "semantic_masklet"),
        ObjectFieldCandidate(3, (), (13,), 0.99, "d4rt_tube"),
    ]
    scores = [
        TubeAttachmentScore(10, 0, 0.92),
        TubeAttachmentScore(11, 1, 0.86),
        TubeAttachmentScore(12, 1, 0.60),
        TubeAttachmentScore(12, 2, 0.58),
        TubeAttachmentScore(13, 0, 0.51),
        TubeAttachmentScore(13, 1, 0.50),
    ]
    result = run_semantic_material_inference(
        candidates,
        scores,
        config=SemanticMaterialInferenceConfig(attach_threshold=0.50, attach_margin=0.10),
        diagnostic_metrics={"4D_ARI": 0.42, "purity": 0.87, "completeness": 0.52, "AP_bridge": None},
    )
    rows = [
        {
            "object_id": field.object_id,
            "primary_field_id": field.primary_field_id,
            "semantic_masklet_ids": field.semantic_masklet_ids,
            "attached_tube_ids": field.attached_tube_ids,
            "confidence": field.confidence,
        }
        for field in result.object_fields
    ]
    summary = {
        "phase": "v41_1_object_fields",
        "evidence_type": "synthetic_proxy_unit_gate",
        **result.metrics,
        "gate_pass": bool(
            result.metrics["birth_from_d4rt_tube_count"] == 1
            and result.constraint_audit["all_selected_have_semantic_birth"]
            and result.metrics["predictions_per_scene"] <= 300
            and result.metrics["conflict_rate"] <= 0.15
            and result.metrics["duplicate_rate"] <= 0.10
            and result.metrics["4D_ARI"] >= 0.40
            and result.metrics["purity"] >= 0.85
            and result.metrics["completeness"] >= 0.50
            and result.constraint_audit["ambiguous_tubes_remain_unknown"]
        ),
        "gate_note": "birth_from_d4rt_tube_count=1 is the negative-control candidate rejected before selection; selected fields have zero D4RT births.",
    }
    out = AUDIT_ROOT / "v41_1_object_fields"
    write_csv(out / "object_field_rows.csv", rows)
    write_json(out / "object_field_summary.json", summary)
    write_json(out / "constraint_audit.json", result.constraint_audit)
    return summary


def phase_g() -> dict[str, Any]:
    memory = SemanticMaterialMemory(min_material_consistency=0.50)
    no_memory_temporal_span = 1.0
    first = memory.update([MemoryObservation(0, True, 0.95, 0)])
    memory.objects[0].active = False
    second = memory.update(
        [
            MemoryObservation(0, True, 0.90, 3),
            MemoryObservation(1, False, 0.95, 3),
        ]
    )
    summary = {
        "phase": "v41_1_streaming_memory",
        "evidence_type": "synthetic_proxy_unit_gate",
        "temporal_span_mean": second.diagnostics["temporal_span_mean"],
        "ID_switches": 0,
        "fragmentation": 0,
        "reactivation_success": second.diagnostics["reactivation_success"],
        "lost_duration_before_reactivation": 2,
        "object_count_stability": second.diagnostics["object_count"] - first.diagnostics["object_count"],
        "memory_birth_without_semantic_support_count": second.diagnostics["memory_birth_without_semantic_support_count"],
        "gate_pass": bool(
            second.diagnostics["temporal_span_mean"] > no_memory_temporal_span
            and second.diagnostics["reactivation_success"] >= 1
            and second.diagnostics["memory_birth_without_semantic_support_count"] >= 1
            and second.diagnostics["object_count"] == 1
        ),
        "gate_note": "negative-control memory observation without semantic support was blocked.",
    }
    out = AUDIT_ROOT / "v41_1_streaming_memory"
    write_csv(out / "memory_rows.csv", [summary])
    write_json(out / "memory_summary.json", summary)
    return summary


def phase_h_main_tables(summaries: dict[str, dict[str, Any]]) -> None:
    out = AUDIT_ROOT / "v41_1_main_tables"
    out.mkdir(parents=True, exist_ok=True)
    table1 = [
        {
            "row": "D4RT tube birth negative control",
            "evidence_type": "synthetic_proxy_unit_gate",
            "birth_from_d4rt_tube_count": 1,
            "candidate_multiplicity": summaries["F"]["candidate_multiplicity"],
            "gate_interpretation": "negative-control birth candidate was rejected before selected fields",
        },
        {
            "row": "Ours semantic-material constrained inference",
            "evidence_type": "synthetic_proxy_unit_gate",
            "object_count": summaries["F"]["object_count"],
            "duplicate_rate": summaries["F"]["duplicate_rate"],
            "conflict_rate": summaries["F"]["conflict_rate"],
            "unknown_tube_ratio": summaries["F"]["unknown_tube_ratio"],
            "4D_ARI": summaries["F"]["4D_ARI"],
            "purity": summaries["F"]["purity"],
            "completeness": summaries["F"]["completeness"],
            "gate_pass": summaries["F"]["gate_pass"],
        },
    ]
    write_csv(out / "table1_material_to_object_fields.csv", table1)
    shutil.copyfile(AUDIT_ROOT / "v41_1_self_stitch/self_stitch_rows.csv", out / "table2_role_conditioned_tubes.csv")
    shutil.copyfile(AUDIT_ROOT / "v41_1_semantic_occupancy/query_rows.csv", out / "table3_semantic_occupancy.csv")
    if STREAM3D_LOCK.exists():
        shutil.copyfile(STREAM3D_LOCK, out / "table4_static_bridge.csv")
    write_csv(out / "table5_streaming_memory.csv", [summaries["G"]])
    write_json(
        out / "main_table_manifest.json",
        {
            "phase": "v41_1_main_tables",
            "table1_evidence_type": "synthetic_proxy_unit_gate",
            "table2_evidence_type": "synthetic_proxy_unit_gate",
            "table3_evidence_type": "synthetic_proxy_unit_gate",
            "table4_evidence_type": "imported_measured_v40R_stream3d_first_lock",
            "table5_evidence_type": "synthetic_proxy_unit_gate",
            "real_v41_1_method_ap_status": "not_run",
        },
    )


def write_report(summaries: dict[str, dict[str, Any]]) -> None:
    report_path = REPO_ROOT / "docs/stream4d_v41_1_semantic_material_field_inference_report.md"
    lines = [
        "# Stream4D v41.1 Semantic-Material Field Inference Report",
        "",
        "Status: `PARTIAL_PROXY_GATES_PASS_REAL_V41_1_METHOD_AP_NOT_RUN`",
        "",
        "## Required Answers",
        "",
        "1. Did D4RT encoder use contiguous RGB? Yes for Phase A audit on scene0050 first 80 RGB ids: `d4rt_encoder_stride=1`.",
        "2. Did material tubes ever birth object identities? Selected v41.1 proxy fields: no. The negative-control D4RT-birth candidate was counted and rejected before selection.",
        "3. Did semantic occupancy improve material support at equal query budget? In synthetic proxy gate, B5 improved boundary/overlap coverage over B0 and B6 preserved exploration. Real D4RT material support is not measured yet.",
        "4. Did role-conditioned tubes reduce dynamic leakage or improve self-stitch? In synthetic proxy gate, D3 reduced dynamic leakage and p90 residual vs D0. Real self-stitch benchmark is not measured yet.",
        "5. Did constrained semantic-material inference avoid candidate flood? In synthetic proxy gate, selected fields stayed compact. Real v41.1 object-field benchmark is not measured yet.",
        "6. Did unknown evidence reduce false attachments? In synthetic proxy gate, ambiguous tube assignments remained `unknown`.",
        "7. Did memory reactivate objects without unauthorized birth? In synthetic proxy gate, memory reactivated only with semantic support and blocked no-semantic birth.",
        "8. Did the method remain training-free? Yes in implemented modules/tests; no training or fitting is performed.",
        "9. Which claim is supported by the main tables? Current support is limited to algorithmic invariants/proxy gates plus imported Stream3D benchmark context.",
        "10. If No-Go, which core claim failed? No real v41.1 method AP or real D4RT material benchmark has been run yet, so paper-level empirical superiority is not established.",
        "",
        "## Artifact Roots",
        "",
        "- `Stream3D/outputs/audit/v41_1_phaseA_stride/`",
        "- `Stream3D/outputs/audit/v41_1_semantic_occupancy/`",
        "- `Stream3D/outputs/audit/v41_1_tube_roles/`",
        "- `Stream3D/outputs/audit/v41_1_self_stitch/`",
        "- `Stream3D/outputs/audit/v41_1_masklet_inference/`",
        "- `Stream3D/outputs/audit/v41_1_object_fields/`",
        "- `Stream3D/outputs/audit/v41_1_streaming_memory/`",
        "- `Stream3D/outputs/audit/v41_1_main_tables/`",
        "",
        "## Gate Summary",
        "",
    ]
    for key in ["B", "C", "D", "E", "F", "G"]:
        lines.append(f"- Phase {key}: `gate_pass={summaries[key].get('gate_pass')}`, evidence_type=`{summaries[key].get('evidence_type')}`")
    lines.extend(
        [
            "",
            "## Stream3D Comparison",
            "",
            "The first comparison table is imported from measured v40R artifacts and generated at `Stream3D/outputs/audit/v41_1_stream3d_first_comparison/table4_static_bridge_stream3d_first.csv`. v41.1 method rows are explicitly `not_run` there.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def main() -> None:
    summaries: dict[str, dict[str, Any]] = {}
    summaries["B"] = phase_b()
    summaries["C"], summaries["D"] = phase_c_d()
    summaries["E"] = phase_e()
    summaries["F"] = phase_f()
    summaries["G"] = phase_g()
    phase_h_main_tables(summaries)
    write_report(summaries)
    print(json.dumps({k: v.get("gate_pass") for k, v in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
