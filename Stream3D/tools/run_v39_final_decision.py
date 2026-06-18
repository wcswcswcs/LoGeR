from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _manifest(phase: str, status: str, reason: str, blocker: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "status": status,
        "not_run_reason": reason,
        "upstream_blocker": blocker,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": False,
        "uses_frozen_visual_backbone": False,
        "visual_backbone_name": None,
        "mask_source": "none_not_run",
        "object_birth_source": "none_not_run",
        "d4rt_role": "none_not_run",
        "geometry_field": "none_not_run",
        "coordinate_frame": "none_not_run",
        "alignment_source": "none_not_run",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    phase_a = _read_json(root / "outputs/audit/v39_phaseA_failure_lock/failure_lock.json")
    phase_b = _read_json(root / "outputs/audit/v39_old_pool_oracle_set_cover/old_pool_oracle_set_cover_summary.json")
    phase_c = _read_json(root / "outputs/audit/v39_masklet_primitive/masklet_primitive_summary.json")
    phase_h = _read_json(root / "outputs/audit/v39_learned_diagnostic/learned_diagnostic_summary.json")
    stronger_source_path = root / "outputs/audit/v39_stronger_mask_source/stronger_mask_source_summary.json"
    stronger_source = _read_json(stronger_source_path) if stronger_source_path.exists() else None
    purity_repair_path = root / "outputs/audit/v39_purity_targeted_repair/v39_summary/purity_targeted_repair_summary.json"
    purity_repair = _read_json(purity_repair_path) if purity_repair_path.exists() else None
    compact_paths = [
        root / "outputs/audit/v39_compact_object_materializer/compact_object_materializer_summary.json",
        root / "outputs/audit/v39_compact_object_materializer_contact/compact_object_materializer_summary.json",
    ]
    compact_summaries = [_read_json(path) for path in compact_paths if path.exists()]
    compact_eval_rows = [
        {**row, "_summary_index": idx}
        for idx, summary in enumerate(compact_summaries)
        for row in summary.get("eval_rows", [])
        if row.get("AP") is not None
    ]
    compact_best = max(compact_eval_rows, key=lambda row: float(row.get("AP") or -1.0), default={})
    compact_best_summary = compact_summaries[int(compact_best.get("_summary_index", 0))] if compact_best else {}

    facts = phase_a["facts"]
    phase_b_gate = phase_b["phaseB_gate"]
    phase_c_gate = phase_c["phaseC_gate"]
    phase_h_best = phase_h["best_stage_stats"]
    phase_h_model = phase_h["model_stats"]
    no_go_reason = (
        "Phase B Stop 1 retired the old candidate pool; Phase C object-birth primitive failed; "
        "Phase H GT-trained pair diagnostic showed learnable pair signal but still failed object gate."
    )
    final_status = "NO_GO_PHASEH_AND_PHASEC_OBJECT_IDENTITY_BLOCKER"
    if purity_repair is not None:
        no_go_reason = (
            "A post-closeout F31 adaptive-density repair recovered the 3D gate and I4 sparse memory recovered the 4D gate, "
            "but AP/export stayed in the candidate-flood regime. Raw export kept 4051.8 predictions/scene with conflict "
            "0.8039312778723964, and the best count/conflict-regularized AP was only 0.00466218049032394."
        )
        final_status = str(purity_repair.get("final_status") or final_status)
    if compact_summaries:
        no_go_reason = (
            "F31/I4 recovered 3D/4D identity, but compact object materialization still failed the AP gate. "
            f"The best completed compact diagnostic was {compact_best_summary.get('best_variant')} with "
            f"AP/AP50/AP25={compact_best.get('AP')}/{compact_best.get('AP50')}/{compact_best.get('AP25')}, "
            f"below the AP gate {compact_best_summary.get('ap_gate')}."
        )
        final_status = "NO_GO_COMPACT_OBJECT_MATERIALIZER_AP_STILL_LOW"

    blocked_phase_roots = {
        "v39_object_set_cover": (
            "Phase D object set cover",
            "NOT_RUN_PHASEC_OBJECT_BIRTH_PRIMITIVE_BLOCKER",
            "No valid v39 2D object masklet primitive passed Phase C.",
        ),
        "v39_d4rt_support_attachment": (
            "Phase E D4RT support attachment",
            "NOT_RUN_PHASED_NOT_AVAILABLE",
            "No selected compact object set was produced by Phase D.",
        ),
        "v39_export_one_object_one_prediction": (
            "Phase F one-object-one-prediction export",
            "NOT_RUN_PHASED_E_NOT_AVAILABLE",
            "No selected objects or D4RT attachment were available for one-object export.",
        ),
        "v39_oracle_attribution": (
            "Phase G oracle attribution",
            "NOT_RUN_NO_V39_OBJECT_EXPORT",
            "No v39 object primitive/export existed to run a meaningful new-object oracle matrix.",
        ),
        "v39_4d_memory_if_allowed": (
            "Phase I 4D memory",
            "NOT_RUN_PRECONDITIONS_FAILED",
            "Phase D/E/F object-set and AP sanity gates did not pass.",
        ),
        "v39_heldout_validation": (
            "Phase J held-out validation",
            "NOT_RUN_PRECONDITIONS_FAILED",
            "No v39 method route passed probe5 gates; held-out success would be ungrounded.",
        ),
    }
    phase_rows = [
        {
            "phase": "Phase A failure lock",
            "artifact_root": "outputs/audit/v39_phaseA_failure_lock",
            "status": "PASS",
            "gate_pass": True,
            "key_metric": "candidate_first_route_retired=true",
            "decision": "old candidate-first main route retired",
        },
        {
            "phase": "Phase B old-pool oracle set cover",
            "artifact_root": "outputs/audit/v39_old_pool_oracle_set_cover",
            "status": "STOP1_FAIL",
            "gate_pass": False,
            "key_metric": f"best_oracle_AP={phase_b_gate.get('best_oracle_AP')}",
            "decision": "old candidate pool unsalvageable as main route",
        },
        {
            "phase": "Phase C masklet primitive",
            "artifact_root": "outputs/audit/v39_masklet_primitive",
            "status": "FAIL_OBJECT_BIRTH_PRIMITIVE",
            "gate_pass": False,
            "key_metric": f"best_masklet_purity={phase_c_gate.get('best_masklet_purity')}",
            "decision": "object_birth_primitive_blocker=true",
        },
        {
            "phase": "Phase H learned diagnostic",
            "artifact_root": "outputs/audit/v39_learned_diagnostic",
            "status": phase_h.get("phaseH_status"),
            "gate_pass": bool(phase_h.get("phaseH_gate_pass")),
            "key_metric": f"mean_AUC={phase_h_model.get('mean_test_AUC')}, best_purity={phase_h_best.get('purity')}",
            "decision": "learned pair signal exists but object gate still fails",
        },
    ]
    if stronger_source is not None:
        stronger_gate = stronger_source.get("stronger_source_gate", {})
        phase_rows.append(
            {
                "phase": "Post-closeout stronger mask/source audit",
                "artifact_root": "outputs/audit/v39_stronger_mask_source",
                "status": stronger_source.get("final_status"),
                "gate_pass": bool(
                    stronger_gate.get("any_phaseC_source_gate_pass")
                    or stronger_gate.get("any_non_oracle_object_gate_pass")
                ),
                "key_metric": (
                    f"best_phaseC_balanced_purity={stronger_gate.get('best_phaseC_balanced_purity')}, "
                    f"best_non_oracle_object_purity={stronger_gate.get('best_object_purity')}"
                ),
                "decision": "stronger existing sources still fail non-oracle Phase C/object gates",
            }
        )
    if purity_repair is not None:
        gates = purity_repair.get("gates", {})
        local = purity_repair.get("local_3d", {})
        memory = purity_repair.get("memory_4d", {})
        best_ap = purity_repair.get("best_ap_row", {})
        phase_rows.append(
            {
                "phase": "Post-closeout purity-targeted 3D/4D repair and AP export",
                "artifact_root": "outputs/audit/v39_purity_targeted_repair",
                "status": purity_repair.get("final_status"),
                "gate_pass": False,
                "key_metric": (
                    f"3D={local.get('status')}, 4D={memory.get('status')}, "
                    f"best_AP={best_ap.get('AP')}, raw_objects={gates.get('raw_mean_exported_objects')}"
                ),
                "decision": "3D/4D identity can pass, but AP/export object-count blocker remains",
            }
        )
    if compact_summaries:
        for idx, summary in enumerate(compact_summaries):
            artifact_root = (
                "outputs/audit/v39_compact_object_materializer"
                if idx == 0
                else "outputs/audit/v39_compact_object_materializer_contact"
            )
            phase_rows.append(
                {
                    "phase": "Post-closeout compact object materializer",
                    "artifact_root": artifact_root,
                    "status": summary.get("final_status"),
                    "gate_pass": False,
                    "key_metric": (
                        f"best_variant={summary.get('best_variant')}, "
                        f"AP={summary.get('best_AP')}, AP50={summary.get('best_AP50')}, AP25={summary.get('best_AP25')}"
                    ),
                    "decision": "soft-merge/contact compact object materialization improves AP over raw but remains below gate",
                }
            )
    for root_name, (phase, status, reason) in blocked_phase_roots.items():
        manifest = _manifest(phase, status, reason, "PHASEC_OBJECT_BIRTH_PRIMITIVE_BLOCKER")
        target = root / "outputs/audit" / root_name / "not_run_manifest.json"
        _write_json(target, manifest)
        phase_rows.append(
            {
                "phase": phase,
                "artifact_root": f"outputs/audit/{root_name}",
                "status": status,
                "gate_pass": False,
                "key_metric": "not_run",
                "decision": reason,
            }
        )

    answers = [
        {
            "question_id": 1,
            "question": "是否正式 retire candidate-first old route？",
            "answer": "Yes. candidate_first_route_retired=true and forbidden_to_continue_as_main_route=true.",
            "evidence": "outputs/audit/v39_phaseA_failure_lock/old_route_retired_manifest.json",
        },
        {
            "question_id": 2,
            "question": "old candidate pool 的 oracle compact set AP 是多少？",
            "answer": f"{phase_b_gate.get('best_oracle_AP')} for {phase_b_gate.get('best_oracle_variant')}; it fails the >=0.35 gate.",
            "evidence": "outputs/audit/v39_old_pool_oracle_set_cover/old_pool_oracle_set_cover_summary.json",
        },
        {
            "question_id": 3,
            "question": "新 masklet primitive 的 oracle compact set AP 是多少？",
            "answer": "Not reached. Phase C primitive gate failed before Phase D/G oracle compact AP.",
            "evidence": "outputs/audit/v39_masklet_primitive/masklet_primitive_summary.json",
        },
        {
            "question_id": 4,
            "question": "non-GT object set cover 是否过 object identity gate？",
            "answer": "No. Phase D was not run because object_birth_primitive_blocker=true.",
            "evidence": "outputs/audit/v39_object_set_cover/not_run_manifest.json",
        },
        {
            "question_id": 5,
            "question": "D4RT 是否只作为 support attachment，而没有 birth object？",
            "answer": "No v39 method route reached Phase E. Not-run manifest forbids D4RT birth claims.",
            "evidence": "outputs/audit/v39_d4rt_support_attachment/not_run_manifest.json",
        },
        {
            "question_id": 6,
            "question": "mean_predictions_per_scene 是否降到 <=300？",
            "answer": f"No v39 method export. Retired old route remains {facts.get('mean_predictions_per_scene')} predictions/scene.",
            "evidence": "outputs/audit/v39_phaseA_failure_lock/failure_lock.json",
        },
        {
            "question_id": 7,
            "question": "candidate_multiplicity_per_GT 是否降到 <=2？",
            "answer": "Only GT/oracle old-pool diagnostics reduce multiplicity; AP remains too low, so this is not method success.",
            "evidence": "outputs/audit/v39_old_pool_oracle_set_cover/old_pool_oracle_set_cover_matrix.csv",
        },
        {
            "question_id": 8,
            "question": "AP/AP50/AP25 是否超过 v38 postprocess？",
            "answer": "No valid v39 method AP was produced after Phase C failed.",
            "evidence": "outputs/audit/v39_export_one_object_one_prediction/not_run_manifest.json",
        },
        {
            "question_id": 9,
            "question": "是否接近 same-support Stream3D？",
            "answer": f"No. Best old-pool oracle AP {phase_b_gate.get('best_oracle_AP')} remains far below same-support Stream3D AP {facts.get('same_support_stream3d_AP')}.",
            "evidence": "outputs/audit/v39_old_pool_oracle_set_cover/old_pool_oracle_set_cover_summary.json",
        },
        {
            "question_id": 10,
            "question": "如果 training-free 失败，learned/calibrated scorer 是否成功？",
            "answer": f"No. Mean LOSO pair AUC={phase_h_model.get('mean_test_AUC')} shows signal, but best purity={phase_h_best.get('purity')} and scene0081_ARI={phase_h_best.get('scene0081_ARI')} fail gates.",
            "evidence": "outputs/audit/v39_learned_diagnostic/learned_diagnostic_summary.json",
        },
        {
            "question_id": 11,
            "question": "是否通过 held-out validation？",
            "answer": "No. Held-out validation was not run because no probe5 method route passed gates.",
            "evidence": "outputs/audit/v39_heldout_validation/not_run_manifest.json",
        },
        {
            "question_id": 12,
            "question": "下一步是继续 training-free、转 calibrated scorer、换 mask source，还是降低 claim？",
            "answer": (
                "Lower the current claim to No-Go/blocker evidence. A stronger existing mask/source audit was tried; "
                "it still found no non-oracle Phase C/object gate pass. Next work needs a new stronger source or trained object primitive, not old-pool repairs."
            ),
            "evidence": "outputs/audit/v39_stronger_mask_source/stronger_mask_source_summary.json",
        },
    ]
    if purity_repair is not None:
        best_ap = purity_repair.get("best_ap_row", {})
        raw = purity_repair.get("raw_ap_export", {})
        for row in answers:
            if row["question_id"] == 8:
                row["answer"] = (
                    f"No. The purity-targeted repair recovered 3D/4D gates, but raw diagnostic AP/AP50/AP25 remained "
                    f"{raw.get('AP')}/{raw.get('AP50')}/{raw.get('AP25')}; the best count/conflict-regularized diagnostic AP was "
                    f"{best_ap.get('AP')} from {best_ap.get('config')}, still below v38 postprocess AP {facts.get('v38_best_postprocess_AP')}."
                )
                row["evidence"] = "outputs/audit/v39_purity_targeted_repair/v39_summary/purity_targeted_repair_summary.json"

    if compact_summaries:
        for row in answers:
            if row["question_id"] == 8:
                row["answer"] = (
                    f"No. Compact object materialization was tried after F31/I4. The best completed compact diagnostic is "
                    f"{compact_best_summary.get('best_variant')} with AP/AP50/AP25 "
                    f"{compact_best.get('AP')}/{compact_best.get('AP50')}/{compact_best.get('AP25')}, below the AP gate "
                    f"{compact_best_summary.get('ap_gate')}."
                )
                row["evidence"] = "outputs/audit/v39_compact_object_materializer/compact_object_materializer_summary.json"
            if row["question_id"] == 12:
                row["answer"] = (
                    "Close out as No-Go. F31/I4 proves the identity stack can pass local 3D/4D gates, but K1-K6 compact "
                    "soft-merge/contact materializers still fail AP. Next work should replace heuristic materialization with a "
                    "learned or stronger source-backed object exporter; more old-pool top-K/NMS/WTA tuning is low value."
                )
                row["evidence"] = "outputs/audit/v39_compact_object_materializer/compact_object_materializer_summary.json"

    visualizations = [
        "old candidate flood visualization",
        "object-set cover before/after",
        "2D masklet tracks",
        "D4RT support attachment overlay",
        "one-object-one-prediction export overlay",
        "top duplicate candidates by GT",
        "top false positives",
        "missed GT objects",
        "same-support Stream3D vs v39 comparison",
        "scene0081 hardcase panel",
    ]
    visualization_rows = [
        {
            "visualization": name,
            "status": "not_produced",
            "reason": "No valid v39 object-set/export route after Phase C/H blockers; producing method visualizations would be misleading.",
            "uses_gt_for_visualization": False,
            "uses_gt_for_prediction": False,
        }
        for name in visualizations
    ]

    summary = {
        "final_status": final_status,
        "method_success_claimed": False,
        "no_go_reason": no_go_reason,
        "phase_rows": phase_rows,
        "required_answers": answers,
        "visualization_rows": visualization_rows,
        "stop_rules_triggered": [
            "Stop 1: old pool oracle AP < 0.35",
            "Phase C all available region sources fail object birth primitive gate",
            "Stop 6-equivalent: learned/calibrated pair diagnostic fails object gate",
            "Post-closeout stronger existing mask/source audit found no non-oracle Phase C/object gate pass",
            "Post-closeout purity-targeted F31/I4 repair passes 3D/4D but fails AP/export object-count gate",
            "Post-closeout compact object materializer K1-K6 fails AP gate",
        ],
        "next_recommendation": [
            "Retire old-pool candidate-first repairs as the main route.",
            "Keep F31 adaptive-density identity repair as the current 3D/4D identity baseline.",
            "Replace heuristic AP materialization/export with a learned or stronger source-backed compact object exporter.",
        ],
    }
    _write_json(output_root / "decision_summary.json", summary)
    _write_json(output_root / "required_answers.json", answers)
    _write_csv(output_root / "phase_status_matrix.csv", phase_rows)
    _write_csv(output_root / "visualization_manifest.csv", visualization_rows)
    _write_json(output_root / "visualization_manifest.json", visualization_rows)
    md = [
        "# Stream4D v39 Final Decision",
        "",
        f"`final_status={final_status}`",
        "",
        "`method_success_claimed=False`",
        "",
        "## Phase Status",
        "",
        "| phase | status | decision |",
        "|---|---|---|",
    ]
    for row in phase_rows:
        md.append(f"| {row['phase']} | {row['status']} | {row['decision']} |")
    md.extend(["", "## Required Answers", "", "| id | answer | evidence |", "|---:|---|---|"])
    for row in answers:
        md.append(f"| {row['question_id']} | {row['answer']} | {row['evidence']} |")
    (output_root / "decision_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write v39 final decision and not-run manifests after stop rules.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--output-root", default="outputs/audit/v39_final_decision")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
